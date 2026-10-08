"""Hyperparameter optimisation of the learned controllers: behaviour cloning
(IL), the DQN (RL), and the clone fine-tuned by the DQN (IL+RL).

The driver is `run_rl_hpo.py`; this module holds what a study IS -- its search
space, its objective, its households, its storage -- and the frames the
notebook (`FIGURES_RL_HPO.ipynb`) draws from. It never imports matplotlib.

What `run_rl_benchmark.tune` already did, and why this is not that again
------------------------------------------------------------------------
`tune` is a 4-point grid over two knobs at a time (gamma x n-step, gamma x BC
regulariser), on the 30 study households' validation weeks. It answered the
question the first screen had leaked (those knobs had been set from scored-year
costs) and nothing more: the clone's optimiser, the network width, the DQN's
learning rate, batch, target sync, exploration schedule and reward scale have
never been searched at all. This searches them jointly, with a model-based
sampler (TPE) and early abandonment of hopeless configurations.

The three-way split is unchanged, and stricter than `tune`
----------------------------------------------------------
    search      trains on TRAIN blocks, scores the VALIDATION weeks -- the same
                `run_rl_benchmark.run_one(score_test=False)` the grid used, so a
                trial is a panel run that stops before the test year.
    households  the search runs on households that are NOT study units: other
                members of the k=30 clustering, stratified over the clusters.
                The 30 study households are never seen by a search, so even
                the validation weeks of the households that are later scored
                stay out of every selection. `tune` could not do this; the
                pooled study (`run_rl_global`) is what solved a teacher MILP
                for the other 269, and those solves are reused here.
    refine      the best few configurations, and the current settings, are
                re-trained under two more seeds on the same households, and the
                winner is picked on the seed average. A search ranks single
                draws; with 8 households and one seed, the top of the ranking
                is partly seed luck, and the seeds of this learner move results
                coherently across households (a re-seeded local clone shifted
                the AU mean by 6 AUD/a, raw p .03 -- `run_rl_global
                .local_seed_runs`).
    confirm     the winner and the current settings are trained on the 30 study
                units under three seeds and scored on the TEST year through the
                arm's own `run_policy` + settle. This is the first and only
                time a tuned configuration meets the test year, and it is a
                verdict, not a selection: nothing reads it back into a choice.

Objective
---------
Validation saving against an idle battery, annualised (currency/a): for each
household, the validation weeks' closed bill of a network that always idles
(`rl.validation_rollout` with an idle net -- the evaluator that scores the
learner, not a second one) minus the learner's closed bill plus lifetime wear
(`val_cost_net_of_wear`, the axis `tune_report` picks on), times 365/140. The
trial's value is the mean over its households. Subtracting the idle bill
changes no ranking -- it is a constant per household -- but makes the number
readable: "the battery saves X a year on the held-out weeks".

Interruption
------------
Everything survives a kill at any moment, and a rerun of the same command
resumes:

    * each (configuration, household, seed) training is `run_one`'s own result
      JSON, keyed by its config digest -- a finished household is never
      retrained, and the scheduler resolves it from disk without a worker;
    * the optuna study is an append-only journal file (`JournalFileBackend`):
      a trial the killed driver had open is RUNNING in it, and is re-queued
      with the same parameters on restart (its finished households then come
      back from disk, so only the households in flight are lost);
    * refine and confirm are plain lists of such evaluations, so they resume
      the same way, and the summaries they write (`winner.json`,
      `confirm.json`) are rebuilt from the cached runs on every rerun.

One driver at a time: `driver.lock` holds its pid, and a second driver refuses
to start while that process is alive. The notebook only reads.

Layout
------
    results_local/rl_hpo/<study>/journal.log         optuna (append-only)
    results_local/rl_hpo/<study>/study.json          what the study is
    results_local/rl_hpo/<study>/runs/<tag>/s<seed>/ run_one results, search + refine
    results_local/rl_hpo/<study>/winner.json         refine verdict
    results_local/rl_hpo/<study>/confirm/<tag>/s<seed>/   test-year runs
    results_local/rl_hpo/<study>/confirm/current/s<seed>/   the panel's settings
                                                     where the panel lacks them
    results_local/rl_hpo/<study>/confirm.json        test-year rows
    results_local/rl_hpo/baselines/<tariff>/<id>.json   idle validation bill
    rl_models/hpo/...                                networks (gitignored)

A study is `<tariff>_<method>` (plus `_<tag>` for a second study of the same
pair). `<tag>` of a run directory is a digest of its full override set and is
spelled out in `runs/<tag>/overrides.json`.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import sys
import time
import warnings
from collections import OrderedDict, deque
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
HPO_ROOT = os.environ.get("ERK_HPO_ROOT",
                          os.path.join(HERE, "results_local", "rl_hpo"))
MODELS_ROOT = os.environ.get("ERK_HPO_MODELS", os.path.join(rb.MODELS, "hpo"))
VARIANT = rb.TUNE_VARIANT           # fc_h24: the deployable feature set
METHODS = ("bc", "dqn", "bc_dqn")
TARIFFS = ("AU", "SI")
STEPS = 500_000                     # the panel's DQN budget (`--steps` default)
N_VAL_DAYS = sum(b - a for a, b in rb.VAL_BLOCKS)

# Bumped whenever a search space or the objective changes shape. It sits in
# study.json, and a study opened under another version is refused rather than
# extended -- TPE would otherwise model two different objectives as one.
SPACE_VERSION = 1


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
# (name, kind, spec). Names are `TrainConfig` fields, or `clone_<field>` for
# the clone's optimiser (`rl.BCOptions`), exactly the keys
# `run_rl_benchmark.effective_config` / `effective_clone` accept -- so a
# winner is a TUNED entry as it stands, nothing to translate.
#
# Every range CONTAINS the current setting (the incumbent is enqueued as trial
# 0 and must be expressible in the space -- `test_rl_hpo` checks it). Discrete
# sets for knobs whose useful values are a handful of magnitudes (widths,
# batch sizes, sync intervals, weight decay including an exact 0); continuous
# ranges only where the value is genuinely a dial.
#
# Not searched, deliberately:
#   total_steps, eval_every, patience   the convergence machinery: early
#       stopping already decides how long a run trains, and a search over the
#       budget would reward whichever setting overfits the validation weeks
#       slowest.
#   network depth                       QNet is fixed at two hidden layers;
#       the warm start of bc_dqn and every saved model assume it.
#   seed                                not a hyperparameter; refine averages it.
SPACES = {
    # Behaviour cloning. The clone's whole optimiser had been hard-coded
    # (batch 512, patience 10, unweighted decay, inverse-frequency class
    # weights); `lr` and `hidden` are the TrainConfig fields the clone reads.
    "bc": [
        ("lr", "log", (1e-4, 1e-2)),
        ("hidden", "cat", (64, 128, 256)),
        ("clone_batch", "cat", (128, 256, 512, 1024, 2048)),
        ("clone_weight_decay", "cat", (0.0, 1e-6, 1e-5, 1e-4, 1e-3)),
        # 1 = inverse-frequency weights (the current fit), 0 = unweighted.
        # Weighting was introduced because an unweighted fit collapsed onto
        # idle; whether FULL inverse frequency is the right strength never was
        # asked.
        ("clone_class_power", "float", (0.0, 1.0)),
        ("clone_label_smoothing", "cat", (0.0, 0.05, 0.1)),
        ("clone_patience", "cat", (5, 10, 20)),
    ],
    # The DQN from a cold start.
    "dqn": [
        ("lr", "log", (1e-4, 3e-3)),
        ("gamma", "float", (0.98, 0.999)),
        ("n_step", "cat", (1, 2, 4, 8, 16)),
        ("batch", "cat", (64, 128, 256)),
        ("hidden", "cat", (64, 128, 256)),
        ("target_sync", "cat", (500, 1000, 2000, 5000, 10000)),
        ("update_every", "cat", (1, 2, 4)),
        ("buffer", "cat", (50_000, 100_000, 200_000, 400_000)),
        ("eps_end", "float", (0.01, 0.1)),
        ("eps_decay_frac", "float", (0.1, 0.6)),
        ("reward_scale", "log", (5.0, 80.0)),
        ("episode_days", "cat", (3, 7, 14)),
    ],
    # The fine-tune of a FIXED clone. `hidden` is not searched: the warm start
    # loads the clone's weights, so the width is the clone's. The clone itself
    # is the IL study's winner (or the current clone, `clone_source`), held
    # fixed so this study measures the fine-tune and nothing else.
    "bc_dqn": [
        ("lr", "log", (1e-4, 3e-3)),
        ("gamma", "float", (0.98, 0.999)),
        ("n_step", "cat", (1, 2, 4, 8, 16)),
        ("batch", "cat", (64, 128, 256)),
        ("target_sync", "cat", (500, 1000, 2000, 5000, 10000)),
        ("eps_decay_frac", "float", (0.1, 0.6)),
        ("reward_scale", "log", (5.0, 80.0)),
        ("bc_reg", "float", (0.0, 3.0)),
        ("bc_reg_decay_frac", "float", (0.1, 1.0)),
        ("bc_guide_prob", "float", (0.0, 1.0)),
    ],
}

# Per-method study shape. A clone trains in ~1 s and its run is dominated by
# preparing the household, so IL buys breadth (many households, many trials,
# evaluated household-major in batches so a worker prepares a household once
# for a whole batch). A DQN run is ~2 min, so RL buys depth with pruning: a
# trial is abandoned after its first rung of households if it sits below the
# median of earlier trials there.
DEFAULTS = {
    "bc": {"trials": 144, "households": 16, "rungs": None, "batch": 24,
           "startup": 24},
    "dqn": {"trials": 40, "households": 8, "rungs": (3, 8), "batch": 1,
            "startup": 10},
    "bc_dqn": {"trials": 40, "households": 8, "rungs": (3, 8), "batch": 1,
               "startup": 10},
}
REFINE_TOP = 3
REFINE_SEEDS = (1, 2)
CONFIRM_SEEDS = (0, 1, 2)
# What one evaluation costs, seconds, before any has been measured (from the
# panel's own wall times): the ETA's prior.
PRIOR_EVAL_S = {"bc": 15.0, "dqn": 140.0, "bc_dqn": 150.0}


def _space(method):
    return SPACES[method]


def suggest(trial, method: str) -> dict:
    """Draw one configuration from `method`'s space on an optuna trial."""
    out = {}
    for name, kind, spec in _space(method):
        if kind == "log":
            out[name] = trial.suggest_float(name, spec[0], spec[1], log=True)
        elif kind == "float":
            out[name] = trial.suggest_float(name, spec[0], spec[1])
        elif kind == "cat":
            out[name] = trial.suggest_categorical(name, list(spec))
        else:
            raise ValueError(kind)
    return out


def in_space(method: str, params: dict) -> list:
    """Names whose value lies outside the space (empty = expressible)."""
    bad = []
    for name, kind, spec in _space(method):
        v = params.get(name)
        if kind == "cat":
            ok = v in spec
        else:
            ok = v is not None and spec[0] - 1e-12 <= v <= spec[1] + 1e-12
        if not ok:
            bad.append(name)
    return bad


def current_value(tariff: str, method: str, name: str, steps: int = STEPS):
    """What the panel trains `name` at today (TrainConfig + TUNED, or BCOptions)."""
    if name.startswith(rb.CLONE_PREFIX):
        clone = rb.effective_clone(tariff, method)
        return getattr(clone, name[len(rb.CLONE_PREFIX):])
    cfg = rb.effective_config(rb.make_config(steps, 0), tariff, method)
    return getattr(cfg, name)


def incumbent_params(tariff: str, method: str, steps: int = STEPS) -> dict:
    """The current settings, as a point of `method`'s space."""
    return {name: current_value(tariff, method, name, steps)
            for name, _, _ in _space(method)}


# ---------------------------------------------------------------------------
# Overrides, tags, households
# ---------------------------------------------------------------------------
def _digest(*objs, n=12) -> str:
    return hashlib.sha256(json.dumps(objs, sort_keys=True, default=str)
                          .encode()).hexdigest()[:n]


def trial_overrides(base: dict, fixed: dict | None, params: dict) -> dict:
    """The FULL override set `run_one` is handed. It replaces TUNED wholesale,
    so the study's `base` (TUNED as it stood when the study was created) is
    carried along, the fixed clone of a bc_dqn study laid over it, and the
    trial's own values over both. Frozen in study.json rather than read from
    TUNED at each trial: pasting a winner into TUNED mid-study must not move
    the configurations a study is still comparing."""
    over = dict(base or {})
    over.update(fixed or {})
    over.update(params)
    return over


def cfg_tag(overrides: dict) -> str:
    return "c" + _digest(overrides)


def tuning_households(n: int, seed: int = 2026) -> list:
    """`n` non-study households, stratified over the k=30 shape clusters.

    One household per cluster in a seeded random cluster order, then a second
    round, and so on -- so 8 households are 8 different consumer types rather
    than whatever a flat draw lands on. Only households whose two-year teacher
    MILP is cached on BOTH tariffs qualify: that is exactly the 269 the pooled
    study could build (a household with a data gap, e.g. Ausgrid 2, never got
    one), and it means a search never stalls on a fresh 15-minute SI solve.
    """
    import pandas as pd
    units = set(int(i) for i in hs.study_units().index)
    df = pd.read_csv(hs.CLUSTERING_CSV)
    df["ident"] = df["user_id"].str.removeprefix("user_").astype(int)
    a, b = rb.TEACH_SPAN

    def _has_teacher(i):
        return all(os.path.exists(os.path.join(rb.MODELS, "teacher", t,
                                               f"d{a}-{b}", f"{i}.npz"))
                   for t in TARIFFS)

    df = df[~df["ident"].isin(units)]
    df = df[df["ident"].map(_has_teacher)]
    rng = np.random.default_rng(seed)
    groups = {c: list(rng.permutation(g["ident"].to_numpy()))
              for c, g in df.groupby("cluster")}
    order = list(rng.permutation(sorted(groups)))
    out = []
    while len(out) < n and any(groups.values()):
        for c in order:
            if groups[c] and len(out) < n:
                out.append(int(groups[c].pop(0)))
    if len(out) < n:
        raise ValueError(f"only {len(out)} tuning households available, {n} asked")
    return out


def study_units() -> list:
    return [int(i) for i in hs.study_units().index]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def study_dir(name: str, root: str | None = None) -> str:
    return os.path.join(root or HPO_ROOT, name)


def run_root(name: str, tag: str, seed: int, root=None, confirm=False) -> str:
    sub = "confirm" if confirm else "runs"
    return os.path.join(study_dir(name, root), sub, tag, f"s{seed}")


def model_root(out_root: str, root=None, models=None) -> str:
    """The networks mirror the result tree under the gitignored rl_models/."""
    rel = os.path.relpath(out_root, root or HPO_ROOT)
    return os.path.join(models or MODELS_ROOT, rel)


def _result_path(out_root, tariff, method, ident):
    return os.path.join(out_root, tariff, f"{VARIANT}__{method}", f"{ident}.json")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# The idle baseline: the evaluator's own number for "no battery"
# ---------------------------------------------------------------------------
class _IdleNet:
    """A 'network' that always picks idle. Callable the way `greedy_rollout`
    calls a QNet, so the baseline goes through the learner's own evaluator."""

    def eval(self):
        return self

    def __call__(self, x):
        import torch
        out = torch.zeros((x.shape[0], rl.N_ACTIONS))
        out[:, rl.A_IDLE] = 1.0
        return out


def _baseline_digest(tariff):
    return _digest({"tariff": tariff, "val": rb.VAL_BLOCKS,
                    "soc": rb.SOC_INIT_USABLE, "variant": VARIANT})


def baseline_path(tariff, ident, root=None):
    return os.path.join(root or HPO_ROOT, "baselines", tariff, f"{ident}.json")


def read_baseline(tariff, ident, root=None):
    p = baseline_path(tariff, ident, root)
    if os.path.exists(p):
        try:
            b = _read_json(p)
            if b.get("digest") == _baseline_digest(tariff):
                return float(b["val_idle_cost"])
        except Exception:
            pass
    return None


def idle_validation_cost(prep, root=None) -> float:
    """Closed validation bill of a battery that never moves, cached."""
    tariff, ident = prep["tariff"], prep["ident"]
    cached = read_baseline(tariff, ident, root)
    if cached is not None:
        return cached
    spec = rb.variant_specs(tariff)[VARIANT]
    fb, static = rb._features(prep, spec, "train")
    rows = np.concatenate([np.arange(a * rb.H, b * rb.H) for a, b in rb.TRAIN_BLOCKS])
    fb.fit_norm(static[rows])
    tr = prep["train"]
    val = rl.validation_rollout(_IdleNet(), fb, fb.normalize(static), tr["sig"],
                                tr["settle"], tr["env"], rb.VAL_BLOCKS,
                                rb.SOC_INIT_USABLE, tariff == "SI")
    _write_json(baseline_path(tariff, ident, root),
                {"digest": _baseline_digest(tariff), "tariff": tariff,
                 "ident": int(ident), "val_idle_cost": val["cost_eur_closed"],
                 "val_days": N_VAL_DAYS})
    return float(val["cost_eur_closed"])


# ---------------------------------------------------------------------------
# One evaluation: (configuration, household, seed) -> row
# ---------------------------------------------------------------------------
# An Eval is a plain dict so it pickles to a worker unchanged:
#   study, trial, tag, tariff, method, ident, seed, overrides (None = TUNED),
#   steps, score_test, out_root, models_root, root
_ID_KEYS = ("study", "trial", "tag", "label", "tariff", "method", "ident", "seed")


def eval_key(ev) -> tuple:
    return (ev["out_root"], ev["tariff"], ev["method"], int(ev["ident"]))


def _row(ev, res, base) -> dict:
    """What every phase keeps of a run_one result.

    The net-of-wear validation cost is RECOMPUTED from the bill and the cycles
    with today's wear formula rather than read from the result: a clone result
    keeps its digest across a change of the wear model (the digest covers the
    fit, not the accounting), and the panel's clones still carry the per-cycle
    price they were scored under before the NPV switch (Ausgrid 138 AU: stored
    88.16 = bill 52.51 + 85.6 EFC x 0.417)."""
    val_net = float(res["val_cost_closed"]) + rb.val_wear_fn(ev["tariff"])(
        float(res["val_efc"]), N_VAL_DAYS)
    row = {k: ev.get(k) for k in _ID_KEYS}
    row.update({
        "val_net": val_net, "val_bill": float(res["val_cost_closed"]),
        "val_efc": float(res["val_efc"]), "val_idle": base,
        # Saving on the held-out weeks against the idle battery, annualised.
        "saving_val_a": (base - val_net) * 365.0 / N_VAL_DAYS,
        "converged": bool(res.get("train_converged", False)),
        "bc_agreement": res.get("bc_val_agreement"),
        "train_steps": res.get("train_steps"),
        "wall_s": res.get("wall_s"), "error": None,
    })
    if ev.get("score_test"):
        efc = float(res["efc"])
        wear = float(hs.cycle_wear_eur(efc, rb.BATTERY_CAP, ev["tariff"]))
        row.update({
            "cost_eur_total": float(res["cost_eur_total"]),
            "cost_eur_closed": float(res["cost_eur_closed"]),
            "fixed_eur": float(res.get("fixed_eur") or 0.0),
            "efc": efc, "wear": wear,
            # The test year's bill plus the lifetime wear `summarize` charges:
            # what a household pays, the axis the comparison is drawn on.
            "test_net": float(res["cost_eur_total"]) + wear,
            "ref_cost_no_battery": res.get("ref_cost_no_battery"),
            "ref_cost_milp_full": res.get("ref_cost_milp_full"),
        })
    return row


def stored_result(ev):
    """`run_one`'s result JSON for this evaluation if it is on disk under the
    evaluation's own digest (and carries a test score when one is wanted)."""
    path = _result_path(ev["out_root"], ev["tariff"], ev["method"], ev["ident"])
    if not os.path.exists(path):
        return None
    try:
        res = _read_json(path)
    except Exception:
        return None
    spec = rb.variant_specs(ev["tariff"])[VARIANT]
    cfg = rb.make_config(ev["steps"], ev["seed"])
    if res.get("digest") != rb.run_digest(cfg, spec, ev["tariff"], ev["method"],
                                          overrides=ev["overrides"]):
        return None
    if ev.get("score_test") and "cost_eur_total" not in res:
        return None
    return res


def cached_row(ev):
    """The row of an evaluation already on disk under its digest, or None.

    Resolved in the driver, without a worker: a resumed search replays its
    finished households from here in milliseconds. Needs the household's idle
    baseline too; until a worker has computed that, the evaluation goes to a
    worker, which finds the stored run and only adds the baseline.
    """
    res = stored_result(ev)
    if res is None:
        return None
    base = read_baseline(ev["tariff"], ev["ident"], ev.get("root"))
    if base is None:
        return None
    return _row(ev, res, base)


_PREP = OrderedDict()
PREP_CACHE = 3          # households a worker keeps prepared (~100 MB each)


def _prep(ident, tariff):
    key = (int(ident), tariff)
    if key in _PREP:
        _PREP.move_to_end(key)
        return _PREP[key]
    prep = rb.prepare_household(ident, tariff)
    _PREP[key] = prep
    while len(_PREP) > PREP_CACHE:
        _PREP.popitem(last=False)
    return prep


def run_job(job) -> list:
    """Worker entry: every evaluation of one (tariff, household). Never raises
    -- a failed evaluation comes back as a row with `error` set, so one bad
    configuration cannot sink the batch it shares a household with."""
    warnings.simplefilter("ignore")
    rows = []
    try:
        prep = _prep(job["ident"], job["tariff"])
        base = idle_validation_cost(prep, job["evals"][0].get("root"))
    except Exception as exc:
        return [dict(_err_row(ev, f"prepare: {exc!r}")) for ev in job["evals"]]
    for ev in job["evals"]:
        t0 = time.time()
        if ev.get("read_only"):
            # Someone else's result tree (the panel's): read it, never train
            # into it. If it went stale since the driver looked, that is an
            # error to rerun, not a reason to overwrite the panel.
            res = stored_result(ev)
            rows.append(_row(ev, res, base) if res is not None else
                        _err_row(ev, "read-only result is no longer current"))
            continue
        try:
            cfg = rb.make_config(ev["steps"], ev["seed"])
            res = rb.run_one(prep, VARIANT, ev["method"], cfg, verbose=False,
                             out_root=ev["out_root"],
                             models_root=ev["models_root"],
                             score_test=bool(ev.get("score_test")),
                             overrides=ev["overrides"])
            if not ev.get("score_test") and "cost_eur_closed" in res:
                # A search must never be handed a test number, even one left
                # on disk by some other path under the same digest.
                res = {k: v for k, v in res.items()
                       if k not in ("cost_eur_closed", "cost_eur_total")}
            row = _row(ev, res, base)
            row["job_s"] = time.time() - t0
        except Exception as exc:
            row = _err_row(ev, repr(exc))
        rows.append(row)
    return rows


def _err_row(ev, msg):
    row = {k: ev.get(k) for k in _ID_KEYS}
    row["error"] = msg
    return row


# ---------------------------------------------------------------------------
# Studies
# ---------------------------------------------------------------------------
def _optuna():
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    return optuna


def storage(name: str, root=None):
    optuna = _optuna()
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend
    d = study_dir(name, root)
    os.makedirs(d, exist_ok=True)
    return JournalStorage(JournalFileBackend(os.path.join(d, "journal.log")))


def study_settings(tariff: str, method: str, tag: str = "", households=None,
                   n_households=None, rungs=None, steps: int = STEPS,
                   clone_source: str = "hpo", root=None) -> dict:
    """What a study is. Frozen into study.json on creation; a reopen under any
    other value of an identity field is refused (see `open_settings`)."""
    d = DEFAULTS[method]
    n = n_households or d["households"]
    hh = list(households) if households is not None else tuning_households(n)
    rungs = tuple(rungs or d["rungs"] or (len(hh),))
    if rungs[-1] != len(hh):
        rungs = tuple(r for r in rungs if r < len(hh)) + (len(hh),)
    fixed = {}
    if method == "bc_dqn":
        fixed = clone_settings(tariff, clone_source, root)
    name = f"{tariff}_{method}" + (f"_{tag}" if tag else "")
    base = dict(rb.TUNED.get((tariff, method), {}))
    inc_params = incumbent_params(tariff, method, steps)
    return {
        "name": name, "tariff": tariff, "method": method, "variant": VARIANT,
        "households": [int(i) for i in hh], "rungs": list(rungs),
        "steps": int(steps), "space_version": SPACE_VERSION,
        "space": [[n_, k, list(s)] for n_, k, s in _space(method)],
        "fixed": fixed, "clone_source": clone_source if method == "bc_dqn" else None,
        "algo": {"rl": rl.ALGO_VERSION, "bc": rl.BC_ALGO_VERSION},
        "split": {"train": rb.TRAIN_BLOCKS, "val": rb.VAL_BLOCKS},
        # Not identity: the reference points, frozen at creation.
        #   base       TUNED's entry for this pair, carried under every trial
        #   incumbent  trial 0 -- the current settings as a point of the space
        #              (for bc_dqn: on the study's fixed clone)
        #   current    what the panel trains (TUNED, current clone): the
        #              comparator `confirm` scores the winner against
        "base": base,
        "incumbent_params": inc_params,
        "incumbent": trial_overrides(base, fixed, inc_params),
        "current": base,
    }


_IDENTITY = ("tariff", "method", "variant", "households", "rungs", "steps",
             "space_version", "space", "fixed", "algo", "split")


def open_settings(settings: dict, root=None) -> dict:
    """Create study.json, or check that the one on disk is the same study."""
    path = os.path.join(study_dir(settings["name"], root), "study.json")
    if os.path.exists(path):
        old = _read_json(path)
        new = json.loads(json.dumps(settings, default=list))
        diff = [k for k in _IDENTITY if old.get(k) != new.get(k)]
        if diff:
            raise RuntimeError(
                f"study {settings['name']} on disk differs in {diff}: it was "
                f"created under other settings, and extending it would model "
                f"two objectives as one. Start a new study with --tag, or move "
                f"{os.path.dirname(path)} aside.")
        return old
    settings = dict(settings, created=datetime.now().isoformat(timespec="seconds"))
    _write_json(path, settings)
    return settings


def read_settings(name: str, root=None) -> dict:
    return _read_json(os.path.join(study_dir(name, root), "study.json"))


def list_studies(root=None) -> list:
    r = root or HPO_ROOT
    if not os.path.isdir(r):
        return []
    return sorted(d for d in os.listdir(r)
                  if os.path.exists(os.path.join(r, d, "study.json")))


def load_study(name: str, root=None):
    optuna = _optuna()
    return optuna.load_study(study_name=name, storage=storage(name, root))


def clone_settings(tariff: str, source: str, root=None) -> dict:
    """The clone a bc_dqn study fine-tunes, as overrides.

    `hpo`: the IL study's refined winner -- the clone is held at the best
    imitation the search found, so IL+RL measures what reinforcement adds to
    the best clone rather than to an arbitrary one. `current`: the clone the
    panel trains today. Either way the clone's learning rate is pinned
    (`clone_lr`), decoupling it from the DQN's: in the panel the two share
    `lr`, which is harmless at one fixed value and wrong once `lr` is searched
    for the fine-tune.
    """
    if source == "current":
        return {"hidden": current_value(tariff, "bc", "hidden"),
                "clone_lr": current_value(tariff, "bc", "lr")}
    if source != "hpo":
        raise ValueError(f"clone source {source!r}: 'hpo' or 'current'")
    w = winner(f"{tariff}_bc", root)
    if w is None:
        raise RuntimeError(
            f"bc_dqn with --clone hpo needs the IL study's winner, and "
            f"{tariff}_bc has none yet: run `search` and `refine` for "
            f"{tariff}:bc first, or pass --clone current.")
    over = w["overrides"]
    out = {"hidden": over.get("hidden", current_value(tariff, "bc", "hidden")),
           "clone_lr": over.get("lr", current_value(tariff, "bc", "lr"))}
    out.update({k: v for k, v in over.items()
                if k.startswith(rb.CLONE_PREFIX) and k != "clone_lr"})
    return out


def recover(study) -> int:
    """Re-queue the trials a killed driver left RUNNING. Returns how many.

    Only ever called under the driver lock, so a RUNNING trial here cannot
    belong to a live process. It is closed as FAIL (the journal keeps it, so
    the interruption stays visible) and its parameters are enqueued again;
    the re-run resolves its finished households from disk.
    """
    optuna = _optuna()
    TS = optuna.trial.TrialState
    n = 0
    for t in study.get_trials(deepcopy=False, states=(TS.RUNNING,)):
        attrs = {"requeued_from": t.number}
        if t.user_attrs.get("incumbent"):
            attrs["incumbent"] = True
        study.tell(t.number, state=TS.FAIL)
        study.enqueue_trial(t.params, user_attrs=attrs)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Sources of work for the scheduler
# ---------------------------------------------------------------------------
class SearchSource:
    """One optuna study, driven ask/tell.

    A trial's households are evaluated rung by rung (`rungs`, cumulative
    counts over the study's fixed household order). At each rung the mean
    validation saving so far is reported; a trial below the pruner's median
    at that rung stops there. The household order is the same for every
    trial, so the comparison at a rung is paired.
    """

    def __init__(self, settings, n_trials, deadline=None, root=None,
                 models=None, seed=0, verbose=True):
        optuna = _optuna()
        self.optuna = optuna
        self.s = settings
        self.name = settings["name"]
        self.tariff, self.method = settings["tariff"], settings["method"]
        self.hh = settings["households"]
        self.rungs = settings["rungs"]
        self.root, self.models = root, models
        self.n_trials = int(n_trials)
        self.deadline = deadline
        self.verbose = verbose
        d = DEFAULTS[self.method]
        self.batch = d["batch"]
        # Group a household's evaluations into one job only where a run is
        # cheap next to preparing the household (the clone).
        self.group = self.batch > 1
        store = storage(self.name, root)
        try:
            n_before = len(optuna.load_study(study_name=self.name, storage=store)
                           .get_trials(deepcopy=False))
        except Exception:           # no such study yet
            n_before = 0
        # Seeded by how much of the study exists: a resumed driver must not
        # restart the sampler's random stream, or every startup trial drawn
        # before the interruption is drawn again.
        sampler = optuna.samplers.TPESampler(
            seed=seed + n_before, multivariate=True, constant_liar=True,
            n_startup_trials=d["startup"])
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=max(d["startup"] // 2, 5), n_warmup_steps=0)
        self.study = optuna.create_study(
            study_name=self.name, storage=store,
            direction="maximize", sampler=sampler, pruner=pruner,
            load_if_exists=True)
        TS = optuna.trial.TrialState
        if not self.study.get_trials(deepcopy=False):
            self.study.enqueue_trial(settings["incumbent_params"],
                                     user_attrs={"incumbent": True})
        self.requeued = recover(self.study)
        self.n_done = len(self.study.get_trials(
            deepcopy=False, states=(TS.COMPLETE, TS.PRUNED)))
        self.open = {}
        self.queue = deque()
        self.t_eval = []
        # Trials in flight at once. A clone study runs two batches; a DQN study
        # as many as keep its share of the workers busy on first rungs, plus
        # one -- set by `search` once it knows the pool. Kept low on purpose:
        # TPE learns only from finished trials.
        self.max_open = 2 * self.batch

    # -- scheduler interface ------------------------------------------------
    @property
    def label(self):
        return self.name

    def _may_open(self):
        if self.n_done + len(self.open) >= self.n_trials:
            return False
        if self.deadline is not None and time.time() >= self.deadline:
            return False
        return len(self.open) < self.max_open

    def _open_trial(self):
        trial = self.study.ask()
        params = suggest(trial, self.method)
        over = trial_overrides(self.s["base"], self.s["fixed"], params)
        tag = cfg_tag(over)
        _write_json(os.path.join(study_dir(self.name, self.root), "runs", tag,
                                 "overrides.json"),
                    {"overrides": over, "params": params, "trial": trial.number})
        trial.set_user_attr("tag", tag)
        trial.set_user_attr("overrides", over)
        st = {"trial": trial, "tag": tag, "over": over, "rung": 0, "rows": {},
              "t0": time.time(), "failed": False}
        self.open[trial.number] = st
        self._enqueue_rung(st)

    def _ev(self, st, ident, seed=0):
        out_root = run_root(self.name, st["tag"], seed, self.root)
        return {"study": self.name, "trial": st["trial"].number,
                "tag": st["tag"], "tariff": self.tariff, "method": self.method,
                "ident": int(ident), "seed": seed, "overrides": st["over"],
                "steps": self.s["steps"], "score_test": False,
                "out_root": out_root,
                "models_root": model_root(out_root, self.root, self.models),
                "root": self.root}

    def _enqueue_rung(self, st):
        lo = 0 if st["rung"] == 0 else self.rungs[st["rung"] - 1]
        hi = self.rungs[st["rung"]]
        for ident in self.hh[lo:hi]:
            ev = self._ev(st, ident)
            row = cached_row(ev)
            if row is not None:
                self.push(ev, row)
                if st["trial"].number not in self.open:
                    return
            else:
                self.queue.append(ev)

    def pull(self):
        """Next job (dict) or None. Opens trials as needed."""
        while not self.queue and self._may_open():
            n = min(self.batch, self.n_trials - self.n_done - len(self.open))
            for _ in range(max(n, 1)):
                if not self._may_open():
                    break
                self._open_trial()
        if not self.queue:
            return None
        ev = self.queue.popleft()
        evals = [ev]
        if self.group:
            keep = deque()
            while self.queue:
                e = self.queue.popleft()
                (evals if e["ident"] == ev["ident"] else keep).append(e)
            self.queue = keep
        return {"tariff": self.tariff, "ident": ev["ident"], "evals": evals}

    def push(self, ev, row):
        st = self.open.get(ev["trial"])
        if st is None:              # trial closed (failed/pruned) meanwhile
            return
        if row.get("error"):
            self._fail(st, row["error"])
            return
        if row.get("job_s"):
            self.t_eval.append(row["job_s"])
        st["rows"][int(ev["ident"])] = row
        hi = self.rungs[st["rung"]]
        if not all(i in st["rows"] for i in self.hh[:hi]):
            return
        vals = [st["rows"][i]["saving_val_a"] for i in self.hh[:hi]]
        value = float(np.mean(vals))
        trial = st["trial"]
        trial.report(value, step=hi)
        last = st["rung"] == len(self.rungs) - 1
        # The incumbent is never pruned: refine and every comparison are
        # paired against it, on all of the study's households.
        if (not last and not trial.user_attrs.get("incumbent")
                and trial.should_prune()):
            self._close(st, value, pruned=True)
        elif last:
            self._close(st, value, pruned=False)
        else:
            st["rung"] += 1
            self._enqueue_rung(st)

    def _close(self, st, value, pruned):
        TS = self.optuna.trial.TrialState
        trial = st["trial"]
        rows = [st["rows"][i] for i in self.hh if i in st["rows"]]
        trial.set_user_attr("hh_saving", {str(r["ident"]): r["saving_val_a"]
                                          for r in rows})
        trial.set_user_attr("val_efc_mean", float(np.mean([r["val_efc"] for r in rows])))
        trial.set_user_attr("converged_share",
                            float(np.mean([r["converged"] for r in rows])))
        trial.set_user_attr("wall_s", time.time() - st["t0"])
        if pruned:
            self.study.tell(trial, state=TS.PRUNED)
        else:
            self.study.tell(trial, value)
        del self.open[trial.number]
        self.n_done += 1
        if self.verbose:
            best = self._best()
            print(f"[{self.name}] trial {trial.number:>3d} "
                  f"{'pruned at ' + str(len(rows)) + ' hh' if pruned else 'complete'}"
                  f"  {value:8.2f}/a  best {best:8.2f}  "
                  f"({self.n_done}/{self.n_trials})", flush=True)

    def _fail(self, st, msg):
        TS = self.optuna.trial.TrialState
        trial = st["trial"]
        trial.set_user_attr("error", msg[:500])
        self.study.tell(trial, state=TS.FAIL)
        del self.open[trial.number]
        # Drop its queued evaluations; in-flight ones are ignored on arrival.
        self.queue = deque(e for e in self.queue if e["trial"] != trial.number)
        print(f"[{self.name}] trial {trial.number} FAILED: {msg[:300]}", flush=True)

    def _best(self):
        try:
            return float(self.study.best_value)
        except ValueError:
            return float("nan")

    def busy(self):
        return bool(self.open) or bool(self.queue) or self._may_open()

    def remaining_s(self, n_jobs):
        """Rough compute still to do, in seconds of one worker."""
        per = (float(np.median(self.t_eval)) if self.t_eval
               else PRIOR_EVAL_S[self.method])
        todo_trials = max(self.n_trials - self.n_done - len(self.open), 0)
        if self.deadline is not None and time.time() >= self.deadline:
            todo_trials = 0
        # Pruning stops about half the trials at the first rung.
        frac = 1.0 if len(self.rungs) == 1 else 0.5 + 0.5 * self.rungs[0] / self.rungs[-1]
        open_left = sum(len(self.hh) - len(st["rows"]) for st in self.open.values())
        return per * (todo_trials * len(self.hh) * frac + open_left)

    def interrupt(self):
        """Close what this driver had open, for a prompt resume."""
        TS = self.optuna.trial.TrialState
        for num, st in list(self.open.items()):
            attrs = {"requeued_from": num}
            if st["trial"].user_attrs.get("incumbent"):
                attrs["incumbent"] = True
            try:
                self.study.tell(st["trial"], state=TS.FAIL)
                self.study.enqueue_trial(st["trial"].params, user_attrs=attrs)
            except Exception:
                pass                # recover() at the next start covers it
        self.open.clear()
        self.queue.clear()


class FixedSource:
    """A plain list of evaluations (refine, confirm). Rows collected in order."""

    def __init__(self, label, evals, verbose=True):
        self.label = label
        self.rows = {}
        self.t_eval = []
        self.verbose = verbose
        self.queue = deque()
        self.total = len(evals)
        method = evals[0]["method"] if evals else "dqn"
        self.prior = PRIOR_EVAL_S[method]
        self.group = method == "bc"
        for ev in evals:
            row = cached_row(ev)
            if row is not None:
                self.rows[eval_key(ev)] = row
            else:
                self.queue.append(ev)
        self.n_cached = len(self.rows)

    def pull(self):
        if not self.queue:
            return None
        ev = self.queue.popleft()
        evals = [ev]
        if self.group:
            keep = deque()
            while self.queue:
                e = self.queue.popleft()
                same = e["ident"] == ev["ident"] and e["tariff"] == ev["tariff"]
                (evals if same else keep).append(e)
            self.queue = keep
        return {"tariff": ev["tariff"], "ident": ev["ident"], "evals": evals}

    def push(self, ev, row):
        self.rows[eval_key(ev)] = row
        if row.get("job_s"):
            self.t_eval.append(row["job_s"])
        if row.get("error") and self.verbose:
            print(f"[{self.label}] FAILED {ev['tag']} s{ev['seed']} "
                  f"Ausgrid {ev['ident']}: {row['error'][:300]}", flush=True)

    def busy(self):
        return len(self.rows) < self.total

    def remaining_s(self, n_jobs):
        per = float(np.median(self.t_eval)) if self.t_eval else self.prior
        return per * (self.total - len(self.rows))

    def interrupt(self):
        self.queue.clear()


# ---------------------------------------------------------------------------
# The scheduler
# ---------------------------------------------------------------------------
def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def _new_pool(n_jobs):
    # Spawned, not forked: a forked worker would inherit the driver's optuna
    # storage handle and file locks, and torch is not fork-safe once it has
    # run. Workers re-import the modules from disk, like the panel's loky ones.
    return cf.ProcessPoolExecutor(max_workers=n_jobs,
                                  mp_context=mp.get_context("spawn"))


def drive(sources, n_jobs: int, job_fn=run_job, report_every_s: float = 600.0):
    """Keep `n_jobs` workers busy with the sources' jobs until all are done.

    Sources are served round-robin, so a study that is waiting on its last
    rung does not idle the pool while another still has work. Identical
    evaluations (two trials that drew the same configuration) are computed
    once: the second waits on the first's result.

    `n_jobs` 0 runs every job inline in this process -- the test path.
    Returns normally when all sources are done; on Ctrl-C / SIGTERM, closes
    the open trials for re-queueing, stops the workers and re-raises
    KeyboardInterrupt.
    """
    sources = list(sources)
    t_start = time.time()
    last_report = t_start
    old_term = signal.signal(signal.SIGTERM, _raise_interrupt)
    pool = _new_pool(n_jobs) if n_jobs > 0 else None
    retry = deque()         # jobs to resubmit after a broken pool
    inflight = {}           # future -> (source, job)
    waiting = {}            # eval key -> [(source, ev)] duplicates
    computing = set()       # eval keys in flight
    rr = 0
    try:
        while True:
            # -- top up -----------------------------------------------------
            cap = n_jobs if n_jobs > 0 else 1
            stalled = 0
            while len(inflight) < cap and stalled < len(sources):
                src = sources[rr % len(sources)]
                rr += 1
                job = src.pull() if src.busy() else None
                if job is None:
                    stalled += 1
                    continue
                stalled = 0
                fresh = []
                for ev in job["evals"]:
                    k = eval_key(ev)
                    if k in computing:
                        waiting.setdefault(k, []).append((src, ev))
                    else:
                        computing.add(k)
                        fresh.append(ev)
                if not fresh:
                    continue
                job = dict(job, evals=fresh)
                if pool is None:
                    _deliver(src, job, job_fn(job), waiting, computing)
                else:
                    inflight[pool.submit(job_fn, job)] = (src, job)
            if not inflight:
                if any(s.busy() for s in sources) and pool is not None:
                    # Busy but nothing to pull: only possible if a source is
                    # waiting on duplicates that are not in flight -- a bug.
                    raise RuntimeError("scheduler stalled with work outstanding")
                if pool is None and any(s.busy() for s in sources):
                    continue
                break
            done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
            broken = False
            for fut in done:
                src, job = inflight.pop(fut)
                try:
                    rows = fut.result()
                except BrokenProcessPool as exc:
                    # A worker was killed (out of memory, a stray `kill`). That
                    # breaks the WHOLE pool, and every job in flight with it,
                    # through no fault of their configurations: rebuild the
                    # pool and resubmit, twice at most, before calling the
                    # evaluations failed.
                    broken = True
                    tries = job.get("tries", 0) + 1
                    if tries <= 2:
                        retry.append((src, dict(job, tries=tries)))
                        continue
                    rows = [_err_row(ev, f"worker: {exc!r}") for ev in job["evals"]]
                except Exception as exc:
                    rows = [_err_row(ev, f"worker: {exc!r}") for ev in job["evals"]]
                _deliver(src, job, rows, waiting, computing)
            if broken or retry:
                if broken:
                    print("a worker died; rebuilding the pool and resubmitting "
                          f"{len(retry) + len(inflight)} job(s)", flush=True)
                    for src_, job_ in inflight.values():
                        retry.append((src_, job_))
                    inflight.clear()
                    pool.shutdown(wait=False, cancel_futures=True)
                    pool = _new_pool(n_jobs)
                while retry:
                    src_, job_ = retry.popleft()
                    inflight[pool.submit(job_fn, job_)] = (src_, job_)
            if time.time() - last_report >= report_every_s:
                last_report = time.time()
                _progress(sources, n_jobs, t_start)
        _progress(sources, n_jobs, t_start, final=True)
    except KeyboardInterrupt:
        print("\ninterrupted: closing open trials for re-queueing, stopping "
              "workers. Rerun the same command to resume.", flush=True)
        for s in sources:
            s.interrupt()
        if pool is not None:
            for p in list(getattr(pool, "_processes", {}).values()):
                try:
                    p.terminate()
                except Exception:
                    pass
            pool.shutdown(wait=False, cancel_futures=True)
            pool = None
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        if pool is not None:
            pool.shutdown(wait=True)


def _deliver(src, job, rows, waiting, computing):
    for ev, row in zip(job["evals"], rows):
        k = eval_key(ev)
        computing.discard(k)
        src.push(ev, row)
        for s2, ev2 in waiting.pop(k, []):
            r2 = dict(row)
            r2.update({key: ev2.get(key) for key in _ID_KEYS})
            s2.push(ev2, r2)


def _progress(sources, n_jobs, t_start, final=False):
    left = sum(s.remaining_s(n_jobs) for s in sources) / max(n_jobs, 1)
    el = time.time() - t_start
    parts = []
    for s in sources:
        if isinstance(s, SearchSource):
            parts.append(f"{s.name} {s.n_done}/{s.n_trials} best {s._best():.1f}")
        else:
            parts.append(f"{s.label} {len(s.rows)}/{s.total}")
    if final:
        print(f"-- done in {el / 60:.1f} min: " + " | ".join(parts), flush=True)
    else:
        eta = datetime.now() + timedelta(seconds=left)
        print(f"-- {el / 60:.0f} min elapsed; ~{left / 3600:.1f} h left, "
              f"finishing ~{eta:%H:%M}: " + " | ".join(parts), flush=True)


# ---------------------------------------------------------------------------
# The driver lock
# ---------------------------------------------------------------------------
def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class DriverLock:
    def __init__(self, root=None):
        self.path = os.path.join(root or HPO_ROOT, "driver.lock")

    def __enter__(self):
        if os.path.exists(self.path):
            try:
                pid = int(_read_json(self.path)["pid"])
            except Exception:
                pid = -1
            if pid > 0 and pid != os.getpid() and _alive(pid):
                raise RuntimeError(
                    f"another HPO driver (pid {pid}) is running; two drivers "
                    f"would re-queue each other's open trials. Wait for it or "
                    f"stop it (kill {pid}); it resumes cleanly.")
        _write_json(self.path, {"pid": os.getpid(), "argv": sys.argv,
                                "started": datetime.now().isoformat(timespec="seconds")})
        return self

    def __exit__(self, *exc):
        try:
            if int(_read_json(self.path)["pid"]) == os.getpid():
                os.remove(self.path)
        except Exception:
            pass
        return False


def driver_running(root=None):
    p = os.path.join(root or HPO_ROOT, "driver.lock")
    if not os.path.exists(p):
        return None
    try:
        info = _read_json(p)
    except Exception:
        return None
    return info if _alive(int(info.get("pid", -1))) else None


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def parse_study(spec: str):
    """'AU:bc' -> ('AU', 'bc')."""
    t, m = spec.split(":")
    if t not in TARIFFS or m not in METHODS:
        raise ValueError(f"bad study {spec!r}: <AU|SI>:<{'|'.join(METHODS)}>")
    return t, m


def search(studies, n_jobs=12, trials=None, hours=None, n_households=None,
           tag="", steps=None, clone_source="hpo", root=None, models=None,
           job_fn=run_job, households=None, verbose=True):
    """Run (or resume) the searches for `studies` [(tariff, method)] together.

    `trials` is the TARGET count of finished (complete + pruned) trials per
    study -- a rerun continues to it, and a larger number extends a study.
    `hours` stops opening new trials after that long; open ones finish.
    """
    # `is not None`: 0 hours (a budget an earlier stage used up) means open
    # nothing new, not "no limit".
    deadline = time.time() + hours * 3600 if hours is not None else None
    sources = []
    # The lock is taken BEFORE the studies are opened: opening one re-queues
    # the trials a dead driver left RUNNING, which is only safe when no live
    # driver can own them.
    with DriverLock(root):
        for t, m in studies:
            name = f"{t}_{m}" + (f"_{tag}" if tag else "")
            hh, st = households, steps
            if name in list_studies(root):
                # Resuming: the study's own households and budget, whatever
                # the defaults are today, unless explicitly given.
                old = read_settings(name, root)
                if hh is None and n_households is None:
                    hh = old["households"]
                if st is None:
                    st = old["steps"]
            s = study_settings(t, m, tag=tag, households=hh,
                               n_households=n_households, steps=st or STEPS,
                               clone_source=clone_source, root=root)
            s = open_settings(s, root)
            n = trials or DEFAULTS[m]["trials"]
            src = SearchSource(s, n, deadline=deadline, root=root, models=models,
                               verbose=verbose)
            if verbose:
                print(f"[{src.name}] {src.n_done}/{n} trials finished, "
                      f"{src.requeued} re-queued from an interrupted run; "
                      f"{len(s['households'])} households, rungs {s['rungs']}",
                      flush=True)
            sources.append(src)
        # A DQN study's share of the pool, in first-rung evaluations: enough
        # open trials to fill it, plus one so a finishing trial never idles a
        # worker. With four RL studies on 12 workers that is 2 per study.
        rl_sources = [s for s in sources if s.batch == 1]
        for s in rl_sources:
            share = max(n_jobs, 1) / len(rl_sources)
            s.max_open = max(2, math.ceil(share / s.rungs[0]) + 1)
        drive(sources, n_jobs, job_fn=job_fn)
    return sources


def _finished_trials(name, root=None):
    optuna = _optuna()
    TS = optuna.trial.TrialState
    st = load_study(name, root)
    return [t for t in st.get_trials(deepcopy=False) if t.state == TS.COMPLETE]


def refine_candidates(name, top=REFINE_TOP, root=None) -> list:
    """(label, tag, overrides, trial number): the incumbent first, then the
    `top` best complete trials by search value (distinct configurations).

    The incumbent is taken from study.json, not from the trials, so it is a
    candidate even when its trial never completed (failed, or the search was
    stopped before it finished)."""
    s = read_settings(name, root)
    trials = _finished_trials(name, root)
    inc_over = s["incumbent"]
    inc_tag = cfg_tag(inc_over)
    inc_num = next((t.number for t in trials
                    if t.user_attrs.get("tag") == inc_tag), None)
    out, seen = [("incumbent", inc_tag, inc_over, inc_num)], {inc_tag}
    for t in sorted(trials, key=lambda t: -t.value):
        tag = t.user_attrs["tag"]
        if tag in seen:
            continue
        seen.add(tag)
        out.append((f"trial {t.number}", tag, t.user_attrs["overrides"], t.number))
        if len(out) >= top + 1:
            break
    return out if len(out) > 1 else []


def _refine_evals(name, seeds, top, root=None, models=None):
    s = read_settings(name, root)
    evals = []
    for label, tag, over, num in refine_candidates(name, top, root):
        for seed in (0,) + tuple(seeds):
            for ident in s["households"]:
                out_root = run_root(name, tag, seed, root)
                evals.append({
                    "study": name, "trial": num, "tag": tag, "label": label,
                    "tariff": s["tariff"], "method": s["method"],
                    "ident": int(ident), "seed": seed, "overrides": over,
                    "steps": s["steps"], "score_test": False,
                    "out_root": out_root,
                    "models_root": model_root(out_root, root, models),
                    "root": root})
    return evals


def refine(names, n_jobs=12, top=REFINE_TOP, seeds=REFINE_SEEDS, root=None,
           models=None, job_fn=run_job, verbose=True):
    """Re-train the incumbent and the top configurations under more seeds;
    write each study's `winner.json`. Validation only."""
    sources = []
    for name in names:
        evals = _refine_evals(name, seeds, top, root, models)
        if not evals:
            print(f"[{name}] no complete trials yet: nothing to refine")
            continue
        sources.append(FixedSource(f"{name} refine", evals, verbose))
    if sources:
        with DriverLock(root):
            drive(sources, n_jobs, job_fn=job_fn)
    out = {}
    for src in sources:
        name = src.label.split()[0]
        out[name] = write_winner(name, src, seeds, top, root)
    return out


def write_winner(name, src, seeds, top, root=None):
    """Pick on the seed-averaged mean validation saving, report the paired
    test of the winner against the incumbent, and write winner.json."""
    from scipy.stats import wilcoxon
    s = read_settings(name, root)
    cands = refine_candidates(name, top, root)
    rows = [r for r in src.rows.values() if not r.get("error")]
    errors = [r for r in src.rows.values() if r.get("error")]
    table = {}
    for label, tag, over, num in cands:
        per_hh = {}
        for r in rows:
            if r["tag"] == tag:
                per_hh.setdefault(int(r["ident"]), []).append(r["saving_val_a"])
        table[tag] = {"label": label, "trial": num, "overrides": over,
                      "per_hh": {str(i): float(np.mean(v)) for i, v in per_hh.items()},
                      "n_seeds": min((len(v) for v in per_hh.values()), default=0)}
        hh_means = list(table[tag]["per_hh"].values())
        table[tag]["mean"] = float(np.mean(hh_means)) if hh_means else float("nan")
    full = {t: v for t, v in table.items()
            if len(v["per_hh"]) == len(s["households"])
            and v["n_seeds"] == 1 + len(seeds)}
    if not full or errors:
        print(f"[{name}] refine incomplete ({len(errors)} failed evaluation(s)); "
              f"rerun refine to finish -- no winner written")
        return None
    best_tag = max(full, key=lambda t: full[t]["mean"])
    inc_tag = cands[0][1]
    d = np.array([full[best_tag]["per_hh"][k] - full[inc_tag]["per_hh"][k]
                  for k in full[inc_tag]["per_hh"]])
    p = (float(wilcoxon(d).pvalue) if (d != 0).sum() >= 6 else float("nan"))
    w = {"study": name, "tariff": s["tariff"], "method": s["method"],
         "tag": best_tag, "label": full[best_tag]["label"],
         "trial": full[best_tag]["trial"],
         "overrides": full[best_tag]["overrides"],
         "mean_saving_val_a": full[best_tag]["mean"],
         "incumbent_tag": inc_tag,
         "incumbent_saving_val_a": full[inc_tag]["mean"],
         "delta_vs_incumbent": float(d.mean()) if len(d) else 0.0,
         "p_vs_incumbent": p, "n_better": int((d > 0).sum()),
         "seeds": [0] + list(seeds), "households": s["households"],
         "candidates": table,
         "written": datetime.now().isoformat(timespec="seconds")}
    _write_json(os.path.join(study_dir(name, root), "winner.json"), w)
    print(f"[{name}] winner: {w['label']} ({best_tag}), "
          f"{w['mean_saving_val_a']:.2f}/a vs incumbent "
          f"{w['incumbent_saving_val_a']:.2f}/a (seed-averaged validation; "
          f"Δ {w['delta_vs_incumbent']:+.2f}, p {p:.3f}, "
          f"{w['n_better']}/{len(d)} households better)", flush=True)
    return w


def winner(name, root=None):
    p = os.path.join(study_dir(name, root), "winner.json")
    return _read_json(p) if os.path.exists(p) else None


def _confirm_evals(name, seeds, units, root=None, models=None):
    """Winner and current settings on the study units, scored on the TEST year.

    The current settings are what the panel trains: overrides None, i.e. TUNED
    as it stands. Their seed-0 runs are usually already on disk in the panel
    (`results_local/rl_screen`); `cached_row` finds them there under the same
    digest, so only what the panel never trained is computed.
    """
    w = winner(name, root)
    if w is None:
        return []
    s = read_settings(name, root)
    t, m = s["tariff"], s["method"]
    current = s["current"]
    # Nothing to confirm when refine kept the incumbent and the incumbent IS
    # the panel's configuration (bc, dqn). For bc_dqn it is not: the incumbent
    # fine-tunes the study's clone, the panel its own.
    if m != "bc_dqn" and w["tag"] == w["incumbent_tag"]:
        print(f"[{name}] refine kept the current settings: nothing to confirm")
        return []
    evals = []
    for seed in seeds:
        for ident in units:
            for label, tag, over, out_root in (
                    ("winner", w["tag"], w["overrides"],
                     run_root(name, w["tag"], seed, root, confirm=True)),
                    ("current", "current", current,
                     run_root(name, "current", seed, root, confirm=True))):
                ev = {"study": name, "trial": None, "tag": tag, "label": label,
                      "tariff": t, "method": m, "ident": int(ident),
                      "seed": seed, "overrides": over, "steps": s["steps"],
                      "score_test": True, "out_root": out_root,
                      "models_root": model_root(out_root, root, models),
                      "root": root}
                if label == "current":
                    # The panel already trained this, if TUNED has not moved
                    # since the study was created and the run sat at the
                    # default budget (stragglers retrained at 1M do not
                    # match, and are retrained here at the study's budget).
                    # Read only: a worker adds the baseline and never writes.
                    panel = dict(ev, out_root=rb.OUT, read_only=True)
                    if stored_result(panel) is not None:
                        ev = panel
                evals.append(ev)
    return evals


def confirm(names, n_jobs=12, seeds=CONFIRM_SEEDS, units=None, root=None,
            models=None, job_fn=run_job, verbose=True):
    """Score each study's winner against the current settings on the test
    year of the study households; write `confirm.json`. A verdict only."""
    units = units or study_units()
    sources = []
    for name in names:
        evals = _confirm_evals(name, seeds, units, root, models)
        if not evals:
            print(f"[{name}] no winner yet: run refine first")
            continue
        sources.append(FixedSource(f"{name} confirm", evals, verbose))
    if sources:
        # Idle baselines on the study units, which no search has touched.
        with DriverLock(root):
            drive(sources, n_jobs, job_fn=job_fn)
    for src in sources:
        name = src.label.split()[0]
        rows = list(src.rows.values())
        _write_json(os.path.join(study_dir(name, root), "confirm.json"),
                    {"study": name, "seeds": list(seeds), "units": list(units),
                     "rows": rows,
                     "written": datetime.now().isoformat(timespec="seconds")})
        bad = [r for r in rows if r.get("error")]
        if bad:
            print(f"[{name}] confirm: {len(bad)} failed evaluation(s); rerun")
    return sources


# ---------------------------------------------------------------------------
# Frames for the notebook (read-only; safe while a driver runs)
# ---------------------------------------------------------------------------
def trials_frame(name, root=None):
    """One row per trial: state, value, parameters, bookkeeping."""
    import pandas as pd
    st = load_study(name, root)
    rows = []
    for t in st.get_trials(deepcopy=False):
        r = {"study": name, "number": t.number, "state": t.state.name,
             "value": t.value,
             "incumbent": bool(t.user_attrs.get("incumbent")),
             "requeued_from": t.user_attrs.get("requeued_from"),
             "tag": t.user_attrs.get("tag"),
             "n_households": len(t.user_attrs.get("hh_saving", {})),
             "last_step": max(t.intermediate_values) if t.intermediate_values else None,
             "last_intermediate": (t.intermediate_values[max(t.intermediate_values)]
                                   if t.intermediate_values else None),
             "val_efc_mean": t.user_attrs.get("val_efc_mean"),
             "converged_share": t.user_attrs.get("converged_share"),
             "wall_s": t.user_attrs.get("wall_s"),
             "start": t.datetime_start, "complete": t.datetime_complete,
             "error": t.user_attrs.get("error")}
        r.update({f"p_{k}": v for k, v in t.params.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def history_frame(name, root=None):
    """The search's progress, relative to the incumbent, trial by trial.

    `delta` is the trial's value minus the INCUMBENT's at the same rung: for a
    complete trial over all households, for a pruned one over the first-rung
    households it was stopped on -- the same households either way, so every
    point is a paired comparison. `best` is the running best complete delta.
    """
    import pandas as pd
    st = load_study(name, root)
    trials = st.get_trials(deepcopy=False)
    inc = next((t for t in trials if t.user_attrs.get("incumbent")
                and t.intermediate_values), None)
    if inc is None:
        return pd.DataFrame()
    ref = dict(inc.intermediate_values)
    rows = []
    for t in sorted(trials, key=lambda t: t.number):
        if t.state.name not in ("COMPLETE", "PRUNED") or not t.intermediate_values:
            continue
        step = max(t.intermediate_values)
        if step not in ref:
            continue
        rows.append({"study": name, "number": t.number, "state": t.state.name,
                     "incumbent": bool(t.user_attrs.get("incumbent")),
                     "step": step, "value": t.intermediate_values[step],
                     "delta": t.intermediate_values[step] - ref[step],
                     "complete_at": t.datetime_complete})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    best = df["delta"].where(df["state"] == "COMPLETE")
    df["best"] = best.cummax().ffill()
    return df


def param_kind(method, name):
    return next(k for n, k, _ in _space(method) if n == name)


def slice_frame(name, root=None):
    """Long frame (param, x, value) over the COMPLETE trials, for the
    objective-versus-parameter panels."""
    import pandas as pd
    tf = trials_frame(name, root)
    if tf.empty:
        return pd.DataFrame()
    s = read_settings(name, root)
    tf = tf[tf.state == "COMPLETE"]
    rows = []
    for pname, kind, _ in _space(s["method"]):
        col = f"p_{pname}"
        if col not in tf:
            continue
        for _, r in tf.iterrows():
            rows.append({"study": name, "param": pname, "kind": kind,
                         "x": r[col], "value": r["value"],
                         "incumbent": r["incumbent"]})
    return pd.DataFrame(rows)


def household_frame(name, root=None):
    """Per (finished trial, household) validation saving, from user attrs."""
    import pandas as pd
    st = load_study(name, root)
    rows = []
    for t in st.get_trials(deepcopy=False):
        for i, v in t.user_attrs.get("hh_saving", {}).items():
            rows.append({"study": name, "number": t.number, "state": t.state.name,
                         "incumbent": bool(t.user_attrs.get("incumbent")),
                         "ident": int(i), "saving_val_a": v})
    return pd.DataFrame(rows)


def importance_frame(name, root=None, min_trials=12, seed=0):
    """fANOVA importance of each parameter over the COMPLETE trials (empty
    below `min_trials`: a forest over a dozen points is a guess)."""
    import pandas as pd
    optuna = _optuna()
    st = load_study(name, root)
    TS = optuna.trial.TrialState
    if len(st.get_trials(deepcopy=False, states=(TS.COMPLETE,))) < min_trials:
        return pd.DataFrame(columns=["study", "param", "importance"])
    imp = optuna.importance.get_param_importances(
        st, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed))
    return pd.DataFrame([{"study": name, "param": k, "importance": v}
                         for k, v in imp.items()])


def refine_frame(name, root=None):
    """Per (candidate, household): seed-averaged validation saving."""
    import pandas as pd
    w = winner(name, root)
    if w is None:
        return pd.DataFrame()
    rows = []
    for tag, c in w["candidates"].items():
        for i, v in c["per_hh"].items():
            rows.append({"study": name, "tag": tag, "label": c["label"],
                         "trial": c["trial"], "ident": int(i),
                         "saving_val_a": v, "winner": tag == w["tag"],
                         "incumbent": tag == w["incumbent_tag"]})
    return pd.DataFrame(rows)


def confirm_frame(name, root=None):
    import pandas as pd
    p = os.path.join(study_dir(name, root), "confirm.json")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.DataFrame(_read_json(p)["rows"])
    if "error" in df:
        df = df[df["error"].isna()]
    return df


def confirm_summary(names=None, root=None):
    """Winner minus current on the test year, per study: the paired change in
    what a household pays (bill + lifetime wear), seed-averaged per household.
    Positive `gain` = the tuned settings save more. Holm over the studies."""
    import pandas as pd
    from scipy.stats import wilcoxon
    names = names or list_studies(root)
    rows = []
    for name in names:
        df = confirm_frame(name, root)
        if df.empty:
            continue
        w = (df.groupby(["ident", "label"])["test_net"].mean()
             .unstack("label").dropna())
        if not {"winner", "current"} <= set(w.columns):
            continue
        d = (w["current"] - w["winner"]).to_numpy()
        lo, hi = _boot_ci(d)
        rows.append({"study": name, "tariff": df["tariff"].iloc[0],
                     "method": df["method"].iloc[0], "n": len(d),
                     "seeds": int(df["seed"].nunique()),
                     "gain": float(d.mean()), "lo": lo, "hi": hi,
                     "p": float(wilcoxon(d).pvalue) if (d != 0).sum() >= 6 else float("nan"),
                     "n_better": int((d > 0).sum()),
                     "efc_current": float(df[df.label == "current"]["efc"].mean()),
                     "efc_winner": float(df[df.label == "winner"]["efc"].mean()),
                     "d": d})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = _holm(out["p"].tolist())
    return out


def _boot_ci(v, n_boot=10_000, alpha=0.05, seed=0):
    v = np.asarray(v, dtype=float)
    if len(v) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def _holm(p):
    return rb.holm(p)


def status_frame(root=None):
    """One row per study: progress and the current verdicts."""
    import pandas as pd
    rows = []
    for name in list_studies(root):
        s = read_settings(name, root)
        try:
            tf = trials_frame(name, root)
        except Exception as exc:          # journal mid-write: try again later
            rows.append({"study": name, "error": repr(exc)})
            continue
        cnt = tf["state"].value_counts() if not tf.empty else {}
        inc = tf[tf.incumbent & (tf.state == "COMPLETE")]
        comp = tf[tf.state == "COMPLETE"]
        w = winner(name, root)
        rows.append({
            "study": name, "households": len(s["households"]),
            "complete": int(cnt.get("COMPLETE", 0)),
            "pruned": int(cnt.get("PRUNED", 0)),
            "failed": int(cnt.get("FAIL", 0)),
            "running": int(cnt.get("RUNNING", 0)),
            "waiting": int(cnt.get("WAITING", 0)),
            "incumbent_val": float(inc["value"].iloc[0]) if len(inc) else np.nan,
            "best_val": float(comp["value"].max()) if len(comp) else np.nan,
            "winner": w["label"] if w else None,
            "winner_delta_val": w["delta_vs_incumbent"] if w else np.nan,
            # Scored test-year rows (failed evaluations excluded): 0 until
            # confirm has run cleanly for at least part of the study.
            "confirmed_rows": len(confirm_frame(name, root)),
        })
    return pd.DataFrame(rows)


def tuned_entries(root=None) -> str:
    """The refine winners in `run_rl_benchmark.TUNED` form, to paste by hand
    -- the same deliberate, reviewable step the grid's winners went through.
    Only the keys that differ from the panel's defaults are listed."""
    lines = []
    for name in list_studies(root):
        w = winner(name, root)
        if w is None:
            continue
        t, m = w["tariff"], w["method"]
        base = rl.TrainConfig()
        bco = rl.BCOptions()
        keep = {}
        for k, v in w["overrides"].items():
            if k.startswith(rb.CLONE_PREFIX):
                dv = getattr(bco, k[len(rb.CLONE_PREFIX):])
            else:
                dv = getattr(base, k)
            if v != dv:
                keep[k] = (round(v, 6) if isinstance(v, float) else v)
        lines.append(f"    ({t!r}, {m!r}): {keep!r},")
    return "\n".join(lines)
