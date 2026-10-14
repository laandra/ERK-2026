"""What MORE INPUT buys a learned controller: the feature study.

    python3 run_rl_features.py all --jobs 12        # everything, in order
    python3 run_rl_features.py status               # progress and ETA, any time
    python3 run_rl_features.py report

The panel (`run_rl_benchmark`) asks what a forecast is worth to a learner and
answers through one observation contract, `fc_h24`: four lookahead bins of the
price list and of a 14-day median. Every other controller in the study reads
more than that (`rl_features`). This driver trains the same three learners
(dqn, bc, bc_dqn), on the same households, split, hyperparameters and
settlement, under WIDER contracts, and prices each addition as a paired
difference against the panel's own `fc_h24` runs, which are read and never
rewritten:

    fine        the lookahead by the hour (24 bins) instead of four bins
    ar          the forecast of the best MPC arm -- the 14-day median plus the
                study's fitted residual AR -- re-issued at every interval
    rank        where the price ranks: the price rules' own readings
    net         the forecast as deficit, surplus and the deficit's cost
    hist        the meter's own past: lags, today's departure from typical,
                the absence signal, the roof's clear-sky envelope
    cal         the tariff's working-day calendar (holidays included)
    peak        SI only: the draw's own top and its forecast against the line
    all         every one of the above
    all_coarse  the same, on four bins
    all_no_*    `all` with one addition taken out (the clone only: seconds)

The phases, each resumable -- a result is one JSON under a config digest, so
rerunning any command recomputes nothing that is on disk and current:

    screen    train every (variant, method, household); VALIDATION numbers
              only. The result files carry no test number at all, so the
              choice below cannot be made on one even by accident.
    retrain   the reinforcement runs that never settled, at 1M steps -- what
              the panel did for its own, so the comparison is like for like.
    select    per (tariff, method), the contract with the lowest mean
              validation cost net of wear, `fc_h24` included. Written to
              `selection.json`; nothing is selected on the test year.
    score     every screened model on the TEST year, from its saved weights,
              through `run_rl_benchmark.score_test_year`.
    report    the selected contract against `fc_h24` (the headline, one test
              per cell, Holm over the cells) and every addition on its own
              (the value-of-information table, Holm within a cell).

TO PAUSE: `python3 run_rl_features.py pause` -- every worker finishes the run
it is on and stops (a kill works too, and loses at most the runs in flight).
`python3 run_rl_features.py all --jobs 12` again resumes.

THE MEASURE THE CHOICE IS MADE ON differs by tariff, because the bill does.
On AU it is the panel's: the closed bill of the held-out weeks plus lifetime
wear. On SI that number is priced against the no-battery contract and cannot
see a contract the agent lowers -- which is where an input about peaks would
pay -- so SI is chosen on `rl_reward`'s contract-aware measure instead: the
held-out weeks of the agent's own full-period trajectory under the contract
its own peaks agree, standing charge included (`contract_validation`). Both
numbers are stored for both tariffs' runs where they exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import rl_features as rf                                         # noqa: E402
import rl_reward as rr                                           # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_local", "rl_features")
SCREEN = os.path.join(OUT, "_screen")          # validation only
MODELS = os.path.join(rb.MODELS, "features")   # gitignored with rl_models/
SELECTION = os.path.join(OUT, "selection.json")
PAUSE = os.path.join(OUT, "PAUSE")

BASELINE = "fc_h24"                            # the panel's contract, read-only
METHODS = ("bc", "dqn", "bc_dqn")
# Both budgets a reinforcement result may legitimately carry: the screen's, and
# the one its stragglers are re-trained at. A result under either is current.
STEPS, RETRAIN_STEPS = 500_000, 1_000_000
H, N_TRAIN = rb.H, rb.N_TRAIN


# ---------------------------------------------------------------------------
# The observation contracts
# ---------------------------------------------------------------------------
def feature_specs(tariff: str) -> dict:
    """{variant: FeatureSpec}. Every one is `fc_h24` plus something, so each
    paired difference against the panel prices exactly that something."""
    peak = tariff == "SI"
    base = dict(horizon=48, peak_features=peak)
    groups = tuple(g for g in rf.GROUPS if g != "peak" or peak)
    v = {
        "fine": rl.FeatureSpec(**base, n_bins=24),
        "ar": rl.FeatureSpec(**base, forecast="ar"),
        "rank": rl.FeatureSpec(**base, groups=("rank",)),
        "net": rl.FeatureSpec(**base, groups=("net",)),
        "hist": rl.FeatureSpec(**base, groups=("history",)),
        "cal": rl.FeatureSpec(**base, groups=("calendar",)),
        "all": rl.FeatureSpec(**base, n_bins=24, forecast="ar", groups=groups),
        "all_coarse": rl.FeatureSpec(**base, forecast="ar", groups=groups),
        # `all` less one addition: what each is worth GIVEN the others, which
        # is not what it is worth alone when two of them carry the same fact
        # (the AR and the raw lags both say "today is not typical"). `all`
        # less the hourly lookahead is `all_coarse`, above.
        "all_no_ar": rl.FeatureSpec(**base, n_bins=24, groups=groups),
    }
    for g in groups:
        v[f"all_no_{SHORT[g]}"] = rl.FeatureSpec(
            **base, n_bins=24, forecast="ar",
            groups=tuple(x for x in groups if x != g))
    if peak:
        v["peak"] = rl.FeatureSpec(**base, groups=("peak",))
    # FOLLOW-UPS, added after the test year had been read (see FOLLOW_UP).
    v["rel"] = rl.FeatureSpec(**base, price_reference="trailing")
    v["all_coarse_rel"] = rl.FeatureSpec(**base, forecast="ar", groups=groups,
                                         price_reference="trailing")
    return v


# Contracts that test an EXPLANATION, trained after the test year was read and
# therefore never a candidate in `select` (whose choice was written before
# they existed) and never part of the headline. The main result -- every input
# at once gains on the test year, one input alone does not, and the price
# rank is what the combination leans on -- has a reading: the scored year's
# prices sit 19 % (AU) and 32 % (SI) above the training years', so a network
# shown price LEVELS is extrapolating and one shown ranks is not. If that is
# right, the panel's own contract with its prices merely re-expressed against
# the trailing week (`rel`) should recover part of the gain with no new
# information at all, and the full contract should keep its gain with its
# price levels taken away (`all_coarse_rel`).
FOLLOW_UP = ("rel", "all_coarse_rel")

SHORT = {"rank": "rank", "net": "net", "history": "hist", "calendar": "cal",
         "peak": "peak"}

# The additions asked about one at a time, and what `all` loses without each.
SINGLES = ["fine", "ar", "rank", "net", "hist", "cal", "peak"]
LEAVE_OUT = {"fine": "all_coarse", "ar": "all_no_ar", "rank": "all_no_rank",
             "net": "all_no_net", "hist": "all_no_hist", "cal": "all_no_cal",
             "peak": "all_no_peak"}
LABELS = {
    "fine": "hourly lookahead", "ar": "AR forecast, re-issued",
    "rank": "price rank", "net": "forecast net load", "hist": "meter history",
    "cal": "working-day calendar",
    "peak": "peak outlook (SI)", "all": "all of them",
    "all_coarse": "all, four bins",
    "rel": "prices against the trailing week",
    "all_coarse_rel": "all, four bins, prices against the trailing week",
}


def methods_for(variant: str) -> tuple:
    """The leave-one-out contracts are trained for the clone only: a clone is
    seconds, a reinforcement run two minutes, and seven more contracts under
    all three learners would double the study to answer a secondary question."""
    return ("bc",) if variant.startswith("all_no_") else METHODS


def jobs_for(ids, tariffs, variants=None, methods=None) -> list:
    out = []
    for tariff in tariffs:
        known = feature_specs(tariff)
        for ident in ids:
            for method in (methods or METHODS):
                for variant in (variants or list(known)):
                    if variant in known and method in methods_for(variant):
                        out.append((tariff, variant, method, int(ident)))
    return out


def baseline_jobs(ids, tariffs) -> list:
    """The panel's own `fc_h24` models, which are never retrained here but on
    SI have to be MEASURED on the axis the choice is made on."""
    return [(t, BASELINE, m, int(i)) for t in tariffs if t in CONTRACT_MEASURED
            for i in ids for m in METHODS]


# ---------------------------------------------------------------------------
# Per-household preparation: the panel's, plus the wider inputs
# ---------------------------------------------------------------------------
# The AR tables and the household's past are built by `run_rl_benchmark`
# (`ar_forecasts`, `add_ar`, `add_context`), lazily, for whichever contract
# asks: they moved there when the wide contract became the panel's.
ar_forecasts = rb.ar_forecasts


def prepare(ident, tariff: str, ar: bool = True) -> dict:
    """`rb.prepare_household` plus what `rb.feature_inputs` asks for on behalf
    of a feature-study spec: the AR forecasts and the household's own past.

    `ar=False` leaves the AR tables out, for the pooled contracts that do not
    read them: outside the 30 study units no AR has been fitted, and asking
    for one is minutes of quantile regression per household."""
    return enrich(rb.prepare_household(ident, tariff), ar)


def enrich(prep: dict, ar: bool = True) -> dict:
    """Add the wider inputs to a household already prepared -- by the panel's
    `prepare_household`, or by another study's that re-bills the same frames
    (`run_rl_contract.prepare`). Eager where `rb.feature_inputs` is lazy: a
    worker that will train a dozen contracts pays for them once, up front."""
    rb.add_context(prep)
    if ar:
        rb.add_ar(prep)
    return prep


# What the re-issued AR is worth as a FORECAST, before it is worth anything as
# an input: by lead time, in the training years (where its coefficients were
# fitted) and in the scored year (where they were not).
SKILL = os.path.join(OUT, "ar_skill.csv")
SKILL_LEADS = (1, 2, 4, 6, 12, 24, 36, 48)


def ar_skill(ids, refresh: bool = False) -> pd.DataFrame:
    """Share of the 14-day median's absolute error the re-issued AR removes,
    per household, channel, period and lead (in intervals). Cached as a CSV."""
    if os.path.exists(SKILL) and not refresh:
        out = pd.read_csv(SKILL)
        if set(out.ident) >= {int(i) for i in ids}:
            return out[out.ident.isin([int(i) for i in ids])]
    rows = []
    s0 = N_TRAIN * H
    for ident in ids:
        frames = hs.load_study_frames(
            os.path.join(rb.DATA, f"Ausgrid {ident}.csv"), H=H,
            delta_t=rb.DELTA_T, n_train=N_TRAIN, n_sim=rb.N_SIM)
        tables = dict(zip(("load", "pv"), ar_forecasts(frames)))
        n = len(tables["load"])
        for ch, col in (("load", "Energy_Consumption"), ("pv", "Energy_Generation")):
            y = frames["df_all_kwh"][col].to_numpy(dtype=float)[:n]
            median = rl.median14_forecast(y, H)
            # The first fifteen days carry the bare median in both columns.
            for period, (a, b) in (("train", (16 * H, s0)), ("test", (s0, n))):
                for lead in SKILL_LEADS:
                    t = np.arange(a, b - lead)
                    err_ar = np.abs(y[t + lead] - tables[ch][t, lead - 1]).mean()
                    err_m = np.abs(y[t + lead] - median[t + lead]).mean()
                    rows.append({"ident": int(ident), "channel": ch,
                                 "period": period, "lead": lead,
                                 "mae_ar": err_ar, "mae_median": err_m,
                                 "skill": 1.0 - err_ar / max(err_m, 1e-12)})
    out = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    out.to_csv(SKILL, index=False)
    return out


# ---------------------------------------------------------------------------
# What is on disk, and whether it is current
# ---------------------------------------------------------------------------
def _path(root, tariff, variant, method, ident) -> str:
    return os.path.join(root, tariff, f"{variant}__{method}", f"{ident}.json")


def _read(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None                                   # half-written: recompute


def digests(tariff, variant, method, seed: int = 0) -> set:
    """The digests a current result may carry: one per legitimate budget."""
    spec = feature_specs(tariff)[variant]
    return {rb.run_digest(rb.make_config(steps, seed), spec, tariff, method)
            for steps in (STEPS, RETRAIN_STEPS)}


def screened(tariff, variant, method, ident):
    """The validation-only result for one run, if on disk, current and -- on a
    tariff chosen on the contract-aware measure -- carrying that measure.

    For the panel's `fc_h24` the file holds the measure alone, keyed to the
    digest of the panel result it was taken from."""
    res = _read(_path(SCREEN, tariff, variant, method, ident))
    if res is None:
        return None
    if variant == BASELINE:
        panel = _read(_path(rb.OUT, tariff, BASELINE, method, ident))
        return res if panel and res.get("digest") == panel.get("digest") else None
    if res.get("digest") not in digests(tariff, variant, method):
        return None
    if tariff in CONTRACT_MEASURED and "val_contract_net" not in res:
        return None
    return res


def scored(tariff, variant, method, ident):
    """The test-year result, if on disk and scored from the CURRENT model."""
    res = _read(_path(OUT, tariff, variant, method, ident))
    base = screened(tariff, variant, method, ident)
    if res is None or base is None or res.get("digest") != base["digest"]:
        return None
    return res


def paused() -> bool:
    return os.path.exists(PAUSE)


# ---------------------------------------------------------------------------
# The validation measure a tariff with a contract is chosen on
# ---------------------------------------------------------------------------
CONTRACT_MEASURED = ("SI",)


def contract_validation(prep, net, fb, spec) -> dict:
    """The held-out weeks WITH the contract the agent's own peaks agree.

    `run_rl_contract._measure`, for a model whose features need more than the
    two forecast arrays that function passes: the reward class is used as a
    meter for a ready-made `Policy` (its own documented use for the rules), so
    nothing about the measure is restated here and the panel clone of Ausgrid
    138 reads 67.41 EUR through either. Three passes, each billed under the
    contract the one before set, as there.

    Leaves the training environment as it found it: the passes re-agree the
    contract on the shared env, and the next run trains on that env.
    """
    train = prep["train"]
    n_val = sum(b - a for a, b in rb.VAL_BLOCKS)
    try:
        meter = rr.reward_class(prep["tariff"], rr.REFERENCE_CONTRACT)(
            train, None, None, val_blocks=rb.VAL_BLOCKS,
            soc_init=rb.SOC_INIT_USABLE, respect_peak=True)
        policy = rl.LearnedPolicy(net, fb, True,
                                  **rb.feature_inputs(prep, spec, "train"))
        v = meter.settle_contract(policy, passes=3)
    finally:
        train["env"].clear_achieved_power()
    return {
        "val_contract_net": float(v["cost_eur_closed"])
        + rb.val_wear_fn(prep["tariff"])(float(v["efc"]), n_val),
        "val_contract_bill": float(v["cost_eur"]),
        "val_contract_eur": float(v["contract_eur"]),
        "val_contract_efc": float(v["efc"]),
    }


def _write(path, res) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, path)


def _measure_baseline(prep, method) -> None:
    """The contract-aware measure of the panel's own `fc_h24` model."""
    tariff, ident = prep["tariff"], prep["ident"]
    panel = _read(_path(rb.OUT, tariff, BASELINE, method, ident))
    net, fb, spec, _ = rl.load_model(os.path.join(HERE, panel["model_path"]))
    _write(_path(SCREEN, tariff, BASELINE, method, ident),
           {"ident": str(ident), "tariff": tariff, "variant": BASELINE,
            "method": method, "digest": panel["digest"],
            **contract_validation(prep, net, fb, spec)})


# ---------------------------------------------------------------------------
# Phases 1 and 2: train, validation only
# ---------------------------------------------------------------------------
def _train_worker(group, steps, quiet):
    """Every run of one (tariff, household). The clone first: it is seconds,
    so every contract has its first number minutes into the sweep."""
    tariff, ident, items = group
    specs = feature_specs(tariff)
    cfg = rb.make_config(steps, 0)
    prep, out = None, []
    for variant, method in items:
        if paused():
            out.append((tariff, variant, method, ident, "paused"))
            continue
        try:
            if prep is None:
                prep = prepare(ident, tariff)
            if variant == BASELINE:
                _measure_baseline(prep, method)
                out.append((tariff, variant, method, ident, None))
                continue
            res = rb.run_one(prep, variant, method, cfg, verbose=not quiet,
                             out_root=SCREEN, models_root=MODELS,
                             score_test=False, specs=specs)
            assert "cost_eur_closed" not in res, "the screen touched the test year"
            if tariff in CONTRACT_MEASURED and "val_contract_net" not in res:
                net, fb, spec, _ = rl.load_model(
                    os.path.join(HERE, res["model_path"]))
                res.update(contract_validation(prep, net, fb, spec))
                _write(_path(SCREEN, tariff, variant, method, ident), res)
            out.append((tariff, variant, method, ident, None))
        except Exception as exc:                 # one run must not sink the sweep
            out.append((tariff, variant, method, ident, repr(exc)))
    return out


def _run_groups(worker, groups, n_jobs, label, *args):
    """Run household groups across processes, printing an ETA as they land."""
    from joblib import Parallel, delayed

    n_runs = sum(len(g[2]) for g in groups)
    if not groups:
        print(f"{label}: nothing to do")
        return []
    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    n_jobs = max(1, min(n_jobs, len(groups)))
    print(f"{label}: {n_runs} run(s) over {len(groups)} household-tariff "
          f"group(s) on {n_jobs} worker(s)", flush=True)
    t0, done, flat = time.time(), 0, []
    stream = Parallel(n_jobs=n_jobs, backend="loky",
                      return_as="generator_unordered")(
        delayed(worker)(g, *args) for g in groups)
    for part in stream:
        flat += part
        done += len(part)
        elapsed = time.time() - t0
        # Groups finish in waves of `n_jobs`, so a per-run rate is only honest
        # as elapsed worker-time over runs landed.
        left = (n_runs - done) * elapsed / max(done, 1)
        print(f"  {label}: {done}/{n_runs} runs, {elapsed/60:.0f} min elapsed, "
              f"~{left/60:.0f} min left, ETA "
              f"{datetime.now() + timedelta(seconds=left):%H:%M}", flush=True)
    failed = [r for r in flat if r[4] not in (None, "paused")]
    held = [r for r in flat if r[4] == "paused"]
    print(f"{label}: {len(flat) - len(failed) - len(held)} run(s) in "
          f"{(time.time() - t0)/60:.1f} min")
    if held:
        print(f"{label}: PAUSED with {len(held)} run(s) outstanding -- remove "
              f"{os.path.relpath(PAUSE, HERE)} (`resume`) and rerun")
    for t, v, m, i, exc in failed:
        print(f"  FAILED {t} {v} {m} Ausgrid {i}: {exc}")
    return flat


def _grouped(jobs) -> list:
    todo = {}
    for tariff, variant, method, ident in jobs:
        todo.setdefault((tariff, ident), []).append((variant, method))
    return [(t, i, items) for (t, i), items in todo.items()]


def screen(ids, tariffs, n_jobs, quiet=True, variants=None, methods=None):
    jobs = jobs_for(ids, tariffs, variants, methods) + baseline_jobs(ids, tariffs)
    todo = [j for j in jobs if screened(*j) is None]
    print(f"screen: {len(jobs) - len(todo)} of {len(jobs)} run(s) cached")
    flat = _run_groups(_train_worker, _grouped(todo), n_jobs, "screen",
                       STEPS, quiet)
    return all(r[4] is None for r in flat)


def unsettled(ids, tariffs) -> list:
    """Reinforcement runs whose validation was still improving at the screen's
    budget -- the panel's own criterion (`rb.unconverged_runs`), read off the
    result files, and only for runs not already at the larger budget."""
    out = []
    for job in jobs_for(ids, tariffs):
        if job[2] == "bc":
            continue
        res = screened(*job)
        if (res is not None and not res.get("train_converged")
                and res["train_config"]["total_steps"] < RETRAIN_STEPS):
            out.append(job)
    return out


def retrain(ids, tariffs, n_jobs, quiet=True):
    todo = unsettled(ids, tariffs)
    print(f"retrain: {len(todo)} unsettled reinforcement run(s) at "
          f"{RETRAIN_STEPS:,} steps")
    flat = _run_groups(_train_worker, _grouped(todo), n_jobs, "retrain",
                       RETRAIN_STEPS, quiet)
    return all(r[4] is None for r in flat)


# ---------------------------------------------------------------------------
# Phase 3: choose on validation
# ---------------------------------------------------------------------------
def _val_net(res, tariff) -> float:
    """Validation cost net of wear, recomputed with today's wear formula --
    the panel's clone files still carry the per-cycle price they were scored
    under before the NPV switch (`rl_hpo._row` documents the same trap)."""
    n_val = sum(b - a for a, b in rb.VAL_BLOCKS)
    return float(res["val_cost_closed"]) + rb.val_wear_fn(tariff)(
        float(res["val_efc"]), n_val)


def validation_frame(ids, tariffs) -> pd.DataFrame:
    """One row per (tariff, variant, method, household) with a validation
    number: the screen's, and the panel's `fc_h24`."""
    rows = []
    for tariff in tariffs:
        for ident in ids:
            for method in METHODS:
                base = _read(_path(rb.OUT, tariff, BASELINE, method, ident))
                if base is None:
                    continue
                if tariff in CONTRACT_MEASURED:
                    extra = screened(tariff, BASELINE, method, ident)
                    if extra is None:
                        continue                  # not yet measured on SI's axis
                    base = {**base, **extra}
                rows.append((tariff, BASELINE, method, int(ident), base))
        for job in jobs_for(ids, [tariff]):
            res = screened(*job)
            if res is not None:
                rows.append((*job, res))
    out = pd.DataFrame([{
        "tariff": t, "variant": v, "method": m, "ident": i,
        "val_net": _val_net(r, t), "val_efc": float(r["val_efc"]),
        "val_contract_net": r.get("val_contract_net"),
        "val_contract_eur": r.get("val_contract_eur"),
        "converged": bool(r.get("train_converged", False)),
        "bc_agreement": r.get("bc_val_agreement"),
        "steps": (r.get("train_config") or {}).get("total_steps"),
        "wall_s": r.get("wall_s"),
    } for t, v, m, i, r in rows])
    if len(out):
        # THE measure a tariff's contract is chosen on (module docstring).
        out["val_select"] = np.where(out.tariff.isin(CONTRACT_MEASURED),
                                     out.val_contract_net.astype(float),
                                     out.val_net)
    return out


def select(ids, tariffs, verbose=True) -> dict:
    """{tariff: {method: variant}}: lowest mean validation cost (`val_select`:
    AU the held-out bill net of wear, SI the same with the contract charge).

    Over the contracts trained for every learner only (the leave-one-out
    contracts exist for the clone alone, and a choice that one method had more
    candidates for would not be the same choice). `fc_h24` is a candidate: a
    learner the wider inputs do not help keeps the panel's contract.
    """
    val = validation_frame(ids, tariffs)
    picks, table = {}, []
    for tariff in tariffs:
        picks[tariff] = {}
        for method in METHODS:
            sub = val[(val.tariff == tariff) & (val.method == method)
                      & ~val.variant.str.startswith("all_no_")
                      & ~val.variant.isin(FOLLOW_UP)]
            wide = sub.pivot(index="ident", columns="variant", values="val_select")
            # Only contracts complete on every household: a mean over a subset
            # is a mean over easier or harder households, not a comparison.
            wide = wide.dropna(axis=1, how="any")
            if BASELINE not in wide.columns:
                continue
            means = wide.mean().sort_values()
            picks[tariff][method] = str(means.index[0])
            for v in means.index:
                d = wide[BASELINE] - wide[v]
                table.append({"tariff": tariff, "method": method, "variant": v,
                              "val_select": means[v],
                              "gain_vs_fc_h24": float(d.mean()),
                              "better_on": int((d > 1e-9).sum()),
                              "n": int(len(d)),
                              "picked": v == means.index[0]})
    table = pd.DataFrame(table)
    os.makedirs(OUT, exist_ok=True)
    with open(SELECTION, "w", encoding="utf-8") as fh:
        json.dump({"picked": picks, "measure": "mean validation cost over the "
                   "held-out weeks, net of wear; SI with the contract charge",
                   "households": len(ids),
                   "table": table.to_dict("records")}, fh, indent=1)
    if verbose and len(table):
        print("\nSELECTED ON VALIDATION (140 held-out days; gain = fc_h24 - "
              "variant, EUR over those days, per household)")
        for (t, m), g in table.groupby(["tariff", "method"]):
            print(f"\n  {t} {m}  -> {picks[t][m]}")
            print(g[["variant", "val_select", "gain_vs_fc_h24", "better_on", "n"]]
                  .round(3).to_string(index=False))
    return picks


# ---------------------------------------------------------------------------
# Phase 4: the test year, once, from the saved weights
# ---------------------------------------------------------------------------
def _score_worker(group, quiet):
    tariff, ident, items = group
    prep, out = None, []
    for variant, method in items:
        if paused():
            out.append((tariff, variant, method, ident, "paused"))
            continue
        try:
            base = screened(tariff, variant, method, ident)
            if base is None:
                raise RuntimeError("no current screened model")
            if prep is None:
                prep = prepare(ident, tariff)
            net, fb, spec, _ = rl.load_model(os.path.join(HERE, base["model_path"]))
            if spec != feature_specs(tariff)[variant]:
                raise RuntimeError("the saved model is not this contract's")
            t0 = time.time()
            res = dict(base)
            res.update(rb.score_test_year(prep, net, fb, spec, method, variant))
            res["score_wall_s"] = time.time() - t0
            res.update(rb._comparators(ident, tariff))
            _write(_path(OUT, tariff, variant, method, ident), res)
            out.append((tariff, variant, method, ident, None))
        except Exception as exc:
            out.append((tariff, variant, method, ident, repr(exc)))
    return out


def score(ids, tariffs, n_jobs, quiet=True):
    jobs = [j for j in jobs_for(ids, tariffs) if screened(*j) is not None]
    todo = [j for j in jobs if scored(*j) is None]
    print(f"score: {len(jobs) - len(todo)} of {len(jobs)} model(s) already "
          f"scored on the test year")
    flat = _run_groups(_score_worker, _grouped(todo), n_jobs, "score", quiet)
    return all(r[4] is None for r in flat)


# ---------------------------------------------------------------------------
# The POOLED clone under a wider contract
# ---------------------------------------------------------------------------
# A household's clone is fitted to 28 000 teacher intervals; a contract five
# times as wide gives it five times the inputs to overfit them with. The
# pooled clone (`run_rl_global`, scheme "global": one network over all ~300
# clustered households, the study's best learned controller) has three
# hundred times the rows, so it is where a wider observation has room to pay.
# Imitation only: a pooled DQN is hours per run and was the one learner
# pooling made worse.
#
# Nothing of the pooled study is rewritten. Its per-household caches supply
# the teacher walk (labels and the dynamic features, which no contract here
# changes), `rl_global.train_bc_pool` fits, and the panel's settings train --
# so a pooled clone here differs from that study's in its STATIC columns and,
# where said, its width. The `fc_h24` clone is retrained through this path as
# the control: same pool, same code, the panel's contract.
POOL = os.path.join(OUT, "_pool")
POOL_SCREEN = os.path.join(POOL, "_screen")
POOL_MODELS = os.path.join(MODELS, "_pool")
POOL_CACHE = os.path.join(MODELS, "_pool_cache")
POOL_SELECTION = os.path.join(POOL, "selection.json")
POOL_WIDTHS = (64, 256)
POOL_SEEDS = (0, 1, 2)


def pool_specs(tariff: str) -> dict:
    """The contracts a pooled clone is trained under.

    None reads the AR: it is fitted for the 30 study units only, and the other
    270 households would each need two quantile-regression fits before a
    single pooled row could be built. `groups` is every input group on the
    panel's four bins; `all_no_ar` adds the hourly lookahead to it.
    """
    peak = tariff == "SI"
    groups = tuple(g for g in rf.GROUPS if g != "peak" or peak)
    return {BASELINE: rb.variant_specs(tariff)[BASELINE],
            "groups": rl.FeatureSpec(horizon=48, peak_features=peak,
                                     groups=groups),
            "all_no_ar": feature_specs(tariff)["all_no_ar"],
            # The follow-ups (FOLLOW_UP), never candidates in `pool_select`:
            # the panel's contract and `groups` with their prices measured
            # against the trailing week.
            "rel": feature_specs(tariff)["rel"],
            "groups_rel": rl.FeatureSpec(horizon=48, peak_features=peak,
                                         groups=groups,
                                         price_reference="trailing")}


POOL_FOLLOW_UP = ("rel", "groups_rel")


def pool_tag(variant: str, width: int, seed: int = 0) -> str:
    return f"{variant}__h{width}" + (f"__s{seed}" if seed else "")


def pool_models(tariffs, seeds=(0,)) -> list:
    return [(t, v, w, s) for t in tariffs for v in pool_specs(t)
            for w in POOL_WIDTHS for s in seeds]


def pool_members(tariff: str) -> list:
    import run_rl_global as rgl
    return rgl.cached_members(tariff, sorted(int(i) for i in rgl.population().index))


def _pool_base_path(tariff, variant, ident) -> str:
    return os.path.join(POOL_CACHE, tariff, variant, f"{ident}.npy")


def _pool_base_worker(ident, tariff, variants):
    """One member's static matrix over the training span, per contract."""
    warnings.simplefilter("ignore")
    try:
        specs = pool_specs(tariff)
        prep = prepare(ident, tariff, ar=False)
        for v in variants:
            path = _pool_base_path(tariff, v, ident)
            if os.path.exists(path):
                continue
            _, static = rb._features(prep, specs[v], "train")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp.npy"
            np.save(tmp, static.astype(np.float32))
            os.replace(tmp, path)
        return ident, None
    except Exception as exc:
        return ident, repr(exc)


def pool_config(tariff, width, seed) -> tuple:
    """(TrainConfig, BCOptions) of one pooled clone: the panel's tuned clone,
    at the width asked for."""
    cfg = rb.effective_config(rb.make_config(STEPS, seed), tariff, "bc")
    cfg.hidden = int(width)
    return cfg, rb.effective_clone(tariff, "bc")


def pool_digest(tariff, variant, width, seed, members) -> str:
    cfg, clone = pool_config(tariff, width, seed)
    return rb._digest(pool_specs(tariff)[variant].config(),
                      rb._method_config(cfg, "bc", clone),
                      {"tariff": tariff, "algo": rl.BC_ALGO_VERSION,
                       "train_days": rb.TRAIN_BLOCKS, "val_days": rb.VAL_BLOCKS,
                       "teach_span": rb.TEACH_SPAN, "members": list(members)})


def _pool_paths(tariff, variant, width, seed):
    tag = pool_tag(variant, width, seed)
    return (os.path.join(POOL_MODELS, tariff, tag + ".pt"),
            os.path.join(POOL, tariff, tag, "model.json"))


def pool_model(tariff, variant, width, seed):
    """The trained pooled clone's metadata, if on disk and current."""
    meta = _read(_pool_paths(tariff, variant, width, seed)[1])
    if meta is None or meta.get("digest") != pool_digest(
            tariff, variant, width, seed, meta.get("members", [])):
        return None
    if meta["members"] != pool_members(tariff):
        return None
    return meta


def _pool_train_worker(tariff, variant, width, seed, threads, quiet):
    """Fit one pooled clone. The rows are every member's teacher walk: its
    training blocks to fit on, its validation weeks to stop on."""
    import torch
    import run_rl_global as rgl
    import rl_global as rg

    warnings.simplefilter("ignore")
    torch.set_num_threads(max(int(threads), 1))
    t0 = time.time()
    try:
        members = pool_members(tariff)
        spec = pool_specs(tariff)[variant]
        cfg, clone = pool_config(tariff, width, seed)
        train_rows = np.concatenate([np.arange(a * H, b * H)
                                     for a, b in rb.TRAIN_BLOCKS])
        # The pooled study's own streaming scaler, handed to a plain builder:
        # an untyped pool has no columns of its own, so the model saves and
        # loads as any local one does (`rl.save_model` / `rl.load_model`).
        scaler = rg.TypedFeatureBuilder(spec, n_types=0)
        scaler.fit_norm_pooled(np.load(_pool_base_path(tariff, variant, i),
                                       mmap_mode="r")[train_rows] for i in members)
        fb = rl.FeatureBuilder(spec)
        fb.norm_mean, fb.norm_std = scaler.norm_mean, scaler.norm_std
        Xtr, ytr, Xva, yva = [], [], [], []
        for i in members:
            with np.load(rgl.cache_path(tariff, i)) as z:
                dyn, labels, s0 = z["dyn"], z["labels"], int(z["teach_start"])
            base = np.load(_pool_base_path(tariff, variant, i))
            n = len(labels)
            X = np.hstack([fb.normalize(base[s0:s0 + n]), dyn]).astype(np.float32)
            mask = rgl._val_mask(n, s0)
            y = labels.astype(np.int64)
            Xtr.append(X[~mask]); ytr.append(y[~mask])
            Xva.append(X[mask]); yva.append(y[mask])
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        Xva, yva = np.concatenate(Xva), np.concatenate(yva)
        net, hist = rg.train_bc_pool(Xtr, ytr, Xva, yva, cfg, verbose=not quiet,
                                     opts=clone)
        pt, js = _pool_paths(tariff, variant, width, seed)
        rl.save_model(pt, net, fb, spec, cfg, {"bc": hist},
                      extra={"tariff": tariff, "variant": variant,
                             "pooled": len(members)})
        _write(js, {"digest": pool_digest(tariff, variant, width, seed, members),
                    "tariff": tariff, "variant": variant, "hidden": int(width),
                    "seed": int(seed), "members": members,
                    "inputs": int(Xtr.shape[1]), "n_train_rows": int(len(Xtr)),
                    "bc_val_agreement": hist.get("final_val_agreement"),
                    "converged": bool(hist.get("converged", False)),
                    "epochs": len(hist.get("val_loss", [])),
                    "train_runtime_s": hist.get("runtime_s"),
                    "wall_s": time.time() - t0, "train_config": cfg.config(),
                    "clone_options": clone.changed(),
                    "model_path": os.path.relpath(pt, HERE)})
        return tariff, variant, width, seed, time.time() - t0, None
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return tariff, variant, width, seed, time.time() - t0, repr(exc)


def pool_train(tariffs, n_jobs, seeds=(0,), quiet=True, concurrent=4):
    """Build the members' static matrices, then fit every pooled clone not on
    disk. `concurrent` fits run at once, each holding its own pool in memory
    (up to ~7 GB at the widest contract), on `n_jobs / concurrent` threads."""
    from joblib import Parallel, delayed

    todo = [m for m in pool_models(tariffs, seeds) if pool_model(*m) is None]
    print(f"pool: {len(pool_models(tariffs, seeds)) - len(todo)} of "
          f"{len(pool_models(tariffs, seeds))} pooled clone(s) trained")
    if not todo:
        return True
    for tariff in sorted({m[0] for m in todo}):
        members = pool_members(tariff)
        variants = sorted({m[1] for m in todo if m[0] == tariff})
        missing = [i for i in members if not all(
            os.path.exists(_pool_base_path(tariff, v, i)) for v in variants)]
        print(f"pool {tariff}: {len(members)} members, building the static "
              f"matrices of {len(missing)}", flush=True)
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_pool_base_worker)(i, tariff, variants) for i in missing)
        bad = [r for r in res if r[1]]
        for i, exc in bad:
            print(f"  FAILED Ausgrid {i}: {exc}")
        if bad:
            return False
        print(f"pool {tariff}: built in {(time.time() - t0)/60:.1f} min", flush=True)
    # Widest first: the long fits start while there is the most left to wait for.
    todo.sort(key=lambda m: (-m[2], m[1] == BASELINE))
    concurrent = max(1, min(int(concurrent), len(todo)))
    threads = max(1, (n_jobs if n_jobs > 0 else os.cpu_count() or 1) // concurrent)
    print(f"pool: fitting {len(todo)} clone(s), {concurrent} at a time on "
          f"{threads} thread(s) each", flush=True)
    t0, ok = time.time(), True
    stream = Parallel(n_jobs=concurrent, backend="loky",
                      return_as="generator_unordered")(
        delayed(_pool_train_worker)(*m, threads, quiet) for m in todo)
    for k, (t, v, w, sd, wall, exc) in enumerate(stream, 1):
        elapsed = time.time() - t0
        left = (len(todo) - k) * elapsed / k
        print(f"  pool: {k}/{len(todo)} {t} {pool_tag(v, w, sd)} in "
              f"{wall/60:.1f} min{'  FAILED ' + exc if exc else ''}; "
              f"~{left/60:.0f} min left, ETA "
              f"{datetime.now() + timedelta(seconds=left):%H:%M}", flush=True)
        ok = ok and exc is None
    return ok


def _pool_eval_worker(ident, tariff, items, with_test):
    """One study unit under every pooled clone: the validation measures, and
    (only once the choice is written) the test year."""
    warnings.simplefilter("ignore")
    out, prep = [], None
    for variant, width, seed in items:
        try:
            meta = pool_model(tariff, variant, width, seed)
            if meta is None:
                raise RuntimeError("no current pooled model")
            tag = pool_tag(variant, width, seed)
            vpath = _path(POOL_SCREEN, tariff, tag, "bc", ident)
            tpath = _path(POOL, tariff, tag, "bc", ident)
            val = _read(vpath)
            if val is not None and val.get("digest") != meta["digest"]:
                val = None
            done = _read(tpath)
            if val is not None and (not with_test or (
                    done is not None and done.get("digest") == meta["digest"])):
                out.append((tariff, tag, ident, None))
                continue
            if prep is None:
                prep = prepare(ident, tariff, ar=False)
            net, fb, spec, _ = rl.load_model(os.path.join(HERE, meta["model_path"]))
            if val is None:
                train = prep["train"]
                static = fb.normalize(fb.build_static(
                    train["sig"], **rb.feature_inputs(prep, spec, "train")))
                v = rl.validation_rollout(net, fb, static, train["sig"],
                                          train["settle"], train["env"],
                                          rb.VAL_BLOCKS, rb.SOC_INIT_USABLE,
                                          tariff == "SI")
                val = {"ident": str(ident), "tariff": tariff, "variant": variant,
                       "hidden": int(width), "seed": int(seed), "method": "bc",
                       "digest": meta["digest"],
                       "val_cost_closed": v["cost_eur_closed"],
                       "val_efc": v["efc"],
                       "bc_val_agreement": meta["bc_val_agreement"]}
                if tariff in CONTRACT_MEASURED:
                    val.update(contract_validation(prep, net, fb, spec))
                _write(vpath, val)
            if with_test:
                res = dict(val)
                res.update(rb.score_test_year(prep, net, fb, spec, "bc", tag))
                res.update(rb._comparators(ident, tariff))
                _write(tpath, res)
            out.append((tariff, tag, ident, None))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            out.append((tariff, pool_tag(variant, width, seed), ident, repr(exc)))
    return out


def pool_eval(ids, tariffs, n_jobs, seeds=(0,), with_test=False):
    from joblib import Parallel, delayed

    items = {}
    for t, v, w, sd in pool_models(tariffs, seeds):
        if pool_model(t, v, w, sd) is not None:
            items.setdefault(t, []).append((v, w, sd))
    tasks = [(int(i), t, items[t], with_test) for t in items for i in ids]
    print(f"pool {'score' if with_test else 'validate'}: {len(tasks)} "
          f"unit-tariff(s) x up to {max(map(len, items.values()), default=0)} "
          f"clone(s)", flush=True)
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_pool_eval_worker)(*task) for task in tasks)
    bad = [r for part in res for r in part if r[3]]
    for t, tag, i, exc in bad:
        print(f"  FAILED {t} {tag} Ausgrid {i}: {exc}")
    return not bad


def pool_frame(ids, tariffs, test: bool = False) -> pd.DataFrame:
    """One row per (tariff, pooled clone, unit): validation, and with `test`
    the scored year on the axis `test_frame` uses."""
    rows = []
    root = POOL if test else POOL_SCREEN
    for t, v, w, sd in pool_models(tariffs, POOL_SEEDS):
        meta = pool_model(t, v, w, sd)
        if meta is None:
            continue
        for i in ids:
            r = _read(_path(root, t, pool_tag(v, w, sd), "bc", i))
            if r is None or r.get("digest") != meta["digest"]:
                continue
            row = {"tariff": t, "variant": v, "hidden": w, "seed": sd,
                   "model": f"{v}__h{w}", "ident": int(i),
                   "val_net": _val_net(r, t), "val_efc": float(r["val_efc"]),
                   "val_contract_net": r.get("val_contract_net"),
                   "bc_agreement": r.get("bc_val_agreement")}
            row["val_select"] = (row["val_contract_net"]
                                 if t in CONTRACT_MEASURED else row["val_net"])
            if test:
                efc = float(r["efc"])
                row.update({"efc": efc,
                            "cost_eur_closed": float(r["cost_eur_closed"]),
                            "fixed_eur": float(r.get("fixed_eur") or 0.0),
                            "test_net": float(r["cost_eur_total"]) + float(
                                hs.cycle_wear_eur(efc, rb.BATTERY_CAP, t))})
            rows.append(row)
    return pd.DataFrame(rows)


POOL_CONTROL = f"{BASELINE}__h{POOL_WIDTHS[0]}"


def pool_select(ids, tariffs, verbose=True) -> dict:
    """{tariff: model}: the pooled clone with the lowest mean validation cost
    over the study units (seed 0), the control included."""
    val = pool_frame(ids, tariffs)
    picks, table = {}, []
    if val.empty:
        return picks
    val = val[(val.seed == 0) & ~val.variant.isin(POOL_FOLLOW_UP)]
    for tariff in tariffs:
        wide = (val[val.tariff == tariff]
                .pivot(index="ident", columns="model", values="val_select")
                .dropna(axis=1, how="any"))
        if POOL_CONTROL not in wide.columns:
            continue
        means = wide.mean().sort_values()
        picks[tariff] = str(means.index[0])
        for m in means.index:
            d = wide[POOL_CONTROL] - wide[m]
            table.append({"tariff": tariff, "model": m, "val_select": means[m],
                          "gain_vs_control": float(d.mean()),
                          "better_on": int((d > 1e-9).sum()), "n": int(len(d))})
    table = pd.DataFrame(table)
    _write(POOL_SELECTION, {"picked": picks, "table": table.to_dict("records")})
    if verbose and len(table):
        print("\nPOOLED CLONE, SELECTED ON VALIDATION (gain = control - model, "
              "over the 140 held-out days)")
        print(table.round(3).to_string(index=False))
        print("  picked:", picks)
    return picks


def pool_report(ids, tariffs):
    picks = (_read(POOL_SELECTION) or {}).get("picked", {})
    df = pool_frame(ids, tariffs, test=True)
    if df.empty:
        print("no pooled clone scored yet")
        return None
    local = test_frame(ids, tariffs)
    local = local[(local.method == "bc") & (local.variant == BASELINE)]
    refs = references(ids, tariffs)
    rows = []
    for tariff in tariffs:
        sub = df[df.tariff == tariff]
        nb = refs[(refs.tariff == tariff) & (refs.controller == "no_battery")
                  ].set_index("ident").test_net
        loc = local[local.tariff == tariff].set_index("ident").test_net
        for (model, seed), g in sub.groupby(["model", "seed"]):
            g = g.set_index("ident")
            ctrl = sub[(sub.model == POOL_CONTROL) & (sub.seed == seed)
                       ].set_index("ident").test_net
            wide = pd.DataFrame({"m": g.test_net, "c": ctrl}).dropna()
            p = (hs.paired_comparison(wide, "c", "m")["p_value"]
                 if model != POOL_CONTROL and len(wide) > 1 else np.nan)
            d = wide.c - wide.m
            rows.append({"tariff": tariff, "model": model, "seed": seed,
                         "picked": picks.get(tariff) == model,
                         "saving": float((nb - g.test_net).mean()),
                         "vs_local_clone": float((loc - g.test_net).mean()),
                         "vs_pooled_control": float(d.mean()),
                         "better_on": int((d > 1e-9).sum()), "n": int(len(d)),
                         "p": p, "efc": float(g.efc.mean()),
                         "agreement": float(g.bc_agreement.mean())})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(f"\n{'=' * 78}\nPOOLED CLONE ({len(pool_members(tariffs[0]))} "
          f"households), test year: saving against no battery and gains, "
          f"total bill + wear\n{'=' * 78}")
    print(out.round(3).to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# SI, PAID FOR THE CONTRACT: the wider inputs under the reward that can use them
# ---------------------------------------------------------------------------
# The panel pays an SI learner for the settled bill against the no-battery
# contract, so nothing it is shown about peaks can be worth anything to it: the
# line a peak moves is not in its reward. `run_rl_contract` built the reward
# that does pay for it (the standing charge the agent's own peaks re-agree,
# "contract" arm), and under it reinforcement became the best learned SI
# controller. This phase trains THAT arm -- its reward, its settings, its
# validation measure, `rc.run_one` itself -- under three wider observations,
# against that study's own `fc_h24` results, which are read and never rewritten.
CONTRACT = os.path.join(OUT, "_contract")
CONTRACT_SCREEN = os.path.join(CONTRACT, "_screen")
CONTRACT_MODELS = os.path.join(MODELS, "_contract")
CONTRACT_SELECTION = os.path.join(CONTRACT, "selection.json")
CONTRACT_TYPE, CONTRACT_ARM = "monthly", "contract"


def contract_specs() -> dict:
    """The observations the contract-paid arm is given. `rc.spec_for` adds the
    arm's own contract features to each."""
    base = dict(horizon=48, peak_features=True)
    return {"peak": rl.FeatureSpec(**base, groups=("peak",)),
            "groups": rl.FeatureSpec(**base, groups=rf.GROUPS),
            "all_coarse": rl.FeatureSpec(**base, forecast="ar", groups=rf.GROUPS),
            # The follow-ups (FOLLOW_UP): never candidates in `contract_select`.
            "rel": rl.FeatureSpec(**base, price_reference="trailing"),
            "all_coarse_rel": rl.FeatureSpec(**base, forecast="ar",
                                             groups=rf.GROUPS,
                                             price_reference="trailing")}


def _contract_file(root, variant, method, ident) -> str:
    return os.path.join(root, variant, CONTRACT_TYPE,
                        f"{CONTRACT_ARM}__{method}", f"{ident}.json")


def contract_baseline(method, ident):
    """The contract study's own `fc_h24` run of this arm."""
    import run_rl_contract as rc
    return _read(os.path.join(rc.OUT, CONTRACT_TYPE,
                              f"{CONTRACT_ARM}__{method}", f"{ident}.json"))


def contract_screened(variant, method, ident):
    import run_rl_contract as rc
    res = _read(_contract_file(CONTRACT_SCREEN, variant, method, ident))
    if res is None:
        return None
    ok = {rc.run_digest(CONTRACT_TYPE, CONTRACT_ARM, method,
                        rb.make_config(steps, 0), base=contract_specs()[variant])
          for steps in (STEPS, RETRAIN_STEPS)}
    return res if res.get("digest") in ok else None


def contract_scored(variant, method, ident):
    res = _read(_contract_file(CONTRACT, variant, method, ident))
    base = contract_screened(variant, method, ident)
    if res is None or base is None or res.get("digest") != base["digest"]:
        return None
    return res


def contract_jobs(ids) -> list:
    return [(v, m, int(i)) for i in ids for m in METHODS for v in contract_specs()]


def _contract_worker(ident, items, steps, quiet, with_test):
    import run_rl_contract as rc

    warnings.simplefilter("ignore")
    out, prep = [], None
    specs = contract_specs()
    for variant, method in items:
        if paused():
            out.append((variant, method, ident, "paused"))
            continue
        try:
            if prep is None:
                prep = enrich(rc.prepare(ident, CONTRACT_TYPE))
            if not with_test:
                res = rc.run_one(
                    prep, CONTRACT_ARM, method, rb.make_config(steps, 0),
                    verbose=not quiet,
                    out_root=os.path.join(CONTRACT_SCREEN, variant),
                    models_root=os.path.join(CONTRACT_MODELS, variant),
                    score_test=False, base=specs[variant])
                assert "cost_eur_total" not in res, "the screen touched the test year"
            else:
                base = contract_screened(variant, method, ident)
                if base is None:
                    raise RuntimeError("no current screened model")
                net, fb, _, _ = rl.load_model(os.path.join(HERE, base["model_path"]))
                res = dict(base)
                res.update(rc.score(
                    prep["sim"], rc.learned_policy(net, fb, base["window"], prep,
                                                   "sim"), rb.N_SIM * H))
                prep["sim"]["env"].clear_achieved_power()
                _write(_contract_file(CONTRACT, variant, method, ident), res)
            out.append((variant, method, ident, None))
        except Exception as exc:
            import traceback
            out.append((variant, method, ident,
                        repr(exc) + "\n" + traceback.format_exc(limit=4)))
    return out


def _contract_run(todo, n_jobs, label, steps, quiet, with_test):
    from joblib import Parallel, delayed

    groups = {}
    for v, m, i in todo:
        groups.setdefault(i, []).append((v, m))
    if not groups:
        print(f"{label}: nothing to do")
        return True
    n_runs = len(todo)
    print(f"{label}: {n_runs} run(s) over {len(groups)} household(s) on "
          f"{min(n_jobs, len(groups))} worker(s)", flush=True)
    t0, done, flat = time.time(), 0, []
    stream = Parallel(n_jobs=max(1, min(n_jobs, len(groups))), backend="loky",
                      return_as="generator_unordered")(
        delayed(_contract_worker)(i, items, steps, quiet, with_test)
        for i, items in groups.items())
    for part in stream:
        flat += part
        done += len(part)
        left = (n_runs - done) * (time.time() - t0) / max(done, 1)
        print(f"  {label}: {done}/{n_runs} runs, {(time.time() - t0)/60:.0f} min "
              f"elapsed, ~{left/60:.0f} min left, ETA "
              f"{datetime.now() + timedelta(seconds=left):%H:%M}", flush=True)
    for v, m, i, exc in flat:
        if exc not in (None, "paused"):
            print(f"  FAILED {v} {m} Ausgrid {i}: {exc}")
    return all(r[3] is None for r in flat)


def contract_frame(ids, test: bool = False) -> pd.DataFrame:
    """One row per (variant, method, household) of the contract-paid arm, the
    contract study's own `fc_h24` rows included. `val` is that study's
    held-out measure with the contract charge; `test_net` the scored year's
    total bill plus wear."""
    rows = []
    for i in ids:
        for m in METHODS:
            base = contract_baseline(m, i)
            if base is not None:
                rows.append((BASELINE, m, int(i), base))
        for v, m, i2 in contract_jobs([i]):
            r = (contract_scored if test else contract_screened)(v, m, i2)
            if r is not None:
                rows.append((v, m, i2, r))
    out = []
    for v, m, i, r in rows:
        row = {"tariff": "SI", "variant": v, "method": m, "ident": i,
               "val": float(r["val_cost"]), "val_efc": float(r["val_efc"]),
               "val_contract_eur": float(r["val_contract"]),
               "converged": bool(r.get("train_converged", False)),
               "steps": (r.get("train_config") or {}).get("total_steps")}
        if test and "cost_eur_total" in r:
            efc = float(r["efc"])
            row.update({"efc": efc, "fixed_eur": float(r["fixed_eur"]),
                        "cost_eur_closed": float(r["cost_eur_closed"]),
                        "energy_eur": float(r["energy_eur"]),
                        "test_net": float(r["cost_eur_total"]) + float(
                            hs.cycle_wear_eur(efc, rb.BATTERY_CAP, "SI"))})
        out.append(row)
    return pd.DataFrame(out)


def contract_select(ids, verbose=True) -> dict:
    """{method: variant}: lowest mean held-out cost, `fc_h24` a candidate."""
    val = contract_frame(ids)
    picks, table = {}, []
    for m in METHODS:
        wide = (val[(val.method == m) & ~val.variant.isin(FOLLOW_UP)]
                .pivot(index="ident", columns="variant", values="val")
                .dropna(axis=1, how="any"))
        if BASELINE not in wide.columns:
            continue
        means = wide.mean().sort_values()
        picks[m] = str(means.index[0])
        for v in means.index:
            d = wide[BASELINE] - wide[v]
            table.append({"method": m, "variant": v, "val": means[v],
                          "gain_vs_fc_h24": float(d.mean()),
                          "better_on": int((d > 1e-9).sum()), "n": int(len(d))})
    table = pd.DataFrame(table)
    _write(CONTRACT_SELECTION, {"picked": picks, "table": table.to_dict("records")})
    if verbose and len(table):
        print("\nSI, CONTRACT-PAID ARM, SELECTED ON VALIDATION (gain = fc_h24 - "
              "variant over the 140 held-out days)")
        print(table.round(3).to_string(index=False))
        print("  picked:", picks)
    return picks


def contract_report(ids):
    picks = (_read(CONTRACT_SELECTION) or {}).get("picked", {})
    df = contract_frame(ids, test=True)
    if df.empty or "test_net" not in df:
        print("contract arm: nothing scored yet")
        return None
    rows = []
    for m in METHODS:
        sub = df[df.method == m]
        for v in contract_specs():
            r = paired(sub, v, BASELINE)
            if r:
                rows.append({"method": m, "variant": v,
                             "selected": picks.get(m) == v, **r,
                             "efc": float(sub[sub.variant == v].efc.mean())})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        sel = out.selected
        out.loc[sel, "p_holm"] = rb.holm(list(out.loc[sel, "p"]))
    nb = references(ids, ["SI"])
    nb = nb[nb.controller == "no_battery"].set_index("ident").test_net
    lvl = df.dropna(subset=["test_net"]).assign(
        saving=lambda d: d.ident.map(nb) - d.test_net)
    pd.set_option("display.width", 220)
    print(f"\n{'=' * 78}\nSI, learners PAID FOR THE CONTRACT: mean saving against "
          f"no battery, total bill + wear, EUR/a\n{'=' * 78}")
    print(lvl.pivot_table(index="variant", columns="method", values="saving")
          .reindex(columns=list(METHODS)).round(1).to_string())
    print("\ngain over that arm's own fc_h24 on the test year (positive = cheaper)")
    print(out.round(4).to_string(index=False))
    return out


def contract_phase(ids, n_jobs, quiet=True):
    """Train (validation only), re-train the unsettled, choose, score."""
    todo = [j for j in contract_jobs(ids) if contract_screened(*j) is None]
    print(f"contract: {len(contract_jobs(ids)) - len(todo)} of "
          f"{len(contract_jobs(ids))} run(s) cached")
    first = _read(CONTRACT_SELECTION)
    if not _contract_run(todo, n_jobs, "contract", STEPS, quiet, False):
        return False
    late = [j for j in contract_jobs(ids) if j[1] != "bc"
            and (lambda r: r is not None and not r.get("train_converged")
                 and r["train_config"]["total_steps"] < RETRAIN_STEPS)(
                     contract_screened(*j))]
    print(f"contract: {len(late)} unsettled reinforcement run(s) at "
          f"{RETRAIN_STEPS:,} steps")
    if not _contract_run(late, n_jobs, "contract retrain", RETRAIN_STEPS, quiet,
                         False):
        return False
    # The choice is made once, among the contracts that existed when it was
    # made: a later pass that trains a follow-up must not re-open it.
    if first is None:
        contract_select(ids)
    todo = [j for j in contract_jobs(ids) if contract_scored(*j) is None]
    if not _contract_run(todo, n_jobs, "contract score", STEPS, quiet, True):
        return False
    contract_report(ids)
    return True


# ---------------------------------------------------------------------------
# Other seeds: is a gain the contract's, or the draw's?
# ---------------------------------------------------------------------------
# A local learner re-trained under another seed moves by up to ~6 AUD/a on its
# own (`run_rl_global.local_seed_runs`), which is the size of the effects
# here. The contracts that carry every input are therefore trained twice more
# and compared with the panel's `fc_h24` under the SAME seed -- which exists:
# the hyperparameter study confirmed the panel's settings on seeds 0-2
# (`rl_hpo`, `confirm/<winner>/s<seed>`), all at the screen's budget with no
# straggler re-trained, so these runs are left at that budget too.
#
# No choice is made on these runs -- both contracts were fixed before any of
# them was trained -- so they are scored on the test year as they finish.
# `all` is the contract named before any test number existed; `all_coarse_rel`
# is the follow-up (FOLLOW_UP) that turned out best, and the one most in need
# of a second and third draw for that reason.
SEED_VARIANTS = ("all", "all_coarse_rel")
SEED_ROOT = os.path.join(OUT, "_seeds")
EXTRA_SEEDS = (1, 2)


def _seed_roots(seed: int) -> tuple:
    return (os.path.join(SEED_ROOT, f"s{seed}"),
            os.path.join(MODELS, "_seeds", f"s{seed}"))


def seed_result(tariff, variant, method, ident, seed):
    res = _read(_path(_seed_roots(seed)[0], tariff, variant, method, ident))
    if res is None or "cost_eur_total" not in res or res.get("digest") != rb.run_digest(
            rb.make_config(STEPS, seed), feature_specs(tariff)[variant], tariff,
            method):
        return None
    return res


def seed_baseline(tariff, method, ident, seed):
    """The panel's `fc_h24` under `seed`, from the hyperparameter study's
    confirmation of the promoted settings -- refused unless its digest is the
    one the panel's settings give at that seed today."""
    study = os.path.join(HERE, "results_local", "rl_hpo", f"{tariff}_{method}")
    winner = _read(os.path.join(study, "winner.json"))
    if winner is None:
        return None
    res = _read(os.path.join(study, "confirm", winner["tag"], f"s{seed}", tariff,
                             f"{BASELINE}__{method}", f"{ident}.json"))
    if res is None or "cost_eur_total" not in res or res.get("digest") != rb.run_digest(
            rb.make_config(STEPS, seed), rb.variant_specs(tariff)[BASELINE],
            tariff, method):
        return None
    return res


def _seed_worker(group, seed, quiet):
    tariff, ident, items = group
    specs = feature_specs(tariff)
    root, models = _seed_roots(seed)
    prep, out = None, []
    for variant, method in items:
        if paused():
            out.append((tariff, variant, method, ident, "paused"))
            continue
        try:
            if prep is None:
                prep = prepare(ident, tariff)
            rb.run_one(prep, variant, method, rb.make_config(STEPS, seed),
                       verbose=not quiet, out_root=root, models_root=models,
                       specs=specs)
            out.append((tariff, variant, method, ident, None))
        except Exception as exc:
            out.append((tariff, variant, method, ident, repr(exc)))
    return out


def seed_runs(ids, tariffs, n_jobs, seeds=EXTRA_SEEDS, quiet=True):
    ok = True
    for seed in seeds:
        jobs = jobs_for(ids, tariffs, SEED_VARIANTS)
        todo = [j for j in jobs if seed_result(*j, seed) is None]
        print(f"seed {seed}: {len(jobs) - len(todo)} of {len(jobs)} run(s) cached")
        flat = _run_groups(_seed_worker, _grouped(todo), n_jobs, f"seed {seed}",
                           seed, quiet)
        ok = ok and all(r[4] is None for r in flat)
    return ok


def seed_frame(ids, tariffs, seeds=(0,) + EXTRA_SEEDS) -> pd.DataFrame:
    """Test-year rows of `fc_h24` and the all-input contracts, per seed. Seed 0
    is the main study's (stragglers re-trained on both sides); the others are
    at the screen's budget on both sides."""
    rows = []
    if 0 in seeds:
        first = test_frame(ids, tariffs)
        first = first[first.variant.isin((BASELINE,) + SEED_VARIANTS)]
        rows += first.assign(seed=0)[["tariff", "variant", "method", "ident",
                                      "seed", "efc", "test_net"]].to_dict("records")
    for seed in [s for s in seeds if s]:
        for tariff in tariffs:
            for ident in ids:
                for method in METHODS:
                    found = {BASELINE: seed_baseline(tariff, method, ident, seed)}
                    for v in SEED_VARIANTS:
                        found[v] = seed_result(tariff, v, method, ident, seed)
                    for v, r in found.items():
                        if r is None:
                            continue
                        efc = float(r["efc"])
                        rows.append({
                            "tariff": tariff, "variant": v, "method": method,
                            "ident": int(ident), "seed": seed, "efc": efc,
                            "test_net": float(r["cost_eur_total"]) + float(
                                hs.cycle_wear_eur(efc, rb.BATTERY_CAP, tariff))})
    return pd.DataFrame(rows)


def seed_table(ids, tariffs) -> pd.DataFrame:
    """Per (tariff, method, contract): the gain over `fc_h24` under each seed,
    and the test on the households' SEED-AVERAGED gains (Holm over the six
    cells of a contract)."""
    df = seed_frame(ids, tariffs)
    rows = []
    for v in SEED_VARIANTS:
        fam = []
        for tariff in tariffs:
            for method in METHODS:
                sub = df[(df.tariff == tariff) & (df.method == method)]
                wide = sub.pivot_table(index=["ident", "seed"], columns="variant",
                                       values="test_net")
                if v not in wide or BASELINE not in wide:
                    continue
                gain = (wide[BASELINE] - wide[v]).dropna().unstack("seed")
                both = gain.dropna()
                if both.empty:
                    continue
                mean = both.mean(axis=1)
                t = pd.DataFrame({"a": mean, "b": 0.0})
                fam.append({
                    "tariff": tariff, "method": method, "variant": v,
                    "n": int(len(both)), "seeds": int(both.shape[1]),
                    **{f"s{int(c)}": float(both[c].mean()) for c in both.columns},
                    "mean": float(mean.mean()), "median": float(mean.median()),
                    "better_on": int((mean > 1e-9).sum()),
                    "p": float(hs.paired_comparison(t, "a", "b")["p_value"])})
        for row, p in zip(fam, rb.holm([f["p"] for f in fam])):
            row["p_holm"] = p
        rows += fam
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A held-out test of the explanation: the validation weeks at the scored
# year's price level
# ---------------------------------------------------------------------------
# The follow-up contracts were trained after the test year had been read, so
# their test-year gain cannot be its own evidence. This is evidence the test
# year's OUTCOMES never touch: the trained models, unchanged, rolled out on
# the held-out weeks of the training years twice -- once as they are, once
# with the market price raised by the difference between the scored year's
# mean and the training years' (a fact about the price list, known the day it
# is published). Everything else is the validation rollout the panel uses. If
# reading price LEVELS is what breaks in the scored year, the level-reading
# contracts should lose saving under the shift and the trailing-week ones
# should not.
SHIFT = os.path.join(OUT, "shift_validation.csv")
SHIFT_VARIANTS = (BASELINE, "rank", "all_coarse", "rel", "all_coarse_rel")


def _shift_worker(ident, tariff):
    warnings.simplefilter("ignore")
    rows = []
    try:
        prep = prepare(ident, tariff)
        shifted = rb.shifted_bundle(prep)
        delta = shifted["delta"]
        bundles = {"none": prep["train"], "shifted": shifted["bundle"]}
        # The idle pack on each price level: what a saving is measured from.
        idle = {k: float(sum(rl.no_battery_cost_trace(
            b["sig"], b["settle"], b["env"])[a * H:c * H].sum()
            for a, c in rb.VAL_BLOCKS)) for k, b in bundles.items()}
        for variant in SHIFT_VARIANTS:
            for method in METHODS:
                res = (_read(_path(rb.OUT, tariff, BASELINE, method, ident))
                       if variant == BASELINE else
                       screened(tariff, variant, method, ident))
                if res is None:
                    continue
                net, fb, spec, _ = rl.load_model(os.path.join(HERE, res["model_path"]))
                for shift, b in bundles.items():
                    sig = b["sig"]
                    static = fb.normalize(fb.build_static(
                        sig, **rb.feature_inputs(prep, spec, "train")))
                    v = rl.validation_rollout(net, fb, static, sig, b["settle"],
                                              b["env"], rb.VAL_BLOCKS,
                                              rb.SOC_INIT_USABLE, tariff == "SI")
                    # Against the idle pack on the SAME prices: a saving, so
                    # the two price levels can be read on one axis.
                    rows.append({"tariff": tariff, "ident": int(ident),
                                 "variant": variant, "method": method,
                                 "shift": shift, "delta_smp": delta,
                                 "val_cost_closed": v["cost_eur_closed"],
                                 "val_efc": v["efc"],
                                 "saving": idle[shift] - v["cost_eur_closed"]})
        return rows, None
    except Exception as exc:
        import traceback
        return rows, repr(exc) + "\n" + traceback.format_exc(limit=4)


def shift_validation(ids, tariffs, n_jobs=1, refresh=False) -> pd.DataFrame:
    """Per (tariff, household, contract, learner, shift): the saving against
    the idle pack over the held-out weeks. Cached as a CSV."""
    if os.path.exists(SHIFT) and not refresh:
        return pd.read_csv(SHIFT)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_shift_worker)(int(i), t) for t in tariffs for i in ids)
    for _, exc in res:
        if exc:
            print("  FAILED", exc)
    out = pd.DataFrame([r for part, _ in res for r in part])
    out.to_csv(SHIFT, index=False)
    return out


# ---------------------------------------------------------------------------
# Promotion: the study's best contract, into the panel under its panel name
# ---------------------------------------------------------------------------
# `rb.variant_specs` names the wide contract `rb.WIDE_VARIANT`; its spec is
# this study's `all_coarse_rel`, so a run of either name carries one digest and
# the models this study trained ARE the panel's. They are copied rather than
# trained a second time -- the same rule the first promotion followed when the
# panel's `fc_h24` reproduced the search's confirmation runs bit for bit.
PROMOTED = "all_coarse_rel"


def promote(ids, tariffs=("AU", "SI"), verbose=True) -> int:
    """Copy the scored seed-0 runs of `PROMOTED` into `rl_screen/` and their
    networks into `rl_models/`, as `rb.WIDE_VARIANT`. A run is copied only if
    its digest is what the panel would compute for it today (at the run's own
    budget), so settings that have moved since it was trained are never
    carried in under the new name. Returns how many were copied."""
    import shutil

    n = 0
    for tariff in tariffs:
        spec = rb.variant_specs(tariff)[rb.WIDE_VARIANT]
        for method in METHODS:
            for ident in ids:
                res = scored(tariff, PROMOTED, method, ident)
                if res is None:
                    continue
                # Under the settings the PANEL trains this cell with: where the
                # wide contract has winners of its own (`rb.VARIANT_TUNED`),
                # this study's runs are not the panel's and are left out.
                want = rb.run_digest(
                    rb.make_config(res["train_config"]["total_steps"], 0),
                    spec, tariff, method,
                    rb.variant_overrides(rb.WIDE_VARIANT, tariff, method))
                if res["digest"] != want:
                    continue
                key = f"{rb.WIDE_VARIANT}__{method}"
                model = os.path.join(rb.MODELS, tariff, key, f"{ident}.pt")
                os.makedirs(os.path.dirname(model), exist_ok=True)
                src = os.path.join(HERE, res["model_path"])
                for suffix in ("", ".history.json"):
                    shutil.copyfile(src + suffix, model + suffix)
                out = dict(res, variant=rb.WIDE_VARIANT,
                           model_path=os.path.relpath(model, HERE))
                _write(_path(rb.OUT, tariff, rb.WIDE_VARIANT, method, ident), out)
                n += 1
    if verbose:
        print(f"promoted {n} run(s) of {PROMOTED} into the panel as "
              f"{rb.WIDE_VARIANT}")
    return n


# ---------------------------------------------------------------------------
# Results as frames (the notebook's half reads these and draws)
# ---------------------------------------------------------------------------
# The study's own controllers a learned one is read beside: name -> (arm
# suffix, checkpoint controller). "mpc" is the MPC under the best deployable
# forecast, the arm whose forecaster the `ar` contract re-issues.
REFERENCES = {
    "no_battery": ("", "no_battery"),
    "self_consumption": ("", "self_consumption"),
    "price_rank_daily": ("", "price_rank_daily"),
    "self_consumption_peak_shaving": ("", "self_consumption_peak_shaving"),
    "mpc_hbd_median14": ("_hbd_median14", "prophet"),
    "milp_full": ("", "milp_full"),
}


def references(ids, tariffs) -> pd.DataFrame:
    """Total bill plus lifetime wear of the study's own controllers, same
    households, read off the arm checkpoints (never recomputed)."""
    rows = []
    for tariff in tariffs:
        for ident in ids:
            for name, (suffix, ctrl) in REFERENCES.items():
                arm = hs.REFERENCE_ARM[tariff] + suffix
                blob = _read(os.path.join(HERE, "results_local", arm,
                                          f"Ausgrid {ident}", "checkpoint.json"))
                m = (blob or {}).get("metrics", {})
                if f"cost_{ctrl}" not in m:
                    continue
                efc = float(m.get(f"efc_{ctrl}") or 0.0)
                rows.append({
                    "tariff": tariff, "ident": int(ident), "controller": name,
                    "efc": efc,
                    "test_net": float(m[f"cost_{ctrl}"])
                    + float(m.get(f"fixed_{ctrl}") or 0.0)
                    + float(hs.cycle_wear_eur(efc, rb.BATTERY_CAP, tariff))})
    return pd.DataFrame(rows)


def test_frame(ids, tariffs) -> pd.DataFrame:
    """One row per scored run: `test_net` is the test year's total bill plus
    the lifetime wear `summarize` charges -- what the household pays, the axis
    `rl_hpo` confirms on. The panel's `fc_h24` rows are included."""
    rows = []
    for tariff in tariffs:
        for ident in ids:
            for method in METHODS:
                base = _read(_path(rb.OUT, tariff, BASELINE, method, ident))
                if base is not None and "cost_eur_total" in base:
                    rows.append((tariff, BASELINE, method, int(ident), base))
        for job in jobs_for(ids, [tariff]):
            res = scored(*job)
            if res is not None:
                rows.append((*job, res))
    out = []
    for t, v, m, i, r in rows:
        efc = float(r["efc"])
        out.append({
            "tariff": t, "variant": v, "method": m, "ident": i, "efc": efc,
            "cost_eur_closed": float(r["cost_eur_closed"]),
            "fixed_eur": float(r.get("fixed_eur") or 0.0),
            "test_net": float(r["cost_eur_total"])
            + float(hs.cycle_wear_eur(efc, rb.BATTERY_CAP, t)),
            "val_net": _val_net(r, t),
            "converged": bool(r.get("train_converged", False)),
            "bc_agreement": r.get("bc_val_agreement"),
        })
    return pd.DataFrame(out)


def paired(df, variant, against, value="test_net") -> dict:
    """`against - variant` per household (positive = the variant is cheaper),
    with the study's own paired test (`hs.paired_comparison`)."""
    wide = df.pivot(index="ident", columns="variant", values=value)
    if variant not in wide.columns or against not in wide.columns:
        return {}
    wide = wide[[variant, against]].dropna()
    if len(wide) < 2:
        return {}
    r = hs.paired_comparison(wide, against, variant)
    d = wide[against] - wide[variant]
    return {"n": int(len(d)), "mean": float(d.mean()),
            "median": float(d.median()), "better_on": int((d > 1e-9).sum()),
            "p": float(r["p_value"])}


def headline(ids, tariffs, picks=None) -> pd.DataFrame:
    """The claim: per (tariff, method), the validation-selected contract and
    `all` against `fc_h24` on the test year. Holm over the selected rows."""
    picks = picks or (_read(SELECTION) or {}).get("picked", {})
    df = test_frame(ids, tariffs)
    rows = []
    for tariff in tariffs:
        for method in METHODS:
            sub = df[(df.tariff == tariff) & (df.method == method)]
            chosen = picks.get(tariff, {}).get(method)
            for role, v in (("selected", chosen), ("all", "all")):
                if v is None:
                    continue
                r = ({"n": int(sub.ident.nunique()), "mean": 0.0, "median": 0.0,
                      "better_on": 0, "p": float("nan")} if v == BASELINE
                     else paired(sub, v, BASELINE))
                if r:
                    level = sub[sub.variant == v]
                    rows.append({"tariff": tariff, "method": method,
                                 "role": role, "variant": v, **r,
                                 "test_net": float(level.test_net.mean()),
                                 "efc": float(level.efc.mean())})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        sel = out.role == "selected"
        out.loc[sel, "p_holm"] = rb.holm(list(out.loc[sel, "p"]))
    return out


def information_value(ids, tariffs) -> pd.DataFrame:
    """Every addition priced twice on the test year: ALONE (variant - fc_h24)
    and, for the clone, as what `all` loses without it. Holm within each
    (tariff, method, question) family."""
    df = test_frame(ids, tariffs)
    rows = []
    for tariff in tariffs:
        for method in METHODS:
            sub = df[(df.tariff == tariff) & (df.method == method)]
            fam = []
            for v in SINGLES + ["all", "all_coarse"]:
                r = paired(sub, v, BASELINE)
                if r:
                    fam.append({"tariff": tariff, "method": method,
                                "question": "alone", "input": v, **r})
            for row, p in zip(fam, rb.holm([f["p"] for f in fam])):
                row["p_holm"] = p
            rows += fam
            fam = []
            for v, without in LEAVE_OUT.items():
                r = paired(sub, "all", without)
                if r:
                    fam.append({"tariff": tariff, "method": method,
                                "question": "given the rest", "input": v, **r})
            for row, p in zip(fam, rb.holm([f["p"] for f in fam])):
                row["p_holm"] = p
            rows += fam
    return pd.DataFrame(rows)


def report(ids, tariffs):
    picks = (_read(SELECTION) or {}).get("picked")
    if not picks:
        picks = select(ids, tariffs, verbose=False)
    df = test_frame(ids, tariffs)
    if df.empty:
        print("nothing scored yet")
        return None
    refs = references(ids, tariffs)
    pd.set_option("display.width", 200)
    for tariff in tariffs:
        sub = df[df.tariff == tariff]
        nb = refs[(refs.tariff == tariff) & (refs.controller == "no_battery")
                  ].set_index("ident").test_net
        print(f"\n{'=' * 78}\n{tariff}: mean saving against no battery, total "
              f"bill + wear, {hs.currency(tariff)}/a "
              f"(n={sub.ident.nunique()})\n{'=' * 78}")
        lvl = sub.assign(saving=sub.ident.map(nb) - sub.test_net)
        print(lvl.pivot_table(index="variant", columns="method",
                              values="saving", aggfunc="mean")
              .reindex(columns=list(METHODS)).round(1).to_string())
        r = refs[refs.tariff == tariff]
        r = r.assign(saving=r.ident.map(nb) - r.test_net)
        print("\n  the study's own controllers, same households:")
        print(r.groupby("controller").saving.mean().round(1).to_string())
    print(f"\n{'=' * 78}\nHEADLINE: gain over fc_h24 on the test year "
          f"(positive = cheaper), paired by household\n{'=' * 78}")
    print(headline(ids, tariffs, picks).round(4).to_string(index=False))
    print(f"\n{'=' * 78}\nEACH INPUT: alone, and what `all` loses without it\n"
          f"{'=' * 78}")
    print(information_value(ids, tariffs).round(4).to_string(index=False))
    return df


def status(ids, tariffs):
    jobs = jobs_for(ids, tariffs)
    done = [(j, screened(*j)) for j in jobs]
    have = [(j, r) for j, r in done if r is not None]
    n_scored = sum(scored(*j) is not None for j, _ in have)
    print(f"screened {len(have)}/{len(jobs)}   scored {n_scored}/{len(jobs)}   "
          f"unsettled {len(unsettled(ids, tariffs))}   "
          f"{'PAUSED' if paused() else 'not paused'}")
    walls = {}
    for (t, v, m, i), r in have:
        walls.setdefault(m, []).append(float(r.get("wall_s") or 0.0))
    left = 0.0
    for m in METHODS:
        todo = sum(1 for j, r in done if r is None and j[2] == m)
        mean = float(np.mean(walls[m])) if walls.get(m) else {"bc": 10.0}.get(m, 130.0)
        left += todo * mean
        print(f"  {m:7s} {len(walls.get(m, [])):4d} done, {todo:4d} to go, "
              f"{mean:6.0f} s/run")
    print(f"  ~{left/3600:.1f} worker-hours of training left "
          f"(divide by --jobs; finish ~"
          f"{datetime.now() + timedelta(seconds=left/12):%H:%M} on 12 workers)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["screen", "retrain", "select", "score",
                                      "report", "status", "all", "pause",
                                      "resume", "pool", "contract", "seeds", "shift",
                                      "promote"])
    ap.add_argument("--ids", nargs="*", type=int, default=None,
                    help="default: the study's 30 units")
    ap.add_argument("--tariffs", nargs="*", default=["AU", "SI"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0],
                    help="pool: the seeds to fit the pooled clones under")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    ids = args.ids or [int(i) for i in hs.dataset_ids()]
    quiet = not args.verbose
    os.makedirs(OUT, exist_ok=True)

    if args.phase == "pause":
        open(PAUSE, "w").close()
        print("pausing: every worker stops after the run it is on")
        return
    if args.phase == "resume" or (args.phase == "all" and paused()):
        if paused():
            os.remove(PAUSE)
        if args.phase == "resume":
            print("pause lifted; rerun the phase")
            return
    if args.phase == "status":
        return status(ids, args.tariffs)
    if args.phase == "promote":
        return promote(ids, args.tariffs)
    if args.phase == "shift":
        df = shift_validation(ids, args.tariffs, args.jobs, refresh=True)
        print(df.pivot_table(index=["tariff", "variant"], columns=["method", "shift"],
                             values="saving").round(2).to_string())
        return
    if args.phase == "seeds":
        seeds = tuple(s for s in args.seeds if s) or EXTRA_SEEDS
        ok = seed_runs(ids, args.tariffs, args.jobs, seeds, quiet)
        pd.set_option("display.width", 220)
        print(seed_table(ids, args.tariffs).round(3).to_string(index=False))
        return print("SEEDS DONE" if ok else "seeds incomplete -- rerun to continue")
    if args.phase == "contract":
        ok = contract_phase(ids, args.jobs, quiet)
        return print("CONTRACT DONE" if ok else
                     "contract incomplete -- rerun to continue")
    if args.phase == "pool":
        # Train, validate, choose, and only then read the test year.
        if not pool_train(args.tariffs, args.jobs, tuple(args.seeds), quiet):
            return print("pool incomplete -- rerun to continue")
        pool_eval(ids, args.tariffs, args.jobs, tuple(args.seeds))
        pool_select(ids, args.tariffs)
        pool_eval(ids, args.tariffs, args.jobs, tuple(args.seeds), with_test=True)
        pool_report(ids, args.tariffs)
        return print("POOL DONE")
    if args.phase in ("screen", "all"):
        ok = screen(ids, args.tariffs, args.jobs, quiet, args.variants,
                    args.methods)
        if args.phase == "all" and not ok:
            return print("screen incomplete -- rerun to continue")
    if args.phase in ("retrain", "all"):
        ok = retrain(ids, args.tariffs, args.jobs, quiet)
        if args.phase == "all" and not ok:
            return print("retrain incomplete -- rerun to continue")
    if args.phase in ("select", "all"):
        select(ids, args.tariffs)
    if args.phase in ("score", "all"):
        ok = score(ids, args.tariffs, args.jobs, quiet)
        if args.phase == "all" and not ok:
            return print("score incomplete -- rerun to continue")
    if args.phase in ("report", "all"):
        report(ids, args.tariffs)
    if args.phase == "all":
        print("ALL DONE")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
