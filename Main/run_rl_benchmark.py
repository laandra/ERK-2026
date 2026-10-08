"""Screen the learned controllers (DQN / behaviour cloning) against the study.

    python3 run_rl_benchmark.py                        # default screen
    python3 run_rl_benchmark.py --ids 138 127 --tariffs SI --variants fc_h24
    python3 run_rl_benchmark.py --methods dqn bc bc_dqn --steps 250000

Resumable exactly like `run_hbd_benchmark.py`: one JSON per (tariff, variant,
method, household) under `results_local/rl_screen/`, keyed by a config digest,
so rerunning until it prints ALL DONE recomputes nothing and a run under
superseded settings is recomputed rather than resumed into. Models land under
`rl_models/` beside their training history.

The grid answers the questions the study asks of every other controller:

    variants    what the agent may SEE -- no forecast at all, the causal
                median14 forecast on either channel, perfect roof or perfect
                everything (diagnostics), the lookahead horizon, and the
                tariff-specific ablations: SI without its contract features
                (is the capacity charge learnable blind?) and AU without its
                price features (is EA025 anything but its price signal?).
    methods     dqn (reinforcement), bc (imitation of the whole-period MILP),
                bc_dqn (imitation warm start, reinforcement fine-tune).

Three disjoint parts: TRAIN (both training years minus every 5th week),
VALIDATION (those held-out weeks: early stopping, and every hyperparameter
choice via `--tune`), TEST (the scored year, untouched until the final
`run_policy` evaluation, which prices the agent through the arm's own
settlement, endogenous contract included).

    python3 run_rl_benchmark.py --tune --ids <30 units> --jobs 12 --quiet
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402
import Battery_Economics as be                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "Input data", "Ausgrid")
OUT = os.path.join(HERE, "results_local", "rl_screen")
MODELS = os.path.join(HERE, "rl_models")

# The study's battery and split, verbatim from `run_pipeline_for_file`'s
# defaults. Restated here ONCE; everything below reads these.
BATTERY_CAP, SOC_MIN, SOC_MAX = 10.0, 0.10, 0.80
P_MAX, EFF, DELTA_T, H = 1.5, 0.95, 0.5, 48
N_TRAIN, N_SIM = 730, 365
SOC_INIT_ABS = 5.0                       # absolute kWh; usable-window = 4.0
SOC_INIT_USABLE = SOC_INIT_ABS - BATTERY_CAP * SOC_MIN

# TRAIN / VALIDATION / TEST. Test is the scored year (days 730-1095) and is
# read exactly once per run, by `run_policy` at the end of `run_one`. Train
# and validation share the two years before it: every 5th week is held out
# for validation, the rest is trained on.
#
# Held-out WEEKS rather than a contiguous tail, and both years rather than
# the last one. The first split trained on year 2 alone and validated on its
# last 60 days -- so the agents never trained on May or June (AU's high
# season) and every validation and tuning verdict was a verdict on early
# winter. A 5-week cycle over 104 weeks lands validation in every month,
# twice over, and keeps ~80 % of both years for training. Whole weeks, not
# random days: adjacent intervals are nearly identical, and a 7-day episode
# must fit inside one training block. The pattern is the same for every
# household, so comparisons stay paired.
#
# Week 4 first, not week 0: the median14 forecast has no history on day 0.
VAL_EVERY_WEEKS, VAL_FIRST_WEEK = 5, 4
VAL_BLOCKS = [(7 * w, 7 * w + 7) for w in range(N_TRAIN // 7)
              if w >= VAL_FIRST_WEEK and (w - VAL_FIRST_WEEK) % VAL_EVERY_WEEKS == 0]


def _complement(blocks, n_days):
    out, lo = [], 0
    for a, b in blocks:
        if a > lo:
            out.append((lo, a))
        lo = b
    if lo < n_days:
        out.append((lo, n_days))
    return out


TRAIN_BLOCKS = _complement(VAL_BLOCKS, N_TRAIN)   # RL episodes / BC fit
# The teacher MILP demonstrates over BOTH training years. It is walked across
# the validation weeks too (a perfect-foresight plan is one trajectory), but
# the clone is only fitted on the training blocks.
TEACH_SPAN = (0, N_TRAIN)

DEFAULT_IDS = [138, 127, 65, 148, 223]         # first five study units


def variant_specs(tariff: str) -> dict:
    """The observation ablations one tariff runs. Keys are result columns."""
    peak = tariff == "SI"
    v = {
        # No load/PV forecast at all: calendar, meter, prices. What the tariff
        # signal alone is worth to a learner.
        "nofc": rl.FeatureSpec(horizon=48, load_channel="none",
                               pv_channel="none", peak_features=peak),
        # The causal forecast on both channels, three horizons.
        "fc_h6": rl.FeatureSpec(horizon=12, peak_features=peak),
        "fc_h12": rl.FeatureSpec(horizon=24, peak_features=peak),
        "fc_h24": rl.FeatureSpec(horizon=48, peak_features=peak),
        # Perfect roof on the causal load forecast: the PV channel's value.
        "loadfc_pvtruth_h24": rl.FeatureSpec(horizon=48, pv_channel="truth",
                                             peak_features=peak),
        # Perfect everything: the learner's own foresight ceiling. Diagnostic.
        "truth_h24": rl.FeatureSpec(horizon=48, load_channel="truth",
                                    pv_channel="truth", peak_features=peak),
    }
    if tariff == "SI":
        # Blind to the contract: no block, agreed power, headroom or running
        # peak. If SI's money is the capacity charge, this one cannot find it.
        v["fc_h24_nopeak"] = rl.FeatureSpec(horizon=48, peak_features=False)
    else:
        # Blind to the price: no rates, no price lookahead. If AU's money is
        # the time-of-use shape, this one cannot find it.
        v["fc_h24_noprice"] = rl.FeatureSpec(horizon=48, price_features=False)
    return v


DEFAULT_VARIANTS = ["nofc", "fc_h6", "fc_h12", "fc_h24",
                    "loadfc_pvtruth_h24", "truth_h24",
                    "fc_h24_nopeak", "fc_h24_noprice"]
DEFAULT_METHODS = ["dqn", "bc"]


# ---------------------------------------------------------------------------
# Per-household preparation (shared across every variant and method)
# ---------------------------------------------------------------------------
def _calendar(tariff: str) -> None:
    """The data's own calendar, exactly as `run_pipeline_for_file` sets it."""
    hs._si_cas.nastavi_koledar(drzava="AU", podrocje="NSW",
                               visja_sezona_meseci={5, 6, 7, 8},
                               casovni_pas="naive")
    hs.TariffCalculator.HOLIDAY_COUNTRY = "AU"
    hs.TariffCalculator.HOLIDAY_SUBDIV = "NSW"
    hs.TariffCalculator.LOCAL_TZ = None


def prepare_household(ident, tariff: str) -> dict:
    """Everything one household needs, built once: train and sim bundles.

    The train bundle exists for learning only; the sim bundle is the arm the
    study already runs (same env factory, same rates, same settlement), so the
    evaluation numbers drop into the study's frame without translation.
    """
    _calendar(tariff)
    path = os.path.join(DATA, f"Ausgrid {ident}.csv")
    frames = hs.load_study_frames(path, H=H, delta_t=DELTA_T, n_train=N_TRAIN,
                                  n_sim=N_SIM)

    def _bundle(df_kwh, df_kw, n_steps):
        env = hs.align_envelope(
            hs.build_study_env(df_kwh, battery_cap=BATTERY_CAP,
                               soc_min_pct=SOC_MIN, soc_max_pct=SOC_MAX,
                               p_max=P_MAX, eff=EFF, delta_t=DELTA_T, H=H),
            P_MAX, EFF, DELTA_T)
        rates = hs.build_rate_vectors(tariff, env, df_kw.index,
                                      df_kw["SMP"].values,
                                      int(round(DELTA_T * 60)))
        settle = hs.build_settlement(tariff, env, rates, DELTA_T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sig = rbc.build_signals(env, n_steps=n_steps, rates=rates)
        return {"env": env, "rates": rates, "settle": settle, "sig": sig}

    df_all_kwh = frames["df_all_kwh"]
    df_train_kwh = df_all_kwh.iloc[: N_TRAIN * H]
    train = _bundle(df_train_kwh, frames["df_train"], N_TRAIN * H)
    sim = _bundle(frames["df_ctrl_kwh"], frames["df_ctrl"], N_SIM * H)

    # The causal forecast, built over the WHOLE series once so the sim year's
    # first fortnight reads real history instead of a cold start.
    fc_con = rl.median14_forecast(df_all_kwh["Energy_Consumption"].values, H)
    fc_gen = rl.median14_forecast(df_all_kwh["Energy_Generation"].values, H)
    s0 = N_TRAIN * H
    n_ctrl = len(frames["df_ctrl_kwh"])
    return {
        "ident": ident, "tariff": tariff, "frames": frames,
        "train": train, "sim": sim,
        "fc_train": (fc_con[:s0], fc_gen[:s0]),
        "fc_sim": (fc_con[s0:s0 + n_ctrl], fc_gen[s0:s0 + n_ctrl]),
    }


def teacher_setpoints(prep) -> np.ndarray:
    """The whole-period MILP over `TEACH_SPAN`, cached as kWh.

    Solved on its own environment over exactly that span -- the contract it
    decides for itself inside the LP is then the contract of a household whose
    history is that span, which is the closest a demonstration can be to the
    conditions the student will meet. Never past day `N_TRAIN`: the teacher
    sees nothing of the scored year.

    The cache path carries the span. It used to be `<tariff>/<ident>.npz`, and
    widening the span from one year to two would have served the stale
    365-day solve under the new labels without a word.
    """
    ident, tariff = prep["ident"], prep["tariff"]
    a, b = TEACH_SPAN
    assert b <= N_TRAIN, "the teacher must not see the scored year"
    cache = os.path.join(MODELS, "teacher", tariff, f"d{a}-{b}", f"{ident}.npz")
    if os.path.exists(cache):
        return np.load(cache)["setpoints_kwh"]

    frames = prep["frames"]
    df_teach_kwh = frames["df_all_kwh"].iloc[a * H: b * H]
    env = hs.align_envelope(
        hs.build_study_env(df_teach_kwh, battery_cap=BATTERY_CAP,
                           soc_min_pct=SOC_MIN, soc_max_pct=SOC_MAX,
                           p_max=P_MAX, eff=EFF, delta_t=DELTA_T, H=H),
        P_MAX, EFF, DELTA_T)
    rates = hs.build_rate_vectors(tariff, env, df_teach_kwh.index,
                                  df_teach_kwh["SMP"].values,
                                  int(round(DELTA_T * 60)))
    sol = hs.solve_full_period(
        env, rates, tariff, n_steps=(b - a) * H,
        soc_init_kwh=SOC_INIT_ABS, delta_t=DELTA_T,
        soc_min_kwh=BATTERY_CAP * SOC_MIN, verbose=True)
    setpoints = (np.asarray(sol["x_ch"]) - np.asarray(sol["x_dis"])) * DELTA_T
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, setpoints_kwh=setpoints,
                        objective=sol["objective"],
                        runtime_s=sol["runtime_s"])
    return setpoints


# ---------------------------------------------------------------------------
# One (tariff, variant, method, household) run
# ---------------------------------------------------------------------------
def _digest(*dicts) -> str:
    return hashlib.sha256(
        json.dumps(dicts, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Which TrainConfig fields each method's result actually depends on. Behaviour
# cloning is a supervised fit: it never reads the replay buffer, the epsilon
# schedule, the n-step window or the BC regulariser, so digesting the WHOLE
# config against a BC result means every DQN tuning pass discards 70 perfectly
# valid clones. Same principle as the study's checkpoint backfill -- a result
# is stale only under a change that could have moved it.
_BC_FIELDS = ("hidden", "lr", "seed")


def _method_config(cfg: rl.TrainConfig, method: str,
                   clone: rl.BCOptions | None = None) -> dict:
    """The settings a result depends on. `clone`, the clone's optimiser, enters
    only for the methods that fit a clone and only through the fields that
    differ from the defaults -- so every result produced before those options
    existed keeps its digest (`test_rl_hpo` checks it)."""
    out = ({k: getattr(cfg, k) for k in _BC_FIELDS} if method == "bc"
           else cfg.config())
    if method != "dqn" and clone is not None and clone.changed():
        out = dict(out, clone=clone.changed())
    return out


def run_digest(cfg: rl.TrainConfig, spec, tariff: str, method: str,
               overrides: dict | None = None) -> str:
    """The identity of one result: its features, its method's settings, its
    split, and the learning rule that produced it.

    The version stamps are in here because the settings alone are not the
    method: a change to the update itself leaves every TrainConfig field
    untouched, and without them a superseded result reports as cached and is
    skipped. Cloning and reinforcement carry separate stamps so an RL fix does
    not discard every clone on disk -- and `bc_dqn` carries BOTH, because it is
    a clone that was then fine-tuned and either half moving moves the result.
    """
    clone = effective_clone(tariff, method, overrides)
    cfg = effective_config(cfg, tariff, method, overrides)
    version = {"bc": (rl.BC_ALGO_VERSION,),
               "dqn": (rl.ALGO_VERSION,),
               "bc_dqn": (rl.BC_ALGO_VERSION, rl.ALGO_VERSION)}[method]
    return _digest(spec.config(), _method_config(cfg, method, clone),
                   {"tariff": tariff, "method": method, "algo": version,
                    "train_days": TRAIN_BLOCKS, "val_days": VAL_BLOCKS,
                    "teach_span": TEACH_SPAN})


def make_config(steps: int, seed: int) -> rl.TrainConfig:
    # No per-cycle wear in the reward: the study prices the pack once, as an
    # NPV over min(12 y, 6000 EFC), so a cycle costs nothing until it would end
    # the pack early -- and that case is caught by `val_wear`, which scores the
    # validation rollouts on the same lifetime wear the results are reported
    # with. It used to be the pack price over its cycle life (0.417 EUR/EFC),
    # the same per-cycle charge the MILP carried; both were dropped together.
    return rl.TrainConfig(total_steps=steps, seed=seed, wear_eur_per_efc=0.0)


def val_wear_fn(tariff: str):
    """`(efc, n_days) -> EUR`: the study's lifetime wear for a validation window.

    The rollout's cycles annualised, priced by `hs.cycle_wear_eur` (the one
    formula `summarize` charges), and scaled back to the window. Zero unless the
    pace would end the pack before its calendar life.
    """
    def _wear(efc, n_days):
        per_year = float(efc) * 365.0 / max(n_days, 1)
        return float(hs.cycle_wear_eur(per_year, BATTERY_CAP, tariff)) * n_days / 365.0
    return _wear


# ---------------------------------------------------------------------------
# Hyperparameters: chosen on VALIDATION, per tariff and method
# ---------------------------------------------------------------------------
# The knobs the first screen set by looking at scored-year costs on household
# 138 (gamma 0.99 -> 0.997, n-step 8, the BC regulariser), re-decided on the
# validation weeks alone. `tune()` trains each grid point on every household
# WITHOUT evaluating the test year, `tune_report()` picks the point with the
# lowest validation cost net of wear, and the winners are written into TUNED
# by hand -- a visible, reviewable step, not one a sweep can take silently.
TUNE_VARIANT = "fc_h24"
TUNE_GRID = {
    "dqn": [{"gamma": g, "n_step": n} for g in (0.99, 0.997) for n in (1, 8)],
    # bc_dqn's exploration is BC-guided, so its continuations are near
    # on-policy and the n-step question is the DQN's; its own knob is how hard
    # the fine-tune is held to the clone.
    "bc_dqn": [{"gamma": g, "bc_reg": r} for g in (0.99, 0.997) for r in (0.0, 1.0)],
}
TUNE_OUT = os.path.join(HERE, "results_local", "rl_tune")
TUNE_MODELS = os.path.join(MODELS, "tune")

# {(tariff, method): {field: value}} -- the tune_report() winners, 2026-10-06,
# ALGO_VERSION 6: no per-cycle wear in the reward, validation on the bill plus
# the lifetime wear (zero here -- no validation rollout cycles fast enough to
# shorten the pack's life). 30 households, 140 validation days, test year never
# evaluated. Ties are kept honest in the comment: on every cell the gamma or
# n-step runner-up is within noise (p 0.26-0.58) EXCEPT AU, where gamma 0.997
# beats 0.99 for both methods (p < 0.001). The BC regulariser separates
# everywhere (p < 0.001) and is 1 on both tariffs: with cycles unpriced there
# is no longer anything for the fine-tune to gain by drifting off the clone --
# under the per-cycle price (ALGO 5) SI had picked 0 for exactly that reason.
TUNED: dict = {
    ("AU", "dqn"): {"gamma": 0.997, "n_step": 8},
    ("SI", "dqn"): {"gamma": 0.997, "n_step": 1},
    ("AU", "bc_dqn"): {"gamma": 0.997, "bc_reg": 1.0},
    ("SI", "bc_dqn"): {"gamma": 0.99, "bc_reg": 1.0},
}


# The clone's optimiser (`rl.BCOptions`) travels in the same override dicts as
# the TrainConfig fields -- a TUNED entry or a search trial is one dict -- under
# a prefix, because two names would collide: `batch` and `lr` are the DQN's.
CLONE_PREFIX = "clone_"


def effective_clone(tariff: str, method: str,
                    overrides: dict | None = None) -> rl.BCOptions:
    """The clone options in force: the `clone_*` keys of TUNED (or of the
    explicitly given overrides) over `rl.BCOptions()`'s defaults."""
    over = TUNED.get((tariff, method), {}) if overrides is None else overrides
    out = rl.BCOptions()
    for k, v in over.items():
        if k.startswith(CLONE_PREFIX):
            name = k[len(CLONE_PREFIX):]
            if not hasattr(out, name):
                raise AttributeError(f"BCOptions has no field {name!r}")
            setattr(out, name, v)
    return out


def tune_tag(overrides: dict) -> str:
    return "_".join(f"{k}{v:g}" for k, v in sorted(overrides.items()))


def effective_config(cfg: rl.TrainConfig, tariff: str, method: str,
                     overrides: dict | None = None) -> rl.TrainConfig:
    """`cfg` with the tuned (or the explicitly given) settings applied.
    Clone options (`clone_*`) are skipped here; `effective_clone` reads them."""
    over = TUNED.get((tariff, method), {}) if overrides is None else overrides
    over = {k: v for k, v in over.items() if not k.startswith(CLONE_PREFIX)}
    if not over:
        return cfg
    out = copy.copy(cfg)
    for k, v in over.items():
        if not hasattr(out, k):
            raise AttributeError(f"TrainConfig has no field {k!r}")
        setattr(out, k, v)
    return out


def _features(prep, spec, bundle_key: str):
    """(FeatureBuilder, normalized static matrix) for one bundle."""
    fb = rl.FeatureBuilder(spec)
    sig = prep[bundle_key]["sig"]
    fc = prep["fc_train"] if bundle_key == "train" else prep["fc_sim"]
    static = fb.build_static(sig, load_fc=fc[0], pv_fc=fc[1])
    return fb, static


def run_one(prep, variant: str, method: str, cfg: rl.TrainConfig,
            verbose: bool = True, out_root: str = OUT,
            models_root: str = MODELS, score_test: bool = True,
            overrides: dict | None = None) -> dict:
    """Train one (variant, method) for one household; score it on the test year.

    `score_test=False` is the TUNING path: the run is trained and validated,
    and the test year is never evaluated -- the result file carries no test
    number at all, so a tuning choice cannot be made on one even by accident.
    `overrides` replaces the TUNED settings (the grid point being tried).
    """
    ident, tariff = prep["ident"], prep["tariff"]
    cfg_in = cfg
    cfg = effective_config(cfg_in, tariff, method, overrides)
    clone = effective_clone(tariff, method, overrides)
    spec = variant_specs(tariff)[variant]
    respect_peak = tariff == "SI"
    key = f"{variant}__{method}"
    out_dir = os.path.join(out_root, tariff, key)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, f"{ident}.json")
    # Digested from the config as given plus the overrides, exactly as the
    # sweep's cache check computes it -- clone options included.
    digest = run_digest(cfg_in, spec, tariff, method, overrides)
    if os.path.exists(result_path):
        with open(result_path, encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("digest") == digest:
            return existing
        print(f"  [stale] {tariff}/{key}/{ident}: config changed, recomputing")

    t0 = time.time()
    train = prep["train"]
    fb, static = _features(prep, spec, "train")
    # Normalisation from the TRAINING blocks only -- the validation weeks and
    # the scored year must not leak into the scaler.
    train_rows = np.concatenate([np.arange(a * H, b * H) for a, b in TRAIN_BLOCKS])
    fb.fit_norm(static[train_rows])
    static_norm = fb.normalize(static)

    hist_bc = hist_dqn = None
    if method in ("bc", "bc_dqn"):
        setpoints = teacher_setpoints(prep)
        # Walk the whole teacher span; fit on the training blocks, early-stop
        # on the validation weeks.
        a, b = TEACH_SPAN
        net, hist_bc = rl.train_bc(
            train["sig"], train["settle"], train["env"], fb, static_norm,
            setpoints, start=a * H, stop=b * H,
            soc_init=SOC_INIT_USABLE, respect_peak=respect_peak, cfg=cfg,
            verbose=verbose, holdout_days=VAL_BLOCKS, opts=clone)
    if method in ("dqn", "bc_dqn"):
        bc_net = net if method == "bc_dqn" else None
        init = net if method == "bc_dqn" else None
        net, hist_dqn = rl.train_dqn(
            train["sig"], train["settle"], train["env"], fb, static_norm, cfg,
            respect_peak=respect_peak, train_days=TRAIN_BLOCKS,
            val_days=VAL_BLOCKS, init_net=init, bc_net=bc_net,
            soc_target=SOC_INIT_USABLE, verbose=verbose,
            val_wear=val_wear_fn(tariff))

    # The convergence measure every method reports on the same axis: the
    # greedy validation rollout, closed cost, summed over the held-out weeks.
    val = rl.validation_rollout(net, fb, static_norm, train["sig"],
                                train["settle"], train["env"], VAL_BLOCKS,
                                SOC_INIT_USABLE, respect_peak)
    # Validation on the axis the paper reports: the bill plus the lifetime wear
    # `summarize` charges (zero below ~500 EFC/a). Early stopping and TUNING
    # both read this one now.
    n_val_days = sum(b - a for a, b in VAL_BLOCKS)
    val_net = val["cost_eur_closed"] + val_wear_fn(tariff)(val["efc"], n_val_days)

    model_path = os.path.join(models_root, tariff, key, f"{ident}.pt")
    rl.save_model(model_path, net, fb, spec, cfg,
                  {"bc": hist_bc, "dqn": hist_dqn},
                  extra={"ident": str(ident), "tariff": tariff,
                         "variant": variant, "method": method})

    hist = hist_dqn or hist_bc or {}
    result = {
        "dataset": f"Ausgrid {ident}", "ident": str(ident), "tariff": tariff,
        "variant": variant, "method": method, "digest": digest,
        "causal": spec.causal, "spec": spec.config(),
        "val_cost_closed": val["cost_eur_closed"],
        "val_efc": val["efc"],
        "val_cost_net_of_wear": val_net,
        "train_converged": bool(hist.get("converged", False)),
        "train_steps": hist.get("steps_run"),
        "train_runtime_s": hist.get("runtime_s"),
        "best_val_cost_closed": hist.get("best_val_cost_closed"),
        "bc_val_agreement": (hist_bc or {}).get("final_val_agreement"),
        "bc_converged": (hist_bc or {}).get("converged"),
        # The budget this result was produced under, recorded rather than
        # implied. A straggler re-trained at a larger budget is otherwise
        # indistinguishable on disk from one that converged at the default,
        # and a paired comparison that silently mixes the two is comparing
        # budgets, not observations.
        "train_config": cfg.config(),
        "model_path": os.path.relpath(model_path, HERE),
    }
    if method != "dqn" and clone.changed():
        result["clone_options"] = clone.changed()
    if not score_test:
        result["wall_s"] = time.time() - t0
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        return result

    # -- score on the TEST year, through the study's own runner -----------
    sim = prep["sim"]
    policy = rl.LearnedPolicy(
        net, fb, respect_peak,
        load_fc=prep["fc_sim"][0], pv_fc=prep["fc_sim"][1],
        name=f"{method}_{variant}", label=f"{method.upper()} {variant}",
        causal=spec.causal)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rbc.run_policy(sim["env"], policy, n_steps=N_SIM * H,
                             settle=sim["settle"],
                             soc_init_kwh=SOC_INIT_USABLE,
                             rates=sim["rates"])

    result.update({
        "cost_eur_closed": out["Cost_EUR_Closed"],
        "cost_eur": out["Cost_EUR"],
        "fixed_eur": out["Fixed_EUR"],
        "cost_eur_total": out["Cost_EUR_Total"],
        "efc": out["Equivalent_Full_Cycles"],
        "import_kwh": out["Import_kWh"], "export_kwh": out["Export_kWh"],
        "peak_import_kw": out["Peak_Import_kW"],
        "agreed_power_iters": out["Agreed_Power_Iters"],
        "agreed_power_converged": out["Agreed_Power_Converged"],
        "wall_s": time.time() - t0,
    })
    result.update(_comparators(ident, tariff))
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


def _comparators(ident, tariff: str) -> dict:
    """The study's own numbers for this household, read off the arm checkpoint.

    Read-only context: the RL screen never recomputes them, it reports beside
    them. Missing checkpoints just leave the keys out.
    """
    arm = hs.REFERENCE_ARM[tariff]
    path = os.path.join(HERE, "results_local", arm, f"Ausgrid {ident}",
                        "checkpoint.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh).get("metrics", {})
    except Exception:
        return {}
    keys = ["cost_no_battery", "cost_self_consumption", "cost_prophet",
            "cost_oracle", "cost_milp_full", "cost_price_threshold",
            "cost_self_consumption_peak_shaving", "cost_tariff_arbitrage"]
    return {f"ref_{k}": blob[k] for k in keys if k in blob}


# ---------------------------------------------------------------------------
# Parallel execution: one process per household
# ---------------------------------------------------------------------------
# The unit of parallelism is the HOUSEHOLD, not the run, because
# `prepare_household` is the expensive shared object: splitting by run would
# rebuild it per worker, and splitting by household lets one worker amortise it
# over every variant and method that household needs. Each worker is a separate
# process with its own calendar globals (`_si_cas.nastavi_koledar` is module
# state, and two tariffs must never share one), and torch is pinned to a single
# thread per worker so N workers ask for N cores rather than N * cores.
def _worker(group, cfg, quiet):
    """Run every job for one (tariff, household). Returns result summaries."""
    tariff, ident, items = group
    prep = None
    out = []
    for variant, method in items:
        if prep is None:
            prep = prepare_household(ident, tariff)
        try:
            res = run_one(prep, variant, method, cfg, verbose=not quiet)
            out.append((tariff, variant, method, ident,
                        res["cost_eur_closed"], res["train_converged"], None))
        except Exception as exc:                     # one run must not sink the sweep
            out.append((tariff, variant, method, ident, None, None, repr(exc)))
    return out


def _run_parallel(jobs, cfg, args):
    """The sweep across processes, grouped by household. Same results, same
    files, same resumability -- only the order they finish in differs."""
    from joblib import Parallel, delayed

    # Drop what is already on disk under this configuration BEFORE grouping, so
    # a resumed sweep does not hand a worker a household with nothing to do.
    todo = {}
    cached = 0
    for tariff, variant, method, ident in jobs:
        path = os.path.join(OUT, tariff, f"{variant}__{method}", f"{ident}.json")
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    existing = json.load(fh)
                if existing.get("digest") == run_digest(
                        cfg, variant_specs(tariff)[variant], tariff, method):
                    cached += 1
                    continue
            except Exception:
                pass                                  # unreadable: recompute
        todo.setdefault((tariff, ident), []).append((variant, method))

    groups = [(t, i, items) for (t, i), items in todo.items()]
    n_runs = sum(len(g[2]) for g in groups)
    if not groups:
        print(f"nothing to do: all {cached} run(s) cached under this config")
        print("ALL DONE")
        summarize()
        return

    n_jobs = args.jobs
    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    n_jobs = min(n_jobs, len(groups))
    print(f"{n_runs} run(s) over {len(groups)} household-tariff group(s) on "
          f"{n_jobs} worker(s); {cached} already cached", flush=True)
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_worker)(g, cfg, args.quiet) for g in groups)

    flat = [r for group in results for r in group]
    failed = [r for r in flat if r[6] is not None]
    unconverged = [r for r in flat if r[5] is False]
    elapsed = time.time() - t0
    print(f"\n{len(flat)} run(s) in {elapsed/60:.1f} min "
          f"({elapsed/max(len(flat),1):.0f} s/run wall, {n_jobs} workers)")
    if unconverged:
        print(f"{len(unconverged)} run(s) did NOT converge:")
        for t, v, m, i, *_ in unconverged:
            print(f"  {t} {v} {m} Ausgrid {i}")
    if failed:
        print(f"{len(failed)} run(s) FAILED:")
        for t, v, m, i, _, _, exc in failed:
            print(f"  {t} {v} {m} Ausgrid {i}: {exc}")
        print("rerun to retry the failures; everything else is cached")
        return
    print("ALL DONE")
    summarize()


# ---------------------------------------------------------------------------
# Tuning: validation only
# ---------------------------------------------------------------------------
def _tune_worker(group, cfg, quiet):
    """Every grid point for one (tariff, household); no test-year scoring."""
    tariff, ident, items = group
    prep = prepare_household(ident, tariff)
    out = []
    for method, over in items:
        tag = tune_tag(over)
        try:
            res = run_one(prep, TUNE_VARIANT, method, cfg, verbose=not quiet,
                          out_root=os.path.join(TUNE_OUT, tag),
                          models_root=os.path.join(TUNE_MODELS, tag),
                          score_test=False, overrides=over)
            assert "cost_eur_closed" not in res, "tuning touched the test year"
            out.append((tariff, method, tag, ident, None))
        except Exception as exc:
            out.append((tariff, method, tag, ident, repr(exc)))
    return out


def tune(ids, tariffs, cfg, n_jobs: int = 1, quiet: bool = True):
    """Train TUNE_GRID on `ids` x `tariffs`, scoring validation weeks only.

    Resumable like the sweep: a grid point already on disk under its digest is
    skipped. Results land in `results_local/rl_tune/<tag>/`, never in the
    screen's own tree, so `load_results()` cannot mix them into the panel.
    """
    from joblib import Parallel, delayed

    todo, cached = {}, 0
    for tariff in tariffs:
        spec = variant_specs(tariff)[TUNE_VARIANT]
        for ident in ids:
            for method, grid in TUNE_GRID.items():
                for over in grid:
                    path = os.path.join(TUNE_OUT, tune_tag(over), tariff,
                                        f"{TUNE_VARIANT}__{method}", f"{ident}.json")
                    if os.path.exists(path):
                        try:
                            with open(path, encoding="utf-8") as fh:
                                if json.load(fh).get("digest") == run_digest(
                                        cfg, spec, tariff, method, overrides=over):
                                    cached += 1
                                    continue
                        except Exception:
                            pass
                    todo.setdefault((tariff, ident), []).append((method, over))
    groups = [(t, i, items) for (t, i), items in todo.items()]
    n_runs = sum(len(g[2]) for g in groups)
    if n_runs:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs) if n_jobs < 0 else n_jobs
        n_jobs = min(n_jobs, len(groups))
        print(f"tuning: {n_runs} run(s) over {len(groups)} group(s) on "
              f"{n_jobs} worker(s); {cached} already cached", flush=True)
        t0 = time.time()
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(_tune_worker)(g, cfg, quiet) for g in groups)
        flat = [r for g in results for r in g]
        failed = [r for r in flat if r[4] is not None]
        print(f"\n{len(flat)} tuning run(s) in {(time.time() - t0) / 60:.1f} min")
        if failed:
            for t, m, tag, i, exc in failed:
                print(f"  FAILED {t} {m} {tag} Ausgrid {i}: {exc}")
            print("rerun to retry the failures; everything else is cached")
            return None
    print("ALL DONE")
    return tune_report()


def tune_report():
    """Validation cost per grid point, and the winner per (tariff, method).

    Picked on the TOTAL validation cost net of wear over every household --
    the paper's axis, paired by construction (same households, same weeks).
    The paired Wilcoxon against the winner says whether the runner-up is
    distinguishable at all; when it is not, either choice is defensible and
    the winner is kept only because some choice has to be.
    """
    import pandas as pd
    from scipy.stats import wilcoxon

    rows = []
    if os.path.isdir(TUNE_OUT):
        for tag in sorted(os.listdir(TUNE_OUT)):
            for tariff in ("AU", "SI"):
                for method in TUNE_GRID:
                    d = os.path.join(TUNE_OUT, tag, tariff,
                                     f"{TUNE_VARIANT}__{method}")
                    if not os.path.isdir(d):
                        continue
                    for f in sorted(os.listdir(d)):
                        if f.endswith(".json"):
                            with open(os.path.join(d, f), encoding="utf-8") as fh:
                                r = json.load(fh)
                            rows.append({"tag": tag, "tariff": tariff,
                                         "method": method, "ident": r["ident"],
                                         "val_net": r["val_cost_net_of_wear"],
                                         "val_bill": r["val_cost_closed"],
                                         "val_efc": r["val_efc"],
                                         "converged": r["train_converged"]})
    if not rows:
        print("no tuning results on disk")
        return None
    df = pd.DataFrame(rows)
    winners = {}
    for (tariff, method), g in df.groupby(["tariff", "method"]):
        wide = g.pivot(index="ident", columns="tag", values="val_net").dropna()
        summary = (g[g["ident"].isin(wide.index)]
                   .groupby("tag")
                   .agg(n=("ident", "size"), val_net=("val_net", "mean"),
                        val_bill=("val_bill", "mean"), efc=("val_efc", "mean"),
                        converged=("converged", "mean"))
                   .sort_values("val_net"))
        best = summary.index[0]
        p = {}
        for tag in summary.index[1:]:
            diff = wide[tag] - wide[best]
            p[tag] = (wilcoxon(diff).pvalue
                      if (diff != 0).sum() >= 6 else float("nan"))
        summary["vs_best_p"] = pd.Series(p)
        print(f"\n{tariff} {method}  (mean validation EUR over "
              f"{len(wide)} households, {sum(b - a for a, b in VAL_BLOCKS)} days)")
        print(summary.round(3).to_string())
        winners[(tariff, method)] = best
    print("\nwinners (copy into TUNED):")
    for k, tag in winners.items():
        over = next(o for o in TUNE_GRID[k[1]] if tune_tag(o) == tag)
        print(f"    {k!r}: {over!r},")
    return df


# ---------------------------------------------------------------------------
# SI on the test year: where the money goes
# ---------------------------------------------------------------------------
# A DIAGNOSTIC of results already scored, not a step in training: nothing here
# feeds back into a model, a setting or a selection. It re-runs the stored
# models and the SI rules through `run_policy` -- the same call that scored
# them -- keeping what the scoring summed away: the bill split into energy,
# excess-power charge and the contract-dependent standing charge, and the
# contract each controller walked itself onto, month by month.
DIAG_OUT = os.path.join(HERE, "results_local", "rl_si_diagnosis")
DIAG_RULES = ["no_battery", "self_consumption", "peak_shaving",
              "self_consumption_peak_shaving"]
DIAG_LEARNED = ["bc", "bc_dqn", "dqn"]


def _diag_one(env, sig, policy, settle, rates):
    """One converged test-year run, with the contract it ended up billed under."""
    out = rbc.run_policy(env, policy, n_steps=N_SIM * H, settle=settle,
                         soc_init_kwh=SOC_INIT_USABLE, rates=rates,
                         keep_traces=True)
    p = np.asarray(out["_setpoints"])
    ch, dis = np.maximum(p, 0.0), np.maximum(-p, 0.0)
    n = len(p)
    net = sig.consumption[:n] + ch - sig.generation[:n] - dis
    kw = np.maximum(net, 0.0) / sig.hours
    # Month by month, per block: the peak drawn and the agreed power billed.
    # `agreed_kw` is read AFTER convergence, so it is the contract this run's
    # own peaks set -- what the household would actually be on.
    sig_c = rbc.rebind_agreed_power(sig, env)
    monthly = {}
    for m in np.unique(sig.windows[:n]):
        sel = sig.windows[:n] == m
        row = {}
        for b in range(1, 6):
            mb = sel & (sig.blocks[:n] == b)
            if mb.any():
                row[str(b)] = {"peak_kw": float(kw[mb].max()),
                               "agreed_kw": float(sig_c.agreed_kw[:n][mb][0]),
                               "steps": int(mb.sum())}
        monthly[str(int(m))] = row
    return {
        "energy_eur": out["Energy_EUR"], "power_eur": out["Power_EUR"],
        "fixed_eur": out["Fixed_EUR"], "termadj_eur": out["Terminal_SOC_Adj_EUR"],
        "cost_eur_closed": out["Cost_EUR_Closed"], "efc": out["Equivalent_Full_Cycles"],
        "import_kwh": out["Import_kWh"], "export_kwh": out["Export_kWh"],
        "peak_import_kw": out["Peak_Import_kW"],
        "agreed_iters": out["Agreed_Power_Iters"],
        "agreed_converged": out["Agreed_Power_Converged"],
        "monthly": monthly,
    }


def _scored_digests(ident):
    """{method: digest} of the scored SI runs a diagnosis re-runs -- its cache key.

    Keyed on the household alone it served the PREVIOUS models after the RL
    panel was retrained (ALGO_VERSION 6): bc_dqn read 13 EUR/a in the
    diagnosis against 44 in the scored panel.
    """
    out = {}
    for method in DIAG_LEARNED:
        path = os.path.join(OUT, "SI", f"fc_h24__{method}", f"{ident}.json")
        with open(path, encoding="utf-8") as fh:
            out[method] = json.load(fh).get("digest")
    return out


def _diag_current(ident):
    path = os.path.join(DIAG_OUT, f"{ident}.json")
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("scored_digests") == _scored_digests(ident)


def _diag_household(ident):
    path = os.path.join(DIAG_OUT, f"{ident}.json")
    if _diag_current(ident):
        return path
    prep = prepare_household(ident, "SI")
    sim = prep["sim"]
    env, sig, settle, rates = sim["env"], sim["sig"], sim["settle"], sim["rates"]
    roster = {p.name: p for p in hs.rule_roster("SI")}
    roster[rbc.NO_BATTERY] = rbc._Idle()        # the runner's own baseline row
    res = {}
    for name in DIAG_RULES:
        res[name] = _diag_one(env, sig, roster[name], settle, rates)
    for method in DIAG_LEARNED:
        net, fb, spec, _ = rl.load_model(
            os.path.join(MODELS, "SI", f"fc_h24__{method}", f"{ident}.pt"))
        pol = rl.LearnedPolicy(net, fb, True, load_fc=prep["fc_sim"][0],
                               pv_fc=prep["fc_sim"][1], name=method)
        res[method] = _diag_one(env, sig, pol, settle, rates)
    os.makedirs(DIAG_OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"ident": str(ident), "controllers": res,
                   "scored_digests": _scored_digests(ident)}, fh)
    return path


def si_test_diagnosis(ids, n_jobs: int = 1):
    """Run `_diag_household` for every id (cached), then check it reproduces
    the scored numbers -- a diagnostic that disagreed with the scoring would be
    diagnosing a different run."""
    from joblib import Parallel, delayed
    todo = [i for i in ids if not _diag_current(i)]
    if todo:
        Parallel(n_jobs=min(max(n_jobs, 1), len(todo)), backend="loky", verbose=5)(
            delayed(_diag_household)(i) for i in todo)
    worst = 0.0
    for ident in ids:
        with open(os.path.join(DIAG_OUT, f"{ident}.json"), encoding="utf-8") as fh:
            c = json.load(fh)["controllers"]
        for method in DIAG_LEARNED:
            with open(os.path.join(OUT, "SI", f"fc_h24__{method}", f"{ident}.json"),
                      encoding="utf-8") as fh:
                scored = json.load(fh)
            worst = max(worst, abs(c[method]["cost_eur_closed"] - scored["cost_eur_closed"]),
                        abs(c[method]["fixed_eur"] - scored["fixed_eur"]))
        ref = _comparators(ident, "SI")
        for name in DIAG_RULES:
            k = f"ref_cost_{name}"
            if k in ref:
                worst = max(worst, abs(c[name]["cost_eur_closed"] - ref[k]))
    print(f"diagnosis reproduces the scored test-year numbers to {worst:.2e} EUR")
    return worst


def si_diagnosis_frames(from_checkpoint=None):
    """(components, monthly) from the cached diagnosis, for drawing.

    `components`: one row per (household, controller), each SI bill line as a
    SAVING against the household's own no-battery row -- energy, excess-power
    charge, the contract-dependent standing charge, the terminal-SOC close-out
    -- plus `wear`, the cost of the pack life the year's cycling uses up
    (`hs.cycle_wear_eur`: zero unless the cycles end the pack before its 12 y
    calendar band), and their sum `net`, which is the study's
    `saving_annual_net`. `wear_shadow` is the per-cycle price the MILP and the
    learned agents DISPATCH against, kept for comparison and not in `net`. The MILP-family controllers come from arm
    checkpoints; their components are the same settlement's, stored by the
    sweep. `from_checkpoint` is `{row name: (arm, checkpoint key)}`, default
    the full-year MILP of the SI reference arm -- pass e.g.
    `{"mpc_best": ("SI_H24_prophet_tuned", "prophet")}` to add an MPC run on
    another arm. They have no monthly contract in the checkpoint, so they are
    absent from `monthly`.

    `monthly`: one row per (household, controller, month, block): the peak drawn
    and the agreed power billed, and both as a change against no battery.
    Learned controllers are named `<method>_fc_h24`, as in the study frame.
    """
    import pandas as pd

    if from_checkpoint is None:
        from_checkpoint = {"milp_full": (hs.REFERENCE_ARM["SI"], "milp_full")}
    shadow = be.cycle_cost_eur_per_efc(BATTERY_CAP)
    wear = lambda efc: -float(hs.cycle_wear_eur(efc, BATTERY_CAP, "SI"))
    rows, mon = [], []
    for f in sorted(os.listdir(DIAG_OUT)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(DIAG_OUT, f), encoding="utf-8") as fh:
            d = json.load(fh)
        ident, ctrls = d["ident"], d["controllers"]
        nb = ctrls[rbc.NO_BATTERY]
        for name, v in ctrls.items():
            c = f"{name}_fc_h24" if name in DIAG_LEARNED else name
            rows.append({"ident": ident, "controller": c,
                         "energy": nb["energy_eur"] - v["energy_eur"],
                         "power": nb["power_eur"] - v["power_eur"],
                         "contract": nb["fixed_eur"] - v["fixed_eur"],
                         "termadj": -v["termadj_eur"],
                         "wear": wear(v["efc"]), "wear_shadow": -shadow * v["efc"],
                         "efc": v["efc"]})
            for m, blocks in v["monthly"].items():
                for b, x in blocks.items():
                    y = nb["monthly"][m][b]
                    mon.append({"ident": ident, "controller": c, "month": int(m),
                                "block": int(b), "peak_kw": x["peak_kw"],
                                "agreed_kw": x["agreed_kw"],
                                "d_peak_kw": x["peak_kw"] - y["peak_kw"],
                                "d_agreed_kw": x["agreed_kw"] - y["agreed_kw"]})
        for name, (arm, key) in from_checkpoint.items():
            ck_path = os.path.join(HERE, "results_local", arm,
                                   f"Ausgrid {ident}", "checkpoint.json")
            if not os.path.exists(ck_path):
                continue
            with open(ck_path, encoding="utf-8") as fh:
                ck = json.load(fh)["metrics"]
            rows.append({"ident": ident, "controller": name,
                         "energy": nb["energy_eur"] - ck[f"energy_{key}"],
                         "power": nb["power_eur"] - ck[f"power_{key}"],
                         "contract": nb["fixed_eur"] - ck[f"fixed_{key}"],
                         "termadj": -ck[f"termadj_{key}"],
                         "wear": wear(ck[f"efc_{key}"]),
                         "wear_shadow": -shadow * ck[f"efc_{key}"],
                         "efc": ck[f"efc_{key}"]})
    comp = pd.DataFrame(rows)
    comp["net"] = comp[["energy", "power", "contract", "termadj", "wear"]].sum(axis=1)
    return comp, pd.DataFrame(mon)


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", nargs="*", type=int, default=DEFAULT_IDS)
    ap.add_argument("--tariffs", nargs="*", default=["AU", "SI"])
    ap.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    ap.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="smoke-test budget: 30k steps, eval every 5k")
    ap.add_argument("--jobs", type=int, default=1,
                    help="households trained in parallel (one process each; "
                         "negative counts back from the core count)")
    ap.add_argument("--retrain-unconverged", action="store_true",
                    help="run only the reinforcement runs whose training never "
                         "settled, at whatever --steps is given. Their result "
                         "files record the budget they were produced under, so "
                         "a mixed-budget panel stays visible rather than "
                         "implied.")
    ap.add_argument("--tune", action="store_true",
                    help="train the TUNE_GRID on --ids, validation only (the "
                         "test year is never evaluated), then print "
                         "tune_report()")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    cfg = make_config(args.steps, args.seed)
    if args.quick:
        cfg.total_steps = 30_000
        cfg.eval_every = 5_000

    # Household outermost: `prepare_household` is tens of seconds of rate
    # vectors and signal building, cached per (household, tariff) and dropped
    # when the household changes -- so the loop order IS the prep bill. With
    # the household innermost the first screen was rebuilding it once per
    # (variant, method) group, ~200 times instead of 10.
    if args.tune:
        return tune(args.ids, args.tariffs, cfg, args.jobs, args.quiet)
    if args.retrain_unconverged:
        jobs = [(t, v, m, int(i)) for t, v, m, i in unconverged_runs()]
        if not jobs:
            print("every reinforcement run settled; nothing to re-train")
            return
        print(f"re-training {len(jobs)} unsettled run(s) at "
              f"{cfg.total_steps:,} steps")
    else:
        jobs = []
        for tariff in args.tariffs:
            known = variant_specs(tariff)
            for ident in args.ids:
                for variant in args.variants:
                    if variant not in known:
                        continue               # tariff-specific ablations
                    for method in args.methods:
                        jobs.append((tariff, variant, method, ident))

    os.makedirs(OUT, exist_ok=True)
    if args.jobs != 1:
        return _run_parallel(jobs, cfg, args)
    done, durations = 0, []
    preps = {}
    for n, (tariff, variant, method, ident) in enumerate(jobs, 1):
        result_path = os.path.join(OUT, tariff, f"{variant}__{method}",
                                   f"{ident}.json")
        header = f"[{n}/{len(jobs)}] {tariff} {variant} {method} Ausgrid {ident}"
        if os.path.exists(result_path):
            with open(result_path, encoding="utf-8") as fh:
                existing = json.load(fh)
            spec = variant_specs(tariff)[variant]
            if existing.get("digest") == run_digest(cfg, spec, tariff, method):
                print(f"{header}: cached "
                      f"({existing['cost_eur_closed']:.2f} closed)", flush=True)
                done += 1
                continue
        key = (ident, tariff)
        if key not in preps:
            print(f"  preparing Ausgrid {ident} / {tariff} ...", flush=True)
            preps[key] = prepare_household(ident, tariff)
            # One household's bundles are ~100 MB of signals; keep only the
            # current household in memory.
            for k in list(preps):
                if k != key:
                    del preps[k]
        t0 = time.time()
        print(header, flush=True)
        res = run_one(preps[key], variant, method, cfg,
                      verbose=not args.quiet)
        durations.append(time.time() - t0)
        done += 1
        remaining = len(jobs) - n
        if durations and remaining:
            eta_s = float(np.mean(durations)) * remaining
            finish = datetime.now() + timedelta(seconds=eta_s)
            print(f"{header}: {res['cost_eur_closed']:.2f} closed, "
                  f"converged={res['train_converged']} "
                  f"[{durations[-1]:.0f}s; ~{eta_s/60:.0f} min left, "
                  f"ETA {finish:%H:%M}]", flush=True)
        else:
            print(f"{header}: {res['cost_eur_closed']:.2f} closed, "
                  f"converged={res['train_converged']}", flush=True)

    if done == len(jobs):
        print("ALL DONE")
        summarize()


def load_results():
    """Every result JSON on disk as one long frame, with the saving column.

    Savings rather than costs are the comparable axis: absolute annual bills
    differ two-fold across these households, so a mean cost is dominated by
    which households happen to be in the sample, while a saving against each
    household's OWN no-battery reference is not.
    """
    import pandas as pd

    rows = []
    for tariff in ("AU", "SI"):
        base = os.path.join(OUT, tariff)
        if not os.path.isdir(base):
            continue
        for key in sorted(os.listdir(base)):
            d = os.path.join(base, key)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.endswith(".json"):
                    with open(os.path.join(d, f), encoding="utf-8") as fh:
                        rows.append(json.load(fh))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "ref_cost_no_battery" in df.columns:
        df["saving"] = df["ref_cost_no_battery"] - df["cost_eur_closed"]
    return df


def summarize():
    """One frame over every result JSON on disk, printed as a pivot."""
    df = load_results()
    if df is None:
        print("no results yet")
        return None
    print(df.pivot_table(index=["tariff", "variant"], columns="method",
                         values="cost_eur_closed", aggfunc="mean")
          .round(2).to_string())
    if "ref_cost_self_consumption" in df.columns:
        refs = (df.groupby("tariff")[["ref_cost_no_battery",
                                      "ref_cost_self_consumption",
                                      "ref_cost_prophet", "ref_cost_milp_full"]]
                .mean(numeric_only=True))
        print("\nreference controllers (study checkpoints, same households):")
        print(refs.round(2).to_string())
    return df


def report(baseline_variant: str = "fc_h24"):
    """The screen read as the study reads its own arms: paired, per household.

    Every comparison here is WITHIN a household -- the same roof, load and
    contract under two observation contracts or two methods -- because the
    between-household spread is several times the effect being measured. On
    the five-household screen that spread was 62.8 EUR (AU) against a
    between-variant spread of 10.2, which is why the first screen could not
    answer the forecast question at all and why these are Wilcoxon tests on
    per-household differences rather than differences of means.

    `hems_study.paired_comparison` does the test, so the RL arms are scored by
    the same statistic as the MILP arms.
    """
    df = load_results()
    if df is None:
        print("no results yet")
        return None

    n_house = df.groupby("tariff")["ident"].nunique().to_dict()
    print(f"households per tariff: {n_house}")
    # fillna BEFORE the inversion: see the note in `convergence_report`.
    bad = df[~df["train_converged"].fillna(False).astype(bool)]
    print(f"runs: {len(df)}   not converged: {len(bad)}")
    if len(bad):
        print(bad.groupby(["tariff", "method"]).size().to_string())

    for tariff in sorted(df["tariff"].unique()):
        sub = df[df["tariff"] == tariff]
        print(f"\n{'='*66}\n{tariff}: mean saving vs no battery "
              f"(EUR-equivalent/year, n={n_house.get(tariff)})\n{'='*66}")
        print(sub.pivot_table(index="variant", columns="method",
                             values="saving", aggfunc="mean")
              .round(1).to_string())

        # The study's own controllers on exactly these households.
        refs = sub.drop_duplicates("ident")
        ref_cols = [c for c in refs.columns if c.startswith("ref_cost_")
                    and c != "ref_cost_no_battery"]
        if ref_cols and refs["ref_cost_no_battery"].notna().all():
            print("\n  reference controllers, same households:")
            for c in sorted(ref_cols):
                if refs[c].notna().all():
                    print(f"    {c[9:]:34s} {(refs.ref_cost_no_battery - refs[c]).mean():7.1f}")

        for method in sorted(sub["method"].unique()):
            wide = sub[sub["method"] == method].pivot(
                index="ident", columns="variant", values="saving")
            if baseline_variant not in wide.columns:
                continue
            print(f"\n  {method}: each variant against {baseline_variant}, "
                  f"paired by household")
            rows = []
            for v in wide.columns:
                if v == baseline_variant:
                    continue
                r = hs.paired_comparison(wide, v, baseline_variant)
                rows.append({"variant": v, "median_diff": r["median_diff"],
                             "wins": r["n_a_greater"], "n": r["n"],
                             "p": r["p_value"]})
            if rows:
                import pandas as pd
                print(pd.DataFrame(rows).set_index("variant")
                      .round(3).to_string())

        # Methods against each other on the baseline observation contract.
        wide_m = sub[sub["variant"] == baseline_variant].pivot(
            index="ident", columns="method", values="saving")
        if len(wide_m.columns) > 1:
            print(f"\n  methods against each other on {baseline_variant}:")
            import pandas as pd
            rows = [hs.paired_comparison(wide_m, a, b)
                    for i, a in enumerate(sorted(wide_m.columns))
                    for b in sorted(wide_m.columns)[i + 1:]]
            print(pd.DataFrame(rows)[["comparison", "n", "median_diff",
                                      "n_a_greater", "p_value"]]
                  .round(3).to_string(index=False))
    return df


# Each pair is one QUESTION about information, asked as a difference between
# two observation contracts run on the same households, battery and
# settlement: (label, the richer contract, the poorer one).
VOI_QUESTIONS = [
    ("a day-ahead forecast at all   (fc_h24 - nofc)", "fc_h24", "nofc"),
    ("a perfect ROOF on top of it   (pvtruth - fc_h24)",
     "loadfc_pvtruth_h24", "fc_h24"),
    ("perfect EVERYTHING, ceiling   (truth - fc_h24)", "truth_h24", "fc_h24"),
    ("horizon 24 h over 12 h        (fc_h24 - fc_h12)", "fc_h24", "fc_h12"),
    ("horizon 12 h over 6 h         (fc_h12 - fc_h6)", "fc_h12", "fc_h6"),
    ("AU: the price signal          (fc_h24 - noprice)",
     "fc_h24", "fc_h24_noprice"),
    ("SI: the contract/peak state   (fc_h24 - nopeak)",
     "fc_h24", "fc_h24_nopeak"),
]


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the order given.

    Six questions are asked of every method on every tariff, so an uncorrected
    0.05 would be expected to manufacture a finding roughly once per family.
    Holm rather than plain Bonferroni because it is uniformly more powerful at
    the same family-wise error rate -- and the distinction it draws here is the
    one that matters: perfect foresight survives at p ~ 1e-3 while every
    marginal result (the price features at 0.020, the horizon steps at 0.047)
    does not. NaNs -- families too small for the exact test -- pass through.
    """
    idx = [i for i, p in enumerate(pvalues) if p == p]
    m = len(idx)
    out = list(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: pvalues[j])):
        # Monotone in rank: an adjusted p may never fall below an earlier one.
        running = max(running, (m - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out


def value_of_information(questions=None):
    """What each piece of information is WORTH, per tariff, paired by household.

    The headline table of this screen. Every row is a within-household
    difference between two observation contracts -- same roof, same load, same
    contract, same battery, same settlement -- so it prices the information
    rather than the households that happened to carry it, and it is corrected
    for the number of questions asked.
    """
    import pandas as pd

    df = load_results()
    if df is None:
        print("no results yet")
        return None
    n = df.groupby("tariff")["ident"].nunique().to_dict()
    print(f"households: {n}\nruns: {len(df)}\n")

    collected = []
    for tariff in sorted(df.tariff.unique()):
        sub = df[df.tariff == tariff]
        print("=" * 74)
        print(f"{tariff}: value of information, EUR-equivalent/year, "
              f"paired over {n[tariff]} households")
        print("=" * 74)
        for method in sorted(sub.method.unique()):
            wide = sub[sub.method == method].pivot(
                index="ident", columns="variant", values="saving")
            rows = []
            for label, rich, poor in (questions or VOI_QUESTIONS):
                if rich not in wide.columns or poor not in wide.columns:
                    continue
                r = hs.paired_comparison(wide, rich, poor)
                rows.append({"question": label,
                             "median": round(r["median_diff"], 2),
                             "q1": round(r["q1_diff"], 2),
                             "q3": round(r["q3_diff"], 2),
                             "wins": f"{r['n_a_greater']}/{r['n']}",
                             "p": round(r["p_value"], 4)
                                  if r["p_value"] == r["p_value"] else float("nan")})
            if not rows:
                continue
            for r, a in zip(rows, holm([x["p"] for x in rows])):
                r["p_holm"] = round(a, 4) if a == a else float("nan")
                r["sig"] = ("**" if a == a and a < 0.01
                            else "*" if a == a and a < 0.05 else "")
                collected.append({"tariff": tariff, "method": method, **r})
            print(f"\n  {method}")
            print(pd.DataFrame(rows).to_string(index=False))

        print(f"\n  {tariff}: mean saving, learned vs the study's own roster")
        best = (sub.groupby(["method", "variant"])["saving"].mean()
                .sort_values(ascending=False))
        for (m, v), s in best.head(6).items():
            print(f"    {m:7s} {v:20s} {s:7.1f}")
        refs = sub.drop_duplicates("ident")
        for c in sorted(c for c in refs.columns if c.startswith("ref_cost_")):
            if c != "ref_cost_no_battery" and refs[c].notna().all():
                print(f"    {'ref':7s} {c[9:]:20s} "
                      f"{(refs.ref_cost_no_battery - refs[c]).mean():7.1f}")
        print()
    return pd.DataFrame(collected)


def unconverged_runs(tail_frac: float = 0.25, tail_limit: float = 0.25):
    """The (tariff, variant, method, ident) tuples whose training did not settle.

    Same rule `convergence_report` prints: the early stop never fired, or a
    quarter or more of the run's total improvement arrived in its final
    quarter -- a curve still descending when the budget ended.
    """
    df = _convergence_frame(tail_frac)
    if df is None:
        return []
    learned = df[df["method"].isin(["dqn", "bc_dqn"])]
    if not len(learned):
        return []
    conv = learned["converged"].fillna(False).astype(bool)
    tail = learned["improvement_tail"].fillna(1.0)
    bad = learned[(~conv) | (tail > tail_limit)]
    return [(r.tariff, r.variant, r.method, r.ident)
            for r in bad.itertuples()]


def _convergence_frame(tail_frac: float = 0.25):
    """Did training actually finish, or did the budget run out mid-descent?

    `train_converged` is the runner's own verdict -- the early stop fired, or
    the best validation cost sat at least `patience` evaluations before the
    last one. It is necessary but not sufficient evidence, because a run can
    also flatten out at a bad policy, so three more things are read off the
    stored validation traces:

        improvement_tail   how much of the run's TOTAL improvement arrived in
                           its last quarter. Near zero is a converged run;
                           a large share means the curve was still falling
                           when the budget ended and the number is a floor,
                           not an estimate.
        best_frac          where in the run the best evaluation sits, as a
                           fraction of it. 1.0 means the final evaluation was
                           the best one -- the signature of a truncated run.
        n_evals            how many validation points the verdict rests on.

    Reads the history JSONs beside the models rather than the result files,
    because the trace is what a convergence figure is drawn from.
    """
    import pandas as pd

    rows = []
    for tariff in ("AU", "SI"):
        base = os.path.join(MODELS, tariff)
        if not os.path.isdir(base):
            continue
        for key in sorted(os.listdir(base)):
            d = os.path.join(base, key)
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if not f.endswith(".pt.history.json"):
                    continue
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    hist = json.load(fh)
                variant, _, method = key.partition("__")
                row = {"tariff": tariff, "variant": variant, "method": method,
                       "ident": f.split(".")[0]}
                dqn = hist.get("dqn")
                if dqn and dqn.get("val_cost_closed"):
                    v = np.asarray(dqn["val_cost_closed"], dtype=float)
                    # Improvement measured on the RUNNING BEST, not on the raw
                    # trace: validation cost is noisy between evaluations, and
                    # a late unlucky rollout would otherwise read as regress.
                    best = np.minimum.accumulate(v)
                    total = best[0] - best[-1]
                    cut = max(int(len(best) * (1.0 - tail_frac)), 1)
                    tail = best[cut - 1] - best[-1]
                    row.update({
                        "n_evals": len(v),
                        "converged": bool(dqn.get("converged")),
                        "stopped_early": bool(dqn.get("stopped_early")),
                        "steps": dqn.get("steps_run"),
                        "best_frac": float(np.argmin(v) / max(len(v) - 1, 1)),
                        "improvement_tail": (float(tail / total) if total > 1e-9
                                             else 0.0),
                        "best_val": float(best[-1]),
                    })
                bc = hist.get("bc")
                if bc:
                    row["bc_converged"] = bool(bc.get("converged"))
                    row["bc_agreement"] = bc.get("final_val_agreement")
                rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


def convergence_report(tail_frac: float = 0.25):
    """`_convergence_frame`, printed: the answer to "did training finish?"."""
    df = _convergence_frame(tail_frac)
    if df is None:
        print("no training histories on disk yet")
        return None

    learned = df[df["method"].isin(["dqn", "bc_dqn"])]
    print(f"{len(learned)} reinforcement run(s)")
    if len(learned):
        print(f"  converged by the early-stop rule : "
              f"{int(learned['converged'].fillna(False).astype(bool).sum())}"
              f"/{len(learned)}")
        print(f"  best eval is the LAST one (truncated): "
              f"{int((learned['best_frac'].fillna(1.0) > 0.99).sum())}")
        print("\n  share of total improvement arriving in the final quarter")
        print("  (near 0 = settled; large = still descending at the buzzer):")
        print(learned.groupby(["tariff", "method"])["improvement_tail"]
              .describe()[["mean", "50%", "max"]].round(3).to_string())
        # `.fillna(False).astype(bool)` before the `~`, and it is load-bearing:
        # the column is object dtype, because the clone-only rows never set it,
        # and `~` on an object Series applies PYTHON's bitwise not elementwise
        # -- where ~True is -2 and ~False is -1, both truthy. Inverting it
        # directly selected all 140 runs, converged ones included.
        conv = learned["converged"].fillna(False).astype(bool)
        tail = learned["improvement_tail"].fillna(1.0)
        stragglers = learned[(~conv) | (tail > 0.25)]
        if len(stragglers):
            print(f"\n  {len(stragglers)} run(s) to re-train with a larger "
                  f"budget (--steps):")
            print(stragglers[["tariff", "variant", "method", "ident",
                              "converged", "improvement_tail"]]
                  .sort_values("improvement_tail", ascending=False)
                  .head(20).round(3).to_string(index=False))
    clones = df[df["bc_agreement"].notna()] if "bc_agreement" in df else df.iloc[:0]
    if len(clones):
        print(f"\n{len(clones)} clone(s): "
              f"{int(clones['bc_converged'].sum())} early-stopped on held-out "
              f"loss; agreement with the MILP teacher "
              f"{clones['bc_agreement'].mean():.3f} mean, "
              f"{clones['bc_agreement'].min():.3f} worst")
    return df


if __name__ == "__main__":
    main()
