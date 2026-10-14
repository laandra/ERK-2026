"""The learned controllers under each SI contract type, each with its own reward.

    python3 run_rl_contract.py refs    --jobs 12          # rules + MILP per contract
    python3 run_rl_contract.py train   --jobs 12 --quiet  # specialists, every type
    python3 run_rl_contract.py retrain --jobs 12 --quiet  # unsettled runs, at 1M steps
    python3 run_rl_contract.py cross   --jobs 12          # every model on every type
    python3 run_rl_contract.py tune    --jobs 12 --quiet  # reward settings, validation only
    python3 run_rl_contract.py report

Three questions, on the 30 study households and the panel's `fc_h24`
observation contract:

    does paying for the contract help?    On the study's own SI contract
        (`monthly`), the panel's learners -- trained on the settled bill, which
        cannot see the standing charge -- against the same learners trained
        with `rl_reward.SIMonthlyContractReward`.
    does each contract need its own reward?   "The SI contract" is two
        choices. The PRICE LIST: dynamic (the study's), active four-tariff, or
        single-tariff (`rl_reward.PRICE_TYPES`). And how the AGREED POWER is
        set: a signed number, re-agreed monthly (the study's), or rolled over
        twelve months (`rl_reward.CONTRACT_TYPES`). Each gets the reward that
        is its bill.
    does each contract need its own MODEL?    Every model is scored on the test
        year of every contract type of its family (`cross`): price lists
        against price lists, agreed-power rules against agreed-power rules. A
        specialist that does no better on its own contract than a model
        trained on another one was not worth training.

Resumable like `run_rl_benchmark.py` and for the same reason: one JSON per
(contract, arm, method, household) under `results_local/rl_contract/`, keyed
by a config digest; rerun any phase until it prints ALL DONE. Everything the
panel fixes is read from `run_rl_benchmark` -- the battery, the split, the
TUNED hyperparameters, the teacher span -- so a number here differs from the
panel's in the contract and the reward and in nothing else.

The test year is scored ONCE per model, by `Rule_Based_Control.run_policy`
with the arm's settlement and the contract's own fixed point. Reward settings
are chosen by `tune`, which never evaluates it.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hems_study as hs                                          # noqa: E402
import rl_control as rl                                          # noqa: E402
import rl_reward as rr                                           # noqa: E402
import run_rl_benchmark as rb                                    # noqa: E402
import Rule_Based_Control as rbc                                 # noqa: E402
from Basic_Functions import max_charge_now, max_discharge_now    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_local", "rl_contract")
MODELS = os.path.join(rb.MODELS, "contract")
TUNE_OUT = os.path.join(OUT, "_tune")
TUNE_MODELS = os.path.join(MODELS, "_tune")

TARIFF = "SI"
VARIANT = "fc_h24"                  # the panel's reference observation contract
METHODS = ["bc", "bc_dqn", "dqn"]
H, N_TRAIN, N_SIM = rb.H, rb.N_TRAIN, rb.N_SIM

# An ARM is a reward plus the features it needs.
#
#   excess     energy + excess power (`SIExcessPowerReward`), the panel's
#              peak features. On `fixed` this IS the bill. On `monthly` and
#              `annual` it is the contract-blind control -- and on `monthly`
#              it is the panel itself, whose stored models are re-scored here
#              rather than re-trained.
#   contract   + the contract charge (`SI{Monthly,Annual}ContractReward`),
#              the contract re-agreed from the agent's own peaks during
#              training, contract features, contract-aware validation.
#              Undefined on `fixed`, where there is no charge to add.
ARMS = {"excess": {"contract_features": False},
        "contract": {"contract_features": True}}
# The reward each contract type is MEANT to be trained with -- its bill.
SPECIALIST = {"fixed": "excess", "monthly": "contract", "annual": "contract",
              "active": "contract", "single": "contract"}

# What the agreed-power feature is scaled by, kW, per contract type.
#
# Every feature is scaled by its own spread over the training blocks. On a
# SIGNED contract the agreed power has none: it is one number for two years
# (per block, and for most households the same number in every block), and the
# test year is billed on ANOTHER number -- the operator's figure moves with the
# metering year it is read from. Scaled by a spread of zero, a 0.6 kW change in
# the signed line reached the network as an input of -600,000, and the first
# fixed-contract panel showed it: 127's clone, which saves 26 EUR/a over the
# training years, ran the whole test year without moving the battery (0.0 EFC),
# and the 30-household mean came out 14 EUR/a BELOW self-consumption.
#
# A floor of half a kW was tried next and is not enough for that contract: the
# clone came back level with self-consumption on the mean, but its two worst
# households were exactly the two whose signed line moved most between the
# training years and the test year (Ausgrid 180, 3.6 -> 5.4 kW: -58 EUR/a
# against self-consumption; 142, 6.2 -> 5.2 kW: -25). A value the network saw
# ONE of in training is not a feature it can be asked to extrapolate. So on the
# signed contract the line itself is taken out (an infinite scale: the input
# is 0 whatever the line is) and the agent reads the contract through what is
# measured AGAINST it and does vary -- the headroom to the line and the running
# peak relative to it.
#
# The rolling twelve-month contract has the first disease in a mild form: the
# line moves a little once a month, its spread over the training blocks is
# under half a kW on 17 of the 30 households (0.15 at the least), and the test
# year's line sits up to seven of those spreads away. There the half-kW floor
# is the remedy: the line still carries information, it is only over-scaled.
#
# Half a kilowatt is the order of the spread the feature has on the monthly
# contract (0.43-1.6 kW over the same households), which is therefore left on
# its own spread -- as the panel's models, and the settings chosen by `tune`,
# were trained.
AGREED_KW_SCALE = {"fixed": float("inf"), "annual": 0.5, "monthly": None,
                   "active": None, "single": None}

# The least the PRICE features are scaled by, EUR/kWh, per contract type.
#
# The same trap one feature over. On the single-tariff list the export credit
# is one number and the import rate moves only with the network's time blocks
# (0.144-0.156 EUR/kWh on Ausgrid 138); on the active list both take a dozen
# posted values. Scaled by their own spread, a model trained there and then
# shown a list whose prices MOVE -- which is exactly what the cross-evaluation
# asks of it -- would read a 10-cent market swing as an input of tens of
# thousands, and "a single-tariff model is useless on a dynamic list" would be
# a statement about a divide by nearly zero. One cent is the order of the
# hour-to-hour spread on the dynamic list. The study's own list is left on its
# own spread, as the panel's models were trained.
PRICE_SCALE = {"fixed": None, "annual": None, "monthly": None,
               "active": 0.01, "single": 0.01}


def arms_for(contract: str) -> list:
    return ["excess"] if contract == "fixed" else ["excess", "contract"]


def spec_for(arm: str, base: rl.FeatureSpec | None = None) -> rl.FeatureSpec:
    """The arm's observation: the panel's `fc_h24`, or `base` -- a wider
    contract of the feature study (`run_rl_features`) -- with the arm's own
    contract features switched on or off."""
    if base is None:
        return rl.FeatureSpec(horizon=48, peak_features=True,
                              contract_features=ARMS[arm]["contract_features"])
    return dataclasses.replace(
        base, contract_features=ARMS[arm]["contract_features"])


def study_ids() -> list:
    return [int(str(i).split()[-1]) for i in hs.study_units().index]


# ---------------------------------------------------------------------------
# One household under one contract type
# ---------------------------------------------------------------------------
def prepare(ident, contract: str) -> dict:
    """`rb.prepare_household` with the bundles re-built on `contract`.

    `monthly` is the panel's household unchanged. For every other type the
    frames, forecasts and split are the panel's and only the environment's
    contract differs (`rb.build_bundle(contract=...)`): the supplier's price
    list for `active` / `single`, the agreed-power rule for `fixed` / `annual`.

    The signed (`fixed`) contract is the operator's figure from the metering
    year BEFORE the one it bills: the test year is billed on the last training
    year's no-battery peaks, so nothing of the scored year reaches the agent
    through the line it is shown. The training period has no year before it
    and is billed on its own first year, the one place the figure is read off
    data it also bills.
    """
    prep = rb.prepare_household(ident, TARIFF)
    prep["contract"] = contract
    prep["contract_kwargs"] = {"train": None, "sim": None}
    if contract == "monthly":
        return prep
    frames = prep["frames"]
    if contract == "fixed":
        sig = prep["train"]["sig"]
        signed = {"train": rr.operator_contract(sig, 0, 365 * H),
                  "sim": rr.operator_contract(sig, (N_TRAIN - 365) * H,
                                              N_TRAIN * H)}
        prep["signed_kw"] = signed
        kw = {k: rr.contract_env_kwargs("fixed", v) for k, v in signed.items()}
    else:
        kw = {k: rr.contract_env_kwargs(contract) for k in ("train", "sim")}
    prep["contract_kwargs"] = kw
    prep["train"] = rb.build_bundle(
        TARIFF, frames["df_all_kwh"].iloc[: N_TRAIN * H], frames["df_train"],
        N_TRAIN * H, contract=kw["train"])
    prep["sim"] = rb.build_bundle(
        TARIFF, frames["df_ctrl_kwh"], frames["df_ctrl"], N_SIM * H,
        contract=kw["sim"])
    return prep


def teacher(prep) -> np.ndarray:
    """The whole-period MILP under THIS contract type, cached per type."""
    if prep["contract"] == "monthly":
        return rb.teacher_setpoints(prep)         # the panel's own cache
    return rb.teacher_setpoints(prep, contract=prep["contract_kwargs"]["train"],
                                cache_tag=prep["contract"])


def _teacher_import_kw(bundle, setpoints) -> np.ndarray:
    """The grid draw the teacher's plan presents to the meter, in kW."""
    sig, env = bundle["sig"], bundle["env"]
    soc = rb.SOC_INIT_USABLE
    out = np.zeros(len(setpoints))
    for idx, target in enumerate(setpoints):
        lo = -max_discharge_now(soc, sig.eta_dis, env.max_discharge_kwh)
        hi = max_charge_now(soc, sig.eta_ch, env.max_charge_kwh,
                            sig.capacity_kwh)
        p = float(np.clip(target, lo, hi))
        ch, dis = max(p, 0.0), max(-p, 0.0)
        out[idx] = max(sig.consumption[idx] + ch - sig.generation[idx] - dis,
                       0.0) / sig.hours
        soc = min(max(soc + ch * sig.eta_ch - dis / sig.eta_dis, 0.0),
                  sig.capacity_kwh)
    return out


# ---------------------------------------------------------------------------
# Scoring: the test year, through the study's own runner
# ---------------------------------------------------------------------------
_LINES = (("energy_eur", "Energy_EUR"), ("power_eur", "Power_EUR"),
          ("fixed_eur", "Fixed_EUR"), ("termadj_eur", "Terminal_SOC_Adj_EUR"),
          ("cost_eur", "Cost_EUR"), ("cost_eur_closed", "Cost_EUR_Closed"),
          ("cost_eur_total", "Cost_EUR_Total"),
          ("efc", "Equivalent_Full_Cycles"),
          ("peak_import_kw", "Peak_Import_kW"),
          ("agreed_power_iters", "Agreed_Power_Iters"),
          ("agreed_power_converged", "Agreed_Power_Converged"))


def _lines(out) -> dict:
    return {k: (bool(out[src]) if k.endswith("converged") else float(out[src]))
            for k, src in _LINES}


def score(bundle, policy, n_steps: int) -> dict:
    """One converged run on `bundle`, as bill lines.

    `cost_eur_total` -- the closed bill plus the standing charge of the
    contract the run walked itself onto -- is the figure this study compares
    on: on a contract the dispatch moves, anything less leaves out the line
    the reward was changed to pay for.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rbc.run_policy(bundle["env"], policy, n_steps=n_steps,
                             settle=bundle["settle"],
                             soc_init_kwh=rb.SOC_INIT_USABLE,
                             rates=bundle["rates"])
    return _lines(out)


def learned_policy(net, fb, window, prep, which: str, name: str = "learned"):
    return rl.LearnedPolicy(
        net, fb, True, name=name, **rb.feature_inputs(prep, fb.spec, which),
        tracker_factory=rr.tracker_factory(
            window if fb.spec.contract_features else None))


# ---------------------------------------------------------------------------
# References: the rules and the MILP under each contract type
# ---------------------------------------------------------------------------
def references(ident, contract: str, verbose: bool = False) -> dict:
    """Every SI rule and the whole-year MILP on the TEST year of one contract.

    The denominators: a learned controller under a contract type is read
    against the rules and the optimum under that SAME contract. For `monthly`
    these are the reference arm's own numbers recomputed (and
    `test_rl_reward` holds them to the arm's checkpoint).
    """
    path = os.path.join(OUT, contract, "refs", f"{ident}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    prep = prepare(ident, contract)
    sim = prep["sim"]
    env, sig, n = sim["env"], sim["sig"], N_SIM * H
    roster = {p.name: p for p in hs.rule_roster(TARIFF)}
    roster[rbc.NO_BATTERY] = rbc._Idle()
    res = {name: score(sim, pol, n) for name, pol in roster.items()}

    env.clear_achieved_power()
    sol = hs.solve_full_period(
        env, sim["rates"], TARIFF, n_steps=n, soc_init_kwh=rb.SOC_INIT_ABS,
        delta_t=rb.DELTA_T, soc_min_kwh=rb.BATTERY_CAP * rb.SOC_MIN,
        verbose=verbose)
    net = (np.asarray(sol["p_buy"]) - np.asarray(sol["p_sell"])) * rb.DELTA_T
    env.clear_achieved_power()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = hs.settle_trajectory(env, net, sim["settle"], sig,
                                 soc_start=rb.SOC_INIT_ABS,
                                 soc_end=float(sol["soc_plan"][-1]))
    discharged = float(np.sum(sol["x_dis"])) * rb.DELTA_T
    res["milp_full"] = {
        "energy_eur": s["Energy_EUR"], "power_eur": s["Power_EUR"],
        "fixed_eur": s["Fixed_EUR"], "termadj_eur": s["Terminal_SOC_Adj_EUR"],
        "cost_eur": s["Cost_EUR"], "cost_eur_closed": s["Cost_EUR_Closed"],
        "cost_eur_total": s["Cost_EUR_Closed"] + s["Fixed_EUR"],
        "efc": discharged / rb.BATTERY_CAP,
        "peak_import_kw": s["Peak_Import_kW"],
        "milp_runtime_s": sol["runtime_s"], "milp_objective": sol["objective"]}
    blob = {"ident": str(ident), "contract": contract, "controllers": res,
            "agreed_power_rule": env.agreed_power_rule,
            "signed_kw": prep.get("signed_kw", {}).get("sim")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(blob, fh, indent=1)
    return blob


# ---------------------------------------------------------------------------
# One (contract, arm, method, household) run
# ---------------------------------------------------------------------------
# Reward and DQN settings an arm may override, e.g. a tuning point. Keys are
# `weight` / `self_consistent` (`rr.ContractOptions`) or TrainConfig fields.
_REWARD_KEYS = ("weight", "self_consistent", "pricing")

# The contract arm's settings, chosen on validation by `tune` (see
# `tune_report`). Empty = the bill as it is (weight 1, the contract type's own
# pricing, the contract re-agreed in training) under the panel's own TUNED
# hyperparameters.
#
# And empty is what the tune of 2026-10-10 chose (monthly contract, 30
# households, held-out weeks only, 300 runs): weights 0.5 and 2 are not
# distinguishable from 1 for either learner (p_holm >= 0.35), pricing by the
# meter's pool is worse by 2.4-3.5 EUR/a (not significant after Holm), and not
# re-agreeing the contract is worse by 5.3 (IL+RL, p_holm 0.0005) and 3.9 EUR/a
# (RL, p_holm 0.02). The exact charge needs no coefficient.
CONTRACT_TUNED: dict = {}


def run_config(contract, arm, method, cfg, overrides=None, base=None):
    """(TrainConfig, BCOptions, ContractOptions, spec) for one run.

    `base` is the observation the arm's contract features are added to
    (`spec_for`); None is the panel's.

    `overrides` are laid OVER the panel's TUNED settings, in the three places
    a key can belong: `weight` / `self_consistent` / `pricing` are the
    reward's (`rr.ContractOptions`), `clone_*` the clone's optimiser
    (`rl.BCOptions`, as in `rb.effective_clone`), anything else a TrainConfig
    field. A search trial (`rl_hpo`) hands over its whole configuration this
    way -- TUNED's entry, the fixed clone and the trial's own values in one
    dict -- and a tuning point just the one or two keys it moves.
    """
    over = dict(CONTRACT_TUNED.get((contract, method), {})
                if overrides is None and arm == "contract" else overrides or {})
    cfg = copy.copy(rb.effective_config(cfg, TARIFF, method))  # the panel's TUNED
    clone = copy.copy(rb.effective_clone(TARIFF, method))
    for k, v in over.items():
        if k in _REWARD_KEYS:
            continue
        if k.startswith(rb.CLONE_PREFIX):
            name = k[len(rb.CLONE_PREFIX):]
            if not hasattr(clone, name):
                raise AttributeError(f"BCOptions has no field {name!r}")
            setattr(clone, name, v)
            continue
        if not hasattr(cfg, k):
            raise AttributeError(f"TrainConfig has no field {k!r}")
        setattr(cfg, k, v)
    options = rr.ContractOptions(**{k: over[k] for k in _REWARD_KEYS if k in over})
    return cfg, clone, options, spec_for(arm, base)


def run_digest(contract, arm, method, cfg, overrides=None, base=None) -> str:
    cfg_e, clone, options, spec = run_config(contract, arm, method, cfg,
                                             overrides, base)
    version = {"bc": (rl.BC_ALGO_VERSION,), "dqn": (rl.ALGO_VERSION,),
               "bc_dqn": (rl.BC_ALGO_VERSION, rl.ALGO_VERSION)}[method]
    reward = ({"reward": "contract", "version": rr.REWARD_VERSION,
               **options.config()} if arm == "contract" else {"reward": "excess"})
    if (contract == "monthly" and arm == "excess" and overrides is None
            and base is None):
        # This cell is the PANEL's model re-scored, so it is as current as the
        # panel is: retrain the panel and this goes stale with it.
        reward["panel"] = rb.run_digest(cfg, rb.variant_specs(TARIFF)[VARIANT],
                                        TARIFF, method)
    if AGREED_KW_SCALE[contract] is not None:
        reward["agreed_kw_scale"] = AGREED_KW_SCALE[contract]
    if PRICE_SCALE[contract] is not None:
        reward["price_scale"] = PRICE_SCALE[contract]
    return rb._digest(spec.config(), rb._method_config(cfg_e, method, clone),
                      reward,
                      {"contract": contract, "arm": arm, "method": method,
                       "algo": version, "train_days": rb.TRAIN_BLOCKS,
                       "val_days": rb.VAL_BLOCKS, "teach_span": rb.TEACH_SPAN})


def _result_path(contract, arm, method, ident, out_root=OUT):
    return os.path.join(out_root, contract, f"{arm}__{method}", f"{ident}.json")


def _model_path(contract, arm, method, ident, models_root=MODELS):
    return os.path.join(models_root, contract, f"{arm}__{method}", f"{ident}.pt")


def panel_model_path(method, ident):
    return os.path.join(rb.MODELS, TARIFF, f"{VARIANT}__{method}", f"{ident}.pt")


def _measure(prep, net, fb, window, contract) -> dict:
    """Validation on the axis every arm of a contract type shares.

    On a contract the dispatch moves: the held-out weeks of the agent's own
    full-period trajectory under its own contract, contract charge included
    (`SIContractChargeReward.held_out`), whatever reward the agent was
    TRAINED with -- so a contract-blind model and a contract-paid one are
    compared on one measure. On a signed contract there is no charge and the
    panel's `validation_rollout` is that measure already.
    """
    train = prep["train"]
    n_val_days = sum(b - a for a, b in rb.VAL_BLOCKS)
    wear = rb.val_wear_fn(TARIFF)
    inputs = rb.feature_inputs(prep, fb.spec, "train")
    if contract == "fixed":
        static = fb.normalize(fb.build_static(train["sig"], **inputs))
        v = rl.validation_rollout(net, fb, static, train["sig"], train["settle"],
                                  train["env"], rb.VAL_BLOCKS,
                                  rb.SOC_INIT_USABLE, True)
        v["contract_eur"] = 0.0
    else:
        cls = rr.reward_class(TARIFF, contract)
        meter = cls(train, fb, None, val_blocks=rb.VAL_BLOCKS,
                    soc_init=rb.SOC_INIT_USABLE, respect_peak=True, **inputs)
        # The pool the MODEL's features read is the model's own, not the
        # contract's it is being measured under.
        meter.tracker_factory = lambda: rr.tracker_factory(
            window if fb.spec.contract_features else None)
        v = meter.settle_contract(net, passes=3)
    return {"val_cost": v["cost_eur_closed"] + wear(v["efc"], n_val_days),
            "val_bill": v["cost_eur"], "val_contract": v["contract_eur"],
            "val_efc": v["efc"]}


def run_one(prep, arm: str, method: str, cfg, verbose: bool = True,
            out_root: str = OUT, models_root: str = MODELS,
            score_test: bool = True, overrides: dict | None = None,
            base: rl.FeatureSpec | None = None) -> dict:
    """Train one (arm, method) on `prep`'s contract type; score the test year.

    `base` trains the arm under a wider observation (`spec_for`); the prep
    must then carry what that observation reads (`rb.feature_inputs`).

    `monthly`/`excess` is the panel: its stored model is loaded and re-scored
    (and checked against the panel's own stored bill) instead of being trained
    a second time under a second name.
    """
    ident, contract = prep["ident"], prep["contract"]
    path = _result_path(contract, arm, method, ident, out_root)
    digest = run_digest(contract, arm, method, cfg, overrides, base)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("digest") == digest:
            return existing
        print(f"  [stale] {contract}/{arm}__{method}/{ident}: recomputing")

    t0 = time.time()
    cfg_e, clone, options, spec = run_config(contract, arm, method, cfg,
                                             overrides, base)
    train = prep["train"]
    train["env"].clear_achieved_power()
    window = rr.CONTRACT_WINDOW[contract] if arm == "contract" else None
    panel = (contract == "monthly" and arm == "excess" and overrides is None
             and base is None)
    hist_bc = hist_dqn = None

    if panel:
        net, fb, spec, _ = rl.load_model(panel_model_path(method, ident))
        model_path = panel_model_path(method, ident)
        with open(os.path.join(rb.OUT, TARIFF, f"{VARIANT}__{method}",
                               f"{ident}.json"), encoding="utf-8") as fh:
            stored = json.load(fh)
    else:
        fb = rl.FeatureBuilder(spec)
        inputs = rb.feature_inputs(prep, spec, "train")
        static = fb.build_static(train["sig"], **inputs)
        # Normalisation from the TRAINING blocks only, as in the panel.
        rows = np.concatenate([np.arange(a * H, b * H) for a, b in rb.TRAIN_BLOCKS])
        fb.fit_norm(static[rows])
        if AGREED_KW_SCALE[contract] is not None:
            col = fb.agreed_kw_column
            fb.norm_std[col] = max(float(fb.norm_std[col]),
                                   AGREED_KW_SCALE[contract])
        if PRICE_SCALE[contract] is not None:
            for col in fb.price_columns:
                fb.norm_std[col] = max(float(fb.norm_std[col]),
                                       PRICE_SCALE[contract])
        common = dict(val_blocks=rb.VAL_BLOCKS, soc_init=rb.SOC_INIT_USABLE,
                      respect_peak=True, **inputs)
        if arm == "contract":
            reward = rr.reward_class(TARIFF, contract)(
                train, fb, fb.normalize(static), options=options, **common)
        else:
            reward = rr.SIExcessPowerReward(train, fb, fb.normalize(static),
                                            **common)

        if method in ("bc", "bc_dqn"):
            setpoints = teacher(prep)
            if arm == "contract" and options.self_consistent:
                # The clone is shown the TEACHER's contract, not the
                # no-battery one: the line the optimum's own peaks agreed is
                # the state it was responding to, and the nearest thing in
                # training to the low contract a good clone is scored under.
                reward.register({"_import_kw": _teacher_import_kw(train, setpoints)})
            a, b = rb.TEACH_SPAN
            net, hist_bc = rl.train_bc(
                reward.sig, train["settle"], train["env"], fb,
                reward.static_norm, setpoints, start=a * H, stop=b * H,
                soc_init=rb.SOC_INIT_USABLE, respect_peak=True, cfg=cfg_e,
                verbose=verbose, holdout_days=rb.VAL_BLOCKS, opts=clone,
                reward_fn=reward)
        if method in ("dqn", "bc_dqn"):
            init = net if method == "bc_dqn" else None
            if arm == "contract":
                train["env"].clear_achieved_power()
                reward.sig = rbc.rebind_agreed_power(reward.sig, train["env"])
                reward.profile = reward.naive
                reward._load_static()
                if init is not None and options.self_consistent:
                    # A fine-tune starts under the contract its clone earns.
                    reward.settle_contract(init, passes=2)
            net, hist_dqn = rl.train_dqn(
                reward.sig, train["settle"], train["env"], fb,
                reward.static_norm, cfg_e, respect_peak=True,
                train_days=rb.TRAIN_BLOCKS, val_days=rb.VAL_BLOCKS,
                init_net=init, bc_net=init, soc_target=rb.SOC_INIT_USABLE,
                verbose=verbose, val_wear=rb.val_wear_fn(TARIFF),
                reward_fn=reward)
        model_path = _model_path(contract, arm, method, ident, models_root)
        rl.save_model(model_path, net, fb, spec, cfg_e,
                      {"bc": hist_bc, "dqn": hist_dqn},
                      extra={"ident": str(ident), "tariff": TARIFF,
                             "contract": contract, "arm": arm,
                             "method": method, "window": window})

    hist = hist_dqn or hist_bc or {}
    result = {
        "dataset": f"Ausgrid {ident}", "ident": str(ident), "tariff": TARIFF,
        "contract": contract, "arm": arm, "method": method, "digest": digest,
        "window": window, "spec": spec.config(),
        "reward": ("panel" if panel else reward.config()),
        "train_converged": (stored.get("train_converged") if panel
                            else bool(hist.get("converged", False))),
        "train_steps": (stored if panel else hist).get(
            "train_steps" if panel else "steps_run"),
        "train_runtime_s": (stored if panel else hist).get(
            "train_runtime_s" if panel else "runtime_s"),
        "bc_val_agreement": (stored.get("bc_val_agreement") if panel
                             else (hist_bc or {}).get("final_val_agreement")),
        "train_config": cfg_e.config(),
        "model_path": os.path.relpath(model_path, HERE),
    }
    result.update(_measure(prep, net, fb, window, contract))
    # The whole training period through the scoring runner: not held out, but
    # the real bill with the real fixed point, on data the test never sees.
    tr = score(train, learned_policy(net, fb, window, prep, "train"), N_TRAIN * H)
    result.update({f"train_{k}": v for k, v in tr.items()})
    if score_test:
        te = score(prep["sim"], learned_policy(net, fb, window, prep, "sim"),
                   N_SIM * H)
        result.update(te)
        if panel:
            gap = abs(stored["cost_eur_closed"] - te["cost_eur_closed"])
            if gap > 1e-6:
                raise RuntimeError(
                    f"re-scoring the panel's {method} model for Ausgrid {ident} "
                    f"gave a bill {gap:.4f} EUR off the panel's own: these are "
                    f"not the models the panel scored")
    result["wall_s"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    return result


# ---------------------------------------------------------------------------
# Cross-evaluation: every model on every contract type
# ---------------------------------------------------------------------------
def cross_one(prep, trained_on: str, arm: str, method: str) -> dict:
    """The model trained on `trained_on` scored on `prep`'s contract type."""
    ident, tested_on = prep["ident"], prep["contract"]
    src = _result_path(trained_on, arm, method, ident)
    with open(src, encoding="utf-8") as fh:
        source = json.load(fh)
    path = os.path.join(OUT, "_cross", tested_on,
                        f"{trained_on}__{arm}__{method}", f"{ident}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("source_digest") == source["digest"]:
            return existing
    if trained_on == tested_on:
        res = {k: source[k] for k, _ in _LINES}
    else:
        net, fb, _, blob = rl.load_model(os.path.join(HERE, source["model_path"]))
        res = score(prep["sim"],
                    learned_policy(net, fb, source["window"], prep, "sim"),
                    N_SIM * H)
    res.update({"ident": str(ident), "tested_on": tested_on,
                "trained_on": trained_on, "arm": arm, "method": method,
                "source_digest": source["digest"]})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
def _train_worker(group, cfg, quiet, tune_root=None):
    contract, ident, items = group
    out = []
    try:
        prep = prepare(ident, contract)
    except Exception as exc:
        return [(contract, a, m, ident, None, repr(exc)) for a, m, _ in items]
    for arm, method, over in items:
        try:
            if tune_root is None:
                res = run_one(prep, arm, method, cfg, verbose=not quiet)
            else:
                tag = tune_tag(over)
                res = run_one(prep, arm, method, cfg, verbose=not quiet,
                              out_root=os.path.join(TUNE_OUT, tag),
                              models_root=os.path.join(TUNE_MODELS, tag),
                              score_test=False, overrides=over)
                assert "cost_eur_total" not in res, "tuning touched the test year"
            out.append((contract, arm, method, ident, res["val_cost"], None))
        except Exception as exc:                  # one run must not sink the sweep
            import traceback
            out.append((contract, arm, method, ident, None,
                        repr(exc) + "\n" + traceback.format_exc(limit=4)))
    return out


def _refs_worker(ident, contract):
    try:
        references(ident, contract)
        return (contract, ident, None)
    except Exception as exc:
        return (contract, ident, repr(exc))


def _cross_worker(ident, tested_on, items):
    out = []
    try:
        prep = prepare(ident, tested_on)
    except Exception as exc:
        return [(tested_on, t, a, m, ident, repr(exc)) for t, a, m in items]
    for trained_on, arm, method in items:
        try:
            cross_one(prep, trained_on, arm, method)
            out.append((tested_on, trained_on, arm, method, ident, None))
        except Exception as exc:
            out.append((tested_on, trained_on, arm, method, ident, repr(exc)))
    return out


def _parallel(fn, tasks, n_jobs, label):
    from joblib import Parallel, delayed
    if not tasks:
        print(f"{label}: nothing to do")
        return []
    n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs) if n_jobs < 0 else n_jobs
    n_jobs = min(n_jobs, len(tasks))
    print(f"{label}: {len(tasks)} group(s) on {n_jobs} worker(s)", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(fn)(*t) for t in tasks)
    print(f"{label}: {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def _report_failures(flat, err_index) -> bool:
    failed = [r for r in flat if r[err_index] is not None]
    for r in failed:
        print("  FAILED", *r[:err_index], "\n   ", r[err_index])
    if failed:
        print(f"{len(failed)} failure(s); rerun to retry, the rest is cached")
    return bool(failed)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def phase_refs(ids, contracts, n_jobs):
    tasks = [(i, c) for c in contracts for i in ids
             if not os.path.exists(os.path.join(OUT, c, "refs", f"{i}.json"))]
    res = _parallel(_refs_worker, tasks, n_jobs, "refs")
    if not _report_failures(res, 2):
        print("ALL DONE")


# The step budget a straggler is re-trained at (`retrain`), as in the panel.
RETRAIN_STEPS = 1_000_000


def _current(contract, arm, method, ident, cfg) -> bool:
    """Is the result on disk this configuration's, at either budget?

    A run whose training never settled is re-trained at `RETRAIN_STEPS`
    (`phase_retrain`) and then carries that budget's digest; it is as current
    as its neighbours at the default budget, and a resumed sweep must not
    throw it away to train the unsettled 500k run again.
    """
    p = _result_path(contract, arm, method, ident)
    if not os.path.exists(p):
        return False
    with open(p, encoding="utf-8") as fh:
        digest = json.load(fh).get("digest")
    longer = copy.copy(cfg)
    longer.total_steps = RETRAIN_STEPS
    return digest in (run_digest(contract, arm, method, cfg),
                      run_digest(contract, arm, method, longer))


def phase_train(ids, contracts, methods, arms, cfg, n_jobs, quiet):
    todo, cached = {}, 0
    for contract in contracts:
        for ident in ids:
            for arm in arms_for(contract):
                if arms and arm not in arms:
                    continue
                for method in methods:
                    if _current(contract, arm, method, ident, cfg):
                        cached += 1
                        continue
                    todo.setdefault((contract, ident), []).append(
                        (arm, method, None))
    print(f"train: {sum(len(v) for v in todo.values())} run(s) to do, "
          f"{cached} cached")
    tasks = [((c, i, items), cfg, quiet) for (c, i), items in todo.items()]
    res = _parallel(_train_worker, tasks, n_jobs, "train")
    if not _report_failures([r for g in res for r in g], 5):
        print("ALL DONE")


def phase_retrain(ids, contracts, cfg, n_jobs, quiet):
    """Re-train the reinforcement runs whose training never settled.

    The panel's own protocol (`run_rl_benchmark --retrain-unconverged`): a run
    the early stop never fired on was still descending when the budget ended,
    so its number is a floor rather than an estimate. Those -- and only those
    -- are trained again at `RETRAIN_STEPS`; the result file records the
    budget it was produced under. The panel's own models (`monthly`/`excess`)
    went through this already and are not touched.
    """
    longer = copy.copy(cfg)
    longer.total_steps = RETRAIN_STEPS
    todo = {}
    for contract in contracts:
        for ident in ids:
            for arm in arms_for(contract):
                if contract == "monthly" and arm == "excess":
                    continue
                for method in ("dqn", "bc_dqn"):
                    p = _result_path(contract, arm, method, ident)
                    if not os.path.exists(p):
                        continue
                    with open(p, encoding="utf-8") as fh:
                        r = json.load(fh)
                    if (not r.get("train_converged")
                            and r["train_config"]["total_steps"] < RETRAIN_STEPS):
                        todo.setdefault((contract, ident), []).append(
                            (arm, method, None))
    print(f"retrain: {sum(len(v) for v in todo.values())} unsettled run(s) at "
          f"{RETRAIN_STEPS:,} steps")
    tasks = [((c, i, items), longer, quiet) for (c, i), items in todo.items()]
    res = _parallel(_train_worker, tasks, n_jobs, "retrain")
    if not _report_failures([r for g in res for r in g], 5):
        print("ALL DONE")


def phase_cross(ids, contracts, methods, n_jobs):
    tasks = []
    for tested_on in contracts:
        for ident in ids:
            items = []
            for trained_on in contracts:
                # Only within a family: a model is a candidate for another
                # price list, or for another agreed-power rule -- not for a
                # contract that differs from its own in both.
                if not any(trained_on in fam for fam in rr.family(tested_on)):
                    continue
                for arm in arms_for(trained_on):
                    for method in methods:
                        if not os.path.exists(
                                _result_path(trained_on, arm, method, ident)):
                            continue
                        items.append((trained_on, arm, method))
            if items:
                tasks.append((ident, tested_on, items))
    res = _parallel(_cross_worker, tasks, n_jobs, "cross")
    if not _report_failures([r for g in res for r in g], 5):
        print("ALL DONE")


# ---------------------------------------------------------------------------
# Tuning the contract reward: validation only
# ---------------------------------------------------------------------------
# What the contract charge leaves open, each asked as one departure from the
# bill as it is (`weight` 1, the contract type's own pricing, the contract
# re-agreed during training):
#
#   weight           is the exact charge also the best TRAINING signal? Under-
#                    and over-paid by a factor of two.
#   pricing          the monthly contract priced by the meter's pool instead of
#                    the episode's own ("ratchet") -- which pool the agent is
#                    held responsible for.
#   self_consistent  the ablation: the charge without the re-agreed contract.
#
# Chosen on the held-out contract validation, never on the test year, and
# written into CONTRACT_TUNED by hand.
_TUNE_POINTS = [{"weight": 1.0}, {"weight": 0.5}, {"weight": 2.0},
                {"weight": 1.0, "pricing": "ratchet"},
                {"weight": 1.0, "self_consistent": False}]
TUNE_GRID = {"dqn": _TUNE_POINTS, "bc_dqn": _TUNE_POINTS}


def tune_tag(over: dict) -> str:
    def _v(v):
        if isinstance(v, bool):
            return str(int(v))
        return v if isinstance(v, str) else f"{v:g}"
    return "_".join(f"{k}{_v(v)}" for k, v in sorted(over.items()))


def phase_tune(ids, contracts, cfg, n_jobs, quiet):
    todo, cached = {}, 0
    for contract in contracts:
        if contract == "fixed":
            continue
        for ident in ids:
            for method, grid in TUNE_GRID.items():
                for over in grid:
                    p = _result_path(contract, "contract", method, ident,
                                     os.path.join(TUNE_OUT, tune_tag(over)))
                    if os.path.exists(p):
                        with open(p, encoding="utf-8") as fh:
                            if json.load(fh).get("digest") == run_digest(
                                    contract, "contract", method, cfg, over):
                                cached += 1
                                continue
                    todo.setdefault((contract, ident), []).append(
                        ("contract", method, over))
    print(f"tune: {sum(len(v) for v in todo.values())} run(s) to do, "
          f"{cached} cached")
    tasks = [((c, i, items), cfg, quiet, True) for (c, i), items in todo.items()]
    res = _parallel(_train_worker, tasks, n_jobs, "tune")
    if not _report_failures([r for g in res for r in g], 5):
        print("ALL DONE")
        tune_report()


def tune_frame():
    import pandas as pd
    rows = []
    if os.path.isdir(TUNE_OUT):
        for tag in sorted(os.listdir(TUNE_OUT)):
            for contract in rr.ALL_TYPES:
                base = os.path.join(TUNE_OUT, tag, contract)
                if not os.path.isdir(base):
                    continue
                for key in sorted(os.listdir(base)):
                    for f in sorted(os.listdir(os.path.join(base, key))):
                        if f.endswith(".json"):
                            with open(os.path.join(base, key, f),
                                      encoding="utf-8") as fh:
                                r = json.load(fh)
                            rows.append({"tag": tag, **{k: r.get(k) for k in (
                                "contract", "method", "ident", "val_cost",
                                "val_bill", "val_contract", "val_efc",
                                "train_cost_eur_total", "train_efc",
                                "train_converged")}})
    return pd.DataFrame(rows)


def tune_report():
    """Held-out contract validation per grid point; the winner per cell."""
    from scipy.stats import wilcoxon
    df = tune_frame()
    if not len(df):
        print("no tuning results on disk")
        return None
    for (contract, method), g in df.groupby(["contract", "method"]):
        wide = g.pivot(index="ident", columns="tag", values="val_cost").dropna()
        summ = (g[g["ident"].isin(wide.index)].groupby("tag")
                .agg(n=("ident", "size"), val_cost=("val_cost", "mean"),
                     val_contract=("val_contract", "mean"),
                     efc=("val_efc", "mean"),
                     train_total=("train_cost_eur_total", "mean"))
                .sort_values("val_cost"))
        best = summ.index[0]
        p = {}
        for tag in summ.index[1:]:
            d = wide[tag] - wide[best]
            p[tag] = wilcoxon(d).pvalue if (d != 0).sum() >= 6 else float("nan")
        summ["vs_best_p"] = p
        print(f"\n{contract} {method} (mean over {len(wide)} households; "
              f"val = held-out weeks, EUR)")
        print(summ.round(3).to_string())
    return df


# ---------------------------------------------------------------------------
# Frames for the notebook
# ---------------------------------------------------------------------------
def load_refs():
    """One row per (contract, household, controller): the references' lines."""
    import pandas as pd
    rows = []
    for contract in rr.ALL_TYPES:
        d = os.path.join(OUT, contract, "refs")
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    blob = json.load(fh)
                for name, v in blob["controllers"].items():
                    rows.append({"contract": contract, "ident": blob["ident"],
                                 "controller": name, **v})
    return pd.DataFrame(rows)


def load_results():
    """One row per trained run, with its saving against no battery."""
    import pandas as pd
    rows = []
    for contract in rr.ALL_TYPES:
        base = os.path.join(OUT, contract)
        if not os.path.isdir(base):
            continue
        for key in sorted(os.listdir(base)):
            if key == "refs":
                continue
            for f in sorted(os.listdir(os.path.join(base, key))):
                if f.endswith(".json"):
                    with open(os.path.join(base, key, f), encoding="utf-8") as fh:
                        r = json.load(fh)
                    r.pop("train_config", None)
                    r.pop("spec", None)
                    rows.append(r)
    return _with_savings(pd.DataFrame(rows), "contract")


def load_cross():
    """One row per (tested_on, trained_on, arm, method, household)."""
    import pandas as pd
    rows = []
    base = os.path.join(OUT, "_cross")
    if os.path.isdir(base):
        for tested in sorted(os.listdir(base)):
            for key in sorted(os.listdir(os.path.join(base, tested))):
                for f in sorted(os.listdir(os.path.join(base, tested, key))):
                    if f.endswith(".json"):
                        with open(os.path.join(base, tested, key, f),
                                  encoding="utf-8") as fh:
                            rows.append(json.load(fh))
    return _with_savings(pd.DataFrame(rows), "tested_on")


def _with_savings(df, contract_col):
    """Each bill line as a saving against the SAME contract's no-battery row."""
    if not len(df):
        return df
    refs = load_refs()
    nb = (refs[refs["controller"] == rbc.NO_BATTERY]
          .set_index(["contract", "ident"]))
    idx = list(zip(df[contract_col], df["ident"].astype(str)))
    for line in ("energy_eur", "power_eur", "fixed_eur", "cost_eur_closed",
                 "cost_eur_total"):
        if line in df:
            base = nb[line].reindex(idx).to_numpy()
            df["saving_" + line.replace("_eur", "").replace("cost_", "")] = (
                base - df[line].to_numpy())
    return df


SAVING_LINES = {"energy": "energy_eur", "power": "power_eur",
                "standing": "fixed_eur", "total": "cost_eur_total"}


def ranking_frame():
    """One row per (contract, household, controller), savings by bill line.

    Rules and the whole-year MILP from the references, learned controllers as
    `<method>:<arm>`; every saving is against the SAME contract's no-battery
    row for the same household. `standing` (the standing charge, which scales
    with the agreed power) is zero by construction on the signed contract and
    is the line the contract charge exists to move on the other two.
    `closeout` is the terminal-SOC adjustment, so the four lines sum to
    `total`. (`contract` is the contract TYPE the row was billed under.)
    """
    import pandas as pd
    refs = load_refs()
    if not len(refs):
        return refs
    refs = refs.assign(kind="reference")
    res = load_results()
    if len(res):
        res = res.assign(controller=res["method"] + ":" + res["arm"],
                         kind="learned")
        refs = pd.concat([refs, res[[c for c in refs.columns if c in res]]],
                         ignore_index=True)
    nb = (refs[refs["controller"] == rbc.NO_BATTERY]
          .set_index(["contract", "ident"]))
    idx = list(zip(refs["contract"], refs["ident"].astype(str)))
    out = refs[["contract", "ident", "controller", "kind", "efc"]].copy()
    for name, col in SAVING_LINES.items():
        out[name] = nb[col].reindex(idx).to_numpy() - refs[col].to_numpy()
    out["closeout"] = out["total"] - out[["energy", "power", "standing"]].sum(axis=1)
    return out


def report():
    import pandas as pd
    refs, res = load_refs(), load_results()
    if not len(refs):
        print("no references yet")
        return
    nb = refs[refs.controller == rbc.NO_BATTERY].set_index(["contract", "ident"])
    refs = refs.join(nb["cost_eur_total"].rename("nb"), on=["contract", "ident"])
    refs["saving"] = refs["nb"] - refs["cost_eur_total"]
    print("mean saving on the total bill, EUR/a (rules and MILP):")
    print(refs.pivot_table(index="controller", columns="contract",
                           values="saving").round(1).to_string())
    if len(res):
        print("\nlearned (arm, method):")
        print(res.pivot_table(index=["arm", "method"], columns="contract",
                              values="saving_total").round(1).to_string())
        print("\n  of which contract line:")
        print(res.pivot_table(index=["arm", "method"], columns="contract",
                              values="saving_fixed").round(1).to_string())
    cross = load_cross()
    if len(cross):
        print("\ncross-evaluation, saving on the total bill (rows: trained on):")
        for method in METHODS:
            sub = cross[cross.method == method]
            print(f"\n  {method}")
            print(sub.pivot_table(index=["trained_on", "arm"],
                                  columns="tested_on", values="saving_total")
                  .round(1).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("phase", choices=["refs", "train", "retrain", "cross", "tune",
                                      "report", "tune-report"])
    ap.add_argument("--ids", nargs="*", type=int, default=None)
    ap.add_argument("--contracts", nargs="*", default=list(rr.ALL_TYPES))
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    ids = args.ids or study_ids()
    cfg = rb.make_config(args.steps, args.seed)
    if args.phase == "refs":
        phase_refs(ids, args.contracts, args.jobs)
    elif args.phase == "train":
        phase_train(ids, args.contracts, args.methods, args.arms, cfg,
                    args.jobs, args.quiet)
    elif args.phase == "retrain":
        phase_retrain(ids, args.contracts, cfg, args.jobs, args.quiet)
    elif args.phase == "cross":
        phase_cross(ids, args.contracts, args.methods, args.jobs)
    elif args.phase == "tune":
        phase_tune(ids, args.contracts, cfg, args.jobs, args.quiet)
    elif args.phase == "tune-report":
        tune_report()
    else:
        report()


if __name__ == "__main__":
    main()
