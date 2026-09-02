"""The invariants the ERK 2026 comparison rests on.

Not a unit-test suite. Every check here is one of the ways this study has
actually been wrong: two batteries, two evaluators, two windows, two starting
states of charge. Each was found by measurement rather than by reading, so each
is pinned here where a re-run says so rather than a figure quietly moving.

    python test_hems_study.py

Runs in about a minute on a short slice; no sweep, no forecast cache.
"""

import sys
import warnings

import numpy as np
import pandas as pd

import hems_study as hs
import Rule_Based_Control as rbc
from Basic_Functions import cumulative_interval_price_series
from MILP_Household import build_household_env

BATTERY = dict(battery_cap=10.0, soc_min_pct=0.10, soc_max_pct=0.80,
               p_max=1.5, eff=0.95)
DELTA_T, H = 0.5, 48
DATA = "../Input data/Ausgrid/Ausgrid 127.csv"

_passed, _failed = [], []


def check(name, ok, detail=""):
    (_passed if ok else _failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")


def load(n_days=10, start="2012-07-01 00:30:00"):
    """A slice in BOTH conventions: kWh per interval, and kW."""
    raw = pd.read_csv(DATA)
    raw.index = pd.to_datetime(raw["Timestamp_UTC"], format="ISO8601")
    cols = ["SMP", "Energy_Generation", "Energy_Consumption"]
    kwh = raw.loc[start:, cols].iloc[:H * n_days].copy()
    kw = kwh.copy()
    kw[["Energy_Generation", "Energy_Consumption"]] /= DELTA_T
    return kwh, kw


def make_env(kwh):
    return hs.align_envelope(
        hs.build_study_env(kwh, delta_t=DELTA_T, H=H, **BATTERY),
        BATTERY["p_max"], BATTERY["eff"], DELTA_T)


def test_one_battery(kwh):
    """A1/A2. The MILP and every rule must drive the SAME battery.

    They did not. MILPScheduler bounded AC power at +-p_max while the rules read
    env.max_charge_kwh, which upstream applies to STORED energy -- +1.579/-1.425
    kW against +-1.500, a 5.3 % larger charge rating and a 5.0 % smaller
    discharge rating, live in every published figure.
    """
    env = make_env(kwh)
    milp = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    rule = rbc.build_signals(env, n_steps=H)
    check("one battery: MILP charge limit == rule charge limit",
          abs(milp.max_ch_kw - rule.max_charge_ac / DELTA_T) < 1e-9,
          f"{milp.max_ch_kw:.4f} vs {rule.max_charge_ac / DELTA_T:.4f} kW")
    check("one battery: MILP discharge limit == rule discharge limit",
          abs(milp.max_dis_kw - rule.max_discharge_ac / DELTA_T) < 1e-9,
          f"{milp.max_dis_kw:.4f} vs {rule.max_discharge_ac / DELTA_T:.4f} kW")

    # An unaligned environment must be refused, not silently scored.
    raw_env = hs.build_study_env(kwh, delta_t=DELTA_T, H=H, **BATTERY)
    try:
        hs.UpstreamMILPScheduler(raw_env, delta_t=DELTA_T, parity=True, **BATTERY)
        check("one battery: unaligned environment is refused", False, "no error raised")
    except ValueError:
        check("one battery: unaligned environment is refused", True)


def test_milp_parity(kwh, kw):
    """A2. parity=True must reproduce the hand-rolled model it replaced.

    This is what makes the previously published numbers reproducible rather than
    merely plausible. If it ever fails, the swap changed the answer.
    """
    env = make_env(kwh)
    old = hs.MILPScheduler(delta_t=DELTA_T, **BATTERY)
    new = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    buy, sell, _ = hs.au_rate_vectors(kwh.index, kwh["SMP"].values, 30)
    gen = kw["Energy_Generation"].tolist()
    con = kw["Energy_Consumption"].tolist()
    worst = 0.0
    for day in range(3):
        a, b = day * H, (day + 1) * H
        ro = old.solve(5.0, buy[a:b], sell[a:b], gen[a:b], con[a:b])
        rn = new.solve(5.0, buy[a:b], sell[a:b], gen[a:b], con[a:b])
        worst = max(worst, abs(ro["cost"] - rn["cost"]))
    check("MILP parity: upstream physics reproduces the old model",
          worst < 1e-6, f"worst cost delta {worst:.2e} EUR over 3 daily solves")


def test_one_evaluator(kwh, kw):
    """The AU settlement must reproduce the controller's own costing exactly.

    Adopting a shared evaluator must move no published number; if it does, the
    change is not "one evaluator" but "a different bill".
    """
    env = make_env(kwh)
    rates = hs.build_rate_vectors("AU", env, kwh.index, kwh["SMP"].values, 30)
    settle = hs.build_settlement("AU", env, rates, DELTA_T)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
    sched = hs.UpstreamMILPScheduler(env, delta_t=DELTA_T, parity=True, **BATTERY)
    ctrl = hs.ReactiveController(
        scheduler=sched, forecaster=None, real_data=kw, soc_init=5.0,
        horizon_steps=H, steps_per_day=H, reoptimize_every=1, freq="30min",
        rate_vectors=rates)
    df = ctrl.run(num_days=len(kwh) // H - 1, use_forecast=False)
    net = (df["Buy_kW"] - df["Sell_kW"]).to_numpy() * DELTA_T
    s = hs.settle_trajectory(env, net, settle, sig)
    own = float(df["Step_Cost"].sum())
    check("one evaluator: AU settlement == controller's own costing",
          abs(own - s["Cost_EUR"]) < 1e-9, f"delta {abs(own - s['Cost_EUR']):.2e} EUR")
    check("one evaluator: AU carries no capacity charge",
          s["Power_EUR"] == 0.0)
    check("one evaluator: the standing charge is recovered, not dropped",
          s["Fixed_EUR"] > 0, f"{s['Fixed_EUR']:.2f} EUR over {len(df)} steps")


def test_si_settlement_matches_upstream(kwh):
    """The SI walk must agree with the shared one for a fixed profile.

    price_interval and Basic_Functions.cumulative_interval_price_series carry the
    same running peak against the same contract; the only difference is that one
    prices a controller's trajectory and the other a fixed profile. On the
    no-battery profile they are the same thing, so they must agree.
    """
    env = make_env(kwh)
    n = len(kwh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        windows = rbc.reset_windows(env, n)
        peak_state = {b: 0.0 for b in rbc._BLOCKS}
        total = 0.0
        for idx in range(n):
            peak_state = rbc._drop_peak_on_window_start(peak_state, windows, idx)
            net = float(env.arr_consumption[idx] - env.arr_generation[idx])
            var, _, _, fixed, peak_state = rbc.price_interval(env, idx, net, peak_state)
            total += var + fixed
        upstream = cumulative_interval_price_series(
            kwh["Energy_Consumption"], kwh["Energy_Generation"], env, kwh)[-1]
    check("SI settlement: agrees with cumulative_interval_price_series",
          abs(total - upstream) < 1e-6, f"{total:.6f} vs {upstream:.6f} EUR")


def test_rules_stay_in_the_envelope(kwh):
    """Every rule is re-clamped by the runner, so drift is 0 by construction."""
    env = make_env(kwh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
        worst_drift, bad_decomp = 0.0, 0.0
        for pol in [rbc._Idle()] + hs.rule_roster("SI"):
            out = rbc.run_policy(env, pol, signals=sig)
            worst_drift = max(worst_drift, out["SOC_Drift_kWh"])
            bad_decomp = max(bad_decomp, abs(
                out["Cost_EUR"] - out["Energy_EUR"] - out["Power_EUR"]))
    check("rules: no policy escapes the battery envelope",
          worst_drift < 1e-9, f"worst SOC drift {worst_drift:.2e} kWh")
    check("rules: Cost_EUR == Energy_EUR + Power_EUR",
          bad_decomp < 1e-6, f"worst residual {bad_decomp:.2e} EUR")


def test_shared_starting_soc(kwh):
    """Every controller must start the year at the same state of charge.

    The rules began at SOC_FRACTION * capacity = 3.5 kWh stored while the MILP
    began at soc_init = 5.0 absolute, which is 4.0 stored. Half a kWh of free
    energy handed to one side of the comparison.
    """
    env = make_env(kwh)
    soc_init, soc_min = 5.0, BATTERY["battery_cap"] * BATTERY["soc_min_pct"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sig = rbc.build_signals(env)
        out = rbc.run_policy(env, rbc._Idle(), signals=sig,
                             soc_init_kwh=soc_init - soc_min)
    check("shared start: an idle pack ends where the MILP started",
          abs(out["Final_SOC_kWh"] + soc_min - soc_init) < 1e-9,
          f"{out['Final_SOC_kWh'] + soc_min:.4f} vs {soc_init:.4f} kWh")


def test_site_selection():
    """F5. The documented rule must be the one that produced the results."""
    published = [138, 127, 65, 148, 223, 179, 261, 142, 128, 168,
                 249, 81, 104, 172, 29, 27, 156, 180, 21, 290,
                 66, 240, 113, 67, 5, 158, 1, 204, 247, 137]
    check("site selection: derived from the clustering == the published list",
          hs.dataset_ids() == published)


def test_wear_cost():
    """F2. The wear rate must be the pack price over the rated cycle life."""
    import Battery_Economics as be
    check("wear: 10 kWh pack prices one full cycle at 0.4167 EUR",
          abs(be.cycle_cost_eur_per_efc(10.0)
              - be.CAPEX_EUR_PER_KWH * 10.0 / be.BATTERY_CYCLE_LIMIT_EFC) < 1e-12,
          f"{be.cycle_cost_eur_per_efc(10.0):.4f} EUR/EFC")


if __name__ == "__main__":
    kwh, kw = load()
    print(f"Ausgrid 127, {len(kwh)} steps ({len(kwh) // H} days)\n")
    test_site_selection()
    test_wear_cost()
    test_one_battery(kwh)
    test_milp_parity(kwh, kw)
    test_one_evaluator(kwh, kw)
    test_si_settlement_matches_upstream(kwh)
    test_rules_stay_in_the_envelope(kwh)
    test_shared_starting_soc(kwh)
    print(f"\n{len(_passed)} passed, {len(_failed)} failed")
    if _failed:
        print("FAILED: " + ", ".join(_failed))
    sys.exit(1 if _failed else 0)
