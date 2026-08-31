"""Validation for the rule-based controllers and the runner that prices them.

In the style of `Battery_Envelope_Check.py` -- a print-driven script with
assertions, run directly rather than through a test runner:

    .venv/bin/python Rule_Based_Control_Check.py            # whole calendar year
    .venv/bin/python Rule_Based_Control_Check.py --quick    # 90 days

What each check is for:

  1  the idle policy is the baseline   the runner reproduces `no_battery_cost`
  2  an empty pack changes nothing     every rule collapses onto that baseline
  3  the envelope holds                no rule can drift the SOC or the inverter
  4  self-consumption stays off-grid   it charges only what the roof gave it
  5  the bill is reproducible          re-pricing a trajectory from the flows
  6  the MILP is a lower bound         no rule may beat perfect foresight
  7  peak shaving shaves               the peak and the excess charge both fall
  8  the signal bundle is re-bindable  only the envelope depends on capacity
  9  foresight is worth something      the forward window beats the trailing one, on average
 10  the daily cycle cap binds         the same budget the MILP is held to
"""

from __future__ import annotations

import sys

import numpy as np

import Horizon_Comparison as hc
import MILP_Household as mh
import Rule_Based_Control as rbc
from Pricing_Functions import calculate_interval_price

DATASET, HOUSEHOLD = "Fluvius_PV", 1
TARIFF = "Dinamični"          # the list that cycles hardest, so every rule moves
CAPACITY_KWH = 10.0
STEPS_PER_DAY = 96

QUICK = "--quick" in sys.argv
N_STEPS = 90 * STEPS_PER_DAY if QUICK else None       # None -> the whole profile


def build(capacity_kwh=CAPACITY_KWH, **envkw):
    """The study environment at one capacity."""
    data = hc.load_user(HOUSEHOLD, DATASET)
    env = hc.build_env(data, capacity_kwh=capacity_kwh, tariff=TARIFF)
    for key, value in envkw.items():
        setattr(env, key, value)
    return env


def steps_for(env):
    return int(env.episode_length if N_STEPS is None else min(N_STEPS, env.episode_length))


# ---------------------------------------------------------------------------
def check_idle_is_the_baseline(env, sig):
    """1. The runner with an idle rule IS `Horizon_Comparison.no_battery_cost`.

    Not approximately: the same intervals, the same evaluator, the same running
    peak. If these two ever part company the whole comparison is measuring the
    plumbing.
    """
    print("\n1. The idle policy reproduces the no-battery baseline")
    n_steps = sig.n_steps
    reference, _ = hc.no_battery_cost(env, n_steps)
    idle = rbc.run_policy(env, rbc._Idle(), signals=sig)["Cost_EUR"]
    difference = abs(idle - reference)
    assert difference < 1e-9, (
        f"the idle runner bills {idle:.9f} where no_battery_cost bills "
        f"{reference:.9f}, a {difference:.2e} EUR difference"
    )
    print(f"   runner {idle:.9f} == no_battery_cost {reference:.9f}  "
          f"({difference:.1e} EUR)  OK")
    return reference


def check_empty_pack(baseline):
    """2. With no battery to command, every rule bills the baseline."""
    print("\n2. An empty pack collapses every rule onto the baseline")
    env = build(capacity_kwh=0.0)
    sig = rbc.build_signals(env, steps_for(env))
    for name in rbc.POLICY_ORDER:
        cost = rbc.run_policy(env, rbc.make_policy(name), signals=sig)["Cost_EUR"]
        assert abs(cost - baseline) < 1e-9, (
            f"{name} bills {cost:.9f} on a 0 kWh pack, baseline is {baseline:.9f}"
        )
    print(f"   all {len(rbc.POLICY_ORDER)} rules bill {baseline:.6f} EUR on 0 kWh  OK")


def check_envelope(env, sig):
    """3. No rule can drift the state of charge or exceed the inverter.

    The runner re-clamps every setpoint, so this is a check that the clamp is
    actually in the path rather than a check on the rules' good behaviour.
    """
    print("\n3. Every rule stays inside the battery envelope")
    capacity = sig.capacity_kwh
    print(f"   {'rule':32}{'max |setpoint|':>16}{'SOC range kWh':>18}{'drift':>10}")
    for name in rbc.POLICY_ORDER:
        out = rbc.run_policy(env, rbc.make_policy(name), signals=sig, keep_traces=True)
        soc, setpoints = out["_soc_trace"], out["_setpoints"]
        ceiling = max(sig.max_charge_ac, sig.max_discharge_ac)
        assert out["SOC_Drift_kWh"] < 1e-12, f"{name} drifted {out['SOC_Drift_kWh']:.2e} kWh"
        assert soc.min() > -1e-9 and soc.max() < capacity + 1e-9, (
            f"{name} left the SOC window: {soc.min():.6f} .. {soc.max():.6f} "
            f"of 0 .. {capacity:.2f}"
        )
        assert np.abs(setpoints).max() <= ceiling + 1e-9, (
            f"{name} asked for {np.abs(setpoints).max():.4f} kWh, inverter allows {ceiling:.4f}"
        )
        print(f"   {name:32}{np.abs(setpoints).max():16.4f}"
              f"{f'{soc.min():.2f} .. {soc.max():.2f}':>18}"
              f"{out['SOC_Drift_kWh']:10.1e}")
    print(f"   none drifted, none exceeded the inverter  OK")


def check_self_consumption_stays_off_grid(env, sig):
    """4. Self-consumption charges only from the roof, and never exports the pack.

    The rule's whole claim is that it does not trade with the grid; if it
    grid-charges, it is quietly arbitraging and is not the floor it is reported
    as.
    """
    print("\n4. Self-consumption never touches the grid")
    out = rbc.run_policy(env, rbc.make_policy("self_consumption"), signals=sig,
                         keep_traces=True)
    assert out["Grid_Charged_kWh"] < 1e-9, (
        f"self_consumption drew {out['Grid_Charged_kWh']:.4f} kWh from the grid"
    )
    # It may only discharge into a real deficit, never into an export.
    discharge = np.maximum(-out["_setpoints"], 0.0)
    overshoot = float(np.max(discharge - sig.deficit))
    assert overshoot < 1e-9, (
        f"self_consumption discharged {overshoot:.6f} kWh past the household load"
    )
    print(f"   grid-charged {out['Grid_Charged_kWh']:.2e} kWh, "
          f"discharged past the load by {overshoot:.2e} kWh  OK")


def check_bill_is_reproducible(env, sig):
    """5. Re-pricing the executed flows from scratch returns the same bill.

    The same guarantee `Battery_Envelope_Check` makes for the MILP: the reported
    cost is a settlement of the flows, not a number carried out of the control
    logic.
    """
    print("\n5. The bill re-prices from the flows alone")
    policy = rbc.make_policy("price_rank_daily")     # the rule that trades most
    out = rbc.run_policy(env, policy, signals=sig, keep_traces=True)
    setpoints = out["_setpoints"]

    total = 0.0
    peak_state = {b: 0.0 for b in (1, 2, 3, 4, 5)}
    for idx in range(sig.n_steps):
        peak_state = hc._drop_peak_on_window_start(peak_state, sig.windows, idx)
        net = sig.consumption[idx] - sig.generation[idx] + setpoints[idx]
        result = calculate_interval_price(
            smp_market_price_kwh=env.arr_price[idx],
            total_consumed_kwh=float(net),
            utc_date=env.dataset.index[idx],
            interval_minutes=env.interval_minutes,
            scheme=env.pricing_scheme,
            dogovorjena_moc=env.agreed_power_at(idx),
            prev_peak_kw=peak_state,
            **env.pricing_options,
        )
        peak_state = dict(result["new_peak_kw"])
        total += float(result["constant_price_aud"]) + float(result["variable_price_aud"])

    billed = out["Cost_EUR"]
    assert abs(total - billed) < 1e-6, (
        f"independent repricing {total:.6f} vs reported {billed:.6f}"
    )
    print(f"   independent repricing {total:.6f} == reported {billed:.6f}  OK")


def check_milp_is_a_lower_bound(env, sig, baseline):
    """6. No rule may beat the whole-year MILP.

    The single most important invariant here. The MILP minimises the same bill
    over the same battery with perfect foresight, so it is a lower bound on
    every rule by construction. A rule that comes in under it is not controlling
    better -- it is being priced differently, and the comparison is void.
    """
    print("\n6. The perfect-foresight MILP is a lower bound on every rule")
    optimum = rbc.run_milp_reference(env, n_steps=sig.n_steps)["Cost_EUR_Closed"]
    achievable = baseline - optimum
    print(f"   no battery {baseline:9.2f}    MILP {optimum:9.2f}    "
          f"achievable {achievable:8.2f} EUR")
    print(f"   {'rule':32}{'EUR':>10}{'gap EUR':>10}{'captured':>10}")
    for name in rbc.POLICY_ORDER:
        policy = rbc.make_policy(name)
        cost = rbc.run_policy(env, policy, signals=sig)["Cost_EUR_Closed"]
        gap = cost - optimum
        # The MILP stops at its MIP gap, so the bound is that, not zero.
        assert gap > -abs(optimum) * mh.FULL_PERIOD_GAP_REL - 1e-6, (
            f"{name} bills {cost:.4f}, under the MILP's {optimum:.4f} by {-gap:.4f} EUR: "
            f"the two are not being priced the same way"
        )
        captured = 100.0 * (baseline - cost) / achievable if achievable > 5.0 else float("nan")
        print(f"   {name:32}{cost:10.2f}{gap:10.2f}{captured:9.0f}%")
    print(f"   every rule at or above the optimum  OK")
    return optimum


def check_peak_shaving_shaves(env, sig):
    """7. The peak shaver lowers both the peak and what the ratchet charges."""
    print("\n7. Peak shaving lowers the peak and the excess-power charge")
    idle = rbc.run_policy(env, rbc._Idle(), signals=sig)
    print(f"   {'rule':32}{'peak kW':>10}{'power EUR':>12}")
    print(f"   {'no battery':32}{idle['Peak_Import_kW']:10.2f}{idle['Power_EUR']:12.2f}")
    for name in ("peak_shaving", "self_consumption_peak_shaving"):
        out = rbc.run_policy(env, rbc.make_policy(name), signals=sig)
        assert out["Peak_Import_kW"] <= idle["Peak_Import_kW"] + 1e-9, (
            f"{name} raised the peak from {idle['Peak_Import_kW']:.3f} to "
            f"{out['Peak_Import_kW']:.3f} kW"
        )
        assert out["Power_EUR"] <= idle["Power_EUR"] + 1e-9, (
            f"{name} raised the excess charge from {idle['Power_EUR']:.4f} to "
            f"{out['Power_EUR']:.4f} EUR"
        )
        print(f"   {name:32}{out['Peak_Import_kW']:10.2f}{out['Power_EUR']:12.2f}")
    print("   both shave the peak and neither raises the charge  OK")


def check_signals_rebind(env, sig):
    """8. Only the envelope depends on capacity.

    `run_unit` prices the year once and re-binds the bundle across the capacity
    sweep. That is only sound while the load, the delivered rates, the calendar
    and the agreed power are all independent of the pack.
    """
    print("\n8. The signal bundle re-binds across capacities")
    other = build(capacity_kwh=30.0)
    rebound = rbc.rebind_signals(sig, other)
    fresh = rbc.build_signals(other, sig.n_steps)
    for field in ("consumption", "generation", "import_rate", "export_credit",
                  "blocks", "windows", "agreed_kw", "local_hour", "day_idx"):
        a, b = getattr(rebound, field), getattr(fresh, field)
        assert np.array_equal(a, b), f"{field} differs between a rebind and a fresh build"
    assert rebound.capacity_kwh == fresh.capacity_kwh == 30.0
    assert rebound.max_charge_ac == fresh.max_charge_ac
    print(f"   9 signal arrays identical, envelope re-bound "
          f"{sig.capacity_kwh:g} -> {rebound.capacity_kwh:g} kWh  OK")


def check_foresight_pays(env, sig):
    """9. The same threshold rule, reading forward, does better ON AVERAGE.

    `price_oracle` is `price_threshold` with the window pointed at the future.
    Nothing else differs, so the difference between them is what foresight is
    worth to a threshold rule.

    Averaged across the price lists, not asserted on one. A threshold rule is a
    heuristic, not an optimizer, so better information moves where its thresholds
    sit without guaranteeing it moves them the right way -- the forward window
    loses on about a fifth of the study's units, and a single-unit assertion here
    would be a coin flip.
    """
    print("\n9. Foresight is worth something to the threshold rule, on average")
    data = hc.load_user(HOUSEHOLD, DATASET)
    gains = {}
    print(f"   {'price list':16}{'trailing':>12}{'forward':>12}{'foresight':>12}")
    for tariff in hc.TARIFF_ORDER:
        if not hc.tariff_allowed(tariff, DATASET):
            continue
        one = hc.build_env(data, capacity_kwh=CAPACITY_KWH, tariff=tariff)
        one_sig = rbc.build_signals(one, steps_for(one))
        causal = rbc.run_policy(one, rbc.make_policy("price_threshold"), signals=one_sig)
        forward = rbc.run_policy(one, rbc.make_policy("price_oracle"), signals=one_sig)
        gains[tariff] = causal["Cost_EUR_Closed"] - forward["Cost_EUR_Closed"]
        print(f"   {tariff:16}{causal['Cost_EUR_Closed']:12.2f}"
              f"{forward['Cost_EUR_Closed']:12.2f}{gains[tariff]:12.2f}")
    mean_gain = sum(gains.values()) / len(gains)
    assert mean_gain > 0, (
        f"the forward window is worth {mean_gain:+.4f} EUR/a averaged over "
        f"{len(gains)} price lists: the pair is not isolating foresight"
    )
    print(f"   foresight worth {mean_gain:+.2f} EUR/a over {len(gains)} lists  OK")


def check_daily_cap_binds():
    """10. A daily cycle cap holds the rules to the budget the MILP is held to.

    Measured against the USABLE window and per LOCAL day, which is what
    `add_household_physics` constrains -- 88- and 104-interval DST days included.
    """
    print("\n10. The daily cycle cap binds, per local day")
    cap = 0.5
    env = build()
    env.max_daily_cycles = cap
    sig = rbc.build_signals(env, steps_for(env))
    free_env = build()
    free_sig = rbc.build_signals(free_env, steps_for(free_env))

    print(f"   {'rule':32}{'EFC free':>10}{'EFC capped':>12}{'worst day':>12}")
    for name in ("price_rank_daily", "self_consumption"):
        free = rbc.run_policy(free_env, rbc.make_policy(name), signals=free_sig)
        held = rbc.run_policy(env, rbc.make_policy(name), signals=sig, keep_traces=True)
        setpoints = held["_setpoints"]
        stored = (np.maximum(setpoints, 0.0) * sig.eta_ch
                  + np.maximum(-setpoints, 0.0) / sig.eta_dis)
        worst = max(
            stored[np.asarray(steps)].sum() / (2.0 * sig.capacity_kwh)
            for steps in sig.day_steps if steps
        )
        assert worst <= cap + 1e-9, (
            f"{name} ran {worst:.4f} cycles on its busiest local day, cap is {cap}"
        )
        print(f"   {name:32}{free['Equivalent_Full_Cycles']:10.1f}"
              f"{held['Equivalent_Full_Cycles']:12.1f}{worst:12.3f}")
    print(f"   no local day exceeded {cap} usable-window cycles  OK")


def main():
    horizon = "90 days" if QUICK else "the whole profile"
    print(f"Rule-based control check -- {DATASET} #{HOUSEHOLD}, {TARIFF}, "
          f"{CAPACITY_KWH:g} kWh, {horizon}")
    print(f"{len(rbc.POLICY_ORDER)} rules: {', '.join(rbc.POLICY_ORDER)}")

    env = build()
    sig = rbc.build_signals(env, steps_for(env))

    baseline = check_idle_is_the_baseline(env, sig)
    check_empty_pack(baseline)
    check_envelope(env, sig)
    check_self_consumption_stays_off_grid(env, sig)
    check_bill_is_reproducible(env, sig)
    check_milp_is_a_lower_bound(env, sig, baseline)
    check_peak_shaving_shaves(env, sig)
    check_signals_rebind(env, sig)
    check_foresight_pays(env, sig)
    check_daily_cap_binds()

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
