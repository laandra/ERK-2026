"""Validation for the battery operating envelope: SOC window, daily cycle cap,
wear price, and the economics that read them.

In the style of `New pricing functions/test_primer.py` -- a print-driven script
with assertions, run directly rather than through a test runner:

    .venv/bin/python Battery_Envelope_Check.py            # full calendar year
    .venv/bin/python Battery_Envelope_Check.py --quick    # 120 days, spring DST only

What each check is for:

  1  defaults are a no-op          nothing new may move an existing result
  2  SOC window derates the pack   usable shrinks, the inverter does not
  3  daily cap binds, per LOCAL day  including the 88- and 104-interval DST days
  4  the wear price is not a bill  it steers dispatch and never reaches Cum_Cost
  5  the wear price bites          more wear priced in, fewer cycles, higher bill
  6  economics                     OPEX enters every figure that spends capital
  7  the two MILPs still agree      community and household price the same pack alike
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pulp

import Battery_Economics as be
import Community_MILP as ccm
import Data_Loader as dl
import MILP_Household as mh
from si_cas import v_lokalni_cas

PRICE_COLUMN = "SMP"
GENERATION_COLUMN = "Feed_In_Volume_kWh"
CONSUMPTION_COLUMN = "Consumption_Volume_kWh"
DATASET, HOUSEHOLD = "Fluvius_PV", 160
PAKET_ID = "GENI_SAMO_DINAMICNI"      # the list that cycles hardest, so the cap bites
REFERENCE_YEAR, PEAK_RESET_MONTHS = 2026, 1
STEPS_PER_DAY = 96

QUICK = "--quick" in sys.argv
N_STEPS = 120 * STEPS_PER_DAY if QUICK else None    # None -> the whole profile

_SOLVER = pulp.PULP_CBC_CMD(
    msg=False, gapRel=mh.FULL_PERIOD_GAP_REL, timeLimit=mh.FULL_PERIOD_TIME_LIMIT_S
)


def load_profile():
    """The household profile with the SI SMP series priced into PRICE_COLUMN."""
    frame = dl.load_household_data(HOUSEHOLD, dataset=DATASET)
    smp = dl.load_smp_data("Slovenia").reindex(frame.index, method="ffill")
    series = pd.to_numeric(smp[PRICE_COLUMN], errors="coerce").ffill().bfill()
    # Auto-detect legacy EUR/MWh inputs, as the studies do.
    scale = 1000.0 if float(series.abs().quantile(0.95)) > 2.0 else 1.0
    frame[PRICE_COLUMN] = (series / scale).astype(float)
    return frame


def solve(frame, capacity_kwh, **envkw):
    """One solve at the given nameplate capacity and envelope."""
    env = mh.build_household_env(
        frame, capacity_kwh=capacity_kwh, scheme="si_samooskrba", paket_id=PAKET_ID,
        pricing_reference_year=REFERENCE_YEAR, peak_reset_months=PEAK_RESET_MONTHS,
        price_column=PRICE_COLUMN, generation_column=GENERATION_COLUMN,
        consumption_column=CONSUMPTION_COLUMN, steps_per_day=STEPS_PER_DAY,
        episode_length=len(frame), **envkw,
    )
    n = len(frame) if N_STEPS is None else N_STEPS
    soc = mh.SOC_FRACTION * env.battery_capacity_kwh
    df = mh.solve_household(
        env, n_steps=n, initial_soc_kwh=soc, final_soc_kwh=soc,
        solver=_SOLVER, verbose=False,
    )
    assert df.attrs["solver_status"] == "Optimal", \
        f"solver ended {df.attrs['solver_status']}, not Optimal"
    return env, df


def summarize(env, df):
    return mh.summarize_trajectory(
        df, capacity_kwh=env.battery_capacity_kwh,
        nominal_capacity_kwh=env.nominal_capacity_kwh, hours_per_step=24.0 / STEPS_PER_DAY,
    )


def stored_throughput_by_local_day(env, df):
    """kWh through the STORE per Brussels-local day: the quantity the cap bounds."""
    days = [v_lokalni_cas(ts).date() for ts in df["Date"]]
    moved = (
        df["Charge_kW"].to_numpy(float) * env.charge_efficiency
        + df["Discharge_kW"].to_numpy(float) / env.discharge_efficiency
    )
    return pd.Series(moved, index=pd.Index(days, name="day")).groupby("day").sum()


# ---------------------------------------------------------------------------
def check_defaults_are_noop(frame):
    print("\n1. Defaults are a no-op")
    env, df = solve(frame, 10.0)
    s = summarize(env, df)
    assert env.nominal_capacity_kwh == env.battery_capacity_kwh == 10.0
    assert env.max_daily_cycles is None and env.cycle_cost_eur_per_efc is None
    assert df.attrs["cycle_cost_eur"] == 0.0
    # With no window the two cycle figures are the same number.
    assert abs(s["Equivalent_Full_Cycles"] - s["EFC_Usable"]) < 1e-12
    print(f"   10 kWh, no envelope: {float(df['Cum_Cost'].iloc[-1]):.4f} EUR, "
          f"{s['Equivalent_Full_Cycles']:.1f} EFC, no wear charged")
    print("   both cycle denominators agree when usable == nameplate  OK")
    return env, df, s


def check_soc_window(frame, base_summary):
    print("\n2. The SOC window derates the pack, not the inverter")
    env, df = solve(frame, 10.0, soc_min_frac=0.1, soc_max_frac=0.9)
    s = summarize(env, df)

    assert env.nominal_capacity_kwh == 10.0, "nameplate must survive the derating"
    assert abs(env.battery_capacity_kwh - 8.0) < 1e-12, "10-90 % of 10 kWh is 8 kWh usable"
    soc = df["SOC_kWh"].to_numpy(float)
    assert soc.min() >= -1e-9 and soc.max() <= 8.0 + 1e-9, "SOC left the usable window"
    # The inverter is unchanged: the per-interval limit still comes off the
    # nameplate pack, min(0.5 * 10, 11) kW held for a quarter hour.
    expected_kwh = min(mh.C_RATE * 10.0, mh.INVERTER_MAX_KW) * 24.0 / STEPS_PER_DAY
    assert abs(env.max_charge_kwh - expected_kwh) < 1e-12, "the window shrank the inverter"
    assert s["Capacity_kWh"] == 10.0, "economics must still price the nameplate pack"
    assert s["Usable_Capacity_kWh"] == 8.0
    # Same energy read against two denominators: usable is the larger figure.
    assert s["EFC_Usable"] > s["Equivalent_Full_Cycles"]

    print(f"   nameplate 10.0 kWh -> usable {env.battery_capacity_kwh:.1f} kWh, "
          f"SOC spans {soc.min():.2f}..{soc.max():.2f} kWh")
    print(f"   power limit {env.max_charge_kwh / (24.0/STEPS_PER_DAY):.1f} kW "
          f"(nameplate, unchanged)")
    print(f"   EFC nameplate {s['Equivalent_Full_Cycles']:.1f}  vs  "
          f"usable {s['EFC_Usable']:.1f}   ({base_summary['Equivalent_Full_Cycles']:.1f} "
          f"undelated)  OK")


def check_daily_cap(frame):
    print("\n3. The daily cycle cap, on the usable window and the local day")
    for label, kw in (("no window", {}),
                      ("10-90 % window", {"soc_min_frac": 0.1, "soc_max_frac": 0.9})):
        env, df = solve(frame, 10.0, max_daily_cycles=2.0, **kw)
        usable = env.battery_capacity_kwh
        budget = 2.0 * 2.0 * usable
        per_day = stored_throughput_by_local_day(env, df)

        assert per_day.max() <= budget + 1e-6, (
            f"{label}: {per_day.idxmax()} moved {per_day.max():.4f} kWh, budget {budget:.4f}"
        )
        # Local days, not a 96-step stride: on a full year the DST days come out
        # 88 and 104 intervals, and each horizon edge leaves a part-day.
        counts = pd.Series([v_lokalni_cas(t).date() for t in df["Date"]]).value_counts()
        odd = counts[counts != STEPS_PER_DAY]
        # The cap has to actually bind, or this check proves nothing.
        n_binding = int((per_day > budget - 1e-6).sum())
        assert n_binding > 0, f"{label}: the cap never bound, nothing was tested"

        print(f"   {label:16} usable {usable:.1f} kWh, budget {budget:5.1f} kWh/day  "
              f"peak day {per_day.max():6.2f}  binding on {n_binding} of {len(per_day)} days")
        if len(odd):
            print(f"   {'':16} DST days handled: "
                  + ", ".join(f"{d} = {int(c)} intervals" for d, c in odd.items()))
            for d in odd.index:
                assert per_day.loc[d] <= budget + 1e-6, f"DST day {d} broke the cap"
    print("   cap holds on every local day, DST included  OK")


def check_wear_price_is_not_a_bill(frame):
    print("\n4. The wear price steers dispatch and never reaches the bill")
    rate = be.cycle_cost_eur_per_efc(10.0)
    env, df = solve(frame, 10.0, cycle_cost_eur_per_efc=rate)
    s = summarize(env, df)

    # Re-price the returned trajectory from scratch, the way an invoice would.
    from Pricing_Functions import calculate_interval_price
    peak = env.compute_seed_peak_kw(0)
    windows = env.reset_window_ids
    total = 0.0
    net = mh.net_grid_kwh(df)
    agreed = mh.agreed_power_by_month(env, 0, mh.month_calendar(df["Date"])[2])
    month_idx = mh.month_calendar(df["Date"])[3]
    for t, ts in enumerate(df["Date"]):
        if t and windows[t] != windows[t - 1]:
            peak = {b: 0.0 for b in peak}
        res = calculate_interval_price(
            smp_market_price_kwh=float(df["SMP_MWh"].iloc[t]), total_consumed_kwh=float(net[t]),
            utc_date=ts, interval_minutes=int(round(env.interval_minutes)),
            scheme=env.pricing_scheme, dogovorjena_moc=agreed[month_idx[t]],
            prev_peak_kw=peak, paket_id=PAKET_ID, pricing_reference_year=REFERENCE_YEAR,
        )
        peak = dict(res["new_peak_kw"])
        total += float(res["constant_price_aud"]) + float(res["variable_price_aud"])

    billed = float(df["Cum_Cost"].iloc[-1])
    assert abs(total - billed) < 1e-6, (
        f"the wear price leaked into the bill: independent repricing {total:.6f} "
        f"vs reported {billed:.6f}"
    )
    charged = df.attrs["cycle_cost_eur"]
    assert charged > 0, "the wear price was configured but charged nothing"
    print(f"   wear price {rate:.4f} EUR/EFC charged the objective {charged:.2f} EUR")
    print(f"   independent repricing {total:.4f} == reported bill {billed:.4f}  OK")
    print(f"   ({s['Equivalent_Full_Cycles']:.1f} EFC under the wear price)")


def check_wear_price_bites(frame, base_summary, base_cost):
    print("\n5. More wear priced in -> fewer cycles, higher bill")
    full = be.cycle_cost_eur_per_efc(10.0)
    rows = [(0.0, base_summary["Equivalent_Full_Cycles"], base_cost)]
    for mult in (0.5, 1.0, 2.0):
        env, df = solve(frame, 10.0, cycle_cost_eur_per_efc=mult * full)
        s = summarize(env, df)
        rows.append((mult * full, s["Equivalent_Full_Cycles"], float(df["Cum_Cost"].iloc[-1])))

    print(f"   {'EUR/EFC':>10} {'EFC':>10} {'bill EUR':>12}")
    for rate, efc, cost in rows:
        print(f"   {rate:10.4f} {efc:10.1f} {cost:12.4f}")
    efcs = [r[1] for r in rows]
    costs = [r[2] for r in rows]
    assert all(b <= a + 1e-6 for a, b in zip(efcs, efcs[1:])), "cycling did not fall"
    assert all(b >= a - 1e-6 for a, b in zip(costs, costs[1:])), "the bill did not rise"
    print(f"   cycling {efcs[0]:.0f} -> {efcs[-1]:.0f} EFC, bill "
          f"{costs[0]:.2f} -> {costs[-1]:.2f} EUR, both monotone  OK")


def check_economics():
    print("\n6. Economics: OPEX enters every figure that spends capital")
    savings, cap, efc = np.array([300.0]), np.array([10.0]), np.array([267.0])

    off = be.battery_economics(savings, cap, efc, opex_frac=0.0)
    on = be.battery_economics(savings, cap, efc)

    capex = float(off["Capex_EUR"][0])
    assert capex == 10.0 * be.CAPEX_EUR_PER_KWH + be.CAPEX_FIXED_EUR
    assert float(off["Annual_OPEX_EUR"][0]) == 0.0
    assert abs(float(on["Annual_OPEX_EUR"][0]) - 0.015 * capex) < 1e-12

    # By hand: Net_Annual = Savings - opex - Capex * crf
    crf = be.capital_recovery_factor(be.DISCOUNT_RATE, float(on["Service_Life_y"][0]))
    expect = 300.0 - 0.015 * capex - capex * crf
    assert abs(float(on["Net_Annual_EUR"][0]) - expect) < 1e-9

    # Break-even must be the price at which the decision is exactly neutral.
    x = float(on["Break_Even_Capex_EUR_kWh"][0])
    at_break_even = be.battery_economics(savings, cap, efc, capex_eur_per_kwh=x)
    assert abs(float(at_break_even["Net_Annual_EUR"][0])) < 1e-9, \
        "Break_Even_Capex_EUR_kWh does not zero Net_Annual_EUR"

    print(f"   capex {capex:,.0f} EUR -> OPEX {float(on['Annual_OPEX_EUR'][0]):.2f} EUR/a "
          f"at {be.OPEX_FRAC_OF_CAPEX_PER_YEAR:.1%}")
    print(f"   Net_Annual  {float(off['Net_Annual_EUR'][0]):+8.2f} (no OPEX)  ->  "
          f"{float(on['Net_Annual_EUR'][0]):+8.2f} EUR/a")
    print(f"   NPV         {float(off['NPV_EUR'][0]):+8.2f}          ->  "
          f"{float(on['NPV_EUR'][0]):+8.2f} EUR")
    print(f"   break-even  {float(off['Break_Even_Capex_EUR_kWh'][0]):8.2f} EUR/kWh   ->  "
          f"{x:8.2f} EUR/kWh, and it zeroes Net_Annual  OK")


def check_models_agree(frame):
    print("\n7. The community MILP and the household MILP still agree")
    # One month, so this stays quick: the community model solves month by month
    # anyway. `SCHEME_INDIVIDUAL` with a single member is the configuration where
    # the two are solving the same problem and must return the same bill.
    start, length, month = ccm.month_slices(frame.index)[5]
    print(f"   {month[0]}-{month[1]:02d}, {length} intervals, one member, no sharing")
    print(f"   {'envelope':28}{'community':>13}{'household':>13}{'diff EUR':>12}{'relative':>11}")

    for label, kw in (
        ("none (defaults)", {}),
        ("max_daily_cycles = 2", {"max_daily_cycles": 2.0}),
        ("SOC 10-90 %", {"soc_min_frac": 0.1, "soc_max_frac": 0.9}),
        ("both + wear price", {"max_daily_cycles": 2.0, "soc_min_frac": 0.1,
                               "soc_max_frac": 0.9,
                               "cycle_cost_eur_per_efc": be.cycle_cost_eur_per_efc(10.0)}),
    ):
        env = mh.build_household_env(
            frame, capacity_kwh=10.0, scheme="si_samooskrba", paket_id=PAKET_ID,
            pricing_reference_year=REFERENCE_YEAR, peak_reset_months=PEAK_RESET_MONTHS,
            price_column=PRICE_COLUMN, generation_column=GENERATION_COLUMN,
            consumption_column=CONSUMPTION_COLUMN, steps_per_day=STEPS_PER_DAY,
            episode_length=len(frame), **kw,
        )
        m = ccm.CommunityMember(
            member_id="A", env=env, role=ccm.ROLE_SENDER, contract_key="check",
            key_weight=1.0, znacilni_primer=5, label="A",
        )
        res = ccm.solve_community_milp(
            [m], scheme=ccm.SCHEME_INDIVIDUAL, start_idx=start, n_steps=length,
            solver=_SOLVER, verbose=False,
        )
        community = float(res["community_cost_eur"])
        soc = mh.SOC_FRACTION * env.battery_capacity_kwh
        df = mh.solve_household(env, start_idx=start, n_steps=length, initial_soc_kwh=soc,
                                final_soc_kwh=soc, solver=_SOLVER, verbose=False)
        household = float(df["Cum_Cost"].iloc[-1])

        # The bound is the MIP gap, not a fixed number of cents. Both models stop
        # at FULL_PERIOD_GAP_REL, so two runs of the same problem may land on
        # different near-optimal vertices; tightening the gap to 1e-7 closes the
        # difference to ~5e-8, which is what proves it is termination and not a
        # difference in what is being modelled.
        relative = abs(community - household) / max(abs(household), 1e-9)
        assert relative < mh.FULL_PERIOD_GAP_REL, (
            f"{label}: the two models disagree by {relative:.2e} relative, more than "
            f"the {mh.FULL_PERIOD_GAP_REL:.0e} gap both solve to"
        )
        print(f"   {label:28}{community:13.6f}{household:13.6f}"
              f"{community - household:+12.2e}{relative:11.1e}")
    print(f"   all within the {mh.FULL_PERIOD_GAP_REL:.0e} MIP gap both models solve to  OK")


def main():
    horizon = "120 days (spring DST only)" if QUICK else "the whole profile"
    print(f"Battery envelope check -- {DATASET} #{HOUSEHOLD}, {PAKET_ID}, {horizon}")
    frame = load_profile()

    env, df, base = check_defaults_are_noop(frame)
    base_cost = float(df["Cum_Cost"].iloc[-1])
    check_soc_window(frame, base)
    check_daily_cap(frame)
    check_wear_price_is_not_a_bill(frame)
    check_wear_price_bites(frame, base, base_cost)
    check_economics()
    check_models_agree(frame)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
