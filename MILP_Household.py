"""The single-household MILP: the model, the environment it runs on, and the
trajectory it produces.

One household is one metering point with a PV roof, a battery and a price list.
Its physics are the same wherever it is solved -- on its own by
`solve_household`, or as one member of a joint community problem in
`Community_MILP` -- so the pieces that describe those physics live here and are
appended to whatever `pulp.LpProblem` the caller is building:

    add_household_physics     buy/sell/spill/charge/discharge/SOC and the
                              constraints that tie them together
    add_excess_power_ratchet  the presezna-moc charge, per (block, month)
    agreed_power_by_month     the per-month dogovorjena obracunska moc
    month_calendar            the local-time month bookkeeping all of the above index by
    floor_export_rates        the guard that keeps a negative import rate from
                              making the LP unbounded

What is NOT here is pricing. `solve_household` prices through
`Pricing_Functions.calculate_interval_price`, the community model prices through
`si_obracun.samooskrba` with VAT applied explicitly and a sharing overlay on top;
those are genuinely different settlements, and only the physics are shared.

`build_household_env` is the one `HouseholdEnvironment` factory. Every study
passes its own price list, capacity and agreed-power rule; nothing about a
particular study is defaulted here.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from Environment import HouseholdEnvironment
from Pricing_Functions import (
    InvoiceBuilder,
    PRIVZETO_REFERENCNO_LETO,
    calculate_interval_price,
    compute_prorated_fixed_charge_eur,
    moc_za_mesec,
)

# The SI tariff primitives live in the spaced folder.
_SI_DIR = Path(__file__).resolve().parent / "New pricing functions"
if str(_SI_DIR) not in sys.path:
    sys.path.append(str(_SI_DIR))

from si_cas import je_visja_sezona, v_lokalni_cas  # noqa: E402
from si_obracun import Pravila  # noqa: E402


# ---------------------------------------------------------------------------
# Battery, horizon and solver defaults
# ---------------------------------------------------------------------------
# Shared by every study so two of them cannot quietly size the same battery
# differently. `Horizon_Comparison` re-exports these, which is where most
# callers still read them from.
STEPS_PER_DAY = 96
SOC_FRACTION = 0.5
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
C_RATE = 0.5
INVERTER_MAX_KW = 11.0

# A whole year at 15-minute resolution is a large MIP; closing the last 0.1 % of
# the gap costs hours and moves a bill in the hundreds by cents.
FULL_PERIOD_GAP_REL = 0.001
FULL_PERIOD_TIME_LIMIT_S = 900


def full_period_solver(gap_rel=FULL_PERIOD_GAP_REL, time_limit_s=FULL_PERIOD_TIME_LIMIT_S):
    """CBC with the whole-period gap and time limit every study solves at."""
    return pulp.PULP_CBC_CMD(msg=False, gapRel=float(gap_rel), timeLimit=float(time_limit_s))


def step_energy_kwh(capacity_kwh, c_rate=C_RATE, inverter_max_kw=INVERTER_MAX_KW,
                    steps_per_day=STEPS_PER_DAY):
    """Per-interval charge/discharge limit for a battery of `capacity_kwh`.

    The power rating scales with capacity up to the inverter, and the interval
    limit is that power held for one interval.
    """
    power_kw = min(float(c_rate) * float(capacity_kwh), float(inverter_max_kw))
    return power_kw * 24.0 / float(steps_per_day)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def build_household_env(
    data,
    *,
    capacity_kwh,
    scheme,
    paket_id,
    pricing_reference_year,
    peak_reset_months,
    price_column,
    generation_column,
    consumption_column,
    steps_per_day=STEPS_PER_DAY,
    charge_efficiency=CHARGE_EFFICIENCY,
    discharge_efficiency=DISCHARGE_EFFICIENCY,
    c_rate=C_RATE,
    inverter_max_kw=INVERTER_MAX_KW,
    episode_length=None,
    contracted_power_kw=None,
    agreed_power=None,
):
    """The `HouseholdEnvironment` every study solves on.

    `episode_length` defaults to `len(data) - 1`; pass `len(data)` to keep the
    closing interval, which a full-calendar-year solve wants.

    `agreed_power` is the dogovorjena-obracunska-moc rule as a dict of
    `connection_power_kw` / `min_agreed_power_kw` / `agreed_power_lag_months` /
    `agreed_power_bootstrap`. Left out, the environment's own defaults apply --
    which is what every study except `Horizon_Comparison` relies on.
    """
    step_kwh = step_energy_kwh(capacity_kwh, c_rate, inverter_max_kw, steps_per_day)
    kwargs = dict(
        dataset=data,
        price_column=price_column,
        generation_column=generation_column,
        consumption_column=consumption_column,
        action_mode="continuous",
        allow_curtailment=True,
        reset_mode="deterministic",
        episode_length=(len(data) - 1) if episode_length is None else int(episode_length),
        steps_per_day=steps_per_day,
        battery_capacity_kwh=float(capacity_kwh),
        charge_efficiency=charge_efficiency,
        discharge_efficiency=discharge_efficiency,
        max_charge_kwh=step_kwh,
        max_discharge_kwh=step_kwh,
        pricing_scheme=scheme,
        pricing_reference_year=pricing_reference_year,
        pricing_options={"paket_id": paket_id},
        peak_reset_months=peak_reset_months,
    )
    if contracted_power_kw is not None:
        kwargs["contracted_power_kw"] = contracted_power_kw
    if agreed_power:
        kwargs.update(agreed_power)
    return HouseholdEnvironment(**kwargs)


# ---------------------------------------------------------------------------
# Calendar and rate bookkeeping
# ---------------------------------------------------------------------------
def month_calendar(dates):
    """Local-time month bookkeeping for one horizon.

    Returns `(local_times, month_key_t, months_sorted, month_idx_t)`. The Fluvius
    stamps carry a `Z` suffix but are Brussels-local, so the month has to be read
    off `v_lokalni_cas` rather than off the raw index.
    """
    local_times = [v_lokalni_cas(ts) for ts in dates]
    month_key_t = [(lt.year, lt.month) for lt in local_times]
    months_sorted = sorted(set(month_key_t))
    month_idx_t = [months_sorted.index(k) for k in month_key_t]
    return local_times, month_key_t, months_sorted, month_idx_t


def agreed_power_by_month(env, start_idx, months_sorted, do_not_use_previous_month=False):
    """`{month index: {block: agreed kW}}` for the months the horizon covers.

    The agreed billing power is a per-month constant, resolved once so every
    constraint, objective term and pricing call reads the same figure.
    """
    run_schedule = env.agreed_power_for_run(
        start_idx, do_not_use_previous_month=do_not_use_previous_month
    )
    return {
        i: moc_za_mesec(run_schedule, y * 12 + mo - 1)
        for i, (y, mo) in enumerate(months_sorted)
    }


def floor_export_rates(import_rates, export_rates):
    """Floor the export credit at the delivered import rate. Returns (rates, n).

    Without it, an interval whose delivered import rate is negative makes the
    buy/sell round trip profitable and the LP unbounded. Flooring makes it
    exactly neutral, and curtailment is free, so the optimum never exports there
    anyway.
    """
    if isinstance(export_rates, np.ndarray) or isinstance(import_rates, np.ndarray):
        imp = np.asarray(import_rates, dtype=float)
        exp = np.asarray(export_rates, dtype=float)
        return np.minimum(exp, imp), int(np.sum(exp > imp))
    n = sum(1 for e, i in zip(export_rates, import_rates) if e > i)
    return [min(e, i) for e, i in zip(export_rates, import_rates)], n


# ---------------------------------------------------------------------------
# The household block
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HouseholdNames:
    """LP variable name templates, formatted with `t`.

    Names are not cosmetic: pulp writes the model sorted by name, so the column
    order CBC sees -- and with it which of several equally optimal trajectories
    comes back -- follows them. Each caller keeps the names its results were
    produced under.
    """

    buy: str = "P_buy_{t}"
    sell: str = "P_sell_{t}"
    spill: str = "P_spill_{t}"
    charge: str = "P_ch_{t}"
    discharge: str = "P_dis_{t}"
    soc: str = "E_{t}"
    charging_flag: str = "B_charging_{t}"

    @classmethod
    def for_member(cls, member_id):
        """The community model's per-member names."""
        return cls(
            buy=f"buy_{member_id}_{{t}}",
            sell=f"sell_{member_id}_{{t}}",
            spill=f"spill_{member_id}_{{t}}",
            charge=f"ch_{member_id}_{{t}}",
            discharge=f"dis_{member_id}_{{t}}",
            soc=f"soc_{member_id}_{{t}}",
            charging_flag=f"bcharge_{member_id}_{{t}}",
        )


@dataclass
class HouseholdBlock:
    """The variables one household contributes to a problem."""

    buy: list
    sell: list
    spill: list
    charge: list
    discharge: list
    soc: Optional[list]
    max_charge_ac: float
    max_discharge_ac: float
    capacity_kwh: float


def add_household_physics(
    prob,
    env,
    *,
    n_steps,
    gen,
    con,
    names=HouseholdNames(),
    initial_soc_kwh=None,
    final_soc_kwh=None,
    exclusivity="binary",
    metering_bounds=False,
):
    """Declare one household's variables and physical constraints on `prob`.

    `exclusivity` picks how simultaneous charge and discharge is forbidden:

      "binary"   one binary per interval, up front. Charging and discharging at
                 once is only ever attractive where the delivered import rate is
                 negative -- the round-trip loss becomes an energy sink -- but
                 where it is, the LP will take it.
      "inverter" `charge + discharge <= the inverter rating`, which is the
                 physical bound and leaves the model a pure LP. The caller is
                 then responsible for repairing any interval that comes back
                 doing both, which is what the community model's lazy loop does.

    `metering_bounds` adds the two inequalities that hold at every physical
    operating point and bound `buy`/`sell` from running away together. The
    binary branch does not need them; the inverter branch does.

    The SOC recursion and the energy balance are emitted in ONE loop, so the
    rows interleave per interval. That is arbitrary mathematically and not
    arbitrary to the solver: on a degenerate LP the row order decides which of
    several equally optimal vertices CBC returns, at the 1e-10 level. Downstream
    of the community model's lazy binary loop that can compound to a few 1e-6
    EUR, which is inside its MIP gap but not zero.
    """
    if exclusivity not in ("binary", "inverter"):
        raise ValueError(f"exclusivity must be 'binary' or 'inverter', got {exclusivity!r}")

    capacity = float(env.battery_capacity_kwh)
    # The environment caps the change in STORED energy, so the AC-side bounds
    # here carry the efficiency factor.
    max_charge_ac = env.max_charge_kwh / env.charge_efficiency
    max_discharge_ac = env.max_discharge_kwh * env.discharge_efficiency

    buy = [pulp.LpVariable(names.buy.format(t=t), lowBound=0) for t in range(n_steps)]
    sell = [pulp.LpVariable(names.sell.format(t=t), lowBound=0) for t in range(n_steps)]

    if capacity > 0:
        soc = [
            pulp.LpVariable(names.soc.format(t=t), lowBound=0, upBound=capacity)
            for t in range(n_steps + 1)
        ]
        charge = [
            pulp.LpVariable(names.charge.format(t=t), lowBound=0, upBound=max_charge_ac)
            for t in range(n_steps)
        ]
        discharge = [
            pulp.LpVariable(names.discharge.format(t=t), lowBound=0, upBound=max_discharge_ac)
            for t in range(n_steps)
        ]
    else:
        # No battery, no variables: the balance below then reads as a constant 0.
        soc, charge, discharge = None, [0.0] * n_steps, [0.0] * n_steps

    # Curtailment, bounded by the interval's own generation: it can switch PV
    # off, never absorb imported energy.
    spill = [
        pulp.LpVariable(names.spill.format(t=t), lowBound=0, upBound=float(gen[t]))
        for t in range(n_steps)
    ]

    if soc is not None:
        if initial_soc_kwh is None:
            initial_soc_kwh = capacity / 2.0
        prob += (soc[0] == min(max(0.0, float(initial_soc_kwh)), capacity))
        # Terminal floor, so the opening charge cannot be sold off as a saving.
        if final_soc_kwh is not None:
            prob += (soc[n_steps] >= min(max(0.0, float(final_soc_kwh)), capacity))

    inverter_kwh = max(max_charge_ac, max_discharge_ac)
    for t in range(n_steps):
        if soc is not None:
            prob += (
                soc[t + 1]
                == soc[t]
                + charge[t] * env.charge_efficiency
                - discharge[t] / env.discharge_efficiency
            )
            if exclusivity == "inverter":
                # One inverter, one rating: charging and discharging cannot sum
                # to more than the unit can pass.
                prob += charge[t] + discharge[t] <= inverter_kwh
        # Energy balance with curtailment.
        prob += (
            gen[t] + buy[t] + discharge[t]
            == con[t] + sell[t] + charge[t] + spill[t]
        )
        if metering_bounds:
            # A meter nets import against export inside the interval. These two
            # hold at every physical operating point and bound the LP, which the
            # energy balance alone does not.
            prob += sell[t] <= max(gen[t] - con[t], 0.0) + discharge[t]
            prob += buy[t] <= max(con[t] - gen[t], 0.0) + charge[t] + spill[t]

    if exclusivity == "binary" and soc is not None:
        flags = [
            pulp.LpVariable(names.charging_flag.format(t=t), cat="Binary")
            for t in range(n_steps)
        ]
        for t in range(n_steps):
            prob += charge[t] <= max_charge_ac * flags[t]
            prob += discharge[t] <= max_discharge_ac * (1 - flags[t])

    return HouseholdBlock(
        buy=buy, sell=sell, spill=spill, charge=charge, discharge=discharge, soc=soc,
        max_charge_ac=max_charge_ac, max_discharge_ac=max_discharge_ac,
        capacity_kwh=capacity,
    )


def add_grid_exclusivity(prob, *, buy_t, sell_t, gen_t, con_t, max_charge_ac,
                         max_discharge_ac, flag):
    """Force one interval to import or export, not both, against binary `flag`."""
    prob += sell_t <= (float(gen_t) + max_discharge_ac) * flag
    prob += buy_t <= (float(con_t) + max_charge_ac + float(gen_t)) * (1 - flag)


def add_battery_exclusivity(prob, *, charge_t, discharge_t, max_charge_ac,
                            max_discharge_ac, flag):
    """Force one interval to charge or discharge, not both, against binary `flag`."""
    prob += charge_t <= max_charge_ac * flag
    prob += discharge_t <= max_discharge_ac * (1 - flag)


# ---------------------------------------------------------------------------
# The excess-power ratchet
# ---------------------------------------------------------------------------
def add_excess_power_ratchet(
    prob,
    env,
    buy,
    *,
    blocks,
    month_idx_t,
    months_sorted,
    agreed_by_month,
    start_idx,
    n_steps,
    hours,
    pravila,
    peak_name="P_peak_b{b}_m{m}",
    excess_name="Excess_b{b}_m{m}",
):
    """The presezna-moc charge: one peak variable per (block, month) occurring.

    The peak is ratcheted within a reset window and floored at the seed peak --
    but only in the window the horizon starts in; a window opening inside it
    resets to zero. It is measured on the METERED import `buy`, which no billing
    overlay can move.

    Returns `(peak_vars, excess_vars, objective_terms)`.
    """
    window_id_arr = env.reset_window_ids[start_idx:start_idx + n_steps]
    seed_peak_kw = env.compute_seed_peak_kw(start_idx)
    seed_window = int(window_id_arr[0])

    month_window = {}
    for t in range(n_steps):
        month_window.setdefault(month_idx_t[t], int(window_id_arr[t]))

    # Exact only at peak_reset_months=1: a wider window telescopes across a
    # boundary where the agreed power may change, i.e. against two contracts.
    for w in set(month_window.values()):
        in_window = [m for m, mw in month_window.items() if mw == w]
        if len({tuple(sorted(agreed_by_month[m].items())) for m in in_window}) > 1:
            raise ValueError(
                f"Ratchet window {w} spans months with different agreed billing power "
                f"({[months_sorted[m] for m in in_window]}). Set peak_reset_months=1 so "
                f"each window is one month, or build the environment with a constant "
                f"contracted_power_kw."
            )

    occurring = sorted({(int(blocks[t]), month_idx_t[t]) for t in range(n_steps)})
    peak_var = {(b, m): pulp.LpVariable(peak_name.format(b=b, m=m), lowBound=0)
                for (b, m) in occurring}
    excess_var = {(b, m): pulp.LpVariable(excess_name.format(b=b, m=m), lowBound=0)
                  for (b, m) in occurring}

    for t in range(n_steps):
        prob += peak_var[(int(blocks[t]), month_idx_t[t])] >= buy[t] / hours

    last_var_by_block, last_window_by_block = {}, {}
    for (b, m) in occurring:
        w = month_window[m]
        if b in last_var_by_block and last_window_by_block[b] == w:
            prob += peak_var[(b, m)] >= last_var_by_block[b]
        elif w == seed_window:
            prob += peak_var[(b, m)] >= seed_peak_kw.get(b, 0.0)
        last_var_by_block[b] = peak_var[(b, m)]
        last_window_by_block[b] = w

        prob += excess_var[(b, m)] >= peak_var[(b, m)] - agreed_by_month[m].get(b, 0.0)

    # Telescoping contribution per (block, month), at each month's own rate.
    objective_terms = []
    prev_excess_by_block, prev_window_by_block = {}, {}
    for (b, m) in occurring:
        y, mo = months_sorted[m]
        w = month_window[m]
        vs = je_visja_sezona(date(y, mo, 1))
        rate_bm = pravila.omreznina.postavka_moc(b, vs)
        faktor = pravila.omreznina.faktor_presezne_moci

        if b in prev_excess_by_block and prev_window_by_block[b] == w:
            prev_term = prev_excess_by_block[b]
        elif w == seed_window:
            prev_term = max(0.0, seed_peak_kw.get(b, 0.0) - agreed_by_month[m].get(b, 0.0))
        else:
            prev_term = 0.0          # a fresh window pays its own excess in full

        objective_terms.append((excess_var[(b, m)] - prev_term) * rate_bm * faktor)
        prev_excess_by_block[b] = excess_var[(b, m)]
        prev_window_by_block[b] = w

    return peak_var, excess_var, objective_terms


# ---------------------------------------------------------------------------
# One household, solved on its own
# ---------------------------------------------------------------------------
def solve_household(
    env,
    start_idx=0,
    n_steps=None,
    initial_soc_kwh=None,
    final_soc_kwh=None,
    verbose=True,
    problem_name="Household_Microgrid_Optimization",
    solver=None,
    annual_netting_rate_eur_per_kwh=None,
    generate_invoice=False,
    invoice_output_dir=None,
    invoice_run_label="milp_eval",
    do_not_use_previous_month=False,
):
    """Cost-minimal battery dispatch over one episode, as a per-step DataFrame.

    `initial_soc_kwh` / `final_soc_kwh` pin the stored energy at the horizon
    ends; `annual_netting_rate_eur_per_kwh` adds the NET-metering settlement
    credit = rate * min(sum P_buy, sum P_sell) to the objective.
    """
    if verbose:
        print("Building MILP Model...")

    # --- 0) Regulatory regime: one reference year for the whole horizon ------
    # The profiles predate the published SI tariff acts, so the year is pinned
    # explicitly rather than resolved from the data date.
    pricing_options = dict(env.pricing_options or {})
    ref_year = pricing_options.get("pricing_reference_year", env.pricing_reference_year)
    ref_year = PRIVZETO_REFERENCNO_LETO if ref_year is None else int(ref_year)
    pricing_options["pricing_reference_year"] = ref_year
    pravila_ref = Pravila.za_leto(ref_year)

    N_STEPS = int(env.episode_length if n_steps is None else n_steps)

    horizon = slice(start_idx, start_idx + N_STEPS)
    gen = env.arr_generation[horizon]
    con = env.arr_consumption[horizon]
    smp_prices = env.arr_price[horizon]
    rel_price = env.arr_relative_price[horizon]
    dates = env.dataset.index[horizon]

    INTERVAL_MINS = int(round(env.interval_minutes))

    _, _, months_sorted, month_idx_t = month_calendar(dates)
    agreed_by_month = agreed_power_by_month(
        env, start_idx, months_sorted, do_not_use_previous_month=do_not_use_previous_month
    )
    agreed_t = [agreed_by_month[month_idx_t[t]] for t in range(N_STEPS)]
    if verbose and do_not_use_previous_month and start_idx:
        y, mo = months_sorted[0]
        print(f"Agreed power: {y}-{mo:02d} bootstrapped "
              f"({env.agreed_power_bootstrap}), previous month deliberately unused.")

    invoice_builder = None
    if generate_invoice:
        invoice_builder = InvoiceBuilder(
            dogovorjena_moc=lambda year, month: env.agreed_power_kw(year * 12 + month - 1),
            pricing_scheme=env.pricing_scheme,
            interval_minutes=INTERVAL_MINS,
            output_dir=invoice_output_dir or env.invoice_output_dir,
            run_label=invoice_run_label,
            write_monthly=True,
            write_period=True,
            pricing_reference_year=ref_year,
        )

    # --- 1) Tariff rates -----------------------------------------------------
    # Priced without dogovorjena_moc/prev_peak_kw so the result stays linear in
    # total_consumed_kwh; peak and fixed charges enter separately below.
    import_rates, export_rates, constant_costs, fixed_monthly_costs = [], [], [], []

    for t in range(N_STEPS):
        import_res = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=1.0,
            utc_date=dates[t],
            interval_minutes=INTERVAL_MINS,
            scheme=env.pricing_scheme,
            **pricing_options,
        )
        import_rates.append(import_res["variable_price_aud"])
        constant_costs.append(import_res["constant_price_aud"])

        export_res = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=-1.0,
            utc_date=dates[t],
            interval_minutes=INTERVAL_MINS,
            scheme=env.pricing_scheme,
            **pricing_options,
        )
        export_rates.append(-export_res["variable_price_aud"])

        fixed_monthly_costs.append(
            compute_prorated_fixed_charge_eur(
                dates[t], INTERVAL_MINS, scheme=env.pricing_scheme,
                dogovorjena_moc=agreed_t[t], **pricing_options,
            )
        )

    export_rates, n_export_floored = floor_export_rates(import_rates, export_rates)

    # --- 2) Model ------------------------------------------------------------
    prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

    block = add_household_physics(
        prob, env, n_steps=N_STEPS, gen=gen, con=con,
        initial_soc_kwh=initial_soc_kwh, final_soc_kwh=final_soc_kwh,
        exclusivity="binary",
    )
    P_buy, P_sell, P_spill = block.buy, block.sell, block.spill
    P_ch, P_dis, E = block.charge, block.discharge, block.soc

    # --- 2b) Excess-power ratchet --------------------------------------------
    block_arr = env.tariff_blocks[horizon]
    window_id_arr = env.reset_window_ids[horizon]
    _, _, peak_objective_terms = add_excess_power_ratchet(
        prob, env, P_buy,
        blocks=block_arr, month_idx_t=month_idx_t, months_sorted=months_sorted,
        agreed_by_month=agreed_by_month, start_idx=start_idx, n_steps=N_STEPS,
        hours=INTERVAL_MINS / 60.0, pravila=pravila_ref,
    )

    # --- 2b') Annual NET-metering settlement ---------------------------------
    # Bounded by both sums, so minimizing pins it to min(imported, exported).
    netting_objective_terms = []
    if annual_netting_rate_eur_per_kwh:
        rate_net = float(annual_netting_rate_eur_per_kwh)
        E_netted = pulp.LpVariable("E_netted", lowBound=0)
        prob += E_netted <= pulp.lpSum(P_buy)
        prob += E_netted <= pulp.lpSum(P_sell)
        netting_objective_terms.append(-rate_net * E_netted)

    # --- 2c) Objective -------------------------------------------------------
    # `constant_costs` is not here: `fixed_monthly_costs` already carries the
    # same prorated fixed charge, and adding both double-counts it.
    prob += pulp.lpSum(
        P_buy[t] * import_rates[t] - P_sell[t] * export_rates[t]
        + fixed_monthly_costs[t]
        for t in range(N_STEPS)
    ) + pulp.lpSum(peak_objective_terms) + pulp.lpSum(netting_objective_terms)

    # --- 3) Solve ------------------------------------------------------------
    if verbose:
        print(f"Solving over {N_STEPS} steps... (This may take a moment)")
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"Warning: Solver ended with status {pulp.LpStatus[prob.status]}")

    # --- 4) Extract ----------------------------------------------------------
    results = []
    cumulative_payment = 0.0
    cumulative_rl_reward = 0.0
    total_bought_kwh = 0.0
    total_sold_kwh = 0.0
    reporting_peak_kw = env.compute_seed_peak_kw(start_idx)

    for t in range(N_STEPS):
        # Drop the running peak at every window turnover, as the objective does.
        if t and window_id_arr[t] != window_id_arr[t - 1]:
            reporting_peak_kw = {b: 0.0 for b in reporting_peak_kw}

        buy_val = P_buy[t].varValue or 0.0
        sell_val = P_sell[t].varValue or 0.0
        e_val = (E[t].varValue or 0.0) if E is not None else 0.0
        ch_val = _value(P_ch[t])
        dis_val = _value(P_dis[t])
        spill_val = P_spill[t].varValue or 0.0
        total_bought_kwh += buy_val
        total_sold_kwh += sell_val

        interval_cost_data = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=buy_val - sell_val,
            utc_date=dates[t],
            interval_minutes=INTERVAL_MINS,
            scheme=env.pricing_scheme,
            dogovorjena_moc=agreed_t[t],
            prev_peak_kw=reporting_peak_kw,
            include_raw=invoice_builder is not None,
            **pricing_options,
        )
        reporting_peak_kw = dict(interval_cost_data["new_peak_kw"])
        if invoice_builder is not None:
            invoice_builder.add_interval(interval_cost_data)

        fixed_cost = interval_cost_data["constant_price_aud"]
        variable_cost = interval_cost_data["variable_price_aud"]
        cumulative_payment += fixed_cost + variable_cost

        soc_norm = e_val / env.battery_capacity_kwh if env.battery_capacity_kwh > 0 else 0.0
        battery_delta_kwh = (
            ch_val * env.charge_efficiency - dis_val / env.discharge_efficiency
        )

        step_reward = env.compute_reward(
            soc_norm, rel_price[t], battery_delta_kwh, variable_cost
        )
        cumulative_rl_reward += step_reward

        results.append({
            "Step": t,
            "Date": dates[t],
            "SMP_MWh": smp_prices[t],
            "Generation": gen[t],
            "Consumption": con[t],
            "Charge_kW": ch_val,
            "Discharge_kW": dis_val,
            "SOC_kWh": e_val,
            "SOC_%": soc_norm * 100,
            "Spill_kW": spill_val,
            "Import_Rate_kWh": import_rates[t],
            "Export_Rate_kWh": export_rates[t],
            "Step_Cost": fixed_cost + variable_cost,
            "Cum_Cost": cumulative_payment,
            "Step_RL_Reward": step_reward,
            "Cum_RL_Reward": cumulative_rl_reward,
            "Energy_Component_EUR": interval_cost_data["energy_component_eur"],
            "Power_Component_EUR": interval_cost_data["power_component_eur"],
        })

    df_results = pd.DataFrame(results)
    # A non-Optimal status means the numbers below are not a solution.
    df_results.attrs["solver_status"] = pulp.LpStatus[prob.status]
    df_results.attrs["export_rate_floored_intervals"] = int(n_export_floored)

    # Booked once on the closing interval, so Cum_Cost.iloc[-1] is the bill for
    # the whole horizon.
    df_results["Netting_Credit_EUR"] = 0.0
    if annual_netting_rate_eur_per_kwh and N_STEPS > 0:
        credit = float(annual_netting_rate_eur_per_kwh) * min(
            total_bought_kwh, total_sold_kwh
        )
        last = df_results.index[-1]
        df_results.loc[last, "Netting_Credit_EUR"] = credit
        df_results.loc[last, "Step_Cost"] -= credit
        df_results.loc[last, "Cum_Cost"] -= credit
        cumulative_payment -= credit

    if invoice_builder is not None and N_STEPS > 0:
        invoice_builder.finalize(period_label=f"{dates[0]:%Y-%m-%d}_{dates[-1]:%Y-%m-%d}")

    if verbose:
        print("\n--- MILP Optimization Complete ---")
        print(f"Total Electricity Cost (Inc GST & Fixed): {cumulative_payment:.4f}")
        print(f"Total RL Equivalent Reward: {cumulative_rl_reward:.4f}")

    return df_results


def _value(x):
    """Variable value, constant, or zero."""
    if isinstance(x, (int, float)):
        return float(x)
    return float(x.varValue or 0.0)


# ---------------------------------------------------------------------------
# Reading a solved trajectory back
# ---------------------------------------------------------------------------
def net_grid_kwh(df):
    """Net grid flow per interval: positive is imported, negative exported.

    The energy balance makes this exact:

        gen + P_buy + P_dis == con + P_sell + P_ch + P_spill
        =>  P_buy - P_sell  == con + P_ch + P_spill - gen - P_dis
    """
    return (
        df["Consumption"].to_numpy(dtype=float)
        + df["Charge_kW"].to_numpy(dtype=float)
        + df["Spill_kW"].to_numpy(dtype=float)
        - df["Generation"].to_numpy(dtype=float)
        - df["Discharge_kW"].to_numpy(dtype=float)
    )


def effective_profile(df, index, *, consumption_column, generation_column):
    """MILP trajectory -> the (consumption, generation) pair the settlement sees.

    A positive net is billed as consumption, a negative one as generation. Both
    settlement paths net within the interval, so the bill is unchanged.
    """
    net = net_grid_kwh(df)
    n = len(net)
    return pd.DataFrame(
        {
            consumption_column: np.maximum(net, 0.0),
            generation_column: np.maximum(-net, 0.0),
        },
        index=index[:n],
    )


def summarize_trajectory(
    df,
    *,
    capacity_kwh,
    hours_per_step,
    annualize=1.0,
    charge_efficiency=CHARGE_EFFICIENCY,
    discharge_efficiency=DISCHARGE_EFFICIENCY,
    blocks=None,
    tariff_blocks=None,
    efc_convention="store",
):
    """Per-run energy and power metrics from one solved trajectory.

    `blocks` is the per-row network tariff block; given it, the peak import is
    also reported per block, which is the power the excess charge is measured
    against. `tariff_blocks` is the set of block ids to report.

    `EFC_AC_Legacy` always counts AC-side discharge only. What
    `Equivalent_Full_Cycles` counts is `efc_convention`: "store" measures energy
    through the store, charge and discharge averaged with the efficiencies
    applied, and "ac" repeats the legacy AC-side figure.
    """
    if efc_convention not in ("store", "ac"):
        raise ValueError(f"efc_convention must be 'store' or 'ac', got {efc_convention!r}")
    net = net_grid_kwh(df)
    charged = float(df["Charge_kW"].sum())
    discharged = float(df["Discharge_kW"].sum())
    pv_surplus = np.maximum(
        df["Generation"].to_numpy(dtype=float) - df["Consumption"].to_numpy(dtype=float), 0.0
    )
    grid_charged = np.maximum(df["Charge_kW"].to_numpy(dtype=float) - pv_surplus, 0.0)

    # Energy moved through the STORE, not through the inverter.
    stored_in_kwh = charged * charge_efficiency
    stored_out_kwh = discharged / discharge_efficiency
    efc = (stored_in_kwh + stored_out_kwh) / (2.0 * capacity_kwh) if capacity_kwh > 0 else 0.0

    out = {
        "Capacity_kWh": float(capacity_kwh),
        "Cost_EUR": float(df["Cum_Cost"].iloc[-1]) * annualize,
        "Charged_kWh": charged * annualize,
        "Discharged_kWh": discharged * annualize,
        "Stored_Out_kWh": stored_out_kwh * annualize,
        "Grid_Charged_kWh": float(grid_charged.sum()) * annualize,
        "Import_kWh": float(net.clip(min=0).sum()) * annualize,
        "Export_kWh": float((-net).clip(min=0).sum()) * annualize,
        "Curtailed_kWh": float(df["Spill_kW"].sum()) * annualize,
        "Peak_Import_kW": float(net.max()) / hours_per_step,
        "EFC_AC_Legacy": (discharged * annualize / capacity_kwh) if capacity_kwh > 0 else 0.0,
    }
    out["Equivalent_Full_Cycles"] = (
        efc * annualize if efc_convention == "store" else out["EFC_AC_Legacy"]
    )

    if blocks is not None:
        # Clipped at zero: a block whose intervals are all net export has a peak
        # import of 0 kW, not a negative one. On the annual maximum the clip
        # never binds, so the largest block reproduces Peak_Import_kW exactly.
        import_kw = np.maximum(net, 0.0) / hours_per_step
        ids = tariff_blocks if tariff_blocks is not None else sorted(set(int(b) for b in blocks))
        out.update({
            f"Peak_Import_B{b}_kW": float(np.max(import_kw[blocks == b], initial=0.0))
            for b in ids
        })
    return out
