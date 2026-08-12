"""Perfect-foresight MILP benchmark for a HouseholdEnvironment episode.

Single shared implementation used by every notebook. It reads all battery and
pricing parameters off the environment, so the MILP always prices exactly like
the RL environment it is benchmarked against. Change the formulation HERE ONLY.
"""

from datetime import date

import numpy as np
import pandas as pd

from Pricing_Functions import (
    InvoiceBuilder,
    PRIVZETO_REFERENCNO_LETO,
    calculate_interval_price,
    compute_prorated_fixed_charge_eur,
    moc_za_mesec,
)


def run_milp_benchmark(
    env,
    use_discrete_actions=False,
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
    """
    Runs a MILP benchmark over a HouseholdEnvironment episode.

    Parameters
    ----------
    env : HouseholdEnvironment
        Fully constructed environment; all battery/pricing parameters
        (battery_capacity_kwh, charge_efficiency, pricing_scheme,
        pricing_options, contracted_power_kw, ...) are read off the env. The SI
        regulatory regime is pinned to env.pricing_reference_year, defaulting to
        2026 when the env leaves it unset (dataset timestamps such as 2012 have
        no published tariff rates).
    use_discrete_actions : bool
        False -> continuous formulation (P_buy, P_sell, P_ch, P_dis), matching
        the environment's native continuous action.
        True  -> choose one of env.action_space.n discrete actions with binary
        vars, matching the legacy discrete action set.
        Both share the same energy balance, battery dynamics and curtailment
        variable; discrete only adds binaries that gate P_ch / P_dis. Continuous
        is therefore a true relaxation of discrete and can never cost more.
    start_idx : int
        First dataset row of the horizon.
    n_steps : int or None
        Horizon length; defaults to env.episode_length.
    initial_soc_kwh : float or None
        Stored energy at the first interval. Defaults to the environment's own
        reset SOC (half the capacity), so existing calls are unchanged. Set it
        explicitly when the starting charge must not scale with the capacity --
        a battery-size sweep, for instance, otherwise hands every larger
        battery a bigger block of free energy at t=0 and reads that gift as a
        saving. It also lets a long horizon be solved in consecutive chunks:
        pass the previous chunk's final SOC and the trajectory stays continuous.
    final_soc_kwh : float or None
        Lower bound on the stored energy left at the end of the horizon. None
        (the default) leaves the terminal SOC free, so the solver empties the
        battery into the last few intervals. Set it equal to `initial_soc_kwh`
        to forbid that: the trajectory then has to give back whatever it
        started with, and the reported cost contains no revenue from simply
        selling off the opening charge.
    verbose : bool
        Print progress / summary lines (turn off for per-user loops).
    problem_name : str
        Name handed to pulp.LpProblem (cosmetic; useful when solving many users).
    solver : pulp solver or None
        Defaults to pulp.PULP_CBC_CMD(msg=False).
    annual_netting_rate_eur_per_kwh : float or None
        NET-metering settlement rate (VAT-inclusive supplier energy price,
        EUR/kWh). None (the default) leaves the bill purely interval-by-interval,
        which is what every non-NET-metering price list does.

        The NET-metering lists (`GENI_NETMETERING`, consents granted before
        2024) credit nothing per interval -- `si_obracun._cena_oddaje` returns
        0.0 for `TipOdkupa.NET_METERING` -- and instead net exported against
        imported energy once a year, on the *supplier energy* component only:
        network charges and levies stay on the gross metered offtake under the
        2024 network act. Setting this rate adds that annual settlement to the
        objective as

            credit = rate * min(sum P_buy, sum P_sell)

        (one free variable bounded by both sums; minimizing drives it to the
        min), so the solver optimizes against the bill the household actually
        receives instead of one where every exported kWh is worthless. The same
        credit is applied to the last row of the returned trajectory -- see the
        `Netting_Credit_EUR` column -- so `Cum_Cost.iloc[-1]` stays the annual
        bill. Exports beyond the annual import volume earn nothing here, which
        is what makes the sizing curve finite under this list. `generate_invoice`
        does not see the settlement -- InvoiceBuilder is fed interval by
        interval, so an emitted invoice is short by exactly this credit.
    generate_invoice : bool
        Emit monthly + whole-period line-item invoices for the solved
        trajectory. The bill is built during the extraction pass below, off the
        same re-priced intervals that produce the cost, so the invoice always
        reconciles with the returned total. Switches those re-pricing calls to
        `include_raw=True`, which the builder needs to read the per-interval
        si_obracun breakdown.
    invoice_output_dir : path-like or None
        Where the invoice CSVs land; defaults to the env's invoice output dir.
    invoice_run_label : str
        Filename prefix for the emitted invoices.

    Returns
    -------
    pandas.DataFrame
        One row per step with energy flows, SOC, per-step and cumulative cost
        and the RL-equivalent reward.

    Notes
    -----
    The solved trajectory is reachable in the environment: feed
    `Charge_kW - Discharge_kW` as the battery setpoint and `Spill_kW` as the
    curtailment to a `HouseholdEnvironment(action_mode="continuous",
    allow_curtailment=True)` and it reproduces `Cum_Cost` exactly. That requires
    curtailment to be enabled -- without it the agent must export its surplus
    even when export is loss-making, and cannot match this cost.
    """
    import pulp

    from si_obracun import Pravila
    from si_cas import v_lokalni_cas, je_visja_sezona

    if verbose:
        print("Building MILP Model...")

    # -------------------------------------------------------------------------
    # 0) REGULATORY REGIME -- ONE regime for the whole horizon
    # -------------------------------------------------------------------------
    # The datasets carry timestamps for which no SI tariff act exists, so
    # resolving rules from the data date raises "Za 2012-06-30 ni nalozenih
    # tarifnih postavk". The MILP therefore always prices under an explicit
    # reference year -- the env's, or 2026 when the env leaves it open -- and
    # pins it into pricing_options so every pricing call below (energy rates,
    # fixed monthly charge, reporting pass) uses exactly the same rates as the
    # block/peak rates derived here.
    pricing_options = dict(env.pricing_options or {})
    ref_year = pricing_options.get("pricing_reference_year", env.pricing_reference_year)
    ref_year = PRIVZETO_REFERENCNO_LETO if ref_year is None else int(ref_year)
    pricing_options["pricing_reference_year"] = ref_year
    pravila_ref = Pravila.za_leto(ref_year)

    # Local name is N_STEPS (not T) so notebooks that alias torch as T are safe.
    N_STEPS = int(env.episode_length if n_steps is None else n_steps)

    horizon = slice(start_idx, start_idx + N_STEPS)
    gen = env.arr_generation[horizon]
    con = env.arr_consumption[horizon]
    smp_prices = env.arr_price[horizon]
    rel_price = env.arr_relative_price[horizon]
    dates = env.dataset.index[horizon]

    INTERVAL_MINS = int(round(env.interval_minutes))

    # Calendar month of every interval, and the agreed billing power in force in
    # it. The agreed power is a per-month constant (the household may revise it
    # on the 1st), so it is resolved once per interval here and every pricing
    # call, constraint and objective term below reads the same figure.
    #
    # A run starting part-way into the dataset lands in a month that has a real
    # predecessor on file, and by default it is billed on it -- the meter was
    # running before the solver was pointed at this window.
    # `do_not_use_previous_month=True` refuses that history and treats the run's
    # first month as a cold start instead; see
    # `HouseholdEnvironment.agreed_power_for_run`.
    lok_t = [v_lokalni_cas(dates[t]) for t in range(N_STEPS)]
    month_key_t = [(lok_t[t].year, lok_t[t].month) for t in range(N_STEPS)]
    months_sorted = sorted(set(month_key_t))
    month_idx_t = [months_sorted.index(month_key_t[t]) for t in range(N_STEPS)]
    run_schedule = env.agreed_power_for_run(
        start_idx, do_not_use_previous_month=do_not_use_previous_month
    )
    agreed_by_month = {
        m: moc_za_mesec(run_schedule, y * 12 + mo - 1)
        for m, (y, mo) in enumerate(months_sorted)
    }
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

    # -------------------------------------------------------------------------
    # 1) PRE-CALCULATE TARIFF RATES
    # -------------------------------------------------------------------------
    # These calls are deliberately made WITHOUT dogovorjena_moc/prev_peak_kw so
    # they stay strictly linear in total_consumed_kwh (required for the +-1kWh
    # unit-rate trick) -- peak/excess charges are handled separately below via
    # explicit MILP variables/constraints, and fixed monthly charges via
    # compute_prorated_fixed_charge_eur (a pure function of the calendar +
    # contract, independent of any decision variable).
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

    # P_buy and P_sell are independent nonnegative variables with no
    # complementarity constraint between them, which is exact only while an
    # exported kWh is worth no more than an imported one costs. Raising both by
    # the same delta always satisfies the energy balance and moves the objective
    # by (import_rate - export_rate), so where that is negative the model is
    # UNBOUNDED -- it buys and sells the same kWh forever.
    #
    # Real contracts do not offer that, but a real COMBINATION does: a price list
    # with no buyback (`tip_odkupa=NI` -> export rate 0.0) in an interval whose
    # delivered import rate is negative, which SIPX-linked lists reach whenever
    # the market price goes below about -0.04 EUR/kWh. GEN-I's plain dynamic list
    # caps SIPX above at 0.22 EUR/kWh and leaves it unbounded below, so 2024 has
    # such intervals and the solve returns "Unbounded" with meaningless values.
    #
    # Flooring the export rate at the import rate removes the loop without adding
    # a binary per interval (which would double the model). The round trip becomes
    # exactly neutral rather than profitable, and because curtailment is free
    # (P_spill, bounded by generation) the optimum never exports in those
    # intervals anyway -- so the trajectory and the cost are the true ones.
    n_export_floored = sum(1 for t in range(N_STEPS) if export_rates[t] > import_rates[t])
    if n_export_floored:
        export_rates = [min(e, i) for e, i in zip(export_rates, import_rates)]

    # -------------------------------------------------------------------------
    # 2) DEFINE MILP
    # -------------------------------------------------------------------------
    prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

    # Shared variables
    P_buy = [pulp.LpVariable(f"P_buy_{t}", lowBound=0) for t in range(N_STEPS)]
    P_sell = [pulp.LpVariable(f"P_sell_{t}", lowBound=0) for t in range(N_STEPS)]
    E = [
        pulp.LpVariable(f"E_{t}", lowBound=0, upBound=env.battery_capacity_kwh)
        for t in range(N_STEPS + 1)
    ]
    # env.max_charge_kwh / max_discharge_kwh cap the change in STORED energy, so
    # the AC-side bounds here carry the efficiency factor -- exactly what
    # Basic_Functions.max_charge_now / max_discharge_now enforce in the
    # environment. Charge and discharge efficiency stay independent.
    max_charge_ac = env.max_charge_kwh / env.charge_efficiency
    max_discharge_ac = env.max_discharge_kwh * env.discharge_efficiency
    P_ch = [
        pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=max_charge_ac)
        for t in range(N_STEPS)
    ]
    P_dis = [
        pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=max_discharge_ac)
        for t in range(N_STEPS)
    ]
    # Curtailment of local production. Bounded by that interval's own generation
    # -- it can only switch PV off, never absorb imported energy. Without this
    # upper bound the solver buys unlimited energy at negative prices and dumps
    # it here, which is not a dispatch any real system (or the environment) can
    # perform. Identical in both formulations.
    P_spill = [
        pulp.LpVariable(f"P_spill_{t}", lowBound=0, upBound=float(gen[t]))
        for t in range(N_STEPS)
    ]

    # Initial SOC
    if initial_soc_kwh is None:
        initial_soc_kwh = env.battery_capacity_kwh / 2.0
    prob += (E[0] == min(max(0.0, float(initial_soc_kwh)), env.battery_capacity_kwh))

    # Terminal SOC floor. Without it the last intervals of the horizon are a
    # free source of energy, and any starting charge shows up as a saving.
    if final_soc_kwh is not None:
        prob += (E[N_STEPS] >= min(max(0.0, float(final_soc_kwh)), env.battery_capacity_kwh))

    for t in range(N_STEPS):
        # Battery dynamics
        prob += (
            E[t + 1]
            == E[t]
            + P_ch[t] * env.charge_efficiency
            - P_dis[t] / env.discharge_efficiency
        )
        # Energy balance with curtailment
        prob += (
            gen[t] + P_buy[t] + P_dis[t]
            == con[t] + P_sell[t] + P_ch[t] + P_spill[t]
        )

    if not use_discrete_actions:
        # ---------------------------------------------------------------------
        # CONTINUOUS MODE -- P_ch[t] - P_dis[t] is the environment's setpoint
        # ---------------------------------------------------------------------
        # Forbid charging and discharging in the same interval. A battery
        # physically cannot, and the environment cannot express it either (one
        # signed setpoint per step). Left unconstrained the LP exploits the
        # round-trip loss as an energy sink whenever the import rate goes
        # negative, producing a trajectory no agent can reproduce. The discrete
        # branch below gets this for free from its one-action-per-step binaries.
        B_charging = [pulp.LpVariable(f"B_charging_{t}", cat="Binary") for t in range(N_STEPS)]
        for t in range(N_STEPS):
            prob += P_ch[t] <= max_charge_ac * B_charging[t]
            prob += P_dis[t] <= max_discharge_ac * (1 - B_charging[t])

    else:
        # ---------------------------------------------------------------------
        # DISCRETE MODE (choose one of env.action_space.n legacy actions)
        # ---------------------------------------------------------------------
        n_actions = int(env.action_space.n)
        A = [
            [pulp.LpVariable(f"A_{t}_{a}", cat="Binary") for a in range(n_actions)]
            for t in range(N_STEPS)
        ]

        for t in range(N_STEPS):
            # one action per step
            prob += pulp.lpSum(A[t][a] for a in range(n_actions)) == 1

            # action semantics (see Environment.ACTION_* constants)
            # 0: charge (PV + grid)
            # 1: charge (PV only)
            # 2: discharge to house / no charge
            # 3: discharge to house + grid / no charge
            # 4: idle / no battery use

            # charging only allowed in actions 0 and 1
            prob += P_ch[t] <= max_charge_ac * (A[t][0] + A[t][1])

            # discharging only allowed in actions 2 and 3
            prob += P_dis[t] <= max_discharge_ac * (A[t][2] + A[t][3])

    # -------------------------------------------------------------------------
    # 2b) PEAK / EXCESS-POWER (ratchet) VARIABLES AND CONSTRAINTS
    # -------------------------------------------------------------------------
    # One P_peak/Excess variable per (block, month) pair that actually occurs in
    # the horizon, ratcheted across consecutive months within the same
    # reset-window (env.peak_reset_months), floored at the historical seed peak
    # entering the horizon (env.compute_seed_peak_kw). This mirrors the RL
    # environment's per-step marginal ratchet charge exactly (see the
    # telescoping-sum proof in si_konica.py) -- both formulations charge the
    # same total for the same trajectory.
    #
    # The seed only belongs to the reset window the horizon STARTS in. A window
    # that opens inside the horizon starts its peak at zero by definition of a
    # reset, so flooring it at the seed too would carry a peak across exactly the
    # boundary the reset exists to break -- and would then charge the next window
    # only the increment over that stale peak instead of its own full excess.
    block_arr = env.tariff_blocks[horizon]
    window_id_arr = env.reset_window_ids[horizon]
    seed_peak_kw = env.compute_seed_peak_kw(start_idx)
    seed_window = int(window_id_arr[0])

    month_window = {}
    for t in range(N_STEPS):
        month_window.setdefault(month_idx_t[t], int(window_id_arr[t]))

    # A ratchet window wider than one month carries a peak across a month
    # boundary, but the agreed power it is charged against may change on exactly
    # that boundary -- and then the telescoping increment below is measured
    # against two different contracts and no longer sums to the real charge. The
    # excess charge is settled monthly in the Akt, so peak_reset_months=1 is the
    # regime this can be exact in; anything else needs a constant agreed power.
    for w in set(month_window.values()):
        in_window = [m for m, mw in month_window.items() if mw == w]
        if len({tuple(sorted(agreed_by_month[m].items())) for m in in_window}) > 1:
            raise ValueError(
                f"Ratchet window {w} spans months with different agreed billing power "
                f"({[months_sorted[m] for m in in_window]}). Set peak_reset_months=1 so "
                f"each window is one month, or build the environment with a constant "
                f"contracted_power_kw."
            )

    ure = INTERVAL_MINS / 60.0
    occurring = sorted({(int(block_arr[t]), month_idx_t[t]) for t in range(N_STEPS)})
    P_peak_month = {(b, m): pulp.LpVariable(f"P_peak_b{b}_m{m}", lowBound=0) for (b, m) in occurring}
    Excess_month = {(b, m): pulp.LpVariable(f"Excess_b{b}_m{m}", lowBound=0) for (b, m) in occurring}

    for t in range(N_STEPS):
        prob += P_peak_month[(int(block_arr[t]), month_idx_t[t])] >= P_buy[t] / ure

    last_var_by_block, last_window_by_block = {}, {}
    for (b, m) in occurring:
        w = month_window[m]
        if b in last_var_by_block and last_window_by_block[b] == w:
            prob += P_peak_month[(b, m)] >= last_var_by_block[b]
        elif w == seed_window:
            prob += P_peak_month[(b, m)] >= seed_peak_kw.get(b, 0.0)
        last_var_by_block[b] = P_peak_month[(b, m)]
        last_window_by_block[b] = w

        prob += Excess_month[(b, m)] >= P_peak_month[(b, m)] - agreed_by_month[m].get(b, 0.0)

    # Incremental (telescoping) objective contribution per (block, month), using
    # each month's own season-correct rate (only block 1's rate depends on season).
    peak_objective_terms = []
    prev_excess_by_block, prev_window_by_block = {}, {}
    for (b, m) in occurring:
        y, mo = months_sorted[m]
        w = month_window[m]
        vs = je_visja_sezona(date(y, mo, 1))
        rate_bm = pravila_ref.omreznina.postavka_moc(b, vs)
        faktor = pravila_ref.omreznina.faktor_presezne_moci

        if b in prev_excess_by_block and prev_window_by_block[b] == w:
            prev_term = prev_excess_by_block[b]
        elif w == seed_window:
            prev_term = max(0.0, seed_peak_kw.get(b, 0.0) - agreed_by_month[m].get(b, 0.0))
        else:
            prev_term = 0.0          # a fresh window pays its own excess in full

        peak_objective_terms.append((Excess_month[(b, m)] - prev_term) * rate_bm * faktor)
        prev_excess_by_block[b] = Excess_month[(b, m)]
        prev_window_by_block[b] = w

    # -------------------------------------------------------------------------
    # 2b') ANNUAL NET-METERING SETTLEMENT (only for NET-metering price lists)
    # -------------------------------------------------------------------------
    # Bounded by both sums, so minimizing pins it to min(imported, exported) --
    # exactly the volume a NET-metering contract nets off at year end.
    netting_objective_terms = []
    if annual_netting_rate_eur_per_kwh:
        rate_net = float(annual_netting_rate_eur_per_kwh)
        E_netted = pulp.LpVariable("E_netted", lowBound=0)
        prob += E_netted <= pulp.lpSum(P_buy)
        prob += E_netted <= pulp.lpSum(P_sell)
        netting_objective_terms.append(-rate_net * E_netted)

    # -------------------------------------------------------------------------
    # 2c) OBJECTIVE (applies identically to both continuous and discrete modes)
    # -------------------------------------------------------------------------
    # `constant_costs` is deliberately NOT in here. It is the same prorated fixed
    # monthly charge `fixed_monthly_costs` carries -- calculate_interval_price
    # returns it as `constant_price_aud` whether or not dogovorjena_moc was
    # passed -- so adding both double-counts the supplier's monthly fee. It is a
    # decision-independent constant either way, so no solution ever changed; it
    # only made the objective value disagree with the reported Cum_Cost.
    prob += pulp.lpSum(
        P_buy[t] * import_rates[t] - P_sell[t] * export_rates[t]
        + fixed_monthly_costs[t]
        for t in range(N_STEPS)
    ) + pulp.lpSum(peak_objective_terms) + pulp.lpSum(netting_objective_terms)

    # -------------------------------------------------------------------------
    # 3) SOLVE
    # -------------------------------------------------------------------------
    if verbose:
        print(f"Solving over {N_STEPS} steps... (This may take a moment)")
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"Warning: Solver ended with status {pulp.LpStatus[prob.status]}")

    # -------------------------------------------------------------------------
    # 4) EXTRACT RESULTS
    # -------------------------------------------------------------------------
    results = []
    cumulative_payment = 0.0
    cumulative_rl_reward = 0.0
    total_bought_kwh = 0.0
    total_sold_kwh = 0.0
    reporting_peak_kw = env.compute_seed_peak_kw(start_idx)

    for t in range(N_STEPS):
        # The reporting pass has to drop the running peak wherever the ratchet
        # window turns over, exactly as the objective above does -- otherwise the
        # per-interval Power_Component_EUR carries a stale peak into a new month
        # and disagrees with the cost the model actually minimized.
        if t and window_id_arr[t] != window_id_arr[t - 1]:
            reporting_peak_kw = {b: 0.0 for b in reporting_peak_kw}

        buy_val = P_buy[t].varValue or 0.0
        sell_val = P_sell[t].varValue or 0.0
        e_val = E[t].varValue or 0.0
        ch_val = P_ch[t].varValue or 0.0
        dis_val = P_dis[t].varValue or 0.0
        spill_val = P_spill[t].varValue or 0.0
        total_bought_kwh += buy_val
        total_sold_kwh += sell_val

        action_val = (
            int(np.argmax([(A[t][a].varValue or 0.0) for a in range(n_actions)]))
            if use_discrete_actions
            else None
        )

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

        row = {
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
        }

        if use_discrete_actions:
            row["Action"] = action_val

        results.append(row)

    df_results = pd.DataFrame(results)
    # Provenance for the caller: a non-Optimal status means the numbers below are
    # not a solution, and the floored count says how many intervals needed the
    # export-rate correction above.
    df_results.attrs["solver_status"] = pulp.LpStatus[prob.status]
    df_results.attrs["export_rate_floored_intervals"] = int(n_export_floored)

    # Annual NET-metering settlement, booked once on the closing interval so
    # Cum_Cost.iloc[-1] is the bill for the whole horizon. Zero (and the column
    # all-zero) for every price list that credits exports per interval.
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
