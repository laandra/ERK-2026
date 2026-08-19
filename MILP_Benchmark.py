"""Perfect-foresight MILP benchmark for a HouseholdEnvironment episode.

Every battery and pricing parameter is read off the environment, so the MILP
prices exactly like the environment it is benchmarked against.
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
    """Cost-minimal battery dispatch over one episode, as a per-step DataFrame.

    `initial_soc_kwh` / `final_soc_kwh` pin the stored energy at the horizon
    ends; `annual_netting_rate_eur_per_kwh` adds the NET-metering settlement
    credit = rate * min(sum P_buy, sum P_sell) to the objective.
    """
    import pulp

    from si_obracun import Pravila
    from si_cas import v_lokalni_cas, je_visja_sezona

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

    # The agreed billing power is a per-month constant, resolved once here so
    # every constraint, objective term and pricing call below reads one figure.
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

    # Export rate floored at the import rate. Without it, an interval whose
    # delivered import rate is negative makes the buy/sell round trip profitable
    # and the LP unbounded. Flooring makes it exactly neutral, and curtailment
    # is free, so the optimum never exports there anyway.
    n_export_floored = sum(1 for t in range(N_STEPS) if export_rates[t] > import_rates[t])
    if n_export_floored:
        export_rates = [min(e, i) for e, i in zip(export_rates, import_rates)]

    # --- 2) Model ------------------------------------------------------------
    prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

    P_buy = [pulp.LpVariable(f"P_buy_{t}", lowBound=0) for t in range(N_STEPS)]
    P_sell = [pulp.LpVariable(f"P_sell_{t}", lowBound=0) for t in range(N_STEPS)]
    E = [
        pulp.LpVariable(f"E_{t}", lowBound=0, upBound=env.battery_capacity_kwh)
        for t in range(N_STEPS + 1)
    ]
    # env caps the change in STORED energy, so the AC-side bounds here carry
    # the efficiency factor.
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
    # Curtailment, bounded by the interval's own generation: it can switch PV
    # off, never absorb imported energy.
    P_spill = [
        pulp.LpVariable(f"P_spill_{t}", lowBound=0, upBound=float(gen[t]))
        for t in range(N_STEPS)
    ]

    # Initial SOC
    if initial_soc_kwh is None:
        initial_soc_kwh = env.battery_capacity_kwh / 2.0
    prob += (E[0] == min(max(0.0, float(initial_soc_kwh)), env.battery_capacity_kwh))

    # Terminal SOC floor, so the opening charge cannot be sold off as a saving.
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
        # Continuous mode: P_ch[t] - P_dis[t] is the environment's setpoint.
        # Simultaneous charge and discharge is forbidden explicitly -- with a
        # negative import rate the LP would otherwise use the round-trip loss as
        # an energy sink. The discrete branch gets this from its binaries.
        B_charging = [pulp.LpVariable(f"B_charging_{t}", cat="Binary") for t in range(N_STEPS)]
        for t in range(N_STEPS):
            prob += P_ch[t] <= max_charge_ac * B_charging[t]
            prob += P_dis[t] <= max_discharge_ac * (1 - B_charging[t])

    else:
        # Discrete mode: one of env.action_space.n legacy actions per step.
        n_actions = int(env.action_space.n)
        A = [
            [pulp.LpVariable(f"A_{t}_{a}", cat="Binary") for a in range(n_actions)]
            for t in range(N_STEPS)
        ]

        for t in range(N_STEPS):
            # one action per step
            prob += pulp.lpSum(A[t][a] for a in range(n_actions)) == 1


            # charging only allowed in actions 0 and 1
            prob += P_ch[t] <= max_charge_ac * (A[t][0] + A[t][1])

            # discharging only allowed in actions 2 and 3
            prob += P_dis[t] <= max_discharge_ac * (A[t][2] + A[t][3])

    # --- 2b) Excess-power ratchet --------------------------------------------
    # One peak variable per (block, month) occurring in the horizon, ratcheted
    # within a reset window and floored at the seed peak -- but only in the
    # window the horizon starts in; a window opening inside it resets to zero.
    block_arr = env.tariff_blocks[horizon]
    window_id_arr = env.reset_window_ids[horizon]
    seed_peak_kw = env.compute_seed_peak_kw(start_idx)
    seed_window = int(window_id_arr[0])

    month_window = {}
    for t in range(N_STEPS):
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

    # Telescoping contribution per (block, month), at each month's own rate.
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
