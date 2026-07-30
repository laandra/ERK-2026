### Basic_Functions.py - Updated to accept parameters as function arguments

import pandas as pd
import numpy as np
from datetime import date
from enum import Enum
from types import SimpleNamespace

from Pricing_Functions import calculate_interval_price

### Helper Functions
def BatMaxPraTrenutno(s, BatUcinkovitost, BatMaxPraznjenje):
    """Vrne maksimalno energijo ki jo lahko baterija odda v tem koraku"""
    return BatUcinkovitost * min(s.Baterija, BatMaxPraznjenje)

def BatMaxPolTrenutno(s, BatUcinkovitost, BatMaxPolnjenje, BatKapaciteta):
    """Vrne maksimalno energijo ki jo lahko baterija sprejme v tem koraku"""
    return (1/BatUcinkovitost) * min(BatKapaciteta - s.Baterija, BatMaxPolnjenje)

def PaneliOdvec(PaneliProizvodnja, Poraba):
    """Energija ki jo lahko paneli oddajo bateriji ali omrežju"""
    if PaneliProizvodnja > Poraba:
        return PaneliProizvodnja - Poraba
    else:
        return 0

def BaterijaSprememba(PaneliBaterija, OmrezjeBaterija, BaterijaDom, BaterijaOmrezje, BatUcinkovitost):
    """Sprememba baterije v tem koraku"""
    return BatUcinkovitost * (PaneliBaterija + OmrezjeBaterija) - (1/BatUcinkovitost) * (BaterijaDom + BaterijaOmrezje)

### State and Action Classes
class State(object):
    """Stanja našega agenta"""
    def __init__(self, CenaEl=0, CenaElMed=0, CenaElRel=1, Baterija=0, BaterijaProp=0.5, Generiranje=0,
                 Poraba=0, Placilo=0, Korak=0):
        self.CenaEl = CenaEl            # Cena elektrike eur/kWh
        self.CenaElMed = CenaElMed      # Povprečna cena elektrike npr v zadnjem mesecu cena elektrike eur/kWh
        self.CenaElRel = CenaElRel      # Relativna cena elektrike = CenaEl/CenaElMed
        self.Baterija = Baterija        # Stanje baterije v kWh
        self.BaterijaProp = BaterijaProp# Proporcionalno stanje baterije v %
        self.Generiranje = Generiranje  # Količina generirane elektrike kWh
        self.Poraba = Poraba            # Količina porabljene elektrike kWh
        self.Placilo = Placilo          # Plačilo za porabljeno elektriko
        self.Korak = Korak              # Trenutni korak

    def to_array(self):
        return np.array([self.CenaEl, self.CenaElMed, self.CenaElRel, self.Baterija, self.BaterijaProp,
                         self.Generiranje, self.Poraba, self.Placilo, self.Korak])

class Action(Enum):
    """Akcije našega agenta"""
    KUPI_POLNI = 0      # Kupi energijo da napajaš hišo in baterijo
    KUPI_HISA = 1       # Kupi energijo da napajaš hišo
    BAT_HISA = 2        # Napajaš hišo iz baterije
    BAT_PRODAJ = 3      # Napajaš hišo in prodajaš iz baterije
    # BAT_POCIVAJ = 4     #Ne uporabi baterije


### MILP Benchmark ###########################################################
# Single, shared implementation of the perfect-foresight MILP benchmark, used
# by smart_home_nanogrid_DQN.ipynb, global_model_DQN.ipynb,
# cluster_model_DQN.ipynb and cluster_sweep_analysis/sweep_analysis.ipynb.
# Change the formulation HERE ONLY -- all notebooks call into this function.

def run_milp_benchmark(
    env,
    use_discrete_actions=False,
    start_idx=0,
    n_steps=None,
    verbose=True,
    problem_name="Household_Microgrid_Optimization",
    solver=None,
    generate_invoice=False,
    invoice_output_dir=None,
    invoice_run_label="milp_eval",
):
    """
    Runs a MILP benchmark over a HouseholdEnvironment episode.

    Parameters
    ----------
    env : HouseholdEnvironment
        Fully constructed environment; all battery/pricing parameters
        (bat_kapaciteta, bat_ucinkovitost, pricing_scheme, pricing_options,
        dogovorjena_moc, ...) are read off the env, so the MILP always prices
        exactly like the RL environment it is benchmarked against. The SI
        regulatory regime is pinned to env.pricing_reference_year, defaulting
        to 2026 when the env leaves it unset (dataset timestamps such as 2012
        have no published tariff rates).
    use_discrete_actions : bool
        False -> continuous formulation (P_buy, P_sell, P_ch, P_dis).
        True  -> choose one of env.action_space.n discrete actions with binary vars.
    start_idx : int
        First dataset row of the horizon.
    n_steps : int or None
        Horizon length; defaults to env.episode_length.
    verbose : bool
        Print progress / summary lines (turn off for per-user loops).
    problem_name : str
        Name handed to pulp.LpProblem (cosmetic; useful when solving many users).
    solver : pulp solver or None
        Defaults to pulp.PULP_CBC_CMD(msg=False).
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
        and the RL-equivalent reward. See `milp_total_cost` for the scalar
        "theoretical optimum cost" used by the multi-user notebooks.
    """
    import pulp

    from Pricing_Functions import (
        compute_prorated_fixed_charge_eur,
        InvoiceBuilder,
        PRIVZETO_REFERENCNO_LETO,
    )
    from si_obracun import Pravila
    from si_cas import v_lokalni_cas, je_visja_sezona

    if verbose:
        print("Building MILP Model...")

    # -------------------------------------------------------------------------
    # 0) REGULATORY REGIME -- ONE regime for the whole horizon
    # -------------------------------------------------------------------------
    # The datasets are Ausgrid 2010-2013, so their timestamps have no published
    # SI tariff act: resolving rules from the data date raises
    # "Za 2012-06-30 ni nalozenih tarifnih postavk". The MILP therefore always
    # prices under an explicit reference year -- the env's, or 2026 when the env
    # leaves it open -- and pins it into pricing_options so every pricing call
    # below (energy rates, fixed monthly charge, reporting pass) uses exactly
    # the same rates as the block/peak rates derived here.
    pricing_options = dict(env.pricing_options or {})
    ref_year = pricing_options.get("pricing_reference_year", env.pricing_reference_year)
    ref_year = PRIVZETO_REFERENCNO_LETO if ref_year is None else int(ref_year)
    pricing_options["pricing_reference_year"] = ref_year
    pravila_ref = Pravila.za_leto(ref_year)

    # Local name is N_STEPS (not T) so notebooks that alias torch as T are safe.
    N_STEPS = int(env.episode_length if n_steps is None else n_steps)

    gen = env.arr_Gen[start_idx : start_idx + N_STEPS]
    con = env.arr_Con[start_idx : start_idx + N_STEPS]
    smp_prices = env.arr_SMP[start_idx : start_idx + N_STEPS]
    rel_price = env.arr_RelativePrice[start_idx : start_idx + N_STEPS]
    dates = env.dataset.index[start_idx : start_idx + N_STEPS]

    INTERVAL_MINS = int(round(1440.0 / env.korakov_na_dan))

    invoice_builder = None
    if generate_invoice:
        invoice_builder = InvoiceBuilder(
            dogovorjena_moc=env.dogovorjena_moc,
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
    # NOTE: scheme=env.pricing_scheme / **pricing_options is required here
    # (previously omitted, so this silently always priced under aus_base
    # regardless of env.pricing_scheme). These calls are deliberately made
    # WITHOUT dogovorjena_moc/prev_peak_kw so they stay strictly linear in
    # total_consumed_kwh (required for the +-1kWh unit-rate trick) -- peak/
    # excess charges are handled separately below via explicit MILP
    # variables/constraints, and fixed monthly charges via
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
                dogovorjena_moc=env.dogovorjena_moc, **pricing_options,
            )
        )

    # -------------------------------------------------------------------------
    # 2) DEFINE MILP
    # -------------------------------------------------------------------------
    prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

    # Shared variables
    P_buy = [pulp.LpVariable(f"P_buy_{t}", lowBound=0) for t in range(N_STEPS)]
    P_sell = [pulp.LpVariable(f"P_sell_{t}", lowBound=0) for t in range(N_STEPS)]
    E = [pulp.LpVariable(f"E_{t}", lowBound=0, upBound=env.bat_kapaciteta) for t in range(N_STEPS + 1)]

    # Initial SOC
    prob += (E[0] == max(0.0, env.bat_kapaciteta / 2.0))

    if not use_discrete_actions:
        # ---------------------------------------------------------------------
        # CONTINUOUS MODE
        # ---------------------------------------------------------------------
        P_ch = [pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=env.bat_max_polnjenje) for t in range(N_STEPS)]
        P_dis = [pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=env.bat_max_praznjenje) for t in range(N_STEPS)]

        for t in range(N_STEPS):
            prob += (gen[t] + P_buy[t] + P_dis[t] == con[t] + P_sell[t] + P_ch[t])
            prob += (
                E[t + 1]
                == E[t] + P_ch[t] * env.bat_ucinkovitost - P_dis[t] / env.bat_ucinkovitost
            )

    else:
        # ---------------------------------------------------------------------
        # DISCRETE MODE (choose one of env.action_space.n actions)
        # ---------------------------------------------------------------------
        n_actions = int(env.action_space.n)
        A = [
            [pulp.LpVariable(f"A_{t}_{a}", cat="Binary") for a in range(n_actions)]
            for t in range(N_STEPS)
        ]

        # Optional auxiliaries:
        P_ch = [pulp.LpVariable(f"P_ch_{t}", lowBound=0, upBound=env.bat_max_polnjenje) for t in range(N_STEPS)]
        P_dis = [pulp.LpVariable(f"P_dis_{t}", lowBound=0, upBound=env.bat_max_praznjenje) for t in range(N_STEPS)]
        P_spill = [pulp.LpVariable(f"P_spill_{t}", lowBound=0) for t in range(N_STEPS)]

        BIG_M = max(
            float(np.max(gen + con)),
            float(env.bat_max_polnjenje),
            float(env.bat_max_praznjenje),
            10.0,
        ) * 10.0

        for t in range(N_STEPS):
            # one action per step
            prob += pulp.lpSum(A[t][a] for a in range(n_actions)) == 1

            # action semantics (based on env.step() action set)
            # 0: charge (PV + grid)
            # 1: charge (PV only)
            # 2: discharge to house / no charge
            # 3: discharge to house + grid / no charge
            # 4: idle / no battery use

            # charging only allowed in actions 0 and 1
            prob += P_ch[t] <= env.bat_max_polnjenje * (A[t][0] + A[t][1])

            # discharging only allowed in actions 2 and 3
            prob += P_dis[t] <= env.bat_max_praznjenje * (A[t][2] + A[t][3])

            # spilling/curtailment only allowed in actions 1 and 4
            prob += P_spill[t] <= BIG_M * (A[t][1] + A[t][4])

            # Energy balance with curtailment
            prob += (
                gen[t] + P_buy[t] + P_dis[t]
                == con[t] + P_sell[t] + P_ch[t] + P_spill[t]
            )

            # Battery dynamics
            prob += (
                E[t + 1]
                == E[t] + P_ch[t] * env.bat_ucinkovitost - P_dis[t] / env.bat_ucinkovitost
            )

    # -------------------------------------------------------------------------
    # 2b) PEAK / EXCESS-POWER (ratchet) VARIABLES AND CONSTRAINTS
    # -------------------------------------------------------------------------
    # Only meaningful for SI schemes (env._blok_arr is precomputed by
    # Environment.py only when pricing_scheme is si_dobava/si_samooskrba).
    # One P_peak/Excess variable per (block, month) pair that actually occurs
    # in the horizon, ratcheted across consecutive months within the same
    # reset-window (env.peak_reset_months), floored at the historical seed
    # peak entering the horizon (env.compute_seed_peak_kw). This mirrors the
    # RL environment's per-step marginal ratchet charge exactly (see the
    # telescoping-sum proof in the project plan) -- both formulations charge
    # the same total for the same trajectory.
    peak_objective_terms = []
    if getattr(env, "_blok_arr", None) is not None:
        blok_arr = env._blok_arr[start_idx : start_idx + N_STEPS]
        window_id_arr = env._window_id_arr[start_idx : start_idx + N_STEPS]
        seed_peak_kw = env.compute_seed_peak_kw(start_idx)

        # Same regime as every pricing call above (section 0).
        pravila_for_blocks = pravila_ref

        lok_t = [v_lokalni_cas(dates[t]) for t in range(N_STEPS)]
        month_key_t = [(lok_t[t].year, lok_t[t].month) for t in range(N_STEPS)]
        months_sorted = sorted(set(month_key_t))
        month_idx_t = [months_sorted.index(month_key_t[t]) for t in range(N_STEPS)]

        month_window = {}
        for t in range(N_STEPS):
            m = month_idx_t[t]
            if m not in month_window:
                month_window[m] = int(window_id_arr[t])

        ure = INTERVAL_MINS / 60.0
        occurring = sorted({(int(blok_arr[t]), month_idx_t[t]) for t in range(N_STEPS)})
        P_peak_month = {(b, m): pulp.LpVariable(f"P_peak_b{b}_m{m}", lowBound=0) for (b, m) in occurring}
        Excess_month = {(b, m): pulp.LpVariable(f"Excess_b{b}_m{m}", lowBound=0) for (b, m) in occurring}

        for t in range(N_STEPS):
            b, m = int(blok_arr[t]), month_idx_t[t]
            prob += P_peak_month[(b, m)] >= P_buy[t] / ure

        last_var_by_block, last_window_by_block = {}, {}
        for (b, m) in occurring:
            w = month_window[m]
            if b in last_var_by_block and last_window_by_block[b] == w:
                prob += P_peak_month[(b, m)] >= last_var_by_block[b]
            else:
                prob += P_peak_month[(b, m)] >= seed_peak_kw.get(b, 0.0)
            last_var_by_block[b] = P_peak_month[(b, m)]
            last_window_by_block[b] = w

            prob += Excess_month[(b, m)] >= P_peak_month[(b, m)] - env.dogovorjena_moc.get(b, 0.0)

        # Incremental (telescoping) objective contribution per (block, month),
        # using each month's own season-correct rate (only block 1's rate
        # depends on season).
        prev_excess_by_block, prev_window_by_block = {}, {}
        for (b, m) in occurring:
            y, mo = months_sorted[m]
            w = month_window[m]
            vs = je_visja_sezona(date(y, mo, 1))
            rate_bm = pravila_for_blocks.omreznina.postavka_moc(b, vs)
            faktor = pravila_for_blocks.omreznina.faktor_presezne_moci

            if b in prev_excess_by_block and prev_window_by_block[b] == w:
                prev_term = prev_excess_by_block[b]
            else:
                prev_term = max(0.0, seed_peak_kw.get(b, 0.0) - env.dogovorjena_moc.get(b, 0.0))

            peak_objective_terms.append((Excess_month[(b, m)] - prev_term) * rate_bm * faktor)
            prev_excess_by_block[b] = Excess_month[(b, m)]
            prev_window_by_block[b] = w

    # -------------------------------------------------------------------------
    # 2c) OBJECTIVE (applies identically to both continuous and discrete modes)
    # -------------------------------------------------------------------------
    total_economic_cost = pulp.lpSum(
        P_buy[t] * import_rates[t] - P_sell[t] * export_rates[t]
        + constant_costs[t] + fixed_monthly_costs[t]
        for t in range(N_STEPS)
    ) + pulp.lpSum(peak_objective_terms)
    prob += total_economic_cost

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
    reporting_peak_kw = env.compute_seed_peak_kw(start_idx)

    for t in range(N_STEPS):
        buy_val = P_buy[t].varValue or 0.0
        sell_val = P_sell[t].varValue or 0.0
        e_val = E[t].varValue or 0.0
        ch_val = P_ch[t].varValue or 0.0
        dis_val = P_dis[t].varValue or 0.0

        if use_discrete_actions:
            spill_val = P_spill[t].varValue or 0.0
            action_val = int(np.argmax([(A[t][a].varValue or 0.0) for a in range(n_actions)]))
        else:
            spill_val = 0.0
            action_val = None

        net_kwh = buy_val - sell_val
        interval_cost_data = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=net_kwh,
            utc_date=dates[t],
            interval_minutes=INTERVAL_MINS,
            scheme=env.pricing_scheme,
            dogovorjena_moc=env.dogovorjena_moc,
            prev_peak_kw=reporting_peak_kw,
            include_raw=invoice_builder is not None,
            **pricing_options,
        )
        reporting_peak_kw = dict(interval_cost_data["new_peak_kw"])
        if invoice_builder is not None:
            invoice_builder.add_interval(interval_cost_data)

        konstantno_placilo = interval_cost_data["constant_price_aud"]
        placilo_zdaj_variabilno = interval_cost_data["variable_price_aud"]
        cumulative_payment += konstantno_placilo + placilo_zdaj_variabilno

        soc_norm = e_val / env.bat_kapaciteta if env.bat_kapaciteta > 0 else 0.0
        sprememba_baterije = (ch_val * env.bat_ucinkovitost) - (dis_val / env.bat_ucinkovitost)

        s_for_reward = SimpleNamespace(
            Baterija_norm=soc_norm,
            CenaElRel=rel_price[t],
        )

        total_step_reward = env._nagrada_skupno(
            s_for_reward, sprememba_baterije, placilo_zdaj_variabilno, None
        )
        cumulative_rl_reward += total_step_reward

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
            "Import_Rate_kWh": import_rates[t],
            "Export_Rate_kWh": export_rates[t],
            "Step_Cost": konstantno_placilo + placilo_zdaj_variabilno,
            "Cum_Cost": cumulative_payment,
            "Step_RL_Reward": total_step_reward,
            "Cum_RL_Reward": cumulative_rl_reward,
            "Energy_Component_EUR": interval_cost_data["energy_component_eur"],
            "Power_Component_EUR": interval_cost_data["power_component_eur"],
        }

        if use_discrete_actions:
            row["Action"] = action_val
            row["Spill_kW"] = spill_val

        results.append(row)

    df_results = pd.DataFrame(results)

    if invoice_builder is not None and N_STEPS > 0:
        invoice_builder.finalize(
            period_label=f"{dates[0]:%Y-%m-%d}_{dates[-1]:%Y-%m-%d}"
        )

    if verbose:
        print("\n--- MILP Optimization Complete ---")
        print(f"Total Electricity Cost (Inc GST & Fixed): {cumulative_payment:.4f}")
        print(f"Total RL Equivalent Reward: {cumulative_rl_reward:.4f}")

    return df_results


def milp_total_cost(env, **kwargs):
    """Scalar convenience wrapper around `run_milp_benchmark`: returns the
    total cost over the horizon (the perfect-foresight theoretical optimum),
    priced exactly like the RL environment prices a trajectory. Accepts the
    same keyword arguments; `verbose` defaults to False here."""
    kwargs.setdefault("verbose", False)
    df = run_milp_benchmark(env, **kwargs)
    if df.empty:
        return 0.0
    return float(df["Cum_Cost"].iloc[-1])


def cumulative_interval_price_series(consumption, generation, pricing_env, dataset):
    cumulative_payment = 0.0
    cumulative_series = []
    interval_minutes = 1440.0 / pricing_env.korakov_na_dan
    peak_kw = pricing_env.compute_seed_peak_kw(0)

    peak_window_id = (
        int(pricing_env._window_id_arr[0])
        if pricing_env._window_id_arr is not None and len(dataset) > 0
        else 0
        )

    for i, timestamp in enumerate(dataset.index):
        if pricing_env._window_id_arr is not None:
            current_window_id = int(pricing_env._window_id_arr[i])
            if current_window_id != peak_window_id:
                peak_kw = {b: 0.0 for b in range(1, 6)}
                peak_window_id = current_window_id

        net_kwh = float(consumption.iloc[i] - generation.iloc[i])
        interval_cost = calculate_interval_price(
            smp_market_price_kwh=float(dataset["SMP"].iloc[i]),
            total_consumed_kwh=net_kwh,
            utc_date=timestamp,
            interval_minutes=interval_minutes,
            scheme=pricing_env.pricing_scheme,
            compare_all=pricing_env.pricing_compare_all,
            dogovorjena_moc=pricing_env.dogovorjena_moc,
            prev_peak_kw=peak_kw,
            **pricing_env.pricing_options,
        )

        peak_kw = dict(interval_cost["new_peak_kw"])
        cumulative_payment += float(interval_cost["constant_price_aud"])
        cumulative_payment += float(interval_cost["variable_price_aud"])
        cumulative_series.append(cumulative_payment)

    return cumulative_series

