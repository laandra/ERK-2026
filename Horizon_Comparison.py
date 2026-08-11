"""Optimization-horizon comparison for the perfect-foresight MILP.

The same MILP formulation is driven in several ways over the same household and
the same battery, on each of four GEN-I price lists, and the executed
trajectories are compared:

    horizon   how much of the future each solve sees (day / week / month / all)
    execution "block"    -- execute everything that was planned, then re-plan
              "receding" -- execute the first interval only, then re-plan (MPC)
    soc_mode  "fixed50"  -- every solve starts AND ends at 50 % of capacity
              "carry"    -- only the first period starts at 50 %; a period ends
                            wherever it likes and the next one starts from
                            exactly that state of charge
    tariff    which price list the dispatch is optimized and billed under
              (Dinamični / Aktivni / Redni 1T / Redni 2T, see TARIFFS)

The price list is a full axis of the study, not a setting: every strategy is run
on every list, and a strategy's gap is always measured against the whole-year
solve of the *same* household on the *same* list. How far ahead a controller must
look is a property of the price signal, so the answer is allowed to differ
between a list that moves every 15 minutes and one that is flat all year.

`full_period` is the perfect-foresight optimum: it is a relaxation of every other
strategy (no interior constraints at all), so its cost is a lower bound and every
gap in the comparison is non-negative.

The two `soc_mode` families answer different questions. `fixed50` asks what a
controller gives up when it must hand the battery over in a defined state at
every period boundary. `carry` removes that constraint -- but note what it
implies: stored energy has no value in a MILP objective that stops at the horizon
end, so a *block* strategy under `carry` empties the battery at every boundary.
Both are real operating rules; neither is a bug.

Design notes
------------
*The MILP only produces a dispatch.* Every strategy's executed trajectory is
priced afterwards by one shared evaluator (`price_interval`), interval by
interval, carrying a single running peak state. That is the only way the variants
are comparable: the excess-power ("konica") charge is a running maximum over the
whole horizon, so a strategy cannot be allowed to price its own peaks.

*The controller is told the truth about its peak state.* `run_milp_benchmark`
seeds each sub-solve from `env.compute_seed_peak_kw(start_idx)`, which is the
peak a household with no battery would have reached by then -- wrong for a
trajectory that has been shaving peaks all year. `_PeakSeedView` wraps the
environment and feeds back the peak the executed trajectory has actually set, so
every sub-solve starts from the state the evaluator is in.

*The year is always closed.* Under both SOC modes the run starts at 50 % and the
final period is required to end at 50 %. Without that last constraint a
free-terminal strategy would sell off its opening charge and book it as a saving.
Only the *interior* boundaries differ between the two families.

*Feasibility near the end of the dataset* is inductive: the horizon shrinks in
lockstep with the remaining steps, so a plan that could reach the terminal SOC at
t can still do so at t+1.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

import Data_Loader as dl
from Environment import HouseholdEnvironment
from MILP_Benchmark import run_milp_benchmark
from Pricing_Functions import calculate_interval_price

# --- Study configuration ---------------------------------------------------
DATASET_GROUPS = [
    "Fluvius",
    "Fluvius_EV",
    "Fluvius_HP",
    "Fluvius_HP_EV",
    "Fluvius_PV",
    "Fluvius_PV_EV",
    "Fluvius_PV_HP",
    "Fluvius_PV_HP_EV",
]
HOUSEHOLDS_PER_GROUP = 5

SMP_COUNTRY_ID = "Slovenia"
PRICE_COLUMN = "SMP"
GENERATION_COLUMN = "Feed_In_Volume_kWh"
CONSUMPTION_COLUMN = "Consumption_Volume_kWh"

# The price lists the horizon question is asked under -----------------------
# Four GEN-I household products, every one of them run through every strategy on
# every household, so the horizon effect can be read both within a price list and
# across them. The naming follows Multuser_Battery_Size_Optimization: "1T"/"2T"
# say whether the list meters on a single rate or on a VT/MT pair.
#
#   Dinamični  GENI_SAMO_DINAMICNI  SIPX +- 0.01199, uncapped, credited per interval
#   Aktivni    GENI_SAMO_AKTIVNI    4 fixed blocks, 0.04090 -> 0.19290 in, 0.00190 -> 0.14990 out
#   Redni 1T   GENI_SAMO_REDNI      0.10290 flat in, 0.05390 flat out
#   Redni 2T   GENI_REDNI           VT 0.11990 / MT 0.09790 in, no buyback at all
#
# Every list runs under `si_samooskrba`, including the plain two-tariff one:
# intra-interval netting is physical, and what the *contract* decides is only
# what a surplus is worth, which is read off the package (`tip_odkupa=NI` on
# GENI_REDNI credits an export 0.00000). Routing GENI_REDNI through `si_dobava`
# instead would credit an exported kWh at the full retail rate including network
# charges and VAT, which makes grid arbitrage unbounded and the comparison
# meaningless.
#
# NET metering is deliberately absent: it is closed to new contracts and settles
# once a year, so it is not a list a household can choose into and it is not a
# baseline. See Multuser_Battery_Size_Optimization for the legacy reference line.
#
# `slug` is the results-file suffix. The default list carries no suffix, so the
# CSVs written before this study grew a tariff axis are still read as its rows.
PRICING_SCHEME = "si_samooskrba"
TARIFFS = {
    "Dinamični": {"paket_id": "GENI_SAMO_DINAMICNI", "scheme": PRICING_SCHEME, "slug": None,
                  "structure": "dynamic (SIPX)", "buyback": True},
    "Aktivni":   {"paket_id": "GENI_SAMO_AKTIVNI",   "scheme": PRICING_SCHEME, "slug": "aktivni",
                  "structure": "4 time blocks", "buyback": True},
    "Redni 1T":  {"paket_id": "GENI_SAMO_REDNI",     "scheme": PRICING_SCHEME, "slug": "redni1t",
                  "structure": "flat (ET)", "buyback": True},
    "Redni 2T":  {"paket_id": "GENI_REDNI",          "scheme": PRICING_SCHEME, "slug": "redni2t",
                  "structure": "two-tariff (VT/MT)", "buyback": False},
}
TARIFF_ORDER = list(TARIFFS)
DEFAULT_TARIFF = "Dinamični"
PAKET_ID = TARIFFS[DEFAULT_TARIFF]["paket_id"]   # kept for callers that want one id

PRICING_REFERENCE_YEAR = 2026
PEAK_RESET_MONTHS = None

BATTERY_CAPACITY_KWH = 30.0
SOC_FRACTION = 0.5
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
C_RATE = 0.5
INVERTER_MAX_KW = 11.0

STEPS_PER_DAY = 96
FULL_PERIOD_GAP_REL = 0.001
FULL_PERIOD_TIME_LIMIT_S = 900

RESULTS_DIR = Path(__file__).resolve().parent / "Results" / "Horizon_Comparison_Groups"

# name -> (horizon, execution, soc_mode)
STRATEGIES = {
    # --- every solve starts and ends at 50 % of capacity ---
    "day_block": ("day", "block", "fixed50"),
    "day_receding": ("day", "receding", "fixed50"),
    "week_block": ("week", "block", "fixed50"),
    #"week_receding": ("week", "receding", "fixed50"),      # ~3 h per household
    "month_block": ("month", "block", "fixed50"),
    "full_period": ("period", "block", "fixed50"),
    # --- SOC carried across period boundaries (50 % only at the very start) ---
   # "day_block_carry": ("day", "block", "carry"),
    #"day_receding_carry": ("day", "receding", "carry"),
    #"week_block_carry": ("week", "block", "carry"),
    #"week_receding_carry": ("week", "receding", "carry"),  # ~3 h per household
    #"month_block_carry": ("month", "block", "carry"),
    #"full_period_carry": ("period", "block", "carry"),
}

REFERENCE_STRATEGY = "full_period"
# A household-unit is one (dataset, household) profile; a run key adds the price
# list, because the whole-year optimum a strategy is scored against is the
# optimum *for that same household on that same list*.
UNIT_COLUMNS = ["Dataset", "Household"]
KEY_COLUMNS = UNIT_COLUMNS + ["Tariff"]

_BLOCKS = (1, 2, 3, 4, 5)


def study_units(groups=None, per_group=HOUSEHOLDS_PER_GROUP):
    """The (dataset, household id) pairs the study runs over."""
    groups = groups or DATASET_GROUPS
    return [(g, i) for g in groups for i in range(1, per_group + 1)]


def study_jobs(units=None, tariffs=None):
    """The (dataset, household id, tariff) triples the batch solves."""
    units = units or study_units()
    tariffs = tariffs or TARIFF_ORDER
    return [(g, i, t) for g, i in units for t in tariffs]


def unit_csv_path(output_dir, dataset, household_id, tariff=DEFAULT_TARIFF):
    """Results file for one (household, price list). The default list keeps the
    un-suffixed name the single-tariff version of this study wrote."""
    slug = TARIFFS[tariff]["slug"]
    suffix = f"_{slug}" if slug else ""
    return Path(output_dir) / f"{dataset}_user_{household_id:03d}{suffix}.csv"


# ---------------------------------------------------------------------------
# Data + environment
# ---------------------------------------------------------------------------
def load_user(household_id, dataset, country=SMP_COUNTRY_ID):
    """Household profile with the country SMP series patched in (EUR/kWh)."""
    data = dl.load_household_data(int(household_id), dataset=dataset)
    smp = dl.load_smp_data(country).reindex(data.index, method="ffill")
    series = pd.to_numeric(smp[PRICE_COLUMN], errors="coerce").ffill().bfill()
    scale = 1000.0 if float(series.abs().quantile(0.95)) > 2.0 else 1.0
    data[PRICE_COLUMN] = (series / scale).astype(float)
    return data


def build_env(data, capacity_kwh=BATTERY_CAPACITY_KWH, tariff=DEFAULT_TARIFF):
    if tariff not in TARIFFS:
        raise ValueError(f"Unknown tariff {tariff!r}. Known: {', '.join(TARIFF_ORDER)}")
    spec = TARIFFS[tariff]
    power_kw = min(C_RATE * capacity_kwh, INVERTER_MAX_KW)
    step_kwh = power_kw * 24.0 / STEPS_PER_DAY
    return HouseholdEnvironment(
        dataset=data,
        price_column=PRICE_COLUMN,
        generation_column=GENERATION_COLUMN,
        consumption_column=CONSUMPTION_COLUMN,
        action_mode="continuous",
        allow_curtailment=True,
        reset_mode="deterministic",
        episode_length=len(data) - 1,
        steps_per_day=STEPS_PER_DAY,
        battery_capacity_kwh=capacity_kwh,
        charge_efficiency=CHARGE_EFFICIENCY,
        discharge_efficiency=DISCHARGE_EFFICIENCY,
        max_charge_kwh=step_kwh,
        max_discharge_kwh=step_kwh,
        pricing_scheme=spec["scheme"],
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        pricing_options={"paket_id": spec["paket_id"]},
        peak_reset_months=PEAK_RESET_MONTHS,
    )


class _PeakSeedView:
    """Environment proxy that reports the peak the executed trajectory has set.

    Everything except `compute_seed_peak_kw` is forwarded untouched, so
    `run_milp_benchmark` reads the real environment's arrays, tariff and battery
    parameters.
    """

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "peak_state", {b: 0.0 for b in _BLOCKS})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def compute_seed_peak_kw(self, start_idx):
        return dict(object.__getattribute__(self, "peak_state"))


# ---------------------------------------------------------------------------
# Shared evaluator -- the single source of cost for every strategy
# ---------------------------------------------------------------------------
def price_interval(env, idx, net_kwh, peak_state):
    """Price one executed interval and return (cost, components, new peak state)."""
    result = calculate_interval_price(
        smp_market_price_kwh=env.arr_price[idx],
        total_consumed_kwh=float(net_kwh),
        utc_date=env.dataset.index[idx],
        interval_minutes=env.interval_minutes,
        scheme=env.pricing_scheme,
        dogovorjena_moc=env.contracted_power_kw,
        prev_peak_kw=peak_state,
        **env.pricing_options,
    )
    cost = float(result["constant_price_aud"]) + float(result["variable_price_aud"])
    return (
        cost,
        float(result["energy_component_eur"]),
        float(result["power_component_eur"]),
        dict(result["new_peak_kw"]),
    )


def no_battery_cost(env, n_steps):
    """Reference cost of the same household with no battery and no curtailment."""
    peak_state = {b: 0.0 for b in _BLOCKS}
    total = 0.0
    for idx in range(n_steps):
        net = float(env.arr_consumption[idx] - env.arr_generation[idx])
        cost, _, _, peak_state = price_interval(env, idx, net, peak_state)
        total += cost
    return total


# ---------------------------------------------------------------------------
# The rolling solve
# ---------------------------------------------------------------------------
def _horizon_steps(kind, n_steps):
    if kind == "day":
        return STEPS_PER_DAY
    if kind == "week":
        return 7 * STEPS_PER_DAY
    if kind == "month":
        return 31 * STEPS_PER_DAY
    return n_steps


def run_strategy(env, horizon_kind, execution, soc_mode="fixed50", n_steps=None,
                 solver=None, verbose=False):
    """Roll one strategy over the horizon and return its executed trajectory.

    horizon_kind : "day" | "week" | "month" | "period"
    execution    : "block" (execute everything that was planned) or
                   "receding" (execute the first interval, then re-plan)
    soc_mode     : "fixed50" (every solve must end at 50 % of capacity) or
                   "carry" (only the final period is pinned; interior periods end
                   where they like and the next one starts from that SOC)
    """
    n_steps = int(env.episode_length if n_steps is None else n_steps)
    soc_target = SOC_FRACTION * env.battery_capacity_kwh
    horizon = _horizon_steps(horizon_kind, n_steps)

    # Month blocks follow the calendar rather than a fixed 31-day stride.
    month_edges = None
    if horizon_kind == "month":
        stamps = env.dataset.index[:n_steps]
        months = np.asarray(stamps.year) * 12 + np.asarray(stamps.month)
        month_edges = [0, *(np.flatnonzero(months[1:] != months[:-1]) + 1).tolist(), n_steps]

    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)

    view = _PeakSeedView(env)
    peak_state = {b: 0.0 for b in _BLOCKS}
    soc = soc_target

    cost = energy_cost = power_cost = 0.0
    charged = discharged = grid_charged = curtailed = 0.0
    import_kwh = export_kwh = 0.0
    peak_import_kw = 0.0
    n_solves = 0
    soc_drift = 0.0
    hours = env.interval_minutes / 60.0
    soc_trace = np.empty(n_steps, dtype=float)
    cost_trace = np.empty(n_steps, dtype=float)

    t = 0
    t_start = time.time()
    while t < n_steps:
        if month_edges is not None:
            block_end = next(e for e in month_edges if e > t)
            span = block_end - t
        else:
            span = min(horizon, n_steps - t)

        # The year is always closed at 50 %; under "carry" that is the only
        # boundary that is pinned, and stored energy is otherwise handed to the
        # next period exactly as the solver left it.
        reaches_end = (t + span) >= n_steps
        final_soc = soc_target if (soc_mode == "fixed50" or reaches_end) else None

        plan = run_milp_benchmark(
            view,
            use_discrete_actions=False,
            start_idx=t,
            n_steps=span,
            initial_soc_kwh=soc,
            final_soc_kwh=final_soc,
            solver=solver,
            verbose=False,
        )
        n_solves += 1

        n_exec = 1 if execution == "receding" else span
        for k in range(n_exec):
            row = plan.iloc[k]
            idx = t + k
            ch, dis, spill = float(row["Charge_kW"]), float(row["Discharge_kW"]), float(row["Spill_kW"])
            gen, con = float(row["Generation"]), float(row["Consumption"])
            net = con + ch + spill - gen - dis

            step_cost, e_part, p_part, peak_state = price_interval(env, idx, net, peak_state)
            cost += step_cost
            energy_cost += e_part
            power_cost += p_part

            soc_trace[idx] = soc
            cost_trace[idx] = cost
            soc += ch * env.charge_efficiency - dis / env.discharge_efficiency
            # A solve that ended non-optimal would silently drift the state; keep
            # the trajectory physical and record how far off it ever got.
            clipped = min(max(soc, 0.0), env.battery_capacity_kwh)
            soc_drift = max(soc_drift, abs(clipped - soc))
            soc = clipped

            charged += ch
            discharged += dis
            grid_charged += max(0.0, ch - max(0.0, gen - con))
            curtailed += spill
            if net >= 0:
                import_kwh += net
                peak_import_kw = max(peak_import_kw, net / hours)
            else:
                export_kwh += -net

        t += n_exec
        if verbose and n_solves % 500 == 0:
            print(f"    ...{t}/{n_steps} steps, {time.time() - t_start:.0f}s", flush=True)

    return {
        "Cost_EUR": cost,
        "Energy_EUR": energy_cost,
        "Power_EUR": power_cost,
        "Final_SOC_kWh": soc,
        "Charged_kWh": charged,
        "Discharged_kWh": discharged,
        "Grid_Charged_kWh": grid_charged,
        "Curtailed_kWh": curtailed,
        "Import_kWh": import_kwh,
        "Export_kWh": export_kwh,
        "Peak_Import_kW": peak_import_kw,
        "Equivalent_Full_Cycles": discharged / env.battery_capacity_kwh,
        "N_Solves": n_solves,
        "SOC_Drift_kWh": soc_drift,
        "Runtime_s": time.time() - t_start,
        "_soc_trace": soc_trace,
        "_cost_trace": cost_trace,
    }


def run_user(household_id, dataset, tariff=DEFAULT_TARIFF, n_steps=None, strategies=None,
             keep_traces=False, verbose=True, checkpoint_path=None):
    """All strategies for one household on one price list. One row each.

    With `checkpoint_path` the CSV is rewritten after every finished strategy and
    strategies already present in it are skipped, so an interrupted batch resumes
    where it stopped instead of redoing hours of solves.
    """
    strategies = strategies or list(STRATEGIES)
    data = load_user(household_id, dataset)
    env = build_env(data, tariff=tariff)
    n_steps = int(env.episode_length if n_steps is None else n_steps)

    rows, traces = [], {}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        done = pd.read_csv(checkpoint_path)
        rows = done.to_dict("records")
        baseline = float(done["No_Battery_EUR"].iloc[0])
        strategies = [s for s in strategies if s not in set(done["Strategy"])]
    else:
        baseline = no_battery_cost(env, n_steps)

    for name in strategies:
        horizon_kind, execution, soc_mode = STRATEGIES[name]
        solver = (
            pulp.PULP_CBC_CMD(msg=False, gapRel=FULL_PERIOD_GAP_REL,
                              timeLimit=FULL_PERIOD_TIME_LIMIT_S)
            if horizon_kind == "period" else pulp.PULP_CBC_CMD(msg=False)
        )
        out = run_strategy(env, horizon_kind, execution, soc_mode=soc_mode,
                           n_steps=n_steps, solver=solver)
        traces[name] = (out.pop("_soc_trace"), out.pop("_cost_trace"))
        out.update(
            Dataset=str(dataset),
            Household=int(household_id),
            Tariff=str(tariff),
            Paket_ID=TARIFFS[tariff]["paket_id"],
            Strategy=name,
            Horizon=horizon_kind,
            Execution=execution,
            SOC_Mode=soc_mode,
            No_Battery_EUR=baseline,
            Savings_EUR=baseline - out["Cost_EUR"],
        )
        rows.append(out)
        if verbose:
            print(f"  [{dataset} {household_id} | {tariff}] {name:20s} {out['Cost_EUR']:9.2f} EUR  "
                  f"({out['N_Solves']:6d} solves, {out['Runtime_s']:7.1f} s)", flush=True)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(rows)
    # A CSV resumed from the single-tariff version of this study has no Tariff
    # column of its own; it is the default list by construction.
    for col, value in (("Tariff", str(tariff)), ("Paket_ID", TARIFFS[tariff]["paket_id"])):
        if col not in df.columns:
            df[col] = value
        else:
            df[col] = df[col].fillna(value)
    lead = ["Dataset", "Household", "Tariff", "Strategy", "Horizon", "Execution", "SOC_Mode",
            "Cost_EUR", "Savings_EUR", "No_Battery_EUR", "Energy_EUR", "Power_EUR"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return (df, traces) if keep_traces else df


# ---------------------------------------------------------------------------
# Batch driver (multiprocessing over households, one CSV per household)
# ---------------------------------------------------------------------------
def _worker(args):
    dataset, household_id, tariff, n_steps, output_dir = args
    out_path = unit_csv_path(output_dir, dataset, household_id, tariff)
    df = run_user(household_id, dataset, tariff=tariff, n_steps=n_steps, verbose=True,
                  checkpoint_path=out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return f"{out_path.name} ({len(df)} strategies)"


def run_batch(units=None, tariffs=None, n_steps=None, output_dir=RESULTS_DIR, n_workers=10):
    """Run every (dataset, household, price list) job in parallel.

    One CSV per (household, price list); finished work is skipped, so adding a
    price list only solves the new one.
    """
    import multiprocessing as mp

    jobs = study_jobs(units, tariffs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(str(g), int(h), str(t), n_steps, str(output_dir)) for g, h, t in jobs]

    if n_workers <= 1:
        for job in jobs:
            _worker(job)
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for done in pool.imap_unordered(_worker, jobs):
                print(f"done -> {done}", flush=True)

    return collect_results(output_dir)


def collect_results(output_dir=RESULTS_DIR):
    """Concatenate every per-(household, price list) CSV written so far.

    Files written before this study grew a tariff axis carry no Tariff column;
    they are the default price list, and are labelled as such on the way in.
    """
    files = sorted(Path(output_dir).glob("*user_*.csv"))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        df = pd.read_csv(f)
        if "Tariff" not in df.columns:
            df["Tariff"] = DEFAULT_TARIFF
        df["Tariff"] = df["Tariff"].fillna(DEFAULT_TARIFF)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score(df_all, reference=REFERENCE_STRATEGY):
    """Add per-unit gap columns measured against the perfect-foresight optimum.

    The optimum is resolved per (household, price list): a strategy is only ever
    compared with the whole-year solve of the same household on the same tariff,
    so the gaps never mix one price list's bill into another's.
    """
    optimum = (
        df_all[df_all["Strategy"] == reference]
        .set_index(KEY_COLUMNS)["Cost_EUR"]
        .rename("Optimum_EUR")
    )
    df = df_all.join(optimum, on=KEY_COLUMNS)
    df["Gap_to_Optimum_EUR"] = df["Cost_EUR"] - df["Optimum_EUR"]

    achievable = (df["No_Battery_EUR"] - df["Optimum_EUR"]).where(lambda s: s.abs() > 1e-9)
    df["Saving_Captured_pct"] = 100.0 * df["Savings_EUR"] / achievable
    # The headline number: the share of the achievable saving a strategy throws
    # away by not seeing the whole year. 0 % is the optimum.
    df["Gap_pct"] = 100.0 - df["Saving_Captured_pct"]
    return df


def summarize(df_all, reference=REFERENCE_STRATEGY):
    """(price list, strategy) comparison table, averaged over household-units."""
    if df_all.empty:
        return pd.DataFrame(), df_all

    df = score(df_all, reference)
    order = [(t, s) for t in TARIFF_ORDER for s in STRATEGIES
             if (t, s) in set(zip(df["Tariff"], df["Strategy"]))]
    summary = (
        df.groupby(["Tariff", "Strategy"])
        .agg(
            Units=("Cost_EUR", "size"),
            SOC_Mode=("SOC_Mode", "first"),
            Gap_pct=("Gap_pct", "mean"),
            Worst_Gap_pct=("Gap_pct", "max"),
            Best_Gap_pct=("Gap_pct", "min"),
            Gap_EUR=("Gap_to_Optimum_EUR", "mean"),
            Cost_EUR=("Cost_EUR", "mean"),
            Savings_EUR=("Savings_EUR", "mean"),
            Cycles=("Equivalent_Full_Cycles", "mean"),
            Peak_kW=("Peak_Import_kW", "mean"),
            # A list with wide fixed price blocks pays for grid-to-battery
            # arbitrage, which is a different business from storing own PV.
            Charged_kWh=("Charged_kWh", "mean"),
            Grid_Charged_kWh=("Grid_Charged_kWh", "mean"),
            Solves=("N_Solves", "mean"),
            Runtime_s=("Runtime_s", "mean"),
        )
        .reindex(pd.MultiIndex.from_tuples(order, names=["Tariff", "Strategy"]))
    )
    return summary, df


def summarize_by_group(df_scored, tariff=None):
    """Mean gap [%] per dataset group.

    With `tariff` the columns are that price list's strategies; without it they
    are a (Tariff, Strategy) MultiIndex covering every list in `df_scored`.
    """
    if tariff is not None:
        df_scored = df_scored[df_scored["Tariff"] == tariff]
        order = [s for s in STRATEGIES if s in set(df_scored["Strategy"])]
        columns, reindex_to = "Strategy", order
    else:
        columns = ["Tariff", "Strategy"]
        reindex_to = pd.MultiIndex.from_tuples(
            [(t, s) for t in TARIFF_ORDER for s in STRATEGIES
             if (t, s) in set(zip(df_scored["Tariff"], df_scored["Strategy"]))],
            names=columns,
        )
    return (
        df_scored.pivot_table(index="Dataset", columns=columns, values="Gap_pct",
                              aggfunc="mean")
        .reindex(columns=reindex_to)
        .reindex([g for g in DATASET_GROUPS if g in set(df_scored["Dataset"])])
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MILP optimization-horizon comparison")
    parser.add_argument("--per-group", type=int, default=HOUSEHOLDS_PER_GROUP)
    parser.add_argument("--groups", type=str, default=",".join(DATASET_GROUPS))
    parser.add_argument("--tariffs", type=str, default=",".join(TARIFF_ORDER),
                        help=f"price lists to run, comma separated ({', '.join(TARIFF_ORDER)})")
    parser.add_argument("--steps", type=int, default=None,
                        help="horizon length in intervals (default: whole dataset)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    units = study_units(args.groups.split(","), args.per_group)
    tariffs = [t.strip() for t in args.tariffs.split(",") if t.strip()]
    unknown = [t for t in tariffs if t not in TARIFFS]
    if unknown:
        parser.error(f"unknown tariff(s) {unknown}; known: {', '.join(TARIFF_ORDER)}")
    print(f"Running {len(units)} household-units x {len(tariffs)} price lists "
          f"x {len(STRATEGIES)} strategies, {args.workers} workers -> {args.output}", flush=True)
    started = time.time()
    run_batch(units, tariffs=tariffs, n_steps=args.steps, output_dir=args.output,
              n_workers=args.workers)
    print(f"Batch finished in {(time.time() - started) / 3600:.2f} h", flush=True)
