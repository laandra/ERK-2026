"""Optimization-horizon comparison for the perfect-foresight MILP.

Six ways of using the same MILP formulation are compared on the same household,
the same battery and the same tariff:

    day_block      solve one day,   execute the whole day,   repeat
    day_receding   solve one day,   execute the first step,  repeat  (MPC)
    week_block     solve one week,  execute the whole week,  repeat
    week_receding  solve one week,  execute the first step,  repeat  (MPC)
    month_block    solve one month, execute the whole month, repeat
    full_period    solve the whole horizon in one go (0.1 % MIP gap)

Only `full_period` sees the entire year at once, so it is the theoretical
optimum; the others show what a controller gives up by looking less far ahead,
and what it gains by re-planning every interval instead of committing to a whole
block.

Design notes
------------
*The MILP only produces a dispatch.* Every strategy's executed trajectory is
priced afterwards by one shared evaluator (`price_interval`), interval by
interval, carrying a single running peak state. That is the only way the six are
comparable: the excess-power ("konica") charge is a running maximum over the
whole horizon, so a strategy cannot be allowed to price its own peaks.

*The controller is told the truth about its peak state.* `run_milp_benchmark`
seeds each sub-solve from `env.compute_seed_peak_kw(start_idx)`, which is the
peak a household with no battery would have reached by then -- wrong for a
trajectory that has been shaving peaks all year. `_PeakSeedView` wraps the
environment and feeds back the peak the executed trajectory has actually set, so
every sub-solve starts from the state the evaluator is in.

*State of charge.* Every solve starts at the SOC the executed trajectory has
reached and is required to end the horizon at `soc_target` (50 % of capacity).
Feasibility near the end of the dataset is inductive: the horizon shrinks in
lockstep with the remaining steps, so a plan that could return to `soc_target`
at t can still do so at t+1.
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
DATASET_NAME = "Fluvius_PV"
SMP_COUNTRY_ID = "Slovenia"
PRICE_COLUMN = "SMP"
GENERATION_COLUMN = "Feed_In_Volume_kWh"
CONSUMPTION_COLUMN = "Consumption_Volume_kWh"

PRICING_SCHEME = "si_samooskrba"
PAKET_ID = "GENI_SAMO_DINAMICNI"
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

RESULTS_DIR = Path(__file__).resolve().parent / "Results" / "Horizon_Comparison"

# name -> (horizon in steps or None for "the whole remaining period", steps executed per solve)
STRATEGIES = {
    "day_block": ("day", "block"),
    "day_receding": ("day", "receding"),
    "week_block": ("week", "block"),
    "week_receding": ("week", "receding"),
    "month_block": ("month", "block"),
    "full_period": ("period", "block"),
}

_BLOCKS = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Data + environment
# ---------------------------------------------------------------------------
def load_user(household_id, dataset=DATASET_NAME, country=SMP_COUNTRY_ID):
    """Household profile with the country SMP series patched in (EUR/kWh)."""
    data = dl.load_household_data(int(household_id), dataset=dataset)
    smp = dl.load_smp_data(country).reindex(data.index, method="ffill")
    series = pd.to_numeric(smp[PRICE_COLUMN], errors="coerce").ffill().bfill()
    scale = 1000.0 if float(series.abs().quantile(0.95)) > 2.0 else 1.0
    data[PRICE_COLUMN] = (series / scale).astype(float)
    return data


def build_env(data, capacity_kwh=BATTERY_CAPACITY_KWH):
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
        pricing_scheme=PRICING_SCHEME,
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        pricing_options={"paket_id": PAKET_ID},
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


def run_strategy(env, horizon_kind, execution, n_steps=None, solver=None, verbose=False):
    """Roll one strategy over the horizon and return its executed trajectory.

    horizon_kind : "day" | "week" | "month" | "period"
    execution    : "block" (execute everything that was planned) or
                   "receding" (execute the first interval, then re-plan)
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

        plan = run_milp_benchmark(
            view,
            use_discrete_actions=False,
            start_idx=t,
            n_steps=span,
            initial_soc_kwh=soc,
            final_soc_kwh=soc_target,
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


def run_user(household_id, n_steps=None, strategies=None, keep_traces=False, verbose=True,
             checkpoint_path=None):
    """All strategies for one household. Returns a tidy DataFrame (one row each).

    With `checkpoint_path` the CSV is rewritten after every finished strategy and
    strategies already present in it are skipped, so an interrupted batch resumes
    where it stopped instead of redoing hours of solves.
    """
    strategies = strategies or list(STRATEGIES)
    data = load_user(household_id)
    env = build_env(data)
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
        horizon_kind, execution = STRATEGIES[name]
        solver = (
            pulp.PULP_CBC_CMD(msg=False, gapRel=FULL_PERIOD_GAP_REL,
                              timeLimit=FULL_PERIOD_TIME_LIMIT_S)
            if name == "full_period" else pulp.PULP_CBC_CMD(msg=False)
        )
        out = run_strategy(env, horizon_kind, execution, n_steps=n_steps, solver=solver)
        traces[name] = (out.pop("_soc_trace"), out.pop("_cost_trace"))
        out.update(
            Household=int(household_id),
            Strategy=name,
            Horizon=horizon_kind,
            Execution=execution,
            No_Battery_EUR=baseline,
            Savings_EUR=baseline - out["Cost_EUR"],
        )
        rows.append(out)
        if verbose:
            print(f"  [{household_id}] {name:14s} {out['Cost_EUR']:9.2f} EUR  "
                  f"({out['N_Solves']:6d} solves, {out['Runtime_s']:7.1f} s)", flush=True)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(rows)
    lead = ["Household", "Strategy", "Horizon", "Execution", "Cost_EUR", "Savings_EUR",
            "No_Battery_EUR", "Energy_EUR", "Power_EUR"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    return (df, traces) if keep_traces else df


# ---------------------------------------------------------------------------
# Batch driver (multiprocessing over households, one CSV per household)
# ---------------------------------------------------------------------------
def _worker(args):
    household_id, n_steps, output_dir = args
    out_path = Path(output_dir) / f"user_{household_id:03d}.csv"
    df = run_user(household_id, n_steps=n_steps, verbose=True, checkpoint_path=out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return f"{out_path} ({len(df)} strategies)"


def run_batch(household_ids, n_steps=None, output_dir=RESULTS_DIR, n_workers=10):
    """Run every household in parallel. Finished households are skipped on re-run."""
    import multiprocessing as mp

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(int(h), n_steps, str(output_dir)) for h in household_ids]

    if n_workers <= 1:
        for job in jobs:
            _worker(job)
    else:
        with mp.get_context("spawn").Pool(n_workers) as pool:
            for path in pool.imap_unordered(_worker, jobs):
                print(f"done -> {path}", flush=True)

    return collect_results(output_dir)


def collect_results(output_dir=RESULTS_DIR):
    """Concatenate every per-household CSV written so far."""
    files = sorted(Path(output_dir).glob("user_*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(df_all):
    """Strategy-level comparison table, averaged over households."""
    if df_all.empty:
        return pd.DataFrame(), df_all

    optimum = (
        df_all[df_all["Strategy"] == "full_period"]
        .set_index("Household")["Cost_EUR"]
        .rename("Optimum_EUR")
    )
    df = df_all.join(optimum, on="Household")
    df["Gap_to_Optimum_EUR"] = df["Cost_EUR"] - df["Optimum_EUR"]
    achievable = df["No_Battery_EUR"] - df["Optimum_EUR"]
    df["Saving_Captured_pct"] = 100.0 * df["Savings_EUR"] / achievable.where(achievable != 0)

    order = list(STRATEGIES)
    summary = (
        df.groupby("Strategy")
        .agg(
            Households=("Household", "nunique"),
            Cost_EUR=("Cost_EUR", "mean"),
            Savings_EUR=("Savings_EUR", "mean"),
            Saving_Captured_pct=("Saving_Captured_pct", "mean"),
            Gap_to_Optimum_EUR=("Gap_to_Optimum_EUR", "mean"),
            Worst_Gap_EUR=("Gap_to_Optimum_EUR", "max"),
            Energy_EUR=("Energy_EUR", "mean"),
            Power_EUR=("Power_EUR", "mean"),
            Cycles=("Equivalent_Full_Cycles", "mean"),
            Peak_kW=("Peak_Import_kW", "mean"),
            Solves=("N_Solves", "mean"),
            Runtime_s=("Runtime_s", "mean"),
        )
        .reindex([s for s in order if s in df["Strategy"].unique()])
    )
    return summary, df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MILP optimization-horizon comparison")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--first-user", type=int, default=1)
    parser.add_argument("--steps", type=int, default=None, help="horizon length (default: whole dataset)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    ids = list(range(args.first_user, args.first_user + args.users))
    print(f"Running {len(ids)} households, {args.workers} workers -> {args.output}", flush=True)
    started = time.time()
    run_batch(ids, n_steps=args.steps, output_dir=args.output, n_workers=args.workers)
    print(f"Batch finished in {(time.time() - started) / 3600:.2f} h", flush=True)
