"""Optimization-horizon comparison for the perfect-foresight MILP.

The same MILP formulation is driven in several ways over the same household and
battery, on each of four GEN-I price lists, and the executed trajectories are
compared:

    horizon   how much of the future each solve sees (day / week / month / all)
    execution "block"    -- execute everything planned, then re-plan
              "receding" -- execute the first interval only, then re-plan (MPC)
    soc_mode  "fixed50"  -- every solve starts AND ends at 50 % of capacity
              "carry"    -- only the first period starts at 50 %; the next one
                            starts wherever the last one ended
    tariff    which price list the dispatch is optimized and billed under

The price list is a full axis of the study: a strategy's gap is always measured
against the whole-year solve of the same household on the same list, because how
far ahead a controller must look is a property of the price signal.
`full_period` is a relaxation of every other strategy, so its cost is a lower
bound and every gap is non-negative.

The MILP only produces a dispatch. Every executed trajectory is priced
afterwards by one shared evaluator carrying a single running peak state, because
the excess-power charge is a running maximum and a strategy cannot be allowed to
price its own peaks. `_PeakSeedView` feeds each sub-solve the peak the executed
trajectory has actually set.

The excess-power peak and the agreed billing power both reset every calendar
month (`PEAK_RESET_MONTHS`, `AGREED_POWER_TAG`). The agreed power is unbounded --
the regulatory floor and the connection-power ceiling both need a connection
agreement the profiles do not carry -- and is derived from the NO-BATTERY profile
so it stays exogenous to the dispatch. The first month has no predecessor and
`AGREED_POWER_BOOTSTRAP = "cyclic"` gives it the last complete month of the same
dataset.

The year is always closed: both SOC modes start at 50 % and the final period must
end at 50 %, so a free-terminal strategy cannot sell off its opening charge and
book it as a saving. Only the interior boundaries differ between the families.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

import Data_Loader as dl
# The battery/solver constants are imported to be RE-EXPORTED: Community_Study
# and both sizing notebooks read them as `hc.<NAME>`, which is the convention
# that keeps every study solving the same battery to the same gap.
from MILP_Household import (  # noqa: F401
    C_RATE,
    CHARGE_EFFICIENCY,
    DISCHARGE_EFFICIENCY,
    FULL_PERIOD_GAP_REL,
    FULL_PERIOD_TIME_LIMIT_S,
    INVERTER_MAX_KW,
    MAX_DAILY_CYCLES,
    SOC_FRACTION,
    SOC_MAX_FRAC,
    SOC_MIN_FRAC,
    STEPS_PER_DAY,
    build_household_env,
    full_period_solver,
    solve_household,
)
from Pricing_Functions import calculate_interval_price, oznaka_razporeda_moci

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
# Four GEN-I household products, every one run through every strategy on every
# household. "1T"/"2T" say whether the list meters on a single rate or a VT/MT
# pair.
#
#   Dinamični  GENI_SAMO_DINAMICNI  SIPX +- 0.01199, uncapped, credited per interval
#   Aktivni    GENI_SAMO_AKTIVNI    4 fixed blocks, 0.04090 -> 0.19290 in, 0.00190 -> 0.14990 out
#   Redni 1T   GENI_SAMO_REDNI      0.10290 flat in, 0.05390 flat out
#   Redni 2T   GENI_REDNI           VT 0.11990 / MT 0.09790 in, no buyback at all
#
# Every list runs under `si_samooskrba`: intra-interval netting is physical, and
# what the contract decides is only what a surplus is worth, read off the package
# (`tip_odkupa=NI` on GENI_REDNI credits an export 0.00000).
#
# NET metering is absent: closed to new contracts, settled once a year, so it is
# not a list a household can choose into and not a baseline.
#
# `scope` says which households may sign the list, enforced in `study_jobs`.
# GEN-I publishes no two-tariff samooskrba list, so "Redni 2T" is the plain
# GENI_REDNI and is restricted to the no-PV groups: run against a PV roof it
# would price every exported kWh at zero and read as a tariff result when it is
# a product-eligibility one.
#
# `slug` is the results-file suffix; the default list carries none.
PRICING_SCHEME = "si_samooskrba"
TARIFFS = {
    "Dinamični": {"paket_id": "GENI_SAMO_DINAMICNI", "scheme": PRICING_SCHEME, "slug": None,
                  "structure": "dynamic (SIPX)", "buyback": True, "scope": "all"},
    "Aktivni":   {"paket_id": "GENI_SAMO_AKTIVNI",   "scheme": PRICING_SCHEME, "slug": "aktivni",
                  "structure": "4 time blocks", "buyback": True, "scope": "all"},
    "Redni 1T":  {"paket_id": "GENI_SAMO_REDNI",     "scheme": PRICING_SCHEME, "slug": "redni1t",
                  "structure": "flat (ET)", "buyback": True, "scope": "all"},
    "Redni 2T":  {"paket_id": "GENI_REDNI",          "scheme": PRICING_SCHEME, "slug": "redni2t",
                  "structure": "two-tariff (VT/MT)", "buyback": False, "scope": "no_pv"},
}
TARIFF_ORDER = list(TARIFFS)
DEFAULT_TARIFF = "Dinamični"
PAKET_ID = TARIFFS[DEFAULT_TARIFF]["paket_id"]   # kept for callers that want one id

PRICING_REFERENCE_YEAR = 2026
# The excess-power ("presezna moc") charge is settled per calendar month, so the
# running peak it is measured on resets on the 1st.
PEAK_RESET_MONTHS = 1
# Stamped on every result row: a CSV written under a different ratchet rule
# holds a different quantity, not an older one.
PEAK_RESET_TAG = "never" if not PEAK_RESET_MONTHS else f"{PEAK_RESET_MONTHS}m"

# --- Dogovorjena obracunska moc (agreed billing power) ----------------------
# The per-block kW vector both the network power charge and the excess-power
# charge are measured against, re-set every month to the peak power the previous
# month realized in each block. `Environment._build_agreed_power_schedule` builds
# it from the no-battery profile, so it stays exogenous to the dispatch.
#
# No floor and no ceiling: both need a connection agreement the Fluvius profiles
# do not carry, and assuming one only manufactures excess charges that measure
# the assumption. Only the Akt's monotonicity rule is kept.
CONNECTION_POWER_KW = None        # no ceiling
MIN_AGREED_POWER_KW = 0.0         # no floor
AGREED_POWER_LAG_MONTHS = 1       # month M is set from month M-1's peaks
# How the first month gets a contract, having no predecessor in the data.
# "cyclic" reads the last complete month of the same dataset.
AGREED_POWER_BOOTSTRAP = "cyclic"
# Stamped on every result row, like PEAK_RESET_TAG, and derived by the same
# function the environment uses so the tag cannot drift from the rule.
AGREED_POWER_TAG = oznaka_razporeda_moci(
    minimalna_moc_kw=MIN_AGREED_POWER_KW,
    prikljucna_moc_kw=CONNECTION_POWER_KW,
    zamik_mesecev=AGREED_POWER_LAG_MONTHS,
    zacetek=AGREED_POWER_BOOTSTRAP,
)
# What rows without the column were priced under.
LEGACY_AGREED_POWER_TAG = "flat_peak_over_1.5"

BATTERY_CAPACITY_KWH = 30.0
# The battery, horizon and solver settings are `MILP_Household`'s, imported
# above rather than re-typed: SOC_FRACTION, CHARGE_EFFICIENCY,
# DISCHARGE_EFFICIENCY, C_RATE, INVERTER_MAX_KW, STEPS_PER_DAY,
# MAX_DAILY_CYCLES, SOC_MIN_FRAC, SOC_MAX_FRAC,
# FULL_PERIOD_GAP_REL and FULL_PERIOD_TIME_LIMIT_S. They are re-exported from
# here, which is where the other studies read them from.

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

# The whole-period solve is the benchmark every rolling horizon is scored
# against, because it is the cheapest bill any of them could reach. That holds
# only while the objective IS the bill: switch a wear price on
# (`cycle_cost_eur_per_efc`) and the full-period solve minimises bill + wear
# instead, so a shorter horizon can post a lower `Cum_Cost` and the regret goes
# negative. Enable it here only after deciding what the reference should be.
REFERENCE_STRATEGY = "full_period"
# Below this the whole-year optimum is not an achievable saving any percentage
# should be taken of.
MIN_ACHIEVABLE_EUR = 5.0
# A household-unit is one (dataset, household) profile; a run key adds the price
# list, because a strategy is scored against the optimum for that same list.
UNIT_COLUMNS = ["Dataset", "Household"]
KEY_COLUMNS = UNIT_COLUMNS + ["Tariff"]

_BLOCKS = (1, 2, 3, 4, 5)


def has_pv(dataset):
    """Whether a Fluvius group carries a PV roof, i.e. can ever export."""
    return "PV" in str(dataset).split("_")


def tariff_allowed(tariff, dataset):
    """Whether `dataset`'s households may sign `tariff` (see TARIFFS["scope"])."""
    scope = TARIFFS[tariff].get("scope", "all")
    if scope == "no_pv":
        return not has_pv(dataset)
    if scope == "pv":
        return has_pv(dataset)
    return True


def study_units(groups=None, per_group=HOUSEHOLDS_PER_GROUP):
    """The (dataset, household id) pairs the study runs over."""
    groups = groups or DATASET_GROUPS
    return [(g, i) for g in groups for i in range(1, per_group + 1)]


def study_jobs(units=None, tariffs=None):
    """The (dataset, household id, tariff) triples the batch solves.
    
    Combinations the household could not contract are dropped rather than solved.
    """
    units = units or study_units()
    tariffs = tariffs or TARIFF_ORDER
    return [(g, i, t) for g, i in units for t in tariffs if tariff_allowed(t, g)]


def unit_csv_path(output_dir, dataset, household_id, tariff=DEFAULT_TARIFF):
    """Results file for one (household, price list)."""
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
    """This study's household environment: one price list, one capacity.

    The only thing it adds to the shared factory is the agreed-power rule above,
    which the other studies leave at the environment's own defaults.
    """
    if tariff not in TARIFFS:
        raise ValueError(f"Unknown tariff {tariff!r}. Known: {', '.join(TARIFF_ORDER)}")
    spec = TARIFFS[tariff]
    return build_household_env(
        data,
        capacity_kwh=capacity_kwh,
        scheme=spec["scheme"],
        paket_id=spec["paket_id"],
        pricing_reference_year=PRICING_REFERENCE_YEAR,
        peak_reset_months=PEAK_RESET_MONTHS,
        price_column=PRICE_COLUMN,
        generation_column=GENERATION_COLUMN,
        consumption_column=CONSUMPTION_COLUMN,
        agreed_power=dict(
            connection_power_kw=CONNECTION_POWER_KW,
            min_agreed_power_kw=MIN_AGREED_POWER_KW,
            agreed_power_lag_months=AGREED_POWER_LAG_MONTHS,
            agreed_power_bootstrap=AGREED_POWER_BOOTSTRAP,
        ),
    )


class _PeakSeedView:
    """Environment proxy that reports the peak the executed trajectory has set.
    
    Everything except `compute_seed_peak_kw` is forwarded untouched. The caller
    must push the evaluator's running peak in with `set_peak_state` before every
    solve.
    """

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "peak_state", {b: 0.0 for b in _BLOCKS})

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_env"), name)

    def set_peak_state(self, peak_state):
        object.__setattr__(self, "peak_state", dict(peak_state))

    def compute_seed_peak_kw(self, start_idx):
        return dict(object.__getattribute__(self, "peak_state"))


# ---------------------------------------------------------------------------
# Shared evaluator: the single source of cost for every strategy
# ---------------------------------------------------------------------------
def price_interval(env, idx, net_kwh, peak_state):
    """Price one executed interval and return (cost, components, new peak state)."""
    result = calculate_interval_price(
        smp_market_price_kwh=env.arr_price[idx],
        total_consumed_kwh=float(net_kwh),
        utc_date=env.dataset.index[idx],
        interval_minutes=env.interval_minutes,
        scheme=env.pricing_scheme,
        dogovorjena_moc=env.agreed_power_at(idx),
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


def reset_windows(env, n_steps):
    """Per-interval ratchet reset-window id (see `PEAK_RESET_MONTHS`).
    
    Read off the environment, so the evaluator and the MILP cannot disagree
    about where a month starts.
    """
    return np.asarray(env.reset_window_ids[:n_steps])


def _drop_peak_on_window_start(peak_state, windows, idx):
    """Zero the running peak when `idx` opens a new ratchet window. Idempotent."""
    if idx and windows[idx] != windows[idx - 1]:
        return {b: 0.0 for b in _BLOCKS}
    return peak_state


def no_battery_cost(env, n_steps):
    """Reference cost of the same household with no battery and no curtailment."""
    windows = reset_windows(env, n_steps)
    peak_state = {b: 0.0 for b in _BLOCKS}
    total = 0.0
    for idx in range(n_steps):
        peak_state = _drop_peak_on_window_start(peak_state, windows, idx)
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
    execution    : "block" (execute everything planned) or "receding" (execute
                   the first interval, then re-plan)
    soc_mode     : "fixed50" (every solve ends at 50 % of capacity) or "carry"
                   (only the final period is pinned)
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
    windows = reset_windows(env, n_steps)
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
        # pinned boundary.
        reaches_end = (t + span) >= n_steps
        final_soc = soc_target if (soc_mode == "fixed50" or reaches_end) else None

        # Hand the solver the peak the executed trajectory is carrying; without
        # it a sub-solve pays round-trip losses to shave a peak already charged.
        peak_state = _drop_peak_on_window_start(peak_state, windows, t)
        view.set_peak_state(peak_state)

        plan = solve_household(
            view,
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

            peak_state = _drop_peak_on_window_start(peak_state, windows, idx)
            step_cost, e_part, p_part, peak_state = price_interval(env, idx, net, peak_state)
            cost += step_cost
            energy_cost += e_part
            power_cost += p_part

            soc_trace[idx] = soc
            cost_trace[idx] = cost
            soc += ch * env.charge_efficiency - dis / env.discharge_efficiency
            # A solve that ended non-optimal would silently drift the state.
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
        # AC-side discharge only, against the NAMEPLATE pack -- the same
        # denominator `summarize_trajectory` uses for its `EFC_AC_Legacy`, so a
        # SOC window does not quietly switch this figure to the usable window.
        "Equivalent_Full_Cycles": discharged / getattr(
            env, "nominal_capacity_kwh", env.battery_capacity_kwh
        ),
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
    strategies already in it are skipped, so an interrupted batch resumes.
    """
    strategies = strategies or list(STRATEGIES)
    data = load_user(household_id, dataset)
    env = build_env(data, tariff=tariff)
    n_steps = int(env.episode_length if n_steps is None else n_steps)

    rows, traces = [], {}
    resumable = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        done = pd.read_csv(checkpoint_path)
        # Rows priced under a different ratchet rule are not this study's rows;
        # resuming into them would make the strategies disagree about the charge.
        if (peak_reset_tag(done) == {PEAK_RESET_TAG}
                and agreed_power_tag(done) == {AGREED_POWER_TAG}):
            resumable = done
        else:
            print(f"  [{dataset} {household_id} | {tariff}] checkpoint written under "
                  f"peak reset {sorted(peak_reset_tag(done))} / agreed power "
                  f"{sorted(agreed_power_tag(done))} != {PEAK_RESET_TAG} / "
                  f"{AGREED_POWER_TAG}; recomputing from scratch", flush=True)
    if resumable is not None:
        rows = resumable.to_dict("records")
        baseline = float(resumable["No_Battery_EUR"].iloc[0])
        strategies = [s for s in strategies if s not in set(resumable["Strategy"])]
    else:
        baseline = no_battery_cost(env, n_steps)

    for name in strategies:
        horizon_kind, execution, soc_mode = STRATEGIES[name]
        solver = (full_period_solver() if horizon_kind == "period"
                  else pulp.PULP_CBC_CMD(msg=False))
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
            Peak_Reset=PEAK_RESET_TAG,
            Agreed_Power=AGREED_POWER_TAG,
        )
        rows.append(out)
        if verbose:
            print(f"  [{dataset} {household_id} | {tariff}] {name:20s} {out['Cost_EUR']:9.2f} EUR  "
                  f"({out['N_Solves']:6d} solves, {out['Runtime_s']:7.1f} s)", flush=True)
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(rows)
    # A CSV with no Tariff column is the default list by construction.
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
    
    One CSV per (household, price list); finished work is skipped.
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


def peak_reset_tag(df):
    """The ratchet rule(s) a result frame was priced under, as a set of tags."""
    if "Peak_Reset" not in df.columns:
        return {"never"}
    return set(df["Peak_Reset"].fillna("never").astype(str))


def sample_tag(df):
    """The `per_group` a cached comparison frame was computed on.
    
    A frame written for 3 households per group is a different sample, not an
    older one.
    """
    if "Per_Group" not in df.columns:
        return {None}
    return {int(v) for v in pd.unique(df["Per_Group"])}


def agreed_power_tag(df):
    """The agreed-power rule(s) a result frame was priced under, as a set of tags."""
    if "Agreed_Power" not in df.columns:
        return {LEGACY_AGREED_POWER_TAG}
    return set(df["Agreed_Power"].fillna(LEGACY_AGREED_POWER_TAG).astype(str))


def collect_results(output_dir=RESULTS_DIR, require_current_peak_reset=True):
    """Concatenate every per-(household, price list) CSV written so far.
    
    Rows priced under a superseded ratchet reset or agreed billing power are
    dropped by default: the charge they contain is a different quantity, so
    mixing them into a mean would compare two studies.
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
        if "Peak_Reset" not in df.columns:
            df["Peak_Reset"] = "never"
        df["Peak_Reset"] = df["Peak_Reset"].fillna("never").astype(str)
        if "Agreed_Power" not in df.columns:
            df["Agreed_Power"] = LEGACY_AGREED_POWER_TAG
        df["Agreed_Power"] = df["Agreed_Power"].fillna(LEGACY_AGREED_POWER_TAG).astype(str)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    if require_current_peak_reset:
        for column, current in (("Peak_Reset", PEAK_RESET_TAG),
                                ("Agreed_Power", AGREED_POWER_TAG)):
            stale = out[column] != current
            if stale.any():
                print(f"collect_results: dropped {int(stale.sum())} of {len(out)} rows priced "
                      f"under {column} {sorted(set(out.loc[stale, column]))} "
                      f"(current: {current!r}). Re-run the batch to replace them.",
                      flush=True)
                out = out[~stale].reset_index(drop=True)
    return out


def score(df_all, reference=REFERENCE_STRATEGY):
    """Add per-unit gap columns measured against the perfect-foresight optimum.
    
    The optimum is resolved per (household, price list), so the gaps never mix
    one price list's bill into another's.
    """
    optimum = (
        df_all[df_all["Strategy"] == reference]
        .set_index(KEY_COLUMNS)["Cost_EUR"]
        .rename("Optimum_EUR")
    )
    df = df_all.join(optimum, on=KEY_COLUMNS)
    df["Gap_to_Optimum_EUR"] = df["Cost_EUR"] - df["Optimum_EUR"]

    # A percentage of the achievable saving is only meaningful while there is
    # one: an optimum worth two cents turns an 8 EUR shortfall into 36 000 %.
    # Such units are dropped from the percentage, never from Gap_to_Optimum_EUR.
    df["Achievable_EUR"] = df["No_Battery_EUR"] - df["Optimum_EUR"]
    achievable = df["Achievable_EUR"].where(lambda s: s > MIN_ACHIEVABLE_EUR)
    df["Saving_Captured_pct"] = 100.0 * df["Savings_EUR"] / achievable
    # The share of the achievable saving a strategy gives up by not seeing the
    # whole year. 0 % is the optimum.
    df["Gap_pct"] = 100.0 - df["Saving_Captured_pct"]
    return df


def _pooled_gap_pct(group):
    """Gap as a share of the achievable saving, pooled over the units.
    
    Sum of gaps over sum of achievable, so every unit is weighted by the euros
    actually at stake.
    """
    denom = group["Achievable_EUR"].sum()
    return 100.0 * group["Gap_to_Optimum_EUR"].sum() / denom if abs(denom) > 1e-9 else np.nan


def summarize(df_all, reference=REFERENCE_STRATEGY):
    """(price list, strategy) comparison table, averaged over household-units."""
    if df_all.empty:
        return pd.DataFrame(), df_all

    df = score(df_all, reference)
    order = [(t, s) for t in TARIFF_ORDER for s in STRATEGIES
             if (t, s) in set(zip(df["Tariff"], df["Strategy"]))]
    pooled = df.groupby(["Tariff", "Strategy"])[
        ["Gap_to_Optimum_EUR", "Achievable_EUR"]
    ].apply(_pooled_gap_pct).rename("Gap_pct_pooled")
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
            # arbitrage, a different business from storing own PV.
            Charged_kWh=("Charged_kWh", "mean"),
            Grid_Charged_kWh=("Grid_Charged_kWh", "mean"),
            Solves=("N_Solves", "mean"),
            Runtime_s=("Runtime_s", "mean"),
        )
        .join(pooled)
        .reindex(pd.MultiIndex.from_tuples(order, names=["Tariff", "Strategy"]))
    )
    return summary, df


def summarize_by_group(df_scored, tariff=None):
    """Mean gap [%] per dataset group.
    
    With `tariff` the columns are that price list's strategies; without it they
    are a (Tariff, Strategy) MultiIndex.
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


# ---------------------------------------------------------------------------
# Which price list should a household with no battery sign?
# ---------------------------------------------------------------------------
# A separate question from the horizon study, which runs the three GEN-I
# samooskrba lists -- all `zahteva_pv=True`, sold only with a self-consumption
# device. A household with no PV chooses from the plain supply lists instead,
# billed under `si_dobava`.
#
# With no PV and no battery the dispatch is fixed, so this is a pure price-list
# question and the answer is the cheapest annual bill. The per-interval unit rate
# depends only on the calendar and the market price, so one pass prices every
# household.
CONSUMER_SCHEME = "si_dobava"
PROSUMER_SCHEME = "si_samooskrba"
NO_PV_GROUPS = [g for g in DATASET_GROUPS if not has_pv(g)]
PV_GROUPS = [g for g in DATASET_GROUPS if has_pv(g)]


# ---------------------------------------------------------------------------
# Price OPTIONS: a price list plus, where the supplier publishes both, which
# metering variant the household is on.
# ---------------------------------------------------------------------------
# A "tarifni" list usually publishes three supplier rates: VT and MT for a
# two-tariff meter and a single ET rate for a one-tariff one. They are
# alternative products on the same cenik, and the household picks one when it
# picks its meter.
#
# `si_obracun._cena_prevzema` bills VT/MT whenever the package carries them, so
# the ET reading is `dataclasses.replace(paket, vt=0, mt=0)`. That variant is
# built here and passed as an object rather than registered in `PAKETI`.
#
# Lists priced on a 4-block (AKTIVNI) or spot-linked (DINAMICNI) rate have no
# such choice: their `et` field is a fallback for missing data, not a product.
VARIANT_2T = "2T"          # VT/MT, two-tariff meter
VARIANT_1T = "1T"          # ET, single-tariff meter
VARIANT_NA = "-"           # the list has no VT/MT-vs-ET choice to make

_OPTION_PAKETI = {}        # option id -> Paket (base or derived variant)


def _register_options(paket_id):
    """The option ids one catalogued list expands into, cached in _OPTION_PAKETI."""
    import dataclasses

    from Pricing_Functions import PAKETI, TipCene

    p = PAKETI[paket_id]
    offers_both = p.tip_cene is TipCene.TARIFNI and (p.vt or p.mt) and p.et
    if not offers_both:
        _OPTION_PAKETI[paket_id] = p
        return [paket_id]

    two, one = f"{paket_id}@{VARIANT_2T}", f"{paket_id}@{VARIANT_1T}"
    # The 2T option is the catalogued package: VT/MT is what _cena_prevzema
    # already reaches for.
    _OPTION_PAKETI[two] = dataclasses.replace(p, id=two)
    _OPTION_PAKETI[one] = dataclasses.replace(p, id=one, vt=0.0, mt=0.0)
    return [two, one]


VARIANT_NOSALE = "NOSALE"  # synthetic: the same list with its export clause struck


def _zero_buyback_control(option_id):
    """Register a synthetic twin of `option_id` whose exported kWh is worth zero.
    
    Every real samooskrba list pays something for a surplus, so the control for
    `pv_tariff_comparison(buyback=False)` has to be built rather than found.
    Synthetic and not on any shelf -- keep it out of price-list tables.
    """
    import dataclasses

    from Pricing_Functions import TipOdkupa

    control = f"{option_base_id(option_id)}@{VARIANT_NOSALE}"
    if control not in _OPTION_PAKETI:
        _OPTION_PAKETI[control] = dataclasses.replace(
            option_paket(option_id), id=control, tip_odkupa=TipOdkupa.NI)
    return control


def option_paket(option_id):
    """The `Paket` an option is billed on, building it on first use."""
    if option_id not in _OPTION_PAKETI:
        _register_options(option_base_id(option_id))
    return _OPTION_PAKETI[option_id]


def option_base_id(option_id):
    """The catalogue key an option belongs to (`GENI_REDNI@2T` -> `GENI_REDNI`)."""
    return str(option_id).split("@", 1)[0]


def option_variant(option_id):
    """Which metering variant an option is: '2T', '1T', or '-' where there is no choice."""
    parts = str(option_id).split("@", 1)
    return parts[1] if len(parts) == 2 else VARIANT_NA


def option_label(option_id):
    """Option id with the variant spelled out, for a table or an axis."""
    variant = option_variant(option_id)
    base = option_base_id(option_id)
    return base if variant == VARIANT_NA else f"{base} [{variant}]"


def tariff_structure_label(option_id):
    """What the household is actually billed on, in words."""
    from Pricing_Functions import TipCene

    p = option_paket(option_id)
    variant = option_variant(option_id)
    if variant == VARIANT_2T:
        return f"2T: VT {p.vt:.5f} / MT {p.mt:.5f}"
    if variant == VARIANT_1T:
        return f"1T: ET {p.et:.5f}"
    if p.tip_cene is TipCene.DINAMICNI:
        return f"dynamic: SIPX + {p.pribitek_odjem:.5f}"
    if p.tip_cene is TipCene.AKTIVNI:
        return (f"4 blocks: {min(p.soncna_ns, p.soncna_vs, p.osnovna, p.konicna):.5f}"
                f" -> {max(p.soncna_ns, p.soncna_vs, p.osnovna, p.konicna):.5f}")
    return f"1T: ET {p.et:.5f}"      # tarifni with no VT/MT published at all


def consumer_price_lists(provider=None):
    """Every option a household without a self-consumption device can sign.
    
    One entry per (list, metering variant), so a supplier publishing both a
    VT/MT and an ET rate contributes two rows.
    """
    from Pricing_Functions import PAKETI

    lists = [pid for pid, p in PAKETI.items() if not p.zahteva_pv]
    if provider is not None:
        lists = [pid for pid in lists if PAKETI[pid].dobavitelj == provider]
    lists.sort(key=lambda pid: (PAKETI[pid].dobavitelj, pid))
    return [opt for pid in lists for opt in _register_options(pid)]


def prosumer_price_lists(provider=None, include_net_metering=False):
    """Every *samooskrba* option a household with a PV roof can sign.
    
    No samooskrba list in the catalogue publishes a VT/MT pair, so the metering
    variant never expands here: a prosumer chooses between lists, not variants.
    NET metering is left out by default, being closed to new consents and
    settled once a year rather than per interval.
    """
    from Pricing_Functions import PAKETI, TipOdkupa

    lists = [
        pid for pid, p in PAKETI.items()
        if p.zahteva_pv
        and (include_net_metering or p.tip_odkupa is not TipOdkupa.NET_METERING)
    ]
    if provider is not None:
        lists = [pid for pid in lists if PAKETI[pid].dobavitelj == provider]
    lists.sort(key=lambda pid: (PAKETI[pid].dobavitelj, pid))
    return [opt for pid in lists for opt in _register_options(pid)]


def buyback_rate_label(option_id):
    """One-line description of what an option pays for an exported kWh."""
    from Pricing_Functions import TipOdkupa

    p = option_paket(option_id)
    if p.tip_odkupa is TipOdkupa.NI:
        return "no buyback"
    if p.tip_odkupa is TipOdkupa.NET_METERING:
        return "annual netting"
    if p.tip_odkupa is TipOdkupa.DINAMICNI:
        return f"SIPX - {p.pribitek_oddaja:.5f}"
    if p.tip_odkupa is TipOdkupa.AKTIVNI:
        rates = [p.odkup_soncna_ns, p.odkup_soncna_vs, p.odkup_osnovna, p.odkup_konicna]
        return f"4 blocks, {min(rates):.5f} -> {max(rates):.5f}"
    return f"{p.odkup_fiksni:.5f} flat"


_RATE_CACHE = {}


def _unit_import_rates(env, option_id, n_steps, scheme=CONSUMER_SCHEME):
    """Per-interval delivered price of one imported kWh [EUR], VAT included.
    
    Supplier energy + network energy + levies, excluding the fixed and
    excess-power charges. A function of the calendar and the market price only,
    so it is cached across households.
    """
    key = ("import", scheme, option_id, n_steps, env.dataset.index[0],
           env.dataset.index[n_steps - 1],
           float(np.round(env.arr_price[:n_steps].sum(), 9)))
    hit = _RATE_CACHE.get(key)
    if hit is not None:
        return hit

    rates = np.empty(n_steps, dtype=float)
    for idx in range(n_steps):
        result = calculate_interval_price(
            smp_market_price_kwh=env.arr_price[idx],
            total_consumed_kwh=1.0,
            utc_date=env.dataset.index[idx],
            interval_minutes=env.interval_minutes,
            scheme=scheme,
            paket_id=option_paket(option_id),
            pricing_reference_year=PRICING_REFERENCE_YEAR,
        )
        rates[idx] = result["variable_price_aud"]
    _RATE_CACHE[key] = rates
    return rates


def _unit_export_credits(env, option_id, n_steps, scheme=PROSUMER_SCHEME):
    """Per-interval credit for one exported kWh [EUR], not subject to VAT.
    
    Read off the package's `tip_odkupa`. Carries no network charge and no levy,
    so it is not the mirror of the import rate, and on a dynamic list it can go
    negative when SIPX does.
    """
    key = ("export", scheme, option_id, n_steps, env.dataset.index[0],
           env.dataset.index[n_steps - 1],
           float(np.round(env.arr_price[:n_steps].sum(), 9)))
    hit = _RATE_CACHE.get(key)
    if hit is not None:
        return hit

    credits = np.empty(n_steps, dtype=float)
    for idx in range(n_steps):
        result = calculate_interval_price(
            smp_market_price_kwh=env.arr_price[idx],
            total_consumed_kwh=0.0,
            total_produced_kwh=1.0,
            utc_date=env.dataset.index[idx],
            interval_minutes=env.interval_minutes,
            scheme=scheme,
            paket_id=option_paket(option_id),
            pricing_reference_year=PRICING_REFERENCE_YEAR,
        )
        credits[idx] = result["dobropis_odkup_eur"]
    _RATE_CACHE[key] = credits
    return credits


def _consumer_fixed_and_excess(env, option_id, n_steps, import_kwh,
                               scheme=CONSUMER_SCHEME):
    """(fixed charge, excess-power charge) for the whole horizon [EUR].
    
    Both are per-month quantities: the fixed part is priced once per month, the
    excess part is the monthly peak per block over the agreed power.
    """
    hours = env.interval_minutes / 60.0
    blocks = np.asarray(env.tariff_blocks[:n_steps])
    windows = reset_windows(env, n_steps)
    power_kw = import_kwh / hours

    # Keyed on the calendar month, not on the ratchet window: the agreed power
    # the fixed charge is computed from changes on the 1st.
    fixed = 0.0
    months = np.asarray(env.month_ids[:n_steps])
    for month in np.unique(months):
        mask = months == month
        first = int(np.flatnonzero(mask)[0])
        per_interval = calculate_interval_price(
            smp_market_price_kwh=env.arr_price[first],
            total_consumed_kwh=0.0,
            utc_date=env.dataset.index[first],
            interval_minutes=env.interval_minutes,
            scheme=scheme,
            paket_id=option_paket(option_id),
            pricing_reference_year=PRICING_REFERENCE_YEAR,
            dogovorjena_moc=env.agreed_power_at(first),
        )["constant_price_aud"]
        fixed += per_interval * int(mask.sum())

    # The excess charge is identical on every list -- it is network, not supply --
    # but it is returned per list so a caller can total one bill in one place.
    excess = 0.0
    peak_state = {b: 0.0 for b in _BLOCKS}
    for idx in range(n_steps):
        peak_state = _drop_peak_on_window_start(peak_state, windows, idx)
        block = int(blocks[idx])
        prev = peak_state.get(block, 0.0)
        if power_kw[idx] > prev:
            result = calculate_interval_price(
                smp_market_price_kwh=env.arr_price[idx],
                total_consumed_kwh=float(import_kwh[idx]),
                utc_date=env.dataset.index[idx],
                interval_minutes=env.interval_minutes,
                scheme=scheme,
                paket_id=option_paket(option_id),
                pricing_reference_year=PRICING_REFERENCE_YEAR,
                dogovorjena_moc=env.agreed_power_at(idx),
                prev_peak_kw=peak_state,
            )
            excess += float(result["power_component_eur"])
            peak_state = dict(result["new_peak_kw"])
    return fixed, excess


def no_pv_tariff_comparison(groups=None, per_group=HOUSEHOLDS_PER_GROUP,
                            options=None, n_steps=None, verbose=False):
    """Annual bill of every no-PV household on every option it could sign.
    
    One row per (dataset, household, price option) with no battery, where an
    option is a price list plus, where published, the metering variant (`2T`
    VT/MT or `1T` ET). `summarize_no_pv_tariffs` turns it into the ranking table.
    """
    groups = groups or NO_PV_GROUPS
    with_pv = [g for g in groups if has_pv(g)]
    if with_pv:
        raise ValueError(
            f"{with_pv} carry PV. This comparison is defined for households that "
            f"never export; a PV roof makes the buyback terms decide the answer."
        )
    options = options or consumer_price_lists()

    rows = []
    for dataset, household_id in study_units(groups, per_group):
        data = load_user(household_id, dataset)
        # The battery parameters are irrelevant here (no dispatch), but the env
        # is what owns the tariff-block, reset-window and agreed-power arrays.
        env = build_env(data)
        steps = int(env.episode_length if n_steps is None else n_steps)
        import_kwh = np.maximum(
            env.arr_consumption[:steps] - env.arr_generation[:steps], 0.0
        )
        if verbose:
            print(f"  {dataset} {household_id}: {import_kwh.sum():.0f} kWh", flush=True)

        for option_id in options:
            energy = float(np.dot(import_kwh, _unit_import_rates(env, option_id, steps)))
            fixed, excess = _consumer_fixed_and_excess(env, option_id, steps, import_kwh)
            rows.append({
                "Dataset": dataset,
                "Household": household_id,
                "Option": option_id,
                "Paket_ID": option_base_id(option_id),
                "Variant": option_variant(option_id),
                "Cost_EUR": energy + fixed + excess,
                "Energy_EUR": energy,
                "Fixed_EUR": fixed,
                "Excess_Power_EUR": excess,
                "Import_kWh": float(import_kwh.sum()),
                "Agreed_Power": AGREED_POWER_TAG,
                "Per_Group": int(per_group),
            })
    return pd.DataFrame(rows)


def pv_tariff_comparison(groups=None, per_group=HOUSEHOLDS_PER_GROUP,
                         options=None, n_steps=None, buyback=True,
                         verbose=False):
    """Annual bill of every PV household on every samooskrba list, no battery.
    
    With no battery there is no dispatch decision, so this is a pure price-list
    question. Netting inside the interval is physical, so what the list decides
    is only what the residual surplus is worth.
    
    `buyback=False` prices every exported kWh at zero while leaving the import
    side untouched, so the difference between the two runs is the annual worth
    of the export clause. No catalogued list is that contract any more.
    
    The Fluvius columns are meter registers, so the roof's self-consumption has
    already been netted away before the data starts: what is priced here is the
    residual, and the flag moves only the surplus.
    """
    groups = groups or PV_GROUPS
    without_pv = [g for g in groups if not has_pv(g)]
    if without_pv:
        raise ValueError(
            f"{without_pv} carry no PV. A samooskrba list requires a "
            f"self-consumption device (zahteva_pv=True); use "
            f"no_pv_tariff_comparison for those groups."
        )
    options = options or prosumer_price_lists()

    rows = []
    for dataset, household_id in study_units(groups, per_group):
        data = load_user(household_id, dataset)
        env = build_env(data)
        steps = int(env.episode_length if n_steps is None else n_steps)
        net_kwh = env.arr_consumption[:steps] - env.arr_generation[:steps]
        import_kwh = np.maximum(net_kwh, 0.0)
        export_kwh = np.maximum(-net_kwh, 0.0)
        # Not self-consumption -- see the note above. Both registers are non-zero
        # in the same 15-minute bucket only where the meter netted at a finer
        # resolution, so this is an aggregation artefact and is reported as a
        # diagnostic on the data, not as a quantity a contract prices.
        overlap_kwh = float(np.minimum(env.arr_consumption[:steps],
                                       env.arr_generation[:steps]).sum())
        if verbose:
            print(f"  {dataset} {household_id}: {import_kwh.sum():.0f} kWh offtake, "
                  f"{export_kwh.sum():.0f} kWh injection", flush=True)

        for option_id in options:
            import_eur = float(np.dot(
                import_kwh, _unit_import_rates(env, option_id, steps, PROSUMER_SCHEME)))
            credit_eur = 0.0 if not buyback else float(np.dot(
                export_kwh, _unit_export_credits(env, option_id, steps, PROSUMER_SCHEME)))
            fixed, excess = _consumer_fixed_and_excess(
                env, option_id, steps, import_kwh, PROSUMER_SCHEME)
            rows.append({
                "Dataset": dataset,
                "Household": household_id,
                "Option": option_id,
                "Paket_ID": option_base_id(option_id),
                "Variant": option_variant(option_id),
                "Buyback": bool(buyback),
                "Cost_EUR": import_eur - credit_eur + fixed + excess,
                "Energy_EUR": import_eur - credit_eur,
                "Import_EUR": import_eur,
                "Export_Credit_EUR": credit_eur,
                "Fixed_EUR": fixed,
                "Excess_Power_EUR": excess,
                "Import_kWh": float(import_kwh.sum()),
                "Export_kWh": float(export_kwh.sum()),
                "Register_Overlap_kWh": overlap_kwh,
                "Agreed_Power": AGREED_POWER_TAG,
                "Per_Group": int(per_group),
            })
    return pd.DataFrame(rows)


def _rank_tariff_bills(df, extra_means=()):
    """Rank price lists by mean annual bill, with the gap to the cheapest.
    
    `Delta_EUR` / `Delta_pct` are against the cheapest list for that same
    household, then averaged. `Cheapest_For` counts the households it wins.
    """
    from Pricing_Functions import PAKETI

    per_household = df.set_index(["Dataset", "Household", "Option"])["Cost_EUR"]
    best = per_household.groupby(level=["Dataset", "Household"]).min()
    winner = per_household.groupby(level=["Dataset", "Household"]).idxmin()

    df = df.copy()
    keys = list(zip(df["Dataset"], df["Household"]))
    df["Best_EUR"] = [best[k] for k in keys]
    df["Delta_EUR"] = df["Cost_EUR"] - df["Best_EUR"]
    df["Delta_pct"] = 100.0 * df["Delta_EUR"] / df["Best_EUR"]

    wins = pd.Series([w[2] for w in winner], index=winner.index).value_counts()
    aggregations = dict(
        Households=("Cost_EUR", "size"),
        Cost_EUR=("Cost_EUR", "mean"),
        Delta_EUR=("Delta_EUR", "mean"),
        Delta_pct=("Delta_pct", "mean"),
        Worst_Delta_pct=("Delta_pct", "max"),
        Energy_EUR=("Energy_EUR", "mean"),
        Fixed_EUR=("Fixed_EUR", "mean"),
        Excess_Power_EUR=("Excess_Power_EUR", "mean"),
    )
    aggregations.update({c: (c, "mean") for c in extra_means if c in df.columns})
    summary = df.groupby("Option").agg(**aggregations).sort_values("Cost_EUR")
    summary.insert(0, "Supplier",
                   [PAKETI[option_base_id(o)].dobavitelj for o in summary.index])
    # What the household is billed on, spelled out: "2T: VT .. / MT .." vs
    # "1T: ET ..", so no table can leave which variant was used implicit.
    summary.insert(1, "Structure", [tariff_structure_label(o) for o in summary.index])
    summary.insert(2, "Variant", [option_variant(o) for o in summary.index])
    summary["Cheapest_For"] = [int(wins.get(p, 0)) for p in summary.index]
    return summary


def summarize_no_pv_tariffs(df_no_pv):
    """Ranking table for `no_pv_tariff_comparison`."""
    return _rank_tariff_bills(df_no_pv)


def summarize_pv_tariffs(df_pv):
    """Ranking table for `pv_tariff_comparison`, with the export side broken out.
    
    `Buyback` is carried through so a credited run and a zero-export run cannot
    be ranked in the same table.
    """
    modes = set(df_pv["Buyback"]) if "Buyback" in df_pv.columns else {True}
    if len(modes) > 1:
        raise ValueError(
            "df_pv mixes credited and zero-export rows. Rank each separately, "
            "then join on Paket_ID."
        )
    summary = _rank_tariff_bills(
        df_pv, extra_means=("Import_EUR", "Export_Credit_EUR", "Export_kWh"))
    summary.insert(3, "Buyback", [buyback_rate_label(o) if modes == {True} else "zeroed"
                                  for o in summary.index])
    return summary


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
