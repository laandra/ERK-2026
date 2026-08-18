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

*The excess-power peak resets every calendar month* (`PEAK_RESET_MONTHS = 1`),
on both sides: the evaluator drops its running peak at each month boundary and
the MILP gives every month its own peak variable starting at zero. The network
bill is monthly, so the excess over the agreed power is a charge a household
pays twelve times a year, and a battery that shaves it earns twelve times. Under
the old never-resetting setting the running maximum was charged once for the
whole year, which made peak shaving worth a rounding error after January and was
the single biggest reason a flat price list showed no battery value at all.

*The agreed power resets every month too* (`AGREED_POWER_TAG`). The dogovorjena
obracunska moc every household is billed on is re-set on the 1st to the peak
power the previous month realized, per tariff block -- a change the Akt allows
free of charge, requested by the 8th and effective the following month. It is
unbounded: the regulatory floor and the connection-power ceiling both need a
connection agreement the profiles do not carry, and only the Akt's monotonicity
rule (a higher block never below a lower one) is applied. It is derived from the
NO-BATTERY profile so it stays exogenous to the dispatch being optimized; see
`Environment._build_agreed_power_schedule`.

*The first month is the one month with no predecessor*, and
`AGREED_POWER_BOOTSTRAP = "cyclic"` gives it the last complete month of the same
dataset -- December standing in for the December before January. The alternative
of letting it read its own peaks is not just non-causal, it is biased: the
agreed power would land exactly on that month's peak, so it could never pay an
excess charge and would contribute no peak-shaving signal. See
`Environment._bootstrap_peak_kw`.

Note the trade all of this models: a household that never touches its assigned
agreed power is permanently exempt from the excess-power charge, so managing the
figure is what makes the charge payable in the first place.

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
# `scope` says which households may sign the list, and it is enforced in
# `study_jobs`. GEN-I publishes no two-tariff *samooskrba* list -- GENI_SAMO_REDNI
# carries a single ET rate and no VT/MT at all -- so "Redni 2T" is GENI_REDNI, a
# plain supply list with `tip_odkupa=NI`. That is a contract for a household that
# never sells anything, and running it against a PV roof would price every
# exported kWh at zero and read as a tariff result when it is really a
# product-eligibility result. It is therefore restricted to the groups with no
# PV; the three samooskrba lists carry the PV groups (and, as the existing study
# defines them, the no-PV groups too, where they reduce to their import side).
# The standalone no-PV comparison across every list a consumer can actually sign
# is `no_pv_tariff_comparison` at the bottom of this module; the matching question
# for a PV roof -- which samooskrba list, and what the buyback term is worth -- is
# `pv_tariff_comparison` beside it. Both drive `No_battery_comparison.ipynb`.
#
# `slug` is the results-file suffix. The default list carries no suffix, so the
# CSVs written before this study grew a tariff axis are still read as its rows.
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
# running peak it is measured on resets on the 1st. See the module docstring.
PEAK_RESET_MONTHS = 1
# Stamped on every result row. A CSV written under a different ratchet rule is
# not comparable with one written under this one -- the excess-power charge is a
# different quantity -- so the tag travels with the numbers.
PEAK_RESET_TAG = "never" if not PEAK_RESET_MONTHS else f"{PEAK_RESET_MONTHS}m"

# --- Dogovorjena obracunska moc (agreed billing power) ----------------------
# Both the network power charge and the excess-power charge are measured against
# a per-block kW vector the household agrees with its operator. It is re-set
# every month to the peak power the previous month realized in each block --
# free of charge under the Akt if requested by the 8th, effective the 1st.
# `Environment._build_agreed_power_schedule` builds the schedule (from the
# no-battery profile, so it stays exogenous to the dispatch being optimized).
#
# NO FLOOR AND NO CEILING. The regulatory minimum and the connection-power
# ceiling are both functions of the connection agreement, which no Fluvius
# profile carries; assuming one only manufactures excess charges that measure
# the assumption rather than the household. The one rule kept is the Akt's
# monotonicity requirement (a higher block is never below a lower one), which
# needs nothing from outside the data and costs nothing to obey.
CONNECTION_POWER_KW = None        # no ceiling
MIN_AGREED_POWER_KW = 0.0         # no floor
AGREED_POWER_LAG_MONTHS = 1       # month M is set from month M-1's peaks
# How the FIRST month gets a contract, given it has no predecessor in the data.
# "cyclic" reads the last complete month of the same dataset -- on a full year
# that is the calendar month right before the first one, so the leading month is
# priced against a real, same-season predecessor instead of against its own
# outcome. See `Environment._bootstrap_peak_kw` for the alternatives.
AGREED_POWER_BOOTSTRAP = "cyclic"
# Stamped on every result row, exactly like PEAK_RESET_TAG: a bill settled
# against a different agreed power is a different number, not an older one.
# Derived from the constants above by the same function the environment uses, so
# the tag on a row cannot drift from the rule that priced it.
AGREED_POWER_TAG = oznaka_razporeda_moci(
    minimalna_moc_kw=MIN_AGREED_POWER_KW,
    prikljucna_moc_kw=CONNECTION_POWER_KW,
    zamik_mesecev=AGREED_POWER_LAG_MONTHS,
    zacetek=AGREED_POWER_BOOTSTRAP,
)
# What rows without the column were priced under: a flat historical peak / 1.5
# in every block, the pre-schedule default.
LEGACY_AGREED_POWER_TAG = "flat_peak_over_1.5"

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
# Below this the whole-year optimum is not an "achievable saving" any percentage
# should be taken of -- see `score`.
MIN_ACHIEVABLE_EUR = 5.0
# A household-unit is one (dataset, household) profile; a run key adds the price
# list, because the whole-year optimum a strategy is scored against is the
# optimum *for that same household on that same list*.
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

    Combinations the household could not contract are dropped rather than
    solved: a price list with no buyback is not a tariff a PV household can be
    scored on.
    """
    units = units or study_units()
    tariffs = tariffs or TARIFF_ORDER
    return [(g, i, t) for g, i in units for t in tariffs if tariff_allowed(t, g)]


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
        connection_power_kw=CONNECTION_POWER_KW,
        min_agreed_power_kw=MIN_AGREED_POWER_KW,
        agreed_power_lag_months=AGREED_POWER_LAG_MONTHS,
        agreed_power_bootstrap=AGREED_POWER_BOOTSTRAP,
    )


class _PeakSeedView:
    """Environment proxy that reports the peak the executed trajectory has set.

    Everything except `compute_seed_peak_kw` is forwarded untouched, so
    `run_milp_benchmark` reads the real environment's arrays, tariff and battery
    parameters.

    The caller must push the evaluator's running peak in with `set_peak_state`
    before every solve; the seed is not derived from anything the environment
    knows, because only the caller has executed the trajectory.
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

    The evaluator has to drop its running peak wherever the tariff does. The
    environment already precomputes the boundaries for the MILP, so both sides
    read the same array and cannot disagree about where a month starts.
    """
    return np.asarray(env.reset_window_ids[:n_steps])


def _drop_peak_on_window_start(peak_state, windows, idx):
    """Zero the running peak when `idx` opens a new ratchet window. Idempotent,
    so it is safe to apply both before a solve and inside the execution loop."""
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
        # boundary that is pinned, and stored energy is otherwise handed to the
        # next period exactly as the solver left it.
        reaches_end = (t + span) >= n_steps
        final_soc = soc_target if (soc_mode == "fixed50" or reaches_end) else None

        # Hand the solver the peak the executed trajectory is actually carrying.
        # Without this every sub-solve believes the month's peak is still zero
        # and pays round-trip losses to shave a peak the evaluator has already
        # been charged for. A solve that opens a new month must see the reset.
        peak_state = _drop_peak_on_window_start(peak_state, windows, t)
        view.set_peak_state(peak_state)

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

            peak_state = _drop_peak_on_window_start(peak_state, windows, idx)
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
    resumable = None
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        done = pd.read_csv(checkpoint_path)
        # Rows priced under a different ratchet rule are not this study's rows.
        # Resuming into them would produce a CSV whose strategies disagree about
        # what the excess-power charge even is, and the gaps would be nonsense.
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


def peak_reset_tag(df):
    """The ratchet rule(s) a result frame was priced under, as a set of tags.

    Rows written before the charge grew a monthly reset carry no column; they
    are the never-resetting variant by construction.
    """
    if "Peak_Reset" not in df.columns:
        return {"never"}
    return set(df["Peak_Reset"].fillna("never").astype(str))


def sample_tag(df):
    """The `per_group` a cached comparison frame was computed on.

    Read alongside `agreed_power_tag` before trusting a cache: a frame written
    for 3 households per group is a different sample, not an older one, and
    averaging it as if it were 5 would silently change every mean in a table.
    Frames written before this column existed return `{None}`.
    """
    if "Per_Group" not in df.columns:
        return {None}
    return {int(v) for v in pd.unique(df["Per_Group"])}


def agreed_power_tag(df):
    """The agreed-power rule(s) a result frame was priced under, as a set of tags.

    Rows written before the agreed power grew a monthly schedule carry no
    column; they were settled against a flat historical peak / 1.5 in every
    block, which is neither the same power charge nor the same excess charge.
    """
    if "Agreed_Power" not in df.columns:
        return {LEGACY_AGREED_POWER_TAG}
    return set(df["Agreed_Power"].fillna(LEGACY_AGREED_POWER_TAG).astype(str))


def collect_results(output_dir=RESULTS_DIR, require_current_peak_reset=True):
    """Concatenate every per-(household, price list) CSV written so far.

    Files written before this study grew a tariff axis carry no Tariff column;
    they are the default price list, and are labelled as such on the way in.

    Rows priced under a superseded network-charge rule -- a different ratchet
    reset, or a different agreed billing power -- are dropped by default. They
    are not merely older: the charge they contain is a different quantity, so
    mixing them into a mean would compare two studies. Pass
    `require_current_peak_reset=False` to read them anyway.
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

    # A percentage of the achievable saving is only meaningful while there IS an
    # achievable saving. On a flat list a household with no PV and no headroom on
    # its agreed power can end the year with an optimum worth two cents, and
    # dividing a real 8 EUR shortfall by it prints 36 000 %. Those units are
    # dropped from the percentage (never from Gap_to_Optimum_EUR) so one
    # degenerate denominator cannot own a column mean.
    df["Achievable_EUR"] = df["No_Battery_EUR"] - df["Optimum_EUR"]
    achievable = df["Achievable_EUR"].where(lambda s: s > MIN_ACHIEVABLE_EUR)
    df["Saving_Captured_pct"] = 100.0 * df["Savings_EUR"] / achievable
    # The headline number: the share of the achievable saving a strategy throws
    # away by not seeing the whole year. 0 % is the optimum.
    df["Gap_pct"] = 100.0 - df["Saving_Captured_pct"]
    return df


def _pooled_gap_pct(group):
    """Gap as a share of the achievable saving, pooled over the units.

    Sum of gaps over sum of achievable, rather than the mean of per-unit
    percentages: it weights every unit by the euros actually at stake, so it
    stays finite and interpretable however small an individual denominator gets.
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
            # arbitrage, which is a different business from storing own PV.
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


# ---------------------------------------------------------------------------
# Which price list should a household with no battery sign?
# ---------------------------------------------------------------------------
# A separate question from the horizon study, and one the horizon study cannot
# answer: it runs the three GEN-I *samooskrba* lists, and all three carry
# `zahteva_pv=True` -- they are only sold with a self-consumption device. A
# household with no PV chooses from the plain supply lists instead, billed under
# `si_dobava` (no export ever occurs, so the netting in `si_samooskrba` would be
# a no-op anyway, but the scheme should still name the contract that exists).
#
# With no PV and no battery the dispatch is fixed: the household imports exactly
# what it consumes, in the interval it consumes it. So the comparison is a pure
# price-list question with no optimisation in it, and the answer is the cheapest
# annual bill. That also makes it cheap to compute: the per-interval unit rate
# depends only on the calendar and the market price, never on the household, so
# one pass per list prices every household.
CONSUMER_SCHEME = "si_dobava"
PROSUMER_SCHEME = "si_samooskrba"
NO_PV_GROUPS = [g for g in DATASET_GROUPS if not has_pv(g)]
PV_GROUPS = [g for g in DATASET_GROUPS if has_pv(g)]


# ---------------------------------------------------------------------------
# Price OPTIONS: a price list plus, where the supplier publishes both, which
# metering variant the household is on.
# ---------------------------------------------------------------------------
# A "tarifni" list usually publishes THREE supplier rates: VT and MT for a
# two-tariff meter, and a single ET rate for a one-tariff meter. They are
# alternative products on the same cenik and the household picks one when it
# picks its meter, so a comparison that silently takes one of them is comparing
# suppliers on an assumption the customer actually gets to make.
#
# `si_obracun._cena_prevzema` bills VT/MT whenever the package carries them and
# falls through to ET only when it does not -- so the ET reading of such a list
# is `dataclasses.replace(paket, vt=0, mt=0)`, which changes the supplier energy
# rate and nothing else. That variant is built here and handed to
# `calculate_interval_price` as an object rather than registered in `PAKETI`,
# so nothing outside this module sees a catalogue that grew rows.
#
# Lists priced on a 4-block (AKTIVNI) or spot-linked (DINAMICNI) rate have no
# such choice: their `et` field is a fallback for missing 15-minute data, not a
# product, and it is never offered as an alternative.
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
    # The 2T option IS the catalogued package -- billing it needs no variant,
    # because VT/MT is what _cena_prevzema already reaches for.
    _OPTION_PAKETI[two] = dataclasses.replace(p, id=two)
    _OPTION_PAKETI[one] = dataclasses.replace(p, id=one, vt=0.0, mt=0.0)
    return [two, one]


VARIANT_NOSALE = "NOSALE"  # synthetic: the same list with its export clause struck


def _zero_buyback_control(option_id):
    """Register a synthetic twin of `option_id` whose exported kWh is worth zero.

    `pv_tariff_comparison(buyback=False)` claims to change the export side and
    nothing else. Until the Petrol samooskrba entry was corrected the catalogue
    held a `tip_odkupa=NI` list that made that claim checkable for free: the two
    runs had to agree on it. Every real samooskrba list now pays *something* for
    a surplus, so the control has to be built rather than found.

    Returns an option id billed on the same package with `tip_odkupa=NI`, for
    which `buyback=True` must reproduce `buyback=False` on the base list exactly.
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
    """What the household is actually billed on, in words.

    The variant alone is not self-explanatory on a list that has no choice to
    offer, so this names the structure instead of printing a bare dash.
    """
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

    One entry per (list, metering variant), so a supplier that publishes both a
    VT/MT and an ET rate contributes two rows rather than one.
    """
    from Pricing_Functions import PAKETI

    lists = [pid for pid, p in PAKETI.items() if not p.zahteva_pv]
    if provider is not None:
        lists = [pid for pid in lists if PAKETI[pid].dobavitelj == provider]
    lists.sort(key=lambda pid: (PAKETI[pid].dobavitelj, pid))
    return [opt for pid in lists for opt in _register_options(pid)]


def prosumer_price_lists(provider=None, include_net_metering=False):
    """Every *samooskrba* option a household with a PV roof can sign.

    Same (list, metering variant) expansion as `consumer_price_lists`, except
    that here it never fires: **no** samooskrba list in the catalogue publishes a
    VT/MT pair, so every option comes back unsuffixed. That is the product and
    not a gap -- Petrol states it outright ("v samooskrbi se vsa električna
    energija obračuna samo po ET"), and GEN-I likewise quotes a single ET rate.
    A prosumer has no metering variant to choose; the tariff choice a roof faces
    is between lists, and on the aktivni and dinamični ones between time blocks.

    NET metering is left out by default: it is closed to new consents and settles
    once a year rather than per interval, so it is neither a list a household can
    choose into nor a bill this per-interval evaluator can produce. Ask for it
    explicitly if a legacy holder is what is being priced.
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

    Supplier energy + network energy + levies. Excludes both fixed charges and
    the excess-power charge, which are not per-kWh and are added separately.

    The rate is a function of the calendar and the market price only, so it is
    cached across households: every household in the study is indexed on the
    same year and carries the same SMP series, and the cache key says so.
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
    """Per-interval credit for one exported kWh [EUR], **not** subject to VAT.

    Read off the package's `tip_odkupa` (`si_obracun._cena_oddaje`): a flat rate,
    the four aktivni buyback blocks, or SIPX minus the supplier's spread. Carries
    no network charge and no levy -- an exported kWh is a credit against the
    bill, not a delivered good -- so it is not the mirror image of the import
    rate above, and on a dynamic list it can go negative when SIPX does.

    Cached on the same calendar/market key as the import rate: within one
    interval the credit is a function of the price list and the clock only.
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

    Both are per-month quantities. The fixed part is constant inside a month, so
    it is priced once per month and multiplied out; the excess part is the
    monthly peak per block over the agreed power, weighted by the block's power
    rate -- the same monthly-resetting ratchet `PEAK_RESET_MONTHS` puts on the
    MILP and the evaluator.
    """
    hours = env.interval_minutes / 60.0
    blocks = np.asarray(env.tariff_blocks[:n_steps])
    windows = reset_windows(env, n_steps)
    power_kw = import_kwh / hours

    # Keyed on the calendar month, not on the ratchet window: the fixed charge
    # is a monthly quantity and the agreed power it is computed from changes on
    # the 1st, so a window wider than a month would price every month in it off
    # the first month's contract.
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
    option is a price list plus -- on a list that publishes both -- which
    metering variant the household is on (`2T` VT/MT or `1T` ET). Returns the
    long frame; `summarize_no_pv_tariffs` turns it into the ranking table.

    `per_group` is recorded on every row as `Per_Group`, so a cached frame
    states the sample it was computed on rather than leaving a caller to assume.
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

    Same shape as `no_pv_tariff_comparison`, and the same reason to exist: with no
    battery there is no dispatch decision, so the answer is a pure price-list
    question. What is new is the export side. Netting inside the 15-minute
    interval is *physical* -- the meter registers the residual whatever the
    contract says -- so what the price list decides is only what the residual
    *surplus* is worth, and that is what separates the lists here as much as the
    import price does.

    `buyback=False` prices every exported kWh at **zero** while leaving the import
    side untouched: the household that produces its own power, nets it inside the
    interval, and gives the rest away. No catalogued list is that contract any
    more -- every samooskrba list a household can sign today values its surplus at
    something -- so the flag now prices a counterfactual rather than a shelf, and
    the difference between the two runs *is* the annual worth of the export clause.
    `_zero_buyback_control` is what remains of the identity check the old (wrong)
    `PETROL_SAMOOSKRBA` entry used to provide for free.

    Note what the flag does *not* do, and what these profiles cannot show. The
    Fluvius columns are **meter registers**: `Consumption_Volume_kWh` is grid
    offtake and `Feed_In_Volume_kWh` is grid injection, and they are very nearly
    mutually exclusive (both non-zero in ~12 % of intervals, which is the 15-min
    aggregation of a meter that netted at a finer resolution). The roof's
    self-consumption has therefore already been netted away before the data
    starts, and no contract term can reach it. What is priced here is the
    *residual* -- the import that survived the roof and the export it could not
    absorb -- which is exactly what a bill is measured on, but it means the flag
    moves only the surplus and never the far larger saving a roof makes by
    displacing retail import in the first place.
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

    `Delta_EUR` / `Delta_pct` are against the cheapest list *for that same
    household*, then averaged, so a list is not flattered by a household mix that
    happens to suit it. `Cheapest_For` counts the households it actually wins.
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

    `Buyback` is carried through so a caller cannot accidentally rank a credited
    run and a zero-export run in the same table -- the two are different
    contracts, and chapter 4 of `No_battery_comparison.ipynb` is where they meet,
    by ranking each separately and joining the top few on `Paket_ID`.
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
