"""The single-household MILP: the model, the environment it runs on, and the
trajectory it produces.

One household is one metering point with a PV roof, a battery and a price list.
Its physics are the same wherever it is solved -- on its own by
`solve_household`, or as one member of a joint community problem in
`Community_MILP` -- so the pieces that describe those physics live here and are
appended to whatever `pulp.LpProblem` the caller is building:

    add_household_physics     buy/sell/spill/charge/discharge/SOC and the
                              constraints that tie them together
    household_data_bounds     the same block re-pointed at new data, for a
                              caller that solves it again every interval
    add_excess_power_ratchet  the presezna-moc charge, per (block, month), as the
                              linear max(peak - A) the study solved until the
                              statutory form below replaced it on SI
    add_statutory_excess_power  the same charge as the invoice bills it,
                              sqrt(sum e^2) with DDV, as cutting planes
                              (`StatutoryExcess`, `tighten_statutory_excess`);
                              ported from Energy_Community's MILP_Household
    agreed_power_by_month     the per-month dogovorjena obracunska moc
    month_calendar            the local-time month bookkeeping all of the above index by
    floor_export_rates        the guard that keeps a negative import rate from
                              making the LP unbounded
    interval_rate_vectors     the per-interval import rate and export credit the
                              objective is built from

What is NOT here is pricing. `solve_household` prices through
`Pricing_Functions.calculate_interval_price`, the community model prices through
`si_obracun.samooskrba` with VAT applied explicitly and a sharing overlay on top;
those are genuinely different settlements, and only the physics are shared.

`build_household_env` is the one `HouseholdEnvironment` factory. Every study
passes its own price list, capacity and agreed-power rule; nothing about a
particular study is defaulted here.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pulp

from Environment import HouseholdEnvironment, converge_agreed_power
from Pricing_Functions import (
    InvoiceBuilder,
    PRIVZETO_REFERENCNO_LETO,
    calculate_interval_price,
    compute_prorated_fixed_charge_eur,
    moc_za_mesec,
    KONICNI_BLOKI,
    ST_KONIC,
)

# The SI tariff primitives live in the spaced folder.
_SI_DIR = Path(__file__).resolve().parent / "New pricing functions"
if str(_SI_DIR) not in sys.path:
    sys.path.append(str(_SI_DIR))

from si_cas import bloki_v_mesecu, je_visja_sezona, v_lokalni_cas  # noqa: E402
from si_obracun import Pravila  # noqa: E402
from si_tarife import DDV  # noqa: E402


# ---------------------------------------------------------------------------
# Battery, horizon and solver defaults
# ---------------------------------------------------------------------------
# Shared by every study so two of them cannot quietly size the same battery
# differently.
STEPS_PER_DAY = 96
SOC_FRACTION = 0.5
CHARGE_EFFICIENCY = 0.95
DISCHARGE_EFFICIENCY = 0.95
C_RATE = 0.5
INVERTER_MAX_KW = 11.0

# Operating envelope and wear price. Every default is "no envelope, no wear
# price", so a study that sets none of them solves exactly what it solved before.
#
# The two cycle figures below do NOT share a denominator, deliberately:
#
#   MAX_DAILY_CYCLES        counts against the USABLE window. Set it to 2 and the
#                           pack may be swept soc_min -> soc_max -> soc_min twice,
#                           after which it neither charges nor discharges again
#                           that day. It is an operating envelope.
#   Equivalent_Full_Cycles  counts against the NAMEPLATE pack, so reserving SOC
#                           headroom reads as a longer life -- the real physics of
#                           shallow cycling -- and the 6000-EFC rating keeps
#                           meaning what it has always meant.
MAX_DAILY_CYCLES = None      # usable-window cycles per LOCAL day; 2.0 is the
                             # recommended residential Li-ion value
SOC_MIN_FRAC = 0.0           # unusable reserve at the bottom of the pack
SOC_MAX_FRAC = 1.0           # unusable headroom at the top

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
    soc_min_frac=SOC_MIN_FRAC,
    soc_max_frac=SOC_MAX_FRAC,
    max_daily_cycles=MAX_DAILY_CYCLES,
    cycle_cost_eur_per_efc=None,
):
    """The `HouseholdEnvironment` every study solves on.

    `episode_length` defaults to `len(data) - 1`; pass `len(data)` to keep the
    closing interval, which a full-calendar-year solve wants.

    `agreed_power` is the dogovorjena-obracunska-moc rule as a dict of
    `connection_power_kw` / `min_agreed_power_kw` / `agreed_power_lag_months` /
    `agreed_power_bootstrap`. Left out, the environment's own defaults apply,
    which is what every caller relies on.

    `capacity_kwh` is the NAMEPLATE pack -- what the invoice is for. The SOC
    window derates it to the usable capacity the physics actually sees, and that
    is what goes into the environment. The power rating stays on the nameplate:
    reserving headroom in the cells does not shrink the inverter.
    """
    nominal_kwh = float(capacity_kwh)
    if not 0.0 <= float(soc_min_frac) < float(soc_max_frac) <= 1.0:
        raise ValueError(
            f"need 0 <= soc_min_frac < soc_max_frac <= 1, got "
            f"{soc_min_frac!r} / {soc_max_frac!r}"
        )
    usable_kwh = (float(soc_max_frac) - float(soc_min_frac)) * nominal_kwh
    step_kwh = step_energy_kwh(nominal_kwh, c_rate, inverter_max_kw, steps_per_day)
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
        battery_capacity_kwh=usable_kwh,
        nominal_capacity_kwh=nominal_kwh,
        soc_min_frac=float(soc_min_frac),
        soc_max_frac=float(soc_max_frac),
        max_daily_cycles=max_daily_cycles,
        cycle_cost_eur_per_efc=cycle_cost_eur_per_efc,
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


def day_calendar(dates):
    """Local-time day bookkeeping for one horizon.

    Returns `(day_key_t, days_sorted, day_idx_t)`, the daily twin of
    `month_calendar` and for the same reason: the Fluvius stamps carry a `Z`
    suffix but are Brussels-local, so the day has to be read off `v_lokalni_cas`
    rather than off the raw index.

    A fixed 96-step stride would be wrong besides. On a calendar year of Fluvius
    profiles the local days come out 96 intervals long except four: the two DST
    days (88 on the spring forward, 104 on the autumn back, the hour of the shift
    compounding with the hour the mislabelled stamps are already off by), and the
    two horizon edges, where the shift leaves a part-day at each end. A daily
    budget sliced on a fixed stride would let those borrow from their neighbours.
    """
    local_times = [v_lokalni_cas(ts) for ts in dates]
    day_key_t = [lt.date() for lt in local_times]
    days_sorted = sorted(set(day_key_t))
    # A dict, not `list.index`: a full year is 35 040 stamps over 366 days, and
    # the linear scan `month_calendar` can afford over 12 months is quadratic here.
    order = {k: i for i, k in enumerate(days_sorted)}
    day_idx_t = [order[k] for k in day_key_t]
    return day_key_t, days_sorted, day_idx_t


def steps_by_day(day_idx_t, n_days=None):
    """`day_idx_t` inverted: a list of the step indices belonging to each day."""
    n_days = (max(day_idx_t) + 1) if n_days is None else int(n_days)
    groups = [[] for _ in range(n_days)]
    for t, d in enumerate(day_idx_t):
        groups[d].append(t)
    return groups


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


def interval_rate_vectors(env, dates, smp_prices, pricing_options, interval_minutes):
    """Per-interval delivered import rate and export credit [EUR/kWh].

    Returns `(import_rates, export_rates, constant_costs)`, all EUR per kWh
    except `constant_costs`, which is the prorated fixed charge the package
    carries with no agreed power supplied.

    Priced at +-1 kWh without `dogovorjena_moc`/`prev_peak_kw`, so the result
    stays linear in the traded energy and the peak and fixed charges can enter
    separately. This is the price signal `solve_household` optimizes against;
    it is a function of the calendar, the market price and the package only, so
    a rule-based controller can read the same two vectors instead of rebuilding
    them and the two cannot end up seeing different prices.

    The export credit is NOT floored here -- `floor_export_rates` is the
    caller's, because only an optimizer needs the guard.
    """
    import_rates, export_rates, constant_costs = [], [], []
    for t in range(len(dates)):
        import_res = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=1.0,
            utc_date=dates[t],
            interval_minutes=interval_minutes,
            scheme=env.pricing_scheme,
            **pricing_options,
        )
        import_rates.append(import_res["variable_price_aud"])
        constant_costs.append(import_res["constant_price_aud"])

        export_res = calculate_interval_price(
            smp_market_price_kwh=smp_prices[t],
            total_consumed_kwh=-1.0,
            utc_date=dates[t],
            interval_minutes=interval_minutes,
            scheme=env.pricing_scheme,
            **pricing_options,
        )
        export_rates.append(-export_res["variable_price_aud"])
    return import_rates, export_rates, constant_costs


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
class HouseholdRows:
    """The rows of a block whose right-hand side is DATA, not structure.

    Everything else `add_household_physics` writes -- the SOC recursion, the
    inverter bound, the binaries -- depends on the battery alone and is the same
    at every step of a receding horizon. These are the rows that are not: they
    carry the opening and closing SOC and the interval's generation and
    consumption. Kept as the constraint objects themselves rather than as
    positions, so a caller that re-points a built block at new data
    (`household_data_bounds`) does not depend on the order the builder emits its
    rows in -- which is load-bearing for other reasons and must stay free to be.
    """

    soc_initial: Optional[object] = None    # soc[0] == opening SOC
    soc_final: Optional[object] = None      # soc[n] >= terminal floor
    balance: list = field(default_factory=list)     # one per interval, ==
    meter_sell: list = field(default_factory=list)  # one per interval, <=
    meter_buy: list = field(default_factory=list)   # one per interval, <=
    fuse: list = field(default_factory=list)        # one per interval, <= (if fused)


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
    capacity_kwh: float          # the usable window, what the physics sees
    nominal_capacity_kwh: float  # the pack on the invoice
    rows: Optional[HouseholdRows] = None    # the data-carrying rows, see above
    max_import_kwh: Optional[float] = None  # the fuse, per interval; None = none


def _clamped_soc(soc_kwh, capacity):
    """An opening or terminal SOC, held inside the usable window."""
    return min(max(0.0, float(soc_kwh)), capacity)


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
    day_idx_t=None,
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

    `day_idx_t` is the per-step local day index from `day_calendar`, and is what
    `env.max_daily_cycles` is enforced over. It is passed in rather than derived
    here, the way `month_idx_t` already is for the ratchet, so the physics
    builder stays free of calendar logic.

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

    # The rows written from DATA are kept by handle as they are added -- the
    # same constraint objects, in the same order, so the model is unchanged.
    # `household_data_bounds` is their other half and must say what they say.
    rows = HouseholdRows()

    if soc is not None:
        if initial_soc_kwh is None:
            initial_soc_kwh = capacity / 2.0
        rows.soc_initial = (soc[0] == _clamped_soc(initial_soc_kwh, capacity))
        prob += rows.soc_initial
        # Terminal floor, so the opening charge cannot be sold off as a saving.
        if final_soc_kwh is not None:
            rows.soc_final = (soc[n_steps] >= _clamped_soc(final_soc_kwh, capacity))
            prob += rows.soc_final

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
        balance = (
            gen[t] + buy[t] + discharge[t]
            == con[t] + sell[t] + charge[t] + spill[t]
        )
        prob += balance
        rows.balance.append(balance)
        if metering_bounds:
            # A meter nets import against export inside the interval. These two
            # hold at every physical operating point and bound the LP, which the
            # energy balance alone does not.
            meter_sell = sell[t] <= max(gen[t] - con[t], 0.0) + discharge[t]
            meter_buy = buy[t] <= max(con[t] - gen[t], 0.0) + charge[t] + spill[t]
            prob += meter_sell
            prob += meter_buy
            rows.meter_sell.append(meter_sell)
            rows.meter_buy.append(meter_buy)

    if exclusivity == "binary" and soc is not None:
        flags = [
            pulp.LpVariable(names.charging_flag.format(t=t), cat="Binary")
            for t in range(n_steps)
        ]
        for t in range(n_steps):
            prob += charge[t] <= max_charge_ac * flags[t]
            prob += discharge[t] <= max_discharge_ac * (1 - flags[t])

    # --- daily cycle cap -----------------------------------------------------
    # Appended here, after every per-interval row, and never inside the loop
    # above: the row order decides which of several equally optimal vertices CBC
    # returns (see the note in this function's docstring), so new rows go at the
    # end where they cannot reshuffle the existing ones.
    #
    # `capacity` is the USABLE window. A cap of 2 therefore buys two sweeps of
    # soc_min -> soc_max -> soc_min, after which the pack idles for the rest of
    # the local day. The factor 2 is because one sweep moves `capacity` kWh
    # through the store in each direction.
    #
    # This is store throughput, not a count of round trips -- many shallow cycles
    # summing to the same energy are equally allowed. That is the standard
    # formulation and the only one that stays linear; a literal "at most two
    # round trips" would need per-day binaries on top of the 35 040 already here.
    max_daily_cycles = getattr(env, "max_daily_cycles", None)
    if max_daily_cycles is not None and soc is not None and capacity > 0:
        if day_idx_t is None:
            raise ValueError(
                "env.max_daily_cycles is set but day_idx_t was not passed; "
                "build it with MILP_Household.day_calendar(dates)"
            )
        if len(day_idx_t) != n_steps:
            raise ValueError(
                f"day_idx_t covers {len(day_idx_t)} steps, the block has {n_steps}"
            )
        budget = float(max_daily_cycles) * 2.0 * capacity
        for steps in steps_by_day(day_idx_t):
            if not steps:
                continue
            prob += pulp.lpSum(
                charge[t] * env.charge_efficiency
                + discharge[t] / env.discharge_efficiency
                for t in steps
            ) <= budget

    # --- the fuse ------------------------------------------------------------
    # The connection carries at most `env.max_import_kwh` an interval, and the
    # battery may not push the draw past it: `buy <= max(fuse, net load)`. On
    # the truth that is `buy <= fuse` -- the connection is sized to carry the
    # house (`hems_study.si_connection`). A forecast may predict a net load the
    # house never drew; the max() keeps that window solvable, and it is exactly
    # the bound every rule runs under (`Basic_Functions.fuse_charge_headroom`):
    # charging stops at the fuse, the house is never cut. Last, for the row-order
    # reason above, and data-carrying, so `household_data_bounds` re-points it.
    max_import_kwh = getattr(env, "max_import_kwh", None)
    if max_import_kwh is not None:
        for t in range(n_steps):
            fuse = buy[t] <= max(float(max_import_kwh), con[t] - gen[t])
            prob += fuse
            rows.fuse.append(fuse)

    return HouseholdBlock(
        buy=buy, sell=sell, spill=spill, charge=charge, discharge=discharge, soc=soc,
        max_charge_ac=max_charge_ac, max_discharge_ac=max_discharge_ac,
        capacity_kwh=capacity,
        nominal_capacity_kwh=float(getattr(env, "nominal_capacity_kwh", capacity)),
        rows=rows,
        max_import_kwh=None if max_import_kwh is None else float(max_import_kwh),
    )


# ---------------------------------------------------------------------------
# The data half of the household block
# ---------------------------------------------------------------------------
# A receding-horizon controller solves the SAME block thousands of times: the
# battery does not change between two steps, only the opening SOC and the
# generation and consumption the window covers. Rebuilding the block for each
# of them is where such a controller's time went -- measured in the HEMS study,
# 4.0 of the 5.6 ms a 48-step solve cost was pulp assembling and translating a
# model whose matrix it had assembled identically the step before, against
# 1.2 ms of actual optimisation.
#
# These two functions are what lets a caller build the block once and re-point
# it instead. They restate, as numbers, exactly what `add_household_physics`
# writes from its `gen`/`con`/SOC arguments, and that is a second statement of
# the same physics -- so it is kept HERE, beside the first, and a caller that
# relies on it is expected to hold the two together by checking a re-pointed
# model against a freshly built one (`UpstreamMILPScheduler` does, on a
# schedule, and `test_hems_study` does on every solve of a slice). Change a
# data-carrying row in the builder and this must change with it.
#
# Nothing here knows about a solver: the bounds come back as plain arrays in the
# convention pulp's own `getLb`/`getUb` use, row activity on the variables only.


def household_data_rows(blk):
    """The block's data-carrying constraints, in `household_data_bounds` order."""
    rows = blk.rows
    if rows is None:
        raise ValueError("this block predates HouseholdRows; rebuild it with "
                         "add_household_physics")
    ordered = [c for c in (rows.soc_initial, rows.soc_final) if c is not None]
    return (ordered + list(rows.balance) + list(rows.meter_sell) + list(rows.meter_buy)
            + list(rows.fuse))


def household_data_bounds(blk, *, gen, con, initial_soc_kwh=None, final_soc_kwh=None):
    """What a built block's data-carrying rows and columns read for NEW data.

    Returns `(row_lower, row_upper, spill_upper)`: the bounds of the rows
    `household_data_rows(blk)` lists, in that order, and the upper bound of each
    interval's curtailment variable. Arguments mean what they mean to
    `add_household_physics`, kWh per interval, and the result is the model that
    function would have built for them -- provided the block was built with the
    same options, which is why a terminal floor cannot be added to or dropped
    from a block here: that is a row, not a number.
    """
    rows = blk.rows
    if rows is None:
        raise ValueError("this block predates HouseholdRows; rebuild it with "
                         "add_household_physics")
    n_steps = len(blk.buy)
    gen = np.asarray(gen, dtype=float)
    con = np.asarray(con, dtype=float)
    if len(gen) != n_steps or len(con) != n_steps:
        raise ValueError(
            f"the block covers {n_steps} steps, gen/con cover {len(gen)}/{len(con)}")
    if (final_soc_kwh is None) != (rows.soc_final is None):
        raise ValueError(
            "a terminal SOC floor is a row of the block: it was built "
            f"{'without' if rows.soc_final is None else 'with'} one and cannot "
            "be re-pointed the other way")

    capacity = blk.capacity_kwh
    lower, upper = [], []
    if rows.soc_initial is not None:
        if initial_soc_kwh is None:
            initial_soc_kwh = capacity / 2.0
        opening = _clamped_soc(initial_soc_kwh, capacity)
        lower.append([opening])
        upper.append([opening])
    if rows.soc_final is not None:
        lower.append([_clamped_soc(final_soc_kwh, capacity)])
        upper.append([np.inf])

    # gen + buy + discharge == con + sell + charge + spill, variables on the left.
    net_load = con - gen
    lower.append(net_load)
    upper.append(net_load)
    if rows.meter_sell:
        unbounded = np.full(n_steps, -np.inf)
        # sell <= max(gen - con, 0) + discharge
        lower.append(unbounded)
        upper.append(np.maximum(gen - con, 0.0))
        # buy <= max(con - gen, 0) + charge + spill
        lower.append(unbounded)
        upper.append(np.maximum(con - gen, 0.0))
    if rows.fuse:
        # buy <= max(fuse, con - gen)
        lower.append(np.full(n_steps, -np.inf))
        upper.append(np.maximum(blk.max_import_kwh, con - gen))

    # Curtailment is bounded by the interval's own generation.
    return np.concatenate(lower), np.concatenate(upper), gen


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
    agreed_vars=None,
    peak_name="P_peak_b{b}_m{m}",
    excess_name="Excess_b{b}_m{m}",
):
    """The presezna-moc charge: one peak variable per (block, month) occurring.

    The peak is ratcheted within a reset window and floored at the seed peak --
    but only in the window the horizon starts in; a window opening inside it
    resets to zero. It is measured on the METERED import `buy`, which no billing
    overlay can move.

    `agreed_vars` is `{(block, month): kW}` where the kW may be an LpVariable --
    the endogenous contract, where the line this month is billed against is set
    by the peaks the solve itself achieved last month. Left None, the constant
    `agreed_by_month` is billed against, which is the exogenous rule.

    Returns `(peak_vars, excess_vars, objective_terms)`.
    """
    def agreed(b, m):
        if agreed_vars is None:
            return agreed_by_month[m].get(b, 0.0)
        return agreed_vars[(b, m)]

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

        prob += excess_var[(b, m)] >= peak_var[(b, m)] - agreed(b, m)

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
            # A max() over a variable would be nonlinear. It never is one: the
            # seed window is the horizon's first, and the first `lag` months of
            # an endogenous contract are the constant bootstrap by construction.
            agreed_seed = agreed(b, m)
            if isinstance(agreed_seed, pulp.LpVariable):
                raise ValueError(
                    "the seed window's agreed power must be a constant; the "
                    "contract of the horizon's leading month is not a decision."
                )
            prev_term = max(0.0, seed_peak_kw.get(b, 0.0) - float(agreed_seed))
        else:
            prev_term = 0.0          # a fresh window pays its own excess in full

        objective_terms.append((excess_var[(b, m)] - prev_term) * rate_bm * faktor)
        prev_excess_by_block[b] = excess_var[(b, m)]
        prev_window_by_block[b] = w

    return peak_var, excess_var, objective_terms


# ---------------------------------------------------------------------------
# The excess-power charge as the statute bills it
# ---------------------------------------------------------------------------
class StatutoryExcess:
    """The statutory excess-power charge, as an LP that cutting planes tighten.

    The invoice bills, per block b and calendar month m,

        1.05 x block rate x sqrt(sum over the month's intervals of e_t^2),
        e_t = max(0, draw_t - A[b, m]),                         plus DDV

    (`si_obracun.MesecniObracun`). The ratchet `add_excess_power_ratchet` puts in
    an LP instead bills `max_t e_t`, without DDV -- the largest term of the same
    norm. The two agree when a month exceeds its contract once and part ways
    with the NUMBER of exceedances: 86 equal exceedances cost sqrt(86) ~ 9.3
    times what the ratchet charges for them. A model priced on the ratchet
    therefore signs too little wherever crossing the line often is cheap one
    interval at a time, which is exactly the levy-carrying block.

    The norm is convex, so its epigraph is the intersection of the half-spaces

        s >= g . e    for every g >= 0 with ||g||_2 = 1

    and any finite set of them is a valid relaxation from below (Kelley's
    cutting-plane method). Three families are put in:

      * every interval on its own, g = unit vector: s >= draw_t - A. This is
        the ratchet's constraint set, so round 0 is the ratchet with DDV;
      * a few tangents read off a reference draw (the no-battery profile),
        where the exceedance pattern of a lower contract is already visible;
      * after each solve, the tangent at the point the solve returned,
        g = e*/||e*||, wherever the model's s falls short of the true norm
        (`add_cuts`). `tighten_statutory_excess` repeats until the shortfall is below a
        tolerance in EUR.

    A tangent weights particular intervals, and a battery can move an
    exceedance off them: measured on a 10 kWh household, forty rounds of
    tangents left February's block 1 under-priced by a third, a new set of
    intervals each round. So a (b, m) that a tangent has not closed after
    `TOPJ_AFTER` rounds also gets the cuts no permutation escapes,

        sqrt(j) * s >= sum of the j largest e_t,     j in TOPJ_LEVELS,

    each exact as an LP through the same k-largest epigraph the contract uses
    (`add_topk_mean_lower_bound`), with the threshold held at or above zero so
    it is the sum of the j largest POSITIVE parts. Equal exceedances over j
    intervals are priced exactly. A battery that runs out of energy leaves a
    DECAYING tail of exceedances instead, on which the best top-j bound is ~13 %
    short (a linear decay: sqrt(3)/2), so the same epigraphs then carry the
    ordered-weighted tangent

        s >= sum_i g_i e_[i],   g decreasing, ||g|| <= 1,   e_[i] the i-th largest,

    valid by Cauchy-Schwarz whatever the order, and linear in the top-j sums
    (sum_i g_i e_[i] = sum_j (g_j - g_j+1) S_j for a g stepping at the levels).
    Read off the solution's own sorted exceedances it is the tangent there up to
    the steps, and no permutation of the intervals escapes it.

    `e_t` variables exist only for the intervals some cut actually weights, so
    the model grows with the exceedances it has to price and not with the year.

    `carried` opens the norm mid-month. A receding controller plans one day of a
    month whose earlier exceedances are already on the meter; the charge it can
    still move is

        rate x sqrt(c0^2 + sum over the window of e_t^2),   c0^2 = metered sum,

    the same cone with one constant coordinate. Every cut above stays valid
    (each weights a sub-vector of it), the tangents take c0 into their norm, and
    `s >= c0` joins the interval cuts. What changes is the marginal price, and
    that is the point: d/de sqrt(c0^2 + e^2) = e / sqrt(c0^2 + e^2), so once a
    month has been crossed by a lot a SMALL further crossing is nearly free and
    a large one is not -- neither the ratchet's "free below the running peak"
    nor its "full rate above it". The rest of the month beyond the window is
    assumed to add nothing, the same myopia the ratchet's seeded peak has.
    """

    TOPJ_LEVELS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)
    TOPJ_AFTER = 2
    # A (b, m) of at most this many intervals gets EVERY level 1..n, which makes
    # the ordered-weighted tangent the exact tangent at the solution and still
    # blind to which intervals carry the exceedances. Only a receding window's
    # day-long groups are this small -- a whole month's run to a thousand -- and
    # on them the stepped levels are what stalls the loop: measured on three
    # captured windows (Fluvius #3, 10 kWh) the stepped family left the model's
    # norm fixed while the battery moved the exceedance between intervals, and
    # 60 rounds did not close 1e-3 EUR; every level closed it in 9-15.
    TOPJ_ALL_LEVELS_UP_TO = 96

    def __init__(self, prob, buy, *, groups, agreed, coefficient, hours, name, carried=None):
        self.prob = prob
        self.buy = buy
        self.groups = groups                 # (b, m) -> [t, ...]
        self.agreed = agreed                 # (b, m) -> LpVariable or float
        self.coefficient = coefficient       # (b, m) -> EUR per RMS kW, DDV in
        self.hours = float(hours)
        self.name = name
        self.s = {key: pulp.LpVariable(name.format(b=key[0], m=key[1]), lowBound=0)
                  for key in groups}
        self.e = {}
        self.n_cuts = 0
        self.rounds_open = {}                # (b, m) -> tangent rounds it stayed open
        self.topj = {}                       # (b, m) -> [(j, S_j expression), ...]
        self.log = []                        # every cut, as data (`replay`)
        # (b, m) -> sqrt of the squared exceedances the month already metered.
        self.c0 = {key: math.sqrt(max(0.0, float((carried or {}).get(key, 0.0))))
                   for key in groups}

    def add_topj_family(self, key):
        """sqrt(j) s >= sum of the j largest e_t, for every level j up to all."""
        if key in self.topj:
            return
        b, m = key
        ts = self.groups[key]
        if len(ts) <= self.TOPJ_ALL_LEVELS_UP_TO:
            levels = list(range(1, len(ts) + 1))
        else:
            levels = [j for j in self.TOPJ_LEVELS if j < len(ts)] + [len(ts)]
        family = []
        for j in levels:
            tau = pulp.LpVariable(f"tauj_b{b}_m{m}_j{j}", lowBound=0)
            slack = []
            for t in ts:
                u = pulp.LpVariable(f"uj_b{b}_m{m}_j{j}_t{t}", lowBound=0)
                self.prob += u >= self.buy[t] * (1.0 / self.hours) - self.agreed[key] - tau
                slack.append(u)
            s_j = j * tau + pulp.lpSum(slack)
            self.prob += math.sqrt(j) * self.s[key] >= s_j
            family.append((j, s_j))
            self.n_cuts += 1
        self.topj[key] = family
        self.log.append(("topj", key))

    def add_ordered_tangent(self, key, exceedances):
        """The ordered-weighted tangent at a sorted exceedance vector.

        g_i = e_[i] / ||e||, stepped DOWN to its value at the end of each level
        so the stepped g stays decreasing with norm <= 1 -- a valid cut that is
        the tangent up to the steps. Needs `add_topj_family(key)` first.
        """
        e = np.sort(np.asarray([v for v in exceedances if v > 0.0], dtype=float))[::-1]
        c0 = self.c0.get(key, 0.0)
        norm = float(np.sqrt(c0 * c0 + np.sum(e * e)))
        if norm <= 0.0 or key not in self.topj:
            return False
        g = e / norm
        family = self.topj[key]
        steps = [float(g[j - 1]) if j - 1 < len(g) else 0.0 for j, _ in family]
        terms = []
        for i, (j, s_j) in enumerate(family):
            w = steps[i] - (steps[i + 1] if i + 1 < len(steps) else 0.0)
            if w > 0.0:
                terms.append(w * s_j)
        if not terms:
            return False
        if c0 > 0.0:
            self.prob += self.s[key] >= c0 * c0 / norm + pulp.lpSum(terms)
        else:
            self.prob += self.s[key] >= pulp.lpSum(terms)
        self.n_cuts += 1
        self.log.append(("ordered", key, tuple(float(v) for v in e)))
        return True

    def _e(self, t, key):
        if t not in self.e:
            v = pulp.LpVariable(f"e_exc_{t}", lowBound=0)
            self.prob += v >= self.buy[t] * (1.0 / self.hours) - self.agreed[key]
            self.e[t] = v
        return self.e[t]

    def add_interval_cuts(self):
        """s >= draw_t - A for every interval: the ratchet's constraint set."""
        for key, ts in self.groups.items():
            for t in ts:
                self.prob += self.s[key] >= self.buy[t] * (1.0 / self.hours) - self.agreed[key]
            if self.c0[key] > 0.0:
                self.prob += self.s[key] >= self.c0[key]

    def add_tangent(self, key, weights):
        """s >= sum_t g_t e_t with g = weights / ||weights||."""
        w = {t: float(v) for t, v in weights.items() if v > 0.0}
        c0 = self.c0.get(key, 0.0)
        norm = math.sqrt(c0 * c0 + sum(v * v for v in w.values()))
        if norm <= 0.0 or not w or (len(w) < 2 and c0 <= 0.0):
            return False                     # a single term is an interval cut
        if c0 > 0.0:
            self.prob += self.s[key] >= c0 * c0 / norm + pulp.lpSum(
                (v / norm) * self._e(t, key) for t, v in w.items())
        else:
            self.prob += self.s[key] >= pulp.lpSum((v / norm) * self._e(t, key)
                                                   for t, v in w.items())
        self.n_cuts += 1
        self.log.append(("tangent", key, tuple(sorted(w.items()))))
        return True

    def replay(self, log):
        """Put in the cuts another model of the same horizon generated.

        Every cut is valid whatever the dispatch, so cuts found on the LP form
        of a household (`exclusivity="inverter"`) carry over to its binary form
        as they are, and the binary solve starts where the LP's loop ended.
        """
        for entry in log or ():
            kind, key = entry[0], tuple(entry[1])
            if key not in self.groups:
                continue
            if kind == "tangent":
                self.add_tangent(key, dict(entry[2]))
            elif kind == "topj":
                self.add_topj_family(key)
            elif kind == "ordered":
                self.add_topj_family(key)
                self.add_ordered_tangent(key, entry[2])

    def add_reference_tangents(self, power_kw, depths=(2, 4, 8, 16, 32, 64, 128)):
        """Tangents at the contract levels a reference draw puts in reach.

        For each (b, m) and each depth j, the contract A = the j-th largest
        reference draw, and the tangent of the norm there. On the reference
        draw itself these are exact at those levels; on any other they are
        still valid cuts.
        """
        power_kw = np.asarray(power_kw, dtype=float)
        for key, ts in self.groups.items():
            ts = np.asarray(ts)
            x = power_kw[ts]
            order = np.argsort(-x)
            for j in depths:
                if j >= len(order):
                    break
                level = float(x[order[j]])
                above = order[:j]
                self.add_tangent(key, {int(ts[i]): float(x[i]) - level for i in above})

    def objective_terms(self):
        return [self.coefficient[key] * self.s[key] for key in self.groups]

    def evaluate(self):
        """{(b, m): (true norm at the solution, the model's s)}."""
        out = {}
        for key, ts in self.groups.items():
            a = _value(self.agreed[key])
            sq = 0.0
            for t in ts:
                d = (self.buy[t].varValue or 0.0) / self.hours - a
                if d > 0.0:
                    sq += d * d
            c0 = self.c0.get(key, 0.0)
            out[key] = (math.sqrt(c0 * c0 + sq), float(self.s[key].varValue or 0.0))
        return out

    def shortfall_eur(self, evaluation=None):
        """What the model under-prices the solution's excess by, EUR."""
        evaluation = self.evaluate() if evaluation is None else evaluation
        return sum(self.coefficient[k] * max(0.0, true - s)
                   for k, (true, s) in evaluation.items())

    def excess_eur(self, evaluation=None):
        """The statutory excess charge of the solution, EUR (DDV in)."""
        evaluation = self.evaluate() if evaluation is None else evaluation
        return sum(self.coefficient[k] * true for k, (true, _) in evaluation.items())

    def add_cuts(self, evaluation=None, rel_tol=1e-6):
        """The tangent at the current solution wherever s falls short. Returns
        how many cuts were added, and sets the new variables' values so the
        previous solution, with s raised to the true norm, is a feasible warm
        start."""
        evaluation = self.evaluate() if evaluation is None else evaluation
        added = 0
        for key, (true, s) in evaluation.items():
            if true <= s * (1.0 + rel_tol) + 1e-9:
                continue
            a = _value(self.agreed[key])
            weights = {}
            for t in self.groups[key]:
                d = (self.buy[t].varValue or 0.0) / self.hours - a
                if d > 0.0:
                    weights[t] = d
            if len(weights) >= 2 or (weights and self.c0.get(key, 0.0) > 0.0):
                self.add_tangent(key, weights)
                added += 1
            self.rounds_open[key] = self.rounds_open.get(key, 0) + 1
            if self.rounds_open[key] >= self.TOPJ_AFTER:
                self.add_topj_family(key)
                if self.add_ordered_tangent(key, list(weights.values())):
                    added += 1
            # Warm start: s at the true norm, every e at its exceedance. The
            # top-j slacks are left unset; CBC completes a partial start.
            self.s[key].varValue = true
            for t in self.groups[key]:
                if t in self.e:
                    d = (self.buy[t].varValue or 0.0) / self.hours - a
                    self.e[t].varValue = max(0.0, d)
        return added


def add_statutory_excess_power(prob, env, buy, *, blocks, month_idx_t, months_sorted,
                               agreed_by_month, agreed_vars, hours, pravila, n_steps,
                               reference_power_kw=None, name="S_exc_b{b}_m{m}",
                               carried_sumsq=None, apply_ddv=True):
    """The statutory excess charge on `buy`, as a `StatutoryExcess`.

    One `s` per (block, month) that occurs, priced at block rate x
    `faktor_presezne_moci` x (1 + DDV) -- the invoice's own figure. Months are
    calendar months; the charge does not telescope across them, so the horizon
    has to open at a month boundary with nothing carried in (checked by the
    caller). `reference_power_kw` (a draw over the same horizon, kW) seeds the
    tangent family; None leaves round 0 at the interval cuts alone.
    `carried_sumsq` is `{(b, m): kW^2}`, the squared exceedances a month has
    already metered before the horizon opens (`StatutoryExcess`, `carried`).

    DEPARTURE from Energy_Community: the rate is `statutory_excess_rate`, the one
    the settlement bills with, and `apply_ddv` follows the environment's pricing
    options instead of being hard-wired on. Every study here prices with DDV, so
    the two agree to the bit; the argument exists so they cannot stop agreeing.
    """
    def agreed(b, m):
        if agreed_vars is None:
            return float(agreed_by_month[m].get(b, 0.0))
        return agreed_vars[(b, m)]

    groups = {}
    for t in range(n_steps):
        groups.setdefault((int(blocks[t]), month_idx_t[t]), []).append(t)
    coefficient = {}
    for (b, m) in groups:
        y, mo = months_sorted[m]
        coefficient[(b, m)] = statutory_excess_rate(pravila, y, mo, b, apply_ddv=apply_ddv)
    model = StatutoryExcess(
        prob, buy, groups=groups, agreed={k: agreed(*k) for k in groups},
        coefficient=coefficient, hours=hours, name=name, carried=carried_sumsq,
    )
    model.add_interval_cuts()
    if reference_power_kw is not None:
        model.add_reference_tangents(reference_power_kw)
    return model


def statutory_excess_rate(pravila, year, month, block, apply_ddv=True):
    """EUR per kW of RMS exceedance, block `block`, calendar month (year, month).

    The invoice's own figure (`si_obracun.MesecniObracun._fiksne`): the block's
    network power rate in that month's season times `faktor_presezne_moci`, and
    DDV on top as the invoice applies it to every taxable line. The ONE place
    that figure is formed: `add_statutory_excess_power` prices the LP with it
    and `Rule_Based_Control.price_interval_statutory` bills the executed meter
    with it, so a solve cannot be optimised against one rate and billed at
    another.
    """
    vs = je_visja_sezona(date(int(year), int(month), 1))
    rate = pravila.omreznina.postavka_moc(int(block), vs) * pravila.omreznina.faktor_presezne_moci
    return rate * ((1.0 + DDV) if apply_ddv else 1.0)


def tighten_statutory_excess(prob, statutory, solve, *, tol_eur, max_rounds,
                             gap_rel=0.0, verbose=False):
    """Cut the statutory model until it prices its own solution, or gives up.

    `solve()` re-solves `prob` and returns its pulp status string. The loop is
    the one `Energy_Community.MILP_Household.solve_household` runs inline,
    lifted out so the whole-period solve and every receding window run the same
    one: evaluate the true norm at the returned point, stop once the model
    under-prices it by at most `tol_eur` -- or by `gap_rel` times the objective,
    for a MIP returned within a relative gap, which moves its point by that much
    between re-solves -- otherwise add the tangents there and solve again.

    Returns `(status, rounds, shortfall_eur)`. A shortfall above the tolerance
    after `max_rounds` is NOT an error: the dispatch is feasible and the bill is
    computed from the meter, never from `s`. It is the caller's to count and
    report (`Statutory_Unclosed`).
    """
    status = pulp.LpStatus[prob.status]
    rounds, shortfall = 0, 0.0
    while status == "Optimal":
        evaluation = statutory.evaluate()
        shortfall = statutory.shortfall_eur(evaluation)
        tol = max(float(tol_eur),
                  float(gap_rel) * abs(float(pulp.value(prob.objective) or 0.0)))
        if shortfall <= tol or rounds >= int(max_rounds):
            break
        if verbose:
            worst = sorted(evaluation.items(),
                           key=lambda kv: -statutory.coefficient[kv[0]] * (kv[1][0] - kv[1][1]))[:3]
            print(f"  round {rounds}: shortfall {shortfall:.4f} EUR; "
                  + ", ".join(f"b{k[0]}m{k[1]} {t:.3f}/{s:.3f}" for k, (t, s) in worst))
        if not statutory.add_cuts(evaluation):
            break
        rounds += 1
        status = solve()
    return status, rounds, shortfall


# ---------------------------------------------------------------------------
# The contract as a decision, not an input
# ---------------------------------------------------------------------------
_AGREED_BLOCKS = (1, 2, 3, 4, 5)


def fixed_charge_coefficients(dates, month_idx_t, months_sorted, interval_minutes,
                              scheme, pricing_options):
    """`({(block, month): EUR per agreed kW}, EUR constant)` over the horizon.

    The prorated fixed charge is linear in the agreed power -- a per-block
    network power rate plus the OVE/SPTE levy on the reference block -- and is
    the same number for every interval of a month. So it is read off
    `compute_prorated_fixed_charge_eur` by differencing a unit contract against
    a zero one, once per month, rather than reimplemented here: whatever the
    tariff act does to the rates, this follows it.
    """
    first_t, counts = {}, {}
    for t, m in enumerate(month_idx_t):
        first_t.setdefault(m, t)
        counts[m] = counts.get(m, 0) + 1

    def fixed(t, moc):
        return compute_prorated_fixed_charge_eur(
            dates[t], interval_minutes, scheme=scheme, dogovorjena_moc=moc,
            **pricing_options,
        )

    coefficients, constant = {}, 0.0
    for m, t in first_t.items():
        zero = fixed(t, {b: 0.0 for b in _AGREED_BLOCKS})
        constant += zero * counts[m]
        for b in _AGREED_BLOCKS:
            unit = fixed(t, {bb: (1.0 if bb == b else 0.0) for bb in _AGREED_BLOCKS})
            coefficients[(b, m)] = (unit - zero) * counts[m]
    return coefficients, constant


def add_endogenous_agreed_power(
    prob, env, *, buy, hours, months_sorted, month_idx_t, blocks, dates,
    interval_minutes, pricing_options, agreed_by_month, n_steps,
    name="A_agreed_b{b}_m{m}", tau_name="tau_b{b}_m{m}",
    slack_name="sk_b{b}_m{m}_i{i}", contract="rule",
):
    """Make the dogovorjena moc a variable of the solve.

    `contract="rule"` ties it to the solve's OWN peaks, the operator's top-k
    statistic read with a lag (`_add_endogenous_agreed_power` has the
    construction). `contract="free"` is the other reading of the Akt: the household FILES its
    agreed power (12. clen (3), by the 20th for the next month) and may file any
    figure between the regulatory floor and its connection power. The
    contract is then a decision in its own right, one variable per billed
    (block, month) -- every month, the leading ones included, since nothing is
    read from a predecessor -- bounded by `env.min_agreed_power_kw` and
    `env.connection_power_kw` and monotone across blocks, with no tie to the
    peaks. Every contract a rule-following controller can be on lies inside
    that set, so the solve is a lower bound on all of them, which "rule" (a
    one-sided `A >= top-k`) is not once the excess charge, as well as the
    standing charge, falls as `A` rises: under the statutory norm the solve then
    sits above the rule while the bill applies it.
"""
    if contract not in ("rule", "free"):
        raise ValueError(f"contract must be 'rule' or 'free', got {contract!r}")
    return _add_endogenous_agreed_power(
        prob, env, buy=buy, hours=hours, months_sorted=months_sorted,
        month_idx_t=month_idx_t, blocks=blocks, dates=dates,
        interval_minutes=interval_minutes, pricing_options=pricing_options,
        agreed_by_month=agreed_by_month, n_steps=n_steps, name=name,
        tau_name=tau_name, slack_name=slack_name, contract=contract)


def _add_endogenous_agreed_power(
    prob, env, *, buy, hours, months_sorted, month_idx_t, blocks, dates,
    interval_minutes, pricing_options, agreed_by_month, n_steps,
    name, tau_name, slack_name, contract,
):
    """`add_endogenous_agreed_power`; the rule form is documented below.

    The contract is not exogenous. A household that flattens its profile
    re-agrees its billing power down the following month, and is then billed --
    both the network power charge and the excess charge -- against the line it
    created. Priced as a constant, that feedback is invisible and the solver has
    no reason to shave anything it is not already paying excess power on.

    So the contract enters the LP as one variable per (block, month), tied to the
    draw by exactly the rule `si_moc.mesecni_razpored` applies to a measured
    profile. The operator's statistic is the MEAN OF THE `k` HIGHEST 15-minute
    peaks per block [Akt, 12. clen], pooled over `n_months_window` months back
    from the lag, so the binding constraint is

        A[b, m] >= mean of the k largest of {import kW in block b over the
                                             months m's contract is read from}
        A[b, m] >= A[b - 1, m]             the Akt's monotonicity across blocks
        A[b, m] >= min_agreed_power_kw     the regulatory floor

    Only blocks 1-4 are read; block 5 has no peak source and comes out of the
    monotonicity rule, which is why a real bill shows P5 == P4.

    The mean of the k largest is convex, so `A >= it` is an exact LP:

        sum of k largest of {x_i} == min over tau of [k*tau + sum_i max(0, x_i - tau)]

    which enters as one free `tau` per (block, month) and one non-negative slack
    per pooled value. `A` carries a positive objective coefficient, so
    minimizing performs that inner minimization too and the bound is tight -- the
    solve is still one solve and still the exact optimum, which matters because
    every rule-based controller is scored against it as a lower bound.
    See Ogryczak & Tamir, "Minimizing the sum of the k largest functions in
    linear time", Inf. Proc. Letters 85(3):117-122 (2003); the same construction
    is better known as Rockafellar & Uryasev's CVaR linearization (2000).

    The leading `lag` months have no in-horizon predecessor and keep the constant
    bootstrap contract. Returns `(agreed_vars, objective_terms)`; `agreed_vars`
    maps every (block, month) to a variable or a float and goes on to
    `add_excess_power_ratchet`.
    """
    free = contract == "free"
    lag = 0 if free else getattr(env, "agreed_power_lag_months", None)
    if not free and not lag:
        raise ValueError("an endogenous contract needs a rolling agreed power "
                         "(agreed_power_lag_months), not a signed constant.")
    cap_kw = None if env.connection_power_kw is None else float(env.connection_power_kw)
    if free and cap_kw is not None and float(env.min_agreed_power_kw) > cap_kw:
        raise ValueError(f"floor {env.min_agreed_power_kw} kW above the connection "
                         f"power {cap_kw} kW: no contract can be filed")
    if not free and env.connection_power_kw is not None:
        # uskladi_bloke clips at the connection power; as a constraint that is
        # an upper bound colliding with a lower one, i.e. an infeasible model
        # for any household that exceeds its own connection.
        raise ValueError("an endogenous contract with a connection-power ceiling "
                         "is not modelled; converge it from the outside instead.")

    n_months = len(months_sorted)
    floor_kw = float(env.min_agreed_power_kw)
    bootstrap = {int(b): [float(v) for v in vals]
                 for b, vals in (env.agreed_power_bootstrap_kw or {}).items()}
    carry = bool(env.agreed_power_carry_missing_blocks)
    k_peaks = int(getattr(env, "agreed_power_n_peaks", ST_KONIC))
    window = int(getattr(env, "agreed_power_n_months_window", 1))

    # The intervals of each (block, month), which are the pool the contract is
    # read from. A block absent from a month has no entry, and the carry rule is
    # what fills the gap.
    group_ts = {}
    for t in range(n_steps):
        group_ts.setdefault((int(blocks[t]), month_idx_t[t]), []).append(t)
    months_with_block = {b: sorted(m for (bb, m) in group_ts if bb == b)
                         for b in _AGREED_BLOCKS}

    # A month is billed `agreed kW x block rate` only for the blocks that occur
    # in it -- blocks 1 and 5 are seasonal, so every month is missing one. A
    # block outside that set carries no charge and no excess term, so its
    # contract variable would be free: unpriced from above, unbounded by
    # anything that costs money, and able to drag the block above it up through
    # the monotonicity rule. It is not part of the month's contract, so it is
    # not a decision either.
    pravila_moc = Pravila.za_leto(int(pricing_options.get(
        "pricing_reference_year", PRIVZETO_REFERENCNO_LETO)))
    billed = {m: set(bloki_v_mesecu(y, mo, pravila_moc.razpored))
              for m, (y, mo) in enumerate(months_sorted)}

    agreed_vars, variable_months = {}, []
    for m in range(n_months):
        if m < lag:
            for b in _AGREED_BLOCKS:
                agreed_vars[(b, m)] = float(agreed_by_month[m].get(b, 0.0))
            continue
        variable_months.append(m)
        for b in _AGREED_BLOCKS:
            agreed_vars[(b, m)] = (
                pulp.LpVariable(name.format(b=b, m=m), lowBound=floor_kw,
                                upBound=cap_kw if free else None)
                if b in billed[m] else 0.0
            )

    # Monotonicity: block 1 is the most expensive and the agreed power may not
    # fall as the block index rises. This is also the ONLY thing that sets
    # block 5, exactly as `uskladi_bloke` does it. The chain runs over the
    # blocks the month is billed for; a block it is not billed for is not in
    # its contract and cannot push the others up.
    for m in variable_months:
        chain = [b for b in _AGREED_BLOCKS if b in billed[m]]
        for lower, upper in zip(chain, chain[1:]):
            prob += agreed_vars[(upper, m)] >= agreed_vars[(lower, m)]

    for m in ([] if free else variable_months):
        newest = m - lag                    # newest month the contract may read
        oldest = newest - window + 1        # oldest month in the window
        for b in KONICNI_BLOKI:
            if b not in billed[m]:
                continue
            pool_ts, pool_const = [], []
            for j in range(max(oldest, 0), newest + 1):
                pool_ts.extend(group_ts.get((b, j), ()))
            if oldest < 0:
                # The window reaches past the start of the horizon; the meter
                # history the household walked in with fills the rest.
                pool_const.extend(bootstrap.get(b, ()))
            if not pool_ts and not pool_const and carry:
                # Seasonal block absent from the whole window: carry the last
                # month it did occur in. Blocks 1 and 5 are seasonal, and
                # reading November's block 1 off an October that has none would
                # agree it to the floor and charge every winter peak as excess.
                for j in range(newest, -1, -1):
                    if group_ts.get((b, j)):
                        pool_ts = list(group_ts[(b, j)])
                        break
                else:
                    pool_const = list(bootstrap.get(b, ()))

            size = len(pool_ts) + len(pool_const)
            if size == 0:
                continue                    # never measured: floor only
            k = min(k_peaks, size)
            terms = [buy[t] * (1.0 / hours) for t in pool_ts] + list(pool_const)

            if k == size:
                # The whole pool is the top k, so the mean is already linear --
                # no epigraph, no slacks. Also the only shape in which the
                # epigraph would be unbounded below (tau -> -inf).
                prob += k * agreed_vars[(b, m)] >= pulp.lpSum(terms)
                continue

            tau = pulp.LpVariable(tau_name.format(b=b, m=m), lowBound=None)
            slacks = []
            for i, term in enumerate(terms):
                s = pulp.LpVariable(slack_name.format(b=b, m=m, i=i), lowBound=0)
                prob += s >= term - tau
                slacks.append(s)
            prob += (k * agreed_vars[(b, m)]
                     >= k * tau + pulp.lpSum(slacks))

    coefficients, constant = fixed_charge_coefficients(
        dates, month_idx_t, months_sorted, interval_minutes,
        env.pricing_scheme, pricing_options,
    )
    objective_terms = [constant]
    objective_terms += [coefficients[(b, m)] * agreed_vars[(b, m)]
                        for m in range(n_months) for b in _AGREED_BLOCKS]
    return agreed_vars, objective_terms


def solved_agreed_power_schedule(env, agreed_vars, months_sorted):
    """`{month id: {block: kW}}` for the contract the solve settled on."""
    return {
        y * 12 + mo - 1: {
            b: float(_value(agreed_vars[(b, m)])) for b in _AGREED_BLOCKS
        }
        for m, (y, mo) in enumerate(months_sorted)
    }


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
    endogenous_agreed_power=None,
):
    """Cost-minimal battery dispatch over one episode, as a per-step DataFrame.

    `initial_soc_kwh` / `final_soc_kwh` pin the stored energy at the horizon
    ends; `annual_netting_rate_eur_per_kwh` adds the NET-metering settlement
    credit = rate * min(sum P_buy, sum P_sell) to the objective.

    `endogenous_agreed_power` rolls the dogovorjena moc from the peaks THIS
    dispatch achieves rather than from the household's no-battery profile. Left
    at None it follows `env.agreed_power_from_dispatch`, which the study
    factories set.

    The contract then becomes part of the decision: it enters the LP as one
    variable per (block, month), tied to the peak variables of the month it is
    read from (`add_endogenous_agreed_power`), so the solve is still one solve
    and still the exact optimum -- which matters, because every rule-based
    controller is scored against it as a lower bound. A household that cannot be
    modelled that way (a connection-power ceiling) falls back to iterating whole
    solves to a fixed point instead, which converges but does not optimize.

    The settled contract is left in force on `env`, so a caller that prices the
    trajectory afterwards bills it against the same line the solve did.
    """
    if endogenous_agreed_power is None:
        endogenous_agreed_power = bool(getattr(env, "agreed_power_from_dispatch", False))
    if endogenous_agreed_power and env.connection_power_kw is not None:
        # No in-LP form: converge whole solves instead.
        hours = env.interval_minutes / 60.0
        kwargs = dict(
            start_idx=start_idx, n_steps=n_steps, initial_soc_kwh=initial_soc_kwh,
            final_soc_kwh=final_soc_kwh, problem_name=problem_name, solver=solver,
            annual_netting_rate_eur_per_kwh=annual_netting_rate_eur_per_kwh,
            invoice_output_dir=invoice_output_dir, invoice_run_label=invoice_run_label,
            do_not_use_previous_month=do_not_use_previous_month,
            endogenous_agreed_power=False,
        )

        def _dispatch():
            # The invoice is written once, from the run that is finally reported.
            df = solve_household(env, verbose=verbose, generate_invoice=False, **kwargs)
            return df, np.maximum(net_grid_kwh(df), 0.0) / hours

        df, info = converge_agreed_power(
            env, _dispatch, start_idx=start_idx, verbose=verbose
        )
        if generate_invoice:
            # Re-run under the settled contract, this time writing the invoice.
            df = solve_household(env, verbose=verbose, generate_invoice=True, **kwargs)
        df.attrs["agreed_power_iterations"] = info["iterations"]
        df.attrs["agreed_power_converged"] = info["converged"]
        return df
    # A signed constant is not a decision, whatever the flag says.
    endogenous_agreed_power = bool(endogenous_agreed_power) and bool(
        getattr(env, "agreed_power_lag_months", None)
    )

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
    # Only needed by the daily cycle cap, and walking 35 040 stamps through
    # `v_lokalni_cas` is not free, so it is built only when the cap is on.
    day_idx_t = None
    if getattr(env, "max_daily_cycles", None) is not None:
        _, _, day_idx_t = day_calendar(dates)
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
    import_rates, export_rates, constant_costs = interval_rate_vectors(
        env, dates, smp_prices, pricing_options, INTERVAL_MINS
    )
    # The fixed charge is the one rate that needs the agreed power, so it is not
    # part of the shared vectors above.
    fixed_monthly_costs = [
        compute_prorated_fixed_charge_eur(
            dates[t], INTERVAL_MINS, scheme=env.pricing_scheme,
            dogovorjena_moc=agreed_t[t], **pricing_options,
        )
        for t in range(N_STEPS)
    ]

    export_rates, n_export_floored = floor_export_rates(import_rates, export_rates)

    # --- 2) Model ------------------------------------------------------------
    prob = pulp.LpProblem(problem_name, pulp.LpMinimize)

    block = add_household_physics(
        prob, env, n_steps=N_STEPS, gen=gen, con=con,
        initial_soc_kwh=initial_soc_kwh, final_soc_kwh=final_soc_kwh,
        exclusivity="binary", day_idx_t=day_idx_t,
    )
    P_buy, P_sell, P_spill = block.buy, block.sell, block.spill
    P_ch, P_dis, E = block.charge, block.discharge, block.soc

    # --- 2b) Excess-power ratchet --------------------------------------------
    block_arr = env.tariff_blocks[horizon]
    window_id_arr = env.reset_window_ids[horizon]

    # --- 2b0) The contract, when the solve gets to set it --------------------
    agreed_vars = None
    agreed_objective_terms = None
    # A horizon no longer than the lag has no month whose contract is set inside
    # it, so there is nothing for the solve to decide.
    if endogenous_agreed_power and len(months_sorted) > int(env.agreed_power_lag_months):
        agreed_vars, agreed_objective_terms = add_endogenous_agreed_power(
            prob, env, buy=P_buy, hours=INTERVAL_MINS / 60.0,
            months_sorted=months_sorted, month_idx_t=month_idx_t,
            blocks=block_arr, dates=dates, interval_minutes=INTERVAL_MINS,
            pricing_options=pricing_options, agreed_by_month=agreed_by_month,
            n_steps=N_STEPS,
        )

    peak_var, _, peak_objective_terms = add_excess_power_ratchet(
        prob, env, P_buy,
        blocks=block_arr, month_idx_t=month_idx_t, months_sorted=months_sorted,
        agreed_by_month=agreed_by_month, agreed_vars=agreed_vars,
        start_idx=start_idx, n_steps=N_STEPS,
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

    # --- 2b'') Battery wear --------------------------------------------------
    # A SHADOW PRICE, not a bill item. It exists so the solver weighs degradation
    # when it decides to cycle, and it must never reach an invoice: `Cum_Cost`
    # below is rebuilt from the solved flows through `calculate_interval_price`,
    # so nothing here can leak into it.
    #
    # `cycle_cost_eur_per_efc` is what one equivalent full cycle costs -- the
    # pack price over its rated cycle life (Battery_Economics). One EFC is
    # `2 * nominal` kWh through the store, hence the divisor.
    #
    # Consequence worth knowing: with this on, the objective is no longer the
    # reported bill, so a full-period solve no longer minimises `Cum_Cost`.
    cycle_objective_terms = []
    rate_efc = getattr(env, "cycle_cost_eur_per_efc", None)
    nominal_kwh = float(getattr(env, "nominal_capacity_kwh", env.battery_capacity_kwh))
    if rate_efc and E is not None and nominal_kwh > 0:
        per_stored_kwh = float(rate_efc) / (2.0 * nominal_kwh)
        cycle_objective_terms = [
            per_stored_kwh * (
                P_ch[t] * env.charge_efficiency + P_dis[t] / env.discharge_efficiency
            )
            for t in range(N_STEPS)
        ]

    # --- 2c) Objective -------------------------------------------------------
    # `constant_costs` is not here: the fixed-charge terms already carry the same
    # prorated fixed charge, and adding both double-counts it. With the contract
    # endogenous the fixed charge is linear in the agreed-power variables rather
    # than a per-interval constant; it is the same quantity either way, and
    # `fixed_charge_coefficients` is read off the same pricing function.
    fixed_terms = (agreed_objective_terms if agreed_objective_terms is not None
                   else fixed_monthly_costs)
    prob += pulp.lpSum(
        P_buy[t] * import_rates[t] - P_sell[t] * export_rates[t]
        for t in range(N_STEPS)
    ) + pulp.lpSum(fixed_terms) \
      + pulp.lpSum(peak_objective_terms) + pulp.lpSum(netting_objective_terms) \
      + pulp.lpSum(cycle_objective_terms)

    # --- 3) Solve ------------------------------------------------------------
    if verbose:
        print(f"Solving over {N_STEPS} steps... (This may take a moment)")
    if solver is None:
        solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"Warning: Solver ended with status {pulp.LpStatus[prob.status]}")

    # The contract the solve agreed to is what the trajectory is billed under,
    # here and in anything that prices it afterwards.
    if agreed_vars is not None:
        settled = env.agreed_power_schedule
        settled.update(solved_agreed_power_schedule(env, agreed_vars, months_sorted))
        env.apply_agreed_power_schedule(settled)
        agreed_by_month = agreed_power_by_month(
            env, start_idx, months_sorted,
            do_not_use_previous_month=do_not_use_previous_month,
        )
        agreed_t = [agreed_by_month[month_idx_t[t]] for t in range(N_STEPS)]

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

    # The envelope this trajectory was solved under, so a cached row can say what
    # physics produced it rather than leaving it to be inferred from the config
    # that happens to be loaded when the row is read back.
    df_results.attrs["nominal_capacity_kwh"] = nominal_kwh
    df_results.attrs["usable_capacity_kwh"] = float(env.battery_capacity_kwh)
    df_results.attrs["soc_min_frac"] = float(getattr(env, "soc_min_frac", 0.0))
    df_results.attrs["soc_max_frac"] = float(getattr(env, "soc_max_frac", 1.0))
    df_results.attrs["max_daily_cycles"] = getattr(env, "max_daily_cycles", None)
    df_results.attrs["cycle_cost_eur_per_efc"] = rate_efc
    # What the shadow price charged, reported so it can be read but never billed.
    df_results.attrs["cycle_cost_eur"] = (
        float(pulp.value(pulp.lpSum(cycle_objective_terms))) if cycle_objective_terms else 0.0
    )

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
    nominal_capacity_kwh=None,
):
    """Per-run energy and power metrics from one solved trajectory.

    `blocks` is the per-row network tariff block; given it, the peak import is
    also reported per block, which is the power the excess charge is measured
    against. `tariff_blocks` is the set of block ids to report.

    `EFC_AC_Legacy` always counts AC-side discharge only. What
    `Equivalent_Full_Cycles` counts is `efc_convention`: "store" measures energy
    through the store, charge and discharge averaged with the efficiencies
    applied, and "ac" repeats the legacy AC-side figure.

    `capacity_kwh` is the USABLE window and `nominal_capacity_kwh` the pack on
    the invoice; they differ only when a SOC window is set, and default to being
    the same. The two cycle figures divide by different ones on purpose:

        Equivalent_Full_Cycles   nameplate -- what the 6000-EFC rating and the
                                 service life are quoted against, so a narrower
                                 window reads as a longer life
        EFC_Usable               the usable window -- what `max_daily_cycles`
                                 constrains, so a cap can be checked from here
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

    usable_kwh = float(capacity_kwh)
    nominal_kwh = usable_kwh if nominal_capacity_kwh is None else float(nominal_capacity_kwh)

    # Energy moved through the STORE, not through the inverter.
    stored_in_kwh = charged * charge_efficiency
    stored_out_kwh = discharged / discharge_efficiency
    stored_kwh = stored_in_kwh + stored_out_kwh
    efc = stored_kwh / (2.0 * nominal_kwh) if nominal_kwh > 0 else 0.0
    efc_usable = stored_kwh / (2.0 * usable_kwh) if usable_kwh > 0 else 0.0

    out = {
        "Capacity_kWh": float(nominal_kwh),
        "Usable_Capacity_kWh": usable_kwh,
        "EFC_Usable": efc_usable * annualize,
        "Cost_EUR": float(df["Cum_Cost"].iloc[-1]) * annualize,
        "Charged_kWh": charged * annualize,
        "Discharged_kWh": discharged * annualize,
        "Stored_Out_kWh": stored_out_kwh * annualize,
        "Grid_Charged_kWh": float(grid_charged.sum()) * annualize,
        "Import_kWh": float(net.clip(min=0).sum()) * annualize,
        "Export_kWh": float((-net).clip(min=0).sum()) * annualize,
        "Curtailed_kWh": float(df["Spill_kW"].sum()) * annualize,
        "Peak_Import_kW": float(net.max()) / hours_per_step,
        "EFC_AC_Legacy": (discharged * annualize / nominal_kwh) if nominal_kwh > 0 else 0.0,
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
