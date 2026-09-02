"""Rule-based battery controllers: the deployable alternative to the MILP.

`MILP_Household.solve_household` is a perfect-foresight optimizer. A household
cannot install one -- it installs an inverter that runs a rule. This module is
the other side of that comparison: eight controllers that decide one interval at
a time, from what they can actually know, and a runner that executes and prices
them through exactly the same settlement the MILP is scored through.

The load-bearing constraint is that a rule and the MILP differ ONLY in how they
pick a setpoint. Everything else is shared, not re-implemented:

    the envelope    `Basic_Functions.max_charge_now` / `max_discharge_now`, the
                    same AC-side bounds `add_household_physics` writes into the LP
    the price       `MILP_Household.interval_rate_vectors`, the same two vectors
                    the MILP objective is built from
    the cost        `price_interval` below, the single evaluator, carrying one
                    running peak state per run
    the calendar    `MILP_Household.day_calendar`, local days, DST-correct

The runner re-clamps every setpoint to the feasible bounds, so a rule cannot
violate the physics even if it asks to; `SOC_Drift_kWh` is 0 by construction
rather than by trust. `run_policy` returns the same keys the MILP runner returns, so the two drop
into one results frame.

The agreed billing power is endogenous, as it is for the MILP: each month's
dogovorjena moc is re-agreed from the peaks the controller's own metered profile
realized the month before, so a rule that shaves its way onto a lower contract is
then held to it, and a rule that charges at night pays for the peak it made. The
runner iterates to that fixed point through `Environment.converge_agreed_power`;
the MILP decides it inside the LP instead (`MILP_Household.
add_endogenous_agreed_power`), so the reference is still an exact optimum and
still a lower bound.

Two asymmetries are handled explicitly rather than hidden:

    terminal SOC    every MILP strategy starts AND ends at 50 % of capacity, so
                    a rule that drains the pack in December would otherwise book
                    a free saving. `Cost_EUR_Closed` values the shortfall at the
                    mean delivered import rate and is the figure to compare.
    curtailment     the MILP may spill PV for free; no rule here does, matching
                    `no_battery_cost`. `Curtailed_kWh` reports it either way.

Causality: every controller except `price_oracle` uses only the past, the
calendar, and the day-ahead price -- which SIPX publishes at 12:45 the previous
day, so a rule reading today's prices is deployable. `price_oracle` reads the
whole year and exists only to split the gap to the MILP into "the rule is too
simple" and "the rule cannot see the future".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Basic_Functions import max_charge_now, max_discharge_now
from Pricing_Functions import calculate_interval_price
from Environment import converge_agreed_power
from MILP_Household import (
    SOC_FRACTION,
    day_calendar,
    floor_export_rates,
    interval_rate_vectors,
    month_calendar,
    steps_by_day,
)

# --- Study configuration ---------------------------------------------------
# The pack sizes a capacity sweep walks. 2.5 kWh is included deliberately: at a
# fixed 1.5 kW inverter it is a C=0.6 pack, above the 0.5 residential default,
# and it is the size at which a battery stops being able to ride an evening out.
CAPACITIES_KWH = [2.5, 5.0, 10.0, 20.0, 30.0]

# The two reference rows every unit carries, so a controller can be read against
# both bounds without joining another table.
NO_BATTERY = "no_battery"
MILP_REFERENCE = "milp_full_period"

_BLOCKS = (1, 2, 3, 4, 5)
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Settlement: the one evaluator every controller is priced through
# ---------------------------------------------------------------------------
# These three used to live in `Horizon_Comparison`, which is gone. They are the
# whole of what this module needed from it, and they are small enough that
# owning them here is better than reviving a study surface for three functions.
#
# The walk is `Basic_Functions.cumulative_interval_price_series`'s, interval for
# interval -- one running per-block peak, dropped on each ratchet reset-window
# boundary, priced against the dogovorjena moc in force. The only difference is
# that this one prices the trajectory a CONTROLLER produced rather than a fixed
# profile, so it takes the realized net draw as an argument.
def reset_windows(env, n_steps=None):
    """Per-interval ratchet reset-window id, as an int array."""
    n_steps = int(env.episode_length if n_steps is None else n_steps)
    ids = env.reset_window_ids
    if ids is None:
        # No ratchet configured: one window for the whole run, so the peak state
        # is never dropped and no excess charge can be billed twice.
        return np.zeros(n_steps, dtype=np.int64)
    return np.asarray(ids[:n_steps], dtype=np.int64)


def _drop_peak_on_window_start(peak_state, windows, idx):
    """Zero the running peak when a new ratchet window opens at `idx`.

    The excess-power charge is billed per window, so the peak it is measured
    against has to start each window at zero -- otherwise a January peak is
    charged for again every month of the year.
    """
    if idx == 0 or windows[idx] == windows[idx - 1]:
        return peak_state
    return {b: 0.0 for b in _BLOCKS}


def price_interval(env, idx, net_kwh, peak_state):
    """Price one executed interval under the SI schemes, carrying the ratchet.

    This is the DEFAULT settlement `run_policy` uses, and the signature every
    settlement has to match: `(env, idx, net_kwh, peak_state) ->
    (variable, energy, power, fixed, new_peak_kw)`. A study on a tariff these
    schemes do not describe -- the Ausgrid time-of-use list, say, which has no
    capacity charge and no per-block agreed power -- passes its own through
    `run_policy(settle=...)`. What must NOT happen is a study pricing its rules
    here and its MILP somewhere else: whatever separates two controllers then
    includes the difference between two evaluators.

    Returns `(variable, energy, power, fixed, new_peak_kw)`, all EUR.

        variable  what the decision moved: energy + power. This is the figure a
                  controller is compared on, because it is the only part any
                  controller can change.
        fixed     the prorated standing charge, reported separately rather
                  than dropped: it is large enough (a few hundred EUR a year)
                  that a saving quoted against the energy bill alone reads as
                  roughly twice the saving against the invoice.

                  NOT decision-independent under the SI schemes, which is easy
                  to assume and wrong. `_prorated_fixed_breakdown_eur` takes
                  `dogovorjena_moc`, so the network's fixed charge scales with
                  the agreed power -- and the agreed power is endogenous, rolled
                  from the peaks the controller itself realized. A peak shaver
                  therefore earns twice: once on `power` (the excess charge it
                  stops paying) and again on `fixed` (the smaller contract it
                  walks itself onto). Measured over 60 days on Ausgrid 127:
                  15.54 -> 13.24 EUR on fixed, on top of 1.78 -> 0.33 on power.
    """
    res = calculate_interval_price(
        smp_market_price_kwh=float(env.arr_price[idx]),
        total_consumed_kwh=float(net_kwh),
        utc_date=env.dataset.index[idx],
        interval_minutes=float(env.interval_minutes),
        scheme=env.pricing_scheme,
        dogovorjena_moc=env.agreed_power_at(idx),
        prev_peak_kw=peak_state,
        **(env.pricing_options or {}),
    )
    return (
        float(res["variable_price_aud"]),
        float(res["energy_component_eur"]),
        float(res["power_component_eur"]),
        float(res["constant_price_aud"]),
        dict(res["new_peak_kw"]),
    )


# ---------------------------------------------------------------------------
# What a rule is allowed to look at
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Signals:
    """Everything a controller may read, built once per run.

    Nothing here is a decision variable: these are the load, the roof, the price
    list and the calendar. A rule that wants a quantity not in this bundle is
    asking for something it could not have in the field.

    What the meter recorded is the one thing a rule may also know and cannot find
    here, because it depends on the rule's own past setpoints. It arrives one
    interval at a time through `Policy.observe`, which is how a real controller
    gets it too.
    """

    env: object
    n_steps: int
    hours: float                # hours per interval
    consumption: np.ndarray     # kWh per interval
    generation: np.ndarray      # kWh per interval
    surplus: np.ndarray         # max(generation - consumption, 0)
    deficit: np.ndarray         # max(consumption - generation, 0)
    import_rate: np.ndarray     # EUR per imported kWh, delivered, VAT included
    export_credit: np.ndarray   # EUR per exported kWh, floored at the import rate
    blocks: np.ndarray          # network tariff block 1..5
    windows: np.ndarray         # ratchet reset-window id
    agreed_kw: np.ndarray       # dogovorjena moc for THIS interval's block
    local_hour: np.ndarray      # hour of the local day, fractional
    day_idx: np.ndarray         # local-day index, DST-correct
    day_steps: list             # day index -> the step indices it covers
    capacity_kwh: float         # usable window
    max_charge_ac: float        # per-interval AC-side charge limit at full SOC
    max_discharge_ac: float     # per-interval AC-side discharge limit at full SOC
    eta_ch: float
    eta_dis: float
    eta_rt: float               # round-trip

    def net_load_kwh(self, idx):
        """Grid draw with the battery idle: positive imports, negative exports."""
        return float(self.consumption[idx] - self.generation[idx])


def build_signals(env, n_steps=None):
    """The signal bundle for one household-environment."""
    n_steps = int(env.episode_length if n_steps is None else n_steps)
    dates = env.dataset.index[:n_steps]

    # One local-time pass feeds both the day index and the clock: walking 35 000
    # stamps through `v_lokalni_cas` is not free, and the Fluvius `Z` suffix is
    # Brussels-local, so a clock rule read off the raw index would fire 1-2 h off
    # the tariff schedule.
    local_times, _, _, _ = month_calendar(dates)
    local_hour = np.array(
        [lt.hour + lt.minute / 60.0 for lt in local_times], dtype=float
    )
    _, _, day_idx_t = day_calendar(dates)
    day_idx = np.asarray(day_idx_t, dtype=int)

    pricing_options = dict(env.pricing_options or {})
    import_rates, export_rates, _ = interval_rate_vectors(
        env, dates, env.arr_price[:n_steps], pricing_options,
        int(round(env.interval_minutes)),
    )
    # The same floor the MILP applies: on a dynamic list SIPX can drive the
    # delivered import rate negative, and an uncapped credit would make the
    # buy/sell round trip pay for itself out of nothing.
    export_rates, _ = floor_export_rates(
        np.asarray(import_rates, dtype=float), np.asarray(export_rates, dtype=float)
    )

    consumption = np.asarray(env.arr_consumption[:n_steps], dtype=float)
    generation = np.asarray(env.arr_generation[:n_steps], dtype=float)
    blocks = np.asarray(env.tariff_blocks[:n_steps], dtype=int)
    agreed_kw = np.array(
        [float(env.agreed_power_at(i).get(int(blocks[i]), 0.0)) for i in range(n_steps)],
        dtype=float,
    )

    return Signals(
        env=env,
        n_steps=n_steps,
        hours=env.interval_minutes / 60.0,
        consumption=consumption,
        generation=generation,
        surplus=np.maximum(generation - consumption, 0.0),
        deficit=np.maximum(consumption - generation, 0.0),
        import_rate=np.asarray(import_rates, dtype=float),
        export_credit=np.asarray(export_rates, dtype=float),
        blocks=blocks,
        windows=reset_windows(env, n_steps),
        agreed_kw=agreed_kw,
        local_hour=local_hour,
        day_idx=day_idx,
        day_steps=steps_by_day(day_idx_t),
        capacity_kwh=float(env.battery_capacity_kwh),
        max_charge_ac=float(env.max_charge_kwh) / float(env.charge_efficiency),
        max_discharge_ac=float(env.max_discharge_kwh) * float(env.discharge_efficiency),
        eta_ch=float(env.charge_efficiency),
        eta_dis=float(env.discharge_efficiency),
        eta_rt=float(env.charge_efficiency) * float(env.discharge_efficiency),
    )


# ---------------------------------------------------------------------------
# The controllers
# ---------------------------------------------------------------------------
class Policy:
    """One rule. `setpoint` returns a signed AC-side kWh: + charges, - discharges.

    `lo` and `hi` are the feasible bounds the runner has already computed from
    the state of charge and the inverter, and the runner re-clamps the answer to
    them, so a rule may reason in whatever units are natural and never has to
    check feasibility itself.

    `peak_state` is the per-block running peak the ratchet has already charged
    for this month. A rule that consults it can tell a peak worth shaving from
    one whose cost is already sunk.
    """

    name = "policy"
    label = ""
    causal = True

    def reset(self, sig):
        """Precompute whatever the whole run needs. Called once, before step 0."""

    def observe(self, sig, idx, net_kwh, setpoint_kwh):
        """The interval just settled: what the METER read, and what was done.

        `net_kwh` is the realized grid flow after the rule's own setpoint -- the
        quantity the ratchet prices and the only draw a field controller can
        actually measure. A rule that adapts to the household's peaks has to
        learn from this rather than from the profile the house would have had
        with the battery idle, which needs a second meter behind the inverter and
        is not the profile the bill is written against.
        """

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.name}>"


def _self_consumption(sig, idx, lo, hi):
    """Soak the PV surplus, cover the deficit, never trade with the grid."""
    if sig.surplus[idx] > _EPS:
        return min(sig.surplus[idx], hi)
    if sig.deficit[idx] > _EPS:
        return -min(sig.deficit[idx], -lo)
    return 0.0


def _grid_charge_room(sig, idx, hi, peak_state, respect_peak, limit_kw=None):
    """Charge headroom that does not set a NEW monthly peak in this block.

    Without it every grid-charging rule buys its arbitrage twice: once at the
    supplier's rate and again as an excess-power charge, because a 2.9 kWh charge
    in a 15-minute interval is 11 kW of import on top of the house load. Real
    inverters cap against the connection just the same.

    The default ceiling is the tariff line -- the agreed power, or the month's
    running peak where that is already higher, since a peak once set is sunk.
    `limit_kw` imposes a tighter one: the shaving rules pass their own threshold,
    because a peak shaver that creates peaks is self-defeating even on a tariff
    that happens not to charge for it.

    PV surplus is never restricted by this: soaking it makes the net draw
    smaller, so the headroom it is measured against is correspondingly larger.
    """
    if not respect_peak:
        return hi
    block = int(sig.blocks[idx])
    ceiling_kw = max(float(sig.agreed_kw[idx]), float(peak_state.get(block, 0.0)))
    if limit_kw is not None:
        ceiling_kw = min(ceiling_kw, float(limit_kw))
    room_kwh = (ceiling_kw - sig.net_load_kwh(idx) / sig.hours) * sig.hours
    return max(0.0, min(hi, room_kwh))


def _cover_load(sig, idx, lo):
    """Discharge just enough to serve the house, never to export."""
    if sig.deficit[idx] <= _EPS:
        return 0.0
    return -min(sig.deficit[idx], -lo)


class SelfConsumption(Policy):
    """Charge the PV surplus, discharge into the deficit. Nothing else.

    The rule every residential inverter ships with, and the honest floor for the
    comparison: it never trades with the grid, so it cannot arbitrage a price
    spread and it cannot shave a peak it did not cause.
    """

    name = "self_consumption"
    label = "Self-consumption"

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        return _self_consumption(sig, idx, lo, hi)


class FixedSchedule(Policy):
    """A clock, and nothing but the clock.

    Charge at full rate through the night window, discharge into the evening
    window, soak the PV surplus in between. It reads no price at all, which is
    the point: on a flat list it is the whole of what a battery can do, and on a
    dynamic one it is what ignoring the price signal costs.

    The windows are half-open in LOCAL time, and may wrap midnight.
    """

    name = "fixed_schedule"
    label = "Fixed schedule"

    def __init__(self, charge_hours=(1.0, 5.0), discharge_hours=(18.0, 22.0),
                 respect_peak=True):
        self.charge_hours = charge_hours
        self.discharge_hours = discharge_hours
        self.respect_peak = respect_peak

    @staticmethod
    def _in_window(hour, window):
        start, end = window
        if start <= end:
            return start <= hour < end
        return hour >= start or hour < end     # wraps midnight

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        hour = sig.local_hour[idx]
        if self._in_window(hour, self.charge_hours):
            return _grid_charge_room(sig, idx, hi, peak_state, self.respect_peak)
        if self._in_window(hour, self.discharge_hours):
            return _cover_load(sig, idx, lo)
        return _self_consumption(sig, idx, lo, hi)


class PriceThreshold(Policy):
    """Buy below a low quantile of the recent price, sell above a high one.

    The thresholds are re-read at the start of each local day from a TRAILING
    window of delivered import rates, so the rule adapts to a moving market
    without ever seeing the future. The first day has no history and uses its
    own day-ahead prices.

    Charging is gated on round-trip breakeven: storing a kWh at `rate` only pays
    if the dear threshold clears it after both efficiencies, otherwise the rule
    is paying losses for the privilege of moving energy through time.
    """

    name = "price_threshold"
    label = "Price threshold"

    def __init__(self, window_days=7, q_low=0.25, q_high=0.75, respect_peak=True):
        self.window_days = int(window_days)
        self.q_low = float(q_low)
        self.q_high = float(q_high)
        self.respect_peak = respect_peak

    def reset(self, sig):
        self._lo_thr = np.empty(sig.n_steps, dtype=float)
        self._hi_thr = np.empty(sig.n_steps, dtype=float)
        window = self.window_days * int(round(24.0 / sig.hours))
        for steps in sig.day_steps:
            if not steps:
                continue
            start = steps[0]
            history = sig.import_rate[max(0, start - window):start]
            if history.size == 0:
                # No past to read: the day's own day-ahead prices, which SIPX
                # publishes the afternoon before and a controller really has.
                history = sig.import_rate[steps[0]:steps[-1] + 1]
            lo_thr = float(np.quantile(history, self.q_low))
            hi_thr = float(np.quantile(history, self.q_high))
            self._lo_thr[steps[0]:steps[-1] + 1] = lo_thr
            self._hi_thr[steps[0]:steps[-1] + 1] = hi_thr

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        rate = sig.import_rate[idx]
        lo_thr, hi_thr = self._lo_thr[idx], self._hi_thr[idx]
        if rate <= lo_thr and hi_thr * sig.eta_rt > rate:
            return _grid_charge_room(sig, idx, hi, peak_state, self.respect_peak)
        if rate >= hi_thr:
            return _cover_load(sig, idx, lo)
        return _self_consumption(sig, idx, lo, hi)


class PriceOracle(PriceThreshold):
    """`price_threshold` reading FORWARD instead of backward. NOT deployable.

    Identical rule, identical quantiles, identical breakeven gate -- the only
    difference is that the window sits in the future rather than in the past.
    That makes the pair a clean experiment: whatever separates them is worth
    exactly what foresight is worth to a threshold rule, with the shape of the
    rule held fixed.

    Any result from it is a diagnostic, never a recommendation.
    """

    name = "price_oracle"
    label = "Price threshold (foresight)"
    causal = False

    def reset(self, sig):
        self._lo_thr = np.empty(sig.n_steps, dtype=float)
        self._hi_thr = np.empty(sig.n_steps, dtype=float)
        window = self.window_days * int(round(24.0 / sig.hours))
        for steps in sig.day_steps:
            if not steps:
                continue
            start = steps[0]
            # The days still to come, this one included -- the mirror image of
            # the trailing window the causal rule reads.
            future = sig.import_rate[start:min(sig.n_steps, start + window)]
            if future.size == 0:
                future = sig.import_rate[steps[0]:steps[-1] + 1]
            self._lo_thr[steps[0]:steps[-1] + 1] = float(
                np.quantile(future, self.q_low)
            )
            self._hi_thr[steps[0]:steps[-1] + 1] = float(
                np.quantile(future, self.q_high)
            )


class PriceRankDaily(Policy):
    """Rank the day's intervals by price and pair the cheapest with the dearest.

    The day-ahead market is known the afternoon before, so ranking today's 96
    delivered rates is something a controller can genuinely do. The pack size
    decides how many intervals it can afford to fill; pairs are kept only while
    the dear one still clears the cheap one after round-trip losses, which is
    where the greedy match stops on its own.

    The ranking is blind to the order of the day, so a discharge slot can fall
    before the charge slot that was meant to fill it. The SOC bound absorbs
    that: the discharge simply does not happen. Fixing it properly is a schedule,
    and a schedule over a horizon is the MILP.
    """

    name = "price_rank_daily"
    label = "Day-ahead price rank"

    def __init__(self, max_share=0.25, respect_peak=True):
        # Never plan more than this share of a day, whatever the pack allows.
        self.max_share = float(max_share)
        self.respect_peak = respect_peak

    def reset(self, sig):
        plan = np.zeros(sig.n_steps, dtype=np.int8)
        # Intervals of charging it takes to fill the usable window once.
        n_fill = int(np.ceil(sig.capacity_kwh / max(sig.max_charge_ac, _EPS)))
        for steps in sig.day_steps:
            if len(steps) < 4:
                continue
            idx = np.asarray(steps)
            rates = sig.import_rate[idx]
            n = min(n_fill, int(self.max_share * len(idx)))
            if n < 1:
                continue
            order = np.argsort(rates, kind="stable")
            cheap = order[:n]                        # ascending
            dear = order[-n:][::-1]                  # descending
            for k in range(n):
                cheap_rate = rates[cheap[k]]
                dear_rate = rates[dear[k]]
                # The k-th best pair no longer pays -> nor does any after it.
                if dear_rate * sig.eta_rt <= cheap_rate:
                    break
                plan[idx[cheap[k]]] = 1
                plan[idx[dear[k]]] = -1
        self._plan = plan

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        action = self._plan[idx]
        if action > 0:
            return _grid_charge_room(sig, idx, hi, peak_state, self.respect_peak)
        if action < 0:
            return _cover_load(sig, idx, lo)
        return _self_consumption(sig, idx, lo, hi)


class PeakShaving(Policy):
    """Hold the grid draw under a threshold, and keep enough charge to do it.

    The ceiling is the tighter of two numbers, re-read at the start of each
    local day:

        the tariff line   `margin * dogovorjena moc` for the interval's block --
                          the point above which the excess-power charge starts
        the household     a high quantile of the METERED draw over a TRAILING
                          window -- the top of what this house actually presents
                          to the grid, battery included

    The second is what makes the rule bite, and it is measured on the meter, not
    on the load behind it. The distinction is the whole rule: a house whose
    battery already flattens its evenings has a metered 98th percentile far below
    its no-battery one, so a threshold read off the raw load would sit above
    everything the meter ever sees and the shaver would never fire. It also cuts
    the other way -- a night grid refill is a peak the raw load cannot show, and
    the meter does.

    Aiming at the metered quantile makes the threshold a moving target the rule
    walks down: shaving to it flattens the distribution, the next window's
    quantile is lower, and the rule follows until the pack runs out of energy to
    hold it there. That is the point rather than a defect. With the dogovorjena
    moc rolled from the same metered peaks (`Environment.set_achieved_power_kw`),
    every kW walked off the top is a kW off next month's contract.

    It is ratchet-aware: once a block's running peak for the month is already
    above this interval's draw, the charge is sunk and shaving buys nothing but
    round-trip losses, so the rule stands down.

    Refilling is sized, not greedy. The target is the largest daily shave the
    trailing window demanded -- measured on the meter with the rule's own shaves
    added back, since what the pack must hold is the excursion it removed, not
    the flat line it left behind. PV surplus is always taken; grid refill happens
    only in the off-peak window and only up to that target -- which is what
    keeps the rule usable on the four Fluvius groups that have no roof at all.
    """

    name = "peak_shaving"
    label = "Peak shaving"

    def __init__(self, margin=1.0, q_peak=0.98, window_days=30,
                 refill_hours=(0.0, 5.0), refill_headroom=1.2, ratchet_aware=True):
        self.margin = float(margin)
        self.q_peak = float(q_peak)
        self.window_days = int(window_days)
        self.refill_hours = refill_hours
        self.refill_headroom = float(refill_headroom)
        self.ratchet_aware = ratchet_aware

    def reset(self, sig):
        self._per_day = int(round(24.0 / sig.hours))
        self._window = self.window_days * self._per_day
        # Filled by `observe` as the run goes: the metered draw, and the same
        # draw with this rule's own shaves added back.
        self._meter_kw = np.full(sig.n_steps, np.nan, dtype=float)
        self._unshaved_kw = np.full(sig.n_steps, np.nan, dtype=float)
        self._shaving = np.zeros(sig.n_steps, dtype=bool)

        self._threshold = np.full(sig.n_steps, np.inf, dtype=float)
        self._target = np.zeros(sig.n_steps, dtype=float)
        # idx -> (first, last) for the first step of each local day. The day's
        # threshold is set there, from history that by then exists.
        self._day_opens = {}
        for steps in sig.day_steps:
            if steps:
                self._day_opens[steps[0]] = (steps[0], steps[-1] + 1)

    def _open_day(self, sig, first, last):
        """Set this day's threshold and refill target from the trailing meter."""
        start = max(0, first - self._window)
        meter = self._meter_kw[start:first]
        unshaved = self._unshaved_kw[start:first]
        if meter.size == 0:
            # Day one: nothing has been metered yet, so the only draw available
            # is the load itself. Every later day reads the meter.
            meter = unshaved = (
                sig.consumption[first:last] - sig.generation[first:last]
            ) / sig.hours

        household_kw = float(np.quantile(meter, self.q_peak))
        tariff_kw = self.margin * float(np.min(sig.agreed_kw[first:last]))
        threshold = min(household_kw, tariff_kw)
        self._threshold[first:last] = threshold

        # What a day of shaving at this threshold would have cost in energy over
        # the trailing window: the pack only has to hold the worst of them.
        # Measured on the unshaved draw -- the shaves already taken are exactly
        # the excursions a future day will have to take again. Sliced on the
        # nominal day, which is all a maximum needs.
        over = np.maximum(unshaved - threshold, 0.0) * sig.hours
        whole = over[: over.size - over.size % self._per_day]
        need = (float(whole.reshape(-1, self._per_day).sum(axis=1).max())
                if whole.size else float(over.sum()))
        self._target[first:last] = min(
            sig.capacity_kwh, self.refill_headroom * need
        )

    def observe(self, sig, idx, net_kwh, setpoint_kwh):
        self._meter_kw[idx] = net_kwh / sig.hours
        # Adding the shave back reconstructs the draw the meter would have
        # recorded had the rule stood down -- the excursion the pack absorbed.
        shaved_back = max(-float(setpoint_kwh), 0.0) if self._shaving[idx] else 0.0
        self._unshaved_kw[idx] = (net_kwh + shaved_back) / sig.hours

    def _shave(self, sig, idx, lo, peak_state):
        """Discharge that pulls the draw back to the threshold, or 0."""
        net_kw = sig.net_load_kwh(idx) / sig.hours
        threshold_kw = self._threshold[idx]
        if net_kw <= threshold_kw:
            return 0.0
        if self.ratchet_aware and float(peak_state.get(int(sig.blocks[idx]), 0.0)) >= net_kw:
            return 0.0
        shave = -min((net_kw - threshold_kw) * sig.hours, -lo)
        # Flagged only when it is really a shave: an empty pack returns 0 here
        # and the rule falls through to whatever else it does, which `observe`
        # must not then mistake for an excursion the battery absorbed.
        if shave < -_EPS:
            self._shaving[idx] = True
        return shave

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        day = self._day_opens.get(idx)
        if day is not None:
            self._open_day(sig, *day)
        shave = self._shave(sig, idx, lo, peak_state)
        if shave < -_EPS:
            return shave
        if sig.surplus[idx] > _EPS:
            return min(sig.surplus[idx], hi)
        shortfall = self._target[idx] - soc_kwh
        if shortfall > _EPS and FixedSchedule._in_window(
            sig.local_hour[idx], self.refill_hours
        ):
            return _grid_charge_room(sig, idx, min(hi, shortfall), peak_state, True,
                                     limit_kw=self._threshold[idx])
        return 0.0


class SelfConsumptionPeakShaving(PeakShaving):
    """Self-consumption with a reserve only peak shaving may spend.

    The hybrid a real home unit runs. Self-consumption earns most of the saving
    but empties the pack by evening, which is exactly when the household's peak
    arrives; holding back what the shaver is sized to need keeps it loaded
    without giving up much of the energy bill.

    The reserve is the peak-shaving refill target -- the worst daily shave the
    trailing METERED window demanded -- rather than a flat fraction of the pack,
    so it scales with the household instead of with the capacity that happens to
    be installed. Self-consumption is what makes reading the meter matter here:
    it is already flattening the profile the shaver then measures, so a reserve
    sized off the raw load would be sized for peaks this controller no longer
    presents.
    """

    name = "self_consumption_peak_shaving"
    label = "Self-consumption + peak shaving"

    def __init__(self, margin=1.0, q_peak=0.98, window_days=30,
                 refill_hours=(0.0, 5.0), reserve_cap_frac=0.5, ratchet_aware=True):
        super().__init__(margin=margin, q_peak=q_peak, window_days=window_days,
                         refill_hours=refill_hours, ratchet_aware=ratchet_aware)
        # However much the shave needs, it may never sit on more than this much
        # of the pack: a reserve that swallows the battery stops it earning.
        self.reserve_cap_frac = float(reserve_cap_frac)

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        day = self._day_opens.get(idx)
        if day is not None:
            self._open_day(sig, *day)
        # Shaving has first claim and may spend the whole pack.
        shave = self._shave(sig, idx, lo, peak_state)
        if shave < -_EPS:
            return shave
        reserve = min(self._target[idx], self.reserve_cap_frac * sig.capacity_kwh)
        if sig.surplus[idx] > _EPS:
            return min(sig.surplus[idx], hi)
        # The reserve has to be RESTOCKED, or the rule shaves once in January and
        # then spends the rest of the year as plain self-consumption -- which is
        # exactly what it does on the four Fluvius groups with no roof to refill
        # it. Grid refill is off-peak only and never goes past the reserve.
        if soc_kwh < reserve - _EPS and FixedSchedule._in_window(
            sig.local_hour[idx], self.refill_hours
        ):
            return _grid_charge_room(sig, idx, min(hi, reserve - soc_kwh), peak_state, True,
                                     limit_kw=self._threshold[idx])
        spendable = max(0.0, soc_kwh - reserve)
        lo_sc = max(lo, -spendable * sig.eta_dis)
        return _self_consumption(sig, idx, lo_sc, hi)


class DelayedPVCharge(Policy):
    """Fill the pack at sunset, not by ten in the morning.

    A greedy self-consumption rule charges flat out on the first sun and is full
    before the roof peaks, so the midday surplus -- the part most likely to be
    curtailed or sold at the worst credit of the day -- goes to the grid. Feed-in
    damping throttles the morning so the pack still has headroom when the peak
    arrives.

    The forecast is persistence: YESTERDAY's surplus profile, aligned by position
    in the day. If what is still to come exceeds the headroom, the rule takes
    only its proportional share. Causal, and about as much forecast as an
    inverter really has.
    """

    name = "delayed_pv_charge"
    label = "Delayed PV charge"

    def reset(self, sig):
        # day index -> the surplus profile of the day before, by position.
        self._previous = {}
        for d, steps in enumerate(sig.day_steps):
            if d == 0 or not sig.day_steps[d - 1]:
                continue
            self._previous[d] = sig.surplus[np.asarray(sig.day_steps[d - 1])]
        self._position = np.zeros(sig.n_steps, dtype=int)
        for steps in sig.day_steps:
            for j, t in enumerate(steps):
                self._position[t] = j

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        if sig.surplus[idx] <= _EPS:
            return -min(sig.deficit[idx], -lo) if sig.deficit[idx] > _EPS else 0.0

        previous = self._previous.get(int(sig.day_idx[idx]))
        if previous is None:
            return min(sig.surplus[idx], hi)          # no yesterday: plain soak

        # What yesterday still had left to give from this point in the day on.
        j = min(int(self._position[idx]), len(previous) - 1)
        still_to_come = float(previous[j:].sum())
        headroom = max(0.0, sig.capacity_kwh - soc_kwh)
        if still_to_come <= headroom + _EPS:
            return min(sig.surplus[idx], hi)          # it all fits: take it all
        share = headroom / still_to_come
        return min(sig.surplus[idx] * share, hi)


# name -> factory. Order is the order every table and chart reports in: the rules
# that only ever move the household's own PV, then the clock, then the ones that
# trade on a price, then the ones that watch the meter, and last the diagnostic
# that reads the future.
POLICIES = {
    "self_consumption": SelfConsumption,
    "fixed_schedule": FixedSchedule,
    "delayed_pv_charge": DelayedPVCharge,
    "price_threshold": PriceThreshold,
    "price_rank_daily": PriceRankDaily,
    "peak_shaving": PeakShaving,
    "self_consumption_peak_shaving": SelfConsumptionPeakShaving,
    "price_oracle": PriceOracle,
}
POLICY_ORDER = list(POLICIES)
# What every table reports in, references included.
CONTROLLER_ORDER = [NO_BATTERY] + POLICY_ORDER + [MILP_REFERENCE]


def make_policy(name, **kwargs):
    """One controller by name, with any of its knobs overridden."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}. Known: {', '.join(POLICY_ORDER)}")
    policy = POLICIES[name](**kwargs)
    policy.name = name
    return policy


# ---------------------------------------------------------------------------
# The runner: execute a rule and price it exactly as the MILP is priced
# ---------------------------------------------------------------------------
def run_policy(env, policy, n_steps=None, signals=None, keep_traces=False,
               settle=None, soc_init_kwh=None):
    """Execute one controller over the horizon and return its priced trajectory.

    The accounting is the MILP runner's, interval for interval, with the solve
    replaced by a call to the rule: the same starting
    SOC, the same running peak state dropped on the same window boundaries, the
    same shared evaluator, the same output keys. Only the setpoint differs.

    `signals` is reusable across every policy on one environment -- building it
    prices the whole year at unit volume -- so a caller running the roster should
    build it once and pass it in.

    `settle` is the evaluator, defaulting to `price_interval`. It is injectable
    so that a study on another tariff can price its rules and its MILP through
    ONE function rather than two, which is the only way the gap between them
    means what it says.

    With `env.agreed_power_from_dispatch` set the run is repeated until the
    dogovorjena moc it is billed under is the one its own peaks agree to, exactly
    as the MILP runner does it for the MILP. A rule is cheap,
    so the loop costs little here -- but it is the same loop, and a controller
    that shaves its way onto a lower contract is then held to it.
    """
    if getattr(env, "agreed_power_from_dispatch", False):
        hours = env.interval_minutes / 60.0
        base = build_signals(env, n_steps) if signals is None else signals
        spent = {"seconds": 0.0}


        def _dispatch():
            # Only the contract moves between iterations, and it enters the
            # bundle as one array; the rates and the calendar are rebound, not
            # rebuilt, so converging costs solves and not price vectors.
            out = _run_policy_once(env, policy, n_steps=n_steps,
                                   signals=rebind_agreed_power(base, env),
                                   keep_traces=True, settle=settle,
                                   soc_init_kwh=soc_init_kwh)
            spent["seconds"] += out["Runtime_s"]
            return out, np.maximum(out["_net_trace"], 0.0) / hours

        out, info = converge_agreed_power(env, _dispatch)
        # Every pass was work done to find this contract, not just the last.
        out["Runtime_s"] = spent["seconds"]
        out["Agreed_Power_Iters"] = info["iterations"]
        out["Agreed_Power_Converged"] = info["converged"]
        if not keep_traces:
            for key in ("_soc_trace", "_cost_trace", "_setpoints", "_net_trace"):
                out.pop(key, None)
        else:
            out.pop("_net_trace", None)
        return out

    out = _run_policy_once(env, policy, n_steps=n_steps, signals=signals,
                           keep_traces=keep_traces, settle=settle,
                           soc_init_kwh=soc_init_kwh)
    out.pop("_net_trace", None)
    out["Agreed_Power_Iters"] = 1
    out["Agreed_Power_Converged"] = True
    return out


def _run_policy_once(env, policy, n_steps=None, signals=None, keep_traces=False,
                     settle=None, soc_init_kwh=None):
    """One pass of the rule under the contract currently in force."""
    settle = price_interval if settle is None else settle
    sig = build_signals(env, n_steps) if signals is None else signals
    n_steps = sig.n_steps
    policy.reset(sig)

    capacity = sig.capacity_kwh
    # Where the pack starts, and the level the terminal adjustment closes it back
    # to. `SOC_FRACTION * capacity` is the default every MILP strategy uses, but a
    # study whose own controller starts somewhere else has to be able to say so:
    # a rule starting at 3.5 kWh stored and a MILP starting at 4.0 are not running
    # the same experiment, and the difference shows up as a saving.
    soc_target = SOC_FRACTION * capacity if soc_init_kwh is None else float(soc_init_kwh)
    soc = soc_target

    # The MILP's daily cap is a constraint on store throughput per LOCAL day;
    # inactive by default, and the runner honours the same budget so a rule is
    # not compared against a battery held to a different duty.
    daily_budget = None
    if getattr(env, "max_daily_cycles", None) is not None:
        daily_budget = float(env.max_daily_cycles) * 2.0 * capacity
    day_used = np.zeros(len(sig.day_steps), dtype=float)

    peak_state = {b: 0.0 for b in _BLOCKS}
    cost = energy_cost = power_cost = fixed_cost = 0.0
    charged = discharged = grid_charged = 0.0
    import_kwh = export_kwh = 0.0
    peak_import_kw = 0.0
    soc_drift = 0.0
    soc_trace = np.empty(n_steps, dtype=float)
    cost_trace = np.empty(n_steps, dtype=float)
    setpoints = np.zeros(n_steps, dtype=float)
    # The draw the trajectory actually presents to the meter, interval by
    # interval. The rules that watch the meter read it back through `observe`,
    # and the endogenous contract is re-rolled from it.
    net_trace = np.zeros(n_steps, dtype=float)

    t_start = time.time()
    for idx in range(n_steps):
        peak_state = _drop_peak_on_window_start(peak_state, sig.windows, idx)

        lo = -max_discharge_now(soc, sig.eta_dis, env.max_discharge_kwh)
        hi = max_charge_now(soc, sig.eta_ch, env.max_charge_kwh, capacity)
        if daily_budget is not None:
            day = int(sig.day_idx[idx])
            remaining = max(0.0, daily_budget - day_used[day])
            hi = min(hi, remaining / sig.eta_ch)
            lo = max(lo, -remaining * sig.eta_dis)

        # The rule may ask for anything; the envelope is the runner's.
        raw = float(policy.setpoint(sig, idx, soc, lo, hi, peak_state))
        p = min(max(raw, lo), hi)
        setpoints[idx] = p

        ch = max(p, 0.0)
        dis = max(-p, 0.0)
        # No rule curtails, matching `no_battery_cost`; the MILP may, and the
        # comparison reports `Curtailed_kWh` for both so the asymmetry shows.
        net = sig.consumption[idx] + ch - sig.generation[idx] - dis

        step_cost, e_part, p_part, f_part, peak_state = settle(
            env, idx, net, peak_state)
        cost += step_cost
        energy_cost += e_part
        power_cost += p_part
        fixed_cost += f_part

        net_trace[idx] = net
        # What the rule may learn from this interval is what the meter recorded,
        # which is the draw AFTER its own setpoint -- not the draw the house
        # would have had with the battery idle.
        policy.observe(sig, idx, net, p)

        soc_trace[idx] = soc
        cost_trace[idx] = cost

        stored = ch * sig.eta_ch - dis / sig.eta_dis
        if daily_budget is not None:
            day_used[int(sig.day_idx[idx])] += ch * sig.eta_ch + dis / sig.eta_dis
        soc += stored
        clipped = min(max(soc, 0.0), capacity)
        soc_drift = max(soc_drift, abs(clipped - soc))
        soc = clipped

        charged += ch
        discharged += dis
        grid_charged += max(0.0, ch - sig.surplus[idx])
        if net >= 0:
            import_kwh += net
            peak_import_kw = max(peak_import_kw, net / sig.hours)
        else:
            export_kwh += -net

    # The MILP closes the year at 50 % of capacity and a rule does not, so a
    # rule that runs the pack down in December would post the difference as a
    # saving. Valuing the shortfall at the mean delivered import rate closes it.
    mean_rate = float(np.mean(sig.import_rate))
    terminal_adj = (soc_target - soc) / sig.eta_ch * mean_rate

    nominal = float(getattr(env, "nominal_capacity_kwh", capacity))
    out = {
        "Cost_EUR": cost,
        "Cost_EUR_Closed": cost + terminal_adj,
        "Terminal_SOC_Adj_EUR": terminal_adj,
        "Energy_EUR": energy_cost,
        "Power_EUR": power_cost,
        # Not a constant across controllers under the SI schemes: it scales
        # with the agreed power, which is endogenous. See `price_interval`.
        "Fixed_EUR": fixed_cost,
        "Cost_EUR_Total": cost + terminal_adj + fixed_cost,
        "Final_SOC_kWh": soc,
        "Charged_kWh": charged,
        "Discharged_kWh": discharged,
        "Grid_Charged_kWh": grid_charged,
        "Curtailed_kWh": 0.0,
        "Import_kWh": import_kwh,
        "Export_kWh": export_kwh,
        "Peak_Import_kW": peak_import_kw,
        # AC-side discharge against the NAMEPLATE pack: the same denominator
        # `run_strategy` and `summarize_trajectory` use for `EFC_AC_Legacy`.
        "Equivalent_Full_Cycles": discharged / nominal if nominal > 0 else 0.0,
        "N_Solves": 0,
        "SOC_Drift_kWh": soc_drift,
        "Runtime_s": time.time() - t_start,
    }
    out["_net_trace"] = net_trace
    if keep_traces:
        out["_soc_trace"] = soc_trace
        out["_cost_trace"] = cost_trace
        out["_setpoints"] = setpoints
    return out


class _Idle(Policy):
    """Never touches the battery. The no-battery reference, priced by the runner.

    Written as a policy rather than as its own loop so the baseline row carries
    every column a controller row carries, and cannot drift from them.
    """

    name = NO_BATTERY
    label = "No battery"

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        return 0.0


def rebind_agreed_power(sig, env):
    """The same bundle with the contract re-read from the environment.

    The agreed power is the one signal an endogenous contract moves between
    convergence iterations, and it is the only one worth rebuilding: the rates,
    the local-time calendar and the load are unchanged by re-agreeing a kW.
    """
    from dataclasses import replace

    blocks = sig.blocks
    return replace(
        sig,
        agreed_kw=np.array(
            [float(env.agreed_power_at(i).get(int(blocks[i]), 0.0))
             for i in range(sig.n_steps)],
            dtype=float,
        ),
    )


def rebind_signals(sig, env):
    """The same load, price and calendar signals, re-bound to another battery.

    Everything expensive in the bundle -- the delivered rates, the local-time
    calendar, the agreed power -- is a property of the household, the price list
    and the year, not of the pack. Only the envelope changes with capacity, so a
    capacity sweep prices the year once instead of once per size.
    """
    from dataclasses import replace

    return replace(
        sig,
        env=env,
        capacity_kwh=float(env.battery_capacity_kwh),
        max_charge_ac=float(env.max_charge_kwh) / float(env.charge_efficiency),
        max_discharge_ac=float(env.max_discharge_kwh) * float(env.discharge_efficiency),
        eta_ch=float(env.charge_efficiency),
        eta_dis=float(env.discharge_efficiency),
        eta_rt=float(env.charge_efficiency) * float(env.discharge_efficiency),
    )

