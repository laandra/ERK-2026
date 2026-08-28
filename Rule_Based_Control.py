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
    the cost        `Horizon_Comparison.price_interval`, the single evaluator,
                    carrying one running peak state per run
    the calendar    `MILP_Household.day_calendar`, local days, DST-correct

The runner re-clamps every setpoint to the feasible bounds, so a rule cannot
violate the physics even if it asks to; `SOC_Drift_kWh` is 0 by construction
rather than by trust. `run_policy` returns the same keys `Horizon_Comparison.
run_strategy` returns, so the two drop into one results frame.

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
from pathlib import Path

import numpy as np
import pandas as pd

import Horizon_Comparison as hc
from Basic_Functions import max_charge_now, max_discharge_now
from MILP_Household import (
    SOC_FRACTION,
    day_calendar,
    floor_export_rates,
    interval_rate_vectors,
    month_calendar,
    steps_by_day,
)

# --- Study configuration ---------------------------------------------------
# The axes are Horizon_Comparison's, read from there rather than re-typed: the
# same eight Fluvius groups, the same four GEN-I price lists with the same
# eligibility scopes, the same agreed-power rule and ratchet reset.
CAPACITIES_KWH = [5.0, 10.0, 20.0, 30.0]

RESULTS_DIR = Path(__file__).resolve().parent / "Results" / "RBC_Comparison"

# The two reference rows every unit carries, so a controller can be read against
# both bounds without joining another table.
NO_BATTERY = "no_battery"
MILP_REFERENCE = "milp_full_period"

_BLOCKS = (1, 2, 3, 4, 5)
_EPS = 1e-12


# ---------------------------------------------------------------------------
# What a rule is allowed to look at
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Signals:
    """Everything a controller may read, built once per run.

    Nothing here is a decision variable: these are the load, the roof, the price
    list and the calendar. A rule that wants a quantity not in this bundle is
    asking for something it could not have in the field.
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
        windows=hc.reset_windows(env, n_steps),
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
        the household     a high quantile of the net load over a TRAILING
                          window -- the top of what this house actually draws

    The second is what makes the rule bite. The agreed power is set from the
    previous month's peaks, so a stationary household spends most of the year
    below it and a shaver aimed only at the tariff line never fires. Aiming at
    the household's own 98th percentile shaves the intervals that set the peak,
    which is the quantity the ratchet prices.

    It is ratchet-aware: once a block's running peak for the month is already
    above this interval's draw, the charge is sunk and shaving buys nothing but
    round-trip losses, so the rule stands down.

    Refilling is sized, not greedy. The target is the largest daily shave the
    trailing window actually demanded, so a 30 kWh pack does not buy 24 kWh
    every night to shave two. PV surplus is always taken; grid refill happens
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
        net_kw = (sig.consumption - sig.generation) / sig.hours
        window = self.window_days * int(round(24.0 / sig.hours))
        self._threshold = np.empty(sig.n_steps, dtype=float)
        self._target = np.zeros(sig.n_steps, dtype=float)

        for steps in sig.day_steps:
            if not steps:
                continue
            first, last = steps[0], steps[-1] + 1
            history = net_kw[max(0, first - window):first]
            if history.size == 0:
                history = net_kw[first:last]      # no past: the day itself
            household_kw = float(np.quantile(history, self.q_peak))
            tariff_kw = self.margin * float(np.min(sig.agreed_kw[first:last]))
            threshold = min(household_kw, tariff_kw)
            self._threshold[first:last] = threshold

            # What a day of shaving at this threshold would have cost in energy
            # over the trailing window: the pack only has to hold the worst of
            # them. Sliced on the nominal day, which is all a maximum needs.
            over = np.maximum(history - threshold, 0.0) * sig.hours
            per_day = int(round(24.0 / sig.hours))
            whole = over[: over.size - over.size % per_day]
            need = (float(whole.reshape(-1, per_day).sum(axis=1).max())
                    if whole.size else float(over.sum()))
            self._target[first:last] = min(
                sig.capacity_kwh, self.refill_headroom * need
            )

    def _shave(self, sig, idx, lo, peak_state):
        """Discharge that pulls the draw back to the threshold, or 0."""
        net_kw = sig.net_load_kwh(idx) / sig.hours
        threshold_kw = self._threshold[idx]
        if net_kw <= threshold_kw:
            return 0.0
        if self.ratchet_aware and float(peak_state.get(int(sig.blocks[idx]), 0.0)) >= net_kw:
            return 0.0
        return -min((net_kw - threshold_kw) * sig.hours, -lo)

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
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
    trailing window demanded -- rather than a flat fraction of the pack, so it
    scales with the household instead of with the capacity that happens to be
    installed.
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
def run_policy(env, policy, n_steps=None, signals=None, keep_traces=False):
    """Execute one controller over the horizon and return its priced trajectory.

    The accounting is `Horizon_Comparison.run_strategy`'s, interval for
    interval, with the solve replaced by a call to the rule: the same starting
    SOC, the same running peak state dropped on the same window boundaries, the
    same shared evaluator, the same output keys. Only the setpoint differs.

    `signals` is reusable across every policy on one environment -- building it
    prices the whole year at unit volume -- so a caller running the roster should
    build it once and pass it in.
    """
    sig = build_signals(env, n_steps) if signals is None else signals
    n_steps = sig.n_steps
    policy.reset(sig)

    capacity = sig.capacity_kwh
    soc_target = SOC_FRACTION * capacity
    soc = soc_target

    # The MILP's daily cap is a constraint on store throughput per LOCAL day;
    # inactive by default, and the runner honours the same budget so a rule is
    # not compared against a battery held to a different duty.
    daily_budget = None
    if getattr(env, "max_daily_cycles", None) is not None:
        daily_budget = float(env.max_daily_cycles) * 2.0 * capacity
    day_used = np.zeros(len(sig.day_steps), dtype=float)

    peak_state = {b: 0.0 for b in _BLOCKS}
    cost = energy_cost = power_cost = 0.0
    charged = discharged = grid_charged = 0.0
    import_kwh = export_kwh = 0.0
    peak_import_kw = 0.0
    soc_drift = 0.0
    soc_trace = np.empty(n_steps, dtype=float)
    cost_trace = np.empty(n_steps, dtype=float)
    setpoints = np.zeros(n_steps, dtype=float)

    t_start = time.time()
    for idx in range(n_steps):
        peak_state = hc._drop_peak_on_window_start(peak_state, sig.windows, idx)

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

        step_cost, e_part, p_part, peak_state = hc.price_interval(env, idx, net, peak_state)
        cost += step_cost
        energy_cost += e_part
        power_cost += p_part

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
    if keep_traces:
        out["_soc_trace"] = soc_trace
        out["_cost_trace"] = cost_trace
        out["_setpoints"] = setpoints
    return out


def run_milp_reference(env, n_steps=None):
    """The perfect-foresight whole-year solve, as the same row shape.

    `Horizon_Comparison.run_strategy` is called rather than re-implemented, so
    the reference is literally the horizon study's `full_period` strategy: same
    solver gap, same terminal SOC, same evaluator.
    """
    out = hc.run_strategy(
        env, "period", "block", soc_mode="fixed50", n_steps=n_steps,
        solver=hc.full_period_solver(),
    )
    out.pop("_soc_trace", None)
    out.pop("_cost_trace", None)
    # Pinned to 50 % at both ends by construction, so there is nothing to close.
    out["Terminal_SOC_Adj_EUR"] = 0.0
    out["Cost_EUR_Closed"] = out["Cost_EUR"]
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


# ---------------------------------------------------------------------------
# One (household, price list): every controller at every capacity
# ---------------------------------------------------------------------------
LEAD_COLUMNS = [
    "Dataset", "Household", "Tariff", "Capacity_kWh", "Controller", "Causal",
    "Cost_EUR_Closed", "Savings_EUR", "No_Battery_EUR", "Cost_EUR",
    "Terminal_SOC_Adj_EUR", "Energy_EUR", "Power_EUR",
]
KEY_COLUMNS = ["Dataset", "Household", "Tariff", "Capacity_kWh"]
CHECKPOINT_KEY = ["Controller", "Capacity_kWh"]


def unit_csv_path(output_dir, dataset, household_id, tariff=hc.DEFAULT_TARIFF):
    """Results file for one (household, price list); all capacities in it."""
    slug = hc.TARIFFS[tariff]["slug"]
    suffix = f"_{slug}" if slug else ""
    return Path(output_dir) / f"{dataset}_user_{household_id:03d}{suffix}.csv"


def _tag_row(row, dataset, household_id, tariff, capacity_kwh, controller, causal,
             baseline):
    row = dict(row)
    row.update(
        Dataset=str(dataset),
        Household=int(household_id),
        Tariff=str(tariff),
        Paket_ID=hc.TARIFFS[tariff]["paket_id"],
        Capacity_kWh=float(capacity_kwh),
        Controller=str(controller),
        Causal=bool(causal),
        No_Battery_EUR=float(baseline),
        # Against the CLOSED cost: a rule that ends the year with an emptier
        # pack than it started has not saved the difference, it has spent it.
        Savings_EUR=float(baseline) - float(row["Cost_EUR_Closed"]),
        Peak_Reset=hc.PEAK_RESET_TAG,
        Agreed_Power=hc.AGREED_POWER_TAG,
    )
    return row


def run_unit(dataset, household_id, tariff=hc.DEFAULT_TARIFF, capacities=None,
             policies=None, with_milp=True, n_steps=None, verbose=True,
             checkpoint_path=None):
    """Every controller at every capacity for one household on one price list.

    One row per (controller, capacity). With `checkpoint_path` the CSV is
    rewritten after each finished pair and pairs already in it are skipped, so
    an interrupted batch resumes.
    """
    capacities = list(CAPACITIES_KWH if capacities is None else capacities)
    policies = list(POLICY_ORDER if policies is None else policies)

    data = hc.load_user(household_id, dataset)

    rows, done = [], set()
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        previous = pd.read_csv(checkpoint_path)
        # Rows written under a different ratchet or agreed-power rule hold a
        # different quantity, not an older one; resuming into them would make
        # the controllers disagree about the charge.
        fresh = (
            hc.peak_reset_tag(previous) == {hc.PEAK_RESET_TAG}
            and hc.agreed_power_tag(previous) == {hc.AGREED_POWER_TAG}
        )
        if fresh:
            rows = previous.to_dict("records")
            done = set(zip(previous["Controller"], previous["Capacity_kWh"].astype(float)))
        else:
            print(f"  [{dataset} {household_id} | {tariff}] checkpoint written under "
                  f"peak reset {sorted(hc.peak_reset_tag(previous))} / agreed power "
                  f"{sorted(hc.agreed_power_tag(previous))}; recomputing from scratch",
                  flush=True)

    def flush():
        if checkpoint_path is not None:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(checkpoint_path, index=False)

    # The signal bundle is built once on the first capacity and re-bound after.
    base_sig = None
    for capacity in capacities:
        env = hc.build_env(data, capacity_kwh=capacity, tariff=tariff)
        if base_sig is None:
            base_sig = build_signals(env, n_steps)
        sig = rebind_signals(base_sig, env)

        # The baseline is the same bill at every capacity, but it is recomputed
        # per capacity rather than carried: it costs one idle pass and it is
        # what every saving on the row is measured from.
        baseline_row = run_policy(env, _Idle(), signals=sig)
        baseline = baseline_row["Cost_EUR"]

        wanted = [(NO_BATTERY, None)] + [(p, p) for p in policies]
        if with_milp:
            wanted.append((MILP_REFERENCE, None))

        for controller, policy_name in wanted:
            if (controller, float(capacity)) in done:
                continue
            if controller == NO_BATTERY:
                out, causal = baseline_row, True
            elif controller == MILP_REFERENCE:
                out, causal = run_milp_reference(env, n_steps=n_steps), False
            else:
                policy = make_policy(policy_name)
                out, causal = run_policy(env, policy, signals=sig), policy.causal
            rows.append(_tag_row(out, dataset, household_id, tariff, capacity,
                                 controller, causal, baseline))
            if verbose:
                print(f"  [{dataset} {household_id} | {tariff} | {capacity:g} kWh] "
                      f"{controller:30s} {rows[-1]['Cost_EUR_Closed']:9.2f} EUR  "
                      f"({rows[-1]['Runtime_s']:6.1f} s)", flush=True)
            flush()

    df = pd.DataFrame(rows)
    lead = [c for c in LEAD_COLUMNS if c in df.columns]
    return df[lead + [c for c in df.columns if c not in lead]]


# ---------------------------------------------------------------------------
# Batch driver (multiprocessing over households, one CSV per household-list)
# ---------------------------------------------------------------------------
def study_jobs(units=None, tariffs=None, per_group=hc.HOUSEHOLDS_PER_GROUP, groups=None):
    """The (dataset, household, price list) jobs, eligibility already applied.

    `Horizon_Comparison.tariff_allowed` does the filtering: a PV roof cannot
    sign Redni 2T, and running it anyway would price every exported kWh at zero
    and read as a tariff result when it is a product-eligibility one.
    """
    units = units or hc.study_units(groups=groups, per_group=per_group)
    return hc.study_jobs(units=units, tariffs=tariffs)


def _worker(args):
    dataset, household_id, tariff, capacities, n_steps, output_dir = args
    out_path = unit_csv_path(output_dir, dataset, household_id, tariff)
    df = run_unit(dataset, household_id, tariff=tariff, capacities=capacities,
                  n_steps=n_steps, verbose=True, checkpoint_path=out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return f"{out_path.name} ({len(df)} rows)"


def run_batch(jobs=None, capacities=None, n_steps=None, output_dir=RESULTS_DIR,
              n_workers=10):
    """Run every (dataset, household, price list) job in parallel.

    One CSV per job holding every capacity; finished work is skipped, so an
    interrupted run resumes where it stopped.
    """
    import multiprocessing as mp

    jobs = jobs or study_jobs()
    capacities = list(CAPACITIES_KWH if capacities is None else capacities)
    payload = [(g, i, t, capacities, n_steps, output_dir) for g, i, t in jobs]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"{len(payload)} jobs x {len(capacities)} capacities x "
          f"{len(POLICY_ORDER) + 2} controllers on {n_workers} workers", flush=True)

    started = time.time()
    with mp.get_context("spawn").Pool(n_workers) as pool:
        for n, message in enumerate(pool.imap_unordered(_worker, payload), start=1):
            print(f"[{n}/{len(payload)}] {message}  "
                  f"({time.time() - started:.0f}s elapsed)", flush=True)
    return len(payload)


def collect_results(output_dir=RESULTS_DIR, require_current_tags=True):
    """Concatenate every per-(household, price list) CSV written so far.

    Rows priced under a superseded ratchet reset or agreed billing power are
    dropped by default, exactly as `Horizon_Comparison.collect_results` drops
    them: the charge they carry is a different quantity, so mixing them into a
    mean would compare two studies.
    """
    files = sorted(Path(output_dir).glob("*user_*.csv"))
    if not files:
        return pd.DataFrame()
    out = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if require_current_tags:
        for column, current in (("Peak_Reset", hc.PEAK_RESET_TAG),
                                ("Agreed_Power", hc.AGREED_POWER_TAG)):
            if column not in out.columns:
                continue
            stale = out[column].astype(str) != current
            if stale.any():
                print(f"collect_results: dropped {int(stale.sum())} of {len(out)} rows "
                      f"priced under {column} {sorted(set(out.loc[stale, column]))} "
                      f"(current: {current!r}). Re-run the batch to replace them.",
                      flush=True)
                out = out[~stale].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score(df_all, reference=MILP_REFERENCE):
    """Add the gap columns, measured against the perfect-foresight optimum.

    The optimum is resolved per (household, price list, capacity), so a gap
    never mixes one pack size or one price list into another's. The semantics
    are `Horizon_Comparison.score`'s: `Gap_pct` is the share of the ACHIEVABLE
    saving a controller gives up, so 0 % is the MILP and 100 % is a battery that
    might as well not be there.
    """
    # Idempotent: `summarize` scores whatever it is handed, and it is routinely
    # handed a frame that has already been through here.
    derived = ["Optimum_EUR", "Gap_to_Optimum_EUR", "Achievable_EUR",
               "Saving_Captured_pct", "Gap_pct"]
    df_all = df_all.drop(columns=[c for c in derived if c in df_all.columns])
    optimum = (
        df_all[df_all["Controller"] == reference]
        .set_index(KEY_COLUMNS)["Cost_EUR_Closed"]
        .rename("Optimum_EUR")
    )
    df = df_all.join(optimum, on=KEY_COLUMNS)
    df["Gap_to_Optimum_EUR"] = df["Cost_EUR_Closed"] - df["Optimum_EUR"]
    df["Achievable_EUR"] = df["No_Battery_EUR"] - df["Optimum_EUR"]
    # A percentage of the achievable saving means nothing when there is barely
    # one: a two-cent optimum turns an 8 EUR shortfall into 36 000 %. Such units
    # drop out of the percentage, never out of `Gap_to_Optimum_EUR`.
    achievable = df["Achievable_EUR"].where(lambda s: s > hc.MIN_ACHIEVABLE_EUR)
    df["Saving_Captured_pct"] = 100.0 * df["Savings_EUR"] / achievable
    df["Gap_pct"] = 100.0 - df["Saving_Captured_pct"]
    return df


def _pooled_gap_pct(group):
    """Gap as a share of the achievable saving, pooled over the units.

    Sum of gaps over sum of achievable, so every unit is weighted by the euros
    actually at stake instead of every household counting the same.
    """
    denominator = group["Achievable_EUR"].sum()
    if abs(denominator) < 1e-9:
        return np.nan
    return 100.0 * group["Gap_to_Optimum_EUR"].sum() / denominator


def summarize(df_all, by=("Tariff", "Controller"), reference=MILP_REFERENCE):
    """Comparison table over `by`, averaged across household-units."""
    if df_all.empty:
        return pd.DataFrame(), df_all
    df = score(df_all, reference)
    by = list(by)
    pooled = (
        df.groupby(by)[["Gap_to_Optimum_EUR", "Achievable_EUR"]]
        .apply(_pooled_gap_pct)
        .rename("Gap_pct_pooled")
    )
    summary = (
        df.groupby(by)
        .agg(
            Units=("Cost_EUR_Closed", "size"),
            Gap_pct=("Gap_pct", "mean"),
            Worst_Gap_pct=("Gap_pct", "max"),
            Best_Gap_pct=("Gap_pct", "min"),
            Gap_EUR=("Gap_to_Optimum_EUR", "mean"),
            Cost_EUR=("Cost_EUR_Closed", "mean"),
            Savings_EUR=("Savings_EUR", "mean"),
            Cycles=("Equivalent_Full_Cycles", "mean"),
            Peak_kW=("Peak_Import_kW", "mean"),
            # A wide fixed price spread pays for grid-to-battery arbitrage,
            # which is a different business from storing your own roof.
            Charged_kWh=("Charged_kWh", "mean"),
            Grid_Charged_kWh=("Grid_Charged_kWh", "mean"),
            Runtime_s=("Runtime_s", "mean"),
        )
        .join(pooled)
    )
    return _order_rows(summary, by), df


def _order_rows(summary, by):
    """Reindex a summary onto the study's reporting order where it applies."""
    orders = {
        "Controller": CONTROLLER_ORDER,
        "Tariff": hc.TARIFF_ORDER,
        "Dataset": hc.DATASET_GROUPS,
    }
    if len(by) == 1:
        order = orders.get(by[0])
        if order is None:
            return summary
        return summary.reindex([k for k in order if k in summary.index])
    levels = [orders.get(name, list(dict.fromkeys(summary.index.get_level_values(name))))
              for name in by]
    wanted = [
        key for key in pd.MultiIndex.from_product(levels, names=by)
        if key in set(summary.index)
    ]
    return summary.reindex(pd.MultiIndex.from_tuples(wanted, names=by))


def cross_group_frame(df_scored):
    """The rows safe to average ACROSS Fluvius groups.

    Redni 2T is a no-PV-only product (`TARIFFS[...]["scope"]`), so a mean taken
    over it and the three samooskrba lists together compares which households
    may sign what, not which list is cheaper. It is dropped here rather than in
    each chart.
    """
    scoped = [t for t, spec in hc.TARIFFS.items() if spec.get("scope", "all") != "all"]
    return df_scored[~df_scored["Tariff"].isin(scoped)].copy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Rule-based controllers vs the MILP, over Fluvius groups, "
                    "price lists and battery sizes."
    )
    parser.add_argument("--per-group", type=int, default=hc.HOUSEHOLDS_PER_GROUP,
                        help="households per Fluvius dataset group")
    parser.add_argument("--groups", type=str, default=None,
                        help=f"comma-separated subset of {', '.join(hc.DATASET_GROUPS)}")
    parser.add_argument("--tariffs", type=str, default=None,
                        help=f"comma-separated subset of {', '.join(hc.TARIFF_ORDER)}")
    parser.add_argument("--capacities", type=str, default=None,
                        help="comma-separated battery sizes in kWh "
                             f"(default {', '.join(f'{c:g}' for c in CAPACITIES_KWH)})")
    parser.add_argument("--steps", type=int, default=None,
                        help="intervals to run (default: the whole dataset)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        unknown = set(groups) - set(hc.DATASET_GROUPS)
        if unknown:
            parser.error(f"unknown groups: {', '.join(sorted(unknown))}")

    tariffs = None
    if args.tariffs:
        tariffs = [t.strip() for t in args.tariffs.split(",") if t.strip()]
        unknown = set(tariffs) - set(hc.TARIFFS)
        if unknown:
            parser.error(f"unknown tariffs: {', '.join(sorted(unknown))}")

    capacities = CAPACITIES_KWH
    if args.capacities:
        capacities = [float(c) for c in args.capacities.split(",") if c.strip()]

    jobs = study_jobs(per_group=args.per_group, groups=groups, tariffs=tariffs)
    run_batch(jobs=jobs, capacities=capacities, n_steps=args.steps,
              output_dir=Path(args.output), n_workers=args.workers)


if __name__ == "__main__":
    main()
