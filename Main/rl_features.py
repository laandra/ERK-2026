"""What a learned controller may be SHOWN beyond the panel's first contract.

The panel's observation (`rl_control.FeatureSpec`, variant `fc_h24`) is the
calendar, the meter's current interval, the published prices and a 14-day
median forecast, the last three squeezed into four lookahead bins each. Every
other controller in the study reads more than that, and reads it sharper:

    the MPC arms     the price list at full resolution over the horizon, and a
                     per-interval forecast from the study's best forecaster
                     (`hbd_median14`: the 14-day median plus a fitted residual
                     AR)
    the price rules  where the current price RANKS within the day and against
                     the trailing week (`PriceRankDaily`, `PriceThreshold`)
    the peak rules   what the house usually presents to the grid -- a high
                     quantile of its own trailing draw (`PeakShaving`)

This module is those inputs, as named GROUPS a `FeatureSpec` switches on, so
what each one is worth to a learner is a paired ablation like every other
question the study asks. Nothing here is a forecast of its own or a second
copy of a rule: a group is a deterministic function of what the controller
already holds at the moment it acts --

    rank       the published price list
    net        the forecast the controller already has, read as the BATTERY
               reads it: the deficit it could serve and the surplus it could
               store, not the load and the roof side by side
    history    the household's own meter, strictly up to the current interval
    calendar   the tariff's working-day calendar and the ratchet window
    peak       the meter's past and the forecast, against the contract line

-- which is also the whole of what a deployed unit could obtain from this
data. There is no weather, irradiance or temperature series in the repository,
so none is used; the trailing clear-sky envelope in `history` is the roof's own
past standing in for one.

The two changes that are not groups live where the thing they change lives:
the lookahead RESOLUTION (`FeatureSpec.n_bins`) in `rl_control`, and the
forecast SOURCE (`FeatureSpec.forecast`, the AR re-issued every interval) in
`hems_study.HbdForecaster.rolling_forecasts`.

Causality is by construction and checked by `test_rl_features`: every column
is recomputed with the series poisoned after a cut, and the rows before the
cut must not move (the price columns excepted, which read the published
horizon like the panel's own price bins do).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from MILP_Household import v_lokalni_cas          # also puts si_cas on the path
import si_cas

GROUPS = ("rank", "net", "history", "calendar", "peak")

_EPS = 1e-9

# Trailing windows, in days. The price week and the 30-day draw are the rules'
# own defaults (`PriceThreshold.window_days`, `PeakShaving.window_days`,
# `PeakShaving.q_peak`), on purpose: the learner is shown what the rule it is
# compared with reads, not a better-tuned version of it.
PRICE_WINDOW_DAYS = 7
PEAK_WINDOW_DAYS = 30
PEAK_QUANTILE = 0.98
ENVELOPE_DAYS = 14


@dataclass
class Context:
    """The household's own past, for the groups that read further back than
    the bundle they are built for.

    A `Signals` bundle starts at its first scored step. A feature that reads
    yesterday, last week or the last 30 days needs the rows BEFORE that, or
    the scored year opens on a month of cold-start values no deployed unit
    would see. So the whole series travels here with the offset of the
    bundle's step 0 in it; everything is indexed `offset + t` and only ever
    backwards.

    `load` / `pv` are kWh per interval, the unit `Signals.consumption` is in.
    `load_m14` / `pv_m14` are the causal 14-day median at every step
    (`rl_control.median14_forecast` over the same series).
    """

    load: np.ndarray
    pv: np.ndarray
    load_m14: np.ndarray
    pv_m14: np.ndarray
    offset: int = 0

    def check(self, sig, n: int) -> None:
        """The series must BE the bundle's, shifted -- a context cut from
        another window scores one day's history against another's meter."""
        a = self.offset
        if a < 0 or a + n > len(self.load):
            raise ValueError("context does not cover the bundle")
        if not (np.array_equal(self.load[a:a + n], sig.consumption[:n])
                and np.array_equal(self.pv[a:a + n], sig.generation[:n])):
            raise ValueError("context series is not the bundle's own meter")


# ---------------------------------------------------------------------------
# Lookahead: one (n, L) view of "the next L intervals", reduced per bin
# ---------------------------------------------------------------------------
def bin_edges(n_bins: int, graded) -> tuple:
    """Bin edges as fractions of the horizon.

    Four bins are the panel's graded edges (finer near now), handed in by
    `rl_control` so there is one copy of them; any other count is uniform --
    at 24 bins over a 24 h horizon, the hourly resolution a 1.5 kW inverter
    that needs five hours to fill the pack can actually use.
    """
    if n_bins == len(graded) - 1:
        return tuple(graded)
    return tuple(np.linspace(0.0, 1.0, int(n_bins) + 1))


def future_matrix(series, n_steps: int, width: int):
    """`(values, valid)`, both `(n_steps, width)`: column j is interval t+1+j.

    `series` is either one array aligned with the bundle (entry t is interval
    t: prices, a day-ahead forecast, the truth) or an `(n, L)` matrix of
    forecasts issued at every step (row t, column j is the forecast made at t
    for t+1+j). Past the end of the bundle `valid` is False -- the same "no
    tail to spare" the panel's own bins apply.
    """
    series = np.asarray(series, dtype=float)
    t = np.arange(n_steps)[:, None] + 1 + np.arange(width)[None, :]
    valid = t < n_steps
    if series.ndim == 2:
        if series.shape[1] < width:
            raise ValueError(f"forecast matrix holds {series.shape[1]} steps "
                             f"ahead, the horizon asks for {width}")
        return series[:n_steps, :width], valid
    return series[np.minimum(t, len(series) - 1)], valid


def lookahead(series, horizon: int, n_steps: int, edges, how: str) -> np.ndarray:
    """`(n_steps, len(edges) - 1)` float32: `how` in sum | mean | max of the
    next `horizon` intervals, per bin. Invalid entries count as nothing: zero
    in a sum, left out of a mean, and a bin with none at all reads zero."""
    out = np.zeros((n_steps, len(edges) - 1), dtype=np.float32)
    if horizon <= 0:
        return out
    values, valid = future_matrix(series, n_steps, horizon)
    for k in range(len(edges) - 1):
        a = int(round(edges[k] * horizon))
        b = max(int(round(edges[k + 1] * horizon)), a + 1)
        v, ok = values[:, a:b], valid[:, a:b]
        count = ok.sum(axis=1)
        if how == "max":
            col = np.where(count > 0,
                           np.where(ok, v, -np.inf).max(axis=1), 0.0)
        else:
            col = np.where(ok, v, 0.0).sum(axis=1)
            if how == "mean":
                col = col / np.maximum(count, 1)
        out[:, k] = col
    return out


def _trailing_mean(x: np.ndarray, at: np.ndarray, width: int) -> np.ndarray:
    """Mean of the `width` entries ending at `at` INCLUSIVE; short at the start."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.maximum(at + 1 - width, 0)
    return (cs[at + 1] - cs[lo]) / np.maximum(at + 1 - lo, 1)


def trailing_price(sig, days: int = PRICE_WINDOW_DAYS) -> np.ndarray:
    """The mean delivered import rate of the `days` before each interval.

    What `FeatureSpec.price_reference == "trailing"` subtracts from every
    price column, so the network reads a price as dear or cheap against the
    last week rather than against the years it was trained on. That matters
    because the level moves and the shape does not: on this data the scored
    year's import rate averages 19 % (AU) and 32 % (SI) above the training
    years' -- on SI its 5th percentile sits at the training years' 95th --
    while the mean daily spread is within 4 % (AU) and inside the two
    training years' own range (SI). A battery trades the spread.

    Strictly past, like `PriceThreshold`'s window, and with its cold start: a
    bundle's first interval has no past and is its own reference.
    """
    n = sig.n_steps
    p = pd.Series(np.asarray(sig.import_rate[:n], dtype=float))
    spd = int(round(24.0 / sig.hours))
    return (p.shift(1).rolling(days * spd, min_periods=1).mean()
            .fillna(p).to_numpy())


# ---------------------------------------------------------------------------
# rank: where the price stands in the list the controller already holds
# ---------------------------------------------------------------------------
def rank_columns(sig, horizon: int) -> list:
    """Eight columns off the published prices.

    The panel hands a learner the current rate and four bin MEANS of the next
    day. A mean cannot say whether this half-hour is the cheapest of the
    night, and that -- not the level -- is what `price_rank_daily`, the best
    rule on the Ausgrid tariff, acts on. These are the rules' own readings:

        rank in the horizon and in the local day   (PriceRankDaily)
        distance to the cheapest and to the dearest interval ahead, the
        latter after round-trip losses             (the breakeven gate both
                                                   price rules carry)
        how many intervals away those two are
        distance to the trailing week's quartiles  (PriceThreshold)
    """
    n = sig.n_steps
    p = np.asarray(sig.import_rate[:n], dtype=float)
    h = max(int(horizon), 1)
    ahead, valid = future_matrix(p, n, h)
    count = np.maximum(valid.sum(axis=1), 1)
    cheaper = (np.where(valid, ahead, np.inf) < p[:, None] - _EPS).sum(axis=1)
    lo = np.where(valid, ahead, np.inf)
    hi = np.where(valid, ahead, -np.inf)
    some = valid.any(axis=1)
    p_min = np.where(some, lo.min(axis=1), p)
    p_max = np.where(some, hi.max(axis=1), p)

    rank_day = np.zeros(n)
    for steps in sig.day_steps:
        idx = np.asarray([s for s in steps if s < n], dtype=int)
        if idx.size:
            # Share of the day's intervals strictly cheaper than this one.
            order = p[idx]
            rank_day[idx] = (order[None, :] < order[:, None] - _EPS).mean(axis=1)

    spd = int(round(24.0 / sig.hours))
    # Strictly past: the window ends at the previous interval. The first
    # interval of a bundle has no past and reads its own price, the cold start
    # `PriceThreshold` documents.
    past = pd.Series(p).shift(1).rolling(PRICE_WINDOW_DAYS * spd, min_periods=1)
    q_lo = past.quantile(0.25).fillna(pd.Series(p)).to_numpy()
    q_hi = past.quantile(0.75).fillna(pd.Series(p)).to_numpy()

    return [
        cheaper / count,
        rank_day,
        p - p_min,                       # > 0: it gets cheaper within the horizon
        p_max * sig.eta_rt - p,          # > 0: a kWh bought now sells at a gain
        np.where(some, lo.argmin(axis=1), 0) / h,
        np.where(some, hi.argmax(axis=1), 0) / h,
        p - q_lo,
        p - q_hi,
    ]


# ---------------------------------------------------------------------------
# net: the forecast as the battery reads it
# ---------------------------------------------------------------------------
def net_columns(sig, load_fc, pv_fc, horizon: int, graded) -> list:
    """Twelve columns: per lookahead bin, the forecast DEFICIT, the forecast
    SURPLUS, and what that deficit would cost at the published price.

    The panel gives a learner the load forecast and the roof forecast as two
    sets of bin sums. What a battery can do with a bin is neither: it can
    serve the intervals where the house runs short and soak the ones where the
    roof runs over, and those are sums of `max(load - pv, 0)` and
    `max(pv - load, 0)` interval by interval -- which the two bin totals do
    not determine (a bin with 2 kWh of each may hold 2 kWh of deficit or
    none). The MILP is handed the per-interval forecast and takes the max for
    itself; a network handed bin totals cannot. The cost column is the same
    deficit weighted by the price it falls on: how much a stored kWh is worth
    there, and to how many of them.
    """
    if load_fc is None or pv_fc is None:
        raise ValueError("the net group reads the forecast net load: it needs "
                         "both the load and the PV channel")
    n = sig.n_steps
    h = max(int(horizon), 1)
    fc_load, valid = future_matrix(load_fc, n, h)
    fc_pv, _ = future_matrix(pv_fc, n, h)
    price, _ = future_matrix(np.asarray(sig.import_rate[:n], dtype=float), n, h)
    deficit = np.where(valid, np.maximum(fc_load - fc_pv, 0.0), 0.0)
    surplus = np.where(valid, np.maximum(fc_pv - fc_load, 0.0), 0.0)
    cols = []
    for part in (deficit, surplus, deficit * np.where(valid, price, 0.0)):
        for k in range(len(graded) - 1):
            a = int(round(graded[k] * h))
            b = max(int(round(graded[k + 1] * h)), a + 1)
            cols.append(part[:, a:b].sum(axis=1))
    return cols


# ---------------------------------------------------------------------------
# history: the household's own meter, up to and including this interval
# ---------------------------------------------------------------------------
def history_columns(sig, ctx: Context, graded) -> list:
    """Eighteen columns off the meter's past.

    What a residual-AR forecaster conditions on, handed over raw so the
    network can condition on it for CONTROL rather than for squared error:

        both channels   the previous interval, the last 3 h and 24 h, the same
                        interval yesterday and a week ago
        both channels   how far the last 3 h ran from the 14-day median --
                        "today is not a typical day", the AR's own input
        load level      the last 24 h against the 14 days before them. Below
                        about a half for days on end is a household that is
                        away (`anomaly.add_absence` finds 37 such runs in the
                        panel, centred; this is its causal half)
        clear sky       the last 3 h of generation against the best the roof
                        did at those clock positions in the last 14 days, and
                        that envelope summed over the panel's four lookahead
                        bins: the ceiling on what is still to come today
    """
    n = sig.n_steps
    ctx.check(sig, n)
    spd = int(round(24.0 / sig.hours))
    at = ctx.offset + np.arange(n)
    cols = []
    for x, m14 in ((ctx.load, ctx.load_m14), (ctx.pv, ctx.pv_m14)):
        x = np.asarray(x, dtype=float)
        prev = np.maximum(at - 1, 0)
        cols += [
            x[prev],
            _trailing_mean(x, prev, 6),
            _trailing_mean(x, prev, spd),
            x[np.where(at >= spd, at - spd, at)],
            x[np.where(at >= 7 * spd, at - 7 * spd,
                       np.where(at >= spd, at - spd, at))],
            _trailing_mean(x - np.asarray(m14, dtype=float), at, 6),
        ]

    load = np.asarray(ctx.load, dtype=float)
    day = _trailing_mean(load, at, spd)
    before = _trailing_mean(load, np.maximum(at - spd, 0), 14 * spd)
    cols.append(np.clip(day / np.maximum(before, _EPS), 0.0, 5.0))

    pv = np.asarray(ctx.pv, dtype=float)
    # The best the roof did at each clock position over the previous
    # ENVELOPE_DAYS days: rows are days back, so the current day is never in it.
    full = len(pv)
    idx = np.arange(full)[None, :] - spd * np.arange(1, ENVELOPE_DAYS + 1)[:, None]
    envelope = np.where(idx >= 0, pv[np.maximum(idx, 0)], 0.0).max(axis=0)
    cols.append(np.clip(_trailing_mean(pv, at, 6)
                        / np.maximum(_trailing_mean(envelope, at, 6), _EPS),
                        0.0, 2.0))
    # The envelope AHEAD is built from rows already past -- each entry reads at
    # least one whole day back -- so a day of it may be read forward freely.
    bins = lookahead(envelope[ctx.offset:], spd, n, graded, "sum")
    cols += [bins[:, k] for k in range(bins.shape[1])]
    return cols


# ---------------------------------------------------------------------------
# calendar: the tariff's own working days, and where the month stands
# ---------------------------------------------------------------------------
def calendar_columns(sig, with_window: bool) -> list:
    """Work-free today, work-free tomorrow, and (with the contract features)
    how far the ratchet window has run.

    Both tariffs price by working day -- EA025's peak and shoulder exist on
    working days only, the Slovenian blocks shift on work-free ones -- and the
    panel shows the learner a day-of-week sine, which cannot represent a
    public holiday at all. `si_cas.je_dela_prost` is the calendar the
    settlement itself asks (weekends and the configured region's holidays), so
    the feature and the bill cannot disagree.

    The window share is the one calendar fact the ratchet turns on: a peak set
    on the 2nd is paid for all month and one avoided on the 30th saves a day.
    """
    n = sig.n_steps
    stamps = sig.env.dataset.index[:n]
    free = np.zeros(n)
    free_next = np.zeros(n)
    cache = {}
    one_day = pd.Timedelta(days=1)
    for i, ts in enumerate(stamps):
        d = v_lokalni_cas(ts).date()
        if d not in cache:
            cache[d] = (float(si_cas.je_dela_prost(d)),
                        float(si_cas.je_dela_prost((pd.Timestamp(d) + one_day).date())))
        free[i], free_next[i] = cache[d]
    cols = [free, free_next]
    if with_window:
        w = np.asarray(sig.windows[:n])
        share = np.zeros(n)
        start = 0
        for i in range(1, n + 1):
            if i == n or w[i] != w[start]:
                # A window the bundle cuts short is measured on what is there:
                # only the first and the last can be, and a month is a month.
                share[start:i] = np.arange(i - start) / max(i - start, 1)
                start = i
        cols.append(share)
    return cols


# ---------------------------------------------------------------------------
# peak: the draw, its own past and its forecast, against the contract line
# ---------------------------------------------------------------------------
def peak_columns(sig, ctx: Context, load_fc, pv_fc, horizon: int, graded) -> list:
    """Seven columns for a tariff that bills power.

    The panel's contract features say where the LINE is (agreed power, the
    idle headroom now, the month's running peak). They do not say what the
    house usually reaches or what it is forecast to reach, which is what a
    peak shaver decides on: `PeakShaving` holds the draw under the 98th
    percentile of the trailing 30 days and earns two thirds of its saving
    from the contract that follows.

        the idle draw now against that percentile, and the percentile against
        the line
        the forecast idle draw over the horizon, per bin, as its worst
        interval's distance OVER the line it will be billed against
        the forecast's worst interval of the whole horizon against the
        percentile

    The percentile is read off the NO-BATTERY draw, not the meter: the meter
    depends on the agent's own past, and a static column cannot. That makes it
    the top of what the house presents unaided -- the thing to be shaved --
    where the rule's metered version is a target that walks down as it works.
    """
    n = sig.n_steps
    ctx.check(sig, n)
    if load_fc is None or pv_fc is None:
        raise ValueError("the peak group reads the forecast draw: it needs "
                         "both the load and the PV channel")
    spd = int(round(24.0 / sig.hours))
    hours = sig.hours
    draw = np.maximum(np.asarray(ctx.load, dtype=float)
                      - np.asarray(ctx.pv, dtype=float), 0.0) / hours
    top = (pd.Series(draw).shift(1)
           .rolling(PEAK_WINDOW_DAYS * spd, min_periods=1)
           .quantile(PEAK_QUANTILE).bfill().to_numpy())[ctx.offset:ctx.offset + n]
    now = np.maximum(sig.consumption[:n] - sig.generation[:n], 0.0) / hours
    agreed = np.asarray(sig.agreed_kw[:n], dtype=float)

    h = max(int(horizon), 1)
    fc_load, valid = future_matrix(load_fc, n, h)
    fc_pv, _ = future_matrix(pv_fc, n, h)
    fc_draw = (fc_load - fc_pv) / hours
    line, _ = future_matrix(agreed, n, h)
    over = np.where(valid, fc_draw - line, -np.inf)
    cols = [now - top, top - agreed]
    for k in range(len(graded) - 1):
        a = int(round(graded[k] * h))
        b = max(int(round(graded[k + 1] * h)), a + 1)
        part = over[:, a:b]
        cols.append(np.where(np.isfinite(part.max(axis=1)),
                             part.max(axis=1), 0.0))
    worst = np.where(valid, fc_draw, -np.inf).max(axis=1)
    cols.append(np.where(np.isfinite(worst), worst - top, 0.0))
    return cols


# ---------------------------------------------------------------------------
# The one entry point `rl_control.FeatureBuilder.build_static` calls
# ---------------------------------------------------------------------------
def group_columns(groups, sig, ctx: Context | None, load_fc, pv_fc,
                  horizon: int, graded, peak_features: bool) -> list:
    """The extra static columns for `groups`, in the fixed order of GROUPS --
    so a spec that lists them in another order is still the same network."""
    unknown = set(groups) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown feature group(s) {sorted(unknown)}; "
                         f"expected a subset of {GROUPS}")
    if ctx is None and {"history", "peak"} & set(groups):
        raise ValueError("the history and peak groups read the household's "
                         "past: they need a Context")
    cols = []
    if "rank" in groups:
        cols += rank_columns(sig, horizon)
    if "net" in groups:
        cols += net_columns(sig, load_fc, pv_fc, horizon, graded)
    if "history" in groups:
        cols += history_columns(sig, ctx, graded)
    if "calendar" in groups:
        cols += calendar_columns(sig, with_window=peak_features)
    if "peak" in groups:
        cols += peak_columns(sig, ctx, load_fc, pv_fc, horizon, graded)
    return [np.asarray(c, dtype=float) for c in cols]
