"""The ERK 2026 HEMS study: what a forecast buys a home battery, and on which tariff.

Everything the study computes lives here; `CODE.ipynb` configures it, loads what
it wrote, and draws the figures. That split is the point. This file used to BE
the notebook -- one 1,980-line cell ending in `if __name__ == "__main__"`, which
in a Jupyter kernel is always true, so opening the notebook and running it
launched the whole sweep. As a module the guard means what it says, and the
notebook can be re-run in seconds to redraw a figure without re-solving a year.

Nothing here imports matplotlib. Figures are the notebook's half of the job and
go through `Plotting_Functions`, so there is one place that decides style, size
and export, and no figure is written from inside a batch run.

Entry points:

    python hems_study.py            the whole sweep, resumable
    hems_study.run_arms(...)        the same, from a notebook or a driver
    hems_study.collect_results()    what the sweep wrote, as one long frame
"""

import collections
import contextlib
import datetime
import glob
import hashlib
import json
import logging
import os
import time
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pulp
from prophet import Prophet

import hbd_forecast as hbd


@contextlib.contextmanager
def _quiet_fit():
    """Silence Prophet's fit chatter, and nothing else.

    The module used to open with a bare `warnings.filterwarnings("ignore")`,
    which is a loaded gun in a study: it hides the pandas and numpy deprecations
    that say a result is about to change, for the whole process, including code
    that has nothing to do with Prophet. Scoped here instead, so a warning
    raised anywhere else still reaches the log.
    """
    stan = logging.getLogger("cmdstanpy")
    prev, stan.disabled = stan.disabled, True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            yield
    finally:
        stan.disabled = prev


# =====================================================================
# 1 — TariffCalculator (unchanged)
# =====================================================================

class TariffCalculator:
    """
    AEMO/Ausgrid (Australia) tariff model, derived from the
    calculate_interval_price function provided by the colleague.
    """

    GST_RATE = 0.10
    MLF = 0.995
    DLF = 1.045
    ENV_MARKET_RATE_KWH = 0.0250

    # The SMP column arrives in EUR/kWh and this arm bills in AUD, so every
    # price on it is divided by this on the way in. It was a bare 0.615 inside
    # `rates`, which made it invisible to anything that needs to put a
    # non-electricity cost -- a battery quote -- on the same axis as a bill.
    # `ARM_CURRENCY_PER_EUR` below is the only other place that converts, and it
    # reads this attribute rather than repeating the number.
    EUR_PER_AUD = 0.615

    MONTHLY_SUBSCRIPTION_EX_GST = 20.00
    DAILY_SUPPLY_EX_GST         = 1.09
    DAYS_IN_MONTH               = 30

    # F9 - AEST is not "strictly UTC+10 year-round": NSW observes AEDT from
    # October to April, so a fixed +10 puts the peak window an hour off for
    # roughly half of every simulated year.
    #
    # F10 - and neither does Australia/Sydney, because the stamps were never UTC.
    # `Timestamp_UTC` carries a `+00:00` suffix the column does not earn: the
    # Ausgrid profiles are LOCAL NSW wall-clock readings laid on a continuous
    # 30-minute index. Measured, over 20 households, the PV centroid steps from
    # 12.49 to 13.13 on 2012-10-07 and back from 13.12 to 12.32 on 2013-04-07 --
    # the first Sunday in October and the first Sunday in April, NSW daylight
    # saving, sitting in the data itself. Converting them to Australia/Sydney
    # therefore added a SECOND +10/+11 h and priced the 15:00-21:00 peak window
    # against the household's 05:00-11:00.
    #
    # So the tariff reads the hour off the stamp and converts nothing, which is
    # what `LOCAL_TZ = None` means. A genuinely UTC-stamped profile sets an IANA
    # name here and gets the old behaviour.
    LOCAL_TZ = None

    # Ausgrid EA025 charges the peak rate on working days. The original model
    # applied it every day including weekends and public holidays, which is both
    # wrong and inconsistent with the Slovenian arm, whose blocks have always
    # distinguished them. Set False to recover the old, calendar-blind behaviour.
    WORKDAY_AWARE = True
    HOLIDAY_COUNTRY, HOLIDAY_SUBDIV = "AU", "NSW"

    @classmethod
    def _local(cls, utc_date: datetime.datetime) -> datetime.datetime:
        if cls.LOCAL_TZ is None:            # the stamp is already local; see F10
            return utc_date.replace(tzinfo=None)
        if utc_date.tzinfo is None:
            utc_date = utc_date.replace(tzinfo=datetime.timezone.utc)
        return utc_date.astimezone(ZoneInfo(cls.LOCAL_TZ))

    @classmethod
    def _is_workday(cls, d: datetime.date) -> bool:
        if d.weekday() >= 5:
            return False
        import holidays as _hol
        key = (d.year, cls.HOLIDAY_COUNTRY, cls.HOLIDAY_SUBDIV)
        cache = getattr(cls, "_holiday_cache", None)
        if cache is None:
            cache = cls._holiday_cache = {}
        if key not in cache:
            cache[key] = frozenset(_hol.country_holidays(
                cls.HOLIDAY_COUNTRY, subdiv=cls.HOLIDAY_SUBDIV, years=d.year).keys())
        return d not in cache[key]

    @classmethod
    def _network_rate_kwh(cls, local_dt: datetime.datetime) -> float:
        hour = local_dt.hour
        if cls.WORKDAY_AWARE and not cls._is_workday(local_dt.date()):
            # Non-working day: no peak. The solar-sponge window is a network
            # condition, not a working-day one, so it still applies.
            return 0.0270 if 10 <= hour < 15 else 0.0720
        if 15 <= hour < 21:
            return 0.2360
        elif 10 <= hour < 15:
            return 0.0270
        else:
            return 0.0720

    @classmethod
    def rates(cls, smp_eur_per_kwh: float, utc_date: datetime.datetime) -> tuple:
        # The SMP column is EUR per kWh (~0.018-0.023, max 0.956), NOT EUR/MWh:
        # the argument used to be called smp_market_price_mwh with a commented-out
        # "/ 1000.0", which is an invitation to a 1000x "fix". /EUR_PER_AUD is
        # the EUR->AUD conversion only.
        spot_price_kwh = smp_eur_per_kwh / cls.EUR_PER_AUD
        adjusted_spot_kwh = spot_price_kwh * cls.MLF * cls.DLF

        network_rate_kwh = cls._network_rate_kwh(cls._local(utc_date))

        buy_rate = (adjusted_spot_kwh + network_rate_kwh + cls.ENV_MARKET_RATE_KWH) * (1 + cls.GST_RATE)
        sell_rate = adjusted_spot_kwh

        return buy_rate, sell_rate

    @classmethod
    def rates_series(cls, smp_series: pd.Series) -> tuple:
        idx = smp_series.index
        buy_arr  = np.empty(len(smp_series))
        sell_arr = np.empty(len(smp_series))
        for i, (ts, smp) in enumerate(zip(idx, smp_series.values)):
            py_ts = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            b, s = cls.rates(float(smp), py_ts)
            buy_arr[i]  = b
            sell_arr[i] = s
        return buy_arr, sell_arr

    @classmethod
    def constant_cost_per_interval(cls, interval_minutes: int = 30) -> float:
        intervals_per_day   = (24 * 60) / interval_minutes
        intervals_per_month = intervals_per_day * cls.DAYS_IN_MONTH
        constant_cost_ex_gst = (
            cls.DAILY_SUPPLY_EX_GST / intervals_per_day
            + cls.MONTHLY_SUBSCRIPTION_EX_GST / intervals_per_month
        )
        return constant_cost_ex_gst * (1 + cls.GST_RATE)


# =====================================================================
# 2 — EnergyForecaster (unchanged)
# =====================================================================

class EnergyForecaster:

    # The settings live here rather than inline in __init__ so they can be
    # overridden per study AND hashed into the forecast cache key: a changed
    # model must not silently reuse predictions made by the old one.
    # yearly_seasonality is ON for both. It used to be off while the study
    # simulated a full year, which for PV removes the dominant signal there is:
    # the model could only offer a fixed daily profile scaled by trend, so the
    # forecast drifted further from the truth the further into the year it got.
    # Two years of training data is the minimum Prophet needs for the annual
    # term, and n_train=730 days is exactly that.
    DEFAULTS_CON = dict(
        seasonality_mode="additive",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        # Nothing downstream reads yhat_lower/yhat_upper -- the controller takes
        # the point forecast and the error metrics score it. Leaving this at its
        # default of 1000 makes every `predict` draw a thousand posterior samples
        # to build an interval that is then thrown away, which across 13 refits x
        # 365 anchors x 30 households is most of the forecasting bill.
        uncertainty_samples=0,
    )
    DEFAULTS_GEN = dict(
        seasonality_mode="multiplicative",
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        uncertainty_samples=0,
    )

    def __init__(self, params_con: dict | None = None, params_gen: dict | None = None):
        self.params_con = {**self.DEFAULTS_CON, **(params_con or {})}
        self.params_gen = {**self.DEFAULTS_GEN, **(params_gen or {})}
        # Built in `fit`, not here. `run_pipeline_for_file` constructs a
        # forecaster before consulting the forecast cache -- it needs
        # `.config()` for the cache key -- so on a cache hit these two models
        # would be built, never fitted, and dropped. `config()` reads the params,
        # not the models, so nothing needs them to exist yet.
        self.model_con = None
        self.model_gen = None
        self._fitted = False

    def config(self) -> dict:
        """The settings this forecaster's predictions depend on."""
        return {"con": self.params_con, "gen": self.params_gen}

    @staticmethod
    def _to_prophet_df(series: pd.Series) -> pd.DataFrame:
        df = series.reset_index()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
        df["y"]  = df["y"].clip(lower=0)
        return df

    def fit(self, df: pd.DataFrame, col_con: str, col_gen: str) -> None:
        with _quiet_fit():
            print("  [Forecaster] Training consumption model...")
            self.model_con = Prophet(**self.params_con)
            self.model_con.fit(self._to_prophet_df(df[col_con]))

            print("  [Forecaster] Training PV generation model...")
            self.model_gen = Prophet(**self.params_gen)
            self.model_gen.fit(self._to_prophet_df(df[col_gen]))

        self._fitted = True
        print("  [Forecaster] Ready.")

    def predict_next_day(self,
                          anchor_ts: pd.Timestamp,
                          horizon_steps: int = 48,
                          freq: str = "30min") -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")

        ts = anchor_ts.tz_localize(None) if anchor_ts.tzinfo else anchor_ts
        future = pd.DataFrame({"ds": pd.date_range(ts, periods=horizon_steps, freq=freq)})

        with _quiet_fit():
            fc_con = self.model_con.predict(future)[["ds", "yhat"]].rename(
                columns={"yhat": "yhat_con"})
            fc_gen = self.model_gen.predict(future)[["ds", "yhat"]].rename(
                columns={"yhat": "yhat_gen"})
        # Clipped at zero: neither a house nor a roof runs backwards, and a
        # negative forecast would ask the battery to absorb energy that is not
        # there.
        fc_con["yhat_con"] = fc_con["yhat_con"].clip(lower=0)
        fc_gen["yhat_gen"] = fc_gen["yhat_gen"].clip(lower=0)

        return fc_con.merge(fc_gen, on="ds")


# =====================================================================
# 2b — Study configuration, forecast cache and run checkpoints
# =====================================================================
#
# The sweep ahead of this pipeline is two tariffs x two horizons x 30 households
# x a simulated year, so nothing may be recomputed that has already been
# computed -- and nothing may be REUSED that was computed under different rules.
# Both halves are solved by one idea: every checkpoint carries a tag describing
# the study it was produced under, and a checkpoint whose tag no longer matches
# is dropped rather than resumed into.
#
# The two caches are keyed DIFFERENTLY, on purpose:
#
#   forecast cache  depends on the household, the training window and the
#                   forecaster -- and deliberately NOT on tariff, horizon or the
#                   leak flag, because none of those change a forecast. That is
#                   what makes the sweep cheap: Prophet is fit once per
#                   household and every later arm reads the same predictions.
#   run checkpoint  depends on all of it, because every one of those axes
#                   changes the answer.

# Anchored to THIS FILE, not to the cwd. A relative "forecast_cache" resolves
# against wherever the process happens to have started -- the notebook runs
# from Main/, a worker process or a test harness need not -- and a cache that
# moves when the cwd moves is a cache that silently misses and refits Prophet.
_HERE = os.path.dirname(os.path.abspath(__file__))
FORECAST_CACHE_DIR = os.environ.get(
    "ERK_FORECAST_CACHE", os.path.join(_HERE, "forecast_cache"))

# The solver every LP in this study goes through.
#
# CBC is a COMMAND-LINE solver: `PULP_CBC_CMD` writes the model to a temp MPS
# file and forks the `cbc` binary once per solve. On a 48-step household that
# round trip is 19.3 of the 24.7 ms a solve costs -- process overhead, not
# optimisation, on a model with 96 binaries that CBC itself dispatches in
# microseconds. Across the sweep's 7.9 million solves it is most of the bill.
#
# HiGHS runs IN PROCESS through `highspy`: no file, no fork. Measured at 4.35 ms
# against CBC's 12.96 ms on this model shape, with the objective identical to
# the last bit over 25 random instances -- the same optimum, reached without the
# subprocess. `gapRel`/`gapAbs` are pinned to zero because HiGHS otherwise stops
# at a near-optimal incumbent inside a default MIP gap, and a controller that
# accepts a 0.01 % worse plan 17,520 times is a changed result, not drift.
#
# Falls back to CBC when highspy is not installed, so the study still runs on an
# environment that has only the wheel PuLP ships with. SOLVER_NAME goes into the
# run checkpoint: which solver produced a number is part of what makes it
# reproducible, and mixing two vintages in one panel is the thing the
# checkpoint tag exists to prevent.
try:
    import highspy as _highspy       # noqa: F401
    SOLVER_NAME = "HiGHS"
except ImportError:
    SOLVER_NAME = "CBC"


def make_solver():
    """A fresh solver instance. Not a module-level singleton, deliberately.

    PuLP solver objects carry per-solve state, and the sweep runs them from
    several worker processes; one shared instance is a race waiting to happen.
    Constructing one is microseconds against a millisecond solve.
    """
    if SOLVER_NAME == "HiGHS":
        return pulp.HiGHS(msg=False, gapRel=0.0, gapAbs=0.0, threads=1)
    return pulp.PULP_CBC_CMD(msg=0)



def config_digest(config: dict) -> str:
    """Short stable digest of a config dict. Sorted, so key order cannot matter."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


class TableForecaster:
    """Serves precomputed day-ahead forecasts, keyed by anchor timestamp.

    Drop-in for EnergyForecaster from ReactiveController's point of view: it
    only ever calls predict_next_day. Reading a table instead of calling Prophet
    inside the loop is what lets a second tariff or horizon reuse the first
    run's forecasts, and it also makes the arm reproducible -- the predictions
    are a file, not a side effect of fitting order.
    """

    def __init__(self, table: pd.DataFrame):
        self._by_anchor = {
            anchor: group[["ds", "yhat_con", "yhat_gen"]].reset_index(drop=True)
            for anchor, group in table.groupby("anchor", sort=False)
        }

    def predict_next_day(self, anchor_ts, horizon_steps: int = 48,
                         freq: str = "30min") -> pd.DataFrame:
        try:
            day = self._by_anchor[anchor_ts]
        except KeyError:
            raise KeyError(
                f"no cached forecast anchored at {anchor_ts}; the cache was "
                f"built for a different simulation window"
            ) from None
        if len(day) < horizon_steps:
            raise ValueError(
                f"cached forecast at {anchor_ts} is {len(day)} steps, "
                f"need {horizon_steps}"
            )
        return day.iloc[:horizon_steps].copy()


def build_forecast_table(forecaster, anchors, horizon_steps: int,
                         freq: str) -> pd.DataFrame:
    """One day-ahead forecast per anchor, stacked into a single frame."""
    frames = []
    for anchor in anchors:
        day = forecaster.predict_next_day(anchor, horizon_steps, freq=freq)
        day.insert(0, "anchor", anchor)
        frames.append(day)
    return pd.concat(frames, ignore_index=True)


def build_forecast_table_refit(params_con, params_gen, df_train, df_sim,
                               horizon_steps: int, freq: str,
                               refit_every_days: int,
                               col_con: str = "Energy_Consumption",
                               col_gen: str = "Energy_Generation") -> pd.DataFrame:
    """Day-ahead forecasts from a model refit every `refit_every_days`.

    Fitting once on the training block and predicting a whole year makes the
    forecast used on simulation day 300 a 300-day-ahead extrapolation, which no
    deployed controller would tolerate and which the trend term is not fit to
    support. Here the window EXPANDS: block b is predicted by a model fit on the
    training data plus every simulated interval before block b starts.

    Causality is the point, so the slice is `df_sim.iloc[:start]` -- strictly
    before the block being predicted. Nothing inside a block sees itself.
    """
    frames = []
    n = len(df_sim)
    block = int(refit_every_days) * horizon_steps
    for start in range(0, n, block):
        history = pd.concat([df_train, df_sim.iloc[:start]]) if start else df_train
        model = EnergyForecaster(params_con, params_gen)
        model.fit(history, col_con, col_gen)
        anchors = list(df_sim.index[start:start + block:horizon_steps])
        frames.append(build_forecast_table(model, anchors, horizon_steps, freq))
        print(f"  [Forecaster] refit {len(frames)} on {len(history)} steps "
              f"-> {len(anchors)} day(s) from {anchors[0]:%Y-%m-%d}")
    return pd.concat(frames, ignore_index=True)


def _naive(index) -> pd.DatetimeIndex:
    """Forecast timestamps, tz-stripped -- the one convention every source uses.

    Prophet drops the timezone in `_to_prophet_df` and never puts it back, so its
    tables carry naive stamps. The sources that read a frame instead of fitting a
    model were handing back that frame's own tz-aware index, which compares
    unequal to an identical naive instant: the hybrid kinds refused to merge with
    "the prophet and persistence tables do not share an index (384 vs 384 rows)",
    which is the guard working and the convention being wrong. The data is UTC
    throughout, so nothing is lost by dropping the label.
    """
    idx = pd.DatetimeIndex(index)
    return idx.tz_localize(None) if idx.tz is not None else idx


class TruthForecaster:
    """Perfect day-ahead knowledge. NOT deployable; a bound, not a controller.

    Day-ahead and no further: the table it serves is anchored at the start of a
    local day and holds exactly that day, so a controller reading it knows the
    next 24 h perfectly and nothing beyond. That is the distinction from the
    oracle arm, which re-reads the truth at every step over a rolling horizon.

    It exists to split the forecast question in two. Paired with a Prophet
    consumption model under `kind="pvtruth"` it answers "what would a perfect PV
    forecast be worth?", which is the only way to read the study's most
    uncomfortable measurement: Prophet's generation model scores WORSE than
    yesterday-same-interval (skill vs seasonal-naive -0.13 over a year on
    Ausgrid 127, -0.08 over 14 days). If pvtruth buys little, the PV forecast is
    not where the money is and fitting a better one is wasted effort.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int):
        self.frame = frame
        self.spd = steps_per_day

    def fit(self, *a, **k):
        return self

    def config(self) -> dict:
        return {"kind": "truth", "steps_per_day": self.spd}

    def predict_next_day(self, anchor_ts, horizon_steps: int = 48,
                         freq: str = "30min") -> pd.DataFrame:
        i = self.frame.index.get_indexer([anchor_ts])[0]
        if i < 0:
            raise KeyError(f"anchor {anchor_ts} is not in the truth frame")
        window = self.frame.iloc[i:i + horizon_steps]
        return pd.DataFrame({
            "ds": _naive(window.index),
            "yhat_con": window["Energy_Consumption"].values,
            "yhat_gen": window["Energy_Generation"].values,
        })


# The fit-free sources, by the name an arm asks for them by: {kind: builder},
# where a builder takes (frame, steps_per_day) and returns a forecaster.
#
# The names are SELF-DESCRIBING on purpose, and that is load-bearing rather than
# cosmetic. `load_or_build_forecasts` folds `kind` into the cache digest but NOT
# a naive method's parameters -- `cfg["forecaster"]` is always the *Prophet*
# params, whatever the kind. So "median7" and "median14" must be different
# strings, or the second one silently serves the first one's cached table.
#
# `truth` is not here: it reads the simulated window rather than history, so it
# is built from `df_sim` and special-cased in `channel_table`.
# Both route to the Prophet path; which SETTINGS they use comes from the arm's
# `forecaster_params_con`/`_gen`, and the cache digest already separates them by
# those params. The second name exists so a tuned arm is distinguishable in a
# group-by -- `AU_H24_leaked` shares `forecaster_kind == "prophet"` with
# `AU_H24`, and a figure that groups by kind silently averages the two.
PROPHET_KINDS = ("prophet", "prophet_tuned")

SIMPLE_KINDS = {
    "persistence": lambda f, spd: SeasonalNaiveForecaster(f, spd, 1),
    "weekly":      lambda f, spd: SeasonalNaiveForecaster(f, spd, 7),
    "daytype":     lambda f, spd: DayTypeNaiveForecaster(f, spd),
    "mean3":       lambda f, spd: ClimatologyForecaster(f, spd, 3, "mean"),
    "mean7":       lambda f, spd: ClimatologyForecaster(f, spd, 7, "mean"),
    "median7":     lambda f, spd: ClimatologyForecaster(f, spd, 7, "median"),
    "median14":    lambda f, spd: ClimatologyForecaster(f, spd, 14, "median"),
}

# Kinds that FIT on the training block and then read recent actuals to predict.
# Separate from SIMPLE_KINDS because they need a third argument -- how many
# leading rows of the frame are training -- and separate from PROPHET_KINDS
# because they are not Prophet and do not take its params.
#
# This is the slot the roster did not have. Every SIMPLE_KIND reads the last
# rows but fits nothing; Prophet fits but reads nothing at prediction time, so
# inside a refit block its answer for tomorrow 08:00 is independent of what
# happened at 07:00 today. `hbd` does both, which is the whole reason it is
# worth adding: {kind: lambda frame, spd, n_train -> forecaster}.
FITTED_KINDS = {
    "hbd":          lambda f, spd, n: HbdForecaster(f, spd, n, use_ar=True),
    # The paper's own ablation: identical object, AR stage switched off, so the
    # gap between the two prices exactly what conditioning on the last 24 h buys
    # and nothing else. Everything else -- features, quantile, refit cadence,
    # clipping -- is held fixed by construction rather than by care.
    "hbd_baseline": lambda f, spd, n: HbdForecaster(f, spd, n, use_ar=False),
    # NOT the paper's method: the paper's second stage on the study's own best
    # first stage. `hbd_median14` is a 14-day median at the same clock position
    # -- byte-identical to `median14` when the AR is switched off, which is what
    # makes the pair a clean control -- with the fitted residual AR on top.
    #
    # It is here because the measurement says the decomposition is the good idea
    # and the Fourier basis is the weak part of it. If this beats `median14`,
    # the reference's contribution ports onto this study's incumbent and the
    # Fourier stage was never the point.
    "hbd_median14": lambda f, spd, n: HbdForecaster(
        f, spd, n, use_ar=True, baseline_kind="climatology"),
}

# A forecast has two channels and they are not equally hard, so these kinds hold
# the consumption channel fixed and vary only the roof:
# {kind: (consumption source, generation source)}.
#
# This comment used to say Prophet earned its place on consumption at +0.2 to
# +0.3 skill. That was one household (Ausgrid 127, where it does score +0.21),
# read as though it were the study. `forecast_benchmark` over all 30 gives a
# MEDIAN skill of -0.10 on consumption and -0.17 on generation: Prophet loses to
# copying yesterday on both channels, and beats it on consumption for only 6 of
# 30 households. The split by channel is still worth keeping -- the two are not
# equally hard, and `median14` beats yesterday by +0.22 on load against +0.11 on
# the roof -- but it is no longer a split between where Prophet wins and where
# it loses.
HYBRID_KINDS = {
    "pvnaive": ("prophet", "persistence"),
    "pvtruth": ("prophet", "truth"),
    # The same question asked of a better naive roof. `forecast_benchmark` over
    # all 30 households puts a 14-day median at +0.11 skill on generation where
    # yesterday is 0.00 and Prophet is -0.17, so if this closes the
    # pvnaive-to-pvtruth gap, the gap was never about modelling the roof -- only
    # about not copying one cloudy day.
    "pvmedian14": ("prophet", "median14"),
    # Perfect roof knowledge offered to EVERY load model, not just Prophet's.
    # `pvtruth` alone answers "what is a perfect PV forecast worth to Prophet",
    # which is only the general question if Prophet is the best load model
    # available -- and it is not. Each of these is the upper bound on what the
    # roof channel can still buy the load model beside it, so the pair
    # (kind, kind_pvtruth) reads as one number: the value of the sun.
    "persist_pvtruth":  ("persistence", "truth"),
    "median14_pvtruth": ("median14", "truth"),
    "pvtruth_tuned":    ("prophet_tuned", "truth"),
    "hbd_pvtruth":      ("hbd", "truth"),
    # The same question asked of the method that actually wins the error axis.
    # Paired with the plain `hbd_median14` arm, the difference is what a perfect
    # roof forecast is still worth once the load channel is as good as this
    # study can make it -- the upper bound on what is left in the PV channel.
    "hbd_median14_pvtruth": ("hbd_median14", "truth"),
}

# Every kind the study can run, in the order a figure should read them, with the
# label it should carry. The notebook derives its axes from this rather than
# from a hardcoded tuple of its own, so a kind added above appears in the
# figures without editing them.
FORECAST_KIND_LABELS = {
    "persistence": "both: yesterday",
    "weekly":      "both: last week",
    "daytype":     "both: last like-day",
    "mean3":       "both: mean of 3 d",
    "mean7":       "both: mean of 7 d",
    "median7":     "both: median of 7 d",
    "median14":    "both: median of 14 d",
    "pvnaive":     "PV: yesterday",
    "pvmedian14":  "PV: median of 14 d",
    # "both", NOT "PV". The `PV:` prefix in this dict means "the GENERATION
    # channel is X, consumption is still Prophet" -- that is what `pvnaive`,
    # `pvmedian14` and `pvtruth` are, and they are `HYBRID_KINDS` entries with
    # ("prophet", other) for exactly that reason. `prophet` is a PROPHET_KINDS
    # entry: `channel_table` is called once with it and `build_forecast_table_refit`
    # fills BOTH yhat_con and yhat_gen from Prophet. It is also the default kind,
    # i.e. `AU_H24`/`SI_H24`, the reference arms everything else is measured
    # against -- so labelling the study's own subject "PV: Prophet" made it read
    # as one of the PV-channel variants and the baseline appeared to be missing
    # from the regret figure entirely.
    "prophet":     "both: Prophet",
    "prophet_tuned":    "both: Prophet, tuned",
    "pvtruth":     "PV: perfect",
    "persist_pvtruth":  "yesterday + perfect PV",
    "median14_pvtruth": "median of 14 d + perfect PV",
    "pvtruth_tuned":    "tuned Prophet + perfect PV",
    "hbd":         "both: season + AR",
    "hbd_baseline": "both: season only",
    "hbd_median14": "both: median of 14 d + AR",
    "hbd_median14_pvtruth": "median of 14 d + AR, perfect PV",
    "hbd_pvtruth": "season + AR + perfect PV",
    "truth":       "both: perfect",
}


def forecast_kind_label(kind: str) -> str:
    """The figure label for a kind, falling back to the kind itself."""
    return FORECAST_KIND_LABELS.get(kind, kind)


# ---------------------------------------------------------------------------
# What KIND of algorithm produced a number, and over how long a horizon
# ---------------------------------------------------------------------------
#
# The on-disk keys stay what they are -- they are the `cost_<name>` columns in
# every checkpoint and the directory names of every arm, and renaming them would
# invalidate a sweep to change a caption. So the algorithm and the horizon live
# HERE, in one table the tables and the figures both read, rather than in a
# `LABEL` dict copied into a notebook cell where a newly added controller can
# quietly appear unlabelled.
#
# The names were doing real damage before this existed. "oracle" was captioned
# "MILP, perfect foresight", which reads as the whole-year optimum and is a
# 24 h (or 11 h!) receding-horizon solve that happens to read realised data --
# so a reader could not tell the study's ceiling from its MPC arm, and the
# horizon, which is the entire point of the H24/H11 axis, appeared nowhere in
# the label at all. `milp_full` is the thing that name was describing.
#
#   RBC        rule-based control. No model, no forecast, no optimisation: a
#              clock, a threshold or a meter reading.
#   MPC        model predictive control. Re-solves a MILP over a finite horizon
#              every interval and commits the first one. The horizon is part of
#              the identity, so it is part of the name.
#   MILP       one solve over the whole scored period. Not deployable; the bound.
CONTROLLER_ALGORITHM = {
    "no_battery":       ("reference", "No battery"),
    "self_consumption": ("RBC", "RBC: self-consumption"),
    "fixed_schedule":   ("RBC", "RBC: fixed schedule (clock)"),
    "delayed_pv_charge": ("RBC", "RBC: delayed PV charge"),
    "price_threshold":  ("RBC", "RBC: price threshold (adaptive)"),
    "price_rank_daily": ("RBC", "RBC: day-ahead price rank"),
    "tariff_arbitrage": ("RBC", "RBC: tariff arbitrage"),
    "peak_shaving":     ("RBC", "RBC: peak shaving"),
    "self_consumption_peak_shaving": ("RBC", "RBC: self-consumption + peak shaving"),
    "price_oracle":     ("RBC", "RBC: price threshold, full-year foresight (diagnostic)"),
    "prophet":          ("MPC", "MPC-MILP {horizon}, Prophet forecast"),
    "oracle":           ("MPC", "MPC-MILP {horizon}, perfect foresight"),
    "milp_full":        ("MILP", "MILP, full {period} horizon (optimum)"),
}


def horizon_label(control_horizon, delta_t: float = 0.5) -> str:
    """A control horizon in STEPS, as the hours a reader thinks in.

    48 half-hour steps is the "H24" arm and 22 is "H11"; the arm names carry the
    hours and the parameter carries the steps, which is exactly the sort of
    mismatch that ends up mislabelled on a figure.
    """
    if control_horizon is None:
        return "?"
    hours = float(control_horizon) * float(delta_t)
    return f"{hours:.0f} h" if abs(hours - round(hours)) < 1e-9 else f"{hours:.1f} h"


def controller_label(name: str, control_horizon=None, delta_t: float = 0.5,
                     n_sim: int | None = None) -> str:
    """The display name for one controller, algorithm and horizon included.

    Unknown names fall back to the key with its underscores opened up, so a
    controller added tomorrow is legible before anyone remembers to label it.
    """
    entry = CONTROLLER_ALGORITHM.get(name)
    if entry is None:
        return name.replace("_", " ")
    template = entry[1]
    return template.format(
        horizon=horizon_label(control_horizon, delta_t),
        period=f"{n_sim} d" if n_sim else "period",
    )


def controller_family(name: str) -> str:
    """RBC / MPC / MILP / reference -- what KIND of thing a row is."""
    entry = CONTROLLER_ALGORITHM.get(name)
    return entry[0] if entry else "RBC"


def seasonal_naive(actual: pd.Series, spd: int) -> pd.Series:
    """Yesterday, same interval. The baseline any forecaster must beat."""
    return actual.shift(spd)


def forecast_error_metrics(table: pd.DataFrame, truth: pd.DataFrame,
                           spd: int, history: pd.DataFrame | None = None) -> dict:
    """Day-ahead error for PV and load, and skill against seasonal-naive.

    Reported for the whole simulation and split first month vs last, because a
    model fit once and extrapolated degrades over the year while a refit one
    should not -- and that difference is invisible in a single average.
    """
    ds = pd.DatetimeIndex(pd.to_datetime(table["ds"], utc=True)).tz_convert(
        truth.index.tz) if truth.index.tz is not None else pd.DatetimeIndex(
        pd.to_datetime(table["ds"]))
    fc = pd.DataFrame({"gen": table["yhat_gen"].values,
                       "con": table["yhat_con"].values}, index=ds)
    fc = fc[fc.index.isin(truth.index)].sort_index()
    act = truth.loc[fc.index]

    ref = pd.concat([history, truth]) if history is not None else truth
    naive = pd.DataFrame({
        "gen": seasonal_naive(ref["Energy_Generation"], spd),
        "con": seasonal_naive(ref["Energy_Consumption"], spd),
    }).loc[fc.index]

    out = {}
    month = 30 * spd
    for name, col in (("gen", "Energy_Generation"), ("con", "Energy_Consumption")):
        err = fc[name] - act[col]
        scale = float(act[col].mean()) or float("nan")
        out[f"{name}_mae"] = float(err.abs().mean())
        out[f"{name}_rmse"] = float(np.sqrt((err ** 2).mean()))
        out[f"{name}_nmae"] = float(err.abs().mean() / scale) if scale == scale else float("nan")
        nerr = (naive[name] - act[col]).abs().mean()
        out[f"{name}_skill_vs_naive"] = (
            float(1.0 - err.abs().mean() / nerr) if nerr and nerr == nerr else float("nan"))
        if len(err) >= 2 * month:
            out[f"{name}_mae_first_month"] = float(err.iloc[:month].abs().mean())
            out[f"{name}_mae_last_month"] = float(err.iloc[-month:].abs().mean())
    return out


def load_or_build_forecasts(dataset_name: str,
                            df_train: pd.DataFrame,
                            df_sim: pd.DataFrame,
                            H: int,
                            freq: str,
                            forecaster: "EnergyForecaster",
                            cache_dir: str | None = None,
                            refit_every_days: int | None = None,
                            kind: str = "prophet",
                            history: pd.DataFrame | None = None) -> tuple:
    """Full-day forecasts for every simulated day, fitting Prophet only on a miss.

    `H` here is the number of steps in a calendar DAY, not the control horizon:
    the table always holds a whole 24 h of consumption and generation per anchor,
    and a shorter-horizon arm simply reads fewer of them. That is what lets the
    24 h and 11 h studies share one cache and one Prophet fit.

    Returns (TableForecaster, cache_path). Gzipped CSV rather than parquet: no
    parquet engine is installed in every environment this runs in, and the cost
    of one is not worth a dependency for 17k rows per household.
    """
    cfg = {
        "dataset":    dataset_name,
        "train_from": str(df_train.index[0]),
        "train_to":   str(df_train.index[-1]),
        "sim_from":   str(df_sim.index[0]),
        "sim_to":     str(df_sim.index[-1]),
        # steps_per_day, NOT the control horizon: the cached table is a full
        # 24 h day-ahead forecast, and both the 24 h and the 11 h arm read it.
        # Keying on the horizon would fit Prophet twice for the same forecast.
        "steps_per_day": int(H),
        "freq":       freq,
        "forecaster": forecaster.config(),
        "refit_every_days": refit_every_days,
        "kind": kind,
    }
    # A fitted kind's settings are NOT in `forecaster.config()` -- the caller
    # hands this function a Prophet object it will never use, because the real
    # forecaster cannot be built until the frame is assembled below. Without
    # this, two `hbd` runs at different quantiles or lookbacks share one cache
    # entry and the second silently serves the first's forecasts. The `kind`
    # string separates hbd from hbd_baseline and nothing else.
    #
    # Constructing the forecaster is cheap; it fits nothing until asked.
    fitted_sources = [src for src in ([kind] if kind not in HYBRID_KINDS
                                      else list(HYBRID_KINDS[kind]))
                      if src in FITTED_KINDS]
    if fitted_sources:
        if history is None:
            raise ValueError(
                f"{kind!r} fits on the training block, so it needs `history=`")
        probe_frame = pd.concat([history, df_sim])
        cfg["fitted"] = {
            src: FITTED_KINDS[src](probe_frame, H, len(history)).config()
            for src in fitted_sources
        }
    # Resolved here rather than as a default argument: a module-level constant
    # bound at def time cannot be overridden by reassigning the constant, which
    # makes the cache location impossible to redirect from a notebook.
    cache_dir = FORECAST_CACHE_DIR if cache_dir is None else cache_dir
    digest = config_digest(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{dataset_name}__{digest}.csv.gz")

    if os.path.exists(path):
        table = pd.read_csv(path, parse_dates=["anchor", "ds"])
        print(f"  [Forecaster] cache hit {os.path.basename(path)} "
              f"({len(table)} rows) -- Prophet not fitted")
        served = TableForecaster(table)
        served.table = table
        return served, path

    anchors = list(df_sim.index[::H])

    def channel_table(source: str) -> pd.DataFrame:
        """One source's day-ahead table over every simulated day."""
        if source in SIMPLE_KINDS:
            # No fitting: past rows only. `history` supplies the days before the
            # simulation, so the first simulated days are copied from real data
            # rather than from themselves -- a 7-day method needs a week of it.
            frame = pd.concat([history, df_sim]) if history is not None else df_sim
            table = build_forecast_table(SIMPLE_KINDS[source](frame, H),
                                         anchors, H, freq)
            print(f"  [Forecaster] {source}: {len(table)} rows, nothing fitted")
            return table
        if source in FITTED_KINDS:
            # Same frame the simple kinds get -- history then simulation, one
            # contiguous series -- plus where the training block ends. The
            # forecaster fits on `frame.iloc[:n_train]` and reads strictly-past
            # rows after it; nothing at or beyond an anchor is ever touched.
            if history is None:
                raise ValueError(
                    f"{source!r} fits on the training block, so it needs "
                    f"`history=` -- got None")
            frame = pd.concat([history, df_sim])
            table = build_forecast_table(
                FITTED_KINDS[source](frame, H, len(history)), anchors, H, freq)
            print(f"  [Forecaster] {source}: {len(table)} rows, "
                  f"fit on {len(history)} training steps")
            return table
        if source == "truth":
            table = build_forecast_table(TruthForecaster(df_sim, H), anchors, H, freq)
            print(f"  [Forecaster] truth: {len(table)} rows, nothing fitted")
            return table
        if source not in PROPHET_KINDS:
            # Previously this fell through to Prophet, so a typo'd kind ran a
            # whole sweep and quietly reported Prophet's numbers under the new
            # name. Fail instead: a kind nobody defined is not a forecast.
            raise ValueError(
                f"unknown forecaster kind {source!r}; expected one of "
                f"{sorted(PROPHET_KINDS)}, 'truth', a simple kind "
                f"{sorted(SIMPLE_KINDS)}, a fitted kind {sorted(FITTED_KINDS)}, "
                f"or a hybrid {sorted(HYBRID_KINDS)}"
            )
        if refit_every_days:
            return build_forecast_table_refit(
                forecaster.params_con, forecaster.params_gen, df_train, df_sim,
                H, freq, refit_every_days)
        forecaster.fit(df_train, "Energy_Consumption", "Energy_Generation")
        # One anchor per simulated day, at the day's first interval -- the same
        # anchors ReactiveController._forecast_slice asks for.
        return build_forecast_table(forecaster, anchors, H, freq)

    if kind in HYBRID_KINDS:
        con_source, gen_source = HYBRID_KINDS[kind]
        table = channel_table(con_source)
        gen = channel_table(gen_source)
        # Both tables are built over the same anchors in the same order, so the
        # rows line up positionally. Checked rather than assumed: a silent
        # misalignment here would score one day's PV against another's load.
        aligned = (len(gen) == len(table)
                   and _naive(gen["ds"]).equals(_naive(table["ds"])))
        if not aligned:
            raise ValueError(
                f"{kind}: the {con_source} and {gen_source} tables do not share "
                f"an index ({len(table)} vs {len(gen)} rows)"
            )
        table = table.assign(yhat_gen=gen["yhat_gen"].to_numpy())
        print(f"  [Forecaster] {kind}: consumption from {con_source}, "
              f"generation from {gen_source}")
    else:
        table = channel_table(kind)
    # ATOMIC, for the same reason the oracle cache is: the key drops the tariff,
    # the horizon and the leak flag, so one entry serves several arms -- and
    # since the sweep was split across three notebooks those arms can be running
    # in two kernels at once. A plain `to_csv` leaves a half-written gzip at the
    # final path that the other kernel reads as a cache hit; `os.replace` means a
    # reader sees the whole file or no file. `_atomic_write` is defined below --
    # module level, resolved at call time.
    _atomic_write(path, lambda t: table.to_csv(t, index=False, compression="gzip"))
    print(f"  [Forecaster] cached {len(table)} rows -> {os.path.basename(path)}")
    served = TableForecaster(table)
    served.table = table
    return served, path


# =====================================================================
# 2c — The oracle/rules cache: what does NOT depend on the forecast
# =====================================================================
#
# Same idea as the forecast cache above, one level up. The forecast cache is
# keyed on what changes a FORECAST; this one is keyed on what changes a
# dispatch that never reads a forecast.
#
# Neither the oracle controller nor any rule-based controller looks at the
# forecast. The oracle reads `_real_slice` -- realised generation and
# consumption -- and every rule reads the meter in front of it. So for a fixed
# (household, window, battery, tariff, calendar, control horizon, solver) they
# produce the SAME answer in every arm, and the study's eleven arms contain only
# four distinct oracle workloads: AU/SI x 24 h/11 h.
#
# The other seven were being re-solved from scratch: 3.7 million LP solves, a
# third of the sweep, to recompute a number already on disk. Measured on
# AU_H24 against AU_H24_leaked, where the leak flag cannot reach the oracle:
# every cost_<rule> was bit-identical and cost_oracle differed by 3e-4 EUR,
# which is CBC picking a different vertex on a tie, not a different answer.
#
# The key drops exactly the four forecast axes and keeps everything else, so an
# arm that changes the battery, the tariff, the horizon or the solver still
# computes its own oracle rather than reading someone else's.

ORACLE_CACHE_DIR = os.environ.get(
    "ERK_ORACLE_CACHE", os.path.join(_HERE, "oracle_cache"))

# What the oracle and the rules cannot see. Everything else in `study_config`
# stays in the key.
_FORECAST_ONLY_KEYS = ("forecaster_kind", "forecaster_digest",
                       "refit_every_days", "leak_current_interval")


def oracle_config(config: dict, dataset_name: str) -> dict:
    """The part of a run's config that a forecast-blind controller can see."""
    cfg = {k: v for k, v in config.items() if k not in _FORECAST_ONLY_KEYS}
    cfg["dataset"] = dataset_name
    return cfg


def _atomic_write(path: str, write_fn) -> None:
    """Write via a temp file in the same directory, then rename.

    The sweep runs from several worker processes and two of them can reach the
    same oracle key at once (AU_H24 and AU_H24_persist are different arms and
    one cache entry). A half-written gzip that a later reader picks up as
    complete is the failure mode; `os.replace` is atomic on POSIX, so a reader
    sees either the old file or the whole new one.
    """
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_or_build_oracle(dataset_name: str, config: dict, build_fn,
                         cache_dir: str | None = None) -> tuple:
    """`(df_pk, rule_metrics, rule_rows, df_full)`, solving only on a miss.

    `build_fn()` returns that tuple and is called only when nothing on disk
    matches the key. The trajectories are stored rather than the metrics derived
    from them: `KPITracker.compare_three` and `settle_trajectory` both need the
    per-interval frame, and re-deriving them from it costs milliseconds while
    re-solving them costs a quarter of an hour.

    `df_full` is the whole-period optimum. It belongs in THIS cache and not in
    one of its own because it is forecast-blind and horizon-blind for exactly
    the same reasons the oracle arm is -- `oracle_config` already strips the
    horizon out of the key -- so one solve serves every arm of the household.
    """
    cache_dir = ORACLE_CACHE_DIR if cache_dir is None else cache_dir
    digest = config_digest(oracle_config(config, dataset_name))
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"{dataset_name}__{digest}")
    pk_path, rules_path = f"{base}__oracle.csv.gz", f"{base}__rules.json"
    full_path = f"{base}__full.csv.gz"

    if all(os.path.exists(q) for q in (pk_path, rules_path, full_path)):
        try:
            df_pk = pd.read_csv(pk_path, index_col=0, parse_dates=[0])
            df_full = pd.read_csv(full_path, index_col=0, parse_dates=[0])
            with open(rules_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            print(f"  [oracle] cache hit {os.path.basename(base)} "
                  f"({len(df_pk)} steps, {len(saved['rows'])} rules, "
                  f"whole-period solve included) -- nothing re-solved")
            return df_pk, saved["metrics"], saved["rows"], df_full
        except (ValueError, OSError, KeyError) as exc:
            # A truncated or superseded entry is recomputed, never resumed into.
            print(f"  [oracle] cache entry unreadable ({exc}); re-solving")

    df_pk, rule_metrics, rule_rows, df_full = build_fn()
    _atomic_write(pk_path, lambda t: df_pk.to_csv(t, encoding="utf-8-sig",
                                                  compression="gzip"))
    _atomic_write(full_path, lambda t: df_full.to_csv(t, encoding="utf-8-sig",
                                                      compression="gzip"))
    _atomic_write(rules_path, lambda t: json.dump(
        {"metrics": rule_metrics, "rows": rule_rows}, open(t, "w", encoding="utf-8"),
        indent=1, default=float))
    print(f"  [oracle] cached {len(df_pk)} steps + {len(rule_rows)} rules "
          f"+ whole-period solve -> {os.path.basename(base)}")
    return df_pk, rule_metrics, rule_rows, df_full


def study_config(**params) -> dict:
    """Everything that changes the ANSWER, and therefore invalidates a result.

    Distinct from the forecast key above: horizon, battery and (once they land)
    tariff and leak flag all belong here and none of them belong there.
    """
    cfg = {k: (round(v, 12) if isinstance(v, float) else v)
           for k, v in sorted(params.items())}
    # Round-trip through JSON so the value compared is exactly the value stored.
    # Without this a tuple in the config (the calendar, say) is written as a JSON
    # array and read back as a list, never equals the fresh tuple, and EVERY run
    # looks stale -- which silently disables checkpointing altogether.
    return json.loads(json.dumps(cfg, sort_keys=True, default=str))


def read_checkpoint(out_dir: str, config: dict):
    """A previous run's metrics, if it was produced under `config`."""
    path = os.path.join(out_dir, "checkpoint.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (ValueError, OSError) as exc:
        print(f"  [checkpoint] unreadable ({exc}); recomputing")
        return None
    if saved.get("config") != config:
        differing = sorted(
            k for k in set(saved.get("config", {})) | set(config)
            if saved.get("config", {}).get(k) != config.get(k)
        )
        print(f"  [checkpoint] stale, recomputing -- differs on: {', '.join(differing)}")
        return None
    return saved.get("metrics")


def write_checkpoint(out_dir: str, config: dict, metrics: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "checkpoint.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": config, "metrics": metrics}, fh, indent=2, default=str)


# =====================================================================
# 3 — MILPScheduler (the parity reference, no longer the study's solver)
# =====================================================================
#
# The hand-rolled battery this study used to solve. It is NOT what runs any more
# -- `UpstreamMILPScheduler` below solves `add_household_physics` instead, so
# the study and the shared modules cannot drift -- but it is kept, and kept
# runnable, as the reference that swap is checked against: `parity=True`
# reproduces this model to 3.6e-9 EUR over seven daily solves, which is what
# makes the previously published numbers reproducible rather than merely
# plausible. Delete it and that check goes with it.

class MILPScheduler:

    def __init__(self,
                 battery_cap:  float = 10.0,
                 soc_min_pct:  float = 0.10,
                 soc_max_pct:  float = 0.80,
                 p_max:        float = 1.5,
                 eff:          float = 0.95,
                 delta_t:      float = 0.5):
        self.battery_cap = battery_cap
        self.soc_min  = battery_cap * soc_min_pct
        self.soc_max  = battery_cap * soc_max_pct
        self.p_max    = p_max
        self.eff      = eff
        self.delta_t  = delta_t
        # AC-side limits the controller enforces per step. Symmetric here
        # because this model bounds AC power directly.
        self.max_ch_kw  = p_max
        self.max_dis_kw = p_max

    def solve(self,
              soc_init:     float,
              buy_rate:     list,
              sell_rate:    list,
              p_gen:        list,
              p_con:        list,
              terminal_soc: float | None = None) -> dict:

        H    = len(buy_rate)
        soc0 = terminal_soc if terminal_soc is not None else soc_init

        mdl = pulp.LpProblem("MILP_HEMS", pulp.LpMinimize)

        x_ch   = pulp.LpVariable.dicts("ch",   range(H), lowBound=0, upBound=self.p_max)
        x_dis  = pulp.LpVariable.dicts("dis",  range(H), lowBound=0, upBound=self.p_max)
        p_buy  = pulp.LpVariable.dicts("buy",  range(H), lowBound=0)
        p_sell = pulp.LpVariable.dicts("sell", range(H), lowBound=0)
        SoC    = pulp.LpVariable.dicts("soc",  range(H),
                                        lowBound=self.soc_min,
                                        upBound=self.soc_max)
        d_ch   = pulp.LpVariable.dicts("dch",  range(H), cat="Binary")
        d_dis  = pulp.LpVariable.dicts("ddis", range(H), cat="Binary")

        mdl += pulp.lpSum(
            p_buy[t]  * buy_rate[t]  * self.delta_t
             - p_sell[t] * sell_rate[t] * self.delta_t
        for t in range(H)
        ), "MinNetCost"

        for t in range(H):
            soc_prev = soc_init if t == 0 else SoC[t - 1]

            mdl += (p_con[t] + x_ch[t] + p_sell[t] == p_gen[t] + x_dis[t] + p_buy[t]), f"balance_{t}"
            mdl += d_ch[t] + d_dis[t] <= 1, f"mutex_{t}"
            mdl += x_ch[t]  <= self.p_max * d_ch[t],  f"ch_bound_{t}"
            mdl += x_dis[t] <= self.p_max * d_dis[t], f"dis_bound_{t}"
            mdl += (SoC[t] == soc_prev
                    + (x_ch[t] * self.eff - x_dis[t] / self.eff) * self.delta_t), f"soc_dyn_{t}"

        mdl += SoC[H - 1] >= soc0, "terminal"

        mdl.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[mdl.status]

        if status != "Optimal":
            return {
                "status":   status,
                "x_ch":     [0.0] * H, "x_dis":    [0.0] * H,
                "p_buy":    [0.0] * H, "p_sell":   [0.0] * H,
                "soc_plan": [soc_init] * H, "cost": 0.0,
            }

        return {
            "status":   status,
            "x_ch":     [pulp.value(x_ch[t])   or 0.0 for t in range(H)],
            "x_dis":    [pulp.value(x_dis[t])  or 0.0 for t in range(H)],
            "p_buy":    [pulp.value(p_buy[t])  or 0.0 for t in range(H)],
            "p_sell":   [pulp.value(p_sell[t]) or 0.0 for t in range(H)],
            "soc_plan": [pulp.value(SoC[t])    or 0.0 for t in range(H)],
            "cost":     pulp.value(mdl.objective) or 0.0,
        }


# =====================================================================
# 3b — Upstream physics adapter (Energy_Community.MILP_Household)
# =====================================================================
#
# Same interface as MILPScheduler, but the constraints come from
# `add_household_physics`, so this study and the community/horizon studies solve
# one battery model instead of two that drift. What arrives with it:
# `floor_export_rates` against the unbounded-LP failure, PV curtailment, the
# metering bounds, an explicit terminal SoC, and the option to drop the binaries
# and solve a pure LP.
#
# Two frames have to be reconciled at this boundary, and both were bugs waiting
# to happen if left implicit:
#
#   units  upstream works in kWh PER INTERVAL, this notebook in kW. gen/con are
#          multiplied by delta_t on the way in, actions divided by it on the way
#          out. (Mixing those two conventions is exactly what F1 was.)
#   SoC    upstream's soc runs 0..usable_capacity; this notebook's runs
#          soc_min..soc_max in absolute kWh. The offset is soc_min.
#
# `parity=True` reproduces the OLD model exactly, so the swap can be validated
# before any of the new capability is switched on:
#   - AC-side charge and discharge limits are both p_max (upstream instead caps
#     the change in STORED energy, which makes them asymmetric by eff**2),
#   - curtailment is forced off,
#   - metering bounds are off,
#   - the charge/discharge binaries are kept.

import sys as _sys


def _find_repo_root(start: str | None = None) -> str:
    """Walk up until the shared modules are in sight.

    Mirrors Data_Loader._find_workspace_root rather than assuming a fixed depth:
    `os.getcwd() + "/.."` is only correct when the cwd happens to be Main/, and
    silently resolves somewhere else when the notebook is run from the repo root,
    from a test harness, or from Colab.
    """
    markers = ("MILP_Household.py", "Environment.py", "Basic_Functions.py")
    here = os.path.abspath(start or os.getcwd())
    for _ in range(6):
        if all(os.path.exists(os.path.join(here, m)) for m in markers):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    raise RuntimeError(
        "Energy_Community modules not found above "
        f"{os.path.abspath(start or os.getcwd())}; set ERK_REPO_ROOT."
    )


_REPO_ROOT = os.environ.get("ERK_REPO_ROOT") or _find_repo_root()
# Only the repo root. "New pricing functions" deliberately stays off sys.path:
# it holds its own Pricing_Functions.py, which would shadow the root shim that
# re-exports the si_* surface Environment.py imports. MILP_Household appends
# that directory itself, which is why si_cas is imported after it below.
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from Data_Loader import load_smp_data   # noqa: E402
from MILP_Household import (            # noqa: E402
    add_battery_exclusivity,
    add_endogenous_agreed_power,
    add_excess_power_ratchet,
    add_household_physics,
    agreed_power_by_month,
    build_household_env,
    floor_export_rates,
    full_period_solver,
    interval_rate_vectors,
    month_calendar,
    step_energy_kwh,
)
import si_cas as _si_cas                # noqa: E402  (after MILP_Household)
from si_obracun import Pravila          # noqa: E402  (the SI ratchet's rule set)
from Basic_Functions import (           # noqa: E402
    battery_delta,
    max_charge_now,
    max_discharge_now,
    pv_surplus,
)


def build_study_env(sample: pd.DataFrame,
                    battery_cap: float,
                    soc_min_pct: float,
                    soc_max_pct: float,
                    p_max: float,
                    eff: float,
                    delta_t: float,
                    H: int,
                    max_daily_cycles: float | None = None,
                    cycle_cost_eur_per_efc: float | None = None):
    """The battery, sized by upstream's own factory rather than by hand.

    c_rate/inverter are chosen so `step_energy_kwh` lands on this study's p_max:
    a 10 kWh pack at 1.5 kW is C=0.15, well under upstream's residential 0.5
    default, and that difference is a study choice rather than an accident.
    """
    c_rate = p_max / float(battery_cap)
    return build_household_env(
        sample,
        capacity_kwh=battery_cap,
        scheme="si_samooskrba",          # unused: pricing stays with TariffCalculator
        paket_id="GENI_SAMO_DINAMICNI",
        pricing_reference_year=2024,
        peak_reset_months=1,
        price_column="SMP",
        generation_column="Energy_Generation",
        consumption_column="Energy_Consumption",
        steps_per_day=H,
        charge_efficiency=eff,
        discharge_efficiency=eff,
        c_rate=c_rate,
        inverter_max_kw=p_max,
        soc_min_frac=soc_min_pct,
        soc_max_frac=soc_max_pct,
        max_daily_cycles=max_daily_cycles,
        cycle_cost_eur_per_efc=cycle_cost_eur_per_efc,
    )


def align_envelope(env, p_max, eff, delta_t):
    """Put the environment on the study's AC-symmetric battery, in place.

    Upstream caps the change in STORED energy; this study's `p_max` caps AC
    power. The two differ by eff**2 -- a 0.75 kWh step is 1.579 kW of charge but
    1.425 kW of discharge -- so an environment left at its own default gives the
    rule-based controllers, which read `env.max_charge_kwh`, a 5.3 % larger
    charge rating and a 5.0 % smaller discharge rating than the MILP's own
    +-p_max bounds. That is a comparison between two different batteries, and it
    was live in every published figure.

    `UpstreamMILPScheduler(parity=True)` used to do this as a side effect of
    being constructed, which made a fair comparison depend on the order two
    objects happened to be built in. Called here, once, on the environment every
    controller in the arm shares, it is a stated property of the study instead.

    Returns `env`, so it can wrap a `build_study_env` call.
    """
    step_kwh = float(p_max) * float(delta_t)
    env.max_charge_kwh = step_kwh * float(eff)
    env.max_discharge_kwh = step_kwh / float(eff)
    return env


def wear_objective_terms(env, blk, n_steps):
    """The battery-wear shadow price, as objective terms. Empty when unpriced.

    Mirrors `MILP_Household.solve_household` exactly -- `cycle_cost_eur_per_efc`
    is what ONE equivalent full cycle costs, one EFC is `2 * nominal` kWh
    through the store, so the per-stored-kWh price is that over the divisor --
    and exists so the receding-horizon solve and the whole-period solve cannot
    price wear two different ways.

    It was missing from the MPC path entirely: `cycle_cost_eur_per_efc` was set
    on the environment, recorded in the run config and described in a comment as
    a shadow price "in its objective", while `UpstreamMILPScheduler.solve` built
    an objective out of `buy` and `sell` alone and never read it. Turning the
    setting on changed the checkpoint key and nothing else.

    Consequence worth keeping in view: with this on the objective is no longer
    the reported bill. `settle_trajectory` still prices the executed trajectory
    without it, and `summarize` reports `wear_eur` beside the bill, so the two
    stay separable.
    """
    rate_efc = getattr(env, "cycle_cost_eur_per_efc", None)
    nominal = float(getattr(env, "nominal_capacity_kwh", env.battery_capacity_kwh))
    if not rate_efc or blk.soc is None or nominal <= 0:
        return []
    per_stored_kwh = float(rate_efc) / (2.0 * nominal)
    return [
        per_stored_kwh * (blk.charge[t] * env.charge_efficiency
                          + blk.discharge[t] / env.discharge_efficiency)
        for t in range(n_steps)
    ]


class UpstreamMILPScheduler:
    """MILPScheduler's interface over add_household_physics."""

    def __init__(self, env, battery_cap: float, soc_min_pct: float,
                 soc_max_pct: float, p_max: float, eff: float, delta_t: float,
                 parity: bool = False, exclusivity: str = "binary",
                 allow_spill: bool = True, metering_bounds: bool = True,
                 solver=None):
        self.env      = env
        self.delta_t  = delta_t
        self.eff      = eff
        self.p_max    = p_max
        self.soc_min  = battery_cap * soc_min_pct
        self.soc_max  = battery_cap * soc_max_pct
        self.parity   = parity
        self.solver   = solver or make_solver()

        if parity:
            # Reproduce the old hand-rolled model exactly: symmetric +-p_max AC
            # bounds, no curtailment, no metering bounds, binaries kept.
            #
            # The envelope itself is `align_envelope`'s job, not this
            # constructor's. Setting it here mutated an environment the
            # rule-based controllers also read, so whether the comparison was
            # fair depended on which object happened to be built first. Checked
            # instead, so a caller that forgets the call is told rather than
            # silently scored against a different battery.
            step_kwh = p_max * delta_t
            want = (step_kwh * eff, step_kwh / eff)
            got = (float(env.max_charge_kwh), float(env.max_discharge_kwh))
            if max(abs(a - b) for a, b in zip(want, got)) > 1e-9:
                raise ValueError(
                    f"parity=True needs an environment on the AC-symmetric "
                    f"envelope: expected max_charge/discharge_kwh "
                    f"{want[0]:.6f}/{want[1]:.6f}, got {got[0]:.6f}/{got[1]:.6f}. "
                    f"Wrap the build in align_envelope(env, p_max, eff, delta_t)."
                )
            self.exclusivity, self.allow_spill, self.metering_bounds = \
                "binary", False, False
        else:
            if exclusivity not in ("binary", "inverter", "auto"):
                raise ValueError(
                    f"exclusivity must be 'binary', 'inverter' or 'auto', "
                    f"got {exclusivity!r}")
            self.exclusivity      = exclusivity
            self.allow_spill      = allow_spill
            self.metering_bounds  = metering_bounds

        # Published in kW for the controller's per-step check, and read AFTER
        # the parity shim above may have rewritten them. Upstream bounds the
        # change in STORED energy, so outside parity these are asymmetric by
        # eff**2: a 0.75 kWh step is 1.579 kW of charge but 1.425 kW of
        # discharge. `p_max` alone is the old model's concept and no longer
        # describes what the battery may do.
        self.max_ch_kw  = env.max_charge_kwh / eff / delta_t
        self.max_dis_kw = env.max_discharge_kwh * eff / delta_t

        # How often the "auto" branch had to fall back to the binaries. Reported
        # rather than assumed: if a price series makes this fire on most solves,
        # the LP is buying nothing and the run should know.
        self.n_solves = 0
        self.n_binary_solves = 0

    def solve(self, soc_init: float, buy_rate: list, sell_rate: list,
              p_gen: list, p_con: list, terminal_soc: float | None = None) -> dict:
        H  = len(buy_rate)
        dt = self.delta_t

        # kW -> kWh per interval, and into upstream's 0..usable SoC frame.
        gen = [g * dt for g in p_gen]
        con = [c * dt for c in p_con]
        soc0 = float(np.clip(soc_init - self.soc_min, 0.0,
                             self.env.battery_capacity_kwh))
        soc_end = soc0 if terminal_soc is None else float(np.clip(
            terminal_soc - self.soc_min, 0.0, self.env.battery_capacity_kwh))

        # The guard against the unbounded LP: where the delivered import rate is
        # negative, an unfloored export credit makes the buy/sell round trip
        # profitable without limit. Flooring makes it exactly neutral.
        export, n_floored = floor_export_rates(buy_rate, sell_rate)

        # Which exclusivity THIS horizon needs. Two different exploits live here
        # and only one of them is closed by the flooring above.
        #
        #   buy/sell     buy at a negative delivered rate, sell the same energy
        #                for a credit. `floor_export_rates` holds the credit at
        #                or below the import rate, so the round trip is exactly
        #                neutral. Closed, always.
        #
        #   charge/dis   charge x and discharge x*eff**2 in the SAME interval.
        #                The stored energy is unchanged and the round-trip loss
        #                x*(1 - eff**2) -- 9.75 % at eff 0.95 -- is drawn from
        #                the grid and thrown away. Where the delivered import
        #                rate is NEGATIVE the household is paid to draw it, so
        #                destroying energy is profitable and an LP will do it.
        #                `charge + discharge <= inverter rating` BOUNDS the pair
        #                but does not forbid it; only the binaries do.
        #
        # Measured, on a horizon forced to -0.10 EUR/kWh: the binary model does
        # both-at-once in 0 of 48 intervals, the pure LP in 33, and books an
        # extra 0.063 EUR of "saving" that is entirely destroyed energy.
        #
        # So the LP is used where it is provably equivalent and the binaries
        # where they are load-bearing, decided per solve on the only thing that
        # decides it. On this study's price series that costs almost nothing --
        # Ausgrid EA025 never goes negative and the SI series does so in 1
        # interval of 17,568 -- while making the guarantee structural rather
        # than a property of the data that happens to be loaded.
        exclusivity = self.exclusivity
        if exclusivity == "auto":
            exclusivity = "binary" if min(buy_rate) <= 0.0 else "inverter"
            self.n_binary_solves += exclusivity == "binary"
            self.n_solves += 1

        prob = pulp.LpProblem("MILP_HEMS", pulp.LpMinimize)
        blk = add_household_physics(
            prob, self.env, n_steps=H, gen=gen, con=con,
            initial_soc_kwh=soc0, final_soc_kwh=soc_end,
            exclusivity=exclusivity,
            metering_bounds=self.metering_bounds,
        )
        if not self.allow_spill:
            for t in range(H):
                prob += blk.spill[t] == 0, f"nospill_{t}"

        prob += pulp.lpSum(
            blk.buy[t] * buy_rate[t] - blk.sell[t] * export[t] for t in range(H)
        ) + pulp.lpSum(wear_objective_terms(self.env, blk, H)), "MinNetCost"

        prob.solve(self.solver)
        status = pulp.LpStatus[prob.status]

        if status != "Optimal":
            # Loud, and never mistaken for a valid do-nothing plan.
            print(f"  !!! solver returned {status} for a {H}-step horizon "
                  f"(soc_init={soc_init:.3f}); holding the battery idle")
            return {"status": status,
                    "x_ch": [0.0] * H, "x_dis": [0.0] * H,
                    "p_buy": [0.0] * H, "p_sell": [0.0] * H,
                    "soc_plan": [soc_init] * H, "cost": 0.0,
                    "n_floored": n_floored}

        val = lambda v: float(pulp.value(v) or 0.0)
        return {
            "status":   status,
            "x_ch":     [val(blk.charge[t]) / dt for t in range(H)],
            "x_dis":    [val(blk.discharge[t]) / dt for t in range(H)],
            "p_buy":    [val(blk.buy[t]) / dt for t in range(H)],
            "p_sell":   [val(blk.sell[t]) / dt for t in range(H)],
            # soc[t+1] is the SoC at the END of step t, which is what the old
            # model's SoC[t] meant; shifted back into absolute kWh.
            "soc_plan": [val(blk.soc[t + 1]) + self.soc_min for t in range(H)],
            "spill":    [val(blk.spill[t]) / dt for t in range(H)],
            "cost":     float(pulp.value(prob.objective) or 0.0),
            "n_floored": n_floored,
        }


# =====================================================================
# 3d — The whole-period MILP: the theoretical optimum
# =====================================================================
#
# Every other controller in this study is a HEURISTIC about the future: the
# rules read a clock or a trailing quantile, and the MILP arms re-solve a 24 h
# or 11 h window and commit one interval of the answer. None of them can be
# compared to "the best a battery could have done" without that number
# existing, and until now it did not -- `gain_share_pct` divided by the
# receding-horizon `oracle`, which is perfect foresight WITHIN 24 h and no more.
# The notebook said so in its own caveats.
#
# This is that number: one solve, the whole scored year, perfect foresight
# throughout. It is not a controller anyone could deploy and is never proposed
# as one. It is the denominator.
#
# It is assembled from the same upstream pieces `MILP_Household.solve_household`
# uses rather than from a second copy of the model, for the reason the rest of
# this file keeps repeating: two models of one battery drift, and the drift
# lands in the comparison. What differs from `solve_household` is only what
# HAS to: the rate vectors come from this study's `build_rate_vectors` so the
# AU arm can be priced at all (upstream's are SI-only), and the exclusivity is
# chosen the way `UpstreamMILPScheduler` chooses it rather than always binary.
#
# WHAT THE OBJECTIVE CONTAINS, and why it is the whole invoice:
#
#   energy        buy x import - sell x floored export.
#   excess power  SI only. The presezna-moc charge on the metered import, over
#                 the agreed line. Without it this is NOT a lower bound: the
#                 peak-shaving rules earn their money here, and a "ceiling"
#                 they beat is not a ceiling. This is the concrete reason the
#                 notebook's old caveat -- "the MILP optimises energy only ...
#                 which is why a peak-shaving rule can beat it there" -- does
#                 not apply to this solve.
#   fixed         SI only, and not optional. The dogovorjena moc is ENDOGENOUS
#                 (`agreed_power_from_dispatch`), so the network power charge is
#                 linear in a variable this solve sets. That variable needs its
#                 positive objective coefficient or the k-largest construction
#                 in `add_endogenous_agreed_power` is not tight -- the contract
#                 would float up for free. So the solve minimises the invoice,
#                 not the part of it `Cost_EUR` reports.
#   wear          the same shadow price every other MILP here now carries.
#
# The consequence to keep in view: on SI the guarantee is over the TOTAL --
# energy + power + fixed + wear -- and not over `Cost_EUR` alone, because the
# fixed charge is part of the same decision and cannot be held constant while
# the rest is optimised. On AU there is no capacity charge and the fixed charge
# is a true constant, so there the bound holds on `Cost_EUR` directly.
# `full_period_bound_check` is what actually verifies this per run, rather than
# leaving it as a claim in a comment.


def solve_full_period(env, rates, tariff, n_steps, soc_init_kwh, delta_t,
                      soc_min_kwh, terminal_soc_kwh=None, solver=None,
                      closeout_rate=None, verbose=True):
    """One MILP over the whole scored window. Perfect foresight, no horizon.

    Returns the same shape `UpstreamMILPScheduler.solve` returns, in kW, plus
    the objective breakdown, so the caller can settle it through the arm's one
    evaluator exactly as it settles the receding-horizon arms.

    On SI the contract is normally decided INSIDE the LP, which is exact. Where
    the horizon is too short for that -- fewer months than the contract lag, so
    no month in it reads its line from another month in it -- the solve cannot
    see the feedback at all while `settle_trajectory` still rolls the contract
    onto the trajectory afterwards, and a solve optimised against one contract
    and billed under another is not a bound. Measured on a 10-day slice, it came
    back 0.79 EUR ABOVE `tariff_arbitrage`. So that case converges whole solves
    to the fixed point instead, exactly as `MILP_Household.solve_household` does
    when the contract has no in-LP form. It costs a handful of extra solves and
    only ever fires on a horizon too short for the study's own arms.
    """
    if _agreed_power_is_endogenous_in_lp(env, n_steps) is False and getattr(
            env, "agreed_power_from_dispatch", False):
        hours = float(delta_t)
        out = {}

        def _dispatch():
            out.update(_solve_full_period_once(
                env, rates, tariff, n_steps, soc_init_kwh, delta_t, soc_min_kwh,
                terminal_soc_kwh, solver, closeout_rate, verbose))
            net = (np.asarray(out["p_buy"]) - np.asarray(out["p_sell"])) * hours
            return out, np.maximum(net, 0.0) / hours

        result, info = rbc.converge_agreed_power(env, _dispatch)
        result["agreed_power_iterations"] = info["iterations"]
        result["agreed_power_converged"] = info["converged"]
        return result
    return _solve_full_period_once(
        env, rates, tariff, n_steps, soc_init_kwh, delta_t, soc_min_kwh,
        terminal_soc_kwh, solver, closeout_rate, verbose)


def _agreed_power_is_endogenous_in_lp(env, n_steps) -> bool:
    """Can this horizon decide its own contract inside one solve?

    Only if a month in it reads its line from another month in it, which needs
    more months than the lag. Same test `_solve_full_period_once` applies; it
    lives here so the wrapper can ask it before building anything.
    """
    if not getattr(env, "agreed_power_from_dispatch", False):
        return False
    lag = getattr(env, "agreed_power_lag_months", None)
    if not lag:
        return False
    _, _, months_sorted, _ = month_calendar(env.dataset.index[:int(n_steps)])
    return len(months_sorted) > int(lag)


def _solve_full_period_once(env, rates, tariff, n_steps, soc_init_kwh, delta_t,
                            soc_min_kwh, terminal_soc_kwh=None, solver=None,
                            closeout_rate=None, verbose=True):
    """One whole-period solve, under the contract currently in force."""
    H_steps = int(n_steps)
    dt = float(delta_t)
    hours = dt

    # kWh per interval -- the environment's own frame, so unlike the receding
    # scheduler there is no kW round trip to get wrong here.
    gen = [float(v) for v in env.arr_generation[:H_steps]]
    con = [float(v) for v in env.arr_consumption[:H_steps]]

    soc0 = float(np.clip(soc_init_kwh - soc_min_kwh, 0.0,
                         env.battery_capacity_kwh))
    # The terminal SoC is left FREE and priced instead, which is not a detail.
    #
    # Pinning it to the opening charge is the obvious move and it is wrong: no
    # other controller here is held to it. `settle_trajectory` closes the year
    # for all of them the same way -- whatever the pack is short at the end is
    # valued at the mean delivered import rate and added to the bill -- so a
    # controller is free to end empty if it can beat that price, and the
    # receding-horizon oracle does exactly that: it ends at soc_min and pays the
    # close-out. Measured before this was fixed, over 14 days on Ausgrid 104,
    # the pinned whole-period solve came out 0.34 EUR ABOVE the oracle it is
    # supposed to bound, entirely because it was made to carry 4 kWh home.
    #
    # So the objective carries the same close-out the evaluator will charge,
    # `(soc_start - soc_end) / eta_ch * mean import rate`, which is linear in
    # the final SoC. The solve then minimises `Cost_EUR_Closed` itself rather
    # than a near neighbour of it, and the bound holds by construction.
    soc_end = None if terminal_soc_kwh is None else float(np.clip(
        terminal_soc_kwh - soc_min_kwh, 0.0, env.battery_capacity_kwh))

    import_rates = np.asarray(rates[0][:H_steps], dtype=float)
    export_rates = np.asarray(rates[1][:H_steps], dtype=float)
    export, n_floored = floor_export_rates(import_rates, export_rates)

    dates = env.dataset.index[:H_steps]
    interval_minutes = int(round(env.interval_minutes))

    prob = pulp.LpProblem("MILP_HEMS_full_period", pulp.LpMinimize)
    blk = add_household_physics(
        prob, env, n_steps=H_steps, gen=gen, con=con,
        initial_soc_kwh=soc0, final_soc_kwh=soc_end,
        exclusivity="inverter",          # see below; binaries added where they bind
        metering_bounds=True,
    )
    # No curtailment, matching the receding-horizon arms.
    for t in range(H_steps):
        prob += blk.spill[t] == 0, f"nospill_{t}"

    # Exclusivity, spent only where it does work. `charge + discharge <= rating`
    # bounds the pair everywhere; it only fails to FORBID the pair where the
    # delivered import rate is negative, because there destroying energy pays.
    # A year of binaries is 17,520 of them and a genuinely hard MIP; a year of
    # this is 1 on SI and 0 on AU. Same guarantee, LP speed.
    negative = [t for t in range(H_steps) if import_rates[t] <= 0.0]
    for t in negative:
        flag = pulp.LpVariable(f"B_charging_{t}", cat="Binary")
        add_battery_exclusivity(
            prob, charge_t=blk.charge[t], discharge_t=blk.discharge[t],
            max_charge_ac=blk.max_charge_ac, max_discharge_ac=blk.max_discharge_ac,
            flag=flag)

    energy_terms = [blk.buy[t] * import_rates[t] - blk.sell[t] * export[t]
                    for t in range(H_steps)]
    # The close-out, exactly as `settle_trajectory` computes it -- and `exactly`
    # is doing work here. The evaluator prices the terminal shortfall at the mean
    # of `sig.import_rate`, so this objective has to use the same number or the
    # solve optimises a near neighbour of the bill instead of the bill. When the
    # two disagreed -- `sig.import_rate` was Slovenian on both arms before
    # `build_signals` took the arm's own rates -- the solve was told that ending
    # low was ~2x dearer than the evaluator would charge, carried 4.2 kWh it did
    # not need, and came out ABOVE the oracle it is meant to bound. They agree
    # now; the argument stays so that they cannot silently stop agreeing.
    mean_rate = (float(np.mean(import_rates)) if closeout_rate is None
                 else float(closeout_rate))
    terminal_terms = []
    if blk.soc is not None:
        terminal_terms = [
            (soc0 - blk.soc[H_steps]) / float(env.charge_efficiency) * mean_rate
        ]
    wear_terms = wear_objective_terms(env, blk, H_steps)
    peak_terms, agreed_terms, agreed_vars = [], [], None

    if tariff == "SI":
        pricing_options = dict(env.pricing_options or {})
        pricing_options.setdefault("pricing_reference_year",
                                   env.pricing_reference_year)
        _, _, months_sorted, month_idx_t = month_calendar(dates)
        block_arr = env.tariff_blocks[:H_steps]
        agreed_by_month = agreed_power_by_month(env, 0, months_sorted)

        # The contract the solve sets for itself, where the horizon is long
        # enough to contain a month whose line is read from one inside it. A
        # whole year always is; the guard mirrors solve_household's.
        if (getattr(env, "agreed_power_from_dispatch", False)
                and getattr(env, "agreed_power_lag_months", None)
                and len(months_sorted) > int(env.agreed_power_lag_months)):
            agreed_vars, agreed_terms = add_endogenous_agreed_power(
                prob, env, buy=blk.buy, hours=hours,
                months_sorted=months_sorted, month_idx_t=month_idx_t,
                blocks=block_arr, dates=dates,
                interval_minutes=interval_minutes,
                pricing_options=pricing_options,
                agreed_by_month=agreed_by_month, n_steps=H_steps,
            )
        _, _, peak_terms = add_excess_power_ratchet(
            prob, env, blk.buy,
            blocks=block_arr, month_idx_t=month_idx_t,
            months_sorted=months_sorted, agreed_by_month=agreed_by_month,
            agreed_vars=agreed_vars, start_idx=0, n_steps=H_steps,
            hours=hours, pravila=Pravila.za_leto(
                int(pricing_options["pricing_reference_year"])),
        )

    prob += (pulp.lpSum(energy_terms) + pulp.lpSum(terminal_terms)
             + pulp.lpSum(wear_terms) + pulp.lpSum(peak_terms)
             + pulp.lpSum(agreed_terms)), "MinInvoice"

    if solver is None:
        # An LP once the binaries are counted, so the study's exact solver; a
        # real MIP only if a price series is negative often enough to matter,
        # and then the whole-period gap upstream already settled on.
        solver = make_solver() if len(negative) <= 8 else full_period_solver()
    if verbose:
        print(f"  [full period] {H_steps} steps, {len(negative)} binary "
              f"interval(s), {n_floored} export rate(s) floored; solving...")
    t0 = datetime.datetime.now()
    prob.solve(solver)
    elapsed = (datetime.datetime.now() - t0).total_seconds()
    status = pulp.LpStatus[prob.status]

    if status != "Optimal":
        # Never silently a do-nothing plan: this number is the denominator every
        # other controller is scored against, and a failed solve that reads as
        # "the optimum is to idle" would make every controller look good.
        raise RuntimeError(
            f"the whole-period solve returned {status} after {elapsed:.0f} s "
            f"({H_steps} steps, {len(negative)} binaries). This is the study's "
            f"denominator; it cannot be defaulted."
        )

    val = lambda v: float(pulp.value(v) or 0.0)
    if verbose:
        print(f"  [full period] {status} in {elapsed:.0f} s, "
              f"objective {float(pulp.value(prob.objective) or 0.0):.2f}")
    return {
        "status": status,
        "x_ch":  [val(blk.charge[t]) / dt for t in range(H_steps)],
        "x_dis": [val(blk.discharge[t]) / dt for t in range(H_steps)],
        "p_buy": [val(blk.buy[t]) / dt for t in range(H_steps)],
        "p_sell": [val(blk.sell[t]) / dt for t in range(H_steps)],
        "soc_plan": [val(blk.soc[t + 1]) + soc_min_kwh for t in range(H_steps)],
        "objective": float(pulp.value(prob.objective) or 0.0),
        "wear_eur": float(sum(pulp.value(x) or 0.0 for x in wear_terms)),
        "n_binary_intervals": len(negative),
        "n_floored": n_floored,
        "runtime_s": elapsed,
    }


# =====================================================================
# 3c — Tariffs as rate vectors
# =====================================================================
#
# Both tariffs are reduced to the same triple upstream's `interval_rate_vectors`
# returns -- (import_rates, export_rates, constant_costs), all EUR per kWh
# except the last -- so the physics, the controller and the evaluator never
# learn which tariff they are running under, and the two are genuinely
# comparable rather than two separate pipelines.

SI_PAKET_ID = "GENI_SAMO_DINAMICNI"     # GEN-I "Dinamicni", samooskrba


def au_rate_vectors(index, smp_eur_per_kwh, interval_minutes: int = 30) -> tuple:
    """Ausgrid EA025 time-of-use, on Australia/Sydney local time.

    Rates are AUD per kWh: the SMP column arrives in EUR/kWh and is divided by
    0.615 on the way in. The SI arm's are EUR per kWh. Neither is USD, which is
    what every column in this frame used to be labelled, so the label is gone
    rather than made wrong in a new way -- an arm's currency is a property of
    its tariff and travels with the arm.
    """
    buy, sell = TariffCalculator.rates_series(
        pd.Series(list(smp_eur_per_kwh), index=index))
    const = TariffCalculator.constant_cost_per_interval(interval_minutes)
    return buy.tolist(), sell.tolist(), [const] * len(index)


def si_rate_vectors(env, index, smp_eur_per_kwh, interval_minutes: int = 30,
                    paket_id: str = SI_PAKET_ID) -> tuple:
    """GEN-I Dinamicni under si_samooskrba, via upstream's own rate builder.

    `meritve_15min=True` is passed explicitly. Left unset it auto-resolves to
    False on a 30-minute interval, which by the rules drops a 4-tariff AKTIVNI
    list onto its flat substitute rate. It is a no-op for DINAMICNI -- measured
    identical for True/False/None -- but the intent here is the one the study
    wants stated: apply the same tariff structure the 15-minute rules describe,
    accepting the resolution the data has.
    """
    return interval_rate_vectors(
        env, list(index), list(smp_eur_per_kwh),
        {"paket_id": paket_id, "meritve_15min": True},
        interval_minutes,
    )


def build_rate_vectors(tariff: str, env, index, smp_eur_per_kwh,
                       interval_minutes: int = 30) -> tuple:
    if tariff == "AU":
        return au_rate_vectors(index, smp_eur_per_kwh, interval_minutes)
    if tariff == "SI":
        return si_rate_vectors(env, index, smp_eur_per_kwh, interval_minutes)
    raise ValueError(f"unknown tariff {tariff!r}; expected 'AU' or 'SI'")


# =====================================================================
# 3d — (was: hand-rolled baselines)
# =====================================================================
#
# SelfConsumptionScheduler, TariffArbitrageScheduler, BASELINE_SCHEDULERS and
# run_baseline lived here. They are gone rather than kept "just in case": a
# second implementation of a controller is exactly how the MILP ended up solving
# a different battery from the rules it was compared against (see
# align_envelope). Their replacements:
#
#   SelfConsumptionScheduler   Rule_Based_Control.SelfConsumption, the same rule,
#                              executed and priced by the same runner as the
#                              other eight
#   TariffArbitrageScheduler   TariffArbitrage in section 3e, ported to a Policy
#                              so nothing is lost -- it is the arm that showed a
#                              forecast-free rule beating MILP+Prophet
#   run_baseline               run_rules in section 3e


class _NaiveForecaster:
    """Shared plumbing for the fit-free baselines: no model, only past rows.

    Causality is the whole contract. A subclass says WHICH past rows it wants,
    through `_source_rows`, and never gets to choose a row at or after the
    anchor: `_anchor_index` resolves the anchor and every subclass computes its
    offsets backwards from it. `test_naive_forecasters` holds all of them to
    that by poisoning the frame from the anchor onwards with NaN and checking
    the forecast does not move.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int):
        self.frame = frame
        self.spd = steps_per_day

    def fit(self, *a, **k):
        return self

    def _anchor_index(self, anchor_ts) -> int:
        i = self.frame.index.get_indexer([anchor_ts])[0]
        if i < 0:
            raise KeyError(
                f"anchor {anchor_ts} is not in the {self.config()['kind']} frame")
        return i

    def _source_rows(self, i: int, horizon_steps: int) -> tuple:
        """-> (consumption, generation) arrays of `horizon_steps` values."""
        raise NotImplementedError

    def _column(self, col: str) -> np.ndarray:
        """The frame's column as an array, converted once per forecaster."""
        cache = self.__dict__.setdefault("_cols", {})
        if col not in cache:
            cache[col] = self.frame[col].to_numpy()
        return cache[col]

    def predict_next_day(self, anchor_ts, horizon_steps: int = 48,
                         freq: str = "30min") -> pd.DataFrame:
        i = self._anchor_index(anchor_ts)
        idx = self.frame.index[i:i + horizon_steps]
        con, gen = self._source_rows(i, horizon_steps)
        return pd.DataFrame({
            "ds": _naive(idx),
            # Clipped for the same reason EnergyForecaster clips: the contract
            # downstream is a non-negative kW profile. On copied real data this
            # is a no-op, which is the point -- it cannot be the thing that
            # differs between a naive kind and Prophet.
            "yhat_con": np.clip(con[:len(idx)], 0.0, None),
            "yhat_gen": np.clip(gen[:len(idx)], 0.0, None),
        })


class SeasonalNaiveForecaster(_NaiveForecaster):
    """The same interval `lag_days` ago. Yesterday at lag 1, last week at lag 7.

    Causal by construction: the day it copies ends `lag_days` full days before
    the anchor it is asked about. `history` supplies the days before the
    simulation starts, so the first simulated days are forecast from real data
    rather than from themselves.

    Lag 1 is the standard baseline any forecaster must beat, and the one
    `forecast_error_metrics` scores skill against. Lag 7 is the other trivial
    answer and a different bet: it is a week stale on the weather, but it never
    predicts a Monday from a Sunday. Which bet wins is a per-channel question --
    that is what `forecast_benchmark` is for.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int,
                 lag_days: int = 1):
        super().__init__(frame, steps_per_day)
        if lag_days < 1:
            raise ValueError(f"lag_days must be >= 1, got {lag_days}")
        self.lag_days = int(lag_days)

    def config(self) -> dict:
        return {"kind": f"naive{self.lag_days}d", "steps_per_day": self.spd,
                "lag_days": self.lag_days}

    def _source_rows(self, i: int, horizon_steps: int) -> tuple:
        j = i - self.lag_days * self.spd
        if j < 0:
            raise ValueError(
                f"no day {self.lag_days} before index {i}; this forecaster "
                f"needs at least {self.lag_days} day(s) of history ahead of "
                f"the simulation"
            )
        prev = self.frame.iloc[j:j + horizon_steps]
        return (prev["Energy_Consumption"].to_numpy(),
                prev["Energy_Generation"].to_numpy())


# Kept so the name that names the concept still resolves: "persistence" is
# seasonal-naive at lag 1, and the study's arms, caches and comments all say
# persistence.
PersistenceForecaster = SeasonalNaiveForecaster


class DayTypeNaiveForecaster(_NaiveForecaster):
    """The most recent day of the SAME TYPE -- weekday from weekday, weekend
    from weekend.

    Between the two trivial answers: 1 to 3 days stale rather than 7, but it
    never forecasts a Monday from a Sunday. A matching day is always found
    within 7, because 7 days back is the same weekday, so the walk terminates.

    A "day" here is the block of `spd` rows the study already partitions on
    (`df_sim.index[::H]`), and that index is UTC. So the weekday label is the
    UTC one, not the Ausgrid local one, and the classification can be a few
    hours off at the boundary. That is the convention every other day-boundary
    assumption in this module uses -- consistent, not newly wrong -- and the
    lags it picks are whole multiples of a day either way.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int,
                 weekend_days: tuple = (5, 6)):
        super().__init__(frame, steps_per_day)
        self.weekend_days = tuple(weekend_days)

    def config(self) -> dict:
        return {"kind": "daytype", "steps_per_day": self.spd,
                "weekend_days": list(self.weekend_days)}

    def _is_weekend(self, i: int) -> bool:
        return self.frame.index[i].dayofweek in self.weekend_days

    def _source_rows(self, i: int, horizon_steps: int) -> tuple:
        want = self._is_weekend(i)
        for back in range(1, 8):
            j = i - back * self.spd
            if j < 0:
                break
            if self._is_weekend(j) == want:
                prev = self.frame.iloc[j:j + horizon_steps]
                return (prev["Energy_Consumption"].to_numpy(),
                        prev["Energy_Generation"].to_numpy())
        raise ValueError(
            f"no matching {'weekend' if want else 'weekday'} day within 7 days "
            f"before index {i}; this forecaster needs a week of history ahead "
            f"of the simulation"
        )


class ClimatologyForecaster(_NaiveForecaster):
    """The last `window_days` days at this interval, averaged.

    One day of history is one noisy draw from the weather. Averaging several
    keeps the shape -- the sunrise, the evening peak -- and drops the day's
    particular cloud, which is the part no fit-free method can know anyway.

    `stat="median"` is the interesting one on generation: it rejects a single
    overcast day outright, where the mean lets it drag the whole profile down.
    On consumption the mean is usually the better of the two, because load
    noise is closer to symmetric. Both are reported by `forecast_benchmark`
    rather than argued about here.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int,
                 window_days: int = 7, stat: str = "mean"):
        super().__init__(frame, steps_per_day)
        if window_days < 1:
            raise ValueError(f"window_days must be >= 1, got {window_days}")
        if stat not in ("mean", "median"):
            raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
        self.window_days = int(window_days)
        self.stat = stat

    def config(self) -> dict:
        return {"kind": f"{self.stat}{self.window_days}",
                "steps_per_day": self.spd,
                "window_days": self.window_days, "stat": self.stat}

    def _source_rows(self, i: int, horizon_steps: int) -> tuple:
        # Each past day contributes a whole `spd`-row day, aligned by position
        # within the day, so interval k of the forecast averages interval k of
        # each of the last `window_days` days. Anchors sit at day starts, so
        # position within the day IS the offset from the anchor.
        if horizon_steps > self.spd:
            raise ValueError(
                f"climatology is a day-shaped average: horizon_steps "
                f"({horizon_steps}) cannot exceed steps_per_day ({self.spd})"
            )
        first = i - self.window_days * self.spd
        if first < 0:
            raise ValueError(
                f"only {i // self.spd} day(s) of history before index {i}; "
                f"this forecaster needs {self.window_days}"
            )
        reduce = np.mean if self.stat == "mean" else np.median
        out = []
        for col in ("Energy_Consumption", "Energy_Generation"):
            # Hoisted: `frame[col].to_numpy()` copies the whole column, and
            # inside the comprehension that was one copy per day per anchor --
            # 14 copies of a three-year column to read 14 slices of 48.
            series = self._column(col)
            days = np.stack([series[i - d * self.spd:
                                    i - d * self.spd + horizon_steps]
                             for d in range(1, self.window_days + 1)])
            out.append(reduce(days, axis=0))
        return out[0], out[1]



class HbdForecaster(_NaiveForecaster):
    """Seasonal baseline plus residual AR, from cvxgrp/home-battery-dispatch.

    See `hbd_forecast` for the method and for every place it departs from the
    reference. This class is the study's adapter: it holds the fitted
    parameters, decides when to refit, and serves `predict_next_day`.

    It sits on `_NaiveForecaster` for one reason -- `_anchor_index` and the
    strictly-past discipline that goes with it. That makes it the first
    forecaster here that BOTH fits a model and reads recent actuals. Prophet
    fits on an expanding window but reads nothing at prediction time, so within
    a refit block its answer for tomorrow 08:00 does not depend on what happened
    at 07:00 today; the naive kinds read the last rows but fit nothing. The AR
    stage is exactly the missing term, and `use_ar=False` is the ablation that
    prices it (the paper's own `baseline_only` sensitivity).

    `frame` is history + simulation concatenated, exactly as SIMPLE_KINDS get
    it, and `n_train` is how many leading rows of it are the training block.
    Everything at or after `n_train` is simulation and is never fit on: a fit
    for the block starting at step s uses `frame.iloc[:s]` only.
    """

    # The AR is fit for one day of horizon and one day of lookback. Beyond L the
    # baseline stands alone, which is what lets a horizon longer than a day be
    # served at all -- see `predict_next_day`.
    LOOKBACK_DAYS = 1

    def __init__(self, frame: pd.DataFrame, steps_per_day: int, n_train: int,
                 eta_con: float = 0.5, eta_gen: float = 0.5,
                 lambd: float = hbd.LAMBDA, n_harmonics: int = hbd.N_HARMONICS,
                 use_ar: bool = True, baseline_kind: str = "fourier",
                 climatology_days: int = 14, refit_every_days: int | None = None,
                 ar_refit_every_days: int | None = None,
                 max_ar_samples: int | None = 12000,
                 lookback_days: int = LOOKBACK_DAYS):
        super().__init__(frame, steps_per_day)
        if n_train < 2 * steps_per_day:
            raise ValueError(
                f"n_train={n_train} is under two days; the baseline has an "
                f"annual harmonic and needs a real training block")
        self.n_train = int(n_train)
        # Both channels default to the MEDIAN, which is not what the paper uses
        # for load (eta=0.2). Deliberate: read `hbd_forecast._pinball` for the
        # sign convention -- eta=0.2 there is the 80th percentile, a baseline
        # biased high, chosen because their tariff punishes walking into a peak
        # tier. Every other method in this study's roster is a median-ish
        # estimator, so importing that bias into the headline arm would confound
        # "the method is better" with "the forecast is biased", and the roster
        # would no longer be like-for-like. The bias is a separate question with
        # a separate experiment: the quantile sweep varies this knob on purpose.
        self.eta = {"Energy_Consumption": float(eta_con),
                    "Energy_Generation": float(eta_gen)}
        self.lambd = float(lambd)
        self.n_harmonics = int(n_harmonics)
        self.use_ar = bool(use_ar)
        # Which unconditional predictor stage 2 corrects. "fourier" is the
        # paper's own and the faithful port. "climatology" is not in the paper:
        # it swaps in the study's own best fit-free method (a `climatology_days`
        # median at the same clock position) as stage 1.
        #
        # The reason it exists: the reference's transferable idea is the
        # DECOMPOSITION -- an unconditional predictor plus a FITTED residual AR
        # -- not the Fourier basis specifically. Measured on this data the
        # Fourier stage is the weak half: on 5 households it scores +0.015 skill
        # on consumption alone, against median14's +0.199, because a household's
        # level is set by the last fortnight and not by where it sits in the
        # year. The AR then adds +0.124 on top of it and still lands short. So
        # the obvious experiment is the same AR on the stronger stage 1, and
        # this switch is what runs it.
        if baseline_kind not in ("fourier", "climatology"):
            raise ValueError(
                f"baseline_kind must be 'fourier' or 'climatology', "
                f"got {baseline_kind!r}")
        self.baseline_kind = baseline_kind
        self.climatology_days = int(climatology_days)
        # Measured, not assumed. On Ausgrid 1 over the full simulation year,
        # refitting the seasonal stage on Prophet's 30-day expanding-window
        # cadence scored con skill +0.140 / gen +0.071 in 308 s; fitting it once
        # on the training block scored +0.139 / +0.070 in 216 s. The cadence
        # buys nothing here and costs 40 % more, so the default is the paper's:
        # fit once. Prophet needs the refit because its trend term extrapolates;
        # a Fourier baseline has no trend to drift.
        self.refit_every_days = refit_every_days
        # None means "fit the AR once and keep it", which is what the paper
        # does. It is a cost decision, not a modelling one: the baseline is ONE
        # small QP, the AR is L of them, so refitting both on Prophet's 30-day
        # cadence costs ~12x the AR fit for a second-order gain. Both cadences
        # stay reachable so the expensive one remains an ablation.
        self.ar_refit_every_days = ar_refit_every_days
        # Also measured. The AR windows overlap at stride 1, so 35k of them
        # carry far less information than 35k independent samples: subsampling
        # to 12k scored con skill +0.140 / gen +0.071 against +0.138 / +0.072
        # on all of them -- indistinguishable -- in 308 s against 1097 s. None
        # uses every window, for anyone who would rather have the 3.6x.
        self.max_ar_samples = max_ar_samples
        self.M = int(lookback_days) * steps_per_day
        self.L = steps_per_day
        # Two caches, not one, because the two stages refit on different
        # cadences and sharing a key makes the expensive one follow the cheap
        # one. See `_ar`.
        self._baselines: dict = {}
        self._ars: dict = {}

    def config(self) -> dict:
        return {
            "kind": "hbd" if self.use_ar else "hbd_baseline",
            # Bumped whenever the METHOD changes in a way the other fields do
            # not describe. The cache digest is built from this dict, so without
            # it a bug fix leaves every table fitted by the broken code in place
            # and silently serves it -- which is exactly what happened to the
            # climatology baseline's causality fix, and the wrong numbers looked
            # entirely plausible.
            #   2: `_climatology_at` took a per-step causality wall. Before it,
            #      in-sample residuals were measured against a baseline built
            #      from t's FUTURE, so the AR was fit to correct residuals it
            #      never meets at prediction time.
            "version": 2,
            "steps_per_day": self.spd,
            "baseline_kind": self.baseline_kind,
            "climatology_days": self.climatology_days,
            "n_train": self.n_train,
            "eta_con": self.eta["Energy_Consumption"],
            "eta_gen": self.eta["Energy_Generation"],
            "lambd": self.lambd,
            "n_harmonics": self.n_harmonics,
            "use_ar": self.use_ar,
            "refit_every_days": self.refit_every_days,
            "ar_refit_every_days": self.ar_refit_every_days,
            "max_ar_samples": self.max_ar_samples,
            "M": self.M,
            "L": self.L,
        }

    # -- fitting -----------------------------------------------------------
    def _fit_end(self, i: int, every_days: int | None) -> int:
        """How many leading rows a forecast at index `i` may be fit on.

        Rounded DOWN to a refit boundary so every anchor inside one block shares
        one fit -- otherwise the cache is useless and every day refits. `None`
        pins it to the training block, i.e. never refit.
        """
        if not every_days:
            return self.n_train
        block = int(every_days) * self.spd
        return self.n_train + ((i - self.n_train) // block) * block

    def _baseline(self, col: str, end: int) -> dict:
        """Baseline for `col` fit on `frame.iloc[:end]`, cached per end."""
        key = (col, end)
        if key not in self._baselines:
            y = self._column(col)[:end]
            theta = self._cached_fit(
                self._param_path(col, end, "baseline"),
                lambda: hbd.train_baseline(
                    y, self.spd, eta=self.eta[col], lambd=self.lambd,
                    n_harmonics=self.n_harmonics, t0=0))
            self._baselines[key] = {"theta": theta}
        return self._baselines[key]

    def _climatology_at(self, col: str, t: np.ndarray,
                        before: int | None) -> np.ndarray:
        """A `climatology_days` median at the same clock position, for steps `t`.

        `before` is the causality wall: every row this reads is strictly before
        it. The naive rule -- go back d whole days for d in 1..N -- is causal
        only while the horizon stays inside one day, which is exactly the
        assumption `ClimatologyForecaster` encodes by refusing a horizon longer
        than a day. Here the source is walked back in whole days UNTIL it clears
        the wall, so any horizon works and a 30-day forecast tiles the last
        fortnight.

        `before=None` means each step is its own wall, which is what fitting
        wants: the residual at t must be measured against the baseline t would
        have been given, using only what preceded t. Passing the end of the
        training block instead makes `t - before` negative for every in-sample
        row and walks the source FORWARD -- the baseline would then be fit
        against values from t's future. In-block rather than a simulation leak,
        but still wrong: the AR would learn to correct residuals that are not
        the ones it meets at prediction time.
        """
        series = self._column(col)
        spd, n = self.spd, self.climatology_days
        wall = t if before is None else np.full(t.shape, before)
        # Whole days back from t until the source clears the wall. For the
        # in-sample and one-day-horizon cases this resolves to exactly t - spd.
        base = t - ((t - wall) // spd + 1) * spd
        idx = base[:, None] - np.arange(n)[None, :] * spd
        if idx.min() < 0:
            raise ValueError(
                f"a {n}-day climatology baseline needs {n} days of history "
                f"before step {int(t.min())}; only "
                f"{int(base.min()) // spd + 1} available")
        if (idx >= wall[:, None]).any():
            raise AssertionError(
                "climatology baseline read at or after its causality wall")
        return np.median(series[idx], axis=1)

    def _baseline_at(self, col: str, t: np.ndarray, end: int,
                     before: int | None) -> np.ndarray:
        """Stage 1 evaluated at absolute step indices `t`."""
        if self.baseline_kind == "climatology":
            return self._climatology_at(col, t, before)
        return hbd.predict_baseline(
            t, self._baseline(col, end)["theta"], self.spd, self.n_harmonics)

    # Fitted parameters are cached to disk as well as in memory. The forecast
    # cache one level up only stores a COMPLETED table, so a run interrupted
    # part-way through a household throws away every fit it had done; at ~150 s
    # per channel that is the difference between a sweep that resumes and one
    # that restarts. Keyed on the config plus a digest of the exact training
    # slice, so it cannot serve one household's parameters to another and does
    # not need the dataset name plumbed in to say so.
    PARAM_CACHE_DIR = os.environ.get(
        "ERK_HBD_PARAM_CACHE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hbd_params"))

    def _param_path(self, col: str, end: int, stage: str) -> str:
        y = self._column(col)[:end]
        key = config_digest({
            "config": self.config(), "col": col, "end": int(end),
            "stage": stage,
            # The data itself, not just its length: two households share every
            # other field.
            "data": hashlib.sha256(np.ascontiguousarray(y)).hexdigest(),
        })
        os.makedirs(self.PARAM_CACHE_DIR, exist_ok=True)
        return os.path.join(self.PARAM_CACHE_DIR, f"{stage}__{key}.npy")

    @staticmethod
    def _cached_fit(path: str, build):
        """`build()`, memoised on disk, written atomically."""
        if os.path.exists(path):
            return np.load(path)
        value = build()
        # .npy suffix required -- see the note in hbd_forecast.train_ar_model.
        tmp = f"{path}.{os.getpid()}.tmp.npy"
        np.save(tmp, value)
        os.replace(tmp, path)          # the sweep is multi-process
        return value

    def _ar(self, col: str, ar_end: int) -> np.ndarray | None:
        """Gamma for `col`, cached per AR-refit block.

        Keyed on `ar_end` ALONE, and fit against the baseline of the same
        `ar_end` rather than whichever baseline the calling anchor is using.
        The two cadences are independent on purpose -- the baseline is one small
        QP and the AR is L of them -- and keying Gamma on the baseline's block
        instead would silently refit the expensive stage every time the cheap
        one moved, which is exactly the default configuration.
        """
        if not self.use_ar:
            return None
        key = (col, ar_end)
        if key not in self._ars:
            y = self._column(col)[:ar_end]
            # The climatology baseline needs a fortnight of history before it
            # can be evaluated, so the residual series starts there rather than
            # at zero. The Fourier one is defined everywhere.
            start = (self.climatology_days * self.spd
                     if self.baseline_kind == "climatology" else 0)
            t = np.arange(start, ar_end)
            resid = y[start:] - self._baseline_at(col, t, ar_end, None)
            whole = self._param_path(col, ar_end, "ar")
            # Per-column checkpoints beside the whole-matrix one, so an
            # interrupted fit resumes mid-channel rather than restarting it.
            stem = whole[:-len(".npy")]
            self._ars[key] = self._cached_fit(
                whole,
                lambda: hbd.train_ar_model(
                    resid, self.M, self.L, eta=self.eta[col], lambd=self.lambd,
                    max_samples=self.max_ar_samples,
                    column_cache=lambda j: f"{stem}__col{j:03d}.npy"))
        return self._ars[key]

    def _range(self, col: str, end: int) -> tuple:
        """The clip range: what the data up to `end` actually occupied.

        Read off the series rather than off a fitted baseline, so it is defined
        for both stage-1 kinds. The reference clips to `data.min()/max()` over
        the WHOLE series, which reads the window it is forecasting.
        """
        y = self._column(col)[:end]
        return float(np.min(y)), float(np.max(y))

    # -- prediction --------------------------------------------------------
    def _source_rows(self, i: int, horizon_steps: int) -> tuple:
        need = self.M + (self.climatology_days * self.spd
                         if self.baseline_kind == "climatology" else 0)
        if i < need:
            raise ValueError(
                f"only {i} row(s) before index {i}; this forecaster needs "
                f"{need} ({self.M} of lookback"
                + (f" plus {self.climatology_days} days of climatology)"
                   if self.baseline_kind == "climatology" else ")"))
        end = self._fit_end(i, self.refit_every_days)
        ar_end = self._fit_end(i, self.ar_refit_every_days)

        out = []
        for col in ("Energy_Consumption", "Energy_Generation"):
            series = self._column(col)
            # Strictly past: [i - M, i). The anchor interval itself is NOT read
            # -- the reference seeds its vector with the realised value at the
            # anchor, which here is the leak `leak_current_interval` measures.
            # `i` is passed as the causality wall for both windows, so stage 1
            # cannot reach the anchor either.
            past = series[i - self.M:i]
            past_bl = self._baseline_at(col, np.arange(i - self.M, i), end, i)
            fut_bl = self._baseline_at(
                col, np.arange(i, i + horizon_steps), end, i)
            lo, hi = self._range(col, end)
            out.append(hbd.compose_forecast(
                past, past_bl, fut_bl, self._ar(col, ar_end), lo, hi))

        return out[0], out[1]

    def predict_next_day(self, anchor_ts, horizon_steps: int = 48,
                         freq: str = "30min") -> pd.DataFrame:
        """As `_NaiveForecaster`, but able to outrun the frame's own index.

        The base class labels its output with `frame.index[i:i + horizon_steps]`,
        which silently TRUNCATES a horizon that runs past the end of the data.
        That is right for a method that can only copy rows it has; this one is a
        function of the clock and can be evaluated arbitrarily far ahead, so the
        stamps are generated instead. Nothing else changes.
        """
        i = self._anchor_index(anchor_ts)
        idx = pd.date_range(start=self.frame.index[i], periods=horizon_steps,
                            freq=freq)
        con, gen = self._source_rows(i, horizon_steps)
        return pd.DataFrame({
            "ds": _naive(idx),
            "yhat_con": np.clip(con, 0.0, None),
            "yhat_gen": np.clip(gen, 0.0, None),
        })


# =====================================================================
# 3e — One settlement, and the rule-based roster
# =====================================================================
#
# The load-bearing rule of the comparison: a rule and the MILP differ ONLY in
# how a setpoint is chosen. Same envelope, same rate vectors, same evaluator.
# The evaluator is the part that is easy to get wrong, because the rules arrive
# with one of their own (`Rule_Based_Control.price_interval`, the SI ratchet
# walk) and the MILP arm has always had another (buy x rate - sell x rate). Run
# both and whatever separates two controllers includes the difference between
# two settlements, which is not a result about control at all.
#
# So: one `settle` callable per arm, matching price_interval's signature, used
# by the rules AND applied to the MILP's executed trajectory.

import Rule_Based_Control as rbc                                  # noqa: E402


def make_au_settlement(rates, delta_t):
    """Ausgrid EA025: the delivered rate vectors, and no capacity charge.

    Returned rather than written inline so it has price_interval's exact
    signature and can be handed to `run_policy(settle=...)` unchanged. The peak
    state passes through untouched: EA025 as modelled here bills energy and a
    standing charge, so there is no running peak to carry and `Power_EUR` is 0
    for every controller on this tariff by construction, not by accident.
    """
    buy, sell, const = (np.asarray(x, dtype=float) for x in rates)

    def settle(env, idx, net_kwh, peak_state):
        imported = max(float(net_kwh), 0.0)
        exported = max(-float(net_kwh), 0.0)
        energy = imported * buy[idx] - exported * sell[idx]
        return energy, energy, 0.0, float(const[idx]), peak_state

    return settle


class TariffArbitrage(rbc.Policy):
    """Self-consumption, plus arbitrage against the PUBLISHED tariff.

    Carried over from the study's own `TariffArbitrageScheduler`, re-expressed
    as a Policy so it is executed and priced by the same runner as the other
    eight instead of by a parallel code path. It is kept because it is the arm
    that isolates what the study is really measuring: on Ausgrid 127 over 14
    days it came in at 26.36 EUR against MILP+Prophet's 27.27: a forecast-free
    rule beating the forecast. If that holds up, the oracle's advantage is
    time-of-use arbitrage rather than better PV capture, and attributing it to
    forecast quality is the misreading this rule exists to prevent.

    It reads the tariff over the day -- published, not forecast -- and the
    generation and consumption of the interval in front of it, which a meter
    measures. It reads no forecast of either and solves no LP.
    """

    name = "tariff_arbitrage"
    label = "Tariff arbitrage"

    CHEAP_Q, PEAK_Q = 0.33, 0.67

    def __init__(self, respect_peak=True):
        self.respect_peak = respect_peak

    def reset(self, sig):
        # Thresholds from the day's OWN published rates: SIPX and the AEMO
        # pre-dispatch both publish the day ahead, so ranking today's intervals
        # is something a controller genuinely has. Quantiles rather than a
        # hard-coded clock, so one policy runs unchanged under either tariff.
        self._cheap = np.empty(sig.n_steps, dtype=float)
        self._peak = np.empty(sig.n_steps, dtype=float)
        for steps in sig.day_steps:
            if not steps:
                continue
            lo, hi = steps[0], steps[-1] + 1
            rates = sig.import_rate[lo:hi]
            self._cheap[lo:hi] = float(np.quantile(rates, self.CHEAP_Q))
            self._peak[lo:hi] = float(np.quantile(rates, self.PEAK_Q))

    def setpoint(self, sig, idx, soc_kwh, lo, hi, peak_state):
        rates = sig.import_rate
        day = next((s for s in sig.day_steps if s and s[0] <= idx <= s[-1]), None)
        end = (day[-1] + 1) if day else sig.n_steps
        ahead = rates[idx + 1:end]

        # The reserve is what makes this more than a rule with a clock: energy
        # is held back from a cheap interval's deficit while priced intervals
        # are still ahead, and the amount held is what the battery could
        # physically deliver into them. Battery spec and tariff table only --
        # never a load forecast.
        n_peak_ahead = int((ahead >= self._peak[idx]).sum())
        reserve = float(min(sig.capacity_kwh, n_peak_ahead * sig.max_discharge_ac))
        is_peak = rates[idx] >= self._peak[idx]
        is_cheap = rates[idx] <= self._cheap[idx]
        # Round-trip guard: never buy unless something ahead clears the purchase
        # after BOTH conversion losses.
        worth_it = bool(ahead.size) and float(ahead.max()) * sig.eta_rt > rates[idx]

        surplus, deficit = sig.surplus[idx], sig.deficit[idx]
        room = rbc._grid_charge_room(sig, idx, hi, peak_state, self.respect_peak)

        if surplus > 1e-9:
            if is_cheap and worth_it:
                return min(surplus + max(reserve - soc_kwh, 0.0), room)
            return min(surplus, hi)
        if deficit > 1e-9:
            if is_peak:
                return -min(deficit, -lo)
            if is_cheap and worth_it and soc_kwh < reserve:
                return min(reserve - soc_kwh, room)
            spare = max(soc_kwh - reserve, 0.0)
            return -min(deficit, -lo, spare * sig.eta_dis)
        if is_cheap and worth_it and soc_kwh < reserve:
            return min(reserve - soc_kwh, room)
        return 0.0


def run_rules(env, settle, tariff, signals=None, only=None, n_steps=None,
              soc_init_kwh=None, rates=None):
    """Every rule this tariff is compared against, executed and priced.

    Returns `(metrics, rows)`: metrics keyed `cost_<rule>` for the checkpoint,
    and one row per controller carrying the full set of columns.

    `no_battery` is run here too, as an idle policy, rather than taken from
    KPITracker: it then goes through the same settlement, the same terminal-SOC
    treatment and the same endogenous contract as everything it is the reference
    for. A reference priced by a different function is not a reference.
    """
    if signals is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signals = rbc.build_signals(env, n_steps=n_steps, rates=rates)

    roster = [rbc._Idle()] + rule_roster(tariff)
    if only is not None:
        roster = [pol for pol in roster if pol.name in set(only)]

    metrics, rows = {}, []
    for pol in roster:
        out = rbc.run_policy(env, pol, signals=signals, settle=settle,
                             soc_init_kwh=soc_init_kwh)
        metrics[f"cost_{pol.name}"] = out["Cost_EUR_Closed"]
        metrics[f"efc_{pol.name}"] = out["Equivalent_Full_Cycles"]
        # The standing charge, per rule. Kept because on SI it is NOT decision
        # independent -- the dogovorjena moc is endogenous, so a peak shaver
        # walks itself onto a smaller contract and a cheaper fixed charge -- and
        # `full_period_bound_check` compares totals that include it. It used to
        # be written only for the MILP arms, which made every rule look 60-70
        # EUR cheaper than the optimum on a total that silently omitted it.
        metrics[f"fixed_{pol.name}"] = out["Fixed_EUR"]
        rows.append({"controller": pol.name, "label": pol.label,
                     "causal": pol.causal, **{
                         k: v for k, v in out.items() if not k.startswith("_")}})
    return metrics, rows


def build_settlement(tariff, env, rates, delta_t):
    """The one evaluator this arm prices every controller through."""
    if tariff == "AU":
        return make_au_settlement(rates, delta_t)
    if tariff == "SI":
        # The full SI walk: a running per-block peak dropped on each ratchet
        # window, priced against the dogovorjena moc in force -- which is itself
        # rolled from the peaks this controller realized the month before.
        return rbc.price_interval
    raise ValueError(f"unknown tariff {tariff!r}; expected 'AU' or 'SI'")


# Which rules run on which tariff. The two peak-shaving rules earn their money
# from a capacity charge; Ausgrid EA025 as modelled has none, so on AU they can
# only spend round-trip losses and including them would be a rigged comparison.
# On SI they are the point.
RULES_BY_TARIFF = {
    "AU": ["self_consumption", "fixed_schedule", "delayed_pv_charge",
           "price_threshold", "price_rank_daily", "tariff_arbitrage",
           "price_oracle"],
    "SI": [n for n in rbc.POLICY_ORDER if n != "price_oracle"]
          + ["tariff_arbitrage", "price_oracle"],
}
# `respect_peak` caps grid charging at the agreed power so a rule does not buy
# its arbitrage twice. On AU there is no excess-power charge to avoid, and the
# environment's agreed power is an SI artefact of build_study_env -- letting it
# throttle an Ausgrid controller would be charging it for a contract it is not
# on. Off there, on under SI.
_PEAK_AWARE_ARGS = {
    "fixed_schedule": "respect_peak", "price_threshold": "respect_peak",
    "price_rank_daily": "respect_peak", "price_oracle": "respect_peak",
    "peak_shaving": "ratchet_aware", "self_consumption_peak_shaving": "ratchet_aware",
    "tariff_arbitrage": "respect_peak",
}


# The clock rule's windows, per tariff, in LOCAL hours. A clock rule answers a
# tariff's shape, and the two tariffs have different shapes, so one pair of
# windows for both was never going to be right for either.
#
# These are SEARCHED, not asserted: a (start x length) sweep over starts 0-23 and
# lengths 3/4/5/6 h, four households, a full simulated year, scored on saving
# NET OF WEAR because a window that buys 20 EUR with 60 extra cycles is not an
# improvement. The search is `scratchpad/tune.py` and is re-runnable.
#
# AU  charge 11-15, discharge 15-21. Top of 96 windows at 187.38 EUR/a net of
#     wear; the runner-up 10-15 is 186.03. It is exactly what EA025's shape
#     nominates -- charge in the 0.0270 solar sponge (10:00-15:00), discharge
#     into the 0.2360 peak (15:00-21:00) -- which is the reassurance that the
#     search found structure and not four households' noise.
#
#     The OLD windows, 01-05 / 18-22, score 61.78 on the same measure. They were
#     not chosen badly; they were chosen against the broken clock, where
#     Ljubljana 01:00-05:00 landed on the Sydney solar sponge by accident. Fix
#     the clock and they price the pack's charge at the flat overnight rate and
#     empty it before the peak begins.
#
# SI  charge 13-16, discharge 16-20. Best of a 40-window shortlist crossing the
#     windows the block schedule and the PV profile nominate -- and it still
#     LOSES 4.79 EUR/a net of wear. That is not a failure to tune it. Over the
#     whole shortlist, two households, a full year, saving net of wear:
#
#         13-16 / 16-20   -4.79    the best window there is
#         01-05 / 18-22  -31.37    what it used to do
#         self-consumption            +12.94
#         self-consumption + shaving  +35.64
#
#     The reason is structural. On GEN-I Dinamicni the delivered energy rate
#     spans 0.074-0.088 EUR/kWh across the whole day, so a clock has almost no
#     spread to arbitrage, while the money on SI is the capacity charge -- which
#     a rule that grid-charges on a schedule cannot help and can only hurt. It
#     buys a few cents of energy with 115 equivalent full cycles a year.
#
#     Which is exactly what this controller is in the study to show: "on a flat
#     list it is the whole of what a battery can do, and on a dynamic one it is
#     what ignoring the price signal costs". On SI that cost is negative, and
#     the honest thing is to report it rather than to keep searching for a
#     window that makes a clock look good on a tariff that does not reward one.
FIXED_SCHEDULE_WINDOWS = {
    "AU": {"charge_hours": (11.0, 15.0), "discharge_hours": (15.0, 21.0)},
    "SI": {"charge_hours": (13.0, 16.0), "discharge_hours": (16.0, 20.0)},
}

# Every other tunable rule, per tariff. Same principle as the windows above and
# the same reason: a quantile is a statement about a price DISTRIBUTION, and the
# two tariffs do not have the same one. Ausgrid EA025 is three network steps
# (0.0270 / 0.0720 / 0.2360) on a spot base -- a wide, strongly trimodal spread.
# GEN-I Dinamicni as it lands on these profiles spans 0.074-0.088 EUR/kWh, which
# is nearly flat; the money there is the capacity charge, not the energy price.
# One pair of quantiles cannot describe both.
#
# Searched by `tune_rules.py`, scored on saving net of wear and, on SI,
# including the standing charge, because the contract is endogenous there.
# Re-searched from scratch once `build_signals` began taking the arm's own
# rates: every quantile in this study had previously been fitted, implicitly,
# against a Slovenian price series on both arms.
#
# HOW THE WINNER IS PICKED. Highest mean saving net of wear, among settings at
# which the rule is still trading -- `tune_rules.MIN_TRADED_SHARE`. The floor is
# there because every one of these rules degenerates into `self_consumption` at
# the quiet end of its own parameter (a `max_share` under 1/48 makes
# `int(max_share * 48) == 0` and the day's plan comes back empty), and a roster
# of four columns of self-consumption under four names cannot answer the
# question the roster exists for. On the corrected signal the floor never binds:
# every winner trades on ~11 % of intervals, well clear of it.
#
# WORTH KNOWING, because it nearly went into the study as a finding: tuned
# against the OLD signal these searches ran to the quiet end of every grid and
# said that price arbitrage does not cover its own cycle wear. That was the bug
# talking. The Slovenian series spans 0.074-0.088 EUR/kWh on these profiles, so
# no trade in it clears a 0.417 EUR cycle; EA025 spans 0.12-0.35, and the same
# rules then want to trade an order of magnitude more and earn 2.6x as much
# (price_threshold 70.96 -> 184.31 EUR/a). A parameter fitted to the wrong
# distribution does not merely land in the wrong place, it inverts the
# conclusion.
RULE_PARAMS = {
    "AU": {
        # The top of this surface is a plateau -- every quantile pair from
        # 0.35/0.65 to 0.50/0.50 lands within 1.8 EUR/a of the best, and the top
        # six within 0.3 -- so the argmax is reported but the choice is
        # insensitive. What is NOT insensitive is the tight end: 0.02/0.98
        # scores 70.96 against 184.31, because on a tariff whose dear block runs
        # six hours a day a rule that only acts on the extreme 2 % sits out the
        # whole of the signal.
        "price_threshold":  {"window_days": 7, "q_low": 0.45, "q_high": 0.55},
        "price_oracle":     {"window_days": 3, "q_low": 0.20, "q_high": 0.80},
        # The class default, and it saturates: above 0.25 the pack's own fill
        # time binds first, so 0.33 and 0.50 return the identical trajectory.
        "price_rank_daily": {"max_share": 0.25},
    },
    "SI": {
        # GEN-I Dinamicni lands on these profiles at 0.074-0.088 EUR/kWh -- an
        # 0.014 spread against a 0.417 EUR cycle. There is no arbitrage here to
        # find, and the search says so: the best price_threshold on the whole
        # 44-point grid earns 0.25 EUR/a, and price_rank_daily's unconstrained
        # optimum is to stop trading (12.94 EUR/a, 0.00 % of intervals). What is
        # recorded is the least-trading setting at which each is still a rule,
        # so the roster keeps a price arm that is honestly worth nothing rather
        # than a fourth copy of self-consumption.
        "price_threshold":  {"window_days": 14, "q_low": 0.02, "q_high": 0.98},
        "price_oracle":     {"window_days": 30, "q_low": 0.02, "q_high": 0.98},
        "price_rank_daily": {"max_share": 0.04},
        # And this is where SI's money actually is: the capacity charge. The two
        # shaving rules earn 33-36 EUR/a against the price rules' 0-3, and every
        # household in the search gains. Note both want a 14-day peak window
        # rather than the 30-day default -- the ratchet resets monthly, so a
        # threshold read over 30 days is still describing the peak the household
        # has already stopped paying for.
        "peak_shaving": {"q_peak": 0.95, "margin": 1.2, "window_days": 14},
        "self_consumption_peak_shaving": {"q_peak": 0.99, "margin": 0.8,
                                          "reserve_cap_frac": 0.5},
    },
}


def rule_roster(tariff):
    """The controllers this tariff is compared against, already configured."""
    respect = tariff == "SI"
    out = []
    for name in RULES_BY_TARIFF[tariff]:
        arg = _PEAK_AWARE_ARGS.get(name)
        kw = {arg: respect} if arg else {}
        kw.update(RULE_PARAMS.get(tariff, {}).get(name, {}))
        if name == "tariff_arbitrage":
            pol = TariffArbitrage(**kw)
            pol.name = name
        elif name == "fixed_schedule":
            pol = rbc.make_policy(name, **kw, **FIXED_SCHEDULE_WINDOWS[tariff])
        else:
            pol = rbc.make_policy(name, **kw)
        out.append(pol)
    return out


def settle_trajectory(env, net_kwh, settle, sig, soc_start=None, soc_end=None):
    """Price an executed trajectory through the arm's evaluator.

    The MILP half of "one evaluator". `ReactiveController.run` decides a
    dispatch and reports what it did; the cost it reports along the way is a
    running convenience, not the bill. This walks the realized net draw through
    exactly the settlement the rules are priced by -- same peak state, same
    window drops, same standing charge -- so `Cost_EUR` means one thing across
    the whole results frame.

    Including the CONTRACT, which is the half that was missing. On SI the
    dogovorjena moc is endogenous: `run_policy` converges every rule onto the
    line its own peaks agree to, and a rule that shaves therefore earns twice --
    once on the excess charge and again on a smaller standing charge. The MILP
    trajectories were priced against whatever contract happened to be left on
    `env`, so all three came back with an identical 68.05 EUR standing charge
    while the peak shavers were billed 61.90 for the same 60 days. That is two
    evaluators wearing one name, and it made the whole-period optimum -- whose
    entire advantage on SI is that it can buy a cheaper contract -- look 1.2 EUR
    WORSE than a rule it strictly dominates.

    So the same fixed point is run here. The dispatch is already decided, so the
    loop is not iterating a controller: it converges the contract onto a fixed
    meter trace, which `converge_agreed_power` settles in one or two passes.
    """
    if getattr(env, "agreed_power_from_dispatch", False):
        hours = sig.hours
        trace = np.maximum(np.asarray(net_kwh, dtype=float), 0.0) / hours

        def _dispatch():
            return _settle_trajectory_once(
                env, net_kwh, settle, sig, soc_start, soc_end), trace

        out, info = rbc.converge_agreed_power(env, _dispatch)
        out["Agreed_Power_Iters"] = info["iterations"]
        out["Agreed_Power_Converged"] = info["converged"]
        return out
    return _settle_trajectory_once(env, net_kwh, settle, sig, soc_start, soc_end)


def _settle_trajectory_once(env, net_kwh, settle, sig, soc_start=None,
                            soc_end=None):
    """One pass of `settle_trajectory`, under the contract currently in force."""
    peak_state = {b: 0.0 for b in rbc._BLOCKS}
    cost = energy = power = fixed = 0.0
    peak_kw = 0.0
    for idx in range(len(net_kwh)):
        peak_state = rbc._drop_peak_on_window_start(peak_state, sig.windows, idx)
        step, e, p_, f, peak_state = settle(env, idx, float(net_kwh[idx]), peak_state)
        cost += step
        energy += e
        power += p_
        fixed += f
        peak_kw = max(peak_kw, float(net_kwh[idx]) / sig.hours)
    # The same close-out the rules get. A controller that ends the year with a
    # flatter pack than it started has spent stored energy it was given, and
    # booking that as a saving is free money; valuing the shortfall at the mean
    # delivered import rate closes it. The MILP's per-solve terminal constraint
    # makes this small, which is the point -- it should be small, and it should be
    # measured rather than assumed.
    adj = 0.0
    if soc_start is not None and soc_end is not None:
        mean_rate = float(np.mean(sig.import_rate[:len(net_kwh)]))
        adj = (float(soc_start) - float(soc_end)) / sig.eta_ch * mean_rate
    return {"Cost_EUR": cost, "Cost_EUR_Closed": cost + adj,
            "Terminal_SOC_Adj_EUR": adj, "Energy_EUR": energy, "Power_EUR": power,
            "Fixed_EUR": fixed, "Peak_Import_kW": peak_kw}


# =====================================================================
# 4 — ReactiveController
# =====================================================================

# Feasibility tolerances for the per-step invariants in ReactiveController.run.
# CBC returns vertices to ~1e-9; these sit well above solver noise and well
# below anything that would move a bill.
SOC_TOL    = 1e-6   # kWh
ACTION_TOL = 1e-6   # kW

class ReactiveController:

    def __init__(self,
                 scheduler:               MILPScheduler,
                 forecaster:              EnergyForecaster,
                 real_data:               pd.DataFrame,
                 soc_init:                float = 10.0,
                 horizon_steps:           int   = 48,
                 reoptimize_every:        int   = 1,
                 freq:                    str   = "30min",
                 steps_per_day:           int   = 48,
                 rate_vectors:            tuple | None = None,
                 leak_current_interval:   bool  = False):
        self.sched    = scheduler
        self.fc       = forecaster
        self.data     = real_data
        self.soc0     = soc_init
        # The CONTROL horizon: how far ahead each solve looks. The day-ahead
        # arm uses 48, the gate-closure arm 22 (11 h). It is deliberately NOT
        # the same number as `steps_per_day`, which is the calendar day the
        # forecast is anchored to and always a full 24 h -- one forecast serves
        # both arms, so shortening the horizon must not re-key the cache or
        # start slicing "days" 11 hours long.
        self.H        = horizon_steps
        self.spd      = steps_per_day
        self.reopt_n  = reoptimize_every
        self.leak_current_interval = bool(leak_current_interval)
        self.freq     = freq
        self._fc_cache: dict = {}

        # The default soc_init=10.0 is above soc_max for the study battery
        # (0.80 * 10.0 = 8.0); catch it here rather than in an infeasible LP.
        if not (scheduler.soc_min <= soc_init <= scheduler.soc_max):
            raise ValueError(
                f"soc_init={soc_init} outside the battery's usable window "
                f"[{scheduler.soc_min}, {scheduler.soc_max}]"
            )

        # Supplied by the study so the controller is tariff-agnostic; the AU
        # calculation is only the fallback for callers that predate the switch.
        if rate_vectors is None:
            buy_arr, sell_arr, const = au_rate_vectors(
                self.data.index, self.data["SMP"].values,
                int(round(self.sched.delta_t * 60)))
        else:
            buy_arr, sell_arr, const = rate_vectors
        if not (len(buy_arr) == len(sell_arr) == len(const) == len(self.data)):
            raise ValueError(
                f"rate vectors cover {len(buy_arr)}/{len(sell_arr)}/{len(const)} "
                f"steps, data has {len(self.data)}"
            )
        self.buy_rate   = np.asarray(buy_arr, dtype=float)
        self.sell_rate  = np.asarray(sell_arr, dtype=float)
        self.fixed_cost = np.asarray(const, dtype=float)

    def _real_slice(self, k: int, h: int) -> tuple:
        sl = self.data.iloc[k : k + h]
        return (
            self.buy_rate[k : k + h].tolist(),
            self.sell_rate[k : k + h].tolist(),
            sl["Energy_Generation"].tolist(),
            sl["Energy_Consumption"].tolist(),
        )

    def _forecast_slice(self, k: int, h: int) -> tuple:
        day_idx    = k // self.spd
        day_offset = k %  self.spd

        for d in [day_idx, day_idx + 1]:
            if d not in self._fc_cache:
                start_k = d * self.spd
                if start_k < len(self.data):
                    anchor = self.data.index[start_k]
                    self._fc_cache[d] = self.fc.predict_next_day(
                        anchor, self.spd, freq=self.freq)

        fc_today = self._fc_cache[day_idx]
        fc_sl    = fc_today.iloc[day_offset : day_offset + h].reset_index(drop=True)

        if len(fc_sl) < h and (day_idx + 1) in self._fc_cache:
            missing     = h - len(fc_sl)
            fc_tomorrow = self._fc_cache[day_idx + 1]
            fc_next     = fc_tomorrow.iloc[:missing].reset_index(drop=True)
            fc_sl       = pd.concat([fc_sl, fc_next], ignore_index=True)

        if len(fc_sl) < h:
            # Past the end of the forecast table: hold the last known interval.
            # The `ds` column is rebuilt rather than repeated -- concatenating
            # the same row n times duplicates its timestamp, and anything that
            # later joins this frame on `ds` (forecast_error_metrics does) would
            # silently fan out those rows. Only the final horizon of a run can
            # reach here, and only when the lookahead tail is short.
            pad = h - len(fc_sl)
            last = fc_sl.iloc[[-1]]
            fc_sl = pd.concat([fc_sl] + [last] * pad, ignore_index=True)
            step = pd.Timedelta(self.freq)
            fc_sl.loc[fc_sl.index[-pad:], "ds"] = [
                fc_sl["ds"].iloc[-pad - 1] + step * (i + 1) for i in range(pad)
            ]

        buy_rate_real  = self.buy_rate[k : k + h].tolist()
        sell_rate_real = self.sell_rate[k : k + h].tolist()

        p_gen_fc = fc_sl["yhat_gen"].tolist()
        p_con_fc = fc_sl["yhat_con"].tolist()

        if self.leak_current_interval:
            # F3 - the applied step's REALIZED generation and consumption,
            # substituted into the forecast the plan is built from. At the
            # moment of deciding interval k the controller cannot know what that
            # interval will total; it spans the next 30 minutes. And because
            # only step 0 is ever executed, this is the one step the leak
            # touches -- the action is optimised against the truth it is then
            # scored on, which flatters the forecast arm precisely where the
            # study measures it.
            #
            # Kept behind a flag rather than deleted so the published numbers
            # remain reproducible and the leak's size is measurable.
            p_gen_fc[0] = self.data["Energy_Generation"].iloc[k]
            p_con_fc[0] = self.data["Energy_Consumption"].iloc[k]

        return buy_rate_real, sell_rate_real, p_gen_fc, p_con_fc

    def run(self, num_days: int = 5, use_forecast: bool = True) -> pd.DataFrame:
        # Days, not horizons: with an 11 h control horizon `num_days * self.H`
        # would simulate under half the year and quietly report it as a year.
        T_total  = min(num_days * self.spd, len(self.data))
        soc_cur  = self.soc0
        plan     = None
        plan_pos = 0
        history  = []

        for k in range(T_total):
            # Clamped by the DATA, not by the scored window. Clamping by
            # `T_total - k` shrinks the horizon over the final day, which
            # penalises a long horizon more than a short one and so contaminates
            # exactly the comparison the H24/H11 arms exist to make. When
            # `self.data` carries a lookahead tail past T_total this never binds.
            horizon = min(self.H, len(self.data) - k)
            if horizon <= 0:
                break

            # F6 - `soc_cur` is the SoC at the END of step k-1, and the plan
            # in hand was solved at that step, so the entry describing the same
            # instant is soc_plan[plan_pos - 1], not soc_plan[plan_pos]. The old
            # index compared the SoC now against the SoC one step into the
            # future, which made this the planned next-step delta rather than a
            # deviation (max observed 0.789 == p_max/eff*delta_t exactly).
            # Corrected, it is 0 by construction: the plan's action is applied
            # verbatim and the SoC recursion carries no noise. It is kept as a
            # live invariant, not as a trigger - see `need_reopt` below.
            soc_dev = 0.0
            if plan is not None and 0 < plan_pos <= len(plan["soc_plan"]):
                soc_dev = abs(soc_cur - plan["soc_plan"][plan_pos - 1])
                # Recorded AND enforced: the executed SoC follows the plan to
                # solver precision, so any real drift is a bug in the loop.
                if soc_dev > SOC_TOL:
                    raise AssertionError(
                        f"step {k}: executed SoC {soc_cur:.9f} kWh drifted "
                        f"{soc_dev:.3e} from the plan"
                    )

            # `soc_deviation_threshold` used to appear here as a third trigger.
            # It is gone rather than merely unused: with the index above correct
            # the deviation is 0 by construction, so no threshold on it can ever
            # fire, and with reoptimize_every=1 `k % self.reopt_n == 0` is true
            # every step regardless. The two study arms now differ only in
            # `use_forecast`, which is the whole point of the comparison.
            need_reopt = (
                plan is None
                or plan_pos >= len(plan["x_ch"])
                or k % self.reopt_n == 0
            )

            if need_reopt:
                fn = self._forecast_slice if use_forecast else self._real_slice
                buy_h, sell_h, p_gen_h, p_con_h = fn(k, horizon)
                plan     = self.sched.solve(soc_cur, buy_h, sell_h, p_gen_h, p_con_h)
                plan_pos = 0

            act_ch  = plan["x_ch"][plan_pos]
            act_dis = plan["x_dis"][plan_pos]

            # Check the APPLIED action before anything is derived from it. A
            # balance check on act_buy/act_sell would be tautological -- both
            # come from p_net_real below, so that residual is identically zero
            # however wrong the plan is. A bad action is the root cause; the SoC
            # bound further down would only catch it as a symptom, and only when
            # it happens to push the pack out of range.
            # The mutex matters from the moment the model stops carrying the
            # d_ch/d_dis binaries (exclusivity="inverter"), which is where a
            # degenerate LP starts being free to do both at once.
            max_ch  = getattr(self.sched, "max_ch_kw",  self.sched.p_max)
            max_dis = getattr(self.sched, "max_dis_kw", self.sched.p_max)
            if not (-ACTION_TOL <= act_ch  <= max_ch  + ACTION_TOL and
                    -ACTION_TOL <= act_dis <= max_dis + ACTION_TOL):
                raise AssertionError(
                    f"step {k}: action outside charge [0, {max_ch:.6f}] / "
                    f"discharge [0, {max_dis:.6f}] kW "
                    f"(charge {act_ch:.9f}, discharge {act_dis:.9f})"
                )
            if act_ch > ACTION_TOL and act_dis > ACTION_TOL:
                raise AssertionError(
                    f"step {k}: charging and discharging at once "
                    f"({act_ch:.9f} / {act_dis:.9f} kW)"
                )

            real_smp       = self.data["SMP"].iloc[k]
            real_buy_rate  = self.buy_rate[k]
            real_sell_rate = self.sell_rate[k]
            real_gen       = self.data["Energy_Generation"].iloc[k]
            real_con       = self.data["Energy_Consumption"].iloc[k]

            p_net_real = real_con + act_ch - real_gen - act_dis
            if p_net_real > 0:
                act_buy  = p_net_real
                act_sell = 0.0
            else:
                act_buy  = 0.0
                act_sell = -p_net_real

            # F7 - clipping here would silently create or destroy energy,
            # because `p_net_real` above was already computed from the unclipped
            # actions. The MILP is re-solved from the true `soc_cur` every step
            # and enforces the same bounds, so a violation is a bug, not a
            # saturation to absorb. Matches Environment.py:461-495 upstream.
            delta_soc = (act_ch * self.sched.eff - act_dis / self.sched.eff) * self.sched.delta_t
            soc_next  = soc_cur + delta_soc
            if not (self.sched.soc_min - SOC_TOL <= soc_next <= self.sched.soc_max + SOC_TOL):
                raise AssertionError(
                    f"step {k}: SoC {soc_next:.9f} kWh outside "
                    f"[{self.sched.soc_min}, {self.sched.soc_max}] "
                    f"(was {soc_cur:.9f}, charge {act_ch:.6f}, discharge {act_dis:.6f})"
                )
            # Only the floating-point overshoot is trimmed.
            soc_cur = float(min(max(soc_next, self.sched.soc_min), self.sched.soc_max))


            step_cost = (act_buy * real_buy_rate - act_sell * real_sell_rate) * self.sched.delta_t

            history.append({
                "Timestamp":        self.data.index[k],
                "Price_SMP":        real_smp,
                "Buy_Rate_kWh":     real_buy_rate,
                "Sell_Rate_kWh":    real_sell_rate,
                "Solar_Gen":        real_gen,
                "Consumption":      real_con,
                "SoC_kWh":          soc_cur,
                # SoC_Planned and SoC_Deviation used to be written here. With
                # reoptimize_every=1 the plan is re-solved every step, so
                # plan_pos is always 0 and both are constant by construction --
                # SoC_Deviation identically 0, which is asserted above as a live
                # invariant rather than stored 17,520 times as a column of zeros.
                "Charge_kW":        act_ch,
                "Discharge_kW":     act_dis,
                "Buy_kW":           act_buy,
                "Sell_kW":          act_sell,
                "Step_Cost":        step_cost,
                "Reoptimized":      int(need_reopt),
            })
            plan_pos += 1

        return pd.DataFrame(history).set_index("Timestamp")


# =====================================================================
# 5 — KPITracker (unchanged)
# =====================================================================

class KPITracker:

    # F10 - a site that never exports gives sell_nb_e == 0, and a net exporter
    # gives cost_nb <= 0; both used to produce inf/NaN or a sign-flipped
    # percentage read as a real result. figure.py already warns about the second
    # case, so it is live. `_pct` returns a marker instead.
    @staticmethod
    def _pct(value: float, baseline: float) -> str:
        # A non-positive baseline has no meaningful percentage: dividing by it
        # inverts the sign, so a net-exporting site (cost_nb <= 0, which
        # figure.py already warns is live) would read as a saving when the
        # comparison is simply undefined.
        if not np.isfinite(baseline) or baseline <= 1e-12:
            return "n/a"
        return f"{100 * value / baseline:+.1f} %"

    @staticmethod
    def compare_three(df_fc: pd.DataFrame,
                       df_pk: pd.DataFrame,
                       delta_t: float = 0.5) -> tuple:

        buy_nb=np.maximum(0,df_fc["Consumption"]-df_fc["Solar_Gen"])
        sell_nb=np.maximum(0,df_fc["Solar_Gen"]-df_fc["Consumption"])

        cost_nb=((buy_nb*df_fc["Buy_Rate_kWh"]-sell_nb*df_fc["Sell_Rate_kWh"])*delta_t).sum()
        cost_pk=df_pk["Step_Cost"].sum()
        cost_fc=df_fc["Step_Cost"].sum()

        buy_nb_e=(buy_nb*delta_t).sum()
        buy_pk=(df_pk["Buy_kW"]*delta_t).sum()
        buy_fc=(df_fc["Buy_kW"]*delta_t).sum()

        sell_nb_e=(sell_nb*delta_t).sum()
        sell_pk=(df_pk["Sell_kW"]*delta_t).sum()
        sell_fc=(df_fc["Sell_kW"]*delta_t).sum()

        pct = KPITracker._pct
        rows=[
        {"KPI":"Total cost",
         "No battery":f"{cost_nb:.2f}",
         "Perfect foresight":f"{cost_pk:.2f} ({pct(cost_pk-cost_nb, cost_nb)})",
         "Forecast (Prophet)":f"{cost_fc:.2f} ({pct(cost_fc-cost_nb, cost_nb)})"},
        {"KPI":"Regret vs perfect foresight (% of no-battery cost)",
         "No battery":"—",
         "Perfect foresight":"0.00 (+0.0 %)",
         "Forecast (Prophet)":f"{cost_fc-cost_pk:.2f} ({pct(cost_fc-cost_pk, cost_nb)})"},
        {"KPI":"Energy bought (kWh)",
         "No battery":f"{buy_nb_e:.1f}",
         "Perfect foresight":f"{buy_pk:.1f} ({pct(buy_pk-buy_nb_e, buy_nb_e)})",
         "Forecast (Prophet)":f"{buy_fc:.1f} ({pct(buy_fc-buy_nb_e, buy_nb_e)})"},
        {"KPI":"Energy sold (kWh)",
         "No battery":f"{sell_nb_e:.1f}",
         "Perfect foresight":f"{sell_pk:.1f} ({pct(sell_pk-sell_nb_e, sell_nb_e)})",
         "Forecast (Prophet)":f"{sell_fc:.1f} ({pct(sell_fc-sell_nb_e, sell_nb_e)})"}]
        return pd.DataFrame(rows).set_index("KPI"), {
            # raw numeric values, reused for the global multi-dataset summary
            "cost_no_battery": cost_nb, "cost_oracle": cost_pk, "cost_prophet": cost_fc,
            "buy_no_battery": buy_nb_e, "buy_oracle": buy_pk, "buy_prophet": buy_fc,
            "sell_no_battery": sell_nb_e, "sell_oracle": sell_pk, "sell_prophet": sell_fc,
        }


# =====================================================================
# 6b — The study's windows, read once
# =====================================================================


def load_study_frames(file_path: str, *, H: int = 48, delta_t: float = 0.5,
                      n_train: int = 730, n_sim: int = 365,
                      start_ts: str = "2010-07-01 00:30:00",
                      smp_source: str | None = None,
                      verbose: bool = False) -> dict:
    """Read one household file and cut the study's windows out of it.

    Extracted verbatim from `run_pipeline_for_file`, which is still its main
    caller, so that `forecast_benchmark` scores its forecasts over exactly the
    same train/sim split the sweep controls over. A screen that cut its own
    windows would be measuring a different year.

    Returns a dict of frames rather than a tuple: there are five of them, two
    pairs differ only by unit, and a positional swap between those pairs is
    precisely the factor-of-two error F1 below exists to prevent.
    """
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    raw = pd.read_csv(file_path)
    raw.index = pd.to_datetime(raw["Timestamp_UTC"], format="ISO8601")

    # F8 - every day-boundary assumption below (k // H, iloc[n_train*H : ...])
    # needs a monotonic, unique, gap-free grid of exactly H rows per day. One
    # missing or duplicated interval silently misaligns the train/sim split and
    # every Prophet anchor, with no error, so it is checked once here.
    if not raw.index.is_monotonic_increasing:
        raise ValueError(f"{file_path}: Timestamp_UTC is not sorted ascending.")
    if not raw.index.is_unique:
        dupes = raw.index[raw.index.duplicated()].unique()
        raise ValueError(f"{file_path}: {len(dupes)} duplicated timestamp(s), "
                         f"first {list(dupes[:3])}.")

    # `start_ts` is naive but the CSV carries +00:00, so localise before slicing
    # rather than relying on pandas' naive-vs-aware comparison.
    start = pd.Timestamp(start_ts)
    if raw.index.tz is not None and start.tz is None:
        start = start.tz_localize(raw.index.tz)
    df_all = raw.loc[start:, ["SMP", "Energy_Generation", "Energy_Consumption"]].copy()

    step = pd.Timedelta(minutes=int(round(delta_t * 60)))
    gaps = df_all.index.to_series().diff().dropna()
    if not (gaps == step).all():
        bad = gaps[gaps != step]
        raise ValueError(f"{file_path}: {len(bad)} irregular interval(s) "
                         f"(expected {step}); first at {bad.index[0]} = {bad.iloc[0]}.")
    if len(df_all) % H:
        raise ValueError(f"{file_path}: {len(df_all)} steps is not a whole number "
                         f"of {H}-step days.")
    if df_all.isna().any().any():
        na = df_all.isna().sum()
        raise ValueError(f"{file_path}: NaNs present -> {na[na > 0].to_dict()}")

    # The price series. Default is the SMP column already in the household file
    # (AEMO, EUR/kWh, half-hourly, 2010-2013). `smp_source` swaps in one of the
    # European series under Input data/SMP -- but note those start in 2015 and
    # the Ausgrid profiles end mid-2013, so the two do not overlap and an
    # alignment rule has to be chosen deliberately rather than left to a ffill.
    if smp_source:
        smp = load_smp_data(smp_source)["SMP"].reindex(df_all.index)
        if smp.isna().any():
            raise ValueError(
                f"SMP series {smp_source!r} does not cover "
                f"{df_all.index[0]}..{df_all.index[-1]} "
                f"({int(smp.isna().sum())} of {len(smp)} intervals missing). "
                f"Choose an explicit alignment rather than forward-filling."
            )
        # EUR/MWh series are stored unscaled; the household column is EUR/kWh.
        if float(smp.abs().quantile(0.95)) > 2.0:
            smp = smp / 1000.0
        df_all["SMP"] = smp.astype(float)

    # F1 - Ausgrid publishes ENERGY in kWh per 30-min interval, but the
    # controller loop below (the power balance, p_max, the `* delta_t` in the
    # costing) works in kW. Upstream Energy_Community keeps kWh/interval
    # throughout instead (MILP_Household.step_energy_kwh). Either convention is
    # fine; mixing them is what halved every absolute figure in the previous
    # results, so BOTH frames are kept explicitly and each consumer is handed
    # the one it expects:
    #
    #   *_kwh   what the file holds, and what HouseholdEnvironment reads. The
    #           environment derives the tariff blocks, the metered peaks and the
    #           agreed power from these columns, so handing it the kW frame
    #           silently doubles every peak -- harmless while nothing priced a
    #           capacity charge, and a factor-of-two error the moment one does.
    #   *_kw    what ReactiveController and the schedulers work in.
    df_all_kwh = df_all
    df_all = df_all.copy()
    df_all[["Energy_Generation", "Energy_Consumption"]] /= delta_t

    if verbose:
        print(f"Native granularity (30 min): {len(df_all)} steps")

    df_train = df_all.iloc[: n_train * H]
    df_sim   = df_all.iloc[n_train * H : (n_train + n_sim) * H]
    # Everything the controller may LOOK at: the scored window plus one horizon
    # of tail, so a 24 h horizon is not truncated over the final day while an
    # 11 h one is. Only `df_sim` is ever scored. If the file has no tail to
    # spare, the old truncating behaviour returns and says so.
    df_ctrl  = df_all.iloc[n_train * H : (n_train + n_sim) * H + H]
    # The same window in the environment's units, for build_study_env below.
    df_ctrl_kwh = df_all_kwh.iloc[n_train * H : (n_train + n_sim) * H + H]
    if verbose:
        if len(df_ctrl) < len(df_sim) + H:
            print(f"  ! only {len(df_ctrl) - len(df_sim)} of {H} lookahead steps "
                  f"available; the last day's horizon will be truncated")
        print(f"Training: {len(df_train)} steps ({n_train} days)")
        print(f"Simulation: {len(df_sim)} steps ({n_sim} days)")

    if len(df_train) == 0 or len(df_sim) == 0:
        raise ValueError(
            f"Not enough data for {dataset_name} "
            f"(train={len(df_train)}, sim={len(df_sim)}). "
            f"Check start_ts / n_train / n_sim for this file."
        )

    return {"dataset_name": dataset_name, "df_all_kwh": df_all_kwh,
            "df_train": df_train, "df_sim": df_sim,
            "df_ctrl": df_ctrl, "df_ctrl_kwh": df_ctrl_kwh}

# The order the screen reports simple kinds in, and the roster it runs by
# default: every fit-free source, cheapest bet first.
BENCHMARK_KINDS = ["persistence", "weekly", "daytype",
                   "mean3", "mean7", "median7", "median14",
                   # Fitted, and minutes rather than milliseconds each -- but
                   # cached, and the cache is shared with the sweep's arms.
                   "hbd_baseline", "hbd", "hbd_median14",
                   # THE TUNED PROPHET, which had a full-year ARM and no
                   # full-roster error number: the only error figure it carried
                   # was the 8-household/90-day screen `tune_prophet` runs, and
                   # that screen is deliberately over a different window from
                   # this table, so the two could not be read against each
                   # other. Reading it here costs nothing -- the tuned arms have
                   # already cached its tables for all 30 households -- and it
                   # is the row that says whether tuning bought back the gap to
                   # seasonal-naive on the sample the study actually reports.
                   "prophet_tuned"]


def forecast_benchmark(data_dir: str,
                       dataset_ids: list | None = None,
                       kinds: list | None = None,
                       filename_template: str = "Ausgrid {id}.csv",
                       include_prophet: bool = False,
                       out_path: str | None = None,
                       H: int = 48,
                       delta_t: float = 0.5,
                       n_train: int = 730,
                       n_sim: int = 365,
                       start_ts: str = "2010-07-01 00:30:00",
                       smp_source: str | None = None,
                       forecast_cache_dir: str | None = None) -> pd.DataFrame:
    """Score every fit-free forecaster on every household. No MILP, no fitting.

    A study arm costs a MILP household-year; a forecast does not. Building a
    naive table is a handful of array copies, so the question "which trivial
    method forecasts this household best" can be answered for the whole roster
    in the time one arm spends on one day. That is what this is for: screen
    here, then promote only what wins to `STUDY_ARMS` and pay for the EUR.

    It is the same measurement the sweep makes -- same windows via
    `load_study_frames`, same anchors, same `forecast_error_metrics` -- so a
    row here and the `gen_*`/`con_*` columns of the matching arm's checkpoint
    must agree. That is the cross-check, and it is why this does not cut its
    own windows.

    `include_prophet` adds Prophet to the comparison by READING its cached
    table. Off by default because a cache miss would fit it, which is the cost
    this function exists to avoid; turn it on once the sweep has run.

    Returns one row per (dataset, kind, channel).
    """
    kinds = list(BENCHMARK_KINDS if kinds is None else kinds)
    if dataset_ids is None:
        dataset_ids = study_units().index.tolist()
    unknown = [k for k in kinds if k not in SIMPLE_KINDS
               and k not in FITTED_KINDS and k not in PROPHET_KINDS
               and k != "truth"]
    if unknown:
        raise ValueError(
            f"unknown forecaster kind(s) {unknown}; expected 'truth', a simple "
            f"kind {sorted(SIMPLE_KINDS)}, a fitted kind "
            f"{sorted(FITTED_KINDS)}, or a Prophet kind {sorted(PROPHET_KINDS)}"
        )

    rows = []
    n_anchors = 0
    for ident in dataset_ids:
        path = os.path.join(data_dir, filename_template.format(id=ident))
        if not os.path.isfile(path):
            print(f"  ! {path} not found, skipped")
            continue
        f = load_study_frames(path, H=H, delta_t=delta_t, n_train=n_train,
                              n_sim=n_sim, start_ts=start_ts,
                              smp_source=smp_source)
        name, train, sim, ctrl = (f["dataset_name"], f["df_train"],
                                  f["df_sim"], f["df_ctrl"])
        # Anchors over df_ctrl and scoring on df_sim, exactly as the pipeline
        # does: the lookahead tail is forecast but never scored.
        anchors = list(ctrl.index[::H])
        # max, not "whichever household was read last". Every household in the
        # study shares one window, so they agree today; a file short by a day
        # would otherwise decide the number written into the provenance column
        # for all of them, silently.
        n_anchors = max(n_anchors, len(anchors))
        # History ahead of the simulation, so day 1 of a 14-day method is built
        # from real data rather than from itself.
        frame = pd.concat([train, ctrl])

        tables = {k: build_forecast_table(SIMPLE_KINDS[k](frame, H),
                                          anchors, H, "30min")
                  for k in kinds if k in SIMPLE_KINDS}
        if "truth" in kinds:
            tables["truth"] = build_forecast_table(
                TruthForecaster(ctrl, H), anchors, H, "30min")
        # The fitted kinds go through the cache rather than being built inline
        # like the fit-free ones. Two reasons, and the second is the important
        # one: their fits cost minutes rather than milliseconds, and the cache
        # key is the same config the sweep will ask for -- so screening a
        # household here WARMS the table its MPC arm then reads instead of
        # fitting the same model twice.
        for k in kinds:
            if k in FITTED_KINDS:
                served, _ = load_or_build_forecasts(
                    name, train, ctrl, H, "30min", EnergyForecaster(None, None),
                    cache_dir=forecast_cache_dir, kind=k, history=train)
                tables[k] = served.table
        # The Prophet kinds, from cache for the same reason: a miss FITS, which
        # is the cost this function exists to avoid. Which settings each kind
        # carries is `PROPHET_KIND_PARAMS`, and it has to be the arm's own pair
        # or the digest misses and the tuned model is refit from scratch -- Stan
        # for an hour against a second of CSV.
        #
        # `include_prophet` predates the roster carrying a Prophet kind at all
        # and still means what it meant: add the DEFAULT Prophet. `prophet_tuned`
        # comes in through `kinds`, like every other method.
        for k in kinds:
            if k in PROPHET_KINDS:
                _con_p, _gen_p = PROPHET_KIND_PARAMS[k]
                served, _ = load_or_build_forecasts(
                    name, train, ctrl, H, "30min",
                    EnergyForecaster(_con_p, _gen_p),
                    cache_dir=forecast_cache_dir,
                    refit_every_days=30, kind=k, history=train)
                tables[k] = served.table
        if include_prophet and "prophet" not in tables:
            served, _ = load_or_build_forecasts(
                name, train, ctrl, H, "30min",
                EnergyForecaster(None, None), cache_dir=forecast_cache_dir,
                refit_every_days=30, kind="prophet", history=train)
            tables["prophet"] = served.table

        for kind, table in tables.items():
            m = forecast_error_metrics(table, sim, H, history=train)
            for ch, label in (("gen", "generation"), ("con", "consumption")):
                rows.append({
                    "dataset": name, "kind": kind, "channel": label,
                    "mae": m[f"{ch}_mae"], "rmse": m[f"{ch}_rmse"],
                    "nmae": m[f"{ch}_nmae"],
                    "skill_vs_naive": m[f"{ch}_skill_vs_naive"],
                    "mae_first_month": m.get(f"{ch}_mae_first_month", float("nan")),
                    "mae_last_month": m.get(f"{ch}_mae_last_month", float("nan")),
                })
        print(f"  [benchmark] {name}: {len(tables)} kind(s) over "
              f"{len(anchors)} day(s)")

    out = pd.DataFrame(rows)
    if out.empty:
        raise FileNotFoundError(f"no household files read from {data_dir}")
    # WHAT WINDOW THIS WAS SCORED OVER, carried in the file. Unlike a run
    # checkpoint, this CSV had no config beside it, so a notebook that loaded it
    # instead of recomputing described it with whatever `N_SIM` happened to be
    # set to in a cell above -- and would have gone on printing "365 simulated
    # days" over a table scored on 90. The caller can now check rather than
    # assume, and `n_anchors` is stored because it is the number of forecast
    # DAYS (n_sim plus the one anchor whose horizon runs past the scored
    # window), which is not the same as `n_sim` and was reported as if it were.
    for col, val in (("n_sim", n_sim), ("n_train", n_train),
                     ("steps_per_day", H), ("start_ts", start_ts),
                     ("n_anchors", n_anchors),
                     ("include_prophet", bool(include_prophet))):
        out[col] = val
    # Ordered so a printed pivot reads in the roster's order, not alphabetically.
    order = [k for k in FORECAST_KIND_LABELS if k in set(out["kind"])]
    out["kind"] = pd.Categorical(out["kind"], categories=order, ordered=True)
    out = out.sort_values(["dataset", "kind", "channel"]).reset_index(drop=True)

    out_path = os.path.join(RESULTS_DIR, "forecast_benchmark.csv") \
        if out_path is None else out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[benchmark] {len(out)} rows -> {out_path}")
    return out


PROVENANCE_COLUMNS = ("n_sim", "n_train", "steps_per_day", "start_ts",
                      "n_anchors", "include_prophet", "refit_every_days")


def provenance(df: pd.DataFrame, **expected) -> str:
    """What window a cached scoring table was produced over, as one line.

    `forecast_benchmark` and `tune_prophet` write a CSV that a notebook may
    LOAD instead of recomputing. A run checkpoint carries the config it was
    produced under and `collect_results` drops a stale one; these two tables had
    no config at all, so the cell that loaded them described the numbers with
    whatever `N_SIM` was set to in a cell above and could not have noticed a
    disagreement. Pass the values the caller believes hold and any that differ
    are named in the returned string rather than left to be assumed.

    A file written before this existed carries no provenance columns and says
    so, which is the honest answer for it.
    """
    have = {c: df[c].dropna().unique() for c in PROVENANCE_COLUMNS if c in df}
    if not have:
        return ("no provenance in this file -- it predates the columns, so what "
                "window it was scored over is not recoverable from it; delete it "
                "and rescore to find out")
    parts, mismatch = [], []
    for col, vals in have.items():
        # More than one value in a column means rows from two different runs
        # were concatenated into one file, which no reader would suspect.
        shown = vals[0] if len(vals) == 1 else f"MIXED{sorted(vals)}"
        parts.append(f"{col}={shown}")
        if col in expected and len(vals) == 1 and vals[0] != expected[col]:
            mismatch.append(f"{col}: file says {vals[0]}, caller says "
                            f"{expected[col]}")
    line = ", ".join(parts)
    if mismatch:
        line += "\n  ! DISAGREES WITH THIS NOTEBOOK -- " + "; ".join(mismatch)
        line += "\n  ! delete the CSV and rescore, or trust the file over the cell"
    return line


def benchmark_ranking(bench: pd.DataFrame) -> pd.DataFrame:
    """Median skill and nMAE per kind and channel, across households.

    The MEDIAN across households, not the mean: one household with a broken
    inverter and an nMAE of 4 would otherwise choose the study's baseline.
    """
    piv = (bench.groupby(["kind", "channel"], observed=True)
                .agg(nmae=("nmae", "median"),
                     skill=("skill_vs_naive", "median"),
                     households=("dataset", "nunique"))
                .reset_index())
    return piv.pivot(index="kind", columns="channel",
                     values=["nmae", "skill", "households"])



# A COORDINATE search, not a grid: each entry moves one axis off the module
# defaults, so a result reads as "this knob is what was wrong" rather than "some
# combination of six knobs scored better". {name: (con overrides, gen overrides)}
#
# The axes, and the suspicion behind each:
#   changepoint_prior_scale   how freely the trend bends. A household has no
#                             trend to speak of, so a flexible one is free
#                             variance the 30-day extrapolation has to pay for.
#   growth="flat"             the same suspicion taken to its conclusion: no
#                             trend term at all.
#   seasonality_prior_scale   how hard the daily/weekly shapes are fit. The
#                             default 10 is effectively unregularised.
#   yearly_seasonality        two years of training data is two observations of
#                             any yearly shape. That is not a season, it is a
#                             pair of anecdotes.
#   seasonality_mode          additive vs multiplicative, flipped per channel.
#   weekly_seasonality        flipped per channel: load has a working week, a
#                             roof does not.
PROPHET_TUNING_GRID = {
    "default":       ({}, {}),
    "cps_0.001":     ({"changepoint_prior_scale": 0.001},
                      {"changepoint_prior_scale": 0.001}),
    "cps_0.01":      ({"changepoint_prior_scale": 0.01},
                      {"changepoint_prior_scale": 0.01}),
    "cps_0.5":       ({"changepoint_prior_scale": 0.5},
                      {"changepoint_prior_scale": 0.5}),
    "flat_trend":    ({"growth": "flat"}, {"growth": "flat"}),
    "sps_0.1":       ({"seasonality_prior_scale": 0.1},
                      {"seasonality_prior_scale": 0.1}),
    "sps_1":         ({"seasonality_prior_scale": 1.0},
                      {"seasonality_prior_scale": 1.0}),
    "no_yearly":     ({"yearly_seasonality": False},
                      {"yearly_seasonality": False}),
    "mode_flip":     ({"seasonality_mode": "multiplicative"},
                      {"seasonality_mode": "additive"}),
    "weekly_flip":   ({"weekly_seasonality": False},
                      {"weekly_seasonality": True}),
}


def _tune_one_household(path, grid, H, delta_t, n_train, n_sim,
                        refit_every_days, start_ts):
    """One household, every config in the grid. Forecast error only, no MILP."""
    f = load_study_frames(path, H=H, delta_t=delta_t, n_train=n_train,
                          n_sim=n_sim, start_ts=start_ts)
    name, train, sim = f["dataset_name"], f["df_train"], f["df_sim"]
    rows = []
    for cfg, (con_over, gen_over) in grid.items():
        fc = EnergyForecaster(con_over, gen_over)
        # Through the REFIT path, not a single fit: the deployed arms refit every
        # 30 days, and a config tuned against a 90-day extrapolation would be
        # tuned against a regime the study never runs.
        table = build_forecast_table_refit(fc.params_con, fc.params_gen,
                                           train, sim, H, "30min",
                                           refit_every_days)
        m = forecast_error_metrics(table, sim, H, history=train)
        for ch, label in (("gen", "generation"), ("con", "consumption")):
            rows.append({"config": cfg, "dataset": name, "channel": label,
                         "mae": m[f"{ch}_mae"], "nmae": m[f"{ch}_nmae"],
                         "skill_vs_naive": m[f"{ch}_skill_vs_naive"]})
    return rows


def tune_prophet(data_dir: str,
                 dataset_ids: list | None = None,
                 grid: dict | None = None,
                 n_train: int = 730,
                 n_sim: int = 90,
                 refit_every_days: int = 30,
                 H: int = 48,
                 delta_t: float = 0.5,
                 start_ts: str = "2010-07-01 00:30:00",
                 filename_template: str = "Ausgrid {id}.csv",
                 out_path: str | None = None,
                 n_jobs: int = 1) -> pd.DataFrame:
    """Score Prophet configurations on forecast error alone. No MILP.

    `forecast_benchmark` found Prophet behind plain seasonal-naive on both
    channels, which is a claim about THIS Prophet -- the module defaults -- and
    not about the model. This is what turns that into a fair question: give it a
    coordinate search over the settings most likely to be responsible, on the
    same metric and the same refit regime the arms use, and see whether any of
    them buys back the gap.

    A subset of households is enough and is the point: this is a screen, and the
    winner is validated on the full roster afterwards. One config on one
    household is about 20 s, so a ten-config grid over eight households is
    minutes, against the hours the same comparison would cost as study arms.

    Returns one row per (config, dataset, channel).
    """
    grid = PROPHET_TUNING_GRID if grid is None else grid
    if dataset_ids is None:
        dataset_ids = study_units().index.tolist()[:8]
    paths = [os.path.join(data_dir, filename_template.format(id=i))
             for i in dataset_ids]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} household file(s) not found, "
                                f"first {missing[0]}")

    args = (grid, H, delta_t, n_train, n_sim, refit_every_days, start_ts)
    if n_jobs == 1:
        rows = [r for p in paths for r in _tune_one_household(p, *args)]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(_tune_one_household)(p, *args) for p in paths)
        rows = [r for chunk in out for r in chunk]

    df = pd.DataFrame(rows)
    # The window this screen was run over, carried in the file for the same
    # reason as in `forecast_benchmark`: a notebook that loads the CSV rather
    # than recomputing it has no other way to know, and the screen deliberately
    # uses a SHORTER window and FEWER households than the arms do -- so a
    # reader told "365 days, 30 households" by a cell above would read this
    # table against the wrong sample.
    for col, val in (("n_sim", n_sim), ("n_train", n_train),
                     ("steps_per_day", H), ("start_ts", start_ts),
                     ("refit_every_days", refit_every_days)):
        df[col] = val
    order = [c for c in grid if c in set(df["config"])]
    df["config"] = pd.Categorical(df["config"], categories=order, ordered=True)
    df = df.sort_values(["dataset", "config", "channel"]).reset_index(drop=True)

    out_path = os.path.join(RESULTS_DIR, "prophet_tuning.csv") \
        if out_path is None else out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[tune] {len(df)} rows over {df['dataset'].nunique()} household(s) "
          f"-> {out_path}")
    return df


def tuning_ranking(tuning: pd.DataFrame) -> pd.DataFrame:
    """Median skill per config and channel, across the tuned households."""
    return (tuning.groupby(["config", "channel"], observed=True)
                  .agg(nmae=("nmae", "median"),
                       skill=("skill_vs_naive", "median"))
                  .reset_index()
                  .pivot(index="config", columns="channel",
                         values=["nmae", "skill"]))


# =====================================================================
# 7 — Pipeline for ONE dataset (formerly main(), now parameterized)
# =====================================================================

def full_period_bound_check(metrics: dict, tariff: str,
                            cycle_cost_eur_per_efc: float | None,
                            bound_is_exact: bool = True,
                            reporting_cycle_cost_eur_per_efc: float | None = None
                            ) -> dict:
    """Is the whole-period solve actually below everything it is the bound for?

    A denominator nobody can name is worse than no denominator, and a "ceiling"
    a rule walks over is a bug wearing a result's clothes. So the property is
    ASSERTED per run rather than argued for in a comment, and a violation is
    printed where the run's own log will carry it.

    What the bound is over depends on the tariff, and this is not a hedge:

      AU   no capacity charge, and the standing charge is a true constant, so
           `cost_*` is the whole of what any controller can move. The bound is
           over `cost_*` directly.
      SI   the dogovorjena moc is endogenous, so the standing charge is part of
           the same decision -- the solve sets it, a peak shaver lowers it. The
           bound is therefore over the TOTAL: cost + fixed + wear. Comparing
           `cost_*` alone would hold the optimum to a number it deliberately
           traded away, and it would lose to a rule that spent the fixed charge.

    `bound_is_exact` is False on an SI horizon shorter than the contract lag.
    There, no month in the window reads its line from another month in the
    window, so `add_endogenous_agreed_power` has nothing to model and the solve
    cannot see that its own peak sets its own standing charge -- while the rules,
    which converge the contract by simply re-running, do. Measured on a 10-day
    slice: the solve took a 3.43 kW peak and a 3.20 EUR standing charge where
    `fixed_schedule` took 2.25 kW and 2.37 EUR, and lost by 0.79 EUR. That is a
    horizon too short for the question, not a missing objective term, and the
    study's own arms are 365 days, so it is reported as a note rather than
    shouted about as a bug.

    Returns the totals it computed, so they land in the checkpoint rather than
    being recomputed differently downstream.

    TWO RATES, and the difference is the whole no-degradation arm. The BOUND is
    a property of the objective, so it is checked at the rate the MILP was
    actually charged (`cycle_cost_eur_per_efc`) -- an optimum that paid nothing
    for its cycles is a lower bound on a total that also charges nothing for
    them, and on no other. The totals WRITTEN OUT are the study's comparison
    money, so they are billed at the rate the pack costs
    (`reporting_cycle_cost_eur_per_efc`), which is the same number on every arm
    but the no-wear ones.

    Getting this wrong is not subtle. With one rate for both, the no-degradation
    arm writes `total_*` with no wear term in it at all, `saving_total` for those
    rows becomes a bill-only saving, and the controller that cycles hardest in
    the whole study appears at the TOP of a chart whose axis says "net of wear".
    """
    rate = float(cycle_cost_eur_per_efc or 0.0)
    report_rate = (rate if reporting_cycle_cost_eur_per_efc is None
                   else float(reporting_cycle_cost_eur_per_efc))
    names = sorted(k[len("cost_"):] for k in metrics if k.startswith("cost_"))

    def total(name, at_rate):
        t = float(metrics[f"cost_{name}"]) + at_rate * float(
            metrics.get(f"efc_{name}", 0.0) or 0.0)
        if tariff == "SI":
            t += float(metrics.get(f"fixed_{name}", 0.0) or 0.0)
        return t

    # `fixed_*` is only written for the three MILP arms; the rules carry it in
    # their own rows. Where it is absent the term is zero for every controller
    # alike, so the comparison stays like for like.
    totals = {f"total_{n}": total(n, report_rate) for n in names}
    # The bound, on the objective's own terms.
    objective = {n: total(n, rate) for n in names}
    ref = objective["milp_full"]
    # One cent of slack: HiGHS is pinned to a zero gap but the settlement walks
    # the trajectory through a different arithmetic path than the objective did.
    beaten = {n: objective[n] for n in names
              if n != "milp_full" and objective[n] < ref - 0.01}
    if beaten and not bound_is_exact:
        print(f"  [full period] not a bound on this horizon -- too short for the "
              f"contract lag, so the solve cannot price its own standing charge. "
              f"Beaten by: {', '.join(sorted(beaten))}")
    elif beaten:
        print("!" * 70)
        print("!!! THE WHOLE-PERIOD SOLVE IS NOT A LOWER BOUND -- its objective is")
        print("!!! missing a term of the bill these controllers are scored on:")
        for n, v in sorted(beaten.items(), key=lambda kv: kv[1]):
            print(f"!!!   {n:32s} {v:10.2f} vs optimum {ref:10.2f} "
                  f"({v - ref:+.2f})")
        print("!" * 70)
    totals["full_period_bound_is_exact"] = bool(bound_is_exact)
    return totals


def run_pipeline_for_file(file_path: str,
                           output_root: str = "results",
                           battery_cap: float = 10.0,
                           soc_min_pct: float = 0.10,
                           soc_max_pct: float = 0.80,
                           p_max: float = 1.5,
                           eff: float = 0.95,
                           delta_t: float = 0.5,
                           soc_init: float = 5.0,
                           H: int = 48,
                           n_train: int = 730,
                           n_sim: int = 365,
                           start_ts: str = "2010-07-01 00:30:00",
                           forecaster_params_con: dict | None = None,
                           forecaster_params_gen: dict | None = None,
                           refit_every_days: int | None = 30,
                           forecast_cache_dir: str | None = None,
                           oracle_cache_dir: str | None = None,
                           control_horizon: int | None = None,
                           tariff: str = "AU",
                           forecaster_kind: str = "prophet",
                           leak_current_interval: bool = False,
                           smp_source: str | None = None,
                           milp_parity: bool = True,
                           milp_exclusivity: str = "auto",
                           cycle_cost_eur_per_efc: float | str | None = "auto",
                           cycle_cost_reporting_eur_per_efc: float | str = "auto",
                           holiday_country: str = "AU",
                           holiday_subdiv: str | None = "NSW",
                           high_season_months: tuple = (5, 6, 7, 8),
                           local_timezone: str = "naive") -> dict:
    """
    Runs the full pipeline (train Prophet, run reactive + oracle,
    KPI, plots) for ONE dataset, and saves all results
    to output_root/<dataset_name>/.

    Returns a dict of numeric metrics (used for the global summary).
    """
    # `H` is the calendar day (48 half-hours). `control_horizon` is how far each
    # solve looks: 48 for the day-ahead arm, 22 (11 h) for the gate-closure arm.
    # Both arms read the SAME cached 24 h forecast.
    control_horizon = H if control_horizon is None else int(control_horizon)
    if not 1 <= control_horizon <= H:
        raise ValueError(
            f"control_horizon={control_horizon} must be within 1..{H} steps"
        )

    # What one equivalent full cycle costs, and therefore what the MILP pays to
    # cycle. "auto" is the DEFAULT and resolves to the pack price over the rated
    # cycle life -- 0.417 EUR/EFC for a 10 kWh pack at 250 EUR/kWh and 6000 EFC.
    #
    # It used to default to None, i.e. free. A battery that wears for nothing is
    # asked to arbitrage a spread of a few cents against a marginal cost of
    # zero, so it cycles for any gain at all: on the SI arm the clock rule alone
    # books 222 EFC/a to save 10 EUR, which is 93 EUR of pack life spent to earn
    # ten. Reported afterwards as `wear_eur` it looked like an observation; in
    # the objective it is a decision, which is what it always was.
    #
    # Pass 0.0 for the old unpriced behaviour, or a float to price it directly.
    import Battery_Economics as _be
    if cycle_cost_eur_per_efc == "auto":
        cycle_cost_eur_per_efc = _be.cycle_cost_eur_per_efc(battery_cap)
    cycle_cost_eur_per_efc = (
        None if not cycle_cost_eur_per_efc else float(cycle_cost_eur_per_efc))

    # WHAT THE CYCLES COST, as opposed to what the MILP was told they cost.
    # These were one number until the no-degradation arm needed them to be two.
    #
    # `cycle_cost_eur_per_efc` is a DISPATCH parameter: it is the shadow price in
    # the objective, and setting it to 0 is the experiment -- "what does a
    # controller do when nobody charges it for pack life". The pack still wears.
    # `summarize` used to read the solved rate back out of the checkpoint and
    # report `wear_eur` at it, which is right while the two agree and catastrophic
    # when they do not: a no-wear arm would report zero wear, and the controller
    # that cycles hardest would come out cheapest on `saving_net_of_wear`,
    # `saving_total`, the NPV and every figure built on them.
    #
    # So the accounting rate is resolved from the PACK, always, and travels in
    # the checkpoint beside the dispatch rate. On every arm but the no-wear ones
    # the two are the same number and nothing moves.
    if cycle_cost_reporting_eur_per_efc == "auto":
        cycle_cost_reporting_eur_per_efc = _be.cycle_cost_eur_per_efc(battery_cap)
    cycle_cost_reporting_eur_per_efc = float(cycle_cost_reporting_eur_per_efc)
    if battery_cap > 0 and not cycle_cost_reporting_eur_per_efc > 0:
        raise ValueError(
            "cycle_cost_reporting_eur_per_efc must be > 0 for a sized pack: it "
            "is what the cycles COST, not what the MILP was charged for them. "
            "Pass cycle_cost_eur_per_efc=0.0 for a no-degradation objective."
        )

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # What this arm COSTS TO RUN, wall clock, written into the checkpoint beside
    # what it earns. An arm is a method, and a method a household could deploy
    # has a compute budget as well as a bill -- but until this was stored the
    # only record of it was the mtime of the checkpoint file, so `arm_runtimes`
    # had to reconstruct it from the gaps between them. Started here rather than
    # below the checkpoint gate so it covers the whole run, which is what a
    # mtime difference covers and what makes the two comparable. A run served
    # from the checkpoint stores nothing: the arm was not executed, and a
    # skipped run recorded as 0.2 s would read as a method that costs nothing.
    _t_start = time.perf_counter()

    print(f"\n{'='*70}\n=== Dataset: {dataset_name} ===\n{'='*70}")

    # Resume: a finished dataset is skipped only if it was finished under THIS
    # configuration. A stale checkpoint is recomputed and says why, rather than
    # being resumed into -- mixing rows produced under different rules is
    # exactly the failure upstream's tag guard exists to prevent.
    cfg = study_config(
        battery_cap=battery_cap, soc_min_pct=soc_min_pct, soc_max_pct=soc_max_pct,
        p_max=p_max, eff=eff, delta_t=delta_t, soc_init=soc_init, H=H,
        n_train=n_train, n_sim=n_sim, start_ts=start_ts,
        control_horizon=control_horizon, tariff=tariff,
        # Both change the dispatch, so both invalidate a checkpoint. milp_parity
        # picks which battery model the MILP solves; cycle_cost_eur_per_efc puts
        # a wear shadow price in its objective (`wear_objective_terms`), which
        # changes what it decides to do and, deliberately, stops the objective
        # being the reported bill. That second half was aspirational until the
        # term was actually added to `UpstreamMILPScheduler.solve`; before that
        # this setting changed the checkpoint key and nothing else.
        # Which battery model the MILP solves. `milp_parity` only means anything
        # on the binary branch -- the inverter branch never consults it -- so
        # recording it there would put "parity=True" in the provenance of a run
        # that did not solve the parity model. None says "not applicable" rather
        # than saying something false.
        milp_parity=bool(milp_parity) if milp_exclusivity == "binary" else None,
        cycle_cost_eur_per_efc=cycle_cost_eur_per_efc,
        # NOT in the config, deliberately. The config is the checkpoint KEY, and
        # this rate changes no dispatch -- adding it here would invalidate all
        # 930 existing runs to record a number that is a pure function of
        # `battery_cap`, which is already in the key. It travels in the metrics
        # instead, where `collect_results` picks it up as a column.
        leak_current_interval=bool(leak_current_interval),
        forecaster_kind=forecaster_kind,
        refit_every_days=refit_every_days,
        smp_source=smp_source or "column",
        # WHICH solver produced the dispatch. CBC and HiGHS agree to the last
        # bit on this model, but they need not on a degenerate tie, and a panel
        # that silently mixes two solvers is the failure the tag guard exists
        # for. Recording it means a solver swap invalidates every checkpoint
        # and the sweep comes back one vintage throughout.
        solver=SOLVER_NAME,
        milp_exclusivity=milp_exclusivity,
        calendar=(holiday_country, holiday_subdiv, tuple(sorted(high_season_months))),
        # The clock every block, local day and clock rule is read on. It changes
        # which interval is peak, so it changes the answer and must invalidate.
        local_timezone=local_timezone,
        # The clock rule's windows. They are a study choice, they were re-tuned
        # once already, and a checkpoint produced under different ones is a
        # different controller wearing the same column name.
        fixed_schedule_windows=sorted(FIXED_SCHEDULE_WINDOWS[tariff].items()),
        rule_params=sorted((k, sorted(v.items()))
                           for k, v in RULE_PARAMS.get(tariff, {}).items()),
        # A DIGEST, not the settings themselves: the forecaster's configuration
        # is already recorded beside the forecasts it produced, and duplicating
        # it here would mean two copies that can disagree. It cannot be dropped
        # altogether though -- a changed forecaster changes the answer, so
        # without this a stale result would be resumed into.
        forecaster_digest=config_digest(
            EnergyForecaster(forecaster_params_con,
                             forecaster_params_gen).config()),
    )
    frames = load_study_frames(
        file_path, H=H, delta_t=delta_t, n_train=n_train, n_sim=n_sim,
        start_ts=start_ts, smp_source=smp_source, verbose=True)
    df_all_kwh = frames["df_all_kwh"]
    df_train, df_sim = frames["df_train"], frames["df_sim"]
    df_ctrl, df_ctrl_kwh = frames["df_ctrl"], frames["df_ctrl_kwh"]

    # Both tariff calendars are evaluated against the DATA's own dates and the
    # data's own hemisphere. Slovenian public holidays on an Australian load
    # profile mark the wrong days non-working (only 5 of the 11 NSW and 14 SI
    # dates coincide in 2013), and the northern high season would put the winter
    # network peak on the Australian summer.
    #
    # `local_timezone` is the third piece of the same decision, and the one that
    # was missing: the CLOCK the blocks, the local days and every clock rule are
    # read on. It was hard-wired to Europe/Ljubljana inside `si_cas` while the
    # AU tariff priced itself on Australia/Sydney, so on the Ausgrid arm a rule
    # and the bill it was scored against sat ~9 hours apart on the same
    # interval. "naive" is right for these profiles for the reason set out in
    # F10 on TariffCalculator: the stamps are already local NSW wall-clock, DST
    # and all, so both the rules and the tariff read the hour off the stamp and
    # neither converts. ONE clock, and it is the household's own.
    _si_cas.nastavi_koledar(drzava=holiday_country, podrocje=holiday_subdiv,
                            visja_sezona_meseci=set(high_season_months),
                            casovni_pas=local_timezone)
    TariffCalculator.HOLIDAY_COUNTRY = holiday_country
    TariffCalculator.HOLIDAY_SUBDIV = holiday_subdiv
    TariffCalculator.LOCAL_TZ = None if local_timezone == "naive" else local_timezone

    # ONE environment for the whole arm, shared by the MILP, every rule and the
    # settlement. There used to be two -- `rate_env` on df_sim for the rates and
    # a second one on df_ctrl inside `run_baseline` -- which is two batteries and
    # two contracts for one household, and nothing forced them to agree.
    #
    # Built on the kWh frame (A7) and put on the AC-symmetric envelope, so every
    # controller in this arm drives the same battery.
    env = align_envelope(
        build_study_env(
            df_ctrl_kwh, battery_cap=battery_cap, soc_min_pct=soc_min_pct,
            soc_max_pct=soc_max_pct, p_max=p_max, eff=eff, delta_t=delta_t, H=H,
            cycle_cost_eur_per_efc=cycle_cost_eur_per_efc,
        ),
        p_max, eff, delta_t,
    )
    rates = build_rate_vectors(tariff, env, df_ctrl.index,
                               df_ctrl["SMP"].values, int(round(delta_t * 60)))
    # The one evaluator this arm prices every controller through -- rules and
    # MILP alike. See section 3e.
    settle = build_settlement(tariff, env, rates, delta_t)
    print(f"Tariff: {tariff} | import rate "
          f"{np.min(rates[0]):.4f}..{np.max(rates[0]):.4f} EUR/kWh")

    # The checkpoint gate sits HERE rather than at the top of the function. The
    # work above it is a CSV read and two rate vectors -- seconds -- while the
    # forecasts and the ~35k LP solves below it are the hours. Paying those
    # seconds buys the backfill below, which is what stops a newly added
    # forecast-free baseline from invalidating every checkpoint on disk.
    _bat = dict(battery_cap=battery_cap, soc_min_pct=soc_min_pct,
                soc_max_pct=soc_max_pct, p_max=p_max, eff=eff)
    cached = read_checkpoint(out_dir, cfg)
    if cached is not None:
        missing = [pol.name for pol in rule_roster(tariff)
                   if f"cost_{pol.name}" not in cached]
        if not missing:
            print("  [checkpoint] already complete under this configuration; skipping")
            # `run_status`: this arm was READ, not executed. Without it the
            # sweep cannot tell the two apart and reports a directory of cache
            # hits as work -- "930/930 runs succeeded in 1.2 min on 10 workers",
            # fifteen cells above a runtime table that says the same sweep cost
            # 22.5 h of household-arm time. Both numbers were right; nothing
            # said they were measuring different things.
            #
            # It is NOT written into the checkpoint -- it is a property of one
            # invocation, not of the result -- so it cannot invalidate one.
            return {"dataset": dataset_name, **cached, "run_status": "cached"}
        # A checkpoint written before one of these rules existed. They need no
        # forecast and no LP, so they are recomputed in seconds and merged rather
        # than discarding MILP results that are still perfectly valid. The tag
        # guard exists to stop rows produced under DIFFERENT RULES from mixing;
        # adding a controller changes no rule that produced the rest.
        print(f"  [checkpoint] backfilling {', '.join(missing)} "
              f"(forecast-free, no LP); MILP results reused")
        extra, _ = run_rules(env, settle, tariff, only=missing,
                             n_steps=n_sim * H, rates=rates,
                             soc_init_kwh=soc_init - battery_cap * soc_min_pct)
        cached = {**cached, **extra}
        write_checkpoint(out_dir, cfg, cached)
        # The rules were recomputed; the MILP -- the expensive part -- was not.
        return {"dataset": dataset_name, **cached, "run_status": "backfilled"}

    forecaster = EnergyForecaster(forecaster_params_con, forecaster_params_gen)
    fc_table, fc_cache_path = load_or_build_forecasts(
        dataset_name, df_train, df_ctrl, H, "30min", forecaster,
        cache_dir=forecast_cache_dir, refit_every_days=refit_every_days,
        kind=forecaster_kind, history=df_train,
    )

    # `add_household_physics`, not the hand-rolled MILPScheduler. b53def7 added
    # this adapter and then never instantiated it, so the swap it announced was
    # inert and the MILP kept solving its own copy of the battery. The study now
    # has one battery model instead of two that can drift, and
    # `test_milp_parity` holds the old one to it: `parity=True` still reproduces
    # the hand-rolled model to 3.0e-9 EUR over three daily solves. It is the
    # checked REFERENCE rather than the default -- see `milp_exclusivity` below
    # for what the study actually solves and what that was measured against.
    # `milp_exclusivity` picks how simultaneous charge and discharge is
    # forbidden, and is the ONLY thing it changes -- the battery, the spill
    # setting and the metering bounds stay where parity puts them.
    #
    #   "auto"      THE DEFAULT. Per solve: the LP where it is provably
    #               equivalent, the binaries where they are load-bearing. See
    #               `UpstreamMILPScheduler.solve` for the rule and why.
    #   "binary"    always one binary per interval. The model the study used to
    #               solve, kept runnable and one flag away.
    #
    # The binaries are NOT redundant in general: where the delivered import rate
    # is negative they are the only thing stopping the household from charging
    # and discharging at once to burn energy it is paid to draw. They ARE
    # inactive wherever the import rate is positive, which on this study's price
    # series is 17,567 intervals of 17,568 on SI and all of them on AU -- so
    # "auto" spends the binaries only where they do work.
    #
    # What the LP buys, measured on the positive-rate majority:
    #   4,800 solves, both tariffs, both horizons, 5 households
    #       same optimum to 8.4e-7 EUR, 4x faster.
    #   6 full simulated years, both tariffs, 3 households
    #       worst annual cost delta 2.4e-4 RELATIVE, forecast-error columns
    #       identical to the bit, 5.6x faster end to end (20.30 -> 3.52 ms per
    #       solve).
    #
    # Going through `parity=False` would also switch curtailment and the
    # metering bounds on, which is three model changes wearing one flag, so the
    # inverter branch is built explicitly instead.
    if milp_exclusivity == "auto":
        scheduler = UpstreamMILPScheduler(
            env,
            battery_cap=battery_cap, soc_min_pct=soc_min_pct,
            soc_max_pct=soc_max_pct, p_max=p_max, eff=eff, delta_t=delta_t,
            parity=False, exclusivity="auto",
            allow_spill=False,        # as parity: no curtailment
            metering_bounds=True,     # the LP branch needs them to stay bounded
        )
        # parity=False skips the envelope check, and that check is the only
        # thing standing between this and the MILP quietly driving a battery
        # 5 % larger than the rules do.
        _step = p_max * delta_t
        _want = (_step * eff, _step / eff)
        _got = (float(env.max_charge_kwh), float(env.max_discharge_kwh))
        if max(abs(a - b) for a, b in zip(_want, _got)) > 1e-9:
            raise ValueError(
                f"milp_exclusivity='auto' needs the same AC-symmetric "
                f"envelope parity does: expected {_want[0]:.6f}/{_want[1]:.6f}, "
                f"got {_got[0]:.6f}/{_got[1]:.6f}")
    elif milp_exclusivity == "binary":
        scheduler = UpstreamMILPScheduler(
            env,
            battery_cap=battery_cap,
            soc_min_pct=soc_min_pct,
            soc_max_pct=soc_max_pct,
            p_max=p_max,
            eff=eff,
            delta_t=delta_t,
            parity=milp_parity,
        )
    else:
        # "inverter" -- always the pure LP -- used to be selectable here and is
        # deliberately not any more. It is unsafe the moment a price series goes
        # negative (see UpstreamMILPScheduler.solve), it was measurably no faster
        # than "auto" on any series that does not, and the only thing it could do
        # for a caller was silently produce a dispatch that destroys energy.
        # "auto" already spends the LP everywhere it is provably equivalent.
        raise ValueError(
            f"milp_exclusivity must be 'auto' (LP where provably equivalent, "
            f"binaries where they bind) or 'binary' (always binaries, the "
            f"parity model); got {milp_exclusivity!r}")

    print("\n--- Reactive mode (consumption + generation via Prophet) ---")
    ctrl_fc = ReactiveController(
        scheduler=scheduler,
        forecaster=fc_table,
        real_data=df_ctrl,
        soc_init=soc_init,
        horizon_steps=control_horizon,
        steps_per_day=H,
        reoptimize_every=1,
        freq="30min",
        rate_vectors=rates,
        leak_current_interval=leak_current_interval,
    )
    df_fc = ctrl_fc.run(num_days=n_sim, use_forecast=True)
    print(f"Simulated steps: {len(df_fc)} | Reopt.: {df_fc['Reoptimized'].sum()}")

    # The SCORED window, not the environment's span. `env` is built on df_ctrl,
    # which carries one extra horizon of lookahead so a 24 h horizon is not
    # truncated over the final day while an 11 h one is -- but only df_sim is
    # ever billed. Building the signal bundle over df_ctrl ran every rule for one
    # day longer than the MILP and charged it for that day: the no-battery
    # reference came out 35.42 EUR through the rules against 34.12 through the
    # MILP path, which is a difference in window length wearing the costume of a
    # difference in control.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The ARM's rates, not the environment's. See `build_signals`: derived
        # from `env.pricing_scheme` they are Slovenian on both arms, and every
        # price rule on AU was then choosing its intervals against a tariff it
        # was not billed under.
        sig = rbc.build_signals(env, n_steps=len(df_sim), rates=rates)

    # --- Everything that never reads a forecast, in one cached block ---------
    #
    # The oracle reads realised data and every rule reads the meter, so for a
    # fixed (household, window, battery, tariff, calendar, horizon, solver) all
    # of this is the same in every arm. Solved once, then read by the other arms
    # that share the key -- see section 2c.
    print(f"\n--- Oracle mode (all real data) + rule-based controllers ({tariff}) ---")

    def _solve_forecast_blind():
        ctrl_pk = ReactiveController(
            scheduler=scheduler,
            forecaster=fc_table,   # unused by the oracle arm, which reads _real_slice
            real_data=df_ctrl,
            soc_init=soc_init,
            horizon_steps=control_horizon,
            steps_per_day=H,
            reoptimize_every=1,
            freq="30min",
            rate_vectors=rates,
        )
        _df_pk = ctrl_pk.run(num_days=n_sim, use_forecast=False)
        # The deployable alternative, on realized data deliberately: an inverter
        # measures the surplus in front of it, it does not forecast one, so
        # giving a rule the realized interval is physical fidelity rather than
        # the look-ahead it would be for a planner. There is correspondingly no
        # "forecast" variant of any of them.
        #
        # `soc_init` is absolute kWh in the controller's frame
        # (soc_min..soc_max); the rules work in upstream's stored frame
        # (0..capacity). Without the offset the rules would start at
        # SOC_FRACTION * capacity = 3.5 kWh stored while the MILP started at
        # 4.0, which is not the same experiment -- and the half kWh difference
        # comes out as a saving.
        _m, _rows = run_rules(
            env, settle, tariff, signals=sig,
            soc_init_kwh=soc_init - battery_cap * soc_min_pct)

        # The whole-period optimum -- the denominator, see section 3d. Solved
        # here so it lands in the same forecast-blind cache as the oracle arm
        # and is paid for once per household rather than once per arm.
        _full = solve_full_period(
            env, rates, tariff, n_steps=len(sig.import_rate),
            soc_init_kwh=soc_init, delta_t=delta_t,
            soc_min_kwh=battery_cap * soc_min_pct,
            closeout_rate=float(np.mean(sig.import_rate)))
        _df_full = pd.DataFrame(
            {"Charge_kW": _full["x_ch"], "Discharge_kW": _full["x_dis"],
             "Buy_kW": _full["p_buy"], "Sell_kW": _full["p_sell"],
             "SoC_kWh": _full["soc_plan"]},
            index=df_ctrl.index[:len(sig.import_rate)])
        _df_full.index.name = "Timestamp"
        _df_full.attrs.update({k: _full[k] for k in
                               ("objective", "wear_eur", "n_binary_intervals",
                                "runtime_s")})
        return _df_pk, _m, _rows, _df_full

    df_pk, rule_metrics, rule_rows, df_full = load_or_build_oracle(
        dataset_name, cfg, _solve_forecast_blind, cache_dir=oracle_cache_dir)

    # Every controller re-priced through the arm's ONE evaluator, the two MILP
    # arms included: ReactiveController accumulates a running cost as it goes,
    # but that is a convenience, not the bill, and it drops the standing charge.
    kpi_table, kpi_raw = KPITracker.compare_three(df_fc, df_pk, delta_t)
    nominal_kwh = float(battery_cap)
    for _name, _frame in (("oracle", df_pk), ("prophet", df_fc),
                          ("milp_full", df_full)):
        _net = (_frame["Buy_kW"] - _frame["Sell_kW"]).to_numpy() * delta_t
        _s = settle_trajectory(env, _net, settle, sig, soc_start=soc_init,
                               soc_end=float(_frame["SoC_kWh"].iloc[-1]))
        # OVERWRITE what KPITracker put here. Its cost_prophet / cost_oracle are
        # the controller's own running total: buy x rate - sell x rate, with no
        # standing charge and -- on SI -- no capacity charge at all. Every rule's
        # cost_<rule> comes from the settlement and carries both. Leaving the two
        # side by side under different names is exactly the unfair comparison
        # this study set out to remove, and it flatters the MILP on SI by the
        # whole excess-power charge.
        # `cost_<controller>` is the comparison figure and the ONLY key allowed
        # to start with "cost_": controller_columns reads that prefix as the
        # controller namespace, so a component named cost_eur_oracle would show
        # up in every table as a controller called "eur_oracle".
        kpi_raw[f"cost_{_name}"] = _s["Cost_EUR_Closed"]
        for _k, _prefix in (("Cost_EUR", "gross"), ("Energy_EUR", "energy"),
                            ("Power_EUR", "power"), ("Fixed_EUR", "fixed"),
                            ("Terminal_SOC_Adj_EUR", "termadj"),
                            ("Peak_Import_kW", "peakkw")):
            kpi_raw[f"{_prefix}_{_name}"] = _s[_k]
        # Cycles, on the same convention the rules report: energy through the
        # STORE against the NAMEPLATE pack, which is what a cycle rating is
        # quoted against.
        _stored = (float(_frame["Charge_kW"].sum()) * delta_t * eff
                   + float(_frame["Discharge_kW"].sum()) * delta_t / eff)
        kpi_raw[f"efc_{_name}"] = _stored / (2.0 * nominal_kwh)

    kpi_raw.update(rule_metrics)
    for _r in rule_rows:
        print(f"  {_r['controller']:30s} {_r['Cost_EUR_Closed']:9.2f} EUR"
              f"   EFC {_r['Equivalent_Full_Cycles']:6.1f}"
              f"   peak {_r['Peak_Import_kW']:5.2f} kW")
    print(f"vs no battery {kpi_raw['cost_no_battery']:.2f} "
          f"vs MPC oracle {kpi_raw['cost_oracle']:.2f} "
          f"vs whole-period optimum {kpi_raw['cost_milp_full']:.2f} EUR")
    kpi_raw.update(full_period_bound_check(
        kpi_raw, tariff, cycle_cost_eur_per_efc,
        reporting_cycle_cost_eur_per_efc=cycle_cost_reporting_eur_per_efc,
        bound_is_exact=(tariff != "SI"
                        or _agreed_power_is_endogenous_in_lp(env, len(sig.import_rate)))))
    # Forecast quality alongside the cost, because a forecasting-in-the-loop
    # result is not interpretable without it.
    fc_err = forecast_error_metrics(fc_table.table, df_sim, H, history=df_train)
    # `fc_table` covers df_ctrl; forecast_error_metrics intersects on df_sim's
    # own index, so the lookahead tail is forecast but never scored.
    kpi_raw = {**kpi_raw, **fc_err}
    print(f"Forecast day-ahead nMAE: gen {fc_err['gen_nmae']:.3f} | "
          f"con {fc_err['con_nmae']:.3f} | skill vs seasonal-naive: "
          f"gen {fc_err['gen_skill_vs_naive']:+.3f} con {fc_err['con_skill_vs_naive']:+.3f}")
    print(kpi_table.to_string())

    # Outputs, all prefixed with the dataset name for easy identification.
    # DATA only: no figure is written from inside a batch run. The notebook
    # draws every figure from these CSVs through Plotting_Functions, which is
    # what lets a chart be restyled without re-solving a household-year.
    kpi_csv_path = os.path.join(out_dir, f"kpi_results_{dataset_name}.csv")
    # Gzipped. These two are 98 % of what the sweep writes -- 5 MB per run, 1.7 GB
    # over the full sweep -- and `collect_results` reads NEITHER: the checkpoint
    # is the source of truth, and the notebook draws every figure from that.
    # They are kept because a finer analysis needs the per-interval trajectory,
    # but there is no reason to keep them uncompressed at 12x the size.
    # pandas infers the codec from the suffix on the way back in, so
    # `pd.read_csv(path)` still just works.
    df_fc_csv_path = os.path.join(out_dir, f"df_fc_{dataset_name}.csv.gz")
    df_pk_csv_path = os.path.join(out_dir, f"df_pk_{dataset_name}.csv.gz")

    # F2 - the previous run lost `Ausgrid 138` (the FIRST id in DATASET_IDS)
    # here, with "Cannot save file into a non-existent directory", and run_all
    # swallowed it: the published summary and figures cover 29 of 30 sites.
    # Re-assert the directory immediately before the writes.
    os.makedirs(out_dir, exist_ok=True)
    kpi_table.to_csv(kpi_csv_path, encoding="utf-8-sig")
    # raw time series also kept, useful for finer analysis later
    df_fc.to_csv(df_fc_csv_path, encoding="utf-8-sig", compression="gzip")
    df_pk.to_csv(df_pk_csv_path, encoding="utf-8-sig", compression="gzip")
    pd.DataFrame(rule_rows).to_csv(
        os.path.join(out_dir, f"rules_{dataset_name}.csv"),
        index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {out_dir}/")

    # Everything above, including the forecast fit, the ~35k LP solves and the
    # CSVs just written. Recorded under contention when the sweep runs in
    # parallel -- `arm_runtimes` says so, because a per-arm time measured with
    # ten workers on the machine is not the time the arm takes alone.
    kpi_raw = {**kpi_raw, "runtime_s": time.perf_counter() - _t_start,
               # What the cycles this run spent are WORTH, as opposed to what
               # the objective was charged for them. Equal on every arm but the
               # no-degradation ones; `summarize` bills `wear_eur` at this and
               # never at the dispatch rate. See the note where it is resolved.
               "cycle_cost_reporting_eur_per_efc":
                   cycle_cost_reporting_eur_per_efc}
    write_checkpoint(out_dir, cfg, kpi_raw)

    return {"dataset": dataset_name, **kpi_raw, "run_status": "computed"}


# =====================================================================
# 8 — Batch runner: chains through all datasets in a folder
# =====================================================================

def run_all(data_dir: str,
            output_root: str = "results",
            pattern: str = "*.csv",
            dataset_ids: list | None = None,
            filename_template: str = "Ausgrid {id}.csv",
            **pipeline_kwargs) -> pd.DataFrame:
    """
    Chains run_pipeline_for_file() over the datasets.

    Two modes:
    - dataset_ids provided (list of IDs, e.g. [138, 127, 65, ...]):
      builds file paths via filename_template.format(id=...) inside
      data_dir, in the EXACT order of the list. This is the mode to use
      when the numbers are not contiguous / there are other
      files in the folder to ignore.
    - dataset_ids=None: falls back to glob.glob(data_dir/pattern), sorted
      alphabetically.

    If a dataset fails, the error is logged and we move to the
    next one (no loss of an entire night's computation for one corrupted file).

    Returns a summary DataFrame (one row per dataset), also
    saved to output_root/summary_all_datasets.csv
    """
    os.makedirs(output_root, exist_ok=True)

    if dataset_ids is not None:
        files = [os.path.join(data_dir, filename_template.format(id=i)) for i in dataset_ids]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            print("!!! Files not found (check name/path):")
            for m in missing:
                print(f"    - {m}")
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, pattern)))

    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    print(f"{len(files)} datasets detected")

    summary_rows = []
    failed = []

    for i, f in enumerate(files, 1):
        print(f"\n\n########## [{i}/{len(files)}] {os.path.basename(f)} ##########")
        try:
            metrics = run_pipeline_for_file(f, output_root=output_root, **pipeline_kwargs)
            summary_rows.append(metrics)
        except Exception as e:
            print(f"!!! ERROR on {f}: {e}")
            traceback.print_exc()
            failed.append({"dataset": os.path.basename(f), "error": str(e)})
            continue

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["regret_prophet"] = (
            summary_df["cost_prophet"] - summary_df["cost_oracle"])

        # A percentage needs a positive denominator, and here it is not
        # guaranteed to have one. A household that exports more than it imports
        # has cost_no_battery <= 0, and dividing by it flips the sign, so a site
        # that saves money reads as one that loses it -- a real result, on a real
        # figure, produced by arithmetic rather than by a battery. figure.py
        # warned about this after the fact; the fix belongs here, where the
        # column is made.
        baseline = summary_df["cost_no_battery"]
        usable = baseline > 1e-9
        for col, cost in (("oracle", "cost_oracle"), ("prophet", "cost_prophet")):
            summary_df[f"gain_{col}_vs_no_battery_pct"] = np.where(
                usable, 100.0 * (baseline - summary_df[cost]) / baseline.where(usable),
                np.nan,
            )
        summary_df["baseline_positive"] = usable
        if not usable.all():
            bad = summary_df.loc[~usable, "dataset"].tolist()
            print(f"\n! {len(bad)} site(s) have a non-positive no-battery cost, so a "
                  f"saving PERCENTAGE is undefined for them: {', '.join(map(str, bad))}")
            print("  Their euro columns are still valid; the _pct columns are NaN.")

        summary_df = summary_df.set_index("dataset")

    summary_path = os.path.join(output_root, "summary_all_datasets.csv")
    summary_df.to_csv(summary_path, encoding="utf-8-sig")
    print(f"\n\n=== DONE: {len(summary_rows)}/{len(files)} datasets succeeded ===")
    print(f"    (re-running is cheap: a finished dataset is skipped via its "
          f"checkpoint, and forecasts are reused from {FORECAST_CACHE_DIR}/)")
    print(f"Global summary: {summary_path}")

    if failed:
        failed_path = os.path.join(output_root, "failed_datasets.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False, encoding="utf-8-sig")
        # Loud, last, and impossible to scroll past: a partial summary that
        # looks complete is how N=29 got reported as N=30.
        print("\n" + "!" * 70)
        print(f"!!! {len(failed)} of {len(files)} DATASET(S) FAILED - SUMMARY IS INCOMPLETE")
        for f in failed:
            print(f"!!!   {f['dataset']}: {f['error']}")
        print(f"!!! details: {failed_path}")
        print("!" * 70)

    return summary_df


# =====================================================================
# 8a — Which households, and why those
# =====================================================================
#
# The 30 sites are not a hand-picked list, though they were written down as one.
# They are the centroid-nearest member of each of 30 k-means clusters over all
# 300 Ausgrid households -- `rank_in_cluster == 1` in the sweep that
# `Clustering/cluster_households.py` runs. One household per cluster is a
# deliberate sample that spans the consumption shapes present in the population,
# and it is a far better story than "30 ids", so it is derived here rather than
# transcribed: the rule is then executable, and a reviewer can check it.

CLUSTERING_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Clustering", "Ausgrid", "user_ids_sorted_by_cluster_30.csv",
)


def study_units(path: str | None = None, k: int = 30) -> pd.DataFrame:
    """The study households, one per cluster, with the cluster they represent.

    Returns a frame indexed by dataset id carrying `cluster` and
    `dist_to_centroid`, in cluster order. Those two columns travel with the
    results so "which kind of household is this controller bad at" is a question
    the study can answer -- which, with a flat list of ids, it could not.
    """
    path = CLUSTERING_CSV if path is None else path
    if not os.path.isfile(path):
        # Said here rather than let out as a pandas FileNotFoundError twenty
        # frames down. This is the FIRST thing the notebook calls and the first
        # thing `collect_results` calls, so a missing file breaks every cell at
        # once with a traceback that names neither the study nor the fix -- and
        # it has already happened twice, on moves of the clustering output.
        raise FileNotFoundError(
            f"the clustering that defines the study households is not at\n"
            f"  {path}\n"
            f"Every household id, and therefore every result, is derived from "
            f"it. Point `hs.CLUSTERING_CSV` at the file or pass `path=`; it is "
            f"`user_ids_sorted_by_cluster_{k}.csv`, which "
            f"`Clustering/cluster_households.py` writes."
        )
    df = pd.read_csv(path)
    picked = df[df["rank_in_cluster"] == 1].sort_values("cluster")
    if len(picked) != k:
        raise ValueError(
            f"{path}: expected {k} clusters with a rank-1 member, found {len(picked)}"
        )
    out = picked.assign(
        dataset_id=picked["user_id"].str.removeprefix("user_").astype(int)
    ).set_index("dataset_id")[["cluster", "dist_to_centroid"]]
    return out


def dataset_ids(path: str | None = None, k: int = 30) -> list:
    """Just the ids, in cluster order -- what `run_all` takes."""
    return study_units(path, k).index.tolist()


# =====================================================================
# 8b — The study arms
# =====================================================================
#
# The axes, and why each one is an axis:
#
#   tariff   AU / SI          the price signal a battery is answering
#   horizon  H24 / H11        24 h assumes a whole day of published prices;
#                             11 h is what a day-ahead market guarantees at its
#                             worst moment (SIPX publishes D+1 at 12:45, so the
#                             known window runs 11.25 h .. 35.25 h). H11 is the
#                             strictly deployable arm.
#   leak     on / off         the current-interval look-ahead, kept runnable so
#                             the published numbers remain reproducible.
#
# Forecasts are keyed on none of these, so the whole sweep costs one Prophet fit
# per household.

# What `tune_prophet` chose, over 8 households and 90 simulated days, under the
# same 30-day refit the arms run. Both are the same finding from two directions:
# the trend term was the problem. A household has no trend over a month, so the
# freedom to fit one is variance the forecast pays for and never earns back --
# on the roof, removing it outright (`growth="flat"`) is worth +0.13 skill.
#
# It is not enough. Tuned, Prophet still scores -0.31 on generation against
# yesterday's 0.00, and its consumption gain over the default is +0.01. Recorded
# here so the arms are reproducible, and so "we only tried the defaults" is not
# available as an explanation of the result.
#
# A round-2 search over combinations of these found nothing further: with
# `growth="flat"` the generation seasonality mode stops mattering at all, and
# `seasonality_prior_scale` moved consumption by +0.001, which is noise.
TUNED_PARAMS_CON = {"changepoint_prior_scale": 0.001}
TUNED_PARAMS_GEN = {"growth": "flat"}

# Which Prophet settings each PROPHET_KIND means, as (consumption, generation).
# It lives here rather than beside `PROPHET_KINDS` because it names the tuned
# tables above; `forecast_benchmark` resolves it at call time. The arms take
# their params from their own `STUDY_ARMS` entry -- this is the same pair, in
# the one place a caller holding only the KIND string can look it up, so a
# screen and an arm cannot score two different models under one name.
PROPHET_KIND_PARAMS = {
    "prophet":       (None, None),
    "prophet_tuned": (TUNED_PARAMS_CON, TUNED_PARAMS_GEN),
}

STUDY_ARMS = [
    {"name": "AU_H24",          "tariff": "AU", "control_horizon": 48},
    {"name": "AU_H11",          "tariff": "AU", "control_horizon": 22},
    {"name": "SI_H24",          "tariff": "SI", "control_horizon": 48},
    {"name": "SI_H11",          "tariff": "SI", "control_horizon": 22},
    {"name": "AU_H24_leaked",   "tariff": "AU", "control_horizon": 48,
     "leak_current_interval": True},
    {"name": "AU_H24_persist",  "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "persistence"},
    {"name": "SI_H24_persist",  "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "persistence"},
    # The forecast-channel arms. A single "with forecast" arm averages the two
    # channels into one number and hides which one the money is in, so these
    # hold consumption at Prophet and vary only the roof. (On the 30-household
    # median Prophet loses to seasonal-naive on BOTH channels -- see the
    # HYBRID_KINDS comment -- which is what the median14 arms below test.)
    {"name": "AU_H24_pvnaive",  "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvnaive"},
    {"name": "SI_H24_pvnaive",  "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvnaive"},
    {"name": "AU_H24_pvtruth",  "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvtruth"},
    {"name": "SI_H24_pvtruth",  "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvtruth"},
    # The fit-free winner, promoted out of `forecast_benchmark`. Screened first
    # over all 30 households and all 7 naive kinds -- 24 s, no LP -- and it took
    # both channels: median skill vs seasonal-naive +0.22 on consumption and
    # +0.11 on generation, against Prophet's -0.10 and -0.17 on the SAME
    # households. These two arms are what turns that into EUR.
    #
    # No hybrid needed: both channels come from one source, so the plain kind
    # already is the "no model anywhere" arm.
    {"name": "AU_H24_median14", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "median14"},
    {"name": "SI_H24_median14", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "median14"},
    # And the roof channel alone, so it reads against pvnaive and pvtruth on the
    # axis those two already define.
    {"name": "AU_H24_pvmedian14", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvmedian14"},
    {"name": "SI_H24_pvmedian14", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvmedian14"},
    # Perfect roof knowledge on top of each load model, so "what is the sun
    # worth" is answered for every one of them rather than for Prophet alone.
    # Each of these pairs with the plain arm above it: the difference IS the
    # value of a perfect PV forecast to that load model.
    {"name": "AU_H24_persist_pvtruth", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "persist_pvtruth"},
    {"name": "SI_H24_persist_pvtruth", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "persist_pvtruth"},
    {"name": "AU_H24_median14_pvtruth", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "median14_pvtruth"},
    {"name": "SI_H24_median14_pvtruth", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "median14_pvtruth"},
    # Prophet given a fair hearing. `forecast_benchmark` found the DEFAULTS
    # behind seasonal-naive, which is a claim about a configuration and not
    # about the model, so these arms carry the settings `tune_prophet` picked.
    {"name": "AU_H24_prophet_tuned", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "prophet_tuned",
     "forecaster_params_con": TUNED_PARAMS_CON,
     "forecaster_params_gen": TUNED_PARAMS_GEN},
    {"name": "SI_H24_prophet_tuned", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "prophet_tuned",
     "forecaster_params_con": TUNED_PARAMS_CON,
     "forecaster_params_gen": TUNED_PARAMS_GEN},
    {"name": "AU_H24_pvtruth_tuned", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvtruth_tuned",
     "forecaster_params_con": TUNED_PARAMS_CON,
     "forecaster_params_gen": TUNED_PARAMS_GEN},
    {"name": "SI_H24_pvtruth_tuned", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvtruth_tuned",
     "forecaster_params_con": TUNED_PARAMS_CON,
     "forecaster_params_gen": TUNED_PARAMS_GEN},
    # The ported baseline-plus-AR method (`HbdForecaster`). It is the first kind
    # in the roster that both fits a model and conditions on the last 24 h, so
    # it is run beside its own ablation rather than alone: `hbd_baseline` is the
    # identical object with the AR stage off, and the gap between the two pairs
    # is the only clean measurement of what that conditioning is worth in euros.
    {"name": "AU_H24_hbd", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "hbd"},
    {"name": "SI_H24_hbd", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "hbd"},
    {"name": "AU_H24_hbd_baseline", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "hbd_baseline"},
    {"name": "SI_H24_hbd_baseline", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "hbd_baseline"},
    # The paper's SECOND stage on this study's own first stage. On the error
    # axis this is the arm that actually beats `median14` -- the faithful port
    # does not -- so it is the one the economic comparison most needs.
    {"name": "AU_H24_hbd_median14", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "hbd_median14"},
    {"name": "SI_H24_hbd_median14", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "hbd_median14"},
    # Both tariffs, because the regret figure only shows a kind that has an arm
    # on every tariff in the panel -- an AU-only arm is computed and then
    # silently dropped from the comparison it was added for.
    {"name": "AU_H24_hbd_median14_pvtruth", "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "hbd_median14_pvtruth"},
    {"name": "SI_H24_hbd_median14_pvtruth", "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "hbd_median14_pvtruth"},

    # THE MILP WITH NO DEGRADATION TERM. `cycle_cost_eur_per_efc = 0` takes the
    # wear shadow price out of the objective (`wear_objective_terms` reads it off
    # the env and contributes nothing at zero), so the three MILP controllers in
    # these arms -- `prophet`, `oracle` and `milp_full` -- optimise the bill
    # alone. Everything else is the H24/H11 spec unchanged, which is what makes
    # the pair a controlled comparison: same battery, same tariff, same forecast,
    # same evaluator, one term removed.
    #
    # It is the ablation the study was missing. Every result above is stated
    # under an objective that already prices pack life, so "the MILP saves less
    # than a price rule on the energy bill" has two possible causes -- the
    # horizon, or the wear term -- and nothing separated them. These arms do.
    #
    # The rules re-run here too and are expected to be IDENTICAL to their H24/H11
    # rows: no rule consults a wear price. That is the control, and a rule that
    # moves between an arm and its no-wear twin is a bug, not a result.
    #
    # The pack still wears. `cycle_cost_reporting_eur_per_efc` stays at the pack
    # price, so `wear_eur`, `saving_net_of_wear`, `saving_total` and every NPV
    # bill these arms for the cycles they spend at the same rate as everyone
    # else. See the note in `summarize`: a solved rate of zero is the one value
    # that is never read back as an accounting rate.
    {"name": "AU_H24_nowear", "tariff": "AU", "control_horizon": 48,
     "cycle_cost_eur_per_efc": 0.0},
    {"name": "SI_H24_nowear", "tariff": "SI", "control_horizon": 48,
     "cycle_cost_eur_per_efc": 0.0},
    {"name": "AU_H11_nowear", "tariff": "AU", "control_horizon": 22,
     "cycle_cost_eur_per_efc": 0.0},
    {"name": "SI_H11_nowear", "tariff": "SI", "control_horizon": 22,
     "cycle_cost_eur_per_efc": 0.0},
]


def forecast_arms(tariff: str, control_horizon: int = 48) -> dict:
    """{forecaster kind: arm name} for one tariff -- the arm a forecast-quality
    figure should read that kind from.

    Diagnostic variants are excluded, and that is the point rather than tidiness.
    `AU_H24_leaked` carries `forecaster_kind == "prophet"` exactly like `AU_H24`,
    so selecting rows by kind alone silently averages the current-interval
    look-ahead into Prophet's number and flatters it by roughly 30 EUR. Reading
    the arm definitions instead makes that impossible to do by accident.
    """
    out = {}
    for a in STUDY_ARMS:
        if a["tariff"] != tariff or a.get("control_horizon") != control_horizon:
            continue
        if a.get("leak_current_interval"):
            continue
        # Same trap, second instance. The no-degradation arms carry no
        # `forecaster_kind` either, so they default to "prophet" exactly like
        # `AU_H24` and would answer "which arm shows me Prophet?" with whichever
        # of the two `setdefault` happened to see first -- i.e. with the roster
        # order, which is not a decision anyone made. They differ in their
        # OBJECTIVE, not their forecast, so they have no business in a
        # forecast-quality comparison at all.
        if "cycle_cost_eur_per_efc" in a:
            continue
        out.setdefault(a.get("forecaster_kind", "prophet"), a["name"])
    return out

# The arm each tariff's comparison is read against, and whose full controller
# roster the "all controllers" figure is drawn from.
REFERENCE_ARM = {"AU": "AU_H24", "SI": "SI_H24"}
ARM_ORDER = [a["name"] for a in STUDY_ARMS]

# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------
# An arm's currency is a property of its tariff and travels with the arm:
# `au_rate_vectors` returns AUD per kWh, `si_rate_vectors` EUR per kWh. Every
# money column downstream -- cost, saving, NPV -- is therefore in whichever of
# the two the ROW's arm bills in, and a figure that puts both on one axis is
# adding AUD to EUR. The unit belongs on the label, so the label has to be able
# to ask for it.
CURRENCY = {"AU": "AUD", "SI": "EUR"}

# How many units of an arm's currency one EUR buys. Needed because
# `Battery_Economics` quotes the pack ONCE, in EUR, and a NPV is that quote
# against a bill: on AU the two are in different money and the quote is the one
# that has to move. 250 EUR/kWh is 406.50 AUD/kWh, so an AU pack is 5691 AUD
# installed against SI's 3500 EUR -- the same pack, said twice.
#
# Worth knowing, because it is NOT applied consistently upstream: the wear
# shadow price the sweep solved under is `cycle_cost_eur_per_efc(10.0)` =
# 0.4167 on BOTH arms, unconverted, so the AU MILP was dispatching against a
# pack it valued at 250 AUD/kWh (154 EUR/kWh) while the NPV here prices it at
# 406.50 AUD/kWh. Converting the shadow price would change the dispatch and
# invalidate every checkpoint, so it is left as it was solved and said out loud
# instead: `wear_eur` and the wear line in the cycles/saving figure are at the
# rate the RUN used, and the lifetime economics are at the honest one.
ARM_CURRENCY_PER_EUR = {"SI": 1.0, "AU": 1.0 / TariffCalculator.EUR_PER_AUD}


def currency(arm_or_tariff: str) -> str:
    """The three-letter code an arm's (or a tariff's) money is in."""
    if arm_or_tariff in CURRENCY:
        return CURRENCY[arm_or_tariff]
    return CURRENCY[arm_tariff(arm_or_tariff)]


def money(arm_or_tariff: str, per: str = "") -> str:
    """An axis unit for a money column: `money("AU", "year") -> "AUD/year"`."""
    return f"{currency(arm_or_tariff)}/{per}" if per else currency(arm_or_tariff)


def arm_tariff(name):
    """Which tariff an arm is on -- the axis its results are comparable along."""
    return next(a["tariff"] for a in STUDY_ARMS if a["name"] == name)


def _run_arms_parallel(data_dir, output_root, dataset_ids, filename_template,
                       arms, n_jobs, kwargs):
    """The sweep, households in parallel. Same results, same checkpoints."""
    from joblib import Parallel, delayed

    if dataset_ids is not None:
        files = [os.path.join(data_dir, filename_template.format(id=i))
                 for i in dataset_ids]
        missing = [f for f in files if not os.path.isfile(f)]
        if missing:
            print("!!! Files not found (check name/path):")
            for m in missing:
                print(f"    - {m}")
        files = [f for f in files if os.path.isfile(f)]
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir}")

    if n_jobs < 0:
        n_jobs = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    n_jobs = min(n_jobs, len(files))
    log_dir = os.path.join(output_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # One thread per worker in every numeric library underneath. HiGHS, BLAS and
    # cmdstan each default to "as many threads as there are cores", so ten
    # workers on fourteen cores would ask for a hundred and forty and spend the
    # difference on contention. The solver is already single-threaded by
    # `make_solver`; this covers everything else.
    for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_var] = "1"

    print(f"\n{'#' * 70}")
    print(f"### PARALLEL SWEEP: {len(arms)} arms x {len(files)} households "
          f"= {len(arms) * len(files)} runs, {n_jobs} workers")
    print(f"### solver {SOLVER_NAME}; per-household logs under {log_dir}/")
    print(f"{'#' * 70}\n", flush=True)

    t0 = datetime.datetime.now()
    batches = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_household_all_arms)(f, output_root, arms, log_dir, kwargs)
        for f in files)
    elapsed = (datetime.datetime.now() - t0).total_seconds()

    rows = [r for batch in batches for r in batch]
    failed = [r for r in rows if "error" in r]
    ok = [r for r in rows if "error" not in r]

    allrows = pd.DataFrame(ok)
    if not allrows.empty:
        allrows = allrows.sort_values(
            ["arm", "dataset"], key=lambda s: s.map(ARM_ORDER.index)
            if s.name == "arm" else s)
        path = sweep_summary_path(output_root, arms)
        allrows.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n{len(arms)} arm(s): {path}")

    # WHAT the elapsed time bought. A sweep whose checkpoints are all valid
    # finishes in a minute because it executed nothing, and reporting that as
    # "930/930 runs succeeded in 1.2 min" reads as a machine forty times faster
    # than the one `arm_runtimes` measured. Split the count so the wall clock
    # can only ever be read against the work it actually covers.
    status = collections.Counter(r.get("run_status", "computed") for r in ok)
    print(f"\n=== {len(ok)}/{len(rows)} runs succeeded in "
          f"{elapsed / 60:.1f} min on {n_jobs} workers ===")
    print("    " + ", ".join(
        f"{status[k]} {lbl}" for k, lbl in
        (("computed", "computed"), ("backfilled", "backfilled (rules only)"),
         ("cached", "served from checkpoints")) if status[k]))
    if not status["computed"] and not status["backfilled"]:
        print(f"    Nothing was executed: every run was already on disk under "
              f"this configuration, so the {elapsed / 60:.1f} min above is the "
              f"cost of READING {len(ok)} checkpoints, not of producing them.\n"
              f"    What producing them cost is `arm_runtimes`.")
    if failed:
        # Loud, last, and impossible to scroll past: a partial summary that
        # looks complete is how N=29 got reported as N=30.
        fp = os.path.join(output_root, "failed_runs.csv")
        pd.DataFrame(failed).to_csv(fp, index=False, encoding="utf-8-sig")
        print("\n" + "!" * 70)
        print(f"!!! {len(failed)} RUN(S) FAILED - THE SWEEP IS INCOMPLETE")
        for r in failed[:20]:
            print(f"!!!   {r['arm']:18s} {r['dataset']:14s} {r['error']}")
        if len(failed) > 20:
            print(f"!!!   ... and {len(failed) - 20} more")
        print(f"!!! details: {fp}   per-household logs: {log_dir}/")
        print("!" * 70)
    if not ok:
        raise RuntimeError("no arm produced any result")
    return allrows


def _household_all_arms(file_path, output_root, arms, log_dir, kwargs):
    """Every arm for ONE household, in one process. The unit of parallelism.

    The household, not the (arm, household) pair, is the unit on purpose. Both
    caches are keyed per household -- Prophet's table and the forecast-blind
    oracle -- so giving one worker every arm of one household means:

      no races      two workers never touch the same cache key, so nothing has
                    to be locked and no pre-warm pass is needed;
      full reuse    the worker fits Prophet once and solves each of its (tariff,
                    horizon) oracles once, then reads them for the remaining
                    arms, exactly as the serial sweep does.

    Splitting by (arm, household) instead would put eleven workers on one
    household's cache at the same moment: each would miss, each would fit its
    own Prophet, and the cheapest work in the study would be done eleven times.
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(log_dir, exist_ok=True)
    rows = []
    # One log per household. Ten workers printing to one stdout interleaves into
    # something unreadable, and these prints are the run's audit trail -- the
    # cache hits, the forecast skill, every rule's bill.
    # buffering=1 -- line buffered. Without it Python holds 8 KB before touching
    # the disk, and a log that only appears when the worker exits is no use
    # during the hours it is running: `tail -f` shows an empty file for the whole
    # sweep and there is no way to tell progress from a hang.
    with open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8",
              buffering=1) as fh:
        with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            for arm in arms:
                spec = {k: v for k, v in arm.items() if k != "name"}
                try:
                    metrics = run_pipeline_for_file(
                        file_path,
                        output_root=os.path.join(output_root, arm["name"]),
                        **{**kwargs, **spec})
                    rows.append({"arm": arm["name"], **metrics})
                except Exception as exc:          # one arm failing is not the run failing
                    traceback.print_exc()
                    rows.append({"arm": arm["name"], "dataset": name,
                                 "error": f"{type(exc).__name__}: {exc}"})
    return rows


def sweep_summary_path(output_root: str, arms) -> str:
    """Where `run_arms` writes its roll-up, named for the arms it actually swept.

    "summary_all_arms.csv" was true while one notebook swept the whole roster.
    The sweep is now partitioned across three notebooks, each passing its own
    `arms=`, and all three writing that one filename means the last one to run
    leaves a file whose name claims every arm and whose contents are one track's
    -- last-writer-wins on a name that lies. Nothing READS it (`collect_results`
    takes the checkpoints as the source of truth, and says so), so this is about
    a person opening the file, which is the only thing it is for.

    The full roster keeps the old name so an existing file is still overwritten
    rather than orphaned beside a new one.
    """
    names = [a["name"] for a in arms]
    if set(names) == {a["name"] for a in STUDY_ARMS}:
        return os.path.join(output_root, "summary_all_arms.csv")
    # Digested, not joined: 21 arm names make a 400-character filename. The arm
    # count is in the name so the file is recognisable without opening it.
    tag = config_digest({"arms": sorted(names)})[:8]
    return os.path.join(output_root, f"summary_{len(names)}_arms_{tag}.csv")


def run_arms(data_dir, output_root="results", dataset_ids=None,
             filename_template="Ausgrid {id}.csv", arms=None, n_jobs=1, **kwargs):
    """Every arm over every dataset, into one long frame.

    Each (arm, dataset) is checkpointed independently, so an interrupted sweep
    resumes where it stopped rather than from the beginning.

    `n_jobs > 1` runs households in parallel, one process each. PROCESSES, not
    threads: `run_pipeline_for_file` sets the tariff calendar through module and
    class globals (`_si_cas.nastavi_koledar`, `TariffCalculator.HOLIDAY_*`),
    which is safe when each worker owns its own interpreter and a data race that
    silently marks the wrong days non-working when they share one.
    """
    arms = arms or STUDY_ARMS
    if n_jobs != 1:
        return _run_arms_parallel(data_dir, output_root, dataset_ids,
                                  filename_template, arms, n_jobs, kwargs)
    rows = []
    for arm in arms:
        spec = {k: v for k, v in arm.items() if k != "name"}
        print(f"\n{'#' * 70}\n### ARM {arm['name']}: {spec}\n{'#' * 70}")
        summary = run_all(
            data_dir, output_root=os.path.join(output_root, arm["name"]),
            dataset_ids=dataset_ids, filename_template=filename_template,
            **{**kwargs, **spec},
        )
        if not summary.empty:
            summary = summary.reset_index()
            summary.insert(0, "arm", arm["name"])
            rows.append(summary)
    if not rows:
        raise RuntimeError("no arm produced any result")
    allrows = pd.concat(rows, ignore_index=True)
    path = sweep_summary_path(output_root, arms)
    allrows.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n{len(arms)} arm(s): {path}")
    return allrows


# =====================================================================
# 9 — Reading the sweep back, and the statistics the article quotes
# =====================================================================

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_local")
KEY_COLUMNS = ["arm", "dataset"]


def collect_results(output_root=None, arms=None) -> pd.DataFrame:
    """Every finished (arm, household) as one long frame, read from checkpoints.

    The checkpoint is the source of truth rather than summary_all_arms.csv: it
    carries the config each row was produced under, so a row computed under
    superseded rules can be DROPPED rather than silently mixed in with current
    ones. Rows are tagged with the arm's tariff and the cluster the household
    was drawn to represent, so both can be grouped by without another join.
    """
    output_root = RESULTS_DIR if output_root is None else output_root
    units = study_units()
    rows = []
    pattern = os.path.join(output_root, "*", "*", "checkpoint.json")
    for path in sorted(glob.glob(pattern)):
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        dataset = os.path.basename(os.path.dirname(path))
        if arms is not None and arm not in arms:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
        except (ValueError, OSError) as exc:
            print(f"  ! unreadable checkpoint {path}: {exc}")
            continue
        cfg = saved.get("config", {})
        row = {"arm": arm, "dataset": dataset,
               # WHICH configuration produced this row. The whole point of
               # storing the config beside the metrics, and until now it was
               # read for four display fields and then thrown away.
               "config_digest": config_digest(cfg),
               "written_at": os.path.getmtime(path)}
        row.update(saved.get("metrics", {}))
        for field in ("tariff", "control_horizon", "forecaster_kind",
                      "milp_parity", "cycle_cost_eur_per_efc", "n_sim",
                      "battery_cap"):
            row[field] = cfg.get(field)
        try:
            uid = int(str(dataset).rsplit(" ", 1)[-1])
            row["cluster"] = int(units.loc[uid, "cluster"])
            row["dist_to_centroid"] = float(units.loc[uid, "dist_to_centroid"])
        except (ValueError, KeyError):
            row["cluster"] = np.nan
            row["dist_to_centroid"] = np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Whether a row carries every controller its tariff is compared on. A
    # checkpoint written before the rule roster existed has cost_prophet and
    # cost_oracle but none of the rules, and reading it as if it did is how a
    # partial panel gets reported as a whole one. Flagged rather than dropped
    # here, so the caller can say what it is holding out and why.
    def complete(row):
        tariff = row.get("tariff")
        if tariff not in RULES_BY_TARIFF:
            return False
        wanted = ["no_battery"] + [pol.name for pol in rule_roster(tariff)]
        return all(f"cost_{name}" in row and pd.notna(row[f"cost_{name}"])
                   for name in wanted)

    df["roster_complete"] = df.apply(complete, axis=1)

    # One vintage per arm, and say so when there was more than one.
    #
    # This function's docstring has always claimed that "a row computed under
    # superseded rules can be DROPPED rather than silently mixed in with current
    # ones". It did not do it: the config was read for four display fields and
    # discarded, so a directory holding last week's checkpoints beside today's
    # returned both, and every count, median and paired test ran over the
    # mixture. It is the exact failure the checkpoint tag exists to prevent, and
    # it is easy to be fooled by -- a half-finished sweep reads as a complete one
    # because the row count is right.
    #
    # The current vintage is the digest of the most recently WRITTEN checkpoint
    # in that arm, which is what a resumed sweep is converging on. Rows on any
    # other are dropped and reported.
    keep, dropped = [], {}
    for arm_name, part in df.groupby("arm", sort=False):
        current = part.sort_values("written_at")["config_digest"].iloc[-1]
        stale = part[part["config_digest"] != current]
        if len(stale):
            dropped[arm_name] = (len(stale), sorted(set(stale["config_digest"])))
        keep.append(part[part["config_digest"] == current])
    if dropped:
        total = sum(n for n, _ in dropped.values())
        print(f"  ! {total} checkpoint(s) predate the current configuration and "
              f"were dropped, not mixed in:")
        for arm_name, (n, digests) in sorted(dropped.items()):
            print(f"      {arm_name:24s} {n:3d} row(s) on {', '.join(digests)}")
        print("    Re-run the sweep to refresh them; it is resumable and will "
              "recompute exactly these.")
    df = pd.concat(keep, ignore_index=True)

    order = {a: i for i, a in enumerate(ARM_ORDER)}
    df["_ord"] = df["arm"].map(order).fillna(len(order))
    return df.sort_values(["_ord", "dataset"]).drop(columns="_ord").reset_index(drop=True)


def arm_runtimes(output_root=None, arms=None, cap_s: float = 1800.0) -> pd.DataFrame:
    """What each arm cost to RUN: seconds per household, most expensive first.

    Two sources, and the frame says which one each row came from:

      measured        `runtime_s`, written into the checkpoint by the run that
                      produced it. Exact.
      reconstructed   the gap between consecutive checkpoint mtimes within one
                      household. `_household_all_arms` gives one worker every
                      arm of one household IN ARM ORDER, so an arm's checkpoint
                      is written when it finishes and the previous arm's when it
                      started -- the difference is that arm's wall time. This is
                      what recovers the sweep that ran before `runtime_s`
                      existed; it is an estimate, and two gaps are not
                      recoverable at all:

                        - the FIRST arm of each household has no predecessor to
                          subtract, and
                        - an arm whose predecessor was served from a checkpoint
                          written in an earlier session measures the gap between
                          sessions, not the work. Those are dropped by `cap_s`
                          rather than reported as an arm that took three days.

                      An arm with no recoverable gap in any household comes back
                      with n = 0 and NaN times, which is the honest answer. The
                      `source` column separates the two reasons -- no predecessor
                      is `unrecoverable`, every gap rejected is `cross-session`.

    THE TIMES CAN COME FROM DIFFERENT SITTINGS, and `sessions` says how many.
    Arms added to `STUDY_ARMS` later are run in a later session, against caches
    the earlier arms already warmed, so their reconstructed times are not
    comparable with the earlier arms' on the cache argument below. `dropped`
    counts the gaps `cap_s` rejected ON ROWS WITH NO RECORDED TIME -- a run that
    stored its own `runtime_s` needs no gap and has lost nothing -- which is
    where those boundaries are.

    The times are WALL CLOCK UNDER CONTENTION -- the sweep runs `n_jobs`
    households at once, each pinned to one thread -- so they rank arms against
    each other on the machine that ran them; they are not the time an arm would
    take alone on an idle box.

    A caching effect is real and visible here, not noise to be averaged out: the
    first arm to need a forecaster or a forecast-blind oracle pays for it and
    every later arm reading the same cache does not. That is why `SI_H24` stands
    above the SI arms after it in `ARM_ORDER` -- it pays for the SI oracle -- and
    why, WITHIN the later hbd session, `AU_H24_hbd_median14` stands above the
    hbd arms after it. Those two are not each other's peers: they are the cache
    payers of two different sittings, which is exactly what `sessions` is for.
    """
    output_root = RESULTS_DIR if output_root is None else output_root
    rows = []
    for path in sorted(glob.glob(os.path.join(output_root, "*", "*", "checkpoint.json"))):
        arm = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if arms is not None and arm not in arms:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                saved = json.load(fh)
        except (ValueError, OSError):
            continue
        rows.append({"arm": arm,
                     "dataset": os.path.basename(os.path.dirname(path)),
                     "written_at": os.path.getmtime(path),
                     "recorded_s": saved.get("metrics", {}).get("runtime_s", np.nan)})
    if not rows:
        return pd.DataFrame()
    runs = pd.DataFrame(rows).sort_values(["dataset", "written_at"])

    # The gap to the previous checkpoint IN THE SAME HOUSEHOLD. Across
    # households it would be the gap between two unrelated workers.
    gap = runs.groupby("dataset")["written_at"].diff()
    runs["gap_s"] = gap.where((gap > 0) & (gap < cap_s))
    runs["runtime_s"] = runs["recorded_s"].fillna(runs["gap_s"])
    runs["source"] = np.where(runs["recorded_s"].notna(), "measured", "reconstructed")

    # WHY a gap is missing, which the single "unrecoverable" label used to hide.
    # The two cases are not the same failure and do not have the same fix:
    #
    #   no predecessor   the arm is first in ARM_ORDER, so there is no earlier
    #                    checkpoint in that household to subtract. Nothing can
    #                    recover it except recording `runtime_s`.
    #   over cap_s       there IS a predecessor and the gap was rejected as a
    #                    session boundary. On this sweep every one of
    #                    `AU_H24_hbd`'s thirty gaps is ~237 000 s -- 2.7 days --
    #                    because the hbd arms were added in a LATER session than
    #                    the arms before them in ARM_ORDER. `cap_s` is right to
    #                    reject them, but it means the surviving rows come from
    #                    two different sessions with two different cache states,
    #                    and a single ranked table presents them as comparable.
    #                    `sessions` below is how a reader sees that.
    runs["dropped_over_cap"] = gap.notna() & runs["gap_s"].isna() & runs["recorded_s"].isna()
    # Which sitting each checkpoint was written in: consecutive writes more than
    # `cap_s` apart are different sessions. Kept per arm, because the number a
    # reader needs is not "how many" -- no arm straddles a boundary -- but
    # WHICH, so two arms measured days apart against different cache states
    # cannot be read off one ranked table as peers.
    order = runs.sort_values("written_at")
    runs["session"] = (order["written_at"].diff().gt(cap_s).cumsum()
                       .reindex(runs.index))
    runs["ran_on"] = pd.to_datetime(runs["written_at"], unit="s")

    out = (runs.groupby("arm")
           .agg(households=("runtime_s", "count"),
                median_s=("runtime_s", "median"),
                mean_s=("runtime_s", "mean"),
                min_s=("runtime_s", "min"),
                max_s=("runtime_s", "max"),
                # NaN, not 0.0, when nothing was recovered: an arm with no
                # time on record did not take no time.
                total_min=("runtime_s", lambda s: s.sum() / 60.0 if s.notna().any() else np.nan),
                dropped=("dropped_over_cap", "sum"),
                sessions=("session", "nunique"),
                ran_on=("ran_on", "median"))
           .reset_index())
    out["ran_on"] = out["ran_on"].dt.strftime("%Y-%m-%d")
    src = (runs[runs["runtime_s"].notna()].groupby("arm")["source"]
           .agg(lambda s: "measured" if (s == "measured").all()
                else "reconstructed" if (s == "reconstructed").all() else "mixed"))
    # "unrecoverable" only where there was nothing to subtract at all. An arm
    # whose every gap was REJECTED is a different statement -- the work happened,
    # in a session this reconstruction cannot see across -- so it says so.
    out["source"] = out["arm"].map(src).fillna("unrecoverable")
    out.loc[out["source"].eq("unrecoverable") & out["dropped"].gt(0),
            "source"] = "cross-session"
    out["tariff"] = out["arm"].map(arm_tariff)
    out["forecaster"] = out["arm"].map(
        {a["name"]: a.get("forecaster_kind", "prophet") for a in STUDY_ARMS})
    # Most expensive on top; an arm with no time at all goes last rather than
    # sorting as if it were instant.
    return (out.sort_values("median_s", ascending=False, na_position="last")
            .reset_index(drop=True)[["arm", "tariff", "forecaster", "households",
                                     "median_s", "mean_s", "min_s", "max_s",
                                     "total_min", "ran_on", "sessions",
                                     "dropped", "source"]])


def controller_columns(df: pd.DataFrame) -> list:
    """The controllers present in a results frame, in reporting order."""
    known = ["no_battery"] + list(rbc.POLICY_ORDER)
    # Reporting order runs from the least informed controller to the most, so
    # `milp_full` -- which sees the whole year at once -- comes last.
    known = known + ["tariff_arbitrage", "prophet", "oracle", "milp_full"]
    present = set()
    for col in df.columns:
        if col.startswith("cost_"):
            present.add(col[len("cost_"):])
    # Only names that are actually controllers. An unrecognised cost_* column is
    # reported rather than sorted in at the end of the list, where it reads as a
    # controller nobody can name -- which is how "eur_closed_oracle" ended up in
    # the paired tables next to "peak shaving".
    unknown = present - set(known)
    if unknown:
        print(f"  ! ignoring cost_* column(s) that name no controller: "
              f"{', '.join(sorted(unknown))}")
    return [c for c in known if c in present]


def summarize(df: pd.DataFrame, reference="no_battery") -> pd.DataFrame:
    """One row per (arm, household, controller): cost, saving, share of oracle.

    Long rather than wide, because every figure and every paired test wants a
    row per controller while the checkpoint stores a column per controller.
    """
    frames = []
    for name in controller_columns(df):
        part = df[KEY_COLUMNS + ["tariff", "cluster"]].copy()
        part["controller"] = name
        part["cost"] = df["cost_" + name]
        part["efc"] = df["efc_" + name] if "efc_" + name in df else np.nan
        frames.append(part)
    long = pd.concat(frames, ignore_index=True)
    # A controller that does not run on this arm's tariff -- peak shaving on AU,
    # where there is no capacity charge to earn from -- is absent, not NaN. A row
    # of NaN reads as "ran and produced nothing", which is a different claim.
    long = long[long["cost"].notna()].reset_index(drop=True)

    indexed = df.set_index(KEY_COLUMNS)
    key = pd.MultiIndex.from_frame(long[KEY_COLUMNS])
    base = indexed["cost_" + reference].reindex(key).to_numpy()
    long["baseline_cost"] = base
    long["saving"] = base - long["cost"]

    # What "fraction of the theoretical gain" divides by: the WHOLE-PERIOD
    # optimum, where the sweep has one. It used to divide by `cost_oracle`, the
    # receding-horizon arm reading realised data -- perfect foresight within
    # 24 h or 11 h and no further -- so a controller could and did score near
    # 100 % of a "theoretical" gain that was itself a horizon-limited heuristic.
    # `gain_share_pct_vs_mpc_oracle` keeps that older reading, because "how much
    # of what an MPC with a perfect forecast got" is a real question; it is just
    # not the one the word "theoretical" was promising.
    #
    # Guarded: a household whose optimum saves nothing has no share to take a
    # percentage of, and dividing by it manufactures a number.
    ceiling = "cost_milp_full" if "cost_milp_full" in indexed else "cost_oracle"
    if ceiling != "cost_milp_full":
        print("  ! no whole-period solve in this frame; gain_share_pct falls "
              "back to the receding-horizon oracle and is not a share of the "
              "optimum")
    full_gain = (indexed["cost_" + reference] - indexed[ceiling]
                 ).reindex(key).to_numpy()
    oracle_gain = (indexed["cost_" + reference] - indexed["cost_oracle"]
                   ).reindex(key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        long["gain_share_pct"] = np.where(
            full_gain > 1e-9, 100.0 * long["saving"] / full_gain, np.nan)
        long["gain_share_pct_vs_mpc_oracle"] = np.where(
            oracle_gain > 1e-9, 100.0 * long["saving"] / oracle_gain, np.nan)
        long["saving_pct"] = np.where(
            base > 1e-9, 100.0 * long["saving"] / base, np.nan)

    # What the cycles cost, at the pack price and rated cycle life
    # Battery_Economics carries. Derived here rather than stored: it is a
    # function of EFC and the pack, both already on the row, and deriving it
    # means a change of pack price does not invalidate a sweep.
    #
    # This is a REPORTED cost, never a billed one -- `Cost_EUR` is the bill and
    # carries no wear. The MILP controllers now DO carry the same price in their
    # objective (`wear_objective_terms`), so for them `saving_net_of_wear` is
    # what they were actually optimising; for the rule-based controllers, which
    # have no objective at all, it stays what the saving is worth after the
    # battery is paid for. A controller with a negative one is buying its saving
    # out of pack life.
    import Battery_Economics as be

    # The rate the RUN was solved under, where the checkpoint carries one, and
    # the Battery_Economics default only as a fallback. They are the same number
    # by default, but they need not be: `cycle_cost_eur_per_efc` is now in the
    # MILP objective, so a sweep at a different pack price optimised against
    # THAT rate, and reporting its wear at a different one would price a
    # decision against a cost nobody made it under.
    caps = pd.to_numeric(
        df.set_index(KEY_COLUMNS)["battery_cap"].reindex(key), errors="coerce"
    ).to_numpy()
    default_rate = np.where(np.isfinite(caps), be.CAPEX_EUR_PER_KWH * caps
                            / be.BATTERY_CYCLE_LIMIT_EFC, np.nan)
    #
    # THE DISPATCH RATE IS NOT THE ACCOUNTING RATE, and conflating them is how
    # the no-degradation arm would have read as the cheapest controller in the
    # study. `cycle_cost_eur_per_efc` is the shadow price the MILP was charged;
    # the no-wear arms set it to 0 on purpose. The pack wears regardless, so a
    # solved rate of 0 must NOT be read back as "these cycles were free" -- that
    # would zero `wear_eur`, `saving_net_of_wear` and `saving_total`, and hand
    # the hardest-cycling controller in the sweep the best NPV on every figure.
    #
    # A solved rate that is POSITIVE and different is still honoured, which is
    # the case the original rule was written for: a sweep at another pack price
    # optimised against that price, and reporting its wear at another one would
    # price a decision against a cost nobody made it under. Zero is the one
    # value that cannot mean "a different pack price".
    #
    # Preference order: the rate the run recorded for accounting, then a
    # positive dispatch rate, then the pack's own price. The first is absent
    # from every checkpoint written before the no-wear arms existed, and the
    # third reproduces exactly what those rows reported.
    report_rate = np.full(len(key), np.nan)
    if "cycle_cost_reporting_eur_per_efc" in df:
        report_rate = pd.to_numeric(
            df.set_index(KEY_COLUMNS)["cycle_cost_reporting_eur_per_efc"]
            .reindex(key), errors="coerce").to_numpy()
    solved_rate = np.full(len(key), np.nan)
    if "cycle_cost_eur_per_efc" in df:
        solved_rate = pd.to_numeric(
            df.set_index(KEY_COLUMNS)["cycle_cost_eur_per_efc"].reindex(key),
            errors="coerce").to_numpy()
    rate = np.where(
        np.isfinite(report_rate) & (report_rate > 0), report_rate,
        np.where(np.isfinite(solved_rate) & (solved_rate > 0), solved_rate,
                 default_rate))
    long["wear_eur"] = long["efc"] * rate
    # Both rates on the row, so "was this controller charged for its cycles?"
    # is a column rather than a thing you have to know about the arm.
    long["wear_rate_charged"] = np.where(np.isfinite(solved_rate), solved_rate, 0.0)
    long["wear_rate_accounted"] = rate
    # Only the MILP family HAS an objective to price wear into. A rule-based
    # controller has none in any arm, so flagging it by the arm's shadow price
    # would say something false about eight rows in ten.
    _has_objective = long["controller"].map(
        lambda c: controller_family(c) in ("MPC", "MILP"))
    long["wear_priced_in_objective"] = np.where(
        _has_objective, long["wear_rate_charged"] > 0, False)
    long["saving_net_of_wear"] = long["saving"] - long["wear_eur"].fillna(0.0)

    # THE quantity the whole-period solve minimises, and therefore the only one
    # it is guaranteed to be a ceiling on: bill + standing charge + wear.
    # `full_period_bound_check` writes it per run as `total_<controller>`, and it
    # is read back rather than recomputed so the figure a chart divides by is the
    # same figure the run asserted its bound on.
    #
    # `saving_net_of_wear` is NOT that quantity: it leaves the standing charge
    # out, on the reasoning that no controller can move it. That is true on AU
    # and false on SI, where the dogovorjena moc is endogenous -- `price_interval`
    # says so in as many words -- so a peak shaver trades a larger energy bill
    # for a smaller contract and comes out ahead on a measure that cannot see the
    # contract. Measured on the 930 household-arms of this sweep:
    # `self_consumption_peak_shaving` beat the optimum on `saving_net_of_wear`
    # in 45 of them, by up to 6.5 %, and `oracle` in a further 14 by up to
    # 0.15 %, while neither ever beat it on the total. Both columns are kept --
    # the energy
    # bill is what a reader compares across tariffs -- but the SHARE divides by
    # the total, because a share of a ceiling has to be a share of the thing the
    # ceiling is a ceiling on.
    if f"total_{reference}" in df and "total_milp_full" in df:
        # Same shape as the `cost` column above: one `total_<controller>` column
        # per controller in the wide frame, gathered into one column of the long
        # one.
        totals = []
        for name in controller_columns(df):
            col = f"total_{name}"
            totals.append(indexed[col].reindex(key).to_numpy() if col in indexed
                          else np.full(len(key), np.nan))
        pick = {name: i for i, name in enumerate(controller_columns(df))}
        rows = np.arange(len(long))
        long["cost_total"] = np.choose(
            long["controller"].map(pick).to_numpy(), totals)[rows] \
            if len(totals) else np.nan
        base_total = indexed[f"total_{reference}"].reindex(key).to_numpy()
        long["baseline_cost_total"] = base_total
        long["saving_total"] = base_total - long["cost_total"]
        opt_gain = base_total - indexed["total_milp_full"].reindex(key).to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            long["gain_share_net_pct"] = np.where(
                opt_gain > 1e-9, 100.0 * long["saving_total"] / opt_gain, np.nan)
            # THE RANKING METRIC, AS A PERCENTAGE. `saving_pct` -- the one the
            # comparison figure has always drawn -- is a share of the ENERGY
            # BILL alone, and the whole-period optimum is not a ceiling on it:
            # the optimum pays for its cycles and a price rule does not, so on
            # AU three rules outscore the "optimum" by up to 6.6 % of the gain.
            # A figure that labels a row "optimum" and then draws three rows
            # past it is not reporting a result, it is reporting the wrong
            # column. This is the share of the quantity the solve actually
            # minimises -- bill, standing charge and wear -- which is the only
            # one the label is true of.
            long["saving_total_pct"] = np.where(
                base_total > 1e-9, 100.0 * long["saving_total"] / base_total,
                np.nan)
            # Wear on the same denominator, so a dumbbell of the two reads in
            # one unit on both tariffs: "this controller earns 9 % of the bill
            # and spends 4 % of it on pack life".
            long["wear_pct"] = np.where(
                base_total > 1e-9, 100.0 * long["wear_eur"] / base_total, np.nan)
    else:
        long["cost_total"] = long["saving_total"] = long["gain_share_net_pct"] = np.nan
        long["saving_total_pct"] = long["wear_pct"] = np.nan
        long["baseline_cost_total"] = np.nan

    # ------------------------------------------------------------------
    # Lifetime economics: is the pack worth buying at all?
    # ------------------------------------------------------------------
    # Everything above ranks controllers against each other on ONE year. None of
    # it answers the question a household asks, which is whether the box pays
    # for itself over its life -- and that answer changes the reading of the
    # table, because on these tariffs it does not, and the controllers are
    # competing over how far short they fall. A ranking that only ever shows
    # positive savings reads as "which of these to buy"; the NPV says the honest
    # thing, which is "none of them at this quote", and the IRR says by how far.
    #
    # Wear is charged ONCE, as cash, at the rate the PACK costs:
    #
    #   annual   `saving_total` -- the operating saving (energy bill, plus on SI
    #            the standing charge the controller's own peaks agreed to) minus
    #            `wear_eur`. This is already THE quantity the whole-period
    #            optimum minimises and the one `gain_share_net_pct` divides by,
    #            so the NPV ranks on the same money the rest of the table does.
    #   life     the 12 y calendar band, with `cycle_limit_efc=None`. The cycle
    #            limit is deliberately OFF: `battery_economics` would otherwise
    #            charge wear a second time, as a shortened service life, on top
    #            of the cash already subtracted.
    #
    #            THIS IS NOW LOAD-BEARING, where it used to be free. While every
    #            arm priced wear in its objective the hardest cycler booked
    #            ~272 EFC/a, i.e. 6000/272 = 22 y of cycle life against a 12 y
    #            calendar band, so the calendar bound every row and the choice
    #            cost nothing. The no-degradation arms are the case that can
    #            break that: a MILP that pays nothing to cycle has no reason to
    #            stop, and above 500 EFC/a cycles bind first. `cycle_life_y` is
    #            on every row so this is checkable rather than asserted, and
    #            `life_binds_on_cycles` says outright where it has happened.
    #            Where it does, the NPV below UNDERSTATES the cost of cycling --
    #            the cash is charged but the shortened life is not -- and the
    #            figures say so rather than quietly discounting a 12 y annuity
    #            from a pack that is dead in eight.
    #   capex    the pack, IN THE ARM'S OWN CURRENCY -- a NPV is a quote against
    #            a bill, and on AU the bill is AUD. See `ARM_CURRENCY_PER_EUR`,
    #            which also records where the sweep did NOT convert.
    #
    # `no_battery` is priced at zero capacity: no capex, NPV 0 by construction.
    # It is the reference the others are a delta against, not an investment.
    op_saving = long["saving_total"] + long["wear_eur"].fillna(0.0)
    long["saving_operating"] = op_saving.where(op_saving.notna(), long["saving"])
    long["saving_annual_net"] = long["saving_total"].where(
        long["saving_total"].notna(), long["saving_net_of_wear"])

    # TWO capex readings, because they answer two different questions and the
    # 1000 EUR that separates them is a third of the bill:
    #
    #   ""            pack + install: 250 EUR/kWh of storage plus the 1000 EUR
    #                 hybrid inverter and fitting. What a household actually
    #                 writes a cheque for, and the honest default.
    #   "_pack_only"  the cells alone, at their own price. "Is storage worth
    #                 what storage costs?", asked separately from "does the whole
    #                 retrofit pay?" -- the distinction `Battery_Economics` was
    #                 already drawing with `Net_Annual_StorageOnly_EUR`.
    #
    # The install fee is not a rounding difference. It is 29 % of the SI capex,
    # it is INDEPENDENT of pack size, and because OPEX is charged as a share of
    # installed capital it also carries 15 EUR/a of O&M with it -- which on SI is
    # more than several controllers save in total. Which of the two is quoted
    # therefore decides whether a controller has an IRR at all, so both are
    # computed and neither is allowed to be the silent default.
    caps_priced = np.where(long["controller"].to_numpy() == reference, 0.0, caps)
    for col in ("cycle_life_y", "service_life_y", "pv_factor", "lifetime_wear",
                "lifetime_saving_operating"):
        long[col] = np.nan
    for suffix, fixed_eur in (("", be.CAPEX_FIXED_EUR), ("_pack_only", 0.0)):
        for col in ("npv", "irr_pct", "payback_y", "capex", "roi_pct",
                    "break_even_capex", "lifetime_saving",
                    "lifetime_saving_undisc"):
            long[col + suffix] = np.nan
        for tariff, per_eur in ARM_CURRENCY_PER_EUR.items():
            m = (long["tariff"] == tariff).to_numpy()
            if not m.any():
                continue
            econ = be.battery_economics(
                long["saving_annual_net"].to_numpy()[m],
                caps_priced[m],
                long["efc"].to_numpy()[m],
                capex_eur_per_kwh=be.CAPEX_EUR_PER_KWH * per_eur,
                capex_fixed_eur=fixed_eur * per_eur,
                cycle_limit_efc=None,
            )
            long.loc[m, "npv" + suffix] = econ["NPV_EUR"]
            long.loc[m, "irr_pct" + suffix] = econ["IRR_pct"]
            long.loc[m, "payback_y" + suffix] = econ["Payback_y"]
            long.loc[m, "capex" + suffix] = econ["Capex_EUR"]
            # THE NPV AS A PERCENTAGE OF THE CAPITAL IT IS A RETURN ON.
            # `battery_economics` has computed this all along and `summarize`
            # threw it away, so every lifetime figure had to be drawn in money
            # -- which puts AU and SI on two axes that cannot be compared, over
            # a pack that is the same pack quoted twice. As a share of capex it
            # is one unit, one axis, and the reading is direct: -60 % means the
            # household gets 40 cents back per unit invested, on either tariff.
            long.loc[m, "roi_pct" + suffix] = econ["ROI_pct"]
            # What the pack price would have to FALL to, for this controller to
            # break even. Defined for every household -- unlike the IRR, which
            # is NaN wherever the saving misses the O&M charge and so is a
            # median over a different subset on every row.
            long.loc[m, "break_even_capex" + suffix] = \
                econ["Break_Even_Capex_EUR_kWh"]
            # LIFETIME, not annual: the discounted sum of what the controller
            # earns over the pack's service life. This is the NPV's own gross
            # side (`npv = lifetime_saving - capex` exactly), so the two figures
            # are the same statement with and without the capital subtracted.
            _pvf = np.array([be.present_value_factor(be.DISCOUNT_RATE, n)
                             for n in econ["Service_Life_y"]])
            long.loc[m, "lifetime_saving" + suffix] = (
                econ["Net_Savings_EUR"] * _pvf)
            if not suffix:
                # The factor itself, on the row, so a figure drawing the
                # lifetime view recomputes rather than multiplying by a constant
                # it typed in. It IS a constant while every pack lives out its
                # 12 y calendar band -- but the no-degradation arms are the ones
                # that can cycle a pack to death early, and on those rows this
                # varies and the lifetime figures have to follow it.
                long.loc[m, "pv_factor"] = _pvf
                # Wear over the same life, discounted the same way, so the
                # lifetime wear/saving figure is the annual one recomputed and
                # not the annual one rescaled.
                long.loc[m, "lifetime_wear"] = long["wear_eur"].to_numpy()[m] * _pvf
                long.loc[m, "lifetime_saving_operating"] = (
                    long["saving_operating"].to_numpy()[m] * _pvf)
            # The undiscounted one too, because it is the number a household
            # hears ("it saves you X over its life") and the gap between the two
            # is what discounting costs.
            long.loc[m, "lifetime_saving_undisc" + suffix] = (
                econ["Net_Savings_EUR"] * econ["Service_Life_y"])
            if not suffix:
                long.loc[m, "service_life_y"] = econ["Service_Life_y"]
                # What the life WOULD be if cycles bound, at the rated 6000 EFC.
                # Not used by the NPV above; reported so "cycles never bind on
                # this sweep" is a number in the frame rather than a claim in a
                # comment.
                long.loc[m, "cycle_life_y"] = be.service_life_years(
                    long["efc"].to_numpy()[m])[1]

    # The IRR does not exist for a controller whose saving, net of wear, does
    # not even cover the O&M charge: NPV(r) is then negative at every discount
    # rate and there is no root to report. That is a DIFFERENT statement from a
    # large negative rate, so it stays NaN -- but it means a median IRR is taken
    # over a different subset of households than a median NPV, which is exactly
    # how a table comes to contradict itself. `irr_defined` makes the subset
    # visible so the comparison can be read honestly.
    long["irr_defined"] = long["irr_pct"].notna() & (caps_priced > 0)
    long["irr_defined_pack_only"] = (long["irr_pct_pack_only"].notna()
                                     & (caps_priced > 0))

    # WHERE THE NPV ABOVE IS OPTIMISTIC. The life it discounts over is the 12 y
    # calendar band; this says where the rated 6000 EFC would have run out
    # first. On every wear-priced arm it is False everywhere, which is why the
    # `cycle_limit_efc=None` choice was free. A no-degradation MILP is the arm
    # that can make it True, and where it is, the row's lifetime figures are an
    # upper bound rather than an estimate -- the cash cost of the cycles is
    # charged, the shortened life is not.
    long["life_binds_on_cycles"] = (long["cycle_life_y"]
                                    < long["service_life_y"] - 1e-9)

    # The lifetime ranking, as a share, on the lifetime bill. The direct
    # analogue of `saving_total_pct` over the pack's life rather than over one
    # year -- and NOT the same number, because the lifetime saving carries the
    # O&M charge that the annual one does not and is discounted over a life that
    # a hard-cycling controller can shorten. Where the two disagree, that
    # difference is the whole reason the article carries both views.
    with np.errstate(divide="ignore", invalid="ignore"):
        _life_bill = long["baseline_cost_total"] * long["pv_factor"]
        long["lifetime_saving_pct"] = np.where(
            _life_bill > 1e-9, 100.0 * long["lifetime_saving"] / _life_bill,
            np.nan)

    # REGRET AS A SHARE OF WHAT WAS THERE TO WIN, so the two tariffs land on one
    # axis. The money version cannot be compared across arms -- AU bills in AUD
    # and SI in EUR, and a 90 AUD regret against an AU bill is not the same
    # statement as a 30 EUR one against an SI bill. Divided by the gain perfect
    # foresight actually achieves on that household, both become "what fraction
    # of the achievable saving did this forecast throw away", which is the
    # question the study is named after and the only form in which AU and SI can
    # be put side by side.
    #
    # Per household, before any median: this is the same pairing argument the
    # regret figure makes, and a ratio of two medians is not any household's
    # share. Households where perfect foresight wins nothing (gain <= 0) have no
    # share to take a fraction of and go to NaN rather than to a large number
    # with an arbitrary sign.
    _gain = long["baseline_cost"] - indexed["cost_oracle"].reindex(key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        long["regret_pct_of_gain"] = np.where(
            _gain > 1e-9,
            100.0 * (long["cost"] - indexed["cost_oracle"].reindex(key).to_numpy())
            / _gain,
            np.nan)
    return long


def best_rule(long: pd.DataFrame, arm: str, exclude=("price_oracle",)) -> str:
    """The deployable rule to carry into that tariff's other arms.

    Lowest median cost across households on the arm's own reference run, with
    the non-causal diagnostics excluded: `price_oracle` reads the whole year and
    is there to bound what foresight is worth to a threshold rule, never to be
    recommended.
    """
    rules = (set(rbc.POLICY_ORDER) | {"tariff_arbitrage"}) - set(exclude)
    sub = long[(long["arm"] == arm) & long["controller"].isin(rules)]
    sub = sub.dropna(subset=["cost"])
    if sub.empty:
        # No rule has run on this arm -- almost always a results directory
        # written before the roster existed. None, not an exception: a notebook
        # should be able to say "not run yet" and carry on drawing what it has.
        return None
    return sub.groupby("controller")["cost"].median().idxmin()


def paired_comparison(df, a, b, label_a=None, label_b=None) -> dict:
    """Wilcoxon signed-rank on the per-site difference between two columns.

    The sites are PAIRED -- every household is run under both conditions -- so
    the per-site difference is the unit of analysis, and a box plot of two
    independent-looking distributions understates the evidence. Wilcoxon rather
    than a t-test because 30 sites of cost differences are not plausibly normal
    and a handful of them (large exporters, dead arrays) sit far out.
    """
    from scipy import stats

    pair = df[[a, b]].dropna()
    diff = (pair[a] - pair[b]).astype(float)
    nonzero = diff[diff != 0]
    out = {
        "comparison": f"{label_a or a} - {label_b or b}",
        "n": int(len(diff)),
        "n_effective": int(len(nonzero)),
        "median_diff": float(diff.median()),
        "q1_diff": float(diff.quantile(0.25)),
        "q3_diff": float(diff.quantile(0.75)),
        "n_a_greater": int((diff > 0).sum()),
        "n_b_greater": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
    }
    if len(nonzero) < 6:
        # Below ~6 non-tied pairs the exact test cannot reach p < 0.05 whatever
        # the data does, so a p-value here would mislead rather than be weak.
        out["p_value"] = float("nan")
        out["note"] = f"only {len(nonzero)} non-tied pairs; test not run"
    else:
        out["p_value"] = float(stats.wilcoxon(nonzero).pvalue)
        out["note"] = ""
    return out


def paired_arms(df: pd.DataFrame, tariff: str, metric="cost_prophet"):
    """Every arm on one tariff against that tariff's reference arm, site by site."""
    sub = df[df["tariff"] == tariff]
    wide = sub.pivot(index="dataset", columns="arm", values=metric)
    ref = REFERENCE_ARM[tariff]
    if ref not in wide.columns:
        return pd.DataFrame()
    rows = [paired_comparison(wide, arm, ref) for arm in wide.columns if arm != ref]
    return pd.DataFrame(rows)


def paired_controllers(long: pd.DataFrame, arm: str, reference="prophet"):
    """Every controller against MILP+Prophet on one arm, household by household."""
    wide = long[long["arm"] == arm].pivot(
        index="dataset", columns="controller", values="cost")
    if reference not in wide.columns:
        return pd.DataFrame()
    rows = [paired_comparison(wide, c, reference)
            for c in wide.columns if c != reference]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Folder containing the CSV files (one per Ausgrid site). Overridable so the
    # notebook runs unchanged on Colab, macOS and Windows.
    DATA_DIR    = os.environ.get(
        "ERK_DATA_DIR",
        os.path.join("..", "Input data", "Ausgrid"),
    )
    OUTPUT_ROOT = os.environ.get("ERK_OUTPUT_ROOT", "results")

    # One household per k-means cluster, read from the clustering rather than
    # transcribed -- see section 8a. The order is cluster order, and it is the
    # same 30 ids the published run used.
    DATASET_IDS = dataset_ids()

    # Every arm over every dataset. Each (arm, dataset) checkpoints on its own,
    # so an interrupted run resumes where it stopped; forecasts are shared
    # across arms, so the whole sweep costs one Prophet fit per household.
    # For a single arm instead, call run_all(...) with the same keywords plus
    # the arm's own (tariff=, control_horizon=, ...).
    summary = run_arms(
        data_dir=DATA_DIR,
        output_root=OUTPUT_ROOT,
        dataset_ids=DATASET_IDS,
        filename_template="Ausgrid {id}.csv",
        battery_cap=10.0,
        soc_min_pct=0.10,
        soc_max_pct=0.80,
        p_max=1.5,
        eff=0.95,
        delta_t=0.5,
        soc_init=5.0,
        H=48,
        n_train=730,
        n_sim=365,
        start_ts="2010-07-01 00:30:00",
    )
    print(summary)
