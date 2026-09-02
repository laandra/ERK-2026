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

import contextlib
import datetime
import glob
import hashlib
import json
import logging
import os
import traceback
import warnings
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pulp
from prophet import Prophet


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

    MONTHLY_SUBSCRIPTION_EX_GST = 20.00
    DAILY_SUPPLY_EX_GST         = 1.09
    DAYS_IN_MONTH               = 30

    # F9 - AEST is not "strictly UTC+10 year-round": NSW observes AEDT from
    # October to April, so a fixed +10 puts the peak window an hour off for
    # roughly half of every simulated year. Australia/Sydney handles it.
    LOCAL_TZ = "Australia/Sydney"

    # Ausgrid EA025 charges the peak rate on working days. The original model
    # applied it every day including weekends and public holidays, which is both
    # wrong and inconsistent with the Slovenian arm, whose blocks have always
    # distinguished them. Set False to recover the old, calendar-blind behaviour.
    WORKDAY_AWARE = True
    HOLIDAY_COUNTRY, HOLIDAY_SUBDIV = "AU", "NSW"

    @classmethod
    def _local(cls, utc_date: datetime.datetime) -> datetime.datetime:
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
        # "/ 1000.0", which is an invitation to a 1000x "fix". /0.615 is the
        # EUR->AUD conversion only.
        spot_price_kwh = smp_eur_per_kwh / 0.615
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
# Upstream solves both halves with one idea (Horizon_Comparison.run_user): every
# checkpoint carries a tag describing the study it was produced under, and a
# checkpoint whose tag no longer matches is dropped rather than resumed into.
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

FORECAST_CACHE_DIR = os.environ.get("ERK_FORECAST_CACHE", "forecast_cache")


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
            "ds": window.index,
            "yhat_con": window["Energy_Consumption"].values,
            "yhat_gen": window["Energy_Generation"].values,
        })


# A forecast has two channels and they are not equally hard. Prophet earns its
# place on consumption (skill vs seasonal-naive about +0.2 to +0.3) and loses to
# yesterday on generation. Averaging the two into one "forecast quality" hides
# that, so these kinds hold the consumption channel fixed at Prophet and vary
# only the roof: {kind: (consumption source, generation source)}.
HYBRID_KINDS = {
    "pvnaive": ("prophet", "persistence"),
    "pvtruth": ("prophet", "truth"),
}


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
        if source == "persistence":
            # No fitting: yesterday, same interval. `history` supplies the day
            # before the simulation so the first day is copied from real data.
            frame = pd.concat([history, df_sim]) if history is not None else df_sim
            table = build_forecast_table(PersistenceForecaster(frame, H),
                                         anchors, H, freq)
            print(f"  [Forecaster] persistence: {len(table)} rows, nothing fitted")
            return table
        if source == "truth":
            table = build_forecast_table(TruthForecaster(df_sim, H), anchors, H, freq)
            print(f"  [Forecaster] truth: {len(table)} rows, nothing fitted")
            return table
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
        if len(gen) != len(table) or not gen["ds"].equals(table["ds"]):
            raise ValueError(
                f"{kind}: the {con_source} and {gen_source} tables do not share "
                f"an index ({len(table)} vs {len(gen)} rows)"
            )
        table = table.assign(yhat_gen=gen["yhat_gen"].to_numpy())
        print(f"  [Forecaster] {kind}: consumption from {con_source}, "
              f"generation from {gen_source}")
    else:
        table = channel_table(kind)
    table.to_csv(path, index=False, compression="gzip")
    print(f"  [Forecaster] cached {len(table)} rows -> {os.path.basename(path)}")
    served = TableForecaster(table)
    served.table = table
    return served, path


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
    `os.getcwd() + "/.."` is only correct when the cwd happens to be Yoann/, and
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
    add_household_physics,
    build_household_env,
    floor_export_rates,
    interval_rate_vectors,
    step_energy_kwh,
)
import si_cas as _si_cas                # noqa: E402  (after MILP_Household)
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
        self.solver   = solver or pulp.PULP_CBC_CMD(msg=0)

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

        prob = pulp.LpProblem("MILP_HEMS", pulp.LpMinimize)
        blk = add_household_physics(
            prob, self.env, n_steps=H, gen=gen, con=con,
            initial_soc_kwh=soc0, final_soc_kwh=soc_end,
            exclusivity=self.exclusivity,
            metering_bounds=self.metering_bounds,
        )
        if not self.allow_spill:
            for t in range(H):
                prob += blk.spill[t] == 0, f"nospill_{t}"

        prob += pulp.lpSum(
            blk.buy[t] * buy_rate[t] - blk.sell[t] * export[t] for t in range(H)
        ), "MinNetCost"

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


class PersistenceForecaster:
    """Yesterday, same interval -- the standard baseline any forecaster must beat.

    Causal by construction: the day it copies ends one full day before the
    anchor it is asked about. `history` supplies the day before the simulation
    starts, so the first simulated day is forecast from real data rather than
    from itself.
    """

    def __init__(self, frame: pd.DataFrame, steps_per_day: int):
        self.frame = frame
        self.spd = steps_per_day

    def fit(self, *a, **k):
        return self

    def config(self) -> dict:
        return {"kind": "persistence", "steps_per_day": self.spd}

    def predict_next_day(self, anchor_ts, horizon_steps: int = 48,
                         freq: str = "30min") -> pd.DataFrame:
        i = self.frame.index.get_indexer([anchor_ts])[0]
        if i < 0:
            raise KeyError(f"anchor {anchor_ts} is not in the persistence frame")
        j = i - self.spd
        if j < 0:
            raise ValueError(
                f"no prior day before {anchor_ts}; the persistence forecaster "
                f"needs at least one day of history ahead of the simulation"
            )
        prev = self.frame.iloc[j:j + horizon_steps]
        idx = self.frame.index[i:i + horizon_steps]
        return pd.DataFrame({
            "ds": idx,
            "yhat_con": prev["Energy_Consumption"].values[:len(idx)],
            "yhat_gen": prev["Energy_Generation"].values[:len(idx)],
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


def run_rules(env, settle, tariff, signals=None, only=None, n_steps=None):
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
            signals = rbc.build_signals(env, n_steps=n_steps)

    roster = [rbc._Idle()] + rule_roster(tariff)
    if only is not None:
        roster = [pol for pol in roster if pol.name in set(only)]

    metrics, rows = {}, []
    for pol in roster:
        out = rbc.run_policy(env, pol, signals=signals, settle=settle)
        metrics[f"cost_{pol.name}"] = out["Cost_EUR_Closed"]
        metrics[f"efc_{pol.name}"] = out["Equivalent_Full_Cycles"]
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


def rule_roster(tariff):
    """The controllers this tariff is compared against, already configured."""
    respect = tariff == "SI"
    out = []
    for name in RULES_BY_TARIFF[tariff]:
        arg = _PEAK_AWARE_ARGS.get(name)
        kw = {arg: respect} if arg else {}
        if name == "tariff_arbitrage":
            pol = TariffArbitrage(**kw)
            pol.name = name
        else:
            pol = rbc.make_policy(name, **kw)
        out.append(pol)
    return out


def settle_trajectory(env, net_kwh, settle, sig):
    """Price an executed trajectory through the arm's evaluator.

    The MILP half of "one evaluator". `ReactiveController.run` decides a
    dispatch and reports what it did; the cost it reports along the way is a
    running convenience, not the bill. This walks the realized net draw through
    exactly the settlement the rules are priced by -- same peak state, same
    window drops, same standing charge -- so `Cost_EUR` means one thing across
    the whole results frame.
    """
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
    return {"Cost_EUR": cost, "Energy_EUR": energy, "Power_EUR": power,
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
# 7 — Pipeline for ONE dataset (formerly main(), now parameterized)
# =====================================================================

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
                           control_horizon: int | None = None,
                           tariff: str = "AU",
                           forecaster_kind: str = "prophet",
                           leak_current_interval: bool = False,
                           smp_source: str | None = None,
                           milp_parity: bool = True,
                           cycle_cost_eur_per_efc: float | None = None,
                           holiday_country: str = "AU",
                           holiday_subdiv: str | None = "NSW",
                           high_season_months: tuple = (5, 6, 7, 8)) -> dict:
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

    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    out_dir = os.path.join(output_root, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

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
        # a wear shadow price in its objective, which changes what it decides to
        # do (and, deliberately, stops the objective being the reported bill).
        milp_parity=bool(milp_parity),
        cycle_cost_eur_per_efc=cycle_cost_eur_per_efc,
        leak_current_interval=bool(leak_current_interval),
        forecaster_kind=forecaster_kind,
        refit_every_days=refit_every_days,
        smp_source=smp_source or "column",
        calendar=(holiday_country, holiday_subdiv, tuple(sorted(high_season_months))),
        # A DIGEST, not the settings themselves: the forecaster's configuration
        # is already recorded beside the forecasts it produced, and duplicating
        # it here would mean two copies that can disagree. It cannot be dropped
        # altogether though -- a changed forecaster changes the answer, so
        # without this a stale result would be resumed into.
        forecaster_digest=config_digest(
            EnergyForecaster(forecaster_params_con,
                             forecaster_params_gen).config()),
    )
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

    # Both tariff calendars are evaluated against the DATA's own dates and the
    # data's own hemisphere. Slovenian public holidays on an Australian load
    # profile mark the wrong days non-working (only 5 of the 11 NSW and 14 SI
    # dates coincide in 2013), and the northern high season would put the winter
    # network peak on the Australian summer.
    _si_cas.nastavi_koledar(drzava=holiday_country, podrocje=holiday_subdiv,
                            visja_sezona_meseci=set(high_season_months))
    TariffCalculator.HOLIDAY_COUNTRY = holiday_country
    TariffCalculator.HOLIDAY_SUBDIV = holiday_subdiv

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
            return {"dataset": dataset_name, **cached}
        # A checkpoint written before one of these rules existed. They need no
        # forecast and no LP, so they are recomputed in seconds and merged rather
        # than discarding MILP results that are still perfectly valid. The tag
        # guard exists to stop rows produced under DIFFERENT RULES from mixing;
        # adding a controller changes no rule that produced the rest.
        print(f"  [checkpoint] backfilling {', '.join(missing)} "
              f"(forecast-free, no LP); MILP results reused")
        extra, _ = run_rules(env, settle, tariff, only=missing,
                             n_steps=n_sim * H)
        cached = {**cached, **extra}
        write_checkpoint(out_dir, cfg, cached)
        return {"dataset": dataset_name, **cached}

    forecaster = EnergyForecaster(forecaster_params_con, forecaster_params_gen)
    fc_table, fc_cache_path = load_or_build_forecasts(
        dataset_name, df_train, df_ctrl, H, "30min", forecaster,
        cache_dir=forecast_cache_dir, refit_every_days=refit_every_days,
        kind=forecaster_kind, history=df_train,
    )

    # `add_household_physics`, not the hand-rolled MILPScheduler. b53def7 added
    # this adapter and then never instantiated it, so the swap it announced was
    # inert and the MILP kept solving its own copy of the battery. parity=True
    # reproduces that copy exactly -- verified to 3.6e-9 EUR over seven daily
    # solves -- so the published numbers are reproducible, while the study now
    # has one battery model instead of two that can drift.
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

    print("\n--- Oracle mode (all real data) ---")
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
    df_pk = ctrl_pk.run(num_days=n_sim, use_forecast=False)

    # Every controller re-priced through the arm's ONE evaluator, the two MILP
    # arms included: ReactiveController accumulates a running cost as it goes,
    # but that is a convenience, not the bill, and it drops the standing charge.
    kpi_table, kpi_raw = KPITracker.compare_three(df_fc, df_pk, delta_t)
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
        sig = rbc.build_signals(env, n_steps=len(df_sim))
    for _name, _frame in (("oracle", df_pk), ("prophet", df_fc)):
        _net = (_frame["Buy_kW"] - _frame["Sell_kW"]).to_numpy() * delta_t
        _s = settle_trajectory(env, _net, settle, sig)
        kpi_raw.update({f"{_k.lower()}_{_name}": _v for _k, _v in _s.items()})

    # The deployable alternative, on realized data deliberately: an inverter
    # measures the surplus in front of it, it does not forecast one, so giving a
    # rule the realized interval is physical fidelity rather than the look-ahead
    # it would be for a planner. There is correspondingly no "forecast" variant
    # of any of them.
    print(f"\n--- Rule-based controllers ({tariff}) ---")
    rule_metrics, rule_rows = run_rules(env, settle, tariff, signals=sig)
    kpi_raw.update(rule_metrics)
    for _r in rule_rows:
        print(f"  {_r['controller']:30s} {_r['Cost_EUR_Closed']:9.2f} EUR"
              f"   EFC {_r['Equivalent_Full_Cycles']:6.1f}"
              f"   peak {_r['Peak_Import_kW']:5.2f} kW")
    print(f"vs no battery {kpi_raw['cost_no_battery']:.2f} "
          f"vs oracle {kpi_raw['cost_oracle']:.2f} EUR")
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
    df_fc_csv_path = os.path.join(out_dir, f"df_fc_{dataset_name}.csv")
    df_pk_csv_path = os.path.join(out_dir, f"df_pk_{dataset_name}.csv")

    # F2 - the previous run lost `Ausgrid 138` (the FIRST id in DATASET_IDS)
    # here, with "Cannot save file into a non-existent directory", and run_all
    # swallowed it: the published summary and figures cover 29 of 30 sites.
    # Re-assert the directory immediately before the writes.
    os.makedirs(out_dir, exist_ok=True)
    kpi_table.to_csv(kpi_csv_path, encoding="utf-8-sig")
    # raw time series also kept, useful for finer analysis later
    df_fc.to_csv(df_fc_csv_path, encoding="utf-8-sig")
    df_pk.to_csv(df_pk_csv_path, encoding="utf-8-sig")
    pd.DataFrame(rule_rows).to_csv(
        os.path.join(out_dir, f"rules_{dataset_name}.csv"),
        index=False, encoding="utf-8-sig")

    print(f"\nResults saved to: {out_dir}/")

    write_checkpoint(out_dir, cfg, kpi_raw)

    return {"dataset": dataset_name, **kpi_raw}


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
# 300 Ausgrid households -- `rank_in_cluster == 1` in the clustering
# `Andraz/cluster_sweep_analysis/sweep_analysis.ipynb` also draws its ten fixed
# evaluation users from. One household per cluster is a deliberate sample that
# spans the consumption shapes present in the population, and it is a far better
# story than "30 ids", so it is derived here rather than transcribed: the rule is
# then executable, and a reviewer can check it.

CLUSTERING_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Andraz", "clustering_results", "user_ids_sorted_by_cluster_30.csv",
)


def study_units(path: str | None = None, k: int = 30) -> pd.DataFrame:
    """The study households, one per cluster, with the cluster they represent.

    Returns a frame indexed by dataset id carrying `cluster` and
    `dist_to_centroid`, in cluster order. Those two columns travel with the
    results so "which kind of household is this controller bad at" is a question
    the study can answer -- which, with a flat list of ids, it could not.
    """
    path = CLUSTERING_CSV if path is None else path
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
    # The forecast-channel arms. Prophet's consumption model beats
    # seasonal-naive; its generation model LOSES to it. A single "with forecast"
    # arm averages those two facts into one number and hides the interesting
    # one, so these hold consumption at Prophet and vary only the roof.
    {"name": "AU_H24_pvnaive",  "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvnaive"},
    {"name": "SI_H24_pvnaive",  "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvnaive"},
    {"name": "AU_H24_pvtruth",  "tariff": "AU", "control_horizon": 48,
     "forecaster_kind": "pvtruth"},
    {"name": "SI_H24_pvtruth",  "tariff": "SI", "control_horizon": 48,
     "forecaster_kind": "pvtruth"},
]

# The arm each tariff's comparison is read against, and whose full controller
# roster the "all controllers" figure is drawn from.
REFERENCE_ARM = {"AU": "AU_H24", "SI": "SI_H24"}
ARM_ORDER = [a["name"] for a in STUDY_ARMS]


def arm_tariff(name):
    """Which tariff an arm is on -- the axis its results are comparable along."""
    return next(a["tariff"] for a in STUDY_ARMS if a["name"] == name)


def run_arms(data_dir, output_root="results", dataset_ids=None,
             filename_template="Ausgrid {id}.csv", arms=None, **kwargs):
    """Every arm over every dataset, into one long frame.

    Each (arm, dataset) is checkpointed independently, so an interrupted sweep
    resumes where it stopped rather than from the beginning.
    """
    arms = arms or STUDY_ARMS
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
    path = os.path.join(output_root, "summary_all_arms.csv")
    allrows.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nAll arms: {path}")
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
        row = {"arm": arm, "dataset": dataset}
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
    order = {a: i for i, a in enumerate(ARM_ORDER)}
    df["_ord"] = df["arm"].map(order).fillna(len(order))
    return df.sort_values(["_ord", "dataset"]).drop(columns="_ord").reset_index(drop=True)


def controller_columns(df: pd.DataFrame) -> list:
    """The controllers present in a results frame, in reporting order."""
    known = ["no_battery"] + list(rbc.POLICY_ORDER)
    known = known + ["tariff_arbitrage", "prophet", "oracle"]
    present = set()
    for col in df.columns:
        if col.startswith("cost_"):
            present.add(col[len("cost_"):])
    return [c for c in known if c in present] + sorted(present - set(known))


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

    indexed = df.set_index(KEY_COLUMNS)
    key = pd.MultiIndex.from_frame(long[KEY_COLUMNS])
    base = indexed["cost_" + reference].reindex(key).to_numpy()
    long["baseline_cost"] = base
    long["saving"] = base - long["cost"]

    # The oracle's gain is what "fraction of theoretical gain" divides by.
    # Guarded: a household whose oracle saves nothing has no share to take a
    # percentage of, and dividing by it manufactures a number.
    oracle_gain = indexed["cost_" + reference] - indexed["cost_oracle"]
    oracle_gain = oracle_gain.reindex(key).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        long["gain_share_pct"] = np.where(
            oracle_gain > 1e-9, 100.0 * long["saving"] / oracle_gain, np.nan)
        long["saving_pct"] = np.where(
            base > 1e-9, 100.0 * long["saving"] / base, np.nan)
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
    if sub.empty:
        raise ValueError(f"no rule rows for arm {arm!r}")
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