"""Days the panel figures average away: public holidays, and households away.

Every figure in `CODE.ipynb` is a median over a year and thirty households. That
is the right unit for "what does a forecast buy", and it is exactly the wrong one
for "when does a forecast break", because the days a model-based forecaster gets
badly wrong are rare -- 1.6 % of this panel -- and a year's median cannot see
them.

Two candidate anomalies, and only one of them is real in this data:

  PUBLIC HOLIDAYS      the obvious hypothesis, and it is not there. Against each
                       household's own rolling baseline the NSW public-holiday
                       effect is +0.017 and only 10 of 30 households consume less
                       on a holiday than on a workday; Christmas Day 2012 sits
                       ABOVE the household median. The tariff's working-day rule
                       still uses the calendar -- that part is real -- but the
                       LOAD does not move, so a figure that looks for a dip finds
                       nothing and has to say so.

  ABSENCES             households leaving for days at a time. 37 runs across 22
                       of 30 households, the longest 12 days at 0.28 of baseline.
                       These are not on any calendar, which is the point: a
                       forecaster that fits a model and never reads a recent
                       actual cannot know about them, and it keeps charging the
                       battery for a household that is not there.

This module finds both and builds the daily panel the figures read. It computes,
it does not draw.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

import hems_study as hs

RESULTS_DIR = hs.RESULTS_DIR
DATA_DIR = os.path.join("..", "Input data", "Ausgrid")

# The methods the anomaly figures compare. Two fit-free, two hybrid, one fitted:
# the roster is chosen to span the ONE axis that turns out to matter here --
# whether the method reads a recent actual at prediction time.
#
#   persistence    reads yesterday and nothing else
#   median14       reads the last 14 like-days
#   hbd            fits a seasonal model AND conditions on yesterday's residuals
#   hbd_median14   the study's own stage-1 under the paper's stage-2
#   prophet        fits a model and reads NO recent actual -- the study's subject
PANEL_KINDS = ("persistence", "median14", "hbd", "hbd_median14", "prophet")

# The last day of the window is TRUNCATED -- `load_study_frames` cuts the
# simulation at an interval, not at midnight, so 2013-06-30 carries a handful of
# intervals and lands at 0.015 of baseline. Left in, it is the panel's single
# strongest "absence" and it is an artefact of the cut. Dropped here rather than
# in each caller, which is where it would be forgotten once.
DROP_PARTIAL_LAST_DAY = True


# ---------------------------------------------------------------------------
# The calendar
# ---------------------------------------------------------------------------
def run_calendar(arm="AU_H24", output_root=None):
    """(country, subdiv) the RUN was priced under, read from its checkpoint.

    Not a retyped "AU"/"NSW": `run_pipeline_for_file` writes the calendar it used
    into every checkpoint, and `TariffCalculator._is_workday` reads the same pair
    off the class. A holiday figure that names its own country is a third source
    of truth, and the one that goes stale is the figure.
    """
    root = RESULTS_DIR if output_root is None else output_root
    for hh in sorted(os.listdir(os.path.join(root, arm))):
        path = os.path.join(root, arm, hh, "checkpoint.json")
        if os.path.isfile(path):
            cal = json.loads(open(path).read())["config"]["calendar"]
            return str(cal[0]), str(cal[1])
    raise FileNotFoundError(f"no checkpoint under {os.path.join(root, arm)}")


def public_holidays(years, country="AU", subdiv="NSW"):
    """{date: name} for the given years, from the same library the tariff uses."""
    import holidays as _hol

    out = {}
    for y in sorted(set(int(y) for y in years)):
        out.update(dict(_hol.country_holidays(country, subdiv=subdiv, years=y)))
    return out


def mark_holidays(days, country="AU", subdiv="NSW"):
    """Boolean Series over `days`: is this a public holiday on that calendar."""
    idx = pd.DatetimeIndex(days)
    cal = public_holidays(idx.year.unique(), country, subdiv)
    return pd.Series([d.date() in cal for d in idx], index=idx)


# ---------------------------------------------------------------------------
# Absence detection
# ---------------------------------------------------------------------------
# THE BASELINE IS A CENTRED ROLLING MEDIAN OF THE HOUSEHOLD'S OWN CONSUMPTION,
# and each of those three words is load-bearing:
#
#   the household's own   households differ by a factor of six in annual kWh, so
#                         an absolute threshold would flag the small ones all year
#                         and the large ones never.
#   rolling               consumption is seasonal on an Australian year; a single
#                         annual level would call every mild week an absence.
#   MEDIAN, centred       a 12-day absence is 12 of the 29 days in the window. A
#                         MEAN would be dragged down by the very run it is meant
#                         to measure and the run would partly hide itself; a
#                         median survives up to half the window being out.
ABSENCE_REL = 0.55        # of the household's own baseline
ABSENCE_MIN_DAYS = 3      # consecutive
ABSENCE_WINDOW = 29       # days, centred


def _runs(mask, min_days):
    """Keep only runs of `min_days` or more consecutive True."""
    a = np.asarray(mask, dtype=bool)
    out = np.zeros(len(a), dtype=bool)
    i = 0
    while i < len(a):
        if a[i]:
            j = i
            while j + 1 < len(a) and a[j + 1]:
                j += 1
            if j - i + 1 >= min_days:
                out[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return out


def add_absence(daily, value="act_con", by="hh", rel=ABSENCE_REL,
                min_days=ABSENCE_MIN_DAYS, window=ABSENCE_WINDOW):
    """Add `base`, `rel` and `absence` columns to a per-(household, day) frame.

    `absence` is True only inside a run of `min_days` or more consecutive days
    under the threshold. A single quiet day is not an absence -- it is a weekend
    -- and the run requirement is what separates the two without a calendar.
    """
    out = daily.sort_values([by, "day"]).copy()
    out["base"] = out.groupby(by)[value].transform(
        lambda s: s.rolling(window, center=True, min_periods=window // 3).median())
    out["rel"] = out[value] / out["base"]
    out["absence"] = out.groupby(by)["rel"].transform(
        lambda s: pd.Series(_runs(s < rel, min_days), index=s.index))
    return out


def absence_runs(daily, by="hh"):
    """One row per detected run: household, start, end, length, depth, drop."""
    rows = []
    for hh, s in daily.groupby(by):
        s = s.sort_values("day").reset_index(drop=True)
        a = s["absence"].to_numpy()
        i = 0
        while i < len(a):
            if a[i]:
                j = i
                while j + 1 < len(a) and a[j + 1]:
                    j += 1
                rows.append({
                    by: hh, "start": s["day"][i], "end": s["day"][j],
                    "days": j - i + 1,
                    "rel": float(s["rel"][i:j + 1].median()),
                    "kwh": float(s["act_con"][i:j + 1].median()),
                    "kwh_base": float(s["base"][i:j + 1].median()),
                })
                i = j + 1
            else:
                i += 1
    return pd.DataFrame(rows)


def absence_sensitivity(daily, rels=(0.45, 0.55, 0.65), mins=(2, 3, 5)):
    """How the run count moves with the two thresholds.

    A threshold nobody varies reads as a threshold that was tuned until the
    result appeared. This is the table that says it was not: the finding has to
    survive the neighbourhood of (0.55, 3), and a reader has to be able to see
    that it does.
    """
    rows = []
    for r in rels:
        for m in mins:
            d = add_absence(daily, rel=r, min_days=m)
            runs = absence_runs(d)
            rows.append({"rel": r, "min_days": m, "runs": len(runs),
                         "households": int(runs["hh"].nunique()) if len(runs) else 0,
                         "days": int(d["absence"].sum()),
                         "pct_of_panel": 100.0 * float(d["absence"].mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# The daily panel
# ---------------------------------------------------------------------------
PANEL_PATH = os.path.join(RESULTS_DIR, "daily_panel.csv")


def _daily_errors(name, train, sim, ctrl, kinds, H=48,
                  forecast_cache_dir=None):
    """Per-day MAE and BIAS for each kind and channel on one household.

    BIAS as well as MAE, and it is the column that carries the mechanism: on an
    absence day every model-based method's error is almost entirely one-signed.
    It is not that they are noisier, it is that they forecast a household that is
    not there, and |error| alone cannot say so.

    Tables come from the same two places the sweep reads them, so a number here
    and the matching arm's `con_*` checkpoint columns are the same measurement:
    `build_forecast_table` for the fit-free kinds, and `load_or_build_forecasts`
    for the fitted ones -- which is a CACHE HIT after the sweep. Against a cold
    cache this fits Prophet and the quantile regressions and takes hours.
    """
    anchors = list(ctrl.index[::H])
    frame = pd.concat([train, ctrl])
    tables = {}
    for k in kinds:
        if k in hs.SIMPLE_KINDS:
            tables[k] = hs.build_forecast_table(hs.SIMPLE_KINDS[k](frame, H),
                                                anchors, H, "30min")
        elif k in hs.PROPHET_KINDS:
            con_p, gen_p = hs.PROPHET_KIND_PARAMS[k]
            served, _ = hs.load_or_build_forecasts(
                name, train, ctrl, H, "30min",
                hs.EnergyForecaster(con_p, gen_p),
                cache_dir=forecast_cache_dir, refit_every_days=30, kind=k,
                history=train)
            tables[k] = served.table
        elif k in hs.FITTED_KINDS:
            served, _ = hs.load_or_build_forecasts(
                name, train, ctrl, H, "30min", hs.EnergyForecaster(None, None),
                cache_dir=forecast_cache_dir, kind=k, history=train)
            tables[k] = served.table
        else:
            raise ValueError(f"unknown forecaster kind {k!r}")

    day = sim.index.normalize()
    out = pd.DataFrame({"act_con": sim["Energy_Consumption"].groupby(day).sum(),
                        "act_gen": sim["Energy_Generation"].groupby(day).sum()})
    for k, t in tables.items():
        ds = pd.DatetimeIndex(pd.to_datetime(t["ds"], utc=True))
        for ch, col in (("con", "Energy_Consumption"), ("gen", "Energy_Generation")):
            fc = pd.Series(t[f"yhat_{ch}"].to_numpy(), index=ds)
            fc = fc[fc.index.isin(sim.index)].sort_index()
            err = fc - sim.loc[fc.index, col]
            grp = err.groupby(err.index.normalize())
            out[f"{k}__{ch}_mae"] = grp.apply(lambda e: e.abs().mean())
            out[f"{k}__{ch}_bias"] = grp.mean()
    return out


def _daily_money(arm, hh, output_root=None):
    """Per-day cost of the three dispatches this study can settle daily.

    `df_fc` is the MPC on the forecast and `df_pk` the same MPC on realised data,
    both written per run; `Step_Cost` is the interval's energy cost and sums over
    a day. The no-battery leg is rebuilt from the SAME file's rate and load
    columns rather than from a second run, so the three legs cannot come from
    different rate vectors.

    AU ONLY, and the restriction is not conservatism. On SI the bill carries an
    excess-power charge measured per network block against a *dogovorjena moc*
    that `Environment.converge_agreed_power` re-agrees from the peaks the
    controller itself made -- so a day is not a settleable unit there, and a
    daily "cost" would omit the term SI controllers spend their effort on.
    """
    root = RESULTS_DIR if output_root is None else output_root
    if not arm.startswith("AU"):
        raise ValueError(
            f"daily money is defined on AU only, not {arm!r}: the SI bill's "
            f"excess-power charge is monthly and endogenous to the controller's "
            f"own peaks, so a day cannot be settled on its own. The error "
            f"columns are tariff-independent and are available on both.")
    d = os.path.join(root, arm, hh)
    fc = pd.read_csv(os.path.join(d, f"df_fc_{hh}.csv.gz"), parse_dates=["Timestamp"])
    pk = pd.read_csv(os.path.join(d, f"df_pk_{hh}.csv.gz"), parse_dates=["Timestamp"])
    day = fc["Timestamp"].dt.tz_localize(None).dt.normalize()
    net = fc["Consumption"] - fc["Solar_Gen"]
    nb = np.where(net >= 0, net * fc["Buy_Rate_kWh"], net * fc["Sell_Rate_kWh"])
    return (pd.DataFrame({"day": day, "cost_nb": nb,
                          "cost_fc": fc["Step_Cost"].to_numpy(),
                          "cost_pk": pk["Step_Cost"].to_numpy()})
            .groupby("day").sum())


def build_daily_panel(arm="AU_H24", kinds=PANEL_KINDS, dataset_ids=None,
                      data_dir=None, output_root=None, H=48, delta_t=0.5,
                      n_train=730, n_sim=365,
                      start_ts="2010-07-01 00:30:00", money=True,
                      forecast_cache_dir=None, verbose=True):
    """One row per (household, day): actual, per-method error, and daily money.

    Windows come from `load_study_frames` with the sweep's own defaults, so the
    days here are the days the arms were scored on and nothing is re-cut.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    ids = hs.dataset_ids() if dataset_ids is None else dataset_ids
    frames = []
    for ident in ids:
        path = os.path.join(data_dir, f"Ausgrid {ident}.csv")
        if not os.path.isfile(path):
            warnings.warn(f"{path} not found, skipped")
            continue
        f = hs.load_study_frames(path, H=H, delta_t=delta_t, n_train=n_train,
                                 n_sim=n_sim, start_ts=start_ts)
        name = f["dataset_name"]
        one = _daily_errors(name, f["df_train"], f["df_sim"], f["df_ctrl"],
                            kinds, H=H, forecast_cache_dir=forecast_cache_dir)
        one.index.name = "day"
        one = one.reset_index()
        one["day"] = pd.DatetimeIndex(one["day"]).tz_localize(None)
        if money:
            one = one.merge(_daily_money(arm, name, output_root).reset_index(),
                            on="day", how="left")
        one.insert(0, "hh", name)
        frames.append(one)
        if verbose:
            print(f"  [daily] {name}: {len(one)} days, {len(kinds)} method(s)")
    if not frames:
        raise FileNotFoundError(f"no household files read from {data_dir}")
    panel = pd.concat(frames, ignore_index=True)

    if DROP_PARTIAL_LAST_DAY:
        last = panel["day"].max()
        n_before = len(panel)
        panel = panel[panel["day"] < last].reset_index(drop=True)
        if verbose:
            print(f"  [daily] dropped the truncated final day {last.date()} "
                  f"({n_before - len(panel)} rows): the window is cut at an "
                  f"interval, not at midnight")

    # The same provenance columns a cached scoring table carries, so a panel
    # built over a different window is NAMED by `hs.provenance` rather than
    # described by whatever the reading notebook happens to have set. This is the
    # trap `forecast_benchmark.csv` fell into before it grew these columns.
    for col, val in (("arm", arm), ("n_sim", n_sim), ("n_train", n_train),
                     ("steps_per_day", H), ("start_ts", start_ts),
                     ("kinds", "|".join(kinds)), ("money", bool(money)),
                     ("absence_rel", ABSENCE_REL),
                     ("absence_min_days", ABSENCE_MIN_DAYS),
                     ("absence_window", ABSENCE_WINDOW)):
        panel[col] = val
    return panel


def load_daily_panel(path=PANEL_PATH, arm="AU_H24", kinds=PANEL_KINDS,
                     rebuild=False, **kwargs):
    """The cached daily panel, rebuilt when the roster or the arm has moved.

    RESCORE WHEN THE ROSTER HAS MOVED, not only when the file is missing. A kind
    added to `PANEL_KINDS` after the file was written would otherwise be silently
    absent from every figure below -- and the newest method is exactly the one a
    reader would assume was included. `forecast_benchmark` learnt this the hard
    way with the `hbd` family; the check is the same one.
    """
    stale = None
    if not rebuild and os.path.isfile(path):
        cached = pd.read_csv(path, parse_dates=["day"])
        have = set(str(cached.get("kinds", pd.Series([""])).iloc[0]).split("|"))
        arm_ok = str(cached.get("arm", pd.Series([""])).iloc[0]) == arm
        missing = set(kinds) - have
        if not missing and arm_ok:
            return add_absence(cached), f"cached: {path}"
        stale = (f"rebuilding {os.path.basename(path)}: "
                 + (f"{sorted(missing)} not in it" if missing else "")
                 + ("" if arm_ok else f"; it holds arm "
                    f"{cached.get('arm', pd.Series(['?'])).iloc[0]}, not {arm}"))
    if stale:
        print(stale)
    panel = build_daily_panel(arm=arm, kinds=kinds, **kwargs)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    panel.to_csv(path, index=False)
    print(f"[daily] {len(panel)} rows -> {path}")
    return add_absence(panel), f"built: {path}"
