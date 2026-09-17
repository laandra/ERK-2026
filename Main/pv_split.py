"""Which arms belong to which of the three PV notebooks, derived not restated.

`CODE.ipynb` ran every arm of the sweep in one notebook, which put two different
questions on one axis. A `*_pvtruth` arm is not another forecaster competing with
Prophet: it is Prophet's own load model handed a roof it cannot have. Ranked
beside the deployable methods it reads as the best forecast in the study, and the
sentence a reader takes away -- "perfect PV wins" -- is not a finding about
forecasting at all.

So the roster is split into three tracks, and each gets a notebook:

  forecast   ONE method answers BOTH channels. Every arm here is deployable and
             every comparison inside it is between things a household could
             install. `CODE_PV_FORECAST.ipynb`.
  perfect    the roof is `TruthForecaster` -- tomorrow's generation, exactly --
             on top of a real load model. Not deployable; the bound on what the
             PV channel can still buy. `CODE_PV_PERFECT.ipynb`.
  mixed      one channel Prophet, the other a naive roof (`pvnaive`,
             `pvmedian14`). Neither deployable-uniform nor perfect: these are the
             INTERMEDIATE rungs of the roof-quality ladder, and they only mean
             anything next to the two ends of it, so they live in the comparison
             notebook rather than in either track. `CODE_PV_VALUE.ipynb`.

THE PARTITION IS COMPUTED FROM `hs.HYBRID_KINDS`, never listed here. An arm added
to `hems_study.STUDY_ARMS` tomorrow lands in the right notebook without anyone
editing this file, and -- the part that matters -- it cannot land in TWO, which
is the failure mode that would have two kernels writing one checkpoint.

WHAT EACH NOTEBOOK MAY WRITE, since three of them now share one cache tree:

  results_local/<arm>/<household>/   partitioned BY ARM, and the tracks are
                                     disjoint by construction, so two notebooks
                                     running at once never write one checkpoint.
  forecast_cache/, oracle_cache/,    keyed on content, shared on purpose. This
  hbd_params/                        is the sharing the split has to preserve:
                                     the oracle and every rule are forecast-blind,
                                     so one solve per (household, tariff,
                                     horizon) serves the `forecast` arm and its
                                     `perfect` twin alike. Writes are atomic.
  results_local/forecast_benchmark.csv   SINGLE WRITER: the `forecast` notebook.
  results_local/prophet_tuning.csv       The other two read them and never
                                     rescore -- see `read_screen` below.
  Results/Figures/<subdir>/          one subdir per notebook (`FIGURE_SUBDIR`),
                                     because every figure name in the three is
                                     the same and one shared subdir means the
                                     last notebook run silently owns them all.
"""

import os

import pandas as pd

import hems_study as hs

FORECAST, PERFECT, MIXED = "forecast", "perfect", "mixed"
TRACKS = (FORECAST, PERFECT, MIXED)

# Where each notebook exports. Not one subdir with different names inside it:
# the two track notebooks draw the SAME figures over different arms, so they
# collide on every single name.
FIGURE_SUBDIR = {
    FORECAST: "hems_pv_forecast",
    PERFECT:  "hems_pv_perfect",
    MIXED:    "hems_pv_value",
}

TRACK_TITLE = {
    FORECAST: "one forecast for both channels",
    PERFECT:  "perfect PV, forecast load",
    MIXED:    "the roof-quality ladder",
}


def channel_sources(kind: str) -> tuple:
    """`(consumption source, generation source)` for a forecaster kind.

    A non-hybrid kind answers both channels itself, which is the whole content
    of the `forecast` track: `("median14", "median14")`, not a special case.
    """
    return hs.HYBRID_KINDS.get(kind, (kind, kind))


def load_kind(kind: str) -> str:
    """The LOAD model inside a kind -- `median14_pvtruth` -> `median14`.

    This is what pairs a `perfect` arm with its `forecast` twin, and what lets a
    perfect-PV arm be looked up in `forecast_benchmark.csv`, which scores load
    models and has never heard of `median14_pvtruth`.
    """
    return channel_sources(kind)[0]


# How to SAY a source in a controller label. `hs.FORECAST_KIND_LABELS` says it
# for a whole kind ("median of 14 d + perfect PV") and that is the right phrase
# on a forecast-quality axis, where the kind IS the subject. It is the wrong
# phrase inside "MPC-MILP 24 h, ..." -- a controller label names the controller
# and then what drives it, and "MPC-MILP 24 h, PV: perfect" reads as a controller
# called PV. These are the per-CHANNEL words `forecast_label` assembles from.
SOURCE_LABEL = {
    "prophet":       "Prophet",
    "prophet_tuned": "tuned Prophet",
    "persistence":   "yesterday",
    "weekly":        "last week",
    "daytype":       "last like-day",
    "mean3":         "mean of 3 d",
    "mean7":         "mean of 7 d",
    "median7":       "median of 7 d",
    "median14":      "median of 14 d",
    "hbd":           "season + AR",
    "hbd_baseline":  "season only",
    "hbd_median14":  "median of 14 d + AR",
    "truth":         "perfect",
}


def source_label(source: str) -> str:
    return SOURCE_LABEL.get(source, source.replace("_", " "))


def forecast_label(kind: str) -> str:
    """What drives the MPC in an arm, as a phrase that follows a comma.

    One method on both channels contracts to one phrase -- "Prophet forecast",
    not "Prophet load, Prophet PV" -- because the channel split is only worth
    naming where the channels differ, and that is exactly where a reader needs
    it: `pvtruth` becomes "Prophet load, perfect PV", which is the one thing
    separating the perfect-PV notebook's reference arm from its twin's.
    """
    con, gen = channel_sources(kind)
    if con == gen:
        return f"{source_label(con)} forecast"
    return f"{source_label(con)} load, {source_label(gen)} PV"


def arm_forecast_kind(arm: str) -> str:
    """The forecaster kind an arm's MPC reads, `prophet` where unstated."""
    return arm_spec(arm).get("forecaster_kind", "prophet")


def mpc_name(arm: str, prefix: str = "MILP+") -> str:
    """The forecast-driven MPC of one arm, named by what actually drives it.

    The controller KEY is `prophet` in every arm of the study, and the controller
    is Prophet in only some of them: on `AU_H24_median14` that row is a 14-day
    median, and on `AU_H24_pvtruth` it is a Prophet load model reading tomorrow's
    roof. A table headed "controller - MILP+Prophet" over either one names a
    forecaster that arm never ran -- which on the perfect-PV track is every table
    in the notebook.
    """
    return prefix + forecast_label(arm_forecast_kind(arm)).replace(" forecast", "")


def track_of_kind(kind: str) -> str:
    con, gen = channel_sources(kind)
    if gen == "truth":
        return PERFECT
    return FORECAST if con == gen else MIXED


def arm_track(arm) -> str:
    """The track of one arm, given its spec dict or its name."""
    spec = arm if isinstance(arm, dict) else arm_spec(arm)
    return track_of_kind(spec.get("forecaster_kind", "prophet"))


def arm_spec(name: str) -> dict:
    for a in hs.STUDY_ARMS:
        if a["name"] == name:
            return a
    raise KeyError(f"{name!r} is not in hems_study.STUDY_ARMS")


def arm_specs(track: str) -> list:
    """The `STUDY_ARMS` entries of one track, in roster order.

    Hand this to `hs.run_arms(arms=...)`: it is the sweep the notebook owns, and
    nothing outside it.
    """
    _check(track)
    return [a for a in hs.STUDY_ARMS if arm_track(a) == track]


def arm_names(track: str) -> list:
    return [a["name"] for a in arm_specs(track)]


def reference_arm(track: str, control_horizon: int = 48) -> dict:
    """`{tariff: arm}` -- the arm a track's controller figures are drawn from.

    DERIVED FROM `hs.REFERENCE_ARM`, so the two tracks answer the same question
    about the same load model: the `forecast` track keeps `AU_H24`/`SI_H24`, and
    the `perfect` track takes the arm carrying that arm's load model with a
    perfect roof on top -- `AU_H24_pvtruth`/`SI_H24_pvtruth`. Naming those
    directly would be two more strings to keep in step with a reference arm that
    is defined elsewhere.
    """
    _check(track)
    out = {}
    for tariff, base in hs.REFERENCE_ARM.items():
        base_kind = arm_spec(base).get("forecaster_kind", "prophet")
        if track == FORECAST:
            out[tariff] = base
            continue
        for a in arm_specs(track):
            if (a["tariff"] == tariff
                    and a.get("control_horizon") == control_horizon
                    and load_kind(a.get("forecaster_kind", "prophet")) == base_kind):
                out[tariff] = a["name"]
                break
    return out


def forecast_arms(track: str, tariff: str, control_horizon: int = 48) -> dict:
    """`{kind: arm}` for one track and tariff -- `hs.forecast_arms`, narrowed.

    The exclusions that function makes are the ones that matter here too: the
    current-interval leak carries `forecaster_kind == "prophet"` and the no-wear
    arms carry no kind at all, so both would answer "which arm shows me Prophet?"
    and neither is a forecast.
    """
    _check(track)
    everything = hs.forecast_arms(tariff, control_horizon)
    return {k: a for k, a in everything.items() if track_of_kind(k) == track}


def study_kind(track: str) -> str:
    """The kind a track's figures mark as "the arm the paper is about"."""
    ref = reference_arm(track)
    if not ref:
        return "prophet"
    return arm_spec(sorted(ref.values())[0]).get("forecaster_kind", "prophet")


# ---------------------------------------------------------------------------
# The pairing the third notebook is built on
# ---------------------------------------------------------------------------

def pv_pairs(tariff: str, control_horizon: int = 48) -> list:
    """`[(load kind, forecast arm, perfect arm)]` -- one row per load model.

    The two arms differ in the GENERATION CHANNEL AND NOTHING ELSE: same
    household, same battery, same tariff, same horizon, same load model, same
    evaluator, same forecast-blind oracle out of the same cache entry. That is
    what makes the per-household difference between them the value of a perfect
    PV forecast rather than a difference between two studies.

    A load model with no perfect twin (`hbd`, `hbd_baseline`) is simply absent --
    there is no arm to difference against, and inventing one from a neighbouring
    model would answer a question nobody asked.
    """
    by_load = {}
    for track in (FORECAST, PERFECT):
        for kind, arm in forecast_arms(track, tariff, control_horizon).items():
            by_load.setdefault(load_kind(kind), {})[track] = arm
    rows = []
    for kind in hs.FORECAST_KIND_LABELS:            # the roster's own order
        pair = by_load.get(kind, {})
        if FORECAST in pair and PERFECT in pair:
            rows.append((kind, pair[FORECAST], pair[PERFECT]))
    return rows


def roof_ladder(tariff: str, control_horizon: int = 48) -> list:
    """`[(generation source, arm)]`, load held at ONE model, roof varying.

    The other reading of the same question. `pv_pairs` is a step -- forecast roof
    to perfect roof -- and says nothing about what lies between; this is the
    ladder, from copying yesterday's roof up to knowing it, with the consumption
    channel pinned so the roof is the only thing moving. `pvnaive` and
    `pvmedian14` exist for exactly this and for nothing else, which is why they
    are not in either track.

    Ordered from the crudest roof to the truth. NOT by measured skill, and the
    difference matters on exactly one rung: the pinned model sits second from the
    top because it is the study's own forecaster, and on the generation channel
    it scores BELOW both naive rungs beneath it (-0.17 skill against a 14-day
    median's +0.11). So a dip at that rung is the roster telling the truth about
    itself, not the ladder being drawn out of order -- and a figure that sorted
    the rungs by measured skill would hide it by construction.
    """
    base = hs.REFERENCE_ARM.get(tariff)
    if base is None:
        return []
    pinned = arm_spec(base).get("forecaster_kind", "prophet")
    rungs = {}
    for a in hs.STUDY_ARMS:
        if a["tariff"] != tariff or a.get("control_horizon") != control_horizon:
            continue
        if a.get("leak_current_interval") or "cycle_cost_eur_per_efc" in a:
            continue
        con, gen = channel_sources(a.get("forecaster_kind", "prophet"))
        if con == pinned:
            rungs.setdefault(gen, a["name"])
    order = ["persistence", "median7", "median14", pinned, "truth"]
    tail = [g for g in rungs if g not in order]
    return [(g, rungs[g]) for g in order + tail if g in rungs]


# ---------------------------------------------------------------------------
# The two screens the tracks share, and which notebook owns them
# ---------------------------------------------------------------------------

SCREEN_FILES = {
    "benchmark": "forecast_benchmark.csv",
    "tuning":    "prophet_tuning.csv",
}


def screen_path(which: str) -> str:
    return os.path.join(hs.RESULTS_DIR, SCREEN_FILES[which])


def read_screen(which: str, owner: str = "CODE_PV_FORECAST.ipynb") -> pd.DataFrame:
    """One of the shared screens, READ-ONLY, with the kind order restored.

    Both screens score FORECASTS, not dispatches: they are keyed on the kind and
    the household and know nothing about a tariff or an arm, so three notebooks
    rescoring them would produce three identical files at one path. They have one
    writer -- the `forecast` notebook, which is the one whose subject they are --
    and everyone else reads. The alternative is two kernels writing one
    non-atomic CSV, which is a truncated file rather than a conflict.
    """
    path = screen_path(which)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{os.path.basename(path)} has not been written yet.\n"
            f"It is scored by {owner}, which owns it; run that notebook's screen "
            f"cell once and this one will read it."
        )
    df = pd.read_csv(path)
    col = "kind" if which == "benchmark" else "config"
    universe = (hs.FORECAST_KIND_LABELS if which == "benchmark"
                else hs.PROPHET_TUNING_GRID)
    order = [k for k in universe if k in set(df[col])]
    if order:
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def _check(track: str) -> None:
    if track not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}, not {track!r}")


def describe(track: str) -> str:
    """One block naming what a notebook owns -- printed by its config cell."""
    _check(track)
    specs = arm_specs(track)
    ref = reference_arm(track)
    lines = [
        f"track          : {track} -- {TRACK_TITLE[track]}",
        f"arms           : {len(specs)} of {len(hs.STUDY_ARMS)} in the roster",
        f"reference arms : {ref}",
        f"figures        : Results/Figures/{FIGURE_SUBDIR[track]}/",
    ]
    w = max((len(a["name"]) for a in specs), default=0)
    for a in specs:
        kind = a.get("forecaster_kind", "prophet")
        con, gen = channel_sources(kind)
        extra = {k: v for k, v in a.items()
                 if k not in ("name", "tariff", "forecaster_kind")}
        lines.append(f"   {a['name']:{w}s}  load={con:<14s} roof={gen:<14s} {extra}")
    return "\n".join(lines)
