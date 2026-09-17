"""The house style two notebooks share, so they cannot drift apart.

Everything here used to live in `CODE.ipynb` cell 10 and cell 11. It moved out
the moment a SECOND notebook needed it: a palette semantics defined twice is a
second source of truth, and the figure that forgets to update is always the one
just added -- which is the same argument `hems_study.CONTROLLER_ALGORITHM` makes
about label dicts, applied one level up.

What stays in the notebooks is what closes over run data: `make_labels`, `LABEL`,
`SAMPLE`, `nowear_twin` and `with_nowear` all need `df_all` or `long`, and a
module-level copy of those would be a cache of a frame the notebook is still
loading.

WHAT A COLOUR MEANS, since that is the part a new figure gets wrong:
colour is the TARIFF (AU blue, SI orange). Within a tariff the controller family
is a marker SHAPE (the strong cue) and a shade of the same hue (the weak one),
never a new colour. Sign, where a figure encodes it, is green/red -- deliberately
not green/orange, because orange is already SI and a reader who has learnt that
over four figures cannot be asked to unlearn it here.

seaborn: import it, but never call `sns.set_theme()` or `sns.set_style()`. They
overwrite the rcParams `pf.use()` sets and every figure in the study would
silently change font, grid and background. Pass `ax=` and an explicit
`color=`/`palette=` taken from `SERIES` instead.
"""

import textwrap
import warnings

import numpy as np

import Plotting_Functions as pf
from Plotting_Functions import INK, INK_2, MUTED, SERIES, SURFACE  # noqa: F401

import hems_study as hs

# ---------------------------------------------------------------------------
# Colour, shape and shade
# ---------------------------------------------------------------------------
TARIFF_COLOR = {"AU": SERIES[0], "SI": SERIES[1]}

# Sign, and only sign. The bar's direction from zero says the same thing, so the
# colour is never load-bearing on its own.
BEATS, LOSES = SERIES[2], "#c0392b"

# What ALGORITHM produced a row. The table lives in `hems_study`
# (CONTROLLER_ALGORITHM) rather than here, for the reason the module docstring
# gives; this only derives the family from it.
FAMILY = {c: hs.controller_family(c) for c in hs.CONTROLLER_ALGORITHM}

# Shade and marker by algorithm family, so a reader can tell an RBC row from an
# MPC row from the optimum without reading a single label.
#
# Widened from 0.75/0.42/0.12/0.0. Those four steps are mixed toward white and
# then drawn at alpha=0.6 on the ranking figure, where 0.12 and 0.0 are the same
# colour to the eye and shape was carrying the whole family encoding on its own.
# Shade is still the WEAK cue -- marker shape is the strong one, and the key on
# the ranking figure names both -- but a redundant cue that cannot be seen is
# not redundancy, it is decoration.
FAMILY_SHADE = {"reference": 0.72, "RBC": 0.46, "MPC": 0.22, "MILP": 0.0}
FAMILY_MARKER = {"reference": "x", "RBC": "o", "MPC": "*", "MILP": "D"}

# WHAT THE SHAPES MEAN, for a key inside the image. The figures are built around
# the RBC/MPC/MILP distinction -- it is what the shape and the shade both encode
# -- and it was once documented nowhere a reader of the figure could see.
FAMILY_LABEL = {"reference": "no battery", "RBC": "rule-based (RBC)",
                "MPC": "MPC-MILP, receding horizon",
                "MILP": "MILP, full-year horizon"}

# The star arms borrow their twin's family, marker and shade -- they ARE the same
# algorithm -- and are told apart by the star in the name and by a hollow face.
# Registered here so `ctrl_marker`, `ctrl_color` and `tick_label` need no special
# case.
for _c in list(FAMILY):
    FAMILY[_c + "*"] = FAMILY[_c]


def is_nowear(controller):
    """True for the `<name>*` rows: the same MILP with no degradation term."""
    return str(controller).endswith("*")


def ctrl_color(tariff, controller):
    """Fill for one (tariff, controller): the tariff's hue, shaded by family."""
    return pf.mix(TARIFF_COLOR[tariff], "#ffffff",
                  FAMILY_SHADE[FAMILY.get(controller, "RBC")])


def ctrl_marker(controller):
    return FAMILY_MARKER[FAMILY.get(controller, "RBC")]


# ---------------------------------------------------------------------------
# Axis labels
# ---------------------------------------------------------------------------
# The two rows that wrapped to a second line, shortened for an axis only.
# `LABEL` in the notebook keeps the full text -- the wide-label figures and every
# printed table read it, and `hs.controller_label` stays the single source of
# truth. At width 30 `self_consumption_peak_shaving` went to two lines on ONE
# character over, and `price_oracle` on genuine length; on the SI ranking panel,
# which carries both, sixteen rows of mixed one- and two-line labels printed into
# each other. `(diagnostic)` is dropped rather than wrapped because no other
# non-deployable row carries it.
TICK_SHORT = {"price_oracle": "price threshold, full-year prices"}


def tick_label(c, label_map, width=34):
    """One controller's name for a y axis: family prefix dropped, then wrapped.

    The prefix is what the marker shape and every figure's family key already
    say, and it is the same five characters on eight of eleven rows -- so it
    spends a third of the axis width repeating the one thing a reader can see at
    a glance. What is left is the part that differs between rows.

    `label_map` is the notebook's per-arm `LABEL`: the horizon is part of an MPC
    controller's identity, so the map has to be built against an arm and cannot
    be a module constant.
    """
    base = c[:-1] if is_nowear(c) else c
    lab = TICK_SHORT.get(base, label_map.get(base, base))
    for prefix in ("RBC: ", "MPC-MILP ", "MILP, "):
        if lab.startswith(prefix):
            lab = lab[len(prefix):]
            break
    # The star is re-appended after the lookup rather than carried through it,
    # so a shortened name and its no-wear twin cannot drift apart.
    return textwrap.fill(lab + (" *" if is_nowear(c) else ""), width)


# ---------------------------------------------------------------------------
# The two readings of every money figure, and the four denominators
# ---------------------------------------------------------------------------
# ONE YEAR or a PACK LIFE. Every money figure is drawn twice, because the two
# answer different questions and the article needs both: the annual reading ranks
# controllers against each other on the year the sweep simulated, and the
# lifetime reading is the one a household decides on.
#
# These are not a rescaling of each other. The lifetime columns discount at 5 %
# over each ROW'S service life (`pv_factor`), so wherever two rows have different
# lives -- which is what a MILP with no degradation term can cause, by cycling a
# pack past its rated 6000 EFC inside the 12 y calendar band -- the ratio between
# them moves between the views.
VIEWS = [
    dict(key="annual", suffix="",
         saving="saving_annual_net", operating="saving_operating",
         wear="wear_eur", ret="npv",
         per="per household-year",
         note="one simulated year"),
    dict(key="lifetime", suffix="_lifetime",
         saving="lifetime_saving", operating="lifetime_saving_operating",
         wear="lifetime_wear", ret="npv",
         per="over the pack's life, at 5 %/a",
         note="the 12 y calendar band, discounted at 5 %/a"),
]

# WHAT PERCENT OF WHAT. A share is a number with a % after it until its
# denominator is said out loud, and this study has four plausible ones. Each
# figure names the one it uses; the strings live here so two figures cannot
# describe the same column differently.
PCT_OF_BILL = "% of the no-battery bill"
PCT_OF_GAIN = "% of what perfect foresight wins"
# THE SAME DENOMINATOR IN FEWER WORDS, and nothing else. Both name
# `regret_pct_of_gain`; this one exists because an axes xlabel is centred on the
# AXES, and a half-width panel holds about 36 characters -- the long form is 41,
# so on a two-panel figure the two copies met in the middle of the canvas and ran
# off both ends. Any figure that uses it defines "achievable" in its caption, in
# the long form's own words. Do not add a third.
PCT_OF_GAIN_SHORT = "% of the achievable gain"
PCT_OF_CAPEX = "% of the capital"
# `lifetime_saving_pct` divides by `baseline_cost_total * pv_factor`, the bill
# over the pack's life -- NOT the year's bill. Naming the denominator is shorter
# than qualifying it, and a qualifier this long ran off both ends of the axis.
PCT_OF_LIFETIME_BILL = "% of the lifetime no-battery bill"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def beeswarm(vals, height=0.34, bins=26):
    """Offsets that spread coincident points instead of stacking them.

    Random jitter was hiding the mode: thirty households on one row with s=30
    markers overlap hardest exactly where the density is, so the fattest part of
    every distribution read as the sparsest. This bins x and spreads each bin
    symmetrically about the row, which is the one thing jitter cannot do -- the
    spread is the count.
    """
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return v
    lo, hi = np.nanmin(v), np.nanmax(v)
    idx = np.zeros(len(v), dtype=int) if hi - lo < 1e-12 else \
        np.clip(((v - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    off = np.zeros(len(v))
    for b in np.unique(idx):
        w = np.where(idx == b)[0]
        n = len(w)
        # Centred, alternating out from the row: -1, +1, -2, +2 ... scaled so a
        # full bin reaches `height` and a lone point sits exactly on the row.
        rank = np.arange(n) - (n - 1) / 2.0
        off[w[np.argsort(v[w])]] = rank / max(n - 1, 1) * 2.0 * height
    return off


def clip_window(pooled, medians, q=0.02, pad=0.06):
    """The x range to show, when one household would otherwise set it.

    On the lifetime panels a single household at -40 % stretches the axis over a
    range three quarters of the data never uses, and the mass of every row is
    squeezed into a third of the width. The answer is a clipped axis, NOT a
    dropped point: the caller draws anything outside the window on the boundary
    as a sideways caret and names the count in the key, so a reader is told
    exactly what is off the edge and how much of it there is.

    The window is then widened to hold every median, because a median outside it
    would be a statistic the figure asserts and does not show. Returns None on an
    empty panel, which is what keeps a partial sweep drawable.
    """
    v = np.asarray([x for x in pooled if x == x], dtype=float)
    if len(v) == 0:
        return None
    lo, hi = float(np.quantile(v, q)), float(np.quantile(v, 1.0 - q))
    m = [float(x) for x in medians if x == x]
    if m:
        lo, hi = min(lo, min(m)), max(hi, max(m))
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span


# ---------------------------------------------------------------------------
# Statistics the figures have to show, not just quote
# ---------------------------------------------------------------------------
# A point estimate with no interval is the defect this module was extended to
# fix. Every figure in the study reports a median or a rho and none of them said
# how firmly it was held, so two bars a reader can see are different lengths --
# 35 and 36 -- carried no way of telling whether the difference exists.
#
# BOOTSTRAP, not a normal approximation: n is 22 to 30 households of cost ratios,
# a handful of them far out (large exporters, dead arrays), and the statistic
# being resampled is usually a MEDIAN, which has no convenient closed form here.

def boot_ci(values, stat=np.median, n_boot=10_000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for `stat` over `values`.

    Percentile rather than BCa: with n around 25 the acceleration term is
    estimated off a jackknife of the same 25 points and adds noise rather than
    accuracy, and nothing here turns on the last half percent of coverage.
    Returns (lo, hi), or (nan, nan) when fewer than 3 finite values survive --
    an interval over two points is not an interval.
    """
    v = np.asarray([x for x in np.ravel(values) if x == x], dtype=float)
    if len(v) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = stat(rng.choice(v, size=(n_boot, len(v)), replace=True), axis=1)
    return (float(np.quantile(draws, alpha / 2.0)),
            float(np.quantile(draws, 1.0 - alpha / 2.0)))


def boot_ci_paired(a, b, stat=np.median, n_boot=10_000, alpha=0.05, seed=0):
    """The same, for the PAIRED difference of two aligned samples.

    The pairing is the whole argument of the study's section 5: every household
    runs under every arm, so the unit is the per-household difference and
    resampling the two arms independently would throw that away and widen the
    interval to something the data does not say.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = (a == a) & (b == b)
    return boot_ci(a[ok] - b[ok], stat=stat, n_boot=n_boot, alpha=alpha,
                   seed=seed)


def spearman_ci(x, y, n_boot=10_000, alpha=0.05, seed=0):
    """Spearman rho with its p and a bootstrap CI, as one call.

    rho and p were already printed on three figures; the INTERVAL was the missing
    half. `rho = -0.64, p = 0.119` and `rho = -0.64, 95 % CI [-0.93, +0.14]` are
    the same two numbers and only the second one says what the figure supports.
    Pairs are resampled together -- resampling x and y apart would test a null
    hypothesis, not estimate an interval.
    """
    from scipy import stats as _st

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = (x == x) & (y == y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(x)
    rho, p = _st.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    # A resample CAN be constant -- on n = 7 the chance of drawing one value
    # seven times is small but not zero -- and Spearman is undefined there. That
    # is a legitimate draw with no statistic, so it is dropped below rather than
    # warned about once per occurrence into the middle of a figure's output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _st.ConstantInputWarning)
        draws = np.array([_st.spearmanr(x[i], y[i]).statistic for i in idx])
    draws = draws[draws == draws]
    if len(draws) < 3:
        return float(rho), float(p), float("nan"), float("nan"), len(x)
    return (float(rho), float(p),
            float(np.quantile(draws, alpha / 2.0)),
            float(np.quantile(draws, 1.0 - alpha / 2.0)), len(x))


def rho_lines(label, rho, p, lo, hi, n):
    """The corner block a correlation panel prints, as text lines.

    One wording, used by every panel that reports a rho, so two figures cannot
    describe the same test differently. `p < 0.05` is rendered as a VERDICT
    rather than left as a bare number: what a figure has to support is "does this
    panel carry the claim", and the threshold is the whole of it.
    """
    ci = "" if lo != lo else f"\n95 % CI [{lo:+.2f}, {hi:+.2f}]"
    verdict = "significant" if p == p and p < 0.05 else "NOT significant"
    return (f"{label}\nSpearman ρ {rho:+.2f}{ci}\n"
            f"n = {n}, p = {p:.3f}\n{verdict} at 5 %")


def corner_block(ax, text, color, values, lims=None, fontsize=8):
    """Print `text` in whichever top/bottom corner the data is not in.

    Fixed at bottom-left, this block printed straight through a label the moment
    two panels stopped sharing a y range -- and on ONE shared axis the panels
    occupy different bands, which is exactly when a figure is most likely to grow
    a shared axis. The rule is mechanical: compare the panel's own median to the
    midpoint of the axis and take the far corner.
    """
    v = np.asarray([x for x in np.ravel(values) if x == x], dtype=float)
    if lims is None:
        lims = (v.min(), v.max()) if len(v) else (0.0, 1.0)
    low = len(v) and np.median(v) < 0.5 * (lims[0] + lims[1])
    y, va = (0.97, "top") if low else (0.03, "bottom")
    ax.annotate(text, xy=(0.03, y), xycoords="axes fraction", ha="left", va=va,
                color=color, fontsize=fontsize)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, in the caller's order.

    A figure that tests every method against the panel's best runs thirteen
    comparisons and then marks whichever cleared 0.05 -- which at thirteen tries
    finds one by chance about half the time. Holm rather than Bonferroni because
    it is uniformly more powerful at the same family-wise error rate, and rather
    than Benjamini-Hochberg because the claim being made is "this method is
    distinguishable from the best", one at a time, not "some of these are".
    """
    p = np.asarray(pvalues, dtype=float)
    ok = np.where(p == p)[0]
    out = np.full(len(p), float("nan"))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out
