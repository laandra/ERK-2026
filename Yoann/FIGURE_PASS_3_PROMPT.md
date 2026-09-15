# Prompt: figure pass 3 on `Yoann/CODE.ipynb`

Paste everything below the line into a fresh session, working in `/Users/summerscholl/Documents/ERK-2026`.

---

Work in `Yoann/`. The notebook is `Yoann/CODE.ipynb`; every figure goes through
`../Plotting_Functions.py` (imported as `pf`), and the study's data layer is `hems_study.py`
(imported as `hs`). Two prior passes closed every item in `FIGURE_FIXES_PROMPT.md` except
§7.4, and added a no-degradation MILP ablation. **Read `FIGURE_FIXES_PROMPT.md`'s
"Status, 2026-09-14" section first** — it is the current state and it explains why several
figures look the way they do. Do not undo those decisions without a reason.

The sweep is complete and cached: 35 arms × 30 households, 12 570 rows. Nothing here needs
a re-run. `python3 test_hems_study.py` reports **67 passed, 0 failed** and must still do so
when you finish — it does not test figures, but it catches an accidental edit to the data
layer.

## How to iterate without a Jupyter UI

```python
import sys, os, warnings, json, io, contextlib
warnings.filterwarnings("ignore")
os.chdir("/Users/summerscholl/Documents/ERK-2026/Yoann")
sys.path.insert(0, "."); sys.path.insert(0, "..")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
nb = json.load(open("CODE.ipynb"))
import Plotting_Functions as pf; pf.SAVE_PDF = False   # don't write into Results/ while iterating
g = {}
for i in (1, 3, 5, 7, 10, 12, 13):                     # setup cells
    src = "".join(nb["cells"][i]["source"])
    if i == 3: src = src.replace("RUN_SWEEP = True", "RUN_SWEEP = False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(src, g)

_orig = pf.show
def _show(fig=None, name=None, **kw):
    f = fig or plt.gcf(); f.savefig(f"/tmp/{name}.png", dpi=110, bbox_inches="tight"); plt.close(f)
pf.show = _show; g["pf"].show = _show

exec("".join(nb["cells"][17]["source"]), g)            # the figure under test
```

Then **Read the rendered PNG and judge it with your eyes.** A figure you have not looked at
is not fixed. The skill figure reads `bench` from cell 13 and the skill-vs-regret figure also
reads `ARM_FOR` from cell 12, which is why the setup list is `(1, 3, 5, 7, 10, 12, 13)`.

Read the `dataviz` skill before writing chart code.

## Cell map

| cell | exports |
|---|---|
| 10 | shared styling: `VIEWS`, `tick_label`, `with_nowear`, `is_nowear`, `PCT_OF_*`, `beeswarm` |
| 11 | `controller_ranking_{au,si}[_lifetime]` |
| 12 | `forecast_channel_regret[_lifetime]` |
| 14 | `forecast_method_skill` |
| 15 | `skill_vs_regret` |
| 17 | `wear_vs_saving_{au,si}[_lifetime]` |
| 18 | `lifetime_economics_{au,si}[_money]` |
| 19 | `break_even_capex_{au,si}` |
| 20 | `horizon_effect` |

Nineteen exports. **Do not rename an existing one** — the names are what the LaTeX side
references. New figures get new names.

## The house style you must stay inside

`pf.use(subdir="hems", titles=False)` is set in cell 10. Colour means the **tariff** (AU
`SERIES[0]` blue, SI `SERIES[1]` orange); within a tariff, marker **shape** means the
controller family (`FAMILY_MARKER`) and shade means the same thing more weakly
(`FAMILY_SHADE`). Sign is green/**red** (`SERIES[2]` / `#c0392b`) and deliberately not
orange — orange is the SI tariff everywhere else, and the two collided in an earlier draft.
A `*` suffix on a controller name means the same MILP with no degradation term; those are
drawn hollow and every key that shows one defines the star.

`titles=False` lifts axes titles out of the image into the export's `SOURCE.md` caption, so
**panel identity can never live in a title** — use an in-axes corner mark or the x label.
Finish figures with `pf.chart_frame(fig, title, subtitle, handles=..., ncol=...)` plus
`pf.show(fig, "<stable_name>", layout="frame")`; `chart_frame` reserves the header and legend
bands in inches and `layout="frame"` stops `show` from running `tight_layout` again and
taking them back.

---

# 1. §7.4 — household heterogeneity. **Read the warning before designing it.**

The goal, from the original list: *"which kind of household is this controller bad at"*. The
caveats cell notes that on Ausgrid 127 the PV model scores −0.13 skill against seasonal-naive
while consumption scores +0.21, and on Ausgrid 148 the ordering is the other way round — so
the roster average hides a real disagreement between households, and nothing plots it.

**The warning.** The original list proposed *"a per-household strip of prophet regret ordered
by cluster"*. Check `hs.study_units()` before you write that:

```
shape (30, 2); columns ['cluster', 'dist_to_centroid']
cluster.value_counts() -> every cluster has exactly 1 household
```

The 30 households are the centroid-nearest member of each of 30 k-means clusters, so **the
cluster id is 1:1 with the household**. Grouping or faceting by cluster gives thirty groups
of one — which is precisely the n=1-median trap the previous pass spent a marker-area
encoding, an `n` column and a suppressed connector on removing from the lifetime figure.
Do not reintroduce it. Ordering rows by cluster id is likewise arbitrary: the ids carry no
order.

`dist_to_centroid` *is* meaningful — how representative each household is of the shape it
stands for — and so are the per-household quantities already in `df_all`: `con_skill_vs_naive`,
`gen_skill_vs_naive`, `con_nmae`, `gen_nmae`, `buy_*`, `sell_*`, and in `long`
`regret_pct_of_gain`, `saving_total_pct`, `efc`, `roi_pct`, `break_even_capex`.

**What to build.** One figure, `household_heterogeneity` (or two panels, one per tariff).
One row or point per household, ordered by something that carries a reading — the household's
own regret, or a load/PV characteristic it can be explained by — not by cluster id. The
question it must answer is "is Prophet's regret concentrated in a few households, and do they
have anything in common?". Some options, in the order I would try them:

- **Regret against a household characteristic**, one point per household, faceted by tariff:
  `regret_pct_of_gain` for the `prophet` controller on y, and on x either the household's PV
  share of its own load (derivable from `sell_no_battery` / `buy_no_battery`) or its forecast
  skill. A correlation here is the answer; the absence of one is also an answer, and is worth
  saying with the same `n` and `p` treatment `skill_vs_regret` now carries (that figure is
  the pattern — copy its Spearman + n + p + verdict annotation).
- **A sorted per-household strip** of `regret_pct_of_gain`, one row per household, with the
  two channel skills (`con_skill_vs_naive`, `gen_skill_vs_naive`) as a second encoding on
  the same row. This directly shows the 127-vs-148 disagreement the caveats describe.

Whichever you pick, if the honest answer is "the households do not cluster into kinds", say
that in the caption rather than drawing a figure that implies structure that is not there.
State `n` wherever a correlation or a median appears.

# 2. Readability pass over all nineteen figures

Render every one and look at it. Fix what a reader has to squint at. Specific things already
suspected, plus whatever you find:

- **Long wrapped y-tick labels crowd their neighbours** on the tall dumbbell figures (11, 17,
  19) — thirteen rows of two-line labels at `labelsize=8`. Either shorten the names in
  `tick_label` further, grow the ratio, or move to a single line with a gutter.
- **Cell 11's `_lifetime` panels have a long left tail** (one household at −40 %) that
  compresses the mass of the distribution into a third of the axis. Consider a clipped axis
  with an out-of-range marker, or a log-ish treatment, but do not silently drop the outlier.
- **Cell 18's `n=30/29 unpaired` annotations** are 6.5 pt and sit right against the markers.
- **Cell 15's labels** are placed by a 4-level alternating offset with leader lines; check it
  still holds now that the full 30-household panel moved the points.
- **Value-label collisions with bar ends** on cells 12 and 14 at small figure scales.

Do not change what a figure *claims* in the name of readability. If a fix would change the
statistic, stop and say so.

# 3. Figure width — **verify the premise before changing anything**

The request was "expand them to the whole page width instead of only the column width."
Measure first:

```python
import Plotting_Functions as pf
pf.use(subdir="hems", titles=False)
pf.PRESET, pf.PRESETS[pf.PRESET], pf.figsize()
# -> 'screen', 7.16, (7.16, 4.4034)
```

`IEEE_PAGE_W = 7.16` is `\textwidth` and `IEEE_COLUMN_W = 3.5` is `\columnwidth`. **Every
figure is already authored at full page width** — `Results/Figures/hems/<name>/SOURCE.md`
confirms e.g. `7.16 x 5.012 in @ 600 dpi`. So nothing in the notebook is column-width, and
widening the figures is not the fix.

The symptom the request describes is almost certainly LaTeX-side: a 7.16 in figure placed in
a single-column `\begin{figure}` float and included at `width=\columnwidth` is scaled to
49 %, which halves every font in it. `pf.latex_figure(name, caption, span=True)` emits the
`figure*` / `width=\textwidth` form that IEEEtran uses for a both-column float; `span=False`
is the default and emits the single-column one.

So: **confirm with the user how the figures are being included in the article** before
touching sizes. Then do whichever applies:

- If they are in single-column `figure` floats → the fix is `figure*` on the article side.
  Provide the `\begin{figure*}` blocks (`pf.latex_figure(..., span=True)` generates them) and
  note it in each `SOURCE.md`-adjacent guidance, rather than resizing anything.
- If they are already in `figure*` and still read small → the figures are at the right width
  and the fonts are too small for it. Raise the base size, not the width.
- If a specific figure genuinely wants to be *taller* at the same width, change its `ratio`
  in `pf.figsize(ratio=...)`. That is the knob; the width is not.

Whatever you conclude, write it down where the next person will find it.

# 4. The remaining regret figures in money → percent

The request was "convert the regret in graph 11 to %". Cell 11 is `controller_ranking_*`,
whose x axis is already `[% of the no-battery bill]` and carries no regret, so the numbering
is ambiguous — **ask the user which figure they meant if it matters.** Two regret axes are
still in currency and both should be converted regardless:

- **`forecast_channel_regret_lifetime`** (cell 12, the `view == "_lifetime"` pass). x is
  `[AUD on AU, EUR on SI, discounted at 5 %/a]`. The annual view of the same figure already
  uses `regret_pct_of_gain` and puts both tariffs on **one shared axis**, which is the whole
  point of it; the lifetime view should do the same. Note that a *share* of the gain is
  invariant under discounting if you divide a discounted regret by a discounted gain — both
  scale by `pv_factor` and it cancels. Cell 17 hit exactly this and solved it by changing the
  **denominator** with the view (annual → the year's bill, lifetime → the capital). Either do
  the same here (lifetime regret as a share of the capital is a real second statement), or
  drop the lifetime view and keep one figure. Do not ship two panels that are the same
  numbers under different axis labels — the first draft of cell 17 did, and it took a second
  look to catch.
- **`skill_vs_regret`** (cell 15). The y axis is `Regret [AUD/household-year]` /
  `[EUR/household-year]`, so the two panels cannot be compared and each needs its own scale.
  `long["regret_pct_of_gain"]` makes it one unit; the panels could then share a y axis and the
  figure would say "skill orders regret" *and* "SI has more regret to order" in one image.
  Recompute the Spearman on the converted values — it is rank-based, so ρ will not move within
  a panel, but **re-render and re-read the printed `n`, `p` and verdict** rather than assuming.

Relevant columns, all already in `long`: `regret_pct_of_gain`, `pv_factor`, `capex`,
`baseline_cost_total`, `lifetime_saving`.

# Done means

- Every figure re-rendered and **visually inspected** by you, not just executed.
- No `tight_layout` warning from any cell; no overprinted tick or axis label anywhere.
- The whole notebook re-executed with `nbclient` (about 95 s, everything cached) with **zero
  cell errors**, so the committed outputs match the committed code:
  ```python
  import nbformat; from nbclient import NotebookClient
  nb = nbformat.read("CODE.ipynb", as_version=4)
  NotebookClient(nb, timeout=3600, resources={"metadata": {"path": "."}}).execute()
  nbformat.write(nb, "CODE.ipynb")
  ```
- `python3 test_hems_study.py` still reports **67 passed, 0 failed**.
- `Results/Figures/hems/<name>/` still carries every existing figure name, plus one folder
  per new figure.
- `FIGURE_FIXES_PROMPT.md`'s status section updated, or a new one written, so the next pass
  starts from the truth.
