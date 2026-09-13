# Prompt: fix and extend the figures in `Yoann/CODE.ipynb`

Paste everything below the line into a fresh session, working in `/Users/summerscholl/Documents/ERK-2026`.

---

Work in `Yoann/`. The notebook is `Yoann/CODE.ipynb`; every figure goes through
`../Plotting_Functions.py` (imported as `pf`), and the study's data layer is `hems_study.py`
(imported as `hs`). A prior review already fixed the notebook's tables, prose and the one
statistical bug in a figure cell (`cell 12` now takes a paired median). **Do not touch
anything outside figure rendering** — the printed tables in cells 3, 5, 8, 13, 15, 17, 19 and
22 were just corrected and re-executed, and `hems_study.py`'s label dict, `arm_runtimes`,
`provenance`, `run_status` plumbing and `study_units` guard are all current.

Before changing a figure, render it and look at it. To iterate without a Jupyter UI:

```python
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(".."))
import matplotlib; matplotlib.use("Agg")
import json, io, contextlib
nb = json.load(open("CODE.ipynb"))
import Plotting_Functions as pf; pf.SAVE_PDF = False   # don't write into Results/ while iterating
g = {}
for i in (1, 3, 5, 7, 10):                              # setup cells
    src = "".join(nb["cells"][i]["source"])
    if i == 3: src = src.replace("RUN_SWEEP = True", "RUN_SWEEP = False")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(src, g)
exec("".join(nb["cells"][17]["source"]), g)             # the figure under test
import matplotlib.pyplot as plt; plt.gcf().savefig("/tmp/f.png", dpi=110)
```

Then `Read /tmp/f.png` and judge it with your eyes. A figure you have not looked at is not
fixed. When you are done, re-execute the whole notebook with `nbclient` (about 80 s, the
sweep is fully cached) so the stored outputs match the code, and verify no cell errors.

Read the `dataviz` skill before writing chart code.

## Status, 2026-09-13

A second review pass has since fixed part of this list and re-executed the notebook, so read
each heading's status line before starting. What is left is §3 (the wear/saving redesign),
§5 (`controller_ranking_*`), §7 (new figures), and the presentational half of §2, §4 and §6.

Also fixed outside the figures, in the same pass: three stale numbers in the caveats (cell 20)
— the Ausgrid 148 skills, the standing-charge example, the 0.44 correlation — `milp_full` added
to cell 0's controller table, and cells 13 and 15 now rescore when the roster or the tuning grid
has moved rather than only when the CSV is missing. Both cached CSVs were regenerated, so
`hs.provenance` reports a window instead of "no provenance" (the numbers came back identical;
only the provenance columns are new).

Note for the harness below: cell 14 reads `bench` from cell 13, so the setup list has to be
`(1, 3, 5, 7, 10, 13)` to render it.

## The house style you must stay inside

`pf.use(subdir="hems", titles=False)` is already set in cell 10. Colour means the **tariff**
(AU `SERIES[0]` blue, SI `SERIES[1]` orange); within a tariff, marker **shape** means the
controller family (`FAMILY_MARKER`) and shade means the same thing more weakly
(`FAMILY_SHADE`). `titles=False` lifts axes titles out of the image into the export's
`SOURCE.md` caption, so **panel identity can never live in a title** — cells 12 and 14 already
solve this correctly, one with an in-axes corner mark and one with the x label. Keep new
figures to `pf.figsize(...)` and finish them with `pf.show(fig, "<stable_name>")`; the name is
what the LaTeX side references, so do not rename an existing one.

---

## 1. DONE — cells 14 and 17: a legend anchored to a half-width axes collapses the panels

**Fixed.** Both cells now build the key with `pf.chart_frame(fig, title, handles=..., ncol=2)`
and close with `pf.show(..., layout="frame")`. `pf.key_legend` was left alone — it has other
callers and is right for an axes-level key. No `tight_layout` warning remains anywhere in the
notebook, both panels are full width on `lifetime_economics_au`, and nothing overprints. The x
labels had to shrink to fit half-width panels — `Lifetime NPV [AUD]` / `IRR [%/a]` in cell 17,
`Skill — the household` / `Skill — the roof` in cell 14 — with what each median is over moved
into the `chart_frame` title, i.e. into the export's caption. Kept for the record:

`pf.key_legend(axes[0], ..., where="below")` sets `bbox_to_anchor=(0.5, -0.16)` on an **axes**
legend, and `pf.show(..., layout="tight")` then runs `fig.tight_layout()`, which shrinks that
axes until its decorations fit. With a two-column legend whose labels are
`"pack + install (5,691 AUD, of which 1,626 is the fee)"` hung under a half-width panel, the
axes is squeezed to roughly 20 px.

Consequences, all currently in the committed outputs:

- **`lifetime_economics_au` is unusable.** Both panels are ~20 px wide, every marker sits on
  one vertical line, and the two x tick labels overprint into `−5000`.
- **`lifetime_economics_si`** is merely bad: its NPV ticks overprint into `−25000` (that is
  `−2500` and `0` on top of each other) and the IRR ticks into `−50`/`0`.
- **`forecast_method_skill`** (cell 14) has its legend *title* — the long
  `"skill = 1 − MAE / MAE_yesterday: …"` string — starting outside the figure's left edge,
  for the same reason: the legend is centred on `axes[0]`, and the title is wider than the
  legend box.

matplotlib emits `UserWarning: Tight layout not applied. tight_layout cannot make Axes width
small enough to accommodate all Axes decorations` on cell 17 — that warning is this bug.

**Fix:** put a shared legend on the **figure**, not on one axes. `pf.chart_frame(fig, title,
subtitle, handles=..., ncol=...)` exists for exactly this — it measures the header and legend
bands in inches and hands `tight_layout` a `rect` that excludes them — and it must be paired
with `pf.show(fig, name, layout="frame")`, otherwise `show` runs `tight_layout` again and
undoes the reserved bands. Check whether `pf.key_legend(..., where="below")` should grow a
figure-level mode instead; if you change `key_legend`, check its other callers in the repo
first (`grep -rn "key_legend" ..`). Either way, verify by reading the rendered PNG that both
panels are full width and no tick label overprints its neighbour.

## 2. PARTLY DONE — cell 17 — the IRR panel plots 1-household medians next to 30-household medians

**The sample size is now visible**, which was the correctness half: marker area is linear in the
households behind the median, and any row not over the full 30 carries `n=<pack only>/<with the
fee>` beside its rightmost mark. The `no IRR with the fee` note was dropped — `n=17/0` says the
same thing in the same units, and the two collided; `no IRR` survives for rows with no mark at
all. The caption states that NPV is over all 30 and the IRR panel is not.

**Still open:** the pack-only/with-fee gap is still invisible on the NPV panel, because the x
range is dominated by the distance to zero. The fee's own NPV delta as a small third panel is
still the suggestion. Original text:

This is a correctness problem expressed as a figure, and it is the one to get right.

`med = sub.groupby("controller")[[...]].median()` takes each column's median independently and
pandas drops NaN first. An IRR is NaN wherever the saving never covers the O&M charge, so on
`SI_H24` the IRR panel currently plots:

| controller | IRR plotted | households behind it |
|---|---|---|
| `milp_full` | −29.9 | 30 |
| `self_consumption_peak_shaving` | −34.6 | 3 |
| `oracle` | −38.5 | **1** |
| `peak_shaving` | −52.0 | **1** |

`peak_shaving` plots far left of `self_consumption_peak_shaving`, which reads as "much worse"
and is one outlier household against three. Pack-only is the same story: `price_oracle` −40.6
and `price_rank_daily` −44.6 are each one household. The `"no IRR"` note only fires when the
median itself is NaN, i.e. at n=0, so a marker backed by one household is visually
indistinguishable from one backed by thirty. And the NPV on the same row is always over all
30, so the two numbers on one row are medians over different samples.

`summarize` already computes `irr_defined` and `irr_defined_pack_only` per row, and the
summary table in cell 7 carries `IRR_n` with a long comment about precisely this trap; the
printed table at the bottom of cell 17 now carries `n` columns too. The figure is the last
place that still hides it.

**Fix:** make the sample size visible in the IRR panel. Options, in the order I'd try them:
annotate `n` beside each IRR marker; scale marker size or opacity by `n`; or draw markers
below some threshold (say n < 10) hollow-and-hairline with the count printed. Whatever you
choose, a reader must not be able to compare two IRR markers without seeing that one is a
median over one household. Also state in the y-axis or the caption that the NPV panel is over
all 30 while the IRR panel is not.

While you are in there: the two marks per row (hollow = pack only, solid = pack + install)
are nearly coincident because the x range is dominated by the distance to zero, so the
"gap = what the fee is worth" reading the cell's comment promises is invisible. Consider
plotting the **fee's** NPV delta as its own small panel, or breaking the x axis.

## 3. Cell 16 — `wear_vs_saving_*` is the wrong chart for this data

Do not try to fix this with better label placement. The `place_labels` helper in cell 10 is
~150 lines of ring search, leader lines, wrapping and a scored fallback, and it is fighting
geometry it cannot win. The medians it has to label:

```
AU_H24   delayed_pv_charge (63.47, 108.65)   self_consumption (63.98, 108.67)   <- ~1 px apart
         price_oracle (222.24, 267.62)  price_threshold (223.08, 265.99)  price_rank_daily (225.25, 270.92)
         milp_full (127.16, 245.92)     oracle (129.81, 244.54)
SI_H24   delayed_pv_charge (63.47, 35.55)    self_consumption (63.98, 36.37)
         price_oracle (82.90, 35.05)  price_rank_daily (83.25, 34.09)  price_threshold (84.53, 33.46)  fixed_schedule (83.41, 20.58)
```

Ten to thirteen points, several pairs a pixel apart, with names up to 54 characters, has no
legible layout. In the committed AU figure two markers coincide and one of their two labels
appears detached with no leader; in the SI figure the leader lines cross each other.

**Fix:** replace the scatter with a **sorted dumbbell**, one row per controller, name on the y
axis where it always fits. Per row draw `saving` and `wear_eur` as two marks joined by a bar,
x in the arm's currency. Break-even becomes "does the bar cross zero", which is the reading the
figure exists for, and it disposes of the sloped wear lines and their captions entirely — the
current cell spends ~40 lines anchoring two line captions onto the visible segment of a line
whose y axis floor is above zero.

Keep the wear-rate sensitivity that cell 16's comment block argues for: on AU the sweep
charged 0.4167 **AUD**/EFC (= 250 AUD/kWh) while the honest quote is 250 EUR/kWh
(= 0.678 AUD/EFC, 63 % steeper). Carry it as a second, fainter wear mark per row, or as a
supplementary panel — not as a dropped caveat. Read `_slope` from the frame
(`sub["wear_eur"] / sub["efc"]`) exactly as the current cell does, never a retyped constant.

Once `place_labels` has no caller, delete it; if you keep one caller, leave it.

## 4. PARTLY DONE — cell 12 — `forecast_channel_regret`

**Fixed:** the limits. An all-NaN arm no longer raises, a negative regret can no longer be
clipped out of the panel, a zero rule is drawn if one appears, and value labels flip to the
outside of the bar end by sign.

**Still open:** sorting the bars by regret, and the reference rule at `both: Prophet`.

The arithmetic is already correct (paired median) and the study's own arm is now correctly
labelled `both: Prophet` rather than `PV: Prophet`. Remaining, all presentational:

- **Sort the bars by regret.** They are in `FORECAST_KIND_LABELS` roster order; nobody reads
  this figure for roster order. Sort within each panel, or sort both by the AU ordering and
  say so.
- **Mark the study's own forecaster.** Add a light reference rule at the `both: Prophet`
  value so every other method reads as better or worse than the arm the paper is about.
- **Guard the limits.** `max(v for v in vals if v == v)` raises `ValueError` on an all-NaN
  arm, and `set_xlim(0, ...)` silently hides a negative regret and puts its value label
  off-axes. Every value is positive today; neither is guaranteed.

## 5. Cell 11 — `controller_ranking_*` (one item done)

The `edgecolor` warning is fixed: the halo is only passed for markers that have a face. The rest
of this section stands.

- The y tick labels eat about a third of the width, `"RBC: price threshold, full-year
  foresight (diagnostic)"` worst. Wrap to two lines, or move the family prefix (`RBC:`,
  `MPC-MILP`) into a gutter and shorten the names.
- `FAMILY_SHADE` (0.75/0.42/0.12/0.0 mixed toward white) is near-invisible at `alpha=0.6`
  — shape is carrying the whole family encoding. Either commit to shape-only and drop the
  shade, or widen the shade range enough to see.
- `UserWarning: You passed an edgecolor/edgecolors ('#fcfcfb') for an unfilled marker ('x')`
  fires on every call — `no_battery` uses `'x'`. Pass `edgecolor` only for filled markers.
- Add a small key for the marker shapes; there is currently none, so the RBC/MPC/MILP
  distinction the figure is built around is undocumented inside the image.

## 6. PARTLY DONE — cell 14 — `forecast_method_skill`

**Fixed:** the legend (§1). The skill definition went into the `chart_frame` title rather than a
`fig.supxlabel`, because `chart_frame` reserves a legend band and nothing else — a supxlabel
lands inside that band. If it has to be in the image rather than in the caption, widen the band
by hand.

**Still open:** the `yesterday` row is still a zero-length bar labelled `+0.00`.

Besides the legend blocker in §1:

- The `yesterday` row is a zero-length bar labelled `+0.00`, which reads as a missing result.
  Drop it, or mark it explicitly as the baseline.
- Move the skill definition out of the legend title into a shared `fig.supxlabel`. Cell 12
  already does exactly this for its two-currency label and is the pattern to copy.

## 7. New figures worth adding

In priority order. The first one is the figure the notebook's title question needs and does
not have.

1. **Skill vs. regret scatter.** Forecast skill (cell 13's `bench`) on x, paired median regret
   (cell 12's computation) on y, one point per forecaster kind, faceted by tariff. Six kinds
   are in both tables — `persistence`, `median14`, `prophet`, `hbd`, `hbd_baseline`,
   `hbd_median14` — which is enough. Right now a reader has to do this join by eye across two
   pages, and the two lists only partly overlap (cell 13 scores 11 kinds, cell 12 plots 14
   arms). Adding `prophet_tuned` to `hs.BENCHMARK_KINDS` would give a seventh point and also
   fix a real gap: the tuned Prophet currently has no full-roster error number at all, only
   the 8-household/90-day screen in cell 15. That is a `hems_study.py` change and a benchmark
   rescore — clear it with the user first, since it invalidates
   `results_local/forecast_benchmark.csv`.
2. **Replacement for cell 16** per §3 — it doubles as the "what a saving costs in pack life"
   figure.
3. **Break-even capex.** `be.battery_economics` already returns
   `Break_Even_Capex_EUR_kWh` and nothing plots it. One bar per controller against the assumed
   250 EUR/kWh answers "how far does the pack price have to fall" far more usefully than a
   column of negative IRRs, and it has no n=1 median problem because it is defined for every
   household. You will need to surface it through `hs.summarize` — check whether it is already
   in `long` before adding it.
4. **Household heterogeneity.** Cell 20's caveats note that Ausgrid 127 and 148 disagree about
   which channel Prophet is good at, and `hs.study_units()` carries `cluster` and
   `dist_to_centroid` specifically so "which kind of household is this controller bad at" is
   answerable. Nothing plots it. A per-household strip of prophet regret ordered by cluster
   uses data already paid for.
5. **Horizon panel.** H24 vs H11 is one of the study's three axes and has no figure at all; it
   appears only as two rows in cell 19's arm table.

## Done means (for what is left)

- Every figure re-rendered and **visually inspected** by you, not just executed.
- No `tight_layout` warning from any cell; no overprinted tick label or axis label anywhere.
- The whole notebook re-executed through `nbclient` with zero cell errors, so the committed
  outputs match the committed code.
- `python3 test_hems_study.py` still reports `42 passed, 0 failed` (it does not test figures,
  but it catches an accidental edit to the data layer).
- The `Results/Figures/hems/<name>/` export folders still carry the same figure names as
  before, plus one folder per new figure.
