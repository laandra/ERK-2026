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

## Status, 2026-09-15 (pass 4) — READ THIS ONE FIRST

Pass 4 was not a geometry pass. Pass 3 left the figures clean to the eye and to a
checker; what was still wrong was **statistical**: three figures printed a number the
picture did not show, and one of the three was costing the paper a result it has. It
also added the two comparisons the study can make and had no figure for.

**26 exports now** (21 + 5 new). No existing name was renamed or dropped.
`python3 test_hems_study.py` is **67 passed, 0 failed**; `hems_study.py`,
`Plotting_Functions.py` and `test_hems_study.py` were not touched. `CODE.ipynb`
re-executes in **99 s with zero cell errors and no warning** but the pre-existing
`ipywidgets` one from cell 1.

### New files, and why the style moved out of cell 10

- **`figure_style.py`** — the house style, minus anything that closes over the sweep.
  It moved because a SECOND notebook now needs the same semantics, and a palette
  defined in two cells is a second source of truth. Cell 10 imports from it and keeps
  only `make_labels`/`LABEL`, `SAMPLE`, `nowear_twin`/`with_nowear` and a one-line
  `tick_label` that binds this notebook's `LABEL` into the shared shortener (which now
  takes the label map as an argument — the horizon is part of an MPC controller's
  identity, so the map is per-arm and cannot be a module constant). `FAMILY_LABEL` and
  `beeswarm` came out of cell 11, `BEATS`/`LOSES` out of cells 14, 17 and 23.
  New in it: `boot_ci`, `boot_ci_paired`, `spearman_ci`, `rho_lines`, `corner_block`,
  `holm`, and `PCT_OF_GAIN_SHORT`. **Do not add a fourth wording for a denominator.**
- **`anomaly.py`** — public holidays off the run's own calendar, absence detection, and
  the cached daily panel (`results_local/daily_panel.csv`, provenance columns and a
  staleness check like `forecast_benchmark.csv`). ~15 s warm for 30 households x 5
  methods; **hours against a cold forecast cache**, so run the sweep first.
- **`FIGURES_FORECAST.ipynb`** — the new figures, plus the defect checker, committed
  this time (see below).

### The three reworked figures — same names, ported back into CODE.ipynb

- **`skill_vs_regret` (cell 15). This one changed a conclusion.** It correlated seven
  method medians against seven and threw away the pairing the whole of section 5 rests
  on. On seven points AU is ρ = −0.64 at **p = 0.119**, and the old caption said AU
  "does not carry the claim". It does: within each household, across the seven methods,
  the correlation is negative on **28/30** households on AU (sign test p = 9e-7) and
  **30/30** on SI (p = 2e-9). Both readings are printed, the between-method one now
  with a bootstrap CI that says how little seven points support. The caveats cell
  carries the correction.
- **`household_heterogeneity` (cell 22).** Spearman drawn on the rank axis it measures,
  not on a raw axis with two thirds of the households under 30 and one at 96. Same ρ
  and p; a Theil–Sen line fitted **in rank space** (the one line a rank statistic
  licenses — the old cell was right to refuse an OLS line in data space, and the comment
  now says why this one is different) and a bootstrap CI on ρ.
- **`forecast_channel_regret` (cell 12, both views).** Percentile bootstrap CI on every
  paired median, and every row tested against its panel's best, Holm-adjusted over the
  thirteen comparisons. All thirteen separate on both tariffs — so the hollow "not
  separable" key never fires on this sweep and is only added when it does, because a
  legend entry with no instance is decoration a reader then hunts for.

### Five new figures

`top_methods_headtohead`, `top_methods_channels`, `holiday_effect`,
`absence_forecast_breakdown`, `absence_map`. The two results worth knowing:

- **The best forecast is not the same one on both tariffs.** `hbd_median14` wins AU by
  0.64 pp of the achievable gain (p = 0.0006, 22/30) and loses SI to plain `median14`
  by 1.90 pp (p = 0.0003, 23/30). Hand both a perfect PV forecast and the swap survives
  and sharpens (2/30 against 28/30), so it is a **consumption-channel** effect.
- **Public holidays do nothing; absences do everything.** The holiday effect is +0.017
  against the households' own workdays at p = 0.18 — the wrong sign — while the same
  statistic finds the weekend at p = 1e-4, which is the positive control that makes the
  negative result mean something. What breaks the forecasts is 37 multi-day absences
  across 22 of the 30 houses: copying yesterday's error falls to **x0.22**, Prophet's
  rises to **x1.53**, every model-based method over-predicts (22/22 households for the
  median methods), and Prophet throws away **34 %** of the day's achievable gain against
  9 % normally. Raw daily cash goes the *other* way — an empty house has less to win —
  which is why the money panel divides by the day's own gain and says so.

### The checker is committed now

It is the last cell of `FIGURES_FORECAST.ipynb`. Two traps cost real time rebuilding it,
both worth knowing before a pass 5 touches it:

- **Ghost ticks.** A locator routinely emits ticks beyond the view limits. They are never
  drawn, but their `Text` artists have positions, and on a right-hand panel they sit past
  the canvas edge — five false "clipped label" reports. Zipping `get_ticklocs()` against
  `get_ticklabels()` does not identify them; ask the `Tick` object for its own `get_loc()`
  and its own `label1`/`label2`. And they must be excluded from the **figure-level** sweep
  as well as the per-axes one, or the figure-level pass picks straight back up everything
  the axes pass just dropped.
- **A trimmed PNG hides the whole clipped class.** `savefig` with `bbox_inches` unset
  resolves to `rcParams["savefig.bbox"] == "tight"`, which grows the canvas to fit
  overhanging text; the exported PDF uses `tight=False` for `layout="frame"` and cuts it.
  Set the rcParam to `None` while iterating.

The residue it reports is one accepted class: a numbered badge on `skill_vs_regret`,
drawn on a surface pad, covers one to five points of a 210-point cloud. There is no
placement in that panel that covers none.

### What is left

- Cell 18 still uses the long `LABEL.get(c, c)` names — unchanged from pass 3, and the
  reason is unchanged: it is the one figure whose key does not carry the family markers.
- Cell 20's value labels still sit close to their markers on short bars.
- The daily money panel is **AU only**, and `anomaly.daily_panel` raises rather than
  guessing: SI's excess-power charge is monthly and endogenous to the controller's own
  peaks, so a day is not a settleable unit there. A per-month SI decomposition is the
  obvious next thing if the absence result needs a second tariff.
- The channel decomposition is **one-sided**. The sweep has a perfect-PV twin for each
  top method and no perfect-LOAD twin, so `top_methods_channels`'s lower segment is "what
  survives a perfect roof" — the load channel plus any interaction — and is labelled as
  that rather than as a measured load-only term. Four `*_loadtruth` arms would close it.

## Status, 2026-09-14 (pass 3) — this list is CLOSED

Kept for the reasoning, not the state. Pass 4 above supersedes it.

Pass 3 closed §7.4, converted the last two money regret axes to shares, ran a measured
readability pass over every figure, and settled the figure-width question. **21 exports now**
(19 + 2 new); no existing name was renamed or dropped. `python3 test_hems_study.py` is
**67 passed, 0 failed**; `hems_study.py`, `Plotting_Functions.py` and `test_hems_study.py`
were not touched — everything below is notebook-side. The whole notebook re-executes in 96 s
with zero cell errors and no `tight_layout` warning.

### §7.4 is done, and it has a real answer

Two new figures, after `horizon_effect`:

- **`household_heterogeneity`** — one point per household, faceted by tariff. x is the
  household's own export over its own import with no battery (`sell_no_battery /
  buy_no_battery`), y is Prophet's `regret_pct_of_gain`. **The sign flips between the
  tariffs**: ρ = **+0.63** (p = 0.0002) on AU and **−0.55** (p = 0.0017) on SI, n = 30 each.
  PV-heavy households are the ones Prophet is worst at on AU and best at on SI. Consistent
  with that, the two tariffs barely agree about which households are hard (ρ = −0.18,
  p = 0.35).
- **`household_regret_strip`** — 30 rows, one per household, sorted by AU regret. Left panel:
  AU and SI regret on one shared axis (legal only because the measure is a share). Right
  panel: both channel skills as bars from zero, coloured by sign like cell 14. Ausgrid 127
  and 148 are bold, so the channel disagreement the caveats describe finally has a picture.

**Do not re-propose `cluster` or `dist_to_centroid` as an axis.** Both were tested:
|ρ| < 0.15, p > 0.4 on both tariffs. The cluster id is 1:1 with the household, exactly as the
pass-3 prompt warned. The honest reading — stated in both captions — is that the households
do not fall into kinds, they fall on a gradient, and forecast skill does not order the regret
either (ρ = −0.33, p = 0.08 against the mean of the two channels).

The caveats cell now carries both findings and points at the two figures by name.

### The last two money axes are shares

- **`forecast_channel_regret_lifetime`** — was `[AUD on AU, EUR on SI]` on two independent
  scales. Now `[% of the capital]` on **one shared axis**. The denominator had to change,
  not just the units: a share of the *gain* is invariant under discounting — `pv_factor`
  multiplies regret and gain alike and cancels — so a lifetime share of the gain is the
  annual panel again under a longer label, which is the trap cell 17 fell into once already.
  `_regret` now reads `pv_factor` and `capex` per household out of `long` rather than using
  the scalar `PVF`.
- **`skill_vs_regret`** — this was the user's "graph 11" (the cell beginning *"Does a better
  forecast buy a smaller regret?"*). y is now `regret_pct_of_gain`, the **same column** cell
  12's annual view plots — verified equal to 10 decimal places on both reference arms. The
  ρ and p **did move**, because dividing by a per-household gain before the median is not a
  rank-preserving transform across kinds: AU went −0.68/p=0.094 → **−0.64/p=0.119**, SI
  −0.93/p=0.003 → **−0.96/p=0.000**. The verdicts are unchanged (AU does not carry the
  claim; SI does).
- It also **gained a caption**. It was the only figure closing on a bare `pf.show`, so with
  `titles=False` its `SOURCE.md` recorded no caption at all and the article had nothing to
  put in `\caption{}`. It now goes through `chart_frame` + `layout="frame"` like the rest.

### Figure width: the premise was false, and the fix is LaTeX-side

`pf.PRESET` is `screen` = `IEEE_PAGE_W` = 7.16 in = `\textwidth`. All 21 `SOURCE.md` files
say `7.16 x ... in`. **Nothing in this notebook is column-width and widening is not
available** — 7.16 is already the maximum. What makes the figures read small is a 7.16 in
graphic included at `width=\columnwidth`, which scales it to 49 % and halves every font.

New cell at the end of the notebook emits `pf.latex_figure(name, caption, subdir="hems",
span=True)` for all 21, reading each figure's recorded caption out of its `source.json` so
the caption and the image cannot drift. Note `latex_figure` does **not** read the module
`SUBDIR` global — `subdir="hems"` must be passed or the include path is wrong.

The knobs, in order: the **float** (`figure*`), then **`ratio`** in `pf.figsize(ratio=...)`
if a figure wants to be taller at the same width, then the **base font size** in
`Plotting_Functions.py` — and only if the figures are already in `figure*` and still read
small. The width is not a knob.

### The readability pass was measured, not eyeballed

Every figure was rendered through the real `pf.show` pipeline and checked mechanically for
text-on-text overlap and text straddling the canvas edge, then looked at. **17 of 19 figures
had at least one defect; all 21 are now clean.** Two classes were invisible to a casual read:

- **Legend rows and axis labels ran off the exported box.** `chart_frame`'s legend is
  centred on the figure and a three-column row of long labels was wider than 7.16 in, so
  `rule-based (RBC)` started 44 px left of the canvas and the star key ended 8 px past its
  right edge. Separately, `finish(ax, xlabel=...)` centres on the **axes**, which thirteen
  rows of tick labels push well right of the figure's centre — the ranking figure's lifetime
  x label overhung by 65 px. Fixed by shortening: `view["per"]` for the lifetime view, the
  star key to `* same MILP, no degradation term` (one wording in all five cells that use
  it), and the ranking figure's lifetime label now **names** its denominator
  (`PCT_OF_LIFETIME_BILL`, new in cell 10) instead of qualifying it — `lifetime_saving_pct`
  divides by `baseline_cost_total * pv_factor`, so the old `PCT_OF_BILL` was also imprecise.
- **The exported PDF is not trimmed, so overhanging text is genuinely cut.** `save_fig` uses
  `tight=False` for `layout="frame"` figures. Watch for this when iterating: passing
  `bbox_inches=None` to `savefig` resolves back through `_val_or_rc` to
  `rcParams["savefig.bbox"] == "tight"`, so a PNG saved that way is trimmed while the PDF is
  not, and the trimmed PNG hides every defect of this class. Only the rcParam turns it off.

Also fixed:

- **`tick_label` (cell 10)** — width 30 → 34 plus one short override, so all sixteen rows are
  now a single line on cells 11, 17 and 19. Only two labels ever wrapped, and one of them
  (`self-consumption + peak shaving`, 31 chars) was one character over.
- **The long left tail on the ranking panels** — new `clip_window` helper in cell 10 (shared,
  since the strip figure uses it too). The axis is clipped to a robust range widened to hold
  every median; households outside it are drawn **on** the boundary as carets and counted in
  the key. The AU lifetime panel went from −42..+27 to −13..+27. Nothing is dropped and the
  medians are still taken over all 30.
- **Cell 18's `n=30/29 unpaired` tags** — 6.5 pt against the markers, now 7.5 pt pinned to
  the axes edge as a right gutter column, the pattern cell 19 already used.
- **`skill_vs_regret`'s label ladder** — the collisions were **horizontal**, so no ladder of
  vertical offsets could fix them; a sweep of the whole offset grid moved them around
  without removing one. Seven names averaging ~90 px cannot fit a ~330 px half-width panel.
  The panels are now **stacked**, full width, sharing both axes — which also makes the new
  finding visible, that the SI regret band sits entirely above the AU band.

### What is left

- **Cell 18 is the only figure still using the long `LABEL.get(c, c)` names** rather than
  `tick_label`, and `RBC: price threshold, full-year foresight (diagnostic)` eats about 38 %
  of its width. It was left alone deliberately: it is the one figure whose key does **not**
  carry the family markers, so the `RBC:`/`MPC-MILP` prefix is the only place the family is
  named. Shortening the labels there means adding a family key first.
- **Cell 20's value labels** sit close to their markers on short bars (`−0.1` on the SI
  Prophet row). Below the overlap threshold, but it is the next thing to tighten.
- The overlap/clipping checker used for this pass is not committed. If a pass 4 touches
  geometry, rebuild it: walk `fig.findobj(Text)` after `fig.canvas.draw()`, drop tick labels
  whose tick is outside the axis view limits (they are ghosts and produce false reports on
  both checks), and test pairwise bbox intersection plus straddling of the renderer's box.
  A text-only check still misses a label sitting on somebody else's **marker** — check label
  boxes against `collection.get_offsets()` too, which is how the caveat-household labels in
  `household_heterogeneity` were caught and turned into rings.

## Status, 2026-09-14 (pass 2) — this list is CLOSED

Everything in sections 1-7 below is done, plus a round of user-requested work that went
past it. **Read this section, not the per-item status lines below, which are stale.**
The sections are kept for the reasoning, not the state.

### Done in this pass

**Data layer (`hems_study.py`).** Two rates where there was one: `cycle_cost_eur_per_efc`
is the MILP's shadow price and `cycle_cost_reporting_eur_per_efc` is what the cycles cost.
`summarize` and `full_period_bound_check` now bill wear at the second and only ever read a
*positive* dispatch rate back — a solved rate of zero is the one value that cannot mean "a
different pack price". Without this the no-degradation arm reports zero wear and the
hardest-cycling controller in the study tops every chart whose axis says "net of wear".
Both places had the bug; the second (`total_*`, and therefore `saving_total`) was found
only by rendering the figure and disbelieving it.

New columns, all with tests: `roi_pct`, `break_even_capex`, `lifetime_saving`,
`lifetime_saving_undisc`, `pv_factor`, `lifetime_wear`, `saving_total_pct`,
`lifetime_saving_pct`, `wear_pct`, `regret_pct_of_gain`, `life_binds_on_cycles`,
`wear_rate_charged` / `_accounted` / `wear_priced_in_objective`. `roi_pct` and
`break_even_capex` were computed by `Battery_Economics` all along and thrown away.

**Four new arms**: `{AU,SI}_{H24,H11}_nowear`, the same arms with
`cycle_cost_eur_per_efc = 0`. They are a controlled ablation — everything else is the
twin's — and the rules in them are expected to be bit-identical, which is the control.
`forecast_arms` excludes them: they carry no `forecaster_kind` either, and would otherwise
answer "which arm shows me Prophet?" by roster order, the same trap `AU_H24_leaked` set.

**`test_hems_study.py` is 67 passed, 0 failed** (was 42).

### The figures

Nineteen exports, every one rendered and looked at. Money figures come in an `annual` and a
`_lifetime` version driven by one `VIEWS` loop; the annual view keeps every pre-existing
export name so the LaTeX side does not move.

- `controller_ranking_{au,si}[_lifetime]` — **now drawn on `saving_total_pct`, not
  `saving_pct`.** That was the one real correctness bug in the set: on the energy bill
  alone, three AU price rules score above the row labelled "optimum" (106.6 % of a supposed
  ceiling) because they buy their saving out of pack life and the measure cannot see it.
  Also sorted, beeswarmed instead of jittered, and the AU/SI roster difference is stated in
  the caption.
- `forecast_channel_regret[_lifetime]` — **`regret_pct_of_gain`, so both tariffs share one
  axis.** Sorted by regret on the AU ordering (panels stay aligned, so a rank disagreement
  stays visible), with a reference rule at the study's own forecaster.
- `forecast_method_skill` — `ratio=0.62` (0.42 was ~11 px a row and would not have
  printed), `yesterday` marked as the baseline instead of a zero bar labelled `+0.00`, and
  sign is green/**red**: orange is the SI tariff everywhere else in this notebook.
- `skill_vs_regret` — prints `n` and `p`. AU is rho -0.61 at **p = 0.15** and does not carry
  the claim; SI does. Printed side by side without them the two read as one finding.
- `wear_vs_saving_{au,si}[_lifetime]` — both axes are shares. The lifetime view divides by
  the **capital**, not the lifetime bill: a discounted saving over a discounted bill cancels
  `pv_factor` out of both and reproduces the annual panel under a new axis label, which the
  first draft duly shipped.
- `lifetime_economics_{au,si}[_money]` — the percentage version is now the default
  (`roi_pct`), money is the variant. The hollow-to-solid connector is drawn **only where the
  same households are behind both ends**; on SI it joined a median over twenty to a median
  over one and asserted the install fee costs 19.2 IRR points. Marker area is documented in
  the key, and both `n` are always printed.
- `break_even_capex_{au,si}` — NEW. Defined for every household, so it has none of the IRR
  panel's n=1 problem. As a share of today's quote: the pack must reach 37 % of it on AU and
  21 % on SI, pack-only — and with the install fee in, every value is negative, i.e. no cell
  price pays, free included.
- `horizon_effect` — NEW, and the H24/H11 axis had no figure at all. Paired per household.
  Perfect foresight gains from the longer horizon; a Prophet forecast does not, and with no
  degradation term the longer horizon actively hurts it.

### What is left

Only §7.4, the household-heterogeneity strip. `study_units()` carries `cluster` and
`dist_to_centroid` for it and nothing plots them.

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

## 3. DONE — cell 16 (now 17) — `wear_vs_saving_*` is the wrong chart for this data

**Replaced by a sorted dumbbell**, one row per controller, name on the y axis. Per row: the
median household's annual saving as the family marker, its annual wear as a tick, a bar between
them coloured by sign (green = the saving covers the wear), and a right-aligned column printing
`saving_net_of_wear` — the paired median the tables quote, which differs from the gap between
the marks by a few units of currency because it is taken per household first — with the row's
EFC beside it. The wear-rate sensitivity survived as a second, pale tick at the 250 EUR/kWh
quote, AU only, drawn from `_slope` read out of the frame. `place_labels` had no callers left
and was deleted; `tick_label` in the styling cell replaced it and is shared with the ranking
figure. Original text:

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

## 5. DONE — cell 11 — `controller_ranking_*`

All four items. The `edgecolor` warning is gone (the halo is only passed for markers with a
face); the family prefix is dropped from the tick labels and the rest wrapped at 30 characters
by `tick_label`, with the figure grown to `ratio=0.78` so thirteen two-line labels have room;
`FAMILY_SHADE` was widened to 0.72/0.46/0.22/0.0, where the four steps are distinguishable at
`alpha=0.6`; and a family key — shape *and* shade, on the figure via `chart_frame` — now
documents the RBC/MPC/MILP encoding inside the image. Original text:

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

1. **DONE — Skill vs. regret scatter.** Built as a new cell after the skill figure, exported as
   `skill_vs_regret`. x is the mean of the two channels' skill per household, then the median;
   y is the paired median regret, unchanged from the regret cell; seven kinds are in both
   tables now that `prophet_tuned` is in `BENCHMARK_KINDS`. Labels alternate above and below
   in x order, which is what keeps the two near-coincident pairs apart. It carries a Spearman
   ρ per panel, and the answer is that skill does order regret: ρ = −0.68 on AU and −0.93 on
   SI. Adding `prophet_tuned` to the benchmark also closed the gap this item flagged — it is
   scored on the full roster now, and it does **not** reproduce what the 8-household screen
   found. Original text: Forecast skill (cell 13's `bench`) on x, paired median regret
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
