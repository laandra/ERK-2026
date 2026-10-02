# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

The **ERK 2026 HEMS study**: *what does a forecast buy a home battery, and on which
tariff?* One continuous MILP drives a 10 kWh / 1.5 kW battery over 30 Ausgrid
households and a full simulated year, measured against the rule-based controllers a
household could actually install, on two price signals:

| arm | tariff |
|---|---|
| **AU** | Ausgrid EA025 time-of-use, working-day aware, spot-linked. Energy + standing charge, **no capacity charge**. |
| **SI** | GEN-I Dinamični under `si_samooskrba`. Energy, standing charge **and an excess-power charge** against a *dogovorjena obračunska moč* that is re-agreed monthly from the peaks the controller itself realised. |

The SI contract is therefore **endogenous**: a controller that shaves its peaks is
then held to the lower contract it earned. The MILP decides it inside the LP
(`MILP_Household.add_endogenous_agreed_power`); rules iterate to the same fixed
point through `Environment.converge_agreed_power`.

Python 3.14, no virtualenv checked in. Dependencies pinned in `requirements.txt`
(pulp/CBC + highspy for the LP, cvxpy for the ported forecaster, Prophet,
gymnasium, matplotlib).

## FIGURES — THE RULE THAT OVERRIDES CONVENIENCE

**Every new graph goes through the unified plotting function.** No exceptions.

- Terminate every figure cell with `pf.show(fig, "export_name")` — never
  `plt.tight_layout()` / `plt.show()`.
- Never set `rcParams` outside `Plotting_Functions.py`. Never call
  `sns.set_theme()` / `sns.set_style()` — they overwrite `pf.use()`'s style and
  silently change every figure in the study.
- Never write into `Results/Figures/` from anywhere but `Plotting_Functions`.
- Never put a `figsize` literal in a cell; geometry comes from the preset.

**If `Plotting_Functions.py` cannot express the figure you need, EXTEND
`Plotting_Functions.py`** — add the helper, the formatter, the palette entry or the
layout there and use it. Do **not** write a one-off helper in a notebook, and do not
copy a style block into a cell. The whole point of the module is that style, size,
language and export are properties of the *article*, not of any one chart; a local
helper is a second source of truth that drifts.

A new shared helper also belongs in `PLOTTING_GUIDE.txt` (section 8) so the next
figure finds it.

What legitimately stays in a notebook: colour *semantics* (`TARIFF_COLOR`,
`FAMILY_COLOR`, …) and anything closing over run data. What a colour **means** is
the study's; the palette itself (`SERIES`) is shared. Cross-notebook semantics live
in `Main/figure_style.py`.

Read `PLOTTING_GUIDE.txt` before touching figures — it is the full reference.
Essentials:

```python
# top of notebook, once
import importlib
import Plotting_Functions as pf
pf = importlib.reload(pf)                       # reload BEFORE use(), never after
pf.use(subdir="hems", titles=False)
from Plotting_Functions import INK, INK_2, MUTED, SURFACE, SERIES, EUR, finish

# bottom of every figure cell, once
pf.show(fig, "cost_vs_capacity")
```

- `pf.use(subdir=, titles=, lang=, preset=, paper=, ieee=, grid=)` — per-notebook
  settings. `titles=False` for the article (the LaTeX caption is the title; the text
  is still written to `SOURCE.md` and `source.json`).
- Presets: `screen` (7.16 in, default), `page` (7.16 in + IEEE text), `column`
  (3.50 in + IEEE text), `as_built`. **Preview equals export** — the preset is
  applied before display, so the inline picture is the one in the paper.
- **Two panels side by side are wrong** for a two-column article. Split into two
  figures in the same cell, one `pf.show` each.
- Labels are written in Slovenian; `lang="en"` translates via `GLOSSARY` /
  `GLOSSARY_PARTS` in preview and PDF alike. Untranslated strings collect in
  `pf.MISSING_TRANSLATIONS`. For new code with both strings: `pf.t("Poraba", "Consumption")`.
- Helpers: `finish` (keyword-only), `chart_frame` (+ `layout="frame"`),
  `panel_grid`, `key_legend`, `edge_label`, `mix`, `ramp`, `figsize`.
  Formatters: `EUR, PLAIN, KW, PCT, PCT_SIGNED, YEARS` (`PCT` and `PCT_SIGNED` are
  deliberately distinct — do not unify).
- An export writes `Results/Figures/<subdir>/<name>/` containing the PDF (always,
  vector, TrueType-embedded), PNG (`png=True`, 600 dpi), `source.json`, `SOURCE.md`,
  `cell.py` and `uncommitted.diff` when the tree was dirty. `Results/Figures/` is
  gitignored — export for the paper from a clean tree.
- `pf.show` returns `None` on purpose. Two copies of a figure = something returned a
  `Figure` as the cell's value.
- `plotMultiY` at the bottom of the module is legacy; do not use it for new work.

## Layout

```
Basic_Functions.py       battery physics helpers shared by env, MILP and rules
Environment.py           HouseholdEnvironment (gymnasium) + CommunityEnvironment;
                         agreed-power machinery (converge_agreed_power, ...)
MILP_Household.py        the LP: add_household_physics, add_excess_power_ratchet,
                         add_endogenous_agreed_power, solve_household;
                         build_household_env is THE env factory; battery/solver
                         constants (STEPS_PER_DAY, C_RATE, ...) live here
Rule_Based_Control.py    eight deployable controllers + run_policy, scored through
                         the same settlement the MILP is
Battery_Economics.py     CAPEX/OPEX/annuity; what a pack costs to own
Pricing_Functions.py     import shim re-exporting "New pricing functions/" under a
                         package-safe name (the folder has spaces)
Data_Loader.py           load_household_data / load_multiple_households / load_smp_data
Plotting_Functions.py    the ONE plotting module (see above)
PLOTTING_GUIDE.txt       its reference manual

New pricing functions/   the Slovenian tariff model
  si_tarife.py           regulated constants, sourced and dated
  si_cas.py              calendar, seasons, time blocks, holidays
  si_paketi.py           supplier price-list catalogue + compatibility rules
  si_moc.py              dogovorjena obračunska moč per block
  si_konica.py           ratchet peak tracking (MILP/RL-friendly)
  si_obracun.py          the monthly settlement math
  si_invoice.py          the ONLY invoice generator (InvoiceBuilder)
  si_poraba_doma.py      bills for real meter exports in Input data/Poraba doma/
  Pricing_Functions.py   calculate_interval_price — the dispatcher
  test_primer.py, test_souporaba.py

Main/                    the study
  hems_study.py          EVERYTHING the study computes (~5.6k lines)
  hbd_forecast.py        Fourier baseline + residual-AR forecaster, ported from
                         cvxgrp/home-battery-dispatch; departures marked DEPARTURE
  pv_split.py            partitions STUDY_ARMS into three notebook tracks
  anomaly.py             holidays and absences; the daily panel (computes, never draws)
  figure_style.py        cross-notebook palette/marker semantics
  tune_rules.py          re-tune price-reading rules per tariff
  tune_fixed_schedule.py where FIXED_SCHEDULE_WINDOWS came from
  run_hbd_benchmark.py   resumable screen of the ported forecaster
  test_*.py              invariant checks (see Testing)
  CODE.ipynb             the notebook of record (all arms)
  CODE_PV_FORECAST.ipynb deployable arms: one method, both channels
  CODE_PV_PERFECT.ipynb  perfect roof on top of a real load model
  CODE_PV_VALUE.ipynb    the difference between the two — a measurement, not a ranking
  FIGURES_FORECAST.ipynb forecast-method figures + the anomaly story

Clustering/              cluster_households.py + committed Ausgrid cluster CSVs
Input data/              Ausgrid (300), Fluvius variants, GreekSmartHome, SMP (per
                         country), Poraba doma (real meter exports)
Results/Figures/         figure exports (gitignored)
```

## Architecture invariants

These are load-bearing. Breaking one makes two numbers in the paper incomparable.

- **One evaluator.** A rule and the MILP differ *only* in how they pick a setpoint.
  Envelope (`Basic_Functions.max_charge_now` / `max_discharge_now`), rates
  (`MILP_Household.interval_rate_vectors`), calendar (`day_calendar`) and settlement
  are shared, never re-implemented.
- **One invoice generator.** Every path — RL env, MILP, real meter profiles —
  accumulates into `si_invoice.InvoiceBuilder`. No invoice math anywhere else.
- **One env factory.** `MILP_Household.build_household_env`. Nothing about a
  particular study is defaulted there; each study passes its own price list,
  capacity and agreed-power rule.
- **One battery.** Physical constants in `MILP_Household`, economics in
  `Battery_Economics`, so two studies cannot quietly price the same pack differently.
- **Physics vs. nameplate.** `battery_capacity_kwh` is the usable window everything
  clamps against; `nominal_capacity_kwh` is the pack on the invoice. Only economics
  and cycle counting read the nameplate.
- **`hems_study.py` never imports matplotlib.** Figures are the notebook's half of
  the job. No figure is written from inside a batch run.
- **Nothing is restated that can be derived.** The household roster comes from the
  clustering (`study_units`), the notebook partition from `HYBRID_KINDS`
  (`pv_split`), labels from `CONTROLLER_ALGORITHM` / `FORECAST_KIND_LABELS`. Adding
  an arm to `STUDY_ARMS` should propagate without editing a list somewhere else.
- **Terminal SOC.** Every MILP strategy starts and ends at 50 % capacity;
  `Cost_EUR_Closed` values a rule's shortfall so the comparison is fair.
- **Causality.** Every controller except `price_oracle` (a deliberate diagnostic)
  uses only the past, the calendar and the day-ahead price.

## Running things

```bash
python Main/hems_study.py                 # the whole sweep, resumable
python Main/tune_rules.py AU|SI
python Main/tune_fixed_schedule.py AU|SI
python Main/run_hbd_benchmark.py [id ...]
python Clustering/cluster_households.py --dataset Ausgrid --k-max 30
```

From a notebook or driver: `hs.run_arms(...)`, then `hs.collect_results()` →
one long frame, then `hs.summarize(df)`.

Env overrides: `ERK_DATA_DIR`, `ERK_OUTPUT_ROOT`, `ERK_FORECAST_CACHE`,
`ERK_ORACLE_CACHE`.

## Caching and concurrency

- Work is checkpointed per **(arm, household)** under `Main/results_local/<arm>/<household>/`.
  A checkpoint carries the config digest it was produced under; a run under
  superseded rules is dropped, not resumed into.
- Content-keyed caches, shared on purpose across notebooks: `Main/forecast_cache/`,
  `Main/oracle_cache/`, `Main/hbd_params/`. All gitignored — they are caches, not
  results. `oracle_config` strips the four forecast-only keys, so one oracle solve
  serves every arm that differs only in its forecast.
- The three PV notebooks own **disjoint** arm sets (`pv_split.arm_specs(TRACK)`), so
  two kernels can run at once. Sweeping without `arms=OWN_SPECS` is the one way to
  get two kernels writing one checkpoint.
- Each notebook writes to its own `Results/Figures/` subdir — the tracks draw the
  same figure *names* over different arms.

## Testing

Plain scripts, no pytest. Each prints PASS/FAIL per check and exits non-zero on
failure.

```bash
cd Main && python test_hems_study.py      # ~1 min, short slice, no sweep
cd Main && python test_hbd_forecast.py
cd Main && python test_pv_split.py
cd "New pricing functions" && python test_primer.py && python test_souporaba.py
```

`test_hems_study.py` is not a unit-test suite: every check is a way this study has
actually been wrong (two batteries, two evaluators, two windows, two starting SOCs).
Treat a failure there as a result, not a flake.

## Working conventions

- **Comments carry the *why*.** This codebase documents the reasoning behind a
  choice — measurements taken, alternatives rejected, bugs found (see the `F9` / `F10`
  timezone notes in `hems_study.TariffCalculator`). Match that density; do not strip
  it. When you change a decision, update the comment that explains it.
- Departures from the upstream `hbd` port are marked `DEPARTURE` with a reason. Keep
  `Main/hbd_forecast.py` diffable against `cvxgrp/home-battery-dispatch`.
- The regulated Slovenian constants in `si_tarife.py` / `si_moc.py` carry sources and
  verification dates. Any change needs one.
- In notebooks, reload modules explicitly (`hs = importlib.reload(hs)`); `import`
  returns the kernel's first-run cache. A `TypeError`/`AttributeError` about code
  plainly present in a `.py` file means a stale module — re-run the imports/style cell.
- `Clustering/Ausgrid/*.csv` predate `cluster_households.py` and are authoritative
  for the published study. Re-running may permute labels; do not overwrite them.
- Don't commit caches, figures or `__pycache__` — `.gitignore` covers them.
