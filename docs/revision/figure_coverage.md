# Figure coverage: manuscript figures vs. rebuilt code

Maps every figure `STEC_Modelling/PNN_main.tex` embeds to the code that produces it, and
checks whether that code has been ported into `stec/`. Read-only audit against the primary
checkout (`/scratch2/arrueegg/WP4/PNN_STEC`, frozen) and the rebuild (`stec/viz/`). No
analysis was run and no figure was generated to produce this document.

**Correction to the task brief**: the manuscript defines **15** `\begin{figure}`
environments, not 16. `grep -c "begin{figure}"` returns 16 because it also matches the
commented-out template at `PNN_main.tex:102-105` (`% \begin{figure} ... % \end{figure}`,
the boilerplate example left over from the AGU template). The real count, confirmed by
`grep -n "^\\\\begin{figure}"` and by the 15 `Figure1.png`…`Figure15.png` files present in
`STEC_Modelling/`, is 15. Everything below is scoped to those 15.

## Figure-by-figure mapping

| # | Label | Caption gist | Generator | In `stec/`? |
|---|---|---|---|---|
| 1 | `temp_split` | Temporal train/val/test split, heatmap over years (2014–2024) × months | `src/data_processing/visualize_temporal_splits.py:create_timeline_heatmap` (default `start_year=2014, end_year=2024`, matches the caption range exactly) | No |
| 2 | `spatial_split` | Global map of IGS stations coloured by split (train blue, val green, test red) | `src/data_processing/split_new.py:plot_station_distribution` (lines 156–259; scatter colours `#215ACC`/`#5ACC21`/`#CC215A` render as blue/green/red as the caption says, though the in-code comments mislabel them "red"/"green"/"blue" — a pre-existing comment bug, flagged here since it sits directly in the generator, not fixed) | No |
| 3 | `network` | ResNet architecture schematic | **Hand-drawn**, not code-generated. Confirmed: `docs/ResNet.drawio` exists (alongside `MLP.drawio`, `MLP_no_unc.drawio`, `ResNet_noDropout.drawio`). Per CLAUDE.md this is expected and is not a coverage gap. | N/A |
| 4 | `pred_density` | Hexbin prediction density, 1:1 line | `src/viz/performance.py:plot_prediction_density` (line 302), reached via `src/viz/__init__.py:plot_test_metrics_for_subset` → `plot_test_metrics`, driven by `src/inference_testset.py` | No |
| 5 | `residuals_elev` | Residual boxplots by 5° elevation bin, MAE (green)/RMSE (orange) overlay | `src/viz/distributions.py:plot_residuals_vs_feature(df, "satele", ...)` (line 298) → `plot_binned_boxplot` (line 22); same `plot_test_metrics` call chain as #4 | No |
| 6 | `residuals_lat` | Residual boxplots by 10° solar-magnetic-latitude bin | `src/viz/spatial.py:plot_box_by_lat` (line 366) — signature and binning (`np.arange(-90, 91, 10)` on `sm_lat_ipp`) match the caption precisely; same call chain | No |
| 7 | `residuals_localtime` | Residual boxplots by hourly local solar time | `src/viz/distributions.py:plot_residuals_vs_local_time` (line 435); same call chain | No |
| 8 | `residuals_year_month` | Monthly residual boxplots, MAE/RMSE overlay | `src/viz/distributions.py:plot_box_by_date` (line 342); same call chain | No |
| 9 | `uncertainty` | Binned-by-predicted-σ absolute-error boxplots, with MAE (orange), mean predicted σ (red), mean epistemic (black), mean aleatoric (blue) curves | `src/viz/uncertainty.py:plot_binned_uncertainty_error_analysis` (line 526) — the four named series match the caption's four curves exactly; called from `plot_test_metrics_for_subset` when uncertainty columns are present | No |
| 10 | `improvements` | Daily % RMSE improvement of Direct STEC over VTEC+Map (orange)/IGS GIM+Map (green), time series | `src/multiday_evaluation.py:generate_aggregate_plots`, section 3 "Improvement statistics" (~line 990) — writes `{metric}_improvement_by_date_{dataset}.png` | No |
| 11 | `mae_rmse_finetuned` | RMSE (top)/MAE (bottom) vs. elevation, 3 curves (STEC blue, VTEC orange, GIM green), jittered error bars | Same function, section 4 "Elevation-dependent plots", "Combined RMSE/MAE Plot" block (~line 1265) — the x-offset jitter per method matches the caption's "slightly shifted... for visibility" note exactly | No |
| 12 | `pos_trend` | Daily 3D RMS positioning error, 4 methods, shaded std band, >10 m outliers excluded | `positioning/scripts/plot_results.py:plot_trends`, part 1 (lines 111–190) — writes `paper_trend_3d_rms_timeseries.png` | No (see below) |
| 13 | `pos_distribution_boxplot` | Overall 3D RMS error distribution, 4 methods, boxplot | `positioning/scripts/plot_results.py:plot_extended_analysis`, part 1 (~lines 242–278) — writes `paper_overall_distribution_boxplot.png` | No |
| 14 | `pos_improvement_timeseries` | Daily % improvement over IGS GIM+Map, 3 methods (STEC blue, VTEC orange, Pretrained purple) | `positioning/scripts/plot_results.py:plot_trends`, part 2 (~lines 191–235) — writes `paper_trend_improvement_timeseries.png` | No |
| 15 | `pos_cdf_3d_rms` | CDF of 3D RMS error, 4 methods | `positioning/scripts/plot_results.py:plot_extended_analysis`, part 2 (~lines 280–317) — writes `paper_cdf_3d_rms.png` | No |

**Every code-based generator is confirmed by content, not just by name**: binning ranges,
axis labels, marker/colour choices and output filenames were read and checked against each
caption rather than matched on a plausible-sounding function name.

## Is any of this actually ported?

**No.** All 14 code-based figures (everything except the hand-drawn #3) are still produced
only by pre-rebuild code — `src/viz/*.py` + `src/inference_testset.py` (#4–9),
`src/multiday_evaluation.py` (#10–11), `src/data_processing/*.py` (#1–2), and
`positioning/scripts/plot_results.py` (#12–15). None of these plotting functions exist
under `stec/`.

For the positioning figures this is not a gap that was missed — it is **explicit, by
design**, and stated in the rebuild's own code:

> `stec/positioning/metrics.py` docstring: "`plot_trends` and its helpers used to be
> duplicated verbatim between `positioning/scripts/run_pipeline.py` and
> `positioning/scripts/recompute_metrics.py`. That duplication is plotting code, out of
> scope here, and is not recreated - this module has no plotting."

`stec/positioning/metrics.py` ports the metrics *computation* behind Table 5 (via
`stec/analysis/positioning_summary.py`) and Table A1 (via `common_set_positioning.py`), and
`stec/analysis/daily_metrics.py` reproduces the STEC metrics behind Tables 3–4 from the
prediction store. So the *numbers* Figures 10–15 would plot are recomputable today with no
GPU and no re-inference — only the plotting step itself has no `stec/` counterpart for any
of the 15 manuscript figures.

**Colour-palette note for Figures 10–11**: `src/multiday_evaluation.py` (the pre-rebuild
generator for the improvement and elevation-RMSE/MAE figures) colours its four series from
`seaborn.color_palette("colorblind")` indices 0/1/2/4 — `#0173b2`/`#de8f05`/`#029e73`/`#cc78bc`
— not the `#1f77b4`/`#ff7f0e`/`#2ca02c`/`#9467bd` hex values CLAUDE.md pins and
`stec/viz/style.py` enforces. This is not a violation of the stated colour rule — CLAUDE.md
scopes that rule to `positioning/scripts/plot_results.py`, which is where the STEC/VTEC/GIM
hex constants actually originate — but it means the already-published Figures 10 and 11 use
visually different (if role-consistent: blue/orange/green/purple-ish) colours than the
`APPROACH_COLORS` palette the rebuild now treats as canonical. Worth knowing before assuming
the two colour sources are interchangeable.

## What `stec/viz/revision_figures.py` actually covers

None of its ~19 figure kinds (per its own module docstring) correspond to a numbered
manuscript figure. They are a **separate, additional figure set** built for the JGR-MLC
response letter (`docs/revision/response_to_reviewers.md`) and evidence summary, one family
per reviewer comment:

| Reviewer comment | Figure(s) |
|---|---|
| R2.2 | `relative_error_absolute`, `relative_error_normalised` |
| R2.5 | `architecture_search` |
| R1.4 | `activity_dst_*`, `activity_f107_*`, `stratified_*` |
| R1.5 | `weighting_ablation` |
| R1.7 | `storm_positioning_absolute`, `storm_positioning_improvement`, `positioning_tail` |
| R1.8 | `oracle_benchmark` |
| R1.3 | `madrigal_reference_offset`, `reference_precision` |
| R1.6 | `calibration_coverage`, `calibration_pit`, `ionex_rms_coverage`, `ionex_rms_crps_skill` |
| R2.3 | `station_independence` |
| R2.6 | `uncertainty_vs_error` |

Two of these are easy to mistake for manuscript figures on a name/topic match but are
confirmed distinct by content:

- `fig_uncertainty_vs_error` (R2.6) plots mean predicted σ against realised RMSE per
  predicted-σ decile, with a 1:1 "perfect calibration" line — a calibration scatter, not
  Figure 9's binned-boxplot-with-four-overlaid-curves (MAE/predicted-σ/epistemic/aleatoric).
- `fig_relative_error_absolute`/`_normalised` (R2.2) plot yearly RMSE and nRMSE of the
  *pretrained* model across the full 2014–2024 span — a new solar-cycle-coverage argument,
  not a stratification of any of Figures 4–11 (which are all 2024-test-set, finetuned-model
  figures).

`response_to_reviewers.md` itself refers to manuscript figures by number in prose ("Figures
5–8", "Figure 12", "without the 10 m outlier exclusion already used in Figure 12") when
building on them — confirming the response letter treats Figures 1–15 as fixed, existing
inputs, and adds new evidence around them rather than regenerating them.

## Colour and `save_plot`/`_notitle` rule check

Both hold in the ported code, verified by reading `stec/viz/style.py` and
`stec/viz/revision_figures.py`, and by the passing test suite:

- `APPROACH_COLORS` in `stec/viz/style.py` is pinned to exactly `#1f77b4`/`#ff7f0e`/
  `#2ca02c`/`#9467bd` for Direct STEC/VTEC + Mapping/IGS GIM + Mapping/Pretrained Direct
  STEC — the same values in `positioning/scripts/plot_results.py`.
- `NON_APPROACH_COLORS` (condition, oracle, CODE-GIM, dataset colours) is checked disjoint
  from `APPROACH_COLORS.values()` both at import time (an `AssertionError` in `style.py`
  itself) and by `tests/viz/test_style.py::test_no_non_approach_series_uses_an_approach_colour`.
  `CODE_GIM_COLOR` (`#7bc47f`) is separately pinned distinct from `GIM_COLOR` (`#2ca02c`).
- Every `fig_*` builder in `revision_figures.py` sources its approach hues only from
  `APPROACH_COLORS`/`METHOD_ORDER` and its non-approach hues only from `CONDITION_COLORS`/
  `ORACLE_COLOR`/`CODE_GIM_COLOR`/`DATASET_COLORS` — spot-checked across
  `fig_relative_error_*`, `fig_uncertainty_vs_error`, and the storm/calibration builders; no
  approach colour is reused for a condition, dataset or the oracle bound.
- `style.save_plot` and `revision_figures._save` both write `<name>.png` (titled, with a
  provenance footnote in `_save`'s case) and `<name>_notitle.png` (title and footnote
  stripped) for every figure, matching the `figures` stage's own caveat: "The `_notitle` and
  `_no_legend` variants are the manuscript figures; the titled copies are working copies."
  Confirmed by `tests/viz/test_style.py::test_save_plot_writes_titled_and_notitle_png` and
  `::test_save_plot_notitle_has_no_title_artist`, both passing.
- `pytest tests/viz -q` → 10 passed. `ruff check stec/viz/` → all checks passed.

## Gaps

**Manuscript figures with no `stec/` generator (reproducibility hole today): 14 of 15** —
every figure except the hand-drawn #3. All 14 have an *identified* pre-rebuild generator
(table above), so nothing would need to be remade by hand from scratch, but nothing in
`stec/pipeline/stages.py`'s `figures` stage (`python -m stec.viz.revision_figures`) touches
any of them. Regenerating any of Figures 1, 2, 4–15 today requires running the pre-rebuild
`src/` and `positioning/scripts/` code directly, not the rebuilt pipeline.

**Generators producing figures the manuscript doesn't use**: all ~19 figure kinds in
`stec/viz/revision_figures.py` (table above) — by design, since they answer reviewer
comments rather than illustrate the frozen manuscript body. Not a defect; the `figures`
stage's own `canonical_for` scope ("one PNG per revision figure, plus the `_notitle`
manuscript variants" — a phrase that itself conflates "revision figure" with "manuscript
variant") is worth tightening so a future reader doesn't assume it covers Figures 1–15.

## Bottom line: what could be regenerated today

- **From `stec/` alone: 0 of the 15.** No manuscript figure has a ported plotting
  function; `stec/viz/revision_figures.py` only builds the separate reviewer-response set.
- **From the pre-rebuild `src/`/`positioning/scripts/` code (unchanged, still present and
  runnable): 14 of 15** — every figure except #3, each traced above to a specific function
  reading data that either still exists (predictions store, `daily_summary*.csv`,
  `experiments/*/positioning/`) or is recomputable via the now-rebuilt `daily_metrics`/
  `positioning_summary`/`common_set_positioning` stages.
- **Not code-generated at all, confirmed and expected: Figure 3** (`docs/ResNet.drawio`).
