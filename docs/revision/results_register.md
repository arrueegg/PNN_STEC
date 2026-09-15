# Results register: every quantitative claim in the manuscript, and where it comes from

**What this is.** A checklist of every table, figure, and prose number in
`STEC_Modelling/PNN_main_revised.tex` (the working copy with tracked changes; read end to end
2026-09-15, 608 lines, md5 of the frozen `PNN_main.tex` comparison copy verified unchanged at
`352234f9b6a08e4aff2e6a009d78243e`), matched to the pipeline stage and artifact file that
currently backs it, plus a consistency verdict across the manuscript's own results. Task-ordered:
the checklist is Task 2, the verdict is Task 3, gaps are listed separately. **Nothing in this
document is a live number.** Every value quoted below is a snapshot read while writing this
register (2026-09-15) from the path named beside it. Re-read that path before using the number
for anything — this repository has a documented history of three separate documents quoting
three different values for the same Table 5 median because an artifact moved and the prose
didn't (CLAUDE.md's own framing of why the canonical-results table stopped carrying numbers).
This document inherits that same discipline and the same risk: it will start going stale the
day after it is written.

**This document does not fix anything.** Task 4 is explicit that reporting is the deliverable;
every open question below is left for the owner to resolve, and no `.tex`, pipeline stage, or
artifact was modified while writing it.

**Verification performed while writing this document**: `stec/pipeline/stages.py` declares 44
stages (`len(stec.pipeline.stages.STAGES)`, confirmed by import), 13 of which carry a
`canonical_for` string — both counts match this task's brief exactly. Every stage name and every
artifact path cited below was checked to exist on disk (`ls`/`test -e`) while writing this
register; none were assumed. `STEC_Modelling/PNN_main.tex`'s md5 was confirmed unchanged before
and is not touched by anything here.

## How this relates to what already exists

This register does not replace any of the following; it sits on top of them and says where they
now disagree with each other or with the revised manuscript.

- **`docs/revision/evidence_summary.md`** (394 lines) is the oldest of the group — written
  mid-rebuild (its own "Still running / pending" section and staleness markers date it to
  roughly 2026-08-24/25) and never fully updated since. Its `R1.5`/`R1.7`/`R1.8` positioning rows
  are superseded by `positioning_reporting.md` and `manuscript_change_list.md` (below); its R1.7
  storm numbers (+25.4%/+19.6%, quiet/storm) are now roughly 4-9x the current artifact's own
  numbers (§Consistency, item F). Its non-positioning rows (R1.2, R1.3, R1.6b, R2.x) were not
  re-checked in depth here and are not flagged as stale by anything found in this pass.
- **`multiday_results/revision_metrics_index.csv`** (50 lines) and
  **`multiday_results/revision_analyses_status.csv`** (21 lines) both point at `src/analysis/*.py`
  scripts and pre-restructure flat paths (`multiday_results/storm_stratification/...` rather than
  `multiday_results/analyses/storm_stratification/rebuilt/...`). Neither file was touched by the
  results-layout restructure (2026-08-21) or the `src/`→`stec/` port. Their reviewer-comment
  mapping (which CSV answers which `R#.#`) is still useful context; their paths are not current
  and should not be followed literally (§Consistency, item J).
- **`docs/revision/manuscript_change_list.md`** (442 lines, dated 2026-09-14 against commit
  `c83643b`) is the closest sibling to this document and did almost all of the Task-1/Task-2 work
  already — but against `STEC_Modelling/PNN_main.tex`, the **frozen pre-revision** copy, not the
  revised one this task reads. Cross-checking directly: essentially every one of its "CLAIM NO
  LONGER SUPPORTED" and "VALUE MOVED" rows has already been acted on in
  `PNN_main_revised.tex` (the 30%→18% median rewrite, the GIM-repair table cells, the Figure
  10/11 replacements, the Table 5 methodology change, the tail caveat in the conclusion, the
  Table 2 KL-warmup row). Two things in it are now themselves stale relative to
  `PNN_main_revised.tex`: its weighting-ablation numbers (rows 32-34, still the old
  mean-based/N≈10,200-10,400 figures) were superseded by a `weighting_ablation` re-run dated
  2026-09-15 08:30 — *after* this file's own 2026-09-14 snapshot — that the revised manuscript's
  Table 7 already reflects (§Consistency, item C); and its "New claims now available" §1 quotes
  a slightly older population-split N (8,442/2,205 paired) than the one the revised manuscript's
  own prose settled on (8,468/8,576 and 2,249/2,277, unpaired per-method N — see the checklist,
  row P8).
- **`docs/revision/positioning_reporting.md`** (338 lines, owner decision 2026-08-28, last
  substantively updated 2026-09-14) is the methodology decision record for Table 5 — median over
  mean, no outcome-based outlier filter, why. Its own §2 recommends pasting `TABLE5_NUMBERS.md`
  (the **full per-method** population, N=10,717 Direct STEC) directly into Table 5. The
  manuscript instead prints the **common-set** population (N=10,387, `TABLE5_COMMON_SET_NUMBERS.md`)
  for the table itself, while using the full per-method population for the surrounding prose and
  Figures 12-15. That divergence between what this document recommended and what the manuscript
  did is real and is not recorded anywhere, including in this file itself (§Consistency, item A).
- **`docs/revision/reviewer_comments_verbatim.md`** (427 lines) supplies the `R1.#`/`R2.#`
  labels used throughout the other documents and this one; its coverage table (which comments are
  answered at all) was not re-derived here and is taken as given.
- **`stec/pipeline/stages.py`** (2,315 lines, 44 stages) is the ground truth for "what stage
  produced this," used directly, not through any of the above documents, everywhere in the
  checklist below.

## How to read the checklist

Each row is one manuscript result. Columns: **Manuscript location** (section, table/figure
number or line); **Claim** (short); **Producing stage** (from `stec/pipeline/stages.py`, or "not
a declared stage" where the number comes from a static file or an undeclared script);
**Artifact path** (read the number from here, not from this table); **Dataset / population /
date range**; **N** (as currently reported by the artifact); **Outlier rule**; **Statistic**;
**Weighting**. `UNKNOWN` means this session could not establish the field from code or artifacts
in the time available — treat it as an open question, not a guess suppressed.

---

## A. Data, methods and the STEC chapter (Tables 1-4, Figures 1-11)

| # | Location | Claim | Stage | Artifact | Dataset / population / dates | N | Outlier rule | Statistic | Weighting |
|---|---|---|---|---|---|---|---|---|---|
| A1 | §2 Data, l.156 | "elevation angles ≥5°" — floor applied when the STEC database itself is built | not a declared `stec/` stage — set at STEC-DB generation time, upstream of everything in this repo | `stec/data/madrigal_reader.py:91` (`elevation_threshold`, default 5.0°, "matching the STEC database's own elevation cut"); `positioning/scripts/generate_reference_corrections.py` (5.0°, per `stec/positioning/metrics.py`'s docstring) | Every STEC-domain table (3, 4) and figure (4-11) inherits this floor | n/a | n/a — a data floor, not a per-analysis exclusion | n/a | n/a |
| A2 | §2 Data, l.170/175, Fig 1-2 | "~70% training, 15% validation, 15% test stations"; static temporal/station splits | `manuscript_figures` (Figures 1, 2 only; `canonical_for=None`) | `stec/data/splits/{train,val,test}_station.list` (360/76/78 stations per CLAUDE.md, = 70.0/14.8/15.2%); `plots/manuscript/dataset_construction/temp_split.csv` | All years 2014-2024, all split stations | 360/76/78 stations | n/a | count | n/a |
| A3 | Table 1, l.192-232 | Input feature list | `paper_tables` (`canonical_for="Tables 1 and 2"`) | `multiday_results/analyses/paper_tables/rebuilt/table1_features.csv` | Model config, not data | n/a | n/a | n/a | n/a |
| A4 | Table 2, l.265-295 | Hyperparameters (training, architecture, Bayesian head) | `paper_tables` (`canonical_for="Tables 1 and 2"`) | `multiday_results/analyses/paper_tables/rebuilt/table2_hyperparameters.csv` | Frozen paper config (`config/paper/pretrain_stec_config.yaml`) | n/a | n/a | n/a | n/a — **see Consistency item I**: the artifact carries 2 rows (variance floor, output bias init) the manuscript still omits, and the manuscript's "Early stopping patience 20 (15)" row has no matching row in this CSV at all |
| A5 | §3 Methods, l.315 | "elevation-angle bins of 5°" for geometry stratification | `elevation_metrics_finetuned` (Figure 11), `stratified_comparison` (R1.4, not in manuscript body) | `elevation_metrics_finetuned/rebuilt/per_day_by_elevation.csv` uses `np.arange(0, 91, 5)` | own, 242-day 2024 test period | varies per bin, ≥100 obs/day/bin/method guard | none | RMSE/MAE per bin | n/a |
| A6 | §4.1, l.332, Fig 4 | Pretrained model: Pearson ≈0.95, R² close to 0.9 | `pretrained_test_diagnostics` (`canonical_for=None`) feeding `manuscript_figures` | `multiday_results/analyses/pretrained_test_diagnostics/rebuilt/observations.parquet` (10,000,000-row cache) | pretrained model, **full multi-year test set 2014-2024** (all years, not the 2024-only fine-tuned test period) | 10,000,000 (cache) | none | Pearson r, R² pooled | n/a |
| A7 | §4.1, l.340-357, Fig 5-7 | Elevation/latitude/local-time error dependence, pretrained model | `pretrained_test_diagnostics` → `manuscript_figures` | same `observations.parquet` cache | same, full multi-year test set | same | none | binned MAE/RMSE | n/a |
| A8 | §4.1, l.365, Fig 8 | Year/month MAE-RMSE: 2018-2020 MAE 2.6-3.0/RMSE 3.8-4.5; 2023 8.0/11.8; 2024 9.2/14.0; Pearson "approximately 0.91-0.96 across all years" (revised from the frozen "0.93-0.95") | `pretrained_test_diagnostics` → `manuscript_figures` | same cache, per-year groupby | full multi-year test set, per year | per-year row counts within the 10M cache | none | per-year MAE/RMSE/Pearson r | n/a |
| A9 | §4.2, l.375, Fig 9 | Monotonic uncertainty-vs-error relationship | `uncertainty_error_relation` (R2.6/R1.2, `canonical_for=None`) | `multiday_results/analyses/uncertainty_error_relation/rebuilt/by_uncertainty.csv` | own, pretrained model's own cache (see caveat: fixed TECU-interval bins, not the retired first-day-sigma-decile scheme) | pooled over test period | none | mean absolute error per uncertainty bin | n/a |
| A10 | §4.2, l.397 | Aleatoric dominates; epistemic "comparatively small" | `uncertainty_error_relation` | `.../by_elevation.csv`'s `epistemic_share_%` column (5.1-6.6% across bins, per `manuscript_change_list.md` row 20 — not independently re-derived in this pass) | own, pretrained | pooled | none | epistemic share % | n/a |
| A11 | §4.3, l.401, Fig 10 | Daily RMSE improvement vs VTEC+Mapping and IGS GIM+Mapping, "gains frequently exceeding 15-30%" | `daily_metrics` (`canonical_for="Tables 3 and 4"`) → `manuscript_figures` | `multiday_results/analyses/daily_metrics/rebuilt/per_day.csv` | own, fine-tuned Direct STEC, 242-day 2024 test period | 242 days | none | per-day RMSE improvement % | n/a |
| A12 | §4.3, l.409-416, Fig 11 | Elevation-binned MAE/RMSE, 4 methods (Direct STEC, Pretrained, VTEC+Mapping, IGS GIM+Mapping); Pretrained "does not converge" at high elevation (revised claim — frozen text said all methods converge) | `elevation_metrics_finetuned` (`canonical_for="Figure 11 per-elevation error bars"`) | `multiday_results/analyses/elevation_metrics_finetuned/rebuilt/per_day_by_elevation.csv` | own, all four methods, 242-day 2024 test period | ≥100 obs/(day, bin, method), else dropped | none | across-day mean RMSE/MAE per 5° bin, error bars = across-day std | n/a |
| A13 | Table 3, l.420-433 | Aggregated RMSE/MAE/R² per model, mean±std across days: Direct STEC 6.92±1.14/3.88±0.49/0.97±0.01; Pretrained 13.45±4.84/9.37±3.86/0.87±0.12; VTEC+Mapping 8.96±1.47/5.21±0.71/0.95±0.01; IGS GIM+Mapping 8.28±0.99/5.30±0.63/0.95±0.01 (GIM row corrected — see Consistency item D) | `daily_metrics` (`canonical_for="Tables 3 and 4"`) | `multiday_results/analyses/daily_metrics/rebuilt/summary.csv`, `dataset=own_vtec_gim` | own, 242-day 2024 test period, all 4 methods matched on the same days | 242 days, 475,111,413 observations | none | **RMSE_mean / MAE_mean / R2_mean — mean of per-day values, not pooled** (`pooled_RMSE` in the same file is a different, consistently-higher number: 6.96/8.30/14.05/9.00 vs the reported 6.92/8.28/13.45/8.96 — see Consistency item D) | n/a |
| A14 | Table 4, l.439-452 | Same 4 rows, evaluated against Madrigal STEC: Direct STEC 14.63±3.43/8.79±1.90/0.85±0.03; Pretrained 17.29±4.78/11.78±3.81/0.79±0.10; VTEC+Mapping 13.60±2.96/8.27±1.71/0.87±0.02 (best); IGS GIM+Mapping 15.47±2.91/10.40±1.85/0.84±0.03 | `daily_metrics` (`canonical_for="Tables 3 and 4"`) | `multiday_results/analyses/daily_metrics/rebuilt/summary.csv`, `dataset=madrigal_vtec_gim` | Madrigal reference STEC, 238 of 242 days (DOY 199-202 absent on this host) | 238 days, 448,938,780 observations | none | RMSE_mean/MAE_mean/R2_mean — mean of per-day values (pooled equivalents: 15.01/15.73/17.98/13.90, all higher — see Consistency item D) | n/a — **must be read alongside `madrigal_reference_offset`** (below); not an accuracy measurement, see l.325/437's own caveats |

## B. GNSS positioning chapter (Table 5-7, Figures 12-15 + Appendix)

| # | Location | Claim | Stage | Artifact | Dataset / population / dates | N | Outlier rule | Statistic | Weighting |
|---|---|---|---|---|---|---|---|---|---|
| P1 | Abstract l.78, PLS l.84, §4.4 l.459/500, Conclusion l.550 | Median 3D RMS improvement ≈18% over IGS GIM+Mapping, "over the common set of station-days solved by all four methods under both weighting schemes" | `positioning_distributions` (`canonical_for="Table 5"`) | `multiday_results/analyses/positioning_distributions/rebuilt/TABLE5_COMMON_SET_NUMBERS.md` | 2024 test period, iono weighting, **4-method × 2-weighting common set** | 10,387 | **none** (no outcome-based exclusion) | median | iono |
| P2 | Table 5, l.487-501 | Median/IQR/P95/P99/exceedance-at-5,10,20,50m, per method | `positioning_distributions` (`canonical_for="Table 5"`) | `multiday_results/analyses/positioning_distributions/rebuilt/TABLE5_COMMON_SET_NUMBERS.md` | same as P1 | 10,387 (all 4 rows) | none | median, IQR, P95, P99, exceedance % | iono |
| P3 | Table 6, l.503-519 | Median 3D/2D(horizontal)/Up(vertical) error, same population as Table 5 | `positioning_distributions` — **same stage as Table 5, but Table 6 itself carries no `canonical_for`** (see Consistency item C) | `multiday_results/analyses/positioning_distributions/rebuilt/common_set_component_medians.csv` | same as P1 | 10,387 | none | median | iono |
| P4 | Table 7, l.521-535 | Elevation-weighting vs predicted-uncertainty-weighting 3D RMS, per correction, same population as Table 5 | `weighting_ablation` (`canonical_for=None`) | `multiday_results/analyses/weighting_ablation/rebuilt/common_set.csv` | same population as P1, both weightings compared directly | 10,387 (all 4 corrections) | none | **median** (`gain_median_%` — changed from mean 2026-09-15, same day this table was last touched; means are still written to the CSV for the sensitivity comparison only) | elev vs iono, paired |
| P5 | §4.4, l.459 | One station-day (URUM, DOY 365) where PPPx did not converge under either weighting moves the Direct STEC elev-weighted **mean** by 25% (2.283 m → 1.706 m) while leaving the median unmoved | `weighting_ablation` | `common_set.csv`'s `elev_mean` column (2.2828) for the "with" value; the "without" value (1.706) is the mean after dropping that one row, not itself a separate declared-stage column — recompute rather than assume a source file for it | same population as P1 | 10,387, minus 1 for the "without" figure | n/a — a single genuine PPPx solve failure, not an outlier rule | mean (used only to illustrate why the table reports medians) | elev |
| P6 | §4.4, l.461, first sentence | Population split by whether the station-day has a real STEC-DB observation ("original") or only a geometry-only fallback ("recovered"): Direct STEC median 0.778 m vs GIM 1.015 m (original, N=8,468/8,576); Direct STEC 2.771 m vs GIM 1.991 m (recovered, N=2,249/2,277) | `positioning_diagnostics` (`canonical_for=None`) feeding `positioning_distributions` | `multiday_results/analyses/positioning_distributions/rebuilt/population_percentile_summary.csv` | 2024 test period, iono weighting, **full per-method population** (not the Table 5 common set — see Consistency item A) | 8,468/8,576/2,249/2,277 (per-method, per-population) | none | median | iono |
| P7 | §4.4, l.461, second sentence | Direct STEC exceeds 10 m on 0.65% of station-days vs 0.15% for IGS GIM; P95/P99 5.29/8.95 m vs 4.13/5.80 m | `positioning_distributions` (`canonical_for="Table 5"`) | `multiday_results/analyses/positioning_distributions/rebuilt/TABLE5_NUMBERS.md` | 2024 test period, iono weighting, **full per-method population** | Direct STEC 10,717; IGS GIM 10,853 | none | exceedance %, P95, P99 | iono |
| P8 | Fig 12, l.463-468 | Daily median 3D RMS + IQR shading, 4 methods, "no station-day is excluded... vertical axis is clipped for readability" | `positioning_distributions_figures` (`canonical_for=None`) | `plots/manuscript/positioning_2024/pos_trend*` (or the CSV twin under `plots/positioning_distributions/`) | 2024 test period, iono weighting, full per-method population | 10,717-10,853 depending on method | none (view-clipped only, not the data) | daily median + IQR | iono |
| P9 | Fig 13, l.471-479 | Boxplot distribution, all 4 methods; mean swings −1.2% (no exclusion) to +16.0% (5 m exclusion) vs IGS GIM; median stays 18-21% across every threshold; exceeds-10m counts 70/115/80/16 for Direct STEC/Pretrained/VTEC/IGS GIM | `positioning_diagnostics` (`canonical_for=None`, headline-sensitivity table) + `positioning_distributions` (exceedance counts) | `multiday_results/analyses/positioning_diagnostics/rebuilt/outlier_headline_sensitivity.csv`; `multiday_results/analyses/positioning_distributions/rebuilt/overall_exceedance.csv` | 2024 test period, iono weighting, full per-method population | Direct STEC 10,717 / GIM 10,853 (the "none" row); the mean/median-swing table itself uses the paired STEC/GIM counts shown in the sensitivity CSV | none for the headline (that's the point of the figure); the sensitivity table sweeps 5/10/20/50 m for comparison | mean (shown to be non-robust) and median (reported) | iono |
| P10 | l.459/§Appendix l.538/566 | Pretrained model's worst positioning excursions coincide with Dst < −300 nT storms, DOY 132-133 and 282-285 | `positioning_diagnostics` (`canonical_for=None`) | `multiday_results/analyses/positioning_diagnostics/rebuilt/daily_timeseries.csv`, `FINDINGS.md` | 2024 test period, iono weighting | n/a (event identification, not a population statistic) | none | daily mean by DOY | iono |
| P11 | Fig 14 (Appendix), l.559-566 | Daily relative improvement vs IGS GIM+Mapping, computed from daily median 3D RMS, 3 methods (Direct STEC, VTEC+Mapping, Pretrained) | `positioning_distributions_figures` / `positioning_diagnostics_figures` (both `canonical_for=None`) | `plots/positioning_diagnostics/positioning_2024/daily_timeseries.csv` or the `positioning_distributions` figure twin | 2024 test period, iono weighting, full per-method population | 10,717-10,853 | none (view-clipped) | daily median, relative % | iono |
| P12 | Fig 15 (Appendix), l.568-573 | CDF of 3D RMS errors, 4 methods, median/P95 markers, "full population reported in Table 5" | `positioning_distributions_figures` (`canonical_for=None`) | `plots/manuscript/positioning_2024/cdf_unfiltered*` | 2024 test period, iono weighting, **full per-method population** (caption itself points to Table 5, which is the narrower common-set population — a cross-reference across the two populations named directly in the caption text) | 10,717-10,853 | none | CDF, median, P95 markers | iono |

---

## Consistency verdict

Grouped by manuscript chapter. Each item states what disagrees, why it matters, and the options
available — this document does not pick one; the owner decides.

### STEC chapter (Tables 1-4, Figures 1-11)

**Item D — Table 3/4 report the mean of per-day RMSE; a pooled (observation-weighted) RMSE sits
in the same source file and is a different, consistently higher number.** `daily_metrics`'s own
caveat states this explicitly: "The published RMSE is RMSE_mean... pooled_RMSE in the same file
is over observations and is consistently higher; the two are not interchangeable." Confirmed
directly in `summary.csv`: Table 4's VTEC+Mapping row is 13.60 (mean-of-days, what the
manuscript prints) against a pooled 13.90 in the same row. **The manuscript itself never mixes
the two** — every number I traced back to Table 3/4 is consistently the mean-of-days figure.
The risk is narrower than "the manuscript is inconsistent": it's that the pooled number sits one
column over in the same CSV, is close enough in magnitude to be mistaken for the mean at a
glance, and has already caused one near-miss — `madrigal_method_offset_comparison`'s first draft
used the pooled ranking (13.90 < 15.01 < 15.73 < 17.98) for a diagnostic, correctly labelled, but
the module needed an owner correction before its headline framing was safe (see
`manuscript_change_list.md` §11). **Options**: state the mean-vs-pooled distinction as a
footnote on Tables 3/4 (cheap, forecloses the ambiguity for a reviewer who pulls the CSV
directly); or leave it, since the manuscript's own text is not actually inconsistent.

**Item H — Table 2 lists "Early stopping patience: 20 (15)," but the canonical stage's own CSV
has no row for it; two other hyperparameters the CSV does carry are still absent from the
table.** `paper_tables/rebuilt/table2_hyperparameters.csv` has 22 rows and no `patience` entry at
all — the manuscript's patience cell is not machine-checked by the stage declared
`canonical_for="Tables 1 and 2"`. Separately, the stage's own caveat says "Table 2 includes three
hyperparameters the submitted manuscript omits: the KL warmup..., the variance floor, and the
output bias initialisation." The revision added the KL-warmup row (now present, in blue, at
l.277) but **not** the variance floor (0.001) or the output bias init (15.5 TECU) rows — both
still sit only in the CSV. The stage's caveat text ("three hyperparameters... omits") is itself
now stale, since one of the three has been added. **Options**: add the two remaining rows (cheap,
the CSV already has them); update the stage's caveat text to say "two" instead of "three"; add a
`patience` row to `table2_hyperparameters.csv` so every Table 2 cell has a machine source, or
accept it as a hand-verified constant that doesn't need one.

**Item I — Figures 5-8 rest on the Gate F equivalence argument, not an independent re-derivation
of the pretrained-model per-bin numbers.** `manuscript_change_list.md` states this directly
("their SAME verdict rests on the Gate F equivalence argument... not on an independent re-binning
of the raw parquet"). This document did not re-derive them either — carried forward as an
open point rather than silently upgraded to "confirmed."

### Positioning chapter (Table 5-7, Figures 12-15, Appendix)

**Item A — Tables 5/6/7 and the surrounding prose/figures use two different populations in the
same subsection, and the manuscript never states this.** Tables 5, 6 and 7 all restrict to the
**4-method × 2-weighting common set** (N=10,387, `coverage_common_station_days` — every
station-day solved by Direct STEC, Pretrained, VTEC+Mapping and IGS GIM under *both* elev and
iono weighting). The prose immediately before Table 5 (l.461, the population-split and tail
paragraphs) and Figures 12-15 all use the **full per-method population** under iono weighting
only (Direct STEC N=10,717, Pretrained N=10,819, VTEC N=10,826, IGS GIM N=10,853 —
`TABLE5_NUMBERS.md`, not `TABLE5_COMMON_SET_NUMBERS.md`). Every number I traced (P6, P7, P9, P11,
P12) is internally correct against its own source file, so this is not a wrong-number problem —
but a reader comparing Table 5's median (0.891 m, N=10,387) against Figure 13's caption
(exceeds-10m counts computed on N=10,717-10,853) is comparing two different denominators without
being told so. `positioning_distributions`'s own caveat documents the split ("The overall/
regime/population sections (Figures 12-15) are iono weighting only... The common_set_* sections
... additionally read WEIGHTING_RUN... to define the population restriction"), so the code is not
confused about which population it is using — the manuscript text simply doesn't carry that
distinction to the reader. **Options**: state explicitly, once, near Table 5 or Figure 12, that
the table restricts to the population solved under both weighting schemes while the figures and
surrounding prose use the (slightly larger) iono-only population, and why (the table needs a
population that supports the elev/iono weighting comparison in Table 7; the figures don't); or
move everything to one population (loses either the weighting-comparison table's exact match to
Table 5, or the figures' larger, more complete sample).

**Item G — `positioning_reporting.md`'s own recommended Table 5 content is not what the
manuscript printed, and the document was never updated to say so.** §2 of that file explicitly
recommends "`TABLE5_NUMBERS.md`... reproduced exactly... can be pasted in directly" — the full
per-method population, N=10,717 for Direct STEC. The manuscript instead prints the common-set
population (N=10,387). This may be a deliberate refinement made after `positioning_reporting.md`
was last edited (2026-09-14) rather than an oversight — the common-set choice is more defensible
for a table that sits next to a weighting-ablation table over the same rows — but nothing records
that the recommendation changed or why. **Options**: add a short dated addendum to
`positioning_reporting.md` §2 recording the final choice and the reason; or treat the manuscript
itself as the record and let this decision document age out for Table 5's specific population
(it would still be current for the median-over-mean, no-outlier-filter methodology, which both
populations share).

**Item C — Table 6 and Table 7 have no `canonical_for` owner, and Table 7's underlying stage has
weaker pipeline protection than Table 5/6's.** Confirmed directly: `positioning_distributions`
carries `canonical_for="Table 5"` only — Table 6 (`common_set_component_medians.csv`) is written
by the same stage but is not itself named, so nothing in the registry's one-owner check protects
it specifically (it does inherit a `min_rows: 4` floor and the stage's general checks, since it's
declared as one of that stage's outputs). `weighting_ablation` (Table 7) carries `canonical_for=
None` and, unlike `positioning_distributions`, declares its output only at the **directory**
level (`outputs=[str(WEIGHTING_ABLATION_DIR)]`) with no `min_rows` and no `checks` entry — the
registry's own stated convention is that a directory-level output carries no row count, so a
truncated or empty `common_set.csv` would not be caught by anything the pipeline currently
asserts. This is the same class of gap the registry was built to prevent for Tables 3/4 (see
`daily_metrics`'s comment about `min_rows={}` being "the audit's smoking gun" before individual
files were declared) — Table 7 is currently in the position Tables 3/4 used to be in. **Options**:
give `weighting_ablation` a `canonical_for="Table 7"` and declare `common_set.csv` (plus a
`min_rows` floor and a same-four-methods check) as an explicit output, mirroring what
`daily_metrics` and `positioning_distributions` already do; or leave it, since nothing found in
this pass shows the current `common_set.csv` is actually wrong — only that nothing would catch it
quickly if it broke.

**Resolved 2026-09-15** (not one of this document's own instructed items, but fixed the same day
as B/F above by the same pass, since it is the exact gap this register just flagged). Took the
first option for both tables: `weighting_ablation` now carries `canonical_for="Table 7"`, with
`common_set.csv` declared as an explicit output, a `min_rows: 4` floor, and a new
`weighting_ablation_common_set_has_all_four_methods` check mirroring
`positioning_summary_overall_has_all_four_methods`'s shape (catches a reindex-guaranteed row
that is present but NaN, which `min_rows` alone cannot). `positioning_distributions` was widened
from `canonical_for="Table 5"` to `canonical_for="Tables 5 and 6"` - one stage legitimately
owning two manuscript tables, the same precedent `daily_metrics` already set for "Tables 3 and
4"; `common_set_component_medians.csv` (Table 6) already had a `min_rows: 4` floor before this
pass, which - unlike `overall_percentile_summary.csv`'s reindex risk - is a complete guarantee
here, since `component_medians` groups by whatever `Method` values are actually present rather
than reindexing, so 4 real rows implies all 4 methods are there. `registry.validate()` passes
with 44 stages, 14 distinct `canonical_for` values (up from 13), no duplicate-ownership error -
confirming this was genuinely two previously-unclaimed deliverables, not a masked double-claim.

**Item B — `storm_stratification`, `positioning_robustness` and `oracle_benchmark` still apply
the retired 10 m outcome-based exclusion, and two of the three still report a mean headline;
none of their numbers currently appear in the manuscript body, but they do appear in
`response_to_reviewers.md`/`evidence_summary.md`.** Confirmed by reading the code directly, not
inferring from caveats:
- `storm_stratification.py:200`, `merged = pm.exclude_outlier_station_days(merged)` — and the
  module's own top-of-file comment (l.82-85) still says "Figure 12 of the paper excludes
  station-days worse than 10 m as extreme outliers," which is no longer true of Figure 12 in the
  revised manuscript (l.466: "No station-day is excluded... does not reflect an outlier
  removal") — the comment is stale in the same direction as the methodology.
- `positioning_robustness.py:99`, same `exclude_outlier_station_days` call; computes and reports
  both mean and median (l.111-112), with mean-based fractions-above-threshold as its main output.
- `oracle_benchmark.py:182`, same call; reports mean and median (l.231-232). Already flagged
  elsewhere as structurally not comparable to Table 5 (elev weighting, own restricted set) — this
  is an additional, independent reason beyond that one.

None of these three stages' numbers are quoted in `PNN_main_revised.tex` — the manuscript body
does not mention storm/quiet percentages, an oracle floor, or a horizontal/vertical robustness
split anywhere (confirmed by grep; the only storm-related manuscript content is the qualitative
Dst<−300 nT DOY identification in P10, which comes from `positioning_diagnostics`, not
`storm_stratification`). The exposure is therefore currently confined to the **response letter
and its supporting docs**, not the paper itself, but it is live there: `response_to_reviewers.md`
(l.285-288, edited 2026-09-15) still states "+25.4% over IGS GIM in quiet and +19.6% in storm
conditions" — the current `storm_stratification/rebuilt/improvement_over_gim.csv` (regenerated
2026-09-14) reads quiet +5.20% / storm **−2.68%** (Direct STEC now loses to GIM on storm days by
this stage's own mean statistic). `manuscript_change_list.md`, edited in the **same commit**
(`ce2a2ed`) as `response_to_reviewers.md`, already documents this exact drift and recommends the
median-based replacement (quiet median 0.862→1.059 m, +22.8%; GIM 1.065→1.234 m, +15.9% —
`docs/revision/positioning_reporting.md` §4) but that fix has not yet been carried into
`response_to_reviewers.md`'s own text. **Options**: apply the same median/no-exclusion
methodology used for Table 5 to these three stages (there is already a working example in
`positioning_distributions`'s `regime_percentile_summary.csv`, which reports the quiet/storm
median split under the correct methodology today); or leave the stages as they are (they answer
`R1.7`/`R2.7`/`R2.8` reviewer comments, not manuscript claims, so a mean-based sensitivity
comparison may be intentional context rather than a bug) but update
`response_to_reviewers.md`/`evidence_summary.md` to the current numbers either way, since those
are live documents that will go out under the authors' names.

**Resolved 2026-09-15.** Took the first option above for two of the three stages, and declared
the third's exclusion explicit rather than following it: `stec/analysis/storm_stratification.py`
and `stec/analysis/positioning_robustness.py` both dropped `pm.exclude_outlier_station_days`,
restricted to the same 4-method × 2-weighting common set Tables 5-7 use
(`common_set_positioning.coverage_common_station_days`, N=10,387 - previously each method's own,
mismatched population, e.g. Direct STEC 10,647 against IGS GIM's 10,837), and made the median the
headline statistic (mean columns kept, explicitly suffixed `_mean_%`/`_mean_m`, for the
sensitivity comparison only). `oracle_benchmark.py` was left untouched, per the owner's explicit
instruction that it answers a self-contained ratio-to-floor question under its own elev-weighted,
all-four-method-restricted methodology - not an unfixed instance of the same problem; its Stage
caveats in `stec/pipeline/stages.py` were extended to say so, so this register (and any future
reader of it) stops flagging it as one.

Both stages' pipeline declarations gained `WEIGHTING_RUN` as a declared input (they now read
`coverage_common_station_days`, which opens `multiday_summary_all_weightings.csv` - undeclared
before this pass, which would have left that file free to change silently under these stages'
skip decisions, the exact class of bug `WEIGHTING_RUN` was already added to `weighting_ablation`
to prevent).

New numbers, `storm_stratification/rebuilt/improvement_over_gim.csv` (median headline,
N=10,387 = 8,706 quiet + 1,681 storm): Direct STEC **+19.5% quiet / +14.5% storm** over IGS GIM
(supersedes this section's own quoted +5.20%/−2.68% mean-based figures above, which themselves
already superseded the response letter's +25.4%/+19.6%, which superseded the published
+31.9%/+26.3% - three superseded restatements, not three disagreeing numbers). The mean, on the
same fixed population, now reads +1.3%/−7.2% (still sign-flips under storm - confirming this was
never a population-size artifact, only a statistic-choice one). `positioning_robustness/rebuilt/`
gained the matching median columns in `tail_distribution.csv` (already had both; median is now
reported first) and `error_components.csv` (previously mean-only). `response_to_reviewers.md`
and `evidence_summary.md`'s R1.7 sections were rewritten with the new median figures throughout
(storm-time, tails, vertical/horizontal), replacing every mean-based number this item flagged.

### Item E — Elevation cutoff reconciliation (cross-cutting, flagged as the largest unmeasured risk)

**Settled from the code, not measured end-to-end.** Three different elevation floors are baked
into three different stages of the pipeline, and the manuscript states only one of them:

| Value | Where | What it gates |
|---|---:|---|
| **5°** | `positioning/scripts/generate_reference_corrections.py` ("matching the STEC database's own elevation cut") | Which observations get a Direct STEC / VTEC correction generated at all — the same floor stated in the manuscript's Data section (l.156) |
| **5°** | `stec/data/madrigal_reader.py`'s `elevation_threshold` default; `src/compare_stec_vtec_gim.py`'s Madrigal loader | Which Madrigal comparison rows enter Table 4 |
| **7°** | `positioning/positioning_eval/generate_ini.py`'s `elev_mask` default | Which observations PPPx actually uses to solve a station-day's position — i.e., every positioning number in Tables 5-7 and Figures 12-15 |

This is documented in the code itself, not newly discovered here: `stec/positioning/metrics.py`'s
module docstring names all three values verbatim and states "no cutoff parameter was added to
avoid implying a resolution that doesn't exist," and `stec/analysis/divergences.py` carries it as
Divergence #6 ("Defect 11: elevation cutoffs never reconciled across the pipeline"), status
`not_yet_ported` / `unmeasurable now` — reconciling it "would require... re-run the STEC
comparison and PPPx over a sample of days at that cutoff," which has not been done.

**Why it matters**: the manuscript's Data section states a single floor (≥5°) and Figure 11's
headline finding is that Direct STEC's advantage over the mapped baselines is largest at low
elevation. But the positioning experiment behind Tables 5-7 never lets PPPx see any observation
between 5° and 7° elevation at all — the elevation band where Figure 11 shows the STEC-domain
gains are largest is silently absent from the positioning solve. The manuscript does not state
this anywhere, and the actual size of the effect has never been measured (both `divergences.py`
and `metrics.py` are explicit that it isn't). This does not mean the positioning result is wrong
— PPPx's own 7° mask is a common, defensible choice independent of this project — but the
STEC-domain and positioning-domain results are not evaluated over the same observation
population, and the manuscript's single stated elevation floor implies otherwise. **Options**:
state the 7°/5° distinction explicitly in the positioning methods paragraph (cheap, and it is
already true of the published result, so it costs no numbers); or treat reconciling the cutoffs
as the retraining-scale investigation `divergences.py` already scopes it as, and leave it stated
as a limitation rather than resolved.

### Documents that are stale as of this pass, but are not manuscript inconsistencies

**Item F — `evidence_summary.md` and `response_to_reviewers.md`'s storm/quiet numbers are stale
relative to the live artifact** (see Item B above; repeated here because it is a documents
problem, not a manuscript problem — the manuscript makes no storm/quiet percentage claim at all).

**Resolved 2026-09-15**, as part of Item B's fix. Both documents' R1.7 sections were rewritten in
place (history kept in git, not in this file) with the current median-based, common-set figures.
Every number changed, old → new:
- Storm-time improvement over IGS GIM: quiet +25.4% → **+19.5%**, storm +19.6% → **+14.5%**
  (both documents; supersedes the even-older published +31.9%/+26.3%, also still named in the
  rewritten text for the record).
- Quiet-to-storm degradation, Direct STEC vs the pretrained-only variant: +19.6%/+41.2% →
  **+23.5%/+46.3%**; both documents now also state the previously-unmentioned fact that VTEC +
  Mapping (+8.9%) and IGS GIM itself (+16.3%) degrade *less* in absolute terms than Direct STEC -
  the "degrades least" framing survives only as a Direct-STEC-vs-pretrained comparison, not as a
  claim about all four methods.
- Tails: p95 3.66 m (Direct STEC) vs 4.10 m (GIM), "best through p95" → **5.27 m vs 4.13 m, GIM
  already ahead at p95**; p99 5.66 m vs 6.24 m → **5.76 m vs 8.92 m**; 15.9% vs 28.1% of
  station-days above 2 m → **24.4% vs 28.5%**. The crossover point moved earlier because the 10 m
  exclusion this item flagged as still-applied is now dropped, letting the extreme tail back in.
- Vertical/horizontal error reduction: 25%/22% → **17.0%/20.6%** (medians; the mean-based figures
  are no longer quoted as the headline in either document).
- Both documents' stale "37,209-row, pre-recovery-sweep, not final" staleness markers were
  removed and replaced with the current, settled N=10,387 common-set population statement -
  the recovery sweep this marker was waiting on has since completed (see the canonical-results
  table in `CLAUDE.md`).

Not touched, and not stale: `docs/revision/positioning_reporting.md`'s own §4 storm/quiet numbers
(0.866→1.076 m, +24.2% quiet-to-storm for Direct STEC; 1.065→1.234 m, +15.9% for GIM) come from
`positioning_distributions/rebuilt/regime_percentile_summary.csv` - the full per-method
population, not this item's 4-method × 2-weighting common set - and were re-checked against that
CSV while writing this resolution (current values: 0.8658→1.0755 m, +24.2%; 1.0648→1.2344 m,
+15.9% - matches to rounding). That document answers the same reviewer comment from a different,
already-current population; it was left as is rather than edited to match this item's numbers,
since the two populations are both legitimate and neither supersedes the other.

**Item J — `multiday_results/revision_metrics_index.csv` and `revision_analyses_status.csv` name
`src/analysis/*.py` scripts and pre-restructure paths that no longer produce anything.** Every
listed script has a `stec/analysis/`-package equivalent declared in `stec/pipeline/stages.py`
today (e.g. `storm_stratification.py`→ stage `storm_stratification`, output now under
`multiday_results/analyses/storm_stratification/rebuilt/`, not the flat path the index names).
Not a manuscript-facing problem, but anyone using these two CSVs as a file-finding index rather
than a reviewer-comment index will be looking in the wrong place.

---

## Gaps

**Asserted in supporting documents, not yet in the manuscript** (each already flagged as a
recommendation in `manuscript_change_list.md`'s "New claims now available," not newly found
here, but confirmed still absent from `PNN_main_revised.tex` by direct reading):
- Computational cost (pretrain ≈6.25 GPU-hours, fine-tuning 15.4 GPU-hours over 242 days,
  inference 8,605 obs/s) — `computational_cost/rebuilt/cost_summary.csv` — currently only in
  `response_to_reviewers.md` (R2.8h), not the manuscript body.
- The station-level attribution result (STEC RMSE predicts absolute Direct STEC positioning
  error, ρ=0.82, but not competitiveness against GIM, ρ=0.14) —
  `positioning_geography/rebuilt/stec_positioning_correlations.csv`, itself flagged as
  provisional (written by the undeclared `positioning_model_attribution.py`, last run
  2026-08-28, not re-run against the current population — see `manuscript_change_list.md` §9).
- The positioning-domain distance-to-nearest-training-station correlation (a second line of
  evidence for R2.3, alongside the STEC-domain `station_independence` result that is already in
  the response letter) — same source file as above, same provisional-freshness caveat.
- Data-coverage limitations stated in `manuscript_change_list.md`'s own "Limitations that must be
  stated" section (DOY 303/338/348 entirely absent from this host; 24 station-days across 6 DOYs
  unreachable in either STEC database; 26 station-days with no SINEX ground truth for that day;
  3 of 242 per-day source files independently truncated at the PPPx level) — none currently
  appear in the manuscript's Data section or as a stated limitation anywhere in the text.

**Declared pipeline stages with `canonical_for` set, answering a reviewer comment, that do not
surface in the manuscript body at all** (expected, not a defect — these back the response letter,
not the paper; listed for completeness per Task 3's "anything an artifact provides that the
manuscript does not yet use"): `temporal_regime_split` / `temporal_regime_activity_matched`
(R2.1, interpolation/extrapolation split), `epistemic_scale_diagnostic` (R1.2),
`madrigal_reference_offset` / `madrigal_method_offset_comparison` (R1.3 — though the manuscript's
own l.325 caveat about Madrigal reference inconsistency is clearly informed by this analysis,
even without citing it by number), `dstec_evaluation` (R1.3, differential STEC vs GIM). Also
undeclared but present in the response letter and not the manuscript: `station_independence`
(R2.3), `uncertainty_calibration`/`uncertainty_calibration_pretrained` (R1.6),
`ionex_rms_benchmark`/`ionex_rms_benchmark_code` (R1.6b), `stratified_comparison`/
`activity_stratification` (R1.4), `hyperparameter_search` (R2.5/R2.8b), `mapping_function_
consistency` (R1.3), `positioning_geography` (R2.3/discussion).

**Where this session could not determine a field (marked `UNKNOWN` rather than guessed)**:
- The exact construction of the frozen manuscript's original "27,205 station-days solved under
  both schemes" figure (row 32 of `manuscript_change_list.md`) — could not be reproduced from any
  single current artifact; that document already flags this rather than guessing, and this
  session did not find a way to resolve it either.
- Whether Figures 5-8's per-bin numbers for the pretrained model have ever been independently
  re-derived from the raw parquet rather than checked only via the Gate F equivalence argument
  (Item I above) — `manuscript_change_list.md` states this gap; this session did not close it.
- The precise current row-count reconciliation between `positioning_coverage`'s "10,712 solved by
  all methods" (its own most recent caveat, iono weighting) and `TABLE5_NUMBERS.md`'s per-method
  Direct STEC N=10,717 — the two are different denominators (four-way intersection vs one
  method's own total) and are not necessarily expected to match, but this session did not verify
  the arithmetic that would confirm the 5-row gap is fully explained by that distinction rather
  than partly by something else.

---

## Tallies

- **Results inventoried**: 26 rows (14 in the STEC chapter, 12 in the positioning chapter),
  covering all 15 figures, all 7 tables, and every prose number this session identified in the
  abstract, plain-language summary, results, conclusion and appendix. Several rows cover a
  restated number appearing in more than one location (e.g. the ≈18% positioning figure appears
  in the abstract, plain-language summary, results prose and conclusion; the row lists all four
  rather than being duplicated four times, per Task 4's instruction to keep this readable).
- **Inconsistencies found**: 10, labelled A-J above. A, B, C and D confirm and extend the four
  the owner already knew about (the Table 5/6/7-vs-figures population split; the three stages
  still on the retired 10 m/mean methodology; Table 6/7's missing ownership; the Table 3/4
  mean-vs-pooled risk). E (the elevation-cutoff reconciliation) and G (the manuscript's Table 5
  population diverging from `positioning_reporting.md`'s own recommendation) are new findings
  from this pass, both in the positioning chapter or cross-cutting. F and J are findings about
  the supporting documents rather than the manuscript itself (stale storm numbers in
  `response_to_reviewers.md`/`evidence_summary.md`; stale paths in the two revision-metrics
  index CSVs). H and I are new, narrower STEC-chapter findings (Table 2's still-missing
  hyperparameter rows and unbacked patience cell; Figures 5-8's evidence resting only on Gate F
  equivalence).
- **Could not determine**: 3 items, listed under Gaps above, marked `UNKNOWN` rather than guessed.
