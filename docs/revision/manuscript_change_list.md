# Manuscript change list: PNN_main.tex against current artifacts

Every concrete claim in `STEC_Modelling/PNN_main.tex` (read from disk 2026-09-14, 557 lines),
checked against the artifact that should back it today. `PNN_main.tex` itself is **not edited
by this document** — the owner revises it by hand (see the manuscript-freeze memory note); this
is the map they work from. `response_to_reviewers.md` and `evidence_summary.md` are also not
edited here, but were read for context and are cited where they already carry a relevant number.

**Method.** Every artifact cited below was re-read or recomputed on 2026-09-14, against the repo
state after commit `c83643b` (tree clean, `python -m stec.pipeline status` reports 0 of 39
stages stale, `python -m pytest tests/ -q --collect-only` collects 1,148 tests). Where a number
required aggregation not already sitting in a CSV (year-by-year pretrained-model stats, the
elevation-binned RMSE table, the daily STEC improvement distribution, the unfiltered positioning
percentile table), it was recomputed directly from the underlying parquet/CSV in this pass —
methodology noted inline so it can be re-run.

**Status vocabulary**: UNCHANGED (artifact reproduces the manuscript number, possibly with
trivial rounding drift), VALUE MOVED (same claim shape, different number), CLAIM NO LONGER
SUPPORTED (the artifact contradicts or cannot support what the sentence asserts), NEW CLAIM
AVAILABLE (evidence exists with no manuscript sentence yet).

**Every "Artifact said (2026-09-14 snapshot)" cell below is exactly that — a snapshot, dated
in the column header, not a live value.** This document's whole purpose is comparing the
manuscript's frozen number against a current one, so both numbers have to be written down
side by side; that is different from CLAUDE.md's canonical-results table, which holds no
numbers at all any more, only paths. Before pasting anything from the "Artifact said" column
into the manuscript, re-read the CSV or stage named in the row (usually the "Population /
statistic" column) — if the pipeline has run again since 2026-09-14, the CSV may already say
something else. Numbers here are pinned to a commit (`c83643b`) precisely so this table is
reproducible, not so it is a substitute for re-checking.

---

## 1. Abstract and Plain Language Summary

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Population / statistic | Status | Action |
|---|---|---|---|---|---|---|
| 1 | Abstract, l.72 | "mean RMSE of 6.92 TECU and a MAE of 3.88 TECU" | RMSE 6.9243, MAE 3.8803 | `daily_metrics/rebuilt/summary.csv`, `own_vtec_gim`/Direct STEC, mean of 242 daily RMSE/MAE | UNCHANGED | none |
| 2 | Abstract, l.72 | "outperforming both a daily ML VTEC baseline... and IGS GIM" | Direct STEC 6.92 < VTEC 8.96 < IGS GIM 8.28 | same table | UNCHANGED (ranking holds; GIM's own value moved, see Table 3 row 25 below) | none |
| 3 | Abstract, l.72 | "largest gains occur at low elevation angles" | 5° bin: Direct STEC 10.74 TECU vs GIM 13.34, VTEC 15.19; 85° bin: 3.63 vs 3.96 vs 3.69 (converges) | `elevation_metrics_finetuned/rebuilt/per_day_by_elevation.csv`, `own` dataset, n-weighted mean RMSE per 5° bin, recomputed this pass | UNCHANGED | none |
| 4 | Abstract, l.72 | "Predicted uncertainties show a monotonic relationship with observed errors" | MAE rises monotonically 1.27→4.36+ TECU across predicted-uncertainty bins 0–1→8–9+ | `uncertainty_error_relation/rebuilt/by_uncertainty.csv` | UNCHANGED | none |
| 5 | Abstract, l.73 | "average improvement of 30% in 3D RMS positioning error relative to IGS GIMs with mapping" | Mean improvement 3.8% (10 m rule, N=10,647/10,837, current population); unfiltered mean −1.2%; median improvement 18.1% (unfiltered, N=10,717/10,853) | `positioning_summary/rebuilt/overall.csv` + `positioning_diagnostics/rebuilt/outlier_headline_sensitivity.csv`, recomputed unfiltered percentiles this pass directly from `positioning_coverage/rebuilt/multiday_summary.csv` | **CLAIM NO LONGER SUPPORTED** | Replace with the median-based figure (~18%) plus the population-split caveat (§3 below), per `docs/revision/positioning_reporting.md`. Do not requote 30% or any single mean without the population split next to it. |
| 6 | Plain Language Summary, l.79 | "reduce positioning errors by about 30 percent on average" | same as #5 | same | **CLAIM NO LONGER SUPPORTED** | Same fix as #5, in plain language: "typically better by about a fifth" or similar median-based phrasing. |
| 7 | Plain Language Summary, l.79 | "improves ionospheric delay predictions... especially for low elevation satellites" | confirmed, #3 | same | UNCHANGED | none |
| 8 | Key Points, l.59 | "Direct STEC modeling improves accuracy especially at low elevation angles and improves GNSS positioning accuracy" | accuracy claim holds (#3); positioning claim holds only as a median/typical-day statement, not as a mean/"30%" statement | — | VALUE MOVED (qualify, don't drop) | Reword to avoid implying a single large average number; "typically improves positioning accuracy" survives, "by 30%" does not. |

## 2. Data and Methods sections

No numeric claim in the Data section (§`secData`, l.149–171: 2014–2024 span, ~70/15/15%
station split) depends on an artifact that has moved — the splits are static files
(`stec/data/splits/*.list`) untouched by the rebuild. No action.

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Status | Action |
|---|---|---|---|---|---|
| 9 | Table 1, l.187–227 | Input feature list | Unchanged — `stec.config.feature_registry` matches | UNCHANGED | none |
| 10 | Table 2, l.260–289 | Hyperparameters; β=0.1, no warmup schedule shown | β anneals **linearly 0→0.1 over 5 warmup epochs** (`stec/training/loss.py`'s `KLWarmupSchedule`, ported from `src/training/training_utils.py:45`) — table shows only the terminal value | Table is not wrong, but incomplete | **NEW CLAIM AVAILABLE** (methods clarification) | Add a row or footnote: "β annealed linearly from 0 over the first 5 epochs." Cheap, prevents a reviewer catching the omission independently. |
| 11 | Methods, implicit (no explicit search claim in the text) | Residual blocks = 4 (Table 2), presented as a fixed architectural choice | `num_layers` was never varied across the paper model's own 711 wandb runs (`response_to_reviewers.md` R2.5 table: 711 runs, one architecture) | — | **NEW CLAIM AVAILABLE** | If the methods text is edited at all for R2.5, add one sentence stating depth was fixed by design, not searched — avoids an implication no evidence supports. |

## 3. Results §4.1 — Pretrained model on the full multi-year test set

Recomputed directly from `pretrained_test_diagnostics/rebuilt/observations.parquet` (10,000,000
rows, `stec.analysis.pretrained_test_diagnostics`'s cache; written 2026-08-24, unaffected by the
positioning recovery sweep — this cache is drawn from `predictions/pretrained_stec/own`, a
different partition from anything the positioning work touches):

```python
import pandas as pd, numpy as np
df = pd.read_parquet(".../observations.parquet", columns=["true_stec","stec_pred","year"])
```

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Status | Action |
|---|---|---|---|---|---|
| 12 | l.326 | "Pearson correlation of approximately 0.95 and R² close to 0.9" | Pearson 0.9475, R² 0.8974 (10M-row cache, all years) | UNCHANGED | none |
| 13 | l.334 | Errors decrease monotonically with elevation; pretrained model stable at low elevation | Consistent with Gate F `mapping_function_consistency`/`stratified_comparison` MATCH against the pre-rebuild predecessor; not independently re-binned by elevation for the pretrained-only model in this pass | UNCHANGED (verified via Gate F, not re-derived here) | none |
| 14 | l.343 | Errors increase near magnetic equator and high latitude; sparse coverage at −80°/−60° | Same basis as #13 | UNCHANGED (verified via Gate F) | none |
| 15 | l.351 | Diurnal MAE/RMSE pattern, small day/night gap | Same basis as #13 | UNCHANGED (verified via Gate F) | none |
| 16 | l.359 | "2018–2020... MAE 2.6 to 3.0 TECU... RMSE 3.8 to 4.5 TECU" | 2018: MAE 3.03/RMSE 4.52; 2019: 2.95/4.31; 2020: 2.55/3.81 | UNCHANGED (matches to first decimal, 2018 MAE nudges the stated upper bound) | none |
| 17 | l.359 | "In 2023 MAE rises to 8.0... RMSE 11.8... in 2024... 9.2... 14.0" | 2023: MAE 7.98/RMSE 11.85; 2024: MAE 9.20/RMSE 14.05 | UNCHANGED | none |
| 18 | l.359 | "Pearson correlation of 0.93–0.95 across all years" | Per-year range is 0.908 (2017) to 0.959 (2022); five of eleven years fall outside 0.93–0.95 (2014 0.928, 2015 0.916, 2017 0.908, 2022 0.959) | VALUE MOVED (minor) | Widen the stated range to ~0.91–0.96, or state the overall-pooled figure (0.9475) instead of a per-year range. Low priority — doesn't change the conclusion ("high correlation throughout"). |

## 4. Results §4.2 — Uncertainty calibration

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Status | Action |
|---|---|---|---|---|---|
| 19 | l.369 | Monotonic relationship, uncertainty tracks error magnitude | Confirmed, #4 above | UNCHANGED | none |
| 20 | l.391 | Aleatoric dominates, epistemic comparatively small | Epistemic share 5.1–6.6% of total across uncertainty bins (`uncertainty_error_relation/rebuilt/by_elevation.csv`'s `epistemic_share_%` column) | UNCHANGED | none |

## 5. Results §4.3 — Fine-tuning, baselines, Tables 3 & 4

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Population | Status | Action |
|---|---|---|---|---|---|---|
| 21 | l.395, Fig 10 | "gains frequently exceeding 15–30%" over both baselines | vs VTEC: median 23.0%, 83.9% of days >15%, 14.5% of days >30%. vs GIM: median 18.3%, 63.6% of days >15%, 2.1% of days >30% | `daily_metrics/rebuilt/per_day.csv`, own dataset, per-day RMSE, recomputed this pass, 242 days | UNCHANGED | none |
| 22 | l.403, Fig 11 | Direct STEC beats both baselines at low elevation, converges at high elevation | 5°: 10.74 (STEC) vs 13.34 (GIM) vs 15.19 (VTEC). 85°: 3.63 vs 3.96 vs 3.69 | `elevation_metrics_finetuned/rebuilt/per_day_by_elevation.csv`, own, recomputed this pass | UNCHANGED | none |
| 23 | Table 3, l.421, Direct STEC row | 6.92±1.14 / 3.88±0.49 / 0.97±0.01 | 6.9243±1.1436 / 3.8803±0.4854 / 0.9673±0.0089 | 242 days | UNCHANGED (exact) | none |
| 24 | Table 3, l.422, Pretrained row | 13.45±4.84 / 9.36±3.86 / 0.87±0.12 | 13.4464±4.8350 / 9.3657±3.8646 / 0.8677±0.1248 | 242 days | UNCHANGED (exact) | none |
| 25 | Table 3, l.423, VTEC row | 8.96±1.47 / 5.21±0.71 / 0.95±0.01 | 8.9636±1.4710 / 5.2125±0.7149 / 0.9461±0.0102 | 242 days | UNCHANGED (exact) | none |
| 26 | Table 3, l.424, IGS GIM row | **8.56±1.86 / 5.52±1.45 / 0.95±0.03** | **8.2826±0.9905 / 5.3008±0.6296 / 0.9534±0.0084** | 242 days | **VALUE MOVED** — the DOY-truncation GIM repair (`docs/revision/response_to_reviewers.md`'s "Correction to the published IGS GIM baseline"; CLAUDE.md Gotchas) | Replace all three GIM cells. Mean, MAE and spread all shrink — the correction makes GIM *more* consistent, not less, so the qualitative "Direct STEC wins" conclusion is unaffected, but every digit in this row is wrong as printed. Already flagged in `response_to_reviewers.md`, not yet applied to the manuscript table. |
| 27 | l.430 | "VTEC + Mapping achieves the lowest RMSE and MAE... while Direct STEC remains competitive and outperforms both Pretrained and IGS GIM" | Ordering: VTEC 13.60 < Direct STEC 14.63 < IGS GIM 15.47 < Pretrained 17.29 | Madrigal, 238 days | UNCHANGED (ordering preserved even after GIM correction) | none |
| 28 | Table 4, l.439, Direct STEC row | 14.70±3.44 / 8.85±1.92 / 0.85±0.03 | 14.6329±3.4256 / 8.7943±1.8959 / 0.8535±0.0349 | 238 of 242 days (DOY 199–202 genuinely absent from Madrigal on this host) | UNCHANGED (near-exact) | none |
| 29 | Table 4, l.440, Pretrained row | 17.37±4.78 / 11.83±3.81 / 0.79±0.10 | 17.2875±4.7822 / 11.7762±3.8061 / 0.7892±0.1034 | 238 days, **now backed by a complete, freshly-generated partition** (`predictions/pretrained_stec/madrigal`, 241/242 files; merged 444.8M rows across 236 days, commit `0d59f00`) rather than the single orphan day this row used to rest on | UNCHANGED, and strengthened | none — but worth a footnote that this row now has real, complete provenance for the first time since the rebuild |
| 30 | Table 4, l.441, VTEC row | 13.60±2.96 / 8.27±1.71 / 0.87±0.02 | 13.6013±2.9568 / 8.2708±1.7113 / 0.8742±0.0209 | 238 days | UNCHANGED (exact) | none |
| 31 | Table 4, l.442, IGS GIM row | **15.64±3.12 / 10.55±2.12 / 0.83±0.04** | **15.4749±2.9078 / 10.3988±1.8463 / 0.8354±0.0252** | 238 days | **VALUE MOVED** — same GIM repair as row 26 | Replace all three GIM cells; same direction (error and spread both shrink). |

## 6. Results §4.4 — GNSS positioning (the section with the largest changes)

All positioning artifacts below use **iono** (predicted-uncertainty) weighting unless stated.
Population sizes moved twice more since the recovery-sweep numbers CLAUDE.md's canonical table
was last updated against: coverage grew from 10,598 solved-by-all (2026-08-27) to **10,712**
(2026-09-14, commit `68a6cb8`), and Direct STEC's own row count grew by 114 station-days between
the `positioning_distributions`/`positioning_diagnostics` prose docs (dated 2026-08-28) and today
— see §9 (corrected later 2026-09-14: both modules are now declared pipeline stages and their
docs have been refreshed, so this is no longer an open gap, only a record of what moved).

| # | Location | Manuscript says | Artifact said (2026-09-14 snapshot) | Population | Status | Action |
|---|---|---|---|---|---|---|
| 32 | l.451 (\add) | Weighting ablation: Direct STEC 3D RMS 1.156→1.121 m switching elev→iono, "an improvement of 3.0%", restricted to 27,205 station-days solved under both schemes | Direct STEC elev 1.5966→iono 1.5367 m, **+3.75%**, N=10,366 paired station-days (self-comparison); cross-check via `common_set_positioning` (all 4 methods × both weightings, N=10,186): elev 1.5442→iono 1.4848, +3.85% | `weighting_ablation/rebuilt/paired.csv`; `common_set_positioning/rebuilt/table5_common_set.csv` | **VALUE MOVED** — direction and rough magnitude (~3–4%) hold, but N dropped from 27,205 to ~10,200–10,400 and both absolute values roughly tripled (harder, larger recovered population) | Requote N and the two absolute values. The "27,205" figure's exact construction (sum across methods? a different N convention?) could not be reproduced from any single current artifact — flag for the owner rather than guess which one it matches. |
| 33 | l.451 (\add) | VTEC + Mapping degrades 1.580→1.624 m under iono weighting | VTEC elev 1.9095→iono 1.9105 m, **−0.05%** (still degrades, far smaller) | N=10,640 paired | VALUE MOVED (direction holds, magnitude much smaller) | Requote |
| 34 | l.451 (\add) | IGS GIM + Mapping "unchanged at 1.630" | GIM elev 1.6270→iono 1.6284 m, **−0.08%** | N=10,733 paired | UNCHANGED (both read as "essentially flat") | none, or requote for precision |
| 35 | Fig 12 caption, l.456; Fig 14 caption, l.512 | "Extreme outlier stations with errors higher than 10 m are excluded" | This is the **outcome-based** exclusion `docs/revision/positioning_reporting.md` argues against: it drops 4.2× more Direct STEC station-days than GIM ones (68 vs 16 at the 10 m line), which is not neutral between methods being compared | `positioning_diagnostics/rebuilt/outlier_threshold_counts.csv` | **CLAIM/METHOD NO LONGER SUPPORTED** | Regenerate Figs 12–15 without the 10 m station-day exclusion; report exceedance rates in the caption instead of hiding the tail. This is a plotting-methodology change, not just a number. |
| 36 | l.451, l.460 | Direct STEC "consistently yields lower 3D RMS errors on most days"; Pretrained "higher and more variable" | Still true directionally: Pretrained Direct STEC mean 2.30 m / median 1.75 m is clearly the worst of the four; Direct STEC leads on median (0.887 m, 10 m rule) | `positioning_summary/rebuilt/overall.csv` | UNCHANGED | none |
| 37 | l.460 | "the gains of direct STEC modelling are robust and not driven by a small subset of favorable days" | The opposite nuance is now demonstrated: the **mean** gain is non-monotonic in the exclusion threshold (−1.2% unfiltered → +16.0% at 5 m → +3.8% at 10 m → −0.2% at 50 m — `positioning_diagnostics/rebuilt/outlier_headline_sensitivity.csv`), and Direct STEC's own worst 0.65% of station-days (70 of 10,717) carry ~30% of its total summed error. The **median** is the robust statistic (18.1–20.5% across every threshold including none), not the mean this sentence is defending. | same | **CLAIM NO LONGER SUPPORTED as written** | Rewrite around the median's stability, not the mean's; state the threshold-sensitivity explicitly per `positioning_reporting.md` §1. |
| 38 | Table 5, l.480–483 | Pretrained 1.96/1.59; Direct STEC 1.12/0.77; VTEC 1.63/1.23; IGS GIM 1.63/1.09 (3D mean/median, m) | 10 m rule, current population: Pretrained 2.3049/1.7482 (N=10,704); Direct STEC 1.5576/0.8870 (N=10,647); VTEC 1.9104/1.3068 (N=10,746); IGS GIM 1.6198/1.0877 (N=10,837). Unfiltered (no exclusion at all): Direct STEC median 0.892 m, IQR [0.577, 1.957], P95 5.289, P99 8.949, N=10,717; IGS GIM median 1.089 m, IQR [0.768, 2.188], P95 4.128, P99 5.804, N=10,853 | `positioning_summary/rebuilt/overall.csv` (10 m rule, superseded methodology); unfiltered percentiles now from `positioning_distributions/rebuilt/TABLE5_NUMBERS.md`, the declared stage's own output (`canonical_for="Table 5"`, corrected 2026-09-14 — see §9) | **CLAIM NO LONGER SUPPORTED** (magnitudes, not just precision) | Replace the whole table per `docs/revision/positioning_reporting.md` §2's replacement content — median, IQR, P95/P99, exceedance rates, no outcome filter. `TABLE5_NUMBERS.md` is now current (regenerated 2026-09-14 by the declared `positioning_distributions` stage against the population this row already quotes) and can be pasted in directly; see §9 for the correction to this row's earlier claim that it could not. |
| 39 | l.451, l.500, l.73 (abstract) | "average improvement of approximately 30% in 3D RMS relative to the IGS GIM-based mapping approach" | Same as #5/#38 | — | **CLAIM NO LONGER SUPPORTED** | Single fix propagates to all three restatements (abstract, results, conclusion) |
| 40 | l.488, appendix l.516 | Pretrained model's worst positioning excursions coincide with Dst < −300 nT storms around DOY 132–133 and 282–285 | Confirmed: DOY 131–133 (Dst_min −406 nT) and 282–285 (Dst_min −333 nT) are the two 2024 storms; Pretrained's daily mean hits 6.8 m on DOY 285 while every other method stays under 3.2 m | `positioning_diagnostics/rebuilt/daily_timeseries.csv` §1 of `FINDINGS.md` (the DOY/Dst identification itself, unlike the population-level percentages nearby, does not depend on the coverage-recovery population and is not stale) | UNCHANGED | none |

## 7. Conclusion (§secConclusion, l.490–503)

| # | Location | Manuscript says | Status | Action |
|---|---|---|---|---|
| 41 | l.498 | "replacing standard elevation-dependent weighting with predicted-uncertainty weighting improves the 3D RMS error by approximately 3%" | VALUE MOVED (minor) — current ~3.8%, same as #32 | Requote alongside #32's fix |
| 42 | l.500 | "~30% relative to IGS GIM" | CLAIM NO LONGER SUPPORTED, duplicate of #5/#39 | Single fix |
| 43 | l.500 | "systematically lower 3D RMS errors than mapping-based solutions over most of the test period" | VALUE MOVED / needs a qualifier — true for the median and for most of the distribution's body, not true at P95/P99 where GIM is lower (§6 row 38) | Add the tail caveat: "except at the extreme tail, where IGS GIM is more consistent (§6, new finding)" |

## 8. Open Research Section (l.532–539)

No numeric claims; data-availability statements are unaffected by anything in the rebuild.
No action.

---

## What the claims-that-are-gone add up to

Only one number drives essentially every "no longer supported" row above: **the abstract's 30%
positioning figure and its five verbatim or near-verbatim restatements** (abstract l.73, plain
language summary l.79, results l.451/l.500, and implicitly Table 5 itself). All five trace to
the same population change — the station-recovery sweep more than doubled the positioning
coverage (35,652 → 43,215+ rows across four methods) and the added station-days are
systematically the ones where Direct STEC has no local calibration data (§3 of
`positioning_reporting.md`; the population crossover, restated in §3 below). This is a single
fix propagating through five locations, not five independent problems.

A second, independent claim is also gone: **that the positioning gain is "robust, not driven by
a small subset of favorable days"** (l.460). The evidence now shows close to the opposite — the
mean-based headline is the fragile statistic, and the median is the robust one. This is a
methodology correction (which statistic to trust), not a population correction, and needs its
own sentence, not just new numbers plugged into the old one.

A third, narrower claim needs dropping or heavily qualifying: **Figure 12/13/14's 10 m
station-day exclusion**. It is not neutral between the methods being compared (§6 row 35) and
should not be presented as an innocuous outlier trim.

## New claims now available

None of these have a manuscript sentence yet. Recommended location and rough content:

1. **The operating envelope (best where the network has data, worse where it does not).**
   Belongs in §4.4, replacing the single pooled headline. Direct STEC's median positioning
   error is 0.776 m on station-days with a real STEC-database observation (N=8,442) against
   1.406 m for GIM (+19–24% depending on mean/median) — and 2.754–3.199 m on geometry-only
   recovered station-days (N≈2,100–2,300) against GIM's 1.988–2.426 m (Direct STEC **loses** by
   roughly 31–38%). Source: `positioning_diagnostics/rebuilt/population_split_stec_vs_gim.csv`
   (current numbers: original +19.4%/+23.5% mean/median, recovered −31.1%/−36.8% — see §9 for why
   these are slightly different from the FINDINGS.md prose's −31.9%/−38.5%). This is the
   single most important new result: it explains *why* the pooled number moved, and it is a
   genuinely stronger, more specific claim than "30% better on average" ever was.
2. **The tail asymmetry.** Direct STEC is typically better (wins on median, IQR, and the bulk of
   the distribution) but IGS GIM is more consistent at the extreme: P95 4.13 m (GIM) vs 5.29 m
   (Direct STEC), P99 5.80 m vs 8.95 m. Belongs beside the new Table 5 and in the discussion —
   framed as "typically better, occasionally worse by more," not hidden by an outlier filter.
   Source: recomputed this pass from `positioning_coverage/rebuilt/multiday_summary.csv`.
3. **The attribution result.** A station's STEC prediction accuracy strongly predicts its
   *absolute* Direct STEC positioning error (Spearman ρ=0.82, p=2.4e-12, N=46 reliable
   stations) but not whether Direct STEC *beats GIM* there (ρ=0.14, p=0.36, not significant).
   Belongs in the discussion, right where the per-station comparison is currently framed as "some
   stations do worse" — the correct framing is "STEC accuracy explains how good the model is, not
   whether it beats the incumbent, because GIM degrades at the same hard stations." Source:
   `positioning_geography/rebuilt/stec_positioning_correlations.csv`. **Correcting this row's
   earlier claim**: this file is written by `positioning_model_attribution.py`, not
   `positioning_geography.py` (they only share an output directory), and checked directly
   2026-09-14 it still carries an Aug 28 mtime — it was **not** rerun against the current
   population, unlike `positioning_geography.py`'s own ten files, which were (see §9). Treat
   these two ρ values as provisional until that module is declared and rerun too.
4. **The completed Madrigal row.** Table 4's Pretrained row is no longer resting on a single
   orphan day — it now has the same 238-day provenance as the other three rows (§5 row 29). Worth
   a one-line footnote or methods-section mention, since a careful reader could otherwise wonder
   why this row alone looked thin in an earlier version.
5. **Computational cost.** Not in the manuscript body at all today (only in the response letter,
   R2.8h). If reviewers pushed on cost (R2.8h did), consider a short methods-section addition:
   pretraining ≈6.25 GPU-hours (150 epochs, I/O-bound, ~7% GPU utilisation because every epoch
   draws 500,000 rows with replacement from the 103 GB training set), daily fine-tuning 15.4
   GPU-hours total over 242 days (median 9.0 s/epoch, 3.6 min/day), inference 8,605
   observations/s at T=100. Source: `computational_cost/rebuilt/cost_summary.csv`.
6. **The oracle floor, if R1.8 gets resolved before submission.** `oracle_benchmark/rebuilt/
   summary.csv` currently reads oracle floor 0.125 m against Direct STEC 1.219 m, VTEC 1.622 m,
   IGS GIM 1.408 m (N=5,514, elevation weighting, all-four-methods intersection) — Direct STEC is
   9.7× the floor against GIM's 11.2×. This is a **third** population for the same comparison,
   different again from the two (N=1,810 and N=5,364) `response_to_reviewers.md`'s R1.8 section
   already flags as disagreeing and unreconciled. Do not add this to the manuscript before that
   reconciliation happens — see the Limitations section below.

## What the reviewers asked that is now answerable differently

- **R2.3 (random station split, proximity to training data).** The response letter already has
  a STEC-domain answer (normalised error 11.1% within 100 km rising to 19.5% beyond 1,000 km,
  Spearman +0.395, N=55 test stations, non-monotonic near end — `station_independence/rebuilt/
  by_distance_bin.csv`). There is now a **second, independent line of evidence in the positioning
  domain**: distance-to-nearest-training-station correlates with a station's STEC RMSE (ρ=0.345,
  p=0.019) and, more weakly, with its absolute Direct STEC positioning error (ρ=0.253, p=0.09)
  and its positioning diff against GIM (ρ=0.294, p=0.047), N=46 reliable stations
  (`positioning_geography/rebuilt/stec_positioning_correlations.csv`). Two independently
  constructed analyses, in two different domains, pointing the same direction is a materially
  stronger answer to R2.3 than either alone — this was not available when the response letter
  section was drafted and should be added to it.
- **R1.7 (storm-time performance).** The mean-based storm/quiet comparison has collapsed to
  noise (Direct STEC vs GIM: quiet +5.2%, storm **−2.7%**, `storm_stratification/rebuilt/
  improvement_over_gim.csv` — Direct STEC now loses to GIM on storm days *by the mean*). The
  **median**-based, unfiltered view recovers a real, physically sensible answer instead: every
  method's storm-day median exceeds its quiet-day median, and Direct STEC's degrades somewhat
  more than GIM's (quiet 0.862 m → storm 1.059 m, +22.8%, against GIM's 1.065 m → 1.234 m,
  +15.9% — `positioning_reporting.md` §4). This directly supersedes the mean-based −2.7%/−0.3%
  figures currently circulating and should be the number that answers R1.7, not the mean.
- **R1.5 (stochastic-model ablation).** Directionally unchanged (Direct STEC benefits from
  uncertainty weighting, the mapped baselines don't) but every number needs replacing — see §6
  rows 32–34.
- **R1.8 (observation-derived upper bound).** Still not answerable with a single number — see
  Limitations below. Worth noting for the response letter that the disagreement has grown a
  third data point, not shrunk.

## Limitations that must be stated

1. **DOY 303, 338, 348 have no raw STEC data on this host at all** (confirmed `68a6cb8`, not
   merely a missing-checkpoint or missing-product issue as earlier framed) — permanently absent
   from every STEC-domain and positioning table in this study. State as a data-coverage
   limitation, not a to-do item.
2. **24 station-days across six DOYs (130, 143, 144, 148, 150, 151) have no observations in
   either the primary STEC database or the geometry-only recovery tree on this host** — genuinely
   unreachable, not an aggregation gap.
3. **26 station-days remain unreachable for a different reason**: a recurring set of ten stations
   (BRMG, LICC, KIR8, MAR7, UCLU, NAUS, NKLG, WUH2, UNSA, YKRO) have no SINEX ground-truth entry
   for the affected day, confirmed by grepping the SINEX file directly — not a pipeline defect.
   (This also resolves an earlier "BRMG/LICC have zero Direct STEC solutions, unexplained" note
   in `FINDINGS.md` — it is now explained.)
4. **Madrigal is a weaker measurement than `own` by construction**: no cycle-slip counter, so
   per-arc analysis on Madrigal is only ever a time-gap heuristic. Already stated in CLAUDE.md;
   should be stated in the manuscript wherever Madrigal dSTEC or per-arc claims are made (the
   current manuscript does not make per-arc Madrigal claims, so this is a forward-looking note
   for R1.3-adjacent additions, not a correction).
5. **Three of 242 per-day source files are truncated independent of the recovery sweep**: DOY
   166 and 176 dropped from ~43 to 2 stations simultaneously across all three ML method trees
   *and* the GIM arm (a PPPx-level failure, not an aggregation bug); DOY 323 dropped from ~41 to
   4 stations in the STEC tree only. Included in every table above as genuinely small samples,
   not backfilled. State as a known small-sample caveat for those three days if per-day figures
   are shown.
6. **Two positioning-adjacent product trees (Oracle, Fixed-Variance) have 158 of 242 days with
   all-dangling non-SINEX product symlinks** — existing results for those days are unaffected
   (already computed and on disk), but the days cannot be re-run from this host. Doesn't block
   any number in this document; flagged because nothing currently tests for it (§9).

## 9. A methodological note: `positioning_distributions`/`positioning_geography` are now
declared stages, and the docs behind this section were refreshed

**Corrected 2026-09-14 (later the same day this section was first written).** This section
used to read "the pipeline is current, but three hand-written docs lag it" and explained that
`positioning_distributions.py` and `positioning_geography.py` were not declared pipeline
stages, so nothing reran them when Direct STEC's row count grew by 114 station-days
(10,603 → 10,717, commit `68a6cb8`). That gap is closed: both modules (and their `stec/viz/`
figure counterparts) are now declared stages in `stec/pipeline/stages.py`, with
`positioning_distributions` carrying `canonical_for="Table 5"` (moved off `positioning_summary`,
which still implements the superseded mean/10 m-exclusion methodology — see
`SUPERSEDED_FOR_TABLE5_NOTE` in `stec/analysis/positioning_summary.py`). Both stages have been
run against the current, 43,215-row population, `python -m stec.pipeline status` reports 0 of
43 stages stale, and `verification/gate_f_figures.py` reports 10/10 MATCH.

Consequently, `TABLE5_NUMBERS.md` (regenerated by the `positioning_distributions` stage, not
hand-edited) and `docs/revision/positioning_reporting.md` (hand-updated against the
regenerated CSVs) are both current as of 2026-09-14 and safe to quote directly — the delta
table this section used to carry (mean/median improvement moving ~1 point in each direction) is
now recorded in `positioning_reporting.md`'s own top note and per-section deltas instead of
here, to avoid the same drift happening to two copies of the same table.

**What is still not current, corrected from this section's own earlier, incorrect claim**: the
previous version of this paragraph said the attribution and R2.3 numbers from
`positioning_geography` did not have the staleness problem because "that module's CSVs carry a
2026-09-14 mtime." Checked directly and found false at the time it was written — every file
under `multiday_results/analyses/positioning_geography/rebuilt/` carried an Aug 28 mtime, this
one included. It is now genuinely true for the ten files `positioning_geography.py` itself
writes (station maps, latitude stratification — all regenerated 2026-09-14 by the newly
declared stage), but **not** for `stec_positioning_correlations.csv`, the source of the
attribution finding (row 3 under "New claims now available" below): that file is written by
`positioning_model_attribution.py`, a different, still-undeclared module that merely shares
`positioning_geography`'s output directory and was not run this pass. Its Spearman ρ values
(0.82 / 0.14) are unchanged from 2026-08-28 because the file itself has not moved — treat them
as provisional, not re-verified against the current population, until that module is declared
too (see `docs/revision/positioning_reporting.md`'s own top note for the same caveat).
`positioning_quality_gate.py` has the identical gap (its quality-gate mean of +0.09%, cited
nowhere in this file directly but present in `positioning_reporting.md` §1, is also
2026-08-28-vintage).
