# Corrections ledger

**What this is.** A standing record of every quantity the rebuilt pipeline now reports
differently from what was previously reported, plus the results that did not exist before and
the ones that are still not safe to quote. It is deliberately *not* organised by manuscript
location or by reviewer comment: it is a statement about what the data currently says, which
outlives any particular draft or letter.

**As of 2026-08-27 11:00.** Every number below was read from its result CSV while writing this
file, not copied from another document. That rule exists because the prose in this repository
has demonstrably drifted from the artifacts: three separate documents still describe the
Madrigal comparison as covering 235 days when the current artifact covers 238.

**Work is in flight while this is being written**, and nothing here was produced by interrupting
it. The station-recovery geometry sweep finished successfully at 03:22 today, having reached
DOY 366; `overnight-chain-20260826.service` cleared its Phase-1 barrier at 03:26 and is now
several hours into Phase 2, rebuilding model predictions and positioning solutions over the
recovered station-days. §6 sets out what is still to come and what each step will change.

Consequently every entry carries one of three states:

- **Settled** — the STEC-domain numbers. Their inputs are complete and are not being rewritten.
- **Live** — every positioning number. The values below are true snapshots of the artifacts on
  disk, which still carry their 2026-08-24 18:23 timestamps because the re-analysis stage has not
  run yet. They *will* move when it does. Re-read before quoting.
- **Open** — §4, things that are missing, stale or unreproducible rather than merely provisional.

---

## 1. Numbers that changed

### 1.1 STEC accuracy, own 2024 test set — **settled**

Source: `multiday_results/analyses/daily_metrics/rebuilt/summary.csv` (242 days,
475,111,413 observations), regenerated 2026-08-26 14:35.

| Model | Previously reported RMSE | Current RMSE | Change |
|---|---|---|---|
| Direct STEC | 6.9245 | **6.9243** | none (−0.0001) |
| Pretrained Direct STEC | 13.4467 | **13.4464** | none (−0.0004) |
| VTEC + Mapping | 8.9636 | **8.9636** | none |
| IGS GIM + Mapping | 8.5552 | **8.2826** | **−0.2727 TECU** |

Only the GIM row moves, and the cause is a defect, not a modelling choice. Day-of-year reaches
the evaluation code as a *denormalised model input*, not as an integer read from a file: it is
normalised to `(doy − 1)/365` and inverted in float32, so 26 days of every year come back just
below the intended integer (DOY 189 inverts to 188.99998). The original aggregation truncated
with `int()` and therefore looked up **the previous day's** IONEX map on 12 of the 242 test days
— DOY 184–189 and 225–230. Rounding instead of truncating recovers the intended day. The GIM
baseline was inflated by the bug; correcting it makes the baseline *stronger*, so every
improvement quoted against it shrinks.

The prediction store itself was never wrong: `repair_gim_baseline` re-derived all 477 stored
days and found `repaired=False` throughout with maximum drift ~1e-5 TECU. The defect lived
purely in the old aggregation script.

### 1.2 STEC accuracy, Madrigal reference — **settled**

Source: same file, `madrigal_vtec_gim` rows (238 days, 448,938,780 observations).

| Model | Previously reported RMSE | Current RMSE | Change |
|---|---|---|---|
| Direct STEC | 14.6963 | **14.6329** | −0.0634 TECU |
| VTEC + Mapping | 13.6013 | **13.6013** | none |
| IGS GIM + Mapping | 15.6450 | **15.4749** | −0.1701 TECU |
| Pretrained Direct STEC | 17.37 | *not reproducible* | see §4 |

Two independent causes, and they act on different rows:

**The GIM row** moves for the same day-lookup reason as §1.1.

**The Direct STEC row** moves because of a second, unrelated erratum: the Madrigal predictions
were generated with `local_time_hours` derived from **receiver** longitude, while every other
dataset in this project derives it from **IPP** longitude. IPP is the physically correct choice —
the ionosphere's diurnal variation follows solar illumination at the pierce point, not at the
receiver — and `local_time_hours` is a real model input, so this changed the predictions, not
just their labelling. Measured on a real day through the real checkpoint with seeded weights and
a zero-perturbation control: mean shift +0.0015 TECU, RMSE 0.80 TECU, maximum |Δ| 13.4 TECU. The
235 affected days have since been re-inferred under the corrected convention, which is why this
table reads 238 days rather than the 235 that older documents describe.

**VTEC + Mapping is untouched by both**, and correctly so: its feature set has
`local_time_hours: false`, and it takes no IONEX lookup. That it reproduces to four decimal
places is a useful control — it shows the two corrections above are the specific mechanisms
claimed, not a general drift in the recomputation.

**The ordering on Madrigal is unchanged.** VTEC + Mapping (13.60) still has the lowest RMSE
against the Madrigal reference, ahead of Direct STEC (14.63). The correction narrowed the gap
from 1.10 to 1.03 TECU; it did not reverse it.

### 1.3 Positioning accuracy — **live, will move tonight**

Three different numbers exist, and they are not competing estimates of one quantity — they
answer different questions over different populations. Quoting one without its population is
the single easiest way to be wrong here.

| Basis | Direct STEC | IGS GIM + Mapping | Improvement | Populations |
|---|---|---|---|---|
| As previously reported (unmatched) | 1.1229 m | 1.6296 m | **31.1%** | 8,280 vs 10,882 station-days |
| Current full population (unmatched) | **1.2229 m** | **1.6184 m** | **24.4%** | 8,636 vs 10,837 station-days |
| Current common set (matched) | **1.1160 m** | **1.4007 m** | **20.3%** | 7,741 for both |

Sources: `analyses/positioning_summary/rebuilt/overall.csv` and
`analyses/common_set_positioning/rebuilt/table5_common_set.csv`.
The first row is re-derived from `analyses/positioning_summary/rebuilt/by_weighting.csv`, which
reads the frozen February 2026 tree; it reproduces the ~30.9% previously reported to within the
vintage difference of that tree, and is shown here so all three rows come from files on disk.

Two distinct effects, pulling the same way:

**The unmatched comparison is not like-for-like.** The IGS GIM arm solves ~2,200 more
station-days than the ML arms, because those stations are absent from the STEC database and so
can never receive an ML correction. Those extra station-days are disproportionately equatorial
and disproportionately hard. Restricting all four arms to the station-days every method solved
costs the GIM arm 3,182 station-days and moves its mean from 1.6184 to 1.4007 m — the GIM arm
was being penalised by a population the ML arms never had to attempt.

**Recovering more station-days made the unmatched comparison *less* favourable, not more.** The
recovery sweep added station-days that are harder than average, so Direct STEC's mean rose from
1.12 to 1.22 m. This is the expected direction and is not evidence of a regression.

**20.3% is the defensible apples-to-apples figure today.** 24.4% is what the original
methodology would now produce. Both are below the 30.9% previously reported, and both will
move again when the sweep and the overnight chain finish.

Supporting figures from the same common-set file, all at N=7,741: Direct STEC wins on 80.1% of
station-days; its paired-median gain over GIM is 27.0%. VTEC + Mapping (1.6034 m) and Pretrained
Direct STEC (1.9209 m) are both *worse* than GIM on the matched set, by 14.5% and 37.1%.

### 1.4 Storm and quiet behaviour — **live (positioning), settled (STEC)**

Positioning improvement over IGS GIM, from `analyses/storm_stratification/rebuilt/`:

| Regime | Previously reported | Current |
|---|---|---|
| Quiet | 31.9% | **25.4%** |
| Storm | 26.3% | **19.6%** |

Both fall for the reasons in §1.3 — the GIM baseline is stronger once repaired, and the
population changed.

**A claim was retracted here, not just a number corrected.** It was previously stated that
Direct STEC degrades *least* from quiet to storm conditions, against +9% for VTEC + Mapping and
+9% for IGS GIM. The artifact does not support this — and the two baselines coming out
identical was itself the symptom. Quiet-to-intense degradation in STEC RMSE
(`analyses/activity_stratification/rebuilt/by_dst.csv`, binned by daily minimum Dst):

| Model | Quiet (Dst > −30) | Intense (Dst ≤ −100) | Relative degradation |
|---|---|---|---|
| Direct STEC | 6.7795 | 8.0416 | **+18.6%** |
| VTEC + Mapping | 8.9470 | 9.5914 | +7.2% |
| IGS GIM + Mapping | 8.1410 | 9.0205 | +10.8% |
| Pretrained Direct STEC | 12.6919 | 24.2562 | +91.1% |

Direct STEC degrades relatively **more** than either operational baseline. What survives, and
is the stronger claim anyway: **Direct STEC has the lowest RMSE of all four methods in every
Dst bin**, quiet through intense. Both statements are true simultaneously precisely because it
starts from the lowest error — the same absolute rise off a smaller base is a larger percentage.

A second consequence of the GIM repair lands here: Direct STEC's margin over GIM now *narrows*
with storm intensity (16.7% quiet → 16.9% weak → 14.3% moderate → 10.9% intense) rather than
widening. Eight of the twelve mis-dated GIM days fell in the quiet bin, which is why that row
moved most.

### 1.5 Weighting ablation — **settled** (reads a frozen February 2026 tree)

Source: `analyses/weighting_ablation/rebuilt/paired.csv`.

| Correction | Elevation weighting | Predicted-uncertainty weighting | Gain | Paired station-days |
|---|---|---|---|---|
| Direct STEC | 1.1558 m | **1.1206 m** | **+3.05%** | 8,170 |
| VTEC + Mapping | 1.5805 m | 1.6238 m | −2.74% | 8,173 |
| IGS GIM + Mapping | 1.6296 m | 1.6311 m | −0.09% | 10,862 |

Two wording problems rather than numeric ones:

- **"IGS GIM + Mapping is unchanged at 1.630 m"** overstates. The values are 1.6296 and 1.6311 —
  a real, if negligible, −0.09% degradation, not an absence of one.
- **"the 27,205 station-days solved under both schemes"** reads as one common set. It is not:
  8,170 + 8,173 + 10,862 = 27,205 exactly. It is the *sum of three method-specific pairings of
  unequal size*. The arithmetic is right and the conclusion is unaffected, but the sentence
  describes a set that does not exist.

Note this stage reads one frozen tree dated February 2026 for both arms, so it is internally
consistent but older than everything in §1.3.

### 1.6 Uncertainty calibration — **settled for the own test set**

Source: `analyses/uncertainty_calibration/rebuilt/finetuned_stec_own/`, 475,111,413 observations.

The VTEC baseline is trained with a Laplacian NLL, so its predictive distribution is a Laplace.
The original evaluation scored every model's uncertainty as Gaussian. Scoring it under the
wrong family is worth roughly eight points of apparent coverage:

| Model | Nominal 50% | Empirical | Family |
|---|---|---|---|
| Direct STEC | 50% | **49.94%** | Gaussian (native) |
| VTEC + Mapping | 50% | 89.44% | Gaussian (**mis-specified**) |
| VTEC + Mapping | 50% | **81.19%** | Laplace (native) |

CRPS and PIT-KS agree independently: scoring VTEC natively gives CRPS 5.534 against 6.585, and
PIT-KS distance 0.222 against 0.273. The rebuilt module always computes both families and tags
which is native, so the choice can no longer be made silently.

Direct STEC's 49.94% empirical coverage at nominal 50%, over 475 million observations, is the
single strongest calibration result in the set and did not change.

### 1.6b Uncertainty calibration, Madrigal reference — **refreshed 2026-08-27**

Source: `analyses/uncertainty_calibration/rebuilt/finetuned_stec_madrigal/`, regenerated
2026-08-27 11:37 over the full 238-day partition.

The refresh did not change any conclusion, but it made the comparison valid for the first time.
Previously Direct STEC was scored over 442,945,913 observations and VTEC + Mapping over
438,900,615 — **different populations**, because days lacking the VTEC uncertainty columns were
silently dropped from one product and not the other. Both now cover **448,938,780**, matching the
store and `daily_metrics` exactly.

| Model | Nominal 50% | Empirical | Predicted scale ÷ realised RMSE |
|---|---|---|---|
| Direct STEC (Gaussian, native) | 50% | **23.58%** | 0.287 |
| VTEC + Mapping (Gaussian, mis-specified) | 50% | 81.62% | 1.925 |
| VTEC + Mapping (Laplace, native) | 50% | **70.09%** | 1.925 |

**The contrast with the own test set is the result worth noting.** The same model, the same
uncertainty head, reads 49.94% empirical at nominal 50% against its own reference (§1.6) and
**23.58%** against Madrigal — severely under-dispersed, predicting a scale only 29% of the
realised error.

That is not straightforwardly a statement about the model's uncertainty. §3.3 establishes that
**45.3% of the Madrigal disagreement is a per-station constant reference offset**, which no
observation-level predictive distribution can anticipate. Under-coverage against a reference
carrying an unmodelled constant offset is the expected outcome, not evidence of overconfidence.
Read the two together or neither.

---

### 1.7 Computational cost — **settled**

Source: `analyses/computational_cost/rebuilt/`, measured on the actual hardware
(RTX 4070 Ti 12 GB, 24 cores) rather than scaled from a per-epoch estimate.

| Item | Previously reported | Measured |
|---|---|---|
| Pretraining, 150 epochs | 0.38 GPU-hours | **6.25 GPU-hours** |
| STEC daily fine-tune, total over 242 days | — | 15.39 GPU-hours (3.58 min/day median) |
| VTEC daily fine-tune, total over 169 days | — | 19.40 GPU-hours |
| Inference throughput at T=100 | — | 8,605 observations/s (4.7 min per evaluation day) |

The old pretraining figure was a scaled estimate and understated the real cost by roughly 16×.
Note the VTEC baseline costs *more* to train than the STEC model it is compared against, because
it is a ten-member deep ensemble.

### 1.8 Hyperparameters that were true of the run but absent from the description

Source: `analyses/paper_tables/rebuilt/table2_hyperparameters.csv`. Three real choices were
made in code and never written down:

| Parameter | Actual value |
|---|---|
| KL weight | **annealed linearly 0 → 0.1 over 5 warmup epochs**, then held (reported as a flat 0.1) |
| Variance floor | **1e-3**, added after the softplus to keep the NLL bounded below |
| Output bias initialisation | **15.5 TECU**, approximately the mean STEC |

The last of these is not cosmetic: omitting it from the fully-Bayesian variant (§3.4) accounted
for about half that variant's apparent accuracy gap.

### 1.9 Epistemic share — **settled**

Source: `analyses/uncertainty_error_relation/rebuilt/by_uncertainty.csv`. The original computed
the epistemic share as the **square of each bin's mean** uncertainties; the variance
decomposition it reports calls for the **mean of the squares**. The square-of-means form is
Jensen-biased and compresses the range toward the middle:

| Formula | Minimum across bins | Maximum across bins |
|---|---|---|
| Original (square of means) | 4.94% | 6.66% |
| Corrected (mean of squares) | **3.07%** | **16.39%** |

Two other things changed in this output at the same time — bin edges moved from the first day's
sigma deciles to fixed absolute-TECU bands, and the column became a fraction rather than a
percentage — so the range shift should not be attributed to the formula alone. The decile
problem was real on its own: edges derived from a single day held between 6.88% and 18.80% of
the full-year population rather than 10% each.

---

### 1.10 The named storm days are not all below −300 nT — **settled**

Measured 2026-08-27 by reading `data/omni_hourly_2010-2025.h5` directly (the `Dst-index,_nT`
column, hourly, `/2024/<DOY>`). This was previously listed as unverified; it is now verified, and
it does not confirm the claim.

| DOY | Minimum Dst | Below −300 nT? |
|---|---|---|
| 132 | **−406 nT** | yes |
| 133 | −150 nT | no |
| 282 | −148 nT | no |
| 283 | −83 nT | no |
| 284 | **−303 nT** | yes |
| 285 | **−333 nT** | yes |

Only **three of the six named days** reach below −300 nT. The phrasing "Dst < −300 nT around
DOY 132–133 and 282–285" reads as though all six do. DOY 283, at −83 nT, does not even reach the
−100 nT "intense" threshold used elsewhere in this ledger, and DOY 133 and 282 sit near −150 nT.

The defensible statement is that there were **two storm episodes, peaking at −406 nT on DOY 132
and −333 nT on DOY 285**, with the adjacent days materially weaker. The days remain correctly
identified as the disturbed periods; it is the uniform −300 nT characterisation that does not
hold.

---

## 2. Two fixes that changed no number, and one gap in the mechanism

The first two are recorded because "we checked and it changed nothing" is a different and more
useful statement than silence. The third is a defect in the provenance machinery itself, found
while refreshing §1.6b.

### 2.1 The 10 m outlier boundary

It was applied with a strict `<` in one analysis and `<=` in two
others, so a station-day at exactly 10.000 m was in two tables and out of the third. Unified to
`<=`. Across all 100,459 station-days in the three trees, **zero** sit exactly at the boundary;
the nearest approaches are 9.9624 m and 10.0236 m.

### 2.2 The subset-index cache ignored its own seed

The cache validated length and subset size but
never read back the `seed` field it stored, so changing a seed silently returned the previous
seed's subset. Fixed and version-gated. All 1,129 cached files carry seed 42, and every call
site passes 42, so no stored selection changes — but a future seed change is now a checked
divergence rather than a silent one.

---

### 2.3 An artifact that no stage owns — found 2026-08-27

Not a fix that changed nothing; a defect found while trying to refresh §1.6b, recorded because
it is a gap in the mechanism the rest of this ledger relies on.

`python -m stec.pipeline run --only uncertainty_calibration --force` completed in 469 s and left
the Madrigal calibration **untouched**. Both calibration stages hardcode `--dataset own`:

| Stage | Reads | Writes |
|---|---|---|
| `uncertainty_calibration` | `predictions/finetuned_stec/own` | `.../finetuned_stec_own/` |
| `uncertainty_calibration_pretrained` | `predictions/pretrained_stec/own` | `.../pretrained_stec_own/` |

`.../finetuned_stec_madrigal/` sits in the same output directory and **no stage declares it**. It
was written by a manual run on 2026-08-25 22:25 and nothing in the 37-stage pipeline regenerates
it, so it has no input fingerprint, no provenance record and no caveats sidecar. A forced re-run
reproduced its observation count byte-for-byte — which reads as confirmation and was in fact a
file that had not been rewritten.

This is the failure mode `CLAUDE.md` records for `positioning_coverage`, one step worse. There,
an artifact had an owner that silently stopped writing it. Here it never had an owner at all, and
the registry cannot catch that: `validate()` rejects *two* stages claiming one output, but nothing
detects an output claimed by *none*.

**Not fixed here.** The repair is a third declared stage for the Madrigal dataset, plus a sweep
for any other artifact under `multiday_results/analyses/` with no owning stage — both code
changes, deliberately left for a decision rather than made silently.

---

## 3. Results that did not exist before

These are not corrections. They are quantities the previous implementation could not produce,
now available from the prediction store.

### 3.1 dSTEC — the accuracy claim that does not depend on the DCB reference

`analyses/dstec_evaluation/rebuilt/summary.csv`, full 242-day store: **672,542 continuous arcs**,
arc boundaries taken from the database's own cycle-slip counter and truth from carrier phase.

| Quantity | Direct STEC | IGS GIM + Mapping |
|---|---|---|
| dSTEC RMSE, pooled | **5.155** | 6.637 TECU |
| dSTEC RMSE, mean of arcs | **3.746** | 5.368 TECU |
| Absolute STEC RMSE, pooled | 6.336 | 7.889 TECU |

This matters because it removes the per-arc constant — the part most sensitive to differential
code bias conventions — and the model's advantage survives.

### 3.2 The model's uncertainty against the GIM products' own published uncertainty

`analyses/ionex_rms_benchmark/`, quiet-day rows, 402,162,623 observations over 203 days.
Empirical coverage at nominal 50%, and CRPS skill against a constant-scale reference:

| Product | Coverage at nominal 50% | CRPS skill |
|---|---|---|
| Direct STEC | **50.3%** | **+0.103** |
| CODE GIM + Mapping | 31.7% | +0.014 |
| IGS GIM + Mapping | 18.2% | −0.077 |
| VTEC + Mapping | 80.8% | −0.367 |

The IONEX products' own RMS values are badly under-dispersed as predictive uncertainties, and
both GIMs score *worse* than a constant scale. This is a like-for-like comparison against the
operational products' own error bars, not against an assumed one.

### 3.3 How much of the Madrigal disagreement is a reference offset

`analyses/madrigal_reference_offset/`, 67 stations, 442,945,913 observations. **45.3% of the
variance is a per-station constant offset**: removing it drops RMSE from 15.045 to 11.130 TECU,
and the mean absolute per-station offset is 6.69 TECU. The model and the IGS GIM disagree with
Madrigal in the same direction at 95.5% of stations, Spearman ρ 0.698.

Read with its own caveat: the all-station Pearson r of 0.925 is leverage-inflated. Restricted to
stations with |offset| ≤ 15 TECU (n=53) it falls to 0.617, and to 0.365 at |offset| ≤ 10 (n=51).
The conclusion — that the Madrigal comparison confounds out-of-distribution model behaviour with
a different processing chain — holds under all four thresholds, but the correlation should be
quoted at a stated threshold, never as the unrestricted 0.925.

### 3.4 The fully-Bayesian variant, actually trained

The fully-Bayesian architecture had never been pretrained. It now has been, twice: the first
attempt omitted the output-layer initialisation of §1.8, and the pervasive −1.93 TECU bias was
the fingerprint. Retrained with the initialisation corrected:

| | Paper model (last-layer Bayesian) | Fully Bayesian, corrected |
|---|---|---|
| Test RMSE | **11.6716** | 15.5389 (was 19.7355 before the init fix) |
| Uncertainty–error correlation | 0.5682 | **0.5752** |

Fixing the initialisation closed 52% of the apparent gap. The remaining 48% is real: last-layer
is substantially more accurate. Two competing explanations were tested and refuted — the KL
weight (the fully-Bayesian share is *smaller*, 0.34% against 1.02%) and undertraining (best
epoch 91, with 20 further epochs producing nothing better).

The honest framing is that pooled RMSE is the wrong scoreboard for this question. Last-layer
Bayesian is expected to under-represent epistemic uncertainty; whether it gives *adequate*
epistemic uncertainty out of distribution is the question worth answering, and the two existing
out-of-distribution probes (Madrigal, station independence) answer it better than RMSE does.

### 3.5 Station independence

`analyses/station_independence/rebuilt/by_distance_bin.csv`, 55 test stations grouped by
distance to the nearest training station:

| Distance | Stations | RMSE | Normalised RMSE |
|---|---|---|---|
| < 100 km | 19 | 6.10 | 11.1% |
| 100–250 km | 8 | 4.36 | 7.8% |
| 250–500 km | 11 | 4.74 | 9.5% |
| 500–1000 km | 10 | 9.47 | 15.9% |
| > 1000 km | 7 | 9.52 | 19.5% |

Degradation with distance is real but not monotonic, and the result is limited by **55 stations**,
not by observation count — more days sharpen each point without strengthening the trend. Making
this claim stronger requires a region-held-out retrain.

### 3.6 The interpolation / extrapolation split is not what the raw RMSE suggests

`analyses/relative_error_metrics/rebuilt/temporal_regime_comparison.csv`:

| Regime | Observations | RMSE | Mean true STEC | Normalised RMSE |
|---|---|---|---|---|
| Before 2024-05-01 (interpolation) | 4,400,934 | 7.6504 | 24.71 | **31.0%** |
| From 2024-05-01 (extrapolation) | 5,599,066 | 14.0463 | 52.17 | **26.9%** |

Absolute RMSE nearly doubles, which reads as a large extrapolation penalty. Normalised by the
mean STEC of each period it *falls*, because mean STEC itself doubles. The degradation is
predominantly the ionosphere getting larger, not the model getting worse — and an
activity-matched comparison at equal F10.7 confirms it within the two overlapping bins.

### 3.7 Distance to an observation-derived floor

`analyses/oracle_benchmark/rebuilt/summary.csv`, 5,364 station-days, elevation weighting:

| Correction | 3D RMS | Ratio to floor |
|---|---|---|
| Reference STEC (oracle) | 0.1223 m | 1.0× |
| Direct STEC | 1.2260 m | **10.0×** |
| IGS GIM + Mapping | 1.4147 m | 11.6× |
| VTEC + Mapping | 1.6339 m | 13.4× |

Read ratios inside this table only. It uses elevation weighting by necessity — the reference
STEC carries a placeholder sigma, so uncertainty weighting would weight by a constant — and is
restricted to station-days solved by all four methods, so its absolute values are not comparable
with §1.3.

---

## 4. Not yet quotable

| Item | State | What would close it |
|---|---|---|
| Pretrained Direct STEC on Madrigal (17.37) | **Not reproducible.** The store partition holds 1 orphan day of ~242, and that day is schema-incomplete (27 of 37 columns). The published figure may also carry the §1.2 local-time erratum, in a direction that cannot be quantified from what is on disk. | ~42 GPU-hours of inference; staged as the overnight chain's Phase 5, off by default |
| All positioning numbers (§1.3, §1.4) | **Live.** The recovery sweep is adding station-days; the overnight chain re-runs the positioning stages afterwards. | Let both finish, then re-read |
| Madrigal uncertainty calibration | **Closed 2026-08-27** — see §1.6b. Was stale at 235 days / 442.9M against a 238-day / 448.9M store. **Correcting this row's own earlier claim: re-running the pipeline stage does not fix it** — no stage owns this artifact (see §2.3). It needs a direct module invocation with `--dataset madrigal`. | done |
| Madrigal DOY 199–202 | Genuinely absent — no Madrigal source data for those days on this host | Nothing local |
| Per-year Pearson r ("0.93–0.95 across all years") | **Unverifiable.** The per-year artifact carries RMSE/MAE/R²/count but no correlation. √R² spans 0.90–0.96, which brackets but does not confirm the narrower claim. | Compute per-year Pearson r from the same test predictions |
| DOY 303, 338, 348 | No checkpoint and no products anywhere on this host — not merely a missing download | Upstream data this host cannot reach |
| DOY 166, 176 (all trees) and DOY 323 (STEC only) | Per-day positioning files truncated on disk to 2–4 stations from ~41–43, from a solver-level failure. Included above as genuinely small samples, not dropped or backfilled | A targeted positioning re-run |

Two of these are worth separating from the rest, because they are *absences of evidence being
read as evidence*: the pretrained-Madrigal partition looks populated if you count files rather
than check schemas, and the Madrigal calibration looks current if you check that the stage ran
rather than what it ran over. Both were found by checking the artifact, not the status.

---

## 5. Where every number came from

All paths relative to `multiday_results/analyses/`. Each directory also carries a
`.caveats.json` sidecar stating the population it was computed over, and each stage writes
`.pipeline/<stage>.json` recording the commit, command and row counts that produced it.

| Section | File |
|---|---|
| §1.1, §1.2 | `daily_metrics/rebuilt/summary.csv` |
| §1.1 (repair check) | `repair_gim_baseline/pre_rebuild/gim_repair_report.csv` |
| §1.3 | `positioning_summary/rebuilt/overall.csv`, `common_set_positioning/rebuilt/table5_common_set.csv` |
| §1.4 | `storm_stratification/rebuilt/`, `activity_stratification/rebuilt/by_dst.csv` |
| §1.5 | `weighting_ablation/rebuilt/paired.csv` |
| §1.6 | `uncertainty_calibration/rebuilt/finetuned_stec_own/{coverage,scores}.csv` |
| §1.7 | `computational_cost/rebuilt/{cost_summary,training_cost}.csv` |
| §1.8 | `paper_tables/rebuilt/table2_hyperparameters.csv` |
| §1.9 | `uncertainty_error_relation/rebuilt/by_uncertainty.csv` |
| §3.1 | `dstec_evaluation/rebuilt/summary.csv` |
| §3.2 | `ionex_rms_benchmark/rebuilt/by_regime_{IGS,CODE}.csv` |
| §3.3 | `madrigal_reference_offset/pre_rebuild/{decomposition,leverage_check}.csv` |
| §3.5 | `station_independence/rebuilt/by_distance_bin.csv` |
| §3.6 | `relative_error_metrics/rebuilt/temporal_regime_comparison.csv` |
| §3.7 | `oracle_benchmark/rebuilt/summary.csv` |

## 6. What happens next

Ordered by dependency, not by importance. Each item says what runs, what it rewrites, and —
where the direction is predictable from what is already known — which way it will move.

### 6.1 In flight right now

| Step | State | Rewrites |
|---|---|---|
| Station-recovery geometry sweep | **done**, 03:22 today, reached DOY 366 | the recovered-observation database (242 days present) |
| Overnight chain Phase 2 — recovery models | **running**, since 03:26 | model predictions and PPPx solutions over recovered station-days |

Phase 2 had completed 135 days by 11:00 (DOY 122 → 280) at roughly 3.3 min/day, with **1,327
RINEX downloads and zero timeouts** — the wrapper-timeout defect that previously killed 1,491 of
1,491 downloads is not recurring. On that rate the remaining ~110 days finish around 17:00 today.

### 6.2 Immediately after, without further intervention

Phase 3 runs the positioning re-analysis (an unfiltered `pipeline run --keep-going`, with
per-stage success judged from `pipeline status` rather than the run's exit code), then Phase 4
regenerates figures and runs the figure-parity gate. Between them these rewrite **§1.3, §1.4 and
the station-day coverage breakdown**.

The chain recorded its own "before" at 03:26, so the comparison will be exact rather than
remembered: solved-by-all **8,195**, all-ML-missing **1,591**, some-ML-missing **1,067**, total
**10,853**.

**Predicted direction, stated now so it can be checked rather than rationalised afterwards.**
Recovery converts all-ML-missing station-days into solved-by-all or some-ML-missing, and the
station-days it adds are harder than average. So the *full-population* improvement (currently
24.4%) should fall further, while the *matched common-set* figure (currently 20.3%) should be
comparatively stable — the two should **converge**, because the gap between them is caused by
precisely the unmatched population that recovery is closing. If instead they diverge, something
in the recovery path needs investigating before the numbers are used.

### 6.3 Deliberately not running

Phase 5 — pretrained-model inference against the Madrigal reference, ~42 GPU-hours — is off by
default and must be enabled explicitly. Until it runs, the Pretrained row of §1.2 stays
unquotable and the published 17.37 has no counterpart. It writes one day at a time and re-checks
its own day list by reading parquet schemas rather than trusting file existence, so it is safe to
stop and safe to resume.

### 6.4 Cheap, unscheduled, and worth doing

None of these needs a sweep or a GPU. The third — confirming the storm-day Dst values — was
done on 2026-08-27 and is now §1.10; it corrected the claim rather than confirming it.

- **Re-run the Madrigal uncertainty calibration.** Currently stale at 235 days / 442.9M
  observations against a store that now holds 238 days / 448.9M, and it predates yesterday's
  backfill of the missing VTEC uncertainty columns on DOY 196/217.
- **Compute per-year Pearson r** from the same test predictions the per-year file was built from,
  which is the only thing standing between the "0.93–0.95 across all years" claim and evidence.

### 6.5 Blocked on data this host does not have

DOY 303, 338 and 348 have neither checkpoint nor products anywhere locally, and Madrigal DOY
199–202 have no source data. DOY 166, 176 and 323 need a targeted positioning re-run to repair
per-day files truncated by a solver failure — possible here, but not part of any queued work.

### 6.6 What this document deliberately does not do

Nothing here has been written into the manuscript, and this ledger is not a manuscript-editing
checklist. It records what the data currently supports. Deciding which of these corrections to
apply, and how to word them, is a separate step that should happen once §6.1 and §6.2 have landed
and the positioning numbers have stopped moving.

---

Related records, kept separate on purpose: `divergences.md` is the machine-backed register of
*deliberate* divergences and their measured effects; `coverage_recovery_status.md` explains why
the positioning population is still moving. This ledger is the human-readable summary of what
those imply for the reported results.
