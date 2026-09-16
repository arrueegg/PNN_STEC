# Metrics and exclusions: the decided reporting methodology

**Status**: design agreed 2026-09-15 (owner + session); **implemented and verified
2026-09-16**. Supersedes nothing; it is the first document that states one exclusion policy
for the *whole* paper rather than per-chapter. All six pipeline changes in Sec 3 are live
declared stages, and every section of Sec 2 has reached `PNN_main_revised.tex`. Sec 3 and
Sec 4 record what was built, not what remains.

**Every number in this document is a dated snapshot** read on 2026-09-15 from the path named
beside it. Read the path, not this document, before using a number. This repository has a
documented history of three documents quoting three different values for one table.

---

## 1. The principle

Two kinds of exclusion were on the table and only one of them is ever admissible.

**Outcome-based exclusion** — dropping a result because the error was large — is not used
anywhere in the paper. Measured: the positioning headline's *mean* improvement over IGS GIM
swings 17.1 points across plausible thresholds (-1.2% at no filter, +16.0% at 5 m, +3.8% at
10 m, -0.2% at 50 m) while the *median* moves 2.4 points across the same range including no
filter at all (`positioning_diagnostics/rebuilt/outlier_headline_sensitivity.csv`). The
retired 10 m station-day rule is not reinstated anywhere.

**Mechanism-based exclusion** — dropping a result because the method never got a fair
attempt — is admissible only when the criterion is defined on *inputs* and is uncorrelated
with difficulty. Every candidate we tested failed that second test:

| Candidate | Why rejected |
|---|---|
| recovered vs original station-days | At matched satellite count and normal local ionospheric level the penalty is x1.01 - it vanishes. The apparent penalty is an ionospheric-activity selection effect (recovered days carry x1.15 higher predicted STEC and x1.19 higher predicted uncertainty at the same station, Wilcoxon p=0.001/0.0003). Excluding them would delete the hardest days. |
| STEC observations below 7 deg elevation | 3.4% of observations, and the band where the model's advantage over the baselines is largest (median abs error 3.30 vs IGS GIM 6.16 TECU, against 1.49 vs 2.37 at 60-90 deg). Excluding it would delete the headline result. |
| high cycle-slip-count arcs | 12% of observations; all four methods degrade together. Cycle slips are *caused* by a disturbed ionosphere, so this is the recovered-days trap again. |

**Consequence**: nothing is excluded. Where a population is genuinely not comparable, it is
**stratified and reported**, never dropped.

## 2. Decisions per result family

### 2.1 Tables 3 and 4 - STEC accuracy

Keep RMSE, MAE and R2 as the columns (the STEC literature runs on them and departing invites
an argument the paper does not need). **Change the spread statistic from mean +/- std across
days to median and quartiles across days, and add one observation-level tail column.**

Reason: mean +/- std misdescribes a skewed day distribution. The Pretrained row is the clear
case - `daily_metrics/rebuilt/per_day.csv` gives RMSE 7.53 min / 12.19 median / 44.63 max
against a reported 13.45 +/- 4.84.

Reason for the tail column: the observation-level error distribution is heavy - on 2024 DOY
200 the worst 1% of observations carry 38.6% of the total squared error - and the current
tables say nothing about it. The positioning chapter reports P95/P99; this makes the two
chapters consistent in kind without restructuring Tables 3/4.

Footnote to add: the tables report the mean of per-day RMSE; `summary.csv` also carries a
pooled, observation-weighted `pooled_RMSE` one column over. They differ by under 3% (6.92 vs
6.96 own, 14.63 vs 15.01 Madrigal) and are not interchangeable.

### 2.2 Table 4 and Madrigal - two panels

Madrigal is not a straight accuracy reference. All four products read **systematically
higher** than Madrigal (mean signed per-station offset: IGS GIM +8.76, Direct STEC +6.51,
Pretrained +5.69, VTEC + Mapping +4.34 TECU; for IGS GIM all 67 stations are positive), and
the discrepancy grows toward the geomagnetic equator (Spearman of offset against |sm_lat|:
-0.70 / -0.63 / -0.66 / -0.57, all p < 2e-5). This is common to all four and is therefore a
property of the reference chain. Table 4's absolute ranking is dominated by which product's
level happens to sit closest to Madrigal's.

**Report both panels:**

1. **Plain agreement** (Table 4 as it stands, unhedged - VTEC + Mapping wins, we say so).
2. **dSTEC**, all four methods, which is free of any constant per-arc offset by construction
   and requires no fitted correction.

The offset-removed re-scoring in `madrigal_method_offset_comparison` stays a labelled
sensitivity diagnostic and is not quoted. State explicitly that dSTEC removes an *additive*
offset, not a multiplicative scale, so the panel is much less contaminated, not clean.

Snapshot, 2026-09-15, both from `dstec_evaluation` after the four-method extension:

| Dataset | Metric | Direct STEC | IGS GIM | VTEC + Mapping | Pretrained |
|---|---|---:|---:|---:|---:|
| own (242 d, 672,542 arcs) | dSTEC RMSE pooled | **5.16** | 6.64 | 8.04 | 10.82 |
| own | absolute RMSE, same masked obs | **6.34** | 7.89 | 8.76 | 14.64 |
| Madrigal (238 d, 944,845 arcs) | dSTEC RMSE pooled | **9.65** | 10.48 | 12.64 | 13.84 |
| Madrigal | absolute RMSE, same masked obs | 15.84 | 16.86 | **15.22** | 19.58 |

The ranking flips on Madrigal between the two rows. That flip is the finding.

The absolute column is computed on the elevation-masked subset (205 M of 449 M observations
on Madrigal) and is **not** Table 4's number - do not quote it as one.

**Reporting unit for dSTEC: the arc, and the statistic is the median across arcs** (owner
decision 2026-09-15). dSTEC is defined per arc, so pooling across arcs mixes the unit the
metric is built on, and arc lengths are highly heterogeneous (1 to 1,395 masked
observations; 5th percentile 20, 95th 706), so pooling weights a long overhead pass ~70x a
short low one. The across-arc distribution is right-skewed - Direct STEC on own reads
median 2.517, mean 3.746, p95 10.888 over 672,542 arcs - so the same median-over-mean
reasoning used for Tables 3/4 and Table 5 applies unchanged.

All three statistics are retained in the artifact so the choice is visible. Improvement of
Direct STEC over IGS GIM, 2026-09-15:

| Dataset | median across arcs | mean across arcs | pooled |
|---|---:|---:|---:|
| own | 38.6% | 30.2% | 22.3% |
| Madrigal | 26.7% | 17.1% | 8.0% |

**Stated plainly because it matters:** the median is also the most favourable of the three.
The justification is that median-over-mean was adopted for the positioning tables before
this was measured, on a threshold-sensitivity argument that applies identically here - and
the mitigation is that all three columns ship, so a reader can apply whichever they prefer.

**No minimum arc length.** Raising a floor from 0 to 100 masked observations drops 21% of
arcs and moves the median-across-arcs improvement 38.6% -> 39.3%. Short arcs are a
different kind of arc (median maximum elevation 27.7 deg against 75.9 deg for the longest
bin) but show the same improvement (35.3% against 40.2%), so there is nothing to correct
for and the no-exclusions rule of Sec 1 stands.

### 2.3 Positioning - population and statistic

- Median 3D RMS with IQR, P95, P99 and exceedance counts (already in place, Tables 5/6).
- **Drop recovered/original as a reporting axis** and replace it with a stratification by
  **local ionospheric activity**, which is what actually drives the difference. The activity
  proxy must be **GIM VTEC at the pierce point**, not the model's own predicted STEC: the
  model's prediction is circular for stratifying the model's own performance, and the 27-day
  sample used to establish the effect must become all 242 days.
- Never quote the median alone: every percentile table stays paired with an exceedance table,
  because the method does produce more large errors than IGS GIM and that is operationally
  real (`positioning_distributions`'s own caveat, unchanged).

### 2.4 Positioning - the constellation limitation

Parked as an upstream data limitation, reported with measured numbers, not fixed in this
revision. Facts to state, all measured 2026-09-15:

- 10 of 55 test stations (KOUG, FAA1, KAT1, NKLG, HRAO, SUTH, ZECK, SOLO, BIK0, UNSA) carry
  corrections for one constellation only; the other 45 have a method-to-method satellite-count
  gap of 0.00-0.10.
- On those station-days the ML arms solve with a median 8.6 satellites against IGS GIM's 17.1
  - 2,525 of 10,715 station-days.
- Within-station cost, controlling for the recovery flag and for ionospheric level:
  **x1.41**, 21 of 23 qualifying stations significant, as reported by the
  `constellation_coverage` stage at its declared `min_days_each=20` threshold (measured
  2026-09-15, superseding the x1.48 first quoted in this session - that came from an
  exploratory script using a >=25-day threshold, not from declared code).

  **The estimate is mildly threshold-dependent, and the paper must say so rather than pick
  the flattering end.** Measured directly:

  | min days of each kind per station | stations | median penalty | significant |
  |---:|---:|---:|---:|
  | >=15 | 26 | x1.402 | 24/26 |
  | **>=20 (declared default)** | **23** | **x1.412** | **21/23** |
  | >=22 | 22 | x1.483 | 20/22 |
  | >=25 | 22 | x1.483 | 20/22 |
  | >=30 | 19 | x1.554 | 17/19 |

  It drifts upward with the threshold because a stricter cut retains the stations that have
  the most of both kinds of day, which are the heavily-affected ones. Quote the range
  **x1.4-1.55** and name the threshold, not a bare point estimate. The stage is the
  authority; any number in the manuscript comes from its artifact.
- 67% of Direct STEC's >10 m station-days and 9 of its 12 >20 m station-days sit there;
  IGS GIM has **zero** >10 m station-days in that population.
- Root cause is upstream: CamaliotGnss reports `Constellations Used: GE` and lists the GPS
  satellites it processed, then writes zero GPS records into the `+SLANT/SOLUTION` block.
  Reproduced directly on BIK0/DOY 122. Ruled out by direct test: our observable selection
  (returns `G,E` correctly), receiver DCB availability (patching a zero C1W-C2W DSB into the
  BSX changes nothing), and unpopulated observables (C1W/C2W present in 467/504 records).
  Not the CAS DCB product's station coverage - an earlier claim in this session, retracted.
- `STEC_DB_estDCB` covers all ten stations with both constellations but is **not** to be used
  (owner decision 2026-09-15: older build, Feb 2025, may carry superseded errors).

**Closed 2026-09-16 (owner decision): the within-station penalty stays at the declared
stage's value and is not refined further.** Three figures for the same measurement were in
circulation, differing only in threshold, not in result: x1.41 (`min_days_each=20`, all days,
23 stations, 21 significant — what `constellation_coverage` writes and what the paper
quotes), x1.48 (`min_days_each=25`, all days, 22 stations), and x1.52 (`min_days_each=20`
restricted to original, non-geometry-recovered station-days, 17 stations). The x1.52 variant
is the better-founded one scientifically, because excluding recovered days removes the
ionospheric-activity confound — but it exists in no artifact, and adding it would have meant
a second column plus a stage regeneration. The owner chose not to pursue it. **x1.41 is the
number; the x1.48/x1.52 variants are recorded here only so a future reader who recomputes
and gets a different figure knows why, and does not mistake it for a disagreement.**

### 2.5 PPPx satellite code biases

One sentence in the methods. PPPx runs single-frequency PPP (`sf_ppp = yes`) and is given no
bias product (`bia` is empty in every arm). The IGS GIM arm's IONEX file carries a
`DIFFERENTIAL CODE BIASES` block (58 satellite records) which PPPx reads; the model and VTEC
arms supply a CSV with no such block. Measured by A/B on WARK/DOY 122 (same station-day, only
the DCB block removed): every one of 2,880 epochs changes, mean 3D difference **7.6 cm**, max
24 cm, flat across the whole day rather than decaying - so it is an applied correction, not a
re-estimated parameter. Receiver-side biases *are* absorbed, by the per-constellation receiver
clock PPPx estimates, so only the satellite side is asymmetric.

### 2.6 Uncertainty

Report the raw calibration **and** the single calibrating factor.

Snapshot 2026-09-15, own test set, 475,111,413 observations, each model under its native
distribution (`uncertainty_calibration/rebuilt/finetuned_stec_own/scores.csv`):

| Model | CRPS | PIT KS | mean predicted scale | RMSE | scale/RMSE |
|---|---:|---:|---:|---:|---:|
| Direct STEC (Gaussian) | 2.89 | 0.030 | 4.05 | 6.96 | 0.58 |
| VTEC + Mapping (Laplace) | 5.53 | 0.222 | 24.47 | 9.00 | 2.72 |

Coverage, Direct STEC: 50% -> 49.9%, 68% -> 65.9%, 90% -> 84.3%, 95% -> 88.9%, 99% -> 94.0%.

RMSE / sigma by elevation (`uncertainty_error_relation/rebuilt/by_elevation.csv`) is
1.61 / 1.64 / 1.64 / 1.67 / 1.68 / 1.65 / 1.65 / 1.64 / 1.54 from 0-10 deg to 80-90 deg.

The justification for stating the factor is precisely that constancy: a uniform x1.6 across
every elevation band is a **scale** error, not a broken uncertainty model, and the table above
is the evidence. Epistemic share is 5.1-6.6% throughout; the separate epistemic-scale
diagnostic finds rescaling the epistemic term alone does not improve error ranking, which is
consistent with the deficiency being in the aleatoric scale.

Each model must be scored under its **native** family - Gaussian for Direct STEC, Laplace for
the VTEC baseline. Scoring the Laplace baseline as Gaussian is a known trap in this repo.

## 3. Pipeline work — DONE, all six

Nothing above is permanent until it is a declared `Stage` in `stec/pipeline/stages.py` with a
`.pipeline/<stage>.json` provenance record. All six landed; `python -m stec.pipeline status`
reports 0 of 48 stages stale as of 2026-09-16. Executed across several concurrent sessions
working from this plan, not by one.

1. **`dstec_evaluation`** - module already extended to all four baselines
   (`COMPARISON_METHODS`, landed 2026-09-15, existing Direct-STEC and GIM numbers verified
   byte-identical). Re-run through the pipeline with `--force` so the canonical artifact and
   its provenance record are regenerated.
2. **New stage `dstec_evaluation_madrigal`** - the Madrigal arm is currently a manual
   invocation writing to `dstec_evaluation/rebuilt/finetuned_stec_madrigal/` with **no
   provenance record at all**. Declare it: `inputs=[predictions/finetuned_stec/madrigal]`,
   its own `min_rows`, and `canonical_for` naming the Madrigal dSTEC panel.
3. **`daily_metrics`** - add across-day median/quartiles and an observation-level tail column
   for Tables 3/4 (Sec 2.1). Keep the existing columns so nothing published moves.
4. **New stage for the ionospheric-activity stratification** of positioning (Sec 2.3), with
   the activity proxy derived from GIM VTEC and run over all 242 days.
5. **New stage for the constellation-coverage limitation** (Sec 2.4) so every number in the
   limitation paragraph is reproducible rather than quoted from a session transcript.
6. **`uncertainty_error_relation`** - surface the calibrating factor as a named summary
   output rather than a column a reader has to compute. Also give it a `min_rows` entry: it
   currently declares its output at directory level only, the same gap that was closed for
   `weighting_ablation` on 2026-09-15.

## 4. Manuscript work — DONE

Every section of Sec 2 is in the manuscript, verified 2026-09-16 by reading the `.tex`
against the artifacts: Tables 3 and 4 in median+IQR form with a P95 column (32 cells checked
cell by cell, one error found and fixed — Table 4's Pretrained R2 printed 0.82 where
`R2_median` is 0.8147, i.e. 0.81); the dSTEC table carrying both datasets and matching
`dstec_evaluation` and `dstec_evaluation_madrigal` exactly (own 2.52/4.10/4.67/6.31,
Madrigal 3.90/5.32/6.00/7.38, medians across arcs); the Madrigal reference-level paragraph;
the activity stratification; the constellation limitation; the satellite-code-bias sentence;
and the uncertainty calibration statement. Figure captions were checked too and are correct.

Edits go to `STEC_Modelling/PNN_main_revised.tex`. `PNN_main.tex` stays frozen as the
pre-revision comparison copy (md5 `352234f9b6a08e4aff2e6a009d78243e`).

Sections touched: Tables 3/4 columns and footnote (2.1); a new Madrigal dSTEC panel and the
reference-level paragraph (2.2); the positioning activity stratification replacing the
population split (2.3); a limitation paragraph on constellation coverage (2.4); one methods
sentence on satellite code biases (2.5); the uncertainty calibration statement including the
constant factor (2.6).

## 5. Open, deliberately not done

- CamaliotGnss GPS-slant behaviour: diagnosed, not fixed. Needs the WP2 pipeline owner.
- No PPPx re-runs. A satellite-matched IGS GIM arm and a constellation-completed ML arm were
  both considered and declined; the limitation is reported instead.
- `oracle_benchmark` keeps its own elev-weighted, 10 m-excluded methodology by explicit owner
  instruction - it answers a self-contained ratio-to-floor question, not one of these tables.
- The x1.52 confound-free constellation penalty (Sec 2.4) - owner decision 2026-09-16 not to
  pursue it.
- No automated check that numbers quoted in figure *captions* match their CSVs. The captions
  were verified by hand 2026-09-16 and are correct. Two things make this harder than it
  looks and would have to be handled first: a caption's population must be resolved before
  its values are compared (Figure 13 quotes the common set, N=10,387, 65/105/72/14, while
  the similarly named `overall_exceedance.csv` holds 70/115/80/16 over N=10,717 - both
  correct), and trackchanges scoping must be respected, since `\remove{}` blocks and the
  first argument of `\change{old}{new}` are a record of superseded text, not live claims.
