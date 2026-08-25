# PNN-STEC revision — what evidence exists, and where

Context for adapting the manuscript. The paper *"Probabilistic Machine Learning for Slant
Total Electron Content Modelling based on GNSS"* (Rüegg, Mao, Pan, Orús Pérez, Soja) was
rejected by JGR: Machine Learning and Computation after two reviews. Neither reviewer
disputed a result — R2 called the work "relevant, novel". The rejection was about framing
and missing diagnostics.

All analysis below is **done and reproducible**. Every number has a CSV and, except where
noted, a figure. Regenerate everything with `python src/analysis/build_all.py --figures`.

- Metric CSVs: `multiday_results/<analysis>/`, indexed in
  `multiday_results/revision_metrics_index.csv` (46 files, mapped to reviewer comment)
- Figures: `plots/revision/<data source>/` — use the **`_notitle`** variants in the paper
- Positioning tables in Table 5 format: `multiday_results/positioning_summary/`
  (`overall.csv`, `by_regime.csv`, `by_weighting.csv`)

---

## Status at a glance — what to write today

| Reviewer comment | Status | Action |
|---|---|---|
| Framing (R2.4 / R1.1) | **READY** | Write now. No data needed; this is what caused the rejection. |
| R2.1 split regimes | **READY** | Final numbers. |
| R2.2 2024 attribution | **READY** | Final numbers. |
| R2.5 / R2.8b architectures | **READY** | Final numbers. Also complete Table 2. |
| R2.8f / R2.8g fine-tune details | **READY** | Already in the text; make prominent. |
| R2.8h computational cost | **READY** | Final numbers. |
| R1.4 activity stratification | **REVISED** | Conclusion reversed by the GIM fix — use the new table, not the old. 8 quiet days still to settle. |
| R1.5 elevation vs uncertainty | **READY** | Final — already all 242 days. |
| R1.7 storm, tails, components | **READY** | Final — already all 242 days. |
| R2.3 station independence | **READY (as a limitation)** | Write as a quantified limitation; it will not improve. |
| R1.3 Madrigal reference offset | **READY** | 67 stations, 235/238 possible days. Quote Spearman +0.698 and 95.5% sign agreement, not Pearson +0.925 (leverage). Offsets are 24x the reference's own stated precision. Computed from the pre-correction Madrigal store (old local-time convention); a re-inference is under way. |
| R1.6 calibration | **PROVISIONAL** | Own-test-set coverage is settled; the storm/quiet split will shift. |
| R1.8 oracle bound | **PENDING** | 48/242 days and still moving. Framing is safe; numbers are not. Uses **elev** weighting and paired station-days, so it is not comparable with Table 5. |
| R1.5 fixed-variance arm | **READY** | 242 days. Constant sigma is 11.5% *worse* than elevation weighting; the model's sigma is 2.6% better. |
| R2.6 uncertainty vs error, fine-tuned | **READY** | Built; final once the store covers 242 days. |
| R1.2 fully-Bayesian comparison | **READY** | Done — matched-init retrain evaluated. Paper model RMSE 11.67 vs fully-Bayesian 15.54 (1.33×); uncertainty–error correlation marginally favours the fully-Bayesian arm (0.575 vs 0.568); epistemic-scale diagnostic shows the paper model's under-dispersion is scale, not structure. |
| R1.4b Figure 4 stratified | **PENDING** | Pretrained pass is queued; the stratification itself is not built. |
| Tables 3 & 4 corrected | **PENDING** | `daily_metrics.py` recomputes them from the store; exact only at 242/242 days. |
| R1.6b uncertainty vs IONEX RMS | **READY** | Final on 43 days; will only firm up as the store grows. |
| IGS GIM baseline correction | **ACTION NEEDED** | Table 4's GIM column and the R1.4 text must be updated before resubmission. |

**So: 12 of 19 items can be written today**, including the framing change that matters most and,
as of this pass, R1.2 (the fully-Bayesian comparison, now done — see below). Two more are
provisional — safe to draft, worth rechecking the exact figures. Four need results that do not
exist yet; draft around them and leave the numbers as placeholders.

Nothing in the remaining PENDING list is expected to *change direction* — the oracle will stay
an order of magnitude below the models. They are missing precision, not missing answers.

---

## The framing issue that caused the rejection

Both reviewers led with the same objection (R2.4, R1.1): daily fine-tuning uses observations
from the day being predicted, so the evaluation looks like it overstates real-time
usefulness.

**This is a framing failure, not an evidentiary one.** Daily fine-tuning mirrors the
production of IGS **final** products, which are themselves generated after the fact from the
full day's observations — the same information regime as the paper's own baseline (final IGS
GIM). The comparison is like-for-like, but the manuscript never says so. State the operating
mode explicitly in the abstract, Section 3.3, Section 3.4 and the conclusion, and present the
pretrained model as the variant applicable without same-day data. No new analysis needed.

---

## Evidence per reviewer comment

### R2.1 — unconventional temporal split
Absolute RMSE is 1.84× higher under extrapolation (2024: 14.05 TECU) than interpolation
(2014–2023: 7.65), but mean STEC is 2.11× higher, so **normalised error is lower** in the
extrapolation regime: 26.9% vs 31.0%.
`multiday_results/temporal_regime_comparison.csv`

### R2.2 — attributing the 2024 degradation to solar maximum
Absolute RMSE spans ×3.7 across 2014–2024; TEC-normalised RMSE spans only ×1.5 and R² stays
0.80–0.91. corr(RMSE, mean STEC) = **+0.954**; corr(nRMSE, mean STEC) = −0.166.
**2024 has the lowest normalised error of any year (26.9%)**, despite the highest absolute
error. 2015 (40.3%) and 2017 (35.8%) are the genuine outliers, both low-sample early years.
`multiday_results/relative_error_metrics.csv` ·
`plots/revision/stec_pretrained_testset/relative_error_{absolute,normalised}_notitle.png`

### R2.3 — random station split may be over-optimistic — **READY, as a limitation**
**Does not exonerate the split.** Normalised error rises from 12.6% for test stations within
100 km of a training station to 20.4% beyond 1000 km; Spearman +0.32 over 55 stations. Not
monotonic at the near end (100–250 km band is best at 8.7%), and distance is confounded with
region — isolated stations sit in sparse Southern-Hemisphere, oceanic and equatorial areas
that are intrinsically harder. **Best presented as a quantified limitation, not a rebuttal.**
Note: n = 55 stations is the binding constraint, so this will not get stronger with more
data; only a region-held-out retrain would settle it.
`multiday_results/station_independence/` ·
`plots/revision/stec_finetuned_2024/station_independence_notitle.png`

### R2.5 / R2.8b — simpler alternatives, hyperparameter selection
729 STEC runs across 5 architectures. Best validation MAE: BayesianResNetSTEC **1.24** (711
runs), FactorizedSTEC 2.87 (7), BNN_NLL 5.43 (4), MLP_NLL 5.78 (5), ResNet_BNN_NLL 13.57 (2).
**Report the run counts** — the search was unbalanced, and ResNet_BNN_NLL has *zero* runs
reaching 20 epochs, so its last place is not usable evidence.
Also: **Table 2 is incomplete** — it omits the KL annealing schedule (linear 0 → 0.1 over 5
warmup epochs), the predictive-variance floor, and the output-bias initialisation.
`multiday_results/hyperparameter_search/` ·
`plots/revision/training_runs/architecture_search_notitle.png`

### R2.8h — computational cost
STEC daily fine-tune: median 25 epochs, 9.0 s/epoch, 3.6 min/day, **15.4 GPU-hours over 242
days**. VTEC: 6.8 min/day, 19.4 GPU-hours. Inference: 8,605 observations/s at T = 100, i.e.
4.7 min per evaluation day. Pretraining: **measured** at ≈6.2 GPU-hours over 150 epochs (2.5
min/epoch, steady) — the pretrain draws 500,000 random rows with replacement from the full
103 GB training set every epoch and is I/O-bound (~7% GPU utilisation), so it does not scale
from the fine-tune's per-epoch cost; an earlier scaled estimate (≈0.4 GPU-hours) was 16× low.
Hardware: RTX 4070 Ti, 24 cores.
`multiday_results/computational_cost/cost_summary.csv` ·
`docs/revision/manuscript_number_audit.md` (measured figure)

### R2.8f, R2.8g — fine-tuning details
Already in the manuscript (Sec 3.4: training stations only; Sec 3.3: all parameters updated,
no freezing). Make more prominent; no new work.

---

### R1.2 — Bayesian only in the output layer — **DONE**
Accepted as a limitation; relabel the small epistemic component in Sec 4.2 accordingly. A
matched fully-Bayesian run (`ResNet_BNN_NLL`, Bayesian residual blocks plus head, identical
hyperparameters and — after a corrected retrain — identical output-layer initialisation to the
paper model) is now evaluated on the same 10 M-observation test set: RMSE 15.54 against the
paper model's 11.67 (1.33×), R² 0.818 vs 0.897, mean predicted uncertainty 19.57 TECU vs 7.14
(2.74×). The fully-Bayesian variant is substantially less accurate; its uncertainty–error
correlation is marginally *better* (0.575 vs 0.568), so what last-layer-only Bayesian costs is
calibration, not ranking. A first comparison with mismatched initialisation overstated the
accuracy gap (1.69× RMSE); matching it closed about half the gap.

**Epistemic-scale diagnostic strengthens the answer**: sweeping a post-hoc scalar `s` on the
paper model's epistemic term alone, `s* = 4.66` restores its badly under-dispersed 1σ coverage
(9.4% vs 68.3% nominal) to nominal, while the uncertainty–error Spearman correlation is
essentially unchanged (0.5609 at s=1 → 0.5625 at s\*, marginally improving). **The deficit is
scale, not structure** — a single post-hoc multiplier repairs coverage without costing ranking
ability.
`docs/revision/r22_fully_bayesian_analysis.md` ·
`multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/*.csv`.
Note: the analysis file names itself "R2.2"; in the response letter's numbering the
fully-Bayesian question is **R1.2** (R2.2 is the solar-maximum attribution, a different
question, below) — use R1.2 consistently when citing this result.

### R1.3 — products may have inconsistent bias references — **PROVISIONAL** ✅ decisive
On the Madrigal geometries there are three independent estimates of the same slant path: the
model, the IGS GIM mapped to that line of sight, and Madrigal. The model and the GIM share
nothing in their construction, yet Spearman ρ = **+0.698** between their per-station offsets
over **67** stations, and both exceed Madrigal at **95.5%** of stations. The Pearson
correlation over all 67 is +0.925, but that is inflated by a sparse arm of large-offset
stations — restricted to |offset| < 15 TECU it falls to +0.617 at n = 53, which is why the
rank correlation and sign agreement are quoted instead. Removing a per-station constant drops
the model's Madrigal RMSE from **15.05 → 11.13 TECU** (mean |offset| 6.69 TECU): **45% of the
Table 4 variance is a reference offset, not model error.**
*[All Madrigal-derived numbers in this section are computed from the pre-correction Madrigal
store, under the old receiver-longitude local-time convention; they will be regenerated once
the local-time re-inference and downstream Madrigal analyses re-run.]*
`multiday_results/madrigal_reference_offset/` ·
`plots/revision/stec_finetuned_2024/madrigal_reference_offset_notitle.png`

### R1.4 — stratify beyond aggregate scatter ✅ — **conclusion revised, do not use the old table**
Elevation, latitude, local time and season are already Figures 5–8. The missing axis was
activity. **The advantage over IGS GIM narrows with disturbance** — the earlier "+18% → +34%,
widens" reading was an artifact of the GIM date defect (see the dedicated section below) and is
retracted.

| Dst bin | days | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|---|
| quiet (> −30 nT) | 165 | 6.78 | 12.69 | 8.95 | 8.14 | +16.7% |
| weak (−50 to −30) | 38 | 7.07 | 13.90 | 8.98 | 8.51 | +16.9% |
| moderate (−100 to −50) | 25 | 7.41 | 15.62 | 9.10 | 8.65 | +14.3% |
| intense (≤ −100) | 14 | **8.04** | 24.26 | 9.59 | 9.02 | **+10.9%** |

The claim that survives: Direct STEC is most accurate in every bin and degrades least from quiet
to intense (+19%, against +9% VTEC + Mapping, +9% IGS GIM, +91% pretrained-only). F10.7
terciles: +20/+15/+17%. Bins: 14/25/38/165 days (Dst), 81/81/80 (F10.7).

All 242 days are now in place, including the 12 whose GIM was recomputed — 8 of them in the
quiet bin, which is why that row moved from +18.4% to +16.7% (and weak from +18.0% to +16.9%)
while moderate and intense did not change. All four rows are final.
`multiday_results/activity_stratification/` ·
`plots/revision/stec_finetuned_2024/activity_{dst,f107}_{absolute,improvement}_notitle.png`

### R1.6b — predicted uncertainty vs the GIM products' own IONEX RMS ✅ decisive
Not a reviewer comment; it is the benchmark the word "Probabilistic" in the title invites, and it
is the strongest uncertainty result in the revision. Each product scored against **its own**
residuals, 43 days, 81.3 M observations.

| | RMSE | 95% cov. | σ scale for nominal | CRPS skill vs constant σ | Spearman(σ,\|err\|) |
|---|---|---|---|---|---|
| Direct STEC | **7.31** | **88.7%** | **×1.42** | **+10.5%** | **0.41** |
| CODE GIM + Mapping | 8.40 | 73.7% | ×2.02 | +1.5% | 0.39 |
| IGS GIM + Mapping | 8.49 | 46.9% | ×4.60 | **−8.1%** | 0.30 |

⏳ **A fourth arm is being added: the VTEC baseline's own uncertainty.** The Mao et al. MLP is
trained with a Laplacian NLL, so it predicts a scale, and `apply_mapping_function` already maps
that to the slant direction — the PPP's `VTEC_iono` arm weights by it. It was being dropped at
the prediction-store write (schema whitelist); fixed, and the backfill now carries it. First day
(DOY 124, Laplace-scored): mean σ 18.7 TECU against an RMSE of 6.4 and a mean absolute error of
3.8, i.e. **over-dispersed by roughly 3×** — coverage 81.6% at nominal 50%, 99.9% at 95%, CRPS
skill −42%. Its Spearman is 0.45, *higher* than ours: it ranks errors well but its scale is far
off. Scope this as "the baseline as configured here", not as a claim about the published model.
The 45 days stored before the fix lack the column and need a re-run for a like-for-like day set.

The IGS combined RMS scores *worse than a single constant* per observation. Caveats to state:
it is an inter-centre spread rather than a validated error, excludes mapping-function error, and
is a 5°/2 h grid quantity judged per observation.
`multiday_results/ionex_rms_benchmark/{overall,by_elevation,by_regime,per_day}_{IGS,CODE}.csv` ·
`plots/revision/stec_finetuned_2024/ionex_rms_{coverage,crps_skill}_notitle.png`

### R2.6 — predicted uncertainty against realised error ✅
`uncertainty_error_relation.py`, over the whole test period rather than the per-day PNGs. Two
views: by predicted-σ decile and by elevation.

**The model is over-confident everywhere**, by a factor RMSE/σ of 1.31–1.53 across σ deciles and
1.60–1.77 across elevation bins. The ratio is U-shaped in σ — best in the middle deciles (1.31),
worse at both the confident and the uncertain end — so it is not a single global scale error that
one constant would fix.

**The epistemic share of the predictive variance is 4.7–6.8%.** That is the number for R1.2: with
only the output layer Bayesian, essentially all the predicted spread is aleatoric.
`multiday_results/uncertainty_error_relation/by_{sigma,elevation}.csv` ·
`plots/revision/stec_finetuned_2024/uncertainty_vs_error_notitle.png`

### Tables 3 and 4 — recomputed from the store
`daily_metrics.py` derives the per-day and pooled metrics from the prediction store instead of
the inference-time aggregation, so it picks up the repaired GIM automatically and needs no GPU.
It writes `vs_published.csv` diffing against the published table, and warns when the store covers
fewer days than the published 242 — until then the deltas reflect the **day subset**, not the
correction. Note the published tables report the *mean of daily RMSE*, not the pooled RMSE;
both are written.
`multiday_results/daily_metrics/{per_day,summary,vs_published}.csv`

### GIM baseline defect — affects Table 4 and R1.4, **not** positioning
`compare_stec_vtec_gim.py` picked the IONEX file from a `doy` that had round-tripped through
float32 normalisation, so a truncating cast loaded the previous day's map on **DOY 184–189 and
225–230** (12 of 242 days). Table 4's IGS GIM entry moves 8.56 → **≈8.31 TECU** (own) and
15.64 → ≈15.50 (Madrigal, *computed from the pre-correction Madrigal store — to be regenerated
once the local-time re-inference re-runs*); the Direct STEC advantage over GIM falls
19.1% → ≈16.7%. Model and
VTEC baselines untouched; **all positioning results untouched** (the positioning pipeline takes
the day from `--date`). Fixed at source; `src/analysis/repair_gim_baseline.py` repairs stored
days and reproduces the unaffected ones to 1.5e-5 TECU.
`multiday_results/gim_baseline_repair/gim_repair_report.csv`

### R1.5 — stochastic-model ablation
Paired station-days, uncertainty vs elevation weighting: Direct STEC **+3.0%** (better on
55.9% of station-days), VTEC + Mapping −2.7%, IGS GIM −0.1%.
**Moderate the manuscript claim accordingly**: uncertainty weighting gives a small real gain
only where the uncertainty is observation-level and model-derived; the bulk of the improvement
over GIM comes from the STEC correction itself, not the weighting — ~20.3% on the matched
population (N = 7,741) and ~24.4% on the full recovered, unmatched population (N = 8,636 vs
10,837), both below the abstract's previously reported 30.9%.
The fixed-variance arm is **running** (full 242 days).
`multiday_results/weighting_ablation/paired.csv` ·
`plots/revision/positioning_2024/weighting_ablation_notitle.png`

### R1.6 — calibration diagnostics — **PROVISIONAL** ✅
Conceded: monotonic association is not calibration. On the own test set (475.1 M observations,
full 242-day store) the uncertainties are close to calibrated centrally and over-confident in
the tails — 50% nominal → 49.9% empirical, 68% → 65.9%, 90% → 84.3%, 95% → 88.9%. **CRPS 2.89
against 3.24 for a constant sigma**, so the per-observation uncertainty is worth ~11% on a
proper score. Coverage degrades under storms (95% → 87.8% vs 89.1% quiet).

⚠️ **The Madrigal calibration figures are largely a reference artefact** — see R1.3. Coverage
at 95% goes 61.4% → 73.8% once the per-station offset is removed *[computed from the
pre-correction Madrigal store; to be regenerated once the local-time re-inference and
downstream Madrigal analyses re-run]*. **Do not cite the Madrigal
calibration as evidence about out-of-distribution uncertainty.** The claims that hold the
processing chain fixed — storm/quiet, and station distance — are the ones that carry
generalisation arguments.
`multiday_results/uncertainty_calibration/` ·
`plots/revision/stec_finetuned_2024/calibration_{coverage,pit}_notitle.png`

### R1.7 — convergence, tails, vertical/horizontal, storm-time ✅
*Storm:* Direct STEC keeps **+25.4% over GIM in quiet and +19.6% in storm** conditions, and
degrades least of the ML models (+19.6% quiet→storm, against the pretrained variant's +41.2%).
Smaller than an earlier version of this document (+31.9%/+26.3%) because the 2026-08-24
station-recovery sweep enlarged the evaluated population; the advantage holds in both regimes.
*Tails:* best through p95 — 3.66 m vs GIM 4.10 m — but IGS GIM's p99 is now lower than Direct
STEC's: 5.66 m vs 6.24 m, a reversal from the earlier, smaller population. 15.9% of
station-days above 2 m against GIM's 28.1%.
*Components:* vertical error cut 25% (1.00 m vs 1.34 m), horizontal 22% (0.68 m vs 0.87 m) —
both smaller than an earlier version of this document (32%/28%), same population change.
*Convergence time:* not derivable from the stored solutions and not meaningful for kinematic,
daily-reprocessed SF-PPP — decline that sub-point explicitly.
⚠️ Methodological note worth including: without the paper's own 10 m outlier rule, a small
number of station-days out of the current 37,209 (grown from 35,652 pre-recovery-sweep)
dominate the quiet-period mean enough to **reverse** the storm/quiet ordering *[the previously
reported count, 102 station-days / 0.29%, has not been recomputed against the recovered
population — no artifact currently supports that recount]*.
`multiday_results/{storm_stratification,positioning_robustness,positioning_summary}/` ·
`plots/revision/positioning_2024/{storm_positioning_*,positioning_tail}_notitle.png`

### R1.8 — observation-derived upper bound — **PENDING (9/242 days)**
Applying the reference STEC directly as the correction gives **0.090 m** 3D RMS against
Direct STEC 1.028, IGS GIM 1.204, VTEC + Mapping 1.284 — models sit 11–14× above the floor.
Currently on a day subset; the full 242-day run is **in progress**.

Framing matters here: the reference STEC *is* the training target, from the same observations,
so this is the pipeline's noise floor rather than reachable headroom. The defensible claim is
**"almost all remaining positioning error is ionospheric modelling error, not orbit, clock or
multipath"** — stronger than what the reviewer asked for, and what this construction supports.
Validation: the oracle experiment's own IGS GIM rerun reproduces the published elevation-weighted
GIM arm at max |Δ| = 0.0000 m over 45 station-days.
`multiday_results/oracle_benchmark/` ·
`plots/revision/positioning_2024/oracle_benchmark_notitle.png`

---

## Still running / pending

| Item | State |
|---|---|
| Prediction store, full 242 days | running (~57 GPU-h) — refreshes R1.3, R1.6, R2.3, R2.6 |
| Oracle + fixed-variance, full 242 days | running (~40 h) — R1.8 and the last R1.5 arm |
| GIM repair on the remaining 8 affected days | waits on the store; re-run `repair_gim_baseline.py --apply`, then `activity_stratification.py` |
| Regenerate daily metrics from the store | replaces `all_results.csv`, retires the repair patch in `activity_stratification.py` |
| R2.6 uncertainty vs error, fine-tuned | pending, needs the store |
| R1.4b Figure 4 stratified (pretrained) | pending, ~30 GPU-min |

Numbers above are from the current data. When the jobs finish, `build_all.py --figures`
refreshes everything; only R1.8 and the storm half of R1.6 are expected to change materially.

## Venue

Recommend resubmitting to JGR-MLC as a new manuscript with a full response document; fallback
GPS Solutions, then Space Weather. Roughly 60% of the criticised material already existed and
was cut for space, so a venue change would not reduce the work. Lead the cover letter with the
operating-mode clarification.
