# PNN-STEC revision — what evidence exists, and where

Context for adapting the manuscript. The paper *"Probabilistic Machine Learning for Slant
Total Electron Content Modelling based on GNSS"* (Rüegg, Mao, Pan, Orús Pérez, Soja) was
rejected by JGR: Machine Learning and Computation after two reviews. Neither reviewer
disputed a result — R1 called the work "relevant, novel". The rejection was about framing
and missing diagnostics.

All analysis below is **done and reproducible**. Every number has a CSV and, except where
noted, a figure. Regenerate everything with `python src/analysis/build_all.py --figures`.

- Metric CSVs: `multiday_results/<analysis>/`, indexed in
  `multiday_results/revision_metrics_index.csv` (46 files, mapped to reviewer comment)
- Figures: `plots/revision/<data source>/` — use the **`_notitle`** variants in the paper
- Positioning tables in Table 5 format: `multiday_results/positioning_summary/`
  (`overall.csv`, `by_regime.csv`, `by_weighting.csv`)

---

## The framing issue that caused the rejection

Both reviewers led with the same objection (R1.4, R2.1): daily fine-tuning uses observations
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

### R1.1 — unconventional temporal split
Absolute RMSE is 1.84× higher under extrapolation (2024: 14.05 TECU) than interpolation
(2014–2023: 7.65), but mean STEC is 2.11× higher, so **normalised error is lower** in the
extrapolation regime: 26.9% vs 31.0%.
`multiday_results/temporal_regime_comparison.csv`

### R1.2 — attributing the 2024 degradation to solar maximum
Absolute RMSE spans ×3.7 across 2014–2024; TEC-normalised RMSE spans only ×1.5 and R² stays
0.80–0.91. corr(RMSE, mean STEC) = **+0.954**; corr(nRMSE, mean STEC) = −0.166.
**2024 has the lowest normalised error of any year (26.9%)**, despite the highest absolute
error. 2015 (40.3%) and 2017 (35.8%) are the genuine outliers, both low-sample early years.
`multiday_results/relative_error_metrics.csv` ·
`plots/revision/stec_pretrained_testset/relative_error_{absolute,normalised}_notitle.png`

### R1.3 — random station split may be over-optimistic
**Does not exonerate the split.** Normalised error rises from 12.6% for test stations within
100 km of a training station to 20.4% beyond 1000 km; Spearman +0.32 over 55 stations. Not
monotonic at the near end (100–250 km band is best at 8.7%), and distance is confounded with
region — isolated stations sit in sparse Southern-Hemisphere, oceanic and equatorial areas
that are intrinsically harder. **Best presented as a quantified limitation, not a rebuttal.**
Note: n = 55 stations is the binding constraint, so this will not get stronger with more
data; only a region-held-out retrain would settle it.
`multiday_results/station_independence/` ·
`plots/revision/stec_finetuned_2024/station_independence_notitle.png`

### R1.5 / R1.8b — simpler alternatives, hyperparameter selection
729 STEC runs across 5 architectures. Best validation MAE: BayesianResNetSTEC **1.24** (711
runs), FactorizedSTEC 2.87 (7), BNN_NLL 5.43 (4), MLP_NLL 5.78 (5), ResNet_BNN_NLL 13.57 (2).
**Report the run counts** — the search was unbalanced, and ResNet_BNN_NLL has *zero* runs
reaching 20 epochs, so its last place is not usable evidence.
Also: **Table 2 is incomplete** — it omits the KL annealing schedule (linear 0 → 0.1 over 5
warmup epochs), the predictive-variance floor, and the output-bias initialisation.
`multiday_results/hyperparameter_search/` ·
`plots/revision/training_runs/architecture_search_notitle.png`

### R1.8h — computational cost
STEC daily fine-tune: median 25 epochs, 9.0 s/epoch, 3.6 min/day, **15.4 GPU-hours over 242
days**. VTEC: 6.8 min/day, 19.4 GPU-hours. Inference: 8,605 observations/s at T = 100, i.e.
4.7 min per evaluation day. Pretraining ≈ 0.4 GPU-hours (scaled from the measured epoch cost,
not measured). Hardware: RTX 4070 Ti, 24 cores.
`multiday_results/computational_cost/cost_summary.csv`

### R1.8f, R1.8g — fine-tuning details
Already in the manuscript (Sec 3.4: training stations only; Sec 3.3: all parameters updated,
no freezing). Make more prominent; no new work.

---

### R2.2 — Bayesian only in the output layer
Accepted as a limitation; relabel the small epistemic component in Sec 4.2 accordingly.
A matched fully-Bayesian run (`config/config_A4_fully_bayesian.yaml`, identical except
`model_type: ResNet_BNN_NLL`) is **pending** (~1 GPU-hour). Confound to report: the published
architecture initialises its output bias to the dataset mean STEC, the fully-Bayesian variant
does not.

### R2.3 — products may have inconsistent bias references ✅ decisive
On the Madrigal geometries there are three independent estimates of the same slant path: the
model, the IGS GIM mapped to that line of sight, and Madrigal. The model and the GIM share
nothing in their construction, yet **corr(offset_model, offset_gim) = +0.946** over 66
stations, both exceed Madrigal at 91% of stations, mean offsets +5.63 and +7.76 TECU.
Removing a per-station constant drops the model's Madrigal RMSE from 13.64 → 10.10 TECU:
**45% of the Table 4 variance is a reference offset, not model error.**
`multiday_results/madrigal_reference_offset/` ·
`plots/revision/stec_finetuned_2024/madrigal_reference_offset_notitle.png`

### R2.4 — stratify beyond aggregate scatter ✅ strong result
Elevation, latitude, local time and season are already Figures 5–8. The missing axis was
activity. **The advantage over IGS GIM widens with disturbance:**

| Dst bin | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|
| quiet (> −30 nT) | 6.8 | 12.7 | 8.9 | 8.3 | +18% |
| weak | 7.1 | 13.9 | 9.0 | 8.6 | +18% |
| moderate | 7.4 | 15.6 | 9.1 | 9.2 | +20% |
| intense (≤ −100) | **8.0** | 24.3 | 9.6 | 12.1 | **+34%** |

Quiet → intense, Direct STEC degrades +19%, IGS GIM +46%, pretrained-only +91%. Across F10.7
terciles the margin holds at +20/+15/+24%. Bins: 14/25/38/165 days (Dst), 81/81/80 (F10.7).
`multiday_results/activity_stratification/` ·
`plots/revision/stec_finetuned_2024/activity_{dst,f107}_{absolute,improvement}_notitle.png`

### R2.5 — stochastic-model ablation
Paired station-days, uncertainty vs elevation weighting: Direct STEC **+3.0%** (better on
55.9% of station-days), VTEC + Mapping −2.7%, IGS GIM −0.1%.
**Moderate the manuscript claim accordingly**: uncertainty weighting gives a small real gain
only where the uncertainty is observation-level and model-derived; the bulk of the ~31%
improvement over GIM comes from the STEC correction itself, not the weighting.
The fixed-variance arm is **running** (full 242 days).
`multiday_results/weighting_ablation/paired.csv` ·
`plots/revision/positioning_2024/weighting_ablation_notitle.png`

### R2.6 — calibration diagnostics ✅
Conceded: monotonic association is not calibration. On the own test set (43.6 M observations)
the uncertainties are close to calibrated centrally and over-confident in the tails —
50% nominal → 48.8% empirical, 90% → 84.0%, 95% → 88.9%. **CRPS 2.80 against 3.11 for a
constant sigma**, so the per-observation uncertainty is worth ~10% on a proper score.
Coverage degrades under storms (95% → 87.6% vs 89.8% quiet).

⚠️ **The Madrigal calibration figures are largely a reference artefact** — see R2.3. Coverage
at 95% goes 63.8% → 77.0% once the per-station offset is removed. **Do not cite the Madrigal
calibration as evidence about out-of-distribution uncertainty.** The claims that hold the
processing chain fixed — storm/quiet, and station distance — are the ones that carry
generalisation arguments.
`multiday_results/uncertainty_calibration/` ·
`plots/revision/stec_finetuned_2024/calibration_{coverage,pit}_notitle.png`

### R2.7 — convergence, tails, vertical/horizontal, storm-time ✅
*Storm:* Direct STEC keeps **+31.9% over GIM in quiet and +26.3% in storm** conditions, and
degrades least of the ML models (+21.1% quiet→storm, against the pretrained variant's +46.6%).
*Tails:* also best — p95 3.16 m vs GIM 4.11 m, p99 5.08 vs 5.67, and 13.7% of station-days
above 2 m against GIM's 28.3%.
*Components:* vertical error cut 32% (0.92 m vs 1.34 m), horizontal 28%.
*Convergence time:* not derivable from the stored solutions and not meaningful for kinematic,
daily-reprocessed SF-PPP — decline that sub-point explicitly.
⚠️ Methodological note worth including: without the paper's own 10 m outlier rule, 102 of
35,652 station-days (0.29%) dominate the quiet-period mean enough to **reverse** the
storm/quiet ordering.
`multiday_results/{storm_stratification,positioning_robustness,positioning_summary}/` ·
`plots/revision/positioning_2024/{storm_positioning_*,positioning_tail}_notitle.png`

### R2.8 — observation-derived upper bound
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
| Prediction store, full 242 days | running (~57 GPU-h) — refreshes R2.3, R2.6, R1.3, R1.6 |
| Oracle + fixed-variance, full 242 days | running (~40 h) — R2.8 and the last R2.5 arm |
| R1.6 uncertainty vs error, fine-tuned | pending, needs the store |
| R2.2 fully-Bayesian comparison | pending, ~1 GPU-hour |
| R2.4b Figure 4 stratified (pretrained) | pending, ~30 GPU-min |

Numbers above are from the current data. When the jobs finish, `build_all.py --figures`
refreshes everything; only R2.8 and the storm half of R2.6 are expected to change materially.

## Venue

Recommend resubmitting to JGR-MLC as a new manuscript with a full response document; fallback
GPS Solutions, then Space Weather. Roughly 60% of the criticised material already existed and
was cut for space, so a venue change would not reduce the work. Lead the cover letter with the
operating-mode clarification.
