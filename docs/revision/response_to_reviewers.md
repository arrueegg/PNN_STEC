# Response to Reviewers

**Manuscript:** Probabilistic Machine Learning for Slant Total Electron Content Modelling
based on GNSS — Rüegg, Mao, Pan, Orús Pérez, Soja
**Status:** draft. Numbers marked ✅ are computed and reproducible from this repository;
⏳ marks results still being computed. `docs/revision/evidence_summary.md` is the companion
reference giving the CSV and figure behind every number, and its status table says which
sections can be written now.

Reproduction: `python src/analysis/build_all.py --figures`.

---

## Framing: what the method is for

Both reviewers led with the same objection (R1.4, R2.1): daily fine-tuning uses observations
from the day being predicted, so the evaluation looks like it overstates operational
usefulness for real-time correction.

**This is a framing failure in the manuscript, and we accept it.** The method was never
intended as a real-time correction service. Daily fine-tuning mirrors the production of IGS
**final** products, which are themselves generated after the fact from the full day's
observations — the same information regime the paper's own baseline (final IGS GIM) occupies.
The comparison is therefore like-for-like, but the manuscript never says so.

We will state the operating mode explicitly in the abstract, Section 3.3, Section 3.4 and the
conclusion, and present the pretrained model as the variant that *is* applicable without
same-day data. ⏳ text

---

## Reviewer 1

### R1.1 — temporal split design ✅
The two regimes the split creates are now reported separately. Absolute RMSE is 1.84× higher
under extrapolation (2024, 14.05 TECU) than interpolation (2014–2023, 7.65 TECU), but mean
STEC is 2.11× higher, so the **normalised error is lower** in the extrapolation regime:
26.9% against 31.0%. The design therefore does not flatter the interpolation years.

### R1.2 — attribution of the 2024 degradation ✅
Accepted, and the corrected analysis strengthens the paper. Absolute RMSE is not comparable
across the solar cycle, since mean STEC itself varies by a factor of 3.7.

| | span 2014–2024 | ratio |
|---|---|---|
| Absolute RMSE | 3.8 → 14.0 TECU | ×3.7 |
| **Normalised RMSE** | 26.8 → 40.3 % | **×1.5** |
| R² | 0.804 → 0.913 | flat |

corr(RMSE, mean STEC) = +0.954; corr(nRMSE, mean STEC) = −0.166. **In relative terms 2024 is
the best year in the record (26.9%), not the worst.** We will replace the solar-maximum
attribution with the statement that absolute error scales with ionospheric amplitude while
relative skill is stable, noting 2015 (40.3%) and 2017 (35.8%) as the genuine outliers.

### R1.3 — random station split ✅ (as a limitation)
We tested the hypothesis directly rather than only discussing it: per-station error against
great-circle distance to the nearest training station. Normalised error rises from 12.6% for
stations within 100 km to 20.4% beyond 1000 km, Spearman +0.32 over 55 stations.

**We do not claim this exonerates the split.** The relationship is not monotonic at the near
end (the 100–250 km band is best at 8.7%), and distance is confounded with region, since
isolated stations sit in the sparsely covered Southern Hemisphere, oceanic and equatorial
areas that are intrinsically harder. We will present it as a quantified limitation. A
region-held-out split would settle it but requires retraining, which we have not done.

### R1.5 — simpler alternatives ✅
729 STEC training runs were logged across five architectures:

| Architecture | best val MAE | runs | runs reaching ≥20 epochs |
|---|---|---|---|
| BayesianResNetSTEC (selected) | 1.24 | 711 | 433 |
| FactorizedSTEC | 2.87 | 7 | 4 |
| BNN_NLL | 5.43 | 4 | 3 |
| MLP_NLL | 5.78 | 5 | 1 |
| ResNet_BNN_NLL | 13.57 | 2 | **0** |

We report the run counts deliberately: the search was unbalanced, and we will say so rather
than imply a rigorous head-to-head.

### R1.8b — hyperparameter selection ⏳ text
Table 2 will be completed. It currently omits the **KL annealing schedule** (linear ramp
0 → 0.1 over 5 warm-up epochs), the predictive-variance floor, and the output-bias
initialisation to the dataset mean STEC.

### R1.8f / R1.8g — fine-tuning details
Already stated (Section 3.4: training stations only; Section 3.3: all parameters updated, no
freezing). We will make both more prominent.

### R1.8h — computational cost ✅
Daily STEC fine-tuning: median 25 epochs, 9.0 s/epoch, 3.6 min/day, 15.4 GPU-hours over the
242 test days. Inference: 8,605 observations/s at T = 100, i.e. 4.7 min per evaluation day.
Pretraining ≈ 0.4 GPU-hours (scaled from the measured epoch cost). Hardware: RTX 4070 Ti.

---

## Reviewer 2

### R2.2 — Bayesian only in the output layer ⏳ evidence
Correct, and we will relabel the small epistemic component in Section 4.2 as a **limitation of
the last-layer design** rather than a finding. A fully-Bayesian variant is implemented; its two
existing runs are not usable evidence (half the hidden width, 10× the learning rate, stopped at
epochs 10 and 7 of 150). A matched run is under way; we will report both accuracy and the
magnitude of the epistemic component, and note that the published architecture initialises its
output bias to the dataset mean STEC while the fully-Bayesian variant does not.

### R2.3 — comparability of STEC products ✅
The reviewer is right that the comparison conflates model error with reference inconsistency,
and we can now quantify the split. On the Madrigal geometries there are three independent
estimates of the same slant path: the model, the IGS GIM mapped to that line of sight, and
Madrigal. The model and the GIM share nothing in their construction, yet their per-station
disagreement with Madrigal is almost identical:

* **corr(offset_model, offset_gim) = +0.946** over 66 stations
* both exceed Madrigal at **91%** of stations; mean offsets +5.63 and +7.76 TECU
* removing a per-station constant drops the model's Madrigal RMSE from 13.64 to 10.10 TECU

**45% of the variance in Table 4 is a station-dependent reference offset, not model error.**
We will report Table 4 with this decomposition alongside, and expand Section 2 on how the
reference STEC is produced (carrier-phase levelling, arc handling, DCB application).

### R2.4 — stratification ✅
Elevation, geomagnetic latitude, local time and season are already Figures 5–8; the missing
axis was activity. The advantage over IGS GIM **widens** with disturbance:

| Daily min Dst | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|
| quiet (> −30 nT) | 6.8 | 12.7 | 8.9 | 8.3 | +18% |
| weak | 7.1 | 13.9 | 9.0 | 8.6 | +18% |
| moderate | 7.4 | 15.6 | 9.1 | 9.2 | +20% |
| intense (≤ −100 nT) | **8.0** | 24.3 | 9.6 | 12.1 | **+34%** |

From quiet to intense, Direct STEC degrades by 19%, IGS GIM by 46% and the pretrained-only
variant by 91%. Across F10.7 terciles the margin holds at +20/+15/+24%.
⏳ The equivalent stratification of Figure 4 itself (pretrained model) is being computed.

### R2.5 — stochastic-model ablation ✅ (one arm ⏳)
Restricting to station-days solved under both weightings:

| Correction | elevation [m] | predicted uncertainty [m] | gain |
|---|---|---|---|
| Direct STEC | 1.156 | 1.121 | **+3.0%** |
| VTEC + Mapping | 1.580 | 1.624 | −2.7% |
| IGS GIM + Mapping | 1.630 | 1.631 | −0.1% |

**We will moderate the manuscript's claim accordingly.** Uncertainty weighting yields a small
but consistent gain, and only where the uncertainty is genuinely observation-level and
model-derived; the majority of the ~31% improvement over IGS GIM comes from the STEC
correction itself. ⏳ A fixed-variance arm is being computed to complete the requested list.

### R2.6 — calibration diagnostics ✅
Conceded: monotonic association is not calibration. Treating each prediction as the Gaussian
the training loss assumes, on the own test set (43.6 M observations):

| nominal | empirical |
|---|---|
| 50% | 48.8% |
| 68% | 65.0% |
| 90% | 84.0% |
| 95% | 88.9% |

The uncertainties are close to calibrated centrally and over-confident in the tails. CRPS is
**2.80 against 3.11** for a constant-sigma model, so the per-observation uncertainty carries
about 10% on a proper score. Coverage degrades under storms (95% → 87.6%, against 89.8% quiet).

**On dataset shift we will be explicit about what the Madrigal numbers do and do not show.**
Coverage there is far below nominal, but so is any estimator's against a reference carrying a
systematic per-station offset: correcting for the offset established in R2.3 moves 95%
coverage from 63.8% to 77.0%. We therefore do not present the Madrigal calibration as evidence
about the model's out-of-distribution behaviour. The diagnostics that hold the processing
chain fixed — the storm/quiet split and the station-distance analysis — are the ones we use
for that.

### R2.7 — convergence, tails, vertical/horizontal, storm-time ✅
*Storm-time.* 39 storm days (daily min Dst ≤ −50 nT) of 242. Direct STEC retains **+31.9% over
IGS GIM in quiet conditions and +26.3% during storms**, and degrades least of the machine-learning
models (+21.1% quiet→storm, against +46.6% for the pretrained-only variant).

*Tails.* Also best: p95 3.16 m against 4.11 m for IGS GIM, p99 5.08 against 5.67, and 13.7%
of station-days above 2 m against 28.3%.

*Vertical vs horizontal.* Vertical error reduced 32% (0.92 m against 1.34 m), horizontal 28%.

*Convergence time.* Not derivable from the stored solutions and not meaningful for the
kinematic, daily-reprocessed single-frequency PPP used here; we will say so rather than
report a quantity the processing strategy does not support.

A methodological note we will include: without the 10 m outlier exclusion already used in
Figure 12, 102 of 35,652 station-days (0.29%) dominate the quiet-period mean strongly enough
to reverse the storm/quiet ordering.

### R2.8 — observation-derived upper bound ⏳
Applying the GNSS-derived reference STEC directly as the ionospheric correction gives 0.090 m
3D RMS, against 1.028 m for Direct STEC, 1.204 m for IGS GIM and 1.284 m for VTEC + Mapping —
the models sit 11–14× above that floor. ⏳ Currently on a subset of days; the full test period
is being computed.

Framing we will use: the reference STEC is the training target itself, derived from the same
observations, so this is the pipeline's own noise floor rather than reachable headroom. The
defensible statement is that **almost all remaining positioning error in this experiment is
ionospheric modelling error**, not orbit, clock or multipath. As a control, the same run
reproduces the published elevation-weighted IGS GIM arm to max |Δ| = 0.0000 m over 45
station-days.

---

## Additional correction, not raised by the reviewers

Validating the figure palette showed that the two baseline series in Figures 11, 12, 13, A1
and A2 — "VTEC + Mapping" (#ff7f0e) and "IGS GIM + Mapping" (#2ca02c) — are separated by only
ΔE = 0.7 in OKLab under simulated protanopia, i.e. they are indistinguishable for red-blind
readers. We have kept the palette for consistency with the published figures and note the
limitation; distinct per-series markers are used wherever the figure form allows.
