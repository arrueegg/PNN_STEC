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

* **corr(offset_model, offset_gim) = +0.924** over 67 stations
* both exceed Madrigal at **93%** of stations; mean offsets +6.04 and +8.50 TECU
* removing a per-station constant drops the model's Madrigal RMSE from 15.76 to 12.06 TECU

**41% of the variance in Table 4 is a station-dependent reference offset, not model error.**
We will report Table 4 with this decomposition alongside, and expand Section 2 on how the
reference STEC is produced (carrier-phase levelling, arc handling, DCB application).

### R2.4 — stratification ✅ (conclusion revised — see the GIM correction below)
Elevation, geomagnetic latitude, local time and season are already Figures 5–8; the missing
axis was activity.

| Daily min Dst | days | Direct STEC | Pretrained | VTEC+Map | IGS GIM | Direct vs GIM |
|---|---|---|---|---|---|---|
| quiet (> −30 nT) | 165 | 6.78 | 12.69 | 8.95 | 8.30 | +18.4% |
| weak (−50 to −30) | 38 | 7.07 | 13.91 | 8.98 | 8.63 | +18.0% |
| moderate (−100 to −50) | 25 | 7.41 | 15.62 | 9.10 | 8.65 | +14.3% |
| intense (≤ −100 nT) | 14 | **8.04** | 24.26 | 9.59 | 9.02 | **+10.9%** |

**The advantage over IGS GIM narrows with disturbance — it does not widen.** An earlier version
of this document reported +18% → +34% and stated the opposite; that was an artifact of the GIM
date defect described below, and it is retracted. Two of the fourteen intense-storm days
(DOY 225, Dst −188; DOY 226, Dst −103) had been compared against the *previous* day's IONEX map,
giving IGS GIM 22.1 and 23.9 TECU instead of 8.96 and 7.85.

What survives, and is the defensible claim: Direct STEC is the most accurate model in every
activity bin and degrades least from quiet to intense (+19%, against +9% for VTEC + Mapping,
+9% for IGS GIM and +91% for the pretrained-only variant). Across F10.7 terciles the margin is
+20/+15/+17%.

⏳ The quiet row still contains 8 days whose GIM is being recomputed; they will move the quiet
margin down by roughly 1.5 points, slightly *reducing* the remaining spread across bins. The
intense, moderate and weak rows are final. ⏳ The equivalent stratification of Figure 4 itself
(pretrained model) is still being computed.

### R2.6b — our uncertainty against the GIM products' own ✅
Not raised by either reviewer, but it is the comparison the title invites. The existing evidence
only benchmarks the predicted uncertainty against *no* uncertainty (a constant σ, elevation
weighting). IGS and CODE ship an RMS map in the same IONEX file, so it can be benchmarked
against a real operational uncertainty. Each product is scored against **its own** residuals,
on 43 test days (81.3 M observations):

| | RMSE [TECU] | 95% coverage | σ scale for nominal | CRPS skill over constant σ | Spearman(σ, \|error\|) |
|---|---|---|---|---|---|
| Direct STEC | **7.31** | **88.7%** | **×1.42** | **+10.5%** | **0.41** |
| CODE GIM + Mapping | 8.40 | 73.7% | ×2.02 | +1.5% | 0.39 |
| IGS GIM + Mapping | 8.49 | 46.9% | ×4.60 | **−8.1%** | 0.30 |

The IGS combined RMS is *worse than a single constant* as a per-observation uncertainty — a
negative skill score. CODE's is far better dispersed but adds almost nothing over a constant.
Ours is the only one that both approaches nominal coverage and earns a positive score.

Three caveats to state rather than have an editor find: the IGS combined RMS reflects the
spread among contributing analysis centres, not a validated error estimate; mapping-function
error is not represented in it at all; and it is a 5°/2 h grid-cell quantity being judged
per observation. It is nonetheless the uncertainty a user of the product actually receives.

### R2.5 — stochastic-model ablation ✅ (one arm ⏳)
Restricting to station-days solved under both weightings:

| Correction | elevation [m] | fixed variance [m] | predicted uncertainty [m] | iono vs elev |
|---|---|---|---|---|
| Direct STEC | 1.226 | **1.367** | 1.195 | **+2.6%** |
| VTEC + Mapping | 1.580 | — | 1.624 | −2.7% |
| IGS GIM + Mapping | 1.630 | — | 1.631 | −0.1% |

**The fixed-variance arm is the one that makes the point.** Replacing the predicted
per-observation sigma with a constant — same STEC values, same `weight_opt iono`, so PPPx still
weights by the uncertainty column — makes positioning **11.5% worse than elevation weighting**,
while the model's own uncertainty makes it 2.6% better. A 14-point spread between "weight by a
constant" and "weight by our uncertainty" is direct evidence that the gain comes from the sigma
*varying per observation* in a way that tracks real error, not from having an uncertainty column
at all. 242 days, 5,422 station-days solved under all three arms.

Note on the sample: pairing across three arms rather than two shrinks the Direct STEC set from
8,170 to 5,422 station-days and moves the iono-vs-elev figure from +3.0% to +2.6%. Both are
reported in `paired.csv` (`gain_iono_%` and `gain_iono_two_arm_%`); the +3.0% quoted previously
is the two-arm number and remains correct on its own sample.

**We will moderate the manuscript's claim accordingly.** Uncertainty weighting yields a small
but consistent gain, and only where the uncertainty is genuinely observation-level and
model-derived; the majority of the ~31% improvement over IGS GIM comes from the STEC
correction itself.

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

### R2.8 — observation-derived upper bound ⏳ provisional
Applying the GNSS-derived reference STEC directly as the ionospheric correction gives **0.128 m**
mean 3D RMS, against 1.149 m for Direct STEC, 1.336 m for IGS GIM and 1.497 m for VTEC + Mapping
— the models sit **9–12×** above that floor.

⏳ These are provisional and have moved as days accumulate (at 9 days the floor read 0.216 m,
at 48 days 0.128 m). Do not quote them until the run covers the full period.

**Three reasons these numbers do not line up with Table 5, none of them an error.** The oracle
run uses **elevation** weighting where Table 5 uses predicted-uncertainty weighting (over all
242 days that is 1.165 vs 1.123 m for Direct STEC); it is restricted to **station-days solved by
all four methods**, currently 1,232 of 11,737 seen for at least one; and it covers 48 days, not
242. Comparisons must be drawn inside this table, never across to Table 5.

Framing we will use: the reference STEC is the training target itself, derived from the same
observations, so this is the pipeline's own noise floor rather than reachable headroom. The
defensible statement is that **almost all remaining positioning error in this experiment is
ionospheric modelling error**, not orbit, clock or multipath. As a control, the same run
reproduces the published elevation-weighted IGS GIM arm to **max |Δ| = 0.0000 m over 1,560
shared station-days**, which is what rules out the pipeline changes as a cause of any difference.

---

## Correction to the published IGS GIM baseline

Building the prediction store surfaced a defect in how the IGS GIM comparison was computed, and
it changes Table 4 and one of the stratified conclusions. It is reported here in full because
the corrected numbers are slightly *less* favourable to the paper.

**What happened.** `compare_stec_vtec_gim.py` selected the IONEX file using the `doy` carried in
the results frame. That value is a denormalised model *input*: day of year is scaled to
(doy−1)/365 and inverted in float32, which for 26 days of the year returns a value just below the
integer — DOY 189 comes back as 188.99998. The truncating `int()` cast then loaded the **previous
day's** global map. Twelve days of the 242-day 2024 test period are affected: **DOY 184–189 and
225–230**.

**What it does and does not touch.**

* The model's own predictions are unaffected — the network consumed the normalised value
  directly and never round-tripped it.
* The VTEC + Mapping baseline is unaffected; it involves no date lookup.
* **All positioning results are unaffected** — Table 5, Figures 12/13/A1/A2 and the ~31%
  improvement claim. The positioning pipeline takes the day from its `--date` argument
  (`run_positioning_evaluation.py`), never from a data frame. This was verified explicitly.
* Only the IGS GIM column of the STEC-domain comparison is wrong, on those 12 days.

**Corrected numbers.** Table 4's IGS GIM entry moves from 8.56 to **≈8.31 TECU** on the own test
set, so the Direct STEC advantage over IGS GIM falls from 19.1% to **≈16.7%**. On Madrigal the
GIM entry moves from 15.64 to ≈15.50 TECU (advantage 6.1% → 5.2%). Four of the twelve days are
recomputed exactly; the remaining eight are projected at the median unaffected daily RMSE and
will be exact once the store covers them.

The knock-on is the R2.4 activity stratification, whose conclusion **reverses** — see that
section.

**Fixes.** The truncating cast is corrected at both sites
([src/compare_stec_vtec_gim.py:316](../../src/compare_stec_vtec_gim.py#L316) and
[src/evaluation/prediction_store.py:176](../../src/evaluation/prediction_store.py#L176), where
the day is now taken from the caller rather than the frame), and
`src/analysis/repair_gim_baseline.py` recomputes the stored days. As a check, that script
reproduces the stored GIM column to 1.5×10⁻⁵ TECU on all 77 unaffected day-datasets.

## Additional correction, not raised by the reviewers

Validating the figure palette showed that the two baseline series in Figures 11, 12, 13, A1
and A2 — "VTEC + Mapping" (#ff7f0e) and "IGS GIM + Mapping" (#2ca02c) — are separated by only
ΔE = 0.7 in OKLab under simulated protanopia, i.e. they are indistinguishable for red-blind
readers. We have kept the palette for consistency with the published figures and note the
limitation; distinct per-series markers are used wherever the figure form allows.
