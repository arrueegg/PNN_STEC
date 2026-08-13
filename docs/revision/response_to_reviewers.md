# Response to Reviewers

**Manuscript:** Probabilistic Machine Learning for Slant Total Electron Content Modelling
based on GNSS — Rüegg, Mao, Pan, Orús Pérez, Soja
**Status:** draft. Numbers marked ✅ are computed and reproducible from this repository;
items marked ⏳ are still in progress.

Reproduction commands are given per item so every figure in the response can be regenerated.

---

## Framing: what the method is for

Both reviewers led with the same objection (R1.4, R2.1): daily fine-tuning uses observations
from the day being predicted, so the evaluation looks like it overstates operational
usefulness for real-time correction.

**This is a framing failure in the manuscript, and we accept it.** The method was never
intended as a real-time correction service. Daily fine-tuning mirrors the production of IGS
**final** products, which are themselves generated after the fact from the full day's
observations — the same information regime the paper's baseline (final IGS GIM) occupies. The
comparison is therefore like-for-like, but the manuscript never says so.

We will state the operating mode explicitly in the abstract, Section 3.3, Section 3.4 and the
conclusion, and present the pretrained model as the variant that *is* applicable without
same-day data. ⏳

---

## Reviewer 1

### R1.2 — attribution of the 2024 degradation ✅

> The interpretation of the model's poorer performance during the high solar activity period
> in 2024 should be more cautious.

Accepted, and the corrected analysis strengthens the paper. Absolute RMSE is not comparable
across the solar cycle, because mean STEC itself varies by a factor of 3.7. Normalising by the
mean observed STEC of each year separates the two effects:

| | span across 2014–2024 | ratio |
|---|---|---|
| Absolute RMSE | 3.8 → 14.0 TECU | ×3.7 |
| **Normalised RMSE** | 26.8 → 40.3 % | **×1.5** |
| R² | 0.804 → 0.913 | flat |

`corr(RMSE, mean STEC) = +0.954`; `corr(nRMSE, mean STEC) = −0.166`. **In relative terms 2024
is the best year in the record (26.9 %), not the worst.** The same holds for the regime split:
the temporal-extrapolation year (2024, 26.9 %) outperforms the interpolation years
(2014–2023, 31.0 %) in normalised terms.

We will therefore replace the solar-maximum attribution with the statement that absolute error
scales with ionospheric amplitude while relative skill is stable, and note that 2015 (40.3 %)
and 2017 (35.8 %) are the genuine outliers — both low-sample early years.

*New figure:* `plots/revision/revision_relative_error.png`
*Reproduce:* `python src/analysis/relative_error_metrics.py`

### R1.1 / R1.5 — split design and architecture alternatives ✅ (partly ⏳)

The interpolation (2014–2023) and extrapolation (2024) regimes are now reported separately
(RMSE 7.65 vs 14.05 TECU; R² 0.900 vs 0.875), with the normalised comparison above. ⏳ write-up

On alternatives: 729 STEC training runs were logged. Best validation MAE per architecture:

| Architecture | best val MAE | runs | runs reaching ≥20 epochs |
|---|---|---|---|
| BayesianResNetSTEC (selected) | 1.24 | 711 | 433 |
| FactorizedSTEC | 2.87 | 7 | 4 |
| BNN_NLL | 5.43 | 4 | 3 |
| MLP_NLL | 5.78 | 5 | 1 |
| ResNet_BNN_NLL | 13.57 | 2 | **0** |

We report the run counts deliberately: the search was **unbalanced**, and we will say so rather
than imply a rigorous head-to-head. The selected architecture received two orders of magnitude
more tuning effort than the alternatives.

*New figure:* `plots/revision/revision_architecture_search.png`
*Reproduce:* `python src/analysis/hyperparameter_search_summary.py`

### R1.8b — hyperparameter selection ⏳

Table 2 will be completed. It currently omits the **KL annealing schedule** (linear ramp
0 → 0.1 over 5 warm-up epochs), the predictive-variance floor, and the output-bias
initialisation to the dataset mean STEC.

### R1.8f / R1.8g — fine-tuning details

Already stated in the manuscript (Section 3.4: training stations only; Section 3.3: all
parameters updated, no layer freezing). We will make both more prominent.

---

## Reviewer 2

### R2.7 — behaviour during disturbed conditions ✅

> A method that improves average RMS but fails during disturbed periods may not be
> operationally reliable.

Directly addressed. The 2024 test period contains 39 storm days (daily minimum Dst ≤ −50 nT),
including two great storms (DOY 131–133, Dst_min = −406 nT; DOY 282–285, −333 nT). Mean 3D RMS
positioning error, applying the same 10 m outlier rule as Figure 12:

| Method | quiet [m] | storm [m] | degradation | vs GIM (quiet) | vs GIM (storm) |
|---|---|---|---|---|---|
| **Direct STEC** | 1.087 | 1.316 | +21.1 % | **+31.9 %** | **+26.3 %** |
| Pretrained Direct STEC | 1.831 | 2.685 | +46.6 % | −14.8 % | −50.5 % |
| VTEC + Mapping | 1.607 | 1.729 | +7.6 % | −0.7 % | +3.1 % |
| IGS GIM + Mapping | 1.595 | 1.785 | +11.9 % | — | — |

The Direct STEC model retains most of its margin over the operational baseline under storm
conditions. The pretrained-only variant is the one that collapses, which quantifies the
qualitative claim in Appendix A and reinforces why day-specific adaptation matters.

*New figure:* `plots/revision/revision_storm_positioning.png`
*Reproduce:* `python src/analysis/storm_stratification.py`

**Methodological note we will include:** without the 10 m exclusion already used in Figure 12,
102 of 35 652 station-days (0.29 %) dominate the quiet-period mean strongly enough to reverse
the ordering and make storms appear *easier*. The medians are unaffected.

### R2.5 — isolating the contribution of the uncertainty estimates ✅

> A stochastic-model ablation is needed to isolate the contribution of uncertainty-based
> observation weighting.

PPPx supports `weight_opt ∈ {elev, snr, iono}`; both `elev` and `iono` were run for all three
correction sources across the full test period. Restricting to station-days solved under
**both** weightings (the arms otherwise differ by several hundred station-days):

| Correction | elevation [m] | predicted uncertainty [m] | gain | uncertainty better on |
|---|---|---|---|---|
| Direct STEC | 1.156 | 1.121 | **+3.0 %** | 55.9 % of station-days |
| VTEC + Mapping | 1.580 | 1.624 | −2.7 % | 41.1 % |
| IGS GIM + Mapping | 1.630 | 1.631 | −0.1 % | 19.0 % |

**We will moderate the manuscript's claim accordingly.** Uncertainty-based weighting yields a
small but consistent gain, and only where the uncertainty is genuinely observation-level and
model-derived; it does not help the mapped baselines. The majority of the ~31 % improvement
over IGS GIM comes from the STEC correction itself rather than from the weighting.

*New figure:* `plots/revision/revision_weighting_ablation.png`
*Reproduce:* `python src/analysis/weighting_ablation.py`

⏳ A fixed-variance arm will be added to complete the requested comparison list.

### R2.2 — the Bayesian component is limited to the output layer ✅ (evidence ⏳)

Correct, and we will relabel the small epistemic component in Section 4.2 as a **limitation of
the last-layer design** rather than presenting it as a finding.

We note that a fully-Bayesian variant (`ResNet_BNN_NLL`, Bayesian weights in every residual
block) is implemented. Its two existing training runs are **not usable evidence** — both used
half the hidden width, a 10× larger learning rate, and stopped at epochs 10 and 7 of 150. ⏳ A
matched run (identical to the published configuration except `model_type`) is under way; we
will report both accuracy and the magnitude of the epistemic component. A confound we will
report explicitly: the selected architecture initialises its output bias to the dataset mean
STEC while the fully-Bayesian variant does not.

### R2.3 — comparability of STEC products ⏳

We will expand Section 2 on how the reference STEC is generated (carrier-phase levelling, arc
handling, DCB application, reference noise level). In addition, the Madrigal discrepancy will
be decomposed into a per-station-per-day offset plus residual scatter, and compared against
Madrigal's own reported per-observation uncertainty (`dlos_tec`), which the processing chain
already carries. Per-**arc** decomposition is possible on our own test set (the cycle-slip
counter is retained) but not against Madrigal, which carries no satellite identity.

### R2.4 — stratification of Figure 4 ⏳

Elevation, geomagnetic latitude, local time and season are already stratified in Figures 5–8.
The genuinely missing axis is storm/quiet, which is being added for the STEC domain (the
positioning equivalent is R2.7 above).

### R2.6 — calibration diagnostics ⏳

Accepted: monotonic association is not calibration. Coverage and σ-interval diagnostics are
already implemented; PIT histograms and CRPS are being added, for the pretrained and fine-tuned
models and under the Madrigal dataset shift.

### R2.8 — observation-derived upper bound ⏳

We will add the requested benchmark, applying the GNSS-derived reference STEC directly as the
ionospheric correction. Feasibility is confirmed: the reference STEC is available for 44–46 of
the 55 positioning stations per day. The station intersection will be stated explicitly.

### R2.7 (second part) — convergence metrics

Tail behaviour (95th percentile) and the vertical/horizontal split are already computed and
will be added to the revised Table 5. Convergence time is not meaningful for the kinematic
single-frequency PPP with daily reprocessing used here, and we will say so rather than report a
quantity the processing strategy does not support.

---

## Additional correction, not raised by the reviewers

While preparing the revision we validated the figure palette and found that the two baseline
series in Figures 11, 12, 13, A1 and A2 — "VTEC + Mapping" (`#ff7f0e`) and "IGS GIM + Mapping"
(`#2ca02c`) — are separated by ΔE = 0.7 in OKLab under simulated protanopia, i.e. they are
indistinguishable for red-blind readers. ⏳ We propose to regenerate the affected figures on a
colourblind-safe palette (Okabe-Ito, worst-pair ΔE ≥ 9.6) with distinct markers per series, so
series identity never depends on colour alone.
