> # ⚠ SUPERSEDED — DO NOT CITE THESE NUMBERS
>
> This analysis compares a run that was **not a clean ablation**. `ResNet_BNN_NLL` lacked the
> output-layer initialisation `BayesianResNetSTEC` has always had:
>
> ```python
> self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)   # 15.5
> self.output_layer.weight_mu.normal_(0, 0.01)
> ```
>
> So every number below measures **the architecture plus that omission**. The −1.93 TECU
> pervasive bias identified in §5 is the fingerprint of an uncorrected output offset, not of
> Bayesian residual blocks. The claim in the opening paragraph that both were trained "with
> identical hyperparameters" is false in exactly the way that matters.
>
> A re-run with matched initialisation is in progress (`fb-retrain`, from 2026-08-23 15:00).
> Both architectures are now verified identical at initialisation from the same seed.
>
> **What survives from this document**: the two refutations, which do not depend on the
> initialisation — the KL-weight confound (`BKLLoss` uses `reduction="mean"`, so the
> fully-Bayesian model's KL share is *smaller*, 0.24% against 1.02%) and undertraining (its
> best checkpoint was epoch 115; epochs 131–136 were worse). Both were measured directly and
> remain valid.
>
> Note also: the predictions this analysis read were later found to have been written into
> `pretrained_stec/own`, overwriting the paper model's partition. They now live at
> `predictions/pretrained_stec_resnet_bnn_nll/`.

# R2.2 — Does a fully Bayesian network improve on the Bayesian-head-only paper model?

Evidence for the reviewer question "why not make the whole network Bayesian instead of only the
output layer?" Compares the paper model `BayesianResNetSTEC` (Bayesian output head only) against
`ResNet_BNN_NLL` (Bayesian residual blocks + head), both pretrained with identical
hyperparameters, seed 42, on the same 10,000,000-observation test set
(`predictions/pretrained_stec/own/`, 544 day-files, 2014–2024).

All numbers below were read from files already on disk — `test_metrics/performance_metrics.txt`,
`test_metrics/uncertainty_analysis/`, `test_metrics/spatial_analysis/`,
`test_metrics/feature_analysis/`, `test_metrics/temporal_analysis/`, `loss_history.csv`, and the
two `.pth` checkpoints (parameter-space only, no forward pass) — under:

```
experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/
experiments/Pretrain_STEC_ResNet_BNN_NLL_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI/
```

No training, no inference, no GPU use, no prediction-store streaming were performed for this
analysis. The two checkpoints (102 MB / 203 MB) were loaded once each, on CPU, purely to
instantiate the saved weights and evaluate torchbnn's closed-form KL divergence — no data was
read for that step.

---

## 1. Answer to R2.2

Making the residual blocks Bayesian as well as the head does not improve the model — it makes it
categorically worse (RMSE 11.67 → 19.74 TECU, R² 0.897 → 0.707) while producing an uncertainty
that is both far larger and far less calibrated (68.3%-nominal 1σ interval covers 60.9% of
observations for the paper model but 94.1% for the fully Bayesian one). The paper's design
choice — Bayesian output layer only, deterministic backbone — is empirically justified by this
ablation, and the degradation is broad-based (worse everywhere, roughly 1.7–2× across elevation,
local time and the 2014–2024 solar cycle) rather than confined to some fixable corner of the
input space, with one exception: it is disproportionately worse at low/mid geomagnetic latitudes.

---

## 2. Evidence table

| | `BayesianResNetSTEC` (paper model) | `ResNet_BNN_NLL` (fully Bayesian) | Ratio |
|---|---|---|---|
| final val loss (checkpoint epoch) | 2.0848 (epoch 148) | 3.5835 (epoch 115) | — |
| final val loss (last logged epoch) | 2.0876 (epoch 150) | 3.6693 (epoch 136) | — |
| RMSE | 11.6716 | 19.7355 | 1.69× |
| MAE | 7.1635 | 13.0620 | 1.82× |
| bias (mean error) | +0.4328 | −1.9304 | sign flip |
| R² | 0.8974 | 0.7068 | — |
| correlation | 0.9475 | 0.8430 | — |
| mean predicted uncertainty | 7.1396 | 32.5918 | 4.57× |
| uncertainty–error correlation | 0.5682 | 0.5448 | 0.96× |
| 1σ coverage (nominal 68.3%) | 60.9% (−7.4 pt) | 94.1% (+25.8 pt) | — |
| 2σ coverage (nominal 95.5%) | 88.8% (−6.7 pt) | 99.6% (+4.1 pt) | — |
| 3σ coverage (nominal 99.7%) | 97.0% (−2.7 pt) | 99.9% (+0.2 pt) | — |

Source: `test_metrics/performance_metrics.txt` and `test_metrics/uncertainty_analysis/sigma_coverage_comparison.png`
in each experiment directory; both match the numbers given in the task brief exactly.

**Checkpoint provenance caveat.** The saved checkpoint (the one `test_metrics/` was computed
from, confirmed via `logs/r22_eval.log`: *"Loading single model: pretrain_ResNet_BNN_NLL_seed42.pth"*)
is the best-val-loss checkpoint under early stopping, not necessarily the last epoch. For
`BayesianResNetSTEC` the two coincide closely (epoch 148 val 2.0848 vs. epoch 150 val 2.0876). For
`ResNet_BNN_NLL` they do not: the saved checkpoint is epoch 115 (val 3.5835), while training
continued to epoch 136 and the loss quoted in the task brief (3.669) matches epoch 136's *last
logged* val loss, not the checkpoint actually used to produce the reported RMSE/MAE/R². This
doesn't change any conclusion here — the checkpoint used for `test_metrics/` is the one analysed
throughout this document — but it means "final val loss" in the task table is not the loss of the
model the other rows describe. Worth fixing before this ablation goes in a table for the paper.

---

## 3. KL-confound assessment

**Parameter count.** Counted directly from the loaded `state_dict`s:

| | Bayesian scalar parameters | Deterministic scalar parameters |
|---|---|---|
| `BayesianResNetSTEC` | 2,050 (output head only: 1024×2 weights + 2 biases) | 8,527,872 |
| `ResNet_BNN_NLL` | 8,398,850 (4 residual blocks × 2 `BayesLinear` × (1024×1024+1024), plus the 2,050-parameter head) | 131,072 |

The fully Bayesian model has **4,097× more Bayesian parameters** — the reviewer's premise is
correct on this point.

**But the loss does not sum over parameters — it averages.** `src/utils/loss_function.py:221`
constructs `bnn.BKLLoss(reduction="mean", last_layer_only=False)`, and torchbnn's
`bayesian_kl_loss` (`torchbnn/functional.py:64-65`) with `reduction='mean'` returns
`kl_sum / n`, where `n` is the total number of Bayesian scalar parameters. This mean, not the
raw sum, is what gets multiplied by `kl_weight` and added to the NLL term
(`train_manager.py:110`: `loss = nll_loss + current_kl_weight * kld_loss`). A larger parameter
count does not mechanically inflate this term — it is normalised away by construction.

Evaluating this analytic KL divergence directly on the two saved checkpoints (CPU, no data,
`torchbnn.functional.bayesian_kl_loss(model, reduction=...)`):

| | KL, sum reduction | KL, **mean reduction (what training used)** |
|---|---|---|
| `BayesianResNetSTEC` | 437.56 | **0.2134** |
| `ResNet_BNN_NLL` | 713,133.75 | **0.0849** |

Under sum reduction the fully Bayesian model's KL term is 1,630× larger — a textbook confound, if
that were what the loss used. Under the mean reduction actually used, it is **smaller**: 0.0849
vs. 0.2134, a 0.40× ratio, the opposite direction from the reviewer's hypothesis. At the trained
`kl_weight = 0.1`, the KL term's contribution to the total loss at convergence is:

- `BayesianResNetSTEC`: 0.1 × 0.2134 = 0.02134, i.e. **1.02%** of the 2.085 total val loss.
- `ResNet_BNN_NLL`: 0.1 × 0.0849 = 0.00849, i.e. **0.24%** of the 3.583 total val loss (using
  the checkpoint's own val loss, 3.5835).

**Conclusion: the confound as usually framed — "more Bayesian parameters means the KL term
dominates the loss at the same weight" — is not what happened here, and is directly refuted by
the trained weights.** The KL penalty is a smaller fraction of the objective for the fully
Bayesian model, not a larger one.

**What this does not rule out.** There is a different, unmeasured effect that the loss-magnitude
argument doesn't capture: `reduction="mean"` keeps the *scalar loss value* comparable across
architectures, but every one of the 8.4M Bayesian parameters in `ResNet_BNN_NLL` still samples
its own weight via the reparameterization trick on every forward pass (training and the T=100 MC
eval, confirmed at `src/inference_testset.py:169`: `num_mc_samples = 100 if is_bayesian else 1`
applies to both models identically). In `BayesianResNetSTEC` only the 2-output head samples; the
four residual blocks are deterministic. This means `ResNet_BNN_NLL`'s entire backbone injects
sampling noise into every activation at every layer, which is a capacity/optimization-dynamics
effect distinct from the KL loss magnitude — and one this analysis has not measured (would need,
e.g., per-layer activation variance under repeated forward passes with fixed input, or a
training re-run with a KL weight scaled to the fully Bayesian parameter count). Flagged, not
resolved — see §6.

**Per-epoch loss components are not available for either run.** `TrainingUtils.save_final_losses`
(`src/training/training_utils.py`) writes only `epoch, train_loss, val_loss` to
`loss_history.csv`, even though `train_manager.py` computes `running_mse`, `running_nll` and
`running_kld` separately every epoch (lines 124-127) — they are simply never persisted.
`wandb/` does log `train_kld`/`val_kld`/`kl_weight` per epoch (confirmed in
`wandb/run-20260630_174354-9ckytbnn/files/wandb-summary.json`), but no wandb run in this
repository's `wandb/` directory (1,529 runs checked) matches either of these two *exact* pretrain
configurations (`hidden_dim=1024, num_layers=4, prior_sigma=0.1, loss_weight=0.1`) — the closest
`BayesianResNetSTEC` pretrain runs use `prior_sigma=0.05`, and the closest `ResNet_BNN_NLL`
pretrain runs use `hidden_dim=512`. The point-in-time analytic KL above (computed from the
converged checkpoint weights) is therefore the only measurement of this term available for the
actual compared runs; the training-trajectory shape of the KL term (in particular during the
5-epoch annealing warmup) cannot be reconstructed from what's on disk.

---

## 4. Calibration versus accuracy

Larger uncertainty is not simply "more honest" here — it is measurably worse-calibrated, in the
opposite direction from the paper model's small existing miscalibration.

From `uncertainty_analysis/sigma_coverage_comparison.png` in each experiment:

- **`BayesianResNetSTEC` under-covers modestly at every level** (60.9% vs. 68.3% expected at 1σ,
  88.8% vs. 95.5% at 2σ, 97.0% vs. 99.7% at 3σ) — intervals are mildly too narrow
  (overconfident), gap shrinking from −7.4 to −2.7 percentage points as σ grows.
- **`ResNet_BNN_NLL` over-covers severely at 1σ** (94.1% vs. 68.3% expected, +25.8 points) and
  the gap only narrows at 2σ/3σ because coverage is bounded at 100% (a ceiling effect, not
  improving calibration) — 99.6% vs. 95.5%, 99.9% vs. 99.7%.
- The **epistemic-uncertainty share** of the reliability curve makes the mechanism visible: for
  `BayesianResNetSTEC`, epistemic coverage is only 9.4%/18.7%/27.5% at 1σ/2σ/3σ — the tiny
  2,050-parameter head contributes almost nothing, and total coverage is carried almost entirely
  by the aleatoric (data-noise) term (60.4%/88.5%/96.8%, essentially identical to total). For
  `ResNet_BNN_NLL`, epistemic coverage jumps to 81.3%/96.7%/99.3% — with 8.4M sampled
  parameters spread through the backbone, parameter-sampling noise becomes a dominant, not
  negligible, contributor to predictive variance, and it dominates *upward*.

`uncertainty_analysis/binned_uncertainty_error_analysis_simplified.png` makes the practical
consequence concrete: for `BayesianResNetSTEC`, mean absolute error tracks predicted uncertainty
almost exactly (the two curves are visually coincident from 0–18 TECU) — this is close to ideal
per-bin calibration. For `ResNet_BNN_NLL`, predicted uncertainty is systematically and
substantially above mean absolute error across the *entire* observed range (e.g. at predicted
σ≈20 TECU, actual MAE≈2 TECU; at predicted σ≈45 TECU, actual MAE≈30 TECU), and — notably — no
observation in the test set has predicted uncertainty below ~20 TECU, so confident, low-error
predictions cannot be flagged as such at all.

**The ranking ability (uncertainty–error correlation) is nearly unchanged**: 0.568 vs. 0.545, a
4% relative difference. So `ResNet_BNN_NLL` still knows *which* predictions are relatively harder
than others about as well as the paper model does — it has simply lost the ability to report that
in correctly-scaled absolute units, and lost it badly (a 4.57× mean-uncertainty inflation against
a 1.69× RMSE inflation). This is not "larger but still useful" uncertainty; it is a scale failure
layered on top of a real accuracy loss, and the two are separable in the data.

---

## 5. Where the degradation concentrates

Per-observation predictions for `ResNet_BNN_NLL` are **not in the parquet store**
(`predictions/` holds only `finetuned_stec` and `pretrained_stec`, both of which are the paper
model — `ls predictions/` confirms no third variant directory exists), and no
`detailed_predictions.csv`-equivalent exists in the experiment directory either. This analysis
therefore uses the pre-aggregated stratified plots each pretrain run's own evaluation pipeline
already produced from the same 10M-row test set (`test_metrics/feature_analysis/`,
`spatial_analysis/`, `temporal_analysis/`), not the store.

- **Elevation** (`feature_analysis/residual_vs_satele_boxplot.png`): both models show the
  expected monotonic decline in error from low to high elevation (longer slant paths are harder).
  The ratio between the two models is close to constant across the whole range — MAE ratio ≈1.8
  at 5–10° (11.0→20.0) down to ≈2.0 at 85–90° (4.3→8.7) — i.e. elevation is **not** where the
  fully Bayesian model disproportionately fails; the degradation scales roughly uniformly.
- **Local solar time** (`feature_analysis/residuals_vs_local_time.png`): both models show the
  same mild diurnal modulation (higher error near local noon, consistent with daytime TEC
  variability), again scaled by a roughly constant factor (`BayesianResNetSTEC` MAE 6.5–7.8,
  `ResNet_BNN_NLL` MAE 12.1–14.1, ratio ≈1.8 throughout). Not a concentration point.
- **Year / solar cycle** (`temporal_analysis/residuals_vs_date.png`, and per-year text summaries
  available only for `BayesianResNetSTEC` — see caveat below): both models track the same
  physically-driven shape — low error during the 2018–2020 solar minimum
  (`BayesianResNetSTEC` RMSE 3.8–4.5 TECU), rising sharply from 2021 toward the Solar Cycle 25
  maximum in 2023–2024 (RMSE 9.2→14.0 TECU). `ResNet_BNN_NLL`'s monthly RMSE/MAE curve has the
  identical shape, scaled up by roughly the same 1.7–2× factor seen elsewhere. Again, not a
  concentration point — this is a shared, physically expected pattern, not an architecture-driven
  failure mode.
- **Solar-magnetic latitude** (`spatial_analysis/mLat_summary.png`) **is the one genuine
  concentration point.** `BayesianResNetSTEC`'s MAE is fairly flat across latitude (roughly
  5–11 TECU, with a mild peak of ~11 TECU near +15° mLat, the equatorial ionisation anomaly
  region). `ResNet_BNN_NLL`'s MAE is markedly non-uniform: ~14–19.6 TECU across low/mid
  latitudes (roughly −60° to +20° mLat) against ~10–14 TECU beyond ±60°, i.e. the equatorial
  degradation is proportionally larger for the fully Bayesian model (≈1.8–2.5× worse than the
  paper model at the equator vs. ≈1.5–1.9× worse at the poles). This region is exactly where
  ionospheric electron content is most variable, so a model whose backbone is now itself a noisy,
  high-variance estimator loses more capacity precisely where the target function is hardest.
- **A systematic bias `BayesianResNetSTEC` does not show**: every stratified plot for
  `ResNet_BNN_NLL` (elevation, local time, magnetic latitude, date) shows a consistent negative
  median residual (roughly −2 to −8 TECU depending on the bin) — i.e. pervasive underprediction —
  matching the global bias of −1.93 TECU in `performance_metrics.txt`. `BayesianResNetSTEC`'s
  median residual sits at or near zero in essentially every bin (global bias +0.43 TECU). This
  bias is present everywhere, not concentrated in one regime, and is consistent with the
  optimizer settling on a systematically lower-STEC solution when regularizing a much larger
  fraction of the network toward a zero-mean prior.

**Caveat on this section**: `ResNet_BNN_NLL`'s `test_metrics/temporal_analysis/` directory holds
only 3 plots (`residuals_vs_date`, `residual_vs_doy_boxplot`, `year_month_summary`) — it is
missing the `total_metrics_summary.txt`, `season_*_metrics_summary.txt`,
`month_*_metrics_summary.txt` and `year_*.0_metrics_summary.txt` files that
`BayesianResNetSTEC`'s directory has. The per-year numbers quoted above for
`BayesianResNetSTEC` (e.g. RMSE 3.81 TECU in 2020, 14.05 TECU in 2024) are exact, read from
those text files; the equivalent numbers for `ResNet_BNN_NLL` were read off the plot axes by eye,
not from text, because the underlying evaluation run for that model produced a smaller set of
artifacts. This asymmetry in what each run's own pipeline computed is itself worth noting — it
means the two evaluations are not bit-for-bit parallel, even though the plots that do exist for
both cover the same 10M-row test set.

---

## 6. What this cannot settle

- **Whether a KL weight tuned to the fully Bayesian parameter count would close the gap.** The
  magnitude confound (§3) is refuted for the loss *value*, but the reparameterization-noise
  effect (every residual-block activation, not just the head, is now stochastic on every forward
  pass) is real, structurally present, and unmeasured here. Settling this needs an actual
  retrain of `ResNet_BNN_NLL` — this analysis was explicitly barred from doing that (no
  training, no GPU) — ideally as a small sweep over `kl_weight` and/or `prior_sigma` for the
  fully Bayesian variant, holding everything else fixed, to see whether accuracy and calibration
  recover as the effective per-layer regularization strength is retuned.
- **Whether the training trajectory (not just the converged endpoint) tells a different story.**
  §3's KL numbers come from the two saved checkpoints' final weights. No per-epoch NLL/KLD
  breakdown survives on disk for either exact run (loss_history.csv doesn't store it; no
  matching wandb run exists), so whether the KL term's *relative* weight behaved differently
  during the 5-epoch annealing warmup — when it could plausibly have had an outsized early
  effect on a network with 4,097× more Bayesian parameters to pull toward the prior
  simultaneously — cannot be reconstructed. A rerun with wandb logging enabled (both configs
  already set `wandb.offline: false`) would resolve this without needing new architectural code.
- **Whether the equatorial-latitude concentration (§5) is specific to this architecture or an
  artifact of a single seed.** Only seed 42 was trained for either model; there is no way from
  what's on disk to tell whether the geomagnetic-latitude pattern is a repeatable property of
  making the backbone Bayesian, or noise from one training run. The same caveat applies to every
  number in this document — no replicate exists for either architecture.
- **Per-observation, store-backed stratification for `ResNet_BNN_NLL`.** Everything in §5 comes
  from the pre-aggregated plots each experiment's evaluation pipeline happened to produce, not
  from re-querying per-observation predictions (none are stored for this model variant — see the
  gap noted in §5). A finer stratification (e.g. joint elevation × magnetic-latitude bins, or
  per-station breakdowns) would require either adding `ResNet_BNN_NLL` to the prediction store's
  supported variants and re-running inference, or reading the existing `test_metrics/` PNGs more
  finely than their bin resolution allows — this analysis did not do either.
- **The checkpoint-vs-"final"-epoch mismatch noted in §2** should be resolved (confirm which
  epoch the task brief's "3.669" was computed from, and whether `test_metrics/` for
  `ResNet_BNN_NLL` should be regenerated from the true final-epoch weights rather than the
  early-stopping checkpoint) before any of these numbers go into a table intended for the
  response to reviewers.
