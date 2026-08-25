# R1.2 — Does a fully Bayesian network improve on the Bayesian-head-only paper model?

**Numbering note (2026-08-25):** this analysis was originally labelled "R2.2" throughout
(including by `CLAUDE.md`, now fixed); in the response letter's actual numbering the
fully-Bayesian question is **R1.2** — R2.2 is a different, closed comment (2024
solar-maximum-degradation attribution). Earlier references to "the R2.2 analysis" elsewhere in
this codebase mean this document. The filename `r22_fully_bayesian_analysis.md` is left as-is
since other documents already link to it by path.

Evidence for the reviewer question "why not make the whole network Bayesian instead of only the
output layer?" Compares the paper model `BayesianResNetSTEC` (Bayesian output head only) against
`ResNet_BNN_NLL` (Bayesian residual blocks + head), both pretrained from the same seed (42) with
**matched output-layer initialisation** and otherwise identical hyperparameters, on the same
10,000,000-observation test set (`predictions/pretrained_stec/own/` and
`predictions/pretrained_stec_resnet_bnn_nll/own/`, 544 day-files, 2014–2024).

This is the corrected re-run. A first comparison (2026-08-23 14:23 eval) used a `ResNet_BNN_NLL`
checkpoint that never received the output-layer initialisation `BayesianResNetSTEC` has always
had, so it measured the architecture plus that omission. It is kept below in §7 as the superseded
predecessor. `fb-retrain` (`logs/fb_retrain.log`, 2026-08-23 17:00 → 2026-08-24 03:28) trained a
new `ResNet_BNN_NLL` checkpoint with the initialisation fixed; `logs/r22_fully_bayesian_eval.log`
(2026-08-24 08:17 → 09:24) evaluated it on the same test set. Every number below is either read
directly from files this run produced, or computed directly from them (see §6 for method).

---

## 1. Answer to R1.2

Two things, read together, are the honest answer.

**Fixing the initialisation closed about half the RMSE gap.** The first (confounded) comparison
put `ResNet_BNN_NLL` at RMSE 19.7355 against the paper model's 11.6716 — a gap of 8.06 TECU. With
matched initialisation the gap is 3.87 TECU (RMSE 15.5389), a 52% reduction. So roughly half of
what the first comparison attributed to "making the backbone Bayesian" was actually the missing
initialisation, and half is architecture. **The fully Bayesian model is still substantially less
accurate** — RMSE 1.33× the paper model's, MAE 1.37×, R² 0.818 against 0.897 — this is not a wash.

**The uncertainty–error correlation now marginally favours the fully Bayesian model** (0.5752
against the paper model's 0.5682), a reversal from the confounded run, where it was worse
(0.5447). The reviewer's stated motivation for asking about a fully-Bayesian variant is that
last-layer-only Bayesian collapses epistemic uncertainty; ranking ability (does the model flag the
*right* observations as uncertain) is the part of that question pooled RMSE doesn't answer. On
that specific question, the fully Bayesian model is now at least as good, not worse. What it does
not buy is calibration: mean predicted uncertainty is 2.74× the paper model's (19.57 vs 7.14 TECU)
against a 1.33× RMSE increase, so the uncertainty scale is inflated well past what the accuracy
loss would justify.

**The honest R1.2 answer is therefore not "last-layer is simply better" — it is "last-layer is
substantially more accurate; the fully-Bayesian variant does not buy better uncertainty ranking
either, and loses absolute calibration."** Both halves of that sentence are now supported with
matched initialisation, where before only the second half was measurable at all.

---

## 2. Evidence table

| | `BayesianResNetSTEC` (paper model) | `ResNet_BNN_NLL`, matched init (corrected) | `ResNet_BNN_NLL`, no init fix (§7, superseded) |
|---|---|---|---|
| RMSE | 11.6716 | 15.5389 (1.33×) | 19.7355 (1.69×) |
| MAE | 7.1635 | 9.8477 (1.37×) | 13.0620 (1.82×) |
| bias (mean error) | +0.4328 | −1.4884 | −1.9304 |
| R² | 0.8974 | 0.8182 | 0.7068 |
| correlation | 0.9475 | 0.9061 | 0.8430 |
| mean predicted uncertainty | 7.1396 | 19.5673 (2.74×) | 32.5918 (4.57×) |
| uncertainty–error correlation | 0.5682 | **0.5752 (1.01×, favours FB)** | 0.5448 (0.96×) |
| 1σ coverage, total (nominal 68.3%) | 60.9% (−7.4 pt) | 90.3% (+22.0 pt) | 94.1% (+25.8 pt) |
| 2σ coverage, total (nominal 95.5%) | 88.8% (−6.7 pt) | 99.3% (+3.8 pt) | 99.6% (+4.1 pt) |
| 3σ coverage, total (nominal 99.7%) | 97.0% (−2.7 pt) | 99.9% (+0.2 pt) | 99.9% (+0.2 pt) |

Source for the first two columns: `test_metrics/performance_metrics.txt` in each experiment
directory, and the 1σ/2σ/3σ coverage computed exactly from the prediction store (§6) — both match
the task brief's numbers exactly. The superseded column is carried over from the first analysis
(read off `sigma_coverage_comparison.png` by eye — see §7's caveat on precision).

---

## 3. What changed between the two `ResNet_BNN_NLL` runs

Both `output_layer.bias_mu[0]` and `output_layer.weight_mu` (and, for `ResNet_BNN_NLL`, the
identical initialisation on every residual block's `BayesLinear`) are now set the same way
`BayesianResNetSTEC` has always initialised its head:

```python
self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)   # 15.5
self.output_layer.weight_mu.normal_(0, 0.01)
```

`src/model/model.py:219-220` confirms this is now present for `ResNet_BNN_NLL`. The two configs
(`experiments/Pretrain_STEC_BayesianResNetSTEC_.../config.yaml` and
`experiments/Pretrain_STEC_ResNet_BNN_NLL_.../config.yaml`) differ only in `model_type`,
`output_dir`/`pretrain_folder` (as they must), and `num_workers` (12 vs 4) — everything that
affects training dynamics (`random_seed: 42`, `prior_sigma: 0.1`, learning rate, batch size,
scheduler, KL annealing) is identical. `fb-retrain` ran 111 epochs (`loss_history.csv`), reaching
its best validation loss (2.9552) at epoch 91 and stopping 20 epochs later with no further
improvement (`patience: 20` in `config.yaml`, matching "Early stopping after 20 epochs without
improvement" in `logs/fb_retrain.log`). The evaluated checkpoint
(`model/pretrain_ResNet_BNN_NLL_seed42.pth`, confirmed via `logs/r22_fully_bayesian_eval.log`:
*"Loading single model: pretrain_ResNet_BNN_NLL_seed42.pth"*) is that epoch-91 checkpoint.

This run also wrote per-observation predictions to the store
(`predictions/pretrained_stec_resnet_bnn_nll/own/`, 544 files, confirmed in
`logs/r22_fully_bayesian_eval.log`) — the first analysis's checkpoint predictions never reached
the store under a variant of their own before being overwritten (see `STATE.md`'s account of the
partition contamination this caused and its repair). That store partition is what §6's coverage
numbers are computed from.

---

## 4. KL-confound assessment (refutation, recomputed on the corrected checkpoint)

**Parameter count is unchanged** — the architecture didn't change, only the initial weights, so
these counts are identical to the first analysis:

| | Bayesian scalar parameters | Deterministic scalar parameters |
|---|---|---|
| `BayesianResNetSTEC` | 2,050 (output head only: 1024×2 weights + 2 biases) | 8,527,872 |
| `ResNet_BNN_NLL` | 8,398,850 (4 residual blocks × 2 `BayesLinear` × (1024×1024+1024), plus the 2,050-parameter head) | 131,072 |

The fully Bayesian model still has **4,097× more Bayesian parameters** — the reviewer's premise on
parameter count is correct.

**The loss still averages, not sums.** `src/utils/loss_function.py:221` constructs
`bnn.BKLLoss(reduction="mean", last_layer_only=False)`; torchbnn's `bayesian_kl_loss` with
`reduction='mean'` returns `kl_sum / n` over all Bayesian scalar parameters, which is what gets
multiplied by `kl_weight` and added to the NLL term. Recomputing the closed-form KL divergence
directly from each checkpoint's `weight_mu`/`weight_log_sigma` tensors against the trained prior
(`N(0, 0.1²)`, matching `prior_sigma: 0.1` in both configs — CPU only, no data, same method as the
first analysis) on the **corrected** checkpoint:

| | KL, sum reduction | KL, **mean reduction (what training used)** |
|---|---|---|
| `BayesianResNetSTEC` (unchanged checkpoint) | 437.56 | **0.2134** |
| `ResNet_BNN_NLL`, matched init | 854,003.44 | **0.1017** |

At the trained `kl_weight = 0.1`, the KL term's share of total val loss at convergence:

- `BayesianResNetSTEC`: 0.1 × 0.2134 = 0.02134, i.e. **1.02%** of 2.0848.
- `ResNet_BNN_NLL`, matched init: 0.1 × 0.1017 = 0.01017, i.e. **0.34%** of 2.9552 (the
  checkpoint's own val loss).

Under mean reduction the fully Bayesian model's KL share is still **smaller** than the paper
model's (0.477× on the mean-KL value itself, and its share of the total loss dropped further,
0.34% vs the superseded run's 0.24%, because the corrected checkpoint's val loss is lower). The
conclusion is unchanged and, if anything, restated more cleanly by the fix: a larger Bayesian
parameter count does not inflate this loss term, because `reduction="mean"` normalises it away by
construction. What this does not rule out — the reparameterization-noise effect of every residual
activation now being stochastic on every forward pass, distinct from the KL loss magnitude — is
still unmeasured and carried forward to §8.

---

## 5. Undertraining assessment (refutation, recomputed on the corrected run)

The corrected `ResNet_BNN_NLL` run was not cut short: it reached its best validation loss (2.9552)
at epoch 91, then trained **20 further epochs (92–111) with no improvement** before early stopping
triggered (`patience: 20`), ending at val loss 3.1076 — worse than the checkpoint actually used.
This is a more direct refutation than the first analysis had available, because it comes from the
early-stopping criterion itself rather than from observing that later epochs happened to be worse:
the run explicitly searched 20 epochs past the optimum and found nothing better.

---

## 6. Calibration versus accuracy

Larger uncertainty is not simply "more honest" here — it is measurably worse-calibrated, and the
corrected run isolates *where* the miscalibration lives.

**Method.** Coverage was computed exactly, not read off a plot: for each model, `iter_days()`
streamed `predictions/<variant>/own/` day by day (`true_stec`, `stec_pred`,
`pred_total_unc`, `pred_epistemic_unc`, `pred_aleatoric_unc`), accumulating
`count(|residual| ≤ k·σ)` for k = 1, 2, 3 across all 10,000,000 rows — the same exact-sum approach
`station_independence`/`madrigal_reference_offset` use for the full store. The result for the
paper model reproduces the values on its (pre-existing, unchanged) `sigma_coverage_comparison.png`
to within plot rounding (60.87%/88.85%/96.96% here vs 60.9%/88.8%/97.0% on the figure), which
cross-checks the method against the original PNG-reading approach.

| | nominal | `BayesianResNetSTEC` (paper) | `ResNet_BNN_NLL`, matched init | `ResNet_BNN_NLL`, no init fix (superseded) |
|---|---|---|---|---|
| **Total**, 1σ / 2σ / 3σ | 68.3 / 95.5 / 99.7 | 60.9 / 88.8 / 97.0 | 90.3 / 99.3 / 99.9 | 94.1 / 99.6 / 99.9 |
| **Epistemic**, 1σ / 2σ / 3σ | 68.3 / 95.5 / 99.7 | 9.4 / 18.7 / 27.5 | **66.8 / 90.2 / 97.3** | 81.3 / 96.7 / 99.3 |
| **Aleatoric**, 1σ / 2σ / 3σ | 68.3 / 95.5 / 99.7 | 60.4 / 88.5 / 96.8 | 86.1 / 98.6 / 99.8 | not broken out in the superseded analysis |

**The initialisation fix specifically repaired epistemic calibration, and moved the problem into
the aleatoric head.** In the superseded run, epistemic coverage was wildly over-confident-wide
(81.3% at 1σ against a 68.3% nominal, +13.0 pt), consistent with the missing initialisation
injecting excess parameter-sampling variance that training never fully corrected. With matched
initialisation, epistemic coverage is now **close to nominal and on the same side as the paper
model** — 66.8%/90.2%/97.3% against nominal 68.3%/95.5%/99.7%, i.e. *slightly under-covering*
rather than badly over-covering. The remaining total over-coverage (90.3% at 1σ) is now
concentrated in the **aleatoric** term (86.1%, +17.8 pt over nominal), not the epistemic one. This
is a materially different picture from what the superseded run's coverage numbers suggested: the
fully Bayesian backbone's epistemic uncertainty is not inherently miscalibrated — the first run's
apparent epistemic miscalibration was largely an artifact of the missing initialisation, and fixing
it left a smaller, differently-located calibration problem in the data-noise (aleatoric) output
instead. This is worth flagging precisely because it changes where a future fix should look: not
at the Bayesian residual blocks' contribution to predictive variance, but at the `GaussianNLLLoss`
variance head's calibration under the fully Bayesian architecture.

The paper model's own miscalibration is unchanged and in the opposite direction — it mildly
**under**-covers at every level (60.9%/88.8%/97.0%, gap shrinking from −7.4 to −2.7 points as σ
grows) — this was never affected by the fully-Bayesian retrain and needed no recomputation.

---

## 7. Superseded predecessor: the confounded run

The first R1.2 comparison (`logs/r22_eval.log`, 2026-08-23 13:25–14:36) evaluated a
`ResNet_BNN_NLL` checkpoint that lacked the output-layer initialisation `BayesianResNetSTEC` has
always had. Every number from that run therefore measured **the architecture plus that missing
initialisation**, not the architecture alone — the pervasive −1.93 TECU bias it reported is the
fingerprint of an uncorrected output offset. The two refutations in §4 and §5 above were first
established against that run and have now been reconfirmed, with updated numbers, against the
corrected one; nothing in this document depends on the confounded run's specific figures any
longer. Its full numbers are retained in the evidence tables above (§2, §6) as the third column,
for the record and because they are the direct demonstration of how large an effect a missing
initialisation can look like when mistaken for an architectural result.

The predictions that run produced were also, separately, the cause of a store-partition
contamination — `predictions/pretrained_stec/own` (the paper model's own partition) was briefly
overwritten by these predictions because `inference_testset.py` chose the write partition from
`mode` alone and both models share `mode: pretrain`. That is now fixed (`evaluation.store_variant`
makes architecture part of the partition identity) and repaired (`pretrained_stec/own` rebuilt);
see `STATE.md` for the full account. It does not affect any number in this document — the
confounded run's predictions now live under `predictions/pretrained_stec_resnet_bnn_nll/own`,
correctly separated, and this analysis's §6 uses the corrected run's predictions
(`predictions/pretrained_stec_resnet_bnn_nll/own`, written fresh by
`logs/r22_fully_bayesian_eval.log` at 09:13, after the repair).

**Caveat on precision carried over from the first analysis**: the confounded-run numbers in the
tables above were read by eye off `sigma_coverage_comparison.png`'s bar labels, not recomputed
from a store partition (none existed for that run at the time), so treat their last significant
figure as approximate; the corrected-run and paper-model numbers in this document are exact sums
over all 10,000,000 rows.

---

## 8. What this cannot settle

- **Whether a KL weight tuned to the fully Bayesian parameter count would close the remaining
  accuracy gap.** The magnitude confound is refuted for the loss *value* (§4), but the
  reparameterization-noise effect — every residual-block activation, not just the head, is
  stochastic on every forward pass — is real, structurally present, and still unmeasured.
  Settling this needs an actual sweep over `kl_weight` and/or `prior_sigma` for the fully Bayesian
  variant, which this analysis was not asked to run.
- **Whether the aleatoric-head miscalibration identified in §6 is specific to the fully Bayesian
  architecture or a property of training a much larger stochastic network with this loss.** Only
  one seed (42) was trained for either architecture in either run; there is no replicate to
  separate "fully Bayesian backbones miscalibrate the aleatoric head" from "this particular
  training run did."
- **Per-observation, store-backed stratification (elevation, local time, magnetic latitude, solar
  cycle) for the corrected checkpoint.** The first analysis's §5 (now removed from this document)
  read stratified numbers off `test_metrics/feature_analysis/`, `spatial_analysis/` and
  `temporal_analysis/` plots generated by the confounded run's own evaluation pipeline. The
  corrected evaluation (`logs/r22_fully_bayesian_eval.log`) regenerated those same plots in the
  same experiment directory, overwriting the confounded run's — so the first analysis's specific
  stratified numbers (e.g. "MAE ratio ≈1.8 at low elevation") can no longer be checked against
  their source and are not repeated here, since presenting them as if they still described the
  current checkpoint would be misleading. Redoing that stratification against the corrected plots
  (or, better, against the now-available per-observation store partition, which the confounded run
  did not have) is open work, not done in this pass.
- **Whether either architecture's calibration story would look different on out-of-distribution
  data** (held-out stations, Madrigal). This document only speaks to the in-distribution 10M-row
  test set; `STATE.md`'s "question R2.2 should actually answer" section covers why that is the
  sharper version of the reviewer's question and what probes already exist for it.
