"""The paper's training loss: Gaussian NLL plus an annealed KL penalty.

`BayesianResNetSTEC` has one Bayesian layer (the output head), so training it needs the
usual evidence lower bound: a data term (Gaussian NLL on the predicted mean/variance) and a
complexity term (KL divergence between the layer's learned weight posterior and its prior).
Ported from `TrainManager.train_epoch` in the old `src/training/train_manager.py` (around
line 109), where the combination is::

    loss = nll_loss + current_kl_weight * kld_loss

**The KL weight is not constant.** It is annealed linearly from 0 to 0.1 over the first 5
epochs (`TrainingUtils.get_current_kl_weight` in the old `src/training/training_utils.py`,
around line 45), and that anneal does not appear anywhere in the paper's hyperparameter
table. Skipping it is not a cosmetic difference: the KL term is a normalising-flow-style
complexity penalty over the *whole* weight posterior, and applying it at full strength from
epoch 0 fights the data term before the network has any useful data-driven signal to
regularise, which is a well-documented BNN failure mode (Blundell et al. 2015 anneal it for
the same reason). Warming it up lets the mean/variance head fit first and only then pulls
the posterior toward the prior.

`KLWarmupSchedule` is the schedule; `AnnealedGaussianNLLWithKL` is the composed loss that
uses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torchbnn as bnn
from torch import nn


@dataclass(frozen=True)
class KLWarmupSchedule:
    """Linear KL weight anneal: `start_weight` at epoch 0 up to `end_weight` at
    `warmup_epochs`, held flat at `end_weight` after that.

    `enabled=False` returns a constant `end_weight` at every epoch, which is what the old
    code did when `kl_annealing.enabled` was unset - it fell back to
    `config["training"]["loss_weight"]` unconditionally. Here that fallback constant *is*
    `end_weight`, since every config that has shipped so far keeps the two in sync (the
    example config's comment on `end_weight` literally says "should match loss_weight
    above").
    """

    enabled: bool
    start_weight: float
    end_weight: float
    warmup_epochs: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> KLWarmupSchedule:
        """Read `training.kl_annealing` (keys: `enabled`, `start_weight`, `end_weight`,
        `warmup_epochs`). Defaults match the paper's `config_BNN.yaml`."""
        block = config.get("training", {}).get("kl_annealing", {})
        return cls(
            enabled=block.get("enabled", False),
            start_weight=block.get("start_weight", 0.0),
            end_weight=block.get("end_weight", 0.1),
            warmup_epochs=block.get("warmup_epochs", 5),
        )

    def weight(self, epoch: int) -> float:
        """The KL weight to use for `epoch` (0-indexed, matching the training loop)."""
        # `epoch >= warmup_epochs` is checked before dividing, so warmup_epochs == 0 (or a
        # disabled schedule) never risks a ZeroDivisionError - it just always returns
        # end_weight, which is the correct answer for "no warmup" anyway.
        if not self.enabled or epoch >= self.warmup_epochs:
            return self.end_weight
        progress = epoch / self.warmup_epochs
        return self.start_weight + progress * (self.end_weight - self.start_weight)


class AnnealedGaussianNLLWithKL(nn.Module):
    """`GaussianNLLLoss(mean, target, variance) + kl_weight(epoch) * BKLLoss(model)`.

    Mirrors the reduction settings `get_criterion` used for `BKLLoss` in the old
    `src/utils/loss_function.py`: `reduction="mean", last_layer_only=False`. Since
    `BayesianResNetSTEC` has exactly one Bayesian layer, `last_layer_only` does not change
    the result for that architecture, but it is kept explicit rather than left to whatever
    torchbnn's default happens to be.
    """

    def __init__(self, schedule: KLWarmupSchedule) -> None:
        super().__init__()
        self.schedule = schedule
        self.gaussian_nll = nn.GaussianNLLLoss()
        self.kl_divergence = bnn.BKLLoss(reduction="mean", last_layer_only=False)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> AnnealedGaussianNLLWithKL:
        return cls(KLWarmupSchedule.from_config(config))

    def forward(
        self,
        pred_mean: torch.Tensor,
        target: torch.Tensor,
        pred_var: torch.Tensor,
        model: nn.Module,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Returns `(total_loss, components)`. `components` is for logging - the training
        loop reports NLL, KLD and the weight actually used separately, same as before."""
        nll_loss = self.gaussian_nll(pred_mean, target, pred_var)
        kld_loss = self.kl_divergence(model)
        kl_weight = self.schedule.weight(epoch)
        total_loss = nll_loss + kl_weight * kld_loss
        components = {
            "nll": float(nll_loss.detach()),
            "kld": float(kld_loss.detach()),
            "kl_weight": kl_weight,
            "total": float(total_loss.detach()),
        }
        return total_loss, components
