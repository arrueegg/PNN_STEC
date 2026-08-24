"""Best-checkpoint selection and early stopping around `fit`'s epoch loop.

`fit` (`stec/training/fit.py`) runs every requested epoch and returns the final weights -
gate-verified against the legacy `TrainManager`/`ValidationManager` loss trajectory (Gate C),
but deliberately missing the checkpoint bookkeeping that loop never needed to reproduce.
Every one of the 3,583 shipped checkpoints was instead selected by
`BaseTrainer.run_training`'s best-val-loss tracking with early stopping
(`src/training/base_trainer.py:251-397`), so this module ports exactly that algorithm:

    best_val_loss = inf
    for epoch in range(epochs):
        train, validate
        step the scheduler
        if val_loss < best_val_loss:            # strict - a tie is not an improvement
            best_val_loss = val_loss
            save the checkpoint
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:    # patience defaults to inf: no early stop
                break

(`config[mode]["early_stopping"]` is never actually read by that code - only `patience` is,
defaulting to `inf` when absent - so this module reads `patience` alone too, the same way
`fit_with_best_checkpoint`'s caller should: the config key's name is misleading, but its
runtime effect is what the shipped checkpoints were trained under.)

**Why this calls `fit`'s private `_run_epoch` instead of calling the public `fit()` once per
epoch.** `fit()` seeds the RNG exactly once, at the top of the call, then runs every epoch
back to back - `BayesianResNetSTEC`'s output layer samples fresh weights from the global RNG
on every forward pass, so the *sequence* of per-epoch draws depends on the RNG state being
carried over from one epoch to the next, not reset between them
(`stec/models/determinism.py`, and the CLAUDE.md gotcha on seeding Bayesian A/B tests makes
the same point about unseeded forward passes generally). Calling `fit(epochs=1, seed=seed)`
once per real epoch to get per-epoch control would re-seed to the *same* state at the start
of every call, so epoch 2 would sample the same weights epoch 1 did instead of continuing
from where epoch 1 left the RNG - a materially different training trajectory, not a faithful
epoch-by-epoch replay of what one `fit(epochs=N)` call produces. Reusing `_run_epoch` directly
and seeding once here, exactly as `fit()` does internally, avoids that: this module's loop is
numerically identical to `fit()`'s own, epoch for epoch, for as many epochs as it runs before
an early stop - it only adds bookkeeping around calls to the same verified unit, rather than
reimplementing the epoch loop or the loss/optimizer/scheduler logic inside it.

The `if scheduler: ... .step(...)` branch is the one exception: it is a few lines inline in
`fit()`'s loop, not factored into `_run_epoch`, so it is duplicated here rather than pulled
out into a shared helper - doing that would mean editing `fit.py`, which is exactly what this
module exists to avoid needing (Gate C stays closed).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .fit import LOSS_HISTORY_COLUMNS, Batch, _run_epoch
from .loss import AnnealedGaussianNLLWithKL


@dataclass(frozen=True)
class BestCheckpointResult:
    """What a run produced: the full per-epoch history (through the epoch that triggered
    early stopping, if any - matching `base_trainer.py`, which records that epoch's losses
    before breaking), and the best epoch's own weights, separately from whatever the model
    holds after the loop ends."""

    history: pd.DataFrame
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_val_loss: float
    stopped_early: bool


def fit_with_best_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None,
    loss_fn: AnnealedGaussianNLLWithKL,
    train_batches: Sequence[Batch],
    val_batches: Sequence[Batch],
    epochs: int,
    seed: int,
    patience: float = float("inf"),
) -> BestCheckpointResult:
    """Run up to `epochs` epochs, returning the best-validation-loss checkpoint rather than
    the final one. `patience` is the number of consecutive non-improving epochs allowed
    before stopping early - `float("inf")` (the default, matching `base_trainer.py`'s own
    `.get("patience", float("inf"))`) never stops early.

    Unlike `fit`, `val_batches` is required: there is no validation loss to select a "best"
    epoch by without it.
    """
    if not val_batches:
        raise ValueError(
            "fit_with_best_checkpoint requires val_batches: best-checkpoint selection has "
            "no validation loss to compare epochs against otherwise."
        )
    if isinstance(scheduler, ReduceLROnPlateau) and val_batches is None:
        raise ValueError("ReduceLROnPlateau requires val_batches to step on")

    torch.manual_seed(seed)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    stopped_early = False
    rows: list[dict[str, float]] = []

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_batches, loss_fn, epoch, optimizer)
        val_loss = _run_epoch(model, val_batches, loss_fn, epoch, optimizer=None)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        rows.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )

        # Strict "<", matching src/training/base_trainer.py:381-390 exactly: a tied
        # val_loss does not replace the saved checkpoint and does not reset patience.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_early = True
                break

    # The first epoch always improves on `best_val_loss = inf`, so this can only be None
    # if the loop above never ran - i.e. epochs <= 0, which is a caller error, not a
    # reachable state for any real training config.
    if best_state_dict is None:
        raise ValueError(f"epochs must be positive, got {epochs}")

    return BestCheckpointResult(
        history=pd.DataFrame(rows, columns=LOSS_HISTORY_COLUMNS),
        best_state_dict=best_state_dict,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        stopped_early=stopped_early,
    )
