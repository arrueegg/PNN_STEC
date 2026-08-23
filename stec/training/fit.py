"""The training loop itself: given a model, an optimiser, a scheduler and batches, run N
epochs and report the per-epoch loss.

Ported from the epoch loop in the old `BaseTrainer.train` (`src/training/base_trainer.py`,
around line 250) plus `TrainManager.train_epoch` / `ValidationManager.validate_epoch`
(`src/training/train_manager.py` line 34, `src/training/validation_manager.py` line 35).
Those three classes also handle ensembles, CRPS loss, checkpointing, W&B logging and early
stopping - none of which the fit loop needs, since `BayesianResNetSTEC` is not an ensemble
and this repo's loss is `AnnealedGaussianNLLWithKL`. What is kept is only what changes the
numbers a checkpoint would produce:

* the KL weight is a function of the epoch, not a constant, so the epoch has to reach the
  loss (`AnnealedGaussianNLLWithKL.forward`'s `epoch` argument);
* `ReduceLROnPlateau.step` takes the validation loss; every other scheduler's `step` takes
  no argument (`base_trainer.py` lines 371-375);
* validation runs the same forward pass and the same loss, under `torch.no_grad()`, with no
  optimiser step (`validation_manager.py` lines 35-92).

`batches` are handed in already batched and already on the model's device - this loop does
no data loading or device placement, so it can be pointed at a handful of fixed batches for
an equivalence check as easily as at a real `DataLoader`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .loss import AnnealedGaussianNLLWithKL

# One batch: model input, target STEC (or training-space target), both already the right
# shape and device for `model`.
Batch = tuple[torch.Tensor, torch.Tensor]

# Columns of the returned history, matching every stored experiment's `loss_history.csv`
# (`TrainingUtils.save_final_losses`, `src/training/training_utils.py` line 152).
LOSS_HISTORY_COLUMNS = ("epoch", "train_loss", "val_loss")


def _run_epoch(
    model: nn.Module,
    batches: Sequence[Batch],
    loss_fn: AnnealedGaussianNLLWithKL,
    epoch: int,
    optimizer: optim.Optimizer | None,
) -> float:
    """One pass over `batches`. Training when `optimizer` is given, otherwise a `no_grad`
    validation pass - `model.train()`/`model.eval()` and the optimiser step are the only
    difference between the two, so one function covers both instead of duplicating the loop
    the way the old `TrainManager`/`ValidationManager` split did."""
    training = optimizer is not None
    model.train(training)

    running_loss = 0.0
    with torch.enable_grad() if training else torch.no_grad():
        for inputs, targets in batches:
            pred_mean, pred_var = model(inputs)
            loss, _ = loss_fn(
                pred_mean.flatten(), targets, pred_var.flatten(), model, epoch
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss += float(loss.detach())
    return running_loss / len(batches)


def fit(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: LRScheduler | None,
    loss_fn: AnnealedGaussianNLLWithKL,
    train_batches: Sequence[Batch],
    epochs: int,
    seed: int,
    val_batches: Sequence[Batch] | None = None,
) -> pd.DataFrame:
    """Train for `epochs` epochs over `train_batches`, seeding once before the loop so a
    run is reproducible given the same model, batches and seed - the model's Bayesian
    output layer samples fresh weights on every forward call
    (`stec/models/determinism.py`), so without a fixed seed even the first epoch's loss
    would not be reproducible.

    Returns a `(epoch, train_loss, val_loss)` frame, one row per epoch, `epoch` 1-indexed
    to match `loss_history.csv`. `val_loss` is `NaN` when `val_batches` is not given.

    `ReduceLROnPlateau` needs a validation loss to step on, so it is a `ValueError` to pair
    it with no validation batches rather than silently stepping on the training loss, which
    the old code never did.
    """
    if isinstance(scheduler, ReduceLROnPlateau) and val_batches is None:
        raise ValueError("ReduceLROnPlateau requires val_batches to step on")

    torch.manual_seed(seed)

    rows: list[dict[str, float]] = []
    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_batches, loss_fn, epoch, optimizer)
        val_loss = (
            _run_epoch(model, val_batches, loss_fn, epoch, optimizer=None)
            if val_batches is not None
            else float("nan")
        )

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        rows.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )

    return pd.DataFrame(rows, columns=LOSS_HISTORY_COLUMNS)
