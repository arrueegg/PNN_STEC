"""Pins the fit loop's contract: reproducible given a seed, the right shape of loss
history, the KL schedule actually advancing epoch by epoch, and each scheduler type
receiving the argument it needs (`ReduceLROnPlateau` a metric, everything else none).
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest
import torch
from torch import optim

from stec.models.architectures import BayesianResNetSTEC
from stec.training.fit import LOSS_HISTORY_COLUMNS, fit
from stec.training.loss import AnnealedGaussianNLLWithKL, KLWarmupSchedule
from stec.training.schedulers import SchedulerCompat, get_scheduler

N_IN = 5
HIDDEN_DIM = 8


def make_model() -> BayesianResNetSTEC:
    return BayesianResNetSTEC(n_in=N_IN, hidden_dim=HIDDEN_DIM, num_layers=1)


def make_batches(
    n_batches: int = 3, batch_size: int = 16
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(batch_size, N_IN, generator=generator),
            torch.randn(batch_size, generator=generator) * 5 + 15.0,
        )
        for _ in range(n_batches)
    ]


def make_loss_fn(**schedule_overrides) -> AnnealedGaussianNLLWithKL:
    defaults = dict(enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5)
    defaults.update(schedule_overrides)
    return AnnealedGaussianNLLWithKL(KLWarmupSchedule(**defaults))


def make_optimizer(model: BayesianResNetSTEC) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-3)


# ---------- shape of the returned history ----------


def test_history_has_one_row_per_epoch_with_the_expected_columns():
    model = make_model()
    history = fit(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        epochs=4,
        seed=0,
    )
    assert list(history.columns) == list(LOSS_HISTORY_COLUMNS)
    assert list(history["epoch"]) == [1, 2, 3, 4]
    assert len(history) == 4


def test_val_loss_is_recorded_when_val_batches_are_given():
    model = make_model()
    history = fit(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(n_batches=2),
        epochs=2,
        seed=0,
    )
    assert history["val_loss"].notna().all()


def test_val_loss_is_nan_when_no_val_batches_are_given():
    model = make_model()
    history = fit(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        epochs=2,
        seed=0,
    )
    assert history["val_loss"].isna().all()


# ---------- reproducibility ----------


def test_same_seed_gives_identical_histories():
    # Model construction has its own randomness (BayesLinear's init draws from the global
    # generator), which is deliberately outside what `fit`'s seed controls - `fit` seeds
    # only the training loop. So both runs start from the *same* initial weights and only
    # the seed passed to `fit` is allowed to differ.
    initial_state = copy.deepcopy(make_model().state_dict())

    def run(seed: int) -> pd.DataFrame:
        model = make_model()
        model.load_state_dict(initial_state)
        return fit(
            model,
            make_optimizer(model),
            scheduler=None,
            loss_fn=make_loss_fn(),
            train_batches=make_batches(),
            epochs=3,
            seed=seed,
        )

    first = run(seed=42)
    second = run(seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_histories():
    initial_state = copy.deepcopy(make_model().state_dict())

    def run(seed: int) -> pd.DataFrame:
        model = make_model()
        model.load_state_dict(initial_state)
        return fit(
            model,
            make_optimizer(model),
            scheduler=None,
            loss_fn=make_loss_fn(),
            train_batches=make_batches(),
            epochs=3,
            seed=seed,
        )

    first = run(seed=1)
    second = run(seed=2)
    assert not first["train_loss"].equals(second["train_loss"])


# ---------- the KL weight actually advances ----------


class _RecordingSchedule:
    """Wraps a real `KLWarmupSchedule` and records which epoch it was asked to weight,
    so a test can confirm `fit` passes the epoch through rather than a constant."""

    def __init__(self, schedule: KLWarmupSchedule) -> None:
        self._schedule = schedule
        self.queried_epochs: list[int] = []

    def weight(self, epoch: int) -> float:
        self.queried_epochs.append(epoch)
        return self._schedule.weight(epoch)


def test_kl_weight_advances_across_epochs():
    recording_schedule = _RecordingSchedule(
        KLWarmupSchedule(
            enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5
        )
    )
    loss_fn = AnnealedGaussianNLLWithKL(recording_schedule)
    model = make_model()

    fit(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=loss_fn,
        train_batches=make_batches(n_batches=1),
        epochs=6,
        seed=0,
    )

    # One query per epoch per batch; the epoch argument must be 0 at the first epoch and
    # reach the post-warmup value (>= warmup_epochs) by the last.
    assert recording_schedule.queried_epochs[0] == 0
    assert recording_schedule.queried_epochs[-1] == 5
    assert recording_schedule._schedule.weight(
        recording_schedule.queried_epochs[0]
    ) == pytest.approx(0.0)
    assert recording_schedule._schedule.weight(
        recording_schedule.queried_epochs[-1]
    ) == pytest.approx(0.1)


# ---------- scheduler.step() argument ----------


def _finetune_config(scheduler_type: str) -> dict:
    return {
        "mode": "finetune",
        "pretrain": {
            "epochs": 150,
            "learning_rate": 0.005,
            "scheduler": scheduler_type,
            "scheduler_step_size": 150,
        },
        "finetune": {
            "epochs": 5,
            "learning_rate": 0.001,
            "scheduler": scheduler_type,
            "scheduler_step_size": 10,
        },
    }


def test_reduce_on_plateau_receives_the_validation_loss():
    model = make_model()
    optimizer = make_optimizer(model)
    scheduler = get_scheduler(
        _finetune_config("ReduceLROnPlateau"),
        optimizer,
        compat=SchedulerCompat.CORRECTED,
    )
    fit(
        model,
        optimizer,
        scheduler=scheduler,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(n_batches=2),
        epochs=2,
        seed=0,
    )
    # ReduceLROnPlateau.step(metric) records the metric as `best` (or worse); if `fit` had
    # called it with no argument this would have raised a TypeError before getting here.
    assert scheduler.best != float("inf")


def test_reduce_on_plateau_without_val_batches_raises():
    model = make_model()
    optimizer = make_optimizer(model)
    scheduler = get_scheduler(
        _finetune_config("ReduceLROnPlateau"),
        optimizer,
        compat=SchedulerCompat.CORRECTED,
    )
    with pytest.raises(ValueError, match="ReduceLROnPlateau"):
        fit(
            model,
            optimizer,
            scheduler=scheduler,
            loss_fn=make_loss_fn(),
            train_batches=make_batches(),
            epochs=2,
            seed=0,
        )


def test_cosine_scheduler_steps_with_no_argument():
    model = make_model()
    optimizer = make_optimizer(model)
    scheduler = get_scheduler(
        _finetune_config("CosineAnnealingLR"),
        optimizer,
        compat=SchedulerCompat.CORRECTED,
    )
    initial_lr = optimizer.param_groups[0]["lr"]
    fit(
        model,
        optimizer,
        scheduler=scheduler,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        epochs=3,
        seed=0,
    )
    # CosineAnnealingLR moves the LR away from its initial value after a few `step()`
    # calls with no validation loss involved at all - passing it a metric would raise.
    assert optimizer.param_groups[0]["lr"] != initial_lr


# ---------- SchedulerCompat is load-bearing for the loop, not just for construction ----------


def test_scheduler_compat_changes_the_lr_trajectory_on_a_finetune_config():
    def run(compat: SchedulerCompat) -> float:
        model = make_model()
        optimizer = make_optimizer(model)
        scheduler = get_scheduler(
            _finetune_config("CosineAnnealingLR"), optimizer, compat=compat
        )
        fit(
            model,
            optimizer,
            scheduler=scheduler,
            loss_fn=make_loss_fn(),
            train_batches=make_batches(n_batches=1),
            epochs=3,
            seed=0,
        )
        return optimizer.param_groups[0]["lr"]

    legacy_lr = run(SchedulerCompat.LEGACY)
    corrected_lr = run(SchedulerCompat.CORRECTED)
    # LEGACY reads T_max=150 (pretrain epochs) so 3 steps barely move the cosine curve;
    # CORRECTED reads T_max=5 (finetune epochs), where the same 3 steps land much further
    # around the curve. The compat flag must reach the scheduler `fit` actually drives.
    assert legacy_lr != corrected_lr
