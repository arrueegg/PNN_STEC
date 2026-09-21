"""Pins `fit_with_best_checkpoint`'s contract against the exact `src/training/
base_trainer.py:251-397` semantics it ports: strict "<" improvement, patience counted in
consecutive non-improving epochs, the *best* epoch's weights returned rather than the final
epoch's, and - as a real-data regression - the paper pretrain's own recorded loss trajectory
(patience=20, best=epoch 149 of 150, early stopping never fires).

Every test below drives `fit_with_best_checkpoint` with a monkeypatched `_run_epoch` that
returns a scripted loss sequence instead of doing real forward/backward passes. That
decouples "does the selection algorithm match `src/`" (what this module ports) from "does
gradient descent converge" (what `fit`'s own gate-verified tests already cover) - the same
separation `tests/training/test_fit.py` uses for its KL-schedule test via
`_RecordingSchedule`, rather than relying on real training dynamics to exercise a specific
control-flow path.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import optim

from stec.models.architectures import BayesianResNetSTEC
from stec.training import checkpointing
from stec.training.checkpointing import fit_with_best_checkpoint
from stec.training.loss import AnnealedGaussianNLLWithKL, KLWarmupSchedule

N_IN = 5
HIDDEN_DIM = 8
REPO_ROOT = Path(__file__).resolve().parents[2]


def make_model() -> BayesianResNetSTEC:
    return BayesianResNetSTEC(n_in=N_IN, hidden_dim=HIDDEN_DIM, num_layers=1)


def make_batches(
    n_batches: int = 2, batch_size: int = 4
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    return [
        (
            torch.randn(batch_size, N_IN, generator=generator),
            torch.randn(batch_size, generator=generator) * 5 + 15.0,
        )
        for _ in range(n_batches)
    ]


def make_loss_fn() -> AnnealedGaussianNLLWithKL:
    return AnnealedGaussianNLLWithKL(
        KLWarmupSchedule(
            enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5
        )
    )


def make_optimizer(model: BayesianResNetSTEC) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-3)


def script_losses(monkeypatch, train_losses: list[float], val_losses: list[float]):
    """Replace `_run_epoch` with one that returns `train_losses[epoch]` /
    `val_losses[epoch]` and, on the training call, stamps `epoch` into the model's first
    parameter so a test can later check *which* epoch's weights a snapshot actually holds -
    not just that some snapshot was taken."""

    def fake_run_epoch(model, batches, loss_fn, epoch, optimizer):
        if optimizer is not None:
            first_param = next(iter(model.state_dict().values()))
            first_param.fill_(float(epoch))
            return train_losses[epoch]
        return val_losses[epoch]

    monkeypatch.setattr(checkpointing, "_run_epoch", fake_run_epoch)


def epoch_marker(state_dict: dict[str, torch.Tensor]) -> float:
    return next(iter(state_dict.values())).flatten()[0].item()


# ---------- core selection semantics ----------


def test_improve_then_degrade_returns_the_best_epoch_not_the_last(monkeypatch):
    # val_loss: 5, 4, 3 (best), 4, 5, 6 - improves for 3 epochs then degrades for 3.
    val_losses = [5.0, 4.0, 3.0, 4.0, 5.0, 6.0]
    script_losses(monkeypatch, train_losses=[0.0] * 6, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=6,
        seed=0,
        patience=float("inf"),
    )

    assert result.best_epoch == 3
    assert result.best_val_loss == pytest.approx(3.0)
    # The snapshot must hold epoch 3's weights (marker == 2, 0-indexed), not epoch 6's.
    assert epoch_marker(result.best_state_dict) == pytest.approx(2.0)
    assert epoch_marker(model.state_dict()) == pytest.approx(
        5.0
    )  # model itself moved on
    assert result.stopped_early is False
    assert len(result.history) == 6


def test_monotonically_improving_run_returns_the_final_epoch(monkeypatch):
    val_losses = [5.0, 4.0, 3.0, 2.0, 1.0]
    script_losses(monkeypatch, train_losses=[0.0] * 5, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=5,
        seed=0,
        patience=3,
    )

    assert result.best_epoch == 5
    assert result.best_val_loss == pytest.approx(1.0)
    assert result.stopped_early is False
    assert len(result.history) == 5


def test_patience_fires_at_the_same_epoch_src_would(monkeypatch):
    # best at epoch 2 (val=1.0), then 3 consecutive non-improving epochs with patience=3:
    # epoch 3 -> patience_counter=1, epoch 4 -> 2, epoch 5 -> 3 == patience, stop after 5.
    val_losses = [5.0, 1.0, 2.0, 2.0, 3.0, 0.5, 0.5]
    script_losses(monkeypatch, train_losses=[0.0] * 7, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=7,
        seed=0,
        patience=3,
    )

    assert result.stopped_early is True
    assert len(result.history) == 5  # stopped after epoch 5, epoch 6/7 never ran
    assert result.best_epoch == 2
    assert result.best_val_loss == pytest.approx(1.0)


def test_a_tied_val_loss_does_not_reset_patience_or_replace_the_checkpoint(monkeypatch):
    # epoch 1 sets best=1.0; epoch 2 ties it (not an improvement under strict "<");
    # epoch 3 also ties. patience=2 must fire after the second consecutive tie.
    val_losses = [1.0, 1.0, 1.0, 1.0]
    script_losses(monkeypatch, train_losses=[0.0] * 4, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=4,
        seed=0,
        patience=2,
    )

    assert result.stopped_early is True
    assert result.best_epoch == 1
    assert (
        len(result.history) == 3
    )  # epoch 1 (improve), 2 (tie->1), 3 (tie->2==patience)
    # The snapshot must still be epoch 1's weights (marker == 0), never overwritten by the
    # ties at epoch 2/3.
    assert epoch_marker(result.best_state_dict) == pytest.approx(0.0)


def test_default_patience_is_infinite_and_never_stops_early(monkeypatch):
    val_losses = [1.0, 2.0, 3.0, 4.0, 5.0]  # degrades every epoch after the first
    script_losses(monkeypatch, train_losses=[0.0] * 5, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=5,
        seed=0,
    )

    assert result.stopped_early is False
    assert len(result.history) == 5
    assert result.best_epoch == 1


def test_requires_val_batches():
    model = make_model()
    with pytest.raises(ValueError, match="val_batches"):
        fit_with_best_checkpoint(
            model,
            make_optimizer(model),
            scheduler=None,
            loss_fn=make_loss_fn(),
            train_batches=make_batches(),
            val_batches=[],
            epochs=3,
            seed=0,
        )


# ---------- real-data regression: the paper pretrain's own recorded trajectory ----------


def test_paper_pretrain_loss_history_never_stops_early_and_best_is_epoch_149(
    monkeypatch,
):
    """`docs/`'s own record of the shipped pretrain checkpoint: 150 epochs, patience=20
    (config.yaml's `pretrain.patience`), and `loss_history.csv` shows the best validation
    loss at epoch 149, with early stopping never firing. Feeding that exact recorded
    sequence back through the ported selection algorithm must reproduce both facts - if it
    stopped early, or picked a different epoch, the port would not match what actually
    trained that checkpoint.
    """
    experiment_dir = (
        REPO_ROOT
        / "experiments"
        / (
            "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_"
            "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
        )
    )
    history_path = experiment_dir / "loss_history.csv"
    if not history_path.exists():
        pytest.skip(f"paper pretrain loss_history.csv not present at {history_path}")

    real_history = pd.read_csv(history_path)
    assert len(real_history) == 150
    train_losses = real_history["train_loss"].tolist()
    val_losses = real_history["val_loss"].tolist()
    script_losses(monkeypatch, train_losses=train_losses, val_losses=val_losses)
    model = make_model()

    result = fit_with_best_checkpoint(
        model,
        make_optimizer(model),
        scheduler=None,
        loss_fn=make_loss_fn(),
        train_batches=make_batches(),
        val_batches=make_batches(),
        epochs=150,
        seed=0,
        patience=20,
    )

    assert result.stopped_early is False
    assert len(result.history) == 150
    assert result.best_epoch == 149
    assert result.best_val_loss == pytest.approx(2.0669411663117447)
