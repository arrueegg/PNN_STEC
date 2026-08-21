"""Pins the KL anneal that the paper's hyperparameter table omits: linear from
`start_weight` to `end_weight` over `warmup_epochs`, flat afterwards. Ported from
`TrainingUtils.get_current_kl_weight` (`src/training/training_utils.py`, around line 45) and
the loss combination in `TrainManager.train_epoch` (`src/training/train_manager.py`, around
line 109).
"""

from __future__ import annotations

import pytest
import torch

from stec.models.architectures import BayesianResNetSTEC
from stec.training.loss import AnnealedGaussianNLLWithKL, KLWarmupSchedule


def _schedule(**overrides) -> KLWarmupSchedule:
    defaults = dict(enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5)
    defaults.update(overrides)
    return KLWarmupSchedule(**defaults)


# ---------- KLWarmupSchedule.weight ----------


def test_weight_is_start_weight_at_epoch_zero():
    schedule = _schedule()
    assert schedule.weight(0) == pytest.approx(0.0)


def test_weight_reaches_end_weight_exactly_at_the_warmup_epoch():
    schedule = _schedule()
    assert schedule.weight(5) == pytest.approx(0.1)


def test_weight_is_linear_between_start_and_warmup():
    schedule = _schedule()
    # Halfway through a 5-epoch warmup from 0.0 to 0.1.
    assert schedule.weight(2) == pytest.approx(0.04)
    assert schedule.weight(3) == pytest.approx(0.06)


def test_weight_stays_flat_at_end_weight_after_warmup():
    schedule = _schedule()
    assert schedule.weight(6) == pytest.approx(0.1)
    assert schedule.weight(1000) == pytest.approx(0.1)


def test_weight_respects_a_nonzero_start_weight():
    schedule = _schedule(start_weight=0.02, end_weight=0.1, warmup_epochs=4)
    assert schedule.weight(0) == pytest.approx(0.02)
    assert schedule.weight(2) == pytest.approx(0.06)
    assert schedule.weight(4) == pytest.approx(0.1)


# ---------- config-driven construction ----------


def test_from_config_reads_the_kl_annealing_block():
    config = {
        "training": {
            "kl_annealing": {
                "enabled": True,
                "start_weight": 0.0,
                "end_weight": 0.2,
                "warmup_epochs": 10,
            }
        }
    }
    schedule = KLWarmupSchedule.from_config(config)
    assert schedule.enabled is True
    assert schedule.start_weight == 0.0
    assert schedule.end_weight == 0.2
    assert schedule.warmup_epochs == 10
    assert schedule.weight(5) == pytest.approx(0.1)  # halfway through a 10-epoch warmup


def test_enabled_false_gives_a_constant_weight_at_every_epoch():
    config = {
        "training": {
            "kl_annealing": {
                "enabled": False,
                "start_weight": 0.0,
                "end_weight": 0.1,
                "warmup_epochs": 5,
            }
        }
    }
    schedule = KLWarmupSchedule.from_config(config)
    assert schedule.weight(0) == pytest.approx(0.1)
    assert schedule.weight(2) == pytest.approx(0.1)
    assert schedule.weight(500) == pytest.approx(0.1)


def test_from_config_defaults_match_config_bnn_yaml():
    """`config/config_BNN.yaml` is the paper's config: enabled, 0.0 -> 0.1 over 5 epochs."""
    schedule = KLWarmupSchedule.from_config({"training": {}})
    assert (
        schedule.enabled is False
    )  # opting in is explicit; missing block anneals nothing
    assert schedule.end_weight == pytest.approx(0.1)
    assert schedule.warmup_epochs == 5


# ---------- AnnealedGaussianNLLWithKL: the composed loss ----------


def _model_and_batch(seed: int = 0):
    torch.manual_seed(seed)
    model = BayesianResNetSTEC(n_in=6, hidden_dim=8, num_layers=1)
    inputs = torch.randn(16, 6)
    targets = torch.randn(16, 1) * 5 + 15.0
    mean, variance = model(inputs)
    return model, mean, variance, targets


def test_total_loss_equals_nll_plus_weighted_kld():
    schedule = _schedule(
        enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5
    )
    loss_fn = AnnealedGaussianNLLWithKL(schedule)
    model, mean, variance, targets = _model_and_batch()

    total, components = loss_fn(mean, targets, variance, model, epoch=2)

    expected_weight = schedule.weight(2)
    expected_total = components["nll"] + expected_weight * components["kld"]
    assert components["kl_weight"] == pytest.approx(expected_weight)
    assert float(total) == pytest.approx(expected_total, rel=1e-5)
    assert components["total"] == pytest.approx(expected_total, rel=1e-5)


def test_kl_term_is_absent_at_epoch_zero_when_start_weight_is_zero():
    """At epoch 0 the anneal contributes nothing, so total loss must equal the NLL alone."""
    schedule = _schedule(
        enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5
    )
    loss_fn = AnnealedGaussianNLLWithKL(schedule)
    model, mean, variance, targets = _model_and_batch()

    total, components = loss_fn(mean, targets, variance, model, epoch=0)

    assert components["kl_weight"] == pytest.approx(0.0)
    assert float(total) == pytest.approx(components["nll"], rel=1e-5)


def test_kl_term_is_fully_weighted_after_warmup():
    schedule = _schedule(
        enabled=True, start_weight=0.0, end_weight=0.1, warmup_epochs=5
    )
    loss_fn = AnnealedGaussianNLLWithKL(schedule)
    model, mean, variance, targets = _model_and_batch()

    total, components = loss_fn(mean, targets, variance, model, epoch=100)

    assert components["kl_weight"] == pytest.approx(0.1)
    assert float(total) == pytest.approx(
        components["nll"] + 0.1 * components["kld"], rel=1e-5
    )


def test_from_config_builds_a_working_loss():
    config = {
        "training": {
            "kl_annealing": {
                "enabled": True,
                "start_weight": 0.0,
                "end_weight": 0.1,
                "warmup_epochs": 5,
            }
        }
    }
    loss_fn = AnnealedGaussianNLLWithKL.from_config(config)
    model, mean, variance, targets = _model_and_batch()
    total, _ = loss_fn(mean, targets, variance, model, epoch=1)
    assert torch.isfinite(total)


def test_disagreeing_kl_weights_are_refused():
    """The legacy trainer annealed to loss_weight; this one uses end_weight.

    config/config_BNN.yaml ships with 1.0 against 0.1, so silently preferring either key
    would train a model tenfold different from the one the other implementation gives.
    """
    config = {
        "training": {
            "loss_weight": 1.0,
            "kl_annealing": {"enabled": True, "end_weight": 0.1, "warmup_epochs": 5},
        }
    }
    with pytest.raises(ValueError, match="disagree"):
        KLWarmupSchedule.from_config(config)


def test_agreeing_kl_weights_are_accepted():
    """Every one of the 853 shipped configs sets both to the same value."""
    config = {
        "training": {
            "loss_weight": 0.1,
            "kl_annealing": {"enabled": True, "end_weight": 0.1, "warmup_epochs": 5},
        }
    }
    assert KLWarmupSchedule.from_config(config).end_weight == pytest.approx(0.1)


def test_a_disabled_schedule_does_not_police_the_weights():
    """With annealing off the legacy code returned loss_weight and never annealed, so
    there is no second key to disagree with."""
    config = {
        "training": {
            "loss_weight": 1.0,
            "kl_annealing": {"enabled": False, "end_weight": 0.1},
        }
    }
    assert KLWarmupSchedule.from_config(config).enabled is False
