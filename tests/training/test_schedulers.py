"""Pins both the legacy `get_scheduler` bug (parameters always from `pretrain`, `StepLR`
step size hardcoded) and the corrected behaviour (parameters from the running mode's own
config block), since both are load-bearing: legacy reproduces ~3,580 existing checkpoints,
corrected is what any newly trained run should use.
"""

from __future__ import annotations

import pytest
import torch
from torch import optim

from stec.training.schedulers import SchedulerCompat, get_scheduler

# Mirrors config_BNN.yaml: pretrain and fine-tune deliberately use different epochs,
# learning rates and step sizes, so a test that reads the wrong block is caught.
PRETRAIN_BLOCK = {
    "epochs": 150,
    "learning_rate": 0.005,
    "scheduler": "CosineAnnealingLR",
    "scheduler_step_size": 150,
}

FINETUNE_BLOCK = {
    "epochs": 5,
    "learning_rate": 0.001,
    "scheduler": "CosineAnnealingLR",
    "scheduler_step_size": 10,
}


def _config(mode: str, scheduler: str) -> dict:
    return {
        "mode": mode,
        "pretrain": {**PRETRAIN_BLOCK, "scheduler": scheduler},
        "finetune": {**FINETUNE_BLOCK, "scheduler": scheduler},
    }


def _optimizer() -> optim.Optimizer:
    model = torch.nn.Linear(2, 2)
    return optim.SGD(model.parameters(), lr=0.1)


def test_scheduler_type_none_returns_no_scheduler():
    config = _config("finetune", "none")
    assert get_scheduler(config, _optimizer()) is None


def test_scheduler_type_is_mode_aware_regardless_of_compat():
    """The type-selection half of the old function was never wrong - both compat modes
    must still read the type from the running mode's own block."""
    config = _config("finetune", "StepLR")
    config["pretrain"]["scheduler"] = "CosineAnnealingLR"

    for compat in SchedulerCompat:
        scheduler = get_scheduler(config, _optimizer(), compat=compat)
        assert isinstance(scheduler, optim.lr_scheduler.StepLR)


def test_default_compat_is_legacy():
    config = _config("finetune", "StepLR")
    scheduler = get_scheduler(config, _optimizer())
    assert scheduler.step_size == 1000


# ---------- Legacy: reproduces the original bug ----------


def test_legacy_finetune_steplr_step_size_is_hardcoded_1000():
    config = _config("finetune", "StepLR")
    config["finetune"]["scheduler_step_size"] = 10  # would be 10 if the bug were fixed
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.LEGACY)
    assert scheduler.step_size == 1000


def test_legacy_finetune_cosine_uses_pretrain_epochs_and_eta_min():
    config = _config("finetune", "CosineAnnealingLR")
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.LEGACY)
    assert scheduler.T_max == PRETRAIN_BLOCK["epochs"]
    assert scheduler.eta_min == pytest.approx(PRETRAIN_BLOCK["learning_rate"] * 0.001)


def test_legacy_finetune_reduce_on_plateau_uses_pretrain_params():
    config = _config("finetune", "ReduceLROnPlateau")
    config["pretrain"]["scheduler_patience"] = 7
    config["pretrain"]["scheduler_gamma"] = 0.25
    config["finetune"]["scheduler_patience"] = 2
    config["finetune"]["scheduler_gamma"] = 0.9
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.LEGACY)
    assert scheduler.patience == 7
    assert scheduler.factor == 0.25
    assert scheduler.min_lrs == [pytest.approx(PRETRAIN_BLOCK["learning_rate"] * 0.001)]


# ---------- Corrected: parameters follow the running mode ----------


def test_corrected_finetune_steplr_uses_configured_step_size():
    config = _config("finetune", "StepLR")
    config["finetune"]["scheduler_step_size"] = 10
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.CORRECTED)
    assert scheduler.step_size == 10


def test_corrected_finetune_cosine_uses_finetune_epochs_and_eta_min():
    config = _config("finetune", "CosineAnnealingLR")
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.CORRECTED)
    assert scheduler.T_max == FINETUNE_BLOCK["epochs"]
    assert scheduler.eta_min == pytest.approx(FINETUNE_BLOCK["learning_rate"] * 0.001)


def test_corrected_finetune_reduce_on_plateau_uses_finetune_params():
    config = _config("finetune", "ReduceLROnPlateau")
    config["pretrain"]["scheduler_patience"] = 7
    config["pretrain"]["scheduler_gamma"] = 0.25
    config["finetune"]["scheduler_patience"] = 2
    config["finetune"]["scheduler_gamma"] = 0.9
    scheduler = get_scheduler(config, _optimizer(), compat=SchedulerCompat.CORRECTED)
    assert scheduler.patience == 2
    assert scheduler.factor == 0.9
    assert scheduler.min_lrs == [pytest.approx(FINETUNE_BLOCK["learning_rate"] * 0.001)]


def test_corrected_pretrain_run_matches_legacy_pretrain_run():
    """When the running mode *is* pretrain, both compat modes read the same block, so they
    must agree - the bug only bites fine-tune runs."""
    config = _config("pretrain", "CosineAnnealingLR")
    legacy = get_scheduler(config, _optimizer(), compat=SchedulerCompat.LEGACY)
    corrected = get_scheduler(config, _optimizer(), compat=SchedulerCompat.CORRECTED)
    assert legacy.T_max == corrected.T_max
    assert legacy.eta_min == corrected.eta_min


# ---------- Scheduler types unaffected by the parameter-source bug ----------


def test_exponential_lr_is_identical_under_both_compat_modes():
    config = _config("finetune", "ExponentialLR")
    for compat in SchedulerCompat:
        scheduler = get_scheduler(config, _optimizer(), compat=compat)
        assert scheduler.gamma == 0.95


def test_unknown_scheduler_type_raises():
    config = _config("finetune", "NotARealScheduler")
    with pytest.raises(ValueError, match="NotARealScheduler"):
        get_scheduler(config, _optimizer())
