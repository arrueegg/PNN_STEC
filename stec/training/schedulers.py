"""LR scheduler construction, ported from `get_scheduler` in the old
`src/utils/optimizers.py` (lines 43-89) - together with the bug that lives there.

The old function picks the scheduler *type* correctly: it reads
`config["finetune"]["scheduler"]` when fine-tuning and `config["pretrain"]["scheduler"]`
when pretraining. But every branch that then builds the scheduler's *parameters* reads
`config["pretrain"]` regardless of which mode is running - including the `ReduceLROnPlateau`
branch, whose `if config["mode"] == "finetune"` and `else` arms are byte-identical. Three
concrete symptoms of that one root cause:

1. `CosineAnnealingLR` during fine-tuning gets `T_max` and `eta_min` from the *pretrain*
   block's `epochs` and `learning_rate`, not the fine-tune block's. A 5-epoch fine-tune run
   configured for a 5-epoch cosine decay instead decays as if it were the 150-epoch
   pretrain run - the schedule barely moves in 5 epochs, so fine-tuning trains at
   close to its initial, undecayed learning rate throughout.
2. `StepLR` hardcodes `step_size=1000`, so `scheduler_step_size` in either config block
   (150 for pretrain, 10 for fine-tune in `config_BNN.yaml`) is dead configuration - it is
   read from the YAML by nothing. In practice `StepLR` never fires within a run shorter
   than 1000 epochs, i.e. it behaves like `scheduler: none` for every run this repo has.
3. `ReduceLROnPlateau`'s patience/factor/min_lr are pretrain values even when fine-tuning,
   for the reason above (both mode branches read `config["pretrain"]`).

**This is preserved on purpose, not just fixed.** ~3,580 checkpoints in `experiments/`
were trained under the buggy path (see the paper CLAUDE.md's canonical-results table), so a
rebuilt pipeline that silently contains only the corrected scheduler could no longer
reproduce how those checkpoints were actually trained - `get_scheduler` would return a
different scheduler for the exact same config that produced them. `SchedulerCompat` makes
the choice explicit at the call site instead: `LEGACY` reproduces the original function
byte-for-byte in its observable behaviour (same type selection, same wrong parameter
source), `CORRECTED` reads every parameter from the block the running mode actually owns.
Nothing here silently changes what a caller gets by default - `get_scheduler` defaults to
`LEGACY` so existing call sites keep reproducing history unless a caller opts into the fix
for a new training run.
"""

from __future__ import annotations

import enum
from typing import Any

import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class SchedulerCompat(enum.Enum):
    """Which parameter source `get_scheduler` uses."""

    # Byte-for-byte reproduction of `src/utils/optimizers.py::get_scheduler`: scheduler
    # parameters always come from `config["pretrain"]`, and `StepLR` ignores
    # `scheduler_step_size` entirely. Required to reproduce checkpoints trained before the
    # bug was found.
    LEGACY = "legacy"

    # Parameters come from `config[mode]` - the block the running mode actually owns - and
    # `StepLR` honours `scheduler_step_size`. Use this for any run being trained from now on.
    CORRECTED = "corrected"


def get_scheduler(
    config: dict[str, Any],
    optimizer: optim.Optimizer,
    compat: SchedulerCompat = SchedulerCompat.LEGACY,
) -> LRScheduler | None:
    """Build the scheduler `config["mode"]` asks for.

    `compat` controls only where scheduler *parameters* are read from - the scheduler
    *type* always comes from `config[mode]["scheduler"]`, since that half of the original
    function was never wrong.
    """
    mode = config["mode"]
    scheduler_type = config[mode]["scheduler"]

    if scheduler_type == "none" or scheduler_type is None:
        return None

    # The one line that both encodes and controls the defect: legacy always points at
    # `pretrain`, no matter which mode is actually running.
    params = config["pretrain"] if compat is SchedulerCompat.LEGACY else config[mode]

    if scheduler_type == "StepLR":
        step_size = (
            1000 if compat is SchedulerCompat.LEGACY else params["scheduler_step_size"]
        )
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    if scheduler_type == "ExponentialLR":
        # Never read from config in either version - not one of the three verified
        # defects, so left exactly as it was.
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    if scheduler_type == "CosineAnnealingLR":
        t_max = params["epochs"]
        eta_min = params["learning_rate"] * 0.001
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    if scheduler_type == "ReduceLROnPlateau":
        patience = params.get("scheduler_patience", 5)
        factor = params.get("scheduler_gamma", 0.5)
        min_lr = params["learning_rate"] * 0.001
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unknown scheduler {scheduler_type!r}")
