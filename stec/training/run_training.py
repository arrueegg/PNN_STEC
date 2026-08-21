"""Training entry point: the driver `stec/training/{fit,loss,schedulers}.py` never had.

Those three modules are gate-verified (Gate C: bit-exact loss trajectory against the legacy
`TrainManager`/`ValidationManager`, fixed batches, 3 and 6 epochs) but nothing in `stec/`
called them - every real training run went through pre-rebuild `src/main.py` +
`src/pretrain.py` / `src/finetune.py`. This module is the missing wiring: resolved config in,
checkpoint and loss-history CSV out, reusing `fit`/`loss`/`schedulers` exactly as verified
rather than reimplementing any part of the epoch loop.

Data comes from `stec.data.day_reader.read_day`, the only ported read path - one day's
`train`/`val` split at a time, assembled into the model's input tensor via
`stec.data.transforms.FeatureAssembler`. That is deliberately narrower than the pre-rebuild
`data_loader` package: this driver reproduces the 258 daily fine-tunes exactly (each one
trains and validates on a single day's own `train_idx`/`val_idx`), because `--train-days`/
`--val-days` accepts any list of days and reads each one the same way. What it does *not*
reproduce is the 150-epoch pretrain's 500,000-observation subsample drawn across 15 years of
`train_dates.list` - that sampling lives in the not-yet-ported `src/data_processing/`
aggregation step (`docs/revision/task_board.md` S2), and multi-day training here is an
honest generalisation of the per-day reader, not a port of that subsampling.

Two things this driver deliberately does not do, both because the module it reuses does not
do them either - `fit.py`'s own docstring lists what it omits, and reimplementing either one
here would mean no longer reusing the gate-verified loop:

* **No best-checkpoint selection or early stopping.** `fit` runs every requested epoch and
  returns the final weights, not the epoch with the lowest validation loss. Every one of the
  ~3,580 shipped checkpoints was instead selected by `BaseTrainer.run_training`'s
  best-val-loss tracking with early stopping, so a checkpoint this driver produces is not a
  drop-in replacement for one of those - only `fit`'s own numbers (the loss trajectory) are
  gate-verified equivalent.
* **`training.log_target` and `<mode>.freeze_body` are refused, not silently ignored.** The
  log-normal target transform and body-freezing helper are pre-rebuild code
  (`training.data_transforms`, `utils.model_utils.freeze_model_body`) that nothing under
  `stec/training/` ports. A config asking for either would train a genuinely different model
  than the one it describes if the request were dropped quietly, so this driver raises
  instead of guessing - the same reasoning `KLWarmupSchedule.from_config` already applies to
  a mismatched `loss_weight`/`end_weight`.

Usage::

    python -m stec.training.run_training --config path/to/config.yaml
    python -m stec.training.run_training --config path/to/config.yaml \\
        --train-days 2024:132 2024:133 --val-days 2024:134 --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from ..data.day_reader import read_day
from ..data.feature_layout import FeatureLayout, layout_from_feature_control
from ..data.spherical_harmonics import SphericalHarmonics
from ..data.transforms import FeatureAssembler
from ..models.architectures import BayesianResNetSTEC, load_checkpoint
from .fit import fit
from .loss import AnnealedGaussianNLLWithKL
from .schedulers import SchedulerCompat, get_scheduler

logger = logging.getLogger(__name__)

Day = tuple[int, int]
Batch = tuple[torch.Tensor, torch.Tensor]


def _distribution_of(config: dict) -> str:
    """Same rule `stec.analysis.paper_tables` uses to pick the SH convention."""
    loss_function = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplac" in loss_function else "gaussian"


def build_layout_and_assembler(config: dict) -> tuple[FeatureLayout, FeatureAssembler]:
    """The input layout a config describes, and the assembler that fills it.

    Mirrors `stec.analysis.paper_tables.feature_table`'s construction exactly, because a
    checkpoint's input width is defined by this same call - a training run and the table
    describing it must agree by construction, not by convention.
    """
    layout = layout_from_feature_control(
        config.get("feature_control", {}),
        sh_degree=int(config.get("data", {}).get("SH_degree", 0)),
        target=str(config.get("target", "stec")),
        distribution=_distribution_of(config),
    )
    sh_encoder = None
    if layout.sh_width:
        legendre_polys = layout.sh_convention.legendre_polys(layout.sh_degree)
        sh_encoder = SphericalHarmonics(legendre_polys)
    return layout, FeatureAssembler(layout, sh_encoder=sh_encoder)


def _numeric_tensors(raw: dict) -> dict[str, torch.Tensor]:
    """Every raw column `read_day` returns that is actually a number - station/sat names
    are read as fixed-width byte strings and are never model inputs, so `FeatureAssembler`
    should never be handed them."""
    return {
        name: torch.from_numpy(values).float()
        for name, values in raw.items()
        if values.dtype.kind in "fiu"
    }


def read_and_assemble(
    days: list[Day],
    split: str,
    assembler: FeatureAssembler,
    database_root: Path | None,
    space_weather: Path | None,
) -> Batch:
    """Every `split` row from `days`, concatenated into one (inputs, targets) pair.

    `targets` is the raw `stec` column, untransformed - the only target space this driver
    supports (see the module docstring on `log_target`).
    """
    all_inputs, all_targets = [], []
    for year, doy in days:
        raw = read_day(
            year,
            doy,
            split=split,
            database_root=database_root,
            space_weather=space_weather,
        )
        raw_tensors = _numeric_tensors(raw)
        all_inputs.append(assembler.assemble(raw_tensors))
        all_targets.append(raw_tensors["stec"])
    return torch.cat(all_inputs), torch.cat(all_targets)


def materialize_batches(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> list[Batch]:
    """`fit`'s expected `Sequence[Batch]`: already batched, already on the model's device.

    `fit` re-iterates the same sequence every epoch - it has no per-epoch reshuffle hook, by
    design (`fit.py`'s docstring: it keeps only what changes the numbers a checkpoint would
    produce). A live `DataLoader` handed to it would therefore serve the *same* shuffled
    order on every epoch anyway, so batching once into a plain list here costs nothing
    against that and avoids re-shuffling machinery `fit` would never exercise.
    """
    dataset = TensorDataset(inputs, targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
    return [(x.to(device), y.to(device)) for x, y in loader]


def _resolve_pretrain_checkpoint(
    config: dict, explicit_path: Path | None
) -> Path | None:
    """The checkpoint a finetune run should load, or None to train from scratch.

    Mirrors `Finetuner.initialize_model`'s filename convention
    (`src/finetune.py`): `pretrain_<model_type>_seed<seed>.pth` under
    `<pretrain_folder>/model/`.
    """
    if explicit_path is not None:
        return explicit_path
    if config["mode"] != "finetune" or config.get("finetune_from_scratch", False):
        return None
    pretrain_folder = config.get("pretrain_folder")
    if not pretrain_folder:
        return None
    seed = config["random_seed"]
    filename = f"pretrain_{config['model']['model_type']}_seed{seed:02}.pth"
    candidate = Path(pretrain_folder) / "model" / filename
    return candidate if candidate.exists() else None


def build_model(
    config: dict,
    layout: FeatureLayout,
    device: torch.device,
    pretrain_checkpoint: Path | None,
) -> torch.nn.Module:
    """A fresh model sized from `layout`, or a pretrained one loaded for fine-tuning.

    Either way the result must have exactly `layout.total_dim` input columns - a checkpoint
    trained under a different feature_control would silently misalign every feature after
    the point the two layouts disagree, which is the exact failure `feature_layout.py`'s
    docstring warns about, so this is checked rather than assumed.
    """
    if pretrain_checkpoint is not None:
        model, shape = load_checkpoint(pretrain_checkpoint, map_location=device)
        if shape["n_in"] != layout.total_dim:
            raise ValueError(
                f"pretrained checkpoint {pretrain_checkpoint} expects {shape['n_in']} "
                f"input columns, but this run's feature_control assembles "
                f"{layout.total_dim} - refusing rather than silently training on "
                "misaligned features."
            )
        logger.info(f"Loaded pretrained weights from {pretrain_checkpoint}")
        return model.to(device)

    model_cfg = config["model"]
    model = BayesianResNetSTEC(
        n_in=layout.total_dim,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        dropout_rate=model_cfg.get("dropout_rate", 0.0),
        prior_sigma=model_cfg.get("prior_sigma", 0.1),
    )
    return model.to(device)


def build_optimizer(config: dict, mode: str, parameters) -> optim.Optimizer:
    training_cfg = config["training"]
    optimizer_type = training_cfg.get("optimizer", "Adam")
    optimizer_cls = getattr(optim, optimizer_type, None)
    if optimizer_cls is None:
        raise ValueError(f"unknown optimizer {optimizer_type!r}")
    return optimizer_cls(
        parameters,
        lr=config[mode]["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )


def train(
    config: dict,
    *,
    output_dir: Path,
    train_days: list[Day],
    val_days: list[Day],
    database_root: Path | None = None,
    space_weather: Path | None = None,
    device: torch.device = torch.device("cpu"),
    pretrain_checkpoint: Path | None = None,
    scheduler_compat: SchedulerCompat = SchedulerCompat.LEGACY,
) -> Path:
    """Train `config['mode']`'s model over `train_days`/`val_days`, returning the
    checkpoint path. Writes the checkpoint and `loss_history.csv` under `output_dir`."""
    mode = config["mode"]
    if mode not in ("pretrain", "finetune"):
        raise ValueError(
            f"config['mode'] must be 'pretrain' or 'finetune', got {mode!r}"
        )
    if config.get("training", {}).get("log_target", False):
        raise NotImplementedError(
            "training.log_target is not ported: the log-normal target transform that maps "
            "a raw STEC value into the space fit()'s loss scores it in "
            "(src/training/data_transforms.py) has no stec/ equivalent, so a target this "
            "driver could hand to AnnealedGaussianNLLWithKL would not be the one the "
            "config asked for. Set log_target: false, which every shipped STEC checkpoint "
            "already uses."
        )
    if config.get(mode, {}).get("freeze_body", False):
        raise NotImplementedError(
            f"{mode}.freeze_body is not ported: utils.model_utils.freeze_model_body has "
            "no stec/ equivalent, so silently ignoring this flag would train the whole "
            "network when the config asked for only the output head."
        )

    layout, assembler = build_layout_and_assembler(config)
    seed = config["random_seed"]

    train_inputs, train_targets = read_and_assemble(
        train_days, "train", assembler, database_root, space_weather
    )
    val_inputs, val_targets = read_and_assemble(
        val_days, "val", assembler, database_root, space_weather
    )
    logger.info(
        f"{mode}: {len(train_inputs):,} train / {len(val_inputs):,} val rows over "
        f"{len(train_days)} train day(s), {len(val_days)} val day(s)"
    )

    # A fresh model's BayesLinear head draws its initial weights from the global RNG at
    # construction time - before fit() gets to seed anything - so a from-scratch run is
    # only reproducible if that draw is seeded too. Mirrors src/main.py's setup_seed(),
    # called before the trainer is constructed, for exactly this reason. A loaded
    # checkpoint is unaffected (load_state_dict overwrites whatever this drew).
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    resolved_checkpoint = _resolve_pretrain_checkpoint(config, pretrain_checkpoint)
    model = build_model(config, layout, device, resolved_checkpoint)

    batch_size = config[mode]["batchsize"]
    train_batches = materialize_batches(
        train_inputs, train_targets, batch_size, shuffle=True, seed=seed, device=device
    )
    val_batches = materialize_batches(
        val_inputs, val_targets, batch_size, shuffle=False, seed=seed, device=device
    )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(config, mode, trainable_params)
    scheduler = get_scheduler(config, optimizer, compat=scheduler_compat)
    loss_fn = AnnealedGaussianNLLWithKL.from_config(config)

    history = fit(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_batches,
        epochs=config[mode]["epochs"],
        seed=seed,
        val_batches=val_batches,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    checkpoint_path = (
        model_dir / f"{mode}_{config['model']['model_type']}_seed{seed:02}.pth"
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(history["epoch"].iloc[-1]),
        },
        checkpoint_path,
    )

    history_path = output_dir / "loss_history.csv"
    history.to_csv(history_path, index=False)

    logger.info(
        f"{mode} finished: {len(history)} epoch(s), "
        f"final train_loss={history['train_loss'].iloc[-1]:.4f}, "
        f"val_loss={history['val_loss'].iloc[-1]:.4f}"
    )
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Loss history: {history_path}")
    return checkpoint_path


def _parse_day(token: str) -> Day:
    year, doy = token.split(":")
    return int(year), int(doy)


def _default_days(config: dict) -> list[Day]:
    return [(int(config["year"]), int(config["doy"]))]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="resolved run config.yaml - a stored experiment config, not a template",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="defaults to config['output_dir']"
    )
    parser.add_argument(
        "--train-days",
        nargs="*",
        type=_parse_day,
        default=None,
        metavar="YYYY:DDD",
        help="defaults to the single day config['year']/config['doy']",
    )
    parser.add_argument(
        "--val-days", nargs="*", type=_parse_day, default=None, metavar="YYYY:DDD"
    )
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--space-weather", type=Path, default=None)
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="override the auto-discovered pretrain checkpoint used for fine-tuning",
    )
    parser.add_argument(
        "--scheduler-compat",
        choices=[compat.value for compat in SchedulerCompat],
        default=SchedulerCompat.LEGACY.value,
        help="LEGACY reproduces the scheduler-parameter bug every shipped checkpoint was "
        "trained under; CORRECTED reads parameters from the running mode's own config "
        "block. See stec.training.schedulers.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = yaml.safe_load(args.config.read_text())
    output_dir = args.output_dir or Path(config["output_dir"])
    train_days = args.train_days or _default_days(config)
    val_days = args.val_days or _default_days(config)

    train(
        config,
        output_dir=output_dir,
        train_days=train_days,
        val_days=val_days,
        database_root=args.database_root,
        space_weather=args.space_weather,
        device=torch.device(args.device),
        pretrain_checkpoint=args.pretrain_checkpoint,
        scheduler_compat=SchedulerCompat(args.scheduler_compat),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
