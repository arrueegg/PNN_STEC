"""Training entry point: the driver `stec/training/{fit,loss,schedulers}.py` never had.

Those three modules are gate-verified (Gate C: bit-exact loss trajectory against the legacy
`TrainManager`/`ValidationManager`, fixed batches, 3 and 6 epochs) but nothing in `stec/`
called them - every real training run went through pre-rebuild `src/main.py` +
`src/pretrain.py` / `src/finetune.py`. This module is the missing wiring: resolved config in,
checkpoint and loss-history CSV out, reusing `fit`/`loss`/`schedulers` exactly as verified
rather than reimplementing any part of the epoch loop.

For `mode: finetune`, data comes from `stec.data.day_reader.read_day` - one day's
`train`/`val` split at a time, assembled into the model's input tensor via
`stec.data.transforms.FeatureAssembler`. This reproduces the 258 daily fine-tunes exactly
(each one trains and validates on a single day's own `train_idx`/`val_idx`), because
`--train-days`/`--val-days` accepts any list of days and reads each one the same way.

For `mode: pretrain` *with* `data.train_subset_size` set in the config, data comes from
`stec.data.aggregated_dataset.AggregatedSplitDataset` instead - a lazy, row-indexed reader
over `data/{train,val}.h5` (`DataPreprocessor.build_split_h5`'s output, ~1.37e9 rows for
train), because the pretrain's 500,000-observation-per-epoch subsample is drawn with
replacement across 15 years of `train_dates.list`, and materialising that span the way
`read_and_assemble`'s `torch.cat` does for a few-day fine-tune would mean holding the whole
aggregate - well over 100 GB once assembled - in memory before a single epoch could sample
from it. `build_pretrain_batches` wires this the same way
`src/data_loader/loaders.py::get_data_loaders` always did:
`stec.data.splits.EpochRandomSampler` draws `data.train_subset_size` rows with replacement
each epoch (reseeded per epoch via `stec.data.splits.ResampledEpochBatches`, since `fit`'s
loop has no per-epoch hook of its own - see that class's docstring for why), and
`stec.data.splits.get_fixed_subset_indices` draws a fixed `data.val_size` validation subset
once, the same set every epoch. `--train-days`/`--val-days` are accepted but unused in this
mode, since the aggregate already spans every split day at once. `data.train_subset_size` is
the gate, not `mode` alone, because it is the same signal the legacy loader branched on
(`elif train_subset and train_subset < len(ds): use EpochRandomSampler`) - every real
pretrain config sets it (`config/config_BNN.yaml`: 500,000), and a bare `mode: pretrain`
with no such key (as every `tests/training/test_run_training.py` fixture that builds a
`pretrain_*` checkpoint to fine-tune from does) keeps using the per-day path unchanged.

**What this closes and what it still leaves open.** The Dataset, sampler wiring and a short
smoke run (a handful of epochs, a few thousand rows, real `data/train.h5`) are verified - see
`tests/data/test_aggregated_dataset.py` and `tests/training/test_run_training_pretrain.py`.
A full 150-epoch pretrain was not run in the session that added this path: the model this
driver would produce from a real pretrain run has not been compared against the shipped
pretrained checkpoint end to end, only its data-loading half.

One thing this driver used to not do, now closed: **best-checkpoint selection and early
stopping.** `fit` itself still runs every requested epoch and returns the final weights, not
the epoch with the lowest validation loss - that has not changed, and `fit.py` was
deliberately left alone (Gate C stays closed). This driver now calls
`stec.training.checkpointing.fit_with_best_checkpoint` instead of `fit` directly: it wraps
the same epoch-level unit `fit` is built from and adds the exact `best_val_loss`/patience
bookkeeping `BaseTrainer.run_training` used to select every one of the 3,583 shipped
checkpoints (`src/training/base_trainer.py:251-397` - see `checkpointing.py`'s own docstring
for the full port rationale, including why it does not just call `fit()` once per epoch).
`patience` is read from `config[mode]["patience"]`, defaulting to `float("inf")` (never stop
early) when absent, matching `base_trainer.py`'s own `.get("patience", float("inf"))` - note
this is *not* gated on `config[mode]["early_stopping"]`, which every shipped config sets but
which `src/training/base_trainer.py` never actually reads.

One thing this driver deliberately still does not do, because the module it reuses does not
do it either - `fit.py`'s own docstring lists what it omits, and reimplementing it here would
mean no longer reusing the gate-verified loop:

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
import functools
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from ..config import paths as stec_paths
from ..data.aggregated_dataset import AggregatedSplitDataset, collate_assembled_batch
from ..data.day_reader import read_day
from ..data.feature_layout import FeatureLayout, layout_from_feature_control
from ..data.spherical_harmonics import SphericalHarmonics
from ..data.splits import (
    EpochRandomSampler,
    ResampledEpochBatches,
    get_fixed_subset_indices,
)
from ..data.transforms import FeatureAssembler
from ..models.architectures import BayesianResNetSTEC, load_checkpoint
from .checkpointing import fit_with_best_checkpoint
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

    Shuffled once, with `seed`, and returned as a plain list - not a live `DataLoader`.
    `fit`/`fit_with_best_checkpoint` re-iterate this same list object every epoch (neither
    has a per-epoch reshuffle hook, by design - `fit.py`'s docstring: it keeps only what
    changes the numbers a checkpoint would produce), so every epoch of a multi-epoch run
    trains on **the same row order**, not a fresh shuffle per epoch.

    **This is a known, unverified divergence from the source**, not an equivalent
    reformulation: `TrainManager.train_epoch` (`src/training/train_manager.py`) iterates a
    live `DataLoader(shuffle=True)` fresh every epoch, and `DataLoader.__iter__` draws a new
    permutation on every call - even from the same seeded `Generator` - so the source
    reshuffles every epoch and this driver does not
    (`tests/training/test_run_training.py::test_materialize_batches_returns_the_same_order_every_call`
    demonstrates the fixed-seed determinism this relies on; a companion check against a live
    `DataLoader` shows the source's behaviour is different, not equivalent). Gate C's fixed
    3-6 epoch synthetic check does not exercise this - both sides there were handed the
    identical fixed batches - so it has not caught this. Whether the difference is large
    enough to matter for a real 50-150 epoch fine-tune has not been measured: doing so needs
    a real fine-tune day's `loss_history.csv` to compare against, which is the kind of
    training-loop equivalence check `docs/revision/src_deletion_runbook.md` requires before
    trusting this driver as a full replacement for a multi-epoch run.
    """
    dataset = TensorDataset(inputs, targets)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )
    return [(x.to(device), y.to(device)) for x, y in loader]


def build_pretrain_batches(
    config: dict,
    assembler: FeatureAssembler,
    batch_size: int,
    seed: int,
    device: torch.device,
    space_weather: Path | None,
) -> tuple[ResampledEpochBatches, list[Batch], int, int]:
    """`train_batches`/`val_batches` for `mode: pretrain`.

    Mirrors `src/data_loader/loaders.py::get_data_loaders`: `EpochRandomSampler` draws
    `data.train_subset_size` rows with replacement from the full train aggregate each epoch,
    and `get_fixed_subset_indices` draws a fixed `data.val_size` subset from the val
    aggregate once - just sourced from `AggregatedSplitDataset` instead of `H5Dataset`.

    `pretrain.num_workers`/`pretrain.prefetch_factor` (config/config_BNN.yaml: 12 / 4) now
    reach the `DataLoader`, not just `src/`'s: `AggregatedSplitDataset` opens its h5py handle
    lazily, one per worker process, rather than once in `__init__` (see that class's own
    docstring), so it no longer needs `num_workers=0` to stay fork-safe. Reading
    `data/train.h5`'s random single-row access pattern under one synchronous reader measured
    659,241 bytes read per 80-byte row (a full 8,192-row/655 KB chunk per row - see the
    module docstring) and only ~1,350 rows/sec; `src/data_loader/datasets.py`'s `H5Dataset`
    reads the same file the same way but under `num_workers=12`, giving the storage many
    outstanding reads at once, and that concurrency - not a change in bytes read - is what
    makes it ~12x faster on this host. `persistent_workers=False`, matching
    `src/data_loader/loaders.py`'s own `# FIXED: Disable to prevent H5 file handle leaks` -
    each epoch's workers open a fresh handle and let it go rather than holding one across
    `ResampledEpochBatches`' per-epoch resample.
    """
    swi_path = space_weather if space_weather is not None else stec_paths.OMNI_INDICES
    train_dataset = AggregatedSplitDataset(
        stec_paths.aggregated_split_h5("train"), space_weather_path=swi_path
    )
    val_dataset: Dataset = AggregatedSplitDataset(
        stec_paths.aggregated_split_h5("val"), space_weather_path=swi_path
    )
    collate = functools.partial(collate_assembled_batch, assembler=assembler)

    # Absent in every existing test fixture's `config["pretrain"]` block (see
    # tests/training/test_run_training_pretrain.py), so this defaults to the old, always-safe
    # num_workers=0 rather than requiring every caller to opt in explicitly.
    num_workers = int(config["pretrain"].get("num_workers", 0))
    # DataLoader raises if prefetch_factor is set without num_workers>0.
    prefetch_factor = (
        config["pretrain"].get("prefetch_factor") if num_workers > 0 else None
    )

    train_subset_size = int(config["data"]["train_subset_size"])
    logger.info(
        f"pretrain: sampling {train_subset_size:,} of {len(train_dataset):,} train rows "
        f"per epoch, with replacement ({num_workers} worker(s))"
    )
    sampler = EpochRandomSampler(
        train_dataset, replacement=True, num_samples=train_subset_size, base_seed=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        collate_fn=collate,
    )
    train_batches = ResampledEpochBatches(train_loader, sampler, device)

    val_size_config = config["data"].get("val_size", "full")
    if val_size_config != "full":
        cache_path = (
            stec_paths.SUBSET_INDEX_CACHE
            / f"pretrain_val_{val_size_config}_seed{seed}.pt"
        )
        idx = get_fixed_subset_indices(
            val_dataset, int(val_size_config), cache_path, seed=seed
        )
        val_dataset = Subset(val_dataset, idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        collate_fn=collate,
    )
    val_batches = [
        (inputs.to(device), targets.to(device)) for inputs, targets in val_loader
    ]

    return train_batches, val_batches, train_subset_size, len(val_dataset)


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
    if config.get(mode, {}).get("save_model_every_epoch", False):
        raise NotImplementedError(
            f"{mode}.save_model_every_epoch is not ported: "
            "fit_with_best_checkpoint only ever snapshots the best-so-far epoch, not "
            "every epoch (src/training/training_utils.py's TrainingUtils.save_checkpoint "
            "would write one file per epoch instead). Every shipped config sets this to "
            "False, so this is refused rather than silently only keeping the best."
        )

    layout, assembler = build_layout_and_assembler(config)
    seed = config["random_seed"]
    batch_size = config[mode]["batchsize"]

    # `data.train_subset_size` is the same signal `src/data_loader/loaders.py::
    # get_data_loaders` branches on (`elif train_subset and train_subset < len(ds):
    # use EpochRandomSampler`) - real pretrain configs set it (config/config_BNN.yaml:
    # 500_000), a bare `mode: pretrain` with no such key does not ask for the aggregate at
    # all. Branching on the mode alone would silently redirect every existing `mode:
    # pretrain` fine-tune-checkpoint fixture in tests/training/test_run_training.py at
    # data/train.h5 instead of its tmp_path fixture - this keeps that path unchanged.
    use_aggregate = mode == "pretrain" and config.get("data", {}).get(
        "train_subset_size"
    )
    if use_aggregate:
        # --train-days/--val-days are accepted (main() always resolves them, defaulting to
        # config['year']/['doy']) but not meaningful here - the aggregate already spans
        # every split day, so this mode ignores them rather than pretending a day filter
        # applies to it.
        train_batches, val_batches, n_train, n_val = build_pretrain_batches(
            config, assembler, batch_size, seed, device, space_weather
        )
        logger.info(f"{mode}: {n_train:,} train rows/epoch / {n_val:,} val rows")
    else:
        train_inputs, train_targets = read_and_assemble(
            train_days, "train", assembler, database_root, space_weather
        )
        val_inputs, val_targets = read_and_assemble(
            val_days, "val", assembler, database_root, space_weather
        )
        n_train, n_val = len(train_inputs), len(val_inputs)
        logger.info(
            f"{mode}: {n_train:,} train / {n_val:,} val rows over "
            f"{len(train_days)} train day(s), {len(val_days)} val day(s)"
        )
        train_batches = materialize_batches(
            train_inputs,
            train_targets,
            batch_size,
            shuffle=True,
            seed=seed,
            device=device,
        )
        val_batches = materialize_batches(
            val_inputs, val_targets, batch_size, shuffle=False, seed=seed, device=device
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

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = build_optimizer(config, mode, trainable_params)
    scheduler = get_scheduler(config, optimizer, compat=scheduler_compat)
    loss_fn = AnnealedGaussianNLLWithKL.from_config(config)

    # float("inf") matches base_trainer.py's own `.get("patience", float("inf"))` - not
    # `config[mode]["early_stopping"]`, which every shipped config sets but which that
    # function never actually reads (see the module docstring above).
    patience = config[mode].get("patience", float("inf"))
    result = fit_with_best_checkpoint(
        model,
        optimizer,
        scheduler,
        loss_fn,
        train_batches,
        val_batches,
        epochs=config[mode]["epochs"],
        seed=seed,
        patience=patience,
    )
    history = result.history

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(exist_ok=True)
    checkpoint_path = (
        model_dir / f"{mode}_{config['model']['model_type']}_seed{seed:02}.pth"
    )
    torch.save(
        {
            "model_state_dict": result.best_state_dict,
            "epoch": result.best_epoch,
        },
        checkpoint_path,
    )

    history_path = output_dir / "loss_history.csv"
    history.to_csv(history_path, index=False)

    if result.stopped_early:
        logger.info(
            f"Early stopping after {len(history)} epoch(s) "
            f"(no improvement for {patience} epochs)"
        )
    logger.info(
        f"{mode} finished: {len(history)} epoch(s) run, "
        f"best epoch={result.best_epoch} (val_loss={result.best_val_loss:.4f}), "
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
