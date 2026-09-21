"""Data-preparation entry point: the driver `stec/data/*.py` never had.

`day_reader`, `feature_layout`, `transforms`, `normalization` and `splits` are gate-verified
(Gate A: bit-exact against the legacy collation, both layout and values, on real HDF5 - see
those modules' own docstrings) but nothing in `stec/` called them to build a *dataset* - the
real `data/{train,val,test}.h5` (103 GB / smaller) were, and still are, built by pre-rebuild
`src/data_processing/` + `src/utils/preprocessing.py`'s `DataPreprocessor.build_split_h5`
(`docs/revision/task_board.md` S2). This module is that missing wiring for the `stec/` side:
a split name and a list of days in, one partitioned parquet file per day out, under
`<output_dir>/<split>/year=<YYYY>/doy=<DDD>.parquet` - the same layout convention
`stec.inference.prediction_store` already uses for the prediction store.

Two things this driver deliberately does differently from the legacy aggregator, both worth
stating plainly rather than leaving to be discovered by a diff:

* **It writes assembled, normalised tensor columns, not raw ones.** The legacy `train.h5`
  stores raw columns and defers assembly to `CollateWithSH` at batch time, which is what
  makes one aggregate reusable across every `feature_control` choice. This driver instead
  reuses `feature_layout`/`transforms`/`normalization` to assemble at write time (the same
  read-then-assemble `stec.training.run_training` and `stec.inference.run_inference` already
  do per-batch), because nothing under `stec/` has an assembly-at-load-time collator to defer
  to - baking in one layout is the honest trade until one exists. The output is therefore
  tied to the config it was built from; a different `feature_control` or `SH_degree` needs a
  separate `--output-dir`.
* **It does not re-derive which rows belong to which split.** `day_reader.read_day` already
  reads a raw day's `train_idx`/`val_idx`/`test_idx` - written once, historically, by
  `src/data_processing/add_split_indices.py`, which applies its own row filter (elevation,
  plausible VTEC range, DCB sanity) and station-list membership. This driver assumes that has
  already run against the raw database (true today; every real database day this project
  reads from already carries those groups, which is what makes `day_reader`'s own Gate-A
  tests possible without re-deriving them). Re-running `add_split_indices.py` itself is out of
  scope here: it is a destructive in-place write against 740 GB of immutable external data,
  which is not a trade this module's caller should be able to trigger by accident.

Per-epoch pretrain sampling is not this module's job
-----------------------------------------------------
The paper's 150-epoch pretrain draws 500,000 observations *with replacement*, resampled fresh
every epoch, from the full 15-year train split (`config['data']['train_subset_size']`, legacy
`data_loader/samplers.py`'s `EpochRandomSampler(ds, replacement=True,
num_samples=train_subset, base_seed=seed)`). That is a training-time sampling decision over
the *whole* aggregated corpus, not a data-preparation one: what this module writes is the
full, unsubsampled corpus the sampler draws from, exactly as the legacy `train.h5` it stands
in for is unsubsampled. Subsampling here would fix one 500,000-row draw for an entire run
instead of resampling every epoch, which is a materially different training signal, not a
faithful port of one.

The sampler itself needs no porting: `stec.data.splits.EpochRandomSampler` already implements
`replacement=True` reseeded per epoch (`base_seed + epoch`), verified in isolation by
`tests/data/test_splits.py`. That file's
`test_epoch_random_sampler_oversamples_with_replacement_for_pretrain` adds the one property
specific to this use - drawing *more* samples than the dataset has rows, the way 500,000
draws from a train split smaller than that would - as evidence the mechanism is faithful.
What remains unbuilt is the multi-day `Dataset` that would sit between this module's per-day
parquet output and that sampler for an actual many-thousand-day pretrain:
`stec.training.run_training`
reads `train_days`/`val_days` straight off `day_reader` for the few-day fine-tune case it was
built to reproduce (its own module docstring says so explicitly), not the pretrain case, and
wiring that is a `stec/training` change - out of this module's scope, and out of reach of
RESOURCE DISCIPLINE for this task, which needs the real 15-year corpus to test meaningfully.

Usage::

    python -m stec.data.run_data_prep --config path/to/config.yaml --split train
    python -m stec.data.run_data_prep --config path/to/config.yaml --split test \\
        --days 2024:132 2024:133 --output-dir artifacts/datasets --force
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from calendar import monthrange
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml

from ..config import paths
from .day_reader import IDENTITY_COLUMNS, TARGET_COLUMN, read_day
from .feature_layout import FeatureLayout, layout_from_feature_control
from .spherical_harmonics import SphericalHarmonics
from .transforms import FeatureAssembler

logger = logging.getLogger(__name__)

Day = tuple[int, int]

MANIFEST_COLUMNS = ("split", "year", "doy", "rows", "status")


def _distribution_of(config: dict) -> str:
    """Same rule `stec.training.run_training`/`stec.inference.run_inference` use."""
    loss_function = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplac" in loss_function else "gaussian"


def build_layout_and_assembler(config: dict) -> tuple[FeatureLayout, FeatureAssembler]:
    """The input layout a config describes, and the assembler that fills it - identical
    construction to the training and inference drivers, so a dataset this module builds
    agrees column-for-column with what those drivers would assemble live from the same
    config."""
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


def _numeric_tensors(raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Every raw column that is actually a number - station/sat/etc. are read as fixed-width
    byte strings and are never assembler input, matching the training/inference drivers."""
    return {
        name: torch.from_numpy(values).float()
        for name, values in raw.items()
        if values.dtype.kind in "fiu"
    }


def _decode_if_bytes(values: np.ndarray) -> np.ndarray:
    """Identity columns (`station`, `sat`) arrive as fixed-width byte strings out of the
    HDF5 table; everything else `read_day` returns is already numeric."""
    if values.dtype.kind == "S":
        return np.array([value.decode("ascii") for value in values])
    return values


def column_names_for_layout(layout: FeatureLayout) -> list[str]:
    """One name per assembled tensor column, in the layout's own tensor order.

    `FeatureBlock.columns` only carries per-column names for temporal, direction and plain
    scalar blocks; a spherical-harmonic block's terms are `Y_l^m` coefficients with no
    individual physical name, so they are numbered within their block instead.
    """
    names: list[str] = []
    for block in layout.blocks():
        if block.columns:
            names.extend(block.columns)
        else:
            names.extend(f"{block.name}_{i}" for i in range(block.width))
    return names


def assembled_frame(
    raw: dict[str, np.ndarray],
    layout: FeatureLayout,
    assembler: FeatureAssembler,
    year: int,
    doy: int,
) -> pd.DataFrame:
    """Every assembled input column, the raw target, and identity metadata, for one day.

    Column layout mirrors `stec.inference.prediction_store`'s day identity convention: `year`
    and `doy` are the caller's arguments, not whatever `read_day` happened to fill in, so a
    later denormalise-and-truncate bug elsewhere can never shift which day a row is filed
    under.
    """
    if TARGET_COLUMN not in raw:
        raise KeyError(f"read_day did not return a {TARGET_COLUMN!r} column")

    assembled = assembler.assemble(_numeric_tensors(raw))
    frame = pd.DataFrame(assembled.numpy(), columns=column_names_for_layout(layout))
    frame[TARGET_COLUMN] = raw[TARGET_COLUMN]
    for name in IDENTITY_COLUMNS:
        if name in raw:
            frame[name] = _decode_if_bytes(raw[name])
    frame["year"] = int(year)
    frame["doy"] = int(doy)
    return frame


def output_path(output_dir: Path, split: str, year: int, doy: int) -> Path:
    return (
        Path(output_dir) / split / f"year={int(year)}" / f"doy={int(doy):03d}.parquet"
    )


def _parquet_row_count(path: Path) -> int | None:
    """Rows in an existing parquet file, or None if it cannot be read as one.

    Reads only the footer metadata, not the data - the same reason
    `prediction_store.available_days` globs paths instead of opening files: deciding what to
    skip must not cost what skipping is supposed to save.
    """
    if not path.exists():
        return None
    try:
        return pq.ParquetFile(path).metadata.num_rows
    except (OSError, ValueError) as exc:
        # A truncated file from a killed process must not be mistaken for a completed day,
        # or resuming would silently carry a corrupt split forward.
        logger.warning(f"Ignoring unreadable dataset partition {path}: {exc}")
        return None


def expand_month_tokens(tokens: list[str]) -> list[Day]:
    """Every day-of-year in each `YYYY-MM` token, mirroring
    `DataPreprocessor._generate_dates` - the original's own definition of what a month token
    in `*_dates.list` means."""
    days: list[Day] = []
    for token in tokens:
        year_str, month_str = token.split("-")
        year, month = int(year_str), int(month_str)
        _, days_in_month = monthrange(year, month)
        for day_of_month in range(1, days_in_month + 1):
            days.append((year, date(year, month, day_of_month).timetuple().tm_yday))
    return days


def resolve_days(split: str, database_root: Path | None = None) -> list[Day]:
    """The days to build for `split`, from its `*_dates.list` - every day of every listed
    month that actually has a raw database file, mirroring `add_split_indices.get_file_list`'s
    own existence check: a `YYYY-MM` token names a month, not a guarantee every day in it was
    captured."""
    tokens = paths.date_list(split).read_text().split()
    root = Path(database_root) if database_root is not None else paths.STEC_DATABASE
    return sorted(
        (year, doy)
        for year, doy in expand_month_tokens(tokens)
        if (root / str(year) / f"{doy:03d}" / f"ccl_{year}{doy:03d}_30_5.h5").exists()
    )


def build_split(
    split: str,
    days: list[Day],
    config: dict,
    output_dir: Path,
    *,
    database_root: Path | None = None,
    space_weather: Path | None = None,
    resume: bool = True,
) -> list[dict]:
    """Stream `days` into `<output_dir>/<split>/year=/doy=.parquet`, one day at a time.

    Never holds more than one day's rows in memory - the corpus this stands in for is 103 GB
    aggregated, out of reach of this host's 30 GB. Resumable by construction rather than by a
    side manifest: a day already on disk and readable is skipped, so a crash costs only the
    day in flight, and a rerun with the same arguments picks up exactly where it stopped
    without a separate progress file that could itself go stale or corrupt.
    """
    layout, assembler = build_layout_and_assembler(config)
    output_dir = Path(output_dir)
    manifest: list[dict] = []
    built = skipped = 0

    for year, doy in days:
        path = output_path(output_dir, split, year, doy)
        existing_rows = _parquet_row_count(path) if resume else None
        if existing_rows is not None:
            logger.debug(f"{split} {year}-{doy:03d}: already built, skipping ({path})")
            manifest.append(
                {
                    "split": split,
                    "year": year,
                    "doy": doy,
                    "rows": existing_rows,
                    "status": "skipped_exists",
                }
            )
            skipped += 1
            continue

        raw = read_day(
            year,
            doy,
            split=split,
            database_root=database_root,
            space_weather=space_weather,
            with_identity=True,
        )
        frame = assembled_frame(raw, layout, assembler, year, doy)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False, compression="snappy")
        manifest.append(
            {
                "split": split,
                "year": year,
                "doy": doy,
                "rows": len(frame),
                "status": "written",
            }
        )
        built += 1
        logger.info(f"{split} {year}-{doy:03d}: wrote {len(frame):,} rows to {path}")

    logger.info(
        f"{split}: {built} day(s) built, {skipped} already present ({len(days)} total)"
    )
    return manifest


def write_manifest(manifest: list[dict], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    return path


def _parse_day(token: str) -> Day:
    year, doy = token.split(":")
    return int(year), int(doy)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="resolved run config.yaml carrying feature_control and data.SH_degree",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument(
        "--days",
        nargs="*",
        type=_parse_day,
        default=None,
        metavar="YYYY:DDD",
        help="defaults to every day --split's *_dates.list resolves to an existing file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="defaults to stec.config.paths.DATASETS",
    )
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--space-weather", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild every day even if a readable output already exists for it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = yaml.safe_load(args.config.read_text())
    output_dir = args.output_dir or paths.DATASETS
    days = args.days or resolve_days(args.split, args.database_root)
    if not days:
        raise RuntimeError(
            f"no days resolved for split {args.split!r}: checked "
            f"{paths.date_list(args.split)} against "
            f"{args.database_root or paths.STEC_DATABASE}"
        )

    manifest = build_split(
        args.split,
        days,
        config,
        output_dir,
        database_root=args.database_root,
        space_weather=args.space_weather,
        resume=not args.force,
    )

    manifest_path = write_manifest(manifest, output_dir / args.split / "manifest.csv")
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} day(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
