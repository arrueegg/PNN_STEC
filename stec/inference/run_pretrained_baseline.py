"""Merge the pretrained model's own prediction into the finetuned_stec store as a
baseline column, for datasets where no driver has ever computed it.

`stec.analysis.daily_metrics` (Tables 3 and 4) and `stec.analysis.stratified_comparison` /
`elevation_metrics_finetuned` (Figure 11) all read the "Pretrained STEC" row from a single
column, `pretrained_stec_pred`, merged into the `finetuned_stec/<dataset>` store -
**never** from the separate `predictions/pretrained_stec/<dataset>` partition directly (see
`daily_metrics.py`'s own `MODELS` comment: "pretrained_stec_pred lives alongside stec_pred
in the finetuned_stec store, not in a separate pretrained_stec partition"). Confirmed by
reading a real file: `predictions/finetuned_stec/own/year=2024/doy=122.parquet` carries
`pretrained_stec_pred` as its only pretrained-related column (no separate uncertainty
columns), and `predictions/pretrained_stec/own/*.parquet` is never opened by
`daily_metrics.collect()` at all.

For "own", that column was written by the legacy `src/compare_stec_vtec_gim.py --pretrained_
baseline` sweep, which loads the pretrained checkpoint and runs it over the same day's data
inline. For "madrigal", no driver - not that legacy script (its pretrained-baseline branch
lives entirely inside its `else: # dataset_type == 'own'` code path, confirmed by reading
it; the Madrigal branch runs only the primary STEC model) and nothing in `stec/` - has ever
computed this column. This is why `predictions/finetuned_stec/madrigal/*.parquet` has no
`pretrained_stec_pred` column on any of its 238 days, and why Table 4's Madrigal half has
never had a Pretrained STEC row: not because `predictions/pretrained_stec/madrigal` was
empty (that partition is not what these analyses read), but because nothing merges its
prediction into the file that is read.

This module is that missing merge, reusing rather than re-running the pretrained model's
Madrigal inference: `stec.inference.run_inference --model-variant pretrained_stec --dataset
madrigal` already produces `predictions/pretrained_stec/<dataset>/year=<Y>/doy=<D>.parquet`
with the model's own `stec_pred` and the full raw geometry frame. This driver reads that
file, verifies it lands on the same rows in the same order as the corresponding
`finetuned_stec/<dataset>` file (two independent readers of the same raw day - own's
`stec.data.day_reader.read_day` vs `stec.data.madrigal_reader.read_madrigal_day`, or for
Madrigal, this reader against the legacy `get_madrigal_data_loader` that built the
`finetuned_stec` file - so alignment is measured, never assumed, exactly the discipline
`stec.inference.run_baselines._verify_alignment` established for the same class of merge),
renames `stec_pred` to `pretrained_stec_pred`, and merges it onto the existing file via
`write_predictions` - which starts from every column already present and refuses to narrow
the schema, so this can never regress the day's GIM/VTEC columns.

Usage::

    python -m stec.inference.run_pretrained_baseline \\
        --dataset madrigal --store-root predictions --doys 2024:132 2024:133
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import paths
from . import prediction_store as ps
from .run_baselines import (
    ALIGNMENT_COLUMNS,
    ALIGNMENT_TOLERANCE,
    ELEVATION_TOLERANCE_DEG,
)

logger = logging.getLogger(__name__)

Day = tuple[int, int]

TARGET_COLUMN = "pretrained_stec_pred"
SOURCE_COLUMN = "stec_pred"

# Only what this merge actually touches: the alignment columns (positional identity
# check) plus the one column being merged. Reading the full ~30-column source frame for a
# ~2M-row day just to keep four columns would cost the same I/O for no benefit.
SOURCE_READ_COLUMNS = (*ALIGNMENT_COLUMNS, "station", SOURCE_COLUMN)

MANIFEST_COLUMNS = ("dataset", "year", "doy", "rows", "status")


def already_merged(path: Path) -> bool:
    """Schema-only check (parquet footer, no row data) - the same discipline
    `scripts/lib/missing_data_selection.py::store_days`'s `required_columns` argument
    documents: a day is "done" by what its schema actually carries, never by file
    existence alone, so a re-run of this driver is a no-op on days it already merged."""
    if not path.exists():
        return False
    return TARGET_COLUMN in pq.ParquetFile(path).schema.names


def _verify_alignment(
    source: pd.DataFrame, existing: pd.DataFrame, year: int, doy: int
) -> None:
    """Refuse to merge unless `source` (the pretrained_stec partition's read of this day)
    landed on the same rows, in the same order, as `existing` (the finetuned_stec file
    already on disk) - the same invariant and the same tolerances
    `stec.inference.run_baselines._verify_alignment` uses for its own baseline merge, only
    adapted to compare two already-written frames instead of a fresh raw read against a
    stored one. Raises rather than returning a bool, for the same reason that module
    gives: a caller that ignored a False here is exactly how a Frankenstein row (one
    observation's STEC prediction merged with a different observation's baseline) would
    happen silently.
    """
    if len(source) != len(existing):
        raise RuntimeError(
            f"{year}-{doy:03d}: row count differs ({len(existing)} in finetuned_stec vs "
            f"{len(source)} in pretrained_stec) - refusing to merge positionally"
        )
    for column in ALIGNMENT_COLUMNS:
        if column not in source.columns or column not in existing.columns:
            continue
        new_values = source[column].to_numpy(dtype=np.float64)
        old_values = existing[column].to_numpy(dtype=np.float64)
        tolerance = (
            ELEVATION_TOLERANCE_DEG if column == "satele" else ALIGNMENT_TOLERANCE
        )
        max_diff = (
            float(np.max(np.abs(new_values - old_values))) if len(new_values) else 0.0
        )
        if max_diff > tolerance:
            raise RuntimeError(
                f"{year}-{doy:03d}: {column} misaligned between pretrained_stec and "
                f"finetuned_stec (max |delta| {max_diff:.4f}) - the two reads landed on "
                "different rows; refusing to merge pretrained_stec_pred onto them"
            )
    if "station" in source.columns and "station" in existing.columns:
        # The store normalises station to uppercase at write time (own emits uppercase
        # already, Madrigal lowercase), so both sides should already agree - upper-case
        # both anyway rather than assume the normalisation ran identically on both files.
        new_station = source["station"].astype(str).str.upper().to_numpy()
        old_station = existing["station"].astype(str).str.upper().to_numpy()
        if not np.array_equal(new_station, old_station):
            raise RuntimeError(
                f"{year}-{doy:03d}: station identity misaligned - refusing to merge"
            )


def merge_pretrained_baseline_for_day(
    year: int,
    doy: int,
    *,
    dataset: str,
    store_root: Path,
    pretrained_store_root: Path | None = None,
) -> dict:
    """Merge one day's `pretrained_stec_pred` column into `finetuned_stec/<dataset>`.

    `status` in the returned row is one of "merged" (wrote the column), "already_merged"
    (no-op, idempotent resume), or "no_pretrained_source" (the `pretrained_stec/<dataset>`
    partition does not have this day yet - not a failure, just not ready; the caller
    decides whether to treat that as fatal).
    """
    if dataset not in ("own", "madrigal"):
        raise ValueError(f"unknown dataset {dataset!r}, expected 'own' or 'madrigal'")

    pretrained_store_root = pretrained_store_root or store_root
    target_path = ps.store_path("finetuned_stec", dataset, year, doy, root=store_root)
    if not target_path.exists():
        raise FileNotFoundError(
            f"{year}-{doy:03d}: no finetuned_stec/{dataset} file at {target_path} - this "
            "driver merges a baseline column onto a day that must already exist."
        )

    if already_merged(target_path):
        logger.info(f"{year}-{doy:03d}: {TARGET_COLUMN} already present - skipping")
        return {
            "dataset": dataset,
            "year": year,
            "doy": doy,
            "rows": 0,
            "status": "already_merged",
        }

    source_path = ps.store_path(
        "pretrained_stec", dataset, year, doy, root=pretrained_store_root
    )
    if not source_path.exists():
        logger.info(
            f"{year}-{doy:03d}: no pretrained_stec/{dataset} source at {source_path} yet "
            "- nothing to merge"
        )
        return {
            "dataset": dataset,
            "year": year,
            "doy": doy,
            "rows": 0,
            "status": "no_pretrained_source",
        }

    source_columns = set(pq.ParquetFile(source_path).schema.names)
    wanted = [c for c in SOURCE_READ_COLUMNS if c in source_columns]
    source = pd.read_parquet(source_path, columns=wanted)
    existing = pd.read_parquet(target_path)

    _verify_alignment(source, existing, year, doy)

    merged = existing.copy()
    merged[TARGET_COLUMN] = source[SOURCE_COLUMN].to_numpy()

    path = ps.write_predictions(
        merged, "finetuned_stec", dataset, year, doy, root=store_root
    )
    logger.info(
        f"{year}-{doy:03d}: merged {TARGET_COLUMN} into {path} ({len(merged):,} rows)"
    )
    return {
        "dataset": dataset,
        "year": year,
        "doy": doy,
        "rows": len(merged),
        "status": "merged",
    }


def write_manifest(manifest: list[dict], path: Path) -> Path:
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
    parser.add_argument("--dataset", choices=["own", "madrigal"], required=True)
    parser.add_argument(
        "--doys", nargs="+", type=_parse_day, required=True, metavar="YYYY:DDD"
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="root holding finetuned_stec/<dataset> (written to) and, unless "
        "--pretrained-store-root overrides it, pretrained_stec/<dataset> (read from). "
        "Defaults to stec.config.paths.PREDICTIONS, the artifacts/ stub - pass this "
        "explicitly for the real store, same trap as run_inference.py/run_baselines.py.",
    )
    parser.add_argument(
        "--pretrained-store-root",
        type=Path,
        default=None,
        help="only if the pretrained_stec/<dataset> source lives under a different root "
        "than --store-root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=paths.PREDICTIONS / "inference_run"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    store_root = args.store_root or ps.DEFAULT_STORE_ROOT
    manifest: list[dict] = []
    for year, doy in args.doys:
        row = merge_pretrained_baseline_for_day(
            year,
            doy,
            dataset=args.dataset,
            store_root=store_root,
            pretrained_store_root=args.pretrained_store_root,
        )
        manifest.append(row)

    manifest_path = write_manifest(
        manifest, args.output_dir / "pretrained_baseline_manifest.csv"
    )
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} day(s))")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
