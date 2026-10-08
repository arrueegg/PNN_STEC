"""Convert the store's VTEC + Mapping columns from MSLM to SLM slant mapping, in place.

The VTEC baseline is trained on the STEC database's `vtec` column, which CamaliotGnss
derived from STEC with the standard single-layer mapping at 450 km (SLM). The baseline
was nevertheless mapped back to slant with MSLM (506.7 km, alpha 0.9782), leaving a -6.9%
bias at 5-10 deg elevation. Decision 2026-10-05: the VTEC + Mapping baseline uses SLM; the
IGS GIM baseline stays on MSLM (this module never touches `gim_stec`).

Mapping is a per-observation multiplicative factor applied to the VTEC mean and, linearly,
to its std-dev columns, so the conversion is exact and needs no re-inference:

    new = old * SLM_MF(satele) / MSLM_MF(satele)

Safety against double-application, which would silently corrupt a baseline: a converted
file is recorded in a manifest CSV *and* carries a `vtec_mapping=SLM` key in its parquet
schema metadata, so the guard survives a lost manifest and a crash between the file
replace and the manifest append. Before a file is rewritten, its four original VTEC
columns are saved to a backup tree (atomic write) so the change is reversible.

Usage::

    python -m stec.inference.convert_vtec_mapping_slm --dry-run
    python -m stec.inference.convert_vtec_mapping_slm
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..baselines.gim import MappingFunction
from ..config import paths

logger = logging.getLogger(__name__)

VTEC_COLUMNS = (
    "vtec_model_stec",
    "vtec_model_stec_total_unc",
    "vtec_model_stec_aleatoric_unc",
    "vtec_model_stec_epistemic_unc",
)
ELEVATION_COLUMN = "satele"
MAPPING_METADATA_KEY = b"vtec_mapping"
CONVERTED_METADATA_VALUE = b"SLM"

PARTITIONS = (
    "finetuned_stec/own",
    "finetuned_stec/madrigal",
    "pretrained_stec/own",
    "pretrained_stec/madrigal",
    "pretrained_stec_resnet_bnn_nll/own",
)

MANIFEST_FIELDS = (
    "path",
    "rows",
    "mean_factor",
    "sha256_before",
    "sha256_after",
    "columns_converted",
    "converted_at_utc",
)


def mapping_ratio(satele_deg: np.ndarray) -> np.ndarray:
    """SLM / MSLM mapping factor per observation (float64)."""
    elevation_rad = np.radians(np.asarray(satele_deg, dtype=np.float64))
    slm = MappingFunction("SLM").get_mapping_factor(elevation_rad)
    mslm = MappingFunction("MSLM").get_mapping_factor(elevation_rad)
    return slm / mslm


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def is_marked_converted(path: Path) -> bool:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    return metadata.get(MAPPING_METADATA_KEY) == CONVERTED_METADATA_VALUE


def vtec_columns_present(path: Path) -> list[str]:
    names = set(pq.ParquetFile(path).schema.names)
    return [column for column in VTEC_COLUMNS if column in names]


def read_manifest_paths(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with open(manifest_path, newline="") as handle:
        return {row["path"] for row in csv.DictReader(handle)}


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    is_new = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def write_table_atomically(
    table: pa.Table, destination: Path, row_group_size: int
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(destination.name + ".tmp")
    try:
        pq.write_table(
            table, temp_path, compression="snappy", row_group_size=row_group_size
        )
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def convert_day_file(
    day_file: Path, store_root: Path, backup_root: Path, manifest_path: Path
) -> dict | None:
    """Convert one day file. Returns its manifest row, or None if it was skipped."""
    relative_path = str(day_file.relative_to(store_root))
    if relative_path in read_manifest_paths(manifest_path):
        logger.info(f"skip {relative_path}: already listed in the manifest")
        return None
    if is_marked_converted(day_file):
        logger.warning(
            f"skip {relative_path}: carries vtec_mapping=SLM metadata but is not in the "
            "manifest (crash between replace and manifest append?) - not converting twice"
        )
        return None
    columns = vtec_columns_present(day_file)
    if not columns:
        return None

    sha_before = file_sha256(day_file)
    parquet_file = pq.ParquetFile(day_file)
    row_group_size = max(parquet_file.metadata.row_group(0).num_rows, 1)
    table = parquet_file.read()

    backup_table = table.select(columns)
    write_table_atomically(
        backup_table, backup_root / relative_path, row_group_size=row_group_size
    )

    ratio = mapping_ratio(table[ELEVATION_COLUMN].to_numpy(zero_copy_only=False))
    for column in columns:
        column_index = table.schema.get_field_index(column)
        original = table[column].to_numpy(zero_copy_only=False).astype(np.float64)
        converted = (original * ratio).astype(
            table.schema.field(column).type.to_pandas_dtype()
        )
        table = table.set_column(
            column_index, table.schema.field(column), pa.array(converted)
        )

    metadata = dict(table.schema.metadata or {})
    metadata[MAPPING_METADATA_KEY] = CONVERTED_METADATA_VALUE
    table = table.replace_schema_metadata(metadata)
    write_table_atomically(table, day_file, row_group_size=row_group_size)

    row = {
        "path": relative_path,
        "rows": table.num_rows,
        "mean_factor": f"{float(np.nanmean(ratio)):.8f}",
        "sha256_before": sha_before,
        "sha256_after": file_sha256(day_file),
        "columns_converted": ";".join(columns),
        "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    append_manifest_row(manifest_path, row)
    return row


def list_day_files(store_root: Path, partitions: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for partition in partitions:
        files.extend(sorted((store_root / partition).glob("year=*/doy=*.parquet")))
    return files


def dry_run(day_files: list[Path], store_root: Path, manifest_path: Path) -> None:
    already_listed = read_manifest_paths(manifest_path)
    per_partition: dict[str, dict[str, float]] = {}
    for day_file in day_files:
        relative = day_file.relative_to(store_root)
        partition = "/".join(relative.parts[:2])
        stats = per_partition.setdefault(
            partition,
            {
                "files": 0,
                "with_vtec": 0,
                "already_done": 0,
                "rows": 0,
                "bytes": 0,
                "backup_bytes": 0,
            },
        )
        stats["files"] += 1
        columns = vtec_columns_present(day_file)
        if not columns:
            continue
        stats["with_vtec"] += 1
        if str(relative) in already_listed or is_marked_converted(day_file):
            stats["already_done"] += 1
            continue
        parquet_file = pq.ParquetFile(day_file)
        size = day_file.stat().st_size
        stats["rows"] += parquet_file.metadata.num_rows
        stats["bytes"] += size
        stats["backup_bytes"] += size * len(columns) / len(parquet_file.schema.names)
    for partition, stats in per_partition.items():
        print(
            f"{partition}: {stats['files']} files, {stats['with_vtec']} with vtec columns, "
            f"{stats['already_done']} already converted, {stats['rows']:,} rows to convert, "
            f"{stats['bytes'] / 1e9:.1f} GB rewritten, ~{stats['backup_bytes'] / 1e9:.1f} GB backup"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--store-root", type=Path, default=paths.LEGACY_PREDICTIONS)
    parser.add_argument("--partitions", nargs="+", default=list(PARTITIONS))
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=None,
        help="default: <store-root>/_backup_vtec_mslm_<YYYYMMDD>",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="default: <store-root>/_backup_vtec_mslm_conversion_manifest.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store_root: Path = args.store_root
    backup_root = args.backup_root or store_root / "_backup_vtec_mslm_20261005"
    manifest_path = (
        args.manifest or store_root / "_backup_vtec_mslm_conversion_manifest.csv"
    )

    day_files = list_day_files(store_root, tuple(args.partitions))
    if args.dry_run:
        dry_run(day_files, store_root, manifest_path)
        return

    converted = 0
    for day_file in day_files:
        row = convert_day_file(day_file, store_root, backup_root, manifest_path)
        if row is not None:
            converted += 1
            if converted % 25 == 0:
                logger.info(f"converted {converted} files, latest {row['path']}")
    logger.info(f"converted {converted} files; manifest at {manifest_path}")


if __name__ == "__main__":
    main()
