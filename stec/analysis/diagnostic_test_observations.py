"""Per-observation cache of the pretrained model's held-out test set, extended with the
columns `pretrained_test_diagnostics` does not carry - feeds `stec.viz.diagnostic_figures`.

`pretrained_test_diagnostics.py` already streams `predictions/pretrained_stec/own`
(544 days, 2014-2024, 10,000,000 observations - see that module's docstring for the size
accounting) into a cache for Figures 4-9, narrowed to the ~11 columns those six figures
read. The diagnostic-plot parity work this module supports needs a few columns that cache
does not carry: `satazi` (azimuth/elevation heatmaps), `lat_ipp` (the geographic spatial
error map - `lon_ipp` is already cached, `lat_ipp` is not) and the space-weather forcing
columns (`Kp_index`, `f107_index`, `Dst-index,_nT`, `AE-index,_nT`, `R_Sunspot_No`,
`ap_index,_nT` - residual-vs-solar-index panels).

This is a second pass over the same 544 files rather than an edit to
`pretrained_test_diagnostics.WANTED_COLUMNS`, deliberately: `stec.viz.manuscript_figures`
(Figures 4-9) reads that module's cache directly and is being worked on concurrently by
another change in this codebase, so widening its column list - and doubling its on-disk
size for consumers that only need the narrower set - is exactly the kind of shared-file
edit to avoid mid-flight. The extra read costs another ~544-file pass over the same
703.9 MB-on-disk store (measured 2026-08-25) and a second, wider parquet cache on disk;
both are one-time costs paid by whoever runs this module, not a recurring cost of every
`pretrained_test_diagnostics` run.

One column the diagnostics this cache feeds want and still cannot get: `sm_lon_ipp`. The
real `pretrained_stec/own` store was written without it (checked directly against the
store's own parquet schema, 2026-08-25 - only `sm_lat_ipp` is present), so the solar-
magnetic-coordinate spatial error map (`src/viz/spatial.py::plot_solar_magnetic_ipp_error_map`)
has no data source in this store and is not ported by this module or by
`stec.viz.diagnostic_figures` - see that module's docstring for the full account.

Usage::

    python -m stec.analysis.diagnostic_test_observations
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..config import paths
from ..inference import prediction_store as ps

logger = logging.getLogger(__name__)

# Same model/dataset as pretrained_test_diagnostics - the diagnostic plots this cache
# feeds are variants of the same family (residuals/uncertainty of the pretrained model's
# own held-out test set), not a different evaluation.
MODEL_VARIANT = "pretrained_stec"
DATASET = "own"

TRUTH_COLUMN = "true_stec"
PRED_COLUMN = "stec_pred"

# The union of columns the diagnostic-plot parity figures read from a per-observation
# frame: everything pretrained_test_diagnostics.WANTED_COLUMNS carries except
# `local_time_hours` (never present in this store - see that module's docstring; derived
# here from `sod`/`lon_ipp` instead via `stratified_comparison.add_local_time`, the same
# fallback `manuscript_figures.fig_residuals_localtime` already uses), plus `satazi`,
# `lat_ipp` and the space-weather forcing columns.
WANTED_COLUMNS = [
    TRUTH_COLUMN,
    PRED_COLUMN,
    "satele",
    "satazi",
    "lat_ipp",
    "lon_ipp",
    "sm_lat_ipp",
    "sod",
    "year",
    "doy",
    "pred_total_unc",
    "pred_epistemic_unc",
    "pred_aleatoric_unc",
    "Kp_index",
    "R_Sunspot_No",
    "Dst-index,_nT",
    "AE-index,_nT",
    "ap_index,_nT",
    "f107_index",
]

DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "diagnostic_test_observations", rebuilt=True
)


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has (same reasoning as
    `pretrained_test_diagnostics._wanted_columns` and `daily_metrics._wanted_columns`)."""
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in WANTED_COLUMNS if column in present]


def collect(
    store_root: Path,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Stream `pretrained_stec/own` one day at a time, narrowed to `WANTED_COLUMNS`.

    `years`/`doys` default to every day the store holds. Mirrors
    `pretrained_test_diagnostics.collect` exactly (same store, same one-file-at-a-time
    read via `prediction_store.iter_days`), only the column list differs.
    """
    day_paths = ps.day_paths(
        MODEL_VARIANT, DATASET, years=years, doys=doys, root=store_root
    )
    if not day_paths:
        raise FileNotFoundError(
            f"No prediction files matched for {MODEL_VARIANT}/{DATASET} "
            f"(years={years}, doys={doys}) under {store_root}"
        )
    logger.info(f"{len(day_paths)} day(s) to read for {MODEL_VARIANT}/{DATASET}")

    frames = []
    for path in day_paths:
        year = int(path.parent.name.split("=")[1])
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if TRUTH_COLUMN not in wanted or PRED_COLUMN not in wanted:
            logger.warning(
                f"{year}-{doy:03d} is missing {TRUTH_COLUMN}/{PRED_COLUMN}, skipping"
            )
            continue
        _, _, frame = next(
            ps.iter_days(
                MODEL_VARIANT,
                DATASET,
                years=[year],
                doys=[doy],
                columns=wanted,
                root=store_root,
            )
        )
        frames.append(frame)

    if not frames:
        raise RuntimeError(
            f"no day under {store_root}/{MODEL_VARIANT}/{DATASET} carried "
            f"{TRUTH_COLUMN}/{PRED_COLUMN} - the store is empty or missing the columns "
            "this cache needs."
        )
    return pd.concat(frames, ignore_index=True)


def manifest(observations: pd.DataFrame) -> pd.DataFrame:
    """One row per year: how many days and observations the cache actually holds."""
    return (
        observations.groupby("year")
        .agg(n_days=("doy", "nunique"), n_observations=("doy", "size"))
        .reset_index()
        .sort_values("year")
    )


def build(
    store_root: Path,
    output_dir: Path,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Collect and persist the cache, returning the observations frame it wrote."""
    observations = collect(store_root, years=years, doys=doys)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(
        output_dir / "observations.parquet", index=False, compression="snappy"
    )
    manifest(observations).to_csv(output_dir / "manifest.csv", index=False)
    return observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these years; default is every year in the store (2014-2024).",
    )
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every day in the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    observations = build(
        args.store_root, args.output_dir, years=args.years, doys=args.doys
    )
    manifest_table = manifest(observations)

    logger.info(
        f"wrote observations.parquet ({len(observations):,} rows, "
        f"{int(manifest_table['n_days'].sum())} days across "
        f"{manifest_table['year'].min()}-{manifest_table['year'].max()}) to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
