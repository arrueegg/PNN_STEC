"""Per-observation cache of the pretrained model's held-out test set, feeding Figures 4-9.

Figures 4-9 (`fig_pred_density`, `fig_residuals_elev`, `fig_residuals_lat`,
`fig_residuals_localtime`, `fig_residuals_year_month`, `fig_uncertainty` in
`stec.viz.manuscript_figures`) were built from `src/inference_testset.py`'s single call to
`plot_test_metrics(test_df, ...)` on the pretrained model's *entire* test set -
`test_df` is never filtered by year before that call. That set is
`predictions/pretrained_stec/own/` in the prediction store: 544 sampled days spanning
2014-2024 (~30/year for 2014-2023, all 242 of 2024), 10,000,000 observations total
(`data.test_size: 10000000` in the paper's pretrain config - a doubled-duty partition,
also read one year at a time by the `uncertainty_calibration_pretrained` stage).

Every other `stec.analysis` module that streams the store accumulates *sums* (RMSE, MAE,
counts), because that is all its output needs, and a running sum never grows with the
number of days read. These six figures cannot work that way: five are boxplots binned by
elevation, latitude, local time or month, and the sixth is a density scatter, so all six
need the actual per-observation values - there is no way to draw an exact box-and-whisker
plot, or a hexbin, from a stream of sums alone.

The 10,000,000-row corpus this module reads is two orders of magnitude smaller than the
580 M-row `finetuned_stec` store that OOM-killed the original analysis driver (CLAUDE.md),
so holding it - narrowed to the columns Figures 4-9 actually need - is a bounded, disclosed
cost: roughly 450 MB of raw float32/int32 data (10 M rows x 11 columns x 4 bytes), well
inside the 0.8-1.3 GB peaks already normal for other streaming analyses here. It is still
read one day at a time via `prediction_store.iter_days`, exactly like every other analysis
module in this package; each day's frame is narrowed to `WANTED_COLUMNS` *before*
concatenation, which is what keeps the final concatenation bounded rather than a copy of
the store's full 25-column, 673 MB-on-disk width.

Two outputs, not one, because a parquet file carries no row count in the pipeline's own
provenance record (`.pipeline/<stage>.json` only counts rows for `.csv` - see the
`inference_smoke`/`data_prep_smoke` stage declarations in `stec/pipeline/stages.py`):
`observations.parquet` is the cache `manuscript_figures._build_pretrained_diagnostics_
figures` reads, and `manifest.csv` (one row per year, with day and observation counts) is
what a `min_rows` check - and a human - can look at without opening the parquet.

Usage::

    python -m stec.analysis.pretrained_test_diagnostics
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

# Figures 4-9 are specifically about the pretrained model's own held-out test set (the
# published figures' data source, see the module docstring) - not a parameter another
# caller would plausibly want to vary, unlike the finetuned_stec analyses that loop over
# both datasets.
MODEL_VARIANT = "pretrained_stec"
DATASET = "own"

TRUTH_COLUMN = "true_stec"
PRED_COLUMN = "stec_pred"

# The union of columns Figures 4-9 (stec.viz.manuscript_figures) read from a per-
# observation frame. local_time_hours is requested defensively even though every sampled
# day of this store lacks it (the pretrained model's own metadata does not carry it) -
# fig_residuals_localtime falls back to deriving it from sod/lon_ipp
# (stratified_comparison.add_local_time) when it is absent, so this module does not need
# to know which case it is in.
WANTED_COLUMNS = [
    TRUTH_COLUMN,
    PRED_COLUMN,
    "satele",
    "sm_lat_ipp",
    "local_time_hours",
    "sod",
    "lon_ipp",
    "year",
    "doy",
    "pred_total_unc",
    "pred_epistemic_unc",
    "pred_aleatoric_unc",
]

# The pre-rebuild store, resolved in one place so this file does not become a sixth copy
# of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a reader
# of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "pretrained_test_diagnostics", rebuilt=True
)


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has (same reasoning as
    `daily_metrics._wanted_columns` and `elevation_metrics_finetuned._wanted_columns`)."""
    present = set(pq.ParquetFile(path).schema.names)
    return [column for column in WANTED_COLUMNS if column in present]


def collect(
    store_root: Path,
    years: Sequence[int] | None = None,
    doys: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Stream `pretrained_stec/own` one day at a time, narrowed to `WANTED_COLUMNS`.

    `years`/`doys` default to every day the store holds - the manuscript figures used the
    model's entire test set, not one year of it. Concatenating the per-day frames here is
    bounded specifically because each is narrowed to ~11 columns before it happens; see
    the module docstring for the measured size of the result.
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
            "Figures 4-9 need."
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

    observations = collect(args.store_root, years=args.years, doys=args.doys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_parquet(
        args.output_dir / "observations.parquet", index=False, compression="snappy"
    )
    manifest_table = manifest(observations)
    manifest_table.to_csv(args.output_dir / "manifest.csv", index=False)

    logger.info(
        f"wrote observations.parquet ({len(observations):,} rows, "
        f"{int(manifest_table['n_days'].sum())} days across "
        f"{manifest_table['year'].min()}-{manifest_table['year'].max()}) to "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()
