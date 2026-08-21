"""Per-day, per-elevation-bin RMSE/MAE for the four correction methods, feeding Figure 11.

Figure 11 (`mae_rmse_finetuned`, `src/multiday_evaluation.py::extract_elevation_metrics_
from_experiment` + `generate_aggregate_plots`'s "Combined RMSE/MAE Plot") needs the
across-day standard deviation of RMSE/MAE within each 5-degree elevation bin - the error
bars in the published figure. The only elevation-binned aggregate that already exists
(`stratified_comparison.py`) pools all days into a single RMSE/MAE per bin with no
variance term, by design: it answers "where does the model still beat the alternatives"
(R1.4), not "how much does that vary day to day". Dropping the error bars to reuse it
would be a different figure, not a port of this one.

This module keeps `doy` as the finest unit instead of summing over it: RMSE/MAE are
computed here, per day, per bin, per method, and the across-day mean/std is left for the
figure builder (`stec.viz.manuscript_figures.fig_mae_rmse_finetuned`) to compute from
this table - matching how `fig_positioning_trend` derives its own mean/SEM from a raw
per-station-day frame rather than reading a pre-aggregated one.

Two things are deliberately different from `stratified_comparison.py`, despite the
shared day-at-a-time accumulation pattern:

* Bin edges are the publication's original 5-degree elevation bins (`np.arange(0, 91,
  5)`), not `stratified_comparison.ELEVATION_BINS` - a coarser partition that serves a
  different figure.
* A (day, bin, method) cell is dropped below `MIN_OBSERVATIONS_PER_DAY_BIN` (the
  source's own `if len(group) > 100` guard), because a mean and std over ~240 nearly-empty
  days would print an error bar that is itself just noise.

Streamed one day at a time via `prediction_store.iter_days`, same reasoning as
`stratified_comparison.py` and `station_independence.py`: the finetuned_stec store holds
~400M rows across 242 days, and per-day RMSE/MAE is exact from that stream, not an
approximation.

Usage::

    python -m stec.analysis.elevation_metrics_finetuned
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import paths
from ..inference import prediction_store as ps
from .stratified_comparison import METHODS, TRUTH_COLUMN

logger = logging.getLogger(__name__)

# Ported unchanged from src/multiday_evaluation.py::extract_elevation_metrics_from_
# experiment (`np.arange(0, 91, 5)`) - independent of, and finer than,
# stratified_comparison.ELEVATION_BINS.
ELEVATION_BIN_EDGES = np.arange(0, 91, 5)

# extract_elevation_metrics_from_experiment's own per-day, per-bin minimum
# ("if len(group) > 100"), so a nearly-empty bin does not contribute a noisy day to the
# across-day std this module exists to produce.
MIN_OBSERVATIONS_PER_DAY_BIN = 100

# The pre-rebuild store, resolved in one place so this file does not become a fifth copy
# of an absolute path. paths.py honours STEC_DATA_ROOT / STEC_ARTIFACT_ROOT, so a reader
# of the published code can point it elsewhere without editing source.
DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_OUTPUT_DIR = Path("multiday_results/elevation_metrics_finetuned_rebuilt")


def accumulate_day(frame: pd.DataFrame, doy: int) -> list[dict]:
    """Per (elevation_bin, Method) RMSE/MAE for one day, NaNs excluded pairwise per method.

    Unlike `stratified_comparison.accumulate_day`, this returns finished RMSE/MAE values
    per day rather than sums to be pooled later: the figure this feeds needs the
    *distribution* of each bin's daily RMSE/MAE across days, not a single value pooled
    from every day's sum.
    """
    if TRUTH_COLUMN not in frame.columns or "satele" not in frame.columns:
        return []
    binned = pd.cut(
        frame["satele"],
        bins=ELEVATION_BIN_EDGES,
        labels=ELEVATION_BIN_EDGES[:-1],
        include_lowest=True,
    )
    truth = frame[TRUTH_COLUMN].to_numpy(float)
    rows: list[dict] = []
    for method_column, method in METHODS.items():
        if method_column not in frame.columns:
            continue
        error = frame[method_column].to_numpy(float) - truth
        # Pairwise exclusion, same reasoning as stratified_comparison: one method's NaN
        # prediction must not drop the observation from another method's tally.
        valid = np.isfinite(error)
        if not valid.any():
            continue
        part = pd.DataFrame(
            {
                "bin": binned.to_numpy()[valid],
                "_sq": error[valid] ** 2,
                "_abs": np.abs(error[valid]),
            }
        ).dropna(subset=["bin"])
        if part.empty:
            continue
        grouped = part.groupby("bin", observed=True).agg(
            n=("_sq", "size"), sum_sq=("_sq", "sum"), sum_abs=("_abs", "sum")
        )
        for bin_value, row in grouped.iterrows():
            if row["n"] <= MIN_OBSERVATIONS_PER_DAY_BIN:
                continue
            rows.append(
                {
                    "doy": doy,
                    "elevation_bin": float(bin_value),
                    "Method": method,
                    "n": int(row["n"]),
                    "RMSE": float(np.sqrt(row["sum_sq"] / row["n"])),
                    "MAE": float(row["sum_abs"] / row["n"]),
                }
            )
    return rows


def _wanted_columns(path: Path) -> list[str]:
    """Restrict the read to columns this day's file actually has (same reasoning as
    `daily_metrics._wanted_columns` and `stratified_comparison._wanted_columns`: not
    every day carries every baseline)."""
    present = set(pq.ParquetFile(path).schema.names)
    wanted = [TRUTH_COLUMN, "satele", *METHODS]
    return [column for column in wanted if column in present]


def collect(
    model_variant: str,
    dataset: str,
    store_root: Path,
    doys: list[int] | None = None,
) -> pd.DataFrame:
    """Stream one dataset's stored days, accumulating per-day-per-bin-per-method RMSE/MAE.

    Raises `FileNotFoundError` if `model_variant`/`dataset` has no store under
    `store_root`, exactly as `prediction_store.day_paths` does - the caller decides
    whether a missing dataset is fatal or just absent (`main` below treats it as absent).
    """
    rows: list[dict] = []
    for path in ps.day_paths(model_variant, dataset, doys=doys, root=store_root):
        doy = int(path.stem.split("=")[1])
        wanted = _wanted_columns(path)
        if TRUTH_COLUMN not in wanted or "satele" not in wanted:
            logger.warning(f"doy={doy:03d} is missing satele/{TRUTH_COLUMN}, skipping")
            continue
        _, _, frame = next(
            ps.iter_days(
                model_variant, dataset, doys=[doy], columns=wanted, root=store_root
            )
        )
        rows.extend(accumulate_day(frame, doy))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default="finetuned_stec")
    parser.add_argument("--datasets", type=str, nargs="*", default=["own", "madrigal"])
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

    frames = []
    for dataset in args.datasets:
        try:
            table = collect(
                args.model_variant, dataset, args.store_root, doys=args.doys
            )
        except FileNotFoundError:
            logger.warning(
                f"no {args.model_variant}/{dataset} store under {args.store_root}"
            )
            continue
        if table.empty:
            continue
        frames.append(table.assign(dataset=dataset))

    if not frames:
        raise RuntimeError(
            "no observations were read for any dataset - the prediction store is empty "
            "or missing under the given root. Run the inference pass that populates it "
            "first; a bare empty-CSV write here would hide which step actually failed."
        )
    combined = pd.concat(frames, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "per_day_by_elevation.csv", index=False)
    logger.info(
        f"wrote per_day_by_elevation.csv ({len(combined):,} day-bin-method rows) "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
