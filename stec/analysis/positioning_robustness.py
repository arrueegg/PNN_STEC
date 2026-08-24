"""Positioning robustness metrics beyond the daily 3D RMS (R2.7b).

Ported from ``src/analysis/positioning_robustness.py`` in the live PNN_STEC checkout,
which supplies evidence for the remainder of reviewer comment R2.7:

    "For PPP applications, daily RMS statistics are insufficient. The authors should
     evaluate convergence time, vertical/horizontal error behavior, tail errors, and
     storm-time positioning performance."

Storm-time behaviour is covered by ``storm_stratification.py``. This module supplies the
other two that the existing solutions already contain:

* **tail errors** - the 95th percentile of the 3D error is already computed per
  station-day, plus the empirical distribution over station-days (p50/p90/p95/p99 and the
  fraction of station-days beyond fixed thresholds).
* **vertical vs horizontal** - the up component is reported separately from the 2D error,
  since ionospheric residuals project mainly into height.

Convergence time is not derivable from these files and is not meaningful for the
kinematic, daily-reprocessed single-frequency PPP used here; that part of the comment is
answered in the text rather than with a number.

The 10 m station-day outlier rule (Figure 12 / Table 5) is reused from
`stec.positioning.metrics.exclude_outlier_station_days` rather than reimplemented.

Usage::

    python -m stec.analysis.positioning_robustness
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from ..positioning import metrics as pm

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# Thresholds a single-frequency user would care about exceeding.
TAIL_THRESHOLDS_M = (2.0, 3.0, 5.0)

# The two positioning trees this module resolves between, mirroring
# `stec.analysis.positioning_summary.canonical_positioning_summary`. Duplicated rather
# than imported - see the identical comment in `storm_stratification.py`, which needs the
# same resolution and has no shared `stec/analysis/paths.py` to pull it from yet.
#
# 2026-08-24: repointed from `positioning_runs/full_coverage/multiday_summary.csv` to
# `positioning_coverage`'s own rebuilt output - see the account in
# `stec/analysis/positioning_summary.py`'s module docstring and `storm_stratification.py`.
FULL_COVERAGE_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
# Frozen - nothing regenerates this any more; kept as the record of what the submitted
# paper reported.
PUBLISHED_SUMMARY = (
    paths.LEGACY_MULTIDAY
    / "positioning_runs"
    / "comparison_3way"
    / "multiday_summary.csv"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_robustness", rebuilt=True)


def canonical_positioning_summary(prefer: Path | None = None) -> Path:
    """The per-station-day positioning table this analysis should read."""
    if prefer is not None:
        return prefer
    if FULL_COVERAGE_SUMMARY.exists():
        logger.info(f"positioning input: {FULL_COVERAGE_SUMMARY} (full coverage)")
        return FULL_COVERAGE_SUMMARY
    logger.warning(
        f"{FULL_COVERAGE_SUMMARY} not found - falling back to {PUBLISHED_SUMMARY}, "
        "which omits the station-days recovered from RINEX."
    )
    return PUBLISHED_SUMMARY


def load(summary_path: Path) -> pd.DataFrame:
    runs = pd.read_csv(summary_path)
    runs = pm.exclude_outlier_station_days(runs)
    runs["Method"] = runs["method"].map(METHOD_LABELS)
    return runs.dropna(subset=["Method"])


def tail_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the daily 3D RMS across station-days, per method."""
    rows = []
    for method, group in runs.groupby("Method"):
        error = group["error_3d_rms"]
        row = {
            "station_days": len(group),
            "mean": error.mean(),
            "median": error.median(),
            "p90": error.quantile(0.90),
            "p95": error.quantile(0.95),
            "p99": error.quantile(0.99),
            # The per-epoch tail inside each station-day, averaged over days.
            "mean_daily_95th_pct": group["error_3d_95th"].mean(),
        }
        for threshold in TAIL_THRESHOLDS_M:
            row[f"frac_above_{threshold:g}m_%"] = 100 * (error > threshold).mean()
        rows.append(pd.Series(row, name=method))
    return pd.DataFrame(rows).reindex(METHOD_ORDER)


def component_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Horizontal vs vertical error, per method."""
    rows = []
    for method, group in runs.groupby("Method"):
        horizontal = group["error_2d_rms"].mean()
        vertical = group["u_rms"].mean()
        rows.append(
            pd.Series(
                {
                    "horizontal_2D_rms": horizontal,
                    "vertical_up_rms": vertical,
                    "vertical_to_horizontal_ratio": vertical / horizontal,
                    "east_rms": group["e_rms"].mean(),
                    "north_rms": group["n_rms"].mean(),
                },
                name=method,
            )
        )
    return pd.DataFrame(rows).reindex(METHOD_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=canonical_positioning_summary(),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs = load(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tails = tail_table(runs)
    tails.to_csv(args.output_dir / "tail_distribution.csv")
    print("=== Tail behaviour of the daily 3D RMS [m] ===")
    print(tails.round(3).to_string())

    components = component_table(runs)
    components.to_csv(args.output_dir / "error_components.csv")
    print("\n=== Horizontal vs vertical error [m] ===")
    print(components.round(3).to_string())

    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
