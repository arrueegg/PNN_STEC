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

**Median is the headline statistic; the 10 m outcome-based outlier rule is dropped; the
population is the same 4-method x 2-weighting common set Tables 5-7 use (owner decision,
applied here 2026-09-15 - see `stec.analysis.storm_stratification`'s module docstring
for the full argument and the evidence it is based on).** `tail_table` already reported
both `mean` and `median` - median is now reported first, as the headline column, with
`mean` kept alongside for the sensitivity comparison
`docs/revision/positioning_reporting.md` argues from. `component_table` (horizontal vs
vertical error) previously reported only the mean and is extended with the matching
median columns, since a mean-only table for exactly the R2.7 "vertical/horizontal error
behavior" comment would carry the same single-outlier sensitivity this module exists to
avoid elsewhere. The 10 m rule (`stec.positioning.metrics.exclude_outlier_station_days`)
is dropped for the same reason Tables 5-7 dropped it: filtering on the outcome being
measured is not neutral between methods. `load()` restricts to
`common_set_positioning.coverage_common_station_days()` instead, so every method in
this module's tables is reported over the same station-days.

Usage::

    python -m stec.analysis.positioning_robustness
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from .common_set_positioning import coverage_common_station_days

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


def load(
    summary_path: Path, common_station_days: pd.MultiIndex | None = None
) -> pd.DataFrame:
    """Load the per-station-day summary, restricted to the common set.

    Restricted to the same 4-method x 2-weighting common set Tables 5-7 use
    (`common_set_positioning.coverage_common_station_days`) rather than each method's
    own coverage - see the module docstring. Computed from the live checkout's own tree
    by default (`None`); tests pass a synthetic index directly, the same
    parameter-injection pattern `weighting_ablation.py` and `positioning_distributions.py`
    use for the identical restriction. No outcome-based (10 m) exclusion is applied;
    that rule was dropped here for the same reason Tables 5-7 dropped it.
    """
    runs = pd.read_csv(summary_path)
    if common_station_days is None:
        common_station_days = coverage_common_station_days()
    n_before = len(runs)
    runs = runs.set_index(["station", "doy"])
    runs = runs[runs.index.isin(common_station_days)].reset_index()
    logger.info(
        f"restricted to the common set solved by all methods under both weightings: "
        f"{len(runs):,} of {n_before:,} station-day rows kept"
    )
    runs["Method"] = runs["method"].map(METHOD_LABELS)
    return runs.dropna(subset=["Method"])


def tail_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Distribution of the daily 3D RMS across station-days, per method.

    `median` is the headline statistic (reported first); `mean` is kept alongside for
    the sensitivity comparison `docs/revision/positioning_reporting.md` argues from.
    """
    rows = []
    for method, group in runs.groupby("Method"):
        error = group["error_3d_rms"]
        row = {
            "station_days": len(group),
            "median": error.median(),
            "mean": error.mean(),
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
    """Horizontal vs vertical error, per method.

    Median is the headline statistic; the matching `_mean` columns are kept only for
    the sensitivity comparison, not as a second number to report (owner decision, see
    the module docstring).
    """
    rows = []
    for method, group in runs.groupby("Method"):
        horizontal_median = group["error_2d_rms"].median()
        vertical_median = group["u_rms"].median()
        horizontal_mean = group["error_2d_rms"].mean()
        vertical_mean = group["u_rms"].mean()
        rows.append(
            pd.Series(
                {
                    "horizontal_2D_median_m": horizontal_median,
                    "vertical_up_median_m": vertical_median,
                    "vertical_to_horizontal_median_ratio": (
                        vertical_median / horizontal_median
                    ),
                    "east_median_m": group["e_rms"].median(),
                    "north_median_m": group["n_rms"].median(),
                    "horizontal_2D_mean_m": horizontal_mean,
                    "vertical_up_mean_m": vertical_mean,
                    "vertical_to_horizontal_mean_ratio": (
                        vertical_mean / horizontal_mean
                    ),
                    "east_mean_m": group["e_rms"].mean(),
                    "north_mean_m": group["n_rms"].mean(),
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
    print(
        f"=== Tail behaviour of the daily 3D RMS [m] (N={len(runs):,} common set) ==="
    )
    print(tails.round(3).to_string())
    print("median is the headline column; mean is the sensitivity comparison.")

    components = component_table(runs)
    components.to_csv(args.output_dir / "error_components.csv")
    print("\n=== Horizontal vs vertical error [m] ===")
    print(components.round(3).to_string())
    print(
        "_median columns are the headline; _mean columns are the sensitivity comparison."
    )

    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
