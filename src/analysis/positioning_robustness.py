"""Positioning robustness metrics beyond the daily 3D RMS.

Evidence for the remainder of reviewer comment R1.7:

    "For PPP applications, daily RMS statistics are insufficient. The authors
     should evaluate convergence time, vertical/horizontal error behavior, tail
     errors, and storm-time positioning performance."

Storm-time behaviour is covered by `storm_stratification.py`. This script
supplies the other two that the existing solutions already contain:

* **tail errors** - the 95th percentile of the 3D error is already computed per
  station-day, plus the empirical distribution over station-days (p50/p90/p95/p99
  and the fraction of station-days beyond fixed thresholds).
* **vertical vs horizontal** - the up component is reported separately from the
  2D error, since ionospheric residuals project mainly into height.

Convergence time is not derivable from these files and is not meaningful for the
kinematic, daily-reprocessed single-frequency PPP used here; that part of the
comment is answered in the text rather than with a number.

Usage::

    python src/analysis/positioning_robustness.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from paths import canonical_positioning_summary

logger = logging.getLogger(__name__)

# Same rule as Figure 12 / Table 5.
OUTLIER_3D_RMS_M = 10.0

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


def load(summary_path: Path) -> pd.DataFrame:
    runs = pd.read_csv(summary_path)
    runs = runs[runs["error_3d_rms"] <= OUTLIER_3D_RMS_M].copy()
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
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("multiday_results/positioning_robustness"),
    )
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

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
