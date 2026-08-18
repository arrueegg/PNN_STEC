"""Regenerate every revision metric table, then index what was produced.

One command to rebuild the numbers behind the response, and an index so a table
can be traced back to the reviewer comment it answers and the script that made
it. Analyses that need artefacts not yet present (an unfinished sweep, an
incomplete oracle batch) are reported as skipped rather than failing the run.

Usage::

    python src/analysis/build_all.py            # tables only
    python src/analysis/build_all.py --figures  # tables, then figures
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# script, reviewer comment, what it answers
ANALYSES = [
    (
        "src/analysis/relative_error_metrics.py",
        "R2.1, R2.2",
        "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
    ),
    (
        "src/analysis/hyperparameter_search_summary.py",
        "R2.5, R2.8b",
        "architecture comparison and hyperparameter sweep from the W&B history",
    ),
    (
        "src/analysis/station_independence.py",
        "R2.3",
        "test-station error against distance to the nearest training station",
    ),
    ("src/analysis/computational_cost.py", "R2.8h", "training and inference cost"),
    (
        # Must precede activity_stratification.py: that analysis reads the
        # corrected IGS GIM daily RMSE this writes.
        "src/analysis/repair_gim_baseline.py --apply",
        "Table 4, R1.4",
        "recompute the IGS GIM baseline against the correct day's IONEX map",
    ),
    (
        # After repair_gim_baseline, so Tables 3 and 4 are derived from the
        # corrected GIM column rather than the published aggregation.
        "src/analysis/daily_metrics.py",
        "Tables 3, 4",
        "per-day and pooled STEC metrics recomputed from the prediction store",
    ),
    (
        "src/analysis/uncertainty_error_relation.py",
        "R2.6, R1.2",
        "predicted uncertainty vs realised error, and the epistemic share",
    ),
    (
        "src/analysis/activity_stratification.py",
        "R1.4",
        "STEC error stratified by Dst and F10.7",
    ),
    (
        "src/analysis/ionex_rms_benchmark.py",
        "R1.6b",
        "predicted uncertainty against the IGS GIM's own IONEX RMS",
    ),
    (
        "src/analysis/ionex_rms_benchmark.py --gim_type CODE",
        "R1.6b",
        "the same benchmark against CODE's finer-resolution RMS",
    ),
    (
        "src/analysis/uncertainty_calibration.py",
        "R1.6",
        "coverage, PIT and CRPS on the own test set",
    ),
    (
        "src/analysis/uncertainty_calibration.py --dataset madrigal",
        "R1.6",
        "the same calibration diagnostics under dataset shift",
    ),
    (
        "src/analysis/madrigal_reference_offset.py",
        "R1.3",
        "reference offset vs model error on Madrigal",
    ),
    (
        "src/analysis/weighting_ablation.py",
        "R1.5",
        "predicted-uncertainty vs elevation weighting, paired",
    ),
    (
        "src/analysis/storm_stratification.py",
        "R1.7",
        "positioning under quiet vs storm conditions",
    ),
    (
        "src/analysis/positioning_robustness.py",
        "R1.7",
        "tail behaviour and horizontal/vertical split",
    ),
    (
        "src/analysis/positioning_summary.py",
        "R1.7, R1.5",
        "Table 5 columns overall, by regime and by weighting",
    ),
    ("src/analysis/oracle_benchmark.py", "R1.8", "observation-derived upper bound"),
]


def run(command: str) -> tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, *command.split()], capture_output=True, text=True
    )
    return result.returncode == 0, result.stderr.strip().splitlines()[
        -1
    ] if result.stderr else ""


def build_index(results_dir: Path, produced: dict[str, str]) -> pd.DataFrame:
    """One row per metric CSV, carrying the comment and script behind it."""
    rows = []
    for path in sorted(results_dir.rglob("*.csv")):
        # Only index the tables produced by the revision analyses, not the raw
        # per-day artefacts the pipeline writes.
        key = path.parts[1] if len(path.parts) > 1 else ""
        if key not in produced and path.name not in produced:
            continue
        comment, description, script = produced.get(key) or produced[path.name]
        frame = pd.read_csv(path, nrows=1)
        rows.append(
            {
                "csv": str(path),
                "reviewer_comment": comment,
                "answers": description,
                "script": script,
                "columns": ", ".join(frame.columns[:10]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=Path("multiday_results"))
    parser.add_argument(
        "--figures", action="store_true", help="also rebuild the figures"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    outcomes = []
    for command, comment, description in ANALYSES:
        ok, error = run(command)
        outcomes.append(
            {
                "script": command,
                "reviewer_comment": comment,
                "answers": description,
                "status": "ok" if ok else "skipped",
                "detail": "" if ok else error,
            }
        )
        logger.info(f"{'✓' if ok else '⚠️ '} {command}" + ("" if ok else f" — {error}"))

    if args.figures:
        ok, error = run("src/viz/revision_figures.py")
        logger.info(f"{'✓' if ok else '⚠️ '} src/viz/revision_figures.py")

    status = pd.DataFrame(outcomes)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    status.to_csv(args.results_dir / "revision_analyses_status.csv", index=False)

    # Map each output directory back to the analysis that owns it.
    produced = {
        "relative_error_metrics.csv": (
            "R2.1, R2.2",
            "error by year",
            "relative_error_metrics.py",
        ),
        "temporal_regime_comparison.csv": (
            "R2.1",
            "interpolation vs extrapolation",
            "relative_error_metrics.py",
        ),
        "hyperparameter_search": (
            "R2.5, R2.8b",
            "architecture and sweep",
            "hyperparameter_search_summary.py",
        ),
        "station_independence": (
            "R2.3",
            "error vs training-network distance",
            "station_independence.py",
        ),
        "computational_cost": (
            "R2.8h",
            "training and inference cost",
            "computational_cost.py",
        ),
        "activity_stratification": (
            "R1.4",
            "error by Dst and F10.7",
            "activity_stratification.py",
        ),
        "uncertainty_calibration": (
            "R1.6",
            "coverage, PIT, CRPS",
            "uncertainty_calibration.py",
        ),
        "madrigal_reference_offset": (
            "R1.3",
            "reference offset vs model error",
            "madrigal_reference_offset.py",
        ),
        "weighting_ablation": ("R1.5", "weighting scheme", "weighting_ablation.py"),
        "storm_stratification": (
            "R1.7",
            "positioning by regime",
            "storm_stratification.py",
        ),
        "positioning_robustness": (
            "R1.7",
            "tail and components",
            "positioning_robustness.py",
        ),
        "positioning_summary": (
            "R1.7, R1.5",
            "Table 5 columns",
            "positioning_summary.py",
        ),
        "oracle_benchmark": (
            "R1.8",
            "observation-derived bound",
            "oracle_benchmark.py",
        ),
    }
    index = build_index(args.results_dir, produced)
    index.to_csv(args.results_dir / "revision_metrics_index.csv", index=False)

    print(status.to_string(index=False))
    print(
        f"\n{len(index)} metric CSVs indexed in {args.results_dir / 'revision_metrics_index.csv'}"
    )
    skipped = status[status["status"] != "ok"]
    if not skipped.empty:
        print(f"{len(skipped)} analysis/analyses skipped - see the status file for why")


if __name__ == "__main__":
    main()
