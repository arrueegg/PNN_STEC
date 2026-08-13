"""Observation-derived upper bound for the positioning experiment.

Evidence for reviewer comment R2.8, which asks for a benchmark in which the
GNSS-derived reference STEC is applied directly as the ionospheric correction,
to show how close the model gets to the best achievable result under the same
STEC processing pipeline.

The oracle runs are produced by::

    python positioning/scripts/generate_reference_corrections.py --year 2024 --doy <DOY>
    python positioning/positioning_eval/run_positioning_evaluation.py \\
        --experiment Reference_STEC_Oracle --date <YYYY-MM-DD> \\
        --all_test_stations --weight_opt elev --no_cleanup

They use elevation weighting, because the reference carries no per-observation
uncertainty, so the comparison is against the other methods' **elevation-weighted**
arm. Mixing in the uncertainty-weighted arm would confound the correction with
the weighting scheme.

The comparison is restricted to station-days solved by every method, since the
oracle covers only the stations present in the processed database on that day
and an unpaired mean would compare different station sets.

What the bound does and does not say: the reference STEC comes from the same
dual-frequency observations, DCB handling and levelling as the training target,
so it bounds what a model of this target can achieve inside this pipeline. It is
not independent truth, and the residual error it leaves is the pipeline's own
noise floor rather than a statement about the ionosphere.

Usage::

    python src/analysis/oracle_benchmark.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTLIER_3D_RMS_M = 10.0
ORACLE_LABEL = "Reference STEC (oracle)"

# The oracle experiment writes the correction run as "model" and its own IGS GIM
# control as "gim"; the latter is what verifies the two pipelines agree.
ORACLE_METHODS = {"model": ORACLE_LABEL, "gim": "IGS GIM + Mapping (oracle run)"}
BASELINE_METHODS = {
    "STEC_elev": "Direct STEC",
    "VTEC_elev": "VTEC + Mapping",
    "gim_elev": "IGS GIM + Mapping",
}
DISPLAY_ORDER = [
    ORACLE_LABEL,
    "Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]


def load_oracle(results_root: Path) -> pd.DataFrame:
    """Collect every daily_summary.csv the oracle experiment has produced."""
    frames = []
    for path in sorted(results_root.glob("*/daily_summary*.csv")):
        frame = pd.read_csv(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No oracle daily_summary*.csv under {results_root}")

    oracle = pd.concat(frames, ignore_index=True)
    oracle["Method"] = oracle["method"].map(ORACLE_METHODS)
    return oracle.dropna(subset=["Method"])


def load_baselines(summary_path: Path) -> pd.DataFrame:
    baselines = pd.read_csv(summary_path)
    baselines["Method"] = baselines["method"].map(BASELINE_METHODS)
    return baselines.dropna(subset=["Method"])


def paired_comparison(oracle: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Restrict to station-days present for every method, then summarise."""
    combined = pd.concat(
        [
            oracle[
                ["station", "doy", "Method", "error_3d_rms", "error_2d_rms", "u_rms"]
            ],
            baselines[
                ["station", "doy", "Method", "error_3d_rms", "error_2d_rms", "u_rms"]
            ],
        ],
        ignore_index=True,
    )
    combined = combined[combined["error_3d_rms"] <= OUTLIER_3D_RMS_M]

    wanted = [m for m in DISPLAY_ORDER if m in set(combined["Method"])]
    pivot = combined[combined["Method"].isin(wanted)].pivot_table(
        index=["station", "doy"], columns="Method", values="error_3d_rms"
    )
    complete = pivot.dropna()
    logger.info(
        f"{len(complete)} station-days solved by all {len(wanted)} methods "
        f"(of {len(pivot)} seen for at least one)"
    )
    if complete.empty:
        # An empty intersection is a coverage problem, not a result. Say which
        # method is responsible instead of returning a table of NaNs.
        coverage = pivot.notna().sum().sort_values()
        logger.warning(
            "⚠️  No station-day is covered by every method. Per-method coverage:\n"
            + coverage.to_string()
            + "\n   The scarcest method above is what limits the comparison."
        )
    return complete[wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle_results",
        type=Path,
        default=Path("experiments/Reference_STEC_Oracle/positioning/results"),
    )
    parser.add_argument(
        "--baseline_summary",
        type=Path,
        default=Path("multiday_results/positioning_20260216_2052/multiday_summary.csv"),
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/oracle_benchmark")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    oracle = load_oracle(args.oracle_results)
    baselines = load_baselines(args.baseline_summary)

    # Sanity check: the oracle experiment reran IGS GIM itself, so its numbers
    # must match the published elevation-weighted GIM arm on the same
    # station-days. A mismatch means the two runs are not comparable.
    control = oracle[oracle["Method"] == "IGS GIM + Mapping (oracle run)"]
    published = baselines[baselines["Method"] == "IGS GIM + Mapping"]
    check = control.merge(
        published, on=["station", "doy"], suffixes=("_rerun", "_published")
    )
    if not check.empty:
        difference = (
            check["error_3d_rms_rerun"] - check["error_3d_rms_published"]
        ).abs()
        logger.info(
            f"IGS GIM control: {len(check)} shared station-days, "
            f"max |Δ 3D RMS| = {difference.max():.4f} m, median {difference.median():.4f} m"
        )
        if difference.max() > 0.01:
            logger.warning(
                "⚠️  Control run disagrees with the published GIM arm by >1 cm"
            )

    paired = paired_comparison(oracle, baselines)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_station_days.csv")

    summary = pd.DataFrame(
        {
            "mean": paired.mean(),
            "median": paired.median(),
            "p95": paired.quantile(0.95),
            "station_days": paired.notna().sum(),
        }
    )
    if ORACLE_LABEL in summary.index:
        floor = summary.loc[ORACLE_LABEL, "mean"]
        # How much of each method's error is above the pipeline's own floor.
        summary["above_oracle_m"] = summary["mean"] - floor
        summary["ratio_to_oracle"] = summary["mean"] / floor
    summary.to_csv(args.output_dir / "summary.csv")

    print("=== Positioning against the observation-derived upper bound ===")
    print("(elevation weighting throughout; paired station-days)\n")
    print(summary.round(3).to_string())
    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
