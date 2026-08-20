"""Observation-derived upper bound for the positioning experiment (R2.8).

Ported from ``src/analysis/oracle_benchmark.py`` in the live PNN_STEC checkout, reusing
the already-ported ``.pos`` parser and metrics in ``stec.positioning.metrics`` instead of
importing the PPPx-adjacent script directly.

Evidence for reviewer comment R2.8, which asks for a benchmark in which the GNSS-derived
reference STEC is applied directly as the ionospheric correction, to show how close the
model gets to the best achievable result under the same STEC processing pipeline.

The oracle runs are produced by::

    python positioning/scripts/generate_reference_corrections.py --year 2024 --doy <DOY>
    python positioning/positioning_eval/run_positioning_evaluation.py \\
        --experiment Reference_STEC_Oracle --date <YYYY-MM-DD> \\
        --all_test_stations --weight_opt elev --no_cleanup

**This is NOT comparable with Table 5, by design and permanently.** It uses elevation
weighting, because the reference STEC carries no per-observation uncertainty - weighting
by ``iono`` would weight every observation by the same placeholder sigma, i.e. by nothing
- so the comparison is against the other methods' **elevation-weighted** arm, never the
uncertainty-weighted one that Table 5 reports. It is also restricted to station-days
solved by every method, since the oracle covers only the stations present in the
processed database on that day and an unpaired mean would compare different station sets.
Read ratios to the floor *within* this table; take absolute positioning numbers from
Table 5.

What the bound does and does not say: the reference STEC comes from the same
dual-frequency observations, DCB handling and levelling as the training target, so it
bounds what a model of this target can achieve inside this pipeline. It is not independent
truth, and the residual error it leaves is the pipeline's own noise floor rather than a
statement about the ionosphere.

Usage::

    python -m stec.analysis.oracle_benchmark
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from ..positioning import metrics as pm
from .positioning_summary import DEFAULT_WEIGHTING_SUMMARY

logger = logging.getLogger(__name__)

ORACLE_LABEL = "Reference STEC (oracle)"

# The oracle experiment writes the correction run as "model" and its own IGS GIM control
# as "gim"; the latter is what verifies the two pipelines agree.
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

DEFAULT_ORACLE_RESULTS = (
    paths.LEGACY_EXPERIMENTS / "Reference_STEC_Oracle" / "positioning" / "results"
)
DEFAULT_OUTPUT_DIR = Path("multiday_results/oracle_benchmark_rebuilt")


def load_oracle(results_root: Path) -> pd.DataFrame:
    """Aggregate the oracle runs straight from their ``.pos`` solutions.

    Deliberately not read from ``daily_summary.csv``: that file is rewritten by whichever
    evaluation ran last, so a single-station rerun silently truncates it to one row while
    the ``.pos`` files stay complete. The solutions are the durable artefact, so they are
    the input.
    """
    rows = []
    for day_dir in sorted(results_root.glob("[0-9]" * 7)):
        year, doy = int(day_dir.name[:4]), int(day_dir.name[4:])
        sinex = sorted(
            (day_dir.parent.parent / "evaluation" / day_dir.name / "products").glob(
                "*CRD.SNX"
            )
        )
        if not sinex:
            logger.warning(f"no SINEX for {day_dir.name} - skipping")
            continue
        truth = pm.load_sinex_coords(sinex[0])

        for method_dir, label in ORACLE_METHODS.items():
            for pos_path in sorted((day_dir / method_dir).glob("*/*.pos")):
                station = pos_path.parent.name
                if station not in truth:
                    continue
                solution = pm.parse_pos_file(pos_path, ref_pos=truth[station])
                metrics = pm.compute_metrics(solution)
                if metrics is None:
                    continue
                rows.append(
                    {
                        "station": station,
                        "year": year,
                        "doy": doy,
                        "Method": label,
                        **metrics,
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No oracle .pos solutions under {results_root}")
    logger.info(f"aggregated {len(rows)} oracle solutions from .pos files")
    return pd.DataFrame(rows)


def load_baselines(summary_path: Path) -> pd.DataFrame:
    baselines = pd.read_csv(summary_path)
    baselines["Method"] = baselines["method"].map(BASELINE_METHODS)
    return baselines.dropna(subset=["Method"])


def paired_comparison(oracle: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Restrict to station-days present for every method, then pivot to one column per
    method - the shape ``main`` summarises into mean/median/p95."""
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
    combined = pm.exclude_outlier_station_days(combined)

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
        # An empty intersection is a coverage problem, not a result. Say which method is
        # responsible instead of returning a table of NaNs.
        coverage = pivot.notna().sum().sort_values()
        logger.warning(
            "No station-day is covered by every method. Per-method coverage:\n"
            + coverage.to_string()
            + "\n   The scarcest method above is what limits the comparison."
        )
    return complete[wanted]


def check_gim_control(oracle: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """The oracle experiment reran IGS GIM itself, so its numbers must match the published
    elevation-weighted GIM arm on the same station-days. Returns the merged comparison so
    callers can assert on it; a mismatch means the two runs are not comparable."""
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
            f"max |delta 3D RMS| = {difference.max():.4f} m, "
            f"median {difference.median():.4f} m"
        )
        if difference.max() > 0.01:
            logger.warning("control run disagrees with the published GIM arm by >1 cm")
    return check


def summarise(paired: pd.DataFrame) -> pd.DataFrame:
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
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-results", type=Path, default=DEFAULT_ORACLE_RESULTS)
    parser.add_argument(
        "--baseline-summary", type=Path, default=DEFAULT_WEIGHTING_SUMMARY
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    oracle = load_oracle(args.oracle_results)
    baselines = load_baselines(args.baseline_summary)
    check_gim_control(oracle, baselines)

    paired = paired_comparison(oracle, baselines)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_station_days.csv")

    summary = summarise(paired)
    summary.to_csv(args.output_dir / "summary.csv")

    print("=== Positioning against the observation-derived upper bound ===")
    print("(elevation weighting throughout; paired station-days)\n")
    print(summary.round(3).to_string())
    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
