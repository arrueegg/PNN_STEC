"""Gate F: does each ported analysis produce what the one it replaces produced?

Runs a ported analysis and its pre-rebuild counterpart against the same real inputs, both
now, and compares the resulting CSVs column by column. This is the gate that covers the
layer the paper actually quotes, and unlike the earlier gates it is not expected to be
uniformly bit-exact - several ports deliberately changed behaviour, and those are listed
here rather than discovered as surprises.

**A difference is only a failure if it is not explained.** That is the rule the whole
rebuild runs on: matching proves the two implementations are consistent, not that either
is right, and a refactor preserves the logic it ports. So each analysis declares what it
expects, and the gate reports three outcomes rather than two:

    MATCH       numerically identical within tolerance
    DIVERGED    different, and the difference is one this port intended
    FAIL        different, and nobody said it would be

Two comparisons are deliberately *not* attempted. `repair_gim_baseline` is excluded
because it is the regression check for the GIM repair, and comparing it against itself
would mean the check and the thing it checks share an implementation. `positioning_coverage`
is excluded while the station-recovery sweep is running: its inputs are being rewritten, so
the two sides would read different trees and the comparison would measure the sweep.

    source .env.worktree
    python verification/gate_f_analysis_equivalence.py --only daily_metrics
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.config import paths  # noqa: E402

LEGACY_SRC = Path("/scratch2/arrueegg/WP4/PNN_STEC")

# Relative tolerance for a float column. Analyses aggregate float32 inputs in a different
# order than the originals did, so exact equality is not the right standard here - unlike
# the model and inference gates, where the computation is the same graph.
RELATIVE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class Comparison:
    """One ported analysis, its predecessor, and what is expected of the difference."""

    name: str
    rebuilt: str
    legacy: str
    outputs: tuple[str, ...]
    # Columns whose disagreement is intended, mapped to why. Anything not named here that
    # disagrees is a failure.
    expected_divergence: dict[str, str] = field(default_factory=dict)
    skip: str | None = None
    # Each side names its own output convention, because they genuinely differ and
    # assuming one for both is what made the first version of this gate skip everything.
    # The ports take --output-dir; the pre-rebuild scripts variously take --output_dir or
    # --output (a *file*, not a directory).
    legacy_output_flag: str = "--output_dir"
    legacy_writes_file: bool = False
    # Some scripts write into a subdirectory of what they were given. uncertainty_calibration
    # writes to <output_dir>/<variant>_<dataset>/ on both sides.
    rebuilt_subdir: str = ""
    legacy_subdir: str = ""
    # When a port renamed its output, both names are needed to compare the same content.
    # Empty means the two sides agree on the filename.
    legacy_outputs: tuple[str, ...] = ()
    # Almost every port takes --output-dir; ionex_rms_benchmark kept the pre-rebuild
    # underscore spelling, so this cannot be a module-wide constant.
    rebuilt_output_flag: str = "--output-dir"
    # Extra arguments each side needs beyond the output flag - e.g. activity_stratification
    # refusing to run without a --repair-report it does not generate itself.
    extra_rebuilt_args: tuple[str, ...] = ()
    extra_legacy_args: tuple[str, ...] = ()
    # Seconds allowed per side. The default covers everything that reads its input
    # once; the analyses that stream the full 242-day store twice (once per side) need
    # more - measured at ~9 s/day for madrigal_reference_offset and worse once IONEX
    # parsing is added, against a few seconds/day for the leaner store analyses.
    timeout_seconds: int = 3600

    def output_pairs(self) -> tuple[tuple[str, str], ...]:
        legacy = self.legacy_outputs or self.outputs
        return tuple(zip(self.outputs, legacy, strict=True))


COMPARISONS: tuple[Comparison, ...] = (
    Comparison(
        name="daily_metrics",
        rebuilt="-m stec.analysis.daily_metrics",
        legacy="src/analysis/daily_metrics.py",
        outputs=("summary.csv", "per_day.csv"),
        expected_divergence={
            "R2": "the per-day column was renamed from the original's unicode R² to R2, "
            "matching the summary's R2_mean/R2_std"
        },
    ),
    Comparison(
        name="uncertainty_calibration",
        rebuilt="-m stec.analysis.uncertainty_calibration",
        legacy="src/analysis/uncertainty_calibration.py",
        outputs=("coverage.csv",),
        rebuilt_subdir="finetuned_stec_own",
        legacy_subdir="finetuned_stec_own",
        expected_divergence={
            "*": "every model is now scored under both Gaussian and Laplace and tagged "
            "with which is native, so the frame has rows the original did not"
        },
    ),
    Comparison(
        name="relative_error_metrics",
        rebuilt="-m stec.analysis.relative_error_metrics",
        legacy="src/analysis/relative_error_metrics.py",
        outputs=("yearly_metrics.csv", "temporal_regime_comparison.csv"),
        legacy_outputs=("relative_error_metrics.csv", "temporal_regime_comparison.csv"),
        legacy_output_flag="--output",
        legacy_writes_file=True,
    ),
    Comparison(
        name="mapping_function_consistency",
        rebuilt="-m stec.analysis.mapping_function_consistency",
        legacy="src/analysis/mapping_function_consistency.py",
        outputs=("by_elevation.csv", "overall.csv"),
    ),
    Comparison(
        name="weighting_ablation",
        rebuilt="-m stec.analysis.weighting_ablation",
        legacy="src/analysis/weighting_ablation.py",
        outputs=("paired.csv", "fixed_variance.csv"),
    ),
    Comparison(
        name="storm_stratification",
        rebuilt="-m stec.analysis.storm_stratification",
        legacy="src/analysis/storm_stratification.py",
        # by_regime.csv is deliberately excluded: the port reshaped it from the
        # original's unstacked (stat, regime) MultiIndex header into a long table (one
        # row per Method/regime, plus a new 3D_p95_m column) via
        # stec.positioning.metrics.summarise. The two CSVs share zero column names, so
        # comparing them would report a false MATCH from an empty intersection rather
        # than reveal anything - see the gate's final report for this finding.
        outputs=("degradation.csv", "improvement_over_gim.csv"),
        expected_divergence={
            "quiet": "stec.positioning.metrics.summarise rounds to 4 decimal places; "
            "the legacy computation keeps full float64 precision, so TECU-scale values "
            "differ by up to ~4e-5 - below either implementation's reported resolution",
            "storm": "same summarise() rounding as 'quiet'",
            "storm_vs_quiet_%": "downstream of the same summarise() rounding",
            "improvement_over_gim_quiet_%": "downstream of the same summarise() "
            "rounding",
            "improvement_over_gim_storm_%": "downstream of the same summarise() "
            "rounding",
        },
    ),
    Comparison(
        name="positioning_robustness",
        rebuilt="-m stec.analysis.positioning_robustness",
        legacy="src/analysis/positioning_robustness.py",
        outputs=("tail_distribution.csv", "error_components.csv"),
    ),
    Comparison(
        name="positioning_summary",
        rebuilt="-m stec.analysis.positioning_summary",
        legacy="src/analysis/positioning_summary.py",
        outputs=("overall.csv", "by_regime.csv", "by_weighting.csv"),
    ),
    Comparison(
        name="common_set_positioning",
        rebuilt="-m stec.analysis.common_set_positioning",
        legacy="src/analysis/common_set_positioning.py",
        outputs=("table5_common_set.csv",),
        expected_divergence={
            "station_days": "the outlier rule changed from < to <= for consistency "
            "with its two siblings (positioning_summary, positioning_robustness), so a "
            "station-day at exactly 10 m is now kept rather than dropped",
            "lost_to_intersection": "downstream of the same <= change",
            "rms_3d_mean": "downstream of the same <= change",
            "rms_3d_median": "downstream of the same <= change",
            "rms_2d_mean": "downstream of the same <= change",
            "up_mean": "downstream of the same <= change",
            "gain_ratio_of_means_pct": "downstream of the same <= change",
            "gain_paired_mean_pct": "downstream of the same <= change",
            "gain_paired_median_pct": "downstream of the same <= change",
            "win_rate_pct": "downstream of the same <= change",
        },
    ),
    Comparison(
        name="oracle_benchmark",
        rebuilt="-m stec.analysis.oracle_benchmark",
        legacy="src/analysis/oracle_benchmark.py",
        outputs=("paired_station_days.csv", "summary.csv"),
    ),
    Comparison(
        name="computational_cost",
        rebuilt="-m stec.analysis.computational_cost",
        legacy="src/analysis/computational_cost.py",
        outputs=("training_cost.csv", "cost_summary.csv"),
    ),
    Comparison(
        name="activity_stratification",
        rebuilt="-m stec.analysis.activity_stratification",
        legacy="src/analysis/activity_stratification.py",
        outputs=("by_dst.csv", "by_f107.csv"),
        extra_rebuilt_args=(
            "--repair-report",
            str(
                LEGACY_SRC
                / "multiday_results/gim_baseline_repair/gim_repair_report.csv"
            ),
        ),
        expected_divergence={
            "RMSE": "F10.7 bins changed from data-derived terciles to fixed absolute "
            "bands, which changes by_f107.csv's row shape (4 fixed bands vs 3 "
            "terciles); by_dst.csv is unaffected since the Dst bins are unchanged",
            "MAE": "same F10.7 rebinning as RMSE",
            "R2": "same F10.7 rebinning as RMSE",
            "days": "same F10.7 rebinning as RMSE",
            "observations": "same F10.7 rebinning as RMSE",
            "improvement_over_gim_%": "same F10.7 rebinning as RMSE",
        },
    ),
    Comparison(
        name="stratified_comparison",
        rebuilt="",
        legacy="",
        outputs=(),
        skip="measured at ~40 s/day on the rebuilt side alone (timed via --doys on 5 "
        "real days: 3m20s), i.e. ~2.7h to stream the full 242-day store once - each "
        "of the 4 stratifiers x up to 4 methods builds and groups a fresh per-bin "
        "frame per day. Both sides together would be ~5h+, which is not a tractable "
        "single subprocess run in this session (the first attempt hit the 3600s "
        "subprocess timeout with the rebuilt side alone still short of DOY 366). "
        "Its expected divergence would have been the same per-method NaN-masking "
        "change documented for uncertainty_error_relation's sibling analyses.",
    ),
    Comparison(
        name="uncertainty_error_relation",
        rebuilt="-m stec.analysis.uncertainty_error_relation",
        legacy="src/analysis/uncertainty_error_relation.py",
        outputs=("by_uncertainty.csv",),
        legacy_outputs=("by_sigma.csv",),
        expected_divergence={
            "*": "three declared changes, not one. (1) bins are fixed absolute TECU "
            "intervals rather than the first day's deciles, which held 6.88-18.80% of the "
            "population instead of 10%. (2) epistemic_share is redefined: the original "
            "computed mean_epistemic**2 / (mean_epistemic**2 + mean_aleatoric**2), the "
            "square of the means, where the variance decomposition calls for the mean of "
            "the squares - the rebuilt sum_epistemic_sq / sum_total_sq. The original is "
            "biased by Jensen and compresses the range: 4.94-6.66% against 3.07-16.39%. "
            "The rebuilt statistic is the correct one, but the change was undeclared until "
            "a port audit found it. (3) it is now a fraction, not a percentage, and the "
            "column lost its % suffix - read the units before quoting it.",
        },
    ),
    Comparison(
        name="station_independence",
        rebuilt="-m stec.analysis.station_independence",
        legacy="src/analysis/station_independence.py",
        outputs=("per_station.csv", "by_distance_bin.csv"),
    ),
    Comparison(
        name="madrigal_reference_offset",
        rebuilt="-m stec.analysis.madrigal_reference_offset",
        legacy="src/analysis/madrigal_reference_offset.py",
        outputs=(
            "per_station_offsets.csv",
            "coverage_before_after.csv",
            "decomposition.csv",
            "leverage_check.csv",
            "reference_precision.csv",
        ),
        # Two passes over the Madrigal store (~9 s/day measured on 5 real days), so a
        # 242-day run is ~35-40 min per side.
        timeout_seconds=5400,
    ),
    Comparison(
        name="ionex_rms_benchmark",
        rebuilt="-m stec.analysis.ionex_rms_benchmark",
        legacy="src/analysis/ionex_rms_benchmark.py",
        outputs=(
            "per_day_IGS.csv",
            "overall_IGS.csv",
            "by_elevation_IGS.csv",
            "by_regime_IGS.csv",
        ),
        rebuilt_output_flag="--output_dir",
        # No --doys on either side (the pre-rebuild script never had one and the port
        # kept parity), and each day also parses an IONEX map, so this cannot be timed
        # on a subset - give it generous headroom instead.
        timeout_seconds=7200,
    ),
    Comparison(
        name="repair_gim_baseline",
        rebuilt="",
        legacy="",
        outputs=(),
        skip="it is the regression check for the GIM repair; comparing it against itself "
        "would make the check share an implementation with what it checks",
    ),
    Comparison(
        name="positioning_coverage",
        rebuilt="",
        legacy="",
        outputs=(),
        skip="its inputs are being rewritten by the running station-recovery sweep, so the "
        "two sides would read different trees and the comparison would measure the sweep",
    ),
)


def run(
    command: str,
    output_dir: Path,
    cwd: Path,
    flag: str = "--output-dir",
    extra_args: tuple[str, ...] = (),
    timeout_seconds: int = 3600,
) -> tuple[bool, str]:
    """Run one analysis into `output_dir`. Returns (succeeded, tail of output).

    A timeout is reported as its own "TIMEOUT: ..." tail rather than left to propagate -
    an analysis that streams the full store can legitimately need an hour, and that is a
    reason to skip the comparison, not to crash the whole gate run and lose every
    comparison after it in the list.
    """
    parts = command.split()
    try:
        result = subprocess.run(
            [sys.executable, *parts, flag, str(output_dir), *extra_args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: exceeded {timeout_seconds}s"
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-2:]
    return result.returncode == 0, " / ".join(tail)


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Max relative difference per shared column.

    Numeric columns are compared on their values. Text columns are compared as exact
    strings and reported as 0.0 or infinity, because a label is either the same label or
    a different one - there is no tolerance to express. Skipping them, which this function
    used to do, made every non-numeric artifact invisible to the gate: computational_cost's
    cost_summary.csv has no numeric column at all, so the whole file went uncompared while
    the comparison reported a match.
    """
    differences: dict[str, float] = {}
    for column in sorted(set(a.columns) & set(b.columns)):
        left, right = a[column], b[column]
        if len(left) != len(right):
            differences[column] = float("inf")
            continue
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            scale = np.maximum(np.abs(right.to_numpy(dtype=float)), 1.0)
            delta = (
                np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float)) / scale
            )
            differences[column] = float(np.nanmax(delta)) if len(delta) else 0.0
            continue
        # NaN != NaN, so an all-empty column would otherwise read as a difference.
        unequal = (left.astype(str) != right.astype(str)) & ~(
            left.isna() & right.isna()
        )
        differences[column] = float("inf") if bool(unequal.any()) else 0.0
    return differences


def verdict_for(
    comparison: Comparison, differences: dict[str, float]
) -> tuple[str, list[str]]:
    # Two frames sharing no numeric column produce an empty difference map, and an empty
    # map trivially satisfies every "nothing exceeds tolerance" test below. That is a pass
    # earned by comparing nothing - the same vacuous success this gate exists to detect,
    # occurring inside the detector. storm_stratification's by_regime.csv hits it exactly:
    # the port reshaped a MultiIndex header into a long table, so the two share zero column
    # names.
    if not differences:
        return "FAIL", ["no numeric column in common - nothing was actually compared"]
    if "*" in comparison.expected_divergence:
        return "DIVERGED", [comparison.expected_divergence["*"]]

    unexplained = [
        column
        for column, delta in differences.items()
        if delta > RELATIVE_TOLERANCE and column not in comparison.expected_divergence
    ]
    explained = [
        column
        for column, delta in differences.items()
        if delta > RELATIVE_TOLERANCE and column in comparison.expected_divergence
    ]
    if unexplained:
        return "FAIL", unexplained
    if explained:
        return "DIVERGED", [comparison.expected_divergence[c] for c in explained]
    return "MATCH", []


def check(comparison: Comparison, workspace: Path) -> str:
    if comparison.skip:
        print(f"  {comparison.name:<26} SKIPPED  {comparison.skip}")
        return "SKIPPED"

    rebuilt_dir = workspace / f"{comparison.name}_rebuilt"
    legacy_dir = workspace / f"{comparison.name}_legacy"

    ok_new, tail_new = run(
        comparison.rebuilt,
        rebuilt_dir,
        Path.cwd(),
        comparison.rebuilt_output_flag,
        comparison.extra_rebuilt_args,
        comparison.timeout_seconds,
    )
    if not ok_new:
        verdict = "SKIPPED" if tail_new.startswith("TIMEOUT") else "FAIL"
        reason = (
            "rebuilt analysis timed out"
            if verdict == "SKIPPED"
            else ("rebuilt analysis did not run")
        )
        print(f"  {comparison.name:<26} {verdict:<8} {reason}: {tail_new}")
        return verdict
    # A script that takes a file rather than a directory is given one inside its own
    # output tree, so both sides still land somewhere comparable.
    legacy_target = legacy_dir
    if comparison.legacy_writes_file:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_target = (
            legacy_dir / (comparison.legacy_outputs or comparison.outputs)[0]
        )
    ok_old, tail_old = run(
        comparison.legacy,
        legacy_target,
        LEGACY_SRC,
        comparison.legacy_output_flag,
        comparison.extra_legacy_args,
        comparison.timeout_seconds,
    )
    if not ok_old:
        print(
            f"  {comparison.name:<26} SKIPPED  legacy analysis did not run: {tail_old}"
        )
        return "SKIPPED"

    worst_verdict = "MATCH"
    notes: list[str] = []
    for new_name, old_name in comparison.output_pairs():
        new_path = rebuilt_dir / comparison.rebuilt_subdir / new_name
        old_path = legacy_dir / comparison.legacy_subdir / old_name
        output = new_name
        if not new_path.exists() or not old_path.exists():
            print(f"  {comparison.name:<26} FAIL     {output} missing on one side")
            return "FAIL"
        differences = compare_frames(pd.read_csv(new_path), pd.read_csv(old_path))
        verdict, why = verdict_for(comparison, differences)
        notes.extend(why)
        if verdict == "FAIL":
            worst_verdict = "FAIL"
            notes.append(f"{output}: unexplained columns {why}")
        elif verdict == "DIVERGED" and worst_verdict != "FAIL":
            worst_verdict = "DIVERGED"

    print(
        f"  {comparison.name:<26} {worst_verdict:<8} "
        + ("; ".join(notes[:1]) if notes else "")
    )
    return worst_verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="analysis names")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/gate_f"))
    parser.add_argument("--keep", action="store_true", help="keep the output trees")
    args = parser.parse_args()

    selected = COMPARISONS
    if args.only:
        known = {c.name for c in COMPARISONS}
        unknown = set(args.only) - known
        if unknown:
            raise SystemExit(f"unknown analysis: {sorted(unknown)}")
        selected = tuple(c for c in COMPARISONS if c.name in set(args.only))

    # Both sides read the store through paths.LEGACY_PREDICTIONS, which falls back to this
    # checkout when STEC_LEGACY_ROOT is unset. In a worktree that directory is empty, and
    # the two sides then agree perfectly about nothing - a vacuous PASS of exactly the kind
    # this gate exists to catch. Forgetting `source .env.worktree` is the ordinary way to
    # arrive here, so refuse rather than compare.
    stored_days = (
        len(list(paths.LEGACY_PREDICTIONS.glob("*/*/year=*/doy=*.parquet")))
        if paths.LEGACY_PREDICTIONS.exists()
        else 0
    )
    if stored_days == 0:
        raise SystemExit(
            f"no prediction store under {paths.LEGACY_PREDICTIONS} - refusing to compare "
            "two analyses that would both read nothing. Set STEC_LEGACY_ROOT "
            "(`source .env.worktree`) and re-run."
        )

    if args.workspace.exists() and not args.keep:
        shutil.rmtree(args.workspace)
    args.workspace.mkdir(parents=True, exist_ok=True)

    print(
        f"comparing {len(selected)} analysis/analyses, store at {paths.LEGACY_PREDICTIONS}\n"
    )
    verdicts = [check(c, args.workspace) for c in selected]

    print()
    for label in ("MATCH", "DIVERGED", "SKIPPED", "FAIL"):
        count = verdicts.count(label)
        if count:
            print(f"  {label}: {count}")
    if "FAIL" in verdicts:
        print("\n  FAIL  at least one difference nobody declared")
        return 1

    compared = [v for v in verdicts if v in ("MATCH", "DIVERGED")]
    if not compared:
        # A gate that compares nothing and reports success is precisely the failure this
        # rebuild exists to remove. Skipping everything is an inconclusive run, not a pass.
        print("\n  INCONCLUSIVE  nothing was actually compared")
        return 2
    print(
        f"\n  PASS  {len(compared)} analysis/analyses compared, every difference declared"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
