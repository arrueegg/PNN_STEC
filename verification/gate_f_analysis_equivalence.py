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
    command: str, output_dir: Path, cwd: Path, flag: str = "--output-dir"
) -> tuple[bool, str]:
    """Run one analysis into `output_dir`. Returns (succeeded, tail of output)."""
    parts = command.split()
    result = subprocess.run(
        [sys.executable, *parts, flag, str(output_dir)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    tail = (result.stderr or result.stdout or "").strip().splitlines()[-2:]
    return result.returncode == 0, " / ".join(tail)


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    """Max relative difference per shared numeric column."""
    differences: dict[str, float] = {}
    for column in sorted(set(a.columns) & set(b.columns)):
        left, right = a[column], b[column]
        if not (
            pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right)
        ):
            continue
        if len(left) != len(right):
            differences[column] = float("inf")
            continue
        scale = np.maximum(np.abs(right.to_numpy(dtype=float)), 1.0)
        delta = np.abs(left.to_numpy(dtype=float) - right.to_numpy(dtype=float)) / scale
        differences[column] = float(np.nanmax(delta)) if len(delta) else 0.0
    return differences


def verdict_for(
    comparison: Comparison, differences: dict[str, float]
) -> tuple[str, list[str]]:
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

    ok_new, tail_new = run(comparison.rebuilt, rebuilt_dir, Path.cwd())
    if not ok_new:
        print(
            f"  {comparison.name:<26} FAIL     rebuilt analysis did not run: {tail_new}"
        )
        return "FAIL"
    # A script that takes a file rather than a directory is given one inside its own
    # output tree, so both sides still land somewhere comparable.
    legacy_target = legacy_dir
    if comparison.legacy_writes_file:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_target = (
            legacy_dir / (comparison.legacy_outputs or comparison.outputs)[0]
        )
    ok_old, tail_old = run(
        comparison.legacy, legacy_target, LEGACY_SRC, comparison.legacy_output_flag
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
