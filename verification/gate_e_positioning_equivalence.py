"""Gate E, positioning equivalence: does the rebuilt metrics code reproduce old numbers?

**This is half of Gate E, and it says so plainly.** The rebuild plan's Gate E asks for "one
day's `.stec` and `.pos` reproduce" - i.e. that PPPx, re-run today, gives the position it
gave before. That half is out of scope here and is not attempted:

* it needs the SuiteSparse 5 shim (`positioning_eval/lib_compat/fetch_libs.sh`) that this
  worktree has no reason to set up,
* most products cannot be downloaded from this host (CODE is FTP-only and firewalled,
  CDDIS needs Earthdata credentials that are not configured here),
* a solved day costs ~766 MB and the primary checkout is already running positioning jobs
  at 81% disk - re-running PPPx there would risk the jobs in progress, and this worktree is
  read-only against `/scratch2/arrueegg/WP4/PNN_STEC` by construction.

So this gate covers the other half: given `.pos` files PPPx has *already* produced, does
`stec/positioning/metrics.py` (a straight port of `positioning/positioning_eval/metrics.py`,
see that module's docstring) compute the same per-station-day numbers the pre-rebuild code
recorded for those same files? The comparison is genuine, not circular - `daily_summary.csv`
is the *old* code's answer, on disk, for exactly the `.pos` files read here; only the
computation is re-run, not the solve.

**What this gate does not cover, and never will by running it harder**: whether PPPx itself
is reproducible, whether the RTKLIB-format corrections fed to PPPx were generated
correctly, and whether the SINEX/orbit/clock products used at solve time match what a fresh
download would give. A green run here says the arithmetic from `.pos` to metrics survived
the rebuild - nothing about the solver.

Sampling: the paper's canonical daily fine-tune experiments
(`Finetune_STEC_2024_<DOY>_BayesianResNetSTEC_h1024_l4_..._SWI`, see CLAUDE.md) each carry
`daily_summary.csv` (elev weighting) and `daily_summary_iono.csv` (iono weighting), one row
per station, produced by the pre-rebuild `aggregate_daily_metrics` + `save_daily_summary`.
This gate re-parses a sample of the underlying `.pos` files with the *rebuilt* module,
recomputes `error_3d_rms` / `error_2d_rms` / `u_rms` against the same SINEX ground truth, and
diffs against the recorded row.

`daily_summary.csv` is written with `float_format='%.4f'`
(`positioning/positioning_eval/metrics.py::save_daily_summary`), so a rounding-only
difference is bounded by half that resolution (5e-5 m); the tolerance below is set well
above that floor without being loose enough to pass a real defect, which - wrong station,
wrong reference position, an unmasked epoch range - shows up orders of magnitude larger.

    source /scratch2/arrueegg/WP4/PNN_STEC/env/bin/activate
    source .env.worktree
    python verification/gate_e_positioning_equivalence.py --days 8 --stations-per-day 3
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.config import paths  # noqa: E402
from stec.positioning import metrics as pm  # noqa: E402

# The hyperparameter suffix `compute_exp_name` gives the paper's daily fine-tunes (CLAUDE.md,
# "The paper model"). Fixed here rather than globbed, because `experiments/` also holds
# superseded hyperparameter-sweep runs for the same DOY (different lr/batch size) that would
# otherwise match a loose pattern.
CANONICAL_STEC_SUFFIX = (
    "BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_"
    "GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
TEST_DOY_RANGE = range(122, 367)  # 2024 test period, per CLAUDE.md's data table

# weight_opt -> (summary CSV name, positioning/results/<day> subdirectory holding the .pos
# files, which is also the "method" value in that CSV - see run_positioning_evaluation.py).
ARMS: dict[str, tuple[str, str]] = {
    "elev": ("daily_summary.csv", "model"),
    "iono": ("daily_summary_iono.csv", "model_iono"),
}

COMPARED_METRICS = ("error_3d_rms", "error_2d_rms", "u_rms")

CSV_ROUNDING_M = 1e-4  # save_daily_summary's float_format='%.4f'
TOLERANCE_M = 5e-4  # 5x the CSV's half-ULP rounding noise (5e-5 m)


def canonical_experiment_dir(doy: int) -> Path:
    return (
        paths.LEGACY_EXPERIMENTS / f"Finetune_STEC_2024_{doy}_{CANONICAL_STEC_SUFFIX}"
    )


def results_dir(experiment_dir: Path, doy: int) -> Path:
    return experiment_dir / "positioning" / "results" / f"2024{doy:03d}"


def sinex_file(experiment_dir: Path, doy: int) -> Path:
    return (
        experiment_dir
        / "positioning"
        / "evaluation"
        / f"2024{doy:03d}"
        / "products"
        / f"IGS0OPSSNX_2024{doy:03d}0000_01D_01D_CRD.SNX"
    )


def pos_file_path(
    experiment_dir: Path, doy: int, method_subdir: str, station: str
) -> Path:
    return (
        results_dir(experiment_dir, doy)
        / method_subdir
        / station
        / f"{station}_{method_subdir}.pos"
    )


def sample_evenly(items: Sequence, n: int) -> list:
    """`n` items spread across `items`, keeping the first and last. Deterministic - no
    seed to pin, unlike a random sample - which matters for a gate whose output should be
    the same across runs against an unchanged checkout."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [items[i] for i in indices]


def discover_sample_days() -> list[int]:
    """DOYs with a canonical experiment directory, a daily_summary.csv, and a SINEX file.

    Empty when the live checkout isn't present, or when nothing solvable was found (e.g.
    DOY 303/338/348, which CLAUDE.md notes have no products anywhere) - both are reported
    by the caller as a clean skip, not a failure.
    """
    if not paths.LEGACY_EXPERIMENTS.exists():
        return []
    days = []
    for doy in TEST_DOY_RANGE:
        experiment_dir = canonical_experiment_dir(doy)
        if not (results_dir(experiment_dir, doy) / "daily_summary.csv").exists():
            continue
        if not sinex_file(experiment_dir, doy).exists():
            continue
        days.append(doy)
    return days


@dataclass(frozen=True)
class StationDayComparison:
    """One station-day: the rebuilt metrics against the pre-rebuild CSV's recorded row."""

    doy: int
    arm: str
    station: str
    method: str
    pos_file: Path
    ref_pos: np.ndarray
    recomputed: dict[str, float]
    recorded: dict[str, float]

    @property
    def metric_diffs(self) -> dict[str, float]:
        return {m: abs(self.recomputed[m] - self.recorded[m]) for m in COMPARED_METRICS}

    @property
    def max_diff(self) -> float:
        return max(self.metric_diffs.values())


def compare_station_day(
    pos_file: Path,
    ref_pos: np.ndarray,
    recorded_row: pd.Series,
    *,
    doy: int,
    arm: str,
    station: str,
    method: str,
) -> StationDayComparison | None:
    """Recompute one station-day's metrics from its `.pos` file and diff against the row
    the pre-rebuild code recorded for it. Returns None if the file can't be re-parsed."""
    df = pm.parse_pos_file(pos_file, ref_pos=ref_pos)
    recomputed = pm.compute_metrics(df)
    if recomputed is None:
        return None
    return StationDayComparison(
        doy=doy,
        arm=arm,
        station=station,
        method=method,
        pos_file=pos_file,
        ref_pos=ref_pos,
        recomputed={m: float(recomputed[m]) for m in COMPARED_METRICS},
        recorded={m: float(recorded_row[m]) for m in COMPARED_METRICS},
    )


def run_day_arm(
    experiment_dir: Path, doy: int, arm: str, stations_per_day: int
) -> list[StationDayComparison]:
    """All sampled station-day comparisons for one DOY and one weighting arm."""
    csv_name, method_subdir = ARMS[arm]
    summary = pd.read_csv(results_dir(experiment_dir, doy) / csv_name)
    summary = summary[summary["method"] == method_subdir].set_index("station")

    gt_coords = pm.load_sinex_coords(sinex_file(experiment_dir, doy))
    stations = sample_evenly(sorted(summary.index), stations_per_day)

    comparisons = []
    for station in stations:
        ref_pos = gt_coords.get(station.upper())
        if ref_pos is None:
            # daily_summary.csv already excludes stations absent from the SINEX file
            # (aggregate_daily_metrics's require_snx), so this should not happen for a
            # station actually listed in the CSV - kept as a defensive skip, not silent.
            continue
        pos_file = pos_file_path(experiment_dir, doy, method_subdir, station)
        if not pos_file.exists():
            continue
        row = summary.loc[station]
        if isinstance(row, pd.DataFrame):  # duplicate station rows would collide here
            row = row.iloc[0]
        comparison = compare_station_day(
            pos_file,
            np.array(ref_pos),
            row,
            doy=doy,
            arm=arm,
            station=station,
            method=method_subdir,
        )
        if comparison is not None:
            comparisons.append(comparison)
    return comparisons


def investigate_disagreement(comparison: StationDayComparison) -> str:
    """Best-effort diagnosis for a station-day that disagrees by more than `TOLERANCE_M`.

    Distinguishes two causes rather than assuming the port is wrong: a genuine defect in
    `stec/positioning/metrics.py`, or the recorded CSV having been produced against a
    different reference (day-mean instead of SINEX ground truth - `aggregate_daily_metrics`
    falls back to that when no SINEX file was supplied at the time).
    """
    df_mean = pm.parse_pos_file(comparison.pos_file, ref_pos=None)
    recomputed_mean = pm.compute_metrics(df_mean)
    if recomputed_mean is None:
        return "pos file became unreadable on re-parse - cannot investigate further"

    mean_diff = abs(
        recomputed_mean["error_3d_rms"] - comparison.recorded["error_3d_rms"]
    )
    if mean_diff <= TOLERANCE_M:
        return (
            "recorded row matches a day-mean reference, not SINEX ground truth "
            f"(day-mean 3D diff {mean_diff:.2e} m) - the CSV was produced without "
            "ground-truth scoring for this station-day, not a defect in the port"
        )
    return (
        f"neither SINEX ground truth ({comparison.metric_diffs['error_3d_rms']:.2e} m off) "
        f"nor a day-mean reference ({mean_diff:.2e} m off) explains the recorded value - "
        "treat as a genuine port defect until shown otherwise"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=8,
        help="DOYs to sample, spread across the test period",
    )
    parser.add_argument("--stations-per-day", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_M)
    args = parser.parse_args()

    print(
        "Gate E (metrics half only - see module docstring): rebuilt stec/positioning/"
        "metrics.py vs. pre-rebuild daily_summary*.csv, on real .pos files.\n"
    )

    available_days = discover_sample_days()
    if not available_days:
        if not paths.LEGACY_EXPERIMENTS.exists():
            reason = f"the live PNN_STEC checkout is not present at {paths.LEGACY_ROOT}"
        else:
            reason = (
                "no canonical Finetune_STEC experiment directory has both a "
                "daily_summary.csv and a matching SINEX file"
            )
        print(f"SKIP  {reason}")
        return 0

    sampled_days = sample_evenly(available_days, args.days)
    print(
        f"sampling {len(sampled_days)} of {len(available_days)} available day(s): "
        f"{sampled_days}"
    )

    comparisons: list[StationDayComparison] = []
    for doy in sampled_days:
        experiment_dir = canonical_experiment_dir(doy)
        for arm in ARMS:
            comparisons.extend(
                run_day_arm(experiment_dir, doy, arm, args.stations_per_day)
            )

    if not comparisons:
        print(
            "SKIP  none of the sampled days yielded a comparable station-day "
            "(no SINEX coordinate or .pos file for any sampled station)"
        )
        return 0

    station_width = max(len(c.station) for c in comparisons)
    print(
        f"\n  {'doy':<8}{'arm':<6}{'station':<{station_width + 2}}"
        f"{'3D diff [m]':>14}{'2D diff [m]':>14}{'Up diff [m]':>14}"
    )
    for c in comparisons:
        diffs = c.metric_diffs
        print(
            f"  {c.doy:<8}{c.arm:<6}{c.station:<{station_width + 2}}"
            f"{diffs['error_3d_rms']:>14.2e}{diffs['error_2d_rms']:>14.2e}"
            f"{diffs['u_rms']:>14.2e}"
        )

    max_diffs = {
        m: max(c.metric_diffs[m] for c in comparisons) for m in COMPARED_METRICS
    }
    print(
        f"\n  {len(comparisons)} station-day(s) compared, {len(sampled_days)} day(s) x "
        f"{len(ARMS)} weighting arm(s)"
    )
    print(
        f"  max |difference|   3D {max_diffs['error_3d_rms']:.2e} m   "
        f"2D (horizontal) {max_diffs['error_2d_rms']:.2e} m   "
        f"Up (vertical) {max_diffs['u_rms']:.2e} m"
    )
    print(
        f"  recorded-CSV rounding floor (±{CSV_ROUNDING_M:.0e} m, half-ULP "
        f"{CSV_ROUNDING_M / 2:.1e} m) - the scale below which a difference is unmeasurable"
    )

    failures = [c for c in comparisons if c.max_diff > args.tolerance]
    if not failures:
        print(
            "\n  PASS  the rebuilt metrics.py reproduces every sampled daily_summary.csv "
            "row within CSV rounding (metrics half of Gate E only - PPPx and corrections "
            "generation remain unverified, see module docstring)"
        )
        return 0

    print(
        f"\n  {len(failures)} of {len(comparisons)} station-day(s) exceed tolerance "
        f"({args.tolerance:.1e} m) - investigating"
    )
    for c in failures:
        print(f"    {c.doy} {c.arm} {c.station}: {investigate_disagreement(c)}")

    print("\n  FAIL  see investigation note(s) above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
