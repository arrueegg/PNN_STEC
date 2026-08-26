"""Recover one day of the station-days the STEC database excluded, end to end.

Chains the pieces that are individually validated in `camaliot_geometry.py` and
`build_recovered_day.py`: fetch the day's inputs, derive geometry with the production
CamaliotGnss binary, write a database-format file, run each model over it, and position.

Only stations classified `all ML methods missing (station absent from STEC DB)` by
`src/analysis/positioning_coverage.py` are processed - the station-days where the CAS DCB
gate, not a positioning failure, is the reason no correction exists.

Broadcast navigation is required (it is what makes the elevation and azimuth), and comes
from the local product tree when present, otherwise CDDIS. The DCB file is required only
because the binary's config demands a path: it calibrates STEC, which this pipeline
discards and the model predicts instead, so a neighbouring day's file is an acceptable
substitute and is logged when used.

Usage::

    python positioning/geometry/recover_day.py --year 2024 --doy 323
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

# stec/ is not an installed package and this file is invoked as a bare script
# (`python positioning/geometry/recover_day.py ...`) by scripts/run_station_recovery.sh,
# which does not put the repo root on sys.path on its own - same situation as
# positioning/positioning_eval/metrics.py, which resolves it the same way.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stec.config.paths import analysis_result_dir  # noqa: E402

# The canonical, currently-absent-station-days coverage file (see
# stec.pipeline.stages.py's positioning_coverage stage). The old default,
# multiday_results/positioning_runs/full_coverage/coverage.csv, is marked superseded
# (.superseded.json) and still lists the original 2,311 absent station-days, including
# ~750 a first recovery sweep already fixed - pointing a re-run at it would redo work
# that is already done. Resolved through stec.config.paths rather than hardcoded again:
# a literal result path here has already broken three times (see CLAUDE.md's Gotchas).
DEFAULT_COVERAGE = (
    analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
)
PRODUCTS = Path("/scratch2/miten/gim_operational_parallel/PRODUCTS")
CDDIS_NAV = (
    "https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy:03d}/"
    "{yy}p/BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx.gz"
)
ABSENT_CAUSE = "all ML methods missing (station absent from STEC DB)"

EXPERIMENT_PATTERNS = {
    "STEC": "Finetune_STEC_2024_{doy:03d}_BayesianResNetSTEC_*_SWI",
    "VTEC": "Finetune_VTEC_2024_{doy:03d}_MLP_LaplacianNLL_*_woYear",
    # Pinned, not globbed: several Pretrain_STEC_*_SWI variants exist and only this one
    # is the paper's model. Globbing picked a dropout variant with no checkpoint.
    "Pretrained_STEC": (
        "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2"
        "_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1"
        "_kl5w0.1_lw1e-1_SWI"
    ),
}


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    logger.info("$ " + " ".join(str(c) for c in command[:6]) + " ...")
    return subprocess.run([str(c) for c in command], cwd=REPO, **kwargs)


def ensure_nav(year: int, doy: int, workdir: Path) -> Path:
    """Broadcast navigation, from the local product tree or CDDIS."""
    local = sorted((PRODUCTS / str(year) / f"{doy:03d}").glob("BRDC00IGS_R_*_MN.rnx"))
    if local:
        return local[0]

    target = workdir / f"BRDC00IGS_R_{year}{doy:03d}0000_01D_MN.rnx"
    if target.exists():
        return target
    url = CDDIS_NAV.format(year=year, doy=doy, yy=str(year)[2:])
    archive = target.with_suffix(".rnx.gz")
    result = run(
        [
            "wget",
            "--netrc",
            "--auth-no-challenge=on",
            "-nv",
            "-t",
            "3",
            "--connect-timeout=15",
            "--read-timeout=90",
            "-O",
            archive,
            url,
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not archive.exists() or archive.stat().st_size == 0:
        raise SystemExit(f"no broadcast navigation for {year} DOY {doy}")
    run(["gunzip", "-f", archive], check=True)
    return target


def ensure_bsx(year: int, doy: int) -> Path:
    """The day's CAS DCB file, or the nearest one - it does not affect geometry."""
    exact = sorted((PRODUCTS / str(year) / f"{doy:03d}").glob("*DCB.BSX"))
    if exact:
        return exact[0]
    candidates = sorted(
        PRODUCTS.glob(f"{year}/*/*DCB.BSX"), key=lambda p: abs(int(p.parent.name) - doy)
    )
    if not candidates:
        raise SystemExit(f"no CAS DCB file available anywhere for {year}")
    logger.warning(
        f"DOY {doy}: no DCB for this day, substituting DOY {candidates[0].parent.name}. "
        "Affects only STEC calibration, which this pipeline discards."
    )
    return candidates[0]


def resolve_experiment(kind: str, doy: int) -> Path | None:
    """The experiment directory for this model, requiring a usable checkpoint."""
    matches = [
        path
        for path in sorted(
            (REPO / "experiments").glob(EXPERIMENT_PATTERNS[kind].format(doy=doy))
        )
        if (path / "model").is_dir() and any((path / "model").glob("*.pth"))
    ]
    if not matches:
        logger.warning(f"DOY {doy}: no {kind} experiment with a checkpoint")
        return None
    return matches[0]


def run_models(args, stations: list[str], rinex_dir: Path) -> None:
    """Inference and positioning for every model over the recovered day.

    `rinex_dir` is shared across all three arms via `--rinex_dir`: without it, each
    of the STEC/VTEC/Pretrained_STEC calls to run_positioning_evaluation.py does its
    own download_rinex_batch into its own experiment-specific directory, so the same
    station-day gets fetched from CDDIS up to three more times here on top of the one
    fetch the geometry stage already did. download_rinex_file() skips any
    station/day already present in `rinex_dir`, so pointing every arm at the same
    directory turns the redundant re-fetches into no-ops instead.

    Also passes `--no_cleanup`, for a reason unrelated to RINEX: any experiment's
    `positioning/evaluation/<day>/products` can be a *lender*, not just a borrower -
    `download_products.py::reuse_from_other_runs` symlinks a missing product in from
    whichever other experiment already fetched it for the same day, globbing across
    all of `experiments/` (see its own docstring). Without `--no_cleanup`,
    run_positioning_evaluation.py's Step 8 `shutil.rmtree()`s this arm's `products_dir`
    on the way out, which silently breaks every symlink any other experiment (present
    or future) has pointed at one of these files - exactly what happened to
    `experiments/Reference_STEC_Oracle`, whose SINEX symlinks into
    `Pretrain_STEC_..._SWI`'s products were destroyed by this exact call, unnoticed
    because `oracle_benchmark`'s declared pipeline input never tracked that tree (see
    `stec/pipeline/stages.py`'s `oracle_benchmark` Stage). One experiment's cleanup
    must never be able to invalidate another experiment's inputs; not cleaning up the
    lender side of that relationship is the cheap way to guarantee it. Cost is small:
    a `products_dir` is ~50-60 MB (SP3/CLK/ERP/OBX/GIM/SNX), not the ~1 GB/day RINEX
    that `--no_cleanup` would also retain elsewhere - RINEX here is unaffected, because
    `rinex_dir` is caller-owned (see `owns_rinex_dir` in
    run_positioning_evaluation.py) and is already left alone regardless of
    `--no_cleanup`; this module cleans it up itself via `--keep_rinex`.
    """
    date = pd.Timestamp(f"{args.year}-01-01") + pd.Timedelta(days=args.doy - 1)
    for kind in EXPERIMENT_PATTERNS:
        experiment = resolve_experiment(kind, args.doy)
        if experiment is None:
            continue
        corrections = run(
            [
                "python",
                "positioning/scripts/generate_stec_corrections.py",
                "--experiment",
                experiment.name,
                "--date",
                date.strftime("%Y-%m-%d"),
                "--gnss_path",
                args.output_root,
            ]
        )
        if corrections.returncode != 0:
            logger.error(
                f"DOY {args.doy}: {kind} corrections failed, skipping its positioning"
            )
            continue
        run(
            [
                "python",
                "positioning/positioning_eval/run_positioning_evaluation.py",
                "--experiment",
                str(experiment.relative_to(REPO)),
                "--date",
                date.strftime("%Y-%m-%d"),
                "--stations",
                *stations,
                "--weight_opt",
                args.weight_opt,
                "--parallel",
                args.parallel,
                "--rinex_dir",
                rinex_dir,
                "--no_cleanup",
            ]
        )

        if not args.keep_diagnostics:
            results = (
                experiment / "positioning" / "results" / f"{args.year}{args.doy:03d}"
            )
            for pattern in ("**/.*.stat", "**/.*.log"):
                for stale in results.glob(pattern):
                    stale.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=DEFAULT_COVERAGE,
    )
    parser.add_argument("--weight_opt", default="iono", choices=["iono", "elev"])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--workdir", type=Path, default=Path("data/recovery_work"))
    parser.add_argument(
        "--output_root", type=Path, default=Path("data/recovered_stec_db")
    )
    parser.add_argument(
        "--stages",
        default="all",
        choices=["all", "geometry", "models"],
        help="'geometry' needs no GPU and can run alongside a training job; "
        "'models' does inference and positioning over an existing file",
    )
    parser.add_argument("--keep_rinex", action="store_true")
    parser.add_argument(
        "--keep_diagnostics",
        action="store_true",
        help="retain PPPx .stat/.log; nothing in the analysis path reads them",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    coverage = pd.read_csv(args.coverage)
    stations = sorted(
        coverage[(coverage.doy == args.doy) & (coverage.cause == ABSENT_CAUSE)]
        .station.str.upper()
        .unique()
    )
    if not stations:
        logger.info(f"DOY {args.doy}: nothing to recover")
        return
    logger.info(
        f"DOY {args.doy}: recovering {len(stations)} station(s): {', '.join(stations)}"
    )

    recovered = (
        args.output_root
        / str(args.year)
        / f"{args.doy:03d}"
        / f"ccl_{args.year}{args.doy:03d}_30_5.h5"
    )
    # Computed unconditionally (not just for the geometry stage below) so a
    # `--stages models`-only invocation can still pass `--rinex_dir` and reuse
    # whatever the geometry stage left behind with `--keep_rinex`, or download once
    # here and share it across all three model arms rather than each downloading its
    # own copy.
    workdir = args.workdir / f"{args.year}{args.doy:03d}"
    rinex_dir = workdir / "rinex"

    if args.stages == "models":
        if not recovered.exists():
            logger.error(
                f"DOY {args.doy}: no recovered file, run the geometry stage first"
            )
            return
        run_models(args, stations, rinex_dir)
        if not args.keep_rinex:
            shutil.rmtree(rinex_dir, ignore_errors=True)
        return

    rinex_dir.mkdir(parents=True, exist_ok=True)
    nav, bsx = ensure_nav(args.year, args.doy, workdir), ensure_bsx(args.year, args.doy)

    sys.path.insert(0, str(REPO / "positioning" / "positioning_eval"))
    from download_rinex import download_rinex_batch  # noqa: E402

    _, rinex_failures = download_rinex_batch(
        stations, args.year, args.doy, rinex_dir, logger, max_workers=4 * args.parallel
    )
    for station in sorted(rinex_failures):
        logger.warning(f"DOY {args.doy}: {station} RINEX: {rinex_failures[station]}")

    built = run(
        [
            "python",
            "positioning/geometry/build_recovered_day.py",
            "--year",
            args.year,
            "--doy",
            args.doy,
            "--stations",
            ",".join(stations),
            "--rinex_dir",
            rinex_dir,
            "--nav",
            nav,
            "--bsx",
            bsx,
            "--workdir",
            workdir / "camaliot",
            "--output_root",
            args.output_root,
            "--parallel",
            args.parallel,
        ]
    )
    if built.returncode != 0:
        logger.error(f"DOY {args.doy}: geometry build failed, nothing to position")
        return

    if args.stages == "geometry":
        if not args.keep_rinex:
            shutil.rmtree(rinex_dir, ignore_errors=True)
        logger.info(f"DOY {args.doy}: geometry done")
        return

    run_models(args, stations, rinex_dir)

    if not args.keep_rinex:
        shutil.rmtree(rinex_dir, ignore_errors=True)
    logger.info(f"DOY {args.doy}: done")


if __name__ == "__main__":
    main()
