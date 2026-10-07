"""Full run of the absent-constellation fill (see `fill_absent_constellation.py`).

Processes every qualifying station-day of `scope_dryrun.csv` (a single-constellation
station-day whose correction CSV carries exactly one of G/E), one day at a time with all of
that day's stations together, so each arm's model loads once per day. Days run concurrently
in a small process pool. Resumable: a JSONL manifest records the final status per
(day, station) and a restart skips `done` and `skipped` ones. Missing products/RINEX, or a
RINEX without the absent constellation, are recorded as skipped, never improvised around.

Usage::

    python positioning/geometry/run_constellation_fill.py [--days 150] [--workers 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "positioning_eval"))

import fill_absent_constellation as fill  # noqa: E402
import independent_geometry as ig  # noqa: E402
from download_rinex import download_rinex_batch  # noqa: E402
from recover_day import resolve_experiment  # noqa: E402
from validate_independent_geometry import products_dir  # noqa: E402

logger = logging.getLogger(__name__)

REPO = fill.REPO
YEAR = fill.YEAR
SCOPE_CSV = fill.WORK_ROOT / "scope_dryrun.csv"
MANIFEST = REPO / "logs" / "constellation_fill_manifest.jsonl"
FULL_MERGED_ROOT = fill.WORK_ROOT / "merged_full"
RINEX_ROOT = fill.WORK_ROOT / "rinex_full"
MIN_FREE_GB = 40
RINEX_DOWNLOAD_THREADS = 4
FINISHED_STATUSES = ("done", "skipped")
# Station-days already completed by the pilot.
PILOT_DONE = {(122, "BIK0"), (150, "ZECK"), (150, "FAA1")}
KEEP_RINEX = os.environ.get("KEEP_RINEX") == "1"
KEEP_DIAGNOSTICS = os.environ.get("KEEP_DIAGNOSTICS") == "1"


def read_manifest() -> dict[tuple[int, str], dict]:
    latest: dict[tuple[int, str], dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            entry = json.loads(line)
            latest[(entry["doy"], entry["station"])] = entry
    return latest


def record(doy: int, station: str, status: str, detail: str = "", **extra) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "doy": doy,
        "station": station,
        "status": status,
        "detail": detail,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }
    with open(MANIFEST, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    logger.info(f"DOY {doy} {station}: {status} {detail}")


def qualifying_station_days() -> dict[int, list[str]]:
    scope = pd.read_csv(SCOPE_CSV)
    qualifying = scope[scope.consts.isin(["G", "E"])]
    return {int(doy): sorted(g.station) for doy, g in qualifying.groupby("doy")}


def snapshot_before(
    experiment: Path, doy: int, backup_dir: Path
) -> dict[str, pd.DataFrame]:
    """Summaries as they were before the fill: the backup if one exists (restart-safe)."""
    snapshot = {}
    for weighting, name in fill.SUMMARY_FILES.items():
        for path in (
            backup_dir / name,
            experiment / "positioning/results" / f"{YEAR}{doy:03d}" / name,
        ):
            if path.exists():
                snapshot[weighting] = pd.read_csv(path).set_index(["station", "method"])
                break
    return snapshot


def prepare_targets(
    doy: int, stations: list[str], products: Path, rinex_dir: Path
) -> dict[str, "pd.DataFrame | None"]:
    """Merged rows per station that can be filled; records a skip for those that cannot."""
    downloaded, failures = download_rinex_batch(
        stations, YEAR, doy, str(rinex_dir), max_workers=RINEX_DOWNLOAD_THREADS
    )
    merged_by_station = {}
    for station in stations:
        if station not in downloaded:
            record(
                doy, station, "skipped", f"rinex unavailable: {failures.get(station)}"
            )
            continue
        try:
            _, original = fill.find_source(station, doy)
            geometry = ig.compute_station_geometry(
                station,
                YEAR,
                doy,
                downloaded[station],
                next(products.glob("*ORB.SP3")),
                next(products.glob("IGS*SNX")),
            )
            merged, letters = fill.merge_absent_constellations(original, geometry)
        except (LookupError, ValueError, StopIteration, OSError) as error:
            record(
                doy,
                station,
                "failed",
                f"geometry/merge: {type(error).__name__}: {error}",
            )
            continue
        if not letters:
            record(
                doy,
                station,
                "skipped",
                "RINEX/SP3 yield no rows for the absent constellation",
            )
            continue
        merged_by_station[station] = (merged, letters, len(merged) - len(original))
    return merged_by_station


def process_day(doy: int, stations: list[str], parallel: int) -> None:
    started = time.time()
    tag = f"{YEAR}{doy:03d}"
    date_text = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)).strftime(
        "%Y-%m-%d"
    )
    log_path = fill.WORK_ROOT / "full_logs" / f"fill_{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    experiments = {arm: resolve_experiment(arm, doy) for arm in fill.ARMS}
    try:
        products = products_dir(doy)
        if not (list(products.glob("*ORB.SP3")) and list(products.glob("IGS*SNX"))):
            raise IndexError("SP3/SINEX missing")
    except IndexError as error:
        for station in stations:
            record(doy, station, "skipped", f"products missing: {error}")
        return
    if any(exp is None for exp in experiments.values()):
        for station in stations:
            record(doy, station, "failed", "no usable experiment checkpoint")
        return
    # Only stations that already have a correction CSV in every arm are in the positioning set.
    stations = [
        s
        for s in stations
        if all(fill.correction_csv(e, doy, s).exists() for e in experiments.values())
    ]
    if not stations:
        return

    rinex_dir = RINEX_ROOT / tag
    rinex_dir.mkdir(parents=True, exist_ok=True)
    merged_by_station = prepare_targets(doy, stations, products, rinex_dir)
    targets = sorted(merged_by_station)
    if not targets:
        shutil.rmtree(rinex_dir, ignore_errors=True)
        return
    # Both the merged file and the generator see only the targets, so no other station's CSV is rewritten.
    fill._write_recovered_day_atomically(
        np.concatenate([merged_by_station[s][0] for s in targets]),
        FULL_MERGED_ROOT / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5",
        YEAR,
        doy,
    )

    day_backup = fill.BACKUP_ROOT / tag
    snapshots = {
        arm: snapshot_before(exp, doy, day_backup / arm)
        for arm, exp in experiments.items()
    }
    for arm, exp in experiments.items():
        results = exp / "positioning" / "results" / tag
        for name in fill.SUMMARY_FILES.values():
            fill.backup_once(results / name, day_backup / arm / name)
        for station in targets:
            fill.backup_once(
                fill.correction_csv(exp, doy, station),
                day_backup / arm / f"{station}.csv",
            )
            for subdir in ("model", "model_iono"):
                fill.backup_once(
                    results / subdir / station, day_backup / arm / f"{subdir}_{station}"
                )

    failures: dict[str, str] = {}
    for arm, exp in experiments.items():
        results = exp / "positioning" / "results" / tag
        if fill.run_logged(
            [
                "python",
                "positioning/scripts/generate_stec_corrections.py",
                "--experiment",
                exp.name,
                "--date",
                date_text,
                "--gnss_path",
                FULL_MERGED_ROOT,
            ],
            log_path,
        ):
            for station in targets:
                failures.setdefault(station, f"{arm} corrections failed")
            continue
        usable = []
        for station in targets:
            comparison = fill.compare_original_rows(
                day_backup / arm / f"{station}.csv",
                fill.correction_csv(exp, doy, station),
            )
            if (
                not comparison["original_rows_same_key_set"]
                or comparison["constellations_new"] != "EG"
            ):
                failures.setdefault(station, f"{arm} CSV check failed: {comparison}")
            else:
                usable.append(station)
        if not usable:
            continue
        for (
            station
        ) in usable:  # an existing .pos counts as done to run_positioning_evaluation
            for subdir in ("model", "model_iono"):
                shutil.rmtree(results / subdir / station, ignore_errors=True)
        for weighting in fill.WEIGHTINGS:
            fill.run_logged(
                [
                    "python",
                    "positioning/positioning_eval/run_positioning_evaluation.py",
                    "--experiment",
                    exp.relative_to(REPO),
                    "--date",
                    date_text,
                    "--stations",
                    *usable,
                    "--weight_opt",
                    weighting,
                    "--parallel",
                    parallel,
                    "--rinex_dir",
                    rinex_dir,
                    "--no_cleanup",
                ],
                log_path,
            )
        if not KEEP_DIAGNOSTICS:
            for pattern in ("**/.*.stat", "**/.*.log"):
                for stale in results.glob(pattern):
                    stale.unlink(missing_ok=True)
        problem = fill.verify_summaries(
            snapshots[arm], fill.summary_snapshot(exp, doy), targets
        )
        if problem:
            for station in targets:
                failures.setdefault(station, f"{arm}: {problem}")
        for station in usable:
            for subdir in ("model", "model_iono"):
                if not (
                    results / subdir / station / f"{station}_{subdir}.pos"
                ).exists():
                    failures.setdefault(
                        station, f"{arm} {subdir}: PPPx produced no .pos"
                    )
    if not KEEP_RINEX:
        shutil.rmtree(rinex_dir, ignore_errors=True)
    seconds = (time.time() - started) / max(len(targets), 1)
    for station in targets:
        _, letters, added = merged_by_station[station]
        if station in failures:
            record(doy, station, "failed", failures[station])
        else:
            record(
                doy,
                station,
                "done",
                appended=letters,
                rows_added=added,
                seconds_per_station=round(seconds, 1),
            )


def free_gb() -> float:
    return shutil.disk_usage(REPO).free / 1e9


def run_day(arguments: tuple[int, list[str], int]) -> int:
    doy, stations, parallel = arguments
    if free_gb() < MIN_FREE_GB:
        logger.error(
            f"DOY {doy}: only {free_gb():.0f} GB free, floor is {MIN_FREE_GB} GB; not started"
        )
        return 3
    try:
        process_day(doy, stations, parallel)
    except Exception as error:  # noqa: BLE001 - a crashed day must not stop the pool; it is recorded and retried on restart
        for station in stations:
            record(
                doy, station, "failed", f"day crashed: {type(error).__name__}: {error}"
            )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--parallel", type=int, default=4, help="PPPx stations in parallel"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    finished = {
        key
        for key, entry in read_manifest().items()
        if entry["status"] in FINISHED_STATUSES
    }
    finished |= PILOT_DONE
    pending = {
        doy: [s for s in stations if (doy, s) not in finished]
        for doy, stations in qualifying_station_days().items()
        if args.days is None or doy in args.days
    }
    pending = {doy: s for doy, s in pending.items() if s}
    logger.info(
        f"{len(pending)} days, {sum(map(len, pending.values()))} station-days pending"
    )
    if args.dry_run:
        return
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        codes = list(
            pool.map(
                run_day, [(doy, s, args.parallel) for doy, s in sorted(pending.items())]
            )
        )
    logger.info(f"finished in {(time.time() - started) / 3600:.2f} h")
    if 3 in codes:
        sys.exit(3)


if __name__ == "__main__":
    main()
