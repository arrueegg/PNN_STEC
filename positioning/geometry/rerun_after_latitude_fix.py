"""Re-run inference and PPPx for the recovered station-days whose latitude was repaired.

Follows `repair_recovered_latitude.py`. For each day that had stations repaired, this
  1. copies the affected stations' rows into a one-station-style input tree, so
     generate_stec_corrections.py (one model load per day) rewrites the correction CSVs of
     the affected stations only;
  2. backs up those stations' old correction CSVs and old model/model_iono result
     directories (never overwriting an existing backup) and removes the old result
     directories, because run_positioning_evaluation.py skips a station whose .pos exists;
  3. regenerates corrections and reruns PPPx under iono and elev weighting for the Direct
     STEC and Pretrained experiments, the same commands recover_day.py uses;
  4. verifies each daily_summary*.csv did not shrink and only the affected stations' model
     rows changed.
Progress is appended to a JSONL manifest, so a restart skips finished station-days. VTEC
experiments are never touched.

Usage::

    python positioning/geometry/rerun_after_latitude_fix.py [--days 133 134] [--parallel 6]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_recovered_day import _write_recovered_day_atomically  # noqa: E402
from recover_day import resolve_experiment  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
YEAR = 2024
RECOVERED_ROOT = REPO / "data" / "recovered_stec_db"
LATITUDE_BACKUP_ROOT = REPO / "data" / "recovered_stec_db.bak_20261005"
WORK_ROOT = REPO / "data" / "latfix_work"
BACKUP_ROOT = REPO / "data" / f"latfix_backup_{date.today():%Y%m%d}"
MANIFEST = REPO / "logs" / "latfix_rerun_manifest.jsonl"
MODEL_KINDS = ("STEC", "Pretrained_STEC")
WEIGHTINGS = ("iono", "elev")
SUMMARY_FILES = {"elev": "daily_summary.csv", "iono": "daily_summary_iono.csv"}
MODEL_METHOD = {"elev": "model", "iono": "model_iono"}
MIN_FREE_GB = 40
# Station-days finished by hand in the pilots, before this driver existed.
PILOT_DONE = {(133, "MRO1"), (257, "MRO1"), (228, "RGDG"), (209, "SUTM")}


def read_manifest() -> dict[tuple[int, str], dict]:
    latest: dict[tuple[int, str], dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            record = json.loads(line)
            latest[(record["doy"], record["station"])] = record
    return latest


def record(doy: int, station: str, status: str, detail: str = "") -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as handle:
        handle.write(
            json.dumps(
                {"doy": doy, "station": station, "status": status, "detail": detail}
            )
            + "\n"
        )
    logger.info(f"DOY {doy} {station}: {status} {detail}")


def _latitudes_by_station(path: Path, doy: int) -> dict[str, float]:
    with h5py.File(path, "r") as handle:
        data = handle[str(YEAR)][f"{doy:03d}"]["all_data"]
        names, latitudes = data["station"][:], data["lat_sta"][:]
    return {
        n.decode(): float(latitudes[np.flatnonzero(names == n)[0]])
        for n in np.unique(names)
    }


def repaired_stations_by_day() -> dict[int, list[str]]:
    """Stations whose lat_sta in the backup (pre-repair) file differs from the current one."""
    by_day: dict[int, list[str]] = {}
    for backup in sorted(LATITUDE_BACKUP_ROOT.glob(f"{YEAR}/*/ccl_*_30_5.h5")):
        doy = int(backup.parent.name)
        current = RECOVERED_ROOT / backup.relative_to(LATITUDE_BACKUP_ROOT)
        before, after = (
            _latitudes_by_station(backup, doy),
            _latitudes_by_station(current, doy),
        )
        changed = [s for s in after if abs(after[s] - before.get(s, after[s])) > 0.01]
        if changed:
            by_day[doy] = sorted(changed)
    return by_day


def run_logged(command: list[str], log_path: Path) -> int:
    with open(log_path, "a") as log:
        return subprocess.run(
            [str(c) for c in command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def backup_once(source: Path, destination: Path) -> None:
    if destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def summary_snapshot(experiment: Path, doy: int) -> dict[str, pd.DataFrame]:
    results = experiment / "positioning" / "results" / f"{YEAR}{doy:03d}"
    return {
        weighting: pd.read_csv(results / name).set_index(["station", "method"])
        for weighting, name in SUMMARY_FILES.items()
        if (results / name).exists()
    }


def verify_summaries(
    experiment: Path, doy: int, before: dict[str, pd.DataFrame], stations: list[str]
) -> str | None:
    after = summary_snapshot(experiment, doy)
    for weighting, old in before.items():
        new = after.get(weighting)
        if new is None or len(new) < len(old):
            return f"{experiment.name[:8]} {weighting} summary shrank or vanished"
        allowed = {(s, MODEL_METHOD[weighting]) for s in stations}
        for key in new.index:
            if key in allowed:
                continue
            if key not in old.index or not old.loc[key].equals(new.loc[key]):
                return f"{experiment.name[:8]} {weighting} unexpected change at {key}"
    return None


def process_day(doy: int, stations: list[str], parallel: int) -> None:
    date_text = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)).strftime(
        "%Y-%m-%d"
    )
    work = WORK_ROOT / f"{YEAR}{doy:03d}"
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "run.log"
    experiments = {kind: resolve_experiment(kind, doy) for kind in MODEL_KINDS}
    if any(exp is None for exp in experiments.values()):
        for station in stations:
            record(doy, station, "failed", "no usable experiment checkpoint")
        return

    # Only stations that were already in the positioning set (had a correction CSV).
    targets = []
    for station in stations:
        has_csv = all(
            (
                exp
                / "positioning"
                / "stec_corrections"
                / f"{YEAR}{doy:03d}"
                / f"{station}.csv"
            ).exists()
            for exp in experiments.values()
        )
        if has_csv:
            targets.append(station)
        else:
            record(
                doy,
                station,
                "skipped",
                "no prior corrections CSV (not in positioning set)",
            )
    if not targets:
        return

    recovered_file = (
        RECOVERED_ROOT / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5"
    )
    with h5py.File(recovered_file, "r") as handle:
        data = handle[str(YEAR)][f"{doy:03d}"]["all_data"][:]
    subset_root = work / "recovered"
    shutil.rmtree(subset_root, ignore_errors=True)
    _write_recovered_day_atomically(
        data[np.isin(data["station"], [s.encode("ascii") for s in targets])],
        subset_root / str(YEAR) / f"{doy:03d}" / recovered_file.name,
        YEAR,
        doy,
    )

    snapshots = {kind: summary_snapshot(exp, doy) for kind, exp in experiments.items()}
    day_backup = BACKUP_ROOT / f"{YEAR}{doy:03d}"
    for kind, exp in experiments.items():
        results = exp / "positioning" / "results" / f"{YEAR}{doy:03d}"
        for name in SUMMARY_FILES.values():
            backup_once(results / name, day_backup / kind / name)
        for station in targets:
            backup_once(
                exp
                / "positioning"
                / "stec_corrections"
                / f"{YEAR}{doy:03d}"
                / f"{station}.csv",
                day_backup / kind / f"{station}.csv",
            )
            for subdir in ("model", "model_iono"):
                backup_once(
                    results / subdir / station,
                    day_backup / kind / f"{subdir}_{station}",
                )
                # Without this PPPx would skip the station: an existing .pos counts as done.
                shutil.rmtree(results / subdir / station, ignore_errors=True)

    rinex_dir = work / "rinex"
    rinex_dir.mkdir(exist_ok=True)
    failures: dict[str, str] = {}
    for kind, exp in experiments.items():
        code = run_logged(
            [
                "python",
                "positioning/scripts/generate_stec_corrections.py",
                "--experiment",
                exp.name,
                "--date",
                date_text,
                "--gnss_path",
                subset_root,
            ],
            log_path,
        )
        if code != 0:
            for station in targets:
                failures[station] = f"{kind} corrections failed"
            continue
        for weighting in WEIGHTINGS:
            run_logged(
                [
                    "python",
                    "positioning/positioning_eval/run_positioning_evaluation.py",
                    "--experiment",
                    exp.relative_to(REPO),
                    "--date",
                    date_text,
                    "--stations",
                    *targets,
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
        results = exp / "positioning" / "results" / f"{YEAR}{doy:03d}"
        for pattern in ("**/.*.stat", "**/.*.log"):
            for stale in results.glob(pattern):
                stale.unlink(missing_ok=True)

    for kind, exp in experiments.items():
        problem = verify_summaries(exp, doy, snapshots[kind], targets)
        if problem:
            for station in targets:
                failures.setdefault(station, problem)
        results = exp / "positioning" / "results" / f"{YEAR}{doy:03d}"
        for station in targets:
            for subdir, weighting in (("model", "elev"), ("model_iono", "iono")):
                if not (
                    results / subdir / station / f"{station}_{subdir}.pos"
                ).exists():
                    failures.setdefault(
                        station, f"{kind} {weighting} PPPx produced no .pos"
                    )
    shutil.rmtree(rinex_dir, ignore_errors=True)
    for station in targets:
        if station in failures:
            record(doy, station, "failed", failures[station])
        else:
            record(doy, station, "done")


def free_gb() -> float:
    return shutil.disk_usage(REPO).free / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", help="restrict to these DOYs")
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    by_day = repaired_stations_by_day()
    finished = {
        key
        for key, rec in read_manifest().items()
        if rec["status"] in ("done", "skipped")
    }
    finished |= PILOT_DONE
    pending_days = {
        doy: [s for s in stations if (doy, s) not in finished]
        for doy, stations in by_day.items()
        if args.days is None or doy in args.days
    }
    pending_days = {doy: s for doy, s in pending_days.items() if s}
    total = sum(len(s) for s in pending_days.values())
    logger.info(f"{len(pending_days)} days, {total} station-days pending")

    started = time.time()
    for doy in sorted(pending_days):
        if free_gb() < MIN_FREE_GB:
            logger.error(
                f"stopping: only {free_gb():.0f} GB free, floor is {MIN_FREE_GB} GB"
            )
            sys.exit(3)
        process_day(doy, pending_days[doy], args.parallel)
    logger.info(f"finished in {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
