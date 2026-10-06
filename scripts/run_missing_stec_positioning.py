"""Run PPPx for Direct STEC and Pretrained on station-days that were never positioned.

At seven stations (HLFX, BRST, DUMG, HKSL, GLSV, HRAO, BAIE) the Direct STEC and Pretrained
corrections CSVs exist, but PPPx was never run for those arms, so the station-days appeared
as "some ML methods missing (per-method failure)" once the VTEC arm was solved there by
`scripts/rerun_vtec_slm_positioning.py`. Nothing is regenerated here: the existing CSVs are
used as they are. Targets per day, arm and weighting are stations that have a corrections CSV, no result
directory for that arm and weighting, and a solved VTEC iono .pos (so that the
station-day is in the positioning set at all).

Per day and arm: back up the daily summaries, run PPPx under iono and elev weighting for the
targets, drop .stat/.log and RINEX, and verify that no existing summary row was lost or
changed (new rows are expected). Progress goes to a JSONL manifest so a restart skips
finished (arm, day, station) entries.

Usage::

    python scripts/run_missing_stec_positioning.py [--days 124 125] [--parallel 6] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "positioning" / "geometry"))

from recover_day import resolve_experiment  # noqa: E402

logger = logging.getLogger(__name__)

YEAR = 2024
FIRST_DOY, LAST_DOY = 122, 366
ARMS = ("STEC", "Pretrained_STEC")
BACKUP_ROOT = REPO / "data" / f"missing_stec_backup_{date.today():%Y%m%d}"
WORK_ROOT = REPO / "data" / "missing_stec_work"
MANIFEST = REPO / "logs" / "missing_stec_positioning_manifest.jsonl"
WEIGHTINGS = ("iono", "elev")
SUMMARY_FILES = {"elev": "daily_summary.csv", "iono": "daily_summary_iono.csv"}
MODEL_SUBDIR = {"elev": "model", "iono": "model_iono"}
MIN_FREE_GB = 40


def finished_entries() -> set[tuple[str, int, str]]:
    done: set[tuple[str, int, str]] = set()
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            entry = json.loads(line)
            if entry["status"] == "done":
                done.add((entry["arm"], entry["doy"], entry["station"]))
    return done


def record(arm: str, doy: int, station: str, status: str, detail: str = "") -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "arm": arm,
        "doy": doy,
        "station": station,
        "status": status,
        "detail": detail,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(MANIFEST, "a") as handle:
        handle.write(json.dumps(entry) + "\n")
    logger.info(f"{arm} DOY {doy} {station}: {status} {detail}")


def run_logged(command: list, log_path: Path) -> int:
    with open(log_path, "a") as log:
        return subprocess.run(
            [str(c) for c in command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def summary_snapshot(results: Path) -> dict[str, pd.DataFrame]:
    return {
        weighting: pd.read_csv(results / name).set_index(["station", "method"])
        for weighting, name in SUMMARY_FILES.items()
        if (results / name).exists()
    }


def existing_rows_intact(results: Path, before: dict[str, pd.DataFrame]) -> str | None:
    after = summary_snapshot(results)
    for weighting, old in before.items():
        new = after.get(weighting)
        if new is None:
            return f"{weighting} summary vanished"
        for key in old.index:
            if key not in new.index:
                return f"{weighting} lost row {key}"
            if not old.loc[key].equals(new.loc[key]):
                return f"{weighting} changed row {key}"
    return None


def targets_for(
    arm_experiment: Path, vtec_experiment: Path, tag: str, subdir: str
) -> list[str]:
    corrections = arm_experiment / "positioning" / "stec_corrections" / tag
    if not corrections.exists():
        return []
    results = arm_experiment / "positioning" / "results" / tag
    vtec_results = vtec_experiment / "positioning" / "results" / tag
    targets = []
    for csv in sorted(corrections.glob("*.csv")):
        station = csv.stem
        # Several stations were solved under one weighting only, so this is per weighting.
        never_positioned = not (results / subdir / station).exists()
        vtec_solved = (
            vtec_results / "model_iono" / station / f"{station}_model_iono.pos"
        ).exists()
        if never_positioned and vtec_solved:
            targets.append(station)
    return targets


def process(arm: str, doy: int, parallel: int, done: set, dry_run: bool) -> int:
    tag = f"{YEAR}{doy:03d}"
    arm_experiment = resolve_experiment(arm, doy)
    vtec_experiment = resolve_experiment("VTEC", doy)
    if arm_experiment is None or vtec_experiment is None:
        return 0
    targets = {
        weighting: [
            s
            for s in targets_for(arm_experiment, vtec_experiment, tag, subdir)
            if (arm, doy, s) not in done
        ]
        for weighting, subdir in MODEL_SUBDIR.items()
    }
    n_targets = sum(len(t) for t in targets.values())
    if dry_run or not n_targets:
        return n_targets

    date_text = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)).strftime(
        "%Y-%m-%d"
    )
    results = arm_experiment / "positioning" / "results" / tag
    work = WORK_ROOT / arm / tag
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "run.log"
    for name in SUMMARY_FILES.values():
        backup = BACKUP_ROOT / arm / tag / name
        if (results / name).exists() and not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results / name, backup)
    before = summary_snapshot(results)

    rinex_dir = work / "rinex"
    rinex_dir.mkdir(exist_ok=True)
    for weighting in WEIGHTINGS:
        if not targets[weighting]:
            continue
        run_logged(
            [
                "python",
                "positioning/positioning_eval/run_positioning_evaluation.py",
                "--experiment",
                arm_experiment.relative_to(REPO),
                "--date",
                date_text,
                "--stations",
                *targets[weighting],
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
    if os.environ.get("KEEP_RINEX") != "1":
        shutil.rmtree(rinex_dir, ignore_errors=True)
    if os.environ.get("KEEP_DIAGNOSTICS") != "1":
        for pattern in ("**/*.stat", "**/*.log"):
            for stale in results.glob(pattern):
                stale.unlink(missing_ok=True)

    problem_all = existing_rows_intact(results, before)
    for station in sorted(set(targets["iono"]) | set(targets["elev"])):
        problem = problem_all
        if problem is None:
            for weighting, subdir in MODEL_SUBDIR.items():
                pos = results / subdir / station / f"{station}_{subdir}.pos"
                if station in targets[weighting] and not pos.exists():
                    problem = f"{weighting} PPPx produced no .pos"
                    break
        record(arm, doy, station, "failed" if problem else "done", problem or "")
    return n_targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*")
    parser.add_argument("--parallel", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    days = args.days or list(range(FIRST_DOY, LAST_DOY + 1))
    done = finished_entries()
    started = time.time()
    total = 0
    for doy in days:
        if not args.dry_run and shutil.disk_usage(REPO).free / 1e9 < MIN_FREE_GB:
            logger.error(f"stopping: less than {MIN_FREE_GB} GB free")
            sys.exit(3)
        for arm in ARMS:
            total += process(arm, doy, args.parallel, done, args.dry_run)
    action = "pending" if args.dry_run else "processed"
    logger.info(
        f"{total} (arm, station-day) entries {action} in "
        f"{(time.time() - started) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
