"""Re-run the VTEC + Mapping positioning arm after the SLM mapping decision (2026-10-05).

The VTEC baseline's slant corrections were mapped with MSLM; they must use SLM (450 km), see
`stec.inference.run_baselines.VTEC_MAPPING_TYPE`. Per day, for every station already in the
VTEC arm's positioning set (it has a corrections CSV), this driver
  1. backs up the old corrections CSVs, daily summaries and model/model_iono result
     directories (never overwriting an existing backup);
  2. regenerates the corrections with the 10-member canonical VTEC ensemble, once from the
     original STEC database and once from `data/recovered_stec_db` (stations missing from the
     original database), and removes any CSV that did not exist before (the generator writes
     every test station it finds, not only the positioning set);
  3. deletes the old model/model_iono result directories, because run_positioning_evaluation
     skips a station whose .pos exists, and reruns PPPx under iono and elev weighting;
  4. drops .stat/.log and the day's RINEX (KEEP_DIAGNOSTICS=1 / KEEP_RINEX=1 keep them);
  5. verifies the daily summaries never shrink and only the VTEC arm's model rows of the
     targeted stations changed.
Progress goes to a JSONL manifest, so a restart skips finished station-days. Only the VTEC
experiments are touched.

Usage::

    python scripts/rerun_vtec_slm_positioning.py [--days 132 133] [--parallel 6] [--dry-run]
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
RECOVERED_ROOT = REPO / "data" / "recovered_stec_db"
BACKUP_ROOT = REPO / "data" / f"vtec_slm_backup_{date.today():%Y%m%d}"
WORK_ROOT = REPO / "data" / "vtec_slm_work"
MANIFEST = REPO / "logs" / "vtec_slm_rerun_manifest.jsonl"
WEIGHTINGS = ("iono", "elev")
SUMMARY_FILES = {"elev": "daily_summary.csv", "iono": "daily_summary_iono.csv"}
MODEL_METHOD = {"elev": "model", "iono": "model_iono"}
MIN_FREE_GB = 40


def read_manifest() -> dict[tuple[int, str], dict]:
    latest: dict[tuple[int, str], dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            entry = json.loads(line)
            latest[(entry["doy"], entry["station"])] = entry
    return latest


def record(doy: int, station: str, status: str, detail: str = "") -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "a") as handle:
        entry = {
            "doy": doy,
            "station": station,
            "status": status,
            "detail": detail,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        handle.write(json.dumps(entry) + "\n")
    logger.info(f"DOY {doy} {station}: {status} {detail}")


def run_logged(command: list, log_path: Path) -> int:
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
            return f"{weighting} summary shrank or vanished"
        allowed = {(s, MODEL_METHOD[weighting]) for s in stations}
        for key in new.index:
            if key in allowed:
                continue
            if key not in old.index or not old.loc[key].equals(new.loc[key]):
                return f"{weighting} unexpected change at {key}"
    return None


def csv_names(directory: Path) -> set[str]:
    return {p.stem for p in directory.glob("*.csv")} if directory.exists() else set()


def generate_corrections(
    experiment: Path, doy: int, date_text: str, log_path: Path
) -> bool:
    corrections_dir = (
        experiment / "positioning" / "stec_corrections" / f"{YEAR}{doy:03d}"
    )
    existing = csv_names(corrections_dir)
    sources: list[list[str]] = [[]]
    if (RECOVERED_ROOT / str(YEAR) / f"{doy:03d}").exists():
        sources.append(["--gnss_path", str(RECOVERED_ROOT)])
    for extra in sources:
        command = [
            "python",
            "positioning/scripts/generate_stec_corrections.py",
            "--experiment",
            experiment.name,
            "--date",
            date_text,
            *extra,
        ]
        if run_logged(command, log_path) != 0:
            return False
    for name in csv_names(corrections_dir) - existing:
        (corrections_dir / f"{name}.csv").unlink()
    return True


def process_day(doy: int, parallel: int) -> None:
    date_text = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)).strftime(
        "%Y-%m-%d"
    )
    experiment = resolve_experiment("VTEC", doy)
    if experiment is None:
        record(doy, "*", "failed", "no usable VTEC experiment checkpoint")
        return
    tag = f"{YEAR}{doy:03d}"
    results = experiment / "positioning" / "results" / tag
    corrections_dir = experiment / "positioning" / "stec_corrections" / tag
    finished = {
        s for (d, s), e in read_manifest().items() if d == doy and e["status"] == "done"
    }
    targets = sorted(csv_names(corrections_dir) - finished)
    if not targets:
        return

    work = WORK_ROOT / tag
    work.mkdir(parents=True, exist_ok=True)
    log_path = work / "run.log"
    before = summary_snapshot(experiment, doy)
    day_backup = BACKUP_ROOT / tag
    for name in SUMMARY_FILES.values():
        backup_once(results / name, day_backup / name)
    for station in targets:
        backup_once(corrections_dir / f"{station}.csv", day_backup / f"{station}.csv")
        for subdir in ("model", "model_iono"):
            backup_once(results / subdir / station, day_backup / f"{subdir}_{station}")

    failure_all = None
    if not generate_corrections(experiment, doy, date_text, log_path):
        failure_all = "corrections generation failed"
    else:
        for station in targets:
            for subdir in ("model", "model_iono"):
                # An existing .pos counts as done to run_positioning_evaluation.
                shutil.rmtree(results / subdir / station, ignore_errors=True)
        rinex_dir = work / "rinex"
        rinex_dir.mkdir(exist_ok=True)
        for weighting in WEIGHTINGS:
            run_logged(
                [
                    "python",
                    "positioning/positioning_eval/run_positioning_evaluation.py",
                    "--experiment",
                    experiment.relative_to(REPO),
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
        if os.environ.get("KEEP_RINEX") != "1":
            shutil.rmtree(rinex_dir, ignore_errors=True)
        if os.environ.get("KEEP_DIAGNOSTICS") != "1":
            for pattern in ("**/*.stat", "**/*.log"):
                for stale in results.glob(pattern):
                    stale.unlink(missing_ok=True)
        failure_all = verify_summaries(experiment, doy, before, targets)

    for station in targets:
        problem = failure_all
        if problem is None:
            for subdir, weighting in (("model", "elev"), ("model_iono", "iono")):
                pos = results / subdir / station / f"{station}_{subdir}.pos"
                if not pos.exists():
                    problem = f"{weighting} PPPx produced no .pos"
                    break
        if problem:
            record(doy, station, "failed", problem)
        else:
            record(doy, station, "done")


def free_gb() -> float:
    return shutil.disk_usage(REPO).free / 1e9


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
    done = {key for key, e in read_manifest().items() if e["status"] == "done"}
    if args.dry_run:
        total = 0
        for doy in days:
            experiment = resolve_experiment("VTEC", doy)
            if experiment is None:
                print(f"DOY {doy}: no VTEC experiment")
                continue
            corrections = (
                experiment / "positioning" / "stec_corrections" / f"{YEAR}{doy:03d}"
            )
            pending = [s for s in csv_names(corrections) if (doy, s) not in done]
            total += len(pending)
        print(f"{total} VTEC station-days pending over {len(days)} day(s)")
        return

    started = time.time()
    for doy in days:
        if free_gb() < MIN_FREE_GB:
            logger.error(f"stopping: {free_gb():.0f} GB free, floor is {MIN_FREE_GB}")
            sys.exit(3)
        process_day(doy, args.parallel)
    logger.info(f"finished in {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
