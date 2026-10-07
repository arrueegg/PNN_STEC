"""Add independently computed rows for a constellation that is absent from a station-day.

CamaliotGnss writes a constellation for a station only when a receiver DCB exists, so some
station-days carry GPS or Galileo rows but not both, and PPPx can use only the satellites
that have a learned correction. This driver, for each (station, DOY):
  1. takes the existing source rows (STEC database, or `data/recovered_stec_db` for stations
     the database lacks) and appends `independent_geometry` rows for any constellation with
     no rows at all (never for DB-dropped arcs inside a constellation that is present);
  2. writes the result to a NEW scratch root, never touching either source tree;
  3. backs up the old correction CSVs, summaries and model/model_iono result directories,
     regenerates the corrections for the Direct STEC, Pretrained and VTEC arms, and reruns
     PPPx under iono and elev weighting;
  4. verifies the summaries only changed at the targeted (station, model) rows, and that
     the original constellation's correction rows are unchanged.

Usage::

    python positioning/geometry/fill_absent_constellation.py --cases ZECK:150 FAA1:150 BIK0:122
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

import independent_geometry as ig  # noqa: E402
from build_recovered_day import (  # noqa: E402
    _write_recovered_day_atomically,
    to_structured,
)
from recover_day import resolve_experiment  # noqa: E402
from validate_independent_geometry import products_dir  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
YEAR = 2024
DATABASE_ROOT = Path("/home/space/data/iono/STEC_DB_CASDCB")
RECOVERED_ROOT = REPO / "data" / "recovered_stec_db"
WORK_ROOT = REPO / "data" / "independent_geometry_work"
MERGED_ROOT = WORK_ROOT / "merged"
RINEX_DIR = WORK_ROOT / "rinex"
BACKUP_ROOT = REPO / "data" / f"constellation_fill_backup_{date.today():%Y%m%d}"
ARMS = ("STEC", "Pretrained_STEC", "VTEC")
WEIGHTINGS = ("iono", "elev")
SUMMARY_FILES = {"elev": "daily_summary.csv", "iono": "daily_summary_iono.csv"}
MODEL_METHOD = {"elev": "model", "iono": "model_iono"}
CONSTELLATIONS = "GE"


# ------------------------------------------------------------------------------ merge


def merge_absent_constellations(
    original: np.ndarray, added_geometry: pd.DataFrame
) -> tuple[np.ndarray, str]:
    """`original` rows (untouched, same order) followed by rows for absent constellations.

    `added_geometry` is `independent_geometry`'s output for the same station-day. Returns the
    merged structured array and the constellation letters that were appended ("" if none).
    """
    present = {sat[:1].decode() for sat in np.unique(original["sat"])}
    absent = "".join(c for c in CONSTELLATIONS if c not in present)
    if not absent or added_geometry.empty:
        return original, ""
    new_rows = added_geometry[added_geometry.sat.str[0].isin(list(absent))]
    if new_rows.empty:
        return original, ""
    appended = to_structured([new_rows.sort_values(["sod", "sat"])])
    merged = np.concatenate([original, appended])
    keys = pd.DataFrame({"sod": merged["sod"], "sat": merged["sat"]})
    if keys.duplicated().any():
        raise ValueError("merge produced duplicate (sod, sat) rows")
    return merged, "".join(sorted(set(new_rows.sat.str[0])))


def read_source_station_rows(root: Path, station: str, doy: int) -> np.ndarray:
    """The station's rows from a day file, original order, requiring all be test rows."""
    path = root / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5"
    with h5py.File(path, "r") as handle:
        group = handle[str(YEAR)][f"{doy:03d}"]
        dataset = group["all_data"]
        indices = np.flatnonzero(dataset.fields("station")[:] == station.encode())
        if len(indices) == 0:
            raise LookupError(f"{station} not in {path}")
        rows = dataset[indices[0] : indices[-1] + 1]
        rows = rows[rows["station"] == station.encode()]
        if not np.isin(indices, group["test_idx"][:]).all():
            raise ValueError(f"{station} {doy}: some source rows are not in test_idx")
    return rows


def find_source(station: str, doy: int) -> tuple[Path, np.ndarray]:
    """Where generate_stec_corrections originally got this station's rows from."""
    for root in (DATABASE_ROOT, RECOVERED_ROOT):
        try:
            return root, read_source_station_rows(root, station, doy)
        except (LookupError, FileNotFoundError, KeyError):
            continue
    raise LookupError(
        f"{station} DOY {doy} in neither the database nor the recovered tree"
    )


# ----------------------------------------------------------------------------- helpers


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
    before: dict[str, pd.DataFrame], after: dict[str, pd.DataFrame], stations: list[str]
) -> str | None:
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


def correction_csv(experiment: Path, doy: int, station: str) -> Path:
    return (
        experiment
        / "positioning"
        / "stec_corrections"
        / f"{YEAR}{doy:03d}"
        / f"{station}.csv"
    )


def compare_original_rows(old_csv: Path, new_csv: Path) -> dict:
    """Original-constellation rows must keep their (epoch, PRN) set; report value drift."""
    old, new = pd.read_csv(old_csv), pd.read_csv(new_csv)
    letters_old = set(old.PRN.str[0])
    kept = new[new.PRN.str[0].isin(letters_old)]
    merged = old.merge(
        kept,
        on=["second_of_day", "PRN"],
        suffixes=("_old", "_new"),
        how="outer",
        indicator=True,
    )
    both = merged[merged._merge == "both"]
    return {
        "old_rows": len(old),
        "new_rows": len(new),
        "constellations_old": "".join(sorted(letters_old)),
        "constellations_new": "".join(sorted(set(new.PRN.str[0]))),
        "original_rows_same_key_set": bool((merged._merge == "both").all()),
        "stec_rms_diff": float(np.sqrt(((both.stec_old - both.stec_new) ** 2).mean())),
        "stec_max_abs_diff": float((both.stec_old - both.stec_new).abs().max()),
        "unc_rms_diff": float(
            np.sqrt(((both.uncertainty_old - both.uncertainty_new) ** 2).mean())
        ),
    }


# ------------------------------------------------------------------------------- driver


def process_day(
    doy: int, stations: list[str], parallel: int, timings: dict
) -> list[dict]:
    date_text = (pd.Timestamp(f"{YEAR}-01-01") + pd.Timedelta(days=doy - 1)).strftime(
        "%Y-%m-%d"
    )
    tag = f"{YEAR}{doy:03d}"
    log_path = WORK_ROOT / f"fill_{tag}.log"
    experiments = {arm: resolve_experiment(arm, doy) for arm in ARMS}
    products = products_dir(doy)

    started = time.time()
    merged_parts, appended = [], {}
    for station in stations:
        source_root, original = find_source(station, doy)
        rinex = next(RINEX_DIR.glob(f"{station}*{tag}0000*"))
        geometry = ig.compute_station_geometry(
            station,
            YEAR,
            doy,
            rinex,
            next(products.glob("*ORB.SP3")),
            next(products.glob("IGS*SNX")),
        )
        merged, letters = merge_absent_constellations(original, geometry)
        appended[station] = (source_root.name, letters, len(merged) - len(original))
        merged_parts.append(merged)
    merged_day = np.concatenate(merged_parts)
    path = MERGED_ROOT / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5"
    _write_recovered_day_atomically(merged_day, path, YEAR, doy)
    timings[f"{doy}_geometry_merge_s"] = time.time() - started
    logger.info(f"DOY {doy}: appended {appended}")

    snapshots = {arm: summary_snapshot(exp, doy) for arm, exp in experiments.items()}
    day_backup = BACKUP_ROOT / tag
    for arm, exp in experiments.items():
        results = exp / "positioning" / "results" / tag
        for name in SUMMARY_FILES.values():
            backup_once(results / name, day_backup / arm / name)
        for station in stations:
            backup_once(
                correction_csv(exp, doy, station), day_backup / arm / f"{station}.csv"
            )
            for subdir in ("model", "model_iono"):
                backup_once(
                    results / subdir / station, day_backup / arm / f"{subdir}_{station}"
                )

    comparisons, problems = [], []
    for arm, exp in experiments.items():
        started = time.time()
        if run_logged(
            [
                "python",
                "positioning/scripts/generate_stec_corrections.py",
                "--experiment",
                exp.name,
                "--date",
                date_text,
                "--gnss_path",
                MERGED_ROOT,
            ],
            log_path,
        ):
            problems.append(f"{arm} corrections failed")
            continue
        timings[f"{doy}_{arm}_corrections_s"] = time.time() - started
        for station in stations:
            comparison = compare_original_rows(
                day_backup / arm / f"{station}.csv", correction_csv(exp, doy, station)
            )
            comparisons.append(
                {"arm": arm, "station": station, "doy": doy, **comparison}
            )
        results = exp / "positioning" / "results" / tag
        for (
            station
        ) in stations:  # An existing .pos counts as done to run_positioning_evaluation.
            for subdir in ("model", "model_iono"):
                shutil.rmtree(results / subdir / station, ignore_errors=True)
        started = time.time()
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
                    *stations,
                    "--weight_opt",
                    weighting,
                    "--parallel",
                    parallel,
                    "--rinex_dir",
                    RINEX_DIR,
                    "--no_cleanup",
                ],
                log_path,
            )
        timings[f"{doy}_{arm}_pppx_both_weightings_s"] = time.time() - started
        for pattern in ("**/.*.stat", "**/.*.log"):
            for stale in results.glob(pattern):
                stale.unlink(missing_ok=True)
        problem = verify_summaries(snapshots[arm], summary_snapshot(exp, doy), stations)
        if problem:
            problems.append(f"{arm}: {problem}")
        for station in stations:
            for subdir in ("model", "model_iono"):
                if not (
                    results / subdir / station / f"{station}_{subdir}.pos"
                ).exists():
                    problems.append(f"{arm} {station} {subdir}: no .pos")

    metrics = []
    for arm, exp in experiments.items():
        after = summary_snapshot(exp, doy)
        for station in stations:
            for weighting in WEIGHTINGS:
                for label, method in (
                    ("model", MODEL_METHOD[weighting]),
                    ("gim", "gim" if weighting == "elev" else "gim_iono"),
                ):
                    key = (station, method)
                    if (
                        key in snapshots[arm][weighting].index
                        and key in after[weighting].index
                    ):
                        old, new = (
                            snapshots[arm][weighting].loc[key],
                            after[weighting].loc[key],
                        )
                        metrics.append(
                            {
                                "station": station,
                                "doy": doy,
                                "arm": arm,
                                "weighting": weighting,
                                "row": label,
                                "nsat_before": old.mean_nsat,
                                "nsat_after": new.mean_nsat,
                                "rmse3d_before": old.error_3d_rms,
                                "rmse3d_after": new.error_3d_rms,
                            }
                        )
    out = WORK_ROOT / f"fill_{tag}_report.json"
    out.write_text(
        json.dumps(
            {
                "appended": appended,
                "comparisons": comparisons,
                "metrics": metrics,
                "problems": problems,
                "timings": timings,
            },
            indent=1,
            default=str,
        )
    )
    logger.info(f"DOY {doy}: problems={problems}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, help="STATION:DOY")
    parser.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    by_day: dict[int, list[str]] = {}
    for case in args.cases:
        station, doy = case.split(":")
        by_day.setdefault(int(doy), []).append(station)
    for doy, stations in sorted(by_day.items()):
        started = time.time()
        process_day(doy, stations, args.parallel, {})
        logger.info(f"DOY {doy} {stations}: full chain {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
