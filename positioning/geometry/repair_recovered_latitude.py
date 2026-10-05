"""Repair sign-flipped southern station latitudes in the recovered STEC-database files.

`camaliot_geometry.parse_site` used to read lat_sta from the unsigned +SITE/ID latitude, so
every southern station in `data/recovered_stec_db` carries a positive lat_sta and wrong
sm_lat_sta/sm_lon_sta (IPP coordinates were never affected). This rewrites those three
columns, per station, from the production database's own lat_sta for the same station.

A station is repaired when its stored lat_sta differs from the correct one by more than
LATITUDE_TOLERANCE_DEGREES, so a second run finds nothing to do. Each modified file is first
copied to the same relative path under the backup root; an existing backup is never
overwritten. Stations with no resolvable correct latitude are reported, not guessed.

Usage::

    python positioning/geometry/repair_recovered_latitude.py --dry-run
    python positioning/geometry/repair_recovered_latitude.py --only 133:MRO1
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_recovered_day import _write_recovered_day_atomically  # noqa: E402
from camaliot_geometry import add_solar_magnetic, parse_site  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
RECOVERED_ROOT = REPO / "data" / "recovered_stec_db"
PRODUCTION_DB_ROOT = Path("/home/space/data/iono/STEC_DB_CASDCB")
RECOVERY_WORK_ROOT = REPO / "data" / "recovery_work"
LATITUDE_TOLERANCE_DEGREES = 0.01
YEAR = 2024
FIRST_DOY, LAST_DOY = 122, 366


def _station_column(path: Path, year: int, doy: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        all_data = handle[str(year)][f"{doy:03d}"]["all_data"]
        return all_data["station"][:], all_data["lat_sta"][:]


def lookup_production_latitudes(stations: set[str]) -> dict[str, float]:
    """lat_sta from the production database, scanning days until every station is found."""
    found: dict[str, float] = {}
    for doy in range(FIRST_DOY, LAST_DOY + 1):
        pending = stations - found.keys()
        if not pending:
            break
        path = (
            PRODUCTION_DB_ROOT
            / str(YEAR)
            / f"{doy:03d}"
            / f"ccl_{YEAR}{doy:03d}_30_5.h5"
        )
        if not path.exists():
            continue
        names, latitudes = _station_column(path, YEAR, doy)
        for name in pending:
            hits = np.flatnonzero(names == name.encode("ascii"))
            if len(hits):
                found[name] = float(latitudes[hits[0]])
    return found


def latitude_from_recovery_work(station: str) -> float | None:
    """Fallback: the (fixed) parser applied to any leftover CamaliotGnss output."""
    for ion_file in sorted(RECOVERY_WORK_ROOT.glob(f"*/camaliot/{station}/*ION.ION")):
        site = parse_site(ion_file)
        if site:
            return site["lat_sta"]
    return None


def recovered_files() -> list[tuple[int, Path]]:
    return [
        (int(path.parent.name), path)
        for path in sorted(RECOVERED_ROOT.glob(f"{YEAR}/*/ccl_*_30_5.h5"))
    ]


def repair_station_rows(
    data: np.ndarray, mask: np.ndarray, doy: int, latitude: float
) -> None:
    """Set lat_sta and recompute sm_lat_sta/sm_lon_sta for the masked rows, in place."""
    rows = data[mask]
    frame = pd.DataFrame(
        {
            "sod": rows["sod"].astype(float),
            "lat_ipp": rows["lat_ipp"].astype(float),
            "lon_ipp": rows["lon_ipp"].astype(float),
            "lat_sta": np.full(len(rows), latitude),
            "lon_sta": rows["lon_sta"].astype(float),
        }
    )
    frame = add_solar_magnetic(frame, YEAR, doy)
    rows["lat_sta"] = latitude
    rows["sm_lat_sta"] = frame["sm_lat_sta"].to_numpy()
    rows["sm_lon_sta"] = frame["sm_lon_sta"].to_numpy()
    data[mask] = rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="DOY:STATION",
        help="restrict to this station-day (repeatable); used for the pilot",
    )
    parser.add_argument(
        "--backup-root",
        type=Path,
        default=REPO / "data" / f"recovered_stec_db.bak_{date.today():%Y%m%d}",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    only = {
        (int(d), s.upper()) for d, s in (item.split(":") for item in args.only)
    } or None
    files = [
        (doy, path)
        for doy, path in recovered_files()
        if only is None or doy in {d for d, _ in only}
    ]

    stored: dict[tuple[int, str], float] = {}
    for doy, path in files:
        names, latitudes = _station_column(path, YEAR, doy)
        for name in np.unique(names):
            stored[(doy, name.decode())] = float(
                latitudes[np.flatnonzero(names == name)[0]]
            )
    stations = {
        station
        for _, station in stored
        if only is None or station in {s for _, s in only}
    }
    logger.info(
        f"{len(files)} recovered files, {len(stations)} distinct stations; looking up latitudes"
    )

    correct = lookup_production_latitudes(stations)
    for station in sorted(stations - correct.keys()):
        fallback = latitude_from_recovery_work(station)
        if fallback is not None:
            correct[station] = fallback
    unresolved = sorted(stations - correct.keys())
    if unresolved:
        logger.warning(f"no correct latitude found for: {', '.join(unresolved)}")

    to_repair = [
        (doy, station)
        for (doy, station), value in sorted(stored.items())
        if station in correct
        and abs(value - correct[station]) > LATITUDE_TOLERANCE_DEGREES
        and (only is None or (doy, station) in only)
    ]
    repaired_stations = sorted({station for _, station in to_repair})
    logger.info(
        f"{len(to_repair)} station-days to repair across "
        f"{len({doy for doy, _ in to_repair})} files, {len(repaired_stations)} stations: "
        f"{', '.join(repaired_stations)}"
    )

    total_rows = 0
    for doy, path in files:
        day_stations = [station for d, station in to_repair if d == doy]
        if not day_stations:
            continue
        with h5py.File(path, "r") as handle:
            data = handle[str(YEAR)][f"{doy:03d}"]["all_data"][:]
        day_rows = 0
        for station in day_stations:
            mask = data["station"] == station.encode("ascii")
            day_rows += int(mask.sum())
            if not args.dry_run:
                repair_station_rows(data, mask, doy, correct[station])
        total_rows += day_rows
        if args.dry_run:
            continue
        backup = args.backup_root / path.relative_to(RECOVERED_ROOT)
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        _write_recovered_day_atomically(data, path, YEAR, doy)
        logger.info(
            f"DOY {doy}: repaired {', '.join(day_stations)} ({day_rows:,} rows), backup {backup}"
        )

    verb = "would change" if args.dry_run else "changed"
    logger.info(f"{verb} {total_rows:,} rows in {len({d for d, _ in to_repair})} files")


if __name__ == "__main__":
    main()
