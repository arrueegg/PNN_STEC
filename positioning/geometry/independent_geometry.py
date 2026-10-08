"""Observation geometry computed without CamaliotGnss or any DCB.

CamaliotGnss only writes a constellation for a station when a CAS receiver DCB exists
for it, so many station-days have learned ionospheric corrections for only one of
GPS / Galileo. The model needs only geometry (elevation, azimuth, pierce point, solar-
magnetic coordinates), all of which follow from the precise orbit (SP3), the station
position and the set of (epoch, satellite) pairs the receiver tracked (RINEX 3). This
module computes exactly that and writes rows in the STEC-database HDF5 layout
(`build_recovered_day.CCL_DTYPE`) so `generate_stec_corrections.py` can consume them.

Conventions verified against the database (see `tests/positioning/test_independent_geometry.py`
and the validation scratch outputs): GPS-time seconds of day, geodetic station latitude
from ECEF, IPP on a thin shell at 450 km above a 6378.137 km sphere, solar-magnetic
coordinates from `camaliot_geometry.add_solar_magnetic`.

Usage::

    python positioning/geometry/independent_geometry.py --year 2024 --doy 133 \
        --station MRO1 --rinex <MRO1...MO.rnx> --sp3 <ORB.SP3> --snx <CRD.SNX> \
        --output_root data/independent_geometry_work/db_like
"""

from __future__ import annotations

import argparse
import logging
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from camaliot_geometry import add_solar_magnetic, ecef_to_geodetic  # noqa: E402

logger = logging.getLogger(__name__)

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0
EARTH_ROTATION_RATE_RAD_PER_S = 7.2921151467e-5
WGS84_SEMI_MAJOR_AXIS_M = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563
# The database was built with the WGS84 equatorial radius here (reproduces its IPPs to 5e-4 deg;
# 6371 km leaves a 0.01 deg systematic offset).
IONOSPHERIC_SHELL_RADIUS_KM = 6378.137
IONOSPHERIC_SHELL_HEIGHT_KM = 450.0
ELEVATION_CUTOFF_DEG = 5.0
INTERPOLATION_POINTS = 10
CRX2RNX = Path("/scratch2/arrueegg/WP2/GNSS_STEC_DB/GNSS_Camaliot_parallel/bin/crx2rnx")

# PPPx observation priority per constellation for the first (L1/E1) frequency, taken from
# `generate_ini.py` ("G = 1 2 PWCLSXYMN", "E = 1 5 BCIQX"). The second-frequency band is
# only used to flag whether a dual-frequency observation (what the database requires)
# was also present.
SINGLE_FREQUENCY_PRIORITY = {"G": "PWCLSXYMN", "E": "BCIQX"}
FIRST_BAND = {"G": "1", "E": "1"}
SECOND_BAND = {"G": "2", "E": "5"}


# ----------------------------------------------------------------------------- SP3 / orbits


@dataclass
class Sp3Orbits:
    """Node times (seconds since the first epoch's midnight) and ECEF metres per PRN."""

    node_times_s: np.ndarray
    positions_m: dict[str, np.ndarray]  # PRN -> (n_nodes, 3), NaN where unavailable


def read_sp3(path: Path, systems: str = "GE") -> Sp3Orbits:
    times: list[float] = []
    records: list[tuple[int, str, list[float]]] = []
    midnight: datetime | None = None
    with open(path, errors="ignore") as handle:
        for line in handle:
            if line.startswith("*"):
                fields = line[1:].split()
                year, month, day, hour, minute = (int(v) for v in fields[:5])
                epoch = datetime(year, month, day, hour, minute)
                midnight = midnight or datetime(year, month, day)
                times.append((epoch - midnight).total_seconds() + float(fields[5]))
            elif line.startswith("P") and line[1] in systems:
                x, y, z = (float(line[4 + 14 * i : 18 + 14 * i]) for i in range(3))
                # SP3 marks missing positions with 0.000000 or 999999.999999.
                if (
                    max(abs(x), abs(y), abs(z)) == 0.0
                    or max(abs(x), abs(y), abs(z)) > 9e5
                ):
                    continue
                records.append(
                    (
                        len(times) - 1,
                        line[1:4].replace(" ", "0"),
                        [x * 1e3, y * 1e3, z * 1e3],
                    )
                )
    positions: dict[str, np.ndarray] = {}
    for epoch_index, prn, xyz in records:
        nodes = positions.setdefault(prn, np.full((len(times), 3), np.nan))
        nodes[epoch_index] = xyz
    return Sp3Orbits(np.asarray(times), positions)


def interpolate_positions(
    node_times_s: np.ndarray,
    node_positions_m: np.ndarray,
    query_times_s: np.ndarray,
    num_points: int = INTERPOLATION_POINTS,
) -> np.ndarray:
    """Lagrange interpolation over the `num_points` nodes surrounding each query time.

    The window slides inward at the ends of the file, so queries near 00:00 or 24:00 use
    one-sided nodes (the SP3 holds the next midnight's node) rather than failing.
    Returns NaN where any node in the window is missing.
    """
    node_step = node_times_s[1] - node_times_s[0]
    num_nodes = len(node_times_s)
    first = np.floor((query_times_s - node_times_s[0]) / node_step).astype(int)
    first = np.clip(first - num_points // 2 + 1, 0, num_nodes - num_points)
    window = first[:, None] + np.arange(num_points)[None, :]  # (queries, points)
    window_times = node_times_s[window]
    weights = np.ones_like(window_times)
    for i in range(num_points):
        for j in range(num_points):
            if i != j:
                weights[:, i] *= (query_times_s - window_times[:, j]) / (
                    window_times[:, i] - window_times[:, j]
                )
    return np.einsum("qp,qpc->qc", weights, node_positions_m[window])


def satellite_positions_at_reception(
    orbits: Sp3Orbits,
    prn: str,
    receive_times_s: np.ndarray,
    receiver_xyz_m: np.ndarray,
    apply_light_time: bool = True,
) -> np.ndarray:
    """Satellite ECEF at transmission, rotated into the Earth-fixed frame of reception."""
    nodes = orbits.positions_m[prn]
    if not apply_light_time:
        return interpolate_positions(orbits.node_times_s, nodes, receive_times_s)
    travel_time_s = np.full(len(receive_times_s), 0.075)
    for _ in range(3):
        satellite = interpolate_positions(
            orbits.node_times_s, nodes, receive_times_s - travel_time_s
        )
        angle = EARTH_ROTATION_RATE_RAD_PER_S * travel_time_s
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotated = np.column_stack(
            (
                cos_a * satellite[:, 0] + sin_a * satellite[:, 1],
                -sin_a * satellite[:, 0] + cos_a * satellite[:, 1],
                satellite[:, 2],
            )
        )
        travel_time_s = (
            np.linalg.norm(rotated - receiver_xyz_m, axis=1) / SPEED_OF_LIGHT_M_PER_S
        )
    return rotated


# ------------------------------------------------------------------------- station position


def read_sinex_station_xyz(path: Path, station: str) -> np.ndarray:
    """ECEF metres of `station` from the SOLUTION/ESTIMATE block (not SOLUTION/APRIORI)."""
    coordinates: dict[str, float] = {}
    inside = False
    with open(path, errors="ignore") as handle:
        for line in handle:
            if line.startswith("+SOLUTION/ESTIMATE"):
                inside = True
            elif line.startswith("-SOLUTION/ESTIMATE"):
                break
            elif inside:
                parts = line.split()
                if (
                    len(parts) >= 9
                    and parts[1] in ("STAX", "STAY", "STAZ")
                    and parts[2].upper() == station.upper()
                ):
                    coordinates[parts[1]] = float(parts[8])
    if len(coordinates) != 3:
        raise ValueError(f"{station} not found in {path.name}")
    return np.array([coordinates["STAX"], coordinates["STAY"], coordinates["STAZ"]])


def ecef_to_geodetic_arrays(xyz_m: np.ndarray) -> tuple[float, float, float]:
    """(longitude deg, geodetic latitude deg, ellipsoidal height m); signed latitude."""
    lon, lat = ecef_to_geodetic(*xyz_m)
    eccentricity_squared = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
    sin_lat = math.sin(math.radians(lat))
    prime_vertical = WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
        1.0 - eccentricity_squared * sin_lat**2
    )
    height = (
        math.hypot(xyz_m[0], xyz_m[1]) / math.cos(math.radians(lat)) - prime_vertical
    )
    return lon, lat, height


# ------------------------------------------------------------------------------- RINEX 3


@dataclass
class RinexObservations:
    header_xyz_m: np.ndarray
    table: pd.DataFrame  # sod, sat, c1_attribute, has_dual


def _decompress_if_needed(path: Path, work_dir: Path) -> Path:
    if path.suffix.lower() == ".crx":
        target = work_dir / (path.stem + ".rnx")
        if not target.exists():
            with open(target, "wb") as out:
                subprocess.run(
                    [str(CRX2RNX), "-"], stdin=open(path, "rb"), stdout=out, check=True
                )
        return target
    return path


def read_rinex3_observations(path: Path, systems: str = "GE") -> RinexObservations:
    """Per (epoch, satellite): which first-band code attribute PPPx would pick, and
    whether a second-band pair was also tracked. GPS time seconds of day."""
    header_xyz = np.full(3, np.nan)
    obs_types: dict[str, list[str]] = {}
    current_system = None
    records = []
    with open(path, errors="ignore") as handle:
        for line in handle:
            label = line[60:].strip()
            if label == "APPROX POSITION XYZ":
                header_xyz = np.array([float(v) for v in line[:60].split()])
            elif label == "SYS / # / OBS TYPES":
                if line[0].strip():
                    current_system = line[0]
                    obs_types[current_system] = []
                obs_types[current_system].extend(line[7:60].split())
            elif label == "END OF HEADER":
                break

        priorities = {}
        for system in systems:
            codes = obs_types.get(system, [])
            band1, band2 = FIRST_BAND[system], SECOND_BAND[system]
            first_band_pairs = []  # (attribute, code_index, phase_index) in PPPx priority order
            for attribute in SINGLE_FREQUENCY_PRIORITY[system]:
                code_name, phase_name = f"C{band1}{attribute}", f"L{band1}{attribute}"
                if code_name in codes and phase_name in codes:
                    first_band_pairs.append(
                        (attribute, codes.index(code_name), codes.index(phase_name))
                    )
            second_band_pairs = [
                (codes.index(name), codes.index("L" + name[1:]))
                for name in codes
                if name.startswith("C" + band2) and "L" + name[1:] in codes
            ]
            priorities[system] = (first_band_pairs, second_band_pairs)

        seconds_of_day = None
        epoch_satellites_left = 0
        for line in handle:
            if line.startswith(">"):
                # Fixed-width RINEX 3 epoch record; event records (flag > 1) may have a blank time.
                event_flag, count = int(line[31:32] or 0), int(line[32:35])
                if event_flag <= 1:
                    seconds_of_day = (
                        int(line[13:15]) * 3600
                        + int(line[16:18]) * 60
                        + float(line[19:30])
                    )
                epoch_satellites_left = count
                skip_lines = event_flag > 1
                continue
            if epoch_satellites_left <= 0:
                continue
            epoch_satellites_left -= 1
            if skip_lines:
                continue
            system = line[0]
            if system not in priorities:
                continue

            def value(index: int) -> bool:
                start = 3 + 16 * index
                return bool(line[start : start + 14].strip())

            first_pairs, second_pairs = priorities[system]
            attribute = next(
                (a for a, c, p in first_pairs if value(c) and value(p)), None
            )
            if attribute is None:
                continue
            has_dual = any(value(c) and value(p) for c, p in second_pairs)
            records.append(
                (seconds_of_day, line[:3].replace(" ", "0"), attribute, has_dual)
            )
    table = pd.DataFrame(records, columns=["sod", "sat", "c1_attribute", "has_dual"])
    return RinexObservations(header_xyz, table)


# ----------------------------------------------------------------------------- geometry


def elevation_azimuth_deg(
    receiver_xyz_m: np.ndarray,
    satellite_xyz_m: np.ndarray,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevation and azimuth (0 = north, clockwise) in the receiver's local ENU frame."""
    lat, lon = math.radians(latitude_deg), math.radians(longitude_deg)
    delta = satellite_xyz_m - receiver_xyz_m
    east = -math.sin(lon) * delta[:, 0] + math.cos(lon) * delta[:, 1]
    north = (
        -math.sin(lat) * math.cos(lon) * delta[:, 0]
        - math.sin(lat) * math.sin(lon) * delta[:, 1]
        + math.cos(lat) * delta[:, 2]
    )
    up = (
        math.cos(lat) * math.cos(lon) * delta[:, 0]
        + math.cos(lat) * math.sin(lon) * delta[:, 1]
        + math.sin(lat) * delta[:, 2]
    )
    elevation = np.degrees(np.arctan2(up, np.hypot(east, north)))
    azimuth = np.degrees(np.arctan2(east, north)) % 360.0
    return elevation, azimuth


def ionospheric_pierce_point_deg(
    latitude_deg: float,
    longitude_deg: float,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin-shell IPP (450 km above a 6378.137 km sphere, the database convention); longitude in [-180, 180).

    Uses atan2 for the longitude offset, which equals the arcsin form of
    `stec.data.coordinate_transforms.calculate_ipp_coordinates` wherever that is defined
    but stays correct for long arcs at high latitude.
    """
    lat = math.radians(latitude_deg)
    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    shell_ratio = IONOSPHERIC_SHELL_RADIUS_KM / (
        IONOSPHERIC_SHELL_RADIUS_KM + IONOSPHERIC_SHELL_HEIGHT_KM
    )
    central_angle = np.pi / 2 - el - np.arcsin(shell_ratio * np.cos(el))
    ipp_lat = np.arcsin(
        math.sin(lat) * np.cos(central_angle)
        + math.cos(lat) * np.sin(central_angle) * np.cos(az)
    )
    delta_lon = np.arctan2(
        np.sin(az) * np.sin(central_angle) * math.cos(lat),
        np.cos(central_angle) - math.sin(lat) * np.sin(ipp_lat),
    )
    ipp_lon = (longitude_deg + np.degrees(delta_lon) + 180.0) % 360.0 - 180.0
    return np.degrees(ipp_lat), ipp_lon


def compute_station_geometry(
    station: str,
    year: int,
    doy: int,
    rinex_path: Path,
    sp3_path: Path,
    sinex_path: Path | None,
    systems: str = "GE",
    apply_light_time: bool = True,
    with_solar_magnetic: bool = True,
    require_dual_frequency: bool = False,
    orbits: Sp3Orbits | None = None,
) -> pd.DataFrame:
    """Geometry rows (database columns, NaN for STEC-derived ones) for one station-day."""
    observations = read_rinex3_observations(rinex_path, systems)
    if sinex_path is not None:
        receiver_xyz = read_sinex_station_xyz(sinex_path, station)
    else:
        receiver_xyz = observations.header_xyz_m
    longitude, latitude, _ = ecef_to_geodetic_arrays(receiver_xyz)
    orbits = orbits or read_sp3(sp3_path, systems)

    table = observations.table
    if require_dual_frequency:
        table = table[table.has_dual]
    frames = []
    for prn, rows in table.groupby("sat"):
        if prn not in orbits.positions_m:
            continue
        times = rows["sod"].to_numpy(float)
        satellite = satellite_positions_at_reception(
            orbits, prn, times, receiver_xyz, apply_light_time
        )
        elevation, azimuth = elevation_azimuth_deg(
            receiver_xyz, satellite, latitude, longitude
        )
        ipp_lat, ipp_lon = ionospheric_pierce_point_deg(
            latitude, longitude, azimuth, elevation
        )
        frames.append(
            pd.DataFrame(
                {
                    "station": station.upper(),
                    "sat": prn,
                    "sod": times,
                    "satele": elevation,
                    "satazi": azimuth,
                    "lat_ipp": ipp_lat,
                    "lon_ipp": ipp_lon,
                    "has_dual": rows["has_dual"].to_numpy(),
                    "c1_attribute": rows["c1_attribute"].to_numpy(),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    geometry = pd.concat(frames, ignore_index=True)
    geometry = geometry[
        (geometry.satele >= ELEVATION_CUTOFF_DEG) & np.isfinite(geometry.satele)
    ].reset_index(drop=True)
    geometry["lat_sta"] = latitude
    geometry["lon_sta"] = longitude
    if with_solar_magnetic:
        geometry = add_solar_magnetic(geometry, year, doy)
    geometry["slipc"] = np.nan
    geometry["gfphase"] = np.nan
    return geometry.sort_values(["sod", "sat"]).reset_index(drop=True)


def write_database_layout(
    geometry: pd.DataFrame, output_root: Path, year: int, doy: int
) -> Path:
    """Write `geometry` in the STEC-database HDF5 layout, replacing that day's file."""
    from build_recovered_day import _write_recovered_day_atomically, to_structured

    out_path = output_root / str(year) / f"{doy:03d}" / f"ccl_{year}{doy:03d}_30_5.h5"
    _write_recovered_day_atomically(to_structured([geometry]), out_path, year, doy)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument("--station", required=True)
    parser.add_argument("--rinex", type=Path, required=True)
    parser.add_argument("--sp3", type=Path, required=True)
    parser.add_argument("--snx", type=Path)
    parser.add_argument("--systems", default="GE")
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    geometry = compute_station_geometry(
        args.station, args.year, args.doy, args.rinex, args.sp3, args.snx, args.systems
    )
    out_path = write_database_layout(geometry, args.output_root, args.year, args.doy)
    logger.info(f"wrote {len(geometry):,} rows for {args.station} to {out_path}")


if __name__ == "__main__":
    main()
