"""Per-epoch parsing and per-station-day/aggregate metrics from PPPx `.pos` files.

Ported from `positioning/positioning_eval/metrics.py` and the aggregation in
`src/analysis/positioning_summary.py`. This module covers the pure computation only:
parsing a solved `.pos` file into per-epoch ENU errors, reducing that to one metrics row
per station-day, and summarising station-days into the tables the paper reports. It does
**not** port the PPPx driver, product download, or RINEX handling - those still run PPPx
itself and stay where they are.

`plot_trends` and its helpers used to be duplicated verbatim between
`positioning/scripts/run_pipeline.py` and `positioning/scripts/recompute_metrics.py`. That
duplication is plotting code, out of scope here, and is not recreated - this module has no
plotting.

Aggregation convention (pinned by `test_summarise_matches_mean_of_station_days`): the
summary tables report the **mean of per-station-day RMSE values**, not an
observation-pooled RMSE recomputed across all epochs of all stations. This matches
`src/analysis/positioning_summary.py::summarise`, which is what produced the published
Table 5.

Known open issue, deliberately not resolved here: the elevation cutoff used upstream of
this module is inconsistent across the codebase - 7 degrees in
`positioning/positioning_eval/generate_ini.py` (the PPPx solve itself, via `elev_mask`) vs.
5.0 degrees in `positioning/scripts/generate_reference_corrections.py` (matching the STEC
database's own elevation cut) and `src/compare_stec_vtec_gim.py` (the Madrigal data
loader's `elevation_threshold`). None of the functions in this module apply an elevation
mask - a `.pos` file already reflects whatever mask solved it, and the metrics here are
computed over every epoch it contains - so no cutoff parameter was added to avoid
implying a resolution that doesn't exist. Anyone adding elevation-dependent computation
on top of this module must pick a value explicitly and cite which of the three it matches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# WGS84 ellipsoid parameters used by the ECEF <-> geodetic conversion below.
WGS84_SEMI_MAJOR_AXIS_M = 6378137.0000
WGS84_SEMI_MINOR_AXIS_M = 6356752.3142

# The 10 m station-day exclusion applied in Figure 12 and Table 5 (see
# `src/analysis/positioning_summary.py::OUTLIER_3D_RMS_M`). A station-day is kept when its
# 3D RMSE is <= this value; strictly greater excludes it.
OUTLIER_3D_RMS_M = 10.0

# Column layout of a PPPx `.pos` file (one header line, whitespace-separated):
#   mjd  sod  nsat  x  y  z  stdx  stdy  stdz  rck(m)  zhd  zwd  dzwd
# Only the columns the metrics need are read.
_POS_FILE_COLUMN_INDICES = (1, 2, 3, 4, 5, 9, 10, 11, 12)
_POS_FILE_COLUMN_NAMES = [
    "sod",
    "nsat",
    "x",
    "y",
    "z",
    "rck",
    "zhd",
    "zwd",
    "dzwd",
]


def xyz2blh(xyz: np.ndarray) -> np.ndarray:
    """Convert ECEF positions [m] to geodetic [lat(deg), lon(deg), height(m)]."""
    r2d = 180.0 / np.pi
    a = WGS84_SEMI_MAJOR_AXIS_M
    b = WGS84_SEMI_MINOR_AXIS_M

    x2 = xyz[:, 0] ** 2
    y2 = xyz[:, 1] ** 2
    z2 = xyz[:, 2] ** 2

    e = np.sqrt(1 - (b / a) ** 2)
    b2 = b * b
    e2 = e**2
    ep = e * (a / b)
    r = np.sqrt(x2 + y2)
    r2 = r * r
    big_e2 = a**2 - b**2
    f = 54 * b2 * z2
    g = r2 + (1 - e2) * z2 - e2 * big_e2
    c = ((e2 * e2) * f * r2) / (g**3)
    s = (1 + c + np.sqrt(c * c + 2 * c)) ** (1 / 3)
    p = f / (3 * (s + 1 / s + 1) ** 2 * g * g)
    q = np.sqrt(1 + 2 * e2 * e2 * p)
    ro = -(p * e2 * r) / (1 + q) + np.sqrt(
        (a * a / 2) * (1 + 1 / q) - (p * (1 - e2) * z2) / (q * (1 + q)) - p * r2 / 2
    )
    tmp = (r - e2 * ro) ** 2
    u = np.sqrt(tmp + z2)
    v = np.sqrt(tmp + (1 - e2) * z2)
    zo = (b2 * xyz[:, 2]) / (a * v)

    h = u * (1 - b2 / (a * v))
    lat = np.arctan((xyz[:, 2] + ep * ep * zo) / r) * r2d

    lon = np.arctan(xyz[:, 1] / xyz[:, 0]) * r2d
    negative_x_positive_y = np.logical_and(xyz[:, 0] < 0, xyz[:, 1] >= 0)
    lon[negative_x_positive_y] += 180
    negative_x_negative_y = np.logical_and(xyz[:, 0] < 0, xyz[:, 1] < 0)
    lon[negative_x_negative_y] -= 180

    return np.column_stack((lat, lon, h))


def xyz2enu(xyz: np.ndarray, org_xyz: np.ndarray) -> np.ndarray:
    """Convert ECEF positions [m] to local East-North-Up relative to `org_xyz`."""
    d2r = np.pi / 180.0
    n = xyz.shape[0]
    dif_xyz = xyz - np.tile(org_xyz, (n, 1))
    org_llh = xyz2blh(org_xyz.reshape(1, -1))
    phi = org_llh[0, 0] * d2r
    lam = org_llh[0, 1] * d2r
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    sin_lam = np.sin(lam)
    cos_lam = np.cos(lam)
    rotation = np.array(
        [
            [-sin_lam, cos_lam, 0],
            [-sin_phi * cos_lam, -sin_phi * sin_lam, cos_phi],
            [cos_phi * cos_lam, cos_phi * sin_lam, sin_phi],
        ]
    )
    return np.dot(rotation, dif_xyz.T).T


def load_sinex_coords(snx_file: Path | str) -> dict[str, list[float]]:
    """Parse an IGS SINEX file's SOLUTION/ESTIMATE block into {STATION: [x, y, z]} (m)."""
    snx_path = Path(snx_file)
    if not snx_path.exists():
        logger.warning(f"SINEX file not found: {snx_file}")
        return {}

    coords: dict[str, list[float]] = {}
    with open(snx_path, "r", errors="ignore") as handle:
        in_estimate = False
        for line in handle:
            if line.startswith("+SOLUTION/ESTIMATE"):
                in_estimate = True
                continue
            if line.startswith("-SOLUTION/ESTIMATE"):
                break
            if not in_estimate:
                continue

            # Example line:
            #    1 STAX  ZIMM  A    1  05:159:43200 m    01  2104332.8845 0.0011
            parts = line.split()
            if len(parts) < 9:
                continue

            entry_type = parts[1]
            if entry_type not in ("STAX", "STAY", "STAZ"):
                continue

            value_str = parts[8]
            if "_" in value_str:
                # Placeholder like '___ESTIMATED_VALUE___' - no coordinate to read.
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue

            station = parts[2].upper()
            coords.setdefault(station, [0.0, 0.0, 0.0])
            axis_index = {"STAX": 0, "STAY": 1, "STAZ": 2}[entry_type]
            coords[station][axis_index] = value

    return coords


def parse_pos_file(
    pos_file_path: Path | str, ref_pos: np.ndarray | None = None
) -> pd.DataFrame | None:
    """Parse a PPPx `.pos` file into a per-epoch frame with ENU and 2D/3D errors.

    Args:
        pos_file_path: Path to the `.pos` file.
        ref_pos: Optional 1x3 ECEF reference position [m]. If None, the day's own mean
            position is used as the reference - that mode reports internal repeatability,
            not true error against ground truth.

    Returns:
        Per-epoch DataFrame, or None if the file could not be read as a `.pos` file.
    """
    try:
        df = pd.read_csv(
            pos_file_path,
            sep=r"\s+",
            usecols=_POS_FILE_COLUMN_INDICES,
            skiprows=1,
            names=_POS_FILE_COLUMN_NAMES,
        )
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        logger.warning(f"could not parse {pos_file_path}: {exc}")
        return None

    df["hour"] = df["sod"] / 3600
    df["ztd"] = df["zhd"] + df["zwd"] + df["dzwd"]

    # Both references and the positions are forced to float. A .pos file with any
    # unparseable field leaves its column as object dtype, and the day-mean branch then
    # produces an object array whose np.sqrt raises "loop of ufunc does not support
    # argument 0 of type float" - a message that names the symptom and hides the cause.
    # The ground-truth branch never showed this because a list of floats already converts
    # cleanly, which is why it survived a gate that only exercised days with a SINEX.
    positions = df[["x", "y", "z"]].astype(float)

    if ref_pos is not None:
        xyz_ref = np.asarray(ref_pos, dtype=float).reshape(1, -1)
        df["ref_source"] = "ground_truth"
    else:
        xyz_ref = positions.mean().to_numpy(dtype=float).reshape(1, -1)
        df["ref_source"] = "mean"

    xyz_array = positions.to_numpy(dtype=float)
    enu = xyz2enu(xyz_array, xyz_ref)

    df["e"] = enu[:, 0]
    df["n"] = enu[:, 1]
    df["u"] = enu[:, 2]

    df["error_2d"] = np.sqrt(df["e"] ** 2 + df["n"] ** 2)
    df["error_3d"] = np.sqrt(df["e"] ** 2 + df["n"] ** 2 + df["u"] ** 2)

    return df


def compute_metrics(df: pd.DataFrame | None) -> dict | None:
    """Reduce a per-epoch frame (from `parse_pos_file`) to one station-day metrics row."""
    if df is None or len(df) == 0:
        return None

    return {
        "n_epochs": len(df),
        "mean_nsat": df["nsat"].mean(),
        "ref_source": df["ref_source"].iloc[0]
        if "ref_source" in df.columns
        else "unknown",
        "e_mean": df["e"].mean(),
        "e_std": df["e"].std(),
        "e_rms": np.sqrt((df["e"] ** 2).mean()),
        "n_mean": df["n"].mean(),
        "n_std": df["n"].std(),
        "n_rms": np.sqrt((df["n"] ** 2).mean()),
        "u_mean": df["u"].mean(),
        "u_std": df["u"].std(),
        "u_rms": np.sqrt((df["u"] ** 2).mean()),
        "error_2d_mean": df["error_2d"].mean(),
        "error_2d_std": df["error_2d"].std(),
        "error_2d_rms": np.sqrt((df["error_2d"] ** 2).mean()),
        "error_2d_95th": df["error_2d"].quantile(0.95),
        "error_3d_mean": df["error_3d"].mean(),
        "error_3d_std": df["error_3d"].std(),
        "error_3d_rms": np.sqrt((df["error_3d"] ** 2).mean()),
        "error_3d_95th": df["error_3d"].quantile(0.95),
    }


def aggregate_daily_metrics(
    results_dir: Path | str,
    year: int,
    doy: int,
    method_name: str,
    stations: list[str] | None = None,
    snx_file: Path | str | None = None,
) -> pd.DataFrame | None:
    """Compute one metrics row per station for all `.pos` files under `results_dir`.

    Args:
        results_dir: Directory holding `.pos` files, either directly or in per-station
            subdirectories (both layouts occur in `experiments/*/positioning/results/`).
        year, doy: Recorded on every output row, not read from the files.
        method_name: Recorded on every output row (e.g. "model" or "gim").
        stations: If given, only look for these stations' subdirectories.
        snx_file: SINEX file with ground-truth coordinates. If given, stations absent
            from it are skipped (no true error can be computed without a reference); if
            omitted, every station falls back to its own day-mean position, which is not
            a true error.

    Returns:
        DataFrame with one row per station, or None if no `.pos` files were found or none
        could be reduced to metrics.
    """
    results_path = Path(results_dir)

    gt_coords: dict[str, list[float]] = {}
    require_snx = False
    if snx_file:
        gt_coords = load_sinex_coords(snx_file)
        logger.info(f"loaded {len(gt_coords)} station coordinate(s) from {snx_file}")
        require_snx = True
    else:
        logger.warning(
            f"no SINEX file for {method_name} {year}/{doy:03d} - using day-mean reference"
        )

    if stations:
        pos_files = []
        for station in stations:
            station_dir = results_path / station
            if station_dir.exists():
                pos_files.extend(station_dir.glob(".*.pos"))
                pos_files.extend(station_dir.glob("*.pos"))
    else:
        pos_files = list(results_path.glob("**/.*.pos")) + list(
            results_path.glob("**/*.pos")
        )

    if not pos_files:
        logger.warning(f"no .pos files found for {method_name} in {results_dir}")
        return None

    all_metrics = []
    for pos_file in pos_files:
        if pos_file.parent.name != results_path.name:
            station = pos_file.parent.name
        else:
            # Hidden-file prefix and method suffix stripped, e.g. ".AMC4_model.pos" -> "AMC4".
            station = pos_file.stem.lstrip(".").split("_")[0]

        ref_pos = gt_coords.get(station.upper())
        if require_snx and not ref_pos:
            logger.info(f"skipping {station.upper()}: not in SINEX file")
            continue

        df = parse_pos_file(pos_file, ref_pos=ref_pos)
        metrics = compute_metrics(df)
        if metrics is None:
            continue

        metrics["station"] = station
        metrics["method"] = method_name
        metrics["year"] = year
        metrics["doy"] = doy
        all_metrics.append(metrics)

    if not all_metrics:
        return None

    metrics_df = pd.DataFrame(all_metrics)
    leading_cols = ["station", "method", "year", "doy"]
    metrics_df = metrics_df[
        leading_cols + [c for c in metrics_df.columns if c not in leading_cols]
    ]
    return metrics_df


def exclude_outlier_station_days(
    frame: pd.DataFrame, threshold: float = OUTLIER_3D_RMS_M
) -> pd.DataFrame:
    """Drop station-days with `error_3d_rms` above `threshold` (default: the paper's 10 m rule).

    A station-day exactly at the threshold is kept - the published tables use `<=`.
    """
    return frame[frame["error_3d_rms"] <= threshold].copy()


def summarise(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Reduce station-day metrics to the Table 5 columns, grouped by `by`.

    Each output value is the **mean (or median/quantile) across station-days**, not an
    observation-pooled statistic recomputed from epochs - the same convention as
    `src/analysis/positioning_summary.py`, which produced the published table. Callers
    that want the paper's numbers must apply `exclude_outlier_station_days` first, as
    `positioning_summary.py` does.
    """
    grouped = frame.groupby(by, observed=True)
    return pd.DataFrame(
        {
            "station_days": grouped.size(),
            "3D_mean_m": grouped["error_3d_rms"].mean(),
            "3D_median_m": grouped["error_3d_rms"].median(),
            "2D_mean_m": grouped["error_2d_rms"].mean(),
            "Up_mean_m": grouped["u_rms"].mean(),
            "3D_p95_m": grouped["error_3d_rms"].quantile(0.95),
        }
    ).round(4)
