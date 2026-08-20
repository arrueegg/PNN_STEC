"""Rebuild one day of STEC-database-format observations for stations the database dropped.

`STEC_DB_CASDCB` only contains stations that appear in the CAS DCB file: the production
script grows its station lists by grepping the daily BSX, so a station without a DCB entry
is dropped before processing and no STEC correction can ever be generated for it. That
accounts for 2,311 of the 2,821 station-days the IGS GIM solves but the ML methods do not
(see `src/analysis/positioning_coverage.py`), which is a systematic exclusion correlated
with station location - exactly the kind of gap that biases a positioning comparison.

The model does not need the calibrated STEC, only the geometry: elevation, azimuth, pierce
point and station position. Those come out of the same CamaliotGnss binary the database was
built with, and are unaffected by the missing DCB - a DCB-less station still yields identical
geometry, only its `stec` is uncalibrated (verified: DCBR=0, negative STEC, geometry exact to
4e-6 degrees). So this runs the production binary, keeps the geometry, and writes a file in
the database's own layout.

`stec` is written as NaN rather than the uncalibrated value, because it is not ground truth
and must never be mistaken for it. Nothing downstream reads it: the model predicts STEC, and
`generate_stec_corrections.py` uses the target only as an unused label.

Point `GNSS_data_path` at the output root to generate corrections for these stations.

Usage::

    python positioning/geometry/build_recovered_day.py --year 2024 --doy 323 \
        --stations ALGO,BIK0 --output_root data/recovered_stec_db
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import subprocess

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "positioning_eval"))

from camaliot_geometry import add_solar_magnetic, run_station  # noqa: E402

logger = logging.getLogger(__name__)

CCL_DTYPE = np.dtype([
    ("station", "S4"), ("sat", "S3"), ("stec", "f4"), ("vtec", "f4"),
    ("vtec_stddev", "f4"), ("satres", "f4"), ("satele", "f4"), ("satazi", "f4"),
    ("dcbs", "f4"), ("dcbr", "f4"), ("lon_ipp", "f4"), ("lat_ipp", "f4"),
    ("sm_lat_ipp", "f4"), ("sm_lon_ipp", "f4"), ("sod", "f4"), ("lat_sta", "f4"),
    ("lon_sta", "f4"), ("sm_lat_sta", "f4"), ("sm_lon_sta", "f4"),
    ("slipc", "f4"), ("gfphase", "f4"),
])
# Fields the geometry cannot supply; the model reads none of them.
UNAVAILABLE = ["stec", "vtec", "vtec_stddev", "satres", "dcbs", "dcbr"]


def to_structured(frames: list[pd.DataFrame]) -> np.ndarray:
    combined = pd.concat(frames, ignore_index=True)
    out = np.empty(len(combined), dtype=CCL_DTYPE)
    out["station"] = combined["station"].str.upper().str.encode("ascii")
    out["sat"] = combined["sat"].astype(str).str.encode("ascii")
    for field in CCL_DTYPE.names:
        if field in ("station", "sat"):
            continue
        out[field] = np.nan if field in UNAVAILABLE else combined[field].to_numpy("f4")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, required=True)
    parser.add_argument("--stations", required=True, help="comma-separated 4-character IDs")
    parser.add_argument("--rinex_dir", type=Path, required=True)
    parser.add_argument("--nav", type=Path, required=True)
    parser.add_argument("--bsx", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=Path("data/recovered_stec_db"))
    parser.add_argument("--parallel", type=int, default=6,
                        help="stations processed concurrently; each is a subprocess")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    args.workdir.mkdir(parents=True, exist_ok=True)

    frames = []

    def skip(station: str, reason: str) -> None:
        logger.warning(f"skipped {station}: {reason}")

    def process(station: str) -> pd.DataFrame | None:
        """Geometry for one station. Each call is an independent CamaliotGnss subprocess."""
        matches = sorted(args.rinex_dir.glob(f"{station}*.rnx")) or \
                  sorted(args.rinex_dir.glob(f"{station.lower()}*.rnx"))
        if not matches:
            skip(station, "no RINEX")
            return None
        try:
            geometry = run_station(matches[0], args.nav, args.bsx, args.workdir / station)
        except (RuntimeError, ValueError, FileNotFoundError, subprocess.SubprocessError) as exc:
            skip(station, f"{type(exc).__name__}: {exc}")
            return None
        if geometry is None or geometry.empty:
            skip(station, "no observations")
            return None
        logger.info(f"{station}: {len(geometry):,} observations")
        return geometry

    # Only the CamaliotGnss subprocesses run concurrently. The solar-magnetic conversion
    # stays on this thread: spacepy's Coords/Ticktock are not documented as thread-safe.
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        raw = [f for f in pool.map(process, stations) if f is not None]
    frames = [add_solar_magnetic(f, args.year, args.doy) for f in raw]

    if not frames:
        raise SystemExit(f"no stations recovered for {args.year} DOY {args.doy}")

    data = to_structured(frames)
    out_dir = args.output_root / str(args.year) / f"{args.doy:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ccl_{args.year}{args.doy:03d}_30_5.h5"
    with h5py.File(out_path, "w") as handle:
        group = handle.require_group(str(args.year)).require_group(f"{args.doy:03d}")
        group.create_dataset("all_data", data=data, compression="gzip")
        # Every recovered observation is a test observation: these stations were never
        # available to train on, precisely because the database excluded them.
        group.create_dataset("test_idx", data=np.arange(len(data), dtype="i8"))

    logger.info(f"💾 {out_path}: {len(data):,} observations, {len(frames)} station(s)")


if __name__ == "__main__":
    main()
