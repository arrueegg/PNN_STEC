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
import os
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

CCL_DTYPE = np.dtype(
    [
        ("station", "S4"),
        ("sat", "S3"),
        ("stec", "f4"),
        ("vtec", "f4"),
        ("vtec_stddev", "f4"),
        ("satres", "f4"),
        ("satele", "f4"),
        ("satazi", "f4"),
        ("dcbs", "f4"),
        ("dcbr", "f4"),
        ("lon_ipp", "f4"),
        ("lat_ipp", "f4"),
        ("sm_lat_ipp", "f4"),
        ("sm_lon_ipp", "f4"),
        ("sod", "f4"),
        ("lat_sta", "f4"),
        ("lon_sta", "f4"),
        ("sm_lat_sta", "f4"),
        ("sm_lon_sta", "f4"),
        ("slipc", "f4"),
        ("gfphase", "f4"),
    ]
)
# Fields the geometry cannot supply; the model reads none of them.
UNAVAILABLE = ["stec", "vtec", "vtec_stddev", "satres", "dcbs", "dcbr"]

# One observation's natural identity: which station observed which satellite at which
# second-of-day. Confirmed against a real recovered file (2024 DOY 166, 71,060 rows, 2
# stations): zero duplicate (station, sat, sod) triples. A station is only ever rebuilt
# from that one day's own RINEX in a single invocation, so its rows are internally unique
# on this key by construction.
IDENTITY_KEY_FIELDS = ("station", "sat", "sod")


class RecoveredDayShrinkError(RuntimeError):
    """A merge would leave fewer observations or fewer stations than were already on disk.

    Mirrors `stec.positioning.summary_writer.SummaryShrinkError`, which exists for the
    identical reason on `daily_summary*.csv`: a recovery run's station list is only the
    stations one day is *still* missing, so it must never be able to erase stations an
    earlier invocation already recovered into this same file. Refuse and let a human
    look, rather than silently writing a smaller file - the same failure shape that
    corrupted 59 daily_summary files before summary_writer.py existed.
    """


def merge_recovered_day(
    new_data: np.ndarray, existing: np.ndarray | None
) -> np.ndarray:
    """Merge `new_data` onto `existing`, replacing whole stations.

    Every station present in `new_data` replaces that station's rows in `existing`
    entirely - a re-run of a station updates its observations rather than appending
    duplicate epochs alongside the old ones. Every station present only in `existing`
    (recovered by an earlier invocation this one's `--stations` list does not cover) is
    carried through untouched. Because one invocation always rebuilds a station from that
    day's own RINEX (see `IDENTITY_KEY_FIELDS`'s docstring), replacing by station and
    de-duplicating on the full `(station, sat, sod)` key produce the same result here, and
    station-level replacement is the simpler of the two to reason about and to verify.
    """
    if existing is None or len(existing) == 0:
        return new_data

    new_stations = np.unique(new_data["station"])
    kept = existing[~np.isin(existing["station"], new_stations)]
    return np.concatenate([kept, new_data])


def raise_if_shrinking(
    existing: np.ndarray, merged: np.ndarray, out_path: Path
) -> None:
    """Raise `RecoveredDayShrinkError` if `merged` has fewer observations or fewer
    stations than `existing`.

    A recovery run's `--stations` list only ever covers what one day is *still*
    missing, so a merge that comes out smaller than what was already on disk means this
    run's rebuild of some station produced fewer rows than a previous run's did - the
    exact shape of bug that corrupted `daily_summary*.csv` before
    `stec.positioning.summary_writer` existed. Call only when `existing` is not None.
    """
    stations_before = len(np.unique(existing["station"]))
    stations_after = len(np.unique(merged["station"]))
    if len(merged) < len(existing) or stations_after < stations_before:
        raise RecoveredDayShrinkError(
            f"{out_path}: merge would shrink {len(existing):,} -> {len(merged):,} "
            f"observations, {stations_before} -> {stations_after} stations; refusing "
            "to write. This invocation's --stations only covers what's still missing, "
            "so a smaller merged result means this run's rebuild of a station produced "
            "fewer rows than were already on disk - check the RINEX/geometry inputs "
            "before overriding."
        )


def _read_existing_data(path: Path, year: int, doy: int) -> np.ndarray | None:
    """The structured `all_data` array already at `path`, or None if there is nothing
    usable there yet."""
    if not path.exists():
        return None
    with h5py.File(path, "r") as handle:
        return handle[str(year)][f"{doy:03d}"]["all_data"][:]


def _write_recovered_day_atomically(
    data: np.ndarray, out_path: Path, year: int, doy: int
) -> None:
    """Write `data` to `out_path` without ever exposing a partially-written file.

    Mirrors `stec.inference.prediction_store._write_parquet_atomically`: build the whole
    file at a temp path in the same directory, then `os.replace()` it into place, so a
    process killed mid-write - this runs unattended for hours as part of the recovery
    sweep - leaves the previous, complete file in place rather than a truncated one. The
    temp name embeds the PID so two concurrent writers to the same day never clobber each
    other's in-progress file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temp_path, "w") as handle:
            group = handle.require_group(str(year)).require_group(f"{doy:03d}")
            group.create_dataset("all_data", data=data, compression="gzip")
            # Every recovered observation is a test observation: these stations were
            # never available to train on, precisely because the database excluded them.
            group.create_dataset("test_idx", data=np.arange(len(data), dtype="i8"))
        os.replace(temp_path, out_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


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
    parser.add_argument(
        "--stations", required=True, help="comma-separated 4-character IDs"
    )
    parser.add_argument("--rinex_dir", type=Path, required=True)
    parser.add_argument("--nav", type=Path, required=True)
    parser.add_argument("--bsx", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--output_root", type=Path, default=Path("data/recovered_stec_db")
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=6,
        help="stations processed concurrently; each is a subprocess",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    stations = [s.strip().upper() for s in args.stations.split(",") if s.strip()]
    args.workdir.mkdir(parents=True, exist_ok=True)

    frames = []

    def skip(station: str, reason: str) -> None:
        logger.warning(f"skipped {station}: {reason}")

    def process(station: str) -> pd.DataFrame | None:
        """Geometry for one station. Each call is an independent CamaliotGnss subprocess."""
        matches = sorted(args.rinex_dir.glob(f"{station}*.rnx")) or sorted(
            args.rinex_dir.glob(f"{station.lower()}*.rnx")
        )
        if not matches:
            skip(station, "no RINEX")
            return None
        try:
            geometry = run_station(
                matches[0], args.nav, args.bsx, args.workdir / station
            )
        except (
            RuntimeError,
            ValueError,
            FileNotFoundError,
            subprocess.SubprocessError,
        ) as exc:
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

    existing = _read_existing_data(out_path, args.year, args.doy)
    merged = merge_recovered_day(data, existing)
    if existing is not None:
        raise_if_shrinking(existing, merged, out_path)

    _write_recovered_day_atomically(merged, out_path, args.year, args.doy)

    logger.info(
        f"💾 {out_path}: {len(merged):,} observations, "
        f"{len(np.unique(merged['station']))} station(s) total "
        f"({len(frames)} recovered this run)"
    )


if __name__ == "__main__":
    main()
