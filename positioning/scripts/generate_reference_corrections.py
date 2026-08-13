"""Write the GNSS-derived reference STEC as a PPPx ionospheric correction file.

Evidence for reviewer comment R2.8:

    "The positioning experiment should include a benchmark in which the
     GNSS-derived reference STEC used for model training/evaluation is directly
     applied as the ionospheric correction. Although this reference STEC is not
     an absolute truth, it would provide an important near-oracle or
     observation-derived upper-bound baseline for the PPP experiment."

The reference STEC is the same quantity the model is trained against, taken
straight from the daily processed database rather than predicted. Feeding it to
PPPx bounds how well any model of this target could possibly do inside this
processing pipeline.

Two properties of that bound are worth stating with the result:

* It is **not an independent truth**. The reference STEC comes from the same
  dual-frequency observations at the same stations, carrying the same DCB
  handling and carrier-phase levelling. It bounds the model, not reality.
* It is **not complete**. Only 44-46 of the 55 positioning stations appear in
  the database on a typical day, so the comparison holds on the intersection and
  the station count should be reported alongside.

Because the reference carries no per-observation uncertainty, the `uncertainty`
column is filled with a constant and the PPPx run should use
``--weight_opt elev``; comparing against the other methods' elevation-weighted
arm keeps the weighting scheme out of the comparison.

Usage::

    python positioning/scripts/generate_reference_corrections.py \\
        --year 2024 --doy 183 --output_dir positioning/outputs/reference_corrections
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("/home/space/data/iono/STEC_DB_CASDCB")
DEFAULT_STATION_LIST = Path("src/data_processing/test_station.list")

# Matches the elevation cut applied when the database was produced.
MIN_ELEVATION_DEG = 5.0

# PPPx needs the column, but the reference has no per-observation sigma. The
# value is irrelevant under elevation weighting; it is written so the file
# format stays identical to the model correction files.
PLACEHOLDER_UNCERTAINTY_TECU = 1.0


def load_stations(path: Path) -> list[str]:
    return [s.upper() for s in np.loadtxt(path, dtype=str)]


def read_reference_day(db_root: Path, year: int, doy: int) -> pd.DataFrame:
    """Read one processed day out of the STEC database."""
    path = db_root / str(year) / f"{doy:03d}" / f"ccl_{year}{doy:03d}_30_5.h5"
    if not path.exists():
        raise FileNotFoundError(f"No processed STEC for {year}-{doy:03d} at {path}")

    with h5py.File(path, "r") as handle:
        data = handle[str(year)][f"{doy:03d}"]["all_data"]
        frame = pd.DataFrame(
            {
                "station": np.char.upper(data["station"][:].astype(str)),
                "sat": data["sat"][:].astype(str),
                "sod": data["sod"][:],
                "lat_ipp": data["lat_ipp"][:],
                "lon_ipp": data["lon_ipp"][:],
                "stec": data["stec"][:],
                "satele": data["satele"][:],
            }
        )
    return frame


def write_station_files(
    frame: pd.DataFrame, stations: list[str], output_dir: Path, year: int, doy: int
) -> list[str]:
    """Write one correction CSV per station, in the format PPPx expects."""
    target = output_dir / f"{year}{doy:03d}"
    target.mkdir(parents=True, exist_ok=True)

    frame = frame[frame["satele"] >= MIN_ELEVATION_DEG]
    frame = frame[frame["station"].isin(stations)]
    frame = frame[np.isfinite(frame["stec"])]

    written = []
    for station, group in frame.groupby("station"):
        export = pd.DataFrame(
            {
                "second_of_day": group["sod"].values,
                "PRN": group["sat"].values,
                "ipp_latitude": group["lat_ipp"].values,
                "ipp_longitude": group["lon_ipp"].values,
                "stec": group["stec"].values,
                "uncertainty": PLACEHOLDER_UNCERTAINTY_TECU,
            }
        ).sort_values(["second_of_day", "PRN"])
        export.to_csv(target / f"{station}.csv", index=False, float_format="%.4f")
        written.append(station)

    return sorted(written)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--doy", type=int, help="Single day of year")
    parser.add_argument("--start_doy", type=int, help="Start of a day range")
    parser.add_argument("--end_doy", type=int, help="End of a day range, inclusive")
    parser.add_argument("--db_root", type=Path, default=DEFAULT_DB)
    parser.add_argument("--station_list", type=Path, default=DEFAULT_STATION_LIST)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("positioning/outputs/reference_corrections"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.doy is not None:
        doys = [args.doy]
    elif args.start_doy is not None and args.end_doy is not None:
        doys = list(range(args.start_doy, args.end_doy + 1))
    else:
        parser.error("give either --doy or both --start_doy and --end_doy")

    stations = load_stations(args.station_list)
    logger.info(f"{len(stations)} candidate stations from {args.station_list}")

    coverage = []
    for doy in doys:
        try:
            frame = read_reference_day(args.db_root, args.year, doy)
        except FileNotFoundError as exc:
            logger.warning(f"⚠️  {exc}")
            continue
        written = write_station_files(frame, stations, args.output_dir, args.year, doy)
        coverage.append(
            {"year": args.year, "doy": doy, "stations_written": len(written)}
        )
        logger.info(
            f"{args.year}-{doy:03d}: wrote {len(written)}/{len(stations)} stations "
            f"to {args.output_dir / f'{args.year}{doy:03d}'}"
        )

    if coverage:
        summary = pd.DataFrame(coverage)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_dir / "station_coverage.csv", index=False)
        logger.info(
            f"Station coverage: median {summary['stations_written'].median():.0f} of "
            f"{len(stations)} per day — report this alongside the benchmark"
        )


if __name__ == "__main__":
    main()
