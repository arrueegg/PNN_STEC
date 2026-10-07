"""Compare independently computed geometry with the STEC database, row by row (V1).

Usage::

    python positioning/geometry/validate_independent_geometry.py \
        --cases MRO1:122 REYK:133 NKLG:132 --output_dir data/independent_geometry_work/v1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import independent_geometry as ig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATABASE_ROOT = Path("/home/space/data/iono/STEC_DB_CASDCB")
RINEX_DIR = REPO / "data" / "independent_geometry_work" / "rinex"
COMPARED_COLUMNS = [
    "satele",
    "satazi",
    "lat_ipp",
    "lon_ipp",
    "sm_lat_ipp",
    "sm_lon_ipp",
    "lat_sta",
    "lon_sta",
    "sm_lat_sta",
    "sm_lon_sta",
]
ANGLE_WRAP_COLUMNS = {"satazi", "lon_ipp", "lon_sta", "sm_lon_ipp", "sm_lon_sta"}


def products_dir(doy: int) -> Path:
    pattern = (
        f"experiments/Finetune_STEC_2024_{doy}_*lr2e-4*SWI/positioning/evaluation/"
        f"2024{doy}/products"
    )
    return Path(glob.glob(str(REPO / pattern))[0])


def read_database_station(station: str, year: int, doy: int) -> pd.DataFrame:
    path = DATABASE_ROOT / str(year) / f"{doy:03d}" / f"ccl_{year}{doy:03d}_30_5.h5"
    with h5py.File(path, "r") as handle:
        dataset = handle[str(year)][f"{doy:03d}"]["all_data"]
        indices = np.where(dataset.fields("station")[:] == station.encode())[0]
        frame = pd.DataFrame(dataset[indices[0] : indices[-1] + 1])
    frame = frame[frame.station == station.encode()].copy()
    frame["sat"] = frame["sat"].str.decode("ascii")
    return frame


def wrapped_abs_difference(
    column: str, mine: pd.Series, theirs: pd.Series
) -> pd.Series:
    difference = (mine - theirs).abs()
    if column in ANGLE_WRAP_COLUMNS:
        difference = np.minimum(difference, 360.0 - difference)
    return difference


def validate_case(
    station: str, doy: int, output_dir: Path, year: int = 2024
) -> tuple[list[dict], dict]:
    rinex = next(RINEX_DIR.glob(f"{station}*{year}{doy:03d}0000*"))
    products = products_dir(doy)
    database = read_database_station(station, year, doy)
    mine = ig.compute_station_geometry(
        station,
        year,
        doy,
        rinex,
        next(products.glob("*ORB.SP3")),
        next(products.glob("IGS*SNX")),
    )
    merged = mine.merge(database, on=["sod", "sat"], suffixes=("", "_db"))
    rows = []
    for column in COMPARED_COLUMNS:
        error = wrapped_abs_difference(column, merged[column], merged[column + "_db"])
        rows.append(
            {
                "station": station,
                "doy": doy,
                "column": column,
                "max": error.max(),
                "p99": error.quantile(0.99),
                "median": error.median(),
            }
        )
    database_keys = set(zip(database.sod, database.sat))
    mine_keys = set(zip(mine.sod, mine.sat))
    extra = mine[[key not in database_keys for key in zip(mine.sod, mine.sat)]]
    missing = database[
        [key not in mine_keys for key in zip(database.sod, database.sat)]
    ]
    summary = {
        "station": station,
        "doy": doy,
        "db_rows": len(database),
        "my_rows": len(mine),
        "matched": len(merged),
        "frac_db_reproduced": len(merged) / len(database),
        "extra_rows": len(extra),
        "extra_without_dual_frequency": int((~extra.has_dual).sum()),
        "extra_elevation_lt_10": int((extra.satele < 10).sum()),
        "db_rows_not_reproduced": len(missing),
        "constellations_db": "".join(sorted(set(database.sat.str[0]))),
    }
    merged.to_parquet(output_dir / f"v1_{station}_{doy}_matched.parquet", index=False)
    extra.to_parquet(output_dir / f"v1_{station}_{doy}_extra.parquet", index=False)
    missing.to_parquet(output_dir / f"v1_{station}_{doy}_missing.parquet", index=False)
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, help="STATION:DOY")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    error_rows, summaries = [], []
    for case in args.cases:
        station, doy = case.split(":")
        rows, summary = validate_case(station, int(doy), args.output_dir)
        error_rows += rows
        summaries.append(summary)
    errors = pd.DataFrame(error_rows)
    summary_frame = pd.DataFrame(summaries)
    errors.to_csv(args.output_dir / "v1_errors.csv", index=False)
    summary_frame.to_csv(args.output_dir / "v1_summary.csv", index=False)
    print(summary_frame.to_string())
    print(errors.to_string())


if __name__ == "__main__":
    main()
