"""V3: how many satellites would independent geometry add on single-constellation station-days?

Compares, per station-day, the satellites the existing correction source (STEC database or
recovered database) carries with what `independent_geometry` produces from the RINEX, and
with the satellite count PPPx's IGS-GIM arm achieved (`positioning_coverage` summary).

Usage::

    python positioning/geometry/check_independent_coverage.py \
        --cases ZECK:150:db FAA1:150:db BIK0:122:recovered REYK:194:db \
        --output_dir data/independent_geometry_work/v3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import independent_geometry as ig  # noqa: E402
from validate_independent_geometry import (  # noqa: E402
    DATABASE_ROOT,
    RINEX_DIR,
    products_dir,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOTS = {"db": DATABASE_ROOT, "recovered": REPO / "data/recovered_stec_db"}
POSITIONING_SUMMARY = (
    REPO / "multiday_results/analyses/positioning_coverage/rebuilt/multiday_summary.csv"
)
# `generate_ini.py`'s default elevation mask for PPPx.
PPPX_ELEVATION_MASK_DEG = 7.0
YEAR = 2024


def mean_satellites_per_epoch(
    frame: pd.DataFrame, constellation: str | None = None
) -> float:
    if constellation:
        frame = frame[frame.sat.str[0] == constellation]
    if frame.empty:
        return 0.0
    return float(frame.groupby("sod").size().mean())


def read_source_rows(root: Path, station: str, doy: int) -> pd.DataFrame:
    path = root / str(YEAR) / f"{doy:03d}" / f"ccl_{YEAR}{doy:03d}_30_5.h5"
    with h5py.File(path, "r") as handle:
        dataset = handle[str(YEAR)][f"{doy:03d}"]["all_data"]
        indices = np.where(dataset.fields("station")[:] == station.encode())[0]
        frame = pd.DataFrame(dataset[indices[0] : indices[-1] + 1])
    frame = frame[frame.station == station.encode()]
    return pd.DataFrame(
        {
            "sod": frame.sod.to_numpy(float),
            "sat": frame.sat.str.decode("ascii").to_numpy(),
            "satele": frame.satele.to_numpy(float),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", nargs="+", required=True, help="STATION:DOY:db|recovered"
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(POSITIONING_SUMMARY)

    rows = []
    for case in args.cases:
        station, doy_text, source_name = case.split(":")
        doy = int(doy_text)
        products = products_dir(doy)
        started = time.time()
        mine = ig.compute_station_geometry(
            station,
            YEAR,
            doy,
            next(RINEX_DIR.glob(f"{station}*{YEAR}{doy:03d}0000*")),
            next(products.glob("*ORB.SP3")),
            next(products.glob("IGS*SNX")),
        )
        seconds = time.time() - started
        source = read_source_rows(SOURCE_ROOTS[source_name], station, doy)
        present = sorted(set(source.sat.str[0]))
        missing = [c for c in "GE" if c not in present]
        masked_mine = mine[mine.satele >= PPPX_ELEVATION_MASK_DEG]
        masked_source = source[source.satele >= PPPX_ELEVATION_MASK_DEG]
        added = masked_mine[masked_mine.sat.str[0].isin(missing)]
        added_dual = added[added.has_dual]
        reported = summary[
            (summary.station == station) & (summary.doy == doy)
        ].set_index("method")["mean_nsat"]
        source_nsat = mean_satellites_per_epoch(masked_source)
        row = {
            "station": station,
            "doy": doy,
            "source": source_name,
            "constellations_in_source": "".join(present),
            "missing_constellation": "".join(missing),
            "source_mean_sats_ge7deg": source_nsat,
            "independent_rows_missing_const_ge7deg": len(added),
            "independent_rows_missing_const_dual_freq": len(added_dual),
            "independent_missing_const_mean_sats": mean_satellites_per_epoch(added),
            "independent_missing_const_mean_sats_dual": mean_satellites_per_epoch(
                added_dual
            ),
            "independent_all_mean_sats_ge7deg": mean_satellites_per_epoch(masked_mine),
            "source_plus_added_mean_sats": source_nsat
            + mean_satellites_per_epoch(added),
            "ppx_ml_mean_nsat": reported.get(
                "STEC_iono", reported.get("Pretrained_STEC_iono")
            ),
            "ppx_gim_mean_nsat": reported.get("gim_iono"),
            "geometry_seconds_incl_solar_magnetic": seconds,
            "independent_rows_total_ge5deg": len(mine),
        }
        rows.append(row)
        mine.to_parquet(
            args.output_dir / f"v3_{station}_{doy}_independent.parquet", index=False
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output_dir / "v3_coverage.csv", index=False)
    print(result.T.to_string())


if __name__ == "__main__":
    main()
