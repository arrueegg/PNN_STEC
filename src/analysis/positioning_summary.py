"""Paper-ready positioning summary tables.

Reproduces and extends Table 5 of the manuscript as CSV, so the tables can be
rebuilt or restratified without re-running PPP. Three tables are written:

* ``overall.csv`` - the Table 5 columns per method: 3D mean, 3D median, 2D mean
  and Up mean, plus the station-day count behind each row.
* ``by_regime.csv`` - the same columns split into quiet and storm days, which is
  what the revised table needs for R2.7.
* ``by_weighting.csv`` - the same columns for the elevation- and
  uncertainty-weighted arms, for R2.5.

All three apply the 10 m station-day exclusion used in Figure 12, so the numbers
line up with the published ones rather than nearly doing so.

Usage::

    python src/analysis/positioning_summary.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTLIER_3D_RMS_M = 10.0
STORM_DST_THRESHOLD = -50.0

PAPER_METHODS = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}
WEIGHTING_METHODS = {
    "STEC_elev": ("Direct STEC", "elevation"),
    "STEC_iono": ("Direct STEC", "predicted uncertainty"),
    "VTEC_elev": ("VTEC + Mapping", "elevation"),
    "VTEC_iono": ("VTEC + Mapping", "predicted uncertainty"),
    "gim_elev": ("IGS GIM + Mapping", "elevation"),
    "gim_iono": ("IGS GIM + Mapping", "predicted uncertainty"),
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]


def summarise(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Table 5 columns, computed over whatever grouping is asked for."""
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


def load_storm_doys(swi_path: Path, year: int) -> set[int] | None:
    import h5py

    if not swi_path.exists():
        logger.warning(f"⚠️  {swi_path} not found - skipping the regime table")
        return None
    with h5py.File(swi_path, "r") as handle:
        group = handle[str(year)]
        doys = sorted(group.keys(), key=int)
        columns = [
            c.decode() if isinstance(c, bytes) else c
            for c in group[doys[0]].attrs["columns"]
        ]
        dst = columns.index("Dst-index,_nT")
        return {
            int(d)
            for d in doys
            if float(np.nanmin(np.asarray(group[d])[:, dst])) <= STORM_DST_THRESHOLD
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper_summary",
        type=Path,
        default=Path(
            "multiday_results/positioning_comparison_3way/multiday_summary.csv"
        ),
        help="The iono-weighted run behind Table 5 and Figures 12/13",
    )
    parser.add_argument(
        "--weighting_summary",
        type=Path,
        default=Path("multiday_results/positioning_20260216_2052/multiday_summary.csv"),
        help="The run carrying both weighting arms",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--swi_path", type=Path, default=Path("data/omni_hourly_2010-2025.h5")
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/positioning_summary")
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paper = pd.read_csv(args.paper_summary)
    paper = paper[paper["error_3d_rms"] <= OUTLIER_3D_RMS_M].copy()
    paper["Method"] = paper["method"].map(PAPER_METHODS)
    paper = paper.dropna(subset=["Method"])

    overall = summarise(paper, ["Method"]).reindex(METHOD_ORDER)
    overall.to_csv(args.output_dir / "overall.csv")
    print("=== Overall (Table 5 columns) ===")
    print(overall.to_string())

    storm_doys = load_storm_doys(args.swi_path, args.year)
    if storm_doys is not None:
        paper["regime"] = np.where(paper["doy"].isin(storm_doys), "storm", "quiet")
        by_regime = summarise(paper, ["Method", "regime"])
        by_regime.to_csv(args.output_dir / "by_regime.csv")
        print("\n=== By geomagnetic regime ===")
        print(by_regime.to_string())

    if args.weighting_summary.exists():
        weighting = pd.read_csv(args.weighting_summary)
        weighting = weighting[weighting["error_3d_rms"] <= OUTLIER_3D_RMS_M].copy()
        mapped = weighting["method"].map(WEIGHTING_METHODS)
        weighting = weighting[mapped.notna()].copy()
        weighting[["Method", "weighting"]] = pd.DataFrame(
            mapped.dropna().tolist(), index=weighting.index
        )
        by_weighting = summarise(weighting, ["Method", "weighting"])
        by_weighting.to_csv(args.output_dir / "by_weighting.csv")
        print("\n=== By observation weighting ===")
        print(by_weighting.to_string())
    else:
        logger.warning(
            f"⚠️  {args.weighting_summary} not found - skipping the weighting table"
        )

    logger.info(f"💾 {args.output_dir}")


if __name__ == "__main__":
    main()
