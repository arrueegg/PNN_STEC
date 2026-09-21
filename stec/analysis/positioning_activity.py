"""Positioning error stratified by local ionospheric activity.

Replaces the recovered/original population split as a reporting axis. That split was
measured to be a proxy for activity, not a property of the recovery pipeline: at matched
satellite count and at a station's own normal ionospheric level, recovered station-days
carry no penalty at all (x1.01), while the penalty at elevated level is x1.51
(docs/revision/metrics_and_exclusions_design.md Sec 1).

The activity proxy is **GIM VTEC at the station**, evaluated from the IONEX maps rather
than from the prediction store. Two reasons, both load-bearing:

* The store is missing exactly the station-days the recovery pipeline added - 49 of 57
  positioning stations are present on a typical day - so a store-derived proxy would be
  absent precisely where the question is sharpest.
* The model's own predicted STEC is circular as a stratifier for the model's own error.

No station-day is excluded. `elevated_ratio` splits each station against **its own**
median VTEC, so an equatorial station's ordinary day is not classified as elevated
merely for being equatorial.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..baselines.gim import GIMMapper
from ..config import paths
from .station_independence import load_station_coordinates

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_activity", rebuilt=True)
POSITIONING_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary.csv"
)
# A day's VTEC sampled on the IONEX epochs; zenith (90 deg) so the mapping factor is 1
# and the result is VTEC at the station, not a slant value.
SAMPLE_SODS = np.arange(0, 86400, 3600, dtype=float)
ZENITH_ELEVATION_DEG = 90.0
# A station-day counts as elevated above 1.1x that station's own median VTEC. Chosen to
# match the bin edge the effect was first measured at (design doc Sec 1); the stratum is
# reported with its N so a reader can see how the population splits.
DEFAULT_ELEVATED_RATIO = 1.1
IONO_WEIGHTING_SUFFIX = "_iono"


def station_day_vtec(
    station_coords: pd.DataFrame, year: int, doys: list[int]
) -> pd.DataFrame:
    """Mean GIM VTEC over each (station, doy), one IONEX load per day."""
    rows: list[dict] = []
    mapper = GIMMapper()
    for doy in doys:
        try:
            mapper.load_for_year_doy(year, doy)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(f"{year}-{doy:03d}: no GIM map ({exc}), skipping")
            continue
        for station, row in station_coords.iterrows():
            lat = np.full_like(SAMPLE_SODS, float(row["lat"]))
            lon = np.full_like(SAMPLE_SODS, float(row["lon"]))
            elev = np.full_like(SAMPLE_SODS, ZENITH_ELEVATION_DEG)
            vtec = mapper.map_vtec_to_stec(SAMPLE_SODS, lat, lon, elev)
            rows.append(
                {
                    "station": station,
                    "doy": doy,
                    "gim_vtec_mean": float(np.nanmean(vtec)),
                    "gim_vtec_max": float(np.nanmax(vtec)),
                }
            )
    return pd.DataFrame(rows)


def stratify(
    frame: pd.DataFrame, elevated_ratio: float = DEFAULT_ELEVATED_RATIO
) -> pd.DataFrame:
    """Label each station-day 'normal' or 'elevated' against its own station baseline."""
    out = frame.copy()
    baseline = out.groupby("station")["gim_vtec_mean"].transform("median")
    out["activity_ratio"] = out["gim_vtec_mean"] / baseline
    out["activity"] = np.where(
        out["activity_ratio"] > elevated_ratio, "elevated", "normal"
    )
    return out


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Median, quartiles, tail quantiles and exceedance counts per (method, stratum).

    Paired with exceedance counts on purpose: reporting the median alone would hide
    that the method produces more large errors than IGS GIM, which is real and
    operationally important (positioning_distributions's own caveat).
    """
    rows: list[dict] = []
    for (method, activity), group in frame.groupby(
        ["Method", "activity"], observed=True
    ):
        errors = group["error_3d_rms"]
        row = {
            "Method": method,
            "activity": activity,
            "n": len(errors),
            "median_m": float(errors.median()),
            "q1_m": float(errors.quantile(0.25)),
            "q3_m": float(errors.quantile(0.75)),
            "p95_m": float(errors.quantile(0.95)),
            "p99_m": float(errors.quantile(0.99)),
            "mean_m": float(errors.mean()),
        }
        for threshold in (5, 10, 20):
            row[f"exceed_{threshold}m"] = int((errors > threshold).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--elevated-ratio", type=float, default=DEFAULT_ELEVATED_RATIO)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    positioning = pd.read_csv(POSITIONING_SUMMARY)
    positioning = positioning[
        positioning["method"].str.endswith(IONO_WEIGHTING_SUFFIX)
    ].copy()
    positioning["station"] = positioning["station"].str.upper()
    positioning = positioning.rename(columns={"method": "Method"})

    coords = load_station_coordinates(paths.IGS_STATION_COORDINATES)
    coords = coords[coords.index.isin(positioning["station"].unique())]
    vtec = station_day_vtec(coords, args.year, sorted(positioning["doy"].unique()))

    merged = positioning.merge(vtec, on=["station", "doy"], how="inner")
    missing = len(positioning) - len(merged)
    if missing:
        logger.warning(f"{missing} station-day rows have no GIM VTEC and are dropped")

    stratified = stratify(merged, args.elevated_ratio)
    summary = summarise(stratified)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stratified.to_csv(args.output_dir / "station_day_activity.csv", index=False)
    summary.to_csv(args.output_dir / "activity_stratification.csv", index=False)
    print(summary.round(3).to_string(index=False))
    logger.info(f"wrote activity_stratification.csv to {args.output_dir}")


if __name__ == "__main__":
    main()
