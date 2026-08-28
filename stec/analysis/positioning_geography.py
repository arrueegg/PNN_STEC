"""Geographic and geomagnetic-latitude analysis of positioning performance (owner
request, 2026-08-28).

Not a declared pipeline stage - like `stec.analysis.positioning_diagnostics`, which this
module reuses rather than duplicates, the owner wants to see this output before deciding
whether/how it becomes part of the manuscript or an appendix.

Motivation (verified against the data, not assumed): ionospheric variability is
dominated by the equatorial ionisation anomaly and the auroral oval, both organised by
**geomagnetic**, not geographic, latitude. `stratified_comparison.py` already stratifies
per-observation STEC error by `sm_lat_ipp` for exactly this reason (its own
`GEOMAGNETIC_BINS`, reused here unchanged as `LATITUDE_BINS` so the geographic and
geomagnetic stratifications below use identical bin edges and are directly comparable
bin-for-bin). This module asks the same question at the *positioning* level: does
Direct STEC's advantage over IGS GIM vary with latitude, and does it vary more sharply
with geomagnetic latitude than with geographic latitude?

Three deliverables, each a function group below:

1. Per-station mean/median 3D error (Direct STEC, and the Direct-STEC-minus-GIM
   difference) with coordinates attached, for `stec.viz.positioning_geography`'s station
   maps.
2. Latitude stratification - per-method error and the paired Direct-STEC-vs-GIM
   difference, binned by both geographic and geomagnetic latitude, mean and median, with
   N per bin.
3. Crossing latitude with the recovered/original population split
   (`positioning_diagnostics.attach_population`): are the geometry-only recovered
   station-days concentrated at particular latitudes, and does Direct STEC's
   underperformance there survive controlling for latitude, or collapse into it?

Station coordinates: `IGSNetwork.csv` (`stec.config.paths.IGS_STATION_COORDINATES`),
read via `stec.analysis.station_independence.load_station_coordinates` rather than a new
copy of the same parsing. Checked directly against the 57 stations in the canonical
positioning summary: **all 57 have a coordinate row** (0 missing), so there is no
coverage reason to prefer the alternative, the SINEX-derived ground truth
`stec.positioning.metrics.load_sinex_coords` reads per experiment per day for the
positioning error computation itself. That SINEX truth is more precise (it is what the
error is measured against) but reading it means walking per-day product trees under
`experiments/*/positioning/evaluation/*/products/` - exactly what the actively-running
`elev-positioning-chain.service` is writing to concurrently - for a precision gain that
does not matter at the tens-of-degrees bin widths this module uses. IGSNetwork.csv and a
station's true ITRF position agree to metres, six orders of magnitude below one degree
of latitude.

Geomagnetic latitude: `stec.data.coordinate_transforms.geographic_to_magnetic_latitude`
(spacepy-based, the repo's own transform - not a new one), evaluated once per station at
a single fixed reference epoch rather than per observation. This is deliberate, not a
shortcut: the SM (solar-magnetic) frame shares its Z-axis with the plain dipole-geomagnetic
frame, and the further rotation from GM to SM that aligns the X-axis with the Sun is
purely a rotation *about* that shared Z-axis - it changes SM longitude, not SM latitude.
So a fixed station's SM latitude should not depend on time of day or season at all, up to
the dipole's own slow secular drift. Verified empirically (see
`tests/analysis/test_positioning_geography.py::test_geomagnetic_latitude_is_effectively_time_invariant`):
for a representative point, SM latitude varies by <=0.01 deg across every UT quarter and
every season of 2024, against bin widths of 10-30 deg - so a single reference epoch per
station is a station property, not an approximation that needs revisiting.
`stec.data.madrigal_builder` computes `sm_lat_sta`/`sm_lat_ipp` per observation second
because it needs the *longitude* half of the same call at real resolution; this module
only needs latitude, so one call per station is enough.

Source data: `stec.analysis.positioning_diagnostics.load_positioning_table` (which
already restricts to the paper's four iono-weighted methods -
`stec.analysis.positioning_summary.PAPER_METHODS` - so every number in this module is
the **iono** arm; the elevation arm is being re-solved as this module is written and
must not be read). Recovered-population membership prefers the CSV
`positioning_diagnostics` already wrote (`analyses/positioning_diagnostics/rebuilt/
recovered_station_days.csv`) over re-scanning the 242-file `data/recovered_stec_db/`
tree, to avoid the extra I/O when the cache is already on disk.

Usage::

    python -m stec.analysis.positioning_geography
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import paths
from ..data.coordinate_transforms import geographic_to_magnetic_latitude
from ..positioning import metrics as pm
from .positioning_coverage import DEFAULT_OUTPUT_DIR as POSITIONING_COVERAGE_DIR
from .positioning_diagnostics import (
    GIM_LABEL,
    STEC_LABEL,
    RECOVERED_STEC_DB_ROOT,
    attach_population,
    load_positioning_table,
    load_recovered_station_days,
    per_station_summary,
)
from .positioning_summary import canonical_positioning_summary
from .station_independence import load_station_coordinates
from .stratified_comparison import GEOMAGNETIC_BINS

logger = logging.getLogger(__name__)

# Same bin edges `stratified_comparison` uses for per-observation geomagnetic
# stratification, reused unchanged (not redefined) for both geographic and geomagnetic
# station latitude here so the two are comparable bin-for-bin.
LATITUDE_BINS: tuple[float, ...] = tuple(GEOMAGNETIC_BINS)

# See the module docstring: SM latitude of a fixed station is time-invariant to within
# <=0.01 deg, so one reference epoch (mid-2024, the middle of the 2024 test period) is a
# station property, not a per-observation quantity.
GEOMAGNETIC_REFERENCE_EPOCH = datetime(2024, 7, 1, 12, 0, 0)

COVERAGE_CSV = POSITIONING_COVERAGE_DIR / "coverage.csv"
CACHED_RECOVERED_CSV = (
    paths.analysis_result_dir("positioning_diagnostics", rebuilt=True)
    / "recovered_station_days.csv"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_geography", rebuilt=True)


# --------------------------------------------------------------------------
# Station coordinates (geographic + geomagnetic)
# --------------------------------------------------------------------------


def add_geomagnetic_latitude(
    coords: pd.DataFrame, reference_time: datetime = GEOMAGNETIC_REFERENCE_EPOCH
) -> pd.DataFrame:
    """Attach `sm_lat` to a station coordinate table indexed by station (`lat`, `lon`
    columns, the shape `station_independence.load_station_coordinates` returns)."""
    out = coords.copy()
    out["sm_lat"] = [
        float(geographic_to_magnetic_latitude(lat, lon, reference_time))
        for lat, lon in zip(out["lat"], out["lon"])
    ]
    return out


def load_station_geography(
    network_csv: Path = paths.IGS_STATION_COORDINATES,
    reference_time: datetime = GEOMAGNETIC_REFERENCE_EPOCH,
) -> pd.DataFrame:
    """Station -> lat, lon, sm_lat, indexed by the 4-character station code."""
    coords = load_station_coordinates(network_csv)
    return add_geomagnetic_latitude(coords, reference_time)


def load_recovered_station_days_cached(
    cache_csv: Path = CACHED_RECOVERED_CSV,
    root: Path = RECOVERED_STEC_DB_ROOT,
) -> pd.DataFrame:
    """(year, doy, station) recovered triples, preferring the CSV
    `positioning_diagnostics` already wrote over re-scanning the HDF5 tree."""
    if cache_csv.exists():
        logger.info(f"recovered station-days: {cache_csv} (cached)")
        return pd.read_csv(cache_csv)
    return load_recovered_station_days(root)


# --------------------------------------------------------------------------
# Building the combined (station-day x method) geography frame
# --------------------------------------------------------------------------


def build_geo_frame(
    positioning_frame: pd.DataFrame,
    station_coords: pd.DataFrame,
    recovered: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """`positioning_frame` (station, Method, doy, error_3d_rms, ...) with `lat`, `lon`,
    `sm_lat` attached from `station_coords`, and `population` attached if `recovered` is
    given. No outlier exclusion here - callers apply
    `stec.positioning.metrics.exclude_outlier_station_days` themselves, the same
    "raw in, excluded where computed" convention `positioning_diagnostics` uses."""
    merged = positioning_frame.merge(
        station_coords[["lat", "lon", "sm_lat"]],
        left_on="station",
        right_index=True,
        how="left",
    )
    missing = sorted(merged.loc[merged["lat"].isna(), "station"].unique())
    if missing:
        logger.warning(f"no coordinates for {len(missing)} station(s): {missing}")
    if recovered is not None:
        merged = attach_population(merged, recovered)
    return merged


# --------------------------------------------------------------------------
# 1. Per-station maps
# --------------------------------------------------------------------------


def per_method_station_summary(
    geo_frame: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M
) -> pd.DataFrame:
    """One row per (station, Method): mean/median/N of `error_3d_rms` plus `lat`,
    `lon`, `sm_lat` - the table `fig_station_map` reads for the plain (non-diff) maps."""
    kept = pm.exclude_outlier_station_days(geo_frame, threshold)
    out = (
        kept.groupby(["station", "Method"])
        .agg(
            mean_error_3d_m=("error_3d_rms", "mean"),
            median_error_3d_m=("error_3d_rms", "median"),
            station_days=("error_3d_rms", "size"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            sm_lat=("sm_lat", "first"),
        )
        .reset_index()
    )
    return out


def station_diff_map_table(
    positioning_frame: pd.DataFrame,
    station_coords: pd.DataFrame,
    threshold: float = pm.OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """`positioning_diagnostics.per_station_summary` (Direct-STEC-minus-GIM diff, mean
    and median) with `lat`, `lon`, `sm_lat` attached - the table `fig_station_diff_map`
    reads."""
    diff = per_station_summary(positioning_frame, threshold)
    return diff.merge(
        station_coords[["lat", "lon", "sm_lat"]],
        left_on="station",
        right_index=True,
        how="left",
    )


# --------------------------------------------------------------------------
# 2. Latitude stratification
# --------------------------------------------------------------------------


def latitude_stratification(
    geo_frame: pd.DataFrame,
    lat_column: str,
    bins: tuple[float, ...] = LATITUDE_BINS,
    threshold: float = pm.OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """Per (lat_bin, Method): mean/median/N of `error_3d_rms`, after the 10 m rule.
    `lat_column` is `"lat"` (geographic) or `"sm_lat"` (geomagnetic) - same function,
    same bins, so the two stratifications are exactly comparable."""
    kept = pm.exclude_outlier_station_days(geo_frame, threshold)
    binned = pd.cut(kept[lat_column], bins=bins, include_lowest=True)
    return (
        kept.assign(lat_bin=binned)
        .groupby(["lat_bin", "Method"], observed=True)["error_3d_rms"]
        .agg(mean="mean", median="median", n="count")
        .reset_index()
    )


def _paired_stec_gim(
    geo_frame: pd.DataFrame, threshold: float, extra_index: list[str] | None = None
) -> pd.DataFrame:
    """(station, doy[, extra_index...]) rows with both a surviving Direct STEC and IGS
    GIM `error_3d_rms` - the building block for every paired STEC-vs-GIM comparison
    below. A row without both methods' outlier-surviving value is dropped, the same
    pairwise handling `positioning_diagnostics.per_station_summary` uses."""
    kept = pm.exclude_outlier_station_days(geo_frame, threshold)
    index_cols = ["station", "doy"] + (extra_index or [])
    pivot = kept.pivot_table(index=index_cols, columns="Method", values="error_3d_rms")
    paired = pivot[[STEC_LABEL, GIM_LABEL]].rename(
        columns={STEC_LABEL: "stec", GIM_LABEL: "gim"}
    )
    paired = paired.dropna(subset=["stec", "gim"]).reset_index()
    paired["diff"] = paired["stec"] - paired["gim"]
    return paired


def _summarise_paired(sub: pd.DataFrame) -> dict:
    stec_mean, gim_mean = sub["stec"].mean(), sub["gim"].mean()
    stec_median, gim_median = sub["stec"].median(), sub["gim"].median()
    return {
        "n": len(sub),
        "stec_mean_m": stec_mean,
        "gim_mean_m": gim_mean,
        "diff_mean_m": stec_mean - gim_mean,
        "mean_improvement_pct": 100 * (gim_mean - stec_mean) / gim_mean,
        "stec_median_m": stec_median,
        "gim_median_m": gim_median,
        "diff_median_m": stec_median - gim_median,
        "median_improvement_pct": 100 * (gim_median - stec_median) / gim_median,
    }


def latitude_stratification_diff(
    geo_frame: pd.DataFrame,
    lat_column: str,
    bins: tuple[float, ...] = LATITUDE_BINS,
    threshold: float = pm.OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """Per lat_bin: Direct STEC vs IGS GIM, paired at station-day granularity, mean AND
    median of each, their difference, and the percent improvement - the same convention
    as `positioning_diagnostics.outlier_headline_sensitivity`, but stratified by
    latitude bin instead of by outlier threshold. Answers "does the model's advantage
    vary with latitude" directly, in the paper's own improvement-percent units."""
    lat_lookup = geo_frame.drop_duplicates("station").set_index("station")[lat_column]
    paired = _paired_stec_gim(geo_frame, threshold)
    paired["lat_bin"] = pd.cut(
        paired["station"].map(lat_lookup), bins=bins, include_lowest=True
    )
    rows = []
    for lat_bin, sub in paired.groupby("lat_bin", observed=True):
        rows.append({"lat_bin": lat_bin, **_summarise_paired(sub)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. Crossing latitude with the recovered/original population split
# --------------------------------------------------------------------------


def recovered_concentration_by_latitude(
    coverage_frame: pd.DataFrame,
    recovered: pd.DataFrame,
    station_coords: pd.DataFrame,
    lat_column: str,
    bins: tuple[float, ...] = LATITUDE_BINS,
) -> pd.DataFrame:
    """Per lat_bin: how many of `coverage.csv`'s (doy, station) station-days are
    "recovered" (geometry-only) vs "original", and the recovered share. Uses
    `coverage.csv` rather than the 4-method-duplicated positioning table, since a
    station-day's recovered/original status does not depend on method and
    `coverage.csv` already holds one row per station-day. Answers "are the recovered
    station-days concentrated at particular latitudes" directly."""
    frame = coverage_frame[["doy", "station"]].copy()
    frame["station"] = frame["station"].str.upper()
    keys = recovered[["doy", "station"]].drop_duplicates()
    keys["population"] = "recovered"
    merged = frame.merge(keys, on=["doy", "station"], how="left")
    merged["population"] = merged["population"].fillna("original")
    merged = merged.merge(
        station_coords[[lat_column]], left_on="station", right_index=True, how="left"
    )
    merged["lat_bin"] = pd.cut(merged[lat_column], bins=bins, include_lowest=True)

    counts = (
        merged.groupby(["lat_bin", "population"], observed=True)
        .size()
        .rename("n")
        .reset_index()
        .pivot(index="lat_bin", columns="population", values="n")
        .fillna(0)
    )
    for col in ("original", "recovered"):
        if col not in counts:
            counts[col] = 0
    counts["total"] = counts["original"] + counts["recovered"]
    counts["pct_recovered"] = (
        100 * counts["recovered"] / counts["total"].replace(0, pd.NA)
    )
    return counts.reset_index()


def population_latitude_diff(
    geo_frame_with_population: pd.DataFrame,
    lat_column: str,
    bins: tuple[float, ...] = LATITUDE_BINS,
    threshold: float = pm.OUTLIER_3D_RMS_M,
) -> pd.DataFrame:
    """Per (lat_bin, population): Direct STEC vs IGS GIM, paired, mean/median/diff/
    improvement - the same shape as `latitude_stratification_diff`, split further by
    `population`. Compares the recovered population's underperformance *within* a
    latitude bin against the original population's, so a latitude effect and a
    population effect can be told apart rather than conflated."""
    lat_lookup = geo_frame_with_population.drop_duplicates("station").set_index(
        "station"
    )[lat_column]
    paired = _paired_stec_gim(
        geo_frame_with_population, threshold, extra_index=["population"]
    )
    paired["lat_bin"] = pd.cut(
        paired["station"].map(lat_lookup), bins=bins, include_lowest=True
    )
    rows = []
    for (lat_bin, population), sub in paired.groupby(
        ["lat_bin", "population"], observed=True
    ):
        rows.append(
            {"lat_bin": lat_bin, "population": population, **_summarise_paired(sub)}
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument(
        "--network-csv", type=Path, default=paths.IGS_STATION_COORDINATES
    )
    parser.add_argument("--coverage-csv", type=Path, default=COVERAGE_CSV)
    parser.add_argument(
        "--recovered-cache-csv", type=Path, default=CACHED_RECOVERED_CSV
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "iono weighting only - the elev arm is being re-solved as of 2026-08-28"
    )
    positioning_frame = load_positioning_table(args.summary_path)
    station_coords = load_station_geography(args.network_csv)
    recovered = load_recovered_station_days_cached(args.recovered_cache_csv)
    geo_frame = build_geo_frame(positioning_frame, station_coords, recovered)

    # 1. Station maps.
    direct_stec = per_method_station_summary(geo_frame)
    direct_stec = direct_stec[direct_stec["Method"] == STEC_LABEL].drop(
        columns="Method"
    )
    direct_stec.to_csv(args.output_dir / "station_map_direct_stec.csv", index=False)

    diff_map = station_diff_map_table(positioning_frame, station_coords)
    diff_map.to_csv(args.output_dir / "station_map_diff.csv", index=False)

    # 2. Latitude stratification, geographic and geomagnetic side by side.
    for lat_kind, lat_column in (("geographic", "lat"), ("geomagnetic", "sm_lat")):
        strat = latitude_stratification(geo_frame, lat_column)
        strat.to_csv(
            args.output_dir / f"latitude_stratification_{lat_kind}.csv", index=False
        )

        strat_diff = latitude_stratification_diff(geo_frame, lat_column)
        strat_diff.to_csv(
            args.output_dir / f"latitude_stratification_diff_{lat_kind}.csv",
            index=False,
        )

        # 3. Cross with the recovered/original population split.
        coverage = pd.read_csv(args.coverage_csv)
        concentration = recovered_concentration_by_latitude(
            coverage, recovered, station_coords, lat_column
        )
        concentration.to_csv(
            args.output_dir / f"population_concentration_by_latitude_{lat_kind}.csv",
            index=False,
        )

        pop_diff = population_latitude_diff(geo_frame, lat_column)
        pop_diff.to_csv(
            args.output_dir / f"population_latitude_diff_{lat_kind}.csv", index=False
        )

        print(f"\n=== Latitude stratification ({lat_kind}): Direct STEC vs IGS GIM ===")
        print(
            strat_diff[
                ["lat_bin", "n", "mean_improvement_pct", "median_improvement_pct"]
            ].to_string(index=False)
        )
        print(f"\n=== Recovered-station-day concentration by latitude ({lat_kind}) ===")
        print(concentration.to_string(index=False))

    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
