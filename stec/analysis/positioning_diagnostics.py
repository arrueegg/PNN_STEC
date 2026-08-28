"""Positioning diagnostics for the coverage-recovery result (owner request, 2026-08-28).

Not a declared pipeline stage yet - the owner wants to see this output before deciding
whether/how it becomes part of Table 5 or a new appendix. See the module docstring of
`stec/pipeline/stages.py` for what "declared stage" means; nothing here is registered
there.

Context: the station-recovery sweep finished (10,598 of 10,853 station-days now solved by
all four methods, up from 8,195 - see CLAUDE.md's coverage-recovery row). An ad-hoc shell
check on the recovered data found the headline improvement is highly sensitive to how
outliers are handled and to whether the mean or the median is reported, and that the
model apparently loses to the IGS GIM on the newly-recovered, geometry-only station-days.
This module turns those ad-hoc numbers into a reproducible analysis with four sections,
each answering one of the leads:

1. `daily_timeseries` - per-DOY mean and median 3D error, one line per method, with a
   `storm` flag from `storm_stratification`'s own daily-Dst classification.
2. `per_station_summary` - per-station mean/median 3D error per method, station-day
   counts, and the Direct-STEC-minus-GIM delta that makes losing stations visible.
3. `outlier_threshold_counts` / `outlier_headline_sensitivity` / `outlier_concentration` -
   how many station-days exceed 5/10/20/50 m per method, what fraction of each method's
   total error mass they carry, whether they concentrate in a few stations or days, and
   how the mean/median improvement over GIM moves as the threshold moves. The project's
   own rule (`stec.positioning.metrics.OUTLIER_3D_RMS_M` = 10 m) is one point on this
   curve, not assumed correct in advance.
4. `attach_population` / `population_split` - Direct STEC vs GIM (and all four methods),
   split into station-days recovered by `positioning/geometry/build_recovered_day.py`
   (no real STEC-database entry that day - see `load_recovered_station_days`) versus
   station-days the original STEC database always covered. This replaces the ad-hoc
   proxy (pre-sweep coverage) with real `data/recovered_stec_db/` membership: a
   station-day is "recovered" only if that exact (doy, station) pair appears in a
   geometry-only file under `data/recovered_stec_db/`, not inferred from which
   station-days used to be missing.

Source data: the per-station-day table `positioning_summary.canonical_positioning_summary()`
resolves - `multiday_results/analyses/positioning_coverage/rebuilt/multiday_summary.csv`
as of 2026-08-28. No `.pos` files are re-read here.

Every ratio this module reports is computed as both mean-of-station-days and
median-of-station-days, side by side - Table 5 reports only the mean
(`stec.positioning.metrics.summarise`), and the point of this module is to show whether
that choice matters.

Usage::

    python -m stec.analysis.positioning_diagnostics
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..config import paths
from ..positioning import metrics as pm
from .positioning_summary import (
    METHOD_ORDER,
    PAPER_METHODS,
    canonical_positioning_summary,
)
from .storm_stratification import STORM_DST_THRESHOLD_NT, load_daily_geomagnetic_indices

logger = logging.getLogger(__name__)

STEC_LABEL = "Direct STEC"
GIM_LABEL = "IGS GIM + Mapping"

# Thresholds probed by the outlier-sensitivity check, in addition to "no exclusion at
# all". stec.positioning.metrics.OUTLIER_3D_RMS_M (10 m) is one of these, not a special
# case - it is the project's existing rule, reused rather than redefined.
OUTLIER_THRESHOLDS_M: tuple[float, ...] = (5.0, 10.0, 20.0, 50.0)

# Geometry-only recovered days (positioning/geometry/build_recovered_day.py), one HDF5
# per (year, doy) holding only the stations that day's STEC database was missing. Not a
# `stec.config.paths` constant yet - this is the only reader of it so far.
RECOVERED_STEC_DB_ROOT = paths.REPO_DATA / "recovered_stec_db"

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_diagnostics", rebuilt=True)


def load_positioning_table(path: Path) -> pd.DataFrame:
    """One row per (station, doy, method), restricted to the paper's four iono-weighted
    methods and labelled with their display names."""
    frame = pd.read_csv(
        path,
        usecols=[
            "station",
            "method",
            "year",
            "doy",
            "error_3d_rms",
            "error_2d_rms",
            "u_rms",
        ],
    )
    frame = frame[frame["method"].isin(PAPER_METHODS)].copy()
    frame["Method"] = frame["method"].map(PAPER_METHODS)
    frame["station"] = frame["station"].str.upper()
    return frame


def load_recovered_station_days(root: Path = RECOVERED_STEC_DB_ROOT) -> pd.DataFrame:
    """(year, doy, station) triples whose row in the recovered tree exists because the
    station was absent from the real STEC database that day - the ground truth for the
    population split, read from the actual recovery mechanism rather than inferred from
    any positioning output.

    `build_recovered_day.py` writes one file per (year, doy) under
    `data/recovered_stec_db/<year>/<doy>/`, holding only the stations that invocation
    rebuilt from RINEX - a station's presence in this tree for that day is exactly what
    "geometry-only, no real STEC-database entry" means (see that module's docstring).
    Returns an empty frame with a warning, not an error, if the tree does not exist -
    the population split then degrades to "everything is 'original'" rather than
    crashing, matching `storm_stratification`'s handling of a missing OMNI archive.
    """
    if not root.exists():
        logger.warning(
            f"{root} not found - population split will have no 'recovered' rows"
        )
        return pd.DataFrame(columns=["year", "doy", "station"])

    rows = []
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        for doy_dir in sorted(year_dir.iterdir()):
            if not doy_dir.is_dir():
                continue
            files = sorted(doy_dir.glob("*.h5"))
            if not files:
                continue
            with h5py.File(files[0], "r") as handle:
                dataset = handle[year_dir.name][doy_dir.name]["all_data"]
                stations = np.unique(dataset.fields("station")[:])
            for raw in stations:
                station = raw.decode() if isinstance(raw, bytes) else raw
                rows.append(
                    {
                        "year": int(year_dir.name),
                        "doy": int(doy_dir.name),
                        "station": station.upper(),
                    }
                )
    frame = pd.DataFrame(rows, columns=["year", "doy", "station"]).drop_duplicates()
    logger.info(f"{len(frame):,} recovered (year, doy, station) triples under {root}")
    return frame


# --------------------------------------------------------------------------
# 1. Per-DOY timeseries
# --------------------------------------------------------------------------


def daily_timeseries(
    frame: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M
) -> pd.DataFrame:
    """Per (doy, Method): mean, median and station-day count of `error_3d_rms`, after
    the project's outlier rule (Table 5's convention)."""
    kept = pm.exclude_outlier_station_days(frame, threshold)
    return (
        kept.groupby(["doy", "Method"])["error_3d_rms"]
        .agg(mean="mean", median="median", station_days="count")
        .reset_index()
    )


def attach_storm_flag(
    daily: pd.DataFrame, year: int = 2024, swi_path: Path = paths.OMNI_INDICES
) -> pd.DataFrame:
    """Add a `storm` column using the same daily-minimum-Dst rule
    `storm_stratification` uses for the R2.7 table, so "storm" means the same thing in
    both places."""
    indices = load_daily_geomagnetic_indices(year, swi_path)
    storm_doys = set(
        indices.loc[indices["dst_min"] <= STORM_DST_THRESHOLD_NT, "doy"].tolist()
    )
    out = daily.copy()
    out["storm"] = out["doy"].isin(storm_doys)
    return out


# --------------------------------------------------------------------------
# 2. Per-station summary
# --------------------------------------------------------------------------


def per_station_summary(
    frame: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M
) -> pd.DataFrame:
    """One row per station: mean/median/count of `error_3d_rms` for Direct STEC and IGS
    GIM, plus their difference (positive = Direct STEC worse than GIM at that station),
    sorted worst-first so a losing station is immediately visible. Station-day counts
    are always alongside the error values so a 3-day station is not read like a 200-day
    one."""
    kept = pm.exclude_outlier_station_days(frame, threshold)
    grouped = (
        kept.groupby(["station", "Method"])["error_3d_rms"]
        .agg(mean="mean", median="median", station_days="count")
        .reset_index()
    )
    stec = grouped[grouped["Method"] == STEC_LABEL].set_index("station")
    gim = grouped[grouped["Method"] == GIM_LABEL].set_index("station")

    out = pd.DataFrame(
        {
            "stec_station_days": stec["station_days"],
            "gim_station_days": gim["station_days"],
            "stec_mean_m": stec["mean"],
            "gim_mean_m": gim["mean"],
            "stec_median_m": stec["median"],
            "gim_median_m": gim["median"],
        }
    )
    out["diff_mean_m"] = out["stec_mean_m"] - out["gim_mean_m"]
    out["diff_median_m"] = out["stec_median_m"] - out["gim_median_m"]
    return out.sort_values("diff_mean_m", ascending=False).reset_index()


# --------------------------------------------------------------------------
# 3. Outlier characterisation
# --------------------------------------------------------------------------


def outlier_threshold_counts(
    frame: pd.DataFrame,
    thresholds: tuple[float, ...] = OUTLIER_THRESHOLDS_M,
    group_cols: tuple[str, ...] = ("Method",),
) -> pd.DataFrame:
    """For each threshold and each group (default: per method), how many station-days
    exceed it and what fraction of that group's *total, unfiltered* error_3d_rms sum
    ("error mass") those station-days carry. `group_cols` can be widened (e.g. to
    `("population", "Method")`) to answer the same question within a subpopulation."""
    group_cols = list(group_cols)
    rows = []
    for keys, sub in frame.groupby(group_cols, observed=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        total_station_days = len(sub)
        total_mass = sub["error_3d_rms"].sum()
        for threshold in thresholds:
            exceeding = sub[sub["error_3d_rms"] > threshold]
            mass = exceeding["error_3d_rms"].sum()
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "threshold_m": threshold,
                    "n_exceeding": len(exceeding),
                    "total_station_days": total_station_days,
                    "pct_station_days_exceeding": (
                        100 * len(exceeding) / total_station_days
                        if total_station_days
                        else float("nan")
                    ),
                    "error_mass_fraction_pct": (
                        100 * mass / total_mass if total_mass else float("nan")
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def outlier_headline_sensitivity(
    frame: pd.DataFrame, thresholds: tuple[float, ...] = OUTLIER_THRESHOLDS_M
) -> pd.DataFrame:
    """Direct STEC vs IGS GIM headline improvement (mean AND median) as the exclusion
    threshold moves from "no exclusion at all" through each of `thresholds`. Answers
    "how much does the outlier rule move the headline number" directly, rather than
    leaving it to be discovered ad hoc."""
    points: list[tuple[str, float | None]] = [("none", None)]
    points += [(f"{t:g}m", t) for t in thresholds]

    rows = []
    for label, threshold in points:
        kept = (
            frame
            if threshold is None
            else pm.exclude_outlier_station_days(frame, threshold)
        )
        stec = kept.loc[kept["Method"] == STEC_LABEL, "error_3d_rms"]
        gim = kept.loc[kept["Method"] == GIM_LABEL, "error_3d_rms"]
        rows.append(
            {
                "threshold_label": label,
                "threshold_m": threshold if threshold is not None else float("inf"),
                "n_stec": len(stec),
                "n_gim": len(gim),
                "stec_mean_m": stec.mean(),
                "gim_mean_m": gim.mean(),
                "mean_improvement_pct": 100 * (gim.mean() - stec.mean()) / gim.mean(),
                "stec_median_m": stec.median(),
                "gim_median_m": gim.median(),
                "median_improvement_pct": 100
                * (gim.median() - stec.median())
                / gim.median(),
            }
        )
    return pd.DataFrame(rows)


def outlier_concentration(
    frame: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Station-days with `error_3d_rms` above `threshold`, broken down by station and by
    day, plus a per-method summary of how concentrated they are (share held by the
    `top_n` worst stations). Returns `(by_station, by_day, concentration_summary)`."""
    outliers = frame[frame["error_3d_rms"] > threshold]

    by_station = (
        outliers.groupby(["Method", "station"])
        .size()
        .rename("n_outlier_station_days")
        .reset_index()
        .sort_values(["Method", "n_outlier_station_days"], ascending=[True, False])
        .reset_index(drop=True)
    )
    by_day = (
        outliers.groupby(["Method", "doy"])
        .size()
        .rename("n_outlier_stations")
        .reset_index()
        .sort_values(["Method", "n_outlier_stations"], ascending=[True, False])
        .reset_index(drop=True)
    )

    summary_rows = []
    for method in METHOD_ORDER:
        sub = by_station[by_station["Method"] == method]
        total = sub["n_outlier_station_days"].sum()
        top_share = sub.nlargest(top_n, "n_outlier_station_days")[
            "n_outlier_station_days"
        ].sum()
        summary_rows.append(
            {
                "Method": method,
                "total_outlier_station_days": int(total),
                "n_distinct_stations": len(sub),
                f"top_{top_n}_station_share_pct": (
                    100 * top_share / total if total else float("nan")
                ),
            }
        )
    concentration_summary = pd.DataFrame(summary_rows)
    return by_station, by_day, concentration_summary


# --------------------------------------------------------------------------
# 4. Population split (recovered vs original)
# --------------------------------------------------------------------------


def attach_population(frame: pd.DataFrame, recovered: pd.DataFrame) -> pd.DataFrame:
    """Add a `population` column: "recovered" if (doy, station) appears in
    `load_recovered_station_days()`'s output, "original" otherwise. Joined on
    (doy, station) only, not year - the paper's positioning runs are all 2024, and the
    recovered tree may not always carry a matching `year` column value type."""
    if recovered.empty:
        out = frame.copy()
        out["population"] = "original"
        return out
    keys = recovered[["doy", "station"]].drop_duplicates()
    keys["population"] = "recovered"
    merged = frame.merge(keys, on=["doy", "station"], how="left")
    merged["population"] = merged["population"].fillna("original")
    return merged


def population_split(
    frame_with_population: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (population, Method): N/mean/median (`by_method`), and Direct STEC vs GIM
    improvement within each population, mean AND median (`stec_vs_gim`)."""
    kept = pm.exclude_outlier_station_days(frame_with_population, threshold)

    by_method = (
        kept.groupby(["population", "Method"])["error_3d_rms"]
        .agg(station_days="count", mean_m="mean", median_m="median")
        .reset_index()
    )

    rows = []
    for population in ("original", "recovered"):
        sub = kept[kept["population"] == population]
        stec = sub.loc[sub["Method"] == STEC_LABEL, "error_3d_rms"]
        gim = sub.loc[sub["Method"] == GIM_LABEL, "error_3d_rms"]
        rows.append(
            {
                "population": population,
                "n_stec": len(stec),
                "n_gim": len(gim),
                "stec_mean_m": stec.mean(),
                "gim_mean_m": gim.mean(),
                "mean_improvement_pct": 100 * (gim.mean() - stec.mean()) / gim.mean(),
                "stec_median_m": stec.median(),
                "gim_median_m": gim.median(),
                "median_improvement_pct": 100
                * (gim.median() - stec.median())
                / gim.median(),
            }
        )
    return by_method, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def overall_summary(
    frame: pd.DataFrame, threshold: float = pm.OUTLIER_3D_RMS_M
) -> pd.DataFrame:
    """The four methods' own Table-5-style summary (mean/median of per-station-day RMSE,
    10 m rule applied), for context alongside the four sections above."""
    kept = pm.exclude_outlier_station_days(frame, threshold)
    return pm.summarise(kept, ["Method"]).reindex(METHOD_ORDER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument("--recovered-root", type=Path, default=RECOVERED_STEC_DB_ROOT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi-path", type=Path, default=paths.OMNI_INDICES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_positioning_table(args.summary_path)
    logger.info(
        f"{args.summary_path}: {len(frame):,} rows, {frame['Method'].nunique()} methods"
    )

    overall = overall_summary(frame)
    overall.to_csv(args.output_dir / "overall_summary.csv")

    daily = attach_storm_flag(daily_timeseries(frame), args.year, args.swi_path)
    daily.to_csv(args.output_dir / "daily_timeseries.csv", index=False)

    per_station = per_station_summary(frame)
    per_station.to_csv(args.output_dir / "per_station_summary.csv", index=False)

    threshold_counts = outlier_threshold_counts(frame)
    threshold_counts.to_csv(
        args.output_dir / "outlier_threshold_counts.csv", index=False
    )

    headline_sensitivity = outlier_headline_sensitivity(frame)
    headline_sensitivity.to_csv(
        args.output_dir / "outlier_headline_sensitivity.csv", index=False
    )

    by_station, by_day, concentration_summary = outlier_concentration(frame)
    by_station.to_csv(args.output_dir / "outlier_by_station.csv", index=False)
    by_day.to_csv(args.output_dir / "outlier_by_day.csv", index=False)
    concentration_summary.to_csv(
        args.output_dir / "outlier_concentration_summary.csv", index=False
    )

    recovered = load_recovered_station_days(args.recovered_root)
    recovered.to_csv(args.output_dir / "recovered_station_days.csv", index=False)
    with_population = attach_population(frame, recovered)

    population_by_method, population_stec_vs_gim = population_split(with_population)
    population_by_method.to_csv(
        args.output_dir / "population_split_by_method.csv", index=False
    )
    population_stec_vs_gim.to_csv(
        args.output_dir / "population_split_stec_vs_gim.csv", index=False
    )

    population_outlier_counts = outlier_threshold_counts(
        with_population, group_cols=("population", "Method")
    )
    population_outlier_counts.to_csv(
        args.output_dir / "outlier_counts_by_population.csv", index=False
    )

    print(
        "=== Overall (Table-5 convention: 10 m rule, mean/median of station-days) ==="
    )
    print(overall.to_string())
    print("\n=== Outlier threshold sensitivity: Direct STEC vs IGS GIM ===")
    print(headline_sensitivity.to_string(index=False))
    print("\n=== Population split: Direct STEC vs IGS GIM ===")
    print(population_stec_vs_gim.to_string(index=False))
    logger.info(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
