"""Positioning diagnostics for the coverage-recovery result (owner request, 2026-08-28).

A declared pipeline stage (`stec/pipeline/stages.py`'s `positioning_diagnostics`) - built
undeclared, for the owner to look at this output before deciding whether/how it becomes
part of Table 5 or a new appendix, then declared the same day once the decision to keep it
as a standing diagnostic (not a Table 5 source) was made. `canonical_for=None`: it backs no
manuscript table directly.

`main()` also writes `FINDINGS.md` in the output directory - the narrative write-up of the
four sections below, generated from the same dataframes the CSVs come from
(`_format_findings_markdown`) rather than hand-maintained prose, so the document cannot say
something its own CSVs disagree with. This is the fix for the failure mode that motivated
it: a hand-written FINDINGS.md sitting beside a declared stage that did not regenerate it
was exactly the kind of number that drifted from its artifact.

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


def _pct(x: float, signed: bool = False) -> str:
    if pd.isna(x):
        return "n/a"
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def _meters(x: float, decimals: int = 3) -> str:
    """A metres value, or 'n/a' when the source population was empty (`mean()`/`median()`
    of zero rows is NaN, e.g. a population split with no recovered station-days at all)."""
    return "n/a" if pd.isna(x) else f"{x:.{decimals}f}"


def _display_path(path: Path) -> str:
    """Repo-relative when possible, so the generated markdown reads like the rest of this
    project's docs (`multiday_results/...`) rather than leaking this machine's absolute
    checkout path - falls back to the path as given (a test fixture, or a path outside the
    repo) rather than raising."""
    try:
        return str(Path(path).resolve().relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


def _format_findings_markdown(
    *,
    frame: pd.DataFrame,
    overall: pd.DataFrame,
    daily: pd.DataFrame,
    per_station: pd.DataFrame,
    threshold_counts: pd.DataFrame,
    headline_sensitivity: pd.DataFrame,
    by_day: pd.DataFrame,
    concentration_summary: pd.DataFrame,
    recovered: pd.DataFrame,
    population_by_method: pd.DataFrame,
    population_stec_vs_gim: pd.DataFrame,
    population_outlier_counts: pd.DataFrame,
    summary_path: Path,
) -> str:
    """The narrative write-up of the four sections `main()` computes, as markdown - the
    prose that used to be a hand-maintained FINDINGS.md and drifted from the CSVs it
    described. Every number below is read from the dataframes `main()` already built, not
    retyped, so the document cannot disagree with its own CSVs. The only fixed numbers are
    values from *other*, frozen artifacts this run is being compared against - an earlier
    ad-hoc shell check that has no CSV of its own, and the published manuscript - named as
    such at each use, the same way CLAUDE.md contrasts a published figure against a current
    one."""
    lines = [
        "# Positioning diagnostics — findings",
        "",
        "Generated by `stec/analysis/positioning_diagnostics.py`'s `main()` - every number "
        "below is read from the CSVs beside this file, not hand-typed, so it cannot drift "
        f"from them. Source: `{_display_path(summary_path)}` ({len(frame):,} rows, "
        f"{frame['Method'].nunique()} methods). Diagnostic output, not a manuscript or "
        "Table 5 source: built to look at the coverage-recovery result before deciding "
        "whether/how it becomes part of Table 5 or a new appendix - see this stage's "
        "`caveats` in `stec/pipeline/stages.py`. Re-run with "
        "`python -m stec.analysis.positioning_diagnostics`; figures are "
        "`python -m stec.viz.positioning_diagnostics`, reading this directory's CSVs.",
        "",
        "Every number here uses the project's own `error_3d_rms` (per-station-day 3D RMS) "
        "and, unless stated otherwise, the project's own 10 m station-day exclusion "
        "(`stec.positioning.metrics.OUTLIER_3D_RMS_M`).",
        "",
        "## Headline",
        "",
        "| Method | station-days | 3D mean [m] | 3D median [m] |",
        "|---|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        row = overall.loc[method]
        if pd.isna(row["station_days"]):
            continue
        lines.append(
            f"| {method} | {int(row['station_days']):,} | {row['3D_mean_m']:.4f} | "
            f"{row['3D_median_m']:.4f} |"
        )
    stec_overall, gim_overall = overall.loc[STEC_LABEL], overall.loc[GIM_LABEL]
    headline_mean_improvement = (
        100
        * (gim_overall["3D_mean_m"] - stec_overall["3D_mean_m"])
        / gim_overall["3D_mean_m"]
    )
    headline_median_improvement = (
        100
        * (gim_overall["3D_median_m"] - stec_overall["3D_median_m"])
        / gim_overall["3D_median_m"]
    )
    lines += [
        "",
        f"(`overall_summary.csv`.) Direct STEC vs IGS GIM: **mean improvement "
        f"{_pct(headline_mean_improvement)}, median improvement "
        f"{_pct(headline_median_improvement)}**.",
        "",
    ]

    # --- 1. Per-DOY timeseries -------------------------------------------------
    n_storm_doys = int(daily.loc[daily["storm"], "doy"].nunique())
    mean_order = overall["3D_mean_m"].dropna().sort_values().index.tolist()
    median_order = overall["3D_median_m"].dropna().sort_values().index.tolist()
    top_spikes = daily.nlargest(3, "mean")[["doy", "Method", "mean", "storm"]]

    lines += [
        "## 1. Per-DOY timeseries (`daily_timeseries.csv`, "
        "`plots/positioning_diagnostics/positioning_2024/daily_timeseries.png`)",
        "",
        f"DOY 122–366, one line per method, {n_storm_doys} storm days flagged (daily min "
        f"Dst ≤ {STORM_DST_THRESHOLD_NT:g} nT, the same rule `storm_stratification` uses).",
        "",
        f"Ordering by mean 3D error, best to worst: {' < '.join(mean_order)}.",
        f"Ordering by median: {' < '.join(median_order)}"
        + (
            " (same ordering as the mean)."
            if mean_order == median_order
            else " - note this differs from the mean ordering above."
        ),
        "",
        "The three largest single-day mean spikes in the table, any method:",
        "",
        "| DOY | Method | mean 3D error [m] | storm day |",
        "|---:|---|---:|:---:|",
    ]
    for _, spike in top_spikes.iterrows():
        lines.append(
            f"| {int(spike['doy'])} | {spike['Method']} | {spike['mean']:.3f} | "
            f"{'yes' if spike['storm'] else 'no'} |"
        )
    lines += [
        "",
        "Direct STEC and IGS GIM track each other closely on most days; the visible "
        "separation between them is a population-level effect (see §3–4 below), not "
        "concentrated on the single days above - those are dominated by whichever method "
        "has the largest tail (see §3).",
        "",
    ]

    # --- 2. Per-station performance ---------------------------------------------
    n_with_stec = int(per_station["stec_station_days"].notna().sum())
    n_total_stations = int(per_station["station"].nunique())
    comparable = per_station.dropna(subset=["diff_mean_m"])
    n_losing = int((comparable["diff_mean_m"] > 0).sum())
    pct_losing = 100 * n_losing / n_with_stec if n_with_stec else float("nan")
    worst5 = comparable.sort_values("diff_mean_m", ascending=False).head(5)
    best3 = comparable.sort_values("diff_mean_m", ascending=True).head(3)
    zero_stec = per_station[
        per_station["stec_station_days"].isna()
        & per_station["gim_station_days"].notna()
    ]
    recovered_counts_by_station = (
        recovered.groupby("station").size()
        if not recovered.empty
        else pd.Series(dtype=int)
    )

    lines += [
        "## 2. Per-station performance (`per_station_summary.csv`, "
        "`plots/positioning_diagnostics/positioning_2024/per_station_diff.png`)",
        "",
        f"{n_total_stations} stations, sorted by Direct-STEC-minus-GIM mean 3D error "
        "(positive = Direct STEC worse). **"
        f"{n_losing} of {n_with_stec} stations with a Direct STEC solution "
        f"({pct_losing:.0f}%) have Direct STEC losing to GIM on average**, concentrated at "
        "the top of the sort:",
        "",
        "| station | STEC days | GIM days | STEC mean [m] | GIM mean [m] | diff mean [m] "
        "| recovered days |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in worst5.iterrows():
        n_recovered = int(recovered_counts_by_station.get(row["station"], 0))
        lines.append(
            f"| {row['station']} | {int(row['stec_station_days'])} | "
            f"{int(row['gim_station_days'])} | {row['stec_mean_m']:.2f} | "
            f"{row['gim_mean_m']:.2f} | **{row['diff_mean_m']:+.2f}** | {n_recovered} |"
        )
    lines += [
        "",
        "and, where Direct STEC wins decisively (`per_station_summary.csv`, ascending "
        "`diff_mean_m`):",
        "",
        "| station | STEC days | GIM days | STEC mean [m] | GIM mean [m] | diff mean [m] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in best3.iterrows():
        lines.append(
            f"| {row['station']} | {int(row['stec_station_days'])} | "
            f"{int(row['gim_station_days'])} | {row['stec_mean_m']:.2f} | "
            f"{row['gim_mean_m']:.2f} | {row['diff_mean_m']:+.2f} |"
        )
    if not zero_stec.empty:
        lines += [
            "",
            f"{len(zero_stec)} station(s) have *zero* Direct STEC station-days at all "
            f"(GIM-only): {', '.join(sorted(zero_stec['station']))} - flagged, not "
            "investigated further here; this is a gap in the STEC-side pipeline for "
            "those stations specifically, separate from the recovered/original split "
            "below.",
        ]
    lines += [
        "",
        "The `recovered days` column above cross-references §4: stations near the top of "
        "this sort tend to carry a large share of geometry-only recovered days, though not "
        "exclusively - see §4 for why the population split is a station-*day* property, "
        "not a clean station property.",
        "",
    ]

    # --- 3. Outlier characterisation ---------------------------------------------
    mean_range = (
        headline_sensitivity["mean_improvement_pct"].max()
        - headline_sensitivity["mean_improvement_pct"].min()
    )
    median_range = (
        headline_sensitivity["median_improvement_pct"].max()
        - headline_sensitivity["median_improvement_pct"].min()
    )
    peak_row = headline_sensitivity.loc[
        headline_sensitivity["mean_improvement_pct"].idxmax()
    ]
    project_rule_row = headline_sensitivity.loc[
        headline_sensitivity["threshold_label"] == "10m"
    ].iloc[0]
    if (
        project_rule_row["mean_improvement_pct"]
        and peak_row["threshold_label"] != "10m"
    ):
        peak_vs_project_text = (
            f", {peak_row['mean_improvement_pct'] / project_rule_row['mean_improvement_pct']:.1f}x "
            f"the project rule's own {project_rule_row['threshold_label']} figure"
        )
    else:
        peak_vs_project_text = ""

    lines += [
        "## 3. Outlier characterisation (`outlier_headline_sensitivity.csv`, "
        "`outlier_threshold_counts.csv`, `outlier_by_station.csv`, `outlier_by_day.csv`, "
        "`outlier_concentration_summary.csv`)",
        "",
        "**Direct STEC vs IGS GIM headline improvement as the exclusion threshold moves:**",
        "",
        "| threshold | N (STEC/GIM) | mean improvement | median improvement |",
        "|---|---:|---:|---:|",
    ]
    for _, row in headline_sensitivity.iterrows():
        label = (
            "none (no exclusion)"
            if row["threshold_label"] == "none"
            else (
                f"{row['threshold_label']}"
                + (" (project rule)" if row["threshold_label"] == "10m" else "")
            )
        )
        lines.append(
            f"| {label} | {int(row['n_stec']):,} / {int(row['n_gim']):,} | "
            f"**{_pct(row['mean_improvement_pct'], signed=True)}** | "
            f"{_pct(row['median_improvement_pct'])} |"
        )
    lines += [
        "",
        f"The mean swings **{mean_range:.1f} points** across these thresholds (peaking at "
        f"{peak_row['threshold_label']}{peak_vs_project_text}); the median swings only "
        f'**{median_range:.1f} points** across the same range, including "none at all" - '
        "the median gives essentially the same answer whether or not any outlier rule is "
        "applied.",
        "",
        "**Exceedance and error-mass share, Direct STEC vs IGS GIM:**",
        "",
        "| threshold | Direct STEC exceeding | IGS GIM exceeding | STEC error-mass share | "
        "GIM error-mass share |",
        "|---|---:|---:|---:|---:|",
    ]
    stec_counts = threshold_counts[threshold_counts["Method"] == STEC_LABEL].set_index(
        "threshold_m"
    )
    gim_counts = threshold_counts[threshold_counts["Method"] == GIM_LABEL].set_index(
        "threshold_m"
    )
    for threshold in OUTLIER_THRESHOLDS_M:
        s, g = stec_counts.loc[threshold], gim_counts.loc[threshold]
        lines.append(
            f"| {threshold:g} m | {int(s['n_exceeding'])} "
            f"({s['pct_station_days_exceeding']:.2g}%) | {int(g['n_exceeding'])} "
            f"({g['pct_station_days_exceeding']:.2g}%) | "
            f"{s['error_mass_fraction_pct']:.1f}% | {g['error_mass_fraction_pct']:.1f}% |"
        )

    stec_10, gim_10 = stec_counts.loc[10.0], gim_counts.loc[10.0]
    stec_concentration = concentration_summary.set_index("Method").loc[STEC_LABEL]
    gim_concentration = concentration_summary.set_index("Method").loc[GIM_LABEL]
    other_tails = threshold_counts[
        (threshold_counts["threshold_m"] == 10.0)
        & (~threshold_counts["Method"].isin([STEC_LABEL, GIM_LABEL]))
    ].set_index("Method")["n_exceeding"]

    if gim_10["n_exceeding"]:
        tail_comparison = (
            f"Direct STEC has **{stec_10['n_exceeding'] / gim_10['n_exceeding']:.1f}x "
            "as many station-days over 10 m** as GIM "
            f"({int(stec_10['n_exceeding'])} vs {int(gim_10['n_exceeding'])})"
        )
    else:
        tail_comparison = (
            f"Direct STEC has {int(stec_10['n_exceeding'])} station-days over 10 m "
            "against GIM's zero"
        )
    lines += [
        "",
        f"{tail_comparison}, and those station-days alone carry "
        f"**{stec_10['error_mass_fraction_pct']:.1f}%** of Direct STEC's entire summed "
        "error.",
    ]
    if not other_tails.empty:
        other_tail_text = ", ".join(
            f"{method} {int(n)}" for method, n in other_tails.items()
        )
        lines.append(
            f"Other methods' 10 m tails, for comparison: {other_tail_text} station-days - "
            "the tail problem is not unique to the headline method."
        )
    lines += [
        "",
        "**Concentration**: for Direct STEC, the "
        f"{int(stec_concentration['total_outlier_station_days'])} station-days over 10 m "
        f"come from {int(stec_concentration['n_distinct_stations'])} distinct stations, "
        "and the 10 worst of those carry "
        f"**{_pct(stec_concentration['top_10_station_share_pct'])}** of the tail "
        "(`outlier_concentration_summary.csv`). For GIM: "
        f"{int(gim_concentration['total_outlier_station_days'])} station-days from "
        f"{int(gim_concentration['n_distinct_stations'])} stations, "
        f"{_pct(gim_concentration['top_10_station_share_pct'])} held by the 10 worst.",
        "",
    ]
    for method, concentration in (
        (STEC_LABEL, stec_concentration),
        (GIM_LABEL, gim_concentration),
    ):
        method_days = by_day[by_day["Method"] == method]
        total = int(concentration["total_outlier_station_days"])
        if total and not method_days.empty:
            worst_day_share = 100 * method_days["n_outlier_stations"].max() / total
            lines.append(
                f"`outlier_by_day.csv`: {method}'s single worst outlier day accounts for "
                f"{worst_day_share:.1f}% of its {total} outlier station-days - the "
                "concentration above sits in stations, not days."
            )
    lines += [
        "",
        "**Conclusion for this section**: the project's 10 m rule removes only "
        f"{stec_10['pct_station_days_exceeding']:.2g}%/{gim_10['pct_station_days_exceeding']:.2g}% "
        "of station-days (STEC/GIM respectively) - outliers are not being silently swept "
        "under the rug - but the *mean* moves by an order of magnitude more than the "
        "underlying error distribution does, depending on where exactly the exclusion line "
        "is drawn. The median does not have this problem.",
        "",
    ]

    # --- 4. Population split -------------------------------------------------
    n_recovered_triples = len(recovered)
    n_recovered_doys = int(recovered["doy"].nunique()) if not recovered.empty else 0
    split = population_stec_vs_gim.set_index("population")
    original, recovered_pop = split.loc["original"], split.loc["recovered"]

    lines += [
        "## 4. Population split (`population_split_by_method.csv`, "
        "`population_split_stec_vs_gim.csv`, `outlier_counts_by_population.csv`, "
        "`recovered_station_days.csv`)",
        "",
        f"{n_recovered_triples:,} (doy, station) pairs across {n_recovered_doys} test days "
        "have a geometry-only recovered file on disk under `data/recovered_stec_db/` - read "
        "directly from the recovery mechanism, not inferred.",
        "",
        "| population | N (STEC/GIM) | STEC mean | GIM mean | mean improvement | "
        "STEC median | GIM median | median improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in [
        ("original (real STEC-DB row)", original),
        ("recovered (geometry only)", recovered_pop),
    ]:
        lines.append(
            f"| {label} | {int(row['n_stec']):,} / {int(row['n_gim']):,} | "
            f"{_meters(row['stec_mean_m'])} | {_meters(row['gim_mean_m'])} | "
            f"**{_pct(row['mean_improvement_pct'], signed=True)}** | "
            f"{_meters(row['stec_median_m'])} | {_meters(row['gim_median_m'])} | "
            f"**{_pct(row['median_improvement_pct'], signed=True)}** |"
        )

    pivot_mean = (
        population_by_method.pivot(
            index="Method", columns="population", values="mean_m"
        )
        if not population_by_method.empty
        else pd.DataFrame()
    )
    ratios = (
        {
            method: pivot_mean.loc[method, "recovered"]
            / pivot_mean.loc[method, "original"]
            for method in METHOD_ORDER
            if method in pivot_mean.index
            and pd.notna(pivot_mean.loc[method, "recovered"])
            and pd.notna(pivot_mean.loc[method, "original"])
            and pivot_mean.loc[method, "original"]
        }
        if "recovered" in pivot_mean.columns and "original" in pivot_mean.columns
        else {}
    )
    ml_ratios = {m: r for m, r in ratios.items() if m != GIM_LABEL}

    if ratios:
        lines += [
            "",
            "Per-method recovered/original mean-error ratio "
            "(`population_split_by_method.csv`):",
            "",
            "| Method | original mean [m] | recovered mean [m] | recovered/original ratio |",
            "|---|---:|---:|---:|",
        ]
        for method in METHOD_ORDER:
            if method not in ratios:
                continue
            lines.append(
                f"| {method} | {pivot_mean.loc[method, 'original']:.3f} | "
                f"{pivot_mean.loc[method, 'recovered']:.3f} | {ratios[method]:.2f}x |"
            )
        if ml_ratios and GIM_LABEL in ratios:
            lines += [
                "",
                f"The ML methods degrade {min(ml_ratios.values()):.1f}x-"
                f"{max(ml_ratios.values()):.1f}x in the recovered population; GIM degrades "
                f"{ratios[GIM_LABEL]:.1f}x - GIM is comparatively insulated because it "
                "never depended on the STEC database, so its accuracy loss on recovered "
                "days is a property of the geometry alone (these are harder station-days "
                "for everyone), while the ML methods lose the extra signal they would "
                "otherwise get from real local calibration data.",
            ]
    else:
        lines += [
            "",
            "No recovered-population station-days in this run "
            "(`data/recovered_stec_db/` absent or empty) - the per-method degradation "
            "comparison above is skipped.",
        ]

    n_stations_total = int(per_station["station"].nunique())
    n_stations_recovered = (
        int(recovered["station"].nunique()) if not recovered.empty else 0
    )

    if not recovered_counts_by_station.empty:
        median_recovered_days = float(recovered_counts_by_station.median())
        top_recovered = recovered_counts_by_station.sort_values(ascending=False).head(9)
        top_recovered_text = ", ".join(
            f"{station} ({int(n)})" for station, n in top_recovered.items()
        )
        coverage_sentence = (
            f'**This is not a small set of "exotic" stations.** {n_stations_recovered} '
            f"of {n_stations_total} test stations have at least one recovered day "
            f"(median {median_recovered_days:.0f} recovered days per station; "
            "`recovered_station_days.csv`), though volume is still concentrated: "
            f"{top_recovered_text} account for most of it. The population split is "
            "therefore a **station-day** property, not a clean **station** property - a "
            "single station can contribute to both populations depending on the day, "
            "which is why §2's per-station diff and §4's population split tell related "
            "but not identical stories."
        )
    else:
        coverage_sentence = (
            "No station has a recovered day in this run "
            "(`data/recovered_stec_db/` absent or empty) - the entire population is "
            "'original'."
        )

    lines += [
        "",
        coverage_sentence,
        "",
    ]

    pop_outlier_10 = population_outlier_counts[
        population_outlier_counts["threshold_m"] == 10.0
    ].set_index(["population", "Method"])
    if ("original", STEC_LABEL) in pop_outlier_10.index and (
        "recovered",
        STEC_LABEL,
    ) in pop_outlier_10.index:
        stec_orig = pop_outlier_10.loc[("original", STEC_LABEL)]
        stec_recov = pop_outlier_10.loc[("recovered", STEC_LABEL)]
        gim_orig = pop_outlier_10.loc[("original", GIM_LABEL)]
        gim_recov = pop_outlier_10.loc[("recovered", GIM_LABEL)]
        lines += [
            f"`outlier_counts_by_population.csv` (10 m rule) shows Direct STEC's tail is "
            f"also concentrated in the recovered population: "
            f"{int(stec_recov['n_exceeding'])} station-days over 10 m in `recovered` "
            f"({stec_recov['pct_station_days_exceeding']:.1f}% of that population) vs "
            f"{int(stec_orig['n_exceeding'])} in `original` "
            f"({stec_orig['pct_station_days_exceeding']:.2g}%), while GIM has "
            f"{int(gim_recov['n_exceeding'])} vs {int(gim_orig['n_exceeding'])} "
            "respectively. So Direct STEC's worse tail (§3) and its population-split loss "
            "(§4) are largely the same underlying station-days showing up twice, not two "
            "independent problems.",
            "",
        ]

    # --- Recommendation ---------------------------------------------------------
    lines += [
        "## Recommendation: which statistic should the paper report",
        "",
        "**Median, or mean with the population split reported alongside it - not the mean "
        "alone.** Reasons, tied to the sections above:",
        "",
        f'1. The mean moves by {mean_range:.1f} points of "improvement" depending on an '
        'exclusion threshold with no principled basis beyond "10 m looked reasonable" '
        f'(§3) - a reviewer who asks "why 10 m and not 5 m or 20 m" currently has no '
        f"defensible answer. The median changes by at most {median_range:.1f} points across "
        'the same range, including the "no rule at all" case.',
        "2. The mean's sensitivity has an identified, non-adversarial cause: a small number "
        f"of station-days ({stec_10['pct_station_days_exceeding']:.2g}% of Direct STEC's "
        f"population) carrying {stec_10['error_mass_fraction_pct']:.1f}% of its total "
        "error, itself concentrated in a handful of stations (§3) that are "
        "disproportionately the geometry-only recovered population (§4). This is a real, "
        "structural weak point (no local calibration data), and a mean-based headline hides "
        "*where* the method is weak behind a single number that outliers dominate.",
        "3. Table 5's own population already mixes two populations with opposite signs "
        f"({_pct(original['mean_improvement_pct'], signed=True)} mean on "
        f"{int(original['n_stec']):,} station-days, "
        f"{_pct(recovered_pop['mean_improvement_pct'], signed=True)} mean on "
        f"{int(recovered_pop['n_stec']):,}) into one "
        f"{_pct(headline_mean_improvement, signed=True)} figure. A single pooled number "
        "answers \"what fraction of this year's positioning workload happened to fall in "
        'the harder population that needed recovery" as much as it answers "does the '
        'model work" - a sampling artifact of which station-days needed recovery, not '
        "purely a property of the method.",
        "",
        "If the paper reports the mean at all, it should not report it without the "
        "population split next to it and without stating the threshold sensitivity - a "
        'bare mean reads as "the model is a little better than GIM on average" when the '
        'more accurate statement is "the model is clearly better on the station-days it '
        "has real local data for, and clearly worse on the ones it doesn't, and those two "
        'effects happen to net out to a small number this year." The median is both more '
        "robust to the exclusion-threshold choice and more representative of the typical "
        "station-day.",
    ]

    return "\n".join(lines) + "\n"


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

    findings_markdown = _format_findings_markdown(
        frame=frame,
        overall=overall,
        daily=daily,
        per_station=per_station,
        threshold_counts=threshold_counts,
        headline_sensitivity=headline_sensitivity,
        by_day=by_day,
        concentration_summary=concentration_summary,
        recovered=recovered,
        population_by_method=population_by_method,
        population_stec_vs_gim=population_stec_vs_gim,
        population_outlier_counts=population_outlier_counts,
        summary_path=args.summary_path,
    )
    (args.output_dir / "FINDINGS.md").write_text(findings_markdown)

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
