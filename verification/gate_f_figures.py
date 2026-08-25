"""Gate F(figures): does each figure plot the values its declared source actually holds?

The audit that produced `docs/revision/independent_audit.md` recorded a gap (finding F3):
every figure generator in `stec/viz/` has been ported and runs, but nothing checks that any
of them plots the *right* numbers. A generator can read the wrong column, join on the wrong
key, apply a stale filter, or silently drop rows - and still produce a plausible-looking PNG.
This gate closes that gap for a defensible subset of figures, the same way
`gate_f_analysis_equivalence.py` closes it for the analyses one layer upstream.

**What this gate compares, and why not pixels.** Every figure's `_save()` helper
(`stec/viz/manuscript_figures.py`, `stec/viz/revision_figures.py`,
`stec/viz/diagnostic_figures.py`) writes the plotted data as `<name>.csv` alongside the PNG -
"the CSV holds what the figure actually draws... so the number a reader checks is the number
they see" (`revision_figures._save`'s own docstring). This gate reads that CSV and compares it
against an **independent recomputation of the same quantity from the upstream artifact** -
plain pandas/numpy, never by calling the `stec.viz` function under test, which would be
self-comparison and prove nothing. A pixel diff was considered and rejected for three
reasons, each sufficient on its own: rendered PNGs differ by matplotlib version, backend and
installed fonts, none of which bears on correctness; the port deliberately changed some
styling (seaborn's `colorblind` palette -> `APPROACH_COLORS`, documented in
`manuscript_figures.py`'s own docstring), so a pixel diff would fail on an intentional
cosmetic change exactly as loudly as on a real defect; and the manuscript's embedded PNGs are
Aug-18 artifacts with no recorded provenance, so there is no trustworthy pixel ground truth to
diff against even in principle.

**What a MATCH here does and does not prove**, restated for this gate specifically because
`docs/ARCHITECTURE.md` section 5 makes the same point about Gate F generally: a MATCH proves
the figure plots what its declared upstream source contains. It does not prove the upstream
source is scientifically right - `daily_metrics`, `positioning_coverage`, `stratified_
comparison` and `pretrained_test_diagnostics` have their own correctness questions, and this
gate is downstream of all of them. It also does not check styling, layout, colour choices, or
that the *right* figure was drawn for the *right* table - only that the numbers on the axes
are the numbers the source data says they should be.

**Scope.** Covering all ~40 figure kinds across three modules was explicitly out of scope for
this pass; the declared subset below prioritises figures that back a specific paper number
(Table 3/4 daily improvement, Table 5's positioning figures) and the one revision family
(`stratified_comparison`) that was silently broken until the day this gate was written - the
exact situation a check like this exists to catch before it reaches a table. Figure 11 is
declared with `skip=` rather than omitted, matching how the sibling gate handles its two
structural skips: an analysis reader should see that this figure was considered and found
not-yet-checkable, not conclude nobody thought about it.

**Design difference from the sibling gate.** `gate_f_analysis_equivalence.compare_frames`
compares two frames column-by-column *by position*, which is valid there because both sides
are two implementations of the same script producing rows in the same order. Here the two
sides are unrelated code paths - a `stec.viz` figure builder and this gate's own
recomputation - with no guarantee of matching row order or even row count, so `compare_on_keys`
below merges on each check's declared `join_keys` before comparing, and a merge that drops
rows (either side has a row the other does not) is folded into the difference map as its own
`_row_count` entry rather than silently comparing a smaller, matched-only slice. Tolerance
otherwise follows the sibling exactly: relative difference against a floor of 1.0 (`np.
maximum(abs(expected), 1.0)`), so a quantity near zero does not fail on floating-point noise.

    python verification/gate_f_figures.py
    python verification/gate_f_figures.py --only fig10_improvement_rmse fig12_positioning_trend
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.config import paths  # noqa: E402

RELATIVE_TOLERANCE = 1e-6

MANUSCRIPT_PLOTS = paths.REPO_ROOT / "plots" / "manuscript"
REVISION_PLOTS = paths.REPO_ROOT / "plots" / "revision"

DAILY_METRICS_DIR = paths.analysis_result_dir("daily_metrics", rebuilt=True)
if not (DAILY_METRICS_DIR / "per_day.csv").exists():
    DAILY_METRICS_DIR = paths.analysis_result_dir("daily_metrics", rebuilt=False)

POSITIONING_COVERAGE_DIR = paths.analysis_result_dir(
    "positioning_coverage", rebuilt=True
)
if not (POSITIONING_COVERAGE_DIR / "multiday_summary.csv").exists():
    POSITIONING_COVERAGE_DIR = paths.analysis_result_dir(
        "positioning_coverage", rebuilt=False
    )

STRATIFIED_COMPARISON_DIR = paths.analysis_result_dir(
    "stratified_comparison", rebuilt=True
)
if not (STRATIFIED_COMPARISON_DIR / "by_elevation.csv").exists():
    STRATIFIED_COMPARISON_DIR = paths.analysis_result_dir(
        "stratified_comparison", rebuilt=False
    )

PRETRAINED_DIAGNOSTICS_CACHE = (
    paths.analysis_result_dir("pretrained_test_diagnostics", rebuilt=True)
    / "observations.parquet"
)

# The station-day outlier rule Table 5 and Figures 12-15 all share
# (`stec.positioning.metrics.OUTLIER_3D_RMS_M`). Recomputation applies the same threshold
# directly (`<=` keeps) rather than importing `exclude_outlier_station_days` - the constant
# is a declared modelling choice, not the logic under test, but the filter itself is
# reimplemented here so this check does not share code with the figure it is checking.
POSITIONING_OUTLIER_THRESHOLD_M = 10.0

POSITIONING_METHOD_LABELS = {
    "STEC_iono": "Direct STEC",
    "Pretrained_STEC_iono": "Pretrained Direct STEC",
    "VTEC_iono": "VTEC + Mapping",
    "gim_iono": "IGS GIM + Mapping",
}

DAILY_METRICS_MODEL_LABELS = {
    "Direct STEC Model": "Direct STEC",
    "Pretrained STEC": "Pretrained Direct STEC",
    "VTEC + Mapping": "VTEC + Mapping",
    "IGS GIM": "IGS GIM + Mapping",
}


@dataclass(frozen=True)
class FigureCheck:
    """One figure, the CSV it wrote, and how to independently recompute what it should say."""

    name: str
    figure: str
    plotted_csv: Path
    upstream: tuple[Path, ...]
    join_keys: tuple[str, ...]
    value_columns: tuple[str, ...]
    # Returns the independently recomputed frame - `join_keys + value_columns`, at minimum.
    recompute: Callable[[], pd.DataFrame]
    # Overridable because a couple of checks compare an aggregate of the plotted CSV (e.g.
    # per-method mean/count) rather than the raw rows it holds - see fig13/fig15 below.
    load_plotted: Callable[[Path], pd.DataFrame] = pd.read_csv
    # Same semantics as gate_f_analysis_equivalence.Comparison.expected_divergence: a column
    # named here is allowed to exceed tolerance, and the value is why. Empty for every check
    # declared below - no known, intended divergence exists yet - kept so a future check that
    # does have one does not need a different mechanism.
    expected_divergence: dict[str, str] = field(default_factory=dict)
    skip: str | None = None


def _relative_difference(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(expected), 1.0)
    return np.abs(actual - expected) / scale


def compare_on_keys(
    plotted: pd.DataFrame,
    recomputed: pd.DataFrame,
    join_keys: tuple[str, ...],
    value_columns: tuple[str, ...],
) -> tuple[dict[str, float], tuple[int, int, int]]:
    """Merge on `join_keys` and return the max relative difference per value column.

    Unlike `gate_f_analysis_equivalence.compare_frames`, this does not assume the two sides
    share row order or row count - see the module docstring's "design difference" section.
    A merge that drops rows on either side is recorded as its own `_row_count` entry (set to
    infinity, the same sentinel `compare_frames` uses for a length mismatch) so it flows
    through `verdict_for` exactly like any other unexplained difference, rather than being
    silently absorbed into a smaller matched-only comparison.
    """
    merged = plotted.merge(
        recomputed,
        on=list(join_keys),
        how="inner",
        suffixes=("_plotted", "_recomputed"),
    )
    differences: dict[str, float] = {}
    if len(merged) != len(plotted) or len(merged) != len(recomputed):
        differences["_row_count"] = float("inf")
    for column in value_columns:
        left = merged[f"{column}_plotted"].to_numpy(dtype=float)
        right = merged[f"{column}_recomputed"].to_numpy(dtype=float)
        if len(merged) == 0:
            # np.nanmax of an empty array never runs; falling through to a bare 0.0 here is
            # exactly the vacuous-MATCH trap gate_f_analysis_equivalence's own history warns
            # about (see its module docstring, "0-row guard"). Infinity forces this into
            # FAIL/DIVERGED rather than letting an empty merge read as agreement.
            differences[column] = float("inf")
            continue
        differences[column] = float(np.nanmax(_relative_difference(left, right)))
    return differences, (len(plotted), len(recomputed), len(merged))


def verdict_for(
    check: FigureCheck, differences: dict[str, float], row_counts: tuple[int, int, int]
) -> tuple[str, list[str]]:
    """MATCH / DIVERGED / FAIL, plus the evidence - mirrors `gate_f_analysis_equivalence.
    verdict_for`'s three-way split and its 0-row guard, adapted for a merge-based comparison.
    """
    n_plotted, n_recomputed, _n_matched = row_counts
    if n_plotted == 0 or n_recomputed == 0:
        return "FAIL", [
            f"{n_plotted} plotted rows vs {n_recomputed} recomputed rows - "
            "nothing was actually compared"
        ]
    if not differences:
        return "FAIL", [
            "no value column produced a difference - nothing was actually compared"
        ]

    unexplained = [
        column
        for column, delta in differences.items()
        if delta > RELATIVE_TOLERANCE and column not in check.expected_divergence
    ]
    explained = [
        column
        for column, delta in differences.items()
        if delta > RELATIVE_TOLERANCE and column in check.expected_divergence
    ]
    if unexplained:
        return "FAIL", [f"{c}={differences[c]:.3g}" for c in unexplained]
    if explained:
        return "DIVERGED", [check.expected_divergence[c] for c in explained]
    return "MATCH", []


def run_check(check: FigureCheck) -> str:
    if check.skip:
        print(f"  {check.name:<32} SKIPPED  {check.skip}")
        return "SKIPPED"

    missing = [p for p in (check.plotted_csv, *check.upstream) if not p.exists()]
    if missing:
        print(
            f"  {check.name:<32} SKIPPED  missing on disk: "
            + ", ".join(str(p) for p in missing)
        )
        return "SKIPPED"

    plotted = check.load_plotted(check.plotted_csv)
    recomputed = check.recompute()
    differences, row_counts = compare_on_keys(
        plotted, recomputed, check.join_keys, check.value_columns
    )
    verdict, notes = verdict_for(check, differences, row_counts)

    n_plotted, n_recomputed, n_matched = row_counts
    counts = f"rows plotted={n_plotted} recomputed={n_recomputed} matched={n_matched}"
    detail = f" - {notes[0]}" if notes else ""
    print(f"  {check.name:<32} {verdict:<8} {counts}{detail}")
    return verdict


# --------------------------------------------------------------------------
# Figure 10 - daily RMSE/MAE improvement, own dataset
# (`stec.viz.manuscript_figures.fig_improvement_by_date`, from `daily_metrics.per_day.csv`)
# --------------------------------------------------------------------------


def _recompute_improvement_by_date(metric: str) -> pd.DataFrame:
    """`fig_improvement_by_date`'s formula, reimplemented from the per_day.csv it reads.

    `_build_improvement_by_date_figures` loops `table.groupby("dataset")` and writes every
    dataset's figure to the same filename (`improvements_{metric}.csv` under
    `stec_finetuned_2024/`, keyed only by `source="finetuned"` in `SOURCE_DIRS`) - the last
    group alphabetically, `own_vtec_gim`, is therefore what survives on disk (`madrigal_vtec_
    gim` sorts first and gets overwritten). This recomputation targets `own_vtec_gim`
    specifically to match what is actually there; see this module's own report for why that
    filename collision is itself worth flagging.
    """
    table = pd.read_csv(DAILY_METRICS_DIR / "per_day.csv")
    table = table[table["dataset"] == "own_vtec_gim"].copy()
    table["Model"] = table["Model"].map(DAILY_METRICS_MODEL_LABELS)
    table["date"] = pd.to_datetime(table["year"], format="%Y") + pd.to_timedelta(
        table["doy"] - 1, unit="D"
    )
    pivot = table.pivot(index="date", columns="Model", values=metric).sort_index()
    stec = pivot["Direct STEC"]

    rows = []
    for baseline in ("VTEC + Mapping", "IGS GIM + Mapping"):
        baseline_values = pivot[baseline].replace(0, np.nan)
        improvement = (1 - stec / baseline_values) * 100
        rows.append(
            pd.DataFrame(
                {
                    "date": improvement.index,
                    "baseline": baseline,
                    "improvement_pct": improvement.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _load_improvement_plotted(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _figure10_check(metric: str) -> FigureCheck:
    return FigureCheck(
        name=f"fig10_improvement_{metric.lower()}",
        figure=f"Figure 10 ({metric}, own dataset)",
        plotted_csv=MANUSCRIPT_PLOTS
        / "stec_finetuned_2024"
        / f"improvements_{metric.lower()}.csv",
        upstream=(DAILY_METRICS_DIR / "per_day.csv",),
        join_keys=("date", "baseline"),
        value_columns=("improvement_pct",),
        load_plotted=_load_improvement_plotted,
        recompute=lambda metric=metric: _recompute_improvement_by_date(metric),
    )


# --------------------------------------------------------------------------
# Figures 12-15 - positioning, from positioning_coverage's multiday_summary.csv
# (`stec.viz.manuscript_figures._build_positioning_figures` and its four `fig_*` callees)
# --------------------------------------------------------------------------


def _load_filtered_positioning() -> pd.DataFrame:
    frame = pd.read_csv(
        POSITIONING_COVERAGE_DIR / "multiday_summary.csv",
        usecols=["date", "method", "error_3d_rms"],
    )
    frame["method"] = frame["method"].map(POSITIONING_METHOD_LABELS)
    frame = frame.dropna(subset=["method"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[frame["error_3d_rms"] <= POSITIONING_OUTLIER_THRESHOLD_M].copy()


def _recompute_positioning_trend() -> pd.DataFrame:
    frame = _load_filtered_positioning()
    daily = (
        frame.groupby(["date", "method"])["error_3d_rms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    daily["sem"] = daily["std"] / np.sqrt(daily["count"])
    return daily


def _recompute_positioning_distribution_stats() -> pd.DataFrame:
    frame = _load_filtered_positioning()
    return (
        frame.groupby("method")["error_3d_rms"]
        .agg(mean="mean", count="count", median="median")
        .reset_index()
    )


def _reduce_plotted_distribution_stats(path: Path) -> pd.DataFrame:
    """fig13's plotted CSV is the raw filtered rows; reduce to the same per-method
    aggregate `_recompute_positioning_distribution_stats` produces, so the comparison is
    "does the population behind the boxplot have the right mean/count/median", not a
    row-for-row identity that would just be re-deriving the same filter twice."""
    raw = pd.read_csv(path)
    return (
        raw.groupby("method")["error_3d_rms"]
        .agg(mean="mean", count="count", median="median")
        .reset_index()
    )


def _recompute_positioning_improvement_timeseries() -> pd.DataFrame:
    frame = _load_filtered_positioning()
    daily_mean = frame.groupby(["date", "method"])["error_3d_rms"].mean()
    pivot = daily_mean.unstack("method").sort_index()
    gim = pivot["IGS GIM + Mapping"]
    rows = []
    for method in [c for c in pivot.columns if c != "IGS GIM + Mapping"]:
        improvement = (gim - pivot[method]) / gim * 100
        rows.append(
            pd.DataFrame(
                {
                    "date": improvement.index,
                    "method": method,
                    "improvement_pct": improvement.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


_CDF_QUANTILE_LEVELS = (0.50, 0.95, 0.99)


def _empirical_quantile(values: np.ndarray, level: float) -> float:
    """The nearest-rank definition `fig_positioning_cdf_3d_rms` itself uses for its ECDF
    (`percentile = np.arange(1, n + 1) / n * 100`, one step per sorted value), picking
    whichever rank's own cumulative percentage is closest to `level * 100` - the same rule
    `_reduce_plotted_cdf_quantiles` applies when reading a level off the plotted CSV. Using
    `np.quantile`'s interpolation here instead was tried first and produced small
    (~0.1-0.2%) but real disagreements at the 95th/99th percentile purely from the two
    sides picking adjacent ranks under two different definitions of "quantile" - not a
    figure defect, an artefact of comparing under mismatched statistics. Unifying on one
    rank-selection rule, applied independently to two different data reads, is what makes
    this a fair comparison.
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative_pct = np.arange(1, n + 1) / n * 100
    index = int(np.argmin(np.abs(cumulative_pct - level * 100)))
    return float(sorted_values[index])


def _recompute_positioning_cdf_quantiles() -> pd.DataFrame:
    frame = _load_filtered_positioning()
    rows = []
    for method, group in frame.groupby("method"):
        values = group["error_3d_rms"].to_numpy(dtype=float)
        for level in _CDF_QUANTILE_LEVELS:
            rows.append(
                {
                    "method": method,
                    "percentile": round(level * 100),
                    "error_3d_rms": _empirical_quantile(values, level),
                }
            )
    return pd.DataFrame(rows)


def _reduce_plotted_cdf_quantiles(path: Path) -> pd.DataFrame:
    """fig15's plotted CSV is (method, error_3d_rms, cumulative_pct) - the full sorted
    per-station-day population and its empirical CDF value, one row per station-day. The
    quantile at a fixed percentile is read off that same population directly (nearest
    cumulative_pct per method) rather than by re-deriving the CDF, giving an independent
    cross-check of the values embedded in the plotted curve without needing the two sides
    to agree on exact `cumulative_pct` floats to join on."""
    raw = pd.read_csv(path)
    rows = []
    for method, group in raw.groupby("method"):
        group = group.sort_values("cumulative_pct")
        for level in _CDF_QUANTILE_LEVELS:
            target_pct = level * 100
            idx = (group["cumulative_pct"] - target_pct).abs().idxmin()
            rows.append(
                {
                    "method": method,
                    "percentile": round(level * 100),
                    "error_3d_rms": float(group.loc[idx, "error_3d_rms"]),
                }
            )
    return pd.DataFrame(rows)


_POSITIONING_UPSTREAM = (POSITIONING_COVERAGE_DIR / "multiday_summary.csv",)

FIGURE12_CHECK = FigureCheck(
    name="fig12_positioning_trend",
    figure="Figure 12 (daily 3D RMS, mean +/- SEM)",
    plotted_csv=MANUSCRIPT_PLOTS / "positioning_2024" / "pos_trend.csv",
    upstream=_POSITIONING_UPSTREAM,
    join_keys=("date", "method"),
    value_columns=("mean", "std", "count", "sem"),
    load_plotted=lambda p: pd.read_csv(p, parse_dates=["date"]),
    recompute=_recompute_positioning_trend,
)

FIGURE13_CHECK = FigureCheck(
    name="fig13_positioning_distribution",
    figure="Figure 13 (overall 3D RMS distribution, per-method aggregates)",
    plotted_csv=MANUSCRIPT_PLOTS / "positioning_2024" / "pos_distribution_boxplot.csv",
    upstream=_POSITIONING_UPSTREAM,
    join_keys=("method",),
    value_columns=("mean", "count", "median"),
    load_plotted=_reduce_plotted_distribution_stats,
    recompute=_recompute_positioning_distribution_stats,
)

FIGURE14_CHECK = FigureCheck(
    name="fig14_positioning_improvement_timeseries",
    figure="Figure 14 (daily % improvement over IGS GIM + Mapping)",
    plotted_csv=MANUSCRIPT_PLOTS
    / "positioning_2024"
    / "pos_improvement_timeseries.csv",
    upstream=_POSITIONING_UPSTREAM,
    join_keys=("date", "method"),
    value_columns=("improvement_pct",),
    load_plotted=lambda p: pd.read_csv(p, parse_dates=["date"]),
    recompute=_recompute_positioning_improvement_timeseries,
)

FIGURE15_CHECK = FigureCheck(
    name="fig15_positioning_cdf",
    figure="Figure 15 (3D RMS CDF, median/p95/p99 per method)",
    plotted_csv=MANUSCRIPT_PLOTS / "positioning_2024" / "pos_cdf_3d_rms.csv",
    upstream=_POSITIONING_UPSTREAM,
    join_keys=("method", "percentile"),
    value_columns=("error_3d_rms",),
    load_plotted=_reduce_plotted_cdf_quantiles,
    recompute=_recompute_positioning_cdf_quantiles,
)


# --------------------------------------------------------------------------
# Two of Figures 4-9 - single-model residual/uncertainty diagnostics, from the
# pretrained_test_diagnostics cache (`stec.viz.manuscript_figures.fig_residuals_elev`,
# `fig_uncertainty`)
# --------------------------------------------------------------------------

_ELEVATION_BIN_EDGES = np.linspace(5.0, 90.0, 18)  # 17 bins, 5-degree width


def _recompute_residuals_elev() -> pd.DataFrame:
    observations = pd.read_parquet(
        PRETRAINED_DIAGNOSTICS_CACHE, columns=["true_stec", "stec_pred", "satele"]
    )
    residual = observations["true_stec"] - observations["stec_pred"]
    elevation_bin = pd.cut(
        observations["satele"], bins=_ELEVATION_BIN_EDGES, include_lowest=True
    )
    grouped = pd.DataFrame(
        {"elevation_bin": elevation_bin, "residual": residual}
    ).groupby("elevation_bin", observed=True)
    rows = []
    for interval, group in grouped:
        values = group["residual"].to_numpy(dtype=float)
        rows.append(
            {
                "bin_left": round(float(interval.left)),
                "bin_right": round(float(interval.right)),
                "mae": float(np.abs(values).mean()),
                "rmse": float(np.sqrt(np.mean(values**2))),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def _reduce_plotted_residuals_elev(path: Path) -> pd.DataFrame:
    """Parses the plotted CSV's en-dash bin label ("5–10") into (bin_left, bin_right)
    independently of `fig_residuals_elev`'s own label-building code, so the join key does
    not depend on reproducing its exact string formatting."""
    raw = pd.read_csv(path)
    left, right = zip(
        *(tuple(float(x) for x in label.split("–")) for label in raw["elevation_bin"])
    )
    return raw.assign(
        bin_left=[round(v) for v in left], bin_right=[round(v) for v in right]
    )


FIGURE5_CHECK = FigureCheck(
    name="fig5_residuals_elev",
    figure="Figure 5 (residual MAE/RMSE by elevation bin, pretrained test set)",
    plotted_csv=MANUSCRIPT_PLOTS / "stec_pretrained_testset" / "residuals_elev.csv",
    upstream=(PRETRAINED_DIAGNOSTICS_CACHE,),
    join_keys=("bin_left", "bin_right"),
    value_columns=("mae", "rmse", "n"),
    load_plotted=_reduce_plotted_residuals_elev,
    recompute=_recompute_residuals_elev,
)


def _recompute_uncertainty() -> pd.DataFrame:
    observations = pd.read_parquet(
        PRETRAINED_DIAGNOSTICS_CACHE,
        columns=[
            "true_stec",
            "stec_pred",
            "pred_total_unc",
            "pred_epistemic_unc",
            "pred_aleatoric_unc",
        ],
    )
    abs_error = (observations["true_stec"] - observations["stec_pred"]).abs().to_numpy()
    total_unc = observations["pred_total_unc"].to_numpy(dtype=float)
    epistemic = observations["pred_epistemic_unc"].to_numpy(dtype=float)
    aleatoric = observations["pred_aleatoric_unc"].to_numpy(dtype=float)

    max_unc = float(np.quantile(total_unc, 0.95))
    bin_width = 1.0
    bin_edges = np.arange(0, max(np.ceil(max_unc), 1.0) + bin_width, bin_width)
    unc_bin = pd.cut(total_unc, bins=bin_edges, include_lowest=True, labels=False)

    rows = []
    for bin_index in range(len(bin_edges) - 1):
        in_bin = unc_bin == bin_index
        if in_bin.sum() < 5:
            continue
        rows.append(
            {
                "unc_bin_center": round(
                    (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2, 3
                ),
                "mean_abs_error": float(abs_error[in_bin].mean()),
                "mean_total_unc": float(total_unc[in_bin].mean()),
                "mean_epistemic_unc": float(epistemic[in_bin].mean()),
                "mean_aleatoric_unc": float(aleatoric[in_bin].mean()),
            }
        )
    return pd.DataFrame(rows)


def _load_uncertainty_plotted(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["unc_bin_center"] = frame["unc_bin_center"].round(3)
    return frame


FIGURE9_CHECK = FigureCheck(
    name="fig9_uncertainty",
    figure="Figure 9 (absolute error vs. predicted-sigma bin, 4 curves)",
    plotted_csv=MANUSCRIPT_PLOTS / "stec_pretrained_testset" / "uncertainty.csv",
    upstream=(PRETRAINED_DIAGNOSTICS_CACHE,),
    join_keys=("unc_bin_center",),
    value_columns=(
        "mean_abs_error",
        "mean_total_unc",
        "mean_epistemic_unc",
        "mean_aleatoric_unc",
    ),
    load_plotted=_load_uncertainty_plotted,
    recompute=_recompute_uncertainty,
)


# --------------------------------------------------------------------------
# Stratified revision family - the one that was silently broken until the
# `positioning_coverage` fix landed (see CLAUDE.md's canonical-results table). Elevation
# axis only, both to keep this gate's scope bounded and because it is the same axis
# `fig5_residuals_elev` above checks for the pretrained model, so a reader can compare the
# two model families on the same bins.
# --------------------------------------------------------------------------


def _interval_label(text: str) -> str:
    """Reimplements `revision_figures._interval_label`'s label independently: parses a
    pandas interval string, rounds an edge within 0.01 of an integer (the `include_lowest`
    padding, e.g. "4.999" for a true 5), and joins with an en dash. Not imported from
    `revision_figures` on purpose - see the module docstring on why the recompute side must
    not call into the code under test.
    """
    left, right = text.strip("()[]").split(", ")
    lo, hi = float(left), float(right)
    lo = round(lo) if abs(lo - round(lo)) < 0.01 else lo
    hi = round(hi) if abs(hi - round(hi)) < 0.01 else hi

    def fmt(value: float) -> str:
        return f"{value:g}"

    return f"{fmt(lo)}–{fmt(hi)}"


def _recompute_stratified_elevation_absolute() -> pd.DataFrame:
    table = pd.read_csv(STRATIFIED_COMPARISON_DIR / "by_elevation.csv")
    return pd.DataFrame(
        {
            "group": table["bin"].map(_interval_label),
            "series": table["Method"],
            "value": table["RMSE"],
        }
    )


FIGURE_STRATIFIED_ELEVATION_CHECK = FigureCheck(
    name="fig_stratified_elevation_absolute",
    figure="Stratified-by-elevation STEC RMSE (R1.4 revision figure family)",
    plotted_csv=REVISION_PLOTS
    / "stec_finetuned_2024"
    / "stratified_elevation_absolute.csv",
    upstream=(STRATIFIED_COMPARISON_DIR / "by_elevation.csv",),
    join_keys=("group", "series"),
    value_columns=("value",),
    recompute=_recompute_stratified_elevation_absolute,
)


# --------------------------------------------------------------------------
# Figure 11 - declared skip, not omitted. Its input analysis
# (`stec.analysis.elevation_metrics_finetuned`) has never been run at full coverage;
# CLAUDE.md's own "src/'s status" section says the Madrigal re-inference this figure
# depends on indirectly is queued, not run. Named here so the gate's summary reports it as
# considered-and-blocked, matching how `gate_f_analysis_equivalence` handles
# `repair_gim_baseline`/`positioning_coverage`.
# --------------------------------------------------------------------------

FIGURE11_CHECK = FigureCheck(
    name="fig11_mae_rmse_finetuned",
    figure="Figure 11 (RMSE/MAE vs. elevation, mean +/- across-day std)",
    plotted_csv=Path(),
    upstream=(),
    join_keys=(),
    value_columns=(),
    recompute=lambda: pd.DataFrame(),
    skip=(
        "its input, stec.analysis.elevation_metrics_finetuned, has never been run at full "
        "coverage - blocked behind another session's Madrigal re-inference (see CLAUDE.md, "
        "'src/'s status')"
    ),
)


CHECKS: tuple[FigureCheck, ...] = (
    _figure10_check("RMSE"),
    _figure10_check("MAE"),
    FIGURE11_CHECK,
    FIGURE12_CHECK,
    FIGURE13_CHECK,
    FIGURE14_CHECK,
    FIGURE15_CHECK,
    FIGURE5_CHECK,
    FIGURE9_CHECK,
    FIGURE_STRATIFIED_ELEVATION_CHECK,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="check names")
    args = parser.parse_args()

    selected = CHECKS
    if args.only:
        known = {c.name for c in CHECKS}
        unknown = set(args.only) - known
        if unknown:
            raise SystemExit(f"unknown check: {sorted(unknown)}")
        selected = tuple(c for c in CHECKS if c.name in set(args.only))

    print(f"checking {len(selected)} figure(s)\n")
    verdicts = [run_check(c) for c in selected]

    print()
    for label in ("MATCH", "DIVERGED", "SKIPPED", "FAIL"):
        count = verdicts.count(label)
        if count:
            print(f"  {label}: {count}")
    if "FAIL" in verdicts:
        print("\n  FAIL  at least one figure disagrees with its declared source")
        return 1

    compared = [v for v in verdicts if v in ("MATCH", "DIVERGED")]
    if not compared:
        print("\n  INCONCLUSIVE  nothing was actually compared")
        return 2
    print(
        f"\n  PASS  {len(compared)} figure(s) compared, every value matched its source"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
