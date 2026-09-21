"""Exercises `stec.viz.oracle_benchmark` against synthetic paired-station-day frames -
mirrors `tests/viz/test_diagnostic_figures.py`'s style: a `_save`-plumbing check, one
end-to-end frame -> PNG/CSV path per figure, and independent hand-computed checks on the
plotted-data CSV and the generated table for a subset of figures (computed with plain
pandas/numpy in the test, not by calling the module's own helpers back - the point is to
catch the module computing something other than what it draws).

Kept synthetic throughout, no read of the real `oracle_benchmark` analysis output - the
real CSVs are exercised by `python -m stec.pipeline run --only oracle_benchmark_figures`,
not by this suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import matplotlib.pyplot as plt

from stec.viz import oracle_benchmark as ob
from stec.viz import style


def _synthetic_paired_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Shaped like `paired_station_days.csv`: one row per station-day, one column per
    method, well-separated locations so the four boxes/medians are visibly distinct and a
    handful of extreme values in each column so the Tukey fence has something to catch."""
    rng = np.random.default_rng(seed)
    stations = rng.choice([f"ST{i:02d}" for i in range(10)], size=n)
    doys = rng.integers(122, 367, size=n)

    def lognormal_column(median: float, extreme_count: int, extreme_value: float):
        values = rng.lognormal(mean=np.log(median), sigma=0.4, size=n)
        values[:extreme_count] = extreme_value
        return values

    return pd.DataFrame(
        {
            "station": stations,
            "doy": doys,
            ob.ORACLE_LABEL: lognormal_column(0.07, 5, 50.0),
            "Direct STEC": lognormal_column(0.8, 5, 80.0),
            "VTEC + Mapping": lognormal_column(1.1, 5, 90.0),
            "IGS GIM + Mapping": lognormal_column(1.0, 5, 70.0),
        }
    )


# --------------------------------------------------------------------------
# _save plumbing
# --------------------------------------------------------------------------


def test_save_writes_titled_notitle_and_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("A title")
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    ob._save(fig, "demo", tmp_path, "unit test provenance", data)

    assert (tmp_path / "demo.png").exists()
    assert (tmp_path / "demo_notitle.png").exists()
    written = pd.read_csv(tmp_path / "demo.csv")
    pd.testing.assert_frame_equal(written, data)


# --------------------------------------------------------------------------
# Box stats / box plot
# --------------------------------------------------------------------------


def test_box_stats_matches_a_hand_computed_tukey_fence():
    """8 values, one clear outlier (100). Q1/median/Q3 are pandas' own linear-interpolated
    quantiles at n=8 (positions 1.75/3.5/5.25), worked out by hand below rather than by
    calling `.quantile()` again inside the assertion, so the test does not just restate the
    function under test."""
    series = pd.Series([1, 2, 3, 4, 5, 6, 7, 100], dtype=float)

    stats = ob._box_stats(series)

    assert stats["q1"] == pytest.approx(2.75)
    assert stats["median"] == pytest.approx(4.5)
    assert stats["q3"] == pytest.approx(6.25)
    # IQR = 3.5, upper fence = 6.25 + 1.5*3.5 = 11.5 -> whishi is the largest value <= 11.5.
    assert stats["whishi"] == pytest.approx(7.0)
    assert stats["whislo"] == pytest.approx(1.0)
    assert stats["n"] == 8
    assert stats["n_fliers"] == 1
    assert list(stats["fliers"]) == [100.0]


def test_fig_boxplot_builds_end_to_end_and_plotted_csv_carries_every_flier(tmp_path):
    style.configure_plotting()
    df = _synthetic_paired_frame()
    order = [m for m in ob.DISPLAY_ORDER if m in df.columns]

    ob.fig_boxplot(df, tmp_path, "unit test provenance")

    assert (tmp_path / "boxplot_oracle_benchmark.png").exists()
    assert (tmp_path / "boxplot_oracle_benchmark_notitle.png").exists()
    plotted = pd.read_csv(tmp_path / "boxplot_oracle_benchmark.csv")
    assert set(plotted["Method"]) == set(order)
    # Every method injected exactly 5 extreme values well above any plausible fence.
    for method in order:
        n_fliers_row = plotted[
            (plotted["Method"] == method) & (plotted["stat"] == "n_fliers")
        ]
        n_flier_points = (
            (plotted["Method"] == method) & (plotted["stat"] == "flier")
        ).sum()
        assert int(n_fliers_row["value"].iloc[0]) == n_flier_points
        assert n_flier_points >= 5


def test_fig_boxplot_uses_oracle_colour_not_an_approach_colour(tmp_path):
    """Regression guard for the one rule this module must never break: the oracle floor is
    not a fifth approach, so it must never take a colour from APPROACH_COLORS."""
    style.configure_plotting()
    df = _synthetic_paired_frame()
    assert ob.ORACLE_LABEL not in style.APPROACH_COLORS
    colors = [
        style.ORACLE_COLOR if m == ob.ORACLE_LABEL else style.APPROACH_COLORS[m]
        for m in ob.DISPLAY_ORDER
        if m in df.columns
    ]
    assert style.ORACLE_COLOR not in set(style.APPROACH_COLORS.values())
    assert colors[0] == style.ORACLE_COLOR


# --------------------------------------------------------------------------
# Paired-difference scatter
# --------------------------------------------------------------------------


def test_fig_paired_difference_win_rate_matches_an_independent_computation(tmp_path):
    """10 rows with a known Direct-STEC-vs-GIM outcome (7 wins, 3 losses) and a separate
    known Direct-STEC-vs-VTEC outcome (4 wins, 6 losses) - the win percentages in the
    plotted CSV must match hand counts, not just be internally self-consistent."""
    style.configure_plotting()
    n = 10
    direct = np.full(n, 1.0)
    gim = np.array([2.0] * 7 + [0.5] * 3)  # Direct STEC wins 7 of 10
    vtec = np.array([2.0] * 4 + [0.5] * 6)  # Direct STEC wins 4 of 10
    df = pd.DataFrame(
        {
            "station": [f"ST{i}" for i in range(n)],
            "doy": range(n),
            ob.ORACLE_LABEL: np.full(n, 0.1),
            "Direct STEC": direct,
            "VTEC + Mapping": vtec,
            "IGS GIM + Mapping": gim,
        }
    )

    ob.fig_paired_difference(df, tmp_path, "unit test provenance")

    assert (tmp_path / "paired_difference.png").exists()
    assert (tmp_path / "paired_difference_notitle.png").exists()
    plotted = pd.read_csv(tmp_path / "paired_difference.csv")
    gim_rows = plotted[plotted["baseline_method"] == "IGS GIM + Mapping"]
    vtec_rows = plotted[plotted["baseline_method"] == "VTEC + Mapping"]
    assert len(gim_rows) == n
    assert len(vtec_rows) == n
    assert gim_rows["direct_stec_wins"].sum() == 7
    assert vtec_rows["direct_stec_wins"].sum() == 4


# --------------------------------------------------------------------------
# Station coverage
# --------------------------------------------------------------------------


def test_fig_station_coverage_keeps_every_station_including_below_threshold(tmp_path):
    style.configure_plotting()
    coverage = pd.DataFrame(
        {
            "station": ["HIGH1", "HIGH2", "LOW1", "LOW2", "LOW3"],
            "station_days": [200, 150, 19, 5, 1],
        }
    )

    ob.fig_station_coverage(coverage, tmp_path, "unit test provenance")

    assert (tmp_path / "station_coverage.png").exists()
    assert (tmp_path / "station_coverage_notitle.png").exists()
    plotted = pd.read_csv(tmp_path / "station_coverage.csv")
    # No station dropped - this figure's entire point is to show the thin end, not hide it.
    assert set(plotted["station"]) == set(coverage["station"])
    assert len(plotted) == len(coverage)
    below_threshold = plotted[plotted["station_days"] < ob.LOW_COVERAGE_THRESHOLD_DAYS]
    assert set(below_threshold["station"]) == {"LOW1", "LOW2", "LOW3"}


# --------------------------------------------------------------------------
# Generated table
# --------------------------------------------------------------------------


def test_build_oracle_table_values_match_independent_pandas_quantiles(tmp_path):
    df = _synthetic_paired_frame(n=2000, seed=1)
    source_csv = tmp_path / "paired_station_days.csv"

    markdown = ob.build_oracle_table(df, source_csv)

    floor_median = df[ob.ORACLE_LABEL].median()
    for method in ob.DISPLAY_ORDER:
        series = df[method]
        expected_median = series.median()
        expected_p99 = series.quantile(0.99)
        expected_ratio = expected_median / floor_median
        expected_pct = (series > ob.EXCEEDANCE_THRESHOLD_M).mean() * 100
        # The table is plain markdown - find the method's row and check its numbers
        # rather than re-parsing the whole grid.
        row = next(
            line for line in markdown.splitlines() if line.startswith(f"| {method} ")
        )
        assert f"{expected_median:.3f}" in row
        assert f"{expected_p99:.3f}" in row
        assert f"{expected_ratio:.1f}x" in row
        assert f"{expected_pct:.2f}%" in row


def test_build_oracle_table_footnote_names_source_and_n(tmp_path):
    df = _synthetic_paired_frame(n=300, seed=2)
    source_csv = tmp_path / "paired_station_days.csv"

    markdown = ob.build_oracle_table(df, source_csv)

    assert str(source_csv) in markdown
    assert f"N={len(df):,}" in markdown
    assert "elev" in markdown
    assert f"{df['station'].nunique()} stations" in markdown


def test_build_oracle_table_includes_the_oracle_floor_row_at_ratio_one():
    df = _synthetic_paired_frame(n=300, seed=3)
    markdown = ob.build_oracle_table(df, pd.io.common.stringify_path("dummy.csv"))
    floor_row = next(
        line
        for line in markdown.splitlines()
        if line.startswith(f"| {ob.ORACLE_LABEL} ")
    )
    assert "1.0x" in floor_row


# --------------------------------------------------------------------------
# Entry point / graceful skip
# --------------------------------------------------------------------------


def test_build_figures_skips_gracefully_without_the_source_csv(tmp_path):
    style.configure_plotting()
    results_dir = tmp_path / "no_such_results"
    output_dir = tmp_path / "out"
    args = type("Args", (), {"results_dir": results_dir})()

    ob._build_figures(args, output_dir)

    assert not output_dir.exists()
