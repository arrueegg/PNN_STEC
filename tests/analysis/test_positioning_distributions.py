"""Tests for `stec.analysis.positioning_distributions`, with hand-computed expected
values.

Built for the owner's decision (2026-08-28) to replace Table 5's mean-with-10 m-exclusion
with unfiltered distributions and medians - see that module's docstring for the full
reasoning. Every aggregation function here is exercised against tiny, hand-computable
frames rather than the real 43,101-row summary, so the expected numbers are arithmetic,
not a second implementation. Plotting (`stec.viz.positioning_distributions`) is not
covered here - only the percentile/exceedance/box-statistic logic the figures read.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import positioning_distributions as pdist

STEC = "Direct STEC"
GIM = "IGS GIM + Mapping"


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (Method, error_3d_rms)."""
    return pd.DataFrame(rows, columns=["Method", "error_3d_rms"])


# ---------------------------------------------------------------------------
# percentile_summary
# ---------------------------------------------------------------------------


def test_percentile_summary_matches_pandas_linear_interpolation_by_hand():
    # 5 evenly spaced points make every default pandas quantile (linear
    # interpolation) land on a value computable by hand:
    #   position = q * (n - 1); q1 -> 0.25*4=1.0 -> 2; median -> 2.0 -> 3;
    #   q3 -> 3.0 -> 4; p95 -> 3.8 -> 4 + 0.8*(5-4) = 4.8; p99 -> 3.96 -> 4.96.
    df = frame([(STEC, v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]])
    result = pdist.percentile_summary(df, ["Method"]).set_index("Method").loc[STEC]

    assert result["n"] == 5
    assert result["mean_m"] == pytest.approx(3.0)
    assert result["median_m"] == pytest.approx(3.0)
    assert result["q1_m"] == pytest.approx(2.0)
    assert result["q3_m"] == pytest.approx(4.0)
    assert result["iqr_m"] == pytest.approx(2.0)
    assert result["p95_m"] == pytest.approx(4.8)
    assert result["p99_m"] == pytest.approx(4.96)


def test_percentile_summary_groups_independently_and_by_multiple_columns():
    df = pd.DataFrame(
        [
            (STEC, "quiet", 1.0),
            (STEC, "quiet", 3.0),
            (STEC, "storm", 10.0),
            (STEC, "storm", 30.0),
            (GIM, "quiet", 100.0),
        ],
        columns=["Method", "regime", "error_3d_rms"],
    )
    result = pdist.percentile_summary(df, ["Method", "regime"]).set_index(
        ["Method", "regime"]
    )

    assert result.loc[(STEC, "quiet"), "median_m"] == pytest.approx(2.0)  # mean(1, 3)
    assert result.loc[(STEC, "storm"), "median_m"] == pytest.approx(
        20.0
    )  # mean(10, 30)
    assert result.loc[(GIM, "quiet"), "n"] == 1
    assert result.loc[(GIM, "quiet"), "median_m"] == pytest.approx(100.0)
    # A single-point group's IQR/whisker spread all collapse to that one point.
    assert result.loc[(GIM, "quiet"), "q1_m"] == pytest.approx(100.0)
    assert result.loc[(GIM, "quiet"), "iqr_m"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# boxplot_stats
# ---------------------------------------------------------------------------


def test_boxplot_stats_tukey_fences_and_fliers_hand_computed():
    # 1..9 plus one far outlier (100). Q1 sits at position 0.25*9=2.25 (between the
    # 3rd and 4th sorted values, 3 and 4): 3 + 0.25*(4-3) = 3.25. Q3 at 0.75*9=6.75
    # (between 7 and 8): 7 + 0.75*(8-7) = 7.75. IQR = 4.5, so the fences are
    # 3.25 - 1.5*4.5 = -3.5 and 7.75 + 1.5*4.5 = 14.5 - both fences are outside the
    # 1..9 range, so the whiskers reach the actual min/max of that range (1 and 9),
    # and only 100 (> 14.5) is a flier.
    values = list(range(1, 10)) + [100.0]
    df = frame([(STEC, v) for v in values])

    stats, fliers = pdist.boxplot_stats(df, ["Method"])
    row = stats.set_index("Method").loc[STEC]

    assert row["n"] == 10
    assert row["q1_m"] == pytest.approx(3.25)
    assert row["q3_m"] == pytest.approx(7.75)
    assert row["median_m"] == pytest.approx(5.5)  # mean(5, 6)
    assert row["whislo_m"] == pytest.approx(1.0)
    assert row["whishi_m"] == pytest.approx(9.0)
    assert row["min_m"] == pytest.approx(1.0)
    assert row["max_m"] == pytest.approx(100.0)
    assert row["n_fliers"] == 1
    assert row["pct_fliers"] == pytest.approx(10.0)

    assert fliers["Method"].tolist() == [STEC]
    assert fliers["error_3d_rms"].tolist() == pytest.approx([100.0])


def test_boxplot_stats_no_fliers_when_all_points_within_fences():
    # A tight, symmetric spread where nothing exceeds 1.5xIQR: fliers must be empty,
    # and the whiskers must equal the true min/max exactly.
    df = frame([(GIM, v) for v in [2.0, 3.0, 4.0, 5.0, 6.0]])
    stats, fliers = pdist.boxplot_stats(df, ["Method"])
    row = stats.set_index("Method").loc[GIM]

    assert row["n_fliers"] == 0
    assert row["whislo_m"] == pytest.approx(2.0)
    assert row["whishi_m"] == pytest.approx(6.0)
    assert fliers.empty


def test_boxplot_stats_groups_by_multiple_columns_independently():
    df = pd.DataFrame(
        [
            (STEC, "original", 1.0),
            (STEC, "original", 2.0),
            (STEC, "recovered", 50.0),
            (STEC, "recovered", 60.0),
        ],
        columns=["Method", "population", "error_3d_rms"],
    )
    stats, fliers = pdist.boxplot_stats(df, ["Method", "population"])
    indexed = stats.set_index(["Method", "population"])

    assert indexed.loc[(STEC, "original"), "median_m"] == pytest.approx(1.5)
    assert indexed.loc[(STEC, "recovered"), "median_m"] == pytest.approx(55.0)
    assert fliers.empty  # only 2 points per group - both always within the fences


# ---------------------------------------------------------------------------
# cdf_points
# ---------------------------------------------------------------------------


def test_cdf_points_sorts_and_computes_empirical_percentile_by_hand():
    # Deliberately unsorted input - the function must sort before assigning percentiles.
    df = frame([(STEC, 3.0), (STEC, 1.0), (STEC, 2.0)])
    result = pdist.cdf_points(df, ["Method"])

    assert result["error_3d_rms"].tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert result["cumulative_pct"].tolist() == pytest.approx([100 / 3, 200 / 3, 100.0])


def test_cdf_points_keeps_groups_separate():
    df = frame([(STEC, 10.0), (GIM, 1.0), (GIM, 2.0)])
    result = pdist.cdf_points(df, ["Method"])

    stec_rows = result[result["Method"] == STEC]
    gim_rows = result[result["Method"] == GIM]
    assert stec_rows["cumulative_pct"].tolist() == pytest.approx([100.0])
    assert gim_rows["cumulative_pct"].tolist() == pytest.approx([50.0, 100.0])


# ---------------------------------------------------------------------------
# exceedance_table (thin wrapper around positioning_diagnostics.outlier_threshold_counts)
# ---------------------------------------------------------------------------


def test_exceedance_table_counts_and_mass_fraction_hand_computed():
    # Values 1..5, threshold 2: station-days exceeding are 3, 4, 5 (n=3 of 5 -> 60%).
    # Total error mass is 1+2+3+4+5=15; the mass of the exceeding rows is 3+4+5=12,
    # which is 12/15 = 80% of the total.
    df = frame([(STEC, v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]])
    result = pdist.exceedance_table(df, ["Method"], thresholds=(2.0,))
    row = result.set_index("threshold_m").loc[2.0]

    assert row["n_exceeding"] == 3
    assert row["pct_station_days_exceeding"] == pytest.approx(60.0)
    assert row["error_mass_fraction_pct"] == pytest.approx(80.0)


def test_exceedance_table_reports_every_requested_threshold():
    df = frame([(STEC, v) for v in [1.0, 6.0, 11.0, 21.0, 51.0]])
    result = pdist.exceedance_table(df, ["Method"])  # default thresholds: 5/10/20/50 m
    counts = result.set_index("threshold_m")["n_exceeding"]

    assert counts.loc[5.0] == 4  # 6, 11, 21, 51 all > 5
    assert counts.loc[10.0] == 3  # 11, 21, 51
    assert counts.loc[20.0] == 2  # 21, 51
    assert counts.loc[50.0] == 1  # 51


# ---------------------------------------------------------------------------
# load_recovered_station_days_cached
# ---------------------------------------------------------------------------


def test_load_recovered_station_days_cached_prefers_the_cache_file(tmp_path):
    cache_path = tmp_path / "recovered_station_days.csv"
    pd.DataFrame([{"year": 2024, "doy": 122, "station": "AAAA"}]).to_csv(
        cache_path, index=False
    )
    # A root that does not exist proves the cache was actually used, not a fallback scan.
    missing_root = tmp_path / "does_not_exist"

    result = pdist.load_recovered_station_days_cached(cache_path, missing_root)

    assert result["station"].tolist() == ["AAAA"]


def test_load_recovered_station_days_cached_falls_back_to_a_fresh_scan(tmp_path):
    missing_cache = tmp_path / "no_cache_here.csv"
    empty_root = tmp_path / "empty_root"
    empty_root.mkdir()

    result = pdist.load_recovered_station_days_cached(missing_cache, empty_root)

    assert list(result.columns) == ["year", "doy", "station"]
    assert result.empty


# ---------------------------------------------------------------------------
# _format_table5_markdown - the exact numbers written to TABLE5_NUMBERS.md
# ---------------------------------------------------------------------------


def test_format_table5_markdown_reports_median_iqr_and_exceedance_per_method():
    df = frame(
        [(STEC, v) for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        + [(GIM, v) for v in [2.0, 4.0, 6.0, 8.0, 10.0]]
    )
    percentiles = pdist.percentile_summary(df, ["Method"])
    exceedance = pdist.exceedance_table(df, ["Method"])

    markdown = pdist._format_table5_markdown(percentiles, exceedance, len(df))

    assert "no outcome-based outlier" in markdown
    assert STEC in markdown and GIM in markdown
    # Direct STEC median = 3.0, IGS GIM median = 6.0 -> 50% improvement.
    assert "**50.0%**" in markdown
