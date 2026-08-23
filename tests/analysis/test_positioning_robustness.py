"""Tests for `stec.analysis.positioning_robustness` (R2.7b).

Tail statistics (percentiles, fraction above a threshold) and the horizontal/vertical
split are checked against hand-computed values on a small synthetic frame, since both are
plain pandas reductions that are easy to get subtly wrong (wrong quantile interpolation,
wrong axis for the ratio).
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import positioning_robustness as pr


def _row(
    station: str,
    doy: int,
    method: str,
    error_3d_rms: float,
    error_3d_95th: float,
    error_2d_rms: float,
    u_rms: float,
    e_rms: float,
    n_rms: float,
) -> dict:
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d_rms,
        "error_3d_95th": error_3d_95th,
        "error_2d_rms": error_2d_rms,
        "u_rms": u_rms,
        "e_rms": e_rms,
        "n_rms": n_rms,
    }


@pytest.fixture
def direct_stec_frame() -> pd.DataFrame:
    """Four Direct STEC station-days with hand-pickable error_3d_rms values so p50/p90
    are unambiguous under pandas' default (linear) quantile interpolation."""
    rows = [
        _row("AMC4", 130, "STEC_iono", 1.0, 1.5, 0.6, 0.3, 0.4, 0.4),
        _row("ZIMM", 131, "STEC_iono", 2.0, 2.5, 1.2, 0.6, 0.8, 0.8),
        _row("WTZR", 132, "STEC_iono", 3.0, 3.5, 1.8, 0.9, 1.2, 1.2),
        _row("ONSA", 133, "STEC_iono", 4.0, 4.5, 2.4, 1.2, 1.6, 1.6),
    ]
    return pd.DataFrame(rows)


def _write(path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load: outlier exclusion and method mapping
# ---------------------------------------------------------------------------


def test_load_excludes_outliers_above_10m_and_drops_unmapped_methods(tmp_path):
    frame = pd.concat(
        [
            pd.DataFrame(
                [_row("AMC4", 130, "STEC_iono", 1.0, 1.5, 0.6, 0.3, 0.4, 0.4)]
            ),
            pd.DataFrame(
                [_row("ZIMM", 131, "STEC_iono", 15.0, 15.0, 9.0, 4.5, 6.0, 6.0)]
            ),
            pd.DataFrame([_row("WTZR", 132, "unmapped", 2.0, 2.0, 1.0, 0.5, 0.5, 0.5)]),
        ],
        ignore_index=True,
    )
    summary_path = tmp_path / "summary.csv"
    _write(summary_path, frame)

    loaded = pr.load(summary_path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["station"] == "AMC4"
    assert loaded.iloc[0]["Method"] == "Direct STEC"


# ---------------------------------------------------------------------------
# tail_table: hand-computed percentiles and threshold fractions
# ---------------------------------------------------------------------------


def test_tail_table_matches_hand_computed_percentiles(direct_stec_frame):
    direct_stec_frame["Method"] = "Direct STEC"

    tails = pr.tail_table(direct_stec_frame)
    row = tails.loc["Direct STEC"]

    assert row["station_days"] == 4
    assert row["mean"] == pytest.approx(2.5)
    # pandas' default linear interpolation on [1, 2, 3, 4]:
    assert row["median"] == pytest.approx(2.5)
    assert row["p90"] == pytest.approx(3.7)  # 1 + 0.90 * (4 - 1) = 3.7
    assert row["p95"] == pytest.approx(3.85)  # 1 + 0.95 * 3
    assert row["p99"] == pytest.approx(3.97)  # 1 + 0.99 * 3
    assert row["mean_daily_95th_pct"] == pytest.approx((1.5 + 2.5 + 3.5 + 4.5) / 4)


def test_tail_table_fraction_above_threshold_matches_hand_count(direct_stec_frame):
    direct_stec_frame["Method"] = "Direct STEC"

    tails = pr.tail_table(direct_stec_frame)
    row = tails.loc["Direct STEC"]

    # error_3d_rms = [1, 2, 3, 4]; strictly above 2.0 -> {3, 4} -> 2/4 = 50%.
    assert row["frac_above_2m_%"] == pytest.approx(50.0)
    # strictly above 3.0 -> {4} -> 1/4 = 25%.
    assert row["frac_above_3m_%"] == pytest.approx(25.0)
    # strictly above 5.0 -> none -> 0%.
    assert row["frac_above_5m_%"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# component_table: horizontal (2D) vs vertical (Up) split
# ---------------------------------------------------------------------------


def test_component_table_matches_hand_computed_horizontal_and_vertical_means(
    direct_stec_frame,
):
    direct_stec_frame["Method"] = "Direct STEC"

    components = pr.component_table(direct_stec_frame)
    row = components.loc["Direct STEC"]

    expected_horizontal = (0.6 + 1.2 + 1.8 + 2.4) / 4
    expected_vertical = (0.3 + 0.6 + 0.9 + 1.2) / 4
    assert row["horizontal_2D_rms"] == pytest.approx(expected_horizontal)
    assert row["vertical_up_rms"] == pytest.approx(expected_vertical)
    assert row["vertical_to_horizontal_ratio"] == pytest.approx(
        expected_vertical / expected_horizontal
    )
    assert row["east_rms"] == pytest.approx((0.4 + 0.8 + 1.2 + 1.6) / 4)
    assert row["north_rms"] == pytest.approx((0.4 + 0.8 + 1.2 + 1.6) / 4)


# ---------------------------------------------------------------------------
# canonical_positioning_summary: prefer full coverage, fall back to published
# ---------------------------------------------------------------------------


def test_canonical_prefers_full_coverage_when_present(tmp_path, monkeypatch):
    full = tmp_path / "full.csv"
    published = tmp_path / "published.csv"
    full.touch()
    published.touch()
    monkeypatch.setattr(pr, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(pr, "PUBLISHED_SUMMARY", published)

    assert pr.canonical_positioning_summary() == full


def test_canonical_falls_back_to_published_when_full_coverage_missing(
    tmp_path, monkeypatch
):
    full = tmp_path / "does_not_exist.csv"
    published = tmp_path / "published.csv"
    published.touch()
    monkeypatch.setattr(pr, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(pr, "PUBLISHED_SUMMARY", published)

    assert pr.canonical_positioning_summary() == published
