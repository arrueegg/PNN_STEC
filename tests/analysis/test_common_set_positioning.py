"""Tests for `stec.analysis.common_set_positioning` (Table A1, R1.5).

Fixtures are written as small CSVs under `tmp_path` rather than depending on the live
checkout's positioning trees - `build()` reads paths, so this exercises the real read
path (`load_tree`, `load_pretrained_elev`) rather than only the in-memory aggregation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import common_set_positioning as csp


def row(station: str, doy: int, method: str, error_3d: float) -> dict:
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d,
        "error_2d_rms": error_3d / 2,
        "u_rms": error_3d / 4,
    }


def write_csv(path, rows: list[dict]) -> None:
    # An empty `rows` still needs the header row - `load_tree` reads with
    # `usecols=lambda c: c in COLUMNS`, which fails on a headerless empty file.
    frame = (
        pd.DataFrame(rows, columns=csp.COLUMNS)
        if rows
        else pd.DataFrame(columns=csp.COLUMNS)
    )
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# build(): intersection across arms, and the reported N
# ---------------------------------------------------------------------------


def test_build_restricts_to_the_station_days_solved_by_every_arm(tmp_path):
    """STEC_iono is missing ZIMM/132 (present for gim_iono), so ZIMM/132 must be dropped
    from the common set even though gim_iono solved it - the intersection, not the union."""
    three_way = tmp_path / "three_way.csv"
    write_csv(
        three_way,
        [
            row("AMC4", 132, "STEC_iono", 2.0),
            row("AMC4", 132, "gim_iono", 4.0),
            row("ZIMM", 132, "gim_iono", 5.0),  # STEC_iono missing for ZIMM/132
        ],
    )
    ablation = tmp_path / "ablation.csv"
    write_csv(ablation, [])  # no additional arms in this fixture
    empty_experiment = tmp_path / "no_such_experiment"

    result = csp.build(three_way, ablation, empty_experiment)

    assert (
        result["arms"] == 2
    )  # "Direct STEC / uncertainty", "IGS GIM + Mapping / uncertainty"
    assert result["common_station_days"] == 1  # only AMC4/132 solved by both arms
    summary = result["summary"]
    assert summary.loc["Direct STEC / uncertainty", "station_days"] == 1
    assert summary.loc["IGS GIM + Mapping / uncertainty", "station_days"] == 1
    # gim_iono had 2 station-days before the intersection, 1 after.
    assert summary.loc["IGS GIM + Mapping / uncertainty", "lost_to_intersection"] == 1
    assert summary.loc["Direct STEC / uncertainty", "lost_to_intersection"] == 0


def test_build_reports_gain_relative_to_the_uncertainty_weighted_gim_baseline(tmp_path):
    three_way = tmp_path / "three_way.csv"
    write_csv(
        three_way,
        [
            row("AMC4", 132, "STEC_iono", 3.0),
            row("AMC4", 132, "gim_iono", 6.0),
            row("ZIMM", 133, "STEC_iono", 1.0),
            row("ZIMM", 133, "gim_iono", 4.0),
        ],
    )
    ablation = tmp_path / "ablation.csv"
    write_csv(ablation, [])
    empty_experiment = tmp_path / "no_such_experiment"

    result = csp.build(three_way, ablation, empty_experiment)
    summary = result["summary"]

    # Direct STEC beats the GIM baseline on both station-days: 3 vs 6, 1 vs 4.
    stec_row = summary.loc["Direct STEC / uncertainty"]
    assert stec_row["win_rate_pct"] == pytest.approx(100.0)
    assert stec_row["gain_paired_mean_pct"] > 0
    # The baseline compared with itself must show zero gain and a 0% win rate.
    gim_row = summary.loc["IGS GIM + Mapping / uncertainty"]
    assert gim_row["gain_paired_mean_pct"] == pytest.approx(0.0)
    assert gim_row["win_rate_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Outlier rule: corrected to <= (matching positioning_summary/oracle_benchmark)
# ---------------------------------------------------------------------------


def test_outlier_boundary_at_exactly_10m_is_kept_not_dropped(tmp_path):
    """The live checkout's version of this analysis used a strict `<` here while the
    other two positioning analyses use `<=`; this port standardises on `<=` via
    `pm.exclude_outlier_station_days`, so a station-day at exactly 10.0 m must survive."""
    three_way = tmp_path / "three_way.csv"
    write_csv(
        three_way,
        [
            row("AMC4", 132, "STEC_iono", 10.0),  # exactly at the boundary
            row("AMC4", 132, "gim_iono", 10.0),
        ],
    )
    ablation = tmp_path / "ablation.csv"
    write_csv(ablation, [])
    empty_experiment = tmp_path / "no_such_experiment"

    result = csp.build(three_way, ablation, empty_experiment)

    assert result["common_station_days"] == 1


# ---------------------------------------------------------------------------
# load_pretrained_elev: the one arm read from per-day summaries, not a tree CSV
# ---------------------------------------------------------------------------


def test_load_pretrained_elev_reads_and_relabels_model_rows(tmp_path):
    experiment = tmp_path / "Pretrain_STEC_example"
    day_dir = experiment / "positioning" / "results" / "2024132"
    day_dir.mkdir(parents=True)
    write_csv(
        day_dir / "daily_summary.csv",
        [
            row("AMC4", 132, "model", 3.0),  # the pretrained elevation run
            row("AMC4", 132, "gim", 5.0),  # not the arm this function extracts
        ],
    )

    frame = csp.load_pretrained_elev(experiment)

    assert list(frame["method"]) == ["Pretrained_STEC_elev"]
    assert frame.iloc[0]["error_3d_rms"] == pytest.approx(3.0)


def test_load_pretrained_elev_returns_empty_frame_when_experiment_has_no_summaries(
    tmp_path,
):
    frame = csp.load_pretrained_elev(tmp_path / "no_such_experiment")

    assert frame.empty
    assert list(frame.columns) == csp.COLUMNS
