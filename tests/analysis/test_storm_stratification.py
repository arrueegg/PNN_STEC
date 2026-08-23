"""Tests for `stec.analysis.storm_stratification` (R2.7).

All fixtures are small synthetic frames built in `tmp_path` - no dependency on the live
checkout's positioning or OMNI trees. `stec.positioning.metrics` already pins the
mean-of-station-days convention and the 10 m outlier boundary in isolation
(`tests/positioning/test_metrics.py`); the tests here focus on what this module adds:
the combined Kp/Dst storm threshold and the regime split built on top of it.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis import storm_stratification as ss

YEAR = 2024


def _write_swi_fixture(
    path, doy_to_hourly: dict[int, list[tuple[float, float]]]
) -> None:
    """`doy_to_hourly` maps doy -> list of (kp, dst) hourly pairs for that day."""
    columns = ["Kp_index", "Dst-index,_nT", "f107_index"]
    with h5py.File(path, "w") as handle:
        group = handle.create_group(str(YEAR))
        for doy, hourly in doy_to_hourly.items():
            data = np.array([[kp, dst, 0.0] for kp, dst in hourly], dtype=np.float64)
            dataset = group.create_dataset(f"{doy:03d}", data=data)
            dataset.attrs["columns"] = columns


def _positioning_summary(path, doys: list[int]) -> None:
    """One STEC_iono and one gim_iono station-day per doy, well under the 10 m outlier
    cutoff.

    Includes error_2d_rms/u_rms because `build_tables` delegates to
    `stec.positioning.metrics.summarise`, which reads them alongside error_3d_rms. Both
    methods are present because `build_tables`'s improvement-over-GIM table needs a GIM
    baseline for every regime it reports.
    """
    frame = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": doy,
                "method": method,
                "error_3d_rms": 1.0,
                "error_2d_rms": 0.6,
                "u_rms": 0.3,
            }
            for doy in doys
            for method in ("STEC_iono", "gim_iono")
        ]
    )
    frame.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# load_daily_geomagnetic_indices: per-day extremes from the hourly archive
# ---------------------------------------------------------------------------


def test_load_daily_geomagnetic_indices_takes_the_daily_extreme_not_the_mean(tmp_path):
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture(swi_path, {132: [(10.0, -5.0), (20.0, -60.0), (5.0, -5.0)]})

    indices = ss.load_daily_geomagnetic_indices(YEAR, swi_path)

    row = indices.set_index("doy").loc[132]
    assert row["kp_max"] == pytest.approx(20.0)
    assert row["dst_min"] == pytest.approx(-60.0)


# ---------------------------------------------------------------------------
# Storm threshold: daily min Dst <= -50, pinned exactly at the boundary.
# ---------------------------------------------------------------------------


def test_storm_threshold_is_the_daily_rule_that_produced_the_published_table():
    """This module answers the positioning question, which is about whole days.

    The per-observation rule in scenario_evaluation.py (Kp >= 37 or Dst <= -33) is a
    different test for a different question, not a variant of this one: applied to days it
    marks 102 of 242 as storms against 39, and moves the published +31.9%/+26.3% to
    +32.2%/+29.1%. Both constants are exposed so the distinction is visible, but only the
    daily one is used here.
    """
    assert ss.STORM_DST_THRESHOLD_NT == -50.0
    assert ss.SCENARIO_KP_THRESHOLD == 37.0
    assert ss.SCENARIO_DST_THRESHOLD_NT == -33.0


@pytest.fixture
def boundary_fixture(tmp_path):
    doy_to_hourly = {
        130: [(20.0, -10.0)],  # quiet
        131: [
            (90.0, -10.0)
        ],  # quiet: high Kp does not make a day a storm under this rule
        132: [
            (10.0, -49.9)
        ],  # quiet: Dst just above (less negative than) the threshold
        133: [(10.0, -50.0)],  # storm: Dst exactly at the threshold
        134: [(10.0, -80.0)],  # storm: well past it
        135: [(40.0, -33.0)],  # quiet: the per-observation threshold is not this one
    }
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture(swi_path, doy_to_hourly)
    summary_path = tmp_path / "summary.csv"
    _positioning_summary(summary_path, list(doy_to_hourly))
    return summary_path, swi_path


def test_stratify_lands_each_boundary_day_on_the_correct_side(boundary_fixture):
    summary_path, swi_path = boundary_fixture

    stratified = ss.stratify(summary_path, YEAR, swi_path)
    regime_by_doy = stratified.set_index("doy")["regime"].to_dict()

    assert regime_by_doy == {
        130: "quiet",
        131: "quiet",
        132: "quiet",
        133: "storm",  # Dst == -50.0, the boundary belongs to storm
        134: "storm",
        135: "quiet",
    }


# ---------------------------------------------------------------------------
# No enable/disable flag: the analysis must run whenever it is invoked, unlike
# `evaluation.enable_scenarios` (defaults False) in the pre-rebuild config.
# ---------------------------------------------------------------------------


def test_stratification_always_produces_both_regimes_with_no_enabling_flag(
    boundary_fixture,
):
    """Regression guard for the class of bug documented in the project notes:
    `evaluation.enable_scenarios` defaulted to False and silently skipped the equivalent
    per-observation stratification for weeks. `stratify`/`build_tables` take no flag that
    could leave the regime split off by default - calling them with only the required
    arguments must produce both regimes, never an empty or single-regime result."""
    summary_path, swi_path = boundary_fixture

    stratified = ss.stratify(summary_path, YEAR, swi_path)
    assert set(stratified["regime"]) == {"storm", "quiet"}

    tables = ss.build_tables(stratified)
    regimes_present = set(tables["by_regime"].index.get_level_values("regime"))
    assert regimes_present == {"storm", "quiet"}


# ---------------------------------------------------------------------------
# build_tables: degradation and improvement-over-GIM derived from the regime split
# ---------------------------------------------------------------------------


def test_build_tables_gim_improvement_over_itself_is_zero(tmp_path):
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture(swi_path, {130: [(20.0, -10.0)], 131: [(20.0, -80.0)]})
    summary_path = tmp_path / "summary.csv"
    common = {"error_2d_rms": 0.6, "u_rms": 0.3}
    frame = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 130,
                "method": "STEC_iono",
                "error_3d_rms": 1.0,
                **common,
            },
            {
                "station": "AMC4",
                "doy": 130,
                "method": "gim_iono",
                "error_3d_rms": 2.0,
                **common,
            },
            {
                "station": "AMC4",
                "doy": 131,
                "method": "STEC_iono",
                "error_3d_rms": 3.0,
                **common,
            },
            {
                "station": "AMC4",
                "doy": 131,
                "method": "gim_iono",
                "error_3d_rms": 4.0,
                **common,
            },
        ]
    )
    frame.to_csv(summary_path, index=False)

    stratified = ss.stratify(summary_path, YEAR, swi_path)
    tables = ss.build_tables(stratified)

    improvement = tables["improvement_over_gim"]
    assert improvement.loc[
        ss.GIM_LABEL, "improvement_over_gim_quiet_%"
    ] == pytest.approx(0.0)
    assert improvement.loc[
        ss.GIM_LABEL, "improvement_over_gim_storm_%"
    ] == pytest.approx(0.0)
    # Direct STEC (1.0) beats GIM (2.0) in quiet by 50%.
    assert improvement.loc[
        "Direct STEC", "improvement_over_gim_quiet_%"
    ] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# canonical_positioning_summary: prefer full coverage, fall back to published
# ---------------------------------------------------------------------------


def test_canonical_prefers_full_coverage_when_present(tmp_path, monkeypatch):
    full = tmp_path / "full.csv"
    published = tmp_path / "published.csv"
    full.touch()
    published.touch()
    monkeypatch.setattr(ss, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(ss, "PUBLISHED_SUMMARY", published)

    assert ss.canonical_positioning_summary() == full


def test_canonical_falls_back_to_published_when_full_coverage_missing(
    tmp_path, monkeypatch
):
    full = tmp_path / "does_not_exist.csv"
    published = tmp_path / "published.csv"
    published.touch()
    monkeypatch.setattr(ss, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(ss, "PUBLISHED_SUMMARY", published)

    assert ss.canonical_positioning_summary() == published
