"""Tests for `stec.analysis.positioning_summary` (Table 5).

All fixtures are small synthetic frames built in-memory or in `tmp_path` - no dependency
on the live checkout's positioning trees. `stec.positioning.metrics` already pins the
mean-of-station-days convention and the 10 m outlier boundary in isolation
(`tests/positioning/test_metrics.py`); the tests here confirm this module actually reuses
those functions end to end, rather than reimplementing them.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis import positioning_summary as psum
from stec.positioning import metrics as pm


def station_day(station: str, doy: int, method: str, error_3d: float) -> dict:
    """One station-day row with a fixed 2D/Up relationship to the 3D error, so every
    aggregate column has a known, hand-computable value."""
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d,
        "error_2d_rms": error_3d / 2,
        "u_rms": error_3d / 4,
    }


# ---------------------------------------------------------------------------
# summarise_overall: Table 5's four iono-weighted methods
# ---------------------------------------------------------------------------


def test_summarise_overall_is_mean_of_station_days_and_excludes_outliers():
    frame = pd.DataFrame(
        [
            station_day("AMC4", 132, "STEC_iono", 2.0),
            station_day("ZIMM", 132, "STEC_iono", 10.0),
            station_day("WTZR", 132, "STEC_iono", 15.0),  # excluded: > 10 m
            station_day("AMC4", 132, "gim_iono", 4.0),
            station_day("AMC4", 132, "unmapped_method", 1.0),  # dropped: not in Table 5
        ]
    )

    overall = psum.summarise_overall(frame)

    assert overall.loc["Direct STEC", "station_days"] == 2  # WTZR excluded
    assert overall.loc["Direct STEC", "3D_mean_m"] == pytest.approx(6.0)  # mean(2, 10)
    assert overall.loc["Direct STEC", "3D_median_m"] == pytest.approx(6.0)
    assert overall.loc["Direct STEC", "2D_mean_m"] == pytest.approx(3.0)
    assert overall.loc["Direct STEC", "Up_mean_m"] == pytest.approx(1.5)
    assert overall.loc["IGS GIM + Mapping", "station_days"] == 1

    # Method not present in this fixture must still appear (as NaN), via METHOD_ORDER.
    assert list(overall.index) == psum.METHOD_ORDER
    assert pd.isna(overall.loc["VTEC + Mapping", "station_days"])

    # The mean-of-station-days convention must differ from epoch/observation pooling -
    # confirms this fixture actually distinguishes the two conventions, not just in theory.
    pooled_rms = np.sqrt(np.mean(np.array([2.0, 10.0]) ** 2))
    assert overall.loc["Direct STEC", "3D_mean_m"] != pytest.approx(pooled_rms)


def test_summarise_overall_boundary_at_exactly_10m_is_kept():
    """Pins that this module reuses `pm.exclude_outlier_station_days` (<=), not a
    reimplementation - the boundary case is where a `<` vs `<=` slip would show up."""
    frame = pd.DataFrame(
        [
            station_day("AMC4", 132, "STEC_iono", 10.0),
            station_day("ZIMM", 132, "STEC_iono", 10.0001),
        ]
    )

    overall = psum.summarise_overall(frame)

    assert overall.loc["Direct STEC", "station_days"] == 1


def test_summarise_overall_reuses_the_named_outlier_constant():
    assert psum.pm.OUTLIER_3D_RMS_M == pm.OUTLIER_3D_RMS_M == 10.0


# ---------------------------------------------------------------------------
# summarise_by_regime: quiet vs storm split (R1.7)
# ---------------------------------------------------------------------------


def test_summarise_by_regime_splits_quiet_and_storm():
    frame = pd.DataFrame(
        [
            station_day("AMC4", 132, "STEC_iono", 2.0),  # storm
            station_day("ZIMM", 200, "STEC_iono", 4.0),  # quiet
        ]
    )
    storm_doys = {132}

    by_regime = psum.summarise_by_regime(frame, storm_doys)

    assert by_regime.loc[("Direct STEC", "storm"), "3D_mean_m"] == pytest.approx(2.0)
    assert by_regime.loc[("Direct STEC", "quiet"), "3D_mean_m"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# summarise_by_weighting: elevation vs uncertainty arms (R1.5)
# ---------------------------------------------------------------------------


def test_summarise_by_weighting_maps_method_to_method_and_weighting():
    frame = pd.DataFrame(
        [
            station_day("AMC4", 132, "STEC_elev", 3.0),
            station_day("AMC4", 132, "STEC_iono", 2.0),
            station_day("AMC4", 132, "unrelated_method", 99.0),
        ]
    )

    by_weighting = psum.summarise_by_weighting(frame)

    assert by_weighting.loc[("Direct STEC", "elevation"), "3D_mean_m"] == pytest.approx(
        3.0
    )
    assert by_weighting.loc[
        ("Direct STEC", "predicted uncertainty"), "3D_mean_m"
    ] == pytest.approx(2.0)
    assert ("unrelated_method",) not in by_weighting.index


# ---------------------------------------------------------------------------
# canonical_positioning_summary: prefer full coverage, fall back to published
# ---------------------------------------------------------------------------


def test_canonical_prefers_full_coverage_when_present(tmp_path, monkeypatch):
    full = tmp_path / "full.csv"
    published = tmp_path / "published.csv"
    full.touch()
    published.touch()
    monkeypatch.setattr(psum, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(psum, "PUBLISHED_SUMMARY", published)

    assert psum.canonical_positioning_summary() == full


def test_canonical_falls_back_to_published_when_full_coverage_missing(
    tmp_path, monkeypatch
):
    full = tmp_path / "does_not_exist.csv"
    published = tmp_path / "published.csv"
    published.touch()
    monkeypatch.setattr(psum, "FULL_COVERAGE_SUMMARY", full)
    monkeypatch.setattr(psum, "PUBLISHED_SUMMARY", published)

    assert psum.canonical_positioning_summary() == published


def test_canonical_prefer_argument_overrides_resolution(tmp_path):
    explicit = tmp_path / "explicit.csv"
    assert psum.canonical_positioning_summary(prefer=explicit) == explicit


# ---------------------------------------------------------------------------
# load_storm_doys: minimum daily Dst against the storm threshold
# ---------------------------------------------------------------------------


def _write_swi_fixture(path, doy_to_dst: dict[int, list[float]]) -> None:
    columns = ["Kp_index", "Dst-index,_nT", "f107_index"]
    with h5py.File(path, "w") as handle:
        group = handle.create_group("2024")
        for doy, dst_values in doy_to_dst.items():
            data = np.array([[0.0, dst, 0.0] for dst in dst_values], dtype=np.float64)
            dataset = group.create_dataset(f"{doy:03d}", data=data)
            dataset.attrs["columns"] = columns


def test_load_storm_doys_flags_days_at_or_below_threshold(tmp_path):
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture(
        swi_path,
        {
            132: [-10.0, -20.0],  # min -20, quiet
            133: [-10.0, -50.0],  # min -50, storm (== threshold)
            134: [-60.0, -5.0],  # min -60, storm
        },
    )

    storm_doys = psum.load_storm_doys(swi_path, 2024)

    assert storm_doys == {133, 134}


def test_load_storm_doys_returns_none_when_file_missing(tmp_path):
    assert psum.load_storm_doys(tmp_path / "missing.h5", 2024) is None
