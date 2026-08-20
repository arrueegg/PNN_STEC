"""Pins the mapping-convention mismatch computation independent of any raw h5 file.

`mapping_mismatch` is the pure part of the port - `mismatch_for_day` only adds the
h5py read around it - so these tests build the stec/vtec/elevation arrays directly
rather than a synthetic STEC-database file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis.mapping_function_consistency import ELEVATION_BINS, mapping_mismatch
from stec.baselines.gim import MappingFunction


def _bin_for(elevation_deg: float) -> pd.Interval:
    """Which ELEVATION_BINS interval one elevation value falls into."""
    return pd.cut([elevation_deg], bins=ELEVATION_BINS)[0]


def test_mismatch_is_zero_at_zenith():
    """Both SLM and MSLM equal 1.0 exactly at 90 degrees elevation, so a reference
    that used *either* convention at zenith must reproduce our MSLM exactly there."""
    vtec = np.array([20.0, 35.0, 50.0])
    elevation = np.array([90.0, 90.0, 90.0])
    stec = vtec.copy()  # zenith: stec == vtec under any thin-shell convention

    result = mapping_mismatch(stec, vtec, elevation, MappingFunction("MSLM"))

    row = result.loc[_bin_for(90.0)]
    assert row["n"] == 3
    assert row["sum_abs"] == pytest.approx(0.0, abs=1e-9)
    assert row["sum_sq"] == pytest.approx(0.0, abs=1e-9)


def test_mismatch_matches_hand_computation_at_a_pinned_low_elevation():
    """Simulate a reference that mapped with SLM while we map with MSLM: the two
    conventions diverge at low elevation (see stec/baselines/gim.py), and the
    resulting discrepancy must equal vtec * (slm_factor - mslm_factor) exactly - pinned
    at 10 degrees, where the two factors are known to differ by construction."""
    vtec_value = 30.0
    elevation_deg = 10.0

    slm_factor = MappingFunction("SLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    mslm_factor = MappingFunction("MSLM").get_mapping_factor(
        np.radians(np.array([elevation_deg]))
    )[0]
    expected_difference = vtec_value * (slm_factor - mslm_factor)
    assert abs(expected_difference) > 1.0  # sanity: not a rounding-scale effect

    stec_from_reference_slm = np.array([vtec_value * slm_factor])
    result = mapping_mismatch(
        stec_from_reference_slm,
        np.array([vtec_value]),
        np.array([elevation_deg]),
        MappingFunction("MSLM"),
    )

    row = result.loc[_bin_for(elevation_deg)]
    assert row["sum_signed"] == pytest.approx(expected_difference, rel=1e-9)
    assert row["sum_abs"] == pytest.approx(abs(expected_difference), rel=1e-9)


def test_mismatch_magnitude_increases_toward_lower_elevation():
    """Not just non-zero, but monotonically larger the lower the elevation - this is
    the qualitative claim the paper makes about low-elevation mapping error."""
    vtec_value = 30.0
    elevations = np.array([80.0, 40.0, 15.0])  # descending
    slm = MappingFunction("SLM")
    mslm = MappingFunction("MSLM")

    stec_from_reference_slm = vtec_value * slm.get_mapping_factor(
        np.radians(elevations)
    )
    magnitudes = np.abs(
        stec_from_reference_slm
        - vtec_value * mslm.get_mapping_factor(np.radians(elevations))
    )
    assert np.all(np.diff(magnitudes) > 0)


def test_observations_below_5_degrees_and_nonfinite_pairs_are_excluded():
    # Each row is invalid for exactly one reason: non-finite stec, non-finite vtec,
    # and vtec == 0 (the mapping ratio is undefined) - none should survive filtering.
    stec = np.array([np.nan, 25.0, 25.0])
    vtec = np.array([20.0, np.nan, 0.0])
    elevation = np.array([45.0, 45.0, 45.0])

    result = mapping_mismatch(stec, vtec, elevation, MappingFunction("MSLM"))
    assert result["n"].sum() == 0

    stec2 = np.array([25.0])
    vtec2 = np.array([20.0])
    elevation2 = np.array([3.0])  # below the 5-degree floor
    result2 = mapping_mismatch(stec2, vtec2, elevation2, MappingFunction("MSLM"))
    assert result2["n"].sum() == 0


def test_elevation_bins_are_fixed_module_constants():
    """The bin edges must not depend on the data being summarised - two batches with
    very different elevation distributions must be partitioned into the same set of
    categories."""
    low_batch = np.array([6.0, 8.0, 10.0])
    high_batch = np.array([70.0, 80.0, 89.0])
    expected_categories = list(pd.cut(np.array([]), bins=ELEVATION_BINS).categories)

    for batch in (low_batch, high_batch):
        result = mapping_mismatch(batch + 1.0, batch, batch, MappingFunction("MSLM"))
        assert list(result.index.categories) == expected_categories
