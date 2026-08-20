"""Reading a day out of the database, including the three derived details."""

from __future__ import annotations

import numpy as np
import pytest

from stec.data.day_reader import HOURS_PER_DAY, compute_local_time_hours, read_day
from stec.config import paths

DATABASE_AVAILABLE = paths.stec_database_day(2024, 132).exists()


def test_local_time_is_utc_shifted_by_longitude():
    """15 degrees of longitude is one hour."""
    noon_utc = np.array([43200.0])
    assert compute_local_time_hours(noon_utc, np.array([0.0]))[0] == pytest.approx(12.0)
    assert compute_local_time_hours(noon_utc, np.array([15.0]))[0] == pytest.approx(
        13.0
    )
    assert compute_local_time_hours(noon_utc, np.array([-15.0]))[0] == pytest.approx(
        11.0
    )


def test_local_time_wraps_into_the_day():
    """It must land in [0, 24), not run off either end."""
    late = compute_local_time_hours(np.array([86000.0]), np.array([170.0]))
    early = compute_local_time_hours(np.array([100.0]), np.array([-170.0]))
    for value in (*late, *early):
        assert 0.0 <= value < HOURS_PER_DAY


def test_local_time_uses_ipp_longitude_not_station():
    """Local time describes where the ray pierces the ionosphere, not where the receiver is."""
    sod = np.array([0.0])
    at_ipp = compute_local_time_hours(sod, np.array([90.0]))
    at_station = compute_local_time_hours(sod, np.array([0.0]))
    assert at_ipp[0] != at_station[0]


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="STEC database not available")
def test_a_real_day_reads_the_columns_the_layout_needs():
    columns = read_day(2024, 132, split="test")
    for required in ("sod", "satele", "satazi", "lat_ipp", "sm_lon_sta", "stec"):
        assert required in columns


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="STEC database not available")
def test_year_and_doy_come_from_the_file_not_the_rows():
    """The table has no such columns; deriving them from data is where float32 drift starts."""
    columns = read_day(2024, 132, split="test")
    assert np.unique(columns["year"]).tolist() == [2024.0]
    assert np.unique(columns["doy"]).tolist() == [132.0]


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="STEC database not available")
def test_row_count_matches_the_split_index():
    """File order and count must be preserved: index joins back to the table depend on it."""
    import h5py  # noqa: PLC0415

    columns = read_day(2024, 132, split="test")
    with h5py.File(paths.stec_database_day(2024, 132), "r") as handle:
        expected = len(handle["2024/132/test_idx"])
    assert len(columns["stec"]) == expected


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="STEC database not available")
def test_a_missing_day_is_an_error_not_an_empty_frame():
    with pytest.raises(FileNotFoundError):
        read_day(1999, 1, split="test")


@pytest.mark.skipif(not DATABASE_AVAILABLE, reason="STEC database not available")
def test_an_unknown_split_is_an_error():
    with pytest.raises(KeyError):
        read_day(2024, 132, split="not_a_split")
