"""Reading a day out of the database, including the three derived details."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from stec.data.day_reader import (
    HOURS_PER_DAY,
    compute_local_time_hours,
    read_day,
    read_space_weather,
)
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


def write_swi_day(path, year: int, doy: int, column_names: list[str]) -> None:
    """A minimal synthetic OMNI file: one day's 24 hourly rows, given column order.

    `read_space_weather` indexes `handle[f"{year}/{doy:03d}"]` directly as the dataset
    (h5py creates the intermediate groups implicitly from the slashed name), with the
    column order recovered from its `columns` attribute - mirroring the real file.
    """
    with h5py.File(path, "w") as handle:
        table = handle.create_dataset(
            f"{year}/{doy:03d}",
            data=np.arange(24 * len(column_names)).reshape(24, len(column_names)),
        )
        table.attrs["columns"] = column_names


def test_a_registry_column_the_file_does_not_carry_is_simply_absent(tmp_path):
    """Not a 0.0 fallback: the legacy per-row `if in_idx is None: value = 0.0` is not
    preserved here. `f107_index` is a real registry SWI name, deliberately left out of
    this file's own column list to stand in for "genuinely missing"."""
    swi_path = tmp_path / "omni.h5"
    write_swi_day(swi_path, 2024, 132, ["YEAR", "DOY", "HR", "Kp_index"])

    columns = read_space_weather(2024, 132, path=swi_path)

    assert "Kp_index" in columns
    assert "f107_index" not in columns
    assert "YEAR" not in columns, "the index columns are masked out, not just unfound"
