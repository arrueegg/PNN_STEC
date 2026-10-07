"""Tests for positioning/geometry/independent_geometry.py (SP3 + RINEX geometry)."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "positioning" / "geometry"))

import independent_geometry as ig  # noqa: E402
from stec.data.coordinate_transforms import calculate_ipp_coordinates  # noqa: E402

# MRO1 (Murchison, Australia) from the 2024 DOY 133 SINEX; a southern station, so a
# latitude sign error shows up immediately.
MRO1_XYZ_M = np.array([-2556630.2462, 5097138.3138, -2848384.5606])
MRO1_LATITUDE_DEG = -26.6966


def test_southern_station_latitude_is_negative():
    longitude, latitude, _ = ig.ecef_to_geodetic_arrays(MRO1_XYZ_M)
    assert latitude == pytest.approx(MRO1_LATITUDE_DEG, abs=0.001)
    assert longitude == pytest.approx(116.637, abs=0.001)


def test_lagrange_interpolation_reproduces_circular_orbit():
    radius_m, period_s = 26_560_000.0, 43_000.0
    node_times = np.arange(0.0, 86_400.0 + 1, 300.0)

    def orbit(times):
        angle = 2 * np.pi * times / period_s
        return np.column_stack(
            (radius_m * np.cos(angle), radius_m * np.sin(angle), np.zeros_like(times))
        )

    query_times = np.array([0.0, 0.03, 137.0, 43_210.5, 86_399.97, 86_400.0])
    interpolated = ig.interpolate_positions(node_times, orbit(node_times), query_times)
    assert np.abs(interpolated - orbit(query_times)).max() < 0.01  # metres, incl. edges


def test_elevation_azimuth_overhead_and_north_horizon():
    station = np.array([ig.WGS84_SEMI_MAJOR_AXIS_M, 0.0, 0.0])  # lat 0, lon 0
    overhead = station + np.array([2.0e7, 0.0, 0.0])
    due_north = station + np.array([0.0, 0.0, 2.0e7])
    elevation, azimuth = ig.elevation_azimuth_deg(
        station, np.vstack((overhead, due_north)), 0.0, 0.0
    )
    assert elevation[0] == pytest.approx(90.0)
    assert elevation[1] == pytest.approx(0.0, abs=1e-9)
    assert azimuth[1] == pytest.approx(0.0, abs=1e-9)


def test_ipp_matches_reference_formula_to_shell_radius_difference():
    azimuth = np.array([10.0, 100.0, 200.0, 300.0])
    elevation = np.array([20.0, 45.0, 10.0, 80.0])
    for station_latitude in (-26.7, 0.4, 64.1):
        ours = ig.ionospheric_pierce_point_deg(
            station_latitude, 12.0, azimuth, elevation
        )
        reference = calculate_ipp_coordinates(
            station_latitude, 12.0, azimuth, elevation
        )
        # The reference uses a 6371 km sphere, the database 6378.137 km: ~0.012 deg.
        assert np.abs(ours[0] - reference[0]).max() < 0.02
        assert np.abs(ours[1] - reference[1]).max() < 0.02


def test_ipp_moves_toward_the_satellite_in_both_hemispheres():
    north_lat, _ = ig.ionospheric_pierce_point_deg(
        50.0, 0.0, np.array([0.0]), np.array([30.0])
    )
    south_lat, _ = ig.ionospheric_pierce_point_deg(
        -50.0, 0.0, np.array([180.0]), np.array([30.0])
    )
    assert north_lat[0] > 50.0
    assert south_lat[0] < -50.0


SP3_TEXT = """#dP2024  5 12  0  0  0.00000000       3 d+D   IGS20 FIT AIUB
*  2024  5 12  0  0  0.00000000
PG01  20000.000000      0.000000      0.000000    100.000000
PE04 999999.999999 999999.999999 999999.999999 999999.999999
*  2024  5 12  0  5  0.00000000
PG01  20000.000000   1000.000000      0.000000    100.000000
PE04  10000.000000  10000.000000  10000.000000    100.000000
EOF
"""


def test_read_sp3_converts_to_metres_and_marks_bad_nodes(tmp_path):
    path = tmp_path / "t.sp3"
    path.write_text(SP3_TEXT)
    orbits = ig.read_sp3(path)
    assert list(orbits.node_times_s) == [0.0, 300.0]
    assert orbits.positions_m["G01"][1].tolist() == [2e7, 1e6, 0.0]
    assert np.isnan(orbits.positions_m["E04"][0]).all()


RINEX_HEADER = [
    ("     3.04           OBSERVATION DATA    M (MIXED)", "RINEX VERSION / TYPE"),
    ("  1000.0000   2000.0000   3000.0000", "APPROX POSITION XYZ"),
    ("G    8 C1C L1C C1W L1W C2W L2W C5X L5X", "SYS / # / OBS TYPES"),
    ("E    4 C1X L1X C5X L5X", "SYS / # / OBS TYPES"),
    ("", "END OF HEADER"),
]


def test_read_rinex3_prefers_ppp_priority_and_flags_dual_frequency(tmp_path):
    path = tmp_path / "t.rnx"

    # Fixed-width 16-char fields: build lines exactly instead of trusting the literal above.
    def field(value):
        return f"{value:14.3f}  " if value is not None else " " * 16

    lines = [text.ljust(60) + label for text, label in RINEX_HEADER]
    lines.append("> 2024 05 12 00 00 30.0000000  0  3")
    lines.append(
        "G01"
        + "".join(
            field(v) for v in (2e7, 1e5, 2e7 + 1, 1e5 + 1, 2e7 + 2, 1e5 + 2, None, None)
        )
    )
    lines.append(
        "G02"
        + "".join(field(v) for v in (2e7, 1e5, None, None, None, None, None, None))
    )
    lines.append("E04" + "".join(field(v) for v in (2.1e7, 1e5, 2.1e7, 1e5)))
    path.write_text("\n".join(lines) + "\n")
    observations = ig.read_rinex3_observations(path)
    table = observations.table.set_index("sat")
    # GPS priority is P, W, C, ...: C1W (index W) beats C1C.
    assert table.loc["G01", "c1_attribute"] == "W"
    assert bool(table.loc["G01", "has_dual"])
    assert table.loc["G02", "c1_attribute"] == "C"
    assert not bool(table.loc["G02", "has_dual"])
    assert bool(table.loc["E04", "has_dual"])
    assert observations.header_xyz_m.tolist() == [1000.0, 2000.0, 3000.0]


def test_read_sinex_uses_estimate_block_not_apriori(tmp_path):
    path = tmp_path / "t.snx"
    path.write_text(
        "+SOLUTION/APRIORI\n"
        "   1 STAX   AAAA  A    1 24:133:43200 m    2  1.0 0.0\n"
        "-SOLUTION/APRIORI\n"
        "+SOLUTION/ESTIMATE\n"
        "   1 STAX   AAAA  A    1 24:133:43200 m    2  1.5e+06 1.0e-03\n"
        "   2 STAY   AAAA  A    1 24:133:43200 m    2 -2.5e+06 1.0e-03\n"
        "   3 STAZ   AAAA  A    1 24:133:43200 m    2  3.5e+06 1.0e-03\n"
        "-SOLUTION/ESTIMATE\n"
    )
    assert ig.read_sinex_station_xyz(path, "aaaa").tolist() == [1.5e6, -2.5e6, 3.5e6]
    with pytest.raises(ValueError):
        ig.read_sinex_station_xyz(path, "BBBB")


def test_light_time_correction_is_below_two_millidegrees_of_elevation():
    # A circular GPS-like orbit: the correction must stay far below the 0.01 deg budget.
    radius_m, period_s = 26_560_000.0, 43_082.0
    node_times = np.arange(0.0, 86_700.0, 300.0)
    angle = 2 * np.pi * node_times / period_s
    # Inclined circular orbit expressed directly in the Earth-fixed frame (rotation ignored:
    # only the *difference* between the two options matters here).
    positions = np.column_stack(
        (
            radius_m * np.cos(angle),
            radius_m * np.sin(angle) * 0.8,
            radius_m * np.sin(angle) * 0.6,
        )
    )
    orbits = ig.Sp3Orbits(node_times, {"G01": positions})
    station = np.array([4.0e6, 3.0e6, 3.9e6])
    longitude, latitude, _ = ig.ecef_to_geodetic_arrays(station)
    times = np.arange(30_000.0, 33_000.0, 30.0)
    with_correction = ig.satellite_positions_at_reception(
        orbits, "G01", times, station, True
    )
    without = ig.satellite_positions_at_reception(orbits, "G01", times, station, False)
    elevation_a, _ = ig.elevation_azimuth_deg(
        station, with_correction, latitude, longitude
    )
    elevation_b, _ = ig.elevation_azimuth_deg(station, without, latitude, longitude)
    assert np.abs(elevation_a - elevation_b).max() < 0.002
