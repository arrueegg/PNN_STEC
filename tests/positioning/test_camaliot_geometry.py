"""Station coordinates parsed from a CamaliotGnss .ION file.

Regression for southern stations: the +SITE/ID block prints latitude unsigned, which
gave every southern station a positive lat_sta and therefore wrong solar-magnetic
station coordinates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "positioning" / "geometry")
)

from camaliot_geometry import ecef_to_geodetic, parse_site  # noqa: E402

# Shaped like a real CamaliotGnss file for CHPG (Cachoeira Paulista, Brazil): +SITE/ID
# prints the unsigned latitude 22.682337 while the true one is about -22.68. The ECEF
# position is approximate, which is enough to pin the hemisphere.
CHPG_ION_EXCERPT = """\
+SITE/ID
*STATION__ PT __DOMES__ T _STATION_DESCRIPTION__ _LONGITUDE _LATITUDE_ _HGT_ELI_ _HGT_MSL_
 CHPG00BRA  A 41612M002 P -                      315.014542  22.682337   617.0    600.0
-SITE/ID
+SITE/COORDINATES
*STATION__ PT SOLN T __DATA_START__ __DATA_END____ __STA_X_____ __STA_Y_____ __STA_Z_____ SYSTEM REMRK
 CHPG00BRA  A 0001 P 2024:133:00000 2024:133:86370  4164616.000 -4161840.000 -2445210.000  IGS14   ETH
-SITE/COORDINATES
"""


def test_southern_station_latitude_is_negative(tmp_path: Path) -> None:
    ion_file = tmp_path / "CHPG.ION"
    ion_file.write_text(CHPG_ION_EXCERPT)

    site = parse_site(ion_file)

    assert site["lat_sta"] == pytest.approx(-22.68, abs=0.05)
    assert site["lon_sta"] == pytest.approx(-44.99, abs=0.05)


def test_ecef_to_geodetic_round_trip_for_northern_and_southern_points() -> None:
    # Point on the equator at lon 90E, and one at the south pole.
    assert ecef_to_geodetic(0.0, 6378137.0, 0.0) == pytest.approx((90.0, 0.0))
    _, south_pole_latitude = ecef_to_geodetic(0.0, 0.0, -6356752.314)
    assert south_pole_latitude == pytest.approx(-90.0)
