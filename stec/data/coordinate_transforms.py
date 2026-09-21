"""Coordinate transformation utilities, ported unchanged from
`src/utils/coordinate_transforms.py`.

IPP (Ionospheric Pierce Point) calculation and geographic-to-solar-magnetic conversion via
spacepy, for the operational scripts that compute model input features from raw
observation logs at inference time (`scripts/infer_from_log.py`,
`stec/data/madrigal_builder.py`) rather than reading them precomputed from the STEC
database.

Faithfully preserves a known limitation, not fixed here: `geographic_to_solar_magnetic`
hardcodes the ionospheric shell height (450 km) for every point passed to it, including
station coordinates near the surface. `stec.data.madrigal_reader._add_sm_coordinates` (the
paper's own rebuilt reader) documents this explicitly and uses a different shell height per
coordinate type instead. This module is used only by standalone scripts that reproduce the
pre-rebuild behaviour on purpose (feeding a station's own coordinates through the same
450 km-shell transform the legacy `infer_from_log.py`/`build_madrigal_h5_sample.py` always
used) - not by anything that also has a Gate-A-verified rebuilt equivalent to disagree with.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    from spacepy import coordinates as coord
    from spacepy.time import Ticktock

    SPACEPY_AVAILABLE = True
except ImportError:
    SPACEPY_AVAILABLE = False
    logger.warning(
        "spacepy not available, will use geographic coordinates as placeholder"
    )


def calculate_ipp_coordinates(
    station_lat: float,
    station_lon: float,
    azimuth: float,
    elevation: float,
    ipp_height: float = 450.0,
) -> Tuple[float, float]:
    """Calculate Ionospheric Pierce Point (IPP) coordinates.

    Args:
        station_lat: Station latitude in degrees.
        station_lon: Station longitude in degrees.
        azimuth: Satellite azimuth angle in degrees (0=North, 90=East).
        elevation: Satellite elevation angle in degrees.
        ipp_height: IPP height in km (default: 450 km).

    Returns:
        (ipp_lat, ipp_lon) in degrees.
    """
    RE = 6371.0  # Earth radius in km

    lat_rad = np.deg2rad(station_lat)
    lon_rad = np.deg2rad(station_lon)
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)

    # Central angle to the IPP via the thin-shell approximation (Klobuchar /
    # Hofmann-Wellenhof): psi = pi/2 - elevation - arcsin(RE/(RE+H) * cos(elevation))
    psi = np.pi / 2 - el_rad - np.arcsin((RE / (RE + ipp_height)) * np.cos(el_rad))

    ipp_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(psi) + np.cos(lat_rad) * np.sin(psi) * np.cos(az_rad)
    )

    delta_lon = np.arcsin(np.sin(psi) * np.sin(az_rad) / np.cos(ipp_lat_rad))
    ipp_lon_rad = lon_rad + delta_lon

    ipp_lat = np.rad2deg(ipp_lat_rad)
    ipp_lon = np.rad2deg(ipp_lon_rad)

    # Normalize longitude to [-180, 180]
    ipp_lon = ((ipp_lon + 180) % 360) - 180

    return ipp_lat, ipp_lon


def coord_transform(
    input_type: str,
    output_type: str,
    lats: Union[float, List[float], np.ndarray],
    lons: Union[float, List[float], np.ndarray],
    epochs: List[datetime],
):
    """Transform coordinates using spacepy.

    Args:
        input_type: Input coordinate system (e.g., 'GEO').
        output_type: Output coordinate system (e.g., 'SM').
        lats: Array of latitudes in degrees.
        lons: Array of longitudes in degrees.
        epochs: Array of datetime objects.

    Returns:
        Transformed coordinates object with a `.data` attribute holding
        `[[alt, lat, lon], ...]`.
    """
    if not SPACEPY_AVAILABLE:
        return None

    try:
        coords = np.array(
            [[1 + 450 / 6371, lat, lon] for lat, lon in zip(lats, lons)],
            dtype=np.float64,
        )
        geo_coords = coord.Coords(coords, input_type, "sph")
        geo_coords.ticks = Ticktock(epochs, "UTC")
        return geo_coords.convert(output_type, "sph")
    except Exception as e:
        logger.warning(f"spacepy coordinate transformation failed: {e}")
        return None


def geographic_to_solar_magnetic(
    geo_lat: Union[float, np.ndarray],
    geo_lon: Union[float, np.ndarray],
    timestamp: datetime,
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert geographic coordinates to solar magnetic coordinates using spacepy.

    Args:
        geo_lat: Geographic latitude in degrees (scalar or array).
        geo_lon: Geographic longitude in degrees (scalar or array).
        timestamp: datetime object.

    Returns:
        (sm_lat, sm_lon) in degrees.
    """
    if not SPACEPY_AVAILABLE:
        logger.warning(
            "spacepy not available, using geographic coordinates as solar magnetic placeholder"
        )
        if not hasattr(geo_lat, "__len__"):
            return float(geo_lat), float(geo_lon)
        return geo_lat, geo_lon

    try:
        if not hasattr(geo_lat, "__len__"):
            geo_lat = [geo_lat]
            geo_lon = [geo_lon]
            is_scalar = True
        else:
            is_scalar = False

        epochs = [timestamp] * len(geo_lat)

        sm_coords = coord_transform("GEO", "SM", geo_lat, geo_lon, epochs)

        if sm_coords is not None:
            sm_lat = sm_coords.data[:, 1]
            sm_lon = sm_coords.data[:, 2]

            if is_scalar:
                return float(sm_lat[0]), float(sm_lon[0])
            return sm_lat, sm_lon

        logger.warning("Using geographic coordinates as solar magnetic placeholder")
        if is_scalar:
            return float(geo_lat[0]), float(geo_lon[0])
        return geo_lat, geo_lon

    except Exception as e:
        logger.warning(
            f"Coordinate transformation failed: {e}, using geographic coordinates"
        )
        if not hasattr(geo_lat, "__len__"):
            return float(geo_lat), float(geo_lon)
        return geo_lat, geo_lon


def geographic_to_magnetic_latitude(
    geo_lat: Union[float, np.ndarray],
    geo_lon: Union[float, np.ndarray],
    timestamp: datetime,
) -> Union[float, np.ndarray]:
    """Magnetic latitude derived from the SM frame (Mao et al. 2025 convention)."""
    if not SPACEPY_AVAILABLE:
        logger.warning(
            "spacepy not available, using geographic latitude as magnetic placeholder"
        )
        return geo_lat

    try:
        sm_lat, _ = geographic_to_solar_magnetic(geo_lat, geo_lon, timestamp)
        return sm_lat
    except Exception as e:
        logger.warning(
            f"Magnetic latitude transform failed: {e}, using geographic latitude as fallback"
        )
        return geo_lat


def geographic_to_sunfixed_longitude(
    geo_lon: Union[float, np.ndarray], timestamp: datetime
) -> Union[float, np.ndarray]:
    """Sun-fixed longitude (0 deg at the subsolar point; Mao et al. 2025 convention)."""
    try:
        j2000 = datetime(2000, 1, 1, 12, 0, 0)

        if isinstance(timestamp, (list, np.ndarray)):
            timestamps = timestamp
            is_array = True
        else:
            timestamps = [timestamp]
            is_array = False

        sf_lons = []
        for ts in timestamps:
            dt = ts - j2000
            T = dt.total_seconds() / (36525 * 86400)  # Julian centuries

            gmst_sec = (
                67310.54841
                + (876600.0 * 3600 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3
            )
            gmst_deg = (gmst_sec / 240.0) % 360.0

            day_of_year = ts.timetuple().tm_yday
            day_angle = 2 * np.pi * (day_of_year - 1) / 365.25
            solar_declination = 23.44 * np.sin(day_angle)  # noqa: F841 - kept for parity

            B = 360 * (day_of_year - 1) / 365
            eot_min = 229.18 * (
                0.000075
                + 0.001868 * np.cos(np.deg2rad(B))
                - 0.032077 * np.sin(np.deg2rad(B))
                - 0.014615 * np.cos(2 * np.deg2rad(B))
                - 0.040849 * np.sin(2 * np.deg2rad(B))
            )
            eot_deg = eot_min / 4.0

            gha = (gmst_deg + eot_deg) % 360.0

            sf_lon = geo_lon - gha
            sf_lon = ((sf_lon + 180) % 360) - 180

            sf_lons.append(sf_lon)

        if is_array:
            if np.isscalar(sf_lons[0]):
                return np.array(sf_lons)
            return np.concatenate(sf_lons)
        return sf_lons[0]

    except Exception as e:
        logger.warning(
            f"Sun-fixed longitude transform failed: {e}, using geographic longitude"
        )
        return geo_lon


def create_global_grid(
    lat_res: float = 1.0, lon_res: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a global grid of latitude and longitude points."""
    lats = np.arange(-90, 90 + lat_res, lat_res)
    lons = np.arange(-180, 180 + lon_res, lon_res)

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    return lat_grid, lon_grid
