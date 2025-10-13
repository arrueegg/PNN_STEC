"""
Coordinate Transformation Utilities for PNN_STEC Project

This module provides coordinate transformation functions for ionospheric modeling:
- IPP (Ionospheric Pierce Point) calculations
- Geographic to Solar Magnetic coordinate conversions using spacepy
- Utility functions for spatial coordinate operations

These functions are extracted from the original inference_map.py for better reusability
across the codebase.
"""

import numpy as np
import logging
from datetime import datetime
from typing import Tuple, Union, List

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
    ipp_height: float = 450.0
) -> Tuple[float, float]:
    """
    Calculate Ionospheric Pierce Point (IPP) coordinates.

    Args:
        station_lat: Station latitude in degrees
        station_lon: Station longitude in degrees
        azimuth: Satellite azimuth angle in degrees (0=North, 90=East)
        elevation: Satellite elevation angle in degrees
        ipp_height: IPP height in km (default: 450 km)

    Returns:
        tuple: (ipp_lat, ipp_lon) in degrees
    """
    # Earth radius in km
    RE = 6371.0

    # Convert to radians
    lat_rad = np.deg2rad(station_lat)
    lon_rad = np.deg2rad(station_lon)
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)

    # Calculate the central angle (psi) to the IPP
    # Using thin shell approximation for ionosphere
    sin_psi = (RE / (RE + ipp_height)) * np.cos(el_rad)
    psi = np.arcsin(sin_psi)

    # Calculate IPP latitude
    ipp_lat_rad = np.arcsin(
        np.sin(lat_rad) * np.cos(psi) + np.cos(lat_rad) * np.sin(psi) * np.cos(az_rad)
    )

    # Calculate IPP longitude
    delta_lon = np.arcsin(np.sin(psi) * np.sin(az_rad) / np.cos(ipp_lat_rad))
    ipp_lon_rad = lon_rad + delta_lon

    # Convert back to degrees
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
    epochs: List[datetime]
):
    """
    Transform coordinates using spacepy.

    Args:
        input_type: Input coordinate system (e.g., 'GEO')
        output_type: Output coordinate system (e.g., 'SM')
        lats: Array of latitudes in degrees
        lons: Array of longitudes in degrees
        epochs: Array of datetime objects

    Returns:
        Transformed coordinates object with .data attribute containing [[alt, lat, lon], ...]
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
    timestamp: datetime
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert geographic coordinates to solar magnetic coordinates using spacepy.

    Args:
        geo_lat: Geographic latitude in degrees (scalar or array)
        geo_lon: Geographic longitude in degrees (scalar or array)
        timestamp: datetime object

    Returns:
        tuple: (sm_lat, sm_lon) in degrees
    """
    if not SPACEPY_AVAILABLE:
        logger.warning(
            "spacepy not available, using geographic coordinates as solar magnetic placeholder"
        )
        if not hasattr(geo_lat, "__len__"):
            return float(geo_lat), float(geo_lon)
        else:
            return geo_lat, geo_lon

    try:
        # Handle scalar inputs
        if not hasattr(geo_lat, "__len__"):
            geo_lat = [geo_lat]
            geo_lon = [geo_lon]
            is_scalar = True
        else:
            is_scalar = False

        epochs = [timestamp] * len(geo_lat)

        # Transform coordinates
        sm_coords = coord_transform("GEO", "SM", geo_lat, geo_lon, epochs)

        if sm_coords is not None:
            # Extract lat/lon from spacepy coords (format: [alt, lat, lon])
            sm_lat = sm_coords.data[:, 1]  # latitude is second column
            sm_lon = sm_coords.data[:, 2]  # longitude is third column

            if is_scalar:
                return float(sm_lat[0]), float(sm_lon[0])
            else:
                return sm_lat, sm_lon
        else:
            # Fallback to geographic coordinates
            logger.warning("Using geographic coordinates as solar magnetic placeholder")
            if is_scalar:
                return float(geo_lat[0]), float(geo_lon[0])
            else:
                return geo_lat, geo_lon

    except Exception as e:
        logger.warning(
            f"Coordinate transformation failed: {e}, using geographic coordinates"
        )
        if not hasattr(geo_lat, "__len__"):
            return float(geo_lat), float(geo_lon)
        else:
            return geo_lat, geo_lon


def create_global_grid(lat_res: float = 1.0, lon_res: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a global grid of latitude and longitude points.

    Args:
        lat_res: Latitude resolution in degrees
        lon_res: Longitude resolution in degrees

    Returns:
        tuple: (lat_grid, lon_grid) as 2D arrays
    """
    # Create 1D arrays
    lats = np.arange(-90, 90 + lat_res, lat_res)
    lons = np.arange(-180, 180 + lon_res, lon_res)

    # Create 2D meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    return lat_grid, lon_grid