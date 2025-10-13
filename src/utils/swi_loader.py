"""
Space Weather Index (SWI) Data Loading Utilities

This module provides utilities for loading and processing Space Weather Index data
from HDF5 files used in the PNN_STEC project.
"""

import h5py
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def load_swi_data(timestamp: datetime, swi_file_path: str = None) -> Optional[Dict[str, float]]:
    """
    Load Space Weather Index (SWI) data for a given timestamp.

    Args:
        timestamp: datetime object
        swi_file_path: Path to SWI HDF5 file. If None, uses default path.

    Returns:
        dict: Dictionary with SWI feature values, or None if data not available
    """
    if swi_file_path is None:
        swi_file_path = "/scratch2/arrueegg/WP4/PNN_STEC/data/omni_hourly_2010-2025.h5"

    if not Path(swi_file_path).exists():
        logger.warning(f"SWI file not found at {swi_file_path}, using default values")
        return None

    try:
        with h5py.File(swi_file_path, "r") as swi_file:
            year = timestamp.year
            doy = timestamp.timetuple().tm_yday
            hour = timestamp.hour

            # Format DOY with 3 digits as in the data structure
            doy3 = f"{doy:03d}"

            if str(year) not in swi_file:
                logger.warning(f"Year {year} not found in SWI file")
                return None

            if doy3 not in swi_file[str(year)]:
                logger.warning(f"DOY {doy3} not found for year {year} in SWI file")
                return None

            # Load the daily data (24 hours)
            daily_data = swi_file[str(year)][doy3][:]

            if hour >= len(daily_data):
                logger.warning(f"Hour {hour} not available for {year}-{doy3}")
                return None

            # Get data for the specific hour
            hourly_data = daily_data[hour]

            # Skip YEAR, DOY, HR columns (first 3) and map correctly to feature names
            # Based on the actual column structure in the HDF5 file
            swi_feature_mapping = [
                ("Bartels_rotation_number", 3),
                ("Scalar_B,_nT", 4),
                ("Vector_B_Magnitude,nT", 5),
                ("Lat_Angle_of_B_GSE", 6),
                ("Long_Angle_of_B_GSE", 7),
                ("BZ,_nT_GSE", 8),
                ("BZ,_nT_GSM", 9),
                ("SW_Plasma_Speed,_km/s", 10),
                ("Flow_pressure", 11),
                ("E_electric_field", 12),  # Note: file has typo 'E_elecrtric_field'
                ("Alfen_mach_number", 13),
                ("Kp_index", 14),
                ("R_Sunspot_No", 15),
                ("Dst-index,_nT", 16),
                ("AE-index,_nT", 17),
                ("ap_index,_nT", 18),
                ("f107_index", 19),
                ("pc-index", 20),
                ("AL-index,_nT", 21),
                ("AU-index,_nT", 22),
                ("Magnetosonic_Much_num", 23),
                ("Lyman_alpha", 24),
            ]

            swi_features = {
                name: hourly_data[idx] if len(hourly_data) > idx else 0.0
                for name, idx in swi_feature_mapping
            }

            return swi_features

    except Exception as e:
        logger.error(f"Error loading SWI data: {e}")
        return None