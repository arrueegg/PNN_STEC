"""
IONEX File Generation Utilities for PNN_STEC Project

This module provides functionality to export STEC maps in the standard 
IONEX (IONosphere map EXchange) format used by the International GNSS Service (IGS).

IONEX Format Specifications:
- ASCII text format with standardized headers
- Global TEC maps on regular grids
- Standard resolution: 2.5° lat × 5° lon (71×73 grid)
- Units in TECU (Total Electron Content Units)
- Single-layer model assumption at 450 km height

References:
- IONEX format specification: ftp://igs.org/pub/data/format/ionex1.pdf
- IGS conventions and standards
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class IONEXWriter:
    """
    Class to generate IONEX files from STEC prediction maps.
    
    Follows the standard IONEX format specification used by IGS and other
    ionospheric data providers.
    """
    
    def __init__(
        self,
        center_code: str = "PNN",
        program: str = "PNN_STEC",
        version: str = "1.0",
        height_km: float = 450.0
    ):
        """
        Initialize IONEX writer.
        
        Args:
            center_code: Analysis center code (e.g., 'PNN', 'COD', 'JPL')
            program: Program name for header
            version: Version string
            height_km: Single-layer height assumption in km
        """
        self.center_code = center_code
        self.program = program
        self.version = version
        self.height_km = height_km
        
    def generate_ionex_header(
        self,
        start_time: datetime,
        end_time: datetime,
        interval_hours: float,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        description: str = "STEC maps from PNN_STEC model",
        elevation: float = 90.0,
        azimuth: float = 180.0
    ) -> List[str]:
        """
        Generate IONEX format header lines.
        
        Args:
            start_time: First map epoch
            end_time: Last map epoch  
            interval_hours: Time interval between maps in hours
            lat_grid: Latitude grid array
            lon_grid: Longitude grid array
            description: Description for header
            
        Returns:
            List of header lines
        """
        header_lines = []
        
        # Version and file type
        header_lines.append(f"     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE")
        
        # Program and creation info
        creation_time = datetime.utcnow()
        header_lines.append(f"{self.program:<20}{self.center_code:<20}{creation_time.strftime('%d-%b-%y %H:%M')}     PGM / RUN BY / DATE")
        
        # Description
        header_lines.append(f"{description:<60}DESCRIPTION")
        
        # Epoch info
        header_lines.append(f"{start_time.year:6d}{start_time.month:6d}{start_time.day:6d}{start_time.hour:6d}{start_time.minute:6d}{start_time.second:6d}                     EPOCH OF FIRST MAP")
        header_lines.append(f"{end_time.year:6d}{end_time.month:6d}{end_time.day:6d}{end_time.hour:6d}{end_time.minute:6d}{end_time.second:6d}                     EPOCH OF LAST MAP")
        
        # Interval
        interval_seconds = int(interval_hours * 3600)
        header_lines.append(f"{interval_seconds:6d}                                                      INTERVAL")
        
        # Number of maps
        n_maps = int((end_time - start_time).total_seconds() / interval_seconds) + 1
        header_lines.append(f"{n_maps:6d}                                                      # OF MAPS IN FILE")
        
        # Mapping function (single layer)
        header_lines.append(f"  NONE                                                       MAPPING FUNCTION")
        
        # Elevation cutoff (not applicable for model predictions)
        header_lines.append(f"     0.0                                                     ELEVATION CUTOFF")
        
        # Observable used
        header_lines.append(f"  TEC                                                        OBSERVABLES USED")
        
        # Number of stations (N/A for model)
        header_lines.append(f"     0                                                       # OF STATIONS")
        
        # Number of satellites (N/A for model) 
        header_lines.append(f"     0                                                       # OF SATELLITES")
        
        # Base radius
        base_radius_km = 6371.0  # Earth radius
        header_lines.append(f"{base_radius_km:8.1f}                                                  BASE RADIUS")
        
        # Map dimension (2 for lat/lon)
        header_lines.append(f"     2                                                       MAP DIMENSION")
        
        # Grid dimensions
        height_single = self.height_km
        lat_min, lat_max = float(np.min(lat_grid)), float(np.max(lat_grid))
        lon_min, lon_max = float(np.min(lon_grid)), float(np.max(lon_grid))
        lat_spacing = float(np.mean(np.diff(lat_grid[:, 0])))
        lon_spacing = float(np.mean(np.diff(lon_grid[0, :])))
        n_lat, n_lon = lat_grid.shape[0], lon_grid.shape[1]
        
        header_lines.append(f"{height_single:8.1f}{height_single:8.1f}{lat_spacing:8.1f}                                HGT1 / HGT2 / DHGT")
        header_lines.append(f"{lat_min:8.1f}{lat_max:8.1f}{lat_spacing:8.1f}                                LAT1 / LAT2 / DLAT")  
        header_lines.append(f"{lon_min:8.1f}{lon_max:8.1f}{lon_spacing:8.1f}                                LON1 / LON2 / DLON")
        
        # Exponent (for scaling TEC values, -1 means values are in 0.1 TECU)
        # We'll use 0 to keep values in TECU
        header_lines.append(f"     0                                                       EXPONENT")
        
        # Comment lines
        header_lines.append(f"Generated by PNN_STEC Bayesian Neural Network model         COMMENT")
        header_lines.append(f"Elevation: {elevation:.1f}°, Azimuth: {azimuth:.1f}°                            COMMENT")
        
        # End of header
        header_lines.append(f"                                                            END OF HEADER")
        
        return header_lines
    
    def format_tec_map(
        self,
        epoch: datetime,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        tec_map: np.ndarray,
        map_type: str = "TEC"
    ) -> List[str]:
        """
        Format a TEC map in IONEX format.
        
        Args:
            epoch: Time of the map
            lat_grid: Latitude grid
            lon_grid: Longitude grid  
            tec_map: TEC values array
            map_type: Type of map ("TEC" or "RMS")
            
        Returns:
            List of formatted map lines
        """
        lines = []
        
        # Start of TEC map
        lines.append(f"     1                                                       START OF {map_type} MAP")
        
        # Epoch line
        lines.append(f"{epoch.year:6d}{epoch.month:6d}{epoch.day:6d}{epoch.hour:6d}{epoch.minute:6d}{epoch.second:6d}                     EPOCH OF CURRENT MAP")
        
        # Map data - IONEX format writes data in latitude bands
        lat_values = lat_grid[:, 0]  # Get unique latitude values
        lon_values = lon_grid[0, :]  # Get unique longitude values
        
        for i, lat in enumerate(lat_values):
            # LAT/LON1/LON2/DLON/H header for each latitude band
            lon_min, lon_max = float(lon_values[0]), float(lon_values[-1])
            lon_spacing = float(np.mean(np.diff(lon_values)))
            lines.append(f"{lat:6.1f}{lon_min:6.1f}{lon_max:6.1f}{lon_spacing:6.1f}{self.height_km:7.1f}                       LAT/LON1/LON2/DLON/H")
            
            # TEC values for this latitude (16 values per line, format I5)
            tec_values = tec_map[i, :]
            
            # Handle NaN values and convert to integers (IONEX typically uses 0.1 TECU units, but we'll use TECU)
            # Replace NaN values with -1 (common IONEX convention for missing data)
            tec_clean = np.where(np.isnan(tec_values), -1, tec_values)
            # Clip extreme values to reasonable range
            tec_clean = np.clip(tec_clean, -999, 9999)
            tec_ints = np.round(tec_clean).astype(int)
            
            # Write in chunks of 16 values per line
            for j in range(0, len(tec_ints), 16):
                chunk = tec_ints[j:j+16]
                line = ""
                for val in chunk:
                    line += f"{val:5d}"
                # Pad line to 80 characters if needed
                line = line.ljust(80)
                lines.append(line)
        
        # End of TEC map
        lines.append(f"     1                                                       END OF {map_type} MAP")
        
        return lines
    
    def write_ionex_file(
        self,
        output_path: str,
        epochs: List[datetime],
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        tec_maps: List[np.ndarray],
        rms_maps: Optional[List[np.ndarray]] = None,
        interval_hours: float = 1.0,
        description: str = "STEC maps from PNN_STEC model",
        elevation: float = 90.0,
        azimuth: float = 180.0
    ) -> None:
        """
        Write complete IONEX file with TEC maps and optional RMS maps.
        
        Args:
            output_path: Output file path
            epochs: List of map epochs
            lat_grid: Latitude grid
            lon_grid: Longitude grid
            tec_maps: List of TEC map arrays
            rms_maps: Optional list of RMS/uncertainty map arrays
            interval_hours: Time interval between maps
            description: Description for header
        """
        if len(epochs) != len(tec_maps):
            raise ValueError("Number of epochs must match number of TEC maps")
        
        if rms_maps and len(rms_maps) != len(tec_maps):
            raise ValueError("Number of RMS maps must match number of TEC maps")
        
        # Generate header
        start_time = min(epochs)
        end_time = max(epochs) 
        header_lines = self.generate_ionex_header(
            start_time, end_time, interval_hours, lat_grid, lon_grid, description, elevation, azimuth
        )
        
        # Write file
        with open(output_path, 'w') as f:
            # Write header
            for line in header_lines:
                f.write(line + '\n')
            
            # Write TEC maps
            for i, (epoch, tec_map) in enumerate(zip(epochs, tec_maps)):
                tec_lines = self.format_tec_map(epoch, lat_grid, lon_grid, tec_map, "TEC")
                for line in tec_lines:
                    f.write(line + '\n')
                
                # Write RMS map if available
                if rms_maps:
                    rms_lines = self.format_tec_map(epoch, lat_grid, lon_grid, rms_maps[i], "RMS")
                    for line in rms_lines:
                        f.write(line + '\n')
            
            # End of file
            f.write("                                                            END OF FILE\n")


def generate_ionex_filename(date: datetime, center_code: str = "PNN") -> str:
    """
    Generate standard IONEX filename following IGS conventions.
    
    Args:
        date: Date for the file (datetime object)
        center_code: Analysis center code
        
    Returns:
        Standard IONEX filename (e.g., "PNNg001.24I")
    """
    # Ensure date is a datetime object
    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d")
    
    # Day of year
    doy = date.timetuple().tm_yday
    
    # Year (last 2 digits)  
    year_2digit = date.year % 100
    
    # IONEX filename format: CCCgDDD.YYI
    # CCC = center code (3 chars)
    # g = GPS
    # DDD = day of year (3 digits)
    # YY = year (2 digits)
    # I = IONEX format indicator
    
    filename = f"{center_code.upper()[:3]:3s}g{doy:03d}.{year_2digit:02d}I"
    
    return filename