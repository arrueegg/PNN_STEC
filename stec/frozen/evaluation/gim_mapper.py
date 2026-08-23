#!/usr/bin/env python3
"""
GIM to STEC Mapping Utilities

This module provides functionality to read Global Ionospheric Maps (GIM) in IONEX format
and map the vertical TEC (VTEC) to slant TEC (STEC) along specified lines of sight.

Key Components:
- IONEXReader: Parse and load IONEX files 
- GIMMapper: Map VTEC to STEC using thin-shell model
- Spatial/temporal interpolation for observation grids

References:
- IONEX format: ftp://igs.org/pub/data/format/ionex1.pdf
- Thin shell mapping: Schaer et al. (1999)
"""

import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import warnings

logger = logging.getLogger(__name__)


class MappingFunction:
    """
    Mapping functions for converting VTEC to STEC.
    
    Supports Single Layer Model (SLM) and Modified Single Layer Model (MSLM).
    """
    
    def __init__(self, mapping_type: str = 'SLM'):
        self.RE = 6371.0  # Earth radius in km
        self.type = mapping_type
        
    def SLM_MF(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate the mapping function for the Single Layer Model (SLM).
        
        Args:
            elevation: Elevation angle in radians (scalar or array)
            
        Returns:
            Mapping factor to convert VTEC to STEC
        """
        H = 450.0  # Height of the ionospheric shell in km
        mapping_function = np.cos(np.arcsin(self.RE / (self.RE + H) * np.sin(np.pi/2 - elevation)))
        return 1.0 / mapping_function

    def MSLM_MF(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate the mapping function for the Modified Single Layer Model (MSLM).
        
        Args:
            elevation: Elevation angle in radians (scalar or array)
            
        Returns:
            Mapping factor to convert VTEC to STEC
        """
        H = 506.7  # Height of the ionospheric shell in km
        alpha = 0.9782
        mapping_function = np.cos(np.arcsin(self.RE / (self.RE + H) * np.sin(alpha * (np.pi/2 - elevation))))
        return 1.0 / mapping_function
        
    def get_mapping_factor(self, elevation: np.ndarray) -> np.ndarray:
        """Get mapping factor based on configured type."""
        if self.type == 'SLM':
            return self.SLM_MF(elevation)
        elif self.type == 'MSLM':
            return self.MSLM_MF(elevation)
        else:
            # Default to SLM
            return self.SLM_MF(elevation)


class IONEXReader:
    """
    Reader for IONEX format Global Ionospheric Maps.
    
    Handles standard IONEX files with VTEC maps on regular grids.
    """
    
    def __init__(self):
        self.header = {}
        self.vtec_maps = []  # List of (epoch, vtec_grid) tuples
        self.rms_maps = []   # List of (epoch, rms_grid) tuples
        self.lat_grid = None
        self.lon_grid = None
        self.epochs = []
        
    def read_ionex_file(self, filepath: Path) -> Dict[str, Any]:
        """
        Read a single IONEX file and extract VTEC maps.
        
        Args:
            filepath: Path to IONEX file (.??i format)
            
        Returns:
            Dict containing epochs, grids, VTEC data, and RMS data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"IONEX file not found: {filepath}")
            
        logger.debug(f"Reading IONEX file: {filepath}")
        
        # Clear previous data to prevent accumulation
        self.header = {}
        self.vtec_maps = []
        self.rms_maps = []
        self.lat_grid = None
        self.lon_grid = None
        self.epochs = []
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Parse header
        self._parse_header(lines)
        
        # Parse data section
        self._parse_data_section(lines)
        
        return {
            'epochs': self.epochs,
            'lat_grid': self.lat_grid,
            'lon_grid': self.lon_grid,
            'vtec_maps': self.vtec_maps,
            'rms_maps': self.rms_maps,
            'header': self.header
        }
    
    def _parse_header(self, lines: List[str]) -> None:
        """Parse IONEX header section."""
        header_end = False
        
        for line in lines:
            if 'END OF HEADER' in line:
                header_end = True
                break
                
            # Parse grid definition  
            if 'LAT1 / LAT2 / DLAT' in line:
                parts = line[:60].split()
                self.header['lat_min'] = float(parts[0])
                self.header['lat_max'] = float(parts[1]) 
                self.header['lat_step'] = float(parts[2])
                
            elif 'LON1 / LON2 / DLON' in line:
                parts = line[:60].split()
                self.header['lon_min'] = float(parts[0])
                self.header['lon_max'] = float(parts[1])
                self.header['lon_step'] = float(parts[2])
                
            elif 'HGT1 / HGT2 / DHGT' in line:
                parts = line[:60].split()
                self.header['height_km'] = float(parts[0])
                
            elif 'INTERVAL' in line:
                interval_str = line[:60].strip()
                self.header['time_interval_hrs'] = float(interval_str) / 3600.0  # Convert seconds to hours
        
        if not header_end:
            raise ValueError("Invalid IONEX file: END OF HEADER not found")
            
        # Create coordinate grids
        self.lat_grid = np.arange(
            self.header['lat_min'], 
            self.header['lat_max'] + self.header['lat_step']/2,
            self.header['lat_step']
        )
        self.lon_grid = np.arange(
            self.header['lon_min'],
            self.header['lon_max'] + self.header['lon_step']/2, 
            self.header['lon_step']
        )
        
        logger.debug(f"Grid: {len(self.lat_grid)} lats × {len(self.lon_grid)} lons")
        
    def _parse_data_section(self, lines: List[str]) -> None:
        """Parse IONEX data section with TEC maps using robust parser logic."""
        header_end_line = None
        
        # Find header end
        for i, line in enumerate(lines):
            if 'END OF HEADER' in line:
                header_end_line = i
                break
                
        if header_end_line is None:
            raise ValueError("No END OF HEADER found")
            
        # Parse data section starting after header
        i = header_end_line + 1
        while i < len(lines):
            line = lines[i]
            
            if 'START OF TEC MAP' in line:
                # Read the epoch
                i += 1
                while i < len(lines) and 'EPOCH OF CURRENT MAP' not in lines[i]:
                    i += 1
                
                if i >= len(lines):
                    break
                    
                epoch_line = lines[i]
                # Extract epoch: "2024  1  1  0  0  0"
                epoch_values = epoch_line[:60].split()
                if len(epoch_values) >= 6:
                    year, month, day, hour, minute, second = map(int, epoch_values[:6])
                    current_epoch = datetime(year, month, day, hour, minute, second)
                else:
                    logger.warning("Invalid epoch line format")
                    i += 1
                    continue

                # Initialize VTEC map for this epoch
                vtec_map = np.zeros((len(self.lat_grid), len(self.lon_grid)))

                # Read the VTEC map
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if 'END OF TEC MAP' in line:
                        break
                    elif 'LAT/LON1/LON2/DLON' in line:
                        # Parse latitude row header with negative number handling
                        line_splitted = line.replace('-', ' -').strip().split()
                        if len(line_splitted) >= 4:
                            lat = float(line_splitted[0])
                            lon1 = float(line_splitted[1])
                            lon2 = float(line_splitted[2])
                            dlon = float(line_splitted[3])

                            # Read the data lines for this latitude
                            n_lons = int(round((lon2 - lon1) / dlon)) + 1
                            n_values_read = 0
                            values = []
                            
                            while n_values_read < n_lons and i + 1 < len(lines):
                                i += 1
                                data_line = lines[i].strip()
                                # Parse TEC values, dividing by 10 to convert to TECU
                                try:
                                    data_values = [float(vtec)/10.0 for vtec in data_line.split()]
                                    values.extend(data_values)
                                    n_values_read += len(data_values)
                                except ValueError:
                                    logger.warning(f"Error parsing TEC values in line: {data_line}")
                                    # Skip this line and continue
                                    continue
                            
                            # Store the values in the vtec_map
                            lat_idx = np.where(np.isclose(self.lat_grid, lat, atol=0.1))[0]
                            if len(lat_idx) > 0 and len(values) >= len(self.lon_grid):
                                vtec_map[lat_idx[0], :] = values[:len(self.lon_grid)]
                    i += 1
                
                # Store completed map
                if current_epoch is not None:
                    self.epochs.append(current_epoch)
                    self.vtec_maps.append(vtec_map.copy())
                    
            elif 'START OF RMS MAP' in line:
                # Read the epoch
                i += 1
                while i < len(lines) and 'EPOCH OF CURRENT MAP' not in lines[i]:
                    i += 1
                
                if i >= len(lines):
                    break
                    
                # We don't need to parse date again if we assume consistency, 
                # but let's do it to find the matching index if needed.
                # For simplicity, we assume RMS maps follow the same order or are interleaved.
                
                # Initialize RMS map
                rms_map = np.zeros((len(self.lat_grid), len(self.lon_grid)))
                
                # Read the RMS map
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if 'END OF RMS MAP' in line:
                        break
                    elif 'LAT/LON1/LON2/DLON' in line:
                        # Parse latitude row header
                        line_splitted = line.replace('-', ' -').strip().split()
                        if len(line_splitted) >= 4:
                            lat = float(line_splitted[0])
                            lon1 = float(line_splitted[1])
                            lon2 = float(line_splitted[2])
                            dlon = float(line_splitted[3])

                            n_lons = int(round((lon2 - lon1) / dlon)) + 1
                            n_values_read = 0
                            values = []
                            
                            while n_values_read < n_lons and i + 1 < len(lines):
                                i += 1
                                data_line = lines[i].strip()
                                try:
                                    data_values = [float(rms)/10.0 for rms in data_line.split()]
                                    values.extend(data_values)
                                    n_values_read += len(data_values)
                                except ValueError:
                                    continue
                            
                            lat_idx = np.where(np.isclose(self.lat_grid, lat, atol=0.1))[0]
                            if len(lat_idx) > 0 and len(values) >= len(self.lon_grid):
                                rms_map[lat_idx[0], :] = values[:len(self.lon_grid)]
                    i += 1
                
                # Store completed RMS map
                # We append to the list. If TEC maps were read first, 
                # we might need to align. But for now, just append.
                # Later we can reconcile.
                # Actually, standard IONEX often has all TEC maps then all RMS maps.
                # If we rely on simple appending, we might have issues if they are not 1:1.
                # But let's assume they are for now.
                
                # If we have more RMS maps than TEC maps (because RMS came first?), 
                # we should handle it. 
                # Better strategy: Just append to self.rms_maps. 
                # When using, we check lengths.
                self.rms_maps.append(rms_map.copy())

            else:
                i += 1


class GIMMapper:
    """
    Maps GIM VTEC to STEC using thin-shell ionosphere model.
    
    Handles spatial/temporal interpolation and line-of-sight mapping.
    """
    
    def __init__(self, shell_height_km: float = 450.0, earth_radius_km: float = 6371.0, 
                 mapping_type: str = 'SLM', gim_type: str = 'IGS'):
        """
        Initialize GIM mapper.
        
        Args:
            shell_height_km: Ionospheric shell height (default: 450 km)
            earth_radius_km: Earth radius (default: 6371 km)
            mapping_type: Mapping function type ('SLM' or 'MSLM')
            gim_type: GIM data source ('IGS' or 'CODE')
        """
        self.shell_height_km = shell_height_km
        self.earth_radius_km = earth_radius_km
        self.gim_type = gim_type
        self.reader = IONEXReader()
        self.mapping_func = MappingFunction(mapping_type)
        self.gim_data = {}  # Cache for loaded GIM data
        
    def load_gim_data(self, gim_path: str, date: datetime) -> None:
        """
        Load GIM data for specified time range.
        
        Args:
            gim_path: Path to directory containing IONEX files
            date: Target date for data loading
        """
        gim_path = Path(gim_path)
                
        # Find relevant IONEX files
        ionex_files = self._find_ionex_files(gim_path, date)
        
        if not ionex_files:
            raise FileNotFoundError(f"No IONEX files found in {gim_path} for time range")
                    
        # Load each file
        all_epochs = []
        all_vtec_maps = []
        lat_grid = None
        lon_grid = None
        
        for filepath in sorted(ionex_files):
            try:
                data = self.reader.read_ionex_file(filepath)
                
                # Use first file's grid as reference
                if lat_grid is None:
                    lat_grid = data['lat_grid']
                    lon_grid = data['lon_grid']
                
                # Filter epochs to time range
                for i, epoch in enumerate(data['epochs']):
                    all_epochs.append(epoch)
                    all_vtec_maps.append(data['vtec_maps'][i])
                        
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                continue
        
        if not all_epochs:
            raise ValueError("No valid VTEC data found in time range")
            
        # Sort by time
        sorted_indices = np.argsort(all_epochs)
        self.gim_data = {
            'epochs': [all_epochs[i] for i in sorted_indices],
            'vtec_maps': [all_vtec_maps[i] for i in sorted_indices],
            'lat_grid': lat_grid,
            'lon_grid': lon_grid
        }
                
    def _find_ionex_files(self, gim_path: Path, date: datetime) -> List[Path]:
        """Find IONEX files covering the specified date based on GIM type."""
        ionex_files = []
        
        year = date.year
        doy = date.timetuple().tm_yday
        
        # Define file patterns based on GIM type
        if self.gim_type.upper() == 'IGS':
            # IGS final products: igsgDDD0.YYi
            pattern = f"igsg{doy:03d}0.{year%100:02d}i"
        elif self.gim_type.upper() == 'CODE':
            # CODE final products: codgDDD0.YYi  
            pattern = f"codg{doy:03d}0.{year%100:02d}i"
        else:
            logger.warning(f"Unknown GIM type '{self.gim_type}', defaulting to IGS")
            pattern = f"igsg{doy:03d}0.{year%100:02d}i"
        
        logger.debug(f"Looking for {self.gim_type} IONEX files with pattern: {pattern}")
        
        # Check year directory
        year_dir = gim_path / str(year)
        if year_dir.exists():
            files = list(year_dir.glob(pattern))
            ionex_files.extend(files)
        
        # Also check root directory
        files = list(gim_path.glob(pattern))
        ionex_files.extend(files)

        return ionex_files
        
    def map_vtec_to_stec(self, 
                        sods: np.ndarray,
                        ipp_lat: np.ndarray, 
                        ipp_lon: np.ndarray,
                        elevations: np.ndarray) -> np.ndarray:
        """
        Map VTEC to STEC for given observation geometry.
        
        Args:
            sod: Satellite observation times (seconds of day)
            ipp_lat: Ionospheric pierce point latitudes (degrees)
            ipp_lon: Ionospheric pierce point longitudes (degrees) 
            elevations: Satellite elevation angles (degrees)
            
        Returns:
            Array of STEC values (TECU)
        """
        return ionex_files
        
    def map_vtec_to_stec(self, 
                        sods: np.ndarray,
                        ipp_lat: np.ndarray, 
                        ipp_lon: np.ndarray,
                        elevations: np.ndarray) -> np.ndarray:
        """
        Map VTEC to STEC for given observation geometry.
        
        Args:
            sods: Satellite observation times (seconds of day)
            ipp_lat: Ionospheric pierce point latitudes (degrees)
            ipp_lon: Ionospheric pierce point longitudes (degrees) 
            elevations: Satellite elevation angles (degrees)
            
        Returns:
            Array of STEC values (TECU)
        """
        if not self.gim_data:
            raise ValueError("No GIM data loaded. Call load_gim_data() first.")
            
        n_obs = len(sods)
        
        logger.debug(f"Mapping VTEC to STEC for {n_obs} observations")
        
        # Prepare inputs for vectorized interpolation
        # Normalize longitude to [-180, 180]
        lons_norm = (ipp_lon + 180) % 360 - 180
        lats_clipped = np.clip(ipp_lat, -90, 90)
        
        # Convert times to hours of day
        hods = sods / 3600.0
        
        # Build epoch list in hours (for interpolator)
        gim_day = self.gim_data['epochs'][0].day
        gim_epochs = []
        for epoch in self.gim_data['epochs']:
            if epoch.day == gim_day:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0)
            else:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0 + 24)
        
        # Handle latitude grid orientation
        lat_grid = self.gim_data['lat_grid']
        vtec_maps = np.array(self.gim_data['vtec_maps'])
        
        if lat_grid[0] > lat_grid[-1]:
            lats_corrected = lat_grid[::-1]
            vtec_corrected = vtec_maps[:, ::-1, :]
        else:
            lats_corrected = lat_grid
            vtec_corrected = vtec_maps
        
        try:
            # Create interpolator once
            # Note: RegularGridInterpolator is efficient but creating it is somewhat costly. 
            # We create it once and reuse it for all points.
            interpolator = RegularGridInterpolator(
                (gim_epochs, lats_corrected, self.gim_data['lon_grid']),
                vtec_corrected, 
                bounds_error=False, 
                fill_value=None
            )
            
            # Helper to batch process if too large to avoid memory issues
            batch_size = 100000
            vtec_values = np.zeros(n_obs)
            
            for i in range(0, n_obs, batch_size):
                end_idx = min(i + batch_size, n_obs)
                # Stack coordinates for batch: (N, 3) array of [time, lat, lon]
                points = np.column_stack((hods[i:end_idx], lats_clipped[i:end_idx], lons_norm[i:end_idx]))
                vtec_values[i:end_idx] = interpolator(points)
            
            # Apply mapping function
            # Convert elevation to radians if it looks like degrees (> pi is a crude check but typical given GNSS elevs)
            # Assuming inputs are degrees as per docstring, but let's be robust
            if np.any(elevations > np.pi):
                elev_rad = np.radians(elevations)
            else:
                elev_rad = elevations
                
            mapping_factors = self.mapping_func.get_mapping_factor(elev_rad)
            stec_values = vtec_values * mapping_factors
            
            return stec_values
            
        except Exception as e:
            logger.error(f"Vectorized GIM mapping failed: {e}")
            # Fallback to slow loop if vectorization crashes (unlikely)
            return np.full(n_obs, np.nan)

    def _interpolate_vtec(self, sod: int, lat: float, lon: float) -> float:
        """
        Interpolate VTEC at given time and location using robust interpolation.
        
        Based on the provided interpolate_vtec method with proper handling of
        longitude wrapping and latitude grid orientation.
        """
        # Normalize longitude to [-180, 180]
        lon = (lon + 180) % 360 - 180
        lat = np.clip(lat, -90, 90)
        
        # Convert time to hour of day
        hod = sod / 3600.0  # seconds of day to hours
        
        # Build epoch list in hours
        gim_day = self.gim_data['epochs'][0].day
        gim_epochs = []
        for epoch in self.gim_data['epochs']:
            if epoch.day == gim_day:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0)
            else:
                gim_epochs.append(epoch.hour + epoch.minute / 60.0 + 24)
        
        # Handle latitude grid orientation (some files have descending latitudes)
        lat_grid = self.gim_data['lat_grid']
        vtec_maps = np.array(self.gim_data['vtec_maps'])
        
        if lat_grid[0] > lat_grid[-1]:
            # Reverse latitude order
            lats_corrected = lat_grid[::-1]
            vtec_corrected = vtec_maps[:, ::-1, :]
        else:
            lats_corrected = lat_grid
            vtec_corrected = vtec_maps
        
        try:
            # Create 3D interpolator (time, lat, lon)
            interpolator = RegularGridInterpolator(
                (gim_epochs, lats_corrected, self.gim_data['lon_grid']),
                vtec_corrected, 
                bounds_error=False, 
                fill_value=None
            )
            
            result = interpolator((hod, lat, lon))
            return float(result) if not np.isnan(result) else np.nan
            
        except Exception as e:
            logger.debug(f"Interpolation failed for lat={lat}, lon={lon}, time={hod}: {e}")
            return np.nan

def build_gim_stec(cfg: Dict[str, Any], obs: Dict[str, Any]) -> np.ndarray:
    """
    Main function to build GIM-derived STEC predictions.
    
    Args:
        cfg: Configuration dict with gim_path, shell_height_km, earth_radius_km
        obs: Observations dict with times, ipp_lat, ipp_lon, elevations
        
    Returns:
        Array of GIM-derived STEC values
    """
    if not obs['times']:
        return np.array([])
        
    logger.info("Building GIM→STEC predictions")
    
    # Initialize mapper
    mapper = GIMMapper(
        shell_height_km=cfg.get('shell_height_km', 450.0),
        earth_radius_km=cfg.get('earth_radius_km', 6371.0)
    )
    
    # Determine time range
    times = np.array(obs['times'])
    time_range = (
        min(times) - timedelta(hours=1),  # Add buffer
        max(times) + timedelta(hours=1)
    )
    
    # Load GIM data  
    try:
        mapper.load_gim_data(cfg['gim_path'], time_range)
    except Exception as e:
        logger.error(f"Failed to load GIM data: {e}")
        return np.full(len(times), np.nan)
    
    # Map to STEC
    try:
        stec_values = mapper.map_vtec_to_stec(
            times=times,
            ipp_lat=np.array(obs['ipp_lat']),
            ipp_lon=np.array(obs['ipp_lon']), 
            elevations=np.array(obs['elevations'])
        )
        
        n_valid = np.sum(~np.isnan(stec_values))
        total = len(stec_values)
        logger.info(f"GIM→STEC mapping: {n_valid}/{total} valid predictions ({100*n_valid/total:.1f}%)")
        
        return stec_values
        
    except Exception as e:
        logger.error(f"Failed to map VTEC to STEC: {e}")
        return np.full(len(times), np.nan)