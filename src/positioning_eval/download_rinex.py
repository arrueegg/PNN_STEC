#!/usr/bin/env python3
"""
RINEX Downloader for CDDIS Archive

Downloads RINEX observation files from CDDIS for specified stations and dates.
Supports both legacy and new CDDIS archive structures.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def doy_to_date(year, doy):
    """Convert year and DOY to datetime object."""
    return datetime.strptime(f"{year} {doy}", "%Y %j")


def download_rinex_file(station, year, doy, output_dir, logger=None):
    """
    Download a single RINEX observation file from CDDIS.
    Bash script caches directory listing to find correct country codes efficiently.
    
    Args:
        station: 4-char station name (uppercase)
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save RINEX files
        logger: Logger instance
    
    Returns:
        Path to downloaded RINEX file or None if failed
    """
    station_upper = station.upper()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get path to download script
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_rinex.sh"
    
    if not download_script.exists():
        logger.error(f"Download script not found: {download_script}")
        return None
    
    logger.info(f"Downloading RINEX for {station_upper}...")
    
    try:
        # Run bash download script (it caches directory listing internally)
        result = subprocess.run(
            ["bash", str(download_script), station_upper, str(year), str(doy), str(output_path)],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            # Find the downloaded file - use glob to match any country code
            yy = str(year)[-2:]
            
            # Search for RINEX 3 long format with any country code (e.g., JPN, CHE, ZAF, etc.)
            rinex3_pattern = f"{station_upper}00???_R_{year}{doy:03d}0000_01D_30S_MO.rnx"
            crx_pattern = f"{station_upper}00???_R_{year}{doy:03d}0000_01D_30S_MO.crx"
            
            # Try glob patterns first
            import glob
            for pattern in [rinex3_pattern, crx_pattern]:
                matches = list(output_path.glob(pattern))
                if matches:
                    logger.info(f"✓ Downloaded: {matches[0].name}")
                    return matches[0]
            
            # Fallback to specific filenames (RINEX 2 format)
            possible_files = [
                output_path / f"{station.lower()}{doy:03d}0.{yy}d",
                output_path / f"{station.lower()}{doy:03d}0.{yy}o",
            ]
            
            for rinex_file in possible_files:
                if rinex_file.exists():
                    logger.info(f"✓ Downloaded: {rinex_file.name}")
                    return rinex_file
            
            logger.warning(f"Download script succeeded but no RINEX file found for {station_upper}")
            return None
        else:
            # wget outputs errors to stdout, not stderr
            error_msg = result.stderr if result.stderr else result.stdout
            logger.warning(f"Download failed for {station_upper}: {error_msg[:500]}")
            return None
    
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout downloading RINEX for {station_upper}")
        return None
    except Exception as e:
        logger.error(f"Error downloading RINEX for {station_upper}: {e}")
        return None


def download_rinex_batch(stations, year, doy, output_dir, logger=None):
    """
    Download RINEX files for multiple stations.
    
    Args:
        stations: List of station names
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save files
        logger: Optional logger instance
    
    Returns:
        Dictionary mapping station names to downloaded file paths
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Downloading RINEX files for {len(stations)} stations ({year}/{doy:03d})")
    
    results = {}
    for station in stations:
        # Bash script now caches directory listing internally
        rinex_path = download_rinex_file(station, year, doy, output_dir, logger)
        if rinex_path:
            results[station] = rinex_path
    
    logger.info(f"Successfully downloaded {len(results)}/{len(stations)} RINEX files")
    return results
