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
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def download_rinex_file(station, year, doy, output_dir, logger=None, cache_dir=None):
    """
    Download a single RINEX observation file from CDDIS.
    Bash script caches directory listing to find correct country codes efficiently.
    
    Args:
        station: 4-char station name (uppercase)
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save RINEX files
        logger: Logger instance
        cache_dir: Optional path to shared temp directory for caching listings
    
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
        # Prepare environment (inject cache dir if provided)
        env = os.environ.copy()
        if cache_dir:
            env["RINEX_CACHE_DIR"] = str(cache_dir)

        # Run bash download script (it caches directory listing internally)
        result = subprocess.run(
            ["bash", str(download_script), station_upper, str(year), str(doy), str(output_path)],
            capture_output=True,
            text=True,
            timeout=120,
            env=env
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


def download_rinex_batch(stations, year, doy, output_dir, logger=None, max_workers=8):
    """
    Download RINEX files for multiple stations in parallel.
    
    Args:
        stations: List of station names
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save files
        logger: Optional logger instance
        max_workers: Number of parallel download threads
    
    Returns:
        Dictionary mapping station names to downloaded file paths
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Downloading RINEX files for {len(stations)} stations ({year}/{doy:03d}) with {max_workers} threads")
    
    results = {}
    
    # Use ThreadPoolExecutor for parallel downloads, with a shared temp dir for caching listings
    with tempfile.TemporaryDirectory(prefix=f"rinex_cache_{year}_{doy:03d}_") as cache_dir, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        # Submit all download tasks
        future_to_station = {
            # Pass the shared cache_dir to each task
            executor.submit(download_rinex_file, station, year, doy, output_dir, logger, cache_dir): station
            for station in stations
        }
        
        # Process results as they complete
        for future in as_completed(future_to_station):
            station = future_to_station[future]
            try:
                rinex_path = future.result()
                if rinex_path:
                    results[station] = rinex_path
            except Exception as e:
                logger.error(f"Thread error downloading {station}: {e}")

    logger.info(f"Successfully downloaded {len(results)}/{len(stations)} RINEX files")
    return results
