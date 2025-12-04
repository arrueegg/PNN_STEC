#!/usr/bin/env python3
"""
Product Downloader for GNSS Positioning

Python wrapper for downloading GNSS products (orbits, clocks, GIM, etc.)
Provides integration with positioning evaluation pipeline.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def download_products(year, doy, output_dir, logger=None):
    """
    Download GNSS products using the existing download_products.sh script.
    
    Args:
        year: Year (int)
        doy: Day of year (int)
        output_dir: Directory to save products
        logger: Optional logger instance
    
    Returns:
        Dictionary with paths to downloaded products
    """
    if logger is None:
        logger = setup_logging()
    
    # Get path to download script
    script_dir = Path(__file__).parent
    download_script = script_dir / "download_products.sh"
    
    if not download_script.exists():
        raise FileNotFoundError(f"Download script not found: {download_script}")
    
    # Create output directory
    products_path = Path(output_dir)
    products_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading products for {year}/{doy:03d} to {output_dir}")
    
    # Run download script - it expects to create ./products in CWD
    try:
        result = subprocess.run(
            ["bash", str(download_script), str(year), str(doy)],
            cwd=str(products_path.parent),  # Run in parent so it creates ./products
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            logger.info("✓ Products downloaded successfully")
            
            # Return paths to key products
            product_files = get_product_paths(year, doy, output_dir)
            return product_files
        else:
            logger.error(f"Product download failed: {result.stderr}")
            logger.debug(f"stdout: {result.stdout}")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error("Product download timed out")
        return None
    except Exception as e:
        logger.error(f"Error downloading products: {e}")
        return None


def get_product_paths(year, doy, products_dir):
    """
    Get paths to product files after download.
    
    Returns:
        Dictionary with product file paths
    """
    products_path = Path(products_dir)
    
    # Expected product filenames (CODE)
    sp3 = f"COD0OPSFIN_{year}{doy:03d}0000_01D_05M_ORB.SP3"
    clk = f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_CLK.CLK"
    erp = f"COD0OPSFIN_{year}{doy:03d}0000_01D_01D_ERP.ERP"
    obx = f"COD0OPSFIN_{year}{doy:03d}0000_01D_30S_ATT.OBX"
    ion = f"COD0OPSFIN_{year}{doy:03d}0000_01D_01H_GIM.INX"
    
    return {
        'sp3': products_path / sp3,
        'clk': products_path / clk,
        'erp': products_path / erp,
        'obx': products_path / obx,
        'ion': products_path / ion
    }


def find_igs_gim(year, doy, gim_base_path="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex"):
    """
    Find IGS GIM file for specified date.
    
    Args:
        year: Year (int)
        doy: Day of year (int)
        gim_base_path: Base path to IGS GIM directory
    
    Returns:
        Path to IGS GIM file or None if not found
    """
    gim_dir = Path(gim_base_path)
    
    # Common IGS GIM naming patterns
    patterns = [
        f"igsg{doy:03d}0.{str(year)[-2:]}i",  # igsYYYY0.YYi
        f"IGSG{doy:03d}0.{str(year)[-2:]}I",  # Upper case
        f"igsg{doy:03d}0.{str(year)[-2:]}i.Z",  # Compressed
        f"IGS0OPSFIN_{year}{doy:03d}0000_01D_02H_GIM.INX",  # New long format
    ]
    
    # Search in year/doy subdirectories and base directory
    search_dirs = [
        gim_dir,
        gim_dir / str(year),
        gim_dir / str(year) / f"{doy:03d}",
    ]
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        for pattern in patterns:
            matches = list(search_dir.glob(pattern))
            if matches:
                return matches[0]
    
    return None
