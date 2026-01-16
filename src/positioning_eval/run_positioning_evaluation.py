#!/usr/bin/env python3
"""
Automated Positioning Evaluation Pipeline

Orchestrates the complete positioning evaluation workflow:
1. Download GNSS products (orbits, clocks, GIM)
2. Download RINEX observation files from CDDIS
3. Generate dynamic INI files for PPPx
4. Run PPPx positioning for each station with:
   - Custom model STEC corrections
   - Reference IGS GIM corrections
5. Compute and aggregate positioning accuracy metrics
6. Generate daily summary reports

Usage:
    python src/run_positioning_evaluation.py \\
        --experiment <experiment_folder> \\
        --date 2024-07-01 \\
        --stations ZIMM BRUS WTZR

    # Or process all test stations
    python src/run_positioning_evaluation.py \\
        --experiment <experiment_folder> \\
        --date 2024-07-01 \\
        --all_test_stations
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules using relative imports
from download_products import download_products, get_product_paths, find_igs_gim
from download_rinex import download_rinex_batch
from generate_ini import generate_pppx_ini
from metrics import parse_pos_file, compute_metrics, aggregate_daily_metrics, save_daily_summary


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def load_test_stations(station_list_path="./src/data_processing/test_station.list"):
    """Load test station list."""
    stations = np.loadtxt(station_list_path, dtype=str)
    return list(stations)


def find_experiment_directory(experiment_name, base_dir="experiments"):
    """Find experiment directory by name or partial match."""
    # Strip leading "experiments/" or "experiment/" if present
    experiment_name = experiment_name.removeprefix("experiments/").removeprefix("experiment/")
    # Remove trailing slash if present
    experiment_name = experiment_name.rstrip("/")
    
    experiments_path = Path(base_dir)
    
    exact_path = experiments_path / experiment_name
    if exact_path.exists():
        return exact_path
    
    matching_dirs = [d for d in experiments_path.iterdir() 
                     if d.is_dir() and experiment_name in d.name]
    
    if len(matching_dirs) == 1:
        return matching_dirs[0]
    elif len(matching_dirs) > 1:
        raise ValueError(f"Multiple experiments match '{experiment_name}': {[d.name for d in matching_dirs]}")
    else:
        raise ValueError(f"No experiment found matching '{experiment_name}'")


def run_pppx_positioning(rinex_file, ini_file, output_dir, pppx_executable, logger):
    """
    Run PPPx positioning for a single RINEX file.
    
    Args:
        rinex_file: Path to RINEX observation file
        ini_file: Path to PPPx INI configuration
        output_dir: Output directory for results
        pppx_executable: Path to pppx executable
        logger: Logger instance
    
    Returns:
        Path to .pos file or None if failed
    """
    rinex_path = Path(rinex_file)
    ini_path = Path(ini_file)
    output_path = Path(output_dir)
    
    if not rinex_path.exists():
        logger.error(f"RINEX file not found: {rinex_path}")
        return None
    
    if not ini_path.exists():
        logger.error(f"INI file not found: {ini_path}")
        return None
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run PPPx from the output directory with short relative paths
        # This avoids long path issues with PPPx
        
        # Copy/symlink RINEX to output directory for short path
        rinex_in_output = output_path / rinex_path.name
        if not rinex_in_output.exists():
            import shutil
            shutil.copy2(rinex_path, rinex_in_output)
        
        # Use just the filename for RINEX (it's in cwd)
        # Use just the INI filename (it's also in cwd - we created it there)
        cmd = [str(pppx_executable), str(rinex_path.name), str(ini_path.name)]
        
        result = subprocess.run(
            cmd,
            cwd=str(output_path),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            # Find generated .pos file (PPPx creates hidden files with . prefix)
            pos_files = list(output_path.glob(f".{rinex_path.stem}*.pos"))
            if not pos_files:
                # Fallback: try non-hidden
                pos_files = list(output_path.glob(f"{rinex_path.stem}*.pos"))
            
            # Clean up copied RINEX file
            if rinex_in_output.exists():
                rinex_in_output.unlink()
            
            if pos_files:
                logger.info(f"✓ PPPx completed: {pos_files[0].name}")
                return pos_files[0]
            else:
                logger.error(f"PPPx succeeded but no .pos file found")
                return None
        else:
            logger.error(f"PPPx failed: {result.stderr}")
            # Clean up copied RINEX even on failure
            if rinex_in_output.exists():
                rinex_in_output.unlink()
            return None
    
    except subprocess.TimeoutExpired:
        logger.error(f"PPPx timeout for {rinex_path.name}")
        return None
    except Exception as e:
        logger.error(f"PPPx error: {e}")
        return None


def process_single_station(
    station,
    year,
    doy,
    experiment_dir,
    products_dir,
    rinex_dir,
    pppx_executable,
    igs_gim_path,
    logger
):
    """
    Process positioning for a single station with both model and GIM.
    
    Returns:
        Dictionary with results status
    """
    results = {
        'station': station,
        'model_success': False,
        'gim_success': False,
        'model_pos': None,
        'gim_pos': None
    }
    
    station_upper = station.upper()
    
    # Find RINEX file - search for any country code match
    rinex_file = None
    
    # Try RINEX3 format with wildcard for country code (e.g., ZIMM00CHE, AIRA00JPN)
    rinex3_pattern = f"{station_upper}00*_R_{year}{doy:03d}0000_01D_30S_MO.rnx"
    matches = list(rinex_dir.glob(rinex3_pattern))
    if matches:
        rinex_file = matches[0]
    else:
        # Fallback to old RINEX format
        rinex_candidates = [
            rinex_dir / f"{station.lower()}{doy:03d}0.{str(year)[-2:]}d",
            rinex_dir / f"{station.lower()}{doy:03d}0.{str(year)[-2:]}o",
        ]
        for candidate in rinex_candidates:
            if candidate.exists():
                rinex_file = candidate
                break
    
    if not rinex_file:
        logger.warning(f"RINEX file not found for {station}")
        return results
    
    # Setup output directories
    model_output_dir = experiment_dir / "positioning" / "results" / f"{year}{doy:03d}" / "model" / station
    gim_output_dir = experiment_dir / "positioning" / "results" / f"{year}{doy:03d}" / "gim" / station
    
    # Find STEC CSV file
    stec_csv = experiment_dir / "positioning" / "stec_corrections" / f"{year}{doy:03d}" / f"{station}.csv"
    
    # 1. Run with model STEC corrections
    if stec_csv.exists():
        logger.info(f"Processing {station} with model STEC...")
        
        ini_file = model_output_dir / "pppx_model.ini"
        generate_pppx_ini(
            year=year,
            doy=doy,
            output_path=ini_file,
            products_dir=products_dir,
            ion_source="IONEX",
            ion_path=stec_csv,
            station_name=station,
            output_dir="./",  # PPPx runs from output directory
            output_ini_dir=model_output_dir
        )
        
        pos_file = run_pppx_positioning(
            rinex_file,
            ini_file,
            model_output_dir,
            pppx_executable,
            logger
        )
        
        if pos_file:
            # Rename for clarity
            final_pos = model_output_dir / f"{station}_model.pos"
            pos_file.rename(final_pos)
            results['model_pos'] = final_pos
            results['model_success'] = True
    else:
        logger.warning(f"STEC CSV not found for {station}: {stec_csv}")
    
    # 2. Run with IGS GIM
    if igs_gim_path and igs_gim_path.exists():
        logger.info(f"Processing {station} with IGS GIM...")
        
        ini_file = gim_output_dir / "pppx_gim.ini"
        generate_pppx_ini(
            year=year,
            doy=doy,
            output_path=ini_file,
            products_dir=products_dir,
            ion_source="IONEX",
            ion_path=igs_gim_path,
            station_name=station,
            output_dir="./",  # PPPx runs from output directory
            output_ini_dir=gim_output_dir
        )
        
        pos_file = run_pppx_positioning(
            rinex_file,
            ini_file,
            gim_output_dir,
            pppx_executable,
            logger
        )
        
        if pos_file:
            # Rename for clarity
            final_pos = gim_output_dir / f"{station}_gim.pos"
            pos_file.rename(final_pos)
            results['gim_pos'] = final_pos
            results['gim_success'] = True
    else:
        logger.warning(f"IGS GIM not found: {igs_gim_path}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run automated positioning evaluation pipeline"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment folder name"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="Date to process (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        help="Station names to process (e.g., ZIMM BRUS WTZR)"
    )
    parser.add_argument(
        "--all_test_stations",
        action="store_true",
        help="Process all stations from test_station.list"
    )
    parser.add_argument(
        "--skip_downloads",
        action="store_true",
        help="Skip product/RINEX downloads (use existing files)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up downloaded products and RINEX after processing"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel station processing jobs"
    )
    parser.add_argument(
        "--pppx_path",
        type=str,
        default="./src/positioning_eval/pppx",
        help="Path to pppx executable"
    )
    parser.add_argument(
        "--gim_base_path",
        type=str,
        default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
        help="Base path to IGS GIM directory"
    )
    
    args = parser.parse_args()
    logger = setup_logging()
    
    try:
        # Parse date
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        year = date_obj.year
        doy = date_obj.timetuple().tm_yday
        
        logger.info("=" * 80)
        logger.info("🚀 POSITIONING EVALUATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Date: {year}-{doy:03d}")
        
        # Find experiment
        experiment_dir = find_experiment_directory(args.experiment)
        logger.info(f"Experiment: {experiment_dir.name}")
        
        # Determine stations to process
        if args.all_test_stations:
            stations = load_test_stations()
            logger.info(f"Processing all {len(stations)} test stations")
        elif args.stations:
            stations = [s.upper() for s in args.stations]
            logger.info(f"Processing {len(stations)} stations: {', '.join(stations)}")
        else:
            logger.error("Must specify --stations or --all_test_stations")
            return 1
        
        # Setup directories
        products_dir = experiment_dir / "positioning" / "evaluation" / f"{year}{doy:03d}" / "products"
        rinex_dir = experiment_dir / "positioning" / "evaluation" / f"{year}{doy:03d}" / "rinex"
        
        # Step 1: Download products
        if not args.skip_downloads:
            logger.info("\n📥 Step 1: Downloading GNSS products...")
            products = download_products(year, doy, str(products_dir), logger)
            
            if not products:
                logger.error("Failed to download products")
                return 1
        else:
            logger.info("\n⏭️  Skipping product download")
        
        # Step 2: Find IGS GIM
        logger.info("\n🗺️  Step 2: Locating IGS GIM...")
        igs_gim_path = find_igs_gim(year, doy, args.gim_base_path)
        
        if igs_gim_path:
            logger.info(f"✓ Found IGS GIM: {igs_gim_path}")
        else:
            logger.warning(f"⚠️  IGS GIM not found - will skip GIM evaluation")
        
        # Step 3: Download RINEX files
        if not args.skip_downloads:
            logger.info("\n📥 Step 3: Downloading RINEX files...")
            # Use more threads for downloading (I/O bound) than processing (CPU bound)
            download_threads = max(4, args.parallel * 4) 
            rinex_results = download_rinex_batch(stations, year, doy, str(rinex_dir), logger, max_workers=download_threads)
            logger.info(f"✓ Downloaded {len(rinex_results)}/{len(stations)} RINEX files")
        else:
            logger.info("\n⏭️  Skipping RINEX download")
        
        # Step 4: Check PPPx executable
        pppx_path = Path(args.pppx_path)
        if not pppx_path.is_absolute():
            pppx_path = Path.cwd() / pppx_path
        
        if not pppx_path.exists():
            logger.error(f"PPPx executable not found: {pppx_path}")
            return 1
        logger.info(f"\n✓ PPPx executable: {pppx_path}")
        
        # Step 5: Process each station
        logger.info(f"\n🔄 Step 5: Running positioning for {len(stations)} stations...")
        logger.info("=" * 80)
        
        all_results = []
        
        if args.parallel > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        process_single_station,
                        station, year, doy, experiment_dir,
                        products_dir, rinex_dir, pppx_path,
                        igs_gim_path, logger
                    ): station
                    for station in stations
                }
                
                for future in tqdm(as_completed(futures), total=len(stations), desc="Stations"):
                    station = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        logger.error(f"Error processing {station}: {e}")
        else:
            # Sequential processing
            for station in tqdm(stations, desc="Stations"):
                result = process_single_station(
                    station, year, doy, experiment_dir,
                    products_dir, rinex_dir, pppx_path,
                    igs_gim_path, logger
                )
                all_results.append(result)
        
        # Step 6: Compute metrics
        logger.info("\n📊 Step 6: Computing positioning metrics...")
        
        model_success = sum(1 for r in all_results if r['model_success'])
        gim_success = sum(1 for r in all_results if r['gim_success'])
        
        logger.info(f"✓ Model positioning: {model_success}/{len(stations)} stations")
        logger.info(f"✓ GIM positioning: {gim_success}/{len(stations)} stations")
        
        # Aggregate metrics
        results_base_dir = experiment_dir / "positioning" / "results" / f"{year}{doy:03d}"
        
        metrics_model = None
        metrics_gim = None
        
        if model_success > 0:
            model_dir = results_base_dir / "model"
            metrics_model = aggregate_daily_metrics(
                model_dir, year, doy, "model",
                stations=[r['station'] for r in all_results if r['model_success']]
            )
        
        if gim_success > 0:
            gim_dir = results_base_dir / "gim"
            metrics_gim = aggregate_daily_metrics(
                gim_dir, year, doy, "gim",
                stations=[r['station'] for r in all_results if r['gim_success']]
            )
        
        # Step 7: Save summary
        if metrics_model is not None or metrics_gim is not None:
            summary_file = experiment_dir / "positioning" / "results" / f"{year}{doy:03d}" / "daily_summary.csv"
            
            # Combine metrics
            metrics_list = []
            if metrics_model is not None:
                metrics_list.append(metrics_model)
            if metrics_gim is not None:
                metrics_list.append(metrics_gim)
            
            if len(metrics_list) == 2:
                save_daily_summary(metrics_model, metrics_gim, summary_file)
            else:
                # Save single method
                combined = pd.concat(metrics_list, ignore_index=True)
                combined.to_csv(summary_file, index=False, float_format='%.4f')
                logger.info(f"\n✓ Summary saved to: {summary_file}")
            
            # Generate plots from summary
            logger.info("\n📊 Step 7b: Generating plots...")
            try:
                from plot_results import plot_positioning_results
                plot_results_output = experiment_dir / "positioning" / "results" / f"{year}{doy:03d}" / "plots"
                plot_positioning_results(summary_file, plot_results_output)
                logger.info(f"✓ Plots saved to: {plot_results_output}")
            except Exception as e:
                logger.warning(f"Could not generate plots: {e}")
        
        # Step 8: Cleanup (optional)
        if args.cleanup:
            logger.info("\n🗑️  Step 8: Cleaning up downloaded files...")
            
            # Remove products
            if products_dir.exists():
                import shutil
                shutil.rmtree(products_dir)
                logger.info("✓ Removed downloaded products")
            
            # Remove RINEX files
            if rinex_dir.exists():
                import shutil
                shutil.rmtree(rinex_dir)
                logger.info("✓ Removed downloaded RINEX files")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ POSITIONING EVALUATION COMPLETED!")
        logger.info("=" * 80)
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
