#!/usr/bin/env python3
"""
Evaluation orchestrator for GNSS-based STEC ML model performance assessment.

This module coordinates three external evaluation flows without implementing evaluation logic:
  1. GIM → slant comparison: Compare model predictions against Global Ionospheric Maps
  2. VLBI dTEC comparison: Validate against Very Long Baseline Interferometry differential TEC
  3. Madrigal comparison: Evaluate against Madrigal database observations

Scope: Configuration management, I/O coordination, and run selection only.
No evaluation implementations are contained within this file.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from evaluation import run_stec_evaluation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        # Output and basic settings
        'output_dir': './eval_results',
        'enable_plots': True,
        'save_csvs': True,
        
        # Legacy evaluation modes
        'run_gim_slant': True,
        'run_vlbi_dtec': False,
        'run_madrigal': False,
        
        # STEC evaluation settings
        'mode': 'stec',
        'dataset': 'testset',
        'gnss_path': None,
        'madrigal_path': None,
        'vgosdb_path': None,
        'gim_path': None,
        'model_path': None,
        'shell_height_km': 450.0,
        'earth_radius_km': 6371.0,
        'group_key': 'station',
        'remove_bias': True
    }
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config_eval.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config from: {config_path}")
        
        # Flatten YAML structure into simple dict
        flat_config = {}
        
        # Basic settings
        flat_config['mode'] = config.get('mode', 'stec')
        flat_config['output_dir'] = config.get('output', {}).get('output_dir', './eval_results')
        flat_config['enable_plots'] = config.get('output', {}).get('enable_plots', True)
        flat_config['save_csvs'] = config.get('output', {}).get('save_csvs', True)
        
        # STEC evaluation settings
        stec_config = config.get('stec_evaluation', {})
        flat_config['dataset'] = stec_config.get('dataset', 'testset')
        flat_config['gnss_path'] = stec_config.get('gnss_path')
        flat_config['madrigal_path'] = stec_config.get('madrigal_path')
        flat_config['vgosdb_path'] = stec_config.get('vgosdb_path')
        flat_config['gim_path'] = stec_config.get('gim_path')
        flat_config['model_path'] = stec_config.get('model_path')
        flat_config['shell_height_km'] = stec_config.get('shell_height_km', 450.0)
        flat_config['earth_radius_km'] = stec_config.get('earth_radius_km', 6371.0)
        flat_config['group_key'] = stec_config.get('group_key', 'station')
        flat_config['remove_bias'] = stec_config.get('remove_bias', True)
        
        # Legacy evaluation settings
        legacy_config = config.get('legacy_evaluation', {})
        flat_config['run_gim_slant'] = legacy_config.get('run_gim_slant', True)
        flat_config['run_vlbi_dtec'] = legacy_config.get('run_vlbi_dtec', False)
        flat_config['run_madrigal'] = legacy_config.get('run_madrigal', False)
        
        return flat_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return defaults


def run_stec_comparison(cfg: Dict[str, Any]) -> None:
    """STEC evaluation coordinator - compare model STEC vs GIM→STEC."""
    logger.info("Running STEC comparison evaluation")
    run_stec_evaluation(cfg)


def run_gim_slant(output_dir: str) -> None:
    """TODO: GIM slant evaluation coordinator."""
    logger.info("Running GIM slant evaluation")
    logger.warning("Implementation not available")
    # TODO: from .eval_gim_slant import run_evaluation


def run_vlbi_dtec(output_dir: str) -> None:
    """TODO: VLBI dTEC evaluation coordinator."""
    logger.info("Running VLBI dTEC evaluation")
    logger.warning("Implementation not available")
    # TODO: from .eval_vlbi_dtec import run_evaluation


def run_madrigal(output_dir: str) -> None:
    """TODO: Madrigal evaluation coordinator."""
    logger.info("Running Madrigal evaluation")
    logger.warning("Implementation not available")
    # TODO: from .eval_madrigal import run_evaluation


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="STEC model evaluation orchestrator")
    
    # Evaluation mode
    parser.add_argument("--mode", choices=["stec", "legacy"], default="stec", 
                       help="Evaluation mode: stec=STEC space comparison, legacy=original flows")
    
    # STEC evaluation options
    parser.add_argument("--dataset", choices=["testset", "madrigal", "grid", "vgosdb"], 
                       help="Observation source for STEC evaluation")
    parser.add_argument("--gnss", type=Path, help="Path to GNSS testset data")
    parser.add_argument("--madrigal", type=Path, help="Path to Madrigal LOS data") 
    parser.add_argument("--vgosdb", type=Path, help="Path to VgosDB VLBI data")
    parser.add_argument("--gim", type=Path, help="Path to GIM data source")
    parser.add_argument("--model", type=Path, help="Path to trained STEC model")
    
    # Physical parameters
    parser.add_argument("--shell-height-km", type=float, help="Shell height in km")
    parser.add_argument("--earth-radius-km", type=float, help="Earth radius in km")
    parser.add_argument("--group-key", choices=["station", "station_sat"], 
                       help="Grouping strategy for bias removal")
    
    # Legacy evaluation options
    parser.add_argument("--run", choices=["gim", "vlbi", "madrigal", "all"], 
                       help="Legacy evaluation type")
    
    # Output options
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--enable-plots", action="store_true", help="Enable plots")
    parser.add_argument("--save-csvs", action="store_true", help="Save CSV results")
    
    args = parser.parse_args()
    
    # Load base config and apply CLI overrides
    config = load_config(args.config)
    
    # Apply CLI overrides (only if explicitly provided)
    if args.mode:
        config['mode'] = args.mode
    if args.dataset:
        config['dataset'] = args.dataset
    if args.gnss:
        config['gnss_path'] = str(args.gnss)
    if args.madrigal:
        config['madrigal_path'] = str(args.madrigal)
    if args.vgosdb:
        config['vgosdb_path'] = str(args.vgosdb)
    if args.gim:
        config['gim_path'] = str(args.gim)
    if args.model:
        config['model_path'] = str(args.model)
    if args.shell_height_km:
        config['shell_height_km'] = args.shell_height_km
    if args.earth_radius_km:
        config['earth_radius_km'] = args.earth_radius_km
    if args.group_key:
        config['group_key'] = args.group_key
    if args.out:
        config['output_dir'] = args.out
    if args.enable_plots:
        config['enable_plots'] = True
    if args.save_csvs:
        config['save_csvs'] = True
    
    logger.info(f"Starting evaluation - Mode: {config['mode']} - Output: {config['output_dir']}")
    
    # Route to appropriate evaluation
    if config['mode'] == 'stec':
        logger.info(f"STEC evaluation with dataset: {config.get('dataset', 'NOT_SET')}")
        run_stec_comparison(config)
    
    elif config['mode'] == 'legacy':
        # Legacy evaluation flows
        run_type = args.run or 'gim'
        do_gim = run_type in ["gim", "all"]
        do_vlbi = run_type in ["vlbi", "all"]
        do_madrigal = run_type in ["madrigal", "all"]
        
        output_dir = config['output_dir']
        evaluations = []
        
        if do_gim:
            run_gim_slant(output_dir)
            evaluations.append("GIM slant")
        if do_vlbi:
            run_vlbi_dtec(output_dir)
            evaluations.append("VLBI dTEC")
        if do_madrigal:
            run_madrigal(output_dir)
            evaluations.append("Madrigal")
        
        if evaluations:
            logger.info(f"Completed legacy evaluations: {', '.join(evaluations)}")
        else:
            logger.warning("No legacy evaluations were executed")
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()