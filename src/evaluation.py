#!/usr/bin/env python3
"""
STEC Evaluation Orchestrator

Clean, focused orchestrator for STEC space model evaluation.
Compares STEC model predictions against GIM VTEC mapped to STEC.

Usage:
    python evaluation.py --dataset testset --gim data/gim/ --out results/
    python evaluation.py --dataset grid --gim data/gim/ --config my_config.yaml
    python evaluation.py --help
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from evaluation import run_stec_evaluation

# Setup logging for evaluation workflow
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        'output_dir': './eval_results',
        'dataset': 'testset',
        'shell_height_km': 450.0,
        'earth_radius_km': 6371.0,
        'group_key': 'station',
        'enable_plots': True,
        'save_csvs': True,
        'remove_bias': True
    }
    
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config_eval.yaml"
    
    if not config_path.exists():
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Flatten YAML structure into simple dict
        flat_config = {}
        flat_config.update(defaults)  # Start with defaults
        
        # Override with YAML values if they exist
        if 'stec_evaluation' in config:
            stec_config = config['stec_evaluation']
            for key in ['dataset', 'shell_height_km', 'earth_radius_km', 'group_key', 'remove_bias']:
                if key in stec_config:
                    flat_config[key] = stec_config[key]
            
            # Handle paths
            for path_key in ['gnss_path', 'madrigal_path', 'vgosdb_path', 'gim_path', 'model_path']:
                if path_key in stec_config and stec_config[path_key]:
                    flat_config[path_key] = stec_config[path_key]
        
        if 'output' in config:
            output_config = config['output']
            for key in ['output_dir', 'enable_plots', 'save_csvs']:
                if key in output_config:
                    flat_config[key] = output_config[key]
        
        return flat_config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return defaults


def main():
    """Main entry point for STEC evaluation."""
    parser = argparse.ArgumentParser(
        description="STEC model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--dataset", 
                       choices=["testset", "madrigal", "grid", "vgosdb"], 
                       default="testset",
                       help="Observation source for evaluation")
    parser.add_argument("--gim", 
                       type=Path, 
                       default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
                       help="Path to GIM data source")
    
    # Dataset-specific paths
    parser.add_argument("--gnss", type=Path, default="data/gnss/", help="Path to GNSS testset data")
    parser.add_argument("--madrigal", type=Path, default="data/madrigal/", help="Path to Madrigal data") 
    parser.add_argument("--vgosdb", type=Path, help="Path to VgosDB data")
    parser.add_argument("--model", type=Path, help="Path to trained STEC model")
    
    # Output and configuration
    parser.add_argument("--out", type=Path, default="./eval_results", 
                       help="Output directory (default: ./eval_results)")
    parser.add_argument("--config", type=Path, 
                       help="YAML config file (default: config/config_eval.yaml)")
    
    # Optional parameters
    parser.add_argument("--shell-height", type=float, default=450.0,
                       help="Shell height in km (default: 450.0)")
    parser.add_argument("--group-by", choices=["station", "station_sat"], default="station",
                       help="Grouping for bias removal (default: station)")
    
    # Flags
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--no-csvs", action="store_true", help="Disable CSV output")
    
    args = parser.parse_args()
    
    # Load base configuration
    config = load_config(args.config)
    
    # Apply CLI arguments (CLI takes precedence)
    config['dataset'] = args.dataset
    config['gim_path'] = str(args.gim)
    config['output_dir'] = str(args.out)
    config['shell_height_km'] = args.shell_height
    config['group_key'] = args.group_by
    config['enable_plots'] = not args.no_plots
    config['save_csvs'] = not args.no_csvs
    
    # Set dataset-specific paths
    if args.gnss:
        config['gnss_path'] = str(args.gnss)
    if args.madrigal:
        config['madrigal_path'] = str(args.madrigal)
    if args.vgosdb:
        config['vgosdb_path'] = str(args.vgosdb)
    if args.model:
        config['model_path'] = str(args.model)
    
    # Validate required paths for dataset
    required_paths = {
        'testset': 'gnss_path',
        'madrigal': 'madrigal_path', 
        'vgosdb': 'vgosdb_path'
    }
    
    if args.dataset in required_paths:
        required_path = required_paths[args.dataset]
        if not config.get(required_path):
            parser.error(f"--{required_path.replace('_path', '')} is required for dataset '{args.dataset}'")
    
    # Run evaluation
    run_stec_evaluation(config)


if __name__ == "__main__":
    main()