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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or return defaults."""
    defaults = {
        'output_dir': './eval_results',
        'enable_plots': True,
        'save_csvs': True,
        'run_gim_slant': True,
        'run_vlbi_dtec': False,
        'run_madrigal': False
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
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return defaults


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
    
    parser.add_argument("--config", type=Path, help="YAML config file")
    parser.add_argument("--run", choices=["gim", "vlbi", "madrigal", "all"], 
                       default="gim", help="Evaluation type")
    parser.add_argument("--out", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--plots", action="store_true", help="Enable plots")
    parser.add_argument("--csvs", action="store_true", help="Save CSV results")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    output_dir = args.out
    
    # Determine what to run
    run_gim = args.run in ["gim", "all"]
    run_vlbi = args.run in ["vlbi", "all"]
    run_madrigal = args.run in ["madrigal", "all"]
    
    logger.info(f"Starting evaluation - Output: {output_dir}")
    
    # Run evaluations
    evaluations = []
    if run_gim:
        run_gim_slant(output_dir)
        evaluations.append("GIM slant")
    
    if run_vlbi:
        run_vlbi_dtec(output_dir)
        evaluations.append("VLBI dTEC")
    
    if run_madrigal:
        run_madrigal(output_dir)
        evaluations.append("Madrigal")
    
    if evaluations:
        logger.info(f"Completed: {', '.join(evaluations)}")
    else:
        logger.warning("No evaluations were executed")
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()