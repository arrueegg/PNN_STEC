#!/usr/bin/env python3
"""
PNN_STEC Command Line Interface

Unified entry point for all PNN_STEC workflows.
Provides clean subcommands for training, evaluation, and inference tasks.

Usage:
    python cli.py <command> [options]
    
Available commands:
    train       - Train models (pretrain/finetune)
    compare     - Comprehensive STEC vs baselines comparison
    evaluate    - Evaluate model on test set
    inference   - Run inference on test data
    positioning - Evaluate positioning accuracy
    map         - Generate spatial STEC maps

Examples:
    # Train a model
    python cli.py train --config config/config.yaml
    
    # Compare STEC model against baselines
    python cli.py compare \\
        --stec_experiment "Finetune_STEC_..." \\
        --vtec_experiment "Finetune_VTEC_..."
    
    # Quick evaluation
    python cli.py evaluate --experiment "Finetune_STEC_..."
    
    # Get help for any command
    python cli.py train --help
    python cli.py compare --help
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def create_train_parser(subparsers):
    """Create parser for training command."""
    parser = subparsers.add_parser(
        "train",
        help="Train models (pretrain or finetune)",
        description="Train STEC/VTEC models using configuration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard pretraining
  python cli.py train --config config/config.yaml
  
  # Finetune on single day
  python cli.py train --config config/config_finetune.yaml
  
  # VTEC baseline training
  python cli.py train --config config/config_vtec_mlp_baseline.yaml
        """
    )
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file (default: config/config.yaml)")
    return parser


def create_compare_parser(subparsers):
    """Create parser for comparison command."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare STEC model against baselines",
        description="Comprehensive comparison: Direct STEC vs VTEC+Mapping vs IGS GIM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full comparison (recommended)
  python cli.py compare \\
      --stec_experiment "Finetune_STEC_2024_183_FactorizedSTEC_..." \\
      --vtec_experiment "Finetune_VTEC_2024_183_MLP_..."
  
  # STEC only (no VTEC baseline)
  python cli.py compare --stec_experiment "Finetune_STEC_..."
  
  # Quick test with subset
  python cli.py compare \\
      --stec_experiment "Finetune_STEC_..." \\
      --vtec_experiment "Finetune_VTEC_..." \\
      --test_size 1000 \\
      --num_inference_samples 10
  
  # Skip GIM comparison
  python cli.py compare \\
      --stec_experiment "Finetune_STEC_..." \\
      --no_gim

Automatic evaluation:
  - Own test set (always)
  - Madrigal independent test set (if available and model is finetuned)
  - VTEC+Mapping baseline (if --vtec_experiment provided)
  - IGS GIM baseline (enabled by default, use --no_gim to skip)
        """
    )
    
    # Required arguments
    parser.add_argument("--stec_experiment", type=str, required=True,
                       help="STEC model experiment name or path")
    parser.add_argument("--vtec_experiment", type=str, default=None,
                       help="VTEC model experiment name or path (optional)")
    
    # Inference parameters
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian inference (default: 100)")
    parser.add_argument("--test_size", default=None,
                       help="Number of test samples (default: full)")
    
    # Data sources
    parser.add_argument("--madrigal_path", type=str,
                       default="/home/space/data/iono/Madrigal_STEC",
                       help="Path to Madrigal STEC data")
    parser.add_argument("--no_gim", action="store_true",
                       help="Skip IGS GIM baseline comparison")
    parser.add_argument("--gim_path", type=str,
                       default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
                       help="Path to GIM/IONEX data")
    
    # Other options
    parser.add_argument("--mapping_function", type=str, default="MSLM",
                       choices=["SLM", "MSLM"],
                       help="Mapping function for VTEC→STEC (default: MSLM)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Additional output directory")
    
    return parser


def create_evaluate_parser(subparsers):
    """Create parser for evaluation command."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on test set",
        description="Basic model evaluation: compute metrics and generate plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set
  python cli.py evaluate --experiment "Finetune_STEC_..."
  
  # Quick evaluation on subset
  python cli.py evaluate --experiment "Finetune_STEC_..." --test_size 10000
        """
    )
    
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment name or path")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Number of test samples (default: full)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian models (default: 100)")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                       help="Checkpoint filename (default: best_checkpoint.pth)")
    
    return parser


def create_inference_parser(subparsers):
    """Create parser for inference command."""
    parser = subparsers.add_parser(
        "inference",
        help="Run inference on test data",
        description="Generate predictions on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference and save predictions
  python cli.py inference --experiment "Finetune_STEC_..."
  
  # Inference on subset
  python cli.py inference --experiment "Finetune_STEC_..." --test_size 10000
        """
    )
    
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment name or path")
    parser.add_argument("--test_size", type=int, default=None,
                       help="Number of test samples (default: full)")
    parser.add_argument("--num_inference_samples", type=int, default=100,
                       help="MC samples for Bayesian models (default: 100)")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                       help="Checkpoint filename (default: best_checkpoint.pth)")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file path (default: experiment_dir/predictions.csv)")
    
    return parser


def create_positioning_parser(subparsers):
    """Create parser for positioning evaluation command."""
    parser = subparsers.add_parser(
        "positioning",
        help="Evaluate positioning accuracy",
        description="Evaluate STEC model impact on GNSS positioning accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run positioning evaluation
  python cli.py positioning --experiment "Finetune_STEC_..."
  
  # Specify date range
  python cli.py positioning \\
      --experiment "Finetune_STEC_..." \\
      --start_date 2024-07-01 \\
      --end_date 2024-07-07
        """
    )
    
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment name or path")
    parser.add_argument("--start_date", type=str, default=None,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                       help="Checkpoint filename (default: best_checkpoint.pth)")
    
    return parser


def create_map_parser(subparsers):
    """Create parser for map generation command."""
    parser = subparsers.add_parser(
        "map",
        help="Generate spatial STEC maps",
        description="Generate spatial maps of STEC predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate map for specific time
  python cli.py map \\
      --experiment "Finetune_STEC_..." \\
      --date 2024-07-01 \\
      --time 12:00
  
  # Generate time series of maps
  python cli.py map \\
      --experiment "Finetune_STEC_..." \\
      --date 2024-07-01 \\
      --time_series 00:00-23:59 \\
      --interval 1h
        """
    )
    
    parser.add_argument("--experiment", type=str, required=True,
                       help="Experiment name or path")
    parser.add_argument("--date", type=str, required=True,
                       help="Date for map generation (YYYY-MM-DD)")
    parser.add_argument("--time", type=str, default=None,
                       help="Specific time (HH:MM)")
    parser.add_argument("--time_series", type=str, default=None,
                       help="Time range for series (HH:MM-HH:MM)")
    parser.add_argument("--interval", type=str, default="1h",
                       help="Time interval for series (default: 1h)")
    parser.add_argument("--checkpoint", type=str, default="best_checkpoint.pth",
                       help="Checkpoint filename (default: best_checkpoint.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for maps")
    
    return parser


def run_train(args):
    """Execute training workflow."""
    import sys
    # Set up arguments for main.py
    sys.argv = ["main.py", "--config", args.config]
    from main import main
    main()


def run_compare(args):
    """Execute comparison workflow."""
    import sys
    # Build argument list
    sys.argv = [
        "compare_stec_vtec_gim.py",
        "--stec_experiment", args.stec_experiment
    ]
    
    if args.vtec_experiment:
        sys.argv.extend(["--vtec_experiment", args.vtec_experiment])
    
    if args.test_size:
        sys.argv.extend(["--test_size", str(args.test_size)])
    
    sys.argv.extend(["--num_inference_samples", str(args.num_inference_samples)])
    sys.argv.extend(["--madrigal_path", args.madrigal_path])
    sys.argv.extend(["--mapping_function", args.mapping_function])
    
    if args.no_gim:
        sys.argv.append("--no_gim")
    else:
        sys.argv.extend(["--gim_path", args.gim_path])
    
    if args.output_dir:
        sys.argv.extend(["--output_dir", args.output_dir])
    
    from compare_stec_vtec_gim import main
    main()


def run_evaluate(args):
    """Execute evaluation workflow."""
    import sys
    sys.argv = [
        "evaluation.py",
        "--experiment", args.experiment,
        "--checkpoint", args.checkpoint,
        "--num_inference_samples", str(args.num_inference_samples)
    ]
    
    if args.test_size:
        sys.argv.extend(["--test_size", str(args.test_size)])
    
    from evaluation import main
    main()


def run_inference(args):
    """Execute inference workflow."""
    import sys
    sys.argv = [
        "inference_testset.py",
        "--experiment", args.experiment,
        "--checkpoint", args.checkpoint,
        "--num_inference_samples", str(args.num_inference_samples)
    ]
    
    if args.test_size:
        sys.argv.extend(["--test_size", str(args.test_size)])
    
    if args.output_file:
        sys.argv.extend(["--output_file", args.output_file])
    
    from inference_testset import main
    main()


def run_positioning(args):
    """Execute positioning evaluation workflow."""
    import sys
    sys.argv = [
        "inference_positioning.py",
        "--experiment", args.experiment,
        "--checkpoint", args.checkpoint
    ]
    
    if args.start_date:
        sys.argv.extend(["--start_date", args.start_date])
    
    if args.end_date:
        sys.argv.extend(["--end_date", args.end_date])
    
    from inference_positioning import main
    main()


def run_map(args):
    """Execute map generation workflow."""
    import sys
    sys.argv = [
        "inference_map.py",
        "--experiment", args.experiment,
        "--date", args.date,
        "--checkpoint", args.checkpoint
    ]
    
    if args.time:
        sys.argv.extend(["--time", args.time])
    
    if args.time_series:
        sys.argv.extend(["--time_series", args.time_series])
        sys.argv.extend(["--interval", args.interval])
    
    if args.output_dir:
        sys.argv.extend(["--output_dir", args.output_dir])
    
    from inference_map import main
    main()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="PNN_STEC: Bayesian Neural Network for STEC Modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  train       Train models (pretrain/finetune)
  compare     Compare STEC model against baselines (VTEC+Mapping, GIM)
  evaluate    Evaluate model on test set
  inference   Run inference on test data
  positioning Evaluate positioning accuracy
  map         Generate spatial STEC maps

Examples:
  python cli.py train --config config/config.yaml
  python cli.py compare --stec_experiment "Finetune_STEC_..." --vtec_experiment "Finetune_VTEC_..."
  python cli.py evaluate --experiment "Finetune_STEC_..."

For detailed help on any command:
  python cli.py <command> --help
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create subparsers for each command
    create_train_parser(subparsers)
    create_compare_parser(subparsers)
    create_evaluate_parser(subparsers)
    create_inference_parser(subparsers)
    create_positioning_parser(subparsers)
    create_map_parser(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute appropriate command
    if args.command == "train":
        run_train(args)
    elif args.command == "compare":
        run_compare(args)
    elif args.command == "evaluate":
        run_evaluate(args)
    elif args.command == "inference":
        run_inference(args)
    elif args.command == "positioning":
        run_positioning(args)
    elif args.command == "map":
        run_map(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
