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
    evaluate    - REMOVED, always broken - see 'inference' instead
    inference   - Run inference on test data
    positioning - REMOVED, never implemented - see positioning/scripts/run_pipeline.py
    map         - Generate spatial STEC maps

Examples:
    # Train a model
    python cli.py train --config config/config.yaml

    # Compare STEC model against baselines
    python cli.py compare \\
        --stec_experiment "Finetune_STEC_..." \\
        --vtec_experiment "Finetune_VTEC_..."

    # Evaluate on the test set (metrics + plots)
    python cli.py inference --experiment "Finetune_STEC_..."

    # Get help for any command
    python cli.py train --help
    python cli.py compare --help
"""

import sys
import argparse
import importlib
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _load_src_main(module_name, subcommand):
    """Import `<module_name>.main` from src/, or exit with a clear diagnosis.

    train/compare/inference/map/multiday are the last cli.py subcommands still backed by
    src/ - each rests on hundreds to thousands of lines of orchestration (training loop
    with early stopping and wandb, Figures 4-9's live-inference path, ensemble/MC-dropout
    decomposition, spatial-grid inference) with no stec/ replacement of equal capability
    yet, so there is nothing to fall back to. A bare `from <module> import main` at the
    call site used to raise a raw ModuleNotFoundError traceback the moment src/ moves or
    is deleted, with nothing at the failure site explaining why - the same "quietly does
    something confusing" failure mode CLAUDE.md warns about, just at the exception level
    instead of the numbers level. This gives the same clear, non-guessing diagnosis
    run_evaluate/run_positioning already print for their two permanently-removed
    subcommands, without changing anything about what happens once the import succeeds.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        print(
            f"cli.py {subcommand} needs src/{module_name}.py, which is not importable "
            f"({exc}). This subcommand has no stec/ replacement with equivalent "
            "capability yet - see CLAUDE.md's \"src/'s status\" section and "
            "docs/revision/retirement_inventory.md for what src/ is still needed for.",
            file=sys.stderr,
        )
        sys.exit(1)
    return module.main


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
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
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
        """,
    )

    # Required arguments
    parser.add_argument(
        "--stec_experiment",
        type=str,
        required=True,
        help="STEC model experiment name or path",
    )
    parser.add_argument(
        "--vtec_experiment",
        type=str,
        default=None,
        help="VTEC model experiment name or path (optional)",
    )

    # Inference parameters
    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=100,
        help="MC samples for Bayesian inference (default: 100)",
    )
    parser.add_argument(
        "--test_size", default=None, help="Number of test samples (default: full)"
    )

    # Data sources
    parser.add_argument(
        "--madrigal_path",
        type=str,
        default="/home/space/data/iono/Madrigal_STEC",
        help="Path to Madrigal STEC data",
    )
    parser.add_argument(
        "--no_gim", action="store_true", help="Skip IGS GIM baseline comparison"
    )
    parser.add_argument(
        "--gim_path",
        type=str,
        default="/home/space/project/2022_shumao_IonoSpatialModeling/07_data/GNSS_ionex",
        help="Path to GIM/IONEX data",
    )

    # Other options
    parser.add_argument(
        "--mapping_function",
        type=str,
        default="MSLM",
        choices=["SLM", "MSLM"],
        help="Mapping function for VTEC→STEC (default: MSLM)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Additional output directory"
    )

    return parser


def create_evaluate_parser(subparsers):
    """Create parser for the removed evaluation command.

    `from evaluation import main` (below, in `run_evaluate`) has resolved to the
    *package* `src/evaluation/` (which has no `main`) rather than the flat module
    `src/evaluation.py` it was written against, ever since `src/evaluation/` was added
    - confirmed by import, not by reading: `hasattr(evaluation, "main")` is `False`.
    This command has therefore always raised an `ImportError` before doing anything,
    independent of and unrelated to the `src/` retirement; `inference_testset.py` (the
    `inference` subcommand) is what real evaluation runs have used instead (CLAUDE.md:
    "That path is not the one used for paper numbers"). Kept registered, rather than
    removed outright, so `python cli.py evaluate --help`/`--help` still lists it with a
    pointer to the working command, instead of argparse's opaque "invalid choice".
    """
    parser = subparsers.add_parser(
        "evaluate",
        help="Removed - always broken (see 'inference' instead)",
        description="Removed: this command's target module was never reachable "
        "(see run_evaluate's docstring). Use 'inference' - it already covers "
        "'compute metrics and generate plots' via the same underlying driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        """,
    )

    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment name or path"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=None,
        help="Number of test samples (default: full)",
    )
    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=100,
        help="MC samples for Bayesian models (default: 100)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_checkpoint.pth",
        help="Checkpoint filename (default: best_checkpoint.pth)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file path (default: experiment_dir/predictions.csv)",
    )

    return parser


def create_positioning_parser(subparsers):
    """Create parser for the removed positioning command.

    `run_positioning` (below) targeted `inference_positioning.py` - a module that does
    not exist anywhere in this repository, under `src/` or otherwise (confirmed by
    `find . -iname "inference_positioning*"`, zero hits). This command has therefore
    never worked, on any branch, independent of the `src/` retirement: it is not a
    module that got lost, it is a caller that was never wired up. The real positioning
    driver is `positioning/positioning_eval/run_positioning_evaluation.py`, normally
    invoked through `positioning/scripts/run_pipeline.py` (CLAUDE.md's documented
    workflow) - both are deliberate, permanent `src/`-independent tooling, unaffected
    by anything in this file. Kept registered, rather than removed outright, so
    `python cli.py positioning --help` still lists it with a pointer to the real
    driver, instead of argparse's opaque "invalid choice".
    """
    parser = subparsers.add_parser(
        "positioning",
        help="Removed - never implemented (see positioning/scripts/run_pipeline.py)",
        description="Removed: this command's target module (inference_positioning.py) "
        "never existed. Use positioning/scripts/run_pipeline.py or "
        "positioning/positioning_eval/run_positioning_evaluation.py directly.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        """,
    )

    parser.add_argument(
        "--experiment", type=str, required=True, help="Experiment name or path"
    )
    parser.add_argument(
        "--date", type=str, required=True, help="Date for map generation (YYYY-MM-DD)"
    )
    parser.add_argument("--time", type=str, default=None, help="Specific time (HH:MM)")
    parser.add_argument(
        "--time_series",
        type=str,
        default=None,
        help="Time range for series (HH:MM-HH:MM)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1h",
        help="Time interval for series (default: 1h)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_checkpoint.pth",
        help="Checkpoint filename (default: best_checkpoint.pth)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for maps"
    )

    return parser


def create_multiday_parser(subparsers):
    """Create parser for multi-day evaluation command."""
    parser = subparsers.add_parser(
        "multiday",
        help="Multi-day evaluation across multiple days",
        description="Automated training and evaluation pipeline for multiple test days",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single day
  python cli.py multiday \\
      --dates "2024-183" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml
  
  # Multiple specific days
  python cli.py multiday \\
      --dates "2024-183,2024-184,2024-185" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml
  
  # Date range (inclusive, one week)
  python cli.py multiday \\
      --dates "2024-183:2024-189" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml \\
      --output_dir multiday_results/july_week1
  
  # Quick test
  python cli.py multiday \\
      --dates "2024-183,2024-184" \\
      --stec_config config/config.yaml \\
      --vtec_config config/config_vtec_mlp_baseline.yaml \\
      --num_inference_samples 10

What it does:
  For each date:
    1. Finetune STEC model on that day
    2. Finetune VTEC model on that day
    3. Run comprehensive comparison (STEC vs VTEC+Mapping vs GIM)
    4. Store results in organized structure
  
  Finally:
    - Generate aggregate statistics across all days
    - Create comparison plots and summary tables
    - Perfect for robust paper evaluations

Date formats:
  - YYYY-DOY: "2024-183"
  - YYYY-MM-DD: "2024-07-01"  
  - Range: "2024-183:2024-192"
  - List: "2024-183,2024-184,2024-185"
        """,
    )

    parser.add_argument(
        "--dates", type=str, help="Date(s) to evaluate (single, list, or range)"
    )
    parser.add_argument(
        "--stec_config", type=str, help="Base config file for STEC training"
    )
    parser.add_argument(
        "--vtec_config", type=str, help="Base config file for VTEC training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multiday_results",
        help="Output directory (default: multiday_results)",
    )
    parser.add_argument(
        "--num_inference_samples",
        type=int,
        default=100,
        help="MC samples for Bayesian inference (default: 100)",
    )
    parser.add_argument(
        "--test_size", type=int, default=None, help="Test set size (default: full)"
    )
    parser.add_argument(
        "--pretrain_folder",
        type=str,
        default=None,
        help="Pretrain experiment folder to use for STEC model (optional)",
    )
    parser.add_argument(
        "--pretrained_baseline",
        type=str,
        default=None,
        help="Pretrained model folder to use as a FIXED baseline in all plots",
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only run evaluation (experiments must exist)",
    )
    parser.add_argument(
        "--update_vtec_only",
        action="store_true",
        help="Reuse existing STEC and GIM results and only update VTEC evaluation",
    )
    parser.add_argument(
        "--skip_plots", action="store_true", help="Skip plot generation to save time"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip training if experiment already exists",
    )
    parser.add_argument(
        "--no_aggregate",
        action="store_true",
        help="Skip aggregate report generation (for parallel execution)",
    )
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip processing, only generate aggregate report from existing results",
    )
    parser.add_argument(
        "--skip_comparison",
        action="store_true",
        help="Skip comparison evaluation (Step 3)",
    )
    parser.add_argument(
        "--positioning",
        action="store_true",
        help="Run positioning evaluation for each day",
    )

    return parser


def run_train(args):
    """Execute training workflow."""
    main = _load_src_main("main", "train")

    main(config_path=args.config)


def run_compare(args):
    """Execute comparison workflow."""
    main = _load_src_main("compare_stec_vtec_gim", "compare")

    main(args)


def run_evaluate(args):
    """Removed: `from evaluation import main` has always resolved to the `src/evaluation/`
    package (no `main` attribute) rather than the flat `src/evaluation.py` module this
    command was written against - an ImportError on every invocation, unrelated to and
    predating the `src/` retirement. See `create_evaluate_parser`'s docstring."""
    print(
        "cli.py evaluate has been removed: its target module was never actually "
        "reachable (see cli.py's run_evaluate docstring for why). Use:\n"
        "  python cli.py inference --experiment <name>\n"
        "which already computes metrics and generates plots via the same underlying "
        "driver (src/inference_testset.py).",
        file=sys.stderr,
    )
    sys.exit(1)


def run_inference(args):
    """Execute inference workflow."""
    import sys

    sys.argv = [
        "inference_testset.py",
        "--experiment",
        args.experiment,
        "--checkpoint",
        args.checkpoint,
        "--num_inference_samples",
        str(args.num_inference_samples),
    ]

    if args.test_size:
        sys.argv.extend(["--test_size", str(args.test_size)])

    if args.output_file:
        sys.argv.extend(["--output_file", args.output_file])

    main = _load_src_main("inference_testset", "inference")

    main()


def run_positioning(args):
    """Removed: targeted `inference_positioning.py`, which does not exist anywhere in
    this repository and never has - not a `src/` retirement casualty. See
    `create_positioning_parser`'s docstring."""
    print(
        "cli.py positioning has been removed: it targeted inference_positioning.py, "
        "which never existed in this repository. Use:\n"
        "  positioning/scripts/run_pipeline.py\n"
        "or, for one experiment/day directly:\n"
        "  positioning/positioning_eval/run_positioning_evaluation.py\n"
        "(see CLAUDE.md's positioning workflow and docs/revision/retirement_inventory.md).",
        file=sys.stderr,
    )
    sys.exit(1)


def run_map(args):
    """Execute map generation workflow."""
    import sys

    sys.argv = [
        "inference_map.py",
        "--experiment",
        args.experiment,
        "--date",
        args.date,
        "--checkpoint",
        args.checkpoint,
    ]

    if args.time:
        sys.argv.extend(["--time", args.time])

    if args.time_series:
        sys.argv.extend(["--time_series", args.time_series])
        sys.argv.extend(["--interval", args.interval])

    if args.output_dir:
        sys.argv.extend(["--output_dir", args.output_dir])

    main = _load_src_main("inference_map", "map")

    main()


def run_multiday(args):
    """Execute multi-day evaluation workflow."""
    import sys

    if args.summary_only:
        sys.argv = [
            "multiday_evaluation.py",
            "--summary_only",
            "--output_dir",
            args.output_dir,
        ]
    else:
        if not args.dates or not args.stec_config or not args.vtec_config:
            print(
                "Error: --dates, --stec_config, and --vtec_config are required unless --summary_only is used"
            )
            return

        sys.argv = [
            "multiday_evaluation.py",
            "--dates",
            args.dates,
            "--stec_config",
            args.stec_config,
            "--vtec_config",
            args.vtec_config,
            "--output_dir",
            args.output_dir,
            "--num_inference_samples",
            str(args.num_inference_samples),
        ]

        if args.test_size:
            sys.argv.extend(["--test_size", str(args.test_size)])

        if args.pretrain_folder:
            sys.argv.extend(["--pretrain_folder", args.pretrain_folder])

        if args.pretrained_baseline:
            sys.argv.extend(["--pretrained_baseline", args.pretrained_baseline])

        if args.skip_training:
            sys.argv.extend(["--skip_training"])

        if args.skip_comparison:
            sys.argv.extend(["--skip_comparison"])

        if args.skip_existing:
            sys.argv.extend(["--skip_existing"])

        if args.update_vtec_only:
            sys.argv.extend(["--update_vtec_only"])

        if args.skip_plots:
            sys.argv.extend(["--skip_plots"])

        if args.no_aggregate:
            sys.argv.extend(["--no_aggregate"])

        if args.positioning:
            sys.argv.extend(["--positioning"])

    main = _load_src_main("multiday_evaluation", "multiday")

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
  evaluate    REMOVED - always broken; use 'inference' instead
  inference   Run inference on test data
  positioning REMOVED - never implemented; use positioning/scripts/run_pipeline.py
  map         Generate spatial STEC maps
  multiday    Multi-day evaluation for robust paper results (NEW!)

Examples:
  python cli.py train --config config/config.yaml
  python cli.py compare --stec_experiment "Finetune_STEC_..." --vtec_experiment "Finetune_VTEC_..."
  python cli.py inference --experiment "Finetune_STEC_..."
  python cli.py multiday --dates "2024-183:2024-189" --stec_config config/config.yaml --vtec_config config/config_vtec_mlp_baseline.yaml

For detailed help on any command:
  python cli.py <command> --help
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create subparsers for each command
    create_train_parser(subparsers)
    create_compare_parser(subparsers)
    create_evaluate_parser(subparsers)
    create_inference_parser(subparsers)
    create_positioning_parser(subparsers)
    create_map_parser(subparsers)
    create_multiday_parser(subparsers)

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
    elif args.command == "multiday":
        run_multiday(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
