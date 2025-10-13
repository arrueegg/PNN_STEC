#!/usr/bin/env python3
"""
Refactored Global Map Inference Script for PNN_STEC Project

This script generates global STEC maps by reusing existing codebase components:
- BaseTrainer for model loading and training infrastructure
- InferenceManager for Bayesian inference and uncertainty quantification  
- CollateWithSH for proper feature transformation
- Existing coordinate transformation utilities
- Feature registry for consistent feature handling

Key Improvements:
- Reuses existing infrastructure instead of custom implementations
- Proper feature transformations using CollateWithSH
- Better error handling and logging
- Cleaner separation of concerns

Usage:
    python src/inference_map.py --date 2024-05-15 --elevation 30.0 --azimuth 180.0

Parameters:
    --date: Date for map generation (YYYY-MM-DD format)
    --elevation: Fixed elevation angle in degrees (default: 30.0)
    --azimuth: Fixed azimuth angle in degrees (default: 180.0)
    --lat_res: Latitude resolution in degrees (default: 2.5)
    --lon_res: Longitude resolution in degrees (default: 5.0)

Output:
- Hourly numpy files (.npz) with global STEC predictions and metadata
- Summary plots and statistics
- All saved to: experiments/<experiment_name>/global_maps/
"""

import torch
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from pathlib import Path
from torch.utils.data import DataLoader

# Add the parent directory to sys.path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config, compute_exp_name
from utils.feature_registry import initialize_feature_registry
from utils.coordinate_transforms import create_global_grid
from utils.ionex_writer import IONEXWriter, generate_ionex_filename
from data_loader.multitemporal_inference_dataset import create_multitemporal_inference_dataloader
from data_loader.collation import CollateWithSH
from training.base_trainer import BaseTrainer
from model.model import get_model
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate global STEC maps")
    parser.add_argument(
        "--date",
        type=str,
        default="2024-01-01",
        help="Date for map generation (YYYY-MM-DD format)",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=90.0,
        help="Fixed elevation angle in degrees (default: 90.0)",
    )
    parser.add_argument(
        "--azimuth",
        type=float,
        default=180.0,
        help="Fixed azimuth angle in degrees (default: 180.0)",
    )
    parser.add_argument(
        "--lat_res",
        type=float,
        default=5.0, 
        help="Latitude resolution in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--lon_res",
        type=float,
        default=5.0,
        help="Longitude resolution in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--time_res",
        type=float,
        default=1.0,
        help="Time resolution in hours (default: 1.0)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
        help="Path to config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--create_gif",
        action="store_true",
        default=True,
        help="Create a GIF animation from all hourly maps",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=0,
        help="Minimum value for STEC colorscale (default: 0)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=80,
        help="Maximum value for STEC colorscale (default: 80)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size for inference (default: 10000)",
    )
    parser.add_argument(
        "--output_format",
        choices=["png", "ionex", "both"],
        default="png",
        help="Output format: png (plots only), ionex (IONEX files only), or both (default: png)",
    )
    parser.add_argument(
        "--ionex_center",
        type=str,
        default="ETH",
        help="Analysis center code for IONEX header (default: ETH)",
    )
    return parser.parse_args()


def find_experiment_directory(experiment_name, base_dir="experiments"):
    """Find the experiment directory that matches the given name."""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory {base_dir} does not exist")

    # Look for exact match first
    exact_path = os.path.join(base_dir, experiment_name)
    if os.path.exists(exact_path):
        return exact_path

    # If no exact match, look for partial matches
    matches = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and experiment_name in item:
            matches.append(item_path)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple experiment directories match '{experiment_name}': {matches}"
        )
    else:
        raise FileNotFoundError(
            f"No experiment directory found matching '{experiment_name}'"
        )


def find_model_checkpoint(experiment_dir):
    """Find the model checkpoint in the experiment directory."""
    model_dir = os.path.join(experiment_dir, "model")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Look for .pth files
    pth_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

    if len(pth_files) == 0:
        raise FileNotFoundError(
            f"No model checkpoint (.pth) files found in {model_dir}"
        )
    elif len(pth_files) == 1:
        return os.path.join(model_dir, pth_files[0])
    else:
        # If multiple, prefer the one with 'pretrain' in the name
        pretrain_files = [f for f in pth_files if "pretrain" in f.lower()]
        if pretrain_files:
            return os.path.join(model_dir, pretrain_files[0])
        else:
            return os.path.join(model_dir, pth_files[0])



def run_inference_with_trainer(trainer, dataloader, model):
    """
    Run inference using the existing InferenceManager.

    Args:
        trainer: BaseTrainer instance with InferenceManager
        dataloader: DataLoader with inference data
        model: Loaded model

    Returns:
        tuple: (predictions, uncertainties) from InferenceManager
    """
    # Use the InferenceManager for proper Bayesian inference
    # The BaseTrainer's InferenceManager handles all the complexity:
    # - Bayesian sampling for uncertainty quantification
    # - Proper target transformations
    # - Memory-efficient processing
    # - Model type detection (BNN, MLP, etc.)
    metrics_dict, results_df = trainer.inference_manager.bayesian_inference_total_uncertainty(
        model=model,
        dataloader=dataloader,
        num_samples=100  # Number of Bayesian samples
    )

    return metrics_dict, results_df


def save_hourly_plot(
    lat_grid,
    lon_grid,
    stec_map,
    uncertainty_map,
    output_path,
    date_str,
    hour,
    elevation,
    azimuth,
    vmin=0,
    vmax=80,
):
    """
    Save individual hourly STEC map as PNG file with GIM-style formatting.

    Args:
        lat_grid: 2D latitude grid
        lon_grid: 2D longitude grid
        stec_map: 2D STEC prediction array
        uncertainty_map: 2D uncertainty array (can be None)
        output_path: Base path to save files (without extension)
        date_str: Date string
        hour: Hour of day
        elevation: Elevation angle
        azimuth: Azimuth angle
        vmin: Minimum value for colorscale
        vmax: Maximum value for colorscale
    """
    # Create figure with cartographic projection matching GIM style
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Create the STEC map plot with fixed colorscale
    im = ax.pcolormesh(
        lon_grid,
        lat_grid,
        stec_map,
        cmap="gist_heat",
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

    # Add coastlines (white like in GIM plots)
    ax.coastlines(color="white")

    # Set title matching GIM style
    ax.set_title(
        f"PNN STEC for {date_str} {int(hour):02d}:{int((hour % 1) * 60):02d} UTC\nElevation: {elevation}°, Azimuth: {azimuth}°",
        fontweight="bold",
        fontsize=16,
    )

    # Set labels and formatting matching GIM style
    ax.set_xlabel("Longitude", fontsize=14)
    ax.set_ylabel("Latitude", fontsize=14)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)

    # Set global extent
    ax.set_global()

    # Add colorbar matching GIM style
    cbar = fig.colorbar(im, ax=ax, label="STEC (TECU)", shrink=0.8)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("STEC (TECU)", fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_gif(image_paths, output_path):
    """
    Create a GIF from a list of image paths.

    Parameters:
    image_paths (list): List of paths to PNG images
    output_path (str): Path for the output GIF file
    duration (int): Duration between frames in milliseconds
    """
    if not image_paths:
        logger.warning("No images found to create GIF")
        return

    # Load all images
    images = []
    for path in image_paths:
        if os.path.exists(path):
            img = Image.open(path)
            images.append(img)

    if images:
        # Save as GIF
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=1000,
            loop=0,
        )
        logger.info(f"GIF created: {output_path}")
    else:
        logger.warning("No valid images found to create GIF")


def main():
    """Main function to generate global STEC maps using existing infrastructure."""
    import gc
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Safety checks for grid size
        n_lat = int(180 / args.lat_res) + 1
        n_lon = int(360 / args.lon_res) + 1
        total_points = n_lat * n_lon
        
        # Calculate time points and warn if many
        n_times = int(24 / args.time_res)
        
        if n_times > 48:
            logger.warning("⚠️  Many time points - this may take a long time!")

        # Load config
        config = load_config(args.config_path)
        
        # Set device with memory consideration
        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            gpu_memory_gb = gpu_props.total_memory / 1e9
            
            if total_points > 10000 and gpu_memory_gb < 8:
                logger.warning("⚠️  Large grid + limited GPU memory - using CPU")
                config["device"] = torch.device("cpu")
            else:
                config["device"] = torch.device("cuda")
        else:
            config["device"] = torch.device("cpu")

        # Initialize feature registry and add it to config
        feature_registry = initialize_feature_registry(config)
        config["feature_registry"] = feature_registry

        # Determine experiment name
        experiment_name = compute_exp_name(config)

        # Find experiment directory
        experiment_dir = find_experiment_directory(experiment_name)

        # Find model checkpoint
        checkpoint_path = find_model_checkpoint(experiment_dir)

        # Create BaseTrainer to handle all the infrastructure
        trainer = BaseTrainer(config, logger)

        # Load model using existing infrastructure
        model = get_model(config).to(config["device"])
        checkpoint = torch.load(
            checkpoint_path, map_location=config["device"], weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        # Create output directory with date subfolder
        output_dir = os.path.join(experiment_dir, "global_maps", args.date)
        os.makedirs(output_dir, exist_ok=True)

        # Create global grid
        lat_grid, lon_grid = create_global_grid(args.lat_res, args.lon_res)

        # Generate time points based on time resolution
        logger.info(f"Generating {args.date} maps with {args.time_res}h resolution")
        date_obj = datetime.strptime(args.date, "%Y-%m-%d")
        
        # Create time points (in hours from start of day)
        time_points = np.arange(0, 24, args.time_res)
        timestamps = [date_obj + timedelta(hours=float(t)) for t in time_points]

        # Create multi-temporal dataset (once for all timestamps)
        print("Initializing multi-temporal inference dataset...")
        import time
        init_start = time.time()
        multitemporal_dataset, dataloader = create_multitemporal_inference_dataloader(
            config, lat_grid, lon_grid, args.elevation, args.azimuth, date_obj, args.batch_size
        )
        grid_shape = multitemporal_dataset.get_grid_shape()
        init_time = time.time() - init_start
        print(f"Initialization completed in {init_time:.2f}s")

        # Storage for results
        image_paths = []  # For GIF creation
        all_tec_maps = []  # For IONEX export
        all_uncertainty_maps = []  # For IONEX RMS export
        all_epochs = []  # For IONEX timestamps

        for i, timestamp in enumerate(timestamps):
            hour_frac = timestamp.hour + timestamp.minute / 60.0
            print(f"Processing {timestamp.strftime('%H:%M')} UTC ({i+1}/{len(timestamps)})")

            # Update dataset for new timestamp (fast operation)
            update_start = time.time()
            multitemporal_dataset.update_timestamp(timestamp)
            update_time = time.time() - update_start
            if i == 1:  # Show timing for second iteration (first may include caching overhead)
                print(f"  → Timestamp update: {update_time:.3f}s")

            # Run inference using existing infrastructure (dataloader reuses updated dataset)
            metrics_dict, results_df = run_inference_with_trainer(trainer, dataloader, model)

            # Extract predictions and uncertainties from results
            # InferenceManager returns metrics dict and DataFrame
            predictions = metrics_dict["mean"].cpu().numpy()
            uncertainties = metrics_dict["total_std"].cpu().numpy() if "total_std" in metrics_dict else None

            # Reshape to grid
            stec_map = predictions.reshape(grid_shape)
            uncertainty_map = uncertainties.reshape(grid_shape) if uncertainties is not None else None

            # Store results for IONEX export
            all_tec_maps.append(stec_map)
            all_uncertainty_maps.append(uncertainty_map)
            all_epochs.append(timestamp)

            # Generate plots if requested
            if args.output_format in ["png", "both"]:
                # Save plot with time-aware filename
                time_str = timestamp.strftime("%H%M")
                base_filename = f"stec_map_{args.date}_{time_str}"
                base_path = os.path.join(output_dir, base_filename)
                save_hourly_plot(
                    lat_grid,
                    lon_grid,
                    stec_map,
                    uncertainty_map,
                    base_path,
                    args.date,
                    timestamp.hour + timestamp.minute / 60.0,
                    args.elevation,
                    args.azimuth,
                    vmin=args.vmin,
                    vmax=args.vmax,
                )

                # Store image path for GIF creation
                image_paths.append(f"{base_path}.png")

        # Create GIF if requested and images were generated
        if args.create_gif and image_paths:
            gif_path = os.path.join(output_dir, f"stec_daily_{args.date}.gif")
            create_gif(image_paths, gif_path)  # Adjust duration for time resolution

        # Generate IONEX file if requested
        if args.output_format in ["ionex", "both"]:
            print("Generating IONEX file...")
            
            # Create IONEX writer
            ionex_writer = IONEXWriter(
                center_code=args.ionex_center,
                program="PNN_STEC",
                version="1.0"
            )
            
            # Generate filename
            ionex_filename = generate_ionex_filename(date_obj, args.ionex_center)
            ionex_path = os.path.join(output_dir, ionex_filename)
            
            # Prepare RMS maps (uncertainties) if available
            rms_maps = all_uncertainty_maps if all_uncertainty_maps[0] is not None else None
            
            # Write IONEX file
            ionex_writer.write_ionex_file(
                output_path=ionex_path,
                epochs=all_epochs,
                lat_grid=lat_grid,
                lon_grid=lon_grid,
                tec_maps=all_tec_maps,
                rms_maps=rms_maps,
                interval_hours=args.time_res,
                description=f"STEC maps from PNN_STEC model - Elevation: {args.elevation}°, Azimuth: {args.azimuth}°",
                elevation=args.elevation,
                azimuth=args.azimuth
            )

        print(f"✅ Complete! Generated {len(timestamps)} maps -> {output_dir}")
        if args.create_gif and image_paths:
            print(f"📹 GIF: {gif_path}")
        if args.output_format in ["ionex", "both"]:
            print(f"📊 IONEX: {ionex_path}")
            
    except KeyboardInterrupt:
        print("⏹️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("For memory issues, try larger --lat_res/--lon_res or smaller --max_grid_points")
        raise
    finally:
        # Cleanup memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()