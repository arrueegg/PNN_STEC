#!/usr/bin/env python3
"""
Positioning Inference Script for PNN_STEC Project

This script generates STEC corrections for positioning applications by:
1. Loading observations from test set for specified date(s)
2. Filtering by test stations only (from test_station.list)
3. Running model inference to generate STEC predictions
4. Exporting results as CSV files (one per station per day)

CSV Output Format:
    - SOD: Seconds of day (UTC)
    - PRN: Satellite code (e.g., 'G01', 'R12')
    - ipp1: Placeholder (IPP latitude)
    - ipp2: Placeholder (IPP longitude)
    - STEC: Slant Total Electron Content prediction

Usage:
    python src/inference_positioning.py --experiment <exp_folder> --date 2024-07-01
    python src/inference_positioning.py --experiment <exp_folder> --start_date 2024-07-01 --end_date 2024-07-05

Output:
    - CSV files saved to: experiments/<experiment_name>/positioning/stec_corrections/YYYYDDD/<station>.csv
"""

import torch
import h5py
import numpy as np
import pandas as pd
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

_repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(_repo_root / "positioning"))

from utils.config_parser import load_config
from utils.feature_registry import initialize_feature_registry, FeatureType
from data_loader.collation import CollateWithSH
from torch.utils.data import Dataset, DataLoader
from evaluation.gim_mapper import MappingFunction  # Import for VTEC -> STEC conversion


def initialize_output_indices_for_registry(registry, config):
    """
    Initialize output indices for the feature registry.
    This mimics what CollateWithSH does but for inference without data loader.
    """
    output_indices = {}
    current_idx = 0

    # Year (normalized)
    temporal_features = registry.get_feature_names(FeatureType.TEMPORAL)
    for feature_name in temporal_features:
        if feature_name == "year":
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1
        elif feature_name in ["doy", "sod", "local_time_hours"]:
            # sin, cos, norm for cyclical features
            output_indices[f"{feature_name}_sin"] = current_idx
            output_indices[f"{feature_name}_cos"] = current_idx + 1
            output_indices[f"{feature_name}_norm"] = current_idx + 2
            current_idx += 3

    # Station features
    station_features = registry.get_feature_names(FeatureType.STATION)
    for feature_name in station_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1

    # Direction features - Cartesian unit vector
    direction_features = registry.get_feature_names(FeatureType.DIRECTION)
    if (
        direction_features
        and "satazi" in direction_features
        and "satele" in direction_features
    ):
        output_indices["e_up"] = current_idx
        output_indices["e_east"] = current_idx + 1
        output_indices["e_north"] = current_idx + 2
        current_idx += 3

    # IPP features
    ipp_features = registry.get_feature_names(FeatureType.IPP)
    for feature_name in ipp_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1

    # SH embeddings if enabled
    sh_degree = config.get("data", {}).get("SH_degree", 0)
    if sh_degree > 0:
        sh_dim = sh_degree * sh_degree
        has_station = len(station_features) > 0

        if has_station:
            output_indices["sh_sta_geo"] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
        else:
            output_indices["sh_sta_geo"] = None

        output_indices["sh_ipp_geo"] = slice(current_idx, current_idx + sh_dim)
        current_idx += sh_dim

        if has_station:
            output_indices["sh_sta_sm"] = slice(current_idx, current_idx + sh_dim)
            current_idx += sh_dim
        else:
            output_indices["sh_sta_sm"] = None

        output_indices["sh_ipp_sm"] = slice(current_idx, current_idx + sh_dim)
        current_idx += sh_dim
    else:
        output_indices["sh_sta_geo"] = None
        output_indices["sh_ipp_geo"] = None
        output_indices["sh_sta_sm"] = None
        output_indices["sh_ipp_sm"] = None

    # SWI features
    swi_features = registry.get_feature_names(FeatureType.SWI)
    for feature_name in swi_features:
        output_indices[f"{feature_name}_norm"] = current_idx
        current_idx += 1

    registry.set_output_indices(output_indices)
    return current_idx


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def load_test_stations(station_list_path="./src/data_processing/test_station.list"):
    """Load test station list."""
    stations = np.loadtxt(station_list_path, dtype=str)
    return set(stations)


def find_experiment_directory(experiment_name, base_dir="experiments"):
    """Find experiment directory by name or partial match."""
    # Strip leading "experiments/" or "experiment/" if present
    experiment_name = experiment_name.removeprefix("experiments/").removeprefix(
        "experiment/"
    )
    # Remove trailing slash if present
    experiment_name = experiment_name.rstrip("/")

    experiments_path = Path(base_dir)

    # First try exact match
    exact_path = experiments_path / experiment_name
    if exact_path.exists():
        return exact_path

    # Try partial match
    matching_dirs = [
        d
        for d in experiments_path.iterdir()
        if d.is_dir() and experiment_name in d.name
    ]

    if len(matching_dirs) == 1:
        return matching_dirs[0]
    elif len(matching_dirs) > 1:
        raise ValueError(
            f"Multiple experiments match '{experiment_name}': {[d.name for d in matching_dirs]}"
        )
    else:
        raise ValueError(f"No experiment found matching '{experiment_name}'")


def find_model_checkpoint(experiment_dir):
    """Find model checkpoint in experiment directory."""
    model_dir = experiment_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pth_files = list(model_dir.glob("*.pth"))

    if len(pth_files) == 0:
        raise FileNotFoundError(
            f"No model checkpoint (.pth) files found in {model_dir}"
        )
    elif len(pth_files) == 1:
        return pth_files[0]
    else:
        # Prefer pretrain checkpoint
        pretrain_files = [f for f in pth_files if "pretrain" in f.name.lower()]
        return pretrain_files[0] if pretrain_files else pth_files[0]


def load_experiment_config(experiment_dir):
    """Load configuration from experiment directory."""
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(str(config_path))
    return config


class PositioningDataset(Dataset):
    """Dataset for loading specific day observations from test set."""

    def __init__(
        self, config, gnss_data_path, year, doy, test_stations, feature_registry
    ):
        """
        Args:
            config: Configuration dictionary
            gnss_data_path: Path to GNSS data folder (STEC_DB_CASDCB)
            year: Year to filter (int)
            doy: Day of year to filter (int)
            test_stations: Set of test station names
            feature_registry: Feature registry for ordering
        """
        self.config = config
        self.feature_registry = feature_registry
        self.year = year
        self.doy = doy

        # Get feature configuration
        from utils.feature_registry import FeatureType

        all_features = feature_registry.get_all_enabled_features()
        self.target_feature = feature_registry.get_features_by_type(FeatureType.TARGET)[
            0
        ]
        self.input_features = [f for f in all_features if f != self.target_feature]
        self.swi_features = (
            feature_registry.get_features_by_type(FeatureType.SWI)
            if feature_registry
            else []
        )

        # Load and filter data from daily CCL file
        logger = logging.getLogger(__name__)
        logger.info(f"Loading data for year={year}, doy={doy:03d}")

        # Construct path to daily CCL file
        ccl_filename = f"ccl_{year}{doy:03d}_30_5.h5"
        ccl_path = os.path.join(gnss_data_path, str(year), f"{doy:03d}", ccl_filename)

        if not os.path.exists(ccl_path):
            raise FileNotFoundError(f"CCL file not found: {ccl_path}")

        logger.info(f"Loading from: {ccl_path}")

        with h5py.File(ccl_path, "r") as f:
            # Load all data and test indices
            all_data = f[str(year)][f"{doy:03d}"]["all_data"][:]
            test_idx = f[str(year)][f"{doy:03d}"]["test_idx"][:]

            # Get test data
            test_data = all_data[test_idx]

            # Filter by test stations
            station_mask = np.isin(
                test_data["station"], [s.encode("ascii") for s in test_stations]
            )

            self.data = test_data[station_mask]
            logger.info(
                f"Loaded {len(self.data):,} observations from {len(np.unique(self.data['station']))} stations"
            )

        # Setup SWI if needed
        self.use_SWI = config["data"].get("use_SWI", False)
        if self.use_SWI:
            swi_path = os.path.join(
                config["data"]["SWI_data_path"], "omni_hourly_2010-2025.h5"
            )
            self.swi_file = h5py.File(swi_path, "r")

            # Build SWI column mapping (same as H5Dataset)
            yrs = list(self.swi_file.keys())
            days = list(self.swi_file[yrs[0]].keys())
            cols = [c.decode() for c in self.swi_file[yrs[0]][days[0]].attrs["columns"]]
            self.swi_col_names = cols
            self.swi_mask = [c not in ("YEAR", "DOY", "HR") for c in cols]

            masked_names = [n for n, m in zip(cols, self.swi_mask) if m]
            self.swi_name_to_idx = {name: i for i, name in enumerate(masked_names)}
            self.swi_indices_in_file_order = [
                self.swi_name_to_idx.get(f, None) for f in self.swi_features
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get features for one observation."""
        row = self.data[idx]

        # Build feature vector (same logic as H5Dataset)
        feature_vector = []

        # Non-SWI features
        for feature_name in self.input_features:
            if feature_name not in self.swi_features:
                if feature_name == "year":
                    value = float(self.year)  # Use year from constructor
                elif feature_name == "doy":
                    value = float(self.doy)  # Use doy from constructor
                elif feature_name == "sod":
                    value = float(row["sod"])
                elif feature_name == "local_time_hours":
                    sod = float(row["sod"])
                    longitude = float(row["lon_ipp"])
                    value = self._compute_local_time_hours(sod, longitude)
                elif feature_name in ["lat_sta", "lon_sta", "sm_lat_sta", "sm_lon_sta"]:
                    value = float(row[feature_name])
                elif feature_name in ["satazi", "satele"]:
                    value = float(row[feature_name])
                elif feature_name in ["lat_ipp", "lon_ipp", "sm_lat_ipp", "sm_lon_ipp"]:
                    value = float(row[feature_name])
                else:
                    raise ValueError(f"Feature {feature_name} not found")
                feature_vector.append(value)

        # SWI features
        if self.use_SWI and self.swi_features:
            for feature_name in self.swi_features:
                year_str = str(self.year)
                doy3 = f"{self.doy:03d}"
                hour = int(row["sod"] // 3600)
                swi_row = self.swi_file[year_str][doy3][hour]
                swi_values_masked = swi_row[self.swi_mask]

                swi_pos = self.swi_features.index(feature_name)
                in_idx = self.swi_indices_in_file_order[swi_pos]
                value = 0.0 if in_idx is None else float(swi_values_masked[in_idx])
                feature_vector.append(value)

        feat = torch.tensor(feature_vector, dtype=torch.float32)
        label = torch.tensor(row[self.target_feature], dtype=torch.float32)

        # Return metadata for CSV export
        metadata = {
            "station": row["station"].decode("utf-8"),
            "sat": row["sat"].decode("utf-8"),
            "sod": float(row["sod"]),
            "lat_ipp": float(row["lat_ipp"]),
            "lon_ipp": float(row["lon_ipp"]),
            "satele": float(row["satele"])
            if "satele" in row.dtype.names
            else 0.0,  # Capture elevation for VTEC mapping
        }

        return feat, label, metadata

    @staticmethod
    def _compute_local_time_hours(sod, longitude):
        """Compute local time in hours from UTC seconds of day and longitude."""
        utc_hours = sod / 3600.0
        longitude_offset = longitude / 15.0
        local_time_hours = (utc_hours + longitude_offset) % 24.0
        return local_time_hours

    def __del__(self):
        if hasattr(self, "swi_file") and self.use_SWI:
            self.swi_file.close()


def run_inference_for_day(
    config, model, feature_registry, year, doy, test_stations, logger
):
    """
    Run inference for all test station observations on a specific day.

    Returns:
        DataFrame with predictions, uncertainties, and metadata
    """
    # Create dataset for this day
    gnss_data_path = config["data"]["GNSS_data_path"]
    dataset = PositioningDataset(
        config, gnss_data_path, year, doy, test_stations, feature_registry
    )

    if len(dataset) == 0:
        logger.warning(f"No observations found for {year}-{doy:03d}")
        return None

    # Create dataloader with collation
    collate_fn = CollateWithSH(config)
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Run inference
    model.eval()
    device = config["device"]

    predictions = []
    uncertainties = []
    metadata_list = []

    logger.info(f"Running inference on {len(dataset):,} observations...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            inputs, labels, batch_metadata = batch
            inputs = inputs.to(device)

            # Model forward pass
            outputs = model(inputs)

            # Extract prediction and uncertainty (handle different model output formats)
            if isinstance(outputs, tuple):
                # Extract mean and the uncertainty parameter
                pred_stec = outputs[0]
                uncertainty_param = outputs[1]

                # Check model type for uncertainty parameterization
                # DeepEnsemble always returns variance (sigma^2) regardless of base distribution
                # Single MLP_LaplacianNLL returns scale (b), while BNN models return variance (sigma^2)
                from model.model import DeepEnsemble

                if isinstance(model, DeepEnsemble):
                    # Ensemble returns variance
                    pred_uncertainty = torch.sqrt(uncertainty_param)
                else:
                    # Single model
                    model_type = config.get("model", {}).get("model_type", "")
                    if "Laplacian" in model_type:
                        # For Laplacian distribution, standard deviation = sqrt(2) * scale (b)
                        # We report the standard deviation to maintain consistency across models
                        pred_uncertainty = uncertainty_param * 1.41421356237
                    else:
                        # Standard BNN/Gaussian models return (mean, variance)
                        pred_uncertainty = torch.sqrt(
                            uncertainty_param
                        )  # Convert variance to std deviation
            else:
                # Deterministic models (MLP) - zero uncertainty
                pred_stec = outputs
                pred_uncertainty = torch.zeros_like(pred_stec)

            predictions.append(pred_stec.cpu())
            uncertainties.append(pred_uncertainty.cpu())
            metadata_list.extend(batch_metadata)

    # Combine results
    all_predictions = torch.cat(predictions).numpy().flatten()
    all_uncertainties = torch.cat(uncertainties).numpy().flatten()

    # Create results DataFrame
    results_df = pd.DataFrame(metadata_list)
    results_df["pred_stec"] = all_predictions
    results_df["uncertainty"] = all_uncertainties

    # Check if we need to map VTEC to STEC
    if config.get("target", "").lower() == "vtec":
        logger.info(
            "ℹ️  Detected VTEC model target - applying mapping function to STEC..."
        )

        # Initialize mapper (defaults to MSLM)
        # Note: mapping_function arg is not passed, assuming default MSLM which is standard
        mapper = MappingFunction(mapping_type="MSLM")

        # Calculate mapping factor
        # satele is in degrees, convert to radians
        if "satele" not in results_df.columns:
            logger.warning(
                "⚠️  'satele' column missing for mapping function! Using M=1 (Vertical). Errors likely."
            )
            mapping_factors = 1.0
        else:
            elev_rad = np.radians(results_df["satele"].values)
            mapping_factors = mapper.get_mapping_factor(elev_rad)

        # Update predictions: STEC = VTEC * M(z)
        # Uncertainty also scales: sigma_stec = sigma_vtec * M(z) (approx)
        results_df["pred_stec"] = results_df["pred_stec"] * mapping_factors
        results_df["uncertainty"] = results_df["uncertainty"] * mapping_factors

        logger.info("✅ VTEC -> STEC conversion complete")

    logger.info(
        f"Completed inference: {len(results_df):,} predictions with uncertainties"
    )

    return results_df


def export_station_csv(station_df, output_path, year, doy, station_name):
    """
    Export predictions for one station to CSV.

    CSV columns: second_of_day, PRN, ipp_latitude, ipp_longitude, stec, uncertainty
    """
    # Prepare export dataframe
    export_df = pd.DataFrame(
        {
            "second_of_day": station_df["sod"].values,
            "PRN": station_df["sat"].values,
            "ipp_latitude": station_df["lat_ipp"].values,
            "ipp_longitude": station_df["lon_ipp"].values,
            "stec": station_df["pred_stec"].values,
            "uncertainty": station_df["uncertainty"].values,
        }
    )

    # Sort by second_of_day and PRN for consistency
    export_df = export_df.sort_values(["second_of_day", "PRN"])

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False, float_format="%.4f")


def process_date(args, config, model, feature_registry, test_stations, logger):
    """Process a single date."""
    year = args.year
    doy = args.doy

    # Create output directory
    experiment_dir = Path(args.experiment_dir)
    date_str = f"{year}{doy:03d}"
    output_dir = experiment_dir / "positioning" / "stec_corrections" / date_str

    logger.info("=" * 80)
    logger.info(f"Processing date: {year}-{doy:03d}")
    logger.info("=" * 80)

    # Run inference
    results_df = run_inference_for_day(
        config, model, feature_registry, year, doy, test_stations, logger
    )

    if results_df is None or len(results_df) == 0:
        logger.warning(f"No data for {year}-{doy:03d}")
        return

    # Group by station and export
    stations_in_data = results_df["station"].unique()
    logger.info(f"Exporting CSV files for {len(stations_in_data)} stations...")

    for station in tqdm(stations_in_data, desc="Exporting CSVs"):
        station_df = results_df[results_df["station"] == station]
        output_path = output_dir / f"{station}.csv"
        export_station_csv(station_df, output_path, year, doy, station)

    logger.info(f"✅ Exported {len(stations_in_data)} CSV files to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate STEC corrections for positioning applications"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Experiment folder name (full or partial match)",
    )
    parser.add_argument("--date", type=str, help="Single date to process (YYYY-MM-DD)")
    parser.add_argument(
        "--start_date", type=str, help="Start date for range (YYYY-MM-DD)"
    )
    parser.add_argument("--end_date", type=str, help="End date for range (YYYY-MM-DD)")
    parser.add_argument("--year", type=int, help="Year (alternative to --date)")
    parser.add_argument("--doy", type=int, help="Day of year (alternative to --date)")
    parser.add_argument(
        "--gnss_path", type=str, help="Override GNSS_data_path from config"
    )

    args = parser.parse_args()
    logger = setup_logging()

    try:
        # Validate date arguments
        if args.date:
            date_obj = datetime.strptime(args.date, "%Y-%m-%d")
            args.year = date_obj.year
            args.doy = date_obj.timetuple().tm_yday
        elif args.year and args.doy:
            pass  # Already set
        elif args.start_date and args.end_date:
            # Date range logic
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

            logger.info("🚀 POSITIONING STEC INFERENCE (Date Range)")
            logger.info("=" * 80)

            # Find and load experiment
            experiment_dir = find_experiment_directory(args.experiment)
            args.experiment_dir = experiment_dir
            logger.info(f"📂 Experiment: {experiment_dir.name}")

            # Load configuration
            config = load_experiment_config(experiment_dir)
            config["device"] = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"💻 Device: {config['device']}")

            # [FIX] Path consistency for local machine vs cluster
            if args.gnss_path:
                config["data"]["GNSS_data_path"] = args.gnss_path
                logger.info(f"📍 Overriding GNSS path: {args.gnss_path}")
            else:
                orig_path = config["data"].get("GNSS_data_path", "")
                if orig_path and not os.path.exists(orig_path):
                    # Try local alternative
                    alt_path = "/home/space/data/iono/STEC_DB_CASDCB"
                    if os.path.exists(alt_path):
                        config["data"]["GNSS_data_path"] = alt_path
                        logger.warning(f"⚠️ GNSS path not found at {orig_path}")
                        logger.info(f"📍 Using local alternative: {alt_path}")

            # Initialize feature registry
            feature_registry = initialize_feature_registry(config)
            config["feature_registry"] = feature_registry

            # Initialize output indices for feature splitter
            total_features = initialize_output_indices_for_registry(
                feature_registry, config
            )
            # logger.info(f"📊 Total features: {total_features}")

            # Load test stations
            test_stations = load_test_stations()
            logger.info(f"📍 Test stations: {len(test_stations)}")

            # Load model (handles single or ensemble)
            from model.model import load_model_for_inference

            model = load_model_for_inference(config, experiment_dir, logger)

            # Iterate through date range
            current_date = start_date
            while current_date <= end_date:
                args.year = current_date.year
                args.doy = current_date.timetuple().tm_yday

                logger.info(
                    f"\n📅 Processing {current_date.strftime('%Y-%m-%d')} ({args.year}-{args.doy:03d})..."
                )
                process_date(
                    args, config, model, feature_registry, test_stations, logger
                )

                current_date += timedelta(days=1)

            logger.info("")
            logger.info("✅ POSITIONING INFERENCE COMPLETED FOR RANGE!")
            return 0

        else:
            logger.error(
                "Must specify either --date or --year/--doy or --start_date/--end_date"
            )
            return 1

        logger.info("🚀 POSITIONING STEC INFERENCE")
        logger.info("=" * 80)

        # Find and load experiment
        experiment_dir = find_experiment_directory(args.experiment)
        args.experiment_dir = experiment_dir
        logger.info(f"📂 Experiment: {experiment_dir.name}")

        # Load configuration
        config = load_experiment_config(experiment_dir)
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"💻 Device: {config['device']}")

        # [FIX] Path consistency for local machine vs cluster
        if args.gnss_path:
            config["data"]["GNSS_data_path"] = args.gnss_path
            logger.info(f"📍 Overriding GNSS path: {args.gnss_path}")
        else:
            orig_path = config["data"].get("GNSS_data_path", "")
            if orig_path and not os.path.exists(orig_path):
                # Try local alternative
                alt_path = "/home/space/data/iono/STEC_DB_CASDCB"
                if os.path.exists(alt_path):
                    config["data"]["GNSS_data_path"] = alt_path
                    logger.warning(f"⚠️ GNSS path not found at {orig_path}")
                    logger.info(f"📍 Using local alternative: {alt_path}")

        # Initialize feature registry
        feature_registry = initialize_feature_registry(config)
        config["feature_registry"] = feature_registry

        # Initialize output indices for feature splitter
        total_features = initialize_output_indices_for_registry(
            feature_registry, config
        )
        logger.info(f"📊 Total features: {total_features}")

        # Load test stations
        test_stations = load_test_stations()
        logger.info(f"📍 Test stations: {len(test_stations)}")

        # Load model (handles single or ensemble)
        from model.model import load_model_for_inference

        model = load_model_for_inference(config, experiment_dir, logger)

        # Process date(s)
        process_date(args, config, model, feature_registry, test_stations, logger)

        logger.info("")
        logger.info("✅ POSITIONING INFERENCE COMPLETED!")

        return 0

    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
