#!/usr/bin/env python3
"""
Model STEC Prediction Module - Leveraging Existing Infrastructure

This module integrates trained STEC models into the evaluation framework by 
reusing the proven inference infrastructure from inference_testset.py and inference_map.py.

Key Features:
- Uses experiment folder's stored config.yaml for proper model loading
- Leverages BaseTrainer and InferenceManager for robust inference
- Reuses existing data loading patterns for feature preparation
- Includes uncertainty quantification from Bayesian models
"""

import os
import sys
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent  # Go up from evaluation/ to src/ to project root
sys.path.append(str(project_root / "src"))

from utils.feature_registry import initialize_feature_registry, FeatureType
from model.model import get_model  
from training.base_trainer import BaseTrainer
from inference_map import find_experiment_directory, find_model_checkpoint, run_inference_with_trainer
from inference_map import find_experiment_directory, find_model_checkpoint, run_inference_with_trainer

logger = logging.getLogger(__name__)


class ModelSTECPredictor:
    """
    Loads trained STEC models using existing inference infrastructure.
    
    Reuses the proven patterns from inference_testset.py and inference_map.py.
    """
    
    def __init__(self):
        self.trainer = None
        self.model = None
        self.config = None
        self.experiment_dir = None
        self.device = None
        
    def load_experiment(self, experiment_folder: str) -> None:
        """Load a trained experiment using its stored configuration."""
        # Convert to absolute path if relative
        if not os.path.isabs(experiment_folder):
            experiment_folder = os.path.join(project_root, experiment_folder)
        
        self.experiment_dir = Path(experiment_folder)
        if not self.experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {self.experiment_dir}")
            
        logger.info(f"Loading experiment from: {self.experiment_dir.name}")
        
        # Load the experiment's stored config (same as inference_testset.py)
        config_path = self.experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        logger.info(f"Loaded config: {self.config['model']['model_type']}")
        
        # Auto-detect correct SH_degree from model input size (reuse logic from inference_map.py)
        self._detect_and_fix_sh_degree()
        
        # Set device (same as inference_testset.py)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config["device"] = self.device
        
        # Initialize feature registry (same as inference_testset.py)
        feature_registry = initialize_feature_registry(self.config)
        self.config["feature_registry"] = feature_registry
        
                # Find model checkpoint (reuse function from inference_map.py)
        checkpoint_path = find_model_checkpoint(str(self.experiment_dir))
        
        # Create BaseTrainer for inference infrastructure (same as inference_testset.py)
        self.trainer = BaseTrainer(self.config, logger)
        
        # Load model using existing patterns (same as inference_testset.py)
        self.model = get_model(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info("✓ Model and trainer loaded successfully")
        logger.info(f"  Model type: {self.config['model']['model_type']}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Total features: {feature_registry.get_total_features()}")
    
    def _detect_and_fix_sh_degree(self):
        """Detect and fix SH_degree from model input size (reuse inference_map.py logic)."""
        model_dir = self.experiment_dir / "model"
        if model_dir.exists():
            pth_files = [f for f in model_dir.iterdir() if f.suffix == ".pth"]
            if pth_files:
                # Load model checkpoint to check actual input size
                checkpoint_path = str(pth_files[0])
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                
                # Check first layer input size to determine correct SH_degree
                first_layer_key = None
                for key in checkpoint["model_state_dict"].keys():
                    if "layers.0.weight" in key or "fc_layers.0.weight" in key:
                        first_layer_key = key
                        break
                
                if first_layer_key:
                    input_size = checkpoint["model_state_dict"][first_layer_key].shape[1]
                    logger.info(f"Detected model input size: {input_size}")
                    
                    # Determine correct SH_degree based on input size
                    if input_size == 41:
                        # 41 features = 25 base + 16 SH (4 coords × 2² = 16)
                        self.config["data"]["SH_degree"] = 2
                        logger.info("Corrected SH_degree to 2 based on model input size")
                    elif input_size == 25:
                        # 25 features = no SH embeddings
                        self.config["data"]["SH_degree"] = 0
                        logger.info("Confirmed SH_degree = 0 based on model input size")
                    else:
                        logger.warning(f"Unexpected input size {input_size}, keeping config SH_degree")
                
    def predict_stec(self, 
                     times: List[datetime],
                     ipp_lat: np.ndarray,
                     ipp_lon: np.ndarray, 
                     elevations: np.ndarray,
                     azimuths: np.ndarray,
                     station_coords: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate STEC predictions using the existing inference infrastructure."""
        if self.trainer is None or self.model is None:
            raise ValueError("Model not loaded. Call load_experiment() first.")
            
        n_obs = len(times)
        logger.debug(f"Generating STEC predictions for {n_obs} observations")
        
        try:
            # Create evaluation dataset using the same pattern as MultiTemporalInferenceDataset
            dataset = STECEvaluationDataset(
                self.config, times, ipp_lat, ipp_lon, elevations, azimuths, station_coords
            )
            
            # Use the same dataloader pattern as inference_map.py
            from data_loader.collation import CollateWithSH
            collate_fn = CollateWithSH(self.config)
            
            # Create dataloader (same as inference patterns)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=min(n_obs, 1000), shuffle=False, num_workers=0,
                collate_fn=collate_fn
            )
            
            # Use existing run_inference_with_trainer function (same as inference_map.py)
            logger.debug("Running Bayesian inference with existing infrastructure")
            metrics_dict, results_df = run_inference_with_trainer(self.trainer, dataloader, self.model)
            
            # Extract predictions and uncertainties (same as inference_map.py)
            if results_df is not None and len(results_df) > 0:
                predictions = results_df["pred_stec"].values
                uncertainties = results_df["pred_total_unc"].values
                
                logger.debug(f"Generated {len(predictions)} predictions")
                logger.debug(f"STEC range: {np.min(predictions):.2f} to {np.max(predictions):.2f} TECU")
                logger.debug(f"Mean uncertainty: {np.mean(uncertainties):.2f} TECU")
                
                return predictions, uncertainties
            else:
                logger.warning("No results from inference manager")
                return np.full(n_obs, np.nan), np.full(n_obs, np.nan)
                
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            return np.full(n_obs, np.nan), np.full(n_obs, np.nan)


class STECEvaluationDataset(torch.utils.data.Dataset):
    """
    Dataset adapter for evaluation observations using existing infrastructure patterns.
    
    Follows the same pattern as MultiTemporalInferenceDataset but for scattered observations.
    """
    
    def __init__(self, config, times: List[datetime], ipp_lat: np.ndarray, ipp_lon: np.ndarray, 
                 elevations: np.ndarray, azimuths: np.ndarray, station_coords: Optional[np.ndarray] = None):
        self.config = config
        self.times = times
        self.n_obs = len(times)
        
        # Get feature registry (same as MultiTemporalInferenceDataset)
        self.feature_registry = config.get("feature_registry")
        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")
        
        # Get input features (same pattern as MultiTemporalInferenceDataset)
        all_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        self.input_features = [f for f in all_features if f not in target_features and f not in swi_features]
        
        # Store observation data
        self.ipp_lat = ipp_lat
        self.ipp_lon = ipp_lon
        self.elevations = elevations
        self.azimuths = azimuths
        self.station_coords = station_coords if station_coords is not None else np.array([[45.0, 8.0]] * self.n_obs)
        
        # Pre-compute solar magnetic coordinates (approximate with geographic for simplicity)
        self.sm_lat_sta = self.station_coords[:, 0]  # Simplified
        self.sm_lon_sta = self.station_coords[:, 1]  # Simplified
        self.sm_lat_ipp = self.ipp_lat  # Simplified
        self.sm_lon_ipp = self.ipp_lon  # Simplified
        
        # Pre-load SWI data (same pattern as MultiTemporalInferenceDataset)
        self.use_SWI = config["data"].get("use_SWI", False)
        if self.use_SWI:
            self._preload_swi_data()
        
        logger.debug(f"Evaluation dataset ready: {self.n_obs} observations")
    
    def _preload_swi_data(self):
        """Pre-load SWI data (simplified for evaluation - use defaults)."""
        self.swi_defaults = np.array([
            20.0,   # Kp_index
            100.0,  # R_Sunspot_No
            -20.0,  # Dst-index,_nT
            200.0,  # AE-index,_nT
            10.0,   # ap_index,_nT
            150.0   # f107_index
        ], dtype=np.float32)
        
    def __len__(self):
        return self.n_obs
        
    def __getitem__(self, idx):
        """Get feature vector for observation (same pattern as MultiTemporalInferenceDataset)."""
        feature_vector = []
        time = self.times[idx]
        
        # Build features in the same order as MultiTemporalInferenceDataset
        for feature_name in self.input_features:
            if feature_name == "year":
                value = float(time.year)
            elif feature_name == "doy":
                value = float(time.timetuple().tm_yday)
            elif feature_name == "sod":
                value = float(time.hour * 3600 + time.minute * 60 + time.second)
            elif feature_name == "sm_lat_sta":
                value = float(self.sm_lat_sta[idx])
            elif feature_name == "sm_lon_sta":
                value = float(self.sm_lon_sta[idx])
            elif feature_name == "lat_sta":
                value = float(self.station_coords[idx, 0])
            elif feature_name == "lon_sta":
                value = float(self.station_coords[idx, 1])
            elif feature_name == "lat_ipp":
                value = float(self.ipp_lat[idx])
            elif feature_name == "lon_ipp":
                value = float(self.ipp_lon[idx])
            elif feature_name == "sm_lat_ipp":
                value = float(self.sm_lat_ipp[idx])
            elif feature_name == "sm_lon_ipp":
                value = float(self.sm_lon_ipp[idx])
            elif feature_name == "satazi":
                value = float(self.azimuths[idx])
            elif feature_name == "satele":
                value = float(self.elevations[idx])
            else:
                raise ValueError(f"Feature {feature_name} not supported in evaluation dataset")
            
            feature_vector.append(value)
        
        feat = torch.tensor(feature_vector, dtype=torch.float32)
        
        # Add SWI features if enabled (same as MultiTemporalInferenceDataset)
        if self.use_SWI:
            swi_feat = torch.tensor(self.swi_defaults, dtype=torch.float32)
            feat = torch.cat((feat, swi_feat), dim=0)
        
        # Placeholder label (same as MultiTemporalInferenceDataset)
        label = torch.tensor(0.0, dtype=torch.float32)
        
        return feat, label


def build_model_stec(cfg: Dict[str, Any], obs: Dict[str, Any]) -> np.ndarray:
    """
    Main function to build model STEC predictions using existing inference infrastructure.
    
    Reuses patterns from inference_testset.py and inference_map.py for robustness.
    
    Args:
        cfg: Configuration dict with experiment_folder
        obs: Observations dict with times, coordinates, and geometry
        
    Returns:
        Array of model STEC predictions (TECU)
    """
    if not obs['times']:
        return np.array([])
        
    logger.info("Building model STEC predictions using existing infrastructure")
    
    # Get experiment folder from config
    experiment_folder = cfg.get('experiment_folder') or cfg.get('model_path')
    if not experiment_folder:
        logger.warning("No experiment_folder specified in config")
        return np.full(len(obs['times']), np.nan)
    
    # If model_path is provided instead of experiment_folder, extract the experiment folder
    if 'experiments/' in str(experiment_folder) and experiment_folder.endswith('.pth'):
        # Extract experiment folder from model path
        experiment_folder = str(Path(experiment_folder).parent.parent)
        logger.info(f"Extracted experiment folder from model path: {experiment_folder}")
    
    # Initialize predictor using existing infrastructure
    try:
        predictor = ModelSTECPredictor()
        predictor.load_experiment(experiment_folder)
    except Exception as e:
        logger.error(f"Failed to load experiment: {e}")
        return np.full(len(obs['times']), np.nan)
    
    # Prepare coordinates
    times = obs['times']
    ipp_lat = np.array(obs['ipp_lat'])
    ipp_lon = np.array(obs['ipp_lon'])
    elevations = np.array(obs['elevations'])
    
    # Handle azimuths (might not be available)
    if 'azimuths' in obs and obs['azimuths']:
        azimuths = np.array(obs['azimuths'])
    else:
        azimuths = np.full(len(times), 180.0)  # Default south direction
        logger.debug("No azimuths provided, using default values")
    
    # Handle station coordinates (might not be available)
    station_coords = None
    if 'stations' in obs and obs['stations']:
        # Use default coordinates for now
        station_coords = np.array([[45.0, 8.0]] * len(times))
        logger.debug("Using default station coordinates")
    
    # Generate predictions using existing infrastructure (same as inference_testset.py)
    try:
        predictions, uncertainties = predictor.predict_stec(
            times=times,
            ipp_lat=ipp_lat, 
            ipp_lon=ipp_lon,
            elevations=elevations,
            azimuths=azimuths,
            station_coords=station_coords
        )
        
        n_valid = np.sum(~np.isnan(predictions))
        total = len(predictions)
        logger.info(f"Model predictions: {n_valid}/{total} valid ({100*n_valid/total:.1f}%)")
        
        if n_valid > 0:
            logger.info(f"STEC range: {np.nanmin(predictions):.2f} to {np.nanmax(predictions):.2f} TECU")
            logger.info(f"Mean uncertainty: {np.nanmean(uncertainties):.2f} TECU")
        
        return predictions
        
    except Exception as e:
        logger.error(f"Failed to generate model predictions: {e}")
        return np.full(len(times), np.nan)