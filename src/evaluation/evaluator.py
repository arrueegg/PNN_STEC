#!/usr/bin/env python3
"""
STEC Evaluation Framework

Memory-efficient STEC evaluation using DataLoader processing.
Groups batches by date for efficient GIM loading (one IONEX file per day).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Iterator, Tuple, Optional, List
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent  
sys.path.append(str(project_root / "src"))

from data_loader import get_test_data_loader
from training.base_trainer import BaseTrainer
from model.model import get_model
from utils.feature_registry import initialize_feature_registry
from .gim_mapper import GIMMapper
from inference_map import find_experiment_directory, find_model_checkpoint

logger = logging.getLogger(__name__)


class EvaluationOrchestrator:
    """
    Memory-efficient STEC evaluation using DataLoader processing.
    
    This orchestrator processes test data efficiently using existing infrastructure,
    avoiding the need to load entire datasets into memory.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config.get('output_dir', './eval_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model predictor using existing infrastructure
        self.trainer = None
        self.model = None
        self._setup_model()
        
        # Initialize GIM processor
        self.gim_processor = GIMProcessor(config)
        
        # Initialize results accumulator
        self.accumulator = ComparisonAccumulator()
        
        # Initialize results streamer for large datasets
        self.streamer = ResultsStreamer(self.output_dir, config)
        
    def _setup_model(self):
        """Setup model using existing infrastructure patterns."""
        experiment_folder = self.config.get('experiment_folder')
        if not experiment_folder:
            raise ValueError("experiment_folder required in config")
            
        # Load experiment config (same as inference_testset.py)
        experiment_dir = Path(experiment_folder)
        if not experiment_dir.is_absolute():
            experiment_dir = project_root / experiment_folder
            
        config_path = experiment_dir / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        import yaml
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
            
        # Merge configs (evaluation config takes precedence)
        model_config.update(self.config)  # This ensures device and other eval settings are preserved
        
        # Ensure device is set
        if 'device' not in model_config:
            model_config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Initialize feature registry
        feature_registry = initialize_feature_registry(model_config)
        model_config["feature_registry"] = feature_registry
        
        # Create trainer using existing infrastructure
        self.trainer = BaseTrainer(model_config, logger)
        self.model_config = model_config  # Store for later use
        
        # Load model
        self.model = get_model(model_config).to(model_config["device"])
        checkpoint_path = find_model_checkpoint(str(experiment_dir))
        checkpoint = torch.load(checkpoint_path, map_location=model_config["device"], weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        logger.info(f"✅ Model loaded: {model_config['model']['model_type']}")
        
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run batch-based evaluation using existing DataLoader infrastructure.
        
        This leverages get_test_data_loader() for memory-efficient processing,
        groups batches by date for efficient GIM loading, and accumulates results.
        """
        logger.info("🚀 Starting STEC evaluation")
        
        # Get test dataloader using existing infrastructure (same as inference_testset.py)
        test_loader = get_test_data_loader(self.model_config, logger)
        
        logger.info(f"📊 Processing {len(test_loader.dataset):,} samples in {len(test_loader)} batches")
        
        # First pass: collect all batches and group by date for efficient GIM processing
        logger.info("📅 Collecting and grouping batches by date...")
        date_groups = {}  # date_str -> list of (batch_idx, features, targets)
        
        for batch_idx, (features, targets) in enumerate(tqdm(test_loader, desc="Collecting batches")):
            # Extract observation data to determine date
            batch_obs = self._extract_observation_data(features, self.model_config["feature_registry"])
            
            # Group by date (assuming all observations in a batch are from the same day)
            if batch_obs['times']:
                batch_date = batch_obs['times'][0].strftime('%Y-%m-%d')
                if batch_date not in date_groups:
                    date_groups[batch_date] = []
                date_groups[batch_date].append((batch_idx, features, targets, batch_obs))
        
        logger.info(f"📅 Found {len(date_groups)} unique dates in test data")
        
        # Second pass: process each date group efficiently
        for date_str, date_batches in tqdm(date_groups.items(), desc="Processing by date"):
            logger.info(f"🌍 Processing {len(date_batches)} batches for date: {date_str}")
            
            # Process all batches for this date (GIM data loaded once per date)
            for batch_idx, features, targets, batch_obs in date_batches:
                
                # Generate model predictions using existing infrastructure
                model_results = self._predict_model_batch(features, targets)
                
                # Generate GIM predictions for the same observations (efficient daily loading)
                gim_results = self._predict_gim_batch_with_obs(batch_obs)
                
                # Compare and accumulate results
                batch_comparison = self._compare_batch(model_results, gim_results, targets)
                self.accumulator.add_batch(batch_comparison)
                
                # Stream results for large datasets
                if len(test_loader.dataset) > 100_000:
                    self.streamer.write_batch(batch_comparison, batch_idx)
                    
            # Memory cleanup after each date
            torch.cuda.empty_cache()
                
        # Finalize evaluation
        final_metrics = self.accumulator.finalize()
        self.streamer.finalize()
        
        # Save comprehensive report
        self._save_evaluation_report(final_metrics)
        
        logger.info("✅ STEC evaluation completed")
        return final_metrics
        
    def _predict_model_batch(self, features: torch.Tensor, targets: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate model predictions for a batch using existing infrastructure."""
        # Create mini-dataloader for this batch (reuse existing patterns)
        from torch.utils.data import TensorDataset, DataLoader
        
        batch_dataset = TensorDataset(features, targets)
        batch_loader = DataLoader(batch_dataset, batch_size=len(features), shuffle=False)
        
        # Use existing bayesian inference infrastructure
        # This handles uncertainty quantification and all model types properly
        with torch.no_grad():
            metrics_dict, results_df = self.trainer.inference_manager.bayesian_inference_total_uncertainty(
                self.model, batch_loader, num_samples=50  # Reduced for efficiency
            )
            
        # Extract predictions and uncertainties
        model_stec = results_df["pred_stec"].values
        model_uncertainty = results_df["pred_total_unc"].values
        
        return {
            'stec': model_stec,
            'uncertainty': model_uncertainty,
            'n_samples': len(model_stec)
        }
        
    def _predict_gim_batch(self, features: torch.Tensor, targets: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate GIM predictions for a batch."""
        # Extract coordinate and time information from features
        # This requires understanding the feature order from the registry
        feature_registry = self.model_config["feature_registry"]
        
        # Map features to observation format for GIM processing
        batch_obs = self._extract_observation_data(features, feature_registry)
        
        # Process through GIM mapper
        gim_stec = self.gim_processor.process_batch(batch_obs)
        
        return {
            'stec': gim_stec,
            'n_samples': len(gim_stec)
        }
        
    def _predict_gim_batch_with_obs(self, batch_obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Generate GIM predictions for a batch using pre-extracted observation data."""
        # Process through GIM mapper (this will use daily loading)
        gim_stec = self.gim_processor.process_batch(batch_obs)
        
        return {
            'stec': gim_stec,
            'n_samples': len(gim_stec)
        }
        
    def _extract_observation_data(self, features: torch.Tensor, feature_registry) -> Dict[str, Any]:
        """Extract observation data from feature tensor for GIM processing using proper denormalization."""
        from datetime import datetime, timedelta
        
        # Get output indices from the feature registry (same as validation_manager.py)
        output_indices = feature_registry._output_indices
        
        logger.debug(f"Extracting observations from tensor shape {features.shape}")
        
        # Extract normalized values and denormalize using feature registry
        feature_map = {}
        for feature_name in ['lat_ipp', 'lon_ipp', 'satele']:
            norm_key = f"{feature_name}_norm"
            if norm_key in output_indices:
                feature_idx = output_indices[norm_key]
                normalized_values = features[:, feature_idx]
                # Use feature registry denormalization
                denormalized_values = feature_registry.denormalize_feature(feature_name, normalized_values)
                feature_map[feature_name] = denormalized_values.cpu().numpy()
            else:
                raise ValueError(f"Feature {norm_key} not found in output_indices")
        
        batch_size = features.shape[0]
        
        # Create placeholder times (this still needs proper time extraction)
        base_time = datetime(2023, 6, 15, 12, 0, 0)
        times = [base_time + timedelta(minutes=i*5) for i in range(batch_size)]
        
        # Extract properly denormalized coordinates
        ipp_lat = feature_map['lat_ipp'].tolist()
        ipp_lon = feature_map['lon_ipp'].tolist() 
        elevations = feature_map['satele'].tolist()
        
        logger.debug(f"Extracted {len(times)} observations")
        logger.debug(f"Lat range: {min(ipp_lat):.2f} to {max(ipp_lat):.2f}")
        logger.debug(f"Lon range: {min(ipp_lon):.2f} to {max(ipp_lon):.2f}")
        logger.debug(f"Elevation range: {min(elevations):.2f} to {max(elevations):.2f}")
        
        logger.warning("Using placeholder times - need proper time extraction from features!")
        
        return {
            'times': times,
            'ipp_lat': ipp_lat,
            'ipp_lon': ipp_lon,
            'elevations': elevations,
        }
        
    def _compare_batch(self, model_results: Dict, gim_results: Dict, targets: torch.Tensor) -> Dict[str, Any]:
        """Compare model and GIM results for a batch."""
        targets_np = targets.cpu().numpy().flatten()
        model_stec = model_results['stec']
        gim_stec = gim_results['stec']
        
        # Create batch comparison data
        batch_data = {
            'target_stec': targets_np,
            'model_stec': model_stec,
            'gim_stec': gim_stec,
            'model_uncertainty': model_results.get('uncertainty', np.zeros_like(model_stec)),
            'model_vs_gim_diff': model_stec - gim_stec,
            'model_vs_truth_diff': model_stec - targets_np,
            'gim_vs_truth_diff': gim_stec - targets_np,
        }
        
        return batch_data
        
    def _save_evaluation_report(self, metrics: Dict[str, Any]):
        """Save comprehensive evaluation report."""
        report_path = self.output_dir / "evaluation_report.yaml"
        
        import yaml
        with open(report_path, 'w') as f:
            yaml.dump({
                'evaluation_config': self.config,
                'final_metrics': metrics,
                'model_info': {
                    'type': self.trainer.config['model']['model_type'],
                    'experiment': self.config.get('experiment_folder')
                }
            }, f, default_flow_style=False, indent=2)
            
        logger.info(f"📋 Saved evaluation report: {report_path}")
        

class GIMProcessor:
    """Processes GIM data efficiently by loading one IONEX file per day."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Initialize GIMMapper with error handling
        try:
            from .gim_mapper import GIMMapper
            self.gim_mapper = GIMMapper(config)
            self.current_date = None  # Track currently loaded date
            self.current_gim_data = {}  # Store current day's GIM data
        except Exception as e:
            logger.warning(f"Failed to initialize GIM mapper: {e}")
            self.gim_mapper = None
        
    def load_daily_gim_data(self, target_date: datetime) -> bool:
        """Load GIM data for a specific date (one IONEX file)."""
        if self.gim_mapper is None:
            return False
            
        # Check if we already have data for this date
        target_date_str = target_date.strftime('%Y-%m-%d')
        if self.current_date == target_date_str:
            return True  # Already loaded
            
        try:
            # Clear previous data to save memory
            self.current_gim_data = {}
            
            # Find IONEX file for this specific date
            gim_path = Path(self.config.get('gim_path', '/default/gim/path'))
            if not gim_path.exists():
                logger.warning(f"GIM path not found: {gim_path}")
                return False
                
            # Find files for just this date
            ionex_files = self.gim_mapper._find_ionex_files(
                gim_path, 
                target_date, 
                target_date + timedelta(days=1)
            )
            
            if not ionex_files:
                logger.warning(f"No IONEX file found for date: {target_date_str}")
                return False
                
            # Load only the first file for this date (should be just one)
            ionex_file = ionex_files[0]
            logger.debug(f"Loading IONEX file for {target_date_str}: {ionex_file.name}")
            
            # Read the IONEX file directly
            file_data = self.gim_mapper.reader.read_ionex_file(ionex_file)
            
            # Store the data in the mapper's format
            self.gim_mapper.gim_data = {
                'epochs': file_data['epochs'],
                'vtec_maps': file_data['vtec_maps'],
                'lat_grid': file_data['lat_grid'],
                'lon_grid': file_data['lon_grid']
            }
            
            self.current_date = target_date_str
            logger.debug(f"✅ Loaded {len(file_data['epochs'])} VTEC maps for {target_date_str}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load GIM data for {target_date_str}: {e}")
            return False
        
    def process_batch(self, batch_obs: Dict[str, Any]) -> np.ndarray:
        """Process observations through GIM mapping with daily data loading."""
        if self.gim_mapper is None:
            logger.warning("GIM mapper not available, returning NaN values")
            return np.full(len(batch_obs['times']), np.nan)
            
        try:
            times = np.array(batch_obs['times'])
            if times.size == 0:
                return np.array([])
                
            # Get the date for this batch (should all be the same day when grouped)
            batch_date = times[0].replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Load GIM data for this specific date only
            if not self.load_daily_gim_data(batch_date):
                logger.warning(f"Failed to load GIM data for {batch_date.strftime('%Y-%m-%d')}")
                return np.full(len(times), np.nan)
            
            # Now process the batch with the loaded daily data
            return self.gim_mapper.map_vtec_to_stec(
                times=times,
                ipp_lat=np.array(batch_obs['ipp_lat']),
                ipp_lon=np.array(batch_obs['ipp_lon']),
                elevations=np.array(batch_obs['elevations'])
            )
            
        except Exception as e:
            logger.warning(f"GIM processing failed for batch: {e}")
            return np.full(len(batch_obs['times']), np.nan)


class ComparisonAccumulator:
    """Accumulates comparison metrics across batches efficiently."""
    
    def __init__(self):
        self.n_total = 0
        self.n_valid = 0
        self.sum_model_truth_diff2 = 0.0
        self.sum_gim_truth_diff2 = 0.0
        self.sum_model_gim_diff2 = 0.0
        self.sum_model_truth_diff = 0.0
        self.sum_gim_truth_diff = 0.0
        self.sum_model_gim_diff = 0.0
        self.batch_summaries = []
        
    def add_batch(self, batch_data: Dict[str, Any]):
        """Add a batch of comparison data."""
        # Filter valid data
        model_stec = batch_data['model_stec']
        gim_stec = batch_data['gim_stec']
        target_stec = batch_data['target_stec']
        
        valid_mask = ~(np.isnan(model_stec) | np.isnan(gim_stec) | np.isnan(target_stec))
        n_valid_batch = np.sum(valid_mask)
        
        if n_valid_batch > 0:
            # Update counters
            self.n_total += len(model_stec)
            self.n_valid += n_valid_batch
            
            # Update running sums for efficient computation
            model_truth_diff = batch_data['model_vs_truth_diff'][valid_mask]
            gim_truth_diff = batch_data['gim_vs_truth_diff'][valid_mask]
            model_gim_diff = batch_data['model_vs_gim_diff'][valid_mask]
            
            self.sum_model_truth_diff += np.sum(model_truth_diff)
            self.sum_gim_truth_diff += np.sum(gim_truth_diff)
            self.sum_model_gim_diff += np.sum(model_gim_diff)
            
            self.sum_model_truth_diff2 += np.sum(model_truth_diff**2)
            self.sum_gim_truth_diff2 += np.sum(gim_truth_diff**2)
            self.sum_model_gim_diff2 += np.sum(model_gim_diff**2)
            
            # Store batch summary for detailed analysis
            self.batch_summaries.append({
                'n_samples': len(model_stec),
                'n_valid': n_valid_batch,
                'model_mae': np.mean(np.abs(model_truth_diff)),
                'gim_mae': np.mean(np.abs(gim_truth_diff)),
                'model_gim_mae': np.mean(np.abs(model_gim_diff))
            })
            
    def finalize(self) -> Dict[str, Any]:
        """Compute final metrics from accumulated statistics."""
        if self.n_valid == 0:
            return {'error': 'No valid predictions across all batches'}
            
        # Compute final metrics
        model_bias = self.sum_model_truth_diff / self.n_valid
        gim_bias = self.sum_gim_truth_diff / self.n_valid
        model_gim_bias = self.sum_model_gim_diff / self.n_valid
        
        model_rmse = np.sqrt(self.sum_model_truth_diff2 / self.n_valid)
        gim_rmse = np.sqrt(self.sum_gim_truth_diff2 / self.n_valid)
        model_gim_rmse = np.sqrt(self.sum_model_gim_diff2 / self.n_valid)
        
        return {
            'n_total': self.n_total,
            'n_valid': self.n_valid,
            'validity_rate': self.n_valid / self.n_total,
            'model_vs_truth': {
                'bias': model_bias,
                'rmse': model_rmse,
                'mae': sum(batch['model_mae'] * batch['n_valid'] for batch in self.batch_summaries) / self.n_valid
            },
            'gim_vs_truth': {
                'bias': gim_bias,
                'rmse': gim_rmse,
                'mae': sum(batch['gim_mae'] * batch['n_valid'] for batch in self.batch_summaries) / self.n_valid
            },
            'model_vs_gim': {
                'bias': model_gim_bias,
                'rmse': model_gim_rmse,
                'mae': sum(batch['model_gim_mae'] * batch['n_valid'] for batch in self.batch_summaries) / self.n_valid
            },
            'batch_summaries': self.batch_summaries
        }


class ResultsStreamer:
    """Streams comparison results to disk for large datasets."""
    
    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        self.output_dir = output_dir
        self.config = config
        self.csv_path = output_dir / "batch_comparison_stream.csv"
        self.csv_written = False
        
    def write_batch(self, batch_data: Dict[str, Any], batch_idx: int):
        """Write batch comparison data to CSV stream."""
        # Convert to DataFrame
        batch_df = pd.DataFrame(batch_data)
        
        # Write to CSV (append mode after first batch)
        mode = 'w' if not self.csv_written else 'a'
        header = not self.csv_written
        
        batch_df.to_csv(self.csv_path, mode=mode, header=header, index=False)
        self.csv_written = True
        
    def finalize(self):
        """Finalize streaming and create summary."""
        if self.csv_written:
            logger.info(f"📊 Streamed comparison data: {self.csv_path}")