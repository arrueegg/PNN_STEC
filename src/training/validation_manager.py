"""
Validation Manager Module for PNN_STEC Training

This module handles validation and testing logic for both single models and ensembles.
It manages model evaluation, metrics computation, and test data processing.

Extracted from BaseTrainer to separate validation and testing concerns.
"""

import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.feature_registry import FeatureType
from utils.loss_function import FairCRPSLoss


class ValidationManager:
    """Manages validation and testing for STEC prediction models."""

    def __init__(self, config, data_transforms, training_utils, logger, device):
        self.config = config
        self.data_transforms = data_transforms
        self.training_utils = training_utils
        self.logger = logger
        self.device = device

        # Training configuration
        self.use_log_target = config["training"].get("log_target", True)
        self.loss_function = config["training"].get("loss_function", "GaussianNLLLoss")
        self.crps_num_samples = config["training"].get("crps_num_samples", 16)
        
        # Initialize CRPS loss if needed
        self.crps_criterion = FairCRPSLoss() if self.loss_function == "FairCRPS" else None

    def validate_epoch(
        self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0
    ):
        """Validate a single epoch for regular (non-ensemble) models."""
        model.eval()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        with torch.no_grad():
            for inputs, targets in tqdm(
                dataloader, desc="Validation", disable=disable_tqdm
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = (
                    self.data_transforms.targets_to_training_space(targets)
                )

                # Check if we're using CRPS loss
                if self.loss_function == "FairCRPS":
                    # Multiple stochastic forward passes for CRPS
                    # Temporarily set to train mode for stochastic sampling
                    was_training = model.training
                    model.train()
                    
                    samples_list = []
                    for _ in range(self.crps_num_samples):
                        outputs = model(inputs)
                        pred_mean_raw, _ = self.data_transforms.compute_mean_var(outputs)
                        samples_list.append(pred_mean_raw.flatten())
                    
                    # Restore original mode
                    if not was_training:
                        model.eval()
                    
                    # Stack samples: [N, B]
                    samples = torch.stack(samples_list, dim=0)
                    
                    # Compute CRPS loss (NOTE: KL divergence NOT included in loss)
                    loss = self.crps_criterion(samples, training_targets)
                    
                    # For logging: use mean of samples as prediction
                    pred_mean_raw = samples.mean(dim=0)
                    pred_var_raw = samples.var(dim=0)
                    
                    # Compute auxiliary losses for logging
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                    kld_loss = criterion_kld(model)
                    
                else:
                    # Standard forward pass
                    outputs = model(inputs)
                    pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(
                        outputs
                    )

                    pred_mean_raw = pred_mean_raw.flatten()
                    pred_var_raw = pred_var_raw.flatten()

                    # Losses in training space
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                    kld_loss = criterion_kld(model)

                    # Use same annealed KL weight as training
                    current_kl_weight = self.training_utils.get_current_kl_weight(epoch)

                    # Use same loss calculation logic as training
                    if self.loss_function == "GaussianNLLLoss":
                        loss = nll_loss + current_kl_weight * kld_loss
                    elif self.loss_function == "LaplacianNLLLoss":  # [PAPER] Mao et al. 2025
                        loss = nll_loss
                    elif self.loss_function == "MSELoss":
                        loss = mse_loss
                    else:
                        # Default to GaussianNLL for backward compatibility
                        loss = nll_loss + current_kl_weight * kld_loss

                # Accumulate losses
                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

                # Back-transform to linear space for metrics/logging
                if self.use_log_target:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_log_to_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )
                else:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_linear_from_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )

                # Track average variance in ORIGINAL space
                running_variance += torch.mean(var_linear).item()

                # Store outputs for metrics (pred, std) and original targets
                all_outputs.append(
                    torch.stack([point_linear, std_linear], dim=1).detach().cpu()
                )
                all_targets.append(original_targets.detach().cpu())

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        n_batches = len(dataloader)
        return (
            running_loss / n_batches,
            running_mse / n_batches,
            running_nll / n_batches,
            running_kld / n_batches,
            running_variance / n_batches,
            all_outputs,
            all_targets,
        )

    def validate_epoch_ensemble(
        self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0
    ):
        """
        Specialized validation for Deep Ensemble models.
        """
        model.eval()
        running_loss = running_mse = running_nll = running_kld = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        with torch.no_grad():
            for inputs, targets in tqdm(
                dataloader, desc="Ensemble Validation", disable=disable_tqdm
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = (
                    self.data_transforms.targets_to_training_space(targets)
                )

                # Check if we're using CRPS loss
                if self.loss_function == "FairCRPS":
                    # Multiple stochastic forward passes for CRPS
                    # Temporarily set to train mode for stochastic sampling
                    was_training = model.training
                    model.train()
                    
                    samples_list = []
                    for _ in range(self.crps_num_samples):
                        outputs = model(inputs)
                        pred_mean_raw, _ = self.data_transforms.compute_mean_var(outputs)
                        samples_list.append(pred_mean_raw.flatten())
                    
                    # Restore original mode
                    if not was_training:
                        model.eval()
                    
                    # Stack samples: [N, B]
                    samples = torch.stack(samples_list, dim=0)
                    
                    # Compute CRPS loss (NOTE: KL divergence NOT included for ensembles)
                    loss = self.crps_criterion(samples, training_targets)
                    
                    # For logging: use mean of samples as prediction
                    pred_mean_raw = samples.mean(dim=0)
                    pred_var_raw = samples.var(dim=0)
                    
                    # Compute auxiliary losses for logging
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                    kld_loss = torch.tensor(0.0, device=self.device)  # No KL loss for ensembles
                    
                else:
                    # Get ensemble prediction (aggregated)
                    outputs = model(inputs)
                    pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(
                        outputs
                    )

                    pred_mean_raw = pred_mean_raw.flatten()
                    pred_var_raw = pred_var_raw.flatten()

                    # Losses in training space
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                    kld_loss = torch.tensor(
                        0.0, device=self.device
                    )  # No KL loss for MLP ensembles

                    if self.loss_function == "GaussianNLLLoss":
                        loss = nll_loss
                    elif self.loss_function == "LaplacianNLLLoss":  # [PAPER] Mao et al. 2025
                        loss = nll_loss
                    elif self.loss_function == "MSELoss":
                        loss = mse_loss
                    else:
                        # Default to GaussianNLL
                        loss = nll_loss

                # Accumulate losses
                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

                # Back-transform to linear space for metrics/logging
                if self.use_log_target:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_log_to_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )
                else:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_linear_from_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )

                all_outputs.append(point_linear.cpu())
                all_targets.append(original_targets.cpu())

                if self.config.get("debug_single_batch", False):
                    break

        # Calculate epoch metrics
        n_batches = len(dataloader)
        avg_loss = running_loss / n_batches
        avg_mse = running_mse / n_batches
        avg_nll = running_nll / n_batches
        avg_kld = running_kld / n_batches

        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_metrics = calculate_metrics(
            all_outputs_tensor, all_targets_tensor, prefix="val"
        )

        # Convert to numpy for compatibility
        all_outputs_numpy = all_outputs_tensor.cpu().numpy()
        all_targets_numpy = all_targets_tensor.cpu().numpy()
        val_metrics["predictions"] = all_outputs_numpy
        val_metrics["targets"] = all_targets_numpy

        return avg_loss, avg_mse, avg_nll, avg_kld, val_metrics

    def test_model(self, model, dataloader):
        """Test the model and return predictions with uncertainties."""
        model.eval()
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        with torch.no_grad():
            for inputs, targets in tqdm(
                dataloader, desc="Testing", disable=disable_tqdm
            ):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = (
                    self.data_transforms.targets_to_training_space(targets)
                )

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(
                    outputs
                )

                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Back-transform to linear space for outputs
                if self.use_log_target:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_log_to_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )
                else:
                    point_linear, std_linear, var_linear = (
                        self.data_transforms.pred_linear_from_linear(
                            pred_mean_raw, pred_var_raw
                        )
                    )

                # Store outputs for metrics (pred, std) and original targets
                all_outputs.append(
                    torch.stack([point_linear, std_linear], dim=1).detach().cpu()
                )
                all_targets.append(original_targets.detach().cpu())

        return torch.cat(all_outputs), torch.cat(all_targets)

    def inverse_transform_features(self, x):
        """Transform normalized features back to original scale using feature registry."""
        feature_registry = self.data_transforms.feature_registry

        # Get output indices from the feature registry (these map to the transformed feature vector)
        output_indices = feature_registry._output_indices

        rescaled_features = {}

        # Process each feature type
        for feature_type in [
            FeatureType.TEMPORAL,
            FeatureType.STATION,
            FeatureType.DIRECTION,
            FeatureType.IPP,
            FeatureType.SWI,
        ]:

            feature_names = feature_registry.get_feature_names(feature_type)

            for feature_name in feature_names:
                if feature_type == FeatureType.TEMPORAL:
                    if feature_name == "year":
                        norm_key = f"{feature_name}_norm"
                        if norm_key in output_indices:
                            norm_idx = output_indices[norm_key]
                            rescaled_features[feature_name] = (
                                feature_registry.denormalize_feature(
                                    feature_name, x[:, norm_idx]
                                )
                            )
                    elif feature_name in ["doy", "sod", "local_time_hours"]:
                        # For cyclic features, we use the normalized version for inverse transform
                        norm_key = f"{feature_name}_norm"
                        if norm_key in output_indices:
                            norm_idx = output_indices[norm_key]
                            rescaled_features[feature_name] = (
                                feature_registry.denormalize_feature(
                                    feature_name, x[:, norm_idx]
                                )
                            )

                elif feature_type == FeatureType.STATION:
                    norm_key = f"{feature_name}_norm"
                    if norm_key in output_indices:
                        norm_idx = output_indices[norm_key]
                        rescaled_features[feature_name] = (
                            feature_registry.denormalize_feature(
                                feature_name, x[:, norm_idx]
                            )
                        )

                elif feature_type == FeatureType.DIRECTION:
                    if feature_name == "satazi":
                        # For azimuth, try multiple possible stored representations
                        # 1) sin/cos pair: satazi_sin / satazi_cos
                        # 2) Cartesian unit vector: e_east / e_north
                        # 3) normalized scalar (fallback)
                        if f"{feature_name}_sin" in output_indices and f"{feature_name}_cos" in output_indices:
                            sin_idx = output_indices[f"{feature_name}_sin"]
                            cos_idx = output_indices[f"{feature_name}_cos"]
                            azi_rad = torch.atan2(x[:, sin_idx], x[:, cos_idx])
                            azi_deg = (azi_rad * 180 / torch.pi) % 360
                            rescaled_features[feature_name] = azi_deg
                        elif "e_east" in output_indices and "e_north" in output_indices:
                            # Reconstruct azimuth from east/north components
                            east_idx = output_indices["e_east"]
                            north_idx = output_indices["e_north"]
                            azi_rad = torch.atan2(x[:, east_idx], x[:, north_idx])
                            azi_deg = (azi_rad * 180 / torch.pi) % 360
                            rescaled_features[feature_name] = azi_deg
                        elif f"{feature_name}_norm" in output_indices:
                            # Fallback: denormalize scalar azimuth
                            norm_idx = output_indices[f"{feature_name}_norm"]
                            rescaled_features[feature_name] = (
                                feature_registry.denormalize_feature(
                                    feature_name, x[:, norm_idx]
                                )
                            )
                        else:
                            # No known representation found - warn and fill with NaNs
                            self.logger.warning(f"No stored representation found for direction feature '{feature_name}' in output_indices")
                            rescaled_features[feature_name] = torch.full((x.size(0),), float('nan'))

                    elif feature_name == "satele":
                        # For elevation, prefer normalized scalar; otherwise reconstruct from e_up
                        if f"{feature_name}_norm" in output_indices:
                            norm_idx = output_indices[f"{feature_name}_norm"]
                            rescaled_features[feature_name] = (
                                feature_registry.denormalize_feature(
                                    feature_name, x[:, norm_idx]
                                )
                            )
                        elif "e_up" in output_indices:
                            up_idx = output_indices["e_up"]
                            # Clamp to valid domain for asin
                            eps = 1e-6
                            e_up = torch.clamp(x[:, up_idx], -1.0 + eps, 1.0 - eps)
                            ele_rad = torch.asin(e_up)
                            ele_deg = ele_rad * 180 / torch.pi
                            rescaled_features[feature_name] = ele_deg
                        else:
                            self.logger.warning(f"No stored representation found for direction feature '{feature_name}' in output_indices")
                            rescaled_features[feature_name] = torch.full((x.size(0),), float('nan'))

                elif feature_type == FeatureType.IPP:
                    norm_key = f"{feature_name}_norm"
                    # Only process if the feature exists in output_indices (it's enabled)
                    if norm_key in output_indices:
                        norm_idx = output_indices[norm_key]
                        rescaled_features[feature_name] = (
                            feature_registry.denormalize_feature(
                                feature_name, x[:, norm_idx]
                            )
                        )

                elif feature_type == FeatureType.SWI:
                    norm_key = f"{feature_name}_norm"
                    if norm_key in output_indices:
                        norm_idx = output_indices[norm_key]
                        rescaled_features[feature_name] = (
                            feature_registry.denormalize_feature(
                                feature_name, x[:, norm_idx]
                            )
                        )

        # Convert to tensor and return in a consistent order
        feature_list = []
        feature_order = []

        # Add features in registry order
        for feature_name in feature_registry.get_all_enabled_features():
            if (
                feature_name in rescaled_features
                and feature_name
                not in feature_registry.get_features_by_type(FeatureType.TARGET)
            ):
                feature_list.append(rescaled_features[feature_name].unsqueeze(1))
                feature_order.append(feature_name)

        if feature_list:
            return torch.cat(feature_list, dim=1), feature_order
        else:
            return torch.empty(x.size(0), 0), []

    def get_feature_indices(self):
        """Get a mapping of feature names to their column indices in the inverse-transformed data."""
        feature_registry = self.data_transforms.feature_registry

        # Get all enabled features excluding targets
        all_enabled = feature_registry.get_all_enabled_features()
        target_features = feature_registry.get_features_by_type(FeatureType.TARGET)
        input_features = [f for f in all_enabled if f not in target_features]

        # Create mapping
        indices = {}
        for idx, feature_name in enumerate(input_features):
            indices[feature_name] = idx

        return indices
