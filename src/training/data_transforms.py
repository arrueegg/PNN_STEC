"""
Data Transforms Module for PNN_STEC Training

This module handles all data transformations for training, including:
- Target normalization/standardization
- Log-space transformations
- Feature transformations
- Prediction space conversions

Extracted from BaseTrainer to separate data processing concerns.
"""

import torch
from utils.feature_registry import FeatureType


class DataTransforms:
    """Handles all data transformations for training and inference."""

    def __init__(self, config, feature_registry, logger, device):
        self.config = config
        self.feature_registry = feature_registry
        self.logger = logger
        self.device = device

        # Transformation configuration
        self.use_log_target = config["training"].get("log_target", True)
        self.use_target_standardization = config["training"].get(
            "standardize_targets", True
        )
        self.log_space_point = config["training"].get("log_space_point", "mean").lower()
        self.eps = 1e-6

        if self.log_space_point not in ("mean", "median"):
            self.log_space_point = "mean"

        # Debug logging flag
        self._debug_logged = False

    def _debug_log_once(self, *messages):
        """Helper method to log debug messages only once."""
        if not self._debug_logged:
            for message in messages:
                self.logger.info(message)
            self._debug_logged = True

    def targets_to_training_space(self, targets):
        """Return targets for the loss computation (standardized + log-space if enabled) AND keep original for metrics."""
        targets = targets.to(self.device, non_blocking=True)
        original_targets = targets.clone()  # for metrics (always linear/original)

        # Debug: Check target ranges
        if torch.isnan(targets).any():
            self.logger.warning("NaN detected in targets!")
        if (targets <= 0).any():
            self.logger.warning(
                f"Non-positive targets detected! Min: {targets.min():.6f}, Count: {(targets <= 0).sum()}"
            )

        # First apply target standardization if enabled
        if self.use_target_standardization:
            targets = self._normalize_targets(targets)
            self._debug_log_once(
                f"Target standardization - Original range: [{original_targets.min():.6f}, {original_targets.max():.6f}]",
                f"Target standardization - Standardized range: [{targets.min():.6f}, {targets.max():.6f}]"
            )

        # Then apply log transformation if enabled (on standardized targets)
        if self.use_log_target:
            # Add epsilon for numerical stability
            targets_shifted = targets + self.eps
            training_targets = torch.log(targets_shifted)
            self._debug_log_once(
                f"Target transform - Standardized range: [{targets.min():.6f}, {targets.max():.6f}]",
                f"Target transform - Log-space range: [{training_targets.min():.6f}, {training_targets.max():.6f}]",
                f"Target transform - Using eps = {self.eps}"
            )
        else:
            training_targets = targets
            if self.use_target_standardization:
                message = f"Target transform - Using standardized linear space, range: [{targets.min():.6f}, {targets.max():.6f}]"
            else:
                message = f"Target transform - Using linear space, range: [{targets.min():.6f}, {targets.max():.6f}]"
            self._debug_log_once(message)

        return training_targets, original_targets

    def pred_log_to_linear(self, mu_log, var_log):
        """
        Convert log-space (Normal) params to linear-space (LogNormal) mean/std/var.
        mu_log: mean of Z = log(Y+eps) (where Y might be standardized)
        var_log: variance of Z
        Returns:
            point (mean or median in original space), std_linear, var_linear
        """
        # Debug: Check for any numerical issues
        if torch.isnan(mu_log).any() or torch.isnan(var_log).any():
            self.logger.warning("NaN detected in log-space predictions!")

        if (var_log < 0).any():
            self.logger.warning(
                f"Negative variances detected! Min: {var_log.min():.6f}, Count: {(var_log < 0).sum()}"
            )
            var_log = torch.clamp(var_log, min=1e-8)

        # First convert from log-space to standardized linear space
        # Log-normal moments in standardized space
        mean_standardized = torch.exp(mu_log + 0.5 * var_log) - self.eps
        var_standardized = (torch.exp(var_log) - 1.0) * torch.exp(2 * mu_log + var_log)

        if self.log_space_point == "median":
            point_standardized = torch.exp(mu_log) - self.eps  # median of log-normal
        else:
            point_standardized = mean_standardized  # unbiased mean

        # Then denormalize to original scale if target standardization is enabled
        if self.use_target_standardization:
            point_original, var_original = self._denormalize_predictions(
                point_standardized, var_standardized
            )
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))
        else:
            point_original = point_standardized
            var_original = var_standardized
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))

        # Debug: Check for negative predictions (would indicate epsilon issues)
        if (point_original < 0).any():
            self.logger.warning(
                f"Negative predictions detected! Min: {point_original.min():.6f}, Count: {(point_original < 0).sum()}"
            )

        return point_original, std_original, var_original

    def pred_linear_from_linear(self, mean, var):
        """
        Handle predictions when training in linear space.
        If target standardization is enabled, denormalize back to original scale.
        Returns (point, std, var) in original scale.
        """
        if self.use_target_standardization:
            # Denormalize from standardized space to original scale
            denorm_mean, denorm_var = self._denormalize_predictions(mean, var)
            denorm_std = torch.sqrt(denorm_var.clamp_min(0.0))
            return denorm_mean, denorm_std, denorm_var
        else:
            # No standardization, return as-is
            std = torch.sqrt(var.clamp_min(0.0))
            return mean, std, var

    def compute_mean_var(self, outputs):
        """
        Model head should return:
            outputs[0] = mean   (mu_log if log-targets, else linear mean)
            outputs[1] = variance (var_log if log-targets, else linear variance)
        """
        pred_mean, pred_var = outputs[0], outputs[1]
        return pred_mean, pred_var

    def inverse_transform_features(self, x):
        """Inverse transform features to original scale using feature registry."""
        if self.feature_registry is None:
            return x

        x_denorm = x.clone()
        features = self.feature_registry.get_feature_names()

        for i, feature_name in enumerate(features):
            feature_col = x_denorm[:, i]
            denorm_col = self.feature_registry.denormalize_feature(
                feature_name, feature_col
            )
            x_denorm[:, i] = denorm_col

        return x_denorm

    def get_feature_indices(self):
        """Get feature indices for different feature types."""
        if self.feature_registry is None:
            return {}

        indices = {}
        for feature_type in FeatureType:
            indices[feature_type.name.lower()] = (
                self.feature_registry.get_indices_by_type(feature_type)
            )

        return indices

    # ---------- Private methods ----------

    def _normalize_targets(self, targets):
        """Normalize targets to [0, 1] range using feature registry."""
        target_name = self.config["target"]
        return self.feature_registry.normalize_feature(target_name, targets)

    def _denormalize_targets(self, normalized_targets):
        """Denormalize targets from [0, 1] range back to original scale."""
        target_name = self.config["target"]
        return self.feature_registry.denormalize_feature(
            target_name, normalized_targets
        )

    def _denormalize_predictions(self, pred_mean, pred_var):
        """
        Denormalize predictions (mean and variance) from standardized space back to original scale.

        For variance: if Y = a*X + b, then Var(Y) = a^2 * Var(X)
        where a = (max - min) is the scaling factor from normalization
        """
        target_name = self.config["target"]
        normalization_params = self.feature_registry.get_normalization_params(
            target_name
        )

        if normalization_params is None:
            return pred_mean, pred_var

        min_val, max_val = normalization_params
        scale_factor = max_val - min_val

        # Denormalize mean: Y = X * scale + min
        denorm_mean = pred_mean * scale_factor + min_val

        # Denormalize variance: Var(Y) = scale^2 * Var(X)
        denorm_var = pred_var * (scale_factor**2)

        return denorm_mean, denorm_var
