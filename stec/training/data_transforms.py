"""`DataTransforms`, ported from `src/training/data_transforms.py`.

Handles target normalization/standardization, log-space transforms and prediction-space
conversions for the legacy `FeatureRegistry`-driven inference scripts
(`scripts/infer_from_log.py`, `vlbi_kband/scripts/infer_vlbi_kband.py`). The paper pipeline's
own equivalent moment-mapping lives inline in `stec.inference.monte_carlo` (see that
module's docstring: "Not ported: the log-target moment mapping" - it declares no log-target
capability because `BayesianResNetSTEC` is trained in linear space). This class is kept as
an independent port, not merged into `monte_carlo.py`, because the scripts that use it also
need `inverse_transform_features`, `pred_linear_from_linear` and the standardize/log-target
toggle combination exactly as the legacy config allows - the parts of the source class the
rebuilt paper pipeline never needed because it only ever produces linear-space,
non-standardized targets.
"""

from __future__ import annotations

import torch

from ..data.feature_registry import FeatureType


class DataTransforms:
    """Handles all data transformations for training and inference."""

    def __init__(self, config: dict, feature_registry, logger, device) -> None:
        self.config = config
        self.feature_registry = feature_registry
        self.logger = logger
        self.device = device

        self.use_log_target = config["training"].get("log_target", True)
        self.use_target_standardization = config["training"].get(
            "standardize_targets", True
        )
        self.log_space_point = config["training"].get("log_space_point", "mean").lower()
        self.eps = 1e-6

        if self.log_space_point not in ("mean", "median"):
            self.log_space_point = "mean"

    def targets_to_training_space(self, targets: torch.Tensor):
        """Targets for the loss computation (standardized + log-space if enabled), plus
        the original targets (always linear/original) for metrics."""
        targets = targets.to(self.device, non_blocking=True)
        original_targets = targets.clone()

        if torch.isnan(targets).any():
            self.logger.warning("NaN detected in targets!")
        if (targets <= 0).any():
            self.logger.warning(
                f"Non-positive targets detected! Min: {targets.min():.6f}, Count: {(targets <= 0).sum()}"
            )

        if self.use_target_standardization:
            targets = self._normalize_targets(targets)

        if self.use_log_target:
            targets_shifted = targets + self.eps
            training_targets = torch.log(targets_shifted)
        else:
            training_targets = targets

        return training_targets, original_targets

    def pred_log_to_linear(self, mu_log: torch.Tensor, var_log: torch.Tensor):
        """Convert log-space (Normal) params to linear-space (LogNormal) mean/std/var.

        `mu_log` is the mean of Z = log(Y + eps) (Y possibly standardized); `var_log` is
        the variance of Z. Returns (point, std_linear, var_linear) in the original scale.
        """
        if torch.isnan(mu_log).any() or torch.isnan(var_log).any():
            self.logger.warning("NaN detected in log-space predictions!")

        if (var_log < 0).any():
            self.logger.warning(
                f"Negative variances detected! Min: {var_log.min():.6f}, Count: {(var_log < 0).sum()}"
            )
            var_log = torch.clamp(var_log, min=1e-8)

        # Log-normal moments in standardized space.
        mean_standardized = torch.exp(mu_log + 0.5 * var_log) - self.eps
        var_standardized = (torch.exp(var_log) - 1.0) * torch.exp(2 * mu_log + var_log)

        if self.log_space_point == "median":
            point_standardized = torch.exp(mu_log) - self.eps
        else:
            point_standardized = mean_standardized

        if self.use_target_standardization:
            point_original, var_original = self._denormalize_predictions(
                point_standardized, var_standardized
            )
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))
        else:
            point_original = point_standardized
            var_original = var_standardized
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))

        if (point_original < 0).any():
            self.logger.warning(
                f"Negative predictions detected! Min: {point_original.min():.6f}, Count: {(point_original < 0).sum()}"
            )

        return point_original, std_original, var_original

    def pred_linear_from_linear(self, mean: torch.Tensor, var: torch.Tensor):
        """Predictions when training in linear space: denormalize if standardization is
        enabled. Returns (point, std, var) in original scale."""
        if self.use_target_standardization:
            denorm_mean, denorm_var = self._denormalize_predictions(mean, var)
            denorm_std = torch.sqrt(denorm_var.clamp_min(0.0))
            return denorm_mean, denorm_std, denorm_var
        std = torch.sqrt(var.clamp_min(0.0))
        return mean, std, var

    def compute_mean_var(self, outputs):
        """Model head returns outputs[0] = mean (mu_log if log-targets, else linear mean),
        outputs[1] = variance (var_log if log-targets, else linear variance)."""
        pred_mean, pred_var = outputs[0], outputs[1]
        return pred_mean, pred_var

    def inverse_transform_features(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse-transform features to original scale using the feature registry."""
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

    def get_feature_indices(self) -> dict:
        """Feature indices for different feature types."""
        if self.feature_registry is None:
            return {}

        indices = {}
        for feature_type in FeatureType:
            indices[feature_type.name.lower()] = (
                self.feature_registry.get_indices_by_type(feature_type)
            )

        return indices

    # ---------- Private methods ----------

    def _normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        target_name = self.config["target"]
        return self.feature_registry.normalize_feature(target_name, targets)

    def _denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        target_name = self.config["target"]
        return self.feature_registry.denormalize_feature(
            target_name, normalized_targets
        )

    def _denormalize_predictions(self, pred_mean: torch.Tensor, pred_var: torch.Tensor):
        """Denormalize (mean, variance) from standardized space back to original scale.

        For variance: if Y = a*X + b, then Var(Y) = a^2 * Var(X), a = (max - min).
        """
        target_name = self.config["target"]
        normalization_params = self.feature_registry.get_normalization_params(
            target_name
        )

        if normalization_params is None:
            return pred_mean, pred_var

        min_val, max_val = normalization_params
        scale_factor = max_val - min_val

        denorm_mean = pred_mean * scale_factor + min_val
        denorm_var = pred_var * (scale_factor**2)

        return denorm_mean, denorm_var
