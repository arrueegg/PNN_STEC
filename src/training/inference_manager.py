"""
Inference Manager Module for PNN_STEC Training

This module handles advanced inference operations including:
- Bayesian inference with uncertainty quantification
- Monte Carlo sampling for uncertainty estimation
- Large-scale inference optimization
- Memory-efficient processing for large datasets

Extracted from BaseTrainer to separate inference concerns.
"""

import torch
import pandas as pd
import gc
from tqdm import tqdm


class InferenceManager:
    """Manages inference operations and uncertainty quantification for STEC prediction models."""

    def __init__(
        self,
        config,
        data_transforms,
        training_utils,
        validation_manager,
        logger,
        device,
    ):
        self.config = config
        self.data_transforms = data_transforms
        self.training_utils = training_utils
        self.validation_manager = validation_manager
        self.logger = logger
        self.device = device

        # Training configuration
        self.use_log_target = config["training"].get("log_target", True)
        self.eps = 1e-6

    def bayesian_inference_total_uncertainty(self, model, dataloader, num_samples=100):
        """
        Compute predictive mean and total uncertainty (epistemic + aleatoric) in ORIGINAL space.
        OPTIMIZED VERSION: Fixes performance degradation issues.

        Supports multiple model types:
        - BNN models: Sample from weight posteriors
        - MC Dropout models: Sample with different dropout masks
        - DE_MLP: Use ensemble-based uncertainty decomposition

        For log-targets, each forward pass is mapped via log-normal moments:
            mean_y_s = exp(mu + 0.5*sigma2) - eps
            var_y_alea_s = (exp(sigma2) - 1) * exp(2*mu + sigma2)
        Then aggregate:
            epistemic_var = Var_s(mean_y_s)
            aleatoric_var = E_s(var_y_alea_s)
        """
        model.eval()

        # Check if this is a Monte Carlo Dropout model and enable dropout
        model_type = self.config["model"]["model_type"]
        is_mc_dropout = model_type in ["MLP_MCDropout_NLL", "MLP_MCDropout_mse"]
        if is_mc_dropout:
            if hasattr(model, "enable_mc_dropout"):
                model.enable_mc_dropout()
                self.logger.info(
                    f"🎲 Using Monte Carlo Dropout inference with {num_samples} samples"
                )
            else:
                raise ValueError(
                    f"Model type {model_type} should support MC Dropout but enable_mc_dropout method not found"
                )
        # Bayesian inference mode

        # OPTIMIZATION 1: Use lists for batch DataFrames, concat once at end
        batch_dataframes = []
        batch_means = []
        batch_epistemic_vars = []
        batch_aleatoric_vars = []
        all_targets = []

        # For large datasets, collect essential features separately to avoid full feature extraction
        dataset_size = len(dataloader.dataset)
        use_minimal_features = dataset_size >= 5_000_000
        batch_essential_features = [] if use_minimal_features else None

        # Progress tracking
        total_batches = len(dataloader)
        processed_samples = 0
        disable_tqdm = self.config.get("cluster", False)

        # Memory management
        gc.collect()
        torch.cuda.empty_cache()

        # Check if dataset returns metadata
        return_metadata = self.config.get("return_metadata", False)
        batch_metadata = [] if return_metadata else None

        # Hoist loop-invariant lookups outside the batch loop
        from model.model import DeepEnsemble, DeepEnsemble_MLP

        is_ensemble = (
            isinstance(model, (DeepEnsemble, DeepEnsemble_MLP))
            or model_type == "DE_MLP"
        )

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(
                tqdm(
                    dataloader,
                    desc="Bayesian Inference",
                    miniters=total_batches // 1000,
                    disable=disable_tqdm,
                )
            ):
                if return_metadata:
                    inputs, targets, metadata = batch_data
                    batch_metadata.append(metadata)
                else:
                    inputs, targets = batch_data

                bs = inputs.size(0)
                inputs = inputs.to(self.device, non_blocking=True)

                if is_ensemble:
                    # For ensemble models, use the decomposed uncertainty method
                    ensemble_mean, aleatoric_var, epistemic_var, total_var = (
                        model.get_uncertainties(inputs)
                    )

                    # OPTIMIZATION 2: Keep on GPU longer, batch CPU transfer
                    stec_mean = ensemble_mean.flatten()
                    epistemic_var = epistemic_var.flatten()
                    aleatoric_var = aleatoric_var.flatten()

                else:
                    # OPTIMIZATION 2: Batch GPU operations, minimize CPU transfers
                    # Original BNN sampling logic - but optimized
                    per_sample_means = []
                    per_sample_alea_vars = []

                    for sample_idx in range(num_samples):
                        outputs = model(inputs)
                        mean_raw, var_raw = self.data_transforms.compute_mean_var(
                            outputs
                        )

                        # [PAPER] Mao et al. 2025: Laplacian variance is 2 * scale^2
                        if "Laplacian" in model_type:
                            var_raw = 2.0 * (var_raw**2)

                        if self.use_log_target:
                            # Log-normal moments for this pass
                            mean_y = torch.exp(mean_raw + 0.5 * var_raw) - self.eps
                            var_alea_y = (torch.exp(var_raw) - 1.0) * torch.exp(
                                2 * mean_raw + var_raw
                            )
                        else:
                            mean_y = mean_raw
                            var_alea_y = var_raw

                        # Detach everything to completely break computation graph
                        mean_y = mean_y.detach()
                        var_alea_y = var_alea_y.detach()

                        # Keep on GPU until all samples done
                        per_sample_means.append(mean_y)
                        per_sample_alea_vars.append(var_alea_y)

                        # Clean up intermediate tensors immediately
                        del outputs, mean_raw, var_raw, mean_y, var_alea_y

                    # OPTIMIZATION 2: Stack and compute on GPU, then transfer once
                    pred_stack = torch.stack(per_sample_means, dim=0)  # [S, B]
                    alea_var_stack = torch.stack(per_sample_alea_vars, dim=0)  # [S, B]

                    stec_mean = pred_stack.mean(dim=0)
                    if num_samples == 1:
                        epistemic_var = torch.zeros_like(
                            pred_stack[0]
                        )  # No epistemic uncertainty
                    else:
                        # [PAPER] Mao et al. 2025: Population variance for Laplacian Ensemble
                        # Otherwise use standard unbiased sample variance for BNNs/Gaussian Ensembles
                        is_laplacian = "Laplacian" in model_type
                        epistemic_var = pred_stack.var(dim=0, unbiased=not is_laplacian)
                    aleatoric_var = alea_var_stack.mean(dim=0)  # Mean aleatoric var

                # OPTIMIZATION 2: Move to CPU immediately and store as CPU tensors (not numpy)
                stec_mean_cpu = stec_mean.detach().cpu()
                epistemic_var_cpu = epistemic_var.detach().cpu()
                aleatoric_var_cpu = aleatoric_var.detach().cpu()
                targets_cpu = targets.detach().cpu()

                batch_means.append(stec_mean_cpu)
                batch_epistemic_vars.append(epistemic_var_cpu)
                batch_aleatoric_vars.append(aleatoric_var_cpu)
                all_targets.append(targets_cpu)

                # OPTIMIZATION 1: Handle DataFrame creation based on dataset size
                if not use_minimal_features:
                    # Small dataset: Create full DataFrame with all features
                    inputs_original, feature_order = (
                        self.validation_manager.inverse_transform_features(inputs)
                    )
                    # Ensure inputs_original is on CPU
                    inputs_original = inputs_original.cpu()

                    batch_df = pd.DataFrame(
                        torch.cat(
                            [
                                inputs_original,
                                targets_cpu.view(bs, -1),
                                stec_mean_cpu.view(bs, -1),
                                torch.sqrt(epistemic_var_cpu).view(bs, -1),
                                torch.sqrt(aleatoric_var_cpu).view(bs, -1),
                                torch.sqrt(
                                    (epistemic_var_cpu + aleatoric_var_cpu)
                                ).view(bs, -1),
                            ],
                            dim=1,
                        ).numpy(),
                        columns=[
                            *feature_order,
                            "target_stec",
                            "pred_stec",
                            "pred_epistemic_unc",
                            "pred_aleatoric_unc",
                            "pred_total_unc",
                        ],
                    )

                    # Add metadata if available
                    if return_metadata and batch_metadata:
                        # Get metadata for this batch (last added entry)
                        batch_meta = batch_metadata[batch_idx]
                        for field in batch_meta[0].keys():
                            batch_df[field] = [
                                sample_meta[field] for sample_meta in batch_meta
                            ]

                    batch_dataframes.append(batch_df)
                else:
                    # Large dataset: Extract only essential features for plotting
                    inputs_original, feature_order = (
                        self.validation_manager.inverse_transform_features(inputs)
                    )

                    # Dynamically build essential features list from the feature registry
                    fr = self.data_transforms.feature_registry
                    # Spatial essentials (IPP and station coordinates)
                    spatial_essentials = [
                        f
                        for f in [
                            "lon_ipp",
                            "lat_ipp",
                            "sm_lat_ipp",
                            "lat_sta",
                            "lon_sta",
                        ]
                        if f in feature_order
                    ]
                    # Temporal essentials: prefer 'sod' or 'local_time_hours' if available, always include 'year' and 'doy' if present
                    temporal_essentials = [
                        f for f in ["year", "doy"] if f in feature_order
                    ]
                    if "sod" in feature_order:
                        temporal_essentials.append("sod")
                    elif "local_time_hours" in feature_order:
                        temporal_essentials.append("local_time_hours")

                    # Direction essentials
                    direction_essentials = [
                        f for f in ["satazi", "satele"] if f in feature_order
                    ]

                    # SWI essentials: pick a small, useful subset if available
                    try:
                        swi_list = fr.get_features_by_type(
                            __import__(
                                "utils.feature_registry", fromlist=["FeatureType"]
                            ).FeatureType.SWI
                        )
                    except Exception:
                        swi_list = []
                    # Pick common SWI features if present (using actual feature names from feature_registry)
                    swi_candidates = [
                        c
                        for c in [
                            "f107_index",
                            "Dst-index,_nT",
                            "Kp_index",
                            "R_Sunspot_No",
                            "AE-index,_nT",
                            "ap_index,_nT",
                        ]
                        if c in feature_order
                    ]

                    essential_feature_names = (
                        spatial_essentials
                        + temporal_essentials
                        + direction_essentials
                        + swi_candidates
                    )

                    # Extract essential features that exist in the feature order
                    batch_essential = {}
                    for i, feature_name in enumerate(feature_order):
                        if feature_name in essential_feature_names:
                            # Use .clone() to create independent copies and break memory references
                            batch_essential[feature_name] = inputs_original[
                                :, i
                            ].clone()

                    batch_essential_features.append(batch_essential)

                # Immediately clear references to CPU tensors to prevent accumulation
                del stec_mean_cpu, epistemic_var_cpu, aleatoric_var_cpu, targets_cpu

                # Comprehensive GPU memory cleanup - IMMEDIATELY after CPU transfer
                del stec_mean, epistemic_var, aleatoric_var, inputs
                if "inputs_original" in locals():
                    del inputs_original

                # For BNN models, clean up sampling tensors immediately
                if model_type != "DE_MLP":
                    if "pred_stack" in locals():
                        del (
                            pred_stack,
                            alea_var_stack,
                            per_sample_means,
                            per_sample_alea_vars,
                        )

                # Clean up remaining variables that could retain references (but keep bs for later use)
                if "feature_order" in locals():
                    del feature_order
                if "batch_essential" in locals():
                    del batch_essential
                if "batch_df" in locals():
                    del batch_df

                # Additional variables from ensemble path
                if "ensemble_mean" in locals():
                    del ensemble_mean, total_var

                processed_samples += bs

        if batch_dataframes:
            final_df = pd.concat(batch_dataframes, ignore_index=True)
            del batch_dataframes
            gc.collect()

        else:
            # Create DataFrame with essential features for plotting
            if use_minimal_features:
                self.logger.info(
                    "📊 Creating DataFrame with essential plotting features for large dataset..."
                )
                mean_tensor = torch.cat(batch_means).squeeze()
                epistemic_tensor = torch.cat(batch_epistemic_vars)
                aleatoric_tensor = torch.cat(batch_aleatoric_vars)
                targets_tensor = torch.cat(all_targets)
                total_std_tensor = torch.sqrt(epistemic_tensor + aleatoric_tensor)

                # Extract essential features from stored batch data
                essential_features = {}
                if batch_essential_features:
                    # Concatenate all essential features safely
                    for feature_name in batch_essential_features[0].keys():
                        try:
                            feature_tensors = [
                                batch[feature_name]
                                for batch in batch_essential_features
                            ]
                            feature_data = torch.cat(feature_tensors)
                            essential_features[feature_name] = (
                                feature_data.cpu().numpy().flatten()
                            )
                            # Clean up intermediate tensors
                            del feature_tensors, feature_data
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to concatenate feature {feature_name}: {e}"
                            )
                            continue

                # Create DataFrame with predictions + essential features
                df_dict = {
                    "target_stec": targets_tensor.cpu().numpy().flatten(),
                    "pred_stec": mean_tensor.cpu().numpy().flatten(),
                    "pred_epistemic_unc": torch.sqrt(epistemic_tensor)
                    .cpu()
                    .numpy()
                    .flatten(),
                    "pred_aleatoric_unc": torch.sqrt(aleatoric_tensor)
                    .cpu()
                    .numpy()
                    .flatten(),
                    "pred_total_unc": total_std_tensor.cpu().numpy().flatten(),
                    **essential_features,
                }

                # Add metadata if available
                if batch_metadata:
                    # Flatten list of batch metadata dicts into single dict with lists
                    metadata_fields = batch_metadata[0][
                        0
                    ].keys()  # Get field names from first sample
                    for field in metadata_fields:
                        # Collect all values for this field across all batches
                        field_values = []
                        for batch_meta in batch_metadata:
                            for sample_meta in batch_meta:
                                field_values.append(sample_meta[field])
                        df_dict[field] = field_values

                final_df = pd.DataFrame(df_dict)
            else:
                final_df = pd.DataFrame()  # Empty DataFrame fallback

        # Final cleanup
        gc.collect()
        torch.cuda.empty_cache()

        # Concatenate CPU tensors directly
        mean = torch.cat(batch_means).squeeze()
        epistemic_var = torch.cat(batch_epistemic_vars)
        aleatoric_var = torch.cat(batch_aleatoric_vars)
        targets = torch.cat(all_targets)

        total_var = epistemic_var + aleatoric_var
        total_std = torch.sqrt(total_var)

        # Disable MC dropout if it was enabled
        if is_mc_dropout and hasattr(model, "disable_mc_dropout"):
            model.disable_mc_dropout()

        # Inference completed

        return {
            "baysian_mae": torch.mean(torch.abs(mean - targets)),
            "baysian_mse": torch.mean((mean - targets) ** 2),
            "mean": mean,
            "epistemic_std": torch.sqrt(epistemic_var),
            "aleatoric_std": torch.sqrt(aleatoric_var),
            "total_std": total_std,
            "targets": targets,
        }, final_df
