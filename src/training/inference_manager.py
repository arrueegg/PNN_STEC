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

        For log-targets, each forward pass is mapped via log-normal moments:
            mean_y_s = exp(mu + 0.5*sigma2) - eps
            var_y_alea_s = (exp(sigma2) - 1) * exp(2*mu + sigma2)
        Then aggregate:
            epistemic_var = Var_s(mean_y_s)
            aleatoric_var = E_s(var_y_alea_s)
        """
        model.eval()

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

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(
                tqdm(
                    dataloader,
                    desc="Bayesian Inference",
                    miniters=total_batches // 1000,
                    disable=disable_tqdm,
                )
            ):
                bs = inputs.size(0)
                inputs = inputs.to(self.device, non_blocking=True)

                # Check if this is an ensemble model
                model_type = self.config["model"]["model_type"]
                if model_type == "DE_MLP":
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
                        with torch.no_grad():  # Extra safety
                            outputs = model(inputs)
                            mean_raw, var_raw = self.data_transforms.compute_mean_var(
                                outputs
                            )

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
                        epistemic_var = pred_stack.var(dim=0)  # Var over means
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
                    batch_dataframes.append(batch_df)
                else:
                    # Large dataset: Extract only essential features for plotting
                    inputs_original, feature_order = (
                        self.validation_manager.inverse_transform_features(inputs)
                    )

                    # Define essential features needed for key plots
                    essential_feature_names = [
                        "lon_ipp",
                        "lat_ipp",
                        "sm_lat_ipp",  # Spatial plotting
                        "year",
                        "doy",
                        "time",  # Temporal plotting
                        "satazi",
                        "satele",  # Directional plotting
                        "kp_binned",
                        "dst",
                        "f107",  # Space weather (if available)
                    ]

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

                # Force CUDA cache cleanup every batch to prevent accumulation
                torch.cuda.empty_cache()

                processed_samples += bs

                # Additional cleanup every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        # MEMORY OPTIMIZATION: For very large datasets, create DataFrame with essential features only
        # to avoid OOM errors during concatenation while preserving key plotting capabilities
        dataset_size = len(dataloader.dataset)
        use_minimal_features = (
            dataset_size >= 5_000_000
        )  # Only use minimal features for ≥5M samples

        if use_minimal_features:
            self.logger.info(
                f"💾 Using essential features only for large dataset ({dataset_size:,} samples) to prevent OOM"
            )

        # OPTIMIZATION 1: Use iterative concatenation to avoid memory spikes
        if batch_dataframes:
            self.logger.info(
                "📊 Combining results from all batches using iterative concatenation..."
            )

            # Iterative concatenation in chunks to avoid memory spikes
            chunk_size = 50  # Process 50 DataFrames at a time
            final_df = None

            for i in range(0, len(batch_dataframes), chunk_size):
                chunk_end = min(i + chunk_size, len(batch_dataframes))
                chunk_dfs = batch_dataframes[i:chunk_end]

                # Concatenate this chunk
                if len(chunk_dfs) == 1:
                    chunk_df = chunk_dfs[0]
                else:
                    chunk_df = pd.concat(chunk_dfs, ignore_index=True)

                # Add to final result
                if final_df is None:
                    final_df = chunk_df
                else:
                    final_df = pd.concat([final_df, chunk_df], ignore_index=True)

                # Free memory immediately
                del chunk_dfs, chunk_df

                # Progress update
                if i + chunk_size < len(batch_dataframes):
                    self.logger.info(
                        f"   Processed {chunk_end}/{len(batch_dataframes)} batch DataFrames..."
                    )

                # Periodic garbage collection
                if (i // chunk_size) % 10 == 0:  # Every 10 chunks (500 DataFrames)
                    gc.collect()

            # Final cleanup of batch list
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
                    "pred_epistemic_unc": torch.sqrt(epistemic_tensor).cpu().numpy().flatten(),
                    "pred_aleatoric_unc": torch.sqrt(aleatoric_tensor).cpu().numpy().flatten(),
                    "pred_total_unc": total_std_tensor.cpu().numpy().flatten(),
                    **essential_features,
                }
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

        self.logger.info(
            f"✅ Bayesian inference completed: {processed_samples:,} samples processed"
        )

        return {
            "baysian_mae": torch.mean(torch.abs(mean - targets)),
            "baysian_mse": torch.mean((mean - targets) ** 2),
            "mean": mean,
            "epistemic_std": torch.sqrt(epistemic_var),
            "aleatoric_std": torch.sqrt(aleatoric_var),
            "total_std": total_std,
            "targets": targets,
        }, final_df
