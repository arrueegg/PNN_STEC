"""
Training Manager Module for PNN_STEC Training

This module handles training epoch logic for both single models and ensembles.
It manages forward/backward passes, loss computation, and optimization steps.

Extracted from BaseTrainer to separate training execution concerns.
"""

import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics
from utils.loss_function import FairCRPSLoss


class TrainManager:
    """Manages training epochs for STEC prediction models."""

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

    def train_epoch(
        self,
        model,
        dataloader,
        criterion_mse,
        criterion_nll,
        criterion_kld,
        optimizer,
        epoch=0,
    ):
        """Train a single epoch for regular (non-ensemble) models."""
        model.train()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=disable_tqdm,
            desc="Training",
        ):
            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = (
                self.data_transforms.targets_to_training_space(targets)
            )

            optimizer.zero_grad()

            # Check if we're using CRPS loss
            if self.loss_function == "FairCRPS":
                # Multiple stochastic forward passes for CRPS
                samples_list = []
                for _ in range(self.crps_num_samples):
                    outputs = model(inputs)
                    pred_mean_raw, _ = self.data_transforms.compute_mean_var(outputs)
                    samples_list.append(pred_mean_raw.flatten())
                
                # Stack samples: [N, B]
                samples = torch.stack(samples_list, dim=0)
                
                # Compute CRPS loss
                crps_loss = self.crps_criterion(samples, training_targets)
                
                # Add KL divergence (same as for GaussianNLL)
                kld_loss = criterion_kld(model)
                current_kl_weight = self.training_utils.get_current_kl_weight(epoch)
                loss = crps_loss + current_kl_weight * kld_loss
                
                # For logging: use mean of samples as prediction
                pred_mean_raw = samples.mean(dim=0)
                pred_var_raw = samples.var(dim=0)
                
                # Compute auxiliary losses for logging (no gradient)
                with torch.no_grad():
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                
            else:
                # Standard forward pass for Gaussian NLL or MSE
                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(outputs)

                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Loss computation
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)

                # Use annealed KL weight
                current_kl_weight = self.training_utils.get_current_kl_weight(epoch)

                # Select loss based on loss_function
                if self.loss_function == "GaussianNLLLoss":
                    loss = nll_loss + current_kl_weight * kld_loss
                elif self.loss_function == "MSELoss":
                    loss = mse_loss
                else:
                    # Default to GaussianNLL for backward compatibility
                    loss = nll_loss + current_kl_weight * kld_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate losses
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_nll += nll_loss.item()
            running_kld += kld_loss.item()

            # Back-transform to linear space for metrics/logging
            if self.use_log_target:
                point_linear, std_linear, var_linear = (
                    self.data_transforms.pred_log_to_linear(pred_mean_raw, pred_var_raw)
                )
            else:
                point_linear, std_linear, var_linear = (
                    self.data_transforms.pred_linear_from_linear(
                        pred_mean_raw, pred_var_raw
                    )
                )

            # Track average variance in ORIGINAL space (more interpretable)
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
            all_outputs.numpy(),
            all_targets.numpy(),
        )

    def train_epoch_ensemble(
        self,
        model,
        dataloader,
        criterion_mse,
        criterion_nll,
        criterion_kld,
        optimizer,
        epoch,
    ):
        """
        Specialized training for Deep Ensemble models.
        Trains all ensemble members simultaneously.
        """
        model.train()
        running_loss = running_mse = running_nll = running_kld = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=disable_tqdm,
            desc="Training Ensemble",
        ):
            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = (
                self.data_transforms.targets_to_training_space(targets)
            )

            optimizer.zero_grad()

            # Check if we're using CRPS loss
            if self.loss_function == "FairCRPS":
                # Multiple stochastic forward passes for CRPS
                samples_list = []
                for _ in range(self.crps_num_samples):
                    outputs = model(inputs)
                    pred_mean_raw, _ = self.data_transforms.compute_mean_var(outputs)
                    samples_list.append(pred_mean_raw.flatten())
                
                # Stack samples: [N, B]
                samples = torch.stack(samples_list, dim=0)
                
                # Compute CRPS loss
                crps_loss = self.crps_criterion(samples, training_targets)
                
                # Add KL divergence with annealing
                kld_loss = criterion_kld(model)
                current_kl_weight = self.training_utils.get_current_kl_weight(epoch)
                loss = crps_loss + current_kl_weight * kld_loss
                
                # For logging: use mean of samples as prediction
                pred_mean_raw = samples.mean(dim=0)
                pred_var_raw = samples.var(dim=0)
                
                # Compute auxiliary losses for logging (no gradient)
                with torch.no_grad():
                    mse_loss = criterion_mse(pred_mean_raw, training_targets)
                    nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                
            else:
                # Standard forward pass for Gaussian NLL or MSE
                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(outputs)

                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Loss computation
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)

                # For ensemble models, KL divergence might be handled differently
                # Select loss based on loss_function
                if self.loss_function == "GaussianNLLLoss":
                    loss = nll_loss + kld_loss
                elif self.loss_function == "MSELoss":
                    loss = mse_loss
                else:
                    # Default to GaussianNLL for backward compatibility
                    loss = nll_loss + kld_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate losses
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_nll += nll_loss.item()
            running_kld += kld_loss.item()

            # Back-transform to linear space for metrics/logging
            if self.use_log_target:
                point_linear, std_linear, var_linear = (
                    self.data_transforms.pred_log_to_linear(pred_mean_raw, pred_var_raw)
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

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        # Calculate avg loss and variance
        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_nll = running_nll / len(dataloader)
        avg_kld = running_kld / len(dataloader)

        # Calculate avg metrics
        train_metrics = calculate_metrics(all_outputs, all_targets, prefix="train")

        return avg_loss, avg_mse, avg_nll, avg_kld, train_metrics
