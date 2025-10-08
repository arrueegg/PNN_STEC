"""
Training Manager Module for PNN_STEC Training

This module handles training epoch logic for both single models and ensembles.
It manages forward/backward passes, loss computation, and optimization steps.

Extracted from BaseTrainer to separate training execution concerns.
"""

import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics


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

        # Timing instrumentation with proper synchronization
        epoch_start_time = self.training_utils.sync_and_time()
        data_load_time = 0.0
        gpu_transfer_time = 0.0
        forward_time = 0.0
        loss_compute_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0
        postprocess_time = 0.0
        tqdm_overhead = 0.0
        batch_count = 0

        # Track data loading time
        iter_start_time = self.training_utils.sync_and_time()

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=disable_tqdm,
            desc="Training",
        ):
            # Measure data loading time (time between iterations)
            data_ready_time = self.training_utils.sync_and_time()
            if i > 0:  # Skip first iteration since there's no previous iteration
                data_load_time += data_ready_time - iter_start_time

            batch_start = self.training_utils.sync_and_time()

            # Time data transfer to GPU
            transfer_start = self.training_utils.sync_and_time()
            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = (
                self.data_transforms.targets_to_training_space(targets)
            )
            transfer_end = self.training_utils.sync_and_time()
            gpu_transfer_time += transfer_end - transfer_start

            optimizer.zero_grad()

            # Time forward pass
            forward_start = self.training_utils.sync_and_time()
            outputs = model(inputs)
            pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(outputs)

            pred_mean_raw = pred_mean_raw.flatten()
            pred_var_raw = pred_var_raw.flatten()
            forward_end = self.training_utils.sync_and_time()
            forward_time += forward_end - forward_start

            # Time loss computation
            loss_start = self.training_utils.sync_and_time()
            mse_loss = criterion_mse(pred_mean_raw, training_targets)
            nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
            kld_loss = criterion_kld(model)

            # Use annealed KL weight
            current_kl_weight = self.training_utils.get_current_kl_weight(epoch)

            if self.config["training"]["loss_function"] == "GaussianNLLLoss":
                loss = nll_loss + current_kl_weight * kld_loss
            elif self.config["training"]["loss_function"] == "MSELoss":
                loss = mse_loss
            loss_end = self.training_utils.sync_and_time()
            loss_compute_time += loss_end - loss_start

            # Time backward pass
            backward_start = self.training_utils.sync_and_time()
            loss.backward()
            backward_end = self.training_utils.sync_and_time()
            backward_time += backward_end - backward_start

            # Time optimizer step
            opt_start = self.training_utils.sync_and_time()
            optimizer.step()
            opt_end = self.training_utils.sync_and_time()
            optimizer_time += opt_end - opt_start

            # Time post-processing
            postprocess_start = self.training_utils.sync_and_time()
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
            postprocess_end = self.training_utils.sync_and_time()
            postprocess_time += postprocess_end - postprocess_start

            batch_count += 1

            # Calculate TQDM and other overhead
            batch_end = self.training_utils.sync_and_time()
            batch_total_measured = batch_end - batch_start
            batch_components_sum = (
                (transfer_end - transfer_start)
                + (forward_end - forward_start)
                + (loss_end - loss_start)
                + (backward_end - backward_start)
                + (opt_end - opt_start)
                + (postprocess_end - postprocess_start)
            )
            tqdm_overhead += batch_total_measured - batch_components_sum

            # Log timing for first few batches if detailed timing is enabled
            if self.training_utils.detailed_batch_timing and i < 3:
                data_load_for_batch = (
                    (data_ready_time - iter_start_time) if i > 0 else 0.0
                )
                self.logger.info(
                    f"  Batch {i+1}: data_load={data_load_for_batch:.3f}s, total={batch_total_measured:.3f}s, components={batch_components_sum:.3f}s"
                )

            # Log periodic timing updates if configured
            elif (
                self.training_utils.timing_enabled
                and self.training_utils.log_timing_frequency > 0
                and (i + 1) % self.training_utils.log_timing_frequency == 0
            ):
                data_load_for_batch = (
                    (data_ready_time - iter_start_time) if i > 0 else 0.0
                )
                self.logger.info(
                    f"  Batch {i+1}/{len(dataloader)}: data_load={data_load_for_batch:.3f}s, total={batch_total_measured:.3f}s"
                )

            # Prepare for next iteration timing
            iter_start_time = self.training_utils.sync_and_time()

            # Debug mode: process only one batch
            if self.config.get("debug_single_batch", False):
                break

        # Log epoch timing summary with synchronized end time
        epoch_total = self.training_utils.sync_and_time() - epoch_start_time
        components_sum = (
            gpu_transfer_time
            + forward_time
            + loss_compute_time
            + backward_time
            + optimizer_time
            + postprocess_time
        )
        unaccounted_time = epoch_total - components_sum - data_load_time - tqdm_overhead

        if self.training_utils.timing_enabled:
            self.training_utils.log_timing(
                "Train Epoch Total", epoch_total, f"{batch_count} batches"
            )
            self.training_utils.log_timing(
                "Train Data Loading",
                data_load_time,
                f"avg={data_load_time/(batch_count-1 if batch_count > 1 else 1):.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Data Transfer",
                gpu_transfer_time,
                f"avg={gpu_transfer_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Forward Pass",
                forward_time,
                f"avg={forward_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Loss Computation",
                loss_compute_time,
                f"avg={loss_compute_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Backward Pass",
                backward_time,
                f"avg={backward_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Optimizer Step",
                optimizer_time,
                f"avg={optimizer_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train Post-processing",
                postprocess_time,
                f"avg={postprocess_time/batch_count:.3f}s/batch",
            )
            self.training_utils.log_timing(
                "Train TQDM/Loop Overhead",
                tqdm_overhead,
                f"avg={tqdm_overhead/batch_count:.3f}s/batch",
            )
            self.logger.info(
                f"⏱️  Components sum: {components_sum:.3f}s, Data loading: {data_load_time:.3f}s, Still unaccounted: {unaccounted_time:.3f}s"
            )

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

    def train_ensemble_epoch(
        self, model, dataloader, optimizer, criterion_mse, criterion_nll, criterion_kld, epoch
    ):
        """
        Specialized training for Deep Ensemble models.
        Trains all ensemble members simultaneously.
        """
        model.train()
        running_loss = running_mse = running_nll = running_kld = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        epoch_start_time = self.training_utils.sync_and_time()

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=disable_tqdm,
            desc="Ensemble Training",
        ):

            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = (
                self.data_transforms.targets_to_training_space(targets)
            )

            optimizer.zero_grad()

            # Train all ensemble members with same data simultaneously
            outputs = model(inputs)  # This calls the ensemble forward method
            pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(
                outputs
            )

            pred_mean_raw = pred_mean_raw.flatten()
            pred_var_raw = pred_var_raw.flatten()

            mse_loss = criterion_mse(pred_mean_raw, training_targets)
            nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
            kld_loss = criterion_kld(model)

            current_kl_weight = self.training_utils.get_current_kl_weight(epoch)

            if self.config["training"]["loss_function"] == "GaussianNLLLoss":
                loss = nll_loss + current_kl_weight * kld_loss
            elif self.config["training"]["loss_function"] == "MSELoss":
                loss = mse_loss

            loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_nll += nll_loss.item()
            running_kld += kld_loss.item()

            # Store outputs for metrics calculation
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
        epoch_end_time = self.training_utils.sync_and_time()
        epoch_duration = epoch_end_time - epoch_start_time

        if self.training_utils.timing_enabled:
            self.training_utils.log_timing(
                f"Ensemble training epoch {epoch}", epoch_duration
            )

        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_nll = running_nll / len(dataloader)
        avg_kld = running_kld / len(dataloader)

        all_outputs_tensor = torch.cat(all_outputs, dim=0).detach()
        all_targets_tensor = torch.cat(all_targets, dim=0)
        train_metrics = calculate_metrics(
            all_outputs_tensor, all_targets_tensor, prefix="train"
        )
        """
        Specialized training for Deep Ensemble models.
        Trains all ensemble members simultaneously or sequentially based on config.
        """
        model.train()
        running_loss = running_mse = running_nll = running_kld = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get("cluster", False)

        # Get ensemble training configuration
        ensemble_config = self.config["model"].get("ensemble_training", {})
        train_method = ensemble_config.get(
            "method", "simultaneous"
        )  # 'simultaneous' or 'sequential'

        epoch_start_time = self.training_utils.sync_and_time()

        for i, (inputs, targets) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            disable=disable_tqdm,
            desc="Ensemble Training",
        ):

            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = (
                self.data_transforms.targets_to_training_space(targets)
            )

            optimizer.zero_grad()
            total_loss = 0.0

            if train_method == "simultaneous":
                # Train all ensemble members with same data simultaneously
                outputs = model(inputs)  # This calls the ensemble forward method
                pred_mean_raw, pred_var_raw = self.data_transforms.compute_mean_var(
                    outputs
                )

                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)

                current_kl_weight = self.training_utils.get_current_kl_weight(epoch)

                if self.config["training"]["loss_function"] == "GaussianNLLLoss":
                    loss = nll_loss + current_kl_weight * kld_loss
                elif self.config["training"]["loss_function"] == "MSELoss":
                    loss = mse_loss

                total_loss = loss

            elif train_method == "sequential":
                # Train each ensemble member with the same batch sequentially
                all_predictions, all_variances = model.forward_individual(inputs)

                for member_idx, (pred_mean, pred_var) in enumerate(
                    zip(all_predictions, all_variances)
                ):
                    pred_mean = pred_mean.flatten()
                    pred_var = pred_var.flatten()

                    mse_loss = criterion_mse(pred_mean, training_targets)
                    nll_loss = criterion_nll(pred_mean, training_targets, pred_var)

                    # For ensemble, KL loss only applies to BNN members (not relevant for MLP ensemble)
                    kld_loss = torch.tensor(0.0, device=self.device)

                    if self.config["training"]["loss_function"] == "GaussianNLLLoss":
                        member_loss = nll_loss
                    elif self.config["training"]["loss_function"] == "MSELoss":
                        member_loss = mse_loss

                    total_loss += member_loss

                # Average loss across ensemble members
                total_loss = total_loss / len(all_predictions)

            total_loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            running_loss += total_loss.item()
            if train_method == "simultaneous":
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

            # Store outputs for metrics calculation
            if train_method == "simultaneous":
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
            else:
                # For sequential training, use ensemble mean for metrics
                ensemble_mean = torch.mean(
                    torch.stack(all_predictions), dim=0
                ).flatten()
                if self.use_log_target:
                    point_linear, _, _ = self.data_transforms.pred_log_to_linear(
                        ensemble_mean, torch.zeros_like(ensemble_mean)
                    )
                else:
                    point_linear, _, _ = self.data_transforms.pred_linear_from_linear(
                        ensemble_mean, torch.zeros_like(ensemble_mean)
                    )
                all_outputs.append(point_linear.cpu())

            all_targets.append(original_targets.cpu())

            if self.config.get("debug_single_batch", False):
                break

        # Calculate epoch metrics
        epoch_end_time = self.training_utils.sync_and_time()
        epoch_duration = epoch_end_time - epoch_start_time

        if self.training_utils.timing_enabled:
            self.training_utils.log_timing(
                f"Ensemble training epoch {epoch}", epoch_duration
            )

        avg_loss = running_loss / len(dataloader)
        avg_mse = (
            running_mse / len(dataloader) if train_method == "simultaneous" else 0.0
        )
        avg_nll = (
            running_nll / len(dataloader) if train_method == "simultaneous" else 0.0
        )
        avg_kld = (
            running_kld / len(dataloader) if train_method == "simultaneous" else 0.0
        )

        all_outputs_tensor = torch.cat(all_outputs, dim=0).detach()
        all_targets_tensor = torch.cat(all_targets, dim=0)
        train_metrics = calculate_metrics(
            all_outputs_tensor, all_targets_tensor, prefix="train"
        )

        # Convert to numpy for compatibility with training loop
        all_outputs_numpy = all_outputs_tensor.cpu().numpy()
        all_targets_numpy = all_targets_tensor.cpu().numpy()
        train_metrics["predictions"] = all_outputs_numpy
        train_metrics["targets"] = all_targets_numpy

        return avg_loss, avg_mse, avg_nll, avg_kld, train_metrics
