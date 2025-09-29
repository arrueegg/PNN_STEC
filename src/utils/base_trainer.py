# base_trainer.py
import os
import time
import torch
import pandas as pd
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import gc
from datetime import datetime, timedelta

from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer, get_scheduler
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics
from utils.feature_registry import create_default_registry, FeatureType

gc.collect()

class BaseTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.get('device', torch.device('cpu'))
        
        # Initialize feature management
        self.feature_registry = config.get('feature_registry') or create_default_registry(config)
        
        self.loss_weight = config['training']['loss_weight']
        self.eps = 1e-6

        # KL annealing configuration
        self.kl_annealing = config['training'].get('kl_annealing', {})
        self.use_kl_annealing = self.kl_annealing.get('enabled', False)
        if self.use_kl_annealing:
            self.kl_warmup_epochs = self.kl_annealing.get('warmup_epochs', 20)
            self.kl_start_weight = self.kl_annealing.get('start_weight', 0.0)
            self.kl_end_weight = self.kl_annealing.get('end_weight', self.loss_weight)
            self.logger.info(f"🔥 KL annealing enabled: {self.kl_start_weight} → {self.kl_end_weight} over {self.kl_warmup_epochs} epochs")

        # Train loss in log space?
        self.use_log_target = self.config['training'].get('log_target', True)
        # Use target standardization (normalize targets to [0,1] before log transform)?
        self.use_target_standardization = self.config['training'].get('standardize_targets', True)
        # How to map log-normal back to linear for the point estimate: "mean" or "median"
        #   "mean": exp(mu + 0.5*sigma2)  (recommended for MSE/sums and GaussianNLLLoss)
        #   "median": exp(mu)             (can be better for heavy tails if you prefer)
        self.log_space_point = self.config['training'].get('log_space_point', 'mean').lower()
        if self.log_space_point not in ('mean', 'median'):
            self.log_space_point = 'mean'

        # Add loss tracking
        self.train_losses = []
        self.val_losses = []
        self.epochs_tracked = []
        
        # Log target weighting configuration
        weighting_config = self.config['training'].get('target_weighting', {})
        if weighting_config.get('enabled', False):
            weight_func = weighting_config.get('weight_function', 'linear')
            if weight_func == 'quantile':
                threshold = weighting_config.get('high_value_threshold', 0.75)
                weight = weighting_config.get('high_value_weight', 3.0)
                self.logger.info(f"🎯 Target weighting enabled: {weight_func} (>{threshold*100}th percentile gets {weight}x weight)")
            else:
                self.logger.info(f"🎯 Target weighting enabled: {weight_func} scaling")
        else:
            self.logger.info("📊 Standard loss weighting (no target-based scaling)")
        
        # Timing instrumentation for performance debugging
        # Enable timing only if explicitly requested
        self.timing_enabled = self.config.get('enable_timing', False)
        
        # Get timing configuration
        timing_config = self.config.get('timing_config', {})
        self.detailed_batch_timing = timing_config.get('detailed_batch_timing', False) and self.timing_enabled
        self.save_timing_to_file = timing_config.get('save_timing_to_file', True)
        self.log_timing_frequency = timing_config.get('log_timing_every_n_batches', 100)
        
        if self.timing_enabled:
            self.logger.info("🔍 Performance timing enabled")
            if self.detailed_batch_timing:
                self.logger.info("⚠️  Detailed batch timing enabled (may slow training)")
            if self.save_timing_to_file:
                self.logger.info("📁 Timing statistics will be saved to file")
        else:
            self.logger.info("⏱️  Performance timing disabled")
        self.timing_stats = {}

    # ---------- small helpers ----------
    
    def _sync_and_time(self):
        """Synchronize CUDA operations and return current time for accurate timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()
    
    def _log_timing(self, step_name, duration, additional_info=""):
        """Log timing information for performance debugging"""
        if self.timing_enabled:
            if step_name not in self.timing_stats:
                self.timing_stats[step_name] = []
            self.timing_stats[step_name].append(duration)
            info_str = f" ({additional_info})" if additional_info else ""
            self.logger.info(f"⏱️  {step_name}: {duration:.3f}s{info_str}")
    
    def _print_timing_summary(self):
        """Print comprehensive summary of timing statistics"""
        if self.timing_enabled and self.timing_stats:
            self.logger.info("📊 PERFORMANCE TIMING SUMMARY:")
            self.logger.info("=" * 70)
            
            # Sort by total time to show most time-consuming operations first
            sorted_stats = sorted(self.timing_stats.items(), 
                                key=lambda x: sum(x[1]), reverse=True)
            
            total_measured_time = sum(sum(times) for times in self.timing_stats.values())
            
            for step_name, times in sorted_stats:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                min_time = min(times)
                max_time = max(times)
                percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
                
                self.logger.info(f"  {step_name}:")
                self.logger.info(f"    Total: {total_time:.1f}s ({percentage:.1f}%) | "
                               f"Avg: {avg_time:.3f}s | Count: {len(times)} | "
                               f"Range: {min_time:.3f}s - {max_time:.3f}s")
            
            self.logger.info(f"\n  Total measured time: {total_measured_time:.1f}s")
            self.logger.info("=" * 70)
    
    def _save_timing_stats(self):
        """Save timing statistics to a CSV file"""
        if self.timing_enabled and self.save_timing_to_file and self.timing_stats:
            import csv
            import os
            
            timing_file = os.path.join(self.config['output_dir'], 'timing_statistics.csv')
            
            with open(timing_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Step', 'Total_Time_s', 'Average_Time_s', 'Count', 'Min_Time_s', 'Max_Time_s', 'Percentage'])
                
                total_measured_time = sum(sum(times) for times in self.timing_stats.values())
                
                # Sort by total time
                sorted_stats = sorted(self.timing_stats.items(), 
                                    key=lambda x: sum(x[1]), reverse=True)
                
                for step_name, times in sorted_stats:
                    total_time = sum(times)
                    avg_time = total_time / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    percentage = (total_time / total_measured_time * 100) if total_measured_time > 0 else 0
                    
                    writer.writerow([step_name, f"{total_time:.3f}", f"{avg_time:.3f}", 
                                   len(times), f"{min_time:.3f}", f"{max_time:.3f}", f"{percentage:.2f}"])
            
            self.logger.info(f"📁 Timing statistics saved to: {timing_file}")

    def _targets_to_training_space(self, targets):
        """Return targets for the loss computation (standardized + log-space if enabled) AND keep original for metrics."""
        targets = targets.to(self.device, non_blocking=True)
        original_targets = targets.clone()  # for metrics (always linear/original)
        
        # Debug: Check target ranges
        if torch.isnan(targets).any():
            self.logger.warning(f"NaN detected in targets!")
        if (targets <= 0).any():
            self.logger.warning(f"Non-positive targets detected! Min: {targets.min():.6f}, Count: {(targets <= 0).sum()}")
        
        # First apply target standardization if enabled
        if self.use_target_standardization:
            targets = self._normalize_targets(targets)
            
            # Debug: Standardization ranges
            if hasattr(self, '_debug_logged') and not self._debug_logged:
                self.logger.info(f"Target standardization - Original range: [{original_targets.min():.6f}, {original_targets.max():.6f}]")
                self.logger.info(f"Target standardization - Standardized range: [{targets.min():.6f}, {targets.max():.6f}]")
        
        # Then apply log transformation if enabled (on standardized targets)
        if self.use_log_target:
            # Add epsilon for numerical stability
            targets_shifted = targets + self.eps
            training_targets = torch.log(targets_shifted)
            
            # Debug: Log transformation ranges
            if hasattr(self, '_debug_logged') and not self._debug_logged:
                self.logger.info(f"Target transform - Standardized range: [{targets.min():.6f}, {targets.max():.6f}]")
                self.logger.info(f"Target transform - Log-space range: [{training_targets.min():.6f}, {training_targets.max():.6f}]")
                self.logger.info(f"Target transform - Using eps = {self.eps}")
                self._debug_logged = True
        else:
            training_targets = targets
            
            # Debug: Linear space
            if hasattr(self, '_debug_logged') and not self._debug_logged:
                if self.use_target_standardization:
                    self.logger.info(f"Target transform - Using standardized linear space, range: [{targets.min():.6f}, {targets.max():.6f}]")
                else:
                    self.logger.info(f"Target transform - Using linear space, range: [{targets.min():.6f}, {targets.max():.6f}]")
                self._debug_logged = True
        
        return training_targets, original_targets

    def _pred_log_to_linear(self, mu_log, var_log):
        """
        Convert log-space (Normal) params to linear-space (LogNormal) mean/std/var.
        mu_log: mean of Z = log(Y+eps) (where Y might be standardized)
        var_log: variance of Z
        Returns:
            point (mean or median in original space), std_linear, var_linear
        """
        # Debug: Check for any numerical issues
        if torch.isnan(mu_log).any() or torch.isnan(var_log).any():
            self.logger.warning(f"NaN detected in log-space predictions!")
        
        if (var_log < 0).any():
            self.logger.warning(f"Negative variances detected! Min: {var_log.min():.6f}, Count: {(var_log < 0).sum()}")
            var_log = torch.clamp(var_log, min=1e-8)
        
        # First convert from log-space to standardized linear space
        # Log-normal moments in standardized space
        mean_standardized = torch.exp(mu_log + 0.5 * var_log) - self.eps
        var_standardized = (torch.exp(var_log) - 1.0) * torch.exp(2 * mu_log + var_log)

        if self.log_space_point == 'median':
            point_standardized = torch.exp(mu_log) - self.eps  # median of log-normal
        else:
            point_standardized = mean_standardized  # unbiased mean

        # Then denormalize to original scale if target standardization is enabled
        if self.use_target_standardization:
            point_original, var_original = self._denormalize_predictions(point_standardized, var_standardized)
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))
        else:
            point_original = point_standardized
            var_original = var_standardized
            std_original = torch.sqrt(torch.clamp(var_original, min=1e-8))

        # Debug: Check for negative predictions (would indicate epsilon issues)
        if (point_original < 0).any():
            self.logger.warning(f"Negative predictions detected! Min: {point_original.min():.6f}, Count: {(point_original < 0).sum()}")

        return point_original, std_original, var_original

    def _pred_linear_from_linear(self, mean, var):
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

    # ---------- target standardization methods ----------

    def _normalize_targets(self, targets):
        """Normalize targets to [0, 1] range using feature registry."""
        target_name = self.config['target']
        return self.feature_registry.normalize_feature(target_name, targets)

    def _denormalize_targets(self, normalized_targets):
        """Denormalize targets from [0, 1] range back to original scale."""
        target_name = self.config['target']
        return self.feature_registry.denormalize_feature(target_name, normalized_targets)

    def _denormalize_predictions(self, pred_mean, pred_var):
        """
        Denormalize predictions (mean and variance) from standardized space back to original scale.
        
        For variance: if Y = a*X + b, then Var(Y) = a^2 * Var(X)
        where a = (max - min) is the scaling factor from normalization
        """
        target_name = self.config['target']
        normalization_params = self.feature_registry.get_normalization_params(target_name)
        
        if normalization_params is None:
            return pred_mean, pred_var
        
        min_val, max_val = normalization_params
        scale_factor = max_val - min_val
        
        # Denormalize mean: Y = X * scale + min
        denorm_mean = pred_mean * scale_factor + min_val
        
        # Denormalize variance: Var(Y) = scale^2 * Var(X)
        denorm_var = pred_var * (scale_factor ** 2)
        
        return denorm_mean, denorm_var

    # ---------- model I/O naming ----------

    def compute_mean_var(self, outputs):
        """
        Model head should return:
            outputs[0] = mean   (mu_log if log-targets, else linear mean)
            outputs[1] = variance (var_log if log-targets, else linear variance)
        """
        pred_mean, pred_var = outputs[0], outputs[1]
        # Ensure consistent shapes by flattening both outputs
        pred_mean = pred_mean.flatten() if pred_mean.dim() > 1 else pred_mean
        pred_var = pred_var.flatten() if pred_var.dim() > 1 else pred_var
        return pred_mean, pred_var

    def train_epoch_ensemble(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, optimizer, epoch=0):
        """
        Specialized training for Deep Ensemble models.
        Trains all ensemble members simultaneously or sequentially based on config.
        """
        model.train()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get('cluster', False)
        
        # Get ensemble training configuration
        ensemble_config = self.config['model'].get('ensemble_training', {})
        train_method = ensemble_config.get('method', 'simultaneous')  # 'simultaneous' or 'sequential'
        
        epoch_start_time = self._sync_and_time()
        
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader),
                                         disable=disable_tqdm, desc="Ensemble Training"):
            
            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = self._targets_to_training_space(targets)
            
            optimizer.zero_grad()
            total_loss = 0.0
            
            if train_method == 'simultaneous':
                # Train all ensemble members with same data simultaneously
                outputs = model(inputs)  # This calls the ensemble forward method
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)
                
                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()
                
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)
                
                current_kl_weight = self.get_current_kl_weight(epoch)
                
                if self.config['training']['loss_function'] == 'GaussianNLLLoss':
                    loss = nll_loss + current_kl_weight * kld_loss
                elif self.config['training']['loss_function'] == 'MSELoss':
                    loss = mse_loss
                
                total_loss = loss
                
            elif train_method == 'sequential':
                # Train each ensemble member with the same batch sequentially
                all_predictions, all_variances = model.forward_individual(inputs)
                
                for member_idx, (pred_mean, pred_var) in enumerate(zip(all_predictions, all_variances)):
                    pred_mean = pred_mean.flatten()
                    pred_var = pred_var.flatten()
                    
                    mse_loss = criterion_mse(pred_mean, training_targets)
                    nll_loss = criterion_nll(pred_mean, training_targets, pred_var)
                    
                    # For ensemble, KL loss only applies to BNN members (not relevant for MLP ensemble)
                    kld_loss = torch.tensor(0.0, device=self.device)
                    
                    if self.config['training']['loss_function'] == 'GaussianNLLLoss':
                        member_loss = nll_loss
                    elif self.config['training']['loss_function'] == 'MSELoss':
                        member_loss = mse_loss
                    
                    total_loss += member_loss
                
                # Average loss across ensemble members
                total_loss = total_loss / len(all_predictions)
            
            total_loss.backward()
            optimizer.step()

            # Accumulate losses for logging
            running_loss += total_loss.item()
            if train_method == 'simultaneous':
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()
            
            # Store outputs for metrics calculation
            if train_method == 'simultaneous':
                if self.use_log_target:
                    point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)
                all_outputs.append(point_linear.cpu())
            else:
                # For sequential training, use ensemble mean for metrics
                ensemble_mean = torch.mean(torch.stack(all_predictions), dim=0).flatten()
                if self.use_log_target:
                    point_linear, _, _ = self._pred_log_to_linear(ensemble_mean, torch.zeros_like(ensemble_mean))
                else:
                    point_linear, _, _ = self._pred_linear_from_linear(ensemble_mean, torch.zeros_like(ensemble_mean))
                all_outputs.append(point_linear.cpu())
            
            all_targets.append(original_targets.cpu())

            if self.config.get('debug_single_batch', False):
                break

        # Calculate epoch metrics
        epoch_end_time = self._sync_and_time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        if self.timing_enabled:
            self._log_timing(f"Ensemble training epoch {epoch}", epoch_duration)

        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader) if train_method == 'simultaneous' else 0.0
        avg_nll = running_nll / len(dataloader) if train_method == 'simultaneous' else 0.0
        avg_kld = running_kld / len(dataloader) if train_method == 'simultaneous' else 0.0

        all_outputs_tensor = torch.cat(all_outputs, dim=0).detach()
        all_targets_tensor = torch.cat(all_targets, dim=0)
        train_metrics = calculate_metrics(all_outputs_tensor, all_targets_tensor, prefix="train")
        
        # Convert to numpy for compatibility with training loop
        all_outputs_numpy = all_outputs_tensor.cpu().numpy()
        all_targets_numpy = all_targets_tensor.cpu().numpy()
        train_metrics['predictions'] = all_outputs_numpy
        train_metrics['targets'] = all_targets_numpy

        return avg_loss, avg_mse, avg_nll, avg_kld, train_metrics

    def validate_epoch_ensemble(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0):
        """
        Specialized validation for Deep Ensemble models.
        """
        model.eval()
        running_loss = running_mse = running_nll = running_kld = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get('cluster', False)
        
        val_start_time = self._sync_and_time()

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Ensemble Validation", disable=disable_tqdm):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = self._targets_to_training_space(targets)

                # Get ensemble prediction (aggregated)
                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)
                
                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Losses in training space
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = torch.tensor(0.0, device=self.device)  # No KL loss for MLP ensembles
                
                if self.config['training']['loss_function'] == 'GaussianNLLLoss':
                    loss = nll_loss
                elif self.config['training']['loss_function'] == 'MSELoss':
                    loss = mse_loss

                # Accumulate losses
                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

                # Back-transform to linear space for metrics/logging
                if self.use_log_target:
                    point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

                all_outputs.append(point_linear.cpu())
                all_targets.append(original_targets.cpu())

                if self.config.get('debug_single_batch', False):
                    break

        # Calculate epoch metrics
        val_end_time = self._sync_and_time()
        val_duration = val_end_time - val_start_time
        
        if self.timing_enabled:
            self._log_timing(f"Ensemble validation epoch {epoch}", val_duration)

        n_batches = len(dataloader)
        avg_loss = running_loss / n_batches
        avg_mse = running_mse / n_batches
        avg_nll = running_nll / n_batches
        avg_kld = running_kld / n_batches

        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)
        val_metrics = calculate_metrics(all_outputs_tensor, all_targets_tensor, prefix="val")
        
        # Convert to numpy for compatibility
        all_outputs_numpy = all_outputs_tensor.cpu().numpy()
        all_targets_numpy = all_targets_tensor.cpu().numpy()
        val_metrics['predictions'] = all_outputs_numpy
        val_metrics['targets'] = all_targets_numpy

        return avg_loss, avg_mse, avg_nll, avg_kld, val_metrics

    def get_current_kl_weight(self, epoch):
        """Compute current KL weight based on annealing schedule"""
        if not self.use_kl_annealing:
            return self.loss_weight
        
        if epoch < self.kl_warmup_epochs:
            # Linear annealing from start_weight to end_weight
            progress = epoch / self.kl_warmup_epochs
            current_weight = self.kl_start_weight + progress * (self.kl_end_weight - self.kl_start_weight)
        else:
            # After warmup, use full weight
            current_weight = self.kl_end_weight
        
        return current_weight

    # ---------- training / validation ----------

    def train_epoch(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, optimizer, epoch=0):
        model.train()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get('cluster', False)
        
        # Timing instrumentation with proper synchronization
        epoch_start_time = self._sync_and_time()
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
        iter_start_time = self._sync_and_time()

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader),
                                         disable=disable_tqdm, desc="Training"):
            # Measure data loading time (time between iterations)
            data_ready_time = self._sync_and_time()
            if i > 0:  # Skip first iteration since there's no previous iteration
                data_load_time += (data_ready_time - iter_start_time)
        
            batch_start = self._sync_and_time()
            
            # Time data transfer to GPU
            transfer_start = self._sync_and_time()
            inputs = inputs.to(self.device, non_blocking=True)
            training_targets, original_targets = self._targets_to_training_space(targets)
            transfer_end = self._sync_and_time()
            gpu_transfer_time += (transfer_end - transfer_start)

            optimizer.zero_grad()

            # Time forward pass
            forward_start = self._sync_and_time()
            outputs = model(inputs)
            pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)
            
            pred_mean_raw = pred_mean_raw.flatten()
            pred_var_raw = pred_var_raw.flatten()
            forward_end = self._sync_and_time()
            forward_time += (forward_end - forward_start)

            # Time loss computation
            loss_start = self._sync_and_time()
            mse_loss = criterion_mse(pred_mean_raw, training_targets)
            nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
            kld_loss = criterion_kld(model)
            
            # Use annealed KL weight
            current_kl_weight = self.get_current_kl_weight(epoch)
            
            if self.config['training']['loss_function'] == 'GaussianNLLLoss':
                loss = nll_loss + current_kl_weight * kld_loss
            elif self.config['training']['loss_function'] == 'MSELoss':
                loss = mse_loss
            loss_end = self._sync_and_time()
            loss_compute_time += (loss_end - loss_start)

            # Time backward pass
            backward_start = self._sync_and_time()
            loss.backward()
            backward_end = self._sync_and_time()
            backward_time += (backward_end - backward_start)
            
            # Time optimizer step
            opt_start = self._sync_and_time()
            optimizer.step()
            opt_end = self._sync_and_time()
            optimizer_time += (opt_end - opt_start)

            # Time post-processing
            postprocess_start = self._sync_and_time()
            # Accumulate losses
            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_nll += nll_loss.item()
            running_kld += kld_loss.item()

            # Back-transform to linear space for metrics/logging
            if self.use_log_target:
                point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
            else:
                point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

            # Track average variance in ORIGINAL space (more interpretable)
            running_variance += torch.mean(var_linear).item()

            # Store outputs for metrics (pred, std) and original targets
            all_outputs.append(torch.stack([point_linear, std_linear], dim=1).detach().cpu())
            all_targets.append(original_targets.detach().cpu())
            postprocess_end = self._sync_and_time()
            postprocess_time += (postprocess_end - postprocess_start)
            
            batch_count += 1
            
            # Calculate TQDM and other overhead
            batch_end = self._sync_and_time()
            batch_total_measured = batch_end - batch_start
            batch_components_sum = (transfer_end - transfer_start) + (forward_end - forward_start) + \
                              (loss_end - loss_start) + (backward_end - backward_start) + \
                              (opt_end - opt_start) + (postprocess_end - postprocess_start)
            tqdm_overhead += (batch_total_measured - batch_components_sum)
            
            # Log timing for first few batches if detailed timing is enabled
            if self.detailed_batch_timing and i < 3:
                data_load_for_batch = (data_ready_time - iter_start_time) if i > 0 else 0.0
                self.logger.info(f"  Batch {i+1}: data_load={data_load_for_batch:.3f}s, total={batch_total_measured:.3f}s, components={batch_components_sum:.3f}s")
            
            # Log periodic timing updates if configured
            elif self.timing_enabled and self.log_timing_frequency > 0 and (i + 1) % self.log_timing_frequency == 0:
                data_load_for_batch = (data_ready_time - iter_start_time) if i > 0 else 0.0
                self.logger.info(f"  Batch {i+1}/{len(dataloader)}: data_load={data_load_for_batch:.3f}s, total={batch_total_measured:.3f}s")
        
            # Prepare for next iteration timing
            iter_start_time = self._sync_and_time()

        # Log epoch timing summary with synchronized end time
        epoch_total = self._sync_and_time() - epoch_start_time
        components_sum = gpu_transfer_time + forward_time + loss_compute_time + backward_time + optimizer_time + postprocess_time
        unaccounted_time = epoch_total - components_sum - data_load_time - tqdm_overhead
        
        if self.timing_enabled:
            self._log_timing("Train Epoch Total", epoch_total, f"{batch_count} batches")
            self._log_timing("Train Data Loading", data_load_time, f"avg={data_load_time/(batch_count-1 if batch_count > 1 else 1):.3f}s/batch")
            self._log_timing("Train Data Transfer", gpu_transfer_time, f"avg={gpu_transfer_time/batch_count:.3f}s/batch")
            self._log_timing("Train Forward Pass", forward_time, f"avg={forward_time/batch_count:.3f}s/batch")
            self._log_timing("Train Loss Computation", loss_compute_time, f"avg={loss_compute_time/batch_count:.3f}s/batch")
            self._log_timing("Train Backward Pass", backward_time, f"avg={backward_time/batch_count:.3f}s/batch")
            self._log_timing("Train Optimizer Step", optimizer_time, f"avg={optimizer_time/batch_count:.3f}s/batch")
            self._log_timing("Train Post-processing", postprocess_time, f"avg={postprocess_time/batch_count:.3f}s/batch")
            self._log_timing("Train TQDM/Loop Overhead", tqdm_overhead, f"avg={tqdm_overhead/batch_count:.3f}s/batch")
            self.logger.info(f"⏱️  Components sum: {components_sum:.3f}s, Data loading: {data_load_time:.3f}s, Still unaccounted: {unaccounted_time:.3f}s")

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        n_batches = len(dataloader)
        return (running_loss / n_batches,
                running_mse / n_batches,
                running_nll / n_batches,
                running_kld / n_batches,
                running_variance / n_batches,
                all_outputs, all_targets)

    def validate_epoch(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0):
        model.eval()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get('cluster', False)
        
        # Timing instrumentation for validation with proper synchronization
        val_start_time = self._sync_and_time()
        val_batch_count = 0

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation", disable=disable_tqdm):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = self._targets_to_training_space(targets)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)
                
                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Losses in training space
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)
                
                # Use same annealed KL weight as training
                current_kl_weight = self.get_current_kl_weight(epoch)
                
                # Use same loss calculation logic as training
                if self.config['training']['loss_function'] == 'GaussianNLLLoss':
                    loss = nll_loss + current_kl_weight * kld_loss
                elif self.config['training']['loss_function'] == 'MSELoss':
                    loss = mse_loss

                # Accumulate losses
                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

                # Back-transform to linear space for metrics/logging
                if self.use_log_target:
                    point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

                # Track average variance in ORIGINAL space
                running_variance += torch.mean(var_linear).item()

                # Store outputs for metrics (pred, std) and original targets
                all_outputs.append(torch.stack([point_linear, std_linear], dim=1).detach().cpu())
                all_targets.append(original_targets.detach().cpu())
                
                val_batch_count += 1

        # Log validation timing with proper synchronization
        val_total = self._sync_and_time() - val_start_time
        if self.timing_enabled:
            self._log_timing("Validation Epoch Total", val_total, f"{val_batch_count} batches")

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        n_batches = len(dataloader)
        return (running_loss / n_batches,
                running_mse / n_batches,
                running_nll / n_batches,
                running_kld / n_batches,
                running_variance / n_batches,
                all_outputs, all_targets)

    def test_model(self, model, dataloader):
        model.eval()
        all_outputs, all_targets = [], []
        disable_tqdm = self.config.get('cluster', False)

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Testing", disable=disable_tqdm):
                inputs = inputs.to(self.device, non_blocking=True)
                training_targets, original_targets = self._targets_to_training_space(targets)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)
                
                pred_mean_raw = pred_mean_raw.flatten()
                pred_var_raw = pred_var_raw.flatten()

                # Back-transform to linear space for outputs
                if self.use_log_target:
                    point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

                # Store outputs for metrics (pred, std) and original targets
                all_outputs.append(torch.stack([point_linear, std_linear], dim=1).detach().cpu())
                all_targets.append(original_targets.detach().cpu())

        return torch.cat(all_outputs), torch.cat(all_targets)

    # ---------- feature inverse-transform ----------

    def inverse_transform_features(self, x):
        """Transform normalized features back to original scale using feature registry."""
        # Get output indices from the feature registry (these map to the transformed feature vector)
        output_indices = self.feature_registry._output_indices
        
        rescaled_features = {}
        
        # Process each feature type
        for feature_type in [FeatureType.TEMPORAL, FeatureType.STATION, FeatureType.DIRECTION, 
                           FeatureType.IPP, FeatureType.SWI]:
            
            feature_names = self.feature_registry.get_feature_names(feature_type)
            
            for feature_name in feature_names:
                if feature_type == FeatureType.TEMPORAL:
                    if feature_name == 'year':
                        norm_idx = output_indices[f'{feature_name}_norm']
                        rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                            feature_name, x[:, norm_idx]
                        )
                    elif feature_name in ['doy', 'sod']:
                        # For cyclic features, we use the normalized version for inverse transform
                        norm_idx = output_indices[f'{feature_name}_norm']
                        rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                            feature_name, x[:, norm_idx]
                        )
                
                elif feature_type == FeatureType.STATION:
                    norm_idx = output_indices[f'{feature_name}_norm']
                    rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                        feature_name, x[:, norm_idx]
                    )
                
                elif feature_type == FeatureType.DIRECTION:
                    if feature_name == 'satazi':
                        # For azimuth, reconstruct from sin/cos components
                        sin_idx = output_indices[f'{feature_name}_sin'] 
                        cos_idx = output_indices[f'{feature_name}_cos']
                        azi_rad = torch.atan2(x[:, sin_idx], x[:, cos_idx])
                        azi_deg = (azi_rad * 180 / torch.pi) % 360
                        rescaled_features[feature_name] = azi_deg
                    elif feature_name == 'satele':
                        norm_idx = output_indices[f'{feature_name}_norm']
                        rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                            feature_name, x[:, norm_idx]
                        )
                
                elif feature_type == FeatureType.IPP:
                    norm_idx = output_indices[f'{feature_name}_norm']
                    rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                        feature_name, x[:, norm_idx]
                    )
                
                elif feature_type == FeatureType.SWI:
                    norm_idx = output_indices[f'{feature_name}_norm']
                    rescaled_features[feature_name] = self.feature_registry.denormalize_feature(
                        feature_name, x[:, norm_idx]
                    )
        
        # Convert to tensor and return in a consistent order
        feature_list = []
        feature_order = []
        
        # Add features in registry order
        for feature_name in self.feature_registry.get_all_enabled_features():
            if feature_name in rescaled_features and feature_name not in self.feature_registry.get_features_by_type(FeatureType.TARGET):
                feature_list.append(rescaled_features[feature_name].unsqueeze(1))
                feature_order.append(feature_name)
        
        if feature_list:
            return torch.cat(feature_list, dim=1), feature_order
        else:
            return torch.empty(x.size(0), 0), []

    def get_feature_indices(self):
        """Get a mapping of feature names to their column indices in the inverse-transformed data."""
        # Get all enabled features excluding targets
        all_enabled = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)
        input_features = [f for f in all_enabled if f not in target_features]
        
        # Create mapping
        indices = {}
        for idx, feature_name in enumerate(input_features):
            indices[feature_name] = idx
            
        return indices

    # ---------- Bayesian inference with proper aggregation in ORIGINAL space ----------

    def bayesian_inference_total_uncertainty(self, model, dataloader, num_samples=100):
        """
        Compute predictive mean and total uncertainty (epistemic + aleatoric) in ORIGINAL space.
        For log-targets, each forward pass is mapped via log-normal moments:
            mean_y_s = exp(mu + 0.5*sigma2) - eps
            var_y_alea_s = (exp(sigma2) - 1) * exp(2*mu + sigma2)
        Then aggregate:
            epistemic_var = Var_s(mean_y_s)
            aleatoric_var = E_s(var_y_alea_s)
        """
        model.eval()

        final_df = pd.DataFrame()
        batch_means = []
        batch_epistemic_vars = []
        batch_aleatoric_vars = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Bayesian Inference",
                                        disable=self.config.get('cluster', False)):
                bs = inputs.size(0)
                inputs = inputs.to(self.device, non_blocking=True)

                # Check if this is an ensemble model
                model_type = self.config['model']['model_type']
                if model_type == 'DE_MLP':
                    # For ensemble models, use the decomposed uncertainty method
                    ensemble_mean, aleatoric_var, epistemic_var, total_var = model.get_uncertainties(inputs)
                    
                    # Move to CPU and handle shapes
                    stec_mean = ensemble_mean.cpu().flatten()
                    epistemic_var = epistemic_var.cpu().flatten()
                    aleatoric_var = aleatoric_var.cpu().flatten()
                    
                    batch_means.append(stec_mean)
                    batch_epistemic_vars.append(epistemic_var)
                    batch_aleatoric_vars.append(aleatoric_var)
                else:
                    # Original BNN sampling logic
                    per_sample_means = []
                    per_sample_alea_vars = []

                    for _ in range(num_samples):
                        outputs = model(inputs)
                        mean_raw, var_raw = self.compute_mean_var(outputs)

                        if self.use_log_target:
                            # Log-normal moments for this pass
                            mean_y = torch.exp(mean_raw + 0.5 * var_raw) - self.eps
                            var_alea_y = (torch.exp(var_raw) - 1.0) * torch.exp(2 * mean_raw + var_raw)
                        else:
                            mean_y = mean_raw
                            var_alea_y = var_raw

                        per_sample_means.append(mean_y.cpu())
                        per_sample_alea_vars.append(var_alea_y.cpu())

                    pred_stack = torch.stack(per_sample_means, dim=0)        # [S, B]
                    alea_var_stack = torch.stack(per_sample_alea_vars, dim=0) # [S, B]

                    stec_mean = pred_stack.mean(dim=0)
                    if num_samples == 1:
                        epistemic_var = torch.zeros_like(pred_stack[0])  # No epistemic uncertainty
                    else:
                        epistemic_var = pred_stack.var(dim=0)             # Var over means
                    aleatoric_var = alea_var_stack.mean(dim=0)            # Mean aleatoric var
                    batch_means.append(stec_mean)
                    batch_epistemic_vars.append(epistemic_var)
                    batch_aleatoric_vars.append(aleatoric_var)
                all_targets.append(targets.cpu())

                # Build per-batch DF using feature registry
                inputs_original, feature_order = self.inverse_transform_features(inputs)

                batch_df = pd.DataFrame(
                    torch.cat([
                        inputs_original.cpu(),
                        targets.cpu().view(bs, -1),
                        stec_mean.cpu().view(bs, -1),
                        torch.sqrt(epistemic_var).cpu().view(bs, -1),
                        torch.sqrt(aleatoric_var).cpu().view(bs, -1),
                        torch.sqrt(epistemic_var + aleatoric_var).cpu().view(bs, -1)
                    ], dim=1).numpy(),
                    columns=[
                        *feature_order,
                        'target_stec', 'pred_stec',
                        'pred_epistemic_unc', 'pred_aleatoric_unc', 'pred_total_unc'
                    ]
                )
                final_df = pd.concat([final_df, batch_df], ignore_index=True)

        mean = torch.cat(batch_means).squeeze()
        epistemic_var = torch.cat(batch_epistemic_vars)
        aleatoric_var = torch.cat(batch_aleatoric_vars)
        targets = torch.cat(all_targets)

        total_var = epistemic_var + aleatoric_var
        total_std = torch.sqrt(total_var)

        return {
            'baysian_mae': torch.mean(torch.abs(mean - targets)),
            'baysian_mse': torch.mean((mean - targets) ** 2),
            'mean': mean,
            'epistemic_std': torch.sqrt(epistemic_var),
            'aleatoric_std': torch.sqrt(aleatoric_var),
            'total_std': total_std,
            'targets': targets
        }, final_df

    # ---------- training driver ----------

    def save_checkpoint(self, config, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir, model_seed):
        self.logger.info(f"Validation loss improved from {best_loss:.2f} to {val_loss:.2f}. Saving checkpoint.")
        if config['mode'] == 'finetune':
            filename = f"{config['mode']}_{config['model']['model_type']}_seed{model_seed:02}.pth"
        elif config['mode'] == 'pretrain':
            filename = f"{config['mode']}_{config['model']['model_type']}_seed{model_seed:02}.pth"
        filepath = os.path.join(checkpoint_dir, filename)
        torch.save({
            'epoch': epoch,
            'model_type': config['model']['model_type'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Save loss history with checkpoint
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'epochs_tracked': self.epochs_tracked,
        }, filepath)
        return val_loss

    def track_losses(self, epoch, train_loss, val_loss):
        """Track training and validation losses for plotting."""
        self.epochs_tracked.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def plot_loss_curve(self, output_dir):
        """Plot and save the loss curve."""
        if not self.train_losses or not self.val_losses:
            self.logger.warning("No loss data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs_tracked, self.train_losses, label='Training Loss', color='blue')
        plt.plot(self.epochs_tracked, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        loss_plot_path = os.path.join(output_dir, f"loss_curve.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs_tracked, self.train_losses, label='Training Loss', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path_train = os.path.join(output_dir, f"train_loss_curve.png")
        plt.savefig(loss_plot_path_train, dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs_tracked, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path_val = os.path.join(output_dir, f"val_loss_curve.png")
        plt.savefig(loss_plot_path_val, dpi=300, bbox_inches='tight')
        plt.close()

    def save_final_losses(self, output_dir):
        """Save final training results including loss curve."""
        # Plot the loss curve
        self.plot_loss_curve(output_dir)
        
        # Optionally save loss data as CSV for further analysis
        if self.train_losses and self.val_losses:
            loss_data = pd.DataFrame({
                'epoch': self.epochs_tracked,
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            })
            csv_path = os.path.join(output_dir, f"loss_history.csv")
            loss_data.to_csv(csv_path, index=False)

    def split_test_data_by_date(self, test_df):
        """
        Split test dataframe into interpolation and extrapolation subsets.
        Simple rule: May 2024 and later = extrapolation, everything before = interpolation.
        
        Args:
            test_df: Test dataframe with 'year' and 'doy' columns
        
        Returns:
            tuple: (interpolation_df, extrapolation_df, split_info)
        """
        if 'year' not in test_df.columns or 'doy' not in test_df.columns:
            self.logger.warning("Year or DOY columns not found in test data. Cannot split by date.")
            return test_df, pd.DataFrame(), {}
        
        # Create datetime from year and doy
        def create_date(row):
            try:
                year = int(row['year'])
                doy = int(row['doy'])
                date = datetime(year, 1, 1) + timedelta(days=doy - 1)
                return date
            except:
                return None
        
        test_df = test_df.copy()
        test_df['date'] = test_df.apply(create_date, axis=1)
        test_df = test_df.dropna(subset=['date'])
        
        # Extract year-month periods for comparison
        test_df['year_month'] = test_df['date'].dt.to_period('M')
        
        # Simple split: May 2024 and later = extrapolation, everything before = interpolation
        cutoff_period = pd.Period('2024-05')
        
        interpolation_mask = test_df['year_month'] < cutoff_period
        extrapolation_mask = test_df['year_month'] >= cutoff_period
        
        interpolation_df = test_df[interpolation_mask].copy().reset_index(drop=True)
        extrapolation_df = test_df[extrapolation_mask].copy().reset_index(drop=True)
        
        # Create split information summary
        interpolation_months = sorted(interpolation_df['year_month'].unique()) if len(interpolation_df) > 0 else []
        extrapolation_months = sorted(extrapolation_df['year_month'].unique()) if len(extrapolation_df) > 0 else []
        
        split_info = {
            'total_samples': len(test_df),
            'interpolation_samples': len(interpolation_df),
            'extrapolation_samples': len(extrapolation_df),
            'interpolation_months': [str(m) for m in interpolation_months],
            'extrapolation_months': [str(m) for m in extrapolation_months],
            'interpolation_percentage': (len(interpolation_df) / len(test_df)) * 100 if len(test_df) > 0 else 0,
            'extrapolation_percentage': (len(extrapolation_df) / len(test_df)) * 100 if len(test_df) > 0 else 0,
            'cutoff_date': str(cutoff_period)
        }
        
        return interpolation_df, extrapolation_df, split_info

    def save_temporal_split_metrics(self, interpolation_df, extrapolation_df, split_info, experiment_dir):
        """
        Calculate and save metrics for interpolation/extrapolation splits.
        
        Args:
            interpolation_df: Test data before May 2024 (interpolation)
            extrapolation_df: Test data May 2024 and later (extrapolation)
            split_info: Dictionary with split information
            experiment_dir: Experiment directory path
        """
        # Create test_metrics subdirectories
        test_metrics_dir = os.path.join(experiment_dir, 'test_metrics')
        interpolation_dir = os.path.join(test_metrics_dir, 'interpolation')
        extrapolation_dir = os.path.join(test_metrics_dir, 'extrapolation')
        
        os.makedirs(interpolation_dir, exist_ok=True)
        os.makedirs(extrapolation_dir, exist_ok=True)
        
        # Calculate metrics for each subset
        metrics_summary = {}
        
        if len(interpolation_df) > 0:
            # Convert dataframe to tensors for metrics calculation
            interpolation_predictions = torch.stack([
                torch.tensor(interpolation_df['pred_stec'].values, dtype=torch.float32),
                torch.tensor(interpolation_df['pred_total_unc'].values, dtype=torch.float32)
            ], dim=1)
            interpolation_targets = torch.tensor(interpolation_df['target_stec'].values, dtype=torch.float32)
            
            interpolation_metrics = calculate_metrics(interpolation_predictions, interpolation_targets, prefix="interpolation")
            metrics_summary['interpolation'] = interpolation_metrics
            
            # Save interpolation period summary
            interpolation_summary_path = os.path.join(interpolation_dir, 'metrics_summary.txt')
            with open(interpolation_summary_path, 'w') as f:
                f.write("METRICS FOR INTERPOLATION (BEFORE MAY 2024)\n")
                f.write("=" * 60 + "\n\n")
                f.write("This includes test months before May 2024.\n")
                f.write("These are months within or close to the training period.\n\n")
                f.write(f"Cutoff date: {split_info['cutoff_date']}\n")
                f.write(f"Number of samples: {len(interpolation_df):,}\n")
                f.write(f"Percentage of total test data: {split_info['interpolation_percentage']:.1f}%\n")
                f.write(f"Months included: {', '.join(split_info['interpolation_months'])}\n\n")
                f.write("METRICS:\n")
                f.write("-" * 20 + "\n")
                for k, v in interpolation_metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
            
            mae_value = interpolation_metrics.get('interpolation_MAE', 'N/A')
            mae_str = f"{mae_value:.4f}" if isinstance(mae_value, (int, float)) else str(mae_value)
            self.logger.info(f"Interpolation - Samples: {len(interpolation_df):,}, MAE: {mae_str}")
        
        if len(extrapolation_df) > 0:
            # Convert dataframe to tensors for metrics calculation  
            extrapolation_predictions = torch.stack([
                torch.tensor(extrapolation_df['pred_stec'].values, dtype=torch.float32),
                torch.tensor(extrapolation_df['pred_total_unc'].values, dtype=torch.float32)
            ], dim=1)
            extrapolation_targets = torch.tensor(extrapolation_df['target_stec'].values, dtype=torch.float32)
            
            extrapolation_metrics = calculate_metrics(extrapolation_predictions, extrapolation_targets, prefix="extrapolation")
            metrics_summary['extrapolation'] = extrapolation_metrics
            
            # Save extrapolation period summary
            extrapolation_summary_path = os.path.join(extrapolation_dir, 'metrics_summary.txt')
            with open(extrapolation_summary_path, 'w') as f:
                f.write("METRICS FOR EXTRAPOLATION (MAY 2024 AND LATER)\n")
                f.write("=" * 60 + "\n\n")
                f.write("This includes test months from May 2024 onwards.\n")
                f.write("These are true forecasting/extrapolation months.\n\n")
                f.write(f"Cutoff date: {split_info['cutoff_date']}\n")
                f.write(f"Number of samples: {len(extrapolation_df):,}\n")
                f.write(f"Percentage of total test data: {split_info['extrapolation_percentage']:.1f}%\n")
                f.write(f"Months included: {', '.join(split_info['extrapolation_months'])}\n\n")
                f.write("METRICS:\n")
                f.write("-" * 20 + "\n")
                for k, v in extrapolation_metrics.items():
                    f.write(f"{k}: {v:.4f}\n")
            
            mae_value = extrapolation_metrics.get('extrapolation_MAE', 'N/A')
            mae_str = f"{mae_value:.4f}" if isinstance(mae_value, (int, float)) else str(mae_value)
            self.logger.info(f"Extrapolation - Samples: {len(extrapolation_df):,}, MAE: {mae_str}")
        
        # Save combined temporal split summary
        split_summary_path = os.path.join(test_metrics_dir, 'temporal_split_summary.txt')
        with open(split_summary_path, 'w') as f:
            f.write("TEMPORAL SPLIT ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Split cutoff: {split_info['cutoff_date']}\n")
            f.write(f"Total test samples: {split_info['total_samples']:,}\n\n")
            
            f.write("INTERPOLATION (BEFORE MAY 2024):\n")
            f.write("-" * 35 + "\n")
            f.write(f"Samples: {split_info['interpolation_samples']:,} ({split_info['interpolation_percentage']:.1f}%)\n")
            f.write(f"Months: {', '.join(split_info['interpolation_months'])}\n")
            if 'interpolation' in metrics_summary:
                f.write("Key Metrics:\n")
                for k, v in metrics_summary['interpolation'].items():
                    if any(metric in k.lower() for metric in ['mae', 'mse', 'rmse']):
                        f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
            
            f.write("EXTRAPOLATION (MAY 2024 AND LATER):\n")
            f.write("-" * 38 + "\n")
            f.write(f"Samples: {split_info['extrapolation_samples']:,} ({split_info['extrapolation_percentage']:.1f}%)\n")
            f.write(f"Months: {', '.join(split_info['extrapolation_months'])}\n")
            if 'extrapolation' in metrics_summary:
                f.write("Key Metrics:\n")
                for k, v in metrics_summary['extrapolation'].items():
                    if any(metric in k.lower() for metric in ['mae', 'mse', 'rmse']):
                        f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
            
            # Performance comparison if both subsets exist
            if 'interpolation' in metrics_summary and 'extrapolation' in metrics_summary:
                f.write("PERFORMANCE COMPARISON:\n")
                f.write("-" * 25 + "\n")
                for metric in ['MAE', 'MSE', 'RMSE']:
                    interpolation_key = f"interpolation_{metric}"
                    extrapolation_key = f"extrapolation_{metric}"
                    if interpolation_key in metrics_summary['interpolation'] and extrapolation_key in metrics_summary['extrapolation']:
                        interpolation_val = metrics_summary['interpolation'][interpolation_key]
                        extrapolation_val = metrics_summary['extrapolation'][extrapolation_key]
                        diff = extrapolation_val - interpolation_val
                        pct_change = (diff / interpolation_val) * 100 if interpolation_val != 0 else 0
                        f.write(f"{metric}:\n")
                        f.write(f"  Interpolation: {interpolation_val:.4f}\n")
                        f.write(f"  Extrapolation: {extrapolation_val:.4f}\n")
                        f.write(f"  Difference: {diff:+.4f} ({pct_change:+.1f}%)\n\n")
        
        return metrics_summary

    def run_training(self, train_loader, val_loader, test_loader, init_model_fn, training_key):
        """
        General ensemble training loop.
        
        Parameters:
          - train_loader, val_loader, test_loader: Dataloaders for training/validation/testing.
          - init_model_fn: Function that takes (seed) and returns an initialized model.
          - training_key: String key to choose the training configuration, e.g. "finetune" or "pretrain".
        """
        seed = self.config['random_seed']
        model_dir = os.path.join(self.config['output_dir'], 'model')
        os.makedirs(model_dir, exist_ok=True)

        self.logger.info("Training model...")
        
        # Time overall training setup with proper synchronization
        setup_start = self._sync_and_time()

        if not self.config["debug"]:
            # Use the sweep-aware wandb setup
            from utils.wandb_sweep_integration import setup_wandb_for_sweep
            experiment_name = os.path.basename(self.config['output_dir'])
            setup_wandb_for_sweep(self.config, experiment_name)

        # Time model initialization
        model_init_start = self._sync_and_time()
        model = init_model_fn(seed)
        if self.timing_enabled:
            self._log_timing("Model Initialization", self._sync_and_time() - model_init_start)
            
        # Time criterion and optimizer setup
        criterion_start = self._sync_and_time()
        criterion_mse = get_criterion(self.config, "MSELoss")
        criterion_nll = get_criterion(self.config, "GaussianNLLLoss")
        criterion_kld = get_criterion(self.config, "BKLLoss")
        optimizer = get_optimizer(self.config, model.parameters())

        scheduler = None
        if self.config[training_key]["scheduler"]:
            scheduler = get_scheduler(self.config, optimizer)
        
        if self.timing_enabled:
            self._log_timing("Criterion & Optimizer Setup", self._sync_and_time() - criterion_start)
            self._log_timing("Total Training Setup", self._sync_and_time() - setup_start)

        best_val_loss = float('inf')
        patience_counter = 0
        epochs = self.config[training_key]["epochs"]

        for epoch in range(epochs):
            gc.collect()
            print(" ")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Update sampler epoch for different data sampling each epoch
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Time data loading and training phases with proper synchronization
            epoch_start = self._sync_and_time()

            train_start = self._sync_and_time()
            
            # Check if model is an ensemble and use appropriate training method
            model_type = self.config['model']['model_type']
            if model_type == 'DE_MLP':
                train_loss, train_mse, train_nll, train_kld, train_metrics = \
                    self.train_epoch_ensemble(model, train_loader, criterion_mse, criterion_nll, criterion_kld, optimizer, epoch)
                # Extract outputs and targets for compatibility
                train_outputs = train_metrics.get('predictions', [])
                train_targets = train_metrics.get('targets', [])
                train_variance = 0.0  # Ensemble handles variance internally
            else:
                train_loss, train_mse, train_nll, train_kld, train_variance, train_outputs, train_targets = \
                    self.train_epoch(model, train_loader, criterion_mse, criterion_nll, criterion_kld, optimizer, epoch)
                train_metrics = calculate_metrics(train_outputs, train_targets, prefix="train")
            
            train_duration = self._sync_and_time() - train_start
            
            val_start = self._sync_and_time()
            
            # Use appropriate validation method for ensemble models
            if model_type == 'DE_MLP':
                val_loss, val_mse, val_nll, val_kld, val_metrics = \
                    self.validate_epoch_ensemble(model, val_loader, criterion_mse, criterion_nll, criterion_kld, epoch)
                # Extract outputs and targets for compatibility
                val_outputs = val_metrics.get('predictions', [])
                val_targets = val_metrics.get('targets', [])
                val_variance = 0.0  # Ensemble handles variance internally
            else:
                val_loss, val_mse, val_nll, val_kld, val_variance, val_outputs, val_targets = \
                    self.validate_epoch(model, val_loader, criterion_mse, criterion_nll, criterion_kld, epoch)
                val_metrics = calculate_metrics(val_outputs, val_targets, prefix="val")
            
            val_duration = self._sync_and_time() - val_start

            # Track losses for plotting
            self.track_losses(epoch + 1, train_loss, val_loss)
            
            # Time metrics calculation
            metrics_start = self._sync_and_time()
            # Metrics already calculated in ensemble methods, just use them
            if model_type != 'DE_MLP':
                train_metrics = calculate_metrics(train_outputs, train_targets, prefix="train")
                val_metrics = calculate_metrics(val_outputs, val_targets, prefix="val")
            metrics_duration = self._sync_and_time() - metrics_start

            if not self.config["debug"]:
                wandb.log({
                    'train_loss': train_loss,
                    'train_mse': train_mse,
                    'train_nll': train_nll,
                    'train_kld': train_kld,
                    'train_variance': train_variance,
                    'val_loss': val_loss,
                    'val_mse': val_mse,
                    'val_nll': val_nll,
                    'val_kld': val_kld,
                    'val_variance': val_variance,
                    'learning_rate': scheduler.get_last_lr()[0] if scheduler else None,
                    'kl_weight': self.get_current_kl_weight(epoch) if self.use_kl_annealing else self.loss_weight,
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch + 1
                })

            if scheduler:
                scheduler.step()

            self.logger.info(f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")
            
            # Log epoch timing summary with synchronized timing
            epoch_total = self._sync_and_time() - epoch_start
            if self.timing_enabled:
                self.logger.info(f"⏱️  Epoch {epoch+1} timing: total={epoch_total:.1f}s, train={train_duration:.1f}s, val={val_duration:.1f}s, metrics={metrics_duration:.3f}s")

            if val_loss < best_val_loss or self.config[training_key]["save_model_every_epoch"]:
                checkpoint_start = self._sync_and_time()
                best_val_loss = self.save_checkpoint(self.config, model, optimizer, epoch, 
                                                   val_loss, best_val_loss, model_dir, seed)
                if self.timing_enabled:
                    self._log_timing("Checkpoint Save", self._sync_and_time() - checkpoint_start)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config[training_key].get("patience", float('inf')):
                    self.logger.info(f"Early stopping after {patience_counter} epochs without improvement")
                    break

        # Clear CUDA cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()

        # Save loss curve
        self.save_final_losses(self.config['output_dir'])

        # Load best checkpoint for testing with synchronized timing
        checkpoint_load_start = self._sync_and_time()
        filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{seed:02}.pth"
        checkpoint_path = os.path.join(model_dir, filename)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        if self.timing_enabled:
            self._log_timing("Checkpoint Load", self._sync_and_time() - checkpoint_load_start)

        # Time testing phase with proper synchronization
        test_start = self._sync_and_time()
        test_outputs, test_targets = self.test_model(model, test_loader)
        test_metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
        self.logger.info("Test metrics: " + ", ".join(f"{k}: {v:.2f}" for k, v in test_metrics.items()))
        test_duration = self._sync_and_time() - test_start
        if self.timing_enabled:
            self._log_timing("Test Phase", test_duration)

        # Clear CUDA cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()

        # Time Bayesian inference with proper synchronization
        bayesian_start = self._sync_and_time()
        num_samples = 100 if "BNN" in self.config['model']['model_type'] else 1
        bayesian_results, test_res_df = self.bayesian_inference_total_uncertainty(model, test_loader,
                                                                                  num_samples=num_samples)
        bayesian_duration = self._sync_and_time() - bayesian_start
        if self.timing_enabled:
            self._log_timing("Bayesian Inference", bayesian_duration, f"{num_samples} samples")

        # Time plotting with proper synchronization
        plot_start = self._sync_and_time()
        # Plotting test metrics from bayesian inference
        plot_test_metrics(test_res_df, output_dir=self.config['output_dir'], 
                         feature_registry=self.feature_registry)
        if self.timing_enabled:
            self._log_timing("Plotting", self._sync_and_time() - plot_start)

        # NEW: Temporal split analysis
        try:
            self.logger.info("Performing temporal split analysis...")
            
            # Simple split: May 2024 and later = extrapolation, everything before = interpolation
            interpolation_df, extrapolation_df, split_info = self.split_test_data_by_date(test_res_df)
            
            # Calculate and save metrics for each subset
            temporal_metrics = self.save_temporal_split_metrics(interpolation_df, extrapolation_df, split_info, self.config['output_dir'])
            
            # Generate separate plots for each subset if they have sufficient data
            if len(interpolation_df) > 1000:  # Minimum threshold for meaningful plots
                try:
                    # plot_test_metrics automatically appends 'test_metrics', so we pass the base directory
                    interpolation_base_dir = os.path.join(self.config['output_dir'], 'interpolation')
                    plot_test_metrics(interpolation_df, output_dir=interpolation_base_dir, 
                                    feature_registry=self.feature_registry)
                    self.logger.info(f"Generated plots for interpolation data")
                except Exception as e:
                    self.logger.warning(f"Could not generate plots for interpolation: {e}")
            
            if len(extrapolation_df) > 1000:  # Minimum threshold for meaningful plots
                try:
                    # plot_test_metrics automatically appends 'test_metrics', so we pass the base directory
                    extrapolation_base_dir = os.path.join(self.config['output_dir'], 'extrapolation')
                    plot_test_metrics(extrapolation_df, output_dir=extrapolation_base_dir,
                                    feature_registry=self.feature_registry)
                    self.logger.info(f"Generated plots for extrapolation data")
                except Exception as e:
                    self.logger.warning(f"Could not generate plots for extrapolation: {e}")
            
            # Log summary of temporal split
            self.logger.info("Temporal split analysis completed:")
            self.logger.info(f"  Total samples: {split_info['total_samples']:,}")
            self.logger.info(f"  Interpolation: {split_info['interpolation_samples']:,} ({split_info['interpolation_percentage']:.1f}%)")
            self.logger.info(f"  Extrapolation: {split_info['extrapolation_samples']:,} ({split_info['extrapolation_percentage']:.1f}%)")
                
        except Exception as e:
            self.logger.warning(f"Temporal split analysis failed: {e}")

        self.logger.info(f"Test MAE: {(bayesian_results['baysian_mae']):.2f}")
        self.logger.info(f"Test MSE: {(bayesian_results['baysian_mse']):.2f}")
        self.logger.info(f"Epistemic uncertainty (mean): {bayesian_results['epistemic_std'].mean():.2f}")
        self.logger.info(f"Aleatoric uncertainty (mean): {bayesian_results['aleatoric_std'].mean():.2f}")
        self.logger.info(f"Total uncertainty (mean): {bayesian_results['total_std'].mean():.2f}")

        if not self.config["debug"]:
            wandb.log({
                **test_metrics,
                'test_baysian_mae': (bayesian_results['baysian_mae']),
                'test_baysian_mse': (bayesian_results['baysian_mse']),
                'test_epistemic_uncertainty': bayesian_results['epistemic_std'].mean().item(),
                'test_aleatoric_uncertainty': bayesian_results['aleatoric_std'].mean().item(),
                'test_total_uncertainty': bayesian_results['total_std'].mean().item(),
            })
            wandb.finish()
            
        # Print comprehensive timing summary and save to file
        if self.timing_enabled:
            self._print_timing_summary()
            if self.save_timing_to_file:
                self._save_timing_stats()
