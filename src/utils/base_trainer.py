# base_trainer.py
import os
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer, get_scheduler
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics
from utils.feature_registry import create_default_registry, FeatureType


class BaseTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.get('device', torch.device('cpu'))
        
        # Initialize feature management
        self.feature_registry = config.get('feature_registry') or create_default_registry(config)
        
        self.loss_weight = config['training']['loss_weight']
        self.eps = 1e-6

        # Train loss in log space?
        self.use_log_target = self.config['training'].get('log_target', True)
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

    # ---------- small helpers ----------

    def _targets_to_training_space(self, targets):
        """Return targets for the loss computation (log-space if enabled) AND keep original for metrics."""
        targets = targets.to(self.device)
        original_targets = targets.clone()  # for metrics (always linear/original)
        if self.use_log_target:
            training_targets = torch.log(targets + self.eps)
        else:
            training_targets = targets
        return training_targets, original_targets

    def _pred_log_to_linear(self, mu_log, var_log):
        """
        Convert log-space (Normal) params to linear-space (LogNormal) mean/std/var.
        mu_log: mean of Z = log(Y+eps)
        var_log: variance of Z
        Returns:
            point (mean or median in linear space), std_linear, var_linear
        """
        # Log-normal moments
        # mean_y = exp(mu + 0.5*sigma2)
        # var_y  = (exp(sigma2) - 1) * exp(2*mu + sigma2)
        mean_y = torch.exp(mu_log + 0.5 * var_log) - self.eps
        var_y = (torch.exp(var_log) - 1.0) * torch.exp(2 * mu_log + var_log)
        std_y = torch.sqrt(var_y)

        if self.log_space_point == 'median':
            point = torch.exp(mu_log) - self.eps  # median of log-normal
        else:
            point = mean_y  # unbiased mean in original units

        return point, std_y, var_y

    def _pred_linear_from_linear(self, mean, var):
        """Identity mapping when training in linear space. Returns (point, std, var)."""
        std = torch.sqrt(var.clamp_min(0.0))
        return mean, std, var

    # ---------- model I/O naming ----------

    def compute_mean_var(self, outputs):
        """
        Model head should return:
            outputs[0] = mean   (mu_log if log-targets, else linear mean)
            outputs[1] = variance (var_log if log-targets, else linear variance)
        """
        pred_mean, pred_var = outputs[0], outputs[1]
        return pred_mean, pred_var

    # ---------- training / validation ----------

    def train_epoch(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, optimizer):
        model.train()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []
        disable_tqdm = "cluster" in os.environ.get('HOME', '')

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader),
                                         disable=disable_tqdm, desc="Training"):
            
                
            inputs = inputs.to(self.device)
            training_targets, original_targets = self._targets_to_training_space(targets)

            optimizer.zero_grad()

            outputs = model(inputs)
            pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

            # Losses in the training space
            mse_loss = criterion_mse(pred_mean_raw, training_targets)
            nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
            kld_loss = criterion_kld(model)
            #loss = nll_loss + self.loss_weight * kld_loss
            loss = mse_loss

            loss.backward()
            optimizer.step()

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

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        n_batches = len(dataloader)
        return (running_loss / n_batches,
                running_mse / n_batches,
                running_nll / n_batches,
                running_kld / n_batches,
                running_variance / n_batches,
                all_outputs, all_targets)

    def validate_epoch(self, model, dataloader, criterion_mse, criterion_nll, criterion_kld):
        model.eval()
        running_loss = running_mse = running_nll = running_kld = running_variance = 0.0
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation", disable="cluster" in os.environ.get('HOME', '')):
                inputs = inputs.to(self.device)
                training_targets, original_targets = self._targets_to_training_space(targets)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

                # Losses in training space
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)
                loss = nll_loss + self.loss_weight * kld_loss

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

        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Testing", disable="cluster" in os.environ.get('HOME', '')):

                inputs = inputs.to(self.device)
                training_targets, original_targets = self._targets_to_training_space(targets)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

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
                                        disable="cluster" in os.environ.get('HOME', '')):
                bs = inputs.size(0)
                inputs = inputs.to(self.device)

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

                if num_samples == 1:
                    pred_stack = torch.stack(per_sample_means, dim=0).squeeze(0)  # [B]
                    alea_var_stack = torch.stack(per_sample_alea_vars, dim=0).squeeze(0)  # [B]
                else:
                    pred_stack = torch.stack(per_sample_means, dim=0)        # [S, B]
                    alea_var_stack = torch.stack(per_sample_alea_vars, dim=0) # [S, B]

                stec_mean = pred_stack.mean(dim=0)
                epistemic_var = pred_stack.var(dim=0)                 # Var over means
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
        
        self.logger.info(f"Loss curve saved to: {loss_plot_path}")

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
            self.logger.info(f"Loss history saved to: {csv_path}")

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

        if not self.config["debug"]:
            wandbname = f"{self.config['mode']} {self.config['model']['model_type']} {self.config['year']}-{self.config['doy']} m{seed+1:02}"
            wandb.init(project=self.config['project_name'], name=wandbname, config=self.config)

        model = init_model_fn(seed)
        criterion_mse = get_criterion(self.config, "MSELoss")
        criterion_nll = get_criterion(self.config, "GaussianNLLLoss")
        criterion_kld = get_criterion(self.config, "BKLLoss")
        optimizer = get_optimizer(self.config, model.parameters())

        scheduler = None
        if self.config[training_key]["scheduler"]:
            scheduler = get_scheduler(self.config, optimizer)

        best_val_loss = float('inf')
        patience_counter = 0
        epochs = self.config[training_key]["epochs"]

        for epoch in range(epochs):
            print(" ")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss, train_mse, train_nll, train_kld, train_variance, train_outputs, train_targets = \
                self.train_epoch(model, train_loader, criterion_mse, criterion_nll, criterion_kld, optimizer)
            val_loss, val_mse, val_nll, val_kld, val_variance, val_outputs, val_targets = \
                self.validate_epoch(model, val_loader, criterion_mse, criterion_nll, criterion_kld)

            # Track losses for plotting
            self.track_losses(epoch + 1, train_loss, val_loss)

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
                    **calculate_metrics(train_outputs, train_targets, prefix="train"),
                    **calculate_metrics(val_outputs, val_targets, prefix="val"),
                    'epoch': epoch + 1
                })

            if scheduler:
                scheduler.step()

            self.logger.info(f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}")

            if val_loss < best_val_loss or self.config[training_key]["save_model_every_epoch"]:
                best_val_loss = self.save_checkpoint(self.config, model, optimizer, epoch, 
                                                   val_loss, best_val_loss, model_dir, seed)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config[training_key].get("patience", float('inf')):
                    self.logger.info(f"Early stopping after {patience_counter} epochs without improvement")
                    break

        # Save loss curve
        self.save_final_losses(self.config['output_dir'], seed)

        # Load best checkpoint for testing.
        filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{seed:02}.pth"
        checkpoint_path = os.path.join(model_dir, filename)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_outputs, test_targets = self.test_model(model, test_loader)
        test_metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
        self.logger.info("Test metrics: " + ", ".join(f"{k}: {v:.2f}" for k, v in test_metrics.items()))

        num_samples = 100 if "BNN" in self.config['model']['model_type'] else 1
        bayesian_results, test_res_df = self.bayesian_inference_total_uncertainty(model, test_loader,
                                                                                  num_samples=num_samples)

        # Plotting test metrics from bayesian inference
        plot_test_metrics(test_res_df, output_dir=self.config['output_dir'], 
                         feature_registry=self.feature_registry)

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
