# base_trainer.py
import os
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import wandb

from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer, get_scheduler
from utils.metrics import calculate_metrics
from utils.plot import plot_test_metrics
from utils.feature_registry import create_default_registry, FeatureType
from utils.feature_monitor import FeatureMonitor


class BaseTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.get('device', torch.device('cpu'))
        
        # Initialize feature management
        self.feature_registry = config.get('feature_registry') or create_default_registry(config)
        self.feature_monitor = config.get('feature_monitor') or FeatureMonitor(self.feature_registry, logger)
        
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
            
            # Feature validation at training input
            if i == 0:  # Only check first batch to avoid overhead
                self.feature_monitor.check_feature_consistency(inputs, "train_input")
                
            inputs = inputs.to(self.device)
            training_targets, original_targets = self._targets_to_training_space(targets)

            optimizer.zero_grad()

            outputs = model(inputs)
            pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

            # Losses in the training space
            mse_loss = criterion_mse(pred_mean_raw, training_targets)
            nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
            kld_loss = criterion_kld(model)
            loss = nll_loss + self.loss_weight * kld_loss

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
            for i, (inputs, targets) in tqdm(dataloader, desc="Validation", disable="cluster" in os.environ.get('HOME', '')):
                
                # Feature validation at validation input
                if i == 0:  # Only check first batch
                    self.feature_monitor.check_feature_consistency(inputs, "val_input")
                    
                inputs = inputs.to(self.device)
                training_targets, original_targets = self._targets_to_training_space(targets)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

                # Losses in training space
                mse_loss = criterion_mse(pred_mean_raw, training_targets)
                nll_loss = criterion_nll(pred_mean_raw, training_targets, pred_var_raw)
                kld_loss = criterion_kld(model)
                loss = nll_loss + self.loss_weight * kld_loss

                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nll += nll_loss.item()
                running_kld += kld_loss.item()

                # Back-transform for metrics
                if self.use_log_target:
                    point_linear, std_linear, var_linear = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, var_linear = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

                running_variance += torch.mean(var_linear).item()

                all_outputs.append(torch.stack([point_linear, std_linear], dim=1).cpu())
                all_targets.append(original_targets.cpu())

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        n_batches = len(dataloader)
        return (running_loss / n_batches,
                running_mse / n_batches,
                running_nll / n_batches,
                running_kld / n_batches,
                running_variance / n_batches,
                all_outputs, all_targets)

    # ---------- testing ----------

    def test_model(self, model, dataloader):
        model.eval()
        all_outputs, all_targets = [], []

        with torch.no_grad():
            for i, (inputs, targets) in tqdm(dataloader, desc="Testing", disable="cluster" in os.environ.get('HOME', '')):
                
                # Feature validation at test input
                if i == 0:  # Only check first batch
                    self.feature_monitor.check_feature_consistency(inputs, "test_input")
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                pred_mean_raw, pred_var_raw = self.compute_mean_var(outputs)

                if self.use_log_target:
                    point_linear, std_linear, _ = self._pred_log_to_linear(pred_mean_raw, pred_var_raw)
                else:
                    point_linear, std_linear, _ = self._pred_linear_from_linear(pred_mean_raw, pred_var_raw)

                all_outputs.append(torch.stack([point_linear, std_linear], dim=1).cpu())
                all_targets.append(targets.cpu())

        return torch.cat(all_outputs), torch.cat(all_targets)

    # ---------- feature inverse-transform ----------

    def inverse_transform_features(self, x):
        use_swi = self.config['data'].get('use_SWI', False)
        sh_degree = self.config['data'].get("SH_degree", 0) or 0
        sh_dim = (sh_degree + 1) ** 2

        swi_cols = [
            'Bartels_rotation_number', 'Scalar_B,_nT', 'Vector_B_Magnitude,nT', 'Lat_Angle_of_B_GSE',
            'Long_Angle_of_B_GSE', 'BZ,_nT_GSE', 'BZ,_nT_GSM', 'SW_Plasma_Speed,_km/s',
            'Flow_pressure', 'E_elecrtric_field', 'Alfen_mach_number', 'Kp_index',
            'R_Sunspot_No', 'Dst-index,_nT', 'AE-index,_nT', 'ap_index,_nT', 'f107_index',
            'pc-index', 'AL-index,_nT', 'AU-index,_nT', 'Magnetosonic_Much_num', 'Lyman_alpha'
        ]
        swi_minmax = [
            (2407, 3000), (0, 70), (0.0, 70), (-90, 90), (0.0, 360.0), (-50, 35), (-50, 35),
            (240.0, 1100.0), (0, 60), (-20, 30), (0, 120), (0.0, 100.0), (0.0, 300.0), (-450, 100),
            (0.0, 2500.0), (0.0, 300.0), (62, 420), (-6, 16), (-2000.0, 20.0), (-200.0, 1200.0),
            (0, 15), (0, 0.015)
        ]

        idx = 0
        rescaled_parts = []

        if use_swi:
            swi_tensor = x[:, :len(swi_cols)]
            for i, (min_val, max_val) in enumerate(swi_minmax):
                rescaled = swi_tensor[:, i] * (max_val - min_val) + min_val
                rescaled_parts.append(rescaled.unsqueeze(1))
            idx += len(swi_cols)

        year = x[:, idx + 0] * 20 + 2010
        doy_norm = x[:, idx + 3]
        sod_norm = (x[:, idx + 6] + 1) * 86400 / 2
        sm_lat   = (x[:, idx + 7] + 1) * 90 - 90
        sm_lon   = (x[:, idx + 8] + 1) * 180 - 180
        azimuth  = torch.atan2(x[:, idx + 8], x[:, idx + 9]) * 180 / torch.pi % 360
        elevation = (x[:, idx + 9] + 1) * 90 / 2
        ipp_lat  = (x[:, idx + 10] + 1) * 90 - 90
        ipp_lon  = (x[:, idx + 11] + 1) * 180 - 180
        doy      = doy_norm * 365 + 1

        rescaled_parts.extend([
            year.unsqueeze(1),
            doy.unsqueeze(1),
            sod_norm.unsqueeze(1),
            sm_lat.unsqueeze(1),
            sm_lon.unsqueeze(1),
            azimuth.unsqueeze(1),
            elevation.unsqueeze(1),
            ipp_lat.unsqueeze(1),
            ipp_lon.unsqueeze(1)
        ])

        return torch.cat(rescaled_parts, dim=1)

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

                pred_stack = torch.stack(per_sample_means, dim=0)        # [S, B]
                alea_var_stack = torch.stack(per_sample_alea_vars, dim=0) # [S, B]

                stec_mean = pred_stack.mean(dim=0)
                epistemic_var = pred_stack.var(dim=0)                 # Var over means
                aleatoric_var = alea_var_stack.mean(dim=0)            # Mean aleatoric var
                batch_means.append(stec_mean)
                batch_epistemic_vars.append(epistemic_var)
                batch_aleatoric_vars.append(aleatoric_var)
                all_targets.append(targets.cpu())

                # Build per-batch DF (optional, unchanged columns)
                inputs_original = self.inverse_transform_features(inputs)
                indices = self.get_feature_indices()
                selected_columns = [indices[key] for key in indices if indices[key] is not None]
                filtered_inputs = inputs_original[:, selected_columns]

                batch_df = pd.DataFrame(
                    torch.cat([
                        filtered_inputs.cpu().view(bs, -1),
                        targets.cpu().view(bs, -1),
                        stec_mean.cpu().view(bs, -1),
                        torch.sqrt(epistemic_var).cpu().view(bs, -1),
                        torch.sqrt(aleatoric_var).cpu().view(bs, -1),
                        torch.sqrt(epistemic_var + aleatoric_var).cpu().view(bs, -1)
                    ], dim=1).numpy(),
                    columns=[
                        *[key for key in indices if indices[key] is not None],
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
        }, filepath)
        return val_loss

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

        # Log feature monitoring summary before training
        self.logger.info("=== Feature Monitoring Summary ===")
        self.feature_monitor.log_feature_summary()
        
        for epoch in range(epochs):
            print(" ")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            train_loss, train_mse, train_nll, train_kld, train_variance, train_outputs, train_targets = \
                self.train_epoch(model, train_loader, criterion_mse, criterion_nll, criterion_kld, optimizer)
            val_loss, val_mse, val_nll, val_kld, val_variance, val_outputs, val_targets = \
                self.validate_epoch(model, val_loader, criterion_mse, criterion_nll, criterion_kld)

            if not self.config["debug"]:
                wandb.log({
                    'train_loss': train_loss,
                    'train_loss_mse': train_mse,
                    'train_loss_nll': train_nll,
                    'train_loss_kld': train_kld,
                    'train_variance': train_variance,
                    'val_loss': val_loss,
                    'val_loss_mse': val_mse,
                    'val_loss_nll': val_nll,
                    'val_loss_kld': val_kld,
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
                patience_counter = 0
                best_val_loss = self.save_checkpoint(self.config, model, optimizer, epoch, val_loss, best_val_loss, model_dir, seed)
            else:
                patience_counter += 1
                if patience_counter >= self.config[training_key]["patience"]:
                    self.logger.info("Early stopping triggered...")
                    break

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
        plot_test_metrics(test_res_df, output_dir=self.config['output_dir'])

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

        # Final feature monitoring summary
        self.feature_monitor.log_feature_summary()
        if self.feature_monitor.has_failures():
            self.logger.warning("Some feature validation checks failed during training!")
