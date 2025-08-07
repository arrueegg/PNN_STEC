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

class BaseTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config['device']
        self.loss_weight = config['training']['loss_weight']

    def compute_stec_unc(self, outputs):
        """
        Compute the integrated STEC and its uncertainty.
        """
        stec, uncertainty = outputs[0], outputs[1]
        return stec, uncertainty
        
    def train_epoch(self, model, dataloader, criterion_mse, criterion_nnl, criterion_kld, optimizer):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_nnl = 0.0
        running_kld = 0.0
        running_variance = 0.0
        all_outputs = []
        all_targets = []
        disable_tqdm = "cluster" in os.environ.get('HOME', '')
        
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), disable=disable_tqdm, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            stec, uncertainty = self.compute_stec_unc(outputs)
            
            mse_loss = criterion_mse(stec, targets)
            nnl_loss = criterion_nnl(stec, targets, uncertainty)
            kld_loss = criterion_kld(model)
            loss = nnl_loss + self.loss_weight * kld_loss
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mse += mse_loss.item()
            running_nnl += nnl_loss.item()
            running_kld += kld_loss.item()
            running_variance += torch.mean(uncertainty).item()
            all_outputs.append(torch.stack([stec, uncertainty], dim=1).detach().cpu())
            all_targets.append(targets.detach().cpu())

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)
        n_samples = len(dataloader)
        avg_loss = running_loss / n_samples
        avg_mse = running_mse / n_samples
        avg_nnl = running_nnl / n_samples
        avg_kld = running_kld / n_samples
        avg_variance = running_variance / n_samples
        return avg_loss, avg_mse, avg_nnl, avg_kld, avg_variance, all_outputs, all_targets

    def validate_epoch(self, model, dataloader, criterion_mse, criterion_nnl, criterion_kld):
        model.eval()
        running_loss = 0.0
        running_mse = 0.0
        running_nnl = 0.0
        running_kld = 0.0
        running_variance = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Validation", disable="cluster" in os.environ.get('HOME', '')):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                stec, uncertainty = self.compute_stec_unc(outputs)
                
                mse_loss = criterion_mse(stec, targets)
                nnl_loss = criterion_nnl(stec, targets, uncertainty)
                kld_loss = criterion_kld(model)
                loss = nnl_loss + self.loss_weight * kld_loss
                
                running_loss += loss.item()
                running_mse += mse_loss.item()
                running_nnl += nnl_loss.item()
                running_kld += kld_loss.item()
                running_variance += torch.mean(uncertainty).item()
                all_outputs.append(torch.stack([stec, uncertainty], dim=1).cpu())
                all_targets.append(targets.cpu())

        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)        
        n_batches = len(dataloader)
        avg_loss = running_loss / n_batches
        avg_mse = running_mse / n_batches
        avg_nnl = running_nnl / n_batches
        avg_kld = running_kld / n_batches
        avg_variance = running_variance / n_batches
        return avg_loss, avg_mse, avg_nnl, avg_kld, avg_variance, all_outputs, all_targets

    def test_model(self, model, dataloader):
        model.eval()
        all_outputs = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Testing", disable="cluster" in os.environ.get('HOME', '')):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                stec, uncertainty = self.compute_stec_unc(outputs)
                all_outputs.append(torch.stack([stec, uncertainty], dim=1).cpu())
                all_targets.append(targets.cpu())
        return torch.cat(all_outputs), torch.cat(all_targets)

    def bayesian_inference(self, model, dataloader, num_samples=100):
        """Memory-efficient Monte Carlo sampling"""
        model.eval()
        
        # Pre-allocate lists for batch-wise processing
        batch_means = []
        batch_stds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                batch_predictions = []
                
                # Sample predictions for this batch
                for _ in range(num_samples):
                    outputs = model(inputs)
                    stec, _ = self.compute_stec_unc(outputs)
                    batch_predictions.append(stec.cpu())
                
                # Stack and compute statistics for this batch
                batch_predictions = torch.stack(batch_predictions, dim=0)
                batch_mean = batch_predictions.mean(dim=0)
                batch_std = batch_predictions.std(dim=0)
                
                batch_means.append(batch_mean)
                batch_stds.append(batch_std)
                all_targets.append(targets.cpu())
        
        # Concatenate across batches
        mean = torch.cat(batch_means)
        std = torch.cat(batch_stds)
        targets = torch.cat(all_targets)
        
        return mean, std, targets

    def get_feature_indices(self):
        """
        Returns a dictionary of feature indices in the input tensor.
        """

        use_swi = self.config['data'].get('use_SWI', False)

        # Define SWI column names (after removing YEAR, DOY, HR)
        swi_columns = [
            'Bartels_rotation_number',         # 0
            'Scalar_B,_nT',                    # 1
            'Vector_B_Magnitude,nT',           # 2
            'Lat_Angle_of_B_GSE',              # 3
            'Long_Angle_of_B_GSE',             # 4
            'BZ,_nT_GSE',                      # 5
            'BZ,_nT_GSM',                      # 6
            'SW_Plasma_Speed,_km/s',           # 7
            'Flow_pressure',                   # 8
            'E_elecrtric_field',               # 9
            'Alfen_mach_number',               # 10
            'Kp_index',                        # 11
            'R_Sunspot_No',                    # 12
            'Dst-index,_nT',                   # 13
            'AE-index,_nT',                    # 14
            'ap_index,_nT',                    # 15
            'f107_index',                      # 16
            'pc-index',                        # 17
            'AL-index,_nT',                    # 18
            'AU-index,_nT',                    # 19
            'Magnetosonic_Much_num',           # 20
            'Lyman_alpha',                     # 21
        ]

        indices = {}

        swi_offset = len(swi_columns) if use_swi else 0

        # Core feature indices
        indices.update({
            'DOY':        swi_offset + 0,
            'SOD':        swi_offset + 1,
            'SM_Lat_sta': swi_offset + 2,
            'SM_Lon_sta': swi_offset + 3,
            'Lat_sta':    swi_offset + 2,  # Alias
            'Lon_sta':    swi_offset + 3,  # Alias
            'Azimuth':    swi_offset + 4,
            'Elevation':  swi_offset + 5,
        })

        if use_swi:
            # Map important SWI feature names to their absolute indices
            indices['Kp_index']         = swi_columns.index('Kp_index')
            indices['Dst_index']       = swi_columns.index('Dst-index,_nT')
            indices['f107']            = swi_columns.index('f107_index')
            indices['R_sunspot_number'] = swi_columns.index('R_Sunspot_No')
        else:
            # Not present in input tensor
            indices['Kp_index']         = None
            indices['Dst_index']        = None
            indices['f107']             = None
            indices['R_sunspot_number'] = None

        return indices
    
    def inverse_transform_features(self, x):
        """
        Inverse-transform normalized features back to original physical units.
        
        Args:
            x: Tensor of shape (N, D), the full model input (normalized + SWI + SH if used).
        
        Returns:
            Tensor of shape (N, 8 [+ SWI]), same column order as before normalization.
        """
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

        # --- SWI features ---
        if use_swi:
            swi_tensor = x[:, :len(swi_cols)]
            for i, (min_val, max_val) in enumerate(swi_minmax):
                rescaled = swi_tensor[:, i] * (max_val - min_val) + min_val
                rescaled_parts.append(rescaled.unsqueeze(1))
            idx += len(swi_cols)

        # --- Core features (13 after transform) ---
        # [doy_sin, doy_cos, doy_norm, sin_t, cos_t, norm_t,
        #  sm_lat_norm, sm_lon_norm, sin_az, cos_az, norm_el,
        #  ipp_lat_norm, ipp_lon_norm]
        doy_norm = x[:, idx + 2]
        sod_norm = (x[:, idx + 5] + 1) * 86400 / 2
        sm_lat   = (x[:, idx + 6] + 1) * 90 - 90
        sm_lon   = (x[:, idx + 7] + 1) * 180 - 180
        azimuth  = torch.atan2(x[:, idx + 8], x[:, idx + 9]) * 180 / torch.pi % 360
        elevation = (x[:, idx + 10] + 1) * 90 / 2
        ipp_lat  = (x[:, idx + 11] + 1) * 90 - 90
        ipp_lon  = (x[:, idx + 12] + 1) * 180 - 180
        doy      = doy_norm * 365 + 1

        # Append in original column order
        rescaled_parts.extend([
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

    def bayesian_inference_total_uncertainty(self, model, dataloader, num_samples=100):
        """Compute total uncertainty = epistemic + aleatoric"""
        model.eval()
        
        final_df = pd.DataFrame()

        batch_means = []
        batch_epistemic_vars = []
        batch_aleatoric_vars = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Bayesian Inference", disable="cluster" in os.environ.get('HOME', '')):
                bs = inputs.size(0)
                inputs = inputs.to(self.device)
                batch_stec_predictions = []
                batch_aleatoric_uncertainties = []
                
                # Sample predictions for this batch
                for _ in range(num_samples):
                    outputs = model(inputs)
                    stec, uncertainty = self.compute_stec_unc(outputs)
                    batch_stec_predictions.append(stec.cpu())
                    batch_aleatoric_uncertainties.append(uncertainty.cpu())
                
                # Stack predictions
                batch_stec_predictions = torch.stack(batch_stec_predictions, dim=0)
                batch_aleatoric_uncertainties = torch.stack(batch_aleatoric_uncertainties, dim=0)
                
                # Compute uncertainties
                stec_mean = batch_stec_predictions.mean(dim=0)
                epistemic_var = batch_stec_predictions.var(dim=0)  # Variance across samples
                aleatoric_var = batch_aleatoric_uncertainties.mean(dim=0)  # Mean predicted variance
                
                batch_means.append(stec_mean)
                batch_epistemic_vars.append(epistemic_var)
                batch_aleatoric_vars.append(aleatoric_var)
                all_targets.append(targets.cpu())

                # Prepare DataFrame for this batch
                # Inverse-transform selected features
                inputs = self.inverse_transform_features(inputs)

                # Get feature indices
                indices = self.get_feature_indices()
                
                # Filter inputs tensor to include only columns specified in indices
                selected_columns = [indices[key] for key in indices if indices[key] is not None]
                filtered_inputs = inputs[:, selected_columns]

                # Concatenate inputs, targets, and predictions for the current batch
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
        
        # Concatenate across batches
        mean = torch.cat(batch_means).squeeze()
        epistemic_var = torch.cat(batch_epistemic_vars)
        aleatoric_var = torch.cat(batch_aleatoric_vars)
        targets = torch.cat(all_targets)
        
        # Total uncertainty
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

        self.logger.info(f"Training model...")

        if not self.config["debug"]:
            wandbname = f"{self.config['mode']} {self.config['model']['model_type']} {self.config['year']}-{self.config['doy']} m{seed+1:02}"
            wandb.init(project=self.config['project_name'], name=wandbname, config=self.config)

        model = init_model_fn(seed)
        criterion_mse = get_criterion(self.config, "MSELoss")
        criterion_nnl = get_criterion(self.config, "GaussianNLLLoss")
        criterion_kld = get_criterion(self.config, "BKLLoss")
        optimizer = get_optimizer(self.config, model.parameters())

        # Set up scheduler from the proper training configuration section.
        scheduler = None
        if self.config[training_key]["scheduler"]:
            scheduler = get_scheduler(self.config, optimizer)

        best_val_loss = float('inf')
        patience_counter = 0
        epochs = self.config[training_key]["epochs"]

        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            # Within your training loop:
            train_loss, train_mse, train_nnl, train_kld, train_variance, train_outputs, train_targets = self.train_epoch(model, train_loader, criterion_mse, criterion_nnl, criterion_kld, optimizer)
            val_loss, val_mse, val_nnl, val_kld, val_variance, val_outputs, val_targets = self.validate_epoch(model, val_loader, criterion_mse, criterion_nnl, criterion_kld)

            if not self.config["debug"]:
                wandb.log({
                    'train_loss': train_loss,
                    'train_loss_mse': train_mse,
                    'train_loss_nnl': train_nnl,
                    'train_loss_kld': train_kld,
                    'train_variance': train_variance,
                    'val_loss': val_loss,
                    'val_loss_mse': val_mse,
                    'val_loss_nnl': val_nnl,
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
                    self.logger.info(f"Early stopping triggered...")
                    break

        # Load best checkpoint for testing.
        filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{seed:02}.pth"
        checkpoint_path = os.path.join(model_dir, filename)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_outputs, test_targets = self.test_model(model, test_loader)
        test_metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
        self.logger.info(f"Test metrics: " +
                            ", ".join(f"{k}: {v:.2f}" for k, v in test_metrics.items()))

        bayesian_results, test_res_df = self.bayesian_inference_total_uncertainty(model, test_loader, num_samples=100)

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

        """all_predictions.append(test_outputs)
        all_targets.append(test_targets)

        # Ensemble testing: average predictions from each model.
        self.logger.info("Testing ensemble models...")
        ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
        ensemble_test_metrics = calculate_metrics(ensemble_predictions, all_targets[0], prefix="test")
        self.logger.info(f"Ensemble Test Metrics: " + ", ".join(f"{k}: {v:.2f}" for k, v in ensemble_test_metrics.items()))"""
