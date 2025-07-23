# base_trainer.py
import os
import torch
import torchbnn as bnn
from datetime import datetime
from tqdm import tqdm
import wandb  # if you use wandb logging
from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer, get_scheduler
from utils.metrics import calculate_metrics

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
        
        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader), disable=disable_tqdm):
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
            for inputs, targets in dataloader:
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
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                stec, uncertainty = self.compute_stec_unc(outputs)
                all_outputs.append(torch.stack([stec, uncertainty], dim=1).cpu())
                all_targets.append(targets.cpu())
        return torch.cat(all_outputs), torch.cat(all_targets)

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
        all_predictions = []
        all_targets = []

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
        if not self.config["debug"]:
            wandb.log(test_metrics)
            wandb.finish()

        """all_predictions.append(test_outputs)
        all_targets.append(test_targets)

        # Ensemble testing: average predictions from each model.
        self.logger.info("Testing ensemble models...")
        ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)
        ensemble_test_metrics = calculate_metrics(ensemble_predictions, all_targets[0], prefix="test")
        self.logger.info(f"Ensemble Test Metrics: " + ", ".join(f"{k}: {v:.2f}" for k, v in ensemble_test_metrics.items()))"""
