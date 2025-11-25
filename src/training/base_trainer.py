"""
Base Trainer Module for PNN_STEC Training

This module provides a lightweight BaseTrainer class that orchestrates training
using composition of specialized manager classes instead of a monolithic design.

The BaseTrainer acts as a coordinator, delegating responsibilities to:
- DataTransforms: Target/feature transformations and normalization
- TrainingUtils: KL annealing, checkpointing, loss tracking
- TrainManager: Training epoch execution and optimization
- ValidationManager: Validation, testing, and feature processing
- InferenceManager: Bayesian inference and uncertainty quantification

This modular approach improves maintainability, testability, and separation of concerns.
"""

import os
import gc
import torch
import wandb

from utils.loss_function import get_criterion
from utils.optimizers import get_optimizer, get_scheduler
from utils.metrics import calculate_metrics
from viz import plot_test_metrics
from utils.feature_registry import create_default_registry

from .data_transforms import DataTransforms
from .training_utils import TrainingUtils
from .train_manager import TrainManager
from .validation_manager import ValidationManager
from .inference_manager import InferenceManager


class BaseTrainer:
    """
    Lightweight orchestrator for STEC model training using modular components.

    This class coordinates training workflow by delegating specific responsibilities
    to specialized manager classes, following composition over inheritance principles.
    """

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = config.get("device", torch.device("cpu"))

        # Initialize feature management
        self.feature_registry = config.get(
            "feature_registry"
        ) or create_default_registry(config)

        # Initialize specialized managers using composition
        self.data_transforms = DataTransforms(
            config, self.feature_registry, logger, self.device
        )
        self.training_utils = TrainingUtils(config, logger)
        self.train_manager = TrainManager(
            config, self.data_transforms, self.training_utils, logger, self.device
        )
        self.validation_manager = ValidationManager(
            config, self.data_transforms, self.training_utils, logger, self.device
        )
        self.inference_manager = InferenceManager(
            config,
            self.data_transforms,
            self.training_utils,
            self.validation_manager,
            logger,
            self.device,
        )
            

    # ---------- Public interface delegating to managers ----------

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
        """Train a single epoch. Delegates to TrainManager."""
        return self.train_manager.train_epoch(
            model,
            dataloader,
            criterion_mse,
            criterion_nll,
            criterion_kld,
            optimizer,
            epoch,
        )

    def train_epoch_ensemble(
        self,
        model,
        dataloader,
        criterion_mse,
        criterion_nll,
        criterion_kld,
        optimizer,
        epoch=0,
    ):
        """Train a single epoch for ensemble models. Delegates to TrainManager."""
        return self.train_manager.train_epoch_ensemble(
            model,
            dataloader,
            criterion_mse,
            criterion_nll,
            criterion_kld,
            optimizer,
            epoch,
        )

    def validate_epoch(
        self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0
    ):
        """Validate a single epoch. Delegates to ValidationManager."""
        return self.validation_manager.validate_epoch(
            model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch
        )

    def validate_epoch_ensemble(
        self, model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch=0
    ):
        """Validate a single epoch for ensemble models. Delegates to ValidationManager."""
        return self.validation_manager.validate_epoch_ensemble(
            model, dataloader, criterion_mse, criterion_nll, criterion_kld, epoch
        )

    def test_model(self, model, dataloader):
        """Test the model. Delegates to ValidationManager."""
        return self.validation_manager.test_model(model, dataloader)

    def bayesian_inference_total_uncertainty(self, model, dataloader, num_samples=100):
        """Perform Bayesian inference with uncertainty quantification. Delegates to InferenceManager."""
        return self.inference_manager.bayesian_inference_total_uncertainty(
            model, dataloader, num_samples
        )

    def inverse_transform_features(self, x):
        """Transform features back to original scale. Delegates to ValidationManager."""
        return self.validation_manager.inverse_transform_features(x)

    def get_feature_indices(self):
        """Get feature indices mapping. Delegates to ValidationManager."""
        return self.validation_manager.get_feature_indices()

    # ---------- Utility methods delegating to TrainingUtils ----------

    def get_current_kl_weight(self, epoch):
        """Get current KL weight with annealing. Delegates to TrainingUtils."""
        return self.training_utils.get_current_kl_weight(epoch)

    def save_checkpoint(
        self, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir, model_seed
    ):
        """Save model checkpoint. Delegates to TrainingUtils."""
        return self.training_utils.save_checkpoint(
            model, optimizer, epoch, val_loss, best_loss, checkpoint_dir, model_seed
        )

    def track_losses(self, epoch, train_loss, val_loss):
        """Track losses for plotting. Delegates to TrainingUtils."""
        self.training_utils.track_losses(epoch, train_loss, val_loss)

    def save_final_losses(self, output_dir):
        """Save final loss plots and data. Delegates to TrainingUtils."""
        self.training_utils.save_final_losses(output_dir)

    def split_test_data_by_date(self, test_df):
        """Split test data by temporal criteria. Delegates to TrainingUtils."""
        return self.training_utils.split_test_data_by_date(test_df)

    def save_temporal_split_metrics(
        self, interpolation_df, extrapolation_df, split_info, experiment_dir
    ):
        """Save temporal split analysis. Delegates to TrainingUtils."""
        return self.training_utils.save_temporal_split_metrics(
            interpolation_df, extrapolation_df, split_info, experiment_dir
        )

    # ---------- Backward compatibility methods ----------

    def compute_mean_var(self, outputs):
        """Backward compatibility: extract mean and variance from model outputs."""
        return self.data_transforms.compute_mean_var(outputs)

    # ---------- Main training orchestration ----------

    def run_training(
        self, train_loader, val_loader, test_loader, init_model_fn, training_key
    ):
        """
        Main training orchestration method.

        Coordinates the full training workflow by delegating to appropriate managers:
        - Setup and initialization
        - Training and validation loops
        - Testing and inference
        - Results visualization and analysis

        Parameters:
          - train_loader, val_loader, test_loader: Dataloaders for training/validation/testing.
          - init_model_fn: Function that takes (seed) and returns an initialized model.
          - training_key: String key to choose the training configuration, e.g. "finetune" or "pretrain".
        """
        seed = self.config["random_seed"]
        model_dir = os.path.join(self.config["output_dir"], "model")
        os.makedirs(model_dir, exist_ok=True)

        self.logger.info("Training model...")

        if not self.config["debug"]:
            # Use the sweep-aware wandb setup
            from utils.wandb_sweep_integration import setup_wandb_for_sweep

            experiment_name = os.path.basename(self.config["output_dir"])
            setup_wandb_for_sweep(self.config, experiment_name)

        model = init_model_fn(seed)

        criterion_mse = get_criterion(self.config, "MSELoss")
        criterion_nll = get_criterion(self.config, "GaussianNLLLoss")
        criterion_kld = get_criterion(self.config, "BKLLoss")
        optimizer = get_optimizer(self.config, model.parameters())

        scheduler = None
        if self.config[training_key]["scheduler"]:
            scheduler = get_scheduler(self.config, optimizer)

        best_val_loss = float("inf")
        patience_counter = 0
        epochs = self.config[training_key]["epochs"]

        for epoch in range(epochs):
            gc.collect()
            print("")
            self.logger.info(f"Epoch {epoch+1}/{epochs}")

            # Update sampler epoch for different data sampling each epoch
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Check if model is an ensemble and use appropriate training method
            model_type = self.config["model"]["model_type"]
            if model_type == "DE_MLP":
                train_loss, train_mse, train_nll, train_kld, train_metrics = (
                    self.train_epoch_ensemble(
                        model,
                        train_loader,
                        criterion_mse,
                        criterion_nll,
                        criterion_kld,
                        optimizer,
                        epoch,
                    )
                )
                # Extract outputs and targets for compatibility
                train_outputs = train_metrics.get("predictions", [])
                train_targets = train_metrics.get("targets", [])
                train_variance = 0.0  # Ensemble handles variance internally
            else:
                (
                    train_loss,
                    train_mse,
                    train_nll,
                    train_kld,
                    train_variance,
                    train_outputs,
                    train_targets,
                ) = self.train_epoch(
                    model,
                    train_loader,
                    criterion_mse,
                    criterion_nll,
                    criterion_kld,
                    optimizer,
                    epoch,
                )
                train_metrics = calculate_metrics(
                    train_outputs, train_targets, prefix="train"
                )

            # Use appropriate validation method for ensemble models
            if model_type == "DE_MLP":
                val_loss, val_mse, val_nll, val_kld, val_metrics = (
                    self.validate_epoch_ensemble(
                        model,
                        val_loader,
                        criterion_mse,
                        criterion_nll,
                        criterion_kld,
                        epoch,
                    )
                )
                # Extract outputs and targets for compatibility
                val_outputs = val_metrics.get("predictions", [])
                val_targets = val_metrics.get("targets", [])
                val_variance = 0.0  # Ensemble handles variance internally
            else:
                (
                    val_loss,
                    val_mse,
                    val_nll,
                    val_kld,
                    val_variance,
                    val_outputs,
                    val_targets,
                ) = self.validate_epoch(
                    model,
                    val_loader,
                    criterion_mse,
                    criterion_nll,
                    criterion_kld,
                    epoch,
                )
                val_metrics = calculate_metrics(val_outputs, val_targets, prefix="val")

            # Track losses for plotting
            self.track_losses(epoch + 1, train_loss, val_loss)

            # Metrics already calculated in ensemble methods, just use them
            if model_type != "DE_MLP":
                train_metrics = calculate_metrics(
                    train_outputs, train_targets, prefix="train"
                )
                val_metrics = calculate_metrics(val_outputs, val_targets, prefix="val")

            if not self.config["debug"]:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "train_mse": train_mse,
                        "train_nll": train_nll,
                        "train_kld": train_kld,
                        "train_variance": train_variance,
                        "val_loss": val_loss,
                        "val_mse": val_mse,
                        "val_nll": val_nll,
                        "val_kld": val_kld,
                        "val_variance": val_variance,
                        "learning_rate": (
                            scheduler.get_last_lr()[0] if scheduler else None
                        ),
                        "kl_weight": self.get_current_kl_weight(epoch),
                        **train_metrics,
                        **val_metrics,
                        "epoch": epoch + 1,
                    }
                )

            if scheduler:
                # ReduceLROnPlateau requires validation loss as argument
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            self.logger.info(
                f"Train Loss: {train_loss:.2f}, Validation Loss: {val_loss:.2f}"
            )

            if (
                val_loss < best_val_loss
                or self.config['pretrain']["save_model_every_epoch"]
            ):
                best_val_loss = self.save_checkpoint(
                    model, optimizer, epoch, val_loss, best_val_loss, model_dir, seed
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config['pretrain'].get(
                    "patience", float("inf")
                ):
                    self.logger.info(
                        f"Early stopping after {patience_counter} epochs without improvement"
                    )
                    break

        # Clear CUDA cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()

        # Save loss curve
        self.save_final_losses(self.config["output_dir"])

        # Load best checkpoint for testing
        filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{seed:02}.pth"
        checkpoint_path = os.path.join(model_dir, filename)
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])

        test_outputs, test_targets = self.test_model(model, test_loader)
        test_metrics = calculate_metrics(test_outputs, test_targets, prefix="test")
        self.logger.info(
            "Test metrics: "
            + ", ".join(f"{k}: {v:.2f}" for k, v in test_metrics.items())
        )

        # Clear CUDA cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()

        # Bayesian inference
        num_samples = 100 if "BNN" in self.config["model"]["model_type"] or "Bayesian" in self.config["model"]["model_type"] else 1
        bayesian_results, test_res_df = self.bayesian_inference_total_uncertainty(
            model, test_loader, num_samples=num_samples
        )

        # Plotting test metrics from bayesian inference
        plot_test_metrics(
            test_res_df,
            output_dir=self.config["output_dir"],
            feature_registry=self.feature_registry,
        )

        # Temporal split analysis
        try:
            print("")
            self.logger.info("Performing temporal split analysis...")

            # Simple split: May 2024 and later = extrapolation, everything before = interpolation
            interpolation_df, extrapolation_df, split_info = (
                self.split_test_data_by_date(test_res_df)
            )

            # Calculate and save metrics for each subset
            self.save_temporal_split_metrics(
                interpolation_df,
                extrapolation_df,
                split_info,
                self.config["output_dir"],
            )

            if self.config["mode"] == "pretrain":
                # Generate separate plots for each subset if they have sufficient data
                if len(interpolation_df) > 1000:  # Minimum threshold for meaningful plots
                    try:
                        interpolation_base_dir = os.path.join(
                            self.config["output_dir"], "interpolation"
                        )
                        plot_test_metrics(
                            interpolation_df,
                            output_dir=interpolation_base_dir,
                            feature_registry=self.feature_registry,
                        )
                        self.logger.info("Generated plots for interpolation data")
                    except Exception as e:
                        self.logger.warning(
                            f"Could not generate plots for interpolation: {e}"
                        )

                if len(extrapolation_df) > 1000:  # Minimum threshold for meaningful plots
                    try:
                        extrapolation_base_dir = os.path.join(
                            self.config["output_dir"], "extrapolation"
                        )
                        plot_test_metrics(
                            extrapolation_df,
                            output_dir=extrapolation_base_dir,
                            feature_registry=self.feature_registry,
                        )
                        self.logger.info("Generated plots for extrapolation data")
                    except Exception as e:
                        self.logger.warning(
                            f"Could not generate plots for extrapolation: {e}"
                        )

            # Log summary of temporal split
            self.logger.info("Temporal split analysis completed:")
            self.logger.info(f"  Total samples: {split_info['total_count']:,}")
            self.logger.info(
                f"  Interpolation: {split_info['interpolation_count']:,} ({split_info['interpolation_percentage']:.1f}%)"
            )
            self.logger.info(
                f"  Extrapolation: {split_info['extrapolation_count']:,} ({split_info['extrapolation_percentage']:.1f}%)"
            )

        except Exception as e:
            self.logger.warning(f"Temporal split analysis failed: {e}")

        print("")
        self.logger.info(f"Test MAE: {(bayesian_results['baysian_mae']):.2f}")
        self.logger.info(f"Test MSE: {(bayesian_results['baysian_mse']):.2f}")
        self.logger.info(
            f"Epistemic uncertainty (mean): {bayesian_results['epistemic_std'].mean():.2f}"
        )
        self.logger.info(
            f"Aleatoric uncertainty (mean): {bayesian_results['aleatoric_std'].mean():.2f}"
        )
        self.logger.info(
            f"Total uncertainty (mean): {bayesian_results['total_std'].mean():.2f}"
        )

        if not self.config["debug"]:
            wandb.log(
                {
                    **test_metrics,
                    "test_baysian_mae": (bayesian_results["baysian_mae"]),
                    "test_baysian_mse": (bayesian_results["baysian_mse"]),
                    "test_epistemic_uncertainty": bayesian_results["epistemic_std"]
                    .mean()
                    .item(),
                    "test_aleatoric_uncertainty": bayesian_results["aleatoric_std"]
                    .mean()
                    .item(),
                    "test_total_uncertainty": bayesian_results["total_std"]
                    .mean()
                    .item(),
                }
            )
            wandb.finish()

        # Clear CUDA cache and garbage collector
        torch.cuda.empty_cache()
        gc.collect()

        return model, test_metrics
