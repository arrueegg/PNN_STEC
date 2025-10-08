"""
Training Utilities Module for PNN_STEC Training

This module provides utility functions for training, including:
- Performance timing and profiling
- KL annealing for Bayesian networks
- Checkpointing and model saving
- Loss tracking and visualization
- Training state management

Extracted from BaseTrainer to separate utility concerns.
"""

import os
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
import csv


class TrainingUtils:
    """Utility functions for training management and monitoring."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # KL annealing configuration
        self.kl_annealing = config["training"].get("kl_annealing", {})
        self.use_kl_annealing = self.kl_annealing.get("enabled", False)
        self.loss_weight = config["training"]["loss_weight"]

        if self.use_kl_annealing:
            self.kl_warmup_epochs = self.kl_annealing.get("warmup_epochs", 20)
            self.kl_start_weight = self.kl_annealing.get("start_weight", 0.0)
            self.logger.info(
                f"🔥 KL annealing enabled: {self.kl_start_weight} → {self.loss_weight} over {self.kl_warmup_epochs} epochs"
            )

        # Timing instrumentation for performance debugging
        self.timing_enabled = config.get("enable_timing", False)
        timing_config = config.get("timing_config", {})
        self.detailed_batch_timing = (
            timing_config.get("detailed_batch_timing", False) and self.timing_enabled
        )
        self.save_timing_to_file = timing_config.get("save_timing_to_file", True)
        self.log_timing_frequency = timing_config.get("log_timing_every_n_batches", 100)

        if self.timing_enabled:
            self.logger.info("🔍 Performance timing enabled")
            if self.detailed_batch_timing:
                self.logger.info("⚠️  Detailed batch timing enabled (may slow training)")
            if self.save_timing_to_file:
                self.logger.info("📁 Timing statistics will be saved to file")
        else:
            self.logger.info("⏱️  Performance timing disabled")

        self.timing_stats = {}

        # Loss tracking
        self.train_losses = []
        self.val_losses = []
        self.epochs_tracked = []

    # ---------- Timing utilities ----------

    def sync_and_time(self):
        """Synchronize CUDA operations and return current time for accurate timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    def log_timing(self, step_name, duration, additional_info=""):
        """Log timing information for performance debugging"""
        if self.timing_enabled:
            if step_name not in self.timing_stats:
                self.timing_stats[step_name] = []
            self.timing_stats[step_name].append(duration)
            info_str = f" ({additional_info})" if additional_info else ""
            self.logger.info(f"⏱️  {step_name}: {duration:.3f}s{info_str}")

    def print_timing_summary(self):
        """Print comprehensive summary of timing statistics"""
        if self.timing_enabled and self.timing_stats:
            self.logger.info("📊 PERFORMANCE TIMING SUMMARY:")
            self.logger.info("=" * 70)

            # Sort by total time to show most time-consuming operations first
            sorted_stats = sorted(
                self.timing_stats.items(), key=lambda x: sum(x[1]), reverse=True
            )

            total_measured_time = sum(
                sum(times) for times in self.timing_stats.values()
            )

            for step_name, times in sorted_stats:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                min_time = min(times)
                max_time = max(times)
                percentage = (
                    (total_time / total_measured_time * 100)
                    if total_measured_time > 0
                    else 0
                )

                self.logger.info(f"  {step_name}:")
                self.logger.info(
                    f"    Total: {total_time:.1f}s ({percentage:.1f}%) | "
                    f"Avg: {avg_time:.3f}s | Count: {len(times)} | "
                    f"Range: {min_time:.3f}s - {max_time:.3f}s"
                )

            self.logger.info(f"\n  Total measured time: {total_measured_time:.1f}s")
            self.logger.info("=" * 70)

    def save_timing_stats(self):
        """Save timing statistics to a CSV file"""
        if self.timing_enabled and self.save_timing_to_file and self.timing_stats:
            timing_file = os.path.join(
                self.config["output_dir"], "timing_statistics.csv"
            )

            with open(timing_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [
                        "Step",
                        "Total_Time_s",
                        "Average_Time_s",
                        "Count",
                        "Min_Time_s",
                        "Max_Time_s",
                        "Percentage",
                    ]
                )

                total_measured_time = sum(
                    sum(times) for times in self.timing_stats.values()
                )

                # Sort by total time
                sorted_stats = sorted(
                    self.timing_stats.items(), key=lambda x: sum(x[1]), reverse=True
                )

                for step_name, times in sorted_stats:
                    total_time = sum(times)
                    avg_time = total_time / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    percentage = (
                        (total_time / total_measured_time * 100)
                        if total_measured_time > 0
                        else 0
                    )

                    writer.writerow(
                        [
                            step_name,
                            f"{total_time:.3f}",
                            f"{avg_time:.3f}",
                            len(times),
                            f"{min_time:.3f}",
                            f"{max_time:.3f}",
                            f"{percentage:.2f}",
                        ]
                    )

            self.logger.info(f"📁 Timing statistics saved to: {timing_file}")

    # ---------- KL annealing ----------

    def get_current_kl_weight(self, epoch):
        """Compute current KL weight based on annealing schedule"""
        if not self.use_kl_annealing:
            return self.loss_weight

        if epoch < self.kl_warmup_epochs:
            # Linear annealing from start_weight to loss_weight
            progress = epoch / self.kl_warmup_epochs
            current_weight = self.kl_start_weight + progress * (
                self.loss_weight - self.kl_start_weight
            )
        else:
            # After warmup, use full loss weight
            current_weight = self.loss_weight

        return current_weight

    # ---------- Checkpointing and state management ----------

    def save_checkpoint(
        self, model, optimizer, epoch, val_loss, best_loss, checkpoint_dir, model_seed
    ):
        """Save model checkpoint with training state."""
        self.logger.info(
            f"Validation loss improved from {best_loss:.2f} to {val_loss:.2f}. Saving checkpoint."
        )
        if self.config["mode"] == "finetune":
            filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{model_seed:02}.pth"
        elif self.config["mode"] == "pretrain":
            filename = f"{self.config['mode']}_{self.config['model']['model_type']}_seed{model_seed:02}.pth"

        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_type": self.config["model"]["model_type"],
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                # Save loss history with checkpoint
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "epochs_tracked": self.epochs_tracked,
            },
            filepath,
        )
        return val_loss

    # ---------- Loss tracking and visualization ----------

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

        # Combined plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs_tracked, self.train_losses, label="Training Loss", color="blue"
        )
        plt.plot(
            self.epochs_tracked, self.val_losses, label="Validation Loss", color="red"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_plot_path = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Training loss only
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs_tracked, self.train_losses, label="Training Loss", color="blue"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path_train = os.path.join(output_dir, "train_loss_curve.png")
        plt.savefig(loss_plot_path_train, dpi=300, bbox_inches="tight")
        plt.close()

        # Validation loss only
        plt.figure(figsize=(10, 6))
        plt.plot(
            self.epochs_tracked, self.val_losses, label="Validation Loss", color="red"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_plot_path_val = os.path.join(output_dir, "val_loss_curve.png")
        plt.savefig(loss_plot_path_val, dpi=300, bbox_inches="tight")
        plt.close()

    def save_final_losses(self, output_dir):
        """Save final training results including loss curve."""
        # Plot the loss curve
        self.plot_loss_curve(output_dir)

        # Optionally save loss data as CSV for further analysis
        if self.train_losses and self.val_losses:
            loss_data = pd.DataFrame(
                {
                    "epoch": self.epochs_tracked,
                    "train_loss": self.train_losses,
                    "val_loss": self.val_losses,
                }
            )
            csv_path = os.path.join(output_dir, "loss_history.csv")
            loss_data.to_csv(csv_path, index=False)

    def split_test_data_by_date(self, test_df):
        """
        Split test dataframe into interpolation and extrapolation subsets.
        Simple rule: May 2024 and later = extrapolation, everything before = interpolation.

        Args:
            test_df: DataFrame with test predictions and targets

        Returns:
            tuple: (interpolation_df, extrapolation_df, split_info)
        """
        from datetime import datetime, timedelta

        # Define the split date
        split_date = datetime(2024, 5, 1)

        def create_date(row):
            """Convert year and day_of_year to datetime"""
            year = int(row["year"])
            doy = int(row["day_of_year"])
            return datetime(year, 1, 1) + timedelta(days=doy - 1)

        # Create datetime column for filtering
        test_df["date"] = test_df.apply(create_date, axis=1)

        # Split the data
        interpolation_df = test_df[test_df["date"] < split_date].copy()
        extrapolation_df = test_df[test_df["date"] >= split_date].copy()

        # Create split information
        split_info = {
            "split_date": split_date,
            "interpolation_count": len(interpolation_df),
            "extrapolation_count": len(extrapolation_df),
            "total_count": len(test_df),
            "interpolation_percentage": len(interpolation_df) / len(test_df) * 100,
            "extrapolation_percentage": len(extrapolation_df) / len(test_df) * 100,
        }

        self.logger.info("📅 Test data temporal split:")
        self.logger.info(f"   Split date: {split_date.strftime('%Y-%m-%d')}")
        self.logger.info(
            f"   Interpolation: {split_info['interpolation_count']:,} samples ({split_info['interpolation_percentage']:.1f}%)"
        )
        self.logger.info(
            f"   Extrapolation: {split_info['extrapolation_count']:,} samples ({split_info['extrapolation_percentage']:.1f}%)"
        )

        return interpolation_df, extrapolation_df, split_info

    def save_temporal_split_metrics(
        self, interpolation_df, extrapolation_df, split_info, experiment_dir
    ):
        """Save metrics for temporal interpolation vs extrapolation performance."""
        from utils.metrics import calculate_metrics

        save_dir = os.path.join(experiment_dir, "temporal_analysis")
        os.makedirs(save_dir, exist_ok=True)

        # Calculate metrics for each subset
        if len(interpolation_df) > 0:
            interp_metrics = calculate_metrics(
                interpolation_df["predictions"].values,
                interpolation_df["targets"].values,
                interpolation_df.get("std_predictions", None),
            )
            interp_metrics["data_type"] = "interpolation"
            interp_metrics["sample_count"] = len(interpolation_df)
        else:
            interp_metrics = {"data_type": "interpolation", "sample_count": 0}

        if len(extrapolation_df) > 0:
            extrap_metrics = calculate_metrics(
                extrapolation_df["predictions"].values,
                extrapolation_df["targets"].values,
                extrapolation_df.get("std_predictions", None),
            )
            extrap_metrics["data_type"] = "extrapolation"
            extrap_metrics["sample_count"] = len(extrapolation_df)
        else:
            extrap_metrics = {"data_type": "extrapolation", "sample_count": 0}

        # Combine metrics and split info
        temporal_analysis = {
            "split_info": split_info,
            "interpolation_metrics": interp_metrics,
            "extrapolation_metrics": extrap_metrics,
        }

        # Save as JSON
        import json

        analysis_file = os.path.join(save_dir, "temporal_split_analysis.json")
        with open(analysis_file, "w") as f:
            json.dump(temporal_analysis, f, indent=2, default=str)

        # Save as CSV for easy analysis
        metrics_data = []
        for subset_name, metrics in [
            ("interpolation", interp_metrics),
            ("extrapolation", extrap_metrics),
        ]:
            if metrics.get("sample_count", 0) > 0:
                metrics_row = {"subset": subset_name}
                metrics_row.update(
                    {k: v for k, v in metrics.items() if k not in ["data_type"]}
                )
                metrics_data.append(metrics_row)

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            csv_file = os.path.join(save_dir, "temporal_split_metrics.csv")
            metrics_df.to_csv(csv_file, index=False)

        self.logger.info(f"📊 Temporal analysis saved to: {save_dir}")

        return temporal_analysis
