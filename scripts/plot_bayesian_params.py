"""
Plot Bayesian layer parameters from trained BayesianResNetSTEC models.

Usage:
    python scripts/plot_bayesian_params.py <experiment_folder>

Example:
    python scripts/plot_bayesian_params.py experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.05_kl5w0.1_lw1e-1_SWI
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

# Add the repo root to sys.path so we can import the stec/ package
sys.path.insert(0, str(Path(__file__).parent.parent))

from stec.models.legacy_factory import get_model
from stec.data.feature_registry import initialize_feature_registry


def find_model_checkpoint(experiment_dir):
    """Find model checkpoint in experiment directory."""
    model_dir = experiment_dir / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    pth_files = list(model_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {model_dir}")

    return pth_files[0]


def load_config(experiment_dir):
    """Load config from experiment directory."""
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_bayesian_params(model):
    """Extract Bayesian layer parameters (mean and log_sigma from torchbnn)."""
    bayesian_params = {}

    for name, module in model.named_modules():
        # torchbnn.BayesLinear uses weight_mu and weight_log_sigma
        # std = exp(log_sigma)
        if hasattr(module, "weight_mu") and hasattr(module, "weight_log_sigma"):
            weight_mu = module.weight_mu.detach().cpu().numpy()
            weight_std = torch.exp(module.weight_log_sigma).detach().cpu().numpy()

            bayesian_params[name] = {
                "weight_mu": weight_mu,
                "weight_std": weight_std,
                "weight_log_sigma": module.weight_log_sigma.detach().cpu().numpy(),
            }

            if (
                hasattr(module, "bias_mu")
                and hasattr(module, "bias_log_sigma")
                and module.bias_mu is not None
            ):
                bayesian_params[name]["bias_mu"] = module.bias_mu.detach().cpu().numpy()
                bayesian_params[name]["bias_std"] = (
                    torch.exp(module.bias_log_sigma).detach().cpu().numpy()
                )
                bayesian_params[name]["bias_log_sigma"] = (
                    module.bias_log_sigma.detach().cpu().numpy()
                )

    return bayesian_params


def plot_parameter_distributions(bayesian_params, experiment_name, output_dir):
    """Plot distributions of Bayesian parameters."""

    num_layers = len(bayesian_params)
    if num_layers == 0:
        print("No Bayesian layers found in the model!")
        return

    # Create figure with subplots for each layer
    fig, axes = plt.subplots(num_layers, 3, figsize=(15, 4 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f"Bayesian Layer Parameters\n{experiment_name}", fontsize=14, y=0.995)

    for idx, (layer_name, params) in enumerate(bayesian_params.items()):
        # Weight mean distribution
        ax = axes[idx, 0]
        weight_mu = params["weight_mu"].flatten()
        ax.hist(weight_mu, bins=50, alpha=0.7, edgecolor="black")
        ax.set_title(f"{layer_name}\nWeight Mean")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.axvline(
            weight_mu.mean(),
            color="red",
            linestyle="--",
            label=f"μ={weight_mu.mean():.4f}",
        )
        ax.legend()

        # Weight std distribution
        ax = axes[idx, 1]
        weight_std = params["weight_std"].flatten()
        ax.hist(weight_std, bins=50, alpha=0.7, edgecolor="black", color="orange")
        ax.set_title(f"{layer_name}\nWeight Std Dev")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.axvline(
            weight_std.mean(),
            color="red",
            linestyle="--",
            label=f"μ={weight_std.mean():.4f}",
        )
        ax.legend()

        # Weight uncertainty ratio (std/|mean|)
        ax = axes[idx, 2]
        uncertainty_ratio = weight_std / (np.abs(weight_mu) + 1e-8)
        ax.hist(uncertainty_ratio, bins=50, alpha=0.7, edgecolor="black", color="green")
        ax.set_title(f"{layer_name}\nUncertainty Ratio (σ/|μ|)")
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Count")
        ax.axvline(
            uncertainty_ratio.mean(),
            color="red",
            linestyle="--",
            label=f"μ={uncertainty_ratio.mean():.4f}",
        )
        ax.legend()
        ax.set_xlim(0, min(10, uncertainty_ratio.max()))

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "bayesian_parameters.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")

    # Also create a summary statistics plot
    plot_summary_statistics(bayesian_params, experiment_name, output_dir)

    plt.show()


def plot_summary_statistics(bayesian_params, experiment_name, output_dir):
    """Plot summary statistics across all Bayesian layers."""

    layer_names = []
    weight_mu_means = []
    weight_mu_stds = []
    weight_std_means = []
    weight_std_stds = []

    for layer_name, params in bayesian_params.items():
        layer_names.append(layer_name.split(".")[-1])  # Short name
        weight_mu = params["weight_mu"].flatten()
        weight_std = params["weight_std"].flatten()

        weight_mu_means.append(weight_mu.mean())
        weight_mu_stds.append(weight_mu.std())
        weight_std_means.append(weight_std.mean())
        weight_std_stds.append(weight_std.std())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Bayesian Layer Statistics Summary\n{experiment_name}", fontsize=14)

    x = np.arange(len(layer_names))
    width = 0.35

    # Plot mean values
    ax = axes[0]
    ax.bar(x - width / 2, weight_mu_means, width, label="Weight Mean (μ)", alpha=0.8)
    ax.bar(x + width / 2, weight_std_means, width, label="Weight Std (σ)", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Value")
    ax.set_title("Average Parameter Values per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot std values
    ax = axes[1]
    ax.bar(x - width / 2, weight_mu_stds, width, label="Weight Mean Spread", alpha=0.8)
    ax.bar(x + width / 2, weight_std_stds, width, label="Weight Std Spread", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Parameter Variability per Layer")
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / "bayesian_parameters_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved summary plot to: {output_path}")


def print_statistics(bayesian_params):
    """Print detailed statistics about Bayesian parameters."""
    print("\n" + "=" * 80)
    print("BAYESIAN LAYER STATISTICS")
    print("=" * 80)

    for layer_name, params in bayesian_params.items():
        print(f"\n{layer_name}:")
        print("-" * 80)

        weight_mu = params["weight_mu"].flatten()
        weight_std = params["weight_std"].flatten()

        print(f"  Weight parameters: {len(weight_mu)}")
        print(
            f"  Weight μ: mean={weight_mu.mean():.6f}, std={weight_mu.std():.6f}, "
            f"min={weight_mu.min():.6f}, max={weight_mu.max():.6f}"
        )
        print(
            f"  Weight σ: mean={weight_std.mean():.6f}, std={weight_std.std():.6f}, "
            f"min={weight_std.min():.6f}, max={weight_std.max():.6f}"
        )

        uncertainty_ratio = weight_std / (np.abs(weight_mu) + 1e-8)
        print(
            f"  Uncertainty ratio (σ/|μ|): mean={uncertainty_ratio.mean():.6f}, "
            f"median={np.median(uncertainty_ratio):.6f}"
        )

        if "bias_mu" in params:
            bias_mu = params["bias_mu"].flatten()
            bias_std = params["bias_std"].flatten()
            print(f"  Bias parameters: {len(bias_mu)}")
            print(f"  Bias μ: mean={bias_mu.mean():.6f}, std={bias_mu.std():.6f}")
            print(f"  Bias σ: mean={bias_std.mean():.6f}, std={bias_std.std():.6f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nError: Please provide experiment folder path")
        sys.exit(1)

    experiment_path = sys.argv[1]
    experiment_dir = Path(experiment_path)

    if not experiment_dir.exists():
        # Try with experiments/ prefix
        experiment_dir = Path("experiments") / experiment_path
        if not experiment_dir.exists():
            print(f"Error: Experiment directory not found: {experiment_path}")
            sys.exit(1)

    experiment_name = experiment_dir.name
    print(f"Loading experiment: {experiment_name}")

    # Load config
    config = load_config(experiment_dir)
    config["device"] = torch.device("cpu")  # Load on CPU for inspection

    # Initialize feature registry
    print("Initializing feature registry...")
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry

    # Load model
    print("Loading model...")
    model = get_model(config).to(config["device"])

    # Load checkpoint
    checkpoint_path = find_model_checkpoint(experiment_dir)
    print(f"Loading checkpoint: {checkpoint_path.name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract Bayesian parameters
    print("Extracting Bayesian layer parameters...")
    bayesian_params = extract_bayesian_params(model)

    if not bayesian_params:
        print("Warning: No Bayesian layers found in this model!")
        print(
            "This script is designed for BayesianResNetSTEC or similar models with Bayesian layers."
        )
        sys.exit(1)

    print(f"Found {len(bayesian_params)} Bayesian layer(s)")

    # Print statistics
    print_statistics(bayesian_params)

    # Create output directory
    output_dir = experiment_dir / "bayesian_analysis"
    output_dir.mkdir(exist_ok=True)

    # Plot distributions
    print("\nGenerating plots...")
    plot_parameter_distributions(bayesian_params, experiment_name, output_dir)

    print("\n✓ Analysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
