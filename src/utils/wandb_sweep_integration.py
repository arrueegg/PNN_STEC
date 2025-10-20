import os
import wandb
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def integrate_wandb_sweep_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Integrate wandb sweep parameters into the existing config.

    This function should be called after loading the base config but before
    initializing the model. It will override config values with wandb sweep
    parameters if a sweep is active.

    Args:
        config: Base configuration dictionary

    Returns:
        Updated configuration dictionary with wandb sweep parameters
    """

    # Check if we're in a wandb sweep
    logger.info(
        f"🔍 W&B integration check: wandb.run={wandb.run}, has_config_keys={hasattr(wandb.config, 'keys') if wandb.run else 'N/A'}"
    )
    if not wandb.run or not hasattr(wandb.config, "keys"):
        logger.info("No active wandb sweep detected, using base config")

        # Still check for cluster mode even without W&B sweep
        cluster_mode = os.environ.get("CLUSTER_MODE", "false").lower() == "true"
        logger.info(
            f"🖥️  Cluster mode check: CLUSTER_MODE={os.environ.get('CLUSTER_MODE', 'not set')}, cluster_mode={cluster_mode}"
        )
        if cluster_mode:
            logger.info("🔧 Applying cluster mode configuration without W&B sweep")
            config["cluster"] = True
            config["data"][
                "scratch_dir"
            ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/"
            config["data"][
                "GNSS_data_path"
            ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/STEC_DB_CASDCB"
            config["data"][
                "SWI_data_path"
            ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/SWI/"
            # By default, disable debug mode on cluster runs unless explicitly overridden
            config["debug"] = False
            logger.info("🔒 Debug mode disabled for cluster run")
            logger.info("✅ Applied cluster mode configuration")

        return config

    logger.info("🔄 Integrating wandb sweep parameters...")

    # Create a copy to avoid modifying the original
    updated_config = config.copy()

    # Map wandb sweep parameters to config structure
    sweep_mappings = {
        # Model parameters
        "model.model_type": ("model", "model_type"),
        "model.hidden_dim": ("model", "hidden_dim"),
        "model.num_layers": ("model", "num_layers"),
        # Training parameters
        "training.loss_function": ("training", "loss_function"),
        "training.loss_weight": ("training", "loss_weight"),
        "training.optimizer": ("training", "optimizer"),
        "training.weight_decay": ("training", "weight_decay"),
        # KL annealing parameters
        "training.kl_annealing.warmup_epochs": (
            "training",
            "kl_annealing",
            "warmup_epochs",
        ),
        # Target weighting parameters
        "training.target_weighting.enabled": (
            "training",
            "target_weighting",
            "enabled",
        ),
        "training.target_weighting.weight_function": (
            "training",
            "target_weighting",
            "weight_function",
        ),
        # Pretrain parameters
        "pretrain.learning_rate": ("pretrain", "learning_rate"),
        "pretrain.batchsize": ("pretrain", "batchsize"),
        "pretrain.scheduler": ("pretrain", "scheduler"),
        "pretrain.scheduler_step_size": ("pretrain", "scheduler_step_size"),
        # Data parameters
        "data.train_subset_size": ("data", "train_subset_size"),
    }

    # Apply sweep parameters
    applied_params = []
    for sweep_key, config_path in sweep_mappings.items():
        if hasattr(wandb.config, sweep_key.replace(".", "_")):
            # Get the value from wandb config
            wandb_key = sweep_key.replace(".", "_")
            value = getattr(wandb.config, wandb_key)

            # Navigate to the nested config location
            current = updated_config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the final value
            current[config_path[-1]] = value
            applied_params.append(f"{sweep_key} = {value}")

    # Handle special cases for model-specific parameters
    if hasattr(wandb.config, "model_model_type"):
        model_type = wandb.config.model_model_type

        # Set appropriate loss function for BNN models
        if "BNN" in model_type and not hasattr(wandb.config, "training_loss_function"):
            updated_config["training"]["loss_function"] = "GaussianNLLLoss"
            applied_params.append(
                "training.loss_function = GaussianNLLLoss (auto for BNN)"
            )
        elif "MLP" in model_type and not hasattr(
            wandb.config, "training_loss_function"
        ):
            updated_config["training"]["loss_function"] = "MSELoss"
            applied_params.append("training.loss_function = MSELoss (auto for MLP)")

    # Handle cluster mode configuration (similar to hyperparameter_search.py)
    cluster_mode = os.environ.get("CLUSTER_MODE", "false").lower() == "true"
    if cluster_mode:
        updated_config["cluster"] = True
        updated_config["data"][
            "scratch_dir"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/"
        updated_config["data"][
            "GNSS_data_path"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/STEC_DB_CASDCB"
        updated_config["data"][
            "SWI_data_path"
        ] = "/cluster/work/igp_psr/arrueegg/WP4/PNN_STEC/data/SWI/"
        applied_params.append("cluster mode enabled with updated data paths")

    # Force debug=False for all sweep runs (production mode)
    updated_config["debug"] = False
    applied_params.append("debug = False (forced for sweep runs)")

    # Log applied parameters
    if applied_params:
        logger.info("✅ Applied wandb sweep parameters:")
        for param in applied_params:
            logger.info(f"   {param}")
    else:
        logger.info("ℹ️  No wandb sweep parameters to apply")

    return updated_config


def setup_wandb_for_sweep(config: Dict[str, Any], experiment_name: str = None) -> None:
    """
    Setup wandb for sweep mode, handling both sweep and regular runs.

    Args:
        config: Configuration dictionary
        experiment_name: Optional experiment name override
    """

    # If wandb.run already exists (from wandb agent), just update config
    if wandb.run:
        logger.info(f"🔗 Using existing wandb run: {wandb.run.name}")
        # Update the run config with our full config for logging
        wandb.config.update(config, allow_val_change=True)
        return

    # If no active run, create one (for non-sweep runs)
    if not config.get("debug"):
        if not experiment_name:
            experiment_name = os.path.basename(config.get("output_dir", "experiment"))

        logger.info(f"🚀 Initializing wandb run: {experiment_name}")
        wandb.init(
            project=config.get("project_name", "PNN_STEC"),
            name=experiment_name,
            config=config,
        )


def log_sweep_metrics(metrics: Dict[str, float]) -> None:
    """
    Log metrics to wandb if a run is active.

    Args:
        metrics: Dictionary of metrics to log
    """
    if wandb.run:
        wandb.log(metrics)
