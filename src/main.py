import torch
import numpy as np
import random
import logging
import os
import gc

from utils.config_parser import parse_config
from data_processing.add_split_indices import add_split_indices
from pretrain import Pretrainer
from finetune import Finetuner
from utils.feature_registry import initialize_feature_registry, FeatureType
from utils.wandb_sweep_integration import integrate_wandb_sweep_config

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main():
    config = parse_config()

    # Set wandb mode based on config and sweep status
    wandb_mode = "offline" if config.get("wandb", {}).get("offline", False) else "online"
    if "WANDB_SWEEP_ID" in os.environ:
        num_agents = int(os.environ.get("WANDB_AGENT_COUNT", 1))
        if num_agents > 1:
            wandb_mode = "online"
    os.environ["WANDB_MODE"] = wandb_mode

    # Integrate wandb sweep parameters if active
    config = integrate_wandb_sweep_config(config)

    setup_seed(config["random_seed"])

    gc.collect()
    torch.cuda.empty_cache()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    config["device"] = device

    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)

    # Add to config so other components can access them
    config["feature_registry"] = feature_registry

    # Log feature configuration
    print("")
    logger.info("=== Feature Configuration ===")
    logger.info(f"Total features: {feature_registry.get_total_features()}")
    for feature_type in FeatureType:
        features = feature_registry.get_features_by_type(feature_type)
        logger.info(f"{feature_type.value}: {len(features)} features")
        if features:
            logger.debug(f"  {feature_type.value} features: {features}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Add split indices to the data
    renew_splits = False
    if renew_splits:
        logger.info("Renewing split indices...")
        add_split_indices(config)
    data_path = config["data"]["GNSS_data_path"]

    if config["mode"] == "pretrain":
        print("")
        logger.info("Starting pretraining...")
        Pretrainer(config, logger)
    elif config["mode"] == "finetune":
        folder = f"{config['pretrain_folder']}/model/"
        model_path = os.listdir(folder) if os.path.exists(folder) else None
        if not model_path:
            print("")
            logger.info("Pretrained model not found.")
            logger.info("Starting pretraining...")
            # Preserve the feature registry when switching to pretrain mode
            original_feature_registry = config["feature_registry"]
            pretrain_config = parse_config(
                mode="pretrain", device=device, data_path=data_path
            )
            pretrain_config["feature_registry"] = original_feature_registry
            Pretrainer(pretrain_config, logger)
            # Preserve feature registry when switching back to finetune
            config = parse_config(mode="finetune", device=device, data_path=data_path)
            config["feature_registry"] = original_feature_registry
        print("")
        logger.info("Starting finetuning...")
        Finetuner(config, logger)
    else:
        logger.error('Invalid mode selected. Choose either "pretrain" or "finetune".')


if __name__ == "__main__":
    main()
