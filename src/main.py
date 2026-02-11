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
from utils.feature_registry import initialize_feature_registry, FeatureType, print_feature_summary
from utils.wandb_sweep_integration import integrate_wandb_sweep_config

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

def setup_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)


def main(config_path=None):
    config = parse_config(config_path=config_path)

    # Set wandb mode based on config and sweep status
    wandb_mode = "offline" if config.get("wandb", {}).get("offline", False) else "online"
    if "WANDB_SWEEP_ID" in os.environ:
        num_agents = int(os.environ.get("WANDB_AGENT_COUNT", 1))
        if num_agents > 1:
            wandb_mode = "online"
    os.environ["WANDB_MODE"] = wandb_mode
    logger.info(f"Set WANDB_MODE to {wandb_mode} (config offline: {config.get('wandb', {}).get('offline', False)}, sweep: {'WANDB_SWEEP_ID' in os.environ}, agents: {os.environ.get('WANDB_AGENT_COUNT', 1)})")

    # Integrate wandb sweep parameters if active
    config = integrate_wandb_sweep_config(config)

    # Enable benchmarking for speed unless in debug mode or explicit deterministic mode
    is_deterministic = config.get("deterministic", config.get("debug", False))
    setup_seed(config["random_seed"], deterministic=is_deterministic)

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

    # Print feature summary
    print_feature_summary(feature_registry, config)

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
        # Check if we should finetune from scratch (no pretrained weights)
        finetune_from_scratch = config.get("finetune_from_scratch", False)
        
        if finetune_from_scratch:
            # Skip pretrain loading - train from scratch on single day
            logger.info("⚠️  Finetuning from scratch (no pretrained model required)")
            config["pretrain_folder"] = None  # Explicitly set to None
            print("")
            logger.info("Starting finetuning from scratch...")
            Finetuner(config, logger)
        else:
            # Try to find pretrained model in experiments folder
            # First check if pretrain_folder is explicitly specified
            if "pretrain_folder" in config:
                pretrain_folder = config["pretrain_folder"]
                logger.info(f"Using specified pretrain_folder: {pretrain_folder}")
            else:
                # Auto-detect based on current experiment settings
                # This will match the pretrain experiment with same model settings
                pretrain_folder = None
                experiments_dir = config.get("output_dir", "experiments/")
                if os.path.exists(experiments_dir):
                    # Look for matching pretrain experiment
                    model_type = config['model']['model_type']
                    for exp_dir in os.listdir(experiments_dir):
                        if exp_dir.startswith(f"Pretrain_STEC_{model_type}"):
                            pretrain_folder = os.path.join(experiments_dir, exp_dir)
                            logger.info(f"Auto-detected pretrain_folder: {pretrain_folder}")
                            break
            
            if pretrain_folder and os.path.exists(pretrain_folder):
                config["pretrain_folder"] = pretrain_folder
                model_folder = os.path.join(pretrain_folder, "model")
                
                if os.path.exists(model_folder) and os.listdir(model_folder):
                    logger.info(f"Found pretrained model in: {model_folder}")
                    print("")
                    logger.info("Starting finetuning...")
                    Finetuner(config, logger)
                else:
                    logger.error(f"Model folder exists but is empty: {model_folder}")
            else:
                logger.error("Pretrained model not found.")
                logger.error("Either specify 'pretrain_folder' in config.yaml or ensure a pretrain experiment exists.")
                logger.error("Alternatively, set 'finetune_from_scratch: true' to train on single day without pretrain.")
                if "pretrain_folder" in config:
                    logger.error(f"Specified folder does not exist: {config['pretrain_folder']}")
    else:
        logger.error('Invalid mode selected. Choose either "pretrain" or "finetune".')


if __name__ == "__main__":
    main()
