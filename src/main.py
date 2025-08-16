import torch
import numpy as np
import random
import logging
import os

from utils.config_parser import parse_config
from data_processing.add_split_indices import add_split_indices
from pretrain import Pretrainer
from finetune import Finetuner
from utils.feature_registry import initialize_feature_registry, FeatureType

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    config = parse_config()
    setup_seed(config['random_seed'])

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    config['device'] = device

    # Initialize feature registry
    feature_registry = initialize_feature_registry(config)
        
    # Add to config so other components can access them
    config['feature_registry'] = feature_registry
    
    # Log feature configuration
    logger.info("=== Feature Configuration ===")
    logger.info(f"Total features: {feature_registry.get_total_features()}")
    for feature_type in FeatureType:
        features = feature_registry.get_features_by_type(feature_type)
        logger.info(f"{feature_type.value}: {len(features)} features")
        if features:
            logger.debug(f"  {feature_type.value} features: {features}")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Split data into train, validation, and test sets
    #data_path = "./temp_data/"
    #os.makedirs(data_path, exist_ok=True)
    #split(config, data_path)
    #config['data']['GNSS_data_path'] = data_path

    # Add split indices to the data
    renew_splits = False
    if renew_splits:
        logger.info("Renewing split indices...")
        add_split_indices(config)
    data_path = config['data']['GNSS_data_path']

    logger.info(f"Starting model training in {config['mode']} mode.")
    
    if config['mode'] == 'pretrain':
        logging.info('Starting pretraining...')
        trainer = Pretrainer(config, logger)
    elif config['mode'] == 'finetune':
        folder = f"{config['pretrain_folder']}/model/"
        model_path = os.listdir(folder)
        if not model_path:
            logging.info(f"Pretrained model not found.")
            logging.info('Starting pretraining...')
            config = parse_config(mode='pretrain', device=device, data_path=data_path)
            Pretrainer(config, logger)
            config = parse_config(mode='finetune', device=device, data_path=data_path)
        logging.info('Starting finetuning...')
        trainer = Finetuner(config, logger)
    else:
        logging.error('Invalid mode selected. Choose either "pretrain" or "finetune".')
        
if __name__ == '__main__':
    main()