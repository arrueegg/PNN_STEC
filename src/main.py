import torch
import numpy as np
import random
import logging
import os

from utils.config_parser import parse_config
from utils.pre_split_subsample_data import split
from data_processing.add_split_indices import add_split_indices
from pretrain import Pretrainer
from finetune import Finetuner


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
    config['device'] = device

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Split data into train, validation, and test sets
    #data_path = "./temp_data/"
    #os.makedirs(data_path, exist_ok=True)
    #split(config, data_path)
    #config['data']['GNSS_data_path'] = data_path

    # Add split indices to the data
    add_split_indices(config)
    data_path = config['data']['GNSS_data_path']

    logger.info(f"Starting model training in {config['mode']} mode.")
    
    if config['mode'] == 'pretrain':
        logging.info('Starting pretraining...')
        Pretrainer(config, logger)
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
        Finetuner(config, logger)
    else:
        logging.error('Invalid mode selected. Choose either "pretrain" or "finetune".')

if __name__ == '__main__':
    main()