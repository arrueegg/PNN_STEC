# finetuner.py
import os
import torch
import wandb
import pandas as pd
from datetime import datetime
from model.model import get_model, init_kaiming
from utils.data import get_data_loaders
from utils.base_trainer import BaseTrainer

class Finetuner(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.year = config['year']
        self.doy = config['doy']
        # Kick off the finetuning process.
        self.finetune(logger)

    def initialize_model(self, model_seed):
        """
        Finetuner-specific model initialization.
        """
        device = self.device
        model = get_model(self.config).to(device)
        init_kaiming(model, self.config['model']['activation'], model_seed)
        return model

    def finetune(self, logger):
        # Get single dataloaders.
        train_loader, val_loader, test_loader = get_data_loaders(self.config, logger)
        # Use the training configuration key
        training_key = self.config.get('mode', 'finetune')
        self.run_training(train_loader, val_loader, test_loader, self.initialize_model, training_key)
