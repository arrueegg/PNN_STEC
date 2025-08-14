# pretrainer.py
import os
import torch
import wandb
import pandas as pd
from model.model import get_model, init_kaiming
from utils.data import get_data_loaders
from utils.base_trainer import BaseTrainer

class Pretrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.year = config['year']
        self.doy = config['doy']
        # Kick off the pretraining process.
        self.pretrain(logger)

    def initialize_model(self, model_seed):
        """
        Pretrainer-specific model initialization.
        Always uses Kaiming initialization.
        """
        device = self.device
        model = get_model(self.config).to(device)
        #init_kaiming(model, self.config["model"]["activation"], model_seed)
        return model

    def pretrain(self, logger):
        # Get multi dataloaders.
        train_loader, val_loader, test_loader = get_data_loaders(self.config, logger)
        # Use "pretrain" config parameters and "pretrain_model" as the checkpoint prefix.
        self.run_training(train_loader, val_loader, test_loader, self.initialize_model, "pretrain")
