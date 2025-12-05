
# finetuner.py

import os
import logging
from model.model import get_model, init_kaiming
from data_loader import get_data_loaders
from training import BaseTrainer
from utils.model_utils import freeze_model_body

logger = logging.getLogger(__name__)

class Finetuner(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.year = config["year"]
        self.doy = config["doy"]
        # Kick off the finetuning process.
        self.finetune(logger)

    def initialize_model(self, model_seed):
        """
        Finetuner-specific model initialization.
        If in finetune mode, loads pretrained weights from pretrain_folder.
        """
        device = self.device
        model = get_model(self.config).to(device)
        pretrain_model_dir = os.path.join(self.config["pretrain_folder"], "model")
        pretrain_filename = f"pretrain_{self.config['model']['model_type']}_seed{model_seed:02}.pth"
        pretrain_checkpoint_path = os.path.join(pretrain_model_dir, pretrain_filename)
        if not os.path.exists(pretrain_checkpoint_path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_checkpoint_path}")
        import torch
        checkpoint = torch.load(pretrain_checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded pretrained weights from {pretrain_checkpoint_path}")
        
        # Freeze body parameters if configured (only train output head)
        freeze_model_body(model, self.config, logger)
        
        return model

    def finetune(self, logger):
        # Get single dataloaders.
        train_loader, val_loader, test_loader = get_data_loaders(self.config, logger)
        # Use the training configuration key
        training_key = self.config.get("mode", "finetune")
        self.run_training(
            train_loader, val_loader, test_loader, self.initialize_model, training_key
        )
