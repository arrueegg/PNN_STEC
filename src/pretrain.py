from model.model import get_model
from data_loader import get_data_loaders
from training import BaseTrainer


class Pretrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.year = config["year"]
        self.doy = config["doy"]
        self.pretrain(logger)

    def initialize_model(self, model_seed):
        return get_model(self.config).to(self.device)

    def pretrain(self, logger):
        train_loader, val_loader, test_loader = get_data_loaders(self.config, logger)
        self.run_training(
            train_loader, val_loader, test_loader, self.initialize_model, "pretrain"
        )
