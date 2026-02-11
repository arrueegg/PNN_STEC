
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
        If in finetune mode, loads pretrained weights from pretrain_folder (unless finetune_from_scratch=True).
        """
        device = self.device
        model = get_model(self.config).to(device)
        
        # Check if we should finetune from scratch (no pretrained weights)
        finetune_from_scratch = self.config.get("finetune_from_scratch", False)
        
        if finetune_from_scratch:
            logger.info("⚠️  Finetuning from scratch (no pretrained weights loaded)")
            logger.info("   This is useful for single-day training without pretrain phase")
        else:
            # Load pretrained weights
            pretrain_model_dir = os.path.join(self.config["pretrain_folder"], "model")
            pretrain_filename = f"pretrain_{self.config['model']['model_type']}_seed{model_seed}.pth"
            pretrain_checkpoint_path = os.path.join(pretrain_model_dir, pretrain_filename)
            if not os.path.exists(pretrain_checkpoint_path):
                raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrain_checkpoint_path}")
            import torch
            checkpoint = torch.load(pretrain_checkpoint_path, weights_only=True, map_location=device)
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
        
        # [PAPER] Mao et al. 2025: Train ensemble if enabled
        train_ensemble = self.config.get("finetune", {}).get("train_ensemble", False)
        ensemble_size = self.config.get("finetune", {}).get("ensemble_size", 1)
        parallel_ensemble = self.config.get("finetune", {}).get("parallel_ensemble", False)
        
        if train_ensemble and ensemble_size > 1:
            logger.info(f"[PAPER] Mao et al. 2025: Training ensemble with {ensemble_size} members")
            
            if parallel_ensemble:
                self._train_ensemble_parallel(
                    train_loader, val_loader, test_loader, training_key, logger
                )
            else:
                self._train_ensemble_sequential(
                    train_loader, val_loader, test_loader, training_key, logger
                )
        else:
            # Standard single model training
            self.run_training(
                train_loader, val_loader, test_loader, self.initialize_model, training_key
            )

    def _train_ensemble_sequential(self, train_loader, val_loader, test_loader, training_key, logger):
        """Train ensemble members sequentially (original behavior)"""
        base_seed = self.config["random_seed"]
        
        for member_idx in range(self.config["finetune"]["ensemble_size"]):
            logger.info(f"\n{'='*80}")
            logger.info(f"Ensemble Member {member_idx + 1}/{self.config['finetune']['ensemble_size']}")
            logger.info(f"{'='*80}")
            
            # Use unique seed for each ensemble member
            ensemble_seed = base_seed + member_idx
            self.config["random_seed"] = ensemble_seed
            
            # Setup seed for this ensemble member
            import torch
            import numpy as np
            import random
            torch.manual_seed(ensemble_seed)
            torch.cuda.manual_seed_all(ensemble_seed)
            
            # Use cluster speed optimizations unless in debug mode
            if not self.config.get("debug", False):
                torch.backends.cudnn.deterministic = False
                torch.backends.cudnn.benchmark = True
            else:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
            np.random.seed(ensemble_seed)
            random.seed(ensemble_seed)
            
            logger.info(f"Ensemble member {member_idx} using seed: {ensemble_seed}")
            
            # Train this ensemble member
            self.run_training(
                train_loader, val_loader, test_loader, self.initialize_model, training_key,
                ensemble_member=member_idx, ensemble_size=self.config["finetune"]["ensemble_size"]
            )

    def _train_ensemble_parallel(self, train_loader, val_loader, test_loader, training_key, logger):
        """Train ensemble members in parallel using joblib and GPU distribution"""
        from joblib import Parallel, delayed
        import torch
        import copy
        
        base_seed = self.config["random_seed"]
        ensemble_size = self.config["finetune"]["ensemble_size"]
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            logger.warning("⚠️  Parallel ensemble training requested but no GPUs available. Falling back to sequential.")
            self._train_ensemble_sequential(train_loader, val_loader, test_loader, training_key, logger)
            return
        
        logger.info(f"🚀 Parallel ensemble training: {ensemble_size} members on {num_gpus} GPU(s)")
        n_jobs = min(num_gpus, ensemble_size)
        logger.info(f"   Using {n_jobs} parallel jobs")
        
        # Define training function for each member
        def train_member(member_idx, config_copy, base_seed_val):
            """Train a single ensemble member in a separate process"""
            import torch
            import numpy as np
            import random
            
            # Assign GPU based on member index (round-robin)
            gpu_id = member_idx % num_gpus
            torch.cuda.set_device(gpu_id)
            
            # Setup seed for this member
            ensemble_seed = base_seed_val + member_idx
            torch.manual_seed(ensemble_seed)
            torch.cuda.manual_seed_all(ensemble_seed)
            
            # Use cluster speed optimizations
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            
            np.random.seed(ensemble_seed)
            random.seed(ensemble_seed)
            
            # Update config for this member
            config_copy["random_seed"] = ensemble_seed
            
            # Create a local trainer instance for this member
            from training import BaseTrainer
            trainer = BaseTrainer(config_copy, logger=None)  # logger=None to avoid conflicts
            trainer.device = torch.device(f"cuda:{gpu_id}")
            
            # Train this member
            trainer.run_training(
                train_loader, val_loader, test_loader, 
                self.initialize_model, training_key,
                ensemble_member=member_idx, ensemble_size=ensemble_size
            )
            
            print(f"✅ Ensemble member {member_idx + 1}/{ensemble_size} completed on GPU {gpu_id}")
        
        # Create a config copy for each member (avoid state sharing)
        configs = [copy.deepcopy(self.config) for _ in range(ensemble_size)]
        
        # Run parallel training
        logger.info(f"Starting parallel training of {ensemble_size} ensemble members...")
        Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
            delayed(train_member)(idx, configs[idx], base_seed) 
            for idx in range(ensemble_size)
        )
        
        logger.info(f"✅ All {ensemble_size} ensemble members trained successfully!")
