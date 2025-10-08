"""
Training Module for PNN_STEC

This module provides a modular training framework for STEC prediction models.
It separates concerns of the original monolithic BaseTrainer into focused components.

Architecture:
    - base_trainer.py: Lightweight orchestrator using composition
    - data_transforms.py: Target normalization and feature transformations
    - training_utils.py: Timing, logging, KL annealing utilities
    - train_manager.py: Training epoch logic and optimization
    - validation_manager.py: Validation and evaluation logic
    - inference_manager.py: Testing and uncertainty estimation

Usage:
    from training import BaseTrainer

    trainer = BaseTrainer(config, logger)
    trainer.run_training(train_loader, val_loader, test_loader, init_model_fn, training_key)
"""

from .base_trainer import BaseTrainer
from .data_transforms import DataTransforms
from .training_utils import TrainingUtils
from .train_manager import TrainManager
from .validation_manager import ValidationManager
from .inference_manager import InferenceManager

__all__ = [
    "BaseTrainer",
    "DataTransforms",
    "TrainingUtils",
    "TrainManager",
    "ValidationManager",
    "InferenceManager",
]
