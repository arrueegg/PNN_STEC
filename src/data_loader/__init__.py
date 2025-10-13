"""
Data Module for PNN_STEC

This module provides a modular data handling framework for STEC prediction models.
It separates concerns of the original monolithic data.py into focused components.

Architecture:
    - datasets.py: Dataset classes (H5Dataset, H5RAMDataset, PyTablesDatasetSplit)
    - multitemporal_inference_dataset.py: Multi-temporal dataset for efficient map inference
    - samplers.py: Sampling utilities (EpochRandomSampler, subset management)
    - collation.py: Data collation and feature transformation (CollateWithSH)
    - loaders.py: Data loader creation and management
    - __init__.py: Public interface

Usage:
    from data import get_data_loaders, get_test_data_loader

    # Get data loaders (backward compatible)
    loaders = get_data_loaders(config, logger)
    train_loader = loaders['train']

    # For multi-temporal inference
    from data_loader.multitemporal_inference_dataset import create_multitemporal_inference_dataloader

    # Or use components directly
    from data import H5Dataset, EpochRandomSampler, CollateWithSH
    dataset = H5Dataset(config, path, split)
    sampler = EpochRandomSampler(dataset)
    collate_fn = CollateWithSH(config)
"""

from .loaders import get_data_loaders, get_test_data_loader
from .datasets import H5Dataset, H5RAMDataset, PyTablesDatasetSplit
from .multitemporal_inference_dataset import create_multitemporal_inference_dataloader
from .samplers import EpochRandomSampler, get_fixed_subset_indices
from .collation import CollateWithSH

__all__ = [
    # Main data loading functions (backward compatibility)
    "get_data_loaders",
    "get_test_data_loader",
    # Dataset classes
    "H5Dataset",
    "H5RAMDataset", 
    "PyTablesDatasetSplit",
    # Multi-temporal inference
    "create_multitemporal_inference_dataloader",
    # Sampling utilities
    "EpochRandomSampler",
    "get_fixed_subset_indices",
    # Collation
    "CollateWithSH",
]
