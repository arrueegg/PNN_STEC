"""
STEC Evaluation Package

This package provides tools for evaluating STEC predictions by comparing
machine learning model outputs against GIM VTEC mapped to STEC space.

Main Components:
- adapters: Observation data loading adapters  
- gim_mapper: GIM VTEC to STEC mapping functionality
- model_predictor: ML model STEC prediction functionality
- stec_eval: Core evaluation workflow coordination
"""

from .adapters import get_adapter
from .gim_mapper import GIMMapper
from .model_predictor import build_model_stec, ModelSTECPredictor

__all__ = [
    'get_adapter',
    'GIMMapper', 
    'build_model_stec',
    'ModelSTECPredictor'
]