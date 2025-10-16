"""
STEC Evaluation Package

This package provides memory-efficient evaluation tools for comparing
machine learning model STEC predictions against GIM VTEC mapped to STEC space.

Main Components:
- evaluator: Memory-efficient evaluation orchestrator  
- gim_mapper: GIM VTEC to STEC mapping functionality

The package uses DataLoader infrastructure to handle datasets of any size 
with constant memory usage and groups batches by date for efficient GIM loading.
"""

from .evaluator import EvaluationOrchestrator
from .gim_mapper import GIMMapper

__all__ = [
    'EvaluationOrchestrator',
    'GIMMapper'
]