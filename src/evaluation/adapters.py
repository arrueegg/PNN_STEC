#!/usr/bin/env python3
"""
Observation source adapters for STEC evaluation.

Provides standardized interfaces to different observation datasets:
- Testset: Held-out GNSS test data (ground truth STEC)
- Madrigal: Madrigal LOS products (STEC where available)
- Grid: Synthetic/gridded observations (diagnostic only)
- VgosDB: VLBI ΔTEC for independent ΔSTEC validation
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DatasetAdapter(ABC):
    """Protocol for observation source adapters."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the dataset."""
        pass
    
    @abstractmethod
    def prepare_observations(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return standardized observation placeholders for evaluation.
        
        Expected return structure:
        {
            'times': array-like,          # Observation timestamps
            'stations': array-like,       # Station identifiers
            'satellites': array-like,     # Satellite identifiers  
            'elevations': array-like,     # Elevation angles
            'ipp_lat': array-like,        # Ionospheric pierce point latitudes
            'ipp_lon': array-like,        # Ionospheric pierce point longitudes
            'stec_obs': array-like,       # Observed STEC (if available)
            'has_truth': bool,            # Whether ground truth is available
            'metadata': dict              # Additional dataset-specific info
        }
        """
        pass


class TestsetAdapter(DatasetAdapter):
    """Adapter for held-out GNSS test dataset (ground truth STEC)."""
    
    @property
    def name(self) -> str:
        return "Testset (GNSS ground truth)"
    
    def prepare_observations(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Load and prepare testset observations."""
        logger.info(f"Preparing {self.name} observations")
        logger.info(f"  Would load GNSS data from: {cfg.get('gnss_path', 'NOT_SET')}")
        logger.info("  Would extract: times, stations, satellites, elevations, IPP coords")
        logger.info("  Would provide ground truth STEC values")
        
        # Return empty placeholders
        return {
            'times': [],
            'stations': [],
            'satellites': [],
            'elevations': [],
            'ipp_lat': [],
            'ipp_lon': [],
            'stec_obs': [],  # Ground truth available
            'has_truth': True,
            'metadata': {'source': 'testset', 'n_obs': 0}
        }


class MadrigalAdapter(DatasetAdapter):
    """Adapter for Madrigal LOS products (STEC where available)."""
    
    @property
    def name(self) -> str:
        return "Madrigal LOS"
    
    def prepare_observations(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Load and prepare Madrigal observations."""
        logger.info(f"Preparing {self.name} observations")
        logger.info(f"  Would load Madrigal data from: {cfg.get('madrigal_path', 'NOT_SET')}")
        logger.info("  Would extract LOS STEC where available")
        logger.info("  Would handle sparse/irregular coverage")
        
        return {
            'times': [],
            'stations': [],
            'satellites': [],
            'elevations': [],
            'ipp_lat': [],
            'ipp_lon': [],
            'stec_obs': [],  # Sparse truth available
            'has_truth': True,
            'metadata': {'source': 'madrigal', 'n_obs': 0, 'coverage': 'sparse'}
        }


class GridAdapter(DatasetAdapter):
    """Adapter for synthetic/gridded observations (diagnostic only)."""
    
    @property
    def name(self) -> str:
        return "Synthetic Grid"
    
    def prepare_observations(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Generate synthetic gridded observations."""
        logger.info(f"Preparing {self.name} observations")
        logger.info("  Would generate systematic grid of LOS")
        logger.info("  Would cover global/regional domain")
        logger.info("  Diagnostic mode: no ground truth available")
        
        return {
            'times': [],
            'stations': [],
            'satellites': [],
            'elevations': [],
            'ipp_lat': [],
            'ipp_lon': [],
            'stec_obs': [],  # No truth for synthetic grid
            'has_truth': False,
            'metadata': {'source': 'grid', 'n_obs': 0, 'type': 'synthetic'}
        }


class VgosdbAdapter(DatasetAdapter):
    """Adapter for VLBI ΔTEC (dTEC) for independent ΔSTEC validation."""
    
    @property
    def name(self) -> str:
        return "VLBI ΔTEC (VgosDB)"
    
    def prepare_observations(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """TODO: Load and prepare VLBI baseline observations."""
        logger.info(f"Preparing {self.name} observations")
        logger.info(f"  Would load VgosDB data from: {cfg.get('vgosdb_path', 'NOT_SET')}")
        logger.info("  Would compute two LOS per baseline")
        logger.info("  Would extract ΔTEC for ΔSTEC comparison")
        
        return {
            'times': [],
            'stations': [],  # Baseline pairs
            'satellites': [],  # Radio sources
            'elevations': [],
            'ipp_lat': [],
            'ipp_lon': [],
            'stec_obs': [],  # ΔTEC converted to ΔSTEC
            'has_truth': True,
            'metadata': {'source': 'vlbi', 'n_obs': 0, 'type': 'baseline_difference'}
        }


def get_adapter(dataset_type: str) -> DatasetAdapter:
    """Factory function to get appropriate adapter by name."""
    adapters = {
        'testset': TestsetAdapter(),
        'madrigal': MadrigalAdapter(),
        'grid': GridAdapter(),
        'vgosdb': VgosdbAdapter()
    }
    
    if dataset_type not in adapters:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from: {list(adapters.keys())}")
    
    return adapters[dataset_type]