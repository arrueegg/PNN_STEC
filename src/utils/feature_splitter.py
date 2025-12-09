"""
Feature Splitting Utility for VTEC × MF Factorized Model

This module provides utilities to split the collated feature tensor into:
- x_vtec: Features for VTEC field prediction (IPP location, time, SWI)
- x_geom: Features for mapping factor prediction (station, elevation, azimuth)
- elev_rad: Elevation angle in radians (for MF constraint)

The feature splitter works with the transformed features output by CollateWithSH,
using the feature registry's output indices to locate specific features.
"""

import torch
from utils.feature_registry import FeatureType


class FeatureSplitter:
    """
    Splits collated feature tensors into VTEC-related and geometry-related components.
    
    This class uses the feature registry's output indices to extract the correct
    slices from the transformed feature tensor produced by CollateWithSH.
    """
    
    def __init__(self, feature_registry):
        """
        Initialize feature splitter with feature registry.
        
        Args:
            feature_registry: Initialized FeatureRegistry instance with output_indices set
        """
        self.feature_registry = feature_registry
        
        # Check if output_indices are available (set by CollateWithSH)
        if not hasattr(feature_registry, '_output_indices'):
            raise ValueError(
                "Feature registry does not have output_indices set. "
                "Make sure CollateWithSH has been initialized first, or call "
                "initialize_output_indices() before creating FeatureSplitter."
            )
        
        self.output_indices = feature_registry._output_indices
        
        # Pre-compute indices for VTEC features
        self.vtec_indices = self._compute_vtec_indices()
        
        # Pre-compute indices for Geometry features
        self.geom_indices = self._compute_geom_indices()
        
        # Store elevation index for extraction
        self.elev_idx = self._get_elevation_index()
        
    def _compute_vtec_indices(self):
        """
        Compute indices for VTEC-related features.
        
        VTEC features include:
        - Temporal features (year, doy, sod, local_time_hours)
        - IPP location features (lat_ipp, lon_ipp, sm_lat_ipp, sm_lon_ipp)
        - Space Weather Indices (SWI) if enabled
        - IPP spherical harmonic embeddings if enabled
        
        Excludes:
        - Station features (station-specific, not part of VTEC field)
        - Direction features (geometry-dependent, used for MF)
        """
        indices = []
        
        # 1. Temporal features (transformed: sin, cos, norm for cyclical; just norm for year)
        temporal_features = self.feature_registry.get_features_by_type(FeatureType.TEMPORAL)
        for feature_name in temporal_features:
            if feature_name == "year":
                indices.append(self.output_indices[f"{feature_name}_norm"])
            elif feature_name in ["doy", "sod", "local_time_hours"]:
                indices.extend([
                    self.output_indices[f"{feature_name}_sin"],
                    self.output_indices[f"{feature_name}_cos"],
                    self.output_indices[f"{feature_name}_norm"]
                ])
        
        # 2. IPP features (normalized)
        ipp_features = self.feature_registry.get_features_by_type(FeatureType.IPP)
        for feature_name in ipp_features:
            indices.append(self.output_indices[f"{feature_name}_norm"])
        
        # 3. IPP Spherical Harmonic embeddings (if enabled)
        if "sh_ipp_geo" in self.output_indices and self.output_indices["sh_ipp_geo"] is not None:
            sh_slice = self.output_indices["sh_ipp_geo"]
            indices.extend(range(sh_slice.start, sh_slice.stop))
        
        if "sh_ipp_sm" in self.output_indices and self.output_indices["sh_ipp_sm"] is not None:
            sh_slice = self.output_indices["sh_ipp_sm"]
            indices.extend(range(sh_slice.start, sh_slice.stop))
        
        # 4. SWI features (normalized) - global space weather conditions
        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            indices.append(self.output_indices[f"{feature_name}_norm"])
        
        return sorted(indices)
    
    def _compute_geom_indices(self):
        """
        Compute indices for geometry-related features.
        
        Geometry features include:
        - Station location features (lat_sta, lon_sta, sm_lat_sta, sm_lon_sta)
        - Direction features (elevation, azimuth as Cartesian unit vector)
        - Station spherical harmonic embeddings if enabled
        
        Note: Elevation is also extracted separately in radians for the MF constraint.
        """
        indices = []
        
        # 1. Station features (normalized)
        station_features = self.feature_registry.get_features_by_type(FeatureType.STATION)
        for feature_name in station_features:
            if f"{feature_name}_norm" in self.output_indices:
                indices.append(self.output_indices[f"{feature_name}_norm"])
        
        # 2. Direction features (Cartesian unit vector: e_up, e_east, e_north)
        # These are already computed from elevation and azimuth in CollateWithSH
        if "e_up" in self.output_indices:
            indices.extend([
                self.output_indices["e_up"],
                self.output_indices["e_east"],
                self.output_indices["e_north"]
            ])
        
        # 3. Station Spherical Harmonic embeddings (if enabled and station features exist)
        if "sh_sta_geo" in self.output_indices and self.output_indices["sh_sta_geo"] is not None:
            sh_slice = self.output_indices["sh_sta_geo"]
            indices.extend(range(sh_slice.start, sh_slice.stop))
        
        if "sh_sta_sm" in self.output_indices and self.output_indices["sh_sta_sm"] is not None:
            sh_slice = self.output_indices["sh_sta_sm"]
            indices.extend(range(sh_slice.start, sh_slice.stop))
        
        return sorted(indices)
    
    def _get_elevation_index(self):
        """
        Get the index of the elevation component (e_up) in the Cartesian direction vector.
        
        Returns:
            int: Index of e_up (sin(elevation)) in the feature tensor
        """
        if "e_up" in self.output_indices:
            return self.output_indices["e_up"]
        else:
            raise ValueError(
                "Elevation component (e_up) not found. "
                "Ensure direction features (satazi, satele) are enabled."
            )
    
    def split_features(self, features):
        """
        Split the collated feature tensor into VTEC and geometry components.
        
        Args:
            features: Tensor of shape [batch_size, total_features] from CollateWithSH
        
        Returns:
            tuple: (x_vtec, x_geom, elev_rad)
                - x_vtec: VTEC field features [batch_size, vtec_dim]
                - x_geom: Geometry features [batch_size, geom_dim]
                - elev_rad: Elevation in radians [batch_size]
        """
        # Extract VTEC features
        if len(self.vtec_indices) > 0:
            x_vtec = features[:, self.vtec_indices]
        else:
            # Handle case where no VTEC features are defined (unlikely but safe)
            x_vtec = torch.empty((features.shape[0], 0), device=features.device)
        
        # Extract Geometry features
        if len(self.geom_indices) > 0:
            x_geom = features[:, self.geom_indices]
        else:
            # Handle case where no geometry features are defined
            x_geom = torch.empty((features.shape[0], 0), device=features.device)
        
        # Extract elevation in radians
        # e_up = sin(elevation_rad), so we use arcsin to recover elevation
        e_up = features[:, self.elev_idx]
        
        # Clamp to valid range for arcsin to avoid numerical issues
        e_up_clamped = torch.clamp(e_up, -1.0, 1.0)
        elev_rad = torch.arcsin(e_up_clamped)
        
        return x_vtec, x_geom, elev_rad
    
    def get_vtec_dim(self):
        """Get the dimensionality of VTEC features."""
        return len(self.vtec_indices)
    
    def get_geom_dim(self):
        """Get the dimensionality of geometry features."""
        return len(self.geom_indices)
    
    def get_total_dim(self):
        """Get the total dimensionality of input features (for validation)."""
        return max(max(self.vtec_indices, default=-1), max(self.geom_indices, default=-1)) + 1
