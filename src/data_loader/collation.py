"""
Data collation and feature transformation functionality.
"""

import torch
from utils.locationencoder.pe import SphericalHarmonics
from utils.feature_registry import FeatureType


class CollateWithSH:
    """
    Collation function that transforms raw features into normalized features
    with optional spherical harmonic embeddings.

    This class handles:
    - Feature normalization based on the feature registry
    - Temporal feature transformations (sin/cos for cyclical features)
    - Spherical harmonic embeddings for spatial coordinates
    - Proper feature ordering and concatenation
    """

    def __init__(self, config):
        # Get feature registry
        self.feature_registry = config.get("feature_registry")

        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")

        # SH degree and flag
        self.sh_degree = config["data"].get("SH_degree", 0) or 0
        self.sh_enabled = self.sh_degree > 0
        if self.sh_enabled:
            # SphericalHarmonics produces degree² features for each location
            self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree)

        # Pre-compute feature slices for efficiency
        self.slices = {
            "temporal": self.feature_registry.get_feature_slice(FeatureType.TEMPORAL),
            "station": self.feature_registry.get_feature_slice(FeatureType.STATION),
            "direction": self.feature_registry.get_feature_slice(FeatureType.DIRECTION),
            "ipp": self.feature_registry.get_feature_slice(FeatureType.IPP),
            "swi": (
                self.feature_registry.get_feature_slice(FeatureType.SWI)
                if config["data"].get("use_SWI", False)
                else slice(0, 0)
            ),
        }

        # Get specific feature indices from registry (for raw input vector)
        self.input_indices = self._compute_input_feature_indices()

        # Compute and store output indices in the registry
        self.expected_dim = self._compute_and_store_output_indices()

    def _compute_input_feature_indices(self):
        """Compute feature indices for the RAW input vector (before transformation)"""
        indices = {}

        # Get all enabled features excluding target
        all_enabled_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)

        # Remove target features from the input features list
        input_features = [f for f in all_enabled_features if f not in target_features]

        # Build indices mapping feature names to their positions in the RAW input vector
        idx = 0
        for feature_name in input_features:
            # Skip SWI features in the main feature vector - they're appended separately
            if feature_name in self.feature_registry.get_features_by_type(
                FeatureType.SWI
            ):
                continue
            indices[feature_name] = idx
            idx += 1

        # Add SWI features at the end (they're concatenated after the main features)
        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            indices[feature_name] = idx
            idx += 1

        return indices

    def _compute_and_store_output_indices(self):
        """Compute and store output indices in the feature registry"""
        output_indices = {}
        current_idx = 0

        # Temporal features
        temporal_features = self.feature_registry.get_feature_names(
            FeatureType.TEMPORAL
        )
        for feature_name in temporal_features:
            if feature_name == "year":
                output_indices[f"{feature_name}_norm"] = current_idx
                current_idx += 1
            elif feature_name == "doy" or feature_name == "sod" or feature_name == "local_time_hours":
                output_indices[f"{feature_name}_sin"] = current_idx
                output_indices[f"{feature_name}_cos"] = current_idx + 1
                output_indices[f"{feature_name}_norm"] = current_idx + 2
                current_idx += 3

        # Station features
        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)
        for feature_name in station_features:
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1

        # Direction features - Cartesian unit vector (e_up, e_east, e_north)
        direction_features = self.feature_registry.get_feature_names(
            FeatureType.DIRECTION
        )
        if direction_features:
            # Check if we have both azimuth and elevation
            if "satazi" in direction_features and "satele" in direction_features:
                output_indices["e_up"] = current_idx
                output_indices["e_east"] = current_idx + 1
                output_indices["e_north"] = current_idx + 2
                current_idx += 3
            else:
                # Fallback to individual processing if not both present
                for feature_name in direction_features:
                    output_indices[f"{feature_name}_norm"] = current_idx
                    current_idx += 1

        # IPP features - only add if there are enabled IPP features
        ipp_features = self.feature_registry.get_feature_names(FeatureType.IPP)
        if ipp_features:
            for feature_name in ipp_features:
                output_indices[f"{feature_name}_norm"] = current_idx
                current_idx += 1

        # SH embeddings if enabled
        if self.sh_enabled:
            # Each location gets degree² SH features
            sh_dim = self.sh_degree * self.sh_degree

            # Check if station features are available (they're excluded for VTEC)
            has_station_features = (
                len(self.feature_registry.get_feature_names(FeatureType.STATION)) > 0
            )
            # Check if IPP features are available
            has_ipp_features = len(ipp_features) > 0

            # Store ranges for SH embeddings - only include station SH if station features available
            if has_station_features:
                output_indices["sh_sta_geo"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_sta_geo"] = None

            # Only add IPP SH embeddings if IPP features are available
            if has_ipp_features:
                output_indices["sh_ipp_geo"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_ipp_geo"] = None

            if has_station_features:
                output_indices["sh_sta_sm"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_sta_sm"] = None

            # Only add IPP SM SH embeddings if IPP features are available
            if has_ipp_features:
                output_indices["sh_ipp_sm"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_ipp_sm"] = None

        # SWI features
        swi_features = self.feature_registry.get_feature_names(FeatureType.SWI)
        for feature_name in swi_features:
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1

        # Store in registry
        self.feature_registry.set_output_indices(output_indices)

        return current_idx

    def transform_temporal(self, features):
        """Transform temporal features using feature registry"""
        temporal_features = self.feature_registry.get_feature_names(
            FeatureType.TEMPORAL
        )
        transformed_features = []

        for feature_name in temporal_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]

            if feature_name == "year":
                # Normalize year
                year_norm = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                transformed_features.extend([year_norm])
            elif feature_name == "doy":
                # Day of year transformations
                doy_norm = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                doy_sin = torch.sin(doy_norm * 2 * torch.pi)
                doy_cos = torch.cos(doy_norm * 2 * torch.pi)
                transformed_features.extend([doy_sin, doy_cos, doy_norm])
            elif feature_name == "sod":
                # Time of day transformations
                norm_sod = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                sin_sod = torch.sin(norm_sod * 2 * torch.pi)
                cos_sod = torch.cos(norm_sod * 2 * torch.pi)
                transformed_features.extend([sin_sod, cos_sod, norm_sod])
            elif feature_name == "local_time_hours":
                # Local time transformations (cyclical feature)
                norm_local_time = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                sin_local_time = torch.sin(norm_local_time * 2 * torch.pi)
                cos_local_time = torch.cos(norm_local_time * 2 * torch.pi)
                transformed_features.extend([sin_local_time, cos_local_time, norm_local_time])
            else:
                raise ValueError(f"Unexpected temporal feature: {feature_name}")

        return torch.cat(transformed_features, dim=1)

    def transform_station(self, features):
        """Transform station features"""
        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)

        # Return None if no station features are available (e.g., for VTEC)
        if not station_features:
            return None

        transformed_features = []

        for feature_name in station_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(
                feature_name, feature_values
            ).unsqueeze(1)
            transformed_features.extend([feature_norm])

        return torch.cat(transformed_features, dim=1)

    def transform_direction(self, features):
        """Transform direction features (azimuth, elevation) to Cartesian unit vector"""
        direction_features = self.feature_registry.get_feature_names(
            FeatureType.DIRECTION
        )

        # Return None if no direction features are available (e.g., for VTEC)
        if not direction_features:
            return None

        # We expect both satazi and satele to be present
        if "satazi" not in direction_features or "satele" not in direction_features:
            raise ValueError("Both 'satazi' and 'satele' must be present for Cartesian transformation")

        # Get indices for azimuth and elevation
        azi_idx = self.input_indices["satazi"]
        ele_idx = self.input_indices["satele"]

        # Get raw values (in degrees, not yet normalized)
        azimuth_deg = features[:, azi_idx]  # 0-360 degrees
        elevation_deg = features[:, ele_idx]  # 0-90 degrees

        # Convert to radians
        azimuth_rad = azimuth_deg * torch.pi / 180.0
        elevation_rad = elevation_deg * torch.pi / 180.0

        # Compute Cartesian unit vector components
        e_up = torch.sin(elevation_rad).unsqueeze(1)  # Vertical component
        e_east = (torch.cos(elevation_rad) * torch.sin(azimuth_rad)).unsqueeze(1)  # Eastward component
        e_north = (torch.cos(elevation_rad) * torch.cos(azimuth_rad)).unsqueeze(1)  # Northward component

        # Return the 3D unit vector
        return torch.cat([e_up, e_east, e_north], dim=1)

    def transform_ipp(self, features):
        """Transform IPP features"""
        ipp_features = self.feature_registry.get_feature_names(FeatureType.IPP)
        
        if not ipp_features:
            return None
        
        transformed_features = []

        for feature_name in ipp_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(
                feature_name, feature_values
            ).unsqueeze(1)
            transformed_features.extend([feature_norm])

        return torch.cat(transformed_features, dim=1)

    def transform_swi(self, features):
        """Transform SWI features using feature registry"""
        swi_features = self.feature_registry.get_feature_names(FeatureType.SWI)

        if not swi_features:
            return None

        transformed_features = []

        for feature_name in swi_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]
            feature_norm = self.feature_registry.normalize_feature(
                feature_name, feature_values
            ).unsqueeze(1)
            transformed_features.append(feature_norm)

        return torch.cat(transformed_features, dim=1)

    def compute_sh_embeddings(self, features):
        """Compute spherical harmonic embeddings for station and IPP"""
        if not self.sh_enabled:
            return None, None, None, None

        # Check if station features are available (they're excluded for VTEC)
        has_station_features = (
            "lon_sta" in self.input_indices and "lat_sta" in self.input_indices
        )
        has_sm_station_features = (
            "sm_lon_sta" in self.input_indices and "sm_lat_sta" in self.input_indices
        )
        
        # Check if IPP features are available
        has_ipp_features = (
            "lon_ipp" in self.input_indices and "lat_ipp" in self.input_indices
        )
        has_sm_ipp_features = (
            "sm_lon_ipp" in self.input_indices and "sm_lat_ipp" in self.input_indices
        )

        # Station SH embeddings (use geographic coordinates for SH) - only if station features available
        if has_station_features:
            sta_lon = features[:, self.input_indices["lon_sta"]]
            sta_lat = features[:, self.input_indices["lat_sta"]]
            sta_lonlat = torch.stack([sta_lon, sta_lat], dim=1)
            sh_sta_geo = self.sh_encoder(sta_lonlat)
        else:
            sh_sta_geo = None

        # IPP SH embeddings (use geographic coordinates for SH) - only if IPP features available
        if has_ipp_features:
            ipp_lon = features[:, self.input_indices["lon_ipp"]]
            ipp_lat = features[:, self.input_indices["lat_ipp"]]
            ipp_lonlat = torch.stack([ipp_lon, ipp_lat], dim=1)
            sh_ipp_geo = self.sh_encoder(ipp_lonlat)
        else:
            sh_ipp_geo = None

        # Station SH embedding (use solar magnetic coordinates for SH) - only if station features available
        if has_sm_station_features:
            sm_lon_sta = features[:, self.input_indices["sm_lon_sta"]]
            sm_lat_sta = features[:, self.input_indices["sm_lat_sta"]]
            sm_lonlat_sta = torch.stack([sm_lon_sta, sm_lat_sta], dim=1)
            sh_sta_sm = self.sh_encoder(sm_lonlat_sta)
        else:
            sh_sta_sm = None

        # IPP SH embedding (use solar magnetic coordinates for SH) - only if IPP features available
        if has_sm_ipp_features:
            sm_lon_ipp = features[:, self.input_indices["sm_lon_ipp"]]
            sm_lat_ipp = features[:, self.input_indices["sm_lat_ipp"]]
            sm_lonlat_ipp = torch.stack([sm_lon_ipp, sm_lat_ipp], dim=1)
            sh_ipp_sm = self.sh_encoder(sm_lonlat_ipp)
        else:
            sh_ipp_sm = None

        # Return in the order expected by _compute_and_store_output_indices
        return sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm

    def __call__(self, batch):
        """Process and collate a batch of (features, labels) or (features, labels, metadata)"""
        # Check if batch contains metadata (3-tuple) or just features and labels (2-tuple)
        if len(batch[0]) == 3:
            # Batch contains metadata
            feats, labels, metadata_list = zip(*batch)
            has_metadata = True
        else:
            # Standard batch without metadata
            feats, labels = zip(*batch)
            has_metadata = False
            metadata_list = None
        
        features = torch.stack(feats, dim=0)
        labels = torch.stack(labels, dim=0)

        # Transform different feature types
        temporal_transformed = self.transform_temporal(features)
        station_transformed = self.transform_station(features)
        direction_transformed = self.transform_direction(features)
        ipp_transformed = self.transform_ipp(features)
        swi_transformed = self.transform_swi(features)

        # Compute SH embeddings if enabled
        sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm = self.compute_sh_embeddings(
            features
        )

        # Combine all transformed features in the SAME ORDER as _compute_and_store_output_indices
        output_features = [temporal_transformed]

        # Only add station and direction features if they exist (excluded for VTEC)
        if station_transformed is not None:
            output_features.append(station_transformed)
        if direction_transformed is not None:
            output_features.append(direction_transformed)

        # Add IPP features if they exist
        if ipp_transformed is not None:
            output_features.append(ipp_transformed)

        # Add SH embeddings if computed (in the exact order from _compute_and_store_output_indices)
        if self.sh_enabled:
            # Only add station SH embeddings if they exist (they're excluded for VTEC)
            if sh_sta_geo is not None:
                output_features.append(sh_sta_geo)
            # Only add IPP SH embeddings if they exist (they're excluded when IPP features are disabled)
            if sh_ipp_geo is not None:
                output_features.append(sh_ipp_geo)
            if sh_sta_sm is not None:
                output_features.append(sh_sta_sm)
            if sh_ipp_sm is not None:
                output_features.append(sh_ipp_sm)

        # Add SWI features if available
        if swi_transformed is not None:
            output_features.append(swi_transformed)

        # Concatenate all features
        final_features = torch.cat(output_features, dim=1)

        # Return with or without metadata
        if has_metadata:
            return final_features, labels, metadata_list
        else:
            return final_features, labels
