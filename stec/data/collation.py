"""`CollateWithSH`, ported from `src/data_loader/collation.py`.

Assembles a raw feature vector into the normalized, transformed tensor a model consumes
(cyclical sin/cos temporal features, Cartesian direction unit vector, spherical-harmonic
embeddings), driven by a `stec.data.feature_registry.FeatureRegistry` instance rather than
the paper pipeline's `FeatureLayout`.

This is a deliberate second implementation, not a stopgap for a missing one:
`stec.data.transforms.FeatureAssembler` is the paper pipeline's own equivalent, verified
bit-exact against this exact class in Gate A. But `FeatureAssembler` is built from a
`FeatureLayout` computed once from `feature_control` at construction time, while every
operational script this class serves (`positioning/scripts/generate_stec_corrections.py`,
`scripts/infer_from_log.py`, `scripts/show_feature_splits.py`,
`vlbi_kband/scripts/infer_vlbi_kband.py`) builds its feature set from an arbitrary
experiment's `config.yaml` via the legacy `FeatureRegistry` object and needs this class's
`__call__(batch)` collate-function contract to hand straight to a `torch.utils.data.DataLoader`.
Retrofitting `FeatureAssembler` to that contract would be the kind of "quiet behaviour
change to make two things share code" `docs/rebuild_plan.md` warns against; keeping this as
an independent, faithful port keeps the equivalence with the legacy pipeline these scripts
were written against exact.

Uses `stec.data.spherical_harmonics.SphericalHarmonics` (the byte-identical port of
`utils.locationencoder.pe.SphericalHarmonics`) rather than the legacy package itself.
"""

from __future__ import annotations

import torch

from .feature_registry import FeatureType
from .spherical_harmonics import SphericalHarmonics


class CollateWithSH:
    """Collation function transforming raw features into normalized features, with
    optional spherical-harmonic embeddings.

    Handles feature normalization based on the feature registry, temporal sin/cos
    transforms, spherical-harmonic embeddings for spatial coordinates, and consistent
    feature ordering and concatenation.
    """

    def __init__(self, config: dict) -> None:
        self.feature_registry = config.get("feature_registry")

        if not self.feature_registry:
            raise ValueError("Feature registry is required but not found in config")

        self.sh_degree = config["data"].get("SH_degree", 0) or 0
        self.sh_enabled = self.sh_degree > 0
        self.target = config.get("target", "stec")
        self.model_type = config.get("model", {}).get("model_type", "")

        if self.sh_enabled:
            # [LEGACY] STEC ResNet models: legendre_polys = sh_degree (sh_degree**2 features)
            # [PAPER] Mao et al. 2025 VTEC: legendre_polys = sh_degree + 1
            if self.target == "vtec" or "Laplacian" in self.model_type:
                self.sh_encoder = SphericalHarmonics(legendre_polys=self.sh_degree + 1)
            else:
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

        self.input_indices = self._compute_input_feature_indices()
        self.expected_dim = self._compute_and_store_output_indices()

    def _compute_input_feature_indices(self) -> dict[str, int]:
        """Feature indices for the RAW input vector (before transformation)."""
        indices: dict[str, int] = {}

        all_enabled_features = self.feature_registry.get_all_enabled_features()
        target_features = self.feature_registry.get_features_by_type(FeatureType.TARGET)

        input_features = [f for f in all_enabled_features if f not in target_features]

        idx = 0
        for feature_name in input_features:
            # SWI features are appended separately, at the end.
            if feature_name in self.feature_registry.get_features_by_type(
                FeatureType.SWI
            ):
                continue
            indices[feature_name] = idx
            idx += 1

        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            indices[feature_name] = idx
            idx += 1

        return indices

    def _compute_and_store_output_indices(self) -> int:
        """Compute and store output indices in the feature registry."""
        output_indices: dict = {}
        current_idx = 0

        temporal_features = self.feature_registry.get_feature_names(
            FeatureType.TEMPORAL
        )
        for feature_name in temporal_features:
            if feature_name == "year":
                output_indices[f"{feature_name}_norm"] = current_idx
                current_idx += 1
            elif feature_name in ("doy", "sod", "local_time_hours"):
                output_indices[f"{feature_name}_sin"] = current_idx
                output_indices[f"{feature_name}_cos"] = current_idx + 1
                output_indices[f"{feature_name}_norm"] = current_idx + 2
                current_idx += 3

        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)
        for feature_name in station_features:
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1

        direction_features = self.feature_registry.get_feature_names(
            FeatureType.DIRECTION
        )
        if direction_features:
            if "satazi" in direction_features and "satele" in direction_features:
                output_indices["e_up"] = current_idx
                output_indices["e_east"] = current_idx + 1
                output_indices["e_north"] = current_idx + 2
                current_idx += 3
            else:
                for feature_name in direction_features:
                    output_indices[f"{feature_name}_norm"] = current_idx
                    current_idx += 1

        ipp_features = self.feature_registry.get_feature_names(FeatureType.IPP)
        if ipp_features:
            for feature_name in ipp_features:
                output_indices[f"{feature_name}_norm"] = current_idx
                current_idx += 1

        if self.sh_enabled:
            if self.target == "vtec" or "Laplacian" in self.model_type:
                sh_dim = (self.sh_degree + 1) * (self.sh_degree + 1)
            else:
                sh_dim = self.sh_degree * self.sh_degree

            has_station_features = (
                len(self.feature_registry.get_feature_names(FeatureType.STATION)) > 0
            )
            has_ipp_features = len(ipp_features) > 0

            if has_station_features:
                output_indices["sh_sta_geo"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_sta_geo"] = None

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

            if has_ipp_features:
                output_indices["sh_ipp_sm"] = slice(current_idx, current_idx + sh_dim)
                current_idx += sh_dim
            else:
                output_indices["sh_ipp_sm"] = None

        swi_features = self.feature_registry.get_feature_names(FeatureType.SWI)
        for feature_name in swi_features:
            output_indices[f"{feature_name}_norm"] = current_idx
            current_idx += 1

        self.feature_registry.set_output_indices(output_indices)

        return current_idx

    def transform_temporal(self, features: torch.Tensor) -> torch.Tensor:
        temporal_features = self.feature_registry.get_feature_names(
            FeatureType.TEMPORAL
        )
        transformed_features = []

        for feature_name in temporal_features:
            feature_idx = self.input_indices[feature_name]
            feature_values = features[:, feature_idx]

            if feature_name == "year":
                year_norm = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                transformed_features.extend([year_norm])
            elif feature_name == "doy":
                doy_norm = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                doy_sin = torch.sin(doy_norm * 2 * torch.pi)
                doy_cos = torch.cos(doy_norm * 2 * torch.pi)
                transformed_features.extend([doy_sin, doy_cos, doy_norm])
            elif feature_name == "sod":
                norm_sod = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                sin_sod = torch.sin(norm_sod * 2 * torch.pi)
                cos_sod = torch.cos(norm_sod * 2 * torch.pi)
                transformed_features.extend([sin_sod, cos_sod, norm_sod])
            elif feature_name == "local_time_hours":
                norm_local_time = self.feature_registry.normalize_feature(
                    feature_name, feature_values
                ).unsqueeze(1)
                sin_local_time = torch.sin(norm_local_time * 2 * torch.pi)
                cos_local_time = torch.cos(norm_local_time * 2 * torch.pi)
                transformed_features.extend(
                    [sin_local_time, cos_local_time, norm_local_time]
                )
            else:
                raise ValueError(f"Unexpected temporal feature: {feature_name}")

        return torch.cat(transformed_features, dim=1)

    def transform_station(self, features: torch.Tensor) -> torch.Tensor | None:
        station_features = self.feature_registry.get_feature_names(FeatureType.STATION)

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

    def transform_direction(self, features: torch.Tensor) -> torch.Tensor | None:
        direction_features = self.feature_registry.get_feature_names(
            FeatureType.DIRECTION
        )

        if not direction_features:
            return None

        if "satazi" not in direction_features or "satele" not in direction_features:
            raise ValueError(
                "Both 'satazi' and 'satele' must be present for Cartesian transformation"
            )

        azi_idx = self.input_indices["satazi"]
        ele_idx = self.input_indices["satele"]

        azimuth_deg = features[:, azi_idx]
        elevation_deg = features[:, ele_idx]

        azimuth_rad = azimuth_deg * torch.pi / 180.0
        elevation_rad = elevation_deg * torch.pi / 180.0

        e_up = torch.sin(elevation_rad).unsqueeze(1)
        e_east = (torch.cos(elevation_rad) * torch.sin(azimuth_rad)).unsqueeze(1)
        e_north = (torch.cos(elevation_rad) * torch.cos(azimuth_rad)).unsqueeze(1)

        return torch.cat([e_up, e_east, e_north], dim=1)

    def transform_ipp(self, features: torch.Tensor) -> torch.Tensor | None:
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

    def transform_swi(self, features: torch.Tensor) -> torch.Tensor | None:
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

    def compute_sh_embeddings(
        self, features: torch.Tensor
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Spherical-harmonic embeddings for station and IPP coordinates."""
        if not self.sh_enabled:
            return None, None, None, None

        has_station_features = (
            "lon_sta" in self.input_indices and "lat_sta" in self.input_indices
        )
        has_sm_station_features = (
            "sm_lon_sta" in self.input_indices and "sm_lat_sta" in self.input_indices
        )
        has_ipp_features = (
            "lon_ipp" in self.input_indices and "lat_ipp" in self.input_indices
        )
        has_sm_ipp_features = (
            "sm_lon_ipp" in self.input_indices and "sm_lat_ipp" in self.input_indices
        )

        if has_station_features:
            sta_lon = features[:, self.input_indices["lon_sta"]]
            sta_lat = features[:, self.input_indices["lat_sta"]]
            sh_sta_geo = self.sh_encoder(torch.stack([sta_lon, sta_lat], dim=1))
        else:
            sh_sta_geo = None

        if has_ipp_features:
            ipp_lon = features[:, self.input_indices["lon_ipp"]]
            ipp_lat = features[:, self.input_indices["lat_ipp"]]
            sh_ipp_geo = self.sh_encoder(torch.stack([ipp_lon, ipp_lat], dim=1))
        else:
            sh_ipp_geo = None

        if has_sm_station_features:
            sm_lon_sta = features[:, self.input_indices["sm_lon_sta"]]
            sm_lat_sta = features[:, self.input_indices["sm_lat_sta"]]
            sh_sta_sm = self.sh_encoder(torch.stack([sm_lon_sta, sm_lat_sta], dim=1))
        else:
            sh_sta_sm = None

        if has_sm_ipp_features:
            sm_lon_ipp = features[:, self.input_indices["sm_lon_ipp"]]
            sm_lat_ipp = features[:, self.input_indices["sm_lat_ipp"]]
            sh_ipp_sm = self.sh_encoder(torch.stack([sm_lon_ipp, sm_lat_ipp], dim=1))
        else:
            sh_ipp_sm = None

        return sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm

    def __call__(self, batch):
        """Collate a batch of (features, labels) or (features, labels, metadata)."""
        if len(batch[0]) == 3:
            feats, labels, metadata_list = zip(*batch)
            has_metadata = True
        else:
            feats, labels = zip(*batch)
            has_metadata = False
            metadata_list = None

        features = torch.stack(feats, dim=0)
        labels = torch.stack(labels, dim=0)

        temporal_transformed = self.transform_temporal(features)
        station_transformed = self.transform_station(features)
        direction_transformed = self.transform_direction(features)
        ipp_transformed = self.transform_ipp(features)
        swi_transformed = self.transform_swi(features)

        sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm = self.compute_sh_embeddings(
            features
        )

        # Combine in the SAME order as _compute_and_store_output_indices.
        output_features = [temporal_transformed]

        if station_transformed is not None:
            output_features.append(station_transformed)
        if direction_transformed is not None:
            output_features.append(direction_transformed)
        if ipp_transformed is not None:
            output_features.append(ipp_transformed)

        if self.sh_enabled:
            if sh_sta_geo is not None:
                output_features.append(sh_sta_geo)
            if sh_ipp_geo is not None:
                output_features.append(sh_ipp_geo)
            if sh_sta_sm is not None:
                output_features.append(sh_sta_sm)
            if sh_ipp_sm is not None:
                output_features.append(sh_ipp_sm)

        if swi_transformed is not None:
            output_features.append(swi_transformed)

        final_features = torch.cat(output_features, dim=1)

        if has_metadata:
            return final_features, labels, metadata_list
        return final_features, labels
