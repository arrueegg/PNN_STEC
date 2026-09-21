"""Feature-tensor splitter for the VTEC x MF factorized model, ported from
`src/utils/feature_splitter.py`.

Splits the collated feature tensor (produced by `stec.data.collation.CollateWithSH`) into:
- x_vtec: features for VTEC field prediction (IPP location, time, SWI)
- x_geom: features for mapping-factor prediction (station, elevation, azimuth)
- elev_rad: elevation angle in radians (for the MF constraint)

Only real caller left: `scripts/show_feature_splits.py`, a diagnostic tool. The factorized
model itself (`FactorizedSTECModel`/`FactorizedSTECModelWrapper` in `src/model/model.py`)
was not ported into `stec/models` - see `stec/models/legacy_factory.py`'s module docstring
for which architectures `get_model` actually reaches.
"""

from __future__ import annotations

import torch

from .feature_registry import FeatureType


class FeatureSplitter:
    """Splits collated feature tensors into VTEC-related and geometry-related components.

    Uses the feature registry's output indices (set by `CollateWithSH`) to extract the
    correct slices from the transformed feature tensor.
    """

    def __init__(self, feature_registry) -> None:
        self.feature_registry = feature_registry

        if not hasattr(feature_registry, "_output_indices"):
            raise ValueError(
                "Feature registry does not have output_indices set. "
                "Make sure CollateWithSH has been initialized first, or call "
                "initialize_output_indices() before creating FeatureSplitter."
            )

        self.output_indices = feature_registry._output_indices

        self.vtec_indices = self._compute_vtec_indices()
        self.geom_indices = self._compute_geom_indices()
        self.elev_idx = self._get_elevation_index()

    def _compute_vtec_indices(self) -> list[int]:
        """VTEC features: temporal, IPP location, SWI, IPP spherical-harmonic embeddings.

        Excludes station features (station-specific, not part of the VTEC field) and
        direction features (geometry-dependent, used for MF instead).
        """
        indices: list[int] = []

        temporal_features = self.feature_registry.get_features_by_type(
            FeatureType.TEMPORAL
        )
        for feature_name in temporal_features:
            if feature_name == "year":
                indices.append(self.output_indices[f"{feature_name}_norm"])
            elif feature_name in ["doy", "sod", "local_time_hours"]:
                indices.extend(
                    [
                        self.output_indices[f"{feature_name}_sin"],
                        self.output_indices[f"{feature_name}_cos"],
                        self.output_indices[f"{feature_name}_norm"],
                    ]
                )

        ipp_features = self.feature_registry.get_features_by_type(FeatureType.IPP)
        for feature_name in ipp_features:
            indices.append(self.output_indices[f"{feature_name}_norm"])

        if (
            "sh_ipp_geo" in self.output_indices
            and self.output_indices["sh_ipp_geo"] is not None
        ):
            sh_slice = self.output_indices["sh_ipp_geo"]
            indices.extend(range(sh_slice.start, sh_slice.stop))

        if (
            "sh_ipp_sm" in self.output_indices
            and self.output_indices["sh_ipp_sm"] is not None
        ):
            sh_slice = self.output_indices["sh_ipp_sm"]
            indices.extend(range(sh_slice.start, sh_slice.stop))

        swi_features = self.feature_registry.get_features_by_type(FeatureType.SWI)
        for feature_name in swi_features:
            indices.append(self.output_indices[f"{feature_name}_norm"])

        return sorted(indices)

    def _compute_geom_indices(self) -> list[int]:
        """Geometry features: station location, direction (Cartesian unit vector), station
        spherical-harmonic embeddings. Elevation is also extracted separately in radians."""
        indices: list[int] = []

        station_features = self.feature_registry.get_features_by_type(
            FeatureType.STATION
        )
        for feature_name in station_features:
            if f"{feature_name}_norm" in self.output_indices:
                indices.append(self.output_indices[f"{feature_name}_norm"])

        if "e_up" in self.output_indices:
            indices.extend(
                [
                    self.output_indices["e_up"],
                    self.output_indices["e_east"],
                    self.output_indices["e_north"],
                ]
            )

        if (
            "sh_sta_geo" in self.output_indices
            and self.output_indices["sh_sta_geo"] is not None
        ):
            sh_slice = self.output_indices["sh_sta_geo"]
            indices.extend(range(sh_slice.start, sh_slice.stop))

        if (
            "sh_sta_sm" in self.output_indices
            and self.output_indices["sh_sta_sm"] is not None
        ):
            sh_slice = self.output_indices["sh_sta_sm"]
            indices.extend(range(sh_slice.start, sh_slice.stop))

        return sorted(indices)

    def _get_elevation_index(self) -> int:
        """Index of e_up (sin(elevation)) in the feature tensor."""
        if "e_up" in self.output_indices:
            return self.output_indices["e_up"]
        raise ValueError(
            "Elevation component (e_up) not found. "
            "Ensure direction features (satazi, satele) are enabled."
        )

    def split_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the collated feature tensor into (x_vtec, x_geom, elev_rad)."""
        if len(self.vtec_indices) > 0:
            x_vtec = features[:, self.vtec_indices]
        else:
            x_vtec = torch.empty((features.shape[0], 0), device=features.device)

        if len(self.geom_indices) > 0:
            x_geom = features[:, self.geom_indices]
        else:
            x_geom = torch.empty((features.shape[0], 0), device=features.device)

        # e_up = sin(elevation_rad), so arcsin recovers elevation.
        e_up = features[:, self.elev_idx]
        e_up_clamped = torch.clamp(e_up, -1.0, 1.0)
        elev_rad = torch.arcsin(e_up_clamped)

        return x_vtec, x_geom, elev_rad

    def get_vtec_dim(self) -> int:
        return len(self.vtec_indices)

    def get_geom_dim(self) -> int:
        return len(self.geom_indices)

    def get_total_dim(self) -> int:
        return (
            max(max(self.vtec_indices, default=-1), max(self.geom_indices, default=-1))
            + 1
        )
