"""Feature registry, ported unchanged from `src/utils/feature_registry.py`.

The rebuilt paper pipeline (`stec.data.feature_layout`, `stec.data.transforms`) replaced
this with a `FeatureLayout` dataclass computed once from `feature_control` - see that
module's docstring for why (Gate A: bit-exact against this class's own derivations across
every experiment config in the repository). This copy exists for the operational scripts
under `positioning/scripts/`, `scripts/` and `vlbi_kband/scripts/` that still build a
`CollateWithSH`-shaped feature tensor at inference time from an arbitrary experiment's
`config.yaml`: they need the *object*, not just the numbers it produces, because
`FeatureSplitter` (the factorized-model diagnostic) and a few of these scripts introspect
`registry.get_features_by_type(...)` / `registry._output_indices` directly rather than
reading a pre-computed layout. Ported as-is, including the mutable, order-dependent
registration API - a divergence-worthy rewrite here would risk changing feature ordering,
and ordering is exactly what determines which column of a loaded checkpoint's first layer
a given feature lands on.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

# =============================================================================
# FEATURE CONTROL - Toggle features on/off for experiments
# =============================================================================
# Set to True to enable a feature, False to disable it
# This serves as the master list of all available features
DEFAULT_FEATURE_CONTROL = {
    # Temporal features
    "year": True,
    "doy": True,
    "sod": True,
    "local_time_hours": True,
    # Station features (only for STEC)
    "sm_lat_sta": True,
    "sm_lon_sta": True,
    "lat_sta": True,
    "lon_sta": True,
    # IPP features
    "lat_ipp": True,
    "lon_ipp": True,
    "sm_lat_ipp": True,
    "sm_lon_ipp": True,
    # Direction features (only for STEC)
    "satazi": True,
    "satele": True,
    # Space Weather Indices (when use_SWI=True)
    "Kp_index": True,
    "R_Sunspot_No": True,
    "Dst-index,_nT": True,
    "AE-index,_nT": True,
    "ap_index,_nT": True,
    "f107_index": True,
    # Target (always enabled - cannot be disabled)
    "stec": True,
    "vtec": True,
}


class FeatureType(Enum):
    SWI = "swi"
    STATION = "station"
    IPP = "ipp"
    TEMPORAL = "temporal"
    DIRECTION = "direction"
    TARGET = "target"


class FeatureRegistry:
    """Central registry for all features with fixed ordering and metadata."""

    def __init__(self) -> None:
        self._features: dict = {}
        self._feature_order: list[str] = []
        self._type_groups: dict = {ft: [] for ft in FeatureType}
        self._indices: dict = {}

    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        position: Optional[int] = None,
        normalization: Optional[tuple[float, float]] = None,
        description: str = "",
    ) -> None:
        """Register a feature with its metadata."""
        if name in self._features:
            raise ValueError(f"Feature {name} already registered")

        feature_info = {
            "name": name,
            "type": feature_type,
            "normalization": normalization,
            "description": description,
            "enabled": True,
        }

        if position is None:
            position = len(self._feature_order)

        self._features[name] = feature_info
        self._feature_order.insert(position, name)
        self._type_groups[feature_type].append(name)
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild feature indices after changes."""
        # Only include enabled features in indices
        enabled_features = [
            name for name in self._feature_order if self._features[name]["enabled"]
        ]
        self._indices = {name: idx for idx, name in enumerate(enabled_features)}

    def set_output_indices(self, output_indices: dict) -> None:
        self._output_indices = output_indices

    def get_indices(self, feature_names: list[str]) -> list[int]:
        """Get indices for specified features."""
        indices = []
        for name in feature_names:
            if name in self._indices and self._features[name]["enabled"]:
                indices.append(self._indices[name])
        return indices

    def get_all_enabled_features(self) -> list[str]:
        """Get all enabled features in order."""
        return [name for name in self._feature_order if self._features[name]["enabled"]]

    def get_features_by_type(self, feature_type: FeatureType) -> list[str]:
        """Get all features of a specific type."""
        return [
            name
            for name in self._type_groups[feature_type]
            if self._features[name]["enabled"]
        ]

    def get_feature_names(self, feature_type: FeatureType) -> list[str]:
        """Get feature names for a specific type (alias for compatibility)."""
        return self.get_features_by_type(feature_type)

    def get_feature_slice(self, feature_type: FeatureType) -> slice:
        """Get slice for features of a specific type."""
        all_enabled = self.get_all_enabled_features()
        type_features = self.get_features_by_type(feature_type)

        if not type_features:
            return slice(0, 0)

        # Find start and end indices in the enabled features list
        start_idx = None
        end_idx = None

        for i, feature in enumerate(all_enabled):
            if feature in type_features:
                if start_idx is None:
                    start_idx = i
                end_idx = i

        if start_idx is None:
            return slice(0, 0)

        return slice(start_idx, end_idx + 1)

    def get_total_features(self) -> int:
        """Get total number of enabled features."""
        return len([f for f in self._features.values() if f["enabled"]])

    def enable_feature(self, name: str) -> None:
        """Enable a feature."""
        if name in self._features:
            self._features[name]["enabled"] = True
            self._rebuild_indices()

    def disable_feature(self, name: str) -> None:
        """Disable a feature."""
        if name in self._features:
            self._features[name]["enabled"] = False
            self._rebuild_indices()

    def validate_feature_data(self, feature_name: str, value: float) -> bool:
        """Validate if a feature value is within expected range."""
        if feature_name not in self._features:
            return False

        normalization = self._features[feature_name]["normalization"]
        if normalization is None:
            return True

        min_val, max_val = normalization
        return min_val <= value <= max_val

    def get_normalization_params(
        self, feature_name: str
    ) -> Optional[tuple[float, float]]:
        """Get normalization parameters for a feature."""
        if feature_name in self._features:
            return self._features[feature_name]["normalization"]
        return None

    def normalize_feature(self, feature_name: str, value):
        """Normalize a feature value to [0, 1] range."""
        normalization = self.get_normalization_params(feature_name)
        if normalization is None:
            return value

        min_val, max_val = normalization
        return (value - min_val) / (max_val - min_val)

    def denormalize_feature(self, feature_name: str, normalized_value):
        """Denormalize a feature value from [0, 1] range."""
        normalization = self.get_normalization_params(feature_name)
        if normalization is None:
            return normalized_value

        min_val, max_val = normalization
        return normalized_value * (max_val - min_val) + min_val

    def get_feature_type(self, feature_name: str) -> FeatureType:
        """Get the feature type for a given feature name."""
        if feature_name not in self._features:
            raise ValueError(f"Feature {feature_name} not found in registry")
        return self._features[feature_name]["type"]


def create_default_registry(config: dict) -> FeatureRegistry:
    """Create default feature registry based on config."""
    registry = FeatureRegistry()

    # Get feature control settings from config (or use defaults)
    feature_control = config.get("feature_control", DEFAULT_FEATURE_CONTROL.copy())

    # Register features in the exact order they appear in the data pipeline

    # 1. Temporal features (first in the data construction)
    if feature_control.get("year", True):
        registry.register_feature(
            "year",
            FeatureType.TEMPORAL,
            normalization=(2010, 2030),
            description="Year of observation",
        )
    if feature_control.get("doy", True):
        registry.register_feature(
            "doy",
            FeatureType.TEMPORAL,
            normalization=(1, 366),
            description="Day of year",
        )
    if feature_control.get("sod", True):
        registry.register_feature(
            "sod",
            FeatureType.TEMPORAL,
            normalization=(0, 86400),
            description="Seconds of day",
        )
    if feature_control.get("local_time_hours", True):
        registry.register_feature(
            "local_time_hours",
            FeatureType.TEMPORAL,
            normalization=(0, 24),
            description="Local time in hours (0-24) based on longitude",
        )

    # 2. Station features (solar magnetic coordinates) - only for STEC
    if config["target"] == "stec":
        if feature_control.get("sm_lat_sta", True):
            registry.register_feature(
                "sm_lat_sta",
                FeatureType.STATION,
                normalization=(-90, 90),
                description="Station solar magnetic latitude",
            )
        if feature_control.get("sm_lon_sta", True):
            registry.register_feature(
                "sm_lon_sta",
                FeatureType.STATION,
                normalization=(-180, 180),
                description="Station solar magnetic longitude",
            )

        if feature_control.get("lat_sta", True):
            registry.register_feature(
                "lat_sta",
                FeatureType.STATION,
                normalization=(-90, 90),
                description="Station geographic latitude",
            )
        if feature_control.get("lon_sta", True):
            registry.register_feature(
                "lon_sta",
                FeatureType.STATION,
                normalization=(-180, 180),
                description="Station geographic longitude",
            )

    # 3. IPP features (Ionospheric Pierce Point)
    if feature_control.get("lat_ipp", True):
        registry.register_feature(
            "lat_ipp",
            FeatureType.IPP,
            normalization=(-90, 90),
            description="IPP geographic latitude",
        )
    if feature_control.get("lon_ipp", True):
        registry.register_feature(
            "lon_ipp",
            FeatureType.IPP,
            normalization=(-180, 180),
            description="IPP geographic longitude",
        )

    if feature_control.get("sm_lat_ipp", True):
        registry.register_feature(
            "sm_lat_ipp",
            FeatureType.IPP,
            normalization=(-90, 90),
            description="IPP solar magnetic latitude",
        )
    if feature_control.get("sm_lon_ipp", True):
        registry.register_feature(
            "sm_lon_ipp",
            FeatureType.IPP,
            normalization=(-180, 180),
            description="IPP solar magnetic longitude",
        )

    # 4. Direction features (satellite direction - azimuth, elevation) - only for STEC
    if config["target"] == "stec":
        if feature_control.get("satazi", True):
            registry.register_feature(
                "satazi",
                FeatureType.DIRECTION,
                normalization=(0, 360),
                description="Satellite azimuth angle",
            )
        if feature_control.get("satele", True):
            registry.register_feature(
                "satele",
                FeatureType.DIRECTION,
                normalization=(0, 90),
                description="Satellite elevation angle",
            )

    # 5. SWI features (if enabled) - these come last in the data construction
    if config["data"].get("use_SWI", False):
        swi_features_with_normalization = [
            ("Kp_index", (0.0, 100.0)),
            ("R_Sunspot_No", (0.0, 300.0)),
            ("Dst-index,_nT", (-450, 100)),
            ("AE-index,_nT", (0.0, 2500.0)),
            ("ap_index,_nT", (0.0, 300.0)),
            ("f107_index", (62, 420)),
        ]

        for feature, normalization in swi_features_with_normalization:
            if feature_control.get(feature, True):
                registry.register_feature(
                    feature, FeatureType.SWI, normalization=normalization
                )

    # 6. Target feature (STEC or VTEC)
    if config["target"] == "stec":
        # Based on dataset statistics: min=0.001, max=~546, mean~26, std~27
        # Using 0 to 200 range to be safe with outliers
        registry.register_feature(
            "stec",
            FeatureType.TARGET,
            normalization=(1.0, 200.0),
            description="Slant Total Electron Content",
        )
    elif config["target"] == "vtec":
        registry.register_feature(
            "vtec",
            FeatureType.TARGET,
            normalization=(0.0, 200.0),
            description="Vertical Total Electron Content",
        )
    else:
        raise ValueError(f"Unknown target type: {config['target']}")

    return registry


def initialize_feature_registry(config: dict) -> FeatureRegistry:
    """Initialize and return the feature registry for the config."""
    registry = create_default_registry(config)

    # Store in config for global access
    config["feature_registry"] = registry

    return registry


def print_feature_summary(registry: FeatureRegistry, config: dict) -> None:
    """Print a summary of enabled/disabled features."""
    print("\n" + "=" * 80)
    print("FEATURE REGISTRY SUMMARY")
    print("=" * 80)

    for feature_type in FeatureType:
        features = registry.get_features_by_type(feature_type)
        if features:
            print(f"\n{feature_type.value.upper()} Features ({len(features)} enabled):")
            for feat in features:
                print(f"  {feat}")

    feature_control = config.get("feature_control", {})
    disabled = [name for name, enabled in feature_control.items() if not enabled]

    if disabled:
        print(f"\nDISABLED Features ({len(disabled)}):")
        for feat in disabled:
            print(f"  {feat}")

    print(f"\n{'=' * 80}")
    print(f"Total enabled features: {registry.get_total_features()}")
    print(f"{'=' * 80}\n")
