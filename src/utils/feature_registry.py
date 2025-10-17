from enum import Enum
from typing import Dict, List, Tuple, Optional


class FeatureType(Enum):
    SWI = "swi"
    STATION = "station"
    IPP = "ipp"
    TEMPORAL = "temporal"
    DIRECTION = "direction"
    TARGET = "target"


class FeatureRegistry:
    """Central registry for all features with fixed ordering and metadata."""

    def __init__(self):
        self._features = {}
        self._feature_order = []
        self._type_groups = {ft: [] for ft in FeatureType}
        self._indices = {}

    def register_feature(
        self,
        name: str,
        feature_type: FeatureType,
        position: Optional[int] = None,
        normalization: Optional[Tuple[float, float]] = None,
        description: str = "",
    ):
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

    def _rebuild_indices(self):
        """Rebuild feature indices after changes."""
        # Only include enabled features in indices
        enabled_features = [
            name for name in self._feature_order if self._features[name]["enabled"]
        ]
        self._indices = {name: idx for idx, name in enumerate(enabled_features)}

    def set_output_indices(self, output_indices: Dict[str, int]):
        self._output_indices = output_indices

    def get_indices(self, feature_names: List[str]) -> List[int]:
        """Get indices for specified features."""
        indices = []
        for name in feature_names:
            if name in self._indices and self._features[name]["enabled"]:
                indices.append(self._indices[name])
        return indices

    def get_all_enabled_features(self) -> List[str]:
        """Get all enabled features in order."""
        return [name for name in self._feature_order if self._features[name]["enabled"]]

    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """Get all features of a specific type."""
        return [
            name
            for name in self._type_groups[feature_type]
            if self._features[name]["enabled"]
        ]

    def get_feature_names(self, feature_type: FeatureType) -> List[str]:
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

    def enable_feature(self, name: str):
        """Enable a feature."""
        if name in self._features:
            self._features[name]["enabled"] = True
            self._rebuild_indices()  # Add this line

    def disable_feature(self, name: str):
        """Disable a feature."""
        if name in self._features:
            self._features[name]["enabled"] = False
            self._rebuild_indices()  # Add this line

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
    ) -> Optional[Tuple[float, float]]:
        """Get normalization parameters for a feature."""
        if feature_name in self._features:
            return self._features[feature_name]["normalization"]
        return None

    def normalize_feature(self, feature_name: str, value: float) -> float:
        """Normalize a feature value to [0, 1] range."""
        normalization = self.get_normalization_params(feature_name)
        if normalization is None:
            return value

        min_val, max_val = normalization
        return (value - min_val) / (max_val - min_val)

    def denormalize_feature(self, feature_name: str, normalized_value: float) -> float:
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


# Create global feature registry
def create_default_registry(config: dict) -> FeatureRegistry:
    """Create default feature registry based on config."""
    registry = FeatureRegistry()

    # Register features in the exact order they appear in your data pipeline

    # 1. Temporal features (first in your data construction)
    registry.register_feature(
        "year",
        FeatureType.TEMPORAL,
        normalization=(2010, 2030),
        description="Year of observation",
    )
    registry.register_feature(
        "doy", FeatureType.TEMPORAL, normalization=(1, 366), description="Day of year"
    )
    registry.register_feature(
        "sod",
        FeatureType.TEMPORAL,
        normalization=(0, 86400),
        description="Seconds of day",
    )
    registry.register_feature(
        "local_time_hours",
        FeatureType.TEMPORAL,
        normalization=(0, 24),
        description="Local time in hours (0-24) based on longitude",
    )

    # 2. Station features (solar magnetic coordinates) - only for STEC
    if config["target"] == "stec":
        registry.register_feature(
            "sm_lat_sta",
            FeatureType.STATION,
            normalization=(-90, 90),
            description="Station solar magnetic latitude",
        )
        registry.register_feature(
            "sm_lon_sta",
            FeatureType.STATION,
            normalization=(-180, 180),
            description="Station solar magnetic longitude",
        )

        registry.register_feature(
            "lat_sta",
            FeatureType.STATION,
            normalization=(-90, 90),
            description="Station geographic latitude",
        )
        registry.register_feature(
            "lon_sta",
            FeatureType.STATION,
            normalization=(-180, 180),
            description="Station geographic longitude",
        )

    # 3. IPP features (Ionospheric Pierce Point)
    registry.register_feature(
        "lat_ipp",
        FeatureType.IPP,
        normalization=(-90, 90),
        description="IPP geographic latitude",
    )
    registry.register_feature(
        "lon_ipp",
        FeatureType.IPP,
        normalization=(-180, 180),
        description="IPP geographic longitude",
    )

    registry.register_feature(
        "sm_lat_ipp",
        FeatureType.IPP,
        normalization=(-90, 90),
        description="IPP solar magnetic latitude",
    )
    registry.register_feature(
        "sm_lon_ipp",
        FeatureType.IPP,
        normalization=(-180, 180),
        description="IPP solar magnetic longitude",
    )

    # 4. Direction features (satellite direction - azimuth, elevation) - only for STEC
    if config["target"] == "stec":
        registry.register_feature(
            "satazi",
            FeatureType.DIRECTION,
            normalization=(0, 360),
            description="Satellite azimuth angle",
        )
        registry.register_feature(
            "satele",
            FeatureType.DIRECTION,
            normalization=(0, 90),
            description="Satellite elevation angle",
        )

    # 5. SWI features (if enabled) - these come last in your data construction
    if config["data"].get("use_SWI", False):
        swi_features_with_normalization = [
            # Bartels rotation number (solar cycle indicator)
            # ('Bartels_rotation_number', (2407, 3000)),
            # Scalar magnetic field strength in nT
            # ('Scalar_B,_nT', (0, 70)),
            # Vector magnetic field magnitude in nT
            # ('Vector_B_Magnitude,nT', (0.0, 70)),
            # Latitude angle of magnetic field in GSE coordinates
            # ('Lat_Angle_of_B_GSE', (-90, 90)),
            # Longitude angle of magnetic field in GSE coordinates
            # ('Long_Angle_of_B_GSE', (0.0, 360.0)),
            # BZ component of magnetic field in GSE coordinates
            # ('BZ,_nT_GSE', (-50, 35)),
            # BZ component of magnetic field in GSM coordinates
            # ('BZ,_nT_GSM', (-50, 35)),
            # Solar wind plasma speed in km/s
            # ('SW_Plasma_Speed,_km/s', (240.0, 1100.0)),
            # Solar wind flow pressure
            # ('Flow_pressure', (0, 60)),
            # Electric field in mV/m
            # ('E_electric_field', (-20, 30)),
            # Alfvén Mach number
            # ('Alfen_mach_number', (0, 120)),
            # Planetary Kp index (geomagnetic activity)
            ("Kp_index", (0.0, 100.0)),
            # Relative sunspot number
            ("R_Sunspot_No", (0.0, 300.0)),
            # Disturbance storm time index in nT
            ("Dst-index,_nT", (-450, 100)),
            # Auroral electrojet index in nT
            ("AE-index,_nT", (0.0, 2500.0)),
            # Planetary ap index in nT
            ("ap_index,_nT", (0.0, 300.0)),
            # Solar radio flux at 10.7 cm
            ("f107_index", (62, 420)),
            # Polar cap index
            # ('pc-index', (-6, 16)),
            # AL index (auroral lower) in nT
            # ('AL-index,_nT', (-2000.0, 20.0)),
            # AU index (auroral upper) in nT
            # ('AU-index,_nT', (-200.0, 1200.0)),
            # Magnetosonic Mach number
            # ('Magnetosonic_Much_num', (0, 15)),
            # Lyman-alpha solar radiation
            # ('Lyman_alpha', (0, 0.015)),
        ]

        for feature, normalization in swi_features_with_normalization:
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
            normalization=(1.0, 200.0),  # More reasonable range
            description="Slant Total Electron Content",
        )
    elif config["target"] == "vtec":
        # Similar range expected for VTEC (needs verification if using VTEC)
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
