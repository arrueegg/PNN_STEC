from enum import Enum
from typing import Dict, List, Tuple, Optional
import torch

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
        
    def register_feature(self, name: str, feature_type: FeatureType, 
                        position: Optional[int] = None, 
                        normalization: Optional[Tuple[float, float]] = None,
                        description: str = ""):
        """Register a feature with its metadata."""
        if name in self._features:
            raise ValueError(f"Feature {name} already registered")
            
        feature_info = {
            'name': name,
            'type': feature_type,
            'normalization': normalization,
            'description': description,
            'enabled': True
        }
        
        if position is None:
            position = len(self._feature_order)
        
        self._features[name] = feature_info
        self._feature_order.insert(position, name)
        self._type_groups[feature_type].append(name)
        self._rebuild_indices()
        
    def _rebuild_indices(self):
        """Rebuild feature indices after changes."""
        self._indices = {name: idx for idx, name in enumerate(self._feature_order)}
        
    def get_indices(self, feature_names: List[str]) -> List[int]:
        """Get indices for specified features."""
        return [self._indices[name] for name in feature_names if name in self._indices]
    
    def get_features_by_type(self, feature_type: FeatureType) -> List[str]:
        """Get all features of a specific type."""
        return [name for name in self._type_groups[feature_type] 
                if self._features[name]['enabled']]
    
    def get_feature_names(self, feature_type: FeatureType) -> List[str]:
        """Get feature names for a specific type (alias for compatibility)."""
        return self.get_features_by_type(feature_type)
    
    def get_feature_slice(self, feature_type: FeatureType) -> slice:
        """Get slice for features of a specific type."""
        features = self.get_features_by_type(feature_type)
        if not features:
            return slice(0, 0)
        indices = self.get_indices(features)
        return slice(min(indices), max(indices) + 1)
    
    def get_total_features(self) -> int:
        """Get total number of enabled features."""
        return len([f for f in self._features.values() if f['enabled']])
    
    def enable_feature(self, name: str):
        """Enable a feature."""
        if name in self._features:
            self._features[name]['enabled'] = True
            
    def disable_feature(self, name: str):
        """Disable a feature."""
        if name in self._features:
            self._features[name]['enabled'] = False

# Create global feature registry
def create_default_registry(config: dict) -> FeatureRegistry:
    """Create default feature registry based on config."""
    registry = FeatureRegistry()
    
    # Register features in the exact order they appear in your data pipeline
    
    # 1. Temporal features (first in your data construction)
    registry.register_feature('year', FeatureType.TEMPORAL, 
                            description="Year of observation")
    registry.register_feature('doy', FeatureType.TEMPORAL, 
                            description="Day of year")
    registry.register_feature('sod', FeatureType.TEMPORAL, 
                            description="Seconds of day")
    
    # 2. Station features (solar magnetic coordinates)
    registry.register_feature('sm_lat_sta', FeatureType.STATION, 
                            normalization=(-90, 90),
                            description="Station solar magnetic latitude")
    registry.register_feature('sm_lon_sta', FeatureType.STATION, 
                            normalization=(-180, 180),
                            description="Station solar magnetic longitude")
    
    # 3. Shared features (satellite direction - azimuth, elevation)
    registry.register_feature('satazi', FeatureType.SHARED, 
                            normalization=(0, 360),
                            description="Satellite azimuth angle")
    registry.register_feature('satele', FeatureType.SHARED, 
                            normalization=(0, 90),
                            description="Satellite elevation angle")
    
    # 4. IPP features (Ionospheric Pierce Point)
    registry.register_feature('lat_ipp', FeatureType.IPP, 
                            normalization=(-90, 90),
                            description="IPP geographic latitude")
    registry.register_feature('lon_ipp', FeatureType.IPP, 
                            normalization=(-180, 180),
                            description="IPP geographic longitude")
    
    # Optional: Add solar magnetic IPP coordinates if they exist in your data
    # Check if these fields exist in your DTYPE
    registry.register_feature('sm_lat_ipp', FeatureType.IPP, 
                            normalization=(-90, 90),
                            description="IPP solar magnetic latitude")
    registry.register_feature('sm_lon_ipp', FeatureType.IPP, 
                            normalization=(-180, 180),
                            description="IPP solar magnetic longitude")
    
    # Optional: Add geographic station coordinates if needed
    registry.register_feature('lat_sta', FeatureType.STATION, 
                            normalization=(-90, 90),
                            description="Station geographic latitude")
    registry.register_feature('lon_sta', FeatureType.STATION, 
                            normalization=(-180, 180),
                            description="Station geographic longitude")
    
    # 5. SWI features (if enabled) - these come last in your data construction
    if config['data'].get('use_SWI', False):
        swi_features = [
            'Bartels_rotation_number', 'Scalar_B,_nT', 'Vector_B_Magnitude,nT',
            'Lat_Angle_of_B_GSE', 'Long_Angle_of_B_GSE', 'BZ,_nT_GSE', 'BZ,_nT_GSM',
            'SW_Plasma_Speed,_km/s', 'Flow_pressure', 'E_elecrtric_field', 
            'Alfen_mach_number', 'Kp_index', 'R_Sunspot_No', 'Dst-index,_nT', 
            'AE-index,_nT', 'ap_index,_nT', 'f107_index', 'pc-index', 
            'AL-index,_nT', 'AU-index,_nT', 'Magnetosonic_Much_num', 'Lyman_alpha'
        ]
        for feature in swi_features:
            registry.register_feature(feature, FeatureType.SWI)
    
    # 6. Target feature (STEC or VTEC)
    if config['target'] == 'stec':
        registry.register_feature('stec', FeatureType.TARGET, 
                                description="Slant Total Electron Content")
    elif config['target'] == 'vtec':
        registry.register_feature('vtec', FeatureType.TARGET, 
                                description="Vertical Total Electron Content")
    else:
        raise ValueError(f"Unknown target type: {config['target']}")
    
    return registry

def initialize_feature_registry(config: dict) -> FeatureRegistry:
    """Initialize and return the feature registry for the config."""
    registry = create_default_registry(config)
    
    # Store in config for global access
    config['feature_registry'] = registry
    
    return registry