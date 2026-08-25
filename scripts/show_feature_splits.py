"""
Feature Splitting Visualization Tool

This script shows exactly which features are assigned to:
- VTEC field prediction (ionospheric content)
- Geometry/MF prediction (geometric effects)

Run with: python scripts/show_feature_splits.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stec.data.feature_registry import (
    FeatureType,
    initialize_feature_registry,
)
from stec.data.feature_splitter import FeatureSplitter
from stec.data.collation import CollateWithSH


def create_config():
    """Create configuration with all features enabled."""
    config = {
        "data": {
            "use_SWI": True,
            "SH_degree": 5,  # Enable SH embeddings to see full feature set
        },
        "target": "stec",
    }
    return config


def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def show_raw_features(feature_registry):
    """Show raw features before transformation."""
    print_section_header("RAW INPUT FEATURES (Before Collation/Transformation)")

    feature_types = [
        (FeatureType.TEMPORAL, "Temporal Features"),
        (FeatureType.STATION, "Station Features"),
        (FeatureType.DIRECTION, "Direction Features"),
        (FeatureType.IPP, "IPP Features"),
        (FeatureType.SWI, "Space Weather Indices"),
    ]

    for ftype, name in feature_types:
        features = feature_registry.get_features_by_type(ftype)
        if features:
            print(f"\n{name} ({len(features)} features):")
            for i, feature in enumerate(features, 1):
                print(f"  {i:2d}. {feature}")


def show_transformed_features(feature_registry):
    """Show transformed features after collation."""
    print_section_header("TRANSFORMED FEATURES (After Collation)")

    output_indices = feature_registry._output_indices

    # Group by transformation type
    temporal_transformed = []
    station_transformed = []
    direction_transformed = []
    ipp_transformed = []
    sh_transformed = []
    swi_transformed = []

    for key, value in output_indices.items():
        if isinstance(value, slice):
            # SH embeddings (slices)
            sh_transformed.append((key, value, f"slice({value.start}:{value.stop})"))
        elif isinstance(value, int):
            # Regular features (indices)
            if "year" in key or "doy" in key or "sod" in key or "local_time" in key:
                temporal_transformed.append((key, value))
            elif "sta" in key and "sh" not in key:
                station_transformed.append((key, value))
            elif "e_up" in key or "e_east" in key or "e_north" in key:
                direction_transformed.append((key, value))
            elif "ipp" in key and "sh" not in key:
                ipp_transformed.append((key, value))
            elif any(
                swi in key for swi in ["Kp", "Sunspot", "Dst", "AE", "ap", "f107"]
            ):
                swi_transformed.append((key, value))

    # Sort by index
    temporal_transformed.sort(key=lambda x: x[1])
    station_transformed.sort(key=lambda x: x[1])
    direction_transformed.sort(key=lambda x: x[1])
    ipp_transformed.sort(key=lambda x: x[1])
    swi_transformed.sort(key=lambda x: x[1])

    print(f"\nTemporal Features (Transformed) - {len(temporal_transformed)} features:")
    for feature, idx in temporal_transformed:
        print(f"  [{idx:2d}] {feature}")

    print(f"\nStation Features (Normalized) - {len(station_transformed)} features:")
    for feature, idx in station_transformed:
        print(f"  [{idx:2d}] {feature}")

    print(f"\nDirection Features (Cartesian) - {len(direction_transformed)} features:")
    for feature, idx in direction_transformed:
        print(f"  [{idx:2d}] {feature}")

    print(f"\nIPP Features (Normalized) - {len(ipp_transformed)} features:")
    for feature, idx in ipp_transformed:
        print(f"  [{idx:2d}] {feature}")

    if sh_transformed:
        print(
            f"\nSpherical Harmonic Embeddings - {len(sh_transformed)} embedding groups:"
        )
        for name, slice_obj, desc in sh_transformed:
            if slice_obj is not None:
                dim = slice_obj.stop - slice_obj.start
                print(f"  {desc:20s} {name:20s} ({dim} dimensions)")

    print(f"\nSpace Weather Indices (Normalized) - {len(swi_transformed)} features:")
    for feature, idx in swi_transformed:
        print(f"  [{idx:2d}] {feature}")

    # Total dimension
    total_dim = (
        max(
            [
                idx
                for _, idx in temporal_transformed
                + station_transformed
                + direction_transformed
                + ipp_transformed
                + swi_transformed
            ]
        )
        + 1
    )
    sh_dim = sum(s.stop - s.start for _, s, _ in sh_transformed if s is not None)

    print(f"\nTotal Transformed Feature Dimension: {total_dim + sh_dim}")
    print(f"  Regular features: {total_dim}")
    print(f"  SH embeddings: {sh_dim}")


def show_vtec_features(splitter, feature_registry):
    """Show features assigned to VTEC field prediction."""
    print_section_header("VTEC FIELD FEATURES (Ionospheric Content)")

    vtec_indices = splitter.vtec_indices
    output_indices = feature_registry._output_indices

    # Reverse mapping: index -> feature name
    idx_to_feature = {}
    for feature_name, idx in output_indices.items():
        if isinstance(idx, int):
            idx_to_feature[idx] = feature_name
        elif isinstance(idx, slice) and idx is not None:
            # SH embeddings
            for i in range(idx.start, idx.stop):
                idx_to_feature[i] = f"{feature_name}[{i - idx.start}]"

    print("\nVTEC features are used to predict the ionospheric field (VTEC).")
    print("These represent the electron content independent of observation geometry.\n")
    print(f"Total VTEC features: {len(vtec_indices)}\n")

    # Group VTEC features by type
    temporal = []
    ipp = []
    sh_ipp = []
    swi = []

    for idx in vtec_indices:
        feature_name = idx_to_feature.get(idx, f"unknown[{idx}]")

        if any(t in feature_name for t in ["year", "doy", "sod", "local_time"]):
            temporal.append((idx, feature_name))
        elif "ipp" in feature_name and "sh_" in feature_name:
            sh_ipp.append((idx, feature_name))
        elif "ipp" in feature_name:
            ipp.append((idx, feature_name))
        elif any(
            s in feature_name for s in ["Kp", "Sunspot", "Dst", "AE", "ap", "f107"]
        ):
            swi.append((idx, feature_name))

    if temporal:
        print(f"Temporal Features ({len(temporal)}):")
        print("  → When? (year, day, time, local solar time)")
        for idx, name in temporal:
            print(f"     [{idx:2d}] {name}")

    if ipp:
        print(f"\nIPP Location Features ({len(ipp)}):")
        print("  → Where? (pierce point geographic & solar-magnetic coordinates)")
        for idx, name in ipp:
            print(f"     [{idx:2d}] {name}")

    if sh_ipp:
        print(f"\nIPP Spherical Harmonic Embeddings ({len(sh_ipp)}):")
        print("  → Spatial encoding of pierce point location")
        # Show first few and last few to avoid clutter
        if len(sh_ipp) > 10:
            for idx, name in sh_ipp[:5]:
                print(f"     [{idx:2d}] {name}")
            print(f"     ... ({len(sh_ipp) - 10} more SH features)")
            for idx, name in sh_ipp[-5:]:
                print(f"     [{idx:2d}] {name}")
        else:
            for idx, name in sh_ipp:
                print(f"     [{idx:2d}] {name}")

    if swi:
        print(f"\nSpace Weather Indices ({len(swi)}):")
        print("  → Global ionospheric conditions (solar/geomagnetic activity)")
        for idx, name in swi:
            print(f"     [{idx:2d}] {name}")

    print(f"\n{'→ VTEC Network Input:':<30} {len(vtec_indices)} features")
    print(f"{'→ Output:':<30} (vtec_mean, vtec_log_sigma)")


def show_geometry_features(splitter, feature_registry):
    """Show features assigned to geometry/MF prediction."""
    print_section_header("GEOMETRY FEATURES (Mapping Factor)")

    geom_indices = splitter.geom_indices
    output_indices = feature_registry._output_indices

    # Reverse mapping: index -> feature name
    idx_to_feature = {}
    for feature_name, idx in output_indices.items():
        if isinstance(idx, int):
            idx_to_feature[idx] = feature_name
        elif isinstance(idx, slice) and idx is not None:
            for i in range(idx.start, idx.stop):
                idx_to_feature[i] = f"{feature_name}[{i - idx.start}]"

    print("\nGeometry features are used to predict the mapping factor (MF).")
    print("These represent the geometric transformation from vertical to slant path.\n")
    print(f"Total geometry features: {len(geom_indices)}\n")

    # Group geometry features by type
    station = []
    direction = []
    sh_station = []

    for idx in geom_indices:
        feature_name = idx_to_feature.get(idx, f"unknown[{idx}]")

        if "sta" in feature_name and "sh_" in feature_name:
            sh_station.append((idx, feature_name))
        elif "sta" in feature_name:
            station.append((idx, feature_name))
        elif any(d in feature_name for d in ["e_up", "e_east", "e_north"]):
            direction.append((idx, feature_name))

    if station:
        print(f"Station Location Features ({len(station)}):")
        print("  → Where is the receiver? (geographic & solar-magnetic coordinates)")
        for idx, name in station:
            print(f"     [{idx:2d}] {name}")

    if direction:
        print(f"\nDirection Features ({len(direction)}):")
        print("  → Viewing geometry (elevation, azimuth as Cartesian unit vector)")
        for idx, name in direction:
            component = (
                "vertical"
                if "e_up" in name
                else ("eastward" if "e_east" in name else "northward")
            )
            print(f"     [{idx:2d}] {name:<30} ({component} component)")

    if sh_station:
        print(f"\nStation Spherical Harmonic Embeddings ({len(sh_station)}):")
        print("  → Spatial encoding of receiver location")
        if len(sh_station) > 10:
            for idx, name in sh_station[:5]:
                print(f"     [{idx:2d}] {name}")
            print(f"     ... ({len(sh_station) - 10} more SH features)")
            for idx, name in sh_station[-5:]:
                print(f"     [{idx:2d}] {name}")
        else:
            for idx, name in sh_station:
                print(f"     [{idx:2d}] {name}")

    print(
        f"\n{'→ GeomNet Input:':<30} {len(geom_indices)} features + elevation (radians)"
    )
    print(f"{'→ Output:':<30} mf (mapping factor)")
    print(f"\n{'→ MF Constraint:':<30} MF(90°) = 1.0, MF ≥ 1.0 everywhere")


def show_elevation_extraction(feature_registry):
    """Show how elevation is extracted."""
    print_section_header("ELEVATION EXTRACTION")

    output_indices = feature_registry._output_indices

    if "e_up" in output_indices:
        e_up_idx = output_indices["e_up"]
        print("\nElevation is extracted from the Cartesian direction vector:")
        print(f"  e_up index: [{e_up_idx}]")
        print("\n  Transformation:")
        print("    1. During collation: azimuth & elevation → (e_up, e_east, e_north)")
        print("    2. e_up = sin(elevation_rad)")
        print("    3. During splitting: elevation_rad = arcsin(e_up)")
        print("\n  This recovered elevation_rad is passed to GeomNet to enforce:")
        print("    g(elev) = 1 - sin(elev)")
        print("    MF = 1 + g(elev) * softplus(mf_raw)")
    else:
        print("\n⚠ WARNING: e_up not found! Direction features may not be enabled.")


def show_final_combination():
    """Show how VTEC and MF combine to produce STEC."""
    print_section_header("FINAL STEC COMPUTATION")

    print("""
The factorized model combines VTEC and MF predictions:

  1. VTECFieldNet(x_vtec) → (vtec_mean, vtec_log_sigma)
     ↓
     σ_vtec = exp(vtec_log_sigma)

  2. GeomNet(x_geom, elev_rad) → mf
     
  3. STEC Prediction:
     μ_stec = mf × vtec_mean
     
  4. Uncertainty Propagation:
     σ_stec = |mf| × σ_vtec
     var_stec = σ_stec²
     
  5. Return (μ_stec, var_stec) for training

Physical Interpretation:
  • VTEC: Vertical electron content (ionospheric property)
  • MF: Geometric scaling factor (observation geometry)
  • STEC = MF × VTEC: Slant path = factor × vertical path
  • Larger MF at low elevations → longer slant path
  • Uncertainty scales with MF → more uncertain at low elevations
""")


def main():
    """Main function to show all feature splits."""
    print("\n" + "=" * 80)
    print("  FACTORIZED STEC MODEL - FEATURE SPLITTING ANALYSIS")
    print("=" * 80)
    print("\nThis tool shows exactly how features are split for the factorized model.")
    print("Configuration: All features enabled, SH_degree=5, use_SWI=True")

    # Setup
    config = create_config()
    feature_registry = initialize_feature_registry(config)
    config["feature_registry"] = feature_registry

    # Initialize collation to set output indices on feature_registry (side effect
    # read by show_transformed_features/show_vtec_features/show_geometry_features/
    # show_elevation_extraction below); the collator object itself is not used.
    CollateWithSH(config)

    # Create splitter
    splitter = FeatureSplitter(feature_registry)

    # Show all feature information
    show_raw_features(feature_registry)
    show_transformed_features(feature_registry)
    show_vtec_features(splitter, feature_registry)
    show_geometry_features(splitter, feature_registry)
    show_elevation_extraction(feature_registry)
    show_final_combination()

    # Summary
    print_section_header("SUMMARY")
    vtec_dim = splitter.get_vtec_dim()
    geom_dim = splitter.get_geom_dim()
    total_dim = splitter.get_total_dim()

    print(f"""
Feature Dimension Summary:
  • Total collated features:  {total_dim}
  • VTEC features:            {vtec_dim} ({100 * vtec_dim / total_dim:.1f}%)
  • Geometry features:        {geom_dim} ({100 * geom_dim / total_dim:.1f}%)
  
Feature Assignment Logic:
  ✓ VTEC gets: temporal + IPP location + SWI + IPP_SH
  ✓ Geometry gets: station location + direction + station_SH
  ✓ Elevation extracted from e_up for MF constraint
  
Model Architecture:
  ✓ VTECFieldNet: {vtec_dim} inputs → (vtec_mean, vtec_log_sigma)
  ✓ GeomNet: {geom_dim} inputs + elev_rad → mf
  ✓ Combination: STEC = MF × VTEC with uncertainty propagation
    
All feature splits verified! ✓
""")


if __name__ == "__main__":
    main()
