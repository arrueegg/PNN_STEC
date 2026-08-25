#!/usr/bin/env python3
"""
Analyze distribution of space weather indices from OMNI data
to determine balanced scenario thresholds.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_all_swi_data(swi_file_path):
    """Load all space weather indices from the OMNI HDF5 file."""
    print(f"Loading OMNI data from: {swi_file_path}")

    # Column indices based on swi_loader.py
    swi_indices = {
        "f107": 19,
        "sunspot": 15,
        "kp": 14,
        "dst": 16,
    }

    data = {key: [] for key in swi_indices.keys()}

    with h5py.File(swi_file_path, "r") as f:
        years = sorted([k for k in f.keys() if k.isdigit()])
        print(f"Processing years: {years[0]} - {years[-1]}")

        total_hours = 0
        for year in years:
            year_group = f[year]
            for doy in sorted(year_group.keys()):
                daily_data = year_group[doy][:]

                # Process each hour in the day
                for hour_data in daily_data:
                    for key, idx in swi_indices.items():
                        if len(hour_data) > idx:
                            val = hour_data[idx]
                            # Skip invalid values (9999, etc.)
                            if val < 9000:
                                data[key].append(val)
                    total_hours += 1

        print(f"Loaded {total_hours:,} hourly samples")

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
        print(f"  {key}: {len(data[key]):,} valid samples")

    return data


def analyze_and_plot(data, output_dir):
    """Analyze distributions and create plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 80)

    percentiles = [5, 10, 25, 33, 50, 67, 75, 90, 95]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    recommendations = {}

    for idx, (key, values) in enumerate(data.items()):
        ax = axes[idx]

        # Statistics
        print(f"\n{key.upper()}:")
        print(f"  Count: {len(values):,}")
        print(f"  Min: {values.min():.2f}")
        print(f"  Max: {values.max():.2f}")
        print(f"  Mean: {values.mean():.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std: {values.std():.2f}")

        print("\n  Percentiles:")
        percs = {}
        for p in percentiles:
            val = np.percentile(values, p)
            percs[p] = val
            print(f"    {p:3d}%: {val:8.2f}")

        # Plot histogram
        ax.hist(values, bins=100, alpha=0.7, edgecolor="black")
        ax.axvline(
            percs[10],
            color="purple",
            linestyle=":",
            linewidth=2,
            label=f"10% ({percs[10]:.1f})",
        )
        ax.axvline(
            percs[33],
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f"33% ({percs[33]:.1f})",
        )
        ax.axvline(
            percs[50],
            color="green",
            linestyle="-",
            linewidth=2,
            label=f"50% ({percs[50]:.1f})",
        )
        ax.axvline(
            percs[67],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"67% ({percs[67]:.1f})",
        )
        ax.axvline(
            percs[90],
            color="orange",
            linestyle=":",
            linewidth=2,
            label=f"90% ({percs[90]:.1f})",
        )

        ax.set_xlabel(key.upper())
        ax.set_ylabel("Frequency")
        ax.set_title(f"{key.upper()} Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Store recommendations
        if key == "dst":
            # For DST, less negative = quiet, more negative = active
            recommendations[key] = {
                "low": percs[67],  # 67th percentile (less negative = quiet)
                "high": percs[33],  # 33rd percentile (more negative = active)
                "storm": percs[10],  # 10th percentile (very negative = storm)
            }
        else:
            recommendations[key] = {
                "low": percs[33],  # 33rd percentile
                "high": percs[67],  # 67th percentile
                "storm": percs[90] if key in ["kp"] else None,
            }

    plt.tight_layout()
    plot_file = output_dir / "swi_distributions.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n✅ Saved plot to: {plot_file}")
    plt.close()

    return recommendations


def test_scenarios(data, recommendations):
    """Test scenario balance with recommended thresholds."""
    print("\n" + "=" * 80)
    print("SCENARIO BALANCE TEST")
    print("=" * 80)

    # Get minimum length to align all arrays
    min_len = min(len(v) for v in data.values())
    print(f"\nUsing {min_len:,} aligned samples for testing")

    # Align arrays
    f107 = data["f107"][:min_len]
    sunspot = data["sunspot"][:min_len]
    kp = data["kp"][:min_len]
    dst = data["dst"][:min_len]

    # Apply thresholds
    is_low = (
        (f107 <= recommendations["f107"]["low"])
        & (sunspot <= recommendations["sunspot"]["low"])
        & (kp <= recommendations["kp"]["low"])
        & (dst >= recommendations["dst"]["low"])
    )

    is_high = (
        (f107 >= recommendations["f107"]["high"])
        | (sunspot >= recommendations["sunspot"]["high"])
        | (kp >= recommendations["kp"]["high"])
        | (dst <= recommendations["dst"]["high"])
    )

    is_storm = (kp >= recommendations["kp"]["storm"]) | (
        dst <= recommendations["dst"]["storm"]
    )

    total = min_len
    print("\nScenario coverage:")
    print(
        f"  Low activity: {is_low.sum():,} samples ({is_low.sum() / total * 100:.1f}%)"
    )
    print(
        f"  High activity: {is_high.sum():,} samples ({is_high.sum() / total * 100:.1f}%)"
    )
    print(
        f"  Storm days: {is_storm.sum():,} samples ({is_storm.sum() / total * 100:.1f}%)"
    )
    print(
        f"  Low + storm: {(is_low & is_storm).sum():,} samples ({(is_low & is_storm).sum() / total * 100:.1f}%)"
    )
    print(
        f"  High + storm: {(is_high & is_storm).sum():,} samples ({(is_high & is_storm).sum() / total * 100:.1f}%)"
    )


def print_recommendations(recommendations):
    """Print final threshold recommendations."""
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS FOR scenario_evaluation.py")
    print("=" * 80)

    print(f"""
LOW_ACTIVITY_THRESHOLD = {{
    'f107': {recommendations["f107"]["low"]:.1f},
    'sunspot': {recommendations["sunspot"]["low"]:.1f},
    'kp': {recommendations["kp"]["low"]:.1f},
    'dst': {recommendations["dst"]["low"]:.1f}
}}

HIGH_ACTIVITY_THRESHOLD = {{
    'f107': {recommendations["f107"]["high"]:.1f},
    'sunspot': {recommendations["sunspot"]["high"]:.1f},
    'kp': {recommendations["kp"]["high"]:.1f},
    'dst': {recommendations["dst"]["high"]:.1f}
}}

STORM_THRESHOLD = {{
    'kp': {recommendations["kp"]["storm"]:.1f},
    'dst': {recommendations["dst"]["storm"]:.1f}
}}
""")

    print("\nThreshold interpretation:")
    print("  - Low activity: ALL conditions must be met (quiet conditions)")
    print("  - High activity: ANY condition triggers (active conditions)")
    print("  - Storm: ANY condition triggers (extreme conditions)")
    print("\nThese thresholds aim for ~33% low, ~67% high, ~10% storm coverage")


if __name__ == "__main__":
    swi_file = Path(__file__).parent.parent / "data" / "omni_hourly_2010-2025.h5"
    output_dir = Path(__file__).parent.parent / "plots" / "swi_analysis"

    # Load data
    data = load_all_swi_data(str(swi_file))

    # Analyze and plot
    recommendations = analyze_and_plot(data, output_dir)

    # Test scenarios
    test_scenarios(data, recommendations)

    # Print recommendations
    print_recommendations(recommendations)
