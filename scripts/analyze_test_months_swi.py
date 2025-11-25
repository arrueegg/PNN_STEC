#!/usr/bin/env python3
"""
Analyze SWI distributions for test months only.
Loads OMNI data and filters by test_dates.list to get realistic thresholds.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Load test dates
test_dates_file = Path(__file__).parent.parent / 'src' / 'data_processing' / 'test_dates.list'
with open(test_dates_file) as f:
    test_months = [line.strip() for line in f if line.strip()]

print(f"Test months: {test_months}")
print(f"Total: {len(test_months)} months\n")

# Parse test months into (year, month) tuples
test_periods = []
for period in test_months:
    year, month = period.split('-')
    test_periods.append((int(year), int(month)))

# Load OMNI data
omni_file = Path(__file__).parent.parent / 'data' / 'omni_hourly_2010-2025.h5'
print(f"Loading OMNI data from: {omni_file}")

# SWI column indices (from swi_loader.py)
swi_indices = {
    'f107': 19,
    'sunspot': 15,
    'kp': 14,
    'dst': 16,
}

data = {key: [] for key in swi_indices.keys()}

with h5py.File(omni_file, 'r') as f:
    for year, month in test_periods:
        year_str = str(year)
        if year_str not in f:
            print(f"  Warning: Year {year} not in OMNI file")
            continue
        
        year_group = f[year_str]
        
        # Get all days in this month
        # Convert month to day-of-year range
        from datetime import datetime, timedelta
        month_start = datetime(year, month, 1)
        if month == 12:
            month_end = datetime(year + 1, 1, 1)
        else:
            month_end = datetime(year, month + 1, 1)
        
        # Get day-of-year for each day in the month
        current_day = month_start
        days_in_month = 0
        hours_in_month = 0
        
        while current_day < month_end:
            doy = current_day.timetuple().tm_yday
            doy3 = f"{doy:03d}"
            
            if doy3 in year_group:
                daily_data = year_group[doy3][:]
                days_in_month += 1
                
                # Process each hour
                for hour_data in daily_data:
                    hours_in_month += 1
                    for key, idx in swi_indices.items():
                        if len(hour_data) > idx:
                            val = hour_data[idx]
                            # Skip invalid values (9999, etc.)
                            if val < 9000:
                                data[key].append(val)
            
            current_day += timedelta(days=1)
        
        print(f"  {year}-{month:02d}: {days_in_month} days, {hours_in_month} hours")

# Convert to numpy arrays
print(f"\nCollected data:")
for key in data:
    data[key] = np.array(data[key])
    print(f"  {key}: {len(data[key]):,} valid hourly samples")

# Analyze distributions
print("\n" + "="*80)
print("SWI DISTRIBUTIONS IN TEST MONTHS")
print("="*80)

percentiles = [5, 10, 25, 33, 50, 67, 75, 90, 95]
recommendations = {}

for name, values in data.items():
    print(f"\n{name.upper()}:")
    print(f"  Count: {len(values):,}")
    print(f"  Min: {values.min():.2f}, Max: {values.max():.2f}")
    print(f"  Mean: {values.mean():.2f}, Median: {np.median(values):.2f}")
    print(f"  Std: {values.std():.2f}")
    
    print(f"\n  Percentiles:")
    percs = {}
    for p in percentiles:
        val = np.percentile(values, p)
        percs[p] = val
        print(f"    {p:3d}%: {val:8.2f}")
    
    # Store recommendations
    if name == 'dst':
        recommendations[name] = {
            'low': percs[67],
            'high': percs[33],
            'storm': percs[10]
        }
    else:
        recommendations[name] = {
            'low': percs[33],
            'high': percs[67],
            'storm': percs[90] if name == 'kp' else None
        }

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, values) in enumerate(data.items()):
    ax = axes[idx]
    
    ax.hist(values, bins=100, alpha=0.7, edgecolor='black')
    
    p33 = np.percentile(values, 33)
    p50 = np.percentile(values, 50)
    p67 = np.percentile(values, 67)
    p90 = np.percentile(values, 90)
    
    ax.axvline(p33, color='blue', linestyle='--', linewidth=2, label=f'33% ({p33:.1f})')
    ax.axvline(p50, color='green', linestyle='-', linewidth=2, label=f'50% ({p50:.1f})')
    ax.axvline(p67, color='red', linestyle='--', linewidth=2, label=f'67% ({p67:.1f})')
    ax.axvline(p90, color='orange', linestyle=':', linewidth=2, label=f'90% ({p90:.1f})')
    
    ax.set_xlabel(name.upper())
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name.upper()} Distribution (Test Months Only)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_dir = Path(__file__).parent.parent / 'plots' / 'swi_analysis'
plot_dir.mkdir(exist_ok=True, parents=True)
plot_file = plot_dir / 'test_months_swi_distributions.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Saved plot to: {plot_file}")
plt.close()

# Test scenario balance
print("\n" + "="*80)
print("SCENARIO BALANCE TEST")
print("="*80)

# Get minimum length to align arrays
min_len = min(len(v) for v in data.values())
print(f"\nUsing {min_len:,} aligned hourly samples")

f107 = data['f107'][:min_len]
sunspot = data['sunspot'][:min_len]
kp = data['kp'][:min_len]
dst = data['dst'][:min_len]

is_low = (
    (f107 <= recommendations['f107']['low']) &
    (sunspot <= recommendations['sunspot']['low']) &
    (kp <= recommendations['kp']['low']) &
    (dst >= recommendations['dst']['low'])
)

is_high = (
    (f107 >= recommendations['f107']['high']) |
    (sunspot >= recommendations['sunspot']['high']) |
    (kp >= recommendations['kp']['high']) |
    (dst <= recommendations['dst']['high'])
)

is_storm = (
    (kp >= recommendations['kp']['storm']) |
    (dst <= recommendations['dst']['storm'])
)

total = min_len
print(f"\nScenario coverage:")
print(f"  Low activity: {is_low.sum():,} samples ({is_low.sum()/total*100:.1f}%)")
print(f"  High activity: {is_high.sum():,} samples ({is_high.sum()/total*100:.1f}%)")
print(f"  Storm days: {is_storm.sum():,} samples ({is_storm.sum()/total*100:.1f}%)")
print(f"  Low + storm: {(is_low & is_storm).sum():,} samples ({(is_low & is_storm).sum()/total*100:.1f}%)")
print(f"  High + storm: {(is_high & is_storm).sum():,} samples ({(is_high & is_storm).sum()/total*100:.1f}%)")

# Print recommendations
print("\n" + "="*80)
print("RECOMMENDED THRESHOLDS FOR scenario_evaluation.py")
print("="*80)

print(f"""
THRESHOLDS = {{
    'low_activity': {{
        'f107': ('<=', {recommendations['f107']['low']:.1f}),
        'sunspot': ('<=', {recommendations['sunspot']['low']:.1f}),
        'kp': ('<=', {recommendations['kp']['low']:.1f}),
        'dst': ('>=', {recommendations['dst']['low']:.1f}),
    }},
    'high_activity': {{
        'f107': ('>=', {recommendations['f107']['high']:.1f}),
        'sunspot': ('>=', {recommendations['sunspot']['high']:.1f}),
        'kp': ('>=', {recommendations['kp']['high']:.1f}),
        'dst': ('<=', {recommendations['dst']['high']:.1f}),
    }},
    'storm': {{
        'kp': ('>=', {recommendations['kp']['storm']:.1f}),
        'dst': ('<=', {recommendations['dst']['storm']:.1f}),
    }}
}}
""")

print("\nThese thresholds are based on your actual test months:")
print(f"  {len(test_months)} months from {test_months[0]} to {test_months[-1]}")
print(f"  Aiming for ~33% low, ~67% high, ~10% storm coverage")
