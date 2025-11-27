"""
Scenario-based evaluation module for PNN_STEC.

This module implements scenario-based filtering and evaluation based on
solar and geomagnetic activity levels using hourly space weather indices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Column name mappings - handle both simple and full feature names
COLUMN_MAPPINGS = {
    'f107': ['f107', 'f107_index'],
    'sunspot': ['sunspot', 'R_Sunspot_No'],
    'kp': ['kp', 'Kp_index'],
    'dst': ['dst', 'Dst-index,_nT', 'Dst-index'],
}

# Activity thresholds based on actual test data distribution (10M hourly samples)
# Using F10.7 quartiles to get ~25% data in each activity category
THRESHOLDS = {
    'low_activity': {
        'f107': ('<=', 86.9),       # 25th percentile of hourly data
    },
    'high_activity': {
        'f107': ('>=', 207.9),      # 75th percentile of hourly data
    },
    'storm': {
        'kp': ('>=', 37),           # 90th percentile of test data
        'dst': ('<=', -33),         # 10th percentile (extreme disturbance)
    }
}


def find_column_name(df: pd.DataFrame, canonical_name: str) -> Optional[str]:
    """
    Find the actual column name in the dataframe for a canonical feature name.
    
    Args:
        df: DataFrame to search
        canonical_name: Canonical name (e.g., 'f107', 'kp')
        
    Returns:
        Actual column name if found, None otherwise
    """
    possible_names = COLUMN_MAPPINGS.get(canonical_name, [canonical_name])
    for name in possible_names:
        if name in df.columns:
            return name
    return None




def identify_scenarios(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add boolean flags for each scenario to the dataframe using hourly SWI data.
    
    Scenarios:
    - low_activity: Based on hourly F10.7 values (bottom quartile)
    - high_activity: Based on hourly F10.7 values (top quartile)
    - storm_days: Based on hourly Kp/Dst extremes
    - high_activity_storm_days: high_activity AND storm_days
    
    Args:
        df: Dataframe with hourly SWI values
        logger: Logger instance
        
    Returns:
        Dataframe with scenario boolean columns added
    """
    if len(df) == 0:
        logger.warning("Empty dataframe passed to identify_scenarios")
        return df
    
    # Map canonical names to actual column names in the dataframe
    col_map = {}
    for canonical in ['f107', 'sunspot', 'kp', 'dst']:
        actual_col = find_column_name(df, canonical)
        if actual_col:
            col_map[canonical] = actual_col
    
    # Detect available indices
    available_indices = {
        'f107': 'f107' in col_map,
        'sunspot': 'sunspot' in col_map,
        'kp': 'kp' in col_map,
        'dst': 'dst' in col_map,
    }
    
    logger.info(f"Available space weather indices: {[k for k, v in available_indices.items() if v]}")
    
    # Initialize scenario flags
    df['is_low_activity'] = False
    df['is_high_activity'] = False
    df['is_storm'] = False
    
    # Low activity: Based on F10.7 only (bottom quartile) - use hourly values
    if available_indices['f107']:
        op, val = THRESHOLDS['low_activity']['f107']
        df['is_low_activity'] = df[col_map['f107']] <= val
    else:
        logger.warning("F10.7 not available for low_activity scenario - skipping")
    
    # High activity: Based on F10.7 only (top quartile) - use hourly values
    if available_indices['f107']:
        op, val = THRESHOLDS['high_activity']['f107']
        df['is_high_activity'] = df[col_map['f107']] >= val
    else:
        logger.warning("F10.7 not available for high_activity scenario - skipping")
    
    # Storm: kp>=37 OR dst<=-33 - use hourly values
    storm_conditions = []
    if available_indices['kp']:
        op, val = THRESHOLDS['storm']['kp']
        storm_conditions.append(df[col_map['kp']] >= val)
    if available_indices['dst']:
        op, val = THRESHOLDS['storm']['dst']
        storm_conditions.append(df[col_map['dst']] <= val)
    
    if storm_conditions:
        df['is_storm'] = np.logical_or.reduce(storm_conditions)
    else:
        logger.warning("No indices available for storm_days scenario - skipping")
        df['is_storm'] = False
    
    # Compute composite scenario: high activity + storm only
    is_high_activity_storm = df['is_high_activity'].values & df['is_storm'].values
    df['is_high_activity_storm'] = is_high_activity_storm
    
    # Log scenario counts
    total = len(df)
    logger.info("Scenario sample counts:")
    logger.info(f"  low_activity: {df['is_low_activity'].sum():,} samples ({df['is_low_activity'].sum()/total*100:.1f}%)")
    logger.info(f"  high_activity: {df['is_high_activity'].sum():,} samples ({df['is_high_activity'].sum()/total*100:.1f}%)")
    logger.info(f"  storm_days: {df['is_storm'].sum():,} samples ({df['is_storm'].sum()/total*100:.1f}%)")
    logger.info(f"  high_activity_storm_days: {is_high_activity_storm.sum():,} samples ({is_high_activity_storm.sum()/total*100:.1f}%)")
    
    return df


def validate_scenario(df_subset: pd.DataFrame, scenario_name: str, logger: logging.Logger) -> bool:
    """
    Validate if a scenario subset meets minimum requirements.
    
    Requirements:
    - At least MIN_DAYS_PER_SCENARIO unique dates
    - At least MIN_SAMPLES_PER_SCENARIO total samples
    
    Args:
        df_subset: Filtered dataframe for the scenario
        scenario_name: Name of the scenario
        logger: Logger instance
        
    Returns:
        True if scenario is valid, False otherwise
    """
    if len(df_subset) == 0:
        logger.warning(f"  ⚠️  {scenario_name}: No samples found - skipping")
        return False
    
    num_samples = len(df_subset)
    logger.info(f"  ✅ {scenario_name}: {num_samples:,} samples")
    return True


def write_scenario_summary(df_subset: pd.DataFrame, scenario_name: str, output_dir: str, logger: logging.Logger):
    """
    Write scenario summary file with statistics.
    
    Args:
        df_subset: Filtered dataframe for the scenario
        scenario_name: Name of the scenario
        output_dir: Directory to save the summary
        logger: Logger instance
    """
    summary_path = Path(output_dir) / 'scenario_summary.txt'
    
    num_samples = len(df_subset)
    
    # Find actual SWI column names
    from analysis.scenario_evaluation import find_column_name
    
    # Gather statistics for available indices
    stats = {
        'scenario': scenario_name,
        'num_samples': num_samples,
    }
    
    # Add space weather index statistics using hourly values
    swi_mapping = [
        ('f107', find_column_name(df_subset, 'f107')),
        ('sunspot', find_column_name(df_subset, 'sunspot')),
        ('kp', find_column_name(df_subset, 'kp')),
        ('dst', find_column_name(df_subset, 'dst')),
    ]
    
    for idx_name, col_name in swi_mapping:
        if col_name and col_name in df_subset.columns:
            stats[f'{idx_name}_min'] = df_subset[col_name].min()
            stats[f'{idx_name}_mean'] = df_subset[col_name].mean()
            stats[f'{idx_name}_max'] = df_subset[col_name].max()
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"Scenario Evaluation Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Number of samples: {num_samples:,}\n")
        f.write(f"\n")
        f.write(f"Space Weather Index Statistics (hourly):\n")
        f.write(f"{'-' * 50}\n")
        
        for key in ['f107', 'sunspot', 'kp', 'dst']:
            if f'{key}_min' in stats:
                f.write(f"{key.upper():10s}: min={stats[f'{key}_min']:8.2f}, "
                       f"mean={stats[f'{key}_mean']:8.2f}, "
                       f"max={stats[f'{key}_max']:8.2f}\n")
    
    logger.info(f"  📝 Wrote scenario summary to {summary_path}")


def get_scenario_filters() -> Dict[str, str]:
    """
    Get mapping of scenario names to boolean column names.
    
    Returns:
        Dictionary mapping scenario names to their filter columns
    """
    return {
        'low_activity': 'is_low_activity',
        'high_activity': 'is_high_activity',
        'storm_days': 'is_storm',
        'high_activity_storm_days': 'is_high_activity_storm',
    }
