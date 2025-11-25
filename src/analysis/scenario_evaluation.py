"""
Scenario-based evaluation module for PNN_STEC.

This module implements scenario-based filtering and evaluation based on
solar and geomagnetic activity levels. It computes daily aggregations
and applies threshold-based criteria to identify specific activity scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Constants
MIN_SAMPLES_PER_DAY = 100
MIN_DAYS_PER_SCENARIO = 3
MIN_SAMPLES_PER_SCENARIO = 10000

# Column name mappings - handle both simple and full feature names
COLUMN_MAPPINGS = {
    'f107': ['f107', 'f107_index'],
    'sunspot': ['sunspot', 'R_Sunspot_No'],
    'kp': ['kp', 'Kp_index'],
    'dst': ['dst', 'Dst-index,_nT', 'Dst-index'],
}

# Activity thresholds (based on OMNI 2010-2025 data distribution analysis)
# Low activity: 33rd percentile (ALL conditions must be met - quiet periods)
# High activity: 67th percentile (ANY condition triggers - active periods)
# Storm: 90th percentile for Kp, 10th percentile for Dst (extreme events)
THRESHOLDS = {
    'low_activity': {
        'f107': ('<=', 80.0),
        'sunspot': ('<=', 25.0),
        'kp': ('<=', 10.0),
        'dst': ('>=', -3.0),
    },
    'high_activity': {
        'f107': ('>=', 125),
        'sunspot': ('>=', 86.0),
        'kp': ('>=', 20.0),
        'dst': ('<=', -14.0),
    },
    'storm': {
        'kp': ('>=', 33.0),
        'dst': ('<=', -31.0),
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


def compute_daily_aggregations(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute daily aggregations from the test dataframe.
    
    Derives a 'date' column from datetime or (year+doy), then computes:
    - daily_mean_f107
    - daily_mean_sunspot
    - daily_max_kp
    - daily_min_dst
    - daily_sample_count
    
    Only keeps days with at least MIN_SAMPLES_PER_DAY samples.
    
    Args:
        df: Test dataframe with predictions
        logger: Logger instance
        
    Returns:
        DataFrame with daily aggregations merged back to original data
    """
    # Avoid full copy - work with view where possible
    
    # Derive date column efficiently
    if 'date' not in df.columns:
        if 'datetime' in df.columns:
            date_series = pd.to_datetime(df['datetime']).dt.date
            logger.info(f"  Derived date from 'datetime' column")
        elif 'year' in df.columns and 'doy' in df.columns:
            # Ensure year and doy are numeric without modifying original df
            year_numeric = pd.to_numeric(df['year'], errors='coerce')
            doy_numeric = pd.to_numeric(df['doy'], errors='coerce')
            
            # Check for invalid values
            valid_mask = year_numeric.notna() & doy_numeric.notna()
            if not valid_mask.all():
                logger.warning(f"  Found {(~valid_mask).sum()} rows with invalid year/doy")
                df = df[valid_mask].copy()
                year_numeric = year_numeric[valid_mask]
                doy_numeric = doy_numeric[valid_mask]
            
            # Compute date efficiently
            datetime_series = pd.to_datetime(year_numeric.astype(int), format='%Y') + pd.to_timedelta(
                doy_numeric.astype(int) - 1, unit='D'
            )
            date_series = datetime_series.dt.date
            logger.info(f"  Derived date from 'year' + 'doy' columns")
        else:
            logger.error("Cannot derive date: missing 'datetime' or 'year'+'doy' columns")
            raise ValueError("Cannot derive date from dataframe")
    else:
        date_series = df['date']
    
    # Count samples per day using the date series
    daily_counts = pd.Series(date_series).value_counts()
    num_dates = len(daily_counts)
    
    logger.info(f"Computing daily aggregations across {num_dates} unique dates...")
    logger.info(f"  Sample distribution: min={daily_counts.min()}, "
                f"max={daily_counts.max()}, mean={daily_counts.mean():.1f}")
    
    # Map canonical names to actual column names
    col_map = {}
    for canonical in ['f107', 'sunspot', 'kp', 'dst']:
        actual_col = find_column_name(df, canonical)
        if actual_col:
            col_map[canonical] = actual_col
    
    if col_map:
        logger.info(f"  Mapped columns: {', '.join([f'{k}->{v}' for k, v in col_map.items()])}")
    
    # Build aggregation dict efficiently
    agg_dict = {}
    rename_dict = {}
    
    if 'f107' in col_map:
        agg_dict[col_map['f107']] = 'mean'
        rename_dict[col_map['f107']] = 'daily_mean_f107'
    if 'sunspot' in col_map:
        agg_dict[col_map['sunspot']] = 'mean'
        rename_dict[col_map['sunspot']] = 'daily_mean_sunspot'
    if 'kp' in col_map:
        agg_dict[col_map['kp']] = 'max'
        rename_dict[col_map['kp']] = 'daily_max_kp'
    if 'dst' in col_map:
        agg_dict[col_map['dst']] = 'min'
        rename_dict[col_map['dst']] = 'daily_min_dst'
    
    # Compute daily statistics in one pass
    if agg_dict:
        # Create temporary df with only needed columns to reduce memory
        cols_needed = list(agg_dict.keys())
        temp_df = df[cols_needed].copy()
        temp_df['date'] = date_series
        
        daily_stats = temp_df.groupby('date', observed=True).agg(agg_dict).reset_index()
        daily_stats.rename(columns=rename_dict, inplace=True)
        
        # Add counts
        daily_stats['daily_sample_count'] = daily_stats['date'].map(daily_counts)
        
        del temp_df  # Free memory
    else:
        daily_stats = pd.DataFrame({
            'date': daily_counts.index,
            'daily_sample_count': daily_counts.values
        })
    
    # Filter days with sufficient samples
    valid_days = daily_stats[daily_stats['daily_sample_count'] >= MIN_SAMPLES_PER_DAY].copy()
    logger.info(f"  Valid days (>={MIN_SAMPLES_PER_DAY} samples): {len(valid_days)}/{len(daily_stats)}")
    
    if len(valid_days) == 0:
        logger.warning(f"  No days meet the minimum sample threshold!")
        # Return empty dataframe with expected columns to avoid downstream errors
        result = df.iloc[:0].copy()
        for col in ['daily_sample_count', 'daily_mean_f107', 'daily_mean_sunspot', 'daily_max_kp', 'daily_min_dst']:
            if col not in result.columns:
                result[col] = pd.Series(dtype=float)
        result['date'] = pd.Series(dtype=object)
        return result
    
    # Add date column to df if not present and merge efficiently
    if 'date' not in df.columns:
        df = df.copy()
        df['date'] = date_series
    
    # Filter df to only valid dates (more memory efficient than merge for large df)
    valid_dates_set = set(valid_days['date'])
    mask = df['date'].isin(valid_dates_set)
    df_filtered = df[mask].copy()
    
    # Merge daily stats
    df_result = df_filtered.merge(valid_days, on='date', how='left')
    
    logger.info(f"  Retained {len(df_result):,} samples after filtering by sample count")
    
    del df_filtered, valid_days  # Free memory
    return df_result


def identify_scenarios(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Add boolean flags for each scenario to the dataframe.
    
    Scenarios:
    - low_activity: f107<=90 AND sunspot<=20 AND kp<=3 AND dst>=-20 (all applicable must be satisfied)
    - high_activity: f107>=150 OR sunspot>=50 OR kp>=4 OR dst<=-50 (any applicable qualifies)
    - storm_days: kp>=6 OR dst<=-100
    - low_activity_storm_days: low_activity AND storm_days
    - high_activity_storm_days: high_activity AND storm_days
    
    Args:
        df: Dataframe with daily aggregations
        logger: Logger instance
        
    Returns:
        Dataframe with scenario boolean columns added
    """
    if len(df) == 0:
        logger.warning("Empty dataframe passed to identify_scenarios")
        return df
    
    # Detect available indices
    available_indices = {
        'f107': 'daily_mean_f107' in df.columns,
        'sunspot': 'daily_mean_sunspot' in df.columns,
        'kp': 'daily_max_kp' in df.columns,
        'dst': 'daily_min_dst' in df.columns,
    }
    
    logger.info(f"Available space weather indices: {[k for k, v in available_indices.items() if v]}")
    
    # Initialize scenario flags
    df['is_low_activity'] = True
    df['is_high_activity'] = False
    df['is_storm'] = False
    
    # Low activity: ALL applicable conditions must be satisfied
    low_conditions = []
    if available_indices['f107']:
        low_conditions.append(df['daily_mean_f107'] <= 90)
    if available_indices['sunspot']:
        low_conditions.append(df['daily_mean_sunspot'] <= 20)
    if available_indices['kp']:
        low_conditions.append(df['daily_max_kp'] <= 3)
    if available_indices['dst']:
        low_conditions.append(df['daily_min_dst'] >= -20)
    
    if low_conditions:
        df['is_low_activity'] = np.logical_and.reduce(low_conditions)
    else:
        logger.warning("No indices available for low_activity scenario - skipping")
        df['is_low_activity'] = False
    
    # High activity: ANY applicable condition qualifies
    high_conditions = []
    if available_indices['f107']:
        high_conditions.append(df['daily_mean_f107'] >= 150)
    if available_indices['sunspot']:
        high_conditions.append(df['daily_mean_sunspot'] >= 50)
    if available_indices['kp']:
        high_conditions.append(df['daily_max_kp'] >= 4)
    if available_indices['dst']:
        high_conditions.append(df['daily_min_dst'] <= -50)
    
    if high_conditions:
        df['is_high_activity'] = np.logical_or.reduce(high_conditions)
    else:
        logger.warning("No indices available for high_activity scenario - skipping")
        df['is_high_activity'] = False
    
    # Storm: kp>=6 OR dst<=-100
    storm_conditions = []
    if available_indices['kp']:
        storm_conditions.append(df['daily_max_kp'] >= 6)
    if available_indices['dst']:
        storm_conditions.append(df['daily_min_dst'] <= -100)
    
    if storm_conditions:
        df['is_storm'] = np.logical_or.reduce(storm_conditions)
    else:
        logger.warning("No indices available for storm_days scenario - skipping")
        df['is_storm'] = False
    
    # Compute composite scenarios efficiently using numpy
    is_low_activity_storm = df['is_low_activity'].values & df['is_storm'].values
    is_high_activity_storm = df['is_high_activity'].values & df['is_storm'].values
    
    df['is_low_activity_storm'] = is_low_activity_storm
    df['is_high_activity_storm'] = is_high_activity_storm
    
    # Log scenario counts
    total = len(df)
    logger.info("Scenario sample counts:")
    logger.info(f"  low_activity: {df['is_low_activity'].sum():,} samples ({df['is_low_activity'].sum()/total*100:.1f}%)")
    logger.info(f"  high_activity: {df['is_high_activity'].sum():,} samples ({df['is_high_activity'].sum()/total*100:.1f}%)")
    logger.info(f"  storm_days: {df['is_storm'].sum():,} samples ({df['is_storm'].sum()/total*100:.1f}%)")
    logger.info(f"  low_activity_storm_days: {is_low_activity_storm.sum():,} samples ({is_low_activity_storm.sum()/total*100:.1f}%)")
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
    
    num_days = df_subset['date'].nunique()
    num_samples = len(df_subset)
    
    if num_days < MIN_DAYS_PER_SCENARIO:
        logger.warning(f"  ⚠️  {scenario_name}: Only {num_days} days (min={MIN_DAYS_PER_SCENARIO}) - skipping")
        return False
    
    if num_samples < MIN_SAMPLES_PER_SCENARIO:
        logger.warning(f"  ⚠️  {scenario_name}: Only {num_samples:,} samples (min={MIN_SAMPLES_PER_SCENARIO:,}) - skipping")
        return False
    
    logger.info(f"  ✅ {scenario_name}: {num_samples:,} samples across {num_days} days")
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
    
    num_days = df_subset['date'].nunique()
    num_samples = len(df_subset)
    
    # Gather statistics for available indices
    stats = {
        'scenario': scenario_name,
        'num_days': num_days,
        'num_samples': num_samples,
    }
    
    # Add space weather index statistics
    for idx_col, daily_col in [
        ('f107', 'daily_mean_f107'),
        ('sunspot', 'daily_mean_sunspot'),
        ('kp', 'daily_max_kp'),
        ('dst', 'daily_min_dst'),
    ]:
        if daily_col in df_subset.columns:
            stats[f'{idx_col}_min'] = df_subset[daily_col].min()
            stats[f'{idx_col}_mean'] = df_subset[daily_col].mean()
            stats[f'{idx_col}_max'] = df_subset[daily_col].max()
    
    # Write summary
    with open(summary_path, 'w') as f:
        f.write(f"Scenario Evaluation Summary\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Number of days: {num_days}\n")
        f.write(f"Number of samples: {num_samples:,}\n")
        f.write(f"\n")
        f.write(f"Space Weather Index Statistics:\n")
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
        'low_activity_storm_days': 'is_low_activity_storm',
        'high_activity_storm_days': 'is_high_activity_storm',
    }
