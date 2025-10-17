#!/usr/bin/env python3
"""
Simple evaluation utilities for CSV saving and statistics.

This replaces the heavy EvaluationOutputManager with lightweight functions
since plotting is now handled by the main plotter.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from scipy.stats import pearsonr
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def save_results_csv(test_df: pd.DataFrame, output_dir: Path) -> None:
    """Save evaluation results to CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stec_comparison.csv"
    test_df.to_csv(csv_path, index=False)
    logger.info(f"💾 Saved results CSV: {csv_path}")


def print_and_save_statistics(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Print and save comprehensive evaluation summary statistics."""
    
    valid_gim_count = results_df['gim_success'].sum()
    total_count = len(results_df)
    
    # Print to console
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"📊 EVALUATION SUMMARY:")
    logger.info(f"   Total observations: {total_count:,}")
    logger.info(f"   Valid GIM comparisons: {valid_gim_count:,} ({100*valid_gim_count/total_count:.1f}%)")
    
    # Prepare statistics content for file
    stats_lines = []
    stats_lines.append(f"STEC EVALUATION SUMMARY")
    stats_lines.append(f"=" * 50)
    stats_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    stats_lines.append(f"")
    stats_lines.append(f"DATASET OVERVIEW:")
    stats_lines.append(f"  Total observations: {total_count:,}")
    stats_lines.append(f"  Valid GIM comparisons: {valid_gim_count:,} ({100*valid_gim_count/total_count:.1f}%)")
    
    if valid_gim_count > 0:
        valid_df = results_df[results_df['gim_success']].copy()
        
        # Basic statistics
        gim_stec_mean = valid_df['gim_stec'].mean()
        gim_stec_std = valid_df['gim_stec'].std()
        model_stec_mean = valid_df['pred_stec'].mean()
        model_stec_std = valid_df['pred_stec'].std()
        
        stats_lines.append(f"")
        stats_lines.append(f"BASIC STATISTICS:")
        stats_lines.append(f"  GIM STEC:   Mean = {gim_stec_mean:.3f} ± {gim_stec_std:.3f} TECU")
        stats_lines.append(f"  Model STEC: Mean = {model_stec_mean:.3f} ± {model_stec_std:.3f} TECU")
        
        if 'target_stec' in valid_df.columns:
            gt_stec_mean = valid_df['target_stec'].mean()
            gt_stec_std = valid_df['target_stec'].std()
            stats_lines.append(f"  Ground Truth: Mean = {gt_stec_mean:.3f} ± {gt_stec_std:.3f} TECU")
                            
            # Calculate comprehensive metrics
            def calc_metrics(true_vals, pred_vals):
                rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
                mae = np.mean(np.abs(true_vals - pred_vals))
                corr, _ = pearsonr(true_vals, pred_vals)
                bias = np.mean(pred_vals - true_vals)
                return rmse, mae, corr, bias
            
            # Model vs Ground Truth
            rmse_model, mae_model, corr_model, bias_model = calc_metrics(
                valid_df['target_stec'], valid_df['pred_stec'])
            
            # GIM vs Ground Truth  
            rmse_gim, mae_gim, corr_gim, bias_gim = calc_metrics(
                valid_df['target_stec'], valid_df['gim_stec'])
            
            # Add detailed metrics to file
            stats_lines.append(f"")
            stats_lines.append(f"DETAILED PERFORMANCE METRICS (vs Ground Truth):")
            stats_lines.append(f"")
            stats_lines.append(f"Model Performance:")
            stats_lines.append(f"  RMSE: {rmse_model:.3f} TECU")
            stats_lines.append(f"  MAE:  {mae_model:.3f} TECU")
            stats_lines.append(f"  r:    {corr_model:.4f}")
            stats_lines.append(f"  Bias: {bias_model:.3f} TECU")
            stats_lines.append(f"")
            stats_lines.append(f"GIM Performance:")
            stats_lines.append(f"  RMSE: {rmse_gim:.3f} TECU")
            stats_lines.append(f"  MAE:  {mae_gim:.3f} TECU")
            stats_lines.append(f"  r:    {corr_gim:.4f}")
            stats_lines.append(f"  Bias: {bias_gim:.3f} TECU")
            
            # Print detailed metrics to console
            logger.info(f"")
            logger.info(f"📈 DETAILED METRICS vs Ground Truth:")
            logger.info(f"   Model Performance:")
            logger.info(f"      RMSE: {rmse_model:.3f} TECU")
            logger.info(f"      MAE:  {mae_model:.3f} TECU") 
            logger.info(f"      r:    {corr_model:.4f}")
            logger.info(f"      Bias: {bias_model:.3f} TECU")
            logger.info(f"   GIM Performance:")
            logger.info(f"      RMSE: {rmse_gim:.3f} TECU")
            logger.info(f"      MAE:  {mae_gim:.3f} TECU")
            logger.info(f"      r:    {corr_gim:.4f}")
            logger.info(f"      Bias: {bias_gim:.3f} TECU")
            
            # Relative performance
            rmse_improvement = ((rmse_gim - rmse_model) / rmse_gim) * 100
            mae_improvement = ((mae_gim - mae_model) / mae_gim) * 100
            
            # Add relative performance to file
            stats_lines.append(f"")
            stats_lines.append(f"RELATIVE PERFORMANCE (Model vs GIM):")
            stats_lines.append(f"  RMSE improvement: {rmse_improvement:+.1f}%")
            stats_lines.append(f"  MAE improvement:  {mae_improvement:+.1f}%")
            
            # Print relative performance to console
            logger.info(f"")
            logger.info(f"🎯 RELATIVE PERFORMANCE (Model vs GIM):")
            logger.info(f"   RMSE improvement: {rmse_improvement:+.1f}%")
            logger.info(f"   MAE improvement:  {mae_improvement:+.1f}%")
    
    # Write statistics to file
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_file_path = output_dir / "evaluation_statistics.txt"
    try:
        with open(stats_file_path, 'w') as f:
            f.write('\n'.join(stats_lines))
        logger.info(f"📄 Statistics saved to: {stats_file_path}")
    except Exception as e:
        logger.warning(f"Failed to save statistics file: {e}")