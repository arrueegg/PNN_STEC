#!/usr/bin/env python3
"""
Test script to generate the improved uncertainty analysis plots.
This script can be used to test the new plotting functions on existing experiment data.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append('/scratch2/arrueegg/WP4/PNN_STEC/src')

from viz import plot_binned_uncertainty_analysis, plot_binned_uncertainty_analysis_lines_only

def test_uncertainty_plots():
    """Test the uncertainty plotting functions with sample data"""
    
    # Use a specific experiment that we know has the data
    exp_name_BNN = 'Pretrain_STEC_BNN_NLL_h1024_l6_lr2e-3_bs512_GNLL_Adam_CosineAnnealingLR_sub500K_SH0_lw5e-1_SWI'
    exp_name_DE = 'Pretrain_STEC_DE_MLP_h1024_l4_lr2e-4_bs512_GNLL_Adam_CosineAnnealingLR_ens5_sub500K_SH0_lw5e-1_SWI'
    exp_name = exp_name_DE  # Change to desired experiment
    exp_path = f'/scratch2/arrueegg/WP4/PNN_STEC/experiments/{exp_name}'
    
    # Use inference_results.csv
    test_csv = os.path.join(exp_path, 'inference_results.csv')
    
    if not os.path.exists(test_csv):
        print(f"Test CSV not found at: {test_csv}")
        return
    
    print(f"Testing with experiment: {exp_name_DE}")
    print(f"Loading data from: {test_csv}")
    
    # Load the test results
    try:
        df = pd.read_csv(test_csv)
        print(f"Loaded {len(df)} test samples")
        
        # Sample only 20% of the data randomly to avoid OOM
        sample_size = int(len(df) * 0.2)
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Using random sample of {len(df)} test samples (20% of total)")
        
        # Check required columns
        required_cols = ['target_stec', 'pred_stec', 'pred_epistemic_unc', 'pred_aleatoric_unc', 'pred_total_unc']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return
        
        # Create output directory for test plots
        output_dir = '/scratch2/arrueegg/WP4/PNN_STEC/test_plots'
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating plots in: {output_dir}")
        
        # Generate both plots
        print("Generating original plot with boxplots + total uncertainty line...")
        plot_binned_uncertainty_analysis(df, output_dir)
        
        print("Generating new lines-only plot...")
        plot_binned_uncertainty_analysis_lines_only(df, output_dir)
        
        print("✅ Successfully generated both plots!")
        print(f"📁 Check plots in: {output_dir}/uncertainty_analysis/")
        print("   - binned_uncertainty_error_analysis.png (original + total uncertainty)")
        print("   - binned_uncertainty_error_analysis_lines_only.png (lines only)")
        
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_uncertainty_plots()