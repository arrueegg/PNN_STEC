"""
STEC Evaluation Plotting System

This module provides a comprehensive plotting system that combines:
1. Basic scatter/density plots (Model vs GIM vs Ground Truth)
2. Enhanced feature-dependent analysis (seasonal, spatial, temporal)
3. Statistical analysis and significance testing

All plots are generated in one location with a single function call.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class STECPlotter:
    """
    Comprehensive plotting system for STEC evaluation.
    
    Combines basic scatter/density plots with enhanced feature analysis
    in a single, comprehensive plotting workflow.
    """
    
    def __init__(self, output_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize the STEC plotter.
        
        Args:
            output_dir: Directory where all plots will be saved
            logger: Optional logger instance
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / "plots"
        self.logger = logger or logging.getLogger(__name__)
        
        # Create plots directory
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_plots(self, test_df: pd.DataFrame) -> None:
        """
        Create all evaluation plots from test dataframe.
        
        This is the main entry point that generates:
        1. Basic comparison plots (scatter, density, residuals)
        2. Enhanced feature analysis plots (seasonal, spatial, temporal)
        
        Args:
            test_df: Complete test dataframe with predictions and GIM data
        """
        # Filter valid comparisons
        valid_mask = test_df['gim_success'] & ~test_df['gim_stec'].isna()
        valid_df = test_df[valid_mask].copy()
        
        if len(valid_df) == 0:
            self.logger.warning("No valid GIM comparisons for plotting")
            return
        
        self.logger.info(f"📊 Creating comprehensive STEC analysis plots for {len(valid_df):,} observations")
        
        # Add derived columns for analysis
        valid_df = self._add_analysis_columns(valid_df)
        
        try:
            # 1. Basic Comparison Plots
            self._create_basic_plots(valid_df)
            
            # 2. Enhanced Feature Analysis Plots
            self._create_feature_analysis_plots(valid_df)
            
            self.logger.info(f"✅ All plots saved to: {self.plots_dir}")
            
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")
            raise
    
    def _add_analysis_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns needed for comprehensive analysis."""
        df = df.copy()
        
        # Performance metrics
        df['model_residual'] = df['pred_stec'] - df['target_stec']
        df['gim_residual'] = df['gim_stec'] - df['target_stec']
        df['model_abs_error'] = np.abs(df['model_residual'])
        df['gim_abs_error'] = np.abs(df['gim_residual'])
        df['performance_diff'] = df['gim_abs_error'] - df['model_abs_error']  # Positive = model better
        df['model_better'] = df['model_abs_error'] < df['gim_abs_error']
        
        # Temporal features
        if 'doy' in df.columns:
            df['season'] = pd.cut(df['doy'],
                                 bins=[0, 80, 172, 266, 365],
                                 labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Spatial features
        if 'lat_ipp' in df.columns:
            df['lat_class'] = pd.cut(df['lat_ipp'], 
                                   bins=[-90, -50, -30, 30, 50, 90],
                                   labels=['South Polar', 'South Mid', 'Equatorial', 'North Mid', 'North Polar'],
                                   include_lowest=True)
        
        # Elevation angle binning
        if 'satele' in df.columns:
            df['elev_bin'] = pd.cut(df['satele'],
                                   bins=[0, 20, 40, 60, 90],
                                   labels=['0-20°', '20-40°', '40-60°', '60-90°'],
                                   include_lowest=True)
        
        # Space weather features (if available)
        if 'kp' in df.columns:
            df['kp_level'] = pd.cut(df['kp'], 
                                   bins=[0, 2, 4, 6, 9],
                                   labels=['Quiet', 'Unsettled', 'Active', 'Storm'],
                                   include_lowest=True)
        
        if 'dst' in df.columns:
            df['dst_level'] = pd.cut(df['dst'],
                                    bins=[-np.inf, -100, -50, -20, np.inf],
                                    labels=['Major Storm', 'Moderate Storm', 'Minor Storm', 'Quiet'],
                                    include_lowest=True)
        
        return df
    
    def _create_basic_plots(self, df: pd.DataFrame) -> None:
        """Create basic comparison plots (scatter, density, residuals)."""
        self.logger.info("  📈 Creating basic comparison plots...")
        
        # 1. Triple comparison density plots
        self._plot_triple_comparison_density(df)
        
        # 2. Triple comparison scatter plots  
        self._plot_triple_comparison_scatter(df)
        
        # 3. Model vs GIM scatter
        self._plot_model_vs_gim_scatter(df)
        
        # 4. Triple residual histograms
        self._plot_triple_residual_histograms(df)
        
        # 5. Prediction scatter (model vs ground truth)
        self._plot_prediction_scatter(df)
        
        # 6. Residuals histogram
        self._plot_residuals_histogram(df)
    
    def _create_feature_analysis_plots(self, df: pd.DataFrame) -> None:
        """Create enhanced feature-dependent analysis plots."""
        self.logger.info("  🔍 Creating enhanced feature analysis plots...")
        
        # Check feature availability
        features_available = {
            'seasonal': 'season' in df.columns,
            'yearly': 'year' in df.columns,
            'elevation': 'elev_bin' in df.columns,
            'latitude': 'lat_class' in df.columns,
            'space_weather': any(col in df.columns for col in ['kp_level', 'dst_level'])
        }
        
        # Create plots for available features
        if features_available['seasonal']:
            self._plot_seasonal_analysis(df)
            self.logger.info("    ✅ Seasonal analysis plot created")
        
        if features_available['yearly']:
            self._plot_yearly_analysis(df)
            self.logger.info("    ✅ Yearly trends plot created")
        
        if features_available['elevation']:
            self._plot_elevation_analysis(df)
            self.logger.info("    ✅ Elevation angle analysis plot created")
        
        if features_available['latitude']:
            self._plot_latitude_analysis(df)
            self.logger.info("    ✅ Latitude classification analysis plot created")
        
        if features_available['space_weather']:
            self._plot_space_weather_analysis(df)
            self.logger.info("    ✅ Space weather analysis plot created")
    
    def _plot_triple_comparison_density(self, df: pd.DataFrame) -> None:
        """Create enhanced triple comparison plot with density plots and metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Calculate metrics for each comparison
        def calc_metrics(true_vals, pred_vals):
            rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
            mae = np.mean(np.abs(true_vals - pred_vals))
            corr, _ = pearsonr(true_vals, pred_vals)
            return rmse, mae, corr
        
        # Ground Truth vs Model
        rmse_model, mae_model, corr_model = calc_metrics(df['target_stec'], df['pred_stec'])
        
        h1 = axes[0].hist2d(df['target_stec'], df['pred_stec'], bins=50, 
                           density=True, cmap='Blues', alpha=0.8)
        plt.colorbar(h1[3], ax=axes[0], label='Density', shrink=0.8)
        axes[0].plot([df['target_stec'].min(), df['target_stec'].max()],
                    [df['target_stec'].min(), df['target_stec'].max()], 'r--', lw=3)
        axes[0].set_xlabel('Ground Truth STEC [TECU]', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Model STEC [TECU]', fontsize=12, fontweight='bold')
        axes[0].set_title('Model vs Ground Truth', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([None, 300])
        axes[0].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_model:.2f} TECU\\nMAE: {mae_model:.2f} TECU\\nr: {corr_model:.3f}'
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Ground Truth vs GIM
        rmse_gim, mae_gim, corr_gim = calc_metrics(df['target_stec'], df['gim_stec'])
        
        h2 = axes[1].hist2d(df['target_stec'], df['gim_stec'], bins=50,
                           density=True, cmap='Greens', alpha=0.8)
        plt.colorbar(h2[3], ax=axes[1], label='Density', shrink=0.8)
        axes[1].plot([df['target_stec'].min(), df['target_stec'].max()],
                    [df['target_stec'].min(), df['target_stec'].max()], 'r--', lw=3)
        axes[1].set_xlabel('Ground Truth STEC [TECU]', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('GIM STEC [TECU]', fontsize=12, fontweight='bold')
        axes[1].set_title('GIM vs Ground Truth', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([None, 300])
        axes[1].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_gim:.2f} TECU\\nMAE: {mae_gim:.2f} TECU\\nr: {corr_gim:.3f}'
        axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Model vs GIM
        rmse_mg, mae_mg, corr_mg = calc_metrics(df['gim_stec'], df['pred_stec'])
        
        h3 = axes[2].hist2d(df['gim_stec'], df['pred_stec'], bins=50,
                           density=True, cmap='Oranges', alpha=0.8)
        plt.colorbar(h3[3], ax=axes[2], label='Density', shrink=0.8)
        axes[2].plot([df['gim_stec'].min(), df['gim_stec'].max()],
                    [df['gim_stec'].min(), df['gim_stec'].max()], 'r--', lw=3)
        axes[2].set_xlabel('GIM STEC [TECU]', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Model STEC [TECU]', fontsize=12, fontweight='bold')
        axes[2].set_title('Model vs GIM', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([None, 300])
        axes[2].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_mg:.2f} TECU\\nMAE: {mae_mg:.2f} TECU\\nr: {corr_mg:.3f}'
        axes[2].text(0.05, 0.95, metrics_text, transform=axes[2].transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "triple_comparison_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log the key metrics
        self.logger.info(f"    📈 COMPARISON METRICS:")
        self.logger.info(f"       Model vs Ground Truth: RMSE={rmse_model:.2f}, MAE={mae_model:.2f}, r={corr_model:.3f}")
        self.logger.info(f"       GIM vs Ground Truth:   RMSE={rmse_gim:.2f}, MAE={mae_gim:.2f}, r={corr_gim:.3f}")
        self.logger.info(f"       Model vs GIM:          RMSE={rmse_mg:.2f}, MAE={mae_mg:.2f}, r={corr_mg:.3f}")
    
    def _plot_triple_comparison_scatter(self, df: pd.DataFrame) -> None:
        """Create enhanced triple comparison scatter plot with metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # Calculate metrics for each comparison
        def calc_metrics(true_vals, pred_vals):
            rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
            mae = np.mean(np.abs(true_vals - pred_vals))
            corr, _ = pearsonr(true_vals, pred_vals)
            return rmse, mae, corr
        
        # Ground Truth vs Model
        rmse_model, mae_model, corr_model = calc_metrics(df['target_stec'], df['pred_stec'])
        axes[0].scatter(df['target_stec'], df['pred_stec'], alpha=0.6, s=1, color='blue')
        axes[0].plot([df['target_stec'].min(), df['target_stec'].max()],
                    [df['target_stec'].min(), df['target_stec'].max()], 'r--', lw=2)
        axes[0].set_xlabel('Ground Truth STEC [TECU]', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Model STEC [TECU]', fontsize=12, fontweight='bold')
        axes[0].set_title('Model vs Ground Truth', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([None, 300])
        axes[0].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_model:.2f} TECU\\nMAE: {mae_model:.2f} TECU\\nr: {corr_model:.3f}'
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
                    bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Ground Truth vs GIM
        rmse_gim, mae_gim, corr_gim = calc_metrics(df['target_stec'], df['gim_stec'])
        axes[1].scatter(df['target_stec'], df['gim_stec'], alpha=0.6, s=1, color='green')
        axes[1].plot([df['target_stec'].min(), df['target_stec'].max()],
                    [df['target_stec'].min(), df['target_stec'].max()], 'r--', lw=2)
        axes[1].set_xlabel('Ground Truth STEC [TECU]', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('GIM STEC [TECU]', fontsize=12, fontweight='bold')
        axes[1].set_title('GIM vs Ground Truth', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([None, 300])
        axes[1].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_gim:.2f} TECU\\nMAE: {mae_gim:.2f} TECU\\nr: {corr_gim:.3f}'
        axes[1].text(0.05, 0.95, metrics_text, transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        # Model vs GIM
        rmse_mg, mae_mg, corr_mg = calc_metrics(df['gim_stec'], df['pred_stec'])
        axes[2].scatter(df['gim_stec'], df['pred_stec'], alpha=0.6, s=1, color='orange')
        axes[2].plot([df['gim_stec'].min(), df['gim_stec'].max()],
                    [df['gim_stec'].min(), df['gim_stec'].max()], 'r--', lw=2)
        axes[2].set_xlabel('GIM STEC [TECU]', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Model STEC [TECU]', fontsize=12, fontweight='bold')
        axes[2].set_title('Model vs GIM', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim([None, 300])
        axes[2].set_ylim([None, 300])
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse_mg:.2f} TECU\\nMAE: {mae_mg:.2f} TECU\\nr: {corr_mg:.3f}'
        axes[2].text(0.05, 0.95, metrics_text, transform=axes[2].transAxes,
                    bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.9),
                    verticalalignment='top', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "triple_comparison_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_vs_gim_scatter(self, df: pd.DataFrame) -> None:
        """Create Model vs GIM scatter plot."""
        plt.figure(figsize=(10, 8))
        plt.scatter(df['gim_stec'], df['pred_stec'], alpha=0.6, s=1)
        plt.plot([df['gim_stec'].min(), df['gim_stec'].max()], 
                [df['gim_stec'].min(), df['gim_stec'].max()], 'r--', lw=2)
        plt.xlabel('GIM STEC [TECU]')
        plt.ylabel('Model STEC [TECU]')
        plt.title('Model vs GIM STEC Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        corr, _ = pearsonr(df['gim_stec'], df['pred_stec'])
        rmse = np.sqrt(np.mean((df['gim_stec'] - df['pred_stec'])**2))
        plt.text(0.05, 0.95, f'r = {corr:.3f}\\nRMSE = {rmse:.2f} TECU', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "model_vs_gim_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_triple_residual_histograms(self, df: pd.DataFrame) -> None:
        """Create triple residual histograms."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Model residuals (Model - Ground Truth)
        model_residuals = df['model_residual']
        axes[0].hist(model_residuals, bins=50, alpha=0.7, color='blue', density=True)
        axes[0].axvline(np.mean(model_residuals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(model_residuals):.2f}')
        axes[0].axvline(np.median(model_residuals), color='green', linestyle='--', 
                       label=f'Median: {np.median(model_residuals):.2f}')
        axes[0].set_xlabel('Model Residuals [TECU]')
        axes[0].set_ylabel('Density')
        axes[0].set_title('Model - Ground Truth')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # GIM residuals (GIM - Ground Truth)
        gim_residuals = df['gim_residual']
        axes[1].hist(gim_residuals, bins=50, alpha=0.7, color='green', density=True)
        axes[1].axvline(np.mean(gim_residuals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(gim_residuals):.2f}')
        axes[1].axvline(np.median(gim_residuals), color='blue', linestyle='--', 
                       label=f'Median: {np.median(gim_residuals):.2f}')
        axes[1].set_xlabel('GIM Residuals [TECU]')
        axes[1].set_ylabel('Density')
        axes[1].set_title('GIM - Ground Truth')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Model vs GIM residuals (Model - GIM)
        model_gim_residuals = df['pred_stec'] - df['gim_stec']
        axes[2].hist(model_gim_residuals, bins=50, alpha=0.7, color='orange', density=True)
        axes[2].axvline(np.mean(model_gim_residuals), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(model_gim_residuals):.2f}')
        axes[2].axvline(np.median(model_gim_residuals), color='blue', linestyle='--', 
                       label=f'Median: {np.median(model_gim_residuals):.2f}')
        axes[2].set_xlabel('Model - GIM [TECU]')
        axes[2].set_ylabel('Density')
        axes[2].set_title('Model - GIM')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "triple_residual_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self, df: pd.DataFrame) -> None:
        """Create prediction scatter plot (model vs ground truth)."""
        plt.figure(figsize=(10, 8))
        plt.scatter(df['target_stec'], df['pred_stec'], alpha=0.6, s=1)
        plt.plot([df['target_stec'].min(), df['target_stec'].max()], 
                [df['target_stec'].min(), df['target_stec'].max()], 'r--', lw=2)
        plt.xlabel('Ground Truth STEC [TECU]')
        plt.ylabel('Predicted STEC [TECU]')
        plt.title('Model Predictions vs Ground Truth')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        corr, _ = pearsonr(df['target_stec'], df['pred_stec'])
        rmse = np.sqrt(np.mean((df['target_stec'] - df['pred_stec'])**2))
        plt.text(0.05, 0.95, f'r = {corr:.3f}\\nRMSE = {rmse:.2f} TECU', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='lightblue'))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "prediction_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residuals_histogram(self, df: pd.DataFrame) -> None:
        """Create residuals histogram (model - ground truth)."""
        plt.figure(figsize=(10, 6))
        residuals = df['model_residual']
        plt.hist(residuals, bins=50, alpha=0.7, density=True)
        plt.axvline(np.mean(residuals), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(residuals):.2f}')
        plt.axvline(np.median(residuals), color='green', linestyle='--', 
                   label=f'Median: {np.median(residuals):.2f}')
        plt.xlabel('Residuals [TECU]')
        plt.ylabel('Density')
        plt.title('Model Residuals Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "residuals_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_seasonal_analysis(self, df: pd.DataFrame) -> None:
        """Create detailed seasonal performance plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Seasonal Performance Analysis: Model vs GIM', fontsize=16)
        
        seasonal_data = df.groupby('season').agg({
            'model_abs_error': ['mean', 'std', 'count'],
            'gim_abs_error': ['mean', 'std', 'count'],
            'model_better': 'mean',
            'performance_diff': ['mean', 'std']
        })
        
        seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_data = seasonal_data.reindex(seasons)
        
        x_pos = range(len(seasons))
        width = 0.35
        
        # MAE comparison by season
        axes[0,0].bar([x - width/2 for x in x_pos], seasonal_data[('model_abs_error', 'mean')],
                     width, label='Model MAE', alpha=0.8, capsize=5,
                     yerr=seasonal_data[('model_abs_error', 'std')])
        axes[0,0].bar([x + width/2 for x in x_pos], seasonal_data[('gim_abs_error', 'mean')],
                     width, label='GIM MAE', alpha=0.8, capsize=5,
                     yerr=seasonal_data[('gim_abs_error', 'std')])
        axes[0,0].set_xlabel('Season')
        axes[0,0].set_ylabel('Mean Absolute Error (TECU)')
        axes[0,0].set_title('MAE by Season')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(seasons)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Model superiority rate by season
        axes[0,1].bar(x_pos, seasonal_data[('model_better', 'mean')], alpha=0.8, color='green')
        axes[0,1].set_xlabel('Season')
        axes[0,1].set_ylabel('Fraction Model Better')
        axes[0,1].set_title('Model Outperformance by Season')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(seasons)
        axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance difference by season
        axes[1,0].bar(x_pos, seasonal_data[('performance_diff', 'mean')], alpha=0.8, color='purple',
                     capsize=5, yerr=seasonal_data[('performance_diff', 'std')])
        axes[1,0].set_xlabel('Season')
        axes[1,0].set_ylabel('Performance Difference\\n(GIM Error - Model Error)')
        axes[1,0].set_title('Performance Difference by Season')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(seasons)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample size by season
        axes[1,1].bar(x_pos, seasonal_data[('model_abs_error', 'count')], alpha=0.8, color='orange')
        axes[1,1].set_xlabel('Season')
        axes[1,1].set_ylabel('Number of Observations')
        axes[1,1].set_title('Sample Size by Season')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(seasons)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "seasonal_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_yearly_analysis(self, df: pd.DataFrame) -> None:
        """Create detailed yearly trends plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Yearly Performance Trends: Model vs GIM', fontsize=16)
        
        yearly_data = df.groupby('year').agg({
            'model_abs_error': ['mean', 'std', 'count'],
            'gim_abs_error': ['mean', 'std', 'count'],
            'model_better': 'mean',
            'performance_diff': ['mean', 'std']
        })
        
        years = yearly_data.index
        
        # MAE trends over years
        axes[0,0].errorbar(years, yearly_data[('model_abs_error', 'mean')],
                          yerr=yearly_data[('model_abs_error', 'std')],
                          marker='o', label='Model MAE', linewidth=2, capsize=5)
        axes[0,0].errorbar(years, yearly_data[('gim_abs_error', 'mean')],
                          yerr=yearly_data[('gim_abs_error', 'std')],
                          marker='s', label='GIM MAE', linewidth=2, capsize=5)
        axes[0,0].set_xlabel('Year')
        axes[0,0].set_ylabel('Mean Absolute Error (TECU)')
        axes[0,0].set_title('MAE Trends Over Years')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Model superiority rate over years
        axes[0,1].plot(years, yearly_data[('model_better', 'mean')], 
                      'go-', linewidth=2, markersize=6)
        axes[0,1].set_xlabel('Year')
        axes[0,1].set_ylabel('Fraction Model Better')
        axes[0,1].set_title('Model Outperformance Rate Over Years')
        axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance difference trends
        axes[1,0].errorbar(years, yearly_data[('performance_diff', 'mean')],
                          yerr=yearly_data[('performance_diff', 'std')],
                          marker='d', color='purple', linewidth=2, capsize=5)
        axes[1,0].set_xlabel('Year')
        axes[1,0].set_ylabel('Performance Difference\\n(GIM Error - Model Error)')
        axes[1,0].set_title('Performance Difference Trends')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample size trends
        axes[1,1].bar(years, yearly_data[('model_abs_error', 'count')], alpha=0.8, color='orange')
        axes[1,1].set_xlabel('Year')
        axes[1,1].set_ylabel('Number of Observations')
        axes[1,1].set_title('Sample Size by Year')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "yearly_performance_trends.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_elevation_analysis(self, df: pd.DataFrame) -> None:
        """Create detailed elevation angle analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Elevation Angle Performance Analysis: Model vs GIM', fontsize=16)
        
        elev_data = df.groupby('elev_bin', observed=True).agg({
            'model_abs_error': ['mean', 'std', 'count'],
            'gim_abs_error': ['mean', 'std', 'count'],
            'model_better': 'mean',
            'performance_diff': ['mean', 'std']
        })
        
        elev_bins = elev_data.index
        x_pos = range(len(elev_bins))
        width = 0.35
        
        # MAE by elevation
        axes[0,0].bar([x - width/2 for x in x_pos], elev_data[('model_abs_error', 'mean')],
                     width, label='Model MAE', alpha=0.8, capsize=5,
                     yerr=elev_data[('model_abs_error', 'std')])
        axes[0,0].bar([x + width/2 for x in x_pos], elev_data[('gim_abs_error', 'mean')],
                     width, label='GIM MAE', alpha=0.8, capsize=5,
                     yerr=elev_data[('gim_abs_error', 'std')])
        axes[0,0].set_xlabel('Elevation Angle')
        axes[0,0].set_ylabel('Mean Absolute Error (TECU)')
        axes[0,0].set_title('MAE by Elevation Angle')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(elev_bins, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Model superiority by elevation
        axes[0,1].bar(x_pos, elev_data[('model_better', 'mean')], alpha=0.8, color='green')
        axes[0,1].set_xlabel('Elevation Angle')
        axes[0,1].set_ylabel('Fraction Model Better')
        axes[0,1].set_title('Model Outperformance by Elevation')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(elev_bins, rotation=45)
        axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance difference by elevation
        axes[1,0].bar(x_pos, elev_data[('performance_diff', 'mean')], alpha=0.8, color='purple',
                     capsize=5, yerr=elev_data[('performance_diff', 'std')])
        axes[1,0].set_xlabel('Elevation Angle')
        axes[1,0].set_ylabel('Performance Difference\\n(GIM Error - Model Error)')
        axes[1,0].set_title('Performance Difference by Elevation')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(elev_bins, rotation=45)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample distribution by elevation
        axes[1,1].bar(x_pos, elev_data[('model_abs_error', 'count')], alpha=0.8, color='orange')
        axes[1,1].set_xlabel('Elevation Angle')
        axes[1,1].set_ylabel('Number of Observations')
        axes[1,1].set_title('Sample Size by Elevation')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(elev_bins, rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "elevation_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latitude_analysis(self, df: pd.DataFrame) -> None:
        """Create detailed latitude analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Latitude Classification Performance Analysis: Model vs GIM', fontsize=16)
        
        lat_data = df.groupby('lat_class', observed=True).agg({
            'model_abs_error': ['mean', 'std', 'count'],
            'gim_abs_error': ['mean', 'std', 'count'],
            'model_better': 'mean',
            'performance_diff': ['mean', 'std']
        })
        
        lat_classes = lat_data.index
        x_pos = range(len(lat_classes))
        width = 0.35
        
        # MAE by latitude class
        axes[0,0].bar([x - width/2 for x in x_pos], lat_data[('model_abs_error', 'mean')],
                     width, label='Model MAE', alpha=0.8, capsize=5,
                     yerr=lat_data[('model_abs_error', 'std')])
        axes[0,0].bar([x + width/2 for x in x_pos], lat_data[('gim_abs_error', 'mean')],
                     width, label='GIM MAE', alpha=0.8, capsize=5,
                     yerr=lat_data[('gim_abs_error', 'std')])
        axes[0,0].set_xlabel('Latitude Class')
        axes[0,0].set_ylabel('Mean Absolute Error (TECU)')
        axes[0,0].set_title('MAE by Latitude Class')
        axes[0,0].set_xticks(x_pos)
        axes[0,0].set_xticklabels(lat_classes)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Model superiority by latitude class
        axes[0,1].bar(x_pos, lat_data[('model_better', 'mean')], alpha=0.8, color='green')
        axes[0,1].set_xlabel('Latitude Class')
        axes[0,1].set_ylabel('Fraction Model Better')
        axes[0,1].set_title('Model Outperformance by Latitude Class')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(lat_classes)
        axes[0,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0,1].grid(True, alpha=0.3)
        
        # Performance difference by latitude class
        axes[1,0].bar(x_pos, lat_data[('performance_diff', 'mean')], alpha=0.8, color='purple',
                     capsize=5, yerr=lat_data[('performance_diff', 'std')])
        axes[1,0].set_xlabel('Latitude Class')
        axes[1,0].set_ylabel('Performance Difference\\n(GIM Error - Model Error)')
        axes[1,0].set_title('Performance Difference by Latitude Class')
        axes[1,0].set_xticks(x_pos)
        axes[1,0].set_xticklabels(lat_classes)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].grid(True, alpha=0.3)
        
        # Sample distribution by latitude class
        axes[1,1].bar(x_pos, lat_data[('model_abs_error', 'count')], alpha=0.8, color='orange')
        axes[1,1].set_xlabel('Latitude Class')
        axes[1,1].set_ylabel('Number of Observations')
        axes[1,1].set_title('Sample Size by Latitude Class')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(lat_classes)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "latitude_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_space_weather_analysis(self, df: pd.DataFrame) -> None:
        """Create space weather analysis plot."""
        # Determine which space weather features are available
        has_kp = 'kp_level' in df.columns
        has_dst = 'dst_level' in df.columns
        
        if not (has_kp or has_dst):
            return
        
        n_plots = sum([has_kp, has_dst])
        fig, axes = plt.subplots(n_plots, 2, figsize=(15, 6*n_plots))
        if n_plots == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Space Weather Performance Analysis: Model vs GIM', fontsize=16)
        
        plot_idx = 0
        
        if has_kp:
            kp_data = df.groupby('kp_level', observed=True).agg({
                'model_abs_error': ['mean', 'std'],
                'gim_abs_error': ['mean', 'std'],
                'model_better': 'mean'
            })
            
            kp_levels = kp_data.index
            x_pos = range(len(kp_levels))
            width = 0.35
            
            # MAE by Kp level
            axes[plot_idx,0].bar([x - width/2 for x in x_pos], kp_data[('model_abs_error', 'mean')],
                                width, label='Model MAE', alpha=0.8, capsize=5,
                                yerr=kp_data[('model_abs_error', 'std')])
            axes[plot_idx,0].bar([x + width/2 for x in x_pos], kp_data[('gim_abs_error', 'mean')],
                                width, label='GIM MAE', alpha=0.8, capsize=5,
                                yerr=kp_data[('gim_abs_error', 'std')])
            axes[plot_idx,0].set_xlabel('Kp Level')
            axes[plot_idx,0].set_ylabel('Mean Absolute Error (TECU)')
            axes[plot_idx,0].set_title('MAE by Kp Activity Level')
            axes[plot_idx,0].set_xticks(x_pos)
            axes[plot_idx,0].set_xticklabels(kp_levels)
            axes[plot_idx,0].legend()
            axes[plot_idx,0].grid(True, alpha=0.3)
            
            # Model superiority by Kp
            axes[plot_idx,1].bar(x_pos, kp_data[('model_better', 'mean')], alpha=0.8, color='green')
            axes[plot_idx,1].set_xlabel('Kp Level')
            axes[plot_idx,1].set_ylabel('Fraction Model Better')
            axes[plot_idx,1].set_title('Model Outperformance by Kp Level')
            axes[plot_idx,1].set_xticks(x_pos)
            axes[plot_idx,1].set_xticklabels(kp_levels)
            axes[plot_idx,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            axes[plot_idx,1].grid(True, alpha=0.3)
            
            plot_idx += 1
        
        if has_dst:
            dst_data = df.groupby('dst_level', observed=True).agg({
                'model_abs_error': ['mean', 'std'],
                'gim_abs_error': ['mean', 'std'],
                'model_better': 'mean'
            })
            
            dst_levels = dst_data.index
            x_pos = range(len(dst_levels))
            width = 0.35
            
            # MAE by Dst level
            axes[plot_idx,0].bar([x - width/2 for x in x_pos], dst_data[('model_abs_error', 'mean')],
                                width, label='Model MAE', alpha=0.8, capsize=5,
                                yerr=dst_data[('model_abs_error', 'std')])
            axes[plot_idx,0].bar([x + width/2 for x in x_pos], dst_data[('gim_abs_error', 'mean')],
                                width, label='GIM MAE', alpha=0.8, capsize=5,
                                yerr=dst_data[('gim_abs_error', 'std')])
            axes[plot_idx,0].set_xlabel('Dst Level')
            axes[plot_idx,0].set_ylabel('Mean Absolute Error (TECU)')
            axes[plot_idx,0].set_title('MAE by Dst Activity Level')
            axes[plot_idx,0].set_xticks(x_pos)
            axes[plot_idx,0].set_xticklabels(dst_levels)
            axes[plot_idx,0].legend()
            axes[plot_idx,0].grid(True, alpha=0.3)
            
            # Model superiority by Dst
            axes[plot_idx,1].bar(x_pos, dst_data[('model_better', 'mean')], alpha=0.8, color='green')
            axes[plot_idx,1].set_xlabel('Dst Level')
            axes[plot_idx,1].set_ylabel('Fraction Model Better')
            axes[plot_idx,1].set_title('Model Outperformance by Dst Level')
            axes[plot_idx,1].set_xticks(x_pos)
            axes[plot_idx,1].set_xticklabels(dst_levels)
            axes[plot_idx,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            axes[plot_idx,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "space_weather_performance_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()


def create_stec_plots(test_df: pd.DataFrame, output_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    """
    Convenience function to create all STEC evaluation plots.
    
    This is the main entry point for plotting that combines:
    1. Basic scatter/density plots  
    2. Enhanced feature analysis plots
    
    Args:
        test_df: Complete test dataframe with predictions and GIM data
        output_dir: Directory where plots will be saved
        logger: Optional logger instance
    """
    plotter = STECPlotter(output_dir, logger)
    plotter.create_all_plots(test_df)