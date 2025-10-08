"""
Test suite for the refactored visualization modules.

This test module ensures that our refactored code works correctly
and maintains backward compatibility.
"""

import sys
import os
import pandas as pd
import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_sample_data():
    """Create sample test data for visualization functions."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic STEC data
    target_stec = np.random.gamma(2, 15, n_samples)  # Realistic STEC distribution
    noise = np.random.normal(0, 3, n_samples)
    pred_stec = target_stec + noise
    
    # Add uncertainty data
    pred_total_unc = np.abs(np.random.normal(3, 1, n_samples))
    pred_epistemic_unc = pred_total_unc * 0.6 
    pred_aleatoric_unc = pred_total_unc * 0.4
    
    # Add feature data
    doy = np.random.randint(1, 366, n_samples)
    time = np.random.uniform(0, 24, n_samples)
    lat_ipp = np.random.uniform(-90, 90, n_samples)
    lon_ipp = np.random.uniform(-180, 180, n_samples)
    satele = np.random.uniform(10, 90, n_samples)
    satazi = np.random.uniform(0, 360, n_samples)
    
    data = {
        'target_stec': target_stec,
        'pred_stec': pred_stec,
        'pred_total_unc': pred_total_unc,
        'pred_epistemic_unc': pred_epistemic_unc,
        'pred_aleatoric_unc': pred_aleatoric_unc,
        'doy': doy,
        'time': time,
        'lat_ipp': lat_ipp,
        'lon_ipp': lon_ipp,
        'satele': satele,
        'satazi': satazi,
    }
    
    return pd.DataFrame(data)


class TestVisualizationImports:
    """Test that all visualization modules can be imported correctly."""
    
    def test_base_module_import(self):
        """Test that the base module imports correctly."""
        from viz.base import FIGSIZE_WIDE, get_scientific_label, ensure_dir
        
        assert FIGSIZE_WIDE == (16, 10)
        assert get_scientific_label('target_stec') == 'True STEC [TECU]'
        
    def test_distributions_import(self):
        """Test that distributions module imports correctly."""
        from viz.distributions import plot_binned_boxplot, plot_histogram_of_residuals
        assert callable(plot_binned_boxplot)
        assert callable(plot_histogram_of_residuals)
        
    def test_performance_import(self):
        """Test that performance module imports correctly.""" 
        from viz.performance import plot_prediction_scatter, plot_az_el_heatmap
        assert callable(plot_prediction_scatter)
        assert callable(plot_az_el_heatmap)
        
    def test_spatial_import(self):
        """Test that spatial module imports correctly."""
        from viz.spatial import plot_spatial_error_map
        assert callable(plot_spatial_error_map)
        
    def test_uncertainty_import(self):
        """Test that uncertainty module imports correctly."""
        from viz.uncertainty import plot_uncertainty_calibration
        assert callable(plot_uncertainty_calibration)
        
    def test_main_interface_import(self):
        """Test that the main interface imports correctly."""
        from viz import plot_test_metrics, FIGSIZE_WIDE
        assert callable(plot_test_metrics)
        assert FIGSIZE_WIDE == (16, 10)


class TestAnalysisModules:
    """Test that analysis modules work correctly."""
    
    def test_analysis_import(self):
        """Test that analysis module imports correctly."""
        from analysis.metrics import modify_df, calculate_performance_metrics
        assert callable(modify_df)
        assert callable(calculate_performance_metrics)
        
    def test_modify_df_function(self):
        """Test that modify_df adds required columns."""
        from analysis.metrics import modify_df
        
        df = create_sample_data()
        df_modified = modify_df(df)
        
        assert 'residual' in df_modified.columns
        assert 'mae' in df_modified.columns
        assert 'abs_residual' in df_modified.columns
        
        # Check calculations are correct
        expected_residual = df['target_stec'] - df['pred_stec']
        np.testing.assert_array_almost_equal(df_modified['residual'], expected_residual)
        
    def test_performance_metrics(self):
        """Test that performance metrics calculation works."""
        from analysis.metrics import calculate_performance_metrics
        
        df = create_sample_data()
        metrics = calculate_performance_metrics(df)
        
        required_metrics = ['mae', 'rmse', 'bias', 'r2_score', 'correlation']
        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


class TestBackwardCompatibility:
    """Test that refactored code maintains backward compatibility."""
    
    def test_legacy_imports_work(self):
        """Test that old import patterns still work."""
        # Test imports that existing code uses
        from viz import plot_test_metrics
        from viz.base import FIGSIZE_WIDE
        
        assert callable(plot_test_metrics)
        assert FIGSIZE_WIDE == (16, 10)
        
    def test_legacy_function_aliases(self):
        """Test that legacy function aliases work."""
        from viz import (
            plot_binned_uncertainty_analysis_lines_only,
            plot_prediction_quality,
            plot_spatial_analysis,
            plot_uncertainty_analysis
        )
        
        assert callable(plot_binned_uncertainty_analysis_lines_only)
        assert callable(plot_prediction_quality)
        assert callable(plot_spatial_analysis)
        assert callable(plot_uncertainty_analysis)


class TestVisualizationFunctions:
    """Test that visualization functions work with sample data."""
    
    def test_plot_functions_run_without_error(self, tmp_path):
        """Test that plot functions can be called without errors."""
        from viz.distributions import plot_histogram_of_residuals
        from viz.performance import plot_prediction_scatter
        
        df = create_sample_data()
        output_dir = str(tmp_path)
        
        # These should not raise exceptions
        plot_histogram_of_residuals(df, output_dir)
        plot_prediction_scatter(df, output_dir)
        
        # Check that files were created
        files = os.listdir(output_dir)
        assert len(files) > 0  # At least some plots were created
        
    def test_uncertainty_plots_with_uncertainty_data(self, tmp_path):
        """Test uncertainty plots work when uncertainty data is present."""
        from viz.uncertainty import plot_uncertainty_calibration
        
        df = create_sample_data() 
        output_dir = str(tmp_path)
        
        # Should not raise exception with uncertainty data
        plot_uncertainty_calibration(df, output_dir)
        
        files = os.listdir(output_dir)
        assert len(files) > 0


if __name__ == '__main__':
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])