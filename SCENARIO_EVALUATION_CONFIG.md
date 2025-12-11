# Scenario-Based Evaluation Configuration

## Overview

Scenario-based evaluation analyzes model performance under different space weather conditions:
- **Low activity**: Quiet ionospheric conditions (F10.7 ≤ 86.9)
- **High activity**: Active solar conditions (F10.7 ≥ 207.9)
- **Storm days**: Geomagnetic storms (high Kp ≥ 37 or disturbed Dst ≤ -33)

While valuable for detailed analysis, these evaluations add significant runtime during training and inference.

## Default Behavior

**Scenario evaluation is now DISABLED by default** to save computational time during normal training runs.

## Configuration

To enable/disable scenario-based evaluation, use the `evaluation.enable_scenarios` parameter in `config/config.yaml`:

```yaml
# Evaluation configuration
evaluation:
  enable_scenarios: false  # Set to true to enable scenario-based evaluation
```

### Disable (default - saves runtime):
```yaml
evaluation:
  enable_scenarios: false
```

### Enable (for detailed analysis):
```yaml
evaluation:
  enable_scenarios: true
```

## When to Enable Scenario Evaluation

**Enable when:**
- Performing final model evaluation for publication/reporting
- Analyzing model behavior under extreme space weather conditions
- Comparing model robustness across different solar activity levels
- Investigating geomagnetic storm performance

**Keep disabled when:**
- Training models (saves ~20-40% evaluation time)
- Running quick inference tests
- Performing hyperparameter sweeps
- Iterating on model architecture

## Implementation Details

The setting affects:
- `src/training/base_trainer.py`: Main test evaluation during training
- `src/inference_testset.py`: Standalone inference script
- `src/viz/__init__.py`: Plotting functions

**Note:** Scenario evaluation is **always disabled** for temporal split analysis (interpolation/extrapolation) regardless of this setting, as those splits already provide temporal generalization insights.

## Files Modified

1. `config/config.yaml`: Added `evaluation.enable_scenarios` parameter (default: false)
2. `src/training/base_trainer.py`: Reads config and passes to plot functions
3. `src/inference_testset.py`: Reads config and passes to plot functions
4. `src/viz/__init__.py`: Changed default from `True` to `False` in `plot_test_metrics()`

## Runtime Savings

Disabling scenario evaluation typically saves:
- **Training evaluation**: ~20-40% faster test evaluation per epoch
- **Inference**: ~30-50% faster overall runtime
- **Storage**: Fewer plots generated (saves disk space)

The exact savings depend on:
- Dataset size
- Number of scenarios with sufficient data
- Complexity of scenario-specific plots generated
