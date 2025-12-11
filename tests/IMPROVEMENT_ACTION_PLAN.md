# Action Plan: Fixing Generalization and Finetuning Issues

## Summary of Problems

### Issue 1: Pretrained Model - Poor Generalization
- High error variance, especially for high STEC values (std=16.5 for STEC 100-300 TECU)
- Systematic bias varies by STEC magnitude (not smooth)
- Uncertainty under-calibrated (only 62.7% within 1σ for DOY 122)

### Issue 2: Finetuning Strategy - Severe Overfitting
- **DOY 122**: Best validation at epoch 5, but trained to epoch 25 (20 epochs wasted!)
- **DOY 183**: Best validation at epoch 1, but trained to epoch 21 (20 epochs wasted!)
- No effective early stopping
- Learning rate too high (1e-4) for finetuning
- Wrong metric monitored (loss instead of bias)

---

## Immediate Actions (Quick Wins)

### 1. Fix Finetuning Strategy - HIGHEST PRIORITY

#### A. Implement Proper Early Stopping
```python
# In your finetuning script
from torch.utils.data import DataLoader

class EarlyStoppingWithBias:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_bias = float('inf')
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_bias, val_loss, model):
        # Monitor BIAS, not just loss
        if abs(val_bias) < self.best_bias - self.min_delta:
            self.best_bias = abs(val_bias)
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
            self.counter = 0
            return False  # Don't stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping! Best bias: {self.best_bias:.4f} at epoch {epoch - self.patience}")
                model.load_state_dict(self.best_weights)  # Restore best weights
                return True  # Stop training
        return False

# Usage in training loop
early_stopping = EarlyStoppingWithBias(patience=5)

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss, val_bias = validate_epoch(...)  # Return both loss and bias
    
    if early_stopping(val_bias, val_loss, model):
        break
```

#### B. Reduce Learning Rate for Finetuning
```python
# Current (TOO HIGH for finetuning)
optimizer = Adam(model.parameters(), lr=1e-4)

# Recommended (try these in order)
optimizer = Adam(model.parameters(), lr=1e-5)  # 10x smaller
# OR
optimizer = Adam(model.parameters(), lr=1e-6)  # 100x smaller

# Even better: use different LR for different layers
optimizer = Adam([
    {'params': model.encoder.parameters(), 'lr': 1e-6},  # freeze or tiny LR
    {'params': model.decoder.parameters(), 'lr': 1e-5},  # small LR
])
```

#### C. Shorter Finetuning Duration
```yaml
# In config.yaml
finetune:
  max_epochs: 10  # Down from 25-30
  early_stopping:
    enabled: true
    patience: 5
    monitor: 'val_bias'  # NOT 'val_loss'
    min_delta: 0.01
```

---

## Medium-Term Actions (1-2 weeks)

### 2. Improve Pretrained Model (Smoother Generalization)

#### A. Add More Regularization
```python
# In model definition
class BayesianResNetSTEC(nn.Module):
    def __init__(self, ...):
        # Current dropout (probably too low)
        self.dropout = nn.Dropout(p=0.1)  
        
        # Recommended: Increase dropout
        self.dropout = nn.Dropout(p=0.3)  # 3x higher
        
        # Add dropout to MORE layers
        self.dropout1 = nn.Dropout(p=0.3)  # After first layer
        self.dropout2 = nn.Dropout(p=0.2)  # After middle layers
        self.dropout3 = nn.Dropout(p=0.1)  # Before output

# In optimizer
optimizer = Adam(
    model.parameters(), 
    lr=1e-4,
    weight_decay=1e-4  # Add L2 regularization (currently missing?)
)
```

#### B. Add Smoothness Regularization to Loss
```python
def smooth_loss(predictions, targets, uncertainties, lambda_smooth=0.1):
    # Standard Gaussian NLL
    nll_loss = gaussian_nll_loss(predictions, targets, uncertainties)
    
    # Add smoothness penalty (penalize high variance predictions)
    # Group by station or temporal neighbors
    smooth_penalty = torch.var(predictions)  # Simple version
    
    # Or more sophisticated: penalize discontinuities
    # smooth_penalty = torch.mean((predictions[1:] - predictions[:-1])**2)
    
    total_loss = nll_loss + lambda_smooth * smooth_penalty
    return total_loss
```

#### C. Better Training Data Selection
```python
# Filter training data to exclude extreme variability days
# Focus on stable ionospheric conditions

def select_stable_training_days(data, threshold=40):
    """Select days with STEC std < threshold"""
    daily_stats = data.groupby('doy').agg({'stec': ['mean', 'std']})
    stable_days = daily_stats[daily_stats['stec']['std'] < threshold].index
    return data[data['doy'].isin(stable_days)]

# Use winter months preferentially (DOY 150-250 for Southern Hemisphere)
winter_mask = (data['doy'] >= 150) & (data['doy'] <= 250)
training_data = data[winter_mask]
```

---

## Long-Term Actions (1-2 months)

### 3. Validation Strategy Overhaul

#### A. Temporal Validation Split
```python
# Instead of random split, use temporal split
# Train on DOY 1-300, validate on DOY 301-365
# Or: Train on year N, validate on year N+1 same DOY range

def temporal_split(data, train_doys, val_doys):
    train_data = data[data['doy'].isin(train_doys)]
    val_data = data[data['doy'].isin(val_doys)]
    return train_data, val_data

# Example: Use DOY 183 (July) for validation
train_doys = list(range(1, 365))
train_doys.remove(183)  # Hold out DOY 183
val_doys = [183]
```

#### B. Spatial Validation Split
```python
# Train on some stations, validate on others
train_stations = ['NNOR', 'POAL', 'SOLO']
val_stations = ['CHPG']

train_data = data[data['station'].isin(train_stations)]
val_data = data[data['station'].isin(val_stations)]
```

#### C. Monitor Multiple Metrics During Training
```python
# Track these metrics on validation set
metrics_to_track = {
    'val_loss': gaussian_nll_loss,
    'val_bias': lambda pred, targ: (pred - targ).mean(),
    'val_mae': lambda pred, targ: torch.abs(pred - targ).mean(),
    'val_rmse': lambda pred, targ: torch.sqrt(torch.mean((pred - targ)**2)),
    'calibration_1sigma': lambda pred, targ, unc: (torch.abs(pred - targ) <= unc).float().mean(),
}

# Use bias as primary early stopping criterion
early_stopping = EarlyStoppingWithBias(
    monitor='val_bias',
    patience=5,
    mode='min_abs'  # Minimize absolute bias
)
```

---

## Verification Plan

### After implementing fixes, verify:

1. **Finetuning converges faster**
   - [ ] Training stops at epoch 5-10 (not 20+)
   - [ ] Validation bias improves or stays stable
   - [ ] Model uses best weights (from early stopping)

2. **Test performance improves**
   - [ ] Bias reduced by >50% on all stations
   - [ ] RMSE reduced by >20%
   - [ ] Positioning quality improves

3. **Pretrained model is smoother**
   - [ ] Error std decreases for high STEC values
   - [ ] Bias is more uniform across STEC ranges
   - [ ] Calibration within 1σ improves to >65%

---

## Expected Improvements

| Metric | Current (DOY 122) | Target | Current (DOY 183) | Target |
|--------|-------------------|---------|-------------------|---------|
| CHPG Bias | +2.79 TECU | <1.0 TECU | +1.47 TECU | <0.5 TECU |
| NNOR Bias | -2.11 TECU | <1.0 TECU | -0.47 TECU | <0.3 TECU |
| Training epochs | 25 | 5-10 | 21 | 5-10 |
| Val loss degradation | +12% | <5% | +21% | <5% |
| Calibration 1σ | 62.7% | >68% | 73.7% | >70% |

---

## Implementation Priority

1. **URGENT** (Do today):
   - [ ] Add bias tracking to validation
   - [ ] Reduce finetuning LR to 1e-5 or 1e-6
   - [ ] Implement early stopping with patience=5

2. **HIGH** (This week):
   - [ ] Increase dropout in pretrained model
   - [ ] Add weight decay (L2 regularization)
   - [ ] Retrain pretrained model with new regularization

3. **MEDIUM** (Next 2 weeks):
   - [ ] Implement temporal validation split
   - [ ] Add smoothness penalty to loss
   - [ ] Test on multiple DOYs to verify generalization

4. **LOW** (Future work):
   - [ ] Ensemble models from different training runs
   - [ ] Investigate architecture changes (fewer layers/units)
   - [ ] Explore alternative loss functions

---

## Files to Modify

1. `src/finetune.py` - Add early stopping with bias monitoring
2. `src/model/model.py` - Increase dropout, add regularization
3. `config/config.yaml` - Update hyperparameters (LR, max_epochs, patience)
4. `src/utils/metrics.py` - Add bias calculation function
5. `src/pretrain.py` - Add smoothness penalty to loss

---

## Questions to Answer

1. What's the current pretrained model's dropout rate?
2. Is weight decay (L2 regularization) currently enabled?
3. What's the current early stopping patience? (seems very high or disabled)
4. How is the validation set currently selected (random? temporal? spatial?)
5. What loss function is used? (Just GNLL or GNLL + other terms?)

Run these checks:
```bash
# Check current config
cat config/config.yaml | grep -A5 "dropout\|weight_decay\|early_stopping\|patience"

# Check pretrained model details
python -c "import torch; model = torch.load('path/to/pretrained.pth'); print(model)"
```
