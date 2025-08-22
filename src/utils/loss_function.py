import torch
import torch.nn as nn
import torchbnn as bnn
import numpy as np

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_function='linear', high_value_threshold=0.75, high_value_weight=3.0):
        super().__init__()
        self.weight_function = weight_function
        self.high_value_threshold = high_value_threshold
        self.high_value_weight = high_value_weight
    
    def forward(self, predictions, targets):
        # Calculate weights based on target magnitude
        if self.weight_function == 'linear':
            # Higher targets get proportionally higher weights
            weights = targets / targets.mean()
        elif self.weight_function == 'quadratic':
            # Quadratic scaling for even more emphasis on high values
            weights = (targets / targets.mean()) ** 2
        elif self.weight_function == 'log':
            # Logarithmic scaling for moderate emphasis
            weights = torch.log(1 + targets / targets.mean())
        elif self.weight_function == 'quantile':
            # Binary weighting: high values get fixed higher weight
            threshold = torch.quantile(targets, self.high_value_threshold)
            weights = torch.where(targets > threshold, self.high_value_weight, 1.0)
        else:
            weights = torch.ones_like(targets)
        
        # Apply weighted MSE
        weighted_errors = weights * (predictions - targets) ** 2
        return weighted_errors.mean()

class WeightedGaussianNLLLoss(nn.Module):
    def __init__(self, weight_function='linear', high_value_threshold=0.75, high_value_weight=3.0, eps=1e-6):
        super().__init__()
        self.weight_function = weight_function
        self.high_value_threshold = high_value_threshold
        self.high_value_weight = high_value_weight
        self.eps = eps
        self.base_loss = nn.GaussianNLLLoss(reduction='none')  # No reduction for element-wise weighting
    
    def forward(self, predictions, targets, variances):
        # Ensure variances are positive
        variances = torch.clamp(variances, min=self.eps)
        
        # Calculate weights based on target magnitude
        if self.weight_function == 'linear':
            weights = targets / (targets.mean() + self.eps)
        elif self.weight_function == 'quadratic':
            weights = (targets / (targets.mean() + self.eps)) ** 2
        elif self.weight_function == 'log':
            weights = torch.log(1 + targets / (targets.mean() + self.eps))
        elif self.weight_function == 'quantile':
            threshold = torch.quantile(targets, self.high_value_threshold)
            weights = torch.where(targets > threshold, self.high_value_weight, 1.0)
        else:
            weights = torch.ones_like(targets)
        
        # Apply base GaussianNLL loss element-wise
        base_losses = self.base_loss(predictions, targets, variances)
        
        # Apply weights and return mean
        weighted_losses = weights * base_losses
        return weighted_losses.mean()

class LaplaceLoss(nn.Module):
    def __init__(self):
        super(LaplaceLoss, self).__init__()
    
    def forward(self, outputs, y):
        mu, std = outputs[:, 0], outputs[:, 1]
        std = std.reshape(-1,) + 1e-6 
        loss = torch.sum(torch.log(2 * std) + torch.abs(y - mu) / std)
        return loss 

def get_criterion(config, loss_fn=None):
    loss_type = config['training']["loss_function"]
    if loss_fn is not None:
        loss_type = loss_fn

    # Get weighting configuration
    weighting_config = config['training'].get('target_weighting', {})
    use_weighting = weighting_config.get('enabled', False)
    weight_function = weighting_config.get('weight_function', 'linear')
    high_value_threshold = weighting_config.get('high_value_threshold', 0.75)
    high_value_weight = weighting_config.get('high_value_weight', 3.0)

    # Initialize the base loss function
    if loss_type == 'MSELoss':
        if use_weighting:
            criterion = WeightedMSELoss(
                weight_function=weight_function,
                high_value_threshold=high_value_threshold,
                high_value_weight=high_value_weight
            )
        else:
            criterion = nn.MSELoss()
    elif loss_type == 'MAELoss':
        criterion = nn.L1Loss() 
    elif loss_type == 'HuberLoss':
        criterion = nn.SmoothL1Loss() 
    elif loss_type == 'LaplaceLoss':
        criterion = LaplaceLoss()
    elif loss_type == 'GaussianNLLLoss':
        if use_weighting:
            criterion = WeightedGaussianNLLLoss(
                weight_function=weight_function,
                high_value_threshold=high_value_threshold,
                high_value_weight=high_value_weight
            )
        else:
            criterion = nn.GaussianNLLLoss()
    elif loss_type == 'BKLLoss':
        criterion = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion
