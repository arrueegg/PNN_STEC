import torch
import torch.nn as nn
import torchbnn as bnn
import numpy as np

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

    # Initialize the base loss function
    if loss_type == 'MSELoss':
        criterion = nn.MSELoss() 
    elif loss_type == 'MAELoss':
        criterion = nn.L1Loss() 
    elif loss_type == 'HuberLoss':
        criterion = nn.SmoothL1Loss() 
    elif loss_type == 'LaplaceLoss':
        criterion = LaplaceLoss()
    elif loss_type == 'GaussianNLLLoss':
        criterion = nn.GaussianNLLLoss()
    elif loss_type == 'BKLLoss':
        criterion = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    else:
        raise Exception(f'unknown loss {loss_type}')

    return criterion
