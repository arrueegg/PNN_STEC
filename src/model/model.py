import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch import nn

import torchbnn as bnn
from utils.feature_registry import FeatureType

# Initialization function
def init_xavier(model, activation, model_seed):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            gain = nn.init.calculate_gain(activation)
            torch.manual_seed(model_seed)  # Set the seed for reproducibility
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
            #torch.nn.init.xavier_uniform_(m.weight, gain=gain)  # Alternative (common choice for small nets)
            if m.bias is not None:
                m.bias.data.fill_(0)
    model.apply(init_weights)

def init_kaiming(model, activation, model_seed):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            nonlinearity = activation
            torch.manual_seed(model_seed)
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                m.bias.data.fill_(0)
    model.apply(init_weights)
    #nn.init.constant_(model.out_layer.bias, 100.0)



class MLP(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        self.layers.append(Linear(n_in, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        
        self.output_layer = Linear(hidden_dim, n_out)
        
        # FIXED: Initialize final layer to predict target mean (~15.5 TECU)
        with torch.no_grad():
            self.output_layer.bias.fill_(15.5)  # Initialize to approximate STEC mean
            self.output_layer.weight.normal_(0, 0.01)  # Small weights initially

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)

        return x, torch.zeros_like(x)  # Return zero variance for MLP
    
class BranchMLP(nn.Module):
    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()

        self.split = 3 + num_SWI_params  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

        # Spatial branch (lat, lon, etc.)
        spatial_layers = []
        spatial_layers.append(nn.Linear(n_in - self.split, hidden_dim))
        spatial_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            spatial_layers.append(nn.Linear(hidden_dim, hidden_dim))
            spatial_layers.append(nn.ReLU())
        self.spatial_net = nn.Sequential(*spatial_layers)
        
        # Temporal branch (sod, cos(doy), solar params, etc.)
        temporal_layers = []
        temporal_layers.append(nn.Linear(self.split, hidden_dim))
        temporal_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            temporal_layers.append(nn.Linear(hidden_dim, hidden_dim))
            temporal_layers.append(nn.ReLU())
        self.temporal_net = nn.Sequential(*temporal_layers)
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Predict STEC
        )
    
    def forward(self, x):
        temporal_features = x[:, :self.split]  # Temporal features (first self.split features)
        spatial_features = x[:, self.split:]   # Spatial features (remaining features)
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance
    
class MLP_NLL(torch.nn.Module):
    def __init__(self, n_in=3, hidden_dim=256, num_layers=2): 
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Linear(n_in, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.output_layer = Linear(hidden_dim, 2)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class MLP_MCDropout_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.layers.append(Linear(n_in, hidden_dim))
        self.dropouts.append(Dropout(p=0.2))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            # Use higher dropout for the last hidden layer
            dropout_p = 0.5 if i == num_layers - 2 else 0.2
            self.dropouts.append(Dropout(p=dropout_p))
        
        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        return x
    

class MLP_MCDropout_NLL(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        self.layers.append(Linear(n_in, hidden_dim))
        self.dropouts.append(Dropout(p=0.2))
        
        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            # Use higher dropout for the last hidden layer
            dropout_p = 0.5 if i == num_layers - 2 else 0.2
            self.dropouts.append(Dropout(p=dropout_p))
        
        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance
    
    
class BNN_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4):
        super().__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_in, out_features=hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim))
        
        # Output layer
        self.output_layer = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=n_out)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x
    
class BNN_NLL(torch.nn.Module):
    def __init__(self, n_in=3, hidden_dim=256, num_layers=4, prior_sigma=0.1):
        super().__init__()
        
        # Create layers dynamically
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=n_in, out_features=hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=hidden_dim))

        # Output layer (2 outputs for mean and variance)
        self.output_layer = bnn.BayesLinear(prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance 
    
class Branch_BNN_NLL(nn.Module):
    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()

        self.split = 6 + num_SWI_params  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

        # Spatial branch (lat, lon, etc.)
        spatial_layers = []
        spatial_layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_in - self.split, out_features=hidden_dim))
        spatial_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            spatial_layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim))
            spatial_layers.append(nn.ReLU())
        self.spatial_net = nn.Sequential(*spatial_layers)
        
        # Temporal branch (sod, cos(doy), solar params, etc.)
        temporal_layers = []
        temporal_layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.split, out_features=hidden_dim))
        temporal_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            temporal_layers.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=hidden_dim))
            temporal_layers.append(nn.ReLU())
        self.temporal_net = nn.Sequential(*temporal_layers)
        
        # Fusion and output
        self.fusion = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=2 * hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=2)  # Predict STEC
        )
    
    def forward(self, x):
        spatial_features = x[:, self.split:]  # Spatial features
        temporal_features = x[:, :self.split]  # Temporal features
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance
    

# Model selection function
def get_model(config):
    model_type = config['model']['model_type']
    hidden_dim = config['model'].get('hidden_dim', 256)  # Default to 256 if not specified
    num_layers = config['model'].get('num_layers', 4)    # Default to 4 if not specified
    prior_sigma = config['model'].get('prior_sigma', 0.1)  # Default prior sigma for BNNs
    
    # Get input features count from feature registry, accounting for transformations
    feature_registry = config.get('feature_registry')
    if not feature_registry:
        raise ValueError("Feature registry is required but not found in config")
    
    # Calculate transformed feature dimensions
    # Note: The collate function transforms features, so we need to account for this
    
    # Temporal features: year (1) + doy (3: sin, cos, norm) + sod (3: sin, cos, norm) = 7
    temporal_features = feature_registry.get_features_by_type(FeatureType.TEMPORAL)
    temporal_dim = 0
    for feature in temporal_features:
        if feature == 'year':
            temporal_dim += 1  # Just normalized year
        elif feature in ['doy', 'sod']:
            temporal_dim += 3  # sin, cos, normalized for each
    
    # Station features (only for STEC target)
    station_features = feature_registry.get_features_by_type(FeatureType.STATION)
    station_dim = len(station_features)  # No transformation applied
    
    # Direction features (only for STEC target) 
    direction_features = feature_registry.get_features_by_type(FeatureType.DIRECTION)
    direction_dim = 0
    for feature in direction_features:
        if feature == 'satazi':
            direction_dim += 3  # sin, cos, normalized for azimuth
        elif feature == 'satele':
            direction_dim += 1  # just normalized for elevation
        else:
            direction_dim += 1  # default: just normalized

    # IPP features
    ipp_features = feature_registry.get_features_by_type(FeatureType.IPP)
    ipp_dim = len(ipp_features)  # No transformation applied
    
    # SWI features
    swi_features = feature_registry.get_features_by_type(FeatureType.SWI)
    swi_dim = len(swi_features)  # No transformation applied
    num_SWI_params = swi_dim
    
    # SH embeddings (if enabled)
    sh_dim = 4 * config['data']['SH_degree']**2  # Currently 0 for SH_degree=0
    
    # Total input features after all transformations
    in_features = temporal_dim + station_dim + direction_dim + ipp_dim + swi_dim + sh_dim

    if model_type == 'MLP':
        return MLP(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'BranchMLP':
        return BranchMLP(n_in=in_features, num_SWI_params=num_SWI_params, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'MLP_NLL':
        model = MLP_NLL(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
        return model
    elif model_type == 'MLP_MCDropout_mse':
        return MLP_MCDropout_mse(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'MLP_MCDropout_NLL':
        model = MLP_MCDropout_NLL(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
        return model
    elif model_type == 'BNN_mse':
        return BNN_mse(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == 'BNN_NLL':
        model = BNN_NLL(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers, prior_sigma=prior_sigma)
        return model
    elif model_type == 'Branch_BNN_NLL':
        model = Branch_BNN_NLL(n_in=in_features, num_SWI_params=num_SWI_params, hidden_dim=hidden_dim, num_layers=num_layers)
        return model
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")

