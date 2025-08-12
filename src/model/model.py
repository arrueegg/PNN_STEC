import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch import nn

import torchbnn as bnn

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
    def __init__(self, n_in=3, n_out=1):
        super().__init__()
        self.Linear_1 = Linear(n_in, 256)
        self.Linear_2 = Linear(256, 256)
        self.Linear_3 = Linear(256, 256)
        self.Linear_4 = Linear(256, 256)
        self.Linear_5 = Linear(256, 1)

    def forward(self, x):
        x = self.Linear_1(x)
        x = F.relu(x)
        x = self.Linear_2(x)
        x = F.relu(x)
        x = self.Linear_3(x)
        x = F.relu(x)
        x = self.Linear_4(x)
        x = F.relu(x)
        x = self.Linear_5(x)

        return x, 0
    
class BranchMLP(nn.Module):
    def __init__(self, n_in, num_SWI_params):
        super().__init__()

        self.split = 3 + num_SWI_params  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

        # Spatial branch (lat, lon, etc.)
        self.spatial_net = nn.Sequential(
            nn.Linear(n_in - self.split, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Temporal branch (sod, cos(doy), solar params, etc.)
        self.temporal_net = nn.Sequential(
            nn.Linear(self.split, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            nn.Linear(2 * 256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)  # Predict STEC
        )
    
    def forward(self, x):
        spatial_features = x[:, :self.split]  # Spatial features
        temporal_features = x[:, self.split:]  # Temporal features
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint

        return mean, variance
    
class MLP_NLL(torch.nn.Module):
    def __init__(self, n_in=3, hidden_dim=256, num_layers=2):  # FIXED: Shallow model
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Linear(n_in, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.output_layer = Linear(hidden_dim, 2)
        
        # FIXED: Add Kaiming initialization
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6  # Positive constraint

        return mean, variance


class MLP_MCDropout_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1):
        super().__init__()
        self.Linear_1 = Linear(n_in, 256)
        self.Dropout_1 = Dropout(p=0.2)
        self.Linear_2 = Linear(256, 256)
        self.Dropout_2 = Dropout(p=0.2)
        self.Linear_3 = Linear(256, 256)
        self.Dropout_3 = Dropout(p=0.2)
        self.Linear_4 = Linear(256, 256)
        self.Dropout_4 = Dropout(p=0.5)
        self.Linear_5 = Linear(256, 1)

    def forward(self, x):
        x = self.Linear_1(x)
        x = F.relu(x)
        x = self.Dropout_1(x)
        x = self.Linear_2(x)
        x = F.relu(x)
        x = self.Dropout_2(x)
        x = self.Linear_3(x)
        x = F.relu(x)
        x = self.Dropout_3(x)
        x = self.Linear_4(x)
        x = F.relu(x)
        x = self.Dropout_4(x)
        x = self.Linear_5(x)

        return x
    

class MLP_MCDropout_NLL(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1):
        super().__init__()
        self.Linear_1 = Linear(n_in, 256)
        self.Dropout_1 = Dropout(p=0.2)
        self.Linear_2 = Linear(256, 256)
        self.Dropout_2 = Dropout(p=0.2)
        self.Linear_3 = Linear(256, 256)
        self.Dropout_3 = Dropout(p=0.2)
        self.Linear_4 = Linear(256, 256)
        self.Dropout_4 = Dropout(p=0.5)
        self.Linear_5 = Linear(256, 2)

    def forward(self, x):
        x = self.Linear_1(x)
        x = F.relu(x)
        x = self.Dropout_1(x)
        x = self.Linear_2(x)
        x = F.relu(x)
        x = self.Dropout_2(x)
        x = self.Linear_3(x)
        x = F.relu(x)
        x = self.Dropout_3(x)
        x = self.Linear_4(x)
        x = F.relu(x)
        x = self.Dropout_4(x)
        x = self.Linear_5(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint

        return mean, variance
    
    
class BNN_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1):
        super().__init__()
        self.BNN_1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_in,
                                     out_features=256)
        self.BNN_2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_4 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_5 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=1)

    def forward(self, x):
        x = self.BNN_1(x)
        x = F.relu(x)
        x = self.BNN_2(x)
        x = F.relu(x)
        x = self.BNN_3(x)
        x = F.relu(x)
        x = self.BNN_4(x)
        x = F.relu(x)
        x = self.BNN_5(x)

        return x
    
class BNN_NLL(torch.nn.Module):
    def __init__(self, n_in=3):
        super().__init__()
        self.BNN_1 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_in,
                                     out_features=256)
        self.BNN_2 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_3 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_4 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=256)
        self.BNN_5 = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256,
                                     out_features=2)

    def forward(self, x):
        x = self.BNN_1(x)
        x = F.relu(x)
        x = self.BNN_2(x)
        x = F.relu(x)
        x = self.BNN_3(x)
        x = F.relu(x)
        x = self.BNN_4(x)
        x = F.relu(x)
        x = self.BNN_5(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6  # Positive constraint

        return mean, variance
    
class Branch_BNN_NLL(nn.Module):
    def __init__(self, n_in, num_SWI_params):
        super().__init__()

        self.split = 6 + num_SWI_params  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

        # Spatial branch (lat, lon, etc.)
        self.spatial_net = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_in - self.split, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=256),
            nn.ReLU()
        )
        
        # Temporal branch (sod, cos(doy), solar params, etc.)
        self.temporal_net = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=self.split, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=256),
            nn.ReLU()
        )
        
        # Fusion and output
        self.fusion = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=2 * 256, out_features=256),
            nn.ReLU(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=256, out_features=2)  # Predict STEC
        )
    
    def forward(self, x):
        spatial_features = x[:, self.split:]  # Spatial features
        temporal_features = x[:, :self.split]  # Temporal features
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6  # Positive constraint

        return mean, variance
    

# Model selection function
def get_model(config):
    model_type = config['model']['model_type']
    in_features = 7 + 4 + 8 + 4 * config['data']['SH_degree']**2 # 7 time&doy&year + 4 azi/ele + 4 sta/ipp coords + SH embeddings
    if config['data']['use_SWI']:
        num_SWI_params = 22
        in_features += num_SWI_params  # Add SWI features
    else:
        num_SWI_params = 0

    if model_type == 'MLP':
        return MLP(n_in=in_features)
    elif model_type == 'BranchMLP':
        return BranchMLP(n_in=in_features, num_SWI_params=num_SWI_params)
    elif model_type == 'MLP_NLL':
        return MLP_NLL(n_in=in_features)
    elif model_type == 'MLP_MCDropout_mse':
        return MLP_MCDropout_mse(n_in=in_features)
    elif model_type == 'MLP_MCDropout_NLL':
        return MLP_MCDropout_NLL(n_in=in_features)
    elif model_type == 'BNN_mse':
        return BNN_mse(n_in=in_features)
    elif model_type == 'BNN_NLL':
        return BNN_NLL(n_in=in_features)
    elif model_type == 'Branch_BNN_NLL':
        return Branch_BNN_NLL(n_in=in_features, num_SWI_params=num_SWI_params)
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")

