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

        return x
    
class MLP_DE(torch.nn.Module):
    def __init__(self, n_in=3, n_out=2):
        super().__init__()
        self.Linear_1 = Linear(n_in, 256)
        self.Linear_2 = Linear(256, 256)
        self.Linear_3 = Linear(256, 256)
        self.Linear_4 = Linear(256, 256)
        self.Linear_5 = Linear(256, 2)

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
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-6 #Positive constraint

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
        variance = F.softplus(variance) + 1e-6 #Positive constraint

        return mean, variance
    

# Model selection function
def get_model(config):
    model_type = config['model']['model_type']
    in_features = 6 + 4 + 2 * config['data']['SH_degree']**2 # 6 azi/ele + 4 sta/ipp coords + SH embeddings
    if config['data']['use_SWI']:
        in_features += 22  # Add SWI features
    if model_type == 'MLP':
        return MLP(n_in=in_features)
    elif model_type == 'MLP_DE':
        return MLP_DE(n_in=in_features)
    elif model_type == 'MLP_MCDropout_mse':
        return MLP_MCDropout_mse(n_in=in_features)
    elif model_type == 'MLP_MCDropout_NLL':
        return MLP_MCDropout_NLL(n_in=in_features)
    elif model_type == 'BNN_mse':
        return BNN_mse(n_in=in_features)
    elif model_type == 'BNN_NLL':
        return BNN_NLL(n_in=in_features)
    else:
        raise ValueError(f"Model type {model_type} is not recognized. Please select from ['MLP', 'RNN']")

