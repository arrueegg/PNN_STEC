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
            # torch.nn.init.xavier_uniform_(m.weight, gain=gain)  # Alternative (common choice for small nets)
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
    # nn.init.constant_(model.out_layer.bias, 100.0)


# ============================================================================
# ResNet-Based Architectures with Skip Connections
# ============================================================================


class ResNetBlock(nn.Module):
    """Residual block for ResNet architecture"""
    def __init__(self, hidden_dim, dropout_rate=0.0):
        super().__init__()
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Residual connection: x + f(x)
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        if self.dropout:
            x = self.dropout(x)
        return x + residual


class ResNet_MSE(torch.nn.Module):
    """ResNet-based MLP with skip connections - MSE loss (deterministic prediction)"""
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4, dropout_rate=0.0):
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)
        
        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias.fill_(15.5)
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.output_layer(x)
        return x, torch.zeros_like(x)  # Return zero variance for MSE


class ResNet_NLL(torch.nn.Module):
    """ResNet-based MLP with skip connections - NLL loss (outputs mean + variance)"""
    def __init__(self, n_in=3, hidden_dim=256, num_layers=4, dropout_rate=0.0):
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)
        
        # Initialize output bias
        with torch.no_grad():
            self.output_layer.bias[0].fill_(15.5)  # Mean bias
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Ensure positive variance
        return mean, variance


# ============================================================================
# Attention-Based Architectures
# ============================================================================


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head self-attention with residual connection and layer normalization"""
    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        # x shape: (batch_size, 1, hidden_dim) - treat as sequence of length 1
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        if self.dropout:
            attn_out = self.dropout(attn_out)
        return attn_out + residual


class AttentionMLP_MSE(torch.nn.Module):
    """Attention-based MLP - MSE loss (deterministic prediction)"""
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4, num_heads=4, dropout_rate=0.0):
        super().__init__()
        
        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 4)  # Fallback to ~4x reduction
        
        # Input projection
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )
        
        # Alternating attention and MLP blocks
        self.attn_blocks = nn.ModuleList([
            MultiHeadAttentionBlock(hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)
        
        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias.fill_(15.5)
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        # Input: (batch_size, n_in)
        x = self.input_layer(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim) - sequence of length 1
        
        for attn_block, mlp_block, norm in zip(self.attn_blocks, self.mlp_blocks, self.norms):
            # Attention
            x = attn_block(x)
            
            # MLP with residual
            residual = x
            x = norm(x)
            x = mlp_block(x)
            x = x + residual
        
        # Remove sequence dimension and project to output
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        x = self.output_layer(x)
        
        return x, torch.zeros_like(x)  # Return zero variance for MSE


class AttentionMLP_NLL(torch.nn.Module):
    """Attention-based MLP - NLL loss (outputs mean + variance)"""
    def __init__(self, n_in=3, hidden_dim=256, num_layers=4, num_heads=4, dropout_rate=0.0):
        super().__init__()
        
        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 4)  # Fallback to ~4x reduction
        
        # Input projection
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )
        
        # Alternating attention and MLP blocks
        self.attn_blocks = nn.ModuleList([
            MultiHeadAttentionBlock(hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.mlp_blocks = nn.ModuleList([
            nn.Sequential(
                Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)
        
        # Initialize output bias
        with torch.no_grad():
            self.output_layer.bias[0].fill_(15.5)  # Mean bias
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        # Input: (batch_size, n_in)
        x = self.input_layer(x)  # (batch_size, hidden_dim)
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim) - sequence of length 1
        
        for attn_block, mlp_block, norm in zip(self.attn_blocks, self.mlp_blocks, self.norms):
            # Attention
            x = attn_block(x)
            
            # MLP with residual
            residual = x
            x = norm(x)
            x = mlp_block(x)
            x = x + residual
        
        # Remove sequence dimension and project to output
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        x = self.output_layer(x)
        
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + 1e-3  # Ensure positive variance
        return mean, variance


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

        self.split = (
            6 + num_SWI_params
        )  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

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
            nn.Linear(hidden_dim, 2),  # Predict STEC
        )

    def forward(self, x):
        temporal_features = x[
            :, : self.split
        ]  # Temporal features (first self.split features)
        spatial_features = x[:, self.split :]  # Spatial features (remaining features)
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = (
            F.softplus(variance) + 1e-3
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

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
        variance = (
            F.softplus(variance) + 1e-3
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class MLP_MCDropout_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4, dropout_rate=0.1):
        super().__init__()

        # Create layers dynamically
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer
        self.layers.append(Linear(n_in, hidden_dim))
        self.dropouts.append(Dropout(p=dropout_rate))

        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            # Use consistent dropout rate for all layers
            self.dropouts.append(Dropout(p=dropout_rate))

        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)

        # Flag to control Monte Carlo mode
        self.mc_mode = False

    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout for uncertainty estimation."""
        self.mc_mode = True
        for dropout in self.dropouts:
            dropout.train()  # Keep dropout active during inference

    def disable_mc_dropout(self):
        """Disable Monte Carlo dropout for standard inference."""
        self.mc_mode = False
        for dropout in self.dropouts:
            dropout.eval()  # Standard eval behavior

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        prediction = self.output_layer(x)

        # Return (prediction, zero_variance) to match expected format
        variance = torch.zeros_like(prediction)
        return prediction, variance


class MLP_MCDropout_NLL(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4, dropout_rate=0.1):
        super().__init__()

        # Create layers dynamically
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # First layer
        self.layers.append(Linear(n_in, hidden_dim))
        self.dropouts.append(Dropout(p=dropout_rate))

        # Hidden layers
        for i in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
            # Use consistent dropout rate for all layers
            self.dropouts.append(Dropout(p=dropout_rate))

        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)

        # Flag to control Monte Carlo mode
        self.mc_mode = False

    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout for uncertainty estimation."""
        self.mc_mode = True
        for dropout in self.dropouts:
            dropout.train()  # Keep dropout active during inference

    def disable_mc_dropout(self):
        """Disable Monte Carlo dropout for standard inference."""
        self.mc_mode = False
        for dropout in self.dropouts:
            dropout.eval()  # Standard eval behavior

    def forward(self, x):
        for layer, dropout in zip(self.layers, self.dropouts):
            x = F.relu(layer(x))
            x = dropout(x)
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = (
            F.softplus(variance) + 1e-3
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class BNN_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4):
        super().__init__()

        # Create layers dynamically
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=n_in, out_features=hidden_dim
            )
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=0.1,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )

        # Output layer
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=n_out
        )

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
        self.layers.append(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=prior_sigma,
                in_features=n_in,
                out_features=hidden_dim,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=prior_sigma,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )

        # Output layer (2 outputs for mean and variance)
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = (
            F.softplus(variance) + 1e-3
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class Branch_BNN_NLL(nn.Module):
    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()

        self.split = (
            6 + num_SWI_params
        )  # time features (sod normalized, cos(doy), sin(doy)) + SWI features

        # Spatial branch (lat, lon, etc.)
        spatial_layers = []
        spatial_layers.append(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=n_in - self.split,
                out_features=hidden_dim,
            )
        )
        spatial_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            spatial_layers.append(
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=0.1,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )
            spatial_layers.append(nn.ReLU())
        self.spatial_net = nn.Sequential(*spatial_layers)

        # Temporal branch (sod, cos(doy), solar params, etc.)
        temporal_layers = []
        temporal_layers.append(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=self.split,
                out_features=hidden_dim,
            )
        )
        temporal_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            temporal_layers.append(
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=0.1,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )
            temporal_layers.append(nn.ReLU())
        self.temporal_net = nn.Sequential(*temporal_layers)

        # Fusion and output
        self.fusion = nn.Sequential(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=0.1,
                in_features=2 * hidden_dim,
                out_features=hidden_dim,
            ),
            nn.ReLU(),
            bnn.BayesLinear(
                prior_mu=0, prior_sigma=0.1, in_features=hidden_dim, out_features=2
            ),  # Predict STEC
        )

    def forward(self, x):
        spatial_features = x[:, self.split :]  # Spatial features
        temporal_features = x[:, : self.split]  # Temporal features
        s_out = self.spatial_net(spatial_features)
        t_out = self.temporal_net(temporal_features)
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)

        mean, variance = torch.split(x, 1, dim=1)
        variance = (
            F.softplus(variance) + 1e-3
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class DeepEnsemble_MLP(torch.nn.Module):
    def __init__(self, n_in=3, hidden_dim=256, num_layers=4, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size

        # Create ensemble of MLP_NLL models
        self.ensemble_models = nn.ModuleList(
            [
                MLP_NLL(n_in=n_in, hidden_dim=hidden_dim, num_layers=num_layers)
                for _ in range(ensemble_size)
            ]
        )

        # Initialize each ensemble member differently for diversity
        self._initialize_ensemble_diversity()

    def _initialize_ensemble_diversity(self):
        """Initialize ensemble members with different seeds for diversity"""
        for i, model in enumerate(self.ensemble_models):
            # Use different seeds for each ensemble member
            torch.manual_seed(42 + i)  # Base seed + member index
            for param in model.parameters():
                if param.dim() > 1:  # Weights
                    nn.init.xavier_normal_(param)
                else:  # Biases
                    nn.init.constant_(param, 0.0)

            # Initialize output bias to STEC mean for each member
            with torch.no_grad():
                model.output_layer.bias.fill_(15.5)
                model.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass through ensemble.
        Returns aggregated mean and total uncertainty (aleatoric + epistemic).
        """
        predictions = []
        variances = []

        # Get predictions from all ensemble members
        for model in self.ensemble_models:
            mean, var = model(x)
            predictions.append(mean)
            variances.append(var)

        # Stack predictions: [ensemble_size, batch_size, 1]
        predictions = torch.stack(predictions, dim=0)
        variances = torch.stack(variances, dim=0)

        # Ensemble mean (epistemic uncertainty reduction)
        ensemble_mean = torch.mean(predictions, dim=0)

        # Aleatoric uncertainty (average of individual model uncertainties)
        aleatoric_uncertainty = torch.mean(variances, dim=0)

        # Epistemic uncertainty (variance of predictions across ensemble)
        epistemic_uncertainty = torch.var(predictions, dim=0, unbiased=True)

        # Total uncertainty = aleatoric + epistemic
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return ensemble_mean, total_uncertainty

    def get_uncertainties(self, x):
        """
        Get decomposed uncertainties for analysis.
        Returns: ensemble_mean, aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty
        """
        predictions = []
        variances = []

        for model in self.ensemble_models:
            mean, var = model(x)
            predictions.append(mean)
            variances.append(var)

        predictions = torch.stack(predictions, dim=0)
        variances = torch.stack(variances, dim=0)

        ensemble_mean = torch.mean(predictions, dim=0)
        aleatoric_uncertainty = torch.mean(variances, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0, unbiased=True)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            ensemble_mean,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            total_uncertainty,
        )


def get_model(config):
    model_type = config["model"]["model_type"]
    hidden_dim = config["model"].get(
        "hidden_dim", 256
    )  # Default to 256 if not specified
    num_layers = config["model"].get("num_layers", 4)  # Default to 4 if not specified
    prior_sigma = config["model"].get(
        "prior_sigma", 0.1
    )  # Default prior sigma for BNNs
    ensemble_size = config["model"].get("ensemble_size", 5)  # Default ensemble size
    dropout_rate = config["model"].get("dropout_rate", 0.1)  # Default dropout rate for MC Dropout

    # Get input features count from feature registry, accounting for transformations
    feature_registry = config.get("feature_registry")
    if not feature_registry:
        raise ValueError("Feature registry is required but not found in config")

    # Calculate transformed feature dimensions
    # Note: The collate function transforms features, so we need to account for this

    # Temporal features: year (1) + doy (3: sin, cos, norm) + sod (3: sin, cos, norm) + local_time_hours (3: sin, cos, norm) = 10
    temporal_features = feature_registry.get_features_by_type(FeatureType.TEMPORAL)
    temporal_dim = 0
    for feature in temporal_features:
        if feature == "year":
            temporal_dim += 1  # Just normalized year
        elif feature in ["doy", "sod", "local_time_hours"]:
            temporal_dim += 3  # sin, cos, normalized for each cyclical feature

    # Station features (only for STEC target)
    station_features = feature_registry.get_features_by_type(FeatureType.STATION)
    station_dim = len(station_features)  # No transformation applied

    # Direction features (only for STEC target) - Cartesian unit vector
    direction_features = feature_registry.get_features_by_type(FeatureType.DIRECTION)
    direction_dim = 0
    if direction_features:
        # Check if we have both azimuth and elevation for Cartesian transformation
        if "satazi" in direction_features and "satele" in direction_features:
            direction_dim = 3  # e_up, e_east, e_north
        else:
            # Fallback to individual processing
            direction_dim = len(direction_features)

    # IPP features
    ipp_features = feature_registry.get_features_by_type(FeatureType.IPP)
    ipp_dim = len(ipp_features)  # No transformation applied

    # SWI features
    swi_features = feature_registry.get_features_by_type(FeatureType.SWI)
    swi_dim = len(swi_features)  # No transformation applied
    num_SWI_params = swi_dim

    # SH embeddings (if enabled)
    sh_dim = 4 * config["data"]["SH_degree"] ** 2  # Currently 0 for SH_degree=0

    # Total input features after all transformations
    in_features = (
        temporal_dim + station_dim + direction_dim + ipp_dim + swi_dim + sh_dim
    )

    if model_type == "MLP":
        return MLP(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == "BranchMLP":
        return BranchMLP(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type == "MLP_NLL":
        model = MLP_NLL(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
        return model
    elif model_type == "DE_MLP":
        model = DeepEnsemble_MLP(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ensemble_size=ensemble_size,
        )
        return model
    elif model_type == "MLP_MCDropout_mse":
        return MLP_MCDropout_mse(
            n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate
        )
    elif model_type == "MLP_MCDropout_NLL":
        model = MLP_MCDropout_NLL(
            n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate
        )
        return model
    elif model_type == "BNN_mse":
        return BNN_mse(n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers)
    elif model_type == "BNN_NLL":
        model = BNN_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            prior_sigma=prior_sigma,
        )
        return model
    elif model_type == "Branch_BNN_NLL":
        model = Branch_BNN_NLL(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return model
    elif model_type == "ResNet_MSE":
        return ResNet_MSE(
            n_in=in_features, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
    elif model_type == "ResNet_NLL":
        return ResNet_NLL(
            n_in=in_features, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            dropout_rate=dropout_rate
        )
    elif model_type == "AttentionMLP_MSE":
        return AttentionMLP_MSE(
            n_in=in_features, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            num_heads=config["model"].get("num_heads", 4),
            dropout_rate=dropout_rate
        )
    elif model_type == "AttentionMLP_NLL":
        return AttentionMLP_NLL(
            n_in=in_features, 
            hidden_dim=hidden_dim, 
            num_layers=num_layers,
            num_heads=config["model"].get("num_heads", 4),
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")
