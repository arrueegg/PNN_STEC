from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch import nn

import torchbnn as bnn
from utils.feature_registry import FeatureType

# Domain constants
STEC_MEAN_TECU = (
    15.5  # Approximate mean STEC in TECU; used for output bias initialization
)
VARIANCE_FLOOR = 1e-3  # Minimum variance to prevent degenerate NLL loss


# Initialization function
def init_xavier(model, activation, model_seed):
    def init_weights(m):
        if isinstance(m, nn.Linear) and m.weight.requires_grad:
            gain = nn.init.calculate_gain(activation)
            torch.manual_seed(model_seed)  # Set the seed for reproducibility
            torch.nn.init.xavier_normal_(m.weight, gain=gain)
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
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)

        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias.fill_(STEC_MEAN_TECU)
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
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)

        # Initialize output bias
        with torch.no_grad():
            self.output_layer.bias[0].fill_(STEC_MEAN_TECU)  # Mean bias
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR  # Ensure positive variance
        return mean, variance


class BayesResNetBlock(nn.Module):
    """Bayesian Residual block for ResNet architecture"""

    def __init__(self, hidden_dim, dropout_rate=0.0, prior_sigma=0.1):
        super().__init__()
        self.fc1 = bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=prior_sigma,
            in_features=hidden_dim,
            out_features=hidden_dim,
        )
        self.fc2 = bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=prior_sigma,
            in_features=hidden_dim,
            out_features=hidden_dim,
        )
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


class ResNet_BNN_NLL(torch.nn.Module):
    """Bayesian ResNet-based MLP with skip connections - NLL loss (outputs mean + variance)"""

    def __init__(
        self, n_in=3, hidden_dim=256, num_layers=4, dropout_rate=0.0, prior_sigma=0.1
    ):
        super().__init__()

        # Input projection (keep standard Linear for input, or make Bayesian if desired)
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )

        # Bayesian Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                BayesResNetBlock(
                    hidden_dim, dropout_rate=dropout_rate, prior_sigma=prior_sigma
                )
                for _ in range(num_layers)
            ]
        )

        # Output layer (2 outputs for mean and variance)
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )

        # Match BayesianResNetSTEC's output initialisation exactly. Without this the two
        # architectures are not a clean ablation: the last-layer variant starts predicting
        # roughly the mean STEC and learns corrections, while this one has to discover the
        # mean through four layers that resample weights on every forward pass. The
        # signature of that difference is a pervasive negative bias - measured at -1.93 TECU
        # overall against +0.43 for the last-layer model - which confounds "Bayesian
        # residual blocks" with "no output initialisation".
        with torch.no_grad():
            self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)  # Mean bias
            self.output_layer.weight_mu.normal_(0, 0.01)

    def forward(self, x):
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.output_layer(x)
        mean, log_var = x.chunk(2, dim=-1)
        variance = F.softplus(log_var) + VARIANCE_FLOOR
        return mean, variance


class BayesianResNetSTEC(torch.nn.Module):
    """Hybrid Bayesian-ResNet for STEC regression with uncertainty quantification.

    Architecture:
    - Deterministic ResNet backbone (residual blocks with standard linear layers)
    - Bayesian output head using torchbnn.BayesLinear

    This design combines the expressiveness of ResNet with principled Bayesian
    uncertainty estimation. The deterministic backbone ensures computational efficiency
    while the Bayesian head captures predictive uncertainty.

    Output: (mean, variance) tuple following repo convention
    """

    def __init__(
        self, n_in=3, hidden_dim=256, num_layers=4, dropout_rate=0.0, prior_sigma=0.1
    ):
        super().__init__()

        # Input projection (deterministic)
        self.input_layer = nn.Sequential(
            Linear(n_in, hidden_dim),
            nn.ReLU(),
        )

        # Deterministic residual blocks (same as ResNet_NLL)
        self.res_blocks = nn.ModuleList(
            [
                ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
                for _ in range(num_layers)
            ]
        )

        # Bayesian output head: outputs (mean, variance) for STEC prediction
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )

        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)  # Mean bias
            self.output_layer.weight_mu.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass through ResNet backbone + Bayesian head.

        Args:
            x: Input features of shape (batch_size, n_in)

        Returns:
            mean: Predicted STEC of shape (batch_size, 1)
            variance: Predicted uncertainty of shape (batch_size, 1)
        """
        # Deterministic backbone
        x = self.input_layer(x)
        for res_block in self.res_blocks:
            x = res_block(x)

        # Bayesian head
        x = self.output_layer(x)  # Shape: (batch_size, 2)
        mean, log_var = torch.split(x, 1, dim=1)

        # Ensure positive variance
        variance = F.softplus(log_var) + VARIANCE_FLOOR

        return mean, variance


# ============================================================================
# Attention-Based Architectures
# ============================================================================


class FeatureAttentionBlock(nn.Module):
    """Multi-head self-attention over feature dimensions (each feature is a token)"""

    def __init__(self, hidden_dim, num_heads=4, dropout_rate=0.0):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim) where seq_len > 1 (multiple feature tokens)
        residual = x
        x_norm = self.norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        if self.dropout:
            attn_out = self.dropout(attn_out)
        return attn_out + residual


class AttentionMLP_MSE(torch.nn.Module):
    """Attention-based MLP with feature-level attention - MSE loss (deterministic prediction)

    Groups input features into tokens and learns their interactions via multi-head attention.
    """

    def __init__(
        self,
        n_in=3,
        n_out=1,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.0,
    ):
        super().__init__()

        # Feature grouping: create tokens from feature groups
        # Divide input into ~8-16 feature tokens for meaningful attention
        self.num_feature_tokens = max(
            2, min(16, n_in // 2)
        )  # Adaptive: 2-16 tokens based on n_in
        self.feature_embed_dim = (
            hidden_dim  # Each feature token has dimension hidden_dim
        )

        # Project raw features to feature tokens
        self.feature_projection = Linear(n_in, self.num_feature_tokens * hidden_dim)

        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 4)  # Fallback to ~4x reduction

        # Attention blocks: attend over feature tokens
        self.attn_blocks = nn.ModuleList(
            [
                FeatureAttentionBlock(
                    hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

        # FFN blocks after attention
        self.mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Pool feature tokens to single representation
        self.pool_layer = nn.AdaptiveAvgPool1d(1)  # Pool over sequence dimension

        # Output layer
        self.output_layer = Linear(hidden_dim, n_out)

        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias.fill_(STEC_MEAN_TECU)
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        # Input: (batch_size, n_in)
        batch_size = x.shape[0]

        # Project to feature tokens: (batch_size, n_in) -> (batch_size, num_tokens, hidden_dim)
        x = self.feature_projection(x)  # (batch_size, num_tokens * hidden_dim)
        x = x.view(
            batch_size, self.num_feature_tokens, -1
        )  # (batch_size, num_tokens, hidden_dim)

        # Apply attention and FFN layers
        for attn_block, mlp_block, norm in zip(
            self.attn_blocks, self.mlp_blocks, self.norms
        ):
            # Attention over feature tokens
            x = attn_block(x)  # (batch_size, num_tokens, hidden_dim)

            # FFN with residual
            residual = x
            x = norm(x)
            x = mlp_block(x)  # (batch_size, num_tokens, hidden_dim)
            x = x + residual

        # Pool tokens to single representation: (batch_size, num_tokens, hidden_dim) -> (batch_size, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, num_tokens)
        x = self.pool_layer(x).squeeze(-1)  # (batch_size, hidden_dim)

        # Project to output
        x = self.output_layer(x)

        return x, torch.zeros_like(x)  # Return zero variance for MSE


class AttentionMLP_NLL(torch.nn.Module):
    """Attention-based MLP with feature-level attention - NLL loss (outputs mean + variance)

    Groups input features into tokens and learns their interactions via multi-head attention.
    """

    def __init__(
        self, n_in=3, hidden_dim=256, num_layers=4, num_heads=4, dropout_rate=0.0
    ):
        super().__init__()

        # Feature grouping: create tokens from feature groups
        # Divide input into ~8-16 feature tokens for meaningful attention
        self.num_feature_tokens = max(
            2, min(16, n_in // 2)
        )  # Adaptive: 2-16 tokens based on n_in
        self.feature_embed_dim = (
            hidden_dim  # Each feature token has dimension hidden_dim
        )

        # Project raw features to feature tokens
        self.feature_projection = Linear(n_in, self.num_feature_tokens * hidden_dim)

        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 4)  # Fallback to ~4x reduction

        # Attention blocks: attend over feature tokens
        self.attn_blocks = nn.ModuleList(
            [
                FeatureAttentionBlock(
                    hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

        # FFN blocks after attention
        self.mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Pool feature tokens to single representation
        self.pool_layer = nn.AdaptiveAvgPool1d(1)  # Pool over sequence dimension

        # Output layer (2 outputs for mean and variance)
        self.output_layer = Linear(hidden_dim, 2)

        # Initialize output bias
        with torch.no_grad():
            self.output_layer.bias[0].fill_(STEC_MEAN_TECU)  # Mean bias
            self.output_layer.weight.normal_(0, 0.01)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.feature_projection(x)
        x = x.view(batch_size, self.num_feature_tokens, -1)
        for attn_block, mlp_block, norm in zip(
            self.attn_blocks, self.mlp_blocks, self.norms
        ):
            x = attn_block(x)
            residual = x
            x = norm(x)
            x = mlp_block(x)
            x = x + residual
        x = x.transpose(1, 2)
        x = self.pool_layer(x).squeeze(-1)
        x = self.output_layer(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR
        return mean, variance


class AttentionMLP_BNN_NLL(torch.nn.Module):
    """Attention-based MLP with Bayesian output layer - NLL loss with uncertainty quantification

    Architecture:
    - Deterministic attention backbone (feature-level multi-head attention)
    - Bayesian output head using torchbnn.BayesLinear for principled uncertainty estimation

    This combines the expressiveness of attention mechanisms with Bayesian uncertainty.
    The deterministic backbone ensures computational efficiency while the Bayesian head
    captures predictive uncertainty through weight posteriors.

    Output: (mean, variance) tuple for NLL loss
    """

    def __init__(
        self,
        n_in=3,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout_rate=0.0,
        prior_sigma=0.1,
    ):
        super().__init__()

        # Feature grouping: create tokens from feature groups
        # Divide input into ~8-16 feature tokens for meaningful attention
        self.num_feature_tokens = max(
            2, min(16, n_in // 2)
        )  # Adaptive: 2-16 tokens based on n_in
        self.feature_embed_dim = (
            hidden_dim  # Each feature token has dimension hidden_dim
        )

        # Project raw features to feature tokens (deterministic)
        self.feature_projection = Linear(n_in, self.num_feature_tokens * hidden_dim)

        # Ensure num_heads divides hidden_dim
        if hidden_dim % num_heads != 0:
            num_heads = max(1, hidden_dim // 4)  # Fallback to ~4x reduction

        # Attention blocks: attend over feature tokens (deterministic)
        self.attn_blocks = nn.ModuleList(
            [
                FeatureAttentionBlock(
                    hidden_dim, num_heads=num_heads, dropout_rate=dropout_rate
                )
                for _ in range(num_layers)
            ]
        )

        # FFN blocks after attention (deterministic)
        self.mlp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
                    Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Pool feature tokens to single representation
        self.pool_layer = nn.AdaptiveAvgPool1d(1)  # Pool over sequence dimension

        # Bayesian output layer: outputs (mean, variance) for STEC prediction with uncertainty
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )

        # Initialize output bias to STEC mean
        with torch.no_grad():
            self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)  # Mean bias
            self.output_layer.weight_mu.normal_(0, 0.01)

    def forward(self, x):
        """Forward pass through attention backbone + Bayesian head.

        Args:
            x: Input features of shape (batch_size, n_in)

        Returns:
            mean: Predicted STEC of shape (batch_size, 1)
            variance: Predicted uncertainty of shape (batch_size, 1)
        """
        # Input: (batch_size, n_in)
        batch_size = x.shape[0]

        # Project to feature tokens: (batch_size, n_in) -> (batch_size, num_tokens, hidden_dim)
        x = self.feature_projection(x)  # (batch_size, num_tokens * hidden_dim)
        x = x.view(
            batch_size, self.num_feature_tokens, -1
        )  # (batch_size, num_tokens, hidden_dim)

        # Apply deterministic attention and FFN layers
        for attn_block, mlp_block, norm in zip(
            self.attn_blocks, self.mlp_blocks, self.norms
        ):
            # Attention over feature tokens
            x = attn_block(x)  # (batch_size, num_tokens, hidden_dim)

            # FFN with residual
            residual = x
            x = norm(x)
            x = mlp_block(x)  # (batch_size, num_tokens, hidden_dim)
            x = x + residual

        # Pool tokens to single representation: (batch_size, num_tokens, hidden_dim) -> (batch_size, hidden_dim)
        x = x.transpose(1, 2)  # (batch_size, hidden_dim, num_tokens)
        x = self.pool_layer(x).squeeze(-1)  # (batch_size, hidden_dim)

        # Bayesian output head - samples from weight posterior
        x = self.output_layer(x)  # Shape: (batch_size, 2)
        mean, log_var = torch.split(x, 1, dim=1)

        # Ensure positive variance
        variance = F.softplus(log_var) + VARIANCE_FLOOR

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

        # FIXED: Initialize final layer to predict target mean (~STEC_MEAN_TECU TECU)
        with torch.no_grad():
            self.output_layer.bias.fill_(
                STEC_MEAN_TECU
            )  # Initialize to approximate STEC mean
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
            F.softplus(variance) + VARIANCE_FLOOR
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
            F.softplus(variance) + VARIANCE_FLOOR
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class MLP_LaplacianNLL(torch.nn.Module):
    """
    [PAPER] Mao et al. 2025: MLP for daily VTEC with Laplacian distribution.

    Architecture: 3 hidden layers × 90 neurons with tanh activation
    Output: (location, scale) for Laplacian distribution
    Uses existing repo collate function and feature pipeline.
    """

    def __init__(self, n_in=259, hidden_dim=90, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(Linear(n_in, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(Linear(hidden_dim, hidden_dim))
        self.output_layer = Linear(hidden_dim, 2)

    def forward(self, x):
        # [PAPER] tanh activation in hidden layers
        for layer in self.layers:
            x = torch.tanh(layer(x))
        x = self.output_layer(x)
        location, log_scale = torch.split(x, 1, dim=1)
        # [PAPER] Ensure scale > 0 using softplus
        scale = F.softplus(log_scale) + VARIANCE_FLOOR

        return location, scale


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
            F.softplus(variance) + VARIANCE_FLOOR
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class BNN_mse(torch.nn.Module):
    def __init__(self, n_in=3, n_out=1, hidden_dim=256, num_layers=4, prior_sigma=0.1):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=prior_sigma,
                in_features=n_in,
                out_features=hidden_dim,
            )
        )
        for _ in range(num_layers - 1):
            self.layers.append(
                bnn.BayesLinear(
                    prior_mu=0,
                    prior_sigma=prior_sigma,
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )
        self.output_layer = bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=prior_sigma,
            in_features=hidden_dim,
            out_features=n_out,
        )

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x, torch.zeros_like(x)


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
            F.softplus(variance) + VARIANCE_FLOOR
        )  # Increased minimum variance to prevent negative GaussianNLLLoss

        return mean, variance


class Branch_MLP_MSE(nn.Module):
    """2-branch MLP (Temporal vs Spatial) with MSE loss"""

    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()
        self.split = 6 + num_SWI_params  # Temporal features

        # Spatial branch
        spatial_layers = [Linear(n_in - self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        # Temporal branch
        temporal_layers = [Linear(self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        # Fusion
        self.fusion = nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            self.fusion[-1].bias.fill_(STEC_MEAN_TECU)

    def forward(self, x):
        s_out = self.spatial_net(x[:, self.split :])
        t_out = self.temporal_net(x[:, : self.split])
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)
        return x, torch.zeros_like(x)


class Branch_MLP_NLL(nn.Module):
    """2-branch MLP (Temporal vs Spatial) with NLL loss"""

    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()
        self.split = 6 + num_SWI_params  # Temporal features

        # Spatial branch
        spatial_layers = [Linear(n_in - self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        # Temporal branch
        temporal_layers = [Linear(self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        # Fusion
        self.fusion = nn.Sequential(
            Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.fusion[-1].bias[0].fill_(STEC_MEAN_TECU)

    def forward(self, x):
        s_out = self.spatial_net(x[:, self.split :])
        t_out = self.temporal_net(x[:, : self.split])
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR
        return mean, variance


class Branch_BNN_NLL(nn.Module):
    """2-branch BNN (Temporal vs Spatial) with NLL loss"""

    def __init__(
        self, n_in, num_SWI_params, hidden_dim=256, num_layers=2, prior_sigma=0.1
    ):
        super().__init__()
        self.split = 6 + num_SWI_params

        def _bayes(in_f, out_f):
            return bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=prior_sigma,
                in_features=in_f,
                out_features=out_f,
            )

        spatial_layers = [_bayes(n_in - self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([_bayes(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        temporal_layers = [_bayes(self.split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([_bayes(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        self.fusion = nn.Sequential(
            _bayes(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            _bayes(hidden_dim, 2),
        )

    def forward(self, x):
        s_out = self.spatial_net(x[:, self.split :])
        t_out = self.temporal_net(x[:, : self.split])
        x = torch.cat([s_out, t_out], dim=-1)
        x = self.fusion(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR
        return mean, variance


class Branch3Way_MLP_MSE(nn.Module):
    """3-branch MLP (Spatial, Temporal, Weather) with MSE loss - OPTIMIZED"""

    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()

        # Feature indices based on feature registry order:
        # Temporal: year(1) + doy(3) + sod(3) + local_time(3) = 10 features
        # Station: sm_lat(1) + sm_lon(1) + lat(1) + lon(1) = 4 features
        # IPP: lat(1) + lon(1) + sm_lat(1) + sm_lon(1) = 4 features
        # Direction: satazi(1) + satele(1) -> cartesian(3) = 3 features
        # SWI: num_SWI_params

        self.temporal_split = 10  # Temporal features
        self.spatial_split = (
            self.temporal_split + 4 + 4 + 3
        )  # Temporal + Station + IPP + Direction
        self.swi_start = self.spatial_split
        self.swi_dim = num_SWI_params

        # Branch 1: Spatial (receiver location + signal geometry)
        spatial_features = self.spatial_split - self.temporal_split
        spatial_layers = [Linear(spatial_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        # Branch 2: Temporal (local time effects)
        temporal_layers = [Linear(self.temporal_split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        # Branch 3: Weather (space weather indices)
        weather_layers = [Linear(self.swi_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            weather_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.weather_net = nn.Sequential(*weather_layers)

        # Fusion
        self.fusion = nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            self.fusion[-1].bias.fill_(STEC_MEAN_TECU)

    def forward(self, x):
        spatial_out = self.spatial_net(x[:, self.temporal_split : self.spatial_split])
        temporal_out = self.temporal_net(x[:, : self.temporal_split])
        weather_out = self.weather_net(x[:, self.swi_start :])
        x = torch.cat([spatial_out, temporal_out, weather_out], dim=-1)
        x = self.fusion(x)
        return x, torch.zeros_like(x)


class Branch3Way_MLP_NLL(nn.Module):
    """3-branch MLP (Spatial, Temporal, Weather) with NLL loss - OPTIMIZED"""

    def __init__(self, n_in, num_SWI_params, hidden_dim=256, num_layers=2):
        super().__init__()

        self.temporal_split = 10
        self.spatial_split = self.temporal_split + 4 + 4 + 3
        self.swi_start = self.spatial_split
        self.swi_dim = num_SWI_params

        # Branch 1: Spatial
        spatial_features = self.spatial_split - self.temporal_split
        spatial_layers = [Linear(spatial_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        # Branch 2: Temporal
        temporal_layers = [Linear(self.temporal_split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        # Branch 3: Weather
        weather_layers = [Linear(self.swi_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            weather_layers.extend([Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.weather_net = nn.Sequential(*weather_layers)

        # Fusion
        self.fusion = nn.Sequential(
            Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 2),
        )
        with torch.no_grad():
            self.fusion[-1].bias[0].fill_(STEC_MEAN_TECU)

    def forward(self, x):
        spatial_out = self.spatial_net(x[:, self.temporal_split : self.spatial_split])
        temporal_out = self.temporal_net(x[:, : self.temporal_split])
        weather_out = self.weather_net(x[:, self.swi_start :])
        x = torch.cat([spatial_out, temporal_out, weather_out], dim=-1)
        x = self.fusion(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR
        return mean, variance


class Branch3Way_BNN_NLL(nn.Module):
    """3-branch BNN (Spatial, Temporal, Weather) with NLL loss - OPTIMIZED"""

    def __init__(
        self, n_in, num_SWI_params, hidden_dim=256, num_layers=2, prior_sigma=0.1
    ):
        super().__init__()

        self.temporal_split = 10
        self.spatial_split = self.temporal_split + 4 + 4 + 3
        self.swi_start = self.spatial_split
        self.swi_dim = num_SWI_params

        def _bayes(in_f, out_f):
            return bnn.BayesLinear(
                prior_mu=0,
                prior_sigma=prior_sigma,
                in_features=in_f,
                out_features=out_f,
            )

        spatial_features = self.spatial_split - self.temporal_split
        spatial_layers = [_bayes(spatial_features, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            spatial_layers.extend([_bayes(hidden_dim, hidden_dim), nn.ReLU()])
        self.spatial_net = nn.Sequential(*spatial_layers)

        temporal_layers = [_bayes(self.temporal_split, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            temporal_layers.extend([_bayes(hidden_dim, hidden_dim), nn.ReLU()])
        self.temporal_net = nn.Sequential(*temporal_layers)

        weather_layers = [_bayes(self.swi_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            weather_layers.extend([_bayes(hidden_dim, hidden_dim), nn.ReLU()])
        self.weather_net = nn.Sequential(*weather_layers)

        self.fusion = nn.Sequential(
            _bayes(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            _bayes(hidden_dim, 2),
        )

    def forward(self, x):
        spatial_out = self.spatial_net(x[:, self.temporal_split : self.spatial_split])
        temporal_out = self.temporal_net(x[:, : self.temporal_split])
        weather_out = self.weather_net(x[:, self.swi_start :])
        x = torch.cat([spatial_out, temporal_out, weather_out], dim=-1)
        x = self.fusion(x)
        mean, variance = torch.split(x, 1, dim=1)
        variance = F.softplus(variance) + VARIANCE_FLOOR
        return mean, variance


class DeepEnsemble(torch.nn.Module):
    """
    General Deep Ensemble wrapper that aggregates predictions from multiple models.
    Supports both Gaussian and Laplacian uncertainty models.
    """

    def __init__(self, models, model_type="Gaussian"):
        super().__init__()
        self.ensemble_models = nn.ModuleList(models)
        self.model_type = model_type  # "Gaussian" or "Laplacian"

    def forward(self, x):
        """
        Forward pass through ensemble.
        Returns:
            ensemble_mean: Mean of all member predictions
            total_uncertainty: Sum of aleatoric (avg per-model var) and epistemic (variance of means)
        """
        predictions = []
        variances_or_scales = []

        # Get outputs from all ensemble members
        for model in self.ensemble_models:
            mean, v = model(x)
            predictions.append(mean)
            variances_or_scales.append(v)

        # Stack: [ensemble_size, batch_size, 1]
        mu_stack = torch.stack(predictions, dim=0)
        v_stack = torch.stack(variances_or_scales, dim=0)

        # 1. Ensemble Mean (Aggregated Prediction)
        ensemble_mean = torch.mean(mu_stack, dim=0)

        # 2. Aleatoric Uncertainty (Expected variance across models)
        if "Laplacian" in self.model_type:
            # For Laplacian: Var = 2 * b^2
            aleatoric_var = torch.mean(2.0 * (v_stack**2), dim=0)
        else:
            # For Gaussian: Var = sigma^2
            aleatoric_var = torch.mean(v_stack, dim=0)

        # 3. Epistemic Uncertainty (Variance of means across ensemble)
        # [PAPER] Mao et al. 2025: Uses population variance for Laplacian ensemble
        # For Gaussian STEC models, we maintain standard unbiased sample variance
        is_laplacian = "Laplacian" in self.model_type
        epistemic_var = torch.var(mu_stack, dim=0, unbiased=not is_laplacian)

        # 4. Total Uncertainty
        total_uncertainty = aleatoric_var + epistemic_var

        return ensemble_mean, total_uncertainty

    def get_uncertainties(self, x):
        """Decomposed uncertainties for analysis."""
        predictions = []
        variances_or_scales = []

        for model in self.ensemble_models:
            mean, v = model(x)
            predictions.append(mean)
            variances_or_scales.append(v)

        mu_stack = torch.stack(predictions, dim=0)
        v_stack = torch.stack(variances_or_scales, dim=0)

        ensemble_mean = torch.mean(mu_stack, dim=0)

        if "Laplacian" in self.model_type:
            aleatoric_var = torch.mean(2.0 * (v_stack**2), dim=0)
        else:
            aleatoric_var = torch.mean(v_stack, dim=0)

        # [PAPER] Mao et al. 2025: Population variance for Laplacian, Unbiased for others
        is_laplacian = "Laplacian" in self.model_type
        epistemic_var = torch.var(mu_stack, dim=0, unbiased=not is_laplacian)
        total_uncertainty = aleatoric_var + epistemic_var

        return (
            ensemble_mean,
            aleatoric_var,
            epistemic_var,
            total_uncertainty,
        )


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
                model.output_layer.bias.fill_(STEC_MEAN_TECU)
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
        # Using unbiased=True (Bessel correction) for standard Gaussian MLP Ensembles
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
        # Using unbiased=True (Bessel correction) for standard Gaussian MLP Ensembles
        epistemic_uncertainty = torch.var(predictions, dim=0, unbiased=True)
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty

        return (
            ensemble_mean,
            aleatoric_uncertainty,
            epistemic_uncertainty,
            total_uncertainty,
        )


# ============================================================================
# Factorized VTEC × MF Model with Uncertainty Propagation
# ============================================================================


class VTECFieldNet(nn.Module):
    """
    Neural network for predicting VTEC (Vertical TEC) with aleatoric and epistemic uncertainty.

    This network takes VTEC-field-related features (IPP location, time, SWI) and
    outputs both the mean VTEC value and its uncertainty (variance).

    Architecture:
        - Multi-layer MLP backbone (deterministic, configurable depth)
        - Bayesian output heads using BayesLinear for epistemic uncertainty
        - ReLU activations (can be swapped for Tanh for smoother fields)

    Uncertainty:
        - Aleatoric: Captured by variance output (data noise)
        - Epistemic: Captured by Bayesian output layers (model uncertainty)
        - Use MC sampling during inference to quantify epistemic uncertainty

    Output:
        vtec_mean: Mean VTEC prediction at the IPP [batch_size]
        vtec_variance: Variance of VTEC prediction [batch_size]

    Note: Changed from log_sigma to variance to match working models (MLP_NLL, BNN_NLL)
          and properly handle STEC-scale uncertainty after MF multiplication.
    """

    def __init__(
        self,
        vtec_in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        activation: str = "relu",
        prior_sigma: float = 0.1,
    ):
        """
        Initialize VTECFieldNet.

        Args:
            vtec_in_dim: Input feature dimension (from FeatureSplitter.get_vtec_dim())
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers in the backbone
            activation: Activation function ("relu" or "tanh")
            prior_sigma: Prior std for Bayesian layers (epistemic uncertainty)
        """
        super().__init__()

        # Select activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. Use 'relu' or 'tanh'."
            )

        # Build deterministic backbone MLP
        layers = []
        in_dim = vtec_in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Bayesian output heads for epistemic uncertainty
        self.mean_head = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=1
        )
        # Changed from log_sigma_head to variance_head to match working models
        self.variance_head = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=1
        )

        # Initialize output heads
        with torch.no_grad():
            # Initialize mean head bias to approximate VTEC mean (~STEC_MEAN_TECU TECU for STEC ≈ VTEC at high elevation)
            self.mean_head.bias_mu.fill_(STEC_MEAN_TECU)
            self.mean_head.weight_mu.normal_(0, 0.01)

            # Initialize variance head for STEC-scale uncertainty
            # Target: NLL ≈ 2.0 requires var_stec ≈ 25-35 at typical elevations
            # With MF averaging ~1.5-2.0, need var_vtec ≈ 10-15 (NOT 22!)
            # softplus_inverse(12) ≈ 3.2
            # Lower than before to compensate for MF² scaling
            self.variance_head.bias_mu.fill_(3.2)  # Will give var ≈ 12 after softplus
            self.variance_head.weight_mu.normal_(0, 0.01)

    def forward(self, x_vtec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VTECFieldNet.

        Args:
            x_vtec: VTEC-related features [batch_size, vtec_in_dim]

        Returns:
            vtec_mean: Mean VTEC prediction [batch_size]
            vtec_variance: Variance of VTEC prediction [batch_size]
        """
        h = self.backbone(x_vtec)
        vtec_mean = self.mean_head(h).squeeze(-1)
        vtec_variance_raw = self.variance_head(h).squeeze(-1)
        # Use softplus + floor like all working models (MLP_NLL, BNN_NLL, etc.)
        vtec_variance = F.softplus(vtec_variance_raw) + VARIANCE_FLOOR
        return vtec_mean, vtec_variance


class GeomNet(nn.Module):
    """
    Neural network for predicting the mapping factor (MF) from geometry features.

    This network takes geometry-related features (station location, elevation, azimuth)
    and outputs a mapping factor that satisfies physical constraints:
        - MF(90°) = 1  (vertical ray has no slant path elongation)
        - MF ≥ 1 for all elevations (slant path ≥ vertical path)
        - MF increases as elevation decreases (longer slant path at low elevations)

    The MF is computed as:
        g(elev) = 1 - sin(elev)  # 0 at 90°, ~1 at 0°
        MF = 1 + g(elev) * softplus(mf_raw)

    This ensures MF(90°) = 1 exactly and MF ≥ 1 everywhere.

    Output:
        mf: Mapping factor [batch_size]
    """

    def __init__(
        self,
        geom_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        activation: str = "relu",
    ):
        """
        Initialize GeomNet.

        Args:
            geom_in_dim: Input feature dimension (from FeatureSplitter.get_geom_dim())
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers in the backbone
            activation: Activation function ("relu" or "tanh")
        """
        super().__init__()

        # Select activation function
        if activation.lower() == "relu":
            act_fn = nn.ReLU
        elif activation.lower() == "tanh":
            act_fn = nn.Tanh
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. Use 'relu' or 'tanh'."
            )

        # Build backbone MLP
        layers = []
        in_dim = geom_in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # MF output head
        self.mf_head = nn.Linear(hidden_dim, 1)

        # Initialize MF head to produce small corrections initially
        with torch.no_grad():
            self.mf_head.bias.fill_(0.0)  # Start with mf_raw ≈ 0
            self.mf_head.weight.normal_(0, 0.01)

    def forward(self, x_geom: torch.Tensor, elev_rad: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GeomNet with elevation-dependent MF constraint.

        Args:
            x_geom: Geometry-related features [batch_size, geom_in_dim]
                   Includes station lat/lon, azimuth info (as direction vector), and SH embeddings
            elev_rad: Elevation in radians [batch_size]

        Returns:
            mf: Mapping factor [batch_size], satisfying MF(90°)=1 and MF≥1
        """
        h = self.backbone(x_geom)
        mf_raw = self.mf_head(h).squeeze(-1)

        # Elevation-dependent scaling: g(90°) = 0, g(0°) ≈ 1
        # This provides the primary elevation dependence (thin-shell approximation)
        g_elev = 1.0 - torch.sin(elev_rad)

        # Final mapping factor with learned corrections:
        # MF = 1 + g_elev * (baseline + learned_correction)
        # - baseline=1: gives thin-shell approximation MF ≈ 1/sin(elev)
        # - mf_raw: learned deviations for latitude/azimuth/local effects
        # - tanh allows both positive and negative corrections in [-1, 1]
        mf = 1.0 + g_elev * (1.0 + torch.tanh(mf_raw))

        return mf


class FactorizedSTECModel(nn.Module):
    """
    Factorized STEC prediction model: STEC = MF × VTEC.

    This model combines:
        - VTECFieldNet: Predicts VTEC and its uncertainty (vtec_mean, vtec_log_sigma)
        - GeomNet: Predicts geometry-dependent mapping factor (MF)

    The STEC prediction and uncertainty are derived as:
        σ_v = exp(vtec_log_sigma)
        μ_stec = MF × vtec_mean
        σ_stec = |MF| × σ_v  (uncertainty propagation)

    This design separates the ionospheric field (VTEC) from geometric effects (MF),
    allowing for better physical interpretation and targeted fine-tuning.

    Inputs (split from collated features via FeatureSplitter):
        x_vtec: VTEC field features (IPP location, time, SWI)
        x_geom: Geometry features (station location, elevation, azimuth)
        elev_rad: Elevation in radians (for MF constraint)

    Output (following repo convention):
        Returns (mu_stec, sigma_stec^2) as (mean, variance) tuple
    """

    def __init__(
        self,
        vtec_in_dim: int,
        geom_in_dim: int,
        vtec_hidden: int = 128,
        geom_hidden: int = 64,
        vtec_layers: int = 3,
        geom_layers: int = 2,
        activation: str = "relu",
        prior_sigma: float = 0.1,
    ):
        """
        Initialize FactorizedSTECModel.

        Args:
            vtec_in_dim: VTEC feature dimension (from FeatureSplitter)
            geom_in_dim: Geometry feature dimension (from FeatureSplitter)
            vtec_hidden: Hidden dimension for VTECFieldNet
            geom_hidden: Hidden dimension for GeomNet
            vtec_layers: Number of layers in VTECFieldNet
            geom_layers: Number of layers in GeomNet
            activation: Activation function ("relu" or "tanh")
            prior_sigma: Prior std for Bayesian layers in VTEC network
        """
        super().__init__()

        self.vtec_net = VTECFieldNet(
            vtec_in_dim,
            hidden_dim=vtec_hidden,
            num_layers=vtec_layers,
            activation=activation,
            prior_sigma=prior_sigma,
        )
        self.geom_net = GeomNet(
            geom_in_dim,
            hidden_dim=geom_hidden,
            num_layers=geom_layers,
            activation=activation,
        )

        # Store dimensions for debugging
        self.vtec_in_dim = vtec_in_dim
        self.geom_in_dim = geom_in_dim

    def forward(
        self, x_vtec: torch.Tensor, x_geom: torch.Tensor, elev_rad: torch.Tensor
    ):
        """
        Forward pass through factorized STEC model.

        Args:
            x_vtec: VTEC field features [batch_size, vtec_in_dim]
            x_geom: Geometry features [batch_size, geom_in_dim]
            elev_rad: Elevation in radians [batch_size]

        Returns:
            For training (when called from training loop):
                Returns (mu_stec, var_stec) as tuple following repo convention

            For inference/analysis (when return_dict=True in config):
                Returns dict with all intermediate values:
                {
                    "vtec_mean": VTEC mean prediction,
                    "vtec_variance": VTEC variance,
                    "sigma_v": VTEC standard deviation,
                    "mf": Mapping factor,
                    "mu_stec": STEC mean prediction,
                    "sigma_stec": STEC standard deviation,
                    "var_stec": STEC variance
                }
        """
        # VTEC prediction with uncertainty
        vtec_mean, vtec_variance = self.vtec_net(x_vtec)
        sigma_v = torch.sqrt(vtec_variance)

        # Mapping factor prediction
        mf = self.geom_net(x_geom, elev_rad)

        # Combine: STEC = MF × VTEC with uncertainty propagation
        mu_stec = mf * vtec_mean

        # Propagate uncertainty: var_stec = MF^2 * var_vtec
        # This is correct variance propagation for multiplication by a constant
        var_stec_prop = (mf**2) * vtec_variance

        # Add minimum variance floor to prevent over-confidence
        # This matches the pattern from working models: F.softplus(var) + VARIANCE_FLOOR
        # The floor is already in vtec_variance, but we add a small STEC-scale floor
        var_stec = var_stec_prop + VARIANCE_FLOOR

        # Return as (mean, variance) tuple following repo convention
        # The training loop expects this format
        return mu_stec, var_stec

    def forward_detailed(
        self, x_vtec: torch.Tensor, x_geom: torch.Tensor, elev_rad: torch.Tensor
    ):
        """
        Forward pass with detailed outputs for analysis and debugging.

        Returns a dictionary with all intermediate values for visualization
        and understanding of the model's internal predictions.

        Args:
            x_vtec: VTEC field features [batch_size, vtec_in_dim]
            x_geom: Geometry features [batch_size, geom_in_dim]
            elev_rad: Elevation in radians [batch_size]

        Returns:
            dict: Detailed outputs including VTEC, MF, STEC, and uncertainties
        """
        # VTEC prediction with uncertainty
        vtec_mean, vtec_variance = self.vtec_net(x_vtec)
        sigma_v = torch.sqrt(vtec_variance)

        # Mapping factor prediction
        mf = self.geom_net(x_geom, elev_rad)

        # Combine: STEC = MF × VTEC with uncertainty propagation
        mu_stec = mf * vtec_mean
        var_stec_prop = (mf**2) * vtec_variance
        var_stec = var_stec_prop + VARIANCE_FLOOR
        sigma_stec = torch.sqrt(var_stec)

        return {
            "vtec_mean": vtec_mean,
            "vtec_variance": vtec_variance,
            "sigma_v": sigma_v,
            "mf": mf,
            "mu_stec": mu_stec,
            "sigma_stec": sigma_stec,
            "var_stec": var_stec,
        }


class FactorizedSTECModelWrapper(nn.Module):
    """
    Wrapper for FactorizedSTECModel that integrates with the existing training pipeline.

    This wrapper:
    1. Receives the full collated feature tensor (as all other models do)
    2. Splits features into VTEC and geometry components using FeatureSplitter
    3. Extracts elevation in radians for the MF constraint
    4. Calls the factorized model with split inputs
    5. Returns (mean, variance) tuple matching repo convention

    This allows the factorized model to work seamlessly with the existing
    training/validation loops without modification.
    """

    def __init__(self, factorized_model, feature_splitter):
        """
        Initialize wrapper with factorized model and feature splitter.

        Args:
            factorized_model: FactorizedSTECModel instance
            feature_splitter: FeatureSplitter instance for splitting features
        """
        super().__init__()
        self.model = factorized_model
        self.splitter = feature_splitter

        # Flag to indicate if this model should use Bayesian inference
        # Currently factorized model has deterministic components, but this
        # can be extended to Bayesian VTEC network
        self._is_bayesian = False

    def forward(self, x):
        """
        Forward pass through wrapped factorized model.

        Args:
            x: Full collated feature tensor [batch_size, total_features]

        Returns:
            (mean, variance) tuple following repo convention
        """
        # Split features into VTEC, geometry, and elevation components
        x_vtec, x_geom, elev_rad = self.splitter.split_features(x)

        # Forward through factorized model
        return self.model(x_vtec, x_geom, elev_rad)

    def forward_detailed(self, x):
        """
        Forward pass with detailed outputs for analysis.

        Args:
            x: Full collated feature tensor [batch_size, total_features]

        Returns:
            dict: Detailed outputs including VTEC, MF, STEC, and uncertainties
        """
        x_vtec, x_geom, elev_rad = self.splitter.split_features(x)
        return self.model.forward_detailed(x_vtec, x_geom, elev_rad)


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
    dropout_rate = config["model"].get(
        "dropout_rate", 0.1
    )  # Default dropout rate for MC Dropout

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
    sh_degree = config["data"]["SH_degree"]

    # [LEGACY] For existing STEC ResNet models, SH dimension was calculated as sh_degree²
    # [PAPER] For Mao et al. 2025 VTEC alignment, SH dim is (sh_degree + 1)²
    if config.get("target") == "vtec" or "Laplacian" in model_type:
        sh_dim_per_location = (sh_degree + 1) ** 2 if sh_degree > 0 else 0
    else:
        sh_dim_per_location = sh_degree**2 if sh_degree > 0 else 0

    # Calculate total SH dimension by counting which coordinate systems are enabled
    total_sh_dim = 0
    if sh_degree > 0:
        num_sh_locations = 0

        # Check for station geographic coordinates
        if "lat_sta" in station_features and "lon_sta" in station_features:
            num_sh_locations += 1

        # Check for station SM coordinates
        if "sm_lat_sta" in station_features and "sm_lon_sta" in station_features:
            num_sh_locations += 1

        # Check for IPP geographic coordinates
        if "lat_ipp" in ipp_features and "lon_ipp" in ipp_features:
            num_sh_locations += 1

        # Check for IPP SM coordinates
        if "sm_lat_ipp" in ipp_features and "sm_lon_ipp" in ipp_features:
            num_sh_locations += 1

        total_sh_dim = num_sh_locations * sh_dim_per_location

    # Total input features after all transformations
    in_features = (
        temporal_dim + station_dim + direction_dim + ipp_dim + swi_dim + total_sh_dim
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
    elif model_type == "MLP_LaplacianNLL":  # [PAPER] Mao et al. 2025
        model = MLP_LaplacianNLL(
            n_in=in_features, hidden_dim=hidden_dim, num_layers=num_layers
        )
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
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif model_type == "MLP_MCDropout_NLL":
        model = MLP_MCDropout_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
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
    elif model_type == "Branch_MLP_MSE":
        return Branch_MLP_MSE(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type == "Branch_MLP_NLL":
        model = Branch_MLP_NLL(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return model
    elif model_type == "Branch_BNN_NLL":
        model = Branch_BNN_NLL(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            prior_sigma=prior_sigma,
        )
        return model
    elif model_type == "Branch3Way_MLP_MSE":
        return Branch3Way_MLP_MSE(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
    elif model_type == "Branch3Way_MLP_NLL":
        model = Branch3Way_MLP_NLL(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        return model
    elif model_type == "Branch3Way_BNN_NLL":
        model = Branch3Way_BNN_NLL(
            n_in=in_features,
            num_SWI_params=num_SWI_params,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            prior_sigma=prior_sigma,
        )
        return model
    elif model_type == "ResNet_MSE":
        return ResNet_MSE(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif model_type == "ResNet_NLL":
        return ResNet_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
        )
    elif model_type == "ResNet_BNN_NLL":
        model = ResNet_BNN_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            prior_sigma=prior_sigma,
        )
        return model
    elif model_type == "BayesianResNetSTEC":
        model = BayesianResNetSTEC(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            prior_sigma=prior_sigma,
        )
        return model
    elif model_type == "AttentionMLP_MSE":
        return AttentionMLP_MSE(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config["model"].get("num_heads", 4),
            dropout_rate=dropout_rate,
        )
    elif model_type == "AttentionMLP_NLL":
        return AttentionMLP_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config["model"].get("num_heads", 4),
            dropout_rate=dropout_rate,
        )
    elif model_type == "AttentionMLP_BNN_NLL":
        return AttentionMLP_BNN_NLL(
            n_in=in_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=config["model"].get("num_heads", 4),
            dropout_rate=dropout_rate,
            prior_sigma=prior_sigma,
        )
    elif model_type == "FactorizedSTEC":
        # Import FeatureSplitter for feature splitting
        from utils.feature_splitter import FeatureSplitter

        # Create feature splitter
        splitter = FeatureSplitter(feature_registry)

        # Get VTEC and geometry dimensions
        vtec_dim = splitter.get_vtec_dim()
        geom_dim = splitter.get_geom_dim()

        # Get model hyperparameters
        vtec_hidden = config["model"].get("vtec_hidden", 128)
        geom_hidden = config["model"].get("geom_hidden", 64)
        vtec_layers = config["model"].get(
            "vtec_layers", num_layers
        )  # Default to num_layers
        geom_layers = config["model"].get("geom_layers", 2)
        activation = config["model"].get("activation", "relu")

        # Create factorized model with Bayesian VTEC network
        factorized_model = FactorizedSTECModel(
            vtec_in_dim=vtec_dim,
            geom_in_dim=geom_dim,
            vtec_hidden=vtec_hidden,
            geom_hidden=geom_hidden,
            vtec_layers=vtec_layers,
            geom_layers=geom_layers,
            activation=activation,
            prior_sigma=prior_sigma,  # Bayesian layers in VTEC network
        )

        # Wrap model with feature splitter integration
        model = FactorizedSTECModelWrapper(factorized_model, splitter)
        return model
    else:
        raise ValueError(f"Model type {model_type} is not recognized.")


def load_model_for_inference(config, experiment_dir, logger=None):
    """
    Unified loader for both single and ensemble models.
    Detects multiple checkpoints in model/ folder and aggregates them if found.
    """

    device = config.get("device", torch.device("cpu"))
    model_dir = Path(experiment_dir) / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find all .pth files
    pth_files = sorted(list(model_dir.glob("*.pth")))
    if not pth_files:
        raise FileNotFoundError(f"No model checkpoints found in {model_dir}")

    # If multiple checkpoints found, create an ensemble
    if len(pth_files) > 1:
        if logger:
            logger.info(
                f"👥 Detected {len(pth_files)} ensemble members. Loading ensemble..."
            )

        models = []
        for pth_path in pth_files:
            model = get_model(config).to(device)
            # Use weights_only=True for security (PyTorch 2.4+)
            checkpoint = torch.load(pth_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)

        # Determine model distribution type for ensemble wrapper
        model_type = config["model"]["model_type"]
        dist_type = "Laplacian" if "Laplacian" in model_type else "Gaussian"

        ensemble = DeepEnsemble(models, model_type=dist_type)
        return ensemble.to(device)

    # Otherwise, load single model
    else:
        if logger:
            logger.info(f"🎯 Loading single model: {pth_files[0].name}")

        model = get_model(config).to(device)
        checkpoint = torch.load(pth_files[0], map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
