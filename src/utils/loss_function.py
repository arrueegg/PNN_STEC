import torch
import torch.nn as nn
import torchbnn as bnn


class FairCRPSLoss(nn.Module):
    """
    Fair CRPS (Continuous Ranked Probability Score) loss for probabilistic regression.
    
    Fair CRPS definition (per sample b in batch, scalar target y_b):
        fCRPS(s_{1:N}, y) = (1/N) * sum_n |s_n - y|  -  (1 / (2*N*(N-1))) * sum_{n != m} |s_n - s_m|
    
    This loss requires multiple stochastic forward passes through the model to generate
    N samples per input. It encourages both accuracy (first term) and diversity (second term).
    
    Args:
        samples: Tensor of shape [N, B] or [N, B, 1] where N is number of stochastic samples
                 and B is batch size
        y: Target tensor of shape [B] or [B, 1]
    
    Returns:
        Scalar loss tensor (mean over batch)
    
    Raises:
        ValueError: If N < 2 (degenerates to MAE and defeats probabilistic training)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, samples: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Squeeze trailing singleton dims for robustness
        if samples.dim() == 3 and samples.size(-1) == 1:
            samples = samples.squeeze(-1)  # [N, B]
        if y.dim() == 2 and y.size(-1) == 1:
            y = y.squeeze(-1)              # [B]

        assert samples.dim() == 2, f"samples must be [N,B], got {samples.shape}"
        assert y.dim() == 1, f"y must be [B], got {y.shape}"

        N, B = samples.shape
        if N < 2:
            raise ValueError(
                f"FairCRPSLoss requires N >= 2 samples per input (got N={N}). "
                "N==1 degenerates to MAE and defeats probabilistic training."
            )

        y = y.view(1, B)  # [1, B]

        # term1: mean_n |s_n - y|
        term1 = torch.mean(torch.abs(samples - y), dim=0)  # [B]

        # term2: (1 / (2*N*(N-1))) * sum_{n!=m} |s_n - s_m|
        # Compute pairwise differences: [N, N, B]
        diffs = torch.abs(samples.unsqueeze(1) - samples.unsqueeze(0))  # [N, N, B]
        
        # Sum all pairwise differences and subtract diagonal (where n==m)
        # Create mask for off-diagonal elements
        mask = ~torch.eye(N, dtype=torch.bool, device=samples.device).unsqueeze(2)  # [N, N, 1]
        off_diag_sum = (diffs * mask).sum(dim=(0, 1))  # [B]
        
        term2 = off_diag_sum / (2.0 * N * (N - 1))  # [B]

        return (term1 - term2).mean()  # scalar


class WeightedMSELoss(nn.Module):
    def __init__(self, weight_function="linear"):
        super().__init__()
        self.weight_function = weight_function

    def forward(self, predictions, targets):
        # Calculate weights based on target magnitude
        if self.weight_function == "linear":
            # Higher targets get proportionally higher weights
            weights = targets / targets.mean()
        elif self.weight_function == "quadratic":
            # Quadratic scaling for even more emphasis on high values
            weights = (targets / targets.mean()) ** 2
        elif self.weight_function == "log":
            # Logarithmic scaling for moderate emphasis
            weights = torch.log(1 + targets / targets.mean())
        else:
            weights = torch.ones_like(targets)

        # Apply weighted MSE
        weighted_errors = weights * (predictions - targets) ** 2
        return weighted_errors.mean()


class WeightedGaussianNLLLoss(nn.Module):
    def __init__(self, weight_function="linear", eps=1e-6):
        super().__init__()
        self.weight_function = weight_function
        self.eps = eps
        self.base_loss = nn.GaussianNLLLoss(
            reduction="none"
        )  # No reduction for element-wise weighting

    def forward(self, predictions, targets, variances):
        # Ensure variances are positive
        variances = torch.clamp(variances, min=self.eps)

        # Calculate weights based on target magnitude
        if self.weight_function == "linear":
            weights = targets / (targets.mean() + self.eps)
        elif self.weight_function == "quadratic":
            weights = (targets / (targets.mean() + self.eps)) ** 2
        elif self.weight_function == "log":
            weights = torch.log(1 + targets / (targets.mean() + self.eps))
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
        std = (
            std.reshape(
                -1,
            )
            + 1e-6
        )
        loss = torch.sum(torch.log(2 * std) + torch.abs(y - mu) / std)
        return loss


def get_criterion(config, loss_fn=None):
    loss_type = config["training"]["loss_function"]
    if loss_fn is not None:
        loss_type = loss_fn

    # Get weighting configuration
    weighting_config = config["training"].get("target_weighting", {})
    use_weighting = weighting_config.get("enabled", False)
    weight_function = weighting_config.get("weight_function", "linear")

    # Initialize the base loss function
    if loss_type == "MSELoss":
        if use_weighting:
            criterion = WeightedMSELoss(weight_function=weight_function)
        else:
            criterion = nn.MSELoss()
    elif loss_type == "MAELoss":
        criterion = nn.L1Loss()
    elif loss_type == "HuberLoss":
        criterion = nn.SmoothL1Loss()
    elif loss_type == "LaplaceLoss":
        criterion = LaplaceLoss()
    elif loss_type == "GaussianNLLLoss":
        if use_weighting:
            criterion = WeightedGaussianNLLLoss(weight_function=weight_function)
        else:
            criterion = nn.GaussianNLLLoss()
    elif loss_type == "BKLLoss":
        criterion = bnn.BKLLoss(reduction="mean", last_layer_only=False)
    else:
        raise Exception(f"unknown loss {loss_type}")

    return criterion
