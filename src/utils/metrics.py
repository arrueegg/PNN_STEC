import torch
import torch.nn.functional as F
import numpy as np


def mse(predictions, targets):
    return F.mse_loss(predictions, targets).item()


def mae(predictions, targets):
    return F.l1_loss(predictions, targets).item()


def rmse(predictions, targets):
    return torch.sqrt(F.mse_loss(predictions, targets)).item()


def mape(predictions, targets):
    epsilon = 1e-7  # to avoid division by zero
    return (
        torch.mean(torch.abs((targets - predictions) / (targets + epsilon))) * 100
    ).item()


def r2_score(predictions, targets):
    ss_res = torch.sum((targets - predictions) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()


def residual_std(predictions, targets):
    residuals = targets - predictions
    return residuals.std().item()


def residual_iqr(predictions, targets):
    residuals = targets - predictions
    q75, q25 = (
        torch.quantile(residuals, 0.75).item(),
        torch.quantile(residuals, 0.25).item(),
    )
    return q75 - q25  # IQR = Q3 - Q1


def calculate_metrics(predictions, targets, prefix):
    """Calculates and returns a dictionary of metrics for each technology type."""
    import torch

    # Convert to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    metrics = {}

    if predictions.numel() > 0:  # Check if there are any samples for this tech
        if predictions.dim() > 1 and predictions.shape[1] > 1:
            stec = predictions[:, 0].squeeze(-1)
            uncertainty = predictions[:, 1].squeeze(-1)
        else:
            stec = predictions.squeeze(-1)
            uncertainty = None

        metrics.update(
            {
                f"{prefix}_MSE": mse(stec, targets),
                f"{prefix}_MAE": mae(stec, targets),
                f"{prefix}_RMSE": rmse(stec, targets),
                f"{prefix}_MAPE": mape(stec, targets),
            }
        )

        if uncertainty is not None:  # If there is an uncertainty column
            metrics.update({f"{prefix}_uncertainty_mean": uncertainty.mean().item()})

    return metrics
