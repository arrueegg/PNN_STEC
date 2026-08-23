"""The paper's architecture, ported so that it is the same function as before.

`BayesianResNetSTEC` is a deterministic input projection, four residual blocks, and a
**Bayesian output layer only** - not a fully Bayesian network. Its second output is a
log-variance passed through softplus with a floor, which is what keeps the Gaussian NLL
from diverging when the model becomes confident.

The port changes no arithmetic. Every layer keeps the name it had, because the equivalence
diagnostic loads a checkpoint written by the old class into this one, and a renamed
parameter would break that silently. What changes is that the model now *declares* its
capabilities instead of the pipeline inferring them from its class name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchbnn as bnn
from torch import nn

from .capabilities import GAUSSIAN_BAYESIAN, Capabilities

# Approximate mean STEC in TECU. Used to initialise the output bias so the model starts in
# the right part of the range rather than at zero.
STEC_MEAN_TECU = 15.5

# Minimum predicted variance. Without it the Gaussian NLL is unbounded below: the model can
# drive variance to zero on an easy observation and take an arbitrarily large reward.
VARIANCE_FLOOR = 1e-3

# Output bias initialisation and the variance floor are both absent from the paper's
# hyperparameter table, along with the KL warmup. They are recorded here so the generated
# table can pick them up rather than relying on someone remembering.
UNDOCUMENTED_HYPERPARAMETERS = {
    "output_bias_init_tecu": STEC_MEAN_TECU,
    "variance_floor": VARIANCE_FLOOR,
}


class ResNetBlock(nn.Module):
    """Pre-norm residual block: norm, fc, relu, norm, fc, then add the input back."""

    def __init__(self, hidden_dim: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


class BayesianResNetSTEC(nn.Module):
    """Deterministic ResNet backbone with a Bayesian output head.

    Returns `(mean, variance)`. The variance is `softplus(log_var) + VARIANCE_FLOOR`, so it
    is strictly positive by construction.
    """

    capabilities: Capabilities = GAUSSIAN_BAYESIAN

    def __init__(
        self,
        n_in: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout_rate: float = 0.0,
        prior_sigma: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(n_in, hidden_dim), nn.ReLU())
        self.res_blocks = nn.ModuleList(
            ResNetBlock(hidden_dim, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        )
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )
        with torch.no_grad():
            self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)
            self.output_layer.weight_mu.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        mean, log_var = torch.split(x, 1, dim=1)
        variance = F.softplus(log_var) + VARIANCE_FLOOR
        return mean, variance


def shape_from_state_dict(state: dict) -> dict:
    """Recover the constructor arguments a checkpoint was written with.

    A checkpoint records the architecture in its tensor shapes, so the model can be rebuilt
    without consulting the config that produced it. That matters for the equivalence
    diagnostic, which has to instantiate a model for a checkpoint written before run_ids
    existed, and it removes one more place where two sources of the same fact can disagree.
    """
    weight = state["input_layer.0.weight"]
    hidden_dim, n_in = int(weight.shape[0]), int(weight.shape[1])
    blocks = {int(key.split(".")[1]) for key in state if key.startswith("res_blocks.")}
    return {
        "n_in": n_in,
        "hidden_dim": hidden_dim,
        "num_layers": len(blocks),
    }


def load_checkpoint(path, map_location="cpu") -> tuple[BayesianResNetSTEC, dict]:
    """Build the model a checkpoint describes, and load it. Returns (model, shape)."""
    state = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    shape = shape_from_state_dict(state)
    model = BayesianResNetSTEC(**shape)
    model.load_state_dict(state)
    model.eval()
    return model, shape
