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

from .capabilities import GAUSSIAN_BAYESIAN, LAPLACE_DETERMINISTIC, Capabilities

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


class BayesResNetBlock(nn.Module):
    """Bayesian residual block: both `fc1`/`fc2` are `bnn.BayesLinear`, not `nn.Linear`.

    Ported from `src/model/model.py`. This is the one structural difference between
    `ResNet_BNN_NLL` (below) and `BayesianResNetSTEC` above - everything else in the two
    architectures, including the output layer and its initialisation, is identical by
    construction (see `ResNet_BNN_NLL`'s docstring).
    """

    def __init__(
        self, hidden_dim: int, dropout_rate: float = 0.0, prior_sigma: float = 0.1
    ) -> None:
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


class ResNet_BNN_NLL(nn.Module):
    """The fully-Bayesian R2.2 variant: Bayesian residual blocks, not just a Bayesian
    output head. Ported from `src/model/model.py` - this is the one architecture CLAUDE.md
    flags as "actively being evaluated right now" and having no prior `stec/models`
    equivalent, so it is ported here verbatim rather than left in `src/`.

    `docs/revision/STATE.md` records why the output-layer initialisation below matters:
    the first R2.2 comparison measured this architecture confounded with a missing
    initialisation match, producing a -1.93 TECU pervasive bias that had nothing to do with
    "Bayesian residual blocks" as a modelling choice. With the matched initialisation, the
    two architectures (this one and `BayesianResNetSTEC`) differ in exactly one way: each
    residual block's `fc1`/`fc2` are Bayesian here, deterministic there.
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
            BayesResNetBlock(
                hidden_dim, dropout_rate=dropout_rate, prior_sigma=prior_sigma
            )
            for _ in range(num_layers)
        )
        self.output_layer = bnn.BayesLinear(
            prior_mu=0, prior_sigma=prior_sigma, in_features=hidden_dim, out_features=2
        )
        # Matches BayesianResNetSTEC's initialisation exactly - see the class docstring
        # for why an unmatched init previously confounded this ablation.
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


class MLP_LaplacianNLL(nn.Module):
    """The canonical VTEC baseline (Mao et al. 2025): 3 hidden layers of 90 neurons with
    tanh activation, outputting a Laplace (location, scale) pair. Ported from
    `src/model/model.py`. Predicts a *scale*, not a standard deviation - see
    `stec.inference.monte_carlo.laplace_scale_to_variance` and CLAUDE.md's prediction-store
    notes for how that is converted downstream.
    """

    capabilities: Capabilities = LAPLACE_DETERMINISTIC

    def __init__(
        self, n_in: int = 259, hidden_dim: int = 90, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_in, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        x = self.output_layer(x)
        location, log_scale = torch.split(x, 1, dim=1)
        scale = F.softplus(log_scale) + VARIANCE_FLOOR
        return location, scale


class DeepEnsemble(nn.Module):
    """Aggregates predictions from multiple (mean, spread) models - Gaussian or Laplacian.

    Ported from `src/model/model.py`, needed by `legacy_factory.load_model_for_inference`
    for any experiment directory holding more than one checkpoint (ensemble members).
    """

    def __init__(self, models: list[nn.Module], model_type: str = "Gaussian") -> None:
        super().__init__()
        self.ensemble_models = nn.ModuleList(models)
        self.model_type = model_type  # "Gaussian" or "Laplacian"

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (ensemble_mean, total_uncertainty) where total_uncertainty is the sum
        of the aleatoric (mean per-model variance) and epistemic (variance of means)
        components."""
        predictions = []
        variances_or_scales = []

        for model in self.ensemble_models:
            mean, v = model(x)
            predictions.append(mean)
            variances_or_scales.append(v)

        mu_stack = torch.stack(predictions, dim=0)
        v_stack = torch.stack(variances_or_scales, dim=0)

        ensemble_mean = torch.mean(mu_stack, dim=0)

        is_laplacian = "Laplacian" in self.model_type
        if is_laplacian:
            aleatoric_var = torch.mean(2.0 * (v_stack**2), dim=0)
        else:
            aleatoric_var = torch.mean(v_stack, dim=0)

        # [PAPER] Mao et al. 2025: population variance for the Laplacian ensemble, unbiased
        # sample variance otherwise.
        epistemic_var = torch.var(mu_stack, dim=0, unbiased=not is_laplacian)

        total_uncertainty = aleatoric_var + epistemic_var

        return ensemble_mean, total_uncertainty

    def get_uncertainties(self, x: torch.Tensor):
        """Decomposed (mean, aleatoric_var, epistemic_var, total_var) for analysis."""
        predictions = []
        variances_or_scales = []

        for model in self.ensemble_models:
            mean, v = model(x)
            predictions.append(mean)
            variances_or_scales.append(v)

        mu_stack = torch.stack(predictions, dim=0)
        v_stack = torch.stack(variances_or_scales, dim=0)

        ensemble_mean = torch.mean(mu_stack, dim=0)

        is_laplacian = "Laplacian" in self.model_type
        if is_laplacian:
            aleatoric_var = torch.mean(2.0 * (v_stack**2), dim=0)
        else:
            aleatoric_var = torch.mean(v_stack, dim=0)

        epistemic_var = torch.var(mu_stack, dim=0, unbiased=not is_laplacian)
        total_uncertainty = aleatoric_var + epistemic_var

        return ensemble_mean, aleatoric_var, epistemic_var, total_uncertainty


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


def shape_from_vtec_state_dict(state: dict) -> dict:
    """`shape_from_state_dict`'s counterpart for `MLP_LaplacianNLL`.

    Reads a different set of tensor names because the two architectures are built
    differently: `BayesianResNetSTEC` names its blocks `input_layer`/`res_blocks`,
    `MLP_LaplacianNLL` keeps every hidden layer in one flat `layers` list ending in
    `output_layer`. A single shape-reader that tried to cover both would have to guess
    which naming convention a checkpoint uses; keeping them separate means each one only
    has to be right about the architecture it names.
    """
    weight = state["layers.0.weight"]
    hidden_dim, n_in = int(weight.shape[0]), int(weight.shape[1])
    layer_indices = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("layers.") and key.endswith(".weight")
    }
    return {
        "n_in": n_in,
        "hidden_dim": hidden_dim,
        "num_layers": len(layer_indices),
    }


def load_vtec_checkpoint(path, map_location="cpu") -> tuple[MLP_LaplacianNLL, dict]:
    """Build the `MLP_LaplacianNLL` checkpoint a VTEC baseline run wrote, and load it.

    Mirrors `load_checkpoint` exactly, one architecture over: the constructor arguments
    come from the checkpoint's own tensor shapes rather than a config, so loading a VTEC
    checkpoint never depends on `stec.models.legacy_factory`'s `FeatureRegistry`-based
    sizing (a separate, config-driven path used elsewhere for ensembles) - it only needs
    the file itself.
    """
    state = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    shape = shape_from_vtec_state_dict(state)
    model = MLP_LaplacianNLL(**shape)
    model.load_state_dict(state)
    model.eval()
    return model, shape
