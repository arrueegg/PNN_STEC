"""The ported architecture, and the capability flags that replace name sniffing."""

from __future__ import annotations

import torch

from stec.models import capabilities
from stec.models.architectures import (
    VARIANCE_FLOOR,
    BayesianResNetSTEC,
    shape_from_state_dict,
)


def test_forward_returns_mean_and_variance():
    torch.manual_seed(0)
    model = BayesianResNetSTEC(n_in=8, hidden_dim=16, num_layers=2).eval()
    mean, variance = model(torch.randn(5, 8))
    assert mean.shape == (5, 1)
    assert variance.shape == (5, 1)


def test_variance_never_falls_below_the_floor():
    """Without the floor the Gaussian NLL is unbounded below and training diverges."""
    torch.manual_seed(0)
    model = BayesianResNetSTEC(n_in=8, hidden_dim=16, num_layers=2).eval()
    with torch.no_grad():
        # Drive the log-variance output as negative as the head allows.
        model.output_layer.bias_mu[1].fill_(-1e4)
        model.output_layer.weight_log_sigma.fill_(-1e4)
        _, variance = model(torch.randn(64, 8))
    assert float(variance.min()) >= VARIANCE_FLOOR


def test_output_bias_starts_at_the_stec_mean():
    model = BayesianResNetSTEC(n_in=4, hidden_dim=8, num_layers=1)
    assert float(model.output_layer.bias_mu[0]) == 15.5


def test_shape_is_recoverable_from_a_state_dict():
    """A checkpoint records its architecture in its tensor shapes, so it needs no config."""
    model = BayesianResNetSTEC(n_in=11, hidden_dim=32, num_layers=3)
    assert shape_from_state_dict(model.state_dict()) == {
        "n_in": 11,
        "hidden_dim": 32,
        "num_layers": 3,
    }


def test_a_recovered_shape_round_trips():
    original = BayesianResNetSTEC(n_in=7, hidden_dim=16, num_layers=2)
    rebuilt = BayesianResNetSTEC(**shape_from_state_dict(original.state_dict()))
    rebuilt.load_state_dict(original.state_dict())


def test_the_model_declares_its_own_capabilities():
    """The pipeline must not have to parse a class name to learn this."""
    caps = BayesianResNetSTEC.capabilities
    assert caps.samples_weights is True
    assert caps.distribution == "gaussian"
    assert caps.spread_kind == "variance"


def test_sample_count_follows_the_capability_not_the_name():
    assert capabilities.GAUSSIAN_BAYESIAN.monte_carlo_samples(100) == 100
    assert capabilities.LAPLACE_DETERMINISTIC.monte_carlo_samples(100) == 1
    assert capabilities.DETERMINISTIC_POINT.monte_carlo_samples(100) == 1


def test_the_vtec_baseline_is_a_laplace_scale():
    """Scoring its scale as a Gaussian std reads 90% coverage at nominal 50%."""
    caps = capabilities.LAPLACE_DETERMINISTIC
    assert caps.distribution == "laplace"
    assert caps.spread_kind == "scale"
