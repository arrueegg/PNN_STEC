"""The ported architecture, and the capability flags that replace name sniffing."""

from __future__ import annotations

import pytest
import torch

from stec.models import capabilities
from stec.models.architectures import (
    VARIANCE_FLOOR,
    BayesianResNetSTEC,
    MLP_LaplacianNLL,
    ResNet_BNN_NLL,
    detect_architecture,
    load_checkpoint,
    load_vtec_checkpoint,
    shape_from_state_dict,
    shape_from_vtec_state_dict,
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


# --- MLP_LaplacianNLL: a different naming convention needs its own shape reader ---------


def test_vtec_shape_is_recoverable_from_a_state_dict():
    """`layers.0`/`output_layer`, not `input_layer`/`res_blocks` - a checkpoint still
    records its own architecture without a config."""
    model = MLP_LaplacianNLL(n_in=13, hidden_dim=24, num_layers=4)
    assert shape_from_vtec_state_dict(model.state_dict()) == {
        "n_in": 13,
        "hidden_dim": 24,
        "num_layers": 4,
    }


def test_a_recovered_vtec_shape_round_trips():
    original = MLP_LaplacianNLL(n_in=9, hidden_dim=12, num_layers=2)
    rebuilt = MLP_LaplacianNLL(**shape_from_vtec_state_dict(original.state_dict()))
    rebuilt.load_state_dict(original.state_dict())


def test_vtec_forward_returns_location_and_scale():
    torch.manual_seed(0)
    model = MLP_LaplacianNLL(n_in=6, hidden_dim=8, num_layers=2).eval()
    location, scale = model(torch.randn(5, 6))
    assert location.shape == (5, 1)
    assert scale.shape == (5, 1)
    assert float(scale.min()) >= VARIANCE_FLOOR


def test_vtec_declares_laplace_deterministic_capabilities():
    caps = MLP_LaplacianNLL.capabilities
    assert caps.samples_weights is False
    assert caps.distribution == "laplace"
    assert caps.spread_kind == "scale"


def test_load_vtec_checkpoint_round_trips_through_disk(tmp_path):
    """Mirrors the real checkpoint format: `{"model_state_dict": ...}`, as written by
    `src/finetune.py` and read by `load_model_for_inference`."""
    original = MLP_LaplacianNLL(n_in=5, hidden_dim=6, num_layers=2)
    checkpoint_path = tmp_path / "finetune_MLP_LaplacianNLL_seed42.pth"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint_path)

    loaded, shape = load_vtec_checkpoint(checkpoint_path)
    assert shape == {"n_in": 5, "hidden_dim": 6, "num_layers": 2}

    inputs = torch.randn(4, 5)
    with torch.no_grad():
        original_out = original(inputs)
        loaded_out = loaded(inputs)
    torch.testing.assert_close(original_out[0], loaded_out[0])
    torch.testing.assert_close(original_out[1], loaded_out[1])


# --- ResNet_BNN_NLL: shares input_layer/res_blocks naming with BayesianResNetSTEC, so it
# needs its own detection path rather than its own shape reader ------------------------


def test_detect_architecture_reads_bayesian_resnet_stec():
    model = BayesianResNetSTEC(n_in=6, hidden_dim=8, num_layers=2)
    assert detect_architecture(model.state_dict()) is BayesianResNetSTEC


def test_detect_architecture_distinguishes_bayesian_residual_blocks():
    """`ResNet_BNN_NLL` and `BayesianResNetSTEC` share every layer name; only the residual
    blocks' own parameter names (`weight_mu` vs plain `weight`) tell them apart."""
    model = ResNet_BNN_NLL(n_in=6, hidden_dim=8, num_layers=2)
    assert detect_architecture(model.state_dict()) is ResNet_BNN_NLL


def test_detect_architecture_reads_the_vtec_mlp():
    model = MLP_LaplacianNLL(n_in=6, hidden_dim=8, num_layers=2)
    assert detect_architecture(model.state_dict()) is MLP_LaplacianNLL


def test_detect_architecture_rejects_an_unrecognised_state_dict():
    with pytest.raises(ValueError, match="none of the ported architectures"):
        detect_architecture({"some.other.weight": torch.zeros(1)})


def test_load_checkpoint_builds_bayesian_resnet_stec(tmp_path):
    """The pre-existing behaviour every current caller relies on must be unchanged."""
    original = BayesianResNetSTEC(n_in=7, hidden_dim=8, num_layers=2)
    checkpoint_path = tmp_path / "finetune_BayesianResNetSTEC_seed42.pth"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint_path)

    loaded, shape = load_checkpoint(checkpoint_path)
    assert type(loaded) is BayesianResNetSTEC
    assert shape == {"n_in": 7, "hidden_dim": 8, "num_layers": 2}


def test_load_checkpoint_builds_resnet_bnn_nll(tmp_path):
    """`run_inference`'s hardcoded `BayesianResNetSTEC` assumption is exactly what this
    closes: the R2.2 ablation checkpoint now builds its own architecture, not the paper
    model's, from the same call site."""
    original = ResNet_BNN_NLL(n_in=7, hidden_dim=8, num_layers=2)
    checkpoint_path = tmp_path / "finetune_ResNet_BNN_NLL_seed42.pth"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint_path)

    loaded, shape = load_checkpoint(checkpoint_path)
    assert type(loaded) is ResNet_BNN_NLL
    assert shape == {"n_in": 7, "hidden_dim": 8, "num_layers": 2}

    inputs = torch.randn(4, 7)
    # Bayesian layers sample weights per forward call - seed both calls identically so this
    # compares the loaded state_dict, not sampling noise (CLAUDE.md's Bayesian A/B gotcha).
    with torch.no_grad():
        torch.manual_seed(123)
        original_out = original.eval()(inputs)
        torch.manual_seed(123)
        loaded_out = loaded(inputs)
    torch.testing.assert_close(original_out[0], loaded_out[0])
    torch.testing.assert_close(original_out[1], loaded_out[1])


def test_load_checkpoint_builds_mlp_laplacian_nll(tmp_path):
    """The VTEC baseline, via the same generic `load_checkpoint` call site."""
    original = MLP_LaplacianNLL(n_in=5, hidden_dim=6, num_layers=2)
    checkpoint_path = tmp_path / "finetune_MLP_LaplacianNLL_seed42.pth"
    torch.save({"model_state_dict": original.state_dict()}, checkpoint_path)

    loaded, shape = load_checkpoint(checkpoint_path)
    assert type(loaded) is MLP_LaplacianNLL
    assert shape == {"n_in": 5, "hidden_dim": 6, "num_layers": 2}

    inputs = torch.randn(4, 5)
    with torch.no_grad():
        original_out = original(inputs)
        loaded_out = loaded(inputs)
    torch.testing.assert_close(original_out[0], loaded_out[0])
    torch.testing.assert_close(original_out[1], loaded_out[1])
