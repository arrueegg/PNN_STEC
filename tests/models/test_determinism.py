"""Whether an equivalence comparison of two Bayesian implementations is possible at all.

The load-bearing test is `test_frozen_output_survives_construction_order`: if the pinned
noise depended on how many random draws happened before a layer was built, then any
refactor that reorders module construction would produce a different posterior draw, and
comparing old code to new code to 1e-6 would be unachievable for reasons unrelated to
correctness.
"""

from __future__ import annotations

import torch
import torchbnn as bnn

from stec.models import determinism


def make_model(prior_sigma: float = 0.1) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        bnn.BayesLinear(
            prior_mu=0.0, prior_sigma=prior_sigma, in_features=8, out_features=2
        ),
    )


def inputs() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(3, 4)


def test_unpinned_forward_passes_disagree():
    """The premise: identical input, identical model, different answer."""
    torch.manual_seed(0)
    model = make_model()
    x = inputs()
    with torch.no_grad():
        spread = float((model(x) - model(x)).abs().max())
    assert spread > 0, "a Bayesian layer that does not resample invalidates these tests"


def test_zero_perturbation_control_is_exactly_zero():
    torch.manual_seed(0)
    model = make_model()
    assert determinism.zero_perturbation_control(model, inputs()) == 0.0


def test_freezing_reports_how_many_layers_it_pinned():
    torch.manual_seed(0)
    model = make_model()
    assert determinism.freeze_bayesian_layers(model, seed=1) == 1


def test_frozen_output_survives_construction_order():
    """The draw must depend on the layer's name, never on what was built before it.

    Two models with identical named layers and identical weights, but different amounts of
    RNG consumed during construction, must produce the same pinned output.
    """
    torch.manual_seed(0)
    reference = make_model()

    # Build a second model after consuming a different amount of the global RNG stream,
    # then give it the same weights. Under torchbnn's own freeze() this would draw
    # different noise; keyed by name, it must not.
    torch.manual_seed(0)
    _ = torch.randn(997)
    other = make_model()
    other.load_state_dict(reference.state_dict())

    x = inputs()
    with torch.no_grad():
        with determinism.frozen(reference, seed=7):
            a = reference(x)
        with determinism.frozen(other, seed=7):
            b = other(x)
    assert torch.equal(a, b)


def test_torchbnn_native_freeze_does_not_survive_construction_order():
    """Documents why the keyed version exists, rather than reusing freeze()."""
    torch.manual_seed(0)
    reference = make_model()
    torch.manual_seed(0)
    _ = torch.randn(997)
    other = make_model()
    other.load_state_dict(reference.state_dict())

    x = inputs()
    with torch.no_grad():
        for module in reference.modules():
            if determinism.is_bayesian_layer(module):
                module.freeze()
        a = reference(x)
        for module in other.modules():
            if determinism.is_bayesian_layer(module):
                module.freeze()
        b = other(x)
    assert not torch.equal(a, b)


def test_different_seeds_give_different_draws():
    torch.manual_seed(0)
    model = make_model()
    x = inputs()
    with torch.no_grad():
        with determinism.frozen(model, seed=1):
            a = model(x)
        with determinism.frozen(model, seed=2):
            b = model(x)
    assert not torch.equal(a, b)


def test_unfreeze_restores_sampling():
    torch.manual_seed(0)
    model = make_model()
    with determinism.frozen(model, seed=1):
        pass
    x = inputs()
    with torch.no_grad():
        assert float((model(x) - model(x)).abs().max()) > 0


def test_monte_carlo_is_reproducible_for_a_seed():
    torch.manual_seed(0)
    model = make_model()
    x = inputs()
    first = determinism.monte_carlo(model, x, samples=16, seed=123)
    second = determinism.monte_carlo(model, x, samples=16, seed=123)
    assert torch.equal(first, second)
    assert first.shape[0] == 16


def test_monte_carlo_samples_actually_differ():
    """A reproducible average still has to be an average over genuinely different draws."""
    torch.manual_seed(0)
    model = make_model()
    draws = determinism.monte_carlo(model, inputs(), samples=8, seed=1)
    assert float(draws.std(dim=0).max()) > 0


def test_monte_carlo_seeds_are_independent():
    torch.manual_seed(0)
    model = make_model()
    x = inputs()
    a = determinism.monte_carlo(model, x, samples=8, seed=1)
    b = determinism.monte_carlo(model, x, samples=8, seed=2)
    assert not torch.equal(a, b)


def test_deterministic_mode_restores_previous_settings():
    before = (
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    with determinism.deterministic_mode():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cuda.matmul.allow_tf32 is False
    after = (
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.get_float32_matmul_precision(),
    )
    assert before == after
