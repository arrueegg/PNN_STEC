"""The uncertainty decomposition, and the two failure modes it must not repeat.

`epistemic = Var_t(mean_t)`, `aleatoric = E_t(variance_t)` looks simple enough to get right
by eye, but the source picked its `unbiased` flag and its sample count from a model-name
substring (`"Laplacian" in model_type`), and its sampling loop was never seeded. Both are
load-bearing here: `test_capabilities_drive_sample_count` pins the replacement for the
former, `test_seeded_draws_reproduce` / `test_different_seeds_give_different_draws` pin the
replacement for the latter.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchbnn as bnn
from torch import nn

from stec.inference import monte_carlo
from stec.models import capabilities as caps
from stec.models import determinism
from stec.models.architectures import DeepEnsemble


class FixedSequenceModel(nn.Module):
    """Returns a pre-scripted (mean, spread) pair per call, cycling if exhausted.

    Not a real network - a stand-in for "T stochastic passes with known outputs" so the
    decomposition arithmetic can be checked exactly, independent of what any actual Bayesian
    layer happens to sample.
    """

    def __init__(self, pairs: list[tuple[float, float]]) -> None:
        super().__init__()
        self.pairs = pairs
        self.calls = 0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_value, spread_value = self.pairs[self.calls % len(self.pairs)]
        self.calls += 1
        batch = x.shape[0]
        mean = torch.full((batch, 1), mean_value)
        spread = torch.full((batch, 1), spread_value)
        return mean, spread


class TinyBayesianSTEC(nn.Module):
    """A miniature `BayesianResNetSTEC`: one `BayesLinear` head, split into (mean, variance).

    Real enough to exercise `determinism.monte_carlo`'s seeding through an actual Bayesian
    layer, without pulling in the full architecture.
    """

    capabilities = caps.GAUSSIAN_BAYESIAN

    def __init__(self) -> None:
        super().__init__()
        self.layer = bnn.BayesLinear(
            prior_mu=0.0, prior_sigma=0.1, in_features=4, out_features=2
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.layer(x)
        mean, log_var = torch.split(out, 1, dim=1)
        variance = F.softplus(log_var) + 1e-3
        return mean, variance


def tiny_inputs() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(3, 4)


def test_decomposition_matches_known_per_pass_values():
    """Three passes with known (mean, variance): every quantity checked by hand."""
    means = [10.0, 12.0, 14.0]
    variances = [1.0, 2.0, 3.0]
    model = FixedSequenceModel(list(zip(means, variances)))

    result = monte_carlo.monte_carlo_uncertainty(
        model, torch.zeros(2, 1), caps.GAUSSIAN_BAYESIAN, requested_samples=3, seed=0
    )

    expected_mean = sum(means) / 3
    expected_aleatoric_var = sum(variances) / 3
    sample_mean = expected_mean
    expected_epistemic_var = sum((m - sample_mean) ** 2 for m in means) / (3 - 1)
    expected_total_var = expected_aleatoric_var + expected_epistemic_var

    assert torch.allclose(result.mean, torch.full((2, 1), expected_mean))
    assert torch.allclose(
        result.aleatoric_std, torch.full((2, 1), math.sqrt(expected_aleatoric_var))
    )
    assert torch.allclose(
        result.epistemic_std, torch.full((2, 1), math.sqrt(expected_epistemic_var))
    )
    assert torch.allclose(
        result.total_std, torch.full((2, 1), math.sqrt(expected_total_var))
    )
    assert result.samples == 3


def test_epistemic_vanishes_when_every_pass_agrees():
    model = FixedSequenceModel([(7.0, 2.5)])
    result = monte_carlo.monte_carlo_uncertainty(
        model, torch.zeros(4, 1), caps.GAUSSIAN_BAYESIAN, requested_samples=5, seed=0
    )
    assert torch.equal(result.epistemic_std, torch.zeros(4, 1))
    assert torch.allclose(result.aleatoric_std, torch.full((4, 1), math.sqrt(2.5)))
    assert torch.equal(result.total_std, result.aleatoric_std)


def test_single_sample_gives_zero_epistemic_not_nan():
    """torch.var(unbiased=True) over one draw is 0/0; the source special-cases this."""
    model = FixedSequenceModel([(3.0, 0.5)])
    result = monte_carlo.monte_carlo_uncertainty(
        model, torch.zeros(2, 1), caps.GAUSSIAN_BAYESIAN, requested_samples=1, seed=0
    )
    assert result.samples == 1
    assert torch.equal(result.epistemic_std, torch.zeros(2, 1))
    assert not torch.isnan(result.epistemic_std).any()


def test_capabilities_drive_sample_count_not_a_model_name():
    bayesian = FixedSequenceModel([(1.0, 1.0), (2.0, 1.0)])
    result = monte_carlo.monte_carlo_uncertainty(
        bayesian,
        torch.zeros(1, 1),
        caps.GAUSSIAN_BAYESIAN,
        requested_samples=100,
        seed=0,
    )
    assert result.samples == 100

    laplace = FixedSequenceModel([(1.0, 1.0)])
    result = monte_carlo.monte_carlo_uncertainty(
        laplace,
        torch.zeros(1, 1),
        caps.LAPLACE_DETERMINISTIC,
        requested_samples=100,
        seed=0,
    )
    assert result.samples == 1


def test_deterministic_point_capability_is_rejected():
    """No spread head means no decomposition to compute; fail loudly, not with zeros."""
    model = FixedSequenceModel([(1.0, 1.0)])
    try:
        monte_carlo.monte_carlo_uncertainty(
            model,
            torch.zeros(1, 1),
            caps.DETERMINISTIC_POINT,
            requested_samples=10,
            seed=0,
        )
    except ValueError as error:
        assert "predicts_spread=False" in str(error)
    else:
        raise AssertionError("expected a ValueError for a non-spread-predicting model")


def test_laplace_scale_is_converted_to_variance_before_aggregation():
    """spread_kind='scale' must go through variance = 2*b**2, not be treated as a variance."""
    scale = 2.0
    model = FixedSequenceModel([(5.0, scale)])
    result = monte_carlo.monte_carlo_uncertainty(
        model,
        torch.zeros(1, 1),
        caps.LAPLACE_DETERMINISTIC,
        requested_samples=10,
        seed=0,
    )
    expected_variance = 2.0 * scale**2
    assert torch.allclose(
        result.aleatoric_std, torch.full((1, 1), math.sqrt(expected_variance))
    )


def test_laplace_scale_to_variance_formula():
    scale = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(monte_carlo.laplace_scale_to_variance(scale), 2.0 * scale**2)


def test_scale_cannot_be_silently_used_as_a_std():
    """There is no function that hands back a scale relabelled as a sigma."""
    assert not hasattr(monte_carlo, "laplace_scale_to_std")
    scale = torch.tensor([2.0])
    variance = monte_carlo.spread_to_variance(scale, caps.LAPLACE_DETERMINISTIC)
    # The only sanctioned reading is variance; using `scale` itself as a std would give 2.0,
    # not sqrt(2 * 2**2) = 2.828... - the two must disagree, or the guard is not doing anything.
    assert not torch.allclose(torch.sqrt(variance), scale)


def test_unknown_spread_kind_is_rejected():
    bogus = caps.Capabilities(
        samples_weights=False, predicts_spread=True, spread_kind="std"
    )
    try:
        monte_carlo.spread_to_variance(torch.tensor([1.0]), bogus)
    except ValueError as error:
        assert "spread_kind" in str(error)
    else:
        raise AssertionError("expected a ValueError for an unrecognised spread_kind")


def test_seeded_draws_reproduce():
    torch.manual_seed(0)
    model = TinyBayesianSTEC()
    x = tiny_inputs()
    first = monte_carlo.monte_carlo_uncertainty(
        model, x, caps.GAUSSIAN_BAYESIAN, requested_samples=16, seed=123
    )
    second = monte_carlo.monte_carlo_uncertainty(
        model, x, caps.GAUSSIAN_BAYESIAN, requested_samples=16, seed=123
    )
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.total_std, second.total_std)


def test_different_seeds_give_different_draws():
    torch.manual_seed(0)
    model = TinyBayesianSTEC()
    x = tiny_inputs()
    first = monte_carlo.monte_carlo_uncertainty(
        model, x, caps.GAUSSIAN_BAYESIAN, requested_samples=16, seed=1
    )
    second = monte_carlo.monte_carlo_uncertainty(
        model, x, caps.GAUSSIAN_BAYESIAN, requested_samples=16, seed=2
    )
    assert not torch.equal(first.mean, second.mean)


def test_zero_perturbation_control_is_exactly_zero_through_the_adapter():
    """The A/B-testing invariant from `determinism.py`, exercised through this module's
    own output-pairing wrapper rather than a bare model call."""
    torch.manual_seed(0)
    model = TinyBayesianSTEC()
    x = tiny_inputs()
    wrapped = monte_carlo._PairedOutputAdapter(model)
    with determinism.frozen(model, seed=7):
        first = wrapped(x)
        second = wrapped(x)
    assert float((first - second).abs().max()) == 0.0


def test_mismatched_mean_and_spread_shapes_are_rejected():
    class Mismatched(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1), torch.zeros(x.shape[0], 2)

    adapter = monte_carlo._PairedOutputAdapter(Mismatched())
    try:
        adapter(torch.zeros(3, 4))
    except ValueError as error:
        assert "different shapes" in str(error)
    else:
        raise AssertionError("expected a ValueError for mismatched mean/spread shapes")


# --- ensemble_uncertainty: DeepEnsemble.get_uncertainties, chunked -------------------------


def test_ensemble_uncertainty_matches_get_uncertainties_unchunked():
    """The whole point of this function is to add row-chunking without changing a single
    number `DeepEnsemble.get_uncertainties` itself would produce."""
    members = [FixedSequenceModel([(mean, 1.0)]) for mean in (10.0, 12.0, 14.0)]
    ensemble = DeepEnsemble(members, model_type="Gaussian")
    inputs = torch.zeros(5, 1)

    ref_mean, ref_alea_var, ref_epi_var, ref_total_var = ensemble.get_uncertainties(
        inputs
    )
    result = monte_carlo.ensemble_uncertainty(ensemble, inputs, batch_size=1000)

    assert torch.allclose(result.mean, ref_mean)
    assert torch.allclose(result.aleatoric_std, torch.sqrt(ref_alea_var))
    assert torch.allclose(result.epistemic_std, torch.sqrt(ref_epi_var))
    assert torch.allclose(result.total_std, torch.sqrt(ref_total_var))
    assert result.samples == 3


def test_ensemble_uncertainty_row_chunking_does_not_change_the_result():
    """Every member is a fixed, already-trained network with no per-call randomness, so
    splitting the row dimension across several forward calls must reproduce the single
    unbatched call exactly - the same invariant `determinism.monte_carlo` proves for a
    Bayesian forward pass."""
    torch.manual_seed(0)
    members = [nn.Linear(4, 2) for _ in range(3)]

    class WrapLinear(nn.Module):
        def __init__(self, linear: nn.Linear) -> None:
            super().__init__()
            self.linear = linear

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            out = self.linear(x)
            mean, raw_spread = torch.split(out, 1, dim=1)
            return mean, F.softplus(raw_spread) + 1e-3

    ensemble = DeepEnsemble([WrapLinear(m) for m in members], model_type="Gaussian")
    inputs = torch.randn(37, 4)  # not a multiple of any chunk size below

    unchunked = monte_carlo.ensemble_uncertainty(ensemble, inputs, batch_size=1000)
    chunked = monte_carlo.ensemble_uncertainty(ensemble, inputs, batch_size=10)

    assert torch.equal(unchunked.mean, chunked.mean)
    assert torch.equal(unchunked.epistemic_std, chunked.epistemic_std)
    assert torch.equal(unchunked.aleatoric_std, chunked.aleatoric_std)
    assert torch.equal(unchunked.total_std, chunked.total_std)


def test_ensemble_uncertainty_laplacian_uses_population_variance():
    """`DeepEnsemble.get_uncertainties`'s own `unbiased=not is_laplacian` (Mao et al.
    2025); this only checks the function this module adds still reaches that branch."""
    members = [FixedSequenceModel([(mean, 1.0)]) for mean in (1.0, 2.0)]
    ensemble = DeepEnsemble(members, model_type="Laplacian")
    inputs = torch.zeros(1, 1)

    result = monte_carlo.ensemble_uncertainty(ensemble, inputs, batch_size=1000)

    # Population variance (ddof=0) of {1.0, 2.0} is 0.25, not the unbiased 0.5.
    assert torch.allclose(result.epistemic_std, torch.full((1, 1), 0.5))
