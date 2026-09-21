"""What a model can do, declared by the model rather than sniffed from its name.

The pipeline decides several things from the architecture: whether to draw Monte Carlo
samples, whether the second output is a variance or a scale, and which distribution to
score against. All of it is currently inferred from substrings of the model type::

    if "Laplacian" in model_type: ...
    is_bayesian = "BNN" in model_type or "Bayesian" in model_type

spread across `model.py`, `inference_manager.py`, `base_trainer.py` and `collation.py`,
each with its own slightly different spelling of the test. A model whose name does not
match is silently treated as deterministic - it gets one forward pass instead of a hundred
- and nothing reports that anything was decided.

The distribution matters more than it looks. The VTEC baseline predicts a Laplace *scale*,
not a standard deviation: the same data reads 90% coverage at nominal 50% under Gaussian
quantiles against 82% under Laplace. Scoring it as a Gaussian is not a small error.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    """How the pipeline must treat a model's outputs."""

    # Weights are sampled per forward pass, so a single call is one draw from the
    # posterior rather than the prediction. Consumers must average over samples, and any
    # A/B comparison must pin the draws first.
    samples_weights: bool

    # The forward pass returns (location, spread) rather than a point estimate.
    predicts_spread: bool

    # How to read the spread: "variance" for the STEC models (softplus + floor),
    # "scale" for the Laplace VTEC baseline, where variance is 2 * scale**2.
    spread_kind: str = "variance"

    # The likelihood the spread parameterises, and therefore the quantiles to score with.
    distribution: str = "gaussian"

    def monte_carlo_samples(self, requested: int) -> int:
        """How many forward passes this model actually needs.

        Replaces `num_mc_samples = 100 if bayesian else 1`, which was copy-pasted with its
        detection logic into four files.
        """
        return requested if self.samples_weights else 1


GAUSSIAN_BAYESIAN = Capabilities(
    samples_weights=True,
    predicts_spread=True,
    spread_kind="variance",
    distribution="gaussian",
)

LAPLACE_DETERMINISTIC = Capabilities(
    samples_weights=False,
    predicts_spread=True,
    spread_kind="scale",
    distribution="laplace",
)

DETERMINISTIC_POINT = Capabilities(samples_weights=False, predicts_spread=False)
