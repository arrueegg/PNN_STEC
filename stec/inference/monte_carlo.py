"""Monte Carlo inference: predictive mean and its uncertainty decomposition.

Ported from `InferenceManager.bayesian_inference_total_uncertainty` in the old
`src/training/inference_manager.py` (the sampling loop and aggregation are around lines
142-191 there). That function decided how many stochastic forward passes to draw, and how
to read the model's second output head, from `model_type` substring checks::

    is_mc_dropout = model_type in ["MLP_MCDropout_NLL", "MLP_MCDropout_mse"]
    if "Laplacian" in model_type: var_raw = 2.0 * (var_raw ** 2)
    is_laplacian = "Laplacian" in model_type

each copy-pasted with its own slightly different spelling into `model.py`,
`inference_manager.py`, `base_trainer.py` and `collation.py` (see
`stec/models/capabilities.py`, which this module reads instead of a model name). Sample
count now comes from `Capabilities.monte_carlo_samples`, never from a substring.

**Decomposition, verified against the source** (`inference_manager.py:177-191`)::

    pred_stack    = stack of T per-pass means                    # [T, ...]
    alea_stack    = stack of T per-pass (converted) variances    # [T, ...]
    stec_mean     = pred_stack.mean(dim=0)
    epistemic_var = pred_stack.var(dim=0, unbiased=not is_laplacian)   # 0 if T == 1
    aleatoric_var = alea_stack.mean(dim=0)
    total_var     = epistemic_var + aleatoric_var

and the columns the source writes are **standard deviations, not variances** - it takes
`torch.sqrt(...)` at the point it builds `pred_epistemic_unc` / `pred_aleatoric_unc` /
`pred_total_unc` (`inference_manager.py:219-223` for the per-batch path, `:397-406` for the
large-dataset path, `:450-452` for the returned summary). `pred_stec` / `stec_mean` is the
one column that is not converted - it is already the target quantity, not a spread. This
module returns sigmas for the same reason: they are what `prediction_store`'s
`pred_total_unc` / `pred_epistemic_unc` / `pred_aleatoric_unc` columns expect.

The source's `unbiased=not is_laplacian` used Bessel's correction (`ddof=1`) for the
Gaussian/BNN case and population variance (`ddof=0`) for the Laplace ensemble, decided by
the same `"Laplacian" in model_type` substring this rebuild removes elsewhere. Here that
becomes `capabilities.distribution != "laplace"` - the same decision, read from a
declaration instead of a name.

**Not ported: the log-target moment mapping.** For a model trained on `log(stec)`, the
source maps each pass's (mean, variance) from log-space to linear-space via log-normal
moments before this decomposition runs (`inference_manager.py:156-164`)::

    mean_y     = exp(mean_raw + 0.5 * var_raw) - eps
    var_alea_y = (exp(var_raw) - 1) * exp(2 * mean_raw + var_raw)

`BayesianResNetSTEC.forward` in the rebuilt `stec/models/architectures.py` returns
`(mean, variance)` already in linear/target space and declares no log-target capability, so
there is nothing yet in this codebase for that mapping to feed into. If a log-target model
is ported later, that conversion belongs between the model's raw output and the
`(mean_t, variance_t)` pairs this module consumes, not inside this module.

**The stored predictions are not reproducible.** The source called `model(inputs)` in a
sampling loop with no seed set between calls (`inference_manager.py:146-147`), so every
`pred_*_unc` value already on disk is a 100-draw average over one unrepeatable realisation
of the posterior - the same defect `stec/models/determinism.py` exists to fix for
model-equivalence checks. `monte_carlo_uncertainty` below seeds the draws
(`determinism.monte_carlo`), so calling it twice with the same seed reproduces exactly - but
neither call reproduces the historical parquet, because that file was never seeded in the
first place. A Gate D check that wants to validate this port must therefore re-run *both*
sides of a comparison fresh, with an explicit seed; comparing against the stored file tests
nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models import determinism
from ..models.capabilities import Capabilities

# A single unbatched forward pass over a ~2M-row Madrigal day asks for one activation
# tensor sized rows x hidden_dim (1024) in float32 - ~7.8 GiB for 2,036,513 rows, which is
# the exact allocation that raised torch.OutOfMemoryError against a 12 GB card shared with
# a 30 GB desktop session (CLAUDE.md's memory gotchas). 50,000 rows keeps that same tensor
# under ~200 MiB, roughly 40x below the failure point, leaving real headroom for cuBLAS
# workspace, the model weights, and this module's own T-sample accumulation.
DEFAULT_INFERENCE_BATCH_SIZE = 50_000


class _PairedOutputAdapter(torch.nn.Module):
    """Wraps a `(mean, spread) -> Tensor` model so `determinism.monte_carlo` can stack it.

    That function does `torch.stack([model(inputs) for _ in range(samples)])`, which needs
    each call to return a single Tensor; `BayesianResNetSTEC.forward` (and the Laplace
    baseline) return a 2-tuple instead. Stacking the pair onto a new trailing axis - rather
    than concatenating along an existing one - works regardless of whether `mean` and
    `spread` carry a trailing size-1 dimension, so this makes no assumption about the
    model's output shape beyond "mean and spread match".

    Wrapping as a submodule (not a bare closure) matters too: `determinism.monte_carlo`
    calls `unfreeze_bayesian_layers(model)` before sampling, which walks `model.modules()`.
    Registering the wrapped model as `self.model` keeps its Bayesian layers reachable
    through that walk; a plain function would hide them.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, spread = self.model(x)
        if mean.shape != spread.shape:
            raise ValueError(
                f"model's mean and spread outputs have different shapes: "
                f"{tuple(mean.shape)} vs {tuple(spread.shape)}"
            )
        return torch.stack([mean, spread], dim=-1)


def laplace_scale_to_variance(scale: torch.Tensor) -> torch.Tensor:
    """Laplace variance = 2 * scale**2 (Mao et al. 2025; `inference_manager.py:154`).

    There is deliberately no `laplace_scale_to_std` next to this: naming the only converter
    after variance, not sigma, means a caller who wants a comparable standard deviation has
    to write `torch.sqrt(laplace_scale_to_variance(scale))` and see the conversion happen,
    rather than reaching for a function whose name would let a raw scale be mistaken for a
    std. Scoring the VTEC baseline's scale as if it already were a Gaussian sigma reads 90%
    empirical coverage at a nominal 50% quantile, against 82% under the correct Laplace
    quantiles - not a small error.
    """
    return 2.0 * scale**2


def spread_to_variance(
    spread: torch.Tensor, capabilities: Capabilities
) -> torch.Tensor:
    """The one sanctioned path from a model's second output to a variance.

    Reads `capabilities.spread_kind` instead of trusting the caller to already know whether
    this particular head predicts a variance or a scale - that ambiguity is exactly what let
    the VTEC baseline's scale be dropped from the prediction store for weeks (see
    `stec/inference/prediction_store.py`).
    """
    if capabilities.spread_kind == "variance":
        return spread
    if capabilities.spread_kind == "scale":
        return laplace_scale_to_variance(spread)
    raise ValueError(f"unknown spread_kind {capabilities.spread_kind!r}")


@dataclass(frozen=True)
class UncertaintyDecomposition:
    """Predictive mean and its epistemic/aleatoric/total spread, as sigmas (std).

    Field names and units match `prediction_store`'s `pred_stec` / `pred_epistemic_unc` /
    `pred_aleatoric_unc` / `pred_total_unc` columns. Tensors keep whatever trailing shape
    the model's own `(mean, spread)` outputs had (typically `(batch, 1)` for
    `BayesianResNetSTEC`); this module does not squeeze them, so a caller sees exactly what
    the model produced rather than a reshape it did not ask for.
    """

    mean: torch.Tensor
    epistemic_std: torch.Tensor
    aleatoric_std: torch.Tensor
    total_std: torch.Tensor
    samples: int  # how many stochastic passes were actually drawn


def monte_carlo_uncertainty(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    capabilities: Capabilities,
    requested_samples: int,
    seed: int,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
) -> UncertaintyDecomposition:
    """Predictive mean and (epistemic, aleatoric, total) sigma over stochastic passes.

    `requested_samples` is a ceiling, not a guarantee: `capabilities.monte_carlo_samples`
    collapses it to 1 for a model that does not sample weights, so a deterministic-point or
    Laplace-scale baseline gets exactly one forward pass and zero epistemic spread, matching
    the source's `num_mc_samples = 1 if not is_bayesian else 100`.

    Sampling goes through `determinism.monte_carlo`, so the same `seed` reproduces the same
    draws - see the module docstring for why that is a difference from, not a match to, the
    stored predictions. `batch_size` is forwarded unchanged: it only bounds how many rows go
    through the network in one CUDA allocation per stochastic pass, and does not change any
    number this function returns (see `determinism.monte_carlo`'s docstring for why).
    """
    if not capabilities.predicts_spread:
        raise ValueError(
            "monte_carlo_uncertainty needs a model that predicts (mean, spread); got a "
            f"model declaring predicts_spread=False ({capabilities})"
        )

    samples = capabilities.monte_carlo_samples(requested_samples)
    stacked = determinism.monte_carlo(
        _PairedOutputAdapter(model),
        inputs,
        samples=samples,
        seed=seed,
        batch_size=batch_size,
    )  # [T, ..., 2]
    mean_per_pass = stacked[..., 0]  # [T, ...]
    variance_per_pass = spread_to_variance(stacked[..., 1], capabilities)

    predictive_mean = mean_per_pass.mean(dim=0)
    aleatoric_var = variance_per_pass.mean(dim=0)

    if samples == 1:
        # torch.var(unbiased=True) over a single sample divides by zero. A lone draw
        # carries no information about spread across draws by construction, so this is
        # exactly zero rather than NaN - source: `inference_manager.py:182-185`.
        epistemic_var = torch.zeros_like(predictive_mean)
    else:
        # Bessel's correction for the Gaussian/BNN posterior, population variance for the
        # Laplace ensemble (`inference_manager.py:187-190`), keyed on
        # `capabilities.distribution` instead of the `"Laplacian" in model_type` substring
        # used there.
        unbiased = capabilities.distribution != "laplace"
        epistemic_var = mean_per_pass.var(dim=0, unbiased=unbiased)

    total_var = epistemic_var + aleatoric_var

    return UncertaintyDecomposition(
        mean=predictive_mean,
        epistemic_std=torch.sqrt(epistemic_var),
        aleatoric_std=torch.sqrt(aleatoric_var),
        total_std=torch.sqrt(total_var),
        samples=samples,
    )
