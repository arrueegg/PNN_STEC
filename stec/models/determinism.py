"""Make a Bayesian model's output reproducible, so two implementations can be compared.

`BayesianResNetSTEC`'s output layer samples fresh weights on **every forward call**, so
`model(a)` and `model(b)` differ by ~1.4 TECU of pure sampling noise even when `a is b`. A
sensitivity test built without accounting for that measures nothing: the zero-perturbation
control once came out *larger* than the perturbed runs, and the spurious 0.33 TECU it
produced was used to reject a correct approach for days.

That makes any comparison of two implementations meaningless unless the sampling is pinned
first. `torchbnn` offers `freeze()`, but it draws from the **global** generator, so the
noise a layer ends up with depends on how many random numbers were consumed before it -
which is a function of module construction order. A refactor that reorders instantiation
would therefore produce a *different posterior draw* rather than a numerically close one,
and "matches to 1e-6" would be unachievable for reasons that have nothing to do with
correctness.

`freeze_bayesian_layers` keys each layer's noise to a generator seeded from the layer's
**name**, so the draw is independent of construction order, of how many layers precede it,
and of anything the process did earlier. Two implementations that build the same named
layers agree exactly, whatever order they build them in. That is what makes an equivalence
comparison possible at all.

Two modes, for two different questions:

* **frozen** - "are these two implementations the same function?" One fixed draw, forward
  is deterministic. Used by the model-equivalence diagnostic.
* **seeded Monte Carlo** - "does this reproduce the stored predictions?" The *sequence* of
  draws is reproducible, so a T-sample average can be recomputed. Used by the inference
  diagnostic.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import warnings
from collections.abc import Iterator

import torch

# torchbnn marks its Bayesian layers with these buffers. Detecting by attribute rather
# than by class keeps this working if the layer is wrapped or subclassed, and avoids the
# `"BNN" in model_type` string sniffing that the rebuild is removing.
_WEIGHT_NOISE = "weight_eps"
_BIAS_NOISE = "bias_eps"


def is_bayesian_layer(module: torch.nn.Module) -> bool:
    return hasattr(module, _WEIGHT_NOISE) and hasattr(module, "weight_log_sigma")


def _name_seed(name: str, seed: int) -> int:
    """A per-layer seed that depends on the layer's name, never on its position.

    Hashing the name rather than using a counter is the whole point: a counter would
    reintroduce the construction-order dependence this function exists to remove.
    """
    digest = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    # torch generators take a 64-bit seed; take it from the digest so it is stable across
    # processes, unlike Python's salted hash().
    return int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF


def freeze_bayesian_layers(model: torch.nn.Module, seed: int = 0) -> int:
    """Pin every Bayesian layer's noise to a draw that depends only on its name.

    Returns the number of layers pinned, so a caller can assert it found any at all - a
    silently unfrozen model would produce a comparison that looks noisy for no visible
    reason.
    """
    frozen = 0
    for name, module in model.named_modules():
        if not is_bayesian_layer(module):
            continue
        generator = torch.Generator(device="cpu").manual_seed(_name_seed(name, seed))

        reference = module.weight_log_sigma
        module.weight_eps = torch.randn(
            reference.shape, generator=generator, dtype=reference.dtype
        ).to(reference.device)

        bias_reference = getattr(module, "bias_log_sigma", None)
        if bias_reference is not None:
            module.bias_eps = torch.randn(
                bias_reference.shape, generator=generator, dtype=bias_reference.dtype
            ).to(bias_reference.device)
        frozen += 1
    return frozen


def unfreeze_bayesian_layers(model: torch.nn.Module) -> int:
    """Restore per-call sampling, which is what Monte Carlo inference needs."""
    thawed = 0
    for module in model.modules():
        if not is_bayesian_layer(module):
            continue
        module.weight_eps = None
        if hasattr(module, _BIAS_NOISE):
            module.bias_eps = None
        thawed += 1
    return thawed


def resample_bayesian_layers(model: torch.nn.Module) -> int:
    """Draw one fresh weight (and bias) sample per Bayesian layer from the *global* RNG,
    then freeze the layer to that draw so several forward calls can reuse it instead of
    each one sampling its own, independent noise.

    This draws exactly what an ordinary *unfrozen* `model(x)` already draws internally -
    `torchbnn`'s `BayesLinear.forward` calls `torch.randn_like(self.weight_log_sigma)`
    itself whenever `weight_eps is None`, and that shape is `(out_features, in_features)`,
    independent of how many rows are in the batch. Doing that draw here, once, before a
    group of forward calls that all reuse it, consumes the global RNG in the same order and
    by the same amount a single unbatched call would - which is what lets `monte_carlo`'s
    row-chunked path reproduce the unbatched one exactly (see its docstring) instead of
    diverging into a second, independent sampling scheme.
    """
    resampled = 0
    for _name, module in model.named_modules():
        if not is_bayesian_layer(module):
            continue
        module.weight_eps = torch.randn_like(module.weight_log_sigma)
        bias_reference = getattr(module, "bias_log_sigma", None)
        if bias_reference is not None:
            module.bias_eps = torch.randn_like(bias_reference)
        resampled += 1
    return resampled


@contextlib.contextmanager
def frozen(model: torch.nn.Module, seed: int = 0) -> Iterator[int]:
    """Temporarily pin the sampling, then restore it."""
    count = freeze_bayesian_layers(model, seed)
    try:
        yield count
    finally:
        unfreeze_bayesian_layers(model)


@contextlib.contextmanager
def deterministic_mode(enabled: bool = True) -> Iterator[None]:
    """Remove the remaining sources of run-to-run difference on GPU.

    The stored runs were produced with none of this: no `deterministic` or `debug` key was
    set in any saved config, so cuDNN chose algorithms by benchmark and TF32 matmul stayed
    at the PyTorch default. That is why the equivalence diagnostics compare old code to new
    code *both re-run now*, rather than either to a historical artifact.

    `warn_only` keeps this usable: some ops have no deterministic implementation, and a
    hard failure would make the diagnostics impossible to run at all rather than merely
    less precise. What the floor actually is, is a measurement - see the determinism
    report - not an assumption.
    """
    if not enabled:
        yield
        return

    previous = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "float32_precision": torch.get_float32_matmul_precision(),
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    # cuBLAS reads CUBLAS_WORKSPACE_CONFIG when its handle is first created, which happens
    # on the first CUDA GEMM. Setting it here is therefore too late in any process that has
    # already run one, and torch says so at the first backward pass. Setting it anyway is
    # still right - it takes effect in a process that has not yet touched CUDA - but a
    # caller who needs the guarantee has to set it before starting python, so say so rather
    # than leave a warning from a lower layer as the only sign.
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ and torch.cuda.is_initialized():
        warnings.warn(
            "CUBLAS_WORKSPACE_CONFIG was not set before CUDA was initialised, so cuBLAS "
            "GEMM determinism is not guaranteed in this process. Export "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting python if an exact "
            "comparison depends on it.",
            RuntimeWarning,
            stacklevel=3,
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = previous["cudnn_deterministic"]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.set_float32_matmul_precision(previous["float32_precision"])
        if previous["cublas_workspace"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous["cublas_workspace"]


@torch.no_grad()
def monte_carlo(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    samples: int,
    seed: int,
    batch_size: int | None = None,
) -> torch.Tensor:
    """`samples` stochastic forward passes, reproducible for a given seed.

    Seeding once before the loop - rather than once per process, as the current inference
    path does - is what makes the resulting average recomputable. The stored predictions
    were produced without it, so they are one unrepeatable realisation of the posterior;
    reproducing them requires re-running both sides, not comparing against the file.

    `batch_size` splits each pass across the row dimension into chunks of that many rows,
    instead of one CUDA allocation sized for the whole input - a 2,036,513-row Madrigal day
    through this model's 1024-wide hidden layer asks for a single ~7.8 GiB activation
    tensor unbatched, which does not fit a 12 GB card once anything else has claimed
    memory. Chunking does not change a single number this returns: `resample_bayesian_layers`
    draws each pass's weights once, from the same position in the global RNG stream a plain
    unfrozen `model(inputs)` call would have consumed, then freezes them so every chunk of
    that pass reuses the identical draw. `F.linear` and this architecture's per-row
    `LayerNorm` are both row-independent (no batch norm, and `nn.Dropout` is a no-op in
    `model.eval()` mode, which every caller here uses), so slicing the batch dimension
    changes nothing about the arithmetic any individual row goes through - only how many
    rows go through the network in one allocation. `batch_size=None` (the default) is one
    chunk covering every row: mechanically the same code path a smaller `batch_size` takes,
    not a separate branch that happens to agree with it, which is what lets a caller that
    never sets it keep exactly today's behaviour.

    Returns a `(samples, ...)` tensor rather than the mean, because the caller usually
    needs the spread too, and computing it from the stack costs nothing.
    """
    unfreeze_bayesian_layers(model)
    torch.manual_seed(seed)
    rows = inputs.shape[0]
    chunk_size = batch_size or rows
    passes = []
    for _ in range(samples):
        resample_bayesian_layers(model)
        chunk_outputs = [
            model(inputs[start : start + chunk_size])
            for start in range(0, rows, chunk_size)
        ]
        passes.append(torch.cat(chunk_outputs, dim=0))
        unfreeze_bayesian_layers(model)
    return torch.stack(passes)


@torch.no_grad()
def zero_perturbation_control(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    seed: int = 0,
) -> float:
    """Feed identical inputs through the pinned model twice; must be exactly 0.0.

    Every A/B comparison in this repository runs this first. A sensitivity study that
    skipped it once measured 1.4 TECU of sampling noise and reported it as signal.
    """
    with frozen(model, seed):
        first = model(inputs)
        second = model(inputs)
    return float((first - second).abs().max())
