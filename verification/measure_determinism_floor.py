"""How closely can two runs of the same model agree? Measure it, do not assume it.

The equivalence diagnostics compare a rebuilt implementation against the pre-rebuild one
and require agreement to a stated tolerance. That tolerance is only meaningful if the
*same* code, run twice, already agrees more closely than it - otherwise the diagnostic is
measuring the hardware's noise floor and calling it a refactoring error.

This measures the floor directly, on the real device, at the paper model's architecture:

  1. the same pinned model, forward twice in one process  (the zero-perturbation control)
  2. two separately constructed models with identical weights, pinned by name
  3. the same, with deterministic algorithms and TF32 disabled
  4. an unpinned Bayesian forward pass, for scale - this is the noise the pinning removes

Run it before trusting any tolerance:

    python verification/measure_determinism_floor.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchbnn as bnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.models import determinism  # noqa: E402

# The paper model: deterministic input projection, 4 residual blocks, Bayesian output.
HIDDEN_DIM = 1024
NUM_LAYERS = 4
N_IN = 92
BATCH = 4096
PRIOR_SIGMA = 0.1
STEC_MEAN_TECU = 15.5


class ResNetBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.relu(self.norm(x + residual))


class BayesianResNet(torch.nn.Module):
    """Structurally the paper's BayesianResNetSTEC: Bayesian output layer only."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(N_IN, HIDDEN_DIM), torch.nn.ReLU()
        )
        self.res_blocks = torch.nn.ModuleList(
            ResNetBlock(HIDDEN_DIM) for _ in range(NUM_LAYERS)
        )
        self.output_layer = bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=PRIOR_SIGMA,
            in_features=HIDDEN_DIM,
            out_features=2,
        )
        with torch.no_grad():
            self.output_layer.bias_mu[0].fill_(STEC_MEAN_TECU)
            self.output_layer.weight_mu.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


def build(device: torch.device, preceding_draws: int = 0) -> BayesianResNet:
    torch.manual_seed(0)
    if preceding_draws:
        _ = torch.randn(preceding_draws)
    return BayesianResNet().to(device).eval()


@torch.no_grad()
def max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max())


@torch.no_grad()
def measure(device_name: str) -> int:
    device = torch.device(device_name)
    print(f"device: {device_name}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(
        f"  architecture: {N_IN} -> {HIDDEN_DIM} x {NUM_LAYERS} residual -> Bayesian(2)"
    )
    print(f"  batch: {BATCH}\n")

    reference = build(device)
    # Constructed after a different amount of RNG consumption, then given identical
    # weights: this is what a refactor that reorders module construction looks like.
    reordered = build(device, preceding_draws=997)
    reordered.load_state_dict(reference.state_dict())

    torch.manual_seed(1)
    x = torch.randn(BATCH, N_IN, device=device)

    results: dict[str, float] = {}

    with determinism.frozen(reference, seed=7):
        results["same model, forward twice (zero-perturbation control)"] = (
            max_abs_difference(reference(x), reference(x))
        )

    with determinism.frozen(reference, seed=7), determinism.frozen(reordered, seed=7):
        results["two constructions, identical weights, pinned by name"] = (
            max_abs_difference(reference(x), reordered(x))
        )

    with determinism.deterministic_mode():
        with (
            determinism.frozen(reference, seed=7),
            determinism.frozen(reordered, seed=7),
        ):
            results["the same, deterministic algorithms + TF32 off"] = (
                max_abs_difference(reference(x), reordered(x))
            )

    determinism.unfreeze_bayesian_layers(reference)
    results["unpinned Bayesian forward, twice (the noise being removed)"] = (
        max_abs_difference(reference(x), reference(x))
    )

    width = max(len(k) for k in results)
    for label, value in results.items():
        print(f"  {label:<{width}}  {value:.3e}")

    control = results["same model, forward twice (zero-perturbation control)"]
    across = results["the same, deterministic algorithms + TF32 off"]
    noise = results["unpinned Bayesian forward, twice (the noise being removed)"]

    print()
    if control != 0.0:
        print(f"  FAIL  zero-perturbation control is {control:.3e}, must be exactly 0")
        return 1
    print("  control is exactly 0 - pinning works on this device")

    if across == 0.0:
        print("  two independent constructions agree exactly")
    else:
        print(f"  two independent constructions agree to {across:.3e}")
    print(
        f"  the pinning removes {noise:.3e} of sampling noise, ~{noise / max(across, 1e-12):.0e}x"
    )
    print()
    print(
        f"  => an equivalence tolerance of 1e-6 is {'achievable' if across < 1e-6 else 'NOT achievable'}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    return measure(args.device)


if __name__ == "__main__":
    sys.exit(main())
