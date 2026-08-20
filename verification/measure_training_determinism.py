"""Is a training run reproducible enough for Gate C to be decidable?

Gate C claims that retraining one day under the rebuilt code reproduces the pre-rebuild
result, and its verdict decides whether ~3,580 checkpoints are reused or 50-90 GPU-hours
are spent retraining. That verdict is only meaningful if the *same* code, trained twice
from the same seed, already agrees more closely than the tolerance being applied.

Forward equivalence is settled: `measure_determinism_floor` gets bit-exact agreement once
the Bayesian noise is pinned. Training is a different question, because the backward pass
introduces reductions - and on GPU some of those use atomics, whose summation order is not
fixed. Seeding does not help with that.

What this measures, over a short run of real training steps (Gaussian NLL + KL, Adam,
matching the paper's loss):

  1. the loss trajectory of two identically seeded runs
  2. the final parameters of two identically seeded runs
  3. the same, with deterministic algorithms and TF32 disabled

Data is a fixed synthetic tensor rather than a loader, deliberately: this isolates the
model, loss and optimiser: adding a DataLoader would fold worker RNG into the answer and
make it impossible to tell which part was responsible.

    python verification/measure_training_determinism.py --device cuda --steps 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchbnn as bnn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.models import determinism  # noqa: E402
from stec.models.architectures import BayesianResNetSTEC  # noqa: E402

N_IN = 127
HIDDEN_DIM = 256
NUM_LAYERS = 4
BATCH = 1024
LEARNING_RATE = 2e-4

# The paper's loss: Gaussian NLL plus a KL term annealed 0 -> 0.1 over 5 warmup epochs.
# The anneal is applied here as a fixed weight, since this measures determinism rather
# than convergence.
KL_WEIGHT = 0.1


def fixed_batch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Identical data for every run, so only the training path can differ."""
    generator = torch.Generator(device="cpu").manual_seed(12345)
    inputs = torch.randn(BATCH, N_IN, generator=generator)
    targets = torch.randn(BATCH, 1, generator=generator) * 10 + 15.5
    return inputs.to(device), targets.to(device)


def train(device: torch.device, steps: int, seed: int) -> tuple[list[float], dict]:
    """Train from a fixed seed and return the loss trajectory and final parameters."""
    torch.manual_seed(seed)
    model = BayesianResNetSTEC(
        n_in=N_IN, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS
    ).to(device)
    model.train()

    nll = torch.nn.GaussianNLLLoss(full=True)
    kl = bnn.BKLLoss(reduction="mean", last_layer_only=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    inputs, targets = fixed_batch(device)
    losses: list[float] = []

    # Seed again immediately before the loop so the weight draws inside each forward pass
    # follow a known sequence, independent of how much RNG construction consumed.
    torch.manual_seed(seed)
    for _ in range(steps):
        optimiser.zero_grad(set_to_none=True)
        mean, variance = model(inputs)
        loss = nll(mean, targets, variance) + KL_WEIGHT * kl(model)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))

    final = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return losses, final


def compare(a: tuple[list[float], dict], b: tuple[list[float], dict]) -> dict:
    losses_a, params_a = a
    losses_b, params_b = b
    loss_diff = max(abs(x - y) for x, y in zip(losses_a, losses_b, strict=True))
    param_diff = max(float((params_a[k] - params_b[k]).abs().max()) for k in params_a)
    return {
        "max |loss difference|": loss_diff,
        "max |parameter difference|": param_diff,
        "final loss": losses_a[-1],
        "relative loss difference": loss_diff / max(abs(losses_a[-1]), 1e-12),
    }


def report(label: str, result: dict) -> None:
    print(f"  {label}")
    for key, value in result.items():
        print(f"      {key:<28} {value:.6e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device: {args.device}")
    if device.type == "cuda":
        print(f"  {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
    print(f"  {args.steps} steps, batch {BATCH}, seed {args.seed}\n")

    default_mode = compare(
        train(device, args.steps, args.seed), train(device, args.steps, args.seed)
    )
    report(
        "as the stored runs were trained (cudnn.benchmark, TF32 default)", default_mode
    )

    with determinism.deterministic_mode():
        strict = compare(
            train(device, args.steps, args.seed), train(device, args.steps, args.seed)
        )
    report("deterministic algorithms, TF32 off", strict)

    different_seed = compare(
        train(device, args.steps, args.seed), train(device, args.steps, args.seed + 1)
    )
    report("a different seed, for scale", different_seed)

    print()
    floor = strict["max |parameter difference|"]
    signal = different_seed["max |parameter difference|"]
    if floor == 0.0:
        print("  Training is bit-exact under deterministic mode.")
        print(
            "  => Gate C can require exact agreement; a difference is a real difference."
        )
    else:
        print(f"  Run-to-run floor under deterministic mode: {floor:.3e}")
        print(f"  A seed change moves parameters by:         {signal:.3e}")
        print(f"  Signal-to-floor ratio: {signal / max(floor, 1e-30):.1e}")
        print(
            "  => Gate C needs a tolerance band above the floor, not exact agreement."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
