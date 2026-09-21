"""Gate B: is the rebuilt model class the same function as the one it replaces?

Loads a real checkpoint into both the pre-rebuild class (`src/model/model.py`) and the
rebuilt one (`stec/models/architectures.py`), pins the Bayesian sampling in both so the
comparison is of the *functions* rather than of two posterior draws, and reports the
largest disagreement over a batch.

This is a diagnostic, not a blocker, and the distinction matters. A match proves the two
implementations are consistent; it does not prove either is correct, because a refactor
preserves the logic it ports. What it catches is the wiring error - a transposed
dimension, a dropped residual, a renamed parameter silently initialised instead of loaded -
which is exactly the class of mistake a port introduces.

    python verification/gate_b_model_equivalence.py --checkpoint <path>
    python verification/gate_b_model_equivalence.py --run-id <id-from-alias-index>

Agreement must be bit-exact: `measure_determinism_floor` establishes that two independent
constructions with identical weights agree to 0.0 on this hardware, so anything larger is
a real difference and not numerical noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.models import architectures, determinism  # noqa: E402

# The pre-rebuild package. Imported from the live checkout by default so the comparison is
# against the code that produced the published numbers, not against a copy of it.
DEFAULT_LEGACY_SRC = Path("/scratch2/arrueegg/WP4/PNN_STEC/src")

BATCH = 4096
PIN_SEED = 7


def load_legacy_class(legacy_src: Path):
    sys.path.insert(0, str(legacy_src))
    from model.model import BayesianResNetSTEC as LegacyModel  # noqa: PLC0415

    return LegacyModel


def resolve_checkpoint(args) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)

    import csv  # noqa: PLC0415

    with Path(args.alias_index).open() as handle:
        for row in csv.DictReader(handle):
            if row["run_id"] == args.run_id:
                if not row["checkpoint"]:
                    raise SystemExit(f"run {args.run_id} has no checkpoint")
                return (
                    Path(args.experiments)
                    / row["exp_name"]
                    / "model"
                    / row["checkpoint"]
                )
    raise SystemExit(f"run_id not found in {args.alias_index}: {args.run_id}")


def compare(checkpoint: Path, legacy_src: Path, device_name: str) -> int:
    device = torch.device(device_name)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    shape = architectures.shape_from_state_dict(state)
    print(f"checkpoint: {checkpoint.name}")
    print(f"  architecture recovered from tensor shapes: {shape}")

    LegacyModel = load_legacy_class(legacy_src)
    legacy = LegacyModel(**shape).to(device).eval()
    rebuilt = architectures.BayesianResNetSTEC(**shape).to(device).eval()

    # Both must accept the checkpoint unchanged. A renamed parameter shows up here rather
    # than as a silently randomly-initialised layer.
    legacy_missing = legacy.load_state_dict(state, strict=True)
    rebuilt_missing = rebuilt.load_state_dict(state, strict=True)
    print(f"  both classes accepted the state dict {legacy_missing}, {rebuilt_missing}")

    torch.manual_seed(1)
    x = torch.randn(BATCH, shape["n_in"], device=device)

    with torch.no_grad(), determinism.deterministic_mode():
        pinned_legacy = determinism.freeze_bayesian_layers(legacy, seed=PIN_SEED)
        pinned_rebuilt = determinism.freeze_bayesian_layers(rebuilt, seed=PIN_SEED)
        if pinned_legacy != pinned_rebuilt or pinned_legacy == 0:
            print(
                f"  FAIL  pinned {pinned_legacy} legacy vs {pinned_rebuilt} rebuilt layer(s)"
            )
            return 1

        legacy_mean, legacy_var = legacy(x)
        rebuilt_mean, rebuilt_var = rebuilt(x)

        control = float((legacy(x)[0] - legacy(x)[0]).abs().max())
        mean_diff = float((legacy_mean - rebuilt_mean).abs().max())
        var_diff = float((legacy_var - rebuilt_var).abs().max())

    print(f"  Bayesian layers pinned: {pinned_legacy}")
    print(f"  zero-perturbation control : {control:.3e}")
    print(f"  max |mean difference|     : {mean_diff:.3e}")
    print(f"  max |variance difference| : {var_diff:.3e}")
    print(
        f"  mean range                : [{float(legacy_mean.min()):.2f}, "
        f"{float(legacy_mean.max()):.2f}] TECU"
    )

    if control != 0.0:
        print(f"\n  FAIL  control is {control:.3e}; the pinning did not take effect")
        return 1
    if mean_diff == 0.0 and var_diff == 0.0:
        print("\n  PASS  the rebuilt class is bit-exactly the same function")
        return 0
    print(
        f"\n  FAIL  the implementations differ (mean {mean_diff:.3e}, var {var_diff:.3e})"
    )
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="path to a .pth")
    source.add_argument("--run-id", help="run_id from the alias index")
    parser.add_argument("--alias-index", default="artifacts/runs/alias_index.csv")
    parser.add_argument(
        "--experiments", default="/scratch2/arrueegg/WP4/PNN_STEC/experiments"
    )
    parser.add_argument("--legacy-src", type=Path, default=DEFAULT_LEGACY_SRC)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    return compare(resolve_checkpoint(args), args.legacy_src, args.device)


if __name__ == "__main__":
    sys.exit(main())
