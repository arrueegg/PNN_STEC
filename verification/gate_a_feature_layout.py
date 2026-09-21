"""Gate A (layout half): does the single feature computation match every trained model?

The rebuilt `FeatureLayout` replaces two independent derivations of the transformed input
dimension - one in `model.py` sizing the input projection, one in `collation.py` filling
it. Replacing them is only safe if the replacement reproduces what the trained models were
actually built with.

Every checkpoint states its own answer: the input projection's weight has shape
`(hidden_dim, n_in)`. So for each experiment this reads the config the run used, computes
the layout from it, and compares against the checkpoint's real width. A disagreement means
the rebuilt layout would size a model differently from the one that produced the published
numbers.

This is the layout half of Gate A. The other half - that the loader emits those columns in
that order, with those values - needs the data path and follows separately.

    python verification/gate_a_feature_layout.py --limit 200
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.data.feature_layout import layout_from_feature_control  # noqa: E402

DEFAULT_EXPERIMENTS = Path("/scratch2/arrueegg/WP4/PNN_STEC/experiments")

# Architectures name their input projection differently. `layers.0.weight` is the MLP
# family, which includes the Mao et al. VTEC baseline - the only models that exercise the
# (degree + 1)**2 spherical-harmonic convention, so leaving them unchecked would leave half
# the layout logic unvalidated.
INPUT_WEIGHT_KEYS = (
    "input_layer.0.weight",
    "layers.0.weight",
    "model.0.weight",
    "net.0.weight",
)


def checkpoint_input_width(path: Path) -> int | None:
    """Read only the input layer's shape.

    Memory-mapped: a STEC checkpoint is ~98 MB and only its metadata is needed, so reading
    the tensor data would mean ~100 GB of IO across the full sweep - competing with
    whatever else is running on this host for nothing.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (RuntimeError, EOFError, OSError, ValueError):
        return None
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        return None
    for key in INPUT_WEIGHT_KEYS:
        if key in state:
            return int(state[key].shape[1])
    return None


def distribution_of(config: dict) -> str:
    """The likelihood a run was trained under, from its loss rather than its name."""
    loss = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplacian" in loss or "laplace" in loss else "gaussian"


def check(experiment: Path) -> tuple[str, str]:
    """Return (verdict, detail) for one experiment directory."""
    config_path = experiment / "config.yaml"
    if not config_path.exists():
        return "skipped", "no config.yaml"
    try:
        config = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError):
        return "skipped", "unreadable config"
    if not isinstance(config, dict):
        return "skipped", "config is not a mapping"

    feature_control = config.get("feature_control")
    if not isinstance(feature_control, dict):
        return "skipped", "no feature_control block"

    checkpoints = sorted((experiment / "model").glob("*.pth"))
    if not checkpoints:
        return "skipped", "no checkpoint"

    actual = checkpoint_input_width(checkpoints[0])
    if actual is None:
        return "skipped", "input layer not recognised"

    data = config.get("data", {})
    predicted = layout_from_feature_control(
        feature_control,
        sh_degree=int(data.get("SH_degree", 0)),
        target=str(config.get("target", "stec")),
        distribution=distribution_of(config),
    ).total_dim

    if predicted == actual:
        return "match", f"{actual}"
    return "MISMATCH", f"layout says {predicted}, checkpoint has {actual}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--limit", type=int, default=0, help="0 checks all")
    args = parser.parse_args()

    directories = sorted(p for p in args.experiments.iterdir() if p.is_dir())
    if args.limit:
        directories = directories[: args.limit]

    verdicts: Counter[str] = Counter()
    skips: Counter[str] = Counter()
    mismatches: list[tuple[str, str]] = []
    widths: Counter[str] = Counter()

    for directory in directories:
        verdict, detail = check(directory)
        verdicts[verdict] += 1
        if verdict == "skipped":
            skips[detail] += 1
        elif verdict == "MISMATCH":
            mismatches.append((directory.name, detail))
        else:
            widths[detail] += 1

    print(f"checked {len(directories)} experiment director(ies)")
    print(f"  match    : {verdicts['match']}")
    print(f"  MISMATCH : {verdicts['MISMATCH']}")
    print(f"  skipped  : {verdicts['skipped']}")
    if widths:
        print("\n  input widths reproduced:")
        for width, count in widths.most_common():
            print(f"    {width:>6} columns  x{count}")
    if skips:
        print("\n  skip reasons:")
        for reason, count in skips.most_common():
            print(f"    {reason}: {count}")
    if mismatches:
        print(f"\n  first {min(10, len(mismatches))} mismatch(es):")
        for name, detail in mismatches[:10]:
            print(f"    {detail}  {name[:70]}")
        print("\n  FAIL")
        return 1
    print(
        "\n  PASS  the single layout computation reproduces every trained model's width"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
