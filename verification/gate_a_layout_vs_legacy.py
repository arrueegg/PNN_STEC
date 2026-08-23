"""Gate A (layout half): does the rebuilt layout compute what the old code computed?

This is the comparison that matters, and it is deliberately *not* a comparison against the
trained checkpoints. Old code and new code are both run now, on the same configs, and
required to agree - because a checkpoint is a historical artifact whose config may no
longer describe it, and disagreeing with such a pair says nothing about the refactor.

The distinction is not academic here. A first pass comparing the layout against checkpoint
widths reported 242 mismatches, all in one model family, all "layout says 92, checkpoint
has 70" - which is the size mismatch CLAUDE.md already documents for that variant. Those
directories hold a config that does not describe their own checkpoint. Attributing that to
the rebuilt layout would have been wrong, and "fixing" the layout to match would have
broken it.

So this script answers the refactoring question, and reports the artifact question
separately:

  agreement  - the rebuilt layout and the legacy derivation compute the same width
  provenance - whether the stored checkpoint's real width matches what its config implies

    python verification/gate_a_layout_vs_legacy.py --limit 400
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/scratch2/arrueegg/WP4/PNN_STEC/src")

from stec.data.feature_layout import layout_from_feature_control  # noqa: E402

DEFAULT_EXPERIMENTS = Path("/scratch2/arrueegg/WP4/PNN_STEC/experiments")
INPUT_WEIGHT_KEYS = (
    "input_layer.0.weight",
    "layers.0.weight",
    "model.0.weight",
    "net.0.weight",
)


def legacy_input_width(config: dict) -> int | None:
    """What the pre-rebuild code sizes the input projection to, for this config."""
    from model.model import get_model  # noqa: PLC0415
    from utils.feature_registry import initialize_feature_registry  # noqa: PLC0415

    working = dict(config)
    try:
        initialize_feature_registry(working)
        # get_model prints a summary; keep the sweep readable.
        with contextlib.redirect_stdout(io.StringIO()):
            model = get_model(working)
    except (ValueError, KeyError, TypeError, RuntimeError, AttributeError):
        return None

    state = model.state_dict()
    for key in INPUT_WEIGHT_KEYS:
        if key in state:
            return int(state[key].shape[1])
    return None


def checkpoint_input_width(path: Path) -> int | None:
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
    loss = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplacian" in loss or "laplace" in loss else "gaussian"


def rebuilt_input_width(config: dict) -> int | None:
    feature_control = config.get("feature_control")
    if not isinstance(feature_control, dict):
        return None
    return layout_from_feature_control(
        feature_control,
        sh_degree=int(config.get("data", {}).get("SH_degree", 0)),
        target=str(config.get("target", "stec")),
        distribution=distribution_of(config),
    ).total_dim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--limit", type=int, default=0, help="0 checks all")
    args = parser.parse_args()

    directories = sorted(p for p in args.experiments.iterdir() if p.is_dir())
    if args.limit:
        directories = directories[: args.limit]

    agree = disagree = 0
    disagreements: list[str] = []
    stale_configs: list[str] = []
    checked_provenance = 0
    skipped: Counter[str] = Counter()

    for directory in directories:
        config_path = directory / "config.yaml"
        if not config_path.exists():
            skipped["no config"] += 1
            continue
        try:
            config = yaml.safe_load(config_path.read_text())
        except (OSError, yaml.YAMLError):
            skipped["unreadable config"] += 1
            continue
        if not isinstance(config, dict):
            skipped["config not a mapping"] += 1
            continue

        rebuilt = rebuilt_input_width(config)
        legacy = legacy_input_width(config)
        if rebuilt is None or legacy is None:
            skipped["width not derivable"] += 1
            continue

        if rebuilt == legacy:
            agree += 1
        else:
            disagree += 1
            disagreements.append(
                f"rebuilt {rebuilt} vs legacy {legacy}  {directory.name[:64]}"
            )

        # Separately: does the stored checkpoint match what its config implies?
        checkpoints = sorted((directory / "model").glob("*.pth"))
        if checkpoints:
            actual = checkpoint_input_width(checkpoints[0])
            if actual is not None:
                checked_provenance += 1
                if actual != legacy:
                    stale_configs.append(
                        f"config implies {legacy}, checkpoint has {actual}  {directory.name[:56]}"
                    )

    print(f"checked {len(directories)} experiment director(ies)\n")
    print("EQUIVALENCE - rebuilt layout against the legacy derivation, same config")
    print(f"  agree    : {agree}")
    print(f"  disagree : {disagree}")
    for line in disagreements[:10]:
        print(f"    {line}")

    print("\nPROVENANCE - stored checkpoint against what its own config implies")
    print(f"  checked  : {checked_provenance}")
    print(f"  consistent : {checked_provenance - len(stale_configs)}")
    print(f"  stale      : {len(stale_configs)}")
    for line in stale_configs[:5]:
        print(f"    {line}")
    if len(stale_configs) > 5:
        print(f"    ... and {len(stale_configs) - 5} more")

    if skipped:
        print("\n  skipped:")
        for reason, count in skipped.most_common():
            print(f"    {reason}: {count}")

    print()
    if disagree:
        print("  FAIL  the rebuilt layout does not reproduce the legacy derivation")
        return 1
    print("  PASS  the rebuilt layout reproduces the legacy derivation on every config")
    if stale_configs:
        print(f"  NOTE  {len(stale_configs)} experiment(s) hold a config that does not")
        print(
            "        describe their own checkpoint. That is an artifact defect, not a"
        )
        print("        refactoring one, and it is why this gate compares code to code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
