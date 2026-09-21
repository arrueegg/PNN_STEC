"""Gate D: does the rebuilt inference path produce the predictions the old one did?

This gate cannot be written the obvious way, and the reason is worth stating before the
code. The stored parquet was produced by a **100-draw unseeded Monte Carlo average**: the
inference path seeds once per process, before data loading and model construction, and
never again. Those predictions are therefore one unrepeatable realisation of the
posterior. Diffing new output against the stored file would measure the difference between
two random draws, not the difference between two implementations, and no tolerance makes
that comparison mean anything.

So both sides are re-run now, from the same checkpoint, with the sampling explicitly
seeded:

    legacy class  + legacy MC loop  -> predictions A
    rebuilt class + rebuilt MC path -> predictions B

with `stec.models.determinism` pinning the weight draws by layer name so the two see the
same posterior sample. Against that, agreement should be bit-exact - the same standard the
model gate already meets - rather than a tolerance band.

The gate also reports the **MC noise floor**: how far apart two runs of the *same*
implementation land at different seeds. That number is what a comparison against the
stored file would have been measuring, and it is the honest scale for judging any
difference this gate does report.

    STEC_REPO_DATA=/scratch2/arrueegg/WP4/PNN_STEC/data \
        python verification/gate_d_inference_equivalence.py --doy 132 --rows 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/scratch2/arrueegg/WP4/PNN_STEC/src")

from stec.data.day_reader import read_day  # noqa: E402
from stec.data.feature_layout import layout_from_feature_control  # noqa: E402
from stec.data.transforms import FeatureAssembler  # noqa: E402
from stec.models import determinism  # noqa: E402
from stec.models.architectures import BayesianResNetSTEC, shape_from_state_dict  # noqa: E402

PAPER_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "local_time_hours": True,
    "lat_sta": True,
    "lon_sta": True,
    "sm_lat_sta": True,
    "sm_lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
    "sm_lat_ipp": True,
    "sm_lon_ipp": True,
    "Kp_index": True,
    "R_Sunspot_No": True,
    "Dst-index,_nT": True,
    "AE-index,_nT": True,
    "ap_index,_nT": True,
    "f107_index": True,
}

SAMPLES = 100
SEED = 12345


def build_inputs(year: int, doy: int, rows: int, device: torch.device) -> torch.Tensor:
    from utils.locationencoder.pe import SphericalHarmonics  # noqa: PLC0415

    columns = read_day(year, doy, split="test")
    layout = layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)
    assembler = FeatureAssembler(
        layout,
        sh_encoder=SphericalHarmonics(
            legendre_polys=layout.sh_convention.legendre_polys(layout.sh_degree)
        ),
    )
    raw = {
        name: torch.from_numpy(np.asarray(values[:rows], dtype=np.float32))
        for name, values in columns.items()
        if name in assembler.required_columns()
    }
    return assembler.assemble(raw).to(device)


def decompose(draws: torch.Tensor) -> dict[str, torch.Tensor]:
    """draws is (samples, rows, 2): per-pass mean and variance."""
    means, variances = draws[..., 0], draws[..., 1]
    epistemic = means.var(dim=0, unbiased=True)
    aleatoric = variances.mean(dim=0)
    return {
        "mean": means.mean(dim=0),
        "epistemic": epistemic.sqrt(),
        "aleatoric": aleatoric.sqrt(),
        "total": (epistemic + aleatoric).sqrt(),
    }


@torch.no_grad()
def sample(
    model: torch.nn.Module, inputs: torch.Tensor, seed: int
) -> dict[str, torch.Tensor]:
    determinism.unfreeze_bayesian_layers(model)
    torch.manual_seed(seed)
    draws = torch.stack([torch.stack(model(inputs), dim=-1) for _ in range(SAMPLES)])
    return decompose(draws)


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float((a[k] - b[k]).abs().max()) for k in a}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--doy", type=int, default=132)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    shape = shape_from_state_dict(state)

    from model.model import BayesianResNetSTEC as LegacyModel  # noqa: PLC0415

    legacy = LegacyModel(**shape).to(device).eval()
    rebuilt = BayesianResNetSTEC(**shape).to(device).eval()
    legacy.load_state_dict(state)
    rebuilt.load_state_dict(state)

    inputs = build_inputs(args.year, args.doy, args.rows, device)
    print(
        f"day {args.year}-{args.doy:03d}, {inputs.shape[0]} observations, "
        f"{SAMPLES} draws, seed {SEED}"
    )

    with determinism.deterministic_mode():
        a = sample(legacy, inputs, SEED)
        b = sample(rebuilt, inputs, SEED)
        # The floor: the same implementation, a different seed.
        c = sample(rebuilt, inputs, SEED + 1)

    equivalence = compare(a, b)
    floor = compare(b, c)

    width = max(len(k) for k in equivalence)
    print("\n  quantity      old vs new (seeded)     same code, different seed")
    for key in ("mean", "total", "epistemic", "aleatoric"):
        print(f"  {key:<{width}}  {equivalence[key]:>18.3e}  {floor[key]:>26.3e}")

    print(
        f"\n  predicted STEC range: [{float(a['mean'].min()):.2f}, "
        f"{float(a['mean'].max()):.2f}] TECU"
    )

    worst = max(equivalence.values())
    if worst == 0.0:
        print("\n  PASS  the rebuilt inference path is bit-exactly the same function")
        print(
            f"        (a seed change alone moves the mean by {floor['mean']:.3e} TECU,"
        )
        print(
            "         which is what a comparison against the stored parquet measures)"
        )
        return 0
    print(
        f"\n  FAIL  implementations differ by {worst:.3e}, against a noise floor of "
        f"{max(floor.values()):.3e}"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
