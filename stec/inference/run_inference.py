"""Inference entry point: run a checkpoint over test days and write the prediction store.

`stec.inference.monte_carlo` and `stec.inference.prediction_store` are gate-verified (Gate
D: bit-exact against a measured 1.275 TECU MC noise floor, one real day, 4096 observations,
100 draws) but nothing in `stec/` called them - the real store the analysis layer reads is
populated entirely by pre-rebuild `src/inference_testset.py` / `src/inference_map.py` /
`src/compare_stec_vtec_gim.py`. This module is that missing wiring: a checkpoint and a list
of days in, one prediction-store parquet file per day out.

Both datasets are wired up. "Madrigal" was a different kind of gap from the other "not
ported" notes in this codebase: it was not that a Madrigal *reader* was missing, it was that
nothing in `stec/` read Madrigal geometry as model *input* at all - `stec.baselines.madrigal`
only loads Madrigal's own reference STEC, to compare a prediction against, never to build
the tensor a prediction is made from. `stec.data.madrigal_reader.read_madrigal_day` is that
reader, ported from `src/data_loader/madrigal_dataset.py`'s `MadrigalSTECDataset` (see its
own module docstring for what it does and does not reproduce from that reference). Its
`local_time_hours` now defaults to `lon_ipp`, this project's own convention and the
physically correct one (divergence #12, `stec.analysis.divergences`) - the legacy
reference's station-longitude convention is a corrected erratum, not a preserved choice.
The published Table 4 numbers and the 235 days already in `predictions/finetuned_stec/madrigal/`
were produced under the wrong (station) convention and do not reflect this default; a
corrected re-run of those 235 days is `stec.inference.reinference_madrigal_local_time`, not
a plain re-invocation of this module (see that module's docstring for why: this driver
writes only the STEC model's own columns, and the 235 files on disk also carry VTEC/GIM
baseline columns this driver cannot recompute, so overwriting them here would silently drop
those columns rather than correct them). Passing `--madrigal-local-time-longitude station`
reproduces the legacy convention exactly, for anyone who needs to regenerate a day matching
the still-published numbers. Its output is shaped exactly
like `read_day`'s, so the branch below is the only dataset-specific code in this driver;
everything downstream of it - assembly, Monte Carlo sampling, the store write - does not
know which dataset it is looking at.

The zero-perturbation control runs once per process, before the first real sampling pass,
because `BayesianResNetSTEC`'s output layer resamples weights on every forward call - this
is the CLAUDE.md gotcha and `stec/models/determinism.py`'s own docstring both require it
before trusting anything this driver writes. `monte_carlo_uncertainty` itself already seeds
its sampling loop (`determinism.monte_carlo`), so the check here is that the *pinning*
machinery those samples rely on actually pins, not a repeat of the sampling seed.

The prediction-store schema in `prediction_store.py` is authoritative. This driver passes
the *entire* assembled frame - every raw column the reader returns (`read_day` for "own",
`read_madrigal_day` for "madrigal") plus the four prediction columns - into
`write_predictions`, which does its own column selection against `STORE_COLUMNS`. Narrowing
the frame here before that call would reintroduce exactly the whitelist-at-the-write-site
defect the store's docstring exists to prevent - which for Madrigal specifically means never
adding a `slipc`/`gfphase` placeholder here either: `read_madrigal_day` has no cycle-slip
counter to build one from, so there is nothing to narrow away for those two. `sat` is
different: `read_madrigal_day` now synthesises it from `sat_id`/`gnss_type` (see that
module's docstring), and `STORE_COLUMNS` already lists `sat` alongside `station` - so it
flows through this driver into the store unchanged, the same way every other raw column
does, with no dataset-specific branch needed here either.

Usage::

    python -m stec.inference.run_inference \\
        --config path/to/config.yaml --checkpoint path/to/model.pth \\
        --model-variant finetuned_stec --dataset own --doys 2024:132 2024:133
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import paths
from ..data.day_reader import read_day
from ..data.feature_layout import FeatureLayout, layout_from_feature_control
from ..data.madrigal_reader import DEFAULT_ELEVATION_THRESHOLD_DEG, read_madrigal_day
from ..data.spherical_harmonics import SphericalHarmonics
from ..data.transforms import FeatureAssembler
from ..models import determinism
from ..models.architectures import load_checkpoint
from . import prediction_store
from .monte_carlo import (
    UncertaintyDecomposition,
    _PairedOutputAdapter,
    monte_carlo_uncertainty,
)

logger = logging.getLogger(__name__)

Day = tuple[int, int]

MANIFEST_COLUMNS = ("model_variant", "dataset", "year", "doy", "rows", "samples")


def _distribution_of(config: dict) -> str:
    """Same rule `stec.analysis.paper_tables` uses to pick the SH convention."""
    loss_function = str(config.get("training", {}).get("loss_function", "")).lower()
    return "laplace" if "laplac" in loss_function else "gaussian"


def build_layout_and_assembler(config: dict) -> tuple[FeatureLayout, FeatureAssembler]:
    """The input layout a config describes, and the assembler that fills it - identical
    construction to `stec.training.run_training`, so a checkpoint and the inference run
    reading it are guaranteed to agree on what a column means."""
    layout = layout_from_feature_control(
        config.get("feature_control", {}),
        sh_degree=int(config.get("data", {}).get("SH_degree", 0)),
        target=str(config.get("target", "stec")),
        distribution=_distribution_of(config),
    )
    sh_encoder = None
    if layout.sh_width:
        legendre_polys = layout.sh_convention.legendre_polys(layout.sh_degree)
        sh_encoder = SphericalHarmonics(legendre_polys)
    return layout, FeatureAssembler(layout, sh_encoder=sh_encoder)


def _numeric_tensors(raw: dict) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(values).float()
        for name, values in raw.items()
        if values.dtype.kind in "fiu"
    }


def _decode_if_bytes(values: np.ndarray) -> np.ndarray:
    """`station`/`sat` arrive as fixed-width byte strings out of the HDF5 table; every
    other raw column is already numeric. Pandas would otherwise stringify the bytes
    objects themselves (`b'AMC4'`) instead of decoding them."""
    if values.dtype.kind == "S":
        return np.array([value.decode("ascii") for value in values])
    return values


def check_zero_perturbation(
    model: torch.nn.Module, inputs: torch.Tensor, seed: int
) -> None:
    """The Bayesian A/B invariant: identical input through pinned weights must return
    bit-identical output. `BayesianResNetSTEC` returns `(mean, variance)`, so this reuses
    `monte_carlo`'s own pairing adapter rather than hand-rolling a second one - the same
    combination `tests/inference/test_monte_carlo.py` pins.

    Runs once per process, on a single row, before any real sampling: a spurious 0.33 TECU
    of pure sampling noise was once measured and used to reject a correct approach for
    days, because nothing checked this first.
    """
    control = determinism.zero_perturbation_control(
        _PairedOutputAdapter(model), inputs[:1], seed=seed
    )
    if control != 0.0:
        raise RuntimeError(
            f"zero-perturbation control returned {control}, not exactly 0.0 - the "
            "pinned-weights invariant (stec/models/determinism.py) does not hold, so no "
            "prediction this run writes can be trusted as a single realised draw"
        )
    logger.info("Zero-perturbation control: 0.0 (pinned-weights invariant holds)")


def build_prediction_frame(
    raw: dict[str, np.ndarray], decomposition: UncertaintyDecomposition
) -> pd.DataFrame:
    """Every raw column plus the four prediction columns, ready for `write_predictions`.

    `raw['stec']` is renamed to `true_stec` - `prediction_store`'s own aliasing only knows
    `target_stec`, the name the pre-rebuild inference manager used; `read_day`'s column is
    plain `stec`.
    """
    frame = pd.DataFrame(
        {name: _decode_if_bytes(values) for name, values in raw.items()}
    )
    frame = frame.rename(columns={"stec": "true_stec"})
    frame["stec_pred"] = decomposition.mean.squeeze(-1).detach().cpu().numpy()
    frame["pred_total_unc"] = decomposition.total_std.squeeze(-1).detach().cpu().numpy()
    frame["pred_epistemic_unc"] = (
        decomposition.epistemic_std.squeeze(-1).detach().cpu().numpy()
    )
    frame["pred_aleatoric_unc"] = (
        decomposition.aleatoric_std.squeeze(-1).detach().cpu().numpy()
    )
    return frame


def run_inference(
    config: dict,
    model: torch.nn.Module,
    days: list[Day],
    *,
    model_variant: str,
    dataset: str,
    split: str = "test",
    samples: int = 100,
    seed: int = 42,
    database_root: Path | None = None,
    space_weather: Path | None = None,
    madrigal_root: Path | None = None,
    madrigal_elevation_threshold: float = DEFAULT_ELEVATION_THRESHOLD_DEG,
    madrigal_local_time_longitude: Literal["station", "ipp"] = "ipp",
    store_root: Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Run `model` over `split` for every day in `days`, writing one store file each.

    `split` means two different things depending on `dataset`, because the two datasets
    have no shared notion of a row-level split. For "own" it selects the STEC database's
    own precomputed `<split>_idx`. Madrigal has no such index, so for "madrigal" it selects
    the station set in `stec.config.paths.station_list(split)` instead - the closest
    analogue Madrigal has, and what `src/compare_stec_vtec_gim.py`'s Madrigal branch did by
    filtering to `test_station.list`. See `stec.data.madrigal_reader`'s module docstring.

    Returns the manifest rows (one per day) the caller writes to CSV - the file `min_rows`
    is keyed on, since a parquet output carries no row count in the pipeline's provenance
    record (only a `.csv`'s does).
    """
    if dataset not in ("own", "madrigal"):
        raise ValueError(f"unknown dataset {dataset!r}, expected 'own' or 'madrigal'")

    _layout, assembler = build_layout_and_assembler(config)
    store_root = store_root or paths.PREDICTIONS

    manifest: list[dict] = []
    checked_zero_perturbation = False
    for year, doy in days:
        if dataset == "own":
            raw = read_day(
                year,
                doy,
                split=split,
                database_root=database_root,
                space_weather=space_weather,
                with_identity=True,
            )
        else:
            raw = read_madrigal_day(
                year,
                doy,
                split=split,
                madrigal_root=madrigal_root,
                space_weather=space_weather,
                elevation_threshold=madrigal_elevation_threshold,
                with_identity=True,
                local_time_longitude=madrigal_local_time_longitude,
            )
        if len(raw.get("stec", [])) == 0:
            raise RuntimeError(f"{year}-{doy:03d} produced zero {split!r} rows")

        raw_tensors = _numeric_tensors(raw)
        inputs = assembler.assemble(raw_tensors).to(device)

        if not checked_zero_perturbation:
            check_zero_perturbation(model, inputs, seed)
            checked_zero_perturbation = True

        decomposition = monte_carlo_uncertainty(
            model, inputs, model.capabilities, requested_samples=samples, seed=seed
        )

        frame = build_prediction_frame(raw, decomposition)
        path = prediction_store.write_predictions(
            frame, model_variant, dataset, year, doy, root=store_root
        )
        manifest.append(
            {
                "model_variant": model_variant,
                "dataset": dataset,
                "year": year,
                "doy": doy,
                "rows": len(frame),
                "samples": decomposition.samples,
            }
        )
        logger.info(f"{year}-{doy:03d}: wrote {len(frame):,} rows to {path}")

    return manifest


def write_manifest(manifest: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest)
    return path


def _parse_day(token: str) -> Day:
    year, doy = token.split(":")
    return int(year), int(doy)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-variant", choices=["finetuned_stec", "pretrained_stec"], required=True
    )
    parser.add_argument("--dataset", choices=["own", "madrigal"], default="own")
    parser.add_argument(
        "--split",
        default="test",
        help="which of this day's train/val/test index sets to run inference over",
    )
    parser.add_argument(
        "--doys", nargs="+", type=_parse_day, required=True, metavar="YYYY:DDD"
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    # A run manifest, not an analysis result - it belongs beside the store output it
    # describes (as inference_smoke's stage explicitly does with an override), not in
    # multiday_results/, which is reserved for stec.analysis outputs and evaluation
    # sweeps (docs/revision/results_layout.md).
    parser.add_argument(
        "--output-dir", type=Path, default=paths.PREDICTIONS / "inference_run"
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=None,
        help="defaults to stec.config.paths.PREDICTIONS",
    )
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--space-weather", type=Path, default=None)
    parser.add_argument(
        "--madrigal-root",
        type=Path,
        default=None,
        help="only used for --dataset madrigal",
    )
    parser.add_argument(
        "--madrigal-elevation-threshold",
        type=float,
        default=DEFAULT_ELEVATION_THRESHOLD_DEG,
        help="minimum elevation (degrees) for a Madrigal row; only used for --dataset madrigal",
    )
    parser.add_argument(
        "--madrigal-local-time-longitude",
        choices=["station", "ipp"],
        default="ipp",
        help=(
            "longitude that feeds local_time_hours for --dataset madrigal; 'ipp' "
            "(default) matches the 'own' dataset's convention and is physically correct "
            "- see divergence #12 in stec.analysis.divergences. 'station' reproduces the "
            "legacy MadrigalSTECDataset convention the published Table 4 numbers and the "
            "current predictions/finetuned_stec/madrigal/ partition were built under; "
            "pass it only to regenerate a day matching those, not to write into that "
            "partition going forward - see "
            "stec.inference.reinference_madrigal_local_time for the corrected re-run"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)

    layout, _ = build_layout_and_assembler(config)
    model, shape = load_checkpoint(args.checkpoint, map_location=device)
    if shape["n_in"] != layout.total_dim:
        raise ValueError(
            f"checkpoint {args.checkpoint} expects {shape['n_in']} input columns, but "
            f"this config's feature_control assembles {layout.total_dim} - refusing "
            "rather than silently reading it through the wrong layout."
        )
    model = model.to(device)
    model.eval()

    manifest = run_inference(
        config,
        model,
        args.doys,
        model_variant=args.model_variant,
        dataset=args.dataset,
        split=args.split,
        samples=args.samples,
        seed=args.seed,
        database_root=args.database_root,
        space_weather=args.space_weather,
        madrigal_root=args.madrigal_root,
        madrigal_elevation_threshold=args.madrigal_elevation_threshold,
        madrigal_local_time_longitude=args.madrigal_local_time_longitude,
        store_root=args.store_root,
        device=device,
    )

    manifest_path = write_manifest(manifest, args.output_dir / "inference_manifest.csv")
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} day(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
