"""Wire the VTEC + Mapping and IGS GIM baselines into the prediction store.

`stec/baselines/gim.py` and `stec/baselines/vtec_mapping.py` already port the baseline
maths (`IONEXReader`, `GIMMapper`, the thin-shell mapping functions, the Laplace scale/std
distinction), and `prediction_store.py`'s schema already reserves `vtec_model_stec`,
`vtec_model_stec_total_unc`/`_aleatoric_unc`/`_epistemic_unc` and `gim_stec`. Nothing in
`stec/` called that math and wrote it back to the store: `stec.inference.run_inference`
writes only the STEC model's own columns. This module is that missing wiring, so Tables 3
and 4's "VTEC + Mapping" and "IGS GIM" rows can be rebuilt from raw without `src/`.

This driver adds baseline columns; it does not run the STEC model. It reads the store file
`stec.inference.run_inference` already wrote for a day (`true_stec`/`stec_pred`/`satele`
and the rest), re-reads the same day's raw observations independently, computes the two
baselines from that raw read, and merges the result back in - `write_predictions` replaces
a day's file whole, so leaving the STEC-model-owned columns untouched means copying them
across a full merged frame, not editing the file in place. `_verify_alignment` checks that
the fresh raw read landed on the same rows, in the same order, as the file already on disk
before trusting that merge: `stec.inference.reinference_madrigal_local_time` hit exactly
this failure mode once, from a different reader taking a different code path over the same
day, and the fix there is the same shape as the check here.

Correctness requirements, each pinned by a test and each corresponding to a bug that
already reached results in this repository:

1. **The VTEC baseline predicts a Laplace scale, not a standard deviation.**
   `monte_carlo_uncertainty` reads `MLP_LaplacianNLL.capabilities.spread_kind == "scale"`
   and converts to variance (`2 * scale**2`) before this module ever sees a number, so
   `decomposition.total_std` arriving here is already `sqrt(2) * b` - the standard
   deviation `vtec_model_stec_total_unc` expects. `compute_vtec_baseline` below only ever
   scales that std by the mapping factor; it never re-derives a variance from it, which is
   the step two independent ports of this codebase got wrong.
2. **Never narrow the schema at a write site.** The merge here starts from every column
   already in the store file (not a hand-picked subset) and adds the baseline columns on
   top; `write_predictions` does its own column selection against `STORE_COLUMNS`.
3. **`year`/`doy` are the caller's own integers, never read back from a results frame.**
   `add_baselines_for_day` takes them as explicit arguments and passes them straight to
   `GIMMapper.load_for_year_doy` (which rounds defensively even so) - there is no
   denormalise-and-round step in this module because there is nothing here to denormalise.
4. **The VTEC baseline is not the obvious variant, and it is an ensemble, not one
   checkpoint.** `vtec_checkpoint_paths_for_doy` builds the canonical
   `MLP_LaplacianNLL_h90_l3_..._SH15_..._woYear` experiment path (CLAUDE.md: "The VTEC
   baseline is not the obvious one"), not a glob that could resolve to the
   `MLP_h512_..._SH5_..._SWI` variant with a different feature set entirely. That
   canonical config also sets `finetune.ensemble_size: 10` / `train_ensemble: true`, and
   242 of 245 real `..._lr1e-3_..._woYear` experiment directories on this host still carry
   all 10 seed checkpoints - `load_vtec_model` loads every `.pth` file beside the one it
   is given and wraps more than one in `DeepEnsemble` (already ported, see
   `stec.models.architectures`), never just the first it finds. Loading only the one
   checkpoint a caller happens to name first reproduces one ensemble member's prediction,
   not the published mean - measured directly while building this module: doing that for
   2024-183 gave a plausible-looking but wrong `vtec_model_stec` (RMSE 2.38 TECU against
   the real column, some rows off by over 40 TECU) and a `vtec_model_stec_epistemic_unc`
   of exactly zero where the real column carries a genuine, non-trivial spread (mean 1.79
   TECU) - the tell that an ensemble collapsed to a single member, not noise.
5. **Seeded, checked determinism.** `compute_vtec_baseline` runs the zero-perturbation
   control before any real forward pass, the same invariant `stec.inference.run_inference`
   checks for the STEC model - neither `MLP_LaplacianNLL` nor `DeepEnsemble` has a
   Bayesian layer to pin, so the control is a check that the model is genuinely
   deterministic (no stray dropout, no uninitialised buffer) rather than a no-op kept only
   for symmetry.

Usage::

    python -m stec.inference.run_baselines \\
        --vtec-config path/to/vtec_config.yaml --vtec-checkpoint path/to/model.pth \\
        --model-variant finetuned_stec --dataset own --doys 2024:132 2024:133

or, to resolve the canonical per-DOY VTEC checkpoint automatically::

    python -m stec.inference.run_baselines \\
        --experiments-root experiments --model-variant finetuned_stec --dataset own \\
        --doys 2024:132
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..baselines.gim import GIMMapper
from ..baselines.vtec_mapping import map_vtec_to_stec
from ..config import paths
from ..data.day_reader import read_day
from ..data.madrigal_reader import DEFAULT_ELEVATION_THRESHOLD_DEG, read_madrigal_day
from ..models.architectures import DeepEnsemble, load_vtec_checkpoint
from . import prediction_store as ps
from .monte_carlo import DEFAULT_INFERENCE_BATCH_SIZE, monte_carlo_uncertainty
from .run_inference import (
    _numeric_tensors,
    build_layout_and_assembler,
    check_zero_perturbation,
)

logger = logging.getLogger(__name__)

Day = tuple[int, int]

# src/compare_stec_vtec_gim.py's production default (--mapping_function, "default: MSLM")
# is what produced the paper's "VTEC + Mapping" and "IGS GIM" numbers - MappingFunction's
# own default ("SLM") is not. See stec/baselines/vtec_mapping.py's module docstring.
MAPPING_TYPE = "MSLM"

# The paper's canonical VTEC fine-tune variant (CLAUDE.md, stec.analysis.
# positioning_coverage.CANONICAL_VTEC_SUFFIX). Quoted rather than imported: that module
# pulls in the positioning stack, which this driver has no other reason to depend on.
CANONICAL_VTEC_SUFFIX = (
    "MLP_LaplacianNLL_h90_l3_lr1e-3_bs2048_LaplacianNLL_Adam_ReduceLROnPlateau_sub500K_"
    "SH15_ps0.1_lw1e+0_woYear"
)
VTEC_CHECKPOINT_FILENAME = "finetune_MLP_LaplacianNLL_seed42.pth"

BASELINE_COLUMNS = (
    "vtec_model_stec",
    "vtec_model_stec_total_unc",
    "vtec_model_stec_aleatoric_unc",
    "vtec_model_stec_epistemic_unc",
    "gim_stec",
)

# Checked for row-for-row alignment between a fresh raw read and the store file already on
# disk before any baseline column is merged in - both reads select the same cached
# `test_idx`/station filter, so they should already agree exactly; this measures that
# rather than assuming it (see the module docstring).
# Same value as stec.inference.reinference_madrigal_local_time's ALIGNMENT_TOLERANCE and
# for the same reason: two independent reads of the same float32 columns disagree by up to
# ~4e-3 on `sod` (values run to 86399, and float32 carries ~7 significant digits) purely
# from the round-trip through parquet, not from genuine misalignment - measured directly
# against real 2024-183 data while building this module.
ALIGNMENT_COLUMNS = ("sod", "satele", "lat_ipp", "lon_ipp")
ALIGNMENT_TOLERANCE = 1e-2

# satele earns its own, looser tolerance for the same reason
# stec.inference.reinference_madrigal_local_time's ELEVATION_TOLERANCE_DEG does: right at
# the zenith singularity (elevation close to 90 deg, where azimuth is undefined and the
# smallest geometry perturbation swings it) two otherwise-identical reads of the same row
# disagree by more than a plain float32 round-trip would predict. Measured directly
# against real 2024-183 data while building this module: 5 of 2,426,735 rows exceed 1e-3
# deg, the single worst (station LICC, elevation 89.98 deg) by 0.0626 deg - still six
# orders of magnitude below any real misalignment (a different satellite's IPP lands
# degrees away, not hundredths of a degree), and every one of the worst rows matches on
# station identity, confirming it is the same observation, not a different one.
ELEVATION_TOLERANCE_DEG = 0.1

MANIFEST_COLUMNS = (
    "model_variant",
    "dataset",
    "year",
    "doy",
    "rows",
    "gim_valid_fraction",
)


def vtec_checkpoint_paths_for_doy(
    doy: int, experiments_root: Path
) -> tuple[Path, Path]:
    """The paper's canonical VTEC fine-tune checkpoint and config for one 2024 DOY."""
    experiment_dir = (
        Path(experiments_root) / f"Finetune_VTEC_2024_{doy}_{CANONICAL_VTEC_SUFFIX}"
    )
    return (
        experiment_dir / "model" / VTEC_CHECKPOINT_FILENAME,
        experiment_dir / "config.yaml",
    )


def compute_gim_baseline(
    raw: dict[str, np.ndarray],
    year: int,
    doy: int,
    *,
    ionex_root: Path | None = None,
    mapping_type: str = MAPPING_TYPE,
) -> dict[str, np.ndarray]:
    """IGS GIM VTEC, mapped to slant STEC. Deterministic - no model, no sampling."""
    mapper = GIMMapper(mapping_type=mapping_type, gim_type="IGS")
    mapper.load_for_year_doy(year, doy, ionex_root=ionex_root)
    gim_stec = mapper.map_vtec_to_stec(
        raw["sod"], raw["lat_ipp"], raw["lon_ipp"], raw["satele"]
    )
    return {"gim_stec": gim_stec}


def load_vtec_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load the VTEC baseline from `checkpoint_path`'s directory.

    Every `.pth` file beside `checkpoint_path` is loaded, not just the one named: the
    canonical VTEC config sets `finetune.ensemble_size: 10` / `train_ensemble: true`, and
    242 of 245 real `..._lr1e-3_..._woYear` experiment directories on this host still hold
    all 10 seed checkpoints (see the module docstring's requirement 4). More than one file
    is wrapped in `DeepEnsemble`, exactly `stec.models.legacy_factory.
    load_model_for_inference`'s ensemble-detection - reimplemented here rather than
    called, because that function needs a legacy `FeatureRegistry`-sized config
    (`load_vtec_checkpoint`'s own docstring explains why this driver avoids it). A
    directory holding only one checkpoint yields a single model instead - not every VTEC
    experiment on disk retains the full ensemble.
    """
    checkpoint_paths = sorted(Path(checkpoint_path).parent.glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"no checkpoint files found beside {checkpoint_path}")

    models = []
    for path in checkpoint_paths:
        model, _shape = load_vtec_checkpoint(path, map_location=device)
        models.append(model.to(device))

    if len(models) == 1:
        return models[0]
    logger.info(
        f"Loaded {len(models)}-member VTEC ensemble from {checkpoint_paths[0].parent}"
    )
    return DeepEnsemble(models, model_type="Laplacian").to(device)


def _ensemble_uncertainty(
    ensemble: DeepEnsemble, inputs: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """`DeepEnsemble.get_uncertainties`, chunked over rows for the same reason
    `determinism.monte_carlo` chunks a Bayesian forward pass: one unbatched call over a
    ~2M-row day allocates one activation tensor per ensemble member sized for the whole
    day at once. Every member is a fixed, already-trained network with no per-call
    randomness, so chunking the row dimension changes nothing about the arithmetic any
    individual row goes through - same argument as `determinism.monte_carlo`'s docstring,
    one layer up.
    """
    rows = inputs.shape[0]
    chunk_size = batch_size or rows
    means, aleatoric_vars, epistemic_vars, total_vars = [], [], [], []
    with torch.no_grad():
        for start in range(0, rows, chunk_size):
            mean, aleatoric_var, epistemic_var, total_var = ensemble.get_uncertainties(
                inputs[start : start + chunk_size]
            )
            means.append(mean)
            aleatoric_vars.append(aleatoric_var)
            epistemic_vars.append(epistemic_var)
            total_vars.append(total_var)
    return (
        torch.cat(means),
        torch.cat(aleatoric_vars),
        torch.cat(epistemic_vars),
        torch.cat(total_vars),
    )


def compute_vtec_baseline(
    raw: dict[str, np.ndarray],
    vtec_config: dict,
    vtec_model: torch.nn.Module,
    *,
    device: torch.device,
    seed: int,
    samples: int,
    batch_size: int,
    mapping_type: str = MAPPING_TYPE,
) -> dict[str, np.ndarray]:
    """VTEC model forward pass, mapped to slant STEC with its Laplace spread carried
    through. See the module docstring's requirement 1 for why the std this function scales
    is already the right quantity, not a scale that still needs converting, and
    requirement 4 for why `vtec_model` may be a single `MLP_LaplacianNLL` or a
    `DeepEnsemble` of them - the two need different uncertainty machinery (Bayesian-weight
    Monte Carlo vs. deterministic-member spread), so this branches on which it was given
    rather than forcing one abstraction over both.
    """
    _layout, assembler = build_layout_and_assembler(vtec_config)
    inputs = assembler.assemble(_numeric_tensors(raw)).to(device)

    # Neither architecture has a Bayesian layer to pin, so this checks that the model is
    # genuinely deterministic (no stray dropout, no uninitialised buffer) - see
    # requirement 5. DeepEnsemble.forward returns the same (mean, spread) shape
    # _PairedOutputAdapter expects, so the same check covers both branches below.
    check_zero_perturbation(vtec_model, inputs, seed)

    if isinstance(vtec_model, DeepEnsemble):
        mean_t, aleatoric_var_t, epistemic_var_t, total_var_t = _ensemble_uncertainty(
            vtec_model, inputs, batch_size
        )
    else:
        decomposition = monte_carlo_uncertainty(
            vtec_model,
            inputs,
            vtec_model.capabilities,
            requested_samples=samples,
            seed=seed,
            batch_size=batch_size,
        )
        mean_t = decomposition.mean
        aleatoric_var_t = decomposition.aleatoric_std**2
        epistemic_var_t = decomposition.epistemic_std**2
        total_var_t = decomposition.total_std**2

    vtec_mean = mean_t.squeeze(-1).detach().cpu().numpy()
    elevation = raw["satele"]
    mapped_mean = map_vtec_to_stec(vtec_mean, elevation, mapping_type=mapping_type)
    columns: dict[str, np.ndarray] = {"vtec_model_stec": mapped_mean.stec}

    # Called once per spread column, as stec.baselines.vtec_mapping's own docstring
    # prescribes: the propagation is identical for each (linear in the mapping factor),
    # but each is a distinct stored quantity.
    for suffix, variance in (
        ("total", total_var_t),
        ("aleatoric", aleatoric_var_t),
        ("epistemic", epistemic_var_t),
    ):
        std_np = variance.squeeze(-1).detach().cpu().numpy() ** 0.5
        mapped_spread = map_vtec_to_stec(
            vtec_mean, elevation, mapping_type=mapping_type, vtec_std=std_np
        )
        columns[f"vtec_model_stec_{suffix}_unc"] = mapped_spread.stec_std.std

    return columns


def _verify_alignment(
    raw: dict[str, np.ndarray], existing: pd.DataFrame, year: int, doy: int
) -> None:
    """Refuse to merge unless the fresh raw read landed on the same rows, in the same
    order, as the store file already on disk. Raises rather than returning a bool: a
    caller that ignored a False here is exactly how a Frankenstein row (one observation's
    STEC prediction merged with a different observation's baseline) would happen silently.
    """
    row_count = len(raw.get("sod", []))
    if row_count != len(existing):
        raise RuntimeError(
            f"{year}-{doy:03d}: row count changed ({len(existing)} on disk vs "
            f"{row_count} freshly read) - refusing to merge positionally"
        )
    for column in ALIGNMENT_COLUMNS:
        if column not in existing.columns:
            continue
        new_values = raw[column].astype(np.float64)
        old_values = existing[column].to_numpy(dtype=np.float64)
        if not len(new_values):
            continue
        tolerance = (
            ELEVATION_TOLERANCE_DEG if column == "satele" else ALIGNMENT_TOLERANCE
        )
        max_diff = float(np.max(np.abs(new_values - old_values)))
        if max_diff > tolerance:
            raise RuntimeError(
                f"{year}-{doy:03d}: {column} misaligned after re-read (max |delta| "
                f"{max_diff:.4f}) - the fresh raw read landed on different rows than the "
                "file on disk; refusing to merge baseline columns onto it"
            )
    if "station" in raw and "station" in existing.columns:
        # The store normalises station to uppercase (own emits uppercase already,
        # Madrigal lowercase) - upper-case both sides so this compares identity, not case.
        new_station = np.char.upper(raw["station"].astype(str))
        old_station = existing["station"].to_numpy().astype(str)
        if not np.array_equal(new_station, old_station):
            raise RuntimeError(
                f"{year}-{doy:03d}: station identity misaligned - refusing to merge"
            )


def add_baselines_for_day(
    year: int,
    doy: int,
    *,
    dataset: str,
    model_variant: str,
    vtec_config: dict,
    vtec_model: torch.nn.Module,
    store_root: Path,
    database_root: Path | None = None,
    madrigal_root: Path | None = None,
    space_weather: Path | None = None,
    ionex_root: Path | None = None,
    split: str | None = "test",
    madrigal_elevation_threshold: float = DEFAULT_ELEVATION_THRESHOLD_DEG,
    mapping_type: str = MAPPING_TYPE,
    seed: int = 42,
    samples: int = 100,
    batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Add the VTEC + Mapping and IGS GIM baseline columns to one day already in the
    prediction store, writing the merged frame back.

    Requires `stec.inference.run_inference` (or an equivalent writer) to have already
    produced this day's file: `write_predictions` refuses to write a frame missing
    `true_stec`/`stec_pred`/`satele`, and this driver computes only the two baselines, not
    the STEC model's own columns - see the module docstring for why that split is
    deliberate rather than a gap.
    """
    if dataset not in ("own", "madrigal"):
        raise ValueError(f"unknown dataset {dataset!r}, expected 'own' or 'madrigal'")

    store_file = ps.store_path(model_variant, dataset, year, doy, root=store_root)
    if not store_file.exists():
        raise FileNotFoundError(
            f"{year}-{doy:03d}: no existing store file at {store_file}. This driver adds "
            "VTEC + GIM baseline columns to a day the STEC model has already produced - "
            "run stec.inference.run_inference for this day first."
        )
    existing = pd.read_parquet(store_file)

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
        )
    if len(raw.get("stec", [])) == 0:
        raise RuntimeError(f"{year}-{doy:03d}: raw read produced zero {split!r} rows")

    _verify_alignment(raw, existing, year, doy)

    gim_columns = compute_gim_baseline(
        raw, year, doy, ionex_root=ionex_root, mapping_type=mapping_type
    )
    vtec_columns = compute_vtec_baseline(
        raw,
        vtec_config,
        vtec_model,
        device=device,
        seed=seed,
        samples=samples,
        batch_size=batch_size,
        mapping_type=mapping_type,
    )

    # Start from every column already on disk - never a hand-picked subset (module
    # docstring, requirement 2) - and add the two baselines on top.
    merged = existing.copy()
    for name, values in {**vtec_columns, **gim_columns}.items():
        merged[name] = values

    path = ps.write_predictions(
        merged, model_variant, dataset, year, doy, root=store_root
    )
    gim_valid_fraction = float(np.isfinite(gim_columns["gim_stec"]).mean())
    logger.info(
        f"{year}-{doy:03d}: merged VTEC + GIM baselines into {path} "
        f"({len(merged):,} rows, GIM valid for {gim_valid_fraction:.1%})"
    )
    return {
        "model_variant": model_variant,
        "dataset": dataset,
        "year": year,
        "doy": doy,
        "rows": len(merged),
        "gim_valid_fraction": round(gim_valid_fraction, 6),
    }


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
    parser.add_argument(
        "--model-variant", choices=["finetuned_stec", "pretrained_stec"], required=True
    )
    parser.add_argument("--dataset", choices=["own", "madrigal"], default="own")
    parser.add_argument(
        "--doys", nargs="+", type=_parse_day, required=True, metavar="YYYY:DDD"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="which of this day's train/val/test index sets to run over - must match "
        "the split the store file on disk was built with, or _verify_alignment rejects "
        "every day",
    )

    # Either point at one fixed VTEC checkpoint/config, or let the per-DOY canonical
    # experiment layout resolve them (mirrors reinference_madrigal_local_time.py's
    # checkpoint_paths_for_doy usage for the STEC model).
    parser.add_argument("--vtec-config", type=Path, default=None)
    parser.add_argument("--vtec-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=paths.LEGACY_EXPERIMENTS,
        help="resolves the canonical per-DOY VTEC checkpoint/config when "
        "--vtec-checkpoint/--vtec-config are not given directly",
    )

    parser.add_argument("--store-root", type=Path, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--space-weather", type=Path, default=None)
    parser.add_argument("--madrigal-root", type=Path, default=None)
    parser.add_argument(
        "--madrigal-elevation-threshold",
        type=float,
        default=DEFAULT_ELEVATION_THRESHOLD_DEG,
    )
    parser.add_argument("--ionex-root", type=Path, default=None)
    parser.add_argument(
        "--mapping-function",
        choices=["SLM", "MSLM"],
        default=MAPPING_TYPE,
        help="thin-shell mapping type; MSLM (default) is what produced the paper's "
        "numbers - see the module docstring",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument(
        "--output-dir", type=Path, default=paths.PREDICTIONS / "baselines_run"
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

    device = torch.device(args.device)
    store_root = args.store_root or ps.DEFAULT_STORE_ROOT

    manifest: list[dict] = []
    loaded_vtec: dict[int, tuple[dict, torch.nn.Module]] = {}
    for year, doy in args.doys:
        if args.vtec_config is not None and args.vtec_checkpoint is not None:
            vtec_config_path, vtec_checkpoint_path = (
                args.vtec_config,
                args.vtec_checkpoint,
            )
        else:
            vtec_checkpoint_path, vtec_config_path = vtec_checkpoint_paths_for_doy(
                doy, args.experiments_root
            )

        if doy not in loaded_vtec:
            vtec_config = yaml.safe_load(vtec_config_path.read_text())
            vtec_model = load_vtec_model(vtec_checkpoint_path, device)
            vtec_model.eval()
            loaded_vtec[doy] = (vtec_config, vtec_model)
        vtec_config, vtec_model = loaded_vtec[doy]

        row = add_baselines_for_day(
            year,
            doy,
            dataset=args.dataset,
            model_variant=args.model_variant,
            vtec_config=vtec_config,
            vtec_model=vtec_model,
            store_root=store_root,
            database_root=args.database_root,
            madrigal_root=args.madrigal_root,
            space_weather=args.space_weather,
            ionex_root=args.ionex_root,
            split=args.split,
            madrigal_elevation_threshold=args.madrigal_elevation_threshold,
            mapping_type=args.mapping_function,
            seed=args.seed,
            samples=args.samples,
            batch_size=args.batch_size,
            device=device,
        )
        manifest.append(row)

    manifest_path = write_manifest(manifest, args.output_dir / "baselines_manifest.csv")
    logger.info(f"Manifest: {manifest_path} ({len(manifest)} day(s))")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
