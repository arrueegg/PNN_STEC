"""Orchestrate the daily STEC sweep: fine-tune, infer, add baselines, one day at a time.

`src/multiday_evaluation.py` (~1,930 lines) used to be the only thing that ran this
sequence, driving pre-rebuild `src/` code end to end and then writing ~1,300 lines of its
own aggregation on top. Every per-day piece it drove now has a `stec/` counterpart -
`stec.training.run_training` (fine-tune), `stec.inference.run_inference` (STEC model
predictions into the store), `stec.inference.run_baselines` (VTEC + GIM baseline columns
into the same store file) and `stec.analysis.daily_metrics` (the aggregation, unchanged,
invoked here through `stec.pipeline` rather than reimplemented). This module is the last
piece: something that calls those four, in order, over a range of days, without
reimplementing what any of them already do.

Each of the three per-day steps runs as its own subprocess (`python -m stec.<module>`,
the same commands documented in each module's own usage), not an in-process function
call. This is not decoration: `stec.pipeline.runner.run_stage` uses the same one-command-
per-subprocess shape for exactly the same reason - a crash in one day's CUDA context (an
OOM, a corrupted device state) must not carry into the next day's fresh process, and a
day's own log is not interleaved with the days around it.

**Resumable, keyed on disk state, not a flag file.** Before running anything for a day,
this driver checks three things that must each already exist for that day to count as
done: the fine-tuned checkpoint file, the store parquet file, and - read from the
parquet's own schema, not assumed from the file merely existing - a baseline column
(`gim_stec`) inside it. A day that has a checkpoint and a store file but no baseline
columns yet (the sweep having been interrupted between `run_inference` and
`run_baselines`) reruns only the missing step. This mirrors the exact trap CLAUDE.md
records against the station-recovery sweep: a skip guard that covers only one of several
stages silently redoes - or silently skips - more than it should. Checking each stage's
own on-disk artifact, not a single day-level marker, is what avoids that here.

**Batched, with a free-space floor between batches**, the same shape
`scripts/backfill_store.sh` uses and for the same reason: a single long-running
invocation covering the whole requested range has no safe point to stop at, so a
disk-full crash lands wherever it lands, potentially mid-parquet-write. The set of
outstanding days is computed fresh from disk at the start of every call to `run_sweep`/
`main`, never assumed from a previous invocation's result - so calling this again after
an earlier run stopped (at the floor, or mid-range) only does whatever that run left
undone, including days it finished. A day already fully done when a call starts is still
reported in that call's summary (`run_day` confirms it from its own on-disk checks, at
the cost of a few stat calls and no subprocess), so a fully-resumed call's summary shows
every requested day as accounted for rather than going empty.

**Guards against a second concurrent sweep** by checking `/proc/*/cmdline` for another
process whose argv contains this module's dotted name as an exact field - not
`pgrep -f`/`ps -eo args`, both of which CLAUDE.md records as having previously matched
the checking shell itself, or truncated a long argv when stdout is not a terminal.

**What this has and has not been exercised against**, as of the session that wrote it:
run twice end to end against `tests/fixtures/pipeline_smoke` (`tests/runs/
test_daily_sweep.py`), including a two-day, two-batch run that proves the batching and
the disk-floor check actually iterate, and a second invocation of the same range that
proves every stage is skipped the second time. That fixture is one synthetic ~40-row day
duplicated under a second DOY, run on CPU in seconds. It has **not** been run against the
real STEC database, the real 640 GB `experiments/` tree, or at anything near the paper's
242-day scale - not the wall-clock behaviour of a real multi-day GPU sweep, not its
interaction with other jobs contending for the GPU or the disk floor for real, and not
failure modes specific to real data (a missing IONEX product, a VTEC experiment directory
with fewer than the full 10-seed ensemble, a raw HDF5 day absent from the database). A
Madrigal re-inference held the GPU for the whole session this driver was written in, and
long training/inference runs were out of scope for the same session - both are why this
gap exists, not an oversight this module's own logic hides.

Usage::

    python -m stec.runs.daily_sweep --year 2024 --start-doy 122 --end-doy 366

    # Explicit VTEC checkpoint/config and a non-default database root, e.g. for a smoke
    # or scratch run against fixture data rather than the real 640 GB tree:
    python -m stec.runs.daily_sweep --year 2024 --start-doy 132 --end-doy 133 \\
        --stec-config path/to/config.yaml --pretrain-checkpoint path/to/pretrain.pth \\
        --vtec-config path/to/vtec_config.yaml --vtec-checkpoint path/to/vtec_model.pth \\
        --database-root path/to/STEC_DB_CASDCB --space-weather path/to/omni.h5 \\
        --ionex-root path/to/GIM_IONEX --batch-days 1 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
import yaml

from ..config import paths
from ..data.madrigal_reader import DEFAULT_ELEVATION_THRESHOLD_DEG
from ..inference import prediction_store as ps
from ..inference.monte_carlo import DEFAULT_INFERENCE_BATCH_SIZE

logger = logging.getLogger(__name__)

Day = tuple[int, int]

# The column that means "run_baselines has already merged its output into this day's
# store file" - checked via the parquet's own schema (cheap: no row data is read), not
# assumed from the file merely existing, since inference and baselines write the same
# file in two separate steps.
BASELINE_MARKER_COLUMN = "gim_stec"

# Matched against /proc/<pid>/cmdline to detect a second concurrent sweep - see the
# module docstring.
DRIVER_MODULE = "stec.runs.daily_sweep"

SUMMARY_COLUMNS = ("year", "doy", "finetune", "inference", "baselines")


# --- disk floor and concurrency guard -----------------------------------------------


def disk_free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def another_sweep_running(proc_root: Path = Path("/proc")) -> bool:
    """True if some other process's argv names this module - see the module docstring
    for why this reads /proc directly rather than shelling out to `pgrep -f`/`ps`.

    `proc_root` defaults to the real `/proc` and is only ever overridden by a test, which
    fabricates a `<proc_root>/<pid>/cmdline` file rather than racing a real subprocess."""
    own_pid = os.getpid()
    for entry in os.scandir(proc_root):
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            cmdline = Path(entry.path, "cmdline").read_bytes()
        except OSError:
            continue  # the process exited between the scandir listing and this read
        if DRIVER_MODULE.encode() in cmdline.split(b"\0"):
            return True
    return False


# --- per-day config -------------------------------------------------------------------


def load_finetune_template(config_path: Path) -> dict:
    """The base STEC config every day's fine-tune starts from - `mode`/`year`/`doy` are
    overridden per day by `write_day_config`, everything else (model, feature_control,
    training hyperparameters, `pretrain_folder`) is used as given.

    `pretrain_folder` is resolved to an absolute path here if the config carries a
    relative one (as `config/paper/pretrain_stec_config.yaml` does): `stec.training.
    run_training._resolve_pretrain_checkpoint` joins it against the training
    subprocess's own working directory, and this driver should not depend on every
    subprocess it launches being started from the same place.
    """
    config = yaml.safe_load(config_path.read_text())
    pretrain_folder = config.get("pretrain_folder")
    if pretrain_folder and not Path(pretrain_folder).is_absolute():
        config["pretrain_folder"] = str(paths.LEGACY_ROOT / pretrain_folder)
    return config


def day_output_dir(models_root: Path, year: int, doy: int) -> Path:
    return models_root / f"{year}_{doy:03d}"


def write_day_config(template: dict, year: int, doy: int, output_dir: Path) -> Path:
    """One day's resolved config, written beside where its checkpoint will land.

    Written unconditionally, even when the day's checkpoint already exists: it is a
    cheap, deterministic YAML dump, and `stec.inference.run_inference` needs this same
    file to rebuild the input layout a checkpoint was trained under, whether or not this
    invocation is the one that trained it.
    """
    config = copy.deepcopy(template)
    config["mode"] = "finetune"
    config["year"] = year
    config["doy"] = doy
    config["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, default_flow_style=False))
    return config_path


def checkpoint_name_for(template: dict) -> str:
    """`stec.training.run_training.train`'s own checkpoint filename convention, for a
    fine-tune run - computed without running anything, so a day's expected checkpoint
    path is known before deciding whether to skip training for it."""
    seed = template["random_seed"]
    model_type = template["model"]["model_type"]
    return f"finetune_{model_type}_seed{seed:02}.pth"


# --- one subprocess step -----------------------------------------------------------


def run_module(module: str, args: list[str]) -> bool:
    """Run one `python -m <module> <args>` step to completion, letting it inherit this
    process's stdout/stderr so a long real fine-tune or inference run shows live
    progress rather than going silent until it exits (unlike `stec.pipeline.runner.
    run_stage`'s own subprocess call, which captures output because its commands are
    short enough that only the tail matters on failure)."""
    command = [sys.executable, "-m", module, *args]
    logger.info(f"  $ {' '.join(str(part) for part in command)}")
    result = subprocess.run(command, cwd=paths.REPO_ROOT)
    return result.returncode == 0


def store_has_column(path: Path, column: str) -> bool:
    if not path.exists():
        return False
    return column in pq.ParquetFile(path).schema.names


# --- one day -------------------------------------------------------------------------


def run_day(
    year: int,
    doy: int,
    *,
    finetune_config_path: Path,
    finetune_output_dir: Path,
    checkpoint_path: Path,
    pretrain_checkpoint: Path | None,
    model_variant: str,
    dataset: str,
    split: str,
    vtec_config: Path | None,
    vtec_checkpoint: Path | None,
    experiments_root: Path,
    store_root: Path,
    database_root: Path | None,
    space_weather: Path | None,
    madrigal_root: Path | None,
    ionex_root: Path | None,
    madrigal_elevation_threshold: float,
    samples: int,
    seed: int,
    batch_size: int,
    device: str,
) -> dict:
    """Fine-tune, infer, then add baselines for one day - each step skipped if its own
    on-disk artifact already exists. Returns one status row: `"skipped"`, `"ok"`,
    `"failed"` or `"not attempted"` (a later step, when an earlier one failed) per stage.
    """
    status: dict = {"year": year, "doy": doy}

    if checkpoint_path.exists():
        status["finetune"] = "skipped"
    else:
        args = [
            "--config",
            str(finetune_config_path),
            "--output-dir",
            str(finetune_output_dir),
            "--train-days",
            f"{year}:{doy}",
            "--val-days",
            f"{year}:{doy}",
            "--device",
            device,
        ]
        if pretrain_checkpoint is not None:
            args += ["--pretrain-checkpoint", str(pretrain_checkpoint)]
        if database_root is not None:
            args += ["--database-root", str(database_root)]
        if space_weather is not None:
            args += ["--space-weather", str(space_weather)]
        status["finetune"] = (
            "ok" if run_module("stec.training.run_training", args) else "failed"
        )

    if status["finetune"] == "failed":
        status["inference"] = status["baselines"] = "not attempted"
        return status

    store_file = ps.store_path(model_variant, dataset, year, doy, root=store_root)
    if store_file.exists():
        status["inference"] = "skipped"
    else:
        args = [
            "--config",
            str(finetune_config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--model-variant",
            model_variant,
            "--dataset",
            dataset,
            "--doys",
            f"{year}:{doy}",
            "--split",
            split,
            "--samples",
            str(samples),
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--store-root",
            str(store_root),
            "--output-dir",
            str(finetune_output_dir / "inference"),
        ]
        if database_root is not None:
            args += ["--database-root", str(database_root)]
        if space_weather is not None:
            args += ["--space-weather", str(space_weather)]
        if dataset == "madrigal":
            if madrigal_root is not None:
                args += ["--madrigal-root", str(madrigal_root)]
            args += [
                "--madrigal-elevation-threshold",
                str(madrigal_elevation_threshold),
            ]
        status["inference"] = (
            "ok" if run_module("stec.inference.run_inference", args) else "failed"
        )

    if status["inference"] == "failed":
        status["baselines"] = "not attempted"
        return status

    if store_has_column(store_file, BASELINE_MARKER_COLUMN):
        status["baselines"] = "skipped"
    else:
        args = [
            "--model-variant",
            model_variant,
            "--dataset",
            dataset,
            "--doys",
            f"{year}:{doy}",
            "--split",
            split,
            "--samples",
            str(samples),
            "--seed",
            str(seed),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "--store-root",
            str(store_root),
            "--experiments-root",
            str(experiments_root),
            "--output-dir",
            str(finetune_output_dir / "baselines"),
        ]
        if vtec_config is not None and vtec_checkpoint is not None:
            args += [
                "--vtec-config",
                str(vtec_config),
                "--vtec-checkpoint",
                str(vtec_checkpoint),
            ]
        if database_root is not None:
            args += ["--database-root", str(database_root)]
        if space_weather is not None:
            args += ["--space-weather", str(space_weather)]
        if ionex_root is not None:
            args += ["--ionex-root", str(ionex_root)]
        if dataset == "madrigal":
            if madrigal_root is not None:
                args += ["--madrigal-root", str(madrigal_root)]
            args += [
                "--madrigal-elevation-threshold",
                str(madrigal_elevation_threshold),
            ]
        status["baselines"] = (
            "ok" if run_module("stec.inference.run_baselines", args) else "failed"
        )

    return status


# --- the batched sweep ---------------------------------------------------------------


def refresh_daily_metrics() -> None:
    """Refresh Tables 3/4 from whatever the store now holds, the same way
    `scripts/backfill_store.sh` does after each of its own batches: so there is usable,
    current output between batches of a long sweep, not only at the very end. Logged but
    not fatal on failure, matching that script's own tolerance - a stale table is not a
    reason to stop the sweep that would fix it."""
    command = [
        sys.executable,
        "-m",
        "stec.pipeline",
        "run",
        "--force",
        "--keep-going",
        "--only",
        "daily_metrics",
    ]
    logger.info(f"  $ {' '.join(command)}")
    result = subprocess.run(command, cwd=paths.REPO_ROOT)
    if result.returncode != 0:
        logger.warning("  daily_metrics refresh failed, continuing")


def run_sweep(
    year: int,
    start_doy: int,
    end_doy: int,
    *,
    models_root: Path,
    finetune_template: dict,
    pretrain_checkpoint: Path | None,
    model_variant: str,
    dataset: str,
    split: str,
    vtec_config: Path | None,
    vtec_checkpoint: Path | None,
    experiments_root: Path,
    store_root: Path,
    database_root: Path | None,
    space_weather: Path | None,
    madrigal_root: Path | None,
    ionex_root: Path | None,
    madrigal_elevation_threshold: float,
    samples: int,
    seed: int,
    batch_size: int,
    device: str,
    batch_days: int,
    min_free_gb: float,
    aggregate: bool,
) -> list[dict]:
    """Fine-tune + infer + add baselines for every day in [start_doy, end_doy] of
    `year`, in batches of `batch_days`, stopping cleanly (not crashing) when free disk
    drops below `min_free_gb`. Safe to call again later: which days are already done is
    computed fresh from disk at the start of the call, so a second call only does
    whatever the first one left undone - see the module docstring."""
    checkpoint_name = checkpoint_name_for(finetune_template)
    all_days: list[Day] = [(year, doy) for doy in range(start_doy, end_doy + 1)]

    def checkpoint_path_for(day: Day) -> Path:
        return day_output_dir(models_root, *day) / "model" / checkpoint_name

    def store_file_for(day: Day) -> Path:
        return ps.store_path(model_variant, dataset, *day, root=store_root)

    def day_done(day: Day) -> bool:
        store_file = store_file_for(day)
        return (
            checkpoint_path_for(day).exists()
            and store_file.exists()
            and store_has_column(store_file, BASELINE_MARKER_COLUMN)
        )

    def process_day(day: Day) -> dict:
        day_year, day_doy = day
        output_dir = day_output_dir(models_root, day_year, day_doy)
        config_path = write_day_config(finetune_template, day_year, day_doy, output_dir)
        status = run_day(
            day_year,
            day_doy,
            finetune_config_path=config_path,
            finetune_output_dir=output_dir,
            checkpoint_path=checkpoint_path_for(day),
            pretrain_checkpoint=pretrain_checkpoint,
            model_variant=model_variant,
            dataset=dataset,
            split=split,
            vtec_config=vtec_config,
            vtec_checkpoint=vtec_checkpoint,
            experiments_root=experiments_root,
            store_root=store_root,
            database_root=database_root,
            space_weather=space_weather,
            madrigal_root=madrigal_root,
            ionex_root=ionex_root,
            madrigal_elevation_threshold=madrigal_elevation_threshold,
            samples=samples,
            seed=seed,
            batch_size=batch_size,
            device=device,
        )
        failed_stage = next(
            (
                s
                for s in ("finetune", "inference", "baselines")
                if status.get(s) == "failed"
            ),
            None,
        )
        if failed_stage:
            logger.error(
                f"{day_year}-{day_doy:03d}: {failed_stage} failed - continuing to the "
                "next day"
            )
        return status

    # Days already fully done are still reported, so a fully-resumed call's summary
    # shows every requested day accounted for rather than going empty - but at the cost
    # of only a few stat calls each (run_day's own per-stage checks confirm it and
    # return without launching a subprocess), so they are processed up front, outside
    # the free-space floor below that exists to protect real writes.
    results = [process_day(day) for day in all_days if day_done(day)]

    outstanding = [day for day in all_days if not day_done(day)]
    if not outstanding:
        logger.info("sweep covers the full requested range, nothing left to do")

    while outstanding:
        free_gb = disk_free_gb(paths.REPO_ROOT)
        if free_gb < min_free_gb:
            logger.warning(
                f"only {free_gb:.0f} GB free, below the {min_free_gb:.0f} GB floor - "
                f"stopping cleanly with {len(outstanding)} day(s) still outstanding"
            )
            break

        batch, outstanding = outstanding[:batch_days], outstanding[batch_days:]
        logger.info(
            f"{len(batch) + len(outstanding)} day(s) outstanding, {free_gb:.0f} GB "
            f"free - running a batch of {len(batch)}"
        )
        results.extend(process_day(day) for day in batch)

        if aggregate:
            refresh_daily_metrics()

    return results


def write_summary(results: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in results:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})
    return path


# --- CLI -------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-doy", type=int, required=True)
    parser.add_argument("--end-doy", type=int, required=True)
    parser.add_argument(
        "--stec-config",
        type=Path,
        default=paths.PAPER_PRETRAINED_CONFIG,
        help="resolved STEC config to fine-tune from; mode/year/doy/output_dir are "
        "overridden per day. Defaults to the paper's own frozen pretrain config, whose "
        "finetune: block already carries the canonical daily fine-tune hyperparameters "
        "(lr2e-4, bs512, 50 epochs, patience 15) - see CLAUDE.md's 'paper model' section.",
    )
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="override --stec-config's pretrain_folder resolution - needed when that "
        "folder is not reachable, e.g. a fixture run with no legacy experiments/ tree",
    )
    parser.add_argument(
        "--models-root", type=Path, default=paths.MODELS / "daily_sweep"
    )
    parser.add_argument(
        "--model-variant",
        choices=["finetuned_stec", "pretrained_stec"],
        default="finetuned_stec",
    )
    parser.add_argument("--dataset", choices=["own", "madrigal"], default="own")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--vtec-config",
        type=Path,
        default=None,
        help="explicit VTEC baseline config; with --vtec-checkpoint, bypasses the "
        "canonical --experiments-root resolution (see stec.inference.run_baselines)",
    )
    parser.add_argument("--vtec-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--experiments-root", type=Path, default=paths.LEGACY_EXPERIMENTS
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
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=25,
        help="matches scripts/backfill_store.sh's own batch size",
    )
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="skip refreshing daily_metrics (Tables 3/4) after each batch",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="defaults to <models-root>/sweep_manifest.csv",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if another_sweep_running():
        logger.error(
            f"another process already has {DRIVER_MODULE} in its argv - refusing to "
            "start a second concurrent sweep"
        )
        return 1

    finetune_template = load_finetune_template(args.stec_config)
    store_root = args.store_root or ps.DEFAULT_STORE_ROOT

    results = run_sweep(
        args.year,
        args.start_doy,
        args.end_doy,
        models_root=args.models_root,
        finetune_template=finetune_template,
        pretrain_checkpoint=args.pretrain_checkpoint,
        model_variant=args.model_variant,
        dataset=args.dataset,
        split=args.split,
        vtec_config=args.vtec_config,
        vtec_checkpoint=args.vtec_checkpoint,
        experiments_root=args.experiments_root,
        store_root=store_root,
        database_root=args.database_root,
        space_weather=args.space_weather,
        madrigal_root=args.madrigal_root,
        ionex_root=args.ionex_root,
        madrigal_elevation_threshold=args.madrigal_elevation_threshold,
        samples=args.samples,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        batch_days=args.batch_days,
        min_free_gb=args.min_free_gb,
        aggregate=not args.no_aggregate,
    )

    summary_path = args.summary_csv or (args.models_root / "sweep_manifest.csv")
    write_summary(results, summary_path)
    failed = [row for row in results if "failed" in row.values()]
    logger.info(
        f"sweep finished: {len(results)} day(s) attempted, {len(failed)} with a failed "
        f"stage. Summary: {summary_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
