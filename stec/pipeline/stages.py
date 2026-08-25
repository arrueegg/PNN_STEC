"""Every analysis that produces a number in the paper, declared once.

Each stage names the command that runs it, what it reads, what it writes, the reviewer
comment it answers, and the minimum it must produce to be believed.

Order is significant and is the order below - `repair_gim_baseline` must precede
`daily_metrics`, which must precede `activity_stratification`.

Inputs are declared at the granularity that actually changes. The prediction store is
named as a directory rather than 242 parquet files, and the raw HDF5 days are not declared
at all: they are immutable external data, and treating them as an input would mean walking
740 GB to decide whether to run a two-second CSV summary.

Three fields carry what used to live only in prose, and they are the reason a reader of an
output CSV can now tell what it means:

* `canonical_for` - which deliverable this stage is the single source for.
* `caveats` - the conditions under which the output must not be read. Two of the
  evaluations here are not what they look like, and both have misled readers before.
* `supersedes` - the older artifact this replaces, which is marked rather than deleted.

No command in this file points at `src/` any more. Most run `-m stec.analysis.<name>`;
the two that still shell out to a standalone script (`repair_gim_baseline`,
`hyperparameter_search`) point at `stec/frozen/analysis/`, not `src/analysis/` - those
two scripts are deliberately unported (see `stec/frozen/README.md`) but were relocated
byte-identically so `src/` itself can be deleted in full. The registry is the contract
layer and can drive scripts as well as `-m` modules, which is what keeps porting from
being a big-bang rewrite: a stage's command changes when its analysis moves, and nothing
else does.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from ..analysis.daily_metrics import DATASET_LABELS, MODELS
from ..analysis.positioning_summary import METHOD_ORDER
from ..config import paths
from .stage import Stage


def _rel(path: Path) -> Path:
    """A `paths.py` result path, relative to `REPO_ROOT`.

    `paths.py` itself always returns absolute paths - correct, since `REPO_ROOT` is
    resolved once, in one place. But every other declared path in this file (`STORE_OWN`,
    `SWI`, `"experiments"`, `"wandb"`) is repository-relative, and `.pipeline/*.json` is
    documented as "the provenance record meant to be published alongside the code" - an
    absolute path baked into a stage's recorded `command`/`outputs` would make that record
    specific to whichever machine happened to run it. This is the one place that turns a
    `paths.py` result back into what a stage command actually needs.
    """
    return path.relative_to(paths.REPO_ROOT)


def _analysis_dir(name: str, *, rebuilt: bool) -> Path:
    return _rel(paths.analysis_result_dir(name, rebuilt=rebuilt))


STORE_OWN = "predictions/finetuned_stec/own"
STORE_PRETRAINED = "predictions/pretrained_stec/own"
STORE_MADRIGAL = "predictions/finetuned_stec/madrigal"
# The fully-Bayesian reference model's own store partition (see CLAUDE.md's
# store-partition gotcha for why this is a separate tree, not a subdirectory of
# STORE_PRETRAINED) - read only by epistemic_scale_diagnostic, as the sweep's reference
# architecture.
STORE_PRETRAINED_BNN_NLL = "predictions/pretrained_stec_resnet_bnn_nll/own"
# `positioning_coverage`'s own rebuilt output, not `positioning_runs/full_coverage/` -
# that tree is what the *pre-rebuild* `src/analysis/positioning_coverage.py` wrote
# directly, and nothing has regenerated it since the results-layout restructure moved
# this stage's default output to `analyses/<name>/rebuilt/` (2026-08-21). It kept
# existing on disk, so every downstream stage's `canonical_positioning_summary()` kept
# silently preferring it and reporting "up to date" against a file no producer owned -
# undetected until the 2026-08-24 station-recovery sweep changed the experiment tree and
# this file didn't move. Repointed here at the stage that actually produces it, so a
# future change to `experiments/` is fingerprinted through to every consumer again. See
# `stec/analysis/positioning_summary.py`'s module docstring for the full account.
POSITIONING = str(
    _rel(paths.analysis_result_dir("positioning_coverage", rebuilt=True))
    / "multiday_summary.csv"
)
WEIGHTING_RUN = str(
    _rel(paths.positioning_result_dir("20260216_2052")) / "multiday_summary.csv"
)
# oracle_benchmark reads this tree directly (stec/analysis/oracle_benchmark.py's
# `load_oracle`: the day directories under positioning/results for the .pos solutions,
# and positioning/evaluation/<day>/products for the SINEX) - never POSITIONING, unlike
# every other stage below that reads the positioning_coverage output. Declared at
# experiment-directory granularity, the same convention positioning_coverage uses for
# the whole `experiments` tree it scans, narrowed to the one experiment oracle_benchmark
# actually reads rather than the entire (640 GB) tree.
#
# A bare literal, like "experiments" above - not `_rel(paths.LEGACY_EXPERIMENTS / ...)`.
# LEGACY_EXPERIMENTS is `LEGACY_ROOT / "experiments"`, and LEGACY_ROOT is deliberately
# overridable *outside* REPO_ROOT via STEC_LEGACY_ROOT - a worktree, or the clean-clone
# test, points it elsewhere on purpose (see LEGACY_ROOT's own docstring). `_rel()`'s
# `relative_to(REPO_ROOT)` throws in exactly that case, which is a real regression this
# literal avoids: fingerprinting still runs relative to CWD like every other declared
# input, but nothing computes or resolves LEGACY_EXPERIMENTS at import time.
ORACLE_EXPERIMENT_DIR = "experiments/Reference_STEC_Oracle"
SWI = "data/omni_hourly_2010-2025.h5"

# Every stec.analysis output directory, named once so a stage's command string,
# `outputs`, `inputs` and `supersedes` can never disagree about where it writes.
# `paths.analysis_result_dir` is the one place the `analyses/<name>/{rebuilt,
# pre_rebuild}` layout is spelled out (docs/revision/results_layout.md) - nothing below
# builds a `multiday_results/...` string by hand.
PAPER_TABLES_DIR = _analysis_dir("paper_tables", rebuilt=True)
RELATIVE_ERROR_METRICS_DIR = _analysis_dir("relative_error_metrics", rebuilt=True)
TEMPORAL_REGIME_SPLIT_DIR = _analysis_dir("temporal_regime_split", rebuilt=True)
TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR = _analysis_dir(
    "temporal_regime_activity_matched", rebuilt=True
)
HYPERPARAMETER_SEARCH_DIR = _analysis_dir("hyperparameter_search", rebuilt=False)
STATION_INDEPENDENCE_DIR = _analysis_dir("station_independence", rebuilt=True)
COMPUTATIONAL_COST_DIR = _analysis_dir("computational_cost", rebuilt=True)
# "repair_gim_baseline" is the stage name; "gim_baseline_repair" is the directory name
# the frozen script has always written - the one irregular case `paths.py`'s docstring
# and `stec.runs.restructure_results` both call out by name.
GIM_BASELINE_REPAIR_DIR = _analysis_dir("repair_gim_baseline", rebuilt=False)
DAILY_METRICS_DIR = _analysis_dir("daily_metrics", rebuilt=True)
UNCERTAINTY_ERROR_RELATION_DIR = _analysis_dir(
    "uncertainty_error_relation", rebuilt=True
)
STRATIFIED_COMPARISON_DIR = _analysis_dir("stratified_comparison", rebuilt=True)
ACTIVITY_STRATIFICATION_DIR = _analysis_dir("activity_stratification", rebuilt=True)
IONEX_RMS_BENCHMARK_DIR = _analysis_dir("ionex_rms_benchmark", rebuilt=True)
UNCERTAINTY_CALIBRATION_DIR = _analysis_dir("uncertainty_calibration", rebuilt=True)
MAPPING_FUNCTION_CONSISTENCY_DIR = _analysis_dir(
    "mapping_function_consistency", rebuilt=True
)
MADRIGAL_REFERENCE_OFFSET_DIR = _analysis_dir("madrigal_reference_offset", rebuilt=True)
WEIGHTING_ABLATION_DIR = _analysis_dir("weighting_ablation", rebuilt=True)
STORM_STRATIFICATION_DIR = _analysis_dir("storm_stratification", rebuilt=True)
POSITIONING_ROBUSTNESS_DIR = _analysis_dir("positioning_robustness", rebuilt=True)
POSITIONING_COVERAGE_DIR = _analysis_dir("positioning_coverage", rebuilt=True)
COMMON_SET_POSITIONING_DIR = _analysis_dir("common_set_positioning", rebuilt=True)
POSITIONING_SUMMARY_DIR = _analysis_dir("positioning_summary", rebuilt=True)
ORACLE_BENCHMARK_DIR = _analysis_dir("oracle_benchmark", rebuilt=True)
RESULTS_MANIFEST_DIR = _analysis_dir("results_manifest", rebuilt=True)
PRETRAINED_TEST_DIAGNOSTICS_DIR = _analysis_dir(
    "pretrained_test_diagnostics", rebuilt=True
)
ELEVATION_METRICS_FINETUNED_DIR = _analysis_dir(
    "elevation_metrics_finetuned", rebuilt=True
)
DSTEC_EVALUATION_DIR = _analysis_dir("dstec_evaluation", rebuilt=True)
EPISTEMIC_SCALE_DIAGNOSTIC_DIR = _analysis_dir(
    "epistemic_scale_diagnostic", rebuilt=True
)
# diagnostic_test_observations is diagnostic_figures's own cache, not a separately
# declared stage (see the diagnostic_figures Stage below) - a second, wider-column pass
# over predictions/pretrained_stec/own, kept apart from PRETRAINED_TEST_DIAGNOSTICS_DIR
# so this stage never has to widen the column list stec.viz.manuscript_figures depends
# on. "plots/diagnostics" is a plain repo-relative literal, matching how the
# manuscript_figures Stage below spells its own "plots/manuscript" output.
DIAGNOSTIC_TEST_OBSERVATIONS_DIR = _analysis_dir(
    "diagnostic_test_observations", rebuilt=True
)
DIAGNOSTIC_FIGURES_DIR = "plots/diagnostics"

# The canonical STEC-metrics sweep (CLAUDE.md's "Which results are canonical" table) is a
# full evaluation tree, not a `stec.analysis` output, so it lives under
# `stec_evaluation/`, not `analyses/` - see docs/revision/results_layout.md.
WITH_PRETRAINED_BASELINE_SUMMARY = (
    _rel(paths.STEC_EVALUATION_RESULTS) / "with_pretrained_baseline" / "summary"
)

# stec.training.run_training / stec.inference.run_inference are the driver layer itself -
# every stage above this point reads results that pre-rebuild src/ code produced, because
# nothing under stec/training or stec/inference had a runnable entry point
# (docs/revision/task_board.md S1-4). These two stages exercise that driver for real, but
# deliberately against a tiny checked-in fixture rather than the paper's actual training
# data: a declared stage runs unattended, by default, as part of `pipeline run`, and
# pointing one at the real ~640 GB tree or a GPU-scale training job would make an ordinary
# `pipeline run` silently start hours of compute. The paper's real checkpoints are still
# produced by the unmodified pre-rebuild `src/main.py` (CLAUDE.md's canonical-results
# table); running the same stec/ driver against real data is a deliberate, manual
# invocation with different flags, not something this registry does on its own.
SMOKE_FIXTURE_DIR = "tests/fixtures/pipeline_smoke"
SMOKE_CONFIG = f"{SMOKE_FIXTURE_DIR}/config.yaml"
SMOKE_DATABASE_ROOT = f"{SMOKE_FIXTURE_DIR}/external_data/STEC_DB_CASDCB"
SMOKE_SWI = f"{SMOKE_FIXTURE_DIR}/repo_data/omni_hourly_2010-2025.h5"
SMOKE_CHECKPOINT = (
    "artifacts/models/pipeline_smoke_finetune/model/"
    "finetune_BayesianResNetSTEC_seed42.pth"
)
SMOKE_STORE_DAY = "artifacts/predictions/finetuned_stec/own/year=2024/doy=132.parquet"
SMOKE_IONEX_ROOT = f"{SMOKE_FIXTURE_DIR}/external_data/GIM_IONEX"
SMOKE_VTEC_CONFIG = f"{SMOKE_FIXTURE_DIR}/vtec_config.yaml"
SMOKE_VTEC_CHECKPOINT = (
    f"{SMOKE_FIXTURE_DIR}/vtec_model/finetune_MLP_LaplacianNLL_seed42.pth"
)

# Reused verbatim wherever a Madrigal number is produced. The comparison changes two things
# at once - the model is out of distribution *and* the reference comes from a different
# processing chain - and 45% of the Madrigal RMSE variance is a per-station reference
# offset, established by the model and the IGS GIM disagreeing with Madrigal identically
# (corr +0.946 over 66 stations).
MADRIGAL_CAVEAT = [
    "Read only alongside madrigal_reference_offset, never standalone: 45% of the RMSE "
    "variance is a per-station reference offset, not model error.",
    "Does not support claims about the model's out-of-distribution uncertainty - dataset "
    "shift and reference-chain difference are confounded here.",
]


# --- Check callables -----------------------------------------------------------------
#
# `min_rows` catches a truncated or empty CSV; it cannot catch one that is the right
# shape but the wrong content - e.g. every `reindex`-guaranteed row present but empty,
# because the input this stage read had nothing for that method. These are the first
# uses of the `checks` field (docs/revision/independent_audit.md F4/F5 found it unused
# on all 34 stages), added to the two canonical stages most exposed to a silently empty
# or partial store: `daily_metrics` (Tables 3/4) and `positioning_summary` (Table 5).


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _missing_csv_columns(path: Path, required: Sequence[str]) -> list[str]:
    """Header columns from `required` absent from `path`.

    Checked before any row is indexed by name. `run_checks` (stec/pipeline/runner.py)
    catches only `AssertionFailed`/`CheckFailed`; a bare `row["Model"]` on a CSV whose
    column was renamed or dropped raises `KeyError`, which escapes uncaught and kills
    the whole `--keep-going` run instead of failing the one stage. Reading the header
    once, rather than every row, is also cheap enough to do unconditionally.
    """
    with path.open(newline="") as handle:
        fieldnames = csv.DictReader(handle).fieldnames or []
    return [column for column in required if column not in fieldnames]


def daily_metrics_summary_has_all_methods_and_datasets(outputs: dict) -> str | None:
    """Tables 3/4's summary.csv must report all four methods on both datasets, not
    merely clear a row-count floor - a store that silently lost its Madrigal partition
    would still write a plausible-looking summary.csv of `own`-only rows.

    Checks marginal coverage only - every model appears in at least one dataset row,
    every dataset appears in at least one model row - not the full 4x2 cross-product.
    This is deliberate, not an oversight: the real store's summary.csv has 7 rows, not
    8, because "Pretrained STEC" x madrigal is legitimately absent -
    predictions/pretrained_stec/madrigal has never been built as a real partition (see
    predictions/pretrained_stec/madrigal/README.md: the one file there is an orphan
    day with no baseline columns, explicitly "not started-and-consistent", and
    CLAUDE.md's prediction-store section confirms the partition "has not been built
    yet"). A strict cross-product check would fail this stage permanently until that
    partition exists, which is a data-availability fact, not a daily_metrics
    correctness bug this stage should be blocked on. tests/pipeline/test_stages.py
    pins both halves of this trade-off: the known real-world 7-row shape passes by
    design, and a single missing (model, dataset) cell elsewhere in the table does
    not fail the check - documented rather than silently absent. Tighten this to a
    cross-product check, with the pretrained/madrigal cell in an explicit
    allowed-missing set (never a silent exception), once that partition is actually
    built end to end.
    """
    path = DAILY_METRICS_DIR / "summary.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(path, ["Model", "dataset"])
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = _read_csv_rows(path)
    seen_models = {row["Model"] for row in rows}
    seen_datasets = {row["dataset"] for row in rows}
    missing_models = set(MODELS.values()) - seen_models
    missing_datasets = set(DATASET_LABELS.values()) - seen_datasets
    if missing_models or missing_datasets:
        return (
            f"{path} is missing model(s) {sorted(missing_models)} and/or "
            f"dataset(s) {sorted(missing_datasets)}"
        )
    return None


def positioning_summary_overall_has_all_four_methods(outputs: dict) -> str | None:
    """Table 5's overall.csv always has exactly 4 rows - `reindex(METHOD_ORDER)`
    guarantees the index even when a method has no station-days, filling it with NaN
    rather than dropping the row. `min_rows=4` alone cannot tell that apart from 4 real
    rows, so this checks `station_days` is actually populated for every method."""
    path = POSITIONING_SUMMARY_DIR / "overall.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(path, ["Method"])
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = {row["Method"]: row for row in _read_csv_rows(path)}
    missing = set(METHOD_ORDER) - set(rows)
    if missing:
        return f"{path} is missing method(s) {sorted(missing)}"
    empty = [method for method in METHOD_ORDER if not rows[method].get("station_days")]
    if empty:
        return f"{path} has no station_days recorded for {sorted(empty)}"
    return None


STAGES: list[Stage] = [
    Stage(
        "training_smoke",
        f"-m stec.training.run_training --config {SMOKE_CONFIG} "
        f"--database-root {SMOKE_DATABASE_ROOT} --space-weather {SMOKE_SWI} "
        "--device cpu",
        "-",
        "proves stec.training.run_training wires fit/loss/schedulers into a real "
        "checkpoint end to end, on a tiny checked-in fixture day",
        inputs=[SMOKE_FIXTURE_DIR],
        outputs=[
            SMOKE_CHECKPOINT,
            "artifacts/models/pipeline_smoke_finetune/loss_history.csv",
        ],
        # 1 row = the fixture config's single fine-tune epoch (finetune.epochs: 1).
        min_rows={"artifacts/models/pipeline_smoke_finetune/loss_history.csv": 1},
        caveats=[
            "Runs the real driver against a tiny checked-in fixture "
            "(tests/fixtures/pipeline_smoke, 200 synthetic observations, hidden_dim=8), "
            "not the paper's actual training data - deliberately, so this stage stays "
            "safe to run unattended (CPU, sub-second) rather than starting a GPU-scale "
            "job the moment it is declared. The paper's checkpoints come from "
            "src/main.py, unmodified; retraining them through this driver against real "
            "data is a separate, manual invocation - see the module docstring.",
            "fit() runs every configured epoch and returns the final weights: no "
            "best-checkpoint selection, no early stopping. Every shipped checkpoint was "
            "instead selected by BaseTrainer.run_training's best-val-loss tracking, so "
            "a checkpoint this driver produces from real data would not be a byte-for-"
            "byte reproduction of one already on disk - only the loss trajectory is "
            "gate-verified equivalent (Gate C).",
            "training.log_target and <mode>.freeze_body are refused, not silently "
            "ignored, if a config sets them: neither transform is ported.",
            "Writes under artifacts/models/, not experiments/ - isolated from every "
            "checkpoint any other stage or analysis reads.",
        ],
    ),
    Stage(
        "inference_smoke",
        f"-m stec.inference.run_inference --config {SMOKE_CONFIG} "
        f"--checkpoint {SMOKE_CHECKPOINT} --model-variant finetuned_stec --dataset own "
        f"--doys 2024:132 --database-root {SMOKE_DATABASE_ROOT} "
        f"--space-weather {SMOKE_SWI} "
        "--output-dir artifacts/predictions/pipeline_smoke_inference --device cpu",
        "-",
        "proves stec.inference.run_inference wires monte_carlo into the prediction "
        "store end to end, on the same tiny checked-in fixture day",
        inputs=[SMOKE_CHECKPOINT, SMOKE_FIXTURE_DIR],
        # SMOKE_STORE_DAY (the parquet itself) is deliberately *not* declared here any
        # more, even though this stage is what creates it - baselines_smoke (below) reads
        # that same file and merges VTEC/GIM columns into it in place, which changes its
        # digest. Declaring it as this stage's output too would make outputs_intact()
        # compare a stale recorded digest against baselines_smoke's edit and conclude
        # inference_smoke's result had been "modified", forcing a rerun that would
        # silently wipe the merged baseline columns back out on the very next `pipeline
        # run` - a stage reporting success while serving a stale file, the exact failure
        # class the registry exists to prevent. The manifest CSV is not touched by
        # baselines_smoke and is sufficient on its own: main()'s run_inference() writes
        # the parquet via write_predictions() and only appends to the manifest after that
        # succeeds for every day, so a manifest with the right row count already implies
        # the parquet was written - nothing is lost by not asserting the parquet directly.
        outputs=[
            "artifacts/predictions/pipeline_smoke_inference/inference_manifest.csv",
        ],
        # Keyed on the manifest, not the parquet: a parquet output carries no row count
        # in the pipeline's provenance record (output_record only counts rows for .csv),
        # which is why this stage writes a manifest CSV alongside the store file at all.
        min_rows={
            "artifacts/predictions/pipeline_smoke_inference/inference_manifest.csv": 1
        },
        caveats=[
            "Runs against the same tiny fixture training_smoke does, and against the "
            "checkpoint training_smoke just produced from it - not the paper's real "
            "checkpoints or test set. See training_smoke's caveats for why.",
            "This smoke run only exercises '--dataset own', because the checked-in "
            "fixture (tests/fixtures/pipeline_smoke) has no Madrigal day to read. "
            "'--dataset madrigal' itself no longer raises NotImplementedError: "
            "stec.data.madrigal_reader.read_madrigal_day is a real model-input reader "
            "since the Madrigal-identity work landed, and stec.inference.run_inference "
            "supports both datasets. predictions/pretrained_stec/madrigal/ still has no "
            "data, but only because that backfill has not been run yet, not because the "
            "dataset is unsupported.",
            "Writes under artifacts/predictions/, the default "
            "prediction_store.DEFAULT_STORE_ROOT - a different tree from the legacy "
            "predictions/ every analysis stage above reads (STORE_OWN etc.), so this "
            "stage's output is never picked up by daily_metrics or any other analysis.",
            "Runs the zero-perturbation control (stec/models/determinism.py) before any "
            "real sampling and fails loudly if it is not exactly 0.0 - the Bayesian A/B "
            "invariant CLAUDE.md requires for every comparison built on this model.",
            "Writes SMOKE_STORE_DAY (the store parquet), which baselines_smoke (below) "
            "then reads and mutates in place - see that stage's own caveats.",
        ],
    ),
    Stage(
        "baselines_smoke",
        f"-m stec.inference.run_baselines --model-variant finetuned_stec --dataset own "
        f"--doys 2024:132 --vtec-config {SMOKE_VTEC_CONFIG} "
        f"--vtec-checkpoint {SMOKE_VTEC_CHECKPOINT} --store-root artifacts/predictions "
        f"--database-root {SMOKE_DATABASE_ROOT} --space-weather {SMOKE_SWI} "
        f"--ionex-root {SMOKE_IONEX_ROOT} "
        "--output-dir artifacts/predictions/pipeline_smoke_baselines --device cpu",
        "-",
        "proves stec.inference.run_baselines wires the VTEC + GIM baselines into the "
        "prediction store end to end, on the same tiny checked-in fixture day",
        # Deliberately does NOT list SMOKE_STORE_DAY as an input, even though this stage
        # reads it: this stage also *writes* SMOKE_STORE_DAY (declared below, in
        # outputs), and an input fingerprint is captured before the command runs, then
        # compared again on the next invocation - if the mutated file were also an input,
        # every run would see "the input changed" (because the *previous* run changed it)
        # and rerun forever, unconditionally. SMOKE_CHECKPOINT and inference_smoke's
        # manifest stand in for "the upstream STEC step is current" without that
        # self-reference; the VTEC checkpoint/config and IONEX fixture are the inputs
        # genuinely unique to this stage.
        inputs=[
            SMOKE_CHECKPOINT,
            "artifacts/predictions/pipeline_smoke_inference/inference_manifest.csv",
            SMOKE_VTEC_CHECKPOINT,
            SMOKE_VTEC_CONFIG,
            SMOKE_IONEX_ROOT,
        ],
        # SMOKE_STORE_DAY is claimed here, not by inference_smoke - see that stage's own
        # comment on why only one of the two may own it.
        outputs=[
            SMOKE_STORE_DAY,
            "artifacts/predictions/pipeline_smoke_baselines/baselines_manifest.csv",
        ],
        min_rows={
            "artifacts/predictions/pipeline_smoke_baselines/baselines_manifest.csv": 1
        },
        caveats=[
            "Runs against the same tiny fixture inference_smoke does, and against the "
            "store file inference_smoke just wrote - not the paper's real checkpoints, "
            "VTEC ensemble or test set. Must run after inference_smoke (list order): "
            "add_baselines_for_day raises FileNotFoundError otherwise, rather than "
            "silently succeeding against a day the STEC model has not produced yet.",
            "Loads a single VTEC checkpoint, not the real 10-seed ensemble - "
            "load_vtec_model wraps >1 checkpoint in DeepEnsemble automatically, but this "
            "fixture carries only one, so vtec_model_stec_epistemic_unc is exactly 0.0 "
            "here by construction. The ensemble path (module docstring requirement 4, the "
            "single-checkpoint-silently-reproduces-one-member regression) is covered by "
            "tests/inference/test_run_baselines.py's dedicated ensemble tests, not by "
            "this stage - a real per-DOY VTEC directory on this host still carries all 10 "
            "seeds, and --experiments-root (unused here) is what resolves that "
            "canonically for a real invocation.",
            "'--dataset own' only, same reason as inference_smoke's caveat: the fixture "
            "has no Madrigal day.",
            "Runs the zero-perturbation control on the VTEC model before any real "
            "sampling, same invariant inference_smoke checks for the STEC model.",
            "Skip detection has one narrow blind spot: it is keyed on "
            "inference_smoke's manifest CSV, not the mutable parquet (see this stage's "
            "own inputs comment). If that manifest were deleted without also changing "
            "SMOKE_CHECKPOINT, inference_smoke would rewrite an identical manifest and "
            "this stage's own input fingerprint would not change, even though "
            "inference_smoke's rerun just reverted the parquet to STEC-only columns - a "
            "case only a manual, partial deletion of pipeline state could trigger.",
        ],
    ),
    Stage(
        "paper_tables",
        f"-m stec.analysis.paper_tables --config {_rel(paths.PAPER_PRETRAINED_CONFIG)} "
        f"--output-dir {PAPER_TABLES_DIR}",
        "Tables 1, 2",
        "input feature list and hyperparameters, generated from the model rather than "
        "maintained beside it",
        inputs=[str(_rel(paths.PAPER_PRETRAINED_CONFIG))],
        outputs=[
            str(PAPER_TABLES_DIR),
            str(PAPER_TABLES_DIR / "table1_features.csv"),
            str(PAPER_TABLES_DIR / "table2_hyperparameters.csv"),
        ],
        # Both tables are close to a fixed size (one row per input block / per
        # hyperparameter, driven by the frozen paper config, not by data volume) -
        # floored comfortably below the real counts (24 and 22 rows respectively) so
        # this catches a header-only write without pinning an exact count a harmless
        # config edit could shift.
        min_rows={
            str(PAPER_TABLES_DIR / "table1_features.csv"): 10,
            str(PAPER_TABLES_DIR / "table2_hyperparameters.csv"): 15,
        },
        canonical_for="Tables 1 and 2",
        caveats=[
            "Generated from a frozen, checked-in copy of the paper's own stored run "
            "config (config/paper/pretrain_stec_config.yaml), not from a hand-maintained "
            "template in config/ and not from the legacy experiments/ tree - this is what "
            "makes the stage runnable on a clean clone with no data mounted. The template "
            "disagreed with the real run on 7 of 8 fields - architecture, prior sigma, "
            "learning rate, batch size, scheduler, SH degree and KL weight - so the table "
            "it used to produce described a model that was never trained.",
            "Both training stages are reported. The paper pretrains and then fine-tunes "
            "daily at a different learning rate, batch size and epoch count; a table "
            "carrying one of them describes half the training.",
            "Table 2 includes three hyperparameters the submitted manuscript omits: the "
            "KL warmup (0 to 0.1 over 5 epochs), the variance floor, and the output bias "
            "initialisation.",
        ],
    ),
    Stage(
        "relative_error_metrics",
        f"-m stec.analysis.relative_error_metrics --output-dir {RELATIVE_ERROR_METRICS_DIR}",
        "R2.1, R2.2",
        "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
        outputs=[
            str(RELATIVE_ERROR_METRICS_DIR),
            str(RELATIVE_ERROR_METRICS_DIR / "yearly_metrics.csv"),
        ],
        # Keyed on the CSV, not the directory: a tree digest carries files/size/mtime but
        # no row count, so a min_rows on a directory can never be satisfied and the stage
        # fails however well it ran.
        min_rows={str(RELATIVE_ERROR_METRICS_DIR / "yearly_metrics.csv"): 5},
    ),
    Stage(
        "temporal_regime_split",
        f"-m stec.analysis.temporal_regime_split --output-dir {TEMPORAL_REGIME_SPLIT_DIR}",
        "R2.1",
        "interpolation vs extrapolation regime comparison, recomputed from the "
        "prediction store instead of src/'s temporal_analysis text files",
        inputs=[STORE_PRETRAINED],
        outputs=[str(TEMPORAL_REGIME_SPLIT_DIR / "temporal_regime_comparison.csv")],
        # Always exactly 2 rows (interpolation, extrapolation) - not a floor.
        min_rows={str(TEMPORAL_REGIME_SPLIT_DIR / "temporal_regime_comparison.csv"): 2},
        canonical_for="R2.1 interpolation/extrapolation temporal split",
        caveats=[
            "Population: every year present in predictions/pretrained_stec/own - "
            "2014-2024, 544 days (302 interpolation days 2014-2023, 242 extrapolation "
            "days 2024) - not the 2024-only period the four-method comparisons "
            "(daily_metrics etc.) use. Deliberate, not an oversight: the "
            "interpolation/extrapolation question this stage answers does not exist "
            "without years on both sides of the 2024-05-01 boundary, so the full "
            "held-out range is the right scope for this standalone characterisation of "
            "the pretrained model, not a population matched to the other baselines.",
            "Answers the R2.1 reviewer-response number (14.05 vs 7.65 TECU, 26.9% vs "
            "31.0% normalised), not a printed manuscript table - the manuscript has 5 "
            "tables and no lettered appendix.",
            "relative_error_metrics (above) already writes a same-named "
            "temporal_regime_comparison.csv into its own directory and already answers "
            "R2.1's numbers, but by parsing total_metrics_summary.txt files that "
            "src/inference_testset.py's live run wrote under experiments/ - a reader of "
            "src/'s output, not an independent computation, and it cannot run once src/ "
            "or that experiment directory is gone. This stage computes the same "
            "comparison directly from predictions/pretrained_stec/own, with no src/ "
            "dependency. relative_error_metrics's own Stage leaves canonical_for unset, "
            "so the two do not collide, but a human should eventually decide whether "
            "relative_error_metrics's regime half is worth keeping once src/ retires.",
            "src/'s split_test_data_by_date builds year/doy from a truncating int() on "
            "a denormalised float, the same class of bug repair_gim_baseline exists to "
            "fix elsewhere - but it was never fixed at this site. This stage reads "
            "year/doy from the store's own directory partition instead (authoritative, "
            "never reconstructed from a float), which sidesteps the bug rather than "
            "reproducing it. Verified to reproduce the src/-produced CSV's RMSE to "
            "4 decimal places and row counts exactly (4,400,934 / 5,599,066) - this "
            "particular boundary (2024-05-01) does not fall on one of the truncation-"
            "affected days, so the bug happens not to move this specific number, but "
            "that is a property of this boundary, not a guarantee of the original code.",
            "This output must not be read as evidence about temporal extrapolation on "
            "its own. The regime split collapses onto a calendar-year split (2024's test "
            "set starts exactly at the 2024-05-01 boundary), so it is perfectly "
            "confounded with solar-cycle phase: every extrapolation day is 2024, the "
            "most active year in the record, and no interpolation day is. "
            "temporal_regime_activity_matched (below) stratifies by F10.7 to check "
            "whether the gap survives at matched activity - read the two together.",
        ],
    ),
    Stage(
        "temporal_regime_activity_matched",
        f"-m stec.analysis.temporal_regime_activity_matched "
        f"--output-dir {TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR}",
        "R2.1",
        "corrects the R2.1 interpolation/extrapolation comparison for its solar-cycle "
        "confound by stratifying on F10.7 before comparing regimes",
        inputs=[STORE_PRETRAINED],
        outputs=[
            str(TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR / "yearly_magnitude.csv"),
            str(
                TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR / "activity_matched_comparison.csv"
            ),
        ],
        # yearly_magnitude.csv: exactly one row per year present in the store (11,
        # 2014-2024) - not a floor, but 10 is a safe minimum in case a future rebuild
        # briefly holds fewer. activity_matched_comparison.csv: one row per (F10.7 band,
        # regime) pair that is non-empty - at most 4 bands x 2 regimes = 8, currently 6
        # because two bands hold only one regime each.
        min_rows={
            str(TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR / "yearly_magnitude.csv"): 10,
            str(
                TEMPORAL_REGIME_ACTIVITY_MATCHED_DIR / "activity_matched_comparison.csv"
            ): 4,
        },
        canonical_for="R2.1 interpolation/extrapolation temporal split, "
        "activity-matched correction",
        caveats=[
            "Same population as temporal_regime_split, for the same reason: "
            "predictions/pretrained_stec/own's full 2014-2024, 544 days, not the "
            "2024-only period the four-method comparisons use. The activity-matched "
            "correction needs both regimes' full day lists to find whatever F10.7 "
            "overlap exists between them, so it cannot be narrowed to a matched "
            "population the way a cross-model comparison would be.",
            "Does not supersede temporal_regime_split - that stage reproduces the "
            "published R2.1 headline number faithfully and is kept for provenance. This "
            "stage is the corrected interpretation: two of the four fixed F10.7 bands "
            "(below 100 sfu, at or above 200 sfu) are structurally unmatched - one "
            "regime holds every day in that band and the other holds none - and those "
            "two bands alone cover 55% of all observations. Only the middle two bands "
            "contain both regimes, and the arms there are unbalanced (7 extrapolation "
            "days against 70 interpolation days in the lower one).",
            "Where a matched comparison is possible at all, extrapolation's normalised "
            "error runs slightly lower than interpolation's, the same direction as the "
            "unmatched headline - this does not mean the confound is resolved in the "
            "model's favour, only that matching does not reverse the naive result "
            "either. The 7-day extrapolation arm in the lower matched band is too thin "
            "to draw a conclusion from on its own.",
            "The defensible reading, and the one this stage's output supports: this "
            "test set cannot cleanly isolate a temporal-extrapolation effect from the "
            "solar-cycle confound, because the only out-of-training-window year is also "
            "the only high-activity year in the test period. That is a limitation of the "
            "test set, not a result about the model, and should be reported to "
            "reviewers as such rather than as a positive finding either way.",
        ],
    ),
    Stage(
        "hyperparameter_search",
        # --wandb_dir passed explicitly: the frozen script's default (`Path("wandb")`)
        # resolves against cwd, and the runner's cwd is REPO_ROOT, which has no wandb/ in
        # this worktree - the run history lives only in the primary checkout.
        # paths.LEGACY_WANDB honours STEC_LEGACY_ROOT, same as repair_gim_baseline's
        # --store_root above. --output_dir is the script's own existing flag (not a new
        # one added for this), so redirecting it into the new layout needs no edit to the
        # frozen script. The script itself now lives at stec/frozen/analysis/ (moved off
        # src/ byte-identically, see stec/frozen/README.md) rather than src/analysis/, so
        # this stage carries no src/ dependency at all - the only src/ dependency left in
        # this file is repair_gim_baseline's, and that one moved with it too.
        f"stec/frozen/analysis/hyperparameter_search_summary.py "
        f"--wandb_dir {paths.LEGACY_WANDB} --output_dir {HYPERPARAMETER_SEARCH_DIR}",
        "R2.5, R2.8b",
        "architecture comparison and hyperparameter sweep from the W&B history",
        inputs=["wandb"],
        outputs=[str(HYPERPARAMETER_SEARCH_DIR)],
        caveats=[
            "Stays unported for a data reason, not a code one: the script itself is "
            "self-contained (only glob/yaml/json/pandas over local wandb/run-*/files/"
            "{config.yaml,wandb-summary.json} pairs, no W&B API call and no network "
            "access) and never depended on the rest of src/, so porting it would be "
            "mechanical. It has not been ported because its input is not reachable "
            "here: wandb/ is untracked (.gitignore'd, ~606 MB, ~1,526 run directories "
            "in the live checkout) and does not exist in this worktree or in any fresh "
            "clone.",
            "Cannot run at all without a populated local wandb/ directory from the "
            "training host.",
        ],
    ),
    Stage(
        "station_independence",
        f"-m stec.analysis.station_independence --output-dir {STATION_INDEPENDENCE_DIR}",
        "R2.3",
        "test-station error against distance to the nearest training station",
        inputs=[STORE_OWN],
        outputs=[str(STATION_INDEPENDENCE_DIR)],
        caveats=[
            "Limited by n = 55 test stations, not by observation count. Adding days "
            "sharpens each point but does not sharpen the Spearman coefficient.",
            "Strengthening this result needs a region-held-out retrain, not more data.",
        ],
    ),
    Stage(
        "computational_cost",
        f"-m stec.analysis.computational_cost --output-dir {COMPUTATIONAL_COST_DIR}",
        "R2.8h",
        "training and inference cost",
        outputs=[str(COMPUTATIONAL_COST_DIR)],
    ),
    Stage(
        "repair_gim_baseline",
        # --store_root passed explicitly: the frozen script's default (`Path("predictions")`)
        # resolves against cwd, and the runner's cwd is REPO_ROOT - which has no predictions/
        # in a worktree that does not carry the 640 GB data tree. paths.LEGACY_PREDICTIONS
        # already honours STEC_LEGACY_ROOT, so this points at the real store without editing
        # the frozen script itself. --output_dir is likewise the script's own existing flag.
        # The script moved from src/analysis/ to stec/frozen/analysis/ byte-identically
        # (checksum-verified, see stec/frozen/README.md) so src/ can be deleted whole
        # without breaking this stage - its own sibling import of `evaluation.*` moved
        # with it, to stec/frozen/evaluation/, for the same reason: this stage must never
        # import stec/'s ported GIMMapper or prediction_store, or the regression check
        # would share an implementation with the thing it checks.
        f"stec/frozen/analysis/repair_gim_baseline.py --apply "
        f"--store_root {paths.LEGACY_PREDICTIONS} --output_dir {GIM_BASELINE_REPAIR_DIR}",
        "Table 4, R1.4",
        "recompute the IGS GIM baseline against the correct day's IONEX map",
        inputs=[STORE_OWN],
        outputs=[str(GIM_BASELINE_REPAIR_DIR)],
        caveats=[
            "Repairs the 12 days of 2024 where a truncating cast on a float32-denormalised "
            "doy loaded the previous day's IONEX map. Unaffected days must reproduce to "
            "~1e-5 TECU; that agreement is the regression check.",
            "Must stay on the frozen, pre-rebuild script, permanently: this stage is the "
            "regression check for the GIM day-lookup repair, and porting it into stec/ "
            "would mean the check and the thing it checks share an implementation.",
        ],
    ),
    Stage(
        # Ported. Verified to reproduce the pre-rebuild implementation exactly - delta 0.0
        # on RMSE_mean, pooled_RMSE, MAE_mean, R2_mean, day and observation counts, across
        # all seven model/dataset combinations over 242 days and 475,111,413 observations.
        "daily_metrics",
        f"-m stec.analysis.daily_metrics --output-dir {DAILY_METRICS_DIR}",
        "Tables 3, 4",
        "per-day and pooled STEC metrics recomputed from the prediction store",
        # STORE_MADRIGAL belongs here as much as STORE_OWN/STORE_PRETRAINED do:
        # daily_metrics.py's own DATASET_LABELS loops "own" and "madrigal" under the same
        # --model-variant (default finetuned_stec) in one invocation, so Table 4's rows
        # come from this same command. Omitting it meant a change to
        # predictions/finetuned_stec/madrigal/ (e.g. the divergence #12 local-time
        # correction) would not move this stage's input fingerprint, and `pipeline run`
        # would skip it - reporting Table 4 as up to date while it was still stale.
        inputs=[
            STORE_OWN,
            STORE_PRETRAINED,
            STORE_MADRIGAL,
            str(GIM_BASELINE_REPAIR_DIR),
        ],
        # summary.csv and per_day.csv declared individually, not only the parent
        # directory: a directory output carries no row count
        # (provenance.output_record only counts rows for a `.csv` file), so
        # `min_rows={}` used to be the only option here - the audit's smoking gun
        # (docs/revision/independent_audit.md F4/F5): a stage canonical for Tables 3/4
        # could record success against an empty or missing store with nothing to catch
        # it. Both files are exactly what daily_metrics.py already writes into this
        # directory, so declaring them adds no new files on disk and does not change
        # this directory's tree digest as an input elsewhere (activity_stratification
        # reads the directory as a whole).
        outputs=[
            str(DAILY_METRICS_DIR),
            str(DAILY_METRICS_DIR / "summary.csv"),
            str(DAILY_METRICS_DIR / "per_day.csv"),
        ],
        min_rows={
            # 4 methods x 2 datasets = 8 rows when every method reports on both
            # datasets - floored below that (8 is already too small for a real
            # "order of magnitude" margin) so this still catches a near-empty or
            # missing-store run without demanding every method survive every rebuild.
            str(DAILY_METRICS_DIR / "summary.csv"): 6,
            # 4 methods x 242 own-days x 2 datasets is on the order of 1,900 rows when
            # every model/day/dataset combination reports - floored an order of
            # magnitude below that.
            str(DAILY_METRICS_DIR / "per_day.csv"): 190,
        },
        checks=[daily_metrics_summary_has_all_methods_and_datasets],
        canonical_for="Tables 3 and 4",
        caveats=[
            "The published RMSE is RMSE_mean - the mean of per-day RMSEs - which is what "
            "the manuscript states. pooled_RMSE in the same file is over observations and "
            "is consistently higher; the two are not interchangeable.",
            "Madrigal rows carry the madrigal_reference_offset caveat.",
        ],
        supersedes=[str(WITH_PRETRAINED_BASELINE_SUMMARY / "summary_statistics.csv")],
    ),
    Stage(
        "uncertainty_error_relation",
        f"-m stec.analysis.uncertainty_error_relation --output-dir {UNCERTAINTY_ERROR_RELATION_DIR}",
        "R2.6, R1.2",
        "predicted uncertainty against realised error, pooled over the test period",
        inputs=[STORE_OWN],
        outputs=[str(UNCERTAINTY_ERROR_RELATION_DIR)],
        caveats=[
            "Bins are now fixed TECU intervals (0-1-2-3-4-5-7-10-15-20-30-inf), not the "
            "previous first-day sigma deciles. Those deciles were computed from "
            "DOY 122's pred_total_unc distribution alone and reused unchanged for the "
            "other 241 days, so a bin labelled 'top decile' held 6.88%-18.80% of the "
            "full-year population rather than 10% - a 'decile' meant something "
            "different on every day but the first.",
        ],
    ),
    Stage(
        "stratified_comparison",
        f"-m stec.analysis.stratified_comparison --output-dir {STRATIFIED_COMPARISON_DIR}",
        "R1.4",
        "all four methods by elevation, geomagnetic latitude, local time and season",
        # Not STORE_PRETRAINED: the invoked command takes no --model-variant/--dataset
        # flags, so it runs against the defaults (finetuned_stec/own = STORE_OWN) only.
        # "Pretrained Direct STEC" in stratified_comparison.py's own METHODS dict reads
        # the pretrained_stec_pred *column*, merged into STORE_OWN's own parquet files
        # alongside stec_pred - the same arrangement daily_metrics.py documents for its
        # own MODELS dict - never a separate read of predictions/pretrained_stec/own.
        # The mirror of the oracle_benchmark defect (see that stage's own comment and
        # tests/pipeline/test_stages.py's dedicated tests): there a real input was
        # undeclared; here a non-input was declared, which can never fail to fingerprint
        # correctly (STORE_PRETRAINED changing is irrelevant to this stage) but invites
        # exactly the same "declared input the module never reads" confusion.
        inputs=[STORE_OWN],
        outputs=[str(STRATIFIED_COMPARISON_DIR)],
    ),
    Stage(
        "activity_stratification",
        f"-m stec.analysis.activity_stratification --output-dir {ACTIVITY_STRATIFICATION_DIR}",
        "R1.4",
        "STEC error stratified by Dst and F10.7",
        inputs=[str(DAILY_METRICS_DIR), SWI],
        outputs=[str(ACTIVITY_STRATIFICATION_DIR)],
        caveats=[
            "Reads the repaired GIM values. Run after repair_gim_baseline: the "
            "un-repaired baseline reversed this comparison's conclusion.",
            "F10.7 bins are now fixed absolute bands (0/100/150/200/1000 sfu), not the "
            "previous data-derived terciles. The terciles split the test period 81/81/80 "
            "by construction - three equal-population groups regardless of what F10.7 "
            "actually did - while the real distribution against the fixed bands is "
            "7/108/127: the old bins ranked the sample rather than the activity level, "
            "and would label a different third of any other period 'high'.",
        ],
    ),
    Stage(
        "ionex_rms_benchmark",
        f"-m stec.analysis.ionex_rms_benchmark --output_dir {IONEX_RMS_BENCHMARK_DIR}",
        "R1.6b",
        "model uncertainty against the IGS and CODE GIM RMS maps",
        inputs=[STORE_OWN],
        outputs=[str(IONEX_RMS_BENCHMARK_DIR)],
    ),
    Stage(
        # Ported. Scores every model under both Gaussian and Laplace, tagging which is
        # native, so the mis-specified number sits beside the correct one. Also
        # restratifies every (model, family) accumulation by geomagnetic regime -
        # "all" plus the daily-Dst "quiet"/"storm" split from the pre-rebuild source,
        # answering R1.6's "uncertainty behaviour under ... disturbed conditions".
        #
        # main() scores exactly one --model-variant per invocation, so covering the
        # pretrained variant this stage's docstring promises ("scored by pointing
        # --model-variant at its own store partition") takes a second Stage below,
        # not a second inputs entry here - a single stage issuing two invocations
        # would blur "what produced this output" back into the ambiguity the
        # provenance record exists to remove (one command, one fingerprint, one
        # duration per stage.json). Each stage's outputs is scoped to the
        # <variant>_<dataset>/ subdirectory its own invocation writes, per the
        # registry's one-owner-per-output check.
        "uncertainty_calibration",
        f"-m stec.analysis.uncertainty_calibration "
        f"--output-dir {UNCERTAINTY_CALIBRATION_DIR} "
        "--model-variant finetuned_stec --dataset own",
        "R1.6, R2.6",
        "coverage, PIT and CRPS for Direct STEC and VTEC + Mapping, each under its "
        "own likelihood and regime",
        inputs=[STORE_OWN, SWI],
        outputs=[str(UNCERTAINTY_CALIBRATION_DIR / "finetuned_stec_own")],
        caveats=[
            "The VTEC baseline is a Laplace, and its stored vtec_model_stec_total_unc is "
            "already a standard deviation (sqrt(2) * scale), not the raw scale - recover "
            "the scale as std / sqrt(2) before any Laplace formula sees it.",
            "Scored as a Gaussian the same data reads 90% coverage at nominal 50% against "
            "82% under Laplace; both scorings are emitted side by side, tagged by which "
            "likelihood is native to each model.",
            "The storm/quiet split uses the daily minimum-Dst rule at -50 nT, matching "
            "storm_stratification.STORM_DST_THRESHOLD_NT - not the per-observation rule "
            "in scenario_evaluation.py.",
        ],
    ),
    Stage(
        # The pretrained model's own uncertainty is not carried alongside stec_pred in
        # the finetuned_stec store (see ionex_rms_benchmark.py's PRODUCTS comment), so
        # scoring it means reading pretrained_stec/own directly, via the second
        # invocation the module docstring documents.
        "uncertainty_calibration_pretrained",
        f"-m stec.analysis.uncertainty_calibration "
        f"--output-dir {UNCERTAINTY_CALIBRATION_DIR} "
        "--model-variant pretrained_stec --dataset own",
        "R1.6, R2.6",
        "coverage, PIT and CRPS for the pretrained variant, same likelihoods and regimes",
        inputs=[STORE_PRETRAINED, SWI],
        outputs=[str(UNCERTAINTY_CALIBRATION_DIR / "pretrained_stec_own")],
        caveats=[
            "No --year passed, so this scores every year present in "
            "predictions/pretrained_stec/own by default: 2014-2024, 544 days - not the "
            "2024 alone that used to be main()'s silent --year default (uncertainty_"
            "calibration.py's 'Coverage default' docstring section and its regression "
            "test name this as the fixed 44%-of-partition defect: 242 of 544 days is "
            "44.5%). Deliberate, not an oversight: this is a standalone characterisation "
            "of the pretrained checkpoint's own calibration, not a comparison against "
            "the other three baselines, so nothing requires it to match "
            "finetuned_stec/own's narrower 2024-only test period.",
            "Only 'Direct STEC' scores here - pretrained_stec/own carries stec_pred and "
            "pred_total_unc but no vtec_model_stec column, so 'VTEC + Mapping' is absent "
            "from this variant's coverage.csv/scores.csv. Expected, not a bug: VTEC + "
            "Mapping is a separate baseline, not a per-STEC-variant one.",
            "Scores a different model from uncertainty_calibration's Direct STEC - the "
            "pretrained checkpoint before daily fine-tuning, not a second reading of the "
            "same predictions.",
        ],
    ),
    Stage(
        # Declared after the fact: this diagnostic was run manually against the real
        # store on 2026-08-24 (commit 2b7172b), before this Stage existed, so the
        # existing multiday_results/analyses/epistemic_scale_diagnostic/rebuilt/
        # output on disk has no .pipeline/epistemic_scale_diagnostic.json - `pipeline
        # status` reporting "never run" for a real, already-answered question is
        # correct here, not a regression; only a fresh run through this Stage earns a
        # provenance record.
        "epistemic_scale_diagnostic",
        f"-m stec.analysis.epistemic_scale_diagnostic --output-dir {EPISTEMIC_SCALE_DIAGNOSTIC_DIR}",
        "R1.2",
        "is the paper model's epistemic under-dispersion fixable with a post-hoc "
        "scalar, or is the frozen deterministic backbone missing information no "
        "rescaling can recover",
        inputs=[STORE_PRETRAINED, STORE_PRETRAINED_BNN_NLL],
        outputs=[
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR),
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "sweep_paper_model.csv"),
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "sweep_fully_bayesian_reference.csv"),
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_elevation.csv"),
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_geomag_lat.csv"),
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_year.csv"),
        ],
        min_rows={
            # SCALE_GRID is a fixed 40-point grid (35 geomspace points + 5 tail
            # points), independent of the data - exactly 40 whenever the sweep runs at
            # all, for both the paper model and the fully-Bayesian reference.
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "sweep_paper_model.csv"): 40,
            str(
                EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "sweep_fully_bayesian_reference.csv"
            ): 40,
            # Stratified tables drop a bin below MIN_STRATUM_OBSERVATIONS (5,000) -
            # floored well below the fixed bin counts (9 elevation, 6 geomag-lat, 11
            # year) so a rebuild that thins out a bin or two still passes, but a store
            # that produced almost no strata does not.
            str(
                EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_elevation.csv"
            ): 5,
            str(
                EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_geomag_lat.csv"
            ): 3,
            str(EPISTEMIC_SCALE_DIAGNOSTIC_DIR / "calibrating_scale_by_year.csv"): 5,
        },
        canonical_for="R1.2 epistemic-scale diagnostic",
        caveats=[
            "A diagnostic, not a retrain: it answers whether a single post-hoc scalar "
            "on the epistemic term can fix under-dispersed coverage, not whether the "
            "model should ship with one. On the real store the answer was scale, not "
            "structure - a calibrating scale exists that restores nominal 1-sigma "
            "coverage without degrading the Spearman ranking against |error| - but "
            "applying that scalar to the shipped model is a separate decision this "
            "stage does not make.",
            "sweep_paper_model.csv scores the paper's own BayesianResNetSTEC "
            "(Bayesian output layer only, --model-variant pretrained_stec); "
            "sweep_fully_bayesian_reference.csv scores a different architecture, "
            "ResNet_BNN_NLL (--reference-model-variant "
            "pretrained_stec_resnet_bnn_nll) - see CLAUDE.md's store-partition "
            "gotcha for why these two live in separate store partitions at all. Do "
            "not read one file's numbers onto the other model.",
            "Both partitions default to every year present (2014-2024, up to 544 days "
            "each), and main() now checks the two day sets match before scoring either "
            "one: this diagnostic's whole premise is comparing the two models' "
            "epistemic terms, so an unmatched population would silently score the "
            "paper model on one set of solar conditions and the reference model on "
            "another. A mismatch is not treated as fatal - both sweeps are restricted "
            "to the days the two partitions share, with a logged warning naming how "
            "many days were excluded from each side - because the two are independent "
            "backfills (CLAUDE.md's store-partition gotcha) that are not guaranteed to "
            "stay in lockstep as more days land on one side before the other.",
            "High coverage at large `s` is not evidence of a good uncertainty "
            "estimate on its own - Spearman rho must be read alongside coverage, "
            "since coverage alone can always be bought by inflating sigma; the "
            "reference model's own sweep shows Spearman falling as `s` grows even "
            "while coverage saturates near 100%.",
            "Real output already exists at this stage's output directory from a "
            "manual invocation (2026-08-24, commit 2b7172b) that predates this "
            "Stage's declaration - `pipeline status` will report 'never run' until "
            "it executes through the registry for real; that is the correct, "
            "un-faked answer, not a bug.",
        ],
    ),
    Stage(
        "mapping_function_consistency",
        f"-m stec.analysis.mapping_function_consistency --output-dir {MAPPING_FUNCTION_CONSISTENCY_DIR}",
        "R1.3",
        "cost of the mapping-function convention mismatch",
        inputs=[STORE_OWN],
        outputs=[str(MAPPING_FUNCTION_CONSISTENCY_DIR)],
    ),
    Stage(
        "madrigal_reference_offset",
        f"-m stec.analysis.madrigal_reference_offset --output-dir {MADRIGAL_REFERENCE_OFFSET_DIR}",
        "R1.3",
        "how much of the Madrigal error is a per-station reference offset",
        inputs=[STORE_MADRIGAL],
        # per_station_offsets.csv, not decomposition.csv, is the file with a
        # station-scale row count (~67 stations clear MIN_OBSERVATIONS_PER_STATION on
        # the real store). decomposition.csv is main()'s summary Series
        # (observations, stations, RMSE_vs_madrigal, RMSE_after_removing_station_offset,
        # variance_explained_by_offset_%, mean_abs_station_offset) turned into one
        # column - always exactly 6 rows when it was written at all, so its floor is
        # exact rather than a margin below a larger expected count.
        outputs=[
            str(MADRIGAL_REFERENCE_OFFSET_DIR),
            str(MADRIGAL_REFERENCE_OFFSET_DIR / "per_station_offsets.csv"),
            str(MADRIGAL_REFERENCE_OFFSET_DIR / "decomposition.csv"),
        ],
        min_rows={
            # Floored well below the ~67 real stations so a rebuild with a handful
            # fewer/more still passes, but a near-empty table (e.g. a truncated
            # Madrigal store) does not.
            str(MADRIGAL_REFERENCE_OFFSET_DIR / "per_station_offsets.csv"): 30,
            str(MADRIGAL_REFERENCE_OFFSET_DIR / "decomposition.csv"): 6,
        },
        canonical_for="Madrigal reference-offset decomposition",
        caveats=MADRIGAL_CAVEAT,
    ),
    Stage(
        "weighting_ablation",
        f"-m stec.analysis.weighting_ablation --output-dir {WEIGHTING_ABLATION_DIR}",
        "R2.5",
        "elevation against predicted-uncertainty weighting, paired station-days",
        inputs=[WEIGHTING_RUN],
        outputs=[str(WEIGHTING_ABLATION_DIR)],
    ),
    Stage(
        # Must precede storm_stratification, positioning_robustness,
        # common_set_positioning, positioning_summary and oracle_benchmark below - they
        # all read POSITIONING, which this stage now owns (see the constant's own
        # comment). Moved here, ahead of them, for exactly that reason; it used to sit
        # after storm_stratification/positioning_robustness with no ordering
        # consequence, back when POSITIONING pointed at a tree nothing in this file
        # produced.
        "positioning_coverage",
        f"-m stec.analysis.positioning_coverage --output-dir {POSITIONING_COVERAGE_DIR}",
        "R1.5",
        "which station-days each method solved, and why the rest are missing",
        inputs=["experiments"],
        outputs=[
            str(POSITIONING_COVERAGE_DIR),
            POSITIONING,
        ],
        # A header-only or drastically truncated multiday_summary.csv is exactly the
        # failure this catches - see the 2026-08-24 finding below: three individual
        # per-day source files on disk are themselves truncated this way, though not
        # enough to sink the whole aggregate below this floor. 30,000 sits comfortably
        # under the 37,209 rows the post-recovery iono run produced and well above
        # anything a truncated run could produce.
        min_rows={POSITIONING: 30_000},
        canonical_for="positioning station-day coverage",
        caveats=[
            "Canonical variant selection is explicit: it matches the canonical "
            "directory name directly rather than de-duplicating multiple "
            "Finetune_STEC_2024_<DOY>_* matches by sort order. Sort-order dedup let "
            "lr1e-4_bs2048/lr1e-4_bs10000 win over the paper's lr2e-4_bs512 for 31 DOYs "
            "once the station-recovery sweep created a second directory per day.",
            "Post station-recovery-sweep (2026-08-24), iono weighting: 8,195 / 1,591 / "
            "1,067 of 10,853 station-days solved by all methods / all ML methods "
            "missing (station absent from STEC DB) / some ML methods missing "
            "(per-method PPPx failure) - against the pre-sweep 8,003 / 2,311 / 510 of "
            "10,824 this caveat used to quote. The solved-by-all population grew by "
            "only 192 station-days (2.4%); most of the sweep's effect moved station-days "
            "from 'all ML missing' into 'some ML missing' rather than completing them, "
            "so the common-set population (common_set_positioning) is still "
            "substantially smaller than the full-recovered-set population "
            "(positioning_summary) - state which one backs a given number.",
            "Three individual per-day source files were found truncated on disk, all "
            "with recovery-sweep mtimes (2026-08-23/24), independent of this stage: "
            "DOY 166 and 176 dropped from ~43 stations to 2 in all three ML methods' "
            "own trees plus the GIM arm simultaneously (identical mtimes across the "
            "independent STEC/VTEC/Pretrained experiment directories, so this is a "
            "PPPx-level failure for those two days, not a positioning_coverage "
            "aggregation bug); DOY 323 dropped from ~41 to 4 stations in the STEC tree "
            "only (VTEC and Pretrained trees for that day are intact, 89 lines each). "
            "These three days are included in the aggregate as genuinely-small samples, "
            "not dropped or backfilled - no PPPx re-run was in scope to fix them. They "
            "are 3 of 242 days and do not materially move the per-station-day-mean "
            "tables, but a per-day breakdown that weights days unevenly should exclude "
            "or flag them.",
        ],
        supersedes=[
            str(paths.positioning_result_dir("full_coverage")),
            str(
                paths.positioning_result_dir("comparison_3way") / "multiday_summary.csv"
            ),
        ],
    ),
    Stage(
        "storm_stratification",
        f"-m stec.analysis.storm_stratification --output-dir {STORM_STRATIFICATION_DIR}",
        "R2.7",
        "positioning accuracy on storm against quiet days",
        inputs=[POSITIONING, SWI],
        outputs=[str(STORM_STRATIFICATION_DIR)],
        caveats=[
            "Classifies a day as storm when its daily minimum Dst reaches -50 nT. This "
            "is a different, deliberately kept-separate threshold from the STEC-domain "
            "scenario_evaluation.py, which classifies individual hours as storm at "
            "Kp>=37 or Dst<=-33: the daily rule finds 39 storm days over the 242-day "
            "positioning test period against 102 under the per-observation rule applied "
            "at the day level. Do not port one rule's day count into the other's table.",
        ],
    ),
    Stage(
        "positioning_robustness",
        f"-m stec.analysis.positioning_robustness --output-dir {POSITIONING_ROBUSTNESS_DIR}",
        "R2.7b",
        "tail behaviour and convergence of the positioning solutions",
        inputs=[POSITIONING],
        outputs=[str(POSITIONING_ROBUSTNESS_DIR)],
    ),
    Stage(
        "common_set_positioning",
        f"-m stec.analysis.common_set_positioning --output-dir {COMMON_SET_POSITIONING_DIR}",
        "R1.5",
        "positioning recomputed on the station-days every method solved",
        inputs=[POSITIONING, WEIGHTING_RUN],
        outputs=[str(COMMON_SET_POSITIONING_DIR)],
        # Not "Table A1": the manuscript has 5 tables and no lettered appendix (Figures
        # 14-15 are the only appendix content, continuously numbered, not a Table A1).
        # This stage's numbers back the R1.5 stochastic-model-ablation answer in
        # docs/revision/response_to_reviewers.md (elevation- vs uncertainty-weighting
        # comparison on the common station-day set), not any printed manuscript table -
        # so it claims no manuscript deliverable here.
        canonical_for=None,
        caveats=[
            "A different station-day population from Table 5, by design: requiring both "
            "weightings costs the IGS GIM ~3,000 station-days. State the N of each table."
        ],
    ),
    Stage(
        "positioning_summary",
        f"-m stec.analysis.positioning_summary --output-dir {POSITIONING_SUMMARY_DIR}",
        "Table 5",
        "headline positioning table, four methods on iono weighting",
        inputs=[POSITIONING],
        outputs=[
            str(POSITIONING_SUMMARY_DIR),
            str(POSITIONING_SUMMARY_DIR / "overall.csv"),
        ],
        # summarise_overall().reindex(METHOD_ORDER) always writes exactly 4 rows, one
        # per method, NaN-filled rather than dropped when a method has no station-days
        # - an exact count, not a floor. The `checks` entry below is what catches the
        # NaN-filled case min_rows cannot see.
        min_rows={str(POSITIONING_SUMMARY_DIR / "overall.csv"): 4},
        checks=[positioning_summary_overall_has_all_four_methods],
        canonical_for="Table 5",
    ),
    Stage(
        "oracle_benchmark",
        f"-m stec.analysis.oracle_benchmark --output-dir {ORACLE_BENCHMARK_DIR}",
        "R2.8",
        "positioning floor from reference STEC, on its own restricted set",
        # Not POSITIONING: oracle_benchmark.py never opens that file. It reads the
        # oracle experiment tree directly (ORACLE_EXPERIMENT_DIR, see that constant's
        # comment) and the frozen weighting run (WEIGHTING_RUN) for the baseline
        # methods it pairs against - the same WEIGHTING_RUN dependency
        # common_set_positioning declares above. Previously declared POSITIONING here
        # instead, which never changes when the oracle tree does, so `pipeline status`
        # reported this stage up to date while 166 of 242 oracle day-directories' SINEX
        # symlinks silently went dangling underneath it (destroyed by another
        # experiment's positioning cleanup rmtree-ing the products they pointed at -
        # see positioning/geometry/recover_day.py's run_models for the fix).
        inputs=[ORACLE_EXPERIMENT_DIR, WEIGHTING_RUN],
        outputs=[str(ORACLE_BENCHMARK_DIR)],
        caveats=[
            "NOT comparable with Table 5, by design and permanently. It uses elev "
            "weighting - the reference STEC carries only a placeholder sigma, so iono "
            "would weight by a constant - and is restricted to station-days solved by all "
            "four methods.",
            "Read ratios to the floor within this table. Take absolute positioning numbers "
            "from Table 5.",
        ],
    ),
    Stage(
        # Streams predictions/pretrained_stec/own (2014-2024, 10,000,000 rows across 544
        # sampled days - the pretrained model's entire held-out test set, not just 2024)
        # into a bounded, narrow-column parquet cache. Figures 4-9 need actual
        # per-observation values, not a sum, so this is the one manuscript-figure input
        # that cannot be a running accumulator like every other stec.analysis stage -
        # see the module's own docstring for the size accounting. Must precede
        # manuscript_figures below, which reads its cache.
        "pretrained_test_diagnostics",
        "-m stec.analysis.pretrained_test_diagnostics "
        f"--output-dir {PRETRAINED_TEST_DIAGNOSTICS_DIR}",
        "Figures 4-9",
        "per-observation cache of the pretrained model's whole 2014-2024 test set",
        inputs=[STORE_PRETRAINED],
        outputs=[
            str(PRETRAINED_TEST_DIAGNOSTICS_DIR),
            str(PRETRAINED_TEST_DIAGNOSTICS_DIR / "manifest.csv"),
        ],
        # Keyed on the manifest, not observations.parquet - a parquet output carries no
        # row count in the pipeline's provenance record, same reasoning as
        # inference_smoke/data_prep_smoke. 11 years, so 11 is exact, not a floor.
        min_rows={str(PRETRAINED_TEST_DIAGNOSTICS_DIR / "manifest.csv"): 11},
        caveats=[
            "Population: predictions/pretrained_stec/own in full - 2014-2024, 544 "
            "sampled days (~30/year for 2014-2023, all 242 of 2024), 10,000,000 "
            "observations - not the 2024-only period the four-method comparisons use. "
            "Deliberate: Figures 4-9 characterise the pretrained checkpoint's own "
            "residual/uncertainty behaviour standalone, matching src/inference_testset."
            "py's original scope (test_df was never filtered by year before "
            "plot_test_metrics), not a like-for-like comparison against Direct STEC/"
            "VTEC/IGS GIM that would need a population matched to theirs.",
        ],
    ),
    Stage(
        # The diagnostic-plot parity port: ~20 residual/spatial/uncertainty plots
        # src/viz/{spatial,performance,distributions,uncertainty}.py's plot_test_metrics
        # chain produced with no prior stec/ counterpart. One command builds its own
        # wider-column cache (stec.analysis.diagnostic_test_observations, a second pass
        # over predictions/pretrained_stec/own - see that module's docstring for why it
        # is not a shared cache with pretrained_test_diagnostics above) and then every
        # figure, because a Stage command is a single `python -m` invocation - there is
        # nowhere to chain a separate cache-building stage in front of it without a
        # second Stage, and this port was scoped to add exactly one.
        "diagnostic_figures",
        "-m stec.viz.diagnostic_figures "
        f"--cache-dir {DIAGNOSTIC_TEST_OBSERVATIONS_DIR} "
        f"--output-dir {DIAGNOSTIC_FIGURES_DIR}",
        "src/ diagnostic-plot parity",
        "spatial error maps, azimuth/elevation heatmaps, residual-vs-feature "
        "boxplots and uncertainty calibration diagnostics for the pretrained model's "
        "own held-out test set",
        inputs=[STORE_PRETRAINED],
        outputs=[
            str(DIAGNOSTIC_TEST_OBSERVATIONS_DIR),
            str(DIAGNOSTIC_TEST_OBSERVATIONS_DIR / "manifest.csv"),
            DIAGNOSTIC_FIGURES_DIR,
            f"{DIAGNOSTIC_FIGURES_DIR}/diagnostic_figures_manifest.csv",
        ],
        min_rows={
            # 11 years in predictions/pretrained_stec/own (2014-2024) - exact, not a
            # floor, same reasoning as pretrained_test_diagnostics above.
            str(DIAGNOSTIC_TEST_OBSERVATIONS_DIR / "manifest.csv"): 11,
            # Up to 23 figures at full coverage (see the module's FIGURE_BUILDERS);
            # floored well below that so a near-empty run is still caught without
            # pinning the exact count fig_spatial_error_map's data-dependent skip can
            # subtract (see the caveat below).
            f"{DIAGNOSTIC_FIGURES_DIR}/diagnostic_figures_manifest.csv": 15,
        },
        canonical_for="src/ diagnostic-plot parity (spatial/az-el/residual-feature/"
        "uncertainty diagnostics)",
        caveats=[
            "Same population as pretrained_test_diagnostics, and for the same reason: "
            "predictions/pretrained_stec/own's full 2014-2024, 544 days - these are "
            "standalone diagnostics of the pretrained model's own residual/uncertainty/"
            "spatial behaviour, not a cross-model comparison, so the full held-out "
            "range is the right scope rather than the 2024-only period the four-method "
            "comparisons use.",
            "Reads its own per-observation cache (stec.analysis."
            "diagnostic_test_observations, a second pass over "
            "predictions/pretrained_stec/own with a wider column set than "
            "pretrained_test_diagnostics's Figures-4-9 cache), not that cache directly - "
            "see this stage's own comment above for why.",
            "plot_solar_magnetic_ipp_error_map (src/viz/spatial.py) is not ported: the "
            "real predictions/pretrained_stec/own store has no sm_lon_ipp column, "
            "checked directly against its parquet schema.",
            "plot_binned_uncertainty_error_analysis (src/viz/uncertainty.py) is not "
            "ported: it duplicates manuscript Figure 9 (fig_uncertainty) for this same "
            "model and dataset.",
            "fig_spatial_error_map needs >=10 observations in a 5-degree lat/lon bin to "
            "plot at all (matching the source's own filter) - met at the full "
            "10,000,000-row store, but a partial or heavily filtered run can "
            "legitimately produce 0 of its 3 files rather than an empty plot.",
        ],
    ),
    Stage(
        # Not a port of stratified_comparison.py despite the shared day-at-a-time
        # accumulation pattern (see the module docstring): bin edges are the
        # publication's original 5-degree elevation bins (np.arange(0, 91, 5)), not
        # stratified_comparison.ELEVATION_BINS, and RMSE/MAE are left per-day rather
        # than pooled - manuscript_figures computes the across-day mean/std (the
        # figure's error bars) from this table itself, matching how
        # fig_positioning_trend derives its own mean/SEM from a raw per-station-day
        # frame. Must precede manuscript_figures below, which reads
        # per_day_by_elevation.csv.
        "elevation_metrics_finetuned",
        "-m stec.analysis.elevation_metrics_finetuned "
        f"--output-dir {ELEVATION_METRICS_FINETUNED_DIR}",
        "Figure 11",
        "per-day, per-elevation-bin RMSE/MAE for all four methods, own and madrigal",
        inputs=[STORE_OWN, STORE_MADRIGAL],
        outputs=[
            str(ELEVATION_METRICS_FINETUNED_DIR),
            str(ELEVATION_METRICS_FINETUNED_DIR / "per_day_by_elevation.csv"),
        ],
        # Keyed on the CSV, not the directory - same reasoning as
        # relative_error_metrics: a tree digest carries files/size/mtime but no row
        # count, so a min_rows on the parent directory can never be satisfied. Floored
        # well below a full run's plausible total - up to (242 own + 235 madrigal)
        # days x 18 five-degree elevation bins x 4 methods, thinned by the >100-
        # observation-per-(day,bin,method) guard - so this catches a near-empty or
        # single-day run (the old floor of 1 could not) without pinning the exact
        # count a real day's elevation distribution determines.
        min_rows={
            str(ELEVATION_METRICS_FINETUNED_DIR / "per_day_by_elevation.csv"): 2000
        },
        canonical_for="Figure 11 per-elevation error bars",
        caveats=[
            "A (day, elevation_bin, method) cell is dropped below 100 observations "
            "(the source's own guard), so the across-day mean/std never averages in a "
            "day where a bin was nearly empty.",
            "own and madrigal are both collected by default; manuscript_figures reads "
            "only the 'own' rows (Tables 3-4's scope) and leaves madrigal for a caller "
            "who wants that variant.",
        ],
    ),
    Stage(
        # Default day list added 2026-08-25 (DEFAULT_TEST_DOYS, the full 2024 test
        # period) - every run before this one only ever covered 18 of 242 days, because
        # --doys had no default and nothing forced a canonical choice. Full-period run
        # (672,542 arcs) barely moved the 18-day numbers (model dSTEC RMSE pooled 5.17 ->
        # 5.16 TECU, GIM 6.68 -> 6.64), so the 18-day estimate was already representative
        # - but that was luck, not something the 18-day invocation guaranteed. Placed
        # here, before figures/manuscript_figures like every other multiday_results
        # producer, not appended after them (see test_figures_and_manuscript_figures_
        # run_last).
        "dstec_evaluation",
        f"-m stec.analysis.dstec_evaluation --output-dir {DSTEC_EVALUATION_DIR}",
        "R1.3",
        "differential STEC (gradient-only) RMSE vs IGS GIM - cancels per-arc DCB/"
        "levelling offsets by construction, isolating the comparability concern from "
        "the model's own accuracy",
        inputs=[STORE_OWN],
        outputs=[
            str(DSTEC_EVALUATION_DIR),
            str(DSTEC_EVALUATION_DIR / "pass_statistics.csv"),
        ],
        # Keyed on the per-arc CSV, not the directory: a tree digest carries files/size/
        # mtime but no row count, so a min_rows on a directory can never be satisfied.
        # 500,000 is a floor, not the expected count (672,542 on the full 242-day store) -
        # comfortably above what an accidental partial run (e.g. the old 18-day default,
        # ~51,547) would produce, comfortably below normal day-to-day variation.
        min_rows={str(DSTEC_EVALUATION_DIR / "pass_statistics.csv"): 500_000},
        canonical_for="dSTEC (differential STEC) RMSE vs GIM, R1.3",
        caveats=[
            "Tests the TEC gradient along a pass, not the absolute level - a low dSTEC "
            "error is evidence the model gets the pass *shape* right, not evidence about "
            "the absolute calibration Tables 3/4 report. Read model_abs_rmse_pooled/ "
            "gim_abs_rmse_pooled alongside the dSTEC numbers, never as a substitute.",
            "Runs on finetuned_stec/own (--model-variant/--dataset default) - the "
            "scientifically sharper Madrigal comparison is parameterised and ready "
            "(see the module docstring) but blocked on the Madrigal local-time "
            "re-inference finishing first.",
        ],
    ),
    # Last: reads the metric CSVs every stage above writes, so it must follow all of them.
    Stage(
        "figures",
        "-m stec.viz.revision_figures",
        "all",
        "one PNG per revision figure, plus the _notitle manuscript variants",
        inputs=[str(paths.RESULTS_ROOT)],
        outputs=["plots/revision"],
        caveats=[
            "The _notitle and _no_legend variants are the manuscript figures; the titled "
            "copies are working copies carrying a provenance footnote.",
            "Approach colours are fixed: blue Direct STEC, orange VTEC + Mapping, green "
            "IGS GIM + Mapping, purple Pretrained. An approach colour must never mean "
            "anything else.",
        ],
    ),
    # Also last: reads daily_metrics's per_day.csv, elevation_metrics_finetuned's
    # per_day_by_elevation.csv, pretrained_test_diagnostics's observations.parquet and
    # positioning_coverage's multiday_summary.csv, so it must follow all of them, same as
    # `figures` above.
    Stage(
        "manuscript_figures",
        "-m stec.viz.manuscript_figures",
        "all",
        "manuscript-numbered figures (dataset split, error/uncertainty, positioning)",
        inputs=[str(paths.RESULTS_ROOT)],
        outputs=[
            "plots/manuscript",
            "plots/manuscript/dataset_construction/temp_split.csv",
        ],
        # Keyed on the CSV, not the directory - same reasoning as relative_error_metrics
        # above: a tree digest carries files/size/mtime but no row count. 132 is exact
        # (train+val+test months within 2014-2024); 100 leaves room without accepting a
        # near-empty run.
        min_rows={"plots/manuscript/dataset_construction/temp_split.csv": 100},
        caveats=[
            "All 14 code-generated manuscript figures (everything but the hand-drawn "
            "Figure 3) are wired into FIGURE_BUILDERS and run here. Figures 4-9 were the "
            "last gap - they read pretrained_test_diagnostics's cache rather than the "
            "store directly, so this stage's own output is only as current as that one's.",
            "Figure 11's error bars need elevation_metrics_finetuned's "
            "per_day_by_elevation.csv, which now has its own Stage (declared just "
            "above this one) and must run first - without it this stage logs a "
            "warning and skips Figure 11 rather than failing.",
            "Depends on stec.config.paths.SPLIT_LISTS (Figures 1-2) and on daily_metrics / "
            "pretrained_test_diagnostics / positioning_coverage's outputs (Figures 4-15) - "
            "a partial multiday_results tree produces a partial figure set with a logged "
            "warning per missing input, not a crash.",
        ],
    ),
    Stage(
        "results_manifest",
        "-m stec.analysis.results_manifest",
        "-",
        "which result trees are canonical and which are superseded",
        outputs=[
            str(RESULTS_MANIFEST_DIR),
            str(RESULTS_MANIFEST_DIR / "manifest.csv"),
        ],
        # One row per declared stage - floored well below today's stage count so
        # adding or removing a stage never needs this number revisited, while still
        # catching a manifest written against an empty or unvalidated registry.
        min_rows={str(RESULTS_MANIFEST_DIR / "manifest.csv"): 10},
        canonical_for="provenance index",
    ),
    Stage(
        "data_prep_smoke",
        f"-m stec.data.run_data_prep --config {SMOKE_CONFIG} --split test "
        f"--days 2024:132 --database-root {SMOKE_DATABASE_ROOT} "
        f"--space-weather {SMOKE_SWI} --output-dir artifacts/datasets/pipeline_smoke",
        "-",
        "proves stec.data.run_data_prep streams day_reader into feature_layout/"
        "transforms and writes a resumable, partitioned dataset - the S1 driver gap "
        "training_smoke and inference_smoke already closed for training and inference "
        "(docs/revision/task_board.md S1) - on the same tiny checked-in fixture day",
        inputs=[SMOKE_FIXTURE_DIR],
        outputs=[
            "artifacts/datasets/pipeline_smoke/test/year=2024/doy=132.parquet",
            "artifacts/datasets/pipeline_smoke/test/manifest.csv",
        ],
        # 1 row = the manifest's single processed day (--days 2024:132). Keyed on the
        # manifest, not the parquet: like inference_smoke's store file, a parquet output
        # carries no row count in the pipeline's provenance record.
        min_rows={"artifacts/datasets/pipeline_smoke/test/manifest.csv": 1},
        caveats=[
            "Runs against the same tiny fixture training_smoke/inference_smoke use "
            "(tests/fixtures/pipeline_smoke, 200 synthetic observations), pinned to one "
            "explicit day via --days rather than a real --split sweep, which would "
            "resolve every day in test_dates.list against the real database and take "
            "hours. A real invocation drops --days so the full split resolves against "
            "--database-root defaulting to stec.config.paths.STEC_DATABASE.",
            "Writes assembled, layout-specific tensor columns (feature_layout/"
            "transforms), not the legacy train.h5's raw, feature_control-agnostic ones - "
            "a deliberate simplification (module docstring) that ties this output to the "
            "config it was built from, unlike the legacy aggregate it stands in for.",
            "Assumes the raw per-day HDF5 already carries train_idx/val_idx/test_idx, "
            "written once, historically, by src/data_processing/add_split_indices.py, "
            "which this driver does not re-run: that would be a destructive in-place "
            "write against 740 GB of immutable external data.",
            "The pretrain-only 500,000-observation-per-epoch resample "
            "(data.train_subset_size, legacy EpochRandomSampler(replacement=True, ...)) "
            "is a training-time concern over this module's output, not part of building "
            "it - see the module docstring's 'Per-epoch pretrain sampling' section. No "
            "stec/ driver wires a multi-day pretrain loop over this output yet.",
        ],
    ),
]
