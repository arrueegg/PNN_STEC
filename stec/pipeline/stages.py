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

The commands still point at `src/`. The registry is the contract layer and can drive the
existing scripts while each is ported, which is what keeps this from being a big-bang
rewrite: a stage's command changes when its analysis moves, and nothing else does.
"""

from __future__ import annotations

from ..config import paths
from .stage import Stage

STORE_OWN = "predictions/finetuned_stec/own"
STORE_PRETRAINED = "predictions/pretrained_stec/own"
STORE_MADRIGAL = "predictions/finetuned_stec/madrigal"
POSITIONING = str(
    paths.positioning_result_dir("full_coverage") / "multiday_summary.csv"
)
WEIGHTING_RUN = str(
    paths.positioning_result_dir("20260216_2052") / "multiday_summary.csv"
)
SWI = "data/omni_hourly_2010-2025.h5"

# Every stec.analysis output directory, named once so a stage's command string,
# `outputs`, `inputs` and `supersedes` can never disagree about where it writes.
# `paths.analysis_result_dir` is the one place the `analyses/<name>/{rebuilt,
# pre_rebuild}` layout is spelled out (docs/revision/results_layout.md) - nothing below
# builds a `multiday_results/...` string by hand.
PAPER_TABLES_DIR = paths.analysis_result_dir("paper_tables", rebuilt=True)
RELATIVE_ERROR_METRICS_DIR = paths.analysis_result_dir(
    "relative_error_metrics", rebuilt=True
)
HYPERPARAMETER_SEARCH_DIR = paths.analysis_result_dir(
    "hyperparameter_search", rebuilt=False
)
STATION_INDEPENDENCE_DIR = paths.analysis_result_dir(
    "station_independence", rebuilt=True
)
COMPUTATIONAL_COST_DIR = paths.analysis_result_dir("computational_cost", rebuilt=True)
# "repair_gim_baseline" is the stage name; "gim_baseline_repair" is the directory name
# the frozen script has always written - the one irregular case `paths.py`'s docstring
# and `stec.runs.restructure_results` both call out by name.
GIM_BASELINE_REPAIR_DIR = paths.analysis_result_dir(
    "repair_gim_baseline", rebuilt=False
)
DAILY_METRICS_DIR = paths.analysis_result_dir("daily_metrics", rebuilt=True)
UNCERTAINTY_ERROR_RELATION_DIR = paths.analysis_result_dir(
    "uncertainty_error_relation", rebuilt=True
)
STRATIFIED_COMPARISON_DIR = paths.analysis_result_dir(
    "stratified_comparison", rebuilt=True
)
ACTIVITY_STRATIFICATION_DIR = paths.analysis_result_dir(
    "activity_stratification", rebuilt=True
)
IONEX_RMS_BENCHMARK_DIR = paths.analysis_result_dir("ionex_rms_benchmark", rebuilt=True)
UNCERTAINTY_CALIBRATION_DIR = paths.analysis_result_dir(
    "uncertainty_calibration", rebuilt=True
)
MAPPING_FUNCTION_CONSISTENCY_DIR = paths.analysis_result_dir(
    "mapping_function_consistency", rebuilt=True
)
MADRIGAL_REFERENCE_OFFSET_DIR = paths.analysis_result_dir(
    "madrigal_reference_offset", rebuilt=True
)
WEIGHTING_ABLATION_DIR = paths.analysis_result_dir("weighting_ablation", rebuilt=True)
STORM_STRATIFICATION_DIR = paths.analysis_result_dir(
    "storm_stratification", rebuilt=True
)
POSITIONING_ROBUSTNESS_DIR = paths.analysis_result_dir(
    "positioning_robustness", rebuilt=True
)
POSITIONING_COVERAGE_DIR = paths.analysis_result_dir(
    "positioning_coverage", rebuilt=True
)
COMMON_SET_POSITIONING_DIR = paths.analysis_result_dir(
    "common_set_positioning", rebuilt=True
)
POSITIONING_SUMMARY_DIR = paths.analysis_result_dir("positioning_summary", rebuilt=True)
ORACLE_BENCHMARK_DIR = paths.analysis_result_dir("oracle_benchmark", rebuilt=True)
RESULTS_MANIFEST_DIR = paths.analysis_result_dir("results_manifest", rebuilt=True)

# The canonical STEC-metrics sweep (CLAUDE.md's "Which results are canonical" table) is a
# full evaluation tree, not a `stec.analysis` output, so it lives under
# `stec_evaluation/`, not `analyses/` - see docs/revision/results_layout.md.
WITH_PRETRAINED_BASELINE_SUMMARY = (
    paths.STEC_EVALUATION_RESULTS / "with_pretrained_baseline" / "summary"
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
        outputs=[
            "artifacts/predictions/finetuned_stec/own/year=2024/doy=132.parquet",
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
            "Only the 'own' dataset is wired up. '--dataset madrigal' raises "
            "NotImplementedError: Madrigal geometry has no stec/ model-input reader - "
            "stec.baselines.madrigal loads Madrigal's reference STEC for comparison, "
            "never model inputs. predictions/pretrained_stec/madrigal/ has no data for "
            "exactly this reason (docs/revision/task_board.md S4).",
            "Writes under artifacts/predictions/, the default "
            "prediction_store.DEFAULT_STORE_ROOT - a different tree from the legacy "
            "predictions/ every analysis stage above reads (STORE_OWN etc.), so this "
            "stage's output is never picked up by daily_metrics or any other analysis.",
            "Runs the zero-perturbation control (stec/models/determinism.py) before any "
            "real sampling and fails loudly if it is not exactly 0.0 - the Bayesian A/B "
            "invariant CLAUDE.md requires for every comparison built on this model.",
        ],
    ),
    Stage(
        "paper_tables",
        f"-m stec.analysis.paper_tables --config {paths.PAPER_PRETRAINED_CONFIG} "
        f"--output-dir {PAPER_TABLES_DIR}",
        "Tables 1, 2",
        "input feature list and hyperparameters, generated from the model rather than "
        "maintained beside it",
        outputs=[str(PAPER_TABLES_DIR)],
        canonical_for="Tables 1 and 2",
        caveats=[
            "Generated from the paper's own stored run config, not from a template in "
            "config/. The template disagreed with it on 7 of 8 fields - architecture, "
            "prior sigma, learning rate, batch size, scheduler, SH degree and KL weight - "
            "so the table it produced described a different model entirely.",
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
        "hyperparameter_search",
        # --wandb_dir passed explicitly: the frozen script's default (`Path("wandb")`)
        # resolves against cwd, and the runner's cwd is REPO_ROOT, which has no wandb/ in
        # this worktree - the run history lives only in the primary checkout.
        # paths.LEGACY_WANDB honours STEC_LEGACY_ROOT, same as repair_gim_baseline's
        # --store_root above. --output_dir is the script's own existing flag (not a new
        # one added for this), so redirecting it into the new layout needs no edit to the
        # frozen script.
        f"src/analysis/hyperparameter_search_summary.py "
        f"--wandb_dir {paths.LEGACY_WANDB} --output_dir {HYPERPARAMETER_SEARCH_DIR}",
        "R2.5, R2.8b",
        "architecture comparison and hyperparameter sweep from the W&B history",
        inputs=["wandb"],
        outputs=[str(HYPERPARAMETER_SEARCH_DIR)],
        caveats=[
            "Stays on the pre-rebuild script for a data reason, not a code one: the "
            "script itself is self-contained (only glob/yaml/json/pandas over local "
            "wandb/run-*/files/{config.yaml,wandb-summary.json} pairs, no W&B API call "
            "and no network access) and has no dependency on the rest of src/, so "
            "porting it would be mechanical. It has not been ported because its input "
            "is not reachable here: wandb/ is untracked (.gitignore'd, ~606 MB, ~1,526 "
            "run directories in the live checkout) and does not exist in this worktree "
            "or in any fresh clone.",
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
        f"src/analysis/repair_gim_baseline.py --apply "
        f"--store_root {paths.LEGACY_PREDICTIONS} --output_dir {GIM_BASELINE_REPAIR_DIR}",
        "Table 4, R1.4",
        "recompute the IGS GIM baseline against the correct day's IONEX map",
        inputs=[STORE_OWN],
        outputs=[str(GIM_BASELINE_REPAIR_DIR)],
        caveats=[
            "Repairs the 12 days of 2024 where a truncating cast on a float32-denormalised "
            "doy loaded the previous day's IONEX map. Unaffected days must reproduce to "
            "~1e-5 TECU; that agreement is the regression check.",
            "Must stay on the pre-rebuild script, permanently: this stage is the "
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
        inputs=[STORE_OWN, STORE_PRETRAINED, str(GIM_BASELINE_REPAIR_DIR)],
        outputs=[str(DAILY_METRICS_DIR)],
        min_rows={},
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
        inputs=[STORE_OWN, STORE_PRETRAINED],
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
        outputs=[str(MADRIGAL_REFERENCE_OFFSET_DIR)],
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
        "positioning_coverage",
        f"-m stec.analysis.positioning_coverage --output-dir {POSITIONING_COVERAGE_DIR}",
        "R1.5",
        "which station-days each method solved, and why the rest are missing",
        inputs=["experiments"],
        # POSITIONING (positioning_result_dir("full_coverage")) is the pre-rebuild tree
        # this stage reads alongside, not something it writes - declaring it here would
        # claim ownership of output this stage never produces.
        outputs=[str(POSITIONING_COVERAGE_DIR)],
        canonical_for="positioning station-day coverage",
        caveats=[
            "Canonical variant selection is explicit: it matches the canonical "
            "directory name directly rather than de-duplicating multiple "
            "Finetune_STEC_2024_<DOY>_* matches by sort order. Sort-order dedup let "
            "lr1e-4_bs2048/lr1e-4_bs10000 win over the paper's lr2e-4_bs512 for 31 DOYs "
            "once the station-recovery sweep created a second directory per day.",
            "Quoting R1.5 needs the pre-sweep 8,003 / 2,311 / 510 of 10,824 "
            "(database-only, iono weighting) until the station-recovery sweep has run "
            "all 242 days with both overwrite sites fixed (metrics.py, patched but not "
            "applied, and run_positioning_evaluation.py:681, unpatched) - a run against "
            "the partially-swept tree corresponds to no coherent configuration.",
        ],
    ),
    Stage(
        "common_set_positioning",
        f"-m stec.analysis.common_set_positioning --output-dir {COMMON_SET_POSITIONING_DIR}",
        "R1.5, Table A1",
        "positioning recomputed on the station-days every method solved",
        inputs=[POSITIONING, WEIGHTING_RUN],
        outputs=[str(COMMON_SET_POSITIONING_DIR)],
        canonical_for="Table A1",
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
        outputs=[str(POSITIONING_SUMMARY_DIR)],
        canonical_for="Table 5",
    ),
    Stage(
        "oracle_benchmark",
        f"-m stec.analysis.oracle_benchmark --output-dir {ORACLE_BENCHMARK_DIR}",
        "R2.8",
        "positioning floor from reference STEC, on its own restricted set",
        inputs=[POSITIONING],
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
    Stage(
        "results_manifest",
        "-m stec.analysis.results_manifest",
        "-",
        "which result trees are canonical and which are superseded",
        outputs=[str(RESULTS_MANIFEST_DIR)],
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
