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
POSITIONING = "multiday_results/positioning_full_coverage/multiday_summary.csv"
WEIGHTING_RUN = "multiday_results/positioning_20260216_2052/multiday_summary.csv"
SWI = "data/omni_hourly_2010-2025.h5"

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
        "--output-dir multiday_results/paper_tables",
        "Tables 1, 2",
        "input feature list and hyperparameters, generated from the model rather than "
        "maintained beside it",
        outputs=["multiday_results/paper_tables"],
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
        "-m stec.analysis.relative_error_metrics --output-dir multiday_results/relative_error_metrics_rebuilt",
        "R2.1, R2.2",
        "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
        outputs=[
            "multiday_results/relative_error_metrics_rebuilt",
            "multiday_results/relative_error_metrics_rebuilt/yearly_metrics.csv",
        ],
        # Keyed on the CSV, not the directory: a tree digest carries files/size/mtime but
        # no row count, so a min_rows on a directory can never be satisfied and the stage
        # fails however well it ran.
        min_rows={
            "multiday_results/relative_error_metrics_rebuilt/yearly_metrics.csv": 5
        },
    ),
    Stage(
        "hyperparameter_search",
        "src/analysis/hyperparameter_search_summary.py",
        "R2.5, R2.8b",
        "architecture comparison and hyperparameter sweep from the W&B history",
        inputs=["wandb"],
        outputs=["multiday_results/hyperparameter_search"],
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
        "-m stec.analysis.station_independence --output-dir multiday_results/station_independence_rebuilt",
        "R2.3",
        "test-station error against distance to the nearest training station",
        inputs=[STORE_OWN],
        outputs=["multiday_results/station_independence_rebuilt"],
        caveats=[
            "Limited by n = 55 test stations, not by observation count. Adding days "
            "sharpens each point but does not sharpen the Spearman coefficient.",
            "Strengthening this result needs a region-held-out retrain, not more data.",
        ],
    ),
    Stage(
        "computational_cost",
        "-m stec.analysis.computational_cost --output-dir multiday_results/computational_cost_rebuilt",
        "R2.8h",
        "training and inference cost",
        outputs=["multiday_results/computational_cost_rebuilt"],
    ),
    Stage(
        "repair_gim_baseline",
        "src/analysis/repair_gim_baseline.py --apply",
        "Table 4, R1.4",
        "recompute the IGS GIM baseline against the correct day's IONEX map",
        inputs=[STORE_OWN],
        outputs=["multiday_results/gim_baseline_repair"],
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
        "-m stec.analysis.daily_metrics --output-dir multiday_results/daily_metrics_rebuilt",
        "Tables 3, 4",
        "per-day and pooled STEC metrics recomputed from the prediction store",
        inputs=[STORE_OWN, STORE_PRETRAINED, "multiday_results/gim_baseline_repair"],
        outputs=["multiday_results/daily_metrics_rebuilt"],
        min_rows={},
        canonical_for="Tables 3 and 4",
        caveats=[
            "The published RMSE is RMSE_mean - the mean of per-day RMSEs - which is what "
            "the manuscript states. pooled_RMSE in the same file is over observations and "
            "is consistently higher; the two are not interchangeable.",
            "Madrigal rows carry the madrigal_reference_offset caveat.",
        ],
        supersedes=[
            "multiday_results/with_pretrained_baseline/summary/summary_statistics.csv"
        ],
    ),
    Stage(
        "uncertainty_error_relation",
        "-m stec.analysis.uncertainty_error_relation --output-dir multiday_results/uncertainty_error_relation_rebuilt",
        "R2.6, R1.2",
        "predicted uncertainty against realised error, pooled over the test period",
        inputs=[STORE_OWN],
        outputs=["multiday_results/uncertainty_error_relation_rebuilt"],
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
        "-m stec.analysis.stratified_comparison --output-dir multiday_results/stratified_comparison_rebuilt",
        "R1.4",
        "all four methods by elevation, geomagnetic latitude, local time and season",
        inputs=[STORE_OWN, STORE_PRETRAINED],
        outputs=["multiday_results/stratified_comparison_rebuilt"],
    ),
    Stage(
        "activity_stratification",
        "-m stec.analysis.activity_stratification --output-dir multiday_results/activity_stratification_rebuilt",
        "R1.4",
        "STEC error stratified by Dst and F10.7",
        inputs=["multiday_results/daily_metrics_rebuilt", SWI],
        outputs=["multiday_results/activity_stratification_rebuilt"],
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
        "-m stec.analysis.ionex_rms_benchmark --output_dir multiday_results/ionex_rms_benchmark_rebuilt",
        "R1.6b",
        "model uncertainty against the IGS and CODE GIM RMS maps",
        inputs=[STORE_OWN],
        outputs=["multiday_results/ionex_rms_benchmark_rebuilt"],
    ),
    Stage(
        # Ported. Scores every model under both Gaussian and Laplace, tagging which is
        # native, so the mis-specified number sits beside the correct one. Also
        # restratifies every (model, family) accumulation by geomagnetic regime -
        # "all" plus the daily-Dst "quiet"/"storm" split from the pre-rebuild source,
        # answering R1.6's "uncertainty behaviour under ... disturbed conditions".
        "uncertainty_calibration",
        "-m stec.analysis.uncertainty_calibration --output-dir multiday_results/uncertainty_calibration_rebuilt",
        "R1.6, R2.6",
        "coverage, PIT and CRPS for every model, each under its own likelihood and regime",
        inputs=[STORE_OWN, STORE_PRETRAINED, SWI],
        outputs=["multiday_results/uncertainty_calibration_rebuilt"],
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
        "mapping_function_consistency",
        "-m stec.analysis.mapping_function_consistency --output-dir multiday_results/mapping_function_consistency_rebuilt",
        "R1.3",
        "cost of the mapping-function convention mismatch",
        inputs=[STORE_OWN],
        outputs=["multiday_results/mapping_function_consistency_rebuilt"],
    ),
    Stage(
        "madrigal_reference_offset",
        "-m stec.analysis.madrigal_reference_offset --output-dir multiday_results/madrigal_reference_offset_rebuilt",
        "R1.3",
        "how much of the Madrigal error is a per-station reference offset",
        inputs=[STORE_MADRIGAL],
        outputs=["multiday_results/madrigal_reference_offset_rebuilt"],
        canonical_for="Madrigal reference-offset decomposition",
        caveats=MADRIGAL_CAVEAT,
    ),
    Stage(
        "weighting_ablation",
        "-m stec.analysis.weighting_ablation --output-dir multiday_results/weighting_ablation_rebuilt",
        "R2.5",
        "elevation against predicted-uncertainty weighting, paired station-days",
        inputs=[WEIGHTING_RUN],
        outputs=["multiday_results/weighting_ablation_rebuilt"],
    ),
    Stage(
        "storm_stratification",
        "-m stec.analysis.storm_stratification --output-dir multiday_results/storm_stratification_rebuilt",
        "R2.7",
        "positioning accuracy on storm against quiet days",
        inputs=[POSITIONING, SWI],
        outputs=["multiday_results/storm_stratification_rebuilt"],
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
        "-m stec.analysis.positioning_robustness --output-dir multiday_results/positioning_robustness_rebuilt",
        "R2.7b",
        "tail behaviour and convergence of the positioning solutions",
        inputs=[POSITIONING],
        outputs=["multiday_results/positioning_robustness_rebuilt"],
    ),
    Stage(
        "positioning_coverage",
        "-m stec.analysis.positioning_coverage --output-dir multiday_results/positioning_coverage_rebuilt",
        "R1.5",
        "which station-days each method solved, and why the rest are missing",
        inputs=["experiments"],
        # The command writes positioning_coverage_rebuilt; positioning_full_coverage is
        # the pre-rebuild tree this stage reads alongside, and declaring it here claimed
        # ownership of output this stage never produces.
        outputs=["multiday_results/positioning_coverage_rebuilt"],
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
        "-m stec.analysis.common_set_positioning --output-dir multiday_results/common_set_positioning_rebuilt",
        "R1.5, Table A1",
        "positioning recomputed on the station-days every method solved",
        inputs=[POSITIONING, WEIGHTING_RUN],
        outputs=["multiday_results/common_set_positioning_rebuilt"],
        canonical_for="Table A1",
        caveats=[
            "A different station-day population from Table 5, by design: requiring both "
            "weightings costs the IGS GIM ~3,000 station-days. State the N of each table."
        ],
    ),
    Stage(
        "positioning_summary",
        "-m stec.analysis.positioning_summary --output-dir multiday_results/positioning_summary_rebuilt",
        "Table 5",
        "headline positioning table, four methods on iono weighting",
        inputs=[POSITIONING],
        outputs=["multiday_results/positioning_summary_rebuilt"],
        canonical_for="Table 5",
    ),
    Stage(
        "oracle_benchmark",
        "-m stec.analysis.oracle_benchmark --output-dir multiday_results/oracle_benchmark_rebuilt",
        "R2.8",
        "positioning floor from reference STEC, on its own restricted set",
        inputs=[POSITIONING],
        outputs=["multiday_results/oracle_benchmark_rebuilt"],
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
        inputs=["multiday_results"],
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
        outputs=["multiday_results/results_manifest"],
        canonical_for="provenance index",
    ),
]
