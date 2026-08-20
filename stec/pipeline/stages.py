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

from .stage import Stage

STORE_OWN = "predictions/finetuned_stec/own"
STORE_PRETRAINED = "predictions/pretrained_stec/own"
STORE_MADRIGAL = "predictions/finetuned_stec/madrigal"
POSITIONING = "multiday_results/positioning_full_coverage/multiday_summary.csv"
WEIGHTING_RUN = "multiday_results/positioning_20260216_2052/multiday_summary.csv"
SWI = "data/omni_hourly_2010-2025.h5"

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
        "paper_tables",
        "-m stec.analysis.paper_tables --config config/config_BNN.yaml",
        "Tables 1, 2",
        "input feature list and hyperparameters, generated from the model rather than "
        "maintained beside it",
        outputs=["multiday_results/paper_tables"],
        canonical_for="Tables 1 and 2",
        caveats=[
            "Generated from a resolved run config. Point it at a stored experiment's "
            "config.yaml to describe what actually trained, not at a template.",
            "Table 2 includes three hyperparameters the submitted manuscript omits: the "
            "KL warmup (0 to 0.1 over 5 epochs), the variance floor, and the output bias "
            "initialisation.",
        ],
    ),
    Stage(
        "relative_error_metrics",
        "src/analysis/relative_error_metrics.py",
        "R2.1, R2.2",
        "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
        outputs=["multiday_results/relative_error_metrics.csv"],
        min_rows={"multiday_results/relative_error_metrics.csv": 5},
    ),
    Stage(
        "hyperparameter_search",
        "src/analysis/hyperparameter_search_summary.py",
        "R2.5, R2.8b",
        "architecture comparison and hyperparameter sweep from the W&B history",
        inputs=["wandb"],
        outputs=["multiday_results/hyperparameter_search"],
    ),
    Stage(
        "station_independence",
        "src/analysis/station_independence.py",
        "R2.3",
        "test-station error against distance to the nearest training station",
        inputs=[STORE_OWN],
        outputs=["multiday_results/station_independence"],
        caveats=[
            "Limited by n = 55 test stations, not by observation count. Adding days "
            "sharpens each point but does not sharpen the Spearman coefficient.",
            "Strengthening this result needs a region-held-out retrain, not more data.",
        ],
    ),
    Stage(
        "computational_cost",
        "src/analysis/computational_cost.py",
        "R2.8h",
        "training and inference cost",
        outputs=["multiday_results/computational_cost"],
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
            "~1e-5 TECU; that agreement is the regression check."
        ],
    ),
    Stage(
        # Ported. Verified to reproduce the pre-rebuild implementation exactly - delta 0.0
        # on RMSE_mean, pooled_RMSE, MAE_mean, R2_mean, day and observation counts, across
        # all seven model/dataset combinations over 242 days and 475,111,413 observations.
        "daily_metrics",
        "-m stec.analysis.daily_metrics",
        "Tables 3, 4",
        "per-day and pooled STEC metrics recomputed from the prediction store",
        inputs=[STORE_OWN, STORE_PRETRAINED, "multiday_results/gim_baseline_repair"],
        outputs=["multiday_results/daily_metrics"],
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
        "src/analysis/uncertainty_error_relation.py",
        "R2.6, R1.2",
        "predicted uncertainty against realised error, pooled over the test period",
        inputs=[STORE_OWN],
        outputs=["multiday_results/uncertainty_error_relation"],
    ),
    Stage(
        "stratified_comparison",
        "src/analysis/stratified_comparison.py",
        "R1.4",
        "all four methods by elevation, geomagnetic latitude, local time and season",
        inputs=[STORE_OWN, STORE_PRETRAINED],
        outputs=["multiday_results/stratified_comparison"],
    ),
    Stage(
        "activity_stratification",
        "src/analysis/activity_stratification.py",
        "R1.4",
        "STEC error stratified by Dst and F10.7",
        inputs=["multiday_results/daily_metrics", SWI],
        outputs=["multiday_results/activity_stratification"],
        caveats=[
            "Reads the repaired GIM values. Run after repair_gim_baseline: the "
            "un-repaired baseline reversed this comparison's conclusion."
        ],
    ),
    Stage(
        "ionex_rms_benchmark",
        "src/analysis/ionex_rms_benchmark.py",
        "R1.6b",
        "model uncertainty against the IGS and CODE GIM RMS maps",
        inputs=[STORE_OWN],
        outputs=["multiday_results/ionex_rms_benchmark"],
    ),
    Stage(
        # Ported. Scores every model under both Gaussian and Laplace, tagging which is
        # native, so the mis-specified number sits beside the correct one.
        "uncertainty_calibration",
        "-m stec.analysis.uncertainty_calibration",
        "R1.6, R2.6",
        "coverage, PIT and CRPS for every model, each under its own likelihood",
        inputs=[STORE_OWN, STORE_PRETRAINED],
        outputs=["multiday_results/uncertainty_calibration"],
        caveats=[
            "The VTEC baseline is a Laplace, and its stored vtec_model_stec_total_unc is "
            "already a standard deviation (sqrt(2) * scale), not the raw scale - recover "
            "the scale as std / sqrt(2) before any Laplace formula sees it.",
            "Scored as a Gaussian the same data reads 90% coverage at nominal 50% against "
            "82% under Laplace; both scorings are emitted side by side, tagged by which "
            "likelihood is native to each model.",
        ],
    ),
    Stage(
        "mapping_function_consistency",
        "src/analysis/mapping_function_consistency.py",
        "R1.3",
        "cost of the mapping-function convention mismatch",
        inputs=[STORE_OWN],
        outputs=["multiday_results/mapping_function_consistency"],
    ),
    Stage(
        "madrigal_reference_offset",
        "src/analysis/madrigal_reference_offset.py",
        "R1.3",
        "how much of the Madrigal error is a per-station reference offset",
        inputs=[STORE_MADRIGAL],
        outputs=["multiday_results/madrigal_reference_offset"],
        canonical_for="Madrigal reference-offset decomposition",
        caveats=MADRIGAL_CAVEAT,
    ),
    Stage(
        "weighting_ablation",
        "src/analysis/weighting_ablation.py",
        "R2.5",
        "elevation against predicted-uncertainty weighting, paired station-days",
        inputs=[WEIGHTING_RUN],
        outputs=["multiday_results/weighting_ablation"],
    ),
    Stage(
        "storm_stratification",
        "src/analysis/storm_stratification.py",
        "R2.7",
        "positioning accuracy on storm against quiet days",
        inputs=[POSITIONING, SWI],
        outputs=["multiday_results/storm_stratification"],
    ),
    Stage(
        "positioning_robustness",
        "src/analysis/positioning_robustness.py",
        "R2.7b",
        "tail behaviour and convergence of the positioning solutions",
        inputs=[POSITIONING],
        outputs=["multiday_results/positioning_robustness"],
    ),
    Stage(
        "positioning_coverage",
        "src/analysis/positioning_coverage.py",
        "R1.5",
        "which station-days each method solved, and why the rest are missing",
        inputs=["experiments"],
        outputs=["multiday_results/positioning_full_coverage"],
        canonical_for="positioning station-day coverage",
    ),
    Stage(
        "common_set_positioning",
        "src/analysis/common_set_positioning.py",
        "R1.5, Table A1",
        "positioning recomputed on the station-days every method solved",
        inputs=[POSITIONING, WEIGHTING_RUN],
        outputs=["multiday_results/common_set_positioning"],
        canonical_for="Table A1",
        caveats=[
            "A different station-day population from Table 5, by design: requiring both "
            "weightings costs the IGS GIM ~3,000 station-days. State the N of each table."
        ],
    ),
    Stage(
        "positioning_summary",
        "src/analysis/positioning_summary.py",
        "Table 5",
        "headline positioning table, four methods on iono weighting",
        inputs=[POSITIONING],
        outputs=["multiday_results/positioning_summary"],
        canonical_for="Table 5",
    ),
    Stage(
        "oracle_benchmark",
        "src/analysis/oracle_benchmark.py",
        "R2.8",
        "positioning floor from reference STEC, on its own restricted set",
        inputs=[POSITIONING],
        outputs=["multiday_results/oracle_benchmark"],
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
        "src/viz/revision_figures.py",
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
