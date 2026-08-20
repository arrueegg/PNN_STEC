"""Every analysis that produces a number in the paper, declared once.

Each stage names the command that runs it, what it reads, what it writes, the reviewer
comment it answers, and the minimum it must produce to be believed. Declaring inputs is
what makes a rerun skippable; declaring outputs is what makes duplication impossible,
because a second stage claiming the same output is a startup error rather than a silent
race; and declaring assertions is what turns a plausible-but-empty result into a failure.

Order is significant and is the order below - `repair_gim_baseline` must precede
`daily_metrics`, which must precede `activity_stratification`.

Inputs are declared at the granularity that actually changes. The prediction store is
named as a directory rather than 242 parquet files, and the raw HDF5 days are not declared
at all: they are immutable external data, and treating them as an input would mean walking
740 GB to decide whether to run a two-second CSV summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STORE_OWN = "predictions/finetuned_stec/own"
STORE_PRETRAINED = "predictions/pretrained_stec/own"
STORE_MADRIGAL = "predictions/finetuned_stec/madrigal"
POSITIONING = "multiday_results/positioning_full_coverage/multiday_summary.csv"
WEIGHTING_RUN = "multiday_results/positioning_20260216_2052/multiday_summary.csv"
SWI = "data/omni_hourly_2010-2025.h5"


@dataclass(frozen=True)
class Stage:
    name: str
    command: str
    answers: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)
    # Minimum rows an output CSV must carry to count as produced. A stage that writes a
    # header and nothing else is the failure mode this exists to catch.
    min_rows: dict[str, int] = field(default_factory=dict)


STAGES: list[Stage] = [
    Stage("relative_error_metrics", "src/analysis/relative_error_metrics.py", "R2.1, R2.2",
          "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
          outputs=["multiday_results/relative_error_metrics.csv"],
          min_rows={"multiday_results/relative_error_metrics.csv": 5}),

    Stage("hyperparameter_search", "src/analysis/hyperparameter_search_summary.py", "R2.5, R2.8b",
          "architecture comparison and hyperparameter sweep from the W&B history",
          inputs=["wandb"], outputs=["multiday_results/hyperparameter_search"]),

    Stage("station_independence", "src/analysis/station_independence.py", "R2.3",
          "test-station error against distance to the nearest training station",
          inputs=[STORE_OWN], outputs=["multiday_results/station_independence"]),

    Stage("computational_cost", "src/analysis/computational_cost.py", "R2.8h",
          "training and inference cost",
          outputs=["multiday_results/computational_cost"]),

    Stage("repair_gim_baseline", "src/analysis/repair_gim_baseline.py --apply", "Table 4, R1.4",
          "recompute the IGS GIM baseline against the correct day's IONEX map",
          inputs=[STORE_OWN], outputs=["multiday_results/gim_baseline_repair"]),

    Stage("daily_metrics", "src/analysis/daily_metrics.py", "Tables 3, 4",
          "per-day and pooled STEC metrics recomputed from the prediction store",
          inputs=[STORE_OWN, STORE_PRETRAINED, "multiday_results/gim_baseline_repair"],
          outputs=["multiday_results/daily_metrics"],
          min_rows={"multiday_results/daily_metrics/per_day.csv": 200}),

    Stage("uncertainty_error_relation", "src/analysis/uncertainty_error_relation.py", "R2.6, R1.2",
          "predicted uncertainty against realised error, pooled over the test period",
          inputs=[STORE_OWN], outputs=["multiday_results/uncertainty_error_relation"]),

    Stage("stratified_comparison", "src/analysis/stratified_comparison.py", "R1.4",
          "all four methods by elevation, geomagnetic latitude, local time and season",
          inputs=[STORE_OWN, STORE_PRETRAINED],
          outputs=["multiday_results/stratified_comparison"]),

    Stage("activity_stratification", "src/analysis/activity_stratification.py", "R1.4",
          "STEC error stratified by Dst and F10.7",
          inputs=["multiday_results/daily_metrics/per_day.csv", SWI],
          outputs=["multiday_results/activity_stratification"]),

    Stage("ionex_rms_benchmark", "src/analysis/ionex_rms_benchmark.py", "R1.6b",
          "model uncertainty against the IGS and CODE GIM RMS maps",
          inputs=[STORE_OWN], outputs=["multiday_results/ionex_rms_benchmark"]),

    Stage("uncertainty_calibration", "src/analysis/uncertainty_calibration.py", "R1.6, R2.6",
          "coverage, PIT and CRPS for every model",
          inputs=[STORE_OWN, STORE_PRETRAINED],
          outputs=["multiday_results/uncertainty_calibration"]),

    Stage("mapping_function_consistency", "src/analysis/mapping_function_consistency.py", "R1.3",
          "cost of the mapping-function convention mismatch",
          inputs=[STORE_OWN], outputs=["multiday_results/mapping_function_consistency"]),

    Stage("madrigal_reference_offset", "src/analysis/madrigal_reference_offset.py", "R1.3",
          "how much of the Madrigal error is a per-station reference offset",
          inputs=[STORE_MADRIGAL], outputs=["multiday_results/madrigal_reference_offset"]),

    Stage("weighting_ablation", "src/analysis/weighting_ablation.py", "R2.5",
          "elevation against predicted-uncertainty weighting, paired station-days",
          inputs=[WEIGHTING_RUN], outputs=["multiday_results/weighting_ablation"]),

    Stage("storm_stratification", "src/analysis/storm_stratification.py", "R2.7",
          "positioning accuracy on storm against quiet days",
          inputs=[POSITIONING, SWI], outputs=["multiday_results/storm_stratification"]),

    Stage("positioning_robustness", "src/analysis/positioning_robustness.py", "R2.7b",
          "tail behaviour and convergence of the positioning solutions",
          inputs=[POSITIONING], outputs=["multiday_results/positioning_robustness"]),

    Stage("positioning_coverage", "src/analysis/positioning_coverage.py", "R1.5",
          "which station-days each method solved, and why the rest are missing",
          inputs=["experiments"],
          outputs=["multiday_results/positioning_full_coverage"],
          min_rows={"multiday_results/positioning_full_coverage/multiday_summary.csv": 30000}),

    Stage("common_set_positioning", "src/analysis/common_set_positioning.py", "R1.5, Table 5",
          "Table 5 recomputed on the station-days every method solved",
          inputs=[POSITIONING, WEIGHTING_RUN],
          outputs=["multiday_results/common_set_positioning"],
          min_rows={"multiday_results/common_set_positioning/table5_common_set.csv": 4}),

    Stage("positioning_summary", "src/analysis/positioning_summary.py", "Table 5",
          "headline positioning table",
          inputs=[POSITIONING], outputs=["multiday_results/positioning_summary"]),

    Stage("oracle_benchmark", "src/analysis/oracle_benchmark.py", "R2.8",
          "positioning floor from reference STEC, on its own restricted set",
          inputs=[POSITIONING], outputs=["multiday_results/oracle_benchmark"]),

    # Last: reads the metric CSVs every stage above writes, so it must follow all of them.
    Stage("figures", "src/viz/revision_figures.py", "all",
          "one PNG per revision figure, plus the _notitle manuscript variants",
          inputs=["multiday_results"], outputs=["plots/revision"]),

    Stage("results_manifest", "src/analysis/results_manifest.py", "-",
          "which result trees are canonical and which are superseded",
          outputs=["multiday_results/runs_manifest.csv"],
          min_rows={"multiday_results/runs_manifest.csv": 10}),
]


def by_name() -> dict[str, Stage]:
    return {stage.name: stage for stage in STAGES}


def check_unique_outputs() -> None:
    """No two stages may claim the same output - that is how duplicates start."""
    owner: dict[str, str] = {}
    for stage in STAGES:
        for output in stage.outputs:
            if output in owner:
                raise ValueError(
                    f"{output} is claimed by both '{owner[output]}' and '{stage.name}'"
                )
            owner[output] = stage.name
