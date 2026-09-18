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
import re
from collections.abc import Sequence
from pathlib import Path

from ..analysis.daily_metrics import DATASET_LABELS, MODELS
from ..analysis.positioning_coverage import EXCLUDED_SOLVER_FAILURES_FILENAME
from ..analysis.positioning_summary import METHOD_ORDER, SUPERSEDED_FOR_TABLE5_NOTE
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
# 2026-08-28: repointed off multiday_results/positioning_runs/20260216_2052/ - a
# 2026-02-16, pre-rebuild snapshot (56,457 rows, 245 dates, 55 stations) that nothing
# regenerated - onto positioning_coverage's own multiday_summary_all_weightings.csv,
# which the same stage's collect() now writes by concatenating its fresh iono and elev
# outputs (see that module's main()). That directory is left on disk, unread by
# default from here on.
WEIGHTING_RUN = str(
    _rel(paths.analysis_result_dir("positioning_coverage", rebuilt=True))
    / "multiday_summary_all_weightings.csv"
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
# relative_error_metrics.py's default --experiment (the paper's pretrained model), read
# for its `temporal_analysis/year_*_metrics_summary.txt` files and its
# `{interpolation,extrapolation}/temporal_analysis/total_metrics_summary.txt` regime
# split. Narrowed to those three subtrees (a few hundred KB total) rather than the whole
# experiment directory: that directory's `positioning/` subtree alone is 37 GB, and
# nothing under it, `model/`, or `global_maps/` is read here. Bare literals, same
# LEGACY_ROOT-override reasoning as ORACLE_EXPERIMENT_DIR above.
_RELATIVE_ERROR_METRICS_EXPERIMENT = (
    "experiments/Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_"
    "bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
RELATIVE_ERROR_METRICS_INPUTS = [
    f"{_RELATIVE_ERROR_METRICS_EXPERIMENT}/temporal_analysis",
    f"{_RELATIVE_ERROR_METRICS_EXPERIMENT}/interpolation/temporal_analysis",
    f"{_RELATIVE_ERROR_METRICS_EXPERIMENT}/extrapolation/temporal_analysis",
]
# weighting_ablation.py's FIXED_VARIANCE_RESULTS: the third, fixed-sigma arm, read
# directly from its own experiment tree rather than from WEIGHTING_RUN's
# multiday_summary.csv (see that module's load_fixed_variance). 4.5 GB / ~27,000 files,
# but fingerprint.py's tree digest only stats - no per-file hashing above
# HASH_LIMIT_BYTES - which measured under 0.4s here, so declaring the whole subtree
# (rather than each of the 242 per-day daily_summary_iono.csv files individually) is
# cheap and simpler to keep in sync with the module. Narrowed to this one experiment,
# not all of `experiments/`, the same convention ORACLE_EXPERIMENT_DIR uses.
WEIGHTING_ABLATION_FIXED_VARIANCE_DIR = (
    "experiments/Fixed_Variance_STEC/positioning/results"
)
SWI = "data/omni_hourly_2010-2025.h5"
# build_recovered_day.py's per-(year, doy) geometry-only HDF5 tree (RINEX + nav only, no
# DCB/target fields - see that module's own UNAVAILABLE constant), the ground truth
# positioning_diagnostics.load_recovered_station_days reads for its recovered-vs-original
# population split. Not a stec.config.paths constant yet - positioning_diagnostics.py is
# still its only reader.
RECOVERED_STEC_DB = "data/recovered_stec_db"

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
# Own output dir - see the Stage's own comment for why this is a separate stage rather
# than a second population read into station_independence's directory.
STATION_INDEPENDENCE_PRETRAINED_DIR = _analysis_dir(
    "station_independence_pretrained", rebuilt=True
)
COMPUTATIONAL_COST_DIR = _analysis_dir("computational_cost", rebuilt=True)
# "repair_gim_baseline" is the stage name; "gim_baseline_repair" is the directory name
# the frozen script has always written - the one irregular case `paths.py`'s docstring
# and `stec.runs.restructure_results` both call out by name.
GIM_BASELINE_REPAIR_DIR = _analysis_dir("repair_gim_baseline", rebuilt=False)
DAILY_METRICS_DIR = _analysis_dir("daily_metrics", rebuilt=True)
UNCERTAINTY_ERROR_RELATION_DIR = _analysis_dir(
    "uncertainty_error_relation", rebuilt=True
)
# Own output dir, not a second population written into uncertainty_error_relation's -
# one owner per output (registry.validate()). The module's output filename suffix keys
# only on --dataset, not --model-variant, so `--model-variant pretrained_stec --dataset
# own` would otherwise collide with the finetuned-model stage's own by_uncertainty.csv/
# by_elevation.csv/calibrating_factor.csv.
UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR = _analysis_dir(
    "uncertainty_error_relation_pretrained", rebuilt=True
)
STRATIFIED_COMPARISON_DIR = _analysis_dir("stratified_comparison", rebuilt=True)
ACTIVITY_STRATIFICATION_DIR = _analysis_dir("activity_stratification", rebuilt=True)
IONEX_RMS_BENCHMARK_DIR = _analysis_dir("ionex_rms_benchmark", rebuilt=True)
UNCERTAINTY_CALIBRATION_DIR = _analysis_dir("uncertainty_calibration", rebuilt=True)
MAPPING_FUNCTION_CONSISTENCY_DIR = _analysis_dir(
    "mapping_function_consistency", rebuilt=True
)
MADRIGAL_REFERENCE_OFFSET_DIR = _analysis_dir("madrigal_reference_offset", rebuilt=True)
MADRIGAL_METHOD_OFFSET_COMPARISON_DIR = _analysis_dir(
    "madrigal_method_offset_comparison", rebuilt=True
)
WEIGHTING_ABLATION_DIR = _analysis_dir("weighting_ablation", rebuilt=True)
STORM_STRATIFICATION_DIR = _analysis_dir("storm_stratification", rebuilt=True)
POSITIONING_ROBUSTNESS_DIR = _analysis_dir("positioning_robustness", rebuilt=True)
POSITIONING_COVERAGE_DIR = _analysis_dir("positioning_coverage", rebuilt=True)
POSITIONING_ACTIVITY_DIR = _analysis_dir("positioning_activity", rebuilt=True)
CONSTELLATION_COVERAGE_DIR = _analysis_dir("constellation_coverage", rebuilt=True)
COMMON_SET_POSITIONING_DIR = _analysis_dir("common_set_positioning", rebuilt=True)
POSITIONING_SUMMARY_DIR = _analysis_dir("positioning_summary", rebuilt=True)
ORACLE_BENCHMARK_DIR = _analysis_dir("oracle_benchmark", rebuilt=True)
POSITIONING_DIAGNOSTICS_DIR = _analysis_dir("positioning_diagnostics", rebuilt=True)
# Owner decision 2026-08-28 (docs/revision/positioning_reporting.md): the positioning-distribution table reports
# distributions/medians with no outcome-based filter. positioning_distributions.py wrote
# that table from the start but was left undeclared while the owner looked at it before
# deciding - now declared, and it is what carries canonical_for="Tables 6 and 7" below
# (moved off positioning_summary, which implements the superseded mean/10 m-exclusion
# methodology; widened from that table alone 2026-09-15, see the Stage's own comment).
POSITIONING_DISTRIBUTIONS_DIR = _analysis_dir("positioning_distributions", rebuilt=True)
# Shared with two further, still-undeclared modules (positioning_quality_gate.py,
# positioning_model_attribution.py both reuse positioning_geography.DEFAULT_OUTPUT_DIR
# rather than defining their own) - only positioning_geography.py's own files are
# declared as this stage's outputs below, never the directory as a whole, so this stage
# never claims ownership of a file one of those two undeclared scripts writes.
POSITIONING_GEOGRAPHY_DIR = _analysis_dir("positioning_geography", rebuilt=True)
# Plain repo-relative literals, matching how every other stec.viz stage in this file
# spells its own "plots/<name>" output (e.g. DIAGNOSTIC_FIGURES_DIR above).
# plots/positioning_distributions/ is exclusive to stec.viz.positioning_distributions.
# plots/positioning_geography/ is shared with positioning_quality_gate.py's and
# positioning_model_attribution.py's own (undeclared) viz counterparts - see the
# positioning_geography_figures Stage's own comment for why only its positioning_2024/
# subdirectory is declared as this stage's output.
POSITIONING_DISTRIBUTIONS_FIGURES_DIR = "plots/positioning_distributions"
POSITIONING_GEOGRAPHY_FIGURES_DIR = "plots/positioning_geography"
# plots/oracle_benchmark/ - exclusive to stec.viz.oracle_benchmark, deployment-ready
# figures for R1.8 held outside the manuscript (owner request, 2026-09-16). See the
# oracle_benchmark_figures Stage below for why this is separate from both
# revision_figures.py's own oracle_benchmark PNG and the manuscript tree.
ORACLE_BENCHMARK_FIGURES_DIR = "plots/oracle_benchmark"
RESULTS_MANIFEST_DIR = _analysis_dir("results_manifest", rebuilt=True)
PRETRAINED_TEST_DIAGNOSTICS_DIR = _analysis_dir(
    "pretrained_test_diagnostics", rebuilt=True
)
PRETRAINED_RESIDUALS_FROM_STORE_DIR = _analysis_dir(
    "pretrained_residuals_from_store", rebuilt=True
)
ELEVATION_METRICS_FINETUNED_DIR = _analysis_dir(
    "elevation_metrics_finetuned", rebuilt=True
)
DSTEC_EVALUATION_DIR = _analysis_dir("dstec_evaluation", rebuilt=True)
# Its own top-level directory, not nested under DSTEC_EVALUATION_DIR where the artifact
# used to live: fingerprint._tree_digest walks a directory output with rglob("*"), so an
# output nested inside dstec_evaluation's own declared directory would fold into that
# stage's recorded output digest and mark it stale every time the Madrigal arm alone ran.
# ionex_rms_benchmark_code already nests inside ionex_rms_benchmark this way - a
# pre-existing instance of the same latent bug, not one to compound here.
DSTEC_EVALUATION_MADRIGAL_DIR = _analysis_dir("dstec_evaluation_madrigal", rebuilt=True)
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

# Everything below feeds the `figures` and `manuscript_figures` Stages' `inputs`, derived
# by reading `stec/viz/revision_figures.py::FIGURE_BUILDERS` and
# `stec/viz/manuscript_figures.py::FIGURE_BUILDERS` directly rather than declaring the
# entire `multiday_results/` tree those two Stages used to point at - see the two
# Stages' own comments for the false-staleness bug that declaration caused.

# revision_figures.py's `_build_relative_error_figures` has its own fallback: it first
# checks `<results_dir>/relative_error_metrics_rebuilt/yearly_metrics.csv`, and only that
# literal path - not `RELATIVE_ERROR_METRICS_DIR` above, whose real layout is
# `analyses/relative_error_metrics/rebuilt/` - so today, with that literal path absent,
# it silently falls through to the pre-rebuild flat file below. Flagged, not fixed here:
# fixing revision_figures.py's own path is out of scope for this file, but declaring
# RELATIVE_ERROR_METRICS_DIR as this stage's input would be wrong - it names a directory
# the figure code never opens - and declaring nothing would repeat the exact
# silently-stale-figure failure this narrowing is meant to prevent. Both of the module's
# own candidate paths are declared so whichever one it actually reads is tracked.
RELATIVE_ERROR_METRICS_LEGACY_FLAT_CSV = "multiday_results/relative_error_metrics.csv"
RELATIVE_ERROR_METRICS_REBUILT_CANDIDATE = (
    "multiday_results/relative_error_metrics_rebuilt"
)
# stratified_comparison's pretrained-model counterpart has no declared stage of its own
# (see CLAUDE.md's "Unreviewed"/"unclassified" note and revision_figures.py's own
# `_build_stratified_figures` comment) - read directly from the restructure's own
# "don't know" bucket, not from a `stec.analysis` output.
STRATIFIED_COMPARISON_PRETRAINED_DIR = _rel(
    paths.UNCLASSIFIED_RESULTS / "stratified_comparison_pretrained"
)

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
# or partial store: `daily_metrics` (Tables 3/4) and `positioning_summary` (the positioning-distribution table).


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


def _extract_tex_table_full_rows(tex_path: Path, label: str) -> list[list[str]]:
    """Every data row of the `tabular` tagged `\\label{label}`, as raw (unstripped-of-
    markup) cell text split on `&`. Shared scoping/filtering logic behind
    `_extract_tex_table_rows` (Table 2's row-*label* check, which only needs the first
    cell) and the manuscript-vs-CSV value checks (Tables 3-5, which need every column).

    Scoped to the text between that label and the next `\\end{tabular}`, skipping the
    header row above the first `\\midrule` (`\\textbf{Parameter} & \\textbf{Value}`, not
    a data row) and every `\\multicolumn{...}` section-title row. A `\\cmidrule`/
    similar rule line is skipped for the same reason a blank line is: it does not end in
    `\\\\`, the line terminator every real table row uses. Nothing about a cell's own
    text is normalised here - callers that need `\\revised{...}` unwrapped or other
    markup stripped do that themselves.
    """
    text = tex_path.read_text()
    marker = f"\\label{{{label}}}"
    start = text.index(marker)
    end = text.index(r"\end{tabular}", start)
    body = text[start:end]
    body = body.split(r"\midrule", 1)[-1] if r"\midrule" in body else body

    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or not line.endswith(r"\\"):
            continue
        if line in (r"\midrule", r"\bottomrule") or line.startswith(r"\multicolumn"):
            continue
        cell_text = line[: -len(r"\\")]
        rows.append([cell.strip() for cell in cell_text.split("&")])
    return rows


def _extract_tex_table_rows(tex_path: Path, label: str) -> list[str]:
    """First-column cell text of every data row in the `tabular` tagged `\\label{label}`.

    A label cell wrapped whole in `\\revised{...}` - the convention this repository uses
    for changed/new table content, since `trackchanges`/`soul` cannot nest inside a real
    `tabular` under `agujournal2019.cls` - is unwrapped to its inner text via brace
    matching, so a revised row's label still compares equal to an unrevised one's.
    Nothing else about the cell is normalised: math mode, `\\textbf{}` etc. inside a
    label are left as written, so a caller's expected-label set must spell them the same
    way the manuscript does.
    """
    return [
        _unwrap_braced_macro(row[0], r"\revised")
        for row in _extract_tex_table_full_rows(tex_path, label)
    ]


def _unwrap_braced_macro(text: str, macro: str) -> str:
    """`text` with a leading `macro{...}` spanning the whole string replaced by `...`.

    Brace-matched rather than a regex snip at the first `}`, so a nested `{...}` inside
    the argument (there is none in this table today, but a future cell might reasonably
    add one, e.g. `\\revised{$1\\times10^{-3}$}`) does not truncate the result early.
    Returns `text` unchanged if it is not exactly one `macro{...}` call.
    """
    prefix = macro + "{"
    if not text.startswith(prefix) or not text.endswith("}"):
        return text
    depth = 0
    for index in range(len(prefix) - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[len(prefix) : index] if index == len(text) - 1 else text
    return text


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


def daily_metrics_summary_has_consistent_day_counts(outputs: dict) -> str | None:
    """Every method present in a dataset must cover the same `Num_days` as its
    siblings there, catching a day that silently lost one baseline column while
    keeping the rest - exactly what happened 2026-08-25 when a `cli.py multiday`
    invocation ran without `--pretrained_baseline`: doy 224/229/294 kept
    `stec_pred`/`vtec_model_stec`/`gim_stec` but dropped `pretrained_stec_pred`,
    so "Pretrained STEC" x own_vtec_gim quietly shrank from 242 to 239 days while
    still writing a plausible, non-empty row. The presence check above cannot see
    this: it only asks whether a (model, dataset) cell has *a* row, not whether
    that row's day count matches its siblings'.

    A method that is legitimately absent from a dataset entirely (Pretrained STEC
    x madrigal - see the presence check's own docstring) writes no row there at
    all, so it never contributes a competing `Num_days` value and this check does
    not need, and must not add, a special case for it - only a *present-but-
    shrunk* row trips this.
    """
    path = DAILY_METRICS_DIR / "summary.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(path, ["Model", "dataset", "Num_days"])
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = _read_csv_rows(path)
    by_dataset: dict[str, dict[str, str]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["Model"]] = row["Num_days"]
    mismatches = {
        dataset: day_counts
        for dataset, day_counts in by_dataset.items()
        if len(set(day_counts.values())) > 1
    }
    if mismatches:
        return (
            f"{path} has models with differing Num_days within the same dataset - "
            f"a day silently missing one baseline column shrinks that model's "
            f"Num_days without failing the presence check: {mismatches}"
        )
    return None


def _strip_tex_number_markup(cell: str) -> str:
    """Plain text of a manuscript table cell, with the markup Tables 3-5 actually wrap
    a number in removed: a whole-cell `\\revised{...}`/`\\add{...}`, an inner
    `\\mathbf{...}` highlight, `$...$` math delimiters and `\\,` thin-space. Digits,
    the surrounding `[...]` interval brackets and the `--` range separator are left
    untouched for a caller to parse. A single helper rather than inlining this in both
    value-comparison checks below, since `\\revised{$\\mathbf{6.87}$ [6.09--7.66]}` is a
    real cell shape and getting the brace-matching wrong silently drops a digit rather
    than raising.
    """
    for macro in (r"\revised", r"\add"):
        cell = _unwrap_braced_macro(cell, macro)
    cell = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", cell)
    cell = cell.replace(r"\,", "").replace("$", "")
    return cell.strip()


_MEDIAN_IQR_CELL = re.compile(
    r"^(?P<median>-?\d+(?:\.\d+)?)\s*\[(?P<q1>-?\d+(?:\.\d+)?)--(?P<q3>-?\d+(?:\.\d+)?)\]$"
)
_PLAIN_NUMBER_CELL = re.compile(r"^(?P<value>-?\d+(?:\.\d+)?)$")


def _parse_median_iqr_cell(cell: str) -> tuple[float, float, float]:
    """A `median [q1--q3]` manuscript cell (Tables 3-5's RMSE/MAE/dSTEC columns) as
    three floats. Raises `ValueError` on a cell that isn't shaped this way, rather than
    returning a sentinel - a cell this cannot parse (a reformatted table, an extra
    footnote marker) is exactly the kind of drift this check must not paper over by
    silently skipping it."""
    match = _MEDIAN_IQR_CELL.match(_strip_tex_number_markup(cell))
    if not match:
        raise ValueError(f"cell {cell!r} is not shaped 'median [q1--q3]'")
    return (
        float(match["median"]),
        float(match["q1"]),
        float(match["q3"]),
    )


def _parse_plain_number_cell(cell: str) -> float:
    """A single-number manuscript cell (Tables 3/4's $R^2$ and P95 columns) as a float.
    Raises `ValueError` on a cell that isn't a bare number, same reasoning as
    `_parse_median_iqr_cell`."""
    match = _PLAIN_NUMBER_CELL.match(_strip_tex_number_markup(cell))
    if not match:
        raise ValueError(f"cell {cell!r} is not a plain number")
    return float(match["value"])


# Manuscript Tables 3/4 row label -> daily_metrics/rebuilt/summary.csv "Model" value.
# Deliberately explicit rather than `.lower()`-normalised, same reasoning as
# `MANUSCRIPT_TABLE2_ROW_BACKING`: the manuscript prints "Direct STEC model" (lower-case
# "model") while the CSV's own column says "Direct STEC Model" (capital "Model") - an
# explicit mapping means a renamed row on either side fails this check instead of
# silently no longer matching.
MANUSCRIPT_DAILY_METRICS_ROW_BACKING: dict[str, str] = {
    "Direct STEC model": "Direct STEC Model",
    "Pretrained Direct STEC model": "Pretrained STEC",
    "VTEC + Mapping": "VTEC + Mapping",
    "IGS GIM + Mapping": "IGS GIM",
}

# (manuscript table name, tex \label, daily_metrics "dataset" value) for the two tables
# this stage backs - Table 3 on the own test set, Table 4 on Madrigal.
_DAILY_METRICS_MANUSCRIPT_TABLES = (
    ("Table 3", "testset_performance", "own_vtec_gim"),
    ("Table 4", "tab:testset_performance_madrigal", "madrigal_vtec_gim"),
)

# Manuscript column -> (csv median column, csv q1 column, csv q3 column), for the two
# "median [q1--q3]" columns Tables 3/4 print in this order after the row label.
_DAILY_METRICS_IQR_COLUMNS = (
    ("RMSE", ("RMSE_median", "RMSE_q1", "RMSE_q3")),
    ("MAE", ("MAE_median", "MAE_q1", "MAE_q3")),
)


def _daily_metrics_row_mismatch(
    table_name: str, row_label: str, csv_row: dict[str, str], cells: list[str]
) -> str | None:
    """`cells` is the manuscript's own column order for Tables 3/4: row label, RMSE,
    MAE, $R^2$, P95. Returns a message naming the table, row, column and both values on
    the first disagreement, or `None` if every column agrees with `csv_row`."""
    _, rmse_cell, mae_cell, r2_cell, p95_cell = cells
    try:
        parsed_iqr = {
            "RMSE": _parse_median_iqr_cell(rmse_cell),
            "MAE": _parse_median_iqr_cell(mae_cell),
        }
        r2 = _parse_plain_number_cell(r2_cell)
        p95 = _parse_plain_number_cell(p95_cell)
    except ValueError as exc:
        return f"manuscript {table_name} row '{row_label}': {exc}"

    for column_label, csv_columns in _DAILY_METRICS_IQR_COLUMNS:
        for stat_name, manuscript_value, csv_column in zip(
            ("median", "q1", "q3"), parsed_iqr[column_label], csv_columns
        ):
            csv_value = round(float(csv_row[csv_column]), 2)
            if csv_value != manuscript_value:
                return (
                    f"manuscript {table_name} row '{row_label}' column "
                    f"{column_label} {stat_name}: manuscript prints {manuscript_value} "
                    f"but {csv_row['dataset']}/{csv_column} in {DAILY_METRICS_DIR / 'summary.csv'} "
                    f"rounds to {csv_value}"
                )

    csv_r2 = round(float(csv_row["R2_median"]), 2)
    if csv_r2 != r2:
        return (
            f"manuscript {table_name} row '{row_label}' column R2: manuscript prints "
            f"{r2} but {csv_row['dataset']}/R2_median in "
            f"{DAILY_METRICS_DIR / 'summary.csv'} rounds to {csv_r2}"
        )
    csv_p95 = round(float(csv_row["AbsErr_p95_median"]), 2)
    if csv_p95 != p95:
        return (
            f"manuscript {table_name} row '{row_label}' column P95: manuscript prints "
            f"{p95} but {csv_row['dataset']}/AbsErr_p95_median in "
            f"{DAILY_METRICS_DIR / 'summary.csv'} rounds to {csv_p95}"
        )
    return None


def daily_metrics_manuscript_tables_match_csv(outputs: dict) -> str | None:
    """Every cell of the manuscript's Table 3 (`testset_performance`, own test set) and
    Table 4 (`tab:testset_performance_madrigal`, Madrigal) must round-trip against
    `daily_metrics/rebuilt/summary.csv` at the manuscript's own 2 dp precision - the gap
    `paper_tables_manuscript_rows_are_backed`'s own docstring names explicitly: a row
    present and correctly named on both sides can still carry two different numbers.
    Both tables share `RMSE_median`/`RMSE_q1`/`RMSE_q3`, `MAE_median`/`MAE_q1`/
    `MAE_q3`, `R2_median` and `AbsErr_p95_median`, keyed by (dataset, Model) via
    `MANUSCRIPT_DAILY_METRICS_ROW_BACKING`.

    A manuscript row not in that mapping fails this check immediately, the same
    triage-before-trust rule `paper_tables_manuscript_rows_are_backed` enforces for
    Table 2 - a newly added or renamed model row must be mapped before either check can
    be believed again. Only once every printed row resolves to a real CSV row are the
    cell values themselves compared, so a value mismatch is always reported against a
    row the mapping already vouches for.
    """
    csv_path = DAILY_METRICS_DIR / "summary.csv"
    if str(csv_path) not in outputs:
        return f"{csv_path} is not a declared output of this stage"
    rows_by_key = {
        (row["dataset"], row["Model"]): row for row in _read_csv_rows(csv_path)
    }

    tex_path = paths.REPO_ROOT / "STEC_Modelling" / "PNN_main_revised.tex"
    if not tex_path.exists():
        return f"manuscript not found at {tex_path}"

    tables = [
        (table_name, dataset, _extract_tex_table_full_rows(tex_path, label))
        for table_name, label, dataset in _DAILY_METRICS_MANUSCRIPT_TABLES
    ]

    unrecognised = [
        (table_name, cells[0])
        for table_name, _dataset, rows in tables
        for cells in rows
        if _unwrap_braced_macro(cells[0], r"\revised")
        not in MANUSCRIPT_DAILY_METRICS_ROW_BACKING
    ]
    if unrecognised:
        return (
            f"manuscript prints row(s) {unrecognised} that "
            "MANUSCRIPT_DAILY_METRICS_ROW_BACKING does not know about - triage them "
            "there before trusting this check again"
        )

    mismatches = []
    for table_name, dataset, rows in tables:
        for cells in rows:
            row_label = _unwrap_braced_macro(cells[0], r"\revised")
            csv_model = MANUSCRIPT_DAILY_METRICS_ROW_BACKING[row_label]
            csv_row = rows_by_key.get((dataset, csv_model))
            if csv_row is None:
                mismatches.append(
                    f"manuscript {table_name} row '{row_label}' has no matching "
                    f"({dataset}, {csv_model}) row in {csv_path}"
                )
                continue
            mismatch = _daily_metrics_row_mismatch(
                table_name, row_label, csv_row, cells
            )
            if mismatch:
                mismatches.append(mismatch)
    if mismatches:
        return "; ".join(mismatches)
    return None


# Manuscript Table 5 (`tab:dstec`) row label -> dstec_evaluation's `COMPARISON_METHODS`
# prefix (`gim`/`vtec`/`pretrained`) or "model" for the Direct STEC model itself, which
# keeps the bare `model_*` prefix rather than one from that dict (see
# stec.analysis.dstec_evaluation's own COMPARISON_METHODS comment). Explicit for the
# same reason MANUSCRIPT_DAILY_METRICS_ROW_BACKING is: a renamed row must be triaged
# here, not silently stop matching.
MANUSCRIPT_DSTEC_ROW_BACKING: dict[str, str] = {
    "Direct STEC model": "model",
    "IGS GIM + Mapping": "gim",
    "VTEC + Mapping": "vtec",
    "Pretrained Direct STEC model": "pretrained",
}


def _read_kv_csv(path: Path) -> dict[str, str]:
    """A `pandas.Series.to_csv(path, header=["value"])` file - dstec_evaluation's own
    `summary.csv` - as `{row label: value string}`. Not `_read_csv_rows`: that assumes
    every column has a real header name, but this file's index column's header is the
    empty string pandas writes for an unnamed index."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)  # header row: "", "value"
        return {row[0]: row[1] for row in reader}


def _dstec_row_mismatch(
    row_label: str,
    prefix: str,
    own_values: dict[str, str],
    madrigal_values: dict[str, str],
    cells: list[str],
) -> str | None:
    """`cells` is Table 5's own column order: row label, own test set, Madrigal - each
    a `median [q1--q3]` dSTEC RMSE cell. Returns a message naming the column, stat,
    and both values on the first disagreement against `{prefix}_dstec_rmse_*_of_arcs`
    in the two summary.csv files, or `None` if every value agrees."""
    _, own_cell, madrigal_cell = cells
    try:
        own_stats = _parse_median_iqr_cell(own_cell)
        madrigal_stats = _parse_median_iqr_cell(madrigal_cell)
    except ValueError as exc:
        return f"manuscript Table 5 row '{row_label}': {exc}"

    for column_label, values, manuscript_stats, source in (
        ("Own test set", own_values, own_stats, DSTEC_EVALUATION_DIR / "summary.csv"),
        (
            "Madrigal",
            madrigal_values,
            madrigal_stats,
            DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv",
        ),
    ):
        for stat_name, suffix, manuscript_value in zip(
            ("median", "q1", "q3"),
            ("median_of_arcs", "q1_of_arcs", "q3_of_arcs"),
            manuscript_stats,
        ):
            csv_key = f"{prefix}_dstec_rmse_{suffix}"
            csv_value = round(float(values[csv_key]), 2)
            if csv_value != manuscript_value:
                return (
                    f"manuscript Table 5 row '{row_label}' column {column_label} "
                    f"{stat_name}: manuscript prints {manuscript_value} but {csv_key} "
                    f"in {source} rounds to {csv_value}"
                )
    return None


def dstec_manuscript_table_matches_csv(outputs: dict) -> str | None:
    """Every cell of the manuscript's Table 5 (`tab:dstec`) must round-trip against the
    two dSTEC stages' `summary.csv` at the manuscript's own 2 dp precision, the same
    value-agreement gap `daily_metrics_manuscript_tables_match_csv` closes for Tables
    3/4. Table 5 needs both dSTEC columns (own test set, Madrigal) judged together per
    row, so this check reads both stages' output rather than being split across them -
    attached only to `dstec_evaluation_madrigal` (the later of the two in run order)
    since the registry's one-owner-per-output rule means a `Check` still belongs to a
    single stage.

    `dstec_evaluation`'s `summary.csv` is read directly off disk, not gated by
    `outputs`, because it is a different, earlier stage's output - this stage cannot
    declare ownership of it. Only `dstec_evaluation_madrigal`'s own `summary.csv` is
    gated that way, the same self-consistency check every other check function in this
    file performs for its own stage's declared outputs.
    """
    madrigal_path = DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv"
    if str(madrigal_path) not in outputs:
        return f"{madrigal_path} is not a declared output of this stage"
    own_path = DSTEC_EVALUATION_DIR / "summary.csv"
    if not own_path.exists():
        return f"{own_path} is not present - run the dstec_evaluation stage first"

    own_values = _read_kv_csv(own_path)
    madrigal_values = _read_kv_csv(madrigal_path)

    tex_path = paths.REPO_ROOT / "STEC_Modelling" / "PNN_main_revised.tex"
    if not tex_path.exists():
        return f"manuscript not found at {tex_path}"

    printed_rows = _extract_tex_table_full_rows(tex_path, "tab:dstec")
    unrecognised = [
        cells[0]
        for cells in printed_rows
        if _unwrap_braced_macro(cells[0], r"\revised")
        not in MANUSCRIPT_DSTEC_ROW_BACKING
    ]
    if unrecognised:
        return (
            f"manuscript Table 5 prints row(s) {unrecognised} that "
            "MANUSCRIPT_DSTEC_ROW_BACKING does not know about - triage them there "
            "before trusting this check again"
        )

    mismatches = []
    for cells in printed_rows:
        row_label = _unwrap_braced_macro(cells[0], r"\revised")
        prefix = MANUSCRIPT_DSTEC_ROW_BACKING[row_label]
        mismatch = _dstec_row_mismatch(
            row_label, prefix, own_values, madrigal_values, cells
        )
        if mismatch:
            mismatches.append(mismatch)
    if mismatches:
        return "; ".join(mismatches)
    return None


def positioning_summary_overall_has_all_four_methods(outputs: dict) -> str | None:
    """The positioning-distribution table's overall.csv always has exactly 4 rows - `reindex(METHOD_ORDER)`
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


def positioning_distributions_overall_has_all_four_methods(outputs: dict) -> str | None:
    """The positioning-distribution table's real source now: `percentile_summary(frame, ["Method"])` groups by
    every `Method` value actually present in the input, so a method silently absent from
    the coverage-repaired positioning table (rather than reindex-guaranteed NaN, unlike
    `positioning_summary_overall_has_all_four_methods`'s target) would still pass
    `min_rows` while missing a manuscript row outright - this checks the row is really
    there for each of the four."""
    path = POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(path, ["Method", "n"])
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = {row["Method"]: row for row in _read_csv_rows(path)}
    missing = set(METHOD_ORDER) - set(rows)
    if missing:
        return f"{path} is missing method(s) {sorted(missing)}"
    empty = [method for method in METHOD_ORDER if not rows[method].get("n")]
    if empty:
        return f"{path} has no station-day count recorded for {sorted(empty)}"
    return None


def weighting_ablation_common_set_has_all_four_methods(outputs: dict) -> str | None:
    """the weighting-ablation table (tab:weighting_ablation)'s real source, `common_set.csv`: `common_set_ablation` groups by
    whatever `correction` values are present in the input and then
    `.reindex(COMMON_SET_CORRECTION_ORDER)`, which fills a missing correction with NaN
    rather than dropping the row - the same reindex-guaranteed-shape gap
    `positioning_summary_overall_has_all_four_methods` checks for the positioning-distribution table. Declared
    2026-09-15 alongside this stage's first `canonical_for` (results_register.md,
    Consistency item C: the weighting-ablation table had a directory-level output only, no `min_rows`, no
    `checks` - the exact gap Tables 3/4 used to have before they were declared)."""
    path = WEIGHTING_ABLATION_DIR / "common_set.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(
        path, ["correction", "common_set_station_days"]
    )
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = {row["correction"]: row for row in _read_csv_rows(path)}
    missing = set(METHOD_ORDER) - set(rows)
    if missing:
        return f"{path} is missing method(s) {sorted(missing)}"
    empty = [
        method
        for method in METHOD_ORDER
        if not rows[method].get("common_set_station_days")
    ]
    if empty:
        return f"{path} has no common_set_station_days recorded for {sorted(empty)}"
    return None


def madrigal_method_offset_comparison_has_all_four_methods(outputs: dict) -> str | None:
    """`pooled_before_after.csv` is built one row per `METHOD_COLUMNS` entry (see
    `build_pooled_before_after`) - always exactly 4, but `min_rows=4` alone cannot tell
    that apart from 4 rows that are the right shape with an empty `RMSE_before` (e.g. a
    method whose Madrigal column was entirely NaN, which would still satisfy the row
    count). Checks the four expected method labels are actually present with a numeric
    RMSE_before, the same shape of check `positioning_summary_overall_has_all_four_methods`
    uses for the positioning-distribution table."""
    path = MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "pooled_before_after.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(
        path, ["Method", "RMSE_before", "RMSE_after"]
    )
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = {row["Method"]: row for row in _read_csv_rows(path)}
    missing = set(MODELS.values()) - set(rows)
    if missing:
        return f"{path} is missing method(s) {sorted(missing)}"
    empty = [
        method
        for method in MODELS.values()
        if not rows[method].get("RMSE_before") or not rows[method].get("RMSE_after")
    ]
    if empty:
        return f"{path} has no RMSE recorded for {sorted(empty)}"
    return None


# --- min_rows floors for the four positioning_distributions/positioning_geography
# stages, measured directly against real output (2026-09-14, current
# 43,215-row coverage-repaired population - see POSITIONING's own history in this
# file's comments). Two shapes, per the convention every other stage in this file
# uses: a handful of tables have a row count fixed by code constants (4 PAPER_METHODS,
# 4 EXCEEDANCE_THRESHOLDS_M, 2 regime/population values, 9 LATITUDE_BINS bins) rather
# than by how much data exists, so their floor is the exact count those constants
# imply; the rest scale with the population (individual station-days, boxplot fliers),
# so their floor sits comfortably below what was actually measured, not pinned to it.
_POSITIONING_DISTRIBUTIONS_MIN_ROWS = {
    # 4 PAPER_METHODS - not reindex-guaranteed the way positioning_summary's overall.csv
    # is (percentile_summary groups by whatever Method values are present), so this is a
    # measured-currently-exact floor, not a structural one; positioning_distributions_
    # overall_has_all_four_methods is what actually enforces the four method identities.
    str(POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"): 4,
    # 4 methods x 4 EXCEEDANCE_THRESHOLDS_M, one row per combination that occurs.
    str(POSITIONING_DISTRIBUTIONS_DIR / "overall_exceedance.csv"): 16,
    str(POSITIONING_DISTRIBUTIONS_DIR / "overall_boxplot_stats.csv"): 4,
    # Individual outlier points - scales with the population's own tail, measured 3,243.
    str(POSITIONING_DISTRIBUTIONS_DIR / "overall_boxplot_fliers.csv"): 2_500,
    # One row per station-day per method - equals the source table's own row count,
    # measured 43,215.
    str(POSITIONING_DISTRIBUTIONS_DIR / "overall_cdf_points.csv"): 35_000,
    # 4 methods x 2 regimes (storm/quiet) - both present for every method given 39 of
    # 242 test-period DOYs are storm days.
    str(POSITIONING_DISTRIBUTIONS_DIR / "regime_percentile_summary.csv"): 8,
    str(POSITIONING_DISTRIBUTIONS_DIR / "regime_exceedance.csv"): 32,
    str(POSITIONING_DISTRIBUTIONS_DIR / "regime_boxplot_stats.csv"): 8,
    str(POSITIONING_DISTRIBUTIONS_DIR / "regime_boxplot_fliers.csv"): 2_500,
    # 4 methods x 2 populations (original/recovered).
    str(POSITIONING_DISTRIBUTIONS_DIR / "population_percentile_summary.csv"): 8,
    str(POSITIONING_DISTRIBUTIONS_DIR / "population_exceedance.csv"): 32,
    str(POSITIONING_DISTRIBUTIONS_DIR / "population_boxplot_stats.csv"): 8,
    str(POSITIONING_DISTRIBUTIONS_DIR / "population_boxplot_fliers.csv"): 2_000,
    # Common set (2026-09-14, owner instruction): 4 PAPER_METHODS, same shape as
    # overall_percentile_summary.csv above, over the smaller N=10,387 coverage-only
    # common-set population rather than the full one.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_percentile_summary.csv"): 4,
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_exceedance.csv"): 16,
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_component_medians.csv"): 4,
    # Common set, extended 2026-09-15 to feed Figures 12-15 (results_register.md
    # consistency item A) - same shapes as the overall_* rows above, over the smaller
    # N=10,387 common-set population rather than the full one.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_boxplot_stats.csv"): 4,
    # Individual outlier points over the common set - scales with its own tail,
    # measured 3,076.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_boxplot_fliers.csv"): 2_000,
    # One row per station-day per method over the common set, measured 41,548.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_cdf_points.csv"): 35_000,
    # 4 methods x 2 regimes (storm/quiet), same shape as regime_boxplot_stats.csv above.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_regime_boxplot_stats.csv"): 8,
    # One row per station-day per method over the common set, measured 41,548 - what
    # Figures 12 and 14 (stec.viz.manuscript_figures) group by (date, method) themselves.
    str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_daily_rows.csv"): 35_000,
}

_POSITIONING_DISTRIBUTIONS_FIGURES_MIN_ROWS = {
    # Every station-day plotted as a box-plot flier plus one summary row per group -
    # scales with the population's tail, measured 3,279.
    f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/boxplot_3d_error.csv": 2_500,
    # One row per station-day per method, same as overall_cdf_points.csv above.
    f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/cdf_unfiltered.csv": 35_000,
    # percentiles (4 rows) concatenated with exceedance (16 rows) - exact given 4
    # methods and 4 thresholds, see fig_percentile_exceedance_table.
    f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
    "percentile_exceedance_table.csv": 20,
    # 4 methods x 2 populations / 2 regimes.
    f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
    "population_split_boxplot.csv": 8,
    f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
    "storm_quiet_boxplot.csv": 8,
}

_POSITIONING_GEOGRAPHY_MIN_ROWS = {
    # One row per station with a surviving Direct STEC / paired STEC-GIM station-day -
    # scales with station coverage, measured 55 and 57 respectively.
    str(POSITIONING_GEOGRAPHY_DIR / "station_map_direct_stec.csv"): 40,
    str(POSITIONING_GEOGRAPHY_DIR / "station_map_diff.csv"): 40,
    # 9 LATITUDE_BINS bins x 4 methods, measured 36 for both geographic and geomagnetic
    # - not reindex-guaranteed (groupby(observed=True) only emits bins that actually
    # have data), so this is a measured-currently-exact floor set below the real count.
    str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_geographic.csv"): 30,
    str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_geomagnetic.csv"): 30,
    # 9 bins, one paired STEC-vs-GIM row each.
    str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_diff_geographic.csv"): 7,
    str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_diff_geomagnetic.csv"): 7,
    str(
        POSITIONING_GEOGRAPHY_DIR
        / "population_concentration_by_latitude_geographic.csv"
    ): 7,
    str(
        POSITIONING_GEOGRAPHY_DIR
        / "population_concentration_by_latitude_geomagnetic.csv"
    ): 7,
    # 9 bins x 2 populations.
    str(POSITIONING_GEOGRAPHY_DIR / "population_latitude_diff_geographic.csv"): 14,
    str(POSITIONING_GEOGRAPHY_DIR / "population_latitude_diff_geomagnetic.csv"): 14,
}

_POSITIONING_GEOGRAPHY_FIGURES_MIN_ROWS = {
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "station_map_direct_stec.csv": 40,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/station_map_diff.csv": 40,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "latitude_stratification_geographic.csv": 30,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "latitude_stratification_geomagnetic.csv": 30,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "population_by_latitude_geographic.csv": 7,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "population_by_latitude_geomagnetic.csv": 7,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "population_latitude_cross_geographic.csv": 30,
    f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
    "population_latitude_cross_geomagnetic.csv": 30,
}


# Every row the manuscript's Table 2 (`\label{tab:hyperparameters}`) prints, mapped to the
# `table2_hyperparameters.csv` "parameter" name that backs it - or to `None` when it
# deliberately has none. Two rows are legitimately `None`: `MC samples (inference)` is set
# at inference time (`src/inference_testset.py`, ported to `stec/inference/monte_carlo.py`)
# rather than read from the training config `hyperparameter_table` builds from, so this
# stage cannot see it without reading a second, unrelated source. Two printed rows -
# `$\beta$` and the KL weight schedule - both describe the same annealed KL coefficient at
# different points in training (its steady-state value vs. its full schedule) and are
# backed by the same "KL weight" CSV row on purpose, not because one of them is spurious.
# This dict is the single place that decision lives; `paper_tables_manuscript_rows_are_backed`
# below is what enforces it.
MANUSCRIPT_TABLE2_ROW_BACKING: dict[str, str | None] = {
    "Optimizer": "Optimiser",
    "Loss": "Loss",
    r"$\beta$": "KL weight",
    "KL weight schedule": "KL weight",
    "Scheduler": "Scheduler",
    "Batch size": "Batch size",
    "Learning rate": "Learning rate",
    "Max epochs": "Epochs",
    "Early stopping patience": "Early stopping patience",
    "Samples per epoch": "Training subset size",
    "Weight decay": "Weight decay",
    "Residual MLP hidden dim.": "Hidden dimension",
    "Residual blocks": "Residual blocks",
    "Activation": "Activation",
    "Dropout rate": "Dropout",
    "Prior std.": "Prior sigma",
    "Variance floor": "Variance floor",
    "Output bias init.": "Output bias init",
    "MC samples (inference)": None,
}


def paper_tables_manuscript_rows_are_backed(outputs: dict) -> str | None:
    """Every row printed in the manuscript's Table 2 must resolve to a real parameter in
    `table2_hyperparameters.csv` - the exact bug this check exists to catch: the
    manuscript's "Early stopping patience & 20 (15)" row was factually correct against
    the shipped configs but had no CSV row backing it at all, so nothing but manual
    inspection could have told the two apart from a row that was simply wrong. Reads the
    CSV this stage just wrote and the real `STEC_Modelling/PNN_main_revised.tex`, not
    copies, so a hand-edit to either side is caught the next time this stage runs rather
    than at the next unrelated audit.

    A label the parser finds in the manuscript that is not a key in
    `MANUSCRIPT_TABLE2_ROW_BACKING` also fails this check - a newly added or renamed
    printed row must be triaged into that mapping (backed by a real parameter, or
    explicitly `None` with a reason) before it can be trusted, not merely printed.

    What this does **not** check, so as not to claim more than it delivers: cell
    *values* agreeing between the manuscript and the CSV (a row present, correctly
    named, on both sides but carrying two different numbers would still pass), and the
    reverse direction - the CSV intentionally carries rows the manuscript does not print
    (`Architecture`, `Random seed`, `SH degree`, the scheduler's own patience; see
    `stec.analysis.paper_tables` for which of the CSV's rows were deliberately left out
    of the printed table and why), so an unprinted CSV parameter is not a violation.
    """
    csv_path = PAPER_TABLES_DIR / "table2_hyperparameters.csv"
    if str(csv_path) not in outputs:
        return f"{csv_path} is not a declared output of this stage"
    csv_parameters = {row["parameter"] for row in _read_csv_rows(csv_path)}

    tex_path = paths.REPO_ROOT / "STEC_Modelling" / "PNN_main_revised.tex"
    if not tex_path.exists():
        return f"manuscript not found at {tex_path}"
    printed_rows = _extract_tex_table_rows(tex_path, "tab:hyperparameters")

    unrecognised = [
        row for row in printed_rows if row not in MANUSCRIPT_TABLE2_ROW_BACKING
    ]
    if unrecognised:
        return (
            f"manuscript Table 2 prints row(s) {unrecognised} that "
            "MANUSCRIPT_TABLE2_ROW_BACKING does not know about - triage them there "
            "before trusting this check again"
        )

    unbacked = [
        row
        for row in printed_rows
        if (backing := MANUSCRIPT_TABLE2_ROW_BACKING[row]) is not None
        and backing not in csv_parameters
    ]
    if unbacked:
        return (
            f"manuscript Table 2 prints row(s) {unbacked} with no matching parameter "
            f"in {csv_path}"
        )
    return None


STORM_STRATIFICATION_STEC_DIR = _analysis_dir("storm_stratification_stec", rebuilt=True)


def storm_stratification_stec_by_regime_has_all_methods(outputs: dict) -> str | None:
    """by_regime.csv (the STEC-domain storm/quiet split) must report all four methods
    for the own test set, not merely clear a row-count floor - the same marginal-
    coverage check `daily_metrics_summary_has_all_methods_and_datasets` performs for
    Tables 3/4, reused here because this stage reads the same per_day.csv MODELS
    vocabulary. Not enforced for Madrigal: `daily_metrics`'s own check already
    documents that a (model, dataset) cell can be legitimately absent there."""
    path = STORM_STRATIFICATION_STEC_DIR / "by_regime.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(path, ["dataset", "method", "regime"])
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = _read_csv_rows(path)
    own_rows = [row for row in rows if row["dataset"] == DATASET_LABELS["own"]]
    missing_methods = set(MODELS.values()) - {row["method"] for row in own_rows}
    if missing_methods:
        return (
            f"{path} is missing method(s) {sorted(missing_methods)} for dataset "
            f"{DATASET_LABELS['own']!r}"
        )
    seen_regimes = {row["regime"] for row in own_rows}
    if seen_regimes != {"quiet", "storm"}:
        return (
            f"{path} does not have both 'quiet' and 'storm' regimes for dataset "
            f"{DATASET_LABELS['own']!r}: {sorted(seen_regimes)}"
        )
    return None


def storm_stratification_stec_by_regime_has_consistent_day_counts(
    outputs: dict,
) -> str | None:
    """Every method within the same (dataset, regime) must cover the same `n_days` as
    its siblings there - `stratify()` inner-joins every method against the identical
    `per_day.csv`, so a mismatch means one method silently dropped a day, the STEC-
    domain analogue of `daily_metrics_summary_has_consistent_day_counts` above."""
    path = STORM_STRATIFICATION_STEC_DIR / "by_regime.csv"
    if str(path) not in outputs:
        return f"{path} is not a declared output of this stage"
    missing_columns = _missing_csv_columns(
        path, ["dataset", "method", "regime", "n_days"]
    )
    if missing_columns:
        return f"{path} is missing column(s) {sorted(missing_columns)}"
    rows = _read_csv_rows(path)
    by_group: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        by_group.setdefault((row["dataset"], row["regime"]), set()).add(row["n_days"])
    mismatches = {key: counts for key, counts in by_group.items() if len(counts) > 1}
    if mismatches:
        return (
            f"{path} has methods with differing n_days within the same "
            f"(dataset, regime): {mismatches}"
        )
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
        f"--space-weather {SMOKE_SWI} --store-root artifacts/predictions "
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
        # floored comfortably below the real counts (24 and 26 rows respectively) so
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
            "The CSV intentionally carries more rows than the manuscript prints - "
            "Architecture, Random seed, SH degree and the ReduceLROnPlateau scheduler's "
            "own patience (as opposed to early stopping's, a different mechanism) were "
            "judged not to need a printed row; see stec.analysis.paper_tables for the "
            "reasoning on each. paper_tables_manuscript_rows_are_backed only checks the "
            "other direction - every printed row resolves to a real CSV parameter - not "
            "that the two carry identical row sets.",
        ],
        checks=[paper_tables_manuscript_rows_are_backed],
    ),
    Stage(
        "relative_error_metrics",
        f"-m stec.analysis.relative_error_metrics --output-dir {RELATIVE_ERROR_METRICS_DIR}",
        "R2.1, R2.2",
        "absolute vs TEC-normalised error by year; interpolation vs extrapolation",
        inputs=RELATIVE_ERROR_METRICS_INPUTS,
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
            "Limited by n = 58 test stations (per_station.csv's own row count - 55 was "
            "stale), not by observation count. Adding days sharpens each point but "
            "does not sharpen the Spearman coefficient.",
            "Strengthening this result needs a region-held-out retrain, not more data.",
        ],
    ),
    Stage(
        # Own directory, not a second population written into station_independence's -
        # one owner per output (registry.validate()). Same module, same CLI, just
        # --model-variant/--dataset/--years pointed at the pretrained model's own store
        # partition instead of the daily fine-tuned models' - see the module's
        # per_station_error() for why --years is mandatory here (pretrained_stec/own
        # spans 2014-2024, and a bare doy would otherwise silently pool other years in).
        "station_independence_pretrained",
        f"-m stec.analysis.station_independence --model-variant pretrained_stec "
        f"--years 2024 --output-dir {STATION_INDEPENDENCE_PRETRAINED_DIR}",
        "R2.3",
        "test-station error against distance to the nearest training station, for the "
        "pretrained model on the same 2024 population as station_independence",
        inputs=[STORE_PRETRAINED],
        outputs=[str(STATION_INDEPENDENCE_PRETRAINED_DIR)],
        canonical_for=None,
        caveats=[
            "Population: pretrained model, own test set, 2024 only (DOY 122-366) - "
            "predictions/pretrained_stec/own holds a fixed random subset of the test "
            "split (10,000,000 observations spanning 2014-2024, all 242 days of 2024 "
            "among them - see stec.analysis.pretrained_test_diagnostics's docstring), "
            "not the full test set the daily fine-tuned model is scored against.",
            "Limited by n = 58 test stations (this stage's own per_station.csv row "
            "count), not by observation count. Adding days sharpens each point but "
            "does not sharpen the Spearman coefficient.",
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
        checks=[
            daily_metrics_summary_has_all_methods_and_datasets,
            daily_metrics_summary_has_consistent_day_counts,
            daily_metrics_manuscript_tables_match_csv,
        ],
        canonical_for="Tables 3 and 4",
        caveats=[
            "The published RMSE is RMSE_mean - the mean of per-day RMSEs - which is what "
            "the manuscript states. pooled_RMSE in the same file is over observations and "
            "is consistently higher; the two are not interchangeable.",
            "Madrigal rows carry the madrigal_reference_offset caveat.",
            "Tables 3/4 report RMSE/MAE/R2 with across-day median and quartiles "
            "(RMSE_median/RMSE_q1/RMSE_q3), not mean +/- std - the day distribution is "
            "skewed and mean +/- std implies a symmetry it does not have. The _mean/_std "
            "columns are kept so nothing already published moves. AbsErr_p95_median/"
            "AbsErr_p99_median are the across-day median of each day's own absolute-error "
            "percentile, not a pooled percentile over all observations.",
        ],
        supersedes=[str(WITH_PRETRAINED_BASELINE_SUMMARY / "summary_statistics.csv")],
    ),
    Stage(
        "uncertainty_error_relation",
        f"-m stec.analysis.uncertainty_error_relation --output-dir {UNCERTAINTY_ERROR_RELATION_DIR}",
        "R2.6, R1.2",
        "predicted uncertainty against realised error, pooled over the test period",
        inputs=[STORE_OWN],
        outputs=[
            str(UNCERTAINTY_ERROR_RELATION_DIR),
            str(UNCERTAINTY_ERROR_RELATION_DIR / "by_elevation.csv"),
            str(UNCERTAINTY_ERROR_RELATION_DIR / "calibrating_factor.csv"),
        ],
        min_rows={
            str(UNCERTAINTY_ERROR_RELATION_DIR / "by_elevation.csv"): 5,
            str(UNCERTAINTY_ERROR_RELATION_DIR / "calibrating_factor.csv"): 5,
        },
        caveats=[
            "Bins are now fixed TECU intervals (0-1-2-3-4-5-7-10-15-20-30-inf), not the "
            "previous first-day sigma deciles. Those deciles were computed from "
            "DOY 122's pred_total_unc distribution alone and reused unchanged for the "
            "other 241 days, so a bin labelled 'top decile' held 6.88%-18.80% of the "
            "full-year population rather than 10% - a 'decile' meant something "
            "different on every day but the first.",
            "calibrating_factor.csv states the single scalar that would calibrate the "
            "predicted uncertainty AND its spread across elevation bands. The spread is "
            "the load-bearing number: a uniform factor is a scale error, a varying one "
            "would be a broken uncertainty model, and only the former is safe to report "
            "as a post-hoc recalibration.",
        ],
    ),
    Stage(
        # Own directory, not a second population written into uncertainty_error_relation's
        # - one owner per output (registry.validate()). Same module, same CLI, just
        # --model-variant/--dataset pointed at the pretrained model's own store partition;
        # the module's output filename suffix keys only on --dataset (empty for "own"),
        # so without a separate directory this would silently overwrite the finetuned
        # model's by_uncertainty.csv/by_elevation.csv/calibrating_factor.csv.
        "uncertainty_error_relation_pretrained",
        f"-m stec.analysis.uncertainty_error_relation --model-variant pretrained_stec "
        f"--dataset own --output-dir {UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR}",
        "R1.6, R2.6",
        "predicted uncertainty against realised error, pretrained model, same "
        "population as Figure 9 (fig_uncertainty)",
        inputs=[STORE_PRETRAINED],
        outputs=[
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR),
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "by_uncertainty.csv"),
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "by_elevation.csv"),
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "calibrating_factor.csv"),
        ],
        min_rows={
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "by_uncertainty.csv"): 5,
            str(UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "by_elevation.csv"): 5,
            str(
                UNCERTAINTY_ERROR_RELATION_PRETRAINED_DIR / "calibrating_factor.csv"
            ): 5,
        },
        canonical_for=None,
        caveats=[
            "Population: pretrained model, own test set (predictions/pretrained_stec/"
            "own) - a fixed random subset of the test split, 10,000,000 observations "
            "spanning 2014-2024 (see stec.analysis.pretrained_test_diagnostics's "
            "docstring), the same population Figure 9 (fig_uncertainty) is drawn from. "
            "Not the 2024-only daily fine-tuned test set uncertainty_error_relation "
            "itself reports on.",
            "Bins are fixed TECU intervals (0-1-2-3-4-5-7-10-15-20-30-inf), not sigma "
            "deciles - see uncertainty_error_relation's own caveat for why a "
            "decile-based partition was rejected.",
            "calibrating_factor.csv states the single scalar that would calibrate the "
            "predicted uncertainty AND its spread across elevation bands. The spread is "
            "the load-bearing number: a uniform factor is a scale error, a varying one "
            "would be a broken uncertainty model, and only the former is safe to report "
            "as a post-hoc recalibration.",
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
        "model uncertainty against the IGS GIM RMS map",
        inputs=[STORE_OWN],
        outputs=[str(IONEX_RMS_BENCHMARK_DIR)],
    ),
    Stage(
        # A second Stage, not a --gim_type parameter on the one above: the registry
        # gives each stage exactly one command and one .pipeline/<name>.json (see
        # uncertainty_calibration/uncertainty_calibration_pretrained's own comment
        # above, "a single stage issuing two invocations would blur 'what produced
        # this output'"). ionex_rms_benchmark.py's main() scores one --gim_type per
        # invocation, so the CODE arm needs its own invocation the same way the
        # pretrained-model uncertainty calibration needed its own. Before this
        # existed, the letter's R1.6b table paired a current IGS row (this file's
        # `rebuilt/` output) with a CODE row that only `pre_rebuild/` had - two
        # different vintages read as if they were one run.
        "ionex_rms_benchmark_code",
        f"-m stec.analysis.ionex_rms_benchmark --gim_type CODE "
        f"--output_dir {IONEX_RMS_BENCHMARK_DIR}",
        "R1.6b",
        "model uncertainty against the CODE GIM RMS map, the table's second arm",
        inputs=[STORE_OWN],
        # Scoped to the four gim_type=CODE files, not the shared directory the IGS
        # stage above already owns - the two invocations write into the same
        # directory (per_day_IGS.csv next to per_day_CODE.csv, etc.), so claiming
        # the directory itself here would collide with the IGS stage under
        # check_unique_outputs. Neither stage's min_rows can be asserted against a
        # directory anyway (provenance.output_record only counts rows for a file).
        outputs=[
            str(IONEX_RMS_BENCHMARK_DIR / "per_day_CODE.csv"),
            str(IONEX_RMS_BENCHMARK_DIR / "overall_CODE.csv"),
            str(IONEX_RMS_BENCHMARK_DIR / "by_elevation_CODE.csv"),
            str(IONEX_RMS_BENCHMARK_DIR / "by_regime_CODE.csv"),
        ],
        min_rows={
            # per_day: up to 3 products (Direct STEC, VTEC + Mapping, CODE GIM +
            # Mapping) x up to 5 elevation rows (all + 4 bins, each bin only
            # written when it clears 1000 observations) x 242 days - 3,630 on the
            # real store; floored well below that so a handful of thin days
            # doesn't fail a real run, but a near-empty file does.
            str(IONEX_RMS_BENCHMARK_DIR / "per_day_CODE.csv"): 2_000,
            # overall/by_elevation/by_regime are groupby aggregates with a fixed
            # shape - 3 products, 3 x 4 elevation bins, 3 x 2 regimes - not a
            # floor with headroom, an exact count.
            str(IONEX_RMS_BENCHMARK_DIR / "overall_CODE.csv"): 3,
            str(IONEX_RMS_BENCHMARK_DIR / "by_elevation_CODE.csv"): 12,
            str(IONEX_RMS_BENCHMARK_DIR / "by_regime_CODE.csv"): 6,
        },
        caveats=[
            "CODE's RMS is a differently constructed product than the IGS combined "
            "RMS this table's other row uses - finer resolution, and not read as a "
            "validated error estimate any more than the IGS one is. The two rows "
            "are complementary independent checks, not a consistency test against "
            "each other.",
            "Unlike the IGS arm, main() has no stored gim_stec column to check the "
            "recomputed CODE GIM mean against - the store only ever carried the "
            "IGS-derived baseline - so this arm has no equivalent of the IGS arm's "
            "1e-3 TECU drift assertion against a second, independent computation of "
            "the same mean. What IS checked: the module's own ionex_path() picks the "
            "codg*.i file, not igsg*.i, and a day with no CODE IONEX on disk is "
            "skipped and logged, not silently zero-filled.",
            "Mapping-function error is not represented in either IONEX RMS, so both "
            "arms are structurally expected to under-cover at low elevation - not "
            "evidence specific to CODE.",
            "A grid-cell uncertainty at 5 degrees / 2 hours, judged here by "
            "per-observation coverage - the comparison the paper needs, not a "
            "like-for-like test of what the RMS was designed to represent.",
        ],
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
        "--model-variant pretrained_stec --dataset own --period-split",
        "R1.6, R2.6",
        "coverage, PIT and CRPS for the pretrained variant, same likelihoods and regimes, "
        "plus a 2014-2023 vs 2024 period split",
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
            "--period-split adds period_2014_2023 (the multi-year interpolation-era "
            "test months) and period_2024 (the DOY 122-366 extrapolation year "
            "finetuned_stec/own also covers) as regimes in coverage.csv/scores.csv, on "
            "top of the existing all/quiet/storm rows, which are unchanged. period_2024 "
            "is further split into period_2024_quiet/period_2024_storm by the same "
            "daily-Dst rule as quiet/storm - there is no period_2014_2023_quiet/_storm, "
            "since only the 2024 population has ever needed an activity split. These "
            "rows replace what used to be an ad-hoc, undeclared computation of the same "
            "period split; read the numbers from the CSV, not from any figure typed "
            "into this file.",
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
        # Answers the question madrigal_reference_offset above never asked: that stage
        # decomposes Direct STEC's (and, for the agreement check, the IGS GIM's) Madrigal
        # error into a per-station offset plus residual, but VTEC + Mapping currently
        # *beats* Direct STEC on every Madrigal metric in daily_metrics's summary.csv -
        # the reverse of the own-test-set result, where Direct STEC wins by 23%. Nobody
        # had checked whether that reversal is a real generalisation difference or the
        # same per-station reference offset carried by a different method. Must follow
        # both madrigal_reference_offset (shares MIN_OBSERVATIONS_PER_STATION, imported
        # from it) and daily_metrics (its summary.csv is the correctness-check input) in
        # run order - both already sit earlier in this file.
        "madrigal_method_offset_comparison",
        "-m stec.analysis.madrigal_method_offset_comparison --output-dir "
        f"{MADRIGAL_METHOD_OFFSET_COMPARISON_DIR}",
        "R1.3",
        "whether VTEC + Mapping's Madrigal win over Direct STEC survives the same "
        "per-station reference-offset correction, for all four methods on identical rows",
        inputs=[STORE_MADRIGAL, str(DAILY_METRICS_DIR / "summary.csv")],
        outputs=[
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR),
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "per_station_offsets.csv"),
            str(
                MADRIGAL_METHOD_OFFSET_COMPARISON_DIR
                / "row_intersection_diagnostics.csv"
            ),
            str(
                MADRIGAL_METHOD_OFFSET_COMPARISON_DIR
                / "correctness_check_vs_daily_metrics.csv"
            ),
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "pooled_before_after.csv"),
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "offset_correlation.csv"),
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "FINDINGS.md"),
        ],
        min_rows={
            # Same ~67-station population madrigal_reference_offset finds, floored the
            # same way - comfortably below the real count, well above an empty or
            # near-empty table.
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "per_station_offsets.csv"): 30,
            # 4 methods + 1 "ALL FOUR METHODS (intersected)" row - exact, fixed by
            # METHOD_COLUMNS' length (build_row_diagnostics), not by the data.
            str(
                MADRIGAL_METHOD_OFFSET_COMPARISON_DIR
                / "row_intersection_diagnostics.csv"
            ): 5,
            # One row per method daily_metrics also reports for dataset=madrigal_vtec_gim
            # - measured 2026-09-14, summary.csv currently carries all 4 Madrigal rows
            # (Direct STEC, Pretrained STEC, VTEC + Mapping, IGS GIM).
            str(
                MADRIGAL_METHOD_OFFSET_COMPARISON_DIR
                / "correctness_check_vs_daily_metrics.csv"
            ): 4,
            # One row per METHOD_COLUMNS entry (build_pooled_before_after) - exact.
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "pooled_before_after.csv"): 4,
            # 4 choose 2 method pairs (offset_correlation_matrix) - exact.
            str(MADRIGAL_METHOD_OFFSET_COMPARISON_DIR / "offset_correlation.csv"): 6,
        },
        checks=[madrigal_method_offset_comparison_has_all_four_methods],
        canonical_for=(
            "Madrigal per-station reference-offset diagnostic (R1.3) - descriptive "
            "only, not a manuscript results table"
        ),
        caveats=[
            *MADRIGAL_CAVEAT,
            "Removing a per-station offset fitted on the same data can only reduce "
            "RMSE/MAE, never increase it - with ~67 station parameters against ~449M "
            "observations the resulting optimism is negligible, but it is not exactly "
            "zero. Stated in FINDINGS.md, not left for a reader to wonder about.",
            "The before/after comparison is restricted to the row set all four methods "
            "have a finite prediction for, and to stations clearing "
            "madrigal_reference_offset.MIN_OBSERVATIONS_PER_STATION - see "
            "row_intersection_diagnostics.csv for whether that population differs "
            "meaningfully from the full store daily_metrics reports against.",
            "Owner decision, 2026-09-14: the offset-removed re-scoring in this stage's "
            "output (FINDINGS.md's 'SENSITIVITY DIAGNOSTIC' section, "
            "pooled_before_after.csv's *_after columns) is fitted on the evaluation "
            "data and reverses the method ranking in the paper's favour. It must NOT "
            "be quoted in the manuscript or the response letter as a corrected result, "
            "and must NOT be used to re-rank the methods. The plain, uncorrected "
            "comparison - matching daily_metrics/Table 4, VTEC + Mapping lowest RMSE "
            "and MAE and highest R2 on Madrigal - is the result; see "
            "docs/revision/manuscript_change_list.md for the decision record.",
        ],
    ),
    Stage(
        # Must precede weighting_ablation, storm_stratification, positioning_robustness,
        # common_set_positioning, positioning_summary and oracle_benchmark below - they
        # all read POSITIONING or WEIGHTING_RUN, both of which this stage now owns (see
        # the constants' own comments). Moved here, ahead of them, for exactly that
        # reason; it used to sit after storm_stratification/positioning_robustness with
        # no ordering consequence, back when POSITIONING pointed at a tree nothing in
        # this file produced. weighting_ablation joined this dependency 2026-08-28, when
        # WEIGHTING_RUN was repointed off the frozen 20260216_2052 snapshot onto this
        # stage's own multiday_summary_all_weightings.csv - it must therefore also sit
        # after this stage now, and was moved here (previously it ran before
        # positioning_coverage, with no dependency on it) for exactly that reason.
        "positioning_coverage",
        f"-m stec.analysis.positioning_coverage --output-dir {POSITIONING_COVERAGE_DIR}",
        "R1.5",
        "which station-days each method solved, and why the rest are missing",
        inputs=["experiments"],
        outputs=[
            str(POSITIONING_COVERAGE_DIR),
            POSITIONING,
            # The elev pair: written by the same `main()` alongside the iono outputs
            # above (weighting="both" is the default), but until now absent from this
            # list, so neither carried a `min_rows` floor nor a digest - a truncated
            # write here would have skipped silently. Added once real row counts existed
            # to derive a floor from (measured 2026-08-26, see min_rows below), not
            # blind.
            str(POSITIONING_COVERAGE_DIR / "multiday_summary_elev.csv"),
            str(POSITIONING_COVERAGE_DIR / "coverage_elev.csv"),
            # The concatenation of the iono and elev multiday_summary files above,
            # written once both have been produced this run - see main()'s own
            # comment. This is what weighting_ablation/common_set_positioning/
            # oracle_benchmark now read as WEIGHTING_RUN instead of the frozen
            # 20260216_2052 snapshot (2026-08-28).
            WEIGHTING_RUN,
            # The solver-failure exclusion (owner decision 2026-09-18, see this
            # stage's own caveats and docs/revision/positioning_reporting.md) - the
            # station-days dropped from every file above because every one of the
            # eight arms exceeded SOLVER_FAILURE_THRESHOLD_M. No min_rows floor: the
            # correct count on a clean day is 0, so a row-count floor would be
            # meaningless (existence, not size, is what a truncated write would break
            # here, and this file is never large enough for that failure mode anyway).
            str(POSITIONING_COVERAGE_DIR / EXCLUDED_SOLVER_FAILURES_FILENAME),
        ],
        # A header-only or drastically truncated multiday_summary.csv is exactly the
        # failure this catches - see the 2026-08-24 finding below: three individual
        # per-day source files on disk are themselves truncated this way, though not
        # enough to sink the whole aggregate below this floor. 30,000 sits comfortably
        # under the 37,209 rows the post-recovery iono run produced and well above
        # anything a truncated run could produce.
        #
        # The elev pair floors use the same margin, derived from what is actually on
        # disk today (measured 2026-08-26, pre-recovery-sweep vintage - see this
        # stage's own caveats): multiday_summary_elev.csv carries 37,241 rows, almost
        # exactly its iono sibling's 37,209, so it gets the same 30,000 floor rather
        # than a separately-tuned number. coverage_elev.csv carries 11,641 rows (this
        # is the "elev (stale, pre-sweep)" row of the coverage table in
        # docs/revision/work_queue.md); 9,000 sits at the same ~80% margin below actual
        # that the iono floor uses.
        min_rows={
            POSITIONING: 30_000,
            str(POSITIONING_COVERAGE_DIR / "multiday_summary_elev.csv"): 30_000,
            str(POSITIONING_COVERAGE_DIR / "coverage_elev.csv"): 9_000,
            # The iono + elev concatenation - sum of the two floors above, same
            # ~80% margin below the 80,342 actual (43,101 + 37,241) measured
            # 2026-08-26 (pre-recovery-sweep vintage, like its two inputs).
            WEIGHTING_RUN: 60_000,
        },
        canonical_for="positioning station-day coverage",
        caveats=[
            "Canonical variant selection is explicit: it matches the canonical "
            "directory name directly rather than de-duplicating multiple "
            "Finetune_STEC_2024_<DOY>_* matches by sort order. Sort-order dedup let "
            "lr1e-4_bs2048/lr1e-4_bs10000 win over the paper's lr2e-4_bs512 for 31 DOYs "
            "once the station-recovery sweep created a second directory per day.",
            "Coverage as of the last regenerated coverage.csv (2026-09-18, this "
            "stage's own .pipeline record), iono weighting: 10,792 / 9 / 35 of "
            "10,836 station-days solved by all methods / all ML methods missing "
            "(station absent from STEC DB) / some ML methods missing (per-method "
            "PPPx failure) - moved from 10,793/9/35 of 10,837 the same day, by the "
            "solver-failure exclusion below (URUM/365 was 'solved by all methods'), "
            "not a coverage-recovery move like the ones narrated next. **Moved twice "
            "more the same day before that, both by the same "
            "underlying fix landing in two steps**: the ref_source mean->ground_truth "
            "repair first operated at whole-file granularity, so any file with even one "
            "unrelated genuine missing-SINEX station (LICC/MAR7/UCLU/KIR8/BRMG/NAUS) "
            "STOPped entirely and left 1,928 contaminated model_iono rows on disk for "
            "`collect()`'s new ref_source filter to exclude - moving 'solved by all' to "
            "9,715 and 'some ML missing' to 1,111, a correct but overly conservative "
            "intermediate state (see the 2026-09-18 coordinator follow-up). "
            "`verification/repair_overwritten_summaries.py` was then extended to handle "
            "missing SINEX per station rather than per file (`find_missing_sinex_drops`): "
            "a station with SINEX is rebuilt against ground truth, a station without it "
            "is dropped from just its own row rather than blocking the whole file. "
            "Applied: 1,765 more rows upgraded to ground_truth and 163 correctly dropped "
            "for missing SINEX (1,765+163, plus the 66 already fixed earlier the same "
            "day = 1,994 - the original contaminated-row count, exactly), restoring "
            "coverage to essentially the same level as the earlier GIM-arm fix "
            "(10,793/9/34 of 10,836) rather than leaving the population artificially "
            "gutted - this is what pulled the 10-11 May 2024 superstorm (DOY 131-132) "
            "back into the common set. Quoted here as a read of coverage.csv's own "
            "'cause' column counts, not re-derived from first principles. This has moved "
            "four times since the number this caveat originally quoted: pre-sweep "
            "8,003 / 2,311 / 510 of 10,824, then post-station-recovery-sweep "
            "(2026-08-24) 8,195 / 1,591 / 1,067 of 10,853, then the second recovery "
            "sweep (2026-08-27) 10,598 / 26 / 229 of 10,853 - the 'all ML missing' "
            "population fell from 1,591 to 26 across that move, which the session "
            "recording it did not investigate the mechanism of. The 2026-09-14 move "
            "(10,598->10,712 solved / 229->115 some-missing, 'all ML missing' "
            "unchanged at 26) closed roughly half the remaining 'some ML missing' "
            "bucket: the STEC arm's correction-generation step had simply never been "
            "pointed at data/recovered_stec_db (the existing geometry-recovery "
            "sweep's own output) for DOY 122-153, even though VTEC/Pretrained_STEC "
            "already read it there - generating STEC corrections against that root "
            "and re-solving PPPx for the dozen affected DOYs closed 114 station-days "
            "with no new geometry work at all. **The 2026-09-17 move (10,712->10,793 "
            "solved, 115->34 some-missing, 26->9 all-missing) is a different fix, not "
            "another round of the same recovery work**: `daily_summary_iono.csv` had "
            "been carrying elevation-weighted 'gim' rows as a per-station fallback "
            "wherever no genuine 'gim_iono' solution existed, and both "
            "`verification/repair_overwritten_summaries.py` (the rebuild) and this "
            "module's own `collect()` (`GENUINE_GIM_METHOD`) now drop those rows "
            "instead of folding them into the iono GIM arm - see this module's own "
            "top-of-file comment. That changed which station-days count as GIM-solved "
            "at all for this weighting (10,853 -> 10,836 total), not only the "
            "solved/missing split within them, so the station-list breakdown below "
            "(the specific ~101/24 counts and the named stations) is keyed to the "
            "pre-2026-09-17 115-row 'some ML missing' population and has not been "
            "re-derived against the new 34-row one - treat the named stations as a "
            "pre-fix sample of the mechanism, not a current enumeration. The pre-fix "
            "115 split two ways, both confirmed rather than assumed at the time: "
            "~101 station-days (a recurring handful of stations - BRMG, LICC, KIR8, "
            "MAR7, UCLU, NAUS, NKLG, WUH2, UNSA, YKRO) had a .pos file but no "
            "daily_summary_iono.csv row for any method, model or GIM alike, because "
            "the day's own SINEX ground truth genuinely has no entry for that station "
            "(confirmed by grepping the SINEX file directly, e.g. DOY 122's has zero "
            "LICC occurrences); the other ~14 had no correction because the "
            "underlying station observations were absent from both the primary STEC "
            "database and data/recovered_stec_db. **The 'second finding' this caveat "
            "used to flag here is now the 2026-09-17 fix, not an open item**: for at "
            "least LICC/DOY122, the 'GIM solved this station-day' signal that put it "
            "into 'some ML missing' rather than 'all ML missing' traced to exactly the "
            "stale, pre-iono/elev-split 'gim' row inside the non-per-DOY "
            "Pretrained_STEC tree's daily_summary_iono.csv that GENUINE_GIM_METHOD now "
            "drops - dropping it removes LICC/DOY122 (and the rest of that mechanism's "
            "station-days) from the GIM-solved population entirely, which is the "
            "10,853->10,836 total drop noted above, rather than reclassifying them "
            "into a different 'missing' bucket. State which vintage backs a given "
            "number, and treat the common-set population (common_set_positioning) as "
            "still meaningfully smaller than the full-recovered-set population "
            "(positioning_summary) unless re-checked against the current coverage.csv.",
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
            "Solver-failure exclusion, owner decision 2026-09-18 "
            "(docs/revision/positioning_reporting.md): a station-day is dropped from "
            "every output of this stage - and so from every downstream population that "
            "reads them (weighting_ablation, storm_stratification, positioning_summary, "
            "oracle_benchmark, positioning_diagnostics, positioning_activity, "
            "constellation_coverage, positioning_robustness, positioning_geography, "
            "common_set_positioning) - when *every one* of the eight arms (all four "
            "methods, both weightings) reports a 3D RMS position error above "
            "`positioning_coverage.SOLVER_FAILURE_THRESHOLD_M` (100 m). This is a "
            "data-quality rule about station/solver failures independent of which "
            "correction method was used, not an outcome-based filter between methods - "
            "the owner's standing rule against those stays in force, so a station-day "
            "where only *some* arms exceed 100 m (e.g. CHPG/248, POVE/124) is left in "
            "untouched. On the data as of 2026-09-18 this removes exactly one "
            "station-day, URUM/DOY 365 (all eight solutions 5,880-5,990 m), recorded "
            "with every arm's error in this stage's own "
            "`excluded_solver_failures.csv`. `common_set_positioning.build()`'s "
            "`load_pretrained_elev` arm is read from a live experiment tree's raw "
            "per-day files rather than this stage's rebuilt summaries, so it applies "
            "this same exclusion list explicitly rather than inheriting it for free.",
        ],
        supersedes=[
            str(paths.positioning_result_dir("full_coverage")),
            str(
                paths.positioning_result_dir("comparison_3way") / "multiday_summary.csv"
            ),
        ],
    ),
    Stage(
        # Moved here, after positioning_coverage, 2026-08-28: WEIGHTING_RUN now points
        # at that stage's own multiday_summary_all_weightings.csv rather than the
        # frozen 20260216_2052 snapshot, so this stage must run after it -
        # check_inputs_are_produced_or_external enforces that ordering. Previously sat
        # ahead of positioning_coverage, with no dependency on it.
        "weighting_ablation",
        f"-m stec.analysis.weighting_ablation --output-dir {WEIGHTING_ABLATION_DIR}",
        "R2.5",
        "elevation against predicted-uncertainty weighting, paired station-days",
        inputs=[WEIGHTING_RUN, WEIGHTING_ABLATION_FIXED_VARIANCE_DIR],
        outputs=[
            str(WEIGHTING_ABLATION_DIR),
            str(WEIGHTING_ABLATION_DIR / "common_set.csv"),
        ],
        min_rows={str(WEIGHTING_ABLATION_DIR / "common_set.csv"): 4},
        checks=[weighting_ablation_common_set_has_all_four_methods],
        # Declared 2026-09-15 (results_register.md, Consistency item C): common_set.csv
        # is the weighting-ablation table, but this stage carried canonical_for=None,
        # so the registry's one-owner-per-canonical_for check gave it no protection at
        # all - the same gap Tables 3/4 used to have before they were declared.
        # paired.csv/fixed_variance.csv are not part of it; only common_set.csv is.
        canonical_for="Table 8",
        caveats=[
            "common_set.csv (2026-09-14, owner instruction) restricts the "
            "elev-vs-iono comparison to positioning_distributions's own common set "
            "(coverage_common_station_days, N=10,674 - see that stage's caveats for "
            "the full history: N=10,387 pre-fix, N=10,674 after the GIM-arm fix, an "
            "overly conservative N=9,606 for a few hours after the ref_source fix's "
            "first, whole-file-granularity pass, restored to N=10,674 once that fix's "
            "per-station follow-up landed) instead "
            "of paired.csv's per-correction pairing (10,366/10,640/"
            "10,733 as of 2026-09-14, not re-derived here), "
            "so the manuscript's weighting-ablation table and the positioning-distribution table share one N. No "
            "10 m outlier exclusion is applied there, unlike paired.csv - but as of "
            "2026-09-18 common_set.csv no longer carries the one genuine PPPx solve "
            "failure this used to leave in regardless (URUM, DOY 365, ~5,988 m under "
            "Direct STEC, both weightings): that row is now excluded upstream, by "
            "positioning_coverage's solver-failure rule, independent of this stage's "
            "own outlier-filtering choice.",
            "common_set.csv's headline is gain_median_%, not gain_mean_% (2026-09-15, "
            "owner instruction - docs/revision/positioning_reporting.md's median-not-"
            "mean decision, 2026-08-28, had missed this table). Before 2026-09-18, the "
            "single PPPx solve failure above (URUM, DOY 365) moved the Direct STEC "
            "elevation-weighted mean by 25% on its own across three successive "
            "population sizes (2.283 m with it / 1.706 m without, as of 2026-09-14; "
            "then 2.249 m over N=10,674 after the GIM-arm fix; briefly 2.355 m over an "
            "artificially smaller N=9,606 mid-ref_source-fix). **2026-09-18: that row "
            "is now excluded upstream** by positioning_coverage's solver-failure rule, "
            "so it is gone from the population entirely rather than merely "
            "present-with-caveats - the current common set (N=10,673) reads a Direct "
            "STEC elevation-weighted mean of 1.688 m against a median of 0.913 m "
            "(common_set.csv). The mean remains the more outlier-sensitive statistic on "
            "whatever tail is left, which is why the median stays the headline. "
            "elev_mean/iono_mean/gain_mean_% are still written to the CSV, but only for "
            "the mean-vs-median sensitivity comparison that document argues from - they "
            "are not a second number for the manuscript to quote. common_set.csv also "
            "now carries a Pretrained Direct STEC row: Pretrained_STEC_elev/iono were "
            "already in multiday_summary_all_weightings.csv, this table just hadn't "
            "read them.",
        ],
    ),
    Stage(
        "storm_stratification",
        f"-m stec.analysis.storm_stratification --output-dir {STORM_STRATIFICATION_DIR}",
        "R2.7",
        "positioning accuracy on storm against quiet days",
        # WEIGHTING_RUN joined 2026-09-15: stratify() now restricts to
        # common_set_positioning.coverage_common_station_days(), which reads
        # multiday_summary_all_weightings.csv - undeclared before this pass, which
        # would have let that file change silently under this stage's skip decision,
        # the same class of bug WEIGHTING_RUN was added to weighting_ablation for.
        inputs=[POSITIONING, SWI, WEIGHTING_RUN],
        outputs=[str(STORM_STRATIFICATION_DIR)],
        caveats=[
            "Classifies a day as storm when its daily minimum Dst reaches -50 nT. This "
            "is a different, deliberately kept-separate threshold from the STEC-domain "
            "scenario_evaluation.py, which classifies individual hours as storm at "
            "Kp>=37 or Dst<=-33: the daily rule finds 39 storm days over the 242-day "
            "positioning test period against 102 under the per-observation rule applied "
            "at the day level. Do not port one rule's day count into the other's table.",
            "Headline changed from mean to median and restricted to the same 4-method "
            "x 2-weighting common set the positioning tables use (N=10,674 - see "
            "positioning_coverage's caveats for the full 10,387 -> 10,674 -> briefly "
            "9,606 -> 10,674 history; owner decision "
            "applied 2026-09-15 - see the module docstring). improvement_over_gim.csv's "
            "headline columns are now _median_% (matching the other positioning tables); the previous "
            "mean-only, per-method-population version is what response_to_reviewers.md "
            "and evidence_summary.md quoted as +25.4%/+19.6% (itself already a stale "
            "restatement of the published +31.9%/+26.3%) - all three numbers are "
            "superseded by this stage's current output, not by each other.",
        ],
    ),
    Stage(
        "storm_stratification_stec",
        "-m stec.analysis.storm_stratification_stec --output-dir "
        f"{STORM_STRATIFICATION_STEC_DIR}",
        "R1.4/R1.7",
        "STEC prediction accuracy on storm against quiet days - the STEC-domain "
        "counterpart to storm_stratification.py's positioning-domain split",
        inputs=[str(DAILY_METRICS_DIR), SWI],
        outputs=[
            str(STORM_STRATIFICATION_STEC_DIR),
            str(STORM_STRATIFICATION_STEC_DIR / "day_classification.csv"),
            str(STORM_STRATIFICATION_STEC_DIR / "by_regime.csv"),
            str(STORM_STRATIFICATION_STEC_DIR / "strongest_storm_days.csv"),
        ],
        min_rows={
            str(STORM_STRATIFICATION_STEC_DIR / "day_classification.csv"): 1,
            str(STORM_STRATIFICATION_STEC_DIR / "by_regime.csv"): 8,
        },
        checks=[
            storm_stratification_stec_by_regime_has_all_methods,
            storm_stratification_stec_by_regime_has_consistent_day_counts,
        ],
        canonical_for=None,
        caveats=[
            "Classifies a day as storm when its daily minimum Dst reaches -50 nT "
            "(STORM_DST_THRESHOLD_NT, imported from storm_stratification.py, not "
            "duplicated). This is a different, deliberately kept-separate threshold "
            "from the per-observation Kp>=37/Dst<=-33 rule in "
            "src/analysis/scenario_evaluation.py, which classifies individual hours "
            "rather than days and is dormant (evaluation.enable_scenarios defaults "
            "False). Do not port one rule's day count into the other's table.",
            "Every method within a dataset is stratified over the same day set (inner "
            "join on doy against the shared per_day.csv), so the four methods are "
            "compared over identical populations, not each method's own count.",
            "STEC-domain per-observation prediction accuracy, not positioning error - "
            "see storm_stratification.py (R2.7) for the positioning-domain "
            "counterpart, which uses the same daily Dst rule and threshold but a "
            "disjoint population (PPPx station-days, not test-set observations).",
            "strongest_storm_days.csv uses a separate, fixed -300 nT threshold "
            "(EXTREME_STORM_DST_THRESHOLD_NT) to surface the two great storms of the "
            "2024 test period individually, rather than folding them into the same "
            "storm bucket as a day that only just crossed -50 nT.",
        ],
    ),
    Stage(
        "positioning_activity",
        f"-m stec.analysis.positioning_activity --output-dir {POSITIONING_ACTIVITY_DIR}",
        "R1.7",
        "positioning error stratified by local ionospheric activity - the axis that "
        "replaces the recovered/original population split, which was measured to be a "
        "proxy for activity rather than a property of the recovery pipeline",
        inputs=[
            str(POSITIONING_COVERAGE_DIR / "multiday_summary.csv"),
            str(paths.GIM_IONEX_ROOT),
            str(paths.IGS_STATION_COORDINATES),
        ],
        outputs=[
            str(POSITIONING_ACTIVITY_DIR),
            str(POSITIONING_ACTIVITY_DIR / "activity_stratification.csv"),
            str(POSITIONING_ACTIVITY_DIR / "station_day_activity.csv"),
        ],
        min_rows={
            str(POSITIONING_ACTIVITY_DIR / "activity_stratification.csv"): 8,
            str(POSITIONING_ACTIVITY_DIR / "station_day_activity.csv"): 30_000,
        },
        canonical_for="positioning ionospheric-activity stratification",
        caveats=[
            "The activity proxy is GIM VTEC at the station from the IONEX maps, not the "
            "model's own predicted STEC (circular) and not the prediction store (absent "
            "on exactly the recovered station-days this axis exists to look at).",
            "'elevated' is relative to each station's own median VTEC, not a global "
            "threshold - otherwise every equatorial station would be permanently "
            "elevated and the stratification would just re-encode latitude.",
            "No station-day is excluded. Every percentile column is paired with an "
            "exceedance count; do not quote the median alone.",
            "iono weighting only, matching the positioning distribution and component tables.",
        ],
    ),
    Stage(
        "constellation_coverage",
        f"-m stec.analysis.constellation_coverage "
        f"--output-dir {CONSTELLATION_COVERAGE_DIR}",
        "R1.5",
        "how many satellites each correction source lets PPPx use, and the within-station "
        "cost of a station-day where only one constellation could be corrected",
        inputs=[str(POSITIONING_COVERAGE_DIR / "multiday_summary.csv")],
        outputs=[
            str(CONSTELLATION_COVERAGE_DIR),
            str(CONSTELLATION_COVERAGE_DIR / "population_summary.csv"),
            str(CONSTELLATION_COVERAGE_DIR / "per_station.csv"),
            str(CONSTELLATION_COVERAGE_DIR / "within_station_penalty.csv"),
        ],
        min_rows={
            str(CONSTELLATION_COVERAGE_DIR / "population_summary.csv"): 3,
            str(CONSTELLATION_COVERAGE_DIR / "per_station.csv"): 40,
            str(CONSTELLATION_COVERAGE_DIR / "within_station_penalty.csv"): 10,
        },
        canonical_for="constellation-coverage limitation",
        caveats=[
            "Descriptive only. single_constellation is never used as a filter - the "
            "paper reports the stratification and excludes nothing.",
            "The root cause is upstream and unfixed: CamaliotGnss reports "
            "'Constellations Used: GE' and lists the GPS satellites it processed, then "
            "writes zero GPS records into +SLANT/SOLUTION. Reproduced on BIK0/DOY 122 "
            "2026-09-15; ruled out by direct test: our observable selection, receiver "
            "DCB availability, and unpopulated observables. It is NOT the CAS DCB "
            "product's station coverage - an earlier claim, retracted.",
            "STEC_DB_estDCB covers all ten affected stations with both constellations "
            "but is not used: owner decision 2026-09-15, it is an older (Feb 2025) "
            "build that may carry superseded errors.",
            "within_station_penalty's median is threshold-dependent on min_days_each, "
            "the per-station qualifying day count: x1.40 (26 stations, 24/26 "
            "significant) at 15, x1.41 (23 stations, 21/23 significant) at the coded "
            "default of 20, x1.48 (22 stations, 20/22 significant) at 22 and 25, and "
            "x1.55 (19 stations, 17/19 significant) at 30. It drifts upward with the "
            "threshold because a stricter cut keeps only the stations with the most of "
            "both kinds of day, which are the heavily-affected ones - so the population "
            "shifts as the threshold moves, not just its size. The manuscript must "
            "quote the range and name the threshold, never a bare point estimate. An "
            "earlier exploratory script in this work reported x1.48 from a >=25-day "
            "threshold; that figure is superseded by this stage, which is the "
            "authority now.",
        ],
    ),
    Stage(
        "positioning_robustness",
        f"-m stec.analysis.positioning_robustness --output-dir {POSITIONING_ROBUSTNESS_DIR}",
        "R2.7b",
        "tail behaviour and convergence of the positioning solutions",
        # WEIGHTING_RUN joined 2026-09-15 for the same reason as storm_stratification's
        # identical addition just above: load() now restricts to
        # coverage_common_station_days(), which reads multiday_summary_all_weightings.csv.
        inputs=[POSITIONING, WEIGHTING_RUN],
        outputs=[str(POSITIONING_ROBUSTNESS_DIR)],
        caveats=[
            "Restricted to the same 4-method x 2-weighting common set the positioning tables use "
            "(N=10,674 - see positioning_coverage's caveats for the full "
            "10,387 -> 10,674 -> briefly 9,606 -> 10,674 history) and the 10 m "
            "outcome-based outlier exclusion dropped, matching "
            "the other positioning tables (owner decision applied 2026-09-15 - see the module docstring). "
            "tail_distribution.csv reports median first (headline); "
            "error_components.csv gained _median_m columns as the headline, alongside "
            "the pre-existing mean-based columns (now suffixed _mean_m) kept only for "
            "the sensitivity comparison.",
        ],
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
            "A different station-day population from the positioning-distribution table, by design: requiring both "
            "weightings costs the IGS GIM ~3,000 station-days. State the N of each table."
        ],
    ),
    Stage(
        # No longer the positioning-summary table (tab:pos_summary)'s canonical source - see SUPERSEDED_FOR_TABLE5_NOTE and the
        # positioning_distributions Stage below, which now carries canonical_for="Table
        # 5" instead. Kept declared (not deleted): overall.csv is the mean/10 m-exclusion
        # side of the mean-vs-median sensitivity docs/revision/positioning_reporting.md
        # Sec 1 reports, and by_regime.csv/by_weighting.csv still back R1.7/R1.5.
        "positioning_summary",
        f"-m stec.analysis.positioning_summary --output-dir {POSITIONING_SUMMARY_DIR}",
        "R1.7/R1.5 sensitivity (superseded mean-based table)",
        "superseded headline positioning table (mean, 10 m exclusion), four methods "
        "on iono weighting - kept for the mean/median sensitivity comparison, not as "
        "the positioning-distribution table's source any more",
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
        canonical_for=None,
        caveats=[SUPERSEDED_FOR_TABLE5_NOTE],
    ),
    Stage(
        "oracle_benchmark",
        f"-m stec.analysis.oracle_benchmark --output-dir {ORACLE_BENCHMARK_DIR}",
        "R1.8",
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
        # Declared 2026-09-16: nothing in the registry recorded that this stage
        # answers a reviewer comment at all, which is the drift canonical_for
        # exists to catch. R1.8 asks for exactly this - the reference STEC applied
        # directly as the correction, as an observation-derived upper bound.
        canonical_for="R1.8 observation-derived oracle PPP benchmark",
        caveats=[
            "NOT comparable with the positioning-distribution table on absolute values: "
            "it uses elev weighting, because the reference STEC carries only a "
            "placeholder sigma so iono would weight by a constant. Since 2026-09-16 it is "
            "otherwise matched to those tables - same four-method/both-weighting common "
            "set, no 10 m exclusion, median headline - so elev weighting is the ONLY "
            "remaining difference, which is exactly when it starts looking like a detail "
            "worth tidying away. It is permanent and physical.",
            "Read ratios to the floor within this table. Take absolute positioning "
            "numbers from the positioning-distribution table.",
            "Methodology matched to the positioning tables 2026-09-16 (owner "
            "instruction, reversing the earlier 'permanently incomparable' position). "
            "Dropping the 10 m filter used to let one genuine PPPx solve failure (URUM, "
            "DOY 365, ~5,989 m under elev weighting) into the oracle arm, inflating the "
            "MEAN floor roughly tenfold while the median floor stayed unchanged to three "
            "figures - which is why the median is the headline and ratio_to_oracle_mean "
            "must not be quoted. **2026-09-18: that row is now excluded upstream** by "
            "positioning_coverage's solver-failure rule (docs/revision/"
            "positioning_reporting.md), independent of this stage's own dropped 10 m "
            "filter, so it no longer reaches the oracle arm at all - the median stays "
            "the headline regardless, because the mean remains the more "
            "outlier-sensitive statistic on whatever heavy tail is left (current oracle "
            "floor: mean 0.142 m against median 0.070 m over N=8,509, summary.csv). "
            "The conclusion has survived every methodology change so far: "
            "median-based ratios for Direct STEC / IGS GIM / VTEC + Mapping read "
            "11.5x/14.1x/15.8x on the original 5,514-station-day population, "
            "11.6x/14.2x/15.9x after the common-set and filter fixes (N=5,442), and "
            "11.4x/14.7x/16.3x after the 2026-09-16 coverage recovery (N=8,223). Read the "
            "current values from summary.csv; the numbers above are dated snapshots kept "
            "to show the finding is robust, not numbers to quote.",
            "Station coverage: 54 of the common set's 55 stations appear. PARC is absent "
            "because it has no reference-STEC corrections generated at all - an upstream "
            "data-generation gap, not a solving gap. The 2026-09-16 recovery solved 4,354 "
            "of 4,493 missing station-days (96.9%), taking the paired population from "
            "5,442 to 8,223 and the top-15-station share from 60% to 41%, with per-station "
            "coverage median rising from ~113 to 188 days. station_median_check.csv's "
            "pooled-versus-per-station-median agreement tightened from 5-8% to 3.6-4.9%, "
            "so the headline is not an artifact of the well-covered stations. Eight "
            "stations remain under 20 station-days, contributing 0.8% between them, but "
            "seven of those are FULLY solved against their available corrections - their "
            "thinness comes from the common-set intersection being sparse for them across "
            "all four methods, which no further oracle solving can fix. Only GLSV is an "
            "oracle-side failure: PPPx SIGSEGVs on it (missing Galileo PRN16 antenna "
            "calibration, 'no E16'), costing 95 of the 139 unrecovered station-days.",
        ],
    ),
    Stage(
        # Deployment-ready figures for R1.8, requested by the owner 2026-09-16 to have
        # "nice plots ready if i use it later for the reviewer comment addressing" - held
        # outside the manuscript until that decision is made, so this is a companion to
        # oracle_benchmark above rather than a replacement for revision_figures.py's own
        # oracle_benchmark bar chart (plots/revision/positioning_2024/oracle_benchmark.png,
        # which still answers R1.8 at manuscript scale). Same pattern as
        # positioning_diagnostics_figures and positioning_distributions_figures: a
        # standalone plots/ tree, not wired into revision_figures.py or
        # manuscript_figures.py, so it cannot be mistaken for either.
        "oracle_benchmark_figures",
        f"-m stec.viz.oracle_benchmark --output_dir {ORACLE_BENCHMARK_FIGURES_DIR}",
        "R1.8",
        "log-scale box plot, paired-difference scatter, station-coverage bar chart and a "
        "generated table for oracle_benchmark's CSVs",
        inputs=[str(ORACLE_BENCHMARK_DIR)],
        outputs=[
            ORACLE_BENCHMARK_FIGURES_DIR,
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/boxplot_oracle_benchmark.csv",
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/paired_difference.csv",
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/station_coverage.csv",
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/oracle_table.md",
        ],
        min_rows={
            # Box geometry (7 stats + n_exceeding_5m + n_above_axis_cap) x 4 methods = 36,
            # plus every flier point beyond the Tukey fence - measured 2,795 fliers on the
            # 2026-09-16 population, 2,831 rows total. Floored below that so a smaller
            # future population (station-day pairing can only shrink, never silently grow
            # past what oracle_benchmark itself paired) does not fail this on its own.
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/boxplot_oracle_benchmark.csv": 2_000,
            # 2 baselines (GIM, VTEC) x N station-days, one row per pairing - exact, not a
            # floor with headroom, because paired_station_days.csv carries no NaNs (every
            # row already has all four methods by construction) - measured 16,446 = 2 x
            # 8,223 pre-2026-09-17, then 17,020 = 2 x 8,510 after that session's
            # positioning_coverage GIM-arm fix. The 2026-09-17 ref_source mean ->
            # ground_truth repair (verification/repair_overwritten_summaries.py) then
            # correctly *dropped* the ~1,928 station-days across VTEC_iono/
            # Pretrained_STEC_iono that could not be repaired at the source (blocked by
            # an unrelated, genuine per-station missing-SINEX row in the same summary
            # file - see positioning_coverage's own caveat) from the common set
            # entirely, rather than leaving them in with a wrong (too-small) error - the
            # paired population shrank to 14,908 = 2 x 7,454 as a direct, expected
            # consequence, not a regression. Floored below that with headroom.
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/paired_difference.csv": 14_000,
            # One row per station in the paired population - measured 54.
            f"{ORACLE_BENCHMARK_FIGURES_DIR}/station_coverage.csv": 40,
        },
        canonical_for=None,
        caveats=[
            "Held outside the manuscript pending the owner's decision on whether/how it "
            "enters the R1.8 reviewer response - not a manuscript or revision-response "
            "figure yet, and must not be treated as one. Lives exclusively in "
            "plots/oracle_benchmark/, never plots/manuscript/ or plots/revision/.",
            "Reads oracle_benchmark's CSVs, not the .pos files directly - must run after "
            "that stage, and is only as current as its output (inherits oracle_benchmark's "
            "own elev-weighting-only caveat unchanged).",
            "The box plot's outlier points are drawn (unlike Figure 13's, which are "
            "omitted on its linear axis) - see stec/viz/oracle_benchmark.py's module "
            "docstring for why a log axis changes that trade-off, and why Figure 13 "
            "itself must not be changed to match. The y-axis is cropped to "
            "[BOXPLOT_Y_MIN_M, BOXPLOT_Y_MAX_M] = [1e-2, 1e2] m (owner review, "
            "2026-09-16) so the boxes are not squeezed by the shared ~6,000 m PPPx "
            "solve-failure cluster; every point above the cap is still a real row and is "
            "still counted, per method, in the title and in boxplot_oracle_benchmark."
            "csv's n_above_axis_cap rows - never silently dropped.",
        ],
    ),
    Stage(
        # Landed 2026-08-28 (commit b844bd4) as an owner-requested look at the
        # coverage-recovery headline before deciding whether/how it becomes part of
        # the positioning-summary table (tab:pos_summary) or a new appendix - see the module docstring. Declared here so its 12
        # CSVs (and FINDINGS.md, the narrative write-up `main()` generates from those
        # same CSVs - see _format_findings_markdown - rather than a hand-maintained
        # document that can drift from them) get the same provenance record as every
        # other analysis, rather than rotting silently the way results_manifest itself
        # once did before it was declared.
        "positioning_diagnostics",
        f"-m stec.analysis.positioning_diagnostics --output-dir {POSITIONING_DIAGNOSTICS_DIR}",
        "-",
        "outlier-threshold sensitivity, per-station losses and a recovered-vs-original "
        "population split for the post-recovery-sweep positioning result",
        inputs=[POSITIONING, SWI, RECOVERED_STEC_DB],
        outputs=[
            str(POSITIONING_DIAGNOSTICS_DIR),
            str(POSITIONING_DIAGNOSTICS_DIR / "overall_summary.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "daily_timeseries.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "per_station_summary.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_threshold_counts.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_headline_sensitivity.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_by_station.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_by_day.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_concentration_summary.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "recovered_station_days.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "population_split_by_method.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "population_split_stec_vs_gim.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_counts_by_population.csv"),
            str(POSITIONING_DIAGNOSTICS_DIR / "FINDINGS.md"),
        ],
        min_rows={
            # overall_summary() is pm.summarise(...).reindex(METHOD_ORDER): always
            # exactly 4 rows, one per method, NaN-filled rather than dropped if a
            # method has no station-days - exact, not a floor, same guarantee
            # positioning_summary_overall_has_all_four_methods checks for the positioning-summary table (tab:pos_summary).
            str(POSITIONING_DIAGNOSTICS_DIR / "overall_summary.csv"): 4,
            # outlier_headline_sensitivity(): "none" + one row per
            # OUTLIER_THRESHOLDS_M entry (4) = 5, fixed by the threshold tuple, not
            # by the data - exact.
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_headline_sensitivity.csv"): 5,
            # outlier_concentration()'s concentration_summary: one row per
            # METHOD_ORDER entry (4), written even when a method has zero outlier
            # station-days - exact.
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_concentration_summary.csv"): 4,
            # population_split()'s stec_vs_gim: one row per population
            # ("original", "recovered") - exact.
            str(POSITIONING_DIAGNOSTICS_DIR / "population_split_stec_vs_gim.csv"): 2,
            # Everything below is data-dependent (which stations/days/thresholds
            # actually appear), so each floor sits comfortably under what is really
            # on disk today (measured 2026-08-28, multiday_results/analyses/
            # positioning_diagnostics/rebuilt/): daily_timeseries 968,
            # per_station_summary 57, outlier_threshold_counts 16,
            # outlier_by_station 70, outlier_by_day 206, recovered_station_days
            # 2,277, population_split_by_method 8, outlier_counts_by_population 32.
            str(POSITIONING_DIAGNOSTICS_DIR / "daily_timeseries.csv"): 700,
            str(POSITIONING_DIAGNOSTICS_DIR / "per_station_summary.csv"): 40,
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_threshold_counts.csv"): 12,
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_by_station.csv"): 50,
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_by_day.csv"): 150,
            str(POSITIONING_DIAGNOSTICS_DIR / "recovered_station_days.csv"): 1_500,
            str(POSITIONING_DIAGNOSTICS_DIR / "population_split_by_method.csv"): 4,
            str(POSITIONING_DIAGNOSTICS_DIR / "outlier_counts_by_population.csv"): 16,
        },
        canonical_for=None,
        caveats=[
            "Diagnostic output, not a manuscript or positioning-table source: built to look at "
            "the coverage-recovery result before deciding whether/how it becomes "
            "part of the positioning-summary table (tab:pos_summary) or a new appendix. Does not supersede positioning_summary "
            "(that table) or common_set_positioning, and nothing here should be quoted "
            "in their place.",
            "Deliberately reports four outlier-exclusion thresholds (5/10/20/50 m) "
            "plus 'no exclusion at all' side by side rather than picking one - "
            "outlier_headline_sensitivity.csv shows the project's own 10 m rule "
            "(stec.positioning.metrics.OUTLIER_3D_RMS_M) sits on a curve where the "
            "Direct-STEC-vs-GIM mean improvement ranges from about -1% (no exclusion) "
            "to +16% (5 m threshold), not a validated single choice.",
            "The recovered-vs-original population split (population_split_by_method."
            "csv, population_split_stec_vs_gim.csv) depends on data/recovered_stec_db/ "
            "membership: a station-day counts as 'recovered' only if that exact "
            "(doy, station) pair has a geometry-only file under that tree "
            "(build_recovered_day.py's output). If the tree is absent, "
            "load_recovered_station_days logs a warning and returns an empty frame, "
            "and every station-day then silently reads as 'original' rather than the "
            "stage failing - a run with zero 'recovered' rows must be checked against "
            "whether data/recovered_stec_db/ was actually present, not read as "
            "'nothing was recovered'.",
            "Restricted to the four iono-weighted PAPER_METHODS (positioning_summary."
            "py's labels), the same population the positioning-distribution table uses - elev weighting is out "
            "of scope here, same as positioning_summary itself.",
        ],
    ),
    Stage(
        # The companion figure family for positioning_diagnostics above: a standalone
        # diagnostic tree (plots/positioning_diagnostics/), deliberately not wired
        # into revision_figures.py or manuscript_figures.py (see the module
        # docstring) - needs its own Stage for the same reason diagnostic_figures
        # does for its own separate tree.
        "positioning_diagnostics_figures",
        "-m stec.viz.positioning_diagnostics",
        "-",
        "daily-timeseries, per-station-diff, outlier-sensitivity and "
        "population-split figures for positioning_diagnostics's CSVs",
        inputs=[str(POSITIONING_DIAGNOSTICS_DIR)],
        outputs=[
            "plots/positioning_diagnostics",
            "plots/positioning_diagnostics/positioning_2024/daily_timeseries.csv",
            "plots/positioning_diagnostics/positioning_2024/per_station_diff.csv",
            "plots/positioning_diagnostics/positioning_2024/"
            "outlier_threshold_sensitivity.csv",
            "plots/positioning_diagnostics/positioning_2024/"
            "outlier_pct_by_threshold.csv",
            "plots/positioning_diagnostics/positioning_2024/population_split.csv",
        ],
        min_rows={
            # fig_outlier_threshold_sensitivity reindexes over the fixed 5-entry
            # _THRESHOLD_ORDER - exact, not data-dependent.
            "plots/positioning_diagnostics/positioning_2024/"
            "outlier_threshold_sensitivity.csv": 5,
            # The rest are data-dependent, same population as the analysis stage;
            # floors sit below what is on disk today (measured 2026-08-28):
            # daily_timeseries 968, per_station_diff 57,
            # outlier_pct_by_threshold 16, population_split 16.
            "plots/positioning_diagnostics/positioning_2024/daily_timeseries.csv": 700,
            "plots/positioning_diagnostics/positioning_2024/per_station_diff.csv": 40,
            "plots/positioning_diagnostics/positioning_2024/"
            "outlier_pct_by_threshold.csv": 8,
            "plots/positioning_diagnostics/positioning_2024/population_split.csv": 8,
        },
        canonical_for=None,
        caveats=[
            "Same diagnostic scope as positioning_diagnostics above - not manuscript "
            "or revision-response figures, kept in its own "
            "plots/positioning_diagnostics/ tree rather than plots/revision/ or "
            "plots/manuscript/ so it is never mistaken for either.",
            "Reads positioning_diagnostics's CSVs, not the per-station-day table "
            "directly - must run after that stage, and is only as current as its "
            "output.",
        ],
    ),
    Stage(
        # The the positioning-summary table (tab:pos_summary) replacement itself (owner decision 2026-08-28,
        # docs/revision/positioning_reporting.md): median, IQR, p95/p99 and exceedance
        # rates at 5/10/20/50 m, over the full unfiltered population - no outcome-based
        # outlier exclusion. Was built the same day as positioning_diagnostics but left
        # undeclared while the owner decided what that table should say (see that module's
        # own docstring, quoted verbatim in positioning_reporting.md's line 11-17); now
        # that the decision is made, this is what canonical_for="Tables 6 and 7" must
        # point at. Must follow positioning_diagnostics (reads its recovered-station-days
        # cache) and positioning_coverage (reads POSITIONING and SWI).
        #
        # canonical_for widened from one table to two (now "Tables 6 and 7" after the dSTEC table was inserted as Table 5) 2026-09-15
        # (results_register.md, Consistency item C): common_set_component_medians.csv
        # is tab:pos_components, written by this same stage, but had
        # no owner of its own in the registry - it only inherited this stage's general
        # min_rows/checks by accident of being one of its outputs, the same gap the weighting-ablation table (tab:weighting_ablation)
        # had (see weighting_ablation's own canonical_for change just below). One stage
        # legitimately owning two manuscript tables is not a double-ownership violation
        # of the registry's one-owner rule - see "Tables 3 and 4" (daily_metrics) for
        # the precedent of a single combined string.
        "positioning_distributions",
        "-m stec.analysis.positioning_distributions "
        f"--output-dir {POSITIONING_DISTRIBUTIONS_DIR}",
        "Tables 6 and 7",
        "positioning error as distributions - median/IQR/p95/p99/exceedance, no "
        "outcome-based outlier filter, per method and by storm/quiet and "
        "original/recovered population",
        inputs=[POSITIONING, SWI, str(POSITIONING_DIAGNOSTICS_DIR), WEIGHTING_RUN],
        outputs=[
            str(POSITIONING_DISTRIBUTIONS_DIR),
            str(POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "overall_exceedance.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "overall_boxplot_stats.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "overall_boxplot_fliers.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "overall_cdf_points.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "regime_percentile_summary.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "regime_exceedance.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "regime_boxplot_stats.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "regime_boxplot_fliers.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "population_percentile_summary.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "population_exceedance.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "population_boxplot_stats.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "population_boxplot_fliers.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "TABLE5_NUMBERS.md"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_percentile_summary.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_exceedance.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_component_medians.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "TABLE5_COMMON_SET_NUMBERS.md"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_boxplot_stats.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_boxplot_fliers.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_cdf_points.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_regime_boxplot_stats.csv"),
            str(POSITIONING_DISTRIBUTIONS_DIR / "common_set_daily_rows.csv"),
        ],
        min_rows=_POSITIONING_DISTRIBUTIONS_MIN_ROWS,
        checks=[positioning_distributions_overall_has_all_four_methods],
        canonical_for="Tables 6 and 7",
        caveats=[
            "The overall/regime/population sections (Figures 12-15) are iono weighting "
            "only, read from POSITIONING alone. The common_set_* sections and "
            "TABLE5_COMMON_SET_NUMBERS.md additionally read WEIGHTING_RUN (both "
            "weightings) to define the population restriction, but still report iono "
            "weighting values within it - every output states 'iono' explicitly rather "
            "than leaving the weighting implicit.",
            "No outcome-based filter anywhere in this stage: every station-day in the "
            "coverage-repaired population counts. Where a figure needs a bounded axis "
            "to stay readable (stec.viz.positioning_distributions), only the *view* is "
            "clipped, never the data - the CSVs here always carry the true fliers.",
            "Reporting only the median would hide that Direct STEC produces more large "
            "errors than GIM does - a real, operationally important property of the "
            "method. Every percentile table here is paired with an exceedance table for "
            "exactly that reason; do not quote the median alone.",
            "common_set_* and TABLE5_COMMON_SET_NUMBERS.md (2026-09-14, owner "
            "instruction) restrict the positioning-distribution table and the new per-component table to the "
            "station-days solved by all four methods under both weighting schemes "
            "(coverage_common_station_days, N=10,674 - moved 10,387 -> 10,674 (GIM-arm "
            "fix) -> briefly 9,606 (ref_source fix's first, whole-file-granularity "
            "pass, over-conservative) -> back to 10,674 (that fix's per-station "
            "follow-up) - a coverage-only "
            "intersection, no 10 m outlier exclusion. weighting_ablation.py's "
            "common_set.csv shares this exact population so the two manuscript "
            "tables report one N. Larger than common_set_positioning's own N=10,467 "
            "(moved 10,186 -> 10,468 -> briefly 9,417 -> 10,467 the same way; same "
            "eight arms) because "
            "that stage still applies the 10 m outlier rule; the 207-row gap (was "
            "206, 189 at the low point, 201 before the GIM-arm fix) is exactly the "
            "station-days where at "
            "least one arm exceeds it - see coverage_common_station_days's own "
            "docstring for the verification.",
            "common_set_boxplot_stats.csv/common_set_boxplot_fliers.csv/"
            "common_set_cdf_points.csv/common_set_regime_boxplot_stats.csv/"
            "common_set_daily_rows.csv (2026-09-15, results_register.md consistency "
            "item A) extend the same common-set restriction to Figures 12-15 and the "
            "standalone storm/quiet figure, which used to read the full per-method "
            "population (10,717-10,853 pre-2026-09-17, 10,803-10,912 after the GIM-arm "
            "fix, briefly 9,802-10,836 after the ref_source fix's whole-file pass, now "
            "10,803-10,837 after that fix's per-station follow-up restored "
            "VTEC/Pretrained_STEC's own per-method row counts; see "
            "overall_percentile_summary.csv) while "
            "the other positioning tables already used the N=10,674 common set - the "
            "same positioning chapter reporting "
            "two N's. "
            "fig_population_split_boxplot is deliberately not moved: see this stage's "
            "module docstring section 6.",
            "The mean is preserved for comparison (percentile_summary's mean_m column, "
            "and positioning_summary's own superseded overall.csv) but is not the "
            "reported statistic: it is non-monotonic in the outlier-exclusion "
            "threshold and swings from negative to strongly positive depending on a "
            "threshold with no principled value - see "
            "docs/revision/positioning_reporting.md Sec 1.",
        ],
    ),
    Stage(
        "positioning_distributions_figures",
        "-m stec.viz.positioning_distributions "
        f"--output_dir {POSITIONING_DISTRIBUTIONS_FIGURES_DIR}",
        "Table 6 (tab:pos_summary)",
        "box/CDF/percentile-exceedance figures for positioning_distributions's CSVs",
        inputs=[str(POSITIONING_DISTRIBUTIONS_DIR)],
        outputs=[
            POSITIONING_DISTRIBUTIONS_FIGURES_DIR,
            f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
            "boxplot_3d_error.csv",
            f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
            "cdf_unfiltered.csv",
            f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
            "percentile_exceedance_table.csv",
            f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
            "population_split_boxplot.csv",
            f"{POSITIONING_DISTRIBUTIONS_FIGURES_DIR}/positioning_2024/"
            "storm_quiet_boxplot.csv",
        ],
        min_rows=_POSITIONING_DISTRIBUTIONS_FIGURES_MIN_ROWS,
        canonical_for=None,
        caveats=[
            "Reads positioning_distributions's CSVs, not the per-station-day table "
            "directly - must run after that stage, and is only as current as its "
            "output.",
            "Not a manuscript or revision-response figure set yet - lives in its own "
            "plots/positioning_distributions/ tree, exclusively (no other module "
            "writes there), the same convention positioning_diagnostics_figures uses.",
        ],
    ),
    Stage(
        # Owner-requested look at whether Direct STEC's advantage over GIM varies with
        # (geographic and geomagnetic) latitude, and whether the recovered/original
        # population split (positioning_diagnostics) is itself latitude-concentrated.
        # Not a manuscript deliverable of its own - claims no canonical_for - but its
        # attribution result (STEC accuracy predicts absolute error, not competitiveness
        # against GIM) is one of the three findings docs/revision/positioning_reporting.md
        # Sec 4 says must reach the manuscript discussion. Must follow
        # positioning_diagnostics (recovered-station-days cache) and positioning_coverage
        # (POSITIONING, coverage.csv).
        "positioning_geography",
        "-m stec.analysis.positioning_geography "
        f"--output-dir {POSITIONING_GEOGRAPHY_DIR}",
        "-",
        "positioning error against geographic and geomagnetic station latitude, and "
        "recovered-population concentration by latitude",
        inputs=[
            POSITIONING,
            str(POSITIONING_COVERAGE_DIR),
            str(POSITIONING_DIAGNOSTICS_DIR),
        ],
        # Only this module's own files, never POSITIONING_GEOGRAPHY_DIR as a whole -
        # positioning_quality_gate.py and positioning_model_attribution.py both
        # deliberately reuse this directory as their own DEFAULT_OUTPUT_DIR (see that
        # constant's own comment), and neither is a declared stage, so claiming the
        # directory here would make this stage's digest depend on files it never writes.
        outputs=[
            str(POSITIONING_GEOGRAPHY_DIR / "station_map_direct_stec.csv"),
            str(POSITIONING_GEOGRAPHY_DIR / "station_map_diff.csv"),
            str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_geographic.csv"),
            str(POSITIONING_GEOGRAPHY_DIR / "latitude_stratification_geomagnetic.csv"),
            str(
                POSITIONING_GEOGRAPHY_DIR
                / "latitude_stratification_diff_geographic.csv"
            ),
            str(
                POSITIONING_GEOGRAPHY_DIR
                / "latitude_stratification_diff_geomagnetic.csv"
            ),
            str(
                POSITIONING_GEOGRAPHY_DIR
                / "population_concentration_by_latitude_geographic.csv"
            ),
            str(
                POSITIONING_GEOGRAPHY_DIR
                / "population_concentration_by_latitude_geomagnetic.csv"
            ),
            str(POSITIONING_GEOGRAPHY_DIR / "population_latitude_diff_geographic.csv"),
            str(POSITIONING_GEOGRAPHY_DIR / "population_latitude_diff_geomagnetic.csv"),
        ],
        min_rows=_POSITIONING_GEOGRAPHY_MIN_ROWS,
        canonical_for=None,
        caveats=[
            "iono weighting only, same population as positioning_distributions/"
            "positioning_diagnostics - the elev arm is out of scope here.",
            "Station coordinates come from IGSNetwork.csv (metres-level accuracy), not "
            "the SINEX ground truth the positioning error itself is measured against - "
            "deliberate, since reading per-day SINEX products would mean walking a "
            "tree an actively-running positioning chain can be writing to concurrently, "
            "for a precision gain that does not matter at this module's tens-of-degrees "
            "bin widths.",
            "Geomagnetic latitude is computed once per station at a fixed 2024-07-01 "
            "reference epoch, not per observation - a station's SM latitude is "
            "time-invariant to <=0.01 deg (module docstring, "
            "tests/analysis/test_positioning_geography.py), so this is a station "
            "property, not an approximation that needs revisiting.",
            "Shares its output directory with two further, undeclared modules "
            "(positioning_quality_gate.py, positioning_model_attribution.py) - only "
            "the files this stage's own command writes are declared as outputs above.",
        ],
    ),
    Stage(
        "positioning_geography_figures",
        f"-m stec.viz.positioning_geography --output_dir {POSITIONING_GEOGRAPHY_FIGURES_DIR}",
        "-",
        "station maps and latitude-stratification figures for positioning_geography's "
        "CSVs",
        inputs=[str(POSITIONING_GEOGRAPHY_DIR / "station_map_direct_stec.csv")],
        # Narrowed to positioning_2024/, not plots/positioning_geography/ as a whole -
        # that top-level tree also holds plots/positioning_geography/quality_gate/ and
        # plots/positioning_geography/model_accuracy_vs_positioning/, written by the
        # viz counterparts of the two undeclared modules above; positioning_2024/ is
        # this module's own, exclusively.
        outputs=[
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "station_map_direct_stec.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/station_map_diff.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "latitude_stratification_geographic.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "latitude_stratification_geomagnetic.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "population_by_latitude_geographic.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "population_by_latitude_geomagnetic.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "population_latitude_cross_geographic.csv",
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024/"
            "population_latitude_cross_geomagnetic.csv",
        ],
        min_rows=_POSITIONING_GEOGRAPHY_FIGURES_MIN_ROWS,
        canonical_for=None,
        caveats=[
            "Reads positioning_geography's CSVs, not the per-station-day table "
            "directly - must run after that stage, and is only as current as its "
            "output.",
            "Writes only plots/positioning_geography/positioning_2024/ - the "
            "quality_gate/ and model_accuracy_vs_positioning/ subtrees under the same "
            "parent belong to two further, undeclared viz modules.",
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
        # Figures 5-8 were the only manuscript figures with no independent check: their
        # correctness rested on Gate F agreeing with the src/ predecessor, which CLAUDE.md
        # is explicit is a weaker claim than correctness, since a refactor preserves the
        # bug it ports. This re-derives the same binned residuals straight from the store
        # with its own binning code, deliberately NOT reading pretrained_test_diagnostics'
        # cache - reading that would only prove the plotting is faithful.
        "pretrained_residuals_from_store",
        "-m stec.analysis.pretrained_residuals_from_store "
        f"--output-dir {PRETRAINED_RESIDUALS_FROM_STORE_DIR}",
        "Figures 5-8 independent verification (results_register.md Item I)",
        "per-bin MAE/RMSE by elevation, latitude, local time and year-month, streamed "
        "from predictions/pretrained_stec/own rather than from the figures' own cache",
        inputs=[STORE_PRETRAINED],
        outputs=[
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_elev.csv"),
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_lat.csv"),
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_localtime.csv"),
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_year_month.csv"),
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "manifest.csv"),
        ],
        # Exact bin counts, not floors: the bin edges are fixed by the figures' own
        # definitions, so a short file means a bin silently vanished.
        min_rows={
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_elev.csv"): 17,
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_lat.csv"): 18,
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_localtime.csv"): 24,
            str(PRETRAINED_RESIDUALS_FROM_STORE_DIR / "residuals_year_month.csv"): 12,
        },
        canonical_for=None,  # verification support, not a manuscript deliverable
        caveats=[
            "Verification only. These CSVs are the independent side of a comparison - "
            "the manuscript's numbers come from pretrained_test_diagnostics and the "
            "figures built on it, never from here. Quoting this stage's output as a "
            "result would defeat the purpose of computing it separately.",
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
        "differential STEC (gradient-only) RMSE vs IGS GIM, VTEC + Mapping and "
        "Pretrained - cancels per-arc DCB/levelling offsets by construction, isolating "
        "the comparability concern from the model's own accuracy",
        inputs=[STORE_OWN],
        outputs=[
            str(DSTEC_EVALUATION_DIR),
            str(DSTEC_EVALUATION_DIR / "pass_statistics.csv"),
            str(DSTEC_EVALUATION_DIR / "summary.csv"),
        ],
        # Keyed on the per-arc CSV, not the directory: a tree digest carries files/size/
        # mtime but no row count, so a min_rows on a directory can never be satisfied.
        # 500,000 is a floor, not the expected count (672,542 on the full 242-day store) -
        # comfortably above what an accidental partial run (e.g. the old 18-day default,
        # ~51,547) would produce, comfortably below normal day-to-day variation.
        # summary.csv holds 5 common rows + 6 for the model + 6 per baseline = 29 with
        # all three baselines; 20 is a floor that survives one baseline being absent
        # from a store partition but fails on the old GIM-only output.
        min_rows={
            str(DSTEC_EVALUATION_DIR / "pass_statistics.csv"): 500_000,
            str(DSTEC_EVALUATION_DIR / "summary.csv"): 20,
        },
        canonical_for="dSTEC (differential STEC) RMSE vs IGS GIM, VTEC + Mapping and "
        "Pretrained, R1.3",
        caveats=[
            "Tests the TEC gradient along a pass, not the absolute level - a low dSTEC "
            "error is evidence the model gets the pass *shape* right, not evidence about "
            "the absolute calibration Tables 3/4 report. Read model_abs_rmse_pooled/ "
            "gim_abs_rmse_pooled alongside the dSTEC numbers, never as a substitute.",
            "Runs on finetuned_stec/own (--model-variant/--dataset default). The "
            "Madrigal arm is its own declared stage, dstec_evaluation_madrigal, with a "
            "different arc rule (time gaps, no slipc) and truth source (code-derived, "
            "no gfphase) - read its own caveats rather than assuming this stage's apply.",
            "Covers all four methods as of 2026-09-15 (COMPARISON_METHODS): Direct "
            "STEC keeps the model_* prefix, IGS GIM gim_*, VTEC + Mapping vtec_*, "
            "Pretrained pretrained_*. The pre-existing model_*/gim_* numbers were "
            "verified byte-identical across that change on both datasets.",
            "The reporting unit is the arc and the headline statistic is the median "
            "across arcs; mean-of-arcs and pooled (observation-weighted) are kept "
            "beside it because they answer different questions and the improvement "
            "over IGS GIM differs materially between them (38.6% / 30.2% / 22.3% on "
            "own, 2026-09-15). No minimum arc length is applied.",
        ],
    ),
    Stage(
        # Its own stage rather than a parameter of dstec_evaluation: the registry's
        # one-owner-per-output rule means two configurations of one module writing two
        # directories have to be two stages, and the Madrigal arm has different inputs
        # (STORE_MADRIGAL), a different arc rule (time gaps, no slipc) and a different
        # truth source (code-derived, no gfphase) - all of which its caveats must state
        # separately from the own-dataset arm's.
        "dstec_evaluation_madrigal",
        f"-m stec.analysis.dstec_evaluation --dataset madrigal "
        f"--output-dir {DSTEC_EVALUATION_MADRIGAL_DIR}",
        "R1.3",
        "differential STEC against the Madrigal reference - the panel that the "
        "per-station reference offset cannot affect, because a constant per-arc offset "
        "cancels by construction rather than by being estimated and subtracted",
        inputs=[STORE_MADRIGAL],
        outputs=[
            str(DSTEC_EVALUATION_MADRIGAL_DIR),
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "pass_statistics.csv"),
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv"),
        ],
        min_rows={
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "pass_statistics.csv"): 700_000,
            str(DSTEC_EVALUATION_MADRIGAL_DIR / "summary.csv"): 20,
        },
        checks=[dstec_manuscript_table_matches_csv],
        supersedes=[str(DSTEC_EVALUATION_DIR / "finetuned_stec_madrigal")],
        canonical_for="Madrigal dSTEC panel (Table 4 companion)",
        caveats=[
            "Both fallbacks are active here and both are weaker than the own-dataset "
            "arm's: Madrigal has no cycle-slip counter, so arcs are inferred from a "
            "30-minute observation gap rather than read off a slip flag, and no "
            "gfphase, so the truth series is the noisier code-derived true_stec. "
            "arc_method/truth_source in the output record which was used. The two "
            "arms' arc counts are therefore not comparable and this is not a "
            "like-for-like comparison with the own-dataset dSTEC numbers.",
            "dSTEC removes an *additive* per-arc offset, not a multiplicative scale. "
            "All four products read systematically higher than Madrigal by an amount "
            "that grows toward the geomagnetic equator (see "
            "madrigal_method_offset_comparison), so this panel is much less "
            "contaminated than the absolute comparison, not free of the reference "
            "difference.",
            "The abs_rmse columns here are computed on the elevation-masked subset "
            "(205M of 449M observations), not Table 4's population. They are for "
            "reading beside the dSTEC columns only and must never be quoted as "
            "Table 4.",
        ],
    ),
    # Last: reads the metric CSVs every stage above writes, so it must follow all of them
    # in run order. Its declared `inputs`, though, name only the specific directories
    # `stec.viz.revision_figures.FIGURE_BUILDERS` actually opens (read from that module
    # directly, not guessed) - not the whole `multiday_results/` tree, which used to make
    # this stage (and manuscript_figures below) go stale every time an unrelated
    # hand-maintained CSV under that tree changed, e.g. revision_metrics_index.csv. See
    # RELATIVE_ERROR_METRICS_LEGACY_FLAT_CSV's own comment above for the one figure that
    # is confirmed to read a different, pre-rebuild file than its own analysis Stage's
    # declared output.
    Stage(
        "figures",
        "-m stec.viz.revision_figures",
        "all",
        "one PNG per revision figure, plus the _notitle manuscript variants",
        inputs=[
            RELATIVE_ERROR_METRICS_LEGACY_FLAT_CSV,
            RELATIVE_ERROR_METRICS_REBUILT_CANDIDATE,
            str(STORM_STRATIFICATION_DIR),
            str(WEIGHTING_ABLATION_DIR),
            str(ORACLE_BENCHMARK_DIR),
            str(HYPERPARAMETER_SEARCH_DIR),
            str(ACTIVITY_STRATIFICATION_DIR),
            str(STRATIFIED_COMPARISON_DIR),
            str(STRATIFIED_COMPARISON_PRETRAINED_DIR),
            str(UNCERTAINTY_ERROR_RELATION_DIR),
            str(IONEX_RMS_BENCHMARK_DIR),
            str(MADRIGAL_REFERENCE_OFFSET_DIR),
            str(DSTEC_EVALUATION_DIR),
            str(UNCERTAINTY_CALIBRATION_DIR),
            str(STATION_INDEPENDENCE_DIR),
            str(STATION_INDEPENDENCE_PRETRAINED_DIR),
            str(POSITIONING_ROBUSTNESS_DIR),
        ],
        outputs=["plots/revision"],
        caveats=[
            "The _notitle and _no_legend variants are the manuscript figures; the titled "
            "copies are working copies carrying a provenance footnote.",
            "Approach colours are fixed: blue Direct STEC, orange VTEC + Mapping, green "
            "IGS GIM + Mapping, purple Pretrained. An approach colour must never mean "
            "anything else.",
        ],
    ),
    # Also last, same reasoning as `figures` above: must run after every stage whose
    # output it reads, but its declared `inputs` name only those specific directories -
    # read from `stec.viz.manuscript_figures.FIGURE_BUILDERS` directly - not the whole
    # `multiday_results/` tree. Figures 1-2 read stec.config.paths.SPLIT_LISTS (not under
    # multiday_results/ at all, and not covered by the old whole-tree declaration
    # either); Figures 4-9 read pretrained_test_diagnostics's cache; Figure 10 reads
    # daily_metrics's per_day.csv; Figure 11 reads elevation_metrics_finetuned's
    # per_day_by_elevation.csv; Figures 12-15 read positioning_distributions's
    # common_set_daily_rows.csv and common_set_*.csv (not positioning_coverage's
    # multiday_summary.csv - that was true before the 2026-09-15 common-set restriction
    # this file's own caveats below already describe, but this comment had not been
    # updated to match). Figure 13's second, elevation-weighted oracle variant
    # (2026-09-17) is the one exception: it reads oracle_benchmark's own paired
    # population directly, plus positioning_coverage's multiday_summary_all_weightings.csv
    # for the Pretrained_STEC_elev arm that oracle_benchmark does not itself carry - see
    # `_build_positioning_oracle_figure`'s docstring.
    Stage(
        "manuscript_figures",
        "-m stec.viz.manuscript_figures",
        "all",
        "manuscript-numbered figures (dataset split, error/uncertainty, positioning)",
        inputs=[
            str(_rel(paths.SPLIT_LISTS)),
            str(PRETRAINED_TEST_DIAGNOSTICS_DIR),
            str(DAILY_METRICS_DIR),
            str(ELEVATION_METRICS_FINETUNED_DIR),
            str(POSITIONING_DISTRIBUTIONS_DIR),
            str(ORACLE_BENCHMARK_DIR),
            str(POSITIONING_COVERAGE_DIR),
        ],
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
            "pretrained_test_diagnostics (Figures 4-11) / positioning_distributions "
            "(Figures 12-15, common-set restricted since 2026-09-15 - see that stage's "
            "module docstring section 6) - a partial multiday_results tree produces a "
            "partial figure set with a logged warning per missing input, not a crash.",
            "Figure 13's oracle variant (boxplot_3d_error_with_oracle, added "
            "2026-09-17) additionally depends on oracle_benchmark and "
            "positioning_coverage - elevation weighting throughout, a different "
            "population from every other positioning figure in this stage (see "
            "stec.analysis.oracle_benchmark's module docstring for why) - and skips "
            "with a logged warning, not this stage's other figures, if either is "
            "missing.",
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
