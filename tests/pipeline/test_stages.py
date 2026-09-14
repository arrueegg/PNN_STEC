"""The declared registry itself, as distinct from the machinery that runs it.

These assert properties of the real stage list: that it is internally consistent, that
each paper deliverable has exactly one owner, and that the two evaluations which are not
what they look like carry the caveat that says so. A caveat lost in a refactor is how a
number ends up in a table it does not belong in.
"""

from __future__ import annotations

import csv
import importlib
import inspect
import re
from collections import Counter
from pathlib import Path

import pytest

from stec.analysis.daily_metrics import DATASET_LABELS, MODELS
from stec.analysis.positioning_summary import METHOD_ORDER
from stec.pipeline import registry
from stec.pipeline.stages import (
    DAILY_METRICS_DIR,
    ORACLE_EXPERIMENT_DIR,
    POSITIONING,
    POSITIONING_COVERAGE_DIR,
    POSITIONING_DIAGNOSTICS_DIR,
    POSITIONING_DISTRIBUTIONS_DIR,
    POSITIONING_DISTRIBUTIONS_FIGURES_DIR,
    POSITIONING_GEOGRAPHY_DIR,
    POSITIONING_GEOGRAPHY_FIGURES_DIR,
    POSITIONING_SUMMARY_DIR,
    RECOVERED_STEC_DB,
    STAGES,
    STORE_PRETRAINED,
    SWI,
    WEIGHTING_RUN,
    daily_metrics_summary_has_all_methods_and_datasets,
    daily_metrics_summary_has_consistent_day_counts,
    positioning_distributions_overall_has_all_four_methods,
    positioning_summary_overall_has_all_four_methods,
)


def stage(name: str):
    found = registry.by_name(STAGES).get(name)
    assert found is not None, f"no stage named {name}"
    return found


def position(name: str) -> int:
    return next(i for i, s in enumerate(STAGES) if s.name == name)


def test_registry_invariants_hold():
    registry.validate(STAGES)


def test_every_stage_names_the_comment_or_table_it_answers():
    for s in STAGES:
        assert s.answers, f"{s.name} does not say what it answers"
        assert s.description, f"{s.name} has no description"


def test_every_stage_declares_an_output():
    """A stage that produces nothing cannot be skipped, checked, or believed."""
    for s in STAGES:
        assert s.outputs, f"{s.name} declares no outputs"


def test_paper_deliverables_have_exactly_one_owner():
    owners = {s.canonical_for: s.name for s in STAGES if s.canonical_for}
    assert owners["Tables 3 and 4"] == "daily_metrics"
    # Moved off positioning_summary 2026-08-28's owner decision
    # (docs/revision/positioning_reporting.md): Table 5 reports distributions/medians
    # with no outcome-based filter, which positioning_distributions.py implements and
    # positioning_summary.py (the mean/10 m-exclusion methodology) does not.
    assert owners["Table 5"] == "positioning_distributions"
    # "Table A1" does not exist in either manuscript copy (5 tables, no lettered
    # appendix - Figures 14/15 are the only appendix content, numbered continuously).
    # Locking in its absence so a future edit cannot silently reintroduce the label.
    assert "Table A1" not in owners


def test_common_set_positioning_backs_no_manuscript_table():
    """It recomputes Table 5's methods on a different, smaller station-day population -
    the one solved under both weightings - to answer R1.5's reviewer-response comparison,
    not a printed manuscript table. Unlike positioning_distributions, it correctly claims
    no `canonical_for`, so it can never collide with that stage's "Table 5"."""
    assert stage("common_set_positioning").canonical_for is None
    assert stage("positioning_distributions").canonical_for == "Table 5"


def test_positioning_summary_no_longer_owns_table_5():
    """positioning_summary.py implements the pre-2026-08-28 mean/10 m-exclusion
    methodology the owner superseded (docs/revision/positioning_reporting.md) - it must
    not claim canonical_for="Table 5" any more, and its caveats must say what replaced
    it and why it is still declared (the mean/median sensitivity comparison), not just
    that it changed."""
    from stec.analysis.positioning_summary import SUPERSEDED_FOR_TABLE5_NOTE

    positioning_summary_stage = stage("positioning_summary")
    assert positioning_summary_stage.canonical_for is None
    assert SUPERSEDED_FOR_TABLE5_NOTE in positioning_summary_stage.caveats
    assert "positioning_distributions" in SUPERSEDED_FOR_TABLE5_NOTE
    assert "median" in SUPERSEDED_FOR_TABLE5_NOTE.lower()


def test_gim_repair_precedes_the_metrics_that_read_it():
    """The un-repaired baseline reversed the R1.4 conclusion, so the order is load-bearing."""
    assert position("repair_gim_baseline") < position("daily_metrics")
    assert position("daily_metrics") < position("activity_stratification")


def test_figures_and_manuscript_figures_run_last():
    """Last among the stages that actually feed them: `figures` and `manuscript_figures`
    both read the metric CSVs every analysis stage above writes to `multiday_results`
    (the latter also reads `daily_metrics` and `positioning_coverage` specifically), so
    both must follow every stage that produces one of those CSVs.

    `results_manifest` and `data_prep_smoke` are excluded, not exempted from a real
    invariant: neither reads `multiday_results` nor is read by either figure stage - the
    former is a standalone provenance index, the latter the data-preparation driver's
    self-contained smoke stage (`stec/data` has no analysis output either figure stage
    depends on). Their position is therefore not load-bearing, only a consequence of
    `stages.py`'s "append only" convention: new stages are added at the end of the
    registry, not inserted. `figures` and `manuscript_figures` are not ordered relative to
    each other - neither reads the other's output.
    """
    trailing_stages = {"results_manifest", "data_prep_smoke"}
    last_producer = max(
        position(s.name)
        for s in STAGES
        if s.name not in trailing_stages | {"figures", "manuscript_figures"}
    )
    assert position("figures") > last_producer
    assert position("manuscript_figures") > last_producer


def test_oracle_benchmark_states_it_is_not_comparable_with_table_5():
    caveats = " ".join(stage("oracle_benchmark").caveats).lower()
    assert "not comparable with table 5" in caveats
    assert "elev weighting" in caveats


def test_madrigal_results_are_never_standalone():
    caveats = " ".join(stage("madrigal_reference_offset").caveats).lower()
    assert "never standalone" in caveats
    assert "out-of-distribution" in caveats


def test_daily_metrics_distinguishes_mean_from_pooled_rmse():
    """Two different statistics with one name is exactly the ambiguity being removed."""
    caveats = " ".join(stage("daily_metrics").caveats).lower()
    assert "pooled" in caveats and "mean of per-day" in caveats


def test_daily_metrics_supersedes_the_unrecomputable_summary():
    assert any(
        "summary_statistics.csv" in path for path in stage("daily_metrics").supersedes
    )


def test_vtec_baseline_is_scored_as_a_laplace():
    caveats = " ".join(stage("uncertainty_calibration").caveats).lower()
    assert "laplace" in caveats


@pytest.mark.parametrize(
    "name",
    ["station_independence", "oracle_benchmark", "madrigal_reference_offset"],
)
def test_the_known_limited_results_carry_their_limitation(name):
    assert stage(name).caveats, f"{name} must state its limitation"


# --- min_rows on the canonical stages the independent audit flagged (F4/F5) ----------
#
# docs/revision/independent_audit.md found 0 of 34 stages declaring `checks`, 22 of 34
# declaring no `min_rows`, and 5 of the 10 `canonical_for` stages with existence-only
# assertions - `daily_metrics` (canonical for Tables 3 and 4) declared `min_rows={}`
# outright, the smoking gun: a stage could record success against a missing or empty
# store with nothing to catch it.


@pytest.mark.parametrize(
    "name",
    [
        "daily_metrics",
        "positioning_summary",
        "madrigal_reference_offset",
        "paper_tables",
        "results_manifest",
        "elevation_metrics_finetuned",
    ],
)
def test_canonical_stages_declare_nonempty_min_rows(name):
    floors = stage(name).min_rows
    assert floors, f"{name} declares no row-count floor"
    assert all(floor > 0 for floor in floors.values()), (
        f"{name} declares a non-positive floor: {floors}"
    )


def test_daily_metrics_floors_are_keyed_on_real_output_files():
    """`min_rows={}` used to be the only option because the stage's sole declared
    output was a directory, which carries no row count - the fix is declaring the CSVs
    daily_metrics.py actually writes as outputs in their own right."""
    floors = stage("daily_metrics").min_rows
    assert str(DAILY_METRICS_DIR / "summary.csv") in floors
    assert str(DAILY_METRICS_DIR / "per_day.csv") in floors


def test_positioning_summary_floor_is_keyed_on_overall_csv():
    floors = stage("positioning_summary").min_rows
    assert str(POSITIONING_SUMMARY_DIR / "overall.csv") in floors


# --- checks: content invariants min_rows cannot see -----------------------------------
#
# A row-count floor cannot tell a plausible-shaped CSV with the wrong content from a
# correct one - e.g. `reindex(METHOD_ORDER)` always writes exactly 4 rows for Table 5's
# overall.csv, NaN-filled rather than dropped when a method has no station-days. These
# pin the two `checks` callables added to catch that: fail on a synthetic wrong CSV,
# pass on a synthetic right one.


def test_daily_metrics_check_is_declared():
    assert (
        daily_metrics_summary_has_all_methods_and_datasets
        in stage("daily_metrics").checks
    )


def test_positioning_summary_check_is_declared():
    assert (
        positioning_summary_overall_has_all_four_methods
        in stage("positioning_summary").checks
    )


def _write_relative(tmp_path: Path, monkeypatch, relative: Path, content: str) -> None:
    """Check callables read a fixed repo-relative path directly (the same one they
    declare in `outputs`), so exercising them means chdir-ing into a scratch tree that
    mirrors the real layout - the same pattern `tests/pipeline/test_runner.py`'s
    `workspace` fixture uses for the runner itself."""
    monkeypatch.chdir(tmp_path)
    full = tmp_path / relative
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


def test_daily_metrics_check_passes_with_all_methods_and_datasets(
    tmp_path, monkeypatch
):
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(
        f"{dataset},{model}\n"
        for dataset in DATASET_LABELS.values()
        for model in MODELS.values()
    )
    _write_relative(tmp_path, monkeypatch, path, "dataset,Model\n" + rows)
    outputs = {str(path): {"present": True}}
    assert daily_metrics_summary_has_all_methods_and_datasets(outputs) is None


def test_daily_metrics_check_fails_when_a_dataset_is_missing(tmp_path, monkeypatch):
    """Simulates a store that silently lost its Madrigal partition while still writing
    a plausible-looking, non-empty summary.csv - exactly the failure a bare row-count
    floor cannot distinguish from four extra rows of a dataset already present."""
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(f"own_vtec_gim,{model}\n" for model in MODELS.values())
    _write_relative(tmp_path, monkeypatch, path, "dataset,Model\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_all_methods_and_datasets(outputs)
    assert violation is not None
    assert "madrigal_vtec_gim" in violation


def test_daily_metrics_check_fails_when_a_model_is_missing(tmp_path, monkeypatch):
    path = DAILY_METRICS_DIR / "summary.csv"
    kept_models = list(MODELS.values())[:-1]
    rows = "".join(
        f"{dataset},{model}\n"
        for dataset in DATASET_LABELS.values()
        for model in kept_models
    )
    _write_relative(tmp_path, monkeypatch, path, "dataset,Model\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_all_methods_and_datasets(outputs)
    assert violation is not None
    assert list(MODELS.values())[-1] in violation


def test_daily_metrics_check_reports_undeclared_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        daily_metrics_summary_has_all_methods_and_datasets({})
        == f"{DAILY_METRICS_DIR / 'summary.csv'} is not a declared output of this stage"
    )


def test_day_count_check_is_declared():
    assert (
        daily_metrics_summary_has_consistent_day_counts in stage("daily_metrics").checks
    )


def test_day_count_check_passes_with_matching_counts_per_dataset(tmp_path, monkeypatch):
    path = DAILY_METRICS_DIR / "summary.csv"
    # Mirrors the real 7-row shape: own carries all four models at 242 days each,
    # madrigal carries the three models it has, each at 238 days - two different
    # counts, but consistent *within* each dataset, which is what this check cares
    # about.
    own_rows = "".join(f"own_vtec_gim,{model},242\n" for model in MODELS.values())
    madrigal_models = [m for m in MODELS.values() if m != "Pretrained STEC"]
    madrigal_rows = "".join(
        f"madrigal_vtec_gim,{model},238\n" for model in madrigal_models
    )
    _write_relative(
        tmp_path,
        monkeypatch,
        path,
        "dataset,Model,Num_days\n" + own_rows + madrigal_rows,
    )
    outputs = {str(path): {"present": True}}
    assert daily_metrics_summary_has_consistent_day_counts(outputs) is None


def test_day_count_check_fails_when_one_model_is_shrunk(tmp_path, monkeypatch):
    """Reproduces the 2026-08-25 regression directly: doy 224/229/294 kept every
    other own-dataset column but lost `pretrained_stec_pred`, so 'Pretrained STEC'
    reported 239 days against every sibling model's 242 while still writing a
    perfectly plausible row - exactly what
    `daily_metrics_summary_has_all_methods_and_datasets` cannot see."""
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(
        f"own_vtec_gim,{model},{239 if model == 'Pretrained STEC' else 242}\n"
        for model in MODELS.values()
    )
    _write_relative(tmp_path, monkeypatch, path, "dataset,Model,Num_days\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_consistent_day_counts(outputs)
    assert violation is not None
    assert "Pretrained STEC" in violation
    assert "239" in violation and "242" in violation


def test_day_count_check_reports_undeclared_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert (
        daily_metrics_summary_has_consistent_day_counts({})
        == f"{DAILY_METRICS_DIR / 'summary.csv'} is not a declared output of this stage"
    )


def test_day_count_check_reports_missing_num_days_column(tmp_path, monkeypatch):
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(f"own_vtec_gim,{model}\n" for model in MODELS.values())
    _write_relative(tmp_path, monkeypatch, path, "dataset,Model\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_consistent_day_counts(outputs)
    assert violation is not None
    assert "Num_days" in violation


def test_daily_metrics_check_reports_renamed_column_instead_of_raising(
    tmp_path, monkeypatch
):
    """A CSV whose 'Model' column was renamed (or dropped) must produce a descriptive
    violation string, not a raw KeyError - runner.run_checks only catches
    AssertionFailed/CheckFailed, so an uncaught KeyError here would kill the whole
    `--keep-going` run instead of failing just this stage."""
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(f"{dataset},x\n" for dataset in DATASET_LABELS.values())
    _write_relative(tmp_path, monkeypatch, path, "dataset,ModelName\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_all_methods_and_datasets(outputs)
    assert violation is not None
    assert "Model" in violation


def test_daily_metrics_check_reports_missing_dataset_column_instead_of_raising(
    tmp_path, monkeypatch
):
    path = DAILY_METRICS_DIR / "summary.csv"
    rows = "".join(f"{model}\n" for model in MODELS.values())
    _write_relative(tmp_path, monkeypatch, path, "Model\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = daily_metrics_summary_has_all_methods_and_datasets(outputs)
    assert violation is not None
    assert "dataset" in violation


def test_daily_metrics_check_passes_on_the_known_pretrained_madrigal_gap_shape(
    tmp_path, monkeypatch
):
    """Pins the real-world shape: predictions/pretrained_stec/madrigal has never been
    built (predictions/pretrained_stec/madrigal/README.md), so the real
    pre_rebuild/summary.csv has 7 rows, not 8 - 'Pretrained STEC' x madrigal is
    legitimately absent. This must pass by design, not by accident, so it is pinned
    with a synthetic fixture mirroring the real shape rather than a read of the real
    file, which would only prove today's file happens to pass."""
    path = DAILY_METRICS_DIR / "summary.csv"
    own_rows = "".join(f"own_vtec_gim,{model}\n" for model in MODELS.values())
    madrigal_models = [m for m in MODELS.values() if m != "Pretrained STEC"]
    madrigal_rows = "".join(f"madrigal_vtec_gim,{model}\n" for model in madrigal_models)
    _write_relative(
        tmp_path, monkeypatch, path, "dataset,Model\n" + own_rows + madrigal_rows
    )
    outputs = {str(path): {"present": True}}
    assert daily_metrics_summary_has_all_methods_and_datasets(outputs) is None


def test_daily_metrics_check_does_not_catch_a_single_missing_cell(
    tmp_path, monkeypatch
):
    """Documents the check's known blind spot directly, independent of the specific
    pretrained/madrigal case above: it verifies marginal coverage (every model
    somewhere, every dataset somewhere), not the full 4x2 cross-product, so a single
    missing (model, dataset) cell - here an arbitrary one, not the known
    pretrained/madrigal gap - still passes. A future tightening to a strict
    cross-product check (see the check's own docstring for the condition) would need
    to update this test too, which is the point: the choice is pinned, not silently
    assumed."""
    path = DAILY_METRICS_DIR / "summary.csv"
    all_models = list(MODELS.values())
    own_rows = "".join(f"own_vtec_gim,{model}\n" for model in all_models)
    # Drop a cell that is NOT the known pretrained/madrigal gap, to show the blind
    # spot is general, not specific to that one documented case.
    madrigal_models = [m for m in all_models if m != "VTEC + Mapping"]
    madrigal_rows = "".join(f"madrigal_vtec_gim,{model}\n" for model in madrigal_models)
    _write_relative(
        tmp_path, monkeypatch, path, "dataset,Model\n" + own_rows + madrigal_rows
    )
    outputs = {str(path): {"present": True}}
    assert daily_metrics_summary_has_all_methods_and_datasets(outputs) is None


def test_positioning_summary_check_passes_with_all_four_methods_populated(
    tmp_path, monkeypatch
):
    path = POSITIONING_SUMMARY_DIR / "overall.csv"
    rows = "".join(f"{method},100\n" for method in METHOD_ORDER)
    _write_relative(tmp_path, monkeypatch, path, "Method,station_days\n" + rows)
    outputs = {str(path): {"present": True}}
    assert positioning_summary_overall_has_all_four_methods(outputs) is None


def test_positioning_summary_check_fails_when_reindex_leaves_a_method_empty(
    tmp_path, monkeypatch
):
    """`reindex(METHOD_ORDER)` guarantees the row exists for every method even when one
    has no station-days - it NaN-fills rather than drops, so `min_rows=4` alone cannot
    tell that apart from four real rows. A NaN written through `DataFrame.to_csv` reads
    back as an empty field, which is what this constructs directly."""
    path = POSITIONING_SUMMARY_DIR / "overall.csv"
    lines = [f"{method},100\n" for method in METHOD_ORDER[:-1]]
    lines.append(f"{METHOD_ORDER[-1]},\n")
    _write_relative(
        tmp_path, monkeypatch, path, "Method,station_days\n" + "".join(lines)
    )
    outputs = {str(path): {"present": True}}
    violation = positioning_summary_overall_has_all_four_methods(outputs)
    assert violation is not None
    assert METHOD_ORDER[-1] in violation


def test_positioning_summary_check_fails_when_a_method_row_is_absent(
    tmp_path, monkeypatch
):
    path = POSITIONING_SUMMARY_DIR / "overall.csv"
    rows = "".join(f"{method},100\n" for method in METHOD_ORDER[:-1])
    _write_relative(tmp_path, monkeypatch, path, "Method,station_days\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = positioning_summary_overall_has_all_four_methods(outputs)
    assert violation is not None
    assert METHOD_ORDER[-1] in violation


def test_positioning_summary_check_reports_renamed_column_instead_of_raising(
    tmp_path, monkeypatch
):
    """A CSV whose 'Method' column was renamed (or dropped) must produce a descriptive
    violation string, not a raw KeyError - same reasoning as the daily_metrics twin
    above."""
    path = POSITIONING_SUMMARY_DIR / "overall.csv"
    rows = "".join(f"{method},100\n" for method in METHOD_ORDER)
    _write_relative(tmp_path, monkeypatch, path, "approach,station_days\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = positioning_summary_overall_has_all_four_methods(outputs)
    assert violation is not None
    assert "Method" in violation


# --- epistemic_scale_diagnostic: the orphaned analysis, now a declared stage ----------


def test_epistemic_scale_diagnostic_is_declared_exactly_once():
    matches = [s for s in STAGES if s.name == "epistemic_scale_diagnostic"]
    assert len(matches) == 1
    assert matches[0].canonical_for == "R1.2 epistemic-scale diagnostic"


def test_epistemic_scale_diagnostic_canonical_for_does_not_collide():
    """registry.validate() (test_registry_invariants_hold) already enforces uniqueness
    globally; this pins the specific string so a future rename cannot silently drop the
    thing this test exists to protect."""
    others = [
        s.canonical_for
        for s in STAGES
        if s.canonical_for and s.name != "epistemic_scale_diagnostic"
    ]
    assert "R1.2 epistemic-scale diagnostic" not in others


def test_epistemic_scale_diagnostic_reads_both_pretrained_store_partitions():
    """Scores the paper's BayesianResNetSTEC against the fully-Bayesian
    ResNet_BNN_NLL reference - two different architectures in two different store
    partitions (CLAUDE.md's store-partition gotcha), not two readings of one."""
    inputs = stage("epistemic_scale_diagnostic").inputs
    assert "predictions/pretrained_stec/own" in inputs
    assert "predictions/pretrained_stec_resnet_bnn_nll/own" in inputs


def test_epistemic_scale_diagnostic_declares_its_diagnostic_not_retrain_caveat():
    caveats = " ".join(stage("epistemic_scale_diagnostic").caveats).lower()
    assert "not a retrain" in caveats


# --- oracle_benchmark: declared inputs must match what the module actually reads -----
#
# Diagnosed 2026-08-25: the stage declared inputs=[POSITIONING], but
# stec/analysis/oracle_benchmark.py never opens that file - it reads the oracle
# experiment tree directly (load_oracle) and the frozen weighting run
# (load_baselines), the same WEIGHTING_RUN dependency common_set_positioning declares.
# Because the declared fingerprint never changed while the real inputs
# (experiments/Reference_STEC_Oracle's SINEX symlinks) silently broke underneath it,
# `pipeline status` reported this stage up to date the entire time its output had
# shrunk from 242 days/~5,364 station-days to 76 days/1,810.


def _module_for(stage_obj) -> str:
    """The `-m <module>` target a stage's command runs, so a test can import the real
    module and check its source against what the stage declares as input."""
    parts = stage_obj.command.split()
    assert parts[0] == "-m", (
        f"{stage_obj.name}'s command does not start with -m: {parts}"
    )
    return parts[1]


def _module_source_mentions(module_name: str, needle: str) -> bool:
    module = importlib.import_module(module_name)
    return needle in inspect.getsource(module)


def test_oracle_benchmark_declares_the_inputs_it_reads():
    inputs = stage("oracle_benchmark").inputs
    assert WEIGHTING_RUN in inputs, (
        "oracle_benchmark.py pairs the oracle run against the frozen weighting "
        "summary (positioning_summary.DEFAULT_WEIGHTING_SUMMARY) - the same "
        "dependency common_set_positioning declares for the same file."
    )
    assert ORACLE_EXPERIMENT_DIR in inputs, (
        "oracle_benchmark.py reads Reference_STEC_Oracle's .pos solutions and SINEX "
        "directly (load_oracle) - this is the tree that silently lost 166/242 SINEX "
        "symlinks while undeclared here."
    )
    assert POSITIONING not in inputs, (
        "oracle_benchmark.py never opens positioning_coverage's multiday_summary.csv "
        "- declaring it here was the original defect: a fingerprint that can never "
        "move when the module's real inputs do."
    )

    # Confirm against the real module source, not just the stage's own comment: the
    # oracle tree is read directly, and the weighting run comes in through the shared
    # DEFAULT_WEIGHTING_SUMMARY constant (the literal "20260216_2052" path segment
    # lives in positioning_summary.py, not here - see that constant's own comment
    # about being reused by both common_set_positioning and oracle_benchmark).
    module = _module_for(stage("oracle_benchmark"))
    assert _module_source_mentions(module, "Reference_STEC_Oracle")
    assert _module_source_mentions(module, "DEFAULT_WEIGHTING_SUMMARY")


def test_oracle_benchmark_and_common_set_positioning_share_the_weighting_run():
    """The two stages read the same frozen weighting summary for the same reason
    (pairing against the baseline methods) - pinned so a future edit cannot drift one
    without the other."""
    assert WEIGHTING_RUN in stage("common_set_positioning").inputs
    assert WEIGHTING_RUN in stage("oracle_benchmark").inputs


# --- population/scope caveats on every stage reading a multi-year-capable partition --
#
# predictions/pretrained_stec/own (and its fully-Bayesian twin) spans 2014-2024, not
# just the 2024 the finetuned_stec/own four-method comparisons use.
# uncertainty_calibration_pretrained used to silently default to 2024 alone - 242 of
# 544 days, 44% of the partition - with nothing in coverage.csv/scores.csv or the
# stage's own caveats saying so. A reader holding only the output CSV could not tell
# which population it covered; these pin that every stage reading that partition now
# says so in its `caveats`, the one documentation field that reaches a
# `<output>.caveats.json` sidecar next to the artifact itself.


@pytest.mark.parametrize(
    "name",
    [
        "temporal_regime_split",
        "temporal_regime_activity_matched",
        "uncertainty_calibration_pretrained",
        "epistemic_scale_diagnostic",
        "pretrained_test_diagnostics",
        "diagnostic_figures",
    ],
)
def test_stages_reading_the_pretrained_partition_state_their_population(name):
    assert STORE_PRETRAINED in stage(name).inputs, (
        f"{name} is not declared to read {STORE_PRETRAINED} - update this "
        "parametrisation if that changed deliberately"
    )
    caveats = " ".join(stage(name).caveats).lower()
    assert caveats, f"{name} must state the population it covers"
    assert "2014-2024" in caveats, (
        f"{name}'s caveats must name the actual year range covered, not just assert "
        "that a scope exists"
    )
    assert "544" in caveats, f"{name}'s caveats must name the day count covered"


def test_epistemic_scale_diagnostic_caveat_explains_the_matched_population():
    """Distinct from the other five: this stage compares *two* partitions, so its
    caveat must explain the day-set match/restriction, not just name a year range."""
    caveats = " ".join(stage("epistemic_scale_diagnostic").caveats).lower()
    assert "day sets" in caveats or "day set" in caveats or "days" in caveats
    assert "warning" in caveats


# --- stratified_comparison: declared inputs must match what the module actually reads,
# the mirror of the oracle_benchmark defect above - there a real input was undeclared;
# here a non-input is declared. ---------------------------------------------------------


def test_stratified_comparison_does_not_declare_the_pretrained_partition_as_input():
    """The invoked command takes no --model-variant/--dataset override, so it only
    ever reads the finetuned_stec/own default (STORE_OWN) - confirmed against the real
    module source: 'Pretrained Direct STEC' reads the pretrained_stec_pred *column*,
    merged into STORE_OWN's own parquet files (the same arrangement daily_metrics.py's
    MODELS dict documents), never a separate read of predictions/pretrained_stec/own."""
    inputs = stage("stratified_comparison").inputs
    assert STORE_PRETRAINED not in inputs

    command = stage("stratified_comparison").command
    assert "--model-variant" not in command
    assert "--dataset" not in command

    module = _module_for(stage("stratified_comparison"))
    assert _module_source_mentions(module, '"pretrained_stec_pred"')


# --- positioning_coverage: the elev pair must carry the same assertion machinery as the
# iono outputs it sits beside, not just exist on disk unfingerprinted. --------------------


def test_positioning_coverage_declares_the_elev_outputs():
    """`positioning_coverage.py`'s `main()` writes `multiday_summary_elev.csv` and
    `coverage_elev.csv` unconditionally (weighting="both" is the default), but until
    this was fixed neither appeared in the stage's `outputs=[...]`, so a truncated or
    missing write there passed `check_assertions` silently - the same defect class as
    the SINEX/positioning_full_coverage incidents this file's other tests pin."""
    outputs = stage("positioning_coverage").outputs
    assert str(POSITIONING_COVERAGE_DIR / "multiday_summary_elev.csv") in outputs
    assert str(POSITIONING_COVERAGE_DIR / "coverage_elev.csv") in outputs


def test_positioning_coverage_elev_outputs_have_real_min_rows_floors():
    """Floors measured against the files actually on disk 2026-08-26
    (multiday_summary_elev.csv: 37,241 rows; coverage_elev.csv: 11,641 rows, matching
    docs/revision/work_queue.md's coverage table) - not copied blind from the iono
    sibling's 30,000. Each floor must sit strictly below what is really there, so a
    header-only or drastically truncated write still fails, while a same-order-of-
    magnitude future run does not."""
    min_rows = stage("positioning_coverage").min_rows
    summary_elev = str(POSITIONING_COVERAGE_DIR / "multiday_summary_elev.csv")
    coverage_elev = str(POSITIONING_COVERAGE_DIR / "coverage_elev.csv")

    assert 0 < min_rows[summary_elev] < 37_241
    assert 0 < min_rows[coverage_elev] < 11_641
    # Comfortably below actual, not merely nonzero - a floor that only barely clears a
    # truncated file is not much of a floor.
    assert min_rows[summary_elev] >= 25_000
    assert min_rows[coverage_elev] >= 8_000


# --- positioning_diagnostics: promoted from ad-hoc output to a declared stage
# (2026-08-28, commit b844bd4 landed stec/analysis/positioning_diagnostics.py +
# stec/viz/positioning_diagnostics.py - this pins the declaration added on top of it,
# not the analysis/viz modules themselves). Two stages, not one: the runner executes
# one `python -m` invocation per stage (stec/pipeline/runner.py), and the analysis
# (12 CSVs) and viz (5 figures + their plotted-value CSVs) are genuinely separate
# entry points, the same split `pretrained_test_diagnostics`/`manuscript_figures` and
# `diagnostic_test_observations`/`diagnostic_figures` use elsewhere in this file. -----


_POSITIONING_DIAGNOSTICS_CSVS = [
    "overall_summary.csv",
    "daily_timeseries.csv",
    "per_station_summary.csv",
    "outlier_threshold_counts.csv",
    "outlier_headline_sensitivity.csv",
    "outlier_by_station.csv",
    "outlier_by_day.csv",
    "outlier_concentration_summary.csv",
    "recovered_station_days.csv",
    "population_split_by_method.csv",
    "population_split_stec_vs_gim.csv",
    "outlier_counts_by_population.csv",
]

_POSITIONING_DIAGNOSTICS_FIGURE_CSVS = [
    "plots/positioning_diagnostics/positioning_2024/daily_timeseries.csv",
    "plots/positioning_diagnostics/positioning_2024/per_station_diff.csv",
    "plots/positioning_diagnostics/positioning_2024/outlier_threshold_sensitivity.csv",
    "plots/positioning_diagnostics/positioning_2024/outlier_pct_by_threshold.csv",
    "plots/positioning_diagnostics/positioning_2024/population_split.csv",
]


def test_positioning_diagnostics_is_declared_exactly_once_and_owns_no_deliverable():
    """Diagnostic output built to look at the coverage-recovery result before deciding
    whether/how it becomes part of Table 5 or a new appendix (see the module docstring)
    - it must not claim a manuscript deliverable while that decision is still open, the
    same reasoning common_set_positioning's canonical_for=None already carries above.
    FINDINGS.md (checked below) is one of this stage's own declared outputs, generated
    by main() from the same dataframes as its CSVs - not a separate hand-maintained
    document that could disagree with canonical_for on its own."""
    matches = [s for s in STAGES if s.name == "positioning_diagnostics"]
    assert len(matches) == 1
    assert matches[0].canonical_for is None

    figure_matches = [s for s in STAGES if s.name == "positioning_diagnostics_figures"]
    assert len(figure_matches) == 1
    assert figure_matches[0].canonical_for is None


def test_positioning_diagnostics_figures_run_after_positioning_diagnostics():
    assert position("positioning_diagnostics") < position(
        "positioning_diagnostics_figures"
    )
    assert position("positioning_diagnostics_figures") < position("figures")
    assert position("positioning_diagnostics_figures") < position("manuscript_figures")


def test_positioning_diagnostics_declares_the_inputs_it_reads():
    """load_positioning_table reads POSITIONING, attach_storm_flag reads SWI, and
    load_recovered_station_days reads RECOVERED_STEC_DB (data/recovered_stec_db/) -
    each declared at the granularity that actually changes (two files, one directory),
    not the whole multiday_results tree the way figures/manuscript_figures do."""
    inputs = stage("positioning_diagnostics").inputs
    assert POSITIONING in inputs
    assert SWI in inputs
    assert RECOVERED_STEC_DB in inputs

    module = _module_for(stage("positioning_diagnostics"))
    assert _module_source_mentions(module, "RECOVERED_STEC_DB_ROOT")
    assert _module_source_mentions(module, "canonical_positioning_summary")


def test_positioning_diagnostics_figures_reads_only_its_own_analysis_directory():
    """Narrowed to POSITIONING_DIAGNOSTICS_DIR, not the whole multiday_results tree -
    declaring the broad tree is exactly why figures/manuscript_figures go stale on
    every unrelated analysis's write, which is what motivated narrowing this one."""
    assert stage("positioning_diagnostics_figures").inputs == [
        str(POSITIONING_DIAGNOSTICS_DIR)
    ]


def test_positioning_diagnostics_states_it_is_diagnostic_not_a_table_5_source():
    caveats = " ".join(stage("positioning_diagnostics").caveats).lower()
    assert "diagnostic" in caveats
    assert "table 5" in caveats


def test_positioning_diagnostics_states_it_reports_several_outlier_thresholds():
    """It deliberately shows the 10 m rule (stec.positioning.metrics.OUTLIER_3D_RMS_M)
    as one point among several rather than picking a single threshold - the audit's own
    framing for why this stage's caveats must say so explicitly."""
    caveats = " ".join(stage("positioning_diagnostics").caveats).lower()
    assert "5 m" in caveats and "50 m" in caveats
    assert "10 m" in caveats


def test_positioning_diagnostics_states_the_recovered_stec_db_dependency():
    """The recovered-vs-original population split depends on data/recovered_stec_db/
    membership, and silently degrades to 'everything is original' if that tree is
    absent (load_recovered_station_days logs a warning, does not raise) - the caveat
    must say so, not just that a population split exists."""
    caveats = " ".join(stage("positioning_diagnostics").caveats).lower()
    assert "data/recovered_stec_db" in caveats
    assert "membership" in caveats


def test_positioning_diagnostics_declares_all_twelve_csv_outputs():
    """stec.analysis.positioning_diagnostics.main() writes exactly these 12 CSVs -
    each needs its own declared output (and its own min_rows floor, checked below) or a
    truncated write to any single one of them passes silently, the same defect class
    positioning_coverage's elev pair had before it carried min_rows of its own."""
    outputs = stage("positioning_diagnostics").outputs
    for name in _POSITIONING_DIAGNOSTICS_CSVS:
        assert str(POSITIONING_DIAGNOSTICS_DIR / name) in outputs, name


def test_positioning_diagnostics_every_csv_has_a_positive_min_rows_floor():
    floors = stage("positioning_diagnostics").min_rows
    for name in _POSITIONING_DIAGNOSTICS_CSVS:
        key = str(POSITIONING_DIAGNOSTICS_DIR / name)
        assert key in floors, f"{name} has no min_rows floor"
        assert floors[key] > 0


def test_positioning_diagnostics_declares_findings_markdown_as_a_generated_output():
    """FINDINGS.md is main()'s 13th write (_format_findings_markdown), not a 13th CSV -
    declared as an output so a truncated/missing write is caught the same way a missing
    CSV would be, but carries no min_rows floor (markdown, not row-shaped data), the
    same treatment positioning_distributions gives TABLE5_NUMBERS.md."""
    outputs = stage("positioning_diagnostics").outputs
    key = str(POSITIONING_DIAGNOSTICS_DIR / "FINDINGS.md")
    assert key in outputs
    assert key not in stage("positioning_diagnostics").min_rows


def test_positioning_diagnostics_structurally_exact_floors_match_the_guarantee():
    """Four of the twelve CSVs are exact row counts by construction, not measured
    floors: overall_summary() reindexes over METHOD_ORDER (4),
    outlier_headline_sensitivity() has one row per threshold plus 'none' (5),
    outlier_concentration()'s summary has one row per METHOD_ORDER entry (4), and
    population_split()'s stec_vs_gim has one row per population (2) - the same kind of
    reindex guarantee positioning_summary_overall_has_all_four_methods checks for
    Table 5's overall.csv."""
    floors = stage("positioning_diagnostics").min_rows
    assert floors[str(POSITIONING_DIAGNOSTICS_DIR / "overall_summary.csv")] == 4
    assert (
        floors[str(POSITIONING_DIAGNOSTICS_DIR / "outlier_headline_sensitivity.csv")]
        == 5
    )
    assert (
        floors[str(POSITIONING_DIAGNOSTICS_DIR / "outlier_concentration_summary.csv")]
        == 4
    )
    assert (
        floors[str(POSITIONING_DIAGNOSTICS_DIR / "population_split_stec_vs_gim.csv")]
        == 2
    )


def test_positioning_diagnostics_data_dependent_floors_sit_below_real_output():
    """Measured against the files actually on disk 2026-08-28
    (multiday_results/analyses/positioning_diagnostics/rebuilt/): daily_timeseries 968,
    per_station_summary 57, outlier_threshold_counts 16, outlier_by_station 70,
    outlier_by_day 206, recovered_station_days 2,277, population_split_by_method 8,
    outlier_counts_by_population 32. Each floor must sit strictly below what is really
    there, so a header-only or drastically truncated write still fails, while a same-
    order-of-magnitude future run does not."""
    floors = stage("positioning_diagnostics").min_rows
    real_counts = {
        "daily_timeseries.csv": 968,
        "per_station_summary.csv": 57,
        "outlier_threshold_counts.csv": 16,
        "outlier_by_station.csv": 70,
        "outlier_by_day.csv": 206,
        "recovered_station_days.csv": 2_277,
        "population_split_by_method.csv": 8,
        "outlier_counts_by_population.csv": 32,
    }
    for name, real_count in real_counts.items():
        floor = floors[str(POSITIONING_DIAGNOSTICS_DIR / name)]
        assert 0 < floor < real_count, (name, floor, real_count)


def test_positioning_diagnostics_figures_declares_the_five_plotted_csvs():
    """Each figure's `_save` writes its plotted numbers to a CSV alongside the PNGs -
    PNGs carry no row count, so these five are what this stage can actually assert
    shape on, the same reasoning min_rows is keyed on CSVs, never directories, wherever
    that is possible in this file."""
    outputs = stage("positioning_diagnostics_figures").outputs
    for path in _POSITIONING_DIAGNOSTICS_FIGURE_CSVS:
        assert path in outputs, path


_COVERAGE_CSV = POSITIONING_COVERAGE_DIR / "coverage.csv"


@pytest.mark.skipif(
    not _COVERAGE_CSV.exists(),
    reason="live positioning_coverage coverage.csv not present on this host",
)
def test_positioning_coverage_caveat_counts_match_live_coverage_csv():
    """The 'Post station-recovery-sweep' caveat's four counts have already drifted
    twice without any test catching either move (8,003/2,311/510 of 10,824 ->
    8,195/1,591/1,067 of 10,853 -> 10,598/26/229 of 10,853, corrected 2026-08-28) -
    the same class of drift test_docstring_station_count_matches_live_per_station_csv
    (tests/analysis/test_station_independence.py) guards for a docstring. Parses the
    numbers straight out of the caveat sentence rather than duplicating them as a
    second hardcoded set here, so the next drift fails this test instead of only being
    caught by hand again."""
    caveats = " ".join(stage("positioning_coverage").caveats)
    match = re.search(
        r"iono weighting: ([\d,]+) / ([\d,]+) / ([\d,]+) of ([\d,]+) station-days "
        r"solved by all methods",
        caveats,
    )
    assert match, (
        "positioning_coverage's caveat no longer states the 'iono weighting: "
        "X / Y / Z of W station-days solved by all methods' sentence - update this "
        "regex if the wording changed deliberately"
    )
    solved_all, all_missing, some_missing, total = (
        int(g.replace(",", "")) for g in match.groups()
    )

    with _COVERAGE_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    causes = Counter(row["cause"] for row in rows)

    assert len(rows) == total, (len(rows), total)
    assert causes["solved by all methods"] == solved_all
    assert causes["all ML methods missing (station absent from STEC DB)"] == all_missing
    assert causes["some ML methods missing (per-method failure)"] == some_missing


def test_positioning_diagnostics_figures_min_rows_floors_sit_below_real_output():
    """outlier_threshold_sensitivity.csv is exact (fig_outlier_threshold_sensitivity
    reindexes over the fixed 5-entry _THRESHOLD_ORDER); the rest are data-dependent,
    measured against plots/positioning_diagnostics/positioning_2024/ 2026-08-28:
    daily_timeseries 968, per_station_diff 57, outlier_pct_by_threshold 16,
    population_split 16."""
    floors = stage("positioning_diagnostics_figures").min_rows
    exact_key = (
        "plots/positioning_diagnostics/positioning_2024/"
        "outlier_threshold_sensitivity.csv"
    )
    assert floors[exact_key] == 5

    real_counts = {
        "plots/positioning_diagnostics/positioning_2024/daily_timeseries.csv": 968,
        "plots/positioning_diagnostics/positioning_2024/per_station_diff.csv": 57,
        "plots/positioning_diagnostics/positioning_2024/"
        "outlier_pct_by_threshold.csv": 16,
        "plots/positioning_diagnostics/positioning_2024/population_split.csv": 16,
    }
    for key, real_count in real_counts.items():
        assert 0 < floors[key] < real_count, (key, floors[key], real_count)


# --- positioning_distributions / positioning_geography: promoted from ad-hoc,
# deliberately-undeclared output to declared stages (2026-09-14). Both were built
# 2026-08-28 alongside positioning_diagnostics and left undeclared "while the owner
# looked at it before deciding what Table 5 should say" (positioning_distributions.py's
# own docstring, quoted in positioning_reporting.md) - once that decision landed
# (Table 5 = distributions/medians, no outcome filter), leaving the module that
# implements it undeclared meant `python -m stec.pipeline status` could report the
# pipeline current while the decided methodology sat unrun against a population that had
# since moved. Four stages, not two: analysis and viz are separate `python -m`
# invocations for each module, the same split positioning_diagnostics uses. -----------


_POSITIONING_DISTRIBUTIONS_CSVS = [
    "overall_percentile_summary.csv",
    "overall_exceedance.csv",
    "overall_boxplot_stats.csv",
    "overall_boxplot_fliers.csv",
    "overall_cdf_points.csv",
    "regime_percentile_summary.csv",
    "regime_exceedance.csv",
    "regime_boxplot_stats.csv",
    "regime_boxplot_fliers.csv",
    "population_percentile_summary.csv",
    "population_exceedance.csv",
    "population_boxplot_stats.csv",
    "population_boxplot_fliers.csv",
    "TABLE5_NUMBERS.md",
]

_POSITIONING_GEOGRAPHY_CSVS = [
    "station_map_direct_stec.csv",
    "station_map_diff.csv",
    "latitude_stratification_geographic.csv",
    "latitude_stratification_geomagnetic.csv",
    "latitude_stratification_diff_geographic.csv",
    "latitude_stratification_diff_geomagnetic.csv",
    "population_concentration_by_latitude_geographic.csv",
    "population_concentration_by_latitude_geomagnetic.csv",
    "population_latitude_diff_geographic.csv",
    "population_latitude_diff_geomagnetic.csv",
]


def test_positioning_distributions_owns_table_5_exactly_once():
    matches = [s for s in STAGES if s.name == "positioning_distributions"]
    assert len(matches) == 1
    assert matches[0].canonical_for == "Table 5"

    figure_matches = [
        s for s in STAGES if s.name == "positioning_distributions_figures"
    ]
    assert len(figure_matches) == 1
    assert figure_matches[0].canonical_for is None


def test_positioning_distributions_runs_after_its_dependencies_and_before_figures():
    """Reads POSITIONING/SWI (positioning_coverage) and the recovered-station-days
    cache (positioning_diagnostics), so both must precede it; its own figures stage
    must follow it and precede the two trailing figure-aggregation stages."""
    assert position("positioning_coverage") < position("positioning_distributions")
    assert position("positioning_diagnostics") < position("positioning_distributions")
    assert position("positioning_distributions") < position(
        "positioning_distributions_figures"
    )
    assert position("positioning_distributions_figures") < position("figures")
    assert position("positioning_distributions_figures") < position(
        "manuscript_figures"
    )


def test_positioning_distributions_declares_the_inputs_it_reads():
    inputs = stage("positioning_distributions").inputs
    assert POSITIONING in inputs
    assert SWI in inputs
    assert str(POSITIONING_DIAGNOSTICS_DIR) in inputs


def test_positioning_distributions_declares_all_fourteen_outputs():
    outputs = stage("positioning_distributions").outputs
    for name in _POSITIONING_DISTRIBUTIONS_CSVS:
        assert str(POSITIONING_DISTRIBUTIONS_DIR / name) in outputs, name


def test_positioning_distributions_every_declared_csv_has_a_positive_min_rows_floor():
    floors = stage("positioning_distributions").min_rows
    for name in _POSITIONING_DISTRIBUTIONS_CSVS:
        if name == "TABLE5_NUMBERS.md":
            continue  # markdown, not a CSV - carries no row count to floor
        key = str(POSITIONING_DISTRIBUTIONS_DIR / name)
        assert key in floors, f"{name} has no min_rows floor"
        assert floors[key] > 0


def test_positioning_distributions_data_dependent_floors_sit_below_real_output():
    """Measured against the files actually on disk 2026-09-14
    (multiday_results/analyses/positioning_distributions/rebuilt/, 43,215-row
    coverage-repaired population): overall_boxplot_fliers 3,243, overall_cdf_points
    43,215, regime_boxplot_fliers 3,204, population_boxplot_fliers 2,537. Each floor
    must sit strictly below what is really there."""
    floors = stage("positioning_distributions").min_rows
    real_counts = {
        "overall_boxplot_fliers.csv": 3_243,
        "overall_cdf_points.csv": 43_215,
        "regime_boxplot_fliers.csv": 3_204,
        "population_boxplot_fliers.csv": 2_537,
    }
    for name, real_count in real_counts.items():
        floor = floors[str(POSITIONING_DISTRIBUTIONS_DIR / name)]
        assert 0 < floor < real_count, (name, floor, real_count)


def test_positioning_distributions_check_is_declared():
    assert (
        positioning_distributions_overall_has_all_four_methods
        in stage("positioning_distributions").checks
    )


def test_positioning_distributions_check_passes_with_all_four_methods_populated(
    tmp_path, monkeypatch
):
    path = POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"
    rows = "".join(f"{method},100,1.0\n" for method in METHOD_ORDER)
    _write_relative(tmp_path, monkeypatch, path, "Method,n,median_m\n" + rows)
    outputs = {str(path): {"present": True}}
    assert positioning_distributions_overall_has_all_four_methods(outputs) is None


def test_positioning_distributions_check_fails_when_a_method_row_is_absent(
    tmp_path, monkeypatch
):
    path = POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"
    rows = "".join(f"{method},100,1.0\n" for method in METHOD_ORDER[:-1])
    _write_relative(tmp_path, monkeypatch, path, "Method,n,median_m\n" + rows)
    outputs = {str(path): {"present": True}}
    violation = positioning_distributions_overall_has_all_four_methods(outputs)
    assert violation is not None
    assert METHOD_ORDER[-1] in violation


def test_positioning_distributions_check_fails_when_n_is_empty(tmp_path, monkeypatch):
    """percentile_summary is not reindex-guaranteed the way positioning_summary's
    overall.csv is - a method could in principle appear with an empty `n` if its group
    were somehow written with no rows. This pins the check catches that shape too, not
    only an outright missing row."""
    path = POSITIONING_DISTRIBUTIONS_DIR / "overall_percentile_summary.csv"
    lines = [f"{method},100,1.0\n" for method in METHOD_ORDER[:-1]]
    lines.append(f"{METHOD_ORDER[-1]},,\n")
    _write_relative(tmp_path, monkeypatch, path, "Method,n,median_m\n" + "".join(lines))
    outputs = {str(path): {"present": True}}
    violation = positioning_distributions_overall_has_all_four_methods(outputs)
    assert violation is not None
    assert METHOD_ORDER[-1] in violation


def test_positioning_distributions_states_no_outcome_filter_and_preserves_the_mean():
    caveats = " ".join(stage("positioning_distributions").caveats).lower()
    assert "no outcome-based filter" in caveats
    assert "mean" in caveats and "not the reported statistic" in caveats


def test_positioning_geography_is_declared_and_owns_no_deliverable():
    matches = [s for s in STAGES if s.name == "positioning_geography"]
    assert len(matches) == 1
    assert matches[0].canonical_for is None

    figure_matches = [s for s in STAGES if s.name == "positioning_geography_figures"]
    assert len(figure_matches) == 1
    assert figure_matches[0].canonical_for is None


def test_positioning_geography_runs_after_its_dependencies_and_before_figures():
    assert position("positioning_coverage") < position("positioning_geography")
    assert position("positioning_diagnostics") < position("positioning_geography")
    assert position("positioning_geography") < position("positioning_geography_figures")
    assert position("positioning_geography_figures") < position("figures")
    assert position("positioning_geography_figures") < position("manuscript_figures")


def test_positioning_geography_declares_only_its_own_files_not_the_shared_directory():
    """positioning_quality_gate.py and positioning_model_attribution.py both reuse
    positioning_geography.DEFAULT_OUTPUT_DIR as their own output directory (neither is a
    declared stage) - declaring the bare directory here would make this stage's digest
    depend on files it never writes, and would let a future declaration of either of
    those two modules collide on check_unique_outputs. Every declared output must be a
    named file this module's own main() writes, and the directory itself must not
    appear."""
    outputs = stage("positioning_geography").outputs
    assert str(POSITIONING_GEOGRAPHY_DIR) not in outputs
    for name in _POSITIONING_GEOGRAPHY_CSVS:
        assert str(POSITIONING_GEOGRAPHY_DIR / name) in outputs, name
    # None of the sibling modules' files may appear even by accident.
    for stray in ("quality_gate_correlations.csv", "stec_positioning_correlations.csv"):
        assert str(POSITIONING_GEOGRAPHY_DIR / stray) not in outputs


def test_positioning_geography_every_declared_csv_has_a_positive_min_rows_floor():
    floors = stage("positioning_geography").min_rows
    for name in _POSITIONING_GEOGRAPHY_CSVS:
        key = str(POSITIONING_GEOGRAPHY_DIR / name)
        assert key in floors, f"{name} has no min_rows floor"
        assert floors[key] > 0


def test_positioning_geography_floors_sit_below_real_output():
    """Measured against the files actually on disk 2026-09-14
    (multiday_results/analyses/positioning_geography/rebuilt/): station_map_direct_stec
    55, station_map_diff 57, latitude_stratification_{geographic,geomagnetic} 36 each,
    latitude_stratification_diff/population_concentration/population_latitude_diff 9/9/18
    each (9 LATITUDE_BINS bins, up to x2 for population). Each floor must sit strictly
    below what is really there."""
    floors = stage("positioning_geography").min_rows
    real_counts = {
        "station_map_direct_stec.csv": 55,
        "station_map_diff.csv": 57,
        "latitude_stratification_geographic.csv": 36,
        "latitude_stratification_geomagnetic.csv": 36,
        "latitude_stratification_diff_geographic.csv": 9,
        "latitude_stratification_diff_geomagnetic.csv": 9,
        "population_concentration_by_latitude_geographic.csv": 9,
        "population_concentration_by_latitude_geomagnetic.csv": 9,
        "population_latitude_diff_geographic.csv": 18,
        "population_latitude_diff_geomagnetic.csv": 18,
    }
    for name, real_count in real_counts.items():
        floor = floors[str(POSITIONING_GEOGRAPHY_DIR / name)]
        assert 0 < floor < real_count, (name, floor, real_count)


def test_positioning_geography_figures_declares_only_its_own_positioning_2024_subtree():
    """plots/positioning_geography/ is shared with positioning_quality_gate.py's and
    positioning_model_attribution.py's own (undeclared) viz counterparts, which write
    plots/positioning_geography/quality_gate/ and
    plots/positioning_geography/model_accuracy_vs_positioning/ - this stage must declare
    only its own positioning_2024/ subtree, never the shared parent."""
    outputs = stage("positioning_geography_figures").outputs
    assert f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}" not in outputs
    for output in outputs:
        assert output.startswith(
            f"{POSITIONING_GEOGRAPHY_FIGURES_DIR}/positioning_2024"
        )


def test_positioning_distributions_figures_declares_only_its_own_tree():
    """plots/positioning_distributions/ is exclusive to this module (unlike
    plots/positioning_geography/), so declaring the parent directory itself is safe -
    this pins that every declared output still sits under it."""
    outputs = stage("positioning_distributions_figures").outputs
    for output in outputs:
        assert output.startswith(POSITIONING_DISTRIBUTIONS_FIGURES_DIR)
