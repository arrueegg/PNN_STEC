"""The declared registry itself, as distinct from the machinery that runs it.

These assert properties of the real stage list: that it is internally consistent, that
each paper deliverable has exactly one owner, and that the two evaluations which are not
what they look like carry the caveat that says so. A caveat lost in a refactor is how a
number ends up in a table it does not belong in.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest

from stec.analysis.daily_metrics import DATASET_LABELS, MODELS
from stec.analysis.positioning_summary import METHOD_ORDER
from stec.pipeline import registry
from stec.pipeline.stages import (
    DAILY_METRICS_DIR,
    ORACLE_EXPERIMENT_DIR,
    POSITIONING,
    POSITIONING_SUMMARY_DIR,
    STAGES,
    WEIGHTING_RUN,
    daily_metrics_summary_has_all_methods_and_datasets,
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
    assert owners["Table 5"] == "positioning_summary"
    # "Table A1" does not exist in either manuscript copy (5 tables, no lettered
    # appendix - Figures 14/15 are the only appendix content, numbered continuously).
    # Locking in its absence so a future edit cannot silently reintroduce the label.
    assert "Table A1" not in owners


def test_common_set_positioning_backs_no_manuscript_table():
    """It recomputes Table 5's methods on a different, smaller station-day population -
    the one solved under both weightings - to answer R1.5's reviewer-response comparison,
    not a printed manuscript table. Unlike every stage above, it correctly claims no
    `canonical_for`, so it can never collide with `positioning_summary`'s "Table 5"."""
    assert stage("common_set_positioning").canonical_for is None
    assert stage("positioning_summary").canonical_for == "Table 5"


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
