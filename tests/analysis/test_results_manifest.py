"""The manifest is generated from the registry, so it cannot drift from it.

The disk-inventory section below (`test_disk_inventory_rows_...` onward) covers the other
half of this module: what is actually on disk, independent of anything a stage claims. All
fixtures are small synthetic trees built in `tmp_path` - never the real, ~640 GB
`multiday_results/` or the prediction store, matching the resource discipline the module
itself is written to respect.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stec.analysis.results_manifest import (
    DISK_INVENTORY_COLUMNS,
    consistency_problems,
    day_directories,
    disk_inventory_rows,
    manifest_rows,
    metrics_index_rows,
    summarise_positioning,
    summarise_stec,
    summarise_store_partitions,
    superseded_rows,
    undeclared_trees,
    write_csv,
)
from stec.pipeline import registry
from stec.pipeline.stage import Stage
from stec.runs.migrate import build_plan


def demo_stages() -> list[Stage]:
    return [
        Stage(
            "producer",
            "-m demo.producer",
            "Table 9",
            "makes a number",
            outputs=["out/table9.csv"],
            canonical_for="Table 9",
            caveats=["only valid on Tuesdays"],
            supersedes=["out/old_table9.csv"],
        ),
        Stage(
            "quiet", "-m demo.quiet", "-", "makes another", outputs=["out/other.csv"]
        ),
    ]


def test_every_declared_output_appears_in_the_metrics_index():
    rows = metrics_index_rows(demo_stages())
    assert {r["output"] for r in rows} == {"out/table9.csv", "out/other.csv"}


def test_the_index_carries_the_reviewer_comment_each_output_answers():
    rows = {r["output"]: r for r in metrics_index_rows(demo_stages())}
    assert rows["out/table9.csv"]["answers"] == "Table 9"


def test_caveats_travel_into_the_index():
    """A number must not be liftable into a table without its condition."""
    rows = {r["output"]: r for r in metrics_index_rows(demo_stages())}
    assert "only valid on Tuesdays" in rows["out/table9.csv"]["caveats"]


def test_manifest_marks_which_stages_have_caveats():
    rows = {r["stage"]: r for r in manifest_rows(demo_stages())}
    assert rows["producer"]["has_caveats"] == "yes"
    assert rows["quiet"]["has_caveats"] == "no"


def test_superseded_artifacts_are_listed_with_what_replaced_them():
    rows = superseded_rows(demo_stages())
    assert len(rows) == 1
    assert rows[0]["superseded_artifact"] == "out/old_table9.csv"
    assert rows[0]["superseded_by_stage"] == "producer"


def test_a_never_run_owner_is_reported_not_silently_omitted():
    """ "Absent from the manifest" and "absent from the pipeline" must differ."""
    problems = " ".join(consistency_problems(demo_stages()))
    assert "never been run" in problems
    assert "Table 9" in problems


def test_a_stage_without_outputs_is_reported():
    orphan = [Stage("nothing", "-m demo.nothing", "-", "produces nothing")]
    assert any("declares no outputs" in p for p in consistency_problems(orphan))


# --- against the real registry --------------------------------------------------------


def test_the_real_registry_generates_a_manifest():
    rows = manifest_rows(registry.STAGES)
    assert len(rows) == len(registry.STAGES)


def test_every_paper_deliverable_has_exactly_one_owner_in_the_manifest():
    owners = [
        r["deliverable"] for r in manifest_rows(registry.STAGES) if r["deliverable"]
    ]
    assert len(owners) == len(set(owners)), (
        "a deliverable with two owners has no answer"
    )
    for expected in ("Tables 1 and 2", "Tables 3 and 4", "Table 5"):
        assert expected in owners
    # "Table A1" does not exist in the manuscript (5 tables, no lettered appendix) -
    # common_set_positioning backs the R1.5 reviewer-response numbers instead and
    # correctly claims no deliverable, so it must never appear here.
    assert "Table A1" not in owners


def test_the_two_dangerous_evaluations_carry_caveats_into_the_index():
    rows = {r["stage"]: r for r in metrics_index_rows(registry.STAGES)}
    assert "not comparable" in rows["oracle_benchmark"]["caveats"].lower()
    assert "never standalone" in rows["madrigal_reference_offset"]["caveats"].lower()


# --- disk inventory: what is actually on disk -------------------------------------------


def write_tree(root: Path, *files: str) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data\n")


def build_legacy_checkout(legacy_root: Path) -> None:
    """The pieces of CLAUDE.md's canonical-results table, laid out under one legacy root -
    mirrors `tests/runs/test_migrate.py::build_legacy_checkout`, since both modules read
    the same on-disk shape."""
    multiday = legacy_root / "multiday_results"
    write_tree(multiday, "with_pretrained_baseline/summary/summary_statistics.csv")
    write_tree(multiday, "positioning_comparison_3way/daily_summary_iono.csv")
    write_tree(multiday, "positioning_20260216_2052/daily_summary_iono.csv")
    write_tree(
        legacy_root / "predictions", "finetuned_stec/own/year=2024/doy=132.parquet"
    )
    # Superseded, named in CLAUDE.md.
    write_tree(multiday, "summary/summary_statistics.csv")
    write_tree(multiday, "positioning/daily_summary.csv")
    # A tree CLAUDE.md's table has never named - the case only a disk walk catches.
    write_tree(multiday, "some_new_experiment_2026/results.csv")


def day_directory_names(*doys: int) -> list[str]:
    return [f"2024_DOY_{doy:03d}/summary.csv" for doy in doys]


# --- day_directories --------------------------------------------------------------------


def test_day_directories_matches_pattern_and_ignores_retries(tmp_path):
    (tmp_path / "2024_DOY_132").mkdir()
    (tmp_path / "2024_DOY_133").mkdir()
    (tmp_path / "2024_DOY_133_try1").mkdir()  # ad-hoc retry, must not count as a day
    (tmp_path / "not_a_day_dir").mkdir()

    found = day_directories(tmp_path)

    assert [p.name for p in found] == ["2024_DOY_132", "2024_DOY_133"]


# --- summarise_positioning ----------------------------------------------------------


def test_summarise_positioning_reads_arms_weighting_span_and_station_days(tmp_path):
    summary_csv = tmp_path / "multiday_summary.csv"
    pd.DataFrame(
        [
            {"method": "STEC_iono", "date": "2024-05-01", "station": "AMC4"},
            {"method": "STEC_iono", "date": "2024-05-01", "station": "ZIMM"},
            {"method": "STEC_iono", "date": "2024-05-02", "station": "AMC4"},
            {"method": "gim_elev", "date": "2024-05-01", "station": "AMC4"},
        ]
    ).to_csv(summary_csv, index=False)

    facts = summarise_positioning(summary_csv)

    assert facts["kind"] == "positioning"
    assert facts["arms"] == "STEC_iono;gim_elev"
    assert facts["weighting"] == "elev;iono"
    assert facts["n_days"] == 2
    assert facts["date_min"] == "2024-05-01"
    assert facts["date_max"] == "2024-05-02"
    assert facts["n_stations"] == 2
    assert facts["station_days_per_arm"] == "STEC_iono=3;gim_elev=1"
    assert facts["n_rows"] == 4


# --- summarise_stec -------------------------------------------------------------------


def test_summarise_stec_reports_span_and_flags_missing_summary_dir(tmp_path):
    tree = tmp_path / "with_pretrained_baseline"
    tree.mkdir()
    for relative in day_directory_names(122, 130):
        write_tree(tree, relative)

    facts = summarise_stec(tree)

    assert facts["kind"] == "stec_evaluation"
    assert facts["n_days"] == 2
    assert facts["date_min"] == "2024-122"
    assert facts["date_max"] == "2024-130"
    assert "pretrained" in facts["arms"]
    assert facts["notes"] == "no summary/"

    (tree / "summary").mkdir()
    assert summarise_stec(tree)["notes"] == "summary/ present"


# --- summarise_store_partitions: filenames only, never parquet content ------------------


def test_summarise_store_partitions_reads_filenames_not_content(tmp_path):
    store = tmp_path / "predictions"
    own = store / "finetuned_stec" / "own" / "year=2024"
    own.mkdir(parents=True)
    (own / "doy=132.parquet").write_bytes(b"x" * 100)
    (own / "doy=133.parquet").write_bytes(b"x" * 300)
    pretrained_2023 = store / "pretrained_stec" / "own" / "year=2023"
    pretrained_2023.mkdir(parents=True)
    (pretrained_2023 / "doy=010.parquet").write_bytes(b"x" * 50)

    rows = {r["name"]: r for r in summarise_store_partitions(store)}

    own_row = rows["finetuned_stec/own"]
    assert own_row["kind"] == "prediction_store_partition"
    assert own_row["status"] == "canonical"
    assert own_row["n_days"] == 2
    assert own_row["date_min"] == "2024-132"
    assert own_row["date_max"] == "2024-133"
    assert own_row["file_count"] == 2
    assert own_row["size_gb"] == round(400 / 1024**3, 2)

    pretrained_row = rows["pretrained_stec/own"]
    # n_days is a total file count across every year (this partition has one, in
    # year=2023); date_min/max stay 2024-only, so a bare day count does not misread a
    # non-2024 file as part of the 2024 test-period span.
    assert pretrained_row["n_days"] == 1
    assert pretrained_row["date_min"] == ""
    assert "2023-2023" in pretrained_row["notes"]


# --- undeclared_trees: the drift a fixed table cannot see coming ------------------------


def test_undeclared_trees_finds_only_what_the_plan_does_not_name(tmp_path):
    build_legacy_checkout(tmp_path)
    multiday = tmp_path / "multiday_results"
    plan = build_plan(multiday, tmp_path / "predictions")
    declared = {t.path for t in plan}

    found = undeclared_trees(multiday, declared)

    assert [p.name for p in found] == ["some_new_experiment_2026"]


def test_undeclared_trees_excludes_root_level_day_directories(tmp_path):
    multiday = tmp_path / "multiday_results"
    for relative in day_directory_names(132):
        write_tree(multiday, relative)

    # Root-level day payloads are rolled up elsewhere, not reported as loose trees.
    assert undeclared_trees(multiday, declared=set()) == []


# --- disk_inventory_rows: the end-to-end walk --------------------------------------------


def test_disk_inventory_rows_classifies_canonical_superseded_and_unreviewed(tmp_path):
    build_legacy_checkout(tmp_path)

    rows = disk_inventory_rows(tmp_path / "multiday_results", tmp_path / "predictions")
    # Rows for a migrate-plan tree are keyed by CLAUDE.md's label, not the bare directory
    # name: the canonical STEC-metrics tree and the superseded top-level `summary/` both
    # end in a directory literally named "summary", so `name` alone would collide.
    by_label = {r["label"]: r for r in rows}

    assert by_label["STEC metrics backing Tables 3 & 4"]["status"] == "canonical"
    assert by_label["Positioning, Figs 12/13/A1/A2 + Table 5"]["status"] == "canonical"
    assert by_label["summary"]["status"] == "superseded"
    assert by_label["positioning"]["status"] == "superseded"
    assert by_label["some_new_experiment_2026"]["status"] == "unreviewed"
    # The top-level with_pretrained_baseline directory must not also show up as its own,
    # spuriously "unreviewed" row - its one canonical artifact is the nested summary/.
    assert not any(r["path"].endswith("with_pretrained_baseline") for r in rows)


def test_disk_inventory_rows_includes_prediction_store_partitions(tmp_path):
    build_legacy_checkout(tmp_path)

    rows = disk_inventory_rows(tmp_path / "multiday_results", tmp_path / "predictions")

    partitions = [r for r in rows if r["kind"] == "prediction_store_partition"]
    assert any(r["name"] == "finetuned_stec/own" for r in partitions)


def test_disk_inventory_rows_reports_a_missing_canonical_tree_as_absent(tmp_path):
    # Everything except the weighting-ablation tree, so its row must still appear.
    multiday = tmp_path / "multiday_results"
    write_tree(multiday, "with_pretrained_baseline/summary/summary_statistics.csv")
    write_tree(multiday, "positioning_comparison_3way/daily_summary_iono.csv")
    write_tree(tmp_path / "predictions", "finetuned_stec/own/year=2024/doy=132.parquet")

    rows = disk_inventory_rows(multiday, tmp_path / "predictions")
    by_label = {r["label"]: r for r in rows}

    weighting_ablation = by_label["Weighting ablation (elev vs iono)"]
    assert weighting_ablation["status"] == "canonical"
    assert weighting_ablation["present"] is False
    assert weighting_ablation["size_gb"] == 0.0


def test_disk_inventory_rows_rolls_up_root_level_day_directories(tmp_path):
    multiday = tmp_path / "multiday_results"
    write_tree(multiday, "with_pretrained_baseline/summary/summary_statistics.csv")
    for relative in day_directory_names(122, 123, 130):
        write_tree(multiday, relative)

    rows = disk_inventory_rows(multiday, tmp_path / "predictions")

    rollup = next(r for r in rows if r["name"] == "2024_DOY_* (root level)")
    assert rollup["status"] == "unreviewed"
    assert rollup["n_days"] == 3
    assert rollup["date_min"] == "2024-122"
    assert rollup["date_max"] == "2024-130"
    # The per-day directories must not also show up as three separate loose trees.
    assert not any(r["name"].startswith("2024_DOY_") for r in rows if r is not rollup)


def test_disk_inventory_rows_picks_up_positioning_facts_for_a_canonical_tree(tmp_path):
    multiday = tmp_path / "multiday_results"
    tree = multiday / "positioning_comparison_3way"
    tree.mkdir(parents=True)
    pd.DataFrame(
        [
            {"method": "STEC_iono", "date": "2024-05-01", "station": "AMC4"},
            {"method": "gim_iono", "date": "2024-05-01", "station": "AMC4"},
        ]
    ).to_csv(tree / "multiday_summary.csv", index=False)

    rows = disk_inventory_rows(multiday, tmp_path / "predictions")
    by_label = {r["label"]: r for r in rows}

    row = by_label["Positioning, Figs 12/13/A1/A2 + Table 5"]
    assert row["kind"] == "positioning"
    assert row["arms"] == "STEC_iono;gim_iono"
    assert row["n_days"] == 1


# --- disk_inventory_rows round-trips through the CSV writer without missing keys --------


def test_disk_inventory_rows_write_csv_round_trips_with_mixed_row_shapes(tmp_path):
    """Positioning rows carry `arms`/`n_days`; a plain analysis-output row does not. Every
    row must still write cleanly against one shared column list."""
    build_legacy_checkout(tmp_path)
    rows = disk_inventory_rows(tmp_path / "multiday_results", tmp_path / "predictions")

    out = tmp_path / "disk_inventory.csv"
    write_csv(rows, DISK_INVENTORY_COLUMNS, out)

    written = pd.read_csv(out, keep_default_na=False)
    assert list(written.columns) == DISK_INVENTORY_COLUMNS
    assert len(written) == len(rows)
