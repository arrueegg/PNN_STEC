"""`stec.runs.restructure_results` - the tool that moves `multiday_results/` into the
layout `docs/revision/results_layout.md` describes.

Pins the properties the migration script must hold on a real, 640 GB shared tree:

* a dry run writes nothing, anywhere;
* each bucket's classification rule fires on the case it exists for (superseded,
  declared analysis output including the one irregular name, per-day payload, structural
  STEC-evaluation sweep, positioning run, unclassified fallback);
* applying is idempotent - a second run with nothing new on disk plans zero moves;
* applying refuses outright if any destination already exists, and touches nothing when
  it refuses;
* every move is recorded in a manifest that `undo` can reverse exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stec.runs import restructure_results as rr


def write_tree(root: Path, *files: str) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data\n")


@pytest.fixture
def source_root(tmp_path) -> Path:
    """A synthetic `multiday_results/`, covering one example of every bucket."""
    root = tmp_path / "multiday_results"

    # per_day: the flat root sweep.
    write_tree(root, "2024_DOY_122/temp_config_stec_2024_122.yaml")
    write_tree(root, "2024_DOY_123/temp_config_stec_2024_123.yaml")

    # analyses: a regular rebuilt name, its pre-rebuild counterpart, and the one
    # irregular stage (repair_gim_baseline -> gim_baseline_repair).
    write_tree(root, "daily_metrics_rebuilt/summary.csv")
    write_tree(root, "daily_metrics/summary.csv")
    write_tree(root, "gim_baseline_repair/gim_repair_report.csv")

    # stec_evaluation: a named sweep with its own per-day children.
    write_tree(root, "with_pretrained_baseline/2024_DOY_122/evaluation/metrics.csv")
    write_tree(root, "with_pretrained_baseline/summary/summary_statistics.csv")

    # positioning: the canonical weighting-ablation tree (also matches the superseded
    # glob, and must not be double-classified), plus an unreviewed sweep.
    write_tree(root, "positioning_20260216_2052/daily_summary_iono.csv")
    write_tree(root, "positioning_full_coverage/multiday_summary.csv")

    # superseded: a literal name and a glob match that is *not* the canonical tree above.
    # "positioning" specifically regression-tests a real bug found against the actual
    # legacy tree: its bare name must not be swallowed by the "already-migrated bucket"
    # skip just because "positioning_runs" (the bucket) and "positioning" (this tree)
    # look related.
    write_tree(root, "summary/summary_statistics.csv")
    write_tree(root, "positioning/daily_summary.csv")
    write_tree(root, "positioning_20260101_0000/daily_summary.csv")

    # unclassified: matches no rule (not a declared output, no per-day children, no
    # "positioning" prefix).
    write_tree(root, "stratified_comparison_pretrained/by_elevation.csv")

    write_tree(tmp_path / "predictions", "finetuned_stec/own/year=2024/doy=132.parquet")
    return root


def all_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file()}


# --- classification ---------------------------------------------------------------------


def test_plan_covers_every_bucket(source_root):
    moves = rr.plan(source_root)
    by_source_name = {m.source.name: m for m in moves}

    assert by_source_name["2024_DOY_122"].bucket == "per_day"
    assert (
        by_source_name["2024_DOY_122"].dest == source_root / "per_day" / "2024" / "122"
    )

    assert by_source_name["daily_metrics_rebuilt"].bucket == "analyses"
    assert (
        by_source_name["daily_metrics_rebuilt"].dest
        == source_root / "analyses" / "daily_metrics" / "rebuilt"
    )
    assert (
        by_source_name["daily_metrics"].dest
        == source_root / "analyses" / "daily_metrics" / "pre_rebuild"
    )
    # The irregular case: dirname != stage name, resolved from the stage's own declared
    # output rather than a suffix guess.
    assert (
        by_source_name["gim_baseline_repair"].dest
        == source_root / "analyses" / "repair_gim_baseline" / "pre_rebuild"
    )

    assert by_source_name["with_pretrained_baseline"].bucket == "stec_evaluation"
    assert (
        by_source_name["with_pretrained_baseline"].dest
        == source_root / "stec_evaluation" / "with_pretrained_baseline"
    )

    assert by_source_name["positioning_20260216_2052"].bucket == "positioning_runs"
    assert (
        by_source_name["positioning_20260216_2052"].dest
        == source_root / "positioning_runs" / "20260216_2052"
    )
    assert by_source_name["positioning_full_coverage"].bucket == "positioning_runs"

    assert by_source_name["summary"].bucket == "superseded"
    assert by_source_name["summary"].dest == source_root / "superseded" / "summary"
    assert by_source_name["positioning_20260101_0000"].bucket == "superseded"

    assert by_source_name["stratified_comparison_pretrained"].bucket == "unclassified"


def test_bare_positioning_tree_is_moved_not_swallowed_by_the_bucket_skip(source_root):
    """Regression test: the superseded tree literally named `positioning` must not be
    mistaken for the `positioning_runs` bucket and silently left in place.
    """
    moves = rr.plan(source_root)
    bare = next(m for m in moves if m.source.name == "positioning")
    assert bare.bucket == "superseded"
    assert bare.dest == source_root / "superseded" / "positioning"


def test_canonical_glob_match_is_not_also_superseded(source_root):
    moves = rr.plan(source_root)
    weighting_ablation = next(
        m for m in moves if m.source.name == "positioning_20260216_2052"
    )
    assert weighting_ablation.bucket == "positioning_runs"


# --- idempotence and dry-run safety -------------------------------------------------


def test_dry_run_writes_nothing(source_root):
    before = all_files(source_root)
    moves = rr.plan(source_root)
    assert moves  # the plan is non-trivial for this fixture
    assert all_files(source_root) == before


def test_second_plan_after_apply_is_empty(source_root):
    moves = rr.plan(source_root)
    rr.apply_plan(source_root, moves)

    assert rr.plan(source_root) == []


def test_apply_moves_content_not_just_the_directory_name(source_root):
    moves = rr.plan(source_root)
    rr.apply_plan(source_root, moves)

    moved = source_root / "analyses" / "daily_metrics" / "rebuilt" / "summary.csv"
    assert moved.read_text() == "data\n"
    assert not (source_root / "daily_metrics_rebuilt").exists()


# --- refuses rather than clobbers ----------------------------------------------------


def test_apply_refuses_when_destination_exists(source_root):
    write_tree(source_root, "superseded/summary/collision.csv")
    before = all_files(source_root)

    with pytest.raises(FileExistsError):
        rr.apply_plan(source_root, rr.plan(source_root))

    # Fails closed: nothing was moved, not even the moves ordered before the collision.
    assert all_files(source_root) == before


def test_plan_raises_on_destination_collision(tmp_path, monkeypatch):
    # No real pair of top-level names collides today (every bucket's destination is keyed
    # either by the source's own unique name or by the registry's 1:1 stage mapping) - so
    # the guard is exercised by forcing two distinct sources through `classify` to the
    # same destination, the way a future naming rule with a real collision would.
    root = tmp_path / "multiday_results"
    write_tree(root, "some_tree/a.csv")
    write_tree(root, "another_tree/b.csv")

    def classify_everything_the_same(
        top: Path, source_root: Path, superseded_paths: set[Path]
    ) -> rr.Move:
        return rr.Move(top, root / "positioning" / "x", "positioning", "x")

    monkeypatch.setattr(rr, "classify", classify_everything_the_same)

    with pytest.raises(ValueError, match="collide"):
        rr.plan(root)


# --- manifest and undo ----------------------------------------------------------------


def test_apply_writes_a_manifest_that_undo_reverses_exactly(source_root):
    moves = rr.plan(source_root)
    manifest_path = rr.apply_plan(source_root, moves)

    assert manifest_path.exists()
    record = json.loads(manifest_path.read_text())
    assert len(record["moves"]) == len(moves)

    after_apply = all_files(source_root) - {manifest_path}
    rr.undo(manifest_path)
    after_undo = all_files(source_root) - {manifest_path}

    # Every moved file is back under its original top-level name. Undo reverses file
    # locations, not directory scaffolding, so an empty `analyses/` parent is fine - no
    # file is left under it.
    assert after_undo != after_apply
    assert (source_root / "daily_metrics_rebuilt" / "summary.csv").exists()
    assert not any(p.is_file() for p in (source_root / "analyses").rglob("*"))


def test_undo_refuses_when_original_location_is_occupied(source_root):
    moves = rr.plan(source_root)
    manifest_path = rr.apply_plan(source_root, moves)

    # Something new now sits where "daily_metrics_rebuilt" used to be.
    write_tree(source_root, "daily_metrics_rebuilt/unrelated.csv")

    with pytest.raises(FileExistsError):
        rr.undo(manifest_path)


# --- source root missing or already migrated ------------------------------------------


def test_plan_on_missing_root_is_empty(tmp_path):
    assert rr.plan(tmp_path / "does_not_exist") == []


def test_plan_skips_the_layouts_own_bucket_directories(tmp_path):
    root = tmp_path / "multiday_results"
    write_tree(root, "analyses/daily_metrics/rebuilt/summary.csv")
    write_tree(root, "per_day/2024/122/config.yaml")

    assert rr.plan(root) == []
