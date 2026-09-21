"""`stec.runs.migrate`, which must never touch the read-only checkout it migrates from.

The four things this pins, each corresponding to a hard constraint of the migration
itself:

* a dry run writes nothing at all, anywhere, even though it fully computes the plan;
* a superseded tree gets a marker, and the tree itself - both its presence and its
  contents - is untouched;
* a canonical or superseded tree named but missing from disk is reported as absent, not
  dropped from the plan;
* a directory's digest is the cheap summary (file count, size, mtime), never a hash of
  its bytes, however large it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stec.pipeline import provenance
from stec.runs import migrate


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fake legacy checkout plus a separate, empty artifact root - never the same tree."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(provenance, "STATE_DIR", tmp_path / ".pipeline")
    legacy_root = tmp_path / "legacy_checkout"
    artifact_root = tmp_path / "artifacts"
    return legacy_root, artifact_root


def write_tree(root: Path, *files: str) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("data\n")


def build_legacy_checkout(legacy_root: Path) -> None:
    """The pieces of CLAUDE.md's table this test needs, laid out under one legacy root."""
    multiday = legacy_root / "multiday_results"
    write_tree(multiday, "with_pretrained_baseline/summary/summary_statistics.csv")
    write_tree(multiday, "positioning_comparison_3way/daily_summary_iono.csv")
    write_tree(multiday, "positioning_20260216_2052/daily_summary_iono.csv")
    write_tree(
        legacy_root / "predictions", "finetuned_stec/own/year=2024/doy=132.parquet"
    )

    # Superseded trees, including the ambiguous glob case: positioning_20260216_2052
    # (already written above, as canonical) also matches "positioning_2026*".
    write_tree(multiday, "summary/summary_statistics.csv")
    write_tree(multiday, "positioning/daily_summary.csv")
    write_tree(multiday, "positioning_20260101_0000/daily_summary.csv")
    # summary_May, summary_122_250, mao_evaluation, positioning_iono, positioning_mean,
    # positioning_snx deliberately left absent, to exercise the "missing" path.


def all_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {p for p in root.rglob("*") if p.is_file()}


# --- dry run writes nothing -----------------------------------------------------------


def test_dry_run_writes_nothing(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)
    before = (
        all_files(legacy_root)
        | all_files(artifact_root)
        | all_files(legacy_root.parent / ".pipeline")
    )

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=False,
    )

    assert not artifact_root.exists()
    assert not (legacy_root.parent / ".pipeline").exists()
    after = (
        all_files(legacy_root)
        | all_files(artifact_root)
        | all_files(legacy_root.parent / ".pipeline")
    )
    assert before == after
    # The plan itself is still fully computed - a dry run describes, it just does not write.
    assert (
        len(result["rows"]) == len(migrate.SUPERSEDED_LITERAL_NAMES) + 4 + 1
    )  # 4 canonical + the one non-canonical glob match


def test_dry_run_reports_every_tree_present_or_absent(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=False,
    )

    by_label = {row["label"]: row for row in result["rows"]}
    assert by_label["summary_May"]["present"] is False
    assert by_label["mao_evaluation"]["present"] is False
    assert by_label["summary"]["present"] is True


# --- canonical trees are recorded, never silently dropped -----------------------------


def test_canonical_trees_are_all_named_even_when_absent(workspace):
    legacy_root, artifact_root = workspace
    # Build everything except the weighting-ablation tree.
    multiday = legacy_root / "multiday_results"
    write_tree(multiday, "with_pretrained_baseline/summary/summary_statistics.csv")
    write_tree(multiday, "positioning_comparison_3way/daily_summary_iono.csv")
    write_tree(
        legacy_root / "predictions", "finetuned_stec/own/year=2024/doy=132.parquet"
    )

    result = migrate.run(
        multiday, legacy_root / "predictions", artifact_root, apply=False
    )

    canonical = [r for r in result["rows"] if r["category"] == "canonical"]
    assert len(canonical) == 4, "a named-but-absent canonical tree must still appear"
    by_label = {r["label"]: r for r in canonical}
    assert by_label["Weighting ablation (elev vs iono)"]["present"] is False


# --- the ambiguous glob: a tree cannot be both canonical and superseded ---------------


def test_glob_match_that_is_canonical_is_not_also_listed_superseded(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=False,
    )

    superseded_paths = {
        r["source_path"] for r in result["rows"] if r["category"] == "superseded"
    }
    canonical_paths = {
        r["source_path"] for r in result["rows"] if r["category"] == "canonical"
    }
    weighting_ablation = str(
        legacy_root / "multiday_results" / "positioning_20260216_2052"
    )
    assert weighting_ablation in canonical_paths
    assert weighting_ablation not in superseded_paths
    # The other glob match, which is not also canonical, is still caught.
    assert (
        str(legacy_root / "multiday_results" / "positioning_20260101_0000")
        in superseded_paths
    )


# --- apply: superseded trees are marked, never deleted or modified --------------------


def test_apply_marks_superseded_without_touching_the_source_tree(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)
    source = legacy_root / "multiday_results" / "summary"
    before = all_files(source)

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=True,
    )

    # The legacy tree itself: unchanged, nothing added or removed.
    assert all_files(source) == before

    row = next(r for r in result["rows"] if r["label"] == "summary")
    assert row["superseded_marker"] != ""
    marker = Path(row["superseded_marker"])
    assert marker.exists()
    assert str(artifact_root) in str(marker), (
        "the marker must live under the artifact root"
    )
    marker_record = json.loads(marker.read_text())
    assert marker_record["superseded_by_stage"] == migrate.STAGE_NAME


def test_apply_does_not_mark_canonical_trees_superseded(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=True,
    )

    canonical_rows = [r for r in result["rows"] if r["category"] == "canonical"]
    assert all(r["superseded_marker"] == "" for r in canonical_rows)


# --- digests use the size-dependent rule: directories are summarised, never hashed ----


def test_directory_digest_is_a_summary_not_a_content_hash(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)
    # A file well past any reasonable hashing threshold, to prove the tree-level digest
    # does not read it - only stat() calls back this up, never a checksum of the bytes.
    big_file = (
        legacy_root / "predictions" / "finetuned_stec" / "own" / "year=2024" / "big.bin"
    )
    big_file.write_bytes(b"0" * (2 * 1024 * 1024))

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=False,
    )

    predictions_row = next(
        r for r in result["rows"] if r["label"] == "Per-observation predictions"
    )
    assert predictions_row["digest_kind"] == "tree"
    assert predictions_row["file_count"] >= 2
    assert int(predictions_row["size_bytes"]) >= 2 * 1024 * 1024


# --- nothing is ever written outside ARTIFACT_ROOT or .pipeline -----------------------


def test_apply_writes_only_under_artifact_root_and_pipeline(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)
    pipeline_dir = legacy_root.parent / ".pipeline"  # cwd is tmp_path, per the fixture
    before_legacy = all_files(legacy_root)

    migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=True,
    )

    assert all_files(legacy_root) == before_legacy, (
        "the read-only checkout must be untouched"
    )
    created = all_files(artifact_root) | all_files(pipeline_dir)
    assert created, "apply should have written something"
    for path in created:
        assert str(path).startswith((str(artifact_root), str(pipeline_dir)))


def test_manifest_csv_has_one_row_per_named_tree(workspace):
    legacy_root, artifact_root = workspace
    build_legacy_checkout(legacy_root)

    result = migrate.run(
        legacy_root / "multiday_results",
        legacy_root / "predictions",
        artifact_root,
        apply=True,
    )

    manifest_path = result["manifest_path"]
    assert manifest_path.exists()
    assert str(manifest_path).startswith(str(artifact_root))
    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) == len(result["rows"]) + 1  # header + one row per tree
