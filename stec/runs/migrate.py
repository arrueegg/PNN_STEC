"""Bring the pre-rebuild canonical result trees under provenance, without copying them.

`CLAUDE.md` in the primary checkout carries a hand-maintained table titled "Which results
are canonical", because the filesystem does not say. `multiday_results/` there holds
`summary/`, `summary_May/`, `summary_122_250/`, `with_pretrained_baseline/summary/`,
`mao_evaluation/` and several positioning trees, and nothing on disk distinguishes the one
that backs the paper from the ones that do not.

This does not migrate data. The four canonical trees plus every superseded tree together
run into the hundreds of gigabytes, the disk that holds the rebuild has ~340 GB free, and
that table's own trees are read-only - this script must never write into the checkout it
reads from. What it migrates is provenance: for every tree the table names, a small
pointer record under `ARTIFACT_ROOT/runs/migration_links/` that says where the real data
lives (`imported_from`, `predates_rebuild: true`), plus a `.pipeline/<slug>.json` record
in the same shape a stage would write. That gives these pre-rebuild results the same
"where did this number come from" answer a rebuilt stage already has.

Superseded trees get the same pointer, and then `provenance.mark_superseded` runs against
*the pointer*, never against the tree itself - the tree lives in the read-only checkout,
and `mark_superseded` writes a marker file beside whatever path it is given. Pointed at a
directory under the legacy tree, that write would land inside the legacy tree; pointed at
the pointer file, which lives under `ARTIFACT_ROOT`, it does not. Nothing named as
superseded is ever deleted - by design, since the superseded trees are the only record of
earlier configurations.

A directory's content digest comes from `stec.pipeline.fingerprint.digest`, which already
does the size-dependent thing this needs: a directory is summarised by file count, total
size and newest mtime rather than hashed, so auditing the 70+ GB prediction store costs a
walk, not a read of every byte.

One glob deserves a note. CLAUDE.md's superseded list includes `positioning_2026*`, and one
of that pattern's matches on disk - `positioning_20260216_2052` - is itself the canonical
weighting-ablation tree. That match is excluded from the superseded set rather than double
-counted: a tree cannot be both current and superseded, and the glob is expanded against
whatever the canonical list has already claimed.

Usage::

    python -m stec.runs.migrate                 # dry run: prints the plan, writes nothing
    python -m stec.runs.migrate --apply          # writes pointers, markers, manifest, provenance
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import paths
from ..pipeline import fingerprint, provenance

STAGE_NAME = "migrate"

DEFAULT_LINKS_DIR = paths.ARTIFACT_ROOT / "runs" / "migration_links"
DEFAULT_MANIFEST = paths.ARTIFACT_ROOT / "runs" / "migration_manifest.csv"

MANIFEST_FIELDS = [
    "category",
    "label",
    "source_path",
    "present",
    "digest_kind",
    "size_bytes",
    "file_count",
    "commit",
    "predates_rebuild",
    "imported_from",
    "pointer",
    "superseded_marker",
    "notes",
]


@dataclass(frozen=True)
class NamedTree:
    """One row of CLAUDE.md's canonical-results table, or one entry of its superseded list."""

    category: str  # "canonical" or "superseded"
    label: str
    path: Path
    notes: str = ""


# --- CLAUDE.md's table, encoded -----------------------------------------------------


def canonical_trees(legacy_multiday: Path, legacy_predictions: Path) -> list[NamedTree]:
    """The four rows of CLAUDE.md's "Which results are canonical" table."""
    return [
        NamedTree(
            "canonical",
            "STEC metrics backing Tables 3 & 4",
            legacy_multiday / "with_pretrained_baseline" / "summary",
            "4 models x 2 datasets x 242 days. summary_statistics.csv reproduces the "
            "paper exactly (6.92 / 13.45 / 8.96 / 8.56).",
        ),
        NamedTree(
            "canonical",
            "Positioning, Figs 12/13/A1/A2 + Table 5",
            legacy_multiday / "positioning_comparison_3way",
            "iono weighting, SINEX ground truth, 4 methods, 2024-05-01 to 12-31.",
        ),
        NamedTree(
            "canonical",
            "Weighting ablation (elev vs iono)",
            legacy_multiday / "positioning_20260216_2052",
            "All six arms: STEC_elev/iono, VTEC_elev/iono, gim_elev/iono.",
        ),
        NamedTree(
            "canonical",
            "Per-observation predictions",
            legacy_predictions,
            "Partitioned parquet store; authoritative going forward.",
        ),
    ]


# Literal names from CLAUDE.md's "Superseded" paragraph, excluding the one glob entry.
SUPERSEDED_LITERAL_NAMES = [
    "summary",
    "summary_May",
    "summary_122_250",
    "mao_evaluation",
    "positioning",
    "positioning_iono",
    "positioning_mean",
    "positioning_snx",
]
SUPERSEDED_GLOB = "positioning_2026*"


def superseded_trees(
    legacy_multiday: Path, canonical_paths: set[Path]
) -> list[NamedTree]:
    """The named-and-globbed superseded list, minus whatever the glob shares with canonical."""
    trees = [
        NamedTree(
            "superseded",
            name,
            legacy_multiday / name,
            "Named superseded in CLAUDE.md's canonical-results table; kept rather than "
            "deleted, as the only record of an earlier configuration.",
        )
        for name in SUPERSEDED_LITERAL_NAMES
    ]
    if legacy_multiday.is_dir():
        for match in sorted(legacy_multiday.glob(SUPERSEDED_GLOB)):
            if match in canonical_paths:
                # positioning_20260216_2052 matches the glob but is itself the canonical
                # weighting-ablation tree - a tree cannot be both.
                continue
            trees.append(
                NamedTree(
                    "superseded",
                    match.name,
                    match,
                    f"Matches CLAUDE.md's '{SUPERSEDED_GLOB}' glob.",
                )
            )
    return trees


def build_plan(legacy_multiday: Path, legacy_predictions: Path) -> list[NamedTree]:
    canonical = canonical_trees(legacy_multiday, legacy_predictions)
    superseded = superseded_trees(legacy_multiday, {t.path for t in canonical})
    return canonical + superseded


def slug_for(tree: NamedTree, legacy_root: Path) -> str:
    """Filesystem- and stage-name-safe identifier, derived from the source path itself.

    Deriving it from the path rather than the label means two trees never collide (the
    canonical `with_pretrained_baseline/summary` and the superseded `summary` differ here
    exactly because their paths differ) and the origin is legible from the slug alone.
    """
    return str(tree.path.relative_to(legacy_root)).replace("/", "_")


# --- migration itself -----------------------------------------------------------------


def run(
    legacy_multiday: Path,
    legacy_predictions: Path,
    artifact_root: Path,
    apply: bool,
) -> dict[str, Any]:
    """Compute the full migration plan; write nothing unless `apply` is True.

    Every write - pointer, superseded marker, caveat sidecar, manifest CSV, per-tree
    provenance record - happens only inside this function's `if apply:` block, so calling
    with `apply=False` is safe to do freely: it is a read-only description of what would
    happen, over trees that may run into the hundreds of gigabytes.
    """
    legacy_root = legacy_multiday.parent
    if legacy_predictions.parent != legacy_root:
        raise ValueError(
            "legacy_multiday and legacy_predictions must share a root: "
            f"{legacy_multiday.parent} != {legacy_predictions.parent}"
        )

    trees = build_plan(legacy_multiday, legacy_predictions)
    canonical = [t for t in trees if t.category == "canonical"]
    replacement = [str(t.path) for t in canonical]

    links_dir = artifact_root / "runs" / "migration_links"
    manifest_path = artifact_root / "runs" / "migration_manifest.csv"
    code = provenance.code_version()

    rows: list[dict[str, Any]] = []
    for tree in trees:
        slug = slug_for(tree, legacy_root)
        tree_digest = fingerprint.digest(tree.path)
        pointer_path = links_dir / f"{slug}.json"
        marker_path: Path | None = None

        if apply:
            links_dir.mkdir(parents=True, exist_ok=True)
            pointer_record = {
                "category": tree.category,
                "label": tree.label,
                "notes": tree.notes,
                "source_path": str(tree.path),
                "imported_from": str(tree.path),
                "predates_rebuild": True,
                "digest": tree_digest,
                "code": code,
            }
            pointer_path.write_text(
                json.dumps(pointer_record, indent=2, sort_keys=True, default=str)
            )

            if tree.category == "canonical":
                provenance.write_caveats(
                    pointer_path, STAGE_NAME, [tree.notes] if tree.notes else []
                )
            else:
                marker_path = provenance.mark_superseded(
                    pointer_path, STAGE_NAME, replacement
                )

            provenance.save(
                f"{STAGE_NAME}_{slug}",
                {
                    "stage": f"{STAGE_NAME}_{slug}",
                    "category": tree.category,
                    "label": tree.label,
                    "notes": tree.notes,
                    "source_path": str(tree.path),
                    "imported_from": str(tree.path),
                    "predates_rebuild": True,
                    "code": code,
                    "outputs": {
                        str(pointer_path): provenance.output_record(pointer_path)
                    },
                },
            )

        rows.append(
            {
                "category": tree.category,
                "label": tree.label,
                "source_path": str(tree.path),
                "present": tree.path.exists(),
                "digest_kind": tree_digest.get("kind"),
                "size_bytes": tree_digest.get("size", ""),
                "file_count": tree_digest.get("files", ""),
                "commit": code["commit"],
                "predates_rebuild": True,
                "imported_from": str(tree.path),
                "pointer": str(pointer_path),
                "superseded_marker": str(marker_path) if marker_path else "",
                "notes": tree.notes,
            }
        )

    if apply:
        write_manifest_csv(rows, manifest_path)

    return {
        "trees": trees,
        "rows": rows,
        "manifest_path": manifest_path,
        "links_dir": links_dir,
    }


def write_manifest_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _format_size(size_bytes: Any) -> str:
    if size_bytes == "":
        return "-"
    size = int(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-multiday", type=Path, default=paths.LEGACY_MULTIDAY)
    parser.add_argument(
        "--legacy-predictions", type=Path, default=paths.LEGACY_PREDICTIONS
    )
    parser.add_argument("--artifact-root", type=Path, default=paths.ARTIFACT_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write pointer records, superseded markers, the manifest and per-tree "
        "provenance. Without this flag, nothing is written.",
    )
    args = parser.parse_args()

    # provenance.save resolves ".pipeline" relative to the current directory, the same
    # way stec.pipeline.runner does before it writes anything.
    os.chdir(paths.REPO_ROOT)

    result = run(
        args.legacy_multiday, args.legacy_predictions, args.artifact_root, args.apply
    )
    rows = result["rows"]

    print("APPLYING" if args.apply else "DRY RUN - pass --apply to write anything")
    width = max(len(r["label"]) for r in rows)
    for row in rows:
        state = "present" if row["present"] else "ABSENT"
        size = _format_size(row["size_bytes"])
        files = row["file_count"] if row["file_count"] != "" else "-"
        print(
            f"  [{row['category']:<10}] {row['label']:<{width}}  {state:<7} "
            f"{size:>10}  {files:>6} files  {row['source_path']}"
        )

    missing_canonical = [
        r for r in rows if r["category"] == "canonical" and not r["present"]
    ]
    if missing_canonical:
        print(
            f"\n  {len(missing_canonical)} canonical tree(s) named in CLAUDE.md are "
            "missing from disk:"
        )
        for row in missing_canonical:
            print(f"    {row['label']}: {row['source_path']}")

    missing_superseded = [
        r for r in rows if r["category"] == "superseded" and not r["present"]
    ]
    if missing_superseded:
        print(
            f"\n  {len(missing_superseded)} named superseded tree(s) are missing from disk:"
        )
        for row in missing_superseded:
            print(f"    {row['label']}: {row['source_path']}")

    if args.apply:
        print(f"\n  manifest -> {result['manifest_path']}")
        print(f"  pointer records -> {result['links_dir']}")
        print("  per-tree provenance -> .pipeline/migrate_*.json")
    else:
        print(
            f"\n  {len(rows)} tree(s) would be recorded "
            f"({sum(r['present'] for r in rows)} present, "
            f"{sum(not r['present'] for r in rows)} absent). Re-run with --apply to write."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
