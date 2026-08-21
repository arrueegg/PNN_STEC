"""Where every reported number comes from, generated from the registry.

`CLAUDE.md` currently carries a hand-maintained table titled "Which results are canonical",
listing which result trees to trust and which are superseded. That table exists because the
filesystem does not say: `multiday_results/` holds `summary/`, `summary_May/`,
`summary_122_250/`, `with_pretrained_baseline/summary/` and several positioning trees, and
nothing on disk distinguishes the one that backs the paper from the four that do not.

A hand-maintained table is the wrong shape for that job. It is written once, consulted by
people who already know the answer, and silently wrong the moment a stage changes. The
registry already knows: which stage owns which deliverable, what each writes, what it
supersedes, and what caveats attach to reading it. This turns that into four files:

* `manifest.csv`        - deliverable, owning stage, outputs, caveats, provenance record
* `superseded.csv`      - artifacts a stage has replaced, and what replaced them
* `metrics_index.csv`   - every metric CSV mapped to the reviewer comment it answers
* `disk_inventory.csv`  - every results tree actually on disk, checked against that table

The registry describes what the *rebuilt* pipeline claims to produce; it says nothing about
the pre-rebuild trees under `stec.config.paths.LEGACY_MULTIDAY` and `LEGACY_PREDICTIONS`
that `CLAUDE.md`'s canonical-results table still names as authoritative
(`with_pretrained_baseline/summary/`, `positioning_comparison_3way/`, ...). Those are read
straight off disk, classified against `stec.runs.migrate`'s encoding of that same table
(reused rather than copied a third time - `migrate.py` already turned it from prose into
data), and every top-level directory the table has never named is reported too, as
"unreviewed" - the case a fixed table cannot see coming and the reason this needs to walk
the filesystem rather than only read the registry. `metrics_index.csv` replaces
`multiday_results/revision_metrics_index.csv`, which was maintained by hand alongside the
analyses it describes.

Running this is also a consistency check, not only a report: a deliverable with no owner,
an output whose provenance record is missing, or a tree on disk the canonical table has
never reviewed is reported rather than omitted, because "absent from the manifest" and
"absent from the pipeline" must not look the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import paths
from ..pipeline import fingerprint, registry
from ..pipeline.stage import Stage
from ..runs.migrate import build_plan, slug_for

PROVENANCE_DIR = Path(".pipeline")


def _provenance(stage: Stage) -> dict[str, Any]:
    record = PROVENANCE_DIR / f"{stage.name}.json"
    if not record.exists():
        return {}
    try:
        return json.loads(record.read_text())
    except json.JSONDecodeError:
        return {}


def manifest_rows(stages: list[Stage]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        provenance = _provenance(stage)
        code = provenance.get("code", {})
        rows.append(
            {
                "deliverable": stage.canonical_for or "",
                "stage": stage.name,
                "answers": stage.answers,
                "command": stage.command,
                "outputs": " | ".join(stage.outputs),
                "caveats": " | ".join(stage.caveats),
                "has_caveats": "yes" if stage.caveats else "no",
                "last_run_commit": code.get("commit", ""),
                "tree_dirty_when_run": code.get("dirty", ""),
                "recorded_at": provenance.get("recorded_at", ""),
                "provenance": "recorded" if provenance else "never run",
            }
        )
    return rows


def superseded_rows(stages: list[Stage]) -> list[dict[str, Any]]:
    return [
        {
            "superseded_artifact": older,
            "superseded_by_stage": stage.name,
            "replacement_outputs": " | ".join(stage.outputs),
            "still_on_disk": "yes" if Path(older).exists() else "no",
        }
        for stage in stages
        for older in stage.supersedes
    ]


def metrics_index_rows(stages: list[Stage]) -> list[dict[str, Any]]:
    """One row per declared output, mapped to the reviewer comment it answers."""
    return [
        {
            "output": output,
            "answers": stage.answers,
            "stage": stage.name,
            "description": stage.description,
            "canonical_for": stage.canonical_for or "",
            "caveats": " | ".join(stage.caveats),
        }
        for stage in stages
        for output in stage.outputs
    ]


def consistency_problems(stages: list[Stage]) -> list[str]:
    """Things a reader would be misled by, reported rather than quietly omitted."""
    problems = []
    for stage in stages:
        if not stage.outputs:
            problems.append(f"{stage.name}: declares no outputs")
        if stage.canonical_for and not _provenance(stage):
            problems.append(
                f"{stage.name}: owns '{stage.canonical_for}' but has never been run, "
                "so that deliverable has no provenance record"
            )
        for older in stage.supersedes:
            if not Path(older).exists():
                problems.append(
                    f"{stage.name}: supersedes {older}, which is no longer on disk"
                )
    return problems


def write_csv(rows: list[dict[str, Any]], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# --- disk inventory: what is actually on disk, not just what the registry claims -------
#
# The pre-rebuild PNN_STEC checkout's `results_manifest.py` walked `multiday_results/` and
# `predictions/` directly and reported, per tree: status (canonical / superseded /
# unreviewed), size, and - where the tree's own layout reveals them - the facts that decide
# whether two numbers may be quoted side by side (arms, weighting scheme, date span,
# station-days per arm). That walk is what is restored below; classification is delegated
# to `stec.runs.migrate.build_plan`, which already encodes CLAUDE.md's canonical/superseded
# table for its own (unrelated) provenance-migration job, so this does not become a third
# hand-maintained copy of the same list.

DOY_DIR_PATTERN = re.compile(r"^2024_DOY_(\d{3})$")

DISK_INVENTORY_COLUMNS = [
    "status",
    "kind",
    "name",
    "label",
    "weighting",
    "n_days",
    "date_min",
    "date_max",
    "n_stations",
    "arms",
    "station_days_per_arm",
    "n_rows",
    "size_gb",
    "file_count",
    "present",
    "path",
    "notes",
]


def day_directories(parent: Path) -> list[Path]:
    """Per-day payload directories, ignoring ad-hoc retries like ``_try1``."""
    return sorted(p for p in parent.glob("2024_DOY_*") if DOY_DIR_PATTERN.match(p.name))


def summarise_positioning(summary_csv: Path) -> dict[str, Any]:
    """Arms, weighting, span and per-arm station-day counts for a positioning tree.

    `multiday_summary.csv` is a small, aggregated per-station-day table (hundreds to a
    few thousand rows) - reading it whole with pandas is not the store-streaming case
    the prediction parquet requires.
    """
    df = pd.read_csv(summary_csv)
    arms = sorted(df["method"].unique())
    counts = df.groupby("method").size().to_dict()
    weightings = sorted({arm.rsplit("_", 1)[-1] for arm in arms})
    return {
        "kind": "positioning",
        "arms": ";".join(arms),
        "weighting": ";".join(weightings),
        "n_days": df["date"].nunique(),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "n_stations": df["station"].nunique(),
        "station_days_per_arm": ";".join(f"{arm}={counts[arm]}" for arm in arms),
        "n_rows": len(df),
    }


def summarise_stec(tree: Path) -> dict[str, Any]:
    """Span and day count for a STEC evaluation tree, from its per-day directory names."""
    day_dirs = day_directories(tree)
    doys = [int(DOY_DIR_PATTERN.match(p.name).group(1)) for p in day_dirs]
    has_summary = (tree / "summary").is_dir()
    return {
        "kind": "stec_evaluation",
        "arms": "stec;vtec;gim" + (";pretrained" if "pretrained" in tree.name else ""),
        "n_days": len(doys),
        "date_min": f"2024-{min(doys):03d}" if doys else "",
        "date_max": f"2024-{max(doys):03d}" if doys else "",
        "notes": "summary/ present" if has_summary else "no summary/",
    }


def summarise_store_partitions(store: Path) -> list[dict[str, Any]]:
    """One row per (model variant, dataset) partition of the prediction store.

    Reads only file *names* (``year=<YYYY>/doy=<DDD>.parquet``) to recover the span, and
    sizes each partition with `fingerprint.digest`, which stats every file and never opens
    one - the store's parquet content is out of scope for an inventory and stays unread.
    """
    rows: list[dict[str, Any]] = []
    for variant_dir in sorted(p for p in store.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            files = sorted(dataset_dir.glob("year=*/doy=*.parquet"))
            # The pretrained variant spans several years, so a bare day count would read
            # as a 2024 span it does not have.
            years = sorted({int(f.parent.name.split("=")[-1]) for f in files})
            doys_2024 = [
                int(f.stem.split("=")[-1])
                for f in files
                if f.parent.name == "year=2024"
            ]
            tree_digest = fingerprint.digest(dataset_dir)
            rows.append(
                {
                    "status": "canonical",
                    "kind": "prediction_store_partition",
                    "name": f"{variant_dir.name}/{dataset_dir.name}",
                    "label": "per-observation predictions",
                    "path": str(dataset_dir),
                    "present": True,
                    "size_gb": round(tree_digest.get("size", 0) / 1024**3, 2),
                    "file_count": tree_digest.get("files", ""),
                    "n_days": len(files),
                    "date_min": f"2024-{min(doys_2024):03d}" if doys_2024 else "",
                    "date_max": f"2024-{max(doys_2024):03d}" if doys_2024 else "",
                    "notes": (
                        f"years {years[0]}-{years[-1]}, {len(doys_2024)} days in 2024"
                        if years
                        else "empty"
                    ),
                }
            )
    return rows


def undeclared_trees(legacy_multiday: Path, declared: set[Path]) -> list[Path]:
    """Top-level directories under ``legacy_multiday`` that `migrate.build_plan` does not
    name - the drift a fixed canonical/superseded table cannot see coming, and the entire
    reason this needs to walk the filesystem rather than only read the registry. Per-day
    payloads of the pre-``with_pretrained_baseline`` root sweep (``2024_DOY_*``) are rolled
    into one row by the caller instead of being listed individually.

    A top-level directory counts as declared if it *contains* a declared path, not only if
    it exactly matches one: `migrate.canonical_trees` points the STEC-metrics entry at
    ``with_pretrained_baseline/summary`` rather than the tree's top-level directory, and an
    exact-match check would flag ``with_pretrained_baseline`` itself as unreviewed even
    though it holds the one canonical artifact that tree has.
    """
    if not legacy_multiday.is_dir():
        return []
    return sorted(
        p
        for p in legacy_multiday.iterdir()
        if p.is_dir()
        and not DOY_DIR_PATTERN.match(p.name)
        and not any(d == p or d.is_relative_to(p) for d in declared)
    )


def _tree_row(
    path: Path, status: str, label: str, notes: str = "", name: str | None = None
) -> dict[str, Any]:
    """One inventory row: presence and cheap size from `fingerprint.digest` (stat only,
    directories are summarised by file count/size/mtime rather than hashed), plus whatever
    kind-specific facts the tree's own layout reveals.

    ``name`` defaults to the tree's own directory name, but a caller must override it for
    a canonical entry that points below a tree's top level (`with_pretrained_baseline`'s
    canonical path is its `summary/` subdirectory) - otherwise that row and the
    superseded, top-level tree literally named `summary` share the key "summary".
    """
    tree_digest = fingerprint.digest(path)
    row: dict[str, Any] = {
        "status": status,
        "kind": "analysis_output",
        "name": name if name is not None else path.name,
        "label": label,
        "path": str(path),
        "present": path.exists(),
        "size_gb": round(tree_digest.get("size", 0) / 1024**3, 2),
        "file_count": tree_digest.get("files", ""),
        "notes": notes,
    }
    summary_csv = path / "multiday_summary.csv"
    if summary_csv.exists():
        row.update(summarise_positioning(summary_csv))
    elif day_directories(path):
        row.update(summarise_stec(path))
    return row


def disk_inventory_rows(
    legacy_multiday: Path, legacy_predictions: Path
) -> list[dict[str, Any]]:
    """One row per results tree actually on disk, so `CLAUDE.md`'s canonical-results table
    can be checked against reality instead of trusted. Classification comes from
    `stec.runs.migrate.build_plan`; trees it does not name are reported as "unreviewed"
    rather than silently skipped.
    """
    plan = build_plan(legacy_multiday, legacy_predictions)
    legacy_root = legacy_multiday.parent
    rows = [
        _tree_row(t.path, t.category, t.label, t.notes, name=slug_for(t, legacy_root))
        for t in plan
    ]

    declared_paths = {t.path for t in plan}
    for path in undeclared_trees(legacy_multiday, declared_paths):
        rows.append(_tree_row(path, "unreviewed", path.name))

    root_days = day_directories(legacy_multiday)
    if root_days:
        doys = [int(DOY_DIR_PATTERN.match(p.name).group(1)) for p in root_days]
        total_bytes = sum(fingerprint.digest(p).get("size", 0) for p in root_days)
        rows.append(
            {
                "status": "unreviewed",
                "kind": "stec_evaluation",
                "name": "2024_DOY_* (root level)",
                "label": "pre-with_pretrained_baseline root sweep",
                "path": str(legacy_multiday),
                "present": True,
                "size_gb": round(total_bytes / 1024**3, 2),
                "file_count": "",
                "n_days": len(root_days),
                "date_min": f"2024-{min(doys):03d}",
                "date_max": f"2024-{max(doys):03d}",
                "notes": "per-day payloads of the root sweep, not a named tree",
            }
        )

    if legacy_predictions.is_dir():
        rows.extend(summarise_store_partitions(legacy_predictions))

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("multiday_results/results_manifest")
    )
    parser.add_argument("--legacy-multiday", type=Path, default=paths.LEGACY_MULTIDAY)
    parser.add_argument(
        "--legacy-predictions", type=Path, default=paths.LEGACY_PREDICTIONS
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any consistency problem is found",
    )
    args = parser.parse_args()

    registry.validate()
    stages = registry.STAGES

    manifest = manifest_rows(stages)
    write_csv(
        manifest,
        [
            "deliverable",
            "stage",
            "answers",
            "command",
            "outputs",
            "caveats",
            "has_caveats",
            "last_run_commit",
            "tree_dirty_when_run",
            "recorded_at",
            "provenance",
        ],
        args.output_dir / "manifest.csv",
    )
    write_csv(
        superseded_rows(stages),
        [
            "superseded_artifact",
            "superseded_by_stage",
            "replacement_outputs",
            "still_on_disk",
        ],
        args.output_dir / "superseded.csv",
    )
    write_csv(
        metrics_index_rows(stages),
        ["output", "answers", "stage", "description", "canonical_for", "caveats"],
        args.output_dir / "metrics_index.csv",
    )
    disk_rows = disk_inventory_rows(args.legacy_multiday, args.legacy_predictions)
    write_csv(disk_rows, DISK_INVENTORY_COLUMNS, args.output_dir / "disk_inventory.csv")

    owned = [row for row in manifest if row["deliverable"]]
    print(f"{len(stages)} stages -> {args.output_dir}")
    print(f"  {len(owned)} named deliverable(s), each with exactly one owner:")
    for row in owned:
        print(f"    {row['deliverable']:<40} {row['stage']}")
    print(
        f"  {sum(1 for r in manifest if r['has_caveats'] == 'yes')} stage(s) carry caveats"
    )

    by_status = Counter(row["status"] for row in disk_rows)
    total_gb = sum(row["size_gb"] for row in disk_rows)
    print(
        f"\n  disk inventory: {len(disk_rows)} results tree(s) under "
        f"{args.legacy_multiday.parent}, {total_gb:.0f} GB"
    )
    for status in ("canonical", "unreviewed", "superseded"):
        if by_status.get(status):
            print(f"    {status}: {by_status[status]}")

    unreviewed = [row for row in disk_rows if row["status"] == "unreviewed"]
    disk_problems = [
        f"unreviewed on disk: {row['name']} ({row['size_gb']} GB) at {row['path']}, "
        "not named in CLAUDE.md's canonical-results table"
        for row in unreviewed
    ]

    problems = consistency_problems(stages) + disk_problems
    if problems:
        print(f"\n  {len(problems)} consistency problem(s):")
        for problem in problems:
            print(f"    {problem}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
