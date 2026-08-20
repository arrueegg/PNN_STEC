"""Where every reported number comes from, generated from the registry.

`CLAUDE.md` currently carries a hand-maintained table titled "Which results are canonical",
listing which result trees to trust and which are superseded. That table exists because the
filesystem does not say: `multiday_results/` holds `summary/`, `summary_May/`,
`summary_122_250/`, `with_pretrained_baseline/summary/` and several positioning trees, and
nothing on disk distinguishes the one that backs the paper from the four that do not.

A hand-maintained table is the wrong shape for that job. It is written once, consulted by
people who already know the answer, and silently wrong the moment a stage changes. The
registry already knows: which stage owns which deliverable, what each writes, what it
supersedes, and what caveats attach to reading it. This turns that into three files:

* `manifest.csv`      - deliverable, owning stage, outputs, caveats, provenance record
* `superseded.csv`    - artifacts a stage has replaced, and what replaced them
* `metrics_index.csv` - every metric CSV mapped to the reviewer comment it answers

The last one replaces `multiday_results/revision_metrics_index.csv`, which was maintained
by hand alongside the analyses it describes.

Running this is also a consistency check, not only a report: a deliverable with no owner,
or an output whose provenance record is missing, is reported rather than omitted, because
"absent from the manifest" and "absent from the pipeline" must not look the same.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from ..pipeline import registry
from ..pipeline.stage import Stage

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("multiday_results/results_manifest")
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

    owned = [row for row in manifest if row["deliverable"]]
    print(f"{len(stages)} stages -> {args.output_dir}")
    print(f"  {len(owned)} named deliverable(s), each with exactly one owner:")
    for row in owned:
        print(f"    {row['deliverable']:<40} {row['stage']}")
    print(
        f"  {sum(1 for r in manifest if r['has_caveats'] == 'yes')} stage(s) carry caveats"
    )

    problems = consistency_problems(stages)
    if problems:
        print(f"\n  {len(problems)} consistency problem(s):")
        for problem in problems:
            print(f"    {problem}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
