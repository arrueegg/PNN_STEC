"""What produced a result, recorded next to the decision to reuse it.

One JSON per stage under `.pipeline/`, holding the code version, the command, the input
fingerprint and a digest and row count of every output. It serves two purposes at once:
the runner reads it to decide whether a stage can be skipped, and a reader (or a reviewer
of the published code) reads it to see exactly what produced a number.

Kept in a single directory rather than scattered beside each artifact so the whole
provenance of the paper is one small, publishable tree, and so writing it never disturbs a
results directory that other tooling globs over.

Two things are written *beside* the artifact rather than here, because they must survive
being read out of context:

* a caveat sidecar, so a CSV cannot be lifted into a table without the condition that
  makes it valid;
* a superseded marker on the older artifact a stage replaces, so a stale number announces
  itself rather than sitting on disk looking current.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(".pipeline")

CAVEAT_SUFFIX = ".caveats.json"
SUPERSEDED_SUFFIX = ".superseded.json"


def code_version() -> dict:
    """Commit the stage ran at, and whether the tree was dirty at the time."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def output_record(path: Path) -> dict:
    """Digest, size and - for a CSV - the row count, which is the number that matters."""
    if not path.exists():
        return {"present": False}
    record = {"present": True, "size": path.stat().st_size}
    if path.is_file():
        data = path.read_bytes()
        record["sha256"] = hashlib.sha256(data).hexdigest()
        if path.suffix == ".csv":
            # -1 for the header; a CSV that came out empty is the failure this catches.
            record["rows"] = max(0, data.count(b"\n") - 1)
    return record


def path_for(stage: str) -> Path:
    return STATE_DIR / f"{stage}.json"


def load(stage: str) -> dict | None:
    path = path_for(stage)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def save(stage: str, record: dict) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    record = {**record, "recorded_at": datetime.now(timezone.utc).isoformat()}
    path = path_for(stage)
    path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str))
    return path


def write_caveats(output: Path, stage: str, caveats: list[str]) -> Path | None:
    """Record, beside the artifact, the conditions under which it must not be read.

    Written for every declared output, including when the list is empty: the presence of
    the sidecar is what lets a consumer distinguish "no caveats" from "nobody recorded
    any", and only the first is safe.
    """
    if not output.exists():
        return None
    sidecar = output.with_suffix(output.suffix + CAVEAT_SUFFIX)
    sidecar.write_text(
        json.dumps(
            {
                "artifact": str(output),
                "produced_by": stage,
                "caveats": caveats,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return sidecar


def mark_superseded(older: Path, by_stage: str, replacement: list[str]) -> Path | None:
    """Stamp an older artifact as replaced, without deleting it.

    Nothing is deleted - the superseded trees are the only record of earlier
    configurations, and storage is not a constraint here. But a number that is no longer
    current should say so where someone reading it will look.
    """
    if not older.exists():
        return None
    marker_dir = older if older.is_dir() else older.parent
    name = "TREE" if older.is_dir() else older.name
    marker = marker_dir / f"{name}{SUPERSEDED_SUFFIX}"
    marker.write_text(
        json.dumps(
            {
                "superseded": str(older),
                "superseded_by_stage": by_stage,
                "replacement_outputs": replacement,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return marker
