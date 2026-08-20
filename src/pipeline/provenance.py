"""What produced a result, recorded next to the decision to reuse it.

One JSON per stage under `.pipeline/`, holding the code version, the command, the input
fingerprint and a digest and row count of every output. It serves two purposes at once:
the runner reads it to decide whether a stage can be skipped, and a reader (or a reviewer
of the published code) reads it to see exactly what produced a number.

Kept in a single directory rather than scattered beside each artifact so the whole
provenance of the paper is one small, publishable tree, and so writing it never disturbs
a results directory that other tooling globs over.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

STATE_DIR = Path(".pipeline")


def code_version() -> dict:
    """Commit the stage ran at, and whether the tree was dirty at the time."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


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
