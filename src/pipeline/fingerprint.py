"""Cheap, stable digests of a stage's inputs, so unchanged work can be skipped.

Content-hashing is right for a 200 kB metric CSV and wrong for a 103 GB HDF5 or a
242-day parquet tree, so the rule is size-dependent: small files are hashed, large files
and directories are summarised by the facts that actually change when they are rewritten
(size, modification time, file count). A summary can in principle miss an edit that
preserves all three; that is an accepted trade for not reading 640 GB on every invocation,
and `--force` exists for when a stage must be rerun regardless.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Above this, summarise rather than read. Covers every metric CSV in the repo while
# excluding the prediction store, the raw HDF5 days and the checkpoints.
HASH_LIMIT_BYTES = 64 * 1024 * 1024


def _file_digest(path: Path) -> dict:
    stat = path.stat()
    if stat.st_size <= HASH_LIMIT_BYTES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"kind": "file", "sha256": digest, "size": stat.st_size}
    return {"kind": "file-summary", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _tree_digest(path: Path) -> dict:
    count = total = 0
    newest = 0
    for entry in path.rglob("*"):
        if not entry.is_file():
            continue
        stat = entry.stat()
        count += 1
        total += stat.st_size
        newest = max(newest, stat.st_mtime_ns)
    return {"kind": "tree", "files": count, "size": total, "mtime_ns": newest}


def digest(path: Path) -> dict:
    """Digest of one declared input, or a marker that it is absent."""
    if not path.exists():
        return {"kind": "missing"}
    return _tree_digest(path) if path.is_dir() else _file_digest(path)


def fingerprint(inputs: list[str], params: dict) -> str:
    """One hash standing for every input and parameter of a stage."""
    payload = {
        "inputs": {name: digest(Path(name)) for name in sorted(inputs)},
        "params": params,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def describe(inputs: list[str]) -> dict[str, dict]:
    """Per-input digests, kept in the provenance record for debugging a stale skip."""
    return {name: digest(Path(name)) for name in sorted(inputs)}
