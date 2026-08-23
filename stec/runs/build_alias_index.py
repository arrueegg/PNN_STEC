"""Map every existing experiment directory onto a run_id.

The equivalence diagnostics have to locate checkpoints that were produced before run_ids
existed, so this index is a prerequisite for them rather than a tidying step. It also
answers the question the directory names cannot: whether two configurations that produced
different directories are in fact identical, and whether any single directory name was
reused by two different configurations.

    python -m stec.runs.build_alias_index
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from ..config import paths
from .identity import build_index

DEFAULT_OUTPUT = paths.ARTIFACT_ROOT / "runs" / "alias_index.csv"

FIELDS = [
    "exp_name",
    "run_id",
    "status",
    "mode",
    "target",
    "model_type",
    "year",
    "doy",
    "random_seed",
    "checkpoints",
    "checkpoint",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=paths.LEGACY_EXPERIMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.experiments.is_dir():
        print(f"no experiments directory at {args.experiments}", file=sys.stderr)
        return 2

    records = build_index(args.experiments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    ok = [r for r in records if r["status"] == "ok"]
    unreadable = [r for r in records if r["status"] != "ok"]

    # A run_id claimed by two directories means the two configurations are identical in
    # everything that affects the result - so one of them is a duplicate run, and the
    # directory names are distinguishing something that does not matter.
    duplicate_ids = {
        rid: n for rid, n in Counter(r["run_id"] for r in ok).items() if n > 1
    }
    # The reverse - one directory name, two configurations - cannot be seen here (a name
    # is a directory), but a name collision would have been silently overwritten on disk.

    print(f"indexed {len(records)} experiment director(ies) -> {args.output}")
    print(f"  {len(ok)} with a recoverable config")
    print(f"  {sum(r['checkpoints'] for r in ok)} checkpoint(s)")
    if unreadable:
        print(f"  {len(unreadable)} without a usable config.yaml:")
        for record in unreadable[:5]:
            print(f"    {record['status']}: {record['exp_name'][:70]}")
        if len(unreadable) > 5:
            print(f"    ... and {len(unreadable) - 5} more")
    if duplicate_ids:
        print(f"  {len(duplicate_ids)} run_id(s) shared by more than one directory:")
        for rid, count in list(duplicate_ids.items())[:5]:
            print(f"    {rid} x{count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
