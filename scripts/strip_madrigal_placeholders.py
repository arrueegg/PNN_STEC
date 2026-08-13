"""Remove placeholder arc columns from Madrigal files in the prediction store.

The Madrigal loader has no satellite identity: it fills `sat` with a single
constant, and leaves `slipc` and `gfphase` at zero. Storing those invites a
per-arc groupby that returns nonsense while looking like it worked.

`compare_stec_vtec_gim.write_prediction_store` now drops them at write time, but
files written before that fix still carry them. This repairs those in place by
rewriting the parquet without the offending columns - no inference needed, since
nothing else about the data changes.

Idempotent: files that are already clean are left untouched, so it can be re-run
after a sweep finishes to catch stragglers.

Usage::

    python scripts/strip_madrigal_placeholders.py --dry_run
    python scripts/strip_madrigal_placeholders.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PLACEHOLDER_COLUMNS = ("sat", "slipc", "gfphase")


def is_placeholder(frame: pd.DataFrame, column: str) -> bool:
    """A column is a placeholder here only if it carries no information at all."""
    return column in frame.columns and frame[column].nunique(dropna=False) <= 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store_root", type=Path, default=Path("predictions"))
    parser.add_argument("--dataset", type=str, default="madrigal")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    paths = sorted(args.store_root.glob(f"*/{args.dataset}/year=*/doy=*.parquet"))
    if not paths:
        logger.warning(f"⚠️  No {args.dataset} files under {args.store_root}")
        return

    repaired, clean, skipped = 0, 0, []
    for path in paths:
        frame = pd.read_parquet(path)
        drop = [c for c in PLACEHOLDER_COLUMNS if is_placeholder(frame, c)]
        present = [c for c in PLACEHOLDER_COLUMNS if c in frame.columns]

        # Refuse to touch a column that actually varies: that would mean the
        # loader started carrying real satellite identity and this repair is
        # no longer the right thing to do.
        informative = [c for c in present if c not in drop]
        if informative:
            skipped.append((path.name, informative))
            continue
        if not drop:
            clean += 1
            continue

        logger.info(
            f"{'would strip' if args.dry_run else 'stripping'} {drop} from {path}"
        )
        if not args.dry_run:
            frame.drop(columns=drop).to_parquet(path, index=False, compression="snappy")
        repaired += 1

    logger.info(f"repaired {repaired}, already clean {clean}, of {len(paths)} file(s)")
    if skipped:
        logger.warning(
            f"⚠️  Left alone because the columns carry real values: {skipped[:5]}"
        )


if __name__ == "__main__":
    main()
