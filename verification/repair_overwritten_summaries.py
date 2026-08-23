"""Rebuild the per-day positioning summaries the recovery sweep overwrote.

`save_daily_summary` writes its frame with `to_csv` and no merge, and `recover_day.py`
calls it with only the handful of stations it has just recovered. Every day the sweep
touched therefore lost the rows for every station that had already been solved: 91
`daily_summary_iono.csv` files fell from roughly 74-81 rows to between 3 and 10.

The loss is recoverable because the summaries are *derived*. Their source is the `.pos`
solutions, which the sweep never touched, so rebuilding needs no PPPx run, no products
download and no solver - only the same arithmetic that produced the rows in the first
place. That arithmetic is `stec.positioning.metrics`, which Gate E has already checked
against the recorded values on 96 station-days and matched to within the CSV's own
`%.4f` rounding floor.

Safety, because this is the one operation in the rebuild that writes into the primary
checkout: dry run unless `--apply`, every file backed up beside itself before it is
replaced, and a refusal to write a summary that would come out *smaller* than the one
already on disk - shrinking is the failure being repaired, so a repair that shrinks a
file is a bug in the repair.

    python verification/repair_overwritten_summaries.py            # dry run
    python verification/repair_overwritten_summaries.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.config import paths  # noqa: E402
from stec.positioning import metrics as pm  # noqa: E402

# The iono arm pairs the uncertainty-weighted model solution with the shared GIM arm.
IONO_SUMMARY = "daily_summary_iono.csv"
IONO_METHODS = (("model_iono", "model_iono"), ("gim", "gim"))

BACKUP_SUFFIX = ".pre_repair"


def summary_dirs(root: Path, since: str) -> list[Path]:
    """Every results directory whose iono summary was rewritten after `since`."""
    cutoff = datetime.fromisoformat(since).timestamp()
    found = [
        path.parent
        for path in root.glob("*/positioning/results/*/" + IONO_SUMMARY)
        if path.stat().st_mtime >= cutoff
    ]
    return sorted(found)


def rebuild(results_dir: Path) -> pd.DataFrame | None:
    """Recompute the iono summary for one results directory from its .pos files."""
    stamp = results_dir.name
    year, doy = int(stamp[:4]), int(stamp[4:])

    snx = (
        results_dir.parents[1]
        / "evaluation"
        / stamp
        / "products"
        / f"IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_CRD.SNX"
    )
    frames = []
    for subdir, method in IONO_METHODS:
        if not (results_dir / subdir).is_dir():
            continue
        frame = pm.aggregate_daily_metrics(
            results_dir / subdir,
            year=year,
            doy=doy,
            method_name=method,
            snx_file=snx if snx.exists() else None,
        )
        if frame is not None and len(frame):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=paths.LEGACY_EXPERIMENTS)
    parser.add_argument(
        "--since",
        default="2026-08-21 00:00",
        help="rebuild summaries rewritten after this",
    )
    parser.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    targets = summary_dirs(args.experiments, args.since)
    if args.limit:
        targets = targets[: args.limit]
    print(
        f"{'APPLY' if args.apply else 'DRY RUN'}: {len(targets)} damaged summary/summaries\n"
    )

    repaired = skipped = refused = 0
    for results_dir in targets:
        summary_path = results_dir / IONO_SUMMARY
        before = max(0, sum(1 for _ in summary_path.open()) - 1)

        frame = rebuild(results_dir)
        if frame is None or frame.empty:
            print(f"  SKIP    {results_dir.name}  no .pos files could be reduced")
            skipped += 1
            continue

        after = len(frame)
        label = f"{results_dir.parent.parent.parent.name[:34]:34s} {results_dir.name}"
        if after < before:
            # Shrinking is the damage; a repair that shrinks is a bug in the repair.
            print(f"  REFUSE  {label}  {before} -> {after} rows, would shrink")
            refused += 1
            continue

        print(
            f"  {'repair' if args.apply else 'would':7s} {label}  {before} -> {after} rows"
        )
        if args.apply:
            backup = summary_path.with_suffix(summary_path.suffix + BACKUP_SUFFIX)
            if not backup.exists():
                shutil.copy2(summary_path, backup)
            frame.to_csv(summary_path, index=False, float_format="%.4f")
        repaired += 1

    print(f"\n  {repaired} repairable, {skipped} skipped, {refused} refused")
    if not args.apply:
        print("  nothing written - rerun with --apply")
    else:
        print(f"  originals kept beside each file as *{BACKUP_SUFFIX}")
    print(f"  recorded at {datetime.now(timezone.utc).isoformat()}")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())
