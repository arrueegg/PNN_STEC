"""Report — and, only when told to, delete — results that are provably redundant.

640 GB of results, of which 302 GB is `detailed_predictions.csv`: the legacy
per-observation format that the partitioned parquet store under `predictions/`
replaced. CLAUDE.md already calls those CSVs "not the source of truth", so the
question is not whether they are redundant but which copies can go without
losing a record that nothing else holds.

Candidates are graded, because the risk is not uniform:

  tier 1  Sweep trees that exist only to have populated the prediction store.
          Every day they contain is verified present in the store before the
          tier is offered, and the store's schema is a strict superset of the
          CSV's. Nothing reads them.

  tier 2  The per-observation CSVs inside superseded evaluation trees. The
          metrics summaries, comparison logs and `temp_config_*.yaml` files
          stay, so what CLAUDE.md wants preserved — the record of earlier
          configurations — survives; only the bulk goes.

  tier 3  The per-observation CSVs inside canonical trees. Blocked: the
          elevation binning in `multiday_evaluation.py` still reads them on a
          `--summary_only` rerun. Reported so the size is visible, never
          offered for deletion here.

Deleting is opt-in per tier and prints what it would remove first. Nothing in
this module removes a metrics file, a config, a plot or a `.pos` solution.

Usage::

    python src/analysis/cleanup_audit.py                    # report only
    python src/analysis/cleanup_audit.py --apply --tier 1   # actually delete
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("multiday_results")
STORE_DIR = Path("predictions")
LEGACY_PREDICTIONS = "detailed_predictions.csv"
DOY_DIR_PATTERN = re.compile(r"^2024_DOY_(\d{3})$")

# Trees that were only ever a vehicle for filling the prediction store.
STORE_SWEEP_TREES = ["store_sweep_full", "store_sweep_priority", "store_sweep_vtec_unc"]
# Superseded evaluation trees, per CLAUDE.md. The root-level 2024_DOY_* sweep is
# handled separately because it also holds the temp_config files that record
# what the canonical runs used, and those must survive.
SUPERSEDED_TREES = ["mao_evaluation"]
CANONICAL_TREES = ["with_pretrained_baseline"]
BLOCKED_REASON = (
    "read by the elevation binning in src/multiday_evaluation.py --summary_only; "
    "delete only after that path reads the parquet store"
)


def store_has_day(doy: int, store_dir: Path = STORE_DIR) -> bool:
    """Is this 2024 day present in the authoritative store for the own test set?"""
    return (store_dir / f"finetuned_stec/own/year=2024/doy={doy:03d}.parquet").exists()


def legacy_csvs(root: Path) -> list[Path]:
    return sorted(root.rglob(LEGACY_PREDICTIONS))


def day_directories(parent: Path) -> list[Path]:
    return sorted(p for p in parent.glob("2024_DOY_*") if DOY_DIR_PATTERN.match(p.name))


def collect_candidates(results_dir: Path, store_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for name in STORE_SWEEP_TREES:
        tree = results_dir / name
        if not tree.is_dir():
            continue
        days = day_directories(tree)
        doys = [int(DOY_DIR_PATTERN.match(p.name).group(1)) for p in days]
        covered = [doy for doy in doys if store_has_day(doy, store_dir)]
        files = legacy_csvs(tree)
        rows.append(
            {
                "tier": 1,
                "target": str(tree),
                "what": LEGACY_PREDICTIONS,
                "n_files": len(files),
                "size_gb": round(sum(f.stat().st_size for f in files) / 1024**3, 2),
                "safe": len(covered) == len(doys),
                "evidence": f"{len(covered)}/{len(doys)} days present in the prediction store",
                "kept": "metrics_summary.csv, comparison logs",
            }
        )

    for name in SUPERSEDED_TREES:
        tree = results_dir / name
        if not tree.is_dir():
            continue
        files = legacy_csvs(tree)
        rows.append(
            {
                "tier": 2,
                "target": str(tree),
                "what": LEGACY_PREDICTIONS,
                "n_files": len(files),
                "size_gb": round(sum(f.stat().st_size for f in files) / 1024**3, 2),
                "safe": True,
                "evidence": "tree marked superseded in CLAUDE.md; no script reads it",
                "kept": "metrics_summary.csv, comparison_summary.txt, logs",
            }
        )

    root_days = day_directories(results_dir)
    if root_days:
        files = [f for day in root_days for f in legacy_csvs(day)]
        rows.append(
            {
                "tier": 2,
                "target": f"{results_dir}/2024_DOY_* (root level)",
                "what": LEGACY_PREDICTIONS,
                "n_files": len(files),
                "size_gb": round(sum(f.stat().st_size for f in files) / 1024**3, 2),
                "safe": True,
                "evidence": "superseded by with_pretrained_baseline",
                "kept": "temp_config_*.yaml and training logs (cited by CLAUDE.md)",
            }
        )

    for name in CANONICAL_TREES:
        tree = results_dir / name
        if not tree.is_dir():
            continue
        files = legacy_csvs(tree)
        rows.append(
            {
                "tier": 3,
                "target": str(tree),
                "what": LEGACY_PREDICTIONS,
                "n_files": len(files),
                "size_gb": round(sum(f.stat().st_size for f in files) / 1024**3, 2),
                "safe": False,
                "evidence": BLOCKED_REASON,
                "kept": "everything, for now",
            }
        )

    return pd.DataFrame(rows)


def delete_tier(candidates: pd.DataFrame, tier: int, results_dir: Path) -> int:
    """Remove the legacy CSVs of one tier. Returns bytes freed."""
    selected = candidates[(candidates["tier"] == tier) & candidates["safe"]]
    if selected.empty:
        logger.warning(
            "tier %d has no candidate cleared as safe; nothing deleted", tier
        )
        return 0

    freed = 0
    for target in selected["target"]:
        root = results_dir if target.endswith("(root level)") else Path(target)
        scope = (
            day_directories(results_dir) if target.endswith("(root level)") else [root]
        )
        for base in scope:
            for path in legacy_csvs(base):
                freed += path.stat().st_size
                path.unlink()
        logger.info("cleared %s", target)
    return freed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--store_dir", type=Path, default=STORE_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "cleanup_candidates.csv"
    )
    parser.add_argument("--apply", action="store_true", help="actually delete")
    parser.add_argument(
        "--tier", type=int, action="append", help="tier(s) to delete with --apply"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    candidates = collect_candidates(args.results_dir, args.store_dir)
    candidates.to_csv(args.output, index=False)
    print(candidates.to_string(index=False))
    print(
        f"\nreclaimable now (tiers 1-2): {candidates[candidates.tier < 3].size_gb.sum():.1f} GB"
    )
    print(
        f"blocked (tier 3):            {candidates[candidates.tier == 3].size_gb.sum():.1f} GB"
    )

    if not args.apply:
        print(f"\nreport only; written to {args.output}")
        return
    if not args.tier:
        parser.error("--apply needs at least one --tier")
    if 3 in args.tier:
        parser.error("tier 3 is blocked; see the evidence column")

    freed = sum(delete_tier(candidates, tier, args.results_dir) for tier in args.tier)
    print(f"\nfreed {freed / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
