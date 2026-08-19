"""Inventory every results tree, so comparability is a lookup rather than a dig.

The repository holds 1588 experiment directories and several parallel results
trees, most of them superseded. Nothing recorded which of them are comparable
with which, and that is what allowed the manuscript's Table 5 to compare four
methods over four different station-day populations for a full review cycle.

The fix is an index, not a reorganisation: moving directories would not have
made a population mismatch visible, whereas a per-arm station-day count sitting
next to the run does. Each row is one results tree with the facts needed to
decide whether two numbers may be quoted side by side — weighting scheme, date
span, arms present, and how many station-days each arm actually solved.

Status is curated rather than inferred; it mirrors the "Which results are
canonical" table of CLAUDE.md, which stays the prose source of truth.

Usage::

    python src/analysis/results_manifest.py
"""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("multiday_results")
STORE_DIR = Path("predictions")

# Curated from CLAUDE.md. Anything not listed is reported as "unreviewed"
# rather than guessed at, so a new tree shows up as needing a decision.
CANONICAL = {
    "with_pretrained_baseline": "STEC metrics behind Tables 3 and 4",
    "positioning_comparison_3way": "positioning, Figures 12/13/A1/A2 and Table 5",
    "positioning_20260216_2052": "weighting ablation, all six arms (R1.5)",
}
SUPERSEDED = {
    "summary": "with_pretrained_baseline",
    "summary_May": "with_pretrained_baseline",
    "summary_122_250": "with_pretrained_baseline",
    "mao_evaluation": "with_pretrained_baseline",
    "positioning": "positioning_comparison_3way",
    "positioning_iono": "positioning_comparison_3way",
    "positioning_mean": "positioning_comparison_3way",
    "positioning_snx": "positioning_comparison_3way",
}
SUPERSEDED_PREFIXES = {"positioning_2026": "positioning_comparison_3way"}
# Trees whose name says what they are and that no table depends on.
ANALYSIS_OUTPUT_HINT = "revision analysis output (CSV only)"


def directory_bytes(path: Path) -> int:
    """Size on disk via ``du``; far faster than walking from Python here."""
    try:
        out = subprocess.run(
            ["du", "-sb", str(path)], capture_output=True, text=True, check=True
        )
        return int(out.stdout.split()[0])
    except (subprocess.CalledProcessError, ValueError, IndexError):
        logger.warning("could not size %s", path)
        return 0


def classify(name: str) -> tuple[str, str]:
    """Return (status, superseded_by) for a tree name."""
    if name in CANONICAL:
        return "canonical", ""
    if name in SUPERSEDED:
        return "superseded", SUPERSEDED[name]
    for prefix, replacement in SUPERSEDED_PREFIXES.items():
        if name.startswith(prefix):
            return "superseded", replacement
    return "unreviewed", ""


def summarise_positioning(summary_csv: Path) -> dict[str, object]:
    """Arms, span and per-arm station-day counts for a positioning tree."""
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


DOY_DIR_PATTERN = re.compile(r"^2024_DOY_(\d{3})$")


def day_directories(parent: Path) -> list[Path]:
    """Per-day payload directories, ignoring ad-hoc retries like ``_try1``."""
    return sorted(p for p in parent.glob("2024_DOY_*") if DOY_DIR_PATTERN.match(p.name))


def summarise_stec(tree: Path) -> dict[str, object]:
    """Span and day count for a STEC evaluation tree."""
    day_dirs = day_directories(tree)
    doys = [int(DOY_DIR_PATTERN.match(p.name).group(1)) for p in day_dirs]
    has_summary = (tree / "summary").is_dir()
    return {
        "kind": "stec_evaluation",
        "arms": "stec;vtec;gim" + (";pretrained" if "pretrained" in tree.name else ""),
        "weighting": "",
        "n_days": len(doys),
        "date_min": f"2024-{min(doys):03d}" if doys else "",
        "date_max": f"2024-{max(doys):03d}" if doys else "",
        "n_stations": "",
        "station_days_per_arm": "",
        "n_rows": "",
        "notes": "summary/ present" if has_summary else "no summary/",
    }


def summarise_store(store: Path) -> list[dict[str, object]]:
    """One row per (model variant, dataset) partition of the prediction store."""
    rows = []
    for variant_dir in sorted(p for p in store.iterdir() if p.is_dir()):
        for dataset_dir in sorted(p for p in variant_dir.iterdir() if p.is_dir()):
            files = sorted(dataset_dir.glob("year=*/doy=*.parquet"))
            # The pretrained variant spans several years, so a bare day count
            # would read as a 2024 span it does not have.
            years = sorted({int(f.parent.name.split("=")[-1]) for f in files})
            doys_2024 = [
                int(f.stem.split("=")[-1])
                for f in files
                if f.parent.name == "year=2024"
            ]
            rows.append(
                {
                    "name": f"{variant_dir.name}/{dataset_dir.name}",
                    "path": str(dataset_dir),
                    "kind": "prediction_store",
                    "status": "canonical",
                    "superseded_by": "",
                    "arms": "per-observation predictions",
                    "weighting": "",
                    "n_days": len(files),
                    "date_min": f"2024-{min(doys_2024):03d}" if doys_2024 else "",
                    "date_max": f"2024-{max(doys_2024):03d}" if doys_2024 else "",
                    "n_stations": "",
                    "station_days_per_arm": "",
                    "n_rows": "",
                    "size_gb": round(directory_bytes(dataset_dir) / 1024**3, 2),
                    "notes": (
                        "authoritative per-observation results; "
                        f"years {years[0]}-{years[-1]}, {len(doys_2024)} days in 2024"
                        if years
                        else "empty"
                    ),
                }
            )
    return rows


def build_manifest(results_dir: Path, store_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for tree in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if tree.name.startswith("2024_DOY_"):
            continue  # per-day payloads of the root-level sweep, rolled up below
        status, superseded_by = classify(tree.name)
        summary_csv = tree / "multiday_summary.csv"
        if summary_csv.exists():
            entry = summarise_positioning(summary_csv)
        elif day_directories(tree):
            entry = summarise_stec(tree)
        else:
            entry = {"kind": "analysis_output", "notes": ANALYSIS_OUTPUT_HINT}
        entry.update(
            {
                "name": tree.name,
                "path": str(tree),
                "status": status,
                "superseded_by": superseded_by,
                "size_gb": round(directory_bytes(tree) / 1024**3, 2),
            }
        )
        rows.append(entry)

    root_days = day_directories(results_dir)
    if root_days:
        doys = [int(DOY_DIR_PATTERN.match(p.name).group(1)) for p in root_days]
        rows.append(
            {
                "name": "2024_DOY_* (root level)",
                "path": str(results_dir),
                "kind": "stec_evaluation",
                "status": "superseded",
                "superseded_by": "with_pretrained_baseline",
                "arms": "stec;vtec;gim",
                "n_days": len(root_days),
                "date_min": f"2024-{min(doys):03d}",
                "date_max": f"2024-{max(doys):03d}",
                "size_gb": round(
                    sum(directory_bytes(p) for p in root_days) / 1024**3, 2
                ),
                "notes": "pre-pretrained-baseline sweep; feeds the superseded summary/",
            }
        )

    if store_dir.is_dir():
        rows.extend(summarise_store(store_dir))

    columns = [
        "name",
        "kind",
        "status",
        "superseded_by",
        "weighting",
        "n_days",
        "date_min",
        "date_max",
        "n_stations",
        "arms",
        "station_days_per_arm",
        "n_rows",
        "size_gb",
        "path",
        "notes",
    ]
    manifest = pd.DataFrame(rows)
    for column in columns:
        if column not in manifest:
            manifest[column] = ""
    manifest = manifest[columns].fillna("")
    order = {"canonical": 0, "unreviewed": 1, "superseded": 2}
    return manifest.sort_values(
        ["status", "kind", "name"],
        key=lambda s: s.map(order).fillna(s) if s.name == "status" else s,
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--store_dir", type=Path, default=STORE_DIR)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / "runs_manifest.csv"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    manifest = build_manifest(args.results_dir, args.store_dir)
    manifest.to_csv(args.output, index=False)
    print(f"{len(manifest)} results trees, {manifest['size_gb'].sum():.0f} GB")
    print(manifest["status"].value_counts().to_string())
    print(f"written to {args.output}")


if __name__ == "__main__":
    main()
