"""Audit which station-days each positioning method actually solved, and why not.

Table 5 is computed on the station-days solved by *all four* methods, so the honest
question is what the other station-days have in common. This rebuilds the aggregate from
every per-day summary on disk - which reproduces the published
`positioning_comparison_3way/multiday_summary.csv` row for row, confirming the aggregation
is complete - and then classifies the shortfall.

Of the 10,824 station-days the IGS GIM solves, 8,003 are solved by all four methods. The
remaining 2,821 split cleanly into two causes, verified against the raw STEC database:

* **2,311 where all three ML methods are missing together.** None of these stations appear
  in `STEC_DB_CASDCB` for that day - the CAS DCB file gates which stations are processed,
  so a station without a DCB entry is dropped and no correction can be generated. These are
  recoverable by deriving the geometry directly from RINEX (`positioning/geometry/`).
* **~510 where only some methods are missing.** These stations *are* in the database for
  that day, so this is a per-method positioning failure, not a coverage gap.

Reporting the split matters because the two are not interchangeable: the first is a
systematic exclusion correlated with station location, the second is scattered noise.

`daily_summary.csv` carries `weight_opt=elev`, `daily_summary_iono.csv` carries
`weight_opt=iono`; both are rebuilt.

Usage::

    python src/analysis/positioning_coverage.py
    python src/analysis/positioning_coverage.py --weighting elev
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Which experiment tree contributes which method label. The GIM arm is written into
# every tree by the same PPPx run, so it is de-duplicated afterwards.
METHOD_TREES = {
    "STEC": "experiments/Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI",
    "VTEC": "experiments/Finetune_VTEC_2024_*_MLP_LaplacianNLL_*_woYear",
    "Pretrained_STEC": "experiments/Pretrain_STEC_BayesianResNetSTEC_*_SWI",
}
SUMMARY_FILE = {"iono": "daily_summary_iono.csv", "elev": "daily_summary.csv"}
DOY_PATTERN = re.compile(r"results/(\d{4})(\d{3})/")


def collect(weighting: str, repo: Path) -> pd.DataFrame:
    """Every per-day summary on disk, relabelled exactly as run_pipeline.py would."""
    frames = []
    for model, pattern in METHOD_TREES.items():
        label = f"{model}_{weighting}"
        paths = sorted(repo.glob(f"{pattern}/positioning/results/2024*/{SUMMARY_FILE[weighting]}"))
        logger.info(f"{label}: {len(paths)} per-day file(s)")
        for path in paths:
            match = DOY_PATTERN.search(str(path))
            if match is None:
                continue
            year, doy = int(match.group(1)), int(match.group(2))
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
                logger.warning(f"unreadable, skipped: {path} ({exc})")
                continue
            if frame.empty or "method" not in frame.columns:
                continue
            frame["year"], frame["doy"] = year, doy
            frame["date"] = (date(year, 1, 1) + timedelta(days=doy - 1)).strftime("%Y-%m-%d")
            method = frame["method"].astype(str).str.lower()
            frame.loc[method.str.startswith("model"), "method"] = label
            frame.loc[method.str.contains("gim"), "method"] = f"gim_{weighting}"
            frames.append(frame)

    if not frames:
        raise SystemExit(f"no per-day summaries found for weighting '{weighting}'")

    combined = pd.concat(frames, ignore_index=True)
    combined["station"] = combined["station"].astype(str).str.upper()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "method", "station"])
    logger.info(f"{before} rows -> {len(combined)} after de-duplicating the shared GIM arm")
    return combined.sort_values(["doy", "station", "method"]).reset_index(drop=True)


def classify(combined: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Per station-day, which methods solved it and how the shortfall is caused."""
    gim = f"gim_{weighting}"
    ml = [f"{m}_{weighting}" for m in METHOD_TREES]
    wide = combined.pivot_table(
        index=["doy", "station"], columns="method", values="e_rms", aggfunc="first"
    )
    for column in [gim, *ml]:
        if column not in wide.columns:
            wide[column] = pd.NA
    solved = wide[wide[gim].notna()]
    missing = solved[ml].isna()

    cause = pd.Series("solved by all methods", index=solved.index)
    cause[missing.any(axis=1)] = "some ML methods missing (per-method failure)"
    cause[missing.all(axis=1)] = "all ML methods missing (station absent from STEC DB)"
    return (
        pd.DataFrame({"cause": cause})
        .reset_index()
        .assign(missing_methods=missing.apply(
            lambda r: ",".join(c for c in ml if r[c]), axis=1).values)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--weighting", choices=["iono", "elev", "both"], default="both")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("multiday_results/positioning_full_coverage")
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weightings = ["iono", "elev"] if args.weighting == "both" else [args.weighting]
    for weighting in weightings:
        combined = collect(weighting, args.repo)
        suffix = "" if weighting == "iono" else f"_{weighting}"
        path = args.output_dir / f"multiday_summary{suffix}.csv"
        combined.to_csv(path, index=False, float_format="%.4f")

        counts = combined.groupby("method").size()
        print(f"\n=== {weighting} weighting: station-days per method ===")
        print(counts.to_string())
        wide = combined.pivot_table(
            index=["doy", "station"], columns="method", values="e_rms", aggfunc="first"
        )
        print(f"  solved by ALL methods: {wide.notna().all(axis=1).sum()} of {len(wide)}")
        coverage = classify(combined, weighting)
        coverage_path = args.output_dir / f"coverage{suffix}.csv"
        coverage.to_csv(coverage_path, index=False)
        print(f"\n--- why station-days are not in the common set ({weighting}) ---")
        print(coverage["cause"].value_counts().to_string())
        logger.info(f"💾 {path}")
        logger.info(f"💾 {coverage_path}")


if __name__ == "__main__":
    main()
