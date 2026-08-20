"""Audit which station-days each positioning method actually solved, and why not.

Ported from ``src/analysis/positioning_coverage.py`` in the live PNN_STEC checkout.
Evidence for reviewer comment R1.5: Table 5 is computed on the station-days solved by
*all four* methods, so the honest question is what the other station-days have in
common. This rebuilds the aggregate from every per-day summary on disk under
``stec.config.paths.LEGACY_EXPERIMENTS`` and classifies the shortfall.

Of the 10,824 station-days the IGS GIM solves, 8,003 are solved by all four methods.
The remaining 2,821 split into two causes:

* **2,311 where all three ML methods are missing together.** None of these stations
  appear in ``STEC_DB_CASDCB`` for that day - the CAS DCB file gates which stations are
  processed, so a station without a DCB entry is dropped and no correction can be
  generated. This was verified separately against the raw database; this script itself
  only observes which methods have a result row for a station-day, it does not read the
  database, so the "absent from STEC DB" label is an established interpretation of the
  cause, not something re-derived here.
* **510 where only some methods are missing.** These stations *are* present in the
  database for that day, so this is a per-method PPPx failure, not a coverage gap - the
  specific cause is unexplained.

The distinction is load-bearing: the first is a systematic exclusion correlated with
station location, the second is scattered noise, and conflating them into one "missing"
bucket would hide that difference.

``daily_summary.csv`` carries ``weight_opt=elev``, ``daily_summary_iono.csv`` carries
``weight_opt=iono``; both are rebuilt.

Usage::

    python -m stec.analysis.positioning_coverage
    python -m stec.analysis.positioning_coverage --weighting elev
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..config import paths

logger = logging.getLogger(__name__)

# Which experiment tree contributes which method label, relative to LEGACY_EXPERIMENTS.
# The GIM arm is written into every tree by the same PPPx run, so it is de-duplicated
# afterwards.
METHOD_TREES = {
    "STEC": "Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI",
    "VTEC": "Finetune_VTEC_2024_*_MLP_LaplacianNLL_*_woYear",
    "Pretrained_STEC": "Pretrain_STEC_BayesianResNetSTEC_*_SWI",
}
SUMMARY_FILE = {"iono": "daily_summary_iono.csv", "elev": "daily_summary.csv"}
DOY_PATTERN = re.compile(r"results/(\d{4})(\d{3})/")

# The causes classify() assigns. Named as constants so main() and the tests compare
# against the same strings rather than retyping them.
SOLVED_BY_ALL = "solved by all methods"
SOME_ML_MISSING = "some ML methods missing (per-method failure)"
ALL_ML_MISSING = "all ML methods missing (station absent from STEC DB)"
UNCLASSIFIED = "unclassified"

DEFAULT_OUTPUT_DIR = Path("multiday_results/positioning_coverage_rebuilt")


def collect(weighting: str, experiments_root: Path) -> pd.DataFrame:
    """Every per-day summary on disk, relabelled exactly as run_pipeline.py would."""
    frames = []
    for model, pattern in METHOD_TREES.items():
        label = f"{model}_{weighting}"
        found = sorted(
            experiments_root.glob(
                f"{pattern}/positioning/results/2024*/{SUMMARY_FILE[weighting]}"
            )
        )
        logger.info(f"{label}: {len(found)} per-day file(s)")
        for path in found:
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
            frame["date"] = (date(year, 1, 1) + timedelta(days=doy - 1)).strftime(
                "%Y-%m-%d"
            )
            method = frame["method"].astype(str).str.lower()
            frame.loc[method.str.startswith("model"), "method"] = label
            frame.loc[method.str.contains("gim"), "method"] = f"gim_{weighting}"
            frames.append(frame)

    if not frames:
        raise SystemExit(
            f"no per-day summaries found for weighting '{weighting}' under "
            f"{experiments_root}"
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["station"] = combined["station"].astype(str).str.upper()
    before = len(combined)
    combined = combined.drop_duplicates(subset=["date", "method", "station"])
    logger.info(
        f"{before} rows -> {len(combined)} after de-duplicating the shared GIM arm"
    )
    return combined.sort_values(["doy", "station", "method"]).reset_index(drop=True)


def classify(combined: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Per station-day the GIM solved, which methods solved it and why the rest didn't.

    Restricted to station-days the GIM itself solved (``wide[gim].notna()``) - that is
    the population the 8,003 / 2,311 / 510 split is defined over. ``cause`` starts as
    ``UNCLASSIFIED`` rather than defaulting to "solved" so a station-day only reads as
    solved when the counts say so explicitly. The three conditions below (zero ML
    methods missing / some missing / all missing) are mutually exclusive and exhaustive
    for any non-empty ``ml`` - ``UNCLASSIFIED`` should therefore never appear in
    practice, but callers get an explicit count instead of a silent drop if it does.
    """
    gim = f"gim_{weighting}"
    ml = [f"{m}_{weighting}" for m in METHOD_TREES]
    assert ml, "METHOD_TREES must be non-empty for the missing-count logic below"

    wide = combined.pivot_table(
        index=["doy", "station"], columns="method", values="e_rms", aggfunc="first"
    )
    for column in [gim, *ml]:
        if column not in wide.columns:
            wide[column] = pd.NA
    solved = wide[wide[gim].notna()]
    missing = solved[ml].isna()
    missing_count = missing.sum(axis=1)

    cause = pd.Series(UNCLASSIFIED, index=solved.index)
    cause[missing_count == 0] = SOLVED_BY_ALL
    cause[(missing_count > 0) & (missing_count < len(ml))] = SOME_ML_MISSING
    cause[missing_count == len(ml)] = ALL_ML_MISSING

    return (
        pd.DataFrame({"cause": cause})
        .reset_index()
        .assign(
            missing_methods=missing.apply(
                lambda r: ",".join(c for c in ml if r[c]), axis=1
            ).values
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-root", type=Path, default=paths.LEGACY_EXPERIMENTS
    )
    parser.add_argument("--weighting", choices=["iono", "elev", "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weightings = ["iono", "elev"] if args.weighting == "both" else [args.weighting]
    for weighting in weightings:
        combined = collect(weighting, args.experiments_root)
        suffix = "" if weighting == "iono" else f"_{weighting}"
        path = args.output_dir / f"multiday_summary{suffix}.csv"
        combined.to_csv(path, index=False, float_format="%.4f")

        counts = combined.groupby("method").size()
        print(f"\n=== {weighting} weighting: station-days per method ===")
        print(counts.to_string())
        wide = combined.pivot_table(
            index=["doy", "station"], columns="method", values="e_rms", aggfunc="first"
        )
        print(
            f"  solved by ALL methods: {wide.notna().all(axis=1).sum()} of {len(wide)}"
        )

        coverage = classify(combined, weighting)
        coverage_path = args.output_dir / f"coverage{suffix}.csv"
        coverage.to_csv(coverage_path, index=False)

        cause_counts = coverage["cause"].value_counts()
        total = len(coverage)
        print(f"\n--- why station-days are not in the common set ({weighting}) ---")
        for cause, count in cause_counts.items():
            print(f"  {cause}: {count} ({100 * count / total:.1f}%)")
        if cause_counts.get(UNCLASSIFIED, 0):
            logger.warning(
                f"{cause_counts[UNCLASSIFIED]} station-day(s) unclassified - the "
                "cause logic in classify() did not cover every case"
            )
        logger.info(f"💾 {path}")
        logger.info(f"💾 {coverage_path}")


if __name__ == "__main__":
    main()
