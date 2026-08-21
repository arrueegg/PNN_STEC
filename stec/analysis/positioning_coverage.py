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

**Variant selection is explicit, not alphabetical.** Each experiment tree glob used to
match every hyperparameter variant on disk (``Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI``),
and ``drop_duplicates(subset=["date", "method", "station"], keep="first")`` resolved any
resulting collision by sorted glob order - not by which variant is the paper's canonical
fine-tune. That was latent while only one directory per DOY held positioning results.
The station-recovery sweep (see ``docs/revision/coverage_variant_selection.md``) created
a second results directory for 31 DOYs, and sorted order silently picked
``lr1e-4_bs2048``/``lr1e-4_bs10000`` over the canonical ``lr2e-4_bs512`` fine-tune for all
of them, because ``"lr1e-4"`` sorts before ``"lr2e-4"``. ``collect()`` now globs the
canonical variant directory name by default (see ``CANONICAL_STEC_SUFFIX`` etc.) and
reports, rather than silently resolves, any remaining collision or any DOY that has only
a non-canonical variant. ``--all-variants`` restores the old broad glob for auditing.
Checking this also turned up an unrelated, narrower anomaly worth guarding structurally:
``Finetune_STEC_2024_170_..._SWI`` contains a stray ``positioning/results/2024122/``
subdirectory whose GIM numbers disagree with DOY 122's own canonical directory. See
`find_foreign_doy_rows` - `collect()` excludes those rows rather than letting them
compete as a second candidate value.

Usage::

    python -m stec.analysis.positioning_coverage
    python -m stec.analysis.positioning_coverage --weighting elev
    python -m stec.analysis.positioning_coverage --all-variants
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

# The paper's canonical variants (CLAUDE.md "The paper model"). These are exact
# directory-name components, not globs that happen to resolve correctly - see the module
# docstring for why that distinction matters.
CANONICAL_STEC_SUFFIX = (
    "BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr2e-4_bs512_GNLL_Adam_"
    "ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
CANONICAL_VTEC_SUFFIX = (
    "MLP_LaplacianNLL_h90_l3_lr1e-3_bs2048_LaplacianNLL_Adam_ReduceLROnPlateau_sub500K_"
    "SH15_ps0.1_lw1e+0_woYear"
)
CANONICAL_PRETRAINED_DIR = (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr1e-3_bs1024_GNLL_Adam_"
    "ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)

# Which experiment tree contributes which method label, relative to LEGACY_EXPERIMENTS.
# "canonical" matches only the paper's variant; "all_variants" is the old broad glob,
# kept for the --all-variants audit escape hatch. The GIM arm is written into every tree
# by the same PPPx run, so it is de-duplicated afterwards regardless of which pattern
# found it.
METHOD_TREES = {
    "STEC": {
        "canonical": f"Finetune_STEC_2024_*_{CANONICAL_STEC_SUFFIX}",
        "all_variants": "Finetune_STEC_2024_*_BayesianResNetSTEC_*_SWI",
    },
    "VTEC": {
        "canonical": f"Finetune_VTEC_2024_*_{CANONICAL_VTEC_SUFFIX}",
        "all_variants": "Finetune_VTEC_2024_*_MLP_LaplacianNLL_*_woYear",
    },
    "Pretrained_STEC": {
        "canonical": CANONICAL_PRETRAINED_DIR,
        "all_variants": "Pretrain_STEC_BayesianResNetSTEC_*_SWI",
    },
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


def find_collisions(combined: pd.DataFrame) -> pd.DataFrame:
    """(doy, method, station) keys supplied by more than one experiment directory.

    Restricted to non-GIM methods: the GIM arm is recomputed independently by every
    method tree's own PPPx run, so small numeric disagreement between trees is routine
    solver noise (typically <0.1 unit, though a handful of outlier stations disagree by
    tens of units for reasons this script does not investigate) rather than a variant
    ambiguity - it is not the defect this function exists to catch, and flagging every
    such pair would bury the real signal under thousands of harmless entries. A real
    collision here means two different hyperparameter variants both produced a row for
    the same station-day and ML method - exactly the situation
    ``drop_duplicates(keep="first")`` used to resolve silently by sorted glob order.
    Returns one row per colliding key with every contributing directory named, so a
    caller can see which variants competed rather than only a count.

    A separate, narrower structural check exists for one real anomaly this module found
    on the live tree that *is* a GIM-arm problem: see `find_foreign_doy_rows`.
    """
    ml_rows = combined[~combined["method"].str.startswith("gim_")]
    if ml_rows.empty:
        return pd.DataFrame(
            columns=["doy", "date", "method", "station", "source_dirs", "n_variants"]
        )
    grouped = ml_rows.groupby(["doy", "date", "method", "station"])["source_dir"].agg(
        lambda values: sorted(set(values))
    )
    collisions = grouped[grouped.apply(len) > 1].reset_index()
    collisions = collisions.rename(columns={"source_dir": "source_dirs"})
    collisions["n_variants"] = collisions["source_dirs"].apply(len)
    return collisions.sort_values(["doy", "method", "station"]).reset_index(drop=True)


# A Finetune_{STEC,VTEC} experiment directory encodes its own DOY in its name; the
# Pretrained tree does not (one directory legitimately holds positioning results for
# every day, since it is not fine-tuned per day).
_FINETUNE_DIR_OWN_DOY = re.compile(r"^Finetune_(?:STEC|VTEC)_2024_(\d+)_")


def find_foreign_doy_rows(
    experiments_root: Path, source_dirs: set[str]
) -> pd.DataFrame:
    """Finetune experiment directories whose own DOY does not match a results
    subdirectory they contain.

    Found once on the live tree and worth checking for structurally rather than trusting
    it stays a one-off: ``Finetune_STEC_2024_170_..._SWI/positioning/results/2024122/``
    exists and disagrees with DOY 122's own canonical directory on every GIM e_rms value
    it shares a station with (e.g. AIRA 0.3302 there vs 0.4626 in the DOY-122 directory).
    Because the glob that finds per-day files wildcards the day twice, independently, in
    ``Finetune_STEC_2024_*_.../results/2024*/``, nothing enforces that the two occurrences
    agree - a fine-tune's own results directory is supposed to hold only its own day, so
    a mismatch means the file does not belong to the day it is being read as. `collect()`
    excludes these rows entirely (not just deduplicates them) rather than letting them
    compete with the correct directory, because a foreign day's GIM run is not a second
    legitimate candidate value - it is contamination.
    """
    rows = []
    for source_dir in sorted(source_dirs):
        match = _FINETUNE_DIR_OWN_DOY.match(source_dir)
        if match is None:
            continue
        own_doy = int(match.group(1))
        results_root = experiments_root / source_dir / "positioning" / "results"
        if not results_root.is_dir():
            continue
        for results_dir in sorted(results_root.iterdir()):
            results_match = re.match(r"^2024(\d{3})$", results_dir.name)
            if results_match is None:
                continue
            results_doy = int(results_match.group(1))
            if results_doy != own_doy:
                rows.append(
                    {
                        "source_dir": source_dir,
                        "own_doy": own_doy,
                        "foreign_results_doy": results_doy,
                    }
                )
    return pd.DataFrame(rows, columns=["source_dir", "own_doy", "foreign_results_doy"])


def find_canonical_gaps(weighting: str, experiments_root: Path) -> pd.DataFrame:
    """DOYs where only a non-canonical variant exists for a method.

    Canonical-only selection (the default) makes these DOYs contribute nothing for that
    method rather than picking a substitute - correct, but silent unless reported
    separately from `find_collisions`, since there is no collision to report: the
    canonical directory is simply missing for that day, not competing with another one.
    """
    gaps = []
    for model, patterns in METHOD_TREES.items():
        summary_name = SUMMARY_FILE[weighting]
        canonical_found = experiments_root.glob(
            f"{patterns['canonical']}/positioning/results/2024*/{summary_name}"
        )
        all_found = experiments_root.glob(
            f"{patterns['all_variants']}/positioning/results/2024*/{summary_name}"
        )
        canonical_doys = _doys_of(canonical_found)
        all_doys = _doys_of(all_found)
        for doy in sorted(all_doys - canonical_doys):
            gaps.append({"model": model, "doy": doy})
    return pd.DataFrame(gaps, columns=["model", "doy"])


def _doys_of(paths_iter) -> set[int]:
    doys = set()
    for path in paths_iter:
        match = DOY_PATTERN.search(str(path))
        if match is not None:
            doys.add(int(match.group(2)))
    return doys


def collect(
    weighting: str, experiments_root: Path, *, all_variants: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every per-day summary on disk, relabelled exactly as run_pipeline.py would.

    Returns ``(combined, collisions, foreign_doy_rows)``. ``combined`` is deduplicated on
    ``(date, method, station)`` after foreign-DOY contamination (see
    `find_foreign_doy_rows`) is dropped entirely, keeping the first remaining row after
    sorting by ``source_dir`` for determinism; ``collisions`` (see `find_collisions`)
    lists every key that deduplication still had to resolve, so an ambiguity is always
    visible to the caller rather than only reflected in which row happened to survive.
    """
    source_dirs: set[str] = set()
    frames = []
    for model, patterns in METHOD_TREES.items():
        pattern = patterns["all_variants"] if all_variants else patterns["canonical"]
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
            source_dir = path.relative_to(experiments_root).parts[0]
            frame["source_dir"] = source_dir
            source_dirs.add(source_dir)
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

    foreign_doy_rows = find_foreign_doy_rows(experiments_root, source_dirs)
    if not foreign_doy_rows.empty:
        logger.warning(
            f"{len(foreign_doy_rows)} experiment directory/directories contain a "
            "results subdirectory for a DOY other than their own - excluding those "
            f"rows rather than treating them as a candidate value: "
            f"{foreign_doy_rows.to_dict('records')}"
        )
        contaminated = foreign_doy_rows[["source_dir", "foreign_results_doy"]].rename(
            columns={"foreign_results_doy": "doy"}
        )
        combined = combined.merge(
            contaminated.assign(_contaminated=True),
            on=["source_dir", "doy"],
            how="left",
        )
        combined = combined[combined["_contaminated"].isna()].drop(
            columns="_contaminated"
        )

    collisions = find_collisions(combined)
    if not collisions.empty:
        doys = sorted(collisions["doy"].unique().tolist())
        logger.warning(
            f"{len(collisions)} (date, method, station) collision(s) across "
            f"{len(doys)} DOY(s) - more than one directory supplied the same "
            f"station-day for one method. Keeping the first by source_dir sort order; "
            f"see the returned collisions frame for which directories competed. "
            f"DOYs: {doys}"
        )

    before = len(combined)
    combined = combined.sort_values(["doy", "station", "method", "source_dir"])
    combined = combined.drop_duplicates(
        subset=["date", "method", "station"], keep="first"
    )
    logger.info(
        f"{before} rows -> {len(combined)} after de-duplicating the shared GIM arm "
        f"and {len(collisions)} reported collision(s)"
    )
    return (
        combined.sort_values(["doy", "station", "method"]).reset_index(drop=True),
        collisions,
        foreign_doy_rows,
    )


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
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Match every hyperparameter variant instead of only the paper's canonical "
        "fine-tune. For auditing what else is on disk; the resulting collisions are "
        "reported but resolved arbitrarily by source_dir sort order, so do not use this "
        "for numbers that get quoted.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    weightings = ["iono", "elev"] if args.weighting == "both" else [args.weighting]
    for weighting in weightings:
        combined, collisions, foreign_doy_rows = collect(
            weighting, args.experiments_root, all_variants=args.all_variants
        )
        suffix = "" if weighting == "iono" else f"_{weighting}"
        path = args.output_dir / f"multiday_summary{suffix}.csv"
        combined.to_csv(path, index=False, float_format="%.4f")

        collisions_path = args.output_dir / f"collisions{suffix}.csv"
        collisions.to_csv(collisions_path, index=False)

        counts = combined.groupby("method").size()
        print(f"\n=== {weighting} weighting: station-days per method ===")
        print(counts.to_string())
        wide = combined.pivot_table(
            index=["doy", "station"], columns="method", values="e_rms", aggfunc="first"
        )
        print(
            f"  solved by ALL methods: {wide.notna().all(axis=1).sum()} of {len(wide)}"
        )

        n_collision_doys = collisions["doy"].nunique() if not collisions.empty else 0
        print(f"\n--- variant collisions ({weighting}) ---")
        print(f"  {len(collisions)} collision(s) across {n_collision_doys} DOY(s)")
        if not collisions.empty:
            print(f"  DOYs: {sorted(collisions['doy'].unique().tolist())}")

        if not foreign_doy_rows.empty:
            foreign_doy_path = args.output_dir / f"foreign_doy_rows{suffix}.csv"
            foreign_doy_rows.to_csv(foreign_doy_path, index=False)
            print(f"\n--- foreign-DOY directories excluded ({weighting}) ---")
            for _, row in foreign_doy_rows.iterrows():
                print(
                    f"  {row['source_dir']} (own DOY {row['own_doy']}) contains "
                    f"results for DOY {row['foreign_results_doy']} - excluded"
                )
            logger.info(f"💾 {foreign_doy_path}")

        if not args.all_variants:
            gaps = find_canonical_gaps(weighting, args.experiments_root)
            if not gaps.empty:
                gaps_path = args.output_dir / f"canonical_gaps{suffix}.csv"
                gaps.to_csv(gaps_path, index=False)
                print(f"\n--- DOYs with only a non-canonical variant ({weighting}) ---")
                for model, doys in gaps.groupby("model")["doy"]:
                    print(f"  {model}: {sorted(doys.tolist())}")
                logger.info(f"💾 {gaps_path}")

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
        logger.info(f"💾 {collisions_path}")
        logger.info(f"💾 {coverage_path}")


if __name__ == "__main__":
    main()
