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

# Corrected 2026-09-17: a daily summary's raw `method` column is not purely this
# weighting's own arm. An iono summary can carry elevation-weighted `gim` rows as a
# per-station fallback wherever no `gim_iono` solution exists (a genuine artifact of
# how the summaries are produced, not a `collect()`-only defect - see
# `verification.repair_overwritten_summaries`'s `IONO_METHODS` for the same finding on
# the repair side). `collect()` used to relabel *any* row containing "gim" to
# `gim_{weighting}`, so those fallback rows silently became the iono GIM arm's value
# wherever no genuine `gim_iono` row competed - this maps each weighting to the one
# raw method value that is genuinely its own; anything else containing "gim" is
# dropped, never relabelled into this arm.
GENUINE_GIM_METHOD = {"iono": "gim_iono", "elev": "gim"}

# Fixed priority for cross-tree GIM dedup, not the alphabetical accident of directory
# names `source_dir` sorting used to provide. The three canonical trees are supposed to
# run byte-identical PPPx GIM solutions, so this only decides which row's floating-point
# noise "wins" once `find_gim_disagreements` has confirmed they actually agree - it must
# not itself depend on a canonical directory name changing.
GIM_SOURCE_PRIORITY = {"STEC": 0, "VTEC": 1, "Pretrained_STEC": 2}

# The causes classify() assigns. Named as constants so main() and the tests compare
# against the same strings rather than retyping them.
SOLVED_BY_ALL = "solved by all methods"
SOME_ML_MISSING = "some ML methods missing (per-method failure)"
ALL_ML_MISSING = "all ML methods missing (station absent from STEC DB)"
UNCLASSIFIED = "unclassified"

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("positioning_coverage", rebuilt=True)

# Owner decision 2026-09-18 (docs/revision/positioning_reporting.md): a station-day is a
# solver/geometry failure - independent of which correction method produced the number -
# when *every one* of the eight arms (all four methods, both weightings) reports a 3D RMS
# position error above this threshold. This is a data-quality rule about station/solver
# failures, not the outcome-based, between-methods filter the owner has separately and
# permanently rejected (docs/revision/positioning_reporting.md): a station-day where only
# *some* arms exceed the threshold (e.g. CHPG/248, VTEC_iono 139.8 m against gim_iono
# 8.5 m; POVE/124, VTEC_iono 142.0 m against gim_iono 2.0 m) says something about one
# method, not about the station-day itself, and must stay in. On the data as of
# 2026-09-18 this removes exactly one station-day: URUM/DOY 365, all eight solutions
# between 5,880 and 5,990 m.
SOLVER_FAILURE_THRESHOLD_M = 100.0

# The eight arms a station-day needs *all* of, at a value above the threshold, to count
# as a solver failure - the same four methods x two weightings
# `common_set_positioning.COVERAGE_COMMON_SET_METHODS` requires present for the common
# set. Built from this module's own `METHOD_TREES` rather than imported from
# `common_set_positioning`, which is a downstream consumer of this module's output, not
# a dependency of it.
SOLVER_FAILURE_ARMS: tuple[str, ...] = tuple(
    f"{model}_{weighting}"
    for model in (*METHOD_TREES.keys(), "gim")
    for weighting in ("iono", "elev")
)

EXCLUDED_SOLVER_FAILURES_FILENAME = "excluded_solver_failures.csv"


def find_solver_failure_station_days(
    all_weightings: pd.DataFrame, threshold_m: float = SOLVER_FAILURE_THRESHOLD_M
) -> pd.DataFrame:
    """(station, doy) pairs where every one of `SOLVER_FAILURE_ARMS` is present and
    exceeds `threshold_m`.

    Takes the same shape as `multiday_summary_all_weightings.csv` (one row per
    station/doy/method with an `error_3d_rms` column, both weightings concatenated).
    Requires all eight arms present - a station-day missing one is already outside the
    common set for a coverage reason (see `common_set_positioning.
    coverage_common_station_days`), so it needs no separate exclusion here, and a
    station-day where the arms that *are* present all exceed the threshold but others
    are simply absent is not "every solution exceeds the threshold" in the sense this
    rule means.

    Returns one row per excluded station-day with every arm's `error_3d_rms`, so a
    caller (and `excluded_solver_failures.csv`) can show which values triggered it
    rather than only the fact of exclusion.
    """
    wide = all_weightings.pivot_table(
        index=["station", "doy"],
        columns="method",
        values="error_3d_rms",
        aggfunc="first",
    )
    for arm in SOLVER_FAILURE_ARMS:
        if arm not in wide.columns:
            wide[arm] = pd.NA
    arms = wide[list(SOLVER_FAILURE_ARMS)]
    is_solver_failure = arms.notna().all(axis=1) & (arms > threshold_m).all(axis=1)
    return arms[is_solver_failure].reset_index()


def solver_failure_station_day_set(csv_path: Path) -> set[tuple[str, int]]:
    """Read an `excluded_solver_failures.csv` back into a `{(station, doy)}` lookup set.

    For callers that build a population from raw per-day files rather than reading
    `positioning_coverage`'s own rebuilt summaries directly - e.g.
    `common_set_positioning.load_pretrained_elev`, which globs a live experiment tree's
    `daily_summary.csv` files and so would not otherwise inherit an exclusion applied to
    this module's own CSVs. Returns an empty set (not an error) when the file does not
    exist yet, matching this module's own tolerance for a stage that has not produced it
    (e.g. a `--weighting elev`-only invocation, which never writes this file - see
    `main()`).
    """
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path, usecols=["station", "doy"])
    return set(zip(frame["station"].astype(str).str.upper(), frame["doy"].astype(int)))


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


# "Beyond rounding" = beyond the CSV's own `%.4f` write precision, the same threshold
# `verification.repair_overwritten_summaries.find_value_mismatches` uses for the
# equivalent question on a single tree's own history.
GIM_DISAGREEMENT_ATOL = 1e-3


def find_gim_disagreements(
    weighting: str,
    experiments_root: Path,
    *,
    all_variants: bool = False,
    atol: float = GIM_DISAGREEMENT_ATOL,
) -> pd.DataFrame:
    """(doy, station) keys where more than one canonical tree's genuine GIM row
    disagrees by more than rounding noise.

    The three canonical trees are supposed to run byte-identical PPPx GIM solutions for
    a given station-day, so `collect()`'s cross-tree dedup (`GIM_SOURCE_PRIORITY`)
    assumes any surviving row is interchangeable with the ones it drops. This is the
    check that verifies that assumption instead of taking it on faith: re-reads each
    tree's genuine GIM rows independently (never the relabelled/deduplicated frame
    `collect()` returns, since the duplicates are exactly what has already been
    resolved away there) and reports any station-day where two trees' values differ by
    more than `atol`.

    **Excludes the same foreign-DOY contamination `collect()` already drops**
    (`find_foreign_doy_rows` - e.g. `Finetune_STEC_2024_170_..._SWI`'s stray
    `results/2024122/` subdirectory). Verified 2026-09-17: without this exclusion, that
    one known-contaminated directory alone produced 50 of 51 reported elev-weighting
    "cross-tree disagreements" - every one of them was the stray directory's foreign row
    being read as a second, disagreeing offering from the same tree that already had a
    genuine one, not an actual disagreement between two trees' PPPx runs.
    """
    genuine_gim_method = GENUINE_GIM_METHOD[weighting]
    summary_name = SUMMARY_FILE[weighting]
    rows = []
    source_dirs: set[str] = set()
    for model, patterns in METHOD_TREES.items():
        pattern = patterns["all_variants"] if all_variants else patterns["canonical"]
        for path in sorted(
            experiments_root.glob(f"{pattern}/positioning/results/2024*/{summary_name}")
        ):
            match = DOY_PATTERN.search(str(path))
            if match is None:
                continue
            doy = int(match.group(2))
            try:
                frame = pd.read_csv(path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if frame.empty or "method" not in frame.columns:
                continue
            source_dir = path.relative_to(experiments_root).parts[0]
            source_dirs.add(source_dir)
            method = frame["method"].astype(str).str.lower()
            gim_rows = frame[method == genuine_gim_method]
            for _, row in gim_rows.iterrows():
                rows.append(
                    {
                        "doy": doy,
                        "station": str(row["station"]).upper(),
                        "tree": model,
                        "e_rms": float(row["e_rms"]),
                        "source_dir": source_dir,
                    }
                )

    columns = ["doy", "station", "min_e_rms", "max_e_rms", "max_abs_diff", "trees"]
    if not rows:
        return pd.DataFrame(columns=columns)

    gim_frame = pd.DataFrame(rows)
    foreign_doy_rows = find_foreign_doy_rows(experiments_root, source_dirs)
    if not foreign_doy_rows.empty:
        contaminated = foreign_doy_rows[["source_dir", "foreign_results_doy"]].rename(
            columns={"foreign_results_doy": "doy"}
        )
        gim_frame = gim_frame.merge(
            contaminated.assign(_contaminated=True),
            on=["source_dir", "doy"],
            how="left",
        )
        gim_frame = gim_frame[gim_frame["_contaminated"].isna()].drop(
            columns="_contaminated"
        )
    gim_frame = gim_frame.drop(columns="source_dir")
    if gim_frame.empty:
        return pd.DataFrame(columns=columns)

    disagreements = []
    for (doy, station), group in gim_frame.groupby(["doy", "station"]):
        if group["tree"].nunique() < 2:
            continue  # only one tree offered this station-day - nothing to compare
        spread = group["e_rms"].max() - group["e_rms"].min()
        if spread > atol:
            disagreements.append(
                {
                    "doy": doy,
                    "station": station,
                    "min_e_rms": group["e_rms"].min(),
                    "max_e_rms": group["e_rms"].max(),
                    "max_abs_diff": spread,
                    "trees": ",".join(sorted(group["tree"].unique())),
                }
            )
    if not disagreements:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(disagreements, columns=columns)
        .sort_values("max_abs_diff", ascending=False)
        .reset_index(drop=True)
    )


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
    genuine_gim_method = GENUINE_GIM_METHOD[weighting]
    source_dirs: set[str] = set()
    frames = []
    foreign_gim_dropped = 0
    non_ground_truth_dropped = 0
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
            frame["_tree"] = model
            source_dirs.add(source_dir)
            method = frame["method"].astype(str).str.lower()

            # Drop the other weighting's GIM rows rather than relabelling them into
            # this arm - see GENUINE_GIM_METHOD's comment for why a "gim"-containing
            # value is not automatically this weighting's own.
            is_foreign_gim = method.str.contains("gim") & (method != genuine_gim_method)
            if is_foreign_gim.any():
                foreign_gim_dropped += int(is_foreign_gim.sum())
                frame = frame[~is_foreign_gim].reset_index(drop=True)
                method = method[~is_foreign_gim].reset_index(drop=True)

            # A `ref_source != "ground_truth"` row (typically "mean": no SINEX was
            # available for that day when the summary was first written) measures
            # internal repeatability around the day's own mean position, not a true
            # error against ground truth - it must never enter the common-set
            # population Tables 6/7/8 are built from. `verification.
            # repair_overwritten_summaries`'s mean->ground_truth upgrade fixes this at
            # the source for most rows once SINEX becomes available; this is the
            # defensive backstop for whatever it can't (no SINEX for the day at all, or
            # a summary this repair pass has not touched).
            if "ref_source" in frame.columns:
                non_ground_truth = frame["ref_source"] != "ground_truth"
                if non_ground_truth.any():
                    non_ground_truth_dropped += int(non_ground_truth.sum())
                    frame = frame[~non_ground_truth].reset_index(drop=True)
                    method = method[~non_ground_truth].reset_index(drop=True)

            frame.loc[method.str.startswith("model"), "method"] = label
            frame.loc[method == genuine_gim_method, "method"] = f"gim_{weighting}"
            frames.append(frame)

    if foreign_gim_dropped:
        logger.warning(
            f"{foreign_gim_dropped} row(s) labelled with a different weighting's GIM "
            f"method were dropped rather than folded into gim_{weighting} (e.g. "
            "elevation-weighted 'gim' rows found inside an iono summary)"
        )
    if non_ground_truth_dropped:
        logger.warning(
            f"{non_ground_truth_dropped} row(s) with ref_source != 'ground_truth' "
            "(day-mean fallback, no SINEX for that day) were dropped rather than "
            "counted as solved station-days"
        )

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
    # Tie-break by GIM_SOURCE_PRIORITY before source_dir, so which tree's row survives
    # a cross-tree GIM duplicate is a declared rule rather than an accident of
    # `source_dir`'s alphabetical sort - see that constant's own comment.
    # `find_gim_disagreements` is what actually verifies the two rows agree.
    combined["_tree_priority"] = combined["_tree"].map(GIM_SOURCE_PRIORITY)
    combined = combined.sort_values(
        ["doy", "station", "method", "_tree_priority", "source_dir"]
    )
    combined = combined.drop_duplicates(
        subset=["date", "method", "station"], keep="first"
    )
    combined = combined.drop(columns=["_tree", "_tree_priority"])
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
    collected = {
        weighting: collect(
            weighting, args.experiments_root, all_variants=args.all_variants
        )
        for weighting in weightings
    }
    combined_by_weighting: dict[str, pd.DataFrame] = {
        weighting: combined for weighting, (combined, _, _) in collected.items()
    }

    # The solver-failure exclusion (SOLVER_FAILURE_THRESHOLD_M) needs all eight arms -
    # both weightings - to decide whether a station-day's *every* solution failed, so it
    # can only run once every weighting in this invocation has been collected. Same
    # restriction the multiday_summary_all_weightings.csv concatenation below already
    # has: a --weighting elev/iono partial run has nothing correct to decide from.
    if set(weightings) == {"iono", "elev"}:
        solver_failures = find_solver_failure_station_days(
            pd.concat(combined_by_weighting.values(), ignore_index=True)
        )
        failures_path = args.output_dir / EXCLUDED_SOLVER_FAILURES_FILENAME
        solver_failures.to_csv(failures_path, index=False, float_format="%.4f")
        print(
            f"\n=== solver-failure exclusion (all {len(SOLVER_FAILURE_ARMS)} arms > "
            f"{SOLVER_FAILURE_THRESHOLD_M:.0f} m) ==="
        )
        if solver_failures.empty:
            print("  none")
        else:
            for _, failure in solver_failures.iterrows():
                print(f"  {failure['station']}/{int(failure['doy'])}")
        logger.info(
            f"💾 {failures_path} ({len(solver_failures)} station-day(s) excluded)"
        )

        if not solver_failures.empty:
            failed_keys = set(zip(solver_failures["station"], solver_failures["doy"]))
            for weighting, combined in combined_by_weighting.items():
                keys = list(zip(combined["station"], combined["doy"]))
                keep = [key not in failed_keys for key in keys]
                combined_by_weighting[weighting] = combined[keep].reset_index(drop=True)

    for weighting in weightings:
        combined = combined_by_weighting[weighting]
        _, collisions, foreign_doy_rows = collected[weighting]
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

        gim_disagreements = find_gim_disagreements(
            weighting, args.experiments_root, all_variants=args.all_variants
        )
        print(f"\n--- cross-tree GIM disagreements beyond rounding ({weighting}) ---")
        if gim_disagreements.empty:
            print("  none - every station-day two or more trees offered agrees")
        else:
            gim_disagreements_path = args.output_dir / f"gim_disagreements{suffix}.csv"
            gim_disagreements.to_csv(gim_disagreements_path, index=False)
            print(
                f"  {len(gim_disagreements)} station-day(s) disagree by more than "
                f"{GIM_DISAGREEMENT_ATOL} m across trees - see {gim_disagreements_path}"
            )
            logger.info(f"💾 {gim_disagreements_path}")

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

    # Both weightings share this module's own canonical variant selection, so
    # concatenating them here - rather than leaving downstream stages to combine the
    # iono and elev files themselves, or to read a frozen external run - is what lets
    # weighting_ablation, common_set_positioning and oracle_benchmark stop reading
    # multiday_results/positioning_runs/20260216_2052/multiday_summary.csv, a 2026-02-16
    # snapshot from before the rebuild that nothing regenerates any more (56,457 rows,
    # 245 dates, 55 stations - a much narrower population than the current recovered
    # set). Only written when both weightings actually ran this invocation; a
    # --weighting elev/iono partial run has nothing correct to concatenate.
    if set(weightings) == {"iono", "elev"}:
        all_weightings_path = args.output_dir / "multiday_summary_all_weightings.csv"
        all_weightings = pd.concat(
            [combined_by_weighting["iono"], combined_by_weighting["elev"]],
            ignore_index=True,
        )
        all_weightings.to_csv(all_weightings_path, index=False, float_format="%.4f")
        logger.info(
            f"💾 {all_weightings_path} ({len(all_weightings):,} rows, both weightings)"
        )


if __name__ == "__main__":
    main()
