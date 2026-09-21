"""Tests for `stec.analysis.positioning_coverage` (R1.5).

The fixture below is a synthetic three-experiment-tree directory laid out exactly like
`stec.config.paths.LEGACY_EXPERIMENTS`, with a known answer for each of the three
coverage causes:

* AMC4/150 - present in all three ML trees and the GIM -> solved by all methods.
* ZIMM/150 - present in the GIM and two ML trees, but the Pretrained_STEC tree has no
  row for it (its correction failed even though the station-day is in the database) ->
  the "510" class, some ML methods missing.
* WARK/150 - present only as a GIM row, no ML tree has any row for it at all (the
  station never appears in the STEC database that day) -> the "2,311" class, all ML
  methods missing.
* NANO/150 - has an ML row but no GIM row at all -> excluded from `classify()`'s output
  entirely, because the split is defined over station-days the GIM itself solved.

The fixture uses the paper's *canonical* directory-name suffixes (see
`CANONICAL_STEC_SUFFIX`, `CANONICAL_VTEC_SUFFIX`, `CANONICAL_PRETRAINED_DIR`) so it also
exercises the canonical-only glob, not just the relabelling logic. Variant-selection
tests below add a second, non-canonical directory alongside these.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stec.analysis import positioning_coverage as pc

STEC_DIR = f"Finetune_STEC_2024_150_{pc.CANONICAL_STEC_SUFFIX}"
VTEC_DIR = f"Finetune_VTEC_2024_150_{pc.CANONICAL_VTEC_SUFFIX}"
PRETRAINED_DIR = pc.CANONICAL_PRETRAINED_DIR

# A non-canonical STEC variant that sorts *after* the canonical one alphabetically
# ("lr2e-4" < "lr9e-9"), so a test relying on this fixture proves canonical selection
# does not depend on glob sort order - it would fail under the old sorted-glob logic,
# which always resolved a collision to whichever path sorted first.
NON_CANONICAL_STEC_SUFFIX = (
    "BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_lr9e-9_bs2048_GNLL_Adam_"
    "ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
NON_CANONICAL_STEC_DIR = f"Finetune_STEC_2024_150_{NON_CANONICAL_STEC_SUFFIX}"


def write_daily_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["station", "method", "e_rms"]).to_csv(path, index=False)


def build_fixture(root: Path) -> None:
    """One experiment directory per entry of `METHOD_TREES`, each with a single day's
    `daily_summary_iono.csv` under `positioning/results/2024150/`."""
    day_dir = "positioning/results/2024150/daily_summary_iono.csv"

    write_daily_summary(
        root / STEC_DIR / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
            {"station": "ZIMM", "method": "model", "e_rms": 2.0},
            {"station": "ZIMM", "method": "gim_iono", "e_rms": 5.0},
            {"station": "WARK", "method": "gim_iono", "e_rms": 7.0},
        ],
    )
    write_daily_summary(
        root / VTEC_DIR / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.1},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
            {"station": "ZIMM", "method": "model", "e_rms": 2.1},
            {"station": "ZIMM", "method": "gim_iono", "e_rms": 5.0},
            {"station": "NANO", "method": "model", "e_rms": 3.0},
        ],
    )
    write_daily_summary(
        root / PRETRAINED_DIR / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.2},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
            # ZIMM deliberately absent: the "510" class - some ML methods missing.
        ],
    )


# ---------------------------------------------------------------------------
# collect(): reading and relabelling every per-day file on disk
# ---------------------------------------------------------------------------


def test_collect_relabels_methods_and_deduplicates_the_shared_gim_arm(tmp_path):
    build_fixture(tmp_path)
    combined, collisions, foreign_doy_rows = pc.collect("iono", tmp_path)
    assert foreign_doy_rows.empty

    # The GIM row for AMC4/150 is written identically into all three trees; it must
    # collapse to a single row rather than being triple-counted, and that expected
    # repeat must not be reported as a collision (it is not a variant ambiguity).
    gim_rows = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "gim_iono")
    ]
    assert len(gim_rows) == 1
    assert collisions.empty

    stec_rows = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "STEC_iono")
    ]
    assert stec_rows["e_rms"].iloc[0] == pytest.approx(1.0)


def test_collect_raises_when_nothing_matches(tmp_path):
    with pytest.raises(SystemExit):
        pc.collect("iono", tmp_path)


# ---------------------------------------------------------------------------
# Variant selection: canonical-only by default, collisions reported not resolved
# silently, sort order must not decide the outcome.
# ---------------------------------------------------------------------------


def test_canonical_only_selection_ignores_a_non_canonical_sibling(tmp_path):
    """A second, non-canonical STEC directory for the same DOY must not appear in the
    canonical-only result at all - the defect this module exists to fix let a
    non-canonical variant silently win by sorted glob order."""
    build_fixture(tmp_path)
    write_daily_summary(
        tmp_path
        / NON_CANONICAL_STEC_DIR
        / "positioning/results/2024150/daily_summary_iono.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 99.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
        ],
    )

    combined, collisions, _ = pc.collect("iono", tmp_path)

    stec_row = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "STEC_iono")
    ]
    assert len(stec_row) == 1
    assert stec_row["e_rms"].iloc[0] == pytest.approx(
        1.0
    )  # the canonical value, not 99.0
    assert NON_CANONICAL_STEC_DIR not in set(combined["source_dir"])
    assert collisions.empty  # nothing competed for the surviving row


def test_canonical_selection_does_not_depend_on_sort_order(tmp_path):
    """`NON_CANONICAL_STEC_DIR` sorts *after* `STEC_DIR` alphabetically (lr9e-9 > lr2e-4),
    the opposite of the historical bug (lr1e-4 sorting before lr2e-4). The canonical
    fine-tune must still be the one selected either way, because selection is by exact
    suffix match, not by glob order."""
    build_fixture(tmp_path)
    assert NON_CANONICAL_STEC_DIR > STEC_DIR  # the non-canonical dir sorts later
    write_daily_summary(
        tmp_path
        / NON_CANONICAL_STEC_DIR
        / "positioning/results/2024150/daily_summary_iono.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 99.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
        ],
    )

    combined, _, _ = pc.collect("iono", tmp_path)

    stec_row = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "STEC_iono")
    ]
    assert stec_row["e_rms"].iloc[0] == pytest.approx(1.0)


def test_all_variants_finds_both_directories(tmp_path):
    """--all-variants (all_variants=True) is the audit escape hatch: both the canonical
    and non-canonical directories must be globbed, even though the final dedup can only
    keep one row per (date, method, station)."""
    build_fixture(tmp_path)
    write_daily_summary(
        tmp_path
        / NON_CANONICAL_STEC_DIR
        / "positioning/results/2024150/daily_summary_iono.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 99.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.0},
        ],
    )

    combined, collisions, _ = pc.collect("iono", tmp_path, all_variants=True)

    # Both source directories were read and competed for the AMC4/STEC_iono row.
    assert not collisions.empty
    amc4_collision = collisions[
        (collisions["station"] == "AMC4") & (collisions["method"] == "STEC_iono")
    ].iloc[0]
    assert set(amc4_collision["source_dirs"]) == {STEC_DIR, NON_CANONICAL_STEC_DIR}
    assert amc4_collision["n_variants"] == 2

    # Canonical-only would never have globbed the non-canonical directory at all.
    combined_canonical, collisions_canonical, _ = pc.collect("iono", tmp_path)
    assert NON_CANONICAL_STEC_DIR not in set(combined_canonical["source_dir"])
    assert collisions_canonical.empty


def test_canonical_gaps_report_doys_with_only_a_non_canonical_variant(tmp_path):
    """A DOY that has *no* canonical directory at all - only a non-canonical one - must
    be reported by `find_canonical_gaps` rather than silently contributing zero rows for
    that method with no trace of why."""
    build_fixture(tmp_path)  # canonical STEC/VTEC/Pretrained all present for DOY 150

    # DOY 200: only a non-canonical STEC variant exists, no canonical STEC directory.
    only_non_canonical_dir = f"Finetune_STEC_2024_200_{NON_CANONICAL_STEC_SUFFIX}"
    write_daily_summary(
        tmp_path
        / only_non_canonical_dir
        / "positioning/results/2024200/daily_summary_iono.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 5.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 4.5},
        ],
    )

    gaps = pc.find_canonical_gaps("iono", tmp_path)

    stec_gaps = gaps[gaps["model"] == "STEC"]
    assert 200 in set(stec_gaps["doy"])
    assert 150 not in set(stec_gaps["doy"])  # DOY 150 has the canonical directory

    # And collect() (canonical-only) correctly contributes nothing for STEC/200 rather
    # than substituting the non-canonical variant.
    combined, _, _ = pc.collect("iono", tmp_path)
    assert not ((combined["doy"] == 200) & (combined["method"] == "STEC_iono")).any()


# ---------------------------------------------------------------------------
# find_foreign_doy_rows() / collect(): a directory's own DOY must match the DOY of any
# results subdirectory it contains - found once on the live tree (see the module
# docstring) and worth guarding structurally rather than trusting it stays a one-off.
# ---------------------------------------------------------------------------


def test_foreign_doy_results_directory_is_excluded_not_merged(tmp_path):
    """DOY 150's own canonical STEC directory additionally contains a
    `results/2024199/` subdirectory - a foreign day's results filed under the wrong
    experiment, exactly like the real `Finetune_STEC_2024_170_..._SWI` anomaly. Its rows
    must not compete with DOY 199's own results (there are none here) nor silently merge
    in; `collect()` must drop them and `find_foreign_doy_rows` must report them."""
    build_fixture(tmp_path)
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024199/daily_summary_iono.csv",
        [
            {"station": "FOREIGN", "method": "model", "e_rms": 42.0},
            {"station": "FOREIGN", "method": "gim_iono", "e_rms": 42.0},
        ],
    )

    foreign = pc.find_foreign_doy_rows(tmp_path, {STEC_DIR})
    assert len(foreign) == 1
    row = foreign.iloc[0]
    assert row["source_dir"] == STEC_DIR
    assert row["own_doy"] == 150
    assert row["foreign_results_doy"] == 199

    combined, collisions, foreign_doy_rows = pc.collect("iono", tmp_path)
    assert not foreign_doy_rows.empty
    assert "FOREIGN" not in set(combined["station"])
    # A structural exclusion, not a collision to arbitrate between competing values.
    assert collisions.empty


def test_pretrained_tree_is_exempt_from_the_foreign_doy_check(tmp_path):
    """The single Pretrained_STEC directory legitimately holds results for every DOY -
    it has no per-directory DOY in its name, so it must never be flagged."""
    build_fixture(tmp_path)
    foreign = pc.find_foreign_doy_rows(tmp_path, {PRETRAINED_DIR})
    assert foreign.empty


# ---------------------------------------------------------------------------
# classify(): the three coverage causes
# ---------------------------------------------------------------------------


@pytest.fixture
def combined(tmp_path):
    build_fixture(tmp_path)
    combined, _, _ = pc.collect("iono", tmp_path)
    return combined


def test_station_day_solved_by_every_method_is_classified_as_solved(combined):
    coverage = pc.classify(combined, "iono")
    amc4 = coverage[coverage["station"] == "AMC4"].iloc[0]
    assert amc4["cause"] == pc.SOLVED_BY_ALL
    assert amc4["missing_methods"] == ""


def test_station_day_missing_from_one_method_is_the_510_class(combined):
    coverage = pc.classify(combined, "iono")
    zimm = coverage[coverage["station"] == "ZIMM"].iloc[0]
    assert zimm["cause"] == pc.SOME_ML_MISSING
    assert zimm["missing_methods"] == "Pretrained_STEC_iono"


def test_station_absent_from_every_ml_tree_is_the_2311_class(combined):
    coverage = pc.classify(combined, "iono")
    wark = coverage[coverage["station"] == "WARK"].iloc[0]
    assert wark["cause"] == pc.ALL_ML_MISSING
    assert set(wark["missing_methods"].split(",")) == {
        "STEC_iono",
        "VTEC_iono",
        "Pretrained_STEC_iono",
    }


def test_station_day_with_no_gim_row_is_excluded_not_misclassified(combined):
    """NANO has an ML row but the GIM never solved it. The split is defined over
    GIM-solved station-days, so NANO must not appear in classify()'s output at all -
    it is out of the population being classified, not a fourth, unclassified cause."""
    coverage = pc.classify(combined, "iono")
    assert "NANO" not in set(coverage["station"])


def test_causes_are_exhaustive_and_percentages_sum_to_the_total(combined):
    """Every GIM-solved station-day must land in exactly one named cause - nothing
    silently dropped, and no residual UNCLASSIFIED bucket for this fixture."""
    coverage = pc.classify(combined, "iono")
    assert len(coverage) == 3  # AMC4, ZIMM, WARK - the three GIM-solved station-days

    cause_counts = coverage["cause"].value_counts()
    assert cause_counts.sum() == len(coverage)
    assert cause_counts.get(pc.UNCLASSIFIED, 0) == 0
    assert set(cause_counts.index) <= {
        pc.SOLVED_BY_ALL,
        pc.SOME_ML_MISSING,
        pc.ALL_ML_MISSING,
    }


def test_elev_weighting_reads_daily_summary_without_iono_suffix(tmp_path):
    """`SUMMARY_FILE` maps weight_opt=elev to `daily_summary.csv`, not the `_iono`
    file - a station-day that only exists under the elev filename must still be found."""
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "model", "e_rms": 1.0}],
    )
    write_daily_summary(
        tmp_path / VTEC_DIR / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "model", "e_rms": 1.1}],
    )
    write_daily_summary(
        tmp_path / PRETRAINED_DIR / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "gim", "e_rms": 4.0}],
    )

    combined, _, _ = pc.collect("elev", tmp_path)
    assert set(combined["method"]) == {"STEC_elev", "VTEC_elev", "gim_elev"}


# ---------------------------------------------------------------------------
# The GENUINE_GIM_METHOD fix (2026-09-17): an iono summary can carry elevation-weighted
# `gim` rows as a per-station fallback wherever no `gim_iono` solution exists (a real
# artifact of how the summaries are produced, not a test fixture invention - see
# verification.repair_overwritten_summaries's IONO_METHODS comment for the same finding
# on the repair side). Those rows must be dropped, never relabelled into gim_iono.
# ---------------------------------------------------------------------------


def test_collect_drops_a_foreign_gim_row_in_an_iono_summary_rather_than_relabelling_it(
    tmp_path,
):
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.0},
            # A per-station elevation-weighted fallback, not this arm's own value.
            {"station": "AMC4", "method": "gim", "e_rms": 99.0},
        ],
    )

    combined, _, _ = pc.collect("iono", tmp_path)

    assert "gim_iono" not in set(combined["method"])
    assert not ((combined["method"] == "STEC_iono") & (combined["e_rms"] == 99.0)).any()


def test_collect_drops_a_non_ground_truth_ref_source_row(tmp_path):
    """A row written with the day-mean fallback (`ref_source="mean"`, no SINEX
    available when the summary was first produced) is not a true positioning error
    against ground truth - it must never enter the common-set population `collect()`
    feeds to Tables 6/7/8, the same way a foreign-weighting GIM row never does. This is
    the defensive filter `verification.repair_overwritten_summaries`'s narrow
    mean->ground_truth upgrade cannot fully replace: a station-day whose rebuild also
    can't get a genuine SINEX row stays `mean` on disk, and must not silently count as
    solved."""
    path = tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary_iono.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "station": "AMC4",
                "method": "model_iono",
                "e_rms": 1.0,
                "ref_source": "ground_truth",
            },
            {
                "station": "ZIMM",
                "method": "model_iono",
                "e_rms": 99.0,
                "ref_source": "mean",
            },
        ]
    ).to_csv(path, index=False)

    combined, _, _ = pc.collect("iono", tmp_path)

    assert "ZIMM" not in set(combined["station"])
    assert set(combined["ref_source"]) == {"ground_truth"}


def test_collect_drops_a_foreign_gim_iono_row_in_an_elev_summary(tmp_path):
    """Symmetric case: an elev summary carrying a `gim_iono` row (never observed live,
    but the same generic rule applies) must be dropped, not folded into gim_elev."""
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary.csv",
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.0},
            {"station": "AMC4", "method": "gim", "e_rms": 4.0},
            {"station": "AMC4", "method": "gim_iono", "e_rms": 99.0},
        ],
    )

    combined, _, _ = pc.collect("elev", tmp_path)

    gim_row = combined[combined["method"] == "gim_elev"]
    assert len(gim_row) == 1
    assert gim_row["e_rms"].iloc[0] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# find_gim_disagreements(): cross-tree GIM rows are supposed to be interchangeable -
# this is what verifies that rather than assuming it.
# ---------------------------------------------------------------------------


def test_find_gim_disagreements_empty_when_trees_agree(tmp_path):
    build_fixture(tmp_path)  # AMC4's gim_iono row is 4.0 in every tree that has one

    disagreements = pc.find_gim_disagreements("iono", tmp_path)

    assert disagreements.empty


def test_find_gim_disagreements_reports_a_real_cross_tree_difference(tmp_path):
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 0.8314}],
    )
    write_daily_summary(
        tmp_path / VTEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 0.7787}],
    )

    disagreements = pc.find_gim_disagreements("iono", tmp_path)

    assert len(disagreements) == 1
    row = disagreements.iloc[0]
    assert row["doy"] == 150
    assert row["station"] == "AMC4"
    assert row["max_abs_diff"] == pytest.approx(0.8314 - 0.7787)
    assert set(row["trees"].split(",")) == {"STEC", "VTEC"}


def test_find_gim_disagreements_ignores_a_single_tree_offering(tmp_path):
    """Nothing to compare when only one tree has a genuine row for this station-day -
    not a disagreement, just an unpaired observation."""
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 5.0}],
    )

    disagreements = pc.find_gim_disagreements("iono", tmp_path)

    assert disagreements.empty


def test_find_gim_disagreements_excludes_a_known_foreign_doy_contaminated_row(tmp_path):
    """Reproduces the real 2026-09-17 finding: 50 of 51 reported elev-weighting
    cross-tree GIM disagreements were not genuine solver differences at all - they were
    `Finetune_STEC_2024_170_..._SWI`'s stray `results/2024122/` subdirectory (a foreign
    DOY's rows filed under the wrong experiment) being read as if it were a second,
    disagreeing STEC-tree offering for DOY 122. `collect()` already excludes exactly
    this via `find_foreign_doy_rows` - `find_gim_disagreements` must apply the same
    exclusion, or it manufactures disagreements out of contamination it already knows
    how to name.

    Fixture: `build_fixture` gives every tree an agreeing AMC4/150 gim_iono=4.0. A
    second STEC-tree directory whose own DOY is 199 additionally contains a stray
    `results/2024150/` subdirectory with a wildly different AMC4 gim_iono value - the
    same shape as the real anomaly, own_doy != foreign_results_doy. Without the fix,
    that stray row reads as a second "STEC" offering and manufactures a disagreement
    against the genuine 4.0 rows every tree actually agrees on.
    """
    build_fixture(tmp_path)
    foreign_owner_dir = f"Finetune_STEC_2024_199_{pc.CANONICAL_STEC_SUFFIX}"
    write_daily_summary(
        tmp_path
        / foreign_owner_dir
        / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 9.0}],
    )

    disagreements = pc.find_gim_disagreements("iono", tmp_path)

    assert disagreements.empty, (
        "a foreign-DOY contaminated row must not manufacture a cross-tree "
        f"disagreement:\n{disagreements}"
    )


def test_find_gim_disagreements_ignores_rounding_sized_differences(tmp_path):
    write_daily_summary(
        tmp_path / STEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 0.83140}],
    )
    write_daily_summary(
        tmp_path / VTEC_DIR / "positioning/results/2024150/daily_summary_iono.csv",
        [{"station": "AMC4", "method": "gim_iono", "e_rms": 0.83149}],
    )

    disagreements = pc.find_gim_disagreements("iono", tmp_path)

    assert disagreements.empty


# ---------------------------------------------------------------------------
# find_solver_failure_station_days(): the solver/geometry-failure exclusion
# (owner decision 2026-09-18, docs/revision/positioning_reporting.md) - a
# station-day drops out only when *every one* of the eight arms (all four
# methods, both weightings) reports error_3d_rms above SOLVER_FAILURE_THRESHOLD_M.
# This is independent of which method produced the number, unlike the outcome-based
# filters the owner has separately and permanently rejected - so a station-day where
# only some arms are bad (CHPG/248, POVE/124 in the live data) must stay in.
# ---------------------------------------------------------------------------


def all_weightings_row(
    station: str, doy: int, method: str, error_3d_rms: float
) -> dict:
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d_rms,
    }


def eight_arm_rows(station: str, doy: int, error_3d_rms: float) -> list[dict]:
    """One row per arm in `pc.SOLVER_FAILURE_ARMS`, all at the same error value -
    the shape a genuine all-methods, all-weightings solver failure takes."""
    return [
        all_weightings_row(station, doy, arm, error_3d_rms)
        for arm in pc.SOLVER_FAILURE_ARMS
    ]


def test_find_solver_failure_station_days_excludes_when_every_arm_exceeds_threshold():
    """URUM/DOY 365 in the live data: all eight solutions between 5,880 and 5,990 m."""
    frame = pd.DataFrame(eight_arm_rows("URUM", 365, 5900.0))

    failures = pc.find_solver_failure_station_days(frame)

    assert list(zip(failures["station"], failures["doy"])) == [("URUM", 365)]


def test_find_solver_failure_station_days_keeps_station_day_with_some_good_arms():
    """CHPG/DOY 248 in the live data: some arms exceed 100 m, others (e.g. the GIM
    arms) are single digits - a per-method problem, not a solver/geometry failure,
    and must not be excluded."""
    rows = eight_arm_rows("CHPG", 248, 150.0)
    for row in rows:
        if row["method"] in ("gim_iono", "gim_elev"):
            row["error_3d_rms"] = 8.0
    frame = pd.DataFrame(rows)

    failures = pc.find_solver_failure_station_days(frame)

    assert failures.empty


def test_find_solver_failure_station_days_keeps_station_day_missing_an_arm():
    """A station-day short of one of the eight arms is not "every solution exceeds
    the threshold" - it is already outside the common set for a different reason
    (missing coverage), so this exclusion must not also claim it."""
    rows = [
        row for row in eight_arm_rows("ZIMM", 133, 500.0) if row["method"] != "gim_elev"
    ]
    frame = pd.DataFrame(rows)

    failures = pc.find_solver_failure_station_days(frame)

    assert failures.empty


def test_find_solver_failure_station_days_boundary_at_exactly_threshold_is_kept():
    """ "Exceeds" means strictly greater than - a station-day sitting exactly at
    100 m on every arm must not be excluded."""
    frame = pd.DataFrame(eight_arm_rows("AMC4", 132, pc.SOLVER_FAILURE_THRESHOLD_M))

    failures = pc.find_solver_failure_station_days(frame)

    assert failures.empty


def test_find_solver_failure_station_days_reports_each_arms_error(tmp_path):
    frame = pd.DataFrame(eight_arm_rows("URUM", 365, 5900.0))

    failures = pc.find_solver_failure_station_days(frame)

    row = failures.iloc[0]
    for arm in pc.SOLVER_FAILURE_ARMS:
        assert row[arm] == pytest.approx(5900.0)


def test_solver_failure_station_day_set_reads_back_a_written_exclusion_csv(tmp_path):
    frame = pd.DataFrame(eight_arm_rows("URUM", 365, 5900.0))
    failures = pc.find_solver_failure_station_days(frame)
    csv_path = tmp_path / pc.EXCLUDED_SOLVER_FAILURES_FILENAME
    failures.to_csv(csv_path, index=False)

    excluded = pc.solver_failure_station_day_set(csv_path)

    assert excluded == {("URUM", 365)}


def test_solver_failure_station_day_set_empty_when_file_does_not_exist(tmp_path):
    assert pc.solver_failure_station_day_set(tmp_path / "does_not_exist.csv") == set()
