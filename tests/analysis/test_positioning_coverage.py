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
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stec.analysis import positioning_coverage as pc


def write_daily_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["station", "method", "e_rms"]).to_csv(path, index=False)


def build_fixture(root: Path) -> None:
    """One experiment directory per entry of `METHOD_TREES`, each with a single day's
    `daily_summary_iono.csv` under `positioning/results/2024150/`."""
    day_dir = "positioning/results/2024150/daily_summary_iono.csv"

    write_daily_summary(
        root / "Finetune_STEC_2024_150_BayesianResNetSTEC_h1_SWI" / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.0},
            {"station": "AMC4", "method": "gim", "e_rms": 4.0},
            {"station": "ZIMM", "method": "model", "e_rms": 2.0},
            {"station": "ZIMM", "method": "gim", "e_rms": 5.0},
            {"station": "WARK", "method": "gim", "e_rms": 7.0},
        ],
    )
    write_daily_summary(
        root / "Finetune_VTEC_2024_150_MLP_LaplacianNLL_h1_woYear" / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.1},
            {"station": "AMC4", "method": "gim", "e_rms": 4.0},
            {"station": "ZIMM", "method": "model", "e_rms": 2.1},
            {"station": "ZIMM", "method": "gim", "e_rms": 5.0},
            {"station": "NANO", "method": "model", "e_rms": 3.0},
        ],
    )
    write_daily_summary(
        root / "Pretrain_STEC_BayesianResNetSTEC_h1_SWI" / day_dir,
        [
            {"station": "AMC4", "method": "model", "e_rms": 1.2},
            {"station": "AMC4", "method": "gim", "e_rms": 4.0},
            # ZIMM deliberately absent: the "510" class - some ML methods missing.
        ],
    )


# ---------------------------------------------------------------------------
# collect(): reading and relabelling every per-day file on disk
# ---------------------------------------------------------------------------


def test_collect_relabels_methods_and_deduplicates_the_shared_gim_arm(tmp_path):
    build_fixture(tmp_path)
    combined = pc.collect("iono", tmp_path)

    # The GIM row for AMC4/150 is written identically into all three trees; it must
    # collapse to a single row rather than being triple-counted.
    gim_rows = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "gim_iono")
    ]
    assert len(gim_rows) == 1

    stec_rows = combined[
        (combined["station"] == "AMC4") & (combined["method"] == "STEC_iono")
    ]
    assert stec_rows["e_rms"].iloc[0] == pytest.approx(1.0)


def test_collect_raises_when_nothing_matches(tmp_path):
    with pytest.raises(SystemExit):
        pc.collect("iono", tmp_path)


# ---------------------------------------------------------------------------
# classify(): the three coverage causes
# ---------------------------------------------------------------------------


@pytest.fixture
def combined(tmp_path):
    build_fixture(tmp_path)
    return pc.collect("iono", tmp_path)


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
        tmp_path
        / "Finetune_STEC_2024_150_BayesianResNetSTEC_h1_SWI"
        / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "model", "e_rms": 1.0}],
    )
    write_daily_summary(
        tmp_path
        / "Finetune_VTEC_2024_150_MLP_LaplacianNLL_h1_woYear"
        / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "model", "e_rms": 1.1}],
    )
    write_daily_summary(
        tmp_path
        / "Pretrain_STEC_BayesianResNetSTEC_h1_SWI"
        / "positioning/results/2024150/daily_summary.csv",
        [{"station": "AMC4", "method": "gim", "e_rms": 4.0}],
    )

    combined = pc.collect("elev", tmp_path)
    assert set(combined["method"]) == {"STEC_elev", "VTEC_elev", "gim_elev"}
