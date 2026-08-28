"""Tests for `stec.analysis.positioning_quality_gate`, with hand-computed expected values.

Built for the owner's request (2026-08-28) to replace the outcome-based 10 m station-day
rule with a method-blind quality gate. Every aggregation function here is exercised
against tiny, hand-computable frames rather than the real 43,101-row summary, the same
convention `test_positioning_diagnostics.py` and `test_positioning_geography.py` use.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import positioning_quality_gate as pqg

STEC = "Direct STEC"
GIM = "IGS GIM + Mapping"
VTEC = "VTEC + Mapping"
PRETRAINED = "Pretrained Direct STEC"


def quality_frame(rows: list[tuple]) -> pd.DataFrame:
    """rows of (station, Method, doy, error_3d_rms, n_epochs, mean_nsat, ref_source)."""
    return pd.DataFrame(
        rows,
        columns=[
            "station", "Method", "doy", "error_3d_rms",
            "n_epochs", "mean_nsat", "ref_source",
        ],
    )  # fmt: skip


# ---------------------------------------------------------------------------
# 1. column_inventory - outcome-derived columns must show r~1, independent ones r~0
# ---------------------------------------------------------------------------


def test_column_inventory_flags_outcome_derived_columns_as_not_independent():
    n = 20
    error = pd.Series(range(1, n + 1), dtype=float)
    df = pd.DataFrame({"error_3d_rms": error})
    for column in pqg.OUTCOME_DERIVED_COLUMNS:
        # Exactly proportional to the outcome - the worst case a real "e_std" etc. is
        # ever slightly less than in practice (r=0.996-1.000), so r=1 here is realistic.
        df[column] = error * 2.0
    df["n_epochs"] = 2880  # constant - genuinely unrelated to the outcome
    df["mean_nsat"] = 15.0  # constant - same

    inventory = pqg.column_inventory(df).set_index("column")

    for column in pqg.OUTCOME_DERIVED_COLUMNS:
        assert inventory.loc[column, "outcome_independent"] == False  # noqa: E712 (numpy bool)
        assert inventory.loc[column, "pearson_r_vs_error_3d_rms"] == pytest.approx(1.0)
    for column in pqg.OUTCOME_INDEPENDENT_NUMERIC_COLUMNS:
        assert inventory.loc[column, "outcome_independent"] == True  # noqa: E712 (numpy bool)
    # A constant column has undefined correlation (zero variance) - NaN, not a spurious 0.
    assert pd.isna(inventory.loc["n_epochs", "pearson_r_vs_error_3d_rms"])


# ---------------------------------------------------------------------------
# 2. cross_method_consistency
# ---------------------------------------------------------------------------


def test_cross_method_consistency_reports_zero_range_when_methods_agree():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("AAAA", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
            ("BBBB", STEC, 100, 1.5, 2880, 16.0, "ground_truth"),
            ("BBBB", GIM, 100, 2.5, 2880, 16.0, "ground_truth"),
        ]
    )
    result = pqg.cross_method_consistency(df, "n_epochs")
    assert result["n_station_days_all_4_methods"].iloc[0] == 2
    assert result["pct_exact_match"].iloc[0] == pytest.approx(100.0)
    assert result["range_max"].iloc[0] == pytest.approx(0.0)


def test_cross_method_consistency_detects_a_real_disagreement():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 14.0, "ground_truth"),
            (
                "AAAA",
                GIM,
                100,
                2.0,
                2880,
                16.0,
                "ground_truth",
            ),  # +2 satellites vs STEC
        ]
    )
    result = pqg.cross_method_consistency(df, "mean_nsat")
    assert result["n_station_days_all_4_methods"].iloc[0] == 1
    assert result["pct_exact_match"].iloc[0] == pytest.approx(0.0)
    assert result["range_max"].iloc[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 3. apply_quality_gate
# ---------------------------------------------------------------------------


def test_apply_quality_gate_drops_mean_referenced_rows_by_default():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("BBBB", STEC, 100, 1.0, 2880, 15.0, "mean"),
        ]
    )
    kept = pqg.apply_quality_gate(df, min_epoch_fraction=0.0)
    assert list(kept["station"]) == ["AAAA"]


def test_apply_quality_gate_can_keep_mean_referenced_rows_if_asked():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("BBBB", STEC, 100, 1.0, 2880, 15.0, "mean"),
        ]
    )
    kept = pqg.apply_quality_gate(
        df, min_epoch_fraction=0.0, require_ground_truth=False
    )
    assert set(kept["station"]) == {"AAAA", "BBBB"}


def test_apply_quality_gate_epoch_fraction_threshold():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),  # full day
            ("BBBB", STEC, 100, 1.0, 1000, 15.0, "ground_truth"),  # ~35% of a day
        ]
    )
    kept_lenient = pqg.apply_quality_gate(df, min_epoch_fraction=0.0)
    assert len(kept_lenient) == 2
    kept_strict = pqg.apply_quality_gate(df, min_epoch_fraction=0.9)
    assert list(kept_strict["station"]) == ["AAAA"]


def test_apply_quality_gate_min_nsat_is_off_by_default():
    df = quality_frame([("AAAA", STEC, 100, 1.0, 2880, 6.0, "ground_truth")])
    # Recommended gate (no min_nsat argument) must not touch a low-nsat row - that
    # column is a rejected variant, not part of the default gate.
    kept = pqg.apply_quality_gate(df, min_epoch_fraction=0.0)
    assert len(kept) == 1
    kept_with_floor = pqg.apply_quality_gate(df, min_epoch_fraction=0.0, min_nsat=7.0)
    assert len(kept_with_floor) == 0


# ---------------------------------------------------------------------------
# 4. quality_gate_sensitivity / _headline_row (Table-5 convention: independent
#    per-method mean/median, not paired station-days)
# ---------------------------------------------------------------------------


def test_headline_row_matches_hand_computed_improvement():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("BBBB", STEC, 100, 3.0, 2880, 15.0, "ground_truth"),
            ("AAAA", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
            ("BBBB", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
        ]
    )
    row = pqg._headline_row(df, "test")
    # stec_mean = 2.0, gim_mean = 2.0 -> 0% improvement.
    assert row["stec_mean_m"] == pytest.approx(2.0)
    assert row["gim_mean_m"] == pytest.approx(2.0)
    assert row["mean_improvement_pct"] == pytest.approx(0.0)
    assert row["n_stec"] == 2
    assert row["n_gim"] == 2


def test_quality_gate_sensitivity_removes_short_sessions_as_fraction_tightens():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("AAAA", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
            ("BBBB", STEC, 100, 5.0, 1000, 15.0, "ground_truth"),  # short session
            ("BBBB", GIM, 100, 5.0, 1000, 15.0, "ground_truth"),
        ]
    )
    result = pqg.quality_gate_sensitivity(df, epoch_fractions=(0.0, 0.9)).set_index(
        "min_epoch_fraction"
    )
    assert result.loc[0.0, "n_stec"] == 2
    assert result.loc[0.9, "n_stec"] == 1  # BBBB's short session dropped


# ---------------------------------------------------------------------------
# 5. nsat_gate_asymmetry - the rejected-variant demonstration
# ---------------------------------------------------------------------------


def test_nsat_gate_asymmetry_shows_unequal_removal_across_methods():
    # GIM systematically has more satellites than STEC for the same station-day -
    # a floor should remove more STEC rows than GIM rows, as in the real data.
    rows = []
    for doy in range(100, 110):
        rows.append(
            ("AAAA", STEC, doy, 1.0, 2880, 6.0, "ground_truth")
        )  # below any floor
        rows.append(
            ("AAAA", GIM, doy, 1.0, 2880, 12.0, "ground_truth")
        )  # above every floor
        rows.append(("AAAA", VTEC, doy, 1.0, 2880, 6.0, "ground_truth"))
        rows.append(("AAAA", PRETRAINED, doy, 1.0, 2880, 6.0, "ground_truth"))
    df = quality_frame(rows)
    result = pqg.nsat_gate_asymmetry(df, nsat_floors=(None, 7.0)).set_index("min_nsat")
    assert result.loc[7.0, "pct_removed_Direct_STEC"] == pytest.approx(100.0)
    assert result.loc[7.0, "pct_removed_IGS_GIM__Mapping"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. station_gim_availability / station_exclusion_candidates
# ---------------------------------------------------------------------------


def test_station_gim_availability_counts_missing_days_against_the_full_universe():
    coverage = pd.DataFrame(
        {
            "doy": [100, 101, 102, 100, 101, 102],
            "station": ["AAAA", "AAAA", "AAAA", "BBBB", "BBBB", "BBBB"],
        }
    )
    # AAAA solved on all 3 days it appears; BBBB removed for doy 102 below, so 1/3
    # missing. Both stations are given all-3-day universes via `all_stations`.
    coverage = coverage.iloc[:-1]  # drop BBBB/doy102 -> BBBB missing 1 of 3 days
    result = pqg.station_gim_availability(coverage, ["AAAA", "BBBB"]).set_index(
        "station"
    )
    assert result.loc["AAAA", "pct_gim_missing"] == pytest.approx(0.0)
    assert result.loc["BBBB", "pct_gim_missing"] == pytest.approx(100 / 3)


def test_station_exclusion_candidates_selects_only_stations_at_or_above_threshold():
    availability = pd.DataFrame(
        {
            "station": ["AAAA", "BBBB", "CCCC"],
            "pct_gim_missing": [90.0, 43.0, 5.0],
        }
    )
    tier1 = pqg.station_exclusion_candidates(availability, threshold_pct=80.0)
    assert tier1 == ["AAAA"]
    lenient = pqg.station_exclusion_candidates(availability, threshold_pct=40.0)
    assert set(lenient) == {"AAAA", "BBBB"}


# ---------------------------------------------------------------------------
# 7. station_exclusion_sensitivity - includes the "within excluded stations" check
#    that guards against the circularity the owner explicitly warned about.
# ---------------------------------------------------------------------------


def test_station_exclusion_sensitivity_reports_within_excluded_population_separately():
    df = quality_frame(
        [
            # Good station: STEC beats GIM.
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("AAAA", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
            # Excluded station: STEC loses badly to GIM.
            ("ZZZZ", STEC, 100, 9.0, 2880, 15.0, "ground_truth"),
            ("ZZZZ", GIM, 100, 1.0, 2880, 15.0, "ground_truth"),
        ]
    )
    result = pqg.station_exclusion_sensitivity(
        df, {"zzzz": ["ZZZZ"]}, threshold=100.0
    ).set_index("label")

    baseline = result.loc["baseline_10m_rule_all_stations"]
    assert baseline["n_stec"] == 2

    excluded = result.loc["exclude_zzzz"]
    assert excluded["n_stec"] == 1
    assert excluded["stec_mean_m"] == pytest.approx(1.0)
    assert excluded["mean_improvement_pct"] == pytest.approx(50.0)  # 1.0 vs 2.0

    within = result.loc["within_excluded_zzzz_only"]
    assert within["n_stec"] == 1
    assert within["mean_improvement_pct"] < 0  # STEC loses inside the excluded station


# ---------------------------------------------------------------------------
# 8. recovered_overlap_with_excluded_stations
# ---------------------------------------------------------------------------


def test_recovered_overlap_computes_baseline_and_recovered_shares_independently():
    coverage = pd.DataFrame(
        {
            "doy": [100, 100, 100, 101],
            "station": ["AAAA", "BBBB", "ZZZZ", "AAAA"],
        }
    )
    recovered = pd.DataFrame({"doy": [100], "station": ["ZZZZ"]})
    result = pqg.recovered_overlap_with_excluded_stations(recovered, coverage, ["ZZZZ"])
    assert result["n_excluded_stations"] == 1
    assert result["baseline_share_pct"] == pytest.approx(25.0)  # 1 of 4 station-days
    assert result["recovered_share_pct"] == pytest.approx(100.0)  # ZZZZ is 1 of 1


# ---------------------------------------------------------------------------
# 9. population_split_under_gate - thin integration test over the reused
#    positioning_diagnostics functions, confirming the gate is applied first.
# ---------------------------------------------------------------------------


def test_population_split_under_gate_excludes_short_sessions_before_splitting():
    df = quality_frame(
        [
            ("AAAA", STEC, 100, 1.0, 2880, 15.0, "ground_truth"),
            ("AAAA", GIM, 100, 2.0, 2880, 15.0, "ground_truth"),
            # Short session - must be gated out before the population split runs.
            ("BBBB", STEC, 100, 50.0, 100, 15.0, "ground_truth"),
            ("BBBB", GIM, 100, 50.0, 100, 15.0, "ground_truth"),
        ]
    )
    recovered = pd.DataFrame(columns=["doy", "station"])
    result = pqg.population_split_under_gate(df, recovered)
    row = result[result["population"] == "original"].iloc[0]
    assert row["n_stec"] == 1  # BBBB's short session excluded, only AAAA survives
    assert row["stec_mean_m"] == pytest.approx(1.0)
