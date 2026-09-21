"""Pins the two-pass streaming design against a direct whole-frame computation, and the
offset-vector-agreement logic against synthetic data with a known common-mode offset and a
known method-specific one.

Follows the style of `test_daily_metrics.py` and `test_madrigal_reference_offset.py`:
synthetic days are written through `prediction_store.write_predictions` so every pass
exercises the real on-disk madrigal partition, not an in-memory shortcut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import madrigal_method_offset_comparison as mmoc
from stec.inference import prediction_store as ps


def madrigal_day_frame(
    rows_per_station: int,
    offsets: dict[str, dict[str, float]],
    seed: int,
    missing: list[str] | None = None,
) -> pd.DataFrame:
    """A Madrigal day where each method's residual against `true_stec` is a per-station
    constant offset plus noise. `offsets` maps method column -> {station: offset}."""
    rng = np.random.default_rng(seed)
    stations = list(next(iter(offsets.values())))
    frames = []
    for station in stations:
        truth = rng.uniform(0, 40, rows_per_station)
        row = {
            "station": station,
            "satele": rng.uniform(5, 90, rows_per_station),
            "true_stec": truth,
        }
        for column, per_station in offsets.items():
            row[column] = (
                truth + per_station[station] + rng.normal(0, 0.5, rows_per_station)
            )
        frames.append(pd.DataFrame(row))
    frame = pd.concat(frames, ignore_index=True)
    return frame.drop(columns=missing or [])


COMMON_MODE_OFFSETS = {
    "stec_pred": {"AAAA": 4.0, "BBBB": -2.0, "CCCC": 1.0},
    "pretrained_stec_pred": {"AAAA": 3.6, "BBBB": -1.8, "CCCC": 0.9},
    "vtec_model_stec": {"AAAA": 4.4, "BBBB": -2.2, "CCCC": 1.1},
    "gim_stec": {"AAAA": 3.8, "BBBB": -1.9, "CCCC": 0.95},
}


def write_days(tmp_path, offsets, days, missing=None):
    for doy, rows, seed in days:
        frame = madrigal_day_frame(rows, offsets, seed, missing=missing)
        ps.write_predictions(
            frame, "finetuned_stec", "madrigal", 2024, doy, root=tmp_path
        )


# --- pass 1: pairwise pooled stats and per-station offsets ---------------------------


def test_pass_one_pairwise_matches_direct_whole_frame_computation(
    tmp_path, monkeypatch
):
    """Pairwise pooled RMSE from streamed accumulation must equal the same statistic
    computed on the days concatenated into one frame - the correctness-check machinery
    this module reports against `daily_metrics`'s pooled_RMSE depends on this holding."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    days = [(150, 40, 1), (151, 60, 2), (152, 20, 3)]
    write_days(tmp_path, COMMON_MODE_OFFSETS, days)

    frames = []
    for doy, rows, seed in days:
        frames.append(madrigal_day_frame(rows, COMMON_MODE_OFFSETS, seed))
    whole = pd.concat(frames, ignore_index=True)

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")

    for column in mmoc.METHOD_COLUMNS:
        error = whole[column].to_numpy(float) - whole["true_stec"].to_numpy(float)
        expected_rmse = float(np.sqrt(np.mean(error**2)))
        assert pass_one.pairwise[column].rmse == pytest.approx(expected_rmse, rel=1e-5)
        assert pass_one.pairwise[column].n == len(whole)


def test_per_station_offset_table_matches_hand_computed_means(tmp_path, monkeypatch):
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    write_days(tmp_path, COMMON_MODE_OFFSETS, [(150, 5000, 7)])

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)

    for column, per_station in COMMON_MODE_OFFSETS.items():
        for station, injected in per_station.items():
            assert offsets.loc[station, f"offset_{column}"] == pytest.approx(
                injected, abs=0.05
            )


def test_offset_table_drops_stations_below_minimum_observations(tmp_path, monkeypatch):
    """A station with too few intersected rows must not appear in the offset table -
    its mean offset is not estimated reliably enough to correct anything with."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1000)
    small_offsets = {
        column: {"AAAA": v["AAAA"], "BBBB": v["BBBB"]}
        for column, v in COMMON_MODE_OFFSETS.items()
    }
    # AAAA clears the floor, BBBB does not.
    frame_a = madrigal_day_frame(1500, small_offsets, seed=1)
    frame_a = frame_a[frame_a["station"] == "AAAA"]
    frame_b = madrigal_day_frame(50, small_offsets, seed=2)
    frame_b = frame_b[frame_b["station"] == "BBBB"]
    combined = pd.concat([frame_a, frame_b], ignore_index=True)
    ps.write_predictions(
        combined, "finetuned_stec", "madrigal", 2024, 150, root=tmp_path
    )

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)

    assert "AAAA" in offsets.index
    assert "BBBB" not in offsets.index


# --- row intersection diagnostics -----------------------------------------------------


def test_row_diagnostics_reports_a_methods_own_nulls_and_the_intersection(
    tmp_path, monkeypatch
):
    """A method with NaN predictions on some rows must show up both as that method's own
    dropped-row count and as a smaller all-four intersection - the population the offset
    table and pass 2 are actually built on."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    rng = np.random.default_rng(3)
    n = 2000
    truth = rng.uniform(0, 40, n)
    frame = pd.DataFrame(
        {
            "station": ["AAAA"] * n,
            "satele": rng.uniform(5, 90, n),
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "pretrained_stec_pred": truth + 2.0,
            "vtec_model_stec": truth + 3.0,
            "gim_stec": truth + 4.0,
        }
    )
    # gim_stec is missing (NaN) on the last 200 rows only.
    frame.loc[frame.index[-200:], "gim_stec"] = np.nan
    ps.write_predictions(frame, "finetuned_stec", "madrigal", 2024, 150, root=tmp_path)

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    diagnostics = mmoc.build_row_diagnostics(pass_one)

    gim_row = diagnostics[diagnostics["Method"] == "IGS GIM"].iloc[0]
    assert gim_row["rows_dropped_to_this_methods_own_nulls"] == 200
    stec_row = diagnostics[diagnostics["Method"] == "Direct STEC Model"].iloc[0]
    assert stec_row["rows_dropped_to_this_methods_own_nulls"] == 0

    intersected_row = diagnostics[
        diagnostics["Method"] == "ALL FOUR METHODS (intersected)"
    ].iloc[0]
    assert intersected_row["rows_with_finite_truth_and_prediction"] == n - 200
    assert pass_one.intersected_rows == n - 200


# --- pass 2: offset-corrected pooled stats ---------------------------------------------


def test_corrected_rmse_and_mae_match_hand_computed_values(tmp_path, monkeypatch):
    """Pass 2 must remove exactly the per-station offset pass 1 computed, leaving a
    residual whose RMSE and MAE are worked out here by hand from the injected noise
    arrays - an independent calculation from `collect_pass_two` under test."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)

    # Deterministic, zero-mean noise per station so pass 1's offset is exactly the
    # injected constant and the corrected residual is exactly the noise array - no
    # reliance on a random seed averaging out.
    noise_a = np.array([1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 2.0, -2.0])
    noise_b = np.array([0.5, -0.5, 1.5, -1.5, 0.5, -0.5, 1.5, -1.5])
    offset_a, offset_b = 5.0, -3.0
    truth = np.full(8, 20.0)

    frame = pd.DataFrame(
        {
            "station": ["AAAA"] * 8 + ["BBBB"] * 8,
            "satele": np.tile(np.linspace(10, 80, 8), 2),
            "true_stec": np.concatenate([truth, truth]),
            "stec_pred": np.concatenate(
                [truth + offset_a + noise_a, truth + offset_b + noise_b]
            ),
            "pretrained_stec_pred": np.concatenate([truth, truth]),
            "vtec_model_stec": np.concatenate([truth, truth]),
            "gim_stec": np.concatenate([truth, truth]),
        }
    )
    ps.write_predictions(frame, "finetuned_stec", "madrigal", 2024, 160, root=tmp_path)

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)
    assert offsets.loc["AAAA", "offset_stec_pred"] == pytest.approx(offset_a)
    assert offsets.loc["BBBB", "offset_stec_pred"] == pytest.approx(offset_b)

    raw, corrected = mmoc.collect_pass_two(tmp_path, "finetuned_stec", offsets)

    residual = np.concatenate([offset_a + noise_a, offset_b + noise_b])
    corrected_residual = np.concatenate([noise_a, noise_b])
    assert raw["stec_pred"].rmse == pytest.approx(
        float(np.sqrt(np.mean(residual**2))), rel=1e-5
    )
    assert raw["stec_pred"].mae == pytest.approx(
        float(np.mean(np.abs(residual))), rel=1e-5
    )
    assert corrected["stec_pred"].rmse == pytest.approx(
        float(np.sqrt(np.mean(corrected_residual**2))), rel=1e-5
    )
    assert corrected["stec_pred"].mae == pytest.approx(
        float(np.mean(np.abs(corrected_residual))), rel=1e-5
    )
    # A method whose residual is already zero-mean per station must not change under
    # correction - the offset removed is exactly zero.
    assert corrected["gim_stec"].rmse == pytest.approx(raw["gim_stec"].rmse, rel=1e-5)


def test_correction_never_increases_pooled_rmse(tmp_path, monkeypatch):
    """Removing a per-station offset fitted on the same data can only reduce (or leave
    unchanged) the pooled RMSE - stated as a caveat in the module docstring and pinned
    here as an actual invariant, for every method."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    write_days(tmp_path, COMMON_MODE_OFFSETS, [(150, 3000, 11), (151, 2000, 12)])

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)
    raw, corrected = mmoc.collect_pass_two(tmp_path, "finetuned_stec", offsets)

    for column in mmoc.METHOD_COLUMNS:
        assert corrected[column].rmse <= raw[column].rmse + 1e-9


# --- ranking and offset-vector agreement -----------------------------------------------


def test_build_pooled_before_after_ranks_methods_correctly():
    raw = {
        "stec_pred": mmoc.RunningStats(n=100, sum_err=0, sum_sq=400, sum_abs=180),
        "pretrained_stec_pred": mmoc.RunningStats(
            n=100, sum_err=0, sum_sq=900, sum_abs=250
        ),
        "vtec_model_stec": mmoc.RunningStats(n=100, sum_err=0, sum_sq=100, sum_abs=90),
        "gim_stec": mmoc.RunningStats(n=100, sum_err=0, sum_sq=225, sum_abs=140),
    }
    corrected = raw  # unused for this ranking check beyond structure
    offsets = pd.DataFrame(
        {
            "offset_stec_pred": [1.0, -1.0],
            "offset_pretrained_stec_pred": [1.0, -1.0],
            "offset_vtec_model_stec": [1.0, -1.0],
            "offset_gim_stec": [1.0, -1.0],
        },
        index=["AAAA", "BBBB"],
    )

    table = mmoc.build_pooled_before_after(raw, corrected, offsets)

    # RMSE_before: VTEC (10) < GIM (15) < Direct STEC (20) < Pretrained (30)
    best = table.loc[table["rank_before_RMSE"] == 1, "Method"].iloc[0]
    worst = table.loc[table["rank_before_RMSE"] == 4, "Method"].iloc[0]
    assert best == "VTEC + Mapping"
    assert worst == "Pretrained STEC"


def test_offset_correlation_high_for_common_mode_offsets(tmp_path, monkeypatch):
    """When all four methods share essentially the same per-station offset pattern (the
    'reference offset' hypothesis), every pairwise Pearson correlation must be strongly
    positive."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    common_mode = {
        "stec_pred": {"AAAA": 5.0, "BBBB": -3.0, "CCCC": 1.0, "DDDD": 8.0},
        "pretrained_stec_pred": {"AAAA": 4.8, "BBBB": -2.9, "CCCC": 1.1, "DDDD": 7.7},
        "vtec_model_stec": {"AAAA": 5.2, "BBBB": -3.1, "CCCC": 0.9, "DDDD": 8.2},
        "gim_stec": {"AAAA": 4.9, "BBBB": -2.8, "CCCC": 1.05, "DDDD": 7.9},
    }
    write_days(tmp_path, common_mode, [(150, 4000, 21)])

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)
    correlation = mmoc.offset_correlation_matrix(offsets)

    assert (correlation["pearson_r"] > 0.9).all()


def test_offset_correlation_low_when_one_method_is_distinct(tmp_path, monkeypatch):
    """The discriminating case this module exists to catch: three methods share a
    per-station offset pattern, the fourth's offsets are unrelated to it. That method's
    pairwise correlations against the other three must come out near zero or negative,
    not swept into a spuriously high aggregate."""
    monkeypatch.setattr(mmoc, "MIN_OBSERVATIONS_PER_STATION", 1)
    mixed = {
        "stec_pred": {"AAAA": 5.0, "BBBB": -3.0, "CCCC": 1.0, "DDDD": 8.0},
        "pretrained_stec_pred": {"AAAA": 4.8, "BBBB": -2.9, "CCCC": 1.1, "DDDD": 7.7},
        "gim_stec": {"AAAA": 4.9, "BBBB": -2.8, "CCCC": 1.05, "DDDD": 7.9},
        # Unrelated to the pattern above by construction (reversed-and-shuffled).
        "vtec_model_stec": {"AAAA": -1.0, "BBBB": 6.0, "CCCC": -4.0, "DDDD": 0.5},
    }
    write_days(tmp_path, mixed, [(150, 4000, 31)])

    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    offsets = mmoc.build_offset_table(pass_one.per_station)
    correlation = mmoc.offset_correlation_matrix(offsets)

    vtec_pairs = correlation[
        (correlation["method_a"] == "VTEC + Mapping")
        | (correlation["method_b"] == "VTEC + Mapping")
    ]
    non_vtec_pairs = correlation[~correlation.index.isin(vtec_pairs.index)]
    assert (vtec_pairs["pearson_r"].abs() < 0.7).all()
    assert (non_vtec_pairs["pearson_r"] > 0.9).all()


# --- correctness check against daily_metrics -------------------------------------------


def test_compare_pairwise_to_daily_metrics_computes_delta(tmp_path):
    summary_path = tmp_path / "summary.csv"
    pd.DataFrame(
        {
            "dataset": ["madrigal_vtec_gim", "madrigal_vtec_gim"],
            "Model": ["Direct STEC Model", "VTEC + Mapping"],
            "pooled_RMSE": [15.01, 13.90],
        }
    ).to_csv(summary_path, index=False)

    pairwise = {
        "stec_pred": mmoc.RunningStats(n=10, sum_err=0, sum_sq=10 * 15.0**2, sum_abs=0),
        "pretrained_stec_pred": mmoc.RunningStats(),
        "vtec_model_stec": mmoc.RunningStats(
            n=10, sum_err=0, sum_sq=10 * 13.9**2, sum_abs=0
        ),
        "gim_stec": mmoc.RunningStats(),
    }

    comparison = mmoc.compare_pairwise_to_daily_metrics(pairwise, summary_path)

    direct = comparison[comparison["Method"] == "Direct STEC Model"].iloc[0]
    assert direct["daily_metrics_pooled_RMSE"] == pytest.approx(15.01)
    assert direct["delta"] == pytest.approx(15.0 - 15.01, abs=1e-6)
    vtec = comparison[comparison["Method"] == "VTEC + Mapping"].iloc[0]
    assert vtec["delta"] == pytest.approx(13.9 - 13.90, abs=1e-6)


def test_compare_pairwise_to_daily_metrics_returns_empty_when_summary_absent(tmp_path):
    pairwise = {col: mmoc.RunningStats() for col in mmoc.METHOD_COLUMNS}
    comparison = mmoc.compare_pairwise_to_daily_metrics(
        pairwise, tmp_path / "does_not_exist.csv"
    )
    assert comparison.empty


# --- edge cases --------------------------------------------------------------------


def test_collect_pass_one_returns_empty_when_store_is_absent(tmp_path):
    pass_one = mmoc.collect_pass_one(tmp_path, "finetuned_stec")
    assert pass_one.total_rows == 0
    assert pass_one.per_station == {}


def test_findings_markdown_reports_ranking_unchanged_when_stable():
    plain_table = pd.DataFrame(
        {
            "Method": ["A", "B", "C", "D"],
            "pooled_RMSE": [1.0, 2.0, 3.0, 4.0],
            "pooled_MAE": [1.0, 2.0, 3.0, 4.0],
            "R2_mean": [0.9, 0.8, 0.7, 0.6],
            "rank_RMSE": [1, 2, 3, 4],
        }
    )
    pooled_table = pd.DataFrame(
        {
            "Method": ["A", "B", "C", "D"],
            "observations": [100, 100, 100, 100],
            "qualifying_stations": [4, 4, 4, 4],
            "mean_abs_station_offset": [1.0, 1.2, 1.4, 1.6],
            "RMSE_before": [1.0, 2.0, 3.0, 4.0],
            "RMSE_after": [0.5, 1.5, 2.5, 3.5],
            "MAE_before": [1.0, 2.0, 3.0, 4.0],
            "MAE_after": [0.5, 1.5, 2.5, 3.5],
            "rank_before_RMSE": [1, 2, 3, 4],
            "rank_after_RMSE": [1, 2, 3, 4],
            "rank_before_MAE": [1, 2, 3, 4],
            "rank_after_MAE": [1, 2, 3, 4],
        }
    )
    correlation = pd.DataFrame(
        {
            "method_a": ["A"],
            "method_b": ["B"],
            "stations": [4],
            "pearson_r": [0.95],
            "pearson_p": [0.01],
            "spearman_rho": [0.9],
            "spearman_p": [0.02],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "Method": ["A"],
            "rows_with_finite_truth_and_prediction": [100],
            "rows_dropped_to_this_methods_own_nulls": [0],
            "pct_dropped": [0.0],
        }
    )

    text = mmoc._format_findings_markdown(
        plain_table, pooled_table, correlation, diagnostics, pd.DataFrame()
    )

    assert "The order does NOT change" in text
    assert "SENSITIVITY DIAGNOSTIC, NOT A RESULT" in text
    assert "A has the lowest RMSE and MAE and the highest R2" in text
