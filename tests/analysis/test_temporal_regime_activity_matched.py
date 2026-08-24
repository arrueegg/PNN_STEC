"""Pins the activity-matching logic that corrects R2.1's solar-cycle confound.

Follows `test_temporal_regime_split.py`'s style: synthetic days are written through
`prediction_store.write_predictions` so the test exercises the real on-disk format, and
every statistic is also computed independently from the written frames rather than
trusted from the module under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import activity_stratification as act
from stec.analysis import temporal_regime_activity_matched as tram
from stec.inference import prediction_store as ps


def day_frame(rows: int, error: float, f107: float, seed: int) -> pd.DataFrame:
    """A day where every prediction misses truth by exactly `error` and every row shares
    one F10.7 value - matching the real store, where F10.7 is constant within a day."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(5, 60, rows)
    return pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(5, 90, rows),
            "true_stec": truth,
            "stec_pred": truth + error,
            "f107_index": [f107] * rows,
        }
    )


# --------------------------------------------------------------------------
# F10.7 bin labels: reused edges, flattened text (see the provenance-count note in the
# module docstring)
# --------------------------------------------------------------------------


def test_bin_edges_are_the_same_object_as_activity_stratification():
    """Bin *edges* must never drift from the paper's own STEC-domain activity
    stratification - collect() reuses act.F107_BINS directly, not a copy."""
    assert tram.F107_BINS is act.F107_BINS


def test_bin_labels_have_no_embedded_newline():
    """activity_stratification.F107_LABELS carry a literal newline for its plot axis;
    the pipeline's row-count provenance counts raw newlines, not CSV rows
    (stec/pipeline/provenance.py), so a label written into a CSV cell must be flat."""
    assert all("\n" not in label for label in tram.F107_BIN_LABELS)
    assert len(tram.F107_BIN_LABELS) == len(act.F107_LABELS)


# --------------------------------------------------------------------------
# collect(): streamed per-day sums against a direct whole-frame computation
# --------------------------------------------------------------------------


def test_collect_assigns_regime_and_f107_bin_per_day(tmp_path):
    # 2023 day: interpolation, low F10.7. 2024 DOY 200: extrapolation, elevated F10.7.
    ps.write_predictions(
        day_frame(50, error=2.0, f107=90.0, seed=1),
        "pretrained_stec",
        "own",
        2023,
        100,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(50, error=4.0, f107=180.0, seed=2),
        "pretrained_stec",
        "own",
        2024,
        200,
        root=tmp_path,
    )

    daily = tram.collect(store_root=tmp_path)

    row_2023 = daily[daily["year"] == 2023].iloc[0]
    assert row_2023["regime"] == "interpolation"
    assert row_2023["f107_bin"] == "low (< 100 sfu)"

    row_2024 = daily[daily["year"] == 2024].iloc[0]
    assert row_2024["regime"] == "extrapolation"
    assert row_2024["f107_bin"] == "elevated (150–200)"


def test_collect_excludes_non_finite_pairs(tmp_path):
    frame = day_frame(10, error=2.0, f107=110.0, seed=4)
    frame.loc[0, "stec_pred"] = np.nan
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 200, root=tmp_path)

    daily = tram.collect(store_root=tmp_path)
    assert daily.iloc[0]["n"] == 9


def test_collect_raises_when_store_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        tram.collect(store_root=tmp_path)


# --------------------------------------------------------------------------
# yearly_summary(): mean/median true STEC, RMSE, nRMSE per year
# --------------------------------------------------------------------------


def test_yearly_summary_mean_and_median_match_manual_computation(tmp_path):
    frame_a = day_frame(200, error=3.0, f107=100.0, seed=1)
    frame_b = day_frame(300, error=3.0, f107=105.0, seed=2)
    ps.write_predictions(frame_a, "pretrained_stec", "own", 2020, 50, root=tmp_path)
    ps.write_predictions(frame_b, "pretrained_stec", "own", 2020, 51, root=tmp_path)

    daily = tram.collect(store_root=tmp_path)
    yearly = tram.yearly_summary(daily)
    row = yearly[yearly["year"] == 2020].iloc[0]

    pooled_truth = np.concatenate(
        [frame_a["true_stec"].to_numpy(), frame_b["true_stec"].to_numpy()]
    )
    assert row["mean_true_stec"] == pytest.approx(pooled_truth.mean(), rel=1e-6)
    assert row["median_true_stec"] == pytest.approx(np.median(pooled_truth), rel=1e-6)
    assert row["n_obs"] == len(pooled_truth)
    assert row["n_days"] == 2

    pooled_error = 3.0  # constant offset for every row in both days
    expected_rmse = np.sqrt(np.mean(np.full(len(pooled_truth), pooled_error) ** 2))
    assert row["RMSE"] == pytest.approx(expected_rmse, rel=1e-6)


# --------------------------------------------------------------------------
# activity_matched_comparison(): pooled RMSE per (bin, regime), matched_bin flag
# --------------------------------------------------------------------------


def test_matched_bin_is_false_when_only_one_regime_present(tmp_path):
    # Only interpolation days, all in the same F10.7 bin - no extrapolation counterpart.
    ps.write_predictions(
        day_frame(100, error=2.0, f107=90.0, seed=1),
        "pretrained_stec",
        "own",
        2019,
        100,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(100, error=2.0, f107=95.0, seed=2),
        "pretrained_stec",
        "own",
        2020,
        100,
        root=tmp_path,
    )

    daily = tram.collect(store_root=tmp_path)
    matched = tram.activity_matched_comparison(daily)

    assert matched["matched_bin"].eq(False).all()
    assert set(matched["regime"]) == {"interpolation"}


def test_matched_bin_is_true_when_both_regimes_share_a_bin(tmp_path):
    # Same F10.7 bin (100-150), one interpolation day and one extrapolation day.
    ps.write_predictions(
        day_frame(100, error=2.0, f107=120.0, seed=1),
        "pretrained_stec",
        "own",
        2019,
        100,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(50, error=5.0, f107=125.0, seed=2),
        "pretrained_stec",
        "own",
        2024,
        200,
        root=tmp_path,
    )

    daily = tram.collect(store_root=tmp_path)
    matched = tram.activity_matched_comparison(daily)

    bin_rows = matched[matched["f107_bin"] == "moderate (100–150)"]
    assert len(bin_rows) == 2
    assert bin_rows["matched_bin"].eq(True).all()
    assert set(bin_rows["regime"]) == {"interpolation", "extrapolation"}


def test_activity_matched_comparison_pools_by_count_within_a_bin(tmp_path):
    """Two days sharing a (bin, regime) cell must pool by count-weighted RMSE, not
    average the two days' RMSE values equally - same convention as
    activity_stratification.py's _pool."""
    frame_a = day_frame(100, error=2.0, f107=110.0, seed=1)
    frame_b = day_frame(300, error=6.0, f107=115.0, seed=2)
    ps.write_predictions(frame_a, "pretrained_stec", "own", 2019, 100, root=tmp_path)
    ps.write_predictions(frame_b, "pretrained_stec", "own", 2019, 101, root=tmp_path)

    daily = tram.collect(store_root=tmp_path)
    matched = tram.activity_matched_comparison(daily)
    row = matched[matched["f107_bin"] == "moderate (100–150)"].iloc[0]

    expected_rmse = np.sqrt((100 * 2.0**2 + 300 * 6.0**2) / 400)
    assert row["RMSE"] == pytest.approx(expected_rmse, rel=1e-6)
    assert row["n_obs"] == 400
    assert row["n_days"] == 2


# --------------------------------------------------------------------------
# naive_regime_totals(): cross-check against temporal_regime_split's own number
# --------------------------------------------------------------------------


def test_naive_regime_totals_matches_temporal_regime_split(tmp_path):
    """This module's own unstratified regime comparison, computed from the same store,
    must agree with temporal_regime_split.collect() to the reported precision - the two
    stages read the identical store partition and must never silently diverge."""
    from stec.analysis import temporal_regime_split as trs

    frames = {
        (2019, 100): day_frame(200, error=2.0, f107=90.0, seed=1),
        (2024, 200): day_frame(300, error=6.0, f107=210.0, seed=2),
    }
    for (year, doy), frame in frames.items():
        ps.write_predictions(frame, "pretrained_stec", "own", year, doy, root=tmp_path)

    daily = tram.collect(store_root=tmp_path)
    naive = tram.naive_regime_totals(daily)

    reference = trs.collect(store_root=tmp_path)
    ref_by_regime = {
        "interpolation": reference.iloc[0],
        "extrapolation": reference.iloc[1],
    }
    for regime in ("interpolation", "extrapolation"):
        assert naive.loc[regime, "RMSE"] == pytest.approx(
            ref_by_regime[regime]["RMSE"], rel=1e-6
        )
        assert naive.loc[regime, "n_obs"] == ref_by_regime[regime]["count"]
