"""Pins pass-1 streaming accumulation against a direct whole-frame computation, and
pass-2's decomposition against a value worked out by hand on synthetic data with a
known injected per-station offset.

Follows the style of `test_daily_metrics.py`: synthetic days are written through
`prediction_store.write_predictions` so both passes exercise the real on-disk madrigal
partition, not an in-memory shortcut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import madrigal_reference_offset as mro
from stec.inference import prediction_store as ps


def madrigal_day_frame(
    rows: int, model_offsets: dict[str, float], gim_offsets: dict[str, float], seed: int
) -> pd.DataFrame:
    """A Madrigal day where `stec_pred - true_stec` and `gim_stec - true_stec` are each
    a per-station constant offset plus noise. `true_stec` plays the role of the
    Madrigal reference in this store partition, matching `_iter_madrigal_days`'
    rename."""
    rng = np.random.default_rng(seed)
    stations = list(model_offsets)
    per_station_rows = rows // len(stations)
    station_col, truth_col, pred_col, gim_col, unc_col = [], [], [], [], []
    for station in stations:
        truth = rng.uniform(0, 40, per_station_rows)
        noise = rng.normal(0, 0.5, per_station_rows)
        station_col.extend([station] * per_station_rows)
        truth_col.extend(truth)
        pred_col.extend(truth + model_offsets[station] + noise)
        gim_col.extend(truth + gim_offsets[station])
        unc_col.extend(rng.uniform(1.0, 3.0, per_station_rows))
    n = len(station_col)
    return pd.DataFrame(
        {
            "station": station_col,
            "satele": rng.uniform(5, 90, n),
            "true_stec": truth_col,
            "stec_pred": pred_col,
            "gim_stec": gim_col,
            "pred_total_unc": unc_col,
        }
    )


def test_per_station_offsets_streaming_matches_whole_frame_direct_computation(
    tmp_path, monkeypatch
):
    """Mean per-station offset from streamed per-day accumulation must equal the
    same statistic computed on the days concatenated into one frame - the entire
    justification for the two-pass streaming design instead of reading the whole
    Madrigal store (which OOM-killed the source driver at 234 days)."""
    monkeypatch.setattr(mro, "MIN_OBSERVATIONS_PER_STATION", 1)
    model_offsets = {"AAAA": 4.0, "BBBB": -2.0}
    gim_offsets = {"AAAA": 3.0, "BBBB": -1.0}
    days = [(2024, 150, 40, 11), (2024, 151, 60, 12), (2024, 152, 20, 13)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = madrigal_day_frame(rows, model_offsets, gim_offsets, seed)
        frames[doy] = frame
        ps.write_predictions(
            frame, "finetuned_stec", "madrigal", year, doy, root=tmp_path
        )

    offsets = mro.per_station_offsets(tmp_path, "finetuned_stec")

    whole = pd.concat(frames.values(), ignore_index=True)
    for station in model_offsets:
        subset = whole[whole["station"] == station]
        direct_offset_model = (
            (subset["stec_pred"] - subset["true_stec"]).to_numpy(float).mean()
        )
        direct_offset_gim = (
            (subset["gim_stec"] - subset["true_stec"]).to_numpy(float).mean()
        )
        row = offsets.loc[station]
        # rel tolerance loosened to float32 precision: the store casts numeric
        # columns to float32 on write, so the round trip alone costs ~1e-7 relative.
        assert row["offset_model"] == pytest.approx(direct_offset_model, rel=1e-5)
        assert row["offset_gim"] == pytest.approx(direct_offset_gim, rel=1e-5)
        assert row["observations"] == len(subset)


def test_per_station_offsets_reads_every_matched_year_without_raising(
    tmp_path, monkeypatch
):
    """`_iter_madrigal_days` used to collapse `available_days()`'s (year, doy) pairs to
    a flat `doys=[...]` list with no `years=` before streaming. That is currently
    harmless only because `madrigal` is single-year for every model_variant that
    exists on disk today; the moment a multi-year Madrigal partition (e.g.
    `pretrained_stec/madrigal`, per CLAUDE.md's roadmap) is built, a doy shared across
    years would hit `prediction_store`'s new multi-year guard and this pass would
    start raising. Confirms it instead reads every matched year's file and pools them,
    which is this function's own intended behaviour."""
    monkeypatch.setattr(mro, "MIN_OBSERVATIONS_PER_STATION", 1)
    model_offsets = {"AAAA": 4.0}
    gim_offsets = {"AAAA": 3.0}
    frame_2020 = madrigal_day_frame(20, model_offsets, gim_offsets, seed=21)
    frame_2024 = madrigal_day_frame(30, model_offsets, gim_offsets, seed=22)
    ps.write_predictions(
        frame_2020, "pretrained_stec", "madrigal", 2020, 132, root=tmp_path
    )
    ps.write_predictions(
        frame_2024, "pretrained_stec", "madrigal", 2024, 132, root=tmp_path
    )

    offsets = mro.per_station_offsets(tmp_path, "pretrained_stec", doys=[132])

    assert offsets.loc["AAAA", "observations"] == 20 + 30


def test_per_station_offsets_returns_empty_frame_when_store_is_absent(tmp_path):
    """No store on disk must not raise - `main()` treats an empty result as a clean
    'nothing to report' rather than crashing pass 1."""
    offsets = mro.per_station_offsets(tmp_path, "finetuned_stec")
    assert offsets.empty


def test_decomposition_matches_hand_computed_residual_variance(tmp_path, monkeypatch):
    """Pass 2 must remove exactly the per-station offset pass 1 computed, leaving a
    residual whose RMSE is worked out here by hand from the injected noise arrays -
    an independent calculation from `decompose_and_coverage` under test."""
    monkeypatch.setattr(mro, "MIN_OBSERVATIONS_PER_STATION", 1)

    # Deterministic, zero-mean noise per station so the per-station offset pass 1
    # computes is exactly the injected constant, and the corrected residual is
    # exactly the noise array - no reliance on a random seed averaging out.
    noise_a = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    noise_b = np.array([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    offset_a, offset_b = 5.0, -3.0
    truth = np.full(10, 20.0)

    frame = pd.DataFrame(
        {
            "station": ["AAAA"] * 10 + ["BBBB"] * 10,
            "satele": np.tile(np.linspace(10, 80, 10), 2),
            "true_stec": np.concatenate([truth, truth]),
            "stec_pred": np.concatenate(
                [truth + offset_a + noise_a, truth + offset_b + noise_b]
            ),
            "gim_stec": np.concatenate([truth + offset_a, truth + offset_b]),
            "pred_total_unc": np.full(20, 2.0),
        }
    )
    ps.write_predictions(frame, "finetuned_stec", "madrigal", 2024, 160, root=tmp_path)

    offsets = mro.per_station_offsets(tmp_path, "finetuned_stec")
    assert offsets.loc["AAAA", "offset_model"] == pytest.approx(offset_a, rel=1e-5)
    assert offsets.loc["BBBB", "offset_model"] == pytest.approx(offset_b, rel=1e-5)

    summary, _coverage = mro.decompose_and_coverage(tmp_path, "finetuned_stec", offsets)

    residual = np.concatenate([offset_a + noise_a, offset_b + noise_b])
    corrected = np.concatenate([noise_a, noise_b])
    expected_rmse = float(np.sqrt(np.mean(residual**2)))
    expected_rmse_corrected = float(np.sqrt(np.mean(corrected**2)))

    assert summary["RMSE_vs_madrigal"] == pytest.approx(expected_rmse, rel=1e-4)
    assert summary["RMSE_after_removing_station_offset"] == pytest.approx(
        expected_rmse_corrected, rel=1e-4
    )
    expected_variance_explained = 100 * (
        1 - (expected_rmse_corrected / expected_rmse) ** 2
    )
    assert summary["variance_explained_by_offset_%"] == pytest.approx(
        expected_variance_explained, rel=1e-4
    )


def test_decompose_and_coverage_raises_when_no_observations_match_offsets(tmp_path):
    """A station table with no matching Madrigal rows in the store must fail loudly,
    not silently report a decomposition over zero observations."""
    offsets = pd.DataFrame(
        {"observations": [10000], "offset_model": [1.0]}, index=["ZZZZ"]
    )
    frame = pd.DataFrame(
        {
            "station": ["AAAA"] * 5,
            "satele": np.linspace(10, 80, 5),
            "true_stec": np.full(5, 20.0),
            "stec_pred": np.full(5, 21.0),
            "gim_stec": np.full(5, 20.5),
            "pred_total_unc": np.full(5, 2.0),
        }
    )
    ps.write_predictions(frame, "finetuned_stec", "madrigal", 2024, 170, root=tmp_path)

    with pytest.raises(RuntimeError, match="no Madrigal observations"):
        mro.decompose_and_coverage(tmp_path, "finetuned_stec", offsets)
