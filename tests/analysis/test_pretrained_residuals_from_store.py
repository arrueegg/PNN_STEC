"""Pins the from-scratch bin definitions and the local-time two-pass edge reproduction
against a direct whole-frame computation - the same equivalence
`test_daily_metrics.py::test_pooled_metrics_match_direct_whole_frame_computation` pins for
its own streamed accumulator, applied here to `pretrained_residuals_from_store.collect`.

Synthetic days are written through `prediction_store.write_predictions` so the streaming
tests exercise the real on-disk parquet format, matching `test_pretrained_test_diagnostics.
py`'s own style, never the real store.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stec.analysis import pretrained_residuals_from_store as prfs
from stec.inference import prediction_store as ps


def day_frame(rows: int, seed: int, missing: list[str] | None = None) -> pd.DataFrame:
    """A day of synthetic per-observation rows shaped like the pretrained store's own
    columns - full elevation/latitude/local-time range so every fixed bin gets some data
    across a handful of days."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 1.0, rows),
            "satele": rng.uniform(5, 90, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "sod": rng.uniform(0, 86400, rows),
            "lon_ipp": rng.uniform(-180, 180, rows),
        }
    )
    return frame.drop(columns=missing or [])


def _direct_bin_stats(
    residual: np.ndarray, values: np.ndarray, edges: np.ndarray, *, right: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A second, independent way to get per-bin MAE/RMSE/count from one concatenated
    frame - a plain groupby, not the module's bincount accumulator - so the streaming
    result is checked against a genuinely different computation, not itself."""
    codes = pd.cut(values, bins=edges, include_lowest=True, right=right).codes
    n_bins = len(edges) - 1
    mae = np.full(n_bins, np.nan)
    rmse = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=np.int64)
    for bin_index in range(n_bins):
        in_bin = codes == bin_index
        count[bin_index] = int(in_bin.sum())
        if count[bin_index] > 0:
            mae[bin_index] = np.abs(residual[in_bin]).mean()
            rmse[bin_index] = np.sqrt(np.mean(residual[in_bin] ** 2))
    return mae, rmse, count


def test_elevation_bins_match_direct_whole_frame_computation(tmp_path):
    days = [(2024, 130, 3000, 1), (2024, 131, 5000, 2), (2014, 152, 2000, 3)]
    for year, doy, rows, seed in days:
        ps.write_predictions(
            day_frame(rows, seed), "pretrained_stec", "own", year, doy, root=tmp_path
        )

    tables = prfs.collect(tmp_path)
    elevation = tables["residuals_elev"].sort_values("bin_left").reset_index(drop=True)

    # Read back what was actually written (float32-cast, per write_predictions), the
    # same bytes `collect` streams, so this comparison isolates the two accumulation
    # strategies (bincount vs a plain per-bin groupby) rather than also absorbing the
    # store's own float32 round-trip as noise.
    whole = ps.read_predictions(
        "pretrained_stec", "own", root=tmp_path, allow_full_scan=True
    )
    residual = (whole["true_stec"] - whole["stec_pred"]).to_numpy(dtype=float)
    expected_mae, expected_rmse, expected_n = _direct_bin_stats(
        residual,
        whole["satele"].to_numpy(dtype=float),
        prfs.ELEVATION_BIN_EDGES,
        right=True,
    )

    assert len(elevation) == len(prfs.ELEVATION_BIN_EDGES) - 1
    np.testing.assert_allclose(elevation["mae"].to_numpy(), expected_mae, rtol=1e-7)
    np.testing.assert_allclose(elevation["rmse"].to_numpy(), expected_rmse, rtol=1e-7)
    np.testing.assert_array_equal(elevation["n"].to_numpy(), expected_n)


def test_latitude_bins_match_direct_whole_frame_computation_and_keep_empty_bins(
    tmp_path,
):
    """Latitude bins are right-open (unlike elevation) and every one of the 18 fixed bins
    must appear even when a bin has no data, matching `fig_residuals_lat`'s `reindex`."""
    rng = np.random.default_rng(7)
    truth = rng.uniform(0, 60, 4000)
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 1.0, 4000),
            "satele": rng.uniform(5, 90, 4000),
            # Confined to [-40, 40): every bin outside that range must come back empty
            # (n=0, mae/rmse NaN) rather than being silently dropped from the table.
            "sm_lat_ipp": rng.uniform(-40, 40, 4000),
            "sod": rng.uniform(0, 86400, 4000),
            "lon_ipp": rng.uniform(-180, 180, 4000),
        }
    )
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 130, root=tmp_path)

    tables = prfs.collect(tmp_path)
    latitude = (
        tables["residuals_lat"].sort_values("lat_bin_center").reset_index(drop=True)
    )

    # Read back the float32-cast, on-disk version - see the elevation test's comment on
    # why this isolates the two accumulation strategies from the store's own round trip.
    whole = ps.read_predictions(
        "pretrained_stec", "own", root=tmp_path, allow_full_scan=True
    )
    residual = (whole["true_stec"] - whole["stec_pred"]).to_numpy(dtype=float)
    expected_mae, expected_rmse, expected_n = _direct_bin_stats(
        residual,
        whole["sm_lat_ipp"].to_numpy(dtype=float),
        prfs.LATITUDE_BIN_EDGES,
        right=False,
    )

    assert len(latitude) == len(prfs.LATITUDE_BIN_EDGES) - 1
    np.testing.assert_array_equal(latitude["n"].to_numpy(), expected_n)
    empty_bins = latitude["n"] == 0
    assert empty_bins.any(), "the -40..40 confinement should leave some bins empty"
    assert latitude.loc[empty_bins, "mae"].isna().all()
    assert latitude.loc[empty_bins, "rmse"].isna().all()
    filled = ~empty_bins
    np.testing.assert_allclose(
        latitude.loc[filled, "mae"].to_numpy(),
        expected_mae[filled.to_numpy()],
        rtol=1e-7,
    )
    np.testing.assert_allclose(
        latitude.loc[filled, "rmse"].to_numpy(),
        expected_rmse[filled.to_numpy()],
        rtol=1e-7,
    )


def test_year_month_accumulates_across_days_sharing_a_month(tmp_path):
    """Two different days in the same calendar month must pool into one row, and a day in
    a different year/month must not leak into it."""
    ps.write_predictions(
        day_frame(1000, seed=10), "pretrained_stec", "own", 2024, 130, root=tmp_path
    )  # 2024-05-09
    ps.write_predictions(
        day_frame(1000, seed=11), "pretrained_stec", "own", 2024, 135, root=tmp_path
    )  # 2024-05-14, same month as above
    ps.write_predictions(
        day_frame(500, seed=12), "pretrained_stec", "own", 2014, 200, root=tmp_path
    )  # a different year and month entirely

    tables = prfs.collect(tmp_path)
    year_month = tables["residuals_year_month"].set_index("year_month")

    assert set(year_month.index) == {"2024-05", "2014-07"}
    assert int(year_month.loc["2024-05", "n"]) == 2000
    assert int(year_month.loc["2014-07", "n"]) == 500


def test_local_time_edges_and_stats_match_direct_whole_frame_pd_cut(tmp_path):
    """The two-pass min/max-then-bin reproduction must land on the exact same 24 edges,
    and the same per-bin MAE/RMSE/count, as calling `pd.cut(bins=24)` once on every row
    concatenated into a single frame - the equivalence the module's docstring claims."""
    days = [(2024, 130, 4000, 21), (2024, 131, 3000, 22), (2014, 152, 2000, 23)]
    for year, doy, rows, seed in days:
        ps.write_predictions(
            day_frame(rows, seed), "pretrained_stec", "own", year, doy, root=tmp_path
        )

    tables = prfs.collect(tmp_path)
    localtime = tables["residuals_localtime"].sort_values("hour").reset_index(drop=True)

    # Read back the float32-cast, on-disk version - see the elevation test's comment.
    whole = ps.read_predictions(
        "pretrained_stec", "own", root=tmp_path, allow_full_scan=True
    )
    local_time_hours = (
        whole["sod"].to_numpy(dtype=float) / 3600.0
        + whole["lon_ipp"].to_numpy(dtype=float) / 15.0
    ) % 24.0
    residual = (whole["true_stec"] - whole["stec_pred"]).to_numpy(dtype=float)
    _, direct_edges = pd.cut(local_time_hours, bins=24, retbins=True)
    expected_mae, expected_rmse, expected_n = _direct_bin_stats(
        residual, local_time_hours, direct_edges, right=True
    )

    assert len(localtime) == 24
    np.testing.assert_array_equal(localtime["hour"].to_numpy(), np.arange(24))
    np.testing.assert_array_equal(localtime["n"].to_numpy(), expected_n)
    filled = expected_n > 0
    np.testing.assert_allclose(
        localtime.loc[filled, "mae"].to_numpy(), expected_mae[filled], rtol=1e-7
    )
    np.testing.assert_allclose(
        localtime.loc[filled, "rmse"].to_numpy(), expected_rmse[filled], rtol=1e-7
    )


def test_local_time_derived_when_column_absent_but_sod_lon_ipp_present(tmp_path):
    """`pretrained_stec/own` never carries a genuine `local_time_hours` column (per
    `pretrained_test_diagnostics.py`'s docstring) - this pins that the module still
    produces the table by deriving it, rather than skipping."""
    frame = day_frame(500, seed=30)
    assert "local_time_hours" not in frame.columns
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 130, root=tmp_path)

    tables = prfs.collect(tmp_path)

    assert "residuals_localtime" in tables
    assert tables["residuals_localtime"]["n"].sum() == 500


def test_no_local_time_source_skips_that_table_with_a_warning(tmp_path, caplog):
    frame = day_frame(500, seed=31, missing=["sod", "lon_ipp"])
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 130, root=tmp_path)

    with caplog.at_level(logging.WARNING):
        tables = prfs.collect(tmp_path)

    assert "residuals_localtime" not in tables
    assert "residuals_elev" in tables
    assert "skipping residuals_localtime" in caplog.text


def test_collect_raises_file_not_found_for_an_absent_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        prfs.collect(tmp_path)


def test_collect_years_filter_prevents_pooling_a_doy_across_years(tmp_path):
    """`pretrained_stec/own` holds the same doy in multiple years - the same pooling risk
    `test_pretrained_test_diagnostics.py` pins for `pretrained_test_diagnostics.collect`,
    checked here for this module's own `day_paths(years=...)` pass-through."""
    ps.write_predictions(
        day_frame(400, seed=40), "pretrained_stec", "own", 2024, 152, root=tmp_path
    )
    ps.write_predictions(
        day_frame(300, seed=41), "pretrained_stec", "own", 2014, 152, root=tmp_path
    )

    tables = prfs.collect(tmp_path, years=[2024])

    assert int(tables["residuals_elev"]["n"].sum()) == 400


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
