"""Exercises the diagnostic-plot parity figures (src/viz/{spatial,performance,
distributions,uncertainty}.py port) against synthetic frames, mirroring
`tests/viz/test_manuscript_figures.py`'s style: a `_save`-plumbing check, then one
end-to-end frame -> PNG/CSV path per figure, plus independent hand-computed checks on
the plotted-data CSV for a subset of figures (not calls back into the module's own
helpers - the point is to catch the module computing something other than what it draws).

Kept synthetic throughout - no read of the real prediction store, per the resource
limits this port was built under (see `stec/viz/diagnostic_figures.py`'s docstring).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stec.viz import diagnostic_figures as dfig
from stec.viz import style


def _synthetic_observation_frame(
    rows: int = 3000, seed: int = 0, offset: float = 3.0
) -> pd.DataFrame:
    """A per-observation frame shaped like the extended diagnostic cache's own columns,
    with a *constant* prediction offset - so every bin's MAE and RMSE are exactly
    `offset`, an exact assertion rather than a statistical one (same trick
    `test_manuscript_figures.py::_synthetic_observation_frame` uses)."""
    rng = np.random.default_rng(seed)
    true_stec = rng.uniform(0, 60, rows)
    total_unc = np.abs(rng.normal(3.0, 1.0, rows)) + 0.5
    return pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": true_stec + offset,
            "satele": rng.uniform(5, 90, rows),
            "satazi": rng.uniform(0, 360, rows),
            "lat_ipp": rng.uniform(-90, 90, rows),
            "lon_ipp": rng.uniform(-180, 180, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "local_time_hours": rng.uniform(0, 24, rows),
            "sod": rng.uniform(0, 86400, rows),
            "year": rng.integers(2020, 2021, rows),
            "doy": rng.integers(1, 366, rows),
            "pred_total_unc": total_unc,
            "pred_epistemic_unc": total_unc * 0.3,
            "pred_aleatoric_unc": total_unc * 0.7,
            "Kp_index": rng.uniform(0, 90, rows),
            "R_Sunspot_No": rng.uniform(0, 200, rows),
            "Dst-index,_nT": rng.uniform(-200, 50, rows),
            "AE-index,_nT": rng.uniform(0, 1500, rows),
            "ap_index,_nT": rng.uniform(0, 300, rows),
            "f107_index": rng.uniform(70, 250, rows),
        }
    )


# --------------------------------------------------------------------------
# _save plumbing
# --------------------------------------------------------------------------


def test_save_writes_titled_notitle_and_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("A title")
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    record = dfig._save(fig, "demo.png", "feature", tmp_path, data)

    target = tmp_path / dfig.CATEGORY_DIRS["feature"]
    assert (target / "demo.png").exists()
    assert (target / "demo_notitle.png").exists()
    written = pd.read_csv(target / "demo.csv")
    pd.testing.assert_frame_equal(written, data)
    assert record == {
        "figure": "demo",
        "category": "feature",
        "filename": str(target / "demo.png"),
        "n_data_rows": 2,
    }


# --------------------------------------------------------------------------
# Spatial error maps
# --------------------------------------------------------------------------


def test_fig_spatial_error_map_bin_values_match_an_independent_computation(tmp_path):
    """20 points, one bin (lat 0-5, lon 0-5), a mix of two known residuals - the
    published mean/MAE for that bin is computed here with plain numpy, not by calling
    any function in `diagnostic_figures`."""
    style.configure_plotting()
    n = 20
    true_stec = np.full(n, 10.0)
    # Half the points get +2 TECU, half get -4 TECU: mean residual (true-pred) is
    # 0.5*(-2) + 0.5*4 = 1.0; MAE is 0.5*2 + 0.5*4 = 3.0.
    pred_stec = true_stec.copy()
    pred_stec[: n // 2] += 2.0
    pred_stec[n // 2 :] -= 4.0
    df = pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": pred_stec,
            "lat_ipp": np.full(n, 2.5),
            "lon_ipp": np.full(n, 2.5),
        }
    )
    expected_mean_residual = float(np.mean(true_stec - pred_stec))
    expected_mae = float(np.mean(np.abs(true_stec - pred_stec)))

    records = dfig.fig_spatial_error_map(df, tmp_path)

    assert len(records) == 3
    mae_csv = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["spatial"] / "spatial_error_map_mae.csv"
    )
    residual_csv = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["spatial"] / "spatial_error_map_residual.csv"
    )
    count_csv = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["spatial"] / "spatial_error_map_count.csv"
    )
    assert len(mae_csv) == 1
    assert mae_csv["value"].iloc[0] == pytest.approx(expected_mae)
    assert residual_csv["value"].iloc[0] == pytest.approx(expected_mean_residual)
    assert count_csv["value"].iloc[0] == n
    for target in (
        "spatial_error_map_mae",
        "spatial_error_map_residual",
        "spatial_error_map_count",
    ):
        folder = tmp_path / dfig.CATEGORY_DIRS["spatial"]
        assert (folder / f"{target}.png").exists()
        assert (folder / f"{target}_notitle.png").exists()


def test_fig_spatial_error_map_drops_bins_below_the_minimum_count(tmp_path):
    style.configure_plotting()
    df = pd.DataFrame(
        {
            "true_stec": np.full(5, 10.0),
            "stec_pred": np.full(5, 11.0),
            "lat_ipp": np.full(5, 2.5),
            "lon_ipp": np.full(5, 2.5),
        }
    )
    records = dfig.fig_spatial_error_map(df, tmp_path)
    assert records == []


def test_fig_spatial_error_map_by_local_time_builds_end_to_end(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=2000)
    records = dfig.fig_spatial_error_map_by_local_time(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["spatial"]
    assert (target / "spatial_error_by_local_time.png").exists()
    assert (target / "spatial_error_by_local_time_notitle.png").exists()
    plotted = pd.read_csv(target / "spatial_error_by_local_time.csv")
    assert set(plotted["period"]) <= set(dfig._LOCAL_TIME_PERIODS)


# --------------------------------------------------------------------------
# Azimuth/elevation heatmap
# --------------------------------------------------------------------------


def test_fig_az_el_heatmap_cell_value_matches_an_independent_computation(tmp_path):
    """All points land in one (az, el) cell with two known residuals - hand-computed
    mean here, not derived by calling the module's own binning helper."""
    style.configure_plotting()
    n = 12
    true_stec = np.full(n, 20.0)
    pred_stec = true_stec.copy()
    pred_stec[:6] += 1.0
    pred_stec[6:] -= 3.0
    df = pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": pred_stec,
            "satazi": np.full(n, 15.0),  # inside the [10, 20) 10-degree bin
            "satele": np.full(n, 47.0),  # inside a [45, 50) 5-degree bin
        }
    )
    expected_mean_residual = float(np.mean(true_stec - pred_stec))
    expected_mae = float(np.mean(np.abs(true_stec - pred_stec)))

    residual_records = dfig.fig_az_el_heatmap(df, tmp_path, metric="residual")
    mae_records = dfig.fig_az_el_heatmap(df, tmp_path, metric="mae")

    assert len(residual_records) == 1
    assert len(mae_records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["spatial"]
    residual_csv = pd.read_csv(target / "residual_azimuth_elevation_heatmap.csv")
    mae_csv = pd.read_csv(target / "mae_azimuth_elevation_heatmap.csv")
    assert len(residual_csv) == 1
    assert len(mae_csv) == 1
    assert residual_csv["value"].iloc[0] == pytest.approx(expected_mean_residual)
    assert mae_csv["value"].iloc[0] == pytest.approx(expected_mae)
    assert residual_csv["count"].iloc[0] == n
    assert (target / "residual_azimuth_elevation_heatmap_notitle.png").exists()


def test_fig_az_el_heatmap_rejects_an_unknown_metric():
    with pytest.raises(ValueError):
        dfig.fig_az_el_heatmap(pd.DataFrame(), object(), metric="bogus")


# --------------------------------------------------------------------------
# Prediction scatter
# --------------------------------------------------------------------------


def test_fig_prediction_scatter_bin_counts_sum_to_the_input_rows(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=500)
    records = dfig.fig_prediction_scatter(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["performance"]
    assert (target / "prediction_scatter.png").exists()
    assert (target / "prediction_scatter_notitle.png").exists()
    plotted = pd.read_csv(target / "prediction_scatter.csv")
    # The histogram bin grid must account for every input point exactly.
    assert plotted["count"].sum() == 500


# --------------------------------------------------------------------------
# Residuals vs. date
# --------------------------------------------------------------------------


def test_fig_residuals_vs_date_builds_end_to_end(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=2000)
    records = dfig.fig_residuals_vs_date(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["temporal"]
    assert (target / "residuals_vs_date.png").exists()
    assert (target / "residuals_vs_date_notitle.png").exists()
    plotted = pd.read_csv(target / "residuals_vs_date.csv")
    assert plotted["count"].sum() == 2000


# --------------------------------------------------------------------------
# Residuals vs. feature axis (8 axes)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", dfig._FEATURE_AXES, ids=lambda a: a.name)
def test_fig_residuals_vs_feature_builds_end_to_end_for_every_axis(tmp_path, axis):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3000)
    records = dfig.fig_residuals_vs_feature(df, axis, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["feature"]
    filename = f"residual_vs_{axis.name}_boxplot"
    assert (target / f"{filename}.png").exists()
    assert (target / f"{filename}_notitle.png").exists()
    plotted = pd.read_csv(target / f"{filename}.csv")
    assert not plotted.empty


def test_fig_residuals_vs_feature_mae_rmse_match_the_constant_offset(tmp_path):
    """Constant +3 TECU prediction offset: every non-empty bin's MAE and RMSE must be
    exactly 3.0, independent of how the bins were formed."""
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=4000, offset=3.0)
    doy_axis = next(a for a in dfig._FEATURE_AXES if a.name == "doy")

    dfig.fig_residuals_vs_feature(df, doy_axis, tmp_path)

    plotted = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["feature"] / "residual_vs_doy_boxplot.csv"
    )
    assert not plotted.empty
    assert plotted["mae"].to_numpy() == pytest.approx(3.0)
    assert plotted["rmse"].to_numpy() == pytest.approx(3.0)


def test_fig_residuals_vs_feature_skips_a_missing_column(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=100).drop(columns=["Kp_index"])
    kp_axis = next(a for a in dfig._FEATURE_AXES if a.name == "kp")
    assert dfig.fig_residuals_vs_feature(df, kp_axis, tmp_path) == []


# --------------------------------------------------------------------------
# Histogram of residuals
# --------------------------------------------------------------------------


def test_fig_histogram_of_residuals_mean_and_std_match_numpy(tmp_path):
    style.configure_plotting()
    true_stec = np.array([10.0, 12.0, 15.0, 9.0, 20.0, 11.0, 14.0, 8.0])
    stec_pred = np.array([9.0, 13.5, 14.0, 9.5, 22.0, 10.0, 16.0, 7.0])
    df = pd.DataFrame({"true_stec": true_stec, "stec_pred": stec_pred})
    residual = true_stec - stec_pred
    expected_mean = float(np.mean(residual))
    expected_std = float(np.std(residual))

    records = dfig.fig_histogram_of_residuals(df, tmp_path)

    assert len(records) == 1
    plotted = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["feature"] / "residuals_histogram.csv"
    )
    assert plotted["mean_residual"].iloc[0] == pytest.approx(expected_mean)
    assert plotted["std_residual"].iloc[0] == pytest.approx(expected_std)
    assert plotted["n"].iloc[0] == len(true_stec)
    # The histogram is a density, so density * bin width must sum to ~1.
    bin_width = plotted["bin_right"].iloc[0] - plotted["bin_left"].iloc[0]
    assert (plotted["density"] * bin_width).sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Residuals vs. solar indices
# --------------------------------------------------------------------------


def test_fig_residuals_vs_solar_indices_builds_end_to_end(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=4000)
    records = dfig.fig_residuals_vs_solar_indices(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["feature"]
    assert (target / "residuals_vs_solar_indices.png").exists()
    plotted = pd.read_csv(target / "residuals_vs_solar_indices.csv")
    assert set(plotted["index"]) <= {a.name for a in dfig._SOLAR_INDEX_AXES}


def test_fig_residuals_vs_solar_indices_skips_when_no_index_column_present(tmp_path):
    df = pd.DataFrame({"true_stec": [1.0, 2.0], "stec_pred": [1.5, 2.5]})
    assert dfig.fig_residuals_vs_solar_indices(df, tmp_path) == []


# --------------------------------------------------------------------------
# Uncertainty diagnostics
# --------------------------------------------------------------------------


def test_fig_uncertainty_calibration_binned_builds_end_to_end(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=4000)
    records = dfig.fig_uncertainty_calibration_binned(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["uncertainty"]
    assert (target / "uncertainty_calibration_binned.png").exists()
    assert (target / "uncertainty_calibration_binned_notitle.png").exists()
    plotted = pd.read_csv(target / "uncertainty_calibration_binned.csv")
    assert plotted["count"].sum() == 4000


def test_fig_uncertainty_calibration_builds_end_to_end(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=4000)
    records = dfig.fig_uncertainty_calibration(df, tmp_path)
    assert len(records) == 1
    target = tmp_path / dfig.CATEGORY_DIRS["uncertainty"]
    assert (target / "uncertainty_calibration_scatter.png").exists()
    plotted = pd.read_csv(target / "uncertainty_calibration_scatter.csv")
    assert len(plotted) == 4000  # below the 10,000-point subsampling threshold


def test_fig_coverage_probability_matches_an_independent_computation(tmp_path):
    """8 points with known |residual| and a constant sigma: coverage at sigma=1 and
    sigma=2 is computed here directly, not through the module's own binning."""
    style.configure_plotting()
    abs_residual = np.array([0.5, 0.5, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5])
    true_stec = np.full(8, 10.0)
    stec_pred = true_stec - abs_residual  # true - pred = abs_residual (all positive)
    total_unc = np.full(8, 1.0)
    df = pd.DataFrame(
        {"true_stec": true_stec, "stec_pred": stec_pred, "pred_total_unc": total_unc}
    )
    expected_at_1sigma = float(np.mean(abs_residual <= 1.0 * total_unc))
    expected_at_2sigma = float(np.mean(abs_residual <= 2.0 * total_unc))

    records = dfig.fig_coverage_probability(df, tmp_path)

    assert len(records) == 1
    plotted = pd.read_csv(
        tmp_path
        / dfig.CATEGORY_DIRS["uncertainty"]
        / "uncertainty_coverage_probability.csv"
    )
    row_1sigma = plotted[np.isclose(plotted["sigma"], 1.0)].iloc[0]
    row_2sigma = plotted[np.isclose(plotted["sigma"], 2.0)].iloc[0]
    assert row_1sigma["observed_total"] == pytest.approx(expected_at_1sigma)
    assert row_2sigma["observed_total"] == pytest.approx(expected_at_2sigma)
    # No epistemic/aleatoric columns were supplied, so only the total series is present.
    assert "observed_epistemic" not in plotted.columns


def test_fig_sigma_coverage_comparison_matches_an_independent_computation(tmp_path):
    """8 points, constant sigma=1.5: 1-sigma coverage (|residual|<=1.5) is exactly 50%,
    2- and 3-sigma coverage (|residual|<=3.0/4.5) are exactly 100% - computed by hand,
    not through `fig_sigma_coverage_comparison`'s own `coverage()` closure."""
    style.configure_plotting()
    abs_residual = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
    true_stec = np.full(8, 10.0)
    stec_pred = true_stec - abs_residual
    total_unc = np.full(8, 1.5)
    df = pd.DataFrame(
        {"true_stec": true_stec, "stec_pred": stec_pred, "pred_total_unc": total_unc}
    )

    records = dfig.fig_sigma_coverage_comparison(df, tmp_path)

    assert len(records) == 1
    plotted = pd.read_csv(
        tmp_path / dfig.CATEGORY_DIRS["uncertainty"] / "sigma_coverage_comparison.csv"
    )
    plotted = plotted.set_index("sigma_level")
    assert plotted.loc["1sigma", "Total"] == pytest.approx(50.0)
    assert plotted.loc["2sigma", "Total"] == pytest.approx(100.0)
    assert plotted.loc["3sigma", "Total"] == pytest.approx(100.0)
    assert plotted.loc["1sigma", "Expected (perfect)"] == pytest.approx(68.27)


def test_fig_uncertainty_distributions_writes_a_distinct_filename_from_any_other_figure(
    tmp_path,
):
    """Regression guard for the src-side collision this port deliberately does not
    reproduce (see the module docstring): the only other histogram-writing figure in
    this module, `fig_histogram_of_residuals`, must use a different filename."""
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=2000)

    unc_records = dfig.fig_uncertainty_distributions(df, tmp_path)
    hist_records = dfig.fig_histogram_of_residuals(df, tmp_path)

    assert unc_records[0]["filename"] != hist_records[0]["filename"]
    target = tmp_path / dfig.CATEGORY_DIRS["uncertainty"]
    assert (target / "uncertainty_distributions.png").exists()
    plotted = pd.read_csv(target / "uncertainty_distributions.csv")
    assert set(plotted["component"]) == {
        "Total",
        "Epistemic (model)",
        "Aleatoric (data noise)",
    }
    for component, group in plotted.groupby("component"):
        assert group["count"].sum() == 2000


def test_fig_uncertainty_functions_skip_gracefully_without_pred_total_unc(tmp_path):
    df = pd.DataFrame({"true_stec": [1.0, 2.0], "stec_pred": [1.5, 2.5]})
    assert dfig.fig_uncertainty_calibration_binned(df, tmp_path) == []
    assert dfig.fig_uncertainty_calibration(df, tmp_path) == []
    assert dfig.fig_sigma_coverage_comparison(df, tmp_path) == []
    assert dfig.fig_uncertainty_distributions(df, tmp_path) == []


# --------------------------------------------------------------------------
# build_all / FIGURE_BUILDERS
# --------------------------------------------------------------------------


def test_build_all_runs_every_registered_builder_and_writes_a_manifest_worth_of_records(
    tmp_path,
):
    style.configure_plotting()
    # 60,000 rows (not the module's usual smaller synthetic size) so the >=10-per-bin
    # floor in fig_spatial_error_map's 2,592 (lat, lon) bins is reliably cleared -
    # otherwise this test's record count would depend on random bin luck.
    df = _synthetic_observation_frame(rows=60_000)

    records = dfig.build_all(df, tmp_path)

    # 3 (spatial map) + 1 (spatial by local time) + 2 (az/el) + 8 (feature axes) +
    # 5 (uncertainty) + 1 (prediction scatter) + 1 (residuals vs date) +
    # 1 (histogram) + 1 (solar indices) = 23, matching the module's coverage table.
    assert len(records) == 23
    for record in records:
        png_path = pd.io.common.stringify_path(record["filename"])
        assert png_path.endswith(".png")


def test_build_all_derives_local_time_when_absent(tmp_path):
    """`build_all` must call `add_local_time` itself so a cache built without
    `local_time_hours` (the real store's shape) still produces the local-time figures."""
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3000).drop(columns=["local_time_hours"])

    records = dfig.build_all(df, tmp_path)

    figures = {r["figure"] for r in records}
    assert "residual_vs_time_boxplot" in figures
    assert "spatial_error_by_local_time" in figures
