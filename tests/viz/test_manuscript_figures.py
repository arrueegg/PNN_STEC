"""Exercises the manuscript figure builders (all 14 code-generated figures) against
synthetic frames, mirroring `tests/viz/test_revision_figures.py`: a `_save`-plumbing
check, then one end-to-end CSV/list -> PNG path per figure family. Figures 4-9 use
deterministic residual/uncertainty offsets (not noise) wherever an exact MAE/RMSE value
makes the assertion tighter than a mere "the file exists" check.

Kept synthetic throughout - no read of `multiday_results/` or the prediction store, per
the resource limits this port was built under.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from stec.config import paths
from stec.viz import manuscript_figures as mf
from stec.viz import style


def test_save_writes_titled_notitle_and_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("A title")
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    mf._save(fig, "demo", "positioning", tmp_path, "source.csv - 2 rows", data)

    target = tmp_path / mf.SOURCE_DIRS["positioning"]
    assert (target / "demo.png").exists()
    assert (target / "demo_notitle.png").exists()
    written = pd.read_csv(target / "demo.csv")
    pd.testing.assert_frame_equal(written, data)


def test_save_requires_a_known_source_key(tmp_path):
    fig, _ax = plt.subplots()
    with pytest.raises(KeyError):
        mf._save(fig, "demo", "not_a_real_source", tmp_path, "prov")
    plt.close(fig)


# --------------------------------------------------------------------------
# Figure 1 - temporal split
# --------------------------------------------------------------------------


def test_fig_temporal_split_counts_only_months_inside_the_window(tmp_path):
    """A month outside [start_year, end_year] must not appear in the legend percentages
    or the plotted-data CSV - `visualize_temporal_splits.py` filters the same way."""
    style.configure_plotting()
    mf.fig_temporal_split(
        train_dates=["2020-01", "2020-02", "2019-12"],  # 2019 is out of window
        val_dates=["2020-03"],
        test_dates=["2020-04"],
        output_dir=tmp_path,
        provenance="synthetic",
        start_year=2020,
        end_year=2020,
    )
    target = tmp_path / mf.SOURCE_DIRS["dataset"]
    assert (target / "temp_split.png").exists()
    assert (target / "temp_split_notitle.png").exists()
    plotted = pd.read_csv(target / "temp_split.csv")
    assert len(plotted) == 4  # 2019-12 excluded
    assert plotted["year"].min() == 2020


def test_build_temporal_split_figure_reads_the_configured_date_lists(
    tmp_path, monkeypatch
):
    """One full list-of-files -> PNG path through the registered builder, with
    `stec.config.paths.SPLIT_LISTS` redirected to a synthetic tmp_path directory."""
    split_dir = tmp_path / "split_lists"
    split_dir.mkdir()
    (split_dir / "train_dates.list").write_text("2020-01\n2020-02\n")
    (split_dir / "val_dates.list").write_text("2020-03\n")
    (split_dir / "test_dates.list").write_text("2020-04\n")
    monkeypatch.setattr(paths, "SPLIT_LISTS", split_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=tmp_path / "results", output_dir=output_dir)
    style.configure_plotting()
    mf._build_temporal_split_figure(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["dataset"]
    assert (target / "temp_split.png").exists()


# --------------------------------------------------------------------------
# Figure 2 - spatial split
# --------------------------------------------------------------------------


def test_fig_spatial_split_builds_end_to_end_from_synthetic_stations(tmp_path):
    style.configure_plotting()
    train = pd.DataFrame(
        {"name": ["AAAA", "BBBB"], "lat": [10.0, -20.0], "lon": [30.0, -40.0]}
    )
    val = pd.DataFrame({"name": ["CCCC"], "lat": [5.0], "lon": [-10.0]})
    test = pd.DataFrame({"name": ["DDDD"], "lat": [50.0], "lon": [100.0]})

    mf.fig_spatial_split(train, val, test, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["dataset"]
    assert (target / "spatial_split.png").exists()
    assert (target / "spatial_split_notitle.png").exists()
    plotted = pd.read_csv(target / "spatial_split.csv")
    assert set(plotted["split"]) == {"Training", "Validation", "Test"}
    assert len(plotted) == 4


# --------------------------------------------------------------------------
# Figures 4-9 - per-observation residual/uncertainty diagnostics
# --------------------------------------------------------------------------


def _synthetic_observation_frame(
    rows: int = 3400, seed: int = 0, offset: float = 3.0
) -> pd.DataFrame:
    """A per-observation frame shaped like the prediction store's own column names, with
    a *constant* prediction offset rather than noise - so a bin's MAE and RMSE are both
    exactly `offset`, an exact assertion rather than a statistical one."""
    rng = np.random.default_rng(seed)
    true_stec = rng.uniform(0, 60, rows)
    total_unc = np.abs(rng.normal(3.0, 1.0, rows)) + 0.5
    return pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": true_stec + offset,
            "satele": rng.uniform(5, 90, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "local_time_hours": rng.uniform(0, 24, rows),
            "sod": rng.uniform(0, 86400, rows),
            "lon_ipp": rng.uniform(-180, 180, rows),
            "year": rng.integers(2020, 2021, rows),
            "doy": rng.integers(1, 366, rows),
            "pred_total_unc": total_unc,
            "pred_epistemic_unc": total_unc * 0.3,
            "pred_aleatoric_unc": total_unc * 0.7,
        }
    )


def test_fig_pred_density_writes_hexbin_and_the_underlying_points(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame()
    mf.fig_pred_density(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "pred_density.png").exists()
    assert (target / "pred_density_notitle.png").exists()
    plotted = pd.read_csv(target / "pred_density.csv")
    assert len(plotted) == len(df)


def test_fig_pred_density_max_limit_uses_a_distinct_filename(tmp_path):
    """The 300 TECU zoomed variant must not overwrite the full-range plot - the source
    wrote them as two separate files for the same reason."""
    style.configure_plotting()
    df = _synthetic_observation_frame()
    mf.fig_pred_density(df, tmp_path, "synthetic")
    mf.fig_pred_density(df, tmp_path, "synthetic", max_limit=300.0)

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "pred_density.png").exists()
    assert (target / "pred_density_limited.png").exists()


def test_fig_residuals_elev_reports_the_exact_constant_offset(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(offset=3.0)
    mf.fig_residuals_elev(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "residuals_elev.png").exists()
    plotted = pd.read_csv(target / "residuals_elev.csv")
    assert len(plotted) == mf._ELEVATION_NUM_BINS
    assert plotted["mae"].apply(lambda v: v == pytest.approx(3.0)).all()
    assert plotted["rmse"].apply(lambda v: v == pytest.approx(3.0)).all()


def test_fig_residuals_lat_keeps_empty_bins_as_nan_not_dropped(tmp_path):
    """Ported from `plot_box_by_lat`'s `reindex(all_bins)`: a latitude band with no
    observations must still appear as a bin (NaN MAE/RMSE), not vanish from the axis."""
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3400, offset=2.0)
    df = df[~df["sm_lat_ipp"].between(60, 70)]  # empty the [60, 70) bin deliberately

    mf.fig_residuals_lat(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    plotted = pd.read_csv(target / "residuals_lat.csv")
    assert len(plotted) == len(mf._GEOMAGNETIC_LAT_BIN_EDGES) - 1
    empty_bin = plotted[plotted["lat_bin_center"] == 65.0]
    assert empty_bin["mae"].isna().all()
    populated = plotted[plotted["lat_bin_center"] != 65.0]
    assert populated["mae"].apply(lambda v: v == pytest.approx(2.0)).all()


def test_fig_residuals_localtime_derives_time_from_sod_and_longitude(tmp_path):
    """The pretrained model was not configured with `local_time_hours` as an input, so
    its store rows lack the column; the figure must derive it via
    `stratified_comparison.add_local_time` rather than skip."""
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3400, offset=1.5).drop(
        columns=["local_time_hours"]
    )
    mf.fig_residuals_localtime(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "residuals_localtime.png").exists()
    plotted = pd.read_csv(target / "residuals_localtime.csv")
    assert len(plotted) == 24
    assert plotted["mae"].dropna().apply(lambda v: v == pytest.approx(1.5)).all()


def test_fig_residuals_localtime_skips_without_time_information(tmp_path, caplog):
    style.configure_plotting()
    df = _synthetic_observation_frame().drop(
        columns=["local_time_hours", "sod", "lon_ipp"]
    )
    with caplog.at_level(logging.WARNING):
        mf.fig_residuals_localtime(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert not target.exists()
    assert "local_time_hours" in caplog.text


def test_fig_residuals_year_month_reports_the_exact_constant_offset(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=600, offset=4.0)
    df["year"] = 2024
    df["doy"] = np.random.default_rng(1).integers(1, 366, len(df))

    mf.fig_residuals_year_month(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "residuals_year_month.png").exists()
    plotted = pd.read_csv(target / "residuals_year_month.csv")
    assert plotted["mae"].apply(lambda v: v == pytest.approx(4.0)).all()
    assert plotted["rmse"].apply(lambda v: v == pytest.approx(4.0)).all()
    # `order = sorted(year_month.unique())`, not `.unique()`'s own (insertion) order.
    assert list(plotted["year_month"]) == sorted(plotted["year_month"])


def test_fig_uncertainty_builds_with_all_four_curves(tmp_path):
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3400, offset=2.0)
    mf.fig_uncertainty(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    assert (target / "uncertainty.png").exists()
    plotted = pd.read_csv(target / "uncertainty.csv")
    assert plotted["mean_abs_error"].apply(lambda v: v == pytest.approx(2.0)).all()
    assert plotted["mean_epistemic_unc"].notna().all()
    assert plotted["mean_aleatoric_unc"].notna().all()


def test_fig_uncertainty_skips_without_pred_total_unc(tmp_path, caplog):
    style.configure_plotting()
    df = _synthetic_observation_frame().drop(columns=["pred_total_unc"])
    with caplog.at_level(logging.WARNING):
        mf.fig_uncertainty(df, tmp_path, "synthetic")

    assert not (tmp_path / mf.SOURCE_DIRS["pretrained"]).exists()
    assert "pred_total_unc" in caplog.text


def test_fig_uncertainty_skips_when_uncertainty_is_degenerate(tmp_path, caplog):
    """A deterministic model (uncertainty ~ 0 everywhere) cannot be binned by
    uncertainty value - the source's own guard, ported unchanged."""
    style.configure_plotting()
    df = _synthetic_observation_frame()
    df["pred_total_unc"] = 0.0
    with caplog.at_level(logging.WARNING):
        mf.fig_uncertainty(df, tmp_path, "synthetic")

    assert not (tmp_path / mf.SOURCE_DIRS["pretrained"]).exists()
    assert "too small to bin" in caplog.text


def _write_synthetic_diagnostics_cache(results_dir, rows: int = 3400) -> None:
    """The on-disk shape `_build_pretrained_diagnostics_figures` reads:
    `stec.analysis.pretrained_test_diagnostics`'s two outputs, built directly here rather
    than by importing that module - `_build_mae_rmse_finetuned_figure`'s own test writes
    its CSV the same self-contained way."""
    diagnostics_dir = mf.analysis_dir(results_dir, "pretrained_test_diagnostics")
    diagnostics_dir.mkdir(parents=True)
    observations = _synthetic_observation_frame(rows=rows)
    observations.to_parquet(diagnostics_dir / "observations.parquet", index=False)
    manifest = (
        observations.groupby("year")
        .agg(n_days=("doy", "nunique"), n_observations=("doy", "size"))
        .reset_index()
    )
    manifest.to_csv(diagnostics_dir / "manifest.csv", index=False)


def test_build_pretrained_diagnostics_figures_draws_all_six_from_one_cache_read(
    tmp_path,
):
    results_dir = tmp_path / "results"
    _write_synthetic_diagnostics_cache(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_pretrained_diagnostics_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["pretrained"]
    for name in (
        "pred_density",
        "pred_density_limited",
        "residuals_elev",
        "residuals_lat",
        "residuals_localtime",
        "residuals_year_month",
        "uncertainty",
    ):
        titled = target / f"{name}.png"
        assert titled.exists() and titled.stat().st_size > 0


def test_build_pretrained_diagnostics_figures_skips_without_raising_when_cache_absent(
    tmp_path, caplog
):
    output_dir = tmp_path / "plots"
    args = argparse.Namespace(
        results_dir=tmp_path / "empty_results", output_dir=output_dir
    )
    with caplog.at_level(logging.WARNING):
        mf._build_pretrained_diagnostics_figures(args, output_dir)

    assert not (output_dir / mf.SOURCE_DIRS["pretrained"]).exists()
    assert "pretrained_test_diagnostics" in caplog.text


def test_build_pretrained_diagnostics_figures_subsamples_pred_density_above_the_cap(
    tmp_path, monkeypatch
):
    """The five boxplot figures must see every row; only fig_pred_density's hexbin is
    bounded - pinned by checking the *other* figures' reported observation counts against
    the full cache while pred_density's own CSV is capped."""
    monkeypatch.setattr(mf, "_PRED_DENSITY_SAMPLE_CAP", 50)
    results_dir = tmp_path / "results"
    _write_synthetic_diagnostics_cache(results_dir, rows=200)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_pretrained_diagnostics_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["pretrained"]
    density = pd.read_csv(target / "pred_density.csv")
    assert len(density) == 50

    # residuals_elev has no sampling cap of its own, so its per-bin observation count
    # must still sum to the full 200-row cache, not the 50-row density subsample.
    elev = pd.read_csv(target / "residuals_elev.csv")
    assert elev["n"].sum() == 200


# --------------------------------------------------------------------------
# Figure 10 - daily % improvement over VTEC/GIM baselines
# --------------------------------------------------------------------------


def test_fig_improvement_by_date_matches_the_1_minus_ratio_formula(tmp_path):
    """`improvement = (1 - direct_stec / baseline) * 100`, ported from
    `src/multiday_evaluation.py`'s improvement-statistics section."""
    style.configure_plotting()
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01"] * 3 + ["2024-05-02"] * 3),
            "Model": ["Direct STEC", "VTEC + Mapping", "IGS GIM + Mapping"] * 2,
            "RMSE": [5.0, 10.0, 8.0, 6.0, 12.0, 6.0],
        }
    )
    mf.fig_improvement_by_date(daily, "RMSE", tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["finetuned"]
    plotted = pd.read_csv(target / "improvements_rmse.csv").set_index(
        ["date", "baseline"]
    )
    assert plotted.loc[
        ("2024-05-01", "VTEC + Mapping"), "improvement_pct"
    ] == pytest.approx(50.0)
    assert plotted.loc[
        ("2024-05-01", "IGS GIM + Mapping"), "improvement_pct"
    ] == pytest.approx(37.5)
    assert plotted.loc[
        ("2024-05-02", "IGS GIM + Mapping"), "improvement_pct"
    ] == pytest.approx(0.0)


def test_build_improvement_by_date_figures_end_to_end_from_synthetic_per_day_csv(
    tmp_path,
):
    """Exercises the Model-name normalisation (`daily_metrics`'s "Direct STEC Model"/
    "IGS GIM" spellings) and the dataset/metric loop through the registered builder."""
    results_dir = tmp_path / "results"
    daily_metrics_dir = mf.analysis_dir(results_dir, "daily_metrics")
    daily_metrics_dir.mkdir(parents=True)
    per_day = pd.DataFrame(
        {
            "date": ["2024-122"] * 3 + ["2024-123"] * 3,
            "year": [2024] * 6,
            "doy": [122] * 3 + [123] * 3,
            "dataset": ["own_vtec_gim"] * 6,
            "Model": ["Direct STEC Model", "VTEC + Mapping", "IGS GIM"] * 2,
            "RMSE": [5.0, 8.0, 7.0, 5.2, 8.1, 7.2],
            "MAE": [3.0, 4.0, 4.5, 3.1, 4.1, 4.4],
        }
    )
    per_day.to_csv(daily_metrics_dir / "per_day.csv", index=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_improvement_by_date_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["finetuned"]
    assert (target / "improvements_rmse.png").exists()
    assert (target / "improvements_mae.png").exists()


def test_build_improvement_by_date_figures_keeps_own_and_madrigal_distinct(tmp_path):
    """Regression for the filename collision: `_build_improvement_by_date_figures` loops
    `table.groupby("dataset")` over `own_vtec_gim` and `madrigal_vtec_gim`, and used to
    write both through the same `improvements_{metric}` name - the second write silently
    discarded the first, so only one dataset's Figure 10 ever reached disk. A test that
    only checked "does improvements_rmse.png exist" would still pass on the broken code
    (one of the two datasets always survives); this asserts both output files exist *and*
    that each carries its own dataset's numbers, which the collision could not satisfy.
    """
    results_dir = tmp_path / "results"
    daily_metrics_dir = mf.analysis_dir(results_dir, "daily_metrics")
    daily_metrics_dir.mkdir(parents=True)
    # Both models present for both datasets on the same day (doy=122 -> 2024-05-01, once
    # `_build_improvement_by_date_figures` rebuilds the real calendar date from year+doy)
    # so the improvement ratio is defined for each: own is Direct STEC 5.0 vs VTEC 10.0 ->
    # 50%; Madrigal is Direct STEC 6.0 vs VTEC 8.0 -> 25%. Deliberately different values
    # so the two CSVs cannot coincidentally agree.
    per_day = pd.DataFrame(
        {
            "date": ["2024-122"] * 4,
            "year": [2024] * 4,
            "doy": [122] * 4,
            "dataset": [
                "own_vtec_gim",
                "own_vtec_gim",
                "madrigal_vtec_gim",
                "madrigal_vtec_gim",
            ],
            "Model": [
                "Direct STEC Model",
                "VTEC + Mapping",
                "Direct STEC Model",
                "VTEC + Mapping",
            ],
            "RMSE": [5.0, 10.0, 6.0, 8.0],
            "MAE": [2.5, 5.0, 3.0, 4.0],
        }
    )
    per_day.to_csv(daily_metrics_dir / "per_day.csv", index=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_improvement_by_date_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["finetuned"]
    own_path = target / "improvements_rmse.csv"
    madrigal_path = target / "improvements_rmse_madrigal.csv"
    assert own_path.exists()
    assert madrigal_path.exists()
    assert (target / "improvements_rmse.png").exists()
    assert (target / "improvements_rmse_madrigal.png").exists()

    own = pd.read_csv(own_path).set_index("date")["improvement_pct"]
    madrigal = pd.read_csv(madrigal_path).set_index("date")["improvement_pct"]
    assert own.loc["2024-05-01"] == pytest.approx(50.0)
    assert madrigal.loc["2024-05-01"] == pytest.approx(25.0)


# --------------------------------------------------------------------------
# Figure 11 - RMSE/MAE vs. elevation, mean +/- across-day std
# --------------------------------------------------------------------------


def _synthetic_daily_by_elevation() -> pd.DataFrame:
    """Three days x two elevation bins x two methods, with Direct STEC's RMSE/MAE fixed
    across days (std = 0, exactly) and VTEC + Mapping's varying (std > 0) - so the
    across-day aggregation `fig_mae_rmse_finetuned` computes internally is checkable
    exactly rather than just "did it run"."""
    rows = []
    for doy, vtec_rmse in ((130, 4.0), (131, 6.0), (132, 8.0)):
        for elevation_bin in (20.0, 40.0):
            rows.append(
                {
                    "doy": doy,
                    "elevation_bin": elevation_bin,
                    "Method": "Direct STEC",
                    "n": 500,
                    "RMSE": 2.0,
                    "MAE": 1.5,
                }
            )
            rows.append(
                {
                    "doy": doy,
                    "elevation_bin": elevation_bin,
                    "Method": "VTEC + Mapping",
                    "n": 500,
                    "RMSE": vtec_rmse,
                    "MAE": vtec_rmse,
                }
            )
    return pd.DataFrame(rows)


def test_fig_mae_rmse_finetuned_computes_mean_and_std_across_days(tmp_path):
    style.configure_plotting()
    daily = _synthetic_daily_by_elevation()
    mf.fig_mae_rmse_finetuned(daily, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["finetuned"]
    assert (target / "mae_rmse_finetuned.png").exists()
    assert (target / "mae_rmse_finetuned_notitle.png").exists()
    plotted = pd.read_csv(target / "mae_rmse_finetuned.csv").set_index(
        ["elevation_bin", "Method"]
    )

    direct = plotted.loc[(20.0, "Direct STEC")]
    assert direct["RMSE_mean"] == pytest.approx(2.0)
    assert direct["RMSE_std"] == pytest.approx(0.0)
    assert direct["days"] == 3
    assert direct["observations"] == 1500

    vtec = plotted.loc[(20.0, "VTEC + Mapping")]
    assert vtec["RMSE_mean"] == pytest.approx(6.0)  # mean(4, 6, 8)
    assert vtec["RMSE_std"] == pytest.approx(2.0)  # sample std of (4, 6, 8)


def test_build_mae_rmse_finetuned_figure_is_scoped_to_the_own_dataset(tmp_path):
    """The manuscript figure is the own test set; a Madrigal slice in the same CSV must
    not silently overwrite it under the shared `mae_rmse_finetuned` filename."""
    results_dir = tmp_path / "results"
    elevation_metrics_dir = mf.analysis_dir(results_dir, "elevation_metrics_finetuned")
    elevation_metrics_dir.mkdir(parents=True)
    own = _synthetic_daily_by_elevation().assign(dataset="own")
    madrigal = _synthetic_daily_by_elevation().assign(
        dataset="madrigal", RMSE=lambda d: d["RMSE"] * 10
    )
    pd.concat([own, madrigal], ignore_index=True).to_csv(
        elevation_metrics_dir / "per_day_by_elevation.csv",
        index=False,
    )

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_mae_rmse_finetuned_figure(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["finetuned"]
    plotted = pd.read_csv(target / "mae_rmse_finetuned.csv").set_index(
        ["elevation_bin", "Method"]
    )
    # own's Direct STEC RMSE is 2.0 everywhere; madrigal's would be 20.0 - a mixed read
    # would show it in the mean.
    assert plotted.loc[(20.0, "Direct STEC"), "RMSE_mean"] == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Figures 12-15 - positioning
# --------------------------------------------------------------------------


def _synthetic_positioning_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for date in ("2024-05-01", "2024-05-02", "2024-05-03"):
        for station in ("AAAA", "BBBB", "CCCC", "DDDD"):
            for raw_method in (
                "STEC_iono",
                "VTEC_iono",
                "gim_iono",
                "Pretrained_STEC_iono",
            ):
                rows.append(
                    {
                        "date": date,
                        "method": raw_method,
                        "error_3d_rms": float(rng.uniform(0.3, 2.5)),
                    }
                )
    # One station-day worse than the paper's 10 m rule, which fig_positioning_* must drop.
    rows.append({"date": "2024-05-01", "method": "STEC_iono", "error_3d_rms": 15.0})
    return pd.DataFrame(rows)


def test_load_positioning_frame_maps_methods_and_drops_10m_outliers(tmp_path):
    path = tmp_path / "multiday_summary.csv"
    _synthetic_positioning_frame().to_csv(path, index=False)

    loaded = mf._load_positioning_frame(path)

    assert set(loaded["method"]) == {
        "Direct STEC",
        "VTEC + Mapping",
        "IGS GIM + Mapping",
        "Pretrained Direct STEC",
    }
    assert (loaded["error_3d_rms"] <= mf.OUTLIER_3D_RMS_M).all()


def test_build_positioning_figures_end_to_end_from_synthetic_multiday_summary(tmp_path):
    results_dir = tmp_path / "results"
    positioning_coverage_dir = mf.analysis_dir(results_dir, "positioning_coverage")
    positioning_coverage_dir.mkdir(parents=True)
    _synthetic_positioning_frame().to_csv(
        positioning_coverage_dir / "multiday_summary.csv",
        index=False,
    )

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_positioning_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["positioning"]
    for name in (
        "pos_trend",
        "pos_improvement_timeseries",
        "pos_distribution_boxplot",
        "pos_cdf_3d_rms",
    ):
        titled = target / f"{name}.png"
        notitle = target / f"{name}_notitle.png"
        assert titled.exists() and titled.stat().st_size > 0
        assert notitle.exists() and notitle.stat().st_size > 0


def test_positioning_figures_are_drawn_at_the_pinned_geometry_not_figsize_wide(
    tmp_path, monkeypatch
):
    """Pins that the four `fig_positioning_*` drawing calls actually pass
    `style.FIGSIZE_POSITIONING_TREND`/`FIGSIZE_POSITIONING_DISTRIBUTION` to
    `plt.subplots` - a constant with the right value but never referenced would pass
    `test_style.py`'s check while the figure still rendered at FIGSIZE_WIDE, which is
    exactly how this port drifted the first time."""
    seen_figsizes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        seen_figsizes.append(kwargs.get("figsize"))
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)

    style.configure_plotting()
    frame = _synthetic_positioning_frame()
    frame["method"] = frame["method"].map(mf._POSITIONING_METHOD_MAP)
    frame = frame.dropna(subset=["method"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[frame["error_3d_rms"] <= mf.OUTLIER_3D_RMS_M]

    mf.fig_positioning_trend(frame, tmp_path, "synthetic")
    mf.fig_positioning_improvement_timeseries(frame, tmp_path, "synthetic")
    mf.fig_positioning_distribution_boxplot(frame, tmp_path, "synthetic")
    mf.fig_positioning_cdf_3d_rms(frame, tmp_path, "synthetic")

    assert seen_figsizes == [
        style.FIGSIZE_POSITIONING_TREND,
        style.FIGSIZE_POSITIONING_TREND,
        style.FIGSIZE_POSITIONING_DISTRIBUTION,
        style.FIGSIZE_POSITIONING_DISTRIBUTION,
    ]


def test_fig_improvement_by_date_is_drawn_at_the_pinned_geometry(tmp_path, monkeypatch):
    """Same regression as the positioning check above, for Figure 10."""
    seen_figsizes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        seen_figsizes.append(kwargs.get("figsize"))
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)

    style.configure_plotting()
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01"] * 2),
            "Model": ["Direct STEC", "IGS GIM + Mapping"],
            "RMSE": [5.0, 8.0],
        }
    )
    mf.fig_improvement_by_date(daily, "RMSE", tmp_path, "synthetic")

    assert seen_figsizes == [style.FIGSIZE_DAILY_IMPROVEMENT]


# --------------------------------------------------------------------------
# Entry point resilience
# --------------------------------------------------------------------------


def test_figure_builders_skip_without_raising_when_inputs_are_absent(
    tmp_path, caplog, monkeypatch
):
    """A partially-populated `multiday_results/` (or a worktree with no split lists) must
    still let every other figure build, matching `revision_figures`'s own contract."""
    monkeypatch.setattr(paths, "SPLIT_LISTS", tmp_path / "no_such_split_lists")
    monkeypatch.setattr(
        paths,
        "IGS_STATION_COORDINATES",
        tmp_path / "no_such_split_lists" / "IGSNetwork.csv",
    )
    args = argparse.Namespace(
        results_dir=tmp_path / "empty_results", output_dir=tmp_path / "plots"
    )
    with caplog.at_level(logging.WARNING):
        for build in mf.FIGURE_BUILDERS:
            build(args, args.output_dir)
    assert not list((tmp_path / "plots").rglob("*.png"))
