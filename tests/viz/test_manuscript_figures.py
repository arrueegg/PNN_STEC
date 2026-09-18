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

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.legend as mlegend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pytest

from stec.analysis import positioning_distributions as pdist
from stec.config import paths
from stec.viz import manuscript_figures as mf
from stec.viz import positioning_distributions as pdist_viz
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


def test_save_strips_legend_for_notitle_when_flagged(tmp_path):
    """`strip_legend_for_notitle` is Figure 4's own flag (see `fig_pred_density`) - the
    figure object itself must lose its legend once the notitle copy is written."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.legend()

    mf._save(
        fig, "demo", "positioning", tmp_path, "prov", strip_legend_for_notitle=True
    )

    assert ax.get_legend() is None


def test_save_keeps_legend_for_notitle_by_default(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="line")
    ax.legend()

    mf._save(fig, "demo", "positioning", tmp_path, "prov")

    assert ax.get_legend() is not None


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


def test_fig_pred_density_colorbar_uses_the_axes_divider(tmp_path, monkeypatch):
    """The colorbar must be exactly as tall as the plot box, which `set_aspect("equal")`
    shrinks - `make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)` ties the
    colorbar's own axes to that shrunk box, unlike a plain `fig.colorbar(..., ax=ax)`."""
    calls = []
    original_locatable = mf.make_axes_locatable

    def spy_locatable(ax):
        divider = original_locatable(ax)
        original_append = divider.append_axes

        def spy_append(*args, **kwargs):
            calls.append((args, kwargs))
            return original_append(*args, **kwargs)

        divider.append_axes = spy_append
        return divider

    monkeypatch.setattr(mf, "make_axes_locatable", spy_locatable)
    style.configure_plotting()
    df = _synthetic_observation_frame()
    mf.fig_pred_density(df, tmp_path, "synthetic")

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("right",)
    assert kwargs == {"size": "5%", "pad": 0.1}


def test_fig_pred_density_notitle_has_no_legend_and_titled_keeps_it(tmp_path):
    """Owner review, 2026-09-17: the manuscript (`_notitle`) copy drops the legend
    (perfect-prediction line, Pearson r, R2) - the working copy keeps it."""
    style.configure_plotting()
    df = _synthetic_observation_frame()

    captured = {}
    original_save = mf._save

    def spy_save(fig, name, source, output_dir, provenance, data=None, **kwargs):
        captured["strip_legend_for_notitle"] = kwargs.get(
            "strip_legend_for_notitle", False
        )
        captured["provenance"] = provenance
        return original_save(fig, name, source, output_dir, provenance, data, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mf, "_save", spy_save)
        mf.fig_pred_density(df, tmp_path, "synthetic")

    assert captured["strip_legend_for_notitle"] is True
    # The r/R2 values must still be recoverable even though the legend is gone.
    assert "Pearson r" in captured["provenance"]
    assert "R2" in captured["provenance"]


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


def test_fig_uncertainty_rmse_per_bin_computed_correctly(tmp_path):
    """A single bin with a non-constant error: RMSE must differ from MAE and match
    sqrt(mean(error**2)) exactly, not merely be present in the output."""
    style.configure_plotting()
    true_stec = np.zeros(5)
    stec_pred = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    df = pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": stec_pred,
            "pred_total_unc": np.full(5, 2.5),
        }
    )

    mf.fig_uncertainty(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    plotted = pd.read_csv(target / "uncertainty.csv")
    assert len(plotted) == 1
    expected_rmse = np.sqrt(np.mean(np.square(stec_pred - true_stec)))
    expected_mae = np.mean(np.abs(stec_pred - true_stec))
    assert expected_rmse != pytest.approx(
        expected_mae
    )  # the test is only meaningful if they differ
    assert plotted["rmse"].iloc[0] == pytest.approx(expected_rmse)
    assert plotted["mean_abs_error"].iloc[0] == pytest.approx(expected_mae)


def test_fig_uncertainty_sigma_columns_computed_correctly(tmp_path):
    """A single bin with non-constant sigma: `n`, `rms_sigma`,
    `rmse_over_mean_sigma` and `rmse_over_rms_sigma` must match hand-computed values
    from exactly the same rows `mean_total_unc`/`rmse` are computed from - mean and RMS
    of sigma must differ, or the test cannot tell them apart."""
    style.configure_plotting()
    true_stec = np.zeros(5)
    stec_pred = np.array([1.0, 1.0, 1.0, 1.0, 5.0])
    total_unc = np.array([1.2, 1.4, 1.6, 1.8, 1.9])  # all in the same (1, 2] bin
    df = pd.DataFrame(
        {
            "true_stec": true_stec,
            "stec_pred": stec_pred,
            "pred_total_unc": total_unc,
        }
    )

    mf.fig_uncertainty(df, tmp_path, "synthetic")

    target = tmp_path / mf.SOURCE_DIRS["pretrained"]
    plotted = pd.read_csv(target / "uncertainty.csv")
    assert len(plotted) == 1

    expected_rmse = np.sqrt(np.mean(np.square(stec_pred - true_stec)))
    expected_mean_sigma = total_unc.mean()
    expected_rms_sigma = np.sqrt(np.mean(np.square(total_unc)))
    assert expected_rms_sigma != pytest.approx(
        expected_mean_sigma
    )  # the test is only meaningful if they differ

    assert plotted["n"].iloc[0] == 5
    assert plotted["rms_sigma"].iloc[0] == pytest.approx(expected_rms_sigma)
    assert plotted["rmse_over_mean_sigma"].iloc[0] == pytest.approx(
        expected_rmse / expected_mean_sigma
    )
    assert plotted["rmse_over_rms_sigma"].iloc[0] == pytest.approx(
        expected_rmse / expected_rms_sigma
    )


def test_fig_uncertainty_rmse_line_present_with_distinct_style(tmp_path, monkeypatch):
    """The RMSE curve is a NON_APPROACH_COLORS colour, distinct from this figure's other
    four curves and from every APPROACH_COLORS hue, with its own marker and "RMSE" label."""
    captured_axes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3400, offset=2.0)
    mf.fig_uncertainty(df, tmp_path, "synthetic")

    ax = captured_axes[0]
    curve_lines = {
        line.get_label(): line
        for line in ax.get_lines()
        if line.get_label() and not line.get_label().startswith("_")
    }
    assert "RMSE" in curve_lines
    rmse_line = curve_lines["RMSE"]
    assert mcolors.to_hex(rmse_line.get_color()) == mf._UNCERTAINTY_RMSE_COLOR
    assert mf._UNCERTAINTY_RMSE_COLOR in style.NON_APPROACH_COLORS
    assert mf._UNCERTAINTY_RMSE_COLOR not in style.APPROACH_COLORS.values()

    other_colors = {
        mcolors.to_hex(line.get_color())
        for label, line in curve_lines.items()
        if label != "RMSE"
    }
    assert mf._UNCERTAINTY_RMSE_COLOR not in other_colors
    other_markers = {
        line.get_marker() for label, line in curve_lines.items() if label != "RMSE"
    }
    assert rmse_line.get_marker() not in other_markers


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


def test_fig_uncertainty_xticks_are_integers_at_bin_edges(tmp_path, monkeypatch):
    """Owner review, 2026-09-17: a label at every 1 TECU bin centre (1.5, 2.5, ...)
    overlapped - ticks now land on whole-TECU bin edges, spaced apart, with no ".0"."""
    captured_axes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)
    style.configure_plotting()
    df = _synthetic_observation_frame(rows=3400, offset=2.0)
    mf.fig_uncertainty(df, tmp_path, "synthetic")

    ax = captured_axes[0]
    xticks = ax.get_xticks()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert all(float(tick).is_integer() for tick in xticks)
    assert all("." not in label for label in labels)
    diffs = np.diff(sorted(xticks))
    assert all(d == pytest.approx(mf._UNCERTAINTY_XTICK_STEP) for d in diffs)
    assert ax.xaxis.get_tick_params()["labelsize"] == mf._UNCERTAINTY_XTICK_LABELSIZE


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


def test_build_pretrained_diagnostics_figures_uses_every_row_for_pred_density(
    tmp_path,
):
    """2026-09-17 (owner review): the old 2,000,000-point, seed-42 hexbin subsample is
    gone - `fig_pred_density` must see every row of the cache, exactly like the five
    boxplot figures, so Pearson r and R2 are exact rather than approximate."""
    assert not hasattr(mf, "_PRED_DENSITY_SAMPLE_CAP")
    results_dir = tmp_path / "results"
    _write_synthetic_diagnostics_cache(results_dir, rows=200)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_pretrained_diagnostics_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["pretrained"]
    density = pd.read_csv(target / "pred_density.csv")
    assert len(density) == 200

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


def test_fig_improvement_by_date_uses_the_same_date_axis_as_figure_12(
    tmp_path, monkeypatch
):
    """Owner review, 2026-09-17: Figure 10's date axis must match Figure 12's
    (`fig_positioning_trend`) locator/formatter/rotation - month-day used to be shown,
    which reads as an inconsistency between the two daily-trend figures."""
    captured_axes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)
    style.configure_plotting()
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
            "Model": ["Direct STEC", "VTEC + Mapping"],
            "RMSE": [5.0, 10.0],
        }
    )
    mf.fig_improvement_by_date(daily, "RMSE", tmp_path, "synthetic")

    ax = captured_axes[0]
    assert isinstance(ax.xaxis.get_major_locator(), mdates.MonthLocator)
    formatter = ax.xaxis.get_major_formatter()
    assert isinstance(formatter, mdates.DateFormatter)
    assert formatter.fmt == "%Y-%m"


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


def _synthetic_daily_by_elevation_with_pretrained() -> pd.DataFrame:
    """Direct STEC fixed at 2.0 RMSE / 1.5 MAE; Pretrained Direct STEC fixed at 20.0
    RMSE / 14.0 MAE - deliberately the highest mean line, matching the real data's shape
    (roughly 20/14 TECU at the lowest elevation bin) - so the y-limit cap and the
    Pretrained-only error-bar fade are both checkable exactly."""
    rows = []
    for doy in (130, 131, 132):
        for elevation_bin in (5.0, 10.0):
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
                    "Method": "Pretrained Direct STEC",
                    "n": 500,
                    "RMSE": 20.0,
                    "MAE": 14.0,
                }
            )
    return pd.DataFrame(rows)


def test_fig_mae_rmse_finetuned_caps_ylim_and_fades_pretrained_errorbars(
    tmp_path, monkeypatch
):
    """Owner review, 2026-09-17 (second pass, 5%/nearest-2): the y-axis is capped a
    little above the highest mean line (data-driven, not hardcoded - see
    `_elevation_ylim_cap`) so Pretrained Direct STEC's large error bars are clipped at
    the top, and those bars (caps + whiskers, not the mean line/markers) are drawn
    semi-transparent so they stop covering the other curves."""
    captured_axes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        captured_axes.append(axes)
        return fig, axes

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)
    style.configure_plotting()
    daily = _synthetic_daily_by_elevation_with_pretrained()
    mf.fig_mae_rmse_finetuned(daily, tmp_path, "synthetic")

    ax_rmse, ax_mae = captured_axes[0]
    # peak RMSE mean 20.0 -> +5% margin = 21.0 -> rounded up to the nearest 2 = 22.0
    assert ax_rmse.get_ylim()[1] == pytest.approx(22.0)
    # peak MAE mean 14.0 -> +5% margin = 14.7 -> rounded up to the nearest 2 = 16.0
    assert ax_mae.get_ylim()[1] == pytest.approx(16.0)

    containers = {c.get_label(): c for c in ax_rmse.containers}
    _, direct_caps, direct_bars = containers["Direct STEC"].lines
    _, pretrained_caps, pretrained_bars = containers["Pretrained Direct STEC"].lines
    assert all(cap.get_alpha() == pytest.approx(0.9) for cap in direct_caps)
    assert all(bar.get_alpha() == pytest.approx(0.9) for bar in direct_bars)
    assert all(
        cap.get_alpha() == pytest.approx(mf._PRETRAINED_ERRORBAR_ALPHA)
        for cap in pretrained_caps
    )
    assert all(
        bar.get_alpha() == pytest.approx(mf._PRETRAINED_ERRORBAR_ALPHA)
        for bar in pretrained_bars
    )


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
    # A station-day worse than the paper's old 10 m rule - kept, not dropped, since
    # 2026-08-28 (docs/revision/positioning_reporting.md): no positioning figure applies
    # an outcome-based filter any more.
    rows.append({"date": "2024-05-01", "method": "STEC_iono", "error_3d_rms": 15.0})
    return pd.DataFrame(rows)


def _write_synthetic_positioning_distributions(output_dir) -> None:
    """Builds `positioning_distributions`'s own CSVs from the same synthetic
    station-day population `_synthetic_positioning_frame` produces, using that module's
    real `boxplot_stats`/`cdf_points`/`percentile_summary` - not hand-built numbers - so
    Figures 13/15's reused generators have something real to read.

    Also writes the `common_set_*` twins Figures 12-15 read since 2026-09-15
    (results_register.md consistency item A) - this fixture has no real coverage
    restriction to model, so every synthetic row is treated as if it were already the
    common set, the same population as the `overall_*`/`multiday_summary.csv` files
    above rather than a genuinely smaller one."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frame = _synthetic_positioning_frame()
    raw_frame[["date", "method", "error_3d_rms"]].to_csv(
        output_dir / "common_set_daily_rows.csv", index=False
    )

    frame = raw_frame.copy()
    frame["Method"] = frame["method"].map(mf._POSITIONING_METHOD_MAP)
    frame = frame.dropna(subset=["Method"])

    stats, fliers = pdist.boxplot_stats(frame, ["Method"])
    stats.to_csv(output_dir / "overall_boxplot_stats.csv", index=False)
    fliers.to_csv(output_dir / "overall_boxplot_fliers.csv", index=False)
    pdist.cdf_points(frame, ["Method"]).to_csv(
        output_dir / "overall_cdf_points.csv", index=False
    )
    pdist.percentile_summary(frame, ["Method"]).to_csv(
        output_dir / "overall_percentile_summary.csv", index=False
    )
    # Figure 13's view-capped y-axis (owner review, 2026-09-14) needs an exceedance
    # table to annotate how many station-days per method fall outside the cap.
    pdist.exceedance_table(frame, ["Method"]).to_csv(
        output_dir / "overall_exceedance.csv", index=False
    )
    # common_set_* twins - same rows, see the docstring above.
    stats.to_csv(output_dir / "common_set_boxplot_stats.csv", index=False)
    fliers.to_csv(output_dir / "common_set_boxplot_fliers.csv", index=False)
    pdist.cdf_points(frame, ["Method"]).to_csv(
        output_dir / "common_set_cdf_points.csv", index=False
    )
    pdist.percentile_summary(frame, ["Method"]).to_csv(
        output_dir / "common_set_percentile_summary.csv", index=False
    )
    pdist.exceedance_table(frame, ["Method"]).to_csv(
        output_dir / "common_set_exceedance.csv", index=False
    )


def test_load_positioning_frame_maps_methods_and_applies_no_filter(tmp_path):
    path = tmp_path / "multiday_summary.csv"
    _synthetic_positioning_frame().to_csv(path, index=False)

    loaded = mf._load_positioning_frame(path)

    assert set(loaded["method"]) == {
        "Direct STEC",
        "VTEC + Mapping",
        "IGS GIM + Mapping",
        "Pretrained Direct STEC",
    }
    # The old >10 m station-day exclusion is gone (docs/revision/positioning_reporting.md,
    # owner decision 2026-08-28) - the synthetic 15.0 m row above must survive.
    assert (loaded["error_3d_rms"] == 15.0).any()


def test_build_positioning_figures_end_to_end_from_synthetic_multiday_summary(tmp_path):
    results_dir = tmp_path / "results"
    positioning_coverage_dir = mf.analysis_dir(results_dir, "positioning_coverage")
    positioning_coverage_dir.mkdir(parents=True)
    _synthetic_positioning_frame().to_csv(
        positioning_coverage_dir / "multiday_summary.csv",
        index=False,
    )
    _write_synthetic_positioning_distributions(
        mf.analysis_dir(results_dir, "positioning_distributions")
    )

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_positioning_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["positioning"]
    for name in (
        "pos_trend",
        "pos_improvement_timeseries",
        "boxplot_3d_error",
        "cdf_unfiltered",
    ):
        titled = target / f"{name}.png"
        notitle = target / f"{name}_notitle.png"
        assert titled.exists() and titled.stat().st_size > 0
        assert notitle.exists() and notitle.stat().st_size > 0


def test_positioning_improvement_ylim_is_data_driven_upper_only():
    """Owner review, 2026-09-18 (third pass): the lower limit is no longer derived from
    the data at all - `_positioning_improvement_ylim` now returns only the upper bound,
    and the caller pairs it with the fixed `_POSITIONING_IMPROVEMENT_YLIM_LOWER` (-150).
    A prior data-driven lower bound let a single series' -290%-ish excursions dictate the
    whole axis height, squashing the readable upper half where every method's actual
    improvement lives - the fixed clip below trades "nothing is ever clipped" for "a
    below-clip value simply runs off the bottom of the visible axis" (see
    `fig_positioning_improvement_timeseries`). The upper-bound math itself (5% margin,
    round up to the nearest 50) is unchanged from
    the previous pass, so it still gives 50 on data topping out at +45.6."""
    values = pd.Series([-258.275654, -221.253584, 45.578029, 9.084547, -76.126536])

    upper = mf._positioning_improvement_ylim(values)

    assert upper > values.max()
    assert upper % 50 == 0
    assert upper == 50.0


def _draw_positioning_improvement_and_capture(monkeypatch, frame, tmp_path):
    """Runs `fig_positioning_improvement_timeseries` and hands back its axes before
    `_save` closes the figure, by intercepting `_save` itself rather than `plt.subplots`
    - the axes object is needed after the function returns to inspect markers/legend."""
    captured = {}
    original_save = mf._save

    def spy_save(fig, name, source, output_dir, provenance, data=None, **kwargs):
        captured["fig"] = fig
        captured["ax"] = fig.axes[0]
        captured["provenance"] = provenance
        return original_save(fig, name, source, output_dir, provenance, data, **kwargs)

    monkeypatch.setattr(mf, "_save", spy_save)
    mf.fig_positioning_improvement_timeseries(frame, tmp_path, "synthetic")
    return captured


def test_positioning_improvement_timeseries_no_triangle_markers_below_clip(
    tmp_path, monkeypatch
):
    """Owner review, 2026-09-18 (fifth round): below-clip values used to get a
    downward-pointing triangle flag; those were removed as too occluded to help, so a
    below-clip day now draws nothing but the ordinary line/marker running off the axis.
    This frame drives two methods past the -150% clip on different days - the same data
    that used to produce two triangle markers - and asserts none are drawn."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-05-01",
                    "2024-05-02",
                    "2024-05-03",
                    "2024-05-01",
                    "2024-05-02",
                    "2024-05-03",
                    "2024-05-01",
                    "2024-05-02",
                    "2024-05-03",
                ]
            ),
            "method": ["IGS GIM + Mapping"] * 3
            + ["Direct STEC"] * 3
            + ["Pretrained Direct STEC"] * 3,
            "error_3d_rms": [
                1.0,
                1.0,
                1.0,  # IGS GIM + Mapping (baseline)
                0.8,
                0.9,
                4.0,  # Direct STEC: 2024-05-03 -> -300%, below clip
                3.5,
                0.6,
                0.7,  # Pretrained Direct STEC: 2024-05-01 -> -250%, below clip
            ],
        }
    )
    style.configure_plotting()

    captured = _draw_positioning_improvement_and_capture(monkeypatch, frame, tmp_path)
    ax = captured["ax"]

    marker_shapes = {line.get_marker() for line in ax.get_lines()}
    assert "v" not in marker_shapes


def test_positioning_improvement_timeseries_ylim_and_ticks_fixed_lower(
    tmp_path, monkeypatch
):
    """The axis bottom is pinned at -150 regardless of how far the data dips, the top
    stays data-driven, and major ticks land every 50 with plain numeric labels (no "%" -
    the axis label already carries the unit)."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-02"] * 2),
            "method": ["IGS GIM + Mapping"] * 2 + ["Pretrained Direct STEC"] * 2,
            "error_3d_rms": [1.0, 1.0, 6.0, 0.5],
        }
    )
    style.configure_plotting()

    captured = _draw_positioning_improvement_and_capture(monkeypatch, frame, tmp_path)
    ax = captured["ax"]

    assert ax.get_ylim()[0] == mf._POSITIONING_IMPROVEMENT_YLIM_LOWER
    assert ax.get_ylim()[0] == -150.0
    locator = ax.yaxis.get_major_locator()
    assert isinstance(locator, mticker.MultipleLocator)
    ticks = ax.get_yticks()
    assert all(tick % 50 == 0 for tick in ticks)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert all("%" not in label for label in labels)


def test_positioning_improvement_timeseries_footnote_mentions_clip_only_in_titled(
    tmp_path, monkeypatch
):
    """The working copy's provenance footnote must say values below -150% are clipped
    (the removed triangle flag is called out by name - "no triangle markers" - so a
    reader of an old copy of this figure knows the marker is gone by design, not
    missing by accident); `_save` already strips all footnote text from the `_notitle`
    copy for every figure, so that half of the requirement is inherited, not re-tested
    here - this test pins the sentence actually reaching `_save` for the titled copy."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-02"] * 2),
            "method": ["IGS GIM + Mapping"] * 2 + ["Direct STEC"] * 2,
            "error_3d_rms": [1.0, 1.0, 0.8, 0.9],
        }
    )
    style.configure_plotting()

    captured = _draw_positioning_improvement_and_capture(monkeypatch, frame, tmp_path)

    assert "synthetic" in captured["provenance"]
    assert "-150" in captured["provenance"]
    assert "clip" in captured["provenance"].lower()

    # `_save` unconditionally clears all footnote text (`footnote.set_text("")`) before
    # writing the `_notitle` copy, regardless of which figure calls it - that behaviour
    # is generic to `_save` and not re-verified per figure here, so this test only needs
    # to confirm the sentence actually reaches `_save` for the titled copy above, and
    # that both files get written.
    target = tmp_path / mf.SOURCE_DIRS["positioning"]
    assert (target / "pos_improvement_timeseries.png").exists()
    assert (target / "pos_improvement_timeseries_notitle.png").exists()


def test_positioning_improvement_timeseries_legend_placement_and_size(
    tmp_path, monkeypatch
):
    """Owner review, 2026-09-18 (fourth round): the legend moved out of the axes to a
    single row below them (`loc="upper center"`, negative `bbox_to_anchor` y, `ncol`
    one per series) - an in-plot legend covered real points in every corner tried (see
    the function's docstring) - at the normal (rcParams-default) legend font size, not
    the shrunk-for-in-plot-placement size that only made sense while the legend shared
    the axes with the data."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-05-01", "2024-05-02"] * 2),
            "method": ["IGS GIM + Mapping"] * 2 + ["Direct STEC"] * 2,
            "error_3d_rms": [1.0, 1.0, 0.8, 0.9],
        }
    )
    style.configure_plotting()

    captured = _draw_positioning_improvement_and_capture(monkeypatch, frame, tmp_path)
    ax = captured["ax"]

    legend = ax.get_legend()
    assert legend is not None
    assert legend._get_loc() == mlegend.Legend.codes["upper center"]
    # `_bbox_to_anchor._bbox` holds the raw (x, y) passed to `bbox_to_anchor`, in axes
    # fraction coordinates, before the transform chain that resolves it to a pixel
    # position at draw time - this is what to compare against, not a resolved position.
    assert legend._bbox_to_anchor._bbox.x0 == pytest.approx(0.5)
    assert legend._bbox_to_anchor._bbox.y0 == pytest.approx(
        mf._POSITIONING_IMPROVEMENT_LEGEND_BBOX_Y
    )
    assert legend._ncols == 1  # single series ("Direct STEC") in this synthetic frame
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(
        plt.rcParams["legend.fontsize"]
    )


def test_fig_positioning_improvement_timeseries_matches_figure_12_style(
    tmp_path, monkeypatch
):
    """Owner review, 2026-09-17: markers reuse Figure 12's own size/linewidth constants,
    the y-label is exactly "Improvement over GIM [%]", and the date axis matches
    Figure 12. Legend placement itself (below the axes, as of the 2026-09-18 fourth
    round) is pinned by
    `test_positioning_improvement_timeseries_legend_placement_and_size` instead of
    here."""
    captured_axes = []
    original_subplots = mf.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(mf.plt, "subplots", spy_subplots)
    style.configure_plotting()
    frame = _synthetic_positioning_frame()
    frame["method"] = frame["method"].map(mf._POSITIONING_METHOD_MAP)
    frame = frame.dropna(subset=["method"])
    frame["date"] = pd.to_datetime(frame["date"])

    mf.fig_positioning_improvement_timeseries(frame, tmp_path, "synthetic")

    ax = captured_axes[0]
    assert ax.get_ylabel() == "Improvement over GIM [%]"
    assert isinstance(ax.xaxis.get_major_locator(), mdates.MonthLocator)
    assert ax.xaxis.get_major_formatter().fmt == "%Y-%m"
    method_lines = [
        line for line in ax.get_lines() if line.get_label() in mf.APPROACH_COLORS
    ]
    assert method_lines
    assert not any(line.get_label().startswith("Imp. by") for line in ax.get_lines())
    for line in method_lines:
        assert line.get_marker() == "o"
        assert line.get_markersize() == pytest.approx(mf._POSITIONING_TREND_MARKERSIZE)
        assert line.get_linewidth() == pytest.approx(mf._POSITIONING_TREND_LINEWIDTH)
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_frame_on()


def test_positioning_figures_are_drawn_at_the_pinned_geometry_not_figsize_wide(
    tmp_path, monkeypatch
):
    """Pins that Figures 12/14's own drawing calls pass `style.FIGSIZE_POSITIONING_TREND`
    to `plt.subplots`, and that the reused Figures 13/15
    (`stec.viz.positioning_distributions.fig_boxplot_3d_error`/`fig_cdf_unfiltered`) are
    drawn at that module's own `FIGSIZE_WIDE` - a constant with the right value but never
    referenced would pass `test_style.py`'s check while a figure still rendered at the
    wrong geometry, which is exactly how this port drifted the first time."""
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

    mf.fig_positioning_trend(frame, tmp_path, "synthetic")
    mf.fig_positioning_improvement_timeseries(frame, tmp_path, "synthetic")

    method_frame = frame.rename(columns={"method": "Method"})
    stats, fliers = pdist.boxplot_stats(method_frame, ["Method"])
    exceedance = pdist.exceedance_table(method_frame, ["Method"])
    mf.fig_boxplot_3d_error(stats, fliers, tmp_path, "synthetic", exceedance)
    cdf = pdist.cdf_points(method_frame, ["Method"])
    percentiles = pdist.percentile_summary(method_frame, ["Method"])
    mf.fig_cdf_unfiltered(cdf, percentiles, tmp_path, "synthetic")

    assert seen_figsizes == [
        style.FIGSIZE_POSITIONING_TREND,
        style.FIGSIZE_POSITIONING_TREND,
        style.FIGSIZE_WIDE,
        style.FIGSIZE_WIDE,
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
# Figure 13, second version - approaches + the observation-derived oracle floor
# --------------------------------------------------------------------------


def test_fig_boxplot_3d_error_with_oracle_places_oracle_first_in_neutral_color(
    tmp_path, monkeypatch
):
    """Five boxes, oracle leftmost in `style.ORACLE_COLOR` (not an approach hue), the
    rest in their usual approach colours and order."""
    captured_axes = []
    original_subplots = pdist_viz.plt.subplots

    def spy_subplots(*args, **kwargs):
        fig, ax = original_subplots(*args, **kwargs)
        captured_axes.append(ax)
        return fig, ax

    monkeypatch.setattr(pdist_viz.plt, "subplots", spy_subplots)
    style.configure_plotting()

    rng = np.random.default_rng(0)
    methods = [
        pdist_viz.ORACLE_LABEL,
        "Direct STEC",
        "Pretrained Direct STEC",
        "VTEC + Mapping",
        "IGS GIM + Mapping",
    ]
    frame = pd.DataFrame(
        {
            "Method": np.repeat(methods, 30),
            "error_3d_rms": rng.uniform(0.1, 5.0, 30 * len(methods)),
        }
    )

    pdist_viz.fig_boxplot_3d_error_with_oracle(frame, tmp_path, "synthetic", 30)

    target = tmp_path / pdist_viz.SOURCE_DIRS["positioning"]
    assert (target / "boxplot_3d_error_with_oracle.png").exists()
    assert (target / "boxplot_3d_error_with_oracle_notitle.png").exists()
    plotted = pd.read_csv(target / "boxplot_3d_error_with_oracle.csv")
    assert set(plotted["Method"]) == set(methods)

    ax = captured_axes[0]
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # Owner review, 2026-09-17: the oracle tick label is two lines ("Reference STEC" /
    # "(oracle)") so it stops nearly touching "Pretrained Direct STEC" - Figure 13
    # itself never carries this method, so its own single-line labels are unaffected.
    expected_labels = [
        "Reference STEC\n(oracle)" if m == pdist_viz.ORACLE_LABEL else m
        for m in methods
    ]
    assert labels == expected_labels
    boxes = ax.patches
    assert mcolors.to_rgb(boxes[0].get_facecolor()[:3]) == mcolors.to_rgb(
        style.ORACLE_COLOR
    )
    assert mcolors.to_rgb(boxes[1].get_facecolor()[:3]) == mcolors.to_rgb(
        style.APPROACH_COLORS["Direct STEC"]
    )


def _write_synthetic_oracle_benchmark_and_coverage(results_dir) -> None:
    """`oracle_benchmark`'s paired population (4 methods, elevation weighting) plus
    `positioning_coverage`'s all-weightings table the builder pulls the
    `Pretrained_STEC_elev` arm from. BBBB/123 is deliberately absent from the coverage
    table - the Pretrained arm need not cover every oracle_benchmark station-day, and the
    builder must drop that row rather than crash or plot a partial box."""
    oracle_dir = mf.analysis_dir(results_dir, "oracle_benchmark")
    oracle_dir.mkdir(parents=True)
    paired = pd.DataFrame(
        {
            "station": ["AAAA", "AAAA", "BBBB", "BBBB"],
            "doy": [122, 123, 122, 123],
            "Reference STEC (oracle)": [0.10, 0.20, 0.15, 0.25],
            "Direct STEC": [1.0, 1.1, 1.2, 1.3],
            "VTEC + Mapping": [1.5, 1.6, 1.7, 1.8],
            "IGS GIM + Mapping": [1.3, 1.4, 1.5, 1.6],
        }
    )
    paired.to_csv(oracle_dir / "paired_station_days.csv", index=False)

    coverage_dir = mf.analysis_dir(results_dir, "positioning_coverage")
    coverage_dir.mkdir(parents=True)
    all_weightings = pd.DataFrame(
        {
            "station": ["AAAA", "AAAA", "BBBB", "CCCC"],
            "method": ["Pretrained_STEC_elev"] * 4,
            "doy": [122, 123, 122, 999],
            "error_3d_rms": [2.0, 2.1, 2.2, 9.9],
        }
    )
    all_weightings.to_csv(
        coverage_dir / "multiday_summary_all_weightings.csv", index=False
    )


def test_build_positioning_oracle_figure_drops_station_days_missing_the_pretrained_arm(
    tmp_path, caplog
):
    results_dir = tmp_path / "results"
    _write_synthetic_oracle_benchmark_and_coverage(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.WARNING):
        mf._build_positioning_oracle_figure(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["positioning"]
    assert (target / "boxplot_3d_error_with_oracle.png").exists()
    plotted = pd.read_csv(target / "boxplot_3d_error_with_oracle.csv")
    # BBBB/123 has no Pretrained_STEC_elev row, so 3 of the 4 paired station-days survive.
    n_rows = plotted[plotted["stat"] == "n"]
    assert (n_rows["value"] == 3).all()
    assert "1 of 4" in caplog.text


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
