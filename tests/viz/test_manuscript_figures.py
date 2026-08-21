"""Exercises the manuscript figure builders (Figures 1, 2, 10, 12-15) against synthetic
frames, mirroring `tests/viz/test_revision_figures.py`: a `_save`-plumbing check, then one
end-to-end CSV/list -> PNG path per figure family.

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
    (results_dir / "daily_metrics_rebuilt").mkdir(parents=True)
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
    per_day.to_csv(results_dir / "daily_metrics_rebuilt" / "per_day.csv", index=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    mf._build_improvement_by_date_figures(args, output_dir)

    target = output_dir / mf.SOURCE_DIRS["finetuned"]
    assert (target / "improvements_rmse.png").exists()
    assert (target / "improvements_mae.png").exists()


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
    (results_dir / "positioning_coverage_rebuilt").mkdir(parents=True)
    _synthetic_positioning_frame().to_csv(
        results_dir / "positioning_coverage_rebuilt" / "multiday_summary.csv",
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
