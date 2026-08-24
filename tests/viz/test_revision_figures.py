"""Exercises the shared `_save` primitive and one full CSV-to-PNG figure build.

Kept deliberately small: the drawing functions themselves are the ported, reviewed
matplotlib calls from `src/viz/revision_figures.py`, so the value here is in the plumbing
that is new to the port - `_save`'s provenance footnote and title clearing, and the
`FIGURE_BUILDERS` table's resilience when an input CSV is missing.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from stec.viz import revision_figures as rf
from stec.viz import style


def test_provenance_footnote_present_only_on_titled_variant(tmp_path):
    """`_save` writes the footnote onto the titled PNG, empties it before the notitle PNG,
    and clears every title-artist location (center/left/right) rather than only the
    default one. Captured with a `savefig` spy, since both saves mutate the same Figure/
    Axes in place - the post-hoc object no longer reflects what either file shows."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("My Title", loc="left")

    seen_footnotes: list[list[str]] = []
    seen_left_titles: list[str] = []
    original_savefig = fig.savefig

    def spy_savefig(*args, **kwargs):
        seen_footnotes.append([t.get_text() for t in fig.texts])
        seen_left_titles.append(ax.get_title(loc="left"))
        return original_savefig(*args, **kwargs)

    fig.savefig = spy_savefig
    rf._save(
        fig, "demo", "positioning", tmp_path, "multiday_results/source.csv - 12 days"
    )

    titled_footnotes, notitle_footnotes = seen_footnotes
    assert any("Data: multiday_results/source.csv" in t for t in titled_footnotes)
    assert all(t == "" for t in notitle_footnotes)
    assert seen_left_titles == ["My Title", ""]

    target = tmp_path / rf.SOURCE_DIRS["positioning"]
    assert (target / "demo.png").exists()
    assert (target / "demo_notitle.png").exists()


def test_save_writes_the_plotted_data_as_csv(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

    rf._save(fig, "demo", "finetuned", tmp_path, "source.csv", data)

    written = pd.read_csv(tmp_path / rf.SOURCE_DIRS["finetuned"] / "demo.csv")
    pd.testing.assert_frame_equal(written, data)


def test_weighting_ablation_figure_builds_end_to_end_from_synthetic_csv(tmp_path):
    """One full CSV -> PNG path through a registered builder, using the smallest input
    shape (`weighting_ablation/paired.csv`) rather than the real store."""
    results_dir = tmp_path / "results"
    weighting_ablation_dir = rf.analysis_dir(results_dir, "weighting_ablation")
    weighting_ablation_dir.mkdir(parents=True)
    paired = pd.DataFrame(
        {
            "method": ["Direct STEC", "IGS GIM + Mapping"],
            "elev_mean": [1.20, 1.55],
            "iono_mean": [1.05, 1.40],
            "paired_station_days": [48, 48],
        }
    ).set_index("method")
    paired.to_csv(weighting_ablation_dir / "paired.csv")

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_weighting_ablation_figure(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["positioning"]
    titled = target / "weighting_ablation.png"
    notitle = target / "weighting_ablation_notitle.png"
    assert titled.exists() and titled.stat().st_size > 0
    assert notitle.exists() and notitle.stat().st_size > 0
    assert (target / "weighting_ablation.csv").exists()


def test_figure_builders_skip_without_raising_when_inputs_are_absent(tmp_path, caplog):
    """Every registered family must behave like the source's main(): warn and move on, not
    raise, when its CSV isn't there - a pipeline run over a partially-populated
    multiday_results/ must still produce every figure it can."""
    args = argparse.Namespace(
        results_dir=tmp_path / "empty_results", output_dir=tmp_path / "plots"
    )
    with caplog.at_level(logging.WARNING):
        for build in rf.FIGURE_BUILDERS:
            build(args, args.output_dir)
    assert not list((tmp_path / "plots").rglob("*.png"))


def test_save_requires_a_known_source_key(tmp_path):
    fig, _ax = plt.subplots()
    with pytest.raises(KeyError):
        rf._save(fig, "demo", "not_a_real_source", tmp_path, "prov")
    plt.close(fig)


def _write_dstec_evaluation_csvs(results_dir, *, with_gim: bool = True) -> None:
    """Synthetic `dstec_evaluation` output: five arcs, matching `pass_statistics.csv`'s
    real columns plus a `summary.csv` whose pooled RMSEs are consistent with them, so a
    test can check the win-rate figure and the RMSE figure against the same numbers.
    """
    dstec_dir = rf.analysis_dir(results_dir, "dstec_evaluation")
    dstec_dir.mkdir(parents=True)

    arcs = pd.DataFrame(
        {
            "model_dstec_rmse": [2.0, 3.0, 6.0, 1.0, 4.0],
            "gim_dstec_rmse": [5.0, 2.0, 7.0, 3.0, 6.0],
            "model_abs_rmse": [4.0, 5.0, 8.0, 2.0, 6.0],
            "gim_abs_rmse": [6.0, 4.0, 9.0, 5.0, 7.0],
            "n_masked": [20, 20, 20, 20, 20],
        }
    )
    if not with_gim:
        arcs["gim_dstec_rmse"] = float("nan")
        arcs["gim_abs_rmse"] = float("nan")
    arcs.to_csv(dstec_dir / "pass_statistics.csv", index=False)

    summary = {
        "n_days": 2,
        "n_arcs": len(arcs),
        "n_masked_obs": int(arcs["n_masked"].sum()),
        "model_dstec_rmse_pooled": 3.5,
        "model_abs_rmse_pooled": 5.5,
    }
    if with_gim:
        summary["gim_dstec_rmse_pooled"] = 4.8
        summary["gim_abs_rmse_pooled"] = 6.2
    pd.Series(summary).to_csv(dstec_dir / "summary.csv", header=["value"])


def test_dstec_evaluation_figures_build_end_to_end_from_synthetic_csvs(tmp_path):
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_dstec_evaluation_figures(args, output_dir)

    target = output_dir / rf.SOURCE_DIRS["finetuned"]
    for stem in ("dstec_absolute_comparison", "dstec_win_rate"):
        assert (target / f"{stem}.png").stat().st_size > 0
        assert (target / f"{stem}_notitle.png").stat().st_size > 0
        assert (target / f"{stem}.csv").exists()


def test_dstec_win_rate_matches_a_direct_per_arc_computation(tmp_path):
    """Pins the win-rate arithmetic against an independent recomputation over the same
    5 synthetic arcs (matching test_dstec_evaluation.py's style: check the module's
    output against a direct computation, not against a value it derived itself)."""
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir)
    arcs = pd.read_csv(
        rf.analysis_dir(results_dir, "dstec_evaluation") / "pass_statistics.csv"
    )

    expected_dstec_win_pct = (
        100 * (arcs["model_dstec_rmse"] < arcs["gim_dstec_rmse"]).mean()
    )
    expected_abs_win_pct = 100 * (arcs["model_abs_rmse"] < arcs["gim_abs_rmse"]).mean()

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    rf._build_dstec_evaluation_figures(args, output_dir)

    written = pd.read_csv(
        output_dir / rf.SOURCE_DIRS["finetuned"] / "dstec_win_rate.csv"
    ).set_index("metric")["win_rate_pct"]
    assert written["dSTEC"] == pytest.approx(expected_dstec_win_pct)
    assert written["Absolute STEC"] == pytest.approx(expected_abs_win_pct)


def test_dstec_evaluation_figures_skip_when_summary_has_no_gim_columns(
    tmp_path, caplog
):
    """dstec_evaluation.summarise only adds gim_* keys when at least one arc had a
    usable gim_stec (see its docstring) - without them there is nothing to compare
    against, so the builder must log and return rather than raise on a missing key."""
    results_dir = tmp_path / "results"
    _write_dstec_evaluation_csvs(results_dir, with_gim=False)

    output_dir = tmp_path / "plots"
    args = argparse.Namespace(results_dir=results_dir, output_dir=output_dir)
    style.configure_plotting()
    with caplog.at_level(logging.INFO):
        rf._build_dstec_evaluation_figures(args, output_dir)

    assert not list(output_dir.rglob("*.png"))
