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
    (results_dir / "weighting_ablation").mkdir(parents=True)
    paired = pd.DataFrame(
        {
            "method": ["Direct STEC", "IGS GIM + Mapping"],
            "elev_mean": [1.20, 1.55],
            "iono_mean": [1.05, 1.40],
            "paired_station_days": [48, 48],
        }
    ).set_index("method")
    paired.to_csv(results_dir / "weighting_ablation" / "paired.csv")

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
