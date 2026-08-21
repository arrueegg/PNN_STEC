"""Pins the approach colour palette and `save_plot`'s title-stripping behaviour.

The colours are pinned as literal hex strings, not derived from `positioning/scripts/
plot_results.py` at test time, because the whole point is that they must never drift even
if that script changes - a test that re-derived them would stop catching the drift it
exists to catch.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from stec.viz import style


def test_approach_colours_match_published_positioning_plots():
    """Pinned from positioning/scripts/plot_results.py: STEC_COLOR, VTEC_COLOR, GIM_COLOR,
    PRETRAINED_COLOR. These four hex values must never change."""
    assert style.APPROACH_COLORS == {
        "Direct STEC": "#1f77b4",
        "VTEC + Mapping": "#ff7f0e",
        "IGS GIM + Mapping": "#2ca02c",
        "Pretrained Direct STEC": "#9467bd",
    }


def test_no_non_approach_series_uses_an_approach_colour():
    """CONDITION_COLORS, ORACLE_COLOR, CODE_GIM_COLOR and DATASET_COLORS encode conditions,
    the oracle bound, and datasets - none of them may reuse an approach's hue, or that hue
    would stop meaning only that approach."""
    approach_hexes = set(style.APPROACH_COLORS.values())
    assert style.NON_APPROACH_COLORS.isdisjoint(approach_hexes)
    # Also check the individual named constants directly, in case NON_APPROACH_COLORS itself
    # were ever assembled incorrectly.
    assert style.ORACLE_COLOR not in approach_hexes
    assert style.CODE_GIM_COLOR not in approach_hexes
    assert set(style.CONDITION_COLORS.values()).isdisjoint(approach_hexes)
    assert set(style.DATASET_COLORS.values()).isdisjoint(approach_hexes)


def test_code_gim_color_is_distinct_from_gim_color():
    """CODE GIM is documented as a lighter shade of the IGS GIM hue, not the same colour -
    otherwise the two products would be indistinguishable on the ionex_rms_benchmark figures."""
    assert style.CODE_GIM_COLOR != style.GIM_COLOR


def test_save_plot_writes_titled_and_notitle_png(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("A Title")

    style.save_plot(fig, "demo.png", tmp_path)

    titled = tmp_path / "demo.png"
    notitle = tmp_path / "demo_notitle.png"
    assert titled.exists() and titled.stat().st_size > 0
    assert notitle.exists() and notitle.stat().st_size > 0


def test_save_plot_notitle_has_no_title_artist(tmp_path):
    """save_plot reuses the same Axes for both saves, clearing the title in place between
    them - so the only reliable check is what the title reads *at the moment each file is
    written*, captured here by spying on `savefig` rather than inspecting the (by-then
    closed) figure afterwards."""
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ax.set_title("A Title")

    titles_when_saved = []
    original_savefig = fig.savefig

    def spy_savefig(*args, **kwargs):
        titles_when_saved.append(ax.get_title())
        return original_savefig(*args, **kwargs)

    fig.savefig = spy_savefig
    style.save_plot(fig, "demo.png", tmp_path)

    assert titles_when_saved == ["A Title", ""]


def test_analysis_method_labels_are_palette_keys():
    """An approach's colour is looked up by its name, so a rename silently unbinds it.

    `stratified_comparison` shipped with "IGS GIM" and "Pretrained" where its predecessor
    and the palette both say "IGS GIM + Mapping" and "Pretrained Direct STEC". Gate F only
    caught it once text columns became comparable; this catches it at import time.

    `daily_metrics` is deliberately excluded: it uses the short spellings, but so does the
    analysis it replaces, so changing it would break a confirmed equivalence. That
    inconsistency is inherited, not introduced, and is recorded rather than silently fixed.
    """
    from stec.analysis import stratified_comparison as sc

    from stec.viz.style import APPROACH_COLORS

    unmatched = set(sc.METHODS.values()) - set(APPROACH_COLORS)
    assert not unmatched, f"labels with no colour: {sorted(unmatched)}"
    assert sc.BASELINE in APPROACH_COLORS
