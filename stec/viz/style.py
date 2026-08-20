"""Shared plotting style: matplotlib defaults, figure sizes, `save_plot`, and the fixed
approach colour palette.

Ported from `src/viz/base.py` (`PLOT_CONFIG`, the `FIGSIZE_*` constants, `configure_plotting`
and `save_plot`) and from `positioning/scripts/plot_results.py` (`get_style`, the source of
the approach colours). Only the pieces `revision_figures.py` needs are ported; `base.py` also
has `get_scientific_label`, `ensure_dir` and `create_temporal_metrics_summaries`, which serve
other, unported analyses and have no bearing on the revision figures.

Colour rule (from CLAUDE.md, non-negotiable): the four approach colours below must never
change, and an approach colour must only ever mean that approach. Conditions (quiet/storm,
weighting scheme), datasets and the oracle bound are drawn from colours outside that palette
- `NON_APPROACH_COLORS` below - so the two sets are checked disjoint at import time as well as
by `tests/viz/test_style.py`. Known and accepted limitation, carried over unchanged: orange
(#ff7f0e) and green (#2ca02c) are separated by only dE = 0.7 in OKLab under simulated
protanopia, so those two series are hard to distinguish for red-blind readers. Consistency
with the already-published figures was chosen over fixing it - do not "fix" it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot is imported anywhere
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Matplotlib defaults, ported unchanged from src/viz/base.py.
# --------------------------------------------------------------------------

PLOT_CONFIG = {
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

FIGSIZE_SQUARE = (12, 12)  # scatter, correlation, calibration
FIGSIZE_WIDE = (
    16,
    10,
)  # spatial maps, single panel timelines - what revision_figures uses
FIGSIZE_DOUBLE_WIDE = (24, 10)  # 2-panel side-by-side
FIGSIZE_QUAD = (20, 16)  # 4-panel 2x2 grids
FIGSIZE_HISTOGRAM = (16, 10)
FIGSIZE_HEATMAP = (16, 10)


def configure_plotting() -> None:
    """Apply the standardized plotting configuration. Idempotent - safe to call per module."""
    plt.rcParams.update(PLOT_CONFIG)


def save_plot(
    fig: matplotlib.figure.Figure, filename: str, output_dir: str | Path
) -> None:
    """Save `filename` with its title, then again as `<stem>_notitle<suffix>` without one.

    Ported from `src/viz/base.py:save_plot`. This is the generic primitive; the revision
    figures use their own `_save` in `revision_figures.py`; which additionally writes the
    plotted data as CSV and a provenance footnote (present on the titled copy, absent on the
    notitle copy) - that behaviour is specific to the revision set and does not belong here.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_path = output_dir / filename
    fig.savefig(full_path, dpi=300, bbox_inches="tight")

    for ax in fig.axes:
        ax.set_title("")
    if fig._suptitle is not None:
        fig.suptitle("")

    stem, suffix = Path(filename).stem, Path(filename).suffix
    fig.savefig(output_dir / f"{stem}_notitle{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Approach colours, ported unchanged from positioning/scripts/plot_results.py's
# STEC_COLOR / VTEC_COLOR / GIM_COLOR / PRETRAINED_COLOR and get_style().
# --------------------------------------------------------------------------

STEC_COLOR = "#1f77b4"  # Direct STEC (blue)
VTEC_COLOR = "#ff7f0e"  # VTEC + Mapping (orange)
GIM_COLOR = "#2ca02c"  # IGS GIM + Mapping (green)
PRETRAINED_COLOR = "#9467bd"  # Pretrained Direct STEC (purple)

APPROACH_COLORS: dict[str, str] = {
    "Direct STEC": STEC_COLOR,
    "VTEC + Mapping": VTEC_COLOR,
    "IGS GIM + Mapping": GIM_COLOR,
    "Pretrained Direct STEC": PRETRAINED_COLOR,
}

# Legend/axis order used throughout the revision figures.
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# --------------------------------------------------------------------------
# Non-approach encodings. These must stay outside APPROACH_COLORS.values() so an approach
# colour never gets reused to mean something else (a condition, a dataset, the oracle bound).
# --------------------------------------------------------------------------

# Geomagnetic regime (quiet/storm) and weighting scheme (elevation/predicted-uncertainty).
CONDITION_COLORS: dict[str, str] = {"baseline": "#7f7f7f", "contrast": "#d62728"}

# The observation-derived positioning bound (oracle_benchmark).
ORACLE_COLOR = "#3f3f3f"

# CODE's GIM is a second instance of the "GIM + Mapping" approach, so it keeps that hue and
# is separated by a lighter shade rather than a new colour, which would read as a fifth
# method. It is deliberately NOT equal to GIM_COLOR - see the ionex_rms_benchmark figures.
CODE_GIM_COLOR = "#7bc47f"

# Evaluation datasets (own test set vs Madrigal vs Madrigal with the station offset removed).
DATASET_COLORS: dict[str, str] = {
    "own": "#7f7f7f",
    "madrigal": "#d62728",
    "madrigal_corrected": "#8c564b",
}

NON_APPROACH_COLORS: frozenset[str] = frozenset(
    {*CONDITION_COLORS.values(), ORACLE_COLOR, CODE_GIM_COLOR, *DATASET_COLORS.values()}
)

if NON_APPROACH_COLORS & set(APPROACH_COLORS.values()):
    # Fail fast at import rather than only when a test happens to run: a shared colour here
    # means a condition/dataset/oracle series would be visually indistinguishable from an
    # approach, which is exactly the confusion the palette split exists to prevent.
    raise AssertionError(
        "a non-approach colour collides with an approach colour: "
        f"{NON_APPROACH_COLORS & set(APPROACH_COLORS.values())}"
    )
