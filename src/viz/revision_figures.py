"""Publication figures for the JGR-MLC revision — one panel per file.

Each figure answers a specific reviewer comment and is built from analyses that
need no re-inference. Figures are written one plot per PNG, grouped into
subfolders by the data that produced them:

    plots/revision/
      stec_pretrained_testset/   pretrained model, held-out test set 2014-2024
      stec_finetuned_2024/       daily fine-tuned models, 242 test days of 2024
      positioning_2024/          SF-PPP solutions, 2024 test period
      training_runs/             W&B training history

Reviewer mapping
----------------
  R1.2  relative_error_absolute, relative_error_normalised
  R1.5  architecture_search
  R2.4  activity_dst_*, activity_f107_*
  R2.5  weighting_ablation
  R2.7  storm_positioning_*
  R2.8  oracle_benchmark

Style
-----
Uses the repository's standard `PLOT_CONFIG` from `viz.base`, the same
configuration behind Figures 4-9, so these sit beside the published figures
without a visual break. Series colours are taken unchanged from
`positioning/scripts/plot_results.py`: blue = Direct STEC, orange = VTEC +
Mapping, green = IGS GIM + Mapping, purple = Pretrained Direct STEC.

The plot area carries no explanatory text — no value labels, correlation
figures, reference-line captions or interpretive notes. All of that belongs in
the manuscript caption and body. Each figure is written twice: a working copy
with a title and a provenance footnote naming the source file and scope, and a
`_notitle` copy with neither, which is the one for the manuscript.

Known limitation, carried over deliberately: orange (#ff7f0e) and green
(#2ca02c) are separated by only dE = 0.7 in OKLab under simulated protanopia, so
those two series are hard to distinguish for red-blind readers. Consistency with
the published figures was chosen over changing the palette.

Usage::

    python src/viz/revision_figures.py --output_dir plots/revision
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from viz.base import FIGSIZE_WIDE, configure_plotting  # noqa: E402

logger = logging.getLogger(__name__)

# Unchanged from positioning/scripts/plot_results.py.
COLORS = {
    "Direct STEC": "#1f77b4",
    "VTEC + Mapping": "#ff7f0e",
    "IGS GIM + Mapping": "#2ca02c",
    "Pretrained Direct STEC": "#9467bd",
}
METHOD_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# Non-approach encodings (geomagnetic regime, weighting scheme, the oracle
# bound). Kept outside the approach palette so an approach colour only ever
# means that approach.
CONDITION_COLORS = {"baseline": "#7f7f7f", "contrast": "#d62728"}
ORACLE_COLOR = "#3f3f3f"
# Evaluation sets, again outside the approach palette.
# CODE's GIM is a second instance of the "GIM + Mapping" approach, so it keeps
# that approach's hue and is separated by a lighter shade rather than a new
# colour, which would read as a fourth method.
CODE_GIM_COLOR = "#7bc47f"

DATASET_COLORS = {
    "own": "#7f7f7f",
    "madrigal": "#d62728",
    "madrigal_corrected": "#8c564b",
}

SOURCE_DIRS = {
    "stec_finetuned": "stec_finetuned_2024",
    "pretrained": "stec_pretrained_testset",
    "finetuned": "stec_finetuned_2024",
    "positioning": "positioning_2024",
    "training": "training_runs",
}


def _save(
    fig,
    name: str,
    source: str,
    output_dir: Path,
    provenance: str,
    data: pd.DataFrame | None = None,
) -> None:
    """Write the working copy (title + provenance), the manuscript copy, and the numbers.

    The CSV holds what the figure actually draws, not the analysis table it came
    from - those often carry extra models, bins or columns that never reach the
    axes. Writing the plotted values means the number a reader checks is the
    number they see, and the two cannot drift apart.
    """
    target = output_dir / SOURCE_DIRS[source]
    target.mkdir(parents=True, exist_ok=True)
    if data is not None:
        data.to_csv(target / f"{name}.csv", index=False)

    # Negative y places the note below the x-axis label; bbox_inches="tight"
    # expands the saved area to include artists outside the figure rectangle.
    footnote = fig.text(
        0.0, -0.04, f"Data: {provenance}", fontsize=11, color="#555555", va="top"
    )
    fig.savefig(target / f"{name}.png", bbox_inches="tight")

    footnote.set_text("")
    for ax in fig.axes:
        # An axes keeps a separate title artist per location, so clearing only
        # the default (centre) one leaves a loc="left" title in place.
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    fig.savefig(target / f"{name}_notitle.png", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"💾 {target / name}.png (+ _notitle)")


def _grouped_bars(ax, groups, series, values, colors, ylabel, xlabel=""):
    """Grouped bars in the paper's style: no value labels, y-grid only.

    Returns the plotted values tidied to one row per bar, so the caller can hand
    them straight to `_save`.
    """
    n_series = len(series)
    x = np.arange(len(groups))
    width = 0.8 / n_series
    for i, name in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        ax.bar(
            x + offset,
            values[name],
            width * 0.94,
            color=colors[i],
            label=name,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    return pd.DataFrame(
        [
            {"group": str(g).replace("\n", " "), "series": name, "value": values[name][i]}
            for i, g in enumerate(groups)
            for name in series
        ]
    )


def _method_labels(names) -> list[str]:
    return [
        n.replace(" Direct STEC", "\nDirect STEC").replace(" + ", "\n+ ") for n in names
    ]


# --------------------------------------------------------------------------
# R1.2 — absolute vs TEC-normalised error across the solar cycle
# --------------------------------------------------------------------------


def fig_relative_error_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(
        d.year,
        d.mean_STEC,
        marker="o",
        markersize=9,
        color=CONDITION_COLORS["baseline"],
        label="Mean observed STEC",
    )
    ax.plot(
        d.year,
        d.RMSE,
        marker="s",
        markersize=9,
        color=COLORS["Direct STEC"],
        label="Model RMSE",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("STEC [TECU]")
    ax.set_xticks(d.year)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Absolute error and mean observed STEC by year")
    _save(fig, "relative_error_absolute", "pretrained", output_dir, provenance,
          d[["year", "mean_STEC", "RMSE"]])


def fig_relative_error_normalised(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(d.year, d["nRMSE_%"], marker="s", markersize=9, color=COLORS["Direct STEC"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalised RMSE [%]")
    ax.set_xticks(d.year)
    ax.set_ylim(0, max(45, d["nRMSE_%"].max() * 1.15))
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("TEC-normalised error by year")
    _save(fig, "relative_error_normalised", "pretrained", output_dir, provenance,
          d[["year", "nRMSE_%"]])


# --------------------------------------------------------------------------
# R2.7 — positioning under quiet vs storm conditions
# --------------------------------------------------------------------------


def fig_storm_positioning_absolute(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        _method_labels(d.index),
        ["quiet", "storm"],
        {"quiet": d["quiet"].values, "storm": d["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.legend(title="Geomagnetic conditions")
    ax.set_title("Positioning error by geomagnetic regime")
    _save(fig, "storm_positioning_absolute", "positioning", output_dir, provenance, plotted)


def fig_storm_positioning_improvement(
    d: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    gim = "IGS GIM + Mapping"
    improvement = pd.DataFrame(
        {
            reg: 100 * (d.loc[gim, reg] - d[reg]) / d.loc[gim, reg]
            for reg in ("quiet", "storm")
        }
    ).drop(index=gim)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        _method_labels(improvement.index),
        ["quiet", "storm"],
        {"quiet": improvement["quiet"].values, "storm": improvement["storm"].values},
        [CONDITION_COLORS["baseline"], CONDITION_COLORS["contrast"]],
        "Improvement over IGS GIM + Mapping [%]",
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=4)
    ax.legend(title="Geomagnetic conditions", loc="lower left")
    ax.set_title("Margin over the operational baseline by geomagnetic regime")
    _save(fig, "storm_positioning_improvement", "positioning", output_dir, provenance, plotted)


# --------------------------------------------------------------------------
# R2.4 — STEC error against geomagnetic activity and solar flux
# --------------------------------------------------------------------------


def _activity_figures(
    table: pd.DataFrame,
    bin_col: str,
    axis_label: str,
    stem: str,
    output_dir: Path,
    provenance: str,
) -> None:
    rename = {
        "Direct STEC Model": "Direct STEC",
        "Pretrained STEC": "Pretrained Direct STEC",
        "VTEC + Mapping": "VTEC + Mapping",
        "IGS GIM": "IGS GIM + Mapping",
    }
    d = table.copy()
    d["Model"] = d["Model"].map(rename)
    bins = list(dict.fromkeys(d[bin_col]))
    series = [m for m in METHOD_ORDER if m in set(d["Model"])]

    values = {
        m: [d[(d.Model == m) & (d[bin_col] == b)]["RMSE"].iloc[0] for b in bins]
        for m in series
    }
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        series,
        values,
        [COLORS[m] for m in series],
        "STEC RMSE [TECU]",
        axis_label,
    )
    ax.legend(ncol=2)
    ax.set_title(f"STEC error by {axis_label.lower()}")
    _save(fig, f"{stem}_absolute", "finetuned", output_dir, provenance, plotted)

    rel_series = [m for m in series if m != "IGS GIM + Mapping"]
    rel_values = {
        m: [
            d[(d.Model == m) & (d[bin_col] == b)]["improvement_over_gim_%"].iloc[0]
            for b in bins
        ]
        for m in rel_series
    }
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        rel_series,
        rel_values,
        [COLORS[m] for m in rel_series],
        "Improvement over IGS GIM [%]",
        axis_label,
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=4)
    ax.legend(loc="lower left")
    ax.set_title(f"Margin over IGS GIM by {axis_label.lower()}")
    _save(fig, f"{stem}_improvement", "finetuned", output_dir, provenance, plotted)


# --------------------------------------------------------------------------
# R2.5, R2.8, R1.5
# --------------------------------------------------------------------------


def fig_weighting_ablation(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        [m.replace(" + ", "\n+ ") for m in d.index],
        ["Elevation weighting", "Fixed variance", "Predicted-uncertainty weighting"],
        {
            "Elevation weighting": d["elev_mean"].values,
            # Only the Direct STEC correction has a fixed-variance run; the bar is
            # simply absent for the others rather than faked.
            "Fixed variance": d.get(
                "fixed_mean", pd.Series(np.nan, index=d.index)
            ).values,
            "Predicted-uncertainty weighting": d["iono_mean"].values,
        },
        [CONDITION_COLORS["baseline"], "#8c6d31", CONDITION_COLORS["contrast"]],
        "3D RMS positioning error [m]",
    )
    ax.legend(loc="upper left")
    ax.set_title("Observation weighting scheme")
    _save(fig, "weighting_ablation", "positioning", output_dir, provenance, plotted)


def fig_oracle_benchmark(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    order = [m for m in ("Reference STEC (oracle)", *METHOD_ORDER) if m in d.index]
    d = d.loc[order]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    colors = [
        ORACLE_COLOR if m.startswith("Reference STEC") else COLORS[m] for m in d.index
    ]
    ax.bar(np.arange(len(d)), d["mean"], 0.62, color=colors, zorder=3)
    ax.axhline(
        d["mean"].iloc[0], color=ORACLE_COLOR, linewidth=1.5, linestyle="--", zorder=4
    )
    ax.set_xticks(np.arange(len(d)))
    ax.set_xticklabels(
        [m.replace(" (oracle)", "\n(oracle)").replace(" + ", "\n+ ") for m in d.index]
    )
    ax.set_ylabel("3D RMS positioning error [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Positioning against the observation-derived bound")
    _save(
        fig, "oracle_benchmark", "positioning", output_dir, provenance,
        d.reset_index()[
            [c for c in (d.index.name or "index", "mean", "median", "p95",
                         "station_days", "above_oracle_m", "ratio_to_oracle")
             if c in d.reset_index().columns]
        ],
    )


def fig_architecture_search(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    d = d.sort_values("best_val_MAE")
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    y = np.arange(len(d))
    colors = [
        COLORS["Direct STEC"] if c > 0 else CONDITION_COLORS["baseline"]
        for c in d["credible_runs"]
    ]
    ax.barh(y, d["best_val_MAE"], 0.62, color=colors, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.invert_yaxis()
    ax.set_xlabel("Best validation MAE [TECU]")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Architecture comparison")
    _save(
        fig, "architecture_search", "training", output_dir, provenance,
        d.reset_index()[
            [c for c in (d.index.name or "index", "best_val_MAE", "runs", "credible_runs")
             if c in d.reset_index().columns]
        ],
    )


def fig_madrigal_reference_offset(
    offsets: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """R2.3 - two unrelated estimates disagree with Madrigal the same way.

    Each point is a station. Agreement along the 1:1 line means the discrepancy
    is a property of the Madrigal reference, since the model and the GIM share
    nothing in how they are produced.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(
        offsets["offset_gim"], offsets["offset_model"], s=90,
        color=CONDITION_COLORS["contrast"], edgecolors="white", linewidths=0.8, zorder=3,
    )
    lo = float(min(offsets["offset_gim"].min(), offsets["offset_model"].min())) - 2
    hi = float(max(offsets["offset_gim"].max(), offsets["offset_model"].max())) + 2
    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.5, zorder=2)
    ax.axhline(0, color=CONDITION_COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.axvline(0, color=CONDITION_COLORS["baseline"], linewidth=1.0, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("IGS GIM − Madrigal, per station [TECU]")
    ax.set_ylabel("Direct STEC − Madrigal, per station [TECU]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("Per-station disagreement with Madrigal")
    _save(fig, "madrigal_reference_offset", "stec_finetuned", output_dir, provenance,
          offsets.reset_index()[["station", "observations", "offset_gim", "offset_model"]])


def fig_calibration_coverage(
    own: pd.DataFrame, madrigal: pd.DataFrame | None,
    corrected: pd.DataFrame | None, output_dir: Path, provenance: str,
) -> None:
    """R2.6 - reliability: nominal against empirical interval coverage."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.5, zorder=2)
    ax.plot(own["nominal"], own["empirical"], marker="o", markersize=10,
            color=DATASET_COLORS["own"], label="Own test set", zorder=3)
    if madrigal is not None:
        ax.plot(madrigal["nominal"], madrigal["empirical"], marker="s", markersize=10,
                color=DATASET_COLORS["madrigal"], label="Madrigal", zorder=3)
    if corrected is not None:
        ax.plot(corrected["nominal"], corrected["empirical_offset_removed"],
                marker="^", markersize=10, color=DATASET_COLORS["madrigal_corrected"],
                label="Madrigal, station offset removed", zorder=3)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("Interval coverage reliability")
    series = [own[["nominal", "empirical"]].assign(series="own test set")]
    if madrigal is not None:
        series.append(madrigal[["nominal", "empirical"]].assign(series="Madrigal"))
    if corrected is not None:
        series.append(
            corrected[["nominal", "empirical_offset_removed"]]
            .rename(columns={"empirical_offset_removed": "empirical"})
            .assign(series="Madrigal, station offset removed")
        )
    _save(
        fig, "calibration_coverage", "stec_finetuned", output_dir, provenance,
        pd.concat(series, ignore_index=True),
    )


def fig_calibration_pit(
    own: pd.DataFrame, madrigal: pd.DataFrame | None, output_dir: Path, provenance: str
) -> None:
    """R2.6 - PIT histogram; uniform under calibration."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    centres = 0.5 * (own["bin_left"] + own["bin_right"])
    width = float(own["bin_right"].iloc[0] - own["bin_left"].iloc[0])
    ax.bar(centres, own["density"], width * 0.94, color=DATASET_COLORS["own"],
           label="Own test set", zorder=3)
    if madrigal is not None:
        ax.step(centres, madrigal["density"], where="mid", linewidth=2.5,
                color=DATASET_COLORS["madrigal"], label="Madrigal", zorder=4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("PIT histogram")
    _save(
        fig, "calibration_pit", "stec_finetuned", output_dir, provenance,
        pd.DataFrame({
            "pit_centre": centres,
            "density_own": own["density"].to_numpy(),
            **({"density_madrigal": madrigal["density"].to_numpy()} if madrigal is not None else {}),
        }),
    )


def fig_station_independence(
    per_station: pd.DataFrame, binned: pd.DataFrame, output_dir: Path, provenance: str
) -> None:
    """R1.3 - does error grow with distance from the nearest training station?"""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.scatter(
        per_station["distance_km"], per_station["nRMSE_%"], s=70,
        color=CONDITION_COLORS["baseline"], alpha=0.65,
        edgecolors="white", linewidths=0.6, zorder=3,
    )
    ax.plot(
        binned["median_distance_km"], binned["nRMSE_pct"], marker="o", markersize=11,
        linewidth=2.5, color=CONDITION_COLORS["contrast"], zorder=4,
        label="Distance-bin mean",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Distance to nearest training station [km]")
    ax.set_ylabel("Normalised RMSE [%]")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title("Test-station error against separation from the training network")
    _save(
        fig, "station_independence", "stec_finetuned", output_dir, provenance,
        pd.concat([
            per_station.reset_index()[["station", "distance_km", "nRMSE_%"]].assign(series="station"),
            binned.rename(columns={"median_distance_km": "distance_km", "nRMSE_pct": "nRMSE_%"})
                  [["distance_km", "nRMSE_%"]].assign(series="distance-bin mean"),
        ], ignore_index=True),
    )


def fig_positioning_tail(tails: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """R2.7 - tail behaviour, not just the mean."""
    quantiles = ["median", "p90", "p95", "p99"]
    order = [m for m in METHOD_ORDER if m in tails.index]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    x = np.arange(len(quantiles))
    for i, method in enumerate(order):
        offset = (i - (len(order) - 1) / 2) * (0.8 / len(order))
        ax.bar(x + offset, tails.loc[method, quantiles].values, 0.8 / len(order) * 0.94,
               color=COLORS[method], label=method, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["Median", "90th", "95th", "99th"])
    ax.set_xlabel("Percentile of the daily 3D RMS across station-days")
    ax.set_ylabel("3D RMS positioning error [m]")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=2)
    ax.set_title("Tail of the positioning error distribution")
    _save(
        fig, "positioning_tail", "positioning", output_dir, provenance,
        tails.loc[order, quantiles].reset_index().melt(
            id_vars=tails.index.name or "index", var_name="percentile", value_name="error_3d_rms_m"
        ),
    )



# --------------------------------------------------------------------------
# IONEX RMS benchmark - our uncertainty against the GIM products' own
# --------------------------------------------------------------------------

# Keyed by the product labels ionex_rms_benchmark emits. "VTEC + Mapping" was
# added to that benchmark later; omitting it here raised KeyError and took every
# figure after it down with the run.
_IONEX_COLORS = {
    "Direct STEC": COLORS["Direct STEC"],
    "VTEC + Mapping": COLORS["VTEC + Mapping"],
    "IGS GIM + Mapping": COLORS["IGS GIM + Mapping"],
    "CODE GIM + Mapping": CODE_GIM_COLOR,
}


def fig_ionex_coverage(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Empirical against nominal coverage; the diagonal is perfect calibration."""
    levels = [50, 68, 90, 95]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    span = [levels[0] - 5, 100]
    ax.plot(
        span,
        span,
        linestyle="--",
        color=CONDITION_COLORS["baseline"],
        linewidth=1.5,
        label="Perfect calibration",
        zorder=2,
    )
    ax.set_xlim(*span)
    for product in [p for p in d.index if p in _IONEX_COLORS]:
        ax.plot(
            levels,
            [100 * d.loc[product, f"cov_{lv}"] for lv in levels],
            marker="o",
            markersize=9,
            color=_IONEX_COLORS[product],
            label=product,
            zorder=3,
        )
    ax.set_xlabel("Nominal coverage [%]")
    ax.set_ylabel("Empirical coverage [%]")
    ax.set_xticks(levels)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Interval coverage of each product's own uncertainty")
    _save(
        fig, "ionex_rms_coverage", "finetuned", output_dir, provenance,
        pd.DataFrame([
            {"product": p, "nominal_%": lv, "empirical_%": 100 * d.loc[p, f"cov_{lv}"],
             "days": d.loc[p, "days"]}
            for p in d.index if p in _IONEX_COLORS for lv in levels
        ]),
    )


def fig_ionex_crps_skill(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """CRPS skill against each product's own constant-sigma reference.

    Positive means the per-observation uncertainty beats a single constant for
    that same set of predictions; negative means it is worse than no
    uncertainty at all.
    """
    bins = ["5-20", "20-40", "40-60", "60-90"]
    products = [p for p in _IONEX_COLORS if p in d.index.get_level_values(0)]
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    plotted = _grouped_bars(
        ax,
        bins,
        products,
        {p: [100 * d.loc[(p, b), "CRPS_skill"] for b in bins] for p in products},
        [_IONEX_COLORS[p] for p in products],
        "CRPS skill over constant σ [%]",
        xlabel="Satellite elevation [°]",
    )
    ax.axhline(0, color="black", linewidth=1.0, zorder=4)
    ax.legend(loc="upper right")
    ax.set_title("Value of the per-observation uncertainty, by elevation")
    _save(fig, "ionex_rms_crps_skill", "finetuned", output_dir, provenance, plotted)



# --------------------------------------------------------------------------
# R1.6 - predicted uncertainty against realised error
# --------------------------------------------------------------------------


def fig_uncertainty_vs_error(d: pd.DataFrame, output_dir: Path, provenance: str) -> None:
    """Mean predicted sigma against realised RMSE, per predicted-sigma decile."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    limit = float(max(d.RMSE.max(), d.mean_sigma.max())) * 1.05
    ax.plot(
        [0, limit],
        [0, limit],
        linestyle="--",
        color=CONDITION_COLORS["baseline"],
        linewidth=1.5,
        label="Perfect calibration",
        zorder=2,
    )
    ax.plot(
        d.mean_sigma,
        d.RMSE,
        marker="o",
        markersize=9,
        color=COLORS["Direct STEC"],
        label="Direct STEC, by predicted-σ decile",
        zorder=3,
    )
    ax.set_xlabel("Mean predicted σ [TECU]")
    ax.set_ylabel("Realised RMSE [TECU]")
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title("Predicted uncertainty against realised error")
    _save(fig, "uncertainty_vs_error", "finetuned", output_dir, provenance,
          d[["bin", "n", "mean_sigma", "RMSE", "rmse_over_sigma"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("plots/revision"))
    parser.add_argument(
        "--relative_metrics",
        type=Path,
        default=Path("multiday_results/relative_error_metrics.csv"),
    )
    parser.add_argument(
        "--storm_degradation",
        type=Path,
        default=Path("multiday_results/storm_stratification/degradation.csv"),
    )
    parser.add_argument(
        "--weighting_paired",
        type=Path,
        default=Path("multiday_results/weighting_ablation/paired.csv"),
    )
    parser.add_argument(
        "--architectures",
        type=Path,
        default=Path("multiday_results/hyperparameter_search/architectures.csv"),
    )
    parser.add_argument(
        "--oracle_summary",
        type=Path,
        default=Path("multiday_results/oracle_benchmark/summary.csv"),
    )
    parser.add_argument(
        "--activity_dir",
        type=Path,
        default=Path("multiday_results/activity_stratification"),
    )
    parser.add_argument(
        "--calibration_dir",
        type=Path,
        default=Path("multiday_results/uncertainty_calibration"),
    )
    parser.add_argument(
        "--madrigal_offset_dir",
        type=Path,
        default=Path("multiday_results/madrigal_reference_offset"),
    )
    parser.add_argument(
        "--station_independence_dir",
        type=Path,
        default=Path("multiday_results/station_independence"),
    )
    parser.add_argument(
        "--uncertainty_error_dir",
        type=Path,
        default=Path("multiday_results/uncertainty_error_relation"),
    )
    parser.add_argument(
        "--ionex_benchmark_dir",
        type=Path,
        default=Path("multiday_results/ionex_rms_benchmark"),
    )
    parser.add_argument(
        "--positioning_robustness_dir",
        type=Path,
        default=Path("multiday_results/positioning_robustness"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    configure_plotting()

    if args.relative_metrics.exists():
        d = pd.read_csv(args.relative_metrics).sort_values("year")
        prov = (
            f"{args.relative_metrics} — pretrained model, held-out test set, "
            f"{d.year.min()}-{d.year.max()} ({int(d['count'].sum()):,} observations)"
        )
        fig_relative_error_absolute(d, args.output_dir, prov)
        fig_relative_error_normalised(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.relative_metrics} not found")

    if args.storm_degradation.exists():
        d = pd.read_csv(args.storm_degradation, index_col=0).reindex(METHOD_ORDER)
        prov = (
            f"{args.storm_degradation} — SF-PPP, 2024 test period, 39 storm days "
            "(daily min Dst ≤ −50 nT) of 242, station-days ≤ 10 m"
        )
        fig_storm_positioning_absolute(d, args.output_dir, prov)
        fig_storm_positioning_improvement(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.storm_degradation} not found")

    if args.weighting_paired.exists():
        d = pd.read_csv(args.weighting_paired, index_col=0)
        prov = (
            f"{args.weighting_paired} — SF-PPP, 2024 test period, paired station-days "
            f"solved under both weightings ({int(d['paired_station_days'].sum()):,} pairs)"
        )
        fig_weighting_ablation(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.weighting_paired} not found")

    if args.oracle_summary.exists():
        d = pd.read_csv(args.oracle_summary, index_col=0)
        prov = (
            f"{args.oracle_summary} — SF-PPP, elevation weighting throughout, "
            f"{int(d['station_days'].max())} station-days solved by every method"
        )
        fig_oracle_benchmark(d, args.output_dir, prov)
    else:
        logger.warning(
            f"⚠️  {args.oracle_summary} not found - run src/analysis/oracle_benchmark.py"
        )

    if args.architectures.exists():
        d = pd.read_csv(args.architectures, index_col=0)
        prov = (
            f"{args.architectures} — local W&B history, "
            f"{int(d['runs'].sum())} STEC runs reporting a validation MAE"
        )
        fig_architecture_search(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {args.architectures} not found")

    for stem, filename, bin_col, axis_label in (
        ("activity_dst", "by_dst.csv", "dst_bin", "Daily minimum Dst"),
        ("activity_f107", "by_f107.csv", "f107_bin", "Daily mean F10.7"),
    ):
        path = args.activity_dir / filename
        if not path.exists():
            logger.warning(
                f"⚠️  {path} not found - run src/analysis/activity_stratification.py"
            )
            continue
        table = pd.read_csv(path)
        days = int(table[table.Model == "Direct STEC Model"]["days"].sum())
        obs = int(table[table.Model == "Direct STEC Model"]["observations"].sum())
        prov = (
            f"{path} — daily fine-tuned models, own test set, "
            f"{days} test days of 2024 ({obs:,} observations)"
        )
        _activity_figures(table, bin_col, axis_label, stem, args.output_dir, prov)

    by_sigma = args.uncertainty_error_dir / "by_sigma.csv"
    if by_sigma.exists():
        d = pd.read_csv(by_sigma)
        prov = (
            f"{by_sigma} — daily fine-tuned models, own test set "
            f"({int(d['n'].sum()):,} observations)"
        )
        fig_uncertainty_vs_error(d, args.output_dir, prov)
    else:
        logger.warning(
            f"⚠️  {by_sigma} not found - run src/analysis/uncertainty_error_relation.py"
        )

    overall = args.ionex_benchmark_dir / "overall_IGS.csv"
    if overall.exists():
        igs = pd.read_csv(overall, index_col=0)
        code = args.ionex_benchmark_dir / "overall_CODE.csv"
        if code.exists():
            extra = pd.read_csv(code, index_col=0)
            igs = pd.concat([igs, extra.loc[[i for i in extra.index if i not in igs.index]]])
        prov = (
            f"{args.ionex_benchmark_dir}/overall_*.csv — daily fine-tuned models, own "
            f"test set, {int(igs['days'].max())} test days of 2024 "
            f"({int(igs['observations'].max()):,} observations)"
        )
        fig_ionex_coverage(igs, args.output_dir, prov)

        by_elev = [
            pd.read_csv(args.ionex_benchmark_dir / f"by_elevation_{g}.csv",
                        index_col=[0, 1])
            for g in ("IGS", "CODE")
            if (args.ionex_benchmark_dir / f"by_elevation_{g}.csv").exists()
        ]
        if by_elev:
            merged = pd.concat(by_elev)
            merged = merged[~merged.index.duplicated()]
            fig_ionex_crps_skill(merged, args.output_dir, prov)
    else:
        logger.warning(
            f"⚠️  {overall} not found - run src/analysis/ionex_rms_benchmark.py"
        )

    # R2.3 - per-station offsets against Madrigal
    offsets_path = args.madrigal_offset_dir / "per_station_offsets.csv"
    coverage_path = args.madrigal_offset_dir / "coverage_before_after.csv"
    if offsets_path.exists():
        offsets = pd.read_csv(offsets_path, index_col=0)
        prov = (
            f"{offsets_path} — daily fine-tuned models on Madrigal geometries, "
            f"{len(offsets)} stations, {int(offsets['observations'].sum()):,} observations"
        )
        fig_madrigal_reference_offset(offsets, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {offsets_path} not found - run madrigal_reference_offset.py")

    # R2.6 - calibration
    own_cov = args.calibration_dir / "finetuned_stec_own" / "coverage_all.csv"
    mad_cov = args.calibration_dir / "finetuned_stec_madrigal" / "coverage_all.csv"
    own_pit = args.calibration_dir / "finetuned_stec_own" / "pit_all.csv"
    mad_pit = args.calibration_dir / "finetuned_stec_madrigal" / "pit_all.csv"
    if own_cov.exists():
        prov = f"{args.calibration_dir} — daily fine-tuned models, prediction store"
        fig_calibration_coverage(
            pd.read_csv(own_cov),
            pd.read_csv(mad_cov) if mad_cov.exists() else None,
            pd.read_csv(coverage_path) if coverage_path.exists() else None,
            args.output_dir,
            prov,
        )
        if own_pit.exists():
            fig_calibration_pit(
                pd.read_csv(own_pit),
                pd.read_csv(mad_pit) if mad_pit.exists() else None,
                args.output_dir,
                prov,
            )
    else:
        logger.warning(f"⚠️  {own_cov} not found - run uncertainty_calibration.py")

    # R1.3 - station independence
    per_station = args.station_independence_dir / "per_station.csv"
    binned = args.station_independence_dir / "by_distance_bin.csv"
    if per_station.exists() and binned.exists():
        d = pd.read_csv(per_station)
        prov = (
            f"{per_station} — daily fine-tuned models, own test set, "
            f"{len(d)} test stations"
        )
        fig_station_independence(d, pd.read_csv(binned), args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {per_station} not found - run station_independence.py")

    # R2.7 - tail of the positioning error distribution
    tails = args.positioning_robustness_dir / "tail_distribution.csv"
    if tails.exists():
        d = pd.read_csv(tails, index_col=0)
        prov = (
            f"{tails} — SF-PPP, 2024 test period, "
            f"{int(d['station_days'].max()):,} station-days per method"
        )
        fig_positioning_tail(d, args.output_dir, prov)
    else:
        logger.warning(f"⚠️  {tails} not found - run positioning_robustness.py")


if __name__ == "__main__":
    main()
