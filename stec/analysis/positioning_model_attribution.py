"""Separating model STEC accuracy from everything else in the PPPx solve (owner
request, 2026-08-28).

Not a declared pipeline stage - like `positioning_geography.py`, the owner wants to see
this output before deciding whether/how it becomes part of the manuscript or a response
letter.

The question: certain stations position badly with Direct STEC even though some sit
close to a training station - HRAO is the named example. Positioning error is a
compound of (a) the model's STEC prediction quality at that station and (b) everything
else in the PPPx solve - satellite geometry, observation count, elevation coverage,
receiver/multipath quality. A station can have excellent STEC and still position badly.
This module is the deliverable that separates the two, joining three already-computed
tables rather than recomputing anything from the raw store:

1. `stec.analysis.station_independence`'s `per_station.csv` - per-station STEC RMSE/
   nRMSE and great-circle distance to the nearest training station, from the full
   242-day `predictions/finetuned_stec/own` store (see `.pipeline/station_independence.json`).
2. `stec.analysis.positioning_geography`'s `station_map_diff.csv` - per-station mean/
   median 3D positioning error for Direct STEC and IGS GIM, and their difference
   (`diff_mean_m`, positive = Direct STEC worse), 10 m outlier rule applied, iono
   weighting, via `positioning_summary.canonical_positioning_summary()`.
3. The canonical positioning table itself, for `n_epochs`/`mean_nsat`/`ref_source` -
   the cheap confound check for "is a bad station-day just short on satellites/epochs,
   independent of model error".

Sample-size discipline first: `station_map_diff.csv`'s `stec_station_days` ranges from
4 (MAR7) to 240. `N_THRESHOLD` below is not assumed - it is chosen by comparing each
station's standard error of the mean 3D error (`std / sqrt(N)`) against the
between-station spread of station means (`n_threshold_diagnostic.csv`): below N=30 at
least one station's SE exceeds ~28% of the between-station spread (HKSL, OWMG); at
N>=30 every station's SE stays under ~17% (the worst case is FAA1). Stations below the
threshold are reported as "suggestive-only", never mixed into the reliable list.

The key result: per-station STEC RMSE correlates strongly with **absolute** Direct STEC
positioning error (Spearman ~0.8) but only weakly, non-significantly with the
**Direct-STEC-minus-GIM** difference that the paper's headline "beats GIM" claim is
built on (Spearman ~0.14, p~0.36). The reason is not a flaw in the join: IGS GIM is
exposed to the *same* non-ionospheric confounds (geometry, satellite count, multipath)
that make a station's STEC hard to predict in the first place, so both methods degrade
together at a hard station and much of the model-accuracy signal cancels out of the
head-to-head comparison. Model-quality claims belong on the absolute-error or STEC-RMSE
scale; a station's head-to-head competitiveness against GIM is a different, noisier
question that this module also answers, but with the opposite verdict.

HRAO directly: distance to its nearest training station (HARB) is 2.0 km - closer than
all but a handful of test stations - and its STEC RMSE (3.53 TECU) sits in the best
quartile of all reliable-tier stations. Its apparent 1.06 m Direct-STEC-minus-GIM
disadvantage is real in the data but rests on only 14 station-days, below the N=30
reliability threshold, and the confound check finds nothing structurally wrong (full
epoch coverage, 100% ground-truth reference, unremarkable satellite count). HRAO is not
evidence the model fails near training stations; it is evidence that 14 station-days is
not enough to conclude anything either way.

Usage::

    python -m stec.analysis.positioning_model_attribution
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..positioning import metrics as pm
from .positioning_diagnostics import STEC_LABEL
from .positioning_geography import DEFAULT_OUTPUT_DIR as GEOGRAPHY_DIR
from .positioning_summary import PAPER_METHODS, canonical_positioning_summary
from .station_independence import DEFAULT_OUTPUT_DIR as STATION_INDEPENDENCE_DIR

logger = logging.getLogger(__name__)

# Chosen from n_threshold_diagnostic.csv - see the module docstring for the derivation.
N_THRESHOLD = 30

NAMED_STATIONS: tuple[str, ...] = (
    "HRAO",
    "PARC",
    "CPVG",
    "SUTH",
    "FAA1",
    "HKSL",
    "LMMF",
    "POVE",
    "KOUG",
    "NAUS",
)

DEFAULT_OUTPUT_DIR = GEOGRAPHY_DIR
DEFAULT_PLOTS_DIR = Path("plots/positioning_geography/model_accuracy_vs_positioning")
STATION_MAP_DIFF_CSV = GEOGRAPHY_DIR / "station_map_diff.csv"
STATION_INDEPENDENCE_CSV = STATION_INDEPENDENCE_DIR / "per_station.csv"


# --------------------------------------------------------------------------
# Loading and filtering the raw (station, doy, method) positioning table
# --------------------------------------------------------------------------


def load_stec_rows_with_confounds(summary_path: Path) -> pd.DataFrame:
    """Direct STEC rows only, 10 m outlier rule applied, with `n_epochs`/`mean_nsat`/
    `ref_source` attached - the one frame both the sample-size diagnostic (item 1) and
    the confound check (item 5) are built from, so the summary CSV is read once."""
    frame = pd.read_csv(
        summary_path,
        usecols=[
            "station",
            "method",
            "doy",
            "error_3d_rms",
            "n_epochs",
            "mean_nsat",
            "ref_source",
        ],
    )
    frame["station"] = frame["station"].str.upper()
    frame = frame[frame["method"].isin(PAPER_METHODS)].copy()
    frame["Method"] = frame["method"].map(PAPER_METHODS)
    kept = pm.exclude_outlier_station_days(frame, pm.OUTLIER_3D_RMS_M)
    return kept[kept["Method"] == STEC_LABEL]


# --------------------------------------------------------------------------
# 1. Sample-size discipline
# --------------------------------------------------------------------------


def station_reliability_diagnostic(stec_rows: pd.DataFrame) -> pd.DataFrame:
    """Per station: mean 3D error, its standard error, N, and SE as a percent of the
    *between-station* spread of station means - the number that decides `N_THRESHOLD`.
    A station whose SE is a large fraction of the signal being compared across stations
    cannot support a "this station is good/bad" claim, regardless of how large N looks
    in isolation."""
    se_table = stec_rows.groupby("station")["error_3d_rms"].agg(
        mean_error_3d="mean", std_error_3d="std", n="count"
    )
    se_table["se_of_mean"] = se_table["std_error_3d"] / np.sqrt(se_table["n"])
    between_station_sd = se_table["mean_error_3d"].std()
    se_table["se_over_between_sd_pct"] = (
        100 * se_table["se_of_mean"] / between_station_sd
    )
    return se_table.sort_values("n")


# --------------------------------------------------------------------------
# 2 & 4. Merge STEC accuracy against positioning performance and distance-to-training
# --------------------------------------------------------------------------


def merge_stec_accuracy_and_positioning(
    station_map_diff: pd.DataFrame,
    station_independence: pd.DataFrame,
    n_threshold: int = N_THRESHOLD,
) -> pd.DataFrame:
    """`station_map_diff.csv` (positioning) joined with `per_station.csv`
    (STEC accuracy + distance-to-training, both from `station_independence.py`, reused
    rather than recomputed). Inner join: a station absent from the prediction store
    (e.g. PARC - zero rows in `predictions/finetuned_stec/own` across all 242 days,
    confirmed directly) positions but cannot be model-accuracy-audited, and is dropped
    with a logged warning rather than silently coerced to NaN."""
    diff = station_map_diff.rename(columns={"stec_station_days": "n_positioning_days"})
    merged = diff.merge(
        station_independence[
            [
                "station",
                "distance_km",
                "nearest_train_station",
                "RMSE",
                "nRMSE_%",
                "mean_true_stec",
                "observations",
            ]
        ],
        on="station",
        how="inner",
    ).rename(columns={"RMSE": "stec_rmse_tecu", "nRMSE_%": "stec_nrmse_pct"})

    dropped = sorted(set(diff["station"]) - set(merged["station"]))
    if dropped:
        logger.warning(
            f"{len(dropped)} positioning station(s) have no STEC-accuracy row (no "
            f"prediction-store rows for that station): {dropped}"
        )

    merged["reliable"] = merged["n_positioning_days"] >= n_threshold
    return merged.sort_values("diff_mean_m", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# 3. THE KEY JOIN - correlations
# --------------------------------------------------------------------------


def correlate(df: pd.DataFrame, x: str, y: str, label: str) -> dict:
    """One row of Spearman + Pearson correlation between two per-station columns,
    dropping rows where either is NaN (a station can have positioning data but no
    surviving Direct STEC row after the outlier rule, e.g. BRMG/LICC)."""
    valid = df.dropna(subset=[x, y])
    sp_r, sp_p = stats.spearmanr(valid[x], valid[y])
    pe_r, pe_p = stats.pearsonr(valid[x], valid[y])
    return {
        "comparison": label,
        "x": x,
        "y": y,
        "spearman_r": sp_r,
        "spearman_p": sp_p,
        "pearson_r": pe_r,
        "pearson_p": pe_p,
        "n": len(valid),
    }


def key_join_correlations(reliable: pd.DataFrame) -> pd.DataFrame:
    """Every correlation the deliverable needs, reliable stations only: does STEC
    accuracy predict positioning outcomes, and does distance-to-training predict either
    STEC accuracy or positioning outcomes."""
    rows = [
        correlate(
            reliable,
            "stec_rmse_tecu",
            "diff_mean_m",
            "STEC RMSE vs (STEC-GIM) positioning diff",
        ),
        correlate(
            reliable,
            "stec_rmse_tecu",
            "stec_mean_m",
            "STEC RMSE vs absolute Direct-STEC positioning error",
        ),
        correlate(
            reliable,
            "stec_nrmse_pct",
            "diff_mean_m",
            "STEC nRMSE% vs (STEC-GIM) positioning diff",
        ),
        correlate(
            reliable,
            "stec_nrmse_pct",
            "stec_mean_m",
            "STEC nRMSE% vs absolute Direct-STEC positioning error",
        ),
        correlate(
            reliable,
            "distance_km",
            "stec_rmse_tecu",
            "Distance-to-training vs STEC RMSE",
        ),
        correlate(
            reliable,
            "distance_km",
            "stec_nrmse_pct",
            "Distance-to-training vs STEC nRMSE%",
        ),
        correlate(
            reliable,
            "distance_km",
            "diff_mean_m",
            "Distance-to-training vs positioning diff",
        ),
        correlate(
            reliable,
            "distance_km",
            "stec_mean_m",
            "Distance-to-training vs absolute Direct-STEC positioning error",
        ),
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Confound check
# --------------------------------------------------------------------------


def station_confound_summary(stec_rows: pd.DataFrame) -> pd.DataFrame:
    """Per station: mean epoch count, mean satellite count, percent of station-days
    using a ground-truth (vs mean-position) reference - the cheap check for "is a bad
    station explained by something the solver saw, independent of model error"."""
    return stec_rows.groupby("station").agg(
        mean_n_epochs=("n_epochs", "mean"),
        mean_nsat=("mean_nsat", "mean"),
        pct_ref_ground_truth=(
            "ref_source",
            lambda s: 100 * (s == "ground_truth").mean(),
        ),
        n=("station", "size"),
    )


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-path", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument(
        "--station-map-diff-csv", type=Path, default=STATION_MAP_DIFF_CSV
    )
    parser.add_argument(
        "--station-independence-csv", type=Path, default=STATION_INDEPENDENCE_CSV
    )
    parser.add_argument("--n-threshold", type=int, default=N_THRESHOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if (
        not args.station_map_diff_csv.exists()
        or not args.station_independence_csv.exists()
    ):
        raise SystemExit(
            f"Missing an input this module reuses rather than recomputes: run "
            f"`python -m stec.analysis.positioning_geography` and "
            f"`python -m stec.analysis.station_independence` first if either "
            f"{args.station_map_diff_csv} or {args.station_independence_csv} is absent."
        )

    stec_rows = load_stec_rows_with_confounds(args.summary_path)

    # 1. Sample-size discipline.
    se_table = station_reliability_diagnostic(stec_rows)
    se_table.to_csv(args.output_dir / "n_threshold_diagnostic.csv")
    n_below = int((se_table["n"] < args.n_threshold).sum())
    n_above = int((se_table["n"] >= args.n_threshold).sum())
    worst_se = se_table.loc[
        se_table["n"] >= args.n_threshold, "se_over_between_sd_pct"
    ].max()
    print(f"=== Sample-size diagnostic (N_THRESHOLD={args.n_threshold}) ===")
    print(
        f"{n_above} stations reliable, {n_below} suggestive-only (positioning table's own station population)."
    )
    print(f"Worst-case SE/between-station-SD among reliable stations: {worst_se:.1f}%")

    # 2 & 4. Merge STEC accuracy + distance-to-training with positioning performance.
    station_map_diff = pd.read_csv(args.station_map_diff_csv)
    station_independence = pd.read_csv(args.station_independence_csv)
    merged = merge_stec_accuracy_and_positioning(
        station_map_diff, station_independence, args.n_threshold
    )
    merged.to_csv(args.output_dir / "station_stec_vs_positioning.csv", index=False)
    reliable = merged[merged["reliable"]].copy()
    suggestive = merged[~merged["reliable"]].copy()
    print(
        f"\n{len(reliable)} reliable / {len(suggestive)} suggestive-only stations with a STEC-accuracy match."
    )

    # 3. THE KEY JOIN.
    correlations = key_join_correlations(reliable)
    correlations.to_csv(
        args.output_dir / "stec_positioning_correlations.csv", index=False
    )
    print("\n=== Correlations, reliable stations only ===")
    print(correlations.to_string(index=False))

    # 4b. Named stations report card.
    cols = [
        "station",
        "n_positioning_days",
        "reliable",
        "diff_mean_m",
        "stec_mean_m",
        "gim_mean_m",
        "distance_km",
        "nearest_train_station",
        "stec_rmse_tecu",
        "stec_nrmse_pct",
    ]
    named = merged[merged["station"].isin(NAMED_STATIONS)][cols].set_index("station")
    named = named.reindex([s for s in NAMED_STATIONS if s in named.index])
    named.to_csv(args.output_dir / "named_stations_report_card.csv")
    print("\n=== Named stations report card ===")
    print(named.to_string())

    # 5. Confound check.
    station_conf = station_confound_summary(stec_rows)
    station_conf.to_csv(args.output_dir / "station_confound_check.csv")
    reliable_conf = reliable.merge(station_conf, on="station", how="inner")
    confound_correlations = pd.DataFrame(
        [
            correlate(
                reliable_conf,
                "mean_nsat",
                "diff_mean_m",
                "mean_nsat vs positioning diff",
            ),
            correlate(
                reliable_conf,
                "mean_nsat",
                "stec_mean_m",
                "mean_nsat vs absolute Direct-STEC error",
            ),
            correlate(
                reliable_conf,
                "mean_n_epochs",
                "diff_mean_m",
                "mean_n_epochs vs positioning diff",
            ),
        ]
    )
    confound_correlations.to_csv(
        args.output_dir / "confound_correlations.csv", index=False
    )
    print("\n=== Confound correlations, reliable stations ===")
    print(confound_correlations.to_string(index=False))

    if not args.no_figures:
        make_figures(merged, reliable, suggestive, args.plots_dir, args.n_threshold)

    logger.info(f"wrote {args.output_dir}")


def make_figures(
    merged: pd.DataFrame,
    reliable: pd.DataFrame,
    suggestive: pd.DataFrame,
    plots_dir: Path,
    n_threshold: int,
) -> None:
    """Two 2-panel figures. Deliberately in this module rather than a separate
    `stec/viz/` file - this is a one-off owner-request join, not a maintained figure
    family, so the analysis/viz split other modules use would add indirection without
    reuse to justify it. `STEC_COLOR` only, per the repo's colour rule: this compares
    Direct STEC's own outcomes against itself (RMSE vs its own positioning error), not
    the four methods against each other, so `APPROACH_COLORS`' other three hues do not
    apply here."""
    import matplotlib.pyplot as plt

    from ..viz.style import STEC_COLOR, configure_plotting, save_plot

    configure_plotting()
    plots_dir.mkdir(parents=True, exist_ok=True)

    def label_points(ax, df, x, y):
        for _, row in df[df["station"].isin(NAMED_STATIONS)].iterrows():
            ax.annotate(
                row["station"],
                (row[x], row[y]),
                fontsize=11,
                xytext=(5, 5),
                textcoords="offset points",
            )

    def scatter_pair(ax, x, y):
        ax.scatter(
            reliable[x],
            reliable[y],
            s=70,
            color=STEC_COLOR,
            label=f"reliable (N>={n_threshold}), n={len(reliable)}",
            zorder=3,
        )
        ax.scatter(
            suggestive[x],
            suggestive[y],
            s=50,
            facecolors="none",
            edgecolors="#888888",
            label=f"suggestive-only (N<{n_threshold}), n={len(suggestive)}",
            zorder=2,
        )
        label_points(ax, merged, x, y)

    # --- Figure 1: the key join - STEC accuracy vs positioning performance ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    scatter_pair(axes[0], "stec_rmse_tecu", "diff_mean_m")
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    axes[0].set_xlabel("Per-station STEC RMSE (TECU)")
    axes[0].set_ylabel(
        "Positioning diff, Direct STEC - IGS GIM (m)\n(positive = STEC worse)"
    )
    axes[0].set_title("vs positioning advantage (weak/no relation)")
    axes[0].legend(loc="upper left", fontsize=12)

    scatter_pair(axes[1], "stec_rmse_tecu", "stec_mean_m")
    axes[1].set_xlabel("Per-station STEC RMSE (TECU)")
    axes[1].set_ylabel("Direct STEC absolute 3D positioning error (m)")
    axes[1].set_title("vs absolute positioning error (strong)")
    axes[1].legend(loc="upper left", fontsize=12)

    fig.suptitle("Per-station model accuracy vs positioning performance (2026-08-28)")
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=6.0)
    save_plot(fig, "stec_accuracy_vs_positioning.png", plots_dir)
    merged[
        [
            "station",
            "n_positioning_days",
            "reliable",
            "stec_rmse_tecu",
            "stec_nrmse_pct",
            "diff_mean_m",
            "stec_mean_m",
            "gim_mean_m",
            "distance_km",
        ]
    ].to_csv(plots_dir / "stec_accuracy_vs_positioning.csv", index=False)

    # --- Figure 2: distance to nearest training station vs both outcomes ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    scatter_pair(axes[0], "distance_km", "stec_rmse_tecu")
    axes[0].set_xscale("symlog", linthresh=10)
    axes[0].set_xlabel("Distance to nearest training station (km)")
    axes[0].set_ylabel("Per-station STEC RMSE (TECU)")
    axes[0].set_title("Distance to training vs STEC accuracy")

    scatter_pair(axes[1], "distance_km", "diff_mean_m")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    axes[1].set_xscale("symlog", linthresh=10)
    axes[1].set_xlabel("Distance to nearest training station (km)")
    axes[1].set_ylabel("Positioning diff, Direct STEC - IGS GIM (m)")
    axes[1].set_title("Distance to training vs positioning advantage")

    fig.suptitle(
        "HRAO check: proximity to training does not guarantee positioning advantage"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=6.0)
    save_plot(fig, "distance_to_training_vs_outcomes.png", plots_dir)
    merged[
        [
            "station",
            "n_positioning_days",
            "reliable",
            "distance_km",
            "nearest_train_station",
            "stec_rmse_tecu",
            "diff_mean_m",
        ]
    ].to_csv(plots_dir / "distance_to_training_vs_outcomes.csv", index=False)

    logger.info(f"wrote figures + CSVs to {plots_dir}")


if __name__ == "__main__":
    main()
