"""Stochastic-model ablation: what does the predicted uncertainty buy in PPP? (R2.5)

Ported from ``src/analysis/weighting_ablation.py`` in the live PNN_STEC checkout, which
answers reviewer comment R2.5: isolate whether the uncertainty estimates themselves
improve positioning, rather than the STEC correction they accompany. No new PPP runs are
needed. Both weighting schemes have already been run for all three correction sources over
the full 2024 test period; this script pairs them.

A third arm exists for the Direct STEC correction: **fixed variance**, the same STEC
values with the per-observation sigma replaced by a constant
(``generate_fixed_variance_corrections.py``, run under ``--weight_opt iono`` so PPPx still
reads the uncertainty column). It is what separates "weighting by a model-derived
uncertainty" from "weighting by anything at all", which is the distinction R2.5 actually
asks about, and it lives in a separate experiment tree rather than in the six-arm sweep.

Weighting provenance (see project notes): ``daily_summary.csv`` means ``weight_opt=elev``;
``daily_summary_iono.csv`` means ``weight_opt=iono``. The fixed-variance experiment was
itself run with ``weight_opt=iono`` - only the sigma *column* it reads was overridden to a
constant upstream of PPPx - so ``load_fixed_variance`` below correctly globs
``daily_summary_iono.csv`` per that provenance rule; the "fixed" label in this module
describes the STEC correction file, not the ``weight_opt`` flag.

**The comparison is paired.** Only station-days that were solved successfully under every
arm being compared are kept - the unpaired arms differ by several hundred station-days,
and comparing their raw means would confound the weighting effect with which days each arm
happened to converge on. `paired_ablation` reports `dropped_unpaired` alongside the paired
count so the cost of pairing is never silent.

The 10 m station-day outlier rule (Figure 12 / Table 5) is reused from
`stec.positioning.metrics.exclude_outlier_station_days` rather than reimplemented.

**Common-set headline: median, not mean (owner decision 2026-08-28, applied here
2026-09-15).** `docs/revision/positioning_reporting.md` decided that positioning results
are reported as medians and distributions, never means - this table was simply missed
when that decision was made. The original evidence for that decision was one station-day
(URUM, DOY 365, a genuine PPPx solve failure at ~5,989 m under Direct STEC/elev) moving
the Direct STEC elevation-weighted mean by 25% while the median did not move at all.
**2026-09-18: that row is now excluded upstream** by `positioning_coverage`'s
solver-failure rule (a station-day where every one of the eight method/weighting arms
exceeds 100 m 3D RMS - `docs/revision/positioning_reporting.md`), so it no longer appears
in this table's population at all. The mean stays the more outlier-sensitive statistic on
whatever tail remains: on the current common set (N=10,673) Direct STEC's
elevation-weighted mean is 1.688 m against a median of 0.913 m
(`weighting_ablation/rebuilt/common_set.csv`).
`common_set_ablation` now reports `gain_median_%` as the headline figure; `gain_mean_%`
and the `*_mean` columns are still written to `common_set.csv` for the mean-vs-median
sensitivity comparison `positioning_reporting.md` argues from, not as a second number for
the manuscript to quote. This pass also adds the Pretrained Direct STEC correction to the
common-set table: `Pretrained_STEC_elev`/`Pretrained_STEC_iono` have always been in
`multiday_summary_all_weightings.csv` alongside the other three corrections, but the
table omitted them. `paired_ablation` (and the revision figure built from its `paired.csv`
output, `stec/viz/revision_figures.py::fig_weighting_ablation`) are untouched by either
change - only `common_set_ablation` gains the row and the median headline.

Usage::

    python -m stec.analysis.weighting_ablation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from ..positioning import metrics as pm
from .common_set_positioning import coverage_common_station_days

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "STEC_elev": ("Direct STEC", "elev"),
    "STEC_iono": ("Direct STEC", "iono"),
    "VTEC_elev": ("VTEC + Mapping", "elev"),
    "VTEC_iono": ("VTEC + Mapping", "iono"),
    "gim_elev": ("IGS GIM + Mapping", "elev"),
    "gim_iono": ("IGS GIM + Mapping", "iono"),
}
CORRECTION_ORDER = ["Direct STEC", "VTEC + Mapping", "IGS GIM + Mapping"]
# Arms in the order they should be reported; elev is the reference for gains.
WEIGHTING_ORDER = ["elev", "fixed", "iono"]
REFERENCE_WEIGHTING = "elev"

# common_set_ablation's own labels/order, kept separate from METHOD_LABELS/
# CORRECTION_ORDER above: those two feed paired_ablation and the published revision
# figure (fig_weighting_ablation), which must keep their existing three arms unchanged.
# Pretrained_STEC_elev/iono have always been in multiday_summary_all_weightings.csv
# alongside the other three corrections - only the common-set table previously omitted
# them.
COMMON_SET_METHOD_LABELS = dict(
    METHOD_LABELS,
    Pretrained_STEC_elev=("Pretrained Direct STEC", "elev"),
    Pretrained_STEC_iono=("Pretrained Direct STEC", "iono"),
)
COMMON_SET_CORRECTION_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]

# 2026-08-28: repointed off positioning_runs/20260216_2052/multiday_summary.csv, a
# 2026-02-16 snapshot from before the rebuild that nothing regenerated (56,457 rows,
# 245 dates, 55 stations), onto positioning_coverage's own
# multiday_summary_all_weightings.csv - the concatenation of its fresh iono and elev
# outputs, written by that stage's own main() once station-recovery elevation solves
# exist to concatenate. Must match stec/pipeline/stages.py's WEIGHTING_RUN constant,
# which is what this stage's declared `inputs` actually names.
DEFAULT_SUMMARY = (
    paths.analysis_result_dir("positioning_coverage", rebuilt=True)
    / "multiday_summary_all_weightings.csv"
)
FIXED_VARIANCE_RESULTS = (
    paths.LEGACY_EXPERIMENTS / "Fixed_Variance_STEC" / "positioning" / "results"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("weighting_ablation", rebuilt=True)


def load_fixed_variance(results_dir: Path) -> pd.DataFrame:
    """The fixed-variance arm, read from its per-day summaries.

    That run lives in its own experiment tree rather than in the six-arm sweep, so it has
    no multiday_summary.csv; its per-day daily_summary_iono.csv files are concatenated
    here into the same shape.
    """
    files = sorted(results_dir.glob("*/daily_summary_iono.csv"))
    if not files:
        logger.warning(f"no fixed-variance summaries under {results_dir}")
        return pd.DataFrame()
    frame = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    logger.info(f"fixed-variance arm: {len(files)} day(s), {len(frame):,} station-days")
    return frame.assign(correction="Direct STEC", weighting="fixed")[
        ["station", "doy", "error_3d_rms", "correction", "weighting"]
    ]


def paired_ablation(summary_path: Path) -> pd.DataFrame:
    """Pair every weighting arm per (station, day) and summarise the effect.

    Pairing is across *all* arms available for a correction, so adding the fixed-variance
    arm necessarily shrinks the Direct STEC sample: a station-day now has to have converged
    under three runs rather than two. That is the price of a like-for-like comparison and
    the count is reported alongside as `dropped_unpaired`.
    """
    runs = pd.read_csv(summary_path)
    runs = pm.exclude_outlier_station_days(runs)

    known = runs["method"].isin(METHOD_LABELS)
    if not known.all():
        logger.warning(
            f"ignoring unlabelled methods: {sorted(runs.loc[~known, 'method'].unique())}"
        )
    runs = runs[known]
    runs[["correction", "weighting"]] = pd.DataFrame(
        runs["method"].map(METHOD_LABELS).tolist(), index=runs.index
    )
    runs = runs[["station", "doy", "error_3d_rms", "correction", "weighting"]]

    rows = []
    for correction, group in runs.groupby("correction"):
        wide = group.pivot_table(
            index=["station", "doy"], columns="weighting", values="error_3d_rms"
        )
        unpaired = len(wide)
        wide = wide.dropna()
        arms = [w for w in WEIGHTING_ORDER if w in wide.columns]
        reference = wide[REFERENCE_WEIGHTING]

        row = {
            "correction": correction,
            "paired_station_days": len(wide),
            "dropped_unpaired": unpaired - len(wide),
            "arms": "+".join(arms),
        }
        for arm in arms:
            row[f"{arm}_mean"] = wide[arm].mean()
            row[f"{arm}_median"] = wide[arm].median()
            if arm != REFERENCE_WEIGHTING:
                difference = reference - wide[arm]
                row[f"gain_{arm}_%"] = 100 * difference.mean() / reference.mean()
                row[f"{arm}_better_frac_%"] = 100 * (difference > 0).mean()
        # Kept under its old name: the headline R2.5 number is iono vs elev.
        row["gain_%"] = row.get("gain_iono_%")

        # Also report iono-vs-elev on the *two-arm* pairing. Adding the fixed-variance arm
        # shrinks the Direct STEC sample (a station-day now needs three converged runs,
        # not two), which moves that number slightly; quoting both makes the shift
        # explicit rather than letting a previously published figure change under the
        # reader without explanation.
        two_arm = group[group["weighting"].isin([REFERENCE_WEIGHTING, "iono"])]
        two_wide = two_arm.pivot_table(
            index=["station", "doy"], columns="weighting", values="error_3d_rms"
        ).dropna()
        if {REFERENCE_WEIGHTING, "iono"}.issubset(two_wide.columns):
            pairwise = two_wide[REFERENCE_WEIGHTING] - two_wide["iono"]
            row["gain_iono_two_arm_%"] = (
                100 * pairwise.mean() / two_wide[REFERENCE_WEIGHTING].mean()
            )
            row["two_arm_station_days"] = len(two_wide)
        rows.append(row)

    return pd.DataFrame(rows).set_index("correction").reindex(CORRECTION_ORDER)


def common_set_ablation(
    summary_path: Path, common_station_days: pd.MultiIndex
) -> pd.DataFrame:
    """The same elevation-vs-predicted-uncertainty comparison as `paired_ablation`, but
    restricted to `common_station_days` (`common_set_positioning.
    coverage_common_station_days`) instead of each correction pairing on its own.

    This puts the ablation table on the same population as Table 5
    (`positioning_distributions.py`, owner instruction 2026-09-14): every correction now
    reports the same N (`len(common_station_days)`), rather than the 10,366/10,640/10,733
    `paired_ablation` reports (each correction's own elev+iono pairing, no cross-method
    restriction). No 10 m outlier exclusion is applied, for the same reason Table 5
    applies none (`docs/revision/positioning_reporting.md`): unlike `paired_ablation`,
    which drops it. **2026-09-18:** the one genuine PPPx solve failure this population
    used to carry regardless of that filter (URUM, DOY 365, ~5,988 m under Direct
    STEC/elev and ~5,940 m under Direct STEC/iono) is now excluded upstream, by
    `positioning_coverage`'s solver-failure rule (every one of the eight
    method/weighting arms over 100 m) - a data-quality exclusion independent of this
    stage's own outlier-filtering choice. The mean remains the more outlier-sensitive
    statistic on whatever tail is left: on the current population (N=10,673) Direct
    STEC's elevation-weighted mean is 1.688 m against a median of 0.913 m.

    **`gain_median_%` is the headline (owner decision 2026-08-28, applied 2026-09-15,
    see the module docstring): the mean-based `gain_mean_%` and the `*_mean` columns are
    kept only for the sensitivity comparison `docs/revision/positioning_reporting.md`
    argues from, not as a second number to report.** Also includes Pretrained Direct
    STEC, present in the source CSV alongside the other three corrections all along but
    previously left out of this table.
    """
    runs = pd.read_csv(summary_path)
    runs = runs.set_index(["station", "doy"])
    runs = runs[runs.index.isin(common_station_days)].reset_index()

    known = runs["method"].isin(COMMON_SET_METHOD_LABELS)
    runs = runs[known].copy()
    runs[["correction", "weighting"]] = pd.DataFrame(
        runs["method"].map(COMMON_SET_METHOD_LABELS).tolist(), index=runs.index
    )

    rows = []
    for correction, group in runs.groupby("correction"):
        wide = group.pivot_table(
            index=["station", "doy"], columns="weighting", values="error_3d_rms"
        ).dropna()
        reference = wide[REFERENCE_WEIGHTING]
        difference = reference - wide["iono"]
        reference_median = reference.median()
        difference_median = reference_median - wide["iono"].median()
        rows.append(
            {
                "correction": correction,
                "common_set_station_days": len(wide),
                "elev_median": wide["elev"].median(),
                "iono_median": wide["iono"].median(),
                "gain_median_%": 100 * difference_median / reference_median,
                "iono_better_frac_%": 100 * (difference > 0).mean(),
                # Sensitivity comparison only - see the docstring above and
                # docs/revision/positioning_reporting.md. Not the reported statistic.
                "elev_mean": wide["elev"].mean(),
                "iono_mean": wide["iono"].mean(),
                "gain_mean_%": 100 * difference.mean() / reference.mean(),
            }
        )
    return (
        pd.DataFrame(rows).set_index("correction").reindex(COMMON_SET_CORRECTION_ORDER)
    )


def fixed_variance_comparison(
    summary_path: Path, fixed_variance_dir: Path
) -> pd.Series | None:
    """Direct STEC under all three stochastic models, on one paired sample."""
    extra = load_fixed_variance(fixed_variance_dir)
    if extra.empty:
        return None
    extra = pm.exclude_outlier_station_days(extra)

    runs = pd.read_csv(summary_path)
    runs = pm.exclude_outlier_station_days(runs)
    runs = runs[runs["method"].isin(["STEC_elev", "STEC_iono"])].copy()
    runs["weighting"] = runs["method"].str.replace("STEC_", "", regex=False)
    combined = pd.concat(
        [runs[["station", "doy", "error_3d_rms", "weighting"]], extra],
        ignore_index=True,
    )

    wide = combined.pivot_table(
        index=["station", "doy"], columns="weighting", values="error_3d_rms"
    ).dropna()
    if not {"elev", "fixed", "iono"}.issubset(wide.columns):
        return None
    return pd.Series(
        {
            "paired_station_days": len(wide),
            "elev_mean_m": wide["elev"].mean(),
            "fixed_variance_mean_m": wide["fixed"].mean(),
            "predicted_uncertainty_mean_m": wide["iono"].mean(),
            "fixed_vs_elev_%": 100
            * (wide["elev"] - wide["fixed"]).mean()
            / wide["elev"].mean(),
            "iono_vs_elev_%": 100
            * (wide["elev"] - wide["iono"]).mean()
            / wide["elev"].mean(),
            "iono_vs_fixed_%": 100
            * (wide["fixed"] - wide["iono"]).mean()
            / wide["fixed"].mean(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--fixed-variance-dir", type=Path, default=FIXED_VARIANCE_RESULTS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    table = paired_ablation(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "paired.csv")

    # --- common set: same population as Table 5 (positioning_distributions.py),
    # so this table and Table 5 stop needing separate N caveats in the manuscript ---
    common_station_days = coverage_common_station_days(args.summary)
    common_table = common_set_ablation(args.summary, common_station_days)
    common_table.to_csv(args.output_dir / "common_set.csv")
    print(
        f"\n=== Predicted-uncertainty vs elevation weighting, common set "
        f"(N={len(common_station_days):,}) ==="
    )
    print(common_table.round(3).to_string())
    print(
        "\ngain_median_% is the headline figure (medians, not means - see the module"
        "\ndocstring for why). gain_mean_%/elev_mean/iono_mean are kept only for the"
        "\nmean-vs-median sensitivity comparison, not as a second number to report."
    )

    # The fixed-variance arm is kept out of the headline table and the figure: elevation
    # weighting is the operational default, so that is the comparison the figure should
    # carry, and a bar for a scheme nobody uses would be clutter. It is still computed,
    # because R2.5 asks for several stochastic models and this is the only arm that
    # separates "our sigma is informative" from "any weighting helps" - the STEC values
    # and weight_opt are identical to the iono arm, only the per-observation sigma
    # becomes a constant.
    fixed = fixed_variance_comparison(args.summary, args.fixed_variance_dir)
    if fixed is not None:
        fixed.to_frame("value").to_csv(args.output_dir / "fixed_variance.csv")
        print("\n=== Fixed variance vs the model's own sigma (Direct STEC) ===")
        print(fixed.round(3).to_string())
        print(
            "\nSame STEC and the same weight_opt iono; only the per-observation sigma"
            "\nis replaced by a constant. Reported as a number, not a figure bar."
        )

    print("=== Predicted-uncertainty vs elevation weighting, paired station-days ===")
    print(table.round(3).to_string())
    print(
        "\nPositive gain_% means uncertainty weighting reduced the 3D RMS error."
        "\nThe effect is confined to the correction whose uncertainty is genuinely"
        "\nobservation-level and model-derived."
    )
    logger.info(f"wrote {args.output_dir / 'paired.csv'}")


if __name__ == "__main__":
    main()
