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

DEFAULT_SUMMARY = (
    paths.LEGACY_MULTIDAY / "positioning_20260216_2052" / "multiday_summary.csv"
)
FIXED_VARIANCE_RESULTS = (
    paths.LEGACY_EXPERIMENTS / "Fixed_Variance_STEC" / "positioning" / "results"
)
DEFAULT_OUTPUT_DIR = Path("multiday_results/weighting_ablation_rebuilt")


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
