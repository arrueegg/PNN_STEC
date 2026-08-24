"""Table 5 rebuilt on a single common station-day set (R1.5).

Backs the R1.5 reviewer-response numbers in `docs/revision/response_to_reviewers.md`, not a
printed manuscript table - the manuscript has 5 tables and no lettered appendix, so there is
no "Table A1" for this to be. `stec/pipeline/stages.py`'s `common_set_positioning` stage
declares `canonical_for=None` for exactly this reason.

Ported from ``src/analysis/common_set_positioning.py`` in the live PNN_STEC checkout.

The published Table 5 compares four corrections over *different* populations. After the
10 m outlier rule the counts are gim_iono 10,809 against STEC_iono 8,280, VTEC_iono 8,266
and Pretrained_STEC_iono 8,195: the IGS GIM is solved for a median of 45 stations per day
where every machine-learning method manages 35.

The cause is upstream and legitimate. Those ~2,810 extra station-days are stations absent
from the STEC database on that day, so no ML correction can exist for them; they are
predominantly equatorial (NKLG, CHPG, KOUG, CPVG, FAA1, LMMF, PTAG, WARK) and they are hard
- 2.24 m against 1.40 m on the shared days. Comparing methods across them therefore mixes a
population effect into the reported improvement, which is what this analysis removes by
intersecting.

**This uses a different station-day population from Table 5 by design.** Requiring both
weightings costs the IGS GIM roughly 3,000 station-days, so the N reported here is smaller
than Table 5's - this module reports its own N (``result["common_station_days"]``) rather
than reusing Table 5's, and callers must state each table's N separately.

Sources, all of which agree where they overlap (max |delta error_3d_rms| = 0.0 in the live
checkout):

* ``canonical_positioning_summary()`` (imported from ``positioning_summary.py``) - the
  ``positioning_coverage`` stage's rebuilt aggregate of every per-day result on disk,
  including ``Pretrained_STEC_iono``, falling back to the frozen
  ``positioning_comparison_3way/`` (the original paper's narrower run) only if that stage
  has never been run.
* ``positioning_20260216_2052/`` - the weighting ablation, 245 dates, six arms.
* the pretrained *elevation* arm, read from its per-day summaries because it post-dates
  both trees.

Everything is restricted to the dates of the three-way tree, so the ablation's extra days
cannot leak in.

**Outlier rule correction from the source script**: the live checkout's version of this
analysis applied the 10 m rule with a strict ``<`` (``error_3d_rms < OUTLIER_3D_RMS_M``),
while ``positioning_summary.py`` and ``oracle_benchmark.py`` both use ``<=``. This port
uses ``stec.positioning.metrics.exclude_outlier_station_days`` (``<=``, matching
``pm.OUTLIER_3D_RMS_M``) throughout, so the rule is now applied identically in all three
ported positioning analyses - a deliberate correction, not a silent behaviour change: it
only affects station-days sitting exactly on the 10.0 m boundary.

Weighting provenance: ``daily_summary.csv`` means ``weight_opt=elev``;
``daily_summary_iono.csv`` means ``weight_opt=iono``.

Usage::

    python -m stec.analysis.common_set_positioning
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from ..positioning import metrics as pm
from .positioning_summary import (
    DEFAULT_WEIGHTING_SUMMARY,
    canonical_positioning_summary,
)

logger = logging.getLogger(__name__)

BASELINE_ARM = "IGS GIM + Mapping / uncertainty"

ARM_LABELS = {
    "STEC_iono": ("Direct STEC", "uncertainty"),
    "STEC_elev": ("Direct STEC", "elevation"),
    "VTEC_iono": ("VTEC + Mapping", "uncertainty"),
    "VTEC_elev": ("VTEC + Mapping", "elevation"),
    "gim_iono": ("IGS GIM + Mapping", "uncertainty"),
    "gim_elev": ("IGS GIM + Mapping", "elevation"),
    "Pretrained_STEC_iono": ("Pretrained Direct STEC", "uncertainty"),
    "Pretrained_STEC_elev": ("Pretrained Direct STEC", "elevation"),
}
ARM_ORDER = [
    "Direct STEC",
    "Pretrained Direct STEC",
    "VTEC + Mapping",
    "IGS GIM + Mapping",
]
COLUMNS = ["station", "doy", "method", "error_3d_rms", "error_2d_rms", "u_rms"]

DEFAULT_EXPERIMENT = paths.LEGACY_EXPERIMENTS / (
    "Pretrain_STEC_BayesianResNetSTEC_h1024_l4_nh4_v128x4_g32x2_"
    "lr1e-3_bs1024_GNLL_Adam_ReduceLROnPlateau_sub500K_SH5_ps0.1_kl5w0.1_lw1e-1_SWI"
)
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir("common_set_positioning", rebuilt=True)


def load_tree(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
    return frame[frame["method"].isin(ARM_LABELS)]


def load_pretrained_elev(experiment: Path) -> pd.DataFrame:
    """The one arm that exists only as per-day summaries.

    ``daily_summary.csv`` is the elevation-weighted output (the uncertainty arm writes
    ``daily_summary_iono.csv``), and inside it the model rows are labelled ``model`` -
    ``run_pipeline`` renames them only when it aggregates, which this arm never went
    through.
    """
    files = sorted(experiment.glob("positioning/results/2024*/daily_summary.csv"))
    if not files:
        logger.warning(f"no elevation-weighted summaries under {experiment}")
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.concat(
        (pd.read_csv(f, usecols=lambda c: c in COLUMNS) for f in files),
        ignore_index=True,
    )
    frame = frame[frame["method"] == "model"].copy()
    frame["method"] = "Pretrained_STEC_elev"
    logger.info(f"pretrained elevation arm: {len(files)} day(s), {len(frame):,} rows")
    return frame


def build(three_way: Path, ablation: Path, experiment: Path) -> dict:
    """Restrict every arm to the three-way tree's dates, intersect on station-day, and
    summarise. Returns the summary table plus the size of the common set, so callers can
    report the N this table used rather than assuming it matches Table 5's."""
    paper = load_tree(three_way)
    dates = set(paper["doy"].unique())
    logger.info(f"restricting everything to the {len(dates)} dates of the 3way tree")

    runs = pd.concat(
        [paper, load_tree(ablation), load_pretrained_elev(experiment)],
        ignore_index=True,
    )
    runs = runs[runs["doy"].isin(dates)]
    runs = pm.exclude_outlier_station_days(runs)
    runs = runs.drop_duplicates(subset=["station", "doy", "method"])

    runs["arm"] = runs["method"].map(lambda m: " / ".join(ARM_LABELS[m]))
    per_arm_before = runs.groupby("arm").size()

    # The common set: station-days solved under every arm present.
    arms = sorted(runs["arm"].unique())
    counts = runs.groupby(["station", "doy"])["arm"].nunique()
    common = set(counts[counts == len(arms)].index)
    logger.info(
        f"{len(arms)} arm(s); {len(common):,} station-days solved under all of them"
    )

    keyed = runs.set_index(["station", "doy"])
    common_runs = keyed[keyed.index.isin(common)].reset_index()

    summary = common_runs.groupby("arm").agg(
        station_days=("error_3d_rms", "size"),
        rms_3d_mean=("error_3d_rms", "mean"),
        rms_3d_median=("error_3d_rms", "median"),
        rms_2d_mean=("error_2d_rms", "mean"),
        up_mean=("u_rms", "mean"),
    )
    summary["lost_to_intersection"] = (
        per_arm_before.reindex(summary.index) - summary["station_days"]
    )

    # Improvement over the uncertainty-weighted GIM, three ways. Ratio-of-means is what
    # the table quotes; the paired statistics answer "is it better on a typical
    # station-day", which a ratio of aggregates cannot.
    wide = common_runs.pivot_table(
        index=["station", "doy"], columns="arm", values="error_3d_rms"
    )
    reference = wide[BASELINE_ARM]
    for arm in summary.index:
        gain = 100 * (reference - wide[arm]) / reference
        summary.loc[arm, "gain_ratio_of_means_pct"] = (
            100 * (reference.mean() - wide[arm].mean()) / reference.mean()
        )
        summary.loc[arm, "gain_paired_mean_pct"] = gain.mean()
        summary.loc[arm, "gain_paired_median_pct"] = gain.median()
        summary.loc[arm, "win_rate_pct"] = 100 * (wide[arm] < reference).mean()

    order = [f"{c} / {w}" for c in ARM_ORDER for w in ("uncertainty", "elevation")]
    summary = summary.reindex([a for a in order if a in summary.index])
    return {"summary": summary, "common_station_days": len(common), "arms": len(arms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--three-way", type=Path, default=canonical_positioning_summary()
    )
    parser.add_argument("--ablation", type=Path, default=DEFAULT_WEIGHTING_SUMMARY)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    result = build(args.three_way, args.ablation, args.experiment)
    summary = result["summary"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "table5_common_set.csv")

    print(
        f"=== Table 5 on the common set: {result['common_station_days']:,} station-days "
        f"solved under all {result['arms']} arm(s) ==="
    )
    print(
        summary[
            ["station_days", "rms_3d_mean", "rms_3d_median", "rms_2d_mean", "up_mean"]
        ]
        .round(3)
        .to_string()
    )
    print(f"\n--- improvement over {BASELINE_ARM} ---")
    print(
        summary[
            [
                "gain_ratio_of_means_pct",
                "gain_paired_mean_pct",
                "gain_paired_median_pct",
                "win_rate_pct",
            ]
        ]
        .round(1)
        .to_string()
    )
    print("\n--- station-days each arm loses to the intersection ---")
    print(summary["lost_to_intersection"].astype(int).to_string())
    logger.info(f"wrote {args.output_dir / 'table5_common_set.csv'}")


if __name__ == "__main__":
    main()
