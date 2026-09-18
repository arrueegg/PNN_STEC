"""STEC prediction accuracy on storm days against quiet days (R1.4/R1.7, STEC domain).

Two other modules already stratify by geomagnetic activity, and this is neither of
them. ``activity_stratification.py`` bins the same per-day metrics into four Dst
severity bands and four F10.7 bands, for a figure. ``storm_stratification.py``
answers the *positioning* question (R2.7) - a binary storm/quiet split of
per-station-day PPPx position error. This module is the STEC-domain twin of that
second one: a binary storm/quiet split (not bands) of per-day STEC prediction error,
so R1.4/R1.7 can say plainly "does the model's STEC accuracy hold up during a storm",
the same way the positioning table already says it for position error.

**Storm threshold: a daily minimum Dst of -50 nT** - imported as
``STORM_DST_THRESHOLD_NT`` from ``storm_stratification.py`` rather than duplicated, so
the STEC-domain and positioning-domain tables can never quietly disagree about what
"storm" means. That is a different rule from the per-observation
``Kp >= 37 or Dst <= -33`` classification in ``src/analysis/scenario_evaluation.py``,
which marks individual hours rather than days and is gated behind
``evaluation.enable_scenarios`` (defaults ``False``, never actually runs). Applying the
per-observation rule at day granularity marks a different set of days as storms; see
``storm_stratification.py``'s own module docstring for the exact counts. Do not port a
day count from one rule into the other's table.

Reads ``daily_metrics.py``'s ``per_day.csv`` directly - no re-inference, no GPU - and
joins it to the daily OMNI extremes the same way ``activity_stratification.py`` does.
Because the IGS GIM baseline in that CSV is only trustworthy once
``repair_gim_baseline`` has run (a float32-denormalised ``doy`` truncated to the wrong
IONEX day on 12 dates and inflated the published GIM RMSE - see CLAUDE.md's Gotchas),
this module reuses ``activity_stratification.py``'s own
``require_repaired_daily_metrics`` gate rather than re-implementing the same check a
second time.

Every method within one dataset is stratified over the *same* day set: the join
against ``per_day.csv`` is an inner join on ``doy``, so a method that silently lost a
day (the exact failure ``daily_metrics_summary_has_consistent_day_counts`` in
``stec/pipeline/stages.py`` was added to catch for Tables 3/4) would show up here as a
mismatched ``n_days`` between methods in the same (dataset, regime) cell rather than
being silently absorbed into a slightly different population per method.

Usage::

    python -m stec.analysis.storm_stratification_stec
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import paths
from .activity_stratification import (
    DEFAULT_REPAIR_REPORT,
    require_repaired_daily_metrics,
)
from .daily_metrics import DATASET_LABELS as DAILY_METRICS_DATASET_LABELS
from .daily_metrics import DEFAULT_OUTPUT_DIR as DAILY_METRICS_DIR
from .daily_metrics import MODELS as DAILY_METRICS_MODELS
from .storm_stratification import STORM_DST_THRESHOLD_NT, load_daily_geomagnetic_indices

logger = logging.getLogger(__name__)

DEFAULT_SWI_PATH = paths.OMNI_INDICES
DEFAULT_DAILY_METRICS_CSV = DAILY_METRICS_DIR / "per_day.csv"
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "storm_stratification_stec", rebuilt=True
)

# The two great storms of the 2024 test period (DOY 131-133, Dst_min ~ -406 nT; DOY
# 282-285, Dst_min ~ -333 nT - see storm_stratification.py's module docstring) are what
# "strongest_storm_days.csv" exists to surface individually, rather than folding them
# into the same "storm" bucket as a day that only just crossed -50 nT. -300 nT is a
# fixed threshold, the same convention DST_BINS/F107_BINS in activity_stratification.py
# use, so it does not move if a future test period has a different worst day.
EXTREME_STORM_DST_THRESHOLD_NT = -300.0

# The four methods, in the order every printed table in this module uses, built from
# daily_metrics.py's own MODELS dict rather than re-typing the label strings - a
# renamed model there is a naming clash here, not a silent divergence.
MODEL_RMSE_COLUMNS: dict[str, str] = {
    DAILY_METRICS_MODELS["stec_pred"]: "direct_stec_rmse",
    DAILY_METRICS_MODELS["gim_stec"]: "igs_gim_rmse",
    DAILY_METRICS_MODELS["vtec_model_stec"]: "vtec_mapping_rmse",
    DAILY_METRICS_MODELS["pretrained_stec_pred"]: "pretrained_rmse",
}
MODEL_ORDER = list(MODEL_RMSE_COLUMNS)


def classify_days(
    doys: pd.Series, year: int, swi_path: Path = DEFAULT_SWI_PATH
) -> pd.DataFrame:
    """One row per DOY in `doys`, with the day's OMNI extremes and storm/quiet regime.

    Regime depends only on the OMNI indices for `year`, never on which dataset or
    method a caller is about to join this against - computed once here and reused by
    every downstream table, rather than re-joined per (dataset, Model) group.
    """
    wanted = sorted(int(doy) for doy in pd.unique(doys))
    indices = load_daily_geomagnetic_indices(year, swi_path)
    indices = indices[indices["doy"].isin(wanted)].copy()
    indices["date"] = indices["doy"].apply(lambda doy: f"{year}-{doy:03d}")
    indices["regime"] = (indices["dst_min"] <= STORM_DST_THRESHOLD_NT).map(
        {True: "storm", False: "quiet"}
    )
    return (
        indices.rename(columns={"dst_min": "min_dst", "kp_max": "max_kp"})[
            ["date", "doy", "min_dst", "max_kp", "regime"]
        ]
        .sort_values("doy")
        .reset_index(drop=True)
    )


def stratify(
    daily_metrics_csv: Path,
    year: int = 2024,
    swi_path: Path = DEFAULT_SWI_PATH,
    repair_report: Path = DEFAULT_REPAIR_REPORT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join `per_day.csv` to the daily storm/quiet classification.

    Refuses to run without a repair report for the same reason
    `activity_stratification.stratify` does: the un-repaired IGS GIM baseline changes
    which method looks best in a storm, not just by how much (see the module
    docstring). Returns `(day_classification, merged)` - the classification alone, and
    the per-day metrics joined to it (inner on `doy`, so a day the OMNI archive has no
    record for is dropped rather than kept with an unclassified regime).
    """
    require_repaired_daily_metrics(daily_metrics_csv, repair_report)

    per_day = pd.read_csv(daily_metrics_csv)
    per_day = per_day[per_day["year"] == year]
    if per_day.empty:
        raise RuntimeError(f"{daily_metrics_csv} has no rows for year {year}")

    day_classification = classify_days(per_day["doy"], year, swi_path)
    merged = per_day.merge(day_classification[["doy", "regime"]], on="doy", how="inner")
    dropped = set(per_day["doy"]) - set(merged["doy"])
    if dropped:
        logger.warning(
            f"no geomagnetic indices for DOY {sorted(dropped)} - excluded from "
            "stratification"
        )
    return day_classification, merged


def build_by_regime(merged: pd.DataFrame) -> pd.DataFrame:
    """Median/Q1/Q3 of daily RMSE and MAE, plus mean daily RMSE, per (dataset, method,
    regime). Reuses the daily values `daily_metrics.py` already computed per day rather
    than re-touching the prediction store - these are statistics of per-day numbers,
    the same convention `daily_metrics.summarise`'s `RMSE_median`/`RMSE_q1`/`RMSE_q3`
    columns use for the pooled (non-stratified) table.
    """

    def aggregate(group: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_days": int(group["doy"].nunique()),
                "rmse_median": group["RMSE"].median(),
                "rmse_q1": group["RMSE"].quantile(0.25),
                "rmse_q3": group["RMSE"].quantile(0.75),
                "rmse_mean": group["RMSE"].mean(),
                "mae_median": group["MAE"].median(),
                "mae_q1": group["MAE"].quantile(0.25),
                "mae_q3": group["MAE"].quantile(0.75),
            }
        )

    by_regime = (
        merged.groupby(["dataset", "Model", "regime"], observed=True)
        .apply(aggregate, include_groups=False)
        .reset_index()
        .rename(columns={"Model": "method"})
    )
    order = {method: position for position, method in enumerate(MODEL_ORDER)}
    by_regime = by_regime.sort_values(
        ["dataset", "method", "regime"],
        key=lambda column: column.map(order) if column.name == "method" else column,
    ).reset_index(drop=True)
    return by_regime


def build_strongest_storm_days(
    merged: pd.DataFrame,
    day_classification: pd.DataFrame,
    threshold: float = EXTREME_STORM_DST_THRESHOLD_NT,
) -> pd.DataFrame:
    """One row per (dataset, day) with min Dst below `threshold`, every method's daily
    RMSE as its own column, and a severity `rank` (1 = most negative Dst) shared by
    every dataset - the same storm is equally severe regardless of which dataset's
    accuracy is being read off it.
    """
    base_columns = ["dataset", "rank", "date", "doy", "min_dst"]
    all_columns = base_columns + list(MODEL_RMSE_COLUMNS.values())

    extreme_days = (
        day_classification[day_classification["min_dst"] < threshold]
        .sort_values("min_dst")
        .reset_index(drop=True)
    )
    if extreme_days.empty:
        return pd.DataFrame(columns=all_columns)
    extreme_days = extreme_days.assign(rank=extreme_days.index + 1)

    subset = merged[merged["doy"].isin(extreme_days["doy"])]
    wide = subset.pivot_table(
        index=["dataset", "doy"], columns="Model", values="RMSE"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns=MODEL_RMSE_COLUMNS).merge(
        extreme_days[["doy", "date", "min_dst", "rank"]], on="doy", how="left"
    )
    present_rmse_columns = [
        column for column in MODEL_RMSE_COLUMNS.values() if column in wide.columns
    ]
    return (
        wide[base_columns + present_rmse_columns]
        .sort_values(["dataset", "rank"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--daily-metrics-csv", type=Path, default=DEFAULT_DAILY_METRICS_CSV
    )
    parser.add_argument("--repair-report", type=Path, default=DEFAULT_REPAIR_REPORT)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--swi-path", type=Path, default=DEFAULT_SWI_PATH)
    parser.add_argument(
        "--extreme-dst-threshold", type=float, default=EXTREME_STORM_DST_THRESHOLD_NT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    day_classification, merged = stratify(
        args.daily_metrics_csv, args.year, args.swi_path, args.repair_report
    )

    tables = {
        "day_classification": day_classification,
        "by_regime": build_by_regime(merged),
        "strongest_storm_days": build_strongest_storm_days(
            merged, day_classification, args.extreme_dst_threshold
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        path = args.output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        logger.info(f"wrote {path}")
        print(f"\n=== {name} ===")
        print(table.round(3).to_string(index=False))

    own_label = DAILY_METRICS_DATASET_LABELS["own"]
    own = merged[merged["dataset"] == own_label]
    if not own.empty:
        print(f"\n=== {own_label}: median daily RMSE, quiet vs storm ===")
        pivot = (
            own.groupby(["Model", "regime"])["RMSE"]
            .median()
            .unstack("regime")
            .reindex(MODEL_ORDER)
        )
        print(pivot.round(2).to_string())

    storm_days = int((day_classification["regime"] == "storm").sum())
    logger.info(
        f"{storm_days} storm day(s) (min Dst <= {STORM_DST_THRESHOLD_NT:.0f} nT) of "
        f"{len(day_classification)} classified days"
    )


if __name__ == "__main__":
    main()
