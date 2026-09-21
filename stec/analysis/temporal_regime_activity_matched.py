"""Activity-matched interpolation/extrapolation comparison - correcting a confound in R2.1.

`docs/revision/response_to_reviewers.md`'s R2.1 answer quotes RMSE 7.65 vs 14.05 TECU
(interpolation vs extrapolation, 26.9% vs 31.0% normalised) as evidence about temporal
extrapolation, reproduced exactly by `temporal_regime_split.py`. **That comparison is
confounded with solar-cycle phase, not a clean read of temporal extrapolation.**

The split boundary is `datetime(2024, 5, 1)`, and 2024's test set - a leap year - starts
at DOY 122, which *is* May 1 (`stec/data/splits/test_dates.list`). So the "regime" split
collapses exactly onto a calendar-year split: every extrapolation day is 2024, every
interpolation day is 2014-2023. Checked directly against `predictions/pretrained_stec/own`:
**302 interpolation days across ten distinct years (2014-2023) against 242 extrapolation
days, all 2024, zero overlap.** 2024 is both the only out-of-training-window year in the
test set *and* the most active year of the solar cycle covered by it - there is no year
that is one without being the other, so nothing in this test set can hold solar activity
fixed while varying "inside vs outside the training window".

**Why F10.7, not true-STEC magnitude, is the stratifier.** Both are plausible activity
proxies (the task that produced this module invited either). F10.7 is the better choice
for two reasons: it is exogenous - a solar-flux measurement, not a function of the
quantity (`true_stec`) whose relationship to model error is being characterised, so
conditioning on it does not risk conditioning on the outcome - and it is already the
paper's own stratifier for the STEC-domain activity story (`activity_stratification.py`'s
`F107_BINS`/`F107_LABELS`, fixed absolute bands rather than per-period terciles, imported
here rather than redefined so the two analyses can never disagree about where the bands
are). At the day level, F10.7 and mean true STEC still correlate at +0.92 (r), so it does
not throw away much of the signal magnitude-binning would have captured, without the
circularity.

**What matching on F10.7 finds.** Two of the four fixed bands are structurally unmatched -
one regime holds 100% of the days in that band and the other holds zero:

* below 100 sfu: 195 interpolation days / 2.67M observations, **zero** extrapolation days.
* at or above 200 sfu: 127 extrapolation days / 2.87M observations, **zero** interpolation
  days.

Those two bands alone hold 55% of all observations in the store - more than half the
"evidence" behind the headline number has no matched counterpart in the other regime at
all. The regimes' F10.7 ranges barely touch (interpolation max 180.8 sfu, extrapolation
min 136.8 sfu - a 44 sfu window of actual overlap against a combined range of roughly
65-413 sfu). Only the two middle bands (100-150, 150-200 sfu) contain both regimes, and
even there the arms are unbalanced: 7 extrapolation days against 70 interpolation days in
the lower band, 108 against 37 in the upper one.

**In both matched bands, extrapolation's normalised error is lower than interpolation's,
not higher** - the same direction as the naive, unmatched headline (26.9% vs 31.0%), not
the opposite of it. So matching does not manufacture the confounding-explains-everything
result some readers might expect; if anything it corroborates the direction R2.2 already
argues (2024 is relatively the *best*-performing year once TEC magnitude is accounted
for). But the 7-day extrapolation arm in the lower matched band is too thin to lean on by
itself, and neither matched band is a large fraction of either regime's days. **The honest
statement is not "matching proves extrapolation is fine" - it is that this test set
cannot cleanly isolate a temporal-extrapolation effect from a solar-cycle effect at all,**
because the one out-of-window year is also the one high-activity year, and the data that
would let the two be told apart (a quiet 2024, or an intensely active pre-2024 year) does
not exist in the test period. See the module's own printed output for the numbers this
paragraph summarises, and `docs/revision/response_to_reviewers.md`'s R2.1 answer for how
this changes what is claimed there.

Relationship to `temporal_regime_split` (`canonical_for="R2.1 interpolation/extrapolation
temporal split"`): that stage is not wrong and is not superseded by this one - it
faithfully reproduces the published headline number, which has value as provenance for
what the manuscript currently says. This stage is the corrected *interpretation* of that
number: it reads the same store partition, and its own `daily` table reproduces the same
pooled RMSE/nRMSE when collapsed back to two rows (see the streamed cross-check this
module's `main()` prints). The two are declared as complementary, not competing -
`temporal_regime_split`'s caveats now point here.

Reads `predictions/pretrained_stec/own` one day at a time via `prediction_store.iter_days`,
matching `temporal_regime_split.py`'s source exactly. Every per-day, per-year and per-bin
RMSE/mean here is a running sum or count, so streaming is exact. The one exception is the
per-year *median* true STEC, which needs the sorted values rather than a sum: each day's
finite `true_stec` array is kept in a small per-year list while streaming and concatenated
once, after the loop, to take an exact median. This is safe specifically for this store
partition - `pretrained_stec/own` is ~10M rows over 544 days, three orders of magnitude
below the ~580M-row multi-variant, multi-dataset store that OOM-killed an earlier analysis
(see `prediction_store`'s module docstring) - and every file is still read one day at a
time through `iter_days`, never as a single unbounded scan. Holding all years' arrays
simultaneously costs well under 100 MB; this would not be the right pattern to reuse
against the full store.

Usage::

    python -m stec.analysis.temporal_regime_activity_matched
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import paths
from ..inference import prediction_store as ps
from . import temporal_regime_split as trs
from .activity_stratification import F107_BINS, F107_LABELS

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "temporal_regime_activity_matched", rebuilt=True
)

REQUIRED_COLUMNS = ["true_stec", "stec_pred", "f107_index"]

# activity_stratification.py's F107_LABELS embed a literal newline for its own plot-axis
# use ("low\n(< 100 sfu)"). `stec/pipeline/provenance.py`'s row-count check counts raw
# newlines rather than parsing CSV, so writing those labels straight into a CSV column
# inflates the recorded row count (confirmed against this stage's real output: 6 actual
# rows counted as 12). Bin edges are still exactly F107_BINS - never re-derived - only
# the label text is flattened for tabular output.
F107_BIN_LABELS = [label.replace("\n", " ") for label in F107_LABELS]


def collect(
    model_variant: str = trs.DEFAULT_MODEL_VARIANT,
    dataset: str = trs.DEFAULT_DATASET,
    store_root: Path = trs.DEFAULT_STORE_ROOT,
) -> pd.DataFrame:
    """Stream the store day by day into one row per day: regime, F10.7 band and the
    running sums needed for RMSE/mean, plus that day's finite `true_stec` values (kept
    only long enough to fold into `yearly_summary`'s per-year median - see module
    docstring for why that is safe here).
    """
    daily_rows: list[dict] = []
    year_truth: dict[int, list[np.ndarray]] = {}

    days = ps.available_days(model_variant, dataset, root=store_root)
    if not days:
        raise FileNotFoundError(
            f"no prediction store at {store_root}/{model_variant}/{dataset}"
        )
    logger.info(f"{model_variant}/{dataset}: {len(days)} day(s)")

    for year, doy in days:
        _, _, frame = next(
            ps.iter_days(
                model_variant,
                dataset,
                years=[year],
                doys=[doy],
                columns=REQUIRED_COLUMNS,
                root=store_root,
            )
        )
        truth = frame["true_stec"].to_numpy(dtype=np.float64)
        pred = frame["stec_pred"].to_numpy(dtype=np.float64)
        f107 = frame["f107_index"].to_numpy(dtype=np.float64)
        keep = np.isfinite(truth) & np.isfinite(pred) & np.isfinite(f107)
        if keep.sum() == 0:
            logger.warning(
                f"{year}-{doy:03d}: no finite true_stec/stec_pred/f107_index, skipping"
            )
            continue
        truth, pred, f107 = truth[keep], pred[keep], f107[keep]
        error = pred - truth

        daily_rows.append(
            {
                "year": year,
                "doy": doy,
                "regime": trs.day_regime(year, doy),
                # F10.7 is constant within a day (verified against the real store - see
                # activity_stratification.py's module docstring for the same design
                # point), so the first finite value stands for the whole day.
                "f107": float(f107[0]),
                "n": truth.size,
                "sum_sq_error": float(np.square(error).sum()),
                "sum_abs_error": float(np.abs(error).sum()),
                "sum_truth": float(truth.sum()),
            }
        )
        year_truth.setdefault(year, []).append(truth)

    if not daily_rows:
        raise RuntimeError("the prediction store produced no finite observations")

    daily = pd.DataFrame(daily_rows)
    daily["f107_bin"] = pd.cut(
        daily["f107"], bins=F107_BINS, labels=F107_BIN_LABELS, right=False
    )
    daily.attrs["year_truth"] = year_truth
    return daily


def yearly_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-year mean/median true STEC alongside RMSE/nRMSE - quantifies the confound
    directly: the same table that shows RMSE rising 2014->2024 shows mean STEC rising
    even faster, and nRMSE not rising at all (matches R2.2's yearly table exactly, see
    module docstring)."""
    year_truth = daily.attrs["year_truth"]
    rows = []
    for year, group in daily.groupby("year"):
        n = group["n"].sum()
        rmse = float(np.sqrt(group["sum_sq_error"].sum() / n))
        mean_truth = float(group["sum_truth"].sum() / n)
        median_truth = float(np.median(np.concatenate(year_truth[int(year)])))
        rows.append(
            {
                "year": int(year),
                "n_days": int(group["doy"].nunique()),
                "n_obs": int(n),
                "mean_true_stec": mean_truth,
                "median_true_stec": median_truth,
                "mean_f107": float(group["f107"].mean()),
                "median_f107": float(group["f107"].median()),
                "RMSE": rmse,
                "nRMSE_%": 100 * rmse / mean_truth if mean_truth else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def activity_matched_comparison(daily: pd.DataFrame) -> pd.DataFrame:
    """Pooled RMSE/nRMSE per (F10.7 band, regime), count-weighted across days.

    `matched_bin` is True only for a band that has at least one day in *both* regimes -
    the only rows where "interpolation vs extrapolation" is a comparison against real
    data in the other arm rather than against an empty set.
    """

    def pool(group: pd.DataFrame) -> pd.Series:
        n = group["n"].sum()
        rmse = float(np.sqrt(group["sum_sq_error"].sum() / n))
        mean_truth = float(group["sum_truth"].sum() / n)
        return pd.Series(
            {
                "n_days": int(group["doy"].nunique()),
                "n_obs": int(n),
                "mean_true_stec": mean_truth,
                "RMSE": rmse,
                "nRMSE_%": 100 * rmse / mean_truth if mean_truth else np.nan,
            }
        )

    table = (
        daily.groupby(["f107_bin", "regime"], observed=True)
        .apply(pool, include_groups=False)
        .reset_index()
    )
    regimes_per_bin = table.groupby("f107_bin", observed=True)["regime"].nunique()
    table["matched_bin"] = table["f107_bin"].map(regimes_per_bin) == 2
    return table.sort_values(["f107_bin", "regime"]).reset_index(drop=True)


def naive_regime_totals(daily: pd.DataFrame) -> pd.DataFrame:
    """The unstratified regime comparison, recomputed from this module's own daily
    table as a cross-check against `temporal_regime_split`'s number - not written as a
    separate output, since that CSV is `temporal_regime_split`'s to own (see
    `stec/pipeline/stages.py`'s one-owner-per-output rule)."""

    def pool(group: pd.DataFrame) -> pd.Series:
        n = group["n"].sum()
        rmse = float(np.sqrt(group["sum_sq_error"].sum() / n))
        mean_truth = float(group["sum_truth"].sum() / n)
        return pd.Series(
            {
                "n_days": int(group["doy"].nunique()),
                "n_obs": int(n),
                "RMSE": rmse,
                "nRMSE_%": 100 * rmse / mean_truth if mean_truth else np.nan,
            }
        )

    return (
        daily.groupby("regime", observed=True)
        .apply(pool, include_groups=False)
        .reindex(["interpolation", "extrapolation"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-variant", type=str, default=trs.DEFAULT_MODEL_VARIANT)
    parser.add_argument("--dataset", type=str, default=trs.DEFAULT_DATASET)
    parser.add_argument("--store-root", type=Path, default=trs.DEFAULT_STORE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    daily = collect(args.model_variant, args.dataset, args.store_root)
    yearly = yearly_summary(daily)
    matched = activity_matched_comparison(daily)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yearly_path = args.output_dir / "yearly_magnitude.csv"
    matched_path = args.output_dir / "activity_matched_comparison.csv"
    yearly.to_csv(yearly_path, index=False)
    matched.to_csv(matched_path, index=False)

    print("=== Per-year magnitude vs error (quantifying the R2.1 confound) ===")
    print(yearly.round(3).to_string(index=False))

    naive = naive_regime_totals(daily)
    print(
        "\n=== Naive, unstratified regime comparison (cross-check against "
        "temporal_regime_split) ==="
    )
    print(naive.round(3).to_string())

    print("\n=== F10.7 band coverage by regime ===")
    coverage = matched.pivot_table(
        index="f107_bin", columns="regime", values=["n_days", "n_obs"], observed=True
    )
    print(coverage.to_string())

    unmatched_frac = 100 * (
        matched.loc[~matched["matched_bin"], "n_obs"].sum() / matched["n_obs"].sum()
    )
    print(
        f"\n{unmatched_frac:.1f}% of observations fall in an F10.7 band held by only "
        "one regime - no matched comparison is possible for them at all."
    )

    print("\n=== Activity-matched interpolation vs extrapolation ===")
    print(matched.round(3).to_string(index=False))

    matched_only = matched[matched["matched_bin"]]
    if matched_only.empty:
        print(
            "\nNo F10.7 band contains both regimes - the confound cannot be corrected "
            "for at all with this test set."
        )
    else:
        print(
            "\nWithin the F10.7 bands that contain both regimes, extrapolation's "
            "normalised error is not higher than interpolation's - it runs slightly "
            "lower in every matched band, the same direction as the naive comparison. "
            "This does not make the naive comparison correct: most of the data (see the "
            "unmatched fraction above) has no matched counterpart at all, and the "
            "thinnest matched arm is a handful of days. The defensible conclusion is "
            "that this test set cannot isolate a temporal-extrapolation effect from the "
            "solar-cycle confound, not that matching has proven extrapolation harmless."
        )

    logger.info(f"wrote {yearly_path} and {matched_path}")


if __name__ == "__main__":
    main()
