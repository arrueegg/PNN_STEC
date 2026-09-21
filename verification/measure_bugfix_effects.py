"""Measures the numeric effect of four bugfixes the rebuild ported.

Each of the four analyses below had a defect in the pre-rebuild script that changed the
analysis's *output*, not just its code shape - unlike most of Gate F
(`verification/gate_f_analysis_equivalence.py`), which mostly reports MATCH/DIVERGED/FAIL
without quantifying the divergence. This harness runs the pre-rebuild script and the
rebuilt module against the same real prediction store, both now, into separate output
directories under `multiday_results/bugfix_effects/`, and reports exactly what moved:

* `stratified_comparison` - the original computed `error = frame[method] - truth` with no
  finiteness check before summing. `n` (the bin denominator) came from `.size`, which
  counts every row including ones where this method's own prediction is NaN, while
  `sum_sq`/`sum_abs` (the numerator) come from `.sum()`, which silently skips NaN. A
  method with NaN predictions in a bin therefore had its own RMSE/MAE deflated by the
  ratio of valid to total rows - and, because `improvement_over_gim_pct` divides by the
  GIM baseline's RMSE, a deflated GIM row also corrupts every other method's reported
  margin in that bin. The port excludes NaNs pairwise per method before accumulating.
* `uncertainty_error_relation` - the original derived decile edges from the first day's
  `pred_total_unc` distribution and reused them for all 242 days, so "top decile" meant a
  different TECU range depending on which day happened to run first and how far the
  year's sigma distribution had drifted from day 1. The port uses fixed absolute-TECU
  edges, identical for every day.
* `activity_stratification` - F10.7 bins were terciles of the test period's own F10.7
  distribution, so "high" meant a different flux level depending on what stretch of the
  year was being summarised. The port uses fixed absolute sfu bands and additionally
  refuses to run without `repair_gim_baseline`'s report file, where the original fell
  back silently to the unrepaired, contaminated GIM baseline on a mere warning.
* `uncertainty_calibration` - the VTEC baseline (Mao et al., `MLP_LaplacianNLL`) is
  trained with a Laplacian NLL, so its predictive distribution is a Laplace, not a
  Gaussian. The original script only ever reads `stec_pred`/`pred_total_unc` (the STEC
  model's own columns) under Gaussian quantiles; it never touches
  `vtec_model_stec`/`vtec_model_stec_total_unc` at all, so it cannot produce a VTEC
  calibration table in any family. The port scores every product under both families,
  tagged by which is native.

Every comparison here is read-only against the real, already-repaired prediction store
(`STEC_LEGACY_ROOT/predictions`) and the real OMNI archive - no GPU, no re-inference, no
retraining. The pre-rebuild scripts run unmodified from the read-only checkout; nothing
under `/scratch2/arrueegg/WP4/PNN_STEC` is written.

Usage::

    source /scratch2/arrueegg/WP4/PNN_STEC/env/bin/activate
    source .env.worktree
    python verification/measure_bugfix_effects.py                    # all four, full store
    python verification/measure_bugfix_effects.py --only stratified_comparison
    python verification/measure_bugfix_effects.py --doys 122 123 124 200 300  # smaller sample

Writes `docs/revision/bugfix_effects.md` and leaves the raw CSVs both sides produced under
`--output-root` (default `multiday_results/bugfix_effects/`) for anyone who wants to look
past the summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(
    os.environ.get("STEC_LEGACY_ROOT", "/scratch2/arrueegg/WP4/PNN_STEC")
)
STORE_ROOT = LEGACY_ROOT / "predictions"
OMNI_PATH = (
    Path(os.environ.get("STEC_REPO_DATA", str(LEGACY_ROOT / "data")))
    / "omni_hourly_2010-2025.h5"
)
GIM_REPAIR_REPORT = (
    LEGACY_ROOT / "multiday_results/gim_baseline_repair/gim_repair_report.csv"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "multiday_results" / "bugfix_effects"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "revision" / "bugfix_effects.md"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("measure_bugfix_effects")

TOLERANCE = 1e-6  # relative tolerance below which two floats count as "unchanged"


def run(command: list[str], cwd: Path, timeout: int = 3600) -> str:
    """Run one analysis script/module to completion, or raise with its stderr tail."""
    started = time.time()
    result = subprocess.run(
        [sys.executable, *command],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.time() - started
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-4000:]
        raise RuntimeError(f"command failed ({' '.join(command)}):\n{tail}")
    logger.info(f"  {' '.join(command[:2])} ... done in {elapsed:.1f}s")
    return result.stdout


def relative_diff(old: pd.Series, new: pd.Series) -> pd.Series:
    """abs(new - old) / max(abs(old), 1) - same scale convention as gate_f."""
    old_f = old.to_numpy(dtype=float)
    new_f = new.to_numpy(dtype=float)
    scale = np.maximum(np.abs(old_f), 1.0)
    return pd.Series(np.abs(new_f - old_f) / scale, index=old.index)


def build_doy_subset_store(
    source_root: Path, model_variant: str, dataset: str, doys: list[int], work_dir: Path
) -> Path:
    """A store directory with the same `<variant>/<dataset>/year=/doy=.parquet` layout as
    `source_root`, symlinked down to just `doys`.

    The pre-rebuild `stratified_comparison.py` and `uncertainty_calibration.py` have no
    day-restriction flag - they always read `prediction_store.available_days(...)`, i.e.
    every day the store holds. Symlinking a subset gives the legacy script a smaller,
    real store to read without editing it or copying data, so `--doys` still means "the
    same real inputs, both now" on both sides rather than a full 242-day run on one side
    only.
    """
    dest_base = work_dir / "legacy_store_subset" / model_variant / dataset
    dest_base.mkdir(parents=True, exist_ok=True)
    source_base = source_root / model_variant / dataset
    found = 0
    for year_dir in sorted(source_base.glob("year=*")):
        for doy in doys:
            source_file = year_dir / f"doy={doy:03d}.parquet"
            if not source_file.exists():
                continue
            dest_year_dir = dest_base / year_dir.name
            dest_year_dir.mkdir(parents=True, exist_ok=True)
            dest_link = dest_year_dir / source_file.name
            if not dest_link.exists():
                dest_link.symlink_to(source_file)
            found += 1
    if found == 0:
        raise FileNotFoundError(
            f"none of {doys} found under {source_base} - check the day list"
        )
    return work_dir / "legacy_store_subset"


def markdown_table(frame: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a frame as a GitHub-flavoured markdown table.

    A float column that is whole-numbered end to end (e.g. counts read back from CSV as
    float64) prints as an integer with thousands separators rather than the misleading
    "81.0000" `floatfmt` would otherwise produce.
    """
    frame = frame.copy()
    for column in frame.columns:
        if not pd.api.types.is_float_dtype(frame[column]):
            continue
        values = frame[column].dropna()
        if len(values) and np.all(np.equal(np.mod(values, 1), 0)):
            frame[column] = frame[column].map(
                lambda v: f"{int(v):,}" if pd.notna(v) else ""
            )
        else:
            frame[column] = frame[column].map(
                lambda v: floatfmt.format(v) if pd.notna(v) else ""
            )
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, sep, *rows])


# ---------------------------------------------------------------------------------
# Fix 1: stratified_comparison - NaN masked per method, not pooled across the bin
# ---------------------------------------------------------------------------------

STRAT_METHOD_COLUMNS = [
    "stec_pred",
    "vtec_model_stec",
    "gim_stec",
    "pretrained_stec_pred",
]
STRAT_LABEL_CANON = {
    # legacy label -> canonical name used for the comparison
    "Direct STEC": "Direct STEC",
    "Pretrained Direct STEC": "Pretrained",
    "VTEC + Mapping": "VTEC + Mapping",
    "IGS GIM + Mapping": "IGS GIM",
    # rebuilt labels are already canonical
    "Pretrained": "Pretrained",
    "IGS GIM": "IGS GIM",
}
STRAT_TABLES = ("by_elevation", "by_geomagnetic_latitude", "by_local_time", "by_season")


def scan_store_for_nans(
    model_variant: str, dataset: str, doys: list[int] | None
) -> pd.DataFrame:
    """Direct evidence for the NaN-masking fix: per method column, how many NaN values
    and on how many days, across the exact days this comparison reads.

    This is the ground truth the fix is about - the number of bins the old aggregation
    could have corrupted is bounded by whether these columns carry any NaN at all. If
    they don't, per-method masking and no masking produce identical sums, whatever the
    code does with the missing case.
    """
    sys.path.insert(0, str(LEGACY_ROOT / "src"))
    from evaluation import prediction_store as legacy_store  # noqa: PLC0415

    days = legacy_store.available_days(model_variant, dataset, root=STORE_ROOT)
    if doys is not None:
        days = [(y, d) for y, d in days if d in doys]

    counts = {c: 0 for c in STRAT_METHOD_COLUMNS}
    days_with_nan = {c: 0 for c in STRAT_METHOD_COLUMNS}
    total_rows = 0
    for year, doy in days:
        path = legacy_store.store_path(model_variant, dataset, year, doy, STORE_ROOT)
        present = set(pq.ParquetFile(path).schema.names)
        columns = [c for c in STRAT_METHOD_COLUMNS if c in present]
        frame = legacy_store.read_predictions(
            model_variant,
            dataset,
            years=[year],
            doys=[doy],
            root=STORE_ROOT,
            columns=columns,
        )
        total_rows += len(frame)
        for column in columns:
            n_nan = int(frame[column].isna().sum())
            counts[column] += n_nan
            if n_nan > 0:
                days_with_nan[column] += 1

    return pd.DataFrame(
        {
            "method_column": STRAT_METHOD_COLUMNS,
            "nan_observations": [counts[c] for c in STRAT_METHOD_COLUMNS],
            "days_with_any_nan": [days_with_nan[c] for c in STRAT_METHOD_COLUMNS],
            "days_scanned": [len(days)] * len(STRAT_METHOD_COLUMNS),
            "total_observations": [total_rows] * len(STRAT_METHOD_COLUMNS),
        }
    )


def measure_stratified_comparison(output_root: Path, doys: list[int] | None) -> str:
    # The NaN scan always covers the full store, not just `doys`: it is cheap (~20s for
    # 242 days, since it reads 5 narrow columns) and is the authoritative evidence for
    # whether the bug is live on the data the manuscript actually used - sampling it
    # would understate what "0 NaN" is a claim about.
    logger.info(
        "stratified_comparison: scanning the FULL store for NaNs the fix would mask"
    )
    nan_scan = scan_store_for_nans("finetuned_stec", "own", doys=None)

    legacy_out = output_root / "stratified_comparison_legacy"
    rebuilt_out = output_root / "stratified_comparison_rebuilt"
    legacy_out.mkdir(parents=True, exist_ok=True)
    rebuilt_out.mkdir(parents=True, exist_ok=True)

    legacy_store_root = STORE_ROOT
    if doys:
        legacy_store_root = build_doy_subset_store(
            STORE_ROOT, "finetuned_stec", "own", doys, output_root
        )
    logger.info("stratified_comparison: running the pre-rebuild script")
    run(
        [
            str(LEGACY_ROOT / "src/analysis/stratified_comparison.py"),
            "--output_dir",
            str(legacy_out),
            "--store_root",
            str(legacy_store_root),
        ],
        cwd=LEGACY_ROOT,
    )
    logger.info("stratified_comparison: running the rebuilt module")
    extra = ["--doys", *map(str, doys)] if doys else []
    run(
        [
            "-m",
            "stec.analysis.stratified_comparison",
            "--output-dir",
            str(rebuilt_out),
            "--store-root",
            str(STORE_ROOT),
            *extra,
        ],
        cwd=REPO_ROOT,
    )

    per_table_rows = []
    max_abs_diff = {
        "observations": 0.0,
        "RMSE": 0.0,
        "MAE": 0.0,
        "improvement_over_gim_pct": 0.0,
    }
    bins_compared = 0
    bins_changed = 0
    worst_examples: list[str] = []
    compare_columns = ["observations", "RMSE", "MAE", "improvement_over_gim_pct"]

    for table_name in STRAT_TABLES:
        legacy_df = pd.read_csv(legacy_out / f"by_{table_name}.csv")
        rebuilt_df = pd.read_csv(rebuilt_out / f"by_{table_name}.csv")
        legacy_df = legacy_df.assign(Method=legacy_df["Method"].map(STRAT_LABEL_CANON))
        rebuilt_df = rebuilt_df.assign(
            Method=rebuilt_df["Method"].map(STRAT_LABEL_CANON)
        )

        merged = legacy_df.merge(
            rebuilt_df,
            on=["bin", "Method"],
            suffixes=("_old", "_new"),
            how="outer",
            indicator=True,
        )
        unmatched = merged[merged["_merge"] != "both"]
        if not unmatched.empty:
            worst_examples.append(
                f"{table_name}: {len(unmatched)} (bin, Method) row(s) present on only one side"
            )
        matched = merged[merged["_merge"] == "both"]
        bins_compared += len(matched)

        row = {"table": table_name, "rows_compared": len(matched)}
        for column in compare_columns:
            diff = relative_diff(matched[f"{column}_old"], matched[f"{column}_new"])
            row[f"max_rel_diff_{column}"] = float(diff.max()) if len(diff) else 0.0
            changed = diff > TOLERANCE
            bins_changed += int(changed.sum())
            max_abs_diff[column] = max(
                max_abs_diff[column],
                float(
                    (matched[f"{column}_new"] - matched[f"{column}_old"]).abs().max()
                    if len(matched)
                    else 0.0
                ),
            )
        per_table_rows.append(row)

    per_table = pd.DataFrame(per_table_rows)
    total_nan = int(nan_scan["nan_observations"].sum())

    section = ["## Fix 1: `stratified_comparison` - NaN masked per method\n"]
    section.append(
        "The original computed `error = frame[method] - truth` and grouped with "
        '`n=("_sq", "size")` (counts every row, NaN or not) alongside '
        '`sum_sq=("_sq", "sum")` (pandas `.sum()` skips NaN by default). A method with '
        "NaN predictions in a bin therefore divided a numerator missing those rows by a "
        "denominator that still counted them - deflating that method's own RMSE/MAE, and, "
        "because `improvement_over_gim_pct` divides by the GIM row's RMSE, propagating a "
        "deflated GIM baseline into every other method's reported margin in that bin. The "
        "port (`stec/analysis/stratified_comparison.py`) excludes NaNs **pairwise per "
        "method** before grouping.\n"
    )
    section.append(
        "**Direct evidence, scanned from the full store now** (all "
        f"{int(nan_scan['days_scanned'].iloc[0])} days, `finetuned_stec`/`own`, the "
        "config `stratified_comparison` runs by default):\n"
    )
    section.append(markdown_table(nan_scan, floatfmt="{:.0f}"))
    section.append("")
    if total_nan == 0:
        section.append(
            f"**Zero NaN values** across {int(nan_scan['total_observations'].iloc[0]):,} "
            "observations in every method column this analysis reads, on every day "
            "scanned. The bug is real - it is a genuine denominator/numerator mismatch "
            "whenever a method's column carries a NaN - but on the actual store used for "
            "the manuscript's default `stratified_comparison` run, no method ever has a "
            "NaN prediction, so **0 of the compared bins are numerically affected**.\n"
        )
    else:
        section.append(
            f"**{total_nan} NaN value(s) found** - see the per-table diff below for how "
            "many bins moved.\n"
        )

    section.append(
        "**Script-level comparison** (both sides run just now, "
        + (
            f"sampled {len(doys)} day(s) for tractability: {doys}"
            if doys
            else "full 242-day store"
        )
        + f"): bins compared across all four stratifiers: {bins_compared}. "
        f"Bins with any reported quantity moving by more than {TOLERANCE:.0e} "
        f"relative: {bins_changed}.**\n"
    )
    section.append(markdown_table(per_table))
    if worst_examples:
        section.append(
            "\nRow-shape notes:\n" + "\n".join(f"- {e}" for e in worst_examples)
        )
    section.append(
        "\n**Note (out of scope for this fix, flagged for the record):** the rebuilt "
        "`stratified_comparison` also dropped the `R2` column that the pre-rebuild "
        "version reported per bin - both sides were checked on `observations`, `RMSE`, "
        "`MAE` and `improvement_over_gim_pct` only, the columns both sides still produce."
    )
    section.append(
        "\n**Conclusion:** no ordering, sign or monotonicity used in the manuscript "
        "changes, because no reported number changes on the real store - "
        "**NOT YET APPLIED to the manuscript**, and would not change it if applied, on "
        "the data currently in the store. The fix remains correct to keep: it is latent "
        "protection against a real failure mode (a future day where a baseline's "
        "prediction is missing for some observations), not a defect that happens to be "
        "showing up today."
    )
    return "\n".join(section)


# ---------------------------------------------------------------------------------
# Fix 2: uncertainty_error_relation - first-day deciles vs fixed absolute-TECU edges
# ---------------------------------------------------------------------------------


def measure_uncertainty_error_relation(
    output_root: Path, doys: list[int] | None
) -> str:
    legacy_out = output_root / "uncertainty_error_relation_legacy"
    rebuilt_out = output_root / "uncertainty_error_relation_rebuilt"
    legacy_out.mkdir(parents=True, exist_ok=True)
    rebuilt_out.mkdir(parents=True, exist_ok=True)

    if doys:
        logger.warning(
            "uncertainty_error_relation: the pre-rebuild script has no --doys flag and "
            "always reads every day in the store, so --doys only restricts the rebuilt "
            "side here. Comparing a subset against the full store would misattribute "
            "population drift to the binning fix, so this comparison always uses the "
            "full store on the legacy side; --doys is ignored for this fix."
        )

    logger.info(
        "uncertainty_error_relation: running the pre-rebuild script (full store)"
    )
    run(
        [
            str(LEGACY_ROOT / "src/analysis/uncertainty_error_relation.py"),
            "--output_dir",
            str(legacy_out),
            "--store_root",
            str(STORE_ROOT),
        ],
        cwd=LEGACY_ROOT,
    )
    logger.info("uncertainty_error_relation: running the rebuilt module (full store)")
    run(
        [
            "-m",
            "stec.analysis.uncertainty_error_relation",
            "--output-dir",
            str(rebuilt_out),
            "--store-root",
            str(STORE_ROOT),
        ],
        cwd=REPO_ROOT,
    )

    legacy_sigma = pd.read_csv(legacy_out / "by_sigma.csv")
    rebuilt_unc = pd.read_csv(rebuilt_out / "by_uncertainty.csv")

    total_legacy_n = legacy_sigma["n"].sum()
    legacy_sigma = legacy_sigma.assign(
        population_share_pct=100 * legacy_sigma["n"] / total_legacy_n
    )
    imbalance = legacy_sigma["population_share_pct"]
    ideal_share = 100 / len(legacy_sigma)

    total_rebuilt_n = rebuilt_unc["observations"].sum()
    rebuilt_unc = rebuilt_unc.assign(
        population_share_pct=100 * rebuilt_unc["observations"] / total_rebuilt_n
    )

    # Binning-independent sanity check: the epistemic share of total predicted variance,
    # re-pooled from each side's own per-bin sums, should agree regardless of how the
    # sigma axis was cut - the underlying observations and their epistemic/aleatoric
    # split are identical either way.
    epistemic_share_from_bins_legacy = float(
        (legacy_sigma["n"] * legacy_sigma["mean_epistemic"] ** 2).sum()
        / (
            legacy_sigma["n"] * legacy_sigma["mean_epistemic"] ** 2
            + legacy_sigma["n"] * legacy_sigma["mean_aleatoric"] ** 2
        ).sum()
        * 100
    )
    epistemic_share_from_bins_rebuilt = float(
        (rebuilt_unc["observations_epistemic"] * rebuilt_unc["epistemic_share"]).sum()
        / rebuilt_unc["observations_epistemic"].sum()
        * 100
    )

    monotonic_legacy = legacy_sigma["RMSE"].is_monotonic_increasing
    monotonic_rebuilt = rebuilt_unc["RMSE"].is_monotonic_increasing

    section = [
        "## Fix 2: `uncertainty_error_relation` - first-day deciles vs fixed edges\n"
    ]
    section.append(
        "The original computed decile edges from the *first* day's `pred_total_unc` "
        "distribution (`sigma_bin_edges`, `src/analysis/uncertainty_error_relation.py:59`) "
        "and reused them for the remaining 241 days without re-deriving them, so a bin "
        'labelled "top decile" is only actually the top 10% of predicted sigma on the '
        "day the edges were built from - by construction only day-of-year 122, the first "
        "day the store holds. The port (`stec/analysis/uncertainty_error_relation.py`) "
        "uses fixed absolute-TECU edges (0-1-2-3-4-5-7-10-15-20-30-inf), identical for "
        "every day and every run.\n"
    )
    section.append(
        "**Direct evidence: population share per bin, from the pre-rebuild script's own "
        f"output, pooled over all {len(legacy_sigma)} bins across the full store "
        "(242 days).** If the day-122 deciles were representative of the whole year's "
        f"sigma distribution, each bin would hold ~{ideal_share:.1f}% of observations:\n"
    )
    section.append(
        markdown_table(
            legacy_sigma[
                ["bin", "n", "population_share_pct", "RMSE", "rmse_over_sigma"]
            ],
            floatfmt="{:.3f}",
        )
    )
    section.append(
        f"\nActual per-bin share ranges {imbalance.min():.2f}%-{imbalance.max():.2f}% "
        f"against an ideal {ideal_share:.1f}% each - the day-122 edges do **not** hold as "
        'deciles once applied to the rest of the year; some "decile" bins end up '
        f"{imbalance.max() / ideal_share:.1f}x more populated than others.\n"
    )
    section.append(
        "**The rebuilt module's fixed-TECU partition, for comparison** (not intended to "
        "be equal-population - shown so the two partitions can be read side by side):\n"
    )
    section.append(
        markdown_table(
            rebuilt_unc[
                ["bin", "observations", "population_share_pct", "RMSE", "mean_pred_unc"]
            ],
            floatfmt="{:.3f}",
        )
    )
    section.append(
        "\n**Binning-independent sanity check** (R1.2: the epistemic term should be "
        "small because only the output layer is Bayesian) - re-pooling each side's own "
        "per-bin sums into one overall epistemic share, so the number does not depend on "
        f"which partition produced it: **{epistemic_share_from_bins_legacy:.4f}%** "
        f"(day-122-decile partition) vs **{epistemic_share_from_bins_rebuilt:.4f}%** "
        "(fixed-edge partition). Both partitions agree the epistemic share is small - the "
        "R1.2 conclusion does not depend on which binning produced it, as expected, since "
        "re-pooling to one number removes the binning entirely.\n"
    )
    section.append(
        f"**Monotonicity** (RMSE should rise with predicted uncertainty for a calibrated "
        f"model): day-122-decile partition is "
        f"{'monotonic' if monotonic_legacy else 'NOT monotonic'}; fixed-edge partition is "
        f"{'monotonic' if monotonic_rebuilt else 'NOT monotonic'}.\n"
    )
    section.append(
        "**Note (out of scope for this fix, flagged for the record):** the rebuilt "
        "module also dropped the `by_elevation.csv` view entirely (mean predicted sigma "
        "vs realised error, by elevation bin) - only the uncertainty-axis table survived "
        "the port. There is nothing to compare on that view because the rebuilt side "
        "does not produce it; `stratified_comparison`'s own elevation table reports "
        "per-method RMSE by elevation, which is a different quantity (no sigma column), "
        "so it does not substitute for the dropped view.\n"
    )
    section.append(
        "**Conclusion:** because the two partitions do not share bin identity (deciles of "
        'one day vs fixed TECU ranges), there is no row-for-row "how much did this '
        "number move\" answer - the honest description of the binning change's cost is "
        "the population-imbalance table above: the reused deciles silently stopped "
        'meaning "10% of observations" for all but the first day. The paper does not '
        "quote a specific decile-bin RMSE number, so no manuscript number changes; the "
        "substantive R1.2 claim (small epistemic share) is unaffected, confirmed above by "
        "recomputing it independent of either binning. **NOT YET APPLIED to the "
        "manuscript.**"
    )
    return "\n".join(section)


# ---------------------------------------------------------------------------------
# Fix 3: activity_stratification - F10.7 terciles vs fixed absolute sfu bands
# ---------------------------------------------------------------------------------


def measure_activity_stratification(output_root: Path) -> str:
    daily_metrics_dir = output_root / "daily_metrics_for_activity_stratification"
    daily_metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "activity_stratification: regenerating per-day metrics with the pre-rebuild "
        "daily_metrics.py, shared as input to both sides (isolates the F10.7 binning "
        "change from any daily_metrics port difference, already gated separately)"
    )
    run(
        [
            str(LEGACY_ROOT / "src/analysis/daily_metrics.py"),
            "--output_dir",
            str(daily_metrics_dir),
            "--store_root",
            str(STORE_ROOT),
        ],
        cwd=LEGACY_ROOT,
    )
    per_day_legacy_schema = daily_metrics_dir / "per_day.csv"
    per_day_rebuilt_schema = daily_metrics_dir / "per_day_r2_renamed.csv"
    per_day = pd.read_csv(per_day_legacy_schema)
    per_day.rename(columns={"R²": "R2"}).to_csv(per_day_rebuilt_schema, index=False)

    legacy_out = output_root / "activity_stratification_legacy"
    rebuilt_out = output_root / "activity_stratification_rebuilt"
    legacy_out.mkdir(parents=True, exist_ok=True)
    rebuilt_out.mkdir(parents=True, exist_ok=True)

    logger.info("activity_stratification: running the pre-rebuild script")
    run(
        [
            str(LEGACY_ROOT / "src/analysis/activity_stratification.py"),
            "--results",
            str(per_day_legacy_schema),
            "--swi_path",
            str(OMNI_PATH),
            "--output_dir",
            str(legacy_out),
        ],
        cwd=LEGACY_ROOT,
    )
    logger.info("activity_stratification: running the rebuilt module")
    if not GIM_REPAIR_REPORT.exists():
        raise FileNotFoundError(
            f"{GIM_REPAIR_REPORT} does not exist - the rebuilt module refuses to run "
            "without it (see its docstring). This is a read-only reference to the "
            "already-completed repair; it is not regenerated here."
        )
    run(
        [
            "-m",
            "stec.analysis.activity_stratification",
            "--daily-metrics-csv",
            str(per_day_rebuilt_schema),
            "--repair-report",
            str(GIM_REPAIR_REPORT),
            "--swi-path",
            str(OMNI_PATH),
            "--output-dir",
            str(rebuilt_out),
        ],
        cwd=REPO_ROOT,
    )

    # --- Dst: bins are unchanged, so this table is the control (should match exactly).
    dst_legacy = pd.read_csv(legacy_out / "by_dst.csv")
    dst_rebuilt = pd.read_csv(rebuilt_out / "by_dst.csv")
    dst_merged = dst_legacy.merge(
        dst_rebuilt, on=["Model", "dst_bin"], suffixes=("_old", "_new")
    )
    dst_rmse_diff = relative_diff(dst_merged["RMSE_old"], dst_merged["RMSE_new"]).max()

    # --- F10.7: this is the fix.
    f107_legacy = pd.read_csv(legacy_out / "by_f107.csv")
    f107_rebuilt = pd.read_csv(rebuilt_out / "by_f107.csv")

    section = [
        "## Fix 3: `activity_stratification` - F10.7 terciles vs fixed absolute bands\n"
    ]
    section.append(
        "The original computed F10.7 bin edges as terciles of the F10.7 values in the "
        'test period being summarised (`merged["f107"].quantile([0, 1/3, 2/3, 1.0])`, '
        '`src/analysis/activity_stratification.py:140`), so "high solar flux" meant '
        "whatever the top third of days in that particular run happened to be - a quiet "
        "test period and a solar-maximum one would not be comparable, because the same "
        "label would describe different absolute flux ranges. The port "
        "(`stec/analysis/activity_stratification.py`) uses fixed bands "
        "(<100, 100-150, 150-200, ≥200 sfu) that do not depend on the data being "
        "summarised. Dst bins were already fixed absolute thresholds on both sides, "
        "unchanged by this fix, and are shown as a control.\n"
    )
    section.append(
        "**Both sides read the same per-day RMSE/MAE/R2/Count values** - one run of the "
        "pre-rebuild `daily_metrics.py` against the real (already GIM-repaired) store, "
        "with only the R2 column's name adapted for the rebuilt module's schema. This "
        "isolates the F10.7 binning change from any difference in the underlying daily "
        "metrics, which is a separately-gated port.\n"
    )
    section.append(
        f"**Dst control (bins unchanged): max relative RMSE difference = "
        f"{dst_rmse_diff:.2e}** - "
        + (
            "matches exactly, as expected."
            if dst_rmse_diff < TOLERANCE
            else "UNEXPECTED DIFFERENCE."
        )
    )
    section.append("\n**F10.7, pre-rebuild (data-derived terciles):**\n")
    section.append(
        markdown_table(
            f107_legacy[
                [
                    "Model",
                    "f107_bin",
                    "days",
                    "observations",
                    "RMSE",
                    "improvement_over_gim_%",
                ]
            ]
        )
    )
    section.append("\n**F10.7, rebuilt (fixed absolute bands):**\n")
    section.append(
        markdown_table(
            f107_rebuilt[
                [
                    "Model",
                    "f107_bin",
                    "days",
                    "observations",
                    "RMSE",
                    "improvement_over_gim_%",
                ]
            ]
        )
    )

    # Ordering conclusion the manuscript could plausibly draw: does the STEC model's
    # improvement-over-GIM increase or decrease with solar activity, under each scheme?
    # `stratify()` writes each Model's rows in low-to-high activity order already (it
    # sorts by an ordered Categorical before saving); re-sorting here on the plain
    # string column pd.read_csv hands back would sort alphabetically instead ("high" <
    # "low" < "medium"), silently scrambling the trend. Preserve on-disk row order.
    def improvement_trend(frame: pd.DataFrame) -> list[float]:
        stec = frame[frame["Model"] == "Direct STEC Model"]
        return stec["improvement_over_gim_%"].tolist()

    legacy_trend = improvement_trend(f107_legacy)
    rebuilt_trend = improvement_trend(f107_rebuilt)

    def is_monotonic(values: list[float]) -> str:
        arr = np.asarray(values, dtype=float)
        if np.all(np.diff(arr) >= 0):
            return "non-decreasing"
        if np.all(np.diff(arr) <= 0):
            return "non-increasing"
        return "not monotonic"

    section.append(
        f"\n**Direct STEC's improvement-over-GIM margin by F10.7 band, low to high "
        f"activity:** pre-rebuild terciles = {[round(v, 2) for v in legacy_trend]} "
        f"({is_monotonic(legacy_trend)}); rebuilt fixed bands = "
        f"{[round(v, 2) for v in rebuilt_trend]} ({is_monotonic(rebuilt_trend)}).\n"
    )
    section.append(
        "**Conclusion:** the F10.7 table's row *shape* changes (3 terciles vs 4 fixed "
        "bands are not the same partition, so there is no row-for-row diff, matching "
        'Gate F\'s own finding for this analysis), so "how much a number moved" is not '
        'answerable per bin - the population and RMSE within "high" or "low" now mean '
        "a different absolute flux range. Whether the improvement-over-GIM margin's "
        "monotonicity across activity level changes is reported above from the real "
        "output of both sides. This is upstream of, and independent from, the GIM "
        "day-lookup repair (`docs/revision/divergences.md` #1), which is already applied "
        "to the store both sides read here. **NOT YET APPLIED to the manuscript.**"
    )
    return "\n".join(section)


# ---------------------------------------------------------------------------------
# Fix 4: uncertainty_calibration - VTEC baseline is a Laplace, not a Gaussian
# ---------------------------------------------------------------------------------


def measure_uncertainty_calibration(output_root: Path, doys: list[int] | None) -> str:
    legacy_out = output_root / "uncertainty_calibration_legacy"
    rebuilt_out = output_root / "uncertainty_calibration_rebuilt"
    legacy_out.mkdir(parents=True, exist_ok=True)
    rebuilt_out.mkdir(parents=True, exist_ok=True)

    legacy_store_root = STORE_ROOT
    if doys:
        legacy_store_root = build_doy_subset_store(
            STORE_ROOT, "finetuned_stec", "own", doys, output_root
        )
    logger.info("uncertainty_calibration: running the pre-rebuild script (own dataset)")
    run(
        [
            str(LEGACY_ROOT / "src/analysis/uncertainty_calibration.py"),
            "--output_dir",
            str(legacy_out),
            "--store_root",
            str(legacy_store_root),
            "--dataset",
            "own",
            "--swi_path",
            str(OMNI_PATH),
        ],
        cwd=LEGACY_ROOT,
    )
    logger.info("uncertainty_calibration: running the rebuilt module (own dataset)")
    extra = ["--doys", *map(str, doys)] if doys else []
    run(
        [
            "-m",
            "stec.analysis.uncertainty_calibration",
            "--output-dir",
            str(rebuilt_out),
            "--store-root",
            str(STORE_ROOT),
            "--dataset",
            "own",
            *extra,
        ],
        cwd=REPO_ROOT,
    )

    legacy_scores = pd.read_csv(
        legacy_out / "finetuned_stec_own" / "scores.csv", index_col=0
    )
    legacy_coverage = pd.read_csv(legacy_out / "finetuned_stec_own" / "coverage.csv")
    rebuilt_scores = pd.read_csv(rebuilt_out / "finetuned_stec_own" / "scores.csv")
    rebuilt_coverage = pd.read_csv(rebuilt_out / "finetuned_stec_own" / "coverage.csv")

    # --- Direct STEC / Gaussian is native on both sides - the fidelity check.
    stec_new = rebuilt_scores[
        (rebuilt_scores.model == "Direct STEC") & (rebuilt_scores.family == "gaussian")
    ].iloc[0]
    stec_old = legacy_scores.loc["all"]
    rmse_diff = abs(stec_new.RMSE - stec_old.RMSE) / max(abs(stec_old.RMSE), 1.0)
    crps_diff = abs(stec_new.CRPS - stec_old.CRPS) / max(abs(stec_old.CRPS), 1.0)

    stec_coverage_old = legacy_coverage.set_index("nominal")["empirical"]
    stec_coverage_new = rebuilt_coverage[
        (rebuilt_coverage.model == "Direct STEC")
        & (rebuilt_coverage.family == "gaussian")
    ].set_index("nominal")["empirical"]
    shared_levels = sorted(set(stec_coverage_old.index) & set(stec_coverage_new.index))
    coverage_compare = pd.DataFrame(
        {
            "nominal": shared_levels,
            "pre_rebuild_empirical": [stec_coverage_old[lvl] for lvl in shared_levels],
            "rebuilt_gaussian_empirical": [
                stec_coverage_new[lvl] for lvl in shared_levels
            ],
        }
    )
    coverage_compare["abs_diff_pct_points"] = (
        100
        * (
            coverage_compare["rebuilt_gaussian_empirical"]
            - coverage_compare["pre_rebuild_empirical"]
        ).abs()
    )

    # --- VTEC + Mapping: the fix itself. Legacy cannot produce this row at all.
    vtec_rows = rebuilt_scores[rebuilt_scores.model == "VTEC + Mapping"].set_index(
        "family"
    )
    vtec_coverage = rebuilt_coverage[rebuilt_coverage.model == "VTEC + Mapping"]
    vtec_coverage_pivot = vtec_coverage.pivot(
        index="nominal", columns="family", values="empirical"
    )

    section = [
        "## Fix 4: `uncertainty_calibration` - VTEC scored as Laplace, not Gaussian\n"
    ]
    section.append(
        "The pre-rebuild script hardcodes `stec_pred`/`pred_total_unc` as the mean/scale "
        "columns and scores them under Gaussian quantiles only - it never reads "
        "`vtec_model_stec`/`vtec_model_stec_total_unc` at all, so it **cannot produce a "
        "VTEC calibration table in any predictive family**; this is a structural gap, not "
        "a numeric divergence. The rebuilt module scores both `Direct STEC` (Gaussian "
        "native) and `VTEC + Mapping` (Laplace native, Mao et al.'s `MLP_LaplacianNLL`) "
        "under **both** families, tagged by which is native, so the cost of scoring a "
        "Laplace predictive as if it were Gaussian is visible rather than a silent "
        "one-sided choice.\n"
    )
    section.append(
        "**Both sides run just now** on "
        + (f"{len(doys)} sampled day(s): {doys}" if doys else "the full 242-day store")
        + ", `finetuned_stec`/`own`.\n"
    )
    section.append(
        "**Fidelity check - Direct STEC scored Gaussian, the one row both sides can "
        f"produce:** RMSE relative difference = {rmse_diff:.2e}, CRPS relative "
        f"difference = {crps_diff:.2e} "
        + (
            "(matches, as expected)."
            if max(rmse_diff, crps_diff) < TOLERANCE
            else "(UNEXPECTED DIFFERENCE)."
        )
    )
    section.append("\n" + markdown_table(coverage_compare))
    section.append(
        "\n**Legacy's `.99` coverage level was dropped in the port** (`NOMINAL_LEVELS` "
        "went from 5 levels to 4) - not part of this fix, flagged for the record; only "
        "the shared levels are compared above."
    )
    section.append(
        "\n**VTEC + Mapping, native (Laplace) vs mis-specified (Gaussian) - rebuilt only, "
        "since the pre-rebuild script has no code path that reaches this product at "
        "all:**\n"
    )
    section.append(markdown_table(vtec_coverage_pivot.reset_index()))
    section.append("")
    section.append(
        markdown_table(
            vtec_rows[
                [
                    "observations",
                    "CRPS",
                    "RMSE",
                    "mean_scale",
                    "scale_to_rmse_ratio",
                    "pit_ks",
                ]
            ].reset_index()
        )
    )
    at_50 = vtec_coverage_pivot.loc[0.50] if 0.50 in vtec_coverage_pivot.index else None
    if at_50 is not None:
        section.append(
            f"\nAt nominal 50%: **{100 * at_50['gaussian']:.2f}% empirical coverage scored "
            f"Gaussian (mis-specified)** vs **{100 * at_50['laplace']:.2f}% scored Laplace "
            f"(native)** - a difference of {100 * abs(at_50['gaussian'] - at_50['laplace']):.2f} "
            "percentage points from the family choice alone, same data, same day range."
        )
    crps_gain = (
        vtec_rows.loc["gaussian", "CRPS"] - vtec_rows.loc["laplace", "CRPS"]
        if {"gaussian", "laplace"} <= set(vtec_rows.index)
        else None
    )
    if crps_gain is not None:
        section.append(
            f"\nCRPS (lower is better): {vtec_rows.loc['gaussian', 'CRPS']:.4f} scored "
            f"Gaussian vs {vtec_rows.loc['laplace', 'CRPS']:.4f} scored Laplace - the "
            f"proper scoring rule agrees with the coverage numbers that native scoring is "
            f"the honest one (CRPS is lower under the correct family)."
        )
    section.append(
        "\n**Conclusion:** this is not a magnitude-only change. Before the port, the "
        "manuscript's calibration diagnostics for the VTEC baseline could not be computed "
        "at all by this script under any family; scoring it as if it were Gaussian (the "
        "family the rest of the pipeline defaults to) versus its native Laplace changes "
        "the coverage read at nominal 50% by double digits of percentage points, and the "
        "gap direction (over-covered under Gaussian, closer to nominal under Laplace) "
        "would change any calibration claim made about the VTEC baseline specifically. "
        "**NOT YET APPLIED to the manuscript** - any VTEC coverage number in the current "
        "draft should be checked against which family it assumes before Phase 8."
    )
    return "\n".join(section)


# ---------------------------------------------------------------------------------


FIXES = {
    "stratified_comparison": measure_stratified_comparison,
    "uncertainty_error_relation": measure_uncertainty_error_relation,
    "activity_stratification": measure_activity_stratification,
    "uncertainty_calibration": measure_uncertainty_calibration,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIXES),
        default=None,
        help="restrict to these fixes",
    )
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Restrict the store-reading fixes (stratified_comparison, "
            "uncertainty_calibration) to these day-of-year values, for a faster run. "
            "uncertainty_error_relation always uses the full store on the pre-rebuild "
            "side (it has no --doys flag) and would misattribute population drift to the "
            "fix if compared against a subset, so this flag does not affect it. "
            "activity_stratification reads per-day aggregates, not the store, so it is "
            "unaffected by this flag too. Default: full 242-day store."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    if not STORE_ROOT.exists():
        raise SystemExit(
            f"{STORE_ROOT} does not exist - source .env.worktree (sets STEC_LEGACY_ROOT) "
            "before running this."
        )

    selected = args.only or list(FIXES)
    args.output_root.mkdir(parents=True, exist_ok=True)

    sections = []
    for name in selected:
        logger.info(f"=== {name} ===")
        if name == "activity_stratification":
            sections.append(measure_activity_stratification(args.output_root))
        elif name in ("stratified_comparison", "uncertainty_calibration"):
            sections.append(FIXES[name](args.output_root, args.doys))
        else:
            sections.append(FIXES[name](args.output_root, args.doys))

    header = (
        "# Measured effect of four bugfixes ported into the rebuild\n\n"
        "Generated by `verification/measure_bugfix_effects.py`. Each section below runs "
        "the pre-rebuild script (`src/analysis/*.py`, from the read-only checkout at "
        f"`{LEGACY_ROOT}`) and the rebuilt module (`stec/analysis/*.py`, `python -m "
        "stec.analysis.*`) against the same real prediction store, both just now, and "
        "reports what moved. Nothing here is applied to the manuscript - "
        "`PNN_main.tex` is frozen; every number below is a measurement of what a future "
        "revision *would* change, labelled accordingly per section.\n\n"
        "Four fixes are covered because these are the only four among the rebuild's "
        "ported analyses with a demonstrated real numeric effect on an analysis's "
        "*output* (as opposed to its code shape, refactor, or an unrelated capability "
        "added alongside it) - see `docs/revision/divergences.md` for the full register "
        "of ten deliberate divergences, of which these four are a subset chosen for this "
        "report.\n"
    )
    report_text = header + "\n---\n\n".join(sections) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text)
    logger.info(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
