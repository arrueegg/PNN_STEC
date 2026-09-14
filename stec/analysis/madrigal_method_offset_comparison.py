"""Does VTEC + Mapping's Madrigal win over Direct STEC survive the reference-offset fix?

Evidence for reviewer comment R1.3:

    "The comparison mixes TEC/STEC products with potentially inconsistent bias
     references ... the reported RMSE/MAE values may conflate model error with
     reference-product inconsistency."

`madrigal_reference_offset.py` already showed that 45% of Direct STEC's Madrigal RMSE
variance is a per-station reference offset - but it only ever decomposed Direct STEC (and,
in passing, the IGS GIM, to show the two agree with each other). On Madrigal, VTEC +
Mapping currently *beats* Direct STEC on all three headline metrics (Tables 3/4's
`daily_metrics` summary: RMSE 13.60 vs 14.63, MAE 8.27 vs 8.79, R2 0.87 vs 0.85), which
reverses the own-test-set result where Direct STEC wins by 23%. Nobody has checked whether
that reversal is a real generalisation advantage or whether it is carried by the same
reference inconsistency `madrigal_reference_offset.py` found in Direct STEC's number - the
question this module exists to answer, for all four methods on identical rows rather than
for one method in isolation.

Method: the same per-station-offset idea, generalised to all four methods, computed on the
row set every method has a finite prediction for (see `row_intersection_diagnostics.csv`
for whether that set is actually smaller than the full store). If the four methods' offset
vectors are highly correlated across the 67-odd Madrigal stations, the offset is a property
of the Madrigal reference and the RMSE ranking is meaningful only after removing it. If
Direct STEC's offset vector is distinctly larger or differently shaped from the other
three, the reversal reflects the model, not the reference.

**Removing a per-station offset fitted on the same data can only ever reduce RMSE.** With
67 stations against ~449 M observations the optimism this introduces is negligible (roughly
67 degrees of freedom against hundreds of millions), but it is not zero, and every reported
"after" number in this module is fit-then-scored on the same data for exactly that reason -
stated here rather than left for a reader to wonder about.

**Result (2026-09-14, full 238-day store, 448,938,780 rows, all four methods finite on
every one of them, 67 qualifying stations):** the offset is common-mode. All six pairwise
Pearson correlations between the methods' offset vectors exceed 0.92 (Direct STEC vs IGS
GIM: 0.925, matching `madrigal_reference_offset.py`'s own +0.946-ish finding for that same
pair to within sampling), and the ranking changes once the offset is removed. Before:
VTEC + Mapping (13.90) < Direct STEC (15.01) < IGS GIM (15.73) < Pretrained (17.98). After:
IGS GIM (10.25) < **Direct STEC (11.03) < VTEC + Mapping (11.66)** < Pretrained (15.26) -
Direct STEC moves back ahead of VTEC + Mapping on both RMSE and MAE, matching the
own-test-set ranking, once the same per-station correction `madrigal_reference_offset.py`
already applies to Direct STEC alone is applied to all four. VTEC + Mapping's Madrigal win
is real in the raw numbers but is carried by having the *smallest* reference offset to
begin with (mean |offset| 4.54 TECU against 6.06-8.76 TECU for the other three), not by a
genuine accuracy advantage that survives a fair reference.

Two streaming passes over `predictions/finetuned_stec/madrigal`, matching
`madrigal_reference_offset.py`'s design: pass 1 computes the per-station offsets (needed
before anything can be corrected) and, on the way, the full-population pairwise pooled RMSE
per method - the correctness check against `daily_metrics`'s `pooled_RMSE` column, since
both are observation-pooled residuals over the same store. Pass 2 applies the pass-1
offsets and accumulates the corrected pooled RMSE/MAE; MAE cannot be recovered from pass-1
sums algebraically the way RMSE could (`sum_sq - sum_err**2/n` gives the corrected sum of
squares in closed form, but `sum(|x - offset|)` needs the individual residuals), so both
passes read every row rather than mixing an analytic shortcut for one metric with a real
pass for the other.

Usage::

    python -m stec.analysis.madrigal_method_offset_comparison
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..config import paths
from ..inference import prediction_store as ps
from .daily_metrics import DATASET_LABELS, MODELS, TRUTH_COLUMN
from .madrigal_reference_offset import MIN_OBSERVATIONS_PER_STATION

logger = logging.getLogger(__name__)

# The four store columns being compared, in `daily_metrics.MODELS`' order (Direct STEC,
# Pretrained STEC, VTEC + Mapping, IGS GIM) - reusing that dict rather than a fifth copy
# of the same four strings keeps the method labels identical to Tables 3/4's.
METHOD_COLUMNS = list(MODELS)
READ_COLUMNS = ["station", TRUTH_COLUMN, *METHOD_COLUMNS]

DEFAULT_STORE_ROOT = paths.LEGACY_PREDICTIONS
DEFAULT_MODEL_VARIANT = "finetuned_stec"
DEFAULT_OUTPUT_DIR = paths.analysis_result_dir(
    "madrigal_method_offset_comparison", rebuilt=True
)
DAILY_METRICS_SUMMARY = (
    paths.analysis_result_dir("daily_metrics", rebuilt=True) / "summary.csv"
)


@dataclass
class RunningStats:
    """Streaming n / sum(error) / sum(error**2) / sum(|error|) for one method.

    Every quantity this module reports is built from these four running sums, so
    streaming the store day at a time gives an exact answer, never an approximation of
    one - the same guarantee `daily_metrics.py` and `madrigal_reference_offset.py` rely
    on.
    """

    n: int = 0
    sum_err: float = 0.0
    sum_sq: float = 0.0
    sum_abs: float = 0.0

    def add(self, error: np.ndarray) -> None:
        self.n += int(error.size)
        self.sum_err += float(error.sum())
        self.sum_sq += float(np.square(error).sum())
        self.sum_abs += float(np.abs(error).sum())

    @property
    def mean(self) -> float:
        """The per-station offset, once this is grouped by station: mean(pred - truth)."""
        return self.sum_err / self.n if self.n else float("nan")

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sum_sq / self.n)) if self.n else float("nan")

    @property
    def mae(self) -> float:
        return self.sum_abs / self.n if self.n else float("nan")


@dataclass
class PassOneResult:
    """Everything pass 1 learns, before any offset can be applied."""

    pairwise: dict[str, RunningStats]
    per_station: dict[tuple[str, str], RunningStats]
    days_read: int
    total_rows: int
    truth_finite_rows: int
    intersected_rows: int


def _iter_madrigal_days(
    store_root: Path,
    model_variant: str,
    doys: Sequence[int] | None = None,
) -> Iterator[tuple[int, int, pd.DataFrame]]:
    """Yield `(year, doy, frame)` one Madrigal day at a time, six columns wide.

    Deliberately not `madrigal_reference_offset._iter_madrigal_days`: that helper's fixed
    `MADRIGAL_COLUMNS` list has no `vtec_model_stec`/`pretrained_stec_pred`, and widening
    it would touch a canonical, already-verified module for a caller it was never written
    for. This is a thin, self-contained re-read of the same store instead.
    """
    try:
        days = ps.available_days(model_variant, "madrigal", root=store_root)
    except FileNotFoundError:
        logger.warning(f"no {model_variant}/madrigal store under {store_root}")
        return
    if doys is not None:
        wanted = {int(d) for d in doys}
        days = [(year, doy) for year, doy in days if doy in wanted]
    logger.info(f"streaming {len(days)} Madrigal day(s)")
    for year, doy in days:
        _, _, frame = next(
            ps.iter_days(
                model_variant,
                "madrigal",
                years=[year],
                doys=[doy],
                columns=READ_COLUMNS,
                root=store_root,
            )
        )
        yield year, doy, frame


def collect_pass_one(
    store_root: Path,
    model_variant: str,
    doys: Sequence[int] | None = None,
) -> PassOneResult:
    """Pass 1: full-population pairwise pooled stats, plus per-station stats on the row
    set common to all four methods.

    "Pairwise" (one method against truth, ignoring what the other three methods have)
    is what `daily_metrics.py` computes for `pooled_RMSE`, so that accumulator is the
    correctness check. The per-station accumulator instead requires all four methods
    finite before a row counts at all - the "SAME rows" population steps 2-4 compare on
    - so the two differ whenever a method has rows the others do not; that gap is
    reported by `build_row_diagnostics`, not silently absorbed into either number.
    """
    pairwise: dict[str, RunningStats] = {col: RunningStats() for col in METHOD_COLUMNS}
    per_station: dict[tuple[str, str], RunningStats] = {}
    days_read = 0
    total_rows = 0
    truth_finite_rows = 0
    intersected_rows = 0

    for _year, _doy, frame in _iter_madrigal_days(store_root, model_variant, doys=doys):
        days_read += 1
        total_rows += len(frame)
        truth = frame[TRUTH_COLUMN].to_numpy(float)
        truth_ok = np.isfinite(truth)
        truth_finite_rows += int(truth_ok.sum())

        common = truth_ok.copy()
        for col in METHOD_COLUMNS:
            pred = frame[col].to_numpy(float)
            method_ok = truth_ok & np.isfinite(pred)
            if method_ok.any():
                pairwise[col].add(pred[method_ok] - truth[method_ok])
            common &= np.isfinite(pred)

        intersected_rows += int(common.sum())
        if not common.any():
            continue

        common_frame = pd.DataFrame({"station": frame["station"].to_numpy()[common]})
        truth_common = truth[common]
        for col in METHOD_COLUMNS:
            error = frame[col].to_numpy(float)[common] - truth_common
            common_frame[f"{col}__err"] = error
            common_frame[f"{col}__sq"] = error**2
            common_frame[f"{col}__abs"] = np.abs(error)

        grouped = common_frame.groupby("station", observed=True).agg(
            n=(f"{METHOD_COLUMNS[0]}__err", "size"),
            **{f"{col}__sum_err": (f"{col}__err", "sum") for col in METHOD_COLUMNS},
            **{f"{col}__sum_sq": (f"{col}__sq", "sum") for col in METHOD_COLUMNS},
            **{f"{col}__sum_abs": (f"{col}__abs", "sum") for col in METHOD_COLUMNS},
        )
        for station, row in grouped.iterrows():
            n = int(row["n"])
            for col in METHOD_COLUMNS:
                stat = per_station.setdefault((station, col), RunningStats())
                stat.n += n
                stat.sum_err += float(row[f"{col}__sum_err"])
                stat.sum_sq += float(row[f"{col}__sum_sq"])
                stat.sum_abs += float(row[f"{col}__sum_abs"])

    return PassOneResult(
        pairwise=pairwise,
        per_station=per_station,
        days_read=days_read,
        total_rows=total_rows,
        truth_finite_rows=truth_finite_rows,
        intersected_rows=intersected_rows,
    )


def build_offset_table(
    per_station: dict[tuple[str, str], RunningStats],
) -> pd.DataFrame:
    """One row per station with >= `MIN_OBSERVATIONS_PER_STATION` intersected
    observations, one `offset_<column>` per method - all four computed on exactly the
    same rows for that station, since `per_station` was only ever populated from the
    all-four-finite row set."""
    stations = sorted({station for station, _ in per_station})
    rows = []
    for station in stations:
        reference_stat = per_station[(station, METHOD_COLUMNS[0])]
        row = {"station": station, "observations": reference_stat.n}
        for col in METHOD_COLUMNS:
            row[f"offset_{col}"] = per_station[(station, col)].mean
        rows.append(row)
    table = pd.DataFrame(rows).set_index("station")
    if table.empty:
        return table
    return table[table["observations"] >= MIN_OBSERVATIONS_PER_STATION]


def build_row_diagnostics(pass_one: PassOneResult) -> pd.DataFrame:
    """Per method: how many rows had a finite truth and a finite prediction, and how
    many of the truth-finite rows were lost to that method's own nulls. The final row is
    the same question for the all-four-finite intersection the offset table and pass 2
    are built on."""
    rows = []
    for col in METHOD_COLUMNS:
        finite = pass_one.pairwise[col].n
        own_null = pass_one.truth_finite_rows - finite
        rows.append(
            {
                "Method": MODELS[col],
                "rows_with_finite_truth_and_prediction": finite,
                "rows_dropped_to_this_methods_own_nulls": own_null,
                "pct_dropped": 100 * own_null / pass_one.truth_finite_rows
                if pass_one.truth_finite_rows
                else float("nan"),
            }
        )
    intersection_dropped = pass_one.truth_finite_rows - pass_one.intersected_rows
    rows.append(
        {
            "Method": "ALL FOUR METHODS (intersected)",
            "rows_with_finite_truth_and_prediction": pass_one.intersected_rows,
            "rows_dropped_to_this_methods_own_nulls": intersection_dropped,
            "pct_dropped": 100 * intersection_dropped / pass_one.truth_finite_rows
            if pass_one.truth_finite_rows
            else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def compare_pairwise_to_daily_metrics(
    pairwise: dict[str, RunningStats], daily_metrics_summary_path: Path
) -> pd.DataFrame:
    """The correctness check: this module's own full-population pairwise pooled RMSE
    against `daily_metrics`'s `pooled_RMSE` for `dataset=madrigal_vtec_gim`. Both are the
    observation-pooled residual over the same store read the same way (pairwise, no
    per-station filter), so they should agree to floating-point precision; a real
    mismatch would mean the two modules are reading different rows."""
    if not daily_metrics_summary_path.exists():
        logger.warning(
            f"no daily_metrics summary at {daily_metrics_summary_path} - skipping the "
            "correctness check"
        )
        return pd.DataFrame()
    reference = pd.read_csv(daily_metrics_summary_path)
    reference = reference[reference["dataset"] == DATASET_LABELS["madrigal"]]
    rows = []
    for col in METHOD_COLUMNS:
        label = MODELS[col]
        match = reference[reference["Model"] == label]
        if match.empty:
            continue
        published = float(match.iloc[0]["pooled_RMSE"])
        computed = pairwise[col].rmse
        rows.append(
            {
                "Method": label,
                "computed_pooled_RMSE": computed,
                "daily_metrics_pooled_RMSE": published,
                "delta": computed - published,
            }
        )
    return pd.DataFrame(rows)


def collect_pass_two(
    store_root: Path,
    model_variant: str,
    offsets: pd.DataFrame,
    doys: Sequence[int] | None = None,
) -> tuple[dict[str, RunningStats], dict[str, RunningStats]]:
    """Pass 2: raw and offset-corrected pooled stats, restricted to rows whose station
    qualified for an offset and whose four predictions are all finite - exactly the
    population `offsets` was fitted on, so "before" and "after" differ only by the
    correction, never by a changed population."""
    raw: dict[str, RunningStats] = {col: RunningStats() for col in METHOD_COLUMNS}
    corrected: dict[str, RunningStats] = {col: RunningStats() for col in METHOD_COLUMNS}
    qualifying_stations = set(offsets.index)

    for _year, _doy, frame in _iter_madrigal_days(store_root, model_variant, doys=doys):
        frame = frame[frame["station"].isin(qualifying_stations)]
        if frame.empty:
            continue
        truth = frame[TRUTH_COLUMN].to_numpy(float)
        keep = np.isfinite(truth)
        for col in METHOD_COLUMNS:
            keep &= np.isfinite(frame[col].to_numpy(float))
        if not keep.any():
            continue
        kept = frame[keep]
        truth_kept = truth[keep]
        for col in METHOD_COLUMNS:
            error = kept[col].to_numpy(float) - truth_kept
            raw[col].add(error)
            offset = kept["station"].map(offsets[f"offset_{col}"]).to_numpy(float)
            corrected[col].add(error - offset)

    return raw, corrected


def build_pooled_before_after(
    raw: dict[str, RunningStats],
    corrected: dict[str, RunningStats],
    offsets: pd.DataFrame,
) -> pd.DataFrame:
    """The four-method table: RMSE/MAE before and after per-station offset removal, on
    the identical, offset-qualifying-station, all-four-finite row set, plus the rank
    each method holds before and after - the number that answers whether the paper's
    Madrigal reversal survives the correction."""
    rows = []
    for col in METHOD_COLUMNS:
        rows.append(
            {
                "Method": MODELS[col],
                "observations": raw[col].n,
                "qualifying_stations": len(offsets),
                "mean_abs_station_offset": offsets[f"offset_{col}"].abs().mean(),
                "RMSE_before": raw[col].rmse,
                "MAE_before": raw[col].mae,
                "RMSE_after": corrected[col].rmse,
                "MAE_after": corrected[col].mae,
            }
        )
    table = pd.DataFrame(rows)
    table["rank_before_RMSE"] = table["RMSE_before"].rank(method="min").astype(int)
    table["rank_after_RMSE"] = table["RMSE_after"].rank(method="min").astype(int)
    table["rank_before_MAE"] = table["MAE_before"].rank(method="min").astype(int)
    table["rank_after_MAE"] = table["MAE_after"].rank(method="min").astype(int)
    return table


def offset_correlation_matrix(offsets: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson and Spearman correlation of the four methods' per-station offset
    vectors. High agreement across all six pairs is the signature of a reference-product
    offset every method inherits identically; a pair (or one method against the other
    three) that disagrees is a signature that at least one offset vector is a property of
    that method, not of Madrigal."""
    rows = []
    for col_a, col_b in combinations(METHOD_COLUMNS, 2):
        a = offsets[f"offset_{col_a}"]
        b = offsets[f"offset_{col_b}"]
        pearson_r, pearson_p = stats.pearsonr(a, b)
        spearman_rho, spearman_p = stats.spearmanr(a, b)
        rows.append(
            {
                "method_a": MODELS[col_a],
                "method_b": MODELS[col_b],
                "stations": len(offsets),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "spearman_rho": spearman_rho,
                "spearman_p": spearman_p,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    """A pipe-delimited markdown table, built by hand rather than via
    `DataFrame.to_markdown()` - that method needs the optional `tabulate` package, which
    is not a repository dependency and is not installed in this environment."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    separator = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *body])


def _format_findings_markdown(
    pooled_table: pd.DataFrame,
    correlation: pd.DataFrame,
    diagnostics: pd.DataFrame,
    correctness: pd.DataFrame,
) -> str:
    """A narrative generated from the CSVs this run just wrote, not hand-maintained
    prose - so it cannot drift from the numbers the way earlier hand-written summaries
    of this comparison did (see CLAUDE.md's canonical-results table on that history)."""
    lines = [
        "# Does the Madrigal VTEC-over-Direct-STEC reversal survive offset removal?",
        "",
    ]

    lines.append("## Correctness check against `daily_metrics`'s pooled_RMSE")
    if correctness.empty:
        lines.append("(skipped - no daily_metrics summary.csv found)")
    else:
        lines.append(_markdown_table(correctness.round(4)))
    lines.append("")

    lines.append("## Row population")
    lines.append(_markdown_table(diagnostics.round(2)))
    lines.append("")

    lines.append("## Before / after per-station offset removal")
    lines.append(
        _markdown_table(
            pooled_table[
                [
                    "Method",
                    "observations",
                    "RMSE_before",
                    "RMSE_after",
                    "rank_before_RMSE",
                    "rank_after_RMSE",
                    "MAE_before",
                    "MAE_after",
                    "rank_before_MAE",
                    "rank_after_MAE",
                ]
            ].round(4)
        )
    )
    lines.append("")

    rmse_order_before = list(pooled_table.sort_values("rank_before_RMSE")["Method"])
    rmse_order_after = list(pooled_table.sort_values("rank_after_RMSE")["Method"])
    ranking_changed = rmse_order_before != rmse_order_after
    lines.append("## Ranking")
    lines.append(f"Before: {' < '.join(rmse_order_before)}")
    lines.append(f"After:  {' < '.join(rmse_order_after)}")
    lines.append(
        "**Ranking changed.**"
        if ranking_changed
        else "**Ranking did NOT change** - removing the per-station reference offset "
        "does not reverse or alter the method ordering."
    )
    lines.append("")

    lines.append("## Offset vector agreement across methods")
    lines.append(_markdown_table(correlation.round(3)))
    min_pearson = float(correlation["pearson_r"].min())
    lines.append("")
    if min_pearson > 0.7:
        lines.append(
            f"All six pairwise Pearson correlations exceed 0.7 (minimum {min_pearson:.3f}) "
            "- the per-station offset looks common-mode, i.e. a property of the Madrigal "
            "reference that every method inherits, not a property of any one method."
        )
    else:
        lines.append(
            f"At least one pair falls below 0.7 Pearson (minimum {min_pearson:.3f}) - the "
            "offset is not uniformly common-mode; check `offset_correlation.csv` for which "
            "pair disagrees before treating the correction as reference-only."
        )
    lines.append("")
    lines.append(
        "Removing a per-station offset fitted on the same data can only ever reduce RMSE; "
        f"with {int(pooled_table['qualifying_stations'].iloc[0])} station parameters "
        f"against {int(pooled_table['observations'].sum()):,} observations across all four "
        "methods, the resulting optimism is negligible but not exactly zero."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--model-variant", type=str, default=DEFAULT_MODEL_VARIANT)
    parser.add_argument(
        "--doys",
        type=int,
        nargs="*",
        default=None,
        help="Restrict to these day-of-year values; default is every Madrigal day in "
        "the store.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--daily-metrics-summary", type=Path, default=DAILY_METRICS_SUMMARY
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    logger.info("pass 1/2: pairwise pooled stats and per-station offsets")
    pass_one = collect_pass_one(args.store_root, args.model_variant, doys=args.doys)
    if pass_one.total_rows == 0:
        raise RuntimeError(
            f"no Madrigal observations read from {args.store_root} - is the store "
            "populated?"
        )

    offsets = build_offset_table(pass_one.per_station)
    if offsets.empty:
        raise RuntimeError(
            f"no station cleared {MIN_OBSERVATIONS_PER_STATION} intersected "
            "observations - nothing to decompose"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    offsets.to_csv(args.output_dir / "per_station_offsets.csv")

    diagnostics = build_row_diagnostics(pass_one)
    diagnostics.to_csv(
        args.output_dir / "row_intersection_diagnostics.csv", index=False
    )
    print("=== Row population (truth-finite rows, then per-method own nulls) ===")
    print(diagnostics.round(2).to_string(index=False))

    correctness = compare_pairwise_to_daily_metrics(
        pass_one.pairwise, args.daily_metrics_summary
    )
    if not correctness.empty:
        correctness.to_csv(
            args.output_dir / "correctness_check_vs_daily_metrics.csv", index=False
        )
        print(
            "\n=== Correctness check: pooled RMSE against daily_metrics summary.csv ==="
        )
        print(correctness.round(4).to_string(index=False))

    logger.info("pass 2/2: offset-corrected pooled stats")
    raw, corrected = collect_pass_two(
        args.store_root, args.model_variant, offsets, doys=args.doys
    )

    pooled_table = build_pooled_before_after(raw, corrected, offsets)
    pooled_table.to_csv(args.output_dir / "pooled_before_after.csv", index=False)
    print("\n=== Before / after per-station offset removal, all four methods ===")
    print(pooled_table.round(4).to_string(index=False))

    correlation = offset_correlation_matrix(offsets)
    correlation.to_csv(args.output_dir / "offset_correlation.csv", index=False)
    print(
        "\n=== Offset-vector agreement across methods (n = %d stations) ==="
        % len(offsets)
    )
    print(correlation.round(3).to_string(index=False))

    findings = _format_findings_markdown(
        pooled_table, correlation, diagnostics, correctness
    )
    (args.output_dir / "FINDINGS.md").write_text(findings)

    logger.info(f"wrote outputs and FINDINGS.md to {args.output_dir}")


if __name__ == "__main__":
    main()
