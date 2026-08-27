#!/usr/bin/env python3
"""Category B/D targeting for scripts/coverage_chain_followon.sh.

`positioning_coverage`'s ``coverage.csv`` classifies every GIM-solved station-day the ML
methods didn't all solve as ``"some ML methods missing (per-method failure)"``. A 2026-08-27
investigation (``docs/revision/work_queue.md``) split the arm-instances behind that cause
into three buckets:

* **Category D** - a ``.pos`` file already exists for that (arm, doy, station); PPPx
  succeeded, but the row never reached ``daily_summary_iono.csv``. Needs re-aggregation
  only, no PPPx. Direct evidence: DOY 122/STEC has 44 ``.pos`` files but only 42 rows in the
  summary (BRMG and LICC dropped during aggregation) - confirmed again against the live
  tree while writing this module.
* **Category B** - the model correction CSV exists on disk but PPPx was never invoked for
  that station in that arm's sweep. Needs a PPPx run; the correction itself does not need
  regenerating (regenerating it needs a GPU forward pass this module has no business
  triggering for data that is already there).
* **Category A** - neither exists. Out of scope here (the STEC arm lagging behind
  VTEC/Pretrained in pointing at recovered data; expected to shrink on its own once the
  concurrent station-recovery sweep finishes) - `categorize()` still labels it so a caller
  can report the count, but nothing here acts on it.

Restricted to **iono weighting only** (``daily_summary_iono.csv``), matching Table 5 and
every other canonical positioning number in this repo - `elev` is a separate, admittedly
stale tree (see CLAUDE.md's weighting-ablation note) and mixing the two here would silently
conflate two different populations.

**Canonical experiment resolution is deliberately NOT
`positioning/geometry/recover_day.py`'s `EXPERIMENT_PATTERNS` + `resolve_experiment()`.**
That helper globs ``Finetune_STEC_2024_{doy}_BayesianResNetSTEC_*_SWI`` and a VTEC
equivalent, then takes ``sorted(matches)[0]`` - alphabetically first, not the paper's
canonical fine-tune. Checked directly against the 212-DOY list the concurrent
`overnight-chain-20260826` recovery sweep is running over (2026-08-27): for 29 DOYs
(122-153) it silently resolves to a hyperparameter-search variant
(``lr1e-4_bs2048``/``lr1e-4_bs10000``) instead of the paper's ``lr2e-4_bs512`` fine-tune,
because the string ``"lr1e-4"`` sorts before ``"lr2e-4"`` - the exact bug
`stec/analysis/positioning_coverage.py`'s own module docstring documents and fixed for its
`collect()`, but that fix was never carried over to `recover_day.py`. DOY 122 hits the same
defect on its VTEC arm too (``lr1e-2`` sorts first). Corrections or positioning results
written against a non-canonical directory would never be read back by
`positioning_coverage`'s canonical-only glob, so this module builds the canonical path
directly from ``CANONICAL_STEC_SUFFIX`` / ``CANONICAL_VTEC_SUFFIX`` /
``CANONICAL_PRETRAINED_DIR`` instead of trusting that helper. This is flagged, not fixed,
here - `recover_day.py` is being actively invoked by the currently-running sweep and is out
of scope for this module to touch.

Usage (invoked by scripts/coverage_chain_followon.sh, not normally run by hand)::

    python3 scripts/lib/coverage_followon.py categorize --out triples.csv --label pre
    python3 scripts/lib/coverage_followon.py reaggregate-category-d --triples triples.csv
    python3 scripts/lib/coverage_followon.py list-category-b-groups --triples triples.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
POSITIONING_EVAL_DIR = REPO_ROOT / "positioning" / "positioning_eval"
if str(POSITIONING_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(POSITIONING_EVAL_DIR))

from stec.analysis.positioning_coverage import (  # noqa: E402
    CANONICAL_PRETRAINED_DIR,
    CANONICAL_STEC_SUFFIX,
    CANONICAL_VTEC_SUFFIX,
    SOME_ML_MISSING,
)
from stec.config.paths import LEGACY_EXPERIMENTS, analysis_result_dir  # noqa: E402

WEIGHTING = "iono"
ARM_SUFFIX = {"STEC": CANONICAL_STEC_SUFFIX, "VTEC": CANONICAL_VTEC_SUFFIX}


def canonical_experiment_dir(arm: str, doy: int) -> Path:
    """The paper's canonical fine-tune directory for `arm`/`doy` - see module docstring
    for why this is not `recover_day.resolve_experiment()`."""
    if arm == "Pretrained_STEC":
        return LEGACY_EXPERIMENTS / CANONICAL_PRETRAINED_DIR
    return LEGACY_EXPERIMENTS / f"Finetune_{arm}_2024_{doy:03d}_{ARM_SUFFIX[arm]}"


def pos_file_path(exp_dir: Path, doy: int, station: str) -> Path:
    tag = f"2024{doy:03d}"
    return (
        exp_dir
        / "positioning"
        / "results"
        / tag
        / "model_iono"
        / station
        / f"{station}_model_iono.pos"
    )


def correction_file_path(exp_dir: Path, doy: int, station: str) -> Path:
    tag = f"2024{doy:03d}"
    return exp_dir / "positioning" / "stec_corrections" / tag / f"{station}.csv"


def categorize() -> pd.DataFrame:
    """Every (arm, doy, station) triple `coverage.csv` currently lists under
    `SOME_ML_MISSING`, labelled 'D' / 'B' / 'A' per the module docstring. Reads
    `coverage.csv` fresh on every call - a caller wanting a stable snapshot across several
    steps should write the result once and reuse the CSV, not call this twice expecting the
    same rows (coverage.csv itself only changes when `positioning_coverage` is re-run).
    """
    coverage_path = (
        analysis_result_dir("positioning_coverage", rebuilt=True) / "coverage.csv"
    )
    coverage = pd.read_csv(coverage_path)
    some_missing = coverage[coverage["cause"] == SOME_ML_MISSING]

    triples = []
    suffix = f"_{WEIGHTING}"
    for _, row in some_missing.iterrows():
        for token in str(row["missing_methods"]).split(","):
            token = token.strip()
            if not token or not token.endswith(suffix):
                continue
            triples.append(
                {
                    "arm": token[: -len(suffix)],
                    "doy": int(row["doy"]),
                    "station": str(row["station"]).upper(),
                }
            )

    detail = []
    for triple in triples:
        arm, doy, station = triple["arm"], triple["doy"], triple["station"]
        exp_dir = canonical_experiment_dir(arm, doy)
        if not exp_dir.is_dir():
            category = "A"  # no experiment directory - nothing this module can act on
        elif pos_file_path(exp_dir, doy, station).exists():
            category = "D"
        elif correction_file_path(exp_dir, doy, station).exists():
            category = "B"
        else:
            category = "A"
        detail.append(
            {
                "arm": arm,
                "doy": doy,
                "station": station,
                "exp_dir": str(exp_dir),
                "category": category,
            }
        )

    return pd.DataFrame(
        detail, columns=["arm", "doy", "station", "exp_dir", "category"]
    )


def cmd_categorize(args: argparse.Namespace) -> None:
    detail_df = categorize()
    detail_df.to_csv(args.out, index=False)
    counts = detail_df["category"].value_counts()
    print(
        f"[SUMMARY][{args.label}] arm-instances: total={len(detail_df)} "
        f"category_D(pos-exists-no-row)={int(counts.get('D', 0))} "
        f"category_B(correction-exists-no-pos)={int(counts.get('B', 0))} "
        f"category_A(no-correction)={int(counts.get('A', 0))}"
    )


def cmd_reaggregate_category_d(args: argparse.Namespace) -> None:
    """Phase B: merge-safe re-aggregation from existing .pos files only, no PPPx.

    Each (doy, arm) group is processed independently and failures are collected rather than
    raised immediately, so one day's corrupted SINEX cannot hide the result of every other,
    independent day. The process still exits non-zero if any group failed - the caller
    (scripts/coverage_chain_followon.sh) stops the chain on that, per this module's own
    contract: category D is supposed to be pure, safe re-aggregation, and a failure there
    is unexpected and worth a human before Phase C/D build on top of it.
    """
    from download_products import download_products  # noqa: PLC0415
    from metrics import aggregate_daily_metrics  # noqa: PLC0415

    from stec.positioning.summary_writer import (  # noqa: PLC0415
        SummaryShrinkError,
        save_daily_summary,
    )

    df = pd.read_csv(args.triples)
    targets = df[df["category"] == "D"]
    if targets.empty:
        print("[phaseB] no category-D rows - nothing to re-aggregate")
        return

    groups = targets.groupby(["doy", "arm"]).agg(exp_dir=("exp_dir", "first"))

    if args.dry_run:
        print(
            f"[DRY RUN][phaseB] {len(groups)} (doy, arm) group(s) would be re-aggregated:"
        )
        for (doy, arm), row in groups.iterrows():
            stations = sorted(
                targets[(targets["doy"] == doy) & (targets["arm"] == arm)]["station"]
            )
            print(
                f"[DRY RUN][phaseB]   DOY {int(doy)} {arm} ({row['exp_dir']}): {stations}"
            )
        return

    recovered_total = 0
    failures: list[tuple[str, int, str]] = []
    for (doy, arm), row in groups.iterrows():
        doy = int(doy)
        exp_dir = Path(row["exp_dir"])
        tag = f"2024{doy:03d}"
        results_dir = exp_dir / "positioning" / "results" / tag
        summary_file = results_dir / "daily_summary_iono.csv"
        model_dir = results_dir / "model_iono"

        products_dir = exp_dir / "positioning" / "evaluation" / tag / "products"
        products_dir.mkdir(parents=True, exist_ok=True)
        snx_file = products_dir / f"IGS0OPSSNX_2024{doy:03d}0000_01D_01D_CRD.SNX"
        if not snx_file.exists():
            try:
                download_products(2024, doy, str(products_dir), None)
            except Exception as exc:  # noqa: BLE001 - isolate one bad day, see docstring
                failures.append((arm, doy, f"SNX download raised: {exc}"))
                continue
        if not snx_file.exists():
            failures.append(
                (
                    arm,
                    doy,
                    "no SNX available even via sibling reuse - skipped rather than "
                    "aggregated against a degraded day-mean reference",
                )
            )
            continue

        try:
            before = 0
            if summary_file.exists():
                before = int(
                    (pd.read_csv(summary_file)["method"] == "model_iono").sum()
                )

            metrics_model = aggregate_daily_metrics(
                model_dir, 2024, doy, "model_iono", stations=None, snx_file=snx_file
            )
            if metrics_model is None:
                failures.append(
                    (arm, doy, "aggregate_daily_metrics found no .pos files")
                )
                continue

            save_daily_summary(metrics_model, None, summary_file)

            after = int((pd.read_csv(summary_file)["method"] == "model_iono").sum())
        except SummaryShrinkError as exc:
            failures.append((arm, doy, f"SummaryShrinkError: {exc}"))
            continue
        except Exception as exc:  # noqa: BLE001 - isolate one bad day, see docstring
            failures.append((arm, doy, f"unexpected error: {exc}"))
            continue

        gained = after - before
        recovered_total += max(gained, 0)
        print(
            f"[phaseB] {arm} DOY {doy}: model_iono rows {before} -> {after} (+{gained})"
        )

    print(
        f"[phaseB] SUMMARY: {recovered_total} row(s) recovered across {len(groups)} "
        f"(doy, arm) group(s), {len(failures)} failure(s)"
    )
    for arm, doy, reason in failures:
        print(f"[phaseB][FAILED] {arm} DOY {doy}: {reason}")
    if failures:
        sys.exit(1)


def cmd_list_category_b_groups(args: argparse.Namespace) -> None:
    """Phase C: print one '<doy>\\t<arm>\\t<exp_dir_name>\\t<space-separated stations>' line
    per (doy, arm) group still needing PPPx, sorted by doy - consumed by a bash while-loop
    in scripts/coverage_chain_followon.sh so DOY-adjacent arms share one --rinex_dir."""
    df = pd.read_csv(args.triples)
    targets = df[df["category"] == "B"]
    if targets.empty:
        return
    grouped = targets.groupby(["doy", "arm"]).agg(
        exp_dir=("exp_dir", "first"),
        stations=("station", lambda s: " ".join(sorted(s))),
    )
    for (doy, arm), row in grouped.sort_index().iterrows():
        exp_name = Path(row["exp_dir"]).name
        print(f"{int(doy)}\t{arm}\t{exp_name}\t{row['stations']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_categorize = sub.add_parser(
        "categorize", help="write the per-triple category CSV"
    )
    p_categorize.add_argument("--out", type=Path, required=True)
    p_categorize.add_argument("--label", required=True)
    p_categorize.set_defaults(func=cmd_categorize)

    p_reagg = sub.add_parser(
        "reaggregate-category-d", help="Phase B: merge existing .pos files, no PPPx"
    )
    p_reagg.add_argument("--triples", type=Path, required=True)
    p_reagg.add_argument("--dry-run", action="store_true")
    p_reagg.set_defaults(func=cmd_reaggregate_category_d)

    p_list = sub.add_parser(
        "list-category-b-groups",
        help="Phase C: print PPPx work groups (doy, arm, stations)",
    )
    p_list.add_argument("--triples", type=Path, required=True)
    p_list.set_defaults(func=cmd_list_category_b_groups)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
