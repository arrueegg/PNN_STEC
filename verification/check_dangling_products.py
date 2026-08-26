"""Detect dangling PPPx product symlinks in a positioning evaluation tree.

`experiments/Reference_STEC_Oracle` and `experiments/Fixed_Variance_STEC` each hold one
`positioning/evaluation/<YYYYDDD>/products/` directory per day. Neither tree has its own
per-day fine-tune, so their products are symlinks into whichever experiment produced the
real files for that day - normally a `Finetune_VTEC_2024_<DOY>_*` directory (in fact
`Fixed_Variance_STEC`'s own `products` directories are themselves symlinks straight into
`Reference_STEC_Oracle`'s, so the two trees share one fate by construction). For 158 of
242 days in both trees, that chain instead ends at
`Pretrain_STEC_.../positioning/evaluation/<day>/products/`, a directory a routine
post-run cleanup deleted - so every non-SINEX product symlink for those days is dangling.

This is the identical mechanism that cost `oracle_benchmark` 166 days of SINEX symlinks
(see that stage's own comment in `stec/pipeline/stages.py` and
`stec/positioning/metrics.py::load_sinex_coords`), just on the other five product types
PPPx needs at run time: ERP, GIM-INX, ORB-SP3, ATT-OBX, CLK.

What this does NOT mean: a day's already-computed `.pos` solution and
`positioning/results/<day>/daily_summary*.csv` are untouched by this - products are only
read at PPPx run time, never at analysis time. The actual loss is narrower and permanent:
that day can no longer be *re-run* on this host, because CODE's FTP is firewalled and
CDDIS needs Earthdata credentials this host does not have. This script therefore reports
two different things under one dangling-products day, not one:

* RECOVERABLE (informational) - products dangling, but `daily_summary*.csv` for that day
  already exists and is non-empty. Nothing today reads the missing products; only a
  future re-run would notice.
* UNRECOVERABLE (real, permanent loss) - products dangling AND no existing results. This
  is the only condition this script fails on.

Not wired as a pipeline `checks` callable: a `checks` entry only runs after a stage's own
outputs are written and only sees that stage's declared output-path mapping, but the
failure here is in an *input* tree neither `oracle_benchmark` nor `weighting_ablation`
produces or owns - the products directories are read directly by the PPPx run, not by any
declared stage output. And even where a stage does declare one of these trees as an input
(`oracle_benchmark` -> ORACLE_EXPERIMENT_DIR, `weighting_ablation` ->
WEIGHTING_ABLATION_FIXED_VARIANCE_DIR), a dangling symlink changes neither the tree's
total size nor its outer mtime in a way `stec/pipeline/fingerprint.py`'s tree digest is
guaranteed to catch, and either stage failing outright over it would block work
(`daily_metrics`, `positioning_summary`, ...) that depends on results already computed
and correct today - the thing this module's own docstring says a `checks` callable must
not do. A standalone, run-on-demand script that never blocks the pipeline is the right
shape for a defect that is real, already known to be permanent for 158 days, and does not
retroactively invalidate anything already on disk.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The five product types PPPx reads at run time beyond the day-invariant SINEX ground
# truth (which `oracle_benchmark`'s own `load_sinex_coords` already guards against going
# dangling - see stec/positioning/metrics.py). Matched by filename suffix, not the full
# CODE product name, because the name's date-stamped prefix
# (`COD0OPSFIN_<YYYYDDD>0000_...`) varies per day; see
# positioning/positioning_eval/download_products.py::get_product_paths for the full
# templates this suffix set is drawn from.
PRODUCT_SUFFIXES = {
    "ERP": "_ERP.ERP",
    "GIM-INX": "_GIM.INX",
    "ORB-SP3": "_ORB.SP3",
    "ATT-OBX": "_ATT.OBX",
    "CLK": "_CLK.CLK",
}

# The two trees known to share this failure (docs/revision/work_queue.md's "Products"
# row). Pass --experiment to check others; nothing here is specific to these two beyond
# being the default.
DEFAULT_EXPERIMENTS = [
    "experiments/Reference_STEC_Oracle",
    "experiments/Fixed_Variance_STEC",
]


def product_status(products_dir: Path) -> dict[str, str]:
    """label -> "ok" | "dangling" | "missing", for each of PRODUCT_SUFFIXES found (or not)
    in one day's products directory.

    "missing" (no file of that type at all) is a related but different gap - a download
    that never happened, not a symlink that broke - and is reported for completeness but
    does not by itself make a day RECOVERABLE/UNRECOVERABLE below; only "dangling" does,
    since that is the failure this script exists to catch. `products_dir` may itself be a
    symlink (see the module docstring) or entirely absent - `Path.iterdir()` follows a
    symlinked directory transparently and raises FileNotFoundError for a genuinely
    missing one, which reads here as every type "missing".
    """
    try:
        entries = list(products_dir.iterdir())
    except (FileNotFoundError, NotADirectoryError):
        entries = []

    status = {}
    for label, suffix in PRODUCT_SUFFIXES.items():
        matches = [p for p in entries if p.name.endswith(suffix)]
        if not matches:
            status[label] = "missing"
        elif any(p.exists() for p in matches):
            status[label] = "ok"
        else:
            status[label] = "dangling"
    return status


def day_has_results(experiment: Path, day: str) -> bool:
    """Whether PPPx has already produced this day's summary, independent of whether its
    products still resolve. Globs `daily_summary*.csv` rather than one hardcoded name,
    because this repo uses `daily_summary.csv` for elev results (Oracle) and
    `daily_summary_iono.csv` for iono results (Fixed-Variance) - see CLAUDE.md's
    "Weighting provenance" note.
    """
    results_dir = experiment / "positioning" / "results" / day
    return any(f.stat().st_size > 0 for f in results_dir.glob("daily_summary*.csv"))


@dataclass
class ExperimentReport:
    experiment: str
    total_days: int
    ok_days: list[str] = field(default_factory=list)
    recoverable_days: list[str] = field(default_factory=list)
    unrecoverable_days: list[str] = field(default_factory=list)
    missing_only_days: list[str] = field(default_factory=list)


def check_experiment(experiment: Path) -> ExperimentReport:
    eval_root = experiment / "positioning" / "evaluation"
    days = sorted(p.name for p in eval_root.iterdir() if p.is_dir())

    report = ExperimentReport(experiment=str(experiment), total_days=len(days))
    for day in days:
        status = product_status(eval_root / day / "products")
        if not any(v != "missing" for v in status.values()):
            # No product of any of the 5 types was found at all - a different, likely
            # pre-existing gap (this day's products were never downloaded/linked), not
            # this script's concern.
            report.missing_only_days.append(day)
            continue

        has_dangling = any(v == "dangling" for v in status.values())
        if not has_dangling:
            report.ok_days.append(day)
        elif day_has_results(experiment, day):
            report.recoverable_days.append(day)
        else:
            report.unrecoverable_days.append(day)

    return report


def print_report(report: ExperimentReport) -> None:
    print(f"{report.experiment}: {report.total_days} day(s)")
    print(f"  products intact:                          {len(report.ok_days)}")
    print(
        f"  dangling, results already computed (info):  {len(report.recoverable_days)}"
    )
    print(
        "  dangling, NO results - cannot be re-run:  "
        f"  {len(report.unrecoverable_days)}"
    )
    if report.unrecoverable_days:
        print(f"    days: {report.unrecoverable_days}")
    if report.missing_only_days:
        print(
            "  no product files of any of the 5 types at all (not a dangling-symlink "
            f"issue): {len(report.missing_only_days)}"
        )
        print(f"    days: {report.missing_only_days}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="experiment directory to check (repeatable); default: the two known-"
        "affected trees, experiments/Reference_STEC_Oracle and "
        "experiments/Fixed_Variance_STEC",
    )
    args = parser.parse_args()
    experiments = args.experiments or DEFAULT_EXPERIMENTS

    unrecoverable_total = 0
    for exp in experiments:
        report = check_experiment(Path(exp))
        print_report(report)
        unrecoverable_total += len(report.unrecoverable_days)

    if unrecoverable_total:
        print(
            f"FAIL: {unrecoverable_total} day(s) have dangling products AND no "
            "existing results - permanently unrunnable on this host without product "
            "recovery."
        )
        return 1

    print(
        "OK: every dangling-products day already has computed results (re-run blocked, "
        "nothing lost today)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
