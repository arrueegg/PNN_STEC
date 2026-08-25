"""Pure day-selection and skip logic for scripts/weekend_missing_data_queue.sh.

Kept out of the bash script (and out of an inline heredoc, which is this repo's usual
pattern - see backfill_store.sh) specifically so it can be unit-tested against synthetic
fixtures instead of only ever being exercised by a real overnight run. Every function here
is pure or touches only the filesystem paths it is given - no dependency on `src/` or
`stec/`, so it works identically from the worktree or the data root.

The only body of missing work with genuinely new selection logic is the finetuned/madrigal
gap-fill: which of the days present in the own-dataset store but absent from madrigal can
actually be re-run. The other two bodies deliberately do NOT get their own copy of
day-selection logic here:

  - Positioning recovery's day list and per-day skip behaviour already live in
    positioning/geometry/recover_day.py and the coverage.csv it reads - the queue script
    delegates to the existing, already-proven scripts/run_station_recovery.sh rather than
    re-deriving which station-days are outstanding. A second implementation of that
    decision is exactly the "two owners of one answer" failure this project's own
    CLAUDE.md warns about (Tables 3 and 4 once existed in three disagreeing places).
  - Pretrained/madrigal inference has no day-selection question yet because nothing in
    this tree can write it at all (see `pretrained_madrigal_driver_available` below) - the
    open question is a missing driver, not which days to feed it.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# The exact marker compare_stec_vtec_gim.py logs when its primary model is not a
# fine-tuned one. Its disappearance is how this module detects that someone has since
# taught the legacy comparison path (or a replacement) to evaluate a pretrained model
# against Madrigal - checked, never assumed, matching this repo's own stated policy for
# the merge-safe writer.
_PRETRAINED_MADRIGAL_GUARD_MARKER = "Pretrained model detected - Madrigal evaluation only supported for finetuned models"


def _has_required_columns(path: Path, required_columns: Sequence[str]) -> bool:
    """Schema-only completeness check: reads the parquet FOOTER, never the row data.

    `ParquetFile(path).schema` is metadata I/O - it does not touch the row groups - which
    matters here because `store_days` is a day-selection scan that can run over a
    partition holding tens of GB; it must never pay for a real read just to decide
    whether a day is done.

    CLAUDE.md documents real per-day store files truncated mid-write by a killed
    recovery-sweep job. Opening one of those raises `pyarrow.ArrowInvalid` ("Parquet
    magic bytes not found in footer" for a torn footer, "Parquet file size is 0 bytes"
    for one caught at the very start of a write) rather than returning a schema -
    confirmed by truncating a real parquet file at several points and reading a genuine
    zero-byte file. Treated the same as a day with a missing column: not done, worth
    logging so the truncation is visible rather than silently re-queued forever without
    explanation.
    """
    try:
        schema_columns = set(pq.ParquetFile(path).schema.names)
    except pa.ArrowInvalid:
        logger.warning(f"{path}: could not read parquet footer (truncated or corrupt)")
        return False
    return set(required_columns).issubset(schema_columns)


def store_days(
    store_root: Path,
    model_variant: str,
    dataset: str,
    required_columns: Sequence[str] | None = None,
) -> set[int]:
    """DOYs already on disk for one (model_variant, dataset) prediction-store partition.

    Reads the parquet filenames directly (`year=*/doy=*.parquet`) rather than importing
    `evaluation.prediction_store.available_days`, which lives under the data root's `src/`
    tree and must not become an import-time dependency of code that also needs to run from
    the worktree.

    `required_columns`, when given, redefines "done" as "the file exists AND its schema
    carries every one of these columns" rather than existence alone. Existence alone is
    not always enough: `predictions/pretrained_stec/madrigal/year=2024/doy=122.parquet`
    is a real, unremarkable-looking file (2,036,513 rows, passed its own
    zero-perturbation control) whose driver died before the run that adds this
    partition's baseline columns finished - existence-only selection would count it as
    done forever and silently skip it on every future gap-fill. Every existing caller
    omits this argument and keeps the old existence-only behavior unchanged.
    """
    base = store_root / model_variant / dataset
    if not base.is_dir():
        return set()
    days: set[int] = set()
    for path in base.glob("year=*/doy=*.parquet"):
        if required_columns is not None and not _has_required_columns(
            path, required_columns
        ):
            continue
        days.add(int(path.stem.split("=")[1]))
    return days


def madrigal_source_exists(madrigal_root: Path, year: int, doy: int) -> bool:
    """Whether this host has the raw Madrigal file a day's inference would read.

    Absence here is permanent, not a scheduling problem: 2024 DOY 199-202 have no
    `los_<date>_IGS.h5` on this host at all (confirmed by directory listing, 2026-08-21),
    the same failure shape CLAUDE.md already documents for positioning DOY 303/338/348 -
    no amount of re-running the queue produces a file this host was never given.
    """
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    return (madrigal_root / str(year) / f"los_{date:%Y%m%d}_IGS.h5").is_file()


def madrigal_gap(own_days: set[int], madrigal_days: set[int]) -> list[int]:
    """Days finished for the own dataset but still missing from madrigal, ascending."""
    return sorted(own_days - madrigal_days)


def partition_recoverable(
    days: list[int], madrigal_root: Path, year: int
) -> tuple[list[int], list[int]]:
    """Split gap days into ones whose raw Madrigal file exists (worth re-running) and
    ones that do not (re-running would just fail again - see `madrigal_source_exists`)."""
    recoverable, unrecoverable = [], []
    for doy in days:
        target = (
            recoverable
            if madrigal_source_exists(madrigal_root, year, doy)
            else unrecoverable
        )
        target.append(doy)
    return recoverable, unrecoverable


def format_dates(year: int, days: list[int]) -> str:
    """`cli.py multiday --dates` wants comma-separated `YYYY-DOY` tokens."""
    return ",".join(f"{year}-{doy:03d}" for doy in days)


def merge_safe_writer_present(repo_root: Path) -> bool:
    """True once `stec/positioning/summary_writer.py` exists in `repo_root`.

    That file's presence is the single, checkable proxy for "the pipeline-rebuild branch
    has merged into this tree": `positioning/positioning_eval/metrics.py` is rewritten to
    import `save_daily_summary` from it in the same merge commit that adds the `stec`
    package, so the two always land together. Positioning recovery must not start against
    a tree where `save_daily_summary` is still the pre-fix bare `to_csv()` overwrite - see
    stec/positioning/summary_writer.py's own docstring for the 59-file data-loss incident
    that fix exists to prevent.
    """
    return (repo_root / "stec" / "positioning" / "summary_writer.py").is_file()


def pretrained_madrigal_driver_available(repo_root: Path) -> bool:
    """True once something in `repo_root` can write predictions/pretrained_stec/madrigal.

    As of the 2026-08-21 investigation behind this script, nothing can:
      - the legacy `src/compare_stec_vtec_gim.py` hard-skips Madrigal whenever the primary
        model's config mode is not "finetune" (it logs `_PRETRAINED_MADRIGAL_GUARD_MARKER`
        and never adds "madrigal" to the datasets it evaluates);
      - the rebuilt `stec.inference.run_inference` raises immediately on `--dataset
        madrigal`, because nothing in `stec/` reads Madrigal geometry as a model *input*
        yet (only as a reference to score against) - see that module's own docstring.
    Detected here as "the legacy guard's marker string is gone", so a future fix on either
    side flips this without the check itself needing an update.
    """
    legacy_comparison = repo_root / "src" / "compare_stec_vtec_gim.py"
    if not legacy_comparison.is_file():
        return False
    return _PRETRAINED_MADRIGAL_GUARD_MARKER not in legacy_comparison.read_text(
        errors="ignore"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    gap = subparsers.add_parser(
        "madrigal-gap",
        help="Print recoverable=... and unrecoverable=... DOY lines for the "
        "finetuned/madrigal gap.",
    )
    gap.add_argument("--store-root", type=Path, required=True)
    gap.add_argument("--madrigal-root", type=Path, required=True)
    gap.add_argument("--model-variant", default="finetuned_stec")
    gap.add_argument("--year", type=int, default=2024)
    gap.add_argument(
        "--required-columns",
        default=None,
        help="Comma-separated column names the MADRIGAL side's parquet footer must "
        "carry for a day to count as done; a day whose file exists but is missing one "
        "of these is still reported as recoverable. Only ever applied to the madrigal "
        "side, never to own - own is the reference set of candidate days, not itself "
        "subject to the completeness question. Omit (the default) to keep the old "
        "exists-only behavior. Needed for pretrained_stec/madrigal, where doy=122 "
        "exists but lacks every baseline column (docs/revision/independent_audit.md's "
        "F8 finding).",
    )

    writer = subparsers.add_parser(
        "merge-safe-writer-present",
        help="Exit 0 if the merge-safe save_daily_summary is present in --root.",
    )
    writer.add_argument("--root", type=Path, required=True)

    driver = subparsers.add_parser(
        "pretrained-madrigal-driver-available",
        help="Exit 0 if --root can write predictions/pretrained_stec/madrigal.",
    )
    driver.add_argument("--root", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "madrigal-gap":
        required_columns = (
            [name.strip() for name in args.required_columns.split(",")]
            if args.required_columns
            else None
        )
        own = store_days(args.store_root, args.model_variant, "own")
        madrigal = store_days(
            args.store_root,
            args.model_variant,
            "madrigal",
            required_columns=required_columns,
        )
        gap_days = madrigal_gap(own, madrigal)
        recoverable, unrecoverable = partition_recoverable(
            gap_days, args.madrigal_root, args.year
        )
        print(f"recoverable={format_dates(args.year, recoverable)}")
        print(f"unrecoverable={format_dates(args.year, unrecoverable)}")
        return 0

    if args.command == "merge-safe-writer-present":
        return 0 if merge_safe_writer_present(args.root) else 1

    if args.command == "pretrained-madrigal-driver-available":
        return 0 if pretrained_madrigal_driver_available(args.root) else 1

    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable, satisfies type checkers


if __name__ == "__main__":
    sys.exit(main())
