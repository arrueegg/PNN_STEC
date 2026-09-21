"""Rebuild the per-day positioning summaries the recovery sweep overwrote.

`save_daily_summary` writes its frame with `to_csv` and no merge, and `recover_day.py`
calls it with only the handful of stations it has just recovered. Every day the sweep
touched therefore lost the rows for every station that had already been solved: 91
`daily_summary_iono.csv` files fell from roughly 74-81 rows to between 3 and 10.

The loss is recoverable because the summaries are *derived*. Their source is the `.pos`
solutions, which the sweep never touched, so rebuilding needs no PPPx run, no products
download and no solver - only the same arithmetic that produced the rows in the first
place. That arithmetic is `stec.positioning.metrics`, which Gate E has already checked
against the recorded values on 96 station-days and matched to within the CSV's own
`%.4f` rounding floor.

Both weight-optimisation summaries can be rebuilt this way: `daily_summary_iono.csv`
(iono weighting, `model_iono`/`gim_iono` arms) and `daily_summary.csv` (elev weighting,
`model`/`gim` arms) - see `stec.analysis.positioning_coverage.SUMMARY_FILE` for the same
pairing. `--weighting` selects which. **Corrected 2026-09-17**: the iono rebuild used to
pair with the elevation-weighted `gim` directory instead of `gim_iono` - the default
(`iono`) no longer reproduces that original, buggy pairing; see `IONO_METHODS`'s comment.

Scoped to the three canonical model trees `stec.analysis.positioning_coverage` reads
(`CANONICAL_STEC_SUFFIX`/`CANONICAL_VTEC_SUFFIX`/`CANONICAL_PRETRAINED_DIR`), not every
one of the 1588+ experiment directories on disk - a non-canonical hyperparameter variant
does not feed any canonical table, so there is nothing to gain and real risk in touching
it here.

Safety, because this is the one operation in the rebuild that writes into the primary
checkout:

* dry run unless `--apply`, every file backed up beside itself before it is replaced,
  dated to the run that touched it (`*.pre_fix_YYYYMMDD`) rather than a fixed name, so a
  file changed on two different days keeps a backup naming each day, and no run ever
  assumes an old backup from a previous version of this script is still the original;
* a refusal to write a summary that would come out *smaller* than the one already on
  disk, once the known-contaminated rows (elevation-weighted `gim` rows in an iono
  summary - see `FOREIGN_GIM_METHOD`) are excluded from that count - shrinking any
  genuine row is the failure being repaired, so a repair that shrinks one is a bug in
  the repair; removing the known contamination is not a shrink, and is reported
  separately (see `count_foreign_gim_rows`);
* a refusal to write when the rebuild would *change the value* of any other row already
  on disk (compared at the CSV's own 4-decimal write precision) - a row that survived the
  original truncation is the only ground truth this script has for whether the rebuild
  arithmetic agrees with what actually produced the file, so a disagreement there means
  something about the rebuild itself is wrong for that day, not that the file should be
  overwritten anyway;
* a refusal to rebuild a day with no SINEX ground truth at all - `aggregate_daily_metrics`
  would otherwise fall back to a day-mean reference (`ref_source="mean"`, not a true
  positioning error) rather than raising, and a canonical summary must never silently
  carry that fallback where a ground-truth row is expected.

    python verification/repair_overwritten_summaries.py                       # dry run, iono
    python verification/repair_overwritten_summaries.py --weighting both --since all
    python verification/repair_overwritten_summaries.py --weighting elev --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stec.analysis.positioning_coverage import (  # noqa: E402
    CANONICAL_PRETRAINED_DIR,
    CANONICAL_STEC_SUFFIX,
    CANONICAL_VTEC_SUFFIX,
)
from stec.config import paths  # noqa: E402
from stec.positioning import metrics as pm  # noqa: E402

# The paper's three canonical result trees - the same patterns
# `stec.analysis.positioning_coverage.METHOD_TREES["canonical"]` glob, repeated here
# rather than imported as a dict because this script needs the bare glob prefixes, not
# the (model -> pattern) mapping that module builds them into.
CANONICAL_TREE_PATTERNS = (
    f"Finetune_STEC_2024_*_{CANONICAL_STEC_SUFFIX}",
    f"Finetune_VTEC_2024_*_{CANONICAL_VTEC_SUFFIX}",
    CANONICAL_PRETRAINED_DIR,
)

# Corrected 2026-09-17: the GIM arm used to be paired with the `gim` (elevation-
# weighted) directory even for the iono rebuild - a leftover from before the iono/elev
# split existed. That silently wrote elevation-weighted solutions into an iono summary
# (2026-09-17: DOY 200/AMC4 reads error_3d_rms 0.8314 m from `gim/` against the iono
# summary's own recorded, correct 0.7787 m from `gim_iono/`), and made every existing
# genuine `gim_iono` row look "dropped by the rebuild" (`find_value_mismatches` STOPped
# on exactly this in the 2026-09-17 dry run - see `IONO_METHODS`'s git history for the
# STOP log). The iono rebuild now reads `model_iono` + `gim_iono` only, matching
# `daily_summary_iono.csv`'s own two genuine arms; `count_foreign_gim_rows` and
# `find_value_mismatches` handle the resulting cleanup of existing `gim` rows an iono
# summary should never have carried (see their docstrings). Each tree aggregates only
# its own directories - reconciling a cross-tree disagreement between two trees' GIM
# runs is `stec.analysis.positioning_coverage`'s job (see its own
# `find_gim_disagreements`), not this script's.
IONO_SUMMARY = "daily_summary_iono.csv"
IONO_METHODS = (("model_iono", "model_iono"), ("gim_iono", "gim_iono"))

ELEV_SUMMARY = "daily_summary.csv"
ELEV_METHODS = (("model", "model"), ("gim", "gim"))

WEIGHTINGS = {
    "iono": {"summary": IONO_SUMMARY, "methods": IONO_METHODS},
    "elev": {"summary": ELEV_SUMMARY, "methods": ELEV_METHODS},
}

# An iono summary's only known contamination: elevation-weighted `gim` rows written by
# the pre-2026-09-17 pairing bug above. `None` for elev - an elev summary carrying a
# `gim_iono` row has never been observed, so there is nothing to strip automatically;
# see `count_foreign_gim_rows`.
FOREIGN_GIM_METHOD = {"iono": "gim", "elev": None}

# Dated per run (not a fixed name) so a file changed on two different days always gets
# a backup naming the day it was actually touched, rather than trusting whatever
# `.pre_repair` happens to already sit beside it from an earlier version of this script.
BACKUP_SUFFIX = f".pre_fix_{datetime.now().strftime('%Y%m%d')}"

# Columns compared by `find_value_mismatches`. Every numeric metric column - everything
# but the three identity/categorical ones, which have their own equality rule.
_IDENTITY_COLUMNS = {"station", "method", "year", "doy"}
_CATEGORICAL_COLUMNS = {"ref_source"}


class MissingSinexError(Exception):
    """No SINEX ground truth exists for this day.

    `aggregate_daily_metrics` accepts `snx_file=None` and falls back to a day-mean
    reference (`ref_source="mean"`) for that legitimate case - a live PPPx run with no
    published SINEX for the day. This repair only ever rebuilds a summary that
    (truncation aside) is supposed to already carry `ref_source="ground_truth"` rows, so
    silently accepting the day-mean fallback here would replace missing rows with rows
    that are the wrong *kind* of number, not merely fewer of them. Raised instead so the
    caller reports and skips the day rather than writing it.
    """


def find_snx(results_dir: Path) -> Path:
    """The SINEX file a `positioning/results/<year><doy>` directory's own day uses."""
    stamp = results_dir.name
    year, doy = int(stamp[:4]), int(stamp[4:])
    return (
        results_dir.parents[1]
        / "evaluation"
        / stamp
        / "products"
        / f"IGS0OPSSNX_{year}{doy:03d}0000_01D_01D_CRD.SNX"
    )


def _parse_since(since: str) -> float:
    """`since` as a Unix timestamp floor. `"all"` (case-insensitive) scans every file
    regardless of mtime, without callers needing to know or type an epoch-spanning date."""
    if since.strip().lower() == "all":
        return 0.0
    return datetime.fromisoformat(since).timestamp()


def summary_dirs(root: Path, since: str, summary_name: str) -> list[Path]:
    """Every canonical results directory whose `summary_name` was rewritten after `since`."""
    cutoff = _parse_since(since)
    found = {
        path.parent
        for pattern in CANONICAL_TREE_PATTERNS
        for path in root.glob(f"{pattern}/positioning/results/2024*/{summary_name}")
        if path.stat().st_mtime >= cutoff
    }
    return sorted(found)


def rebuild(results_dir: Path, weighting: str) -> pd.DataFrame | None:
    """Recompute `weighting`'s summary for one results directory from its `.pos` files.

    Raises `MissingSinexError` if the day has no SINEX file - see that class's docstring
    for why this does not fall back to `snx_file=None`.
    """
    stamp = results_dir.name
    year, doy = int(stamp[:4]), int(stamp[4:])

    snx = find_snx(results_dir)
    if not snx.exists():
        raise MissingSinexError(f"no SINEX file at {snx}")

    frames = []
    for subdir, method in WEIGHTINGS[weighting]["methods"]:
        if not (results_dir / subdir).is_dir():
            continue
        frame = pm.aggregate_daily_metrics(
            results_dir / subdir,
            year=year,
            doy=doy,
            method_name=method,
            snx_file=snx,
        )
        if frame is not None and len(frame):
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def count_foreign_gim_rows(existing: pd.DataFrame, weighting: str) -> int:
    """Existing rows that do not belong in this weighting's summary at all.

    Currently only the elevation-weighted `gim` rows the pre-2026-09-17 pairing bug
    wrote into iono summaries (see `IONO_METHODS`'s comment) - not a real disagreement
    to STOP on, and not a row the rebuild is expected to reproduce. Reported separately
    from `find_value_mismatches` so `main()` can print how many were removed per file
    without conflating a known, intentional removal with a genuine mismatch.
    """
    foreign_method = FOREIGN_GIM_METHOD[weighting]
    if foreign_method is None or existing.empty:
        return 0
    return int((existing["method"] == foreign_method).sum())


def _is_ref_source_upgrade(old_ref: object, new_ref: object) -> bool:
    """The one ref_source change that is expected, not a disagreement: a row first
    written with the day-mean fallback (no SINEX at the time) whose rebuild now has a
    genuine SINEX-referenced value, because SINEX has since become available for that
    day. Every other direction - staying `mean`, staying `ground_truth`, or genuinely
    *losing* SINEX (`ground_truth` -> `mean`) - is not this and must still be compared
    normally."""
    return old_ref == "mean" and new_ref == "ground_truth"


def find_ref_source_upgrades(
    existing: pd.DataFrame, rebuilt: pd.DataFrame
) -> list[dict]:
    """Rows `find_value_mismatches` lets through under the mean -> ground_truth
    allowance (see `_is_ref_source_upgrade`), reported separately so `main()` can log
    exactly what changed - old and new `error_3d_rms` - rather than folding an expected,
    desirable change into a generic "repaired" count next to genuine bug fixes.
    """
    if existing.empty:
        return []
    rebuilt_keyed = rebuilt.set_index(["station", "method"])
    upgrades = []
    for _, row in existing.iterrows():
        if row.get("ref_source") != "mean":
            continue
        key = (row["station"], row["method"])
        if key not in rebuilt_keyed.index:
            continue
        new_row = rebuilt_keyed.loc[key]
        if isinstance(new_row, pd.DataFrame):
            new_row = new_row.iloc[0]
        if not _is_ref_source_upgrade(row.get("ref_source"), new_row.get("ref_source")):
            continue
        upgrades.append(
            {
                "station": row["station"],
                "method": row["method"],
                "old_error_3d_rms": float(row["error_3d_rms"]),
                "new_error_3d_rms": float(new_row["error_3d_rms"]),
            }
        )
    return upgrades


def find_missing_sinex_drops(
    existing: pd.DataFrame, rebuilt: pd.DataFrame, weighting: str
) -> list[dict]:
    """Existing `ref_source="mean"` rows the rebuild correctly drops rather than
    reproduces, because that specific station has no SINEX entry for this day.

    2026-09-18 follow-up: the original mean -> ground_truth allowance
    (`find_ref_source_upgrades`) operated at whole-file granularity through
    `find_value_mismatches`'s generic "on disk but dropped by the rebuild" check - one
    station with no SINEX for the day (a real, unrelated condition; recurring stations
    include LICC/MAR7/UCLU/KIR8/BRMG/NAUS) made the whole file STOP, blocking the repair
    for every *other* station in it too. Applied at scale this gutted the common set for
    DOY 122-151, including the 10-11 May 2024 superstorm (DOY 131-132), because those
    files' many genuinely-repairable rows never got a chance to be rebuilt.

    `aggregate_daily_metrics` never falls back to a day-mean value for one station once
    a SINEX file is given at all (`require_snx and not ref_pos: skip` -
    `stec/positioning/metrics.py`) - a station absent from the rebuild's output while a
    SINEX file was used for the day is not a rebuild bug, it is the correct absence of
    ground truth for that one station. Dropping the row is therefore safe *only* when
    the existing row was already `ref_source="mean"` (never valid ground truth to begin
    with, and `positioning_coverage.collect()`'s ref_source filter would exclude it from
    the common set regardless) - a dropped `ground_truth` row is a genuine regression
    and is not covered here, so `find_value_mismatches` still flags it.

    `weighting`'s foreign-gim rows (see `FOREIGN_GIM_METHOD`/`count_foreign_gim_rows`)
    are excluded first, the same way `find_value_mismatches` excludes them - they are
    *always* absent from a same-weighting rebuild's output because they belong to a
    different weighting entirely, not because their station lacks SINEX, so counting
    them here would mislabel known contamination as a missing-SINEX drop and
    double-subtract it from the shrink floor alongside `count_foreign_gim_rows`. Found
    live in the 2026-09-18 dry run: elevation-weighted 'gim' rows for stations that do
    have SINEX (WARK, VALD, ...) were showing up under "no SINEX for that station"
    inside `[iono]` runs.
    """
    if existing.empty:
        return []
    foreign_method = FOREIGN_GIM_METHOD[weighting]
    if foreign_method is not None:
        existing = existing[existing["method"] != foreign_method]
        if existing.empty:
            return []
    rebuilt_keyed = rebuilt.set_index(["station", "method"])
    drops = []
    for _, row in existing.iterrows():
        if row.get("ref_source") != "mean":
            continue
        key = (row["station"], row["method"])
        if key in rebuilt_keyed.index:
            continue  # present in the rebuild - find_ref_source_upgrades's row, not this
        drops.append(
            {
                "station": row["station"],
                "method": row["method"],
                "old_error_3d_rms": float(row["error_3d_rms"]),
            }
        )
    return drops


def find_value_mismatches(
    existing: pd.DataFrame, rebuilt: pd.DataFrame, weighting: str, atol: float = 1e-3
) -> list[str]:
    """Rows already on disk that `rebuilt` does not reproduce.

    Compared at the CSV's own `%.4f` write precision (rounding both sides to 4 decimals
    before differencing), not raw float64 equality - the existing file was written at
    that precision, so anything tighter would flag the write's own rounding as a
    mismatch. A key present on disk but absent from the rebuild (a station the rebuild
    dropped entirely) is reported the same way as a value that changed - both mean the
    rebuild's arithmetic disagrees with the row's own history, which is exactly the
    condition `--apply` must refuse rather than paper over.

    Three deliberate exceptions:
    * an iono summary's `gim` rows (see `count_foreign_gim_rows`) are known
      contamination the rebuild correctly never reproduces, so they are excluded from
      this comparison rather than reported as "dropped" - `main()` reports their
      removal separately instead;
    * a `ref_source="mean"` -> `"ground_truth"` upgrade (see `_is_ref_source_upgrade`
      and `find_ref_source_upgrades`) is expected to change every numeric column on
      that row - the whole point is that SINEX has since become available - so the row
      is skipped here entirely rather than compared, and `main()` logs it via
      `find_ref_source_upgrades` instead;
    * a `ref_source="mean"` row the rebuild drops entirely because that one station has
      no SINEX entry for the day (see `find_missing_sinex_drops`) is not reported as
      "dropped" either - it was never valid ground truth, so its disappearance is not a
      regression, and `main()` logs it via `find_missing_sinex_drops` instead. A dropped
      `ref_source="ground_truth"` row is a different, genuine regression and is not
      covered by this exception.
    Every other ref_source change (including `ground_truth` -> `mean`) still STOPs
    exactly as before.
    """
    if existing.empty:
        return []
    foreign_method = FOREIGN_GIM_METHOD[weighting]
    if foreign_method is not None:
        existing = existing[existing["method"] != foreign_method]
        if existing.empty:
            return []
    rebuilt_keyed = rebuilt.set_index(["station", "method"])
    numeric_columns = [
        column
        for column in existing.columns
        if column not in _IDENTITY_COLUMNS and column not in _CATEGORICAL_COLUMNS
    ]

    mismatches = []
    for _, row in existing.iterrows():
        key = (row["station"], row["method"])
        if key not in rebuilt_keyed.index:
            if row.get("ref_source") == "mean":
                continue  # missing SINEX for this one station - see find_missing_sinex_drops
            mismatches.append(f"{key}: on disk but dropped by the rebuild")
            continue
        new_row = rebuilt_keyed.loc[key]
        if isinstance(new_row, pd.DataFrame):
            new_row = new_row.iloc[0]

        old_ref = row.get("ref_source")
        new_ref = new_row.get("ref_source")
        if _is_ref_source_upgrade(old_ref, new_ref):
            continue
        if old_ref != new_ref:
            mismatches.append(f"{key}: ref_source {old_ref!r} -> {new_ref!r}")
            continue

        for column in numeric_columns:
            if column not in new_row.index:
                continue
            old_value = round(float(row[column]), 4)
            new_value = round(float(new_row[column]), 4)
            if abs(old_value - new_value) > atol:
                mismatches.append(f"{key} {column}: {old_value} -> {new_value}")
    return mismatches


def _read_existing(summary_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=paths.LEGACY_EXPERIMENTS)
    parser.add_argument(
        "--since",
        default="2026-08-21 00:00",
        help="rebuild summaries rewritten after this (ISO datetime, or 'all' for every "
        "file regardless of mtime)",
    )
    parser.add_argument(
        "--weighting",
        choices=["iono", "elev", "both"],
        default="iono",
        help="which weight-optimisation summary to rebuild (default: iono, matching "
        "this script's original behaviour)",
    )
    parser.add_argument("--apply", action="store_true", help="write; otherwise dry run")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    weightings = ["iono", "elev"] if args.weighting == "both" else [args.weighting]

    targets_by_weighting = {
        weighting: summary_dirs(
            args.experiments, args.since, WEIGHTINGS[weighting]["summary"]
        )
        for weighting in weightings
    }
    total = sum(len(targets) for targets in targets_by_weighting.values())
    print(
        f"{'APPLY' if args.apply else 'DRY RUN'}: {total} damaged summary/summaries\n"
    )

    repaired = skipped = refused = missing_snx = ref_source_upgrades = 0
    missing_sinex_dropped_rows = 0
    for weighting in weightings:
        summary_name = WEIGHTINGS[weighting]["summary"]
        targets = targets_by_weighting[weighting]
        if args.limit:
            targets = targets[: args.limit]

        for results_dir in targets:
            summary_path = results_dir / summary_name
            existing = _read_existing(summary_path)
            before = len(existing)
            foreign_gim_count = count_foreign_gim_rows(existing, weighting)
            # The floor a correct rebuild must clear: every existing row minus the ones
            # known to be contamination (see FOREIGN_GIM_METHOD) that the rebuild is
            # supposed to drop, not reproduce.
            expected_floor = before - foreign_gim_count
            label = f"[{weighting}] {results_dir.parent.parent.parent.name[:34]:34s} {results_dir.name}"

            try:
                frame = rebuild(results_dir, weighting)
            except MissingSinexError as exc:
                print(f"  STOP    {label}  {exc}")
                missing_snx += 1
                continue

            if frame is None or frame.empty:
                print(f"  SKIP    {label}  no .pos files could be reduced")
                skipped += 1
                continue

            after = len(frame)
            # A ref_source='mean' row the rebuild drops because that one station has no
            # SINEX entry for the day is an expected, per-station shrink - not the
            # regression the floor below exists to catch (see find_missing_sinex_drops).
            missing_sinex_drops = find_missing_sinex_drops(existing, frame, weighting)
            adjusted_floor = expected_floor - len(missing_sinex_drops)
            if after < adjusted_floor:
                # Shrinking beyond the known, intentional gim-row removal and the
                # per-station missing-SINEX drops is the damage; a repair that shrinks
                # the genuine rows is a bug in the repair.
                print(
                    f"  REFUSE  {label}  {before} -> {after} rows "
                    f"({foreign_gim_count} elev-weighted gim row(s), "
                    f"{len(missing_sinex_drops)} missing-SINEX row(s) expected "
                    "removed), would shrink"
                )
                refused += 1
                continue
            if after == before and foreign_gim_count == 0 and not missing_sinex_drops:
                continue  # unchanged - nothing to report or touch

            mismatches = find_value_mismatches(existing, frame, weighting)
            if mismatches:
                print(f"  STOP    {label}  {before} -> {after} rows, but disagrees on:")
                for mismatch in mismatches[:10]:
                    print(f"            {mismatch}")
                if len(mismatches) > 10:
                    print(f"            ... and {len(mismatches) - 10} more")
                refused += 1
                continue

            upgrades = find_ref_source_upgrades(existing, frame)
            if upgrades:
                print(
                    f"  {label}  {len(upgrades)} ref_source mean->ground_truth "
                    "upgrade(s):"
                )
                for upgrade in upgrades[:10]:
                    print(
                        f"            {upgrade['station']}/{upgrade['method']}: "
                        f"{upgrade['old_error_3d_rms']:.4f} -> "
                        f"{upgrade['new_error_3d_rms']:.4f}"
                    )
                if len(upgrades) > 10:
                    print(f"            ... and {len(upgrades) - 10} more")
                ref_source_upgrades += len(upgrades)

            if missing_sinex_drops:
                print(
                    f"  {label}  {len(missing_sinex_drops)} row(s) dropped, no SINEX "
                    "for that station:"
                )
                for drop in missing_sinex_drops[:10]:
                    print(
                        f"            {drop['station']}/{drop['method']}: "
                        f"{drop['old_error_3d_rms']:.4f} -> dropped"
                    )
                if len(missing_sinex_drops) > 10:
                    print(f"            ... and {len(missing_sinex_drops) - 10} more")
                missing_sinex_dropped_rows += len(missing_sinex_drops)

            removed_note = (
                f", removing {foreign_gim_count} elev-weighted gim row(s)"
                if foreign_gim_count
                else ""
            ) + (
                f", dropping {len(missing_sinex_drops)} missing-SINEX row(s)"
                if missing_sinex_drops
                else ""
            )
            print(
                f"  {'repair' if args.apply else 'would':7s} {label}  "
                f"{before} -> {after} rows{removed_note}"
            )
            if args.apply:
                backup = summary_path.with_suffix(summary_path.suffix + BACKUP_SUFFIX)
                if not backup.exists():
                    shutil.copy2(summary_path, backup)
                frame.to_csv(summary_path, index=False, float_format="%.4f")
            repaired += 1

    print(
        f"\n  {repaired} repairable, {skipped} skipped, {refused} refused, "
        f"{missing_snx} missing SINEX, {ref_source_upgrades} ref_source "
        f"mean->ground_truth upgrade(s), {missing_sinex_dropped_rows} row(s) dropped "
        "for a station missing SINEX"
    )
    if not args.apply:
        print("  nothing written - rerun with --apply")
    else:
        print(f"  originals kept beside each file as *{BACKUP_SUFFIX}")
    print(f"  recorded at {datetime.now(timezone.utc).isoformat()}")
    return 1 if (refused or missing_snx) else 0


if __name__ == "__main__":
    sys.exit(main())
