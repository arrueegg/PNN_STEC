"""Restructure `multiday_results/` from a flat, 300+-entry tree into the layout
`docs/revision/results_layout.md` describes, by renaming top-level directories - never by
copying their contents.

Before this layout, `multiday_results/` held 312 directories at depth 1: 246 per-day
`2024_DOY_*` sweep trees, ~20 `stec.analysis` outputs, ~40 positioning runs and a dozen
superseded records, all as siblings. The new layout groups each by what it structurally
*is* - a rename is the only operation this script performs, and on one filesystem a
directory rename is metadata-only, not a copy: `os.rename` refuses (`OSError`, not a
silent copy-then-delete) if source and destination ever end up on different devices, which
is exactly the failure mode "never copy 640 GB" exists to catch.

Classification is layered, most specific first, and reuses two things this repository
already computes rather than re-deriving them:

1. **Superseded.** `stec.runs.migrate.build_plan` already encodes CLAUDE.md's
   canonical/superseded table (including the one glob tree that is canonical, not
   superseded, despite matching the superseded glob). A directory that table names
   superseded moves to `superseded/<name>/`, regardless of what kind of tree it is -
   being retired changes where a reader is told to look, not what the tree structurally
   is.
2. **A declared analysis output.** `stec.pipeline.registry.STAGES` already says, for every
   `stec.analysis` stage, its exact declared output path and whether its command invokes
   `stec/` (rebuilt) or a pre-rebuild script. Reading that - rather than guessing from a
   `_rebuilt` suffix - is what resolves the one irregular case (`repair_gim_baseline`'s
   output is named `gim_baseline_repair`, not `repair_gim_baseline_rebuilt`) and the two
   permanently-one-variant cases (`paper_tables`, `hyperparameter_search`) without a hand
   list of exceptions.
3. **A per-day payload.** `2024_DOY_<ddd>`, the same pattern
   `stec.analysis.results_manifest.DOY_DIR_PATTERN` already matches.
4. **A structural fallback.** A directory that contains its own `2024_DOY_*` children (the
   shape `with_pretrained_baseline` and the `store_sweep_*` trees have) is an STEC
   evaluation sweep; a directory whose name starts with `positioning` is a positioning run.
5. **Unclassified.** Anything left over moves to `unclassified/<name>/`, unchanged in
   content, flagged rather than silently placed - `stratified_comparison_pretrained` is
   the one real example on disk: its shape matches `stratified_comparison`'s output but
   its name does not, and guessing would risk merging two different result sets.

The plan already skips a re-run's own output: a top-level entry named `per_day`,
`stec_evaluation`, `analyses`, `positioning`, `superseded` or `unclassified` is the layout
itself, not something to classify, so running this script twice with nothing new on disk
plans zero moves the second time.

Usage::

    python -m stec.runs.restructure_results                  # dry run, LEGACY_MULTIDAY
    python -m stec.runs.restructure_results --source-root PATH
    python -m stec.runs.restructure_results --apply           # writes the moves + a manifest
    python -m stec.runs.restructure_results --undo MANIFEST   # reverses one recorded run
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..analysis.results_manifest import DOY_DIR_PATTERN, day_directories
from ..config import paths
from ..pipeline import fingerprint
from ..pipeline.registry import STAGES
from .migrate import build_plan as legacy_canonical_and_superseded_plan

# The layout's own top-level directories - present once this script has run, and never a
# thing to classify on a later run. Names come from `stec.config.paths`, not a literal
# copy, so the two never drift: this script must work against an arbitrary root (a
# synthetic tmp_path tree in tests, the read-only legacy checkout, or - one day -
# `paths.RESULTS_ROOT` itself), which is exactly why it takes the root as a parameter
# instead of importing `paths.PER_DAY_RESULTS` and friends directly as destinations.
BUCKET_NAMES = frozenset(
    {
        paths.PER_DAY_RESULTS.name,
        paths.STEC_EVALUATION_RESULTS.name,
        paths.ANALYSES_RESULTS.name,
        paths.POSITIONING_RESULTS.name,
        paths.SUPERSEDED_RESULTS.name,
        paths.UNCLASSIFIED_RESULTS.name,
    }
)

POSITIONING_PREFIX = "positioning_"


@dataclass(frozen=True)
class Move:
    source: Path
    dest: Path
    bucket: str
    tag: str


# --- classification ---------------------------------------------------------------------


# The one stage whose historical flat directory name does not match its stage name.
# Every other analysis's old `<name>` / `<name>_rebuilt` pair is derived directly from
# `stage.name` below - `stages.py` itself only records the *new* nested location
# (`paths.analysis_result_dir`), so the flat name being migrated away from cannot be read
# back out of `stage.outputs` and has to be named here instead, the same way
# `stec.runs.migrate`'s canonical/superseded tables are hand-written rather than derived.
_IRREGULAR_LEGACY_DIRNAMES = {"repair_gim_baseline": "gim_baseline_repair"}


def _stage_output_dirnames() -> dict[str, tuple[str, bool]]:
    """`{legacy_flat_dirname: (stage_name, is_rebuilt)}` for every stage that owns a
    dedicated top-level analysis directory - the historical flat name each one wrote to
    before this layout existed.

    A stage's `command` starting with `-m stec.` means it invokes the ported `stec/`
    implementation, which historically wrote `<name>_rebuilt`; anything else
    (`src/analysis/...`) is a pre-rebuild script, which wrote the bare `<name>`. A stage
    only qualifies if `paths.analysis_result_dir(stage.name, ...)` is itself one of its
    declared outputs (equal to it, or an ancestor of it) - `uncertainty_calibration_
    pretrained` fails this on purpose: it writes *inside* `uncertainty_calibration`'s own
    directory rather than owning one under its own name, so it never had a flat legacy
    directory of its own to migrate. The counterpart name (the other implementation's
    output, if one happens to exist on disk) is not itself declared anywhere, so it is
    synthesised here and only ever fills a gap (`setdefault`) - a real declared entry,
    processed in either order, always wins.
    """
    mapping: dict[str, tuple[str, bool]] = {}
    for stage in STAGES:
        is_rebuilt = stage.command.startswith("-m stec.")
        expected_dir = paths.analysis_result_dir(stage.name, rebuilt=is_rebuilt)
        # A declared output is repository-relative (stages.py builds it that way so
        # .pipeline/*.json stays portable across clones - see stages.py's `_rel`), so it
        # is joined onto REPO_ROOT before comparing; `Path.__truediv__` leaves an already-
        # absolute `output` unchanged, so this handles either representation correctly
        # regardless of the caller's own current working directory.
        owns_it = any(
            (paths.REPO_ROOT / output) == expected_dir
            or expected_dir in (paths.REPO_ROOT / output).parents
            for output in stage.outputs
        )
        if not owns_it:
            continue

        flat_name = _IRREGULAR_LEGACY_DIRNAMES.get(stage.name, stage.name)
        rebuilt_name = f"{flat_name}_rebuilt"
        this_name = rebuilt_name if is_rebuilt else flat_name
        counterpart_name = flat_name if is_rebuilt else rebuilt_name
        mapping[this_name] = (stage.name, is_rebuilt)
        mapping.setdefault(counterpart_name, (stage.name, not is_rebuilt))
    return mapping


def _positioning_tag(name: str) -> str:
    if name.startswith(POSITIONING_PREFIX) and len(name) > len(POSITIONING_PREFIX):
        return name[len(POSITIONING_PREFIX) :]
    return name


def classify(top: Path, root: Path, superseded_paths: set[Path]) -> Move | None:
    """Where one top-level `multiday_results` entry belongs, or None to leave it alone.

    `root` is the tree being restructured (any `multiday_results`-shaped directory, not
    necessarily `paths.LEGACY_MULTIDAY`) - every destination is built under it, never
    under the fixed `paths.RESULTS_ROOT`, so the same classification is exact whether it
    is planning against the real legacy tree or a synthetic test fixture.
    """
    if top.name in BUCKET_NAMES:
        return None

    if top in superseded_paths:
        return Move(
            top, root / paths.SUPERSEDED_RESULTS.name / top.name, "superseded", top.name
        )

    doy_match = DOY_DIR_PATTERN.match(top.name)
    if doy_match:
        doy = int(doy_match.group(1))
        # DOY_DIR_PATTERN itself only matches "2024_DOY_<ddd>" - the year is baked into
        # that pattern already, not assumed fresh here.
        dest = root / paths.PER_DAY_RESULTS.name / "2024" / f"{doy:03d}"
        return Move(top, dest, "per_day", f"2024/{doy:03d}")

    stage_name, is_rebuilt = _stage_output_dirnames().get(top.name, (None, None))
    if stage_name is not None:
        variant = "rebuilt" if is_rebuilt else "pre_rebuild"
        dest = root / paths.ANALYSES_RESULTS.name / stage_name / variant
        return Move(top, dest, "analyses", f"{stage_name}/{variant}")

    if day_directories(top):
        dest = root / paths.STEC_EVALUATION_RESULTS.name / top.name
        return Move(top, dest, "stec_evaluation", top.name)

    if top.name.startswith(POSITIONING_PREFIX) or top.name == "positioning":
        tag = _positioning_tag(top.name)
        dest = root / paths.POSITIONING_RESULTS.name / tag
        return Move(top, dest, paths.POSITIONING_RESULTS.name, tag)

    dest = root / paths.UNCLASSIFIED_RESULTS.name / top.name
    return Move(top, dest, "unclassified", top.name)


def plan(source_root: Path) -> list[Move]:
    """Every move this run would make, in a stable order. Reads the filesystem; writes
    nothing - safe to call as many times as needed to inspect a tree.
    """
    if not source_root.is_dir():
        return []
    superseded_paths = {
        t.path
        for t in legacy_canonical_and_superseded_plan(
            source_root, source_root.parent / "predictions"
        )
        if t.category == "superseded"
    }
    moves = [
        move
        for top in sorted(p for p in source_root.iterdir() if p.is_dir())
        if (move := classify(top, source_root, superseded_paths)) is not None
    ]

    dests = [m.dest for m in moves]
    duplicates = {d for d in dests if dests.count(d) > 1}
    if duplicates:
        names = ", ".join(str(d) for d in sorted(duplicates))
        raise ValueError(
            f"two source directories would collide at the same destination: {names}"
        )
    return moves


# --- apply / undo ------------------------------------------------------------------------


def _write_manifest(source_root: Path, moves: list[Move]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = source_root / f"RESTRUCTURE_MANIFEST_{timestamp}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source_root),
                "moves": [
                    {
                        "source": str(m.source),
                        "dest": str(m.dest),
                        "bucket": m.bucket,
                        "tag": m.tag,
                    }
                    for m in moves
                ],
            },
            indent=2,
        )
    )
    return manifest_path


def apply_plan(source_root: Path, moves: list[Move]) -> Path:
    """Perform every move, then write the manifest that makes it reversible.

    Validates the whole batch - no destination already present, no destination shared
    between two sources - before touching anything, so a bad plan fails closed rather than
    leaving the tree half migrated. `Path.rename` is used directly rather than
    `shutil.move`: the latter falls back to copy-then-delete across filesystems, silently
    doing the one thing this script must never do to a 640 GB tree.
    """
    for move in moves:
        if move.dest.exists():
            raise FileExistsError(
                f"destination already exists, refusing to run: {move.dest}"
            )

    for move in moves:
        move.dest.parent.mkdir(parents=True, exist_ok=True)
        move.source.rename(move.dest)

    return _write_manifest(source_root, moves)


def undo(manifest_path: Path) -> list[Move]:
    """Reverse exactly the moves one manifest recorded, refusing if any original location
    is occupied by something new.
    """
    record = json.loads(manifest_path.read_text())
    moves = [
        Move(Path(m["dest"]), Path(m["source"]), m["bucket"], m["tag"])
        for m in record["moves"]
    ]
    for move in moves:
        if move.dest.exists():
            raise FileExistsError(
                f"original location is occupied, refusing to undo: {move.dest}"
            )
    for move in moves:
        move.dest.parent.mkdir(parents=True, exist_ok=True)
        move.source.rename(move.dest)
    return moves


# --- reporting -----------------------------------------------------------------------


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _print_plan(moves: list[Move]) -> None:
    by_bucket: dict[str, list[Move]] = {}
    for move in moves:
        by_bucket.setdefault(move.bucket, []).append(move)

    total_bytes = 0
    for bucket in sorted(by_bucket):
        bucket_moves = by_bucket[bucket]
        bucket_bytes = sum(
            fingerprint.digest(m.source).get("size", 0) for m in bucket_moves
        )
        total_bytes += bucket_bytes
        print(
            f"\n  {bucket}/  ({len(bucket_moves)} dir(s), {_format_size(bucket_bytes)})"
        )
        for move in bucket_moves:
            print(f"    {move.source.name:<40} -> {move.dest}")

    print(f"\n  {len(moves)} directory(ies) total, {_format_size(total_bytes)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=paths.LEGACY_MULTIDAY)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the moves; without this, dry run only",
    )
    parser.add_argument(
        "--undo",
        type=Path,
        default=None,
        metavar="MANIFEST",
        help="reverse one recorded run",
    )
    args = parser.parse_args()

    if args.undo is not None:
        moves = undo(args.undo)
        print(f"undone: {len(moves)} directory(ies) restored from {args.undo}")
        return 0

    moves = plan(args.source_root)
    if not moves:
        print(f"nothing to move under {args.source_root} - already in the new layout")
        return 0

    print("APPLYING" if args.apply else "DRY RUN - pass --apply to write anything")
    _print_plan(moves)

    if args.apply:
        manifest_path = apply_plan(args.source_root, moves)
        print(f"\n  manifest -> {manifest_path}")
        print(
            f"  undo with: python -m stec.runs.restructure_results --undo {manifest_path}"
        )
    else:
        print(
            "\n  re-run with --apply to write these moves (and a manifest to undo them)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
