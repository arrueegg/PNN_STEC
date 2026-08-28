#!/usr/bin/env python3
"""Elevation-weighting PPPx targeting for scripts/elev_positioning_chain.sh.

The station-recovery sweep that finished 2026-08-27 re-solved positioning under **iono**
weighting only (CLAUDE.md's canonical-results table). Every one of the 1,212 elev
``daily_summary.csv`` files under ``experiments/`` has a max mtime of 2026-08-20 13:54,
before the sweep started - so elev never saw the recovered population at all, on top of
never having been re-solved for anything the sweep's own iono-only pass changed.

**Corrections are reused, not regenerated.** ``run_positioning_evaluation.py`` never
writes ``positioning/stec_corrections/<year><doy>/<station>.csv`` itself - it only reads
one if present (see that script's ``process_single_station``) and warns if it is missing.
Those files already exist from the iono runs (STEC/VTEC's own per-day fine-tune, or the
Pretrained model's forward pass), and are identical regardless of ``--weight_opt``: only
the PPPx weighting differs, not the ionospheric correction fed into it. So the elev target
for one (arm, doy) is simply "every station with a correction CSV on disk that does not
already have an elev ``.pos`` file" - no correction generation, no model inference, ever
happens here.

**Canonical experiment resolution reuses ``positioning/geometry/recover_day.py``'s
``resolve_experiment()`` directly**, not a second definition. That function used to pick a
non-canonical hyperparameter-search variant via ``sorted(matches)[0]`` (the same bug
``scripts/lib/coverage_followon.py``'s module docstring documents and worked around by
reimplementing canonical selection itself) - commit ``faf06bd`` (2026-08-27) fixed
``resolve_experiment()`` in place to prefer the canonical directory explicitly
(``CANONICAL_STEC_SUFFIX`` / ``CANONICAL_VTEC_SUFFIX`` / ``CANONICAL_PRETRAINED_DIR``,
imported from ``stec.analysis.positioning_coverage``), falling back to the old sorted-glob
behaviour only - loudly - when the canonical directory itself has no checkpoint for that
DOY. Reusing it here (rather than re-copying ``coverage_followon.py``'s workaround) is
correct now that the bug it worked around is fixed, and keeps a single source of truth for
what "canonical" means.

Usage (invoked by scripts/elev_positioning_chain.sh, not normally run by hand)::

    python3 scripts/lib/elev_positioning_targets.py list-groups --out groups.tsv
    python3 scripts/lib/elev_positioning_targets.py summary --out groups.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from positioning.geometry.recover_day import (  # noqa: E402
    EXPERIMENT_PATTERNS,
    resolve_experiment,
)

YEAR = 2024
# The 2024 test period (CLAUDE.md: "Daily fine-tunes (258 days, DOY 122-366 of 2024)");
# resolve_experiment() returns None for any DOY a given arm has no canonical directory
# for, so this range does not need to be narrowed further here.
DOYS = range(122, 367)
ARMS = list(EXPERIMENT_PATTERNS)  # ["STEC", "VTEC", "Pretrained_STEC"]


def find_pending(kind: str, doy: int) -> tuple[Path, list[str]] | None:
    """The canonical experiment for (kind, doy) and the stations still needing an elev
    solve, or None if that arm has no canonical experiment for this DOY, no corrections
    yet, or nothing pending."""
    experiment = resolve_experiment(kind, doy)
    if experiment is None:
        return None

    day_tag = f"{YEAR}{doy:03d}"
    corrections_dir = experiment / "positioning" / "stec_corrections" / day_tag
    if not corrections_dir.is_dir():
        return None
    stations_with_correction = sorted(p.stem for p in corrections_dir.glob("*.csv"))
    if not stations_with_correction:
        return None

    elev_model_dir = experiment / "positioning" / "results" / day_tag / "model"
    pending = [
        station
        for station in stations_with_correction
        if not (elev_model_dir / station / f"{station}_model.pos").exists()
    ]
    return (experiment, pending) if pending else None


def iter_groups() -> list[tuple[int, str, str, list[str]]]:
    """One (doy, arm, exp_name, stations) group per arm-day with pending elev work,
    doy-major so scripts/elev_positioning_chain.sh's RINEX-sharing loop can rely on
    every arm for a given day being contiguous."""
    groups = []
    for doy in DOYS:
        for kind in ARMS:
            found = find_pending(kind, doy)
            if found is None:
                continue
            experiment, pending = found
            groups.append((doy, kind, experiment.name, pending))
    return groups


def cmd_list_groups(args: argparse.Namespace) -> None:
    groups = iter_groups()
    lines = [
        f"{doy}\t{arm}\t{exp_name}\t{' '.join(stations)}"
        for doy, arm, exp_name, stations in groups
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + ("\n" if lines else ""))
    total_stations = sum(len(stations) for _, _, _, stations in groups)
    print(
        f"[elev_positioning_targets] {len(groups)} (doy, arm) group(s), "
        f"{total_stations} station-instance(s) pending across "
        f"{len({doy for doy, *_ in groups})} day(s) -> {args.out}"
    )


def cmd_summary(args: argparse.Namespace) -> None:
    """Re-derive the same counts from an already-written groups file, for a cheap
    DRY_RUN / post-hoc report that does not re-walk experiments/."""
    lines = [line for line in args.out.read_text().splitlines() if line.strip()]
    by_arm: dict[str, int] = {}
    days: set[int] = set()
    total_stations = 0
    for line in lines:
        doy_str, arm, _exp_name, stations = line.split("\t")
        stations_list = stations.split()
        by_arm[arm] = by_arm.get(arm, 0) + len(stations_list)
        days.add(int(doy_str))
        total_stations += len(stations_list)
    print(
        f"[elev_positioning_targets] {len(lines)} group(s), {total_stations} "
        f"station-instance(s), {len(days)} day(s)"
    )
    for arm, count in sorted(by_arm.items()):
        print(f"  {arm}: {count} station-instance(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser(
        "list-groups", help="write the (doy, arm, exp_name, stations) TSV"
    )
    p_list.add_argument("--out", type=Path, required=True)
    p_list.set_defaults(func=cmd_list_groups)

    p_summary = sub.add_parser(
        "summary", help="summarise an already-written groups TSV"
    )
    p_summary.add_argument("--out", type=Path, required=True)
    p_summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
