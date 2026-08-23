"""Run the declared stages, skipping whatever is already up to date.

A stage is up to date when its recorded input fingerprint matches the current one *and*
every declared output is still present with the digest that was recorded. The second half
matters as much as the first: a fingerprint match alone would happily "skip" a stage whose
output someone has since deleted or truncated.

Every run writes a provenance record - code version, command, input digests, output
digests and row counts - which is both the skip decision for next time and the answer to
"what produced this number".

Three things happen after the command and before the record is written, so that a stage
which exits zero without having actually produced its result is a failure rather than a
cached success:

1. assertions - every declared output exists and carries the rows it promised;
2. checks - the invariants that make the result believable, not merely present;
3. caveats and superseded markers are stamped beside the artifacts.

Usage::

    python -m stec.pipeline status            # what would run, and why
    python -m stec.pipeline run               # run what is out of date
    python -m stec.pipeline run --only daily_metrics --force
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from ..config import paths
from . import fingerprint, provenance
from .registry import STAGES, by_name, validate
from .stage import Stage

logger = logging.getLogger(__name__)


class AssertionFailed(RuntimeError):
    """A stage finished but did not produce what it declared."""


class CheckFailed(RuntimeError):
    """A stage produced its outputs, but an invariant on them does not hold."""


def outputs_intact(stage: Stage, record: dict) -> bool:
    """Every declared output still present, with the digest that was recorded."""
    recorded = record.get("outputs", {})
    for output in stage.outputs:
        current = provenance.output_record(Path(output))
        if not current.get("present"):
            return False
        was = recorded.get(output, {})
        if "sha256" in was and was.get("sha256") != current.get("sha256"):
            return False
    return True


def reason_to_run(stage: Stage, force: bool) -> str | None:
    """Why this stage must run, or None when it is up to date."""
    if force:
        return "forced"
    record = provenance.load(stage.name)
    if record is None:
        return "never run"
    if record.get("fingerprint") != fingerprint.fingerprint(stage.inputs, stage.params):
        return "inputs or parameters changed"
    if record.get("command") != stage.command:
        return "command changed"
    if not outputs_intact(stage, record):
        return "outputs missing or modified"
    return None


def check_assertions(stage: Stage) -> dict:
    """Outputs exist and carry at least the rows the stage promised."""
    records = {}
    for output in stage.outputs:
        path = Path(output)
        record = provenance.output_record(path)
        if not record.get("present"):
            raise AssertionFailed(f"{stage.name}: declared output missing: {output}")
        records[output] = record

    for output, minimum in stage.min_rows.items():
        record = records[output]
        rows = record.get("rows")
        if rows is None:
            raise AssertionFailed(f"{stage.name}: cannot count rows of {output}")
        if rows < minimum:
            raise AssertionFailed(
                f"{stage.name}: {output} has {rows} rows, expected at least {minimum}"
            )
    return records


def run_checks(stage: Stage, outputs: dict) -> None:
    """Invariants that decide whether the result is believable, not merely present."""
    for check in stage.checks:
        violation = check(outputs)
        if violation:
            name = getattr(check, "__name__", "check")
            raise CheckFailed(f"{stage.name}: {name}: {violation}")


def record_context(stage: Stage) -> None:
    """Stamp caveats beside each output, and mark whatever this stage supersedes."""
    for output in stage.outputs:
        provenance.write_caveats(Path(output), stage.name, stage.caveats)
    for older in stage.supersedes:
        provenance.mark_superseded(Path(older), stage.name, stage.outputs)


def run_stage(stage: Stage, reason: str) -> bool:
    logger.info(f"▶ {stage.name} ({reason})")
    started = time.time()
    before = fingerprint.fingerprint(stage.inputs, stage.params)

    result = subprocess.run(
        [sys.executable, *stage.command.split()], capture_output=True, text=True
    )
    duration = time.time() - started
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        logger.error(f"✗ {stage.name} exited {result.returncode}: {' / '.join(tail)}")
        return False

    try:
        outputs = check_assertions(stage)
        run_checks(stage, outputs)
    except (AssertionFailed, CheckFailed) as exc:
        logger.error(f"✗ {exc}")
        return False

    record_context(stage)
    provenance.save(
        stage.name,
        {
            "stage": stage.name,
            "command": stage.command,
            "answers": stage.answers,
            "description": stage.description,
            "canonical_for": stage.canonical_for,
            "caveats": stage.caveats,
            "supersedes": stage.supersedes,
            "code": provenance.code_version(),
            "fingerprint": before,
            "inputs": fingerprint.describe(stage.inputs),
            "outputs": outputs,
            "params": stage.params,
            "duration_s": round(duration, 1),
        },
    )
    rows = ", ".join(
        f"{Path(k).name}={v['rows']}" for k, v in outputs.items() if "rows" in v
    )
    logger.info(f"✓ {stage.name} in {duration:.0f}s" + (f" [{rows}]" if rows else ""))
    return True


def select(names: list[str] | None) -> list[Stage]:
    if not names:
        return STAGES
    known = by_name()
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(unknown)}")
    return [known[n] for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--only", nargs="+", help="stage names, in the order given")
    parser.add_argument("--force", action="store_true", help="rerun even if up to date")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    os.chdir(paths.REPO_ROOT)
    validate()
    stages = select(args.only)

    if not stages:
        print("  no stages declared yet")
        return

    if args.action == "status":
        width = max(len(s.name) for s in stages)
        for stage in stages:
            reason = reason_to_run(stage, args.force)
            print(f"  {stage.name:<{width}}  {reason or 'up to date'}")
        pending = sum(reason_to_run(s, args.force) is not None for s in stages)
        print(f"\n  {pending} of {len(stages)} stage(s) would run")
        return

    failed = []
    for stage in stages:
        reason = reason_to_run(stage, args.force)
        if reason is None:
            logger.info(f"· {stage.name} up to date")
            continue
        if not run_stage(stage, reason):
            failed.append(stage.name)
            if not args.keep_going:
                raise SystemExit(f"stopped at failing stage: {stage.name}")

    if failed:
        raise SystemExit(f"{len(failed)} stage(s) failed: {', '.join(failed)}")
    logger.info("all stages up to date")


if __name__ == "__main__":
    main()
