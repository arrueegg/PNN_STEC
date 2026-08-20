"""The stage list, and the invariants that hold across it.

These checks run at startup rather than at write time, because the failures they catch -
two stages writing the same file, two stages claiming to be the source for Table 3 - are
configuration mistakes that should never reach a compute run.
"""

from __future__ import annotations

from .stage import Stage

# Populated as layers are ported. Each entry moves here from src/pipeline/stages.py once
# its analysis has a declared owner in the rebuilt package.
STAGES: list[Stage] = []


def by_name(stages: list[Stage] | None = None) -> dict[str, Stage]:
    return {stage.name: stage for stage in (STAGES if stages is None else stages)}


def check_unique_outputs(stages: list[Stage] | None = None) -> None:
    """No two stages may claim the same output - that is how duplicates start."""
    owner: dict[str, str] = {}
    for stage in STAGES if stages is None else stages:
        for output in stage.outputs:
            if output in owner:
                raise ValueError(
                    f"{output} is claimed by both '{owner[output]}' and '{stage.name}'"
                )
            owner[output] = stage.name


def check_unique_canonical(stages: list[Stage] | None = None) -> None:
    """One stage per deliverable.

    Tables 3 and 4 previously existed in three result trees that disagreed, and choosing
    between them meant consulting prose. A deliverable with two claimants has no answer to
    "where does this number come from", which is the whole point of the rebuild.
    """
    owner: dict[str, str] = {}
    for stage in STAGES if stages is None else stages:
        if stage.canonical_for is None:
            continue
        if stage.canonical_for in owner:
            raise ValueError(
                f"'{stage.canonical_for}' is claimed as canonical by both "
                f"'{owner[stage.canonical_for]}' and '{stage.name}'"
            )
        owner[stage.canonical_for] = stage.name


def check_inputs_are_produced_or_external(stages: list[Stage] | None = None) -> None:
    """A stage may not depend on an output that no stage produces and no data provides.

    Catches the rename that silently orphans a dependency: the consuming stage keeps
    pointing at a path nothing writes any more, and only fails once its input has aged out
    of the filesystem.
    """
    selected = STAGES if stages is None else stages
    produced = {output for stage in selected for output in stage.outputs}
    order = {stage.name: i for i, stage in enumerate(selected)}

    for stage in selected:
        for dependency in stage.inputs:
            if dependency not in produced:
                # External data, or a legacy tree being migrated. Not this check's business.
                continue
            producer = next(s for s in selected if dependency in s.outputs)
            if order[producer.name] > order[stage.name]:
                raise ValueError(
                    f"'{stage.name}' consumes {dependency}, which '{producer.name}' "
                    f"produces later in the run order"
                )


def validate(stages: list[Stage] | None = None) -> None:
    check_unique_outputs(stages)
    check_unique_canonical(stages)
    check_inputs_are_produced_or_external(stages)
