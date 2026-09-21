"""What a stage declares about itself.

A stage is the unit that produces a result. Declaring inputs is what makes a rerun
skippable; declaring outputs is what makes duplication impossible; declaring assertions is
what turns a plausible-but-empty result into a failure.

Four of these fields exist specifically to remove ambiguity about what a number means,
each corresponding to a way this repository has previously confused itself:

* `canonical_for` - the deliverable this stage is the single source for. Tables 3 and 4
  once existed in three trees that disagreed, and knowing which to believe required
  reading a table in CLAUDE.md. Two stages claiming one deliverable is now a startup
  error, the same way two claiming one file is.
* `caveats` - conditions under which the output must not be read. `oracle_benchmark` is
  "not comparable with Table 5, by design and permanently"; Madrigal numbers "must be read
  alongside madrigal_reference_offset, never standalone". Recorded here, these travel into
  the artifact rather than living in prose that a reader of the CSV never sees.
* `supersedes` - older artifacts this replaces, stamped so a superseded number announces
  itself instead of sitting on disk looking current.
* `checks` - invariants that must hold for the result to be believable, as distinct from
  assertions that it merely exists. A shape check catches a truncated CSV; an invariant
  catches a plausible CSV full of wrong numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# An invariant takes the stage's outputs as a mapping of declared path to its record and
# returns None when satisfied, or a string explaining the violation.
Check = Callable[[dict], str | None]


@dataclass(frozen=True)
class Stage:
    name: str
    command: str
    answers: str
    description: str

    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    # Minimum rows an output CSV must carry to count as produced. A stage that writes a
    # header and nothing else is the failure mode this exists to catch.
    min_rows: dict[str, int] = field(default_factory=dict)

    canonical_for: str | None = None
    caveats: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)

    def __post_init__(self) -> None:
        undeclared = set(self.min_rows) - set(self.outputs)
        if undeclared:
            raise ValueError(
                f"stage '{self.name}' asserts row counts for paths it does not declare as "
                f"outputs: {sorted(undeclared)}"
            )
