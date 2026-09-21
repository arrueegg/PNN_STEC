"""The top-level `cli.py`'s src/ dispatch fails clearly instead of with a raw traceback.

`cli.py`'s train/compare/inference/map/multiday subcommands all resolve their real work
through `_load_src_main`, added so a missing/deleted `src/` produces a named, actionable
message instead of a bare `ModuleNotFoundError` traceback pointing nowhere in particular -
the same "fail clearly, don't guess" pattern `tests/test_cli.py` already pins for
`stec/cli.py`'s own `_run_module`. This file has no `stec/` equivalent to sit next to
because `cli.py` lives at the repo root, not under `stec/` - it is the file being retired,
not a `stec/` module.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import cli

# src/ is not always present (it is being retired - see CLAUDE.md's "src/'s status"). The
# golden-path test below needs a real src/main.py to import, so it skips rather than fails
# once src/ is genuinely gone, matching the skipif pattern the six Gate-A equivalence tests
# already use for the same reason.
SRC_MAIN_AVAILABLE = (
    Path(__file__).resolve().parent.parent / "src" / "main.py"
).exists()


@pytest.mark.skipif(
    not SRC_MAIN_AVAILABLE, reason="src/main.py not present in this checkout"
)
def test_load_src_main_returns_the_real_entry_point_when_src_is_present():
    """The production path, unchanged: every queued training/inference run still gets
    exactly the `src/` module's own `main`, not a stec/ substitute."""
    main = cli._load_src_main("main", "train")
    assert callable(main)
    assert main.__module__ == "main"


def test_load_src_main_reports_a_missing_module_instead_of_a_raw_traceback(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli._load_src_main("not_a_real_module_xyz", "train")
    assert exc_info.value.code == 1

    message = capsys.readouterr().err
    assert "cli.py train needs src/not_a_real_module_xyz.py" in message
    assert "no stec/ replacement" in message


def test_load_src_main_does_not_swallow_a_module_that_imports_but_lacks_main(
    monkeypatch,
):
    """'Module not found' (clear message, exit 1) and 'module found but has no main' (an
    AttributeError, matching plain `from X import main` semantics) are different failures
    - the latter must not be silently folded into the former's friendlier message."""
    module = types.ModuleType("module_without_main")
    monkeypatch.setitem(sys.modules, "module_without_main", module)

    with pytest.raises(AttributeError):
        cli._load_src_main("module_without_main", "train")
