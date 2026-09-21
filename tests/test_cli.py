"""Every subcommand's arguments reach the thing it dispatches to.

The two broken subcommands this replaces both failed the same way: the CLI accepted an
argument and the callee never saw it. These assert the opposite property directly.
"""

from __future__ import annotations

import sys

import pytest

from stec import cli


def test_every_registered_subcommand_has_a_handler():
    parser = cli.build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, type(parser._subparsers._group_actions[0]))
    )
    for name, sub in subparsers.choices.items():
        assert sub.get_default("handler") is not None, f"{name} has no handler"


def test_a_missing_command_is_an_error_not_a_silent_noop():
    with pytest.raises(SystemExit):
        cli.main([])


def test_an_unknown_command_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["not-a-command"])


def test_arguments_reach_the_dispatched_module(monkeypatch):
    """The defect being replaced: an accepted argument that the callee never sees."""
    seen: dict[str, list[str]] = {}

    def fake_run(module: str, args: list[str]) -> int:
        seen["module"] = module
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    cli.main(["tables", "--config", "some/config.yaml", "--output-dir", "out"])

    assert seen["module"] == "stec.analysis.paper_tables"
    assert "--config" in seen["args"]
    assert "some/config.yaml" in seen["args"]
    assert "out" in seen["args"]


def test_optional_arguments_are_omitted_rather_than_passed_empty(monkeypatch):
    seen: dict[str, list[str]] = {}
    monkeypatch.setattr(
        cli, "_run_module", lambda module, args: seen.setdefault("args", args) and 0
    )
    cli.main(["manifest"])
    assert seen["args"] == []


def test_doys_are_forwarded_individually(monkeypatch):
    seen: dict[str, list[str]] = {}

    def fake_run(module: str, args: list[str]) -> int:
        seen["args"] = args
        return 0

    monkeypatch.setattr(cli, "_run_module", fake_run)
    cli.main(["metrics", "--doys", "132", "133"])
    assert "132" in seen["args"] and "133" in seen["args"]


def test_argv_is_restored_after_dispatch(monkeypatch):
    """Two dispatches in one process must not inherit each other's arguments."""
    import types  # noqa: PLC0415

    module = types.ModuleType("fake_entry")
    module.main = lambda: 0  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fake_entry", module)

    before = list(sys.argv)
    cli._run_module("fake_entry", ["--flag"])
    assert sys.argv == before


def test_a_module_without_main_is_reported(monkeypatch):
    import types  # noqa: PLC0415

    module = types.ModuleType("no_entry")
    monkeypatch.setitem(sys.modules, "no_entry", module)
    with pytest.raises(SystemExit, match="no main"):
        cli._run_module("no_entry", [])


def test_pipeline_subcommand_requires_an_action():
    with pytest.raises(SystemExit):
        cli.main(["pipeline"])
