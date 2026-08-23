"""One entry point, with subcommands that own their arguments.

The existing `cli.py` dispatches by rebuilding `sys.argv` and calling each script's own
`main()`. Two consequences, both live in the current tree:

* `cli.py evaluate --experiment X` **ignores X**. It rebuilds argv, then calls a `main()`
  that never parses argv at all and instead reads `config/config_eval.yaml`, taking the
  experiment from there. In fact it does not even get that far: the module it imports is
  shadowed by the `evaluation/` package of the same name, whose lazy `__getattr__` exposes
  no `main`, so the command raises `ImportError`.
* `cli.py positioning` imports `inference_positioning`, which **does not exist anywhere in
  the repository**.

Both are the same failure: the CLI does not know what its subcommands need, so nothing
checks that it is passing it. Here each subcommand declares its own arguments and calls a
function with them. A missing module is an import error at startup rather than a runtime
surprise, and an argument that is accepted is an argument that is used.

Deliberately thin. Long-running work belongs to the pipeline runner, which knows what is
already up to date:

    python -m stec.pipeline status
    python -m stec.pipeline run --only daily_metrics
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


def _run_module(module: str, args: list[str]) -> int:
    """Invoke a ported module's `main()` with an explicit argument list.

    argv is set for the duration of the call rather than mutated globally, so a caller that
    dispatches twice in one process does not inherit the first call's arguments - which is
    the mechanism behind the two broken subcommands this replaces.
    """
    import importlib  # noqa: PLC0415

    loaded = importlib.import_module(module)
    entry: Callable[[], int] | None = getattr(loaded, "main", None)
    if entry is None:
        raise SystemExit(f"{module} has no main() to call")

    saved = sys.argv
    sys.argv = [module.replace(".", "/") + ".py", *args]
    try:
        return int(entry() or 0)
    finally:
        sys.argv = saved


def add_metrics(subparsers) -> None:
    parser = subparsers.add_parser(
        "metrics", help="per-day and pooled STEC metrics (Tables 3 and 4)"
    )
    parser.add_argument("--store-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--doys", type=int, nargs="*")

    def run(args: argparse.Namespace) -> int:
        passthrough: list[str] = []
        if args.store_root:
            passthrough += ["--store-root", str(args.store_root)]
        if args.output_dir:
            passthrough += ["--output-dir", str(args.output_dir)]
        if args.dataset:
            passthrough += ["--dataset", args.dataset]
        if args.doys:
            passthrough += ["--doys", *(str(d) for d in args.doys)]
        return _run_module("stec.analysis.daily_metrics", passthrough)

    parser.set_defaults(handler=run)


def add_tables(subparsers) -> None:
    parser = subparsers.add_parser(
        "tables", help="generate manuscript Tables 1 and 2 from a run config"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)

    def run(args: argparse.Namespace) -> int:
        passthrough = ["--config", str(args.config)]
        if args.output_dir:
            passthrough += ["--output-dir", str(args.output_dir)]
        return _run_module("stec.analysis.paper_tables", passthrough)

    parser.set_defaults(handler=run)


def add_manifest(subparsers) -> None:
    parser = subparsers.add_parser(
        "manifest", help="where every reported number comes from"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--strict", action="store_true", help="fail on any consistency problem"
    )

    def run(args: argparse.Namespace) -> int:
        passthrough: list[str] = []
        if args.output_dir:
            passthrough += ["--output-dir", str(args.output_dir)]
        if args.strict:
            passthrough.append("--strict")
        return _run_module("stec.analysis.results_manifest", passthrough)

    parser.set_defaults(handler=run)


def add_runs(subparsers) -> None:
    parser = subparsers.add_parser(
        "runs", help="index the experiment directories by run identity"
    )
    parser.add_argument("--experiments", type=Path)
    parser.add_argument("--output", type=Path)

    def run(args: argparse.Namespace) -> int:
        passthrough: list[str] = []
        if args.experiments:
            passthrough += ["--experiments", str(args.experiments)]
        if args.output:
            passthrough += ["--output", str(args.output)]
        return _run_module("stec.runs.build_alias_index", passthrough)

    parser.set_defaults(handler=run)


def add_pipeline(subparsers) -> None:
    parser = subparsers.add_parser(
        "pipeline", help="run or inspect the declared stages"
    )
    parser.add_argument("action", choices=["run", "status"])
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")

    def run(args: argparse.Namespace) -> int:
        passthrough = [args.action]
        if args.only:
            passthrough += ["--only", *args.only]
        if args.force:
            passthrough.append("--force")
        if args.keep_going:
            passthrough.append("--keep-going")
        return _run_module("stec.pipeline.runner", passthrough)

    parser.set_defaults(handler=run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stec", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for register in (add_pipeline, add_metrics, add_tables, add_manifest, add_runs):
        register(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
