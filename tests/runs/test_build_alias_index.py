"""`stec.runs.build_alias_index`: the CSV it writes and the duplicate-run_id detection.

`test_cli.py`'s `runs` subcommand tests monkeypatch `_run_module`, so they only pin argv
wiring and never execute this module's own logic - the CSV writing (`FIELDS`, the
`csv.DictWriter`) and the `Counter`-based duplicate check are otherwise untested. Both
matter: `docs/revision/rebuild_status.md` cites this index's "zero collisions" as a fact
about the real experiment tree, and a silently-broken duplicate check would keep reporting
that even if it stopped being true.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from stec.runs import build_alias_index, identity


def write_experiment(
    root: Path, name: str, config: dict, num_checkpoints: int = 0
) -> Path:
    """A synthetic experiment directory: config.yaml plus optional model/*.pth files."""
    experiment_dir = root / name
    experiment_dir.mkdir(parents=True)
    _write_yaml(experiment_dir / "config.yaml", config)
    if num_checkpoints:
        model_dir = experiment_dir / "model"
        model_dir.mkdir()
        for i in range(num_checkpoints):
            (model_dir / f"epoch_{i}.pth").write_text("checkpoint\n")
    return experiment_dir


def _write_yaml(path: Path, config: dict) -> None:
    # Every value used below is a plain str/int/bool, so a hand-rolled writer avoids
    # pulling in PyYAML's dumper just to serialise a couple of flat and one-deep dicts.
    lines = []
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            lines.extend(f"  {k}: {v!r}" for k, v in value.items())
        else:
            lines.append(f"{key}: {value!r}")
    path.write_text("\n".join(lines) + "\n")


def base_config(**overrides) -> dict:
    config = {
        "mode": "finetune",
        "target": "stec",
        "year": 2024,
        "doy": 132,
        "random_seed": 42,
        "model": {"model_type": "BayesianResNetSTEC"},
    }
    config.update(overrides)
    return config


def run_main(experiments_dir: Path, output_path: Path, monkeypatch) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_alias_index.py",
            "--experiments",
            str(experiments_dir),
            "--output",
            str(output_path),
        ],
    )
    return build_alias_index.main()


def read_rows(output_path: Path) -> list[dict]:
    with output_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# --- CSV columns and contents -----------------------------------------------------------


def test_csv_header_matches_declared_fields(tmp_path, monkeypatch):
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    write_experiment(experiments_dir, "exp_a", base_config())
    output_path = tmp_path / "alias_index.csv"

    exit_code = run_main(experiments_dir, output_path, monkeypatch)

    assert exit_code == 0
    with output_path.open(newline="") as handle:
        header = next(csv.reader(handle))
    assert header == build_alias_index.FIELDS


def test_csv_row_reports_the_resolved_run_id_and_checkpoint(tmp_path, monkeypatch):
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    config = base_config()
    write_experiment(experiments_dir, "exp_a", config, num_checkpoints=2)
    output_path = tmp_path / "alias_index.csv"

    run_main(experiments_dir, output_path, monkeypatch)

    (row,) = read_rows(output_path)
    assert row["exp_name"] == "exp_a"
    assert row["run_id"] == identity.run_id(config)
    assert row["status"] == "ok"
    assert row["mode"] == "finetune"
    assert row["target"] == "stec"
    assert row["model_type"] == "BayesianResNetSTEC"
    assert row["year"] == "2024"
    assert row["doy"] == "132"
    assert row["random_seed"] == "42"
    assert row["checkpoints"] == "2"
    assert row["checkpoint"] == "epoch_0.pth"


def test_csv_row_for_missing_config_reports_status_without_a_run_id(
    tmp_path, monkeypatch
):
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    (experiments_dir / "no_config_here").mkdir()
    output_path = tmp_path / "alias_index.csv"

    run_main(experiments_dir, output_path, monkeypatch)

    (row,) = read_rows(output_path)
    assert row["exp_name"] == "no_config_here"
    assert row["run_id"] == ""
    assert row["status"] == "no config.yaml"


# --- duplicate run_id detection ----------------------------------------------------------


def test_zero_duplicates_reports_no_shared_run_id(tmp_path, monkeypatch, capsys):
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    write_experiment(experiments_dir, "exp_a", base_config(random_seed=1))
    write_experiment(experiments_dir, "exp_b", base_config(random_seed=2))
    output_path = tmp_path / "alias_index.csv"

    run_main(experiments_dir, output_path, monkeypatch)

    rows = read_rows(output_path)
    assert len({row["run_id"] for row in rows}) == 2
    out = capsys.readouterr().out
    assert "shared by more than one directory" not in out


def test_two_directories_with_the_same_config_are_reported_as_duplicates(
    tmp_path, monkeypatch, capsys
):
    """Two directory names, one resolved configuration - the case the index exists to catch."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    config = base_config()
    write_experiment(experiments_dir, "exp_a_first_run", config)
    write_experiment(experiments_dir, "exp_b_rerun_with_a_different_name", config)
    output_path = tmp_path / "alias_index.csv"

    run_main(experiments_dir, output_path, monkeypatch)

    rows = read_rows(output_path)
    run_ids = [row["run_id"] for row in rows]
    assert len(set(run_ids)) == 1, "both directories must resolve to the same run_id"

    out = capsys.readouterr().out
    assert "1 run_id(s) shared by more than one directory" in out
    assert f"{identity.run_id(config)} x2" in out
