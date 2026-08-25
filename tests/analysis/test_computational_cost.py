"""Tests for `stec.analysis.computational_cost` (R2.8h).

Training cost is genuinely recomputed from synthetic log files here - `parse_training_log`
and `collect` are exercised against real files under `tmp_path`, not mocked. The two
values the module documents as *not* recomputable (pretraining wall-clock and the
inference throughput in `MEASURED_INFERENCE`) are pinned as recorded constants instead:
the tests check they are reported as such, and that a genuinely missing input (no
training logs, no pretrain loss history) is surfaced rather than silently defaulted to
zero.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stec.analysis import computational_cost as cc


def write_training_log(
    path: Path, start: datetime, epoch_gaps_s: list[float], max_epochs: int
) -> None:
    """A log with one `Epoch i/max_epochs` banner per gap, spaced by `epoch_gaps_s`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = start
    lines = []
    for i, gap in enumerate([0.0, *epoch_gaps_s]):
        stamp = stamp + timedelta(seconds=gap)
        lines.append(
            f"{stamp.strftime(cc.TIMESTAMP_FORMAT)},000 - INFO - Epoch {i + 1}/{max_epochs} - loss=0.1"
        )
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# parse_training_log: the real recomputation this module performs
# ---------------------------------------------------------------------------


def test_parse_training_log_computes_median_gap_and_wall_clock(tmp_path):
    log = tmp_path / "temp_config_stec_2024_150_training.log"
    start = datetime(2024, 6, 1, 10, 0, 0)
    write_training_log(log, start, epoch_gaps_s=[10.0, 20.0, 10.0], max_epochs=30)

    parsed = cc.parse_training_log(log)

    assert parsed["epochs_run"] == 4
    assert parsed["max_epochs"] == 30
    assert parsed["median_epoch_s"] == pytest.approx(
        10.0
    )  # gaps [10, 20, 10] -> median 10
    assert parsed["wall_clock_s"] == pytest.approx(40.0)


def test_parse_training_log_returns_none_with_fewer_than_two_epoch_banners(tmp_path):
    log = tmp_path / "temp_config_stec_2024_151_training.log"
    log.write_text("2024-06-01 10:00:00,000 - INFO - Epoch 1/30 - loss=0.1\n")
    assert cc.parse_training_log(log) is None


def test_collect_reads_only_logs_matching_the_glob_pattern(tmp_path):
    start = datetime(2024, 6, 1, 10, 0, 0)
    write_training_log(
        tmp_path / "2024_DOY_150" / "temp_config_stec_2024_150_training.log",
        start,
        epoch_gaps_s=[9.0, 9.0],
        max_epochs=30,
    )
    write_training_log(
        tmp_path / "2024_DOY_151" / "temp_config_vtec_2024_151_training.log",
        start,
        epoch_gaps_s=[7.0, 7.0],
        max_epochs=60,
    )

    stec = cc.collect(tmp_path, "*/temp_config_stec_*_training.log")
    assert len(stec) == 1
    assert stec.iloc[0]["median_epoch_s"] == pytest.approx(9.0)

    vtec = cc.collect(tmp_path, "*/temp_config_vtec_*_training.log")
    assert len(vtec) == 1


# ---------------------------------------------------------------------------
# summarise: medians and GPU-hour totals across days
# ---------------------------------------------------------------------------


def test_summarise_reports_medians_and_total_gpu_hours():
    table = pd.DataFrame(
        {
            "epochs_run": [10, 20, 30],
            "median_epoch_s": [8.0, 10.0, 12.0],
            "wall_clock_s": [80.0, 200.0, 360.0],
        }
    )
    row = cc.summarise("STEC daily fine-tune", table)
    assert row["days"] == 3
    assert row["median_epochs_run"] == pytest.approx(20.0)
    assert row["median_epoch_s"] == pytest.approx(10.0)
    assert row["median_wall_clock_min"] == pytest.approx(200.0 / 60)
    assert row["total_gpu_hours"] == pytest.approx((80.0 + 200.0 + 360.0) / 3600)


# ---------------------------------------------------------------------------
# main(): missing inputs are reported, not defaulted to zero; units are pinned
# ---------------------------------------------------------------------------


def run_main(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["computational_cost.py", *argv])
    cc.main()


def test_main_raises_when_no_stec_training_logs_exist(tmp_path, monkeypatch):
    """No STEC training logs at all is a setup error, not a table with zeroed rows."""
    with pytest.raises(FileNotFoundError):
        run_main(
            monkeypatch,
            [
                "--multiday-dir",
                str(tmp_path),
                "--pretrain-loss-history",
                str(tmp_path / "missing_loss_history.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )


def test_main_omits_pretrain_row_when_loss_history_is_missing(
    tmp_path, monkeypatch, caplog
):
    """A missing pretrain loss history must drop the pretraining estimate from the
    output entirely - not report 0 GPU-hours, which would read as a real measurement."""
    write_training_log(
        tmp_path
        / "per_day"
        / "2024"
        / "150"
        / "temp_config_stec_2024_150_training.log",
        datetime(2024, 6, 1, 10, 0, 0),
        epoch_gaps_s=[9.0, 9.0],
        max_epochs=30,
    )
    output_dir = tmp_path / "out"

    run_main(
        monkeypatch,
        [
            "--multiday-dir",
            str(tmp_path),
            "--pretrain-loss-history",
            str(tmp_path / "missing_loss_history.csv"),
            "--output-dir",
            str(output_dir),
        ],
    )

    cost_summary = pd.read_csv(output_dir / "cost_summary.csv")
    assert "pretraining, 150 epochs" not in set(cost_summary["item"])
    assert any("not found" in message for message in caplog.messages)


def test_main_reports_recorded_units_for_measured_items(tmp_path, monkeypatch):
    """Pins the unit strings a reader of cost_summary.csv depends on, and confirms the
    inference row reflects `MEASURED_INFERENCE` and the pretrain row reflects
    `MEASURED_PRETRAIN` - the two numbers this module cannot re-measure and instead
    reports as recorded constants (see the module docstring)."""
    write_training_log(
        tmp_path
        / "per_day"
        / "2024"
        / "150"
        / "temp_config_stec_2024_150_training.log",
        datetime(2024, 6, 1, 10, 0, 0),
        epoch_gaps_s=[9.0, 9.0],
        max_epochs=30,
    )
    loss_history = tmp_path / "loss_history.csv"
    pd.DataFrame({"epoch": range(150), "loss": [0.1] * 150}).to_csv(
        loss_history, index=False
    )
    output_dir = tmp_path / "out"

    run_main(
        monkeypatch,
        [
            "--multiday-dir",
            str(tmp_path),
            "--pretrain-loss-history",
            str(loss_history),
            "--output-dir",
            str(output_dir),
        ],
    )

    cost_summary = pd.read_csv(output_dir / "cost_summary.csv").set_index("item")
    assert cost_summary.loc["STEC daily fine-tune, median epoch", "unit"] == "s"
    assert (
        cost_summary.loc["STEC daily fine-tune, median wall clock", "unit"] == "min/day"
    )
    assert (
        cost_summary.loc["STEC daily fine-tune, total over all days", "unit"]
        == "GPU-hours"
    )
    pretrain_row = cost_summary.loc["pretraining, 150 epochs"]
    assert pretrain_row["unit"] == "GPU-hours"
    assert pretrain_row["measured"] == "yes"
    expected_pretrain_hours = 150 * cc.MEASURED_PRETRAIN["epoch_minutes"] / 60
    assert float(pretrain_row["value"]) == pytest.approx(expected_pretrain_hours)

    inference_row = cost_summary.loc["inference throughput"]
    assert (
        inference_row["unit"]
        == f"observations/s at T={cc.MEASURED_INFERENCE['mc_samples']}"
    )
    expected_rate = round(
        cc.MEASURED_INFERENCE["observations"] / cc.MEASURED_INFERENCE["seconds"], 0
    )
    assert float(inference_row["value"]) == pytest.approx(expected_rate)


def test_pretrain_row_uses_measured_epoch_rate_not_finetune_epoch_time(
    tmp_path, monkeypatch
):
    """Regression test for the bug this stage used to ship: the pretrain row must come
    from `MEASURED_PRETRAIN`, not from whatever the fine-tune logs happen to report.
    Scaling from the fine-tune's epoch time was wrong (the pretrain is I/O-bound on a
    500,000-row random resample every epoch, the fine-tune is not) and read 16x low -
    0.38 GPU-hours instead of the ~6.2 actually measured. A fine-tune epoch time picked
    deliberately far from `MEASURED_PRETRAIN['epoch_minutes']` must not move the
    pretrain row at all."""
    write_training_log(
        tmp_path
        / "per_day"
        / "2024"
        / "150"
        / "temp_config_stec_2024_150_training.log",
        datetime(2024, 6, 1, 10, 0, 0),
        epoch_gaps_s=[500.0, 500.0],
        max_epochs=30,
    )
    loss_history = tmp_path / "loss_history.csv"
    pd.DataFrame({"epoch": range(150), "loss": [0.1] * 150}).to_csv(
        loss_history, index=False
    )
    output_dir = tmp_path / "out"

    run_main(
        monkeypatch,
        [
            "--multiday-dir",
            str(tmp_path),
            "--pretrain-loss-history",
            str(loss_history),
            "--output-dir",
            str(output_dir),
        ],
    )

    cost_summary = pd.read_csv(output_dir / "cost_summary.csv").set_index("item")
    pretrain_row = cost_summary.loc["pretraining, 150 epochs"]
    assert pretrain_row["measured"] == "yes"
    expected_hours = 150 * cc.MEASURED_PRETRAIN["epoch_minutes"] / 60
    assert float(pretrain_row["value"]) == pytest.approx(expected_hours)
