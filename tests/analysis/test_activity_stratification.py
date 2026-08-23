"""Pins the two things most likely to go silently wrong in this port.

1. **Fixed bin edges.** Dst and F10.7 must partition by fixed constants, not by the
   distribution of the test period being summarised - two synthetic periods with very
   different F10.7 spreads must still place the same absolute value in the same bin.
2. **The repair-report gate.** `stratify()`/`require_repaired_daily_metrics` must
   refuse to compute anything when either the daily-metrics CSV or the repair report
   is missing, rather than silently falling back to an unrepaired source - that
   silent fallback is what reversed the R1.4 conclusion in the pre-rebuild code.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis.activity_stratification import (
    DST_BINS,
    DST_LABELS,
    F107_BINS,
    F107_LABELS,
    require_repaired_daily_metrics,
    stratify,
)


def write_swi(path: Path, year: int, daily: dict[int, dict[str, float]]) -> None:
    """A minimal OMNI-hourly-style h5 file: 24 identical hourly rows per day, holding
    only the three columns `load_daily_indices` reads."""
    columns = ["Dst-index,_nT", "Kp_index", "f107_index"]
    with h5py.File(path, "w") as handle:
        group = handle.create_group(str(year))
        for doy, values in daily.items():
            row = [values["dst"], values["kp"], values["f107"]]
            table = np.tile(row, (24, 1))
            dataset = group.create_dataset(f"{doy:03d}", data=table)
            dataset.attrs["columns"] = columns


def write_repair_report(path: Path) -> None:
    """`repair_gim_baseline`'s own report - content does not matter here, only its
    presence, which is the evidence the check ran (see the module docstring)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dataset": ["own"], "doy": [132], "repaired": [False]}).to_csv(
        path, index=False
    )


def write_per_day_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def day_row(
    doy: int, model: str, rmse: float, mae: float, r2: float, count: int
) -> dict:
    return {
        "date": f"2024-{doy:03d}",
        "year": 2024,
        "doy": doy,
        "dataset": "own_vtec_gim",
        "Model": model,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Bias": 0.0,
        "Std": rmse,
        "Count": count,
    }


# --------------------------------------------------------------------------
# 1. Fixed bin edges
# --------------------------------------------------------------------------


def test_dst_bins_are_fixed_constants_not_derived_from_the_data():
    assert DST_BINS == [-1000, -100, -50, -30, 1000]
    assert len(DST_LABELS) == len(DST_BINS) - 1


def test_f107_bins_are_fixed_constants_not_derived_from_the_data():
    assert F107_BINS == [0, 100, 150, 200, 1000]
    assert len(F107_LABELS) == len(F107_BINS) - 1


def test_two_differently_distributed_periods_place_the_same_value_in_the_same_bin(
    tmp_path,
):
    """A quiet-period frame and a storm-period frame differ wildly in their F10.7 and
    Dst spread; a value of 120 sfu / -40 nT must still land in the same bin either way,
    which only holds if the bin edges are fixed rather than computed per call (e.g. as
    the original source did with F10.7 terciles)."""
    swi_path = tmp_path / "swi.h5"
    daily_metrics_csv = tmp_path / "per_day.csv"
    repair_report = tmp_path / "gim_repair_report.csv"
    write_repair_report(repair_report)

    quiet_period = {
        132: {"dst": -5.0, "kp": 1.0, "f107": 90.0},
        133: {"dst": -10.0, "kp": 1.0, "f107": 95.0},
        # the probe day, embedded in an otherwise very quiet/low-flux period
        134: {"dst": -40.0, "kp": 3.0, "f107": 120.0},
    }
    storm_period = {
        200: {"dst": -300.0, "kp": 8.0, "f107": 240.0},
        201: {"dst": -250.0, "kp": 8.0, "f107": 230.0},
        # the same probe values, now embedded in a very disturbed/high-flux period
        202: {"dst": -40.0, "kp": 3.0, "f107": 120.0},
    }

    bins_seen = {}
    for label, period in (("quiet", quiet_period), ("storm", storm_period)):
        write_swi(swi_path, 2024, period)
        rows = [
            day_row(doy, "Direct STEC Model", rmse=5.0, mae=4.0, r2=0.9, count=1000)
            for doy in period
        ] + [
            day_row(doy, "IGS GIM", rmse=8.0, mae=6.0, r2=0.7, count=1000)
            for doy in period
        ]
        write_per_day_csv(daily_metrics_csv, rows)

        tables = stratify(
            daily_metrics_csv,
            year=2024,
            swi_path=swi_path,
            repair_report=repair_report,
        )
        # The probe day (f107=120.0) is the only day of its period in the "moderate"
        # bin - the other two days sit at the opposite end of that period's own
        # spread - so it is the unique row with a single day pooled into it.
        probe_row = tables["f107"][
            (tables["f107"]["Model"] == "Direct STEC Model")
            & (tables["f107"]["days"] == 1)
        ]
        assert len(probe_row) == 1
        bins_seen[label] = probe_row["f107_bin"].iloc[0]

    assert bins_seen["quiet"] == bins_seen["storm"]


# --------------------------------------------------------------------------
# 2. Hand-computed per-bin values on a small synthetic frame
# --------------------------------------------------------------------------


def test_stratify_reproduces_hand_computed_pooled_rmse_per_dst_bin(tmp_path):
    swi_path = tmp_path / "swi.h5"
    daily_metrics_csv = tmp_path / "per_day.csv"
    repair_report = tmp_path / "gim_repair_report.csv"
    write_repair_report(repair_report)

    # Two days in the "quiet" Dst bin (> -30), one in "weak" (-50 to -30).
    write_swi(
        swi_path,
        2024,
        {
            132: {"dst": -10.0, "kp": 2.0, "f107": 110.0},
            133: {"dst": -20.0, "kp": 2.0, "f107": 110.0},
            134: {"dst": -35.0, "kp": 4.0, "f107": 110.0},
        },
    )
    write_per_day_csv(
        daily_metrics_csv,
        [
            day_row(132, "Direct STEC Model", rmse=4.0, mae=3.0, r2=0.9, count=100),
            day_row(133, "Direct STEC Model", rmse=6.0, mae=5.0, r2=0.8, count=300),
            day_row(134, "Direct STEC Model", rmse=10.0, mae=8.0, r2=0.5, count=50),
            day_row(132, "IGS GIM", rmse=8.0, mae=6.0, r2=0.7, count=100),
            day_row(133, "IGS GIM", rmse=8.0, mae=6.0, r2=0.7, count=300),
            day_row(134, "IGS GIM", rmse=8.0, mae=6.0, r2=0.7, count=50),
        ],
    )

    tables = stratify(
        daily_metrics_csv, year=2024, swi_path=swi_path, repair_report=repair_report
    )
    dst_table = tables["dst"]

    quiet_row = dst_table[
        (dst_table["Model"] == "Direct STEC Model")
        & (dst_table["dst_bin"] == DST_LABELS[3])  # "quiet\n(> −30)"
    ].iloc[0]
    # Count-weighted pool of the two quiet days: sqrt((100*4^2 + 300*6^2) / 400).
    expected_rmse = np.sqrt((100 * 4.0**2 + 300 * 6.0**2) / 400)
    assert quiet_row["RMSE"] == pytest.approx(expected_rmse)
    assert quiet_row["observations"] == 400
    assert quiet_row["days"] == 2

    weak_row = dst_table[
        (dst_table["Model"] == "Direct STEC Model")
        & (dst_table["dst_bin"] == DST_LABELS[2])  # "weak\n(−50 to −30)"
    ].iloc[0]
    assert weak_row["RMSE"] == pytest.approx(10.0)
    assert weak_row["observations"] == 50

    # Scale-free companion: improvement over the IGS GIM baseline in the quiet bin.
    gim_quiet_rmse = dst_table[
        (dst_table["Model"] == "IGS GIM") & (dst_table["dst_bin"] == DST_LABELS[3])
    ].iloc[0]["RMSE"]
    expected_improvement = 100 * (gim_quiet_rmse - expected_rmse) / gim_quiet_rmse
    assert quiet_row["improvement_over_gim_%"] == pytest.approx(expected_improvement)


# --------------------------------------------------------------------------
# 3. The repaired-input gate
# --------------------------------------------------------------------------


def test_missing_daily_metrics_csv_fails_loudly(tmp_path):
    repair_report = tmp_path / "gim_repair_report.csv"
    write_repair_report(repair_report)

    with pytest.raises(FileNotFoundError, match="daily_metrics"):
        require_repaired_daily_metrics(tmp_path / "does_not_exist.csv", repair_report)


def test_missing_repair_report_fails_loudly_even_if_daily_metrics_exists(tmp_path):
    daily_metrics_csv = tmp_path / "per_day.csv"
    write_per_day_csv(
        daily_metrics_csv,
        [day_row(132, "Direct STEC Model", rmse=4.0, mae=3.0, r2=0.9, count=100)],
    )

    with pytest.raises(FileNotFoundError, match="repair_gim_baseline"):
        require_repaired_daily_metrics(
            daily_metrics_csv, tmp_path / "gim_repair_report.csv"
        )


def test_stratify_refuses_to_run_without_a_repair_report(tmp_path):
    """The end-to-end entry point, not just the guard function directly - a caller
    that only ever calls `stratify()` must still be protected."""
    swi_path = tmp_path / "swi.h5"
    daily_metrics_csv = tmp_path / "per_day.csv"
    write_swi(swi_path, 2024, {132: {"dst": -10.0, "kp": 2.0, "f107": 110.0}})
    write_per_day_csv(
        daily_metrics_csv,
        [day_row(132, "Direct STEC Model", rmse=4.0, mae=3.0, r2=0.9, count=100)],
    )

    with pytest.raises(FileNotFoundError, match="repair_gim_baseline"):
        stratify(
            daily_metrics_csv,
            year=2024,
            swi_path=swi_path,
            repair_report=tmp_path / "missing_report.csv",
        )
