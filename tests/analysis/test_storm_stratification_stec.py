"""Pins the STEC-domain storm/quiet split against hand-computed numbers.

1. **Classification uses the shared threshold.** A day at exactly -50 nT is a storm
   (`<=`, matching `storm_stratification.STORM_DST_THRESHOLD_NT`), not `<`.
2. **by_regime reproduces hand-computed median/Q1/Q3/mean** on a small synthetic frame.
3. **strongest_storm_days ranks by severity** (most negative Dst first) and carries one
   RMSE column per method.
4. **The repair-report gate is real**, not merely imported and unused.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from stec.analysis.storm_stratification_stec import (
    EXTREME_STORM_DST_THRESHOLD_NT,
    MODEL_RMSE_COLUMNS,
    build_by_regime,
    build_strongest_storm_days,
    classify_days,
    stratify,
)
from stec.analysis.storm_stratification import STORM_DST_THRESHOLD_NT


def write_swi(path: Path, year: int, daily: dict[int, dict[str, float]]) -> None:
    """A minimal OMNI-hourly-style h5 file: 24 identical hourly rows per day, holding
    only the two columns `load_daily_geomagnetic_indices` reads."""
    columns = ["Dst-index,_nT", "Kp_index"]
    with h5py.File(path, "w") as handle:
        group = handle.create_group(str(year))
        for doy, values in daily.items():
            row = [values["dst"], values["kp"]]
            table = np.tile(row, (24, 1))
            dataset = group.create_dataset(f"{doy:03d}", data=table)
            dataset.attrs["columns"] = columns


def write_repair_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"dataset": ["own"], "doy": [132], "repaired": [False]}).to_csv(
        path, index=False
    )


def write_per_day_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def day_row(
    doy: int,
    model: str,
    rmse: float,
    mae: float,
    dataset: str = "own_vtec_gim",
    year: int = 2024,
) -> dict:
    return {
        "date": f"{year}-{doy:03d}",
        "year": year,
        "doy": doy,
        "dataset": dataset,
        "Model": model,
        "RMSE": rmse,
        "MAE": mae,
        "R2": 0.9,
        "Bias": 0.0,
        "Std": rmse,
        "Count": 1000,
    }


MODELS = list(
    MODEL_RMSE_COLUMNS
)  # Direct STEC Model, IGS GIM, VTEC + Mapping, Pretrained STEC


# --------------------------------------------------------------------------
# 1. Classification threshold
# --------------------------------------------------------------------------


def test_a_day_at_exactly_the_threshold_is_a_storm_not_quiet(tmp_path):
    swi_path = tmp_path / "swi.h5"
    write_swi(
        swi_path,
        2024,
        {
            132: {"dst": STORM_DST_THRESHOLD_NT, "kp": 5.0},  # exactly -50
            133: {"dst": STORM_DST_THRESHOLD_NT + 0.1, "kp": 1.0},  # -49.9, quiet
        },
    )
    classified = classify_days(pd.Series([132, 133]), 2024, swi_path)
    regimes = classified.set_index("doy")["regime"]
    assert regimes.loc[132] == "storm"
    assert regimes.loc[133] == "quiet"


def test_stratify_refuses_to_run_without_a_repair_report(tmp_path):
    swi_path = tmp_path / "swi.h5"
    daily_metrics_csv = tmp_path / "per_day.csv"
    write_swi(swi_path, 2024, {132: {"dst": -10.0, "kp": 2.0}})
    write_per_day_csv(daily_metrics_csv, [day_row(132, "Direct STEC Model", 4.0, 3.0)])

    with pytest.raises(FileNotFoundError, match="repair_gim_baseline"):
        stratify(
            daily_metrics_csv,
            year=2024,
            swi_path=swi_path,
            repair_report=tmp_path / "missing_report.csv",
        )


def test_stratify_runs_once_the_repair_report_is_present(tmp_path):
    swi_path = tmp_path / "swi.h5"
    daily_metrics_csv = tmp_path / "per_day.csv"
    repair_report = tmp_path / "gim_repair_report.csv"
    write_repair_report(repair_report)
    write_swi(swi_path, 2024, {132: {"dst": -10.0, "kp": 2.0}})
    write_per_day_csv(daily_metrics_csv, [day_row(132, "Direct STEC Model", 4.0, 3.0)])

    day_classification, merged = stratify(
        daily_metrics_csv, year=2024, swi_path=swi_path, repair_report=repair_report
    )
    assert len(day_classification) == 1
    assert len(merged) == 1
    assert merged.iloc[0]["regime"] == "quiet"


# --------------------------------------------------------------------------
# 2. by_regime: hand-computed median/Q1/Q3/mean
# --------------------------------------------------------------------------


def test_build_by_regime_reproduces_hand_computed_statistics():
    # Three quiet days (RMSE 4, 6, 10) and one storm day (RMSE 20) for one method.
    rows = [
        day_row(132, "Direct STEC Model", rmse=4.0, mae=3.0),
        day_row(133, "Direct STEC Model", rmse=6.0, mae=5.0),
        day_row(134, "Direct STEC Model", rmse=10.0, mae=8.0),
        day_row(200, "Direct STEC Model", rmse=20.0, mae=15.0),
    ]
    merged = pd.DataFrame(rows)
    merged["regime"] = ["quiet", "quiet", "quiet", "storm"]

    by_regime = build_by_regime(merged)
    quiet_row = by_regime[
        (by_regime["method"] == "Direct STEC Model") & (by_regime["regime"] == "quiet")
    ].iloc[0]

    expected_rmse = pd.Series([4.0, 6.0, 10.0])
    assert quiet_row["n_days"] == 3
    assert quiet_row["rmse_median"] == pytest.approx(expected_rmse.median())
    assert quiet_row["rmse_q1"] == pytest.approx(expected_rmse.quantile(0.25))
    assert quiet_row["rmse_q3"] == pytest.approx(expected_rmse.quantile(0.75))
    assert quiet_row["rmse_mean"] == pytest.approx(expected_rmse.mean())

    storm_row = by_regime[
        (by_regime["method"] == "Direct STEC Model") & (by_regime["regime"] == "storm")
    ].iloc[0]
    assert storm_row["n_days"] == 1
    assert storm_row["rmse_median"] == pytest.approx(20.0)


def test_by_regime_keeps_datasets_and_methods_separate():
    rows = [
        day_row(132, "Direct STEC Model", rmse=4.0, mae=3.0, dataset="own_vtec_gim"),
        day_row(132, "IGS GIM", rmse=8.0, mae=6.0, dataset="own_vtec_gim"),
        day_row(
            132, "Direct STEC Model", rmse=40.0, mae=30.0, dataset="madrigal_vtec_gim"
        ),
    ]
    merged = pd.DataFrame(rows)
    merged["regime"] = "quiet"

    by_regime = build_by_regime(merged)
    assert len(by_regime) == 3  # one row per (dataset, method) here, all "quiet"
    own_direct = by_regime[
        (by_regime["dataset"] == "own_vtec_gim")
        & (by_regime["method"] == "Direct STEC Model")
    ].iloc[0]
    assert own_direct["rmse_median"] == pytest.approx(4.0)
    madrigal_direct = by_regime[
        (by_regime["dataset"] == "madrigal_vtec_gim")
        & (by_regime["method"] == "Direct STEC Model")
    ].iloc[0]
    assert madrigal_direct["rmse_median"] == pytest.approx(40.0)


# --------------------------------------------------------------------------
# 3. strongest_storm_days: severity rank and per-method RMSE columns
# --------------------------------------------------------------------------


def test_strongest_storm_days_ranks_by_severity_and_carries_every_method():
    day_classification = pd.DataFrame(
        [
            {
                "date": "2024-131",
                "doy": 131,
                "min_dst": -339.0,
                "max_kp": 8.0,
                "regime": "storm",
            },
            {
                "date": "2024-132",
                "doy": 132,
                "min_dst": -406.0,
                "max_kp": 9.0,
                "regime": "storm",
            },
            # not extreme enough (-50 threshold storm, but not < -300)
            {
                "date": "2024-150",
                "doy": 150,
                "min_dst": -60.0,
                "max_kp": 4.0,
                "regime": "storm",
            },
        ]
    )
    rows = []
    for doy, values in {
        131: {
            "Direct STEC Model": 7.81,
            "IGS GIM": 9.30,
            "VTEC + Mapping": 10.45,
            "Pretrained STEC": 15.24,
        },
        132: {
            "Direct STEC Model": 7.31,
            "IGS GIM": 7.71,
            "VTEC + Mapping": 8.39,
            "Pretrained STEC": 29.25,
        },
        150: {
            "Direct STEC Model": 5.0,
            "IGS GIM": 6.0,
            "VTEC + Mapping": 6.5,
            "Pretrained STEC": 7.0,
        },
    }.items():
        for model, rmse in values.items():
            rows.append(day_row(doy, model, rmse=rmse, mae=rmse * 0.6))
    merged = pd.DataFrame(rows)

    result = build_strongest_storm_days(
        merged, day_classification, threshold=EXTREME_STORM_DST_THRESHOLD_NT
    )

    # Only the two days below -300 nT are included; DOY 150 (-60) is excluded even
    # though it is itself a storm day under the -50 nT rule.
    assert set(result["doy"]) == {131, 132}
    assert list(result["rank"]) == [1, 2]
    # Rank order: 132 (-406 nT) is more severe than 131 (-339 nT), so it comes first.
    assert list(result["doy"]) == [132, 131]

    row_132 = result[result["doy"] == 132].iloc[0]
    assert row_132["direct_stec_rmse"] == pytest.approx(7.31)
    assert row_132["igs_gim_rmse"] == pytest.approx(7.71)
    assert row_132["vtec_mapping_rmse"] == pytest.approx(8.39)
    assert row_132["pretrained_rmse"] == pytest.approx(29.25)

    row_131 = result[result["doy"] == 131].iloc[0]
    assert row_131["rank"] == 2
    assert row_131["direct_stec_rmse"] == pytest.approx(7.81)


def test_strongest_storm_days_is_empty_but_correctly_shaped_when_no_day_qualifies():
    day_classification = pd.DataFrame(
        [
            {
                "date": "2024-132",
                "doy": 132,
                "min_dst": -60.0,
                "max_kp": 4.0,
                "regime": "storm",
            }
        ]
    )
    merged = pd.DataFrame([day_row(132, "Direct STEC Model", rmse=5.0, mae=3.0)])

    result = build_strongest_storm_days(
        merged, day_classification, threshold=EXTREME_STORM_DST_THRESHOLD_NT
    )
    assert result.empty
    assert "direct_stec_rmse" in result.columns
    assert "rank" in result.columns
