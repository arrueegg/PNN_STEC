"""Pins the day-at-a-time streaming and column set for `diagnostic_test_observations`,
the wider-column sibling of `pretrained_test_diagnostics` that `stec.viz.diagnostic_figures`
reads. Follows `test_pretrained_test_diagnostics.py`'s style: synthetic days are written
through `prediction_store.write_predictions` so the streaming tests exercise the real
on-disk parquet format, never an in-memory shortcut, and never the real store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import diagnostic_test_observations as dto
from stec.inference import prediction_store as ps


def day_frame(rows: int, seed: int, missing: list[str] | None = None) -> pd.DataFrame:
    """A day of synthetic per-observation rows shaped like the pretrained store's own
    columns, extended with the azimuth/lat_ipp/forcing columns this module adds."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    total_unc = np.abs(rng.normal(3.0, 1.0, rows)) + 0.5
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "satele": rng.uniform(5, 90, rows),
            "satazi": rng.uniform(0, 360, rows),
            "lat_ipp": rng.uniform(-90, 90, rows),
            "lon_ipp": rng.uniform(-180, 180, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "sod": rng.uniform(0, 86400, rows),
            "pred_total_unc": total_unc,
            "pred_epistemic_unc": total_unc * 0.3,
            "pred_aleatoric_unc": total_unc * 0.7,
            "Kp_index": rng.uniform(0, 90, rows),
            "R_Sunspot_No": rng.uniform(0, 200, rows),
            "Dst-index,_nT": rng.uniform(-200, 50, rows),
            "AE-index,_nT": rng.uniform(0, 1500, rows),
            "ap_index,_nT": rng.uniform(0, 300, rows),
            "f107_index": rng.uniform(70, 250, rows),
        }
    )
    return frame.drop(columns=missing or [])


def test_collect_streams_multiple_days_with_the_extended_column_set(tmp_path):
    days = [(2024, 130, 400, 1), (2024, 131, 500, 2), (2014, 152, 300, 3)]
    for year, doy, rows, seed in days:
        ps.write_predictions(
            day_frame(rows, seed), "pretrained_stec", "own", year, doy, root=tmp_path
        )

    observations = dto.collect(tmp_path)

    assert len(observations) == sum(rows for _, _, rows, _ in days)
    assert {"satazi", "lat_ipp", "Kp_index", "f107_index", "Dst-index,_nT"} <= set(
        observations.columns
    )
    # local_time_hours is deliberately not in WANTED_COLUMNS - it is never present in
    # the real store and is derived downstream from sod/lon_ipp instead.
    assert "local_time_hours" not in observations.columns


def test_collect_years_filter_prevents_pooling_a_doy_across_years(tmp_path):
    ps.write_predictions(
        day_frame(400, 1), "pretrained_stec", "own", 2024, 152, root=tmp_path
    )
    ps.write_predictions(
        day_frame(300, 2), "pretrained_stec", "own", 2014, 152, root=tmp_path
    )

    observations = dto.collect(tmp_path, years=[2024])

    assert set(observations["year"]) == {2024}
    assert len(observations) == 400


def test_collect_handles_a_day_missing_optional_columns(tmp_path):
    ps.write_predictions(
        day_frame(200, 4, missing=["ap_index,_nT", "AE-index,_nT"]),
        "pretrained_stec",
        "own",
        2024,
        160,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(200, 5), "pretrained_stec", "own", 2024, 161, root=tmp_path
    )

    observations = dto.collect(tmp_path)

    assert len(observations) == 400
    day_160 = observations[observations["doy"] == 160]
    assert day_160["ap_index,_nT"].isna().all()
    day_161 = observations[observations["doy"] == 161]
    assert day_161["ap_index,_nT"].notna().all()


def test_collect_raises_file_not_found_for_an_absent_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        dto.collect(tmp_path)


def test_manifest_reports_per_year_day_and_observation_counts():
    observations = pd.DataFrame(
        {"year": [2024, 2024, 2024, 2014], "doy": [130, 130, 131, 152]}
    )
    table = dto.manifest(observations).set_index("year")

    assert table.loc[2024, "n_days"] == 2
    assert table.loc[2024, "n_observations"] == 3
    assert table.loc[2014, "n_days"] == 1


def test_build_writes_parquet_and_manifest(tmp_path):
    ps.write_predictions(
        day_frame(150, 7), "pretrained_stec", "own", 2024, 200, root=tmp_path
    )
    output_dir = tmp_path / "cache"

    observations = dto.build(tmp_path, output_dir)

    assert (output_dir / "observations.parquet").exists()
    assert (output_dir / "manifest.csv").exists()
    reread = pd.read_parquet(output_dir / "observations.parquet")
    assert len(reread) == len(observations) == 150
