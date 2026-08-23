"""Pins the day-at-a-time streaming, the year-scoping (`pretrained_stec/own` spans
2014-2024, so a bare `doy` filter would silently pool years), the column narrowing that
keeps the concatenation bounded, and the manifest Figures 4-9's builder reads for
provenance.

Follows `test_elevation_metrics_finetuned.py`'s style: synthetic days are written through
`prediction_store.write_predictions` so the streaming tests exercise the real on-disk
parquet format, never an in-memory shortcut, and never the real store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stec.analysis import pretrained_test_diagnostics as ptd
from stec.inference import prediction_store as ps


def day_frame(rows: int, seed: int, missing: list[str] | None = None) -> pd.DataFrame:
    """A day of synthetic per-observation rows shaped like the pretrained store's own
    columns, with a known prediction offset so downstream figure tests can assert exact
    residual values."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    total_unc = np.abs(rng.normal(3.0, 1.0, rows)) + 0.5
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + 1.0,
            "satele": rng.uniform(5, 90, rows),
            "sm_lat_ipp": rng.uniform(-90, 90, rows),
            "sod": rng.uniform(0, 86400, rows),
            "lon_ipp": rng.uniform(-180, 180, rows),
            "pred_total_unc": total_unc,
            "pred_epistemic_unc": total_unc * 0.3,
            "pred_aleatoric_unc": total_unc * 0.7,
        }
    )
    return frame.drop(columns=missing or [])


def test_collect_streams_multiple_days_into_one_narrow_frame(tmp_path):
    """End-to-end through the on-disk parquet format: `collect` must return every
    requested day's rows, narrowed to `WANTED_COLUMNS`, with `local_time_hours` (absent
    from every written day here, matching the real store) simply not appearing rather than
    raising."""
    days = [(2024, 130, 400, 1), (2024, 131, 500, 2), (2014, 152, 300, 3)]
    for year, doy, rows, seed in days:
        ps.write_predictions(
            day_frame(rows, seed), "pretrained_stec", "own", year, doy, root=tmp_path
        )

    observations = ptd.collect(tmp_path)

    assert len(observations) == sum(rows for _, _, rows, _ in days)
    assert set(zip(observations["year"], observations["doy"])) == {
        (year, doy) for year, doy, _, _ in days
    }
    assert "local_time_hours" not in observations.columns
    assert {"true_stec", "stec_pred", "satele", "sm_lat_ipp"} <= set(
        observations.columns
    )


def test_collect_years_filter_prevents_pooling_a_doy_across_years(tmp_path):
    """`pretrained_stec/own` holds the same doy in multiple years (2014-2024); passing
    `years` must restrict to the requested one(s) rather than silently matching doy=152
    in every year it exists - the exact bug CLAUDE.md documents `uncertainty_calibration`
    being bitten by."""
    ps.write_predictions(
        day_frame(400, 1), "pretrained_stec", "own", 2024, 152, root=tmp_path
    )
    ps.write_predictions(
        day_frame(300, 2), "pretrained_stec", "own", 2014, 152, root=tmp_path
    )

    observations = ptd.collect(tmp_path, years=[2024])

    assert set(observations["year"]) == {2024}
    assert len(observations) == 400


def test_collect_handles_a_day_missing_optional_columns(tmp_path):
    """A day written without the epistemic/aleatoric split (an older or ensemble-only
    run) must still contribute its other columns, not fail the whole read - `pd.concat`
    fills the gap with NaN for that day only."""
    ps.write_predictions(
        day_frame(200, 4, missing=["pred_epistemic_unc", "pred_aleatoric_unc"]),
        "pretrained_stec",
        "own",
        2024,
        160,
        root=tmp_path,
    )
    ps.write_predictions(
        day_frame(200, 5), "pretrained_stec", "own", 2024, 161, root=tmp_path
    )

    observations = ptd.collect(tmp_path)

    assert len(observations) == 400
    day_160 = observations[observations["doy"] == 160]
    assert day_160["pred_epistemic_unc"].isna().all()
    day_161 = observations[observations["doy"] == 161]
    assert day_161["pred_epistemic_unc"].notna().all()


def test_collect_skips_a_day_missing_truth_or_prediction(tmp_path, caplog):
    """A file that somehow lacks true_stec/stec_pred (write_predictions itself refuses
    this, so simulate it by writing the parquet directly) must be skipped with a warning,
    not crash the whole sweep."""
    import logging

    good = day_frame(300, 6)
    ps.write_predictions(good, "pretrained_stec", "own", 2024, 170, root=tmp_path)

    broken_path = ps.store_path("pretrained_stec", "own", 2024, 171, root=tmp_path)
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"satele": [10.0, 20.0]}).to_parquet(broken_path, index=False)

    with caplog.at_level(logging.WARNING):
        observations = ptd.collect(tmp_path)

    assert len(observations) == 300
    assert "2024-171" in caplog.text


def test_collect_raises_file_not_found_for_an_absent_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        ptd.collect(tmp_path)


def test_manifest_reports_per_year_day_and_observation_counts():
    observations = pd.DataFrame(
        {
            "year": [2024, 2024, 2024, 2014],
            "doy": [130, 130, 131, 152],
        }
    )
    table = ptd.manifest(observations).set_index("year")

    assert table.loc[2024, "n_days"] == 2
    assert table.loc[2024, "n_observations"] == 3
    assert table.loc[2014, "n_days"] == 1
    assert table.loc[2014, "n_observations"] == 1
