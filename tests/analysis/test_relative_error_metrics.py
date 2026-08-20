"""Pins the interpolation/extrapolation boundary and the TEC-normalised error's
behaviour at a zero denominator.

`collect_yearly_metrics`/`collect_regime_metrics` read pre-computed text summaries
rather than raw observations, so this builds those text files under `tmp_path` in the
exact format `parse_year_summary`'s regexes expect, matching the style of the
temporal-analysis output the pretrained-model evaluation actually writes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from stec.analysis import relative_error_metrics as rem

SUMMARY_TEMPLATE = """Metrics Summary
Sample Count: {count}
RMSE: {rmse}
MAE: {mae}
R²: {r2}
Mean Target STEC: {mean_stec}
"""


def write_summary(
    path: Path, *, count=1000, rmse=5.0, mae=3.0, r2=0.9, mean_stec=20.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        SUMMARY_TEMPLATE.format(
            count=count, rmse=rmse, mae=mae, r2=r2, mean_stec=mean_stec
        )
    )


# --------------------------------------------------------------------------
# Interpolation / extrapolation boundary: named constant, not a literal
# --------------------------------------------------------------------------


def test_boundary_constant_is_2024_05_01():
    assert rem.EXTRAPOLATION_START == datetime(2024, 5, 1)


def test_day_before_the_boundary_is_interpolation():
    assert rem.regime_of(datetime(2024, 4, 30, 23, 59, 59)) == "interpolation"


def test_boundary_day_itself_is_extrapolation():
    """Matches `training_utils.split_test_data_by_date`'s `>=` comparison: the split
    day belongs to extrapolation, not interpolation."""
    assert rem.regime_of(datetime(2024, 5, 1)) == "extrapolation"


def test_day_after_the_boundary_is_extrapolation():
    assert rem.regime_of(datetime(2024, 5, 2)) == "extrapolation"


def test_regime_labels_are_built_from_the_constant_not_a_second_literal():
    labels = dict(rem.REGIME_LABELS)
    assert "2024-05-01" in labels["interpolation"]
    assert "2024-05-01" in labels["extrapolation"]


# --------------------------------------------------------------------------
# TEC-normalised error: undefined (not guarded) at mean_STEC == 0
# --------------------------------------------------------------------------


def test_nrmse_divides_by_the_yearly_mean_not_a_per_observation_value(tmp_path):
    summary_dir = tmp_path / "temporal_analysis"
    write_summary(
        summary_dir / "year_2020.0_metrics_summary.txt", rmse=4.0, mean_stec=8.0
    )
    table = rem.collect_yearly_metrics(tmp_path)
    assert table.loc[0, "nRMSE_%"] == pytest.approx(100 * 4.0 / 8.0)


def test_nrmse_is_inf_not_raising_when_mean_stec_is_exactly_zero(tmp_path):
    """Pins the source's actual (unguarded) behaviour: a zero denominator produces
    `inf`, not an exception and not a silently wrong finite number. This is a
    documentation test, not an endorsement - `mean_STEC == 0` does not occur in the
    real per-year summaries, but the formula is undefined there and this is what it
    does about it."""
    summary_dir = tmp_path / "temporal_analysis"
    write_summary(
        summary_dir / "year_2020.0_metrics_summary.txt", rmse=4.0, mean_stec=0.0
    )
    table = rem.collect_yearly_metrics(tmp_path)
    assert np.isinf(table.loc[0, "nRMSE_%"])


def test_nrmse_is_nan_when_both_rmse_and_mean_stec_are_zero(tmp_path):
    summary_dir = tmp_path / "temporal_analysis"
    write_summary(
        summary_dir / "year_2020.0_metrics_summary.txt", rmse=0.0, mean_stec=0.0
    )
    table = rem.collect_yearly_metrics(tmp_path)
    assert np.isnan(table.loc[0, "nRMSE_%"])


# --------------------------------------------------------------------------
# Parsing and aggregation
# --------------------------------------------------------------------------


def test_collect_yearly_metrics_reads_every_year_and_sorts_them(tmp_path):
    summary_dir = tmp_path / "temporal_analysis"
    write_summary(
        summary_dir / "year_2023.0_metrics_summary.txt", rmse=6.0, mean_stec=25.0
    )
    write_summary(
        summary_dir / "year_2014.0_metrics_summary.txt", rmse=2.0, mean_stec=6.0
    )
    table = rem.collect_yearly_metrics(tmp_path)
    assert list(table["year"]) == [2014, 2023]
    assert table.loc[0, "nMAE_%"] == pytest.approx(100 * 3.0 / 6.0)


def test_collect_yearly_metrics_raises_when_nothing_is_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        rem.collect_yearly_metrics(tmp_path)


def test_parse_year_summary_returns_none_on_a_missing_field(tmp_path):
    path = tmp_path / "broken.txt"
    path.write_text("Sample Count: 100\nRMSE: 5.0\n")  # missing MAE, R2, mean_STEC
    assert rem.parse_year_summary(path) is None


def test_collect_regime_metrics_reads_interpolation_and_extrapolation_trees(tmp_path):
    write_summary(
        tmp_path / "interpolation" / "temporal_analysis" / "total_metrics_summary.txt",
        rmse=3.0,
        mean_stec=10.0,
    )
    write_summary(
        tmp_path / "extrapolation" / "temporal_analysis" / "total_metrics_summary.txt",
        rmse=9.0,
        mean_stec=37.0,
    )
    table = rem.collect_regime_metrics(tmp_path)
    assert table is not None
    assert list(table["regime"]) == [label for _, label in rem.REGIME_LABELS]
    interp, extrap = table.iloc[0], table.iloc[1]
    assert extrap["RMSE"] / interp["RMSE"] == pytest.approx(3.0)
    # Higher absolute RMSE but a *lower* normalised error - the R2.1 finding this
    # analysis exists to surface.
    assert extrap["nRMSE_%"] < interp["nRMSE_%"]


def test_collect_regime_metrics_returns_none_when_a_tree_is_missing(tmp_path):
    write_summary(
        tmp_path / "interpolation" / "temporal_analysis" / "total_metrics_summary.txt"
    )
    assert rem.collect_regime_metrics(tmp_path) is None
