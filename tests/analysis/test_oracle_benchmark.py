"""Tests for `stec.analysis.oracle_benchmark` (R2.8).

`load_oracle` is exercised end to end against synthetic `.pos`/SINEX files, reusing the
same fixture shape as `tests/positioning/test_metrics.py`. The remaining functions
(`paired_comparison`, `summarise`, `check_gim_control`) are tested against small in-memory
frames, since they operate purely on already-parsed metrics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import oracle_benchmark as ob
from stec.positioning import metrics as pm

_POS_FIXTURE = """\
 mjd     sod   nsat   x             y             z          stdx     stdy     stdz    rck(m)   zhd     zwd     dzwd
60609     0.00   4  -3530194.195   4118798.368   3344042.673    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
60609    30.00   4  -3530194.840   4118798.715   3344043.220    0.000    0.000    0.000      0.0   2.232   0.079   0.3739
"""

_SINEX_FIXTURE = (
    "+SOLUTION/ESTIMATE\n"
    " 1 STAX  AIRA  A    1  05:159:43200 m    01  -3530200.0000 0.0011\n"
    " 2 STAY  AIRA  A    1  05:159:43200 m    01   4118800.0000 0.0011\n"
    " 3 STAZ  AIRA  A    1  05:159:43200 m    01   3344040.0000 0.0011\n"
    "-SOLUTION/ESTIMATE\n"
)


def row(station: str, doy: int, method: str, error_3d: float) -> dict:
    return {
        "station": station,
        "doy": doy,
        "method": method,
        "error_3d_rms": error_3d,
        "error_2d_rms": error_3d / 2,
        "u_rms": error_3d / 4,
    }


# ---------------------------------------------------------------------------
# load_oracle: aggregation straight from .pos + SINEX, not daily_summary.csv
# ---------------------------------------------------------------------------


def _build_oracle_results_tree(root, year=2024, doy=132):
    day_dir = root / "positioning" / "results" / f"{year}{doy:03d}"
    for method_dir in ("model", "gim"):
        station_dir = day_dir / method_dir / "AIRA"
        station_dir.mkdir(parents=True)
        (station_dir / "AIRA_run.pos").write_text(_POS_FIXTURE)

    products_dir = root / "positioning" / "evaluation" / f"{year}{doy:03d}" / "products"
    products_dir.mkdir(parents=True)
    (products_dir / "IGS0OPSSNX_CRD.SNX").write_text(_SINEX_FIXTURE)
    return day_dir


def test_load_oracle_reads_both_model_and_gim_pos_files(tmp_path):
    experiment_root = tmp_path / "Reference_STEC_Oracle"
    _build_oracle_results_tree(experiment_root)
    results_root = experiment_root / "positioning" / "results"

    oracle = ob.load_oracle(results_root)

    assert set(oracle["Method"]) == {ob.ORACLE_LABEL, "IGS GIM + Mapping (oracle run)"}
    assert (oracle["station"] == "AIRA").all()
    assert (oracle["doy"] == 132).all()


def test_load_oracle_skips_days_with_no_sinex(tmp_path, caplog):
    experiment_root = tmp_path / "Reference_STEC_Oracle"
    day_dir = experiment_root / "positioning" / "results" / "2024132"
    (day_dir / "model" / "AIRA").mkdir(parents=True)
    (day_dir / "model" / "AIRA" / "AIRA_run.pos").write_text(_POS_FIXTURE)
    # No products/SINEX directory created for this day.
    results_root = experiment_root / "positioning" / "results"

    with pytest.raises(FileNotFoundError):
        ob.load_oracle(results_root)


def test_load_oracle_raises_when_nothing_is_found(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError):
        ob.load_oracle(empty_root)


# ---------------------------------------------------------------------------
# load_baselines: elevation weighting only, not the uncertainty-weighted arms
# ---------------------------------------------------------------------------


def test_load_baselines_keeps_only_elevation_weighted_methods(tmp_path):
    summary_path = tmp_path / "weighting_summary.csv"
    pd.DataFrame(
        [
            row("AIRA", 132, "STEC_elev", 1.0),
            row("AIRA", 132, "STEC_iono", 2.0),  # must be dropped
            row("AIRA", 132, "gim_elev", 3.0),
        ]
    ).to_csv(summary_path, index=False)

    baselines = ob.load_baselines(summary_path)

    assert set(baselines["Method"]) == {"Direct STEC", "IGS GIM + Mapping"}
    assert "predicted uncertainty" not in " ".join(baselines["Method"].unique())


# ---------------------------------------------------------------------------
# paired_comparison: restricted to station-days solved by every method present
# ---------------------------------------------------------------------------


def test_paired_comparison_restricts_to_all_methods_solved():
    """gim_elev is missing ZIMM/133, so ZIMM/133 must be dropped from the paired table
    even though the oracle and Direct STEC both solved it."""
    oracle = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": ob.ORACLE_LABEL,
                "error_3d_rms": 1.0,
                "error_2d_rms": 0.5,
                "u_rms": 0.25,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": ob.ORACLE_LABEL,
                "error_3d_rms": 1.2,
                "error_2d_rms": 0.6,
                "u_rms": 0.3,
            },
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": "Direct STEC",
                "error_3d_rms": 2.0,
                "error_2d_rms": 1.0,
                "u_rms": 0.5,
            },
            {
                "station": "AMC4",
                "doy": 132,
                "Method": "IGS GIM + Mapping",
                "error_3d_rms": 3.0,
                "error_2d_rms": 1.5,
                "u_rms": 0.75,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": "Direct STEC",
                "error_3d_rms": 2.5,
                "error_2d_rms": 1.25,
                "u_rms": 0.6,
            },
            # No "IGS GIM + Mapping" row for ZIMM/133 - that station-day is unsolved by it.
        ]
    )

    paired = ob.paired_comparison(oracle, baselines)

    assert list(paired.index) == [("AMC4", 132)]
    assert paired.loc[("AMC4", 132), ob.ORACLE_LABEL] == pytest.approx(1.0)


def test_paired_comparison_applies_the_10m_outlier_rule():
    oracle = pd.DataFrame(
        [
            # AMC4/132: oracle error is an outlier (>10 m) and must be dropped, which
            # then leaves AMC4/132 with only "Direct STEC" - not a complete station-day.
            {
                "station": "AMC4",
                "doy": 132,
                "Method": ob.ORACLE_LABEL,
                "error_3d_rms": 15.0,
                "error_2d_rms": 7.5,
                "u_rms": 3.0,
            },
            # ZIMM/133: both methods present and within the outlier bound.
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": ob.ORACLE_LABEL,
                "error_3d_rms": 1.0,
                "error_2d_rms": 0.5,
                "u_rms": 0.25,
            },
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": "Direct STEC",
                "error_3d_rms": 2.0,
                "error_2d_rms": 1.0,
                "u_rms": 0.5,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": "Direct STEC",
                "error_3d_rms": 2.5,
                "error_2d_rms": 1.25,
                "u_rms": 0.6,
            },
        ]
    )

    paired = ob.paired_comparison(oracle, baselines)

    # AMC4/132 dropped: the oracle's outlier row was excluded, leaving only one method
    # for that station-day. ZIMM/133 survives with both methods.
    assert list(paired.index) == [("ZIMM", 133)]


def test_paired_comparison_reuses_the_named_outlier_constant():
    assert ob.pm.OUTLIER_3D_RMS_M == pm.OUTLIER_3D_RMS_M == 10.0


# ---------------------------------------------------------------------------
# summarise: ratio-to-oracle floor
# ---------------------------------------------------------------------------


def test_summarise_computes_ratio_to_oracle_floor():
    paired = pd.DataFrame(
        {
            ob.ORACLE_LABEL: [1.0, 1.0],
            "Direct STEC": [2.0, 2.0],
        }
    )

    summary = ob.summarise(paired)

    assert summary.loc[ob.ORACLE_LABEL, "ratio_to_oracle"] == pytest.approx(1.0)
    assert summary.loc["Direct STEC", "ratio_to_oracle"] == pytest.approx(2.0)
    assert summary.loc["Direct STEC", "above_oracle_m"] == pytest.approx(1.0)


def test_summarise_has_no_ratio_columns_when_oracle_missing():
    paired = pd.DataFrame({"Direct STEC": [2.0, 3.0]})

    summary = ob.summarise(paired)

    assert "ratio_to_oracle" not in summary.columns


# ---------------------------------------------------------------------------
# check_gim_control: the oracle's own GIM rerun must match the published GIM arm
# ---------------------------------------------------------------------------


def test_check_gim_control_compares_rerun_against_published():
    oracle = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": "IGS GIM + Mapping (oracle run)",
                "error_3d_rms": 3.001,
                "error_2d_rms": 1.5,
                "u_rms": 0.75,
            },
        ]
    )
    baselines = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": "IGS GIM + Mapping",
                "error_3d_rms": 3.000,
                "error_2d_rms": 1.5,
                "u_rms": 0.75,
            },
        ]
    )

    check = ob.check_gim_control(oracle, baselines)

    assert len(check) == 1
    assert (
        check["error_3d_rms_rerun"] - check["error_3d_rms_published"]
    ).abs().max() == pytest.approx(0.001)
