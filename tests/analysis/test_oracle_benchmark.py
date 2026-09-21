"""Tests for `stec.analysis.oracle_benchmark` (R1.8).

`load_oracle` is exercised end to end against synthetic `.pos`/SINEX files, reusing the
same fixture shape as `tests/positioning/test_metrics.py`. The remaining functions
(`paired_comparison`, `summarise`, `check_gim_control`, `station_coverage`,
`station_median_check`) are tested against small in-memory frames, since they operate
purely on already-parsed metrics.

`paired_comparison` now takes a `common_station_days` argument (2026-09-16, matching
this stage to the positioning tables) rather than computing its own population and
applying the 10 m outcome-based exclusion - see the module's own docstring. Tests below
pass a synthetic `pd.MultiIndex` directly rather than depending on the production
`coverage_common_station_days()` path, the same parameter-injection pattern
`test_weighting_ablation.py`/`test_storm_stratification.py` use for the identical
restriction.
"""

from __future__ import annotations

import pandas as pd
import pytest

from stec.analysis import oracle_benchmark as ob

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
# load_oracle's day-coverage assertion: whole days silently vanishing, the shape of the
# 2026-08-25 regression (166 of 242 days aggregated nothing while `pipeline status` still
# reported success), not partial station-day loss within a day.
# ---------------------------------------------------------------------------


def _build_day(root, year, doy, *, with_sinex=True):
    day_dir = root / "positioning" / "results" / f"{year}{doy:03d}"
    for method_dir in ("model", "gim"):
        station_dir = day_dir / method_dir / "AIRA"
        station_dir.mkdir(parents=True)
        (station_dir / "AIRA_run.pos").write_text(_POS_FIXTURE)

    if with_sinex:
        products_dir = (
            root / "positioning" / "evaluation" / f"{year}{doy:03d}" / "products"
        )
        products_dir.mkdir(parents=True)
        (products_dir / "IGS0OPSSNX_CRD.SNX").write_text(_SINEX_FIXTURE)
    return day_dir


def test_load_oracle_passes_when_every_found_day_aggregates(tmp_path):
    experiment_root = tmp_path / "Reference_STEC_Oracle"
    doys = [132, 133, 134, 135, 136]
    for doy in doys:
        _build_day(experiment_root, 2024, doy)
    results_root = experiment_root / "positioning" / "results"

    oracle = ob.load_oracle(results_root)  # must not raise

    assert set(oracle["doy"]) == set(doys)


def test_load_oracle_raises_when_many_days_drop(tmp_path):
    """Reproduces the regression's shape: most found day directories contribute zero
    rows (dangling/missing SINEX), well beyond ALLOWED_MISSING_DAYS. The message must
    name the doys that vanished, not just report a count."""
    experiment_root = tmp_path / "Reference_STEC_Oracle"
    good_doys = [132, 133, 134, 135]
    dropped_doys = [140, 141, 142, 143, 144, 145]
    for doy in good_doys:
        _build_day(experiment_root, 2024, doy, with_sinex=True)
    for doy in dropped_doys:
        _build_day(experiment_root, 2024, doy, with_sinex=False)
    results_root = experiment_root / "positioning" / "results"

    with pytest.raises(AssertionError) as exc_info:
        ob.load_oracle(results_root)

    message = str(exc_info.value)
    for doy in dropped_doys:
        assert f"2024{doy:03d}" in message


def test_load_oracle_passes_when_only_known_missing_days_drop(tmp_path):
    """DOY 303/338/348 (2024) have no positioning products on this host at all (see
    CLAUDE.md and ALLOWED_MISSING_DAYS's own comment) - a day directory that exists but
    never got a SINEX is the closest reproducible shape. Exactly ALLOWED_MISSING_DAYS of
    them dropping must still pass; the assertion's boundary is inclusive."""
    experiment_root = tmp_path / "Reference_STEC_Oracle"
    good_doys = [122, 150, 200, 250, 290, 320, 360]
    missing_doys = [303, 338, 348]
    assert len(missing_doys) == ob.ALLOWED_MISSING_DAYS
    for doy in good_doys:
        _build_day(experiment_root, 2024, doy, with_sinex=True)
    for doy in missing_doys:
        _build_day(experiment_root, 2024, doy, with_sinex=False)
    results_root = experiment_root / "positioning" / "results"

    oracle = ob.load_oracle(results_root)  # must not raise

    assert set(oracle["doy"]) == set(good_doys)


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
# paired_comparison: restricted to the common set, then to station-days solved by every
# method present - no 10 m outcome-based exclusion any more (2026-09-16).
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
    common_station_days = pd.MultiIndex.from_tuples(
        [("AMC4", 132), ("ZIMM", 133)], names=["station", "doy"]
    )

    paired = ob.paired_comparison(oracle, baselines, common_station_days)

    assert list(paired.index) == [("AMC4", 132)]
    assert paired.loc[("AMC4", 132), ob.ORACLE_LABEL] == pytest.approx(1.0)


def test_paired_comparison_restricts_to_the_common_station_days():
    """AMC4/132 is solved by all four methods but is not in the common set (e.g. it
    failed under the uncertainty-weighted arm elsewhere) - it must be dropped even
    though every method present here agrees on it."""
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
                "station": "AMC4",
                "doy": 132,
                "Method": "VTEC + Mapping",
                "error_3d_rms": 3.5,
                "error_2d_rms": 1.75,
                "u_rms": 0.9,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": "Direct STEC",
                "error_3d_rms": 2.5,
                "error_2d_rms": 1.25,
                "u_rms": 0.6,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": "IGS GIM + Mapping",
                "error_3d_rms": 2.8,
                "error_2d_rms": 1.4,
                "u_rms": 0.65,
            },
            {
                "station": "ZIMM",
                "doy": 133,
                "Method": "VTEC + Mapping",
                "error_3d_rms": 3.0,
                "error_2d_rms": 1.5,
                "u_rms": 0.7,
            },
        ]
    )
    # Only ZIMM/133 is in the common set - AMC4/132 is solved by all four methods but
    # excluded here, standing in for a station-day that failed elsewhere in the
    # four-method/both-weighting intersection this common set represents.
    common_station_days = pd.MultiIndex.from_tuples(
        [("ZIMM", 133)], names=["station", "doy"]
    )

    paired = ob.paired_comparison(oracle, baselines, common_station_days)

    assert list(paired.index) == [("ZIMM", 133)]


def test_paired_comparison_keeps_outlier_station_days_no_10m_filter():
    """The 10 m outcome-based exclusion was dropped 2026-09-16 to match the positioning
    tables - a station-day with a >10 m error must now be kept, not silently removed."""
    oracle = pd.DataFrame(
        [
            {
                "station": "AMC4",
                "doy": 132,
                "Method": ob.ORACLE_LABEL,
                "error_3d_rms": 15.0,
                "error_2d_rms": 7.5,
                "u_rms": 3.0,
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
        ]
    )
    common_station_days = pd.MultiIndex.from_tuples(
        [("AMC4", 132)], names=["station", "doy"]
    )

    paired = ob.paired_comparison(oracle, baselines, common_station_days)

    assert list(paired.index) == [("AMC4", 132)]
    assert paired.loc[("AMC4", 132), ob.ORACLE_LABEL] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# summarise: mean- and median-based ratios to the oracle floor, explicitly named
# (2026-09-16) so neither can be mistaken for the other; ratio_to_oracle_median is the
# headline.
# ---------------------------------------------------------------------------


def test_summarise_computes_ratio_to_oracle_floor_mean_and_median():
    paired = pd.DataFrame(
        {
            ob.ORACLE_LABEL: [1.0, 1.0],
            "Direct STEC": [2.0, 2.0],
        }
    )

    summary = ob.summarise(paired)

    assert summary.loc[ob.ORACLE_LABEL, "ratio_to_oracle_median"] == pytest.approx(1.0)
    assert summary.loc["Direct STEC", "ratio_to_oracle_median"] == pytest.approx(2.0)
    assert summary.loc["Direct STEC", "above_oracle_median_m"] == pytest.approx(1.0)
    # Constant series here, so mean and median coincide - both columns must agree.
    assert summary.loc["Direct STEC", "ratio_to_oracle_mean"] == pytest.approx(2.0)
    assert summary.loc["Direct STEC", "above_oracle_mean_m"] == pytest.approx(1.0)


def test_summarise_mean_and_median_ratios_diverge_on_a_skewed_sample():
    """A single large outlier in the oracle arm should distort ratio_to_oracle_mean
    without moving ratio_to_oracle_median much - the exact URUM/DOY 365 shape the module
    docstring documents (dropping the 10 m filter lets one PPPx solve failure dominate
    the oracle's mean floor)."""
    paired = pd.DataFrame(
        {
            ob.ORACLE_LABEL: [0.07, 0.07, 0.07, 0.07, 100.0],
            "Direct STEC": [0.8, 0.8, 0.8, 0.8, 0.8],
        }
    )

    summary = ob.summarise(paired)

    # The median floor barely moves (still 0.07), so Direct STEC's median ratio stays
    # large; the mean floor is dragged up by the outlier, collapsing the mean ratio.
    assert summary.loc["Direct STEC", "ratio_to_oracle_median"] > 10
    assert summary.loc["Direct STEC", "ratio_to_oracle_mean"] < 1


def test_summarise_has_no_ratio_columns_when_oracle_missing():
    paired = pd.DataFrame({"Direct STEC": [2.0, 3.0]})

    summary = ob.summarise(paired)

    assert "ratio_to_oracle_mean" not in summary.columns
    assert "ratio_to_oracle_median" not in summary.columns


# ---------------------------------------------------------------------------
# station_coverage / station_median_check: the coverage-imbalance diagnostics
# ---------------------------------------------------------------------------


def test_station_coverage_counts_station_days_per_station():
    paired = pd.DataFrame(
        {ob.ORACLE_LABEL: [1.0, 1.0, 1.0]},
        index=pd.MultiIndex.from_tuples(
            [("AMC4", 132), ("AMC4", 133), ("ZIMM", 133)],
            names=["station", "doy"],
        ),
    )

    coverage = ob.station_coverage(paired)

    assert coverage.loc["AMC4"] == 2
    assert coverage.loc["ZIMM"] == 1
    # Sorted descending, so the best-covered station comes first.
    assert list(coverage.index)[0] == "AMC4"


def test_station_median_check_matches_pooled_median_for_a_single_station():
    """With one station, the pooled median and the "median of station medians" must be
    identical by construction - this is the degenerate case that pins the shape."""
    paired = pd.DataFrame(
        {ob.ORACLE_LABEL: [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [("AMC4", 132), ("AMC4", 133), ("AMC4", 134)],
            names=["station", "doy"],
        ),
    )

    check = ob.station_median_check(paired)

    assert check.loc[ob.ORACLE_LABEL, "pooled_median"] == pytest.approx(2.0)
    assert check.loc[ob.ORACLE_LABEL, "median_of_station_medians"] == pytest.approx(2.0)
    assert check.loc[ob.ORACLE_LABEL, "n_stations"] == 1


def test_station_median_check_diverges_when_one_station_dominates():
    """One well-covered station (AMC4, 5 days) against one thinly-covered station
    (ZIMM, 1 day) with a very different error - the pooled median is set almost
    entirely by AMC4, while the median-of-station-medians weights ZIMM equally."""
    paired = pd.DataFrame(
        {ob.ORACLE_LABEL: [1.0, 1.0, 1.0, 1.0, 1.0, 9.0]},
        index=pd.MultiIndex.from_tuples(
            [
                ("AMC4", 130),
                ("AMC4", 131),
                ("AMC4", 132),
                ("AMC4", 133),
                ("AMC4", 134),
                ("ZIMM", 133),
            ],
            names=["station", "doy"],
        ),
    )

    check = ob.station_median_check(paired)

    assert check.loc[ob.ORACLE_LABEL, "pooled_median"] == pytest.approx(1.0)
    # median of {AMC4's median=1.0, ZIMM's median=9.0} = 5.0
    assert check.loc[ob.ORACLE_LABEL, "median_of_station_medians"] == pytest.approx(5.0)


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
