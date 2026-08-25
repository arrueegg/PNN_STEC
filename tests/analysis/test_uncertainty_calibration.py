"""Pins the closed-form CRPS formulas against numerical integration, checks that
coverage of synthetic, perfectly-calibrated samples converges to nominal for both
predictive families, and pins the headline defect this port fixes: scoring the VTEC
baseline's stored uncertainty as a Gaussian standard deviation over-reports coverage at
nominal 50%, while scoring it as the Laplace it actually is does not.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from stec.analysis import uncertainty_calibration as uc
from stec.inference import prediction_store as ps


# --- CRPS closed forms vs numerical integration ---------------------------------------


def numerical_crps(cdf, y: float, lower: float, upper: float) -> float:
    """CRPS from its definition, integral (F(x) - 1{x >= y})**2 dx, evaluated by
    quadrature over a range wide enough that the integrand is negligible outside it."""
    integrand = lambda x: (cdf(x) - (x >= y)) ** 2  # noqa: E731
    value, _ = integrate.quad(integrand, lower, upper, points=[y], limit=200)
    return value


GAUSSIAN_TRIPLES = [
    (0.0, 1.0, 0.0),
    (2.0, 1.0, 5.0),
    (-3.0, 2.5, 1.0),
    (10.0, 0.3, 9.4),
]
LAPLACE_TRIPLES = [(0.0, 1.0, 0.0), (2.0, 1.0, 5.0), (-3.0, 2.5, 1.0), (10.0, 0.3, 9.4)]


@pytest.mark.parametrize("mu,sigma,y", GAUSSIAN_TRIPLES)
def test_gaussian_crps_matches_numerical_integration(mu, sigma, y):
    closed_form = uc.gaussian_crps(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    numerical = numerical_crps(
        lambda x: norm.cdf((x - mu) / sigma), y, mu - 20 * sigma, mu + 20 * sigma
    )
    assert closed_form == pytest.approx(numerical, abs=1e-5)


@pytest.mark.parametrize("mu,std,y", LAPLACE_TRIPLES)
def test_laplace_crps_matches_numerical_integration(mu, std, y):
    """`std` is what the store carries (Laplace std = sqrt(2) * scale); the closed
    form and the numerical CDF both take that std directly, as `laplace_crps` does."""
    scale = std / np.sqrt(2.0)

    def laplace_cdf(x):
        z = x - mu
        return 0.5 + 0.5 * np.sign(z) * (1 - np.exp(-np.abs(z) / scale))

    closed_form = uc.laplace_crps(np.array([y]), np.array([mu]), np.array([std]))[0]
    numerical = numerical_crps(laplace_cdf, y, mu - 30 * scale, mu + 30 * scale)
    assert closed_form == pytest.approx(numerical, abs=1e-5)


# --- Coverage of perfectly-calibrated synthetic samples converges to nominal ----------


def test_gaussian_coverage_converges_to_nominal():
    rng = np.random.default_rng(0)
    n = 200_000
    mu = rng.uniform(-10, 10, n)
    sigma = rng.uniform(0.5, 3.0, n)
    y = rng.normal(mu, sigma)

    accumulator = uc.CalibrationAccumulator("gaussian")
    accumulator.update(y, mu, sigma)
    coverage = accumulator.coverage_table().set_index("nominal")["empirical"]
    for level in uc.NOMINAL_LEVELS:
        assert coverage[level] == pytest.approx(level, abs=0.01)


def test_laplace_coverage_converges_to_nominal():
    rng = np.random.default_rng(1)
    n = 200_000
    mu = rng.uniform(-10, 10, n)
    scale = rng.uniform(0.5, 3.0, n)
    y = mu + rng.laplace(0.0, scale, n)
    # The store's convention: the column passed in is the Laplace *standard
    # deviation* (sqrt(2) * scale), not the scale itself - update() expects that.
    std = scale * np.sqrt(2.0)

    accumulator = uc.CalibrationAccumulator("laplace")
    accumulator.update(y, mu, std)
    coverage = accumulator.coverage_table().set_index("nominal")["empirical"]
    for level in uc.NOMINAL_LEVELS:
        assert coverage[level] == pytest.approx(level, abs=0.01)


# --- The headline check: Gaussian quantiles over-report coverage for Laplace data -----


def test_gaussian_scoring_overreports_coverage_of_laplace_residuals_at_nominal_50():
    """Construct residuals that are genuinely Laplace-distributed with a known scale,
    stored the way the prediction store stores it (as a standard deviation). Scoring
    with Gaussian quantiles must read a coverage well above the true 50% at nominal
    50%; scoring with (correctly-scaled) Laplace quantiles must read close to 50%.
    This pins the exact defect the family correction fixes - see the module docstring
    for the real-data magnitude (90% Gaussian vs 82% Laplace).
    """
    rng = np.random.default_rng(2)
    n = 500_000
    mu = np.zeros(n)
    true_scale = 4.0
    y = mu + rng.laplace(0.0, true_scale, n)
    stored_std = np.full(n, true_scale * np.sqrt(2.0))

    gaussian_scored = uc.CalibrationAccumulator("gaussian")
    gaussian_scored.update(y, mu, stored_std)
    laplace_scored = uc.CalibrationAccumulator("laplace")
    laplace_scored.update(y, mu, stored_std)

    gaussian_coverage_50 = (
        gaussian_scored.coverage_table().set_index("nominal").loc[0.50, "empirical"]
    )
    laplace_coverage_50 = (
        laplace_scored.coverage_table().set_index("nominal").loc[0.50, "empirical"]
    )

    # Laplace scoring recovers the true scale exactly (stored_std / sqrt(2) ==
    # true_scale), so it must land on the nominal level.
    assert laplace_coverage_50 == pytest.approx(0.50, abs=0.01)
    # Gaussian scoring treats the same std as a Gaussian sigma without the sqrt(2)
    # correction, which widens its interval beyond what the Laplace data needs -
    # over-reporting coverage relative to both the nominal level and the correct score.
    assert gaussian_coverage_50 > 0.55
    assert gaussian_coverage_50 > laplace_coverage_50 + 0.05


def test_pit_ks_distance_is_small_when_calibrated_and_large_when_not():
    rng = np.random.default_rng(3)
    n = 100_000
    mu = np.zeros(n)
    sigma = np.full(n, 2.0)

    calibrated = uc.CalibrationAccumulator("gaussian")
    calibrated.update(rng.normal(0.0, 2.0, n), mu, sigma)
    assert calibrated.pit_ks_distance() < 0.02

    # Same data scored as if it were Laplace: PIT is no longer uniform.
    miscalibrated = uc.CalibrationAccumulator("laplace")
    miscalibrated.update(rng.normal(0.0, 2.0, n), mu, sigma)
    assert miscalibrated.pit_ks_distance() > 0.02


# --- Streaming accumulation equals whole-frame computation on a synthetic store -------


def day_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_stec = rng.uniform(0, 60, rows)
    stec_sigma = rng.uniform(0.5, 3.0, rows)
    vtec_std = rng.uniform(0.5, 3.0, rows)
    return pd.DataFrame(
        {
            "station": ["AMC4"] * rows,
            "sat": ["G01"] * rows,
            "satele": rng.uniform(5, 90, rows),
            "true_stec": true_stec,
            "stec_pred": true_stec + rng.normal(0, stec_sigma),
            "pred_total_unc": stec_sigma,
            "vtec_model_stec": true_stec + rng.laplace(0, vtec_std / np.sqrt(2.0)),
            "vtec_model_stec_total_unc": vtec_std,
        }
    )


def test_streaming_accumulation_matches_whole_frame_computation(tmp_path):
    days = [(2024, 132, 4_000, 10), (2024, 133, 9_000, 11), (2024, 134, 1_500, 12)]
    frames = []
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames.append(frame)
        ps.write_predictions(frame, "finetuned_stec", "own", year, doy, root=tmp_path)

    results = uc.accumulate("finetuned_stec", "own", tmp_path)

    whole = pd.concat(frames, ignore_index=True)
    for name, (mean_col, scale_col, _native) in uc.PRODUCTS.items():
        y = whole["true_stec"].to_numpy(float)
        mu = whole[mean_col].to_numpy(float)
        sigma = whole[scale_col].to_numpy(float)
        for family in uc.FAMILIES:
            direct = uc.CalibrationAccumulator(family)
            direct.update(y, mu, sigma)

            # No storm_doys was passed, so "all" is the only regime accumulated.
            streamed = results["all"][name][family]
            assert streamed.n == direct.n == len(whole)
            # rel tolerance loosened to float32 precision: the store casts numeric
            # columns to float32 on write, so the round trip alone costs ~1e-7 relative
            # (see daily_metrics' equivalent test for the same reasoning).
            assert streamed.scores()["CRPS"] == pytest.approx(
                direct.scores()["CRPS"], rel=1e-6
            )
            assert streamed.scores()["RMSE"] == pytest.approx(
                direct.scores()["RMSE"], rel=1e-6
            )
            pd.testing.assert_frame_equal(
                streamed.coverage_table(), direct.coverage_table(), check_exact=False
            )


def test_missing_product_column_is_skipped_not_errored(tmp_path):
    """A run with no VTEC comparison must drop that product rather than crash."""
    frame = day_frame(2_000, seed=20).drop(
        columns=["vtec_model_stec", "vtec_model_stec_total_unc"]
    )
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 132, root=tmp_path)

    results = uc.accumulate("finetuned_stec", "own", tmp_path)
    assert "VTEC + Mapping" not in results["all"]
    assert "Direct STEC" in results["all"]


def test_accumulate_raises_when_store_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        uc.accumulate("finetuned_stec", "own", tmp_path)


# --- Small structural checks on the output tables --------------------------------------


def test_coverage_table_tags_native_family(tmp_path):
    frame = day_frame(2_000, seed=30)
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 132, root=tmp_path)
    results = uc.accumulate("finetuned_stec", "own", tmp_path)

    coverage = uc.coverage_table(results)
    # No storm_doys was passed, so "all" is the only regime in the output.
    assert set(coverage["regime"]) == {"all"}
    direct_stec_native = coverage[
        (coverage["model"] == "Direct STEC") & coverage["native"]
    ]
    assert set(direct_stec_native["family"]) == {"gaussian"}
    vtec_native = coverage[(coverage["model"] == "VTEC + Mapping") & coverage["native"]]
    assert set(vtec_native["family"]) == {"laplace"}
    # Every (model, family) pair reports all four required nominal levels.
    assert set(uc.NOMINAL_LEVELS) <= set(coverage["nominal"])


# --- Storm/quiet regime split (R1.6: "uncertainty behaviour under ... disturbed
# conditions"), restored on top of the model x family axes above -------------------------


def _write_swi_fixture(path, doy_to_dst_min: dict[int, float]) -> None:
    """A minimal OMNI-shaped archive: one hourly row per day, holding only the daily
    minimum Dst `load_storm_doys` actually reads (real files carry 24 rows and more
    columns; a single row reproduces the same `nanmin` result more simply)."""

    with h5py.File(path, "w") as handle:
        group = handle.create_group("2024")
        for doy, dst_min in doy_to_dst_min.items():
            dataset = group.create_dataset(f"{doy:03d}", data=np.array([[dst_min]]))
            dataset.attrs["columns"] = ["Dst-index,_nT"]


def test_load_storm_doys_applies_the_daily_minimum_dst_threshold(tmp_path):
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture(
        swi_path,
        {
            130: -49.9,  # quiet: just above (less negative than) the threshold
            131: -50.0,  # storm: exactly at the threshold
            132: -80.0,  # storm: well past it
        },
    )

    storm_doys = uc.load_storm_doys(swi_path, 2024)

    assert storm_doys == {131, 132}
    # Matches storm_stratification.STORM_DST_THRESHOLD_NT so a day classified as storm
    # here is a storm there too.
    assert uc.STORM_DST_THRESHOLD == -50.0


def test_load_storm_doys_returns_none_when_archive_missing(tmp_path):
    assert uc.load_storm_doys(tmp_path / "does_not_exist.h5", 2024) is None


def _write_swi_fixture_multi_year(
    path, year_to_doy_dst: dict[int, dict[int, float]]
) -> None:
    """Multi-year variant of `_write_swi_fixture`: one OMNI-shaped group per year."""
    with h5py.File(path, "w") as handle:
        for year, doy_to_dst_min in year_to_doy_dst.items():
            group = handle.create_group(str(year))
            for doy, dst_min in doy_to_dst_min.items():
                dataset = group.create_dataset(f"{doy:03d}", data=np.array([[dst_min]]))
                dataset.attrs["columns"] = ["Dst-index,_nT"]


def test_load_storm_doys_by_year_classifies_each_year_from_its_own_dst_record(
    tmp_path,
):
    """DOY 131 is a storm in 2016 but quiet in 2024 - the same day-of-year must not
    borrow another year's Dst record, which is exactly the bug a flat cross-year
    storm_doys set would reintroduce."""
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture_multi_year(
        swi_path,
        {
            2016: {130: -10.0, 131: -80.0},  # 131 is a storm in 2016
            2024: {130: -10.0, 131: -20.0},  # 131 is quiet in 2024
        },
    )

    storm_doys = uc.load_storm_doys_by_year(swi_path, [2016, 2024])

    assert storm_doys == {2016: {131}, 2024: set()}


def test_load_storm_doys_by_year_skips_a_year_absent_from_the_archive(tmp_path):
    """A year requested but not present in the OMNI archive is dropped with a warning,
    not raised on and not silently mislabelled from a neighbouring year."""
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture_multi_year(swi_path, {2024: {131: -80.0}})

    storm_doys = uc.load_storm_doys_by_year(swi_path, [2016, 2024])

    assert storm_doys == {2024: {131}}
    assert 2016 not in storm_doys


def test_load_storm_doys_by_year_returns_none_when_archive_missing(tmp_path):
    assert uc.load_storm_doys_by_year(tmp_path / "does_not_exist.h5", [2024]) is None


def test_load_storm_doys_by_year_returns_none_when_no_requested_year_is_present(
    tmp_path,
):
    swi_path = tmp_path / "omni.h5"
    _write_swi_fixture_multi_year(swi_path, {2024: {131: -80.0}})
    assert uc.load_storm_doys_by_year(swi_path, [2017]) is None


def regime_day_frame(rows: int, seed: int) -> pd.DataFrame:
    """Same shape as `day_frame`, but every mean is offset by `seed` so quiet-day and
    storm-day accumulators are trivially distinguishable in the assertions below."""
    frame = day_frame(rows, seed)
    frame["stec_pred"] += seed
    return frame


def test_regime_split_is_an_additional_axis_not_a_replacement(tmp_path):
    """Passing storm_doys must add "quiet"/"storm" entries alongside "all", and the
    quiet + storm accumulators must partition "all" exactly - same rows, split by day,
    not a second, independent pass over the store."""
    quiet_frame = regime_day_frame(3_000, seed=1)
    storm_frame = regime_day_frame(2_000, seed=2)
    ps.write_predictions(quiet_frame, "finetuned_stec", "own", 2024, 130, root=tmp_path)
    ps.write_predictions(storm_frame, "finetuned_stec", "own", 2024, 131, root=tmp_path)

    results = uc.accumulate("finetuned_stec", "own", tmp_path, storm_doys={2024: {131}})

    assert set(results.keys()) == {"all", "quiet", "storm"}
    all_acc = results["all"]["Direct STEC"]["gaussian"]
    quiet_acc = results["quiet"]["Direct STEC"]["gaussian"]
    storm_acc = results["storm"]["Direct STEC"]["gaussian"]

    assert quiet_acc.n == len(quiet_frame)
    assert storm_acc.n == len(storm_frame)
    assert all_acc.n == quiet_acc.n + storm_acc.n == len(quiet_frame) + len(storm_frame)


def test_regime_split_is_absent_when_storm_doys_not_given(tmp_path):
    frame = day_frame(1_000, seed=5)
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 132, root=tmp_path)

    results = uc.accumulate("finetuned_stec", "own", tmp_path, storm_doys=None)

    assert set(results.keys()) == {"all"}


def test_coverage_and_scores_tables_carry_a_regime_column(tmp_path):
    quiet_frame = regime_day_frame(1_500, seed=1)
    storm_frame = regime_day_frame(1_500, seed=2)
    ps.write_predictions(quiet_frame, "finetuned_stec", "own", 2024, 130, root=tmp_path)
    ps.write_predictions(storm_frame, "finetuned_stec", "own", 2024, 131, root=tmp_path)

    results = uc.accumulate("finetuned_stec", "own", tmp_path, storm_doys={2024: {131}})
    coverage = uc.coverage_table(results)
    scores = uc.scores_table(results)

    assert set(coverage["regime"]) == {"all", "quiet", "storm"}
    assert set(scores["regime"]) == {"all", "quiet", "storm"}
    # Every regime still reports every (model, family) combination.
    for regime in ("all", "quiet", "storm"):
        assert set(scores.loc[scores["regime"] == regime, "model"]) == set(uc.PRODUCTS)


# --- Multi-year coverage (the 44%-of-data defect this session fixes) ------------------
#
# `pretrained_stec/own` spans 2014-2024, not just the 2024 that `finetuned_stec/own`
# holds. `accumulate()`'s own `years` parameter already accepted `None` to mean "every
# year present" before this fix - the defect lived entirely in `main()`'s `--year`
# argparse default (hardcoded to 2024) and the stage command that never overrode it, so
# the regression tests below exercise `main()` itself, not just `accumulate()`.


def test_available_years_lists_years_present_optionally_restricted_to_doys(tmp_path):
    ps.write_predictions(
        day_frame(10, seed=1), "pretrained_stec", "own", 2016, 130, root=tmp_path
    )
    ps.write_predictions(
        day_frame(10, seed=2), "pretrained_stec", "own", 2016, 200, root=tmp_path
    )
    ps.write_predictions(
        day_frame(10, seed=3), "pretrained_stec", "own", 2024, 130, root=tmp_path
    )

    assert uc._available_years("pretrained_stec", "own", tmp_path, doys=None) == [
        2016,
        2024,
    ]
    assert uc._available_years("pretrained_stec", "own", tmp_path, doys=[200]) == [2016]


def _run_main(monkeypatch, argv: list[str]) -> None:
    monkeypatch.setattr("sys.argv", ["uncertainty_calibration", *argv])
    uc.main()


def test_main_default_scores_every_year_present_not_just_2024(tmp_path, monkeypatch):
    """This is the exact defect: with no `--year` passed - the shape of the real
    `uncertainty_calibration_pretrained` stage command - `main()` used to default to
    2024 alone and silently drop every other year in the partition. It must now cover
    every year on disk. `--swi-path` points at a nonexistent file so the run degrades to
    the unstratified "all" regime only, keeping this test about year coverage rather
    than the storm/quiet split (covered separately below).
    """
    old_year_frame = day_frame(1_000, seed=40)
    new_year_frame = day_frame(1_200, seed=41)
    ps.write_predictions(
        old_year_frame, "pretrained_stec", "own", 2016, 130, root=tmp_path
    )
    ps.write_predictions(
        new_year_frame, "pretrained_stec", "own", 2024, 130, root=tmp_path
    )
    output_dir = tmp_path / "output"

    _run_main(
        monkeypatch,
        [
            "--store-root",
            str(tmp_path),
            "--model-variant",
            "pretrained_stec",
            "--dataset",
            "own",
            "--swi-path",
            str(tmp_path / "no_such_omni.h5"),
            "--output-dir",
            str(output_dir),
        ],
    )

    scores = pd.read_csv(output_dir / "pretrained_stec_own" / "scores.csv")
    row = scores[
        (scores["regime"] == "all")
        & (scores["model"] == "Direct STEC")
        & (scores["family"] == "gaussian")
    ]
    assert len(row) == 1
    assert int(row["observations"].iloc[0]) == len(old_year_frame) + len(new_year_frame)


def test_main_year_flag_still_scopes_to_one_year_on_request(tmp_path, monkeypatch):
    """`--year` must still work for a caller that deliberately wants one year - it just
    must not be the silent default a paper artifact gets by doing nothing."""
    old_year_frame = day_frame(900, seed=42)
    new_year_frame = day_frame(1_100, seed=43)
    ps.write_predictions(
        old_year_frame, "pretrained_stec", "own", 2016, 130, root=tmp_path
    )
    ps.write_predictions(
        new_year_frame, "pretrained_stec", "own", 2024, 130, root=tmp_path
    )
    output_dir = tmp_path / "output"

    _run_main(
        monkeypatch,
        [
            "--store-root",
            str(tmp_path),
            "--model-variant",
            "pretrained_stec",
            "--dataset",
            "own",
            "--year",
            "2016",
            "--swi-path",
            str(tmp_path / "no_such_omni.h5"),
            "--output-dir",
            str(output_dir),
        ],
    )

    scores = pd.read_csv(output_dir / "pretrained_stec_own" / "scores.csv")
    row = scores[
        (scores["regime"] == "all")
        & (scores["model"] == "Direct STEC")
        & (scores["family"] == "gaussian")
    ]
    assert int(row["observations"].iloc[0]) == len(old_year_frame)


def test_own_dataset_default_scope_matches_explicit_single_year_scope(tmp_path):
    """`finetuned_stec/own` only ever holds one year, so the new "every year present"
    default must be numerically identical to the old explicit `years=[2024]` call - this
    is what pins that the fix leaves the own-dataset invocation's numbers unchanged."""
    frame = day_frame(2_000, seed=50)
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 132, root=tmp_path)

    default_results = uc.accumulate("finetuned_stec", "own", tmp_path)
    pinned_year_results = uc.accumulate("finetuned_stec", "own", tmp_path, years=[2024])

    pd.testing.assert_frame_equal(
        uc.scores_table(default_results), uc.scores_table(pinned_year_results)
    )


# --- Storm/quiet pooled correctly across years -----------------------------------------


def test_regime_split_looks_up_storm_status_per_year_not_flat_doy(tmp_path):
    """DOY 131 is a storm in 2016 but quiet in 2024 in this fixture. A flat, doy-only
    storm_doys set (the pre-fix shape) cannot represent that at all - it would apply one
    year's label to the other. The per-year dict must classify each year independently
    and then pool "storm" (and "quiet") observations across years, the same way "all"
    already pools every year."""
    storm_2016 = regime_day_frame(1_000, seed=10)
    quiet_2024 = regime_day_frame(1_200, seed=11)
    ps.write_predictions(storm_2016, "pretrained_stec", "own", 2016, 131, root=tmp_path)
    ps.write_predictions(quiet_2024, "pretrained_stec", "own", 2024, 131, root=tmp_path)

    results = uc.accumulate(
        "pretrained_stec",
        "own",
        tmp_path,
        years=None,
        allow_multi_year=True,
        storm_doys={2016: {131}, 2024: set()},
    )

    storm_acc = results["storm"]["Direct STEC"]["gaussian"]
    quiet_acc = results["quiet"]["Direct STEC"]["gaussian"]
    all_acc = results["all"]["Direct STEC"]["gaussian"]

    assert storm_acc.n == len(storm_2016)
    assert quiet_acc.n == len(quiet_2024)
    assert all_acc.n == storm_acc.n + quiet_acc.n


def test_regime_split_treats_a_year_missing_from_storm_doys_as_all_only(tmp_path):
    """A year present in the store but absent from the storm_doys dict (e.g. the OMNI
    archive didn't cover it) must not be silently folded into "quiet" - it is
    unclassified, not known-quiet, so its days stay out of both stratified regimes."""
    unclassified_year = regime_day_frame(800, seed=12)
    classified_year = regime_day_frame(600, seed=13)
    ps.write_predictions(
        unclassified_year, "pretrained_stec", "own", 2016, 131, root=tmp_path
    )
    ps.write_predictions(
        classified_year, "pretrained_stec", "own", 2024, 131, root=tmp_path
    )

    results = uc.accumulate(
        "pretrained_stec",
        "own",
        tmp_path,
        years=None,
        allow_multi_year=True,
        storm_doys={2024: set()},  # 2016 absent entirely
    )

    all_acc = results["all"]["Direct STEC"]["gaussian"]
    quiet_acc = results["quiet"]["Direct STEC"]["gaussian"]
    assert all_acc.n == len(unclassified_year) + len(classified_year)
    assert quiet_acc.n == len(classified_year)
    assert "storm" not in results


def _calibrated_sample(family: str, n: int = 200_000, seed: int = 7):
    """Residuals drawn from exactly the family they will be scored under."""
    rng = np.random.default_rng(seed)
    sigma = rng.uniform(0.5, 6.0, n)
    if family == "gaussian":
        return rng.normal(0.0, sigma), sigma
    return rng.laplace(0.0, sigma / np.sqrt(2.0)), sigma


@pytest.mark.parametrize("family", ["gaussian", "laplace"])
def test_nominal_levels_include_99_and_track_a_calibrated_sample(family):
    # 0.99 is the level at which an over-confident model is most visible, and it was
    # dropped from NOMINAL_LEVELS during the port.
    assert 0.99 in uc.NOMINAL_LEVELS
    y, sigma = _calibrated_sample(family)
    accumulator = uc.CalibrationAccumulator(family)
    accumulator.update(y, np.zeros_like(y), sigma)
    coverage = accumulator.coverage_table().set_index("nominal")["empirical"]
    for level in uc.NOMINAL_LEVELS:
        assert abs(coverage[level] - level) < 0.005


@pytest.mark.parametrize("family", ["gaussian", "laplace"])
def test_per_observation_uncertainty_beats_the_constant_scale_reference(family):
    """The reference is what makes CRPS interpretable, so it must be beatable."""
    y, sigma = _calibrated_sample(family)
    accumulator = uc.CalibrationAccumulator(family)
    accumulator.update(y, np.zeros_like(y), sigma)
    scores = accumulator.scores()
    assert scores["CRPS"] < scores["CRPS_constant_scale"]


@pytest.mark.parametrize("family", ["gaussian", "laplace"])
def test_a_constant_scale_predictor_scores_its_own_reference(family):
    """A model emitting one scale everywhere *is* the reference, so the two must agree.

    This is what pins the reference to the right parameterisation: a Laplace scored at
    scale = RMSE rather than RMSE/sqrt(2) fails here by roughly 8%.
    """
    y, sigma = _calibrated_sample(family)
    constant = np.full_like(y, float(np.sqrt(np.mean(sigma**2))))
    accumulator = uc.CalibrationAccumulator(family)
    accumulator.update(y, np.zeros_like(y), constant)
    scores = accumulator.scores()
    assert scores["CRPS"] == pytest.approx(scores["CRPS_constant_scale"], rel=1e-3)
