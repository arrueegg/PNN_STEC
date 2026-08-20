"""Pins the closed-form CRPS formulas against numerical integration, checks that
coverage of synthetic, perfectly-calibrated samples converges to nominal for both
predictive families, and pins the headline defect this port fixes: scoring the VTEC
baseline's stored uncertainty as a Gaussian standard deviation over-reports coverage at
nominal 50%, while scoring it as the Laplace it actually is does not.
"""

from __future__ import annotations

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

            streamed = results[name][family]
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
    assert "VTEC + Mapping" not in results
    assert "Direct STEC" in results


def test_accumulate_raises_when_store_is_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        uc.accumulate("finetuned_stec", "own", tmp_path)


# --- Small structural checks on the output tables --------------------------------------


def test_coverage_table_tags_native_family(tmp_path):
    frame = day_frame(2_000, seed=30)
    ps.write_predictions(frame, "finetuned_stec", "own", 2024, 132, root=tmp_path)
    results = uc.accumulate("finetuned_stec", "own", tmp_path)

    coverage = uc.coverage_table(results)
    direct_stec_native = coverage[
        (coverage["model"] == "Direct STEC") & coverage["native"]
    ]
    assert set(direct_stec_native["family"]) == {"gaussian"}
    vtec_native = coverage[(coverage["model"] == "VTEC + Mapping") & coverage["native"]]
    assert set(vtec_native["family"]) == {"laplace"}
    # Every (model, family) pair reports all four required nominal levels.
    assert set(uc.NOMINAL_LEVELS) <= set(coverage["nominal"])
