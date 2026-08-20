"""Pins the three things most likely to go silently wrong in this port.

1. **The Laplace sqrt(2) conversion.** `vtec_model_stec_total_unc` is stored as a
   Laplace *standard deviation*, not its scale; `laplace_crps`/`half_width` must divide
   by `sqrt(2)` before using it as a Laplace scale. Checked against numerical
   integration of the CRPS definition, and against the wrong (un-converted) formula to
   show the two actually differ.
2. **round(), never int(), when a year/doy pair becomes an IONEX date.** The module
   must route through `GIMMapper.load_for_year_doy` rather than reintroducing a
   truncating `datetime.strptime(f"{year}-{doy:03d}", ...)`-style conversion.
3. **Streaming equals whole-frame.** `pool()` recombines per-day `diagnostics()` rows;
   for RMSE and the plain per-observation means that must reproduce the same numbers a
   single `diagnostics()` call on the concatenated data would give, which is the
   entire justification for scoring the store one day at a time.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.stats import norm

from stec.analysis import ionex_rms_benchmark as irb
from stec.baselines.gim import GIMMapper, date_from_year_doy

# --------------------------------------------------------------------------
# 1. Laplace sqrt(2) conversion
# --------------------------------------------------------------------------


def numerical_crps(cdf, y: float, lower: float, upper: float) -> float:
    """CRPS from its definition, integral (F(x) - 1{x >= y})**2 dx, evaluated by
    quadrature over a range wide enough that the integrand is negligible outside it."""
    integrand = lambda x: (cdf(x) - (x >= y)) ** 2  # noqa: E731
    value, _ = integrate.quad(integrand, lower, upper, points=[y], limit=200)
    return value


@pytest.mark.parametrize(
    "mu,std,y", [(0.0, 1.0, 0.0), (2.0, 1.0, 5.0), (-3.0, 2.5, 1.0), (10.0, 0.3, 9.4)]
)
def test_laplace_crps_matches_numerical_integration_of_the_std_parameterised_cdf(
    mu, std, y
):
    """`std` is what the store carries. The closed form takes `std` directly; the
    numerical check builds its CDF from `scale = std / sqrt(2)`, so agreement here
    is exactly the evidence that the closed form performs that same conversion."""
    scale = std / np.sqrt(2.0)

    def laplace_cdf(x):
        z = x - mu
        return 0.5 + 0.5 * np.sign(z) * (1 - np.exp(-np.abs(z) / scale))

    closed_form = irb.laplace_crps(np.array([y]), np.array([mu]), np.array([std]))[0]
    numerical = numerical_crps(laplace_cdf, y, mu - 30 * scale, mu + 30 * scale)
    assert closed_form == pytest.approx(numerical, abs=1e-5)


def test_laplace_crps_would_differ_if_std_were_used_as_the_scale_directly():
    """Documents the defect this pins against: treating the stored std as if it were
    already the Laplace scale (skipping the /sqrt(2)) changes the result, so a
    regression that drops the conversion would silently start scoring the wrong
    number rather than crashing."""
    y, mu, std = np.array([5.0]), np.array([0.0]), np.array([4.0])
    correct = irb.laplace_crps(y, mu, std)[0]

    deviation = abs(y[0] - mu[0])
    wrong_scale = std[0]  # the bug: no /sqrt(2)
    wrong = (
        deviation + wrong_scale * np.exp(-deviation / wrong_scale) - 0.75 * wrong_scale
    )
    assert correct != pytest.approx(wrong)


def test_half_width_laplace_uses_the_converted_scale():
    # -(sigma/sqrt(2)) * ln(1 - 0.5), pinned numerically.
    width = irb.half_width(np.array([1.0]), 0.5, "laplace")[0]
    assert width == pytest.approx(-(1.0 / np.sqrt(2.0)) * np.log(0.5), abs=1e-12)


@pytest.mark.parametrize(
    "mu,sigma,y", [(0.0, 1.0, 0.0), (2.0, 1.0, 5.0), (-3.0, 2.5, 1.0)]
)
def test_gaussian_crps_matches_numerical_integration(mu, sigma, y):
    closed_form = irb.gaussian_crps(np.array([y]), np.array([mu]), np.array([sigma]))[0]
    numerical = numerical_crps(
        lambda x: norm.cdf((x - mu) / sigma), y, mu - 20 * sigma, mu + 20 * sigma
    )
    assert closed_form == pytest.approx(numerical, abs=1e-5)


# --------------------------------------------------------------------------
# 2. round(), never int(), for the year/doy -> IONEX date conversion
# --------------------------------------------------------------------------


def test_module_never_reintroduces_a_truncating_strptime_conversion():
    """The source this was ported from built the date with
    `datetime.strptime(f"{year}-{doy:03d}", "%Y-%j")`, which would raise (or worse,
    silently misformat) on a fractional doy rather than rounding it. This module must
    route through the shared rounding helper instead."""
    source = inspect.getsource(irb)
    assert "strptime" not in source
    assert "load_for_year_doy" in inspect.getsource(irb.main)


def test_load_for_year_doy_rounds_a_doy_just_below_the_integer(monkeypatch):
    """The exact float DOY 189 round-trips to through the model's normalise/
    denormalise-in-float32 path (see `stec/baselines/gim.py`); it must resolve to
    DOY 189, not the DOY 188 a truncating int() would give."""
    mapper = GIMMapper(mapping_type=irb.MAPPING_TYPE, gim_type="IGS")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        mapper, "load_gim_data", lambda date, **kw: captured.update(date=date)
    )

    mapper.load_for_year_doy(2024, 188.99998, ionex_root="/unused")

    assert captured["date"].timetuple().tm_yday == 189
    assert int(188.99998) == 188  # what a truncating cast would have given


def test_date_from_year_doy_is_the_one_conversion_site():
    """Sanity check on the shared helper this module leans on: an exact integer day is
    unaffected by rounding."""
    date = date_from_year_doy(2024, 132)
    assert (date.year, date.timetuple().tm_yday) == (2024, 132)


# --------------------------------------------------------------------------
# 3. Streaming (per-day diagnostics + pool) equals whole-frame
# --------------------------------------------------------------------------


def _synthetic_day(
    rows: int, sigma_true: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian-consistent synthetic observations: errors truly drawn from N(0,
    sigma_true), reported sigma constant and equal to the truth, so the day's
    RMSE/coverage/CRPS are unambiguous."""
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0, 40, rows)
    y = mu + rng.normal(0, sigma_true, rows)
    sigma = np.full(rows, sigma_true)
    return y, mu, sigma


def test_pooled_diagnostics_match_direct_whole_frame_computation():
    """`pool()` over per-day `diagnostics()` rows must reproduce RMSE, coverage, CRPS
    and mean_sigma computed directly on the concatenated observations - the exactness
    `iter_days` streaming depends on (see `pool`'s docstring; `spearman` and
    `scale_95` are both quantile/rank statistics that do not decompose additively
    across days and are deliberately not checked for exact equality here)."""
    days = [(132, 4000, 2.0), (133, 9000, 5.0), (134, 500, 1.0)]
    rows = []
    all_y, all_mu, all_sigma = [], [], []
    for doy, n, sigma_true in days:
        y, mu, sigma = _synthetic_day(n, sigma_true, seed=doy)
        all_y.append(y)
        all_mu.append(mu)
        all_sigma.append(sigma)
        rows.append({"doy": doy, **irb.diagnostics(y, mu, sigma, "gaussian")})

    pooled = irb.pool(pd.DataFrame(rows))

    whole = irb.diagnostics(
        np.concatenate(all_y),
        np.concatenate(all_mu),
        np.concatenate(all_sigma),
        "gaussian",
    )

    assert pooled["observations"] == whole["observations"]
    assert pooled["RMSE"] == pytest.approx(whole["RMSE"], rel=1e-9)
    assert pooled["cov_95"] == pytest.approx(whole["cov_95"], rel=1e-9)
    assert pooled["cov_50"] == pytest.approx(whole["cov_50"], rel=1e-9)
    assert pooled["CRPS"] == pytest.approx(whole["CRPS"], rel=1e-9)
    assert pooled["mean_sigma"] == pytest.approx(whole["mean_sigma"], rel=1e-9)
    # scale_95 is a 95th-percentile quantile, not a plain mean, so a count-weighted
    # average of per-day quantiles only approximates the whole-frame quantile -
    # close, but deliberately not asserted exact (see pool()'s docstring).
    assert pooled["scale_95"] == pytest.approx(whole["scale_95"], rel=0.05)


def test_pooled_rmse_differs_from_the_unweighted_mean_of_daily_rmse():
    """Regression guard: a sparse, error-prone day must not be weighted the same as a
    dense one - if `pool()` ever regressed to an unweighted mean of `RMSE`, this would
    stop failing."""
    rows = [
        {
            "doy": 1,
            "observations": 10,
            "RMSE": 10.0,
            "cov_50": 0.5,
            "cov_68": 0.68,
            "cov_90": 0.9,
            "cov_95": 0.95,
            "CRPS": 1.0,
            "CRPS_const": 1.0,
            "mean_sigma": 1.0,
            "scale_95": 1.0,
            "spearman": 0.0,
        },
        {
            "doy": 2,
            "observations": 990,
            "RMSE": 1.0,
            "cov_50": 0.5,
            "cov_68": 0.68,
            "cov_90": 0.9,
            "cov_95": 0.95,
            "CRPS": 1.0,
            "CRPS_const": 1.0,
            "mean_sigma": 1.0,
            "scale_95": 1.0,
            "spearman": 0.0,
        },
    ]
    pooled = irb.pool(pd.DataFrame(rows))
    unweighted_mean = (10.0 + 1.0) / 2
    pooled_expected = np.sqrt((10 * 10.0**2 + 990 * 1.0**2) / 1000)
    assert pooled["RMSE"] == pytest.approx(pooled_expected)
    assert pooled["RMSE"] != pytest.approx(unweighted_mean)


# --------------------------------------------------------------------------
# ionex_path: locating the file for one day
# --------------------------------------------------------------------------


def test_ionex_path_finds_file_under_the_year_subdirectory(tmp_path):
    year_dir = tmp_path / "2024"
    year_dir.mkdir()
    target = year_dir / "igsg1320.24i"
    target.touch()
    assert irb.ionex_path(tmp_path, 2024, 132, "IGS") == target


def test_ionex_path_finds_file_directly_under_the_flat_root(tmp_path):
    target = tmp_path / "codg0100.24i"
    target.touch()
    assert irb.ionex_path(tmp_path, 2024, 10, "CODE") == target


def test_ionex_path_returns_none_when_absent(tmp_path):
    assert irb.ionex_path(tmp_path, 2024, 132, "IGS") is None


# --------------------------------------------------------------------------
# slant_rms: variance-domain interpolation of the IONEX RMS maps
# --------------------------------------------------------------------------


def _constant_rms_data(rms_value: float, lat_descending: bool = False) -> dict:
    lat_grid = np.arange(-90.0, 90.1, 5.0)
    if lat_descending:
        lat_grid = lat_grid[::-1]
    lon_grid = np.arange(-180.0, 180.1, 5.0)
    from datetime import datetime

    epochs = [datetime(2024, 5, 1, 0), datetime(2024, 5, 1, 2)]
    rms_maps = [np.full((len(lat_grid), len(lon_grid)), rms_value) for _ in epochs]
    return {
        "rms_maps": rms_maps,
        "epochs": epochs,
        "lat_grid": lat_grid,
        "lon_grid": lon_grid,
    }


def test_slant_rms_returns_the_constant_at_zenith():
    """A spatially and temporally constant RMS field must interpolate back to itself
    exactly, and at the zenith (elevation 90) the MSLM mapping factor is 1, so the
    slant result equals the vertical RMS."""
    from stec.baselines.gim import MappingFunction

    data = _constant_rms_data(rms_value=2.5)
    frame = pd.DataFrame(
        {
            "lat_ipp": [10.0, -30.0],
            "lon_ipp": [20.0, -170.0],
            "sod": [1800.0, 5400.0],
            "satele": [90.0, 90.0],
        }
    )
    result = irb.slant_rms(data, frame, MappingFunction("MSLM"))
    np.testing.assert_allclose(result, [2.5, 2.5], atol=1e-9)


def test_slant_rms_handles_descending_latitude_grid():
    """IONEX stores latitude descending; the function must flip it internally rather
    than mis-interpolating."""
    from stec.baselines.gim import MappingFunction

    data = _constant_rms_data(rms_value=1.7, lat_descending=True)
    frame = pd.DataFrame(
        {"lat_ipp": [0.0], "lon_ipp": [0.0], "sod": [0.0], "satele": [90.0]}
    )
    result = irb.slant_rms(data, frame, MappingFunction("MSLM"))
    np.testing.assert_allclose(result, [1.7], atol=1e-9)


def test_slant_rms_returns_none_when_no_rms_maps_present():
    data = {
        "rms_maps": [],
        "epochs": [],
        "lat_grid": np.array([]),
        "lon_grid": np.array([]),
    }
    frame = pd.DataFrame(
        {"lat_ipp": [0.0], "lon_ipp": [0.0], "sod": [0.0], "satele": [45.0]}
    )
    from stec.baselines.gim import MappingFunction

    assert irb.slant_rms(data, frame, MappingFunction("MSLM")) is None


# --------------------------------------------------------------------------
# diagnostics: low-sigma / non-finite rows are dropped, not propagated
# --------------------------------------------------------------------------


def test_diagnostics_drops_non_finite_and_degenerate_sigma_rows():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    mu = np.array([1.5, np.nan, 3.5, 4.5])
    sigma = np.array([1.0, 1.0, 0.0, 1.0])  # sigma=0 is below MIN_SIGMA_TECU
    out = irb.diagnostics(y, mu, sigma, "gaussian")
    assert out["observations"] == 2  # only rows 0 and 3 survive


def test_diagnostics_returns_empty_dict_when_nothing_survives():
    y = np.array([np.nan])
    mu = np.array([1.0])
    sigma = np.array([1.0])
    assert irb.diagnostics(y, mu, sigma, "gaussian") == {}
