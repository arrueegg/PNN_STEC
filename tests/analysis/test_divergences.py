"""Tests for `stec.analysis.divergences` - the divergence registry and its pure
measurement functions.

Two things are pinned separately, matching the module's own split:

* The registry's structure (every entry has an id/description/deliverable/status, ids are
  unique, statuses come from a closed set, "measured" carries a number and
  "unmeasurable_now" carries a reason with no number) - checked directly against
  `REGISTRY`, no I/O.
* The pure measurement functions (`doy_lookup_disagrees`, `count_boundary_rows`,
  `classify_storm_days_daily`, `classify_storm_days_combined`, `interval_coverage`) -
  checked against small synthetic inputs with hand-computed answers, independent of the
  registry and of any real data tree.

Deliberately not exercised here: `Divergence.measure()` for the six measured entries
does real I/O against the read-only legacy trees (`stec.config.paths`) - #12 goes further
and loads a real checkpoint for a live forward pass - which is what makes it a genuine,
re-runnable harness rather than a frozen constant, but that also makes it
environment-dependent, so it does not belong in a hermetic unit test. The
`recorded_effect` snapshot each carries is what stays fast and reproducible, and is what
this suite pins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from stec.analysis import divergences as dv

# --- registry structure -----------------------------------------------------------------


def test_every_divergence_has_id_description_deliverable_and_status():
    assert len(dv.REGISTRY) == 12
    for divergence in dv.REGISTRY:
        assert divergence.id
        assert divergence.description
        assert divergence.deliverable
        assert divergence.status in ("measured", "unmeasurable_now")


def test_ids_unique_and_statuses_and_application_come_from_closed_sets():
    ids = [d.id for d in dv.REGISTRY]
    assert len(ids) == len(set(ids)), f"duplicate ids in {ids}"

    allowed_status = {"measured", "unmeasurable_now"}
    allowed_applied = {"applied", "available_but_off", "not_yet_ported"}
    for divergence in dv.REGISTRY:
        assert divergence.status in allowed_status
        assert divergence.applied in allowed_applied


def test_duplicate_id_is_rejected():
    """`_check_unique_ids` is what `test_ids_unique...` above trusts; pin it directly
    against a constructed collision so a future refactor that drops the check is caught
    here rather than only by eyeballing the registry."""
    duplicate = (dv.REGISTRY[0], dv.REGISTRY[0])
    with pytest.raises(ValueError, match="duplicate"):
        dv._check_unique_ids(duplicate)


def test_measured_divergence_carries_numeric_effect():
    measured = [d for d in dv.REGISTRY if d.status == "measured"]
    assert {d.id for d in measured} == {"1", "4", "9", "10", "11", "12"}
    for divergence in measured:
        effect = divergence.recorded_effect
        assert isinstance(effect, dv.MeasuredEffect)
        assert effect.method
        assert len(effect.measurements) > 0
        for m in effect.measurements:
            assert isinstance(m.new_value, (int, float))


def test_unmeasurable_divergence_carries_reason_and_no_number():
    unmeasurable = [d for d in dv.REGISTRY if d.status == "unmeasurable_now"]
    assert {d.id for d in unmeasurable} == {"2", "3", "5", "6", "7", "8"}
    for divergence in unmeasurable:
        effect = divergence.recorded_effect
        assert isinstance(effect, dv.UnmeasurableEffect)
        assert effect.reason
        assert effect.would_require
        # An UnmeasurableEffect has no `measurements`/`new_value` attribute at all - the
        # type itself is the "no number" guarantee, not a value that happens to be None.
        assert not hasattr(effect, "measurements")
        assert not hasattr(effect, "new_value")


def test_by_id_looks_up_and_raises_on_unknown():
    assert dv.by_id("1") is dv.REGISTRY[0]
    with pytest.raises(KeyError):
        dv.by_id("no-such-id")


def test_measured_effect_rejects_empty_measurements():
    with pytest.raises(ValueError, match="at least one Measurement"):
        dv.MeasuredEffect(method="x", measurements=())


def test_unmeasurable_effect_construction_requires_reason_and_requirement():
    with pytest.raises(ValueError, match="reason"):
        dv.Divergence(
            id="test-empty-reason",
            description="d",
            deliverable="x",
            reviewer_comment=None,
            applied="applied",
            recorded_effect=dv.UnmeasurableEffect(
                reason="", would_require="do a thing"
            ),
            measure=lambda: dv.UnmeasurableEffect(
                reason="", would_require="do a thing"
            ),
        )


def test_every_unmeasurable_measure_call_returns_the_recorded_snapshot():
    """`measure()` for an unmeasurable entry must never attempt I/O or fabricate a
    number - calling it should be exactly as cheap and as fixed as reading
    `recorded_effect` directly."""
    for divergence in dv.REGISTRY:
        if divergence.status != "unmeasurable_now":
            continue
        assert divergence.measure() == divergence.recorded_effect


# --- pure measurement functions, synthetic inputs, hand-computed answers ----------------


def test_doy_lookup_disagrees_matches_the_known_affected_days():
    """The float32 round-trip corrupts a specific, hand-verifiable set of days: DOY 189
    (the CLAUDE.md example, 188.99998) disagrees, DOY 1 and DOY 366 (exact endpoints, no
    float32 rounding loss near them) do not."""
    assert dv.doy_lookup_disagrees(189) is True
    assert dv.doy_lookup_disagrees(1) is False
    assert dv.doy_lookup_disagrees(366) is False

    affected = [d for d in range(1, 367) if dv.doy_lookup_disagrees(d)]
    # Hand-verified against CLAUDE.md's own account: 26 days/year, including the two
    # named ranges 184-189 and 225-230.
    assert len(affected) == 26
    assert set(range(184, 190)).issubset(affected)
    assert set(range(225, 231)).issubset(affected)


def test_count_boundary_rows_counts_only_exact_matches():
    values = [9.999, 10.0, 10.0, 10.0001, 9.5, 10]
    assert dv.count_boundary_rows(values, threshold=10.0) == 3
    assert dv.count_boundary_rows([9.9, 10.1], threshold=10.0) == 0
    assert dv.count_boundary_rows([], threshold=10.0) == 0


def test_classify_storm_days_daily_hand_computed():
    dst_min = pd.Series([-60.0, -50.0, -49.9, -10.0])
    # -60 and -50 both reach the threshold (<=); -49.9 and -10 do not.
    expected = pd.Series([True, True, False, False])
    pd.testing.assert_series_equal(dv.classify_storm_days_daily(dst_min), expected)


def test_classify_storm_days_combined_hand_computed():
    # (kp_max, dst_min): storm if kp_max >= 37 OR dst_min <= -33.
    dst_min = pd.Series([-40.0, -20.0, -33.0, -10.0])
    kp_max = pd.Series([10.0, 37.0, 10.0, 36.9])
    # row0: dst -40 <= -33 -> storm. row1: kp 37 >= 37 -> storm.
    # row2: dst -33 <= -33 -> storm (boundary, inclusive). row3: neither -> quiet.
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(
        dv.classify_storm_days_combined(dst_min, kp_max), expected
    )


def test_combined_storm_rule_marks_more_days_than_daily_rule_on_synthetic_days():
    """Pins the qualitative direction of divergence #10's effect (the combined rule is
    strictly inclusive of the daily rule's marked days plus more) on a small synthetic
    calendar, independent of the real OMNI archive."""
    dst_min = pd.Series([-60.0, -20.0, -20.0, -5.0])
    kp_max = pd.Series([20.0, 40.0, 20.0, 10.0])
    daily = dv.classify_storm_days_daily(dst_min)
    combined = dv.classify_storm_days_combined(dst_min, kp_max)
    assert int(daily.sum()) == 1  # only the -60 day
    assert int(combined.sum()) == 2  # the -60 day, plus the kp=40 day
    assert (daily & ~combined).sum() == 0  # daily never marks a day combined misses


def test_interval_coverage_gaussian_hand_computed():
    """sigma=1 for every point, level=0.5 -> Gaussian half-width is exactly
    `norm.ppf(0.75)` (~0.6745). Residuals [0.5, 0.7, 1.0, 0.2]: two are within the
    half-width (0.5, 0.2), two are outside (0.7, 1.0) -> coverage 0.5."""
    mu = np.zeros(4)
    y = np.array([0.5, 0.7, 1.0, 0.2])
    sigma = np.ones(4)
    half_width = norm.ppf(0.75)
    assert half_width == pytest.approx(0.6744897501960817)
    coverage = dv.interval_coverage(y, mu, sigma, level=0.5, family="gaussian")
    assert coverage == pytest.approx(0.5)


def test_interval_coverage_laplace_hand_computed():
    """Same residuals as the Gaussian test, but the Laplace half-width at level=0.5 is
    `-scale * ln(0.5)` with `scale = sigma / sqrt(2)`: for sigma=1, half-width
    ~= 0.49015. Only residuals 0.5 and 0.2 are within/at the width; 0.5 is strictly
    greater than 0.49015 so it is excluded -> coverage 0.25 (only the 0.2 residual)."""
    mu = np.zeros(4)
    y = np.array([0.5, 0.7, 1.0, 0.2])
    sigma = np.ones(4)
    scale = 1.0 / np.sqrt(2.0)
    half_width = -scale * np.log(0.5)
    assert half_width == pytest.approx(0.4901290717724337)
    coverage = dv.interval_coverage(y, mu, sigma, level=0.5, family="laplace")
    assert coverage == pytest.approx(0.25)


def test_interval_coverage_family_gap_matches_reported_direction():
    """The same data must read higher coverage under Gaussian than under (correctly
    scaled) Laplace quantiles for a Laplace-distributed sample at nominal 50% - the
    qualitative shape of divergence #4's effect - built from an actual Laplace draw
    rather than asserted only on the two hand-picked residual sets above."""
    rng = np.random.default_rng(0)
    scale = 2.0
    sigma = np.full(
        20_000, scale * np.sqrt(2.0)
    )  # stored as std, per the module docstring
    mu = np.zeros_like(sigma)
    y = rng.laplace(loc=0.0, scale=scale, size=sigma.size)

    laplace_coverage = dv.interval_coverage(y, mu, sigma, level=0.5, family="laplace")
    gaussian_coverage = dv.interval_coverage(y, mu, sigma, level=0.5, family="gaussian")

    # True Laplace data must be closer to nominal under its native family.
    assert abs(laplace_coverage - 0.5) < abs(gaussian_coverage - 0.5)
    assert gaussian_coverage > laplace_coverage


def test_interval_coverage_rejects_unknown_family():
    with pytest.raises(ValueError, match="unknown predictive family"):
        dv.interval_coverage(
            np.zeros(1), np.zeros(1), np.ones(1), level=0.5, family="cauchy"
        )
