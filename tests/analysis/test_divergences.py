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

Deliberately not exercised unconditionally here: `Divergence.measure()` for the eight
measured entries does real I/O against the read-only legacy trees (`stec.config.paths`) -
#12 goes further and loads a real checkpoint for a live forward pass, #15 loads ~1,100
small cache files - which is what makes each a genuine, re-runnable harness rather than a
frozen constant, but that also makes it environment-dependent, so it does not belong in a
hermetic unit test. The `recorded_effect` snapshot each carries is what stays fast and
reproducible, and is what most of this suite pins.

Two `skipif`-guarded exceptions run `measure()` for real when the artifact it reads is
actually present on the host (the same "live checkout, real data" idiom used elsewhere in
`tests/`, e.g. `tests/baselines/test_gim.py`): #4's coverage numbers and #15's seed count
have both drifted from a frozen snapshot before without anything catching it, so those two
are worth checking live rather than only pinning the frozen constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from stec.analysis import divergences as dv

# --- registry structure -----------------------------------------------------------------


def test_every_divergence_has_id_description_deliverable_and_status():
    assert len(dv.REGISTRY) == 16
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
    assert {d.id for d in measured} == {
        "1",
        "4",
        "9",
        "10",
        "11",
        "12",
        "14",
        "15",
        "16",
    }
    for divergence in measured:
        effect = divergence.recorded_effect
        assert isinstance(effect, dv.MeasuredEffect)
        assert effect.method
        assert len(effect.measurements) > 0
        for m in effect.measurements:
            assert isinstance(m.new_value, (int, float))


def test_unmeasurable_divergence_carries_reason_and_no_number():
    unmeasurable = [d for d in dv.REGISTRY if d.status == "unmeasurable_now"]
    assert {d.id for d in unmeasurable} == {"2", "3", "5", "6", "7", "8", "13"}
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


# --- drift guards: frozen fallbacks must match the live artifact when it exists ---------
#
# #4's frozen VTEC-family coverage numbers (85.91/76.67) went stale once the live
# uncertainty_calibration store grew and the stage re-ran - nothing caught the drift until
# an independent audit compared the frozen constant against the CSV by hand. These tests
# make that comparison automatic, on whichever host happens to have the artifact checked
# out (`pytest.mark.skipif`, matching the "live checkout, real data" idiom already used in
# tests/baselines/test_gim.py and tests/positioning/test_metrics.py) rather than depending
# on someone remembering to re-run this check.

_VTEC_COVERAGE_CSV = (
    dv.paths.analysis_result_dir("uncertainty_calibration", rebuilt=True)
    / "finetuned_stec_own"
    / "coverage.csv"
)


@pytest.mark.skipif(
    not _VTEC_COVERAGE_CSV.exists(),
    reason="live uncertainty_calibration coverage.csv not present on this host",
)
def test_frozen_vtec_family_fallback_matches_live_coverage_csv():
    live = dv.by_id("4").measure()
    frozen = dv.by_id("4").recorded_effect
    assert isinstance(live, dv.MeasuredEffect)
    assert isinstance(frozen, dv.MeasuredEffect)
    live_values = [m.new_value for m in live.measurements]
    frozen_values = [m.new_value for m in frozen.measurements]
    assert live_values == frozen_values, (
        "the frozen _VTEC_FAMILY_EFFECT fallback has drifted from the live "
        f"coverage.csv - live={live_values}, frozen={frozen_values}. Refresh the "
        "constant in stec/analysis/divergences.py and docs/revision/divergences.md."
    )


_SUBSET_CACHE_DIR = dv.paths.SUBSET_INDEX_CACHE


@pytest.mark.skipif(
    not _SUBSET_CACHE_DIR.exists(),
    reason="live subset-index cache not present on this host",
)
def test_frozen_subset_cache_seed_fallback_matches_live_scan():
    live = dv.by_id("15").measure()
    frozen = dv.by_id("15").recorded_effect
    assert isinstance(live, dv.MeasuredEffect)
    assert isinstance(frozen, dv.MeasuredEffect)
    # Only the "carries a different (or unreadable/missing) seed" count matters here -
    # that is the number that must stay zero. The total file count is expected to grow
    # over time (a new cache is added whenever a new call site or config runs), so pinning
    # it exactly would make this test fail on drift that is not a bug.
    live_other = live.measurements[1].new_value
    assert live_other == 0, (
        f"{live_other} cached subset files no longer carry seed 42 - the seed-check fix "
        "did not hold, or the cache genuinely drifted."
    )


def test_subset_cache_seed_check_counts_a_truncated_file_instead_of_raising(
    tmp_path, monkeypatch
):
    """A truncated or non-torch .pt file raises `pickle.UnpicklingError`, not `OSError` -
    the same failure mode `stec/data/splits.py`'s `_load_cached` already guards against for
    these exact cache files, and CLAUDE.md documents truncated files as a real, observed
    failure mode on this filesystem. One corrupt file among the ~1,129 real ones must be
    counted in `other`, not crash this diagnostic."""
    import torch

    monkeypatch.setattr(dv.paths, "SUBSET_INDEX_CACHE", tmp_path)

    (tmp_path / "truncated.pt").write_bytes(b"not a real torch checkpoint")
    torch.save(
        {"len": 10, "k": 5, "seed": 42, "indices": [0, 1, 2, 3, 4]},
        tmp_path / "good_seed42.pt",
    )

    effect = dv._measure_subset_cache_seed_check()

    assert isinstance(effect, dv.MeasuredEffect)
    seed_42, other = (m.new_value for m in effect.measurements)
    assert seed_42 == 1, "the one genuinely good, seed-42 file must still be counted"
    assert other == 1, "the truncated file must land in `other`, not raise"


_COST_SUMMARY_CSV = (
    dv.paths.analysis_result_dir("computational_cost", rebuilt=True)
    / "cost_summary.csv"
)


@pytest.mark.skipif(
    not _COST_SUMMARY_CSV.exists(),
    reason="live computational_cost cost_summary.csv not present on this host",
)
def test_frozen_pretrain_compute_cost_fallback_matches_live_cost_summary_csv():
    live = dv.by_id("16").measure()
    frozen = dv.by_id("16").recorded_effect
    assert isinstance(live, dv.MeasuredEffect)
    assert isinstance(frozen, dv.MeasuredEffect)
    live_value = live.measurements[0].new_value
    frozen_value = frozen.measurements[0].new_value
    assert live_value == frozen_value, (
        "the frozen _PRETRAIN_COMPUTE_COST_EFFECT fallback has drifted from the live "
        f"cost_summary.csv - live={live_value}, frozen={frozen_value}. Refresh the "
        "constant in stec/analysis/divergences.py and docs/revision/divergences.md."
    )
    # The scaled-estimate figure must stay dead, not merely superseded - a regression that
    # silently reintroduced the old scaling would still read as "measured" without this.
    assert live_value != 0.38


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
