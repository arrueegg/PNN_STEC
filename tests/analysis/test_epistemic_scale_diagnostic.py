"""Pins the crux logic of `epistemic_scale_diagnostic`: that scaling the epistemic term
changes the *combined* ranking (not a tautology), that the bisection search actually
lands on the target coverage, that streaming through `iter_days` reproduces a direct
whole-frame computation, and that missing columns degrade gracefully rather than crash.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from stec.analysis import epistemic_scale_diagnostic as esd
from stec.inference import prediction_store as ps


def day_frame(rows: int, seed: int, missing: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0, 60, rows)
    frame = pd.DataFrame(
        {
            "true_stec": truth,
            "stec_pred": truth + rng.normal(0, 3, rows),
            "pred_epistemic_unc": rng.uniform(0.1, 2.0, rows),
            "pred_aleatoric_unc": rng.uniform(1.0, 10.0, rows),
            "satele": rng.uniform(5, 90, rows),
            "sm_lat_ipp": rng.uniform(-80, 80, rows),
        }
    )
    return frame.drop(columns=missing or [])


# --- sigma_total / coverage --------------------------------------------------------


def test_sigma_total_reduces_to_aleatoric_alone_at_s_zero():
    epistemic = np.array([1.0, 2.0, 3.0])
    aleatoric = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(esd.sigma_total(epistemic, aleatoric, 0.0), aleatoric)


def test_sigma_total_is_quadrature_sum():
    epistemic = np.array([3.0])
    aleatoric = np.array([4.0])
    # s=1: sqrt(3^2 + 4^2) = 5, the textbook 3-4-5 triangle.
    assert esd.sigma_total(epistemic, aleatoric, 1.0)[0] == pytest.approx(5.0)


def test_coverage_matches_a_hand_worked_example():
    abs_error = np.array([0.5, 1.5, 2.5, 3.5])
    sigma = np.full(4, 1.0)
    # Only the first observation falls within 1 sigma.
    assert esd.coverage(abs_error, sigma, level=1) == pytest.approx(0.25)
    # All but the last fall within 3 sigma.
    assert esd.coverage(abs_error, sigma, level=3) == pytest.approx(0.75)


# --- the crux test: scaling one component changes the combined ranking -------------


def test_scaling_epistemic_leaves_its_own_ranking_unchanged():
    """Spearman(epistemic, error) cannot depend on a constant positive scale - ranks
    are invariant to a positive multiplier. This is the "not a tautology" premise the
    diagnostic's whole design rests on: the combined sigma_total's ranking DOES move
    with s (checked below) precisely because epistemic's own ranking does not."""
    rng = np.random.default_rng(11)
    epistemic = rng.uniform(0.1, 5.0, 5_000)
    abs_error = rng.uniform(0.1, 30.0, 5_000)
    rho_at_1 = spearmanr(epistemic, abs_error).statistic
    rho_at_50 = spearmanr(50.0 * epistemic, abs_error).statistic
    assert rho_at_1 == pytest.approx(rho_at_50, abs=1e-9)


def test_combined_ranking_moves_with_s_when_components_disagree():
    """Construct epistemic and aleatoric so they rank error oppositely: aleatoric
    increasing with error, epistemic decreasing. As s grows from 0, sigma_total's
    correlation with error must move from "aleatoric's" (positive) towards
    "epistemic's" (negative) - the combined ranking is not fixed by either
    component's own ranking alone, which is the reweighting the diagnostic exploits."""
    n = 4_000
    rank = np.arange(n, dtype=float)
    abs_error = rank
    aleatoric = rank + 1.0  # increasing with error
    epistemic = (n - rank) + 1.0  # decreasing with error

    rho_small_s = esd.spearman_rank_correlation(
        esd.sigma_total(epistemic, aleatoric, 0.01), abs_error
    )
    rho_large_s = esd.spearman_rank_correlation(
        esd.sigma_total(epistemic, aleatoric, 100.0), abs_error
    )
    assert rho_small_s > 0.9  # dominated by aleatoric, which ranks with error
    assert rho_large_s < -0.9  # dominated by epistemic, which ranks against error


# --- find_calibrating_scale: bisection actually hits the target --------------------


def test_find_calibrating_scale_recovers_a_known_answer():
    """Build data where s=3 is exactly the calibrating scale for 1-sigma coverage:
    residuals are Gaussian with true std `sigma_total(epistemic, aleatoric, 3)`, so
    the empirical coverage at s=3 should sit at nominal and bisection should find it."""
    rng = np.random.default_rng(5)
    n = 300_000
    epistemic = rng.uniform(0.2, 1.0, n)
    aleatoric = rng.uniform(0.5, 2.0, n)
    true_sigma = esd.sigma_total(epistemic, aleatoric, 3.0)
    abs_error = np.abs(rng.normal(0.0, true_sigma))

    s_star = esd.find_calibrating_scale(epistemic, aleatoric, abs_error, level=1)
    assert s_star == pytest.approx(3.0, rel=0.05)

    coverage_at_s_star = esd.coverage(
        abs_error, esd.sigma_total(epistemic, aleatoric, s_star), level=1
    )
    assert coverage_at_s_star == pytest.approx(esd.NOMINAL_COVERAGE[1], abs=0.005)


def test_find_calibrating_scale_returns_low_when_already_over_covering():
    """If coverage at `low` already meets the target, the calibrating scale is `low`
    itself - inflating further would only over-cover more."""
    rng = np.random.default_rng(6)
    n = 50_000
    epistemic = rng.uniform(0.1, 1.0, n)
    # aleatoric alone already massively over-covers 1 sigma.
    aleatoric = np.full(n, 1000.0)
    abs_error = rng.uniform(0, 5.0, n)

    s_star = esd.find_calibrating_scale(
        epistemic, aleatoric, abs_error, level=1, low=0.0
    )
    assert s_star == 0.0


def test_find_calibrating_scale_raises_when_high_is_not_wide_enough():
    rng = np.random.default_rng(7)
    n = 10_000
    epistemic = rng.uniform(0.1, 1.0, n)
    aleatoric = rng.uniform(0.1, 1.0, n)
    abs_error = rng.uniform(50.0, 100.0, n)  # errors far larger than any sigma below

    with pytest.raises(ValueError, match="widen `high`"):
        esd.find_calibrating_scale(epistemic, aleatoric, abs_error, level=1, high=1.0)


# --- streaming (collect_arrays / sweep_scale) matches direct computation -----------


def test_collect_arrays_streaming_matches_whole_frame(tmp_path):
    days = [(2024, 140, 500, 11), (2024, 141, 700, 12), (2024, 142, 300, 13)]
    frames = {}
    for year, doy, rows, seed in days:
        frame = day_frame(rows, seed)
        frames[doy] = frame
        ps.write_predictions(frame, "pretrained_stec", "own", year, doy, root=tmp_path)

    arrays = esd.collect_arrays("pretrained_stec", "own", tmp_path)
    whole = pd.concat(frames.values(), ignore_index=True)

    assert arrays["abs_error"].size == len(whole)
    direct_abs_error = (
        (whole["stec_pred"] - whole["true_stec"]).abs().to_numpy(np.float32)
    )
    # Row order across concatenated days is preserved by day_paths' sorted iteration
    # (140, 141, 142, matching insertion order into `frames`), so a direct positional
    # comparison is valid, not just matching distributions. atol covers residuals near
    # zero, where a relative tolerance alone is too tight for the float32 round trip.
    np.testing.assert_allclose(
        arrays["abs_error"], direct_abs_error, rtol=1e-5, atol=1e-4
    )

    direct_sweep_rho = spearmanr(
        esd.sigma_total(
            whole["pred_epistemic_unc"].to_numpy(np.float32),
            whole["pred_aleatoric_unc"].to_numpy(np.float32),
            5.0,
        ),
        direct_abs_error,
    ).statistic
    streamed_sweep = esd.sweep_scale(arrays, scales=np.array([5.0]))
    assert streamed_sweep.loc[0, "spearman_rho"] == pytest.approx(
        direct_sweep_rho, rel=1e-3
    )


def test_missing_geomag_column_degrades_gracefully(tmp_path):
    """A store variant without the geomagnetic-latitude column (`satele` is one of
    the store's REQUIRED_COLUMNS, so it can never actually be absent) must still
    produce the core sweep - that column only feeds the geomag-latitude stratified
    view, not coverage or Spearman."""
    frame = day_frame(2_000, seed=9, missing=["sm_lat_ipp"])
    ps.write_predictions(frame, "pretrained_stec", "own", 2024, 150, root=tmp_path)

    arrays = esd.collect_arrays("pretrained_stec", "own", tmp_path)
    assert arrays["abs_error"].size == 2_000
    assert not np.isnan(arrays["elevation"]).any()
    assert np.isnan(arrays["geomag_lat"]).all()

    sweep = esd.sweep_scale(arrays, scales=np.array([1.0, 10.0]))
    assert len(sweep) == 2
    assert sweep["coverage_1sigma"].between(0, 1).all()


def test_collect_arrays_raises_on_empty_store(tmp_path):
    with pytest.raises(FileNotFoundError):
        esd.collect_arrays("pretrained_stec", "own", tmp_path)


# --- stratified_calibrating_scale ---------------------------------------------------


def test_stratified_calibrating_scale_recovers_different_s_per_bin():
    """Two elevation bins with deliberately different calibrating scales (3 and 10)
    must be recovered separately, not averaged into one global value - this is what
    the diagnostic uses to decide whether a single scalar suffices."""
    rng = np.random.default_rng(8)
    n_per_bin = 200_000

    low_elevation = rng.uniform(5, 15, n_per_bin)
    epistemic_low = rng.uniform(0.2, 1.0, n_per_bin)
    aleatoric_low = rng.uniform(0.5, 2.0, n_per_bin)
    sigma_low = esd.sigma_total(epistemic_low, aleatoric_low, 3.0)
    error_low = np.abs(rng.normal(0.0, sigma_low))

    high_elevation = rng.uniform(75, 85, n_per_bin)
    epistemic_high = rng.uniform(0.2, 1.0, n_per_bin)
    aleatoric_high = rng.uniform(0.5, 2.0, n_per_bin)
    sigma_high = esd.sigma_total(epistemic_high, aleatoric_high, 10.0)
    error_high = np.abs(rng.normal(0.0, sigma_high))

    arrays = {
        "epistemic": np.concatenate([epistemic_low, epistemic_high]),
        "aleatoric": np.concatenate([aleatoric_low, aleatoric_high]),
        "abs_error": np.concatenate([error_low, error_high]),
        "elevation": np.concatenate([low_elevation, high_elevation]),
    }

    table = esd.stratified_calibrating_scale(
        arrays, "elevation", bin_edges=[0, 20, 90], min_observations=1_000
    )
    # Sort by calibrating_scale rather than by bin label: the low-elevation bin must
    # land near 3, the high-elevation bin near 10.
    scales_sorted = table.sort_values("calibrating_scale")[
        "calibrating_scale"
    ].to_numpy()
    assert scales_sorted[0] == pytest.approx(3.0, rel=0.1)
    assert scales_sorted[1] == pytest.approx(10.0, rel=0.1)


def test_stratified_calibrating_scale_drops_bins_below_min_observations():
    rng = np.random.default_rng(10)
    arrays = {
        "epistemic": rng.uniform(0.1, 1.0, 100),
        "aleatoric": rng.uniform(0.5, 2.0, 100),
        "abs_error": rng.uniform(0.1, 5.0, 100),
        "elevation": rng.uniform(5, 15, 100),  # a single, small bin
    }
    table = esd.stratified_calibrating_scale(
        arrays, "elevation", bin_edges=[0, 20, 90], min_observations=5_000
    )
    assert table.empty


def test_stratified_calibrating_scale_year_is_categorical_not_binned():
    rng = np.random.default_rng(12)
    n = 20_000
    arrays = {
        "epistemic": rng.uniform(0.1, 1.0, n),
        "aleatoric": rng.uniform(0.5, 2.0, n),
        "abs_error": rng.uniform(0.1, 5.0, n),
        "year": np.concatenate([np.full(n // 2, 2020), np.full(n - n // 2, 2021)]),
    }
    table = esd.stratified_calibrating_scale(
        arrays, "year", bin_edges=None, min_observations=1_000
    )
    assert set(table["bin"]) == {"2020", "2021"}


# --- matched_day_pairs: the paper and reference partitions are independent backfills,
# not guaranteed to cover the same days - this is what checks that before either is
# streamed and scored -----------------------------------------------------------------


def test_matched_day_pairs_returns_full_set_with_no_warning_when_matched(
    tmp_path, caplog
):
    ps.write_predictions(
        day_frame(10, seed=1), "pretrained_stec", "own", 2024, 130, root=tmp_path
    )
    ps.write_predictions(
        day_frame(10, seed=2),
        "pretrained_stec_resnet_bnn_nll",
        "own",
        2024,
        130,
        root=tmp_path,
    )

    with caplog.at_level("WARNING", logger="stec.analysis.epistemic_scale_diagnostic"):
        common = esd.matched_day_pairs(
            "pretrained_stec", "pretrained_stec_resnet_bnn_nll", "own", tmp_path
        )

    assert common == {(2024, 130)}
    # write_predictions warns separately about this fixture's missing uncertainty
    # column (day_frame carries no pred_total_unc) - filter to this module's own
    # records so that unrelated warning does not fail this assertion.
    own_records = [
        r
        for r in caplog.records
        if r.name == "stec.analysis.epistemic_scale_diagnostic"
    ]
    assert not own_records


def test_matched_day_pairs_restricts_to_intersection_and_warns_when_mismatched(
    tmp_path, caplog
):
    """The concrete regression this check exists to catch: the paper model's partition
    has a day (2024, 131) the reference partition does not - a real possibility since
    the two are independent backfills (CLAUDE.md's store-partition gotcha). The
    mismatch must not pass silently."""
    ps.write_predictions(
        day_frame(10, seed=1), "pretrained_stec", "own", 2024, 130, root=tmp_path
    )
    ps.write_predictions(
        day_frame(10, seed=2), "pretrained_stec", "own", 2024, 131, root=tmp_path
    )
    ps.write_predictions(
        day_frame(10, seed=3),
        "pretrained_stec_resnet_bnn_nll",
        "own",
        2024,
        130,
        root=tmp_path,
    )

    with caplog.at_level("WARNING", logger="stec.analysis.epistemic_scale_diagnostic"):
        common = esd.matched_day_pairs(
            "pretrained_stec", "pretrained_stec_resnet_bnn_nll", "own", tmp_path
        )

    assert common == {(2024, 130)}
    own_records = [
        r
        for r in caplog.records
        if r.name == "stec.analysis.epistemic_scale_diagnostic"
    ]
    assert len(own_records) == 1
    assert "1 day(s) only in pretrained_stec" in own_records[0].message


def test_collect_arrays_day_pairs_restricts_to_the_given_days(tmp_path):
    ps.write_predictions(
        day_frame(500, seed=20), "pretrained_stec", "own", 2024, 140, root=tmp_path
    )
    ps.write_predictions(
        day_frame(700, seed=21), "pretrained_stec", "own", 2024, 141, root=tmp_path
    )

    arrays = esd.collect_arrays(
        "pretrained_stec", "own", tmp_path, day_pairs={(2024, 140)}
    )
    assert arrays["abs_error"].size == 500


def test_main_passes_the_same_common_day_pairs_to_both_collect_arrays_calls(
    tmp_path, monkeypatch
):
    """End-to-end: both `collect_arrays` calls `main()` makes must be restricted to the
    same shared day set, not just that `matched_day_pairs` computes it correctly in
    isolation. (2024, 151) exists only in the paper model's partition and must be
    excluded from both."""
    ps.write_predictions(
        day_frame(400, seed=30), "pretrained_stec", "own", 2024, 150, root=tmp_path
    )
    ps.write_predictions(
        day_frame(300, seed=31), "pretrained_stec", "own", 2024, 151, root=tmp_path
    )
    ps.write_predictions(
        day_frame(400, seed=32),
        "pretrained_stec_resnet_bnn_nll",
        "own",
        2024,
        150,
        root=tmp_path,
    )
    output_dir = tmp_path / "output"

    seen_day_pairs = []
    real_collect_arrays = esd.collect_arrays

    def spy(*args, **kwargs):
        seen_day_pairs.append(kwargs.get("day_pairs"))
        return real_collect_arrays(*args, **kwargs)

    monkeypatch.setattr(esd, "collect_arrays", spy)
    monkeypatch.setattr(
        "sys.argv",
        [
            "epistemic_scale_diagnostic",
            "--store-root",
            str(tmp_path),
            "--model-variant",
            "pretrained_stec",
            "--reference-model-variant",
            "pretrained_stec_resnet_bnn_nll",
            "--dataset",
            "own",
            "--output-dir",
            str(output_dir),
        ],
    )
    esd.main()

    assert len(seen_day_pairs) == 2
    assert seen_day_pairs[0] == seen_day_pairs[1] == {(2024, 150)}
