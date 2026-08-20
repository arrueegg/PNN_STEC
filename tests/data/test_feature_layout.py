"""One computation of the input dimension, and named blocks instead of offsets.

The defect these replace does not raise when it happens: a disagreement between the two
independent dimension computations shifts every later feature by a constant, and the model
trains to something plausible and wrong.

The anchor is `test_paper_model_layout_matches_the_published_checkpoint`: the layout has to
reproduce the 127 input columns of the real pretrained checkpoint from the real config. A
formula that is self-consistent but disagrees with the trained model is worth nothing.
"""

from __future__ import annotations

import pytest

from stec.data.feature_layout import (
    FeatureGroup,
    SHConvention,
    convention_for,
    layout_from_feature_control,
)

# feature_control exactly as stored in the paper model's config.yaml: everything enabled.
PAPER_FEATURE_CONTROL = {
    "year": True,
    "doy": True,
    "sod": True,
    "local_time_hours": True,
    "lat_sta": True,
    "lon_sta": True,
    "sm_lat_sta": True,
    "sm_lon_sta": True,
    "satazi": True,
    "satele": True,
    "lat_ipp": True,
    "lon_ipp": True,
    "sm_lat_ipp": True,
    "sm_lon_ipp": True,
    "Kp_index": True,
    "R_Sunspot_No": True,
    "Dst-index,_nT": True,
    "AE-index,_nT": True,
    "ap_index,_nT": True,
    "f107_index": True,
}


def paper_layout():
    return layout_from_feature_control(PAPER_FEATURE_CONTROL, sh_degree=5)


def test_paper_model_layout_matches_the_published_checkpoint():
    """127 is the input width of pretrain_BayesianResNetSTEC_seed42.pth."""
    assert paper_layout().total_dim == 127


def test_the_127_decomposes_the_way_the_collation_emits_it():
    layout = paper_layout()
    widths = {
        group: layout.group_slice(group).stop - layout.group_slice(group).start
        for group in FeatureGroup
    }
    # year 1 + three cyclical x 3
    assert widths[FeatureGroup.TEMPORAL] == 10
    assert widths[FeatureGroup.STATION] == 4
    # azimuth and elevation become one unit vector
    assert widths[FeatureGroup.DIRECTION] == 3
    assert widths[FeatureGroup.IPP] == 4
    assert widths[FeatureGroup.SWI] == 6
    assert layout.sh_width == 100


def test_cyclical_temporal_features_emit_sin_cos_and_norm():
    layout = paper_layout()
    assert layout.block("sod").columns == ("sod_sin", "sod_cos", "sod_norm")
    assert layout.block("year").columns == ("year_norm",)


def test_direction_becomes_a_unit_vector_when_both_angles_are_present():
    layout = paper_layout()
    assert layout.block("direction").columns == ("e_up", "e_east", "e_north")


def test_a_lone_direction_angle_falls_back_to_one_column():
    control = {**PAPER_FEATURE_CONTROL, "satazi": False}
    layout = layout_from_feature_control(control, sh_degree=5)
    assert layout.block("satele").width == 1
    assert layout.total_dim == 127 - 3 + 1


def test_stec_convention_is_degree_squared():
    assert SHConvention.SQUARED.terms(5) == 25


def test_vtec_convention_is_degree_plus_one_squared():
    assert SHConvention.PLUS_ONE_SQUARED.terms(15) == 256


def test_degree_zero_contributes_nothing():
    assert SHConvention.SQUARED.terms(0) == 0
    assert SHConvention.PLUS_ONE_SQUARED.terms(0) == 0


def test_convention_follows_the_run_not_the_model_name():
    assert convention_for("vtec", "gaussian") is SHConvention.PLUS_ONE_SQUARED
    assert convention_for("stec", "laplace") is SHConvention.PLUS_ONE_SQUARED
    assert convention_for("stec", "gaussian") is SHConvention.SQUARED


def test_all_four_coordinate_pairs_expand_in_the_collation_order():
    """Grouped by coordinate system, not by location: both geographic, then both magnetic.

    This is the order the collation emits (sh_sta_geo, sh_ipp_geo, sh_sta_sm, sh_ipp_sm)
    and therefore the order every trained checkpoint expects. Getting it wrong produces a
    tensor of the correct width holding permuted columns, which trains a plausible and
    wrong model rather than failing - it is what the legacy-comparison test caught.
    """
    assert paper_layout().sh_locations == (
        "station_geographic",
        "ipp_geographic",
        "station_magnetic",
        "ipp_magnetic",
    )


def test_a_half_disabled_pair_does_not_expand():
    """An expansion needs both members; one alone is not a location."""
    control = {**PAPER_FEATURE_CONTROL, "lon_ipp": False}
    layout = layout_from_feature_control(control, sh_degree=5)
    assert "ipp_geographic" not in layout.sh_locations
    # One scalar column gone, and the whole 25-term expansion with it.
    assert layout.total_dim == 127 - 1 - 25


def test_disabling_a_cyclical_feature_removes_three_columns():
    """The failure the Branch models' hardcoded splits cannot survive."""
    control = {**PAPER_FEATURE_CONTROL, "local_time_hours": False}
    layout = layout_from_feature_control(control, sh_degree=5)
    assert layout.total_dim == 127 - 3


def test_blocks_tile_the_tensor_without_gap_or_overlap():
    blocks = paper_layout().blocks()
    assert blocks[0].start == 0
    for earlier, later in zip(blocks[:-1], blocks[1:], strict=True):
        assert earlier.stop == later.start
    assert blocks[-1].stop == 127


def test_group_slice_replaces_a_hardcoded_split():
    """Branch models hardcode temporal_split = 10; this derives it."""
    layout = paper_layout()
    assert layout.group_slice(FeatureGroup.TEMPORAL) == slice(0, 10)
    assert (
        layout.group_slice(FeatureGroup.SWI).stop
        - layout.group_slice(FeatureGroup.SWI).start
        == 6
    )


def test_the_collation_order_is_not_the_order_the_branch_comment_claims():
    """The comment says Temporal + Station + IPP + Direction; direction comes third."""
    layout = paper_layout()
    assert (
        layout.group_slice(FeatureGroup.DIRECTION).start
        < layout.group_slice(FeatureGroup.IPP).start
    )


def test_unknown_block_is_an_error_not_a_silent_zero():
    with pytest.raises(KeyError):
        paper_layout().block("sh_moon_geographic")


def test_empty_group_lookup_is_an_error():
    control = {k: v for k, v in PAPER_FEATURE_CONTROL.items()}
    for name in (
        "Kp_index",
        "R_Sunspot_No",
        "Dst-index,_nT",
        "AE-index,_nT",
        "ap_index,_nT",
        "f107_index",
    ):
        control[name] = False
    layout = layout_from_feature_control(control, sh_degree=5)
    with pytest.raises(KeyError):
        layout.group_slice(FeatureGroup.SWI)


def test_layout_is_immutable():
    """A shared mutable registry made a consumer's view depend on construction order."""
    with pytest.raises(AttributeError):
        paper_layout().sh_degree = 15


def test_vtec_baseline_layout_matches_its_published_checkpoint():
    """261 is the real input width of the Mao et al. replication's layers.0.weight.

    Its stored feature_control enables only sod, sm_lat_ipp and sm_lon_ipp, at SH degree 15
    under the (degree + 1)**2 convention: 3 + 2 + 256. This is the only model family that
    exercises the second convention, so it is what keeps that branch honest - 487 of the
    trained checkpoints reproduce it.
    """
    control = {"sod": True, "sm_lat_ipp": True, "sm_lon_ipp": True}
    layout = layout_from_feature_control(
        control, sh_degree=15, target="vtec", distribution="laplace"
    )
    assert layout.sh_terms_per_location == 256
    assert layout.sh_locations == ("ipp_magnetic",)
    assert layout.total_dim == 261
