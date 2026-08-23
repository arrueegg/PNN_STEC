"""`SphericalHarmonics`: shape/degree contract, and numerical equivalence with the
pre-rebuild `src/utils/locationencoder` implementation it was ported from.

The equivalence test is the second half of the self-containment fix: `stec/` used to have no
native spherical-harmonic encoder at all, so `stec.data.transforms.FeatureAssembler` could
only be exercised in tests by reaching into `src/` (see `test_transforms.sh_encoder_for` and
`test_clean_clone.py`, both now pointed at this module instead). Because this class computes
100 of the paper model's 127 input columns, "ported" has to mean "identical to float
round-off", not merely "structurally similar" - a systematic drift here would silently
retrain a different model under the published hyperparameters.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from stec.data.spherical_harmonics import SphericalHarmonics

LEGACY_SRC = "/scratch2/arrueegg/WP4/PNN_STEC/src"


def legacy_available() -> bool:
    """The pre-rebuild tree, which a clean clone will not have."""
    if LEGACY_SRC not in sys.path:
        sys.path.insert(0, LEGACY_SRC)
    return importlib.util.find_spec("utils.locationencoder.pe") is not None


def random_lonlat(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    lon = torch.rand(rows, generator=generator) * 360 - 180
    lat = torch.rand(rows, generator=generator) * 180 - 90
    return torch.stack([lon, lat], dim=1)


# --- shape / degree contract ------------------------------------------------------------


@pytest.mark.parametrize(
    "legendre_polys,expected_width",
    [
        (5, 25),  # STEC models: legendre_polys == sh_degree, so degree**2
        (16, 256),  # VTEC baseline: legendre_polys == sh_degree + 1 == 16
    ],
)
def test_output_width_is_legendre_polys_squared(legendre_polys, expected_width):
    encoder = SphericalHarmonics(legendre_polys=legendre_polys)
    assert encoder.embedding_dim == expected_width
    out = encoder(random_lonlat(rows=8, seed=0))
    assert out.shape == (8, expected_width)


def test_degree_zero_term_is_the_constant_1_over_2_sqrt_pi():
    # Yl0_m0 is a bare Python float in the analytic table (it does not depend on phi/theta),
    # so this also proves the constant-broadcast branch in forward() produces a real batch
    # column rather than leaking a 0-dim tensor into torch.stack.
    encoder = SphericalHarmonics(legendre_polys=1)
    out = encoder(random_lonlat(rows=5, seed=1))
    assert out.shape == (5, 1)
    assert torch.allclose(out, torch.full((5, 1), 0.282094791773878), atol=1e-12)


def test_no_nan_at_poles_or_the_dateline():
    # cos(theta) == +-1 exactly at the poles, so any (1 - cos(theta)**2)**0.5 term is
    # differentiating a fractional power at its singular point - the case most likely to
    # produce a NaN from floating-point overshoot past 1.0.
    edge_cases = torch.tensor(
        [
            [-180.0, -90.0],
            [180.0, 90.0],
            [179.999, 45.0],
            [-179.999, -45.0],
            [0.0, 0.0],
        ]
    )
    out = SphericalHarmonics(legendre_polys=16)(edge_cases)
    assert not torch.isnan(out).any()


# --- equivalence with the pre-rebuild implementation -------------------------------------


@pytest.mark.skipif(
    not legacy_available(), reason="pre-rebuild source tree not available"
)
@pytest.mark.parametrize("legendre_polys", [5, 16])
def test_matches_legacy_locationencoder_exactly(legendre_polys):
    """Same generated Yl_m table, so the two must agree to float round-off - in practice,
    exactly, since both call the identical torch.jit.script-compiled formulas."""
    from utils.locationencoder.pe import SphericalHarmonics as LegacySphericalHarmonics  # noqa: PLC0415

    lonlat = random_lonlat(rows=2000, seed=2)
    ported = SphericalHarmonics(legendre_polys=legendre_polys)(lonlat)
    legacy = LegacySphericalHarmonics(legendre_polys=legendre_polys)(lonlat)

    max_abs_diff = float((ported - legacy).abs().max())
    assert max_abs_diff == 0.0, (
        f"max abs diff {max_abs_diff:.3e} at legendre_polys={legendre_polys}"
    )
