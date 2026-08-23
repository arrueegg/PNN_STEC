"""Spherical-harmonic feature encoder.

Ported from `src/utils/locationencoder/pe/spherical_harmonics.py` (part of a small
"locationencoder" package pulled in from elsewhere), which is the single piece of that
package `stec/` actually needs: `FeatureAssembler` (`stec/data/transforms.py`) takes its
encoder by injection rather than constructing one, and every call site - the paper's model
and its VTEC baseline alike - passes a `SphericalHarmonics` instance. The sibling encoders
in the original package (`Theory`, `GridAndSphere`, `Direct`, `Cartesian3D`, `Wrap`, and the
`_cal_freq_list` helper they share) implement position-encoding schemes from other papers
that nothing in this codebase ever calls; they were not ported.

The original also accepted a `harmonics_calculation="closed-form"` mode, but no branch ever
implemented it - `self.SH` was only ever assigned for `"analytic"` - and every call site in
this codebase uses the default, so that dead branch is dropped here rather than carried
forward.
"""

from __future__ import annotations

import torch
from torch import nn

from .spherical_harmonics_ylm import SH


class SphericalHarmonics(nn.Module):
    """Real spherical-harmonic expansion of a (longitude, latitude) pair, up to a degree.

    `legendre_polys` sets both the degree and order bound `L = M`; the module emits every
    Y_l^m for l in [0, L) and m in [-l, l], for `L**2` output columns. The two conventions
    the paper's models use - `degree**2` for the STEC models, `(degree+1)**2` for the Mao et
    al. VTEC baseline - both come from picking what `legendre_polys` is set to
    (`stec.data.feature_layout.SHConvention` does that); this class only ever sees the
    already-resolved value.
    """

    def __init__(self, legendre_polys: int) -> None:
        super().__init__()
        self.L = self.M = int(legendre_polys)
        self.embedding_dim = self.L * self.M

    def forward(self, lonlat: torch.Tensor) -> torch.Tensor:
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # The analytic Yl_m formulas are colatitude/longitude functions of a unit sphere,
        # so degrees and the geographic (lon in [-180, 180], lat in [-90, 90]) convention
        # have to be shifted into the [0, pi] / [0, 2*pi] ranges they were derived in.
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        columns = []
        for degree in range(self.L):
            for m in range(-degree, degree + 1):
                y = SH(m, degree, phi, theta)
                if isinstance(y, float):
                    # Yl0_m0 is a constant, so it comes back as a bare Python float rather
                    # than a tensor; broadcast it so every column has the batch shape.
                    y = y * torch.ones_like(phi)
                columns.append(y)

        return torch.stack(columns, dim=-1)
