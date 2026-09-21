"""VTEC + Mapping baseline: turn a predicted vertical TEC into a slant TEC.

Ported from `apply_mapping_function` in `src/compare_stec_vtec_gim.py`. The VTEC
model itself (Mao et al. 2025's MLP, trained with a Laplacian NLL, config
`MLP_LaplacianNLL`) lives outside this module and outside this port; this module
only carries out the thin-shell mapping step the source function performs once
the model has already produced a VTEC (and optionally a predicted spread).

Thin-shell mapping function
----------------------------
Reused, not re-implemented: `MappingFunction` in `stec/baselines/gim.py` already
carries the SLM and MSLM formulas verbatim from `src/evaluation/gim_mapper.py`
(itself unchanged there from `src/compare_stec_vtec_gim.py`'s import of the same
class). Both baselines share identical thin-shell geometry - the GIM baseline
additionally reads IONEX grids, this one does not - so this module imports the
one definition rather than keeping a second copy of the constants that could
drift from it. For the record, quoting the source (`src/evaluation/gim_mapper.py`):

    def SLM_MF(self, elevation):
        H = 450.0  # Height of the ionospheric shell in km
        mapping_function = np.cos(np.arcsin(self.RE / (self.RE + H) * np.sin(np.pi/2 - elevation)))
        return 1.0 / mapping_function

    def MSLM_MF(self, elevation):
        H = 506.7  # Height of the ionospheric shell in km
        alpha = 0.9782
        mapping_function = np.cos(np.arcsin(self.RE / (self.RE + H) * np.sin(alpha * (np.pi/2 - elevation))))
        return 1.0 / mapping_function

with `self.RE = 6371.0`. No constant here has been changed.

`MappingFunction.__init__` defaults to `mapping_type="SLM"`, but the source's
own production callers do not use that default: `compare_stec_vtec_gim.py`'s
CLI flag `--mapping_function` defaults to `"MSLM"` (`choices=["SLM", "MSLM"]`,
`help="... (default: MSLM)"`) and that is what `apply_mapping_function` is
actually invoked with when the paper's "VTEC + Mapping" numbers were produced.
SLM and MSLM are not interchangeable - they disagree by multiple TECU at low
elevation (see the tests). `map_vtec_to_stec` below therefore takes
`mapping_type` as a required keyword, with no default of its own, so a caller
cannot silently inherit `MappingFunction`'s "SLM" default by omission.

Uncertainty propagation
------------------------
`apply_mapping_function` (source, `src/compare_stec_vtec_gim.py` around lines
264-280) propagates the VTEC model's predicted spread like this:

    vtec_var = vtec_val**2          # "if uncertainty is std, square it first"
    stec_var_mapped = vtec_var * mapping_factors**2
    stec_unc = np.sqrt(stec_var_mapped)

For a single term this square/multiply/root sequence is algebraically just
`stec_unc = vtec_val * mapping_factors`, because squaring and then taking the
(non-negative) square root of a positive number is the identity - the source
only needs the variance form when it is *summing* several variance components
(aleatoric + epistemic) before mapping, not for the mapping step itself. So:
STEC = VTEC * mapping_factor is a linear map, and the spread of a linear
transform of any location-scale distribution scales linearly with the
coefficient (variance scales with its square) - this holds regardless of
whether the underlying distribution is Gaussian or Laplace. Verified directly
against the source above: this module reproduces `stec_unc = spread *
mapping_factor` rather than the square/root round trip, since they are the
same computation and the direct form is not misreadable as "squares, therefore
must be variance-only".

The Laplace point
------------------
The VTEC model is trained with a Laplacian NLL. Its raw network output is a
scale parameter `b`; `inference_manager` (`src/training/inference_manager.py`)
turns that into the reported `pred_total_unc` as the predictive's true
standard deviation, `std = sqrt(2) * b` (a Laplace's variance is `2 *
scale**2`), and that is the convention `stec/analysis/uncertainty_calibration.py`
already documents and relies on for the column this module's caller writes,
`vtec_model_stec_total_unc`: "vtec_model_stec_total_unc is stored as that
Laplace's standard deviation ... Recovering the Laplace scale therefore needs
scale = std / sqrt(2) before it goes into any Laplace formula". Propagating
linearly by the mapping factor (as derived above) preserves that: the mapped
value is still a standard deviation, with `variance = std**2 = 2 * scale**2`
and `scale = std / sqrt(2)`.

Scoring this spread as if it were a Gaussian sigma is a real, measured error,
not a rounding difference: the same stored numbers read 90% empirical coverage
at nominal 50% under Gaussian quantiles, against 82% under (correctly-scaled)
Laplace quantiles. To make that mistake structurally harder, the mapped spread
is returned here as a `LaplaceStd`, not a bare array: getting a variance or the
native scale out of it requires calling `.variance()` or `.scale()`
explicitly, so a caller cannot hand the number to a Gaussian formula without a
visible, named step declaring which family it is assuming.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gim import MappingFunction

__all__ = ["LaplaceStd", "MappedVtec", "map_vtec_to_stec"]


@dataclass(frozen=True)
class LaplaceStd:
    """A Laplace distribution's spread, held as its standard deviation.

    This exists so the scale-vs-std distinction cannot be skipped silently.
    There is no `.std`-shaped accessor and no bare-array return path: the only
    ways to read a number out of this are `.variance()` and `.scale()`, and
    both name the distributional assumption a caller is making. Plugging the
    raw standard deviation into a Laplace PDF/CDF/quantile formula as if it
    were the scale parameter inflates the implied spread by `sqrt(2)`; scoring
    it as a Gaussian sigma instead assumes the wrong distribution shape
    entirely. Both are the same failure mode this type exists to block.
    """

    std: np.ndarray

    def variance(self) -> np.ndarray:
        """Variance of the distribution: `std**2` (equivalently `2 * scale()**2`)."""
        return self.std**2

    def scale(self) -> np.ndarray:
        """The Laplace distribution's native scale parameter, `b = std / sqrt(2)`.

        Required by any Laplace PDF/CDF/quantile formula. Using `std` in place
        of this is a second, easy-to-miss error on top of picking the wrong
        distribution family in the first place.
        """
        return self.std / np.sqrt(2.0)


@dataclass(frozen=True)
class MappedVtec:
    """A VTEC prediction, and optionally its spread, mapped to the slant direction."""

    stec: np.ndarray
    mapping_factor: np.ndarray
    stec_std: LaplaceStd | None = None


def map_vtec_to_stec(
    vtec: np.ndarray,
    elevation_deg: np.ndarray,
    *,
    mapping_type: str,
    vtec_std: np.ndarray | None = None,
) -> MappedVtec:
    """Map a predicted VTEC (and optionally its predicted std) to slant STEC.

    Port of `apply_mapping_function` in `src/compare_stec_vtec_gim.py`,
    restricted to the single-model case that function's `pred_stec` branch
    covers (its `pred_mean`/`pred_var` branch serves a different ensemble
    code path not needed here). Call this once per spread column a caller has
    (aleatoric/epistemic/total): the propagation is identical for each, since
    it is the same linear scaling regardless of which variance component is
    being mapped.

    Args:
        vtec: Predicted vertical TEC (TECU).
        elevation_deg: Satellite elevation angle, in degrees - matches
            `vtec_df["satele"]` in the source, which the source converts with
            `np.radians` before calling `get_mapping_factor`.
        mapping_type: `"SLM"` or `"MSLM"`, passed straight to
            `MappingFunction`. Required, with no default: see the module
            docstring - `MappingFunction`'s own default (`"SLM"`) is not what
            produced the paper's "VTEC + Mapping" numbers (`"MSLM"`).
        vtec_std: Predicted standard deviation of the VTEC model's Laplace
            predictive (TECU), if the caller has one to propagate. `None`
            skips uncertainty propagation and leaves `MappedVtec.stec_std`
            as `None`.

    Returns:
        `MappedVtec` holding the slant STEC, the per-observation mapping
        factor that was applied, and - when `vtec_std` was given - the mapped
        spread as a `LaplaceStd`.
    """
    elevation_rad = np.radians(np.asarray(elevation_deg, dtype=np.float64))
    mapping_factor = MappingFunction(mapping_type=mapping_type).get_mapping_factor(
        elevation_rad
    )

    stec = np.asarray(vtec, dtype=np.float64) * mapping_factor

    stec_std = None
    if vtec_std is not None:
        # Linear in the mapping factor - see "Uncertainty propagation" above.
        stec_std = LaplaceStd(np.asarray(vtec_std, dtype=np.float64) * mapping_factor)

    return MappedVtec(stec=stec, mapping_factor=mapping_factor, stec_std=stec_std)
