#!/usr/bin/env python
# coding: utf-8
"""
Shared machinery for the reconstruction sensitivity analyses.

Usage
-----
    from sensitivity_common import (
        PRECIP_DATASETS,
        SST_DATASETS,
        INPUTS_DIR,
        RESULTS_DIR,
        TREND_PERIODS,
        get_all_pairs,
        parse_pair_args,
        drop_scalar_coords,
        load_and_prepare_pair,
        fit_regression,
        area_weights,
        reconstruct,
        diagnostics,
        convert_time_to_years,
        linear_trend,
        trend_over,
        CI_Z,
        AGREEMENT_THRESHOLD,
        ensemble_trend_significance,
        sign_agreement,
        three_categories,
    )

The loading, alignment, detrending, regression and area-weighting steps here
reproduce 11_Linear_Regression_Bootstrap_SE.py exactly, so a sensitivity run at
the baseline settings is bit-identical to the pipeline.

One ordering matters throughout: `slope` is the sensitivity and is never
area-weighted. Cell area is a property of the spatial integral, so it is applied
inside `reconstruct` and nowhere else. Anything that presents a sensitivity --
a map, a pattern correlation, a standard error -- uses the raw slope.

What is deliberately NOT here: the bootstrap. `slope` and `pval` do not depend on
the significance threshold, the FDR family, the area convention or the grid, so
every sensitivity sweep fits the regression once and varies only what comes
after. That is what makes these analyses minutes rather than hours.
"""

# Defers annotation evaluation, so PEP 585 generics (list[tuple[str, str]]) and
# PEP 604 unions (X | None) parse on Python 3.7+ rather than only 3.9+/3.10+.
# Keeps these modules runnable under whichever interpreter picks them up.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import detrend_dim, grid_area, regression_slope_se

# ---------------------------------------------------------------------------
# Paths and datasets — must match 11_Linear_Regression_Bootstrap_SE.py
# ---------------------------------------------------------------------------
INPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
RESULTS_DIR = DATA_DIR

PRECIP_DATASETS = [
    "GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN",
]
SST_DATASETS = ["ERSSTv6", "COBE-SST3"]


# ---------------------------------------------------------------------------
# Trend windows
# ---------------------------------------------------------------------------
# The first period is the one the figures report; the rest are carried so the
# trend window can be changed without re-running the bootstrap, which is the
# only place the per-replicate trend distribution can be computed.
TREND_PERIODS = [
    ("1979-01-01", "2014-12-31"),
    ("1980-01-01", "2019-12-31"),
]


# ---------------------------------------------------------------------------
# Pair enumeration and CLI
# ---------------------------------------------------------------------------

def get_all_pairs() -> list[tuple[str, str]]:
    """
    Ordered (precip, SST) pairs.

    Must stay in the same order as scripts 11 and 17 so that --pair-index N
    means the same pair everywhere and outputs can be matched up.
    """
    return [(p, s) for p in PRECIP_DATASETS for s in SST_DATASETS]


def parse_pair_args(description: str) -> tuple[str, str, int]:
    """
    Standard --pair-index / --list-pairs handling shared by the sensitivity
    scripts. Exits on --list-pairs or on an out-of-range index.

    Returns
    -------
    (precip_name, sst_name, pair_index)
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Index of the (precip, SST) pair to process (0-based).")
    parser.add_argument("--list-pairs", action="store_true",
                        help="Print the pair index table and exit.")
    args = parser.parse_args()

    pairs = get_all_pairs()

    if args.list_pairs:
        print(f"\n{'Index':<8}{'Precip':<16}{'SST':<16}")
        print("-" * 40)
        for i, (p, s) in enumerate(pairs):
            print(f"{i:<8}{p:<16}{s:<16}")
        sys.exit(0)

    if args.pair_index is None:
        parser.error("--pair-index is required unless using --list-pairs")
    if not 0 <= args.pair_index < len(pairs):
        print(f"ERROR: --pair-index {args.pair_index} outside 0-{len(pairs) - 1}")
        sys.exit(1)

    p_name, sst_name = pairs[args.pair_index]
    return p_name, sst_name, args.pair_index


# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------

def drop_scalar_coords(obj):
    """
    Drop coordinates that are not dimensions.

    SST files carry a scalar depth coordinate `lev`, and its value differs
    between ERSSTv6 and COBE-SST3. Anything derived from the SST field inherits
    it, so concatenating across (precip x SST) members raises

        MergeError: conflicting values for variable 'lev'

    17_PCA_method.py:648-649 squeezes it away with drop=True for the same
    reason. Dropping every non-dimension coordinate is the general form: they
    are metadata, nothing downstream selects on them, and leaving any of them
    attached is a latent merge conflict.
    """
    return obj.drop_vars([c for c in obj.coords if c not in obj.dims])


def load_and_prepare_pair(p_name: str, sst_name: str, verbose: bool = True) -> tuple:
    """
    Load one (precip, SST) pair, align on common time, and detrend.

    Mirrors 11_Linear_Regression_Bootstrap_SE.py:328-337. The intersection (not
    union) is used for the time axis, matching script 11 line 330 — note that
    17_PCA_method.py:669 uses a NaN-padded union instead, so anything comparing
    the two methods must realign explicitly.

    Returns
    -------
    p_anom   : (time, basin)          anomalies, NOT detrended
    sst_anom : (time, lat, lon)       anomalies, NOT detrended — this is what the
                                      reconstruction is built from, as at 11:384
    p_det    : (time, basin)          detrended, float32 — what the fit uses
    sst_det  : (time, lat, lon)       detrended, float32 — what the fit uses
    """
    p_anom   = xr.open_dataarray(INPUTS_DIR / f"precip_anom_{p_name}.nc").load()
    sst_anom = xr.open_dataarray(INPUTS_DIR / f"sst_anom_{sst_name}.nc").squeeze().load()

    # Strip scalar coords (notably `lev`) before anything derives from these —
    # see drop_scalar_coords. Doing it here means reconstructions, masks and
    # diagnostics are all born concat-safe.
    p_anom   = drop_scalar_coords(p_anom)
    sst_anom = drop_scalar_coords(sst_anom)

    common_time = np.intersect1d(p_anom["time"].values, sst_anom["time"].values)
    p_anom      = p_anom.sel(time=common_time)
    sst_anom    = sst_anom.sel(time=common_time)

    p_det   = detrend_dim(p_anom,   "time").astype(np.float32)
    sst_det = detrend_dim(sst_anom, "time").astype(np.float32)

    if verbose:
        print(f"  Common time steps: {len(common_time)}")
        print(f"  Basins: {p_anom.sizes['basin']}  |  Grid: "
              f"{sst_anom.sizes['lat']} x {sst_anom.sizes['lon']}")

    return p_anom, sst_anom, p_det, sst_det


# ---------------------------------------------------------------------------
# Regression and weighting
# ---------------------------------------------------------------------------

def fit_regression(sst_det: xr.DataArray, p_det: xr.DataArray,
                   verbose: bool = True) -> tuple:
    """
    Per-cell, per-basin OLS of detrended precip on detrended SST.

    Mirrors 11_Linear_Regression_Bootstrap_SE.py:343-351. Output dims are
    (lat, lon, basin): every SST cell is regressed independently against every
    basin, so `slope` is a marginal univariate sensitivity, not a coefficient
    from a joint fit.

    This is the expensive step and the only one the sweeps share, so callers
    should compute it once and vary the masking and weighting afterwards.

    Returns
    -------
    slope, pval : (lat, lon, basin) float32
    """
    slope, _, pval = xr.apply_ufunc(
        regression_slope_se,
        sst_det,
        p_det,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32],
    )

    if verbose:
        n_ocean = int((~np.isnan(pval.isel(basin=0))).sum())
        print(f"  Valid slopes: {int((~np.isnan(slope)).sum())}  "
              f"|  ocean cells per basin: {n_ocean}")

    return slope, pval


def area_weights(slope: xr.DataArray) -> tuple:
    """
    Grid-cell area weights, broadcast to the slope's basin axis.

    `grid_area` returns spherical-cap fractional areas that sum to 4*pi
    steradians over the globe (regression_functions.py:130-154), so a weighted
    sum is an area-weighted mean times ~12.57 rather than a normalised mean.
    Grid spacing is inferred from the coordinate, so this adapts to any regular
    grid without change.

    Returns the broadcast field as well as the (lat, lon) one because
    `reconstruct` contracts against (lat, lon, basin) and broadcasting once here
    is cheaper than doing it per replicate.

    Note what this does NOT return: an area-weighted slope. Multiplying area into
    the sensitivity is what made the saved fields impossible to label, so the
    product is formed only where the reconstruction is.

    Returns
    -------
    area      : (lat, lon)             steradians
    area_full : (lat, lon, basin)      area broadcast over basins
    """
    area      = grid_area(slope).astype(np.float32)
    area_full = area * xr.ones_like(slope)
    return area, area_full


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def reconstruct(sst_anom: xr.DataArray, slope_sig: xr.DataArray,
                area_full: xr.DataArray) -> xr.DataArray:
    """
    SST-forced reconstruction: sum over grid cells of area * beta * SST anomaly.

    Equivalent to `(sst_anom * area_full * slope_sig).sum(('lat','lon'))` in
    script 11 — xarray's `.sum` skips NaN, and filling NaN with 0 before the
    contraction gives the same answer — but uses a tensor contraction instead of
    materialising the full (time, lat, lon, basin) product, which is ~14 GB on
    the 2 degree grid.

    `area_full` is passed in rather than folded into `slope_sig` upstream: this
    is the step that owns the area factor, because it is the only step that is a
    spatial integral. `slope_sig` stays a sensitivity everywhere else.

    Note this uses the NON-detrended `sst_anom`, as script 11 does, even though
    the slopes were fitted on detrended SST.
    """
    return xr.dot(
        sst_anom.fillna(0.0),
        (area_full * slope_sig).fillna(0.0),
        dims=("lat", "lon"),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnostics(recon: xr.DataArray, p_det: xr.DataArray,
                mask: xr.DataArray, area: xr.DataArray) -> dict:
    """
    Per-basin summary statistics for one reconstruction variant.

    Returns a dict of (basin,) DataArrays:

      n_sig_cells  count of grid cells passing the significance mask
      area_sig     total steradian weight of those cells
      std_recon    std of the reconstruction over time
      std_ratio    100 * std(recon) / std(obs), the magnitude metric Figure 7 uses
      correlation  Pearson r between reconstruction and observed

    `p_det` is the detrended precipitation, so std_ratio and correlation are on
    the same footing as 17_PCA_method.py:490-496. Script 11 scores against the
    non-detrended `p_anom` instead (11:416, 436), so its saved correlation and
    any std_ratio derived from its saved `observed_precip` are not directly
    comparable to these.
    """
    std_recon = recon.std("time")
    std_obs   = p_det.std("time")

    return {
        "n_sig_cells": mask.sum(("lat", "lon")),
        "area_sig":    (area * mask).sum(("lat", "lon")),
        "std_recon":   std_recon,
        "std_ratio":   xr.where(std_obs > 0, std_recon / std_obs * 100.0, np.nan),
        "correlation": xr.corr(recon, p_det, dim="time"),
    }


# ---------------------------------------------------------------------------
# Trends
# ---------------------------------------------------------------------------
# Moved here from plotting_functions.py and amip_functions.py, which carried
# byte-identical copies. Scripts 11 and 16 now need it too, and code/scripts
# may not import from code/figures, so this module is the only home that serves
# all three.

def convert_time_to_years(da: xr.DataArray) -> xr.DataArray:
    """Convert time coordinate from datetime64 to fractional years (e.g. 1980.5)."""
    return da.assign_coords(
        time=da["time"].dt.year + (da["time"].dt.dayofyear - 1) / 365.0
    )


def linear_trend(da: xr.DataArray) -> xr.DataArray:
    """
    Linear trend per decade along the time dimension.

    Assumes time is in fractional years (use convert_time_to_years first).
    Output units: [input units / decade].
    """
    coeff = da.polyfit(dim="time", deg=1, skipna=True)
    trend = coeff.polyfit_coefficients.sel(degree=1) * 10.0
    # .sel leaves a scalar `degree` coordinate attached. Nothing selects on it,
    # but it survives into saved NetCDF and then into every to_dataframe() as a
    # spurious column, which collides the third time a GeoDataFrame is merged
    # (pandas has already used degree_x and degree_y by then).
    return trend.drop_vars("degree", errors="ignore")


def trend_over(da: xr.DataArray, period: tuple[str, str]) -> xr.DataArray:
    """Per-decade trend of `da` restricted to `period`, handling the time conversion."""
    return linear_trend(convert_time_to_years(da.sel(time=slice(*period))))


# ---------------------------------------------------------------------------
# Trend significance
# ---------------------------------------------------------------------------
# The test compares the ensemble-mean trend to the standard deviation of the
# bootstrap distribution OF THAT MEAN. It is the same construction already used
# for the sensitivity and reconstruction standard errors -- the SD of a
# bootstrap distribution -- applied to a third quantity, so it needs neither a
# percentile interval nor a bias correction.
#
# Why the bootstrap is needed at all: inter-member spread measures disagreement
# between dataset choices, and all members see the same record. Whatever that
# realization got wrong, they all get wrong together, and it cancels out of
# their spread entirely. Sampling uncertainty is invisible to them by
# construction, and it is the larger of the two sources: on the observational
# ensemble the within-member SD is 0.431 against 0.159 between members, so a
# test built only on inter-member spread uses a standard error roughly six
# times too small.

CI_Z                = 1.96   # two-sided 95%
AGREEMENT_THRESHOLD = 0.75   # fraction of members that must share the sign


def _member_trend_arrays(results_dict, period_label):
    """Stacked (bootstrap, point) member trends on a common basin axis."""
    boot, point = [], []
    for res in results_dict.values():
        tb = res["trend_boot"].sel(period=period_label)
        pt = res["reconstruction_trend"].sel(period=period_label)
        boot.append(tb.drop_vars([c for c in ("period",) if c in tb.coords]))
        point.append(pt.drop_vars([c for c in ("period",) if c in pt.coords]))

    common = boot[0]["basin"].values
    for da in boot[1:]:
        common = np.intersect1d(common, da["basin"].values)

    return (xr.concat([d.sel(basin=common) for d in boot],  dim="member"),
            xr.concat([d.sel(basin=common) for d in point], dim="member"))


def ensemble_trend_significance(results_dict, period_label, z: float = CI_Z):
    """
    Returns (point_trend, sd, significant), each (basin,).

    `sd` is the spread of the ensemble mean across replicates: average over
    members WITHIN replicate b, then take the standard deviation over b.

    Averaging within the replicate is what keeps the sampling error the members
    share. Drawing replicates independently per member averages that component
    away and understates the spread by roughly a factor of two.
    """
    boot_stack, point_stack = _member_trend_arrays(results_dict, period_label)
    theta_bar = boot_stack.mean("member")        # (bootstrap, basin)
    sd        = theta_bar.std("bootstrap")
    point     = point_stack.mean("member")
    return point, sd, (abs(point) > z * sd)


def sign_agreement(results_dict, period_label):
    """Fraction of members whose trend sign matches the ensemble-mean sign."""
    _, point_stack = _member_trend_arrays(results_dict, period_label)
    ens   = np.sign(point_stack.mean("member"))
    match = (np.sign(point_stack) == ens).where(np.isfinite(point_stack))
    frac  = match.mean("member", skipna=True)
    return xr.where(ens == 0, 0.5, frac)


def three_categories(significant, agreement, threshold: float = AGREEMENT_THRESHOLD):
    """
    0 = not significant, 1 = significant but low agreement, 2 = significant and robust.

    Two axes rather than one verdict. Three tests ANDed at nominal 5% each have
    no stated error rate and the result cannot be described as a 5% procedure;
    this keeps one claim that carries an error rate alongside one descriptive
    robustness statement, following IPCC map convention.
    """
    return xr.where(~significant, 0, xr.where(agreement >= threshold, 2, 1))
