#!/usr/bin/env python
# coding: utf-8
"""
Regression and spatial utility functions for sst-precipitation-sensitivity.

Usage
-----
    from regression_functions import (
        waterbasin,
        proccess_sst,
        detrend_dim,
        grid_area,
        fdr_correction,
        fdr_correction_field,
        regression_slope,
        regression_slope_se,
        regression_with_lags,
        regression_with_conditioning_var,
        regression_with_conditioning_var_se,
        regression_with_conditioning_var_and_interaction,
        regression_with_conditioning_var_and_interaction_se,
        regression_with_conditioning_var_return_all,
        regression_with_conditioning_var_and_interaction_return_all,
        regression_poly2_raw,
        regression_log_raw,
    )

Note: convert_to_mm_month is defined in data_processing_functions.py.
      Import it from there to avoid duplication.
"""

import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from scipy.stats import f

# geopandas and regionmask are imported inside waterbasin(), the only function
# that uses them. At module level they are a hard requirement for every consumer
# of this module, including the regression, trend and FDR helpers that never
# touch a shapefile — sensitivity_common imports detrend_dim, grid_area and
# fdr_correction from here, so scripts 11 and 16 and Figure 10 would all need
# geopandas installed to run.
#
# It also matters across environments: xesmf_env carries a regionmask too old
# for its NumPy 2.x (regionmask still references np.NaN, removed in 2.0), so a
# module-level import breaks anything needing xesmf but not basin masking.

# Shared utility — import from sibling module, do not redefine here
from data_processing_functions import convert_to_mm_month

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Spatial / basin utilities
# ---------------------------------------------------------------------------

def waterbasin(data: xr.DataArray) -> xr.DataArray:
    """
    Compute area-weighted basin mean precipitation using GRDC basin boundaries.

    Parameters
    ----------
    data : xr.DataArray
        Input data with lat/lon coordinates.

    Returns
    -------
    xr.DataArray
        Basin-mean time series with a 'basin' dimension.
    """
    import geopandas as gpd
    import regionmask

    # Normalize longitude to -180/180
    data = data.assign_coords(lon=((data.lon + 180) % 360 - 180)).sortby("lon")

    # Load GRDC basin boundaries
    grdc_basins = gpd.read_file(
        CMIG_DATA / "Data" / "Other" / "grdc_basins"
    ).rename(columns={"MRBID": "region_id", "RIVER_BASI": "region_name"})

    # Build RegionMask from basin geometries
    regions = regionmask.from_geopandas(
        grdc_basins[["region_id", "geometry"]],
        names="region_id",
        numbers="region_id",
    )

    # 3D fractional overlap mask and latitude weights
    fracmask = regions.mask_3D_frac_approx(data)
    weights  = np.cos(np.deg2rad(data.lat))

    # Area-weighted basin mean
    basin_mean = (
        data.weighted(fracmask * weights)
            .mean(dim=("lat", "lon"))
            .rename(region="basin")
    )

    # Drop metadata coords added by regionmask that downstream code doesn't need
    for coord in ["names", "abbrevs"]:
        if coord in basin_mean.coords:
            basin_mean = basin_mean.drop_vars(coord)

    return basin_mean



# ---------------------------------------------------------------------------
# Grid / detrending utilities
# ---------------------------------------------------------------------------

def detrend_dim(da: xr.DataArray, dim: str, deg: int = 1) -> xr.DataArray:
    """Remove polynomial trend of degree `deg` along dimension `dim`."""
    p   = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit


def grid_area(xarray: xr.DataArray) -> xr.DataArray:
    """
    Compute fractional surface area of each grid cell on a lat/lon grid.

    Returns
    -------
    xr.DataArray with dims ['lat', 'lon'], values in fractional Earth surface area.
    """
    lat = xarray["lat"]
    lon = xarray["lon"]

    lat_interval = abs(float(lat[1] - lat[0]))
    lon_interval = abs(float(lon[1] - lon[0]))

    lat_rad = np.deg2rad(lat)
    dlat    = lat_interval * (np.pi / 180.0)
    dlon    = lon_interval * (np.pi / 180.0)

    area = dlon * (np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2))

    return xr.DataArray(
        np.broadcast_to(area, (len(lon), len(lat))).T,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
    )


# ---------------------------------------------------------------------------
# FDR correction
# ---------------------------------------------------------------------------

def fdr_correction_single(p_values_1d: np.ndarray, alpha_FDR: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a 1D array, ignoring NaNs.

    Implements the BH *step-up* procedure: find the largest k for which
    p_(k) <= k*alpha/N, then reject every hypothesis up to and including k.
    The rejection set is therefore always a prefix of the sorted p-values.
    """
    valid_mask    = ~np.isnan(p_values_1d)
    valid_p       = p_values_1d[valid_mask]

    result = np.full(len(p_values_1d), False, dtype=bool)
    if len(valid_p) == 0:
        return result

    sorted_idx    = np.argsort(valid_p)
    sorted_p      = valid_p[sorted_idx]
    N             = len(sorted_p)
    threshold     = np.arange(1, N + 1) * alpha_FDR / N

    # Step-up: largest passing rank, then reject everything at or below it.
    passing    = np.nonzero(sorted_p <= threshold)[0]
    sig_sorted = np.zeros(N, dtype=bool)
    if passing.size:
        sig_sorted[: passing[-1] + 1] = True

    sig_unsorted = np.empty(N, dtype=bool)
    sig_unsorted[sorted_idx] = sig_sorted

    result[valid_mask] = sig_unsorted
    return result


def fdr_correction_field(p_values_nd: np.ndarray, alpha_FDR: float = 0.05) -> np.ndarray:
    """Apply BH to an entire N-D block of p-values treated as a single family."""
    p_values_nd = np.asarray(p_values_nd)
    flat        = fdr_correction_single(p_values_nd.ravel(), alpha_FDR)
    return flat.reshape(p_values_nd.shape)


def fdr_correction(
    p_values: xr.DataArray,
    alpha_FDR: float = 0.05,
    core_dims: tuple = ("lat", "lon"),
) -> xr.DataArray:
    """
    Apply BH FDR correction with the test family defined by `core_dims`.

    The dimensions listed in `core_dims` are pooled into one family of
    simultaneous tests; every other dimension is looped over independently.

    The default ("lat", "lon") treats all SST grid cells for a given basin as
    one family. That is the family the reconstruction subsequently sums over
    (11_Linear_Regression_Bootstrap_SE.py:384), so this is the correction that
    actually controls the false discovery rate of the reconstruction.

    Pass core_dims=("basin",) to reproduce the pre-2026-08 behaviour, in which
    the family was basins and each grid cell was corrected independently.
    """
    dims = [d for d in core_dims if d in p_values.dims]
    if not dims:
        raise ValueError(
            f"None of core_dims={core_dims} found in p_values.dims={p_values.dims}"
        )

    return xr.apply_ufunc(
        fdr_correction_field,
        p_values,
        input_core_dims=[dims],
        output_core_dims=[dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        kwargs={"alpha_FDR": alpha_FDR},
    )


# ---------------------------------------------------------------------------
# Regression functions
# ---------------------------------------------------------------------------

def regression_slope(sst, precip):
    """
    OLS regression: precip ~ intercept + sst.

    Returns: slope, p-value, AIC, RMSE, NRMSE
    """
    sst    = np.asarray(sst)
    precip = np.asarray(precip)
    valid  = ~np.isnan(sst) & ~np.isnan(precip)

    if np.sum(valid) < 2 or np.all(sst[valid] == sst[valid][0]):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    X     = sm.add_constant(sst[valid])
    model = sm.OLS(precip[valid], X).fit()
    resid = precip[valid] - model.predict(X)
    rmse  = np.sqrt(np.mean(resid ** 2))

    return (
        model.params[1],
        model.pvalues[1],
        model.aic,
        rmse,
        rmse / np.std(precip[valid]),
    )


def regression_slope_se(sst, precip):
    """
    OLS regression: precip ~ intercept + sst.

    Returns: slope, slope_se, p-value
    """
    sst    = np.asarray(sst)
    precip = np.asarray(precip)
    valid  = ~np.isnan(sst) & ~np.isnan(precip)

    if np.sum(valid) < 2 or np.all(sst[valid] == sst[valid][0]):
        return np.nan, np.nan, np.nan

    X     = sm.add_constant(sst[valid])
    model = sm.OLS(precip[valid], X).fit()
    return model.params[1], model.bse[1], model.pvalues[1]


def regression_with_lags(sst, precip, n_lags: int = 3):
    """
    Regress precip on SST with n lagged SST predictors.

    Returns: lag_coef_0, ..., lag_coef_n, f_p_value, AIC, RMSE, NRMSE
    """
    nan_return = tuple([np.nan] * (n_lags + 4))

    if np.isnan(sst).any() or np.isnan(precip).any():
        return nan_return
    if n_lags < 1 or len(sst) <= n_lags:
        return nan_return

    X_cols = [sst[n_lags - 1 - lag: -lag if lag != 0 else None] for lag in range(n_lags)]
    X = np.hstack([np.ones((len(X_cols[0]), 1)), np.column_stack(X_cols)])
    Y = precip[n_lags - 1:]

    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        Y_pred   = X @ beta
        rss_full = np.sum((Y - Y_pred) ** 2)
        rss_null = np.sum((Y - Y.mean()) ** 2)

        n, p = len(Y), X.shape[1] - 1
        df1, df2 = p, n - p - 1

        if df2 <= 0 or rss_full <= 0:
            return tuple([np.nan] * (n_lags + 3))

        F_stat    = ((rss_null - rss_full) / df1) / (rss_full / df2)
        f_p_value = 1 - f.cdf(F_stat, df1, df2)
        rmse      = np.sqrt(np.mean((Y - Y_pred) ** 2))

        return tuple(beta[1:]) + (f_p_value, n * np.log(rss_full / n) + 2 * (p + 1), rmse, rmse / np.std(Y))

    except (np.linalg.LinAlgError, ValueError):
        return nan_return


def regression_with_conditioning_var(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con.

    Returns: slope_sst, pvalue_sst, AIC, RMSE, NRMSE
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    X     = sm.add_constant(np.column_stack((sst, con)))
    model = sm.OLS(precip, X).fit()
    rmse  = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    return model.params[1], model.pvalues[1], model.aic, rmse, rmse / np.std(precip)


def regression_with_conditioning_var_se(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con, with standard errors.

    Returns: slope_sst, slope_sst_se, pvalue_sst, slope_con, slope_con_se, pvalue_con
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    X     = sm.add_constant(np.column_stack((sst, con)))
    model = sm.OLS(precip, X).fit()
    return (
        model.params[1], model.bse[1], model.pvalues[1],
        model.params[2], model.bse[2], model.pvalues[2],
    )


def regression_with_conditioning_var_return_all(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con.

    Returns: slope_sst, pvalue_sst, slope_con, pvalue_con, r (corr obs vs pred)
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    X     = sm.add_constant(np.column_stack((sst, con)))
    model = sm.OLS(precip, X).fit()
    r     = np.corrcoef(model.fittedvalues, precip)[0, 1]
    return model.params[1], model.pvalues[1], model.params[2], model.pvalues[2], r


def regression_with_conditioning_var_and_interaction(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con + sst*con.

    Returns: slope_sst, slope_interaction, pvalue_sst, pvalue_interaction, AIC, RMSE, NRMSE
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    X     = sm.add_constant(np.column_stack((sst, con, sst * con)))
    model = sm.OLS(precip, X).fit()
    rmse  = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    return (
        model.params[1], model.params[3],
        model.pvalues[1], model.pvalues[3],
        model.aic, rmse, rmse / np.std(precip),
    )


def regression_with_conditioning_var_and_interaction_se(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con + sst*con, with standard errors.

    Returns: slope_sst, se_sst, p_sst, slope_rh, se_rh, p_rh, slope_int, se_int, p_int
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return (np.nan,) * 9

    X     = sm.add_constant(np.column_stack((sst, con, sst * con)))
    model = sm.OLS(precip, X).fit()
    return (
        model.params[1], model.bse[1], model.pvalues[1],
        model.params[2], model.bse[2], model.pvalues[2],
        model.params[3], model.bse[3], model.pvalues[3],
    )


def regression_with_conditioning_var_and_interaction_return_all(sst, con, precip):
    """
    OLS: precip ~ intercept + sst + con + sst*con.

    Returns: slope_sst, slope_rh, slope_int, p_sst, p_rh, p_int, r (corr obs vs pred)
    """
    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return (np.nan,) * 7

    X     = sm.add_constant(np.column_stack((sst, con, sst * con)))
    model = sm.OLS(precip, X).fit()
    r     = np.corrcoef(model.fittedvalues, precip)[0, 1]
    return (
        model.params[1], model.params[2], model.params[3],
        model.pvalues[1], model.pvalues[2], model.pvalues[3],
        r,
    )


def regression_poly2_raw(sst, precip):
    """
    OLS: precip ~ intercept + sst + sst^2.

    Returns: intercept, beta1, beta2, p1, p2, AIC, RMSE, NRMSE
    """
    if np.isnan(sst).any() or np.isnan(precip).any():
        return (np.nan,) * 8

    try:
        X     = sm.add_constant(np.column_stack((sst, sst ** 2)))
        model = sm.OLS(precip, X).fit()
        rmse  = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
        return (
            model.params[0], model.params[1], model.params[2],
            model.pvalues[1], model.pvalues[2],
            model.aic, rmse, rmse / np.std(precip),
        )
    except Exception:
        return (np.nan,) * 8


def regression_log_raw(sst, precip):
    """
    OLS: precip ~ intercept + sst + log(sst).

    Returns: intercept, beta1, beta2, p1, p2, AIC, RMSE, NRMSE
    SST must be strictly positive to compute log.
    """
    if np.any(sst <= 0) or np.isnan(sst).any() or np.isnan(precip).any():
        return (np.nan,) * 8

    try:
        X     = sm.add_constant(np.column_stack((sst, np.log(sst))))
        model = sm.OLS(precip, X).fit()
        rmse  = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
        return (
            model.params[0], model.params[1], model.params[2],
            model.pvalues[1], model.pvalues[2],
            model.aic, rmse, rmse / np.std(precip),
        )
    except Exception:
        return (np.nan,) * 8