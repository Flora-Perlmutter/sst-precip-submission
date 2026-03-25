#!/usr/bin/env python
# coding: utf-8
"""
Plotting and ensemble utility functions for sst-precipitation-sensitivity.

Usage
-----
    from plotting_functions import (
        basin_name_for_id,
        basin_id_for_name,
        load_and_combine,
        apply_fixdates_to_results,
        apply_fixdates_to_sst,
        compute_ensemble_means,
        compute_ensemble_mean_sst,
        convert_time_to_years,
        linear_trend,
        plot_ensemble_on_ax,
    )
"""

import pickle
import warnings
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load GRDC basin boundaries once at import time
# ---------------------------------------------------------------------------
_GRDC_PATH = CMIG_DATA / "Data" / "Other" / "grdc_basins"
grdc_basins = gpd.read_file(_GRDC_PATH)

# ---------------------------------------------------------------------------
# Fix Dates function
# ---------------------------------------------------------------------------
def fixdates(da: xr.DataArray) -> xr.DataArray:
    """Standardize monthly timestamps to the first day of each month."""
    if not hasattr(da, "time"):
        return da
    st_y = int(pd.to_datetime(da.time.values[0]).year)
    st_m = int(pd.to_datetime(da.time.values[0]).month)
    e_y  = int(pd.to_datetime(da.time.values[-1]).year)
    e_m  = int(pd.to_datetime(da.time.values[-1]).month)
    newdates = pd.date_range(f"{st_y}-{st_m}-1", f"{e_y}-{e_m}-1", freq="MS")
    return da.assign_coords(time=newdates)

# ---------------------------------------------------------------------------
# Basin name / ID lookups
# ---------------------------------------------------------------------------

def basin_name_for_id(basin_id: int) -> str:
    """Return a clean, title-cased basin name for a given GRDC basin ID."""
    try:
        nm = grdc_basins.loc[grdc_basins["MRBID"] == basin_id, "RIVER_BASI"].values[0]
    except Exception:
        return str(basin_id)
    return nm.strip().split("(")[0].strip().title()


def basin_id_for_name(basin_name: str):
    """Return the GRDC basin ID for a given basin name, or None if not found."""
    try:
        clean = basin_name.strip().split("(")[0]
        match = grdc_basins.loc[
            grdc_basins["RIVER_BASI"].str.strip().str.split("(").str[0] == clean,
            "MRBID",
        ]
        if not match.empty:
            return match.values[0]
        print(f"No match found for basin name '{basin_name}'")
        return None
    except Exception as e:
        print(f"Error looking up basin name '{basin_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_combine(files: list, exclude_amip: bool = False) -> dict:
    """
    Load and merge a list of pickle files into a single dict.

    Parameters
    ----------
    files        : list of Path objects
    exclude_amip : if True, skip files with 'amip' in the name;
                   if False, only load files with 'r1i1p1f1' in the name.
    """
    combined = {}
    for file_path in files:
        if exclude_amip and "amip" in file_path.name:
            continue
        if not exclude_amip and "r1i1p1f1" not in file_path.name:
            continue
        with open(file_path, "rb") as f:
            combined.update(pickle.load(f))
    return combined


# ---------------------------------------------------------------------------
# fixdates helpers
# ---------------------------------------------------------------------------

def apply_fixdates_to_results(results_dict: dict) -> dict:
    """Apply fixdates() to every DataArray with a time coordinate in a nested results dict."""
    return {
        key: {
            name: fixdates(val) if hasattr(val, "time") else val
            for name, val in subdict.items()
        }
        for key, subdict in results_dict.items()
    }


def apply_fixdates_to_sst(results_dict: dict) -> dict:
    """Apply fixdates() to every DataArray with a time coordinate in a flat results dict."""
    return {
        key: fixdates(sst) if hasattr(sst, "time") else sst
        for key, sst in results_dict.items()
    }


# ---------------------------------------------------------------------------
# Ensemble statistics
# ---------------------------------------------------------------------------

def drop_extra_coords(da):
    coords_to_drop = [c for c in da.coords if c not in da.dims]
    return da.drop_vars(coords_to_drop)


def compute_ensemble_means(results_dict: dict, label: str = "") -> dict:
    """
    Compute ensemble mean and spread across all (precip, SST) dataset combinations.

    Returns a dict with keys:
        marginal_sensitivity_sst, marginal_sensitivity_sst_se,
        marginal_sensitivity_sst_std, sst_reconstruction,
        sst_reconstruction_std, observed_precip, observed_precip_std,
        correlation_sst, correlation_sst_std
    """
    keys = ["marginal_sensitivity", "marginal_sensitivity_se",
            "reconstruction", "observed_precip", "correlation"]
    lists = {k: [] for k in keys}

    for result in results_dict.values():
        for k in keys:
            lists[k].append(drop_extra_coords(result[k]))

    def _stack_mean(k): return xr.concat(lists[k], dim="ensemble").mean(dim="ensemble")
    def _stack_std(k):  return xr.concat(lists[k], dim="ensemble").std(dim="ensemble")

    ensemble_mean = {
        "marginal_sensitivity_sst":     _stack_mean("marginal_sensitivity"),
        "marginal_sensitivity_sst_se":  _stack_mean("marginal_sensitivity_se"),
        "marginal_sensitivity_sst_std": _stack_std("marginal_sensitivity"),
        "sst_reconstruction":           _stack_mean("reconstruction"),
        "sst_reconstruction_std":       _stack_std("reconstruction"),
        "observed_precip":              _stack_mean("observed_precip"),
        "observed_precip_std":          _stack_std("observed_precip"),
        "correlation_sst":              _stack_mean("correlation"),
        "correlation_sst_std":          _stack_std("correlation"),
    }

    print(f"Computed ensemble means for {label}")
    return ensemble_mean


def compute_ensemble_mean_sst(sst_dict: dict) -> xr.DataArray:
    """
    Compute ensemble mean SST over the common overlapping time period.

    Parameters
    ----------
    sst_dict : dict of {name: xr.DataArray}

    Returns
    -------
    xr.DataArray : ensemble mean SST
    """
    print("Computing ensemble mean SST...")

    datasets     = list(sst_dict.values())
    common_start = max(ds["time"].min().values for ds in datasets)
    common_end   = min(ds["time"].max().values for ds in datasets)
    print(f"  Common time period: {common_start} to {common_end}")

    sst_list = []
    for name, ds in sst_dict.items():
        subset = ds.sel(time=slice(common_start, common_end))
        print(f"  {name}: {len(subset['time'])} time steps")
        sst_list.append(drop_extra_coords(subset))

    ensemble_sst = xr.concat(sst_list, dim="ensemble").mean(dim="ensemble")
    print(f"  Ensemble mean SST shape: {ensemble_sst.shape}")
    return ensemble_sst


# ---------------------------------------------------------------------------
# Time utilities (duplicated from amip_functions for standalone use)
# ---------------------------------------------------------------------------

def convert_time_to_years(da: xr.DataArray) -> xr.DataArray:
    """Convert time coordinate from datetime64 to fractional years (e.g. 1980.5)."""
    return da.assign_coords(
        time=da["time"].dt.year + (da["time"].dt.dayofyear - 1) / 365.0
    )


def linear_trend(da: xr.DataArray) -> xr.DataArray:
    """
    Compute linear trend per decade along the time dimension.
    Assumes time is in fractional years (use convert_time_to_years first).
    """
    coeff = da.polyfit(dim="time", deg=1, skipna=True)
    return coeff.polyfit_coefficients.sel(degree=1) * 10.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_ensemble_on_ax(linear_results: dict, basin_id: int, ax) -> None:
    """
    Plot ensemble SST reconstructions (each dataset mean ± SE) and observed
    precipitation (ensemble mean ± SD) on a provided Axes object.

    Parameters
    ----------
    linear_results : dict keyed by (precip_name, sst_name)
    basin_id       : GRDC basin ID integer
    ax             : matplotlib Axes
    """
    recons, recons_se, precip_obs, times_list = [], [], [], []

    for res in linear_results.values():
        if "reconstruction" not in res or "reconstruction_se" not in res:
            continue

        r     = res["reconstruction"].sel(basin=basin_id)
        r_se  = res["reconstruction_se"].sel(basin=basin_id)
        p_obs = res["observed_precip"].sel(basin=basin_id)

        valid = np.isfinite(r) & np.isfinite(r_se)
        r, r_se = r.where(valid, drop=True), r_se.where(valid, drop=True)

        recons.append(r)
        recons_se.append(r_se)
        precip_obs.append(p_obs)
        times_list.append(r["time"])

    if not recons:
        raise ValueError("No reconstructions found in the provided result dictionary.")

    # Find common time across all datasets
    common_time = times_list[0].values
    for t in times_list[1:]:
        common_time = np.intersect1d(common_time, t.values)
    if len(common_time) == 0:
        raise ValueError("No overlapping time period across datasets.")

    recons     = [drop_extra_coords(r.sel(time=common_time))     for r in recons]
    recons_se  = [drop_extra_coords(r.sel(time=common_time))     for r in recons_se]
    precip_obs = [drop_extra_coords(p.sel(time=common_time))     for p in precip_obs]

    p_obs_stack = xr.concat(precip_obs, dim="ensemble")
    p_obs_mean  = p_obs_stack.mean("ensemble")
    p_obs_std   = p_obs_stack.std("ensemble")

    time = mdates.date2num(xr.concat(recons, dim="ensemble")["time"].values)

    # Observed precipitation
    ax.plot(time, p_obs_mean, color="steelblue", linewidth=1.8,
            label="AMIP precipitation mean", zorder=3)
    ax.fill_between(time,
                    p_obs_mean - p_obs_std,
                    p_obs_mean + p_obs_std,
                    color="steelblue", alpha=0.25, label="±1 SD", zorder=2)

    # Each dataset's reconstruction
    for i, (r, r_se) in enumerate(zip(recons, recons_se)):
        if r.values.shape != r_se.values.shape:
            raise ValueError(f"Shape mismatch between reconstruction and SE for dataset {i}")
        ax.plot(time, r.values, color="crimson", alpha=0.6, linewidth=0.8,
                label="Reconstruction dataset mean" if i == 0 else None, zorder=2)
        ax.fill_between(time,
                        r.values - r_se.values,
                        r.values + r_se.values,
                        color="crimson", alpha=0.15,
                        label="±1 SE" if i == 0 else None, zorder=1)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
