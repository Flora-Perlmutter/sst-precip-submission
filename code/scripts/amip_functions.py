#!/usr/bin/env python
# coding: utf-8
"""
AMIP/CMIP6 data processing utility functions for sst-precipitation-sensitivity.

Usage
-----
Import into processing or figure scripts as needed:
    from amip_functions import (
        standardize_time_calendar,
        preprocess_pr,
        preprocess_tas,
        extract_model_info,
        group_files_by_model,
        convert_time_to_years,
        linear_trend,
        remove_duplicate_times,
        ensure_compatible_time_coords,
        make_periodic,
    )
"""

import os
import sys
from pathlib import Path

import cftime
import numpy as np
import pandas as pd
import xarray as xr

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

def standardize_time_calendar(ds: xr.Dataset) -> xr.Dataset:
    """Convert any cftime calendar to standard datetime64[ns]."""
    if "time" in ds.coords:
        if isinstance(
            ds.time.values[0],
            (
                cftime.datetime,
                cftime.DatetimeNoLeap,
                cftime.DatetimeGregorian,
                cftime.DatetimeJulian,
                cftime.DatetimeProlepticGregorian,
            ),
        ):
            time_as_pd = [
                pd.Timestamp(f"{t.year:04d}-{t.month:02d}-{t.day:02d}")
                for t in ds.time.values
            ]
            ds = ds.assign_coords(time=time_as_pd)
    return ds


def convert_time_to_years(da: xr.DataArray) -> xr.DataArray:
    """Convert time coordinate from datetime64 to fractional years (e.g. 1980.5)."""
    return da.assign_coords(
        time=da["time"].dt.year + (da["time"].dt.dayofyear - 1) / 365.0
    )


def remove_duplicate_times(da: xr.DataArray) -> xr.DataArray:
    """Remove duplicate time values, keeping first occurrence. Returns sorted result."""
    _, unique_indices = np.unique(da.time.values, return_index=True)
    return da.isel(time=np.sort(unique_indices))


def ensure_compatible_time_coords(
    da1: xr.DataArray,
    da2: xr.DataArray,
    da3: xr.DataArray,
) -> tuple:
    """
    Remove duplicates from three DataArrays and trim to their common time overlap.

    Returns
    -------
    (da1_aligned, da2_aligned, da3_aligned)
    """
    da1_clean = remove_duplicate_times(da1)
    da2_clean = remove_duplicate_times(da2)
    da3_clean = remove_duplicate_times(da3)

    common_times = sorted(
        set(pd.DatetimeIndex(da1_clean.time.values))
        & set(pd.DatetimeIndex(da2_clean.time.values))
        & set(pd.DatetimeIndex(da3_clean.time.values))
    )

    if not common_times:
        raise ValueError("No overlapping time periods found between datasets.")

    return (
        da1_clean.sel(time=common_times),
        da2_clean.sel(time=common_times),
        da3_clean.sel(time=common_times),
    )


# ---------------------------------------------------------------------------
# CMIP6 preprocessing
# ---------------------------------------------------------------------------

def preprocess_pr(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess CMIP6 precipitation: standardize calendar, convert to mm/month.
    Returns a Dataset for xr.open_mfdataset compatibility.
    """
    from data_processing_functions import convert_to_mm_month

    ds = standardize_time_calendar(ds)
    return xr.Dataset({"pr": convert_to_mm_month(ds["pr"])})


def preprocess_tas(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess CMIP6 surface temperature: standardize calendar only.
    Returns a Dataset for xr.open_mfdataset compatibility.
    """
    return standardize_time_calendar(ds)


# ---------------------------------------------------------------------------
# CMIP6 file utilities
# ---------------------------------------------------------------------------

def extract_model_info(filepath: str) -> tuple:
    """
    Extract model name and ensemble member from a CMIP6 filename.

    Example
    -------
    tas_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197901-201412.nc
    → ("ACCESS-CM2", "r1i1p1f1", "ACCESS-CM2_r1i1p1f1")
    """
    parts = os.path.basename(filepath).split("_")
    if len(parts) >= 5:
        model_name      = parts[2]
        ensemble_member = parts[4]
        return model_name, ensemble_member, f"{model_name}_{ensemble_member}"
    return "unknown", "unknown", "unknown"


def group_files_by_model(file_list: list) -> dict:
    """
    Group a list of CMIP6 filepaths by model and ensemble member.

    Returns
    -------
    dict : {full_identifier: [list of files]}
    """
    grouped = {}
    for filepath in file_list:
        _, _, full_id = extract_model_info(filepath)
        grouped.setdefault(full_id, []).append(filepath)
    return grouped


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def linear_trend(da: xr.DataArray) -> xr.DataArray:
    """
    Compute linear trend per decade along the time dimension.

    Assumes time coordinate is in fractional years (use convert_time_to_years first).
    Output units: [input_units / decade].
    """
    coeff = da.polyfit(dim="time", deg=1, skipna=True)
    return coeff.polyfit_coefficients.sel(degree=1) * 10.0


# ---------------------------------------------------------------------------
# Longitude handling
# ---------------------------------------------------------------------------

def make_periodic(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure longitude is periodic by adding a wraparound point at 360°.
    Normalizes to 0-360 and sorts before adding the wrap point.
    """
    da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    if not np.isclose(da.lon.values[-1], 360.0, atol=1e-6):
        wrap_pt = da.sel(lon=0, method="nearest").assign_coords(lon=360.0)
        da = xr.concat([da, wrap_pt], dim="lon")
    return da