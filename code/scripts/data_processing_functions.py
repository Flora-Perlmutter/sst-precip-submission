#!/usr/bin/env python
# coding: utf-8
"""
Core data processing utility functions for sst-precipitation-sensitivity.

Usage
-----
Import into processing or figure scripts as needed:
    from data_processing_functions import (
        convert_to_mm_month,
        fix_lons,
        fixdates,
        load_and_preprocess_data,
        mask_sea_ice,
        compute_monthly_anomaly,
        determine_common_time_period,
    )
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def convert_to_mm_month(data: xr.DataArray) -> xr.DataArray:
    """
    Convert precipitation data to mm/month from any common input unit.

    Supported input units
    ---------------------
    m/s, kg/m2/s (and variants), W/m2 (and variants),
    mm/day, m, cm, mm/month (and variants)
    """
    if not hasattr(data, "units"):
        print("No units attribute found — returning data unchanged.")
        return data

    units = data.units

    if units == "m/s":
        mmperday   = data * 86_400_000
        mmpermonth = mmperday * mmperday.time.dt.days_in_month
    elif units in ["kg/m2/s", "kg/m^2/s", "kg m-2 s-1"]:
        mmperday   = data * 86_400
        mmpermonth = mmperday * mmperday.time.dt.days_in_month
    elif units in ["W/m2", "W/m^2", "W m-2"]:
        mmperday   = (data * 86_400) / (2.25e6)
        mmpermonth = mmperday * mmperday.time.dt.days_in_month
    elif units == "mm/day":
        mmpermonth = data * data.time.dt.days_in_month
    elif units == "m":
        mmpermonth = data * 1000
    elif units == "cm":
        mmpermonth = data * 10
    elif units in ["mm/month", "mm month-1"]:
        mmpermonth = data
    else:
        print(f"Unrecognized units '{units}' — returning data unchanged.")
        return data

    mmpermonth.attrs.update(data.attrs)
    mmpermonth.attrs["units"] = "mm/month"
    return mmpermonth


# ---------------------------------------------------------------------------
# Coordinate standardization
# ---------------------------------------------------------------------------

def fix_lons(da: xr.DataArray) -> xr.DataArray:
    """Convert longitude from 0-360 to -180/180 if needed, then sort."""
    lon = da["lon"]
    if lon.min() >= 0 and lon.max() > 180:
        da = da.assign_coords(lon=((lon.values + 180) % 360 - 180))
        da = da.sortby("lon")
    return da


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
# Dataset loading
# ---------------------------------------------------------------------------

def load_and_preprocess_data(
    dset_dict: dict,
    time_slice: slice = slice("1979-01-01", "2025-12-31"),
) -> xr.DataArray:
    """
    Load and standardize a single precipitation or SST dataset.

    Parameters
    ----------
    dset_dict : dict with keys:
        'path'  — file path or glob pattern
        'var'   — variable name to extract
        'name'  — (optional) human-readable label for error messages
    time_slice : slice
        Time range to select after loading.

    Returns
    -------
    xr.DataArray
    """
    path     = dset_dict["path"]
    var_name = dset_dict["var"]
    name     = dset_dict.get("name", path)

    try:
        if "*" in str(path):
            ds = xr.open_mfdataset(path, combine="by_coords", chunks={"time": 120})
        else:
            ds = xr.open_dataset(path, chunks={"time": 120})
    except Exception as e:
        raise RuntimeError(f"Failed to open dataset '{name}': {e}")

    # Standardize dimension names
    rename_map = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in ds.dims}
    if rename_map:
        ds = ds.rename(rename_map)

    da = fix_lons(ds[var_name])
    da = convert_to_mm_month(da)
    da = fixdates(da.sortby(["time", "lat", "lon"]))
    da = da.sel(time=time_slice, drop=True).sortby(["time", "lat", "lon"])

    return da


# ---------------------------------------------------------------------------
# Ocean / sea ice masking
# ---------------------------------------------------------------------------

def mask_sea_ice(sst: xr.DataArray) -> xr.DataArray:
    """
    Mask grid points where SST is near or below freezing more than 20% of the time.
    Threshold: 273.15 K.
    """
    freeze_fraction = (sst <= 273.15).sum("time") / sst["time"].size
    return sst.where(freeze_fraction < 0.2)


# ---------------------------------------------------------------------------
# Anomaly computation
# ---------------------------------------------------------------------------

def compute_monthly_anomaly(
    da: xr.DataArray,
    climatology_period: slice = None,
) -> xr.DataArray:
    """
    Compute monthly anomalies by removing the monthly climatology.

    Parameters
    ----------
    da : xr.DataArray
        Input data with a 'time' dimension.
    climatology_period : slice, optional
        Time slice to compute the climatology over (e.g. slice('1981-01-01','2010-12-31')).
        If None or fewer than 12 months available, uses the full record.
    """
    if climatology_period is not None:
        try:
            clim_slice = da.sel(time=climatology_period)
            clim = (
                clim_slice.groupby("time.month").mean("time", skipna=True)
                if clim_slice.time.size >= 12
                else da.groupby("time.month").mean("time", skipna=True)
            )
        except Exception:
            clim = da.groupby("time.month").mean("time", skipna=True)
    else:
        clim = da.groupby("time.month").mean("time", skipna=True)

    return da.groupby("time.month") - clim


# ---------------------------------------------------------------------------
# Time period utilities
# ---------------------------------------------------------------------------

def determine_common_time_period(
    precip_dict: dict,
    sst_dict: dict,
) -> tuple:
    """
    Find the overlapping time period across all precipitation and SST datasets.

    Parameters
    ----------
    precip_dict : dict of {name: xr.DataArray}
    sst_dict    : dict of {name: xr.DataArray}

    Returns
    -------
    (common_start, common_end) as pandas Timestamps
    """
    all_das = list(precip_dict.values()) + list(sst_dict.values())
    starts  = [pd.Timestamp(da.time.values[0])  for da in all_das]
    ends    = [pd.Timestamp(da.time.values[-1]) for da in all_das]

    common_start = max(starts)
    common_end   = min(ends)
    n_months     = len(pd.date_range(common_start, common_end, freq="MS"))

    print(f"\n{'='*60}")
    print(f"COMMON TIME PERIOD")
    print(f"{'='*60}")
    print(f"  Start:        {common_start.strftime('%Y-%m-%d')}")
    print(f"  End:          {common_end.strftime('%Y-%m-%d')}")
    print(f"  Total months: {n_months}")
    print(f"{'='*60}\n")

    return common_start, common_end