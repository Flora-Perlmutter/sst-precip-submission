import xarray as xr
import glob
import pandas as pd
import numpy as np
import cftime
import os
import pickle
import sys

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')

os.chdir(functions_dir)
from regression_functions import waterbasin, convert_to_mm_month
from data_processing_functions import compute_monthly_anomaly

def standardize_time_calendar(ds):
    """
    Convert any cftime calendar to standard datetime64[ns].
    This fixes the mixed calendar issue.
    """
    if 'time' in ds.coords:
        # Convert cftime to pandas datetime
        if isinstance(ds.time.values[0], (cftime.datetime, cftime.DatetimeNoLeap, 
                                          cftime.DatetimeGregorian, cftime.DatetimeJulian,
                                          cftime.DatetimeProlepticGregorian)):
            time_as_pd = [pd.Timestamp(f"{t.year:04d}-{t.month:02d}-{t.day:02d}") 
                         for t in ds.time.values]
            ds = ds.assign_coords(time=time_as_pd)
    return ds

def preprocess_pr(ds):
    """
    Preprocess precipitation: convert to mm/month and extract DataArray.
    Returns a Dataset for mfdataset compatibility.
    """
    # Standardize calendar first
    ds = standardize_time_calendar(ds)
    
    # Extract pr variable
    pr = ds['pr']
    
    pr_mm_month = convert_to_mm_month(pr)
    
    # Create a new dataset with just the converted pr variable
    return xr.Dataset({'pr': pr_mm_month})

def preprocess_tas(ds):
    """
    Preprocess surface temperature.
    Returns a Dataset for mfdataset compatibility.
    """
    # Standardize calendar first
    ds = standardize_time_calendar(ds)
    
    # Return the dataset (already has 'tas' variable)
    return ds

def extract_model_info(filepath):
    """
    Extract model name and ensemble member from CMIP6 filename.
    Example: tas_Amon_ACCESS-CM2_amip_r1i1p1f1_gn_197901-201412.nc
    Returns: (model_name, ensemble_member, full_identifier)
    """
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    if len(parts) >= 5:
        model_name = parts[2]  # e.g., ACCESS-CM2
        ensemble_member = parts[4]  # e.g., r1i1p1f1
        full_id = f"{model_name}_{ensemble_member}"
        return model_name, ensemble_member, full_id
    return "unknown", "unknown", "unknown"

def group_files_by_model(file_list):
    """
    Group files by model and ensemble member.
    Returns dict: {full_identifier: [list of files]}
    """
    grouped = {}
    for filepath in file_list:
        model_name, ensemble_member, full_id = extract_model_info(filepath)
        if full_id not in grouped:
            grouped[full_id] = []
        grouped[full_id].append(filepath)
    return grouped

def convert_time_to_years(da):
    """
    Convert time coordinate from datetime64 to fractional years (e.g. 1980.5).
    """
    return da.assign_coords(
        time = da['time'].dt.year + (da['time'].dt.dayofyear - 1) / 365.0
    )

def linear_trend(da):
    """
    Compute linear trend (per decade) along the time dimension for a DataArray.
    Assumes time coordinate is in years (float).
    Output has units of [input_units / decade].
    """
    coeff = da.polyfit(dim='time', deg=1, skipna=True)
    trend = coeff.polyfit_coefficients.sel(degree=1) * 10.0  # per decade
    return trend

def remove_duplicate_times(da):
    """
    Remove duplicate time values by keeping only the first occurrence.
    Also ensures time is sorted.
    """
    # Get unique times while preserving order
    _, unique_indices = np.unique(da.time.values, return_index=True)
    unique_indices = np.sort(unique_indices)  # Keep chronological order
    
    # Select only unique times
    da_unique = da.isel(time=unique_indices)
    
    return da_unique

def ensure_compatible_time_coords(da1, da2, da3):
    """
    Ensure three DataArrays have compatible time coordinates.
    Removes duplicates and finds common time overlap.
    """
    # Remove duplicates from each
    da1_clean = remove_duplicate_times(da1)
    da2_clean = remove_duplicate_times(da2)
    da3_clean = remove_duplicate_times(da3)
    
    # Find common time values (intersection)
    times1 = set(pd.DatetimeIndex(da1_clean.time.values))
    times2 = set(pd.DatetimeIndex(da2_clean.time.values))
    times3 = set(pd.DatetimeIndex(da3_clean.time.values))
    
    common_times = sorted(times1 & times2 & times3)
    
    if len(common_times) == 0:
        raise ValueError("No overlapping time periods found between datasets!")
    
    # Select common times
    da1_aligned = da1_clean.sel(time=common_times)
    da2_aligned = da2_clean.sel(time=common_times)
    da3_aligned = da3_clean.sel(time=common_times)
    
    return da1_aligned, da2_aligned, da3_aligned

def make_periodic_old(da):
    """
    Ensure longitude is periodic so interpolation at lon=0 won't produce NaNs.
    Adds a synthetic lon=0 point copied from the last grid point.
    """
    # Normalize to 0–360 and sort
    da = da.assign_coords(lon=(da.lon % 360))
    da = da.sortby("lon")

    lons = da.lon.values
    dlon = float(lons[1] - lons[0])
    
    # If grid does not include 0, add synthetic 0° point
    if not np.isclose(lons[0], 0.0, atol=1e-6):
        # Copy last point (e.g. lon=359.0625) and relabel as 0
        zero_pt = da.sel(lon=lons[-1])
        zero_pt = zero_pt.assign_coords(lon=0.0)
        
        # Insert at beginning
        da = xr.concat([zero_pt, da], dim="lon")
        da = da.sortby("lon")

    return da

def make_periodic(da):
    """
    Ensure longitude is periodic by adding a wraparound point at 360°.
    """
    da = da.assign_coords(lon=(da.lon % 360))
    da = da.sortby("lon")

    lons = da.lon.values

    if not np.isclose(lons[-1], 360.0, atol=1e-6):
        zero_pt = da.sel(lon=0, method="nearest")
        wrap_pt = zero_pt.assign_coords(lon=360.0)
        da = xr.concat([da, wrap_pt], dim="lon")

    return da
