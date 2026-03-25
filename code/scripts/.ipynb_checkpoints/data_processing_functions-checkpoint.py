import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import pickle
import glob
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import regionmask
import cartopy.io.shapereader as shpreader

def convert_to_mm_month(data):
    # Check if data has units attribute
    if not hasattr(data, 'units'):
        print("No units")
        return data
    
    units = data.units
    
    if units == 'm/s':
        # get mm per month: (originally m/s)
        mmperday = data * 86400000
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['kg/m2/s', 'kg/m^2/s', 'kg m-2 s-1']:  
        # get mm per month: (originally kg/m2/s)
        mmperday = data * 86400
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['W/m2', 'W/m^2', 'W m-2']:
        # get mm per month: (originally W/m^2)
        mmperday = data * 86400
        mmperday = mmperday / (2.25*10**6)
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'mm/day':
        # get mm per month: (originally mm/day)
        mmperday = data
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'm':
        # get mm per month: (originally m/month)
        mmpermonth = data * 1000
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'cm':
        # get mm per month: (originally cm/month)
        mmpermonth = data * 10
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['mm/month', 'mm month-1']:
        # already in mm per month
        mmpermonth = data
    else:
        # Exception: units don't fall into a recognized category
        print(f"Exception units = {units}")
        return data
    
    return mmpermonth

def fix_lons(da):
    """Converts longitude values from (0 to 360) to (-180 to 180)."""
    lon = da['lon']
    if lon.min() >= 0 and lon.max() > 180:
        lon360 = lon.values
        lon180 = (lon360 + 180) % 360 - 180
        da = da.assign_coords(lon=lon180)
        da = da.sortby('lon')
    return da

def fixdates(da):
    """Standardizes monthly timestamps to the first day of each month."""
    if not hasattr(da, 'time'):
        return da
    st_y = int(pd.to_datetime(da.time.values[0]).year)
    st_m = int(pd.to_datetime(da.time.values[0]).month)
    e_y = int(pd.to_datetime(da.time.values[-1]).year)
    e_m = int(pd.to_datetime(da.time.values[-1]).month)
    newdates = pd.date_range(f'{st_y}-{st_m}-1', f'{e_y}-{e_m}-1', freq='MS')
    da = da.assign_coords(time=newdates)
    return da

def load_and_preprocess_data(dset_dict, time_slice=slice('1980-01-01', '2024-12-31')):
    """Loads and standardizes a single dataset (Precip or SST)."""
    path = dset_dict['path']
    var_name = dset_dict['var']
    name = dset_dict.get('name', '')

    try:
        if  '*' in path:
            ds = xr.open_mfdataset(path, combine='by_coords', preprocess=lambda x: x)
        else:
            ds = xr.open_dataset(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open {path} for dataset {name}: {e}")

    # Standardize dimension names
    rename_map = {}
    if 'latitude' in ds.dims:
        rename_map['latitude'] = 'lat'
    if 'longitude' in ds.dims:
        rename_map['longitude'] = 'lon'
    if rename_map:
        ds = ds.rename(rename_map)

    da = ds[var_name]
    da = fix_lons(da)

    
    da = convert_to_mm_month(da)

    da = da.sortby(['time', 'lat', 'lon'])
    da = fixdates(da)
    da = da.sel(time=time_slice, drop=True).sortby(['time', 'lat', 'lon'])

    return da


def mask_sea_ice(sst):
    """Mask grid points where SST is frequently near/below freezing."""
    freeze_threshold = (sst <= 273.15)
    fraction_near_freeze = freeze_threshold.sum('time') / sst['time'].size
    
    # Mask grid points where >20% of the time SST is near freezing
    sst_clean = sst.where(fraction_near_freeze < 0.2)
    return sst_clean


def compute_monthly_anomaly(da, climatology_period=None):
    """Compute monthly anomalies by removing the monthly climatology."""
    if climatology_period is not None:
        try:
            clim_slice = da.sel(time=climatology_period)
            if clim_slice.time.size < 12:
                clim = da.groupby('time.month').mean('time', skipna=True)
            else:
                clim = clim_slice.groupby('time.month').mean('time', skipna=True)
        except Exception:
            clim = da.groupby('time.month').mean('time', skipna=True)
    else:
        clim = da.groupby('time.month').mean('time', skipna=True)

    anom = da.groupby('time.month') - clim
    return anom


def determine_common_time_period(precip_dict, sst_dict):
    """
    Determine the common overlapping time period across all precipitation and SST datasets.
    
    Returns:
        tuple: (common_start, common_end) as pandas Timestamps
    """
    all_starts = []
    all_ends = []
    
    # Get time ranges from precipitation datasets
    for p_name, p_da in precip_dict.items():
        all_starts.append(pd.Timestamp(p_da.time.values[0]))
        all_ends.append(pd.Timestamp(p_da.time.values[-1]))
    
    # Get time ranges from SST datasets
    for sst_name, sst_da in sst_dict.items():
        all_starts.append(pd.Timestamp(sst_da.time.values[0]))
        all_ends.append(pd.Timestamp(sst_da.time.values[-1]))
    
    # Common period is the latest start and earliest end
    common_start = max(all_starts)
    common_end = min(all_ends)
    
    print(f"\n{'='*80}")
    print(f"COMMON TIME PERIOD DETERMINATION")
    print(f"{'='*80}")
    print(f"Common start: {common_start.strftime('%Y-%m-%d')}")
    print(f"Common end:   {common_end.strftime('%Y-%m-%d')}")
    print(f"Total months: {len(pd.date_range(common_start, common_end, freq='MS'))}")
    print(f"{'='*80}\n")
    
    return common_start, common_end

