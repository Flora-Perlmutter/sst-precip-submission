#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive analysis comparing Observations vs AMIP ensemble results
Focus on differences and what they reveal about model performance
"""

import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.gridspec as gridspec
import cartopy.feature as cfeature
import pickle
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from scipy.stats import linregress, pearsonr
from matplotlib.dates import DateFormatter
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import matplotlib.dates as mdates
import sys
import glob
import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import cartopy.feature as cfeature
import regionmask
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import seaborn as sns
import xskillscore
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import geopandas as gpd
import seaborn as sns
import xarray as xr
import matplotlib.gridspec as gridspec
from matplotlib.cm import RdBu
from scipy.stats import linregress
from scipy import stats
from scipy.stats import t
import pickle
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import warnings
import glob
from pathlib import Path
warnings.filterwarnings("ignore")


root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')

# load basin boundaries
grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other', "grdc_basins"),)

os.chdir(functions_dir)
from data_processing_functions import fixdates
warnings.filterwarnings("ignore")


def basin_name_for_id(basin_id):
    try:
        nm = grdc_basins.loc[grdc_basins['MRBID'] == basin_id, 'RIVER_BASI'].values[0]
    except Exception:
        nm = str(basin_id)
    # Clean up: drop leading/trailing spaces and anything after the first '('
    nm = nm.strip().split('(')[0]
    # Capitalize the first letter of each word
    nm = nm.title()
    return nm

def basin_id_for_name(basin_name):

    try:
        # Normalize name for comparison
        basin_name_clean = basin_name.strip().split('(')[0]
        match = grdc_basins.loc[
            grdc_basins['RIVER_BASI'].str.strip().str.split('(').str[0] == basin_name_clean,
            'MRBID'
        ]
        if not match.empty:
            return match.values[0]
        else:
            print(f"No match found for basin name '{basin_name}'")
            return None
    except Exception as e:
        print(f"Error looking up basin name '{basin_name}': {e}")
        return None

def load_and_combine(files, exclude_amip=False):
    """Load and combine pickle files"""
    combined = {}
    for file_path in files:
        if exclude_amip and 'amip' in file_path.name:
            continue
        if not exclude_amip and 'r1i1p1f1' not in file_path.name:
            continue
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        combined.update(data)
    return combined

def apply_fixdates_to_results(results_dict):
    """Apply fixdates() to every DataArray inside the results dictionary."""
    new_dict = {}

    for key, subdict in results_dict.items():
        new_subdict = {}
        for name, val in subdict.items():
            # apply only to DataArrays with time coordinates
            if hasattr(val, "time"):
                new_subdict[name] = fixdates(val)
            else:
                new_subdict[name] = val
        new_dict[key] = new_subdict

    return new_dict

def compute_ensemble_means(results_dict, label=""):
    """Compute ensemble mean and standard deviation across all model combinations"""
    marginal_sst_list = []
    marginal_sst_se_list = []
    recon_sst_list = []
    obs_precip_list = []
    corr_sst_list = []
    
    for (p_name, sst_name), result in results_dict.items():
        marginal_sst_list.append(result['marginal_sensitivity'])
        marginal_sst_se_list.append(result['marginal_sensitivity_se'])
        recon_sst_list.append(result['reconstruction'])
        obs_precip_list.append(result['observed_precip'])
        corr_sst_list.append(result['correlation'])
    
    ensemble_mean = {
        'marginal_sensitivity_sst': xr.concat(marginal_sst_list, dim='ensemble').mean(dim='ensemble'),
        'marginal_sensitivity_sst_se': xr.concat(marginal_sst_se_list, dim='ensemble').mean(dim='ensemble'),
        'marginal_sensitivity_sst_std': xr.concat(marginal_sst_list, dim='ensemble').std(dim='ensemble'),
        'sst_reconstruction': xr.concat(recon_sst_list, dim='ensemble').mean(dim='ensemble'),
        'sst_reconstruction_std': xr.concat(recon_sst_list, dim='ensemble').std(dim='ensemble'),
        'observed_precip': xr.concat(obs_precip_list, dim='ensemble').mean(dim='ensemble'),
        'observed_precip_std': xr.concat(obs_precip_list, dim='ensemble').std(dim='ensemble'),
        'correlation_sst': xr.concat(corr_sst_list, dim='ensemble').mean(dim='ensemble'),
        'correlation_sst_std': xr.concat(corr_sst_list, dim='ensemble').std(dim='ensemble'),
    }
    
    print(f"\nComputed ensemble means for {label}")
    return ensemble_mean

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

def plot_ensemble_on_ax(linear_results, basin_id, ax):
    """
    Plot ensemble reconstructions (each dataset mean ± SE)
    and ensemble mean observed precipitation (mean ± SD).
    Handles datasets with different time lengths safely.
    """
    import matplotlib.dates as mdates
    import xarray as xr
    import numpy as np

    recons = []
    recons_se = []
    precip_obs = []
    times_list = []

    # --- Collect reconstructions, SEs, and observations ---
    for (_, _), res in linear_results.items():
        if "reconstruction" not in res or "reconstruction_se" not in res:
            continue

        r = res["reconstruction"].sel(basin=basin_id)
        r_se = res["reconstruction_se"].sel(basin=basin_id)
        p_obs = res["observed_precip"].sel(basin=basin_id)

        # Drop any NaNs along time
        valid_mask = np.isfinite(r) & np.isfinite(r_se)
        r = r.where(valid_mask, drop=True)
        r_se = r_se.where(valid_mask, drop=True)

        recons.append(r)
        recons_se.append(r_se)
        precip_obs.append(p_obs)
        times_list.append(r["time"])

    if len(recons) == 0:
        raise ValueError("No reconstructions found in the provided result dictionary.")

    # --- Find common time range across all datasets ---
    common_time = sorted(set(times_list[0].values))
    for t in times_list[1:]:
        common_time = np.intersect1d(common_time, t.values)
    if len(common_time) == 0:
        raise ValueError("No overlapping time period across datasets.")

    # Reindex all to the common time base
    recons = [r.sel(time=common_time) for r in recons]
    recons_se = [r_se.sel(time=common_time) for r_se in recons_se]
    precip_obs = [p.sel(time=common_time) for p in precip_obs]

    # --- Stack into ensemble dimension ---
    recons_stack = xr.concat(recons, dim="ensemble")
    recons_se_stack = xr.concat(recons_se, dim="ensemble")
    p_obs_stack = xr.concat(precip_obs, dim="ensemble")

    # --- Compute ensemble mean and SD of observed precip ---
    p_obs_mean = p_obs_stack.mean("ensemble")
    p_obs_std = p_obs_stack.std("ensemble")

    # --- Convert time for Matplotlib ---
    time = mdates.date2num(recons_stack["time"].values)

    # ========================
    # Plot observed precip (mean ± SD)
    # ========================
    ax.plot(time, p_obs_mean, color="steelblue", linewidth=1.8,
            label="AMIP precipitation mean", zorder=3)
    ax.fill_between(time,
                    p_obs_mean - p_obs_std,
                    p_obs_mean + p_obs_std,
                    color="steelblue", alpha=0.25,
                    label="±1 SD", zorder=2)

    # ========================
    # Plot each dataset's reconstruction (mean ± SE)
    # ========================
    for i, (r, r_se) in enumerate(zip(recons, recons_se)):
        # ensure SE is paired correctly with its reconstruction
        r_vals = r.values
        r_se_vals = r_se.values
        if r_vals.shape != r_se_vals.shape:
            raise ValueError(f"Shape mismatch between reconstruction and SE for dataset {i}")
        
        ax.plot(time, r_vals, color="crimson", alpha=0.6, linewidth=.8,
                label="Reconstruction dataset mean" if i == 0 else None, zorder=2)
        ax.fill_between(time,
                        r_vals - r_se_vals,
                        r_vals + r_se_vals,
                        color="crimson", alpha=0.15, label="±1 SE" if i == 0 else None,
                        zorder=1)

    # --- Format x-axis ---
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    
def compute_ensemble_mean_sst(sst_dict):
    """
    Compute ensemble mean SST from multiple SST datasets with different time lengths.
    Uses only overlapping time periods across all datasets.
    
    Parameters:
    -----------
    sst_dict : dict
        Dictionary of SST datasets, e.g., {'sst1': xr.DataArray, 'sst2': xr.DataArray}
    
    Returns:
    --------
    ensemble_sst : xr.DataArray
        Ensemble mean SST over common time period
    """
    print("\nComputing ensemble mean SST...")
    
    # Get all datasets
    sst_datasets = list(sst_dict.values())
    
    # Find common time period across all datasets
    time_starts = [ds['time'].min().values for ds in sst_datasets]
    time_ends = [ds['time'].max().values for ds in sst_datasets]
    
    common_start = max(time_starts)
    common_end = min(time_ends)
    
    print(f"Common time period: {common_start} to {common_end}")
    
    # Subset each dataset to common time period
    sst_list = []
    for name, ds in sst_dict.items():
        ds_subset = ds.sel(time=slice(common_start, common_end))
        print(f"  {name}: {len(ds_subset['time'])} time steps")
        sst_list.append(ds_subset)
    
    # Compute ensemble mean
    ensemble_sst = xr.concat(sst_list, dim='ensemble').mean(dim='ensemble')
    
    print(f"Ensemble mean SST shape: {ensemble_sst.shape}")
    
    return ensemble_sst



def apply_fixdates_to_sst(results_dict):
    """Apply fixdates() to every DataArray inside the results dictionary."""
    new_dict = {}

    for key, sst in results_dict.items():
        # apply only to DataArrays with time coordinates
        if hasattr(sst, "time"):
            new_dict[key] = fixdates(sst)
        else:
            new_dict[key] = sst

    return new_dict