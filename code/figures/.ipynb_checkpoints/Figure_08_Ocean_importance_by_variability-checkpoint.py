#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[5]:


##-------------------------------------------------------------------------------##

#Define directories

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Scripts')
data_dir = os.path.join(root_dir, 'Data/Observations')
amip_data_dir='/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/CAM4_Greens_Project/Data/Raw'
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
figures_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Figures/Paper_Figures')
        
##-------------------------------------------------------------------------------##

#Import functions

os.chdir(os.path.join(functions_dir))
from regression_functions import waterbasin, proccess_sst, detrend_dim, convert_to_mm_month, grid_area, fdr_correction, regression_slope_se
from plotting_functions import basin_name_for_id, basin_id_for_name, load_and_combine, apply_fixdates_to_results, convert_time_to_years, linear_trend, compute_ensemble_means, plot_ensemble_on_ax, compute_ensemble_mean_sst, apply_fixdates_to_sst

##-------------------------------------------------------------------------------##

#Import and process Data

# Load basin boundaries
grdc_basins = gpd.read_file(os.path.join(root_dir,'Data','Other', "grdc_basins"))

# Load SST anomalies
sst_dict = {}
for sst_name in ["ERSSTv5", "COBE-SST2"]:
    nc_path = os.path.join(outputs_dir, f"sst_anom_{sst_name}.nc")
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path)

# ============================================================================
# LOAD ALL BOOTSTRAP RESULTS
# ============================================================================

outputs_dir = '/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/Observational_Regressions_Project/Data/Processed'

# Define precipitation datasets
PRECIP_DATASETS = {
    'GPCP': None,
    'CRU': None,
    'GPCC': None,
    'CPC': None,
    'UDel': None,
    'PREC': None,
    'TerraClimate': None,
    'REGEN': None,
}

SST_DATASETS = ['ERSSTv5', 'COBE-SST2']

# Create nested dictionary to store all results
linear_results = {}

print("Loading bootstrap results...")
for p_name in PRECIP_DATASETS.keys():
    for sst_name in SST_DATASETS:
        output_file = os.path.join(
            outputs_dir,
            f'global_linear_regression_bootstrap_{p_name}_{sst_name}.nc'
        )
        
        if os.path.exists(output_file):
            try:
                result_ds = xr.open_dataset(output_file)
                
                # Reconstruct the result dictionary structure
                result = {
                    'model_id': result_ds.attrs['model_id'],
                    'description': result_ds.attrs['description'],
                    'variable': result_ds.attrs['variable'],
                    'alpha': result_ds.attrs['alpha'],
                    'n_bootstrap': result_ds.attrs['n_bootstrap'],
                    'reconstruction': result_ds['reconstruction'],
                    'reconstruction_se': result_ds['reconstruction_se'],
                    'observed_precip': result_ds['observed_precip'],
                    'correlation': result_ds['correlation'],
                    'marginal_sensitivity_se': result_ds['marginal_sensitivity_se'],
                    'marginal_sensitivity': result_ds['marginal_sensitivity'],
                }
                
                # Store with key (p_name, sst_name)
                linear_results[(p_name, sst_name)] = result
                
                print(f"Loaded: {p_name} vs {sst_name}")
                
            except Exception as e:
                print(f"Failed to load {p_name} vs {sst_name}: {e}")
        else:
            print(f"Not found: {output_file}")


# In[6]:


# ==============================================================================
# COMPUTE ENSEMBLE MEAN FOR OBS
# ==============================================================================

ensemble_obs = compute_ensemble_means(linear_results, "Observations")
sst_ensemble = compute_ensemble_mean_sst(sst_dict)


# In[ ]:


def minmax_norm(da, dim=None):
    da_min = da.min(dim=dim, skipna=True)
    da_max = da.max(dim=dim, skipna=True)
    return (da - da_min) / (da_max - da_min)


# In[65]:


sst_variability = sst_ensemble.std('time')

# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# --- Marginal sensitivity ---
ms = ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids)

# --- Convolved quantity |MS| * SST variability ---
convolved = abs(ms) * sst_variability
convolved_basin_mean = convolved.mean(dim='basin', skipna=True)

# Min–max normalization → [0, 1]
convolved_norm = minmax_norm(convolved_basin_mean)

# --- Importance count (unchanged, looks good) ---
important_mask = xr.where(ms.notnull(), 1, np.nan)
important_ssts = xr.where(ms.notnull(), 1, 0)

ocean_importance_count = important_ssts.sum(dim='basin', skipna=True)

all_nan_mask = important_mask.isnull().all(dim='basin')
ocean_importance_count = ocean_importance_count.where(~all_nan_mask)

# --- MS magnitude only ---
ms_abs = abs(ms)
ms_basinmean = ms_abs.mean(dim='basin', skipna=True)

# Min–max normalization → [0, 1]
ms_norm = minmax_norm(ms_basinmean)

lats = ms.lat.values
lons = ms.lon.values


# In[97]:


# -------------------------------
# Create figure with GridSpec
# -------------------------------
fig = plt.figure(figsize=(30, 6))
gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.2, hspace=0.25, figure=fig)
plt.rcParams.update({'font.size': 24})

# -------------------------------
# Define colormaps and norms
# -------------------------------
# SST sensitivity (PuRd)
sst_vmin, sst_vmax = 0, 1
num_levels_sst = 20
sst_cmap = plt.get_cmap('PuRd', num_levels_sst)
sst_norm = BoundaryNorm(np.linspace(sst_vmin, sst_vmax, num_levels_sst + 1), sst_cmap.N)

# Unnormalized sensitivity norm
sst_vmin_unnorm, sst_vmax_unnorm = 0, 2
sst_norm_unnorm = BoundaryNorm(np.linspace(sst_vmin_unnorm, sst_vmax_unnorm, num_levels_sst + 1), sst_cmap.N)

# Ocean count (PuRd)
ocean_vmax = len(grdc_basins)
num_levels_ocean = 20
ocean_cmap = plt.get_cmap('PuRd', num_levels_ocean)
ocean_norm = BoundaryNorm(np.linspace(0, ocean_vmax, num_levels_ocean + 1), ocean_cmap.N)

# -------------------------------
# Panel A: Normalized Total Sensitivity
# -------------------------------
ax1 = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson(central_longitude=180))
ax1.set_global()
im1 = ax1.pcolormesh(lons, lats, convolved_norm, transform=ccrs.PlateCarree(), 
                     cmap=sst_cmap, norm=sst_norm)
ax1.coastlines()
ax1.add_feature(cfeature.LAND, facecolor="white")
# Colorbar for Panel A
sm1 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap)
sm1.set_array([])
cbar_ticks_1 = [0, 0.25, 0.5, 0.75, 1]
cbar1 = fig.colorbar(
    sm1, 
    ax=ax1, 
    ticks=cbar_ticks_1, 
    orientation='horizontal', 
    extend='neither',
    shrink=0.7,
    pad=0.05
)
cbar1.set_label("Normalized Total Sensitivity", labelpad=2)
cbar1.ax.minorticks_off()

# -------------------------------
# Panel B: Normalized Marginal Sensitivity
# -------------------------------
ax3 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax3.set_global()

im3 = ax3.pcolormesh(lons, lats, ms_norm, transform=ccrs.PlateCarree(), 
                     cmap=sst_cmap, norm=sst_norm)
ax3.coastlines()
ax3.add_feature(cfeature.LAND, facecolor="white")

# Colorbar for Panel C
sm3 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap)
sm3.set_array([])
cbar_ticks_3 = [0, 0.25, 0.5, 0.75, 1]
cbar3 = fig.colorbar(
    sm3, 
    ax=ax3, 
    ticks=cbar_ticks_3, 
    orientation='horizontal', 
    extend='neither',
    shrink=0.7,
    pad=0.05
)
cbar3.set_label("Normalized Marginal Sensitivity", labelpad=2)
cbar3.ax.minorticks_off()

# -------------------------------
# Panel C: Number of Basins with Significant Marginal Sensitivity
# -------------------------------
ax4 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax4.set_global()

im4 = ax4.pcolormesh(lons, lats, ocean_importance_count,
                     cmap=ocean_cmap, norm=ocean_norm,
                     transform=ccrs.PlateCarree())
ax4.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax4.coastlines(zorder=3)

# Colorbar for Panel D
sm4 = plt.cm.ScalarMappable(norm=ocean_norm, cmap=ocean_cmap)
sm4.set_array([])
cbar_ticks_4 = np.linspace(0, int(ocean_vmax), 5).astype(int)
cbar4 = fig.colorbar(
    sm4, 
    ax=ax4, 
    ticks=cbar_ticks_4, 
    orientation='horizontal',
    extend='neither',
    shrink=0.7,
    pad=0.05
)
cbar4.set_label("Number of Basins Influenced by SST", labelpad=2)
cbar4.ax.minorticks_off()

# -------------------------------
# Panel labels
# -------------------------------
for ax, label in zip([ax4, ax3, ax1], ['a', 'b', 'c']):
    ax.text(-0.05, 1.08, label, transform=ax.transAxes, 
            fontsize=24, fontweight='bold', va='top')
    ax.set_aspect('auto')

plt.savefig(os.path.join(figures_dir, "Figure_8_SST_importance.png"), 
            dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.show()


# In[ ]:




