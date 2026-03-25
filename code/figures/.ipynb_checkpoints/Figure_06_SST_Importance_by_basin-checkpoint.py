#!/usr/bin/env python
# coding: utf-8

# In[15]:


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


# In[16]:


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
from plotting_functions import basin_name_for_id, basin_id_for_name, load_and_combine, apply_fixdates_to_results, convert_time_to_years, linear_trend, compute_ensemble_means, plot_ensemble_on_ax

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


# In[17]:


# ==============================================================================
# COMPUTE ENSEMBLE MEAN FOR OBS
# ==============================================================================

ensemble_obs = compute_ensemble_means(linear_results, "Observations")


# In[18]:


# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# Compute mean correlations
corr_mean = ensemble_obs['correlation_sst'].reindex(basin=basin_ids)

# Process MS data (for top panels)
ms_sq = ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids) ** 2
ms_sq_basinmean = ms_sq.mean(dim='basin', skipna=True)
ms_norm = ms_sq_basinmean / ms_sq_basinmean.max()

# Process importance data (for bottom panels)
important_mask = xr.where(ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids).notnull(), 1, np.nan)

important_ssts = xr.where(ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids).notnull(), 1, 0)

# Compute the sum, but keep NaN where all are NaN
ocean_importance_count = important_ssts.sum(dim='basin', skipna=True)

# Reapply mask so grid cells that were all NaN stay NaN
all_nan_mask = important_mask.isnull().all(dim='basin')
ocean_importance_count = ocean_importance_count.where(~all_nan_mask)

n_ocean_cells = np.isfinite(sst_dict['ERSSTv5'].mean('time')).sum(dim=('lat', 'lon'))
basin_importance_fraction = important_mask.sum(dim=('lat', 'lon')) / n_ocean_cells

# Merge correlation data with GRDC basins
metrics_df = gpd.GeoDataFrame({'correlation': corr_mean.values}, index=grdc_basins['MRBID'].values)
fraction_df = gpd.GeoDataFrame({'fraction_important': basin_importance_fraction.values},
                               index=grdc_basins['MRBID'].values)

grdc_basins = grdc_basins.copy()
try:
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(int)
    metrics_df.index = metrics_df.index.astype(int)
    fraction_df.index = fraction_df.index.astype(int)
except (ValueError, TypeError):
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(str)
    metrics_df.index = metrics_df.index.astype(str)
    fraction_df.index = fraction_df.index.astype(str)

gdf_corr = grdc_basins.merge(metrics_df, left_on="MRBID", right_index=True, how="left")
gdf_frac = grdc_basins.merge(fraction_df, left_on="MRBID", right_index=True, how="left")

lats = ms_sq.lat.values
lons = ms_sq.lon.values


# In[19]:


# ---------------------------------------
# Compute std ratio metric from xarrays
# ---------------------------------------

# Take std over time for each basin
sst_recon_std = ensemble_obs['sst_reconstruction'].std('time')
obs_precip_std = ensemble_obs['observed_precip'].std('time')

# Correct ratio: std(SST_recon) / std(precip_obs)
std_ratio = (sst_recon_std / obs_precip_std)*100

# ---------------------------------------
# Create GeoDataFrame with std ratio
# ---------------------------------------
std_ratio_df = gpd.GeoDataFrame(
    {'std_ratio': std_ratio.values},
    index=std_ratio['basin'].values
)

grdc_basins_copy = grdc_basins.copy()
try:
    grdc_basins_copy["MRBID"] = grdc_basins_copy["MRBID"].astype(int)
    std_ratio_df.index = std_ratio_df.index.astype(int)
except (ValueError, TypeError):
    grdc_basins_copy["MRBID"] = grdc_basins_copy["MRBID"].astype(str)
    std_ratio_df.index = std_ratio_df.index.astype(str)

gdf_std_ratio = grdc_basins_copy.merge(
    std_ratio_df, left_on="MRBID", right_index=True, how="left"
)


# In[22]:


print(std_ratio.sel(basin=3203.))


# In[28]:


print(std_ratio.max())


# In[29]:


print(std_ratio.min())


# In[26]:


print(corr_mean.sel(basin=3203.))


# In[24]:


print(corr_mean.max())


# In[27]:


print(corr_mean.min())


# In[ ]:


# --------------------------------------------
# Create 2x2 figure with maps and histograms
# --------------------------------------------
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1.2, 1], hspace=0.3, wspace=0.25)
plt.rcParams.update({'font.size': 18})

# ---------------------------------------
# Panel A: Correlation Map
# ---------------------------------------
ax_map = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))

# Define colormap and normalization
corr_vmin, corr_vmax = 0, 60
corr_cmap = plt.get_cmap('Blues', 20)
corr_norm = BoundaryNorm(np.linspace(corr_vmin, corr_vmax, 21), corr_cmap.N)

# Plot background ocean
ax_map.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.3)

# Significant basins
sig_gdf = gdf_corr.copy()
sig_gdf['corr_plot'] = corr_mean.values*100
sig_gdf = sig_gdf[sig_gdf['corr_plot'] > 0]

for idx, row in sig_gdf.iterrows():
    ax_map.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                          facecolor=corr_cmap(corr_norm(row.corr_plot)),
                          edgecolor='black', linewidth=0.3, alpha=0.9)

# Insignificant or missing
insig_gdf = gdf_corr[(gdf_corr["correlation"] <= 0) | (gdf_corr["correlation"].isna())]
if not all(insig_gdf.is_empty):
    insig_gdf.plot(ax=ax_map, facecolor='none', hatch='///', transform=ccrs.PlateCarree(),
                   linewidth=0.3, edgecolor="black")

ax_map.coastlines()
ax_map.add_feature(cfeature.LAND, facecolor="white")
ax_map.set_aspect('auto')

# Add colorbar
sm_corr = plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
sm_corr.set_array([])
cbar_map = fig.colorbar(sm_corr, ax=ax_map, ticks=[0, 15, 30, 45, 60], 
                        orientation='horizontal', shrink=.7, pad=0.05)
cbar_map.set_label("Average SST Forced Precipitation Variability (%)", fontsize=14)
cbar_map.ax.minorticks_off()

# Add panel label
ax_map.text(-0.05, 1.08, 'a', transform=ax_map.transAxes, 
            fontsize=16, fontweight='bold', va='top')

# ---------------------------------------
# Panel B: Correlation Histogram
# ---------------------------------------
ax_hist = plt.subplot(gs[0, 1])

corr_values = gdf_corr["correlation"].dropna().values*100
# Match bins to colorbar ticks: [0, 15, 30, 45, 60]
hist_bins = np.linspace(0, 60, 21)  # Creates bins that align with 15% intervals
n, bins, patches = ax_hist.hist(corr_values, bins=hist_bins, color="lightblue", 
                                 edgecolor="black", alpha=0.8)
ax_hist.set_xlabel("Average SST Forced Precipitation Variability (%)", fontsize=14)
ax_hist.set_ylabel("Number of Basins")
# Set x-axis ticks to match colorbar
ax_hist.set_xticks([0, 15, 30, 45, 60])

# Highlight basins
highlight_basins = ["MURRAY",  "YELLOW RIVER", 
                    "AMAZON (also AMAZONAS)", "MISSISSIPPI", "CONNECTICUT", "ZARUMILLA"]

for basin_name in highlight_basins:
    

    
    basin_id = float(basin_id_for_name(basin_name))
    if basin_id is None:
        continue
        
    basin_name = basin_name_for_id(basin_id)
    row = gdf_corr.loc[gdf_corr['MRBID'] == basin_id]
    if row.empty:
        continue
    val = row['correlation'].values[0]*100
    bin_idx = np.clip(np.digitize(val, bins) - 1, 0, len(bins)-2)
    bin_center = 0.5 * (bins[bin_idx] + bins[bin_idx+1])
    bin_height = n[bin_idx]
    
    text_x = bin_center + 0.0015
    ha_text = "left"
    text_y = bin_height + 8
    
    # Special positioning for Connecticut
    if basin_name.upper() == "ZARUMILLA":
        text_x = bin_center + 0.04 
        ha_text = "left"
    elif basin_name.upper() == "AMAZON ":
        text_x = bin_center + 0.04  
        text_y = bin_height + 14  
        ha_text = "left"
    elif basin_name.upper() == "MURRAY":
        text_y = bin_height + 12  
    elif basin_name.upper() == "YELLOW RIVER":
        text_y = bin_height + 14
        ha_text = "left"
    elif basin_name.upper() == "CONNECTICUT":
        text_y = bin_height + 14  
        ha_text = "right"
    else:
        text_x = bin_center + 0.0015
        ha_text = "left"
    
    plt.annotate(
        basin_name,
        xy=(bin_center, bin_height),
        xytext=(text_x, text_y),
        ha=ha_text,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
    )

ax_hist.set_ylim(0, 120)
ax_hist.set_xlim(0, 65)

# Add panel label
ax_hist.text(-0.17, 1.08, 'b', transform=ax_hist.transAxes, 
             fontsize=16, fontweight='bold', va='top', ha='left')


# ---------------------------------------
# Panel C: Std Ratio Map
# ---------------------------------------
ax_map_2 = plt.subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=0))

# Define colormap and normalization
ratio_vmin, ratio_vmax = 0, 60
ratio_cmap = plt.get_cmap('Blues', 21)
ratio_norm = BoundaryNorm(np.linspace(ratio_vmin, ratio_vmax, 21), ratio_cmap.N)

# Plot background ocean
ax_map_2.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.3)

# Plot basins with valid std ratio values
valid_gdf = gdf_std_ratio[gdf_std_ratio['std_ratio'].notna()]

for idx, row in valid_gdf.iterrows():
    ratio = np.clip(row.std_ratio, ratio_vmin, ratio_vmax)
    ax_map_2.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                          facecolor=ratio_cmap(ratio_norm(ratio)),
                          edgecolor='black', linewidth=0.3, alpha=0.9)

# Plot basins with missing values (hatched)
invalid_gdf = gdf_std_ratio[gdf_std_ratio['std_ratio'].isna()]
if not all(invalid_gdf.is_empty):
    invalid_gdf.plot(ax=ax_map_2, facecolor='none', hatch='///', transform=ccrs.PlateCarree(),
                     linewidth=0.3, edgecolor="black")

ax_map_2.set_global()
ax_map_2.coastlines()
ax_map_2.add_feature(cfeature.LAND, facecolor="white")
ax_map_2.set_aspect('auto')

# Add colorbar
sm_ratio = plt.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
sm_ratio.set_array([])
cbar_map_2 = fig.colorbar(sm_ratio, ax=ax_map_2, ticks=np.linspace(0, 60, 5), 
                        orientation='horizontal', shrink=0.7, pad=0.05)
cbar_map_2.set_label("Average SST Forced Precipitation Magnitude (%)", fontsize=14)
cbar_map_2.ax.minorticks_off()

# Add panel label
ax_map_2.text(-0.05, 1.08, 'c', transform=ax_map_2.transAxes, 
            fontsize=16, fontweight='bold', va='top')

# ---------------------------------------
# Panel D: Std Ratio Histogram
# ---------------------------------------
ax_hist_2 = plt.subplot(gs[1, 1])

ratio_values_clean = gdf_std_ratio["std_ratio"].dropna().values
ratio_values_filtered = ratio_values_clean[(ratio_values_clean >= 0) & (ratio_values_clean <= 200)]

# Match bins to colorbar ticks: [0, 15, 30, 45, 60]
hist_bins_2 = np.linspace(0, 60, 21)  # Creates bins that align with 15% intervals
n, bins, patches = ax_hist_2.hist(
    ratio_values_filtered, bins=hist_bins_2, color="lightblue", edgecolor="black", alpha=0.8
)
ax_hist_2.set_xlabel("Average SST Forced Precipitation Magnitude (%)", fontsize=14)
ax_hist_2.set_ylabel("Number of Basins")
# Set x-axis ticks to match colorbar
ax_hist_2.set_xticks([0, 15, 30, 45, 60])

# Highlight basins
highlight_basins_2 = ["MURRAY",  "YELLOW RIVER", 
                    "AMAZON (also AMAZONAS)", "MISSISSIPPI", "CONNECTICUT", "ZARUMILLA"]

for basin_name in highlight_basins_2:
    bid = float(basin_id_for_name(basin_name))
    if bid is None:
        continue
    canonical_name = basin_name_for_id(bid)
    
    row = gdf_std_ratio.loc[gdf_std_ratio['MRBID'] == bid]
    if row.empty:
        print(f"No std ratio value found for basin {canonical_name}")
        continue
    
    val = row['std_ratio'].values[0]
    if not np.isfinite(val) or val < 0 or val > 160:
        continue
    
    bin_idx = np.clip(np.digitize(val, bins) - 1, 0, len(bins)-2)
    bin_center = 0.5 * (bins[bin_idx] + bins[bin_idx+1])
    bin_height = n[bin_idx]
    
    # Special positioning (similar to first histogram)
    if canonical_name.upper() in ['YELLOW RIVER', 'GANGES','MISSISSIPPI']:
        text_x = bin_center  + 0.018
        ha_text = "left"
        text_y = 13
    # Special positioning (similar to first histogram)
    elif canonical_name.upper() in ['AMAZON ']:
        text_x = bin_center  + 0.018
        ha_text = "left"
        text_y = 15
    else:
        text_x = bin_center + 0.015
        ha_text = "left"
        text_y = 0
    
    ax_hist_2.annotate(
        canonical_name,
        xy=(bin_center, bin_height),
        xytext=(text_x, bin_height+text_y + 12),
        ha=ha_text,
        arrowprops=dict(arrowstyle="->", lw=1.2, color="black"),
        fontsize=11,
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
    )
    
# Add panel label
ax_hist_2.text(-0.17, 1.08, 'd', transform=ax_hist_2.transAxes, 
             fontsize=16, fontweight='bold', va='top', ha='left')

ax_hist_2.set_ylim(0, 280)
ax_hist_2.set_xlim(0, 65)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "Figure_6_sst_importance_by_basin.png"),
            dpi=300, bbox_inches="tight", pad_inches = .1)
plt.show()


# In[ ]:




