#!/usr/bin/env python
# coding: utf-8
"""
Figure 8: SST importance — normalized sensitivity and basin count maps.

Author: Flora Perlmutter

Description
-----------
3-panel global map figure showing where ocean SST matters for precipitation:
  Panel A: Min-max normalized total sensitivity
                     (|MS| × SST variability), basin-averaged then normalized
  Panel B: Min-max normalized marginal sensitivity magnitude
                     (|MS|), basin-averaged then normalized
  Panel C: Number of GRDC basins for which each ocean grid cell
                     has a significant marginal sensitivity

Required data files
-------------------------------
  global_linear_regression_bootstrap_{P}_{SST}.nc  (run_bootstrap_se_obs.py)
  sst_anom_{SST}.nc                                (load_and_process_obs.py)
  grdc_basins/

Output
------
  figures/paper_figures/Figure_8_SST_importance.png

"""

import warnings
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import DATA_DIR, PAPER_FIGURE_DIR
from plotting_functions import compute_ensemble_means, compute_ensemble_mean_sst

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR
BASINS_DIR  = DATA_DIR / "grdc_basins"

#Import and process Data

# Load basin boundaries
grdc_basins = gpd.read_file(BASINS_DIR)

# Load SST anomalies
sst_dict = {}
for sst_name in ["ERSSTv6", "COBE-SST3"]:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()

# ============================================================================
# LOAD ALL BOOTSTRAP RESULTS
# ============================================================================

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

SST_DATASETS = ['ERSSTv6', 'COBE-SST3']

# Create nested dictionary to store all results
linear_results = {}

print("Loading bootstrap results...")
for p_name in PRECIP_DATASETS.keys():
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f'global_linear_regression_bootstrap_{p_name}_{sst_name}.nc'
        
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
            
# ==============================================================================
# COMPUTE ENSEMBLE MEAN FOR OBS
# ==============================================================================

ensemble_obs = compute_ensemble_means(linear_results, "Observations")
sst_ensemble = compute_ensemble_mean_sst(sst_dict)

def minmax_norm(da, dim=None):
    da_min = da.min(dim=dim, skipna=True)
    da_max = da.max(dim=dim, skipna=True)
    return (da - da_min) / (da_max - da_min)

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


# -------------------------------
# Create figure with GridSpec
# -------------------------------
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(6.86, 1.35), dpi=600)
gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.25, hspace=0.25, figure=fig)


# -------------------------------
# Define colormaps and norms
# -------------------------------
# SST sensitivity (PuRd)
sst_vmin, sst_vmax = 0, 1
num_levels_sst = 20
sst_cmap = plt.get_cmap('PuRd', num_levels_sst)
sst_norm = BoundaryNorm(np.linspace(sst_vmin, sst_vmax, num_levels_sst + 1), sst_cmap.N)

# Ocean count (PuRd)
ocean_vmax = len(grdc_basins)
num_levels_ocean = 20
ocean_cmap = plt.get_cmap('PuRd', num_levels_ocean)
ocean_norm = BoundaryNorm(np.linspace(0, ocean_vmax, num_levels_ocean + 1), ocean_cmap.N)

# -------------------------------
# Panel A: Number of Basins with Significant Marginal Sensitivity
# -------------------------------
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax1.set_global()

im1 = ax1.pcolormesh(lons, lats, ocean_importance_count,
                     cmap=ocean_cmap, norm=ocean_norm,
                     transform=ccrs.PlateCarree())
ax1.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax1.coastlines(linewidth=.3, zorder=3)

# Colorbar for Panel A
sm1 = plt.cm.ScalarMappable(norm=ocean_norm, cmap=ocean_cmap)
sm1.set_array([])
cbar_ticks_1 = np.linspace(0, int(ocean_vmax), 5).astype(int)
cbar1 = fig.colorbar(
    sm1, 
    ax=ax1, 
    ticks=cbar_ticks_1, 
    orientation='horizontal',
    extend='neither',
    shrink=0.8,
    pad=0.06
)
ax1.set_title("SST Sensitivity")
cbar1.set_label('Number of River Basins', ) 
cbar1.ax.minorticks_off()

# -------------------------------
# Panel B: Normalized Marginal Sensitivity
# -------------------------------
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax2.set_global()

im2 = ax2.pcolormesh(lons, lats, ms_norm, transform=ccrs.PlateCarree(), 
                     cmap=sst_cmap, norm=sst_norm)
ax2.coastlines(linewidth=.3)
ax2.add_feature(cfeature.LAND, facecolor="white")

# Colorbar for Panel B
sm2 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap)
sm2.set_array([])
cbar_ticks_2 = [0, 0.25, 0.5, 0.75, 1]
cbar2 = fig.colorbar(
    sm2, 
    ax=ax2, 
    ticks=cbar_ticks_2, 
    orientation='horizontal', 
    extend='neither',
    shrink=0.8,
    pad=0.06
)
ax2.set_title("Marginal Sensitivity")
cbar2.ax.minorticks_off()

# -------------------------------
# Panel C: Normalized Total Sensitivity
# -------------------------------
ax3 = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson(central_longitude=180))
ax3.set_global()
im3 = ax3.pcolormesh(lons, lats, convolved_norm, transform=ccrs.PlateCarree(), 
                     cmap=sst_cmap, norm=sst_norm)
ax3.coastlines(linewidth=.3)
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
    shrink=0.8,
    pad=0.06
)
ax3.set_title("Total Sensitivity")
cbar3.ax.minorticks_off()



# -------------------------------
# Panel labels
# -------------------------------
for ax, label in zip([ax1, ax2, ax3], ['a', 'b', 'c']):
    ax.text(-0.05, 1.23, label, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top')
    ax.set_aspect('auto')

plt.savefig(FIGURES_DIR / "Figure_8_SST_importance.png", 
            dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()