#!/usr/bin/env python
# coding: utf-8
"""
Figure 12: AMIP vs observed marginal sensitivity pattern correlation.

Author: Flora Perlmutter

Description
-----------
2-panel figure comparing AMIP and observed marginal sensitivity spatial
patterns across GRDC basins:
  Panel A: Basin choropleth map colored by pattern correlation between
           AMIP and observed marginal sensitivity (RdYlGn)
  Panel B: Histogram of per-basin pattern correlations with selected
           basins annotated

Pattern correlation is computed per basin: AMIP MS spatial field vs
observed MS spatial field (obs interpolated to AMIP grid).

Required data files
-------------------------------
  global_linear_regression_bootstrap_{P}_{SST}.nc     (run_bootstrap_se_obs.py)
  global_linear_regression_bootstrap_amip_{M}_{M}.nc  (run_bootstrap_se_amip.py)
  amip_dataset_names.json                             (load_and_process_amip.py)
  grdc_basins/

Output
------
  figures/paper_figures/Figure_12_amip_vs_obs_marginal_sensitivity.png
"""

import json
import warnings
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import DATA_DIR, PAPER_FIGURE_DIR
from plotting_functions import (
    apply_fixdates_to_results,
    compute_ensemble_means,basin_id_for_name,basin_name_for_id 
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR
BASINS_DIR  = DATA_DIR / "grdc_basins"

# Load basin boundaries
grdc_basins = gpd.read_file(BASINS_DIR)

# Load SST anomalies
sst_dict = {}
for sst_name in ["ERSSTv6", "COBE-SST3"]:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()

# ============================================================================
# LOAD ALL OBSERVED BOOTSTRAP RESULTS
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
        output_file = OUTPUTS_DIR /  f'global_linear_regression_bootstrap_{p_name}_{sst_name}.nc'
        
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
            
# =============================================================================
# LOAD ALL AMIP BOOTSTRAP RESULTS
# =============================================================================
# Load AMIP dataset names manifest
names_path = OUTPUTS_DIR / "amip_dataset_names.json"
with open(names_path, "r") as f:
    names = json.load(f)
    
# Load SST anomalies
amip_sst_dict = {}
for sst_name in names['sst_datasets']:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        amip_sst_dict[sst_name] = xr.open_dataarray(nc_path)


AMIP_MODELS = names['precip_datasets']  # same list used for SSTs

# Container for results
linear_results_amip = {}

print("Loading AMIP bootstrap results...")

for model_id in AMIP_MODELS:
    output_file = OUTPUTS_DIR / f'global_linear_regression_bootstrap_amip_{model_id}_{model_id}.nc'

    if not os.path.exists(output_file):
        print(f"Not found: {output_file}")
        continue

    try:
        ds = xr.open_dataset(output_file)

        result = {
            'model_id': ds.attrs.get('model_id', 'P ~ β*SST (AMIP)'),
            'description': ds.attrs.get('description'),
            'variable': ds.attrs.get('variable'),
            'alpha': ds.attrs.get('alpha'),
            'n_bootstrap': ds.attrs.get('n_bootstrap'),
            'reconstruction': ds['reconstruction'],
            'reconstruction_se': ds['reconstruction_se'],
            'observed_precip': ds['observed_precip'],
            'correlation': ds['correlation'],
            'marginal_sensitivity_se': ds['marginal_sensitivity_se'],
            'marginal_sensitivity': ds['marginal_sensitivity'],
        }

        # Store using (precip_model, sst_model) key for symmetry
        linear_results_amip[(model_id, model_id)] = result

        print(f"Loaded AMIP: {model_id}")

    except Exception as e:
        print(f"Failed to load AMIP {model_id}: {e}")
        
# ==============================================================================
# COMPUTE ENSEMBLE MEANS FOR BOTH AMIP AND OBS
# ==============================================================================
ensemble_obs = compute_ensemble_means(linear_results, "Observations")
amip_results_fixed = apply_fixdates_to_results(linear_results_amip)
ensemble_amip = compute_ensemble_means(amip_results_fixed, "AMIP")

def calculate_pattern_metrics(amip_data, obs_data):
    """
    Calculate pattern correlation and RMSE between AMIP and observations.
    
    Parameters:
    -----------
    amip_data : xarray.DataArray or numpy array
        AMIP spatial field
    obs_data : xarray.DataArray or numpy array
        Observed spatial field (regridded to match AMIP)
    
    Returns:
    --------
    corr : float
        Pattern correlation coefficient
    rmse : float
        Root mean square error
    """
    # Flatten and remove NaN values
    amip_flat = np.asarray(amip_data).flatten()
    obs_flat = np.asarray(obs_data).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(amip_flat) | np.isnan(obs_flat))
    amip_clean = amip_flat[mask]
    obs_clean = obs_flat[mask]
    
    if len(amip_clean) == 0:
        return np.nan, np.nan
    
    # Pattern correlation (normalized)
    corr = np.corrcoef(amip_clean, obs_clean)[0, 1]
    
    # RMSE
    rmse = np.sqrt(np.mean((amip_clean - obs_clean)**2))
    
    return corr, rmse

# =================================================================
# Calculate Pattern Correlation and RMSE for all basins
# =================================================================

ms_name = 'marginal_sensitivity_sst'
# Extract unique basin IDs from your basin data
basin_ids = grdc_basins['MRBID'].values
n_basins = len(basin_ids)

# Initialize arrays to store metrics
correlations = np.full(n_basins, np.nan)
rmses = np.full(n_basins, np.nan)
basin_names = []

# Calculate metrics for each basin
for idx, basin_id in enumerate(basin_ids):
    try:
        # Extract AMIP and observations for this basin
        amip_mean = ensemble_amip[ms_name].sel(basin=basin_id)
        obs_mean = ensemble_obs[ms_name].sel(basin=basin_id)
        
        # Regrid obs to amip grid
        obs_mean_rg = obs_mean.interp(lat=amip_mean.lat, lon=amip_mean.lon)
        
        # Calculate metrics
        corr, rmse = calculate_pattern_metrics(amip_mean, obs_mean_rg)
        correlations[idx] = corr
        rmses[idx] = rmse
        basin_names.append(str(int(basin_id)))
        
    except Exception as e:
        print(f"Error processing basin {basin_id}: {e}")
        continue

# Create a DataFrame for easier plotting
metrics_df = pd.DataFrame({
    'basin_id': basin_ids,
    'correlation': correlations,
    'rmse': rmses,
    'basin_name': basin_names
})

# Remove NaN entries
metrics_df = metrics_df.dropna()

print(f"Processed {len(metrics_df)} basins")
print(f"Mean correlation: {metrics_df['correlation'].mean():.3f}")
print(f"Mean RMSE: {metrics_df['rmse'].mean():.3f}")

# =================================================================
# Map basins colored by pattern correlation
# =================================================================

# Create GeoDataFrame with metrics
metrics_gdf = grdc_basins.copy()
metrics_gdf['correlation'] = metrics_gdf['MRBID'].map(
    dict(zip(metrics_df['basin_id'], metrics_df['correlation']))
)
metrics_gdf['rmse'] = metrics_gdf['MRBID'].map(
    dict(zip(metrics_df['basin_id'], metrics_df['rmse']))
)

plt.rcParams.update({'font.size': 7})

# Create figure with 2 maps
fig = plt.figure(figsize=(6.25, 2.1), dpi=600)
gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0.3, wspace=0.35, width_ratios=[1.2, 1])

# --- Map 1: Colored by correlation ---
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))
ax1.set_title('AMIP vs Observed Marginal Sensitivity')
ax1.set_global()
num_levels = 20
vmax=1
vmin=-1

corr_cmap = plt.get_cmap("RdYlGn", num_levels)
corr_norm = BoundaryNorm(np.linspace(vmin, vmax, num_levels + 1), corr_cmap.N)

for _, row in metrics_gdf.iterrows():
    corr_val = row['correlation']
    if np.isfinite(corr_val):
        color = corr_cmap(corr_norm(corr_val))
    else:
        color = 'lightgray'
    ax1.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                       facecolor=color, edgecolor='black', linewidth=0.3)

ax1.add_feature(cfeature.LAND, facecolor='white', zorder=0)
ax1.coastlines(zorder=3, linewidth=.3)
sm1 = plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
sm1.set_array([])

cb1 = fig.colorbar(sm1, ax=ax1, orientation='horizontal', 
                   ticks=np.linspace(-1, 1, 5), shrink=0.7, pad=0.05)
cb1.set_label('Pattern Correlation')
cb1.ax.minorticks_off()

# Add panel label
ax1.text(-0.15, 1.17, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top')

ax_hist_2 = plt.subplot(gs[0, 1])

ratio_values_clean = metrics_gdf["correlation"].dropna().values

# Match bins to colorbar ticks
hist_bins_2 = np.linspace(-1, 1, 21)  # Creates bins that align with 15% intervals
n, bins, patches = ax_hist_2.hist(
    ratio_values_clean, bins=hist_bins_2, linewidth=.5, color="lightblue", edgecolor="black", alpha=0.8
)
ax_hist_2.set_xlabel("Average AMIP-Observed MS Correlation")
ax_hist_2.set_ylabel("Number of Basins")
# Set x-axis ticks to match colorbar
ax_hist_2.set_xticks([-1, -.5, 0, .5, 1])

# Highlight basins
highlight_basins_2 = ["MURRAY",  "YELLOW RIVER", 
                    "AMAZON (also AMAZONAS)", "MISSISSIPPI", "CONNECTICUT", "ZARUMILLA"]

for basin_name in highlight_basins_2:
    bid = float(basin_id_for_name(basin_name))
    if bid is None:
        continue
    canonical_name = basin_name_for_id(bid)
    
    row = metrics_gdf.loc[metrics_gdf['MRBID'] == bid]
    if row.empty:
        print(f"No std ratio value found for basin {canonical_name}")
        continue
    
    val = row['correlation'].values[0]
    bin_idx = np.clip(np.digitize(val, bins) - 1, 0, len(bins)-2)
    bin_center = 0.5 * (bins[bin_idx] + bins[bin_idx+1])
    bin_height = n[bin_idx]
    
    # Special positioning (similar to first histogram)
    if canonical_name.upper() in ['YELLOW RIVER', 'GANGES','MISSISSIPPI']:
        text_x = bin_center  + 0.018
        ha_text = "right"
        text_y = 9
    # Special positioning (similar to first histogram)
    elif canonical_name.upper() in ['AMAZON']:
        text_x = bin_center  + 0.015
        ha_text = "right"
        text_y = 27
        canonical_name='Amazon'
    # Special positioning (similar to first histogram)
    elif canonical_name.upper() in ['CONNECTICUT']:
        text_x = bin_center  + 0.015
        ha_text = "right"
        text_y = 27
    # Special positioning (similar to first histogram)
    elif canonical_name.upper() in ['MURRAY']:
        text_x = bin_center   + 0.015
        ha_text = "right"
        text_y = 18
        
    else:
        text_x = bin_center + 0.015
        ha_text = "right"
        text_y = 0
    if canonical_name.upper() in ['AMAZON','MISSISSIPPI', 'MURRAY']:
        ax_hist_2.annotate(
            canonical_name,
            xy=(bin_center, bin_height),
            xytext=(text_x, bin_height + text_y + 14),
            ha=ha_text,
            fontsize=5,
            #arrowprops=dict(arrowstyle="->", lw=.8, color="black", zorder=1),
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2", zorder=1),
            zorder=15,
            clip_on=False
    )
    else:
            ax_hist_2.annotate(
            canonical_name,
            xy=(bin_center, bin_height),
            xytext=(text_x, bin_height + text_y + 14),
            ha=ha_text,
            fontsize=5,
            arrowprops=dict(arrowstyle="->", lw=.8, color="black", zorder=0),
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2"),
            zorder=10,
            clip_on=False
    )
    
    
# Add panel label
ax_hist_2.text(-0.26, 1.04, 'b', transform=ax_hist_2.transAxes, 
             fontsize=10, fontweight='bold', va='top', ha='left')

ax_hist_2.set_ylim(0, 175)
ax_hist_2.set_xlim(-1, 1)

plt.savefig(FIGURES_DIR / f"Figure_12_amip_vs_obs_marginal_sensitivity.png", 
    dpi=600, bbox_inches='tight'
)
plt.show()