#!/usr/bin/env python
# coding: utf-8
"""
Figure 10: AMIP average marginal sensitivity and SST variability maps.

Author: Flora Perlmutter

Description
-----------
AMIP counterpart to Figure 7. Identical 4-panel layout using AMIP ensemble
quantities instead of observational ensemble quantities:
  Panel A: AMIP ensemble mean marginal sensitivity (dP/dSST), with
           significance hatching where the ensemble mean is not
           distinguishable from zero at p<0.05 (t-test, df=5) OR
           fewer than 75% of members agree on the sign of the trend.
  Panel B: AMIP convolved quantity: MS × SST variability, same hatching.
  Panel C: AMIP bootstrap SE of the marginal sensitivity
  Panel D: AMIP ensemble mean SST variability (std over time)

Required data files
-------------------------------
  global_linear_regression_bootstrap_amip_{M}_{M}.nc  (run_bootstrap_se_amip.py)
  sst_anom_{M}.nc                                     (load_and_process_amip.py)
  amip_dataset_names.json                             (load_and_process_amip.py)
  grdc_basins/

Output
------
  figures/paper_figures/Figure_10_average_sensitivity_importance_AMIP.png

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
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import stats
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import DATA_DIR, PAPER_FIGURE_DIR
from plotting_functions import (
    apply_fixdates_to_results,
    apply_fixdates_to_sst,
    compute_ensemble_mean_sst,
    compute_ensemble_means,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR
BASINS_DIR  = DATA_DIR / "grdc_basins"

##-------------------------------------------------------------------------------##

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
        amip_sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()


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

for key in amip_sst_dict:
    if "height" in amip_sst_dict[key].coords:
        amip_sst_dict[key] = amip_sst_dict[key].drop_vars("height")
        
# ==============================================================================
# COMPUTE ENSEMBLE MEAN FOR OBS
# ==============================================================================

amip_sst_ensemble = compute_ensemble_mean_sst(amip_sst_dict)
sst_ensemble = compute_ensemble_mean_sst(sst_dict)


# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# --- Marginal sensitivity ---
amip_ms = ensemble_amip['marginal_sensitivity_sst'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
amip_ms_se = ensemble_amip['marginal_sensitivity_sst_se'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
amip_ms_se = amip_ms_se.where(amip_ms_se != 0, np.nan)

amip_sst_variability = amip_sst_ensemble.std('time')

# --- Convolved quantity MS * SST variability ---
amip_convolved = amip_ms * amip_sst_variability

amip_lats = amip_ms.lat.values
amip_lons = amip_ms.lon.values

# -----------------------------------------------------------------------
# SIGNIFICANCE MASKING (AMIP: df = 5, so 6 members)
# -----------------------------------------------------------------------
N_MEMBERS_AMIP = len(linear_results_amip)   # should be 6
DF_AMIP        = N_MEMBERS_AMIP - 1         # 5

# --- Inter-member std from individual AMIP members ---
member_arrays = []
for key, result in linear_results_amip.items():
    if 'marginal_sensitivity' in result:
        m = result['marginal_sensitivity'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
        member_arrays.append(m)

members_stacked = xr.concat(member_arrays, dim='member')   # (member, lat, lon)

amip_ms_std      = members_stacked.std(dim='member')
inter_member_se  = amip_ms_std / np.sqrt(N_MEMBERS_AMIP)

# --- t-test: is ensemble mean distinguishable from zero? ---
t_stat = amip_ms / inter_member_se.where(inter_member_se != 0)

p_val = xr.apply_ufunc(
    lambda t, df: 2 * stats.t.sf(np.abs(t), df),
    t_stat,
    DF_AMIP,
    dask='parallelized',
    output_dtypes=[float],
)

not_significant = p_val >= 0.05   # True where NOT significant

# --- Sign agreement: fewer than 75% of members agree on sign ---
sign_agree     = (np.sign(members_stacked) == np.sign(amip_ms)).sum(dim='member') / N_MEMBERS_AMIP
poor_agreement = sign_agree < 0.75

# Combined mask: hatch where EITHER criterion is met
valid = np.isfinite(amip_ms)
hatch_mask_AB = (not_significant | poor_agreement) & valid


print(f"AMIP members used: {N_MEMBERS_AMIP}  (df={DF_AMIP})")
print(f"Fraction of grid cells hatched: {float(hatch_mask_AB.mean()):.2%}")

# -------------------------------
# Create figure with GridSpec
# -------------------------------
plt.rcParams.update({'font.size': 7})
plt.rcParams['hatch.linewidth'] = 0.4
fig = plt.figure(figsize=(6.95, 4.67), dpi=600)
gs = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.2, hspace=0.3, figure=fig)

# -------------------------------
# Define colormaps and norms
# -------------------------------
# SST Variance
sst_vmin, sst_vmax = 0, 2
num_levels_sst = 20
sst_cmap = plt.get_cmap('Blues', num_levels_sst+1)
colors = sst_cmap(np.arange(num_levels_sst + 1))
sst_cmap = ListedColormap(colors[:-1])
sst_cmap.set_over(colors[-1])
sst_norm = BoundaryNorm(np.linspace(sst_vmin, sst_vmax, num_levels_sst + 1), sst_cmap.N)

# MS
ms_vmin, ms_vmax = -.04, .04
ms_cmap = plt.get_cmap('RdBu', num_levels_sst+1)
colors = ms_cmap(np.arange(num_levels_sst + 1))
ms_cmap = ListedColormap(colors[:-1])
ms_cmap.set_over(colors[-1])
ms_norm = BoundaryNorm(np.linspace(ms_vmin, ms_vmax, num_levels_sst + 1), ms_cmap.N)

# SE
se_vmin, se_vmax = 0, .01
se_cmap = plt.get_cmap('Blues', num_levels_sst+1)
colors = se_cmap(np.arange(num_levels_sst + 1))
se_cmap = ListedColormap(colors[:-1])
se_cmap.set_over(colors[-1])
se_norm = BoundaryNorm(np.linspace(se_vmin, se_vmax, num_levels_sst + 1), se_cmap.N)

# Convolved
total_vmin, total_vmax = -.02, .02
total_cmap = plt.get_cmap('RdBu', num_levels_sst+1)
colors = total_cmap(np.arange(num_levels_sst + 1))
total_cmap = ListedColormap(colors[:-1])
total_cmap.set_over(colors[-1])
total_norm = BoundaryNorm(np.linspace(total_vmin, total_vmax, num_levels_sst + 1), total_cmap.N)

# =============================
# TOP ROW: Total Sensitivity
# =============================

# -------------------------------
# Panel A: Average Marginal Sensitivity
# -------------------------------
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax1.set_global()
im1 = ax1.pcolormesh(amip_lons, amip_lats, amip_ms, transform=ccrs.PlateCarree(),
                     cmap=ms_cmap, norm=ms_norm)
ax1.coastlines(linewidth=.3, zorder=6)
ax1.add_feature(cfeature.LAND, facecolor="white", zorder=5)

# --- significance hatching ---
ax1.contourf(amip_lons, amip_lats, hatch_mask_AB.values.astype(float),
             levels=[0.5, 1.5],
             hatches=['////////'],
             colors='none',
             transform=ccrs.PlateCarree(),
             zorder=3)

# Colorbar for Panel A
sm1 = plt.cm.ScalarMappable(norm=ms_norm, cmap=ms_cmap)
sm1.set_array([])
cbar_ticks_1 = np.round(np.linspace(ms_vmin, ms_vmax, 5), 2)
cbar1 = fig.colorbar(
    sm1,
    ax=ax1,
    ticks=cbar_ticks_1,
    orientation='horizontal',
    extend='max',
    shrink=0.8,
    pad=0.05
)
cbar1.formatter.set_powerlimits((-2, 2))
ax1.set_title("AMIP Average Marginal Sensitivity")
cbar1.set_label(r'mm month$^{-1}$ K$^{-1}$', labelpad=2)
cbar1.ax.minorticks_off()
cbar1.ax.xaxis.get_offset_text().set_x(1.1)


# -------------------------------
# Panel B: Total Sensitivity
# -------------------------------
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax2.set_global()
im2 = ax2.pcolormesh(amip_lons, amip_lats, amip_convolved,
                     cmap=total_cmap, norm=total_norm,
                     transform=ccrs.PlateCarree())
ax2.add_feature(cfeature.LAND, facecolor="white", zorder=5)
ax2.coastlines(zorder=6, linewidth=.3)

# --- significance hatching ---
ax2.contourf(amip_lons, amip_lats, hatch_mask_AB.values.astype(float),
             levels=[0.5, 1.5],
             hatches=['////////'],
             colors='none',
             transform=ccrs.PlateCarree(),
             zorder=4)

# Colorbar for Panel B
sm2 = plt.cm.ScalarMappable(norm=total_norm, cmap=total_cmap)
sm2.set_array([])
cbar_ticks_2 = np.round(np.linspace(total_vmin, total_vmax, 5), 2)
cbar2 = fig.colorbar(
    sm2,
    ax=ax2,
    ticks=cbar_ticks_2,
    orientation='horizontal',
    extend='max',
    shrink=0.8,
    pad=0.05
)
ax2.set_title("AMIP Average Total Sensitivity")
cbar2.formatter.set_powerlimits((-2, 2))
cbar2.set_label(r'mm month$^{-1}$', labelpad=2)
cbar2.ax.minorticks_off()
cbar2.ax.xaxis.get_offset_text().set_x(1.1)


# =============================
# BOTTOM ROW: SE and SST variability (no hatching)
# =============================

# -------------------------------
# Panel C: Average SE
# -------------------------------
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180))
ax3.set_global()
im3 = ax3.pcolormesh(amip_lons, amip_lats, amip_ms_se, transform=ccrs.PlateCarree(),
                     cmap=se_cmap, norm=se_norm)
ax3.coastlines(zorder=4, linewidth=.3)
ax3.add_feature(cfeature.LAND, facecolor="white", zorder=1)

# Colorbar for Panel C
sm3 = plt.cm.ScalarMappable(norm=se_norm, cmap=se_cmap)
sm3.set_array([])
cbar_ticks_3 = np.round(np.linspace(se_vmin, se_vmax, 5), 4)
cbar3 = fig.colorbar(
    sm3,
    ax=ax3,
    ticks=cbar_ticks_3,
    orientation='horizontal',
    extend='max',
    shrink=0.8,
    pad=0.05
)
ax3.set_title("AMIP Average MS Standard Error")
cbar3.set_label(r'mm month$^{-1}$ K$^{-1}$', labelpad=2)
cbar3.formatter.set_powerlimits((-3, -3))
cbar3.ax.minorticks_off()
cbar3.ax.xaxis.get_offset_text().set_x(1.1)

# -------------------------------
# Panel D: SST Variability
# -------------------------------
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180))
ax4.set_global()
im4 = ax4.pcolormesh(amip_lons, amip_lats, amip_sst_variability, transform=ccrs.PlateCarree(),
                     cmap=sst_cmap, norm=sst_norm)
ax4.coastlines(linewidth=.3)
ax4.add_feature(cfeature.LAND, facecolor="white")

# Colorbar for Panel D
sm4 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap)
sm4.set_array([])
cbar_ticks_4 = np.round(np.linspace(sst_vmin, sst_vmax, 5), 2)
cbar4 = fig.colorbar(
    sm4,
    ax=ax4,
    ticks=cbar_ticks_4,
    orientation='horizontal',
    extend='max',
    shrink=0.8,
    pad=0.05
)
ax4.set_title('AMIP Average SST Variability')
cbar4.set_label("K", labelpad=2)
cbar4.ax.minorticks_off()

# -------------------------------
# Panel labels
# -------------------------------
for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.text(0, 1.15, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_aspect('auto')

plt.savefig(FIGURES_DIR / "Figure_10_average_sensitivity_importance_AMIP.png",
            dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()