#!/usr/bin/env python
# coding: utf-8
"""
Figure 8: Average marginal sensitivity and SST variability maps.

Author: Flora Perlmutter

Description
-----------
4-panel global map figure:
  Panel A (top-left):     Ensemble mean marginal sensitivity (dP/dSST)
                          basin-averaged across all GRDC basins
  Panel B (top-right):    Convolved quantity: mean(|MS|) × SST variability
                          ("total sensitivity")
  Panel C (bottom-left):  Bootstrap SE of the marginal sensitivity
  Panel D (bottom-right): Ensemble mean SST variability (std over time)

Required data files
-------------------------------
  global_linear_regression_bootstrap_{P}_{SST}.nc  (run_bootstrap_se_obs.py)
  sst_anom_{SST}.nc                                (load_and_process_obs.py)

Output
------
  figures/paper_figures/Figure_08_average_sensitivity_importance.png

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
from scipy import stats
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

sst_variability = sst_ensemble.std('time')

# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# --- Marginal sensitivity ---
ms = ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
#ms_std = ensemble_obs['marginal_sensitivity_sst_std'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
ms_se = ensemble_obs['marginal_sensitivity_sst_se'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
ms_se = ms_se.where(ms_se != 0, np.nan)

# --- Convolved quantity |MS| * SST variability ---
convolved = ms * sst_variability

lats = ms.lat.values
lons = ms.lon.values

# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# --- Marginal sensitivity ---
ms = ensemble_obs['marginal_sensitivity_sst'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
ms_se = ensemble_obs['marginal_sensitivity_sst_se'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
ms_se = ms_se.where(ms_se != 0, np.nan)

# --- Convolved quantity |MS| * SST variability ---
convolved = ms * sst_variability

# -----------------------------------------------------------------------
# SIGNIFICANCE MASKING (Observations: df = 15)
# -----------------------------------------------------------------------


N_MEMBERS_OBS = 16   # df = 15  →  8 members
DF_OBS        = N_MEMBERS_OBS - 1   # 15

# --- t-test: is ensemble mean distinguishable from zero? ---
# SE here is the bootstrap SE averaged across members; we need the
# inter-member SE = ensemble_std / sqrt(n_members).
# Pull the per-member std that compute_ensemble_means stores:
ms_std = ensemble_obs['marginal_sensitivity_sst_std'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)

inter_member_se = ms_std / np.sqrt(N_MEMBERS_OBS)
t_stat          = ms / inter_member_se.where(inter_member_se != 0)

# two-tailed p-value
p_val = xr.apply_ufunc(
    lambda t, df: 2 * stats.t.sf(np.abs(t), df),
    t_stat,
    DF_OBS,
    dask='parallelized',
    output_dtypes=[float],
)

not_significant = p_val >= 0.05   # True where NOT significant

# --- Sign agreement: fraction of members agreeing on sign of ensemble mean ---
# Stack individual member MS fields, averaged over basins the same way
member_arrays = []
for key, result in linear_results.items():
    if 'marginal_sensitivity' in result:
        m = result['marginal_sensitivity'].reindex(basin=basin_ids).mean(dim='basin', skipna=True)
        member_arrays.append(m)

members_stacked = xr.concat(member_arrays, dim='member')  # (member, lat, lon)

N_MEMBERS_OBS = len(member_arrays)   # 16 (8 precip × 2 SST)
DF_OBS        = N_MEMBERS_OBS - 1

sign_agree   = (np.sign(members_stacked) == np.sign(ms)).sum(dim='member') / N_MEMBERS_OBS
poor_agreement = sign_agree < 0.75


# Combined mask: hatch where EITHER criterion is met
valid = np.isfinite(ms) & np.isfinite(t_stat)
hatch_mask_A = (not_significant | poor_agreement) & valid          # based on ms
hatch_mask_B = (not_significant | poor_agreement) & np.isfinite(convolved)  # based on convolved

lats = ms.lat.values
lons = ms.lon.values


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
sst_cmap = plt.get_cmap('Blues', num_levels_sst)
sst_norm = BoundaryNorm(np.linspace(sst_vmin, sst_vmax, num_levels_sst + 1), sst_cmap.N)

# MS
ms_vmin, ms_vmax = -.04, .04
num_levels_sst = 20
ms_cmap = plt.get_cmap('RdBu', num_levels_sst)
ms_norm = BoundaryNorm(np.linspace(ms_vmin, ms_vmax, num_levels_sst + 1), ms_cmap.N)

# SE
se_vmin, se_vmax = 0, .01
se_cmap = plt.get_cmap('Blues', num_levels_sst)
se_norm = BoundaryNorm(np.linspace(se_vmin, se_vmax, num_levels_sst + 1), se_cmap.N)

# Convolved
total_vmin, total_vmax = -.04, .04
total_cmap = plt.get_cmap('RdBu', num_levels_sst)
total_norm = BoundaryNorm(np.linspace(total_vmin, total_vmax, num_levels_sst + 1), total_cmap.N)

# =============================
# TOP ROW: Total Sensitivity
# =============================

# -------------------------------
# Panel A: Average Marginal Sensitivity
# -------------------------------
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax1.set_global()
im1 = ax1.pcolormesh(lons, lats, ms, transform=ccrs.PlateCarree(),
                     cmap=ms_cmap, norm=ms_norm)
ax1.coastlines(linewidth=.3, zorder=6)
ax1.add_feature(cfeature.LAND, facecolor="white", zorder=5)

# --- significance hatching ---
ax1.contourf(lons, lats, hatch_mask_A.values.astype(float),
             levels=[0.5, 1.5],
             hatches=['////////'],
             colors='none',
             transform=ccrs.PlateCarree(),
             zorder=3)

# Colorbar
sm1 = plt.cm.ScalarMappable(norm=ms_norm, cmap=ms_cmap)
sm1.set_array([])
cbar_ticks_1 = np.round(np.linspace(ms_vmin, ms_vmax, 5), 2)
cbar1 = fig.colorbar(sm1, ax=ax1, ticks=cbar_ticks_1,
                     orientation='horizontal', extend='neither',
                     shrink=0.8, pad=0.05)
cbar1.formatter.set_powerlimits((-2, 2))
ax1.set_title("Average Marginal Sensitivity")
cbar1.set_label(r'mm month$^{-1}$ K$^{-1}$', labelpad=2)
cbar1.ax.minorticks_off()
cbar1.ax.xaxis.get_offset_text().set_x(1.1)


# -------------------------------
# Panel B: Total Sensitivity
# -------------------------------
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax2.set_global()
im2 = ax2.pcolormesh(lons, lats, convolved,
                     cmap=total_cmap, norm=total_norm,
                     transform=ccrs.PlateCarree())
ax2.coastlines(zorder=6, linewidth=.3)
ax2.add_feature(cfeature.LAND, facecolor="white", zorder=5)

# --- significance hatching ---
ax2.contourf(lons, lats, hatch_mask_B.values.astype(float),
             levels=[0.5, 1.5],
             hatches=['////////'],
             colors='none',
             transform=ccrs.PlateCarree(),
             zorder=4)

# Colorbar
sm2 = plt.cm.ScalarMappable(norm=total_norm, cmap=total_cmap)
sm2.set_array([])
cbar_ticks_2 = np.round(np.linspace(total_vmin, total_vmax, 5), 2)
cbar2 = fig.colorbar(sm2, ax=ax2, ticks=cbar_ticks_2,
                     orientation='horizontal', extend='neither',
                     shrink=0.8, pad=0.05)
ax2.set_title("Average Total Sensitivity")
cbar2.formatter.set_powerlimits((-2, 2))
cbar2.set_label(r'mm month$^{-1}$', labelpad=2)
cbar2.ax.minorticks_off()
cbar2.ax.xaxis.get_offset_text().set_x(1.1)


# =============================
# BOTTOM ROW: Marginal Sensitivity
# =============================

# -------------------------------
# Panel C: Average SE
# -------------------------------
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=180))
ax3.set_global()
im3 = ax3.pcolormesh(lons, lats, ms_se, transform=ccrs.PlateCarree(), 
                     cmap=se_cmap, norm=se_norm)

ax3.coastlines(zorder=4, linewidth=.3)
ax3.add_feature(cfeature.LAND, facecolor="white", zorder=1)
# Colorbar for Panel C
sm3 = plt.cm.ScalarMappable(norm=se_norm, cmap=se_cmap)
sm3.set_array([])
cbar_ticks_3 = np.linspace(se_vmin, se_vmax, 5)
cbar_ticks_3 = np.round(cbar_ticks_3, 4)
cbar3 = fig.colorbar(
    sm3, 
    ax=ax3, 
    ticks=cbar_ticks_3, 
    orientation='horizontal', 
    extend='neither',
    shrink=0.8,
    pad=0.05
)
# |MS| × SST Variability
ax3.set_title("Average MS Standard Error")
cbar3.set_label(r'mm month$^{-1}$ K$^{-1}$', labelpad=2)
cbar3.formatter.set_powerlimits((-3, -3))
cbar3.ax.minorticks_off()
offset_text = cbar3.ax.xaxis.get_offset_text()
offset_text.set_x(1.1)

# -------------------------------
# Panel D: SST Variability
# -------------------------------
ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson(central_longitude=180))
ax4.set_global()
im4 = ax4.pcolormesh(lons, lats, sst_variability, transform=ccrs.PlateCarree(), 
                     cmap=sst_cmap, norm=sst_norm)
ax4.coastlines(linewidth=.3)
ax4.add_feature(cfeature.LAND, facecolor="white")

# Colorbar for Panel D
sm4 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap)
sm4.set_array([])
cbar_ticks_4 = np.linspace(sst_vmin, sst_vmax, 5)
cbar_ticks_4 = np.round(cbar_ticks_4, 2)
cbar4 = fig.colorbar(
    sm4, 
    ax=ax4, 
    ticks=cbar_ticks_4, 
    orientation='horizontal', 
    extend='neither',
    shrink=0.8,
    pad=0.05
)
ax4.set_title('Average SST Variability')
cbar4.set_label("K", labelpad=2)
cbar4.ax.minorticks_off()

# -------------------------------
# Panel labels
# -------------------------------
for ax, label in zip([ax1, ax2, ax3, ax4], ['a', 'b', 'c', 'd']):
    ax.text(0, 1.15, label, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', va='top')
    ax.set_aspect('auto')

plt.savefig(FIGURES_DIR / "Figure_08_average_sensitivity_importance.png", 
            dpi=600, bbox_inches='tight', pad_inches=0.05)
plt.show()


# ----------------------------------------------------------
# Compute significance statistics for manuscript text
# ----------------------------------------------------------
# A grid cell is "significant" if it passes BOTH:
#   - t-test (p < 0.05): ensemble mean distinguishable from zero
#   - sign agreement >= 75% across members
# These are the cells that are NOT hatched

is_significant = (~not_significant) & (~poor_agreement) & valid

# Total ocean cells = cells with a finite MS value (land is NaN)
ocean_mask = np.isfinite(ms)
n_total_ocean = int(ocean_mask.sum().values)

# Significant cells
n_sig = int(is_significant.sum().values)

# Positive vs negative significant cells
sig_positive = is_significant & (ms > 0)
sig_negative = is_significant & (ms < 0)

n_pos = int(sig_positive.sum().values)
n_neg = int(sig_negative.sum().values)

# Percentages of all ocean cells
pct_pos = 100.0 * n_pos / n_total_ocean if n_total_ocean > 0 else np.nan
pct_neg = 100.0 * n_neg / n_total_ocean if n_total_ocean > 0 else np.nan

# Percentages of significant cells
pct_pos_sig = 100.0 * n_pos / n_sig if n_sig > 0 else np.nan
pct_neg_sig = 100.0 * n_neg / n_sig if n_sig > 0 else np.nan

# Mean sensitivity values (positive cells, negative cells)
mean_pos = float(ms.where(sig_positive).mean(skipna=True).values)
mean_neg = float(ms.where(sig_negative).mean(skipna=True).values)

# ----------------------------------------------------------
# Print summary
# ----------------------------------------------------------
print("Observed Marginal Sensitivity Summary")
print("-" * 50)
print(f"Total ocean grid cells                : {n_total_ocean:,}")
print(f"Total significant grid cells          : {n_sig:,}")
print()
print(f"Positive significant cells            : {n_pos:,}")
print(f"Negative significant cells            : {n_neg:,}")
print()
print(f"Percent of all oceans (positive)      : {pct_pos:.1f}%")
print(f"Percent of all oceans (negative)      : {pct_neg:.1f}%")
print()
print(f"Mean positive sensitivity             : {mean_pos:.3f} mm month^-1 K^-1")
print(f"Mean negative sensitivity             : {mean_neg:.3f} mm month^-1 K^-1")
print()
print(f"Percent of significant cells positive : {pct_pos_sig:.1f}%")
print(f"Percent of significant cells negative : {pct_neg_sig:.1f}%")


