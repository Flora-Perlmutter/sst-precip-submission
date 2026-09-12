#!/usr/bin/env python
# coding: utf-8
"""
Figure 6: Amazon basin model construction.

Author: Flora Perlmutter

Description
-----------
4-panel figure illustrating the SST-forced precipitation reconstruction
for the Amazon basin:
  Panel A: Ensemble mean SST sensitivity (dP/dSST)
  Panel B: January 2016 SST anomaly
  Panel C: SST-forced contribution for January 2016,
           area × sensitivity × SST anomaly
  Panel D: Full reconstruction time series (ensemble mean ± SE) vs
           observed precipitation (ensemble mean ± SD)

Weighting convention
--------------------
This figure is the pipeline in miniature, and the panels run in the order the
weighting does: panel A is the raw sensitivity (mm month-1 K-1), panel C applies
the grid-cell area to turn it into a contribution, and panel D sums that over the
ocean. Only panel A changed when the area factor moved downstream.

Required data files (in data/)
-------------------------------
  sst_anom_{SST}.nc                                   (load_and_process_obs.py)
  global_linear_regression_bootstrap_{P}_{SST}.nc     (run_bootstrap_se_obs.py)
  grdc_basins/  (shapefile)   →  Data/Other/grdc_basins on HPC;
                                 copy or symlink to data/grdc_basins

Output
------
  figures/paper_figures/Figure_06_amazon_model_construction.png

"""

import warnings
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR, PAPER_FIGURE_DIR, bootstrap_file
from plotting_functions import drop_extra_coords, grid_area

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR  = DATA_DIR
FIGURES_DIR  = PAPER_FIGURE_DIR
BASINS_DIR   = DATA_DIR / "grdc_basins"

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
        output_file = bootstrap_file(p_name, sst_name)
        
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

        recons.append(drop_extra_coords(r))
        recons_se.append(drop_extra_coords(r_se))
        precip_obs.append(drop_extra_coords(p_obs))
        times_list.append(r["time"])

    if len(recons) == 0:
        raise ValueError("No reconstructions found in the provided result dictionary.")

    # --- Find common time range across all datasets ---
    common_time = sorted(set(times_list[0].values))
    for t in times_list[1:]:
        common_time = np.intersect1d(common_time, t.values)
    if len(common_time) == 0:
        raise ValueError("No overlapping time period across datasets.")

    # --- Reindex all to the common time base --- 
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
    ax.plot(time, p_obs_mean, color="steelblue", linewidth=.8,
            label="Observed precipitation mean", zorder=3)
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
        
        ax.plot(time, r_vals, color="crimson", alpha=0.6, linewidth=.5,
                label="Reconstruction ensemble member mean" if i == 0 else None, zorder=2)
        ax.fill_between(time,
                        r_vals - r_se_vals,
                        r_vals + r_se_vals,
                        color="crimson", alpha=0.15, label="±1 SE" if i == 0 else None,
                        zorder=1)

    # --- Format x-axis ---
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_xlim(time.min(), time.max())
    ax.set_ylim(-90, 90)

# Select basin
basin_id = 3203.  # Adjust as needed
basin_name = 'Amazon'
alpha = 0.05  # Adjust as needed

# --- Compute ensemble means across datasets ---
# Marginal sensitivity ensemble mean
marginal_sensitivities = []
for (precip_name, sst_name), res in linear_results.items():
    if "marginal_sensitivity" in res:
        marginal_sensitivities.append(res["marginal_sensitivity"].sel(basin=basin_id))
marginal_sens_mean = xr.concat(marginal_sensitivities, dim="ensemble").mean("ensemble")

# SST ensemble mean (across all datasets)
sst_datasets = []
for sst_name, sst_data in sst_dict.items():
    # Keep only dimension coordinates (lat, lon, time), drop any scalar/extra coords
    sst_datasets.append(drop_extra_coords(sst_data))
sst_ensemble = xr.concat(sst_datasets, dim="ensemble")
sst_mean = sst_ensemble.mean("ensemble")

# Select January 2016
date = '2016-01-01'
selected_sst = sst_mean.sel(time=date)

# Convolved result -- the SST-forced contribution for this month.
# `marginal_sens_mean` is the raw sensitivity (mm month-1 K-1), which is what the
# left panel shows. Turning it into a contribution is a spatial integral, so the
# cell area enters here; summing this map over lat/lon reproduces the value the
# reconstruction time series carries in the bottom panel.
area = grid_area(marginal_sens_mean)
convolved_result = selected_sst * area * marginal_sens_mean

# The sensitivity panel's vmax is hardcoded below and its right value moved by
# ~3 orders of magnitude when the area factor left the sensitivity. Print what it
# should be set from, so a stale limit shows up in the log rather than as
# saturated colour.
_sens_vals = np.abs(marginal_sens_mean.values[np.isfinite(marginal_sens_mean.values)])
if _sens_vals.size:
    print(f"Amazon sensitivity |x|: p95={np.percentile(_sens_vals, 95):.3g}  "
          f"p98={np.percentile(_sens_vals, 98):.3g}  max={_sens_vals.max():.3g} "
          f"mm month-1 K-1")

# --- Setup figure ---
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(6.32, 4.95), dpi=600)

# Create gridspec: 2 rows, 3 columns
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], 
                       height_ratios=[1.6, 1.2], hspace=0.2, wspace=0.3)

ax_map_sensitivity = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax_map_sst_2016 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax_map_convolved = fig.add_subplot(gs[0, 2], projection=ccrs.Robinson(central_longitude=180))
ax_line = fig.add_subplot(gs[1, :])  # Full width bottom panel

for ax in [ax_map_sensitivity, ax_map_sst_2016, ax_map_convolved]:
    ax.set_global()

# Basin geometry
basin_geom = gpd.GeoDataFrame(
    geometry=[grdc_basins.loc[grdc_basins["MRBID"] == basin_id, "geometry"].iloc[0]]
)

# =====================
# SST SENSITIVITY MAP (Top Left)
# =====================
# The sensitivity panel, mm month-1 K-1. Limits changed when the cell-area factor
# moved out of the saved sensitivity: on the 2 degree grid a cell subtends
# ~1.2e-3 * cos(lat) sr, so the raw slope is ~800/cos(lat) times the old
# area-weighted field. Matches Figures 8 and 11 so the three are comparable.
vmax = 30
vmin = -vmax
num_levels = 20
cmap = plt.get_cmap("RdBu", num_levels)
norm = BoundaryNorm(np.linspace(vmin, vmax, num_levels + 1), cmap.N)

# Remove the scalar basin coordinate
marginal_sens_mean_plot = marginal_sens_mean.reset_coords('basin', drop=True)

marginal_sens_mean_plot.plot(
    ax=ax_map_sensitivity,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    add_colorbar=False
)

basin_geom.plot(
    ax=ax_map_sensitivity,
    transform=ccrs.PlateCarree(),
    cmap='gist_rainbow_r',
    edgecolor="black",
    linewidth=.3
)

ax_map_sensitivity.coastlines(linewidth=.3)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar_ticks = np.linspace(vmin, vmax, 5)
cbar_ticks = np.round(cbar_ticks, 2)
cbar = fig.colorbar(
    sm,
    ax=ax_map_sensitivity,
    orientation="horizontal",
    ticks=cbar_ticks,
    extend="neither",
    shrink=0.8,
    pad=0.03
)
cbar.ax.minorticks_off()
cbar.set_label(r'mm month$^{-1}$ K$^{-1}$', labelpad=2)
ax_map_sensitivity.set_title(f"Amazon SST Sensitivity")

# =====================
# 2016 SST ANOMALY MAP (Top Middle)
# =====================
vmax_sst = 4
vmin_sst = -vmax_sst
cmap_sst = plt.get_cmap("RdBu_r", num_levels)
norm_sst = BoundaryNorm(np.linspace(vmin_sst, vmax_sst, num_levels+1), cmap_sst.N)

selected_sst.plot(
    ax=ax_map_sst_2016,
    cmap=cmap_sst,
    norm=norm_sst,
    transform=ccrs.PlateCarree(),
    add_colorbar=False
)

ax_map_sst_2016.coastlines(linewidth=.3)

sm_sst = plt.cm.ScalarMappable(norm=norm_sst, cmap=cmap_sst)
sm_sst.set_array([])
cbar_ticks_sst = np.linspace(vmin_sst, vmax_sst, 5)
cbar_ticks_sst = np.round(cbar_ticks_sst, 1)
cbar_sst = fig.colorbar(
    sm_sst,
    ax=ax_map_sst_2016,
    orientation="horizontal",
    ticks=cbar_ticks_sst,
    extend="neither",
    shrink=0.8,
    pad=0.03
)
cbar_sst.formatter.set_powerlimits((-2, 2))
cbar_sst.ax.minorticks_off()
cbar_sst.set_label("K", labelpad=2)
ax_map_sst_2016.set_title("Jan 2016 SST Anomaly")

# =====================
# CONVOLVED SST × SENSITIVITY MAP (Top Right)
# =====================
# The contribution panel is numerically unchanged -- area is re-applied above --
# so these limits stay as they were.
vmax_conv = .04
vmin_conv = -vmax_conv
cmap_conv = plt.get_cmap("RdBu", num_levels)
norm_conv = BoundaryNorm(np.linspace(vmin_conv, vmax_conv, num_levels+1), cmap_conv.N)

convolved_result.plot(
    ax=ax_map_convolved,
    cmap=cmap_conv,
    norm=norm_conv,
    transform=ccrs.PlateCarree(),
    add_colorbar=False
)

basin_geom.plot(
    ax=ax_map_convolved,
    transform=ccrs.PlateCarree(),
    cmap='gist_rainbow_r',
    edgecolor="black",
    linewidth=.3
)

ax_map_convolved.coastlines(linewidth=.3)

sm_conv = plt.cm.ScalarMappable(norm=norm_conv, cmap=cmap_conv)
sm_conv.set_array([])
cbar_ticks_conv = np.linspace(vmin_conv, vmax_conv, 5)
cbar_ticks_conv = np.round(cbar_ticks_conv, 2)
cbar_conv = fig.colorbar(
    sm_conv,
    ax=ax_map_convolved,
    orientation="horizontal",
    ticks=cbar_ticks_conv,
    extend="neither",
    shrink=0.8,
    pad=0.03
)
cbar_conv.formatter.set_powerlimits((-2, 2))
cbar_conv.ax.minorticks_off()

# Move the scientific notation (×10ⁿ) closer to the axis
offset_text = cbar_conv.ax.xaxis.get_offset_text()
offset_text.set_x(1.2)  # center horizontally

cbar_conv.set_label(r'mm month$^{-1}$', labelpad=2)
ax_map_convolved.set_title("Jan 2016 SST-Forced Precipitation")

# =====================
# RECONSTRUCTION TIME SERIES (Bottom, Full Width)
# =====================
plot_ensemble_on_ax(linear_results, basin_id, ax_line)

# Add vertical line at January 2016
jan_2016 = np.datetime64('2016-01-01')
ax_line.axvline(mdates.date2num(jan_2016), color='orange', linestyle='--', 
                linewidth=1, label='Jan 2016', zorder=4)

ax_line.set_xlabel("Time")
ax_line.set_ylabel(r'Precipitation Anomaly (mm month$^{-1}$)', labelpad=2)
ax_line.legend(loc='upper left', fontsize=5, ncol=2)
ax_line.set_title(f"{basin_name.capitalize()} SST-Forced Precipitation")
ax_line.grid(False)
ax_line.axhline(y=0, color='grey', alpha=.5, linestyle='-', linewidth=1)

# =====================
# Panel labels
# =====================
for ax, label in zip([ax_map_sensitivity], 
                     ['a']):
    ax.text(-0.22, 1.27, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
for ax, label in zip([ax_map_sst_2016], 
                     ['b']):
    ax.text(-0.15, 1.27, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
for ax, label in zip([ ax_map_convolved], 
                     [ 'c']):
    ax.text(-0.29, 1.27, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')
for ax, label in zip([ax_line], 
                     ['d']):
    ax.text(-.04, 1.13, label, transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')


plt.savefig(FIGURES_DIR / f"Figure_06_amazon_model_construction.png", 
            dpi=600, bbox_inches='tight', pad_inches = .05)
plt.show()

