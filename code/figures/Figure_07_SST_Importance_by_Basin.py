#!/usr/bin/env python
# coding: utf-8
"""
Figure 7: SST-forced precipitation variability and magnitude by basin.

Author: Flora Perlmutter

Description
-----------
4-panel figure summarising across-basin SST importance:
  Panel A: Map of ensemble mean correlation^2 between SST reconstruction
           and observed precipitation (SST-forced variability %)
  Panel B: Histogram of the same correlation^2, with selected basins annotated
  Panel C: Map of std(SST reconstruction) / std(observed precip) × 100
           (SST-forced magnitude %)
  Panel D: Histogram of the same ratio, with selected basins annotated

Required data files 
-------------------------------
  global_linear_regression_bootstrap_{P}_{SST}.nc  (run_bootstrap_se_obs.py)
  sst_anom_{SST}.nc                                (load_and_process_obs.py)
  grdc_basins/                                     (copy from HPC Data/Other/)

Output
------
  figures/paper_figures/Figure_07_sst_importance_by_basin.png

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
from plotting_functions import compute_ensemble_means, basin_id_for_name, basin_name_for_id

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

# -------------------------------
# Prepare data for all panels
# -------------------------------
basin_ids = grdc_basins['MRBID'].values

# Mean correlations
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

n_ocean_cells = np.isfinite(sst_dict['ERSSTv6'].mean('time')).sum(dim=('lat', 'lon'))
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

# --------------------------------------------
# Create 2x2 figure with maps and histograms
# --------------------------------------------
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(6.25, 4.7), dpi=600)
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig, width_ratios=[1.2, 1], hspace=0.4, wspace=0.35)


# ---------------------------------------
# Panel A: Correlation^2 Map
# ---------------------------------------
ax_map = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))

# Define colormap and normalization
corr_vmin, corr_vmax = 0, 30
corr_cmap = plt.get_cmap('Blues', 20)
corr_norm = BoundaryNorm(np.linspace(corr_vmin, corr_vmax, 21), corr_cmap.N)

# Plot background ocean
ax_map.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.3)

# Significant basins
sig_gdf = gdf_corr.copy()
sig_gdf['corr_plot'] = (corr_mean.values**2)*100
sig_gdf = sig_gdf[sig_gdf['corr_plot'] > 0]

for idx, row in sig_gdf.iterrows():
    ax_map.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                          facecolor=corr_cmap(corr_norm(row.corr_plot)),
                          edgecolor='black', linewidth=0.3, alpha=0.9)

# Negative or missing
insig_gdf = gdf_corr[(gdf_corr["correlation"] <= 0) | (gdf_corr["correlation"].isna())]
if not all(insig_gdf.is_empty):
    for _, row in insig_gdf.iterrows():
        ax_map.add_geometries([row.geometry], crs=ccrs.PlateCarree(),
                              facecolor='none', hatch='///',
                              edgecolor='black', linewidth=0.3)
    
ax_map.coastlines(linewidth=.3)
ax_map.add_feature(cfeature.LAND, facecolor="white")
ax_map.set_aspect('auto')

# Add colorbar
sm_corr = plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
sm_corr.set_array([])
cbar_map = fig.colorbar(sm_corr, ax=ax_map, ticks=[0, 6, 12, 18, 24, 30], 
                        orientation='horizontal', shrink=.8, pad=0.05)
ax_map.set_title("Average SST-Forced Precipitation Variability")
cbar_map.set_label("%")
cbar_map.ax.minorticks_off()


# ---------------------------------------
# Panel B: Correlation^2 Histogram
# ---------------------------------------
ax_hist = plt.subplot(gs[0, 1])

corr_values = (gdf_corr["correlation"].dropna().values**2)*100
# Match bins to colorbar ticks: [0, 6, 12, 18, 24, 30]
hist_bins = np.linspace(0, 30, 21)  # Creates bins that align with 15% intervals
n, bins, patches = ax_hist.hist(corr_values, bins=hist_bins, color="lightblue", linewidth=.5, 
                                 edgecolor="black", alpha=0.7)
ax_hist.set_xlabel("Average SST-Forced Precipitation Variability (%)")
ax_hist.set_ylabel("Number of Basins")
# Set x-axis ticks to match colorbar
ax_hist.set_xticks([0, 6, 12, 18, 24, 30])

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
    val = (row['correlation'].values[0]**2)*100
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
    elif basin_name.upper() == "AMAZON":
        text_x = bin_center 
        text_y = bin_height + 4 
        ha_text = "left"
    elif basin_name.upper() == "MURRAY":
        text_y = bin_height + 12  
    elif basin_name.upper() == "YELLOW RIVER":
        text_y = bin_height + 18
        ha_text = "left"
    elif basin_name.upper() == "CONNECTICUT":
        text_y = bin_height + 14  
        ha_text = "right"
    else:
        text_x = bin_center + 0.0015
        ha_text = "left"
    if basin_name.upper() in ["YELLOW RIVER"]:
        plt.annotate(
        basin_name,
        xy=(bin_center, bin_height),
        xytext=(text_x, text_y),
        ha=ha_text,
        fontsize=5,
        #arrowprops=dict(arrowstyle="->", lw=.8, color="black"),
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
    )
    else:
        plt.annotate(
            basin_name,
            xy=(bin_center, bin_height),
            xytext=(text_x, text_y),
            ha=ha_text,
            fontsize=5,
            arrowprops=dict(arrowstyle="->", lw=.8, color="black"),
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
        )
#ax_hist.set_title('Distribution of SST-Forced Precipitation Variabilities')
ax_hist.set_ylim(0, 120)
ax_hist.set_xlim(0, 30)


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
ax_map_2.coastlines(linewidth=.3)
ax_map_2.add_feature(cfeature.LAND, facecolor="white")
ax_map_2.set_aspect('auto')

# Add colorbar
sm_ratio = plt.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
sm_ratio.set_array([])
cbar_map_2 = fig.colorbar(sm_ratio, ax=ax_map_2, ticks=np.linspace(0, 60, 5), 
                        orientation='horizontal', shrink=0.8, pad=0.05)
ax_map_2.set_title("Average SST-Forced Precipitation Magnitude")
cbar_map_2.set_label("%")
cbar_map_2.ax.minorticks_off()


# ---------------------------------------
# Panel D: Std Ratio Histogram
# ---------------------------------------
ax_hist_2 = plt.subplot(gs[1, 1])

ratio_values_clean = gdf_std_ratio["std_ratio"].dropna().values
ratio_values_filtered = ratio_values_clean[(ratio_values_clean >= 0) & (ratio_values_clean <= 200)]

# Match bins to colorbar ticks: [0, 15, 30, 45, 60]
hist_bins_2 = np.linspace(0, 60, 21)  # Creates bins that align with 15% intervals
n, bins, patches = ax_hist_2.hist(
    ratio_values_filtered, bins=hist_bins_2, color="lightblue", edgecolor="black", linewidth=.5, alpha=0.7
)
ax_hist_2.set_xlabel("Average SST Forced Precipitation Magnitude (%)")
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
    elif canonical_name.upper() in ['AMAZON']:
        text_x = bin_center  + 0.018
        ha_text = "left"
        text_y = 17
    elif canonical_name.upper() in ['MURRAY', 'ZARUMILLA']:
        text_x = bin_center  + 0.015
        ha_text = "left"
        text_y = 8
    else:
        text_x = bin_center + 0.015
        ha_text = "left"
        text_y = 0
    
    if canonical_name.upper() in ['MISSISSIPPI']:
        ax_hist_2.annotate(
            canonical_name,
            xy=(bin_center, bin_height),
            xytext=(text_x, bin_height+text_y + 17),
            ha=ha_text,
            fontsize=5,
            #arrowprops=dict(arrowstyle="->", lw=.8, color="black"),
            bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
        )
    else:
        ax_hist_2.annotate(
        canonical_name,
        xy=(bin_center, bin_height),
        xytext=(text_x, bin_height+text_y + 14),
        ha=ha_text,
        fontsize=5,
        arrowprops=dict(arrowstyle="->", lw=.8, color="black"),
        bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle="round,pad=0.2")
    )
    
ax_hist_2.set_ylim(0, 300)
ax_hist_2.set_xlim(0, 60)
    
# Add panel labels
ax_map.text(-0.15, 1.16, 'a', transform=ax_map.transAxes, 
            fontsize=10, fontweight='bold', va='top')
ax_hist.text(-0.11, 1.12, 'b', transform=ax_hist.transAxes, 
             fontsize=10, fontweight='bold', va='top', ha='left')
ax_map_2.text(-0.15, 1.16, 'c', transform=ax_map_2.transAxes, 
            fontsize=10, fontweight='bold', va='top')
ax_hist_2.text(-0.11, 1.12, 'd', transform=ax_hist_2.transAxes, 
             fontsize=10, fontweight='bold', va='top', ha='left')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "Figure_07_sst_importance_by_basin.png",
            dpi=600, pad_inches = .1)

# ============================================================================
# NUMBERS FOR THE MANUSCRIPT PARAGRAPH
# ============================================================================
# Panels a/b plot correlation**2 as a percentage (r-squared, the shared
# variance). Panels c/d plot std(reconstruction)/std(observed) as a
# percentage. These are different quantities: the std ratio is a ratio of
# magnitudes and must NOT be squared.

r_vals     = gdf_corr["correlation"].dropna().values.astype(float)
r2_pct     = (r_vals ** 2) * 100
ratio_vals = gdf_std_ratio["std_ratio"].dropna().values.astype(float)

print("\n" + "=" * 78)
print("FIGURE 7 -- NUMBERS FOR THE MANUSCRIPT PARAGRAPH")
print("=" * 78)

print(f"\nBasins with a correlation value : {r_vals.size}")
print(f"Basins with a std-ratio value   : {ratio_vals.size}")
print(f"Basins with r > 0               : {(r_vals > 0).sum()} "
      f"({(r_vals > 0).mean() * 100:.2f}%)")
print(f"Basins with r <= 0              : {(r_vals <= 0).sum()} "
      f"(hatched in panel a; note r^2 hides their sign)")

# ---------------------------------------------------------------------------
# Panels a/b -- correlation and shared variance
# ---------------------------------------------------------------------------
print("\n--- Panels a/b: correlation r and r^2 ---")
print("(percentiles are computed independently per column; they refer to the")
print(" same basin only where every r > 0)")
print(f"{'stat':>8}  {'r':>8}  {'r^2 (%)':>9}")
for label, q in [('min', 0), ('5th', 5), ('25th', 25), ('median', 50),
                 ('75th', 75), ('95th', 95), ('max', 100)]:
    print(f"{label:>8}  {np.percentile(r_vals, q):8.3f}  {np.percentile(r2_pct, q):9.1f}")
print(f"{'mean':>8}  {r_vals.mean():8.3f}  {r2_pct.mean():9.1f}")

counts, edges = np.histogram(r2_pct, bins=hist_bins)
peak = int(counts.argmax())
print(f"\nModal bin of panel b : {edges[peak]:.1f}-{edges[peak + 1]:.1f}% "
      f"({counts[peak]} basins)")
print(f"Middle 50% of basins : {np.percentile(r2_pct, 25):.1f}% to "
      f"{np.percentile(r2_pct, 75):.1f}% r^2")
print(f"Middle 90% of basins : {np.percentile(r2_pct, 5):.1f}% to "
      f"{np.percentile(r2_pct, 95):.1f}% r^2")

# ---------------------------------------------------------------------------
# Panels c/d -- relative magnitude
# ---------------------------------------------------------------------------
print("\n--- Panels c/d: std(reconstruction) / std(observed), % ---")
for label, q in [('min', 0), ('5th', 5), ('25th', 25), ('median', 50),
                 ('75th', 75), ('95th', 95), ('max', 100)]:
    print(f"{label:>8}  {np.percentile(ratio_vals, q):8.1f}")
print(f"{'mean':>8}  {ratio_vals.mean():8.1f}")

in_range = (ratio_vals >= 1) & (ratio_vals <= 15)
print(f"\nBasins with ratio 1-15% : {in_range.sum()} of {ratio_vals.size} "
      f"({in_range.mean() * 100:.1f}%)")
print(f"Basins with ratio > 30% : {(ratio_vals > 30).sum()}")

# ---------------------------------------------------------------------------
# Basins annotated in the figure
# ---------------------------------------------------------------------------
print("\n--- Annotated basins ---")
print(f"{'basin':<24}{'r':>8}{'r^2 (%)':>10}{'std ratio (%)':>15}")
for basin_name in highlight_basins:
    bid = basin_id_for_name(basin_name)
    if bid is None:
        continue
    bid = float(bid)
    canonical = basin_name_for_id(bid)
    r_row = gdf_corr.loc[gdf_corr['MRBID'] == bid, 'correlation']
    s_row = gdf_std_ratio.loc[gdf_std_ratio['MRBID'] == bid, 'std_ratio']
    r_b = float(r_row.values[0]) if len(r_row) else np.nan
    s_b = float(s_row.values[0]) if len(s_row) else np.nan
    print(f"{canonical:<24}{r_b:>8.3f}{(r_b ** 2) * 100:>10.1f}{s_b:>15.1f}")

# ---------------------------------------------------------------------------
# Direct comparison against the numbers currently in the paragraph
# ---------------------------------------------------------------------------
print("\n--- Paragraph numbers, r vs r^2 ---")
print("If a quoted figure was a correlation, its r^2 equivalent is:")
for quoted in [0.11, 0.20, 0.30, 0.41, 0.50]:
    print(f"  r = {quoted:.2f}  ->  r^2 = {quoted ** 2 * 100:.1f}%")
print("The 42% Amazon magnitude is a std ratio, not a correlation -- leave it")
print("as it is; squaring it would be a category error.")
print("=" * 78 + "\n")

plt.show()

