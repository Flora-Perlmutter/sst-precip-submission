#!/usr/bin/env python
# coding: utf-8
"""
Figure 6 (PCA version): SST-forced precipitation variability and magnitude by basin.

Author: Flora Perlmutter

Description
-----------
Identical 4-panel layout to Figure_6_sst_importance_by_basin.py, but reads
PCA-based reconstruction results instead of univariate bootstrap results.

  Panel A: Map of ensemble mean correlation between PCA SST reconstruction
           and observed precipitation (SST-forced variability %)
  Panel B: Histogram of the same correlation, with selected basins annotated
  Panel C: Map of std(PCA reconstruction) / std(observed precip) × 100
           (SST-forced magnitude %)
  Panel D: Histogram of the same ratio, with selected basins annotated

Required data files
-------------------
  global_pca_regression_{P}_{SST}.nc   (run_pca_regression_global.py)
  sst_anom_{SST}.nc                    (load_and_process_obs.py)
  grdc_basins/                         (copy from HPC Data/Other/)

Output
------
  figures/paper_figures/Figure_6_pca_sst_importance_by_basin.png

Relationship to original Figure 6
----------------------------------
The only substantive differences from Figure_6_sst_importance_by_basin.py are:
  1. Input files: global_pca_regression_*.nc instead of
                  global_linear_regression_bootstrap_*.nc
  2. The 'correlation' variable now contains in-sample corr(PCA recon, obs)
     rather than the bootstrap correlation.
  3. The 'std_ratio' variable is read directly from the PCA output
     (std(recon)/std(obs)*100) rather than being derived from
     'reconstruction' and 'observed_precip' on the fly.
  4. Output filename has '_pca_' in it.
Everything else — colormaps, bin edges, annotations, panel labels — is unchanged.
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
from plotting_functions import (
    compute_ensemble_means,
    basin_id_for_name,
    basin_name_for_id,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR
BASINS_DIR  = DATA_DIR / "grdc_basins"

# ---------------------------------------------------------------------------
# Datasets (must match filenames produced by run_pca_regression_global.py)
# ---------------------------------------------------------------------------
PRECIP_DATASETS = {
    "GPCP"        : None,
    "CRU"         : None,
    "GPCC"        : None,
    "CPC"         : None,
    "UDel"        : None,
    "PREC"        : None,
    "TerraClimate": None,
    "REGEN"       : None,
}
SST_DATASETS = ["ERSSTv6", "COBE-SST3"]

# ---------------------------------------------------------------------------
# Load basin boundaries
# ---------------------------------------------------------------------------
grdc_basins = gpd.read_file(BASINS_DIR)

# ---------------------------------------------------------------------------
# Load SST anomalies (needed only for n_ocean_cells denominator — same as
# original Figure 6; kept for parity, not used in PCA panels A/B/C/D).
# ---------------------------------------------------------------------------
sst_dict = {}
for sst_name in SST_DATASETS:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()

# =============================================================================
# LOAD ALL PCA REGRESSION RESULTS
# =============================================================================
print("Loading PCA regression results...")
pca_results = {}

for p_name in PRECIP_DATASETS:
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f"global_pca_regression_{p_name}_{sst_name}.nc"

        if os.path.exists(output_file):
            try:
                ds = xr.open_dataset(output_file)

                # Build result dict with same keys that compute_ensemble_means
                # expects from the univariate bootstrap results, so we can reuse
                # compute_ensemble_means without modification.
                #
                # Key mapping (PCA → univariate schema):
                #   'correlation'    — already a (basin,) array of corr(recon,obs)
                #   'reconstruction' — (basin, time) PCA reconstruction
                #   'observed_precip'— (basin, time) observed precip
                #   'std_ratio'      — (basin,) precomputed std ratio [%]
                #
                # NOTE: 'marginal_sensitivity_sst' is expected by compute_ensemble_means
                # for the importance panels (Panels C/D in the *original* Fig 6 use
                # a different metric — std_ratio — so we map it here).
                result = {
                    "model_id"           : f"pca_{p_name}_{sst_name}",
                    "description"        : "Global PCA regression",
                    "variable"           : "precip",
                    "alpha"              : ds.attrs.get("alpha", 0.05),
                    "n_bootstrap"        : 0,               # not applicable
                    "reconstruction"     : ds["reconstruction"],
                    "observed_precip"    : ds["observed_precip"],
                    "correlation"        : ds["correlation"],  # corr(recon, obs)
                    "std_ratio"          : ds["std_ratio"],    # std(recon)/std(obs)*100
                    # The two fields below are used by compute_ensemble_means to
                    # derive sensitivity / importance maps.  For the PCA case we
                    # pass sensitivity_map as 'marginal_sensitivity' so that any
                    # ensemble-averaging code that touches it still works.
                    "marginal_sensitivity"    : ds["sensitivity_map"].mean("lat").mean("lon"),  # placeholder scalar per basin
                    "marginal_sensitivity_se" : xr.zeros_like(ds["correlation"]),
                    # Extra PCA-specific fields (not used by plotting code)
                    "optimal_k"               : ds["optimal_k"],
                    "cv_re_optimal"           : ds["cv_re_optimal"],
                    "cv_corr_optimal"         : ds["cv_corr_optimal"],
                }

                pca_results[(p_name, sst_name)] = result
                print(f"  Loaded: {p_name} × {sst_name}")

            except Exception as e:
                print(f"  Failed to load {p_name} × {sst_name}: {e}")
        else:
            print(f"  Not found: {output_file}")

# =============================================================================
# COMPUTE ENSEMBLE MEANS
# =============================================================================
ensemble_pca = compute_ensemble_means(pca_results, "Observations")

# ---------------------------------------------------------------------------
# Prepare common data
# ---------------------------------------------------------------------------
basin_ids = grdc_basins["MRBID"].values

# --- Panel A / B data: ensemble-mean correlation ---
corr_mean = ensemble_pca["correlation_sst"].reindex(basin=basin_ids)

# --- Panel C / D data: ensemble-mean std_ratio ---
# We average std_ratio directly across (precip, SST) ensemble members.
# This mirrors how the original script averages std(recon)/std(obs).
std_ratio_arrays = []
for (p_name, sst_name), res in pca_results.items():
    arr = res["std_ratio"].reindex(basin=basin_ids)
    std_ratio_arrays.append(arr)

if std_ratio_arrays:
    std_ratio_ensemble = xr.concat(std_ratio_arrays, dim="member").mean("member", skipna=True)
else:
    std_ratio_ensemble = xr.full_like(corr_mean, np.nan)

# ---------------------------------------------------------------------------
# Merge into GeoDataFrames (same type-casting logic as original)
# ---------------------------------------------------------------------------
metrics_df   = gpd.GeoDataFrame(
    {"correlation": corr_mean.values}, index=grdc_basins["MRBID"].values
)
std_ratio_df = gpd.GeoDataFrame(
    {"std_ratio": std_ratio_ensemble.values}, index=grdc_basins["MRBID"].values
)

grdc_basins = grdc_basins.copy()
try:
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(int)
    metrics_df.index   = metrics_df.index.astype(int)
    std_ratio_df.index = std_ratio_df.index.astype(int)
except (ValueError, TypeError):
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(str)
    metrics_df.index   = metrics_df.index.astype(str)
    std_ratio_df.index = std_ratio_df.index.astype(str)

gdf_corr      = grdc_basins.merge(metrics_df,   left_on="MRBID", right_index=True, how="left")
gdf_std_ratio = grdc_basins.merge(std_ratio_df, left_on="MRBID", right_index=True, how="left")

# Add corr_plot column (correlation as %, clipped for colormapping)
gdf_corr["corr_plot"] = corr_mean.values * 100

# =============================================================================
# FIGURE
# =============================================================================
plt.rcParams.update({"font.size": 7})
plt.rcParams['hatch.linewidth'] = 0.4
fig = plt.figure(figsize=(6.25, 4.7), dpi=600)
gs  = gridspec.GridSpec(
    nrows=2, ncols=2, figure=fig,
    width_ratios=[1.2, 1], hspace=0.4, wspace=0.35
)


# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Correlation Map
# ─────────────────────────────────────────────────────────────────────────────
ax_map = plt.subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))
ax_map.set_global()

corr_vmin, corr_vmax = 0, 60
corr_cmap = plt.get_cmap("Blues", 20)
corr_norm = BoundaryNorm(np.linspace(corr_vmin, corr_vmax, 21), corr_cmap.N)

ax_map.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.3)

sig_gdf = gdf_corr[gdf_corr["corr_plot"] > 0].copy()
for idx, row in sig_gdf.iterrows():
    ax_map.add_geometries(
        [row.geometry], crs=ccrs.PlateCarree(),
        facecolor=corr_cmap(corr_norm(row.corr_plot)),
        edgecolor="black", linewidth=0.3, alpha=0.9,
    )

insig_gdf = gdf_corr[(gdf_corr["correlation"] <= 0) | (gdf_corr["correlation"].isna())]
if not all(insig_gdf.is_empty):
    insig_gdf.plot(
        ax=ax_map, facecolor="none", 
        transform=ccrs.PlateCarree(), linewidth=0.05, edgecolor="black", hatch='////////',  alpha=0.8
    )

ax_map.coastlines(linewidth=0.3)
ax_map.add_feature(cfeature.LAND, facecolor="white")
ax_map.set_aspect("auto")

sm_corr = plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
sm_corr.set_array([])
cbar_map = fig.colorbar(
    sm_corr, ax=ax_map, ticks=[0, 15, 30, 45, 60],
    orientation="horizontal", shrink=0.8, pad=0.05,
)
ax_map.set_title("Average SST-Forced Precipitation Variability\n(PCA Reconstruction)")
cbar_map.set_label("%")
cbar_map.ax.minorticks_off()

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Correlation Histogram
# ─────────────────────────────────────────────────────────────────────────────
ax_hist = plt.subplot(gs[0, 1])

corr_values = gdf_corr["correlation"].dropna().values * 100
hist_bins   = np.linspace(0, 60, 21)
n, bins, patches = ax_hist.hist(
    corr_values, bins=hist_bins,
    color="lightblue", linewidth=0.5, edgecolor="black", alpha=0.7,
)
ax_hist.set_xlabel("Average SST-Forced Precipitation Variability (%)")
ax_hist.set_ylabel("Number of Basins")
ax_hist.set_xticks([0, 15, 30, 45, 60])
ax_hist.set_ylim(0, 170)
ax_hist.set_xlim(0, 60)

# Annotate selected basins — identical basin list and positioning logic to
# the original Figure 6 (adjustments copied verbatim so panels look the same).
highlight_basins = [
    "MURRAY", "YELLOW RIVER",
    "AMAZON (also AMAZONAS)", "MISSISSIPPI", "CONNECTICUT", "ZARUMILLA",
]

for basin_name in highlight_basins:
    basin_id = basin_id_for_name(basin_name)
    if basin_id is None:
        continue
    basin_id = float(basin_id)
    canonical = basin_name_for_id(basin_id)
    row = gdf_corr.loc[gdf_corr["MRBID"] == basin_id]
    if row.empty:
        continue
    val = row["correlation"].values[0] * 100
    if not np.isfinite(val):
        continue

    bin_idx    = np.clip(np.digitize(val, bins) - 1, 0, len(bins) - 2)
    bin_center = 0.5 * (bins[bin_idx] + bins[bin_idx + 1])
    bin_height = n[bin_idx]

    text_x = bin_center + 0.0015
    ha_text = "left"
    text_y  = bin_height + 8

    cu = canonical.upper()
    if cu == "ZARUMILLA":
        text_x = bin_center + 0.04
        text_y = bin_height + 14
    elif cu == "AMAZON":
        text_x  = bin_center + 0.04
        text_y  = bin_height + 24
    elif cu == "MURRAY":
        text_y  = bin_height + 18
    elif cu == "YELLOW RIVER":
        text_y  = bin_height + 14
        ha_text = "right"
    elif cu == "MISSISSIPPI":
        text_y  = bin_height + 14
        

    arrow_kw = dict(arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    if cu == "YELLOW RIVER":
        ax_hist.annotate(
            canonical,
            xy=(bin_center, bin_height),
            xytext=(text_x, text_y),
            ha=ha_text,
            fontsize=5,
            bbox=dict(facecolor="white",  edgecolor="none", boxstyle="round,pad=0.2"),
            **arrow_kw,
        )
    else:
        ax_hist.annotate(
            canonical,
            xy=(bin_center, bin_height),
            xytext=(text_x, text_y),
            ha=ha_text,
            fontsize=5,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
            **arrow_kw,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Panel C: Std Ratio Map
# ─────────────────────────────────────────────────────────────────────────────
ax_map_2 = plt.subplot(gs[1, 0], projection=ccrs.Robinson(central_longitude=0))

ratio_vmin, ratio_vmax = 0, 60
ratio_cmap = plt.get_cmap("Blues", 21)
ratio_norm = BoundaryNorm(np.linspace(ratio_vmin, ratio_vmax, 21), ratio_cmap.N)

ax_map_2.add_feature(cfeature.OCEAN, facecolor="lightgray", alpha=0.3)

valid_gdf = gdf_std_ratio[gdf_std_ratio["std_ratio"].notna()]
for idx, row in valid_gdf.iterrows():
    ratio = np.clip(row.std_ratio, ratio_vmin, ratio_vmax)
    ax_map_2.add_geometries(
        [row.geometry], crs=ccrs.PlateCarree(),
        facecolor=ratio_cmap(ratio_norm(ratio)),
        edgecolor="black", linewidth=0.3, alpha=0.9,
    )

invalid_gdf = gdf_std_ratio[gdf_std_ratio["std_ratio"].isna()]
if not all(invalid_gdf.is_empty):
    insig_gdf.plot(
        ax=ax_map_2, facecolor="none", 
        transform=ccrs.PlateCarree(), linewidth=0.05, edgecolor="black", hatch='////////',  alpha=0.8
    )

ax_map_2.set_global()
ax_map_2.coastlines(linewidth=0.3)
ax_map_2.add_feature(cfeature.LAND, facecolor="white")
ax_map_2.set_aspect("auto")

sm_ratio = plt.cm.ScalarMappable(norm=ratio_norm, cmap=ratio_cmap)
sm_ratio.set_array([])
cbar_map_2 = fig.colorbar(
    sm_ratio, ax=ax_map_2, ticks=np.linspace(0, 60, 5),
    orientation="horizontal", shrink=0.8, pad=0.05,
)
ax_map_2.set_title("Average SST-Forced Precipitation Magnitude\n(PCA Reconstruction)")
cbar_map_2.set_label("%")
cbar_map_2.ax.minorticks_off()

# ─────────────────────────────────────────────────────────────────────────────
# Panel D: Std Ratio Histogram
# ─────────────────────────────────────────────────────────────────────────────
ax_hist_2 = plt.subplot(gs[1, 1])

ratio_values_clean    = gdf_std_ratio["std_ratio"].dropna().values
ratio_values_filtered = ratio_values_clean[(ratio_values_clean >= 0) & (ratio_values_clean <= 200)]

hist_bins_2 = np.linspace(0, 60, 21)
n2, bins2, _ = ax_hist_2.hist(
    ratio_values_filtered, bins=hist_bins_2,
    color="lightblue", edgecolor="black", linewidth=0.5, alpha=0.7,
)
ax_hist_2.set_xlabel("Average SST Forced Precipitation Magnitude (%)")
ax_hist_2.set_ylabel("Number of Basins")
ax_hist_2.set_xticks([0, 15, 30, 45, 60])
ax_hist_2.set_ylim(0, 280)
ax_hist_2.set_xlim(0, 60)

# Annotate selected basins (same list and positioning as original Panel D)
highlight_basins_2 = [
    "MURRAY", "YELLOW RIVER",
    "AMAZON (also AMAZONAS)", "MISSISSIPPI", "CONNECTICUT", "ZARUMILLA",
]

for basin_name in highlight_basins_2:
    bid = basin_id_for_name(basin_name)
    if bid is None:
        continue
    bid = float(bid)
    canonical = basin_name_for_id(bid)

    row = gdf_std_ratio.loc[gdf_std_ratio["MRBID"] == bid]
    if row.empty:
        print(f"  No std_ratio value for basin {canonical}")
        continue

    val = row["std_ratio"].values[0]
    if not np.isfinite(val) or val < 0 or val > 160:
        continue

    bin_idx    = np.clip(np.digitize(val, bins2) - 1, 0, len(bins2) - 2)
    bin_center = 0.5 * (bins2[bin_idx] + bins2[bin_idx + 1])
    bin_height = n2[bin_idx]

    cu = canonical.upper()
    if cu in ["YELLOW RIVER", "GANGES", "MISSISSIPPI"]:
        text_x = bin_center + 0.018;  ha_text = "left";  text_y = 13
    elif cu == "AMAZON":
        text_x = bin_center + 0.018;  ha_text = "left";  text_y = 22
    elif cu in ["MURRAY", "ZARUMILLA"]:
        text_x = bin_center + 0.015;  ha_text = "left";  text_y = 8
    else:
        text_x = bin_center + 0.015;  ha_text = "left";  text_y = 0

    arrow_kw = dict(arrowprops=dict(arrowstyle="->", lw=0.8, color="black"))
    if cu == "YELLOW RIVER":
        arrow_kw = {}

    if cu == "AMAZON":
        ax_hist_2.annotate(
        canonical,
        xy=(bin_center, bin_height),
        xytext=(text_x, bin_height + text_y + 14),
        ha=ha_text,
        fontsize=5,
        zorder=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
        **arrow_kw,
    )
        
    else: 
        ax_hist_2.annotate(
            canonical,
            xy=(bin_center, bin_height),
            xytext=(text_x, bin_height + text_y + 14),
            ha=ha_text,
            fontsize=5,
            zorder=1,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"),
            **arrow_kw,
        )

# ─────────────────────────────────────────────────────────────────────────────
# Panel labels
# ─────────────────────────────────────────────────────────────────────────────
ax_map.text(-0.15,  1.16, "a", transform=ax_map.transAxes,
            fontsize=10, fontweight="bold", va="top")
ax_hist.text(-0.11, 1.12, "b", transform=ax_hist.transAxes,
             fontsize=10, fontweight="bold", va="top", ha="left")
ax_map_2.text(-0.15, 1.16, "c", transform=ax_map_2.transAxes,
              fontsize=10, fontweight="bold", va="top")
ax_hist_2.text(-0.11, 1.12, "d", transform=ax_hist_2.transAxes,
               fontsize=10, fontweight="bold", va="top", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_fig = FIGURES_DIR / "Figure_6_pca_sst_importance_by_basin.png"
plt.savefig(out_fig, dpi=600, bbox_inches='tight', pad_inches=0.2)
print(f"\nFigure saved → {out_fig}")
plt.show()