#!/usr/bin/env python
# coding: utf-8
"""
Figure 3: Observed regression vs observed PCA SST sensitivity pattern correlation.

Author: Flora Perlmutter

Description
-----------
2-panel figure comparing observed bootstrap and observed PCA SST
sensitivity spatial patterns across GRDC basins:

  Panel A: Basin choropleth map colored by pattern correlation between
           obs bootstrap marginal_sensitivity and PCA sensitivity_map
           (RdYlGn colormap, −1 to 1)
  Panel B: Histogram of per-basin pattern correlations with selected
           basins annotated

Pattern correlation is computed per basin: for each basin the obs bootstrap
`marginal_sensitivity` field (lat × lon) is correlated against the PCA
`sensitivity_map` field (same lat × lon grid — no regridding needed).
Both fields are ensemble-averaged across their respective (precip × SST)
members before the per-basin spatial correlation is computed.

Required data files
-------------------
  global_linear_regression_bootstrap_{P}_{SST}.nc   (run_bootstrap_se_obs.py)
  global_pca_regression_{P}_{SST}.nc                (run_pca_regression_global.py)
  grdc_basins/

Output
------
  figures/paper_figures/Figure_03_regression_vs_pca_SST_sensitivity.png
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
# Datasets
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
basin_ids   = grdc_basins["MRBID"].values

# =============================================================================
# LOAD OBS BOOTSTRAP RESULTS
# =============================================================================
print("Loading obs bootstrap results...")
linear_results = {}

for p_name in PRECIP_DATASETS:
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f"global_linear_regression_bootstrap_{p_name}_{sst_name}.nc"
        if os.path.exists(output_file):
            try:
                ds = xr.open_dataset(output_file)
                linear_results[(p_name, sst_name)] = {
                    "model_id"             : ds.attrs["model_id"],
                    "description"          : ds.attrs["description"],
                    "variable"             : ds.attrs["variable"],
                    "alpha"                : ds.attrs["alpha"],
                    "n_bootstrap"          : ds.attrs["n_bootstrap"],
                    "reconstruction"       : ds["reconstruction"],
                    "reconstruction_se"    : ds["reconstruction_se"],
                    "observed_precip"      : ds["observed_precip"],
                    "correlation"          : ds["correlation"],
                    "marginal_sensitivity_se": ds["marginal_sensitivity_se"],
                    "marginal_sensitivity" : ds["marginal_sensitivity"],
                }
                print(f"  Loaded bootstrap: {p_name} × {sst_name}")
            except Exception as e:
                print(f"  Failed bootstrap {p_name} × {sst_name}: {e}")
        else:
            print(f"  Not found: {output_file}")

# =============================================================================
# LOAD OBS PCA RESULTS
# =============================================================================
print("Loading PCA regression results...")
pca_results = {}

for p_name in PRECIP_DATASETS:
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f"global_pca_regression_{p_name}_{sst_name}.nc"
        if os.path.exists(output_file):
            try:
                ds = xr.open_dataset(output_file)
                pca_results[(p_name, sst_name)] = {
                    "sensitivity_map": ds["sensitivity_map"],   # (basin, lat, lon)
                }
                print(f"  Loaded PCA: {p_name} × {sst_name}")
            except Exception as e:
                print(f"  Failed PCA {p_name} × {sst_name}: {e}")
        else:
            print(f"  Not found: {output_file}")

# =============================================================================
# COMPUTE ENSEMBLE MEANS
# =============================================================================
# Obs bootstrap: use compute_ensemble_means to get ensemble-mean
# marginal_sensitivity (basin, lat, lon)
ensemble_obs = compute_ensemble_means(linear_results, "Observations")
obs_ms_mean  = ensemble_obs["marginal_sensitivity_sst"]   # (basin, lat, lon)

# PCA: average sensitivity_map across all (precip × SST) members
sensitivity_map_arrays = [res["sensitivity_map"] for res in pca_results.values()]
if not sensitivity_map_arrays:
    raise RuntimeError("No PCA results loaded. Check OUTPUTS_DIR and filenames.")
pca_ms_mean = xr.concat(sensitivity_map_arrays, dim="member").mean("member", skipna=True)
# pca_ms_mean shape: (basin, lat, lon)

# =============================================================================
# COMPUTE PER-BASIN PATTERN CORRELATION
# =============================================================================
def pattern_correlation(field_a, field_b):
    """
    Spatial pattern correlation between two (lat × lon) fields.
    NaNs are excluded pairwise.
    Returns (corr, rmse).
    """
    a = np.asarray(field_a).flatten()
    b = np.asarray(field_b).flatten()
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return np.nan, np.nan
    corr = np.corrcoef(a, b)[0, 1]
    rmse = np.sqrt(np.mean((a - b) ** 2))
    return corr, rmse


print("Computing per-basin pattern correlations...")
correlations = np.full(len(basin_ids), np.nan)
rmses        = np.full(len(basin_ids), np.nan)

for idx, basin_id in enumerate(basin_ids):
    try:
        obs_field = obs_ms_mean.sel(basin=basin_id)   # (lat, lon)
        pca_field = pca_ms_mean.sel(basin=basin_id)   # (lat, lon) — same grid

        # Both fields are on the same lat/lon grid: no regridding needed.
        corr, rmse = pattern_correlation(obs_field, pca_field)
        correlations[idx] = corr
        rmses[idx]        = rmse
    except Exception as e:
        print(f"  Error for basin {basin_id}: {e}")

print(f"Basins with valid correlation: {np.sum(np.isfinite(correlations))}")
print(f"Mean correlation: {np.nanmean(correlations):.3f}")

# =============================================================================
# MERGE INTO GeoDataFrame
# =============================================================================
grdc_basins = grdc_basins.copy()
try:
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(int)
    basin_ids_typed = basin_ids.astype(int)
except (ValueError, TypeError):
    grdc_basins["MRBID"] = grdc_basins["MRBID"].astype(str)
    basin_ids_typed = basin_ids.astype(str)

grdc_basins["correlation"] = correlations
grdc_basins["rmse"]        = rmses

# =============================================================================
# FIGURE
# =============================================================================
plt.rcParams.update({"font.size": 7})
plt.rcParams["hatch.linewidth"] = 0.4

fig = plt.figure(figsize=(6.25, 2.1), dpi=600)
gs  = fig.add_gridspec(
    nrows=1, ncols=2, hspace=0.3, wspace=0.35, width_ratios=[1.2, 1]
)

num_levels = 20
vmin, vmax = -1, 1
corr_cmap  = plt.get_cmap("RdYlGn", num_levels)
corr_norm  = BoundaryNorm(np.linspace(vmin, vmax, num_levels + 1), corr_cmap.N)

# ─────────────────────────────────────────────────────────────────────────────
# Panel A: Choropleth map
# ─────────────────────────────────────────────────────────────────────────────
ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0))
ax_map.set_global()
ax_map.set_title("Regression vs PCA SST Sensitivity")

for _, row in grdc_basins.iterrows():
    corr_val = row["correlation"]
    if np.isfinite(corr_val):
        color = corr_cmap(corr_norm(corr_val))
    else:
        color = "none"
    ax_map.add_geometries(
        [row.geometry], crs=ccrs.PlateCarree(),
        facecolor=color, edgecolor="black", linewidth=0.3,
    )

# Hatch missing basins
missing_gdf = grdc_basins[~np.isfinite(grdc_basins["correlation"])]
if len(missing_gdf) > 0 and not all(missing_gdf.geometry.is_empty):
    missing_gdf.plot(
        ax=ax_map, facecolor="none",
        transform=ccrs.PlateCarree(), linewidth=0.05,
        edgecolor="black", hatch="////////", alpha=0.8,
    )

ax_map.add_feature(cfeature.LAND, facecolor="white", zorder=0)
ax_map.coastlines(linewidth=0.3, zorder=3)

sm = plt.cm.ScalarMappable(norm=corr_norm, cmap=corr_cmap)
sm.set_array([])
cb = fig.colorbar(
    sm, ax=ax_map, orientation="horizontal",
    ticks=np.linspace(-1, 1, 5), shrink=0.7, pad=0.05,
)
cb.set_label("Pattern Correlation")
cb.ax.minorticks_off()

ax_map.text(-0.15, 1.17, "a", transform=ax_map.transAxes,
            fontsize=10, fontweight="bold", va="top")

# ─────────────────────────────────────────────────────────────────────────────
# Panel B: Histogram
# ─────────────────────────────────────────────────────────────────────────────
ax_hist = plt.subplot(gs[0, 1])

corr_values = grdc_basins["correlation"].dropna().values
hist_bins   = np.linspace(-1, 1, 21)

n, bins, patches = ax_hist.hist(
    corr_values, bins=hist_bins,
    color="lightblue", edgecolor="black", linewidth=0.5, alpha=0.8,
)

## Colour each bar to match the map colormap
#for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
#    bin_center = 0.5 * (left_edge + right_edge)
#    patch.set_facecolor(corr_cmap(corr_norm(bin_center)))

ax_hist.set_xlabel("Regression vs PCA Correlation")
ax_hist.set_ylabel("Number of Basins")
ax_hist.set_xticks([-1, -0.5, 0, 0.5, 1])
ax_hist.set_xlim(-1, 1)
ax_hist.set_ylim(0, 175)

ax_hist.text(-0.26, 1.04, "b", transform=ax_hist.transAxes,
             fontsize=10, fontweight="bold", va="top", ha="left")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
out_fig = FIGURES_DIR / "Figure_03_regression_vs_pca_SST_sensitivity.png"
plt.savefig(out_fig, dpi=600, bbox_inches="tight")
print(f"\nFigure saved → {out_fig}")
plt.show()