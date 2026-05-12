#!/usr/bin/env python
# coding: utf-8
"""
Combined Figure 5: Randomization experiment and marginal sensitivity
pattern correlation stability.

Author: Flora Perlmutter

Description
-----------
2-panel figure (1 row × 2 columns):

  Panel A (left):  Distribution of original SST-precipitation correlations
                   across river basins and ensemble members, overlaid with
                   the distribution of correlations from the randomization
                   experiment (SST resampled, precipitation held fixed).
                   Vertical dashed lines mark the mean of each distribution.

  Panel B (right): Correlation between each 30-year rolling-window marginal
                   sensitivity estimate and the full-period ensemble mean,
                   across all basins. Shaded bands (±1 SD across basins) and
                   per-member lines show pattern stability over time.

Required data files
-------------------------------
  randomization_experiment_{P}_{SST}.nc   (run_bootstrap_randomization.py)
  pattern_correlations_all_basins.nc       (compute_pattern_correlations.py)
  
  Output
------
  <PAPER_FIGURE_DIR>/Figure_05_regression_robustness_tests.png

"""

import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import DATA_DIR, PAPER_FIGURE_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR

# ===========================================================================
# randomization experiment
# ===========================================================================
PRECIP_DATASETS = {
    'CPC': None,
    'CRU': None,
    'GPCC': None,
    'GPCP': None,
    'PREC': None,
    'REGEN': None,
    'TerraClimate': None,
    'UDel': None,
}
SST_DATASETS = ['ERSSTv6', 'COBE-SST3']

results = {}
print("Loading randomization results...")
for p_name in PRECIP_DATASETS:
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f'randomization_experiment_{p_name}_{sst_name}.nc'
        if os.path.exists(output_file):
            try:
                result_ds = xr.open_dataset(output_file)
                results[(p_name, sst_name)] = {
                    'model_id':               result_ds.attrs['model_id'],
                    'description':            result_ds.attrs['description'],
                    'variable':               result_ds.attrs['variable'],
                    'alpha':                  result_ds.attrs['alpha'],
                    'n_bootstrap':            result_ds.attrs['n_bootstrap'],
                    'original_correlation':   result_ds['original_correlation'],
                    'bootstrap_correlations': result_ds['bootstrap_correlations'],
                    'bootstrap_mean':         result_ds['bootstrap_mean'],
                    'bootstrap_std':          result_ds['bootstrap_std'],
                }
                print(f"  Loaded: {p_name} × {sst_name}")
            except Exception as e:
                print(f"  Failed: {p_name} × {sst_name}: {e}")
        else:
            print(f"  Not found: {output_file}")

# Pool all original and randomized correlations across basins and ensemble members
original_vals = []
randomized_vals = []
for (p_name, sst_name), result in results.items():
    if "original_correlation" in result:
        orig = result["original_correlation"]
        if hasattr(orig, "values"):
            orig = orig.values
        original_vals.extend(np.ravel(orig).tolist())
    if "bootstrap_correlations" in result:
        boot = result["bootstrap_correlations"]
        if hasattr(boot, "values"):
            boot = boot.values
        randomized_vals.extend(np.ravel(boot).tolist())

original_vals = np.asarray(original_vals, dtype=float)
randomized_vals = np.asarray(randomized_vals, dtype=float)
original_vals = original_vals[np.isfinite(original_vals)]
randomized_vals = randomized_vals[np.isfinite(randomized_vals)]

mean_original   = float(np.mean(original_vals))
mean_randomized = float(np.mean(randomized_vals))

print(f"  Original   : n={len(original_vals)}, mean={mean_original:.3f}")
print(f"  Randomized : n={len(randomized_vals)}, mean={mean_randomized:.3f}")

# ===========================================================================
# pattern correlations
# ===========================================================================
print("Loading pattern correlation results...")
output_filename = OUTPUTS_DIR / 'pattern_correlations_all_basins.nc'
results_ds = xr.open_dataset(output_filename)

pattern_corr_per_member = results_ds['pattern_corr_per_member']   # (basin, ensemble, window)
rolling_years = results_ds['window'].values

print(f"  Time period  : {results_ds.attrs['time_period']}")
print(f"  Window size  : {results_ds.attrs['window_size_years']} years")
print(f"  Precip dsets : {results_ds.attrs['precip_datasets']}")
print(f"  SST dsets    : {results_ds.attrs['sst_datasets']}")

# ===========================================================================
# FIGURE
# ===========================================================================
plt.rcParams.update({'font.size': 7})

fig = plt.figure(figsize=(7.0, 3.2), dpi=600)
gs = gridspec.GridSpec(
    nrows=1, ncols=2,
    figure=fig,
    wspace=0.30,
)

# Colors
COLOR_ORIGINAL   = 'steelblue'   # match Panel B blue
COLOR_RANDOMIZED = '#E69F00'     # orange

# ---------------------------------------------------------------------------
# Panel A: Distribution of original vs. randomized correlations
# ---------------------------------------------------------------------------
ax_A = fig.add_subplot(gs[0, 0])

# Histograms (density-normalized) with KDE overlays
bins = np.linspace(-1, 1, 61)

ax_A.hist(
    randomized_vals, bins=bins, density=True,
    color=COLOR_RANDOMIZED, alpha=0.45,
    edgecolor=COLOR_RANDOMIZED, linewidth=0.3,
    label='Null', zorder=2,
)
ax_A.hist(
    original_vals, bins=bins, density=True,
    color=COLOR_ORIGINAL, alpha=0.55,
    edgecolor=COLOR_ORIGINAL, linewidth=0.3,
    label='Observed', zorder=3,
)

# KDE overlays for smooth shape
sns.kdeplot(randomized_vals, ax=ax_A, color=COLOR_RANDOMIZED,
            linewidth=1.0, zorder=4)
sns.kdeplot(original_vals, ax=ax_A, color=COLOR_ORIGINAL,
            linewidth=1.0, zorder=5)

# Mean lines
ax_A.axvline(
    mean_randomized, color=COLOR_RANDOMIZED, linestyle='--', linewidth=1.0,
    zorder=6, label=f'Null mean',
)
ax_A.axvline(
    mean_original, color=COLOR_ORIGINAL, linestyle='--', linewidth=1.0,
    zorder=7, label=f'Observed mean',
)
print(f'Original mean = {mean_original:.3f}')
print(f'Randomized mean = {mean_randomized:.3f}')
ax_A.axvline(0, color='grey', alpha=0.5, linestyle='-', linewidth=0.8, zorder=1)

ax_A.set_title('Distribution of SST-Precipitation Correlations')
ax_A.set_xlabel('Correlation')
ax_A.set_ylabel('Density')
ax_A.set_xlim(-1, 1)
ax_A.grid(False)
ax_A.legend(loc='best', fontsize=5, framealpha=0.85)

# ---------------------------------------------------------------------------
# Panel B: Pattern correlation stability
# ---------------------------------------------------------------------------
ax_B = fig.add_subplot(gs[0, 1])

# initialize min/max trackers
global_min = np.inf
global_max = -np.inf

if 'basin' in pattern_corr_per_member.dims and 'ensemble' in pattern_corr_per_member.dims:
    datasets = pattern_corr_per_member.ensemble.values
    first = True
    for ds in datasets:
        member      = pattern_corr_per_member.sel(ensemble=ds)   # (basin, window)
        member_mean = member.mean('basin')
        member_std  = member.std('basin')
        
        lower = member_mean - member_std
        upper = member_mean + member_std

        # update global min/max
        global_min = min(global_min, float(lower.min()))
        global_max = max(global_max, float(upper.max()))

        ax_B.fill_between(
            rolling_years,
            lower,
            upper,
            alpha=0.15, color='steelblue', zorder=2,
            label='±1 SD river basins' if first else None,
        )
        ax_B.plot(
            rolling_years, member_mean,
            color='crimson', linewidth=0.5, alpha=0.6, zorder=10,
            label='Ensemble member mean' if first else None,
        )
        first = False
        
# print results after loop
print(f"Min plotted value: {global_min}")
print(f"Max plotted value: {global_max}")

ax_B.set_ylabel('Pattern Correlation')
ax_B.set_title('Stability of the Marginal Sensitivity Over Time')
ax_B.set_xlabel('Last Year of 30-Year Window')
ax_B.set_ylim([0.4, 1.0])
ax_B.set_xlim(rolling_years.min(), rolling_years.max())

x_ticks = np.arange(int(rolling_years.min()), int(rolling_years.max()) + 1, step=3)
x_ticks = np.append(x_ticks, 2025)
ax_B.set_xticks(x_ticks)
ax_B.tick_params(axis='x', labelsize=6)

ax_B.legend(loc='best', fontsize=5)
ax_B.grid(False)

# ---------------------------------------------------------------------------
# Panel labels
# ---------------------------------------------------------------------------
ax_A.text(-0.14, 1.07, 'a', transform=ax_A.transAxes,
          fontsize=10, fontweight='bold', va='top', ha='left')
ax_B.text(-0.18, 1.07, 'b', transform=ax_B.transAxes,
          fontsize=10, fontweight='bold', va='top', ha='left')

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
plt.tight_layout()
save_path = FIGURES_DIR / "Figure_05_regression_robustness_tests.png"
plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
plt.show()
print(f"Saved → {save_path}")