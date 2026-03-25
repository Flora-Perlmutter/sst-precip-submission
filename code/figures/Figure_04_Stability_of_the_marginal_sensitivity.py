#!/usr/bin/env python
# coding: utf-8
"""
Figure 4: Marginal sensitivity pattern correlation stability.

Author: Flora Perlmutter

Description
-----------
Plots the correlation between each 30-year rolling window marginal
sensitivity estimate and the full-period ensemble mean sensitivity,
across all basins. Shaded percentile bands (5th–95th, 25th–75th)
and the median show how stable the SST-precipitation sensitivity
pattern is across different 30-year periods.

Required data files
-------------------------------
  pattern_correlations_all_basins.nc   (compute_pattern_correlations.py)

Output
------
  figures/paper_figures/Figure_4_marginal_sensitivity_stability.png

"""

import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
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

# ============================================================================
# LOAD DATA
# ============================================================================

# Load the results
output_filename = OUTPUTS_DIR / 'pattern_correlations_all_basins.nc'
results_ds = xr.open_dataset(output_filename)


# Access individual variables
pattern_corr_per_member = results_ds['pattern_corr_per_member']  # (window, ensemble)
full_period_beta = results_ds['full_period_beta']
rolling_years = results_ds['window'].values

# Access metadata
print(f"Time period: {results_ds.attrs['time_period']}")
print(f"Window size: {results_ds.attrs['window_size_years']} years")
print(f"Precip datasets: {results_ds.attrs['precip_datasets']}")
print(f"SST datasets: {results_ds.attrs['sst_datasets']}")

# ============================================================================
# FIGURE: PATTERN CORRELATION PER DATASET, DISTRIBUTION ACROSS BASINS
# ============================================================================
print("\nCreating figure: Pattern correlations per dataset, distribution across basins...")
plt.rcParams.update({'font.size': 7})
fig, axes = plt.subplots(1, 1, figsize=(3.3, 3.3), dpi=600)
ax = axes

if 'basin' in pattern_corr_per_member.dims and 'ensemble' in pattern_corr_per_member.dims:
    datasets = pattern_corr_per_member.ensemble.values

    # Accumulate per-dataset basin means for ensemble statistics
    all_dataset_means = []

    first = True
    for ds in datasets:
        member = pattern_corr_per_member.sel(ensemble=ds)   # dims: (basin, rolling_year)

        member_mean = member.mean('basin')               # mean across basins
        member_std  = member.std('basin')                # spread across basins

        all_dataset_means.append(member_mean.values)

        # Each dataset member: fill_between ± 1 SD across basins + line for mean
        ax.fill_between(
            rolling_years,
            member_mean - member_std,
            member_mean + member_std,
            alpha=0.15, color='steelblue', zorder=2,
            label='±1 SD river basins' if first else None,
        )
        ax.plot(
            rolling_years, member_mean,
            color='crimson', linewidth=0.5, alpha=0.6, zorder=10,
            label='Ensemble member mean' if first else None,
        )
        first = False

ax.set_ylabel('Pattern Correlation')
ax.set_title('Stability of the Marginal Sensitivity Over Time')
ax.set_xlabel('Last Year of 30-Year Window')
ax.set_ylim([.4, 1.0])
ax.set_xlim(rolling_years.min(), rolling_years.max())
# Set integer ticks and left-align the labels
x_ticks = np.arange(int(rolling_years.min()), int(rolling_years.max()) + 1, step=3)  # adjust step as needed
x_ticks = np.append(x_ticks, 2025)
ax.set_xticks(x_ticks)

ax.legend(loc='best', fontsize=5)
ax.grid(False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'Figure_4_marginal_sensitivity_stability.png',
            dpi=600, bbox_inches='tight', pad_inches=0.05)
print("Saved: Figure_4_marginal_sensitivity_stability.png")
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)