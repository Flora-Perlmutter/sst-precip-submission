#!/usr/bin/env python
# coding: utf-8
"""
Figure 3: Randomization experiment — bootstrap null distribution.

Author: Flora Perlmutter

Description
-----------
Boxplot of bootstrap null distributions (SST resampled, precipitation
held fixed) versus the original SST-precipitation correlation for each
precipitation-SST ensemble member. Tests the null hypothesis of no
temporal relationship between SST and basin precipitation.

  Blue boxes  : distribution of correlations under random SST resampling
  Orange dots : original (unshuffled) correlation for each member
  Green dashed: mean original correlation across all members

Required data files
-------------------------------
  randomization_experiment_{P}_{SST}.nc   (run_bootstrap_randomization.py)

Output
------
  figures/paper_figures/Figure_3_randomization_experiment.png
"""

import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import matplotlib.gridspec as gridspec
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

 
##-------------------------------------------------------------------------------##

# Define precipitation datasets
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

# Create nested dictionary to store all results
results = {}

print("Loading bootstrap results...")
for p_name in PRECIP_DATASETS.keys():
    for sst_name in SST_DATASETS:
        output_file = OUTPUTS_DIR / f'randomization_experiment_{p_name}_{sst_name}.nc'
        
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
                    'original_correlation': result_ds['original_correlation'],
                    'bootstrap_correlations': result_ds['bootstrap_correlations'],
                    'bootstrap_mean': result_ds['bootstrap_mean'],
                    'bootstrap_std': result_ds['bootstrap_std'],
                }
                
                # Store with key (p_name, sst_name)
                results[(p_name, sst_name)] = result
                
                print(f"Loaded: {p_name} vs {sst_name}")
                
            except Exception as e:
                print(f"Failed to load {p_name} vs {sst_name}: {e}")
        else:
            print(f"Not found: {output_file}")
            

# -------------------------------------------------
# Gather data for box plot
# -------------------------------------------------
plot_data = []
for key, result in results.items():
    p_name, sst_name = key
    dataset_label = f"{p_name}-{sst_name}"
    
    if "bootstrap_correlations" in result:
        for val in np.ravel(result["bootstrap_correlations"]):
            plot_data.append({
                "Dataset": dataset_label,
                "Correlation": float(val),
                "Type": "Randomized"
            })
    
    if "original_correlation" in result:
        orig_corr = result["original_correlation"]
        if hasattr(orig_corr, "values"):
            orig_corr = orig_corr.values
        if np.ndim(orig_corr) > 0:
            orig_corr = np.nanmean(orig_corr)
        plot_data.append({
            "Dataset": dataset_label,
            "Correlation": float(orig_corr),
            "Type": "Original"
        })

df_plot = pd.DataFrame(plot_data)

# -------------------------------------------------
# Apply name mapping
# -------------------------------------------------
name_mapping = {
    "CRU-ERSSTv6": "CRU TS-ERSSTv6",
    "CRU-COBE-SST3": "CRU TS-COBE-SST3",
    "PREC-ERSSTv6": "PREC/L-ERSSTv6",
    "PREC-COBE-SST3": "PREC/L-COBE-SST3",
    "UDel-ERSSTv6": "UDEL-TS-ERSSTv6",
    "UDel-COBE-SST3": "UDEL-TS-COBE-SST3",
}
df_plot["Dataset"] = df_plot["Dataset"].map(lambda x: name_mapping.get(x, x))

# -------------------------------------------------
# Compute overall mean of original correlations
# -------------------------------------------------
mean_original_corr = df_plot.loc[df_plot["Type"] == "Original", "Correlation"].mean()

# -------------------------------------------------
# Create figure
# -------------------------------------------------
plt.rcParams.update({'font.size': 7})
fig = plt.figure(figsize=(5.6, 4.1), dpi=600)
gs = gridspec.GridSpec(1, 1, figure=fig)

# -------------------------------------------------
# Panel A: Box plot
# -------------------------------------------------
ax_box = fig.add_subplot(gs[0, 0])

sns.boxplot(
    data=df_plot[df_plot["Type"] == "Randomized"],
    x="Dataset",
    y="Correlation",
    ax=ax_box,
    width=0.6,
    linewidth=.5,
    fliersize=1,
    color="lightblue",
)


# Get the category order seaborn used for the x-axis
category_order = [tick.get_text() for tick in ax_box.get_xticklabels()]
position_map = {name: i for i, name in enumerate(category_order)}

# Overlay original correlations
orig_df = df_plot[df_plot["Type"] == "Original"].copy()
orig_df["x_pos"] = orig_df["Dataset"].map(position_map)

ax_box.scatter(
    x=orig_df["x_pos"],
    y=orig_df["Correlation"],
    color="orange",
    s=40,
    edgecolor="black",
    linewidth=.5,
    zorder=10,
    label="Observed correlation"
)

ax_box.axhline(mean_original_corr, color="green", linestyle="--", linewidth=1,
               alpha=1, label=f"Observed mean (ρ={mean_original_corr:.3f})")
ax_box.axhline(y=0, color='grey', alpha=0.5, linestyle='-', linewidth=1)

ax_box.set_title("Randomization Experiment Across Ensemble Members")
ax_box.set_xlabel("Ensemble Member")
ax_box.set_ylabel("Correlation")
ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha="right")
ax_box.set_ylim(-.9, .9)
ax_box.legend(fontsize=5)
plt.grid(False)


fig.suptitle("")



# -------------------------------------------------
# Save
# -------------------------------------------------
plt.tight_layout()
save_path = FIGURES_DIR / "Figure_3_linear_validation.png"
plt.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
plt.show()
print(f"Saved: {save_path}")