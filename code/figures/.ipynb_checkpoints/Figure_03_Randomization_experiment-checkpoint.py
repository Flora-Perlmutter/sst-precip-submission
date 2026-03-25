#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import cartopy.feature as cfeature
import regionmask
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import cartopy
import seaborn as sns
import xskillscore
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import geopandas as gpd
import seaborn as sns
import xarray as xr
import matplotlib.gridspec as gridspec
from matplotlib.cm import RdBu
from scipy.stats import linregress
from scipy import stats
from scipy.stats import t
import pickle
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import numpy as np
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import warnings
import glob
from pathlib import Path
warnings.filterwarnings("ignore")


# In[2]:


##-------------------------------------------------------------------------------##

#Define directories

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Scripts')
data_dir = os.path.join(root_dir, 'Data/Observations')
amip_data_dir='/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/CAM4_Greens_Project/Data/Raw'
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
figures_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Figures/Paper_Figures')
        
##-------------------------------------------------------------------------------##

outputs_dir = '/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/Observational_Regressions_Project/Data/Processed'

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

SST_DATASETS = ['ERSSTv5', 'COBE-SST2']

# Create nested dictionary to store all results
results = {}

print("Loading bootstrap results...")
for p_name in PRECIP_DATASETS.keys():
    for sst_name in SST_DATASETS:
        output_file = os.path.join(
            outputs_dir,
            f'randomization_experiment_{p_name}_{sst_name}.nc'
        )
        
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
            


# In[3]:


# -------------------------------------------------
# Gather data for box plot
# -------------------------------------------------
plot_data = []
for key, result in results.items():
    p_name, sst_name = key
    dataset_label = f"{p_name}-{sst_name}"
    
    # Bootstrapped/randomization correlations
    if "bootstrap_correlations" in result:
        for val in np.ravel(result["bootstrap_correlations"]):
            plot_data.append({
                "Dataset": dataset_label,
                "Correlation": float(val),
                "Type": "Randomized"
            })
    
    # Original correlation (ensure scalar)
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
# APPLY NAME MAPPING HERE (BEFORE PLOTTING!)
# -------------------------------------------------
name_mapping = {
    "CRU-ERSSTv5": "CRU TS-ERSSTv5",
    "CRU-COBE-SST2": "CRU TS-COBE-SST2",
    "PREC-ERSSTv5": "PREC/L-ERSSTv5",
    "PREC-COBE-SST2": "PREC/L-COBE-SST2",
    "UDel-ERSSTv5": "UDEL-TS-ERSSTv5",
    "UDel-COBE-SST2": "UDEL-TS-COBE-SST2",
}
df_plot["Dataset"] = df_plot["Dataset"].map(lambda x: name_mapping.get(x, x))

# -------------------------------------------------
# Compute overall mean of original correlations
# -------------------------------------------------
mean_original_corr = df_plot.loc[df_plot["Type"] == "Original", "Correlation"].mean()

# -------------------------------------------------
# Create box plot (NOW with renamed datasets)
# -------------------------------------------------
plt.figure(figsize=(max(8, len(results) * 0.8), 6))
sns.boxplot(
    data=df_plot[df_plot["Type"] == "Randomized"],
    x="Dataset",
    y="Correlation",
    width=0.6,
    linewidth=1,
    fliersize=2,
    color="skyblue",
    boxprops=dict(alpha=0.6)
)

# Overlay original (unshuffled) correlation for each dataset
orig_df = df_plot[df_plot["Type"] == "Original"]
plt.scatter(
    x=np.arange(len(orig_df)),
    y=orig_df["Correlation"],
    color="orange",
    s=80,
    edgecolor="black",
    linewidth=1.2,
    zorder=10,
    label="Observed Correlation"
)

# Add green dashed line for mean original correlation
plt.axhline(
    mean_original_corr,
    color="green",
    linestyle="--",
    linewidth=2,
    alpha=0.8,
    label=f"Observed Mean (ρ={mean_original_corr:.3f})"
)
plt.axhline(
    0,
    color="black",
    linewidth=2,
    alpha=0.8,
    zorder=0
)

# -------------------------------------------------
# Format
# -------------------------------------------------
plt.title("Randomization Experiment Across Ensemble Members", fontsize=14, fontweight="bold")
plt.xlabel("Ensemble Member", fontsize=12)
plt.ylabel("Correlation", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

# Save and show
save_path = os.path.join(figures_dir, "Figure_3_randomization_boxplot.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
print(f" Saved combined box plot to {save_path}")


# In[ ]:




