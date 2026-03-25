#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ============================================================================
# SETUP
# ============================================================================

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
figures_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Figures/Paper_Figures')


# ============================================================================
# LOAD DATA
# ============================================================================

# Load the results
output_filename = os.path.join(outputs_dir, 'pattern_correlations_all_basins.nc')
results_ds = xr.open_dataset(output_filename)

# Access individual variables
pattern_corr_means_da = results_ds['pattern_corr_mean']
pattern_corr_stds = results_ds['pattern_corr_std']
full_period_beta = results_ds['full_period_beta']
rolling_years = results_ds['window'].values

# Access metadata
print(f"Time period: {results_ds.attrs['time_period']}")
print(f"Window size: {results_ds.attrs['window_size_years']} years")
print(f"Precip datasets: {results_ds.attrs['precip_datasets']}")
print(f"SST datasets: {results_ds.attrs['sst_datasets']}")


# In[2]:


# ============================================================================
# FIGURE: PATTERN CORRELATION FOR ALL BASINS (SINGLE PANEL)
# ============================================================================

# Correlation of a 30 year marginal sensitivity estimate with the marginal sensitivity of the basin estimated from the entire dataset

print("\nCreating figure: Pattern correlations for all basins (improved visualization)...")
plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(1, 1, figsize=(8, 8))



basins = pattern_corr_means_da.basin.values if 'basin' in pattern_corr_means_da.dims else None

if basins is not None:
    
    # ============ Percentile bands ============
    ax = axes
    
    corr_data = np.array([
        pattern_corr_means_da.sel(basin=basin).values 
        for basin in basins
    ])
    
    # Calculate percentiles across basins at each time point
    p5 = np.percentile(corr_data, 5, axis=0)
    p25 = np.percentile(corr_data, 25, axis=0)
    p50 = np.percentile(corr_data, 50, axis=0)
    p75 = np.percentile(corr_data, 75, axis=0)
    p95 = np.percentile(corr_data, 95, axis=0)
    
    ax.fill_between(rolling_years, p5, p95, alpha=0.2, color='blue', label='5th–95th percentile')
    ax.fill_between(rolling_years, p25, p75, alpha=0.4, color='blue', label='25th–75th percentile')
    ax.plot(rolling_years, p50, color='darkblue', linewidth=2.5, label='Median')
    
    ax.set_ylabel('Pattern Correlation')
    ax.set_xlabel('Last Year of 30-Year Window')
    ax.set_ylim([0, 1.0])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    #ax.set_title('Percentile Bands Across Basins', fontsize=12, fontweight='bold')
    

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, f'Figure_4_marginal_sensitivity_stability.png'), 
            dpi=300, bbox_inches='tight', pad_inches=0.05)
print("Saved: Figure_S5_marginal_sensitivity_stability.png")
print("\n" + "="*80)
print("Analysis complete!")
print("="*80)


# In[7]:


print(p25)


# In[8]:


print(p75)


# In[4]:


for sst_name, sst_anom in sst_dict.items():
    for p_name, p_anom in precip_dict.items():
        print(f"  Processing: {p_name} vs {sst_name}")
        
        # Find common time across basin dimension
        common_time = np.intersect1d(p_anom['time'].values, sst_anom['time'].values)
        
        p_common = p_anom.sel(time=common_time)
        sst_common = sst_anom.sel(time=common_time)
        
        print(common_time.min(), common_time.max())


# In[ ]:




