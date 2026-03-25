#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
# In[7]:
import numpy as np
import xarray as xr
from tqdm import tqdm
import gc
import os
import sys
import pickle
import argparse
import psutil
from joblib import Parallel, delayed
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import json
warnings.filterwarnings("ignore")

# ============================================================================
# SETUP
# ============================================================================
root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
data_dir = os.path.join(root_dir, 'Data/Observations')
figures_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Figures')  # Define figures directory
os.makedirs(figures_dir, exist_ok=True)

sys.path.append(functions_dir)
from regression_functions import detrend_dim

# Load dataset names
names_path = os.path.join(outputs_dir, "amip_dataset_names.json")
with open(names_path, "r") as f:
    names = json.load(f)

# Load SST anomalies using manifest
sst_dict = {}
for sst_name in names['sst_datasets']:
    nc_path = os.path.join(outputs_dir, f"sst_anom_{sst_name}.nc")
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path)
    else:
        print(f"Warning: {nc_path} not found")
    



# In[5]:


# ----------------------------
# Autocorrelation and PACF calculation
# ----------------------------
def calculate_autocorr(data, max_lag=24, threshold=1/np.e):
    """
    Calculate temporal autocorrelation, partial autocorrelation, and determine block length.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Time series data [time, ...]
    max_lag : int
        Maximum lag to consider
    threshold : float
        Threshold for determining significant autocorrelation (default: 1/e ≈ 0.37)
        
    Returns:
    --------
    block_length : int
        Suggested block length for block bootstrap
    autocorr : np.array
        Autocorrelation values at each lag
    pacf : np.array
        Partial autocorrelation values at each lag
    """
    from statsmodels.tsa.stattools import acf, pacf
    
    # Flatten spatial dimensions and compute mean autocorrelation
    if len(data.shape) > 1:
        # Average over space
        data_mean = data.mean(dim=[d for d in data.dims if d != 'time'])
    else:
        data_mean = data
    
    # Remove NaNs
    data_clean = data_mean.dropna(dim='time')
    
    # Calculate autocorrelation
    autocorr = acf(data_clean.values, nlags=max_lag, fft=True)
    
    # Find first lag where autocorrelation drops below threshold
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    if len(below_threshold) > 0:
        block_length = below_threshold[0]
    else:
        block_length = max_lag
    
    # Ensure minimum block length of 1
    block_length = max(1, block_length)
    
    return block_length, autocorr

# ----------------------------
# Calculate autocorrelation and PACF for both datasets
# ----------------------------
print("Calculating temporal autocorrelation and PACF for SST datasets...")

results_dict = {}
for dataset_name, data in sst_dict.items():
    print(f"\nProcessing {dataset_name}...")
    data_constrained = data.sel(time=slice('1981-01-01', '2014-12-01'))
    data_detrended=detrend_dim(data_constrained, 'time')
    block_length, autocorr = calculate_autocorr(data_detrended, max_lag=24)
    
    results_dict[dataset_name] = {
        'block_length': block_length,
        'autocorr': autocorr,
        'lags': np.arange(len(autocorr))
    }
    
    print(f"  Suggested block length: {block_length} months")
    print(f"  Autocorrelation at lag 1: {autocorr[1]:.3f}")


# Convert numpy arrays to lists for JSON serialization
results_dict_json = {}
for dataset_name, data in results_dict.items():
    results_dict_json[dataset_name] = {
        'block_length': int(data['block_length']),
        'autocorr': data['autocorr'].tolist(),
        'lags': data['lags'].tolist()
    }

# Save
results_path = os.path.join(outputs_dir, "autocorr_block_length_results_amip.json")
with open(results_path, 'w') as f:
    json.dump(results_dict_json, f, indent=2)
print(f"✓ Saved autocorrelation results to: {results_path}")


# In[6]:


results_dict_json


# In[8]:


# ----------------------------
# Plot ACF for both datasets
# ----------------------------
dataset_names = list(results_dict.keys())
fig, axes = plt.subplots(len(dataset_names), 1, figsize=(7, 30))
for idx, dataset_name in enumerate(dataset_names):
    results = results_dict[dataset_name]
    lags = results['lags']
    autocorr = results['autocorr']
    block_length = results['block_length']
    
    ax_acf = axes[idx]
    ax_acf.bar(lags, autocorr, color='steelblue', alpha=0.7)
    ax_acf.axhline(1/np.e, color='red', linestyle='--', linewidth=2, 
                   label=f'1/e threshold ({1/np.e:.3f})')
    ax_acf.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax_acf.axvline(block_length, color='green', linestyle='--', linewidth=2, 
                   label=f'Block length = {block_length}')
    ax_acf.set_xlabel('Lag (months)', fontsize=11)
    ax_acf.set_ylabel('Autocorrelation', fontsize=11)
    ax_acf.set_title(f'{dataset_name} - ACF', fontsize=12, fontweight='bold')
    ax_acf.legend(fontsize=9)
    ax_acf.grid(alpha=0.3)


plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "sst_acf_block_length_determination_amip.png"), 
            dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




