#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8
# In[7]:
import numpy as np
import xarray as xr
from tqdm import tqdm
import gc
import os
import sys
import argparse
import psutil
from joblib import Parallel, delayed
import glob
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
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
    
# Load SST anomalies
sst_dict = {}
for sst_name in ["ERSSTv5", "COBE-SST2"]:
    nc_path = os.path.join(outputs_dir, f"sst_anom_{sst_name}.nc")
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path)

# ----------------------------
# Autocorrelation and PACF calculation
# ----------------------------
def calculate_autocorr_and_pacf(data, max_lag=24, threshold=1/np.e):
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
    
    # Calculate partial autocorrelation
    pacf_values = pacf(data_clean.values, nlags=max_lag, method='ywm')
    
    # Find first lag where autocorrelation drops below threshold
    below_threshold = np.where(np.abs(autocorr) < threshold)[0]
    if len(below_threshold) > 0:
        block_length = below_threshold[0]
    else:
        block_length = max_lag
    
    # Ensure minimum block length of 1
    block_length = max(1, block_length)
    
    return block_length, autocorr, pacf_values

# ----------------------------
# Calculate autocorrelation and PACF for both datasets
# ----------------------------
print("Calculating temporal autocorrelation and PACF for SST datasets...")

results_dict = {}
for dataset_name, data in sst_dict.items():
    print(f"\nProcessing {dataset_name}...")
    data_constrained = data.sel(time=slice('1981-01-01', '2019-12-01'))
    data_detrended=detrend_dim(data_constrained, 'time')
    block_length, autocorr, pacf_values = calculate_autocorr_and_pacf(data_detrended, max_lag=24)
    
    results_dict[dataset_name] = {
        'block_length': block_length,
        'autocorr': autocorr,
        'pacf': pacf_values,
        'lags': np.arange(len(autocorr))
    }
    
    print(f"  Suggested block length: {block_length} months")
    print(f"  Autocorrelation at lag 1: {autocorr[1]:.3f}")
    print(f"  PACF at lag 1: {pacf_values[1]:.3f}")

# ----------------------------
# Plot ACF and PACF for both datasets
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
dataset_names = list(results_dict.keys())

for idx, dataset_name in enumerate(dataset_names):
    results = results_dict[dataset_name]
    lags = results['lags']
    autocorr = results['autocorr']
    pacf_values = results['pacf']
    block_length = results['block_length']
    
    # Plot ACF (left column)
    ax_acf = axes[idx, 0]
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
    
    # Plot PACF (right column)
    ax_pacf = axes[idx, 1]
    ax_pacf.bar(lags, pacf_values, color='coral', alpha=0.7)
    ax_pacf.axhline(0, color='black', linestyle='-', linewidth=0.5)
    # Add confidence intervals (approximate 95% CI)
    ci = 1.96 / np.sqrt(len(data.time))
    ax_pacf.axhline(ci, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, 
                    label=f'95% CI (±{ci:.3f})')
    ax_pacf.axhline(-ci, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    ax_pacf.set_xlabel('Lag (months)', fontsize=11)
    ax_pacf.set_ylabel('Partial Autocorrelation', fontsize=11)
    ax_pacf.set_title(f'{dataset_name} - PACF', fontsize=12, fontweight='bold')
    ax_pacf.legend(fontsize=9)
    ax_pacf.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "sst_acf_pacf_comparison.png"), 
            dpi=300, bbox_inches='tight')
plt.show()

# ----------------------------
# Save results
# ----------------------------
output_file = os.path.join(outputs_dir, "sst_autocorr_pacf_results.pkl")
with open(output_file, "wb") as f:
    pickle.dump(results_dict, f)
print(f"\nResults saved to: {output_file}")

print("\nSummary:")
print("-" * 60)
for dataset_name, results in results_dict.items():
    print(f"{dataset_name}:")
    print(f"  Block length: {results['block_length']} months")
    print(f"  ACF(1): {results['autocorr'][1]:.3f}")
    print(f"  PACF(1): {results['pacf'][1]:.3f}")
    print()


# In[4]:


results_dict


# In[3]:


# ----------------------------
# Plot ACF for both datasets
# ----------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 10))
dataset_names = list(results_dict.keys())

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
plt.savefig(os.path.join(figures_dir, "sst_acf_block_length_determination.png"), 
            dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




