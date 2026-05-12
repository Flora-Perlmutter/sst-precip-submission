#!/usr/bin/env python
# coding: utf-8
"""
Figure 2: RH conditioning attenuation analysis.

Author: Flora Perlmutter

Description
-----------
Loads RH-conditioned regression results, linear regression results, and
RH-SST correlations to assess whether column relative humidity acts as
a confounder in the SST-precipitation relationship. Produces a 
figure showing: a line plot of attenuation ratios (dP/dSST from RH model) / (dP/dSST from linear model) vs RH-SST correlations

Required data files
---------------------------------------------------
  global_rh_regression_bootstrap_{P}_{SST}.nc   (run_bootstrap_se_rh_obs.py)
  global_linear_regression_bootstrap_{P}_{SST}.nc (run_bootstrap_se_obs.py)
  rh_sst_correlation_{P}_{SST}.nc               (compute_rh_sst_correlation.py)

Output
------
  figures/paper_figures/Figure_02_bad_control.png  (repo-tracked)

"""

import numpy as np
import xarray as xr
import os
import sys
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import DATA_DIR, PAPER_FIGURE_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUTS_DIR = DATA_DIR
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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

# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_rh_results():
    """Load RH conditioning regression results"""
    rh_results = {}
    print("\n" + "="*80)
    print("LOADING RH CONDITIONING RESULTS")
    print("="*80)
    
    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            path = INPUTS_DIR / (
                f'global_rh_regression_bootstrap_{p_name}_{sst_name}.nc'
            )
            
            if path.exists():
                try:
                    result_ds = xr.open_dataset(path)
                    rh_results[(p_name, sst_name)] = result_ds
                    print(f" Loaded: {p_name} vs {sst_name}")
                except Exception as e:
                    print(f" Failed: {p_name} vs {sst_name} - {e}")
            else:
                print(f"  - Missing: {p_name} vs {sst_name}")
    
    print(f"\nTotal loaded: {len(rh_results)}/{len(PRECIP_DATASETS) * len(SST_DATASETS)}")
    return rh_results


def load_linear_results():
    """Load linear regression results"""
    linear_results = {}
    print("\n" + "="*80)
    print("LOADING LINEAR REGRESSION RESULTS")
    print("="*80)
    
    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            output_file = INPUTS_DIR / (
        f'global_linear_regression_bootstrap_{p_name}_{sst_name}.nc'
    )
            
            if os.path.exists(output_file):
                try:
                    result_ds = xr.open_dataset(output_file)
                    
                    result = {
                        'model_id': result_ds.attrs.get('model_id', 'unknown'),
                        'description': result_ds.attrs.get('description', ''),
                        'variable': result_ds.attrs.get('variable', 'sst'),
                        'alpha': result_ds.attrs.get('alpha', 0.05),
                        'n_bootstrap': result_ds.attrs.get('n_bootstrap', 1000),
                        'reconstruction': result_ds['reconstruction'],
                        'reconstruction_se': result_ds['reconstruction_se'],
                        'observed_precip': result_ds['observed_precip'],
                        'correlation': result_ds['correlation'],
                        'marginal_sensitivity_se': result_ds['marginal_sensitivity_se'],
                        'marginal_sensitivity': result_ds['marginal_sensitivity'],
                    }
                    
                    linear_results[(p_name, sst_name)] = result
                    print(f"   Loaded: {p_name} vs {sst_name}")
                    
                except Exception as e:
                    print(f"   Failed: {p_name} vs {sst_name} - {e}")
            else:
                print(f"  - Missing: {p_name} vs {sst_name}")
    
    print(f"\nTotal loaded: {len(linear_results)}/{len(PRECIP_DATASETS) * len(SST_DATASETS)}")
    return linear_results


def load_rh_sst_correlations():
    """Load RH-SST correlation results"""
    rh_sst_corr_dict = {}
    print("\n" + "="*80)
    print("LOADING RH-SST CORRELATIONS")
    print("="*80)
    
    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            corr_file = OUTPUTS_DIR / (
    f'rh_sst_correlation_{p_name}_{sst_name}.nc'
)
            
            if os.path.exists(corr_file):
                try:
                    rh_sst_corr_dict[(p_name, sst_name)] = xr.open_dataset(corr_file)
                    print(f"   Loaded: {p_name} vs {sst_name}")
                except Exception as e:
                    print(f"   Failed: {p_name} vs {sst_name} - {e}")
            else:
                print(f"  - Missing: {p_name} vs {sst_name}")
    
    print(f"\nTotal loaded: {len(rh_sst_corr_dict)}/{len(PRECIP_DATASETS) * len(SST_DATASETS)}")
    return rh_sst_corr_dict


# ============================================================================
# COMPUTE METRICS
# ============================================================================

def compute_attenuation_ratios(rh_results, linear_results):
    """
    Compute the ratio of SST→P correlation in RH-conditioned vs linear models.
    
    Attenuation ratio = MS_RH_Model / MS_Linear_Model
    
    Values < 1 indicate that RH conditioning reduces the SST-P correlation,
    suggesting RH acts as a confounder in the relationship.
    """
    print("\n" + "="*80)
    print("COMPUTING ATTENUATION RATIOS")
    print("="*80)
    
    attenuation_dict = {}
    
    for key in rh_results.keys():
        if key in linear_results:
            corr_rh = rh_results[key]['marginal_sensitivity_sst']
            corr_linear = linear_results[key]['marginal_sensitivity']
            
            if corr_rh is not None and corr_linear is not None:
                attenuation = corr_rh / (corr_linear + 1e-14)
                attenuation_dict[key] = attenuation
                
                mean_attn = float(attenuation.mean())
                median_attn = float(attenuation.median())
                pct_attenuated = float((attenuation < 1).mean() * 100)
                
    
    print(f"\nTotal attenuation ratios computed: {len(attenuation_dict)}")
    return attenuation_dict


# In[2]:


def plot_combined_analysis(attenuation_dict, rh_sst_corr_dict, figures_dir):
    """
    Plot RH-SST correlation vs Attenuation Ratio.
    Each dataset member is plotted as a line (mean across basins binned by 
    RH-SST correlation), with fill_between for ±1 SD across basins.
    Ensemble mean ± SD across all members is overlaid.
    """
    print("\nCreating figure: RH-SST Correlation vs Attenuation Ratio...")
    plt.rcParams.update({'font.size': 7})
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 3.3), dpi=600)

    N_BINS = 10  # number of bins along the RH-SST correlation axis

    # ------------------------------------------------------------------ #
    # Collect all basin-level (corr, attn) pairs across all datasets
    # ------------------------------------------------------------------ #
    all_member_corr_means = []
    all_member_attn_means = []

    # First pass: find global corr range to define shared bin edges
    all_corr_vals_global = []
    for key in attenuation_dict:
        if key not in rh_sst_corr_dict:
            continue
        corr_ds  = rh_sst_corr_dict[key]
        corr_da  = (corr_ds[list(corr_ds.data_vars)[0]]
                    if isinstance(corr_ds, xr.Dataset) else corr_ds)
        corr_vals = corr_da.values.ravel()
        all_corr_vals_global.append(corr_vals[np.isfinite(corr_vals)])

    all_corr_vals_global = np.concatenate(all_corr_vals_global)
    bin_edges   = np.linspace(all_corr_vals_global.min(),
                              all_corr_vals_global.max(), N_BINS + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # ------------------------------------------------------------------ #
    # Second pass: bin each dataset member and plot
    # ------------------------------------------------------------------ #
    # Accumulate binned values across members for ensemble mean/SD
    binned_attn_all_members = []  # shape: (n_members, N_BINS)

    first_member = True
    for key in attenuation_dict:
        if key not in rh_sst_corr_dict:
            continue

        # --- flatten basin-level arrays ---
        attn_da   = attenuation_dict[key]
        attn_vals = attn_da.values.ravel()

        corr_ds   = rh_sst_corr_dict[key]
        corr_da   = (corr_ds[list(corr_ds.data_vars)[0]]
                     if isinstance(corr_ds, xr.Dataset) else corr_ds)
        corr_vals = corr_da.values.ravel()

        # keep only finite, co-valid pairs
        valid = np.isfinite(attn_vals) & np.isfinite(corr_vals)
        attn_vals = attn_vals[valid]
        corr_vals = corr_vals[valid]

        # --- bin by RH-SST correlation value ---
        bin_idx       = np.digitize(corr_vals, bin_edges) - 1
        bin_idx       = np.clip(bin_idx, 0, N_BINS - 1)

        bin_mean = np.full(N_BINS, np.nan)
        bin_std  = np.full(N_BINS, np.nan)

        for b in range(N_BINS):
            mask = bin_idx == b
            if mask.sum() > 1:
                bin_mean[b] = np.nanmean(attn_vals[mask])
                bin_std[b]  = np.nanstd(attn_vals[mask])

        # --- plot each member: fill_between ± SD, line for mean ---
        valid_bins = np.isfinite(bin_mean)
        ax.fill_between(
            bin_centers[valid_bins],
            (bin_mean - bin_std)[valid_bins],
            (bin_mean + bin_std)[valid_bins],
            alpha=0.15, color='steelblue', zorder=2,
            label='±1 SD river basins' if first_member else None,
        )
        ax.plot(
            bin_centers[valid_bins], bin_mean[valid_bins],
            color='crimson', linewidth=0.5, alpha=0.6, zorder=30,
            label='Ensemble member mean' if first_member else None,
        )
        first_member = False

    # ------------------------------------------------------------------ #
    # Reference line and formatting
    # ------------------------------------------------------------------ #
    ax.axhline(1.0, color='red', linewidth=1, linestyle='--', alpha=1, zorder=7, label='No attenuation')
    ax.axhline(y=0, color='grey', alpha=.5, linestyle='-', linewidth=1)

    ax.set_xlabel('RH–SST Correlation')
    ax.set_ylabel('Attenuation Ratio')
    ax.set_title('Attenuation vs RH–SST Correlation')
    ax.legend(loc='best', fontsize=5)
    ax.grid(False)
    ax.set_xlim(min(bin_centers), max(bin_centers))

    plt.tight_layout()
    plt.savefig(
        figures_dir / 'Figure_2_bad_control.png',
        dpi=600, bbox_inches='tight', pad_inches=0.05
    )
    print("Saved: Figure_2_bad_control.png")
    return np.array(all_member_attn_means), np.array(all_member_corr_means)


# In[3]:


def main():
    
    print("\n" + "="*80)
    print("RH CONDITIONING ANALYSIS")
    print("="*80)
    
    # Load all results
    rh_results = load_rh_results()
    linear_results = load_linear_results()
    rh_sst_corr_dict = load_rh_sst_correlations()
    
    if len(rh_results) == 0 or len(linear_results) == 0:
        print("\nERROR: Missing required regression results!")
        print("Please ensure both RH conditioning and linear regression results are available.")
        sys.exit(1)
    
    # Compute attenuation ratios
    attenuation_dict = compute_attenuation_ratios(rh_results, linear_results)
    
    if len(attenuation_dict) == 0:
        print("\nERROR: No attenuation ratios could be computed!")
        sys.exit(1)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"\nPlots will be saved to: {FIGURES_DIR}")
    
    print("\n" + "="*80)
    print("GENERATING COMBINED 3-PANEL FIGURE")
    print("="*80)
    
    if len(rh_sst_corr_dict) > 0:
        attn_values, corr_values = plot_combined_analysis(
            attenuation_dict, rh_sst_corr_dict, FIGURES_DIR
        )
    else:
        print("\nWarning: No RH-SST correlations found. Cannot create complete figure.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nCombined figure saved to: {FIGURES_DIR}/Figure_02_bad_control.png")


if __name__ == "__main__":
    main()


# In[ ]:




