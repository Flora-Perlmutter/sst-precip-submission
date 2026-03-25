#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import xarray as xr
import os
import sys
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# SETUP
# ============================================================================

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
figures_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Figures')

# Define precipitation and SST datasets
PRECIP_DATASETS = [ 'CPC','CRU', 'GPCC','GPCP', 'PREC','REGEN', 'TerraClimate', 'UDel' ]
SST_DATASETS = ['ERSSTv5', 'COBE-SST2']


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
            output_file = os.path.join(
                outputs_dir,
                f'global_rh_regression_bootstrap_{p_name}_{sst_name}.nc'
            )
            
            if os.path.exists(output_file):
                try:
                    result_ds = xr.open_dataset(output_file)
                    rh_results[(p_name, sst_name)] = result_ds
                    print(f"  ✓ Loaded: {p_name} vs {sst_name}")
                except Exception as e:
                    print(f"  ✗ Failed: {p_name} vs {sst_name} - {e}")
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
            output_file = os.path.join(
                outputs_dir,
                f'global_linear_regression_bootstrap_{p_name}_{sst_name}.nc'
            )
            
            if os.path.exists(output_file):
                try:
                    result_ds = xr.open_dataset(output_file)
                    
                    # Reconstruct the result dictionary structure
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
            corr_file = os.path.join(
                outputs_dir, 
                f'rh_sst_correlation_{p_name}_{sst_name}.nc'
            )
            
            if os.path.exists(corr_file):
                try:
                    rh_sst_corr_dict[(p_name, sst_name)] = xr.open_dataarray(corr_file)
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
    
    Parameters:
    -----------
    rh_results : dict
        RH conditioning regression results
    linear_results : dict
        Linear regression results
        
    Returns:
    --------
    dict : Attenuation ratios for each precipitation-SST pair
    """
    print("\n" + "="*80)
    print("COMPUTING ATTENUATION RATIOS")
    print("="*80)
    
    attenuation_dict = {}
    
    for key in rh_results.keys():
        if key in linear_results:
            # Get correlations from both models
            corr_rh = rh_results[key]['marginal_sensitivity_sst']
            corr_linear = linear_results[key]['marginal_sensitivity']
            
            if corr_rh is not None and corr_linear is not None:
                # Attenuation ratio: how much is correlation reduced?
                # Add small epsilon to avoid division by zero
                attenuation = corr_rh / (corr_linear + 1e-14)
                attenuation_dict[key] = attenuation
                
                mean_attn = float(attenuation.mean())
                median_attn = float(attenuation.median())
                pct_attenuated = float((attenuation < 1).mean() * 100)
                
                print(f"  {key[0]:12s} vs {key[1]:10s}: "
                      f"mean={mean_attn:6.3f}, median={median_attn:6.3f}, "
                      f"{pct_attenuated:5.1f}% attenuated")
    
    print(f"\nTotal attenuation ratios computed: {len(attenuation_dict)}")
    return attenuation_dict


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_combined_analysis(attenuation_dict, rh_sst_corr_dict, figures_dir):
    """
    Create a 3-panel figure with:
    Panel A (top left): Attenuation ratio histogram
    Panel B (top right): RH-SST correlation histogram
    Panel C (bottom): Attenuation vs RH-SST correlation scatter plot
    """
    print("\nGenerating combined 3-panel analysis figure...")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 16))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.8], hspace=0.1, wspace=0.1)
    
    # ========================================================================
    # PANEL A: Attenuation Distribution
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Collect attenuation values
    all_attn = []
    for key, attn in attenuation_dict.items():
        values = attn.values.flatten()
        values = values[~np.isnan(values) & ~np.isinf(values)]
        all_attn.extend(values)
    
    all_attn = np.array(all_attn)
    
    if len(all_attn) > 0:
        ax_a.hist(all_attn, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax_a.axvline(x=1, color='red', linestyle='--', linewidth=2, label='No attenuation')
        ax_a.axvline(x=np.median(all_attn), color='orange', linestyle='-', linewidth=2, 
                   label=f'Median = {np.median(all_attn):.2f}')
        
        ax_a.set_xlabel('Attenuation Ratio', fontsize=12)
        ax_a.set_ylabel('Frequency', fontsize=12)
        ax_a.set_title('Distribution of Attenuation Ratios\ndP/dSST (RH Model) / dP/dSST (Linear Model)', 
                      fontsize=12, fontweight='bold')
        ax_a.legend(fontsize=10)
        ax_a.grid(True, alpha=0.3)
        ax_a.set_ylim(0, 3.2e5)
        ax_a.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax_a.text(0.5, 0.5, 'No valid attenuation values', 
                 ha='center', va='center', transform=ax_a.transAxes)
    
    # ========================================================================
    # PANEL B: RH-SST Correlation Distribution
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
        
    # Collect correlation values — built jointly with attenuation to match scatter plot exactly
    all_corr = []
    all_attn_for_hist = []
    for key in attenuation_dict.keys():
        if key in rh_sst_corr_dict:
            attn = attenuation_dict[key].values.flatten()
            corr = rh_sst_corr_dict[key].values.flatten()
            valid_mask = ~(np.isnan(attn) | np.isnan(corr) |
                          np.isinf(attn) | np.isinf(corr))
            all_corr.extend(corr[valid_mask])
            all_attn_for_hist.extend(attn[valid_mask])

    all_corr = np.array(all_corr)
    
    all_corr = np.array(all_corr)
    
    if len(all_corr) > 0:
        ax_b.hist(all_corr, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
        ax_b.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No correlation')
        ax_b.set_xlabel('RH-SST Correlation', fontsize=12)
        ax_b.set_ylabel('Frequency', fontsize=12)
        ax_b.set_title('Distribution of RH-SST Correlations', 
                      fontsize=12, fontweight='bold')
        ax_b.legend(fontsize=10)
        ax_b.grid(True, alpha=0.3)
        ax_b.set_ylim(0, 3.2e5)
        ax_b.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    else:
        ax_b.text(0.5, 0.5, 'No valid correlation values', 
                 ha='center', va='center', transform=ax_b.transAxes)
    
    # ========================================================================
    # PANEL C: Scatter Plot
    # ========================================================================
    ax_c = fig.add_subplot(gs[1, :])
    
    # Collect matching pairs
    attn_values = []
    corr_values = []
    
    for key in attenuation_dict.keys():
        if key in rh_sst_corr_dict:
            attn = attenuation_dict[key].values.flatten()
            corr = rh_sst_corr_dict[key].values.flatten()
            
            # Filter out nan and inf
            valid_mask = ~(np.isnan(attn) | np.isnan(corr) | 
                          np.isinf(attn) | np.isinf(corr))
            
            attn_values.extend(attn[valid_mask])
            corr_values.extend(corr[valid_mask])
    
    attn_values = np.array(attn_values)
    corr_values = np.array(corr_values)
    
    if len(attn_values) > 0:
        # Scatter plot with transparency
        ax_c.scatter(corr_values, attn_values, alpha=0.05, s=1, c='steelblue', rasterized=True)
        
        # Add reference lines
        ax_c.axhline(y=1, color='red', linestyle='--', linewidth=2, 
                   label='No attenuation', zorder=10)
        ax_c.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add binned statistics overlay
        n_bins = 20
        bin_edges = np.linspace(corr_values.min(), corr_values.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_means = []
        bin_stds = []
        
        for i in range(n_bins):
            mask = (corr_values >= bin_edges[i]) & (corr_values < bin_edges[i+1])
            if mask.sum() > 0:
                bin_means.append(attn_values[mask].mean())
                bin_stds.append(attn_values[mask].std())
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)
        
        bin_means = np.array(bin_means)
        bin_stds = np.array(bin_stds)
        
        # Plot binned means
        valid = ~np.isnan(bin_means)
        ax_c.plot(bin_centers[valid], bin_means[valid], 'o-', color='darkred', 
                linewidth=2, markersize=6, label='Binned mean', zorder=11)
        ax_c.fill_between(bin_centers[valid], 
                         bin_means[valid] - bin_stds[valid],
                         bin_means[valid] + bin_stds[valid],
                         alpha=0.2, label="±1 SD", color='darkred', zorder=9)
        
        ax_c.set_xlabel('RH-SST Correlation', fontsize=12)
        ax_c.set_ylabel('Attenuation Ratio', fontsize=12)
        ax_c.set_title('Attenuation vs RH-SST Correlation',
                     fontsize=12, fontweight='bold')
        ax_c.legend(fontsize=10)
        ax_c.grid(True, alpha=0.3)
    else:
        ax_c.text(0.5, 0.5, 'No valid paired values', 
                 ha='center', va='center', transform=ax_c.transAxes)
        
    for ax, label in zip([ax_a, ax_b, ax_c], 
                     ['a', 'b', 'c']):
        ax.text(-0.05, 1.08, label, transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    # Save combined figure
    output_path = os.path.join(figures_dir, 'Figure_2_bad_control.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()
    
    return all_attn, all_corr


# In[12]:


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    
    print("\n" + "="*80)
    print("RH CONDITIONING ANALYSIS")
    print("="*80)
    
    # Load all results
    rh_results = load_rh_results()
    linear_results = load_linear_results()
    rh_sst_corr_dict = load_rh_sst_correlations()
    
    # Check if we have data
    if len(rh_results) == 0 or len(linear_results) == 0:
        print("\nERROR: Missing required regression results!")
        print("Please ensure both RH conditioning and linear regression results are available.")
        sys.exit(1)
    
    # Compute attenuation ratios
    attenuation_dict = compute_attenuation_ratios(rh_results, linear_results)
    
    if len(attenuation_dict) == 0:
        print("\nERROR: No attenuation ratios could be computed!")
        sys.exit(1)
    
    # Create output directory for plots
    plot_dir = os.path.join(figures_dir)
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {plot_dir}")
    
    # Generate combined 3-panel plot
    print("\n" + "="*80)
    print("GENERATING COMBINED 3-PANEL FIGURE")
    print("="*80)
    
    if len(rh_sst_corr_dict) > 0:
        attn_values, corr_values = plot_combined_analysis(attenuation_dict, rh_sst_corr_dict, plot_dir)
    else:
        print("\nWarning: No RH-SST correlations found. Cannot create complete figure.")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nCombined figure saved to: {plot_dir}/rh_analysis_combined_v2.png")


if __name__ == "__main__":
    main()


# In[ ]:




