# coding: utf-8
"""
Bootstrap randomization experiment for SST-precipitation regression.

Author: Flora Perlmutter

Description
-----------
For each precipitation-SST pair, runs a block bootstrap randomization
experiment to build a null distribution of reconstruction-precipitation
correlations. Unlike run_bootstrap_se_obs.py (which resamples both SST
and precip together to estimate SE), this script resamples only SST
while keeping precip fixed — generating a distribution of correlations
under the null hypothesis of no temporal relationship.

Designed for sbatch array submission — one job per pair via --pair-index.

"""

import argparse
import gc
import sys
import warnings
import os
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import detrend_dim, grid_area, fdr_correction, regression_slope_se

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OUTPUTS_DIR = DATA_DIR

# ============================================================================
# SETUP
# ============================================================================

# Define precipitation datasets (must match what was saved)
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

# Configuration
ALPHA = 0.05
N_BOOTSTRAP = 100
BLOCK_SIZE = 11
RANDOM_SEED = 42


# ============================================================
# Parallelized Bootstrap Loop using joblib
# ============================================================

def run_single_bootstrap(seed, n_time, sst_detrended, p_detrended, sst_anom, p_anom,
                         area_precomputed, alpha, n_blocks, block_size):
    """
    Run a single bootstrap iteration with block resampling.
    Returns the correlation between reconstruction and precipitation anomaly.
    """
    np.random.seed(seed)
    
    # Block bootstrap resampling
    block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
    resampled_indices = []
    for block_idx in block_indices:
        start = block_idx * block_size
        end = min(start + block_size, n_time)
        resampled_indices.extend(range(start, end))
    
    # Ensure we have exactly n_time points
    if len(resampled_indices) > n_time:
        resampled_indices = resampled_indices[:n_time]
    elif len(resampled_indices) < n_time:
        # If short, sample additional indices to fill
        shortage = n_time - len(resampled_indices)
        additional = np.random.choice(n_time, size=shortage, replace=True)
        resampled_indices.extend(additional.tolist())

    # Resample SST data
    sst_boot = sst_detrended.isel(time=resampled_indices)
    sst_boot = sst_boot.assign_coords(time=p_detrended.time)

    # Run regressions
    slope_boot, _, pval_boot = xr.apply_ufunc(
        regression_slope_se,
        sst_boot,
        p_detrended,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32]
    )

    # Apply area weighting (precomputed)
    slope_area_boot = area_precomputed * slope_boot

    # FDR correction
    fdr_mask_boot = fdr_correction(pval_boot, alpha_FDR=alpha)
    slope_sig_boot = slope_area_boot.where(fdr_mask_boot, 0.0)

    # Reconstruction using original (non-resampled) SST anomalies
    reconstruction_boot = (sst_anom * slope_sig_boot).sum(('lat', 'lon'))
    
    # Calculate correlation between reconstruction and original precipitation anomaly
    corr_boot = xr.corr(reconstruction_boot, p_anom, 'time').values.astype(np.float32)
    
    return corr_boot


# ============================================================================
# BOOTSTRAP CORRELATION DISTRIBUTION
# ============================================================================

def bootstrap_correlation_distribution(sst_detrended, p_detrended, sst_anom, p_anom,
                                        area_precomputed, alpha, n_bootstrap=1000, 
                                        block_size=11, random_seed=42):
    """
    Compute distribution of correlations from bootstrap resampling.
    
    Parameters
    ----------
    sst_detrended : xr.DataArray
        SST detrended data (time, lat, lon) [float32]
    p_detrended : xr.DataArray
        Precipitation detrended data (time, basin) [float32]
    sst_anom : xr.DataArray
        SST anomalies for reconstruction [float32]
    p_anom : xr.DataArray
        Precipitation anomalies [float32]
    area_precomputed : xr.DataArray
        Precomputed area weights (lat, lon, basin) [float32]
    alpha : float
        FDR alpha threshold
    n_bootstrap : int
        Number of bootstrap iterations
    block_size : int
        Block size for temporal correlation
    random_seed : int
        Random seed
        
    Returns
    -------
    bootstrap_correlations : np.ndarray
        Array of correlations from bootstrap (n_bootstrap, n_basins)
    """
    np.random.seed(random_seed)
    
    n_time = len(sst_detrended.time)
    n_blocks = int(np.ceil(n_time / block_size))
    n_basins = len(p_detrended.basin)
    
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP RANDOMIZATION EXPERIMENT")
    print(f"{'='*60}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Block size: {block_size} months")
    print(f"  Basins: {n_basins}")
    print(f"  Time points: {n_time}")
    
    # ========================================================================
    # Process bootstrap in batches
    # ========================================================================
    BATCH_SIZE = 4  
    n_batches = int(np.ceil(n_bootstrap / BATCH_SIZE))
    
    print(f"  Processing in {n_batches} batches of {BATCH_SIZE}")
    
    bootstrap_correlations = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, n_bootstrap)
        
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: iterations {start_idx}-{end_idx-1}")
        
        # Run parallel bootstrap for this batch
        results = Parallel(n_jobs=4, backend='loky')(
            delayed(run_single_bootstrap)(
                seed, n_time, sst_detrended, p_detrended, sst_anom, p_anom,
                area_precomputed, alpha, n_blocks, block_size
            )
            for seed in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx + 1}")
        )
        
        # Collect results
        bootstrap_correlations.extend(results)
        
        # Clean up
        del results
        gc.collect()
    
    # Convert to numpy array (n_bootstrap, n_basins)
    bootstrap_correlations = np.array(bootstrap_correlations, dtype=np.float32)
    
    print(f"\n  Bootstrap correlations shape: {bootstrap_correlations.shape}")
    
    return bootstrap_correlations


# ============================================================================
# PROCESS SINGLE PAIR
# ============================================================================

def process_pair(p_name, p_anom, sst_name, sst_anom):
    """
    Process a single precipitation-SST pair with randomization experiment.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")
    
    p_anom = p_anom.load()
    
    common_time = np.intersect1d(p_anom['time'].values, sst_anom['time'].values)
    p_anom = p_anom.sel(time=common_time)
    sst_anom = sst_anom.sel(time=common_time)
    
    # Detrending
    print("Detrending...")
    p_detrended = detrend_dim(p_anom, 'time').astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, 'time').astype(np.float32)
    
    
    # ========================================================================
    # Original regression (non-bootstrapped)
    # ========================================================================
    print("\nRunning original regressions...")
    slope, slope_se, pval = xr.apply_ufunc(
        regression_slope_se,
        sst_detrended,
        p_detrended,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32]
    )
    
    print(f"  Valid slopes: {(~np.isnan(slope)).sum().values}")
    
    # ========================================================================
    # Precompute area weights
    # ========================================================================
    print("\nPrecomputing area weights...")
    area = grid_area(slope).astype(np.float32)
    area_precomputed = area * xr.ones_like(slope)
    
    # ========================================================================
    # FDR correction on original
    # ========================================================================
    print("Applying FDR correction...")
    slope_area = area_precomputed * slope
    
    fdr_mask = fdr_correction(pval, alpha_FDR=ALPHA)
    n_sig_total = int(fdr_mask.sum().values)
    print(f"  Significant cells: {n_sig_total}")
    
    slope_sig = slope_area.where(fdr_mask)
    
    # ========================================================================
    # Original reconstruction and correlation
    # ========================================================================
    print("Computing original reconstruction...")
    reconstruction = (sst_anom * slope_sig).sum(('lat', 'lon'))
    
    print("Computing original correlation...")
    original_corr = xr.corr(reconstruction, p_anom, 'time')
    
    # ========================================================================
    # Bootstrap randomization experiment
    # ========================================================================
    print("\nRunning bootstrap randomization experiment...")

    # Automatically set block size based on SST dataset name
    if "ERSSTv6" in sst_name:
        block_size = 11
    elif "COBE-SST3" in sst_name:
        block_size = 9
    else:
        block_size = BLOCK_SIZE  # default fallback
    
    print(f"  Selected block size: {block_size} months based on SST dataset: {sst_name}")
    
    bootstrap_correlations = bootstrap_correlation_distribution(
        sst_detrended, p_detrended, sst_anom, p_anom, area_precomputed,
        ALPHA, N_BOOTSTRAP, block_size, RANDOM_SEED
    )
    
    # ========================================================================
    # Convert to xarray for easier handling
    # ========================================================================
    bootstrap_corr_da = xr.DataArray(
        bootstrap_correlations,
        dims=['bootstrap', 'basin'],
        coords={
            'bootstrap': np.arange(N_BOOTSTRAP),
            'basin': p_detrended.basin
        }
    )
    
    # ========================================================================
    # Compute summary statistics
    # ========================================================================
    print("\nCalculating bootstrap correlation statistics")
    boot_mean = bootstrap_corr_da.mean('bootstrap')
    boot_std = bootstrap_corr_da.std('bootstrap')
    
    # ========================================================================
    # Clean up before returning
    # ========================================================================
    del p_detrended, sst_detrended, sst_anom, slope, slope_se, pval
    del area, area_precomputed, slope_area, fdr_mask, reconstruction
    gc.collect()
    
    result = {
        'model_id': 'P ~ β*SST (Randomization)',
        'description': 'Bootstrap randomization experiment',
        'variable': 'precip',
        'alpha': ALPHA,
        'n_bootstrap': N_BOOTSTRAP,
        'block_size': block_size,
        'original_correlation': original_corr,
        'bootstrap_correlations': bootstrap_corr_da,
        'bootstrap_mean': boot_mean,
        'bootstrap_std': boot_std
    }
    
    print(f"\n  SUCCESS: {p_name} vs {sst_name}")
    
    return ((p_name, sst_name), result)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bootstrap randomization experiment for one pair')
    parser.add_argument('--pair-index', type=int, required=True,
                        help='Index of precipitation-SST pair to process (0-based)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OPTIMIZED BOOTSTRAP SE ANALYSIS")
    print("="*80)
    print(f"Pair index: {args.pair_index}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Block size: {BLOCK_SIZE}")
    print(f"Alpha: {ALPHA}")
        
    # Load precipitation anomalies
    precip_dict = {}
    for p_name in PRECIP_DATASETS.keys():
        nc_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
        if os.path.exists(nc_path):
            precip_dict[p_name] = xr.open_dataarray(nc_path)
    
    # Load SST anomalies
    sst_dict = {}
    for sst_name in ["ERSSTv6", "COBE-SST3"]:
        nc_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if os.path.exists(nc_path):
            sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()
    
    print(f"  Precipitation datasets: {list(precip_dict.keys())}")
    print(f"  SST datasets: {list(sst_dict.keys())}")
    
    # Create all pairs
    pairs = [(p_name, p_anom, sst_name, sst_anom)
             for p_name, p_anom in precip_dict.items()
             for sst_name, sst_anom in sst_dict.items()]
    
    total_pairs = len(pairs)
    print(f"  Total pairs: {total_pairs}")
    
    if args.pair_index >= total_pairs:
        print(f"ERROR: pair-index {args.pair_index} >= total pairs {total_pairs}")
        sys.exit(1)
    
    # Process the specified pair
    p_name, p_da, sst_name, sst_da = pairs[args.pair_index]
    
    print(f"\nProcessing pair {args.pair_index}/{total_pairs-1}: {p_name} vs {sst_name}")
    
    key, result = process_pair(p_name, p_da, sst_name, sst_da)
    
    # Convert result dict to xarray Dataset for saving
    result_ds = xr.Dataset({
        'original_correlation': result['original_correlation'],
        'bootstrap_correlations': result['bootstrap_correlations'],
        'bootstrap_mean': result['bootstrap_mean'],
        'bootstrap_std': result['bootstrap_std'],
    })
    
    # Add metadata as attributes
    result_ds.attrs.update({
        'model_id': result['model_id'],
        'description': result['description'],
        'variable': result['variable'],
        'alpha': result['alpha'],
        'n_bootstrap': result['n_bootstrap'],
        'block_size': result['block_size'],
        'precip_dataset': p_name,
        'sst_dataset': sst_name,
    })
    
        
    # Save result
    output_file = OUTPUTS_DIR / f'randomization_experiment_{p_name}_{sst_name}.nc'
    
    result_ds.to_netcdf(output_file)
    
    
    print(f"\n  Saved: {output_file}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()