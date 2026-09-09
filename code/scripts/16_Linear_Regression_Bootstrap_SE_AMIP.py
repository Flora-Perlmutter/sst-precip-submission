#!/usr/bin/env python
# coding: utf-8
"""
Bootstrap standard error estimation for AMIP linear regression.

Author: Flora Perlmutter

Description
-----------
AMIP counterpart to run_bootstrap_se_obs.py. For each matched
precipitation-SST model pair, fits P ~ β*SST using basin-scale data,
applies FDR correction, computes SST reconstructions, and estimates
uncertainty via a memory-efficient block bootstrap with Welford's
incremental variance algorithm.

Key differences from the observational version:
  - Loads AMIP dataset names from amip_dataset_names.json manifest
  - Pairs are matched by model ID (same model for both pr and tas)
  - Block sizes loaded from amip_block_lengths.json (from determine_block_lengths_amip.py)
  - Input/output files use the amip_ prefix
  - Default block size is 8 months (vs 11 for obs)

Designed for sbatch array submission — one job per pair via --pair-index.

"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path
import os

import numpy as np
import psutil
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR, bootstrap_file
from regression_functions import detrend_dim, grid_area, fdr_correction, regression_slope_se
from sensitivity_common import TREND_PERIODS, trend_over

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OUTPUTS_DIR = DATA_DIR

# Configuration
ALPHA = 0.05
N_BOOTSTRAP = 100
BLOCK_SIZE = 8
RANDOM_SEED = 42

# ============================================================================
# MEMORY MONITORING
# ============================================================================

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)

def get_available_memory():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / (1024 ** 3)

def print_memory_status(step_name):
    """Print current memory status."""
    used = get_memory_usage()
    available = get_available_memory()
    total = psutil.virtual_memory().total / (1024 ** 3)
    percent = psutil.virtual_memory().percent
    
    print(f"  [MEMORY] {step_name}:")
    print(f"    Process: {used:.2f} GB | Available: {available:.2f} GB | "
          f"Total: {total:.2f} GB | Used: {percent:.1f}%")
    
    if percent > 80:
        print(f"WARNING: System memory usage above 80%!")
    
    return used, available

# ============================================================================
# WELFORD'S INCREMENTAL VARIANCE ALGORITHM
# ============================================================================

class IncrementalStats:
    """
    Compute mean and variance incrementally using Welford's online algorithm.
    Avoids storing all bootstrap samples in memory.
    """
    
    def __init__(self, shape, dtype=np.float32):
        self.n = 0
        self.mean = np.zeros(shape, dtype=dtype)
        self.m2 = np.zeros(shape, dtype=dtype)  # Sum of squared deviations
    
    def update(self, x):
        """Add a new sample."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
    
    def get_mean(self):
        """Return current mean."""
        return self.mean
    
    def get_variance(self):
        """Return current variance."""
        if self.n < 2:
            return np.zeros_like(self.mean)
        return self.m2 / (self.n - 1)
    
    def get_std(self):
        """Return current standard deviation."""
        return np.sqrt(self.get_variance())



# ============================================================
# Parallelized Bootstrap Loop using joblib
# ============================================================

def run_single_bootstrap(seed, n_time, sst_detrended, p_detrended, sst_anom,
                         area_precomputed, alpha, n_blocks, block_size):
    np.random.seed(seed)
    # Block bootstrap resampling
    block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
    resampled_indices = []
    for block_idx in block_indices:
        start = block_idx * block_size
        end = min(start + block_size, n_time)
        resampled_indices.extend(range(start, end))
    resampled_indices = resampled_indices[:n_time]

    # Resample data
    sst_boot = sst_detrended.isel(time=resampled_indices)
    p_boot = p_detrended.isel(time=resampled_indices)

    # Run regressions
    slope_boot, _, pval_boot = xr.apply_ufunc(
        regression_slope_se,
        sst_boot,
        p_boot,
        input_core_dims=[['time'], ['time']],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32]
    )

    # FDR correction on the RAW slope -- see 11_Linear_Regression_Bootstrap_SE.py.
    # The sensitivity SE is accumulated from slope_sig_boot, so it must be in the
    # same per-unit-SST units the sensitivity is reported in.
    fdr_mask_boot = fdr_correction(pval_boot, alpha_FDR=alpha)
    slope_sig_boot = slope_boot.where(fdr_mask_boot, 0.0)

    # Reconstruction -- the one place area weighting enters.
    reconstruction_boot = (sst_anom * area_precomputed * slope_sig_boot).sum(('lat', 'lon'))

    # Derived quantities are computed here because this is the only place a
    # replicate exists — the caller keeps Welford summaries, not the samples.
    # See 11_Linear_Regression_Bootstrap_SE.py for why a trend over a replicate
    # is well defined.
    trend_boot  = xr.concat(
        [trend_over(reconstruction_boot, period) for period in TREND_PERIODS],
        dim='period',
    )

    result = (
        reconstruction_boot.values.astype(np.float32),
        slope_sig_boot.values.astype(np.float32),
        trend_boot.values.astype(np.float32),
    )

    return (result)

# ============================================================================
# OPTIMIZED BOOTSTRAP FUNCTIONS - SEQUENTIAL
# ============================================================================

def bootstrap_se_incremental(sst_detrended, p_detrended, sst_anom, area_precomputed,
                              alpha, n_bootstrap=200, block_size=11, random_seed=42):
    """
    Memory-efficient bootstrap using incremental variance calculation.
    Processes bootstrap samples in batches to avoid memory buildup.
    
    Parameters
    ----------
    sst_detrended : xr.DataArray
        SST detrended data (time, lat, lon) [float32]
    p_detrended : xr.DataArray
        Precipitation detrended data (time, basin) [float32]
    sst_anom : xr.DataArray
        SST anomalies for reconstruction [float32]
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
    reconstruction_se : xr.DataArray
        Bootstrap SE for reconstruction (time, basin)
    marginal_sensitivity_se : xr.DataArray
        Bootstrap SE for marginal sensitivity (lat, lon, basin)
    """
    np.random.seed(random_seed)
    
    n_time = len(sst_detrended.time)
    n_blocks = int(np.ceil(n_time / block_size))
    n_basins = len(p_detrended.basin)
    n_lat = len(sst_detrended.lat)
    n_lon = len(sst_detrended.lon)
    
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP SE CALCULATION (Batched + Incremental)")
    print(f"{'='*60}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Block size: {block_size} months")
    print(f"  Using float32 precision")
    print(f"  Grid: {n_lat} x {n_lon}")
    print(f"  Basins: {n_basins}")
    print(f"  Time points: {n_time}")
    
    # Initialize incremental statistics
    recon_stats = IncrementalStats((n_time, n_basins), dtype=np.float32)
    slope_stats = IncrementalStats((n_lat, n_lon, n_basins), dtype=np.float32)

    # Full distributions rather than Welford summaries; see script 11.
    trend_boot_all  = np.full((n_bootstrap, len(TREND_PERIODS), n_basins),
                              np.nan, dtype=np.float32)
    
    print_memory_status("After initializing incremental stats")
    
    # ========================================================================
    # Process bootstrap in batches
    # ========================================================================
    from joblib import Parallel, delayed
    
    BATCH_SIZE = 4  # Process 4 bootstraps at a time
    n_batches = int(np.ceil(n_bootstrap / BATCH_SIZE))
    
    print(f"  Processing in {n_batches} batches of {BATCH_SIZE}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, n_bootstrap)
        batch_size = end_idx - start_idx
        
        print(f"\n  Batch {batch_idx + 1}/{n_batches}: iterations {start_idx}-{end_idx-1}")
        
        # Run parallel bootstrap for this batch
        results = Parallel(n_jobs=4, backend='loky')(
            delayed(run_single_bootstrap)(
                seed, n_time, sst_detrended, p_detrended, sst_anom, area_precomputed,
                alpha, n_blocks, block_size
            )
            for seed in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx + 1}")
        )
        
        print_memory_status(f"After batch {batch_idx + 1} computation")
        
        # Update incremental statistics with batch results
        for offset, (recon_values, slope_values, trend_values) in enumerate(results):
            recon_stats.update(recon_values)
            slope_stats.update(slope_values)
            trend_boot_all[start_idx + offset]  = trend_values
        
        # Aggressively clean up batch results
        del results
        gc.collect()
        
        print_memory_status(f"After batch {batch_idx + 1} cleanup")
    
    print_memory_status("After all batches complete")
    
    # Get final statistics
    reconstruction_se_np = recon_stats.get_std()
    marginal_sensitivity_se_np = slope_stats.get_std()
    
    # Convert to xarray
        
    reconstruction_se = xr.DataArray(
        reconstruction_se_np,
        dims=['time', 'basin'],
        coords={'time': p_detrended.time, 'basin': p_detrended.basin}
    )
    
    marginal_sensitivity_se = xr.DataArray(
        marginal_sensitivity_se_np,
        dims=['lat', 'lon', 'basin'],
        coords={
            'lat': sst_detrended.lat,
            'lon': sst_detrended.lon,
            'basin': p_detrended.basin
        })
    
    period_labels = [f"{start[:4]}-{end[:4]}" for start, end in TREND_PERIODS]

    trend_boot = xr.DataArray(
        trend_boot_all,
        dims=['bootstrap', 'period', 'basin'],
        coords={
            'bootstrap': np.arange(n_bootstrap),
            'period': period_labels,
            'basin': p_detrended.basin
        })

    print_memory_status("After converting to xarray")

    return (reconstruction_se, marginal_sensitivity_se, trend_boot)

# ============================================================================
# PROCESS SINGLE PAIR
# ============================================================================

def process_pair(p_name, p_anom, sst_name, sst_anom, block_size):
    """
    Process a single precipitation-SST pair with optimized bootstrap.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")
    
    p_anom = p_anom.load()
    sst_anom = sst_anom.load()
    
    common_time = np.intersect1d(p_anom['time'].values, sst_anom['time'].values)
    p_anom = p_anom.sel(time=common_time)
    sst_anom = sst_anom.sel(time=common_time)
    
    # Detrending
    print("Detrending...")
    p_detrended = detrend_dim(p_anom, 'time').astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, 'time').astype(np.float32)
    
    # ========================================================================
    # Original regression
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
    
    print_memory_status("After regression")
    
    # ========================================================================
    # Precompute area weights
    # ========================================================================
    print("\nPrecomputing area weights...")
    area = grid_area(slope).astype(np.float32)
    
    # Broadcast area to all basins
    area_precomputed = area * xr.ones_like(slope)
    
    print_memory_status("After area computation")
    
    # ========================================================================
    # FDR correction on original
    # ========================================================================
    # Masked on the RAW slope, so `slope_sig` -- saved as `marginal_sensitivity` --
    # is a sensitivity in mm month-1 K-1. Area belongs to the spatial integral below.
    print("Applying FDR correction...")
    fdr_mask = fdr_correction(pval, alpha_FDR=ALPHA)
    n_sig_total = int(fdr_mask.sum().values)
    print(f"  Significant cells: {n_sig_total}")
    
    slope_sig = slope.where(fdr_mask)
    
    # ========================================================================
    # Original reconstruction
    # ========================================================================
    # Area weighting enters here, and only here.
    print("Computing original reconstruction...")
    reconstruction = (sst_anom * area_precomputed * slope_sig).sum(('lat', 'lon'))
    
    print_memory_status("After original reconstruction")
    
    # ========================================================================
    # Bootstrap SE (incremental, memory-efficient, sequential)
    # ========================================================================
    print("\nRunning bootstrap SE calculation...")
    
    print(f"  Selected block size: {block_size} months based on SST dataset name: {sst_name}")
    
    reconstruction_se, marginal_sensitivity_se, trend_boot = bootstrap_se_incremental(
        sst_detrended, p_detrended, sst_anom, area_precomputed,
        ALPHA, N_BOOTSTRAP, block_size, RANDOM_SEED
    )
    
    print_memory_status("After bootstrap")

    
    # ========================================================================
    # Compute metrics per basin
    # ========================================================================    
    print("  Computing metrics...")
    
    corr = xr.corr(reconstruction, p_anom, 'time')

    # Trend point estimates; see script 11.
    reconstruction_trend = xr.concat(
        [trend_over(reconstruction, period) for period in TREND_PERIODS],
        dim='period',
    ).assign_coords(period=trend_boot['period'])

    # ========================================================================
    # Clean up before returning
    # ========================================================================
    del p_detrended, sst_detrended, sst_anom, slope, slope_se, pval
    del area, area_precomputed, fdr_mask
    gc.collect()
    
    print_memory_status("Before return")
    
    result = {
        'model_id': 'P ~ β*SST',
        'description': 'Linear regression with bootstrap SE (optimized)',
        'variable': 'precip',
        'alpha': ALPHA,
        'n_bootstrap': N_BOOTSTRAP,
        'block_size': block_size,
        'reconstruction': reconstruction,
        'reconstruction_se': reconstruction_se,
        'observed_precip': p_anom,
        'correlation': corr,
        'marginal_sensitivity_se': marginal_sensitivity_se,
        'marginal_sensitivity': slope_sig,
        'reconstruction_trend': reconstruction_trend,
        'trend_boot': trend_boot,
    }
    
    print(f"\n SUCCESS: {p_name} vs {sst_name}")
    
    return ((p_name, sst_name), result)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bootstrap SE for one pair')
    parser.add_argument('--pair-index', type=int, required=True,
                        help='Index of precipitation-SST pair to process (0-based)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OPTIMIZED BOOTSTRAP SE ANALYSIS")
    print("="*80)
    print(f"Pair index: {args.pair_index}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print(f"Alpha: {ALPHA}")
    
    # Load data
    # Load dataset names
    names_path = INPUTS_DIR / "amip_dataset_names.json"
    with open(names_path, "r") as f:
        names = json.load(f)
    
    print(f"\nDataset names loaded:")
    
    # Load results
    results_path = DATA_DIR / "amip_block_lengths.json"
    with open(results_path, 'r') as f:
        autocorr_results = json.load(f)
    
    # Load precipitation anomalies using manifest
    precip_dict = {}
    for p_name in names['precip_datasets']:
        nc_path = INPUTS_DIR / f"amip_precip_anom_{p_name}.nc"
        if os.path.exists(nc_path):
            precip_dict[p_name] = xr.open_dataarray(nc_path)
        else:
            print(f"Warning: {nc_path} not found")
    
    # Load SST anomalies using manifest
    sst_dict = {}
    for sst_name in names['sst_datasets']:
        nc_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if os.path.exists(nc_path):
            sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()
        else:
            print(f"Warning: {nc_path} not found")
    
    print(f"\nLoaded {len(precip_dict)} precipitation datasets and {len(sst_dict)} SST datasets")
    
    # Create pairs only for model IDs appearing in BOTH
    common_models = sorted(set(precip_dict.keys()) & set(sst_dict.keys()))
    
    print(f"  Common models: {common_models}")
    print(f"  Total paired models: {len(common_models)}")
    
    # Build aligned pairs
    pairs = [
        (model_id, precip_dict[model_id], model_id, sst_dict[model_id])
        for model_id in common_models
    ]
    
    total_pairs = len(pairs)
    print(f"  Total pairs: {total_pairs}")
    
    if args.pair_index >= total_pairs:
        print(f"ERROR: pair-index {args.pair_index} >= total pairs {total_pairs}")
        sys.exit(1)
    
    # Process the specified pair
    p_name, p_da, sst_name, sst_da = pairs[args.pair_index]
    
    print(f"\nProcessing pair {args.pair_index}/{total_pairs-1}: {p_name} vs {sst_name}")
    
    # Get block size for this specific SST dataset
    if sst_name in autocorr_results:
        block_size = autocorr_results[sst_name]
        print(f"  Using computed block size: {block_size} months (from autocorrelation analysis)")
    else:
        block_size = BLOCK_SIZE  # Your default fallback value
        print(f"  WARNING: No autocorrelation data for {sst_name}, using default block size: {block_size}")
    
    # Pass block_size to process_pair
    key, result = process_pair(p_name, p_da, sst_name, sst_da, block_size)

    # Save result as NetCDF instead of pickle
    output_file = bootstrap_file(p_name, sst_name, amip=True)

    # Convert result dict to xarray Dataset for saving
    result_ds = xr.Dataset({
        'reconstruction': result['reconstruction'],
        'reconstruction_se': result['reconstruction_se'],
        'observed_precip': result['observed_precip'],
        'correlation': result['correlation'],
        'marginal_sensitivity_se': result['marginal_sensitivity_se'],
        'marginal_sensitivity': result['marginal_sensitivity'],
        'reconstruction_trend': result['reconstruction_trend'],
        'trend_boot': result['trend_boot'],
    })

    # Same convention as script 11: the saved sensitivity is the raw FDR-masked
    # slope, with cell area applied only inside the reconstruction.
    for _v in ('marginal_sensitivity', 'marginal_sensitivity_se'):
        result_ds[_v].attrs.update({
            'units': 'mm month-1 K-1',
            'note': ('FDR-masked OLS slope, NOT area-weighted. Grid-cell area is '
                     'applied where the reconstruction is formed, so multiply by '
                     'regression_functions.grid_area() before summing over lat/lon.'),
        })

    result_ds['reconstruction'].attrs.update({
        'units': 'mm month-1',
        'note': 'sum over ocean cells of area * beta * SST anomaly',
    })

    result_ds['trend_boot'].attrs.update({
        'long_name': 'per-decade reconstruction trend, one value per bootstrap replicate',
        'units': 'mm/month/decade',
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

    result_ds.to_netcdf(output_file)

    print(f"\n  Saved: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()