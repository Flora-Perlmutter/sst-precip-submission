#!/usr/bin/env python
# coding: utf-8
"""
Bootstrap standard error estimation for RH-conditioned regression (observational).

Author: Flora Perlmutter

Description
-----------
Fits P ~ β₀·SST + β₁·column RH using basin-scale data, applies FDR
correction to both SST and RH slopes, computes SST and RH reconstructions,
and estimates uncertainty via block bootstrap with Welford's incremental
variance algorithm.

Column-integrated RH is computed from MERRA-2 as a pressure-weighted
vertical average across all available levels.

Designed for sbatch array submission — one job per pair via --pair-index.

Block sizes per SST dataset (from determine_block_lengths.py):
  ERSSTv6   → 11 months
  COBE-SST3 → 9 months

"""

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
from pathlib import Path

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import (
    waterbasin, detrend_dim, grid_area, fdr_correction,
    regression_with_conditioning_var_se,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
INPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OUTPUTS_DIR = DATA_DIR
MERRA2_RH_DIR = CMIG_DATA / "Data/Observations/MERRA-2/RH"

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
        print(f" WARNING: System memory usage above 80%!")
    
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

# ============================================================================
# PARALLELIZED BOOTSTRAP LOOP
# ============================================================================

def run_single_bootstrap(seed, n_time, sst_detrended, rh_detrended, p_detrended,
                         sst_anom, rh_anom, area_precomputed, alpha, n_blocks, block_size):
    """
    Run one bootstrap iteration for RH conditioning regression.
    
    Returns
    -------
    tuple of numpy arrays:
        - reconstruction_sst_boot: SST reconstruction (time, basin)
        - reconstruction_rh_boot: RH reconstruction (time, basin)
        - reconstruction_total_boot: Total reconstruction (time, basin)
        - slope_sst_sig_boot: Significant SST slopes (lat, lon, basin)
        - slope_rh_sig_boot: Significant RH slopes (basin)
    """
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
    rh_boot = rh_detrended.isel(time=resampled_indices)
    p_boot = p_detrended.isel(time=resampled_indices)
    
    # Expand RH to match SST grid
    rh_expanded_boot = rh_boot.expand_dims(lat=sst_boot.lat, lon=sst_boot.lon)
    
    # Run regression with conditioning variable
    slope_sst_boot, _, pval_sst_boot, slope_rh_boot, _, pval_rh_boot = xr.apply_ufunc(
        regression_with_conditioning_var_se,
        sst_boot,
        rh_expanded_boot,
        p_boot,
        input_core_dims=[['time'], ['time'], ['time']],
        vectorize=True,
        output_core_dims=[[], [], [], [], [], []],
        output_dtypes=[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    )
    
    # FDR correction on the RAW slopes. Figure 2 divides this script's SST
    # sensitivity by script 11's, so the two must carry the same convention or the
    # attenuation ratio picks up a spurious cos(lat) factor.
    fdr_mask_sst_boot = fdr_correction(pval_sst_boot, alpha_FDR=alpha)
    fdr_mask_rh_boot = fdr_correction(pval_rh_boot, alpha_FDR=alpha)
    
    slope_sst_sig_boot = slope_sst_boot.where(fdr_mask_sst_boot, 0.0)
    slope_rh_sig_boot = slope_rh_boot.where(fdr_mask_rh_boot, 0.0)
    
    rh_anom_expanded = rh_anom.expand_dims(lat=sst_anom.lat, lon=sst_anom.lon)
    
    # Reconstructions with original anomalies. Area weighting enters here only.
    reconstruction_sst_boot = (sst_anom * area_precomputed * slope_sst_sig_boot).sum(('lat', 'lon'))
    reconstruction_rh_boot = (rh_anom_expanded * slope_rh_sig_boot).mean(('lat', 'lon'))
    reconstruction_total_boot = reconstruction_sst_boot + reconstruction_rh_boot
    
    result = (
        reconstruction_sst_boot.values.astype(np.float32),
        reconstruction_rh_boot.values.astype(np.float32),
        reconstruction_total_boot.values.astype(np.float32),
        slope_sst_sig_boot.values.astype(np.float32),
        slope_rh_sig_boot.values.astype(np.float32)
    )

    return (result)

# ============================================================================
# OPTIMIZED BOOTSTRAP FUNCTIONS
# ============================================================================

def bootstrap_se_incremental(sst_detrended, rh_detrended, p_detrended,
                              sst_anom, rh_anom, area_precomputed,
                              alpha, n_bootstrap=200, block_size=11, random_seed=42):
    """
    Memory-efficient bootstrap for RH conditioning regression using incremental variance.
    
    Parameters
    ----------
    sst_detrended : xr.DataArray
        SST detrended data (time, lat, lon) [float32]
    rh_detrended : xr.DataArray
        RH detrended data (time, basin) [float32]
    p_detrended : xr.DataArray
        Precipitation detrended data (time, basin) [float32]
    sst_anom : xr.DataArray
        SST anomalies for reconstruction [float32]
    rh_anom : xr.DataArray
        RH anomalies for reconstruction [float32]
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
    dict with bootstrap SEs:
        - 'reconstruction_sst_se': Bootstrap SE for SST reconstruction
        - 'reconstruction_rh_se': Bootstrap SE for RH reconstruction
        - 'reconstruction_total_se': Bootstrap SE for total reconstruction
        - 'marginal_sensitivity_sst_se': Bootstrap SE for SST slopes
        - 'marginal_sensitivity_rh_se': Bootstrap SE for RH slopes
    """
    np.random.seed(random_seed)
    
    n_time = len(sst_detrended.time)
    n_blocks = int(np.ceil(n_time / block_size))
    n_basins = len(p_detrended.basin)
    n_lat = len(sst_detrended.lat)
    n_lon = len(sst_detrended.lon)
    
    print(f"\n{'='*60}")
    print(f"BOOTSTRAP SE CALCULATION (RH Conditioning)")
    print(f"{'='*60}")
    print(f"  Bootstrap iterations: {n_bootstrap}")
    print(f"  Block size: {block_size} months")
    print(f"  Using float32 precision")
    print(f"  Grid: {n_lat} x {n_lon}")
    print(f"  Basins: {n_basins}")
    print(f"  Time points: {n_time}")
    
    # Initialize incremental statistics
    recon_sst_stats = IncrementalStats((n_time, n_basins), dtype=np.float32)
    recon_rh_stats = IncrementalStats((n_time, n_basins), dtype=np.float32)
    recon_total_stats = IncrementalStats((n_time, n_basins), dtype=np.float32)
    slope_sst_stats = IncrementalStats((n_lat, n_lon, n_basins), dtype=np.float32)
    slope_rh_stats = IncrementalStats((n_lat, n_lon, n_basins), dtype=np.float32)
    
    print_memory_status("After initializing incremental stats")
    
    # ========================================================================
    # Process bootstrap in batches
    # ========================================================================
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
                seed, n_time, sst_detrended, rh_detrended, p_detrended,
                sst_anom, rh_anom, area_precomputed, alpha, n_blocks, block_size
            )
            for seed in tqdm(range(start_idx, end_idx), desc=f"Batch {batch_idx + 1}")
        )
        
        print_memory_status(f"After batch {batch_idx + 1} computation")
        
        # Update incremental statistics with batch results
        for recon_sst_vals, recon_rh_vals, recon_total_vals, slope_sst_vals, slope_rh_vals in results:
            recon_sst_stats.update(recon_sst_vals)
            recon_rh_stats.update(recon_rh_vals)
            recon_total_stats.update(recon_total_vals)
            slope_sst_stats.update(slope_sst_vals)
            slope_rh_stats.update(slope_rh_vals)
        
        # Aggressively clean up batch results
        del results
        gc.collect()
        
        print_memory_status(f"After batch {batch_idx + 1} cleanup")
    
    print_memory_status("After all batches complete")
    
    # Get final statistics
    reconstruction_sst_se_np = recon_sst_stats.get_std()
    reconstruction_rh_se_np = recon_rh_stats.get_std()
    reconstruction_total_se_np = recon_total_stats.get_std()
    marginal_sensitivity_sst_se_np = slope_sst_stats.get_std()
    marginal_sensitivity_rh_se_np = slope_rh_stats.get_std()
    
    # Convert to xarray
    reconstruction_sst_se = xr.DataArray(
        reconstruction_sst_se_np,
        dims=['time', 'basin'],
        coords={'time': p_detrended.time, 'basin': p_detrended.basin}
    )
    
    reconstruction_rh_se = xr.DataArray(
        reconstruction_rh_se_np,
        dims=['time', 'basin'],
        coords={'time': p_detrended.time, 'basin': p_detrended.basin}
    )
    
    reconstruction_total_se = xr.DataArray(
        reconstruction_total_se_np,
        dims=['time', 'basin'],
        coords={'time': p_detrended.time, 'basin': p_detrended.basin}
    )
    
    marginal_sensitivity_sst_se = xr.DataArray(
        marginal_sensitivity_sst_se_np,
        dims=['lat', 'lon', 'basin'],
        coords={
            'lat': sst_detrended.lat,
            'lon': sst_detrended.lon,
            'basin': p_detrended.basin
        }
    )
    
    marginal_sensitivity_rh_se = xr.DataArray(
        marginal_sensitivity_rh_se_np,
        dims=['lat', 'lon', 'basin'],
        coords={
            'lat': sst_detrended.lat,
            'lon': sst_detrended.lon,
            'basin': p_detrended.basin
        }
    )
    
    print_memory_status("After converting to xarray")
    
    return {
        'reconstruction_sst_se': reconstruction_sst_se,
        'reconstruction_rh_se': reconstruction_rh_se,
        'reconstruction_total_se': reconstruction_total_se,
        'marginal_sensitivity_sst_se': marginal_sensitivity_sst_se,
        'marginal_sensitivity_rh_se': marginal_sensitivity_rh_se
    }

# ============================================================================
# PROCESS SINGLE PAIR
# ============================================================================

def process_pair(p_name, p_da, sst_name, sst_da, rh_da):
    """
    Process a single precipitation-SST pair with RH conditioning and optimized bootstrap.
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name} (with RH conditioning)")
    print(f"{'='*80}")
    
    p_anom = p_da.load()
    
    common_time = np.intersect1d(p_anom['time'].values, sst_da['time'].values)
    common_time = np.intersect1d(common_time, rh_da['time'].values)
    p_anom = p_anom.sel(time=common_time)
    sst_anom = sst_da.sel(time=common_time)
    rh_basin = waterbasin(rh_da)
    rh_constrained = rh_basin.sel(time=common_time)
    
    # Find common basins
    common_basin = np.intersect1d(p_anom['basin'].values, rh_constrained['basin'].values)
    p_final = p_anom.sel(basin=common_basin)
    rh_final = rh_constrained.sel(basin=common_basin)

    
    print(f"\nData dimensions:")
    print(f"  Time: {len(common_time)} months")
    print(f"  Basins: {len(common_basin)}")
    print(f"  SST grid: {len(sst_anom.lat)} x {len(sst_anom.lon)}")
    
    # Cast to float32
    print("\nCasting to float32...")
    p_final = p_final.astype(np.float32)
    sst_anom = sst_anom.astype(np.float32)
    rh_final = rh_final.astype(np.float32)
    
    # Calculate anomalies
    print("Calculating anomalies...")
    p_anom = p_final
    rh_anom = (rh_final.groupby(rh_final.time.dt.month) -
               rh_final.groupby(rh_final.time.dt.month).mean('time'))
    rh_anom = rh_anom.load()
    
    del rh_basin, rh_constrained, p_final, rh_final
    gc.collect()
    
    print_memory_status("After anomaly calculation")
    
    # Detrending
    print("Detrending...")
    p_detrended = detrend_dim(p_anom, 'time').astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, 'time').astype(np.float32)
    rh_detrended = detrend_dim(rh_anom, 'time').astype(np.float32)
    
    print_memory_status("After detrending")
    
    # ========================================================================
    # Original regression
    # ========================================================================
    print("\nRunning original regressions...")
    
    # Expand RH to match SST grid
    rh_expanded = rh_detrended.expand_dims(lat=sst_detrended.lat, lon=sst_detrended.lon)
    
    slope_sst, slope_se_sst, pval_sst, slope_rh, slope_se_rh, pval_rh = xr.apply_ufunc(
        regression_with_conditioning_var_se,
        sst_detrended,
        rh_expanded,
        p_detrended,
        input_core_dims=[['time'], ['time'], ['time']],
        vectorize=True,
        output_core_dims=[[], [], [], [], [], []],
        output_dtypes=[np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    )
    
    print(f"  Valid SST slopes: {(~np.isnan(slope_sst)).sum().values}")
    print(f"  Valid RH slopes: {(~np.isnan(slope_rh)).sum().values}")
    
    del rh_expanded
    gc.collect()
    
    print_memory_status("After regression")
    
    # ========================================================================
    # Precompute area weights
    # ========================================================================
    print("\nPrecomputing area weights...")
    area = grid_area(slope_sst).astype(np.float32)
    
    # Broadcast area to all basins
    area_precomputed = area * xr.ones_like(slope_sst)
    
    print_memory_status("After area computation")
    
    # ========================================================================
    # FDR correction on original
    # ========================================================================
    # Masked on the RAW slopes, matching 11_Linear_Regression_Bootstrap_SE.py: the
    # saved SST sensitivity is per unit SST, and cell area is applied only in the
    # reconstruction below.
    print("Applying FDR correction...")
    fdr_mask_sst = fdr_correction(pval_sst, alpha_FDR=ALPHA)
    fdr_mask_rh = fdr_correction(pval_rh, alpha_FDR=ALPHA)
    
    n_sig_sst = int(fdr_mask_sst.sum().values)
    n_sig_rh = int(fdr_mask_rh.sum().values)
    print(f"  Significant SST cells: {n_sig_sst}")
    print(f"  Significant RH cells: {n_sig_rh}")
    
    slope_sst_sig = slope_sst.where(fdr_mask_sst)
    slope_sst_se_sig = slope_se_sst.where(fdr_mask_sst)
    slope_rh_sig = slope_rh.where(fdr_mask_rh)
    slope_rh_se_sig = slope_se_rh.where(fdr_mask_rh)
    
    # ========================================================================
    # Original reconstruction
    # ========================================================================
    print("Computing original reconstruction...")
    
    rh_anom_expanded = rh_anom.expand_dims(lat=sst_anom.lat, lon=sst_anom.lon)
    
    reconstruction_sst = (sst_anom * area_precomputed * slope_sst_sig).sum(('lat', 'lon'))
    reconstruction_rh = (rh_anom_expanded * slope_rh_sig).mean(('lat', 'lon'))
    reconstruction_total = reconstruction_sst + reconstruction_rh
    
    print_memory_status("After original reconstruction")
    
    # ========================================================================
    # Bootstrap SE (incremental, memory-efficient)
    # ========================================================================
    print("\nRunning bootstrap SE calculation...")
    
    # Automatically set block size based on SST dataset name
    if "ERSSTv6" in sst_name:
        block_size = 11
    elif "COBE-SST3" in sst_name:
        block_size = 9
    else:
        block_size = BLOCK_SIZE  # default fallback (e.g., 11)
    
    print(f"  Selected block size: {block_size} months based on SST dataset name: {sst_name}")
    
    bootstrap_results = bootstrap_se_incremental(
        sst_detrended, rh_detrended, p_detrended,
        sst_anom, rh_anom, area_precomputed,
        ALPHA, N_BOOTSTRAP, block_size, RANDOM_SEED
    )
    print_memory_status("After bootstrap")
    
    # ========================================================================
    # Compute metrics per basin
    # ========================================================================    
    print("  Computing metrics...")
    
    corr_total = xr.corr(reconstruction_total, p_anom, 'time')
    corr_sst = xr.corr(reconstruction_sst, p_anom, 'time')
    
    # ========================================================================
    # Clean up before returning
    # ========================================================================
    del p_detrended, sst_detrended, rh_detrended
    del slope_sst, slope_se_sst, pval_sst, slope_rh, slope_se_rh, pval_rh
    del area, area_precomputed
    del fdr_mask_sst, fdr_mask_rh, rh_anom_expanded
    gc.collect()
    
    print_memory_status("Before return")
    
    result = {
        'model_id': 'P ~ β₀·SST + β₁·column RH',
        'description': 'Linear regression with RH conditioning, bootstrap SE',
        'variable': 'precip',
        'time_scale': 'monthly',
        'alpha': ALPHA,
        'n_bootstrap': N_BOOTSTRAP,
        'marginal_sensitivity_sst': slope_sst_sig,
        'marginal_sensitivity_sst_se': bootstrap_results['marginal_sensitivity_sst_se'],
        'marginal_sensitivity_rh': slope_rh_sig,
        'marginal_sensitivity_rh_se': bootstrap_results['marginal_sensitivity_rh_se'],
        'sst_reconstruction': reconstruction_sst,
        'sst_reconstruction_se': bootstrap_results['reconstruction_sst_se'],
        'reconstruction_total': reconstruction_total,
        'reconstruction_total_se': bootstrap_results['reconstruction_total_se'],
        'observed_precip_anomaly': p_anom,
        'correlation_sst': corr_sst,
        'correlation_total': corr_total
    }
    
    print(f"\n SUCCESS: {p_name} vs {sst_name}")
    
    return ((p_name, sst_name), result)

# ============================================================================
# LOAD RH DATA
# ============================================================================

def load_rh_data():
    """Load and process MERRA-2 RH data."""
    print("\nLoading MERRA-2 RH data...")
    
    files = sorted(MERRA2_RH_DIR.glob("*.nc4"))
    
    if not files:
        raise FileNotFoundError(f"No RH files found in {rh_dir}")
    
    print(f"  Found {len(files)} RH files")
    
    # Open and combine into one dataset
    rh_merra = xr.open_mfdataset(files, combine='by_coords')
    
    # Compute pressure weights
    pressure = rh_merra.lev.values
    dp = np.gradient(pressure)
    weights = dp / np.sum(dp)
    weights_da = xr.DataArray(weights, coords={'lev': rh_merra.lev}, dims='lev')
    
    # Compute pressure-weighted column-integrated RH
    rh_da = (rh_merra.RH * weights_da).sum(dim='lev')
    
    print(f"  RH data shape: {rh_da.shape}")
    print(f"  RH time range: {rh_da.time.values[0]} to {rh_da.time.values[-1]}")
    
    return rh_da

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    
    parser = argparse.ArgumentParser(description='Bootstrap SE for RH conditioning regression')
    parser.add_argument('--pair-index', type=int, required=True,
                        help='Index of precipitation-SST pair to process (0-based)')
    args = parser.parse_args()
    
    
    print("\nLoading observational ensembles...")
    
    # Load precipitation anomalies from .nc files
    precip_dict = {}
    for p_name in PRECIP_DATASETS.keys():
        nc_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
        if os.path.exists(nc_path):
            precip_dict[p_name] = xr.open_dataarray(nc_path)
    
    # Load SST anomalies from .nc files
    sst_dict = {}
    for sst_name in ["ERSSTv6", "COBE-SST3"]:
        nc_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if os.path.exists(nc_path):
            sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()
    
    print(f"  Precipitation datasets: {list(precip_dict.keys())}")
    print(f"  SST datasets: {list(sst_dict.keys())}")
    
    
    # Load RH data
    rh_da = load_rh_data()
    
    # Create all pairs
    pairs = [(p_name, p_da, sst_name, sst_da)
             for p_name, p_da in precip_dict.items()
             for sst_name, sst_da in sst_dict.items()]
    
    total_pairs = len(pairs)
    print(f"  Total pairs: {total_pairs}")
    
    if args.pair_index >= total_pairs:
        print(f"ERROR: pair-index {args.pair_index} >= total pairs {total_pairs}")
        sys.exit(1)
    
    # Process the specified pair
    p_name, p_da, sst_name, sst_da = pairs[args.pair_index]
    
    print(f"\nProcessing pair {args.pair_index}/{total_pairs-1}: {p_name} vs {sst_name}")
    
    key, result = process_pair(p_name, p_da, sst_name, sst_da, rh_da)
    
    output_file = OUTPUTS_DIR / f'global_rh_regression_bootstrap_{p_name}_{sst_name}.nc'
    
    # Convert result dict to xarray Dataset for saving
    result_ds = xr.Dataset({
        'marginal_sensitivity_sst': result['marginal_sensitivity_sst'],
        'marginal_sensitivity_sst_se': result['marginal_sensitivity_sst_se'],
        'marginal_sensitivity_rh': result['marginal_sensitivity_rh'],
        'marginal_sensitivity_rh_se': result['marginal_sensitivity_rh_se'],
        'sst_reconstruction': result['sst_reconstruction'],
        'sst_reconstruction_se': result['sst_reconstruction_se'],
        'reconstruction_total': result['reconstruction_total'],
        'reconstruction_total_se': result['reconstruction_total_se'],
        'observed_precip_anomaly': result['observed_precip_anomaly'],
        'correlation_sst': result['correlation_sst'],
        'correlation_total': result['correlation_total'],
    })
    
    # Same convention as script 11 -- Figure 2 takes the ratio of the two files'
    # SST sensitivities, so both must be raw for the ratio to be an attenuation.
    for _v in ('marginal_sensitivity_sst', 'marginal_sensitivity_sst_se'):
        if _v in result_ds:
            result_ds[_v].attrs.update({
                'units': 'mm month-1 K-1',
                'note': 'FDR-masked OLS slope, NOT area-weighted.',
            })

    # Add metadata as attributes
    result_ds.attrs.update({
        'model_id': result['model_id'],
        'description': result['description'],
        'variable': result['variable'],
        'time_scale': result['time_scale'],
        'alpha': result['alpha'],
        'n_bootstrap': result['n_bootstrap'],
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