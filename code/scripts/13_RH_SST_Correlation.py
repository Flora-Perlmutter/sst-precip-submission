#!/usr/bin/env python
# coding: utf-8
"""
Compute RH-SST correlations for each precipitation-SST pair.

Author: Flora Perlmutter

Description
-----------
For each precipitation-SST pair, computes the spatial correlation between
basin-scale detrended column RH (MERRA-2, pressure-weighted) and SST at
each grid point. Results are saved as NetCDF for use in downstream
conditioning variable analysis.

Designed for sbatch array submission — one job per pair via --pair-index.

"""

import argparse
import gc
import sys
import warnings
import os
import numpy as np
import glob
import xarray as xr
import xskillscore as xs
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import waterbasin, detrend_dim

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"

OUTPUTS_DIR = DATA_DIR

# Define precipitation and SST datasets
PRECIP_DATASETS = ['GPCP', 'CRU', 'GPCC', 'CPC', 'UDel', 'PREC', 'TerraClimate', 'REGEN']
SST_DATASETS = ['ERSSTv6', 'COBE-SST3']


# ============================================================================
# PROCESS SINGLE PAIR
# ============================================================================

def process_pair(p_name, p_da, sst_name, sst_da, rh_da):
    """
    Process a single precipitation-SST pair and compute the correlation between SST and RH.
    
    Parameters:
    -----------
    p_name : str
        Name of precipitation dataset
    p_da : xr.DataArray
        Precipitation anomalies
    sst_name : str
        Name of SST dataset
    sst_da : xr.DataArray
        SST anomalies
    rh_da : xr.DataArray
        Relative humidity data
        
    Returns:
    --------
    tuple : ((p_name, sst_name), rh_sst_corr)
        Key tuple and correlation DataArray
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")
    
    # Load precipitation data
    p_anom = p_da.load()
    
    # Find common time periods
    common_time = np.intersect1d(p_anom['time'].values, sst_da['time'].values)
    common_time = np.intersect1d(common_time, rh_da['time'].values)
    
    print(f"\nTime alignment:")
    print(f"  Precip time range: {p_anom['time'].values[0]} to {p_anom['time'].values[-1]}")
    print(f"  SST time range: {sst_da['time'].values[0]} to {sst_da['time'].values[-1]}")
    print(f"  RH time range: {rh_da['time'].values[0]} to {rh_da['time'].values[-1]}")
    print(f"  Common time: {len(common_time)} months ({common_time[0]} to {common_time[-1]})")
    
    # Subset to common time
    p_anom = p_anom.sel(time=common_time)
    sst_anom = sst_da.sel(time=common_time)
    
    # Process RH to basin-level
    print("\nProcessing RH to basin level...")
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
    
    # Cast to float32 to save memory
    print("\nCasting to float32...")
    p_final = p_final.astype(np.float32)
    sst_anom = sst_anom.astype(np.float32)
    rh_final = rh_final.astype(np.float32)
    
    # Calculate RH anomalies
    print("Calculating RH anomalies...")
    p_anom = p_final
    rh_anom = (rh_final.groupby(rh_final.time.dt.month) -
               rh_final.groupby(rh_final.time.dt.month).mean('time'))
    rh_anom = rh_anom.load()
    
    # Clean up
    del rh_basin, rh_constrained, p_final, rh_final
    gc.collect()
    
    # Detrending
    print("Detrending...")
    p_detrended = detrend_dim(p_anom, 'time').astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, 'time').astype(np.float32)
    rh_detrended = detrend_dim(rh_anom, 'time').astype(np.float32)
    
    del p_anom, sst_anom, rh_anom
    gc.collect()
    
    # ========================================================================
    # Calculate RH-SST Correlation
    # ========================================================================
    
    print("\nCalculating RH-SST correlation...")
    
    # Expand RH to match SST grid (each basin gets correlated with each SST gridpoint)
    rh_expanded = rh_detrended.expand_dims(lat=sst_detrended.lat, lon=sst_detrended.lon)
    
    # Expand SST to have basin dimension
    sst_expanded = sst_detrended.expand_dims(basin=rh_expanded.basin)
    
    # Compute correlation
    rh_sst_corr = xs.pearson_r(rh_expanded, sst_expanded, dim='time')
    p_values = xs.pearson_r_p_value(rh_expanded, sst_expanded, dim='time')
    
    print(f"  Correlation shape: {rh_sst_corr.shape}")
    print(f"  Mean correlation: {float(rh_sst_corr.mean()):.3f}")
    print(f"  Median correlation: {float(rh_sst_corr.median()):.3f}")
    
    # Clean up
    del rh_expanded, sst_expanded
    del p_detrended, sst_detrended, rh_detrended
    gc.collect()
    
    print(f"\nSUCCESS: {p_name} vs {sst_name}")
    
    return ((p_name, sst_name), rh_sst_corr, p_values)


# ============================================================================
# LOAD RH DATA
# ============================================================================

def load_rh_data():
    """Load and process MERRA-2 RH data."""
    print("\n" + "="*80)
    print("LOADING MERRA-2 RH DATA")
    print("="*80)
    
    rh_dir = CMIG_DATA / 'Data/Observations/MERRA-2/RH'
    files = sorted(glob.glob(os.path.join(rh_dir, "*.nc4")))
    
    if not files:
        raise FileNotFoundError(f"No RH files found in {rh_dir}")
    
    print(f"\nFound {len(files)} RH files")
    
    # Open and combine into one dataset
    print("Opening files...")
    rh_merra = xr.open_mfdataset(files, combine='by_coords')
    
    # Compute pressure weights
    print("Computing pressure-weighted column integration...")
    pressure = rh_merra.lev.values
    dp = np.gradient(pressure)
    weights = dp / np.sum(dp)
    weights_da = xr.DataArray(weights, coords={'lev': rh_merra.lev}, dims='lev')
    
    # Compute pressure-weighted column-integrated RH
    rh_da = (rh_merra.RH * weights_da).sum(dim='lev')
    
    print(f"\nRH data loaded:")
    print(f"  Shape: {rh_da.shape}")
    print(f"  Time range: {rh_da.time.values[0]} to {rh_da.time.values[-1]}")
    print(f"  Spatial extent: lat [{rh_da.lat.min().values:.1f}, {rh_da.lat.max().values:.1f}], "
          f"lon [{rh_da.lon.min().values:.1f}, {rh_da.lon.max().values:.1f}]")
    
    return rh_da


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    
    parser = argparse.ArgumentParser(
        description='Compute RH-SST correlations for precipitation-SST pairs'
    )
    parser.add_argument(
        '--pair-index', 
        type=int, 
        required=True,
        help='Index of precipitation-SST pair to process (0-based)'
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("RH-SST CORRELATION COMPUTATION")
    print("="*80)
    
    # Load precipitation anomalies
    print("\nLoading precipitation datasets...")
    precip_dict = {}
    for p_name in PRECIP_DATASETS:
        nc_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
        if os.path.exists(nc_path):
            precip_dict[p_name] = xr.open_dataarray(nc_path)
            print(f"  Loaded: {p_name}")
        else:
            print(f"  Missing: {p_name}")
    
    # Load SST anomalies
    print("\nLoading SST datasets...")
    sst_dict = {}
    for sst_name in SST_DATASETS:
        nc_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"        
        if os.path.exists(nc_path):
            sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()
            print(f"  Loaded: {sst_name}")
        else:
            print(f"  Missing: {sst_name}")
    
    if not precip_dict or not sst_dict:
        print("\nERROR: Missing required precipitation or SST data!")
        sys.exit(1)
    
    # Load RH data
    rh_da = load_rh_data()
    
    # Create all pairs
    pairs = [
        (p_name, p_da, sst_name, sst_da)
        for p_name, p_da in precip_dict.items()
        for sst_name, sst_da in sst_dict.items()
    ]
    
    total_pairs = len(pairs)
    print(f"\n" + "="*80)
    print(f"Total precipitation-SST pairs: {total_pairs}")
    print("="*80)
    
    # Validate pair index
    if args.pair_index >= total_pairs:
        print(f"\nERROR: pair-index {args.pair_index} is out of range!")
        print(f"Valid range: 0 to {total_pairs - 1}")
        sys.exit(1)
    
    # Process the specified pair
    p_name, p_da, sst_name, sst_da = pairs[args.pair_index]
    
    print(f"\nProcessing pair {args.pair_index + 1}/{total_pairs}:")
    print(f"  Precipitation: {p_name}")
    print(f"  SST: {sst_name}")
    
    # Compute correlation
    key, rh_sst_corr, p_values = process_pair(p_name, p_da, sst_name, sst_da, rh_da)
    
    # Save result
    output_file = OUTPUTS_DIR / f'rh_sst_correlation_{p_name}_{sst_name}.nc'
    
    print(f"\nSaving results...")
    ds_out = xr.Dataset({'correlation': rh_sst_corr, 'p_value': p_values})
    ds_out.to_netcdf(output_file)
    print(f"  Saved to: {output_file}")
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()