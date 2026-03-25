#!/usr/bin/env python
# coding: utf-8
"""

Author: Flora Perlmutter

Description
-----------
Loads cross-validation results (produced by run_cv_obs.py), spatially
averages over lat/lon, and saves the result for plotting.

Output
------
  <OUTPUTS_DIR>/processed/ds_avg.nc

"""

import gc
import warnings

import xarray as xr
import os
from pathlib import Path
import sys
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
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
import xskillscore
from matplotlib.cm import RdBu
from scipy.stats import linregress
from scipy import stats
from scipy.stats import t
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import glob
import gc

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR   = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
PROCESSED_DIR = DATA_DIR


# ============================================================================
# OPTIMIZED DATA LOADING AND PROCESSING
# ============================================================================

def load_cv_results_optimized(verbose=True):
    """
    Load CV result files efficiently with proper error handling.
    Adds pair dimension to the combined dataset.
    
        verbose: Print loading info
    
    Returns:
        xarray.Dataset: Combined CV results with pair dimension
    """
    cv_files = sorted(OUTPUTS_DIR.glob("cv_results_*.nc"))
    
    if not cv_files:
        print(f"ERROR: No CV result files found in: {OUTPUTS_DIR}")
        print("Make sure files are named 'cv_results_*.nc'")
        raise FileNotFoundError(f"No CV result files in {OUTPUTS_DIR}")
    
    if verbose:
        print(f"Found {len(cv_files)} CV result files:")
        for f in cv_files[:5]:
            print(f"  - {os.path.basename(f)}")
        if len(cv_files) > 5:
            print(f"  ... and {len(cv_files) - 5} more")
    
    try:
        # Load each file and add pair dimension
        datasets = []
        pair_names = []
        
        for i, file in enumerate(cv_files):
            ds = xr.open_dataset(file)
            
            # Extract pair name from filename or metadata
            # Adjust this logic based on how your pair info is stored
            basename = os.path.basename(file)
            pair_name = basename.replace('cv_results_', '').replace('.nc', '')
            
            # Add pair as a coordinate
            ds = ds.expand_dims(pair=[pair_name])
            datasets.append(ds)
            pair_names.append(pair_name)
            
            
            # Close dataset to free memory
            ds.close()
            del ds
            
            gc.collect()
            print(pair_name)
        
        # Concatenate along pair dimension
        ds_combined = xr.concat(datasets, dim='pair')
        
        if verbose:
            print(f"Successfully loaded and combined dataset")
            print(f"Shape: {dict(ds_combined.sizes)}")
            print(f"Pairs: {list(ds_combined.pair.values)}")
        
        return ds_combined
        
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        raise

def spatial_average_before_ranking(ds, spatial_dims=['lat', 'lon'], verbose=True):
    """
    Spatial averaging BEFORE ranking (crucial for per-basin comparisons).
    This averages over lat/lon within each basin/grid cell before ranking models.
    
    Args:
        ds: xarray.Dataset with spatial dimensions
        spatial_dims: Dimensions to average over (default: lat, lon)
        verbose: Print info
    
    Returns:
        xarray.Dataset: Spatially averaged data
    """
    # Check which spatial dimensions actually exist
    existing_spatial_dims = [d for d in spatial_dims if d in ds.dims]
    
    if not existing_spatial_dims:
        if verbose:
            print("No spatial dimensions found to average over")
            print(f"Available dimensions: {list(ds.dims)}")
        return ds
    
    if verbose:
        print(f"\nSpatially averaging over: {existing_spatial_dims}")
        print(f"Original shape: {dict(ds.sizes)}")
    
    try:
        # Average all data variables over spatial dims
        ds_averaged = ds.mean(dim=existing_spatial_dims, skipna=True)
        
        if verbose:
            print(f"  After averaging: {dict(ds_averaged.sizes)}")
        
        return ds_averaged
    except Exception as e:
        print(f"ERROR during spatial averaging: {e}")
        raise


def extract_metric_stats_optimized(ds, metric_name, verbose=True):
    """
    Extract metric data and compute statistics across pairs.
    
    Args:
        ds: xarray.Dataset from CV results
        metric_name: Name of metric variable
        verbose: Print info
    
    Returns:
        tuple: (mean_across_pairs, std_across_pairs)
    """
    if metric_name not in ds.data_vars:
        print(f"ERROR: Metric '{metric_name}' not found")
        print(f"Available variables: {list(ds.data_vars)}")
        raise ValueError(f"Metric {metric_name} not found in dataset")
    
    metric_da = ds[metric_name]
    
    # Get mean and std across pairs (if pair dimension exists)
    if 'pair' in metric_da.dims:
        metric_mean = metric_da.mean("pair", skipna=True)
        metric_std = metric_da.std("pair", skipna=True)
        if verbose:
            print(f"  Averaged across 'pair' dimension")
    else:
        metric_mean = metric_da
        metric_std = None
        if verbose:
            print(f"  No 'pair' dimension found")
    
    if verbose:
        print(f"  Shape after averaging pairs: {metric_mean.shape}")
    
    return metric_mean, metric_std


def calculate_ranks_per_basin_fast(da, lower_is_better=True, verbose=True):
    """
    Fast ranking using numpy.
    Ranks models within each basin/location independently.
    
    Args:
        da: xarray.DataArray with metric values (basin x model or similar)
        lower_is_better: If True, lower values get rank 1
        verbose: Print info
    
    Returns:
        xarray.DataArray: Ranks along model dimension
    """
    # Get data
    data = da.values.copy()
    
    # Prepare for ranking: invert if needed
    if not lower_is_better:
        data = -data
    
    # Get shape info
    shape = data.shape
    model_axis = da.dims.index('model')
    n_models = shape[model_axis]
    
    if verbose:
        print(f"  Ranking {n_models} models across basins/locations")
    
    # Move model dimension to last axis for easier processing
    data_moved = np.moveaxis(data, model_axis, -1)
    original_shape = data_moved.shape
    
    # Reshape to 2D: (all other dimensions, model)
    data_2d = data_moved.reshape(-1, n_models)
    
    # Fast ranking using numpy argsort
    # argsort returns indices of sorted elements
    # argsort of argsort gives ranks (1-indexed by adding 1)
    ranks_2d = np.argsort(np.argsort(np.nan_to_num(data_2d, nan=np.inf), axis=1), axis=1) + 1
    
    # Reshape back
    ranks_moved = ranks_2d.reshape(original_shape).astype(float)
    data_ranked = np.moveaxis(ranks_moved, -1, model_axis)
    
    # Convert back to DataArray
    ranks_da = xr.DataArray(
        data_ranked,
        dims=da.dims,
        coords=da.coords,
        name=f"{da.name}_rank" if da.name else "rank"
    )
    
    return ranks_da


def model_ranking(ds, metrics=None, verbose=True):
    results = {}
    """
    Rank models according to spatial average, dataset average for each basin.
    
    Args:
        ds: xarray.Dataset (spatially averaged before ranking)
        metrics: List of metric names to plot. If None, use defaults
        verbose: Print info
    """
    # Default metrics
    if metrics is None:
        metrics = ['rmse_mean', 'adjr2_mean']
    
    metric_info = {
        'rmse_mean': {
            'label': 'RMSE Rank',
            'lower_is_better': True,
            'title': 'Model Ranking by RMSE'
        },
        'adjr2_mean': {
            'label': 'Adjusted R² Rank',
            'lower_is_better': False,
            'title': 'Model Ranking by Adjusted R²'
        },
        'rmse': {
            'label': 'RMSE Rank',
            'lower_is_better': True,
            'title': 'Model Ranking by RMSE'
        },
        'r2': {
            'label': 'R² Rank',
            'lower_is_better': False,
            'title': 'Model Ranking by R²'
        }
    }
    
    # Validate dataset
    if 'model' not in ds.dims:
        print("ERROR: No 'model' dimension in dataset")
        print(f"Available dimensions: {list(ds.dims)}")
        raise ValueError("Dataset missing 'model' dimension")
    
    all_models = list(ds.model.values)
    if verbose:
        print(f"\nFound {len(all_models)} models: {all_models}")
    
    # Filter to metrics that exist
    valid_metrics = [m for m in metrics if m in ds.data_vars]
    if not valid_metrics:
        print(f"ERROR: None of the metrics {metrics} found in dataset")
        print(f"Available variables: {list(ds.data_vars)}")
        raise ValueError(f"No valid metrics found")
    
    if verbose:
        print(f"Plotting metrics: {valid_metrics}")
    
    # Process each metric
    for idx, metric in enumerate(valid_metrics):
        metric_config = metric_info.get(metric, {
            'label': metric,
            'lower_is_better': True,
            'title': f'Model Ranking by {metric}'
        })
        
        if verbose:
            print(f"\nProcessing {metric}...")
        
        try:
            # Extract metric
            metric_mean, _ = extract_metric_stats_optimized(ds, metric, verbose=verbose)
            
            # Rank (already spatially averaged)
            ranks_da = calculate_ranks_per_basin_fast(
                metric_mean,
                lower_is_better=metric_config['lower_is_better'],
                verbose=verbose
            )

            
            # Fix dimension order: we need (basin, model) for proper DataFrame
            # 1. Ensure 'model' is the SECOND dimension (columns) for the DataFrame
            # We want the shape to be (basins, models)
            if 'model' in ranks_da.dims:
                # Identify all other dimensions (basins, lat, lon, etc.)
                other_dims = [d for d in ranks_da.dims if d != 'model']
                # Transpose so other_dims are first, and model is LAST
                ranks_da = ranks_da.transpose(*other_dims, 'model')
            
            # 2. Stack any remaining spatial dimensions into a single index
            non_model_dims = [d for d in ranks_da.dims if d != 'model']
            if non_model_dims:
                # This creates a MultiIndex for rows, leaving 'model' as the columns
                ranks_df = ranks_da.stack(stacked_index=non_model_dims).to_pandas().T
                # Note: .T (transpose) if stacking puts model back in the index
            else:
                ranks_df = ranks_da.to_pandas()
            
            # If the DataFrame still has models as rows, force a transpose:
            if ranks_df.index.name == 'model' or 'model' in ranks_df.index.names:
                ranks_df = ranks_df.T
            
            results[metric] = ranks_df
            
            if verbose:
                print(f"  DataFrame shape: {ranks_df.shape}")
                print(f"  Rows (basins): {len(ranks_df.index)}, Cols (models): {len(ranks_df.columns)}")
            
    
                    
        except Exception as e:
            print(f"  ERROR processing {metric}: {e}")
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CV RESULTS ANALYSIS WITH SPATIAL AVERAGING")
    print("="*80)
    
    # Step 1: Load CV results
    print(f"\nStep 1: Loading CV results from:\n  {OUTPUTS_DIR}")
    ds = load_cv_results_optimized(verbose=True)
    
    print(f"\nDataset structure (before averaging):")
    print(f"  Dimensions: {dict(ds.sizes)}")
    print(f"  Variables: {list(ds.data_vars)}")
    
    # Step 2: Spatial averaging BEFORE ranking
    print(f"\nStep 2: Spatial averaging (mean over lat, lon)")
    ds_averaged = spatial_average_before_ranking(ds, spatial_dims=['lat', 'lon'], verbose=True)
    
    print(f"\nDataset structure (after spatial averaging):")
    print(f"  Dimensions: {dict(ds_averaged.sizes)}")
    print(f"  Variables: {list(ds_averaged.data_vars)}")
    
    # Step 3: Create and save plots
    print(f"\nStep 3: Creating plots with spatially-averaged data")
    print("="*80)
    
    ranks = model_ranking(ds_averaged, verbose=True)
    
    ranks['rmse_mean'].to_csv(PROCESSED_DIR / 'cross_validation_ranks_rmse_mean.csv')
    ranks['adjr2_mean'].to_csv(PROCESSED_DIR / 'cross_validation_ranks_adjr2_mean.csv')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return ds_averaged  # Return for interactive use


if __name__ == "__main__":
    ds = main()

