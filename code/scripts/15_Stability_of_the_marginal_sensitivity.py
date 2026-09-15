#!/usr/bin/env python
# coding: utf-8
"""
Rolling window pattern correlation analysis.

Author: Flora Perlmutter

Description
-----------
Computes the pattern correlation between the full-period ensemble mean
SST-precipitation sensitivity and rolling 30-year window sensitivities,
across all basins. Designed to assess temporal stationarity of the
SST-precipitation relationship.

Every dataset is truncated to ANALYSIS_START-ANALYSIS_END (January 1979 to
December 2024). Each ensemble member's full-period sensitivity is computed
over that same period, and the rolling windows are built from the calendar
months within it that are common to EVERY selected member, so the reference
and the windows cover the same record for all members -- see DETERMINE TIME
WINDOWS below. The number of windows follows from the length of that period
(1979-2024 gives seventeen).

Both the full-period and the rolling-window sensitivities are FDR-corrected
before they are correlated, with the same (lat, lon)-per-basin test family
used by 11_Linear_Regression_Bootstrap_SE.py, so the correlation is between
the fields that actually enter the reconstruction rather than the raw slopes.

Two time period modes are available via USE_LONGER_PERIOD:
  True  → CRU, CPC, PREC, TerraClimate × ERSSTv6         (1980-2023)
  False → Full observational ensemble × ERSSTv6 + COBE-SST3 (1980-2016)

"""

import warnings
import sys
from functools import reduce
from pathlib import Path
import os

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.stats import pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import detrend_dim, fdr_correction, regression_slope_se

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"

OUTPUTS_DIR = DATA_DIR

# ============================================================================
# SETUP
# ============================================================================


# Configuration
WINDOW_SIZE = 30  # years (360 months)
WINDOW_STEP = 12  # step by 1 year (12 months) for each window
ALPHA = 0.05
ANALYSIS_START = "1979-01-01"  # full-period reference and rolling windows never use dates before this
ANALYSIS_END = "2024-12-31"    # ...or after this; every dataset is truncated here

# Time period options
USE_LONGER_PERIOD = True  # Set to True to use longer period with fewer datasets

if USE_LONGER_PERIOD:
    PRECIP_DATASETS_TO_USE = ['GPCP', 'CRU', 'GPCC', 'CPC', 'PREC', 'TerraClimate']
    SST_DATASETS_TO_USE = ['ERSSTv6', 'COBE-SST3']
else:
    PRECIP_DATASETS_TO_USE = ['GPCP', 'CRU', 'GPCC', 'CPC', 'UDel', 'PREC', 'TerraClimate', 'REGEN']
    SST_DATASETS_TO_USE = ['ERSSTv6', 'COBE-SST3']

# TIME_PERIOD describes the rolling-window common period. It isn't known until
# every member's record has been loaded and intersected (see DETERMINE TIME
# WINDOWS below), so it's set there rather than hardcoded here.
TIME_PERIOD = None

print(f"\n{'='*80}")
print(f"PATTERN CORRELATION ANALYSIS - ALL BASINS (PARALLELIZED)")
print(f"{'='*80}")
print(f"Window size: {WINDOW_SIZE} years ({WINDOW_SIZE * 12} months)")
print(f"Window step: {WINDOW_STEP} months")
print(f"Precip datasets: {PRECIP_DATASETS_TO_USE}")
print(f"SST datasets: {SST_DATASETS_TO_USE}")
print(f"Number of ensemble members: {len(PRECIP_DATASETS_TO_USE) * len(SST_DATASETS_TO_USE)}")

print(f"FDR alpha: {ALPHA}")

# ============================================================================
# FDR MASKING
# ============================================================================

def fdr_mask_sensitivity(slope, pval, alpha=ALPHA):
    """
    Zero out grid cells whose slope fails the BH-FDR test.

    Cells are pooled into one family per basin, matching
    11_Linear_Regression_Bootstrap_SE.py. Insignificant cells become 0.0, while
    cells that were already NaN -- land, missing data -- stay NaN so they are
    dropped from the correlation instead of entering it as a block of identical
    zeros.

    `slope` is the raw sensitivity, not an area-weighted one. This script's
    product is a spatial pattern correlation between sensitivity fields, and a
    correlation is not invariant to a latitude-dependent multiplier: weighting by
    cell area would tilt the answer toward the tropics. Area weighting belongs to
    the reconstruction, which this script never forms.
    """
    mask = fdr_correction(pval, alpha_FDR=alpha)
    return slope.where(mask | slope.isnull(), 0.0)

# ============================================================================
# LOAD DATA
# ============================================================================

# Load SST anomalies (using selected datasets)
sst_dict = {}
for sst_name in SST_DATASETS_TO_USE:
    nc_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()
        print(f"Loaded: {sst_name}")

# Load precipitation anomalies (using selected datasets)
precip_dict = {}
for p_name in PRECIP_DATASETS_TO_USE:
    nc_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
    if os.path.exists(nc_path):
        precip_dict[p_name] = xr.open_dataarray(nc_path)
        print(f"Loaded: {p_name}")

# ============================================================================
# COMPUTE FULL-PERIOD ENSEMBLE MEAN SENSITIVITY (ALL BASINS)
# ============================================================================

print("\nComputing full-period ensemble mean sensitivity for all basins...")

all_full_period_sensitivities = []

for sst_name, sst_anom in sst_dict.items():
    for p_name, p_anom in precip_dict.items():
        print(f"  Processing: {p_name} vs {sst_name}")

        # Full period for this member: ANALYSIS_START to ANALYSIS_END, the same
        # for every member. Fail loudly if a member doesn't cover all of it.
        common_time = np.intersect1d(p_anom['time'].values, sst_anom['time'].values)
        common_time = common_time[(common_time >= np.datetime64(ANALYSIS_START))
                                  & (common_time <= np.datetime64(ANALYSIS_END))]
        if (str(common_time[0])[:7] != ANALYSIS_START[:7]
                or str(common_time[-1])[:7] != ANALYSIS_END[:7]):
            raise ValueError(
                f"{p_name} x {sst_name} covers {str(common_time[0])[:7]} to "
                f"{str(common_time[-1])[:7]}, not the full {ANALYSIS_START[:7]} to "
                f"{ANALYSIS_END[:7]} analysis period"
            )
        print(f"    Full-period record: {str(common_time[0])[:7]} to {str(common_time[-1])[:7]}"
              f" ({len(common_time)} months)")

        p_common = p_anom.sel(time=common_time)
        sst_common = sst_anom.sel(time=common_time)
        
        # Detrend
        p_detrended = detrend_dim(p_common, 'time').astype(np.float32)
        sst_detrended = detrend_dim(sst_common, 'time').astype(np.float32)
        
        # Regression
        slope, _, pval = xr.apply_ufunc(
            regression_slope_se,
            sst_detrended,
            p_detrended,
            input_core_dims=[['time'], ['time']],
            vectorize=True,
            output_core_dims=[[], [], []],
            output_dtypes=[np.float32, np.float32, np.float32]
        )

        # FDR correction, per member, before the ensemble mean
        slope_sig = fdr_mask_sensitivity(slope, pval)

        all_full_period_sensitivities.append(slope_sig)

# Compute ensemble mean
full_period_beta = xr.concat(all_full_period_sensitivities, dim='ensemble').mean(dim='ensemble')
print(f"Full-period ensemble mean computed from {len(all_full_period_sensitivities)} model combinations")

# ============================================================================
# DETERMINE TIME WINDOWS (ONCE, BEFORE PARALLELIZATION)
# ============================================================================
#
# The windows must be built from the calendar months common to EVERY selected
# ensemble member, not just one arbitrary pair. Each member's own record can
# end at a different date (e.g. GPCP through Dec 2025, CRU only through Dec
# 2024) -- using one pair's time index positions to slice into a shorter
# member's array (as this used to do) silently truncates/misaligns that
# member's later windows instead of raising an error. Every date in
# `common_time` below is guaranteed present in every member's own time index,
# so downstream `.sel(time=...)` by date label is always exact.

common_time = reduce(
    np.intersect1d,
    [p.time.values for p in precip_dict.values()] + [s.time.values for s in sst_dict.values()],
)
common_time = common_time[(common_time >= np.datetime64(ANALYSIS_START))
                          & (common_time <= np.datetime64(ANALYSIS_END))]

n_times = len(common_time)
window_months = WINDOW_SIZE * 12

# Calculate number of windows
n_windows = (n_times - window_months) // WINDOW_STEP + 1
print(f"\nCommon period across all {len(precip_dict) * len(sst_dict)} ensemble members: "
      f"{str(common_time[0])[:7]} to {str(common_time[-1])[:7]}")
print(f"Total time points: {n_times}")
print(f"Window size: {window_months} months")
print(f"Window step: {WINDOW_STEP} months")
print(f"Number of 30-year windows: {n_windows}")

TIME_PERIOD = f"{str(common_time[0])[:7]} to {str(common_time[-1])[:7]}"

# Pre-compute rolling windows indices to avoid repeated indexing
window_indices = []
rolling_years = []
for window_idx in range(n_windows):
    window_start = window_idx * WINDOW_STEP
    window_end = window_start + window_months
    window_indices.append((window_start, window_end))
    last_year = int(str(common_time[window_end-1])[:4])
    rolling_years.append(last_year)

# ============================================================================
# DEFINE FUNCTION TO PROCESS SINGLE DATASET PAIR (FOR PARALLELIZATION)
# ============================================================================

def process_dataset_pair(sst_name, sst_anom, p_name, p_anom,
                        common_time_all, window_indices, rolling_years):
    """
    Process all windows for one SST-Precip pair.

    `common_time_all` is the calendar-month grid common to every selected
    SST-precip pair (computed once in DETERMINE TIME WINDOWS above), so each
    window below is selected by actual date labels rather than by index
    position into this pair's own time array. That guarantees every window
    is a complete WINDOW_SIZE-year span for this member -- a member with a
    shorter or differently-aligned record than another can never end up with
    a truncated or shifted window.

    Returns:
        windows_stacked: xarray.DataArray with dims (window, spatial_dims...)
    """

    window_sensitivities_all = []

    for window_idx, (window_start, window_end) in enumerate(window_indices):
        window_time = common_time_all[window_start:window_end]

        # Select window
        p_window = p_anom.sel(time=window_time)
        sst_window = sst_anom.sel(time=window_time)
        
        # Detrend
        p_detrended = detrend_dim(p_window, 'time').astype(np.float32)
        sst_detrended = detrend_dim(sst_window, 'time').astype(np.float32)
        
        # Regression
        slope, _, pval = xr.apply_ufunc(
            regression_slope_se,
            sst_detrended,
            p_detrended,
            input_core_dims=[['time'], ['time']],
            vectorize=True,
            output_core_dims=[[], [], []],
            output_dtypes=[np.float32, np.float32, np.float32]
        )

        # FDR correction, applied window by window
        slope_sig = fdr_mask_sensitivity(slope, pval)

        window_sensitivities_all.append(slope_sig)
    
    # Stack windows along new dimension for this dataset pair
    windows_stacked = xr.concat(window_sensitivities_all, dim='window')
    
    return windows_stacked

# ============================================================================
# COMPUTE ROLLING WINDOW SENSITIVITIES (ALL BASINS) - PARALLELIZED
# ============================================================================

print("\nComputing rolling window sensitivities for all basins (parallelized)...")

# Create list of tasks
tasks = []
for sst_name, sst_anom in sst_dict.items():
    for p_name, p_anom in precip_dict.items():
        tasks.append((sst_name, sst_anom, p_name, p_anom))

print(f"Processing {len(tasks)} dataset pairs in parallel...")

# Run in parallel: n_jobs=-1 uses all available cores
rolling_beta_ensemble = Parallel(n_jobs=-1, verbose=10)(
    delayed(process_dataset_pair)(sst_name, sst_anom, p_name, p_anom,
                                   common_time, window_indices, rolling_years)
    for sst_name, sst_anom, p_name, p_anom in tasks
)

print(f"Computed {len(rolling_beta_ensemble)} dataset pairs × {n_windows} windows")

# ============================================================================
# STACK ALL WINDOWS AND COMPUTE CORRELATIONS WITH SINGLE UFUNC CALL
# ============================================================================

print("\nStacking all windows for vectorized correlation computation...")

# Stack windows along new dimension: shape will be (window, ensemble, lat, lon, ...)
rolling_beta_all_windows = xr.concat(rolling_beta_ensemble, dim='ensemble')
rolling_beta_all_windows['window'] = rolling_years

print(f"All windows stacked: {rolling_beta_all_windows.shape}")

def compute_pattern_corr_all_windows(full_beta_values, window_ensemble_stack):
    """
    Returns per-ensemble-member correlations, not just mean/std.
    - corr_all: (window, ensemble) - correlation for each member and window
    """
    full_flat = full_beta_values.flatten()
    n_windows = window_ensemble_stack.shape[0]
    n_ensemble = window_ensemble_stack.shape[1]
    
    corr_all = np.full((n_windows, n_ensemble), np.nan, dtype=np.float32)
    
    for w_idx in range(n_windows):
        for e_idx in range(n_ensemble):
            window_flat = window_ensemble_stack[w_idx, e_idx].flatten()
            valid_mask = ~(np.isnan(full_flat) | np.isnan(window_flat))
            full_valid = full_flat[valid_mask]
            window_valid = window_flat[valid_mask]
            if len(full_valid) > 1:
                corr_all[w_idx, e_idx] = pearsonr(full_valid, window_valid)[0]
    
    return corr_all

print("\nComputing pattern correlations for all basins and windows with single ufunc call...")

# Single apply_ufunc call across all basins at once
result = xr.apply_ufunc(
    compute_pattern_corr_all_windows,
    full_period_beta,
    rolling_beta_all_windows,
    input_core_dims=[['lat', 'lon'], ['window', 'ensemble', 'lat', 'lon']],
    vectorize=True,
    output_core_dims=[['window', 'ensemble']],   # now one output with both dims
    output_dtypes=[np.float32]
)

pattern_corr_per_member = result  # dims: (basin, window, ensemble) or just (window, ensemble)

# ============================================================================
# SAVE STATEMENTS
# ============================================================================

# Create output filename with descriptive name
output_filename = OUTPUTS_DIR / 'pattern_correlations_all_basins_fdr_corrected.nc'


# Assign dataset labels to ensemble dimension
dataset_labels = [f"{p_name}_{sst_name}" for sst_name, _, p_name, _ in tasks]
pattern_corr_per_member['ensemble'] = dataset_labels
pattern_corr_per_member = pattern_corr_per_member.assign_coords(
    ensemble=dataset_labels,
    window=rolling_years,
)

results_ds = xr.Dataset(
    {
        'pattern_corr_per_member': pattern_corr_per_member,  # (window, ensemble)
        'full_period_beta':        full_period_beta           # (lat, lon)
    }
)

results_ds.attrs['description']        = ('Pattern correlation analysis results for all basins, '
                                          'computed on FDR-corrected sensitivities')
results_ds.attrs['precip_datasets']    = ', '.join(PRECIP_DATASETS_TO_USE)
results_ds.attrs['sst_datasets']       = ', '.join(SST_DATASETS_TO_USE)
results_ds.attrs['time_period']        = TIME_PERIOD
results_ds.attrs['window_size_years']  = WINDOW_SIZE
results_ds.attrs['window_step_months'] = WINDOW_STEP
results_ds.attrs['alpha_FDR']          = ALPHA
results_ds.attrs['fdr_family']         = 'lat/lon grid cells, one family per basin'

# Save to NetCDF
results_ds.to_netcdf(output_filename)
print(f"\nResults saved to: {output_filename}")
