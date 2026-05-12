#!/usr/bin/env python
# coding: utf-8
"""
Process reconstruction cross-validation results: Regression vs PCA.

Author: Flora Perlmutter

Description
-----------
Loads per-pair CV results (produced by run_cv_reconstruction.py),
combines across the ensemble, computes ensemble-mean skill metrics
and method differences, and saves processed outputs for plotting.

Output
------
  <DATA_DIR>/cv_reconstruction_ensemble_metrics.nc
  <DATA_DIR>/cv_reconstruction_method_comparison.csv
"""
import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import CMIG_DATA, DATA_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CV_DIR       = DATA_DIR          # where run_cv_reconstruction.py saved output
OUTPUTS_DIR  = DATA_DIR

PRECIP_DATASETS = [
    "GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN",
]
SST_DATASETS = ["ERSSTv6", "COBE-SST3"]

METRICS = ["re", "correlation", "rmse", "nrmse", "mae"]


# =============================================================================
# LOADING
# =============================================================================
def load_cv_results(verbose=True):
    """
    Load all cv_reconstruction_{P}_{SST}.nc files and combine
    along a 'pair' dimension.

    Returns
    -------
    xr.Dataset with dims (pair, basin, method) for each metric,
    plus (pair, basin, time) for predictions.
    """
    datasets = []
    pair_names = []

    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            path = CV_DIR / f"cv_reconstruction_{p_name}_{sst_name}.nc"
            if not path.exists():
                if verbose:
                    print(f"  [MISSING] {path.name}")
                continue

            ds = xr.open_dataset(path)
            pair_label = f"{p_name}_{sst_name}"
            ds = ds.expand_dims(pair=[pair_label])
            datasets.append(ds)
            pair_names.append(pair_label)
            ds.close()

            if verbose:
                print(f"  Loaded {pair_label}")

    if not datasets:
        raise FileNotFoundError("No CV reconstruction files found.")

    ds_combined = xr.concat(datasets, dim="pair")

    if verbose:
        print(f"\nCombined dataset:")
        print(f"  Dimensions: {dict(ds_combined.sizes)}")
        print(f"  Pairs: {len(pair_names)}")

    return ds_combined


# =============================================================================
# ENSEMBLE STATISTICS
# =============================================================================
def compute_ensemble_stats(ds, verbose=True):
    """
    Compute ensemble mean and std of skill metrics across pairs.

    Parameters
    ----------
    ds : xr.Dataset with dims (pair, basin, method) for metrics

    Returns
    -------
    ds_mean : xr.Dataset  — mean across pairs, dims (basin, method)
    ds_std  : xr.Dataset  — std across pairs, dims (basin, method)
    """
    metric_vars = [v for v in ds.data_vars if v in METRICS]

    ds_metrics = ds[metric_vars]
    ds_mean = ds_metrics.mean(dim="pair", skipna=True)
    ds_std = ds_metrics.std(dim="pair", skipna=True)

    if verbose:
        print(f"\nEnsemble statistics computed for: {metric_vars}")
        print(f"  Shape: {dict(ds_mean.sizes)}")

    return ds_mean, ds_std


# =============================================================================
# METHOD COMPARISON TABLE
# =============================================================================
def build_comparison_table(ds_mean, ds_std, verbose=True):
    """
    Build a summary DataFrame comparing regression vs PCA.

    For each metric, computes:
      - Ensemble-mean value for each method (averaged across basins)
      - Ensemble-std
      - Basin-level win counts
      - Mean difference (PCA - regression)

    Returns
    -------
    pd.DataFrame with one row per metric
    """
    rows = []

    for metric in METRICS:
        if metric not in ds_mean.data_vars:
            continue

        da_mean = ds_mean[metric]  # (basin, method)
        da_std = ds_std[metric]

        reg_vals = da_mean.isel(method=0).values
        pca_vals = da_mean.isel(method=1).values
        
        valid = np.isfinite(reg_vals) & np.isfinite(pca_vals)

        if not valid.any():
            continue

        reg_valid = reg_vals[valid]
        pca_valid = pca_vals[valid]

        # For RMSE, MAE, NRMSE: lower is better → PCA wins if pca < reg
        # For RE, correlation: higher is better → PCA wins if pca > reg
        lower_is_better = metric in ("rmse", "nrmse", "mae")

        if lower_is_better:
            pca_wins = int(np.sum(pca_valid < reg_valid))
            reg_wins = int(np.sum(reg_valid < pca_valid))
        else:
            pca_wins = int(np.sum(pca_valid > reg_valid))
            reg_wins = int(np.sum(reg_valid > pca_valid))

        ties = int(valid.sum()) - pca_wins - reg_wins

        rows.append({
            "metric": metric,
            "reg_mean": float(np.nanmean(reg_valid)),
            "reg_std": float(np.nanstd(reg_valid)),
            "pca_mean": float(np.nanmean(pca_valid)),
            "pca_std": float(np.nanstd(pca_valid)),
            "diff_pca_minus_reg": float(np.nanmean(pca_valid - reg_valid)),
            "pca_wins": pca_wins,
            "reg_wins": reg_wins,
            "ties": ties,
            "n_basins": int(valid.sum()),
        })

    df = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 80)
        print("METHOD COMPARISON SUMMARY (ensemble-mean, averaged across basins)")
        print("=" * 80)
        print(df.to_string(index=False))

    return df


# =============================================================================
# PER-BASIN DETAILED TABLE
# =============================================================================
def build_per_basin_table(ds_mean, verbose=True):
    """
    Build a per-basin DataFrame with ensemble-mean metrics for both methods.

    Returns
    -------
    pd.DataFrame with columns: basin, metric, regression, pca, difference, winner
    """
    rows = []

    for metric in METRICS:
        if metric not in ds_mean.data_vars:
            continue

        da = ds_mean[metric]  # (basin, method)
        basins = da["basin"].values

        reg_vals = da.isel(method=0).values
        pca_vals = da.isel(method=1).values

        lower_is_better = metric in ("rmse", "nrmse", "mae")

        for i, bid in enumerate(basins):
            r = reg_vals[i]
            p = pca_vals[i]

            if not (np.isfinite(r) and np.isfinite(p)):
                winner = "N/A"
            elif lower_is_better:
                winner = "pca" if p < r else ("regression" if r < p else "tie")
            else:
                winner = "pca" if p > r else ("regression" if r > p else "tie")

            rows.append({
                "basin": bid,
                "metric": metric,
                "regression": float(r),
                "pca": float(p),
                "difference": float(p - r),
                "winner": winner,
            })

    df = pd.DataFrame(rows)

    if verbose:
        print(f"\nPer-basin table: {len(df)} rows "
              f"({len(df['basin'].unique())} basins × {len(df['metric'].unique())} metrics)")

    return df


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("PROCESSING RECONSTRUCTION CV RESULTS")
    print("=" * 80)

    # Step 1: Load
    print("\nStep 1: Loading CV results...")
    ds = load_cv_results(verbose=True)

    # Step 2: Ensemble statistics
    print("\nStep 2: Computing ensemble statistics...")
    ds_mean, ds_std = compute_ensemble_stats(ds, verbose=True)

    # Step 3: Summary comparison
    print("\nStep 3: Building method comparison summary...")
    comparison_df = build_comparison_table(ds_mean, ds_std, verbose=True)

    # Step 4: Per-basin detail
    print("\nStep 4: Building per-basin table...")
    basin_df = build_per_basin_table(ds_mean, verbose=True)

    # Step 5: Save
    print("\nStep 5: Saving outputs...")

    # Save ensemble-mean metrics as NetCDF
    out_nc = OUTPUTS_DIR / "cv_reconstruction_ensemble_metrics.nc"
    ds_out = xr.merge([
        ds_mean.rename({v: f"{v}_mean" for v in ds_mean.data_vars}),
        ds_std.rename({v: f"{v}_std" for v in ds_std.data_vars}),
    ])
    ds_out.attrs["description"] = (
        "Ensemble-mean and std of LOO reconstruction CV metrics: regression vs PCA"
    )
    ds_out.to_netcdf(out_nc)
    print(f"  Saved → {out_nc.name}")

    # Save comparison summary as CSV
    out_summary = OUTPUTS_DIR / "cv_reconstruction_method_comparison.csv"
    comparison_df.to_csv(out_summary, index=False)
    print(f"  Saved → {out_summary.name}")

    # Save per-basin detail as CSV
    out_basin = OUTPUTS_DIR / "cv_reconstruction_per_basin.csv"
    basin_df.to_csv(out_basin, index=False)
    print(f"  Saved → {out_basin.name}")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80 + "\n")

    return ds_mean, ds_std, comparison_df, basin_df


if __name__ == "__main__":
    main()