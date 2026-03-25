#!/usr/bin/env python
# coding: utf-8
"""
Bootstrap standard error estimation for AMIP linear regression.

Author: Flora Perlmutter
Refactored for reproducibility: February 24, 2026

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

Usage
-----
    python run_bootstrap_se_amip.py --pair-index 3
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import psutil
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

# --- project paths ---
from paths import CMIG_DATA
from regression_functions import waterbasin, detrend_dim, grid_area, fdr_correction, regression_slope_se

# Reuse bootstrap infrastructure from the observational script
from run_bootstrap_se_obs import (
    IncrementalStats,
    run_single_bootstrap,
    bootstrap_se_incremental,
    print_memory_status,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ALPHA        = 0.05
N_BOOTSTRAP  = 100
BLOCK_SIZE   = 8    # default for AMIP (shorter period than obs)
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_pair(
    p_name: str,
    p_anom: xr.DataArray,
    sst_name: str,
    sst_anom: xr.DataArray,
    block_size: int,
) -> tuple:
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")

    # Align time and load
    p_anom   = p_anom.load()
    sst_anom = sst_anom.load()
    common_time = np.intersect1d(p_anom["time"].values, sst_anom["time"].values)
    p_anom   = p_anom.sel(time=common_time)
    sst_anom = sst_anom.sel(time=common_time)

    # Detrend
    print("Detrending...")
    p_detrended   = detrend_dim(p_anom,   "time").astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, "time").astype(np.float32)

    # Regression
    print("Running regression...")
    slope, slope_se, pval = xr.apply_ufunc(
        regression_slope_se,
        sst_detrended, p_detrended,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32],
    )
    print(f"  Valid slopes: {int((~np.isnan(slope)).sum().values)}")
    print_memory_status("After regression")

    # Area weights
    print("Precomputing area weights...")
    area_precomputed = grid_area(slope).astype(np.float32) * xr.ones_like(slope)
    print_memory_status("After area computation")

    # FDR correction
    print("Applying FDR correction...")
    slope_area = area_precomputed * slope
    fdr_mask   = fdr_correction(pval, alpha_FDR=ALPHA)
    slope_sig  = slope_area.where(fdr_mask)
    print(f"  Significant cells: {int(fdr_mask.sum().values)}")

    # Reconstruction
    print("Computing reconstruction...")
    reconstruction = (sst_anom * slope_sig).sum(("lat", "lon"))
    print_memory_status("After reconstruction")

    # Bootstrap SE
    print(f"Bootstrap block size: {block_size} months")
    reconstruction_se, marginal_sensitivity_se = bootstrap_se_incremental(
        sst_detrended, p_detrended, sst_anom, area_precomputed,
        ALPHA, N_BOOTSTRAP, block_size, RANDOM_SEED,
    )
    print_memory_status("After bootstrap")

    # Correlation
    corr = xr.corr(reconstruction, p_anom, "time")

    # Clean up
    del p_detrended, sst_detrended, sst_anom, slope, slope_se, pval
    del area_precomputed, slope_area, fdr_mask
    gc.collect()
    print_memory_status("Before return")

    result = {
        "model_id":               "P ~ β*SST",
        "description":            "Linear regression with bootstrap SE (AMIP)",
        "variable":               "precip",
        "alpha":                  ALPHA,
        "n_bootstrap":            N_BOOTSTRAP,
        "reconstruction":         reconstruction,
        "reconstruction_se":      reconstruction_se,
        "observed_precip":        p_anom,
        "correlation":            corr,
        "marginal_sensitivity":     slope_sig,
        "marginal_sensitivity_se":  marginal_sensitivity_se,
    }

    print(f"\n  SUCCESS: {p_name} vs {sst_name}")
    return (p_name, sst_name), result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bootstrap SE for one AMIP pair")
    parser.add_argument("--pair-index", type=int, required=True,
                        help="Index of model pair to process (0-based, for sbatch array)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("BOOTSTRAP SE ANALYSIS — AMIP")
    print("=" * 80)
    print(f"  Pair index:  {args.pair_index}")
    print(f"  Bootstraps:  {N_BOOTSTRAP}")
    print(f"  Alpha:       {ALPHA}")

    # Load AMIP dataset manifest
    manifest_path = OUTPUTS_DIR / "amip_dataset_names.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run load_and_process_amip.py first."
        )
    with open(manifest_path) as f:
        names = json.load(f)

    # Load block lengths from autocorrelation analysis
    block_lengths_path = OUTPUTS_DIR / "amip_block_lengths.json"
    if block_lengths_path.exists():
        with open(block_lengths_path) as f:
            amip_block_lengths = json.load(f)
        print(f"  Loaded block lengths from {block_lengths_path}")
    else:
        print(f"  Warning: {block_lengths_path} not found — using default {BLOCK_SIZE} months")
        amip_block_lengths = {}

    # Load precipitation anomalies
    precip_dict = {}
    for p_name in names["precip_datasets"]:
        path = OUTPUTS_DIR / f"amip_precip_anom_{p_name}.nc"
        if path.exists():
            precip_dict[p_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found")

    # Load SST anomalies
    sst_dict = {}
    for sst_name in names["sst_datasets"]:
        path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if path.exists():
            sst_dict[sst_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found")

    print(f"\n  Loaded {len(precip_dict)} precip datasets, {len(sst_dict)} SST datasets")

    # Pairs matched by model ID (same model for pr and tas)
    common_models = sorted(set(precip_dict) & set(sst_dict))
    print(f"  Common models: {len(common_models)}")

    pairs = [
        (model_id, precip_dict[model_id], model_id, sst_dict[model_id])
        for model_id in common_models
    ]

    if not pairs:
        print("ERROR: No valid pairs found.")
        sys.exit(1)

    if args.pair_index >= len(pairs):
        print(f"ERROR: pair-index {args.pair_index} >= total pairs {len(pairs)}")
        sys.exit(1)

    p_name, p_da, sst_name, sst_da = pairs[args.pair_index]
    print(f"\nProcessing pair {args.pair_index}/{len(pairs)-1}: {p_name} vs {sst_name}")

    # Block size: use computed value if available, else default
    block_size = amip_block_lengths.get(sst_name, BLOCK_SIZE)
    if sst_name in amip_block_lengths:
        print(f"  Block size: {block_size} months (from autocorrelation analysis)")
    else:
        print(f"  Block size: {block_size} months (default — no autocorr data for {sst_name})")

    (p_name, sst_name), result = process_pair(p_name, p_da, sst_name, sst_da, block_size)

    # Save
    out = OUTPUTS_DIR / f"global_linear_regression_bootstrap_amip_{p_name}_{sst_name}.nc"
    xr.Dataset({
        "reconstruction":          result["reconstruction"],
        "reconstruction_se":       result["reconstruction_se"],
        "observed_precip":         result["observed_precip"],
        "correlation":             result["correlation"],
        "marginal_sensitivity":    result["marginal_sensitivity"],
        "marginal_sensitivity_se": result["marginal_sensitivity_se"],
    }, attrs={
        "model_id":       result["model_id"],
        "description":    result["description"],
        "variable":       result["variable"],
        "alpha":          result["alpha"],
        "n_bootstrap":    result["n_bootstrap"],
        "precip_dataset": p_name,
        "sst_dataset":    sst_name,
    }).to_netcdf(out)

    print(f"\n  Saved → {out}")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()