#!/usr/bin/env python
# coding: utf-8
"""
Determine block bootstrap lengths from SST temporal autocorrelation.

Author: Flora Perlmutter

Description
-----------
Computes the autocorrelation function (ACF) and partial autocorrelation
function (PACF) for each SST dataset over the 1981-2019 period, after
detrending. The suggested block length is the first lag at which the
ACF drops below 1/e (~0.37). Results are printed, plotted, and saved.

Outputs
-------
  Processed/sst_block_lengths.json       — block length per SST dataset
  Figures/sst_acf_pacf_comparison.png    — combined ACF + PACF panel
  Figures/sst_acf_block_length.png       — ACF-only panel (paper quality)

Usage
-----
    python determine_block_lengths.py
"""

import json
import warnings
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from statsmodels.tsa.stattools import acf, pacf
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA
from regression_functions import detrend_dim

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
FIGURES_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Figures"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SST_NAMES        = ["ERSSTv6", "COBE-SST3"]
ANALYSIS_PERIOD  = slice("1981-01-01", "2019-12-01")
MAX_LAG          = 24
ACF_THRESHOLD    = 1 / np.e   # ~0.37


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def calculate_autocorr_and_pacf(
    data: xr.DataArray,
    max_lag: int = MAX_LAG,
    threshold: float = ACF_THRESHOLD,
) -> tuple:
    """
    Compute ACF and PACF for a spatiotemporal DataArray and suggest a block length.

    Spatial dimensions are averaged before computing statistics. The suggested
    block length is the first lag at which |ACF| drops below `threshold`.

    Parameters
    ----------
    data      : xr.DataArray with a 'time' dimension
    max_lag   : maximum lag to compute
    threshold : ACF magnitude below which autocorrelation is considered negligible

    Returns
    -------
    block_length : int
    autocorr     : np.ndarray (max_lag + 1,)
    pacf_values  : np.ndarray (max_lag + 1,)
    """
    # Average over all spatial dimensions
    spatial_dims = [d for d in data.dims if d != "time"]
    data_mean    = data.mean(dim=spatial_dims) if spatial_dims else data
    data_clean   = data_mean.dropna(dim="time")

    autocorr    = acf(data_clean.values,  nlags=max_lag, fft=True)
    pacf_values = pacf(data_clean.values, nlags=max_lag, method="ywm")

    below = np.where(np.abs(autocorr) < threshold)[0]
    block_length = max(1, int(below[0]) if len(below) > 0 else max_lag)

    return block_length, autocorr, pacf_values


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_acf_pacf(results_dict: dict, n_time: int) -> None:
    """Combined 2×2 ACF + PACF panel for all SST datasets."""
    dataset_names = list(results_dict.keys())
    fig, axes     = plt.subplots(len(dataset_names), 2, figsize=(14, 5 * len(dataset_names)))
    ci = 1.96 / np.sqrt(n_time)

    for idx, name in enumerate(dataset_names):
        res          = results_dict[name]
        lags         = res["lags"]
        autocorr     = res["autocorr"]
        pacf_values  = res["pacf"]
        block_length = res["block_length"]

        # ACF
        ax = axes[idx, 0]
        ax.bar(lags, autocorr, color="steelblue", alpha=0.7)
        ax.axhline(ACF_THRESHOLD, color="red",   linestyle="--", linewidth=2,
                   label=f"1/e threshold ({ACF_THRESHOLD:.3f})")
        ax.axhline(0,            color="black",  linestyle="-",  linewidth=0.5)
        ax.axvline(block_length, color="green",  linestyle="--", linewidth=2,
                   label=f"Block length = {block_length}")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{name} — ACF", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # PACF
        ax = axes[idx, 1]
        ax.bar(lags, pacf_values, color="coral", alpha=0.7)
        ax.axhline( ci, color="blue", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"95% CI (±{ci:.3f})")
        ax.axhline(-ci, color="blue", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(0,   color="black", linestyle="-",  linewidth=0.5)
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Partial Autocorrelation")
        ax.set_title(f"{name} — PACF", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "sst_acf_pacf_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_acf_only(results_dict: dict) -> None:
    """ACF-only panel — one subplot per SST dataset."""
    dataset_names = list(results_dict.keys())
    fig, axes     = plt.subplots(len(dataset_names), 1,
                                 figsize=(7, 5 * len(dataset_names)))
    if len(dataset_names) == 1:
        axes = [axes]

    for idx, name in enumerate(dataset_names):
        res          = results_dict[name]
        block_length = res["block_length"]

        ax = axes[idx]
        ax.bar(res["lags"], res["autocorr"], color="steelblue", alpha=0.7)
        ax.axhline(ACF_THRESHOLD, color="red",  linestyle="--", linewidth=2,
                   label=f"1/e threshold ({ACF_THRESHOLD:.3f})")
        ax.axhline(0,             color="black", linestyle="-",  linewidth=0.5)
        ax.axvline(block_length,  color="green", linestyle="--", linewidth=2,
                   label=f"Block length = {block_length}")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{name} — ACF", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "sst_acf_block_length.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load SST anomalies
    sst_dict = {}
    for sst_name in SST_NAMES:
        path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if path.exists():
            sst_dict[sst_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found — skipping {sst_name}")

    if not sst_dict:
        raise FileNotFoundError("No SST anomaly files found. Run load_and_process_obs.py first.")

    # Compute ACF / PACF
    print("Computing temporal autocorrelation and PACF for SST datasets...")
    results_dict = {}
    n_time       = None

    for name, da in sst_dict.items():
        print(f"\n  Processing {name}...")
        da_constrained = da.sel(time=ANALYSIS_PERIOD)
        da_detrended   = detrend_dim(da_constrained, "time")
        block_length, autocorr, pacf_values = calculate_autocorr_and_pacf(da_detrended)

        results_dict[name] = {
            "block_length": block_length,
            "autocorr":     autocorr,
            "pacf":         pacf_values,
            "lags":         np.arange(len(autocorr)),
        }
        n_time = int(da_constrained.time.size)

        print(f"    Suggested block length: {block_length} months")
        print(f"    ACF(lag=1):  {autocorr[1]:.3f}")
        print(f"    PACF(lag=1): {pacf_values[1]:.3f}")

    # Plots
    print("\nGenerating plots...")
    plot_acf_pacf(results_dict, n_time)
    plot_acf_only(results_dict)

    # Save block lengths as JSON (no pickle dependency)
    block_lengths = {name: res["block_length"] for name, res in results_dict.items()}
    out_json      = OUTPUTS_DIR / "sst_block_lengths.json"
    with open(out_json, "w") as f:
        json.dump(block_lengths, f, indent=2)
    print(f"\n  Saved block lengths → {out_json}")

    # Summary
    print("\nSummary")
    print("-" * 60)
    for name, res in results_dict.items():
        print(f"  {name}:")
        print(f"    Block length: {res['block_length']} months")
        print(f"    ACF(1):       {res['autocorr'][1]:.3f}")
        print(f"    PACF(1):      {res['pacf'][1]:.3f}")


if __name__ == "__main__":
    main()