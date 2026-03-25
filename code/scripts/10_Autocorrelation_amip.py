#!/usr/bin/env python
# coding: utf-8
"""
Determine block bootstrap lengths from AMIP SST temporal autocorrelation.

Author: Flora Perlmutter

Description
-----------
AMIP counterpart to determine_block_lengths.py. Loads SST dataset names
from the AMIP manifest JSON, constrains to the AMIP analysis period
(1981-2014), detrends, and computes ACF to suggest block lengths.

Outputs
-------
  Processed/amip_block_lengths.json          — block length per AMIP SST dataset
  Figures/sst_acf_block_length_amip.png      — ACF panel for all AMIP SST models

Usage
-----
    python determine_block_lengths_amip.py

Note: Run load_and_process_amip.py first to generate the manifest and
      SST anomaly files this script depends on.
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
from paths import CMIG_DATA, DATA_DIR
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
ANALYSIS_PERIOD = slice("1981-01-01", "2014-12-01")   # AMIP period
MAX_LAG         = 24

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

def plot_acf(results_dict: dict) -> None:
    """ACF panel — one subplot per AMIP SST dataset."""
    names     = list(results_dict.keys())
    fig_h     = max(5 * len(names), 8)
    fig, axes = plt.subplots(len(names), 1, figsize=(7, fig_h))
    if len(names) == 1:
        axes = [axes]

    for idx, name in enumerate(names):
        res          = results_dict[name]
        block_length = res["block_length"]
        ax           = axes[idx]

        ax.bar(res["lags"], res["autocorr"], color="steelblue", alpha=0.7)
        ax.axhline(ACF_THRESHOLD, color="red",   linestyle="--", linewidth=2,
                   label=f"1/e threshold ({ACF_THRESHOLD:.3f})")
        ax.axhline(0,             color="black",  linestyle="-",  linewidth=0.5)
        ax.axvline(block_length,  color="green",  linestyle="--", linewidth=2,
                   label=f"Block length = {block_length}")
        ax.set_xlabel("Lag (months)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"{name} — ACF", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "sst_acf_block_length_amip.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load AMIP SST names from manifest
    manifest_path = OUTPUTS_DIR / "amip_dataset_names.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run load_and_process_amip.py first."
        )
    with open(manifest_path) as f:
        names = json.load(f)

    # Load AMIP SST anomalies
    sst_dict = {}
    for sst_name in names["sst_datasets"]:
        path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if path.exists():
            sst_dict[sst_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found — skipping {sst_name}")

    if not sst_dict:
        raise FileNotFoundError(
            "No AMIP SST anomaly files found. Run load_and_process_amip.py first."
        )

    # Compute ACF for each dataset
    print("Computing temporal autocorrelation for AMIP SST datasets...")
    results_dict = {}

    for name, da in sst_dict.items():
        print(f"\n  Processing {name}...")
        da_constrained = da.sel(time=ANALYSIS_PERIOD)
        da_detrended   = detrend_dim(da_constrained, "time")

        block_length, autocorr, _ = calculate_autocorr_and_pacf(
            da_detrended, max_lag=MAX_LAG
        )

        results_dict[name] = {
            "block_length": block_length,
            "autocorr":     autocorr,
            "lags":         np.arange(len(autocorr)),
        }

        print(f"    Suggested block length: {block_length} months")
        print(f"    ACF(lag=1): {autocorr[1]:.3f}")

    # Plot
    print("\nGenerating ACF plot...")
    plot_acf(results_dict)

    # Save block lengths as JSON
    block_lengths = {name: res["block_length"] for name, res in results_dict.items()}
    out_json      = DATA_DIR / "amip_block_lengths.json"
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


if __name__ == "__main__":
    main()