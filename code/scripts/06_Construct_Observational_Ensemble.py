#!/usr/bin/env python
# coding: utf-8
"""
Load and process observational precipitation and SST data.

Author: Flora Perlmutter

Description
-----------
This script loads and preprocesses all observational precipitation and SST
datasets used in the SST-precipitation sensitivity analysis. It:
  1. Loads and preprocesses 8 precipitation datasets
  2. Loads and preprocesses ERSSTv5 and COBE-SST2 SST datasets
  3. Determines the common overlapping time period
  4. Computes monthly anomalies for precipitation (basin-scaled) and SST
  5. Saves all processed data as NetCDF to the outputs directory
  6. Generates verification plots (saved to HPC figures directory)

Excluded datasets:
  - MERRA2_P: missing dates
  - CHIRPS: spatial extent limited to 50S-50N
  - ERA5-Land: known biases
  - MSWEP: lat/lon not evenly spaced, incompatible with waterbasin()

Usage
-----
    python load_and_process_obs.py
"""

import os
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import regionmask
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import gc

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA
from data_processing_functions import (
    fix_lons,
    fixdates,
    load_and_preprocess_data,
    mask_sea_ice,
    compute_monthly_anomaly,
    determine_common_time_period,
)
from regression_functions import waterbasin
from amip_functions import make_periodic

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
DATA_OBS_DIR = CMIG_DATA / "Data/Observations"
INTERIM_P    = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
INTERIM_SST    = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OUTPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
FIGURES_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Figures"

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
PRECIP_DATASETS = {
    "GPCP":        {"path": str(INTERIM_P / "P.GPCP.1979.2025.nc"), "var": "precip", "name": "GPCP"},
    "CRU":         {"path": str(DATA_OBS_DIR / "CRU_TS/pre/v4.09_1901_2024/cru_ts4.09.1901.2024.pre.dat.nc"), "var": "pre", "name": "CRU"},
    "GPCC":         {"path": str(DATA_OBS_DIR / "GPCC/v2025_monthly/05/*.nc"), "var": "precip",      "name": "GPCC"},
    "CPC":         {"path": str(INTERIM_P / "P.CPC.*.nc"), "var": "P", "name": "CPC"},
    "UDel":        {"path": str(DATA_OBS_DIR / "UDel/monthly/precip.mon.total.v501.nc"), "var": "precip", "name": "UDel"},
    "PREC":        {"path": str(DATA_OBS_DIR / "PREC_L/2026/precip.mon.mean.0.5x0.5.nc"), "var": "precip", "name": "PREC"},
    'TerraClimate': {'path': str(INTERIM_P / "P.TerraClimate.*.nc"), 'var': 'P', 'name': 'TerraClimate'},
    "REGEN":       {"path": str(INTERIM_P / "P.REGEN.1950-2016.nc"), "var": "P", "name": "REGEN"},
}

SST_PATHS = {
    "ERSSTv6":   INTERIM_SST / "ersstv6_preprocessed.nc",
    "COBE-SST3": INTERIM_SST / "cobe-sst3_regridded.nc",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_ocean_mask(target_da: xr.DataArray) -> xr.DataArray:
    """Build a boolean ocean mask on the target grid using Natural Earth land shapes."""
    print("  Creating ocean mask...")
    land_shp = shpreader.natural_earth(resolution="10m", category="physical", name="land")
    land_gdf = gpd.read_file(land_shp)
    land_mask = regionmask.mask_geopandas(land_gdf, target_da.lon, target_da.lat)
    return land_mask.isnull()   # True where ocean


# ---------------------------------------------------------------------------
# Section 1: Load precipitation
# ---------------------------------------------------------------------------

def load_precip() -> dict:
    print("\n" + "=" * 80)
    print("LOADING PRECIPITATION DATASETS")
    print("=" * 80)

    precip_raw = {}
    for p_name, p_meta in PRECIP_DATASETS.items():
        print(f"\n  Loading {p_name}...")
        try:
            p_da = load_and_preprocess_data(p_meta)
            precip_raw[p_name] = p_da
            print(f"    {p_name}: {p_da.time.size} timesteps, "
                  f"{p_da.lat.size} lats, {p_da.lon.size} lons")
        except Exception as e:
            print(f"    FAILED to load {p_name}: {e}")

    return precip_raw


# ---------------------------------------------------------------------------
# Section 2: Load SST
# ---------------------------------------------------------------------------

def load_sst() -> dict:
    print("\n" + "=" * 80)
    print("LOADING SST DATASETS")
    print("=" * 80)

    ersst = xr.open_dataset(SST_PATHS["ERSSTv6"])
    
    cobe = xr.open_dataset(SST_PATHS["COBE-SST3"])
    
    # Build mask on ERSST grid
    ocean_mask = _build_ocean_mask(ersst)
    
    # Apply mask
    cobe_masked = mask_sea_ice(
        cobe.where(ocean_mask)
    )
    
    ersst_masked = mask_sea_ice(ersst.where(ocean_mask))
    ersst_masked = ersst_masked.sst.squeeze()
    
    cobe_masked = cobe_masked.__xarray_dataarray_variable__
    
    print("  Masking land and sea ice...")
    sst_dict_raw = {
        "ERSSTv6":   ersst_masked,
        "COBE-SST3": cobe_masked,
    }

    return sst_dict_raw


# ---------------------------------------------------------------------------
# Section 3: Compute anomalies
# ---------------------------------------------------------------------------

def compute_anomalies(precip_raw: dict, sst_dict_raw: dict) -> tuple:
    common_start, common_end = determine_common_time_period(precip_raw, sst_dict_raw)
    common_period = slice(
        common_start.strftime("%Y-%m-%d"),
        common_end.strftime("%Y-%m-%d"),
    )

    print("\n" + "=" * 80)
    print("COMPUTING PRECIPITATION ANOMALIES WITH BASIN SCALING")
    print("=" * 80)
            
    for p_name, p_da in precip_raw.items():
        print(f"  Processing {p_name}...")
        try:
            p_basin = waterbasin(p_da)
            del p_da
            p_anom = compute_monthly_anomaly(p_basin, climatology_period=common_period).load()
            del p_basin
            
            # Save immediately and don't accumulate in dict (memory intensive)
            out = OUTPUTS_DIR / f"precip_anom_{p_name}.nc"
            p_anom.to_netcdf(out)
            del p_anom
            gc.collect()
            print(f"    Saved {p_name}")
        except Exception as e:
            print(f"    Failed to process {p_name}: {e}")
    
    print("\n" + "=" * 80)
    print("COMPUTING SST ANOMALIES")
    print("=" * 80)
    
    sst_anomalies = {}
    for sst_name, sst_da in sst_dict_raw.items():
        print(f"  Processing {sst_name}...")
        try:
            sst_anom = compute_monthly_anomaly(sst_da, climatology_period=common_period)
            sst_anom = sst_anom.assign_coords(dataset=sst_name)
            sst_anomalies[sst_name] = sst_anom
            print(f"    Anomalies computed: {sst_anom.time.size} timesteps")
        except Exception as e:
            print(f"    Failed to process {sst_name}: {e}")

    return sst_anomalies, common_period


# ---------------------------------------------------------------------------
# Section 4: Save outputs
# ---------------------------------------------------------------------------

def save_outputs(sst_anomalies: dict, sst_dict_raw: dict) -> None:
    print("\n" + "=" * 80)
    print("SAVING PROCESSED DATA")
    print("=" * 80)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for sst_name, sst_anom in sst_anomalies.items():
        out = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        sst_anom.to_netcdf(out)
        print(f"  Saved {sst_name} → {out}")

    for sst_name, sst_raw in sst_dict_raw.items():
        out = OUTPUTS_DIR / f"sst_raw_{sst_name}.nc"
        sst_raw.to_netcdf(out)
        print(f"  Saved {sst_name} → {out}")


# ---------------------------------------------------------------------------
# Section 5: Verification plots (HPC only, not committed to repo)
# ---------------------------------------------------------------------------

def plot_dataset_verification_precip() -> None:
    verify_dir = FIGURES_DIR / "verification_plots"
    verify_dir.mkdir(parents=True, exist_ok=True)

    # Combined time series — load only basin mean for each dataset (tiny)
    fig, ax = plt.subplots(figsize=(16, 8))
    for p_name in PRECIP_DATASETS:
        nc = OUTPUTS_DIR / f"precip_anom_{p_name}.nc"
        if not nc.exists():
            continue
        try:
            p_da = xr.open_dataarray(nc, chunks={"time": 120})
            basin_mean = p_da.mean(dim="basin").compute()  # only loads the mean, not full array
            ax.plot(basin_mean.time, basin_mean.values, label=p_name, alpha=0.7)
            del p_da, basin_mean
        except Exception as e:
            print(f"  Could not plot {p_name}: {e}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Precipitation Anomaly (mm/month)")
    ax.set_title("Basin-Averaged Precipitation Anomaly — All Datasets", fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(verify_dir / "precip_anomaly_timeseries_all.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Per-dataset basin variability — load one at a time
    for p_name in PRECIP_DATASETS:
        nc = OUTPUTS_DIR / f"precip_anom_{p_name}.nc"
        if not nc.exists():
            continue
        try:
            p_da = xr.open_dataarray(nc, chunks={"time": 120})
            std_p = p_da.std(dim="time").compute()
            del p_da
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(range(len(std_p.basin)), std_p.values)
            ax.set_xlabel("Basin Index")
            ax.set_ylabel("Std Dev (mm/month)")
            ax.set_title(f"Precipitation Anomaly Variability by Basin — {p_name}", fontweight="bold")
            ax.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(verify_dir / f"precip_anomaly_basin_variability_{p_name}.png", dpi=150, bbox_inches="tight")
            plt.close()
            del std_p
        except Exception as e:
            print(f"  Could not plot variability for {p_name}: {e}")


def plot_dataset_verification_sst() -> None:
    """QC plots for SST anomalies. Saved to HPC figures dir."""
    print("\n" + "=" * 80)
    print("GENERATING SST VERIFICATION PLOTS")
    print("=" * 80)

    verify_dir = FIGURES_DIR / "verification_plots"
    verify_dir.mkdir(parents=True, exist_ok=True)

    # Load SST anomalies from disk
    sst_dict = {}
    for sst_name in SST_PATHS:
        nc = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if not nc.exists():
            print(f"  Skipping {sst_name}: file not found at {nc}")
            continue
        try:
            sst_dict[sst_name] = xr.open_dataarray(nc)
            print(f"  Loaded {sst_name} from {nc}")
        except Exception as e:
            print(f"  Could not load {sst_name}: {e}")

    if not sst_dict:
        print("  No SST anomaly files found — skipping verification plots.")
        return

    # Global mean time series
    fig, ax = plt.subplots(figsize=(16, 8))
    for sst_name, sst_da in sst_dict.items():
        try:
            ax.plot(sst_da.time, sst_da.mean(dim=["lat", "lon"]).values,
                    label=sst_name, alpha=0.7, linewidth=2)
        except Exception as e:
            print(f"  Could not plot {sst_name}: {e}")
    ax.set_xlabel("Time")
    ax.set_ylabel("SST Anomaly (K)")
    ax.set_title("Global Mean SST Anomaly — All Datasets", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out = verify_dir / "sst_anomaly_timeseries_all.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # Spatial std dev maps
    for sst_name, sst_da in sst_dict.items():
        try:
            fig = plt.figure(figsize=(14, 6))
            ax  = plt.axes(projection=ccrs.PlateCarree())
            sst_da.std(dim="time").plot(
                ax=ax, transform=ccrs.PlateCarree(),
                cmap="RdBu_r", vmin=0, vmax=2.5,
                cbar_kwargs={"label": "SST Anomaly Std Dev (K)", "shrink": 0.8},
            )
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":", alpha=0.5)
            ax.set_title(f"SST Anomaly Variability — {sst_name}", fontweight="bold")
            plt.tight_layout()
            out = verify_dir / f"sst_anomaly_map_{sst_name}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved {out}")
        except Exception as e:
            print(f"  Could not plot map for {sst_name}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    precip_raw   = load_precip()
    sst_dict_raw = load_sst()

    sst_anomalies, _ = compute_anomalies(precip_raw, sst_dict_raw)

    save_outputs(sst_anomalies, sst_dict_raw)

    plot_dataset_verification_precip()
    plot_dataset_verification_sst()

    print("\n" + "=" * 80)
    print("DATA PROCESSING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()