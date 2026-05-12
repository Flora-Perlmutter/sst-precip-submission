#!/usr/bin/env python
# coding: utf-8
"""
Load and process AMIP CMIP6 precipitation and surface temperature data.

Author: Flora Perlmutter

Description
-----------
This script loads and preprocesses CMIP6 AMIP ensemble precipitation (pr)
and surface air temperature (tas, used as SST proxy) for the
SST-precipitation sensitivity analysis. It:
  1. Loads and filters AMIP pr files by model, coverage, and resolution
  2. Loads and filters AMIP tas files by model, coverage, and resolution
  3. Regrids all tas models to the coarsest common grid
  4. Determines the common overlapping time period
  5. Computes monthly anomalies for precipitation (basin-scaled) and tas
  6. Filters to models with both pr and tas data
  7. Saves all processed data as NetCDF + a JSON manifest to the outputs directory
  8. Generates verification plots (saved to HPC figures directory)

Excluded models (noted for reproducibility):
  - GISS-E2-1-G_r1i1p1f1: missing dates in 1980-2014
  - NorESM2-LM_r1i1p1f1:  missing dates in 1980-2014
  - Any model with resolution > 2.5 degrees
  
"""


import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import itertools
import pickle
import glob
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import sys
import regionmask
import cartopy.io.shapereader as shpreader
import json
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

##-------------------------------------------------------------------------------##
# SECTION 1: SETUP AND CONFIGURATION
##-------------------------------------------------------------------------------##

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from data_processing_functions import (
    convert_to_mm_month,
    compute_monthly_anomaly,
    determine_common_time_period,
    mask_sea_ice,
)
from amip_functions import (
    preprocess_pr,
    preprocess_tas,
    group_files_by_model,
    make_periodic,
)
from regression_functions import waterbasin

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
AMIP_PR_DIR  = CMIG_DATA / "Data/ClimateModels/CMIP6/amip/pr_Amon"
AMIP_TAS_DIR = CMIG_DATA / "Data/ClimateModels/CMIP6/amip/tas_Amon"
OUTPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
FIGURES_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Figures"


# --- Define the datasets for the ensemble ---

pr_files = glob.glob('/dartfs-hpc/rc/lab/C/CMIG/Data/ClimateModels/CMIP6/amip/pr_Amon/pr_Amon_*.nc')
print(f"Found {len(pr_files)} precipitation files")

# Problematic precip datasets to exclude
# Missing some dates between 1980-2014
exclude_pr_models = {
    "GISS-E2-1-G_r1i1p1f1",
    "NorESM2-LM_r1i1p1f1",
}

# Group files by model and ensemble member
pr_grouped = group_files_by_model(pr_files)
print(f"Grouped into {len(pr_grouped)} unique model/ensemble combinations")

# Keep only r1i1p1f1 and exclude known bad precip datasets
pr_grouped = {
    k: v
    for k, v in pr_grouped.items()
    if ("r1i1p1f1" in k) and (k not in exclude_pr_models)
}


tas_files = glob.glob('/dartfs-hpc/rc/lab/C/CMIG/Data/ClimateModels/CMIP6/amip/tas_Amon/tas_Amon_*.nc')
print(f"Found {len(tas_files)} temperature files")

# Group files by model and ensemble member
tas_grouped = group_files_by_model(tas_files)
print(f"Grouped into {len(tas_grouped)} unique model/ensemble combinations")

tas_grouped = {
    k: v
    for k, v in tas_grouped.items()
    if "r1i1p1f1" in k
}

##-------------------------------------------------------------------------------##
# SECTION 2: LOAD AND PROCESS PRECIPITATION DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("LOADING PRECIPITATION DATASETS")
print("="*80)

# Load each model/ensemble separately with mfdataset
amip_pr_dict = {}
for model_id, file_list in pr_grouped.items():
    print(f"\nProcessing {model_id} ({len(file_list)} file(s))...")
    
    try:
        # Open multiple files with preprocessing
        pr_ds = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            decode_times=True,
            use_cftime=True,
            preprocess=preprocess_pr
        )
        
        # Extract the DataArray
        pr_da = pr_ds['pr']
        
        # Sort coordinates
        pr_da = pr_da.sortby(['time', 'lat', 'lon'])
        
        # Ensure longitude is 0-360
        if pr_da.lon.min() < 0:
            pr_da = pr_da.assign_coords(lon=(pr_da.lon % 360))
            pr_da = pr_da.sortby('lon')
        
        print(f"    Loaded: {pr_da.time.size} timesteps, {pr_da.lat.size} lats, {pr_da.lon.size} lons")
        print(f"    Time range: {pr_da.time.values[0]} to {pr_da.time.values[-1]}")
        
        # ===== ADD CHECK HERE =====
        # Check if data covers 1980-2014 period
        data_constrained = pr_da.sel(time=slice('1980-01-01', '2014-12-31'))
        if len(data_constrained.time) == 0:
            print(f"     Skipping {model_id} - no data in 1980-2014 period")
            continue
        
        # Check if coverage is complete enough (e.g., at least 400 months = ~33 years)
        if len(data_constrained.time) < 400:
            print(f"     Skipping {model_id} - insufficient data in 1980-2014 ({len(data_constrained.time)} months)")
            continue
        # ===== END CHECK =====
        
        amip_pr_dict[model_id] = pr_da
        
    except Exception as e:
        print(f"     Failed to load {model_id}: {e}")

# ---------------------------------------------------------
# Compute resolution for each precipitation model
# ---------------------------------------------------------
pr_model_resolutions = {}

for model_id, pr_da in amip_pr_dict.items():
    dlat = abs(float(pr_da.lat[1] - pr_da.lat[0]))
    dlon = abs(float(pr_da.lon[1] - pr_da.lon[0]))

    pr_model_resolutions[model_id] = (dlat, dlon)
    print(f"{model_id}: (lat: {dlat:.2f}°, lon: {dlon:.2f}°)")

# ---------------------------------------------------------
# Filter precipitation models by maximum allowed resolution
# ---------------------------------------------------------
max_resolution = 2.5
filtered_pr_dict = {}
dropped_pr_models = []

for model_id, pr_da in amip_pr_dict.items():
    dlat, dlon = pr_model_resolutions[model_id]

    if (dlat <= max_resolution) and (dlon <= max_resolution):
        filtered_pr_dict[model_id] = pr_da
    else:
        dropped_pr_models.append(model_id)

print(f"\n   Dropped {len(dropped_pr_models)} precipitation model(s) with resolution > {max_resolution}°:")
for model in dropped_pr_models:
    dlat, dlon = pr_model_resolutions[model]
    print(f"  - {model}: (lat: {dlat:.2f}°, lon: {dlon:.2f}°)")

print(f"\n  Kept {len(filtered_pr_dict)} precipitation model(s) with resolution ≤ {max_resolution}°")

precip_raw = filtered_pr_dict
        
##-------------------------------------------------------------------------------##
# SECTION 3: LOAD AND PROCESS SST DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("LOADING & PROCESSING AMIP TAS DATASETS")
print("="*80)

amip_tas_dict = {}

for model_id, file_list in tas_grouped.items():
    print(f"\nProcessing {model_id} ({len(file_list)} file(s))...")

    try:
        # Load dataset
        tas_ds = xr.open_mfdataset(
            file_list,
            combine='by_coords',
            decode_times=True,
            use_cftime=True,
            preprocess=preprocess_tas
        )

        tas_da = tas_ds['tas']

        # Sort coordinates
        tas_da = tas_da.sortby(['time', 'lat', 'lon'])

        # Ensure longitude is 0–360
        if tas_da.lon.min() < 0:
            tas_da = tas_da.assign_coords(lon=(tas_da.lon % 360))
            tas_da = tas_da.sortby('lon')

        print(f"    Loaded {tas_da.time.size} timesteps")
        print(f"    Time range: {tas_da.time.values[0]} → {tas_da.time.values[-1]}")

        # ===== ADD CHECK HERE =====
        # Check if data covers 1980-2014 period
        data_constrained = tas_da.sel(time=slice('1980-01-01', '2014-12-31'))
        if len(data_constrained.time) == 0:
            print(f"     Skipping {model_id} - no data in 1980-2014 period")
            continue
        
        # Check if coverage is complete enough
        if len(data_constrained.time) < 400:
            print(f"     Skipping {model_id} - insufficient data in 1980-2014 ({len(data_constrained.time)} months)")
            continue
        # ===== END CHECK =====

        # ---------------------------------------------------------
        # Check & convert units (C → K)
        # ---------------------------------------------------------
        units = tas_da.attrs.get('units', '').lower()
        needs_conversion = units in ['c', 'celsius', 'degc', 'degrees_celsius', 'degree_celsius']

        if needs_conversion:
            print("  Converting units from Celsius → Kelvin")
            tas_da = tas_da + 273.15
            tas_da.attrs['units'] = 'K'
        elif units not in ['k', 'kelvin']:
            print(f"  Warning: units='{units}', assuming Kelvin")

        # ---------------------------------------------------------
        # Create land/ocean mask (ERSST-style)
        # ---------------------------------------------------------
        print("  Creating land/ocean mask...")

        land_shp = shpreader.natural_earth(
            resolution='10m',
            category='physical',
            name='land'
        )
        land_gdf = gpd.read_file(land_shp)

        land_mask = regionmask.mask_geopandas(
            land_gdf,
            tas_da.lon,
            tas_da.lat
        )

        ocean_mask = land_mask.isnull()

        # ---------------------------------------------------------
        # Mask land + apply sea-ice mask (ERSST-style)
        # ---------------------------------------------------------
        print("  Applying land + sea-ice mask...")

        tas_ocean = tas_da.where(ocean_mask)
        tas_wo_ice = mask_sea_ice(tas_ocean)

        # Save processed version
        amip_tas_dict[model_id] = tas_wo_ice

        print(f"    Finished processing: {model_id}")

    except Exception as e:
        print(f"     Failed to process {model_id}: {e}")

print("\n" + "="*80)
print("FILTERING MODELS BY RESOLUTION")
print("="*80)

# ---------------------------------------------------------
# Compute resolution for each model and store as tuple
# ---------------------------------------------------------
model_resolutions = {}

for model_id, tas_da in amip_tas_dict.items():
    dlat = abs(float(tas_da.lat[1] - tas_da.lat[0]))
    dlon = abs(float(tas_da.lon[1] - tas_da.lon[0]))

    model_resolutions[model_id] = (dlat, dlon)

    print(f"{model_id}: (lat: {dlat:.2f}°, lon: {dlon:.2f}°)")


# ---------------------------------------------------------
# Filter by maximum allowed resolution
# ---------------------------------------------------------
max_resolution = 2.5
filtered_dict = {}
dropped_models = []

for model_id, tas_da in amip_tas_dict.items():
    dlat, dlon = model_resolutions[model_id]
    
    if (dlat <= max_resolution) and (dlon <= max_resolution):
        filtered_dict[model_id] = tas_da
    else:
        dropped_models.append(model_id)

print(f"\n   Dropped {len(dropped_models)} model(s) with resolution > {max_resolution}°:")
for model in dropped_models:
    dlat, dlon = model_resolutions[model]
    print(f"  - {model}: (lat: {dlat:.2f}°, lon: {dlon:.2f}°)")

print(f"\n  Kept {len(filtered_dict)} model(s) with resolution ≤ {max_resolution}°")


# ---------------------------------------------------------
# Determine the coarsest (lowest-resolution) model
# ---------------------------------------------------------
if filtered_dict:
    # Only consider models that survived filtering
    lowest_res_model = max(
        filtered_dict.keys(),
        key=lambda m: (model_resolutions[m][0], model_resolutions[m][1])
    )

    print("\n" + "="*80)
    print("REGRIDDING TO LOWEST RESOLUTION (SAFE LON INTERP)")
    print("="*80)

    target = make_periodic(filtered_dict[lowest_res_model])
    target_lats = target.lat
    target_lons = target.lon
    
    # Remove the duplicate 360° from target if it exists
    if target_lons.max() >= 360.0:
        target_lons = target_lons.where(target_lons < 360.0, drop=True)
    
    regridded_dict = {}
    
    for model_id, da in filtered_dict.items():
        print(f"Regridding {model_id}...")
    
        da = make_periodic(da)
    
        regridded = da.interp(
            lat=target_lats,
            lon=target_lons,
            method="linear"
        )
    
        regridded_dict[model_id] = regridded
        
        # Verify no gaps
        lon_diff = regridded.lon.diff('lon')
        max_gap = float(lon_diff.max())
        expected_gap = float(lon_diff.diff('lon').median())
        if max_gap > expected_gap * 1.5:
            print(f"    Warning: Large longitude gap detected ({max_gap:.3f}° vs expected {expected_gap:.3f}°)")
        else:
            print(f"    Longitude spacing OK (max: {max_gap:.3f}°, expected: {expected_gap:.3f}°)")        
else:
    print("\n   No models remaining after filtering!")
    amip_tas_dict = {}


print("\n  All AMIP SST-style steps applied to AMIP tas datasets.")
print("="*80)

# Store raw SST data (before anomalies)
sst_dict_raw = regridded_dict


##-------------------------------------------------------------------------------##
# SECTION 4: DETERMINE COMMON TIME PERIOD AND COMPUTE ANOMALIES
##-------------------------------------------------------------------------------##

common_start, common_end = determine_common_time_period(precip_raw, sst_dict_raw)
common_period = slice(common_start.strftime('%Y-%m-%d'), common_end.strftime('%Y-%m-%d'))

# Compute precipitation anomalies with respect to common time period
print("\n" + "="*80)
print("COMPUTING PRECIPITATION ANOMALIES WITH BASIN SCALING")
print("="*80)

precip_anomalies = {}
for p_name, p_da in precip_raw.items():
    print(f"  Processing {p_name}...")
    try:

        # Apply basin scaling
        print(f"    Applying basin scaling...")
        p_basin = waterbasin(p_da)
        
        # Calculate anomalies
        print(f"    Computing anomalies...")
        p_anom = compute_monthly_anomaly(p_basin, climatology_period=common_period)
        p_anom = p_anom.load()
        
        precip_anomalies[p_name] = p_anom
        print(f"      Anomalies computed for {p_anom.time.size} timesteps")
    except Exception as e:
        print(f"       Failed to process {p_name}: {e}")

# Compute SST anomalies with respect to common time period
print("\n" + "="*80)
print("COMPUTING SST ANOMALIES")
print("="*80)

sst_anomalies = {}
for sst_name, sst_da in sst_dict_raw.items():
    print(f"  Processing {sst_name}...")
    try:
        
        # Compute monthly anomalies
        sst_anom = compute_monthly_anomaly(sst_da, climatology_period=common_period)
        sst_anom = sst_anom.assign_coords(dataset=sst_name)
        
        sst_anomalies[sst_name] = sst_anom
        print(f"      Anomalies computed for {sst_anom.time.size} timesteps")
    except Exception as e:
        print(f"       Failed to compute anomalies for {sst_name}: {e}")

##-------------------------------------------------------------------------------##
# SECTION 5: FILTER FOR MODELS WITH BOTH PRECIP AND SST DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("FILTERING FOR MODELS WITH BOTH PRECIPITATION AND SST DATA")
print("="*80)

# Get the set of model names from both dictionaries
precip_models = set(precip_anomalies.keys())
sst_models = set(sst_anomalies.keys())

# Find models that exist in both
common_models = precip_models.intersection(sst_models)

print(f"\nPrecipitation models: {len(precip_models)}")
print(f"SST models: {len(sst_models)}")
print(f"Models with both: {len(common_models)}")

# Models only in precip
precip_only = precip_models - sst_models
if precip_only:
    print(f"\n   Dropping {len(precip_only)} model(s) with precipitation but no SST:")
    for model in sorted(precip_only):
        print(f"  - {model}")

# Models only in SST
sst_only = sst_models - precip_models
if sst_only:
    print(f"\n   Dropping {len(sst_only)} model(s) with SST but no precipitation:")
    for model in sorted(sst_only):
        print(f"  - {model}")

# Filter both dictionaries to keep only common models
precip_anomalies = {k: v for k, v in precip_anomalies.items() if k in common_models}
sst_anomalies = {k: v for k, v in sst_anomalies.items() if k in common_models}
sst_dict_raw = {k: v for k, v in sst_dict_raw.items() if k in common_models}

print(f"\n  Kept {len(common_models)} model(s) with both precipitation and SST data")
print("="*80)

##-------------------------------------------------------------------------------##
# SECTION 6: SAVE PROCESSED DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)
# Save precipitation anomalies as NetCDF instead of pickle
print("  Saving precipitation anomalies...")
saved_precip = []
for p_name, p_anom in precip_anomalies.items():
    nc_path = OUTPUTS_DIR / f"amip_precip_anom_{p_name}.nc"
    p_anom.to_netcdf(nc_path)
    saved_precip.append(p_name)
    print(f"      Saved {p_name} to: {nc_path}")

# Save SST anomalies as NetCDF
print("  Saving SST anomalies...")
saved_sst = []
for sst_name, sst_anom in sst_anomalies.items():
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    sst_anom.to_netcdf(nc_path)
    saved_sst.append(sst_name)
    print(f"      Saved {sst_name} to: {nc_path}")

# Save raw SST data as NetCDF
print("  Saving raw SST data...")
saved_sst_raw = []
for sst_name, sst_raw in sst_dict_raw.items():
    nc_path = OUTPUTS_DIR / f"sst_raw_{sst_name}.nc"
    sst_raw.to_netcdf(nc_path)
    saved_sst_raw.append(sst_name)
    print(f"      Saved {sst_name} to: {nc_path}")

# Save target grid as NetCDF
print("  Saving target grid...")
xr.Dataset({
    "lats": (["lat"], target_lats.values),
    "lons": (["lon"], target_lons.values),
}).to_netcdf(OUTPUTS_DIR / "amip_target_grid.nc")
print(f"      Saved target grid to: {nc_path}")

# Save manifest of dataset names as JSON
names = {
    'precip_datasets': saved_precip,
    'sst_datasets': saved_sst,
    'sst_raw_datasets': saved_sst_raw
}
names_path = DATA_DIR / "amip_dataset_names.json"
with open(names_path, "w") as f:
    json.dump(names, f, indent=2)
print(f"    Saved dataset names to: {names_path}")

print("\n" + "="*80)
print("DATA PROCESSING COMPLETE")
print("="*80 + "\n")


##-------------------------------------------------------------------------------##
# SECTION 7: VERIFICATION PLOTS
##-------------------------------------------------------------------------------##

def plot_dataset_verification_precip(precip_dict, output_dir):
    """
    Create verification plots for all preprocessed precipitation datasets.
    Note: These datasets have basin dimension instead of lat/lon after waterbasin().
    """
    print("\n" + "="*80)
    print("GENERATING PRECIPITATION VERIFICATION PLOTS")
    print("="*80)
    
    # Create subdirectory for verification plots
    verify_dir = output_dir / "verification_plots"
    os.makedirs(verify_dir, exist_ok=True)
    
    # Plot 1: Precipitation Time Series (INDIVIDUAL plots for each dataset)
    print("\n--- Plotting Precipitation Anomaly Time Series (Individual) ---")
    for p_name, p_da in precip_dict.items():
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Average across basins for global mean
            basin_mean = p_da.mean(dim='basin')
            ax.plot(basin_mean.time, basin_mean.values, linewidth=2, color='steelblue')
            
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Precipitation Anomaly (mm/month)', fontsize=12)
            ax.set_title(f'Basin-Averaged Precipitation Anomaly - {p_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # Add statistics text box
            mean_val = float(basin_mean.mean())
            std_val = float(basin_mean.std())
            textstr = f'Mean: {mean_val:.3f} mm/month\nStd: {std_val:.3f} mm/month'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(verify_dir / f'amip_precip_anomaly_timeseries_{p_name}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: precip_anomaly_timeseries_{p_name}.png")
        except Exception as e:
            print(f"     Could not plot {p_name}: {e}")
    
    # Plot 2: Basin-wise variability for each precipitation dataset
    print("\n--- Plotting Precipitation Anomaly Basin Variability ---")
    for p_name, p_da in precip_dict.items():
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate std dev for each basin
            std_precip = p_da.std(dim='time')
            
            # Plot as bar chart
            basins = std_precip.basin.values
            ax.bar(range(len(basins)), std_precip.values, color='steelblue', alpha=0.7)
            ax.set_xlabel('Basin Index', fontsize=12)
            ax.set_ylabel('Std Dev (mm/month)', fontsize=12)
            ax.set_title(f'Precipitation Anomaly Variability by Basin - {p_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(verify_dir / f'amip_precip_anomaly_basin_variability_{p_name}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: precip_anomaly_basin_variability_{p_name}.png")
        except Exception as e:
            print(f"     Could not plot variability for {p_name}: {e}")
        
    print("\n" + "="*80)
    print(f"PRECIPITATION VERIFICATION PLOTS COMPLETE - Saved to: {verify_dir}")
    print("="*80 + "\n")

def plot_dataset_verification_sst(sst_dict, output_dir):
    """
    Create verification plots for all preprocessed SST datasets.
    """
    print("\n" + "="*80)
    print("GENERATING SST VERIFICATION PLOTS")
    print("="*80)
    
    # Create subdirectory for verification plots
    verify_dir = output_dir / 'verification_plots'
    os.makedirs(verify_dir, exist_ok=True)
    
    # Plot 1: SST Time Series (all datasets)
    print("\n--- Plotting SST Anomaly Time Series ---")
    fig, ax = plt.subplots(figsize=(16, 8))
    for sst_name, sst_da in sst_dict.items():
        try:
            global_mean = sst_da.mean(dim=['lat', 'lon'])
            ax.plot(global_mean.time, global_mean.values, label=sst_name, alpha=0.7, linewidth=2)
        except Exception as e:
            print(f"     Could not plot {sst_name}: {e}")
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('SST Anomaly (K)', fontsize=12)
    ax.set_title('Global Mean SST Anomaly - All Datasets', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(verify_dir / 'amip_sst_anomaly_timeseries_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: sst_anomaly_timeseries_all.png")
        
    # Plot 2: Spatial maps for each SST dataset
    print("\n--- Plotting SST Anomaly Spatial Maps ---")
    for sst_name, sst_da in sst_dict.items():
        try:
            fig = plt.figure(figsize=(14, 6))
            ax = plt.axes(projection=ccrs.PlateCarree())
            
            std_sst = sst_da.std(dim='time')
            im = std_sst.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap='RdBu_r',
                vmin=0, vmax=2.5,
                cbar_kwargs={'label': 'SST Anomaly Std Dev (K)', 'shrink': 0.8}
            )
            
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
            ax.set_title(f'SST Anomaly Variability - {sst_name}', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(verify_dir / f'amip_sst_anomaly_map_{sst_name}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: sst_anomaly_map_{sst_name}.png")
        except Exception as e:
            print(f"     Could not plot map for {sst_name}: {e}")
       
    print("\n" + "="*80)
    print(f"SST VERIFICATION PLOTS COMPLETE - Saved to: {verify_dir}")
    print("="*80 + "\n")


# Generate verification plots
plot_dataset_verification_precip(precip_anomalies, FIGURES_DIR)
plot_dataset_verification_sst(sst_anomalies, FIGURES_DIR)