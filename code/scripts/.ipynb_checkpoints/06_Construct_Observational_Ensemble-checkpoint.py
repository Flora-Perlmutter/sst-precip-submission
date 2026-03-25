#!/usr/bin/env python
# coding: utf-8


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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

##-------------------------------------------------------------------------------##
# SECTION 1: SETUP AND CONFIGURATION
##-------------------------------------------------------------------------------##

# Define directories
root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')
data_dir = os.path.join(root_dir, 'Data/Observations')
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
figures_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Figures')

# Ensure output directory exists
os.makedirs(outputs_dir, exist_ok=True)

# Add functions dir to path and import helpers
sys.path.append(functions_dir)
from regression_functions import (
    waterbasin, detrend_dim, convert_to_mm_month, grid_area,
    fdr_correction, regression_with_conditioning_var_se
)

from data_processing_functions import (
    fix_lons, fixdates, convert_to_mm_month, load_and_preprocess_data, mask_sea_ice,
    compute_monthly_anomaly, determine_common_time_period
)


os.chdir(root_dir)

# --- Define the datasets for the ensemble ---
#excluding MERRA2_P because there are missing dates, excluding CHIRPS because of 50S-50N spatial extent, excluding ERA5-Land because of known biases, excluding MSWEP because it is not compatible with the river basin function due to lat/lon not being evenly spaced

PRECIP_DATASETS = {
    'GPCP': {'path': os.path.join(data_dir, 'GPCP/precip.mon.mean.nc'), 'var': 'precip', 'name': 'GPCP'},
    'CRU': {'path': 'nsiegert/projects/aridity/data/P/P.CRU.1901.2024.nc', 'var': 'P', 'name': 'CRU'},
    'GPCC': {'path': 'nsiegert/projects/aridity/data/P/P.GPCC.1891.2020.nc', 'var': 'P', 'name': 'GPCC'},
    'CPC': {'path': 'nsiegert/projects/aridity/data/P/P.CPC.*.nc', 'var': 'P', 'name': 'CPC'},
    'UDel': {'path': 'Data/Observations/UDel/monthly/precip.mon.total.v501.nc', 'var': 'precip', 'name': 'UDel'},
    'PREC': {'path': 'Data/Observations/PREC_L/precip.mon.mean.0.5x0.5.nc', 'var': 'precip', 'name': 'PREC'},
    'TerraClimate': {'path': 'nsiegert/projects/aridity/data/P/P.TerraClimate.*.nc', 'var': 'P', 'name': 'TerraClimate'},
    'REGEN': {'path': 'nsiegert/projects/aridity/data/P/P.REGEN.1950-2016.nc', 'var': 'P', 'name': 'REGEN'},
}

##-------------------------------------------------------------------------------##
# SECTION 2: LOAD AND PROCESS PRECIPITATION DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("LOADING PRECIPITATION DATASETS")
print("="*80)

precip_raw = {}
for p_name, p_meta in PRECIP_DATASETS.items():
    print(f"\nLoading and preprocessing precipitation dataset: {p_name}")
    try:
        p_da = load_and_preprocess_data(p_meta)
        precip_raw[p_name] = p_da
        print(f"    {p_name}: {p_da.time.size} timesteps, "
              f"{p_da.lat.size} lats, {p_da.lon.size} lons")
    except Exception as e:
        print(f"    FAILED to load {p_name}: {e}")


##-------------------------------------------------------------------------------##
# SECTION 3: LOAD AND PROCESS SST DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("LOADING SST DATASETS")
print("="*80)

ersst_path = os.path.join(data_dir, 'ERSST/sst.mnmean.nc')
cobe_path  = '/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/CAM4_Greens_Project/Data/Raw/sst.mon.mean.nc'

ersst = xr.open_dataset(ersst_path)
cobe  = xr.open_dataset(cobe_path)

# Select variable names
sst_ersst = ersst['sst']
sst_cobe  = cobe['sst']

# Convert units if in degC to K
units = sst_cobe.attrs.get('units', '').lower()
needs_conversion = units in ['c', 'celsius', 'degc', 'degree_celsius', 'degrees_celsius', 'deg_c']

if needs_conversion:
    print(f"  Converting COBE-SST2 from Celsius to Kelvin")
    sst_cobe = sst_cobe + 273.15
    sst_cobe.attrs['units'] = 'K'
elif units not in ['k', 'kelvin']:
    print(f"  Warning: Units for COBE-SST2 are '{units}', assuming Kelvin")
    
# Convert units if in degC to K
units = sst_ersst.attrs.get('units', '').lower()
needs_conversion = units in ['c', 'celsius', 'degc', 'degree_celsius', 'degrees_celsius', 'deg_c']

if needs_conversion:
    print(f"  Converting ERSSTv5 from Celsius to Kelvin")
    sst_ersst = sst_ersst + 273.15
    sst_ersst.attrs['units'] = 'K'
elif units not in ['k', 'kelvin']:
    print(f"  Warning: Units for ERSSTv5 are '{units}', assuming Kelvin")

# Select target lats/lons
target_lats = sst_ersst.lat
target_lons = sst_ersst.lon

# Shift COBE by -0.5 degrees to align the grids
sst_cobe = sst_cobe.assign_coords(lon=(sst_cobe.lon - 0.5) % 360)

# Sort longitude to ensure proper ordering
sst_cobe = sst_cobe.sortby('lon')
    
# Regrid to target grid
print("  Regridding COBE-SST2 to ERSSTv5 grid...")
sst_cobe_regridded = sst_cobe.interp(lat=target_lats, lon=target_lons, method='linear')

# Load high-resolution land shapefile from Natural Earth
print("  Creating ocean mask...")
land_shp = shpreader.natural_earth(resolution='10m',
                                   category='physical',
                                   name='land')
land_gdf = gpd.read_file(land_shp)

# Use regionmask to mask ocean or land
land_mask = regionmask.mask_geopandas(
    land_gdf, 
    sst_ersst.lon, 
    sst_ersst.lat
)

# Create a mask for ocean
ocean_mask = land_mask.isnull()

# Apply the mask to the data
print("  Masking land and sea ice regions...")
ersst_masked = sst_ersst.where(ocean_mask)
ersst_wo_ice = mask_sea_ice(ersst_masked)

sst_cobe_masked = sst_cobe_regridded.where(ocean_mask)
cobe_sst_wo_ice = mask_sea_ice(sst_cobe_masked)

# Store raw SST data (before anomalies)
sst_dict_raw = {
    "ERSSTv5": ersst_wo_ice,
    "COBE-SST2": cobe_sst_wo_ice
}


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
        print(f"      Failed to process {p_name}: {e}")

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
        print(f"      Failed to compute anomalies for {sst_name}: {e}")


##-------------------------------------------------------------------------------##
# SECTION 5: SAVE PROCESSED DATA
##-------------------------------------------------------------------------------##

print("\n" + "="*80)
print("SAVING PROCESSED DATA")
print("="*80)

# Save precipitation anomalies as NetCDF instead of pickle
print("  Saving precipitation anomalies...")
for p_name, p_anom in precip_anomalies.items():
    nc_path = os.path.join(outputs_dir, f"precip_anom_{p_name}.nc")
    p_anom.to_netcdf(nc_path)
    print(f"      Saved {p_name} to: {nc_path}")

# Save SST anomalies as NetCDF
print("  Saving SST anomalies...")
for sst_name, sst_anom in sst_anomalies.items():
    nc_path = os.path.join(outputs_dir, f"sst_anom_{sst_name}.nc")
    sst_anom.to_netcdf(nc_path)
    print(f"      Saved {sst_name} to: {nc_path}")

# Save raw SST data as NetCDF
print("  Saving raw SST data...")
for sst_name, sst_raw in sst_dict_raw.items():
    nc_path = os.path.join(outputs_dir, f"sst_raw_{sst_name}.nc")
    sst_raw.to_netcdf(nc_path)
    print(f"      Saved {sst_name} to: {nc_path}")

print("\n" + "="*80)
print("DATA PROCESSING COMPLETE")
print("="*80 + "\n")


##-------------------------------------------------------------------------------##
# SECTION 6: VERIFICATION PLOTS
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
    verify_dir = os.path.join(output_dir, 'verification_plots')
    os.makedirs(verify_dir, exist_ok=True)
    
    # Plot 1: Precipitation Time Series (all datasets)
    print("\n--- Plotting Precipitation Anomaly Time Series (Basin-averaged) ---")
    fig, ax = plt.subplots(figsize=(16, 8))
    for p_name, p_da in precip_dict.items():
        try:
            # Average across basins for global mean
            basin_mean = p_da.mean(dim='basin')
            ax.plot(basin_mean.time, basin_mean.values, label=p_name, alpha=0.7)
        except Exception as e:
            print(f"    Could not plot {p_name}: {e}")
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Precipitation Anomaly (mm/month)', fontsize=12)
    ax.set_title('Basin-Averaged Precipitation Anomaly - All Datasets', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(verify_dir, 'precip_anomaly_timeseries_all.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: precip_anomaly_timeseries_all.png")
    
    # Plot 2: Basin-wise variability for each precipitation dataset
    print("\n--- Plotting Precipitation Anomaly Basin Variability ---")
    for p_name, p_da in precip_dict.items():
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Calculate std dev for each basin
            std_precip = p_da.std(dim='time')
            
            # Plot as bar chart
            basins = std_precip.basin.values
            ax.bar(range(len(basins)), std_precip.values)
            ax.set_xlabel('Basin Index', fontsize=12)
            ax.set_ylabel('Std Dev (mm/month)', fontsize=12)
            ax.set_title(f'Precipitation Anomaly Variability by Basin - {p_name}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(verify_dir, f'precip_anomaly_basin_variability_{p_name}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: precip_anomaly_basin_variability_{p_name}.png")
        except Exception as e:
            print(f"    Could not plot variability for {p_name}: {e}")
        
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
    verify_dir = os.path.join(output_dir, 'verification_plots')
    os.makedirs(verify_dir, exist_ok=True)
    
    # Plot 1: SST Time Series (all datasets)
    print("\n--- Plotting SST Anomaly Time Series ---")
    fig, ax = plt.subplots(figsize=(16, 8))
    for sst_name, sst_da in sst_dict.items():
        try:
            global_mean = sst_da.mean(dim=['lat', 'lon'])
            ax.plot(global_mean.time, global_mean.values, label=sst_name, alpha=0.7, linewidth=2)
        except Exception as e:
            print(f"    Could not plot {sst_name}: {e}")
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('SST Anomaly (K)', fontsize=12)
    ax.set_title('Global Mean SST Anomaly - All Datasets', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(verify_dir, 'sst_anomaly_timeseries_all.png'), dpi=150, bbox_inches='tight')
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
            plt.savefig(os.path.join(verify_dir, f'sst_anomaly_map_{sst_name}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            print(f"    Saved: sst_anomaly_map_{sst_name}.png")
        except Exception as e:
            print(f"    Could not plot map for {sst_name}: {e}")
       
    print("\n" + "="*80)
    print(f"SST VERIFICATION PLOTS COMPLETE - Saved to: {verify_dir}")
    print("="*80 + "\n")


# Generate verification plots
plot_dataset_verification_precip(precip_anomalies, figures_dir)
plot_dataset_verification_sst(sst_anomalies, figures_dir)