#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
import xesmf
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir,'fperlmutter/Observational_Regressions_Project/Scripts')

outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
os.chdir(functions_dir)
warnings.filterwarnings("ignore")


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
DATA_OBS_DIR = os.path.join(root_dir,"Data/Observations")

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

SST_PATHS = {
    "ERSSTv6":   os.path.join(DATA_OBS_DIR , 'ERSST/ERSST6/2026.ersst.v6'),
    "COBE-SST3": os.path.join(DATA_OBS_DIR , 'COBE-SST/COBE-SST3'),
}

def fix_lons(da: xr.DataArray) -> xr.DataArray:
    """Convert longitude from 0-360 to -180/180 if needed, then sort."""
    lon = da["lon"]
    if lon.min() >= 0 and lon.max() > 180:
        da = da.assign_coords(lon=((lon.values + 180) % 360 - 180))
        da = da.sortby("lon")
    return da


def fixdates(da: xr.DataArray) -> xr.DataArray:
    """Standardize monthly timestamps to the first day of each month."""
    if not hasattr(da, "time"):
        return da
    st_y = int(pd.to_datetime(da.time.values[0]).year)
    st_m = int(pd.to_datetime(da.time.values[0]).month)
    e_y  = int(pd.to_datetime(da.time.values[-1]).year)
    e_m  = int(pd.to_datetime(da.time.values[-1]).month)
    newdates = pd.date_range(f"{st_y}-{st_m}-1", f"{e_y}-{e_m}-1", freq="MS")
    return da.assign_coords(time=newdates)

def _convert_sst_to_kelvin(da: xr.DataArray, name: str) -> xr.DataArray:
    """Convert SST from Celsius to Kelvin if needed, with informative print."""
    units = da.attrs.get("units", "").lower()
    celsius_aliases = {"c", "celsius", "degc", "degree_celsius", "degrees_celsius", "deg_c", "degree celsius", "degree_c"}
    if units in celsius_aliases:
        print(f"  Converting {name} from Celsius to Kelvin")
        da = da + 273.15
        da.attrs["units"] = "K"
    elif units not in {"k", "kelvin"}:
        print(f"  Warning: units for {name} are '{units}', assuming Kelvin")
    return da




# In[19]:


# ---------------------------------------------------------------------------
# Section 2: Load SST
# ---------------------------------------------------------------------------

print("\n" + "=" * 80)
print("LOADING SST DATASETS")
print("=" * 80)

ersst = xr.open_mfdataset(
    os.path.join(SST_PATHS["ERSSTv6"], "ersst.v6.*.nc"),
    combine="by_coords"
)
cobe = xr.open_mfdataset(
    os.path.join(SST_PATHS["COBE-SST3"], "cobe-sst3.monthly.*.nc"),
    combine="by_coords"
)

sst_ersst = _convert_sst_to_kelvin(ersst["sst"], "ERSSTv6")
sst_cobe  = _convert_sst_to_kelvin(cobe["sst"],  "COBE-SST3")

rename_map = {k: v for k, v in [("latitude", "lat"), ("longitude", "lon")] if k in sst_cobe.dims}
if rename_map:
    sst_cobe = sst_cobe.rename(rename_map)
    
# Fix ERSSTv6 dates
sst_ersst = fixdates(sst_ersst)

regridder = xesmf.Regridder(
    sst_cobe,
    sst_ersst,
    method="bilinear",       
    periodic=True            # for longitudinal periodicity
)

sst_cobe_regridded = regridder(sst_cobe)

# Save to NetCDF
sst_cobe_regridded.to_netcdf(os.path.join(outputs_dir, "cobe-sst3_regridded.nc"))
sst_ersst.to_netcdf(os.path.join(outputs_dir, "ersstv6_preprocessed.nc"))


# In[10]:





# In[ ]:




