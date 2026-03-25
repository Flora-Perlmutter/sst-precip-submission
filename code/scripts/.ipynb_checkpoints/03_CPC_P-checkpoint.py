#!/usr/bin/env python
# coding: utf-8

# ---
# ### Prepare CPC P data for analysis
# #### Noel Siegert, 3/21/2023
# ### Revised by Leah Brown 07/09/2025
# ---

# In[14]:


# this data is already at 0.5 degree resolution, but is daily sum, units are mm/day
# so this script will double check that grid lines up w/ reference, then resample to monthly.
# Mask out ocean.

# New data (added 06/2025) is of the same units and scale as before (daily, 0.25 deg), but longitude is of the form 0.25, 0.75, etc, rather than -179, so on.
##Latitude is also listed backwards relative to normal (should be -89 to +89). Need to regrid


# In[7]:


# import
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from datetime import datetime

import warnings
warnings.filterwarnings(action='ignore') 


# In[8]:


# directories
root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
output_dir = 'nsiegert/projects/aridity/data/P'
os.chdir(root_dir)


# In[9]:


# list cpc p ds files to make it easy to load them all in
files = glob.glob(os.path.join(root_dir, 'Data/Observations/CPC/ppt/*.nc'))
files.sort()
files


# In[ ]:





# In[ ]:


# First, fix the grid


# In[13]:


#Open Noel's refernce grid for regridding
# May need to rename halfdeg_ref_grid_repaired.nc to halfdeg_ref_grid.nc
ds_ref = xr.open_dataset(os.path.join(root_dir, 'nsiegert/projects/aridity/data', 'halfdeg_ref_grid_repaired.nc'))

# Load in land-ocean mask
LO_mask = xr.open_dataset('nsiegert/projects/aridity/data/landseamask/era5_landmask_halfdeg.nc')
LO_mask = LO_mask['landfrac']


# In[11]:


# convert longitude without need for full regridding
def convert_lon(ds):
    ds = ds.copy()
    lon_new = (((ds.lon + 180) % 360) - 180).sortby('lon')
    ds.coords['lon'] = lon_new
    return ds.sortby('lon')


# In[14]:


#Can now open files and finish regridding lat
#Mask out ocean area. Easy for CPC as ocean is represented as 0
#Once regridding is done, convert from daily to monthly

for f in files:
    #Open each of the CPC files listed
    ds = xr.open_dataset(f)

    
    #Apply above function to fix lonigtude
    ds_in_reoriented = convert_lon(ds)
    
    
    # Create the regridder based on each new observation file
    regridder = xe.Regridder(ds_in_reoriented, ds_ref, method='conservative')#, periodic=True, reuse_weights=False)

    # Step 3: Regrid the precip variable
    precip_regridded = regridder(ds_in_reoriented['precip'])

    # Step 4: Create a new dataset with regridded precipitation and reference coords
    ds_out = xr.Dataset(
        {
            'precip': precip_regridded
        },
        coords={
            'time': ds['time'],       # keep original time dimension
            'lat': ds_ref['lat'],
            'lon': ds_ref['lon']
        }
    )

    
    ## Now regrid to monthly
    # save startdate
    st_dt = ds.time.values[0]


    # groupby and sum on DataArray 
    precip_monthly_da = ds_out.groupby('time.month').sum(dim='time')

    # Convert to Dataset
    ds_mo = precip_monthly_da#.to_dataset(name='precip')

    # Rename variable precip -> P
    ds_mo = ds_mo.rename_vars({'precip': 'P'})

    
   
    # gen monthly dates (know each file is 1 yr or 12 months)
    mo_dates = pd.date_range(start=st_dt, periods=12, freq='MS')

    # add that to the dataset
    ds_mo = ds_mo.rename({'month':'time'})
    ds_mo['time'] = mo_dates
    
    # Mask out anywhere not land according to mask
    masked_cpc = ds_mo.where(LO_mask > 0)
    
    
    # Add attributes
    masked_cpc.attrs['units'] = 'mm/month'
    masked_cpc.attrs['desc'] = 'monthly sum precipitation'
    masked_cpc.attrs['script'] = 'nsiegert/projects/aridity/code/dataprep/CPC_P.ipynb'
    masked_cpc.attrs['info'] = 'Daily data, resampled to monthly. 0.5 deg resolution. Masked out ocean. https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html'
    now = datetime.now() # get datetime
    masked_cpc.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

    # save (P.CPC.<year>.nc)
    masked_cpc.to_netcdf(os.path.join(output_dir, 'P.CPC.{}'.format(f[-7:])))
    
    print('saved P.CPC.{}'.format(f[-7:]))

    


# In[ ]:




