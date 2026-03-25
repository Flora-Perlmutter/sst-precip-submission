#!/usr/bin/env python
# coding: utf-8

# ---
# ### Prepare REGEN P data for analysis
# #### Leah Brown, 07/29/2025
# ---

# Daily data, make monthly total precip. From 1950-2016. 1 degree lat and lon- need to regrid to 0.5 deg. Raw data in mm/day, make mm/month.

# In[ ]:





# In[1]:


# import
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
#import rioxarray as rxr
import xesmf as xe
from datetime import datetime
import warnings
warnings.filterwarnings(action='ignore')


# In[2]:


# interactive plotting stuff 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import matplotlib.gridspec as gridspec
import seaborn as sns
#import matplotlib.dates as mdates
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 12, 6
#%config InlineBackend.figure_format = 'retina'

import cartopy
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

from datetime import datetime


# In[3]:


# directories
script = os.getcwd()+ '/nsiegert/projects/aridity/code/dataprep/REGEN_P.ipynb'
#Save this data to Noeal's P folder
output_dir = 'nsiegert/projects/aridity/data/P'

root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
os.chdir(root_dir)


# In[ ]:





# In[4]:


# List all REGEN data (1950-2016). Open files
files = glob.glob('Data/Observations/REGEN/AllStns/REGEN_AllStns_V1-2019_*.nc')
files.sort()
regen = xr.open_mfdataset(files)
regen


# In[5]:


# Create REGEN data set, keeping precip data only (not keeping sd, ek, and s). Load out of dask format
regen_P = xr.Dataset({
            'P': regen.p.load()
        }
    )

regen_P


# In[6]:


#Take monthly total precip. Have daily data over multiple years, so using resample(1MS) to sum by month without combining all year's months
monthly_sum = regen_P['P'].resample(time='1MS').sum()

# Make into dataarray with designated variable. Not sure if this is necessary but works more consistently
monthly_sum = xr.Dataset(
    {
        'P': monthly_sum
    }
)
monthly_sum


# In[7]:


# Regrid to 0.5 deg

 # load in Noel's reference grid
ds_ref = xr.open_dataset(os.path.join(root_dir, 'nsiegert/projects/aridity/data', 'halfdeg_ref_grid_repaired.nc'))
#Define the regridder
regridder = xe.Regridder(ds_in=monthly_sum, ds_out=ds_ref, method='conservative')
#Regrid the data
da_regrid = regridder(monthly_sum.P, keep_attrs=True)

# Make data array
regen_P_sum = xr.Dataset(
    {
        'P': da_regrid
    }
)
regen_P_sum


# In[8]:


# Load in land-ocean mask
LO_mask = xr.open_dataset('nsiegert/projects/aridity/data/landseamask/era5_landmask_halfdeg.nc')
LO_mask = LO_mask['landfrac']

# Mask out anywhere not land according to mask
masked_regen = regen_P_sum.where(LO_mask > 0)


# In[12]:


# Finally, add attrs
masked_regen.P.attrs['desc'] = 'Total monthly precipitation'
masked_regen.P.attrs['units'] = 'mm/month'
masked_regen.attrs['info'] = 'Monthly REGEN data, 1950-2016. Originally daily, 1.0 deg resolution, regrid to monthly, 0.5 deg'
masked_regen.attrs['script'] = 'nsiegert/projects/aridity/code/dataprep/TerraClimate_P.ipynb'
now = datetime.now() # get datetime
masked_regen.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

# save all years of data in one file in Noel's P folder
masked_regen.to_netcdf(os.path.join(output_dir, 'P.REGEN.1950-2016.nc'))

print('saved P.REGEN.1950-2016')

