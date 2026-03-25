import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import datetime as dt
import numpy as np
import geopandas as gpd
from rasterio import features
from scipy import stats
import math
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import statsmodels.api as sm
from shapely.geometry import Polygon, MultiPolygon, Point
import cartopy.io.shapereader as shpreader
import geopandas as gpd
import pickle
import csv
import glob
import xarray as xr
import numpy as np
import cftime
import xskillscore
from scipy.stats import linregress
from scipy.stats import t
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


def waterbasin(data):

    # --- Fix longitude to [-180, 180] ---
    data = data.assign_coords(
        lon=((data.lon + 180) % 360 - 180)
    ).sortby("lon")

    # --- Load basin boundaries ---
    root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
    grdc_basins = gpd.read_file(
        os.path.join(root_dir, "Data", "Other", "grdc_basins")
    )

    # Ensure column names are correct
    grdc_basins = grdc_basins.rename(
        columns={"MRBID": "region_id", "RIVER_BASI": "region_name"}
    )

    # --- Build RegionMask ---
    regions = regionmask.from_geopandas(
        grdc_basins[['region_id', 'geometry']],
        names='region_id',
        numbers='region_id'
    )

    # --- 3D fractional overlap mask ---
    fracmask = regions.mask_3D_frac_approx(data)

    # --- Latitude weights ---
    weights = np.cos(np.deg2rad(data.lat))

    # --- Compute area-weighted basin mean for each basin ---
    #      (region dimension is "region")
    basin_mean = (
        data.weighted(fracmask * weights)
            .mean(dim=("lat", "lon"))
            .rename(region="basin")
    )
    
    for coord in ['names', 'abbrevs']:
        if coord in basin_mean.coords:
            basin_mean = basin_mean.drop_vars(coord)

    return basin_mean



def proccess_sst(sst_observed, string):
    if 'nv' in sst_observed.dims: 
        sst_observed=sst_observed.drop_dims('nv')
    if 'bnds' in sst_observed.dims:  
        sst_observed=sst_observed.drop_dims('bnds')
    if 'nbnds' in sst_observed.dims:  
        sst_observed=sst_observed.drop_dims('nbnds')
    if 'latitude' in sst_observed.dims:
        sst_observed=sst_observed.rename({'latitude':'lat', 'longitude':'lon'})
    
    if(string=='hadsst4'):
        #change the lons to be from 0 to 360
        sst_observed['lon'] = (sst_observed['lon'] + 360) %360
        sst_observed=sst_observed.sortby('lon')
        #convert to the water year, changing into seasons first
        wy_sst_obs=wy(sst_observed.tos, 'sst')
        #hadsst4 is already in anomaly values
        sst_anom=wy_sst_obs  
    
    elif(string=='tas'):
                #convert to kelvin
        if sst_observed[string].units != 'K':
            sst_observed=sst_observed+273
            sst_observed.assign_attrs(units="K")
        #convert to the water year, changing into seasons first
        wy_sst_obs=wy(sst_observed[string], string)
        #find the anomaly of sst relative to 1961-1990
        sst_anom=wy_sst_obs-wy_sst_obs.sel(wateryear=slice(1961, 1990)).mean(dim='wateryear')
    
    else:
        #convert to kelvin
        if sst_observed['sst'].units != 'K':
            sst_observed=sst_observed+273
            sst_observed.assign_attrs(units="K")
        #convert to the water year, changing into seasons first
        wy_sst_obs=wy(sst_observed['sst'], 'sst')
        #find the anomaly of sst relative to 1961-1990
        sst_anom=wy_sst_obs-wy_sst_obs.sel(wateryear=slice(1961, 1990)).mean(dim='wateryear')

    return(sst_anom)


def convert_to_mm_month(data):
    # Check if data has units attribute
    if not hasattr(data, 'units'):
        print("No units")
        return data
    
    units = data.units
    
    if units == 'm/s':
        # get mm per month: (originally m/s)
        mmperday = data * 86400000
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['kg/m2/s', 'kg/m^2/s', 'kg m-2 s-1']:  
        # get mm per month: (originally kg/m2/s)
        mmperday = data * 86400
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['W/m2', 'W/m^2', 'W m-2']:
        # get mm per month: (originally W/m^2)
        mmperday = data * 86400
        mmperday = mmperday / (2.25*10**6)
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'mm/day':
        # get mm per month: (originally mm/day)
        mmperday = data
        mmpermonth = (mmperday * mmperday.time.dt.days_in_month)
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'm':
        # get mm per month: (originally m/month)
        mmpermonth = data * 1000
        mmpermonth.assign_attrs(units='mm/month')
    elif units == 'cm':
        # get mm per month: (originally cm/month)
        mmpermonth = data * 10
        mmpermonth.assign_attrs(units='mm/month')
    elif units in ['mm/month', 'mm month-1']:
        # already in mm per month
        mmpermonth = data
    else:
        # Exception: units don't fall into a recognized category
        print(f"Exception units = {units}")
        return data
    
    return mmpermonth

def detrend_dim(da, dim, deg=1):
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def grid_area(xarray):
#fractional area of each grid cell in a lat by lon grid
    # Constants
    R = 6371  # Earth's radius in km
    deg_to_rad = np.pi / 180.0

    # Latitude and longitude arrays
    lat = xarray['lat']
    lon = xarray['lon']

    # Compute area of each grid cell using the spherical quadrilateral formula
    lat_rad = np.deg2rad(lat)

    # Latitude and longitude intervals
    if lat[1]>lat[0]:
        lat_interval = lat[1]-lat[0]  # degrees
    else:
        lat_interval = lat[0]-lat[1]  # degrees
    if lon[1]>lon[0]:
        lon_interval = lon[1]-lon[0]  # degrees
    else:
        lon_interval = lon[0]-lon[1]  # degrees

    # Calculate the grid cell area in fractional Earth's surface area

    dlat = lat_interval * deg_to_rad
    dlon = lon_interval * deg_to_rad

    area = dlon * (np.sin(lat_rad + dlat / 2) - np.sin(lat_rad - dlat / 2))

    # Convert the area to a 2D array matching lat/lon grid
    return( xr.DataArray(
        np.broadcast_to(area, (len(lon), len(lat))).T,
        coords={'lat': lat, 'lon': lon},
        dims=['lat', 'lon']
    ))

def fdr_correction_single(p_values_1d, alpha_FDR=0.05):
    """Apply BH FDR correction to a 1D array, ignoring NaNs."""
    valid_mask = ~np.isnan(p_values_1d)
    valid_p_values = p_values_1d[valid_mask]
    
    if len(valid_p_values) == 0:
        return np.full_like(p_values_1d, False, dtype=bool)
    
    # Sort and apply BH procedure
    sorted_indices = np.argsort(valid_p_values)
    sorted_p_values = valid_p_values[sorted_indices]
    N = len(sorted_p_values)
    
    threshold = np.arange(1, N + 1) * alpha_FDR / N
    significant_sorted = sorted_p_values <= threshold
    
    # Unsort the significant mask
    significant_unsorted = np.empty_like(significant_sorted)
    significant_unsorted[sorted_indices] = significant_sorted
    
    # Create result array
    result = np.full(len(p_values_1d), False, dtype=bool)
    result[valid_mask] = significant_unsorted
    
    return result

def fdr_correction(p_values, alpha_FDR=0.05, basin_dim='basin'):
    """Apply BH FDR correction separately for each basin."""
    return xr.apply_ufunc(
        fdr_correction_single,
        p_values,
        input_core_dims=[[basin_dim]],  # Apply along basin dimension
        output_core_dims=[[basin_dim]],
        vectorize=True,  # Vectorize over other dimensions
        dask='parallelized',  # Enable dask parallelization
        output_dtypes=[bool],
        kwargs={'alpha_FDR': alpha_FDR}
    )

def regression_slope(sst, precip):
    """
    Linear regression: precip ~ intercept + sst
    Returns: slope, p-value, AIC, RMSE, NRMSE
    """
    import statsmodels.api as sm
    import numpy as np
    
    sst = np.asarray(sst)
    precip = np.asarray(precip)

    valid = ~np.isnan(sst) & ~np.isnan(precip)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    sst_valid, precip_valid = sst[valid], precip[valid]
    if np.all(sst_valid == sst_valid[0]):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    X = sm.add_constant(sst_valid)
    model = sm.OLS(precip_valid, X).fit()

    slope = model.params[1]
    p_value = model.pvalues[1]
    aic = model.aic

    residuals = precip_valid - model.predict(X)
    rmse = np.sqrt(np.mean(residuals ** 2))
    nrmse = rmse / np.std(precip_valid)

    return slope, p_value, aic, rmse, nrmse

def regression_slope_se(sst, precip):
    """
    Linear regression: precip ~ intercept + sst
    Returns: slope, slope_se, p-value
    """
    import statsmodels.api as sm
    import numpy as np
    
    sst = np.asarray(sst)
    precip = np.asarray(precip)

    valid = ~np.isnan(sst) & ~np.isnan(precip)
    if np.sum(valid) < 2:
        return np.nan, np.nan, np.nan

    sst_valid, precip_valid = sst[valid], precip[valid]
    if np.all(sst_valid == sst_valid[0]):
        return np.nan, np.nan, np.nan

    X = sm.add_constant(sst_valid)
    model = sm.OLS(precip_valid, X).fit()

    slope = model.params[1]
    slope_se = model.bse[1]   # standard error of the slope
    p_value = model.pvalues[1]

    return slope, slope_se, p_value

def regression_with_lags(sst, precip, n_lags=3):
    """
    Regression of precip on SST with lags.
    Returns:
        - lag coefficients
        - F-statistic p-value for overall regression
        - AIC
        - RMSE (root mean square error of prediction)
    """
    import numpy as np
    from scipy.stats import f

    if np.isnan(sst).any() or np.isnan(precip).any():
        return tuple([np.nan] * (n_lags + 4)) 

    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")

    if len(sst) <= n_lags or len(precip) <= n_lags:
        return tuple([np.nan] * (n_lags + 4))

    # Create lagged predictors
    X_cols = [sst[n_lags - 1 - lag: -lag if lag != 0 else None] for lag in range(n_lags)]
    X = np.column_stack(X_cols)
    Y = precip[n_lags - 1:]

    # Add intercept
    X = np.hstack([np.ones((X.shape[0], 1)), X])

    try:
        # Fit model
        beta, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
        Y_pred = X @ beta
        rss_full = np.sum((Y - Y_pred) ** 2)

        # RMSE calculation
        rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
        nrmse = rmse / np.std(Y)

        # Null model RSS
        Y_mean = np.mean(Y)
        rss_null = np.sum((Y - Y_mean) ** 2)

        # Degrees of freedom
        n = len(Y)
        p = X.shape[1] - 1
        df1 = p
        df2 = n - p - 1

        # F-statistic and p-value
        if df2 <= 0 or rss_full <= 0:
            return tuple([np.nan] * (n_lags + 3))

        F_stat = ((rss_null - rss_full) / df1) / (rss_full / df2)
        f_p_value = 1 - f.cdf(F_stat, df1, df2)

        # AIC
        aic = n * np.log(rss_full / n) + 2 * (p + 1)

        lag_coefs = beta[1:]  # exclude intercept
        return tuple(lag_coefs) + (f_p_value, aic, rmse, nrmse)

    except (np.linalg.LinAlgError, ValueError):
        return tuple([np.nan] * (n_lags + 4))


def regression_with_conditioning_var_and_interaction(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    interaction = sst * con
    X = np.column_stack((sst, con, interaction))
    X = sm.add_constant(X)

    model = sm.OLS(precip, X).fit()
    slope_sst = model.params[1]
    slope_interaction = model.params[3]
    pvalue_sst = model.pvalues[1]
    pvalue_interaction = model.pvalues[3]
    aic = model.aic
    rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    nrmse = rmse / np.std(precip)

    return slope_sst, slope_interaction, pvalue_sst, pvalue_interaction, aic, rmse, nrmse

def regression_with_conditioning_var(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan 

    X = np.column_stack((sst, con))
    X = sm.add_constant(X)

    model = sm.OLS(precip, X).fit()
    slope_sst = model.params[1]
    pvalue_sst = model.pvalues[1]
    aic = model.aic
    rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    nrmse = rmse / np.std(precip)
    return slope_sst, pvalue_sst, aic, rmse, nrmse


def regression_poly2_raw(sst, precip):
    """
    Regress raw precipitation on raw SST and SST^2.
    Returns:
        - intercept (const)
        - beta1: coefficient for SST
        - beta2: coefficient for SST^2
        - p1: p-value for SST
        - p2: p-value for SST^2
        - AIC
        - RMSE
        - NRMSE
    """
    import numpy as np
    import statsmodels.api as sm

    if np.isnan(sst).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    sst_squared = sst ** 2
    X = np.column_stack((sst, sst_squared))
    X = sm.add_constant(X)  # adds intercept

    try:
        model = sm.OLS(precip, X).fit()
        intercept = model.params[0]
        beta1 = model.params[1]
        beta2 = model.params[2]
        p1 = model.pvalues[1]
        p2 = model.pvalues[2]
        aic = model.aic
        rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
        nrmse = rmse / np.std(precip)
        return intercept, beta1, beta2, p1, p2, aic, rmse, nrmse

    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
def regression_log_raw(sst, precip):
    """
    Regress raw precipitation on SST and log(SST).
    Returns:
        - intercept (const)
        - beta1: coefficient for SST
        - beta2: coefficient for log(SST)
        - p1: p-value for SST
        - p2: p-value for log(SST)
        - AIC
        - RMSE
        - NRMSE
    """
    import numpy as np
    import statsmodels.api as sm

    # Basic sanity check: SST must be positive to take log
    if np.any(sst <= 0) or np.isnan(sst).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    sst_log = np.log(sst)
    X = np.column_stack((sst, sst_log))
    X = sm.add_constant(X)  # adds intercept

    try:
        model = sm.OLS(precip, X).fit()
        intercept = model.params[0]
        beta1 = model.params[1]
        beta2 = model.params[2]
        p1 = model.pvalues[1]
        p2 = model.pvalues[2]
        aic = model.aic
        rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
        nrmse = rmse / np.std(precip)
        return intercept, beta1, beta2, p1, p2, aic, rmse, nrmse

    except Exception:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def regression_with_conditioning_var_return_all(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # Build design matrix
    X = np.column_stack((sst, con))
    X = sm.add_constant(X)

    # Fit regression
    model = sm.OLS(precip, X).fit()

    # Extract parameters
    slope_sst = model.params[1]
    pvalue_sst = model.pvalues[1]

    slope_con = model.params[2]
    pvalue_con = model.pvalues[2]

    # Model evaluation metrics
    aic = model.aic
    rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    nrmse = rmse / np.std(precip)
    
    # Correlation between observed and predicted values
    r = np.corrcoef(model.fittedvalues, precip)[0, 1]

    return slope_sst, pvalue_sst, slope_con, pvalue_con, r

def regression_with_conditioning_var_and_interaction_return_all(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan 

    interaction = sst * con
    X = np.column_stack((sst, con, interaction))
    X = sm.add_constant(X)

    model = sm.OLS(precip, X).fit()
    slope_sst = model.params[1]
    slope_rh = model.params[2]
    slope_interaction = model.params[3]
    pvalue_sst = model.pvalues[1]
    pvalue_rh = model.pvalues[2]
    pvalue_interaction = model.pvalues[3]
    aic = model.aic
    rmse = np.sqrt(np.mean((model.fittedvalues - precip) ** 2))
    nrmse = rmse / np.std(precip)
    
    
    # Correlation between observed and predicted values
    r = np.corrcoef(model.fittedvalues, precip)[0, 1]


    return slope_sst, slope_rh, slope_interaction, pvalue_sst, pvalue_rh, pvalue_interaction, r


def regression_with_conditioning_var_se(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Build design matrix
    X = np.column_stack((sst, con))
    X = sm.add_constant(X)

    # Fit regression
    model = sm.OLS(precip, X).fit()

    # Extract parameters
    slope_sst = model.params[1]
    slope_sst_se = model.bse[1]
    pvalue_sst = model.pvalues[1]

    slope_con = model.params[2]
    slope_con_se = model.bse[2]
    pvalue_con = model.pvalues[2]

    return slope_sst, slope_sst_se, pvalue_sst, slope_con, slope_con_se, pvalue_con

def regression_with_conditioning_var_and_interaction_se(sst, con, precip):
    import statsmodels.api as sm
    import numpy as np

    if np.isnan(sst).any() or np.isnan(con).any() or np.isnan(precip).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    interaction = sst * con
    X = np.column_stack((sst, con, interaction))
    X = sm.add_constant(X)

    model = sm.OLS(precip, X).fit()
    slope_sst = model.params[1]
    slope_sst_se = model.bse[1]
    pvalue_sst = model.pvalues[1]
    
    slope_rh = model.params[2]
    slope_rh_se = model.bse[2]
    pvalue_rh = model.pvalues[2]
    
    slope_interaction = model.params[3]
    slope_interaction_se = model.bse[3]
    pvalue_interaction = model.pvalues[3]

    return (slope_sst, slope_sst_se, pvalue_sst, slope_rh, slope_rh_se, pvalue_rh, slope_interaction, slope_interaction_se, pvalue_interaction)

