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
from affine import Affine
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
import statsmodels.api as sm
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def cv_regression_model(sst, precip, model_id=None, n_splits=5):
    """
    Applies K-Fold CV for different regression models for basin-scale SST-precip relationship.
    Returns RMSE (mean,std), MAE (mean,std), coef (mean,std), adjR2 (mean,std).
    """
    import statsmodels.api as sm
    
    sst = np.asarray(sst)
    precip = np.asarray(precip)
    
    # Mask for valid data
    valid = ~np.isnan(sst) & ~np.isnan(precip)
    
    if np.sum(valid) < n_splits:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    sst = sst[valid]
    precip = precip[valid]

    # Construct design matrix X based on model_id
    if model_id == f'P ~ β*SST':
        X_list = [sst]
    elif model_id == f'P ~ intercept + β1*SST + β2*SST²':
        X_list = [sst, sst ** 2]
    elif model_id == f'P ~ intercept + β1*SST + β2*log(SST)':
        X_list = [sst, np.log(np.where(sst > 0, sst, np.nan))]
    elif model_id == f'P(t) ~ β₀·SST(t) + β₁·SST(t-1)':
        X_list = [sst[:-1], sst[1:]]
        precip = precip[1:]
    elif model_id == f'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)':
        X_list = [sst[:-2], sst[1:-1], sst[2:]]
        precip = precip[2:]
    elif model_id == f'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)':
        X_list = [sst[:-3], sst[1:-2], sst[2:-1], sst[3:]]
        precip = precip[3:]
    elif model_id == f'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)':
        X_list = [sst[:-4], sst[1:-3], sst[2:-2], sst[3:-1], sst[4:]]
        precip = precip[4:]
    else:
        X_list = [sst]

    # Ensure consistent shape
    X = np.column_stack(X_list)
    X = sm.add_constant(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list, adjr2_list, rmse_list, mae_list, nrmse_list, coef_list = [], [], [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = precip[train_idx], precip[test_idx]

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            n, k = len(y_test), X_test.shape[1]-1  # exclude constant
            adj_r2 = 1 - (1-r2) * (n-1)/(n-k-1) if n > k+1 else np.nan

            r2_list.append(r2)
            adjr2_list.append(adj_r2)
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_list.append(rmse_val)
            nrmse_list.append(rmse_val / np.std(y_test) if np.std(y_test) > 0 else np.nan)
            mae_list.append(mean_absolute_error(y_test, y_pred))
            coef_list.append(model.params[1])  

        except:
            continue

    if len(r2_list) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # FIXED: Convert all outputs to Python floats (scalars), not np.array
    def to_scalar(x):
        """Force conversion to Python float scalar"""
        return float(np.asarray(x).item())

    return (
        to_scalar(np.mean(rmse_list)), to_scalar(np.std(rmse_list)),
        to_scalar(np.mean(mae_list)), to_scalar(np.std(mae_list)),
        to_scalar(np.mean(nrmse_list)), to_scalar(np.std(nrmse_list)),
        to_scalar(np.mean(coef_list)), to_scalar(np.std(coef_list)),
        to_scalar(np.mean(adjr2_list)), to_scalar(np.std(adjr2_list))
    )


def cv_regression_model_with_predictor(sst, precip, predictor, model_id=None, n_splits=5):
    """
    Applies K-Fold CV for different regression models for basin-scale SST-precip relationship.
    predictor: arrays (olr, lr, gmst, rh, etc.), aligned in time.
    Returns RMSE (mean,std), MAE (mean,std), coef (mean,std), adjR2 (mean,std).
    """
    import statsmodels.api as sm
    
    sst = np.asarray(sst)
    precip = np.asarray(precip)
    predictor = np.asarray(predictor)
    
    # Mask for valid data
    valid = ~np.isnan(sst) & ~np.isnan(precip) & ~np.isnan(predictor)
    
    if np.sum(valid) < n_splits:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    sst = sst[valid]
    precip = precip[valid]
    predictor = predictor[valid]

    # Construct design matrix X based on model_id
    if model_id == f'P ~ β₀*SST(SST ≤ T_c) + β₁*SST(SST > T_c)':
        threshold = predictor  
        sst_below = sst * (sst <= threshold)
        sst_above = sst * (sst > threshold)
        if np.all(sst_below == 0) or np.all(sst_above == 0):
            X_list = [sst]
        else:
            X_list = [sst_below, sst_above]
    elif model_id in [f'P ~ β₀·SST + β₁·OLR + β₂·SST·OLR',
                      f'P ~ β₀·SST + β₁·LR + β₂·SST·LR',
                      f'P ~ β₀·SST + β₁·GMST + β₂·SST·GMST',
                      f'P ~ β₀·SST + β₁·RH + β₂·SST·RH',
                     'P ~ β₀·SST + β₁·z500 RH + β₂·SST·z500 RH',
                     'P ~ β₀·SST + β₁·column RH + β₂·SST·column RH',
                     'P ~ β₀·SST + β₁·sfc RH + β₂·SST·sfc RH']:
        X_list = [sst, predictor, sst * predictor]
    else:
        X_list = [sst, predictor]

    # Ensure consistent shape
    X = np.column_stack(X_list)
    X = sm.add_constant(X)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2_list, adjr2_list, rmse_list, mae_list, nrmse_list, coef_list = [], [], [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = precip[train_idx], precip[test_idx]

        try:
            model = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            n, k = len(y_test), X_test.shape[1]-1
            adj_r2 = 1 - (1-r2) * (n-1)/(n-k-1) if n > k+1 else np.nan

            r2_list.append(r2)
            adjr2_list.append(adj_r2)
            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_list.append(rmse_val)
            nrmse_list.append(rmse_val / np.std(y_test) if np.std(y_test) > 0 else np.nan)
            mae_list.append(mean_absolute_error(y_test, y_pred))
            coef_list.append(model.params[1])  

        except:
            continue

    if len(r2_list) == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    # FIXED: Convert all outputs to Python floats (scalars), not np.array
    def to_scalar(x):
        """Force conversion to Python float scalar"""
        return float(np.asarray(x).item())

    return (
        to_scalar(np.mean(rmse_list)), to_scalar(np.std(rmse_list)),
        to_scalar(np.mean(mae_list)), to_scalar(np.std(mae_list)),
        to_scalar(np.mean(nrmse_list)), to_scalar(np.std(nrmse_list)),
        to_scalar(np.mean(coef_list)), to_scalar(np.std(coef_list)),
        to_scalar(np.mean(adjr2_list)), to_scalar(np.std(adjr2_list))
    )