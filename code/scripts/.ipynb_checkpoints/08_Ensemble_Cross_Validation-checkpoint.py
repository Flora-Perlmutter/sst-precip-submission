import numpy as np
import xarray as xr
import os
import sys
import argparse
import warnings
import pandas as pd
import glob
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")

# ============================================================================
# SETUP
# ============================================================================
root_dir = '/dartfs-hpc/rc/lab/C/CMIG'
functions_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Scripts')
outputs_dir = os.path.join(root_dir, 'fperlmutter/Observational_Regressions_Project/Data/Processed')
data_dir = os.path.join(root_dir, 'Data/Observations')

sys.path.append(functions_dir)
from regression_functions import waterbasin, detrend_dim
from cross_validation_functions import cv_regression_model, cv_regression_model_with_predictor

# Define precipitation datasets
PRECIP_DATASETS = {
    'GPCP': None,
    'CRU': None,
    'GPCC': None,
    'CPC': None,
    'UDel': None,
    'PREC': None,
    'TerraClimate': None,
    'REGEN': None,
}

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================
def get_models_to_run(sst_detrended, precip_detrended, sst_raw, precip_raw, olr_regridded=None, lapse_rate_regridded=None, 
                      gmst_detrended_expanded=None,
                      rh_z500_detrended_expanded=None, rh_column_detrended_expanded=None,
                      observed_precip_constrained_detrended_rh= None):
    """
    Define all models to run for cross-validation.
    Only includes models for which all required predictors are available.
    """
    models = []
    
    # Models without conditioning variables
    models.extend([
        {
            'model_id': 'P ~ β*SST',
            'description': 'Linear regression on monthly data',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_detrended,
            'precip': precip_detrended
        },
        {
            'model_id': 'P ~ intercept + β1*SST + β2*SST²',
            'description': 'Quadratic regression on monthly raw time series',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_raw,
            'precip': precip_raw
        },
        {
            'model_id': 'P ~ intercept + β1*SST + β2*log(SST)',
            'description': 'Log-linear regression on monthly raw time series',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_raw,
            'precip': precip_raw
        },
        {
            'model_id': 'P(t) ~ β₀·SST(t) + β₁·SST(t-1)',
            'description': 'Linear regression with lags on monthly data',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_detrended,
            'precip': precip_detrended
        },
        {
            'model_id': 'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)',
            'description': 'Linear regression with lags on monthly data',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_detrended,
            'precip': precip_detrended
        },
        {
            'model_id': 'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)',
            'description': 'Linear regression with lags on monthly data',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_detrended,
            'precip': precip_detrended
        },
        {
            'model_id': 'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)',
            'description': 'Linear regression with lags on monthly data',
            'time_scale': 'monthly',
            'variable': 'precip',
            'sst': sst_detrended,
            'precip': precip_detrended
        }
    ])
    
    if olr_regridded is not None:
        models.extend([
            {
                'model_id': 'P ~ β₀·SST + β₁·OLR + β₂·SST·OLR',
                'description': 'Linear regression with OLR as a conditioning variable and an interaction term, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': olr_regridded
            },
            {
                'model_id': 'P ~ β₀·SST + β₁·OLR',
                'description': 'Linear regression with OLR as a conditioning variable, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': olr_regridded
            }
        ])
    
    if lapse_rate_regridded is not None:
        models.extend([
            {
                'model_id': 'P ~ β₀·SST + β₁·LR + β₂·SST·LR',
                'description': 'Linear regression with lapse rate as a conditioning variable and an interaction term, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': lapse_rate_regridded
            },
            {
                'model_id': 'P ~ β₀·SST + β₁·LR',
                'description': 'Linear regression with lapse rate as a conditioning variable, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': lapse_rate_regridded
            }
        ])
    
    if gmst_detrended_expanded is not None:
        models.extend([
            {
                'model_id': 'P ~ β₀·SST + β₁·GMST + β₂·SST·GMST',
                'description': 'Linear regression with global mean surface temperature as a conditioning variable and an interaction term, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': gmst_detrended_expanded
            },
            {
                'model_id': 'P ~ β₀·SST + β₁·GMST',
                'description': 'Linear regression with global mean surface temperature as a conditioning variable, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': precip_detrended,
                'predictor': gmst_detrended_expanded
            }
        ])
    
    if rh_z500_detrended_expanded is not None:
        models.extend([
            {
                'model_id': 'P ~ β₀·SST + β₁·z500 RH + β₂·SST·z500 RH',
                'description': 'Linear regression with relative humidity as a conditioning variable and an interaction term, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': observed_precip_constrained_detrended_rh,
                'predictor': rh_z500_detrended_expanded
            },
            {
                'model_id': 'P ~ β₀·SST + β₁·z500 RH',
                'description': 'Linear regression with relative humidity as a conditioning variable, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': observed_precip_constrained_detrended_rh,
                'predictor': rh_z500_detrended_expanded
            }
        ])
    
    if rh_column_detrended_expanded is not None:
        models.extend([
            {
                'model_id': 'P ~ β₀·SST + β₁·column RH + β₂·SST·column RH',
                'description': 'Linear regression with relative humidity as a conditioning variable and an interaction term, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': observed_precip_constrained_detrended_rh,
                'predictor': rh_column_detrended_expanded
            },
            {
                'model_id': 'P ~ β₀·SST + β₁·column RH',
                'description': 'Linear regression with relative humidity as a conditioning variable, monthly data',
                'time_scale': 'monthly',
                'variable': 'precip',
                'sst': sst_detrended,
                'precip': observed_precip_constrained_detrended_rh,
                'predictor': rh_column_detrended_expanded
            }
        ])
    
    return models

# ============================================================================
# CROSS-VALIDATION PROCESSING
# ============================================================================
def process_model_cv(model_dict):
    """
    Run cross-validation for a single model.
    """
    model_id = model_dict['model_id']
    sst_input = model_dict['sst']
    precip_input = model_dict['precip']
    predictor = model_dict.get('predictor', None)
    
    if model_id in [
        'P ~ intercept + β1*SST + β2*SST²',
        'P ~ intercept + β1*SST + β2*log(SST)',
        'P ~ β*SST',
        'P(t) ~ β₀·SST(t) + β₁·SST(t-1)',
        'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)',
        'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)',
        'P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)'
    ]:
    
        cv_output = xr.apply_ufunc(
            cv_regression_model,
            sst_input,
            precip_input,
            kwargs={'model_id': model_id},
            input_core_dims=[['time'], ['time']],
            output_core_dims=[[], [], [], [], [], [], [], [], [], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float] * 10
        )
    else:
        cv_output = xr.apply_ufunc(
            cv_regression_model_with_predictor,
            sst_input,
            precip_input,
            predictor,
            kwargs={'model_id': model_id},
            input_core_dims=[['time'], ['time'], ['time']],
            output_core_dims=[[], [], [], [], [], [], [], [], [], []],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float] * 10
        )
    

    # Create Dataset with only data variables (not metadata strings)
    cv_ds = xr.Dataset(
        data_vars=dict(
            rmse_mean=cv_output[0],
            rmse_std=cv_output[1],
            mae_mean=cv_output[2],
            mae_std=cv_output[3],
            coef_mean=cv_output[6],
            coef_std=cv_output[7],
            adjr2_mean=cv_output[8],
            adjr2_std=cv_output[9],
            nrmse_mean=cv_output[4],
            nrmse_std=cv_output[5],
        ),
        attrs=dict(
            model_id=model_dict['model_id'],
            description=model_dict['description'],
            time_scale=model_dict['time_scale'],
            variable=model_dict['variable'],
        )
    )
    
    return model_id, cv_ds




def process_ensemble_pair(p_name, p_da, sst_name, sst_da, sst_raw, n_jobs=16):
    """
    Process a single precipitation-SST pair: prepare data and run CV for all models.
    
    Returns:
        dict: Cross-validation results for all models
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")

    
    p_anom = p_da.load()
    
    common_time = np.intersect1d(p_anom['time'].values, sst_da['time'].values)
    p_anom = p_anom.sel(time=common_time)
    sst_anom = sst_da.sel(time=common_time)
    sst_raw = sst_raw.sel(time=common_time)
    
    # Detrending
    print("Detrending...")
    p_detrended = detrend_dim(p_anom, 'time').astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, 'time').astype(np.float32)
    
    #Process OLR
    print("Processing OLR...")
    os.chdir('/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/NCEP-DOE-R2/monthly/radiation')
    try:
        olr = xr.open_dataset('ulwrf.ntat.mon.mean.nc')
        olr = olr.drop_dims('nbnds')
        olr_constrained = olr.sel(time=common_time)
        olr_anom = olr_constrained.groupby(olr_constrained.time.dt.month) - olr_constrained.groupby(olr_constrained.time.dt.month).mean('time')
        # Detrend using your existing function
        olr_detrended = detrend_dim(olr_anom.ulwrf, 'time')
        olr_regridded = olr_detrended.interp(coords={'lat': sst_detrended['lat'], 'lon': sst_detrended['lon']})
    except Exception as e:
        print(f"  Warning: Could not process OLR: {e}")
        olr_regridded = None
    
    #Process Lapse Rate
    print("Processing lapse rate...")
    os.chdir('/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/NCEP-DOE-R2/monthly/radiation')
    try:
        hgt = xr.open_dataset('hgt.mon.mean.nc')
        temp = xr.open_dataset('air.mon.mean.nc')  
                       
        # lapse rate between 850 hPa and 500 hPa
        T850 = temp['air'].sel(level=850)  # in Kelvin
        T500 = temp['air'].sel(level=500)
        Z850 = hgt['hgt'].sel(level=850)
        Z500 = hgt['hgt'].sel(level=500)
        # Calculate lapse rate (convert Kelvin to °C/km)
        lapse_rate = (T850 - T500) / ((Z500 - Z850) / 1000)  # °C/km       
        lapse_rate_constrained = lapse_rate.sel(time=common_time)
        # find the lapse rate anomaly
        lapse_rate_anom = lapse_rate_constrained.groupby(lapse_rate_constrained.time.dt.month) - lapse_rate_constrained.groupby(lapse_rate_constrained.time.dt.month).mean('time')
        # Detrend
        lapse_rate_detrended = detrend_dim(lapse_rate_anom, 'time')
        # regrid to sst coordinates
        lapse_rate_regridded = lapse_rate_detrended.interp(coords={'lat': sst_detrended['lat'], 'lon': sst_detrended['lon']})                   
        lapse_rate_regridded = lapse_rate_regridded.transpose('lat', 'lon', 'time')
    except Exception as e:
        print(f"  Warning: Could not process lapse rate: {e}")
        lapse_rate_regridded = None

    #GMST processing 
    print("Processing GMST...")
    os.chdir('/dartfs-hpc/rc/lab/C/CMIG/Data/Observations/BerkeleyEarth')
    try:
        gmst = pd.read_csv('gmst.csv')
        # Convert to datetime
        gmst['time'] = pd.to_datetime(gmst['time'])
        # Convert to xarray DataArray with time coordinate
        gmst_xr = xr.DataArray(gmst['t_abs'].values, coords=[gmst['time']], dims='time', name='gmst')
        gmst_xr_constrained = gmst_xr.sel(time=common_time)
        # find the anomaly
        gmst_anom = gmst_xr_constrained.groupby(gmst_xr_constrained.time.dt.month) - gmst_xr_constrained.groupby(gmst_xr_constrained.time.dt.month).mean('time')
        # Detrend 
        gmst_detrended = detrend_dim(gmst_anom, 'time')
        # Expand gmst_detrended to match dimensions
        gmst_detrended_expanded = gmst_detrended.expand_dims(
            lat=sst_detrended.lat,
            lon=sst_detrended.lon
        ).transpose("time", "lat", "lon")
    except Exception as e:
        print(f"  Warning: Could not process GMST: {e}")
        gmst_detrended_expanded = None
    
    
    #Relative humidity processing
    #MERRA-2
    print("Processing relative humidity...")
    rh_z500_detrended_expanded = None
    rh_column_detrended_expanded = None
    observed_precip_constrained_detrended_rh = None
    
    try:
        # --- Load full MERRA-2 RH (all levels)
        rh_dir = os.path.join(root_dir, 'Data/Observations/MERRA-2/RH')
        files = sorted(glob.glob(os.path.join(rh_dir, "*.nc4")))
        
        if len(files) == 0:
            print(f"  Warning: No RH files found in {rh_dir}")
        else:
            rh = xr.open_mfdataset(files, combine='by_coords')
            
            # ============================================================
            # Mid-level (500 hPa)
            # ============================================================
            z500_rh = rh.RH.sel(lev=500)
            z500_rh_basin = waterbasin(z500_rh)
            
            common_basin = np.intersect1d(p_detrended['basin'], z500_rh_basin['basin'])
            rh_constrained = z500_rh_basin.sel(time=common_time, basin=common_basin)
            observed_precip_constrained_detrended_rh = p_detrended.sel(basin=common_basin)
            
            rh_anom = rh_constrained.groupby(rh_constrained.time.dt.month) - \
                      rh_constrained.groupby(rh_constrained.time.dt.month).mean('time')
            rh_detrended = detrend_dim(rh_anom, 'time')
            rh_z500_detrended_expanded = rh_detrended.expand_dims(
                lat=sst_anom.lat,
                lon=sst_anom.lon
            ).load()
            
            # ============================================================
            # Column-integrated RH (pressure-weighted)
            # ============================================================
            pressure = rh.lev.values
            dp = np.gradient(pressure)
            weights = dp / np.sum(dp)
            weights_da = xr.DataArray(weights, coords={'lev': rh.lev}, dims='lev')
            
            rh_column = (rh.RH * weights_da).sum(dim='lev')
            column_rh_basin = waterbasin(rh_column)
            
            common_basin = np.intersect1d(p_detrended['basin'], column_rh_basin['basin'])
            rh_constrained = column_rh_basin.sel(time=common_time, basin=common_basin)
            
            rh_anom = rh_constrained.groupby(rh_constrained.time.dt.month) - \
                      rh_constrained.groupby(rh_constrained.time.dt.month).mean('time')
            rh_detrended = detrend_dim(rh_anom, 'time')
            rh_column_detrended_expanded = rh_detrended.expand_dims(
                lat=sst_anom.lat,
                lon=sst_anom.lon
            ).load()
    except Exception as e:
        print(f"  Warning: Could not process relative humidity: {e}")

        
    # Get models to run
    models_to_run = get_models_to_run(
        sst_detrended, 
        p_detrended,        
        sst_raw, 
        p_anom,                
        olr_regridded=olr_regridded, 
        lapse_rate_regridded=lapse_rate_regridded,
        gmst_detrended_expanded=gmst_detrended_expanded,
        rh_z500_detrended_expanded=rh_z500_detrended_expanded, 
        rh_column_detrended_expanded=rh_column_detrended_expanded,
        observed_precip_constrained_detrended_rh=observed_precip_constrained_detrended_rh
    )
    
    # Run cross-validation in parallel
    print(f"Running {len(models_to_run)} models in parallel...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_model_cv)(model_dict) for model_dict in models_to_run
    )
    
    
    
    # Collect results
    cv_results = {model_id: cv_ds for model_id, cv_ds in results}
    
    return cv_results
    

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Cross-validation for observational ensemble')
    parser.add_argument('--pair-index', type=int, default=None,
                        help='Index of precipitation-SST pair to process (0-based). If not provided, processes all pairs.')
    parser.add_argument('--n-jobs', type=int, default=16,
                        help='Number of parallel jobs for model processing')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION ANALYSIS FOR OBSERVATIONAL ENSEMBLE")
    print("="*80)
    
    # Load precipitation anomalies
    print("\nLoading observational ensembles...")
    precip_dict = {}
    for p_name in PRECIP_DATASETS.keys():
        nc_path = os.path.join(outputs_dir, f"precip_anom_{p_name}.nc")
        if os.path.exists(nc_path):
            precip_dict[p_name] = xr.open_dataarray(nc_path)
        else:
            print(f"  Warning: {nc_path} not found")
    
    # Load SST anomalies
    sst_dict = {}
    for sst_name in ["ERSSTv5", "COBE-SST2"]:
        nc_path = os.path.join(outputs_dir, f"sst_anom_{sst_name}.nc")
        if os.path.exists(nc_path):
            sst_dict[sst_name] = xr.open_dataarray(nc_path)
        else:
            print(f"  Warning: {nc_path} not found")
    
    # Load raw SST data
    sst_raw_dict = {}
    for sst_name in ["ERSSTv5", "COBE-SST2"]:
        nc_path = os.path.join(outputs_dir, f"sst_raw_{sst_name}.nc")
        if os.path.exists(nc_path):
            sst_raw_dict[sst_name] = xr.open_dataarray(nc_path)
        else:
            print(f"  Warning: {nc_path} not found")
    
    print(f"  Loaded {len(precip_dict)} precipitation datasets")
    print(f"  Loaded {len(sst_dict)} SST anomaly datasets")
    print(f"  Loaded {len(sst_raw_dict)} raw SST datasets")
    
    # Create all pairs where SST and raw SST have the same name
    pairs = [
        (p_name, p_da, sst_name, sst_da, sst_raw_dict[sst_name])
        for p_name, p_da in precip_dict.items()
        for sst_name, sst_da in sst_dict.items()
        if sst_name in sst_raw_dict
    ]
    
    total_pairs = len(pairs)
    
    if total_pairs == 0:
        print("ERROR: No valid precipitation-SST pairs found!")
        sys.exit(1)
    
    # Process pairs
    if args.pair_index is not None:
        # Process single pair
        if args.pair_index >= total_pairs:
            print(f"ERROR: pair-index {args.pair_index} >= total pairs {total_pairs}")
            sys.exit(1)
        
        p_name, p_da, sst_name, sst_da, sst_raw = pairs[args.pair_index]
        print(f"\nProcessing pair {args.pair_index}/{total_pairs-1}: {p_name} vs {sst_name}")
        
        cv_results = process_ensemble_pair(p_name, p_da, sst_name, sst_da, sst_raw, args.n_jobs)
        
        # Save result as NetCDF
        print("\n" + "="*80)
        print("SAVING CROSS-VALIDATION RESULTS")
        print("="*80)
        
        output_file = os.path.join(
            outputs_dir,
            f'cv_results_{p_name}_{sst_name}.nc'
        )
        
        cv_results_ds = xr.concat(
            cv_results.values(),
            dim="model",
            coords='minimal'
        )
        
        cv_results_ds = cv_results_ds.assign_coords(
            model=list(cv_results.keys())
        )
        
        cv_results_ds.to_netcdf(output_file)
        
        
        print(f"    Saved CV results to: {output_file}")
    
    else:
        # Process all pairs
        all_results = {}
        
        for idx, (p_name, p_da, sst_name, sst_da, sst_raw) in enumerate(pairs):
            print(f"\nProcessing pair {idx+1}/{total_pairs}: {p_name} vs {sst_name}")
            
            cv_results = process_ensemble_pair(p_name, p_da, sst_name, sst_da, sst_raw, args.n_jobs)
            
            key = f"{p_name}_{sst_name}"
            all_results[key] = cv_results
        
        # Save all results
        print("\n" + "="*80)
        print("SAVING CROSS-VALIDATION RESULTS")
        print("="*80)
        
        output_file = os.path.join(outputs_dir, 'cv_results_ensemble_all.nc')
        
        # Convert results to xarray Dataset
        cv_results_ds = xr.Dataset(all_results)
        cv_results_ds.to_netcdf(output_file)
        
        print(f"    Saved all CV results to: {output_file}")
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()