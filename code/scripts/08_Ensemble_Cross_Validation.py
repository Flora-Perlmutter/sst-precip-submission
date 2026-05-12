#!/usr/bin/env python
# coding: utf-8
"""
Cross-validation analysis for the observational precipitation-SST ensemble.

Author: Flora Perlmutter

Description
-----------
Runs leave-one-out (or k-fold) cross-validation across all combinations of
observational precipitation and SST datasets. For each pair, the following
predictors are prepared and passed to a suite of regression models:
  - Detrended SST anomalies
  - Raw SST
  - OLR (NCEP-DOE R2)
  - Lapse rate 850-500 hPa (NCEP-DOE R2)
  - GMST (Berkeley Earth)
  - Relative humidity at 500 hPa and column-integrated (MERRA-2)

Processes all pairs sequentially
python run_cv_obs.py --n-jobs 16

"""

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA
from regression_functions import waterbasin, detrend_dim
from cross_validation_functions import cv_regression_model, cv_regression_model_with_predictor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
OUTPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OLR_DIR      = CMIG_DATA / "Data/Observations/NCEP-DOE-R2/monthly/radiation"
NCEP_DIR     = CMIG_DATA / "Data/Observations/NCEP-DOE-R2/monthly/radiation"
GMST_FILE    = CMIG_DATA / "Data/Observations/BerkeleyEarth/gmst.csv"
MERRA2_RH_DIR= CMIG_DATA / "Data/Observations/MERRA-2/RH"

# ---------------------------------------------------------------------------
# Dataset names (must match filenames written by load_and_process_obs.py)
# ---------------------------------------------------------------------------
PRECIP_NAMES = ["GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN"]
SST_NAMES    = ["ERSSTv6", "COBE-SST3"]

# Models that use only SST (no conditioning variable)
SST_ONLY_MODEL_IDS = {
    "P ~ β*SST",
    "P ~ intercept + β₁*SST + β₂*SST²",
    "P ~ intercept + β₁*SST + β₂*log(SST)",
    "P(t) ~ β₀·SST(t) + β₁·SST(t-1)",
    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)",
    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)",
    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)",
}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models_to_run(
    sst_detrended, precip_detrended,
    sst_raw, precip_raw,
    olr_regridded=None,
    lapse_rate_regridded=None,
    gmst_detrended_expanded=None,
    rh_z500_detrended_expanded=None,
    rh_column_detrended_expanded=None,
    observed_precip_constrained_detrended_rh=None,
) -> list:
    """
    Build the list of model dicts to run for one precipitation-SST pair.
    Conditioning-variable models are only included if that predictor loaded
    successfully (i.e. is not None).
    """
    models = [
        {
            "model_id":    "P ~ β*SST",
            "description": "Linear regression on monthly data",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_detrended,
            "precip":      precip_detrended,
        },
        {
            "model_id":    "P ~ intercept + β₁*SST + β₂*SST²",
            "description": "Quadratic regression on monthly raw time series",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_raw,
            "precip":      precip_raw,
        },
        {
            "model_id":    "P ~ intercept + β₁*SST + β₂*log(SST)",
            "description": "Log-linear regression on monthly raw time series",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_raw,
            "precip":      precip_raw,
        },
        {
            "model_id":    "P(t) ~ β₀·SST(t) + β₁·SST(t-1)",
            "description": "Linear regression with 1 lag",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_detrended,
            "precip":      precip_detrended,
        },
        {
            "model_id":    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)",
            "description": "Linear regression with 2 lags",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_detrended,
            "precip":      precip_detrended,
        },
        {
            "model_id":    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)",
            "description": "Linear regression with 3 lags",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_detrended,
            "precip":      precip_detrended,
        },
        {
            "model_id":    "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)",
            "description": "Linear regression with 4 lags",
            "time_scale":  "monthly",
            "variable":    "precip",
            "sst":         sst_detrended,
            "precip":      precip_detrended,
        },
    ]

    # Conditioning variable models — only added if predictor is available
    if olr_regridded is not None:
        models += [
            {"model_id": "P ~ β₀·SST + β₁·OLR + β₂·SST·OLR",
             "description": "SST + OLR interaction, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": olr_regridded},
            {"model_id": "P ~ β₀·SST + β₁·OLR",
             "description": "SST + OLR, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": olr_regridded},
        ]

    if lapse_rate_regridded is not None:
        models += [
            {"model_id": "P ~ β₀·SST + β₁·LR + β₂·SST·LR",
             "description": "SST + lapse rate interaction, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": lapse_rate_regridded},
            {"model_id": "P ~ β₀·SST + β₁·LR",
             "description": "SST + lapse rate, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": lapse_rate_regridded},
        ]

    if gmst_detrended_expanded is not None:
        models += [
            {"model_id": "P ~ β₀·SST + β₁·GMST + β₂·SST·GMST",
             "description": "SST + GMST interaction, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": gmst_detrended_expanded},
            {"model_id": "P ~ β₀·SST + β₁·GMST",
             "description": "SST + GMST, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended, "precip": precip_detrended, "predictor": gmst_detrended_expanded},
        ]

    if rh_z500_detrended_expanded is not None:
        models += [
            {"model_id": "P ~ β₀·SST + β₁·z500 RH + β₂·SST·z500 RH",
             "description": "SST + z500 RH interaction, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended,
             "precip": observed_precip_constrained_detrended_rh,
             "predictor": rh_z500_detrended_expanded},
            {"model_id": "P ~ β₀·SST + β₁·z500 RH",
             "description": "SST + z500 RH, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended,
             "precip": observed_precip_constrained_detrended_rh,
             "predictor": rh_z500_detrended_expanded},
        ]

    if rh_column_detrended_expanded is not None:
        models += [
            {"model_id": "P ~ β₀·SST + β₁·column RH + β₂·SST·column RH",
             "description": "SST + column RH interaction, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended,
             "precip": observed_precip_constrained_detrended_rh,
             "predictor": rh_column_detrended_expanded},
            {"model_id": "P ~ β₀·SST + β₁·column RH",
             "description": "SST + column RH, monthly",
             "time_scale": "monthly", "variable": "precip",
             "sst": sst_detrended,
             "precip": observed_precip_constrained_detrended_rh,
             "predictor": rh_column_detrended_expanded},
        ]

    return models


# ---------------------------------------------------------------------------
# Cross-validation runner
# ---------------------------------------------------------------------------

def process_model_cv(model_dict: dict) -> tuple:
    """Run cross-validation for a single model dict. Returns (model_id, cv_ds)."""
    model_id  = model_dict["model_id"]
    predictor = model_dict.get("predictor", None)

    sst = model_dict["sst"]
    precip = model_dict["precip"]

    if predictor is None:
        sst, precip = xr.align(sst, precip, join="inner")
    else:
        sst, precip, predictor = xr.align(sst, precip, predictor, join="inner")

    apply_kwargs = dict(
        kwargs={"model_id": model_id},
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float] * 10,
    )
    

    if model_id in SST_ONLY_MODEL_IDS:
        cv_output = xr.apply_ufunc(
            cv_regression_model,
            sst,
            precip,
            input_core_dims=[["time"], ["time"]],
            output_core_dims=[[], [], [], [], [], [], [], [], [], []],
            **apply_kwargs,
        )
    else:
        cv_output = xr.apply_ufunc(
            cv_regression_model_with_predictor,
            sst,
            precip,
            predictor,
            input_core_dims=[["time"], ["time"], ["time"]],
            output_core_dims=[[], [], [], [], [], [], [], [], [], []],
            **apply_kwargs,
        )

    cv_ds = xr.Dataset(
        {
            "rmse_mean":   cv_output[0],
            "rmse_std":    cv_output[1],
            "mae_mean":    cv_output[2],
            "mae_std":     cv_output[3],
            "nrmse_mean":  cv_output[4],
            "nrmse_std":   cv_output[5],
            "coef_mean":   cv_output[6],
            "coef_std":    cv_output[7],
            "adjr2_mean":  cv_output[8],
            "adjr2_std":   cv_output[9],
        },
        attrs={
            "model_id":    model_dict["model_id"],
            "description": model_dict["description"],
            "time_scale":  model_dict["time_scale"],
            "variable":    model_dict["variable"],
        },
    )
    return model_id, cv_ds


# ---------------------------------------------------------------------------
# Predictor preparation
# ---------------------------------------------------------------------------

def _load_olr(common_time, sst_detrended):
    """Load, constrain, detrend, and regrid OLR from NCEP-DOE R2."""
    try:
        olr = xr.open_dataset(OLR_DIR / "ulwrf.ntat.mon.mean.nc").drop_dims("nbnds")
        common_time = np.intersect1d(common_time, olr.time.values)
        olr_anom = (
            olr.sel(time=common_time)
               .groupby("time.month")
               .__sub__(olr.sel(time=common_time).groupby("time.month").mean("time"))
        )
        olr_detrended = detrend_dim(olr_anom.ulwrf, "time")
        return olr_detrended.interp(lat=sst_detrended.lat, lon=sst_detrended.lon)
    except Exception as e:
        print(f"  Warning: Could not load OLR: {e}")
        return None


def _load_lapse_rate(common_time, sst_detrended):
    """Compute 850-500 hPa lapse rate anomaly from NCEP-DOE R2, regrid to SST grid."""
    try:
        hgt  = xr.open_dataset(NCEP_DIR / "hgt.mon.mean.nc")
        temp = xr.open_dataset(NCEP_DIR / "air.mon.mean.nc")

        lapse_rate = (
            (temp["air"].sel(level=850) - temp["air"].sel(level=500))
            / ((hgt["hgt"].sel(level=500) - hgt["hgt"].sel(level=850)) / 1000)
        )
        common_time = np.intersect1d(common_time, lapse_rate.time.values)
        lr_constrained = lapse_rate.sel(time=common_time)
        lr_anom = (
            lr_constrained.groupby("time.month")
            - lr_constrained.groupby("time.month").mean("time")
        )
        lr_detrended = detrend_dim(lr_anom, "time")
        return (
            lr_detrended
            .interp(lat=sst_detrended.lat, lon=sst_detrended.lon)
            .transpose("lat", "lon", "time")
        )
    except Exception as e:
        print(f"  Warning: Could not load lapse rate: {e}")
        return None


def _load_gmst(common_time, sst_detrended):
    """Load Berkeley Earth GMST, compute anomaly, detrend, expand to SST grid."""
    try:
        gmst    = pd.read_csv(GMST_FILE)
        gmst_xr = xr.DataArray(
            gmst["t_abs"].values,
            coords=[pd.to_datetime(gmst["time"])],
            dims="time",
            name="gmst",
        )
        common_time = np.intersect1d(common_time, gmst_xr.time.values)
        gmst_xr = gmst_xr.sel(time=common_time)
        gmst_anom     = gmst_xr.groupby("time.month") - gmst_xr.groupby("time.month").mean("time")
        gmst_detrended = detrend_dim(gmst_anom, "time")
        return gmst_detrended.expand_dims(
            lat=sst_detrended.lat, lon=sst_detrended.lon
        ).transpose("time", "lat", "lon")
    except Exception as e:
        print(f"  Warning: Could not load GMST: {e}")
        return None


def _load_rh(common_time, p_detrended, sst_anom):
    """Load MERRA-2 RH, compute z500 and column-integrated basin anomalies."""
    rh_z500, rh_col, precip_rh = None, None, None
    try:
        files = sorted(MERRA2_RH_DIR.glob("*.nc4"))
        if not files:
            print(f"  Warning: No RH files found in {MERRA2_RH_DIR}")
            return None, None, None

        rh = xr.open_mfdataset([str(f) for f in files], combine="by_coords")

        # --- 500 hPa ---
        z500_basin = waterbasin(rh.RH.sel(lev=500))
        common_basin = np.intersect1d(p_detrended["basin"], z500_basin["basin"])
        common_time = np.intersect1d(common_time, z500_basin.time.values)
        rh_c = z500_basin.sel(time=common_time, basin=common_basin)
        rh_anom = rh_c.groupby("time.month") - rh_c.groupby("time.month").mean("time")
        rh_z500 = (
            detrend_dim(rh_anom, "time")
            .expand_dims(lat=sst_anom.lat, lon=sst_anom.lon)
            .load()
        )
        precip_rh = p_detrended.sel(basin=common_basin)

        # --- Column-integrated (pressure-weighted) ---
        pressure = rh.lev.values
        dp       = np.gradient(pressure)
        weights  = xr.DataArray(dp / dp.sum(), coords={"lev": rh.lev}, dims="lev")
        col_basin = waterbasin((rh.RH * weights).sum(dim="lev"))
        common_basin = np.intersect1d(p_detrended["basin"], col_basin["basin"])
        rh_c  = col_basin.sel(time=common_time, basin=common_basin)
        rh_anom = rh_c.groupby("time.month") - rh_c.groupby("time.month").mean("time")
        rh_col = (
            detrend_dim(rh_anom, "time")
            .expand_dims(lat=sst_anom.lat, lon=sst_anom.lon)
            .load()
        )

    except Exception as e:
        print(f"  Warning: Could not load relative humidity: {e}")

    return rh_z500, rh_col, precip_rh


# ---------------------------------------------------------------------------
# Per-pair processing
# ---------------------------------------------------------------------------

def process_ensemble_pair(
    p_name: str,
    p_da: xr.DataArray,
    sst_name: str,
    sst_da: xr.DataArray,
    sst_raw: xr.DataArray,
    n_jobs: int = 16,
) -> dict:
    print(f"\n{'='*80}")
    print(f"PROCESSING: {p_name} vs {sst_name}")
    print(f"{'='*80}")

    # Align to common time
    common_time = np.intersect1d(p_da.load()["time"].values, sst_da["time"].values)
    p_anom      = p_da.sel(time=common_time)
    sst_anom    = sst_da.sel(time=common_time)
    sst_raw     = sst_raw.sel(time=common_time)

    # Detrend
    print("  Detrending...")
    p_detrended   = detrend_dim(p_anom,   "time").astype(np.float32)
    sst_detrended = detrend_dim(sst_anom, "time").astype(np.float32)

    # Load conditioning variables
    print("  Loading OLR...")
    olr_regridded = _load_olr(common_time, sst_detrended)

    print("  Loading lapse rate...")
    lapse_rate_regridded = _load_lapse_rate(common_time, sst_detrended)

    print("  Loading GMST...")
    gmst_detrended_expanded = _load_gmst(common_time, sst_detrended)

    print("  Loading relative humidity...")
    rh_z500, rh_col, precip_rh = _load_rh(common_time, p_detrended, sst_anom)

    # Build and run models
    models_to_run = get_models_to_run(
        sst_detrended, p_detrended, sst_raw, p_anom,
        olr_regridded=olr_regridded,
        lapse_rate_regridded=lapse_rate_regridded,
        gmst_detrended_expanded=gmst_detrended_expanded,
        rh_z500_detrended_expanded=rh_z500,
        rh_column_detrended_expanded=rh_col,
        observed_precip_constrained_detrended_rh=precip_rh,
    )

    print(f"  Running {len(models_to_run)} models in parallel (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_model_cv)(m) for m in models_to_run
    )

    return {model_id: cv_ds for model_id, cv_ds in results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-validation for observational ensemble")
    parser.add_argument("--pair-index", type=int, default=None,
                        help="Index of precipitation-SST pair to process (0-based). "
                             "If omitted, all pairs are processed sequentially.")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Number of parallel jobs for model processing")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION ANALYSIS — OBSERVATIONAL ENSEMBLE")
    print("=" * 80)

    # Load preprocessed data
    print("\nLoading preprocessed datasets...")
    precip_dict  = {}
    sst_dict     = {}
    sst_raw_dict = {}

    for p_name in PRECIP_NAMES:
        path = OUTPUTS_DIR / f"precip_anom_{p_name}.nc"
        if path.exists():
            precip_dict[p_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found")

    for sst_name in SST_NAMES:
        path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if path.exists():
            sst_dict[sst_name] = xr.open_dataarray(path).squeeze()
        else:
            print(f"  Warning: {path} not found")

        path = OUTPUTS_DIR / f"sst_raw_{sst_name}.nc"
        if path.exists():
            sst_raw_dict[sst_name] = xr.open_dataarray(path)
        else:
            print(f"  Warning: {path} not found")

    print(f"  Loaded {len(precip_dict)} precipitation datasets")
    print(f"  Loaded {len(sst_dict)} SST anomaly datasets")
    print(f"  Loaded {len(sst_raw_dict)} raw SST datasets")

    pairs = [
        (p_name, p_da, sst_name, sst_da, sst_raw_dict[sst_name])
        for p_name, p_da in precip_dict.items()
        for sst_name, sst_da in sst_dict.items()
        if sst_name in sst_raw_dict
    ]

    if not pairs:
        print("ERROR: No valid precipitation-SST pairs found.")
        sys.exit(1)

    print(f"\n  Total pairs to process: {len(pairs)}")

    def _save_pair(cv_results: dict, p_name: str, sst_name: str) -> None:
        out = OUTPUTS_DIR / f"cv_results_{p_name}_{sst_name}.nc"
        (
            xr.concat(cv_results.values(), dim="model", coords="minimal")
              .assign_coords(model=list(cv_results.keys()))
              .to_netcdf(out)
        )
        print(f"  Saved CV results → {out}")

    if args.pair_index is not None:
        if args.pair_index >= len(pairs):
            print(f"ERROR: pair-index {args.pair_index} >= total pairs {len(pairs)}")
            sys.exit(1)
        p_name, p_da, sst_name, sst_da, sst_raw = pairs[args.pair_index]
        print(f"\nProcessing pair {args.pair_index}/{len(pairs)-1}: {p_name} vs {sst_name}")
        cv_results = process_ensemble_pair(p_name, p_da, sst_name, sst_da, sst_raw, args.n_jobs)
        _save_pair(cv_results, p_name, sst_name)

    else:
        for idx, (p_name, p_da, sst_name, sst_da, sst_raw) in enumerate(pairs):
            print(f"\nProcessing pair {idx+1}/{len(pairs)}: {p_name} vs {sst_name}")
            cv_results = process_ensemble_pair(p_name, p_da, sst_name, sst_da, sst_raw, args.n_jobs)
            _save_pair(cv_results, p_name, sst_name)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()