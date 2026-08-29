#!/usr/bin/env python
# coding: utf-8
"""
Reconstruction cross-validation: Regression vs PCA.

Author: Flora Perlmutter

Description
-----------
For each (precip_dataset, SST_dataset) pair, runs leave-one-year-out
cross-validation comparing the reconstruction skill of two methods:

  1. Linear regression:  P̂(t) = Σ sensitivity_reg(lat,lon) · SST(lat,lon,t)
     where sensitivity_reg = grid_area · β · FDR_mask, refit each fold.

  2. PCA regression:     P̂(t) = Σ βᵢ · PCᵢ(t)
     where EOFs and coefficients are refit each fold, optimal k selected
     via 1-SE rule on an inner LOO within the training fold.

Both methods produce a single predicted precipitation time series per basin.
Skill metrics (RE, correlation, RMSE, NRMSE, MAE) are computed on the
LOO predictions and saved per basin.

"""
import argparse
import gc
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import signal
from scipy.stats import pearsonr
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import CMIG_DATA, DATA_DIR
from regression_functions import (
    detrend_dim,
    grid_area,
    fdr_correction,
    regression_slope_se,
)

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUTS_DIR  = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"
OUTPUTS_DIR = DATA_DIR

PRECIP_DATASETS = [
    "GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN",
]
SST_DATASETS = ["ERSSTv6", "COBE-SST3"]

ALPHA                    = 0.05   # FDR / Bonferroni significance threshold
K_MAX                    = 15     # maximum PCs to test in PCA inner CV
K_FALLBACK               = 4      # fallback when PCA CV fails
MIN_YEARS                = 20     # skip basins with fewer valid years
SKIP_EXISTING            = True
PRECIP_VAR_MIN           = 1e-12  # minimum precip variance to process basin
K_FALLBACK_WARN_FRAC     = 0.05   # warn if > 5 % of basins use fallback k
BASIN_ID_TOL             = 1e-6   # tolerance for float basin-ID matching


# =============================================================================
# PAIR ENUMERATION
# =============================================================================
def get_all_pairs() -> list[tuple[str, str]]:
    """Return ordered list of (precip_name, sst_name) pairs."""
    return list(product(PRECIP_DATASETS, SST_DATASETS))


# =============================================================================
# DETRENDING (numpy, for use inside CV folds)
# =============================================================================
def detrend_along_time(arr: np.ndarray) -> np.ndarray:
    """Linear detrend along axis=0 for a 1-D or 2-D numpy array."""
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        if not np.any(np.isfinite(arr)):
            return arr
        return signal.detrend(arr, axis=0).astype(np.float32)
    out = np.empty_like(arr)
    finite_cols = np.all(np.isfinite(arr), axis=0)
    out[:, ~finite_cols] = np.nan
    if finite_cols.any():
        out[:, finite_cols] = signal.detrend(
            arr[:, finite_cols], axis=0
        ).astype(np.float32)
    return out


def detrend_using_training(
    trn_sst: np.ndarray,
    trn_years: np.ndarray,
    tst_sst: np.ndarray,
    tst_years: np.ndarray,
) -> np.ndarray:
    """
    Detrend test SST using the linear trend estimated on the training fold.

    This prevents test-fold detrending from being computed on a single year
    (where the trend is undefined), and ensures both branches receive test
    data that has been preprocessed with the same transformation as the
    training data.

    Parameters
    ----------
    trn_sst   : (n_train, n_grid)  raw training SST anomalies
    trn_years : (n_train,)         integer years for training times
    tst_sst   : (n_test, n_grid)   raw test SST anomalies
    tst_years : (n_test,)          integer years for test times

    Returns
    -------
    tst_sst_detrended : (n_test, n_grid)  test SST with training trend removed
    """
    X_trn = np.column_stack([
        np.ones(len(trn_years), dtype=np.float32),
        trn_years.astype(np.float32),
    ])
    # coeffs shape: (2, n_grid)  — row 0: intercept, row 1: slope
    coeffs, _, _, _ = np.linalg.lstsq(X_trn, trn_sst, rcond=None)

    X_tst = np.column_stack([
        np.ones(len(tst_years), dtype=np.float32),
        tst_years.astype(np.float32),
    ])
    trend_tst = X_tst @ coeffs  # (n_test, n_grid)

    return (tst_sst - trend_tst).astype(np.float32)


# =============================================================================
# SST UTILITIES
# =============================================================================
def sst_area_weights(lat: np.ndarray, n_lon: int) -> np.ndarray:
    """sqrt(cos(lat)) weights for PCA, shape (n_lat, n_lon)."""
    w = np.cos(np.deg2rad(lat))
    w = np.where(w > 0, np.sqrt(w), 0.0)
    w2d = np.broadcast_to(w[:, None], (len(lat), n_lon))
    return np.ascontiguousarray(w2d, dtype=np.float32)


def flatten_sst(sst_vals: np.ndarray, lat: np.ndarray, n_lon: int):
    """
    (n_time, n_lat, n_lon) → (n_time, n_ocean), plus mask and weights.

    Returns
    -------
    flat_vals  : (n_time, n_ocean)
    w_flat     : (n_ocean,)
    ocean_mask : (n_lat, n_lon) bool
    """
    w2d = sst_area_weights(lat, n_lon)
    ocean = np.all(np.isfinite(sst_vals), axis=0)
    return (
        sst_vals[:, ocean].astype(np.float32),
        w2d[ocean].astype(np.float32),
        ocean,
    )


# =============================================================================
# PCA HELPERS
# =============================================================================
def pc_regression_numpy(PCs, p_1d, alpha=ALPHA):
    """
    OLS: p ~ intercept + PC_1 + ... + PC_k (numpy lstsq).
    Bonferroni-corrected significance over k tests.

    Returns: betas (k,), pvals (k,), sig_mask (k,), r2 float
    """
    from scipy.stats import t as t_dist

    valid = ~np.isnan(p_1d) & np.all(np.isfinite(PCs), axis=1)
    k = PCs.shape[1]
    empty = (
        np.zeros(k, dtype=np.float32),
        np.ones(k, dtype=np.float32),
        np.zeros(k, dtype=bool),
        0.0,
    )
    if valid.sum() < k + 2:
        return empty

    X = np.column_stack([np.ones(valid.sum(), dtype=np.float32), PCs[valid]])
    y = p_1d[valid].astype(np.float32)
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    betas = coeffs[1:].astype(np.float32)
    y_hat = X @ coeffs
    resid = y - y_hat
    dof = len(y) - k - 1
    if dof < 1:
        return empty

    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sigma2 = ss_res / dof

    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return empty

    se = np.sqrt(np.maximum(np.diag(XtXinv)[1:] * sigma2, 0.0)).astype(np.float32)
    t_stat = np.where(se > 0, betas / se, 0.0)
    pvals = (2 * t_dist.sf(np.abs(t_stat), dof)).astype(np.float32)
    sig_mask = pvals <= (alpha / k)

    return betas, pvals, sig_mask, r2


def pca_predict_fold(
    sst_train_flat: np.ndarray,
    sst_test_flat: np.ndarray,
    weights: np.ndarray,
    p_train: np.ndarray,
    k: int,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Fit PCA regression on training data, predict on test data.

    Parameters
    ----------
    sst_train_flat : (n_train, n_ocean)  detrended training SST
    sst_test_flat  : (n_test, n_ocean)   test SST detrended with training trend
    weights        : (n_ocean,)
    p_train        : (n_train,)          detrended training precip
    k              : number of PCs

    Returns
    -------
    p_hat_test : (n_test,)
    """
    # Truncated SVD on training fold
    X_w_train = sst_train_flat * weights[None, :]
    _, s_tr, Vt_tr = randomized_svd(
        X_w_train, n_components=k, n_iter=4, random_state=0
    )
    safe_w = np.where(weights > 0, weights, 1.0)
    EOFs_tr = (Vt_tr / safe_w[None, :]).astype(np.float32)

    # Project onto training EOFs
    def project(sst_2d, EOFs):
        X_w = sst_2d * weights[None, :]
        return (X_w @ (EOFs * weights[None, :]).T).astype(np.float32)

    PCs_train = project(sst_train_flat, EOFs_tr)
    PCs_test  = project(sst_test_flat,  EOFs_tr)

    # Regress and predict
    betas, _, sig_mask, _ = pc_regression_numpy(PCs_train[:, :k], p_train, alpha)
    b = np.where(sig_mask, betas, 0.0)
    return (PCs_test[:, :k] @ b).astype(np.float32)


def select_k_inner_cv(
    sst_flat_det: np.ndarray,
    weights: np.ndarray,
    p_det: np.ndarray,
    years: np.ndarray,
    alpha: float = ALPHA,
    k_max: int = K_MAX,
) -> int:
    """
    Inner LOO CV to select optimal k via 1-SE rule.
    Recomputes SVD per fold (no leakage).
    Taken directly from your PCA script's cross_validate_k.

    Returns
    -------
    optimal_k : int
    """
    unique_years = np.unique(years)
    p_var = np.nanvar(p_det)
    if p_var == 0:
        return K_FALLBACK

    k_max_eff = min(k_max, len(p_det) - 2)
    if k_max_eff < 1:
        return K_FALLBACK
    
    p_hat_all = np.full((k_max_eff, len(p_det)), np.nan, dtype=np.float32)

    for yr in unique_years:
        test = (years == yr)
        train = ~test
        if train.sum() < k_max_eff + 2:
            continue

        sst_train = sst_flat_det[train, :]
        sst_test = sst_flat_det[test, :]

        _, s_tr, Vt_tr = randomized_svd(
            sst_train * weights[None, :],
            n_components=k_max_eff,
            n_iter=4,
            random_state=0,
        )
        safe_w = np.where(weights > 0, weights, 1.0)
        EOFs_tr = (Vt_tr / safe_w[None, :]).astype(np.float32)

        def project(sst_2d, EOFs):
            X_w = sst_2d * weights[None, :]
            return (X_w @ (EOFs * weights[None, :]).T).astype(np.float32)

        PCs_train = project(sst_train, EOFs_tr)
        PCs_test = project(sst_test, EOFs_tr)

        for k in range(1, k_max_eff + 1):
            betas, _, sig, _ = pc_regression_numpy(
                PCs_train[:, :k], p_det[train], alpha
            )
            b = np.where(sig, betas, 0.0)
            p_hat_all[k - 1, test] = (PCs_test[:, :k] @ b).astype(np.float32)

    # 1-SE rule
    cv_re = np.full(k_max_eff, np.nan, dtype=np.float32)
    for k in range(1, k_max_eff + 1):
        p_hat = p_hat_all[k - 1]
        valid = np.isfinite(p_hat) & np.isfinite(p_det)
        if valid.sum() < 3:
            continue
        mse = np.mean((p_hat[valid] - p_det[valid]) ** 2)
        cv_re[k - 1] = float(1.0 - mse / p_var)

    finite_mask = np.isfinite(cv_re)
    if not finite_mask.any():
        return K_FALLBACK

    best_re = float(np.max(cv_re[finite_mask]))
    se_re = float(np.std(cv_re[finite_mask]) / np.sqrt(finite_mask.sum()))
    threshold = best_re - se_re

    for k in range(1, k_max_eff + 1):
        if np.isfinite(cv_re[k - 1]) and cv_re[k - 1] >= threshold:
            return k
    return K_FALLBACK

# =============================================================================
# REGRESSION RECONSTRUCTION — PER FOLD
# =============================================================================
def regression_predict_fold(
    sst_anom_train: xr.DataArray,
    sst_anom_test: xr.DataArray,
    p_train: xr.DataArray,
    area: xr.DataArray,
    alpha: float = ALPHA,
) -> np.ndarray:
    """
    Fit regression on training data, reconstruct on test data.

    Steps:
        1. Detrend training SST and precip
        2. regression_slope_se at every grid cell
        3. FDR correction
        4. Apply area-weighted significant slopes to test SST anomalies
           to produce reconstruction

    Parameters
    ----------
    sst_anom_train : (time_train, lat, lon)  raw anomalies (will be detrended)
    sst_anom_test  : (time_test, lat, lon)   raw anomalies (used as-is for reconstruction)
    p_train        : (time_train, basin)     raw anomalies (will be detrended)
    area           : (lat, lon)              grid cell fractional areas
    alpha          : FDR threshold

    Returns
    -------
    p_hat_test : (n_test, n_basins) float32
    """
    # Detrend training data
    sst_det_train = detrend_dim(sst_anom_train, "time").astype(np.float32)
    p_det_train = detrend_dim(p_train, "time").astype(np.float32)

    slope, _, pval = xr.apply_ufunc(
        regression_slope_se,
        sst_det_train,
        p_det_train,
        input_core_dims=[["time"], ["time"]],
        vectorize=True,
        output_core_dims=[[], [], []],
        output_dtypes=[np.float32, np.float32, np.float32],
    )
    
    # Re-attach spatial + basin coords that apply_ufunc may have dropped
    coords = {"lat": sst_det_train.lat, "lon": sst_det_train.lon}
    if "basin" in p_det_train.coords:
        coords["basin"] = p_det_train.basin

    slope = slope.assign_coords(coords)
    pval  = pval.assign_coords(coords)
    
    if "basin" in p_train.dims:
        slope_area = (area * slope).transpose("lat", "lon", "basin")
        pval       = pval.transpose("lat", "lon", "basin")
    else:
        slope_area = (area * slope).transpose("lat", "lon")
        pval       = pval.transpose("lat", "lon")

    # The test family is the SST grid cells within this basin — the same set the
    # reconstruction sums over below. Works with or without a basin dimension, so
    # no dummy axis is needed.
    #
    # This previously expanded a length-1 basin axis and corrected over it, which
    # made N=1 and reduced BH to an uncorrected p <= alpha threshold.
    fdr_mask = fdr_correction(pval, alpha_FDR=alpha, core_dims=("lat", "lon"))


    slope_sig  = slope_area.where(fdr_mask, 0.0)
    recon_test = (sst_anom_test * slope_sig).sum(("lat", "lon"))

    return recon_test.values.astype(np.float32)


# =============================================================================
# SKILL METRICS
# =============================================================================
def compute_skill_metrics(
    p_obs: np.ndarray,
    p_hat: np.ndarray,
) -> dict:
    """
    Compute reconstruction skill metrics.

    Parameters
    ----------
    p_obs : (n_time,) observed precipitation (detrended anomalies)
    p_hat : (n_time,) LOO predictions

    Returns
    -------
    dict with: re, correlation, correlation_pval, rmse, nrmse, mae
    """
    valid = np.isfinite(p_obs) & np.isfinite(p_hat)
    if valid.sum() < 3:
        return {
            "re": np.nan, "correlation": np.nan,
            "correlation_pval": np.nan, "rmse": np.nan,
            "nrmse": np.nan, "mae": np.nan,
        }

    obs = p_obs[valid]
    hat = p_hat[valid]

    mse     = np.mean((obs - hat) ** 2)
    obs_var = np.var(obs)

    # Guard against zero variance
    re = np.nan if obs_var < PRECIP_VAR_MIN else 1.0 - mse / obs_var

    r, p    = pearsonr(obs, hat)
    rmse    = np.sqrt(mse)
    std_obs = np.std(obs)
    nrmse   = np.nan if std_obs < PRECIP_VAR_MIN ** 0.5 else rmse / std_obs
    mae     = np.mean(np.abs(obs - hat))

    return {
        "re": float(re),
        "correlation": float(r),
        "correlation_pval": float(p),
        "rmse": float(rmse),
        "nrmse": float(nrmse),
        "mae": float(mae),
    }


# =============================================================================
# PER-BASIN LOO CROSS-VALIDATION
# =============================================================================
def cv_basin(
    basin_idx: int,
    basin_id: float,
    sst_anom_vals: np.ndarray,
    sst_lat: np.ndarray,
    sst_lon: np.ndarray,
    sst_times: np.ndarray,
    p_anom_vals: np.ndarray,
    p_times: np.ndarray,
    area_vals: np.ndarray,
    area_lat: np.ndarray,
    area_lon: np.ndarray,
) -> dict | None:
    """
    Run LOO CV for both methods on a single basin.

    Returns dict with LOO predictions and skill metrics for both methods,
    or None if the basin is skipped.
    """
    # ---- Align times ----
    common_time = np.intersect1d(p_times, sst_times)
    if len(common_time) < MIN_YEARS:
        return None

    idx_p   = np.isin(p_times,   common_time)
    idx_sst = np.isin(sst_times, common_time)

    p_1d       = p_anom_vals[idx_p, basin_idx].astype(np.float32)
    sst_common = sst_anom_vals[idx_sst]

    if not np.any(np.isfinite(p_1d)):
        return None

    # Guard against zero-variance precip
    if np.nanvar(p_1d) < PRECIP_VAR_MIN:
        print(f"    WARNING: Basin {basin_id:.0f} has near-zero precip variance — skipping")
        return None

    years        = common_time.astype("datetime64[Y]").astype(int) + 1970
    unique_years = np.unique(years)
    n_time       = len(common_time)

    # ---- Flatten SST for PCA ----
    sst_flat, w_flat, ocean_mask = flatten_sst(sst_common, sst_lat, len(sst_lon))

    # ---- Detrend full precip record ONCE, outside the fold loop ----
    p_det_full = detrend_along_time(p_1d)

    # ---- LOO storage ----
    p_hat_reg = np.full(n_time, np.nan, dtype=np.float32)
    p_hat_pca = np.full(n_time, np.nan, dtype=np.float32)

    # ---- Build xarray objects once for regression fold construction ----
    sst_anom_xr = xr.DataArray(
        sst_common,
        dims=["time", "lat", "lon"],
        coords={"time": common_time, "lat": sst_lat, "lon": sst_lon},
    )
    area_xr = xr.DataArray(
        area_vals,
        dims=["lat", "lon"],
        coords={"lat": area_lat, "lon": area_lon},
    )

    for yr in unique_years:
        test_mask  = years == yr
        train_mask = ~test_mask
        n_train    = train_mask.sum()

        if n_train < MIN_YEARS:
            continue

        test_times  = common_time[test_mask]
        train_times = common_time[train_mask]

        # ---- Raw SST blocks (needed by detrend_using_training) ----
        sst_flat_train = sst_flat[train_mask]
        sst_flat_test  = sst_flat[test_mask]

        # Detrend training SST within fold; detrend test SST using the trend from the training data
        sst_flat_det_train = detrend_along_time(sst_flat_train)
        sst_flat_det_test  = detrend_using_training(
            sst_flat_train,
            years[train_mask],
            sst_flat_test,
            years[test_mask],
        )
        # Use the already-detrended precip slices
        p_train_det = p_det_full[train_mask]

        # ==============================================================
        # REGRESSION FOLD
        # Pass detrended test SST (same as PCA)
        # ==============================================================
        sst_train_xr = sst_anom_xr.sel(time=train_times)

        # Reconstruct detrended test SST as xarray for regression_predict_fold
        n_test = test_mask.sum()
        sst_test_det_3d = np.zeros((n_test, len(sst_lat), len(sst_lon)), dtype=np.float32)
        sst_test_det_3d[:, ocean_mask] = sst_flat_det_test
        
        sst_test_xr = xr.DataArray(
            sst_test_det_3d,
            dims=["time", "lat", "lon"],
            coords={"time": test_times, "lat": sst_lat, "lon": sst_lon},
        )
        
        p_train_xr = xr.DataArray(
            p_train_det,
            dims=["time"],
            coords={"time": train_times},
        )

        p_hat_reg[test_mask] = regression_predict_fold(
            sst_train_xr, sst_test_xr, p_train_xr, area_xr, ALPHA
        )

        # ==============================================================
        # PCA FOLD
        # ==============================================================
        
        # Select k via inner LOO on training fold
        years_train = years[train_mask]
        optimal_k = select_k_inner_cv(
            sst_flat_det_train, w_flat, p_train_det,
            years_train, ALPHA, K_MAX,
        )
    
        p_hat_pca[test_mask] = pca_predict_fold(
            sst_flat_det_train,
            sst_flat_det_test,   # detrended with training trend
            w_flat,
            p_train_det,         # consistent detrended precip
            optimal_k,
            ALPHA,
        )

        # Free fold-local temporaries to avoid memory accumulation
        del sst_flat_train, sst_flat_test, sst_flat_det_train, sst_flat_det_test
        del sst_test_det_3d, sst_test_xr, p_train_xr, p_train_det
        gc.collect()

    # ---- Compute metrics (both methods compared to same p_det_full) ----
    reg_metrics = compute_skill_metrics(p_det_full, p_hat_reg)
    pca_metrics = compute_skill_metrics(p_det_full, p_hat_pca)

    return {
        "basin_id":   basin_id,
        "p_hat_reg":  p_hat_reg,
        "p_hat_pca":  p_hat_pca,
        "p_obs":      p_det_full,
        "time":       common_time,
        "reg_metrics": reg_metrics,
        "pca_metrics": pca_metrics,
    }


# =============================================================================
# BUILD OUTPUT DATASET
# =============================================================================
def build_output_dataset(
    basin_results: dict,
    all_times: np.ndarray,
    p_name: str,
    sst_name: str,
) -> xr.Dataset:
    """Pack per-basin results into an xr.Dataset."""
    basin_ids    = sorted(basin_results.keys())
    n_basins     = len(basin_ids)
    n_time       = len(all_times)
    metric_names = ["re", "correlation", "correlation_pval", "rmse", "nrmse", "mae"]
    method_names = ["regression", "pca"]

    # Prediction arrays
    p_hat_reg_arr = np.full((n_basins, n_time), np.nan, dtype=np.float32)
    p_hat_pca_arr = np.full((n_basins, n_time), np.nan, dtype=np.float32)
    p_obs_arr     = np.full((n_basins, n_time), np.nan, dtype=np.float32)

    # Metric arrays: (n_basins, n_methods)
    metrics_arr = {
        m: np.full((n_basins, 2), np.nan, dtype=np.float32)
        for m in metric_names
    }

    for i, bid in enumerate(basin_ids):
        res   = basin_results[bid]
        t_idx = np.isin(all_times, res["time"])

        p_hat_reg_arr[i, t_idx] = res["p_hat_reg"]
        p_hat_pca_arr[i, t_idx] = res["p_hat_pca"]
        p_obs_arr[i, t_idx]     = res["p_obs"]

        for j, method_key in enumerate(["reg_metrics", "pca_metrics"]):
            for m in metric_names:
                metrics_arr[m][i, j] = res[method_key][m]

    ds = xr.Dataset(
        {
            "p_hat_regression": xr.DataArray(
                p_hat_reg_arr, dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"long_name": "LOO reconstruction — linear regression"},
            ),
            "p_hat_pca": xr.DataArray(
                p_hat_pca_arr, dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"long_name": "LOO reconstruction — PCA regression"},
            ),
            "p_observed": xr.DataArray(
                p_obs_arr, dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"long_name": "Observed detrended precipitation anomaly"},
            ),
            **{
                metric: xr.DataArray(
                    metrics_arr[metric],
                    dims=["basin", "method"],
                    coords={"basin": basin_ids, "method": method_names},
                    attrs={"long_name": f"LOO CV {metric}"},
                )
                for metric in metric_names
            },
        },
        attrs={
            "description":       "LOO reconstruction cross-validation: regression vs PCA",
            "cv_method":         "Leave-one-year-out",
            "regression_method": "P ~ β*SST, area-weighted, FDR-corrected, refit per fold",
            "pca_method":        "P ~ Σ βᵢ·PCᵢ(SST), per-fold SVD, 1-SE rule for k",
            "precip_dataset":    p_name,
            "sst_dataset":       sst_name,
            "alpha":             ALPHA,
            "k_max":             K_MAX,
            "k_fallback":        K_FALLBACK,
            "min_years":         MIN_YEARS,
        },
    )
    return ds


# =============================================================================
# PROCESS ONE PAIR
# =============================================================================
def process_pair(p_name: str, sst_name: str) -> None:
    """Load data, run all basins, save output for one (precip, SST) pair."""
    out_path = OUTPUTS_DIR / f"cv_reconstruction_{p_name}_{sst_name}.nc"
    if SKIP_EXISTING and out_path.exists():
        print(f"  [EXISTS] {out_path.name} — skipping")
        return

    # --- Load SST anomalies ---
    sst_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if not sst_path.exists():
        print(f"  [SKIP] SST file not found: {sst_path}")
        return
    print(f"  Loading SST: {sst_name}")
    sst_anom = xr.open_dataarray(sst_path).squeeze().load()
    if "lev" in sst_anom.dims:
        sst_anom = sst_anom.squeeze("lev", drop=True)

    sst_lat       = sst_anom["lat"].values
    sst_lon       = sst_anom["lon"].values
    sst_times     = sst_anom["time"].values
    sst_anom_vals = sst_anom.values.astype(np.float32)

    # --- Precompute grid area ---
    area      = grid_area(sst_anom).astype(np.float32)
    area_vals = area.values
    area_lat  = area["lat"].values
    area_lon  = area["lon"].values

    # --- Load precip anomalies ---
    p_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
    if not p_path.exists():
        print(f"  [SKIP] Precip file not found: {p_path}")
        return
    print(f"  Loading precip: {p_name}")
    p_anom = xr.open_dataarray(p_path).load()

    basin_ids   = p_anom["basin"].values
    p_times     = p_anom["time"].values
    p_anom_vals = p_anom.values.astype(np.float32)
    all_times   = np.union1d(p_times, sst_times)

    print(f"  Basins: {len(basin_ids)}")
    print(f"  SST grid: {len(sst_lat)} x {len(sst_lon)}")
    print(f"  Time overlap: checking per basin")


    # --- Basin loop ---
    basin_results   = {}
    skipped         = 0

    for i, bid in enumerate(basin_ids):
        if i % 25 == 0:
            print(f"    Basin {i}/{len(basin_ids)} (id={bid}) ...")

        res = cv_basin(
            basin_idx=i,
            basin_id=float(bid),
            sst_anom_vals=sst_anom_vals,
            sst_lat=sst_lat,
            sst_lon=sst_lon,
            sst_times=sst_times,
            p_anom_vals=p_anom_vals,
            p_times=p_times,
            area_vals=area_vals,
            area_lat=area_lat,
            area_lon=area_lon,
        )

        if res is None:
            skipped += 1
        else:
            basin_results[float(bid)] = res

    print(f"  Completed: {len(basin_results)} basins  |  Skipped: {skipped}")

    if not basin_results:
        print("  No valid basins — skipping output.")
        return

    # --- Build and save ---
    ds_out   = build_output_dataset(basin_results, all_times, p_name, sst_name)
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f"  Saved → {out_path.name}")

    del sst_anom, p_anom, basin_results, ds_out
    gc.collect()


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="LOO reconstruction CV: regression vs PCA."
    )
    parser.add_argument(
        "--pair-index", type=int, default=None,
        help="Index of (precip, SST) pair to process (0-based).",
    )
    parser.add_argument(
        "--list-pairs", action="store_true",
        help="Print pair index table and exit.",
    )
    args = parser.parse_args()

    pairs = get_all_pairs()

    if args.list_pairs:
        print(f"\n{'Index':<8}{'Precip':<16}{'SST':<16}{'Output exists?'}")
        print("-" * 55)
        for i, (p, s) in enumerate(pairs):
            out   = OUTPUTS_DIR / f"cv_reconstruction_{p}_{s}.nc"
            exist = "YES" if out.exists() else "no"
            print(f"{i:<8}{p:<16}{s:<16}{exist}")
        return

    if args.pair_index is None:
        parser.error("--pair-index is required unless using --list-pairs")

    if args.pair_index >= len(pairs):
        print(f"ERROR: --pair-index {args.pair_index} >= total pairs {len(pairs)}")
        sys.exit(1)

    p_name, sst_name = pairs[args.pair_index]

    print("\n" + "=" * 80)
    print("RECONSTRUCTION CROSS-VALIDATION: REGRESSION vs PCA")
    print("=" * 80)
    print(f"Pair index   : {args.pair_index} / {len(pairs) - 1}")
    print(f"Precip       : {p_name}")
    print(f"SST          : {sst_name}")
    print(f"CV method    : Leave-one-year-out")
    print(f"Regression   : P ~ β*SST, area-weighted, FDR per fold")
    print(f"PCA          : per-fold SVD, inner LOO for k (1-SE rule)")
    print(f"K_MAX        : {K_MAX}")
    print(f"Alpha        : {ALPHA}")
    print("=" * 80)

    process_pair(p_name, sst_name)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()