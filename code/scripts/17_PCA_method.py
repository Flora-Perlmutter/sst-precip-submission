#!/usr/bin/env python
# coding: utf-8
"""
Global PCA Regression: all river basins × all precip datasets × all SST datasets.

Author: Flora Perlmutter

Description
-----------
For every (precip_dataset, SST_dataset, basin) combination:
  1. Load pre-computed anomalies, align times, detrend.
  2. Area-weighted PCA on global SST (sqrt-cosine weighting).
  3. Leave-one-year-out cross-validation to select optimal k.
     EOFs are recomputed on each training fold (no data leakage).
     Optimal k is selected via the 1-SE rule (simplest k within one
     standard error of the best RE), which prefers parsimony.
  4. Final PCA regression on full record at optimal k.
  5. Back-transform PC coefficients to a spatial sensitivity map.
  6. Store per-basin scalar results in a schema that mirrors
     global_linear_regression_bootstrap_{P}_{SST}.nc.

Required input files
--------------------
  precip_anom_{P}.nc       (load_and_process_obs.py)
  sst_anom_{SST}.nc        (load_and_process_obs.py)

Output files
------------
  global_pca_regression_{P}_{SST}.nc   — one file per (P, SST) pair
"""

import argparse
import gc
import os
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import signal
from scipy.stats import t as t_dist
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import CMIG_DATA, DATA_DIR
from regression_functions import detrend_dim, grid_area

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

ALPHA         = 0.05  # significance threshold (Bonferroni-corrected per basin)
K_MAX         = 15    # maximum PCs to test in CV
                      # Covers higher-order modes beyond ENSO while keeping
                      # per-fold SVD cost feasible. Run a sensitivity check
                      # on a basin subset with K_MAX=20 if needed.
K_FALLBACK    = 4     # used when CV fails or time series is too short
MIN_YEARS     = 20    # basins with fewer valid years are skipped
SKIP_EXISTING = True  # set False to recompute and overwrite existing output files


# =============================================================================
# PAIR ENUMERATION
# =============================================================================

def get_all_pairs() -> list[tuple[str, str]]:
    """Return ordered list of (precip_name, sst_name) pairs."""
    return list(product(PRECIP_DATASETS, SST_DATASETS))


# =============================================================================
# DETRENDING
# =============================================================================

def detrend_along_time(arr: np.ndarray) -> np.ndarray:
    """
    Linear detrend along axis=0 for a 1-D or 2-D numpy array.

    Parameters
    ----------
    arr : (n_time,) or (n_time, n_space)

    Returns
    -------
    detrended array, same shape, float32
    """
    arr = arr.astype(np.float32)
    if arr.ndim == 1:
        if not np.any(np.isfinite(arr)):
            return arr
        return signal.detrend(arr, axis=0).astype(np.float32)
    out         = np.empty_like(arr)
    finite_cols = np.all(np.isfinite(arr), axis=0)
    out[:, ~finite_cols] = np.nan
    if finite_cols.any():
        out[:, finite_cols] = signal.detrend(
            arr[:, finite_cols], axis=0
        ).astype(np.float32)
    return out


# =============================================================================
# SST FLATTENING / UNFLATTENING
# =============================================================================

def sst_area_weights(lat: np.ndarray, n_lon: int) -> np.ndarray:
    """sqrt(cos(lat)) weights, shape (n_lat, n_lon)."""
    w   = np.cos(np.deg2rad(lat))
    w   = np.where(w > 0, np.sqrt(w), 0.0)
    w2d = np.broadcast_to(w[:, None], (len(lat), n_lon))
    return np.ascontiguousarray(w2d, dtype=np.float32)


def flatten_sst(sst: xr.DataArray) -> tuple:
    """
    (time, lat, lon) → (time, n_ocean), plus land/sea mask and weight vector.

    Ocean mask = grid cells that are finite across ALL time steps.
    Computed once and reused for every basin so that unflatten()
    always produces a (lat, lon) array on the same grid.

    Returns
    -------
    flat_vals  : (n_time, n_ocean) float32
    w_flat     : (n_ocean,)        float32
    ocean_mask : (n_lat, n_lon)    bool
    """
    vals  = sst.values.astype(np.float32)
    lat   = sst["lat"].values
    n_lon = len(sst["lon"])
    w2d   = sst_area_weights(lat, n_lon)
    ocean = np.all(np.isfinite(vals), axis=0)
    return (
        vals[:, ocean].astype(np.float32),
        w2d[ocean].astype(np.float32),
        ocean,
    )


def unflatten(flat: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """(n_ocean,) → (n_lat, n_lon), NaN on land."""
    full             = np.full(ocean_mask.shape, np.nan, dtype=np.float32)
    full[ocean_mask] = flat
    return full


# =============================================================================
# PCA — truncated SVD
# =============================================================================

def compute_pca_truncated(
    sst_2d  : np.ndarray,
    weights : np.ndarray,
    k       : int,
    n_iter  : int = 4,
) -> tuple:
    """
    Area-weighted PCA via randomized truncated SVD (sklearn).

    Only computes the top-k components, which is much faster than full
    SVD when k << min(n_time, n_ocean).  n_iter=4 power iterations is
    sufficient accuracy for smooth SST fields.

    Parameters
    ----------
    sst_2d  : (n_time, n_ocean)  already detrended
    weights : (n_ocean,)
    k       : number of leading PCs to retain
    n_iter  : power iteration steps for randomized SVD accuracy

    Returns
    -------
    PCs     : (n_time, k)
    EOFs    : (k, n_ocean)   physical units (not weighted)
    var_exp : (k,)           fraction of total weighted variance
    """
    X_w      = sst_2d * weights[None, :]
    U, s, Vt = randomized_svd(X_w, n_components=k, n_iter=n_iter, random_state=0)
    PCs      = (U * s[None, :]).astype(np.float32)
    safe_w   = np.where(weights > 0, weights, 1.0)
    EOFs     = (Vt / safe_w[None, :]).astype(np.float32)
    var_exp  = (s ** 2 / np.sum(X_w ** 2)).astype(np.float32)
    return PCs, EOFs, var_exp


# =============================================================================
# OLS — numpy-based (replaces statsmodels)
# =============================================================================

def pc_regression_numpy(
    PCs  : np.ndarray,
    p_1d : np.ndarray,
    alpha: float = ALPHA,
) -> tuple:
    """
    OLS via numpy lstsq: p ~ intercept + PC_1 + ... + PC_k.
    Bonferroni-corrected significance over k tests.

    Replaces sm.OLS for speed; eliminates statsmodels Python overhead
    in the inner CV loop.

    Returns
    -------
    betas    : (k,)
    pvals    : (k,)
    sig_mask : (k,) bool
    r2       : float
    """
    valid = ~np.isnan(p_1d) & np.all(np.isfinite(PCs), axis=1)
    k     = PCs.shape[1]
    empty = (
        np.zeros(k, dtype=np.float32),
        np.ones(k,  dtype=np.float32),
        np.zeros(k, dtype=bool),
        0.0,
    )
    if valid.sum() < k + 2:
        return empty

    X = np.column_stack([np.ones(valid.sum(), dtype=np.float32), PCs[valid]])
    y = p_1d[valid].astype(np.float32)

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    betas  = coeffs[1:].astype(np.float32)
    y_hat  = X @ coeffs
    resid  = y - y_hat
    dof    = len(y) - k - 1

    if dof < 1:
        return empty

    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    sigma2 = ss_res / dof
    try:
        XtXinv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return empty

    se       = np.sqrt(np.maximum(np.diag(XtXinv)[1:] * sigma2, 0.0)).astype(np.float32)
    t_stat   = np.where(se > 0, betas / se, 0.0)
    pvals    = (2 * t_dist.sf(np.abs(t_stat), dof)).astype(np.float32)
    sig_mask = pvals <= (alpha / k)

    return betas, pvals, sig_mask, r2


# =============================================================================
# BACK-TRANSFORM AND RECONSTRUCTION
# =============================================================================

def back_transform(
    betas    : np.ndarray,
    EOFs     : np.ndarray,
    sig_mask : np.ndarray,
) -> np.ndarray:
    """sensitivity_map = Σ beta_i * EOF_i  (significant PCs only)."""
    b = np.where(sig_mask, betas, 0.0)
    return (b[:, None] * EOFs).sum(axis=0).astype(np.float32)


def pca_reconstruction(
    PCs      : np.ndarray,
    betas    : np.ndarray,
    sig_mask : np.ndarray,
) -> np.ndarray:
    """P̂(t) = Σ beta_i * PC_i(t)  (significant PCs only)."""
    b = np.where(sig_mask, betas, 0.0)
    return (PCs @ b).astype(np.float32)


# =============================================================================
# CROSS-VALIDATION — per-fold SVD, 1-SE rule
# =============================================================================

def cross_validate_k(
    sst_flat_det : np.ndarray,
    weights      : np.ndarray,
    p_1d         : np.ndarray,
    years        : np.ndarray,
    alpha        : float = ALPHA,
    k_max        : int   = K_MAX,
) -> tuple:
    """
    LOO cross-validation over k = 1 … k_max.

    Key design choices
    ------------------
    Per-fold SVD (no leakage):
        EOFs are recomputed on each training set so the held-out year's
        SST pattern never influences the modes used to predict it.
        This is required for an unbiased estimate of predictive skill
        and for a fair comparison against linear regression CV skill.

    Truncated SVD:
        randomized_svd computes only the top k_max components, which is
        much faster than full SVD for large SST grids.

    1-SE rule for k selection:
        Picks the SMALLEST k whose LOO RE is within one standard error
        of the best RE across all tested k values. Implements the goal
        of minimum modes that genuinely predict rainfall in a
        statistically principled way (simpler model preferred when
        performance is indistinguishable from the best).

    Parameters
    ----------
    sst_flat_det : (n_time, n_ocean)  already detrended
    weights      : (n_ocean,)
    p_1d         : (n_time,)          already detrended
    years        : (n_time,)          integer year per time step
    alpha        : significance level
    k_max        : maximum PCs to test

    Returns
    -------
    cv_re     : (k_max,)  LOO Reduction of Error per k
    cv_corr   : (k_max,)  LOO Pearson r per k
    optimal_k : int       1-SE rule selection
    """
    unique_years = np.unique(years)
    p_var        = np.nanvar(p_1d)

    if p_var == 0:
        return (
            np.zeros(k_max, dtype=np.float32),
            np.zeros(k_max, dtype=np.float32),
            K_FALLBACK,
        )

    # Accumulate LOO predictions for all k simultaneously.
    # p_hat_all[k_idx, t] = predicted precip at time t when using k=k_idx+1 PCs
    p_hat_all = np.full((k_max, len(p_1d)), np.nan, dtype=np.float32)

    for yr in unique_years:
        test  = (years == yr)
        train = ~test

        if train.sum() < k_max + 2:
            continue

        sst_train = sst_flat_det[train, :]
        sst_test  = sst_flat_det[test,  :]

        # Truncated SVD on training fold only — the held-out year's SST
        # pattern never influences the EOFs used to predict it.
        _, s_tr, Vt_tr = randomized_svd(
            sst_train * weights[None, :],
            n_components=k_max,
            n_iter=4,
            random_state=0,
        )
        safe_w  = np.where(weights > 0, weights, 1.0)
        EOFs_tr = (Vt_tr / safe_w[None, :]).astype(np.float32)

        # Project train and test onto training-fold EOFs
        def project(sst_2d: np.ndarray, EOFs: np.ndarray) -> np.ndarray:
            X_w = sst_2d * weights[None, :]
            return (X_w @ (EOFs * weights[None, :]).T).astype(np.float32)

        PCs_train = project(sst_train, EOFs_tr)   # (n_train, k_max)
        PCs_test  = project(sst_test,  EOFs_tr)   # (n_test,  k_max)

        for k in range(1, k_max + 1):
            betas, _, sig, _ = pc_regression_numpy(
                PCs_train[:, :k], p_1d[train], alpha
            )
            p_hat_all[k - 1, test] = pca_reconstruction(
                PCs_test[:, :k], betas, sig
            )

    # Summarise CV scores
    cv_re   = np.full(k_max, np.nan, dtype=np.float32)
    cv_corr = np.full(k_max, np.nan, dtype=np.float32)

    for k in range(1, k_max + 1):
        p_hat = p_hat_all[k - 1]
        valid = np.isfinite(p_hat) & np.isfinite(p_1d)
        if valid.sum() < 3:
            continue
        mse            = np.mean((p_hat[valid] - p_1d[valid]) ** 2)
        cv_re[k - 1]   = float(1.0 - mse / p_var)
        cv_corr[k - 1] = float(np.corrcoef(p_hat[valid], p_1d[valid])[0, 1])

    # 1-SE rule: smallest k within one SE of the best RE
    finite_mask = np.isfinite(cv_re)
    if not finite_mask.any():
        return cv_re, cv_corr, K_FALLBACK

    finite_re = cv_re[finite_mask]
    best_re   = float(np.max(finite_re))
    se_re     = float(np.std(finite_re) / np.sqrt(len(finite_re)))
    threshold = best_re - se_re

    optimal_k = K_FALLBACK
    for k in range(1, k_max + 1):
        if np.isfinite(cv_re[k - 1]) and cv_re[k - 1] >= threshold:
            optimal_k = k
            break

    return cv_re, cv_corr, optimal_k


# =============================================================================
# PER-BASIN PCA REGRESSION
# =============================================================================

def run_basin(
    basin_id          : float,
    sst_flat_full     : np.ndarray,   # (n_sst_time, n_ocean)  NOT detrended
    sst_flat_det_full : np.ndarray,   # (n_sst_time, n_ocean)  detrended
    w_flat            : np.ndarray,   # (n_ocean,)
    ocean_mask        : np.ndarray,   # (n_lat, n_lon) bool
    sst_times         : np.ndarray,
    sst_lat           : np.ndarray,
    sst_lon           : np.ndarray,
    p_vals            : np.ndarray,   # (n_p_time, n_basins)
    p_times           : np.ndarray,
    p_basin_ids       : np.ndarray,
) -> dict | None:
    """Run PCA regression for a single basin."""

    # 1. Locate basin
    basin_idx = np.where(p_basin_ids == basin_id)[0]
    if len(basin_idx) == 0:
        return None
    basin_idx = basin_idx[0]

    # 2. Align times
    common_time = np.intersect1d(p_times, sst_times)
    if len(common_time) < MIN_YEARS:
        return None

    idx_p   = np.isin(p_times,   common_time)
    idx_sst = np.isin(sst_times, common_time)

    p_1d_raw = p_vals[idx_p, basin_idx].astype(np.float32)
    if not np.any(np.isfinite(p_1d_raw)):
        return None

    # 3. Slice pre-detrended SST to common time window
    sst_vals = sst_flat_det_full[idx_sst, :]

    # 4. Detrend precip
    p_det = detrend_along_time(p_1d_raw)
    years = (p_times[idx_p].astype("datetime64[Y]").astype(int) + 1970)

    k_max_effective = min(K_MAX, len(common_time) - 2)
    if k_max_effective < 1:
        return None

    # 5. Cross-validation: per-fold SVD, 1-SE rule
    cv_re, cv_corr, optimal_k = cross_validate_k(
        sst_vals, w_flat, p_det, years,
        alpha=ALPHA, k_max=k_max_effective,
    )

    # 6. Final model at optimal k on full record
    PCs, EOFs, var_exp = compute_pca_truncated(sst_vals, w_flat, optimal_k)
    betas, _, sig_mask, r2 = pc_regression_numpy(PCs, p_det, ALPHA)

    recon_1d  = pca_reconstruction(PCs, betas, sig_mask)
    sens_flat = back_transform(betas, EOFs, sig_mask)

    # 7. Unflatten sensitivity to (lat, lon) using global ocean_mask
    sens_2d = unflatten(sens_flat, ocean_mask)
    if sens_2d.shape != ocean_mask.shape:
        raise ValueError(
            f"Basin {basin_id}: sens_2d shape {sens_2d.shape} "
            f"!= expected {ocean_mask.shape}."
        )

    # 8. Diagnostics
    valid = np.isfinite(recon_1d) & np.isfinite(p_det)
    insample_corr = (
        float(np.corrcoef(recon_1d[valid], p_det[valid])[0, 1])
        if valid.sum() >= 3 else np.nan
    )
    std_obs   = float(np.nanstd(p_det))
    std_recon = float(np.nanstd(recon_1d))
    std_ratio = (std_recon / std_obs * 100.0) if std_obs > 0 else np.nan

    cv_re_opt   = float(cv_re[optimal_k - 1])   if np.isfinite(cv_re[optimal_k - 1])   else np.nan
    cv_corr_opt = float(cv_corr[optimal_k - 1]) if np.isfinite(cv_corr[optimal_k - 1]) else np.nan

    return {
        "reconstruction"         : recon_1d,
        "observed_precip"        : p_det,
        "time"                   : common_time,
        "sensitivity_map"        : sens_2d,
        "correlation"            : insample_corr,
        "std_ratio"              : std_ratio,
        "cv_re_optimal"          : cv_re_opt,
        "cv_corr_optimal"        : cv_corr_opt,
        "optimal_k"              : optimal_k,
        "variance_explained_sum" : float(np.sum(var_exp)),
        "r2_insample"            : r2,
    }


# =============================================================================
# ASSEMBLE OUTPUT DATASET
# =============================================================================

def build_output_dataset(
    basin_results : dict,
    sst_lat       : np.ndarray,
    sst_lon       : np.ndarray,
    all_times     : np.ndarray,
    p_name        : str,
    sst_name      : str,
    k_max         : int,
) -> xr.Dataset:
    """Pack per-basin dicts into a single xr.Dataset."""
    basin_ids = sorted(basin_results.keys())
    n_basins  = len(basin_ids)
    n_lat     = len(sst_lat)
    n_lon     = len(sst_lon)
    n_time    = len(all_times)

    recon_arr   = np.full((n_basins, n_time),       np.nan, dtype=np.float32)
    obs_arr     = np.full((n_basins, n_time),       np.nan, dtype=np.float32)
    sens_arr    = np.full((n_basins, n_lat, n_lon), np.nan, dtype=np.float32)
    corr_arr    = np.full(n_basins,                 np.nan, dtype=np.float32)
    ratio_arr   = np.full(n_basins,                 np.nan, dtype=np.float32)
    cv_re_arr   = np.full(n_basins,                 np.nan, dtype=np.float32)
    cv_corr_arr = np.full(n_basins,                 np.nan, dtype=np.float32)
    opt_k_arr   = np.full(n_basins,                 np.nan, dtype=np.float32)
    var_exp_arr = np.full(n_basins,                 np.nan, dtype=np.float32)

    for i, bid in enumerate(basin_ids):
        res   = basin_results[bid]
        t_idx = np.isin(all_times, res["time"])

        recon_arr[i, t_idx] = res["reconstruction"]
        obs_arr[i,   t_idx] = res["observed_precip"]
        sens_arr[i]         = res["sensitivity_map"]
        corr_arr[i]         = res["correlation"]
        ratio_arr[i]        = res["std_ratio"]
        cv_re_arr[i]        = res["cv_re_optimal"]
        cv_corr_arr[i]      = res["cv_corr_optimal"]
        opt_k_arr[i]        = res["optimal_k"]
        var_exp_arr[i]      = res["variance_explained_sum"]

    ds = xr.Dataset(
        {
            "reconstruction": xr.DataArray(
                recon_arr, dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"units": "mm/month",
                       "long_name": "PCA-based SST-forced precipitation reconstruction"},
            ),
            "observed_precip": xr.DataArray(
                obs_arr, dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"units": "mm/month",
                       "long_name": "Observed basin-mean precipitation anomaly (detrended)"},
            ),
            "sensitivity_map": xr.DataArray(
                sens_arr, dims=["basin", "lat", "lon"],
                coords={"basin": basin_ids, "lat": sst_lat, "lon": sst_lon},
                attrs={"units": "mm/month/K",
                       "long_name": "Back-transformed PCA sensitivity dP/dSST"},
            ),
            "correlation": xr.DataArray(
                corr_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "In-sample Pearson r(recon, obs)"},
            ),
            "std_ratio": xr.DataArray(
                ratio_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"units": "%", "long_name": "std(recon)/std(obs)*100"},
            ),
            "cv_re_optimal": xr.DataArray(
                cv_re_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "LOO Reduction of Error at optimal k"},
            ),
            "cv_corr_optimal": xr.DataArray(
                cv_corr_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "LOO Pearson r at optimal k"},
            ),
            "optimal_k": xr.DataArray(
                opt_k_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "Cross-validated optimal number of PCs (1-SE rule)"},
            ),
            "variance_explained_sum": xr.DataArray(
                var_exp_arr, dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "Fraction of SST variance explained by k PCs"},
            ),
        },
        attrs={
            "method"        : "Global PCA regression P ~ Σ beta_i*PC_i(SST)",
            "cv_method"     : "LOO with per-fold SVD (no data leakage)",
            "k_selection"   : "1-SE rule (simplest k within 1 SE of best LOO RE)",
            "precip_dataset": p_name,
            "sst_dataset"   : sst_name,
            "alpha"         : ALPHA,
            "significance"  : "Bonferroni over k PC tests per basin",
            "k_max_tested"  : k_max,
            "k_fallback"    : K_FALLBACK,
            "min_years"     : MIN_YEARS,
        },
    )
    return ds


# =============================================================================
# PROCESS ONE PAIR
# =============================================================================

def process_pair(p_name: str, sst_name: str) -> None:
    """Load data, run all basins, and save output for one (precip, SST) pair."""

    out_path = OUTPUTS_DIR / f"global_pca_regression_{p_name}_{sst_name}.nc"

    if SKIP_EXISTING and out_path.exists():
        print(f"  [EXISTS] {out_path.name} — skipping")
        return

    # --- Load SST ---
    sst_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if not sst_path.exists():
        print(f"  [SKIP] SST file not found: {sst_path}")
        return

    print(f"  Loading SST: {sst_name}")
    sst_anom = xr.open_dataarray(sst_path).squeeze()
    if "lev" in sst_anom.dims:
        sst_anom = sst_anom.squeeze("lev", drop=True)
    sst_anom.load()

    sst_lat   = sst_anom["lat"].values
    sst_lon   = sst_anom["lon"].values
    sst_times = sst_anom["time"].values

    sst_flat_full, w_flat, ocean_mask = flatten_sst(sst_anom)
    sst_flat_det_full = detrend_along_time(sst_flat_full)
    print(f"  Ocean cells : {sst_flat_full.shape[1]:,} / {ocean_mask.size:,}")

    # --- Load precip ---
    p_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
    if not p_path.exists():
        print(f"  [SKIP] Precip file not found: {p_path}")
        return

    print(f"  Loading precip: {p_name}")
    p_anom    = xr.open_dataarray(p_path).load()
    basin_ids = p_anom["basin"].values
    all_times = np.union1d(p_anom["time"].values, sst_times)
    print(f"  Basins: {len(basin_ids)}")

    p_vals      = p_anom.values
    p_times     = p_anom["time"].values
    p_basin_ids = p_anom["basin"].values

    # --- Basin loop ---
    basin_results = {}
    skipped       = 0

    for i, bid in enumerate(basin_ids):
        if i % 50 == 0:
            print(f"    Basin {i}/{len(basin_ids)} ...")
        res = run_basin(
            basin_id          = float(bid),
            sst_flat_full     = sst_flat_full,
            sst_flat_det_full = sst_flat_det_full,
            w_flat            = w_flat,
            ocean_mask        = ocean_mask,
            sst_times         = sst_times,
            sst_lat           = sst_lat,
            sst_lon           = sst_lon,
            p_vals            = p_vals,
            p_times           = p_times,
            p_basin_ids       = p_basin_ids,
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
    ds_out   = build_output_dataset(
        basin_results = basin_results,
        sst_lat       = sst_lat,
        sst_lon       = sst_lon,
        all_times     = all_times,
        p_name        = p_name,
        sst_name      = sst_name,
        k_max         = K_MAX,
    )
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f"  Saved → {out_path.name}")

    del sst_anom, sst_flat_full, sst_flat_det_full, p_anom, basin_results, ds_out
    gc.collect()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCA regression for one (precip, SST) dataset pair."
    )
    parser.add_argument(
        "--pair-index", type=int, default=None,
        help="Index of the (precip, SST) pair to process (0-based). "
             "Run --list-pairs to see the full table.",
    )
    parser.add_argument(
        "--list-pairs", action="store_true",
        help="Print the full pair index table and exit.",
    )
    args = parser.parse_args()

    pairs = get_all_pairs()

    # --list-pairs: print table and exit
    if args.list_pairs:
        print(f"\n{'Index':<8}{'Precip':<16}{'SST':<16}{'Output exists?'}")
        print("-" * 55)
        for i, (p, s) in enumerate(pairs):
            out   = OUTPUTS_DIR / f"global_pca_regression_{p}_{s}.nc"
            exist = "YES" if out.exists() else "no"
            print(f"{i:<8}{p:<16}{s:<16}{exist}")
        return

    # Require --pair-index when actually running
    if args.pair_index is None:
        parser.error("--pair-index is required unless using --list-pairs")

    total_pairs = len(pairs)
    if args.pair_index >= total_pairs:
        print(f"ERROR: --pair-index {args.pair_index} >= total pairs {total_pairs}")
        sys.exit(1)

    p_name, sst_name = pairs[args.pair_index]

    print("\n" + "=" * 80)
    print("GLOBAL PCA REGRESSION")
    print("=" * 80)
    print(f"Pair index   : {args.pair_index} / {total_pairs - 1}")
    print(f"Precip       : {p_name}")
    print(f"SST          : {sst_name}")
    print(f"K_MAX        : {K_MAX}")
    print(f"CV method    : per-fold SVD (no data leakage)")
    print(f"k selection  : 1-SE rule")
    print(f"Alpha        : {ALPHA}")
    print("=" * 80)

    process_pair(p_name, sst_name)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()