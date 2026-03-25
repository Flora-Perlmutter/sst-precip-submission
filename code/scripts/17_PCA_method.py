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
  3. Leave-one-year-out cross-validation to select optimal k
     (with a fixed-k fallback if CV fails or yields k=0).
  4. Final PCA regression on full record at optimal k.
  5. Back-transform PC coefficients to a spatial sensitivity map.
  6. Store per-basin scalar results (correlation, std_ratio, reconstruction)
     in a schema that mirrors global_linear_regression_bootstrap_{P}_{SST}.nc
     so that Figure 6 can be plotted with minimal changes.

Required input files (same as univariate pipeline)
--------------------------------------------------
  precip_anom_{P}.nc           (load_and_process_obs.py)
  sst_anom_{SST}.nc            (load_and_process_obs.py)
  grdc_basins/                 (copy from HPC Data/Other/)

Output files
------------
  global_pca_regression_{P}_{SST}.nc   — one file per (P, SST) pair

Output dataset schema (mirrors univariate bootstrap output)
-----------------------------------------------------------
  reconstruction         (basin, time)   — PCA-based P reconstruction
  observed_precip        (basin, time)   — observed basin-mean P anomaly
  correlation            (basin,)        — in-sample corr(recon, obs)
  sensitivity_map        (basin,lat,lon) — back-transformed dP/dSST [mm/month/K]
  std_ratio              (basin,)        — std(recon)/std(obs)*100  [%]
  cv_re_optimal          (basin,)        — LOO RE at optimal k
  cv_corr_optimal        (basin,)        — LOO Pearson r at optimal k
  optimal_k              (basin,)        — selected number of PCs
  variance_explained_sum (basin,)        — fraction of SST var explained by k PCs

Key design principle for sensitivity maps
-----------------------------------------
  SST is flattened and masked ONCE globally in main(), producing:
    sst_flat   : (n_time, n_ocean)   — pre-masked, area-weighted inputs
    w_flat     : (n_ocean,)          — sqrt-cos(lat) weights
    ocean_mask : (n_lat, n_lon) bool — global land/sea mask

  These three arrays are passed into every worker unchanged.  Inside
  run_basin the correct time slice is obtained by integer indexing:

    sst_vals = sst_flat_detrended[common_idx_sst, :]

  Because the same ocean_mask is used for both sst_vals → PCA and
  unflatten(sens_flat, ocean_mask), the back-projected sensitivity map
  is guaranteed to be (n_lat, n_lon) and spatially consistent across
  all basins.  Re-deriving the mask inside the worker (the earlier bug)
  is never done.

Optimizations
-------------
  1. Parallel basin processing via ProcessPoolExecutor (N_WORKERS controls cores).
  2. Fast CV: SVD computed ONCE on the full time record; only OLS is re-fit per
     LOO fold. This reduces SVD calls from k_max×n_years → 1.
  3. Early stopping in CV: stops testing higher k when RE has not improved for
     CV_PATIENCE consecutive values.
  4. SST detrending is done ONCE per (P, SST) pair outside the basin loop,
     so workers only receive an already-detrended flat array.

Configuration
-------------
Edit the CONFIGURATION block below to adjust paths, datasets, and PCA settings.
"""

import gc
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import statsmodels.api as sm
import xarray as xr
from scipy import signal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA, DATA_DIR
from regression_functions import detrend_dim, grid_area

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"


OUTPUTS_DIR = DATA_DIR

PRECIP_DATASETS = [
    "GPCP",
    "CRU",
    "GPCC",
    "CPC",
    "UDel",
    "PREC",
    "TerraClimate",
    "REGEN",
]

SST_DATASETS = ["ERSSTv6", "COBE-SST3"]

ALPHA        = 0.05   # significance threshold (Bonferroni-corrected per basin)
K_MAX        = 30     # maximum number of PCs to test in CV
K_FALLBACK   = 4      # fixed-k fallback used when CV fails or time series is short
MIN_YEARS    = 20     # basins with fewer valid years are skipped
SKIP_EXISTING = True  # set False to recompute and overwrite existing output files

# --- Optimization settings ---
N_WORKERS   = 8       # parallel worker processes for basin loop; set to 1 to disable
CV_PATIENCE = 3       # early-stop CV when RE hasn't improved for this many consecutive k


# =============================================================================
# DETRENDING UTILITY
# =============================================================================

def detrend_along_time(arr: np.ndarray) -> np.ndarray:
    """
    Linear detrend along axis=0 (time) for a 1-D or 2-D numpy array.

    Uses scipy.signal.detrend which handles NaN-free data efficiently.
    Any column that is all-NaN is left as NaN.

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
    # 2-D: detrend each column that has valid data
    out = np.empty_like(arr)
    finite_cols = np.all(np.isfinite(arr), axis=0)
    out[:, ~finite_cols] = np.nan
    if finite_cols.any():
        out[:, finite_cols] = signal.detrend(
            arr[:, finite_cols], axis=0
        ).astype(np.float32)
    return out


# =============================================================================
# PCA UTILITIES
# =============================================================================

def sst_area_weights(lat: np.ndarray, n_lon: int) -> np.ndarray:
    """
    sqrt(cos(lat)) weights, shape (n_lat, n_lon).

    Parameters
    ----------
    lat   : 1-D array of latitude values in degrees
    n_lon : number of longitude points
    """
    w   = np.cos(np.deg2rad(lat))
    w   = np.where(w > 0, np.sqrt(w), 0.0)
    w2d = np.broadcast_to(w[:, None], (len(lat), n_lon))
    return np.ascontiguousarray(w2d, dtype=np.float32)


def flatten_sst(sst: xr.DataArray) -> tuple:
    """
    (time, lat, lon) → (time, n_ocean), plus land/sea mask and weight vector.

    The ocean mask is defined as grid cells that are finite across ALL time
    steps.  This mask is computed ONCE and reused for every basin so that
    unflatten() always produces a (lat, lon) array on the same grid.

    Returns
    -------
    flat_vals  : (n_time, n_ocean) float32
    w_flat     : (n_ocean,)        float32  sqrt-cos weights
    ocean_mask : (n_lat, n_lon)    bool     True = ocean cell
    """
    vals  = sst.values.astype(np.float32)
    lat   = sst["lat"].values
    n_lon = len(sst["lon"])
    w2d   = sst_area_weights(lat, n_lon)
    ocean = np.all(np.isfinite(vals), axis=0)          # (n_lat, n_lon)
    return (
        vals[:, ocean].astype(np.float32),             # (n_time, n_ocean)
        w2d[ocean].astype(np.float32),                 # (n_ocean,)
        ocean,                                         # (n_lat, n_lon) bool
    )


def unflatten(flat: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """
    (n_ocean,) → (n_lat, n_lon), NaN on land.

    ocean_mask must be the SAME mask used to produce flat.
    """
    full             = np.full(ocean_mask.shape, np.nan, dtype=np.float32)
    full[ocean_mask] = flat
    return full


def compute_pca(sst_2d: np.ndarray, weights: np.ndarray, k: int) -> tuple:
    """
    Area-weighted PCA via truncated SVD.

    Parameters
    ----------
    sst_2d  : (n_time, n_ocean)  already detrended
    weights : (n_ocean,)         sqrt-cos area weights
    k       : number of leading PCs to retain

    Returns
    -------
    PCs     : (n_time, k)
    EOFs    : (k, n_ocean)   physical units (not weighted)
    var_exp : (k,)           fraction of total weighted variance
    """
    X_w      = sst_2d * weights[None, :]
    U, s, Vt = np.linalg.svd(X_w, full_matrices=False)
    U, s, Vt = U[:, :k], s[:k], Vt[:k, :]
    PCs      = (U * s[None, :]).astype(np.float32)
    safe_w   = np.where(weights > 0, weights, 1.0)
    EOFs     = (Vt / safe_w[None, :]).astype(np.float32)
    var_exp  = (s ** 2 / np.sum(X_w ** 2)).astype(np.float32)
    return PCs, EOFs, var_exp


def project_onto_eofs(sst_2d: np.ndarray, weights: np.ndarray, EOFs: np.ndarray) -> np.ndarray:
    """
    Project SST onto pre-computed EOFs — no data leakage in LOO CV.

    Parameters
    ----------
    sst_2d  : (n_time, n_ocean)
    weights : (n_ocean,)
    EOFs    : (k, n_ocean)

    Returns
    -------
    PCs : (n_time, k)
    """
    X_w = sst_2d * weights[None, :]
    return (X_w @ (EOFs * weights[None, :]).T).astype(np.float32)


def pc_regression(PCs: np.ndarray, p_1d: np.ndarray, alpha: float = ALPHA) -> tuple:
    """
    OLS: p ~ intercept + PC_1 + ... + PC_k, Bonferroni-corrected significance.

    Returns
    -------
    betas    : (k,)   regression coefficients
    betas_se : (k,)   standard errors
    pvals    : (k,)   p-values
    sig_mask : (k,)   bool, True = significant after Bonferroni correction
    r2       : float  in-sample R²
    """
    valid = ~np.isnan(p_1d) & np.all(np.isfinite(PCs), axis=1)
    k     = PCs.shape[1]
    if valid.sum() < k + 2:
        return (
            np.zeros(k, dtype=np.float32),
            np.zeros(k, dtype=np.float32),
            np.ones(k,  dtype=np.float32),
            np.zeros(k, dtype=bool),
            0.0,
        )
    model    = sm.OLS(p_1d[valid], sm.add_constant(PCs[valid])).fit()
    betas    = model.params[1:].astype(np.float32)
    betas_se = model.bse[1:].astype(np.float32)
    pvals    = model.pvalues[1:].astype(np.float32)
    sig_mask = pvals <= (alpha / k)                   # Bonferroni
    return betas, betas_se, pvals, sig_mask, float(model.rsquared)


def back_transform(betas: np.ndarray, EOFs: np.ndarray, sig_mask: np.ndarray) -> np.ndarray:
    """
    sensitivity_map = Σ beta_i * EOF_i  (significant PCs only).

    Returns
    -------
    (n_ocean,) float32
    """
    b = np.where(sig_mask, betas, 0.0)
    return (b[:, None] * EOFs).sum(axis=0).astype(np.float32)


def pca_reconstruction(PCs: np.ndarray, betas: np.ndarray, sig_mask: np.ndarray) -> np.ndarray:
    """
    P̂(t) = Σ beta_i * PC_i(t)  (significant PCs only).

    Returns
    -------
    (n_time,) float32
    """
    b = np.where(sig_mask, betas, 0.0)
    return (PCs @ b).astype(np.float32)


# =============================================================================
# CROSS-VALIDATION — fast version (single SVD + early stopping)
# =============================================================================

def cross_validate_k_fast(
    sst_flat_det : np.ndarray,
    weights      : np.ndarray,
    p_1d         : np.ndarray,
    years        : np.ndarray,
    alpha        : float = ALPHA,
    k_max        : int   = K_MAX,
    patience     : int   = CV_PATIENCE,
) -> tuple:
    """
    LOO cross-validation over k = 1 … k_max.

    Optimization: Single SVD
      EOFs are computed ONCE on the full time record. The LOO loop only
      re-fits OLS (cheap), eliminating k_max × n_years redundant SVD calls.

    Optimization: Early stopping
      Stops incrementing k when CV RE has not improved for `patience`
      consecutive values, avoiding wasted OLS fits at high k.

    Parameters
    ----------
    sst_flat_det : (n_time, n_ocean)  already detrended SST
    weights      : (n_ocean,)
    p_1d         : (n_time,)          already detrended precip
    years        : (n_time,)          integer year for each time step
    alpha        : significance level
    k_max        : maximum PCs to test
    patience     : early-stop patience

    Returns
    -------
    cv_re     : (k_max,)  Reduction of Error per k  (NaN beyond early-stop)
    cv_corr   : (k_max,)  Pearson r per k
    optimal_k : int       argmax(RE), minimum 1
    """
    unique_years = np.unique(years)
    p_var        = np.nanvar(p_1d)
    if p_var == 0:
        return (
            np.zeros(k_max, dtype=np.float32),
            np.zeros(k_max, dtype=np.float32),
            K_FALLBACK,
        )

    # Single SVD on the full record — EOFs fixed for all LOO folds
    _, EOFs_full, _ = compute_pca(sst_flat_det, weights, k_max)
    PCs_full        = project_onto_eofs(sst_flat_det, weights, EOFs_full)  # (n_t, k_max)

    cv_re   = np.full(k_max, np.nan, dtype=np.float32)
    cv_corr = np.full(k_max, np.nan, dtype=np.float32)

    best_re    = -np.inf
    no_improve = 0

    for k in range(1, k_max + 1):
        PCs_k = PCs_full[:, :k]
        p_hat = np.full_like(p_1d, np.nan)

        for yr in unique_years:
            test  = (years == yr)
            train = ~test
            if train.sum() < k + 2:
                continue
            betas, _, _, sig, _ = pc_regression(PCs_k[train], p_1d[train], alpha)
            p_hat[test]         = pca_reconstruction(PCs_k[test], betas, sig)

        valid = np.isfinite(p_hat) & np.isfinite(p_1d)
        if valid.sum() < 3:
            no_improve += 1
        else:
            mse            = np.mean((p_hat[valid] - p_1d[valid]) ** 2)
            cv_re[k - 1]   = 1.0 - mse / p_var
            cv_corr[k - 1] = float(np.corrcoef(p_hat[valid], p_1d[valid])[0, 1])

            if cv_re[k - 1] > best_re + 1e-4:
                best_re    = cv_re[k - 1]
                no_improve = 0
            else:
                no_improve += 1

        if no_improve >= patience:
            break

    finite_re = np.where(np.isfinite(cv_re), cv_re, -np.inf)
    if np.all(finite_re == -np.inf):
        return cv_re, cv_corr, K_FALLBACK

    optimal_k = int(np.argmax(finite_re)) + 1
    return cv_re, cv_corr, optimal_k


# =============================================================================
# PER-BASIN PCA REGRESSION
# =============================================================================

def run_basin(
    basin_id          : float,
    # SST — pre-flattened with GLOBAL ocean mask, FULL time record
    sst_flat_full     : np.ndarray,   # (n_sst_time, n_ocean)  NOT yet detrended
    sst_flat_det_full : np.ndarray,   # (n_sst_time, n_ocean)  detrended version
    w_flat            : np.ndarray,   # (n_ocean,)
    ocean_mask        : np.ndarray,   # (n_lat, n_lon) bool  — GLOBAL, FIXED
    sst_times         : np.ndarray,   # (n_sst_time,) datetime64
    sst_lat           : np.ndarray,
    sst_lon           : np.ndarray,
    # Precip — passed as numpy for worker-process compatibility
    p_vals            : np.ndarray,   # (n_p_time, n_basins)
    p_times           : np.ndarray,   # (n_p_time,)
    p_basin_ids       : np.ndarray,   # (n_basins,)
) -> dict | None:
    """
    Run PCA regression for a single basin.

    SST is NEVER re-flattened or re-masked here.  The global ocean_mask
    passed in is used both to select SST columns AND inside unflatten(),
    guaranteeing that sensitivity_map is always (n_lat, n_lon).

    Returns a dict of results, or None if the basin is skipped.
    """
    # ------------------------------------------------------------------
    # 1. Locate basin in precip array
    # ------------------------------------------------------------------
    basin_idx = np.where(p_basin_ids == basin_id)[0]
    if len(basin_idx) == 0:
        return None
    basin_idx = basin_idx[0]

    # ------------------------------------------------------------------
    # 2. Align times between precip and SST
    # ------------------------------------------------------------------
    common_time = np.intersect1d(p_times, sst_times)
    if len(common_time) < MIN_YEARS:
        return None

    idx_p   = np.isin(p_times,   common_time)
    idx_sst = np.isin(sst_times, common_time)

    p_1d_raw = p_vals[idx_p, basin_idx].astype(np.float32)

    # Guard: all-NaN basin
    if not np.any(np.isfinite(p_1d_raw)):
        return None

    # ------------------------------------------------------------------
    # 3. Slice pre-detrended SST to common time
    #    (mask already applied — no re-flattening needed)
    # ------------------------------------------------------------------
    sst_vals = sst_flat_det_full[idx_sst, :]   # (n_common, n_ocean)

    # ------------------------------------------------------------------
    # 4. Detrend precip
    # ------------------------------------------------------------------
    p_det = detrend_along_time(p_1d_raw)       # (n_common,)

    years = p_times[idx_p]
    # Convert datetime64 → integer years
    years = years.astype("datetime64[Y]").astype(int) + 1970

    n_t = len(common_time)

    # ------------------------------------------------------------------
    # 5. Cross-validation to choose optimal k
    # ------------------------------------------------------------------
    k_max_effective = min(K_MAX, n_t - 2)
    if k_max_effective < 1:
        return None

    cv_re, cv_corr, optimal_k = cross_validate_k_fast(
        sst_vals, w_flat, p_det, years,
        alpha=ALPHA, k_max=k_max_effective, patience=CV_PATIENCE,
    )

    # ------------------------------------------------------------------
    # 6. Final model at optimal k
    # ------------------------------------------------------------------
    PCs, EOFs, var_exp = compute_pca(sst_vals, w_flat, optimal_k)
    betas, _, _, sig_mask, r2 = pc_regression(PCs, p_det, ALPHA)

    recon_1d  = pca_reconstruction(PCs, betas, sig_mask)   # (n_common,)
    sens_flat = back_transform(betas, EOFs, sig_mask)       # (n_ocean,)

    # ------------------------------------------------------------------
    # 7. Back-project sensitivity to (lat, lon) using GLOBAL ocean_mask
    # ------------------------------------------------------------------
    sens_2d = unflatten(sens_flat, ocean_mask)              # (n_lat, n_lon)

    # Sanity check (cheap — catches any future mask mismatch immediately)
    expected_shape = ocean_mask.shape
    if sens_2d.shape != expected_shape:
        raise ValueError(
            f"Basin {basin_id}: sens_2d shape {sens_2d.shape} "
            f"!= expected {expected_shape}.  "
            "ocean_mask mismatch — check that sst_flat was produced "
            "with the same mask passed to run_basin."
        )

    # ------------------------------------------------------------------
    # 8. Scalar diagnostics
    # ------------------------------------------------------------------
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
        "reconstruction"         : recon_1d,          # (n_common,)
        "observed_precip"        : p_det,              # (n_common,)
        "time"                   : common_time,        # (n_common,)
        "sensitivity_map"        : sens_2d,            # (n_lat, n_lon) ← KEY FIX
        "correlation"            : insample_corr,
        "std_ratio"              : std_ratio,
        "cv_re_optimal"          : cv_re_opt,
        "cv_corr_optimal"        : cv_corr_opt,
        "optimal_k"              : optimal_k,
        "variance_explained_sum" : float(np.sum(var_exp)),
        "r2_insample"            : r2,
    }


# =============================================================================
# WORKER — top-level function for ProcessPoolExecutor (must be picklable)
# =============================================================================

def _basin_worker(args: tuple) -> tuple:
    """
    Thin picklable wrapper around run_basin.

    All data passed as numpy arrays (no xarray objects) so that
    multiprocessing serialisation is fast and reliable.

    Returns (basin_id, result_dict_or_None).
    """
    (basin_id,
     sst_flat_full, sst_flat_det_full,
     w_flat, ocean_mask,
     sst_times, sst_lat, sst_lon,
     p_vals, p_times, p_basin_ids) = args

    result = run_basin(
        basin_id          = basin_id,
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
    return basin_id, result


def run_basins_parallel(
    basin_ids         : np.ndarray,
    sst_flat_full     : np.ndarray,
    sst_flat_det_full : np.ndarray,
    w_flat            : np.ndarray,
    ocean_mask        : np.ndarray,
    sst_times         : np.ndarray,
    sst_lat           : np.ndarray,
    sst_lon           : np.ndarray,
    p_anom            : xr.DataArray,
    n_workers         : int,
    desc              : str = "",
) -> tuple[dict, int]:
    """
    Dispatch run_basin across N_WORKERS processes (Optimization 1).

    Falls back to a sequential loop when n_workers == 1.

    Pre-extracts all numpy arrays from xarray ONCE here so each worker
    receives only plain numpy data (fast pickle, no xarray overhead).

    Returns
    -------
    basin_results : dict  {basin_id → result_dict}
    n_skipped     : int
    """
    # Extract numpy once — workers share read-only copies via pickle
    p_vals      = p_anom.values                  # (n_p_time, n_basins)
    p_times     = p_anom["time"].values           # (n_p_time,) datetime64
    p_basin_ids = p_anom["basin"].values          # (n_basins,)

    def make_args(bid: float) -> tuple:
        return (
            float(bid),
            sst_flat_full, sst_flat_det_full,
            w_flat, ocean_mask,
            sst_times, sst_lat, sst_lon,
            p_vals, p_times, p_basin_ids,
        )

    basin_results: dict = {}
    skipped = 0

    if n_workers == 1:
        for bid in tqdm(basin_ids, desc=desc):
            _, res = _basin_worker(make_args(bid))
            if res is None:
                skipped += 1
            else:
                basin_results[float(bid)] = res
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_basin_worker, make_args(bid)): bid
                for bid in basin_ids
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                bid, res = future.result()
                if res is None:
                    skipped += 1
                else:
                    basin_results[float(bid)] = res

    return basin_results, skipped


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
    """
    Pack per-basin dicts into a single xr.Dataset mirroring the
    univariate bootstrap schema expected by the Figure 6 script.

    sensitivity_map is stored as (basin, lat, lon) — one full spatial
    field per basin.
    """
    basin_ids = sorted(basin_results.keys())
    n_basins  = len(basin_ids)
    n_lat     = len(sst_lat)
    n_lon     = len(sst_lon)
    n_time    = len(all_times)

    recon_arr   = np.full((n_basins, n_time),         np.nan, dtype=np.float32)
    obs_arr     = np.full((n_basins, n_time),         np.nan, dtype=np.float32)
    sens_arr    = np.full((n_basins, n_lat, n_lon),   np.nan, dtype=np.float32)
    corr_arr    = np.full(n_basins,                   np.nan, dtype=np.float32)
    ratio_arr   = np.full(n_basins,                   np.nan, dtype=np.float32)
    cv_re_arr   = np.full(n_basins,                   np.nan, dtype=np.float32)
    cv_corr_arr = np.full(n_basins,                   np.nan, dtype=np.float32)
    opt_k_arr   = np.full(n_basins,                   np.nan, dtype=np.float32)
    var_exp_arr = np.full(n_basins,                   np.nan, dtype=np.float32)

    for i, bid in enumerate(basin_ids):
        res   = basin_results[bid]
        t_idx = np.isin(all_times, res["time"])

        recon_arr[i, t_idx] = res["reconstruction"]
        obs_arr[i,   t_idx] = res["observed_precip"]
        sens_arr[i]         = res["sensitivity_map"]   # (n_lat, n_lon) ← always 2D
        corr_arr[i]         = res["correlation"]
        ratio_arr[i]        = res["std_ratio"]
        cv_re_arr[i]        = res["cv_re_optimal"]
        cv_corr_arr[i]      = res["cv_corr_optimal"]
        opt_k_arr[i]        = res["optimal_k"]
        var_exp_arr[i]      = res["variance_explained_sum"]

    ds = xr.Dataset(
        {
            "reconstruction": xr.DataArray(
                recon_arr,
                dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"units": "mm/month",
                       "long_name": "PCA-based SST-forced precipitation reconstruction"},
            ),
            "observed_precip": xr.DataArray(
                obs_arr,
                dims=["basin", "time"],
                coords={"basin": basin_ids, "time": all_times},
                attrs={"units": "mm/month",
                       "long_name": "Observed basin-mean precipitation anomaly (detrended)"},
            ),
            "sensitivity_map": xr.DataArray(
                sens_arr,
                dims=["basin", "lat", "lon"],
                coords={"basin": basin_ids, "lat": sst_lat, "lon": sst_lon},
                attrs={"units": "mm/month/K",
                       "long_name": "Back-transformed PCA sensitivity dP/dSST"},
            ),
            "correlation": xr.DataArray(
                corr_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "In-sample Pearson r(recon, obs)"},
            ),
            "std_ratio": xr.DataArray(
                ratio_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"units": "%",
                       "long_name": "std(recon)/std(obs)*100"},
            ),
            "cv_re_optimal": xr.DataArray(
                cv_re_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "LOO Reduction of Error at optimal k"},
            ),
            "cv_corr_optimal": xr.DataArray(
                cv_corr_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "LOO Pearson r at optimal k"},
            ),
            "optimal_k": xr.DataArray(
                opt_k_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "Cross-validated optimal number of PCs"},
            ),
            "variance_explained_sum": xr.DataArray(
                var_exp_arr,
                dims=["basin"],
                coords={"basin": basin_ids},
                attrs={"long_name": "Fraction of SST variance explained by k PCs"},
            ),
        },
        attrs={
            "method"         : "Global PCA regression P ~ Σ beta_i*PC_i(SST)",
            "precip_dataset" : p_name,
            "sst_dataset"    : sst_name,
            "alpha"          : ALPHA,
            "significance"   : "Bonferroni over k PC tests per basin",
            "k_max_tested"   : k_max,
            "k_fallback"     : K_FALLBACK,
            "min_years"      : MIN_YEARS,
            "cv_patience"    : CV_PATIENCE,
            "n_workers"      : N_WORKERS,
        },
    )
    return ds


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("GLOBAL PCA REGRESSION — all basins × precip datasets × SST datasets")
    print(f"Workers: {N_WORKERS}  |  CV patience: {CV_PATIENCE}  |  K_MAX: {K_MAX}")
    print("=" * 80)

    for sst_name in SST_DATASETS:
        # ------------------------------------------------------------------
        # Load SST once per SST dataset
        # ------------------------------------------------------------------
        sst_path = INPUTS_DIR / f"sst_anom_{sst_name}.nc"
        if not sst_path.exists():
            print(f"\n[SKIP] SST file not found: {sst_path}")
            continue

        print(f"\n{'='*60}")
        print(f"SST dataset: {sst_name}")
        print(f"{'='*60}")

        sst_anom_full = xr.open_dataarray(sst_path).squeeze()
        if "lev" in sst_anom_full.dims:
            sst_anom_full = sst_anom_full.squeeze("lev", drop=True)
        sst_anom_full.load()

        sst_lat   = sst_anom_full["lat"].values
        sst_lon   = sst_anom_full["lon"].values
        sst_times = sst_anom_full["time"].values

        # ------------------------------------------------------------------
        # Flatten SST ONCE with a GLOBAL ocean mask
        # sst_flat_full     : (n_sst_time, n_ocean)  raw anomalies
        # sst_flat_det_full : (n_sst_time, n_ocean)  detrended anomalies
        # w_flat            : (n_ocean,)
        # ocean_mask        : (n_lat, n_lon) bool
        #
        # Key: ocean_mask is fixed for ALL basins and ALL precip datasets.
        # Workers use it ONLY via integer indexing and unflatten() —
        # never re-derived — so sens_2d is always (n_lat, n_lon).
        # ------------------------------------------------------------------
        sst_flat_full, w_flat, ocean_mask = flatten_sst(sst_anom_full)
        sst_flat_det_full = detrend_along_time(sst_flat_full)

        print(f"  Ocean cells : {sst_flat_full.shape[1]:,} / {ocean_mask.size:,}")

        for p_name in PRECIP_DATASETS:
            out_path = OUTPUTS_DIR / f"global_pca_regression_{p_name}_{sst_name}.nc"

            if SKIP_EXISTING and out_path.exists():
                print(f"\n  [EXISTS] {out_path.name} — skipping")
                continue

            p_path = INPUTS_DIR / f"precip_anom_{p_name}.nc"
            if not p_path.exists():
                print(f"\n  [SKIP] Precip file not found: {p_path}")
                continue

            print(f"\n  Precip: {p_name}")
            p_anom    = xr.open_dataarray(p_path).load()
            basin_ids = p_anom["basin"].values
            print(f"  Basins    : {len(basin_ids)}")

            all_times = np.union1d(p_anom["time"].values, sst_times)

            # ------------------------------------------------------------------
            # Run per-basin PCA regression (parallel)
            # ------------------------------------------------------------------
            basin_results, skipped = run_basins_parallel(
                basin_ids         = basin_ids,
                sst_flat_full     = sst_flat_full,
                sst_flat_det_full = sst_flat_det_full,
                w_flat            = w_flat,
                ocean_mask        = ocean_mask,
                sst_times         = sst_times,
                sst_lat           = sst_lat,
                sst_lon           = sst_lon,
                p_anom            = p_anom,
                n_workers         = N_WORKERS,
                desc              = f"    {p_name}/{sst_name}",
            )

            print(f"    Completed: {len(basin_results)} basins  |  Skipped: {skipped}")

            if not basin_results:
                print("    No valid basins — skipping output.")
                del p_anom
                gc.collect()
                continue

            # ------------------------------------------------------------------
            # Build and save output dataset
            # ------------------------------------------------------------------
            ds_out = build_output_dataset(
                basin_results = basin_results,
                sst_lat       = sst_lat,
                sst_lon       = sst_lon,
                all_times     = all_times,
                p_name        = p_name,
                sst_name      = sst_name,
                k_max         = K_MAX,
            )

            encoding = {
                v: {"zlib": True, "complevel": 4}
                for v in ds_out.data_vars
            }
            ds_out.to_netcdf(out_path, encoding=encoding)
            print(f"    Saved → {out_path.name}")

            del p_anom, basin_results, ds_out
            gc.collect()

        del sst_anom_full, sst_flat_full, sst_flat_det_full, w_flat, ocean_mask
        gc.collect()

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()