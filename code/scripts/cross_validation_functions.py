#!/usr/bin/env python
# coding: utf-8
"""
K-fold cross-validation functions for SST-precipitation regression models.

Usage
-----
    from cross_validation_functions import (
        cv_regression_model,
        cv_regression_model_with_predictor,
    )
"""

import warnings

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_scalar(x) -> float:
    """Force conversion to a Python float scalar (required by xr.apply_ufunc)."""
    return float(np.asarray(x).item())


def _nan_result():
    """Return a 12-tuple of NaNs for failed fits."""
    return (np.nan,) * 12


def _run_kfold(X: np.ndarray, precip: np.ndarray, n_splits: int) -> tuple:
    """
    Fit OLS with K-fold CV on a pre-built design matrix X.

    Returns
    -------
    12-tuple: rmse_mean, rmse_std, mae_mean, mae_std, nrmse_mean, nrmse_std,
              coef_mean, coef_std, adjr2_mean, adjr2_std, r2_mean, r2_std
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_list, mae_list, nrmse_list, coef_list, adjr2_list, r2_list = [], [], [], [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = precip[train_idx], precip[test_idx]

        try:
            model  = sm.OLS(y_train, X_train).fit()
            y_pred = model.predict(X_test)

            r2  = r2_score(y_test, y_pred)
            n, k = len(y_test), X_test.shape[1] - 1   # k excludes constant
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else np.nan

            rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_list.append(rmse_val)
            mae_list.append(mean_absolute_error(y_test, y_pred))
            nrmse_list.append(rmse_val / np.std(y_test) if np.std(y_test) > 0 else np.nan)
            coef_list.append(model.params[1])
            adjr2_list.append(adj_r2)
            r2_list.append(r2)

        except Exception:
            continue

    if not rmse_list:
        return _nan_result()

    return (
        _to_scalar(np.mean(rmse_list)),  _to_scalar(np.std(rmse_list)),
        _to_scalar(np.mean(mae_list)),   _to_scalar(np.std(mae_list)),
        _to_scalar(np.mean(nrmse_list)), _to_scalar(np.std(nrmse_list)),
        _to_scalar(np.mean(coef_list)),  _to_scalar(np.std(coef_list)),
        _to_scalar(np.mean(adjr2_list)), _to_scalar(np.std(adjr2_list)),
        _to_scalar(np.mean(r2_list)),    _to_scalar(np.std(r2_list)),
    )


# ---------------------------------------------------------------------------
# SST-only models
# ---------------------------------------------------------------------------

def cv_regression_model(sst, precip, model_id: str = None, n_splits: int = 5) -> tuple:
    """
    K-fold cross-validation for SST-only regression models.

    Supported model_ids
    -------------------
    'P ~ β*SST'
    'P ~ intercept + β1*SST + β2*SST²'
    'P ~ intercept + β1*SST + β2*log(SST)'
    'P(t) ~ β₀·SST(t) + β₁·SST(t-1)'               (1 lag)
    'P(t) ~ β₀·SST(t) + ... + β₂·SST(t-2)'          (2 lags)
    'P(t) ~ β₀·SST(t) + ... + β₃·SST(t-3)'          (3 lags)
    'P(t) ~ β₀·SST(t) + ... + β₄·SST(t-4)'          (4 lags)

    Returns
    -------
    12-tuple: rmse_mean, rmse_std, mae_mean, mae_std, nrmse_mean, nrmse_std,
              coef_mean, coef_std, adjr2_mean, adjr2_std, r2_mean, r2_std
    """
    sst    = np.asarray(sst)
    precip = np.asarray(precip)

    valid = ~np.isnan(sst) & ~np.isnan(precip)
    if np.sum(valid) < n_splits:
        return _nan_result()

    sst    = sst[valid]
    precip = precip[valid]

    # Build lagged design matrix
    if model_id == "P ~ β*SST":
        X_list = [sst]
    elif model_id == "P ~ intercept + β1*SST + β2*SST²":
        X_list = [sst, sst ** 2]
    elif model_id == "P ~ intercept + β1*SST + β2*log(SST)":
        X_list = [sst, np.log(np.where(sst > 0, sst, np.nan))]
    elif model_id == "P(t) ~ β₀·SST(t) + β₁·SST(t-1)":
        X_list, precip = [sst[:-1], sst[1:]], precip[1:]
    elif model_id == "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2)":
        X_list, precip = [sst[:-2], sst[1:-1], sst[2:]], precip[2:]
    elif model_id == "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3)":
        X_list, precip = [sst[:-3], sst[1:-2], sst[2:-1], sst[3:]], precip[3:]
    elif model_id == "P(t) ~ β₀·SST(t) + β₁·SST(t-1) + β₂·SST(t-2) + β₃·SST(t-3) + β₄·SST(t-4)":
        X_list, precip = [sst[:-4], sst[1:-3], sst[2:-2], sst[3:-1], sst[4:]], precip[4:]
    else:
        X_list = [sst]

    X = sm.add_constant(np.column_stack(X_list))
    return _run_kfold(X, precip, n_splits)


# ---------------------------------------------------------------------------
# Conditioning-variable models
# ---------------------------------------------------------------------------

# Model IDs that use an interaction term (sst * predictor)
_INTERACTION_MODELS = {
    "P ~ β₀·SST + β₁·OLR + β₂·SST·OLR",
    "P ~ β₀·SST + β₁·LR + β₂·SST·LR",
    "P ~ β₀·SST + β₁·GMST + β₂·SST·GMST",
    "P ~ β₀·SST + β₁·RH + β₂·SST·RH",
    "P ~ β₀·SST + β₁·z500 RH + β₂·SST·z500 RH",
    "P ~ β₀·SST + β₁·column RH + β₂·SST·column RH",
    "P ~ β₀·SST + β₁·sfc RH + β₂·SST·sfc RH",
}


def cv_regression_model_with_predictor(
    sst,
    precip,
    predictor,
    model_id: str = None,
    n_splits: int = 5,
) -> tuple:
    """
    K-fold cross-validation for SST + conditioning variable regression models.

    The predictor array (OLR, lapse rate, GMST, RH, etc.) must be
    time-aligned with sst and precip before calling this function.

    Returns
    -------
    12-tuple: rmse_mean, rmse_std, mae_mean, mae_std, nrmse_mean, nrmse_std,
              coef_mean, coef_std, adjr2_mean, adjr2_std, r2_mean, r2_std
    """
    sst       = np.asarray(sst)
    precip    = np.asarray(precip)
    predictor = np.asarray(predictor)

    valid = ~np.isnan(sst) & ~np.isnan(precip) & ~np.isnan(predictor)
    if np.sum(valid) < n_splits:
        return _nan_result()

    sst       = sst[valid]
    precip    = precip[valid]
    predictor = predictor[valid]

    if model_id in _INTERACTION_MODELS:
        X_list = [sst, predictor, sst * predictor]
    else:
        # Simple additive model (sst + predictor, no interaction)
        X_list = [sst, predictor]

    X = sm.add_constant(np.column_stack(X_list))
    return _run_kfold(X, precip, n_splits)