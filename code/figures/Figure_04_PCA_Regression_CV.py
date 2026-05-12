#!/usr/bin/env python
# coding: utf-8
"""
Figure 4: Reconstruction CV comparison — Regression vs PCA.

Author: Flora Perlmutter

Description
-----------
Reads processed CV results and creates a multi-panel figure comparing
regression and PCA reconstruction skill across river basins.

Each point = one basin × dataset-pair combination.
Ensemble-mean win counts shown in corner annotation.

Panels:
  (a) Paired scatter: RE for regression vs PCA
  (b) Paired scatter: Correlation for regression vs PCA

Input
-----
  <DATA_DIR>/cv_reconstruction_ensemble_metrics.nc   (for ensemble-mean stats)
  <DATA_DIR>/cv_reconstruction_per_basin.csv         (for individual points)

Output
------
  <PAPER_FIGURE_DIR>/Figure_04_cv_reconstruction_comparison.png
"""
import sys
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PAPER_FIGURE_DIR, DATA_DIR

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROCESSED_DIR = DATA_DIR
FIGURES_DIR   = PAPER_FIGURE_DIR

# ---------------------------------------------------------------------------
# Load Complete CV Data
# ---------------------------------------------------------------------------

PRECIP_DATASETS = [
    "GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN",
]
SST_DATASETS = ["ERSSTv6", "COBE-SST3"]

CV_DIR = DATA_DIR


def load_cv_results(verbose=True):
    datasets = []
    pair_names = []

    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            path = CV_DIR / f"cv_reconstruction_{p_name}_{sst_name}.nc"
            if not path.exists():
                if verbose:
                    print(f"  [MISSING] {path.name}")
                continue

            ds = xr.open_dataset(path)
            pair_label = f"{p_name}_{sst_name}"
            ds = ds.expand_dims(pair=[pair_label])
            datasets.append(ds)
            pair_names.append(pair_label)
            ds.close()

            if verbose:
                print(f"  Loaded {pair_label}")

    if not datasets:
        raise FileNotFoundError("No CV reconstruction files found.")

    return xr.concat(datasets, dim="pair")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})


# =============================================================================
# PANEL FUNCTIONS
# =============================================================================
def plot_scatter_panel(ax, basin_df, metric, metric_label, panel_label,
                       ensemble_pca_wins, ensemble_reg_wins):
    """
    Paired scatter: regression (x) vs PCA (y).
    Each point = one basin × dataset-pair.
    Ensemble-mean win counts shown in the top-left corner.
    """
    sub = basin_df[basin_df["metric"] == metric].copy()

    # Each row already has regression and pca columns (ensemble-mean per basin).
    # To plot individual pairs we need the raw pair-level data, which is stored
    # in the wide per_basin CSV produced by build_per_basin_table.

    r = sub["regression"].values
    p = sub["pca"].values

    valid = np.isfinite(r) & np.isfinite(p)
    r, p = r[valid], p[valid]

    # 1:1 line
    lo = min(r.min(), p.min())
    hi = max(r.max(), p.max())
    margin = 0.05 * (hi - lo) if hi > lo else 0.1
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        "k--", lw=0.8, alpha=0.5, zorder=5, label='1:1 line'
    )

    ax.scatter(r, p, s=10, alpha=0.3, edgecolors="black", linewidth=0.5, zorder=2)

    ax.set_xlabel(f"Regression {metric_label}")
    ax.set_ylabel(f"PCA {metric_label}")
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)
    ax.set_aspect("equal", adjustable="box")

    # Ensemble-mean win counts in top-left corner
    ax.text(
        0.025, 0.91,
        f"PCA better: {ensemble_pca_wins}\nRegression better: {ensemble_reg_wins}",
        transform=ax.transAxes,
        fontsize=5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", alpha=0.8),
    )
    

    ax.legend(fontsize=5, loc='upper left')
    ax.text(
        -.08, 1.05, panel_label,
        transform=ax.transAxes,
        fontsize=10, fontweight="bold", va="top",
    )



# =============================================================================
# HELPERS
# =============================================================================
def load_pair_level_data(ds_raw):
    """
    Reshape the raw (pair, basin, method) Dataset into a long DataFrame
    with columns: pair, basin, metric, regression, pca.

    This is what gives you one point per basin × dataset-pair in the scatter.
    """
    records = []
    metrics = ["re", "correlation", "rmse", "nrmse", "mae"]

    for metric in metrics:
        if metric not in ds_raw.data_vars:
            continue

        da = ds_raw[metric]  # (pair, basin, method)
        pairs  = da["pair"].values
        basins = da["basin"].values

        for pi, pair in enumerate(pairs):
            for bi, basin in enumerate(basins):
                r_val = float(da.isel(pair=pi, basin=bi, method=0).values)
                p_val = float(da.isel(pair=pi, basin=bi, method=1).values)
                records.append({
                    "pair":       pair,
                    "basin":      basin,
                    "metric":     metric,
                    "regression": r_val,
                    "pca":        p_val,
                })

    return pd.DataFrame(records)

def load_cv_results(verbose=True):
    """
    Load all cv_reconstruction_{P}_{SST}.nc files and combine
    along a 'pair' dimension.

    Returns
    -------
    xr.Dataset with dims (pair, basin, method) for each metric,
    plus (pair, basin, time) for predictions.
    """
    datasets = []
    pair_names = []

    for p_name in PRECIP_DATASETS:
        for sst_name in SST_DATASETS:
            path = CV_DIR / f"cv_reconstruction_{p_name}_{sst_name}.nc"
            if not path.exists():
                if verbose:
                    print(f"  [MISSING] {path.name}")
                continue

            ds = xr.open_dataset(path)
            pair_label = f"{p_name}_{sst_name}"
            ds = ds.expand_dims(pair=[pair_label])
            datasets.append(ds)
            pair_names.append(pair_label)
            ds.close()

            if verbose:
                print(f"  Loaded {pair_label}")

    if not datasets:
        raise FileNotFoundError("No CV reconstruction files found.")

    ds_combined = xr.concat(datasets, dim="pair")

    if verbose:
        print(f"\nCombined dataset:")
        print(f"  Dimensions: {dict(ds_combined.sizes)}")
        print(f"  Pairs: {len(pair_names)}")

    return ds_combined

# =============================================================================
# MAIN FIGURE
# =============================================================================
def make_figure():
    """Create the 2-panel comparison figure."""

    # --- Load ensemble-mean metrics (for corner annotation) ---
    ds_mean = xr.open_dataset(PROCESSED_DIR / "cv_reconstruction_ensemble_metrics.nc")
    comparison_df = pd.read_csv(PROCESSED_DIR / "cv_reconstruction_method_comparison.csv")

    # --- Load raw pair-level data for individual points ---
    ds_raw = load_cv_results(verbose=False)
    pair_df = load_pair_level_data(ds_raw)

    # --- Pull ensemble-mean win counts for annotation ---
    def _wins(metric, col):
        row = comparison_df[comparison_df["metric"] == metric]
        return int(row[col].iloc[0]) if not row.empty else "?"

    # --- Figure layout ---
    fig = plt.figure(figsize=(7.2, 3.2), dpi=600)
    gs = gridspec.GridSpec(1, 2, hspace=0.38, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # --- Panels ---
    plot_scatter_panel(
        ax_a, pair_df, "re", "RE", "a",
        ensemble_pca_wins=_wins("re", "pca_wins"),
        ensemble_reg_wins=_wins("re", "reg_wins"),
    )
    plot_scatter_panel(
        ax_b, pair_df, "correlation", "Correlation", "b",
        ensemble_pca_wins=_wins("correlation", "pca_wins"),
        ensemble_reg_wins=_wins("correlation", "reg_wins"),
    )

    # --- Save ---
    outfile = FIGURES_DIR / "Figure_04_cv_reconstruction_comparison.png"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=600, bbox_inches="tight")
    print(f"Figure saved → {outfile}")
    plt.show()
    plt.close()

    


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 80)
    print("PLOTTING RECONSTRUCTION CV COMPARISON")
    print("=" * 80)

    make_figure()

    print("\n" + "=" * 80)
    print("PLOTTING COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()