#!/usr/bin/env python
# coding: utf-8
"""
Figure 1: Model rankings by cross-validation metrics.

Author: Flora Perlmutter

Description
-----------
Reads pre-processed model rank files (produced by process_cv_results.py) and
produces boxplots showing the distribution of per-basin ranks for each metric.

Input
-----
  <PROCESSED_DIR>/ranks_rmse_mean.nc
  <PROCESSED_DIR>/ranks_adjr2_mean.nc

Output
------
  figures/paper_figures/Figure_1_cross_validation.png  (repo-tracked)

"""

import xarray as xr
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import MaxNLocator
import cartopy.feature as cfeature
import regionmask
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns
import xskillscore
from matplotlib.cm import RdBu
from scipy.stats import linregress
from scipy import stats
from scipy.stats import t
import pickle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
import warnings
import glob
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import PAPER_FIGURE_DIR, DATA_DIR

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIGURES_DIR   = PAPER_FIGURE_DIR
PROCESSED_DIR = DATA_DIR

def plot_model_rankings(cv_ranks, figures_dir, metrics=None, verbose=True):
    """
    Create boxplots of model rankings.

    Args:
        cv_ranks: dict of DataFrames returned by model_ranking()
        figures_dir: Directory to save figures
        metrics: metrics to plot (keys of cv_ranks)
        verbose: Print info
    """

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------
    if metrics is None:
        metrics = list(cv_ranks.keys())

    metric_info = {
        'rmse_mean': {
            'label': 'RMSE Rank',
            'title': 'Model Ranking by RMSE'
        },
        'adjr2_mean': {
            'label': 'Adjusted R² Rank',
            'title': 'Model Ranking by Adjusted R²'
        }
    }

    n_metrics = len(metrics)

    # --------------------------------------------------
    # Figure setup
    # --------------------------------------------------
    plt.rcParams.update({'font.size': 7})
    plt.rcParams['axes.titlesize']='large'
    
    fig, axes = plt.subplots(
        n_metrics, 1,
        figsize=(6.27, 7.1),
        dpi=600,
        sharex=True
    )

    if n_metrics == 1:
        axes = [axes]

    fig.subplots_adjust(hspace=0.14)


    # --------------------------------------------------
    # Plot each metric
    # --------------------------------------------------
    for idx, metric in enumerate(metrics):

        ax = axes[idx]

        metric_config = metric_info.get(metric, {
            'label': metric,
            'title': f'Model Ranking by {metric}'
        })

        try:
            ranks_df = cv_ranks[metric]

            if verbose:
                print(f"\nProcessing {metric}")
                print(f"  shape: {ranks_df.shape}")

            # --------------------------------------------------
            # Plot
            # --------------------------------------------------
            # Convert to long format for seaborn
            ranks_long = ranks_df.melt(
                var_name="Model",
                value_name="Rank"
            )
            
            sns.boxplot(
                data=ranks_long,
                x="Model",
                y="Rank",
                ax=ax,
                width=0.6,
                linewidth=0.5,
                fliersize=1,
                color="lightblue",
            )
            ax.set_ylabel(metric_config['label'])
            ax.set_title(metric_config['title'])

            n_models = len(ranks_df.columns)

            ax.set_xticks(range(1, n_models))
            ax.set_xticklabels(
                ranks_df.columns[1:],
                rotation=45,
                ha='right'
            )
            
            ax.set_xlim(.5, n_models - .5)
            ax.set_ylim(0, n_models)
            
            ax.grid(False)

            if idx == n_metrics - 1:
                ax.set_xlabel('Model')
            else:
                ax.set_xlabel('')

        except Exception as e:
            print(f"ERROR processing {metric}: {e}")
            ax.text(
                0.5, 0.5,
                f"Error: {metric}",
                ha='center',
                va='center',
                transform=ax.transAxes
            )
            

    # --------------------------------------------------
    # Panel labels
    # --------------------------------------------------
    for ax, label in zip(fig.axes, ['a', 'b']):
        ax.text(
            -.037, 1.11, label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top'
        )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    plt.tight_layout()

    outfile = FIGURES_DIR / 'Figure_1_cross_validation.png'
    plt.savefig(outfile, dpi=600, bbox_inches='tight')
    print(f"Figure saved to: {outfile}")

    plt.show()
    plt.close()
    


# In[22]:


def main():
    print("\n" + "="*80)
    print("CV RESULTS PLOTTING")
    print("="*80)

    ranks_rmse_path = PROCESSED_DIR / 'cross_validation_ranks_rmse_mean.csv'
    ranks_adjr2_path = PROCESSED_DIR / 'cross_validation_ranks_adjr2_mean.csv'
    cv_rmse_ranks = pd.read_csv(ranks_rmse_path)
    cv_adjr2_ranks = pd.read_csv(ranks_adjr2_path)
    
    cv_ranks = {}
    cv_ranks['rmse_mean']=cv_rmse_ranks
    cv_ranks['adjr2_mean']=cv_adjr2_ranks

    print(f"\nCreating plots...")
    print("="*80)
    
    plot_model_rankings(cv_ranks, FIGURES_DIR, verbose=True)
    
    print("\n" + "="*80)
    print("PLOTTING COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
