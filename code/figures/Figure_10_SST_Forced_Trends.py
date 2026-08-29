#!/usr/bin/env python
# coding: utf-8
"""
Figure 10: SST-forced precipitation trends (observed vs AMIP).

Author: Flora Perlmutter

The reconstruction is plotted at the amplitude the method actually produces.
Significance compares each trend against the regression's own sampling error,
taken from the bootstrap distribution the pipeline saves per replicate, so the
magnitude and the uncertainty come from the same fit.

Significance and agreement (two axes, not one verdict)
------------------------------------------------------
Reported as three categories, following IPCC AR5/AR6 map convention:

  significant, robust agreement  no hatching
  significant, low agreement     backslash hatching
  not significant                slash hatching

SIGNIFICANCE compares the ensemble-mean trend to the standard deviation of the
bootstrap distribution of that mean:

    significant if  |theta_hat| > CI_Z * SD(theta_bar_b)

where theta_bar_b averages the member trends WITHIN bootstrap replicate b, and
the SD is taken over b. This is the same construction the paper already uses for
the sensitivity and reconstruction standard errors -- the SD of a bootstrap
distribution -- applied to a third quantity, so it needs neither a percentile
interval nor a bias correction.

Averaging within the replicate is what retains the sampling error the members
share. Drawing replicates independently per member averages that component away
and understates the spread by roughly a factor of two.

AGREEMENT is the fraction of members whose trend sign matches the ensemble mean,
reported on its own axis rather than ANDed into the significance verdict. Three
tests ANDed at nominal 5% each have no stated error rate and the result could
not be described as a 5% procedure.

Why the bootstrap is needed: inter-member spread measures disagreement between
dataset choices, and all 16 members see the same 36-year record. Whatever that
realization got wrong they all get wrong together, so it cancels out of their
spread entirely. On this ensemble the within-member SD is 0.431 against 0.159
between members, so the superseded two-gate scheme used a standard error about
six times too small and called 267 of 518 basins significant.

Observed precipitation has no bootstrap, so those panels keep the inter-member
t-test and are marked with slashes only.

Limitations to state: no correction is made for testing ~518 basins, so roughly
26 false positives are expected at nominal 5%; and replicate b means a different
resampling for most member pairs, which makes the interval mildly lenient.
  Note on the signal-to-noise paradox: AMIP models forced with observed SSTs
  are known to exhibit higher inter-model spread than observed, meaning the
  t-test is conservative for AMIP. Hatched AMIP basins may still have a
  real forced signal that the models underestimate in consistency.
  
  Output
------
  figures/paper_figures/Figure_10_SST_forced_trends.png

"""

import json
import warnings
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import linregress, t as t_dist
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import DATA_DIR, PAPER_FIGURE_DIR, bootstrap_file
from plotting_functions import (
    apply_fixdates_to_results,
    apply_fixdates_to_sst,
    compute_ensemble_mean_sst,
    compute_ensemble_means,
    convert_time_to_years,
    linear_trend,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sensitivity_common import (
    AGREEMENT_THRESHOLD,
    CI_Z,
    ensemble_trend_significance,
    sign_agreement,
    three_categories,
    TREND_PERIODS,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Tuneable parameters
# ---------------------------------------------------------------------------
# Gate 1: t-test across member trends
# Conservative df: obs has 16 independent precip datasets (16 members = 8x2
# combos; the two SST datasets are highly correlated so don't double the dof)
N_INDEPENDENT_OBS  = 16
N_INDEPENDENT_AMIP = 6   # all 6 AMIP models treated as independent
ALPHA              = 0.05  # two-tailed significance level

# Gate 2: sign agreement fraction threshold
SIGN_THRESHOLD_OBS  = 0.75  # ≥ 12/16 independent obs members
SIGN_THRESHOLD_AMIP = 0.75  # ≥ 5/6 AMIP models

# Reported trend window. Was hard-coded in five places in this file.
TREND_PERIOD       = TREND_PERIODS[0]
TREND_PERIOD_LABEL = f"{TREND_PERIOD[0][:4]}-{TREND_PERIOD[1][:4]}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUTS_DIR = DATA_DIR
FIGURES_DIR = PAPER_FIGURE_DIR
BASINS_DIR  = DATA_DIR / "grdc_basins"

# ---------------------------------------------------------------------------
# Load basin boundaries
# ---------------------------------------------------------------------------
grdc_basins = gpd.read_file(BASINS_DIR)

# Load SST anomalies
sst_dict = {}
for sst_name in ["ERSSTv6", "COBE-SST3"]:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        sst_dict[sst_name] = xr.open_dataarray(nc_path).squeeze()

# ============================================================================
# LOAD ALL OBSERVED BOOTSTRAP RESULTS
# ============================================================================
PRECIP_DATASETS = {
    'GPCP': None, 'CRU': None, 'GPCC': None, 'CPC': None,
    'UDel': None, 'PREC': None, 'TerraClimate': None, 'REGEN': None,
}
SST_DATASETS = ['ERSSTv6', 'COBE-SST3']

linear_results = {}

print("Loading bootstrap results...")
for p_name in PRECIP_DATASETS.keys():
    for sst_name in SST_DATASETS:
        output_file = bootstrap_file(p_name, sst_name)
        if os.path.exists(output_file):
            try:
                result_ds = xr.open_dataset(output_file)
                result = {
                    'model_id':              result_ds.attrs['model_id'],
                    'description':           result_ds.attrs['description'],
                    'variable':              result_ds.attrs['variable'],
                    'alpha':                 result_ds.attrs['alpha'],
                    'n_bootstrap':           result_ds.attrs['n_bootstrap'],
                    'reconstruction':        result_ds['reconstruction'],
                    'reconstruction_se':     result_ds['reconstruction_se'],
                    'observed_precip':       result_ds['observed_precip'],
                    'correlation':           result_ds['correlation'],
                    'trend_boot':            result_ds['trend_boot'],
                    'reconstruction_trend':  result_ds['reconstruction_trend'],
                }
                linear_results[(p_name, sst_name)] = result
                print(f"  Loaded: {p_name} vs {sst_name}")
            except Exception as e:
                print(f"  Failed to load {p_name} vs {sst_name}: {e}")

# ============================================================================
# LOAD ALL AMIP BOOTSTRAP RESULTS
# ============================================================================
names_path = OUTPUTS_DIR / "amip_dataset_names.json"
with open(names_path, "r") as f:
    names = json.load(f)

amip_sst_dict = {}
for sst_name in names['sst_datasets']:
    nc_path = OUTPUTS_DIR / f"sst_anom_{sst_name}.nc"
    if os.path.exists(nc_path):
        amip_sst_dict[sst_name] = xr.open_dataarray(nc_path)

AMIP_MODELS = names['precip_datasets']

linear_results_amip = {}

print("Loading AMIP bootstrap results...")
for model_id in AMIP_MODELS:
    output_file = bootstrap_file(model_id, model_id, amip=True)
    if not os.path.exists(output_file):
        print(f"  Not found: {output_file}")
        continue
    try:
        ds = xr.open_dataset(output_file)
        result = {
            'model_id':              ds.attrs.get('model_id', 'P ~ β*SST (AMIP)'),
            'description':           ds.attrs.get('description'),
            'variable':              ds.attrs.get('variable'),
            'alpha':                 ds.attrs.get('alpha'),
            'n_bootstrap':           ds.attrs.get('n_bootstrap'),
            'reconstruction':        ds['reconstruction'],
            'reconstruction_se':     ds['reconstruction_se'],
            'observed_precip':       ds['observed_precip'],
            'correlation':           ds['correlation'],
            'trend_boot':            ds['trend_boot'],
            'reconstruction_trend':  ds['reconstruction_trend'],
        }
        linear_results_amip[(model_id, model_id)] = result
        print(f"  Loaded AMIP: {model_id}")
    except Exception as e:
        print(f"  Failed to load AMIP {model_id}: {e}")

# ============================================================================
# ENSEMBLE MEANS
# ============================================================================
ensemble_obs  = compute_ensemble_means(linear_results,       "Observations")
amip_results_fixed = apply_fixdates_to_results(linear_results_amip)
ensemble_amip = compute_ensemble_means(amip_results_fixed,   "AMIP")


amip_sst_dict  = apply_fixdates_to_sst(amip_sst_dict)
for name, da in amip_sst_dict.items():
    if 'height' in da.coords:
        amip_sst_dict[name] = da.reset_coords('height', drop=True)
amip_sst_mean  = compute_ensemble_mean_sst(amip_sst_dict)
sst_ensemble   = compute_ensemble_mean_sst(sst_dict)

# ============================================================================
# TRENDS FROM ENSEMBLE MEANS
# ============================================================================
def _constrained_trend(ensemble, sst_mean, key, time_slice=TREND_PERIOD):
    da = ensemble[key].sel(time=slice(*time_slice))
    return linear_trend(convert_time_to_years(da))

sst_constrained_amip = amip_sst_mean.sel(time=slice(*TREND_PERIOD))
sst_trend_amip = linear_trend(convert_time_to_years(sst_constrained_amip))

sst_constrained_obs = sst_ensemble.sel(time=slice(*TREND_PERIOD))
sst_trend_obs = linear_trend(convert_time_to_years(sst_constrained_obs))

# The reconstruction is plotted at the amplitude the method actually produces.
precip_sst_trend_amip = _constrained_trend(ensemble_amip, amip_sst_mean, 'sst_reconstruction')
precip_obs_trend_amip = _constrained_trend(ensemble_amip,     amip_sst_mean, 'observed_precip')
precip_sst_trend_obs  = _constrained_trend(ensemble_obs,  sst_ensemble,  'sst_reconstruction')
precip_obs_trend_obs  = _constrained_trend(ensemble_obs,      sst_ensemble,  'observed_precip')



# ============================================================================
# TWO-GATE SIGNIFICANCE
# ============================================================================
def compute_member_trends(results_dict, data_key='reconstruction',
                          time_slice=TREND_PERIOD,
                          member_keys=None):
    """
    Return a (n_members × n_basins) array of linear trends and the common
    basin coordinate array.  Members are aligned to the intersection of their
    basin coordinates before stacking to handle datasets with different spatial
    coverage.

    Parameters
    ----------
    results_dict : dict
    data_key     : str   — 'reconstruction' or 'observed_precip'
    time_slice   : tuple of str
    member_keys  : list or None — subset of keys (used in bootstrap test)

    Returns
    -------
    trends_arr : np.ndarray (n_members, n_basins)
    basins     : xr.DataArray coordinate
    """
    keys = member_keys if member_keys is not None else list(results_dict.keys())

    member_trend_das = []
    for key in keys:
        result = results_dict[key]
        da     = result[data_key].sel(time=slice(*time_slice))
        trend  = linear_trend(convert_time_to_years(da))
        member_trend_das.append(trend)

    # Intersect basin coordinates across all members
    common_basins = member_trend_das[0].coords['basin'].values
    for trend_da in member_trend_das[1:]:
        common_basins = np.intersect1d(common_basins, trend_da.coords['basin'].values)

    if len(common_basins) == 0:
        raise ValueError(
            f"No basins in common across all ensemble members for key='{data_key}'."
        )

    n_dropped = len(member_trend_das[0].coords['basin']) - len(common_basins)
    if n_dropped > 0:
        print(f"    [compute_member_trends] Aligning to {len(common_basins)} common basins "
              f"({n_dropped} dropped due to missing coverage in ≥1 member).")

    stacked = np.stack(
        [t.sel(basin=common_basins).values for t in member_trend_das], axis=0
    )
    basins = xr.DataArray(common_basins, dims=['basin'], name='basin')
    return stacked, basins


def calculate_two_gate_significance(results_dict, data_key='reconstruction',
                                    time_slice=TREND_PERIOD,
                                    n_independent=16,
                                    sign_threshold=0.75,
                                    alpha=ALPHA,
                                    member_keys=None):
    """
    Two-gate significance test applied per basin across ensemble members.

    Gate 1 — t-test across member trends:
        t = mean(trends) / (std(trends) / sqrt(n_independent))
        df = n_independent - 1
        Basin passes if two-tailed p-value < alpha.

    Gate 2 — sign agreement:
        Fraction of members whose trend sign matches the ensemble mean sign.
        Basin passes if fraction >= sign_threshold.

    A basin is significant (not hatched) only when it passes BOTH gates.

    Parameters
    ----------
    results_dict   : dict  — keyed by member identifier
    data_key       : str   — 'reconstruction' or 'observed_precip'
    time_slice     : tuple of str
    n_independent  : int   — conservative degrees of freedom for the t-test
                             (16 for obs, 6 for AMIP)
    sign_threshold : float — minimum fraction of members agreeing on sign
                             (0.75 for obs, 0.75 for AMIP)
    alpha          : float — two-tailed significance level for Gate 1
    member_keys    : list or None — subset of keys (used for bootstrap test)

    Returns
    -------
    hatch_mask : xr.DataArray (basin,) bool — True where NOT significant
    gate1_pass : xr.DataArray (basin,) bool — True where Gate 1 passes
    gate2_pass : xr.DataArray (basin,) bool — True where Gate 2 passes
    """
    trends_arr, basins = compute_member_trends(
        results_dict, data_key=data_key,
        time_slice=time_slice, member_keys=member_keys
    )
    n_members = trends_arr.shape[0]

    # ── Gate 1: t-test across member trends ──────────────────────────────
    mean_trend = np.mean(trends_arr, axis=0)
    std_trend  = np.std(trends_arr,  axis=0, ddof=1)
    df         = n_independent - 1

    with np.errstate(invalid='ignore', divide='ignore'):
        se     = std_trend / np.sqrt(n_independent)
        t_stat = np.where(se > 0, mean_trend / se, 0.0)

    # Two-tailed p-value
    p_values   = 2 * t_dist.sf(np.abs(t_stat), df=df)
    gate1_pass = p_values < alpha

    # ── Gate 2: sign agreement ────────────────────────────────────────────
    signs          = np.sign(trends_arr)                        # (n_members, n_basins)
    ensemble_sign  = np.sign(mean_trend)                        # (n_basins,)
    agree_frac     = np.where(
        ensemble_sign == 0,
        0.5,
        np.mean(signs == ensemble_sign[np.newaxis, :], axis=0)
    )
    gate2_pass = agree_frac >= sign_threshold

    # ── Combined: must pass both ──────────────────────────────────────────
    sig        = gate1_pass & gate2_pass
    hatch_mask = ~sig

    coords = {'basin': basins}
    return (
        xr.DataArray(hatch_mask,  coords=coords, dims=['basin'], name='hatch_mask'),
        xr.DataArray(gate1_pass,  coords=coords, dims=['basin'], name='gate1_ttest'),
        xr.DataArray(gate2_pass,  coords=coords, dims=['basin'], name='gate2_sign'),
    )

def report_significance(label, point, sd, agree, gate1=None, gate2=None,
                        threshold=AGREEMENT_THRESHOLD):
    """Detailed per-criterion counts, so the cost of each test is visible."""
    sig    = np.abs(point) > CI_Z * sd
    robust = agree >= threshold
    n      = int(np.isfinite(point).sum())

    print(f"\n  {label}  ({n} basins)")
    print(f"    median trend                : {float(abs(point).median()):.4f}")
    print(f"    median SD of ensemble mean  : {float(sd.median()):.4f}")
    print(f"    significant (|t| > {CI_Z})     : {int(sig.sum()):4d}"
          f"  ({100*float(sig.mean()):5.1f}%)")
    print(f"    sign agreement >= {threshold:.0%}      : {int(robust.sum()):4d}"
          f"  ({100*float(robust.mean()):5.1f}%)")
    # Categories come from the shared helper so the figures and the notebook
    # cannot drift on what "robust" means.
    cat = three_categories(sig, agree, threshold)
    print(f"    -> significant AND robust   : {int((cat == 2).sum()):4d}")
    print(f"    -> significant, low agree   : {int((cat == 1).sum()):4d}")
    print(f"    -> not significant          : {int((cat == 0).sum()):4d}")

    if gate1 is not None and gate2 is not None:
        old = gate1 & gate2
        common = np.intersect1d(sig['basin'].values, old['basin'].values)
        s, o = sig.sel(basin=common), old.sel(basin=common)
        print(f"    superseded two-gate scheme  : {int(o.sum()):4d}")
        print(f"      kept by both              : {int((s & o).sum()):4d}")
        print(f"      lost when adding the SD   : {int((~s & o).sum()):4d}")
        print(f"      gained                    : {int((s & ~o).sum()):4d}")
    return sig, robust


print("\nSignificance: ensemble-mean trend vs the SD of its bootstrap distribution")
print(f"  z = {CI_Z}, agreement threshold = {AGREEMENT_THRESHOLD:.0%}, "
      f"period {TREND_PERIOD_LABEL}")
print("  Reconstruction panels use the bootstrap; observed precipitation has no")
print("  bootstrap and keeps the inter-member t-test.")

# --- reconstruction panels: the bootstrap significance test ------------------
pt_sst_obs, sd_sst_obs, sig_sst_obs = ensemble_trend_significance(
    linear_results, TREND_PERIOD_LABEL)
agree_sst_obs = sign_agreement(linear_results, TREND_PERIOD_LABEL)

pt_sst_amip, sd_sst_amip, sig_sst_amip = ensemble_trend_significance(
    linear_results_amip, TREND_PERIOD_LABEL)
agree_sst_amip = sign_agreement(linear_results_amip, TREND_PERIOD_LABEL)

# The old two-gate result, computed only so the comparison can be printed.
_h, g1_sst_obs, g2_sst_obs = calculate_two_gate_significance(
    linear_results, data_key='reconstruction',
    n_independent=N_INDEPENDENT_OBS, sign_threshold=SIGN_THRESHOLD_OBS)
_h, g1_sst_amip, g2_sst_amip = calculate_two_gate_significance(
    linear_results_amip, data_key='reconstruction',
    n_independent=N_INDEPENDENT_AMIP, sign_threshold=SIGN_THRESHOLD_AMIP)

report_significance("Obs  - SST-forced", pt_sst_obs, sd_sst_obs,
                    agree_sst_obs, g1_sst_obs, g2_sst_obs)
report_significance("AMIP - SST-forced", pt_sst_amip, sd_sst_amip,
                    agree_sst_amip, g1_sst_amip, g2_sst_amip)


# Hatching: slashes where not significant, backslashes where significant but
# the datasets disagree on sign.
hatch_sst_obs   = ~sig_sst_obs
lowagr_sst_obs  = sig_sst_obs & (agree_sst_obs < AGREEMENT_THRESHOLD)
hatch_sst_amip  = ~sig_sst_amip
lowagr_sst_amip = sig_sst_amip & (agree_sst_amip < AGREEMENT_THRESHOLD)

# --- observed precipitation: no bootstrap exists, so the two gates stand -----
hatch_obs_obs,  _, _ = calculate_two_gate_significance(
    linear_results, data_key='observed_precip',
    n_independent=N_INDEPENDENT_OBS, sign_threshold=SIGN_THRESHOLD_OBS)

hatch_obs_amip, _, _ = calculate_two_gate_significance(
    linear_results_amip, data_key='observed_precip',
    n_independent=N_INDEPENDENT_AMIP, sign_threshold=SIGN_THRESHOLD_AMIP)


print(f"\n  Obs  - observed precip (two-gate) : {int((~hatch_obs_obs).sum())} significant")
print(f"  AMIP - modelled precip (two-gate) : {int((~hatch_obs_amip).sum())} significant")

# ============================================================================
# HELPER: merge hatch mask into GeoDataFrame
# ============================================================================
def merge_hatch(gdf, hatch_mask_da, column="hatch"):
    # Strip non-dimension coordinates before the frame conversion. Anything
    # left attached becomes a column, and merging a third mask then collides:
    # pandas has already used the _x and _y suffixes on the first two.
    hatch_mask_da = hatch_mask_da.drop_vars(
        [c for c in hatch_mask_da.coords if c not in hatch_mask_da.dims]
    )
    hatch_df = hatch_mask_da.to_dataframe(name=column).reset_index()
    merged   = gdf.merge(hatch_df, left_on='MRBID', right_on='basin', how='left')
    cols = merged.columns.tolist()
    if 'basin_x' in cols:
        merged = merged.rename(columns={'basin_x': 'basin'}).drop(columns='basin_y')
    # Basins in the shapefile but absent from the mask merge to NaN. NaN is
    # truthy, so leaving it would silently mark them as failing the test rather
    # than as having no test at all.
    merged[column] = merged[column].fillna(False).astype(bool)
    return gpd.GeoDataFrame(merged, geometry='geometry')


# ============================================================================
# BUILD GeoDataFrames
# ============================================================================
def _make_gdf(trend_da, hatch_mask_da, name="P", lowagr_da=None):
    trend_da = trend_da.copy()
    trend_da = trend_da.drop_vars(
        [c for c in trend_da.coords if c not in trend_da.dims]
    )
    trend_da.name = name
    df  = trend_da.to_dataframe().reset_index()
    gdf = gpd.GeoDataFrame(
        grdc_basins.merge(df, left_on='MRBID', right_on='basin', how='left'),
        geometry='geometry'
    )
    gdf = merge_hatch(gdf, hatch_mask_da)
    # Second mark: significant, but the datasets disagree on the sign.
    if lowagr_da is not None:
        gdf = merge_hatch(gdf, lowagr_da, column='lowagr')
    else:
        gdf['lowagr'] = False
    return gdf

precip_sst_gdf_amip = _make_gdf(precip_sst_trend_amip, hatch_sst_amip, lowagr_da=lowagr_sst_amip)
precip_obs_gdf_amip = _make_gdf(precip_obs_trend_amip, hatch_obs_amip)
precip_sst_gdf_obs  = _make_gdf(precip_sst_trend_obs,  hatch_sst_obs,  lowagr_da=lowagr_sst_obs)
precip_obs_gdf_obs  = _make_gdf(precip_obs_trend_obs,  hatch_obs_obs)


# ============================================================================
# COLOUR MAPS AND NORMS
# ============================================================================
sst_levels        = np.linspace(-0.5, 0.5, 21)
precip_sst_levels = np.linspace(-2,   2,   21)
precip_levels     = np.linspace(-4,   4,   21)

def make_cmap(base_name, levels):
    raw    = plt.get_cmap(base_name, len(levels) + 1)
    colors = raw(np.arange(len(levels) + 1))
    cmap   = ListedColormap(colors[1:-1])
    cmap.set_under(colors[0])
    cmap.set_over(colors[-1])
    return cmap

sst_cmap        = make_cmap("RdBu_r", sst_levels)
precip_sst_cmap = make_cmap("BrBG",   precip_sst_levels)
precip_cmap     = make_cmap("BrBG",   precip_levels)

sst_norm        = BoundaryNorm(sst_levels,        sst_cmap.N)
precip_sst_norm = BoundaryNorm(precip_sst_levels, precip_sst_cmap.N)
precip_norm     = BoundaryNorm(precip_levels,     precip_cmap.N)


# ============================================================================
# PLOTTING HELPER
# ============================================================================
def plot_basin_choropleth(ax, gdf, cmap, norm, hatch_col='hatch',
                          lowagr_col='lowagr'):
    for _, row in gdf.iterrows():
        val   = row["P"]
        color = cmap(norm(val)) if np.isfinite(val) else "lightgray"
        ax.add_geometries(
            [row.geometry], crs=ccrs.PlateCarree(),
            facecolor=color, edgecolor="black", linewidth=0.2
        )
        if row.get(hatch_col, False):
            ax.add_geometries(
                [row.geometry], crs=ccrs.PlateCarree(),
                facecolor='none', edgecolor='black',
                linewidth=0.05, hatch='////////', alpha=0.8
            )
        # Backslashes: distinguishable from zero, but fewer than
        # AGREEMENT_THRESHOLD of the members agree on the sign.
        if row.get(lowagr_col, False):
            ax.add_geometries(
                [row.geometry], crs=ccrs.PlateCarree(),
                facecolor='none', edgecolor='0.25',
                linewidth=0.05, hatch=r'\\\\\\\\', alpha=0.8
            )


# ============================================================================
# FIGURE
# ============================================================================
plt.rcParams.update({'font.size': 7})
plt.rcParams['hatch.linewidth'] = 0.4

fig = plt.figure(figsize=(6.2, 8.8), dpi=600)
gs  = fig.add_gridspec(nrows=4, ncols=2, hspace=0.35, wspace=0.3)

# ── Row 1: SST trends ──────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=180))
ax1.set_title("Observed SST Trend")
ax1.set_global()
ax1.pcolormesh(sst_trend_obs.lon, sst_trend_obs.lat, sst_trend_obs,
               transform=ccrs.PlateCarree(), cmap=sst_cmap, norm=sst_norm)
ax1.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax1.coastlines(zorder=3, linewidth=.3)
sm1 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap); sm1.set_array([])
cb1 = fig.colorbar(sm1, ax=ax1, ticks=np.linspace(-0.5, 0.5, 5),
                   orientation='horizontal', extend='max', shrink=0.7, pad=0.05)
cb1.set_label(r"K decade$^{-1}$", labelpad=2); cb1.ax.minorticks_off()

ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson(central_longitude=180))
ax2.set_title("AMIP SST Trend")
ax2.set_global()
ax2.pcolormesh(sst_trend_amip.lon, sst_trend_amip.lat, sst_trend_amip,
               transform=ccrs.PlateCarree(), cmap=sst_cmap, norm=sst_norm)
ax2.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax2.coastlines(zorder=3, linewidth=.3)
sm2 = plt.cm.ScalarMappable(norm=sst_norm, cmap=sst_cmap); sm2.set_array([])
cb2 = fig.colorbar(sm2, ax=ax2, ticks=np.linspace(-0.5, 0.5, 5),
                   orientation='horizontal', extend='max', shrink=0.7, pad=0.05)
cb2.set_label("K decade$^{-1}$", labelpad=2); cb2.ax.minorticks_off()

# ── Row 2: Observed precipitation trends ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
ax3.set_title("Observed Precipitation Trend")
ax3.set_global()
plot_basin_choropleth(ax3, precip_obs_gdf_obs, precip_cmap, precip_norm)
ax3.add_feature(cfeature.LAND, facecolor="white", zorder=0)
ax3.coastlines(zorder=3, linewidth=.3)
sm3 = plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap); sm3.set_array([])
cb3 = fig.colorbar(sm3, ax=ax3, ticks=np.linspace(-4, 4, 5),
                   orientation='horizontal', extend='both', shrink=0.7, pad=0.05)
cb3.set_label(r"mm month$^{-1}$ decade$^{-1}$", labelpad=2); cb3.ax.minorticks_off()

ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
ax4.set_title("AMIP Precipitation Trend")
ax4.set_global()
plot_basin_choropleth(ax4, precip_obs_gdf_amip, precip_cmap, precip_norm)
ax4.add_feature(cfeature.LAND, facecolor="white", zorder=0)
ax4.coastlines(zorder=3, linewidth=.3)
sm4 = plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap); sm4.set_array([])
cb4 = fig.colorbar(sm4, ax=ax4, ticks=np.linspace(-4, 4, 5),
                   orientation='horizontal', extend='both', shrink=0.7, pad=0.05)
cb4.set_label(r"mm month$^{-1}$ decade$^{-1}$", labelpad=2); cb4.ax.minorticks_off()

# ── Row 3: SST-forced precipitation trends ────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0], projection=ccrs.Robinson())
ax5.set_title("SST-Forced Precipitation Trend in Obs")
ax5.set_global()
plot_basin_choropleth(ax5, precip_sst_gdf_obs, precip_sst_cmap, precip_sst_norm)
ax5.add_feature(cfeature.LAND, facecolor="white", zorder=0)
ax5.coastlines(zorder=3, linewidth=.3)
sm5 = plt.cm.ScalarMappable(norm=precip_sst_norm, cmap=precip_sst_cmap); sm5.set_array([])
cb5 = fig.colorbar(sm5, ax=ax5, ticks=np.linspace(-2, 2, 5),
                   orientation='horizontal', extend='both', shrink=0.7, pad=0.05)
cb5.set_label(r"mm month$^{-1}$ decade$^{-1}$", labelpad=2); cb5.ax.minorticks_off()

ax6 = fig.add_subplot(gs[2, 1], projection=ccrs.Robinson())
ax6.set_title("SST-Forced Precipitation Trend in AMIP")
ax6.set_global()
plot_basin_choropleth(ax6, precip_sst_gdf_amip, precip_sst_cmap, precip_sst_norm)
ax6.add_feature(cfeature.LAND, facecolor="white", zorder=0)
ax6.coastlines(zorder=3, linewidth=.3)
sm6 = plt.cm.ScalarMappable(norm=precip_sst_norm, cmap=precip_sst_cmap); sm6.set_array([])
cb6 = fig.colorbar(sm6, ax=ax6, ticks=np.linspace(-2, 2, 5),
                   orientation='horizontal', extend='max', shrink=0.7, pad=0.05)
cb6.set_label(r"mm month$^{-1}$ decade$^{-1}$", labelpad=2); cb6.ax.minorticks_off()

# ── Row 4: Scatter plots ───────────────────────────────────────────────────
ax7 = fig.add_subplot(gs[3, 0])
obs_data  = precip_sst_gdf_obs[['MRBID', 'P']].merge(
    precip_obs_gdf_obs[['MRBID', 'P']], on='MRBID', suffixes=('_sst', '_obs'))
obs_valid = obs_data.dropna()
ax7.scatter(obs_valid['P_sst'], obs_valid['P_obs'],
            alpha=0.6, s=10, edgecolors='black', linewidth=0.5)
ax7.axhline(y=0, color='grey', alpha=0.5, linestyle='-', linewidth=1)
ax7.set_xlim(-5, 10); ax7.set_ylim(-12, 27)
lims = [min(ax7.get_xlim()[0], ax7.get_ylim()[0]),
        max(ax7.get_xlim()[1], ax7.get_ylim()[1])]
ax7.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='1:1 line')
ax7.set_xlabel(r"SST-Forced Trend in Obs (mm month$^{-1}$ decade$^{-1}$)")
ax7.set_ylabel(r"Observed Trend (mm month$^{-1}$ decade$^{-1}$)")
ax7.grid(False); ax7.legend(fontsize=5)

ax8 = fig.add_subplot(gs[3, 1])
amip_data  = precip_sst_gdf_amip[['MRBID', 'P']].merge(
    precip_obs_gdf_amip[['MRBID', 'P']], on='MRBID', suffixes=('_sst', '_obs'))
amip_valid = amip_data.dropna()
ax8.scatter(amip_valid['P_sst'], amip_valid['P_obs'],
            alpha=0.6, s=10, edgecolors='black', linewidth=0.5)
ax8.axhline(y=0, color='grey', alpha=0.5, linestyle='-', linewidth=1)
ax8.set_xlim(-5, 10); ax8.set_ylim(-12, 27)
lims = [min(ax8.get_xlim()[0], ax8.get_ylim()[0]),
        max(ax8.get_xlim()[1], ax8.get_ylim()[1])]
ax8.plot(lims, lims, 'k--', alpha=0.3, zorder=0, label='1:1 line')
ax8.set_xlabel(r"SST-Forced Trend in AMIP (mm month$^{-1}$ decade$^{-1}$)")
ax8.set_ylabel(r"AMIP Trend (mm month$^{-1}$ decade$^{-1}$)")
ax8.grid(False); ax8.legend(fontsize=5)

# ── Panel labels ──────────────────────────────────────────────────────────
for ax, label in zip([ax1, ax2, ax3, ax4, ax6, ax7, ax8],
                     ['a', 'b', 'c', 'd', 'f', 'g', 'h']):
    ax.text(-0.1, 1.16, label, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.set_aspect('auto')

ax5.text(-0.15, 1.16, 'e', transform=ax5.transAxes,
         fontsize=10, fontweight='bold', va='top')
ax5.set_aspect('auto')

plt.tight_layout()
plt.savefig(FIGURES_DIR / "Figure_10_SST_forced_trends.png",
            dpi=600,  pad_inches=0.1)
print("\nFigure saved.")
plt.close()

