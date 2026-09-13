#!/usr/bin/env python
# coding: utf-8
"""
Diagnostic: per-dataset time coverage for the stability ensemble.

Author: Flora Perlmutter

Description
-----------
For the twelve (precip, SST) pairs used by
15_Stability_of_the_marginal_sensitivity.py (USE_LONGER_PERIOD=True), prints:
  - each individual dataset's own first/last available month
  - each pair's full-period record (>= ANALYSIS_START) -- what that member's
    full-period reference sensitivity would be computed over
  - the true common period across all twelve members -- what the rolling
    windows are built from
  - whether the ANALYSIS_START floor (1979-01) is actually binding, or
    whether some member's record starts later (e.g. 1980-01)
  - how many complete 30-year rolling windows the common period supports,
    and their "last year" range

Run this on Discovery before regenerating Figure 5 to confirm the ensemble
matches its manuscript description ("twelve ensemble members with data from
at least 1980-2024, eighteen 30-year rolling windows").

Usage
-----
    python check_stability_ensemble_coverage.py
"""

import sys
from functools import reduce
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

# ---------------------------------------------------------------------------
# Config -- mirrors 15_Stability_of_the_marginal_sensitivity.py, USE_LONGER_PERIOD=True
# ---------------------------------------------------------------------------
INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"

WINDOW_SIZE = 30  # years
WINDOW_STEP = 12  # months
ANALYSIS_START = "1979-01-01"

PRECIP_DATASETS_TO_USE = ['GPCP', 'CRU', 'GPCC', 'CPC', 'PREC', 'TerraClimate']
SST_DATASETS_TO_USE = ['ERSSTv6', 'COBE-SST3']


def month_str(dt64):
    return str(dt64)[:7]


def load(name, kind):
    path = INPUTS_DIR / f"{kind}_anom_{name}.nc"
    da = xr.open_dataarray(path)
    if kind == "sst":
        da = da.squeeze()
    return da


# ---------------------------------------------------------------------------
# Per-dataset coverage
# ---------------------------------------------------------------------------
print("=" * 80)
print("PER-DATASET TIME COVERAGE")
print("=" * 80)

precip_dict = {}
for p_name in PRECIP_DATASETS_TO_USE:
    da = load(p_name, "precip")
    precip_dict[p_name] = da
    print(f"  {p_name:14s}: {month_str(da.time.values[0])} to {month_str(da.time.values[-1])}"
          f"  ({da.time.size} months)")

sst_dict = {}
for s_name in SST_DATASETS_TO_USE:
    da = load(s_name, "sst")
    sst_dict[s_name] = da
    print(f"  {s_name:14s}: {month_str(da.time.values[0])} to {month_str(da.time.values[-1])}"
          f"  ({da.time.size} months)")

# ---------------------------------------------------------------------------
# Per-pair full-period record
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print(f"PER-PAIR FULL-PERIOD RECORD (>= {ANALYSIS_START[:7]})")
print("=" * 80)

for s_name, s_da in sst_dict.items():
    for p_name, p_da in precip_dict.items():
        common = np.intersect1d(p_da.time.values, s_da.time.values)
        common = common[common >= np.datetime64(ANALYSIS_START)]
        print(f"  {p_name:14s} x {s_name:10s}: {month_str(common[0])} to {month_str(common[-1])}"
              f"  ({len(common)} months)")

# ---------------------------------------------------------------------------
# Twelve-member common period (what the rolling windows use)
# ---------------------------------------------------------------------------
print()
print("=" * 80)
print("TWELVE-MEMBER COMMON PERIOD (rolling-window grid)")
print("=" * 80)

common_time = reduce(
    np.intersect1d,
    [p.time.values for p in precip_dict.values()] + [s.time.values for s in sst_dict.values()],
)

print(f"  True earliest common month across all 12 members: {month_str(common_time[0])}")
print(f"  True latest common month across all 12 members:   {month_str(common_time[-1])}")
print()

if common_time[0] > np.datetime64(ANALYSIS_START):
    print(f"  NOTE: the true common start ({month_str(common_time[0])}) is LATER than "
          f"ANALYSIS_START ({ANALYSIS_START[:7]}).")
    print("  At least one of the 12 datasets does not have data back to "
          f"{ANALYSIS_START[:7]} -- check the per-dataset coverage above to see which.")
    print("  15_Stability_of_the_marginal_sensitivity.py needs no code change for this: "
          "the ANALYSIS_START value is only a floor, so the true intersection "
          "(and the dynamically-set TIME_PERIOD attribute) will already reflect "
          f"{month_str(common_time[0])} as the real start.")
else:
    print(f"  All 12 datasets cover {ANALYSIS_START[:7]} or earlier -- "
          f"the {ANALYSIS_START[:7]} floor is not binding.")

common_time_floored = common_time[common_time >= np.datetime64(ANALYSIS_START)]
n_times = len(common_time_floored)
window_months = WINDOW_SIZE * 12
n_windows = (n_times - window_months) // WINDOW_STEP + 1

print()
print(f"  Common period used for rolling windows: {month_str(common_time_floored[0])} to "
      f"{month_str(common_time_floored[-1])}  ({n_times} months)")
print(f"  Number of complete {WINDOW_SIZE}-year rolling windows (step={WINDOW_STEP} mo): {n_windows}")

if n_windows > 0:
    last_years = []
    for window_idx in range(n_windows):
        window_start = window_idx * WINDOW_STEP
        window_end = window_start + window_months
        last_years.append(int(str(common_time_floored[window_end - 1])[:4]))
    print(f"  Window 'last year' range: {last_years[0]} to {last_years[-1]}")
    print(f"  All window last-years: {last_years}")
