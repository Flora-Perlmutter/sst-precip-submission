#!/usr/bin/env python
# coding: utf-8
"""
Prepare GPCP precipitation data for analysis.

Updated: Flora Perlmutter
Original: Noel Siegert

Description
-----------
GPCP v2.3 monthly data at 2.5-degree resolution (mm/day), spanning 1979-2025.
This script:
  1. Opens the GPCP monthly file
  2. Fixes the time dimension
  3. Writes the dataset to the output directory

"""

import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import os
import numpy as np
import pandas as pd
import xarray as xr
import cftime

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
GPCP_RAW_DIR   = CMIG_DATA / 'Data/Observations/GPCP/precip.mon.mean_1979_2026.nc'

OUTPUT_DIR     = CMIG_DATA / 'fperlmutter/Observational_Regressions_Project/Data/Processed'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Processing GPCC precipitation data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #  Load without decoding times
    gpcp = xr.open_dataset(GPCP_RAW_DIR, decode_times=False)
    
    #  Fix the time dimension
    # Units are 'days since 1800-1-1 00:00:0.0', which is not compatible with decode_times
    gpcp["time"] = pd.date_range("1979-01-01", periods=len(gpcp.time), freq="MS")

    # Drop time_bnds entirely
    if "time_bnds" in gpcp:
        gpcp = gpcp.drop_vars("time_bnds")
        gpcp["time"].attrs.pop("bounds", None)  # remove the bounds reference

    #  Attributes
    gpcp.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    #  Save
    out_path = OUTPUT_DIR / "P.GPCP.1979.2025.nc"
    gpcp.to_netcdf(out_path)
    print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()