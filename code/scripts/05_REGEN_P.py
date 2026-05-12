#!/usr/bin/env python
# coding: utf-8
"""
Prepare REGEN precipitation data for analysis.

Updated: Flora Perlmutter
Original: Leah Brown

Description
-----------
REGEN AllStns V1-2019 daily precipitation at 1.0-degree resolution (mm/day),
spanning 1950-2016. This script:
  1. Opens and concatenates all REGEN daily files
  2. Resamples from monthly to daily data
  3. Writes the dataset to the output directory

"""

import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import xarray as xr

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
REGEN_RAW_DIR  = CMIG_DATA / "Data/Observations/REGEN/AllStns"
OUTPUT_DIR     = CMIG_DATA / 'fperlmutter/Observational_Regressions_Project/Data/Processed'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Processing REGEN precipitation data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all REGEN daily files
    files = sorted(glob.glob(str(REGEN_RAW_DIR / "REGEN_AllStns_V1-2019_*.nc")))
    if not files:
        raise FileNotFoundError(f"No REGEN files found in {REGEN_RAW_DIR}")
    print(f"Found {len(files)} files to process.")

    regen = xr.open_mfdataset(files)

    # Isolate precipitation and resample to monthly totals
    regen_P      = xr.Dataset({"P": regen.p})
    monthly_sum  = xr.Dataset({"P": regen_P["P"].resample(time="1MS").sum()})

    # Attributes
    monthly_sum.P.attrs["desc"]  = "Total monthly precipitation"
    monthly_sum.P.attrs["units"] = "mm/month"
    monthly_sum.attrs["info"]    = (
        "Monthly REGEN AllStns V1-2019 data, 1950-2016. "
        "Originally daily at 1.0-deg resolution, resampled to monthly"
    )
    monthly_sum.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save
    out_path = OUTPUT_DIR / "P.REGEN.1950-2016.nc"
    monthly_sum.to_netcdf(out_path)
    print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()