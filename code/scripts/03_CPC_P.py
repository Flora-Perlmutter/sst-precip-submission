#!/usr/bin/env python
# coding: utf-8
"""
Prepare CPC precipitation data for analysis.

Updated: Flora Perlmutter
Original: Noel Siegert
Revised:  Leah Brown

Description
-----------
CPC data is daily precipitation at 0.25-degree resolution (mm/day).
Longitude is in 0-360 form and latitude is reversed relative to convention.
This script:
  1. Loads CPC data
  2. Resamples from daily to monthly sums
  3. Writes one output file per input year to output_dir

"""

import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import xarray as xr

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
CPC_RAW_DIR   = CMIG_DATA / "Data/Observations/CPC/ppt"
OUTPUT_DIR    = CMIG_DATA / 'fperlmutter/Observational_Regressions_Project/Data/Processed'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def process_one_file(f: str) -> None:
    """Regrid, resample to monthly, mask, and save one CPC annual file."""

    ds = xr.open_dataset(f)

    # Resample to monthly sums
    st_dt = ds.time.values[0]
    precip_monthly = ds.groupby("time.month").sum(dim="time")
    ds_mo = precip_monthly.rename_vars({"precip": "P"}).rename({"month": "time"})
    ds_mo["time"] = pd.date_range(start=st_dt, periods=12, freq="MS")

    # Attributes
    ds_mo.P.attrs["units"] = "mm/month"
    ds_mo.attrs.update(
        {
            "desc":      "monthly sum precipitation",
            "info":      (
                "Daily CPC data resampled to monthly. https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html"
            ),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )

    # Save  ->  P.CPC.<year>.nc
    out_path = OUTPUT_DIR / "P.CPC.{}".format(Path(f).name[-7:])
    ds_mo.to_netcdf(out_path)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Processing CPC precipitation data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(CPC_RAW_DIR / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {CPC_RAW_DIR}")
    print(f"Found {len(files)} files to process.")

    for f in files:
        process_one_file(f)

    print("Done.")


if __name__ == "__main__":
    main()