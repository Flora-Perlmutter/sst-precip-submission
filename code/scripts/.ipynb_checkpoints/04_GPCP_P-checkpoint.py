#!/usr/bin/env python
# coding: utf-8
"""
Prepare GPCC precipitation data for analysis.

Original: Leah Brown, 06/13/2025
Refactored for reproducibility: February 24, 2026

Description
-----------
GPCC v2022 monthly data at 1.0-degree resolution (mm/month), spanning 1891-2020.
This script:
  1. Opens and concatenates all GPCC monthly files
  2. Regrids from 1.0-degree to 0.5-degree using the project reference grid
  3. Masks out ocean using the ERA5 land fraction mask
  4. Writes the masked dataset to the output directory

Note: 0.25-degree GPCC data (1991-2020) was considered but not used here,
as the 1.0-degree product covers the full 1891-2020 period needed.

Usage
-----
    python GPCC_P.py
"""

import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import xarray as xr
import xesmf as xe

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# A collaborator replaces these to match their own directory structure
# ---------------------------------------------------------------------------
GPCC_RAW_DIR   = CMIG_DATA / "Data/Observations/GPCC/v2022_monthly/10"
REF_GRID_FILE  = CMIG_DATA / "nsiegert/projects/aridity/data/halfdeg_ref_grid_repaired.nc"
LAND_MASK_FILE = CMIG_DATA / "nsiegert/projects/aridity/data/landseamask/era5_landmask_halfdeg.nc"
OUTPUT_DIR     = CMIG_DATA / "nsiegert/projects/aridity/data/P"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Processing GPCC precipitation data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and concatenate all GPCC files
    files = sorted(glob.glob(str(GPCC_RAW_DIR / "*.nc")))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {GPCC_RAW_DIR}")
    print(f"Found {len(files)} files to process.")

    combo_ds = xr.open_mfdataset(files, combine="by_coords")[["precip"]]
    combo_ds.load()

    # 2. Regrid from 1.0-degree to 0.5-degree (conservative)
    ds_ref    = xr.open_dataset(REF_GRID_FILE)
    regridder = xe.Regridder(ds_in=combo_ds, ds_out=ds_ref, method="conservative")
    da_regrid = regridder(combo_ds.precip, keep_attrs=True)

    # 3. Load land-ocean mask and apply
    land_mask   = xr.open_dataset(LAND_MASK_FILE)["landfrac"]
    masked_gpcc = da_regrid.where(land_mask > 0).to_dataset(name="P")

    # 4. Attributes
    masked_gpcc.P.attrs["desc"]  = "Precipitation"
    masked_gpcc.P.attrs["units"] = "mm/month"
    masked_gpcc.attrs["info"]    = (
        "Monthly data from GPCC v2022. Originally 1.0-deg resolution, "
        "regridded to 0.5-deg. Ocean masked."
    )
    masked_gpcc.attrs["script"]    = "code/scripts/GPCC_P.py"
    masked_gpcc.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 5. Save
    out_path = OUTPUT_DIR / "P.GPCC.1891.2020.nc"
    masked_gpcc.to_netcdf(out_path)
    print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()