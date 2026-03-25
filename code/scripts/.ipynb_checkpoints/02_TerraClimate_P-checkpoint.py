#!/usr/bin/env python
# coding: utf-8
"""
Prepare TerraClimate precipitation data for analysis.

Original: Leah Brown, 07/28/2025
Refactored for reproducibility: February 24, 2026

Description
-----------
TerraClimate monthly precipitation at ~0.04166-degree resolution (mm/month),
spanning 1958-2024. Only files from 1979 onward are processed.
This script:
  1. Loops over individual annual TerraClimate files from 1979 onward
  2. Regrids each from ~0.04166-degree to 0.5-degree (conservative remapping)
  3. Masks out ocean using the ERA5 land fraction mask
  4. Writes one output file per year to the output directory

Note: This script is memory- and time-intensive (~20hrs, 40+GB on HPC).
Submit via sbatch with appropriate resource requests.

Source: https://climatedataguide.ucar.edu/climate-data/terraclimate-global-high-resolution-gridded-temperature-precipitation-and-other-water

Usage
-----
    python TerraClimate_P.py
    sbatch submit_TerraClimate_P.sh
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
TERRACLIMATE_RAW_DIR = CMIG_DATA / "Data/Observations/TerraClimate/ppt"
REF_GRID_FILE        = CMIG_DATA / "nsiegert/projects/aridity/data/halfdeg_ref_grid_repaired.nc"
LAND_MASK_FILE       = CMIG_DATA / "nsiegert/projects/aridity/data/landseamask/era5_landmask_halfdeg.nc"
OUTPUT_DIR           = CMIG_DATA / "nsiegert/projects/aridity/data/P"

# First year to process (files before this are skipped)
START_YEAR = 1979


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def process_one_file(
    f: str,
    ds_ref: xr.Dataset,
    land_mask: xr.DataArray,
    regridder: xe.Regridder,
) -> None:
    """Regrid, mask, and save one TerraClimate annual file."""

    terrCl_P = xr.Dataset({"P": xr.open_dataset(f).ppt.load()})

    masked_TC = xr.Dataset(
        {"P": regridder(terrCl_P.P, keep_attrs=True)}
    ).where(land_mask > 0)

    masked_TC.P.attrs["desc"]  = "Total monthly precipitation"
    masked_TC.P.attrs["units"] = "mm/month"
    masked_TC.attrs["info"]    = (
        f"Monthly TerraClimate data, {START_YEAR}-2024. "
        "Originally ~0.04166-deg resolution, regridded to 0.5-deg. Ocean masked."
    )
    masked_TC.attrs["script"]    = "code/scripts/TerraClimate_P.py"
    masked_TC.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUTPUT_DIR / "P.TerraClimate.{}".format(Path(f).name[-7:])
    masked_TC.to_netcdf(out_path)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Processing TerraClimate precipitation data...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect files and filter to START_YEAR onward
    all_files = sorted(glob.glob(str(TERRACLIMATE_RAW_DIR / "TerraClimate_ppt_*.nc")))
    if not all_files:
        raise FileNotFoundError(f"No TerraClimate files found in {TERRACLIMATE_RAW_DIR}")

    files = [f for f in all_files if int(Path(f).stem[-4:]) >= START_YEAR]
    if not files:
        raise ValueError(f"No files found from {START_YEAR} onward.")
    print(f"Processing {len(files)} files ({START_YEAR} onward).")

    # Load shared reference files once before the loop
    ds_ref    = xr.open_dataset(REF_GRID_FILE)
    land_mask = xr.open_dataset(LAND_MASK_FILE)["landfrac"]

    # Build regridder once using the first file as a template
    template = xr.Dataset({"P": xr.open_dataset(files[0]).ppt})
    regridder = xe.Regridder(ds_in=template, ds_out=ds_ref, method="conservative")

    for f in files:
        process_one_file(f, ds_ref, land_mask, regridder)

    print("Done.")


if __name__ == "__main__":
    main()