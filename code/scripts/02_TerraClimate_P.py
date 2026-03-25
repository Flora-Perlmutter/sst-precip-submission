#!/usr/bin/env python
# coding: utf-8
"""
Prepare TerraClimate precipitation data for analysis.

Updated: Flora Perlmutter, 3/8/2026
Original: Leah Brown, 07/28/2025

Description
-----------
TerraClimate monthly precipitation at ~0.04166-degree resolution (mm/month),
spanning 1958-2025. Only files from 1979 onward are processed.
This script:
  1. Loops over individual annual TerraClimate files from 1979 onward
  2. Regrids each from ~0.04166-degree to 0.5-degree (conservative remapping)
  3. Writes one output file per year to the output directory

Note: This script is memory- and time-intensive 
Submit via sbatch with appropriate resource requests.
Requires environment with xesmf.

Source: https://climatedataguide.ucar.edu/climate-data/terraclimate-global-high-resolution-gridded-temperature-precipitation-and-other-water

"""

import glob
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

# --- project paths ---
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root
from paths import CMIG_DATA

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# HPC paths — all relative to CMIG_DATA
# ---------------------------------------------------------------------------
TERRACLIMATE_RAW_DIR = CMIG_DATA / "Data/Observations/TerraClimate/ppt"
OUTPUT_DIR    = CMIG_DATA / 'fperlmutter/Observational_Regressions_Project/Data/Processed'

# First year to process (files before this are skipped)
START_YEAR = 1979


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_target_grid(res: float = 0.5) -> xr.Dataset:
    """Construct a global regular lat/lon grid at the given resolution."""
    lats = np.arange(-90  + res / 2,  90,  res)
    lons = np.arange(-180 + res / 2, 180,  res)
    return xr.Dataset(
        {"lat": (["lat"], lats), "lon": (["lon"], lons)}
    )


def process_one_file(
    f: str,
    ds_ref: xr.Dataset,
    regridder: xe.Regridder,
) -> None:
    """Regrid and save one TerraClimate annual file."""

    terrCl_P = xr.Dataset({"P": xr.open_dataset(f).ppt.load()})

    regridded = xr.Dataset(
        {"P": regridder(terrCl_P.P, keep_attrs=True)}
    )

    regridded.P.attrs["desc"]  = "Total monthly precipitation"
    regridded.P.attrs["units"] = "mm/month"
    regridded.attrs["info"]    = (
        f"Monthly TerraClimate data, {START_YEAR}-2025. "
        "Originally ~0.04166-deg resolution, regridded to 0.5-deg."
    )
    regridded.attrs["script"]    = "code/scripts/TerraClimate_P.py"
    regridded.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    out_path = OUTPUT_DIR / "P.TerraClimate.{}".format(Path(f).name[-7:])
    regridded.to_netcdf(out_path)
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

    # Build target grid and regridder once using the first file as a template
    ds_ref    = make_target_grid(res=0.5)
    template  = xr.Dataset({"P": xr.open_dataset(files[0]).ppt})
    regridder = xe.Regridder(ds_in=template, ds_out=ds_ref, method="conservative")

    for f in files:
        process_one_file(f, ds_ref, regridder)

    print("Done.")


if __name__ == "__main__":
    main()