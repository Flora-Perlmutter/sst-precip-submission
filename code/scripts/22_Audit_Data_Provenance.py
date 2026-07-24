#!/usr/bin/env python
# coding: utf-8
"""
Audit which input record each result file was computed from, and roll the
verdict up to each paper figure.

Author: Flora Perlmutter

Motivation
----------
The observational anomaly inputs were extended (e.g. 1980-2024 -> 1979-2025).
Recomputing every downstream result is expensive, so this script reports, for
each result file, the time span actually baked into it and compares that to the
current span of the specific inputs it depends on. A figure is up to date only
if every result it reads matches its current inputs.

Two signals are reported per result:
  * embedded time span  -- authoritative where the file carries a 'time' (or
    'window') coordinate; it is the record the computation actually saw.
  * mtime vs input mtime -- the only signal available for files with no time
    axis (cv_results, randomization, ranks CSVs). Treated as heuristic, since
    a git checkout can reset the mtime of repo-tracked outputs.

Run
---
    python code/scripts/22_Audit_Data_Provenance.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import CMIG_DATA, DATA_DIR

INPUTS_DIR = CMIG_DATA / "fperlmutter/Observational_Regressions_Project/Data/Processed"

PRECIP = ["GPCP", "CRU", "GPCC", "CPC", "UDel", "PREC", "TerraClimate", "REGEN"]
SST    = ["ERSSTv6", "COBE-SST3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mtime(path: Path) -> datetime:
    return datetime.fromtimestamp(os.path.getmtime(path))


def _times(path: Path):
    """Return the sorted time-coordinate values of a .nc file, or None."""
    if not path.exists():
        return None
    try:
        try:
            da = xr.open_dataarray(path)
        except Exception:
            da = xr.open_dataset(path)
        t = da["time"].values if "time" in da.coords or "time" in da.dims else None
        da.close()
        return np.sort(t) if t is not None else None
    except Exception:
        return None


def _span_str(path: Path):
    """(first, last, n) from a 'time' coord, else from a 'window' coord, else None."""
    if not path.exists():
        return None
    try:
        ds = xr.open_dataset(path)
    except Exception:
        return None
    try:
        if "time" in ds.coords:
            t = np.sort(ds["time"].values)
            return (str(t[0])[:7], str(t[-1])[:7], len(t))
        if "window" in ds.coords:
            w = np.sort(ds["window"].values.astype(int))
            return (f"{w[0]}", f"{w[-1]}", len(w))     # window end-years
        return None
    except Exception:
        return None
    finally:
        ds.close()


def _pair_input_span(p_name: str, sst_name: str, need_raw: bool):
    """Current common time span of the inputs for one (precip, SST) pair."""
    files = [INPUTS_DIR / f"precip_anom_{p_name}.nc",
             INPUTS_DIR / f"sst_anom_{sst_name}.nc"]
    if need_raw:
        files.append(INPUTS_DIR / f"sst_raw_{sst_name}.nc")

    common = None
    newest_mtime = None
    for f in files:
        t = _times(f)
        if t is None:
            return None
        common = t if common is None else np.intersect1d(common, t)
        m = _mtime(f)
        newest_mtime = m if newest_mtime is None else max(newest_mtime, m)
    if common is None or len(common) == 0:
        return None
    return (str(common[0])[:7], str(common[-1])[:7], len(common), newest_mtime)


def _parse_pair(stem: str, prefix: str):
    """'<prefix>GPCP_COBE-SST3' -> ('GPCP', 'COBE-SST3'); SST never contains '_'."""
    core = stem[len(prefix):]
    if "_" not in core:
        return None
    return tuple(core.rsplit("_", 1))


def _verdict(result_end, input_end, result_mtime, input_mtime):
    """
    Return (verdict, basis). Prefer the embedded-time comparison; fall back to
    mtime only when the result carries no time axis.
    """
    if result_end is not None and input_end is not None:
        if result_end >= input_end:
            return "UP-TO-DATE", "time-span"
        return "STALE", "time-span"
    if result_mtime is not None and input_mtime is not None:
        if result_mtime >= input_mtime:
            return "up-to-date?", "mtime-only"
        return "STALE", "mtime-only"
    return "unknown", "no-signal"


# ---------------------------------------------------------------------------
# Result families and figure dependencies
# ---------------------------------------------------------------------------
# per_pair families are parsed into (precip, SST) and matched to their inputs.
# aggregate families (no per-pair inputs) are reported by embedded span only.

FAMILIES = {
    "cv_results": dict(
        dir=INPUTS_DIR, glob="cv_results_*.nc", prefix="cv_results_",
        per_pair=True, need_raw=True, script="08",
    ),
    "linreg_bootstrap": dict(
        dir=DATA_DIR, glob="global_linear_regression_bootstrap_*.nc",
        prefix="global_linear_regression_bootstrap_", per_pair=True,
        need_raw=False, script="11", exclude="_amip_",
    ),
    "linreg_bootstrap_amip": dict(
        dir=DATA_DIR, glob="global_linear_regression_bootstrap_amip_*.nc",
        prefix=None, per_pair=False, need_raw=False, script="16",
    ),
    "rh_bootstrap": dict(
        dir=DATA_DIR, glob="global_rh_regression_bootstrap_*.nc",
        prefix="global_rh_regression_bootstrap_", per_pair=True,
        need_raw=False, script="12",
    ),
    "rh_sst_corr": dict(
        dir=DATA_DIR, glob="rh_sst_correlation_*.nc",
        prefix="rh_sst_correlation_", per_pair=True, need_raw=False, script="13",
    ),
    "randomization": dict(
        dir=DATA_DIR, glob="randomization_experiment_*.nc",
        prefix="randomization_experiment_", per_pair=True, need_raw=False,
        script="14",
    ),
    "pattern_corr": dict(
        dir=DATA_DIR, glob="pattern_correlations_all_basins.nc",
        prefix=None, per_pair=False, need_raw=False, script="15",
        compare_newest=True,
    ),
    "pca_regression": dict(
        dir=DATA_DIR, glob="global_pca_regression_*.nc",
        prefix="global_pca_regression_", per_pair=True, need_raw=False,
        script="17",
    ),
    "cv_reconstruction": dict(
        dir=DATA_DIR, glob="cv_reconstruction_*.nc",
        prefix=None, per_pair=False, need_raw=False, script="19/20",
        compare_newest=True,
    ),
    "cv_ranks": dict(
        dir=DATA_DIR, glob="cross_validation_ranks_*.csv",
        prefix=None, per_pair=False, need_raw=False, script="18 (from cv_results)",
    ),
}

FIGURE_DEPS = {
    "Figure_01": ["cv_results (-> cv_ranks)"],
    "Figure_02": ["rh_bootstrap", "linreg_bootstrap", "rh_sst_corr"],
    "Figure_03": ["linreg_bootstrap", "pca_regression"],
    "Figure_04": ["cv_reconstruction"],
    "Figure_05": ["randomization", "pattern_corr"],
    "Figure_06": ["linreg_bootstrap"],
    "Figure_07": ["linreg_bootstrap"],
    "Figure_08": ["linreg_bootstrap"],
    "Figure_09": ["linreg_bootstrap"],
    "Figure_10": ["linreg_bootstrap", "linreg_bootstrap_amip"],
    "Figure_11": ["linreg_bootstrap", "linreg_bootstrap_amip"],
    "Figure_12": ["linreg_bootstrap", "linreg_bootstrap_amip"],
}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    print("=" * 92)
    print("DATA PROVENANCE AUDIT")
    print("=" * 92)

    # --- 1. Current inputs -------------------------------------------------
    print("\n[1] CURRENT INPUT ANOMALY FILES (the reference record)\n")
    print(f"    {'file':28}{'span':22}{'n':>6}   {'modified':19}")
    input_ends = []
    input_mtimes = []
    for name in [f"precip_anom_{p}" for p in PRECIP] + [f"sst_anom_{s}" for s in SST]:
        f = INPUTS_DIR / f"{name}.nc"
        sp = _span_str(f)
        if sp is None:
            print(f"    {name:28}{'MISSING':22}")
            continue
        print(f"    {name:28}{sp[0]+'..'+sp[1]:22}{sp[2]:>6}   "
              f"{_mtime(f):%Y-%m-%d %H:%M}")
        input_ends.append(sp[1])
        input_mtimes.append(_mtime(f))
    newest_end = max(input_ends) if input_ends else None
    newest_input_mtime = max(input_mtimes) if input_mtimes else None
    if input_ends:
        print(f"\n    Newest input end month: {newest_end}")
        print("    (Datasets with a naturally shorter record, e.g. REGEN, end earlier")
        print("     by design; each result is checked against ITS OWN inputs below.)")

    # --- 2. Result families ------------------------------------------------
    print("\n[2] RESULT FILES vs THEIR INPUTS\n")
    family_status = {}   # family -> list of verdict strings

    for fam, cfg in FAMILIES.items():
        files = sorted(cfg["dir"].glob(cfg["glob"]))
        excl = cfg.get("exclude")
        if excl:
            files = [f for f in files if excl not in f.name]

        print(f"  --- {fam}  (script {cfg['script']}, {len(files)} files) ---")
        if not files:
            print("      none found\n")
            family_status[fam] = ["missing"]
            continue

        verdicts = []
        print(f"      {'file':46}{'result span':20}{'input span':20}{'verdict':14}")
        for f in files:
            r_span = _span_str(f)
            r_end  = r_span[1] if r_span else None
            r_mtime = _mtime(f)

            in_end = in_mtime = None
            in_disp = "-"
            if cfg["per_pair"]:
                pair = _parse_pair(f.stem, cfg["prefix"])
                if pair:
                    ps = _pair_input_span(pair[0], pair[1], cfg["need_raw"])
                    if ps:
                        in_end, in_mtime = ps[1], ps[3]
                        in_disp = f"{ps[0]}..{ps[1]}"
            elif cfg.get("compare_newest") and newest_end is not None:
                # obs-based aggregate: compare to the newest obs input as a proxy
                in_end, in_mtime = newest_end, newest_input_mtime
                in_disp = f"<= {newest_end}"

            verdict, basis = _verdict(r_end, in_end, r_mtime, in_mtime)
            verdicts.append(verdict)
            r_disp = f"{r_span[0]}..{r_span[1]}" if r_span else "no time axis"
            print(f"      {f.name[:44]:46}{r_disp:20}{in_disp:20}"
                  f"{verdict:12} ({basis})")
        family_status[fam] = verdicts
        print()

    # --- 3. Figure roll-up -------------------------------------------------
    print("[3] FIGURE ROLL-UP\n")

    def fam_state(fam_label):
        fam = fam_label.split(" ")[0]     # strip "(<- cv_results)" annotation
        vs = family_status.get(fam, ["missing"])
        if any(v == "STALE" for v in vs):
            return "STALE"
        if any(v in ("missing", "unknown") for v in vs):
            return "unknown"
        if all(v == "UP-TO-DATE" for v in vs):
            return "up-to-date"
        return "up-to-date?"     # mtime-only or mixed heuristic

    order = ["STALE", "unknown", "up-to-date?", "up-to-date"]
    for fig, deps in FIGURE_DEPS.items():
        states = {d: fam_state(d) for d in deps}
        worst = min((states[d] for d in deps), key=order.index)
        flag = {"STALE": "RERUN", "unknown": "CHECK",
                "up-to-date?": "likely ok", "up-to-date": "OK"}[worst]
        print(f"  {fig:12} {flag:10}  " +
              ", ".join(f"{d}={s}" for d, s in states.items()))

    print("\n" + "=" * 92)
    print("Legend: 'time-span' verdicts are authoritative; 'mtime-only' are heuristic")
    print("(a git checkout can reset an mtime). RERUN = at least one input is newer")
    print("than the record baked into the result.")
    print("=" * 92)


if __name__ == "__main__":
    main()
