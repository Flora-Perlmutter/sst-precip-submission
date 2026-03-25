import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — directory containing this file
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Repo-tracked output directories (committed to GitHub)
# ---------------------------------------------------------------------------
DATA_DIR         = PROJECT_ROOT / "data"
FIGURE_DIR       = PROJECT_ROOT / "figures"
PAPER_FIGURE_DIR = FIGURE_DIR / "paper_figures"

# ---------------------------------------------------------------------------
# Machine-local HPC base
# Set this in your environment before running any scripts:
#   export CMIG_DATA=/dartfs-hpc/rc/lab/C/CMIG
# ---------------------------------------------------------------------------
CMIG_DATA = Path(os.environ.get("CMIG_DATA", "/dartfs-hpc/rc/lab/C/CMIG"))

# ---------------------------------------------------------------------------
# Auto-create repo-tracked output directories on import
# ---------------------------------------------------------------------------
for _dir in [DATA_DIR, FIGURE_DIR, PAPER_FIGURE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)