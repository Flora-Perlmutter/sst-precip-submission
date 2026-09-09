#!/bin/bash
#SBATCH --job-name='run_rh_sst_correlation'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/array_%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/array_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=24:00:00
# Email address
#SBATCH --mail-user=flora.l.perlmutter.gr@dartmouth.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=CMIG

# Order matters: `module load python` puts anaconda3 python3.8 on PATH, so it
# has to run BEFORE conda is set up. With it after, it shadowed the activated
# environment and jobs silently ran under 3.8, which has no netCDF4 backend.
module load python
source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate fp1225

# Call the interpreter by absolute path rather than trusting PATH. Activation
# does not reliably win inside a batch shell here -- `module load python` puts
# anaconda3 python3.8 ahead of the activated environment, and that interpreter
# has no netCDF4 backend. Resolving the prefix from conda keeps the path out of
# the script while making PATH order irrelevant.
PYTHON="$(conda env list | awk '$1=="fp1225" {print $NF}')/bin/python3"
echo "Python: $PYTHON"
"$PYTHON" -c "import xarray, netCDF4" || { echo "FATAL: fp1225 interpreter unusable"; exit 1; }

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

"$PYTHON" /dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precip-submission/code/scripts/13_RH_SST_Correlation.py --pair-index $SLURM_ARRAY_TASK_ID

