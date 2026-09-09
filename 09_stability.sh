#!/bin/bash
#SBATCH --job-name='ms_stability'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=300G
#SBATCH --time=48:00:00
# Email address
#SBATCH --mail-user=<flora.l.perlmutter.gr@dartmouth.edu>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=preemptable
#SBATCH --account=CMIG

# Run from the project root, or from anywhere



set -e  # stop on first error

# repo location
REPO_DIR=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precip-submission

cd $REPO_DIR

SCRIPT_DIR="$REPO_DIR/code/scripts"

echo "Running figure scripts from: $SCRIPT_DIR"

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

for script in \
    '15_Stability_of_the_marginal_sensitivity'; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    "$PYTHON" -u "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All data processing complete."
