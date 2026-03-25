#!/bin/bash
#SBATCH --job-name='data_processing'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=260G
#SBATCH --time=36:00:00
# Email address
#SBATCH --mail-user=<flora.l.perlmutter.gr@dartmouth.edu>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=CMIG

# Run Data Processing Scripts 1-4.
# Run from the project root, or from anywhere
#    '01_CPC_P' \
#    '02_GPCP_P' \
#    '03_REGEN_P' \
#    '04_TerraClimate_P' \


set -e  # stop on first error

# repo location
REPO_DIR=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precipitation-sensitivity

cd $REPO_DIR

SCRIPT_DIR="$REPO_DIR/code/scripts"

echo "Running figure scripts from: $SCRIPT_DIR"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
module load python
conda activate fp1225

for script in \
    '04_TerraClimate_P' \
    '05_Construct_Observational_Ensemble'; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    python -u "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All data processing complete."
