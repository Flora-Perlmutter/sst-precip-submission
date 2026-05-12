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
REPO_DIR=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precipitation-sensitivity

cd $REPO_DIR

SCRIPT_DIR="$REPO_DIR/code/scripts"

echo "Running figure scripts from: $SCRIPT_DIR"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
module load python
conda activate fp1225

for script in \
    '15_Stability_of_the_marginal_sensitivity'; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    python -u "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All data processing complete."
