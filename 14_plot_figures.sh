#!/bin/bash
#SBATCH --job-name='plot_figures'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=4:00:00
# Email address
#SBATCH --mail-user=<flora.l.perlmutter.gr@dartmouth.edu>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=CMIG
#SBATCH --partition=preemptable

#     'Figure_01_Regression_Cross_Validation' \
#     'Figure_02_Bad_Control' \
#     'Figure_03_Regression_vs_PCA_SST_Sensitivity' \
#     'Figure_04_PCA_Regression_CV_hexabin' \
#     'Figure_05_Regression_Robustness_Tests' \
#     'Figure_06_Amazon_Model_Construction' \
#     'Figure_08_Average_SST_Sensitivity' \
#     'Figure_09_Ocean_Region_Importance' \
#     'Figure_10_SST_Forced_Trends' \
#     'Figure_11_AMIP_Average_SST_Sensitivity' \
#     'Figure_12_AMIP_vs_Obs_SST_Sensitivity'

# Regenerate all paper figures (Figures 1-12).

set -e  # stop on first error
# repo location
REPO_DIR=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precip-submission

cd $REPO_DIR

SCRIPT_DIR="$REPO_DIR/code/figures"

echo "Running figure scripts from: $SCRIPT_DIR"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
module load python
conda activate fp1225

for script in \
     'Figure_07_SST_Importance_by_Basin'; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    python -u "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All figures complete."
