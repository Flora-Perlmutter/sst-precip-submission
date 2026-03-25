#!/bin/bash
#SBATCH --job-name='plot_figures'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
# Email address
#SBATCH --mail-user=<flora.l.perlmutter.gr@dartmouth.edu>
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=CMIG
# \
#    'Figure_01_Cross_validation' \
#    'Figure_02_Bad_Control' \
#    'Figure_03_Randomization_experiment' \
#    'Figure_04_Stability_of_the_marginal_sensitivity' \
#    'Figure_05_Amazon_Model_Construction' \
#    'Figure_06_SST_Importance_by_basin' \
#    'Figure_07_Average_marginal_sensitivity' \
#    'Figure_08_Ocean_importance_by_variability' \
#    'Figure_09_SST_Trends_with_significance' \
#    'Figure_10_AMIP_average_MS' \
#    'Figure_11_AMIP_vs_Obs_marginal_sensitivity'

# Run Data Processing Script 16.
# Regenerate all paper figures (Figures 1-11).

set -e  # stop on first error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/code/figures"

echo "Running figure scripts from: $SCRIPT_DIR"

for script in \
    '16_Process_Cross_Validation_Data'; do
    echo "----------------------------------------"
    echo "$(date '+%H:%M:%S')  $script"
    python "$SCRIPT_DIR/${script}.py"
done

echo "----------------------------------------"
echo "$(date '+%H:%M:%S')  All figures complete."
