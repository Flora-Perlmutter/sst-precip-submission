#!/bin/bash
#SBATCH --job-name='run_pca'
#SBATCH --output=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/array_%A_%a.out
#SBATCH --error=/dartfs-hpc/rc/lab/C/CMIG/fperlmutter/jobs/array_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=48:00:00
# Email address
#SBATCH --mail-user=flora.l.perlmutter.gr@dartmouth.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=CMIG

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
module load python
conda activate fp1225

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Running on $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

python3 /dartfs-hpc/rc/lab/C/CMIG/fperlmutter/git_repos/sst-precipitation-sensitivity/code/scripts/17_PCA_method.py --pair-index $SLURM_ARRAY_TASK_ID

