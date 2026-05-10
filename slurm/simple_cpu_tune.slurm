#!/bin/bash
#SBATCH --job-name=ray_cpu_test
#SBATCH --output=slurm/ray_cpu_test-%j.out
#SBATCH --error=slurm/ray_cpu_test-%j.err
#SBATCH --partition=plgrid-testing
#SBATCH --account=plglscclass26-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate ray-clean

cd /net/afscra/people/plgpatim423/ray-slurm-project/ray-tune-slurm-demo

export WANDB_MODE=offline
export RAY_TMPDIR=$SCRATCH/ray_tmp
mkdir -p $RAY_TMPDIR

python src/rtstest/dothetune_nowandb.py --tune
