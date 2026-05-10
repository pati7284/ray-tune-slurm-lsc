#!/bin/bash
#SBATCH --job-name=ray_gpu_tune
#SBATCH --output=slurm/ray_gpu_tune-%j.out
#SBATCH --error=slurm/ray_gpu_tune-%j.err
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --account=plglscclass26-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1 


module load miniconda3
eval "$(conda shell.bash hook)"
conda activate ray-gpu

cd /net/afscra/people/plgpatim423/ray-slurm-project/ray-tune-slurm-demo

export WANDB_MODE=offline
export RAY_TMPDIR=$SCRATCH/ray_tmp
mkdir -p $RAY_TMPDIR

python src/rtstest/dothetune_nowandb.py --tune --gpu
