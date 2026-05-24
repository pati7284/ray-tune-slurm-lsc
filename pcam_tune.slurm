#!/bin/bash
#SBATCH --job-name=pcam_optuna
#SBATCH --account=plglscclass26-gpu
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate ray-gpu

# Wymuszenie użycia krótkiej ścieżki /tmp, aby ominąć błąd 107 znaków
export RAY_TMPDIR=/tmp

python run_pcam.py
