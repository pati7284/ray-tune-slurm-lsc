#!/bin/bash
#SBATCH --job-name=gpu_check
#SBATCH --output=slurm/gpu_check-%j.out
#SBATCH --error=slurm/gpu_check-%j.err
#SBATCH --partition=plgrid-gpu-v100
#SBATCH --account=plglscclass26-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

module load miniconda3
eval "$(conda shell.bash hook)"
conda activate ray-gpu

echo "HOSTNAME: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "NVIDIA-SMI:"
nvidia-smi



python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))



PY
