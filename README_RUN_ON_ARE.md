# Ray Tune + Optuna + SLURM on PLGrid Ares
This repository contains a working Ray Tune + Optuna setup adapted for the PLGrid Ares cluster.

The original repository was modified so that it works reliably on Ares with SLURM, Conda environments, CPU jobs, GPU jobs, and offline W&B logging.
---

## 1. Important rule

Do **not** run training or tuning directly on the login node with commands such as:


`python src/rtstest/dothetune_nowandb.py --tune --gpu`


The login node should only be used for:

logging into Ares,
editing files,
preparing SLURM scripts,
submitting jobs,
checking logs and results.

Actual training and hyperparameter tuning should be submitted through SLURM using sbatch.

Example:
`sbatch slurm/simple_gpu_tune.slurm`

## 2. Why this repo has changes compared to the original demo

Several changes were needed to make the original Ray Tune SLURM demo work on Ares.

### 2.1 Conda/library compatibility issues

There were compatibility issues between some Conda packages, especially around:

* PyTorch,
* torchvision,
* **Pillow**,
* **libtiff**,
* CUDA-enabled PyTorch,
* Ray,
* **W&B.**

One recurring error was:
`ImportError: libtiff.so.5: cannot open shared object file: No such file or directory` 

This was caused by an incompatible Pillow/libtiff combination.

The stable workaround used here is:
`pip uninstall -y pillow
pip install --no-cache-dir pillow`

after creating the Conda environment.

There was also a separate issue where a CPU-only PyTorch environment could not see the GPU:

`torch.cuda.is_available() = False`

Therefore, two environments are used:

* ray-clean for CPU jobs,
* ray-gpu for GPU jobs.

### 2.2 SLURM scripts were adapted for Ares

The original SLURM scripts were modified so that they match the Ares cluster configuration.

Important Ares-specific changes include:

* correct partition names,
* correct PLGrid grant/account names,
* GPU resource request with --gres=gpu:1,
* correct Conda module loading,
* correct Conda environment activation,
* use of $SCRATCH,
* disabling unstable Ray W&B callback,
* using a stable Python script without WandbLoggerCallback.

For example, GPU jobs use:
``` #SBATCH --partition=plgrid-gpu-v100
#SBATCH --account=plglscclass26-gpu
#SBATCH --gres=gpu:1
```

CPU jobs use:

```#SBATCH --partition=plgrid-testing
#SBATCH --account=plglscclass26-cpu
```

These values were selected based on the available Ares partitions and active PLGrid grants.

--- 
## 3. Repository setup after logging into Ares -- HOW TO RUN

After logging into Ares (``` ssh your_name_acc@cyfronet.pl```), go to your scratch directory:

`cd $SCRATCH` 
Clone this repository:

`git clone https://github.com/pati7284/ray-tune-slurm-lsc.git`
`cd ray-tune-slurm-lsc`

Load Conda:
`module load miniconda3`
`eval "$(conda shell.bash hook)"`

## 4. Conda environments

Two separate environments are recommended.

### 4.1 CPU environment: ray-clean

Use this for CPU-only tests and CPU tuning.

Create it with:
```
conda create -y -n ray-clean -c conda-forge \
  python=3.10 \
  pytorch torchvision \
  ray-tune optuna wandb click joblib pandas
```

Activate it:
`conda activate ray-clean`

Fix Pillow/libtiff compatibility:

```
pip uninstall -y pillow
pip install --no-cache-dir pillow
```

Install the repository package without forcing dependency changes:
`pip install -e . --no-deps`

Check basic imports:

```
python - <<'PY'
import torch
import torchvision
from PIL import Image
import ray
import optuna
import wandb

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PIL ok")
print("ray:", ray.__version__)
print("optuna:", optuna.__version__)
print("wandb:", wandb.__version__)
print("cuda available:", torch.cuda.is_available())
PY
```

For the CPU environment, `cuda available`can be `False`.


### 4.2 GPU environment: ray-gpu

Use this for GPU tests and GPU tuning.

Create it with:
```
conda create -y -n ray-gpu -c pytorch -c nvidia -c conda-forge \
  python=3.10 \
  pytorch torchvision pytorch-cuda=12.1 \
  ray-tune optuna wandb click joblib pandas
```

Activate it:
`conda activate ray-gpu`

Fix Pillow/libtiff compatibility:

```
pip uninstall -y pillow
pip install --no-cache-dir pillow
```

Install the repository package without forcing dependency changes:

`pip install -e . --no-deps`

Check basic imports on the login node:
```
python - <<'PY'
import torch
import torchvision
from PIL import Image
import ray
import optuna
import wandb

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("PIL ok")
print("ray:", ray.__version__)
print("optuna:", optuna.__version__)
print("wandb:", wandb.__version__)
print("cuda available on login node:", torch.cuda.is_available())
PY
```

On the login node, `cuda availabl`e on login node may still be `False`. **This is normal because the login node does not allocate a GPU.**

The real GPU check must be done through SLURM.
## 5. Main Python files
Original script
`src/rtstest/dothetune.py`

This is the original script from the demo repository.

It contains Ray s built-in W&B callback:

`WandbLoggerCallback(...)`

This callback was unstable on Ares and caused segmentation faults.

#### Stable script used for experiments

`src/rtstest/dothetune_nowandb.py`

This is the stable version used in the SLURM jobs.

The important change is that the Ray W&B callback is disabled:

`callbacks=[]`
insted of: 
```
callbacks=[
    WandbLoggerCallback(project=study_name),
]
```

Use `dothetune_nowandb.py` for CPU and GPU experiments.


## 6. SLURM scripts

All SLURM scripts are in:
`slurm/`

The scripts were adapted for Ares. In particular:

* CPU scripts use plgrid-testing and plglscclass26-cpu,
* GPU scripts use plgrid-gpu-v100 and plglscclass26-gpu,
* GPU scripts request one GPU with #SBATCH --gres=gpu:1,
* CPU scripts activate ray-clean,
* GPU scripts activate ray-gpu,
* training uses dothetune_nowandb.py.


### 6.1 CPU single test

`sbatch slurm/simple_cpu_test.slurm`

This runs one simple Ray Tune trial on CPU.

The script runs:
`python src/rtstest/dothetune_nowandb.py`

###6.2 CPU tuning

`sbatch slurm/simple_cpu_tune.slurm`

This runs Ray Tune + Optuna on CPU.

The script runs:
`python src/rtstest/dothetune_nowandb.py --tune`

### 6.3 GPU check

`sbatch slurm/gpu_check_ray_gpu.slurm`
This checks whether SLURM correctly allocates a GPU and whether PyTorch can see CUDA.

Expected output should include something similar to:

```
HOSTNAME: ag0001
CUDA_VISIBLE_DEVICES: 0
SLURM_JOB_GPUS: 0
NVIDIA-SMI:
...
Tesla V100-SXM2-32GB
...
torch: 2.5.1
cuda available: True
cuda device count: 1
device name: Tesla V100-SXM2-32GB
```


If this test returns:

`cuda available: True`

then the GPU environment is working.

### 6.4 GPU single test

`sbatch slurm/simple_gpu_test.slurm`

This runs one Ray Tune trial on GPU.

The script runs:

`python src/rtstest/dothetune_nowandb.py --gpu`

### 6.5 GPU tuning

`sbatch slurm/simple_gpu_tune.slurm`

This is the main GPU experiment.

It runs Ray Tune + Optuna on GPU:

`python src/rtstest/dothetune_nowandb.py --tune --gpu`

This was tested successfully on Ares. The run completed 30 trials and used one V100 GPU.

### 6.6 W&B callback test

`sbatch slurm/wandb_callback_test.slurm`

This script was used only for debugging.

It showed that Rays built-in `WandbLoggerCallback` crashes on Ares with segmentation faults in `_WandbLoggingActor.`

This script is kept for documentation/debugging purposes, but it is not recommended for normal runs.

## 7. Checking job status

After submitting a job, SLURM returns a job ID:
`Submitted batch job JOB_ID`

Check status with:

`squeue -j JOB_ID`

Example:
`squeue -j 20073628`

If the job is running, you will see status R.

If you see:

`slurm_load_jobs error: Invalid job id specified`

it usually means the job has already finished and is no longer in the active queue.

## 8. Checking logs

Each SLURM job writes logs to the slurm/ directory.

Examples:

```
tail -80 slurm/ray_gpu_tune-JOB_ID.out
tail -80 slurm/ray_gpu_tune-JOB_ID.err
```

The `.out` file usually contains Ray Tune progress tables.

The `.err `file may contain Ray warnings. Some warnings about Ray dashboard or metrics exporter can appear on Ares. They are not necessarily fatal if the trial finishes successfully.

A successful run should contain something like:
```
Trial status: 30 TERMINATED
Logical resource usage: 1.0/32 CPUs, 1.0/1 GPUs
```

## 9. Ray Tune results

Ray stores experiment results in:
`~/ray_results/ray-tune-slurm-test`


Inside each trial directory, useful files include:

```
progress.csv  - metrics per training iteration
params.json   - hyperparameters for the trial
result.json   - final result
```
For example:

`find ~/ray_results/ray-tune-slurm-test -maxdepth 2 -type f | head`

To inspect metrics:

`head -5 ~/ray_results/ray-tune-slurm-test/Trainable_*/progress.csv`

## 10. Summarizing Ray Tune results

The project includes summary CSV files generated from Ray Tune results:

```
slurm_cpu_tuning_summary.csv
slurm_gpu_tuning_summary.csv
```


These files may need to be regenerated after a new run.

Example summary generation logic:
```
python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path.home() / "ray_results" / "ray-tune-slurm-test"

rows = []
for progress_file in base.glob("Trainable_*/progress.csv"):
    df = pd.read_csv(progress_file)
    if "mean_accuracy" not in df.columns:
        continue
    best_row = df.loc[df["mean_accuracy"].idxmax()]
    rows.append({
        "trial": progress_file.parent.name,
        "best_accuracy": best_row["mean_accuracy"],
        "best_iteration": int(best_row["training_iteration"]),
        "path": str(progress_file.parent),
    })

results = pd.DataFrame(rows).sort_values("best_accuracy", ascending=False)
results.to_csv("slurm_gpu_tuning_summary.csv", index=False)
print(results.head(10).to_string(index=False))
PY
```

In the completed GPU run (not our dataset) , the best observed result was:
```
mean_accuracy = 0.990625
conf_out_channels = 9
lr = 0.0861
momentum = 0.6357
```

## 11. W&B workflow

### 11.1 What works

Manual W&B offline logging works:

```
import wandb

run = wandb.init(
    project="ray-slurm-wandb-test",
    mode="offline",
)

wandb.log({"accuracy": 0.99})
wandb.finish()
```

### 11.2 What does not work

Rays built-in W&B callback is unstable on Ares:
`WandbLoggerCallback(...)`

It crashes with errors like:

```
_WandbLoggingActor
Fatal Python error: Segmentation fault
```

Therefore, do not use WandbLoggerCallback in the main experiment.

### 11.3 Stable workaround

Use this workflow instead:

1. Run Ray Tune without W&B callback.
2. Let Ray write progress.csv files.
3. Generate a summary CSV.
4. Log the summary manually to W&B offline.

The script:

`log_gpu_summary_to_wandb.py`

ogs the GPU tuning summary to W&B offline.

Run:
`python log_gpu_summary_to_wandb.py`

To sync the offline run later:
`wandb sync wandb/offline-run-...`

Example from a completed run:

`wandb sync /net/afscra/people/plgpatim423/ray-slurm-project/ray-tune-slurm-demo/wandb/offline-run-20260510_142939-czew69uh`
