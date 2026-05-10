# Person A notes - Ray Tune + SLURM

## Done
- Cloned ray-tune-slurm-demo on Ares in $SCRATCH.
- Created clean Conda environment: ray-clean.
- Installed required packages from conda-forge.
- Installed repository with: pip install -e . --no-deps.
- Created dothetune_nowandb.py by disabling WandbLoggerCallback.
- Ran dothetune_nowandb.py manually on login node for a short technical test.
- Created simple CPU SLURM job: slurm/simple_cpu_test.slurm.
- Submitted job with sbatch.
- Job 20072206 ran on compute node ac0682.
- Ray Tune completed 1 trial with 5 iterations.
- Results were saved to ~/ray_results/ray-tune-slurm-test.

## Observations
- W&B callback caused segmentation fault, so it was disabled for the first SLURM test.
- Ray dashboard/metrics exporter produced warnings, but training completed successfully.
- Current test used CPU only. GPU test is the next step.

## GPU results
- GPU was allocated successfully through SLURM on partition plgrid-gpu-v100.
- GPU account used: plglscclass26-gpu.
- GPU device detected: Tesla V100-SXM2-32GB.
- Created and used Conda environment: ray-gpu.
- GPU single run completed through SLURM.
- GPU tuning completed through SLURM with 30 trials.
- Best observed GPU tuning accuracy: 0.990625.
- Best observed GPU config: conf_out_channels=9, lr=0.0861, momentum=0.6357.
- GPU tuning summary saved to: slurm_gpu_tuning_summary.csv.

## Remaining
- Fix or replace W&B logging.
- Decide final integration approach for the real model/dataset.

## W&B status
- Manual W&B offline logging works.
- Ray AIR WandbLoggerCallback crashes with segmentation fault in _WandbLoggingActor.
- The crash occurs also in a minimal SLURM test, so it is not caused by the model code.
- Stable workaround: log Ray Tune results to W&B manually after the run from CSV files.
- Created script: log_gpu_summary_to_wandb.py.
- Logged GPU tuning summary to W&B offline.
