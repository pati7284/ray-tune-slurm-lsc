# Person B - ML Pipeline Update

## What was done:
1. **Dataset Selection & Setup:** Decided on the **PCAM** (PatchCamelyon) dataset for binary classification (tumor detection). It's a great fit for Large Scale Computing (large enough for GPU, but small enough file size to not overload the I/O).
2. **Modular Architecture:** Wrote the ML pipeline and split it into three clean modules to satisfy the "dataset/model-agnostic" requirement:
   - `dataset.py`: Handles data loading, ResNet-specific resizing (96x96 -> 224x224), ImageNet normalization, and augmentations for Optuna.
   - `model.py`: Loads pre-trained `ResNet18` and modifies the final `fc` layer for binary output (1 neuron).
   - `train.py`: The main training loop utilizing `BCEWithLogitsLoss`. Includes the `train.report({"loss": ..., "mean_accuracy": ...})` hook to ensure compatibility with Patrycja's offline W&B logger.
3. **Dependencies Fixed:** Identified that PCAM requires `h5py` (to read the files) and `gdown` (to download from GDrive).

## Status:
- **Sanity check passed.** Tested locally on the Ares login node. 
- The script successfully downloaded the ~6.4GB training image file to `$SCRATCH/pcam_data` and properly initiated the PyTorch pipeline.
- *Note:* The test was intentionally killed by the system (OOM) on the login node due to strict RAM limits when loading the `.h5` file, which confirms the logic is solid and ready for the compute nodes.

## ⚠️ Important Note for Patrycja (Person A):
When you plug `train.py` into the Ray Tune script and send it to SLURM, **please ensure you request a lot of RAM** (e.g., `#SBATCH --mem=64G` or `128G`). 
Loading the PCAM `.h5` files into memory is heavy. If Ray spawns multiple parallel trials on one node, each trial will load the dataset, which can easily cause an Out-Of-Memory crash if the SLURM allocation is too small.
