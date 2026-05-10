import pandas as pd
import wandb

df = pd.read_csv("slurm_gpu_tuning_summary.csv")

run = wandb.init(
    project="ray-slurm-wandb-test",
    name="gpu-tuning-summary",
    mode="offline",
)

for rank, row in df.reset_index(drop=True).iterrows():
    wandb.log({
        "rank": rank + 1,
        "best_accuracy": row["best_accuracy"],
        "best_iteration": row["best_iteration"],
        "trial": row["trial"],
    })

best = df.iloc[0]
wandb.summary["best_accuracy"] = best["best_accuracy"]
wandb.summary["best_iteration"] = int(best["best_iteration"])
wandb.summary["best_trial"] = best["trial"]

wandb.finish()

print("Logged GPU tuning summary to W&B offline.")
print("Best accuracy:", best["best_accuracy"])
print("Best trial:", best["trial"])
