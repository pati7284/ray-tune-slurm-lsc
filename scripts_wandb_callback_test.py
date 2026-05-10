import os
from ray import tune
from ray.air import RunConfig
from ray.air.integrations.wandb import WandbLoggerCallback

os.environ["WANDB_MODE"] = "offline"

def trainable(config):
    for step in range(3):
        tune.report({"score": step + config["x"]})

tuner = tune.Tuner(
    trainable,
    param_space={"x": 1},
    run_config=RunConfig(
        name="wandb-callback-minimal-test",
        callbacks=[
            WandbLoggerCallback(project="ray-slurm-wandb-test")
        ],
    ),
)

tuner.fit()
print("minimal WandbLoggerCallback test completed")
