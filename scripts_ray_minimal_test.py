from ray import tune

def trainable(config):
    for step in range(3):
        tune.report({"score": step + config["x"]})

tuner = tune.Tuner(
    trainable,
    param_space={"x": 1},
)
tuner.fit()
print("minimal Ray Tune test completed")
