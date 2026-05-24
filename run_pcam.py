import os
import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from train import train_model

def main():
    if "redis_password" in os.environ:
        _node_ip_addr = os.environ["ip_head"].split(":")[0]
        ray.init(address=os.environ["ip_head"], _redis_password=os.environ["redis_password"], _node_ip_address=_node_ip_addr)
    else:
        ray.init() 

    my_storage_path = os.path.join(os.environ.get("SCRATCH", "."), "ray_results")

    search_space = {
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64]),
        "epochs": 5 
    }

    optuna_search = OptunaSearch(metric="mean_accuracy", mode="max")

    tuner = tune.Tuner(
        tune.with_resources(train_model, resources={"gpu": 1, "cpu": 4}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            num_samples=30,
        ),
        param_space=search_space,
        run_config=tune.RunConfig(
            name="pcam-resnet-tune",
            storage_path=my_storage_path
        )
    )
    
    print("Rozpoczynam optymalizację PCAM (30 prób)...")
    results = tuner.fit()
    best_result = results.get_best_result(metric="mean_accuracy", mode="max")
    
    print("\n=== EKSPERYMENT ZAKOŃCZONY ===")
    print(f"Najlepsza dokładność: {best_result.metrics['mean_accuracy']}")
    print(f"Najlepsze parametry: {best_result.config}")

if __name__ == "__main__":
    main()
