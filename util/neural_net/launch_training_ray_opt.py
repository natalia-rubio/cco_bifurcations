from operator import ne
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from numpy import save
from util.neural_net.nn_model import NeuralNet
from util.neural_net.train_nn import train_nn
from util.tools.basic import load_dict, save_dict
from ray import tune
from hyperopt import hp
from ray.tune.search.hyperopt import HyperOptSearch
from ray.train import ScalingConfig
scaling_config = ScalingConfig(
    # Number of distributed workers.
    num_workers=4,
    # Turn on/off GPU.
    use_gpu=False,
)
anatomy = sys.argv[1]
num_geos = int(sys.argv[2])
set_type = "random"

global output_type
global coef_ind

search_space = { 
    "num_layers": tune.uniform(0, 4),
    "layer_width": tune.lograndint(1, 300),
    "learning_rate": tune.loguniform(1e-6, 1e-2),
    #"batch_size": tune.choice([20, 50, 100,]),
}
hyperopt_search = HyperOptSearch(search_space, metric="val_loss", mode="min")
def obj(config):
    split_ind_dict = load_dict(f"/Users/natalia/Desktop/cco_bifurcations/data/split_indices/{anatomy}/{set_type}/train_val_ind_{anatomy}_num_geos_{num_geos}")

    network_params = {"num_input_features": 10,
                        "num_layers": 2,
                        "layer_width":20,
                        "output_type": output_type,
                        "anatomy": anatomy,
                        "set_type": set_type,
                        "num_geos": num_geos,
                        "pred_mode": "m1"}

    training_params = {"num_epochs": 1000, 
                        "batch_size": 100,
                        "train_inds": split_ind_dict["train_ind"],
                        "val_inds": split_ind_dict["val_ind"],
                        "num_offsets": split_ind_dict["num_offsets"],}

    optimizer_params = {#"step_size": 0.0002,
                        "init" : 0.02,
                        "transition_steps": 5000,
                        "decay_rate" : 0.9}

    network_params["target_coef_ind"] = coef_ind
    network_params["output_type"] = output_type
    
    network_params["num_layers"] = int(config["num_layers"])
    network_params["layer_width"] = int(config["layer_width"])
    optimizer_params["init"] = config["learning_rate"]
    #training_params["batch_size"] = int(config["batch_size"])
    
    model = NeuralNet(network_params, optimizer_params)
    val_loss = train_nn(model, training_params)
    return {"val_loss": val_loss}


print("Hyperparameter optimization RRI Resistance")
output_type = "rri"
coef_ind = 0
tuner = tune.Tuner(obj, tune_config=tune.TuneConfig(
        num_samples=50,
        search_alg=hyperopt_search,
        max_concurrent_trials=4)
    )
results = tuner.fit()
print(results.get_best_result(metric="val_loss", mode="min").config)
if not os.path.exists(f"/Users/natalia/Desktop/cco_bifurcations/results/ray_tune"):
    os.makedirs(f"/Users/natalia/Desktop/cco_bifurcations/results/ray_tune")
save_dict(results.get_best_result(metric="val_loss", mode="min").config, f"/Users/natalia/Desktop/cco_bifurcations/results/ray_tune/{output_type}_{coef_ind}_best_config_{anatomy}_{num_geos}.json")

# print("Hyperparameter optimization RRI Quadratic Resistance")
# output_type = "rri"
# coef_ind = 1
# tuner = tune.Tuner(obj, tune_config=tune.TuneConfig(
#         num_samples=50,
#         search_alg=hyperopt_search,
#     )) 
# results = tuner.fit()
# print(results.get_best_result(metric="val_loss", mode="min").config)
# save_dict(results.get_best_result(metric="val_loss", mode="min").config, f"results/ray_tune/{output_type}_{coef_ind}_best_config_{anatomy}_{num_geos}.json")

# print("Hyperparameter optimization RRI Inductor")
# output_type = "rri"
# coef_ind = 2
# tuner = tune.Tuner(obj, param_space=search_space) 
# results = tuner.fit()
# print(results.get_best_result(metric="val_loss", mode="min").config)
# save_dict(results.get_best_result(metric="val_loss", mode="min").config, f"results/ray_tune/{output_type}_{coef_ind}_best_config_{anatomy}_{num_geos}.json")

# print("Hyperparameter optimization RRI Resistance")
# output_type = "ri"
# coef_ind = 0
# tuner = tune.Tuner(obj, param_space=search_space) 
# results = tuner.fit()
# print(results.get_best_result(metric="val_loss", mode="min").config)
# save_dict(results.get_best_result(metric="val_loss", mode="min").config, f"results/ray_tune/{output_type}_{coef_ind}_best_config_{anatomy}_{num_geos}.json")

# print("Hyperparameter optimization RRI Quadratic Resistance")
# output_type = "ri"
# coef_ind = 1
# tuner = tune.Tuner(obj, param_space=search_space) 
# results = tuner.fit()
# print(results.get_best_result(metric="val_loss", mode="min").config)
# save_dict(results.get_best_result(metric="val_loss", mode="min").config, f"results/ray_tune/{output_type}_{coef_ind}_best_config_{anatomy}_{num_geos}.json")
