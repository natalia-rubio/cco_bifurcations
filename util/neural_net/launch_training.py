import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.neural_net.nn_model import NeuralNet
from util.neural_net.train_nn import train_nn
from util.tools.basic import load_dict


def launch_training(network_params, optimizer_params, training_params):
    
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)
    return

if __name__ == "__main__":
    anatomy = sys.argv[1]
    num_geos = int(sys.argv[2])
    set_type = "random"

    split_ind_dict = load_dict(f"data/split_indices/{anatomy}/{set_type}/train_val_ind_{anatomy}_num_geos_{num_geos}")

    network_params = {"num_input_features": 10,
                      "num_layers": 1,
                      "layer_width": 120,
                      "num_output_features": 3,
                      "anatomy": anatomy,
                      "set_type": set_type,
                      "num_geos": num_geos,
                      "pred_mode": "m1"}
    
    training_params = {"num_epochs": 8000, 
                       "batch_size": 50,
                       "train_inds": split_ind_dict["train_ind"],
                       "val_inds": split_ind_dict["val_ind"]}
    
    optimizer_params = {#"step_size": 0.0002,
                        "init" : 0.00045,
                        "transition_steps": 10000,
                        "decay_rate" : 0.95}
    
    launch_training(network_params, optimizer_params, training_params)