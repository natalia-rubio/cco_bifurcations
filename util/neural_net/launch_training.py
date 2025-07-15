from operator import ne
import pdb
import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.neural_net.nn_model import NeuralNet
from util.neural_net.train_nn import train_nn
from util.tools.basic import load_dict


def launch_training(network_params, optimizer_params, training_params):
    if network_params["output_type"] == "rri":
        print("Training RRI model...")

        # print(f"training model 1:  Linear Resistor")
        # network_params["target_coef_ind"] = 0
        # network_params["layer_width"] = 202
        # network_params["num_layers"] = 3
        # training_params["num_epochs"] = 1000
        # training_params["learning_rate"] = 0.0011
        # model = NeuralNet(network_params, optimizer_params)
        # train_nn(model, training_params)
        
        print(f"training model 2:  Quadratic Resistor")
        network_params["target_coef_ind"] = 1
        network_params["layer_width"] = 50
        training_params["num_epochs"] = 1000
        training_params["num_layers"] = 2
        training_params["learning_rate"] = 0.00012
        model = NeuralNet(network_params, optimizer_params)
        train_nn(model, training_params)
        
        print(f"training model 3:  Inductor")
        network_params["target_coef_ind"] = 2
        network_params["layer_width"] = 200
        training_params["num_epochs"] = 1000
        model = NeuralNet(network_params, optimizer_params)
        train_nn(model, training_params)
    if network_params["output_type"] == "ri":
        print("Training RI model...")
        print(f"training model 1:  Linear Resistor")
        network_params["target_coef_ind"] = 0
        network_params["layer_width"] = 100
        network_params["num_layers"] = 1
        model = NeuralNet(network_params, optimizer_params)
        train_nn(model, training_params)
        
        print(f"training model 3:  Inductor")
        network_params["target_coef_ind"] = 1
        network_params["layer_width"] = 200
        training_params["num_epochs"] = 1000
        model = NeuralNet(network_params, optimizer_params)
        train_nn(model, training_params)
    return

if __name__ == "__main__":
    anatomy = sys.argv[1]
    num_geos = int(sys.argv[2])
    output_type = sys.argv[3]  # "rri", "ri", or "rr"
    set_type = "random"
    #pdb.set_trace()

    split_ind_dict = load_dict(f"data/split_indices/{anatomy}/{set_type}/train_val_ind_{anatomy}_num_geos_{num_geos}")

    network_params = {"num_input_features": 10,
                      "num_layers": 2,
                      "layer_width":20,
                      "output_type": output_type,
                      "anatomy": anatomy,
                      "set_type": set_type,
                      "num_geos": num_geos,
                      "pred_mode": "m1"}
    
    training_params = {"num_epochs": 1000, 
                       "batch_size": 400,
                       "train_inds": split_ind_dict["train_ind"],
                       "val_inds": split_ind_dict["val_ind"],
                       "num_offsets": split_ind_dict["num_offsets"],}
    
    optimizer_params = {#"step_size": 0.0002,
                        "init" : 0.02,
                        "transition_steps": 5000,
                        "decay_rate" : 0.9}
    
    launch_training(network_params, optimizer_params, training_params)