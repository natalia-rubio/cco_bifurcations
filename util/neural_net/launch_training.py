
import pdb
import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.neural_net.nn_model import NeuralNet
from util.neural_net.train_nn import train_nn
from util.tools.basic import load_dict


def launch_training(network_params, optimizer_params, training_params):
    network_params["output_type"] = "rri"
    
    #for outlet_number in ["o1", "o2"]:
    
    print("Training RRI model...")

    # print(f"training model 1:  Linear Resistor Outlet 1")
    # network_params["outlet_number"] = "o1"
    # network_params["target_coef_ind"] = 0
    # network_params["layer_width"] = 9
    # network_params["num_layers"] = 1
    # training_params["num_epochs"] = 500
    # optimizer_params["decay_rate"] = 0.7
    # optimizer_params["init"] = 0.01
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 1:  Linear Resistor Outlet 2")
    # network_params["outlet_number"] = "o2"
    # network_params["target_coef_ind"] = 0
    # network_params["layer_width"] = 40
    # network_params["num_layers"] = 1
    # training_params["num_epochs"] = 500
    # optimizer_params["decay_rate"] = 0.7
    # optimizer_params["init"] = 0.01
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 2:  Quadratic Resistor Outlet 1")
    # network_params["outlet_number"] = "o1"
    # network_params["target_coef_ind"] = 1
    # network_params["layer_width"] = 30
    # training_params["num_epochs"] = 500
    # network_params["num_layers"] = 1
    # training_params["learning_rate"] = 0.00012
    # optimizer_params["decay_rate"] = 0.7
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 2:  Quadratic Resistor Outlet 2")
    # network_params["outlet_number"] = "o2"
    # network_params["target_coef_ind"] = 2
    # network_params["layer_width"] = 25
    # training_params["num_epochs"] = 500
    # network_params["num_layers"] = 1
    # training_params["learning_rate"] = 0.00012
    # optimizer_params["decay_rate"] = 0.6
    # optimizer_params["init"] = 0.02
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 3:  Inductor")
    # network_params["outlet_number"] = "o1"
    # network_params["num_layers"] = 1
    # network_params["target_coef_ind"] = 2
    # network_params["layer_width"] = 12
    # training_params["num_epochs"] = 300
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    print(f"training model 3:  Inductor")
    network_params["outlet_number"] = "o2"
    network_params["num_layers"] = 1
    network_params["target_coef_ind"] = 2
    network_params["layer_width"] = 12
    training_params["num_epochs"] = 300
    model = NeuralNet(network_params, optimizer_params)
    train_nn(model, training_params)
        
    # network_params["output_type"] = "ri"
    # print("Training RI model...")
    # print(f"training model 1:  Linear Resistor Outlet 1")
    # network_params["outlet_number"] = "o1"
    # network_params["target_coef_ind"] = 0
    # network_params["layer_width"] = 10
    # network_params["num_layers"] = 1
    # training_params["num_epochs"] = 800
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # network_params["output_type"] = "ri"
    # print("Training RI model...")
    # print(f"training model 1:  Linear Resistor Outlet 2")
    # network_params["outlet_number"] = "o2"
    # network_params["target_coef_ind"] = 0
    # network_params["layer_width"] = 20
    # network_params["num_layers"] = 1
    # training_params["num_epochs"] = 800
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 3:  Inductor Outlet 1")
    # network_params["outlet_number"] = "o1"
    # network_params["target_coef_ind"] = 1
    # network_params["layer_width"] = 20
    # training_params["num_epochs"] = 500
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    # print(f"training model 3:  Inductor Outlet 2")
    # network_params["outlet_number"] = "o2"
    # network_params["target_coef_ind"] = 1
    # network_params["layer_width"] = 40
    # training_params["num_epochs"] = 500
    # optimizer_params["init"] = 0.01
    # model = NeuralNet(network_params, optimizer_params)
    # train_nn(model, training_params)
    
    return

if __name__ == "__main__":
    anatomy = sys.argv[1]
    num_geos = int(sys.argv[2])
    output_type = sys.argv[3]  # "rri", "ri", or "rr"
    set_type = "random_pared"
    #pdb.set_trace()

    split_ind_dict = load_dict(f"data/split_indices/{anatomy}/{set_type}/train_val_ind_{anatomy}_num_geos_{num_geos}")

    network_params = {"num_input_features": 10,
                      "num_layers": 1,
                      "layer_width":20,
                      "output_type": output_type,
                      "anatomy": anatomy,
                      "set_type": set_type,
                      "num_geos": num_geos,
                      "pred_mode": "m1"}
    
    training_params = {"num_epochs": 1000, 
                       "batch_size": 200,
                       "train_inds": split_ind_dict["train_ind"],
                       "val_inds": split_ind_dict["val_ind"],
                       "num_offsets": split_ind_dict["num_offsets"],}
    
    optimizer_params = {#"step_size": 0.0002,
                        "init" : 0.02,
                        "transition_steps": 1000,
                        "decay_rate" : 0.95}
    
    launch_training(network_params, optimizer_params, training_params)