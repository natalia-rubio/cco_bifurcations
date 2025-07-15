import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.neural_net.nn_util import get_L2, init_weights, batched_forward_pass, inv_scale_jax, relu, dill_load
import optax

class NeuralNet():
   
    def __init__(self, network_params, optimizer_params):
        self.anatomy        = network_params["anatomy"]; self.set_type = network_params["set_type"]
        self.data_dict      = load_dict(f"/Users/natalia/Desktop/cco_bifurcations/data/jax_arrays/{self.anatomy}/{self.set_type}/jax_arrays_num_geos_{network_params["num_geos"]}") # Load all data
        self.scaling_dict   = load_dict(f"/Users/natalia/Desktop/cco_bifurcations/data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        
        self.output_type    = network_params["output_type"]
        self.target_coef_ind = network_params["target_coef_ind"]
        #model_name = "tree_20_ng_1220_nl_1_lw_40_ne_5000_bs_50_dr_0.95_model"
        #nn_model = dill_load(f"results/models/{network_params["anatomy"]}/{model_name}")
        self.weights        =  init_weights(network_params) # 
        #self.weights = nn_model.weights# jnp.asarray([1.005]) #
        self.num_input_features = network_params["num_input_features"]
        self.num_layers     = network_params["num_layers"]
        self.layer_width    = network_params["layer_width"]
        if self.output_type == "rri":
            self.num_output_coefs = 3
            self.output = self.data_dict["output_rri"]
        elif self.output_type == "ri":
            self.num_output_coefs = 2
            self.output = self.data_dict["output_ri"]
        elif self.output_type == "rr":
            self.num_output_coefs = 2
            self.output = self.data_dict["output_rr"]
        self.num_output_features = 1
        # self.num_output_features = 1
        # self.output = self.data_dict["output_rri"][:,2:3]
            
        self.num_geos       = network_params["num_geos"]
        self.decay_rate     = optimizer_params["decay_rate"]

        self.scheduler = optax.exponential_decay(init_value = optimizer_params["init"], 
                                                 transition_steps = optimizer_params["transition_steps"], 
                                                 decay_rate = optimizer_params["decay_rate"])
        self.optimizer = optax.adam(learning_rate = self.scheduler)
        self.opt_state = self.optimizer.init(self.weights)
        return
    
    def update(self, indices):
        grads = grad(loss, argnums = -1)(self.data_dict["input"][indices,:],
            self.output[indices,:],
            self.data_dict["scaling_factors"][indices,:],
            self.scaling_dict,
            self.target_coef_ind,
            self.weights,
            )
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
        return


@jit
def predict(input, weights):
    output = batched_forward_pass(input, weights)
    return output 

@jit
def loss(input, outputs, scaling_factors, scaling_dict, target_coef_ind, weights):
    coefs_pred = predict(input, weights)
    #pdb.set_trace()
    L2_penalty = get_L2(weights)/(len(weights) * jnp.size(weights[0][0]))
    return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:,target_coef_ind] - outputs[:,target_coef_ind]))) + L2_penalty*0 #*1#L2 regularization term

@jit
def loss_pure(input, outputs, scaling_factors, scaling_dict, target_coef_ind, weights):
    coefs_pred = predict(input, weights)
    #pdb.set_trace()
    L2_penalty = get_L2(weights)/(len(weights) * jnp.size(weights[0][0]))
    return jnp.sqrt(jnp.mean(jnp.square(coefs_pred[:,target_coef_ind] - outputs[:,target_coef_ind]))) #L2 regularization term



