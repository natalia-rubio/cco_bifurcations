import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from util.tools.basic import *
from util.neural_net.nn_util import init_weights, batched_forward_pass, inv_scale_jax, relu
import optax

#   m1: inlet_pressure_recovery_resistor
#       outlet1_pressure_recovery_resistor
#       outlet1_pressure_resistor
#       outlet2_pressure_recovery_resistor
#       outlet2_pressure_resistor


class NeuralNet():
   
    def __init__(self, network_params, optimizer_params):
        self.anatomy        = network_params["anatomy"]; self.set_type = network_params["set_type"]
        self.data_dict      = load_dict(f"data/jax_arrays/{self.anatomy}/{self.set_type}/jax_arrays_num_geos_{network_params["num_geos"]}") # Load all data
        self.scaling_dict   = load_dict(f"data/scaling_dictionaries/{self.anatomy}_{self.set_type}_scaling_dict")
        
        self.pred_mode      = network_params["pred_mode"]
        self.weights        = init_weights(network_params)
        self.num_input_features = network_params["num_input_features"]
        self.num_layers     = network_params["num_layers"]
        self.layer_width    = network_params["layer_width"]
        if self.pred_mode == "m1":
            self.num_output_features = network_params["num_output_features"]
        self.num_geos       = network_params["num_geos"]
        self.decay_rate     = optimizer_params["decay_rate"]

        # rate_factor = ((count - transition_begin) / transition_steps)
        # decayed_value = init_value * (decay_rate ** rate_factor)
        self.scheduler = optax.exponential_decay(init_value = optimizer_params["init"], 
                                                 transition_steps = optimizer_params["transition_steps"], 
                                                 decay_rate = optimizer_params["decay_rate"])
        self.optimizer = optax.adam(learning_rate = self.scheduler)
        self.opt_state = self.optimizer.init(self.weights)
        return
    
    def update(self, indices):
        grads = grad(loss, argnums = -1)(self.data_dict["input"][indices,:],
            self.data_dict["flows"][indices,:,:],
            self.data_dict["dPs"][indices,:,:],
            self.data_dict["scaling_factors"][indices,:],
            self.scaling_dict,
            self.weights,
            )
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.weights = optax.apply_updates(self.weights, updates)
        return

#input = self.data_dict["geo"][indices,:]
@jit
def predict(input, weights):
    output = batched_forward_pass(input, weights)
    return output 

@jit
def loss(input, flow, dP_true, scaling_factors, scaling_dict, weights):

    coefs_pred = predict(input, weights)
    #print(coefs_pred)

    R_lin_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[:,0], "inlet_R_lin_star") * 0
    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[:,1], "daughter1_R_lin_star")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[:,2], "daughter2_R_lin_star") * 0
    R_quad_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[:,3], "inlet_R_quad_star") * 0
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[:,4], "daughter1_R_quad_star")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[:,5], "daughter2_R_quad_star") * 0


    R_lin_star_pred_inlet = 0 * R_lin_star_pred_inlet # No linear inlet resistor
    R_quad_star_pred_inlet = relu(R_quad_star_pred_inlet) # Negative quadratic inlet resistor

    A_char = scaling_factors[:,0].reshape(-1,1)
    U_char = scaling_factors[:,1].reshape(-1,1)

    dP_star_pred = jnp.zeros(flow.shape)
    inflow = flow[:,:,0] + flow[:,:,1]
    dP_star_pred = dP_star_pred.at[:,:,0].set(jnp.divide(R_lin_star_pred_inlet * inflow, A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred_inlet * jnp.square(inflow), jnp.square(A_char * U_char)) +
                                            jnp.divide(R_lin_star_pred1 * flow[:,:,0], A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred1 * jnp.square(flow[:,:,0]), jnp.square(A_char * U_char)))
    
    dP_star_pred = dP_star_pred.at[:,:,1].set(jnp.divide(R_lin_star_pred_inlet * inflow, A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred_inlet * jnp.square(inflow), jnp.square(A_char * U_char)) +
                                            jnp.divide(R_lin_star_pred2 * flow[:,:,1], (A_char * U_char)) + 
                                            jnp.divide(R_quad_star_pred2 * jnp.square(flow[:,:,1]), jnp.square(A_char * U_char)))
    
    dP_pred = jnp.multiply(dP_star_pred, 1.06 * jnp.square(U_char.reshape(-1,1,1)))

    return jnp.sqrt(jnp.mean(jnp.square((dP_pred - dP_true)/1333)))

#@jit
def coef_loss(output, flow, dP_true, scaling_factors, scaling_dict):

    R_lin_star_pred_inlet = 0 #inv_scale_jax(scaling_dict, output[:,0], "R_lin_star_inlet")
    R_lin_star_pred1 = inv_scale_jax(scaling_dict, output[:,1], "daughter1_R_lin_star")
    R_lin_star_pred2 = 0 #inv_scale_jax(scaling_dict, output[:,2], "daughter2_R_lin_star")
    R_quad_star_pred_inlet = 0 #inv_scale_jax(scaling_dict, output[:,3], "R_quad_star_inlet")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, output[:,4], "daughter1_R_quad_star")
    R_quad_star_pred2 = 0 #inv_scale_jax(scaling_dict, output[:,5], "R_quad_star2")


    A_char = scaling_factors[:,0].reshape(-1,1)
    U_char = scaling_factors[:,1].reshape(-1,1)

    dP_star_pred = jnp.zeros(flow.shape)
    inflow = flow[:,:,0] + flow[:,:,1]
    dP_star_pred = dP_star_pred.at[:,:,0].set(jnp.divide(R_lin_star_pred_inlet * inflow, A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred_inlet * jnp.square(inflow), jnp.square(A_char * U_char)) +
                                            jnp.divide(R_lin_star_pred1 * flow[:,:,0], A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred1 * jnp.square(flow[:,:,0]), jnp.square(A_char * U_char)))
    
    dP_star_pred = dP_star_pred.at[:,:,1].set(jnp.divide(R_lin_star_pred_inlet * inflow, A_char * U_char) + 
                                            jnp.divide(R_quad_star_pred_inlet * jnp.square(inflow), jnp.square(A_char * U_char)) +
                                            jnp.divide(R_lin_star_pred2 * flow[:,:,1], (A_char * U_char)) + 
                                            jnp.divide(R_quad_star_pred2 * jnp.square(flow[:,:,1]), jnp.square(A_char * U_char)))
    
    dP_pred = jnp.multiply(dP_star_pred, 1.06 * jnp.square(U_char.reshape(-1,1,1)))
    print(f"output: {output}")
    print(f"Predicted dP: {dP_pred}")
    print(f"True dP: {dP_true}")
    #pdb.set_trace()
    return jnp.sqrt(jnp.mean(jnp.square((dP_pred - dP_true)/1333)))
