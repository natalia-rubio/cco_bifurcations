from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os

import pandas as pd

#from util.zerod.svzerod_to_casadi_exact import R_lin

#from util.zerod.svzerod_to_casadi_exact import R_quad
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, add_solution_values, add_3D_resistance

def check_out_of_dist(param, param_name, scaling_dict, verbose = True):
    if param < scaling_dict[param_name][2]:
        #if verbose and param_name != "daughter1_length_star" and param_name != "daughter2_length_star" and param *0.9 < scaling_dict[param_name][2]:
        if verbose:
            print(f"{param_name} smaller than training set minimum: {param}, {scaling_dict[param_name][2]}")
        param = scaling_dict[param_name][2]
    if param > scaling_dict[param_name][3]:
        #if verbose and param_name != "daughter1_length_star" and param_name != "daughter2_length_star" and param *1.1 > scaling_dict[param_name][3]:
        if verbose:
            print(f"{param_name} larger than training set maximum: {param}, {scaling_dict[param_name][3]}")
        param = scaling_dict[param_name][3]
    return param



anatomy = "tree_20"
set_type = "random"
num_geos = 2968
data_dict      = load_dict(f"data/jax_arrays/{anatomy}/{set_type}/jax_arrays_num_geos_{num_geos}") # Load all data
scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
model_name = "rri_tree_20_ng_2968_nl_1_lw_5_ne_10_bs_100_dr_0.95_model" #"rri_tree_20_ng_1310_nl_1_lw_120_ne_5000_bs_50_dr_0.95_model"
#model_name = "tree_20_ng_950_nl_2_lw_200_ne_5000_bs_100_dr_0.95_model" #"tree_20_ng_400_nl_2_lw_100_ne_1000_bs_20_dr_0.95_model"
#model_name = "tree_20_ng_280_nl_3_lw_500_ne_2500_bs_20_dr_0.95_model" # "tree_20_ng_400_nl_2_lw_100_ne_1000_bs_20_dr_0.95_model"
nn_model = dill_load(f"results/models/{anatomy}/{model_name}")


re_char = 4500
A_char = 1
U_char = re_char * 0.04 / (1.06 * 2 *np.sqrt(A_char/np.pi))
L_char = np.sqrt(A_char/np.pi)



input_tens1 = jnp.asarray(data_dict["input"][0:1,:])

coefs_pred1 = predict(input_tens1, nn_model.weights)


R_lin_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][0], "daughter1_R_lin_star")[0][0])
R_lin_star_pred1 = check_out_of_dist(R_lin_star_pred1, "daughter1_R_lin_star", scaling_dict)

C = np.exp(-4)
R_quad_star_pred1_log = float(inv_scale_jax(scaling_dict, coefs_pred1[0][1], "daughter1_R_quad_star_logC")[0][0]); 
R_quad_star_pred1_log = check_out_of_dist(R_quad_star_pred1_log, "daughter1_R_quad_star_logC", scaling_dict)
R_quad_star_pred1 = np.sign(R_quad_star_pred1_log) * C * (np.exp(np.abs(R_quad_star_pred1_log))-1)
R_quad_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][1], "daughter1_R_quad_star")[0][0])
R_quad_star_pred1 = check_out_of_dist(R_quad_star_pred1, "daughter1_R_quad_star", scaling_dict)

if coefs_pred1.size > 2:
    L_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][2], "daughter1_L_star")[0][0]); 
    L_star_pred1 = check_out_of_dist(L_star_pred1, "daughter1_L_star", scaling_dict)
else:
    L_star_pred1 = 0.0


daughter1_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)))
daughter1_R_lin_final = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char))) 
#pdb.set_trace()
daughter1_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char))
daughter1_L = float(L_star_pred1) *1.06*L_char/A_char
pdb.set_trace()
