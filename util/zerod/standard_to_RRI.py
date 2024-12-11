import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from neural_net.nn_util import scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict

def get_RRI_values(areas, tangents):
    anatomy = "CCO"
    set_type = "random"

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    
    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(areas[1]/areas[0], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(areas[2]/areas[0], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(tangents[0], tangents[1]), dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(tangents[0], tangents[2]), dtype=jnp.float32), "daughter2_angle"),
                                ])
    
    model_name = "CCO_ng_88_nl_0_lw_25_ne_5000_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

    coefs_pred = predict(input_tens, nn_model.weights)

    return r_lin, r_quad, L


tree_name = "tree_80"
with open(f'trees/zerod_input_standard/{tree_name}/solver_0d.json') as json_file:
    input_file = json.load(json_file)

for junction in input_file["junctions"]:
    if len(junction["outlet_vessels"]) > 2:
        print("Junction with more than 2 outlets.")
    elif len(junction["outlet_vessels"]) == 2:
        junction["junction_type"] = "BloodVesselJunction"

    r_lin, r_quad, L = get_RRI_values(areas = junction["areas"], tangets = junction["tangents"])

    junction["R_poiseuille"] = r_lin
    junction["stenosis_coefficient"] = r_quad
    junction["L"] = L
    junction["C"] = 0.0000001
pdb.set_trace()