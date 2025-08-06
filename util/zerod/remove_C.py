from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os

import pandas as pd

#from util.zerod.svzerod_to_casadi_exact import R_quad
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, add_solution_values, add_3D_resistance


def remove_C(tree_name, junction_mode):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    
    input_file_standard = f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    for i in range(len(input_file["vessels"])):
        input_file["vessels"][i]["zero_d_element_values"]["C"] = 10**-10
        #input_file["vessels"][i]["zero_d_element_values"]["L"] = 0


    if not os.path.exists(f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    #print(f"RRI 0D input file saved to trees/zerod_input/RR/{tree_name_base}/{tree_name}/solver_0d.json")
        
if __name__ == "__main__":
    remove_C(sys.argv[1], sys.argv[2])