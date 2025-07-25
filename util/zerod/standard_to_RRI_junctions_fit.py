from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os
from tabnanny import check

import pandas as pd

#from util.zerod.svzerod_to_casadi_exact import R_lin

#from util.zerod.svzerod_to_casadi_exact import R_quad
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, add_solution_values, add_3D_resistance_junction_only

def transform_standard_to_RRI_junctions_fit(tree_name):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    junction_dict_master = get_input_file_junction_dict_master(tree_name)

    time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["12","25", "50", "100"]
    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    add_downstream_resistance_values(junction_dict_master)
    for flow_mag in flow_mag_list:
        add_solution_values(junction_dict_master, tree_name, flow_mag = flow_mag, time_step = time_step1)
    add_3D_resistance_junction_only(junction_dict_master, tree_name, flow_mag_list = flow_mag_list, time_step = time_step1)

    #print("converting to RR")
    input_file_standard = f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    for i in range(len(input_file["junctions"])):
        junction_name = input_file["junctions"][i]["junction_name"]
        if junction_name not in junction_dict_master:
            #print(f"Junction {junction_name} not found in junction dictionary master")
            continue
        junction_dict = junction_dict_master[junction_name]
        daughter1_flow_ratio = junction_dict["0D_geo_flow_split"]/(1 + junction_dict["0D_geo_flow_split"])
        daughter2_flow_ratio = 1/(1 + junction_dict["0D_geo_flow_split"])


        input_file["junctions"][i]["junction_type"] = "BloodVesselJunction"
        input_file["junctions"][i]["junction_values"] = {"R_poiseuille": [junction_dict["3D_daughter1_R_lin_junction_only"],
                                                                          junction_dict["3D_daughter2_R_lin_junction_only"]],
                                                         "pressure_recovery_coefficient": [junction_dict["3D_daughter1_R_quad_junction_only"],
                                                                          junction_dict["3D_daughter2_R_quad_junction_only"]],
                                                         "stenosis_coefficient": [0,0],
                                                         "L": [0,0],
                                                         "flow_split": [daughter1_flow_ratio,daughter2_flow_ratio],
                                                         }
        
    
    if flow_mag == "12":
        num_pts = 5
    elif flow_mag == "25":
        num_pts = 10
    elif flow_mag == "50":
        num_pts = 20
    elif flow_mag == "100":
        num_pts = 40
    t = input_file["boundary_conditions"][0]["bc_values"]["t"] 
    t = np.linspace(t[0], t[-1], num_pts).tolist()  # Create a fine time vector
    Q = input_file["boundary_conditions"][0]["bc_values"]["Q"]
    Q = np.linspace(0, Q[-1], num_pts).tolist()  # Create a fine flow vector
        
    input_file["boundary_conditions"][0]["bc_values"]["t"] = t
    input_file["boundary_conditions"][0]["bc_values"]["Q"] = Q
        
    if not os.path.exists(f'trees/zerod_input/RRI_junctions_fit/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/RRI_junctions_fit/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/RRI_junctions_fit/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/RRI_junctions_fit/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/RRI_junctions_fit/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    #print(f"RRI 0D input file saved to trees/zerod_input/RR/{tree_name_base}/{tree_name}/solver_0d.json")