from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os
from tabnanny import check
from tkinter import N

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

def split_junctions(tree_name):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    # junction_dict_master = get_input_file_junction_dict_master(tree_name)

    # time_step1 = "700"; time_step2 = "600"
    # flow_mag_list = ["12","25", "50", "100"]
    # add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    # add_downstream_resistance_values(junction_dict_master)
    # for flow_mag in flow_mag_list:
    #     add_solution_values(junction_dict_master, tree_name, flow_mag = flow_mag, time_step = time_step1)
    # add_3D_resistance(junction_dict_master, tree_name, flow_mag_list = flow_mag_list, time_step = time_step1)

    # #print("converting to RR")
    input_file_standard = f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)
    if not os.path.exists(f'trees/zerod_input/unsplit_bifs/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/unsplit_bifs/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/unsplit_bifs/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/unsplit_bifs/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/unsplit_bifs/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
        
    max_branch = 0
    for vessel in input_file["vessels"]:
        branch =  int(vessel["vessel_name"].split("_")[0][6:])
        max_branch = max(max_branch, branch)
        
    num_junctions = len(input_file["junctions"])
    num_vessels = len(input_file["vessels"])
    original_junctions = copy.deepcopy(input_file["junctions"])
    #pdb.set_trace()
    for i, junction_dict in enumerate(original_junctions):
        outlet_vessel_list = input_file["junctions"][i]["outlet_vessels"]
        junction_name = input_file["junctions"][i]["junction_name"]
        if len(outlet_vessel_list) == 1:
            continue
        input_file["junctions"][i]["junction_type"] = "BloodVesselJunction"
        try:
            area_list = input_file["junctions"][i]["areas"]
        except:
            pdb.set_trace()
        total_outlet_area = sum(area_list[1:])
        length_list = input_file["junctions"][i]["lengths"]
        tangent_list = input_file["junctions"][i]["tangents"]
        num_connectors = 0
        if len(outlet_vessel_list) == 2:
            continue
        while len(outlet_vessel_list) > 2:
            num_vessels += 1
            # Fix the outlet list for the current junction
            junction_dict["outlet_vessels"] = [outlet_vessel_list[0], copy.copy(num_vessels)]
            junction_dict["areas"] = [area_list[0], area_list[1], sum(area_list[1:])]
            junction_dict["lengths"] = [length_list[0], 0]
            junction_dict["tangents"] = [tangent_list[1], tangent_list[0]]
            
            outlet_vessel_list.pop(0);area_list.pop(1);length_list.pop(0);tangent_list.pop(1)
            # Add the connector vessel
            input_file["vessels"].append({
                "vessel_id": num_vessels,
                "vessel_name": f"{junction_dict['junction_name']}_connector_{num_connectors}",
                "vessel_type": "BloodVessel",
                "zero_d_element_values": {
                    "R_poiseuille": 0,
                    "stenosis_coefficient": 0,
                    "L": 0,
                    "C": 0,},
                "vessel_length": 0,
                })
            # Add the new junction
            input_file["junctions"].append({
                "inlet_vessels": [num_vessels,],
                "junction_name": f"J{num_junctions}",
                "junction_type": "BloodVesselJunction",
                "outlet_vessels": outlet_vessel_list,
            })
            num_connectors += 1
            num_junctions += 1
            junction_dict = input_file["junctions"][-1]
            #pdb.set_trace()
        junction_dict["outlet_vessels"] = [outlet_vessel_list[0], outlet_vessel_list[1]]
        junction_dict["areas"] = [area_list[0], area_list[1], sum(area_list[1:])]
        junction_dict["lengths"] = [length_list[0], 0]
        junction_dict["tangents"] = [tangent_list[1], tangent_list[0]]
            


        
    for vessel in input_file["vessels"]:
        if "branch0" not in vessel["vessel_name"]:
            vessel["zero_d_element_values"]["R_poiseuille"] = 0
            vessel["zero_d_element_values"]["stenosis_coefficient"] = 0
            vessel["zero_d_element_values"]["L"] = 0
        else:
            continue
        
    if not os.path.exists(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/standard/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/standard/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    #print(f"RRI 0D input file saved to trees/zerod_input/RR/{tree_name_base}/{tree_name}/solver_0d.json")
    
    
if __name__ == "__main__":
    

    tree_name = sys.argv[1]
    print(f"Splitting junctions for {tree_name}")
    split_junctions(tree_name)
    print("Done splitting junctions")