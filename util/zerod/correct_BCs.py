import json
import pdb
import sys
import os

import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import *


def correct_BCs(tree_name):
    #pdb.set_trace()
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    junction_dict_master = get_input_file_junction_dict_master(tree_name)

    time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["25", "50", "100", "150"]
    flow_mag_index = flow_mag_list.index(flow_mag)
    #pdb.set_trace()
    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    for time_step in [time_step1, time_step2]:
        for flow_mag in flow_mag_list:
            add_solution_values(junction_dict_master, f"{tree_name_base}_flow_{flow_mag}", flow_mag, time_step = time_step)

    add_3D_resistance(junction_dict_master, tree_name_base, flow_mag_list, time_step = time_step1)
    add_3D_outlet_resistance(junction_dict_master, tree_name_base, flow_mag_list, time_step1)
    
    try:
        input_file_standard = f'trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}/solver_0d.json'

        with open(input_file_standard) as json_file:
            input_file = json.load(json_file)
    except:
        input_file_standard = f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json'

        with open(input_file_standard) as json_file:
            input_file = json.load(json_file)

    
    if not os.path.exists(f'trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/original_BCs/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/original_BCs/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)

    #pdb.set_trace()
    #print(f"Original 0D input file saved to trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}/solver_0d.json")

    bc_ind_dict = {}
    for ind, bc in enumerate(input_file["boundary_conditions"]):
        bc_ind_dict[bc["bc_name"]] = ind

    for junction_name, junction_dict in junction_dict_master.items():
        if junction_dict["0D_termination"] == "resistance":
            #pdb.set_trace()
            bc_ind = bc_ind_dict[junction_dict["0D_bc_resistance_name"]]
            terminal_outlet_vessel_id = junction_dict["0D_terminal_outlet_vessel_id"]
            input_file["boundary_conditions"][bc_ind]["bc_values"]["R"] = junction_dict["3D_outlet1_boundary_resistances"][flow_mag_index][0]
            #print(junction_dict["0D_bc_resistance_name"])
            #pdb.set_trace()
        
        if junction_dict["0D_aux_termination"] == "resistance":
            bc_ind = bc_ind_dict[junction_dict["0D_aux_bc_resistance_name"]]
            terminal_outlet_vessel_id = junction_dict["0D_aux_terminal_outlet_vessel_id"]
            input_file["boundary_conditions"][bc_ind]["bc_values"]["R"] = junction_dict["3D_outlet2_boundary_resistances"][flow_mag_index][0]
            #print(junction_dict["0D_aux_bc_resistance_name"])
        # try:
        #     if junction_dict["0D_aux_bc_resistance_name"] == "RESISTANCE_15" or junction_dict["0D_bc_resistance_name"]== "RESISTANCE_15":
        #         #pdb.set_trace()
        # except:
        #     continue
    # Save the modified input file
    if not os.path.exists(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/standard/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/standard/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    #print(f"Original 0D input file saved to trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json")