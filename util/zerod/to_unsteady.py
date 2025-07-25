import json
import pdb
import sys
import os

from matplotlib import lines
import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import *
from scipy.interpolate import interp1d

if __name__ == "__main__":
    
    tree_name = sys.argv[1]
    junction_type = sys.argv[2]
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    input_file_standard = f'trees/zerod_input/{junction_type}/{tree_name_base}/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    t = []
    Q = []

    #with open("util/zerod/inflow_svFSI_flow_unsteady.flow", 'r') as file:
    try:    
        with open(f"trees/geo_files/{tree_name_base}/{tree_name_base}_flow_unsteady/inflow_svFSI_flow_unsteady.flow", 'r') as file:
            for line in file:
                # Split the line by whitespace
                columns = line.split("    ")
                #pdb.set_trace()
                if len(columns) == 2:
                    # Append data from each column to the corresponding list
                    t.append(float(columns[0]))
                    Q.append(-1*float(columns[1][:-2]))
                else:
                    print(f"Skipping line with unexpected format: {line}")
                    
            inlet_area = np.sqrt(8*np.pi*0.04*input_file["vessels"][0]["vessel_length"]/input_file["vessels"][0]["zero_d_element_values"]["R_poiseuille"])
            #pdb.set_trace()
            t = t[1:801]  # Remove the first time point
            Q = [q * inlet_area for q in Q[1:801]]  # Scale flow by inlet area
        #inlet_area = input_file["junctions"][0]["areas"][0]
    except:
        pdb.set_trace()
        with open(f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/inflow.flow", 'r') as file:
            for line in file:
                # Split the line by whitespace
                columns = line.split(" ")
                #pdb.set_trace()
                if len(columns) == 2:
                    # Append data from each column to the corresponding list
                    t.append(float(columns[0]))
                    Q.append(-1*float(columns[1][:-2]))
                else:
                    print(f"Skipping line with unexpected format: {line}")
        #inlet_area = input_file["junctions"][0]["areas"][0]
        

    #Q = [q * 80/max(Q) for q in Q]  # Scale flow by 80/84 to match the steady state flow
    
    t_fine = jnp.linspace(t[0], t[-1], 400)
    Q_fine = interp1d(t, Q, kind='linear', fill_value="extrapolate")(t_fine)
    t_fine = t_fine.tolist()
    Q_fine = Q_fine.tolist()

    num_repeats =  2
    input_file["boundary_conditions"][0]["bc_values"]["Q"]= num_repeats * Q_fine
    input_file["boundary_conditions"][0]["bc_values"]["t"]= np.linspace(0, num_repeats * (t_fine[-1] - t_fine[0]), num_repeats * len(t_fine)).tolist()
    
    input_file["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = len(input_file["boundary_conditions"][0]["bc_values"]["t"]) 
    input_file["simulation_parameters"]["number_of_cardiac_cycles"] = 1
    
    for vessel in input_file["vessels"]:
        if "branch0" in vessel["vessel_name"]:
            vessel["zero_d_element_values"]["C"] = 0

    tree_name_split = tree_name.split("_")
    tree_name_split[-1] = "unsteady"
    tree_name = "_".join(tree_name_split)

    if not os.path.exists(f'trees/zerod_input/{junction_type}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/{junction_type}/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/{junction_type}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/{junction_type}/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/{junction_type}/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input/{junction_type}/{tree_name_base}/{tree_name}/solver_0d.json")