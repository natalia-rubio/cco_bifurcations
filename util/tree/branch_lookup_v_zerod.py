import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from collections import defaultdict


if __name__ == "__main__":
    flow_amp = "full"
    tree_name = "tree_80"
    zerod_gen = "sv"
    if zerod_gen == "CCO":
        input_file_standard = f'trees/zerod_input_CCO_standard/{tree_name}_{flow_amp}/solver_0d.json'
    else:
        input_file_standard = f'trees/zerod_input_sv_standard/{tree_name}_{flow_amp}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    resistance_dict = load_dict(f"data/tree_80_resistance_dict_branch")

    zerod_branch_dict = defaultdict(lambda: defaultdict(float))

    for vessel in input_file["vessels"]:
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        print(f"Branch ID: {branch_id}, Seg ID: {seg_id}")

        zerod_branch_dict[branch_id]["R"] += vessel["zero_d_element_values"]["R_poiseuille"] 
        zerod_branch_dict[branch_id]["length"] += vessel["vessel_length"]

    R_diff_list = []
    R_diff_perc_list = []

    #pdb.set_trace()
    for branch_id in resistance_dict["vessels"].keys():
        resistance_dict["vessels"][branch_id].update({"R_zerod": zerod_branch_dict[branch_id]["R"]})
        resistance_dict["vessels"][branch_id].update({"length_zerod": zerod_branch_dict[branch_id]["length"]})
        
        resistance_dict["vessels"][branch_id].update({"R_diff": -resistance_dict["vessels"][branch_id]["R"] + resistance_dict["vessels"][branch_id]["R_zerod"]})
        resistance_dict["vessels"][branch_id].update({"R_diff_perc": resistance_dict["vessels"][branch_id]["R_diff"] / abs(resistance_dict["vessels"][branch_id]["R"])})
        R_diff_list.append(resistance_dict["vessels"][branch_id]["R_diff"])
        R_diff_perc_list.append(resistance_dict["vessels"][branch_id]["R_diff_perc"])
        

    plt.hist(R_diff_perc_list, bins=50)
    plt.xlabel("0D Vessel Resistance Percent Discrepancy (0D - True)")
    plt.ylabel("Frequency")
    pdb.set_trace()
    plt.savefig(f"trees/geo_files/{tree_name}_resistance_diff.png", bbox_inches="tight")

    
    

