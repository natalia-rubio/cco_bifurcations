import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
# from util.tree.get_0d_junction_dict import get_input_file_junction_dict_master
from util.tree.get_0d_res_dict import get_input_file_junction_dict_master
from util.tree.extract_true_junctions_helpers import *


if __name__ == "__main__":
    tree_name = "tree_20"; time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["25", "50", "100", "150"]
    junction_dict_master = get_input_file_junction_dict_master(tree_name) #get_input_file_junction_dict_master(tree_name)
    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    for time_step in [time_step1, time_step2]:
        for flow_mag in flow_mag_list:
            add_solution_values(junction_dict_master, tree_name, flow_mag, time_step = time_step)

    add_downstream_resistance_values(junction_dict_master)
    add_3D_resistance(junction_dict_master, tree_name, flow_mag_list, time_step = time_step1)

    add_0D_resistance(junction_dict_master, tree_name)
    converged = check_steady_state_convergence(junction_dict_master, tree_name, flow_mag_list, time_step1, time_step2)
    if converged:   print("Verified 3D solution steady state convergence!")
    add_3D_isol_values(junction_dict_master, tree_name)
    
    make_pdfs(junction_dict_master, tree_name, flow_mag_list=flow_mag_list, time_step=time_step1)
    save_dict(junction_dict_master, f"trees/reports/{tree_name}/junction_dict.json")
    # Check if the solution is converged
