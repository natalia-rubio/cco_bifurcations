import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
# from util.tree.get_0d_junction_dict import get_input_file_junction_dict_master
from util.tree.get_0d_res_dict import get_input_file_junction_dict_master
from util.tree.extract_true_junctions_helpers import *


if __name__ == "__main__":
    tree_name_base = "tree_20"; time_step1 = "700"; time_step2 = "600"
    tree_name_full = f"{tree_name_base}_flow_100"
    flow_mag_list = ["25", "50", "100", "150"]
    isol_set_list = ["dict_res_fs_ext", "dict_flat_3D_dim_ext"]
    junction_dict_master = get_input_file_junction_dict_master(tree_name_full) #get_input_file_junction_dict_master(tree_name_base)
    add_geometry_values(junction_dict_master, tree_name_full, flow_mag = flow_mag_list[0], time_step = time_step1)
    for time_step in [time_step1, time_step2]:
        for flow_mag in flow_mag_list:
            add_solution_values(junction_dict_master, f"{tree_name_base}_flow_{flow_mag}", flow_mag, time_step = time_step)

    add_downstream_resistance_values(junction_dict_master)
    add_3D_resistance(junction_dict_master, tree_name_full, flow_mag_list, time_step = time_step1)

    add_0D_resistance(junction_dict_master, tree_name_base)
    converged = check_steady_state_convergence(junction_dict_master, tree_name_base, flow_mag_list, time_step1, time_step2)
    if converged:   print("Verified 3D solution steady state convergence!")
    for isol_set_name in isol_set_list:
        print(f"Adding isol values for {isol_set_name}")
        add_3D_isol_values(junction_dict_master, tree_name_base, isol_set_name)
    plot_flow_splits(junction_dict_master, tree_name_base, flow_mag_list, time_step)
    get_statistics(junction_dict_master, tree_name_base, flow_mag_list, time_step, isol_set_list)
    make_pdfs(junction_dict_master, tree_name_base, flow_mag_list=flow_mag_list, time_step=time_step1, isol_set_list = isol_set_list)
    save_dict(junction_dict_master, f"trees/reports/{tree_name_base}/junction_dict.json")
    # Check if the solution is converged
