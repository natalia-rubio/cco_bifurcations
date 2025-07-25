import os
import sys
import pdb
import time
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tree.centerline_proj import extract_results
from util.fidelity_comparison.compare_to_3d_inlet import compare_to_3d_inlet
from util.zerod.standard_to_RR import transform_standard_to_RR
from util.zerod.standard_to_RI import transform_standard_to_RI
from util.zerod.standard_to_RRI_full_fit import transform_standard_to_RRI_full_fit
from util.zerod.standard_to_RRI_junctions_fit import transform_standard_to_RRI_junctions_fit
from util.zerod.standard_to_RRI_nn_branch_fit import transform_standard_to_RRI_nn_branch_fit
from util.zerod.standard_to_RI_full_fit import transform_standard_to_RI_full_fit
from util.zerod.standard_to_RI_nn_branch_fit import transform_standard_to_RI_nn_branch_fit
from util.zerod.standard_to_RI_junctions_fit import transform_standard_to_RI_junctions_fit
#from util.zerod.svzerod_to_casadi_ws import solve_casadi_single
#from util.zerod.svzerod_to_casadi_ws import solve_casadi_single
from util.zerod.correct_BCs import correct_BCs

inflow_dict = {"tree_3_flow_12": 7.56, #12,
               "tree_3_flow_25": 16,#20,
                "tree_3_flow_50": 32,#40,
                "tree_3_flow_100": 63, #80,
                "tree_3_flow_150": 120,
                "tree_5_flow_12": 11.25, #12*.5,
                "tree_5_flow_25": 22.5,#45*.5,
                "tree_5_flow_50": 45,#*.5,
                "tree_5_flow_100": 90,#179*.5,
                "tree_5_flow_150": 134, #270*.5,
                "tree_10_flow_12": 7.65, #12*1.875,
                "tree_10_flow_25": 16,
                "tree_10_flow_50": 32,
                "tree_10_flow_100": 64,
                "tree_10_flow_150": 108,
                "tree_20_flow_25": 21,
                "tree_20_flow_50": 42,
                "tree_20_flow_100": 84,
                "tree_20_flow_150": 126,
                "tree_40_flow_12": 9.57, #12*3.75,
                "tree_40_flow_25": 20,
                "tree_40_flow_50": 40,
                "tree_40_flow_100": 80,
                }


def test_junction_model_single(tree_name, junction_mode):
    """
    Test the junction model for a given tree and junction mode.
    """
    # if junction_mode not in ["standard", "RRI", "RI", "RI_full_fit", "RI_junctions_fit", "RI_nn_branch_fit"]:
    #     raise ValueError("Invalid junction mode. Choose from 'standard', 'RRI', or 'RI'.")
    
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    time_step = 700
    if not os.path.exists(f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_{time_step}.vtp"):
        fpath_out = f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_{time_step}.vtp"
        fpath_3d  = f"trees/threed_results/{tree_name_base}/{tree_name}/{tree_name_base}_{flow_mag}_result_{time_step}.vtu"
        fpath_1d  = f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/centerlines/centerlines.vtp"
        if not os.path.exists(fpath_3d):    print(f"3D results for {tree_name} not found at {fpath_3d}. Exiting.")
        else:
            os.makedirs(f"trees/threed_output_cent/{tree_name_base}/{tree_name}", exist_ok=True)
            extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 50)


    inflow = inflow_dict[tree_name]
    # If a standard 0D file doesn't exist, create it
    if not os.path.exists(f"trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json"):
        os.makedirs(f"trees/zerod_input/standard/{tree_name_base}/{tree_name}", exist_ok=True)
        os.system(f"python3 util/zerod/get_inp_standard.py {tree_name} {inflow}")

    # Correct BCs
    # correct_BCs(tree_name)

    # If special junction handling is needed, create and run the modified 0D file
    if junction_mode == "RI":
        transform_standard_to_RI(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RI")
    elif junction_mode == "RRI":
        transform_standard_to_RR(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RRI")
    elif junction_mode == "RI_full_fit":
        transform_standard_to_RI_full_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RI_full_fit")
    elif junction_mode == "RI_junctions_fit":
        transform_standard_to_RI_junctions_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RI_junctions_fit")
    elif junction_mode == "RI_nn_branch_fit":
        transform_standard_to_RI_nn_branch_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RI_nn_branch_fit")
    elif junction_mode == "RRI_full_fit":
        transform_standard_to_RRI_full_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RRI_full_fit")
    elif junction_mode == "RRI_junctions_fit":
        transform_standard_to_RRI_junctions_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RRI_junctions_fit")
    elif junction_mode == "RRI_nn_branch_fit":
        transform_standard_to_RRI_nn_branch_fit(tree_name)
        os.system(f"python3 util/zerod/svzerod_to_casadi_ws.py {tree_name} RRI_nn_branch_fit")
        

    # Project the 0D results to the 3D centerline
    os.system(f"python3 util/centerline_projection/project_0d_to_3d.py {tree_name} {junction_mode}")


    # Compare the 0D and 3D results
    error_dict = compare_to_3d_inlet(junction_mode, tree_name)

    return error_dict

if __name__ == "__main__":
    tree_name = sys.argv[1]
    junction_mode = sys.argv[2]
    test_junction_model_single(tree_name, junction_mode)







