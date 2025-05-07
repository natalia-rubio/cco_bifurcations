import os
import sys
import pdb
inflow_dict = {"tree_20": 84,
                "tree_20_flow_100": 84,
                "tree_20_flow_50": 42,
                "tree_20_flow_150": 126,
               "tree_80": 339,
               "tree_dec1": 646,
               "tree_dec":483.4}

tree_name = sys.argv[1]
junction_mode = sys.argv[2]

inflow = inflow_dict[tree_name]
casadi = True

# If a standard 0D file doesn't exist, create it
if not os.path.exists(f"trees/zerod_input/standard/{tree_name}/solver_0d.json"):
    os.makedirs(f"trees/zerod_input/standard/{tree_name}")
    os.system(f"python3 util/zerod/get_inp_standard.py {tree_name} {inflow}")

# If special junction handling is needed, create and run the modified 0D file
if junction_mode != "standard":

    if junction_mode == "RR":
        os.system(f"python3 util/zerod/standard_to_RR.py {tree_name}")
    elif junction_mode == "TP":
        os.system(f"python3 util/zerod/standard_to_TP.py {tree_name}")
    else:
        print("Invalid junction mode. Exiting.")
    
    # Solve the 0D equations with CasADi
    os.system(f"python3 util/zerod/svzerod_to_casadi_single.py {tree_name} {junction_mode}")

# Project the 0D results to the 3D centerline
os.system(f"python3 util/centerline_projection/project_0d_to_3d.py {tree_name} {junction_mode}")

# Compare the 0D and 3D results
print("Centerline errors:")
os.system(f"python3 util/fidelity_comparison/compare_to_3d.py {junction_mode} {tree_name}")
print("Inlet errors:")
os.system(f"python3 util/fidelity_comparison/compare_to_3d_inlet.py {junction_mode} {tree_name}")

    







