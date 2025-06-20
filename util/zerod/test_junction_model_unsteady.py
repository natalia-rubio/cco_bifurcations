import os
import sys
import pdb
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.zerod.standard_to_RR import transform_standard_to_RR

tree_name = sys.argv[1]
junction_mode = sys.argv[2]

casadi = True
tree_name_split = tree_name.split("_")
tree_name_base = "_".join(tree_name_split[0:2])

tree_name_split[-1] = "unsteady"
tree_name_unsteady = "_".join(tree_name_split)
# If a standard 0D file doesn't exist, create it
if not os.path.exists(f"trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json"):
    print(f"Steady 0D input file not found for {tree_name} in {junction_mode}. Run test_junction_model.py first.")
if junction_mode == "RR":
    transform_standard_to_RR(tree_name)
    #os.system(f"python3 util/zerod/standard_to_RR.py {tree_name}")
os.system(f"python3 util/zerod/to_unsteady.py {tree_name} {junction_mode}")



# # If special junction handling is needed, create and run the modified 0D file
if junction_mode != "standard" and not os.path.exists(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name_unsteady}/sol_casadi.csv"):
    os.system(f"python3 util/zerod/svzerod_to_casadi_unsteady.py {tree_name_unsteady} {junction_mode}")
    
# Project the 0D results to the 3D centerline
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_unsteady} {junction_mode}")

# Compare the 0D and 3D results
# print("Centerline errors:")
# os.system(f"python3 util/fidelity_comparison/compare_to_3d.py {junction_mode} {tree_name}")
print("Inlet errors:")
os.system(f"python3 util/fidelity_comparison/compare_to_3d_inlet_unsteady.py {junction_mode} {tree_name_unsteady}")

    







