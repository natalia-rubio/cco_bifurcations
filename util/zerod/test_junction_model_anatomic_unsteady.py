import os
import sys
import pdb
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.zerod.standard_to_RR import transform_standard_to_RR
from util.zerod.standard_to_RI import transform_standard_to_RI

tree_name = sys.argv[1]
junction_mode = sys.argv[2]

casadi = True
tree_name_split = tree_name.split("_")
tree_name_base = "_".join(tree_name_split[0:2])

tree_name_split[-1] = "unsteady"
tree_name_unsteady = "_".join(tree_name_split)
# If a standard 0D file doesn't exist, create it


print("In anatomic unsteady junction model test script")
if not os.path.exists(f"trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json"):
    print("Making directory for 0D input file")

    os.makedirs(f"trees/zerod_input/standard/{tree_name_base}/{tree_name}", exist_ok=True)
    print("Creating 0D input file")
    os.system(f"python3 util/zerod/get_inp_standard_anatomical.py {tree_name} 100")

    #os.system(f"python3 util/zerod/standard_to_RR.py {tree_name}")
#os.system(f"python3 util/zerod/remove_C.py {tree_name} standard")
print("Splitting junctions")
os.system(f"python3 util/zerod/split_junctions.py {tree_name}")
os.system(f"python3 util/zerod/to_unsteady.py {tree_name} standard")

transform_standard_to_RR(tree_name_unsteady)
transform_standard_to_RI(tree_name_unsteady)

# # If special junction handling is needed, create and run the modified 0D file
#if junction_mode != "standard" and not os.path.exists(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name_unsteady}/sol_casadi.csv"):
os.system(f"python3 util/zerod/svzerod_to_casadi_unsteady_acc.py {tree_name_unsteady} RRI")
os.system(f"python3 util/zerod/svzerod_to_casadi_unsteady_acc.py {tree_name_unsteady} RI")
    
# Project the 0D results to the 3D centerline
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_unsteady} standard")
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_unsteady} RRI")
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_unsteady} RI")

# Compare the 0D and 3D results
# print("Centerline errors:")
# os.system(f"python3 util/fidelity_comparison/compare_to_3d.py {junction_mode} {tree_name}")
print("Inlet errors:")
os.system(f"python3 util/fidelity_comparison/compare_to_3d_inlet_unsteady.py {junction_mode} {tree_name_unsteady}")