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

tree_name_split[-1] = "real"
tree_name_real = "_".join(tree_name_split)
#If a standard 0D file doesn't exist, create it
# if not os.path.exists(f"trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json"):
#     print(f"Steady 0D input file not found for {tree_name} in {junction_mode}. Run test_junction_model.py first.")

# #     os.system(f"python3 util/zerod/standard_to_RR.py {tree_name}")
os.system(f"python3 util/zerod/to_unsteady_real.py {tree_name} standard")
os.system(f"python3 util/zerod/remove_C.py {tree_name_real} standard")


transform_standard_to_RR(tree_name_real)
transform_standard_to_RI(tree_name_real)

# If special junction handling is needed, create and run the modified 0D file
if junction_mode != "standard" and not os.path.exists(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name_real}/sol_casadi.csv"):
    pass
    #os.system(f"python3 util/zerod/svzerod_to_casadi_real_acc.py {tree_name_real} RRI")
    #os.system(f"python3 util/zerod/svzerod_to_casadi_real_acc.py {tree_name_real} RI")
    
# # # # Project the 0D results to the 3D centerline
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_real} standard")
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_real} RRI")
os.system(f"python3 util/centerline_projection/project_0d_to_3d_unsteady.py {tree_name_real} RI")

# Compare the 0D and 3D results

print("Inlet errors:")
os.system(f"python3 util/fidelity_comparison/compare_to_3d_inlet_real.py {junction_mode} {tree_name_real}")

    







