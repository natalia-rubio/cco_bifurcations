import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.synthetic_data_processing.extract_synthetic_data import *
from util.synthetic_data_processing.synthesize_synthetic_data import *
# from util.synthetic_data_processing.assemble_graphs import *
from util.synthetic_data_processing.train_val_split import *
from util.synthetic_data_processing.get_jax_arrays import *
from util.synthetic_data_processing.get_scaling_dict import *

anatomy = sys.argv[1]
set_type= sys.argv[2]
unsteady_text = sys.argv[3] #false
unsteady = False
if unsteady_text == "unsteady":
    unsteady = True
print(f"Unsteady: {unsteady}")
extract_steady_flow_data(anatomy = anatomy, set_type = set_type, require4 =False)
print("Extracted simulation results.")

#get_coefs(anatomy = anatomy, set_type = set_type, rm_low_r2 = True, unsteady = unsteady)
print("Fitted dP(Q) coefficients.")

get_data_lists(anatomy, set_type = set_type, unsteady = unsteady)
get_scaling_dict(anatomy, set_type = set_type, unsteady = unsteady)
print("Generated scaling dictionary.")

generate_train_val_indices(anatomy = anatomy, set_type = set_type, unsteady = unsteady)
print("Generated train and validation indices.")

get_jax_arrays(anatomy = anatomy, set_type = set_type, unsteady = unsteady)
print("Saved jax arrays.")
