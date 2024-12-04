import sys
sys.path.append("Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
# import tensorflow as tf
from os.path import exists



def generate_train_val_indices(anatomy, set_type, unsteady = True):
    seed = 0
    char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    num_geos = len(char_val_dict["inlet_area"])
    train_ind, val_ind = get_random_ind(num_pts = num_geos, percent_train = 80, seed = seed)

    if not os.path.exists(f"data/split_indices"):
        os.mkdir(f"data/split_indices")
    if not os.path.exists(f"data/split_indices/{anatomy}"):
        os.mkdir(f"data/split_indices/{anatomy}")  
    if not os.path.exists(f"data/split_indices/{anatomy}/{set_type}"):
        os.mkdir(f"data/split_indices/{anatomy}/{set_type}")
    save_dict({"train_ind": train_ind, "val_ind": val_ind}, 
              f"data/split_indices/{anatomy}/{set_type}/train_val_ind_{anatomy}_num_geos_{num_geos}")
    return

# def prepare_tensors(anatomy, set_type, unsteady = True):
#     char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
#     num_geos = len(char_val_dict["inlet_area"])
    
#     input_tens = tf.concat((tf.reshape(tf.convert_to_tensor(char_val_dict["daughter1_area_ratio"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["daughter2_area_ratio"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["daughter1_angle"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["daughter2_angle"], dtype=tf.float64), (num_geos, 1)),
#                                 ), axis = -1)

#     output_tens = tf.concat((tf.reshape(tf.convert_to_tensor(char_val_dict["R_lin_star1"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["R_lin_star2"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["R_quad_star1"], dtype=tf.float64), (num_geos, 1)),
#                             tf.reshape(tf.convert_to_tensor(char_val_dict["R_quad_star2"], dtype=tf.float64), (num_geos, 1)),
#                             ), axis = -1)
    
#     scaling_factors = tf.concat((tf.reshape(tf.convert_to_tensor(char_val_dict["inlet_area"], dtype=tf.float64), (num_geos, 1)),
#                                 tf.reshape(tf.convert_to_tensor(char_val_dict["U_char"], dtype=tf.float64), (num_geos, 1))), axis = -1)
    
#     flows = tf.stack((tf.convert_to_tensor(np.asarray(char_val_dict["daughter1_flows"]), dtype=tf.float64),
#                             tf.convert_to_tensor(np.asarray(char_val_dict["daughter2_flows"]), dtype=tf.float64),
#                             ), axis = -1)
    
#     dPs = tf.stack((tf.convert_to_tensor(np.asarray(char_val_dict["daughter1_dPs"]), dtype=tf.float64),
#                             tf.convert_to_tensor(np.asarray(char_val_dict["daughter2_dPs"]), dtype=tf.float64),
#                             ), axis = -1)
    
#     if not os.path.exists(f"data/tensors"):
#         os.mkdir(f"data/tensors")
#     if not os.path.exists(f"data/tensors/{anatomy}"):
#         os.mkdir(f"data/tensors/{anatomy}")
#     if not os.path.exists(f"data/tensors/{anatomy}/{set_type}"):
#         os.mkdir(f"data/tensors/{anatomy}/{set_type}")
        
#     save_dict({"input": input_tens, "output": output_tens, "scaling_factors": scaling_factors, "flows": flows, "dPs": dPs},
#                 f"data/tensors/{anatomy}/{set_type}/tensors_num_geos_{num_geos}")
#     return 
    