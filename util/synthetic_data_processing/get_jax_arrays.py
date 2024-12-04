import sys
sys.path.append("Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
import jax.numpy as jnp

def scale_jax(scaling_dict, field, field_name):
    # Inverse data normalization function
    mean = scaling_dict[field_name][0]
    std = scaling_dict[field_name][1]
    scaled_field = jnp.divide(jnp.subtract(field, mean), std)
    return jnp.reshape(scaled_field, (-1,1))

def get_jax_arrays(anatomy, set_type, unsteady = True):
    char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    num_geos = len(char_val_dict["inlet_area"])
    
    input_tens = jnp.concatenate((scale_jax(scaling_dict, jnp.asarray(char_val_dict["daughter1_area_ratio"], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["daughter2_area_ratio"], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["daughter1_angle"], dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["daughter2_angle"], dtype=jnp.float32), "daughter2_angle"),
                                ), axis = -1)

    output_tens = jnp.concatenate((scale_jax(scaling_dict, jnp.asarray(char_val_dict["R_lin_star1"], dtype=jnp.float32), "R_lin_star1"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["R_lin_star2"], dtype=jnp.float32), "R_lin_star2"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["R_quad_star1"], dtype=jnp.float32), "R_quad_star1"),
                            scale_jax(scaling_dict, jnp.asarray(char_val_dict["R_quad_star2"], dtype=jnp.float32), "R_quad_star2"),
                            ), axis = -1)
    
    scaling_factors = jnp.concatenate((jnp.reshape(jnp.asarray(char_val_dict["inlet_area"], dtype=jnp.float32), (num_geos, 1)),
                                jnp.reshape(jnp.asarray(char_val_dict["U_char"], dtype=jnp.float32), (num_geos, 1))), axis = -1)
    
    flows = jnp.stack((jnp.asarray(char_val_dict["daughter1_flows"], dtype=jnp.float32),
                            jnp.asarray(char_val_dict["daughter2_flows"], dtype=jnp.float32),
                            ), axis = -1)
    
    dPs = jnp.stack((jnp.asarray(char_val_dict["daughter1_dPs"], dtype=jnp.float32),
                            jnp.asarray(char_val_dict["daughter2_dPs"], dtype=jnp.float32),
                            ), axis = -1)
    
    if not os.path.exists(f"data/jax_arrays"):
        os.mkdir(f"data/jax_arrays")
    if not os.path.exists(f"data/jax_arrays/{anatomy}"):
        os.mkdir(f"data/jax_arrays/{anatomy}")
    if not os.path.exists(f"data/jax_arrays/{anatomy}/{set_type}"):
        os.mkdir(f"data/jax_arrays/{anatomy}/{set_type}")
        
    save_dict({"input": input_tens, "output": output_tens, "scaling_factors": scaling_factors, "flows": flows, "dPs": dPs},
                f"data/jax_arrays/{anatomy}/{set_type}/jax_arrays_num_geos_{num_geos}")
    return 
    