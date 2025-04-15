import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict


def check_out_of_dist(param, param_name, scaling_dict):
    if param < scaling_dict[param_name][2]:
        print(f"{param_name} smaller than training set minimum: {param}, {scaling_dict[param_name][2]}")
        param = scaling_dict[param_name][2]
    if param > scaling_dict[param_name][3]:
        print(f"{param_name} larger than training set maximum: {param}, {scaling_dict[param_name][3]}")
        param = scaling_dict[param_name][3]
    return

def get_R_values_bif(areas, tangents, length):
    anatomy = "angles_CCO"
    set_type = "random"

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    re_char = 4500
    A_char = areas[0]
    U_char = re_char * 0.04 / (1.06 * 2*np.sqrt(A_char/np.pi))
    
    L_char = np.sqrt(A_char/np.pi)
    

    daughter1_area_ratio = areas[1]/A_char; check_out_of_dist(daughter1_area_ratio, "daughter1_area_ratio", scaling_dict)
    daughter2_area_ratio = areas[2]/A_char; check_out_of_dist(daughter2_area_ratio, "daughter2_area_ratio", scaling_dict)
    daughter1_area_ratio_inv2 = (areas[1]/A_char)**-2; check_out_of_dist(daughter1_area_ratio_inv2, "daughter1_area_ratio_inv2", scaling_dict)
    daughter2_area_ratio_inv2 = (areas[2]/A_char)**-2; check_out_of_dist(daughter2_area_ratio_inv2, "daughter2_area_ratio_inv2", scaling_dict)
    daughter1_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])); check_out_of_dist(daughter1_angle, "daughter1_angle", scaling_dict)
    daughter2_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2])); check_out_of_dist(daughter2_angle, "daughter2_angle", scaling_dict)
    daughter1_length = length/L_char; check_out_of_dist(daughter1_length, "daughter1_length", scaling_dict)
    
    length_add = max([length - L_char * scaling_dict["daughter1_length"][3], 0])
    res_add = length_add * 8 * np.pi * 0.04 / (areas[1]**2)
    if res_add > 0:
        print(f"Length add: {length_add}, res_add: {res_add}")
    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_length, dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)

    model_name = "angles_CCO_ng_1902_nl_3_lw_200_ne_1000_bs_40_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

    coefs_pred = predict(input_tens, nn_model.weights)

    #R_lin_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star_inlet"); check_out_of_dist(R_lin_star_pred_inlet, "R_lin_star_inlet", scaling_dict)
    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star"); check_out_of_dist(R_lin_star_pred1, "daughter1_R_lin_star", scaling_dict)
    # R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "daughter2_R_lin_star"); check_out_of_dist(R_lin_star_pred2, "daughter2_R_lin_star", scaling_dict)
    #R_quad_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star_inlet"); check_out_of_dist(R_quad_star_pred_inlet, "R_quad_star_inlet", scaling_dict)
    #R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "R_quad_star1"); check_out_of_dist(R_quad_star_pred1, "R_quad_star1", scaling_dict)
    #R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][5], "R_quad_star2"); check_out_of_dist(R_quad_star_pred2, "R_quad_star2", scaling_dict)

    inlet_R_lin = 0 #* -1.06 * jnp.square(U_char) * R_lin_star_pred_inlet /  (A_char * U_char)
    daughter1_R_lin = (1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)) + res_add
    daughter2_R_lin = 0 #-1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)

    inlet_R_quad = 0 #* -1.06 * jnp.square(U_char) * R_quad_star_pred_inlet / jnp.square(A_char * U_char)
    daughter1_R_quad = 0 #* -1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    daughter2_R_quad = 0 #* -1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)

    R_dict = {"inlet_R_lin": 0, #float(inlet_R_lin[0][0]), 
              "daughter1_R_lin": float(daughter1_R_lin[0][0]), 
              "daughter2_R_lin": 0, #float(daughter2_R_lin[0][0]), 
              "inlet_R_quad": 0, #float(inlet_R_quad[0][0]), 
              "daughter1_R_quad": 0, #float(daughter1_R_quad[0][0]), 
              "daughter2_R_quad": 0}#float(daughter2_R_quad[0][0])}

    return R_dict

if __name__ == "__main__":
    flow_amp = "full"
    tree_name = "tree_20"
    zerod_gen = "sv"
    if zerod_gen == "CCO":
        input_file_standard = f'trees/zerod_input_CCO_standard/{tree_name}_{flow_amp}/solver_0d.json'
    else:
        input_file_standard = f'trees/zerod_input_sv_standard/{tree_name}_{flow_amp}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    num_junctions = 0; num_non_bifs = 0; out_of_dist_cnt = 0
    r_lin_list = [];r_quad_list = [];areas_list = []

    # Get the maximum vessel ID
    max_vessel_id = 0
    for i, vessel in enumerate(input_file["vessels"]):
        max_vessel_id = max(max_vessel_id, vessel["vessel_id"])
        input_file["vessels"][i]["zero_d_element_values"]["C"] = 0
        input_file["vessels"][i]["zero_d_element_values"]["L"] = 0

    max_junction_id = 0
    for junction in input_file["junctions"]:
        junction_id = int(junction["junction_name"][1:])
        max_junction_id = max(max_junction_id, junction_id)

    lengths = []; vessel_ids = []; branch_ids = []
    for vessel in input_file["vessels"]:
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        branch_ids.append(branch_id)
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        vessel_ids.append( vessel["vessel_id"])
        lengths.append(vessel["vessel_length"])
        vessel["zero_d_element_values"]["pressure_recovery_coefficient"] = 0
        if not branch_id == 0:
            vessel["zero_d_element_values"]["R_poiseuille"] = 0
            vessel["zero_d_element_values"]["stenosis_coefficient"] = 0
            
            vessel["zero_d_element_values"]["L"] = 0
            vessel["zero_d_element_values"]["C"] = 0
    length_dict = {"branch_ids": np.asarray(branch_ids), "vessel_ids": np.asarray(vessel_ids), "lengths": np.asarray(lengths)}
    inds = []
    new_junction_list = []

    for junction in input_file["junctions"]:
        original_inlet_vessel_id = copy.copy(junction["inlet_vessels"][0])
        junction_name = junction["junction_name"]
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."

        # Skip junctions that have only one outlet vessel
        if len(junction["outlet_vessels"]) == 1:
            continue

        r_lin = []
        r_quad = []
        r_lin_inlet_list = []
        r_quad_inlet_list = []

        num_outlets = len(junction["outlet_vessels"])
        for i in range(num_outlets):
            aux_area = sum([junction["areas"][j+1] for j in range(0, num_outlets) if j != i])
            tangent_array = np.asarray(junction["tangents"])
            
            aux_tangent = 0 * tangent_array[0,:]
            for j in range(num_outlets):
                if j != i:
                    aux_tangent += (tangent_array[j+1,:]*junction["areas"][j+1]/aux_area)
            outlet_branch = length_dict["branch_ids"][np.where(length_dict["vessel_ids"] == junction["outlet_vessels"][i])]
            
            length = junction["lengths"][i] + float(np.sum(length_dict["lengths"][np.where(length_dict["branch_ids"] == outlet_branch)]))

            R_dict = get_R_values_bif([junction["areas"][0], junction["areas"][i+1], aux_area], 
                                      [junction["tangents"][0], junction["tangents"][i+1], list(aux_tangent)],
                                      length)

            r_lin.append(R_dict["daughter1_R_lin"])
            r_quad.append(R_dict["daughter1_R_quad"])


        L = [0 * area for area in junction["areas"][1:]]
        inlet_area = junction["areas"][0]
            
        junction["junction_type"] = "BloodVesselJunction"
        junction["junction_values"] = {"R_poiseuille": r_lin, 
                            "stenosis_coefficient": L,
                            "pressure_recovery_coefficient": r_quad,
                            "L": L,}
        
        # Add inlet resistor vessel
        # 

    # num_steps = 100
    # max_Q = input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]
    # input_file["boundary_conditions"][0]["bc_values"]["Q"] = list(np.linspace(0, max_Q, num_steps))
    # input_file["boundary_conditions"][0]["bc_values"]["t"] = list(np.linspace(0, 1, num_steps))
    # input_file["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = num_steps
    # input_file["simulation_parameters"]["number_of_cardiac_cycles"] = 1
    # input_file["simulation_parameters"]["steady_initial"] = False
    # input_file["simulation_parameters"]["maximum_nonlinear_iterations"] = 100

    input_file["junctions"] += new_junction_list
    if not os.path.exists(f'trees/zerod_input_{zerod_gen}_R/{tree_name}_{flow_amp}'):
        os.makedirs(f'trees/zerod_input_{zerod_gen}_R/{tree_name}_{flow_amp}')
    if not os.path.exists(f'trees/zerod_output_{zerod_gen}_R/{tree_name}_{flow_amp}'):
        os.makedirs(f'trees/zerod_output_{zerod_gen}_R/{tree_name}_{flow_amp}')
    with open(f'trees/zerod_input_{zerod_gen}_R/{tree_name}_{flow_amp}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input_R/{tree_name}_{flow_amp}/solver_0d.json")