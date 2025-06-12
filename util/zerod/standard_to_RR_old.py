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
    return param

def get_R_values_bif(areas, tangents, length):
    verbose = False
    # anatomy = "ideal" #
    # anatomy = "tree_20" #
    #anatomy = "ideal_fix_areas"
    #anatomy = "angles_CCO"
    anatomy = "tree_20_res2"
    set_type = "dict" #
    #set_type = "random"

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    re_char = 4500
    A_char = areas[0]
    U_char = re_char * 0.04 / (1.06 * 2 *np.sqrt(A_char/np.pi))

    L_char = np.sqrt(A_char/np.pi)

    daughter1_area_ratio = areas[1]/A_char; daughter1_area_ratio = check_out_of_dist(daughter1_area_ratio, "daughter1_area_ratio", scaling_dict)
    daughter2_area_ratio = areas[2]/A_char; daughter2_area_ratio = check_out_of_dist(daughter2_area_ratio, "daughter2_area_ratio", scaling_dict)
    daughter1_area_ratio_inv2 = (areas[1]/A_char)**-2;  daughter1_area_ratio_inv2 = check_out_of_dist(daughter1_area_ratio_inv2, "daughter1_area_ratio_inv2", scaling_dict)
    daughter2_area_ratio_inv2 = (areas[2]/A_char)**-2;  daughter2_area_ratio_inv2 = check_out_of_dist(daughter2_area_ratio_inv2, "daughter2_area_ratio_inv2", scaling_dict)
    daughter1_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])); daughter1_angle = check_out_of_dist(daughter1_angle, "daughter1_angle", scaling_dict)
    daughter2_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2])); daughter2_angle = check_out_of_dist(daughter2_angle, "daughter2_angle", scaling_dict)
    #pdb.set_trace()
    daughter1_length_star = length/L_char; daughter1_length_star = check_out_of_dist(daughter1_length_star, "daughter1_length_star", scaling_dict)
    
    length_add = max([length - L_char * scaling_dict["daughter1_length_star"][3], 0])
    res_add = length_add * 8 * np.pi * 0.04 / (areas[1]**2)
    if res_add > 0 and verbose:
        print(f"Length add: {length_add}, res_add: {res_add}")
    print(f"Length star: {daughter1_length_star}")
    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_length_star, dtype=jnp.float32), "daughter1_length_star"),
                                ]).reshape(1,-1)

    #model_name = "angles_CCO_ng_1434_nl_3_lw_500_ne_1000_bs_70_dr_0.9_model"
    # model_name = "ideal_ng_2094_nl_3_lw_500_ne_1000_bs_70_dr_0.9_model"
    # model_name = "ideal_fix_areas_ng_1768_nl_2_lw_40_ne_1000_bs_70_dr_0.9_model"
    model_name = "tree_20_ng_630_nl_2_lw_400_ne_3000_bs_20_dr_0.9_model"
    #model_name = "rand_res_ng_2816_nl_2_lw_40_ne_1000_bs_70_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

    coefs_pred = predict(input_tens, nn_model.weights)

    # R_lin_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star_inlet"); check_out_of_dist(R_lin_star_pred_inlet, "R_lin_star_inlet", scaling_dict)
    R_lin_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred[0][0], "daughter1_R_lin_star")[0][0]); R_lin_star_pred1 = check_out_of_dist(R_lin_star_pred1, "daughter1_R_lin_star", scaling_dict)
    # R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "daughter2_R_lin_star"); check_out_of_dist(R_lin_star_pred2, "daughter2_R_lin_star", scaling_dict)
    # R_quad_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star_inlet"); check_out_of_dist(R_quad_star_pred_inlet, "R_quad_star_inlet", scaling_dict)
    R_quad_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_quad_star")[0][0]); R_quad_star_pred1 = check_out_of_dist(R_quad_star_pred1, "daughter1_R_quad_star", scaling_dict)
    # R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][5], "R_quad_star2"); check_out_of_dist(R_quad_star_pred2, "R_quad_star2", scaling_dict)
    
    inlet_R_lin = 0 #* -1.06 * jnp.square(U_char) * R_lin_star_pred_inlet /  (A_char * U_char)
    daughter1_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char))) + res_add
    # daughter1_R_lin_poiseuille = length * 8 * np.pi * 0.04 / (areas[1]**2)
    daughter2_R_lin = 0 #-1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)

    inlet_R_quad = 0 #* -1.06 * jnp.square(U_char) * R_quad_star_pred_inlet / jnp.square(A_char * U_char)
    daughter1_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char))
    # daughter1_R_quad = check_out_of_dist(daughter1_R_quad, "daughter1_R_quad", scaling_dict)
    daughter2_R_quad = 0 #* -1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)


    R_dict = {"inlet_R_lin": 0,
              "daughter1_R_lin": daughter1_R_lin,
              "daughter2_R_lin": 0,
              "inlet_R_quad": 0,
              "daughter1_R_quad": daughter1_R_quad,
              "daughter2_R_quad": 0}

    # pdb.set_trace()
    return R_dict

if __name__ == "__main__":
    
    tree_name = sys.argv[1]
    
    input_file_standard = f'trees/zerod_input/standard/{tree_name}/solver_0d.json'
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

    lengths = []; areas = []; vessel_ids = []; branch_ids = []
    for vessel in input_file["vessels"]:
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        branch_ids.append(branch_id)
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        vessel_ids.append( vessel["vessel_id"])
        lengths.append(vessel["vessel_length"])
        areas.append(np.sqrt(0.04*8*np.pi*vessel["vessel_length"] / vessel["zero_d_element_values"]["R_poiseuille"]))
        vessel["zero_d_element_values"]["pressure_recovery_coefficient"] = 0
        if not branch_id == 0:
            vessel["zero_d_element_values"]["R_poiseuille"] = 0
            vessel["zero_d_element_values"]["stenosis_coefficient"] = 0
            
            vessel["zero_d_element_values"]["L"] = 0
            vessel["zero_d_element_values"]["C"] = 0
    length_dict = {"branch_ids": np.asarray(branch_ids), "vessel_ids": np.asarray(vessel_ids), "lengths": np.asarray(lengths), "areas": np.asarray(areas)}
    # pdb.set_trace()
    inds = []
    new_junction_list = []

    for junction in input_file["junctions"]:
        original_inlet_vessel_id = copy.copy(junction["inlet_vessels"][0])
        junction_name = junction["junction_name"]
        print(f"Processing junction {junction_name}")
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."

        # Skip junctions that have only one outlet vessel
        if len(junction["outlet_vessels"]) == 1:
            continue

        r_lin = []
        r_quad = []
        r_lin_inlet_list = []
        r_quad_inlet_list = []

        num_outlets = len(junction["outlet_vessels"])
        assert num_outlets == 2, "Junction with more than two outlet vessels."
        for i in range(num_outlets):
            if i == 0:
                aux_i  = 1
            if i == 1: 
                aux_i = 0

            outlet_branch = length_dict["branch_ids"][np.where(length_dict["vessel_ids"] == junction["outlet_vessels"][i])]
            aux_outlet_branch = length_dict["branch_ids"][np.where(length_dict["vessel_ids"] == junction["outlet_vessels"][aux_i])]

            
            length = junction["lengths"][i] + float(np.sum(length_dict["lengths"][np.where(length_dict["branch_ids"] == outlet_branch)]))
            primary_area = min(length_dict["areas"][np.where(length_dict["branch_ids"] == outlet_branch)])
            #print(f"Primary area: {primary_area} vs {junction['areas'][i+1]}")
            aux_area = min(length_dict["areas"][np.where(length_dict["branch_ids"] == aux_outlet_branch)])
            #print(f"Aux area: {aux_area} vs {junction['areas'][aux_i+1]}")
            aux_area_tan = sum([junction["areas"][j+1] for j in range(0, num_outlets) if j != i])
            tangent_array = np.asarray(junction["tangents"])
            
            aux_tangent = 0 * tangent_array[0,:]
            for j in range(num_outlets):
                if j != i:
                    aux_tangent += (tangent_array[j+1,:]*junction["areas"][j+1]/aux_area_tan)

            R_dict = get_R_values_bif([junction["areas"][0], primary_area, aux_area], 
                                      [junction["tangents"][0], junction["tangents"][i+1], list(aux_tangent)],
                                      length)

            r_lin.append(R_dict["daughter1_R_lin"])
            r_quad.append(R_dict["daughter1_R_quad"])


        L = [0 * area for area in junction["areas"][1:]]
        inlet_area = junction["areas"][0]

        #print(f"Junction {junction_name} has daughter 1 R_lin values: {r_lin}")
        junction["junction_type"] = "BloodVesselJunction"
        junction["junction_values"] = {"R_poiseuille": r_lin, 
                            "stenosis_coefficient": L,
                            "pressure_recovery_coefficient": r_quad,
                            "L": L,}
        
    input_file["junctions"] += new_junction_list
    if not os.path.exists(f'trees/zerod_input/RR/{tree_name}'):
        os.makedirs(f'trees/zerod_input/RR/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/RR/{tree_name}'):
        os.makedirs(f'trees/zerod_output/RR/{tree_name}')
    with open(f'trees/zerod_input/RR/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input/RR/{tree_name}/solver_0d.json")