from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os

import pandas as pd

#from util.zerod.svzerod_to_casadi_exact import R_lin

#from util.zerod.svzerod_to_casadi_exact import R_quad
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, add_solution_values, add_3D_resistance

def check_out_of_dist(param, param_name, scaling_dict, verbose = False):
    if param < scaling_dict[param_name][2]:
        if verbose:
            print(f"{param_name} smaller than training set minimum: {param}, {scaling_dict[param_name][2]}")
        param = scaling_dict[param_name][2]
    if param > scaling_dict[param_name][3]:
        if verbose:
            print(f"{param_name} larger than training set maximum: {param}, {scaling_dict[param_name][3]}")
        param = scaling_dict[param_name][3]
    return param

def get_R_values_bif(inlet_area,
                     outlet1_area,
                     outlet2_area,
                     daughter1_angle,
                     daughter2_angle,
                     length1,
                     length2,
                     daughter1_flow_split,
                     daughter2_flow_split):
    verbose = False
    anatomy = "tree_20"
    set_type = "random"#
    #set_type = "combined" #"dict_res_fs_ext" 
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    model_name = "tree_20_ng_1220_nl_1_lw_70_ne_10000_bs_50_dr_0.95_model"
    #model_name = "tree_20_ng_950_nl_2_lw_200_ne_5000_bs_100_dr_0.95_model" #"tree_20_ng_400_nl_2_lw_100_ne_1000_bs_20_dr_0.95_model"
    #model_name = "tree_20_ng_280_nl_3_lw_500_ne_2500_bs_20_dr_0.95_model" # "tree_20_ng_400_nl_2_lw_100_ne_1000_bs_20_dr_0.95_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")


    re_char = 4500
    A_char = inlet_area
    U_char = re_char * 0.04 / (1.06 * 2 *np.sqrt(A_char/np.pi))
    L_char = np.sqrt(A_char/np.pi)

    daughter1_area_ratio = outlet1_area/A_char; daughter1_area_ratio = check_out_of_dist(daughter1_area_ratio, "daughter1_area_ratio", scaling_dict)
    daughter2_area_ratio = outlet2_area/A_char; daughter2_area_ratio = check_out_of_dist(daughter2_area_ratio, "daughter2_area_ratio", scaling_dict)
    total_daughter_area_ratio = (outlet1_area + outlet2_area)/A_char; total_daughter_area_ratio = check_out_of_dist(total_daughter_area_ratio, "total_daughter_area_ratio", scaling_dict)
    daughter1_area_ratio_inv2 = (outlet1_area/A_char)**-2;  daughter1_area_ratio_inv2 = check_out_of_dist(daughter1_area_ratio_inv2, "daughter1_area_ratio_inv2", scaling_dict)
    daughter2_area_ratio_inv2 = (outlet2_area/A_char)**-2;  daughter2_area_ratio_inv2 = check_out_of_dist(daughter2_area_ratio_inv2, "daughter2_area_ratio_inv2", scaling_dict)
    total_area_ratio_inv2 = ((outlet1_area + outlet2_area)/A_char)**2; total_area_ratio_inv2 = check_out_of_dist(total_area_ratio_inv2, "total_area_ratio_inv2", scaling_dict)
    daughter1_angle = check_out_of_dist(daughter1_angle, "daughter1_angle", scaling_dict)
    daughter2_angle = check_out_of_dist(daughter2_angle, "daughter2_angle", scaling_dict)

    daughter1_length_star = length1/L_char; daughter1_length_star = check_out_of_dist(daughter1_length_star, "daughter1_length_star", scaling_dict)
    daughter1_length_star_sq = jnp.square(daughter1_length_star); daughter1_length_star_sq = check_out_of_dist(daughter1_length_star_sq, "daughter1_length_star_sq", scaling_dict)
    length_add1 = max([length1 - L_char * scaling_dict["daughter1_length_star"][3], 0])
    length_sub1 = max([L_char * scaling_dict["daughter1_length_star"][2] - length1, 0])
    res_add1 = length_add1 * 8 * np.pi * 0.04 / (outlet1_area**2)
    res_sub1 = length_sub1 * 8 * np.pi * 0.04 / (outlet1_area**2)

    daughter2_length_star = length2/L_char; daughter2_length_star = check_out_of_dist(daughter2_length_star, "daughter2_length_star", scaling_dict)
    daughter2_length_star_sq = jnp.square(daughter2_length_star); daughter2_length_star_sq = check_out_of_dist(daughter2_length_star_sq, "daughter2_length_star_sq", scaling_dict)
    length_add2 = max([length2 - L_char * scaling_dict["daughter2_length_star"][3], 0])
    length_sub2 = max([L_char * scaling_dict["daughter2_length_star"][2] - length2, 0])
    res_add2 = length_add2 * 8 * np.pi * 0.04 / (outlet2_area**2)
    res_sub2 = length_sub2 * 8 * np.pi * 0.04 / (outlet2_area**2)


    input_tens1 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                    #scale_jax(scaling_dict, jnp.asarray(total_daughter_area_ratio, dtype=jnp.float32), "total_daughter_area_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                    #scale_jax(scaling_dict, jnp.asarray(total_area_ratio_inv2, dtype=jnp.float32), "total_area_ratio_inv2"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter1_angle"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter2_angle"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_length_star, dtype=jnp.float32), "daughter1_length_star"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_length_star_sq, dtype=jnp.float32), "daughter1_length_star_sq"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_flow_split, dtype=jnp.float32), "daughter1_flow_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_flow_split**2, dtype=jnp.float32), "daughter1_flow_ratio_sq"),
                        ]).reshape(1,-1)
    input_tens2 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                    #scale_jax(scaling_dict, jnp.asarray(total_daughter_area_ratio, dtype=jnp.float32), "total_daughter_area_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                    #scale_jax(scaling_dict, jnp.asarray(total_area_ratio_inv2, dtype=jnp.float32), "total_area_ratio_inv2"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter1_angle"),
                    scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter2_angle"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_length_star, dtype=jnp.float32), "daughter2_length_star"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_length_star_sq, dtype=jnp.float32), "daughter2_length_star_sq"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_flow_split, dtype=jnp.float32), "daughter2_flow_ratio"),
                    scale_jax(scaling_dict, jnp.asarray(daughter2_flow_split**2, dtype=jnp.float32), "daughter2_flow_ratio_sq"),
                        ]).reshape(1,-1)

    coefs_pred1 = predict(input_tens1, nn_model.weights)
    coefs_pred2 = predict(input_tens2, nn_model.weights)

    R_lin_star_pred1_log = float(inv_scale_jax(scaling_dict, coefs_pred1[0][0], "daughter1_R_lin_star_log")[0][0]); 
    R_lin_star_pred1_log = check_out_of_dist(R_lin_star_pred1_log, "daughter1_R_lin_star_log", scaling_dict)
    #R_lin_star_pred1 = np.exp(R_lin_star_pred1_log )
    R_lin_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][0], "daughter1_R_lin_star")[0][0])

    # R_quad_star_pred1_log = float(inv_scale_jax(scaling_dict, coefs_pred1[0][1], "daughter1_R_quad_star_log")[0][0]); 
    # R_quad_star_pred1_log = check_out_of_dist(R_quad_star_pred1_log, "daughter1_R_quad_star_log", scaling_dict)
    # R_quad_star_pred1 = np.sign(R_quad_star_pred1_log) * np.exp(np.abs(R_quad_star_pred1_log)-4)
    C = np.exp(-4)
    R_quad_star_pred1_log = float(inv_scale_jax(scaling_dict, coefs_pred1[0][1], "daughter1_R_quad_star_logC")[0][0]); 
    R_quad_star_pred1_log = check_out_of_dist(R_quad_star_pred1_log, "daughter1_R_quad_star_logC", scaling_dict)
    R_quad_star_pred1 = np.sign(R_quad_star_pred1_log) * C * (np.exp(np.abs(R_quad_star_pred1_log))-1)

    if coefs_pred1.size > 2:
        L_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][2], "daughter1_L_star")[0][0]); 
        L_star_pred1 = check_out_of_dist(L_star_pred1, "daughter1_L_star", scaling_dict)
    else:
        L_star_pred1 = 0.0

    R_lin_star_pred2_log = float(inv_scale_jax(scaling_dict, coefs_pred2[0][0], "daughter2_R_lin_star_log")[0][0]); 
    R_lin_star_pred2_log = check_out_of_dist(R_lin_star_pred2_log, "daughter2_R_lin_star_log", scaling_dict)
    #R_lin_star_pred2 = np.exp(R_lin_star_pred2_log)
    R_lin_star_pred2 = float(inv_scale_jax(scaling_dict, coefs_pred2[0][0], "daughter2_R_lin_star")[0][0])

    # R_quad_star_pred2_log = float(inv_scale_jax(scaling_dict, coefs_pred2[0][1], "daughter2_R_quad_star_log")[0][0]); 
    # R_quad_star_pred2_log = check_out_of_dist(R_quad_star_pred2_log, "daughter2_R_quad_star_log", scaling_dict)
    # R_quad_star_pred2 = np.sign(R_quad_star_pred2_log) * np.exp(np.abs(R_quad_star_pred2_log)-4)
    R_quad_star_pred2_log = float(inv_scale_jax(scaling_dict, coefs_pred2[0][1], "daughter1_R_quad_star_logC")[0][0]); 
    R_quad_star_pred2_log = check_out_of_dist(R_quad_star_pred2_log, "daughter1_R_quad_star_logC", scaling_dict)
    R_quad_star_pred2 = np.sign(R_quad_star_pred2_log) * C * (np.exp(np.abs(R_quad_star_pred2_log))-1)
    #.set_trace()
    if coefs_pred2.size > 2:
        L_star_pred2 = float(inv_scale_jax(scaling_dict, coefs_pred2[0][2], "daughter2_L_star")[0][0]); 
        L_star_pred2 = check_out_of_dist(L_star_pred2, "daughter2_L_star", scaling_dict)
    else:   
        L_star_pred2 = 0.0

    #pdb.set_trace()

    daughter1_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)))
    daughter1_R_lin_final = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char))) + (res_add1 - res_sub1) * daughter1_flow_split
    #pdb.set_trace()
    daughter1_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char))
    daughter1_L = float(L_star_pred1) *1.06*L_char/A_char

    daughter2_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)))
    daughter2_R_lin_final = float((1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char))) + (res_add2 - res_sub2) * daughter2_flow_split
    daughter2_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char))
    daughter2_L = float(L_star_pred2) *1.06*L_char/A_char


    R_dict = {"inlet_R_lin": 0,
              "daughter1_R_lin": daughter1_R_lin_final,
              "daughter2_R_lin": daughter2_R_lin_final,
              "inlet_R_quad": 0,
              "daughter1_R_quad": daughter1_R_quad,
              "daughter2_R_quad": daughter2_R_quad,
              "daughter1_L": daughter1_L,
              "daughter2_L": daughter2_L,}
    
    return R_dict

#if __name__ == "__main__":
def transform_standard_to_RR(tree_name):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    junction_dict_master = get_input_file_junction_dict_master(tree_name)

    time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["25", "50", "100", "150"]
    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    add_downstream_resistance_values(junction_dict_master)

    #print("converting to RR")
    input_file_standard = f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    for i in range(len(input_file["junctions"])):
        junction_name = input_file["junctions"][i]["junction_name"]
        if junction_name not in junction_dict_master:
            #print(f"Junction {junction_name} not found in junction dictionary master")
            continue
        junction_dict = junction_dict_master[junction_name]
        daughter1_flow_ratio = junction_dict["0D_geo_flow_split"]/(1 + junction_dict["0D_geo_flow_split"])
        daughter2_flow_ratio = 1/(1 + junction_dict["0D_geo_flow_split"])
        R_dict = get_R_values_bif(junction_dict["0D_inlet_area"],
                                    junction_dict["0D_outlet1_area"],
                                    junction_dict["0D_outlet2_area"],
                                    get_angle_diff(np.asarray(junction_dict["0D_inlet_tangent"]), np.asarray(junction_dict["0D_daughter1_tangent"])),
                                    get_angle_diff(np.asarray(junction_dict["0D_inlet_tangent"]), np.asarray(junction_dict["0D_daughter2_tangent"])),
                                    junction_dict["0D_length1"],
                                    junction_dict["0D_length2"],
                                    daughter1_flow_ratio,
                                    daughter2_flow_ratio)

        input_file["junctions"][i]["junction_type"] = "BloodVesselJunction"
        input_file["junctions"][i]["junction_values"] = {"R_poiseuille": [R_dict["daughter1_R_lin"]/daughter1_flow_ratio, 
                                                                          R_dict["daughter2_R_lin"]/daughter2_flow_ratio], 
                                                                    "stenosis_coefficient": [0, 0],
                                                                    "pressure_recovery_coefficient": [R_dict["daughter1_R_quad"]/(daughter1_flow_ratio**2), 
                                                                                                      R_dict["daughter2_R_quad"]/(daughter2_flow_ratio**2)],
                                                                    "L": [R_dict["daughter1_L"]/daughter1_flow_ratio, 
                                                                          R_dict["daughter2_L"]/daughter2_flow_ratio],
                                                                    "flow_split": [daughter1_flow_ratio, daughter2_flow_ratio],}
        # pdb.set_trace()
        # if tree_name_base == "tree_20":
        #     for time_step in [time_step1, time_step2]:
        #         for flow_mag in flow_mag_list:
        #             add_solution_values(junction_dict_master, f"{tree_name_base}_flow_{flow_mag}", flow_mag, time_step = time_step)
        #     add_3D_resistance(junction_dict_master, tree_name, flow_mag_list, time_step = time_step1)
        #     input_file["junctions"][i]["junction_values"] = {"R_poiseuille": [junction_dict["3D_daughter1_R_lin"]/daughter1_flow_ratio, junction_dict["3D_daughter2_R_lin"]/daughter2_flow_ratio], 
        #                                                 "stenosis_coefficient": [0, 0],
        #                                                 "pressure_recovery_coefficient": [junction_dict["3D_daughter1_R_quad"]/(daughter1_flow_ratio**2), junction_dict["3D_daughter2_R_quad"]/(daughter2_flow_ratio**2)],
        #                                                 "L": [0,0],
        #                                                 "flow_split": [daughter1_flow_ratio, daughter2_flow_ratio],}

    if not os.path.exists(f'trees/zerod_input/RR/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_input/RR/{tree_name_base}/{tree_name}')
    if not os.path.exists(f'trees/zerod_output/RR/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/RR/{tree_name_base}/{tree_name}')
    with open(f'trees/zerod_input/RR/{tree_name_base}/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    #print(f"RRI 0D input file saved to trees/zerod_input/RR/{tree_name_base}/{tree_name}/solver_0d.json")