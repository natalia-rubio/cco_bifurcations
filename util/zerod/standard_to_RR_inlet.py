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

def get_R_values_bif(areas, tangents):
    anatomy = "angles_CCO"
    set_type = "random"

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    re_char = 4500
    U_char = re_char * 0.04 / (1.06 * 2*np.sqrt(areas[0]/np.pi))
    A_char = areas[0]
    

    daughter1_area_ratio = areas[1]/areas[0]
    daughter2_area_ratio = areas[2]/areas[0]
    daughter1_area_ratio_inv2 = (areas[1]/areas[0])**-2
    daughter2_area_ratio_inv2 = (areas[2]/areas[0])**-2
    daughter1_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1]))
    daughter2_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2]))
    
    if daughter1_area_ratio < scaling_dict["daughter1_area_ratio"][2]:
        print(f"Daughter1 area ratio smaller than training set minimum: {daughter1_area_ratio}, {scaling_dict['daughter1_area_ratio'][2]}")
        daughter1_area_ratio = scaling_dict["daughter1_area_ratio"][2]
    if daughter1_area_ratio > scaling_dict["daughter1_area_ratio"][3]:
        print(f"Daughter1 area ratio larger than training set maximum: {daughter1_area_ratio}, {scaling_dict['daughter1_area_ratio'][3]}")
        daughter1_area_ratio = scaling_dict["daughter1_area_ratio"][3]
    if daughter2_area_ratio < scaling_dict["daughter2_area_ratio"][2]:
        print(f"Daughter2 area ratio smaller than training set minimum: {daughter2_area_ratio}, {scaling_dict['daughter2_area_ratio'][2]}")
        daughter2_area_ratio = scaling_dict["daughter2_area_ratio"][2]
    if daughter2_area_ratio > scaling_dict["daughter2_area_ratio"][3]:
        print(f"Daughter2 area ratio larger than training set maximum: {daughter2_area_ratio}, {scaling_dict['daughter2_area_ratio'][3]}")
        daughter2_area_ratio = scaling_dict["daughter2_area_ratio"][3]
    if daughter1_area_ratio_inv2 < scaling_dict["daughter1_area_ratio_inv2"][2]:
        print(f"Daughter1 area ratio inv2 smaller than training set minimum: {daughter1_area_ratio_inv2}, {scaling_dict['daughter1_area_ratio_inv2'][2]}")
        daughter1_area_ratio_inv2 = scaling_dict["daughter1_area_ratio_inv2"][2]
    if daughter1_area_ratio_inv2 > scaling_dict["daughter1_area_ratio_inv2"][3]:
        print(f"Daughter1 area ratio inv2 larger than training set maximum: {daughter1_area_ratio_inv2}, {scaling_dict['daughter1_area_ratio_inv2'][3]}")
        daughter1_area_ratio_inv2 = scaling_dict["daughter1_area_ratio_inv2"][3]
    if daughter2_area_ratio_inv2 < scaling_dict["daughter2_area_ratio_inv2"][2]:
        print(f"Daughter2 area ratio inv2 smaller than training set minimum: {daughter2_area_ratio_inv2}, {scaling_dict['daughter2_area_ratio_inv2'][2]}")
        daughter2_area_ratio_inv2 = scaling_dict["daughter2_area_ratio_inv2"][2]
    if daughter2_area_ratio_inv2 > scaling_dict["daughter2_area_ratio_inv2"][3]:
        print(f"Daughter2 area ratio inv2 larger than training set maximum: {daughter2_area_ratio_inv2}, {scaling_dict['daughter2_area_ratio_inv2'][3]}")
        daughter2_area_ratio_inv2 = scaling_dict["daughter2_area_ratio_inv2"][3]
    if daughter1_angle < scaling_dict["daughter1_angle"][2]:
        print(f"Daughter1 angle smaller than training set minimum: {daughter1_angle}, {scaling_dict['daughter1_angle'][2]}")
        daughter1_angle = scaling_dict["daughter1_angle"][2]
    if daughter1_angle > scaling_dict["daughter1_angle"][3]:
        print(f"Daughter1 angle larger than training set maximum: {daughter1_angle}, {scaling_dict['daughter1_angle'][3]}")
        daughter1_angle = scaling_dict["daughter1_angle"][3]
    if daughter2_angle < scaling_dict["daughter2_angle"][2]:
        print(f"Daughter2 angle smaller than training set minimum: {daughter2_angle}", scaling_dict["daughter2_angle"][2])
        daughter2_angle = scaling_dict["daughter2_angle"][2]
    if daughter2_angle > scaling_dict["daughter2_angle"][3]:
        print(f"Daughter2 angle larger than training set maximum: {daughter2_angle}", scaling_dict["daughter2_angle"][3])
        daughter2_angle = scaling_dict["daughter2_angle"][3]

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)

    if areas[1] < 0.01:
        pdb.set_trace()
    model_name = "angles_CCO_ng_172_nl_2_lw_50_ne_2000_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star_inlet")

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_lin_star2")
    R_quad_star_pred_inlet = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star_inlet")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][5], "R_quad_star2")

    # if R_quad_star_pred_inlet > scaling_dict["R_lin_star_inlet"][3]:
    #     print(f"R_lin_star_inlet larger than training set maximum: {R_lin_star_pred_inlet}, {scaling_dict['R_lin_star_inlet'][3]}")
    #     R_lin_star_pred_inlet = scaling_dict["R_lin_star_inlet"][3]
    # if R_lin_star_pred1 < scaling_dict["R_lin_star1"][2]:
    #     print(f"R_lin_star1 smaller than training set minimum: {R_lin_star_pred1}, {scaling_dict['R_lin_star1'][2]}")
    #     R_lin_star_pred1 = scaling_dict["R_lin_star1"][2]
    # if R_lin_star_pred1 > scaling_dict["R_lin_star1"][3]:
    #     print(f"R_lin_star1 larger than training set maximum: {R_lin_star_pred1}, {scaling_dict['R_lin_star1'][3]}")
    #     R_lin_star_pred1 = scaling_dict["R_lin_star1"][3]
    # if R_lin_star_pred2 < scaling_dict["R_lin_star2"][2]:
    #     print(f"R_lin_star2 smaller than training set minimum: {R_lin_star_pred2}, {scaling_dict['R_lin_star2'][2]}")
    #     R_lin_star_pred2 = scaling_dict["R_lin_star2"][2]
    # if R_lin_star_pred2 > scaling_dict["R_lin_star2"][3]:
    #     print(f"R_lin_star2 larger than training set maximum: {R_lin_star_pred2}, {scaling_dict['R_lin_star2'][3]}")
    #     R_lin_star_pred2 = scaling_dict["R_lin_star2"][3]
    # if R_quad_star_pred1 < scaling_dict["R_quad_star1"][2]:
    #     print(f"R_quad_star1 smaller than training set minimum: {R_quad_star_pred1}, {scaling_dict['R_quad_star1'][2]}")
    #     R_quad_star_pred1 = scaling_dict["R_quad_star1"][2]
    # if R_quad_star_pred1 < 0:
    #     print(f"R_quad_star1 smaller than 0: {R_quad_star_pred1}")
    #     R_quad_star_pred1 *= 0
    # if R_quad_star_pred1 > scaling_dict["R_quad_star1"][3]:
    #     print(f"R_quad_star1 larger than training set maximum: {R_quad_star_pred1}, {scaling_dict['R_quad_star1'][3]}")
    #     R_quad_star_pred1 = scaling_dict["R_quad_star1"][3]
    # if R_quad_star_pred2 < scaling_dict["R_quad_star2"][2]:
    #     print(f"R_quad_star2 smaller than training set minimum: {R_quad_star_pred2}, {scaling_dict['R_quad_star2'][2]}")
    #     R_quad_star_pred2 = scaling_dict["R_quad_star2"][2]
    # if R_quad_star_pred2 < 0:
    #     print(f"R_quad_star2 smaller than 0: {R_quad_star_pred2}")
    #     R_quad_star_pred2 *= 0
    # if R_quad_star_pred2 > scaling_dict["R_quad_star2"][3]:
    #     print(f"R_quad_star2 larger than training set maximum: {R_quad_star_pred2}, {scaling_dict['R_quad_star2'][3]}")
    #     R_quad_star_pred2 = scaling_dict["R_quad_star2"][3]

    R_lin_inlet = 0 *-1.06 * jnp.square(U_char) * R_lin_star_pred_inlet /  (A_char * U_char)
    R_lin1 = -1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = -1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)

    R_quad_inlet = 0 * -1.06 * jnp.square(U_char) * R_quad_star_pred_inlet / jnp.square(A_char * U_char)
    R_quad1 = 0 * -1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 0 * -1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)

    R_dict = {"R_lin_inlet": 0*float(R_lin_inlet[0][0]), 
              "R_lin1": float(R_lin1[0][0]), 
              "R_lin2": float(R_lin2[0][0]), 
              "R_quad_inlet": 1 * max([float(R_quad_inlet[0][0]), 0]), 
              "R_quad1": float(R_quad1[0][0]), 
              "R_quad2": float(R_quad2[0][0])}

    return R_dict

if __name__ == "__main__":
    flow_amp = "full"
    tree_name = "tree_80"
    zerod_gen = "sv"
    if zerod_gen == "CCO":
        input_file_standard = f'trees/zerod_input_CCO_standard/{tree_name}_{flow_amp}/solver_0d.json'
    else:
        input_file_standard = f'trees/zerod_input_sv_standard/{tree_name}_{flow_amp}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)

    num_junctions = 0
    num_non_bifs = 0
    out_of_dist_cnt = 0

    r_lin_list = []
    r_quad_list = []
    areas_list = []

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

    inds = []
    new_junction_list = []
    for junction in input_file["junctions"]:
        original_inlet_vessel_id = copy.copy(junction["inlet_vessels"][0])
        junction_name = junction["junction_name"]
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."

        # Skip junctions that have only one outlet vessel
        if len(junction["outlet_vessels"]) == 1:
            continue

        r_lin = [0,]
        r_quad = [0,]
        r_lin_inlet_list = []
        r_quad_inlet_list = []

        num_aux_outlets = len(junction["outlet_vessels"])-1
        for i in range(num_aux_outlets):
            R_dict = get_R_values_bif([junction["areas"][0], junction["areas"][1], junction["areas"][i+1]], 
                                      [junction["tangents"][0], junction["tangents"][1], junction["tangents"][i+1]])
            r_lin[0] += R_dict["R_lin1"] #/(num_aux_outlets)
            r_quad[0] += R_dict["R_quad1"] #/(num_aux_outlets)
            r_lin.append(R_dict["R_lin2"])
            r_quad.append(R_dict["R_quad2"])
            r_lin_inlet_list.append(R_dict["R_lin_inlet"])
            r_quad_inlet_list.append(R_dict["R_quad_inlet"]) 


        L = [0 * area for area in junction["areas"][1:]]
        inlet_area = junction["areas"][0]
            
        junction["junction_type"] = "BloodVesselJunction"
        junction["junction_values"] = {"R_poiseuille": r_lin, 
                            "stenosis_coefficient": L,
                            "pressure_recovery_coefficient": r_quad,
                            "L": L,}
        
        # Add inlet resistor vessel
        if num_aux_outlets > 1:
            print(f"Junction {junction_name} has {num_aux_outlets} outlets.")
            print(f"Inlet vessel has R_lin = {r_lin_inlet_list} and R_quad = {r_quad_inlet_list}")
        new_vessel_id = max_vessel_id + 1
        inlet_vessel_dict = {'vessel_id': new_vessel_id, 
                       'vessel_length': 0, 
                       'vessel_name':  f'branch{new_vessel_id}_seg0', 
                       'zero_d_element_type': 'BloodVessel', 
                       'zero_d_element_values': {'C': 0, 'L': 0, 
                                                 'R_poiseuille': r_lin_inlet_list[0], 
                                                 'stenosis_coefficient': 0,
                                                 'pressure_recovery_coefficient': r_quad_inlet_list[0]},}
        input_file["vessels"].append(inlet_vessel_dict)
        max_vessel_id += 1
        junction["inlet_vessels"] = [new_vessel_id]

        # Add a junction to connect the inlet vessel to the bifurcation
        new_junction_id = max_junction_id + 1
        inlet_vessel_2_bif_connector = {'inlet_vessels': [original_inlet_vessel_id], 
                               'junction_name': f"J{new_junction_id}", 
                               'junction_type': 'NORMAL_JUNCTION', 
                               'outlet_vessels': [new_vessel_id]}
        max_junction_id += 1
        new_junction_list.append(inlet_vessel_2_bif_connector)
        #pdb.set_trace()

    num_steps = 100
    max_Q = input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]
    input_file["boundary_conditions"][0]["bc_values"]["Q"] = list(np.linspace(0, max_Q, num_steps))
    input_file["boundary_conditions"][0]["bc_values"]["t"] = list(np.linspace(0, 1, num_steps))
    input_file["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = num_steps
    input_file["simulation_parameters"]["number_of_cardiac_cycles"] = 1
    input_file["simulation_parameters"]["steady_initial"] = False
    input_file["simulation_parameters"]["maximum_nonlinear_iterations"] = 100

    input_file["junctions"] += new_junction_list
    if not os.path.exists(f'trees/zerod_input_{zerod_gen}_RR_inlet/{tree_name}_{flow_amp}'):
        os.makedirs(f'trees/zerod_input_{zerod_gen}_RR_inlet/{tree_name}_{flow_amp}')
    with open(f'trees/zerod_input_{zerod_gen}_RR_inlet/{tree_name}_{flow_amp}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input_RR_inlet/{tree_name}_{flow_amp}/solver_0d.json")