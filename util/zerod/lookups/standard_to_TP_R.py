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
    
    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(areas[1]/areas[0], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(areas[2]/areas[0], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])), dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2])), dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)

    if areas[1] < 0.01:
        pdb.set_trace()
    model_name = "angles_CCO_ng_222_nl_1_lw_50_ne_2000_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")

    R_lin1 = -1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = -1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)

    return [float(R_lin1[0][0]), float(R_lin2[0][0])]

if __name__ == "__main__":
    tree_name = "tree_80"
    with open(f'trees/zerod_input_standard/{tree_name}/solver_0d.json') as json_file:
        input_file = json.load(json_file)

    results_dir = f"data/characteristic_value_dictionaries/{tree_name}_char_val_dict"
    char_val_dict = load_dict(results_dir)

    num_junctions = 0
    num_non_bifs = 0
    out_of_dist_cnt = 0

    r_lin_list = []
    r_quad_list = []
    areas_list = []

    # Get the maximum vessel ID
    max_vessel_id = 0
    for vessel in input_file["vessels"]:
        max_vessel_id = max(max_vessel_id, vessel["vessel_id"])

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

        if len(junction["outlet_vessels"]) == 1:
            continue
        r_lin = [0,]
        for i in range(len(junction["outlet_vessels"])-1):
            R_vals = get_R_values_bif([junction["areas"][0], junction["areas"][1], junction["areas"][i+1]], 
                                      [junction["tangents"][0], junction["tangents"][1], junction["tangents"][i+1]])
            r_lin[0] += R_vals[0]/(len(junction["outlet_vessels"])-1)
            r_lin.append(R_vals[1])

        L = [0 * area for area in junction["areas"][1:]]
        inlet_area = junction["areas"][0]
        r_quad = [1.06 * 0.5 * outlet_area**(-2) for outlet_area in junction["areas"][1:]]
            
        junction["junction_type"] = "BloodVesselJunction"
        junction["junction_values"] = {"R_poiseuille": r_lin, 
                            "stenosis_coefficient": r_quad,
                            "L": L,}
        
        # Add inlet resistor vessel
        new_vessel_id = max_vessel_id + 1
        inlet_vessel_dict = {'vessel_id': new_vessel_id, 
                       'vessel_length': 0, 
                       'vessel_name':  f'branch{new_vessel_id}_seg0', 
                       'zero_d_element_type': 'BloodVessel', 
                       'zero_d_element_values': {'C': 0, 'L': 0, 'R_poiseuille': 0, 'stenosis_coefficient': -1.06 * 0.5 * inlet_area**(-2)}}
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

    input_file["junctions"] += new_junction_list
    input_file["boundary_conditions"][0]["bc_values"]["Q"] = list(np.linspace(0, 340, 40)) + 200 * [340,]
    input_file["boundary_conditions"][0]["bc_values"]["t"] = list(np.linspace(0, 100, 240))
    input_file["simulation_parameters"]["number_of_cardiac_cycles"] = 1
    input_file["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = 240
    input_file["simulation_parameters"]["steady_initial"] = True
    
    if not os.path.exists(f'trees/zerod_input_TP_R/{tree_name}'):
        os.makedirs(f'trees/zerod_input_TP_R/{tree_name}')
    with open(f'trees/zerod_input_TP_R/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input_TP_R/{tree_name}/solver_0d.json")