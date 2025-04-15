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

    resistance_dict = load_dict(f"data/tree_80_resistance_dict_branch")
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
        junction_id = int(junction_name[1:])
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."

        # Skip junctions that have only one outlet vessel
        if len(junction["outlet_vessels"]) == 1:
            continue


        L = [0 * area for area in junction["areas"][1:]]
            
        junction["junction_type"] = "BloodVesselJunction" 
        #pdb.set_trace()
        junction["junction_values"] = {"R_poiseuille": resistance_dict["junctions"][junction_id]["R"], 
                            "stenosis_coefficient": L,
                            "pressure_recovery_coefficient": L,
                            "L": L,}
        
        # Add inlet resistor vessel
        new_vessel_id = max_vessel_id + 1
        inlet_vessel_dict = {'vessel_id': new_vessel_id, 
                       'vessel_length': 0, 
                       'vessel_name':  f'branch{new_vessel_id}_seg0', 
                       'zero_d_element_type': 'BloodVessel', 
                       'zero_d_element_values': {'C': 0, 'L': 0, 
                                                 'R_poiseuille': 0, 
                                                 'stenosis_coefficient': 0,
                                                 'pressure_recovery_coefficient': 0,}}
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
    
    for vessel in input_file["vessels"]:
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        print(f"Branch ID: {branch_id}, Seg ID: {seg_id}")

        vessel["zero_d_element_values"]["R_poiseuille"] = 0
        vessel["zero_d_element_values"]["stenosis_coefficient"] = 0
        #if "pressure_recovery_coefficient" in vessel["zero_d_element_values"].keys():
        vessel["zero_d_element_values"]["pressure_recovery_coefficient"] = 0
        vessel["zero_d_element_values"]["C"] = 0
        vessel["zero_d_element_values"]["L"] = 0

        if branch_id in resistance_dict["vessels"].keys():
            if seg_id == 0:
                vessel["zero_d_element_values"]["R_poiseuille"] = resistance_dict["vessels"][branch_id]["R"]


    # num_steps = 100
    # max_Q = input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]
    # input_file["boundary_conditions"][0]["bc_values"]["Q"] = list(np.linspace(0, max_Q, num_steps))
    # input_file["boundary_conditions"][0]["bc_values"]["t"] = list(np.linspace(0, 1, num_steps))
    # input_file["simulation_parameters"]["number_of_time_pts_per_cardiac_cycle"] = num_steps
    # input_file["simulation_parameters"]["number_of_cardiac_cycles"] = 1
    # input_file["simulation_parameters"]["steady_initial"] = False
    # input_file["simulation_parameters"]["maximum_nonlinear_iterations"] = 100

    input_file["junctions"] += new_junction_list
    if not os.path.exists(f'trees/zerod_input_{zerod_gen}_lookup_branch_PR0/{tree_name}_{flow_amp}'):
        os.makedirs(f'trees/zerod_input_{zerod_gen}_lookup_branch_PR0/{tree_name}_{flow_amp}')
    with open(f'trees/zerod_input_{zerod_gen}_lookup_branch_PR0/{tree_name}_{flow_amp}/solver_0d.json', 'w') as fp:
        json.dump(input_file, indent = 4, fp = fp)
    print(f"RRI 0D input file saved to trees/zerod_input_lookup_branch_PR0/{tree_name}_{flow_amp}/solver_0d.json")