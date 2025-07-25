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
#from util.tree.extract_true_junctions_helpers import *

def get_input_file_junction_dict_master(tree_name):
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    
    try:
        input_file_standard = f'trees/zerod_input/original_BCs/{tree_name_base}/{tree_name}/solver_0d.json'

        with open(input_file_standard) as json_file:
            input_file = json.load(json_file)
    except:
        input_file_standard = f'trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json'

        with open(input_file_standard) as json_file:
            input_file = json.load(json_file)

    
    terminal_junction_dict = {}
    for junction in input_file["junctions"]:
        if len(junction["outlet_vessels"]) == 1:
            continue
        junction_name = junction["junction_name"]
        junction_id = int(junction_name[1:])
        inlet_vessel = junction["inlet_vessels"][0]
        terminal_junction_dict[inlet_vessel] = {}
        terminal_junction_dict[inlet_vessel]["terminal_junction_id"] = junction_id
        terminal_junction_dict[inlet_vessel]["terminal_junction_name"] = junction_name

    bc_dict = {}
    for bc in input_file["boundary_conditions"]:
        bc_dict[bc["bc_name"]] = {}
        for key, value in bc.items():
            bc_dict[bc["bc_name"]][key] = value

    lengths = []; areas = []; vessel_ids = []; branch_ids = []; R_poiseuille = []
    vessel_dict = {}
    for vessel in input_file["vessels"]:
        if "connector" in vessel["vessel_name"]:
            continue
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        branch_ids.append(branch_id)
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        vessel_ids.append( vessel["vessel_id"])
        lengths.append(vessel["vessel_length"])
        R_poiseuille.append(vessel["zero_d_element_values"]["R_poiseuille"])
        
        if vessel["zero_d_element_values"]["R_poiseuille"] == 0:
            area = 0
        else:
            area = np.sqrt(0.04*8*np.pi*vessel["vessel_length"] / vessel["zero_d_element_values"]["R_poiseuille"])
        areas.append(area)
        vessel_dict[vessel["vessel_id"]] = {"branch_id": branch_id,
                                                "seg_id": seg_id,
                                                "vessel_id": vessel["vessel_id"],
                                                "length": vessel["vessel_length"],
                                                "area": area,
                                                "R_poiseuille": vessel["zero_d_element_values"]["R_poiseuille"],}
        if "boundary_conditions" in vessel.keys(): 
            vessel_dict[vessel["vessel_id"]]["boundary_conditions"] =  vessel["boundary_conditions"]
        

        vessel["zero_d_element_values"]["pressure_recovery_coefficient"] = 0
        if not branch_id == 0:
            vessel["zero_d_element_values"]["R_poiseuille"] = 0
            vessel["zero_d_element_values"]["stenosis_coefficient"] = 0
            
            vessel["zero_d_element_values"]["L"] = 0
            vessel["zero_d_element_values"]["C"] = 0
    vessel_arr_dict = {"branch_ids": np.asarray(branch_ids), "vessel_ids": np.asarray(vessel_ids), "lengths": np.asarray(lengths), "areas": np.asarray(areas), "R_poiseuille": np.asarray(R_poiseuille)}
    # pdb.set_trace()
    # junction_geo_dict = {"primary_area_ratio": [], "total_area_ratio": [], "primary_angle": [], "angle_diff": [], "length": [], "norm_length": []}
    junction_dict = {}
    for junction in input_file["junctions"]:

        if len(junction["outlet_vessels"]) == 1:
            continue
        
        # Initialize the dictionary for the junction
        junction_name = junction["junction_name"]
        junction_id = int(junction_name[1:])
        #print(f"Processing junction {junction_name}")
        junction_dict[junction_name] = {}
        junction_dict[junction_name]["junction_id"] = junction_id

        # Get the inlet vessel id and branch id
        original_inlet_vessel_id = copy.copy(junction["inlet_vessels"][0])
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."
        junction_dict[junction_name]["inlet_vessel_id"] = original_inlet_vessel_id
        junction_dict[junction_name]["inlet_branch_id"] = vessel_arr_dict["branch_ids"][np.where(vessel_arr_dict["vessel_ids"] == original_inlet_vessel_id)]
        
        # Get the outlet vessel ids and branch ids
        num_outlets = len(junction["outlet_vessels"]); i = 0; aux_i = 1
        assert num_outlets == 2, "Junction with more than two outlet vessels."
        outlet_branch = vessel_arr_dict["branch_ids"][np.where(vessel_arr_dict["vessel_ids"] == junction["outlet_vessels"][i])]
        aux_outlet_branch = vessel_arr_dict["branch_ids"][np.where(vessel_arr_dict["vessel_ids"] == junction["outlet_vessels"][aux_i])]
        junction_dict[junction_name]["0D_outlet1_vessel_id"] = junction["outlet_vessels"][i] 
        junction_dict[junction_name]["0D_outlet1_branch_id"] = outlet_branch[0]
        junction_dict[junction_name]["0D_outlet2_vessel_id"] = junction["outlet_vessels"][aux_i]
        junction_dict[junction_name]["0D_outlet2_branch_id"] = aux_outlet_branch[0]

        terminal_outlet_vessel_id = np.max(vessel_arr_dict["vessel_ids"][np.where(vessel_arr_dict["branch_ids"] == outlet_branch)])
        junction_dict[junction_name]["0D_terminal_outlet_vessel_id"] = terminal_outlet_vessel_id
        if "boundary_conditions" in vessel_dict[terminal_outlet_vessel_id].keys():
            junction_dict[junction_name]["0D_termination"] = "resistance"
            junction_dict[junction_name]["0D_termination_RI"] = "resistance"
            junction_dict[junction_name]["0D_bc_resistance_name"] = vessel_dict[terminal_outlet_vessel_id]["boundary_conditions"]["outlet"]
            junction_dict[junction_name]["0D_bc_geo_resistance"] = bc_dict[junction_dict[junction_name]["0D_bc_resistance_name"]]["bc_values"]["R"]
            junction_dict[junction_name]["0D_bc_geo_resistance_RI"] = bc_dict[junction_dict[junction_name]["0D_bc_resistance_name"]]["bc_values"]["R"]

        else:
            junction_dict[junction_name]["0D_termination"] = "junction"
            junction_dict[junction_name]["0D_termination_RI"] = "junction"
            junction_dict[junction_name]["0D_terminal_junction_name"] = terminal_junction_dict[terminal_outlet_vessel_id]["terminal_junction_name"]
            junction_dict[junction_name]["0D_terminal_junction_id"] = terminal_junction_dict[terminal_outlet_vessel_id]["terminal_junction_id"]


        aux_terminal_outlet_vessel_id = np.max(vessel_arr_dict["vessel_ids"][np.where(vessel_arr_dict["branch_ids"] == aux_outlet_branch)])
        junction_dict[junction_name]["0D_aux_terminal_outlet_vessel_id"] = aux_terminal_outlet_vessel_id
        if "boundary_conditions" in vessel_dict[aux_terminal_outlet_vessel_id].keys():
            junction_dict[junction_name]["0D_aux_termination"] = "resistance"
            junction_dict[junction_name]["0D_aux_termination_RI"] = "resistance"
            junction_dict[junction_name]["0D_aux_bc_resistance_name"] = vessel_dict[aux_terminal_outlet_vessel_id]["boundary_conditions"]["outlet"]
            junction_dict[junction_name]["0D_aux_bc_geo_resistance"] = bc_dict[junction_dict[junction_name]["0D_aux_bc_resistance_name"]]["bc_values"]["R"]
            junction_dict[junction_name]["0D_aux_bc_geo_resistance_RI"] = bc_dict[junction_dict[junction_name]["0D_aux_bc_resistance_name"]]["bc_values"]["R"]
        else:
            junction_dict[junction_name]["0D_aux_termination"] = "junction"
            junction_dict[junction_name]["0D_aux_termination_RI"] = "junction"
            junction_dict[junction_name]["0D_aux_terminal_junction_name"] = terminal_junction_dict[aux_terminal_outlet_vessel_id]["terminal_junction_name"]
            junction_dict[junction_name]["0D_aux_terminal_junction_id"] = terminal_junction_dict[aux_terminal_outlet_vessel_id]["terminal_junction_id"]
            
        # Get outlet lengths
        A_char = junction["areas"][0]
        length = junction["lengths"][i] + float(np.sum(vessel_arr_dict["lengths"][np.where(vessel_arr_dict["branch_ids"] == outlet_branch)]))
        aux_length = junction["lengths"][aux_i] + float(np.sum(vessel_arr_dict["lengths"][np.where(vessel_arr_dict["branch_ids"] == aux_outlet_branch)]))
        L_char = np.sqrt(A_char/np.pi)
        junction_dict[junction_name]["0D_R_poiseuille_outlet1"] = np.sum(vessel_arr_dict["R_poiseuille"][np.where(vessel_arr_dict["branch_ids"] == outlet_branch)])
        junction_dict[junction_name]["0D_R_poiseuille_outlet2"] = np.sum(vessel_arr_dict["R_poiseuille"][np.where(vessel_arr_dict["branch_ids"] == aux_outlet_branch)])
        junction_dict[junction_name]["0D_length1"] = length
        junction_dict[junction_name]["0D_length1_base"] =  junction["lengths"][0]
        junction_dict[junction_name]["0D_length_star"] = (length - 0*junction_dict[junction_name]["0D_length1_base"])/L_char
        junction_dict[junction_name]["0D_length2"] = aux_length
        junction_dict[junction_name]["0D_length2_base"] =  junction["lengths"][1]
        junction_dict[junction_name]["0D_length2_star"] = (aux_length - 0*junction_dict[junction_name]["0D_length2_base"])/L_char

        
        primary_area = min(vessel_arr_dict["areas"][np.where(vessel_arr_dict["branch_ids"] == outlet_branch)])
        primary_area_ratio = primary_area/A_char
        aux_area = min(vessel_arr_dict["areas"][np.where(vessel_arr_dict["branch_ids"] == aux_outlet_branch)])
        aux_area_ratio = aux_area/A_char
        junction_dict[junction_name]["0D_inlet_area"] = A_char
        L_char = np.sqrt(A_char/np.pi)
        junction_dict[junction_name]["0D_L_char"] = L_char
        junction_dict[junction_name]["0D_outlet1_area"] = primary_area
        junction_dict[junction_name]["0D_outlet1_junction_area"] = junction["areas"][1]
        junction_dict[junction_name]["0D_outlet2_area"] = aux_area
        junction_dict[junction_name]["0D_outlet2_junction_area"] = junction["areas"][2]
        #pdb.set_trace()
        
        aux_area_tan = sum([junction["areas"][j+1] for j in range(0, num_outlets) if j != i])
        tangent_array = np.asarray(junction["tangents"])
        tangents = junction["tangents"]
        aux_tangent = 0 * tangent_array[0,:]
        for j in range(num_outlets):
            if j != i:
                aux_tangent += (tangent_array[j+1,:]*junction["areas"][j+1]/aux_area_tan)
        
        assert np.linalg.norm(aux_tangent - np.asarray(tangents[2])) < 0.01, "Tangent calculation not consistent."
        tangents = [junction["tangents"][0], junction["tangents"][i+1], list(aux_tangent)]
        daughter1_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1]))[0]
        daughter2_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2]))[0]
        junction_dict[junction_name]["0D_daughter1_angle"] = daughter1_angle
        junction_dict[junction_name]["0D_daughter2_angle"] = daughter2_angle
        inlet_tangent = np.asarray(tangents[0])
        junction_dict[junction_name]["0D_inlet_tangent"] = inlet_tangent
        junction_dict[junction_name]["0D_daughter1_tangent"] = tangents[1]
        junction_dict[junction_name]["0D_daughter2_tangent"] = tangents[2]
    # pdb.set_trace()
    return junction_dict
