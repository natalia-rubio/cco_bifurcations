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
    
    tree_name = sys.argv[1]
    
    input_file_standard = f'trees/zerod_input/standard/{tree_name}/solver_0d.json'
    with open(input_file_standard) as json_file:
        input_file = json.load(json_file)


    lengths = []; areas = []; vessel_ids = []; branch_ids = []
    for vessel in input_file["vessels"]:
        branch_id = int(vessel["vessel_name"].split("_")[0][6:])
        branch_ids.append(branch_id)
        seg_id = int(vessel["vessel_name"].split("_")[1][3:])
        vessel_ids.append( vessel["vessel_id"])
        lengths.append(vessel["vessel_length"])
        areas.append(np.sqrt(0.04*8*np.pi*vessel["vessel_length"] /vessel["zero_d_element_values"]["R_poiseuille"]))

    length_dict = {"branch_ids": np.asarray(branch_ids), "vessel_ids": np.asarray(vessel_ids), "lengths": np.asarray(lengths), "areas": np.asarray(areas)}

    junction_geo_dict = {"primary_area_ratio": [], "total_area_ratio": [], "primary_angle": [], "angle_diff": [], "length": [], "norm_length": []}

    for junction in input_file["junctions"]:
        original_inlet_vessel_id = copy.copy(junction["inlet_vessels"][0])
        junction_name = junction["junction_name"]
        print(f"Processing junction {junction_name}")
        assert len(junction["inlet_vessels"]) == 1; "Junction with more than one inlet vessel."

        # Skip junctions that have only one outlet vessel
        if len(junction["outlet_vessels"]) == 1:
            continue

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

            A_char = junction["areas"][0]
            L_char = np.sqrt(A_char/np.pi)
            primary_area = min(length_dict["areas"][np.where(length_dict["branch_ids"] == outlet_branch)])
            primary_area_ratio = primary_area/A_char
            aux_area = min(length_dict["areas"][np.where(length_dict["branch_ids"] == aux_outlet_branch)])
            aux_area_ratio = aux_area/A_char
            
            aux_area_tan = sum([junction["areas"][j+1] for j in range(0, num_outlets) if j != i])
            tangent_array = np.asarray(junction["tangents"])
            aux_tangent = 0 * tangent_array[0,:]
            for j in range(num_outlets):
                if j != i:
                    aux_tangent += (tangent_array[j+1,:]*junction["areas"][j+1]/aux_area_tan)
            tangents = [junction["tangents"][0], junction["tangents"][i+1], list(aux_tangent)]
            daughter1_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1]))[0]
            daughter2_angle = get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2]))[0]
            #pdb.set_trace()
            junction_geo_dict["primary_area_ratio"].append(primary_area_ratio)
            junction_geo_dict["total_area_ratio"].append(aux_area_ratio + primary_area_ratio)
            junction_geo_dict["primary_angle"].append(daughter1_angle)
            junction_geo_dict["angle_diff"].append(daughter2_angle + daughter1_angle)
            junction_geo_dict["length"].append(length)
            junction_geo_dict["norm_length"].append(length/L_char)

    fig, axes = plt.subplots(nrows=int(len(junction_geo_dict)/2), ncols=2, figsize=(8, 8))

    for i, (key, values) in enumerate(junction_geo_dict.items()):
        axes[int(i/2), int(i%2)].hist(values, bins=5, edgecolor='black')
        axes[int(i/2), int(i%2)].set_xlabel(f'{key}')
        axes[int(i/2), int(i%2)].set_ylabel('freq')
    plt.tight_layout()
    if not os.path.exists(f'results/tree_junction_geo_hists'):
        os.makedirs(f'results/tree_junction_geo_hists')
    plt.savefig(f'results/tree_junction_geo_hists/{tree_name}_junction_geo_hists.png', bbox_inches='tight')