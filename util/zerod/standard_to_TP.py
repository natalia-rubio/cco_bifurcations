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

    inds = []
    for junction in input_file["junctions"]:
        if len(junction["outlet_vessels"]) == 1:
            continue

        r_lin = [0 * area for area in junction["areas"][1:]]
        L = [0 * area for area in junction["areas"][1:]]
        inlet_area = junction["areas"][0]
        r_quad = [1.06 * 0.5 * (outlet_area**(-2) - inlet_area**(-2)) for outlet_area in junction["areas"][1:]]
            
        junction["junction_type"] = "BloodVesselJunction"
        junction["junction_values"] = {"R_poiseuille": r_lin, 
                            "stenosis_coefficient": r_quad,
                            "L": L,}


    print(f"{num_junctions} junctions processed.")

    plt.clf()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(areas_list[0::2], r_lin_list[0::2])
    axs[0, 0].set_title('Daughter 1')
    axs[0, 0].set_ylabel('R_lin')
    axs[0,1].scatter(areas_list[1::2], r_lin_list[1::2])
    axs[0,1].set_title('Daughter 2')
    axs[1,0].scatter(areas_list[0::2], r_quad_list[0::2])
    axs[1,0].set_ylabel('R_quad')
    axs[1,0].set_xlabel('Area')
    axs[1,1].scatter(areas_list[1::2], r_quad_list[1::2])
    axs[1,1].set_xlabel('Area')
    fig.savefig(f"results/CCO_hist_{tree_name}/RRI_values.png", bbox_inches='tight', transparent=False, format = "png")

    if not os.path.exists(f'trees/zerod_input_TP/{tree_name}'):
        os.makedirs(f'trees/zerod_input_TP/{tree_name}')
    with open(f'trees/zerod_input_TP/{tree_name}/solver_0d.json', 'w') as fp:
        json.dump(input_file, fp)
    print(f"RRI 0D input file saved to trees/zerod_input_TP/{tree_name}/solver_0d.json")