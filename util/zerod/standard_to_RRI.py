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

def get_RRI_values(areas, tangents):
    anatomy = "CCO_80"
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
    
    #model_name = "CCO_ng_88_nl_0_lw_25_ne_5000_bs_10_dr_0.9_model"
    model_name = "CCO_80_ng_126_nl_2_lw_100_ne_500_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    if R_lin_star_pred1 < 0:
        R_lin_star_pred1 = R_lin_star_pred1 * 0
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    print(f"Predicted R_lin_star1: {R_lin_star_pred1}, R_lin_star2: {R_lin_star_pred2}, R_quad_star1: {R_quad_star_pred1}, R_quad_star2: {R_quad_star_pred2}")
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = 1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)
    L = [0, 0]
    return [float(R_lin1[0][0]), float(R_lin2[0][0])], [float(R_quad1[0][0]), float(R_quad2[0][0])], L


tree_name = "tree_80"
with open(f'trees/zerod_input_standard/{tree_name}/solver_0d.json') as json_file:
    input_file = json.load(json_file)

num_junctions = 0
num_non_bifs = 0
out_of_dist_cnt = 0

r_lin_list = []
r_quad_list = []
areas_list = []

for junction in input_file["junctions"]:
    
    if len(junction["outlet_vessels"]) > 2:
        num_non_bifs += 1
        continue
    elif len(junction["outlet_vessels"]) == 2:
        num_junctions += 1
        r_lin, r_quad, L = get_RRI_values(areas = junction["areas"], tangents = junction["tangents"])

        areas_list += junction["areas"][1:]
        r_lin_list +=r_lin
        r_quad_list += r_quad

        in_dist = True
        # for r_lin_val in r_lin:
        #     if abs(r_lin_val) > 50:
        #         print(f"Out of distribution R_lin value: {r_lin_val}")
        #         in_dist = False
            # if r_lin_val < 0:
            #     print(f"Negative R_lin value: {r_lin_val}")
            #     in_dist = False
        # for r_quad_val in r_quad:
        #     if abs(r_quad_val) > 10:
        #         print(f"Out of distribution R_quad value: {r_quad_val}")
        #         in_dist = False
            # if r_quad_val < 0:  
            #     print(f"Negative R_quad value: {r_quad_val}")
            #     in_dist = False
        if in_dist:
                # print(f"R_lin values: {r_lin}")
                # print(f"R_quad values: {r_quad}") 
            junction["junction_type"] = "BloodVesselJunction"
            junction["junction_values"] = {"R_poiseuille": r_lin, 
                                "stenosis_coefficient": r_quad,
                                "L": L,}
        else:
            out_of_dist_cnt += 1

print(f"{num_junctions} junctions processed.")
print(f"{num_non_bifs} junctions with more than 2 outlets.")
print(f"{out_of_dist_cnt} junctions with out of distribution values.")

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

if not os.path.exists(f'trees/zerod_input_RRI/{tree_name}'):
    os.makedirs(f'trees/zerod_input_RRI/{tree_name}')
with open(f'trees/zerod_input_RRI/{tree_name}/solver_0d.json', 'w') as fp:
    json.dump(input_file, fp)
print(f"RRI 0D input file saved to trees/zerod_input_RRI/{tree_name}/solver_0d.json")