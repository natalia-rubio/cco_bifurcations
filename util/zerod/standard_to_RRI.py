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
    anatomy = "Jan_CCO_80"
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
    #print(f"area ratio: {(areas[1]/areas[0], areas[2]/areas[0])}")
    if areas[1] < 0.01:
        pdb.set_trace()
    print(f"angle diff rri: {(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])), get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2])))}")
    #model_name = "CCO_ng_88_nl_0_lw_25_ne_5000_bs_10_dr_0.9_model"
    model_name = "Jan_CCO_80_ng_120_nl_1_lw_4_ne_2000_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    # if R_lin_star_pred1 < 0:
    #     R_lin_star_pred1 = R_lin_star_pred1 * 0
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    #print(f"Predicted R_lin_star1: {R_lin_star_pred1}, R_lin_star2: {R_lin_star_pred2}, R_quad_star1: {R_quad_star_pred1}, R_quad_star2: {R_quad_star_pred2}")
    R_lin1 = -1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = -1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = -1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = -1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)
    L = [0, 0]
    return [float(R_lin1[0][0]), float(R_lin2[0][0])], [float(R_quad1[0][0]), float(R_quad2[0][0])], L, [float(R_lin_star_pred1[0][0]), float(R_lin_star_pred2[0][0])]

def get_RRI_values_trif(areas, tangents):
    anatomy = "CCO_80"
    set_type = "random"

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    re_char = 4500
    U_char = re_char * 0.04 / (1.06 * 2*np.sqrt(areas[0]/np.pi))
    A_char = areas[0]
    if areas[1] < 0.01:
        pdb.set_trace()
    model_name = "CCO_80_ng_120_nl_3_lw_200_ne_2000_bs_10_dr_0.9_model"
    nn_model = dill_load(f"results/models/{anatomy}/{model_name}")
    
    input_tens1 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(areas[1]/areas[0], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(areas[2]/areas[0], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])), dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[2])), dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)

    coefs_pred = predict(input_tens1, nn_model.weights)

    R_lin_star_pred1_1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1_1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")


    input_tens2 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(areas[1]/areas[0], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(areas[3]/areas[0], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[1])), dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(get_angle_diff(np.asarray(tangents[0]), np.asarray(tangents[3])), dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)

    coefs_pred = predict(input_tens1, nn_model.weights)

    R_lin_star_pred1_2 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred3 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1_2 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred3 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")

    R_lin_star_pred1 = (R_lin_star_pred1_1 + R_lin_star_pred1_2)/2
    R_quad_star_pred1 = (R_quad_star_pred1_1 + R_quad_star_pred1_2)/2

    #print(f"Predicted R_lin_star1: {R_lin_star_pred1}, R_lin_star2: {R_lin_star_pred2}, R_quad_star1: {R_quad_star_pred1}, R_quad_star2: {R_quad_star_pred2}")
    R_lin1 = -1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = -1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_lin3 = -1.06 * jnp.square(U_char) * R_lin_star_pred3 /  (A_char * U_char)
    R_quad1 = -1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = -1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)
    R_quad3 = -1.06 * jnp.square(U_char) * R_quad_star_pred3 / jnp.square(A_char * U_char)
    L = [0, 0, 0]
    return [float(R_lin1[0][0]), float(R_lin2[0][0]), float(R_lin3[0][0])], [float(R_quad1[0][0]), float(R_quad2[0][0]), float(R_quad3[0][0])], L, [float(R_lin_star_pred1[0][0]), float(R_lin_star_pred2[0][0])]


def look_up_RRI_values(areas, tangents, tree_dict):
    anatomy = "CCO"
    for i in range(len(tree_dict[anatomy]["inlet_area"])):
        #pdb.set_trace()
        if np.linalg.norm((areas[0] - tree_dict[anatomy]["inlet_area"][i])/areas[0]) > 5e-2:
            continue
        if np.linalg.norm((areas[1] - tree_dict[anatomy]["outlet_area"][i][0])/areas[1]) > 5e-2: 
            continue
        if np.linalg.norm((areas[2] - tree_dict[anatomy]["outlet_area"][i][1])/areas[2]) > 5e-2:
            continue
        if np.linalg.norm((np.asarray(tangents[0]).reshape(3,1) - tree_dict[anatomy]["inlet_angles"][i])) > 5e-2:
            continue
    # assert areas[0] == tree_dict[anatomy]["inlet_area"][junction_number]
    # assert areas[1] == tree_dict[anatomy]["daughter1_area"][junction_number]
    # assert areas[2] == tree_dict[anatomy]["daughter2_area"][junction_number]

        r_lin_1 = -1*tree_dict[anatomy]["r_lin_1"][i]
        r_lin_2 = -1*tree_dict[anatomy]["r_lin_2"][i]
        r_quad_1 = -1*tree_dict[anatomy]["r_quad_1"][i]
        r_quad_2 = -1*tree_dict[anatomy]["r_quad_2"][i]
        return [r_lin_1, r_lin_2], [r_quad_1, r_quad_2], i
    pdb.set_trace()
    return

def look_up_RRI_values_trif(areas, tangents, tree_dict):
    anatomy = "CCO"
    for i1 in range(len(tree_dict[anatomy]["inlet_area"])):
        #pdb.set_trace()
        if np.linalg.norm((areas[0] - tree_dict[anatomy]["inlet_area"][i1])/areas[0]) > 5e-2:
            continue
        if np.linalg.norm((areas[1] - tree_dict[anatomy]["outlet_area"][i1][0])/areas[1]) > 5e-2: 
            continue
        if np.linalg.norm((areas[2] - tree_dict[anatomy]["outlet_area"][i1][1])/areas[2]) > 5e-2:
            continue
        if np.linalg.norm((np.asarray(tangents[0]).reshape(3,1) - tree_dict[anatomy]["inlet_angles"][i1])) > 5e-2:
            continue
    # assert areas[0] == tree_dict[anatomy]["inlet_area"][junction_number]
    # assert areas[1] == tree_dict[anatomy]["daughter1_area"][junction_number]
    # assert areas[2] == tree_dict[anatomy]["daughter2_area"][junction_number]

        r_lin_1_1 = -1*tree_dict[anatomy]["r_lin_1"][i1]
        r_lin_2 = -1*tree_dict[anatomy]["r_lin_2"][i1]
        r_quad_1_1 = -1*tree_dict[anatomy]["r_quad_1"][i1]
        r_quad_2 = -1*tree_dict[anatomy]["r_quad_2"][i1]
        break

    for i2 in range(len(tree_dict[anatomy]["inlet_area"])):
        #pdb.set_trace()
        if i2 == i1:
            continue
        if np.linalg.norm((areas[0] - tree_dict[anatomy]["inlet_area"][i2])/areas[0]) > 5e-2:
            continue
        if np.linalg.norm((areas[1] - tree_dict[anatomy]["outlet_area"][i2][0])/areas[1]) > 5e-2: 
            continue
        if np.linalg.norm((areas[2] - tree_dict[anatomy]["outlet_area"][i2][1])/areas[3]) > 5e-2:
            continue
        if np.linalg.norm((np.asarray(tangents[0]).reshape(3,1) - tree_dict[anatomy]["inlet_angles"][i2])) > 5e-2:
            continue

    # assert areas[0] == tree_dict[anatomy]["inlet_area"][junction_number]
    # assert areas[1] == tree_dict[anatomy]["daughter1_area"][junction_number]
    # assert areas[2] == tree_dict[anatomy]["daughter2_area"][junction_number]

        r_lin_1_2 = -1*tree_dict[anatomy]["r_lin_1"][i2]
        r_lin_3 = -1*tree_dict[anatomy]["r_lin_2"][i2]
        r_quad_1_2 = -1*tree_dict[anatomy]["r_quad_1"][i2]
        r_quad_3 = -1*tree_dict[anatomy]["r_quad_2"][i2]
        break
    r_lin_1 = (r_lin_1_1 + r_lin_1_2)
    r_quad_1 = (r_quad_1_1 + r_quad_1_2)
    return [r_lin_1, r_lin_2, r_lin_3], [r_quad_1, r_quad_2, r_quad_3], i1, i2


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
        
        if len(junction["outlet_vessels"]) > 2:
            num_non_bifs += 1
            print(f"Junction with more than 2 outlets: {junction["junction_name"]}.  Areas {junction['areas']}")
            #r_lin, r_quad, L, r_lin_star = get_RRI_values_trif(areas = junction["areas"], tangents = junction["tangents"])
            r_lin, r_quad, ind1, ind2 = look_up_RRI_values_trif(areas = junction["areas"], tangents = junction["tangents"], tree_dict = char_val_dict)
            L = [0,0,0]
            if ind1 not in inds:
                inds.append(ind1)
            else:
                print(f"Duplicate junction index: {ind1}")
                pdb.set_trace()
            if ind2 not in inds:
                inds.append(ind2)
            else:
                print(f"Duplicate junction index: {ind2}")
                pdb.set_trace()   
            num_junctions += 1
            #continue
        elif len(junction["outlet_vessels"]) == 2:
            
            #r_lin, r_quad, L, r_lin_star = get_RRI_values(areas = junction["areas"], tangents = junction["tangents"])
            r_lin, r_quad,ind  = look_up_RRI_values(areas = junction["areas"], tangents = list(np.asarray(junction["tangents"])), tree_dict = char_val_dict)
            if ind not in inds:
                inds.append(ind)
            else:
                print(f"Duplicate junction index: {ind}")
                pdb.set_trace()
            areas_list += junction["areas"][1:]
            r_lin_list +=r_lin
            r_quad_list += r_quad
            L = 0
            num_junctions += 1

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