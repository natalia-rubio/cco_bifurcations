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


anatomy = "angles_CCO"
set_type = "random"

scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
model_name = "angles_CCO_ng_1728_nl_3_lw_200_ne_1000_bs_40_dr_0.9_model"
nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

re_char = 4500
A_char = 0.25**2 * np.pi #0.0516194110 #0.25**2 * np.pi
print(f"A_char: {A_char}")
U_char = re_char * 0.04 / (1.06 * 2*np.sqrt(A_char/np.pi))

# daughter1_area_ratio_default = scaling_dict["daughter1_area_ratio"][3]
# daughter2_area_ratio_default = scaling_dict["daughter2_area_ratio"][3]
# daughter1_area_ratio_inv2_default = daughter1_area_ratio_default**-2
# daughter2_area_ratio_inv2_default = daughter2_area_ratio_default**-2
# daughter1_angle_default = scaling_dict["daughter1_angle"][0]
# daughter2_angle_default = scaling_dict["daughter2_angle"][0]
# length_default = scaling_dict["daughter1_length"][3]

daughter1_area_ratio_default = 0.9402850520484878 #scaling_dict["daughter1_area_ratio"][3]
daughter2_area_ratio_default = 0.3783777656544973 #scaling_dict["daughter2_area_ratio"][3]
daughter1_area_ratio_inv2_default = daughter1_area_ratio_default**-2
daughter2_area_ratio_inv2_default = daughter2_area_ratio_default**-2
daughter1_angle_default = 0.38699742 #scaling_dict["daughter1_angle"][0]
daughter2_angle_default = 0.73518744 #scaling_dict["daughter2_angle"][0]
length_default = 11.793104770350494 #scaling_dict["daughter1_length"][3]

n_pts = 100
daughter1_area_ratio_arr = np.linspace(scaling_dict["daughter1_area_ratio"][2], scaling_dict["daughter1_area_ratio"][3], n_pts)
daughter2_area_ratio_arr = np.linspace(scaling_dict["daughter2_area_ratio"][2], scaling_dict["daughter2_area_ratio"][3], n_pts)
daughter1_area_ratio_inv2_arr = daughter1_area_ratio_arr**-2
daughter2_area_ratio_inv2_arr = daughter2_area_ratio_arr**-2
daughter1_angle_arr = np.linspace(scaling_dict["daughter1_angle"][2], scaling_dict["daughter1_angle"][3], n_pts)
daughter2_angle_arr = np.linspace(scaling_dict["daughter2_angle"][2], scaling_dict["daughter2_angle"][3],n_pts)
length_arr = np.linspace(scaling_dict["daughter1_length"][2], scaling_dict["daughter1_length"][3], n_pts)

R_lin_star_pred1_arr = np.zeros(n_pts)
R_lin_star_pred2_arr = np.zeros(n_pts)
R_quad_star_pred1_arr = np.zeros(n_pts)
R_quad_star_pred2_arr = np.zeros(n_pts)

R_lin1_arr = np.zeros(n_pts)
R_lin2_arr = np.zeros(n_pts)
R_quad1_arr = np.zeros(n_pts)
R_quad2_arr = np.zeros(n_pts)

fig, axs = plt.subplots(6, 2, figsize = (16, 25))
fig_star, axs_star = plt.subplots(6, 2, figsize = (16, 25))

# Loop over Daughter 1 area ratio
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_arr[i], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2_arr[i], dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2_default, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(length_default, dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "daughter1_R_quad_star")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])

axs[0, 0].plot(daughter1_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[1, 0].plot(daughter1_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[1, 0].set_xlabel("Daughter 1 Area Ratio")
axs[1, 0].set_ylabel("R_quad")
axs[0, 0].set_ylabel("R_lin")

axs_star[0, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred1_arr, label = "daughter1_R_lin_star")
axs_star[0, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred2_arr, label = "daughter2_R_lin_star")
axs_star[1, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred1_arr, label = "daughter1_R_quad_star")
axs_star[1, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred2_arr, label = "daughter2_R_quad_star")
axs_star[1, 0].set_xlabel("Daughter 1 Area Ratio")
axs_star[1, 0].set_ylabel("R_quad^*")
axs_star[0, 0].set_ylabel("R_lin^*")


# Loop over Daughter 2 area ratio
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_arr[i], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2_default, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2_arr[i], dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(length_default, dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "daughter1_R_quad_star")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])

axs[0, 1].plot(daughter2_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[1, 1].plot(daughter2_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[1, 1].set_xlabel("Daughter 2 Area Ratio")
axs[1, 1].set_ylabel("R_quad")
axs[0, 1].set_ylabel("R_lin")


axs_star[0, 1].plot(daughter2_area_ratio_arr, R_lin_star_pred1_arr, label = "daughter1_R_lin_star")
axs_star[1, 1].plot(daughter2_area_ratio_arr, R_quad_star_pred1_arr, label = "daughter1_R_quad_star")
axs_star[1, 1].set_xlabel("Daughter 2 Area Ratio")
axs_star[1, 1].set_ylabel("R_quad^*")
axs_star[0, 1].set_ylabel("R_lin^*")

# Loop over Daughter 1 angle
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2_default, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2_default, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_arr[i], dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(length_default, dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "daughter1_R_quad_star")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])

axs[2, 0].plot(daughter1_angle_arr, R_lin1_arr, label = "R_lin1")
axs[3, 0].plot(daughter1_angle_arr, R_quad1_arr, label = "R_quad1")
axs[3, 0].set_xlabel("Daughter 1 Angle")
axs[3, 0].set_ylabel("R_quad")
axs[2, 0].set_ylabel("R_lin")

axs_star[2, 0].plot(daughter1_angle_arr, R_lin_star_pred1_arr, label = "daughter1_R_lin_star")
axs_star[3, 0].plot(daughter1_angle_arr, R_quad_star_pred1_arr, label = "daughter1_R_quad_star")
axs_star[3, 0].set_xlabel("Daughter 1 Angle")
axs_star[3, 0].set_ylabel("R_quad^*")
axs_star[2, 0].set_ylabel("R_lin^*")

# Loop over Daughter 2 angle
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2_default, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2_default, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_arr[i], dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(length_default, dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "daughter1_R_quad_star")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])

axs[2, 1].plot(daughter2_angle_arr, R_lin1_arr, label = "R_lin1")
axs[3, 1].plot(daughter2_angle_arr, R_quad1_arr, label = "R_quad1")
axs[3, 1].set_xlabel("Daughter 2 Angle")
axs[3, 1].set_ylabel("R_quad")
axs[2, 1].set_ylabel("R_lin")

axs_star[2, 1].plot(daughter1_area_ratio_arr, R_lin_star_pred1_arr, label = "daughter1_R_lin_star")
axs_star[3, 1].plot(daughter1_area_ratio_arr, R_quad_star_pred1_arr, label = "daughter1_R_quad_star")
axs_star[3, 1].set_xlabel("Daughter 2 Angle")
axs_star[3, 1].set_ylabel("R_quad^*")
axs_star[2, 1].set_ylabel("R_lin^*")

# Loop over length
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2_default, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2_default, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                            scale_jax(scaling_dict, jnp.asarray(length_arr[i], dtype=jnp.float32), "daughter1_length"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "daughter1_R_lin_star")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][4], "daughter1_R_quad_star")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])

axs[4, 0].plot(length_arr, R_lin1_arr, label = "R_lin1")
axs[5, 0].plot(length_arr, R_quad1_arr, label = "R_quad1")
axs[5, 0].set_xlabel("Daughter 2 Length")
axs[5, 0].set_ylabel("R_quad")
axs[4, 0].set_ylabel("R_lin")

axs_star[4, 0].plot(length_arr, R_lin_star_pred1_arr, label = "daughter1_R_lin_star")
axs_star[5, 0].plot(length_arr, R_quad_star_pred1_arr, label = "daughter1_R_quad_star")
axs_star[5, 0].set_xlabel("Daughter 2 Length")
axs_star[5, 0].set_ylabel("R_quad^*")
axs_star[4, 0].set_ylabel("R_lin^*")

fig.savefig(f"results/models/{anatomy}_RRI_vs_input.png", bbox_inches = "tight")
fig_star.savefig(f"results/models/{anatomy}_star_vs_input.png", bbox_inches = "tight")