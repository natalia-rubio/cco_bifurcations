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


anatomy = "CCO_80"
set_type = "random"

scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
model_name = "CCO_80_ng_120_nl_1_lw_4_ne_2000_bs_10_dr_0.9_model"
nn_model = dill_load(f"results/models/{anatomy}/{model_name}")

re_char = 4500
A_char = 0.25
U_char = re_char * 0.04 / (1.06 * 2*np.sqrt(A_char/np.pi))

daughter1_area_ratio_default = 0.8
daughter2_area_ratio_default = 0.8
daughter1_angle_default = 0.85
daughter2_angle_default = 1.5

n_pts = 100
daughter1_area_ratio_arr = np.linspace(0.4, 1.2, n_pts)
daughter2_area_ratio_arr = np.linspace(0.2, 1.4, n_pts)
daughter1_angle_arr = np.linspace(0.4, 1.4, n_pts)
daughter2_angle_arr = np.linspace(1.0, 1.9, n_pts)

R_lin_star_pred1_arr = np.zeros(n_pts)
R_lin_star_pred2_arr = np.zeros(n_pts)
R_quad_star_pred1_arr = np.zeros(n_pts)
R_quad_star_pred2_arr = np.zeros(n_pts)

R_lin1_arr = np.zeros(n_pts)
R_lin2_arr = np.zeros(n_pts)
R_quad1_arr = np.zeros(n_pts)
R_quad2_arr = np.zeros(n_pts)

fig, axs = plt.subplots(4, 2, figsize = (20, 20))
fig_star, axs_star = plt.subplots(4, 2, figsize = (20, 20))

# Loop over Daughter 1 area ratio
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_arr[i], dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = 1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)
    
    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_lin_star_pred2_arr[i] = float(R_lin_star_pred2[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])
    R_quad_star_pred2_arr[i] = float(R_quad_star_pred2[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_lin2_arr[i] = float(R_lin2[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])
    R_quad2_arr[i] = float(R_quad2[0][0])

axs[0, 0].plot(daughter1_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[0, 0].plot(daughter1_area_ratio_arr, R_lin2_arr, label = "R_lin_2")
axs[1, 0].plot(daughter1_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[1, 0].plot(daughter1_area_ratio_arr, R_quad2_arr, label = "R_quad2")
axs[1, 0].set_xlabel("Daughter 1 Area Ratio")
axs[1, 0].set_ylabel("R_quad")
axs[0, 0].set_ylabel("R_lin")

axs_star[0, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred1_arr, label = "R_lin_star1")
axs_star[0, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred2_arr, label = "R_lin_star2")
axs_star[1, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred1_arr, label = "R_quad_star1")
axs_star[1, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred2_arr, label = "R_quad_star2")
axs_star[1, 0].set_xlabel("Daughter 1 Area Ratio")
axs_star[1, 0].set_ylabel("R_quad^*")
axs_star[0, 0].set_ylabel("R_lin^*")


# Loop over Daughter 2 area ratio
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_arr[i], dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = 1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)

    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_lin_star_pred2_arr[i] = float(R_lin_star_pred2[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])
    R_quad_star_pred2_arr[i] = float(R_quad_star_pred2[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_lin2_arr[i] = float(R_lin2[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])
    R_quad2_arr[i] = float(R_quad2[0][0])

axs[0, 1].plot(daughter2_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[0, 1].plot(daughter2_area_ratio_arr, R_lin2_arr, label = "R_lin_2")
axs[1, 1].plot(daughter2_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[1, 1].plot(daughter2_area_ratio_arr, R_quad2_arr, label = "R_quad2")
axs[1, 1].set_xlabel("Daughter 2 Area Ratio")
axs[1, 1].set_ylabel("R_quad")
axs[0, 1].set_ylabel("R_lin")


axs_star[0, 1].plot(daughter2_area_ratio_arr, R_lin_star_pred1_arr, label = "R_lin_star1")
axs_star[0, 1].plot(daughter2_area_ratio_arr, R_lin_star_pred2_arr, label = "R_lin_star2")
axs_star[1, 1].plot(daughter2_area_ratio_arr, R_quad_star_pred1_arr, label = "R_quad_star1")
axs_star[1, 1].plot(daughter2_area_ratio_arr, R_quad_star_pred2_arr, label = "R_quad_star2")
axs_star[1, 1].set_xlabel("Daughter 2 Area Ratio")
axs_star[1, 1].set_ylabel("R_quad^*")
axs_star[0, 1].set_ylabel("R_lin^*")

# Loop over Daughter 1 angle
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_arr[i], dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_default, dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = 1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)

    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_lin_star_pred2_arr[i] = float(R_lin_star_pred2[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])
    R_quad_star_pred2_arr[i] = float(R_quad_star_pred2[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_lin2_arr[i] = float(R_lin2[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])
    R_quad2_arr[i] = float(R_quad2[0][0])

axs[2, 0].plot(daughter1_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[2, 0].plot(daughter1_area_ratio_arr, R_lin2_arr, label = "R_lin_2")
axs[3, 0].plot(daughter1_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[3, 0].plot(daughter1_area_ratio_arr, R_quad2_arr, label = "R_quad2")
axs[3, 0].set_xlabel("Daughter 1 Angle")
axs[3, 0].set_ylabel("R_quad")
axs[2, 0].set_ylabel("R_lin")

axs_star[2, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred1_arr, label = "R_lin_star1")
axs_star[2, 0].plot(daughter1_area_ratio_arr, R_lin_star_pred2_arr, label = "R_lin_star2")
axs_star[3, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred1_arr, label = "R_quad_star1")
axs_star[3, 0].plot(daughter1_area_ratio_arr, R_quad_star_pred2_arr, label = "R_quad_star2")
axs_star[3, 0].set_xlabel("Daughter 1 Angle")
axs_star[3, 0].set_ylabel("R_quad^*")
axs_star[2, 0].set_ylabel("R_lin^*")

# Loop over Daughter 2 angle
for i in range(n_pts):

    input_tens = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_default, dtype=jnp.float32), "daughter1_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_default, dtype=jnp.float32), "daughter2_area_ratio"),
                            scale_jax(scaling_dict, jnp.asarray(daughter1_angle_default, dtype=jnp.float32), "daughter1_angle"),
                            scale_jax(scaling_dict, jnp.asarray(daughter2_angle_arr[i], dtype=jnp.float32), "daughter2_angle"),
                                ]).reshape(1,-1)
    
    coefs_pred = predict(input_tens, nn_model.weights)

    R_lin_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][0], "R_lin_star1")
    R_lin_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][1], "R_lin_star2")
    R_quad_star_pred1 = inv_scale_jax(scaling_dict, coefs_pred[0][2], "R_quad_star1")
    R_quad_star_pred2 = inv_scale_jax(scaling_dict, coefs_pred[0][3], "R_quad_star2")
    
    R_lin1 = 1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)
    R_lin2 = 1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)
    R_quad1 = 1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)
    R_quad2 = 1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)

    R_lin_star_pred1_arr[i] = float(R_lin_star_pred1[0][0])
    R_lin_star_pred2_arr[i] = float(R_lin_star_pred2[0][0])
    R_quad_star_pred1_arr[i] = float(R_quad_star_pred1[0][0])
    R_quad_star_pred2_arr[i] = float(R_quad_star_pred2[0][0])

    R_lin1_arr[i] = float(R_lin1[0][0])
    R_lin2_arr[i] = float(R_lin2[0][0])
    R_quad1_arr[i] = float(R_quad1[0][0])
    R_quad2_arr[i] = float(R_quad2[0][0])

axs[2, 1].plot(daughter1_area_ratio_arr, R_lin1_arr, label = "R_lin1")
axs[2, 1].plot(daughter1_area_ratio_arr, R_lin2_arr, label = "R_lin_2")
axs[3, 1].plot(daughter1_area_ratio_arr, R_quad1_arr, label = "R_quad1")
axs[3, 1].plot(daughter1_area_ratio_arr, R_quad2_arr, label = "R_quad2")
axs[3, 1].set_xlabel("Daughter 2 Angle")
axs[3, 1].set_ylabel("R_quad")
axs[2, 1].set_ylabel("R_lin")

axs_star[2, 1].plot(daughter1_area_ratio_arr, R_lin_star_pred1_arr, label = "R_lin_star1")
axs_star[2, 1].plot(daughter1_area_ratio_arr, R_lin_star_pred2_arr, label = "R_lin_star2")
axs_star[3, 1].plot(daughter1_area_ratio_arr, R_quad_star_pred1_arr, label = "R_quad_star1")
axs_star[3, 1].plot(daughter1_area_ratio_arr, R_quad_star_pred2_arr, label = "R_quad_star2")
axs_star[3, 1].set_xlabel("Daughter 2 Angle")
axs_star[3, 1].set_ylabel("R_quad^*")
axs_star[2, 1].set_ylabel("R_lin^*")

fig.savefig("results/models/CCO_80_RRI_vs_input.png")
fig_star.savefig("results/models/CCO_80_RRI_star_vs_input.png")