from ctypes.wintypes import PDWORD
import json
import pdb
from pyexpat import model
import sys
import os
from tabnanny import check
from matplotlib.pyplot import plot
from numpy import mean
import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import add_0D_resistance, add_0D_RI_resistance, get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, add_downstream_resistance_values_RI, add_solution_values, add_3D_resistance_total



def plot_resistances(tree_name):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    junction_dict_master = get_input_file_junction_dict_master(tree_name)

    time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["12", "25", "50", "100"]

    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    for flow_mag in flow_mag_list:
        add_solution_values(junction_dict_master, f"{tree_name_base}_flow_{flow_mag}", flow_mag, time_step = time_step1)

    add_3D_resistance_total(junction_dict_master, tree_name, flow_mag_list, time_step = time_step1)
    
    add_downstream_resistance_values(junction_dict_master)
    add_0D_RI_resistance(junction_dict_master, tree_name_base)
    add_downstream_resistance_values_RI(junction_dict_master)
    
    depth = []
    res_3d = []
    res_0d = []
    res_0d_RI = []
    
    for junction_name, junction_dict in junction_dict_master.items():
        depth.append(junction_dict["depth"])
        res_3d.append(junction_dict["3D_R_lin_total"])
        res_0d.append(((junction_dict["0D_geo_resistance"])**-1 + \
                        (junction_dict["0D_aux_geo_resistance"])**-1)**-1)
        #pdb.set_trace()
        res_0d_RI.append(((junction_dict["0D_geo_resistance_RI"])**-1 + \
                        (junction_dict["0D_aux_geo_resistance_RI"])**-1)**-1)
    
    depth = np.asarray(depth)
    res_3d = np.asarray(res_3d)
    res_0d = np.asarray(res_0d)
    res_0d_RI = np.asarray(res_0d_RI)
    
    plt.scatter(depth, res_3d, color='black', s = 20, label='3D Resistance')
    plt.scatter(depth, res_0d, color='orangered', s = 20, label='0D Resistance')
    plt.scatter(depth, res_0d_RI, color='royalblue', s = 20, label='0D RI Resistance')
    plt.xlabel("Depth (max number of bifurcations downstream)")
    plt.ylabel("Total Downstream Resistance ($\Omega$)")
    plt.legend()
    plt.savefig(f"results/tree_resistances/{tree_name_base}/resistance_plot.pdf", bbox_inches='tight')
    plt.clf()
    fig = plt.figure(figsize=(3,2.5))
    for depth_val in np.unique(depth):
        #pdb.set_trace()
        
        idx = np.where(depth == depth_val)[0]
        mean_0d_err = np.mean(np.abs(res_0d[idx] - res_3d[idx])/np.abs(res_3d[idx]))
        std_0d_err = np.std(np.abs(res_0d[idx] - res_3d[idx])/np.abs(res_3d[idx]))
        mean_0d_RI_err = np.mean(np.abs(res_0d_RI[idx] - res_3d[idx])/np.abs(res_3d[idx]))
        std_0d_RI_err = np.std(np.abs(res_0d_RI[idx] - res_3d[idx])/np.abs(res_3d[idx]))
        
        # plt.scatter(depth_val, mean_0d_err*100, color='orangered', s = 20, label='0D Error' if depth_val == 1 else "")
        # plt.scatter(depth_val, mean_0d_RI_err*100, color='royalblue', s = 20, label='0D RI Error' if depth_val == 1 else "")
        plt.errorbar(depth_val, mean_0d_RI_err, yerr=std_0d_RI_err, fmt='o', color='forestgreen', label='0D RI Error' if depth_val == 1 else "")
        plt.errorbar(depth_val, mean_0d_err, yerr=std_0d_err, fmt='o', color='orangered', label='0D Error' if depth_val == 1 else "")

    #plt.yscale("log")
    plt.xlabel("Depth ($\#$ bifurcations downstream)")
    plt.ylabel("Total Downstream Resistance \n Relative Error ($\%$)")
    plt.xscale("log")
    plt.legend(loc = "upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    if not os.path.exists(f"results/tree_resistances/{tree_name_base}"):
        os.makedirs(f"results/tree_resistances/{tree_name_base}")
    plt.savefig(f"results/tree_resistances/{tree_name_base}/resistance_error_plot.pdf", bbox_inches='tight')

    
    return

if __name__ == "__main__":
    plot_resistances("tree_3_flow_100")
    plot_resistances("tree_10_flow_100")
    plot_resistances("tree_40_flow_100")