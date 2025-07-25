from ctypes.wintypes import PDWORD
import json
from operator import index
import pdb
from pyexpat import model
import sys
import os
from tabnanny import check
from matplotlib.pyplot import plot
from numpy import add, mean
import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import add_0D_resistance, add_0D_RI_resistance, add_0D_solution_values, \
    get_input_file_junction_dict_master, add_geometry_values, add_downstream_resistance_values, \
        add_downstream_resistance_values_RI, add_solution_values, add_3D_resistance_total, add_0D_RI_resistance_total


def plot_resistances(tree_name):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    junction_dict_master = get_input_file_junction_dict_master(tree_name)

    time_step1 = "700"; time_step2 = "600"
    flow_mag_list = ["12", "25", "50", "100"]
    #flow_mag_list = [ "100"]
    add_geometry_values(junction_dict_master, tree_name, flow_mag = flow_mag_list[0], time_step = time_step1)
    for flow_mag in flow_mag_list:
        print(f"Processing flow magnitude: {flow_mag}")
        add_solution_values(junction_dict_master, f"{tree_name_base}_flow_{flow_mag}", flow_mag, time_step = time_step1)
        add_0D_solution_values(junction_dict_master, tree_name, flow_mag, time_step = time_step1, junction_mode="standard")
        add_0D_solution_values(junction_dict_master, tree_name, flow_mag, time_step = time_step1, junction_mode="RI")
        add_0D_solution_values(junction_dict_master, tree_name, flow_mag, time_step = time_step1, junction_mode="RRI")
        #pdb.set_trace()

    add_3D_resistance_total(junction_dict_master, tree_name, flow_mag_list, time_step = time_step1)
    add_0D_RI_resistance_total(junction_dict_master, tree_name, flow_mag_list, time_step = time_step1)

    
    add_downstream_resistance_values(junction_dict_master)
    add_0D_RI_resistance(junction_dict_master, tree_name_base)
    add_downstream_resistance_values_RI(junction_dict_master)
    
    depth = []
    res_3d = []
    res_0d = []
    res_0d_RI = []
    p_3d = []
    p_0d = []
    p_0d_RI = []
    p_0d_RRI = []
    re = []
    flow = []
    time_step = time_step1
    for junction_name, junction_dict in junction_dict_master.items():
        depth.append(junction_dict["depth2"])
        res_3d.append(junction_dict["3D_R_lin_total"])
        res_0d.append(((junction_dict["0D_geo_resistance"])**-1 + \
                        (junction_dict["0D_aux_geo_resistance"])**-1)**-1)
        #pdb.set_trace()
        # res_0d_RI.append(((junction_dict["0D_geo_resistance_RI"])**-1 + \
        #                 (junction_dict["0D_aux_geo_resistance_RI"])**-1)**-1)
        res_0d_RI.append(junction_dict["0D_RI_R_lin_total"])
        
        p_3d.append([junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] for flow_mag in flow_mag_list][-1])
        p_0d.append([junction_dict[f"0D_standard_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] for flow_mag in flow_mag_list][-1])
        p_0d_RI.append([junction_dict[f"0D_RI_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] for flow_mag in flow_mag_list][-1])
        p_0d_RRI.append([junction_dict[f"0D_RRI_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] for flow_mag in flow_mag_list][-1] )
        
        re.append(junction_dict[f"3D_junc_inlet_re_fm_{flow_mag}_ts_{time_step}"]/junction_dict_master["J0"][f"3D_junc_inlet_re_fm_{flow_mag}_ts_{time_step}"])
        flow.append(junction_dict[f"3D_junc_inlet_flow_fm_{flow_mag}_ts_{time_step}"]/junction_dict_master["J0"][f"3D_junc_inlet_flow_fm_{flow_mag}_ts_{time_step}"])
    
    depth = np.asarray(depth)
    res_3d = np.asarray(res_3d)
    res_0d = np.asarray(res_0d)
    res_0d_RI = np.asarray(res_0d_RI)
    p_3d = np.asarray(p_3d)
    p_0d = np.asarray(p_0d)
    p_0d_RI = np.asarray(p_0d_RI)
    p_0d_RRI = np.asarray(p_0d_RRI)
    re = np.asarray(re)
    flow = np.asarray(flow)
    
    
    #print(flow)
    # pdb.set_trace()
    # plt.scatter(depth, res_3d, color='black', s = 20, label='3D Resistance')
    # plt.scatter(depth, res_0d, color='tomato', s = 20, label='0D Resistance')
    # plt.scatter(depth, res_0d_RI, color='cornflowerblue', s = 20, label='0D RI Resistance')
    # plt.xlabel("Depth (max number of bifurcations downstream)")
    # plt.ylabel("Total Downstream Resistance ($\Omega$)")
    # plt.legend()
    # plt.savefig(f"results/tree_resistances/{tree_name_base}/resistance_plot.pdf", bbox_inches='tight')
    # plt.clf()
    
    fig, axs = plt.subplots(1,3, figsize=(8, 2))
    
    #fig = plt.figure(figsize=(3,2.5))
    for depth_val in np.unique(depth):
        #pdb.set_trace()
        
        idx = np.where(depth == depth_val)[0]
        mean_0d_err = np.mean(np.abs(res_0d[idx] - res_3d[idx])/np.abs(res_3d[idx]))*100
        std_0d_err = np.std(np.abs(res_0d[idx] - res_3d[idx])/np.abs(res_3d[idx]))*100
        mean_0d_RI_err = np.mean(np.abs(max((0,res_0d_RI[idx][0])) - res_3d[idx])/np.abs(res_3d[idx]))*100
        std_0d_RI_err = np.std(np.abs(max((0,res_0d_RI[idx][0]))  - res_3d[idx])/np.abs(res_3d[idx]))*100
        # mean_0d_err = np.mean(np.abs(res_0d[idx] - res_3d[idx]))
        # std_0d_err = np.std(np.abs(res_0d[idx] - res_3d[idx]))
        # mean_0d_RI_err = np.mean(np.abs(res_0d_RI[idx] - res_3d[idx]))
        # std_0d_RI_err = np.std(np.abs(res_0d_RI[idx] - res_3d[idx]))
        #pdb.set_trace()
        # plt.scatter(depth_val, mean_0d_err*100, color='tomato', s = 20, label='0D Error' if depth_val == 1 else "")
        # plt.scatter(depth_val, mean_0d_RI_err*100, color='cornflowerblue', s = 20, label='0D RI Error' if depth_val == 1 else "")
        axs[2].errorbar(depth_val, mean_0d_RI_err, yerr=std_0d_RI_err, fmt='o', color='cornflowerblue', label='0D RI Error' if depth_val == 1 else "")
        axs[2].errorbar(depth_val, mean_0d_err, yerr=std_0d_err, fmt='o', color='tomato', label='0D Error' if depth_val == 1 else "")

    #plt.yscale("log")
    #axs[2].set_xlabel("Depth ($\#$ bifurcations downstream)")
    axs[2].set_title("Downstream Resistance Error (\%)", fontsize=10)
    #axs[2].set_xscale("log")
    
    
    #axs[0].legend(loc = "upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    # if not os.path.exists(f"results/tree_resistances/{tree_name_base}"):
    #     os.makedirs(f"results/tree_resistances/{tree_name_base}")
    # plt.savefig(f"results/tree_resistances/{tree_name_base}/resistance_error_plot.pdf", bbox_inches='tight')

    # plt.clf()
    for depth_val in np.unique(depth):
        mean_re = np.mean(re[depth == depth_val])*100
        std_re = np.std(re[depth == depth_val])*100
        mean_flow = np.mean(flow[depth == depth_val])*100
        std_flow = np.std(flow[depth == depth_val])*100
        axs[1].errorbar(depth_val, mean_re, yerr=std_re, fmt='o', color='black', label='Re' if depth_val == 1 else "")
        axs[1].errorbar(depth_val, mean_flow, yerr=std_flow, fmt='o', color='slategrey', label='Flow' if depth_val == 1 else "")
        
        idx = np.where(depth == depth_val)[0]
        mean_0d_p_err = np.mean(np.abs(p_0d[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
        std_0d_p_err = np.std(np.abs(p_0d[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
        
        mean_0d_RI_p_err = np.mean(np.abs(p_0d_RI[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
        std_0d_RI_p_err = np.std(np.abs(p_0d_RI[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
        
        mean_0d_RRI_p_err = np.mean(np.abs(p_0d_RRI[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
        std_0d_RRI_p_err = np.std(np.abs(p_0d_RRI[idx] - p_3d[idx])/np.abs(p_3d[idx]))*100
    
        # idx = np.where(depth == depth_val)[0]
        # mean_0d_p_err = np.mean(np.abs(p_0d[idx] - p_3d[idx]))/1333
        # std_0d_p_err = np.std(np.abs(p_0d[idx] - p_3d[idx]))/1333
        
        # mean_0d_RI_p_err = np.mean(np.abs(p_0d_RI[idx] - p_3d[idx]))/1333
        # std_0d_RI_p_err = np.std(np.abs(p_0d_RI[idx] - p_3d[idx]))/1333
        
        # mean_0d_RRI_p_err = np.mean(np.abs(p_0d_RRI[idx] - p_3d[idx]))/1333
        # std_0d_RRI_p_err = np.std(np.abs(p_0d_RRI[idx] - p_3d[idx]))/1333
        print(f"Num bifurcations: {idx.size}")
        axs[0].errorbar(depth_val, mean_0d_RRI_p_err, yerr=std_0d_RRI_p_err, fmt='o', color='limegreen', label='0D RRI Error' if depth_val == 1 else "")
        axs[0].errorbar(depth_val, mean_0d_RI_p_err, yerr=std_0d_RI_p_err, fmt='o', color='cornflowerblue', label='0D RI Error' if depth_val == 1 else "")
        axs[0].errorbar(depth_val, mean_0d_p_err, yerr=std_0d_p_err, fmt='o', color='tomato', label='0D Error' if depth_val == 1 else "")
        # if depth_val == np.max(depth):
        #     pdb.set_trace()
        #print(f"Depth: {depth_val}, 0D STD: {std_0d_p_err}")

    #plt.yscale("log")
    #axs[0].set_xlabel("Depth ($\#$ bifurcations downstream)")
    axs[0].set_title("Pressure Error (mmHg)", fontsize=10)
    #axs[0].set_xscale("log")
    axs[0].legend(loc = "upper center", bbox_to_anchor=(1.8, 1.4), ncol=3, frameon=False)
    axs[1].set_xlabel("Depth ($\#$ bifurcations upstream)")
    axs[1].set_title("Relative Reynolds Number (\%)", fontsize=10)
    #axs[1].set_xscale("log")
    axs[1].legend(frameon=False)
    
    plt.subplots_adjust(wspace=0.3)
    if not os.path.exists(f"results/tree_resistances/{tree_name_base}"):
        os.makedirs(f"results/tree_resistances/{tree_name_base}")
    fig.savefig(f"results/tree_resistances/{tree_name_base}/pressure_error_plot.pdf", bbox_inches='tight')
    
    return

if __name__ == "__main__":
    # plot_resistances("tree_3_flow_100")
    # plot_resistances("tree_10_flow_100")
    plot_resistances("tree_40_flow_100")