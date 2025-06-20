from calendar import c
from math import e
import os
import pdb
import sys
import numpy as np

from pytest import mark
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
import matplotlib.pyplot as plt
from util.zerod.test_junction_model_single import test_junction_model_single
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True

tree_list = ["tree_3", "tree_20",]
num_outlets_list = [3,20,]
flow_list = ["25", "50", "100", "150"]
junction_mode_list = ["standard", "RR"]
color_list = ["royalblue", "seagreen", "darkorange", "crimson"]


error_dict = {}
for tree_name in tree_list:
    print(f"Testing tree: {tree_name}")
    error_dict.update({tree_name: {}})
    for flow_mag in flow_list:
        print(f"    Flow magnitude: {flow_mag}")
        error_dict[tree_name].update({flow_mag: {}})
        for junction_mode in junction_mode_list:
            print(f"        Junction mode: {junction_mode}")
            try:
                sample_error_dict = test_junction_model_single(f"{tree_name}_flow_{flow_mag}", junction_mode)
                error_dict[tree_name][flow_mag].update({junction_mode: sample_error_dict})
            except:
                continue
            
            #pdb.set_trace()

#plt.figure(figsize=(10, 6))
plt.clf()
for i, flow_mag in enumerate(flow_list):
    RR_error = []; standard_error = []; re_list = []
    for tree_name in tree_list:
        RR_error.append(np.abs(error_dict[tree_name][flow_mag]['RR']['pressure_error_0d_tot']))
        standard_error.append(np.abs(error_dict[tree_name][flow_mag]['standard']['pressure_error_0d_tot']))
        re_list.append(error_dict[tree_name][flow_mag]['RR']['inlet_re'])
    re_avg = sum(re_list) / len(re_list)
    plt.plot(num_outlets_list, RR_error, label=f"{re_avg}", markersize=7, marker='o', color=color_list[i], linestyle='-', linewidth=2)
    plt.plot(num_outlets_list, standard_error, label=f"standard", markersize=7, marker='o', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle='--', linewidth=2)
    
plt.xlabel("Tree Size (Number of Outlets)")
plt.ylabel("Inlet Pressure Error (mmHg)")
#plt.yscale("symlog")
plt.legend(bbox_to_anchor=(0.5, 2), loc='upper center')
plt.savefig(f"results/steady_error_plot.pdf", bbox_inches='tight')