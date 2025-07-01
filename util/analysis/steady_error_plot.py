from calendar import c
from math import e
import os
import pdb
import sys
import numpy as np

import pandas as pd
from pytest import mark
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
import matplotlib.pyplot as plt
from util.zerod.test_junction_model_single import test_junction_model_single
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True

#tree_list = ["tree_3", "tree_5", "tree_10", "tree_20", "tree_40"]
tree_list = ["tree_3",  "tree_10", "tree_40"]
num_outlets_list = [3, 10, 40]
flow_mag_list = ["25", "50", "100"]
junction_mode_list = ["standard", "RR"]
color_list = ["royalblue", "seagreen", "darkorange", "crimson"]


error_dict = {}
for tree_name in tree_list:
    print(f"Testing tree: {tree_name}")
    error_dict.update({tree_name: {}})
    for flow_mag in flow_mag_list:
        print(f"    Flow magnitude: {flow_mag}")
        error_dict[tree_name].update({flow_mag: {}})
        #try:
        for junction_mode in junction_mode_list:
            print(f"        Junction mode: {junction_mode}")
            sample_error_dict = test_junction_model_single(f"{tree_name}_flow_{flow_mag}", junction_mode)
            error_dict[tree_name][flow_mag].update({junction_mode: sample_error_dict})
        # except:
        #     error_dict[tree_name][flow_mag].update({"standard": None})
        #     error_dict[tree_name][flow_mag].update({"RR": None})
        #     continue

            
            #pdb.set_trace()

#plt.figure(figsize=(10, 6))

plt.clf()
for i, flow_mag in enumerate(flow_mag_list):
    RR_error = []; standard_error = []; re_list = []; flow_list = []
    for tree_name in tree_list:
        if error_dict[tree_name][flow_mag]['RR'] is None:
            continue
        RR_error.append(np.abs(error_dict[tree_name][flow_mag]['RR']['pressure_error_0d_tot']))
        standard_error.append(np.abs(error_dict[tree_name][flow_mag]['standard']['pressure_error_0d_tot']))
        re_list.append(error_dict[tree_name][flow_mag]['RR']['inlet_re'])
    re_avg = sum(re_list) / len(re_list)
    plt.plot(num_outlets_list, RR_error, label=f"{re_avg}", markersize=7, marker='o', color=color_list[i], linestyle='-', linewidth=2)
    plt.plot(num_outlets_list, standard_error, label=f"standard", markersize=7, marker='o', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle='--', linewidth=2)
    print(f"RR error list: {RR_error}")
    print(f"Standard error list: {standard_error}")
plt.xlabel("Tree Size (Number of Outlets)")
plt.ylabel("Inlet Pressure Error (mmHg)")
plt.yscale("symlog")
plt.xscale("log")
plt.legend(bbox_to_anchor=(0.5, 2), loc='upper center')
plt.savefig(f"results/steady_error_plot.pdf", bbox_inches='tight')

plt.clf()
for i, tree_name in enumerate(tree_list):

    RR_error = []; standard_error = []; flow_list = []; re_list = []
    for j, flow_mag in enumerate(flow_mag_list):
        if error_dict[tree_name][flow_mag]['RR'] is None:
            continue
        RR_error.append(np.abs(error_dict[tree_name][flow_mag]['RR']['pressure_error_0d_rel']))
        standard_error.append(np.abs(error_dict[tree_name][flow_mag]['standard']['pressure_error_0d_rel']))
        flow_list.append(error_dict[tree_name][flow_mag]['RR']['inflow'])
        re_list.append(error_dict[tree_name][flow_mag]['RR']['inlet_re'])

    #re_avg = sum(re_list) / len(re_list)
    plt.plot(re_list, RR_error, markersize=7, marker='o', color=color_list[i], linestyle='-', linewidth=2, label=tree_name)
    plt.plot(re_list, standard_error, label=f"standard", markersize=7, marker='o', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle='--', linewidth=2)
    print(f"RR error list: {RR_error}")
    print(f"Standard error list: {standard_error}")
plt.xlabel("Inlet Reynolds Number")
plt.ylabel("Inlet Pressure Error (mmHg)")
#plt.yscale("symlog")
plt.xscale("log")
plt.legend(bbox_to_anchor=(0.5, 2), loc='upper center')
plt.savefig(f"results/steady_error_plot_re.pdf", bbox_inches='tight')

plt.clf()
fig, ax = plt.subplots()

# Bar width and offsets
bar_width = 0.20
indices = np.arange(len(outer_categories))

# Plotting nested bars
for i, middle in enumerate(middle_categories):
    for j, inner in enumerate(range(inner_categories_count)):
        offset = (i * inner_categories_count + inner) * bar_width
        heights = [values[outer][middle][inner] for outer in outer_categories]
        bar_positions = indices + offset - (bar_width * inner_categories_count * len(middle_categories)) / 2
        ax.bar(bar_positions, heights, width=bar_width, label=f'{middle} - Type {inner + 1}' if j == 0 else "")

# Adjustments for readability
ax.set_xticks(indices)
ax.set_xticklabels(outer_categories)
ax.legend(title="Legend", bbox_to_anchor=(1.05, 1), loc='upper left')

# Labels and title
ax.set_xlabel('Outer Categories')
ax.set_ylabel('Values')
ax.set_title('Nested Bar Chart with 3 Levels')

# Show plot
plt.tight_layout()
plt.show()
plt.savefig(f"results/steady_error_bars_re.pdf", bbox_inches='tight')