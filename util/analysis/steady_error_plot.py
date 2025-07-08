from calendar import c
from math import e
import os
import pdb
import sys
import numpy as np

import pandas as pd
from pytest import mark
from sklearn import tree
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
import matplotlib.pyplot as plt
from util.zerod.test_junction_model_single import test_junction_model_single
from util.tools.basic import save_dict, load_dict
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True


#tree_list = ["tree_3", "tree_5", "tree_10", "tree_20", "tree_40"]
tree_list = ["tree_3",  "tree_10", "tree_40"]
num_outlets_list = [3, 10, 40]
flow_mag_list = ["12", "25", "50", "100"]
junction_mode_list = ["standard", "RI", "RRI"]
color_list = ["orangered", "royalblue", "seagreen",]
hatch_list = ["xxxx", "....", ""]
color_list_light = ['mistyrose', "lightblue", "lightgreen"]

redo = True
if redo:
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
    save_dict(error_dict, f"util/analysis/steady_error_dict.pkl")
else:
    error_dict = load_dict(f"util/analysis/steady_error_dict.pkl")
pdb.set_trace()
# plt.clf()
# for i, flow_mag in enumerate(flow_mag_list):
#     RR_error = []; RI_error = []; standard_error = []; re_list = []; flow_list = []
#     for tree_name in tree_list:
#         if error_dict[tree_name][flow_mag]['RR'] is None:
#             continue
#         RR_error.append(np.abs(error_dict[tree_name][flow_mag]['RR']['pressure_error_0d_tot']))
#         RI_error.append(np.abs(error_dict[tree_name][flow_mag]['RI']['pressure_error_0d_tot']))
#         standard_error.append(np.abs(error_dict[tree_name][flow_mag]['standard']['pressure_error_0d_tot']))
#         re_list.append(error_dict[tree_name][flow_mag]['RR']['inlet_re'])
#     re_avg = sum(re_list) / len(re_list)
#     plt.plot(num_outlets_list, RR_error, label=f"{re_avg}", markersize=7, marker='o', color=color_list[i], linestyle='-', linewidth=2)
#     plt.plot(num_outlets_list, RI_error, label=f"RI", markersize=7, marker='s', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle=':', linewidth=2)
#     plt.plot(num_outlets_list, standard_error, label=f"standard", markersize=7, marker='P', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle='--', linewidth=2)
#     print(f"RR error list: {RR_error}")
#     print(f"Standard error list: {standard_error}")
# plt.xlabel("Tree Size (Number of Outlets)")
# plt.ylabel("Inlet Pressure Error (mmHg)")
# plt.yscale("symlog")
# plt.xscale("log")
# plt.legend(bbox_to_anchor=(0.5, 2), loc='upper center')
# plt.savefig(f"results/steady_error_plot.pdf", bbox_inches='tight')

plt.clf()
for i, tree_name in enumerate(tree_list):

    RR_error = []; RI_error = []; standard_error = []; flow_list = []; re_list = []
    for j, flow_mag in enumerate(flow_mag_list):
        if error_dict[tree_name][flow_mag]['RRI'] is None:
            continue
        RR_error.append(np.abs(error_dict[tree_name][flow_mag]['RRI']['pressure_error_0d_rel']))
        RI_error.append(np.abs(error_dict[tree_name][flow_mag]['RI']['pressure_error_0d_rel']))
        standard_error.append(np.abs(error_dict[tree_name][flow_mag]['standard']['pressure_error_0d_rel']))
        flow_list.append(error_dict[tree_name][flow_mag]['RRI']['inflow'])
        re_list.append(error_dict[tree_name][flow_mag]['RRI']['inlet_re'])

    plt.plot(re_list, RR_error, markersize=7, marker='P', color=color_list[i], linestyle='-', linewidth=2, label=tree_name)
    plt.plot(re_list, RI_error, label=f"RI", markersize=7, marker='s', markerfacecolor='none', markeredgecolor=color_list[i], color = color_list[i], linestyle=':', linewidth=2)
    plt.plot(re_list, standard_error, label=f"standard", markersize=7, marker='o',color = color_list[i], linestyle='--', linewidth=2)
    print(f"RR error list: {RR_error}")
    print(f"Standard error list: {standard_error}")
plt.xlabel("Inlet Reynolds Number")
plt.ylabel("Inlet Pressure Error (mmHg)")
plt.xscale("log")
plt.legend(bbox_to_anchor=(0.0, 1.4), loc='upper center', ncols = 3)
plt.savefig(f"results/steady_error_plot_re.pdf", bbox_inches='tight')

# ------------------- Relative Error Plot -------------------
plt.clf()
fig, ax = plt.subplots()
fig.set_size_inches(8, 4)

# Bar width and offsets
bar_width = 0.20
indices = np.arange(len(tree_list))

# Plotting nested bars
num_junction_modes = len(junction_mode_list)
num_flows = len(flow_mag_list)
re_list = []
tick_list = []

for i, middle in enumerate(flow_mag_list):


    for j, inner in enumerate(junction_mode_list):
        offset = (i * (num_junction_modes+1) + j) * bar_width
        heights = [np.abs(error_dict[outer][middle][inner]["pressure_error_0d_rel"][0])*100 for outer in tree_list]
        bar_positions = indices*(((num_junction_modes+1)*len(flow_mag_list)+2)*bar_width) + offset - (bar_width * num_junction_modes * len(flow_mag_list)+1) / 2
        if i == j == 0:
            start_pos = bar_positions[0]
        ax.bar(bar_positions, heights, width=bar_width, color = color_list_light[j], edgecolor = color_list[j], hatch = hatch_list[j],label= inner if i == 0 else "")
        
for t, tree_name in enumerate(tree_list):     
    for i, middle in enumerate(flow_mag_list):
        tick_list.append((i * (num_junction_modes+1)) * bar_width + (t * (num_flows * (num_junction_modes+1) + 2) * bar_width) + start_pos + bar_width)
        re_list.append(f"{round(int(error_dict[tree_list[t]][middle]['RRI']['inlet_re'][0]), -2)}")

# Adjustments for readability
ax.tick_params(axis='x', labelsize=9, bottom=False, top=False, labelbottom=True)
tick_list.insert(0, tick_list[0] - 4*bar_width)
re_list.insert(0, "Inlet Re:")
ax.set_xticks(tick_list)
print(tick_list)
ax.set_xticklabels(re_list)
tick_list.pop(0)

sec = ax.secondary_xaxis(location=-0.15)
tl1 = tick_list[2::4]
tl2 = tick_list[1::4]
sec.set_xticks([(tl1[i] + tl2[i])/2 for i in range(len(tl1))])
sec.set_xticklabels(["2 Junction Tree", "10 Junction Tree", "40 Junction Tree"],  style='italic')
for label in sec.get_xticklabels():
    label.set_fontstyle('italic')
sec.tick_params('x', length=0)
sec.spines['bottom'].set_linewidth(0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol =3 , frameon=False)

# Labels and title
ax.set_ylabel('Inlet Pressure (\%)')

# Show plot
plt.tight_layout()
plt.savefig(f"results/steady_error_bars_re.pdf", bbox_inches='tight')

# ------------------- Total Error Plot -------------------

plt.clf()
fig, ax = plt.subplots()
fig.set_size_inches(12, 4)

# Bar width and offsets
bar_width = 0.20
indices = np.arange(len(tree_list))

# Plotting nested bars
num_junction_modes = len(junction_mode_list)
num_flows = len(flow_mag_list)
re_list = []
tick_list = []

for i, middle in enumerate(flow_mag_list):


    for j, inner in enumerate(junction_mode_list):
        offset = (i * (num_junction_modes+1) + j) * bar_width
        heights = [np.abs(error_dict[outer][middle][inner]["pressure_error_0d_tot"][0])*100 for outer in tree_list]
        bar_positions = indices*(((num_junction_modes+1)*len(flow_mag_list)+2)*bar_width) + offset - (bar_width * num_junction_modes * len(flow_mag_list)+1) / 2
        if i == j == 0:
            start_pos = bar_positions[0]
        ax.bar(bar_positions, heights, width=bar_width, color = color_list_light[j], edgecolor = color_list[j], hatch = hatch_list[j],label= inner if i == 0 else "")
        
for t, tree_name in enumerate(tree_list):     
    for i, middle in enumerate(flow_mag_list):
        tick_list.append((i * (num_junction_modes+1)) * bar_width + (t * (num_flows * (num_junction_modes+1) + 2) * bar_width) + start_pos + bar_width)
        re_list.append(f"Re={round(int(error_dict[tree_list[t]][middle]['RRI']['inlet_re'][0]), -2)}")

# Adjustments for readability
ax.tick_params(axis='x', bottom=False, top=False, labelbottom=True)
ax.set_xticks(tick_list)
print(tick_list)
ax.set_xticklabels(re_list)

sec = ax.secondary_xaxis(location=-0.1)
tl1 = tick_list[2::4]
tl2 = tick_list[1::4]
sec.set_xticks([(tl1[i] + tl2[i])/2 for i in range(len(tl1))])
sec.set_xticklabels(["2 Junction Tree", "10 Junction Tree", "40 Junction Tree"],  style='italic')
for label in sec.get_xticklabels():
    label.set_fontstyle('italic')
sec.tick_params('x', length=0)
sec.spines['bottom'].set_linewidth(0)
ax.legend(loc='upper left', frameon=False)

# Labels and title
ax.set_ylabel('Inlet Pressure Error (mmHg)')

plt.tight_layout()
plt.savefig(f"results/steady_error_total_bars_re.pdf", bbox_inches='tight')