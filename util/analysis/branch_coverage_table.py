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
junction_mode_list = ["RI_junctions_fit", "RI_nn_branch_fit", "RI_full_fit"]
color_list = ["orangered", "royalblue", "seagreen",]
hatch_list = ["xxxx", "....", ""]
color_list_light = ['mistyrose', "lightblue", "lightgreen"]

redo = True
if redo:
    error_dict = {}
    for tree_name in tree_list:
        print(f"Testing tree: {tree_name}")
        error_dict.update({tree_name: {}})

        for junction_mode in junction_mode_list:
            print(f"        Junction mode: {junction_mode}")
            err_list = []
            for flow_mag in flow_mag_list:
            
                print(f"    Flow magnitude: {flow_mag}")
                err_list.append(np.abs(test_junction_model_single(f"{tree_name}_flow_{flow_mag}", junction_mode)['pressure_error_0d_rel']))
                
            error_dict[tree_name][junction_mode] = np.mean(np.asarray(err_list))
    save_dict(error_dict, f"util/analysis/coverage_error_dict.pkl")
else:
    error_dict = load_dict(f"util/analysis/coverage_error_dict.pkl")
    
df = pd.DataFrame(error_dict).T

# Convert the DataFrame to a LaTeX table string
latex_table = df.to_latex()

print(latex_table)
