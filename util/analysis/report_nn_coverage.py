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
from util.zerod.get_nn_coverage import get_nn_coverage
from util.tools.basic import save_dict, load_dict
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True


#tree_list = ["tree_3", "tree_5", "tree_10", "tree_20", "tree_40"]
tree_list = ["tree_3_flow_12",  "tree_10_flow_12", "tree_40_flow_12"]

for tree_name in tree_list:
    print(f"Testing tree: {tree_name}")
    covg = get_nn_coverage(tree_name)
    print(f"Coverage for {tree_name} is {covg[1]/sum(covg)}")
        
