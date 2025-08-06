import json
import pdb
import sys
import os

from matplotlib import lines
import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.extract_true_junctions_helpers import *
from scipy.interpolate import interp1d

if __name__ == "__main__":
    
    tree_name = sys.argv[1]
    junction_type = sys.argv[2]
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    
    geo_name = tree_name_base
    if geo_name  == "tree_20":   
        inlet_rad = 0.28
    elif geo_name == "tree_5":
        inlet_rad = 0.279
    elif geo_name == "tree_3":
        inlet_rad = 0.20
    elif geo_name == "tree_10":
        inlet_rad = 0.204 #0.176
    elif geo_name == "tree_40":
        inlet_rad = 0.252
        
        
    u_max_des = 0.04*5000/(1.06*inlet_rad*2)
    
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    t = []
    Q = []


    with open(f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/inflow.flow", 'r') as file:
        for line in file:
            # Split the line by whitespace
            columns = line.split(" ")
            #pdb.set_trace()
            if len(columns) == 2:
                # Append data from each column to the corresponding list
                t.append(float(columns[0]))
                Q.append(-1*float(columns[1]))
            else:
                print(f"Skipping line with unexpected format: {line}")
                
    Q_final = [-QQ * u_max_des / np.max(Q) for QQ in Q]
        
    num_time_steps = 1000
    dt = 1/num_time_steps
    t_fine = np.linspace(0, t[-2], num_time_steps)
    f = interp1d(t[:-1], Q_final[:-1], kind='linear', fill_value="extrapolate")
    q = f(t_fine)
    
    flow = f"{int(num_time_steps)}    32\n"
    # updated!
    for i in range(3*len(t_fine)):
        flow = flow + "%1.5f    %1.3f\n" %(i*dt, q[i%num_time_steps])
    if not os.path.exists(f"trees/geo_files/{tree_name_base}/{tree_name_base}_real"):
        os.mkdir(f"trees/geo_files/{tree_name_base}/{tree_name_base}_real")
    f = open(f"trees/geo_files/{tree_name_base}/{tree_name_base}_real" + f"/inflow_svFSI_flow_real.flow", "w")
    f.write(flow)
    f.close()
    pdb.set_trace()
    