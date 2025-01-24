
import vtk
import sys
sys.path.append('/Users/natalia/Desktop/cco_bifurcations')
import os
import platform
import numpy as np
import pickle
import pdb

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict


tree_name = "Jan_CCO_80"
set_type = "random"
sampled_params_dict = load_dict(f"data/sampled_params_dict/{tree_name}_sampled_params_dict".format(tree_name=tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions'):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions')
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type))

for geo_index in range(len(sampled_params_dict["daughter1_angle"])):


    geo_dir = '/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}/CCO_{:03d}/mesh-complete'.format(tree_name, set_type, geo_index)


    geo_params={"daughter1_angle": sampled_params_dict["daughter1_angle"][geo_index],
                "daughter2_angle": sampled_params_dict["daughter2_angle"][geo_index],
                "daughter1_area_ratio": sampled_params_dict["daughter1_area_ratio"][geo_index],
                "daughter2_area_ratio": sampled_params_dict["daughter1_area_ratio"][geo_index]*sampled_params_dict["daughter12_area_ratio"][geo_index]}
    
    if geo_params["daughter2_angle"] < 1.4 and geo_params["daughter2_angle"] > 1.1:
        print(f"CCO {geo_index} has daughter 2 angle between 1.1 and 1.4.")
        if os.path.exists(geo_dir):
            print(f"..... CCO {geo_index} has a mesh.")
        else:
            print(f"..... CCO {geo_index} does not have a mesh.")
        continue