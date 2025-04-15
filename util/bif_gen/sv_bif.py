from sv import *
import vtk
import sys
sys.path.append('/Users/natalia/Desktop/cco_bifurcations')
import os
import platform
import numpy as np
import pathplanning
import util.bif_gen.meshing as meshing
from util.bif_gen.plan_junction import plan_junction
import pickle
import pdb

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict


tree_name = "ideal_fix_areas"
set_type = "random"
sampled_params_dict = load_dict("data/sampled_params_dict/{tree_name}_sampled_params_dict".format(tree_name=tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions'):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions')
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type))

for geo_index in range(len(sampled_params_dict["daughter1_angle"])):
    
    geo_dir = '/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}/CCO_{:03d}'.format(tree_name, set_type, geo_index)
    if os.path.exists(geo_dir):
        print("Geometry directory {:03d} already exists".format(geo_index))
        continue
    else:
        os.mkdir(geo_dir)
    print("Generating geometry for CCO {:03d}".format(geo_index))
    #pdb.set_trace()
    geo_params={"daughter1_angle": sampled_params_dict["daughter1_angle"][geo_index],
                "daughter2_angle": sampled_params_dict["daughter2_angle"][geo_index],
                "daughter1_area_ratio": max([sampled_params_dict["daughter1_area_ratio"][geo_index], sampled_params_dict["total_daughter_area_ratio"][geo_index]-sampled_params_dict["daughter1_area_ratio"][geo_index]]),
                "total_daughter_area_ratio": sampled_params_dict["total_daughter_area_ratio"][geo_index],
                "geo_dir": geo_dir,}
    print(geo_params)
    
    
    try:
        plan_junction(geo_params=geo_params)
        contours, polydata = pathplanning.get_contours(geo_params=geo_params)
        print(geo_params)
        #pdb.set_trace()
        model, walls = meshing.build_model(contours)
        mesher, msh, cap_dict = meshing.get_mesh(model = model, contours = contours, walls = walls, edge_size = 0.8)
        meshing.save_mesh(mesher = mesher, model = model, walls = walls, cap_dict=cap_dict, geo_dir = geo_dir)

    except:
        print("Geometry {:02d} failed".format(geo_index))
        pdb.set_trace()
        continue


    print("Geometry {:02d} saved".format(geo_index))
    #pdb.set_trace()








