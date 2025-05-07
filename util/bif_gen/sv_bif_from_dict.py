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


tree_name = "tree_20"
set_type = "dict"
junction_dict = load_dict("trees/reports/{tree_name}/junction_dict.json".format(tree_name=tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions'):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions')
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}'.format(tree_name))
if not os.path.exists('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type)):
    os.mkdir('/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}'.format(tree_name, set_type))

for geo_index in range(len(junction_dict.keys())):
    junction_name = list(junction_dict.keys())[geo_index]
    geo_dir = '/Users/natalia/Desktop/cco_bifurcations/data/synthetic_junctions/{}/{}/CCO_{:03d}'.format(tree_name, set_type, geo_index)
    if os.path.exists(geo_dir):
        print("Geometry directory {:03d} already exists".format(geo_index))
        #continue
    else:
        os.mkdir(geo_dir)
    print("Generating geometry for CCO {:03d}".format(geo_index))
    #pdb.set_trace()

    geo_params={"daughter1_angle": junction_dict[junction_name]["3D_junc_outlet1_angle"],
                "daughter2_angle": junction_dict[junction_name]["3D_junc_outlet2_angle"],
                "daughter1_area_ratio": junction_dict[junction_name]["3D_branch1_outlet_area"]/junction_dict[junction_name]["3D_junc_inlet_area"],
                "total_daughter_area_ratio": (junction_dict[junction_name]["3D_branch2_outlet_area"]+
                                              junction_dict[junction_name]["3D_branch1_outlet_area"])/
                                              junction_dict[junction_name]["3D_junc_inlet_area"],
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








