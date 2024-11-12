from sv import *
import vtk
import os
import platform
import numpy as np
import pathplanning
import mesh_utils
import pickle

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

sampled_params_dict = load_dict("data/CCO_tree/CCO_sampled_params_dict")
geometry = "CCO"
set_type = "random"
if not os.path.exists('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions'):
    os.mkdir('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions')
if not os.path.exists('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions/{}'.format(geometry)):
    os.mkdir('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions/{}'.format(geometry))
if not os.path.exists('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions/{}/{}'.format(geometry, set_type)):
    os.mkdir('/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions/{}/{}'.format(geometry, set_type))

for geo_index in range(len(sampled_params_dict["daughter1_angle"])):

    print("Generating geometry for CCO {:03d}".format(geo_index))

    geo_dir = '/Users/natalia/Desktop/CCO_junctions/data/synthetic_junctions/{}/{}/CCO_{:03d}'.format(geometry, set_type, geo_index)
    if os.path.exists(geo_dir):
        print("Geometry directory {:03d} already exists".format(geo_index))
        continue
    else:
        os.mkdir(geo_dir)

    geo_params={"daughter1_angle": sampled_params_dict["daughter1_angle"][geo_index],
                                                        "daughter2_angle": sampled_params_dict["daughter2_angle"][geo_index],
                                                        "daughter1_area_ratio": sampled_params_dict["daughter1_area_ratio"][geo_index],
                                                        "daughter2_area_ratio": sampled_params_dict["daughter2_area_ratio"][geo_index]}

    contours, polydata = pathplanning.get_contours(geo_params=geo_params)
    print(geo_params)
    model, walls = mesh_utils.build_model(contours)
    mesher, msh = mesh_utils.get_mesh(model = model, contours = contours, walls = walls)
    mesh_utils.save_mesh(mesher = mesher, model = model, walls = walls, geo_dir = geo_dir)
    print("Geometry {:02d} saved".format(geo_index))








