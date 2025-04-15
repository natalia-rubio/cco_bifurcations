from sv import *
import vtk
import os
import platform
import numpy as np
import pickle
import pdb


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_contours(geo_params):
    branch_dict = load_dict(geo_params["geo_dir"] + "/tree_dict.pkl")
    contour_list = []
    contour_polydata_list = []
    branch_names = list(branch_dict.keys())
    branch_names.sort()
    print(branch_names)
    for branch_name in branch_names:
        starts = branch_dict[branch_name]["starts"]
        ends = branch_dict[branch_name]["ends"]
        locs = branch_dict[branch_name]["locs"]
        norms = branch_dict[branch_name]["norms"]
        radii = branch_dict[branch_name]["radii"]

        path = pathplanning.Path()
        print("Branch name: " + branch_name)
        print(locs)
        print(radii)
        path.set_control_points(locs)
        curve_points = path.get_curve_points()
        contour_list_i = []
        contour_polydata_list_i = []
        for i in range(len(locs)):
            #pdb.set_trace()
            #contour_list_i.append(segmentation.Circle(radii[i], center = locs[i], normal = norms[i]))
            contour_list_i.append(segmentation.Circle(radii[i], center = locs[i], normal = path.get_curve_tangent(curve_points.index(locs[i]))))
            contour_polydata_list_i.append(contour_list_i[-1].get_polydata())

        contour_list.append(contour_list_i)
        contour_polydata_list.append(contour_polydata_list_i)
    return contour_list, contour_polydata_list