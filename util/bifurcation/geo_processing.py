
import os
import sys
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
import random
import copy
import pickle
#from get_avg_sol import *

np.seterr(all='raise')
np.set_printoptions(threshold=sys.maxsize)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res

def get_all_arrays(geo):
    # collect all arrays
    #cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())
    return point_data

def read_centerline(fpath_1d):
    fname = fpath_1d

    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def load_centerline_data(fpath_1d):
    cent = read_centerline(fpath_1d).GetOutput()
    cent_array = get_all_arrays(cent)

    #Extract Geometry ----------------------------------------------------
    pt_id = cent_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = cent_array["BranchId"].astype(int)
    junction_id = cent_array["BifurcationId"].astype(int)
    axial_distance = cent_array["Path"]  # distance along centerline
    area = cent_array["CenterlineSectionArea"]
    direction = cent_array["CenterlineSectionNormal"]  # vector normal direction
    direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
    direction = np.transpose(np.divide(direction,direction_norm))  # normalized direction vector
    angle1 = direction[0,:].reshape(-1,)
    angle2 = direction[1,:].reshape(-1,)
    angle3 = direction[2,:].reshape(-1,)
    return pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3
