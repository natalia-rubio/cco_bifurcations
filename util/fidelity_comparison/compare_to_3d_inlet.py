import sys
import vtk
import os
import numpy as np
import pdb
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.tools.get_bc_integrals import get_res_names
from util.tools.junction_proc import *
from util.tools.basic import compute_rmse
#from geo_processing import *
from util.tools.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle

def compare_to_3d_inlet(junction_mode = "standard",
                  tree_name = "tree_20_flow_100",
                  time_step = "700"):
        
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])

    reader_0d = read_geo(f"trees/zerod_output_cent/{junction_mode}/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_{time_step}.vtp").GetOutput()

    arrays_0d = get_all_arrays(reader_0d)
    arrays_3d = get_all_arrays(reader_3d)

    #pdb.set_trace()

    flow_0d = arrays_0d["flow"][arrays_3d["GlobalNodeId"] == 10]
    flow_3d = arrays_3d["Velocity"][arrays_3d["GlobalNodeId"] == 10]
    inlet_area = arrays_0d["CenterlineSectionArea"][arrays_3d["GlobalNodeId"] == 10]
    inlet_u = flow_3d / inlet_area
    inlet_radius = np.sqrt(inlet_area / np.pi)
    inlet_re = 1.06 * inlet_u * 2 * inlet_radius / 0.04

    area = arrays_3d["CenterlineSectionArea"]

    pressure_0d = arrays_0d["pressure"][arrays_3d["GlobalNodeId"] == 10]
    pressure_3d = arrays_3d["Pressure"][arrays_3d["GlobalNodeId"] == 10]

    flow_error_0d_rel = (flow_0d-flow_3d)/flow_3d
    flow_error_0d_tot = (flow_0d-flow_3d)

    print(f"                Flow error      (Relative):     {flow_error_0d_rel}")
    #print(f"Flow error      (Total):        {flow_error_0d_tot}")
   
    pressure_error_0d_rel = (pressure_0d - pressure_3d)/pressure_3d
    pressure_error_0d_tot = (pressure_0d - pressure_3d)/1333

    print(f"                Pressure error  (Relative):     {pressure_error_0d_rel}")
    print(f"                Pressure error  (Total mmHg):        {pressure_error_0d_tot}")

    error_dict = {
        "flow_error_0d_rel": flow_error_0d_rel,
        "flow_error_0d_tot": flow_error_0d_tot,
        "pressure_error_0d_rel": pressure_error_0d_rel,
        "pressure_error_0d_tot": pressure_error_0d_tot,
        "inlet_re": inlet_re,
    }
    #pdb.set_trace()
    return error_dict

if __name__ == "__main__":
    compare_to_3d_inlet(junction_mode = sys.argv[1], tree_name= sys.argv[2])
                  