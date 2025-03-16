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

def compare_to_3d(junction_mode = "standard",
                  tree_name = "tree_80",
                  flow_amp = "full",
                  gen_mode = "sv"):

    reader_0d = read_geo(f"trees/zerod_output_cent_{gen_mode}_{junction_mode}/{tree_name}_{flow_amp}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_{flow_amp}.vtp").GetOutput()
    #was 300

    arrays_0d_0d = get_all_arrays(reader_0d)
    arrays_3d = get_all_arrays(reader_3d)

    flow_0d = arrays_0d_0d["flow"]
    flow_3d = arrays_3d["Velocity"]

    area = arrays_3d["CenterlineSectionArea"]

    pressure_0d = arrays_0d_0d["pressure"]
    pressure_3d = arrays_3d["Pressure"]

    flow_error_0d = (flow_0d-flow_3d)/flow_3d
    flow_error_0d[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Flow error: {compute_rmse(flow_error_0d,0*flow_error_0d)}")

   
    pressure_error_0d = (pressure_0d - pressure_3d)/pressure_3d
    pressure_error_0d[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Pressure error: {compute_rmse(pressure_error_0d,0*pressure_error_0d)}")


    array = n2v(pressure_error_0d)
    array.SetName("pressure_error_0d")
    array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
    reader_3d.GetPointData().AddArray(array) 

    pressure_0d[arrays_3d["BifurcationId"] >= 0] = 0
    array = n2v(pressure_0d)
    array.SetName("pressure_0d")
    array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
    reader_3d.GetPointData().AddArray(array) 

    array = n2v(flow_error_0d)
    array.SetName("flow_error_0d")
    array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
    reader_3d.GetPointData().AddArray(array) 

    flow_0d[arrays_3d["BifurcationId"] >= 0] = 0
    array = n2v(flow_0d)
    array.SetName("flow_0d")
    array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
    reader_3d.GetPointData().AddArray(array)

    write_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_{junction_mode}.vtp", reader_3d)
    return

if __name__ == "__main__":
    compare_to_3d(junction_mode = sys.argv[1])
                  