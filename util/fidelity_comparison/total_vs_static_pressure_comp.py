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

tree_name = "tree_80"
reader_0d_standard = read_geo(f"trees/zerod_output_cent_standard/{tree_name}/centerline_sol.vtp").GetOutput()
reader_0d_rri = read_geo(f"trees/zerod_output_cent_RRI/{tree_name}/centerline_sol.vtp").GetOutput()
reader_3d = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_300.vtp").GetOutput()
reader_3d_200 = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_200.vtp").GetOutput()

arrays_0d_standard = get_all_arrays(reader_0d_standard)
#arrays_0d_rri = get_all_arrays(reader_0d_rri)
arrays_3d = get_all_arrays(reader_3d)
#arrays_3d_200 = get_all_arrays(reader_3d_200)

flow_3d = arrays_3d["Velocity"]#/arrays_3d["area"]
area_3d = arrays_3d["area"]
avg_velocity_3d = flow_3d/area_3d
dynamic_pressure_3d = 0.5*1.06*avg_velocity_3d**2
flow_standard = arrays_0d_standard["flow"]
flow_error_standard = (arrays_0d_standard["flow"]-flow_3d)/flow_3d
flow_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Flow error standard: {compute_rmse(flow_error_standard,0*flow_error_standard)}")

pressure_3d = arrays_3d["Pressure"]
pressure_standard = arrays_0d_standard["pressure"]
pressure_error_standard = (pressure_standard - pressure_3d)/pressure_3d
pressure_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Pressure error standard: {compute_rmse(pressure_error_standard,0*pressure_error_standard)}")

area = arrays_0d_standard["CenterlineSectionArea"]
avg_velocity_standard = flow_standard/area
dynamic_pressure_standard = 0.5*1.06*avg_velocity_standard**2
min_pressure_ind = np.argmin(pressure_standard)
total_pressure_offset = dynamic_pressure_standard[min_pressure_ind]
total_pressure_standard = pressure_standard - (dynamic_pressure_standard) #- total_pressure_offset)
total_pressure_error_standard = (total_pressure_standard - pressure_3d)/pressure_3d
total_pressure_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Total pressure error standard: {compute_rmse(total_pressure_error_standard,0*total_pressure_error_standard)}")


array = n2v(dynamic_pressure_standard)
array.SetName("standard_dynamic_pressure")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(dynamic_pressure_3d)
array.SetName("dynamic_pressure")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(pressure_error_standard)
array.SetName("standard_pressure_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(flow_error_standard)
array.SetName("standard_flow_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(total_pressure_error_standard)
array.SetName("total_pressure_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 


re = 1.06*(flow_3d/arrays_3d["area"])*(2*np.sqrt(arrays_3d["area"]/np.pi))/0.04
re[arrays_3d["BifurcationId"] >= 0] = 0
array = n2v(re)
array.SetName("re")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

re = arrays_3d["area"]
re[arrays_3d["BifurcationId"] >= 0] = 0
array = n2v(re)
array.SetName("Area")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 


write_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_total_pressure_error.vtp", reader_3d)

minid = np.argmin(arrays_3d["GlobalNodeId"])
inlet_area = arrays_3d["area"][minid]
pdb.set_trace()
