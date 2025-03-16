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
flow_amp = "full"
gen_mode = "sv"

reader_0d_standard = read_geo(f"trees/zerod_output_cent_{gen_mode}_standard/{tree_name}_{flow_amp}/centerline_sol.vtp").GetOutput()
reader_0d_lookup_branch = read_geo(f"trees/zerod_output_cent_{gen_mode}_lookup_branch/{tree_name}_{flow_amp}/centerline_sol.vtp").GetOutput()
reader_3d = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_{flow_amp}.vtp").GetOutput()
#was 300

arrays_0d_standard = get_all_arrays(reader_0d_standard)
arrays_0d_lookup_branch = get_all_arrays(reader_0d_lookup_branch)
arrays_3d = get_all_arrays(reader_3d)

flow_lookup_branch = arrays_0d_lookup_branch["flow"]
flow_standard = arrays_0d_standard["flow"]
flow_3d = arrays_3d["Velocity"]

area = arrays_3d["CenterlineSectionArea"]

pressure_lookup_branch = arrays_0d_lookup_branch["pressure"]
pressure_standard = arrays_0d_standard["pressure"]
pressure_3d = arrays_3d["Pressure"]

energy_lookup_branch = pressure_lookup_branch + 0.5*1.06*flow_lookup_branch**2/area
energy_standard = pressure_standard + 0.5*1.06*flow_standard**2/area
energy_3d = pressure_3d + 0.5*1.06*flow_3d**2/area


flow_error_lookup_branch = (flow_lookup_branch-flow_3d )/flow_3d
flow_error_lookup_branch[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Flow error lookup_branch: {compute_rmse(flow_error_lookup_branch,0*flow_error_lookup_branch)}")
flow_error_standard = (flow_standard-flow_3d)/flow_3d
flow_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Flow error standard: {compute_rmse(flow_error_standard,0*flow_error_standard)}")

pressure_error_lookup_branch = (pressure_lookup_branch - pressure_3d)/pressure_3d
pressure_error_lookup_branch[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Pressure error lookup_branch: {compute_rmse(pressure_error_lookup_branch,0*pressure_error_lookup_branch)}")
pressure_error_standard = (pressure_standard - pressure_3d)/pressure_3d
pressure_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Pressure error standard: {compute_rmse(pressure_error_standard,0*pressure_error_standard)}")

energy_error_lookup_branch = (energy_lookup_branch - energy_3d)/energy_3d
energy_error_lookup_branch[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Energy error lookup_branch: {compute_rmse(energy_error_lookup_branch,0*energy_error_lookup_branch)}")
energy_error_standard = (energy_standard - energy_3d)/energy_3d
energy_error_standard[arrays_3d["BifurcationId"] >= 0] = 0
print(f"Energy error standard: {compute_rmse(energy_error_standard,0*energy_error_standard)}")

array = n2v(pressure_error_standard)
array.SetName("standard_pressure_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(flow_error_standard)
array.SetName("standard_flow_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(energy_error_standard)
array.SetName("standard_energy_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array)

array = n2v(pressure_error_lookup_branch)
array.SetName("lookup_branch_pressure_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(flow_error_lookup_branch)
array.SetName("lookup_branch_flow_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(energy_error_lookup_branch)
array.SetName("lookup_branch_energy_error")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array)

array = n2v(flow_lookup_branch)
array.SetName("lookup_branch_flow")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

array = n2v(pressure_lookup_branch)
array.SetName("lookup_branch_pressure")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array) 

energy_lookup_branch[arrays_3d["BifurcationId"] >= 0] = 0
array = n2v(energy_lookup_branch)
array.SetName("lookup_branch_energy")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array)

energy_standard[arrays_3d["BifurcationId"] >= 0] = 0
array = n2v(energy_standard)
array.SetName("standard_energy")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array)

energy_3d[arrays_3d["BifurcationId"] >= 0] = 0
array = n2v(energy_3d)
array.SetName("energy")
array.SetNumberOfValues(reader_3d.GetNumberOfPoints())
reader_3d.GetPointData().AddArray(array)

re = 1.06*(flow_3d/arrays_3d["CenterlineSectionArea"])*(2*np.sqrt(arrays_3d["CenterlineSectionArea"]/np.pi))/0.04
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


write_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_aug_lookup_branch.vtp", reader_3d)

minid = np.argmin(arrays_3d["GlobalNodeId"])
inlet_area = arrays_3d["area"][minid]
pdb.set_trace()
