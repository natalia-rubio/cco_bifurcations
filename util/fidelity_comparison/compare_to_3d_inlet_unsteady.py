from cProfile import label
import sys
from turtle import color
from pyparsing import line
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
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex']=True
colors = ["royalblue", "orangered", "seagreen", "peru", "blueviolet"]


def compare_to_3d_inlet_unsteady(junction_mode = "standard",
                  tree_name = "tree_20_unsteady",
                  time_step = "700"):

    reader_0d = read_geo(f"trees/zerod_output_cent/standard/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_rr = read_geo(f"trees/zerod_output_cent/RR/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_unsteady.vtp").GetOutput()
    #was 300

    arrays_0d = get_all_arrays(reader_0d)
    arrays_rr = get_all_arrays(reader_rr)
    arrays_3d = get_all_arrays(reader_3d)

    
    times_3d = [int(key[9:]) for key in arrays_3d.keys() if "pressure" in key]
    times_3d = [time for time in times_3d if time % 20 == 0]  # Only take every 10th time step
    times_0d = [float(key[9:]) for key in arrays_0d.keys() if "pressure" in key]
    times_rr = [float(key[9:]) for key in arrays_rr.keys() if "pressure" in key]
    dt = times_0d[1] - times_0d[0]
   

    flows_0d = []; pressures_0d = []
    for time in times_0d:
        flows_0d.append(arrays_0d[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == 23][0])
        pressures_0d.append(arrays_0d[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == 23][0]/1333)

    flows_rr = []; pressures_rr = []
    for time in times_rr:
        flows_rr.append(arrays_rr[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == 23][0])
        pressures_rr.append(arrays_rr[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == 23][0]/1333)


    flows_3d = []; pressures_3d = []
    for time in times_3d:
        flows_3d.append(arrays_3d[f"velocity_{time:03d}"][arrays_3d["GlobalNodeId"] == 23][0])
        pressures_3d.append(arrays_3d[f"pressure_{time:03d}"][arrays_3d["GlobalNodeId"] == 23][0]/1333)
    times_3d = [time * dt for time in times_3d]


    
    plt.clf()
    plt.plot(flows_0d, pressures_0d, label="0D standard", color="slategrey")
    plt.plot(flows_rr, pressures_rr, label="0D RR", color="mediumturquoise")
    plt.scatter(flows_3d, pressures_3d, marker = "*", s = 70,  color="goldenrod", label="3D")
    plt.xlabel("Flow (cm$^3$/s)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    os.makedirs(f"results/{tree_name}", exist_ok=True)
    plt.savefig(f"results/{tree_name}/0d_standard_pf.png")

    plt.clf()
    plt.plot(times_0d, flows_0d, color="slategrey", linewidth=4)
    plt.plot(times_rr, flows_rr, color="mediumturquoise")
    plt.scatter(times_3d, flows_3d, marker = "*", s = 70,  color="goldenrod")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (cm$^3$/s)")
    plt.savefig(f"results/{tree_name}/0d_ft.png")

    plt.clf()
    plt.plot(times_0d, pressures_0d, color="slategrey")
    plt.plot(times_rr, pressures_rr, color="mediumturquoise")
    plt.scatter(times_3d, pressures_3d, marker = "*", s = 70,  color="goldenrod")
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (mmHg)")
    plt.savefig(f"results/{tree_name}/0d_pt.png")
    pdb.set_trace()    

    flow_3d = arrays_3d["Velocity"][arrays_3d["GlobalNodeId"] == 100]

    area = arrays_3d["CenterlineSectionArea"]

    pressure_0d = arrays_0d["pressure"][arrays_3d["GlobalNodeId"] == 10]
    pressure_3d = arrays_3d["Pressure"][arrays_3d["GlobalNodeId"] == 10]

    flow_error_0d_rel = (flow_0d-flow_3d)/flow_3d
    flow_error_0d_tot = (flow_0d-flow_3d)
    # flow_error_0d_rel[arrays_3d["BifurcationId"] >= 0] = 0
    # flow_error_0d_tot[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Flow error      (Relative):     {flow_error_0d_rel}")
    print(f"Flow error      (Total):        {flow_error_0d_tot}")
   
    pressure_error_0d_rel = (pressure_0d - pressure_3d)/pressure_3d
    pressure_error_0d_tot = (pressure_0d - pressure_3d)/1333
    # pressure_error_0d_rel[arrays_3d["BifurcationId"] >= 0] = 0
    # pressure_error_0d_tot[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Pressure error  (Relative):     {pressure_error_0d_rel}")
    print(f"Pressure error  (Total mmHg):        {pressure_error_0d_tot}")

    return

if __name__ == "__main__":
    compare_to_3d_inlet_unsteady(junction_mode = sys.argv[1], tree_name= sys.argv[2])
                  