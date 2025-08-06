from cProfile import label
import sys
from turtle import color
from matplotlib import lines
from pyparsing import line
import vtk
import os
import numpy as np
import pdb
from scipy.interpolate import interp1d
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

    
    tree_name_split = tree_name.split("_")[0:2]
    tree_name_base = "_".join(tree_name_split)

    reader_0d = read_geo(f"trees/zerod_output_cent/standard/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_rri = read_geo(f"trees/zerod_output_cent/RRI/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_ri = read_geo(f"trees/zerod_output_cent/RI/{tree_name_base}/{tree_name}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_unsteady.vtp").GetOutput()
    #was 300

    arrays_0d = get_all_arrays(reader_0d)
    arrays_rri = get_all_arrays(reader_rri)
    arrays_ri = get_all_arrays(reader_ri)
    arrays_3d = get_all_arrays(reader_3d)

    
    times_3d = [int(key[9:]) for key in arrays_3d.keys() if "pressure" in key]
    times_3d = [time for time in times_3d if time % 20 == 0]  # Only take every 10th time step
    times_0d = [float(key[9:]) for key in arrays_0d.keys() if "pressure" in key]
    times_rri = [float(key[9:]) for key in arrays_rri.keys() if "pressure" in key]
    times_ri = [float(key[9:]) for key in arrays_ri.keys() if "pressure" in key]

    dt = times_0d[1] - times_0d[0]
    dt_3d = 0.001
    branch0_locs = np.where(arrays_3d["BranchId"] == 0)[0]
    branch0_valid_locs = np.where(~np.isnan(arrays_3d["pressure_100"][branch0_locs])) # Get the first value of the pressure for branch 0
    inlet_gid = arrays_3d["GlobalNodeId"][branch0_locs[branch0_valid_locs[0][0]]]
    #inlet_gid = arrays_3d["GlobalNodeId"][branch0_valid_locs[0][0]]
    #inlet_gid = 23
    flows_0d = []; pressures_0d = []
    for time in times_0d:
        flows_0d.append(arrays_0d[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_0d.append(arrays_0d[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)

    flows_rri = []; pressures_rri = []
    for time in times_rri:
        flows_rri.append(arrays_rri[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_rri.append(arrays_rri[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)

    flows_ri = []; pressures_ri = []
    for time in times_rri:
        flows_ri.append(arrays_ri[f"flow_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_ri.append(arrays_ri[f"pressure_{time:0.5f}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)
        
    flows_3d = []; pressures_3d = []

    for time in times_3d:
        flows_3d.append(arrays_3d[f"velocity_{time:03d}"][arrays_3d["GlobalNodeId"] == inlet_gid][0])
        pressures_3d.append(arrays_3d[f"pressure_{time:03d}"][arrays_3d["GlobalNodeId"] == inlet_gid][0]/1333)
    times_3d = [time * dt_3d for time in times_3d]
    
    zipped_lists = zip(times_3d, flows_3d, pressures_3d)
    sorted_zipped_lists = sorted(zipped_lists)
    # Unzip the sorted lists
    times_3d, flows_3d, pressures_3d = zip(*sorted_zipped_lists)
    n_reps = 2
    times_3d_last = times_3d[-len(times_3d)//n_reps:]  # Only take the second half of the time steps
    flows_3d_last = flows_3d[-len(flows_3d)//n_reps:]
    pressures_3d_last = pressures_3d[-len(pressures_3d)//n_reps:]
    
    times_3d_lst_fine = np.linspace(times_3d_last[0], times_3d_last[-1], len(times_3d_last)*100)
    flows_3d_last_fine = interp1d(times_3d_last, flows_3d_last, kind='quadratic')(times_3d_lst_fine)
    pressures_3d_last_fine = interp1d(times_3d_last, pressures_3d_last, kind='quadratic')(times_3d_lst_fine)
    n_reps = 4
    pressures_0d_last = pressures_0d[-len(pressures_0d)//n_reps:];flows_0d_last = flows_0d[-len(flows_0d)//n_reps:]; times_0d_last = times_0d[-len(times_0d)//n_reps:]
    pressures_rri_last = pressures_rri[-len(pressures_rri)//n_reps:];flows_rri_last = flows_rri[-len(flows_rri)//n_reps:]; times_rri_last = times_rri[-len(times_rri)//n_reps:]
    pressures_ri_last = pressures_ri[-len(pressures_ri)//n_reps:];flows_ri_last = flows_ri[-len(flows_ri)//n_reps:]; times_ri_last = times_rri[-len(times_rri)//n_reps:]
    
    


    # half_pt = int(len(times_3d)/2)
    # times_3d = times_3d[:half_pt]  # Only take the second half of the time steps
    # flows_3d = flows_3d[half_pt:]
    # pressures_3d = pressures_3d[half_pt:]
    
    # --------------------- FLOW PLOT ---------------------
    plt.clf()
    plt.plot(times_0d_last, pressures_0d_last, label="0D standard", color="tomato")
    plt.plot(times_rri_last, pressures_rri_last, label="0D RRI", color="seagreen")
    plt.plot(times_ri_last, pressures_ri_last, label="0D RI", color="royalblue")
    plt.plot(times_3d_last, pressures_3d_last, color="slategrey", label="3D")
    plt.xlabel("Time (cm$^3$/s)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend()
    os.makedirs(f"results/unsteady/{tree_name}", exist_ok=True)
    plt.savefig(f"results/unsteady/{tree_name}/0d_standard_pt.png")

    # --------------------- PRESSURE PLOT ---------------------
    plt.clf()
    plt.plot(times_0d_last, flows_0d_last, color="tomato", linewidth=4)
    plt.plot(times_rri_last, flows_rri_last, color="seagreen")
    plt.plot(times_ri_last, flows_ri_last, color="royalblue")
    plt.plot(times_3d_last, flows_3d_last, color="slategrey")
    plt.xlabel("Time (s)")
    plt.ylabel("Flow (cm$^3$/s)")
    plt.savefig(f"results/unsteady/{tree_name}/0d_ft.png")

    # --------------------- PRESSURE LOOP ---------------------
    plt.clf()
    fig = plt.figure(figsize=(2.5, 2.5))
    plt.plot(flows_0d_last,         pressures_0d_last,          color="tomato",             dashes=(1, 1),     label="0D standard", linewidth=3)
    plt.plot(flows_rri_last,        pressures_ri_last,          color="cornflowerblue",     linestyle='dashdot',    label="0D RI", linewidth=2)
    plt.plot(flows_rri_last,        pressures_rri_last,         color="limegreen",          linestyle='dashed',     label="0D RRI", linewidth=2)
    
    plt.plot(flows_3d_last_fine,    pressures_3d_last_fine,     color="slategray",          linestyle='solid',      label="3D", linewidth=2)
    plt.xlabel("Flow (cm$^3$/s)")
    plt.ylabel("Pressure (mmHg)")
    plt.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols = 4)
    plt.savefig(f"results/unsteady/{tree_name}/0d_pressure_loop.pdf", bbox_inches='tight')
    
    pdb.set_trace()    


    return

if __name__ == "__main__":
    compare_to_3d_inlet_unsteady(junction_mode = sys.argv[1], tree_name= sys.argv[2])
                  