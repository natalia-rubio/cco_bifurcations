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

def get_outlet_nodes(arrays_3d):
    num_branchs = np.max(arrays_3d["BranchId"])
    outlet_node_pt_list = []
    for branch_ind in range(num_branchs):
        branch_pts = np.where(arrays_3d["BranchId"] == branch_ind)[0]
        branch_path = arrays_3d["CenterlineSectionArea"][branch_pts]
        branch_ids = arrays_3d["GlobalNodeId"][branch_pts]
        max_branch_ind = np.argmax(branch_path)
        branch_outlet_pt = branch_ids[max_branch_ind]
        outlet_node_pt_list.append(branch_outlet_pt)
    assert len(outlet_node_pt_list) == num_branchs
    
    outlet_node_pt_arr = False * arrays_3d["GlobalNodeId"]
    for node_id in outlet_node_pt_list:
        outlet_node_pt_arr[np.where(arrays_3d["GlobalNodeId"] == node_id)[0]] = True

    return outlet_node_pt_arr.astype(bool)

def compare_to_3d(junction_mode = "standard",
                  tree_name = "tree_dec1",
                  flow_amp = "full",
                  gen_mode = "sv"):

    reader_0d = read_geo(f"trees/zerod_output_cent_{gen_mode}_{junction_mode}/{tree_name}_{flow_amp}/centerline_sol.vtp").GetOutput()
    reader_3d = read_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_{flow_amp}.vtp").GetOutput()
    #was 300

    arrays_0d = get_all_arrays(reader_0d)
    arrays_3d = get_all_arrays(reader_3d)

    outlet_pts = get_outlet_nodes(arrays_3d)
    

    flow_0d = arrays_0d["flow"][outlet_pts]
    flow_3d = arrays_3d["Velocity"][outlet_pts]

    area = arrays_3d["CenterlineSectionArea"][outlet_pts]

    pressure_0d = arrays_0d["pressure"][outlet_pts]
    pressure_3d = arrays_3d["Pressure"][outlet_pts]

    

    flow_error_0d = (flow_0d-flow_3d)/flow_3d
    #flow_error_0d[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Flow error: {compute_rmse(flow_error_0d,0*flow_error_0d)}")


    pressure_error_0d = (pressure_0d - pressure_3d)/pressure_3d
    #pressure_error_0d[arrays_3d["BifurcationId"] >= 0] = 0
    print(f"Pressure error: {compute_rmse(pressure_error_0d,0*pressure_error_0d)}")

    write_geo(f"trees/threed_output_cent/{tree_name}/centerline_sol_{junction_mode}.vtp", reader_3d)
    return

if __name__ == "__main__":
    compare_to_3d(junction_mode = sys.argv[1])
                  