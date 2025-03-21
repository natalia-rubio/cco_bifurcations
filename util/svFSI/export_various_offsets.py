import os
import sys
import math
import numpy as np
import pickle
import shutil
import pdb
import subprocess
import time
import copy
import get_avg_sol
import util.junction_proc
import centerline_proj

def check_convergence_offsets(geo_name, flow_index, anatomy, set_type, num_time_steps, inc):
    flow_name = f"flow_{flow_index}"
    
    centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
    print("Averaging 3D results.")
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, points = util.junction_proc.load_centerline_data(fpath_1d = centerline_dir)
    #print("Extracted centerline data.")
    offset_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for percent_offset in offset_list:
        print(f"Percent offset: {percent_offset}")
        junction_dict = util.junction_proc.identify_junctions_percent_offset(junction_id, branch_id, pt_id, path, points, percent_offset)
        results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{flow_index}_offset_{int(percent_offset*100)}"
        print(f"Results dir: {results_dir}")
        print(f"Junction dict: {junction_dict}")
        
        soln_dict, conv = get_avg_sol.get_avg_steady_results(ss_tol= 0.02, inc = inc,
                        fpath_1d = centerline_dir,
                        fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps):03d}.vtu",
                        fpath_3d_prev = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps)-inc:03d}.vtu",
                        fpath_out = results_dir,
                        pt_inds = junction_dict[0]["branch_pts_offset"], 
                        offsets = junction_dict[0]["total_lengths"],
                        only_caps=False)



    return

geo_name = sys.argv[1]; flow_index = sys.argv[2]; anatomy = sys.argv[3]; set_type = sys.argv[4];num_time_steps = sys.argv[5]; inc = int(sys.argv[6])
if __name__ == "__main__":
    anatomy = sys.argv[1]
    set_type = sys.argv[2]
    num_time_steps = int(sys.argv[3])
    num_cores = int(sys.argv[4])
    num_geos = int(sys.argv[5])
    num_flows = int(sys.argv[6])
    inc = int(sys.argv[7])

    dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}"
    geos = os.listdir(dir); geos.sort(); geo_ind = 0;#
    num_geos = len(geos)
    num_launched = 0

    while num_launched < num_geos:
        geo = geos[geo_ind]; geo_name = geo; print(f"Geometry: {geo_name}")
        geo_ind += 1

        for i, inlet_flow_fac in enumerate([0.25, 0.5, 0.75, 1]):
            flow_index = i; flow_name = f"flow_{flow_index}"
            try:
                check_convergence_offsets(geo_name = geo_name, flow_index = flow_index, anatomy = anatomy, set_type = set_type, num_time_steps = num_time_steps, inc = inc)
            except Exception as error:
                # handle the exception
                print("An exception occurred:", type(error).__name__) 
                print(error)