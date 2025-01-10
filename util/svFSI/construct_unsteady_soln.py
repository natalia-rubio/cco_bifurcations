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

def construct_soln(geo_name, anatomy, set_type, max_time_step, inc):
    flow_name = f"unsteady"
    flow_index = "unsteady"
    results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/{flow_name}"
    centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
    print("Averaging 3D results.")
    print("Centerline dir: " + centerline_dir)
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = util.junction_proc.load_centerline_data(fpath_1d = centerline_dir)
    junction_dict, junc_pt_ids = util.junction_proc.identify_junctions_offset(junction_id, branch_id, pt_id, offset = 20)
    
    soln_dict = get_avg_sol.get_avg_unsteady_results(ss_tol= 0.02, 
                    inc = inc,
                    max_time_step= max_time_step,
                    fpath_1d = centerline_dir,
                    fpath_3d_base = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_{flow_index}_",
                    fpath_out = results_dir,
                    pt_inds = junc_pt_ids, only_caps=False)
    
    print("Done constructing unsteady solution.")
    return

geo_name = sys.argv[1]; anatomy = sys.argv[2]; set_type = sys.argv[3];num_time_steps = int(sys.argv[4]); inc = int(sys.argv[5])
construct_soln(geo_name = geo_name, anatomy = anatomy, set_type = set_type, max_time_step=num_time_steps, inc = inc)
