from util.sherlock_util import *
from write_solver_files import *
import get_avg_sol
# from util.svFSI.util.projection import * For solution initialization
def check_convergence(geo_name, flow_index, anatomy, set_type, num_time_steps, inc):
    flow_name = f"flow_{flow_index}"
    results_dir = f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{flow_index}"
    centerline_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/centerlines/centerline.vtp"
    print("Averaging 3D results.")
    print("Centerline dir: " + centerline_dir)
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path = util.junction_proc.load_centerline_data(fpath_1d = centerline_dir)
    junction_dict, offsets, junc_pt_ids = util.junction_proc.identify_junctions_offset(junction_id, branch_id, pt_id, path, offset = 20)
    soln_dict, conv = get_avg_sol.get_avg_steady_results(ss_tol= 0.02, inc = inc,
                    fpath_1d = centerline_dir,
                    fpath_3d = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps):03d}.vtu",
                    fpath_3d_prev = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/solution_flow_{flow_index}_{int(num_time_steps)-inc:03d}.vtu",
                    fpath_out = results_dir,
                    pt_inds = junc_pt_ids, 
                    offsets = offsets,
                    only_caps=False)

    return

anatomy = sys.argv[1]
set_type = sys.argv[2]
num_time_steps = int(sys.argv[3])
num_cores = int(sys.argv[4])
num_geos = int(sys.argv[5])
num_flows = int(sys.argv[6])
inc = int(sys.argv[7])

time_step_size = 0.001
num_launched = 0
print(f"Launching {num_geos} steady flow sweeps.")
dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}"
geos = os.listdir(dir); geos.sort(); geo_ind = 0;#

while num_launched < num_geos:

    geo = geos[geo_ind]; geo_name = geo; print(f"Geometry: {geo_name}")
    geo_ind += 1

    if not check_geo_name(geo):
        continue
    if not check_for_centerline(anatomy, set_type, geo_name):
        continue
    
    try:
        inlet_cap_number, cap_numbers = get_cap_info(anatomy, set_type, geo_name, correct_cap_numbers = 3)
    except:
        print("Problem with caps.")
        continue
        
    inlet_area = np.pi * 0.28382253272887237 **2
    re = 5500
    inlet_vel = re * 0.04 / (1.06 * 2 * 0.28382253272887237)
    inlet_flow = inlet_area * inlet_vel


    for i, inlet_flow_fac in enumerate([0.25, 0.5, 0.75, 1]):
        
        
        try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if num_flows == 2:
                if i == 0 or i == 2:
                    continue
            if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{i}_red_sol"):
                print(f"Simulation already complete for flow {flow_index}")
                continue
            print(f"Launching flow {i}.")
            #print(f"Initializing solution.")
            #project_0d_to_3D(anatomy, set_type, geo_name, flow_index)

            set_up_sim_directories(anatomy, set_type, geo_name, flow_name, num_cores)
            flow_params = {"flow_amp": inlet_flow*inlet_flow_fac,
                            "vel_in": inlet_vel*inlet_flow_fac}
            #pdb.set_trace()
            cap_dict = load_dict(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/cap_dict")
            res_caps = list(cap_dict.keys())
            res_caps.remove(inlet_cap_number)
            #print(cap_dict)
            outlet_area_total = sum([cap_dict[res_cap] for res_cap in res_caps])
            #print(f"outlet_area_total: {outlet_area_total}")

            time_step_size = (np.sqrt(inlet_area/np.pi))/flow_params["vel_in"]
            # print(f"Time step size: {time_step_size}")
            # write_svfsi(anatomy, set_type, geo_name, flow_index, flow_params, cap_dict, inlet_cap_number, num_time_steps, time_step_size, inc = inc)
            # write_flow_steady(anatomy, set_type, geo_name, flow_index, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
            # print("Done writing solver files.")
            # f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()

            check_convergence(geo_name = geo_name, flow_index = flow_index, anatomy = anatomy, set_type = set_type, num_time_steps = num_time_steps, inc = inc)
            
        except:
            continue
    num_launched +=1
