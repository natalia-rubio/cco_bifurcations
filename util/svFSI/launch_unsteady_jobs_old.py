from util.sherlock_util import *
from write_solver_files_unsteady import *
# from util.svFSI.util.projection import * For solution initialization
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


    #try:
    flow_index = "unsteady"; flow_name = f"unsteady"

    if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/{flow_name}_red_sol"):
        print(f"Simulation already complete for flow {flow_index}")
        continue


    set_up_sim_directories(anatomy, set_type, geo_name, flow_name, num_cores)
    flow_params = {"flow_amp": inlet_flow,
                    "vel_in": inlet_vel,
                    "res_1": 100,
                    "res_2": 100}

    #time_step_size = (np.sqrt(inlet_area/np.pi))/flow_params["vel_in"]
    time_step_size = 2*0.4/num_time_steps
    print(f"Time step size: {time_step_size}")
    write_svfsi_unsteady(anatomy, set_type, geo_name, flow_index, flow_params, copy.deepcopy(cap_numbers), inlet_cap_number, num_time_steps, time_step_size, inc = inc)
    write_flow_unsteady(anatomy, set_type, geo_name, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
    print("Done writing solver files.")
    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()


    write_job_unsteady(anatomy, set_type, geo_name, flow_name = flow_name, flow_index = flow_index, num_cores = num_cores, num_time_steps = num_time_steps, inc = inc)
    i = "unsteady"
    os.system(f"sbatch /scratch/users/nrubio/job_scripts/{geo}_{i}.sh")
    print(f"Started job for {geo} flow {flow_index}")
    print("\n\
            ---------------------------------\n")    

    # except:
    #     continue
    num_launched +=1
