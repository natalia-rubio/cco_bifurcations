from util.sherlock_util import *
from write_solver_files import *
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
        
    
    geo_dict = load_dict(f"{dir}/{geo_name}/geo_params_dict")
    flow_split = geo_dict["flow_split"][0]
    inlet_flows_3D = geo_dict["inlet_flows_3D"]
    inlet_area_3D = geo_dict["inlet_area_3D"]
    inlet_area = geo_dict["inlet_area"] #np.pi * 0.28382253272887237 **2
    
    inlet_REs = [1.06 * (inlet_flows_3D[i]/inlet_area_3D) * ((inlet_area_3D/np.pi)**0.5) *2 /0.04 for i in range(len(inlet_flows_3D))]
    inlet_flows_scaled = [inlet_area * re * 0.04 /(1.06 * ((inlet_area/np.pi)**0.5) *2) for re in inlet_REs]
    for i in range(len(inlet_flows_scaled)):
        
        try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if num_flows == 2:
                if i == 0 or i == 2:
                    continue
                    
            for offset in range(1,10):
                if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{i}_offset_{offset*10}_red_sol"):
                    print(f"Simulation already complete for flow {flow_index} offset {offset*10}")
                    continue
                else:
                    print(f"Launching flow {i}.")

                    set_up_sim_directories(anatomy, set_type, geo_name, flow_name, num_cores)
                    flow_params = {"flow_amp": inlet_flows_scaled[i], #inlet_flow*inlet_flow_fac,
                                "vel_in": inlet_flows_scaled[i]/inlet_area} #inlet_vel*inlet_flow_fac}

                    cap_dict = load_dict(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/cap_dict")
                    print(cap_dict)
                    res_caps = list(cap_dict.keys())
                    #res_caps.remove(inlet_cap_number)
                    
                    outlet_area_total = sum([cap_dict[res_cap] for res_cap in res_caps])
                    min_cap_area = min([cap_dict[res_cap] for res_cap in res_caps])
                    new_cap_dict = {"inlet": {},
                            "daughter1_outlet": {},
                            "daughter2_outlet": {}}
                    
                    
                    for res_cap in res_caps:
                            
                            if res_cap == inlet_cap_number:
                                new_cap_dict["inlet"]["id"] = res_cap
                                new_cap_dict["inlet"]["area"] = cap_dict[res_cap]
                                new_cap_dict["inlet"]["flow"] = flow_params["flow_amp"]
                                print(f'Inflow {new_cap_dict["inlet"]["flow"]}')
                            
                            elif cap_dict[res_cap] == min_cap_area:
                                new_cap_dict["daughter2_outlet"]["id"] = res_cap
                                new_cap_dict["daughter2_outlet"]["area"] = cap_dict[res_cap]
                                new_cap_dict["daughter2_outlet"]["res"] = geo_dict["flow_split"][0]*10000
                            else:
                                new_cap_dict["daughter1_outlet"]["id"] = res_cap
                                new_cap_dict["daughter1_outlet"]["area"] = cap_dict[res_cap]
                                new_cap_dict["daughter1_outlet"]["res"] =  10000

                    time_step_size = (np.sqrt(inlet_area/np.pi))/flow_params["vel_in"]
                    print(f"Time step size: {time_step_size}")
                    #write_svfsi_2flow(anatomy, set_type, geo_name, flow_name, flow_params, new_cap_dict, inlet_cap_number, num_time_steps, time_step_size, inc = inc)
                    write_svfsi(anatomy, set_type, geo_name, flow_name, flow_params, new_cap_dict, inlet_cap_number, num_time_steps, time_step_size, inc = inc)
                    #write_flow_steady(anatomy, set_type, geo_name, flow_index, flow_params["flow_amp"], inlet_cap_number, num_time_steps, time_step_size)
                    print("Done writing solver files.")
                    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/numstart.dat", "w"); f.write("0"); f.close()


                    write_job_steady(anatomy, set_type, geo_name, flow_name = flow_name, flow_index = flow_index, num_cores = num_cores, num_time_steps = num_time_steps, inc = inc)
                    os.system(f"sbatch /scratch/users/nrubio/job_scripts/{geo}_f{i}.sh")
                    print(f"Started job for {geo} flow {flow_index}")
                    print("\n\
                        ---------------------------------\n")    
                    break

        except:
            continue
    num_launched +=1
