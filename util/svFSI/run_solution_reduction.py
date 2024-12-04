from util.sherlock_util import *
from write_solver_files import *
# from util.svFSI.util.projection import * For solution initialization
anatomy = sys.argv[1]
set_type = sys.argv[2]
num_time_steps = int(sys.argv[3])
num_cores = int(sys.argv[4])
num_geos = int(sys.argv[5])
num_flows = int(sys.argv[6])

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
    inlet_flow = inlet_area * 300


    for i, inlet_flow_fac in enumerate([0.25, 0.5, 0.75, 1]):
        print(f"Launching flow {i}.")
        
        try:
            flow_index = i; flow_name = f"flow_{flow_index}"
            if num_flows == 2:
                if i == 0 or i == 2:
                    continue
            if os.path.exists(f"/scratch/users/nrubio/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo_name}/flow_{i}_red_sol"):
                print(f"Simulation already complete for flow {flow_index}")
            
            os.system(f"python3 /home/users/nrubio/SV_scripts/svFSI/check_convergence.py {geo_name} {flow_index} {anatomy} {set_type} {num_time_steps}")

        except:
            continue
    num_launched +=1
