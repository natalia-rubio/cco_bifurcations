import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff

def extract_steady_flow_data(anatomy, set_type, require4):
    re_char = 4500

    geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}"); geos.sort(); print(f"Geometries: {geos}")
    CCO_params_dict = {"daughter1_angle": [],
                    "daughter2_angle": [],
                    "daughter1_area_ratio": [],
                    "daughter2_area_ratio": [],
                    "inlet_area": [],
                    "daughter1_dP": [],
                    "daughter2_dP": [],
                    "daughter1_dP_original": [],
                    "daughter2_dP_original": [],
                    "daughter1_dP_total": [],
                    "daughter2_dP_total": [],
                    "daughter1_dP_dyn": [],
                    "daughter2_dP_dyn": [],
                    "inlet_P_dyn": [],
                    "daughter1_P_dyn": [],
                    "daughter2_P_dyn": [],
                    "daughter1_flow": [],
                    "daughter2_flow": [],
                    "inlet_flow": [],
                    "daughter1_velocity": [],
                    "daughter2_velocity": [],
                    "inlet_velocity": [],
                    "daughter1_energy": [],
                    "daughter2_energy": [],
                    "inlet_energy": [],
                    "U_char": [],
                    "daughter1_dP_star": [],
                    "daughter2_dP_star": [],
                    "daughter1_flow_star": [],
                    "daughter2_flow_star": []
                    }
    
    for j, geo in enumerate(geos[0:5]):
        results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}/"
        
        daughter1_dPs = [0, ]
        daughter2_dPs = [0, ]
        daughter1_dP_tots = [0, ]
        daughter2_dP_tots = [0, ]
        daughter1_flows = [0, ]
        daughter2_flows = [0, ]
        daughter1_energy = [0, ]
        daughter2_energy = [0, ]
        inlet_flows = [0, ]
        inlet_energy = [0, ]

        # Compose lists of flow and pressure data for each outlet
        for i in [0,1,2,3]:
            #print(f"Flow {i}")
            try:
                # pdb.set_trace()
                flow_result_dir = results_dir + f"flow_{i}_red_sol"
                if not os.path.exists(flow_result_dir):
                        #print(f"Flow {i} missing for geometry {geo} at {results_dir}")
                        continue
                
                soln_dict = load_dict(flow_result_dir)
                #print(f"soln_dict: {soln_dict}")
                daughter1_dPs.append(soln_dict["pressure_in_time"][1] - soln_dict["pressure_in_time"][0])
                daughter2_dPs.append(soln_dict["pressure_in_time"][2] - soln_dict["pressure_in_time"][0])
                inlet_flows.append(soln_dict["flow_in_time"][0])
                daughter1_flows.append(soln_dict["flow_in_time"][1])
                daughter2_flows.append(soln_dict["flow_in_time"][2])
                inlet_energy.append(soln_dict["energy_in_time"][0])
                daughter1_energy.append(soln_dict["energy_in_time"][1])
                daughter2_energy.append(soln_dict["energy_in_time"][2])
                daughter1_dP_tots.append(soln_dict["pressure_in_time"][1]+soln_dict["energy_in_time"][1] - (soln_dict["pressure_in_time"][0]+soln_dict["energy_in_time"][0]))
                daughter2_dP_tots.append(soln_dict["pressure_in_time"][2]+soln_dict["energy_in_time"][2] - (soln_dict["pressure_in_time"][0]+soln_dict["energy_in_time"][0]))
                #print(f'Daughter1 dP tots: {daughter1_dP_tots}')
                #pdb.set_trace()
                assert len(daughter1_dPs) == len(daughter1_flows); "Lengths of daughter1_dPs and daughter1_flows do not match."
                assert len(daughter2_dPs) == len(daughter2_flows); "Lengths of daughter2_dPs and daughter2_flows do not match."
                assert len(daughter1_dP_tots) == len(daughter1_flows); "Lengths of daughter1_dP_tots and daughter1_flows do not match."
                assert len(daughter2_dP_tots) == len(daughter2_flows); "Lengths of daughter2_dP_tots and daughter2_flows do not match."
            except:
                if require4:
                    raise ValueError(f"Could not extract steady data from {geo}, flow {i}.\n\
                                    Solution dict: {soln_dict}")
                continue

        if len(daughter1_dPs) <  5:
            print(f"Fewer than 4 flow data points for {geo}.")
            continue
        #pdb.set_trace()
        
        plt.clf()
        fig, axs = plt.subplots(2,2, figsize = (10,10))
        axs[0,0].plot(daughter1_flows, daughter1_dPs, label = "daughter1_dP")
        axs[0,1].plot(daughter2_flows, daughter2_dPs, label = "daughter2_dP")
        axs[0,0].set_xlabel("Inlet Flow")
        axs[0,1].set_xlabel("Inlet Flow")
        axs[0,0].set_ylabel("Daughter1 dP")
        axs[0,1].set_ylabel("Daughter2 dP")
        pdb.set_trace()
        axs[1,0].plot(daughter1_flows, daughter1_dP_tots, label = "daughter1_flow")
        axs[1,1].plot(daughter2_flows, daughter2_dP_tots, label = "daughter2_flow")
        axs[1,0].set_xlabel("Inlet Flow")
        axs[1,1].set_xlabel("Inlet Flow")
        axs[1,0].set_ylabel("Daughter1 dP total")
        axs[1,1].set_ylabel("Daughter2 dP total")
        if not os.path.exists(f"results/flow_vs_dp"):
            os.mkdir(f"results/flow_vs_dp")
        plt.savefig(f"results/flow_vs_dp/{geo}")
    return CCO_params_dict

extract_steady_flow_data("angles_CCO", "random", False)