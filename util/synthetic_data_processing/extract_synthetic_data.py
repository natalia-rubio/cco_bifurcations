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
    for j, geo in enumerate(geos[0:]):
        results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}/"
        
        daughter1_dPs = [0, ]
        daughter2_dPs = [0, ]
        daughter1_flows = [0, ]
        daughter2_flows = [0, ]
        daughter1_energy = [0, ]
        daughter2_energy = [0, ]
        daughter1_dP_total = [0, ]
        daughter2_dP_total = [0, ]
        inlet_flows = [0, ]
        inlet_energy = [0, ]

        # Compose lists of flow and pressure data for each outlet
        for i in [1,3]:
            try:
                # pdb.set_trace()
                flow_result_dir = results_dir + f"flow_{i}_red_sol"
                if not os.path.exists(flow_result_dir):
                        print(f"Flow {i} missing for geometry {geo} at {results_dir}")
                        continue
                soln_dict = load_dict(flow_result_dir)

                daughter1_dPs.append(soln_dict["pressure_in_time"][1] - soln_dict["pressure_in_time"][0])
                daughter2_dPs.append(soln_dict["pressure_in_time"][2] - soln_dict["pressure_in_time"][0])
                inlet_flows.append(soln_dict["flow_in_time"][0])
                daughter1_flows.append(soln_dict["flow_in_time"][1])
                daughter2_flows.append(soln_dict["flow_in_time"][2])
                inlet_energy.append(soln_dict["energy_in_time"][0])
                daughter1_energy.append(soln_dict["energy_in_time"][1])
                daughter2_energy.append(soln_dict["energy_in_time"][2])
                daughter1_dP_total.append(soln_dict["pressure_in_time"][1]+soln_dict["energy_in_time"][1] - (soln_dict["pressure_in_time"][0]+soln_dict["energy_in_time"][0]))
                daughter2_dP_total.append(soln_dict["pressure_in_time"][2]+soln_dict["energy_in_time"][2] - (soln_dict["pressure_in_time"][0]+soln_dict["energy_in_time"][0]))

                #pdb.set_trace()
                assert len(daughter1_dPs) == len(daughter1_flows); "Lengths of daughter1_dPs and daughter1_flows do not match."
                assert len(daughter2_dPs) == len(daughter2_flows); "Lengths of daughter2_dPs and daughter2_flows do not match."
                # if geo == "CCO_221":
                #     pdb.set_trace()
            except:
                if require4:
                    raise ValueError(f"Could not extract steady data from {geo}, flow {i}.\n\
                                    Solution dict: {soln_dict}")
                continue
        # if daughter2_dP_total[-1] > 0:
        #     pdb.set_trace()
        if len(daughter1_dPs) < 3 or len(daughter2_dPs) < 3:
            print(f"Fewer than 3 flow data points for {geo}.")
            continue
        #pdb.set_trace()
        CCO_params_dict["daughter1_angle"].append(get_angle_diff(soln_dict["tangents"][:,1], soln_dict["tangents"][:,0])[0])
        CCO_params_dict["daughter2_angle"].append(get_angle_diff(soln_dict["tangents"][:,2], soln_dict["tangents"][:,0])[0])
        CCO_params_dict["daughter1_area_ratio"].append(soln_dict["areas"][0,1]/soln_dict["areas"][0,0])
        CCO_params_dict["daughter2_area_ratio"].append(soln_dict["areas"][0,2]/soln_dict["areas"][0,0])
        CCO_params_dict["inlet_area"].append(soln_dict["areas"][0,0])
        #pdb.set_trace()

        CCO_params_dict["daughter1_flow"].append(daughter1_flows)
        CCO_params_dict["daughter2_flow"].append(daughter2_flows)
        CCO_params_dict["inlet_flow"].append(inlet_flows)
        CCO_params_dict["daughter1_velocity"].append([daughter1_flow/soln_dict["areas"][0,1] for daughter1_flow in daughter1_flows])
        CCO_params_dict["daughter2_velocity"].append([daughter2_flow/soln_dict["areas"][0,2] for daughter2_flow in daughter2_flows])
        CCO_params_dict["inlet_velocity"].append([inlet_flow/soln_dict["areas"][0,0] for inlet_flow in inlet_flows])
        CCO_params_dict["daughter1_energy"].append(daughter1_energy)
        CCO_params_dict["daughter2_energy"].append(daughter2_energy)
        CCO_params_dict["inlet_energy"].append(inlet_energy)
        CCO_params_dict["U_char"].append(re_char * 0.04 / (1.06 * 2*np.sqrt(soln_dict["areas"][0,0]/np.pi)))

        CCO_params_dict["daughter1_dP_original"].append(daughter1_dPs)
        CCO_params_dict["daughter2_dP_original"].append(daughter2_dPs)

        poiseulle_res_1 = 8*0.04*np.pi*soln_dict["paths"][0][1]/(soln_dict["areas"][0,1]**2)
        poiseulle_res_2 = 8*0.04*np.pi*soln_dict["paths"][0][2]/(soln_dict["areas"][0,2]**2)
        print(f"Poiseuille resistance 1: {poiseulle_res_1}")
        print(f"Poiseuille resistance 2: {poiseulle_res_2}")
        CCO_params_dict["daughter1_dP"].append([daughter1_dP + 0*poiseulle_res_1*daughter1_flow for daughter1_dP, daughter1_flow in zip(daughter1_dPs, daughter1_flows)])
        CCO_params_dict["daughter2_dP"].append([daughter2_dP + 0*poiseulle_res_2*daughter2_flow for daughter2_dP, daughter2_flow in zip(daughter2_dPs, daughter2_flows)])

        CCO_params_dict["daughter1_dP_total"].append([daughter1_dP_total + poiseulle_res_1*flow for daughter1_dP_total, flow in zip(daughter1_dP_total, daughter1_flows)])
        CCO_params_dict["daughter2_dP_total"].append([daughter2_dP_total + poiseulle_res_2*flow for daughter2_dP_total, flow in zip(daughter2_dP_total, daughter2_flows)])

        # # CCO_params_dict["daughter1_P_dyn"].append([0.5*1.06*daughter1_velocity**2 for daughter1_velocity in CCO_params_dict["daughter1_velocity"][-1]])
        # # CCO_params_dict["daughter2_P_dyn"].append([0.5*1.06*daughter2_velocity**2 for daughter2_velocity in CCO_params_dict["daughter2_velocity"][-1]])
        # # CCO_params_dict["inlet_P_dyn"].append([0.5*1.06*inlet_velocity**2 for inlet_velocity in CCO_params_dict["inlet_velocity"][-1]])
        # # CCO_params_dict["daughter1_dP_dyn"].append([daughter1_P_dyn - inlet_P_dyn for daughter1_P_dyn, inlet_P_dyn in zip(CCO_params_dict["daughter1_P_dyn"][-1], CCO_params_dict["inlet_P_dyn"][-1])])
        # # CCO_params_dict["daughter2_dP_dyn"].append([daughter2_P_dyn - inlet_P_dyn for daughter2_P_dyn, inlet_P_dyn in zip(CCO_params_dict["daughter2_P_dyn"][-1], CCO_params_dict["inlet_P_dyn"][-1])])
        # CCO_params_dict["daughter1_dP_dyn"].append([daughter1_P_dyn - inlet_P_dyn for daughter1_P_dyn, inlet_P_dyn in zip(CCO_params_dict["daughter1_energy"][-1], CCO_params_dict["inlet_energy"][-1])])
        # CCO_params_dict["daughter2_dP_dyn"].append([daughter2_P_dyn - inlet_P_dyn for daughter2_P_dyn, inlet_P_dyn in zip(CCO_params_dict["daughter2_energy"][-1], CCO_params_dict["inlet_energy"][-1])])
        # # CCO_params_dict["daughter1_dP_static"].append([daughter1_dP - daughter1_P_dyn for daughter1_dP, daughter1_P_dyn in zip(CCO_params_dict["daughter1_dP"][-1], CCO_params_dict["daughter1_P_dyn"][-1])])
        # # CCO_params_dict["daughter2_dP_static"].append([daughter2_dP - daughter2_P_dyn for daughter2_dP, daughter2_P_dyn in zip(CCO_params_dict["daughter2_dP"][-1], CCO_params_dict["daughter2_P_dyn"][-1])])
        # CCO_params_dict["daughter1_dP_total"].append([daughter1_dP + daughter1_dP_dyn for daughter1_dP, daughter1_dP_dyn in zip(CCO_params_dict["daughter1_dP"][-1], CCO_params_dict["daughter1_dP_dyn"][-1])])
        # CCO_params_dict["daughter2_dP_total"].append([daughter2_dP + daughter2_dP_dyn for daughter2_dP, daughter2_dP_dyn in zip(CCO_params_dict["daughter2_dP"][-1], CCO_params_dict["daughter2_dP_dyn"][-1])])
        # # Adjust for Poiseuille pressure drop in outlets

        # Add non-dimensionalized parameters
        daughter1_dP_stars = [daughter1_dP/(1.06 * CCO_params_dict["U_char"][-1]**2) for daughter1_dP in CCO_params_dict["daughter1_dP_total"][-1]]
        daughter2_dP_stars = [daughter2_dP/(1.06 * CCO_params_dict["U_char"][-1]**2) for daughter2_dP in CCO_params_dict["daughter2_dP_total"][-1]]
        CCO_params_dict["daughter1_dP_star"].append(daughter1_dP_stars)
        CCO_params_dict["daughter2_dP_star"].append(daughter2_dP_stars)

        daughter1_flow_stars = [daughter1_flow/(CCO_params_dict["U_char"][-1]*CCO_params_dict["inlet_area"][-1]) for daughter1_flow in daughter1_flows]
        daughter2_flow_stars = [daughter2_flow/(CCO_params_dict["U_char"][-1]*CCO_params_dict["inlet_area"][-1]) for daughter2_flow in daughter2_flows]
        CCO_params_dict["daughter1_flow_star"].append(daughter1_flow_stars)
        CCO_params_dict["daughter2_flow_star"].append(daughter2_flow_stars)

        r_lin_calc = (daughter1_dPs[-1]/daughter1_flows[-1])*(CCO_params_dict["inlet_area"][-1]/CCO_params_dict["U_char"][-1])
        #print(f"Calculated R_lin_star: {r_lin_calc}")
    #pdb.set_trace()
    save_dict(CCO_params_dict, f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    return CCO_params_dict