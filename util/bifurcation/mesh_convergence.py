import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff

anatomy = "CCO_80"
set_type = "mesh_convergence_4"

require4 = False
re_char = 4500
fig, ax = plt.subplots(1, 2, figsize=(10, 10))
geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}"); geos.sort(); print(f"Geometries: {geos}")
results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/"
CCO_params_dict = {"daughter1_angle": [],
            "daughter2_angle": [],
            "daughter1_area_ratio": [],
            "daughter2_area_ratio": [],
            "inlet_area": [],
            "daughter1_dP": [],
            "daughter2_dP": [],
            "daughter1_flow": [],
            "daughter2_flow": [],
            "U_char": [],
            "daughter1_dP_star": [],
            "daughter2_dP_star": [],
            "daughter1_flow_star": [],
            "daughter2_flow_star": [],
            "name": []
            }

for j, geo in enumerate(geos[0:]):

    daughter1_dPs = [0, ]
    daughter2_dPs = [0, ]
    daughter1_flows = [0, ]
    daughter2_flows = [0, ]

    # Compose lists of flow and pressure data for each outlet
    for i in [1,3]:
        try:
            flow_result_dir = results_dir + f"{geo}/flow_{i}_red_sol"
            if not os.path.exists(flow_result_dir):
                    print(f"Flow {i} missing for geometry {geo} at {flow_result_dir}")
                    continue
            soln_dict = load_dict(flow_result_dir)
            print(soln_dict["paths"])
            poiseuille_res1 = 8*np.pi*0.04*soln_dict["paths"][0,1]/soln_dict["areas"][0,1]**2
            poiseuille_res2 = 8*np.pi*0.04*soln_dict["paths"][0,2]/soln_dict["areas"][0,2]**2
            daughter1_dPs.append(soln_dict["pressure_in_time"][1]/1333 - soln_dict["pressure_in_time"][0]/1333 - poiseuille_res1 * soln_dict["flow_in_time"][1]/1333)
            daughter2_dPs.append(soln_dict["pressure_in_time"][2]/1333 - soln_dict["pressure_in_time"][0]/1333 - poiseuille_res2 * soln_dict["flow_in_time"][2]/1333)
            daughter1_flows.append(soln_dict["flow_in_time"][1])
            daughter2_flows.append(soln_dict["flow_in_time"][2])
            assert len(daughter1_dPs) == len(daughter1_flows); "Lengths of daughter1_dPs and daughter1_flows do not match."
            assert len(daughter2_dPs) == len(daughter2_flows); "Lengths of daughter2_dPs and daughter2_flows do not match."
        except:
            if require4:
                raise ValueError(f"Could not extract steady data from {geo}, flow {i}.\n\
                                Solution dict: {soln_dict}")
            continue


    if len(daughter1_dPs) < 3 or len(daughter2_dPs) < 3:
        print(f"Fewer than 3 flow data points for {geo}.")
        continue
    
    ax[0].scatter(daughter1_flows, daughter1_dPs, s = 50, alpha = 0.5, label = f"{geo.split("_")[-1]} elements")
    ax[1].scatter(daughter2_flows, daughter2_dPs, s = 50, alpha = 0.5, label = f"{geo.split("_")[-1]} elements")

ax[0].set_title("Daughter 1")
ax[0].set_ylabel("Pressure Drop (mmHg)")
ax[0].set_xlabel("Flow (cm^3/s)")
ax[1].set_title("Daughter 2")
ax[1].set_xlabel("Flow (cm^3/s)")
ax[0].legend()
ax[1].legend()

if not os.path.exists(f"results/bifurcation_analysis"):
    os.mkdir(f"results/bifurcation_analysis")
fig.savefig(f"results/bifurcation_analysis/{anatomy}_mesh_convergence.png")
    