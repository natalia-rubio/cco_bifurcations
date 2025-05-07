import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff

def extract_flow_behavior(geo_results_dir, offset):
    verbose = True
    daughter1_dPs = []
    daughter2_dPs = []
    inlet_flows = []
    daughter1_flows = []
    daughter2_flows = []
    inlet_velocity = []
    daughter1_velocity = []
    daughter2_velocity = []
    daughter1_energy = []
    daughter2_energy = []
    daughter1_dP_total = []
    daughter2_dP_total = []
    inlet_energy = []
    offset_dict = {}
    re_char = 4500
    # Compose lists of flow and pressure data for each outlet
    for i in [0,1,2,3,4,5]:
        try:
            flow_result_dir = f"{geo_results_dir}/flow_{i}_offset_{int(10*offset)}_red_sol"
            print(f"Flow result dir: {flow_result_dir}")
            # pdb.set_trace()
            # pdb.set_trace()
            if not os.path.exists(flow_result_dir):
                    raise ValueError(f"Could not find {flow_result_dir}.")
            
            soln_dict = load_dict(flow_result_dir)
            A_char = soln_dict["areas"][0,0]
            
            L_char = np.sqrt(A_char/np.pi)
            U_char = re_char * 0.04/(1.06 * 2*np.sqrt(A_char/np.pi))
            
            daughter1_dPs.append(-soln_dict["pressure_in_time"][1] + soln_dict["pressure_in_time"][0])
            daughter2_dPs.append(-soln_dict["pressure_in_time"][2] + soln_dict["pressure_in_time"][0])
            inlet_flows.append(soln_dict["flow_in_time"][0])
            daughter1_flows.append(soln_dict["flow_in_time"][1])
            daughter2_flows.append(soln_dict["flow_in_time"][2])
            
            if verbose:
                print(f"Extracted flow data from {flow_result_dir}.")
            assert len(daughter1_dPs) == len(daughter1_flows); "Lengths of daughter1_dPs and daughter1_flows do not match."
            assert len(daughter2_dPs) == len(daughter2_flows); "Lengths of daughter2_dPs and daughter2_flows do not match."

        
        except:
            require4 = False
            if require4:
                raise ValueError(f"Could not extract steady data from {flow_result_dir}.\n\
                                Solution dict: {soln_dict}")
            continue
    print(f"Daughter 1 flows: {daughter1_flows}")
    print(f"Daughter 1 dPs: {daughter1_dPs}")
    # for k in range(1, len(daughter1_flows)):
    #     daughter1_flows[k] = 0
    #     daughter2_flows[k] = 0
    #     daughter1_dPs[k] = 0
    #     daughter2_dPs[k] = 0


    assert len(daughter1_dPs) > 0; "Length of daughter1_dPs should be > 0."
    daughter1_flow_stars = [daughter1_flow/(U_char * A_char) for daughter1_flow in daughter1_flows]
    daughter2_flow_stars = [daughter2_flow/(U_char * A_char) for daughter2_flow in daughter2_flows]
    
    Q_star1 = np.asarray(daughter1_flow_stars).reshape(-1,)
    Q_star2 = np.asarray(daughter2_flow_stars).reshape(-1,)
    Q_star_inlet = Q_star1 + Q_star2

    daughter1_dP_stars = [daughter1_dP/(1.06 * U_char**2) for daughter1_dP in daughter1_dPs]
    daughter2_dP_stars = [daughter2_dP/(1.06 * U_char**2) for daughter2_dP in daughter2_dPs]

    dP_star1 = np.asarray(daughter1_dP_stars).reshape(-1,)
    dP_star2 = np.asarray(daughter2_dP_stars).reshape(-1,)
    dP_vec_star = np.hstack([dP_star1, dP_star2])

    Q1 = np.asarray(daughter1_flows).reshape(-1,)
    Q2 = np.asarray(daughter2_flows).reshape(-1,)
    Q_inlet = Q1 + Q2

    dP1 = np.asarray(daughter1_dPs).reshape(-1,)
    dP2 = np.asarray(daughter2_dPs).reshape(-1,)
    dP_vec = np.hstack([dP1, dP2])

    num_flows = len(daughter1_flows)
    num_coefs = 4

    A_mat_star = np.zeros((2*num_flows, num_coefs))
    A_mat = np.zeros((2*num_flows, num_coefs))

    # Inlet flows
    inlet_resistance = False
    if inlet_resistance:
        A_mat_star[0:num_flows,0] = Q_star_inlet
        A_mat_star[0:num_flows,1] = np.square(Q_star_inlet)
        A_mat_star[num_flows:2*num_flows,0] = Q_star_inlet
        A_mat_star[num_flows:2*num_flows,1] = np.square(Q_star_inlet)
    
        A_mat[0:num_flows,0] = Q_inlet
        A_mat[0:num_flows,1] = np.square(Q_inlet)
        A_mat[num_flows:2*num_flows,0] = Q_inlet
        A_mat[num_flows:2*num_flows,1] = np.square(Q_inlet)


    # Daughter 1 flows
    A_mat[0:num_flows,0] = Q1
    A_mat[num_flows:2*num_flows,2] = Q2
    A_mat_star[0:num_flows,0] = Q_star1
    A_mat_star[num_flows:2*num_flows,2] = Q_star2

    quadratic_resistors = True
    if quadratic_resistors:
        A_mat[0:num_flows,1] = np.square(Q1)
        A_mat[num_flows:2*num_flows,3] = np.square(Q2)
        A_mat_star[0:num_flows,1] = np.square(Q_star1)
        A_mat_star[num_flows:2*num_flows,3] = np.square(Q_star2)

    # Solve
    coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
    # R_lin_star_inlet    = coefs_star[0]
    # R_quad_star_inlet   = coefs_star[1]
    R_lin_star1         = coefs_star[0]
    R_quad_star1        = coefs_star[1]
    R_lin_star2         = coefs_star[2]
    R_quad_star2        = coefs_star[3]

    # offset_dict["inlet_R_lin_star"] = copy.copy(R_lin_star_inlet)
    # offset_dict["inlet_R_quad_star"] = copy.copy(R_quad_star_inlet)
    offset_dict["daughter1_R_lin_star"] = copy.copy(R_lin_star1)
    offset_dict["daughter2_R_lin_star"] = copy.copy(R_lin_star2)
    offset_dict["daughter1_R_quad_star"] = copy.copy(R_quad_star1)
    offset_dict["daughter2_R_quad_star"] = copy.copy(R_quad_star2)

    # Solve
    coefs, residuals, t, q = np.linalg.lstsq(A_mat, dP_vec, rcond=None)
    #pdb.set_trace()
    residuals = dP_vec - A_mat @ coefs
    print(f"Residuals: {np.linalg.norm(residuals/dP_vec)}")
    # if np.linalg.norm(residuals/dP_vec) > 1:
    #     raise ValueError(f"Residuals are too large: {np.linalg.norm(residuals/dP_vec)}")

    # R_lin_inlet    = coefs[0]
    # R_quad_inlet   = coefs[1]
    R_lin1         = coefs[0]
    R_quad1        = coefs[1]
    R_lin2         = coefs[2]
    R_quad2        = coefs[3]

    # offset_dict["inlet_R_lin"] = copy.copy(R_lin_inlet)
    # offset_dict["inlet_R_quad"] = copy.copy(R_quad_inlet)
    offset_dict["daughter1_R_lin"] = copy.copy(R_lin1)
    offset_dict["daughter2_R_lin"] = copy.copy(R_lin2)
    offset_dict["daughter1_R_quad"] = copy.copy(R_quad1)
    offset_dict["daughter2_R_quad"] = copy.copy(R_quad2)
    if verbose:
        print("Solved for resistances.")

    # Check consistency of non-dimensionalization
    
    # assert abs(offset_dict["inlet_R_lin"] - offset_dict["inlet_R_lin_star"]*1.06*U_char/A_char) < 0.1; "Inlet linear resistances do not match."
    # if "CCO_004" in geo_results_dir:
    #     pdb.set_trace()
    assert abs(offset_dict["daughter1_R_lin"] - offset_dict["daughter1_R_lin_star"]*1.06*U_char/A_char) < 0.1; "Daughter 1 linear resistances do not match."
    assert abs(offset_dict["daughter2_R_lin"] - offset_dict["daughter2_R_lin_star"]*1.06*U_char/A_char) < 0.1; "Daughter 2 linear resistances do not match."
    # assert abs(offset_dict["inlet_R_quad"] - offset_dict["inlet_R_quad_star"]*1.06/A_char**2) < 0.1; "Inlet quadratic resistances do not match."
    assert abs(offset_dict["daughter1_R_quad"] - offset_dict["daughter1_R_quad_star"]*1.06/A_char**2) < 0.1; "Daughter 1 quadratic resistances do not match."
    assert abs(offset_dict["daughter2_R_quad"] - offset_dict["daughter2_R_quad_star"]*1.06/A_char**2) < 0.1; "Daughter 2 quadratic resistances do not match."
    if verbose:
        print("Passed non-dimensionalization consistency check.")

    offset_dict["daughter1_flow"] = daughter1_flows
    offset_dict["daughter1_Re"] = [1.06*(flow/soln_dict["areas"][0,1])*np.sqrt(soln_dict["areas"][0,1]/np.pi)/0.04 for flow in daughter1_flows]
    offset_dict["daughter2_Re"] = [1.06*(flow/soln_dict["areas"][0,2])*np.sqrt(soln_dict["areas"][0,2]/np.pi)/0.04 for flow in daughter2_flows]
    offset_dict["daughter2_flow"] = daughter2_flows
    offset_dict["daughter1_dP"] = daughter1_dPs
    offset_dict["daughter2_dP"] = daughter2_dPs
    offset_dict["daughter1_flow_star"] = daughter1_flow_stars
    offset_dict["daughter2_flow_star"] = daughter2_flow_stars
    offset_dict["daughter1_dP_star"] = daughter1_dP_stars
    offset_dict["daughter2_dP_star"] = daughter2_dP_stars

    offset_dict["U_char"] = U_char
    offset_dict["A_char"] = A_char
    offset_dict["L_char"] = L_char

    offset_dict["daughter1_length_star"] = soln_dict["lengths"][0][0]/L_char
    offset_dict["daughter2_length_star"] = soln_dict["lengths"][1][0]/L_char

    offset_dict["daughter1_length"] = soln_dict["lengths"][0][0]
    offset_dict["daughter2_length"] = soln_dict["lengths"][1][0]

    offset_dict["daughter1_angle"] = get_angle_diff(soln_dict["tangents"][:,1], soln_dict["tangents"][:,0])[0]
    offset_dict["daughter2_angle"] = get_angle_diff(soln_dict["tangents"][:,2], soln_dict["tangents"][:,0])[0]

    offset_dict["daughter1_area_ratio"] = soln_dict["areas"][0,1]/A_char
    offset_dict["daughter2_area_ratio"] = soln_dict["areas"][0,2]/A_char
    total_daughter_area_ratio = soln_dict["areas"][0,1]/A_char + soln_dict["areas"][0,2]/A_char
    # print(f"Total daughter area ratio: {total_daughter_area_ratio}")
    # assert abs(total_daughter_area_ratio) < 1.5; "Total daughter area ratio too big."

    offset_dict["daughter1_area"] = soln_dict["areas"][0,1]
    offset_dict["daughter2_area"] = soln_dict["areas"][0,2]

    offset_dict["daughter1_area_ratio_inv2"] = (A_char/soln_dict["areas"][0,1])**2
    offset_dict["daughter2_area_ratio_inv2"] = (A_char/soln_dict["areas"][0,2])**2
    #pdb.set_trace()
    
    return offset_dict

def plot_geo(offset_dict, anatomy, set_type, geo):
    num_flows = len(offset_dict[list(offset_dict.keys())[0]]["daughter1_flow"])
    colors = ['b', 'g', 'y', 'r',"orange", "c", "m", "k"]
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10,10), sharex = True)
    offset_list = []; hp_list = []
    for offset_name in offset_dict.keys():
        offset = int(offset_name.split("_")[1])
        num_flows = len(offset_dict[offset_name]["daughter1_flow"])
        ax2.scatter(offset_dict[offset_name]["daughter1_length_star"], offset_dict[offset_name]["daughter1_R_lin"], color = "k")
        ax3.scatter(offset_dict[offset_name]["daughter1_length_star"], offset_dict[offset_name]["daughter1_R_quad"], color = "k")
        offset_list.append(offset_dict[offset_name]["daughter1_length_star"])
        hp_list.append(offset_dict[offset_name]["daughter1_length_star"] * 8 * np.pi *0.04 / (offset_dict[offset_name]["daughter1_area"]**2))
        for flow_ind in range(len(offset_dict[offset_name]["daughter1_flow"])):
            ax1.scatter(offset_dict[offset_name]["daughter1_length_star"], offset_dict[offset_name]["daughter1_dP"][flow_ind]/1333, color = colors[flow_ind])
    ax1.set_ylabel("dP (in - out) (mmHg)")
    ax1.legend([f"Re = {offset_dict[offset_name]['daughter1_Re'][i]}" for i in range(num_flows)])
    ax1.set_title(f"Geometry: {geo} \n\
        Inlet Area = {offset_dict[offset_name]['A_char']} cm^2, \n\
        Daughter Area Ratio = {offset_dict[offset_name]['daughter1_area_ratio']}, Auxilliary Area Ratio = {offset_dict[offset_name]['daughter2_area_ratio']} \n\
        Daughter Angle = {offset_dict[offset_name]['daughter1_angle']}, Auxilliary Angle = {offset_dict[offset_name]['daughter2_angle']}",
        fontsize = 8)
    ax2.plot(offset_list, hp_list, "--", color = "k", label = "Hagen-Poiseuille Law")
    ax2.set_ylabel("Linear Resistance")
    ax3.set_ylabel("Quadratic Resistance")
    ax3.set_xlabel("Junction + Branch Length (normalized)")
    ax2.legend()
    fig.savefig(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}/resistances.png")

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize = (10,10), sharex = True)
    offset_list = []; hp_list = []
    for offset_name in offset_dict.keys():
        num_flows = len(offset_dict[offset_name]["daughter2_flow"])
        offset = int(offset_name.split("_")[1])
        ax2.scatter(offset_dict[offset_name]["daughter2_length_star"], offset_dict[offset_name]["daughter2_R_lin"], color = "k")
        ax3.scatter(offset_dict[offset_name]["daughter2_length_star"], offset_dict[offset_name]["daughter2_R_quad"], color = "k")
        offset_list.append(offset_dict[offset_name]["daughter2_length_star"])
        hp_list.append(offset_dict[offset_name]["daughter2_length_star"] * 8 * np.pi *0.04 / (offset_dict[offset_name]["daughter2_area"]**2))
        for flow_ind in range(len(offset_dict[offset_name]["daughter2_flow"])):
            ax1.scatter(offset_dict[offset_name]["daughter2_length_star"], offset_dict[offset_name]["daughter2_dP"][flow_ind]/1333, color = colors[flow_ind])
    ax1.set_ylabel("dP (in - out) (mmHg)")
    ax1.legend([f"Re = {offset_dict[offset_name]['daughter2_Re'][i]}" for i in range(num_flows)])
    ax1.set_title(f"Geometry: {geo} \n\
        Inlet Area = {offset_dict[offset_name]['A_char']} cm^2, \n\
        Daughter Area Ratio = {offset_dict[offset_name]['daughter1_area_ratio']}, Auxilliary Area Ratio = {offset_dict[offset_name]['daughter2_area_ratio']} \n\
        Daughter Angle = {offset_dict[offset_name]['daughter1_angle']}, Auxilliary Angle = {offset_dict[offset_name]['daughter2_angle']}",
        fontsize = 8)
    ax2.plot(offset_list, hp_list, "--", color = "k", label = "Hagen-Poiseuille Law")
    ax2.set_ylabel("Linear Resistance")
    ax3.set_ylabel("Quadratic Resistance")
    ax3.set_xlabel("Junction + Branch Length (normalized)")
    ax2.legend()
    fig.savefig(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}/resistances2.png")
    return

def extract_steady_flow_data(anatomy, set_type, require4):
    
    CCO_data_dict = {}
    geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}"); geos.sort(); print(f"Geometries: {geos}")

    for j, geo in enumerate(geos[0:]):
        if "DS" in geo:
            continue
        geo_results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}"
        geo_dict = {}
        for offset in range(1,10):
            try:
                offset_dict = extract_flow_behavior(geo_results_dir, offset)
                geo_dict[f"offset_{int(10*offset)}"] = offset_dict
                
            except Exception as error:
                # handle the exception
                print("An exception occurred:", type(error).__name__)
                print(error) 
                print(f"Could not extract steady data from {geo}, offset {offset}.")
                
                continue
        if len(geo_dict.keys()) > 0:
            plot_geo(geo_dict, anatomy, set_type, geo)
        CCO_data_dict[geo] = geo_dict

    if not os.path.exists(f"data/data_dicts"):
        os.makedirs(f"data/data_dicts")
    save_dict(CCO_data_dict, f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    #pdb.set_trace()
    return