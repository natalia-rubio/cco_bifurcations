from cProfile import label
import re
import sys
import os
import time

import pandas as pd
from pyparsing import line

sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from fpdf import FPDF
from util.tools.junction_proc import get_angle_diff
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import r2_score
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 10
plt.rcParams.update({
         "text.usetex": True,
         "font.family": "serif",
         "font.serif": ["Computer Modern"]
     })
from scipy.interpolate import CubicSpline
def extract_flow_behavior_unsteady(geo_results_dir, offset):
    
    verbose = True
    daughter1_dPs = []
    daughter2_dPs = []
    inlet_flows = []
    daughter1_flows = []
    daughter2_flows = []
    offset_dict = {}
    re_char = 4500
    # Compose lists of flow and pressure data for each outlet
    try:
        flow_result_dir = f"{geo_results_dir}/flow_unsteady_offset_{int(10*offset)}_red_sol"
        if not os.path.exists(flow_result_dir):
            print(f"Could not find {flow_result_dir}.")
            raise ValueError(f"Could not find {flow_result_dir}.")
        
        soln_dict = load_dict(flow_result_dir)

        areas = np.asarray(soln_dict["areas"]).reshape((-1, 3))
        pressure = np.asarray(soln_dict["pressure_in_time"])
        flow = np.asarray(soln_dict["flow_in_time"])
        times = np.asarray(soln_dict["times"])

        # time derivatives
        
        time_steps = times[1:,0] - times[:-1,0]
        if np.any(time_steps != 25):
            print(f"Time steps are not constant in {flow_result_dir}.")
            #raise ValueError(f"Time steps are not constant in {flow_result_dir}.")
        #pdb.set_trace()
        dt = 0.001 # convert to seconds, assuming 2*0.4 is the time step in ms
        #print(f"dt: {dt}")
        dflow_dt = np.zeros_like(flow)
        #dflow_dt[1:-1,:] = (flow[2:,:] - flow[:-2,:])/(2*dt) # central difference
        cd_time_steps = np.tile((time_steps[0:-1]+time_steps[1:]),(3,1)).T
        #pdb.set_trace()
        dflow_dt[1:-1,:] = (flow[2:,:] - flow[:-2,:])/(cd_time_steps*dt) # central difference with variable time step
        
        # dflow_dt[1:,:] = (flow[1:,:] - flow[:-1,:])/(dt) # central difference
        
        #flow = flow[1:-1,:] # remove first and last time step
        #pressure = pressure[1:-1,:] # remove first and last time step
        #times = times[1:-1,:] # remove first and last time step
        #dflow_dt = dflow_dt[1:-1,:] # remove first and last time step
        dflow_dt[0,:] = (flow[1,:] - flow[0,:])/((times[1][0] - times[0][0]) * 2*0.4/800) # forward difference for first time step
        dflow_dt[-1,:] = (flow[-1,:] - flow[-2,:])/((times[-1][0] - times[-2][0]) * 2*0.4/800) # backward difference for last time step
        #dflow_dt[0,:] = dflow_dt[1,:]# forward difference for first time step
        #dflow_dt[-1,:] = dflow_dt[-2,:] # backward difference for last time step
        #pdb.set_trace()
        flow = flow[1:-1,:] # remove first and last time step
        pressure = pressure[1:-1,:] # remove first and last time step
        times = times[1:-1,:] # remove first and last time step
        dflow_dt = dflow_dt[1:-1,:] # remove first and last time step

        tangents = np.asarray(soln_dict["tangents"])
        lengths = np.asarray(soln_dict["lengths"]).reshape((-1, 2))
        
        
        if verbose:
            print(f"Extracted flow data from {flow_result_dir}.")

    except:
        print(f"Could not extract flow data from {geo_results_dir}.")
        return


    A_char = areas[0,0]
    L_char = np.sqrt(A_char/np.pi)
    U_char = re_char * 0.04/(1.06 * 2*np.sqrt(A_char/np.pi))
    t_char = L_char/U_char

    flow_ratio1 = np.median(flow[:,1]/flow[:,0])
    flow_ratio2 = np.median(flow[:,2]/flow[:,0])


    flow_star = flow/(U_char * A_char)
    dflow_dt_star = dflow_dt/(U_char * A_char/t_char)
    p_star = pressure/(1.06 * U_char**2)


    dP_vec = np.hstack([pressure[:,0]-pressure[:,1],
                        pressure[:,0]-pressure[:,2]])
    dP_vec_star = np.hstack([p_star[:,0]-p_star[:,1],
                             p_star[:,0]-p_star[:,2]])
    
    num_flows = flow.shape[0]
    num_coefs = 6
    A_mat_star = np.zeros((2*num_flows, num_coefs))
    A_mat = np.zeros((2*num_flows, num_coefs))
    
    # Linear resistors
    A_mat[0:num_flows,0] = flow[:,1]
    A_mat[num_flows:2*num_flows,3] = flow[:,2]
    A_mat_star[0:num_flows,0] = flow_star[:,1]
    A_mat_star[num_flows:2*num_flows,3] = flow_star[:,2]

    # Quadratic resistors
    A_mat[0:num_flows,1] = np.square(flow[:,1])
    A_mat[num_flows:2*num_flows,4] = np.square(flow[:,2])
    A_mat_star[0:num_flows,1] = np.square(flow_star[:,1])
    A_mat_star[num_flows:2*num_flows,4] = np.square(flow_star[:,2])

    # Inductors
    A_mat[0:num_flows,2] = dflow_dt[:,1]
    A_mat[num_flows:2*num_flows,5] = dflow_dt[:,2]
    A_mat_star[0:num_flows,2] = dflow_dt_star[:,1]
    A_mat_star[num_flows:2*num_flows,5] = dflow_dt_star[:,2]

    # Solve
    coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
    
    R_lin_star1         = coefs_star[0]
    R_quad_star1        = coefs_star[1]
    L_star1             = coefs_star[2]
    R_lin_star2         = coefs_star[3]
    R_quad_star2        = coefs_star[4]
    L_star2             = coefs_star[5]

    if R_lin_star1 < 0:
            # Linear resistors
       # raise ValueError(f"Linear resistance for daughter 1 is negative: {R_lin_star1}.")
        A_mat[0:num_flows,0] = flow[:,1]*0
        A_mat_star[0:num_flows,0] = flow_star[:,1]*0
        R_lin_star1 = 0
        
    if R_lin_star2 < 0:
            # Linear resistors
        #raise ValueError(f"Linear resistance for daughter 2 is negative: {R_lin_star2}.")
        A_mat[num_flows:2*num_flows,3] = flow[:,2]*0
        A_mat_star[num_flows:2*num_flows,3] = flow_star[:,2]*0
        R_lin_star2 = 0
     
        # Solve
    coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
    

    R_quad_star1        = coefs_star[1]
    L_star1             = coefs_star[2]
    R_quad_star2        = coefs_star[4]
    L_star2             = coefs_star[5]
   
    offset_dict["daughter1_R_lin_star"]     = copy.copy(R_lin_star1)
    #offset_dict["daughter1_R_lin_star_log"] = np.log(np.abs(np.max((0, offset_dict["daughter1_R_lin_star"]))))
    offset_dict["daughter2_R_lin_star"]     = copy.copy(R_lin_star2)
    #offset_dict["daughter2_R_lin_star_log"] = np.log(np.abs(np.max((0, offset_dict["daughter2_R_lin_star"]))))

    offset_dict["daughter1_R_quad_star"]    = copy.copy(R_quad_star1)
    #offset_dict["daughter1_R_quad_star_log"] = np.sign(offset_dict["daughter1_R_quad_star"]) * np.max((np.log(np.abs(offset_dict["daughter1_R_quad_star"]))+4, 0))
    offset_dict["daughter2_R_quad_star"]    = copy.copy(R_quad_star2)
    #offset_dict["daughter2_R_quad_star_log"] = np.sign(offset_dict["daughter2_R_quad_star"]) * np.max((np.log(np.abs(offset_dict["daughter2_R_quad_star"]))+4, 0))

    offset_dict["daughter1_L_star"]         = copy.copy(L_star1)
    offset_dict["daughter2_L_star"]         = copy.copy(L_star2)

    # Solve
    coefs, residuals, t, q = np.linalg.lstsq(A_mat, dP_vec, rcond=None)
    
    residuals = (dP_vec - A_mat @ coefs) / np.max(dP_vec)
    r2_rri = r2_score(dP_vec, A_mat @ coefs)
    print(f"Residuals: {np.mean(residuals)}")
    residuals_avg = np.mean(residuals)
    if np.abs(np.mean(residuals)) > 0.1:
        print(f"Residuals are too high: {np.mean(residuals)}.")
        residuals_avg = np.mean(residuals)
        pdb.set_trace()
        #return None

    R_lin1         = coefs[0]
    R_quad1        = coefs[1]
    L1             = coefs[2]
    R_lin2         = coefs[3]
    R_quad2        = coefs[4]
    L2             = coefs[5]
    
    if R_lin_star1 < 0:
        R_lin1 = 0
    if R_lin_star2 < 0:
        R_lin2 = 0

    offset_dict["daughter1_R_lin"]      = copy.copy(R_lin1)
    #offset_dict["daughter1_R_lin_log"] = np.log(np.abs(np.max((0,offset_dict["daughter1_R_lin"]))))
    offset_dict["daughter2_R_lin"]      = copy.copy(R_lin2)
    #offset_dict["daughter2_R_lin_log"] = np.log(np.abs(np.max((0,offset_dict["daughter2_R_lin"]))))

    offset_dict["daughter1_R_quad"]     = copy.copy(R_quad1)
    #offset_dict["daughter1_R_quad_log"]     = np.sign(offset_dict["daughter1_R_quad"]) * np.max((np.log(np.abs(offset_dict["daughter1_R_quad"]))+4, 0))
    offset_dict["daughter2_R_quad"]     = copy.copy(R_quad2)
    #offset_dict["daughter2_R_quad_log"]     = np.sign(offset_dict["daughter2_R_quad"]) * np.max((np.log(np.abs(offset_dict["daughter2_R_quad"]))+4, 0))

    offset_dict["daughter1_L"]          = copy.copy(L1)
    offset_dict["daughter2_L"]          = copy.copy(L2)
    
    tol = 0.02
    assert abs(offset_dict["daughter1_R_lin"]   - offset_dict["daughter1_R_lin_star"]*1.06*U_char/A_char)   < tol; "Daughter 1 linear resistances do not match."
    assert abs(offset_dict["daughter2_R_lin"]   - offset_dict["daughter2_R_lin_star"]*1.06*U_char/A_char)   < tol; "Daughter 2 linear resistances do not match."
    assert abs(offset_dict["daughter1_R_quad"]  - offset_dict["daughter1_R_quad_star"]*1.06/A_char**2)      < tol; "Daughter 1 quadratic resistances do not match."
    assert abs(offset_dict["daughter2_R_quad"]  - offset_dict["daughter2_R_quad_star"]*1.06/A_char**2)      < tol; "Daughter 2 quadratic resistances do not match."
    assert abs(offset_dict["daughter1_L"]       - offset_dict["daughter1_L_star"]*1.06*L_char/A_char)       < tol; "Daughter 1 inductances do not match."
    assert abs(offset_dict["daughter2_L"]       - offset_dict["daughter2_L_star"]*1.06*L_char/A_char)       < tol; "Daughter 2 inductances do not match."
    
    # ONLY LINEAR RESISTANCE
    num_flows = flow.shape[0]
    num_coefs = 4
    A_mat_star = np.zeros((2*num_flows, num_coefs))
    A_mat = np.zeros((2*num_flows, num_coefs))

    #pdb.set_trace()
    # Linear resistors
    A_mat[0:num_flows,0] = flow[:,1]
    A_mat[num_flows:2*num_flows,2] = flow[:,2]
    A_mat_star[0:num_flows,0] = flow_star[:,1]
    A_mat_star[num_flows:2*num_flows,2] = flow_star[:,2]

    # Inductors
    A_mat[0:num_flows,1] = dflow_dt[:,1]
    A_mat[num_flows:2*num_flows,3] = dflow_dt[:,2]
    A_mat_star[0:num_flows,1] = dflow_dt_star[:,1]
    A_mat_star[num_flows:2*num_flows,3] = dflow_dt_star[:,2]

    # Solve
    coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
    
    R_lin_star1_m2         = coefs_star[0]
    L_star1_m2             = coefs_star[1]
    R_lin_star2_m2         = coefs_star[2]
    L_star2_m2             = coefs_star[3]


    offset_dict["daughter1_R_lin_star_m2"]     = copy.copy(R_lin_star1_m2)
    offset_dict["daughter2_R_lin_star_m2"]     = copy.copy(R_lin_star2_m2)
    offset_dict["daughter1_L_star_m2"]         = copy.copy(L_star1_m2)
    offset_dict["daughter2_L_star_m2"]         = copy.copy(L_star2_m2)

    # Solve
    coefs, residuals, t, q = np.linalg.lstsq(A_mat, dP_vec, rcond=None)
    residuals = (dP_vec - A_mat @ coefs) / np.max(dP_vec)
    r2_ri = r2_score(dP_vec, A_mat @ coefs)
    residuals_avg_m2 = np.mean(residuals)
    print(f"Residuals: {np.mean(residuals)}")
    if np.abs(np.mean(residuals)) > 0.1:
        print(f"Residuals are too high: {np.mean(residuals)}.")
        
        #return None

    R_lin1_m2         = coefs[0]
    L1_m2             = coefs[1]
    R_lin2_m2         = coefs[2]
    L2_m2             = coefs[3]

    offset_dict["daughter1_R_lin_m2"]      = copy.copy(R_lin1_m2)
    offset_dict["daughter2_R_lin_m2"]      = copy.copy(R_lin2_m2)

    offset_dict["daughter1_L_m2"]          = copy.copy(L1_m2)
    offset_dict["daughter2_L_m2"]          = copy.copy(L2_m2)

    if verbose:
        print("Solved for resistances.")

    # Check consistency of non-dimensionalization
    tol = 0.02
    assert abs(offset_dict["daughter1_R_lin_m2"]   - offset_dict["daughter1_R_lin_star_m2"]*1.06*U_char/A_char)   < tol; "Daughter 1 linear resistances do not match."
    assert abs(offset_dict["daughter2_R_lin_m2"]   - offset_dict["daughter2_R_lin_star_m2"]*1.06*U_char/A_char)   < tol; "Daughter 2 linear resistances do not match."
    assert abs(offset_dict["daughter1_L_m2"]       - offset_dict["daughter1_L_star_m2"]*1.06*L_char/A_char)       < tol; "Daughter 1 inductances do not match."
    assert abs(offset_dict["daughter2_L_m2"]       - offset_dict["daughter2_L_star_m2"]*1.06*L_char/A_char)       < tol; "Daughter 2 inductances do not match."

    if verbose:
        print("Passed non-dimensionalization consistency check.")

    offset_dict["flow"] = flow
    offset_dict["dflow_dt"] = dflow_dt
    offset_dict["pressure"] = pressure
    offset_dict["dp1"] = pressure[:,0] - pressure[:,1]
    offset_dict["dp2"] = pressure[:,0] - pressure[:,2]
    
    offset_dict["times"] = times * 0.001

    offset_dict["daughter1_flow_ratio"] = flow_ratio1
    offset_dict["daughter2_flow_ratio"] = flow_ratio2
    offset_dict["daughter1_flow_ratio_inv"] = flow_ratio1**-1
    offset_dict["daughter2_flow_ratio_inv"] = flow_ratio2**-1

    offset_dict["U_char"] = U_char
    offset_dict["A_char"] = A_char
    offset_dict["L_char"] = L_char
    
    offset_dict["daughter1_length_star"] = lengths[0,0]/L_char
    offset_dict["daughter1_length_star_sq"] = offset_dict["daughter1_length_star"]**2
    offset_dict["daughter2_length_star"] = lengths[0,1]/L_char
    offset_dict["daughter2_length_star_sq"] = offset_dict["daughter2_length_star"]**2
    
    offset_dict["daughter1_length"] = lengths[0,0]
    offset_dict["daughter2_length"] = lengths[0,1]

    offset_dict["daughter1_angle"] = get_angle_diff(tangents[0,:,1], tangents[0,:,0])[0]
    offset_dict["daughter2_angle"] = get_angle_diff(tangents[0,:,2], tangents[0,:,0])[0]

    offset_dict["daughter1_area_ratio"] = areas[0,1]/A_char
    offset_dict["daughter1_outlet_area"] = areas[0,1]
    offset_dict["daughter2_area_ratio"] = areas[0,2]/A_char
    offset_dict["daughter2_outlet_area"] = areas[0,2]
    total_daughter_area_ratio = areas[0,1]/A_char + areas[0,2]/A_char
    offset_dict["total_daughter_area_ratio"] = total_daughter_area_ratio
    if total_daughter_area_ratio > 1.7 or total_daughter_area_ratio < 0.4:
        raise ValueError(f"Total daughter area ratio is too high: {total_daughter_area_ratio}.")
    if offset_dict["daughter1_area_ratio"] > 1.2 or offset_dict["daughter2_area_ratio"] > 1.2:
        raise ValueError(f"Daughter area ratios are too high: {offset_dict['daughter1_area_ratio']}, {offset_dict['daughter2_area_ratio']}.")
    if offset_dict["daughter1_angle"] > 1.8 or offset_dict["daughter2_angle"] > 1.8:
        raise ValueError(f"Daughter angles are too high: {offset_dict['daughter1_angle']}, {offset_dict['daughter2_angle']}.")
    if offset_dict["daughter1_flow_ratio"] > 0.95 or offset_dict["daughter2_flow_ratio"] > 0.95:
        raise ValueError(f"Daughter flow ratios are too high: {offset_dict['daughter1_flow_ratio']}, {offset_dict['daughter2_flow_ratio']}.")
    # if offset_dict["daughter1_length_star"] > 35 or offset_dict["daughter2_length_star"] > 35:
    #     raise ValueError(f"Daughter length stars are too low: {offset_dict['daughter1_length_star']}, {offset_dict['daughter2_length_star']}.")
    if offset_dict["daughter1_R_quad_star"] < -10 or offset_dict["daughter2_R_quad_star"] < -10:
        raise ValueError(f"Quadratic resistance is too low: {offset_dict['daughter1_R_quad_star']}.")
    if offset_dict["daughter2_R_quad_star"] > 10 and offset_dict["daughter2_area_ratio"] > 0.4:
        raise ValueError(f"Daughter 2 quadratic resistance is too high: {offset_dict['daughter2_R_quad_star']}.")
    if offset_dict["daughter1_R_quad_star"] > 10 and offset_dict["daughter1_area_ratio"] > 0.4:
        raise ValueError(f"Daughter 2 quadratic resistance is too high: {offset_dict['daughter2_R_quad_star']}.")
    # if offset_dict["daughter1_flow_ratio"] < 0.1 or offset_dict["daughter2_flow_ratio"] < 0.1:
    #     raise ValueError(f"Daughter flow ratios are too low: {offset_dict['daughter1_flow_ratio']}, {offset_dict['daughter2_flow_ratio']}.")
    if A_char < 0.8 or A_char > 1.3:
        raise ValueError(f"A_char is too low or too high: {A_char}.")

    offset_dict["daughter1_area"] = areas[0,1]
    offset_dict["daughter2_area"] = areas[0,2]

    offset_dict["daughter1_area_ratio_inv2"] = (A_char/areas[0,1])**2
    offset_dict["daughter2_area_ratio_inv2"] = (A_char/areas[0,2])**2
    offset_dict["total_area_ratio_inv2"] = (A_char/(areas[0,1] + areas[0,2]))**2
    print(f"Extracted flow behavior for offset {offset} in geometry {geo_results_dir}.")
    #pdb.set_trace()
    return offset_dict, (r2_rri, r2_ri)

def plot_geo(geo_dict, anatomy, set_type, geo):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.size'] = 10
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

    for offset_name, offset_dict in geo_dict.items():
        colors = ['b', 'g', 'y', 'r',"orange", "c", "m", "k"]
        ms = 20
        fig, axs = plt.subplots(3,2, width_ratios=[1.5, 1])
        fig.set_size_inches(7, 7)
        ax1 = axs[0,0]
        offset_list = []; hp_list = []
        offset = int(offset_name.split("_")[1])
        
        R_poiseuille1 = 8 * 0.04 * np.pi * offset_dict["daughter1_length"] / offset_dict["daughter1_area"]**2
        R_poiseuille2 = 8 * 0.04 * np.pi * offset_dict["daughter2_length"] / offset_dict["daughter2_area"]**2
        ind1 = offset_dict["daughter1_L"] * 1.06 / offset_dict["daughter1_area"]
        ind2 = offset_dict["daughter2_L"] * 1.06 / offset_dict["daughter2_area"]
        
        def flow_to_re(flow):
            u = flow/offset_dict["A_char"]
            d = 2 * np.sqrt(offset_dict["A_char"]/np.pi)
            return 1.06 * u * d / 0.04
        def re_to_flow(re):
            d = 2 * np.sqrt(offset_dict["A_char"]/np.pi)
            u = re * 0.04 / (1.06 * d)
            return u * offset_dict["A_char"]
        
        times_arr = np.asarray(offset_dict["times"])[:,0]
        flow_arr = np.asarray(offset_dict["flow"])[:,0]
        flow_arr1 = np.asarray(offset_dict["flow"])[:,1]
        flow_arr2 = np.asarray(offset_dict["flow"])[:,2]

        cs = CubicSpline(times_arr, flow_arr)
        times_arr_fine = np.linspace(times_arr[0], times_arr[-1], 1000)
        flow_arr_fine = cs(times_arr_fine)
        flow_arr_fine1 = interp1d(times_arr, flow_arr1, kind='linear')(times_arr_fine)
        flow_arr_fine2 = interp1d(times_arr, flow_arr2, kind='linear')(times_arr_fine)
        
        dflow_dt_arr = offset_dict["dflow_dt"][:,0]
        dflow_dt_arr1 = offset_dict["dflow_dt"][:,1]
        dflow_dt_arr2 = offset_dict["dflow_dt"][:,2]
        dflow_dt_arr_fine = interp1d(times_arr, dflow_dt_arr, kind='linear')(times_arr_fine)
        dflow_dt_arr_fine1 = interp1d(times_arr, dflow_dt_arr1, kind='linear')(times_arr_fine)
        dflow_dt_arr_fine2 = interp1d(times_arr, dflow_dt_arr2, kind='linear')(times_arr_fine)

        dp1_arr = offset_dict["dp1"]
        dp2_arr = offset_dict["dp2"]
        dp1_arr_fine = flow_arr_fine1 * offset_dict["daughter1_R_lin"] + offset_dict["daughter1_R_quad"] * np.square(flow_arr_fine1) + offset_dict["daughter1_L"] * dflow_dt_arr_fine1
        dp2_arr_fine = flow_arr_fine2 * offset_dict["daughter2_R_lin"] + offset_dict["daughter2_R_quad"] * np.square(flow_arr_fine2) + offset_dict["daughter2_L"] * dflow_dt_arr_fine2  
        dp1_arr_fine_stan = flow_arr_fine1 * R_poiseuille1 + ind1 * dflow_dt_arr_fine1
        dp2_arr_fine_stan = flow_arr_fine2 * R_poiseuille2 + ind2 * dflow_dt_arr_fine2
        dp1_arr_fine_stan_tp = dp1_arr_fine_stan + 0.5*1.06*((flow_arr_fine1/offset_dict["daughter1_outlet_area"])**2 - (flow_arr_fine/offset_dict["A_char"])**2)
        dp2_arr_fine_stan_tp = dp2_arr_fine_stan + 0.5*1.06*((flow_arr_fine2/offset_dict["daughter2_outlet_area"])**2 - (flow_arr_fine/offset_dict["A_char"])**2)
        
        ax2 = ax1.twinx()

        ax1.scatter(offset_dict["times"], offset_dict["dp1"]/1333, color = "orange", s = ms, label = "Outlet 1 $\Delta P$ (Simulation)")
        ax1.plot(times_arr_fine, dp1_arr_fine/1333, color = "orange", label = "Outlet 1 $\Delta P$ (RRI Fit)")
        #ax1.plot(times_arr_fine, dp1_arr_fine_stan/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Standard 0D)", linestyle = "--")
        #ax1.plot(times_arr_fine, dp1_arr_fine_stan_tp/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Standard 0D TP)", linestyle = "-.")
        ax1.scatter(offset_dict["times"], offset_dict["dp2"]/1333, color = "mediumpurple", s = ms,  label = "Outlet 2 $\Delta P$")
        ax1.plot(times_arr_fine, dp2_arr_fine/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (RRI Fit)")
        #ax1.plot(times_arr_fine, dp2_arr_fine_stan/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Standard 0D)", linestyle = "--")
        #ax1.plot(times_arr_fine, dp2_arr_fine_stan_tp/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Standard 0D TP)", linestyle = "-.")

        #ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("RRI Fit \n \n $\Delta P$ (mmHg)")

        ax2.plot(times_arr_fine, flow_arr_fine, "--", color = "black", label = "Inlet Flow")

        ax2.set_ylabel("Inlet Flow (cm$ ^3$/s)", rotation=270, labelpad=15)
        #labels = ["Outlet 1 $\Delta P$ (Simulation)", "Outlet 1 $\Delta P$ (Fit)", "Outlet 2 $\Delta P$ (Simulation)", "Outlet 2 $\Delta P$ (Fit)", "Inlet Flow"]


        flow_steady1 = flow_arr_fine1[230:500]
        flow_steady2 = flow_arr_fine2[230:500]
        flow_steady = flow_arr_fine[230:500]
        
        dp_steady1 = dp1_arr - offset_dict["daughter1_L"] * dflow_dt_arr1
        dp_steady_calc = flow_steady1 * offset_dict["daughter1_R_lin"] + \
                        offset_dict["daughter1_R_quad"] * np.square(flow_steady1)
        dp_steady2 = dp2_arr - offset_dict["daughter2_L"] * dflow_dt_arr2
        dp_steady_calc2 = flow_steady2 * offset_dict["daughter2_R_lin"] + \
                        offset_dict["daughter2_R_quad"] * np.square(flow_steady2)
        

        dp1_arr_fine_steady = flow_steady1 * R_poiseuille1
        dp1_arr_fine_steady_tp = dp1_arr_fine_steady + 0.5*1.06*((flow_steady1/offset_dict["daughter1_outlet_area"])**2 - (flow_steady/offset_dict["A_char"])**2)
        dp2_arr_fine_steady = flow_steady2 * R_poiseuille2
        dp2_arr_fine_steady_tp = dp2_arr_fine_steady + 0.5*1.06*((flow_steady2/offset_dict["daughter2_outlet_area"])**2 - (flow_steady/offset_dict["A_char"])**2)
        ax3 = axs[0,1]
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax3.set_xticks([])
        

        ax3.scatter(flow_arr, dp_steady1/1333, color = "orange", s = ms, label = "Outlet 1 Steady $\Delta P$")
        ax3.plot(flow_steady, dp_steady_calc/1333, color = "orange")
        #ax3.plot(flow_steady, dp1_arr_fine_steady/1333, color = "orange", linestyle = "--", label = "Outlet 1 Steady $\Delta P$ (Standard 0D)")
        #ax3.plot(flow_steady, dp1_arr_fine_steady_tp/1333, color = "orange", linestyle = "-.", label = "Outlet 1 Steady $\Delta P$ (Standard 0D TP)")
        ax3.scatter(flow_arr, dp_steady2/1333, color = "mediumpurple", s = ms, label = "Outlet 2 Steady $\Delta P$")
        ax3.plot(flow_steady, dp_steady_calc2/1333, color = "mediumpurple")
        #ax3.plot(flow_steady, dp2_arr_fine_steady/1333, color = "mediumpurple", linestyle = "--", label = "Outlet 2 Steady $\Delta P$ (Standard 0D)")
        #ax3.plot(flow_steady, dp2_arr_fine_steady_tp/1333, color = "mediumpurple", linestyle = "-.", label = "Outlet 2 Steady $\Delta P$ (Standard 0D TP)")
        
        #ax3.set_xlabel("Inlet Flow (cm$ ^3$/s)")
        ax3.set_ylabel("$\Delta P_{\mathrm{steady}}$ (mmHg)")
        secax_3 = ax3.secondary_xaxis('top', xlabel = "Reynolds Number", functions = (flow_to_re, re_to_flow))





        ax4 = axs[1,0]
        dp1_arr_fine_m2 = flow_arr_fine1 * offset_dict["daughter1_R_lin_m2"] + + offset_dict["daughter1_L_m2"] * dflow_dt_arr_fine1
        dp2_arr_fine_m2 = flow_arr_fine2 * offset_dict["daughter2_R_lin_m2"] + + offset_dict["daughter2_L_m2"] * dflow_dt_arr_fine2  
        
        ax5 = ax4.twinx()

        ax4.scatter(offset_dict["times"], offset_dict["dp1"]/1333, color = "orange", s = ms, label = "Outlet 1 $\Delta P$ (Simulation)")
        ax4.scatter(offset_dict["times"], offset_dict["dp2"]/1333, color = "mediumpurple", s = ms,  label = "Outlet 2 $\Delta P$")
        ax4.plot(times_arr_fine, dp1_arr_fine_m2/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Fit)")
        #ax4.plot(times_arr_fine, dp1_arr_fine_stan/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Standard 0D)", linestyle = "--")
        
        ax4.plot(times_arr_fine, dp2_arr_fine_m2/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Fit)")
        #ax4.plot(times_arr_fine, dp2_arr_fine_stan/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Standard 0D)", linestyle = "--")

        
        ax4.set_ylabel("RI Fit \n\n $\Delta P$ (mmHg)")

        ax5.plot(times_arr_fine, flow_arr_fine, "--", color = "black", label = "Inlet Flow")
        ax5.set_ylabel("Inlet Flow (cm$ ^3$/s)", rotation=270, labelpad=15)
        #labels = ["Outlet 1 $\Delta P$ (Simulation)", "Outlet 1 $\Delta P$ (RI Fit)", "Outlet 2 $\Delta P$ (Simulation)", "Outlet 2 $\Delta P$ (RI Fit)", "Inlet Flow"]
        

        dp_steady1_m2       = dp1_arr - offset_dict["daughter1_L_m2"] * dflow_dt_arr1
        dp_steady_calc_m2   = flow_steady1 * offset_dict["daughter1_R_lin_m2"]
        dp_steady2_m2       = dp2_arr - offset_dict["daughter2_L_m2"] * dflow_dt_arr2
        dp_steady_calc2_m2  = flow_steady2* offset_dict["daughter2_R_lin_m2"]
        ax6 = axs[1,1]

        ax6.scatter(flow_arr, dp_steady1_m2/1333, color = "orange", s = ms, label = "Outlet 1 Steady $\Delta P$")
        ax6.plot(flow_steady, dp_steady_calc_m2/1333, color = "orange")
        #ax6.plot(flow_steady, dp1_arr_fine_steady/1333, color = "orange", linestyle = "--", label = "Outlet 1 Steady $\Delta P$ (Standard 0D)")
        ax6.scatter(flow_arr, dp_steady2_m2/1333, color = "mediumpurple", s = ms, label = "Outlet 2 Steady $\Delta P$")
        ax6.plot(flow_steady, dp_steady_calc2_m2/1333, color = "mediumpurple")
        #ax6.plot(flow_steady , dp2_arr_fine_steady/1333, color = "mediumpurple", linestyle = "--", label = "Outlet 2 Steady $\Delta P$ (Standard 0D)")
        
        ax6.set_ylabel("$\Delta P_{\mathrm{steady}}$ (mmHg)")
        ax4.set_xticks([])
        ax5.set_xticks([])
        ax6.set_xticks([])
        
        
        ax7 = axs[2,0]
        offset_list = []; hp_list = []
        offset = int(offset_name.split("_")[1])
        ax8 = ax7.twinx()
        ax9 = axs[2,1]

        ax7.scatter(offset_dict["times"], offset_dict["dp1"]/1333, color = "orange", s = ms, label = "Outlet 1 $\Delta P$ (Simulation)")
        ax7.scatter(offset_dict["times"], offset_dict["dp2"]/1333, color = "mediumpurple", s = ms,  label = "Outlet 2 $\Delta P$")
        ax7.plot(times_arr_fine, dp1_arr_fine_stan/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Standard 0D)", linestyle = "--")
        ax7.plot(times_arr_fine, dp1_arr_fine_stan_tp/1333, color = "orange", label = "Outlet 1 $\Delta P$ (Standard 0D TP)", linestyle = ":")
        
        ax7.plot(times_arr_fine, dp2_arr_fine_stan/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Standard 0D)", linestyle = "--")
        ax7.plot(times_arr_fine, dp2_arr_fine_stan_tp/1333, color = "mediumpurple", label = "Outlet 2 $\Delta P$ (Standard 0D TP)", linestyle = ":")

        ax7.set_ylabel("Standard \n \n $\Delta P$ (mmHg)")
        ax7.set_xlabel("Time (s)")
        ax8.plot(times_arr_fine, flow_arr_fine, "--", color = "black", label = "Inlet Flow")
        ax8.set_ylabel("Inlet Flow (cm$ ^3$/s)", rotation=270, labelpad=15)

        ax9.scatter(flow_arr, dp_steady1/1333, color = "orange", s = ms, label = "Outlet 1 Steady $\Delta P$")
        ax9.scatter(flow_arr, dp_steady2/1333, color = "mediumpurple", s = ms, label = "Outlet 2 Steady $\Delta P$")
        ax9.plot(flow_steady, dp1_arr_fine_steady/1333, color = "orange", linestyle = "--", label = "Outlet 1 Steady $\Delta P$ (Standard 0D)")
        ax9.plot(flow_steady, dp1_arr_fine_steady_tp/1333, color = "orange", linestyle = ":", label = "Outlet 1 Steady $\Delta P$ (Standard 0D TP)")
        
        ax9.plot(flow_steady, dp2_arr_fine_steady/1333, color = "mediumpurple", linestyle = "--", label = "Outlet 2 Steady $\Delta P$ (Standard 0D)")
        ax9.plot(flow_steady, dp2_arr_fine_steady_tp/1333, color = "mediumpurple", linestyle = ":", label = "Outlet 2 Steady $\Delta P$ (Standard 0D TP)")
        
        #ax9.set_xlabel("Inlet Flow (cm$ ^3$/s)")
        ax9.set_ylabel("$\Delta P_{\mathrm{steady}}$ (mmHg)")
        ax9.set_xlabel("Inlet Flow (cm$ ^3$/s)")
        #ax7.legend(labels, bbox_to_anchor=(0.7, 3.6), loc='upper center', ncol = 2, frameon=False)
        #fig.tight_layout()
        ax1.tick_params(direction="in")
        ax2.tick_params(direction="in")
        ax3.tick_params(direction="in")
        ax4.tick_params(direction="in")
        ax5.tick_params(direction="in")
        ax6.tick_params(direction="in")
        ax7.tick_params(direction="in")
        ax8.tick_params(direction="in")
        ax9.tick_params(direction="in")
        
        labels = ["Outlet 1 (Simulation)", "Outlet 1 (Fit)", "Outlet 1 (Standard 0D)", "Outlet 1 (Standard 0D TP)",
            "Outlet 2 (Simulation)", "Outlet 2 (Fit)", "Outlet 2 (Standard 0D)", "Outlet 2 (Standard 0D TP)"]
        labels2 = ["Inlet Flow"]
        handles = ax1.get_legend_handles_labels()[0][0:2] + ax7.get_legend_handles_labels()[0][2:4] + ax1.get_legend_handles_labels()[0][2:4] + ax7.get_legend_handles_labels()[0][4:6]
        labels = ax1.get_legend_handles_labels()[1][0:2] + ax7.get_legend_handles_labels()[1][2:4] + ax1.get_legend_handles_labels()[1][2:4] + ax7.get_legend_handles_labels()[1][4:6]
        
        ax1.legend(handles, labels, bbox_to_anchor=(0.7, 1.8), loc='upper center', ncol = 2, frameon=False)
        ax2.legend(labels2, bbox_to_anchor=(1.8, 1.8), loc='upper center', ncol = 1, frameon=False)
        
        plt.subplots_adjust(wspace=0.5)
        plt.subplots_adjust(hspace=0.1)
        fig.savefig(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}/unsteady_plot_{offset_name}.pdf", bbox_inches='tight')

    return

def extract_unsteady_flow_data(anatomy, set_type, require4, num_offsets = 5):
    residuals_total = (0,0)
    residuals_cnt = 0
    CCO_data_dict = {}
    if set_type == "combined":
        set_type_list = ["dict_res_fs_ext", "dict_flat_3D_dim_ext"]
    else:
        set_type_list = [set_type]
    for set_type in set_type_list:
        geos = os.listdir(f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}"); geos.sort(); print(f"Geometries: {geos}")
        
        for j, geo in enumerate(geos[0:]):
            print(f"Processing geometry {geo}")
            if "DS" in geo:
                continue
            geo_results_dir = f"data/synthetic_junctions_reduced_results/{anatomy}/{set_type}/{geo}"
            geo_dict = {}
            for offset in range(10-num_offsets,10):
                try:
                    offset_dict, res = extract_flow_behavior_unsteady(geo_results_dir, offset)
                    
                    if offset_dict is None:
                        print(f"Could not extract flow behavior for offset {offset} in geometry {geo}.")
                        continue
                    residuals_total = (residuals_total[0] + res[0], residuals_total[1] + res[1])
                    residuals_cnt += 1
                    geo_dict[f"offset_{int(10*offset)}"] = offset_dict
                    
                    if offset == 7:
                        print(f"Plotting offset {offset} is the steady state for {geo}.")
                        #plot_geo(geo_dict, anatomy, set_type, geo)

                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", type(error).__name__)
                    print(error) 
                    print(f"Could not extract steady data from {geo}, offset {offset}.")
                    
                    continue
            if len(geo_dict.keys()) != num_offsets:
                print(f"Could not extract all offsets for geometry {geo}. Found {len(geo_dict.keys())} instead of {num_offsets}.")
                continue
            
            CCO_data_dict[f"{geo}_{set_type}"] = geo_dict

    if not os.path.exists(f"data/data_dicts"):
        os.makedirs(f"data/data_dicts")
    save_dict(CCO_data_dict, f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    #print(f"Average RRI residuals: {residuals_total[0]/residuals_cnt}, average RI residuals: {residuals_total[1]/residuals_cnt}")
    print(f"Average RRI R2: {residuals_total[0]/residuals_cnt}, average RI R2: {residuals_total[1]/residuals_cnt}")
    #pdb.set_trace()
    return