import json
import pdb
import sys
import os

import pandas as pd
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import *
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.get_0d_res_dict import get_input_file_junction_dict_master
from util.tree.centerline_proj import extract_results
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load 
#from util.zerod.standard_to_RR import check_out_of_dist

from fpdf import FPDF
import matplotlib.pyplot as plt
import io
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10
plt.rcParams['text.usetex']=True
colors = ["royalblue", "orangered", "seagreen", "peru", "blueviolet"]

def check_out_of_dist(param, param_name, scaling_dict):
    if param < scaling_dict[param_name][2]:
        print(f"{param_name} smaller than training set minimum: {param}, {scaling_dict[param_name][2]}")
        param = scaling_dict[param_name][2]
    if param > scaling_dict[param_name][3]:
        print(f"{param_name} larger than training set maximum: {param}, {scaling_dict[param_name][3]}")
        param = scaling_dict[param_name][3]
    return param

def get_recursive_resistance(junction_dict_master, junction_name):
    
    # Base Case
    if junction_dict_master[junction_name]["0D_termination"] == "resistance":
        junction_dict_master[junction_name]["0D_geo_resistance"] = junction_dict_master[junction_name]["0D_bc_geo_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet1"]
    elif junction_dict_master[junction_name]["0D_termination"] == "junction":
        downstream_junction_name = junction_dict_master[junction_name]["0D_terminal_junction_name"]
        junction_dict_master[junction_name]["0D_bc_geo_resistance"] = get_recursive_resistance(junction_dict_master, downstream_junction_name)
        junction_dict_master[junction_name]["0D_geo_resistance"] = junction_dict_master[junction_name]["0D_bc_geo_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet1"]
        junction_dict_master[junction_name]["0D_termination"] = "resistance"
        #downstream_resistance = junction_dict_master[downstream_junction_name]["0D_bc_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet1"]

    if junction_dict_master[junction_name]["0D_aux_termination"] == "resistance":
        junction_dict_master[junction_name]["0D_aux_geo_resistance"] = junction_dict_master[junction_name]["0D_aux_bc_geo_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet2"]
    elif junction_dict_master[junction_name]["0D_aux_termination"] == "junction":
        downstream_aux_junction_name = junction_dict_master[junction_name]["0D_aux_terminal_junction_name"]
        junction_dict_master[junction_name]["0D_aux_bc_geo_resistance"] = get_recursive_resistance(junction_dict_master, downstream_aux_junction_name)
        junction_dict_master[junction_name]["0D_aux_geo_resistance"] = junction_dict_master[junction_name]["0D_aux_bc_geo_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet2"]
        junction_dict_master[junction_name]["0D_aux_termination"] = "resistance"
        #downstream_aux_resistance = junction_dict_master[downstream_aux_junction_name]["0D_aux_bc_resistance"] + junction_dict_master[junction_name]["0D_R_poiseuille_outlet2"]

    junction_resistance = (junction_dict_master[junction_name]["0D_geo_resistance"]**-1 + junction_dict_master[junction_name]["0D_aux_geo_resistance"]**-1)**-1
    #junction_dict_master[junction_name]["resistance_at_junction"] = junction_resistance
    # elif junction_dict_master[junction_name]["0D_termination"] == "resistance":
    #     get_recursive_resistance
    return junction_resistance #, downstream_aux_resistance

def add_solution_values(junction_dict_master, tree_name, flow_mag, time_step):
    
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    fpath_out = f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_{time_step}.vtp"
    # Project the 3D solution onto the centerline if not already done
    if not os.path.exists(fpath_out):
        fpath_1d = f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/centerlines/centerlines.vtp"
        fpath_3d = f"trees/threed_results/{tree_name_base}/{tree_name}/{tree_name}_result_{time_step}.vtu"
        if not os.path.exists(f"trees/threed_output_cent/{tree_name_base}/{tree_name}"):
            os.makedirs(f"trees/threed_output_cent/{tree_name}")
        extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 50)

    # Load the solution data
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, direction, pressure_in_time, flow_in_time, times, time_interval= \
    load_vmr_model_data(f"centerline_sol_{time_step}.vtp", f"trees/threed_output_cent/{tree_name_base}/{tree_name}")
    reader_1d = read_geo(fpath_out).GetOutput(); points = v2n(reader_1d.GetPoints().GetData())

    # Fit a curve to the flow and pressure solutions, get extra timesteps and derivatives, as necessary
    pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
    flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time, pressure_in_time, times)

    # Find the endpoints of the junctions
    junction_dict_3D, offsets, branch_id_dict = identify_junctions_offset(junction_id, branch_id, pt_id, path, offset=0)
    # Find the inlet and outlet points of the branches
    branch_dict_3D = identify_branches_offset(branch_id, pt_id, path, offset=0)

    for junction_name, junction_dict in junction_dict_master.items():
        # Decide which points are inlets and outlets, and find the indices of relevant points, from their pt_ids
        junc_inlet_pt, junc_outlet_pts = classify_branches_backflow_allowed(flow_in_time_aug, junc_pts = junction_dict_3D[junction_dict["junction_id"]], pt_arr = pt_id)
        junc_inlet_ind =    get_inds(arr = pt_id, vals = junc_inlet_pt)[0]
        junc_outlet_inds =  get_inds(arr = pt_id, vals = junc_outlet_pts)

        # Identify the first branch coming out of the junction
        branch1_id = junction_dict["0D_outlet1_branch_id"]
        # Get the indices of the inlet and outlet points of the first branch
        branch1_inlet_ind =     get_inds(arr = pt_id, vals = [branch_dict_3D[branch1_id]["min_pt"]])
        branch1_outlet_ind =    get_inds(arr = pt_id, vals = [branch_dict_3D[branch1_id]["max_pt"]])
        assert (np.linalg.norm(points[branch1_inlet_ind,:] - points[junc_outlet_inds[0],:]) < 1e-2), "Inlet point of branch 1 is not the same as outlet point of junction."
        #junction_dict["3D_branch1_id"] = branch1_id

        # Identify the second branch coming out of the junction
        branch2_id = junction_dict["0D_outlet2_branch_id"]
        # Get the indices of the inlet and outlet points of the first branch
        branch2_inlet_ind =     get_inds(arr = pt_id, vals = [branch_dict_3D[branch2_id]["min_pt"]])
        branch2_outlet_ind =    get_inds(arr = pt_id, vals = [branch_dict_3D[branch2_id]["max_pt"]])
        assert (np.linalg.norm(points[branch2_inlet_ind,:] - points[junc_outlet_inds[1],:]) < 1e-2), "Inlet point of branch 1 is not the same as outlet point of junction."
        
        # Extract the relevant pressure values
        junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"]      = pressure_in_time_aug[0, junc_inlet_ind]
        junction_dict[f"3D_junc_outlet1_pressure_fm_{flow_mag}_ts_{time_step}"]    = pressure_in_time_aug[0, junc_outlet_inds[0]]
        junction_dict[f"3D_junc_outlet2_pressure_fm_{flow_mag}_ts_{time_step}"]    = pressure_in_time_aug[0, junc_outlet_inds[1]]
        junction_dict[f"3D_branch1_inlet_pressure_fm_{flow_mag}_ts_{time_step}"]   = pressure_in_time_aug[0, branch1_inlet_ind]
        junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step}"]  = pressure_in_time_aug[0, branch1_outlet_ind]
        junction_dict[f"3D_branch2_inlet_pressure_fm_{flow_mag}_ts_{time_step}"]   = pressure_in_time_aug[0, branch2_inlet_ind]
        junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step}"]  = pressure_in_time_aug[0, branch2_outlet_ind]
        

        # Extract the relevant flow values
        junction_dict[f"3D_junc_inlet_flow_fm_{flow_mag}_ts_{time_step}"]      = flow_in_time_aug[0, junc_inlet_ind]
        junction_dict[f"3D_junc_outlet1_flow_fm_{flow_mag}_ts_{time_step}"]    = flow_in_time_aug[0, junc_outlet_inds[0]]
        junction_dict[f"3D_junc_outlet2_flow_fm_{flow_mag}_ts_{time_step}"]    = flow_in_time_aug[0, junc_outlet_inds[1]]
        junction_dict[f"3D_branch1_inlet_flow_fm_{flow_mag}_ts_{time_step}"]   = flow_in_time_aug[0, branch1_inlet_ind]
        junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step}"]  = flow_in_time_aug[0, branch1_outlet_ind]
        junction_dict[f"3D_branch2_inlet_flow_fm_{flow_mag}_ts_{time_step}"]   = flow_in_time_aug[0, branch2_inlet_ind]
        junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step}"]  = flow_in_time_aug[0, branch2_outlet_ind]

        junction_dict[f"3D_flow_split_flow_fm_{flow_mag}_ts_{time_step}"]      = flow_in_time_aug[0, branch1_outlet_ind]/flow_in_time_aug[0, branch2_outlet_ind]
        junction_dict[f"3D_branch1_outlet_resistance_fm_{flow_mag}_ts_{time_step}"] = junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step}"]/junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step}"]
        junction_dict[f"3D_branch2_outlet_resistance_fm_{flow_mag}_ts_{time_step}"] = junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step}"]/junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step}"]


    return

def add_downstream_resistance_values(junction_dict_master):
    junction_name = "J0"
    junction_resistance = get_recursive_resistance(junction_dict_master, junction_name)
    for junction_name, junction_dict in junction_dict_master.items():
        junction_dict["0D_geo_flow_split"] = junction_dict["0D_aux_geo_resistance"] / junction_dict["0D_geo_resistance"]
        # junction_dict["0D_bc_resistance"] = downstream_resistance
        # junction_dict["0D_aux_bc_resistance"] = downstream_aux_resistance
    
    return

def add_geometry_values(junction_dict_master, tree_name, flow_mag, time_step):

    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    # Project the 3D solution onto the centerline if not already done
    fpath_out = f"trees/threed_output_cent/{tree_name_base}/{tree_name}/centerline_sol_{time_step}.vtp"
    if not os.path.exists(fpath_out):
        fpath_1d = f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/centerlines/centerlines.vtp"
        fpath_3d = f"trees/threed_results/{tree_name_base}/{tree_name}_flow_{flow_mag}/{tree_name}_{flow_mag}_result_{time_step}.vtu"
        if not os.path.exists(f"trees/threed_output_cent/{tree_name_base}/{tree_name}_flow_{flow_mag}"):
            os.makedirs(f"trees/threed_output_cent/{tree_name_base}/{tree_name}_flow_{flow_mag}")
        extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 50)

    # Load the solution data
    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, direction, pressure_in_time, flow_in_time, times, time_interval= \
    load_vmr_model_data(f"centerline_sol_{time_step}.vtp", f"trees/threed_output_cent/{tree_name_base}/{tree_name}")
    reader_1d = read_geo(fpath_out).GetOutput(); points = v2n(reader_1d.GetPoints().GetData())
    
    # Fit a curve to the flow and pressure solutions, get extra timesteps and derivatives, as necessary
    pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
    flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time, pressure_in_time, times)
    
    # Find the endpoints of the junctions
    junction_dict_3D, offsets, branch_id_dict = identify_junctions_offset(junction_id, branch_id, pt_id, path, offset=0)
    # Find the inlet and outlet points of the branches
    branch_dict_3D = identify_branches_offset(branch_id, pt_id, path, offset=0)
    
    for junction_name, junction_dict in junction_dict_master.items():
        # Decide which points are inlets and outlets, and find the indices of relevant points, from their pt_ids
        junc_inlet_pt, junc_outlet_pts = classify_branches_backflow_allowed(flow_in_time_aug.T, junc_pts = junction_dict_3D[junction_dict["junction_id"]], pt_arr = pt_id)
        junc_inlet_ind =    get_inds(arr = pt_id, vals = junc_inlet_pt)[0]
        junc_outlet_inds =  get_inds(arr = pt_id, vals = junc_outlet_pts)
        junction_dict["3D_junc_inlet_area"] = area[junc_inlet_ind]
        junction_dict["3D_L_char"] = (junction_dict["3D_junc_inlet_area"] / np.pi)**0.5
        junction_dict["3D_junc_outlet1_area"] = area[junc_outlet_inds[0]]
        junction_dict["3D_junc_outlet2_area"] = area[junc_outlet_inds[1]]
        junction_dict["3D_junc_inlet_tangent"] = direction[junc_inlet_ind,:]
        junction_dict["3D_junc_outlet1_tangent"] = direction[junc_outlet_inds[0],:]
        junction_dict["3D_junc_outlet2_tangent"] = direction[junc_outlet_inds[1],:]
        junction_dict["3D_junc_outlet1_angle"] = get_angle_diff(np.asarray(junction_dict["3D_junc_inlet_tangent"]), np.asarray(junction_dict["3D_junc_outlet1_tangent"]))[0]
        junction_dict["3D_junc_outlet2_angle"] = get_angle_diff(np.asarray(junction_dict["3D_junc_inlet_tangent"]), np.asarray(junction_dict["3D_junc_outlet2_tangent"]))[0]

        # Identify the first branch coming out of the junction
        branch1_id = junction_dict["0D_outlet1_branch_id"]
        # Get the indices of the inlet and outlet points of the first branch
        branch1_inlet_ind =     get_inds(arr = pt_id, vals = [branch_dict_3D[branch1_id]["min_pt"],])[0]
        branch1_outlet_ind =    get_inds(arr = pt_id, vals = [branch_dict_3D[branch1_id]["max_pt"],])[0]

        assert (np.linalg.norm(points[branch1_inlet_ind,:] - points[junc_outlet_inds[0],:]) < 1e-2), "Inlet point of branch 1 is not the same as outlet point of junction."
        junction_dict["3D_branch1_id"] = branch1_id
        junction_dict["3D_branch1_length"] = np.linalg.norm(points[branch1_outlet_ind, :] - points[branch1_inlet_ind, :])
        junction_dict["3D_branch1_length_star"] = junction_dict["3D_branch1_length"] / junction_dict["3D_L_char"]
        junction_dict["3D_branch1_inlet_area"] = area[branch1_inlet_ind]
        junction_dict["3D_branch1_outlet_area"] = area[branch1_outlet_ind]
        junction_dict["3D_branch1_inlet_tangent"] = direction[branch1_inlet_ind,:]
        junction_dict["3D_branch1_outlet_tangent"] = direction[branch1_outlet_ind,:]


        # Identify the second branch coming out of the junction
        branch2_id = junction_dict["0D_outlet2_branch_id"]
        # Get the indices of the inlet and outlet points of the first branch
        branch2_inlet_ind =     get_inds(arr = pt_id, vals = [branch_dict_3D[branch2_id]["min_pt"],])[0]
        branch2_outlet_ind =    get_inds(arr = pt_id, vals = [branch_dict_3D[branch2_id]["max_pt"],])[0]
        assert (np.linalg.norm(points[branch2_inlet_ind, :] - points[junc_outlet_inds[1], :]) < 1e-2), "Inlet point of branch 1 is not the same as outlet point of junction."
        junction_dict["3D_branch2_id"] = branch2_id
        junction_dict["3D_branch2_length"] = np.linalg.norm(points[branch2_outlet_ind, :] - points[branch2_inlet_ind, :])
        junction_dict["3D_branch2_length_star"] = junction_dict["3D_branch2_length"] / junction_dict["3D_L_char"]
        junction_dict["3D_branch2_inlet_area"] = area[branch2_inlet_ind]
        junction_dict["3D_branch2_outlet_area"] = area[branch2_outlet_ind]
        junction_dict["3D_branch2_inlet_tangent"] = direction[branch2_inlet_ind,:]
        junction_dict["3D_branch2_outlet_tangent"] = direction[branch2_outlet_ind,:]
    
    return

def add_3D_resistance(junction_dict_master, tree_name, flow_mag_list, time_step):
    for junction_name, junction_dict in junction_dict_master.items():

        re_char = 4500
        A_char = junction_dict["3D_junc_inlet_area"]
        L_char = np.sqrt(A_char/np.pi)
        U_char = re_char * 0.04/(1.06 * 2*np.sqrt(A_char/np.pi))
        junction_dict["3D_U_char"] = U_char

        daughter1_dPs = []
        daughter2_dPs = []
        daughter1_flows = []
        daughter2_flows = []

        # Compose lists of flow and pressure data for each outlet
        for flow_mag in flow_mag_list:

                daughter1_dPs.append(
                    junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] - 
                    junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step}"])
                daughter2_dPs.append(
                    junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step}"] - 
                    junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step}"])
                
                daughter1_flows.append(
                    junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step}"])
                daughter2_flows.append(
                    junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step}"])
        
                assert len(daughter1_dPs) == len(daughter1_flows); "Lengths of daughter1_dPs and daughter1_flows do not match."
                assert len(daughter2_dPs) == len(daughter2_flows); "Lengths of daughter2_dPs and daughter2_flows do not match."
        
        junction_dict["3D_daughter1_flows"] = copy.copy(daughter1_flows)
        junction_dict["3D_daughter2_flows"] = copy.copy(daughter2_flows)
        junction_dict["3D_inlet_flows"] = [daughter1_flow + daughter2_flow for daughter1_flow, daughter2_flow in zip(daughter1_flows, daughter2_flows)]

        junction_dict["3D_daughter1_dPs"] = copy.copy(daughter1_dPs)
        junction_dict["3D_daughter2_dPs"] = copy.copy(daughter2_dPs)

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

        # Daughter 1 flows
        A_mat[0:num_flows,0] = Q_inlet
        A_mat[num_flows:2*num_flows,2] = Q_inlet
        A_mat_star[0:num_flows,0] = Q_star_inlet
        A_mat_star[num_flows:2*num_flows,2] = Q_star_inlet

        quadratic_resistors = True
        if quadratic_resistors:
            A_mat[0:num_flows,1] = np.square(Q_inlet)
            A_mat[num_flows:2*num_flows,3] = np.square(Q_inlet)
            A_mat_star[0:num_flows,1] = np.square(Q_star_inlet)
            A_mat_star[num_flows:2*num_flows,3] = np.square(Q_star_inlet)

        # Solve
        coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
        R_lin_star1         = coefs_star[0]
        R_quad_star1        = coefs_star[1]
        R_lin_star2         = coefs_star[2]
        R_quad_star2        = coefs_star[3]

        junction_dict["3D_daughter1_R_lin_star"] = copy.copy(R_lin_star1)
        junction_dict["3D_daughter2_R_lin_star"] = copy.copy(R_lin_star2)
        junction_dict["3D_daughter1_R_quad_star"] = copy.copy(R_quad_star1)
        junction_dict["3D_daughter2_R_quad_star"] = copy.copy(R_quad_star2)

        # Solve
        coefs, residuals, t, q = np.linalg.lstsq(A_mat, dP_vec, rcond=None)
        #pdb.set_trace()
        residuals = dP_vec - A_mat @ coefs
        # print(f"Residuals: {np.linalg.norm(residuals/dP_vec)}")

        R_lin1         = coefs[0]
        R_quad1        = coefs[1]
        R_lin2         = coefs[2]
        R_quad2        = coefs[3]

        junction_dict["3D_daughter1_R_lin"] = copy.copy(R_lin1)
        junction_dict["3D_daughter2_R_lin"] = copy.copy(R_lin2)
        junction_dict["3D_daughter1_R_quad"] = copy.copy(R_quad1)
        junction_dict["3D_daughter2_R_quad"] = copy.copy(R_quad2)

        # Check consistency of non-dimensionalization
        
        assert abs(junction_dict["3D_daughter1_R_lin"] - junction_dict["3D_daughter1_R_lin_star"]*1.06*U_char/A_char) < 0.1; "Daughter 1 linear resistances do not match."
        assert abs(junction_dict["3D_daughter2_R_lin"] - junction_dict["3D_daughter2_R_lin_star"]*1.06*U_char/A_char) < 0.1; "Daughter 2 linear resistances do not match."
        assert abs(junction_dict["3D_daughter1_R_quad"] - junction_dict["3D_daughter1_R_quad_star"]*1.06/A_char**2) < 0.1; "Daughter 1 quadratic resistances do not match."
        assert abs(junction_dict["3D_daughter2_R_quad"] - junction_dict["3D_daughter2_R_quad_star"]*1.06/A_char**2) < 0.1; "Daughter 2 quadratic resistances do not match."

    return

def add_3D_outlet_resistance(junction_dict_master, tree_name, flow_mag_list, time_step):

    for junction_name, junction_dict in junction_dict_master.items():

        junction_dict["3D_outlet1_boundary_resistances"] = []
        if junction_dict["0D_aux_termination"] == "resistance":
            for flow_mag in flow_mag_list:
                junction_dict["3D_outlet1_boundary_resistances"].append(
                    junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step}"] /
                    junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step}"])
                
        junction_dict["3D_outlet2_boundary_resistances"] = []        
        if junction_dict["0D_aux_termination"] == "resistance":
            for flow_mag in flow_mag_list:
                junction_dict["3D_outlet2_boundary_resistances"].append(
                    junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step}"] /
                    junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step}"])
                
    return

def add_0D_resistance(junction_dict_master, tree_name):
    for junction_name, junction_dict in junction_dict_master.items():

        anatomy = "tree_20"; 
        set_type = "random" 
        #set_type = "combined" #"dict_res_fs_ext"; 
        scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
        #model_name = "tree_20_ng_720_nl_2_lw_70_ne_1000_bs_20_dr_0.95_model" #
        model_name ="tree_20_ng_498_nl_1_lw_30_ne_2000_bs_20_dr_0.95_model"#"tree_20_ng_400_nl_2_lw_60_ne_2000_bs_20_dr_0.95_model" # "tree_20_ng_400_nl_2_lw_100_ne_1000_bs_20_dr_0.95_model"
        nn_model = dill_load(f"results/models/{anatomy}/{model_name}")
        
        A_char = junction_dict["0D_inlet_area"]; L_char = np.sqrt(A_char/np.pi)
        junction_dict["0D_L_char"] = L_char
        re_char = 4500; U_char = re_char * 0.04 / (1.06 * 2 *np.sqrt(A_char/np.pi))
        junction_dict["0D_U_char"] = U_char
        assert np.linalg.norm((junction_dict["0D_U_char"] - junction_dict["3D_U_char"])/junction_dict["3D_U_char"]) < 1e-1; "0D and 3D U_char do not match."
        
        daughter1_area_ratio = junction_dict["0D_outlet1_area"]/A_char; daughter1_area_ratio = check_out_of_dist(daughter1_area_ratio, "daughter1_area_ratio", scaling_dict)
        daughter2_area_ratio = junction_dict["0D_outlet2_area"]/A_char; daughter2_area_ratio = check_out_of_dist(daughter2_area_ratio, "daughter2_area_ratio", scaling_dict)
        total_daughter_area_ratio = daughter1_area_ratio + daughter2_area_ratio; total_daughter_area_ratio = check_out_of_dist(total_daughter_area_ratio, "total_daughter_area_ratio", scaling_dict)
        daughter1_area_ratio_inv2 = (junction_dict["0D_outlet1_area"]/A_char)**-2;  daughter1_area_ratio_inv2 = check_out_of_dist(daughter1_area_ratio_inv2, "daughter1_area_ratio_inv2", scaling_dict)
        daughter2_area_ratio_inv2 = (junction_dict["0D_outlet2_area"]/A_char)**-2;  daughter2_area_ratio_inv2 = check_out_of_dist(daughter2_area_ratio_inv2, "daughter2_area_ratio_inv2", scaling_dict)
        total_area_ratio_inv2 = 1/(total_daughter_area_ratio**2); total_area_ratio_inv2 = check_out_of_dist(total_area_ratio_inv2, "total_area_ratio_inv2", scaling_dict)
        daughter1_angle = get_angle_diff(np.asarray(junction_dict["0D_inlet_tangent"]), np.asarray(junction_dict["0D_daughter1_tangent"])); 
        daughter1_angle = np.cos(daughter1_angle)
        daughter1_angle = check_out_of_dist(daughter1_angle, "daughter1_angle", scaling_dict)
        
        daughter2_angle = get_angle_diff(np.asarray(junction_dict["0D_inlet_tangent"]), np.asarray(junction_dict["0D_daughter2_tangent"])); 
        daughter2_angle = np.cos(daughter2_angle)
        daughter2_angle = check_out_of_dist(daughter2_angle, "daughter2_angle", scaling_dict)
        daughter1_length = junction_dict["0D_length1"]
        daughter1_length_star = daughter1_length/L_char; daughter1_length_star_trim = check_out_of_dist(daughter1_length_star, "daughter1_length_star", scaling_dict)
        
        junction_dict["0D_length1_star"] = daughter1_length_star
        daughter2_length = junction_dict["0D_length2"]
        junction_dict["0D_length2_star"] = daughter2_length/L_char
        daughter2_length_star = daughter2_length/L_char; daughter2_length_star_trim = check_out_of_dist(daughter2_length_star, "daughter2_length_star", scaling_dict)
        print(f"Length star 1: {daughter1_length_star_trim} Length star 2: {daughter2_length_star_trim}")

        length_add1 = max([daughter1_length - L_char * scaling_dict["daughter1_length_star"][3], 0])
        junction_dict["0D_length_add1"] = length_add1
        length_add2 = max([daughter2_length - L_char * scaling_dict["daughter2_length_star"][3], 0])
        junction_dict["0D_length_add2"] = length_add2
        #pdb.set_trace()
        junction_dict["max_length"] = scaling_dict["daughter1_length_star"][3] * L_char

        daughter1_flow_split = junction_dict["0D_geo_flow_split"]/(1+ junction_dict["0D_geo_flow_split"])
        daughter2_flow_split = 1 - daughter1_flow_split
        #pdb.set_trace()

        assert scaling_dict["daughter1_length_star"][3] == scaling_dict["daughter2_length_star"][3]; "Length scaling factors do not match."
        res_add1 = length_add1 * 8 * np.pi * 0.04 / (junction_dict["0D_outlet1_area"]**2)
        junction_dict["0D_res_add1"] = res_add1
        res_add2 = length_add2 * 8 * np.pi * 0.04 / (junction_dict["0D_outlet2_area"]**2)
        junction_dict["0D_res_add2"] = res_add2
        # print(f"Length star 1: {daughter1_length_star} Length star 2: {daughter2_length_star}")
        # FIX LAST INPUT ARG
        input_tens1 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(total_daughter_area_ratio, dtype=jnp.float32), "total_daughter_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(total_area_ratio_inv2, dtype=jnp.float32), "total_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter1_angle"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter2_angle"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_length_star_trim, dtype=jnp.float32), "daughter1_length_star"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_length_star_trim, dtype=jnp.float32), "daughter2_length_star"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_flow_split, dtype=jnp.float32), "daughter1_flow_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_flow_split**2, dtype=jnp.float32), "daughter1_flow_ratio_sq"),
                            ]).reshape(1,-1)
        input_tens2 = jnp.asarray([scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio, dtype=jnp.float32), "daughter1_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio, dtype=jnp.float32), "daughter2_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(total_daughter_area_ratio, dtype=jnp.float32), "total_daughter_area_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_area_ratio_inv2, dtype=jnp.float32), "daughter1_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_area_ratio_inv2, dtype=jnp.float32), "daughter2_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(total_area_ratio_inv2, dtype=jnp.float32), "total_area_ratio_inv2"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_angle, dtype=jnp.float32), "daughter1_angle"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_angle, dtype=jnp.float32), "daughter2_angle"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_length_star_trim, dtype=jnp.float32), "daughter1_length_star"),
                        scale_jax(scaling_dict, jnp.asarray(daughter1_length_star_trim, dtype=jnp.float32), "daughter2_length_star"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_flow_split, dtype=jnp.float32), "daughter2_flow_ratio"),
                        scale_jax(scaling_dict, jnp.asarray(daughter2_flow_split**2, dtype=jnp.float32), "daughter2_flow_ratio_sq"),
                            ]).reshape(1,-1)
        # print(f"Input tensor 2 {junction_name}: {input_tens2}")


        coefs_pred1 = predict(input_tens1, nn_model.weights)
        coefs_pred2 = predict(input_tens2, nn_model.weights)

        R_lin_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][0], "daughter1_R_lin_star")[0][0]); R_lin_star_pred1 = check_out_of_dist(R_lin_star_pred1, "daughter1_R_lin_star", scaling_dict)
        R_quad_star_pred1 = float(inv_scale_jax(scaling_dict, coefs_pred1[0][1], "daughter1_R_quad_star")[0][0]); R_quad_star_pred1 = check_out_of_dist(R_quad_star_pred1, "daughter1_R_quad_star", scaling_dict)
        
        R_lin_star_pred2 = float(inv_scale_jax(scaling_dict, coefs_pred2[0][0], "daughter2_R_lin_star")[0][0]); R_lin_star_pred2 = check_out_of_dist(R_lin_star_pred2, "daughter2_R_lin_star", scaling_dict)
        R_quad_star_pred2 = float(inv_scale_jax(scaling_dict, coefs_pred2[0][1], "daughter2_R_quad_star")[0][0]); R_quad_star_pred2 = check_out_of_dist(R_quad_star_pred2, "daughter2_R_quad_star", scaling_dict)
        
        junction_dict["0D_daughter1_R_lin_star"] = copy.copy(R_lin_star_pred1)
        junction_dict["0D_daughter1_R_quad_star"] = copy.copy(R_quad_star_pred1)
        junction_dict["0D_daughter2_R_lin_star"] = copy.copy(R_lin_star_pred2)
        junction_dict["0D_daughter2_R_quad_star"] = copy.copy(R_quad_star_pred2)

        daughter1_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char))); junction_dict["0D_daughter1_R_lin"] = copy.copy(daughter1_R_lin)
        daughter1_R_lin_final = float((1.06 * jnp.square(U_char) * R_lin_star_pred1 /  (A_char * U_char)))+ res_add1; junction_dict["0D_daughter1_R_lin_final"] = copy.copy(daughter1_R_lin_final)
        daughter1_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred1 / jnp.square(A_char * U_char)); junction_dict["0D_daughter1_R_quad"] = copy.copy(daughter1_R_quad)

        daughter2_R_lin = float((1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char))); junction_dict["0D_daughter2_R_lin"] = copy.copy(daughter2_R_lin)
        daughter2_R_lin_final = float((1.06 * jnp.square(U_char) * R_lin_star_pred2 /  (A_char * U_char)))+ res_add2; junction_dict["0D_daughter2_R_lin_final"] = copy.copy(daughter2_R_lin_final)
        daughter2_R_quad = float(1.06 * jnp.square(U_char) * R_quad_star_pred2 / jnp.square(A_char * U_char)); junction_dict["0D_daughter2_R_quad"] = copy.copy(daughter2_R_quad)
        #pdb.set_trace()
    return

def add_3D_isol_values(junction_dict_master, tree_name, isol_set_name):
    #isol_junc_dict_master = load_dict(f"data/data_dicts/{tree_name}_res2_dict_synthetic_data_dict")
    isol_junc_dict_master = load_dict(f"data/data_dicts/{tree_name}_{isol_set_name}_synthetic_data_dict")
    J_to_CCO_dict = {}
    for isol_junc_name in isol_junc_dict_master.keys():
        try:
            geo_name = load_dict(f"data/synthetic_junctions/{tree_name}/{isol_set_name}/{isol_junc_name}/geo_params_dict")["geo_name"]
            J_to_CCO_dict[geo_name] = isol_junc_name
        except:
            print(f"Junction {isol_junc_name} does not have a geo name.")
            continue

    # Match the junction geometry
    for junction_name, junction_dict in junction_dict_master.items():
        j_geo = np.asarray([junction_dict["3D_branch1_outlet_area"]/junction_dict["3D_junc_inlet_area"],
                            junction_dict["3D_branch2_outlet_area"]/junction_dict["3D_junc_inlet_area"],
                            junction_dict["3D_junc_outlet1_angle"],
                            junction_dict["3D_junc_outlet2_angle"]])
        diff_list = []
        isol_junc_list = list(isol_junc_dict_master.keys())
        for isol_junc in isol_junc_list:
            if len(isol_junc_dict_master[isol_junc].keys()) == 0:
                continue
            #pdb.set_trace()
            offset_name = max(list(isol_junc_dict_master[isol_junc].keys()))
            cco_geo = np.asarray([isol_junc_dict_master[isol_junc][offset_name]["daughter1_area_ratio"],
                                isol_junc_dict_master[isol_junc][offset_name]["daughter2_area_ratio"],
                                isol_junc_dict_master[isol_junc][offset_name]["daughter1_angle"],
                                isol_junc_dict_master[isol_junc][offset_name]["daughter2_angle"]])
            diff_list.append(np.linalg.norm((j_geo - cco_geo)/j_geo))
        diff_list = np.asarray(diff_list)

        try:
            junction_dict[f"isol_{isol_set_name}_junc_name"] = J_to_CCO_dict[junction_name]# f"CCO_0{int(junction_name[1:]):02d}" #isol_junc_list[np.argmin(diff_list)]
        except:
            continue
        junction_dict[f"isol_{isol_set_name}_junc_dict"] = isol_junc_dict_master[junction_dict[f"isol_{isol_set_name}_junc_name"]]
        
    # for junction_name, junction_dict in junction_dict_master.items():
        # Get the inlet and outlet areas
        offset_dict = isol_junc_dict_master[junction_dict[f"isol_{isol_set_name}_junc_name"]] # isol_junc_dict_master[f"CCO_{junction_dict["junction_id"]:03d}"]
        if len(list(offset_dict.keys())) == 0:
            if junction_name[0]=="J": 
                print(f"Junction {junction_name} has no isolated geometry.");# 
                continue
                #return
                #pdb.set_trace()
            else:
                continue
            #continue
            pdb.set_trace()
        length1_list = []; length2_list = []
        for offset, isol_dict in offset_dict.items():
            length1_list.append(isol_dict["daughter1_length"]/isol_dict["L_char"])
            length2_list.append(isol_dict["daughter2_length"]/isol_dict["L_char"])

        best_offset1 = list(offset_dict.keys())[
            np.argmin(np.abs(np.array(length1_list) - junction_dict["0D_length1_star"]))]
        best_offset2 = list(offset_dict.keys())[
            np.argmin(np.abs(np.array(length2_list) - junction_dict["0D_length2_star"]))]

        junction_dict[f"isol_{isol_set_name}_length1"] = offset_dict[best_offset1]["daughter1_length"]
        junction_dict[f"isol_{isol_set_name}_length2"] = offset_dict[best_offset2]["daughter2_length"]

        junction_dict[f"isol_{isol_set_name}_inlet_area"] = offset_dict[best_offset1]["A_char"]
        junction_dict[f"isol_{isol_set_name}_daughter1_area"] = offset_dict[best_offset1]["daughter1_area"]
        junction_dict[f"isol_{isol_set_name}_daughter2_area"] = offset_dict[best_offset2]["daughter2_area"]

        junction_dict[f"isol_{isol_set_name}_daughter1_area_ratio"] = offset_dict[best_offset1]["daughter1_area_ratio"]
        junction_dict[f"isol_{isol_set_name}_daughter2_area_ratio"] = offset_dict[best_offset2]["daughter2_area_ratio"]
        junction_dict[f"isol_{isol_set_name}_daughter1_angle"] = offset_dict[best_offset1]["daughter1_angle"]
        junction_dict[f"isol_{isol_set_name}_daughter2_angle"] = offset_dict[best_offset2]["daughter2_angle"]

        junction_dict[f"isol_{isol_set_name}_L_char"] = offset_dict[best_offset1]["L_char"]
        junction_dict[f"isol_{isol_set_name}_daughter1_length"] = offset_dict[best_offset1]["daughter1_length"]
        junction_dict[f"isol_{isol_set_name}_daughter2_length"] = offset_dict[best_offset2]["daughter2_length"]
        
        junction_dict[f"isol_{isol_set_name}_daughter1_length_star"] = offset_dict[best_offset1]["daughter1_length"]/offset_dict[best_offset1]["L_char"]
        junction_dict[f"isol_{isol_set_name}_daughter1_length_add_star"] = max([junction_dict["0D_length1_star"]-junction_dict[f"isol_{isol_set_name}_daughter1_length_star"],0])
        junction_dict[f"isol_{isol_set_name}_daughter1_length_add"] = junction_dict[f"isol_{isol_set_name}_daughter1_length_add_star"] * junction_dict["0D_L_char"]
        if junction_dict[f"isol_{isol_set_name}_daughter1_length_add"] > 0:
            print(f"Junction {junction_name} has daughter1 length add: {junction_dict[f'isol_{isol_set_name}_daughter1_length_add_star']}")

        junction_dict[f"isol_{isol_set_name}_daughter2_length"] = offset_dict[best_offset2]["daughter2_length"]
        junction_dict[f"isol_{isol_set_name}_daughter2_length_star"] = offset_dict[best_offset2]["daughter2_length"]/offset_dict[best_offset2]["L_char"]
        junction_dict[f"isol_{isol_set_name}_daughter2_length_add_star"] = max([junction_dict["0D_length2_star"]-junction_dict[f"isol_{isol_set_name}_daughter2_length_star"],0])
        junction_dict[f"isol_{isol_set_name}_daughter2_length_add"] = junction_dict[f"isol_{isol_set_name}_daughter2_length_add_star"] * junction_dict["0D_L_char"]
        if junction_dict[f"isol_{isol_set_name}_daughter2_length_add"] > 0:
            print(f"Junction {junction_name} has daughter2 length add: {junction_dict[f'isol_{isol_set_name}_daughter2_length_add_star']}")

        junction_dict[f"isol_{isol_set_name}_daughter1_flows"] = offset_dict[best_offset1]["daughter1_flow"]
        junction_dict[f"isol_{isol_set_name}_daughter2_flows"] = offset_dict[best_offset2]["daughter2_flow"]
        junction_dict[f"isol_{isol_set_name}_inlet_flows"] = [junction_dict[f"isol_{isol_set_name}_daughter1_flows"][i] + junction_dict[f"isol_{isol_set_name}_daughter2_flows"][i] for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter1_flows"]))]
        junction_dict[f"isol_{isol_set_name}_flow_splits"] = [junction_dict[f"isol_{isol_set_name}_daughter1_flows"][i]/junction_dict[f"isol_{isol_set_name}_daughter2_flows"][i] for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter1_flows"]))]
        #pdb.set_trace()

        # junction_dict[f"isol_{isol_set_name}_daughter1_flow_star"] = offset_dict[best_offset1]["daughter1_flow_star"]
        junction_dict[f"isol_{isol_set_name}_daughter1_flow_star"] = offset_dict[best_offset2]["daughter1_flow_star"]
        junction_dict[f"isol_{isol_set_name}_daughter2_flow_star"] = offset_dict[best_offset2]["daughter2_flow_star"]

        junction_dict[f"isol_{isol_set_name}_daughter1_dPs"] = offset_dict[best_offset1]["daughter1_dP"]
        junction_dict[f"isol_{isol_set_name}_daughter2_dPs"] = offset_dict[best_offset2]["daughter2_dP"]

        junction_dict[f"isol_{isol_set_name}_daughter1_dP_star"] = offset_dict[best_offset1]["daughter1_dP_star"]
        junction_dict[f"isol_{isol_set_name}_daughter2_dP_star"] = offset_dict[best_offset2]["daughter2_dP_star"]

        # junction_dict[f"isol_{isol_set_name}_daughter1_flow_redim"] = [flow_star * junction_dict["3D_junc_inlet_area"] * 
        #                                               junction_dict["3D_U_char"] 
        #                                               for flow_star in junction_dict[f"isol_{isol_set_name}_daughter1_flow_star"]]
        re_char = 4500
        junction_dict[f"isol_{isol_set_name}_U_char"] = re_char * 0.04/(1.06 * 2*np.sqrt(junction_dict[f"isol_{isol_set_name}_inlet_area"]/np.pi))
        junction_dict[f"isol_{isol_set_name}_inlet_velocity"] = [flow / junction_dict[f"isol_{isol_set_name}_inlet_area"] for flow in junction_dict[f"isol_{isol_set_name}_inlet_flows"]]
        junction_dict[f"isol_{isol_set_name}_inlet_vel_star"] = [vel/junction_dict[f"isol_{isol_set_name}_U_char"] for vel in junction_dict[f"isol_{isol_set_name}_inlet_velocity"]]
        junction_dict[f"isol_{isol_set_name}_redim_inlet_velocity"] = [vel_star*junction_dict["3D_U_char"] for vel_star in junction_dict[f"isol_{isol_set_name}_inlet_vel_star"]]
        junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"] = [vel * junction_dict["3D_junc_inlet_area"] for vel in junction_dict[f"isol_{isol_set_name}_redim_inlet_velocity"]]

        junction_dict[f"isol_{isol_set_name}_daughter1_flow_redim"] = [flow_star * junction_dict["3D_junc_inlet_area"] *
                                                        junction_dict["3D_U_char"]
                                                        for flow_star in junction_dict[f"isol_{isol_set_name}_daughter1_flow_star"]]
        junction_dict[f"isol_{isol_set_name}_daughter2_flow_redim"] = [flow_star * junction_dict["3D_junc_inlet_area"] *
                                                        junction_dict["3D_U_char"]
                                                        for flow_star in junction_dict[f"isol_{isol_set_name}_daughter2_flow_star"]]
        junction_dict["3D_inlet_velocity"] = [flow / junction_dict["3D_junc_inlet_area"] for flow in junction_dict["3D_inlet_flows"]]
        junction_dict[f"isol_{isol_set_name}_daughter1_dP_star"] = [junction_dict[f"isol_{isol_set_name}_daughter1_dPs"][i]/ (1.06 * junction_dict[f"isol_{isol_set_name}_U_char"]**2) for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter1_dPs"]))]
        junction_dict[f"isol_{isol_set_name}_daughter2_dP_star"] = [junction_dict[f"isol_{isol_set_name}_daughter2_dPs"][i]/ (1.06 * junction_dict[f"isol_{isol_set_name}_U_char"]**2) for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter2_dPs"]))]
        
        junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim"] = [junction_dict[f"isol_{isol_set_name}_daughter1_dP_star"][i] * (1.06 * junction_dict["3D_U_char"]**2) for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter1_dPs"]))]
        junction_dict[f"isol_{isol_set_name}_daughter2_dP_redim"] = [junction_dict[f"isol_{isol_set_name}_daughter2_dP_star"][i] * (1.06 * junction_dict["3D_U_char"]**2) for i in range(len(junction_dict[f"isol_{isol_set_name}_daughter2_dPs"]))]

        #junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim"] = #[dP_star * 1.06 * junction_dict["3D_U_char"]**2 for dP_star in junction_dict[f"isol_{isol_set_name}_daughter1_dP_star"]]
        # junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim"] = [dP_star * 1.06 * junction_dict["3D_U_char"]**2
        #                                                 for dP_star in junction_dict[f"isol_{isol_set_name}_daughter1_dP_star"]]
        # junction_dict[f"isol_{isol_set_name}_daughter2_dP_redim"] = [dP_star * 1.06 * junction_dict["3D_U_char"]**2
        #                                                  for dP_star in junction_dict[f"isol_{isol_set_name}_daughter2_dP_star"]]
        
        junction_dict[f"isol_{isol_set_name}_daughter1_res_add"] = 8 * np.pi * 0.04 * junction_dict[f"isol_{isol_set_name}_daughter1_length_add"]*junction_dict["3D_branch1_outlet_area"]**-2
        junction_dict[f"isol_{isol_set_name}_daughter2_res_add"] = 8 * np.pi * 0.04 * junction_dict[f"isol_{isol_set_name}_daughter2_length_add"]*junction_dict["3D_branch2_outlet_area"]**-2

        junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim_len_add"] = [junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim"][k] +
                                                1*junction_dict[f"isol_{isol_set_name}_daughter1_flow_redim"][k] * junction_dict[f"isol_{isol_set_name}_daughter1_res_add"]
                                                for k in range(len(junction_dict[f"isol_{isol_set_name}_daughter1_flow_redim"]))]
        
        junction_dict[f"isol_{isol_set_name}_daughter2_dP_redim_len_add"] = [junction_dict[f"isol_{isol_set_name}_daughter2_dP_redim"][k] +
                                                1*junction_dict[f"isol_{isol_set_name}_daughter2_flow_redim"][k] * junction_dict[f"isol_{isol_set_name}_daughter2_res_add"]
                                                for k in range(len(junction_dict[f"isol_{isol_set_name}_daughter2_flow_redim"]))]
        
        junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_star"] = offset_dict[best_offset1]["daughter1_R_lin_star"]
        junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_star"] = offset_dict[best_offset2]["daughter2_R_lin_star"]
        junction_dict[f"isol_{isol_set_name}_daughter1_R_quad_star"] = offset_dict[best_offset1]["daughter1_R_quad_star"]
        junction_dict[f"isol_{isol_set_name}_daughter2_R_quad_star"] = offset_dict[best_offset2]["daughter2_R_quad_star"]

        junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim_base"] = offset_dict[best_offset1]["daughter1_R_lin_star"]*1.06*junction_dict["3D_U_char"] /junction_dict["3D_junc_inlet_area"]
        junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim_base"] = offset_dict[best_offset2]["daughter2_R_lin_star"]*1.06*junction_dict["3D_U_char"] /junction_dict["3D_junc_inlet_area"]
        junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim"] = junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim_base"] + junction_dict[f"isol_{isol_set_name}_daughter1_res_add"]
        junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim"] = junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim_base"] + junction_dict[f"isol_{isol_set_name}_daughter2_res_add"]
        junction_dict[f"isol_{isol_set_name}_daughter1_R_quad_redim"] = offset_dict[best_offset1]["daughter1_R_quad_star"]*1.06/junction_dict["3D_junc_inlet_area"]**2
        junction_dict[f"isol_{isol_set_name}_daughter2_R_quad_redim"] = offset_dict[best_offset2]["daughter2_R_quad_star"]*1.06/junction_dict["3D_junc_inlet_area"]**2

        
        # if junction_dict[f"isol_{isol_set_name}_junc_name"] == "CCO_002":
        #     pdb

        

    return

def check_steady_state_convergence(junction_dict_master, tree_name, flow_mag_list, time_step1, time_step2):

    converged = True
    tol = 0.05
    for junction_name, junction_dict in junction_dict_master.items():

        re_char = 4500
        A_char = junction_dict["3D_junc_inlet_area"]
        L_char = np.sqrt(A_char/np.pi)
        U_char = re_char * 0.04/(1.06 * 2*np.sqrt(A_char/np.pi))

        daughter1_dPs = []
        daughter2_dPs = []
        daughter1_flows = []
        daughter2_flows = []


        # Compose lists of flow and pressure data for each outlet
        for flow_mag in flow_mag_list:

                converged = np.linalg.norm((junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step1}"] - 
                                           junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step2}"])/
                                           junction_dict[f"3D_junc_inlet_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                if not converged:   print(f"Junction {junction_name} not converged"); return False

                converged = np.linalg.norm((junction_dict[f"3D_junc_outlet1_pressure_fm_{flow_mag}_ts_{time_step1}"] -  
                                        junction_dict[f"3D_junc_outlet1_pressure_fm_{flow_mag}_ts_{time_step2}"])/
                                        junction_dict[f"3D_junc_outlet1_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                if not converged:   print(f"Junction {junction_name} not converged"); return False

                converged = np.linalg.norm((junction_dict[f"3D_junc_outlet2_pressure_fm_{flow_mag}_ts_{time_step1}"] -
                                        junction_dict[f"3D_junc_outlet2_pressure_fm_{flow_mag}_ts_{time_step2}"])/
                                         junction_dict[f"3D_junc_outlet2_pressure_fm_{flow_mag}_ts_{time_step2}"])< tol
                if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch1_inlet_pressure_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch1_inlet_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch1_outlet_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch2_inlet_pressure_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch2_inlet_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch2_outlet_pressure_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False

                # converged = np.linalg.norm((junction_dict[f"3D_junc_inlet_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_junc_inlet_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # pdb.set_trace()
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_junc_outlet1_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_junc_outlet1_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_junc_outlet2_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_junc_outlet2_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch1_inlet_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch1_inlet_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch1_outlet_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch2_inlet_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch2_inlet_flow_fm_{flow_mag}_ts_{time_step2}"])/
                #                         junction_dict[f"3D_branch2_inlet_flow_fm_{flow_mag}_ts_{time_step2}"])) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
                # converged = np.linalg.norm((junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step1}"] -
                #                         junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step2}"])/
                #                         junction_dict[f"3D_branch2_outlet_flow_fm_{flow_mag}_ts_{time_step2}"]) < tol
                # if not converged:   print(f"Junction {junction_name} not converged"); return False
               
    return True

def plot_flow_splits(junction_dict_master, tree_name, flow_mag_list, time_step):
    plt.rcParams['font.size'] = 14
    plt.clf()
    for junction_name, junction_dict in junction_dict_master.items():
        id = junction_dict["junction_id"]
        fs = junction_dict["0D_geo_flow_split"]
        if id == 0:
            plt.scatter(id, 100*fs/(fs+1), marker="o", color="red", label="Predicted")
        else:
            plt.scatter(id, 100*fs/(fs+1), marker="o", color="red")
        for i, flow_mag in enumerate(flow_mag_list):
            fs = junction_dict[f"3D_flow_split_flow_fm_{flow_mag}_ts_{time_step}"][0]
            if i == 0 and id ==0:
                plt.scatter(id, 100*fs/(fs+1), facecolors='none', edgecolors = "blue", label = "3D Sim")
            else:
                plt.scatter(id, 100*fs/(fs+1), facecolors='none', edgecolors = "blue")
    plt.xlabel("Junction ID")
    plt.ylabel("% Flow through Daughter 1")
    plt.ylim(0, 100)
    plt.legend()
    plt.savefig(f"trees/reports/flow_splits_{tree_name}.pdf", bbox_inches='tight')
    return

def make_pdfs(junction_dict_master, tree_name, flow_mag_list, time_step, isol_set_list):

    
    plt.rcParams['font.size'] = 10
    for junction_name, junction_dict in junction_dict_master.items():
        # Create a PDF report for each junction
        isol_set_name = isol_set_list[0]  # Assuming you want to use the first isol set for the report
        class PDF(FPDF):
            
            def header(self):
                self.set_font("Times", "B", 12)
                try:
                    self.cell(0, 10, f"{tree_name} - Junction {junction_name} - {junction_dict[f"isol_{isol_set_name}_junc_name"]}", 0, 1, "C")
                except:
                    self.cell(0, 10, f"{tree_name} - Junction {junction_name}", 0, 1, "C")

            def footer(self):
                self.set_y(-15)
                self.set_font("Times", "I", 8)
                self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "R")

            def chapter_title(self, title):
                self.set_font("Times", "B", 12)
                self.cell(0, 6, title, 0, 1, "L")
                #self.ln()

            def chapter_body(self, body):
                self.set_font("Times", "", 12)
                self.multi_cell(0, 10, body)
                self.ln()

            def add_plot(self, img_path):
                self.image(img_path, x=10, y=None, w=180)

            def add_resistance_table(self, table_data):
                cw1 = 40
                cw2 = 55
                self.set_font("Times", "B", 10)
                self.set_fill_color(255,255,190)
                self.cell(cw2, 4, "Value", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "3D Tree Sim.", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "3D Isolated Sim.", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "Neural Net", border = 0, ln = 1, align ="R")

                self.set_font("Times", "", 10)

                self.ln()
                self.cell(cw2, 4, "Daughter 1 Linear Resistance ND", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_daughter1_R_lin_star"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_star"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_R_lin_star"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 1 Linear Resistance", border = 0, ln = 0, align ="R", fill=True); 
                self.cell(cw1, 4, f"{junction_dict["3D_daughter1_R_lin"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_R_lin_final"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Daughter 1 Linear Resistance (Base)", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim_base"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_R_lin"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 1 Added Resistance", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_res_add"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_res_add1"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 1 Added Length", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_length_add"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_length_add1"]:.3f}", border = 0, ln = 1, align ="R")
                self.ln()

                self.cell(cw2, 4, "Daughter 2 Linear Resistance ND", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_daughter2_R_lin_star"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_star"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_R_lin_star"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 2 Linear Resistance", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_daughter2_R_lin"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_R_lin_final"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Daughter 2 Linear Resistance (Base)", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim_base"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_R_lin"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 2 Added Resistance", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_res_add"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_res_add2"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 2 Added Length", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_length_add"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_length_add2"]:.3f}", border = 0, ln = 1, align ="R")
                self.ln()

                self.cell(cw2, 4, "Daughter 1 Quadratic Resistance ND", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_daughter1_R_quad_star"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_R_quad_star"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_R_quad_star"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 1 Quadratic Resistance", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_daughter1_R_quad"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_R_quad_redim"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_R_quad"]:.3f}", border = 0, ln = 1, align ="R", fill=True)
                self.ln()

                self.cell(cw2, 4, "Daughter 2 Quadratic Resistance ND", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_daughter2_R_quad_star"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_R_quad_star"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_R_quad_star"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Daughter 2 Quadratic Resistance", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_daughter2_R_quad"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_R_quad_redim"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_R_quad"]:.3f}", border = 0, ln = 1, align ="R", fill=True)
                self.ln()
       
            def add_geometry_table(self, table_data):
                cw1 = 40
                cw2 = 55
                
                self.set_fill_color(255,255,190)
                self.set_font("Times", "B", 10)
                self.cell(cw2, 4, "Value", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "3D (in tree)", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "3D (isolated)", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, "0D", border = 0, ln = 1, align ="R")

                self.set_font("Times", "", 10)
                self.ln()
                self.cell(cw2, 4, "Inlet Area", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_junc_inlet_area"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_inlet_area"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=False)
                self.cell(cw1, 4, f"{junction_dict["0D_inlet_area"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Outlet 1 Area", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_branch1_outlet_area"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_area"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=False)
                self.cell(cw1, 4, f"{junction_dict["0D_outlet1_area"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Outlet 1 Area ratio", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_branch1_outlet_area"]/junction_dict["3D_junc_inlet_area"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_area_ratio"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_outlet1_area"]/junction_dict["0D_inlet_area"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Outlet 1 Area @ Junction", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_junc_outlet1_area"]:.3f}", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_outlet1_junction_area"]:.3f}", border = 0, ln = 1, align ="R")
                
                self.cell(cw2, 4, "Outlet 2 Area", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_branch2_outlet_area"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_area"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=False)
                self.cell(cw1, 4, f"{junction_dict["0D_outlet2_area"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Outlet 2 Area ratio", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_branch2_outlet_area"]/junction_dict["3D_junc_inlet_area"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_area_ratio"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_outlet2_area"]/junction_dict["0D_inlet_area"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

  
                self.cell(cw2, 4, "Outlet 2 Area @ Junction", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_junc_outlet2_area"]:.3f}", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["0D_outlet2_junction_area"]:.3f}", border = 0, ln = 1, align ="R")

                self.ln()

                self.cell(cw2, 4, "Outlet 1 Angle", border = 0, ln = 0, align ="R", fill=True); 
                self.cell(cw1, 4, f"{junction_dict["3D_junc_outlet1_angle"]*180/np.pi:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_angle"]*180/np.pi:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter1_angle"]*180/np.pi:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Outlet 2 Angle", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["3D_junc_outlet2_angle"]*180/np.pi:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_angle"]*180/np.pi:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_daughter2_angle"]*180/np.pi:.3f}", border = 0, ln = 1, align ="R", fill=True)
                
                self.ln()
                self.cell(cw2, 4, "Outlet 1 Length", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"{junction_dict["3D_branch1_length"]+junction_dict["0D_length1_base"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_length"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=False)
                self.cell(cw1, 4, f"{junction_dict["0D_length1"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Outlet 1 Length ND", border = 0, ln = 0, align ="R", fill=True); 
                self.cell(cw1, 4, f"{junction_dict["3D_branch1_length_star"]:.3f} ", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter1_length_star"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_length1_star"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Outlet 2 Length", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["3D_branch2_length"]+junction_dict["0D_length2_base"]:.3f}", border = 0, ln = 0, align ="R")
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_length"]:.3f}", border = 0, ln = 0, align ="R")
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=False)
                self.cell(cw1, 4, f"{junction_dict["0D_length2"]:.3f}", border = 0, ln = 1, align ="R")

                self.cell(cw2, 4, "Outlet 2 Length ND", border = 0, ln = 0, align ="R", fill=True); 
                self.cell(cw1, 4, f"{junction_dict["3D_branch2_length_star"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                try:
                    self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_daughter2_length_star"]:.3f}", border = 0, ln = 0, align ="R", fill=True)
                except:
                    self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R", fill=True)
                self.cell(cw1, 4, f"{junction_dict["0D_length2_star"]:.3f}", border = 0, ln = 1, align ="R", fill=True)

                self.cell(cw2, 4, "Max Handleable Length", border = 0, ln = 0, align ="R"); 
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                self.cell(cw1, 4, f"{junction_dict["max_length"]:.3f}", border = 0, ln = 1, align ="R")

                self.ln()
                for i, flow_mag in enumerate(flow_mag_list):
                    self.cell(cw2, 4, "Flow Split (Outlet 1/Outlet 2)", border = 0, ln = 0, align ="R"); 
                    self.cell(cw1, 4, f"{junction_dict[f"3D_flow_split_flow_fm_{flow_mag}_ts_{time_step}"][0]/(1+junction_dict[f"3D_flow_split_flow_fm_{flow_mag}_ts_{time_step}"][0]):.3f}", border = 0, ln = 0, align ="R")
                    try:
                        if i < len(junction_dict[f"isol_{isol_set_name}_flow_splits"]):
                            
                            self.cell(cw1, 4, f"{junction_dict[f"isol_{isol_set_name}_flow_splits"][i]/(1+junction_dict[f"isol_{isol_set_name}_flow_splits"][i]):.3f}", border = 0, ln = 0, align ="R")
                        else:
                            self.cell(cw1, 4, f"", border = 0, ln = 0, align ="R")
                    except:
                        
                        continue

                    self.cell(cw1, 4, f"{junction_dict["0D_geo_flow_split"]/(junction_dict["0D_geo_flow_split"]+1):.3f}", border = 0, ln = 1, align ="R")
                

        # Create a plot and save to a temporary file
        fname = f"{junction_name}_plot.png"
        fig, axs = plt.subplots(1,2, sharey=False, figsize=(10,3))
        m_size = 80

        axs[0].scatter(junction_dict["3D_inlet_flows"], [dP/1333 for dP in junction_dict["3D_daughter1_dPs"]], marker = "*", s = m_size, label="3D Tree Sim", color="g")
        axs[0].plot(junction_dict["3D_inlet_flows"], 
                    [junction_dict["3D_daughter1_R_lin"] * Q /1333 +
                    junction_dict["3D_daughter1_R_quad"] * Q**2 /1333 
                    for Q in junction_dict["3D_inlet_flows"]], color="g") #, label="3D Tree Fit", color="g")
        axs[0].plot(junction_dict["3D_inlet_flows"],
                    [junction_dict["0D_R_poiseuille_outlet1"] * Q /1333
                        for Q in junction_dict["3D_daughter1_flows"]], "--", label="Poiseuille", color="slategrey")    

        axs[0].plot(junction_dict["3D_inlet_flows"],
                    [junction_dict["0D_daughter1_R_lin_final"] * Q /1333 +
                        junction_dict["0D_daughter1_R_quad"] * Q**2 /1333
                        for Q in junction_dict["3D_inlet_flows"]], label="NN Pred", color="deeppink")
        isol_colors = [ 'darkblue',"cornflowerblue"]
        for j, isol_set_name in enumerate(isol_set_list):
            set_name = copy.copy(isol_set_list[j])
            try:
                print(f"isol_{isol_set_name}_flow_splits: {junction_dict[f'isol_{isol_set_name}_flow_splits']}")
            except:
                print(f"isol_{isol_set_name}_flow_splits not found in junction_dict")
                continue
            #pdb.set_trace()
            if f"isol_{isol_set_name}_daughter1_flow_redim" in junction_dict.keys():
                axs[0].plot(junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"],
                            [junction_dict[f"isol_{isol_set_name}_daughter1_R_lin_redim"] * Q /1333 +
                                junction_dict[f"isol_{isol_set_name}_daughter1_R_quad_redim"] * Q**2 /1333
                                for Q in junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"]], color=isol_colors[j])
        
                axs[0].scatter(junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"],
                        [dP/1333 for dP in junction_dict[f"isol_{isol_set_name}_daughter1_dP_redim_len_add"]], marker = ".", s = m_size, label=isol_set_name, color=isol_colors[j])
                print(f"Junction {junction_name} has no isolated geometry")

        set_name = copy.copy(isol_set_list[0])
            
        axs[0].set_xlabel("Inlet Flow (cm$^3$/s)")
        axs[0].set_ylabel("Pressure Drop (mmHg)")
        axs[0].legend()
        axs[0].set_title(f"Daughter 1")

        axs[1].scatter(junction_dict["3D_inlet_flows"], [dP/1333 for dP in junction_dict["3D_daughter2_dPs"]], marker = "*", s = m_size, label="3D Tree Sim", color="g")
        axs[1].plot(junction_dict["3D_inlet_flows"], 
                    [junction_dict["3D_daughter2_R_lin"] * Q /1333 +
                    junction_dict["3D_daughter2_R_quad"] * Q**2 /1333 
                    for Q in junction_dict["3D_inlet_flows"]], color="g") #, label="3D Tree Fit", color="g")
        axs[1].plot(junction_dict["3D_inlet_flows"],
                    [junction_dict["0D_R_poiseuille_outlet2"] * Q /1333
                        for Q in junction_dict["3D_daughter2_flows"]], "--", label="Poiseuille", color="slategrey")
        axs[1].plot(junction_dict["3D_inlet_flows"],
            [junction_dict["0D_daughter2_R_lin_final"] * Q /1333 +
                junction_dict["0D_daughter2_R_quad"] * Q**2 /1333
                for Q in junction_dict["3D_inlet_flows"]], label="NN Pred", color="deeppink")
        
        for j, isol_set_name in enumerate(isol_set_list):
            set_name = copy.copy(isol_set_list[j])
            print(f"isol_set_name: {isol_set_name}")
            if f"isol_{isol_set_name}_daughter2_flow_redim" in junction_dict.keys():
                axs[1].plot(junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"],
                            [junction_dict[f"isol_{isol_set_name}_daughter2_R_lin_redim"] * Q /1333 +
                                junction_dict[f"isol_{isol_set_name}_daughter2_R_quad_redim"] * Q**2 /1333
                                for Q in junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"]],  color=isol_colors[j])
        
                axs[1].scatter(junction_dict[f"isol_{isol_set_name}_inlet_flow_redim"],
                        [dP/1333 for dP in junction_dict[f"isol_{isol_set_name}_daughter2_dP_redim_len_add"]], marker = ".", s = m_size, label=isol_set_name, color=isol_colors[j])
                # axs[1].scatter(junction_dict["isol_inlet_flows"],
                #         [dP/1333 for dP in junction_dict["isol_daughter2_dPs"]], marker = "*", s = m_size, label="3D Isol. Sim", color="orange")
                #pdb.set_trace()
            else:
                print(f"Junction {junction_name} has no isolated geometry")
        set_name = copy.copy(isol_set_list[0])

        
        axs[1].set_xlabel("Inlet Flow (cm$^3$/s)")
        axs[1].set_ylabel("Pressure Drop (mmHg)")
        axs[1].legend()
        axs[1].set_title(f"Daughter 2")
        plt.savefig(fname, format="PNG",bbox_inches='tight', dpi=300)

        # Generate PDF
        pdf = PDF()

        pdf.add_page()
        pdf.add_plot(fname)

        pdf.chapter_title("Resistances:")
        pdf.add_resistance_table(junction_dict)

        pdf.chapter_title("Geometric Data")
        pdf.add_geometry_table(junction_dict)

        report_loc = f"trees/reports/{tree_name}/{junction_name}_report.pdf"
        if not os.path.exists(os.path.dirname(report_loc)):
            os.makedirs(os.path.dirname(report_loc))
        pdf.output(report_loc)

        if os.path.exists(fname):
            os.remove(fname)

        # if junction_name == "J2":
        #     pdb.set_trace()
        print(f"PDF saved to {report_loc}")

def get_statistics(junction_dict_master, tree_name, flow_mag_list, time_step, isol_set_list):
    param_dict = {"daughter1_angle": [],
                "daughter2_angle": [],
                "daughter1_area_ratio": [],
                "daughter2_area_ratio": [],
                "flow_split": [],
                "max_inlet_re": [],
                }
    for junction_name, junction_dict in junction_dict_master.items():
        param_dict["daughter1_area_ratio"].append(junction_dict["3D_branch1_outlet_area"]/junction_dict["3D_junc_inlet_area"])
        param_dict["daughter2_area_ratio"].append(junction_dict["3D_branch2_outlet_area"]/junction_dict["3D_junc_inlet_area"])
        param_dict["daughter1_angle"].append(junction_dict["3D_junc_outlet1_angle"]*180/np.pi)
        param_dict["daughter2_angle"].append(junction_dict["3D_junc_outlet2_angle"]*180/np.pi)
        param_dict["flow_split"].append(junction_dict[f"3D_flow_split_flow_fm_{flow_mag_list[0]}_ts_{time_step}"][0]/(1+junction_dict[f"3D_flow_split_flow_fm_{flow_mag_list[0]}_ts_{time_step}"][0]))

    stats_dict = {}
    for key, values in param_dict.items():
        if len(values) == 0:
            continue
        stats_dict[key] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values)
        }
    save_dict(stats_dict, f"trees/reports/{tree_name}/junction_statistics")
    pdb.set_trace()