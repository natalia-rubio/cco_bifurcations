# -*- coding: utf-8 -*-
"""
Natalia Rubio
August 2022
Get junction GNN graph lists from VMR models.
"""
# VTK processing from Martin:

import os
import sys
#sys.path[0] = "/home/users/nrubio/miniconda3/lib/python3.9/site-packages"
#sys.path.append("/home/users/nrubio/SV_scripts")
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
import pdb
import random
import copy
from scipy import interpolate
import pickle
from get_avg_sol import *

np.seterr(all='raise')
np.set_printoptions(threshold=sys.maxsize)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res

def get_all_arrays(geo):
    # collect all arrays
    #cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())
    return point_data

def read_solution(geo_name, flow_name):
    #home_dir = os.path.expanduser("~")
    fname = "/home/nrubio/Desktop/res_sweep/%s/avg_solution_res_%s.vtu"%(geo_name, flow_name)
    print(fname)
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def read_centerline(fpath_1d):
    fname = fpath_1d

    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError("File extension " + ext + " unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader

def augment_time(field, times_before, aug_factor):
    ntimes_new = aug_factor*len(times_before)
    end_time = max(times_before)
    start_time = min(times_before)
    n_points = field.shape[1]
    times_new = np.linspace(start_time, end_time, ntimes_new, endpoint=True)

    field_new = np.zeros((ntimes_new, n_points))
    field_new_der = np.zeros((ntimes_new, n_points))
    field_new_der2 = np.zeros((ntimes_new, n_points))
    for point_i in range(n_points):
      y = field[:, point_i]
      tck = interpolate.splrep(times_before, y, s=0)
      field_new[:,point_i] = interpolate.splev(times_new, tck, der=0)
      field_new_der[:,point_i] = interpolate.splev(times_new, tck, der=1)
      field_new_der2[:,point_i] = interpolate.splev(times_new, tck, der=2)

    return field_new, field_new_der, field_new_der2


def load_centerline_data(fpath_1d):
    cent = read_centerline(fpath_1d).GetOutput()
    cent_array = get_all_arrays(cent)

    #Extract Geometry ----------------------------------------------------
    pt_id = cent_array["GlobalNodeId"].astype(int)
    num_pts = np.size(pt_id)  # number of points in mesh
    branch_id = cent_array["BranchId"].astype(int)
    junction_id = cent_array["BifurcationId"].astype(int)
    axial_distance = cent_array["Path"]  # distance along centerline
    area = cent_array["CenterlineSectionArea"]
    direction = cent_array["CenterlineSectionNormal"]  # vector normal direction
    direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
    direction = np.transpose(np.divide(direction,direction_norm))  # normalized direction vector
    angle1 = direction[0,:].reshape(-1,)
    angle2 = direction[1,:].reshape(-1,)
    angle3 = direction[2,:].reshape(-1,)
    return pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3

def identify_junctions(junction_id, branch_id, pt_id):
    junction_ids = np.linspace(0,max(junction_id),max(junction_id)+1).astype(int)
    branch_ids = np.linspace(0,max(branch_id),max(branch_id)+1).astype(int)
    junction_dict = {}
    for i in junction_ids:

        junction_pts = pt_id[junction_id == i] # find all points in junction
        branch_pts_junc = [] # inlet and outlet point ids of junction
        branch_ids_junc = [] # branch ids of junction

        base_branch = branch_id[pt_id == min(junction_pts)-1]
        base_branch_pts = pt_id[branch_id == base_branch]

        branch_pts_junc.append(max([min(junction_pts)-1, min(base_branch_pts)])) # find "inlet" of junction (point with smallest Id)
        branch_ids_junc.append(branch_id[pt_id == min(junction_pts)-1][0]) # find the branch to which the inlet belongs
        branch_counter = 1 # initialize counter for the number of branches
        # loop over all branches in model
        for j in branch_ids:
            branch_pts = pt_id[branch_id == j] # find points belonging to branch
            shared_pts = np.intersect1d(junction_pts+1, branch_pts) # find points adjacent to the junction
            # if there is an adjacent point
            if len(shared_pts) != 0 and j not in branch_ids_junc : # if there is a shared point in the branch
                branch_counter = branch_counter + 1 # increment branch counter
                branch_ids_junc.append(j.astype(int)) # add outlet branch Id to outlet branch array
                branch_pts_junc.append(min([min(branch_pts).astype(int), max(branch_pts)])) # add outlet point Id to outlet point array
        junction_dict.update({i : branch_pts_junc})
        #assert i == 0, "There should only be one junction,"
    return junction_dict, branch_pts_junc

def identify_junctions100(junction_id, branch_id, pt_id):
    junction_ids = np.linspace(0,max(junction_id),max(junction_id)+1).astype(int)
    branch_ids = np.linspace(0,max(branch_id),max(branch_id)+1).astype(int)
    junction_dict = {}
    for i in junction_ids:

        junction_pts = pt_id[junction_id == i] # find all points in junction
        branch_pts_junc = [] # inlet and outlet point ids of junction
        branch_ids_junc = [] # branch ids of junction
        branch_pts_junc.append(min(pt_id[branch_id == 0])+5) # find "inlet" of junction (point with smallest Id)
        branch_ids_junc.append(branch_id[pt_id == min(junction_pts)-1][0]) # find the branch to which the inlet belongs
        branch_counter = 1 # initialize counter for the number of branches
        # loop over all branches in model
        for j in branch_ids:
            branch_pts = pt_id[branch_id == j] # find points belonging to branch
            shared_pts = np.intersect1d(junction_pts+1, branch_pts) # find points adjacent to the junction
            # if there is an adjacent point
            if len(shared_pts) != 0 and j not in branch_ids_junc : # if there is a shared point in the branch
                branch_counter = branch_counter + 1 # increment branch counter
                branch_ids_junc.append(j.astype(int)) # add outlet branch Id to outlet branch array
                branch_pts_junc.append(max(branch_pts).astype(int)-10) # add outlet point Id to outlet point array
        junction_dict.update({i : branch_pts_junc})
        assert i == 0, "There should only be one junction,"
    return junction_dict, branch_pts_junc

def identify_junctions_end(junction_id, branch_id, pt_id):
    junction_ids = np.linspace(0,max(junction_id),max(junction_id)+1).astype(int)
    branch_ids = np.linspace(0,max(branch_id),max(branch_id)+1).astype(int)
    junction_dict = {}
    for i in junction_ids:

        junction_pts = pt_id[junction_id == i] # find all points in junction
        branch_pts_junc = [] # inlet and outlet point ids of junction
        branch_ids_junc = [] # branch ids of junction
        branch_pts_junc.append(min(pt_id[branch_id == 0])+1) # find "inlet" of junction (point with smallest Id)
        branch_ids_junc.append(branch_id[pt_id == min(junction_pts)-1][0]) # find the branch to which the inlet belongs
        branch_counter = 1 # initialize counter for the number of branches
        # loop over all branches in model
        for j in branch_ids:
            branch_pts = pt_id[branch_id == j] # find points belonging to branch
            shared_pts = np.intersect1d(junction_pts+1, branch_pts) # find points adjacent to the junction
            # if there is an adjacent point
            if len(shared_pts) != 0 and j not in branch_ids_junc : # if there is a shared point in the branch
                branch_counter = branch_counter + 1 # increment branch counter
                branch_ids_junc.append(j.astype(int)) # add outlet branch Id to outlet branch array
                branch_pts_junc.append(max(branch_pts).astype(int)-1) # add outlet point Id to outlet point array
        junction_dict.update({i : branch_pts_junc})
        assert i == 0, "There should only be one junction,"
    return junction_dict, branch_pts_junc


def get_junction_pts(flow, junction_id, junction_dict):

    inlets = []; outlets = [] # initialize inlet and outlet list
    branch_pts_junc = junction_dict[junction_id] #
    return branch_pts_junc

def load_soln_data(soln_dict):

    pressure_in_time = soln_dict["pressure_in_time"]
    flow_in_time = soln_dict["flow_in_time"]
    times = soln_dict["times"]
    time_interval = 0.1

    return pressure_in_time, flow_in_time, times, time_interval

def classify_branches(flow, junction_id, junction_dict):
    inlets = []; outlets = [] # initialize inlet and outlet list
    branch_pts_junc = junction_dict[junction_id] #
    for branch_pt in branch_pts_junc:
      if branch_pt == min(branch_pts_junc) and flow[np.isin(branch_pts_junc, branch_pt)] > 0:
          inlets.append(branch_pt)
      elif branch_pt == min(branch_pts_junc) and flow[np.isin(branch_pts_junc, branch_pt)] < 0:
          outlets.append(branch_pt)
      elif flow[np.isin(branch_pts_junc, branch_pt)] > 0:
          outlets.append(branch_pt)
      elif flow[np.isin(branch_pts_junc, branch_pt)] < 0:
          inlets.append(branch_pt)
      else:
          outlets.append(branch_pt)
    #import pdb; pdb.set_trace()
    return inlets, outlets

def get_inlet_outlet_pairs(num_inlets, num_outlets):

    inlet_list = []; outlet_list = []
    for inlet in range(num_inlets):
        for outlet in range(num_outlets):
            inlet_list.append(inlet); outlet_list.append(outlet)
    inlet_outlet_pairs = (tf.convert_to_tensor(inlet_list, dtype=tf.int32),
                            tf.convert_to_tensor(outlet_list, dtype=tf.int32))
    return inlet_outlet_pairs

def get_outlet_pairs(num_outlets):
    outlet_list1 = []; outlet_list2 = []
    for outlet1 in range(num_outlets):
        for outlet2 in range(num_outlets):
            outlet_list1.append(outlet1); outlet_list2.append(outlet2)
    outlet_pairs = (tf.convert_to_tensor(outlet_list1, dtype=tf.int32),
                            tf.convert_to_tensor(outlet_list2, dtype=tf.int32))
    return outlet_pairs

def get_angle_diff(angle1, angle2):
    try:
        angle_diff = np.arccos(np.dot(angle1, angle2))
    except:
        pdb.set_trace()
    return angle_diff

def get_flow_hist(flow_in_time_aug, time_index, num_time_steps_model):
    if time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[0]:
      flow_hist1 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist2 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    elif time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[1]:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    elif time_index == np.linspace(0, num_time_steps_model-1, num_time_steps_model).astype(int)[2]:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = flow_in_time_aug[time_index-2, :] # velocity at timestep
      flow_hist3 = 0*flow_in_time_aug[time_index, :] # velocity at timestep
    else:
      flow_hist1 = flow_in_time_aug[time_index-1, :] # velocity at timestep
      flow_hist2 = flow_in_time_aug[time_index-2, :] # velocity at timestep
      flow_hist3 = flow_in_time_aug[time_index-3, :] # velocity at timestep
    return flow_hist1, flow_hist2, flow_hist3

def scale(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
    scaled_field = (field-mean)/std
    return scaled_field

def extract_model_data():
    aug_factor = 1

    pressure_in_sum = 0
    pressure_in_rel_sum = 0
    pressure_der_in_sum = 0
    flow_in_sum = 0
    flow_der_in_sum = 0
    flow_der2_in_sum = 0
    flow_hist1_in_sum = 0
    flow_hist2_in_sum = 0
    flow_hist3_in_sum = 0
    area_in_sum = 0
    time_interval_in_sum = 0
    flow_out_sum = 0
    flow_der_out_sum = 0
    flow_der2_out_sum = 0
    flow_hist1_out_sum = 0
    flow_hist2_out_sum = 0
    flow_hist3_out_sum = 0
    area_out_sum = 0
    pressure_out_sum = 0
    pressure_out_rel_sum = 0

    pressure_in_sum2 = 0
    pressure_in_rel_sum2 = 0
    pressure_der_in_sum2 = 0
    flow_in_sum2 = 0
    flow_der_in_sum2 = 0
    flow_der2_in_sum2 = 0
    flow_hist1_in_sum2 = 0
    flow_hist2_in_sum2 = 0
    flow_hist3_in_sum2 = 0
    area_in_sum2 = 0
    time_interval_in_sum2 = 0
    flow_out_sum2 = 0
    flow_der_out_sum2 = 0
    flow_der2_out_sum2 = 0
    flow_hist1_out_sum2 = 0
    flow_hist2_out_sum2 = 0
    flow_hist3_out_sum2 = 0
    area_out_sum2 = 0
    pressure_out_sum2 = 0
    pressure_out_rel_sum2 = 0

    home_dir = os.path.expanduser("~")
    #dir = home_dir + "/Desktop/junction_sim_files"
    geos = os.listdir(f"{home_dir}/Desktop/synthetic_junction_results"); #
    #geos = os.listdir(f"{home_dir}/Desktop/res_sweep"); #
    geos.sort()
    print(geos)
    cnt = 0
    #import pdb; pdb.set_trace()
    for geo in geos:
        print(geo)
        if os.path.exists(f"{home_dir}/Desktop/synthetic_junction_results/{geo}") == False:
            print("making directory")
            os.mkdir(f"{home_dir}/Desktop/synthetic_junction_results/{geo}")
        try:
            pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = f"{home_dir}/Desktop/centerlines/{geo}/centerline.vtp")
            junction_dict, junc_pt_ids = identify_junctions(junction_id, branch_id, pt_id)
            #print("centerline data loaded")
        except:
            print("Geometry Error.")
            continue

        for flow_ind in range(8):

            try:
                if os.path.exists(f"{home_dir}/Desktop/synthetic_junction_results/{geo}/flow_{flow_ind}_avg_sol"):
                    soln_dict = load_dict(f"{home_dir}/Desktop/synthetic_junction_results/{geo}/flow_{flow_ind}_avg_sol")
                    print(f"Flow {flow_ind} average results already exist.")
                else:
                    print("Averaging 3D results.")

                    #import pdb; pdb.set_trace()

                    soln_dict = get_avg_results(fpath_1d = f"{home_dir}/Desktop/centerlines/{geo}/centerline.vtp",
                                    fpath_3d = f"{home_dir}/Desktop/synthetic_junctions/{geo}/solution_flow_{flow_ind}.vtu",
                                    fpath_out = f"{home_dir}/Desktop/synthetic_junction_results/{geo}/flow_{flow_ind}_avg_sol",
                                    pt_inds = junc_pt_ids, only_caps=False)

                pressure_in_time, flow_in_time, times, time_interval = load_soln_data(soln_dict)


                # Compute flow time_index derivative ----------------------------------------
                times = np.asarray(times)
                time_sort = np.argsort(times); times = times[time_sort] # sort timestep array
                pressure_in_time = pressure_in_time[time_sort, :] # sort pressure_in_time array
                flow_in_time = flow_in_time[time_sort, :] # sort flow_in_time_index array
                pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2 = augment_time(pressure_in_time, times, aug_factor)
                flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2 = augment_time(flow_in_time, times, aug_factor)
                num_time_steps_model = flow_in_time_aug.shape[0]


                # Extract feature values at every timestep ----------------------------------------

                for time_index in np.linspace(0, num_time_steps_model, num_time_steps_model, endpoint=False).astype(int):
                    if time_index < 3:
                        continue
                    pressure = pressure_in_time_aug[time_index, :] # pressure at timestep
                    if np.any(np.isnan(pressure)):
                        print("NaN: skipping model.")
                        print(pressure)
                        continue
                    pressure_der = pressure_in_time_aug_der[time_index, :] # pressure at timestep
                    flow = flow_in_time_aug[time_index, :] # velocity at timestep

                    flow_hist1, flow_hist2, flow_hist3 = get_flow_hist(flow_in_time_aug, time_index, num_time_steps_model) # get flow history
                    flow_der = flow_in_time_aug_der[time_index, :] # velocity at timestep
                    flow_der2 = flow_in_time_aug_der2[time_index, :] # velocity at timestep

                    time_interval = 0.01
                    for junction_id in junction_dict.keys():

                        inlets, outlets = classify_branches(flow, junction_id, junction_dict)
                        inlet_pts = np.isin(pt_id, inlets); outlet_pts = np.isin(pt_id, outlets)
                        if len(inlets) != 1 or len(outlets) != 2:
                            continue
                        if area[inlet_pts] < area[outlet_pts][0] or area[inlet_pts] < area[outlet_pts][1]:
                            continue
                        flow = np.abs(flow)

                        min_pressure_in = 2* np.min(pressure[np.isin(junc_pt_ids, inlets)])
                        pressure_in_sum += 2* pressure[np.isin(junc_pt_ids, inlets)]
                        pressure_in_rel_sum += 2* pressure[np.isin(junc_pt_ids, inlets)] - min_pressure_in
                        pressure_der_in_sum += 2* pressure_der[np.isin(junc_pt_ids, inlets)]
                        flow_in_sum += 2* flow[np.isin(junc_pt_ids, inlets)]
                        vel_in_sum += 2* flow[np.isin(junc_pt_ids, inlets)]/area[inlet_pts]
                        flow_der_in_sum += 2* flow_der[np.isin(junc_pt_ids, inlets)]
                        vel_der_in_sum += 2* flow_der[np.isin(junc_pt_ids, inlets)]/area[inlet_pts]
                        flow_der2_in_sum += 2* flow_der2[np.isin(junc_pt_ids, inlets)]
                        flow_hist1_in_sum += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]
                        flow_hist2_in_sum += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]
                        flow_hist3_in_sum += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]
                        area_in_sum += 2* area[inlet_pts]
                        time_interval_in_sum += 2* time_interval

                        flow_out_sum += sum(flow[np.isin(junc_pt_ids, outlets)])
                        flow_der_out_sum += sum(flow_der[np.isin(junc_pt_ids, outlets)])
                        flow_der2_out_sum += sum(flow_der2[np.isin(junc_pt_ids, outlets)])
                        flow_hist1_out_sum += sum(flow_hist1[np.isin(junc_pt_ids, outlets)])
                        flow_hist2_out_sum += sum(flow_hist1[np.isin(junc_pt_ids, outlets)])
                        flow_hist3_out_sum += sum(flow_hist1[np.isin(junc_pt_ids, outlets)])
                        area_out_sum += sum(area[outlet_pts])
                        pressure_out_rel_sum += sum(pressure[np.isin(junc_pt_ids, outlets)] - min_pressure_in)
                        pressure_out_sum += sum(pressure[np.isin(junc_pt_ids, outlets)])


                        pressure_in_sum2 += 2* pressure[np.isin(junc_pt_ids, inlets)]**2
                        pressure_in_rel_sum2 += 2* (pressure[np.isin(junc_pt_ids, inlets)] - min_pressure_in)**2
                        pressure_der_in_sum2 += 2* pressure_der[np.isin(junc_pt_ids, inlets)]**2
                        flow_in_sum2 += 2* flow[np.isin(junc_pt_ids, inlets)]**2
                        flow_der_in_sum2 += 2* flow_der[np.isin(junc_pt_ids, inlets)]**2
                        flow_der2_in_sum2 += 2* flow_der2[np.isin(junc_pt_ids, inlets)]**2
                        flow_hist1_in_sum2 += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]**2
                        flow_hist2_in_sum2 += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]**2
                        flow_hist3_in_sum2 += 2* flow_hist1[np.isin(junc_pt_ids, inlets)]**2
                        area_in_sum2 += 2* area[inlet_pts]**2
                        time_interval_in_sum2 += 2* time_interval**2

                        flow_out_sum2 += sum(flow[np.isin(junc_pt_ids, outlets)]**2)
                        flow_der_out_sum2 += sum(flow_der[np.isin(junc_pt_ids, outlets)]**2)
                        flow_der2_out_sum2 += sum(flow_der2[np.isin(junc_pt_ids, outlets)]**2)
                        flow_hist1_out_sum2 += sum(flow_hist1[np.isin(junc_pt_ids, outlets)]**2)
                        flow_hist2_out_sum2 += sum(flow_hist1[np.isin(junc_pt_ids, outlets)]**2)
                        flow_hist3_out_sum2 += sum(flow_hist1[np.isin(junc_pt_ids, outlets)]**2)
                        area_out_sum2 += sum(area[outlet_pts]**2)
                        pressure_out_rel_sum2 += sum((pressure[np.isin(junc_pt_ids, outlets)] - min_pressure_in)**2)
                        pressure_out_sum2 += sum(pressure[np.isin(junc_pt_ids, outlets)]**2)

                        cnt += 2
            except:
                print("Flow error.")
                continue


    scaling_dict = {"pressure_in": [pressure_in_sum/cnt, np.sqrt((pressure_in_sum2 - ((pressure_in_sum**2)/cnt))/cnt)],
                    "pressure_der_in": [pressure_der_in_sum/cnt, np.sqrt((pressure_der_in_sum2 - ((pressure_der_in_sum**2)/cnt))/cnt)],
                    "pressure_in_rel": [pressure_in_rel_sum/cnt, np.sqrt((pressure_in_rel_sum2 - ((pressure_in_rel_sum**2)/cnt))/cnt)],
                    "flow_in": [flow_in_sum/cnt, np.sqrt((flow_in_sum2 - ((flow_in_sum**2)/cnt))/cnt)],
                    "flow_der_in": [flow_der_in_sum/cnt, np.sqrt((flow_der_in_sum2 - ((flow_der_in_sum**2)/cnt))/cnt)],
                    "flow_der2_in": [flow_der2_in_sum/cnt, np.sqrt((flow_der2_in_sum2 - ((flow_der2_in_sum**2)/cnt))/cnt)],
                    "flow_hist1_in": [flow_hist1_in_sum/cnt, np.sqrt((flow_hist1_in_sum2 - ((flow_hist1_in_sum**2)/cnt))/cnt)],
                    "flow_hist2_in": [flow_hist2_in_sum/cnt, np.sqrt((flow_hist2_in_sum2 - ((flow_hist2_in_sum**2)/cnt))/cnt)],
                    "flow_hist3_in": [flow_hist3_in_sum/cnt, np.sqrt((flow_hist3_in_sum2 - ((flow_hist3_in_sum**2)/cnt))/cnt)],
                    "area_in": [area_in_sum/cnt, np.sqrt((area_in_sum2 - ((area_in_sum**2)/cnt))/cnt)],
                    "time_interval_in": [time_interval_in_sum/cnt, np.sqrt(0)], # (time_interval_in_sum2 - ((time_interval_in_sum**2)/cnt))/cnt)
                    "flow_out": [flow_out_sum/cnt, np.sqrt((flow_out_sum2 - ((flow_out_sum**2)/cnt))/cnt)],
                    "flow_der_out": [flow_der_out_sum/cnt, np.sqrt((flow_der_out_sum2 - ((flow_der_out_sum**2)/cnt))/cnt)],
                    "flow_der2_out": [flow_der2_out_sum/cnt, np.sqrt((flow_der2_out_sum2 - ((flow_der2_out_sum**2)/cnt))/cnt)],
                    "flow_hist1_out": [flow_hist1_out_sum/cnt, np.sqrt((flow_hist1_out_sum2 - ((flow_hist1_out_sum**2)/cnt))/cnt)],
                    "flow_hist2_out": [flow_hist2_out_sum/cnt, np.sqrt((flow_hist2_out_sum2 - ((flow_hist2_out_sum**2)/cnt))/cnt)],
                    "flow_hist3_out": [flow_hist3_out_sum/cnt, np.sqrt((flow_hist3_out_sum2 - ((flow_hist3_out_sum**2)/cnt))/cnt)],
                    "area_out": [area_out_sum/cnt, np.sqrt((area_out_sum2 - ((area_out_sum**2)/cnt))/cnt)],
                    "pressure_out": [pressure_out_sum/cnt, np.sqrt((pressure_out_sum2 - ((pressure_out_sum**2)/cnt))/cnt)],
                    "pressure_out_rel": [pressure_out_rel_sum/cnt, np.sqrt((pressure_out_rel_sum2 - ((pressure_out_rel_sum**2)/cnt))/cnt)]}

    print(f"Scaling Dictionary: \n {scaling_dict}")
    dir_save = f"{home_dir}/Desktop/junction_GNN/data/"
    save_dict(scaling_dict, dir_save + "synthetic_scaling_dict")
    #save_dict(data_dict, dir_save + "synthetic_data_lists")

def get_avg_sol_res_sweep():

    home_dir = os.path.expanduser("~")
    geos = os.listdir(f"{home_dir}/Desktop/res_sweep"); #
    geos.sort()
    print(geos)
    cnt = 0
    #import pdb; pdb.set_trace()
    for geo in geos:
        print(geo)
        if os.path.exists(f"{home_dir}/Desktop/res_sweep/synthetic_junction_results/{geo}") == False:
            print("making directory")
            os.mkdir(f"{home_dir}/Desktop/res_sweep/synthetic_junction_results/{geo}")
        try:
            pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(geo_name = geo)
            junction_dict, junc_pt_ids = identify_junctions(junction_id, branch_id, pt_id)
            #print("centerline data loaded")
        except:
            print("Geometry Error.")
            continue

        for res in [1000, 2000, 5000, 10000, 20000, 30000]:
            print(f"Res = {res}")
            # try:

            try:
                if os.path.exists(f"{home_dir}/Desktop/res_sweep/synthetic_junction_results/{geo}/res_{res}_avg_sol"):
                    soln_dict = load_dict(f"{home_dir}/Desktop/res_sweep/synthetic_junction_results/{geo}/res_{res}_avg_sol")
                else:
                    print("Averaging 3D results.")

                    #import pdb; pdb.set_trace()

                    soln_dict = get_avg_results(fpath_1d = f"{home_dir}/Desktop/centerlines/{geo}/centerline.vtp",
                                    fpath_3d = f"{home_dir}/Desktop/res_sweep/{geo}/solution_res_{res}.vtu",
                                    fpath_out = f"{home_dir}/Desktop/res_sweep/synthetic_junction_results/{geo}/res_{res}_avg_sol",
                                    pt_inds = junc_pt_ids, only_caps=False)
            except:
                print("Error")
    return

def get_avg_sol_inflow_sweep():

    home_dir = os.path.expanduser("~")
    geos = os.listdir(f"{home_dir}/Desktop/inflow_sweep"); #
    geos.sort()
    print(geos)
    cnt = 0
    #import pdb; pdb.set_trace()
    for geo in geos:
        print(geo)
        if os.path.exists(f"{home_dir}/Desktop/inflow_sweep/synthetic_junction_results/{geo}") == False:
            print("making directory")
            os.mkdir(f"{home_dir}/Desktop/inflow_sweep/synthetic_junction_results/{geo}")
        try:
            pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3 = load_centerline_data(fpath_1d = \
            f"{home_dir}/Desktop/centerlines/geom_0/centerline.vtp")
            junction_dict, junc_pt_ids = identify_junctions(junction_id, branch_id, pt_id)
            #print("centerline data loaded")
        except:
            print("Geometry Error.")
            continue

        for inflow in [5,10,20,40,80]:
            print(f"inflow = {inflow}")
            # try:

            try:
                if os.path.exists(f"{home_dir}/Desktop/inflow_sweep/synthetic_junction_results/{geo}/inflow_{inflow}_avg_sol"):
                    soln_dict = load_dict(f"{home_dir}/Desktop/inflow_sweep/synthetic_junction_results/{geo}/inflow_{inflow}_avg_sol")
                else:
                    print("Averaging 3D results.")

                    #import pdb; pdb.set_trace()

                    soln_dict = get_avg_results(fpath_1d = f"{home_dir}/Desktop/centerlines/{geo}/centerline.vtp",
                                    fpath_3d = f"{home_dir}/Desktop/inflow_sweep/{geo}/solution_inflow_{inflow}.vtu",
                                    fpath_out = f"{home_dir}/Desktop/inflow_sweep/synthetic_junction_results/{geo}/inflow_{inflow}_avg_sol",
                                    pt_inds = junc_pt_ids, only_caps=False)
            except:
                print("Error")
    return

if __name__ == "__main__":
    extract_model_data()
    #get_avg_sol_inflow_sweep()
