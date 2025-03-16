import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.junction_proc import *
from util.tools.vtk_functions import *
import matplotlib.pyplot as plt
from util.zerod.standard_to_RRI import get_RRI_values
#from sklearn.linear_model import LinearRegression
#from util.junction_extraction_util.graph_handling import *

def print_stats(char_val_dict, anatomy, value):
    dat = np.asarray(char_val_dict[anatomy][value])
    print(f"{anatomy} {value} statistics:")

    print(f"Min: {np.min(dat)}.  Max {np.max(dat)}.  Med {np.median(dat)}")
    print(f"Mean: {np.mean(dat)}.  Standard Deviation {np.std(dat)}.")
    print("---------------------------------------------------------")
    return

def check_energy_conservation(tree_name):
    """
    Compile list of junction graphs (from synthetic data)
    """

    # relevant file paths:
    soln_dir = f"trees/threed_output_cent/{tree_name}"
    soln_dir = f"trees/zerod_output_cent_TP/{tree_name}"

    pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, pressure_in_time, flow_in_time, times, time_interval = \
    load_vmr_model_data("centerline_sol_half.vtp", soln_dir)
    junction_dict, offsets, branch_pts_junc = identify_junctions_offset(junction_id, branch_id, pt_id, path, offset=00)

    pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
    flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time, pressure_in_time, times)

    for junction_id in junction_dict.keys():

        max_flow_ind = np.argmax(flow_in_time_aug[:,np.argmax(np.abs(flow_in_time_aug[:,get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])

        flow, flow_hist1, flow_hist2, flow_hist3, flow_der, flow_der2, pressure, pressure_der = get_soln_at_time(\
        flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, pressure_in_time_aug, pressure_in_time_aug_der, max_flow_ind)
        
        inlets, outlets = classify_branches(flow, junc_pts = junction_dict[junction_id], pt_arr = pt_id)
        inlet_pts = get_inds(arr = pt_id, vals = inlets); outlet_pts = get_inds(arr = pt_id, vals = outlets)

        inlet_area = area[inlet_pts][0]
        outlet_area = area[outlet_pts]
        inlet_angles = np.asarray([angle1[inlet_pts],
                                angle2[inlet_pts],
                                angle3[inlet_pts]])

        outlet_angles = np.asarray([angle1[outlet_pts],
                                angle2[outlet_pts],
                                angle3[outlet_pts]])

        angle_diff1 = get_angle_diff(inlet_angles, outlet_angles[:,0])
        angle_diff2 = get_angle_diff(inlet_angles, outlet_angles[:,1])
        angle_diffs = [angle_diff1, angle_diff2] #angle_diffs.reshape((len(outlet_pts),))
        if len(junction_dict[junction_id]) == 4:
            angle_diff3 = get_angle_diff(inlet_angles, outlet_angles[:,2])
            angle_diffs.append(angle_diff3)
        
        #outlet_angle_diffs = get_angle_diff(outlet_angles[0,:],outlet_angles[1,:])
        
        #pdb.set_trace()
        flow = list(flow_in_time_aug[max_flow_ind, outlet_pts])
        inlet_flow = flow_in_time_aug[max_flow_ind, inlet_pts][0]

        p = list(pressure_in_time_aug[max_flow_ind, outlet_pts])
        inlet_p = pressure_in_time_aug[max_flow_ind, inlet_pts][0]

        dp = list(pressure_in_time_aug[max_flow_ind, outlet_pts] - pressure_in_time_aug[max_flow_ind, inlet_pts])

        radius = list(np.sqrt(area[outlet_pts]/np.pi))
        inlet_radius = np.sqrt(area[inlet_pts]/np.pi)

        max_out_ind = np.argmax(flow_in_time_aug[max_flow_ind, outlet_pts])
        min_out_ind = np.argmin(flow_in_time_aug[max_flow_ind, outlet_pts])

        inlet_energy = inlet_flow * (inlet_p + 0.5*(inlet_flow/inlet_area)**2)
        outlet_energy = [flow[out_ind] * (p[out_ind] + 0.5*(flow[out_ind]/outlet_area[out_ind])**2) for out_ind in range(len(outlet_pts))]
        print(f"Percent energy loss over junction: {(inlet_energy - sum(outlet_energy))/inlet_energy}")

    pdb.set_trace()
    return

if __name__ == "__main__":
    tree_name = "tree_80"
    check_energy_conservation(tree_name)