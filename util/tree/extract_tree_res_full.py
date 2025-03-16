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

def extract_characteristic_values(tree_name):
    """
    Compile list of junction graphs (from synthetic data)
    """

    # relevant file paths:
    # results_dir = "data/CCO_tree/char_val_dict"
    results_dir = f"data/tree_80_resistance_dict"
    if not os.path.exists("data/characteristic_value_dictionaries"):
        os.makedirs("data/characteristic_value_dictionaries")
    # soln_dir = "data/CCO_tree"
    soln_dir = f"trees/threed_output_cent/{tree_name}"
    # centerline_dir = "data/CCO_tree/geometry/centerlines.vtp"
    char_val_dict = {}
    models = ["centerline_sol_aug.vtp",]#[x.strip() for x in content].copy()
    resistance_dict = defaultdict(list)
    for model in models:

        anatomy = "CCO"
        print(f"Model {model} ({anatomy})")

        if anatomy not in char_val_dict.keys():
            char_val_dict.update({anatomy: {"daughter1_radius": [],
                                            "daughter2_radius": [],
                                            "inlet_radius": [],
                                            "inlet_area": [],
                                            "outlet_area": [],
                                            "daughter1_flow": [],
                                            "daughter2_flow": [],
                                            "inlet_flow": [],
                                            "inlet_velocity": [],
                                            "daughter1_velocity": [],
                                            "daughter2_velocity": [],
                                            "daughter1_P": [],
                                            "daughter2_P": [],
                                            "inlet_P": [],
                                            "daughter1_P_dyn": [],
                                            "daughter2_P_dyn": [],
                                            "inlet_P_dyn": [],
                                            "daughter1_dP": [],
                                            "daughter2_dP": [],
                                            "daughter1_dP_dyn": [],
                                            "daughter2_dP_dyn": [],
                                            "daughter1_dP_stat": [],
                                            "daughter2_dP_stat": [],
                                            "daughter1_angle": [],
                                            "daughter2_angle": [],
                                            "inlet_angles": [],
                                            "outlet_angles": [],
                                            "r_lin_1": [],
                                            "r_quad_1": [],
                                            "r_lin_star1": [],
                                            "r_lin_2": [],
                                            "r_quad_2": [],
                                            "r_lin_star2": [],
                                            "r_lin_1_pred": [],
                                            "r_lin_star1_pred": [],
                                            "r_lin_2_pred": [],
                                            "r_lin_star2_pred": [],
                                            "name": []}})

        pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, path, pressure_in_time, flow_in_time, times, time_interval = \
        load_vmr_model_data("centerline_sol_full.vtp", soln_dir)

        junction_dict, offsets, branch_pts_junc = identify_junctions_offset(junction_id, branch_id, pt_id, path, offset=00)


        pressure_in_time_aug, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
        flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time, pressure_in_time, times)

        for junction_id in junction_dict.keys():

            
            max_flow_ind = np.argmax(flow_in_time_aug[:,np.argmax(np.abs(flow_in_time_aug[:,get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])


            flow, flow_hist1, flow_hist2, flow_hist3, flow_der, flow_der2, pressure, pressure_der = get_soln_at_time(\
            flow_in_time_aug, flow_in_time_aug_der, flow_in_time_aug_der2, pressure_in_time_aug, pressure_in_time_aug_der, max_flow_ind)
            #if verify_bifurcation(flow, area, pressure = pressure, junc_pts = junction_dict[junction_id], pt_arr = pt_id) == False: continue
            inlets, outlets = classify_branches(flow, junc_pts = junction_dict[junction_id], pt_arr = pt_id)
            inlet_pts = get_inds(arr = pt_id, vals = inlets); outlet_pts = get_inds(arr = pt_id, vals = outlets)

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

            flow = list(flow_in_time_aug[max_flow_ind, outlet_pts])
            inlet_flow = flow_in_time_aug[max_flow_ind, inlet_pts]

            p = list(pressure_in_time_aug[max_flow_ind, outlet_pts])
            inlet_p = pressure_in_time_aug[max_flow_ind, inlet_pts]

            dp = list(pressure_in_time_aug[max_flow_ind, outlet_pts] - pressure_in_time_aug[max_flow_ind, inlet_pts])


            r_lin_list = []
            for outlet_ind, outlet_pt in enumerate(outlet_pts):
                # Add first bifurcation
                outlet_flow = flow[outlet_ind]
                outlet_dP = dp[outlet_ind]
                R = outlet_dP/outlet_flow
                resistance_dict[junction_id].append(-R)
    save_dict(resistance_dict, results_dir)
    pdb.set_trace()
    return

if __name__ == '__main__':
    extract_characteristic_values(tree_name="tree_80")

