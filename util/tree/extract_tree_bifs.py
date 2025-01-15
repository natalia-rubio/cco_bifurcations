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
    results_dir = f"data/characteristic_value_dictionaries/{tree_name}_char_val_dict"
    if not os.path.exists("data/characteristic_value_dictionaries"):
        os.makedirs("data/characteristic_value_dictionaries")
    # soln_dir = "data/CCO_tree"
    soln_dir = f"trees/threed_output_cent/{tree_name}"
    # centerline_dir = "data/CCO_tree/geometry/centerlines.vtp"
    char_val_dict = {}
    models = ["centerline_sol_aug.vtp",]#[x.strip() for x in content].copy()

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
                                            "daughter1_P": [],
                                            "daughter2_P": [],
                                            "inlet_P": [],
                                            "daughter1_dP": [],
                                            "daughter2_dP": [],
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

        pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, pressure_in_time_half, flow_in_time_half, times, time_interval = \
        load_vmr_model_data("centerline_sol_half.vtp", soln_dir)
        junction_dict, branch_pts_junction = identify_junctions_offset(junction_id, branch_id, pt_id, offset=00)

        pt_id, num_pts, branch_id, junction_id, area, angle1, angle2, angle3, pressure_in_time_full, flow_in_time_full, times, time_interval = \
        load_vmr_model_data("centerline_sol_full.vtp", soln_dir)

        pressure_in_time_aug_half, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
        flow_in_time_aug_half, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time_half, pressure_in_time_half, times)

        pressure_in_time_aug_full, pressure_in_time_aug_der, pressure_in_time_aug_der2,\
        flow_in_time_aug_full, flow_in_time_aug_der, flow_in_time_aug_der2, num_time_steps_model = process_soln(flow_in_time_full, pressure_in_time_full, times)

        for junction_id in junction_dict.keys():
            #import pdb; pdb.set_trace()
            #max_flow_ind = np.argmax(flow_in_time_aug[:,np.argmax(np.abs(flow_in_time_aug[:, get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])
            if len(junction_dict[junction_id]) > 3:
                print(f"Junction {junction_id} has more than 3 outlets.")
                continue
            max_flow_ind_half = np.argmax(flow_in_time_aug_half[:,np.argmax(np.abs(flow_in_time_aug_half[:,get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])
            max_flow_ind_full = np.argmax(flow_in_time_aug_full[:,np.argmax(np.abs(flow_in_time_aug_full[:,get_inds(arr = pt_id, vals = junction_dict[junction_id])]))])


            flow_half, flow_hist1, flow_hist2, flow_hist3, flow_der, flow_der2, pressure_half, pressure_der = get_soln_at_time(\
            flow_in_time_aug_half, flow_in_time_aug_der, flow_in_time_aug_der2, pressure_in_time_aug_half, pressure_in_time_aug_der, max_flow_ind_half)

            flow_full, flow_hist1, flow_hist2, flow_hist3, flow_der, flow_der2, pressure_full, pressure_der = get_soln_at_time(\
            flow_in_time_aug_full, flow_in_time_aug_der, flow_in_time_aug_der2, pressure_in_time_aug_full, pressure_in_time_aug_der, max_flow_ind_full)
            #if verify_bifurcation(flow, area, pressure = pressure, junc_pts = junction_dict[junction_id], pt_arr = pt_id) == False: continue
            inlets, outlets = classify_branches(flow_half, junc_pts = junction_dict[junction_id], pt_arr = pt_id)
            inlet_pts = get_inds(arr = pt_id, vals = inlets); outlet_pts = get_inds(arr = pt_id, vals = outlets)

            inlet_angles = np.asarray([angle1[inlet_pts],
                                    angle2[inlet_pts],
                                    angle3[inlet_pts]])

            outlet_angles = np.asarray([angle1[outlet_pts],
                                    angle2[outlet_pts],
                                    angle3[outlet_pts]])

            angle_diffs = get_angle_diff(inlet_angles, outlet_angles)
            angle_diffs = angle_diffs.reshape((len(outlet_pts),))

            outlet_angle_diffs = get_angle_diff(outlet_angles[0,:],outlet_angles[1,:])
            

            flow_half = list(flow_in_time_aug_half[max_flow_ind_half, outlet_pts])
            flow_full = list(flow_in_time_aug_full[max_flow_ind_full, outlet_pts])
            inlet_flow_half = flow_in_time_aug_half[max_flow_ind_half, inlet_pts]
            inlet_flow_full = flow_in_time_aug_full[max_flow_ind_full, inlet_pts]

            p_half = list(pressure_in_time_aug_half[max_flow_ind_half, outlet_pts])
            p_full = list(pressure_in_time_aug_full[max_flow_ind_full, outlet_pts])
            inlet_p_half = pressure_in_time_aug_half[max_flow_ind_half, inlet_pts]
            inlet_p_full = pressure_in_time_aug_full[max_flow_ind_full, inlet_pts]

            dp_half = list(pressure_in_time_aug_half[max_flow_ind_half, outlet_pts] - pressure_in_time_aug_half[max_flow_ind_half, inlet_pts])
            dp_full = list(pressure_in_time_aug_full[max_flow_ind_full, outlet_pts] - pressure_in_time_aug_full[max_flow_ind_full, inlet_pts])

            radius = list(np.sqrt(area[outlet_pts]/np.pi))
            inlet_radius = np.sqrt(area[inlet_pts]/np.pi)

            max_out_ind_half = np.argmax(flow_in_time_aug_half[max_flow_ind_half, outlet_pts])
            min_out_ind_half = np.argmin(flow_in_time_aug_half[max_flow_ind_half, outlet_pts])
            max_out_ind_full = np.argmax(flow_in_time_aug_full[max_flow_ind_full, outlet_pts])
            min_out_ind_full = np.argmin(flow_in_time_aug_full[max_flow_ind_full, outlet_pts])
            #min_out_ind = np.argmin(flow_in_time_aug[max_flow_ind, outlet_pts])
            if len(junction_dict[junction_id]) == 4:
                print(f"Junction {junction_id} has 3 outlets.  Converting into 2 bifurcations")
                char_val_dict[anatomy]["inlet_angles"] += [inlet_angles,]
                char_val_dict[anatomy]["outlet_angles"] += [outlet_angles,]
                char_val_dict[anatomy]["daughter1_flow"] += [0, float(flow_half[max_out_ind_half]), float(flow_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_flow"] += [0, float(flow_half[min_out_ind_half]), float(flow_full[min_out_ind_full]),]
                char_val_dict[anatomy]["inlet_flow"] += [0, float(inlet_flow_half), float(inlet_flow_full),]
                char_val_dict[anatomy]["daughter1_P"] += [0, float(p_half[max_out_ind_half]), float(p_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_P"] += [0, float(p_half[min_out_ind_half]), float(p_full[min_out_ind_full]),]
                char_val_dict[anatomy]["inlet_P"] += [0, float(inlet_p_half), float(inlet_p_full),]
                char_val_dict[anatomy]["daughter1_dP"] += [0, float(dp_half[max_out_ind_half]), float(dp_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_dP"] += [0, float(dp_half[min_out_ind_half]), float(dp_full[min_out_ind_full]),]

                char_val_dict[anatomy]["inlet_radius"] += [float(inlet_radius),]
                char_val_dict[anatomy]["inlet_area"] += [area[inlet_pts],]
                char_val_dict[anatomy]["outlet_area"] += [area[outlet_pts],]
                char_val_dict[anatomy]["daughter1_radius"] += [float(radius[max_out_ind_half]),]
                char_val_dict[anatomy]["daughter2_radius"] += [float(radius[min_out_ind_half]),]
                char_val_dict[anatomy]["daughter1_angle"] += [float(angle_diffs[max_out_ind_half]),]
                char_val_dict[anatomy]["daughter2_angle"] += [float(angle_diffs[min_out_ind_half]),]


                Q1 = np.asarray(char_val_dict[anatomy]["daughter1_flow"]).reshape(-1,1)
                Q2 = np.asarray(char_val_dict[anatomy]["daughter2_flow"]).reshape(-1,1)

                dP1 = np.asarray(char_val_dict[anatomy]["daughter1_dP"]).reshape(-1,1)
                dP2 = np.asarray(char_val_dict[anatomy]["daughter2_dP"]).reshape(-1,1)

                A_mat1 = np.hstack([Q1, np.square(Q1)])
                A_mat2 = np.hstack([Q2, np.square(Q2)])

                coefs1, residuals1, t, q = np.linalg.lstsq(A_mat1, dP1, rcond=None)
                coefs2, residuals2, t, q = np.linalg.lstsq(A_mat2, dP2, rcond=None)

                err1 = np.linalg.norm(residuals1)/(1333**2)
                err2 = np.linalg.norm(residuals2)/(1333**2)

                char_val_dict[anatomy]["r_lin_1"] += [coefs1[0][0],]
                char_val_dict[anatomy]["r_lin_2"] += [coefs2[0][0],]
                char_val_dict[anatomy]["r_quad_1"] += [coefs1[1][0],]
                char_val_dict[anatomy]["r_quad_2"] += [coefs2[1][0],]


                U_c = 4500 * 0.04 / (1.06 * 2*np.sqrt(area[inlet_pts]/np.pi))
                char_val_dict[anatomy]["r_lin_star1"] += [char_val_dict[anatomy]["r_lin_1"][-1]*area[inlet_pts]/(1.06*U_c),]
                char_val_dict[anatomy]["r_lin_star2"] += [char_val_dict[anatomy]["r_lin_2"][-1]*area[inlet_pts]/(1.06*U_c),]
                areas = [area[inlet_pts[0]], area[outlet_pts[max_out_ind_half]], area[outlet_pts[min_out_ind_half]]]
                tangents = [[angle1[inlet_pts[0]], angle2[inlet_pts[0]], angle3[inlet_pts[0]]],
                            [angle1[outlet_pts[max_out_ind_half]], angle2[outlet_pts[max_out_ind_half]], angle3[outlet_pts[max_out_ind_half]]],
                            [angle1[outlet_pts[min_out_ind_half]], angle2[outlet_pts[min_out_ind_half]], angle3[outlet_pts[min_out_ind_half]]]
                            ]
                r_lin, r_quad, L, r_lin_star = get_RRI_values(areas = areas, tangents = tangents)
                char_val_dict[anatomy]["r_lin_1_pred"] += [r_lin[0],]
                char_val_dict[anatomy]["r_lin_star1_pred"] += [r_lin_star[0],]
                char_val_dict[anatomy]["r_lin_2_pred"] += [r_lin[1],]
                char_val_dict[anatomy]["r_lin_star2_pred"] += [r_lin_star[1],]
            else:
                char_val_dict[anatomy]["inlet_angles"] += [inlet_angles,]
                char_val_dict[anatomy]["outlet_angles"] += [outlet_angles,]
                char_val_dict[anatomy]["daughter1_flow"] += [0, float(flow_half[max_out_ind_half]), float(flow_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_flow"] += [0, float(flow_half[min_out_ind_half]), float(flow_full[min_out_ind_full]),]
                char_val_dict[anatomy]["inlet_flow"] += [0, float(inlet_flow_half), float(inlet_flow_full),]
                char_val_dict[anatomy]["daughter1_P"] += [0, float(p_half[max_out_ind_half]), float(p_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_P"] += [0, float(p_half[min_out_ind_half]), float(p_full[min_out_ind_full]),]
                char_val_dict[anatomy]["inlet_P"] += [0, float(inlet_p_half), float(inlet_p_full),]
                char_val_dict[anatomy]["daughter1_dP"] += [0, float(dp_half[max_out_ind_half]), float(dp_full[max_out_ind_full]),]
                char_val_dict[anatomy]["daughter2_dP"] += [0, float(dp_half[min_out_ind_half]), float(dp_full[min_out_ind_full]),]

                char_val_dict[anatomy]["inlet_radius"] += [float(inlet_radius),]
                char_val_dict[anatomy]["inlet_area"] += [area[inlet_pts],]
                char_val_dict[anatomy]["outlet_area"] += [area[outlet_pts],]
                char_val_dict[anatomy]["daughter1_radius"] += [float(radius[max_out_ind_half]),]
                char_val_dict[anatomy]["daughter2_radius"] += [float(radius[min_out_ind_half]),]
                char_val_dict[anatomy]["daughter1_angle"] += [float(angle_diffs[max_out_ind_half]),]
                char_val_dict[anatomy]["daughter2_angle"] += [float(angle_diffs[min_out_ind_half]),]


                Q1 = np.asarray(char_val_dict[anatomy]["daughter1_flow"]).reshape(-1,1)
                Q2 = np.asarray(char_val_dict[anatomy]["daughter2_flow"]).reshape(-1,1)

                dP1 = np.asarray(char_val_dict[anatomy]["daughter1_dP"]).reshape(-1,1)
                dP2 = np.asarray(char_val_dict[anatomy]["daughter2_dP"]).reshape(-1,1)

                A_mat1 = np.hstack([Q1, np.square(Q1)])
                A_mat2 = np.hstack([Q2, np.square(Q2)])

                coefs1, residuals1, t, q = np.linalg.lstsq(A_mat1, dP1, rcond=None)
                coefs2, residuals2, t, q = np.linalg.lstsq(A_mat2, dP2, rcond=None)

                err1 = np.linalg.norm(residuals1)/(1333**2)
                err2 = np.linalg.norm(residuals2)/(1333**2)

                char_val_dict[anatomy]["r_lin_1"] += [coefs1[0][0],]
                char_val_dict[anatomy]["r_lin_2"] += [coefs2[0][0],]
                char_val_dict[anatomy]["r_quad_1"] += [coefs1[1][0],]
                char_val_dict[anatomy]["r_quad_2"] += [coefs2[1][0],]


                U_c = 4500 * 0.04 / (1.06 * 2*np.sqrt(area[inlet_pts]/np.pi))
                char_val_dict[anatomy]["r_lin_star1"] += [char_val_dict[anatomy]["r_lin_1"][-1]*area[inlet_pts]/(1.06*U_c),]
                char_val_dict[anatomy]["r_lin_star2"] += [char_val_dict[anatomy]["r_lin_2"][-1]*area[inlet_pts]/(1.06*U_c),]
                areas = [area[inlet_pts[0]], area[outlet_pts[max_out_ind_half]], area[outlet_pts[min_out_ind_half]]]
                tangents = [[angle1[inlet_pts[0]], angle2[inlet_pts[0]], angle3[inlet_pts[0]]],
                            [angle1[outlet_pts[max_out_ind_half]], angle2[outlet_pts[max_out_ind_half]], angle3[outlet_pts[max_out_ind_half]]],
                            [angle1[outlet_pts[min_out_ind_half]], angle2[outlet_pts[min_out_ind_half]], angle3[outlet_pts[min_out_ind_half]]]
                            ]
                r_lin, r_quad, L, r_lin_star = get_RRI_values(areas = areas, tangents = tangents)
                char_val_dict[anatomy]["r_lin_1_pred"] += [r_lin[0],]
                char_val_dict[anatomy]["r_lin_star1_pred"] += [r_lin_star[0],]
                char_val_dict[anatomy]["r_lin_2_pred"] += [r_lin[1],]
                char_val_dict[anatomy]["r_lin_star2_pred"] += [r_lin_star[1],]
            # print(f"inlet radius: {char_val_dict[anatomy]['inlet_radius'][-1]}")
            # print(f"U_char: {U_c}")
            # print(f"Angle diff: {angle_diffs[max_out_ind_half]}, {angle_diffs[min_out_ind_half]}")
            # print(f"Daughter 1 R_lin calc: {char_val_dict[anatomy]['r_lin_1'][-1]}.  R_lin pred: {r_lin[0]}")
            # print(f"Daughter 1 R_lin_star calc: {char_val_dict[anatomy]['r_lin_star1'][-1]}.  R_lin_star pred: {char_val_dict[anatomy]['r_lin_star1_pred'][-1]}")
            # print(f"Daughter 2 R_lin calc: {char_val_dict[anatomy]['r_lin_2'][-1]}.  R_lin pred: {r_lin[1]}")
            # print(f"Daughter 2 R_lin_star calc: {char_val_dict[anatomy]['r_lin_star2'][-1]}.  R_lin_star pred: {char_val_dict[anatomy]['r_lin_star2_pred'][-1]}")
    save_dict(char_val_dict, results_dir)
    pdb.set_trace()
    return

def print_aorta_study():

    anatomy = "Aorta"
    char_val_dict = load_dict("data/characteristic_value_dictionaries/vmr_char_val_dict")
    param_stat_dict = {}

    for anatomy in char_val_dict.keys():

        if anatomy == "Pulmonary":
            low_rad_ind = np.asarray(char_val_dict[anatomy]["inlet_radius"]) < 0.7
            for value in char_val_dict[anatomy].keys():
                char_val_dict[anatomy][value] = list(np.asarray(char_val_dict[anatomy][value])[low_rad_ind])

        char_val_dict[anatomy].update({"velocity": np.asarray(char_val_dict[anatomy]["flow"])/ (np.pi*np.square(char_val_dict[anatomy]["radius"]))})
        char_val_dict[anatomy].update({"inlet_velocity": np.asarray(char_val_dict[anatomy]["inlet_flow"])/ (np.pi*np.square(char_val_dict[anatomy]["inlet_radius"]))})
        char_val_dict[anatomy].update({"radius_ratio": np.asarray(char_val_dict[anatomy]["radius"])/ np.asarray(char_val_dict[anatomy]["inlet_radius"])})
        char_val_dict[anatomy].update({"angle_diff": np.asarray(char_val_dict[anatomy]["angle"])[0::2] + \
                                       np.asarray(char_val_dict[anatomy]["angle"])[1::2]})
        value_list = ["flow", "angle","angle_diff", "inlet_radius", "velocity", "radius_ratio", "inlet_velocity"]

        param_stat_dict.update({anatomy:{}})

        for value in value_list:
            dat = np.asarray(char_val_dict[anatomy][value])
            print_stats(char_val_dict, anatomy, value)
            param_stat_dict[anatomy].update({value: [np.min(dat),
                                            np.max(dat),
                                            np.mean(dat),
                                            np.std(dat)]})

    #pprint(param_stat_dict)
    params_stat_dir = "data/param_stat_dict"
    save_dict(param_stat_dict, params_stat_dir)
    return

def print_pulmo_study():

    anatomy = "Pulmonary"
    char_val_dict = load_dict("data/characteristic_value_dictionaries/vmr_char_val_dict")
    param_stat_dict = {}

    char_val_dict[anatomy].update({"velocity": np.asarray(char_val_dict[anatomy]["flow"])/ (np.pi*np.square(char_val_dict[anatomy]["radius"]))})
    char_val_dict[anatomy].update({"inlet_velocity": np.asarray(char_val_dict[anatomy]["inlet_flow"])/ (np.pi*np.square(char_val_dict[anatomy]["inlet_radius"]))})
    char_val_dict[anatomy].update({"radius_ratio": np.asarray(char_val_dict[anatomy]["radius"])/ np.asarray(char_val_dict[anatomy]["inlet_radius"])})
    value_list = ["flow", "angle", "inlet_radius", "velocity", "radius_ratio", "inlet_velocity"]

    plt.hist(char_val_dict[anatomy]["inlet_radius"])
    plt.xlabel("inlet radius (cm)")
    plt.ylabel("frequency")
    plt.savefig("results/vmr/pulmonary_inlet_ratios.png")
    return



if __name__ == '__main__':
    extract_characteristic_values(tree_name="tree_80")

