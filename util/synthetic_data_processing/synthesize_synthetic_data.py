import sys
from util.tools.basic import *
plt.rcParams.update(plt.rcParamsDefault)
def get_name_end( unsteady = False, use_steady_ab = True):
    name_end = ""
    if unsteady and use_steady_ab:
        name_end += "steady_ab"
    if not unsteady:
        name_end += "steady"

def get_coefs(anatomy, set_type, rm_low_r2 = True, unsteady = False, use_steady_ab = True):

    num_outlets = 2
    char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    char_val_dict.update({"R_lin_star1": [],
                    "R_lin_star2": [],
                    "R_quad_star1": [],
                    "R_quad_star2": [],
                    "R_lin1": [],
                    "R_lin2": [],
                    "R_quad1": [],
                    "R_quad2": [],})
    to_rm = []
    for geo_ind in range(len(char_val_dict["inlet_area"])):

        Q_star1 = np.asarray(char_val_dict["daughter1_flow_star"][geo_ind]).reshape(-1,1)
        Q_star2 = np.asarray(char_val_dict["daughter2_flow_star"][geo_ind]).reshape(-1,1)

        r2_unsteady = 0; r2_steady = 0; r2_L = 0

        dP1_star = np.asarray(char_val_dict["daughter1_dP_star"][geo_ind]).reshape(-1,1)
        dP2_star = np.asarray(char_val_dict["daughter2_dP_star"][geo_ind]).reshape(-1,1)

        A_mat1_star = np.hstack([Q_star1, np.square(Q_star1)])
        A_mat2_star = np.hstack([Q_star2, np.square(Q_star2)])

        coefs_star1, residuals1, t, q = np.linalg.lstsq(A_mat1_star, dP1_star, rcond=None)
        R_lin_star1 = coefs_star1[0][0]
        R_quad_star1 = coefs_star1[1][0]

        if True:
            A_mat1_star = np.hstack([Q_star1,])
            coefs_star1, residuals1, t, q = np.linalg.lstsq(A_mat1_star, dP1_star, rcond=None)
            R_lin_star1 = coefs_star1[0][0]
            #R_quad_star1 = 0
            
        if R_lin_star1 > 0:
            print(f"Removing {geo_ind} for positive R_lin_star1.")
            to_rm.append(geo_ind)

        #print(f"Condition number 1: {cond1}")
        coefs_star2, residuals2, t, q = np.linalg.lstsq(A_mat2_star, dP2_star, rcond=None)
        R_lin_star2 = coefs_star2[0][0]
        R_quad_star2 = coefs_star2[1][0]

        if True:
            A_mat2_star = np.hstack([Q_star2,])
            coefs_star2, residuals2, t, q = np.linalg.lstsq(A_mat2_star, dP2_star, rcond=None)
            R_lin_star2 = coefs_star2[0][0]
            #R_quad_star2 = 0
            
        if R_lin_star2 > 0:
            print(f"Removing {geo_ind} for positive R_lin_star2.")
            to_rm.append(geo_ind)

        # r2_steady1 = get_r2(A_mat1_star, dP1_star, coefs1.reshape(-1,1))
        # r2_steady2 = get_r2(A_mat2_star, dP2_star, coefs2.reshape(-1,1))

        
        # err1 = np.linalg.norm(residuals1)/(1333**2)
        # err2 = np.linalg.norm(residuals2)/(1333**2)

        char_val_dict["R_lin_star1"].append(R_lin_star1)
        char_val_dict["R_lin_star2"].append(R_lin_star2)
        char_val_dict["R_quad_star1"].append(R_quad_star1)
        char_val_dict["R_quad_star2"].append(R_quad_star2)


        Q1 = np.asarray(char_val_dict["daughter1_flow"][geo_ind]).reshape(-1,1)
        Q2 = np.asarray(char_val_dict["daughter2_flow"][geo_ind]).reshape(-1,1)

        r2_unsteady = 0; r2_steady = 0; r2_L = 0

        dP1 = np.asarray(char_val_dict["daughter1_dP_total"][geo_ind]).reshape(-1,1)
        dP2 = np.asarray(char_val_dict["daughter2_dP_total"][geo_ind]).reshape(-1,1)

        A_mat1 = np.hstack([Q1, np.square(Q1)])
        A_mat2 = np.hstack([Q2, np.square(Q2)])

        coefs1, residuals1, t, q = np.linalg.lstsq(A_mat1, dP1, rcond=None)
        R_lin1 = coefs1[0][0]
        R_quad1 = coefs1[1][0]

        if True > 0:
            A_mat1 = np.hstack([Q1,])
            coefs1, residuals1, t, q = np.linalg.lstsq(A_mat1, dP1, rcond=None)
            R_lin1 = coefs1[0][0]
            #R_quad1 = 0



        coefs2, residuals2, t, q = np.linalg.lstsq(A_mat2, dP2, rcond=None)
        R_lin2 = coefs2[0][0]
        R_quad2 = coefs2[1][0]

        if True > 0:
            A_mat2 = np.hstack([Q2,])
            coefs2, residuals2, t, q = np.linalg.lstsq(A_mat2, dP2, rcond=None)
            R_lin2 = coefs2[0][0]
            #R_quad2 = 0

        r2_steady1 = get_r2(A_mat1, dP1, coefs1.reshape(-1,1))
        r2_steady2 = get_r2(A_mat2, dP2, coefs2.reshape(-1,1))

        # err1 = np.linalg.norm(residuals1)/(1333**2)
        # err2 = np.linalg.norm(residuals2)/(1333**2)
        char_val_dict["R_lin1"].append(R_lin1)
        char_val_dict["R_lin2"].append(R_lin2)
        char_val_dict["R_quad1"].append(R_quad1)
        char_val_dict["R_quad2"].append(R_quad2)

        print(f"R_lin_star1: {char_val_dict['R_lin_star1'][-1]}, R_lin1: {char_val_dict['R_lin1'][-1]}")
        print(f"dP 1: {char_val_dict['daughter1_dP_total'][geo_ind]}")
        print(f"R_lin_star2: {char_val_dict['R_lin_star2'][-1]}, R_lin2: {char_val_dict['R_lin2'][-1]}")
        print(f"dP 2: {char_val_dict['daughter2_dP_total'][geo_ind]}")
        #print(f"R_quad_star1: {char_val_dict['R_quad_star1'][-1]}, R_quad1: {char_val_dict['R_quad1'][-1]}")
        #print(f"R_quad_star2: {char_val_dict['R_quad_star2'][-1]}, R_quad2: {char_val_dict['R_quad2'][-1]}")
        #pdb.set_trace()
        # Check consistency of non-dimensionalization
        assert abs(char_val_dict["R_lin1"][-1] - char_val_dict["R_lin_star1"][-1]*1.06*char_val_dict["U_char"][geo_ind]/char_val_dict["inlet_area"][geo_ind]) < 0.1
        assert abs(char_val_dict["R_lin2"][-1] - char_val_dict["R_lin_star2"][-1]*1.06*char_val_dict["U_char"][geo_ind]/char_val_dict["inlet_area"][geo_ind]) < 0.1

        # assert abs(char_val_dict["R_quad1"][-1] - char_val_dict["R_quad_star1"][-1]*1.06/char_val_dict["inlet_area"][geo_ind]**2) < 0.1
        # assert abs(char_val_dict["R_quad2"][-1] - char_val_dict["R_quad_star2"][-1]*1.06/char_val_dict["inlet_area"][geo_ind]**2) < 0.1
       
        # if err1 > 0.1 or err2 > 0.1: #r2_steady < 0.90 and r2_unsteady < 0.90 and r2_UO < 0.90:
        #     to_rm.append(geo_ind)
        #     print(f"Removing {geo_ind} for high residual {(np.linalg.norm(residuals1)/(1333**2))}, {(np.linalg.norm(residuals2)/(1333**2))} mmHg.")
        # if char_val_dict["R_quad_star2"][-1] < -10:
        #     to_rm.append(geo_ind)
        #     print(f"Removing {geo_ind} for OOD R_quad_star2.")
        # if char_val_dict["R_lin_star2"][-1] < -0.1:
        #     to_rm.append(geo_ind)
        #     print(f"Removing {geo_ind} for OOD R_quad_star2.")
        # if char_val_dict["R_lin_star1"][-1] > 0.08:
        #     to_rm.append(geo_ind)
        #     print(f"Removing {geo_ind} for OOD R_quad_star2.")

    # if rm_low_r2:
    #     print(f"Removing {len(to_rm)} outlets for low r2 values.")
    #     to_keep = []
    #     for i in range(len(char_val_dict["inlet_area"])):
    #         if i in to_rm:
    #             continue
    #         else:
    #             to_keep.append(i)

    #     for key in char_val_dict:
    #         try:
    #             char_val_dict[key] = [char_val_dict[key][ind] for ind in to_keep]
    #         except:
    #             continue
    plt.savefig("results/flow_vs_dp")
    save_dict(char_val_dict, f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    return

def remove_outlier_coefs(anatomy, set_type, sd_tol):
    char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")

    outlier_inds, non_outlier_inds = get_outlier_inds(char_val_dict["coef_a"][::2], m = sd_tol)

    non_outlier_inds = [2*ind for ind in non_outlier_inds] + [2*ind+1 for ind in non_outlier_inds]
    non_outlier_inds.sort()

    for key in char_val_dict.keys():
        try:
            full_array = char_val_dict[key]
            char_val_dict[key] = list(np.asarray(char_val_dict[key])[2*np.asarray(non_outlier_inds).astype(int)])
        except:
            continue

    print(f"a outlier_inds: {outlier_inds}")
    save_dict(char_val_dict, f"data/characteristic_value_dictionaries/{anatomy}_{set_type}_synthetic_data_dict")
    return

def get_geo_scalings(anatomy, set_type, unsteady = False):

    #plt.style.use('dark_background')

    char_val_dict = load_dict(f"data/param_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    scaling_dict = {}
    to_normalize = list(char_val_dict.keys())
    if unsteady:
        to_normalize.append("coef_L")
        to_normalize.append("coef_a_UO")
        to_normalize.append("coef_b_UO")
        to_normalize.append("coef_L_UO")

    if not os.path.exists(f"results/synthetic_data_trends"):
        os.mkdir(f"results/synthetic_data_trends")
    if not os.path.exists(f"results/synthetic_data_trends/geo_dist"):
        os.mkdir(f"results/synthetic_data_trends/geo_dist")


    values_of_interest = ["R_lin_star1", "R_lin_star2", "R_quad_star1", "R_quad_star2"]
    values_to_skip = ["daughter1_dP_star", "daughter2_dP_star", 
                      "daughter1_dP_original", "daughter2_dP_original",
                      "daughter1_flow_star", "daughter2_flow_star", 
                      "daughter1_dP", "daughter2_dP", 
                      "daughter1_flow", "daughter2_flow",
                      "daughter1_dP_total", "daughter2_dP_total","daughter1_dP_dyn", "daughter2_dP_dyn",
                      "inlet_dP_dyn", "daughter1_dP_dyn", "daughter2_dP_dyn", 
                        "daughter1_P_dyn", "daughter2_P_dyn", "inlet_P_dyn",
                      "daughter1_velocity", "daughter2_velocity", "inlet_velocity", 
                      "daughter1_energy", "daughter2_energy", "inlet_energy", 
                      "inlet_flow", "daughter1_flow", "daughter2_flow",]
    for value in to_normalize:
        if value in values_to_skip:
            continue
        print(f"Normalizing {value}")
        scaling_dict.update({value: [np.mean(char_val_dict[value]), np.std(char_val_dict[value])]})

        plt.clf()
        plt.hist(char_val_dict[value], bins = 30, alpha = 0.5,)
        plt.xlabel(value); plt.ylabel("frequency"); plt.title(f"Synthetic {value} distribution")

        plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_{set_type}_{value}.png", bbox_inches='tight', transparent=False, format = "png")

        for value_of_interest in values_of_interest:

            if not os.path.exists(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends"):
                os.mkdir(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends")
            plt.clf()
            plt.scatter(char_val_dict[value], char_val_dict[value_of_interest])
            plt.xlabel(value); plt.ylabel(value_of_interest); plt.title(f"{value} vs {value_of_interest}")
            plt.savefig(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends/{value}_{value_of_interest}.png", bbox_inches='tight', transparent=False, format = "png")
        
            

    if not os.path.exists(f"data/scaling_dictionaries"):
        os.mkdir(f"data/scaling_dictionaries")
    save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    return
