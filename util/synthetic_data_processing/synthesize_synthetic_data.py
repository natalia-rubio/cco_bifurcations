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
    char_val_dict.update({"R_lin_star_inlet": [],
                    "R_lin_star1": [],
                    "R_lin_star2": [],
                    "R_quad_star_inlet": [],
                    "R_quad_star1": [],
                    "R_quad_star2": [],
                    "R_lin_inlet": [],
                    "R_lin1": [],
                    "R_lin2": [],
                    "R_quad_inlet": [],
                    "R_quad1": [],
                    "R_quad2": [],})
    to_rm = []
    for geo_ind in range(len(char_val_dict["inlet_area"])):

        Q_star1 = np.asarray(char_val_dict["daughter1_flow_star"][geo_ind]).reshape(-1,)
        Q_star2 = np.asarray(char_val_dict["daughter2_flow_star"][geo_ind]).reshape(-1,)
        Q_star_inlet = Q_star1 + Q_star2

        r2_unsteady = 0; r2_steady = 0; r2_L = 0

        dP1_star = np.asarray(char_val_dict["daughter1_dP_star"][geo_ind]).reshape(-1,)
        dP2_star = np.asarray(char_val_dict["daughter2_dP_star"][geo_ind]).reshape(-1,)
        dP_vec_star = np.hstack([dP1_star, dP2_star])

        A_mat_star = np.zeros((8, 6))
        # Inlet flows
        A_mat_star[0:4,0] = Q_star_inlet
        A_mat_star[0:4,1] = np.square(Q_star_inlet)
        A_mat_star[4:8,0] = Q_star_inlet
        A_mat_star[4:8,1] = np.square(Q_star_inlet)
        
        # Daughter 1 flows
        A_mat_star[0:4,2] = Q_star1
        A_mat_star[0:4,3] = np.square(Q_star1)
        
        # Daughter 2 flows
        A_mat_star[4:8,4] = Q_star2
        A_mat_star[4:8,5] = np.square(Q_star2)

        # Solve
        coefs_star, residuals, t, q = np.linalg.lstsq(A_mat_star, dP_vec_star, rcond=None)
        #pdb.set_trace()
        R_lin_star_inlet    = coefs_star[0]
        R_quad_star_inlet   = coefs_star[1]
        R_lin_star1         = coefs_star[2]
        R_quad_star1        = coefs_star[3]
        R_lin_star2         = coefs_star[4]
        R_quad_star2        = coefs_star[5]

        char_val_dict["R_lin_star_inlet"].append(copy.copy(R_lin_star_inlet))
        char_val_dict["R_quad_star_inlet"].append(copy.copy(R_quad_star_inlet))
        char_val_dict["R_lin_star1"].append(copy.copy(R_lin_star1))
        char_val_dict["R_lin_star2"].append(copy.copy(R_lin_star2))
        char_val_dict["R_quad_star1"].append(copy.copy(R_quad_star1))
        char_val_dict["R_quad_star2"].append(copy.copy(R_quad_star2))

        # Check consistency of non-dimensionalization
        # Dimensional solution
        Q1 = np.asarray(char_val_dict["daughter1_flow"][geo_ind]).reshape(-1,)
        Q2 = np.asarray(char_val_dict["daughter2_flow"][geo_ind]).reshape(-1,)
        Q_inlet = Q1 + Q2

        dP1 = np.asarray(char_val_dict["daughter1_dP"][geo_ind]).reshape(-1,)
        dP2 = np.asarray(char_val_dict["daughter2_dP"][geo_ind]).reshape(-1,)
        dP_vec = np.hstack([dP1, dP2])

        A_mat = np.zeros((8, 6))
        # Inlet flows
        A_mat[0:4,0] = Q_inlet
        A_mat[0:4,1] = np.square(Q_inlet)
        A_mat[4:8,0] = Q_inlet
        A_mat[4:8,1] = np.square(Q_inlet)


        # Daughter 1 flows
        A_mat[0:4,2] = Q1
        A_mat[0:4,3] = np.square(Q1)

        # Daughter 2 flows
        A_mat[4:8,4] = Q2
        A_mat[4:8,5] = np.square(Q2)

        # Solve
        coefs, residuals, t, q = np.linalg.lstsq(A_mat, dP_vec, rcond=None)
        print(f"Residuals: {np.linalg.norm(np.sqrt(residuals)/1333)}")
        
        R_lin_inlet    = coefs[0]
        R_quad_inlet   = coefs[1]
        R_lin1         = coefs[2]
        R_quad1        = coefs[3]
        R_lin2         = coefs[4]
        R_quad2        = coefs[5]

        char_val_dict["R_lin_inlet"].append(copy.copy(R_lin_inlet))
        char_val_dict["R_quad_inlet"].append(copy.copy(R_quad_inlet))
        char_val_dict["R_lin1"].append(copy.copy(R_lin1))
        char_val_dict["R_lin2"].append(copy.copy(R_lin2))
        char_val_dict["R_quad1"].append(copy.copy(R_quad1))
        char_val_dict["R_quad2"].append(copy.copy(R_quad2))

        # print(f"R_lin_inlet_star: {char_val_dict['R_lin_inlet_star'][-1]}, R_lin_inlet: {char_val_dict['R_lin_inlet'][-1]}")
        # print(f"R_lin_star1: {char_val_dict['R_lin_star1'][-1]}, R_lin1: {char_val_dict['R_lin1'][-1]}")
        # print(f"dP 1: {char_val_dict['daughter1_dP_total'][geo_ind]}")
        # print(f"R_lin_star2: {char_val_dict['R_lin_star2'][-1]}, R_lin2: {char_val_dict['R_lin2'][-1]}")
        # print(f"dP 2: {char_val_dict['daughter2_dP_total'][geo_ind]}")

        # Check consistency of non-dimensionalization
        #pdb.set_trace()
        assert abs(char_val_dict["R_lin_inlet"][-1] - char_val_dict["R_lin_star_inlet"][-1]*1.06*char_val_dict["U_char"][geo_ind]/char_val_dict["inlet_area"][geo_ind]) < 0.1
        assert abs(char_val_dict["R_lin1"][-1] - char_val_dict["R_lin_star1"][-1]*1.06*char_val_dict["U_char"][geo_ind]/char_val_dict["inlet_area"][geo_ind]) < 0.1
        assert abs(char_val_dict["R_lin2"][-1] - char_val_dict["R_lin_star2"][-1]*1.06*char_val_dict["U_char"][geo_ind]/char_val_dict["inlet_area"][geo_ind]) < 0.1

        assert abs(char_val_dict["R_quad_inlet"][-1] - char_val_dict["R_quad_star_inlet"][-1]*1.06/char_val_dict["inlet_area"][geo_ind]**2) < 0.1
        assert abs(char_val_dict["R_quad1"][-1] - char_val_dict["R_quad_star1"][-1]*1.06/char_val_dict["inlet_area"][geo_ind]**2) < 0.1
        assert abs(char_val_dict["R_quad2"][-1] - char_val_dict["R_quad_star2"][-1]*1.06/char_val_dict["inlet_area"][geo_ind]**2) < 0.1

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


    values_of_interest = ["R_lin_star1", "R_lin_star2", "R_quad_star1", "R_quad_star2", "R_lin_star_inlet", "R_quad_star_inlet",]
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
        if len(char_val_dict[value]) == 0:
            continue
        print(f"Normalizing {value}")
        scaling_dict.update({value: [np.mean(char_val_dict[value]), np.std(char_val_dict[value]), np.min(char_val_dict[value]), np.max(char_val_dict[value])]})

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
