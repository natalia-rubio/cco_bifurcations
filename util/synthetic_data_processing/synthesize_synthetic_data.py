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
                    "R_quad_star2": [],})
    to_rm = []
    for geo_ind in range(len(char_val_dict["inlet_area"])):

        Q_star1 = np.asarray(char_val_dict["daughter1_flow_star"][geo_ind]).reshape(-1,1)
        Q_star2 = np.asarray(char_val_dict["daughter2_flow_star"][geo_ind]).reshape(-1,1)

        r2_unsteady = 0; r2_steady = 0; r2_L = 0

        dP1 = np.asarray(char_val_dict["daughter1_dP_star"][geo_ind]).reshape(-1,1)
        dP2 = np.asarray(char_val_dict["daughter2_dP_star"][geo_ind]).reshape(-1,1)

        A_mat1 = np.hstack([np.square(Q_star1), Q_star1])
        A_mat2 = np.hstack([np.square(Q_star2), Q_star2])

        coefs1, residuals1, t, q = np.linalg.lstsq(A_mat1, dP1, rcond=None)
        coefs2, residuals2, t, q = np.linalg.lstsq(A_mat2, dP2, rcond=None)

        r2_steady1 = get_r2(A_mat1, dP1, coefs1.reshape(-1,1))
        r2_steady2 = get_r2(A_mat2, dP2, coefs2.reshape(-1,1))

        err1 = np.linalg.norm(residuals1)/(1333**2)
        err2 = np.linalg.norm(residuals2)/(1333**2)

        # print(f"Geo: {geo_ind} Outlet 1 R2: {r2_steady1} residual: {(np.linalg.norm(residuals1)/(1333**2))}")
        # print(f"Geo: {geo_ind} Outlet 2 R2: {r2_steady2} residual: {(np.linalg.norm(residuals2)/(1333**2))}")

        char_val_dict["R_lin_star1"].append(coefs1[0][0])
        char_val_dict["R_lin_star2"].append(coefs2[0][0])
        char_val_dict["R_quad_star1"].append(coefs1[1][0])
        char_val_dict["R_quad_star2"].append(coefs2[1][0])
        # if unsteady:
        #     Q_unsteady = char_val_dict["unsteady_flow_list"][outlet_ind].reshape(-1,1)
        #     dQdt = char_val_dict["unsteady_flow_der_list"][outlet_ind].reshape(-1,1)

        #     dP_unsteady_total = char_val_dict["unsteady_dP_list"][outlet_ind].reshape(-1,1)
        #     dP_unsteady_component = char_val_dict["unsteady_dP_list"][outlet_ind].reshape(-1,1) - (a * np.square(Q_unsteady) + b * Q_unsteady)


        #     coefs, residuals, t, q = np.linalg.lstsq(dQdt, dP_unsteady_component, rcond=None);
        #     r2_unsteady = get_r2(dQdt, dP_unsteady_component, coefs.reshape(-1,1))
        #     L = coefs[0][0]
        #     char_val_dict["coef_L"].append(L)

        #     X_unsteady = np.hstack([np.square(Q_unsteady), Q_unsteady, dQdt])

        #     coefs, residuals, t, q = np.linalg.lstsq(X_unsteady, dP_unsteady_total, rcond=None);
        #     r2_UO = get_r2(X_unsteady, dP_unsteady_total, coefs.reshape(-1,1))
        #     a_UO = coefs[0][0]; b_UO = coefs[1][0]; L_UO =  coefs[2][0]
        #     char_val_dict["coef_a_UO"].append(a_UO); char_val_dict["coef_b_UO"].append(b_UO); char_val_dict["coef_L_UO"].append(L_UO)

        if err1 > 0.1 or err2 > 0.1: #r2_steady < 0.90 and r2_unsteady < 0.90 and r2_UO < 0.90:
            to_rm.append(geo_ind)
            print(f"Removing {geo_ind} for high residual {(np.linalg.norm(residuals1)/(1333**2))}, {(np.linalg.norm(residuals2)/(1333**2))} mmHg.")

    if rm_low_r2:
        print(f"Removing {len(to_rm)} outlets for low r2 values.")
        to_keep = []
        for i in range(len(char_val_dict["inlet_area"])):
            if i in to_rm:
                continue
            else:
                to_keep.append(i)

        for key in char_val_dict:
            try:
                char_val_dict[key] = [char_val_dict[key][ind] for ind in to_keep]
            except:
                continue
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
                      "daughter1_flow_star", "daughter2_flow_star", 
                      "daughter1_dPs", "daughter2_dPs", 
                      "daughter1_flows", "daughter2_flows"]
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
