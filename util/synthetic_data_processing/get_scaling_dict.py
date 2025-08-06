import sys

from numpy import save
import pandas as pd
from util.tools.basic import *
from collections import defaultdict
#plt.rcParams.update(plt.rcParamsDefault)
def get_name_end( unsteady = False, use_steady_ab = True):
    name_end = ""
    if unsteady and use_steady_ab:
        name_end += "steady_ab"
    if not unsteady:
        name_end += "steady"

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

def get_data_lists(anatomy, set_type, unsteady = False):
    CCO_data_dict = load_dict(f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_dict")
    data_list_dict = defaultdict(list)
    for geo in CCO_data_dict.keys():
        for offset in CCO_data_dict[geo].keys():
            for value in CCO_data_dict[geo][offset].keys():
                try:
                    data_list_dict[value].append(CCO_data_dict[geo][offset][value])
                except:
                    pdb.set_trace()

    # values = list(data_list_dict.keys())
    # #pdb.set_trace()
    # daughter1_values = [value for value in values if "daughter1" in value]
    # daughter2_values = [value for value in values if "daughter2" in value]
    # for value in daughter1_values:
    #     print(f"Swapping {value}")
    #     value_name = value[10:]
    #     daughter1_list = copy.deepcopy(data_list_dict["daughter1_" + value_name])
    #     daughter2_list = copy.deepcopy(data_list_dict["daughter2_" + value_name])
    #     data_list_dict["daughter1_" + value_name] += daughter2_list
    #     data_list_dict["daughter2_" + value_name] += daughter1_list

    # for value in values:
    #     if not value in daughter1_values and not value in daughter2_values:
    #         print(f"Doubling {value}")
    #         data_list_dict[value] += data_list_dict[value]
    #     print(value)
    #     try:
    #         data_list_dict[value] = np.asarray(data_list_dict[value])
    #     except:
    #         print(f"Could not convert {value} to numpy array, skipping")
    #         continue
    save_dict(data_list_dict, f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_list_dict")
    return

def get_scaling_dict(anatomy, set_type, doubled = True, unsteady = False):

    data_list_dict = load_dict(f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_list_dict")
    scaling_dict = {}
    

    if not os.path.exists(f"results/synthetic_data_trends"):
        os.mkdir(f"results/synthetic_data_trends")
    if not os.path.exists(f"results/synthetic_data_trends/geo_dist"):
        os.mkdir(f"results/synthetic_data_trends/geo_dist")

    r_quad = data_list_dict["daughter1_R_quad_star"]
    C = np.exp(-4)
    r_quad_scaled = np.sign(r_quad) * np.log(1 + np.abs(r_quad) / C)
    data_list_dict["daughter1_R_quad_star_logC"] = r_quad_scaled
    save_dict(data_list_dict, f"data/data_dicts/{anatomy}_{set_type}_synthetic_data_list_dict")
    to_normalize = list(data_list_dict.keys())

    values_of_interest = ["daughter1_R_lin_star_m2", "daughter1_L_m2", "daughter1_R_lin_star", "daughter2_R_lin_star", "daughter1_R_quad_star", "daughter2_R_quad_star", "A_char"]
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
                      "inlet_flow", "daughter1_flow", "daughter2_flow","daughter1_Re", "daughter2_Re", "inlet_Re","flow", "pressure", "dp1","dp2","times","dflow_dt"]
    
    for value in to_normalize:
        if value in values_to_skip:
            continue
        if len(data_list_dict[value]) == 0:
            continue
        print(f"Normalizing {value}")
        scaling_dict.update({value: [np.mean(data_list_dict[value]), np.std(data_list_dict[value]), np.min(data_list_dict[value]), np.max(data_list_dict[value])]})

        if True:
            plt.clf()
            plt.hist(data_list_dict[value], bins = 30, alpha = 0.5,)
            plt.xlabel(value); plt.ylabel("frequency"); plt.title(f"Synthetic {value} distribution")

            plt.savefig(f"results/synthetic_data_trends/geo_dist/{anatomy}_{set_type}_{value}.png", bbox_inches='tight', transparent=False, format = "png")

            for value_of_interest in values_of_interest:

                if not os.path.exists(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends"):
                    os.mkdir(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends")
                plt.clf()
                plt.scatter(data_list_dict[value], data_list_dict[value_of_interest])
                plt.xlabel(value); plt.ylabel(value_of_interest); plt.title(f"{value} vs {value_of_interest}")
                plt.savefig(f"results/synthetic_data_trends/{anatomy}_{set_type}_{value_of_interest}_trends/{value}_{value_of_interest}.png", bbox_inches='tight', transparent=False, format = "png")
            
                

    if not os.path.exists(f"data/scaling_dictionaries"):
        os.mkdir(f"data/scaling_dictionaries")
    if not doubled:
        save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict_not_doubled")
    else:
        save_dict(scaling_dict, f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    return
