import sys
sys.path.append("/Users/natalia/Desktop/junction_pressure_differentials")
from util.unified0D_plus.apply_unified0D_plus  import *
from util.unified0D_plus.graph_to_junction_dict import *

from util.regression.neural_network.training_util import *
from util.tools.graph_handling import *
from util.tools.basic import *
plt.rcParams.update(plt.rcParamsDefault)
import tensorflow as tf
import dgl
from dgl.data import DGLDataset

def get_re(inlet_flow, inlet_radius):
    viscosity = 0.04
    density = 1.06
    inlet_velocity = 2*inlet_flow/(np.pi *inlet_radius**2)
    re = (density * inlet_velocity * 2 * inlet_radius)/viscosity
    return re

def dP_poiseuille(flow, radius, length):
    mu = 0.04
    dP = 8 * mu * length * flow /(np.pi * radius**4)
    return dP

def plot_mesh_convergence(anatomy):
    set_type = "mesh_convergence"
    color = "royalblue"
    dP_type = "end"
    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_{set_type}_synthetic_data_dict")

    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")
    dPs = []
    marker_list = ["o", "v", "s", "d", "X", "*"]

    if anatomy[0:5] == "Aorta":
        ref_list = ["1.9E5 cells", "2.7E5 cells", "4.4E5 cells", "9.0E5 cells", "1.5E6 cells" , "2.3E6 cells"] # Aorta
    elif anatomy[0:5] == "Pulmo":
        ref_list = ["6.4E4 cells", "9.3E4 cells", "1.6E5 cells", "2.5E5 cells", "5.1E5 cells" , "9.1E5 cells"]
    elif anatomy[0:5] == "mynar":
        ref_list = ["4.5E5 cells", "8.0E5 cells", "1.3E6 cells", "1.8E6 cells"] # Mynard
    elif anatomy == "AP":
        ref_list = ["1.6E4", "3.1E4", "8.8E4", "1.8E5", "5.3E5", "1.2E6"]
    re_max = 0

    flow_arr = []
    dP_arr = []
    area_arr = []


    for i in range(int(len(char_val_dict["name"])/2)):
        print(char_val_dict["name"][2*i])
        outlet_ind = 0
        if char_val_dict["outlet_area"][2*i] < char_val_dict["outlet_area"][2*i +1 ]:
            outlet_ind = 0

        print(outlet_ind)

        inlet_data = np.stack((scale(scaling_dict, char_val_dict["inlet_area"][2*i], "inlet_area").reshape(1,-1),
                                )).T
        print(char_val_dict["outlet_area"][2*i:2*i+2])
        outlet_data = np.stack((
            scale(scaling_dict, np.asarray(char_val_dict["outlet_area"][2*i: 2*(i+1)]), "outlet_area"),
            scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i: 2*(i+1)]), "angle"),
            )).T

        outlet_flows = np.stack((np.asarray(char_val_dict["flow_list"][2*i]).T,
                                np.asarray(char_val_dict["flow_list"][2*i + 1]).T))

        outlet_dPs = np.stack((np.asarray(char_val_dict["dP_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_list"][2*i + 1]).T))

        outlet_junction_dPs = np.stack((np.asarray(char_val_dict["dP_junc_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_junc_list"][2*i + 1]).T))

        outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b")]).T

        geo_name = "".join([let for let in char_val_dict["name"][2*i] if let.isnumeric()])
        geo_name = int(geo_name)

        inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
        outlet_pairs = get_outlet_pairs(2)
        graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
        graph  = graph.to("/cpu:0")

        with tf.device("/cpu:0"):

            graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float32), [1,1])
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
            graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float32)
            if dP_type == "end":
                graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float32)
            elif dP_type == "junction":
                graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_junction_dPs, dtype=tf.float32)
            graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
            graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float32)
        #print(graph.nodes["outlet"].data)

        master_tensor = get_master_tensors_steady([graph])
        input_tensor = master_tensor[0]
        flow_tensor = master_tensor[2]
        dP = master_tensor[4]
        print(flow_tensor)

        
        plt.scatter(np.asarray(flow_tensor)[outlet_ind,:], np.asarray(dP)[outlet_ind,:]/1333, facecolors='none', edgecolors = color, marker = marker_list[i], s = 50, label = f"{ref_list[i]} mesh elements")
        re_max = max(re_max, get_re(inlet_flow = np.max(np.asarray(flow_tensor)[outlet_ind,:]), inlet_radius = char_val_dict["outlet_radius"][2*i+outlet_ind]))

    flow_tensor_cont = tf.linspace(flow_tensor[outlet_ind,0], flow_tensor[outlet_ind,-1], 100)
    inflow_tensor_cont =  tf.linspace(flow_tensor[0,0], flow_tensor[0,-1], 100) \
                        + tf.linspace(flow_tensor[1,0], flow_tensor[1,-1], 100)


    plt.xlabel("$Q \;  (\mathrm{cm^3/s})$" + f"    (Outlet Re {int(re_max)})")
    plt.ylabel("$\Delta P$ (mmHg)")
    plt.legend(fontsize="12", bbox_to_anchor=(1.05, 1.0), loc='upper left')
    if os.path.exists(f"results") == False:
        os.mkdir(f"results")
    if os.path.exists(f"results/mesh_convergence") == False:
        os.mkdir(f"results/mesh_convergence")
    plt.savefig(f"results/mesh_convergence/mesh_refinement_study_{anatomy}.pdf", bbox_inches='tight', format = "pdf")
    return

anatomy = sys.argv[1]
plot_mesh_convergence(anatomy)


    # junction_dict_global = graphs_to_junction_dict_steady([graph], scaling_dict)
    # flow_arr = flow_tensor.numpy()
    # dP_mynard_list = []
    # if dP_type == "end":
    #     for j in range(1,100):
    #             dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[j])[outlet_ind] \
    #                             - dP_poiseuille(flow = inflow_tensor_cont[j], radius = char_val_dict["inlet_radius"][2*i], length = char_val_dict["inlet_length"][2*i]) \
    #                             - dP_poiseuille(flow = flow_tensor_cont[j], radius = char_val_dict["outlet_radius"][2*i+outlet_ind], length = char_val_dict["outlet_length"][2*i+outlet_ind])]
    # elif dP_type == "junction":
    #     for j in range(1,100):
    #             dP_mynard_list = dP_mynard_list + [apply_unified0D_plus(junction_dict_global[j])[outlet_ind]]
    # dP_mynard = np.asarray(dP_mynard_list)
    #plt.plot(np.asarray(flow_tensor_cont)[1:], dP_mynard/1333, "--", linewidth=2, color = color, label = "Unified0D+")

        # if i/int(len(char_val_dict["name"])/2) < 0.5:
        #     outlet_ind = 1
        # else:
        #     outlet_ind = 1
        #print(outlet_ind)

        # if i/int(len(char_val_dict["name"])/2) < 0.5:
        #     outlet_ind = 1
        # else:
        #     outlet_ind = 0

    #ref_list = ["6.1E5 elements","3.3E5 elements","1.7E5 elements", "9.9E5 elements"]
    #ref_list = ["1.3E5 cells", "1.6E5 cells", "3.5E5 cells", "6.9E5 cells", "1.5E6 cells", "1.9E6 cells"]# ["1", "2", "3", "4", "5"]
    #ref_list = ["4.5E5 cells", "8.0E5 cells", "1.3E6 cells", "1.8E6 cells"] # Mynard
    #ref_list = ["1.9E5 cells", "2.7E5 cells", "4.4E5 cells", "9.0E5 cells", "1.5E6 cells" , "2.3E6 cells"] # Aorta
    #ref_list = ["6.4E4 cells", "9.3E4 cells", "1.6E5 cells", "2.5E5 cells", "5.1E5 cells" , "9.1E5 cells"]