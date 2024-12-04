import sys
sys.path.append("/home/nrubio/Desktop/junction_pressure_differentials")
from util.tools.basic import *
from util.tools.junction_proc import *
import tensorflow as tf
import dgl

def assemble_graphs(anatomy, set_type, unsteady = False):

    graph_list = []

    char_val_dict = load_dict(f"data/characteristic_value_dictionaries/{anatomy}_{set_type}_synthetic_data_dict")
    # if unsteady:
    #     scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_scaling_dict")
    # else:
    scaling_dict = load_dict(f"data/scaling_dictionaries/{anatomy}_{set_type}_scaling_dict")

    if not os.path.exists(f"data/graph_lists/{anatomy}"):
        os.mkdir(f"data/graph_lists/{anatomy}")
    if not os.path.exists(f"data/graph_lists/{anatomy}/{set_type}"):
        os.mkdir(f"data/graph_lists/{anatomy}/{set_type}")

    for i in range(int(len(char_val_dict["inlet_radius"])/2)):

        inlet_data = np.stack((scale(scaling_dict, np.asarray(char_val_dict["inlet_area"][2*i]), "inlet_area").reshape(1,-1),
                                scale(scaling_dict, np.asarray(char_val_dict["inlet_length"][2*i]), "inlet_length").reshape(1,-1)
                                )).T

        outlet_data = np.stack((scale(scaling_dict, np.asarray(char_val_dict["outlet_area"][2*i: 2*(i+1)]), "outlet_area"),
                                scale(scaling_dict, np.asarray(char_val_dict["outlet_length"][2*i: 2*(i+1)]), "outlet_length"),
                                scale(scaling_dict, np.asarray(char_val_dict["angle"][2*i: 2*(i+1)]), "angle")
                                )).T

        outlet_flows = np.stack((np.asarray(char_val_dict["flow_list"][2*i]).T,
                                np.asarray(char_val_dict["flow_list"][2*i + 1]).T))

        outlet_dPs = np.stack((np.asarray(char_val_dict["dP_list"][2*i]).T,
                                np.asarray(char_val_dict["dP_list"][2*i + 1]).T))

        outlet_lengths = np.stack((np.asarray(char_val_dict["outlet_length"][2*i]).T,
                                np.asarray(char_val_dict["outlet_length"][2*i + 1]).T))

        if unsteady:

            unsteady_outlet_flows = np.stack((np.asarray(char_val_dict["unsteady_flow_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_flow_list"][2*i + 1]).T))

            unsteady_outlet_flow_ders = np.stack((np.asarray(char_val_dict["unsteady_flow_der_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_flow_der_list"][2*i + 1]).T))

            unsteady_outlet_dPs = np.stack((np.asarray(char_val_dict["unsteady_dP_list"][2*i]).T,
                                np.asarray(char_val_dict["unsteady_dP_list"][2*i + 1]).T))

            outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                    scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b"),
                                    scale(scaling_dict, char_val_dict["coef_L"][2*i: 2*(i+1)], "coef_L")]).T

            outlet_coefs_UO = np.asarray([scale(scaling_dict, char_val_dict["coef_a_UO"][2*i: 2*(i+1)], "coef_a_UO"),
                                    scale(scaling_dict, char_val_dict["coef_b_UO"][2*i: 2*(i+1)], "coef_b_UO"),
                                    scale(scaling_dict, char_val_dict["coef_L_UO"][2*i: 2*(i+1)], "coef_L_UO")]).T


        else:
            outlet_coefs = np.asarray([scale(scaling_dict, char_val_dict["coef_a"][2*i: 2*(i+1)], "coef_a"),
                                    scale(scaling_dict, char_val_dict["coef_b"][2*i: 2*(i+1)], "coef_b")]).T

        geo_name = "".join([let for let in char_val_dict["name"][2*i] if let.isnumeric()])
        geo_name = int(geo_name)

        inlet_outlet_pairs = get_inlet_outlet_pairs(1, 2)
        outlet_pairs = get_outlet_pairs(2)
        graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
        graph  = graph.to("/cpu:0")


        with tf.device("/cpu:0"):
            graph.nodes["inlet"].data["inlet_features"] = tf.reshape(tf.convert_to_tensor(inlet_data, dtype=tf.float64), [1,-1])
            #graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float64)
            graph.nodes["inlet"].data["inlet_length"] = tf.reshape(tf.convert_to_tensor(char_val_dict["inlet_length"][2*i], dtype=tf.float64), [1,1])
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_flows"] = tf.convert_to_tensor(outlet_flows, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_length"] = tf.convert_to_tensor(outlet_lengths, dtype=tf.float64)
            graph.nodes["outlet"].data["outlet_dP"] = tf.convert_to_tensor(outlet_dPs, dtype=tf.float64)
            graph.nodes["inlet"].data["geo_name"] = tf.constant([geo_name])
            graph.nodes["outlet"].data["outlet_coefs"] = tf.convert_to_tensor(outlet_coefs, dtype=tf.float64)


            if unsteady:
                graph.nodes["outlet"].data["outlet_coefs_UO"] = tf.convert_to_tensor(outlet_coefs_UO, dtype=tf.float64)
                graph.nodes["outlet"].data["unsteady_outlet_flows"] = tf.convert_to_tensor(unsteady_outlet_flows, dtype=tf.float32)
                graph.nodes["outlet"].data["unsteady_outlet_flow_ders"] = tf.convert_to_tensor(unsteady_outlet_flow_ders, dtype=tf.float32)
                graph.nodes["outlet"].data["unsteady_outlet_dP"] = tf.convert_to_tensor(unsteady_outlet_dPs, dtype=tf.float32)

        graph_list.append(graph)
        #print(graph.nodes["inlet"])
        #pdb.set_trace()
    if unsteady:
        dgl.save_graphs(f"data/graph_lists/{anatomy}/{set_type}/graph_list", graph_list)

    else:
        dgl.save_graphs(f"data/graph_lists/{anatomy}/{set_type}/graph_list_steady", graph_list)
    return graph


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
