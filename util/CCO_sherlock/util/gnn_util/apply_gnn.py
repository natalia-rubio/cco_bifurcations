from util.gnn_util.reconstruct_gnn import GraphNetReconstructed
import numpy as np
import pickle
import pdb
import tensorflow as tf
import dgl
from dgl.data import DGLDataset

def apply_gnn_normalized(model_name, graph_list):

    f = open("../../data/scaling_dict_synthetic", "rb"); scaling_dict = pickle.load(f)
    graphs_dataset = DGL_Dataset(graph_list)
    graph_data_loader = dgl.batch([graph.to("/gpu:0") for graph in graphs_dataset])

    gnn_model = GraphNet(model_name = model_name)
    junc_ids = graph_data_loader.nodes['outlet'].data['outlet_identifiers'].numpy()

    pred_outlet_pressures_rel = inv_scale(scaling_dict,  gnn_model.forward(graph_data_loader).numpy(),"pressure_out_rel")
    true_outlet_pressures_rel = inv_scale(scaling_dict, graph_data_loader.nodes["outlet"].data["outlet_pressure"].numpy(), "pressure_out_rel")
    pred_outlet_pressures_rel = gnn_model.forward(graph_data_loader).numpy()
    true_outlet_pressures_rel = graph_data_loader.nodes["outlet"].data["outlet_pressure"].numpy()
    print("GNN evaluated!")
    return pred_outlet_pressures_rel, true_outlet_pressures_rel, junc_ids
    # return pred_outlet_pressures, true_outlet_pressures, junc_ids

def apply_gnn(model_name, graph_list):

    f = open("../../data/scaling_dict_synthetic", "rb"); scaling_dict = pickle.load(f)
    graphs_dataset = DGL_Dataset(graph_list)
    graph_data_loader = dgl.batch([graph.to("/gpu:0") for graph in graphs_dataset])

    gnn_model = GraphNet(model_name = model_name)
    junc_ids = graph_data_loader.nodes['outlet'].data['outlet_identifiers'].numpy()

    pred_outlet_pressures_rel = inv_scale(scaling_dict,  gnn_model.forward(graph_data_loader).numpy(),"pressure_out_rel")
    true_outlet_pressures_rel = inv_scale(scaling_dict, graph_data_loader.nodes["outlet"].data["outlet_pressure"].numpy(), "pressure_out_rel")
    print("GNN evaluated!")
    return pred_outlet_pressures_rel, true_outlet_pressures_rel, junc_ids
    # return pred_outlet_pressures, true_outlet_pressures, junc_ids



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

def scale(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
    scaled_field = (field-mean)/std
    return scaled_field

def inv_scale(scaling_dict, field, field_name):
    mean = scaling_dict[field_name][0]; std = scaling_dict[field_name][1]
    scaled_field = (field*std)+mean
    return scaled_field

def get_graph(junction_dict):
    f = open("../../data/scaling_dict_synthetic", "rb"); scaling_dict = pickle.load(f)
    num_inlets = junction_dict["inlet_area"].size
    num_outlets = junction_dict["outlet_area"].size
    #import pdb; pdb.set_trace()
    min_pressure_in = np.min(junction_dict["inlet_pressure"])
    try:
        inlet_data = np.stack((scale(scaling_dict, junction_dict["inlet_pressure"], "pressure_in"),
            scale(scaling_dict, junction_dict["inlet_pressure_der"], "pressure_der_in"),
            scale(scaling_dict, junction_dict["inlet_pressure"]-min_pressure_in, "pressure_in_rel"),
            scale(scaling_dict, junction_dict["inlet_flow"], "flow_in"),
            scale(scaling_dict, junction_dict["inlet_flow_der"], "flow_der_in"),
            scale(scaling_dict, junction_dict["inlet_flow_der2"], "flow_der2_in"),
            scale(scaling_dict, junction_dict["inlet_flow_hist1"], "flow_hist1_in"),
            scale(scaling_dict, junction_dict["inlet_flow_hist1"], "flow_hist2_in"),
            scale(scaling_dict, junction_dict["inlet_flow_hist1"], "flow_hist3_in"),
            scale(scaling_dict, junction_dict["inlet_area"], "area_in"),
            np.asarray(junction_dict["inlet_angle"][:,0]),
            np.asarray(junction_dict["inlet_angle"][:,1]),
            np.asarray(junction_dict["inlet_angle"][:,2]),
            scale(scaling_dict, junction_dict["inlet_time_interval"], "time_interval_in"))).T
    except:
        import pdb; pdb.set_trace()


    outlet_data = np.stack((scale(scaling_dict, junction_dict["outlet_flow"], "flow_out"),
        scale(scaling_dict, junction_dict["outlet_flow_der"], "flow_der_out"),
        scale(scaling_dict, junction_dict["outlet_flow_der2"], "flow_der2_out"),
        scale(scaling_dict, junction_dict["outlet_flow_hist1"], "flow_hist1_out"),
        scale(scaling_dict, junction_dict["outlet_flow_hist1"], "flow_hist2_out"),
        scale(scaling_dict, junction_dict["outlet_flow_hist1"], "flow_hist3_out"),
        scale(scaling_dict, junction_dict["outlet_area"], "area_out"),
        np.asarray(junction_dict["outlet_angle"][:,0]),
        np.asarray(junction_dict["outlet_angle"][:,1]),
        np.asarray(junction_dict["outlet_angle"][:,2]),
        scale(scaling_dict, junction_dict["outlet_time_interval"], "time_interval_in"))).T



    inlet_outlet_pairs = get_inlet_outlet_pairs(num_inlets, num_outlets)
    outlet_pairs = get_outlet_pairs(num_outlets)

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)

    return graph

class DGL_Dataset(DGLDataset):

    def __init__(self, graphs):
        self.graphs = graphs
        super().__init__(name='dgl_dataset')

    def process(self):

        pass

    def __getitem__(self, i):

        return self.graphs[i]

    def __len__(self):

        return len(self.graphs)
