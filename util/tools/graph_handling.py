from util.tools.junction_proc import *

def get_reduced_graph(scaling_dict, pressure, flow, area, angle1, angle2, angle3, inlet_pts, outlet_pts):
    """
    Build reduced-feature DGL graph
    Returns: reduced-feature graph
    """
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_data = np.stack((scale(scaling_dict, pressure[inlet_pts], "pressure_in"), #0
        scale(scaling_dict, np.abs(flow[inlet_pts]), "flow_in"),
        scale(scaling_dict, area[inlet_pts], "area_in")
        )).T

    outlet_data = np.stack((scale(scaling_dict, np.abs(flow[outlet_pts]), "flow_out"),
        scale(scaling_dict, area[outlet_pts], "area_out"),#1
        np.array([get_angle_diff(
                np.array([angle1[inlet_pts][0], angle2[inlet_pts][0], angle3[inlet_pts][0]]),
                np.array([angle1[outlet_pts][0], angle2[outlet_pts][0], angle3[outlet_pts][0]])),
                get_angle_diff(np.array([angle1[inlet_pts][0], angle2[inlet_pts][0], angle3[inlet_pts][0]]), np.array([angle1[outlet_pts][1], angle2[outlet_pts][1], angle3[outlet_pts][1]]))]))).T#2

    pressure_out = scale(scaling_dict, pressure[outlet_pts]-min_pressure_in, "pressure_out_rel").reshape(-1,1)

    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out, dtype=tf.float32)

    return graph

def get_geo_graph_dP(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    print("Getting graph (geo features + dP output)")
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_angles = np.asarray([angle1[inlet_pts],
                            angle2[inlet_pts],#8
                            angle3[inlet_pts]])

    outlet_angles = np.asarray([angle1[outlet_pts],
                            angle2[outlet_pts],#8
                            angle3[outlet_pts]])

    angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
    angle_diffs = angle_diffs.reshape((len(outlet_pts),))

    inlet_data = np.stack((
        scale(scaling_dict, area[inlet_pts], "area_in")
        )).reshape((-1,1))

    outlet_data = np.stack((
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle_diffs)).T#10

    pressure_out = (pressure[outlet_pts]-min_pressure_in).reshape(-1,1)
    flow_out = np.abs(flow[outlet_pts]).reshape(-1,1)
    res_out = scale(scaling_dict, pressure_out/flow_out, "res_out")
    #import pdb; pdb.set_trace()
    pressure_out_scaled = scale(scaling_dict, pressure_out, "pressure_out_rel")
    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})

    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_resistance"] = tf.convert_to_tensor(res_out, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out_scaled, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_flow"] = tf.convert_to_tensor(flow_out, dtype=tf.float32)
    #import pdb; pdb.set_trace()
    return graph


def get_geo_graph_res(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_angles = np.asarray([angle1[inlet_pts],
                            angle2[inlet_pts],#8
                            angle3[inlet_pts]])

    outlet_angles = np.asarray([angle1[outlet_pts],
                            angle2[outlet_pts],#8
                            angle3[outlet_pts]])

    angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
    angle_diffs = angle_diffs.reshape((len(outlet_pts),))

    inlet_data = np.stack((
        scale(scaling_dict, area[inlet_pts], "area_in")
        )).reshape((-1,1))

    outlet_data = np.stack((
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle_diffs)).T#10

    pressure_out = (pressure[outlet_pts]-min_pressure_in).reshape(-1,1)
    flow_out = np.abs(flow[outlet_pts]).reshape(-1,1)
    res_out = scale(scaling_dict, pressure_out/flow_out, "res_out")
    #import pdb; pdb.set_trace()
    pressure_out_scaled = scale(scaling_dict, pressure_out, "pressure_out_rel")
    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})

    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_resistance"] = tf.convert_to_tensor(res_out, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out_scaled, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_flow"] = tf.convert_to_tensor(flow_out, dtype=tf.float32)
    #import pdb; pdb.set_trace()
    return graph

def get_geo_graph_energy_res(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_angles = np.asarray([angle1[inlet_pts],
                            angle2[inlet_pts],#8
                            angle3[inlet_pts]])

    outlet_angles = np.asarray([angle1[outlet_pts],
                            angle2[outlet_pts],#8
                            angle3[outlet_pts]])

    angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
    angle_diffs = angle_diffs.reshape((len(outlet_pts),))

    inlet_data = np.stack((
        scale(scaling_dict, area[inlet_pts], "area_in")
        )).reshape((-1,1))

    outlet_data = np.stack((
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle_diffs)).T#10

    pressure_out = (pressure[outlet_pts]-min_pressure_in).reshape(-1,1)
    flow_out = np.square(np.abs(flow[outlet_pts])).reshape(-1,1)
    res_out = scale(scaling_dict, pressure_out/flow_out, "energy_res_out")
    #import pdb; pdb.set_trace()
    pressure_out_scaled = scale(scaling_dict, pressure_out, "pressure_out_rel")
    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})

    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_energy_resistance"] = tf.convert_to_tensor(res_out, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out_scaled, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_flow"] = tf.convert_to_tensor(flow_out, dtype=tf.float32)
    #import pdb; pdb.set_trace()
    return graph

def get_full_res_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    print("getting full resistance graph")
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_angles = np.asarray([angle1[inlet_pts],
                            angle2[inlet_pts],#8
                            angle3[inlet_pts]])

    outlet_angles = np.asarray([angle1[outlet_pts],
                            angle2[outlet_pts],#8
                            angle3[outlet_pts]])

    angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
    angle_diffs = angle_diffs.reshape((len(outlet_pts),))

    min_pressure_in = np.min(pressure[inlet_pts])

    inlet_data = np.stack((scale(scaling_dict, pressure[inlet_pts], "pressure_in"), #0
        scale(scaling_dict, pressure_der[inlet_pts], "pressure_der_in"),
        pressure[inlet_pts]-min_pressure_in,#2
        scale(scaling_dict, np.abs(flow[inlet_pts]), "flow_in"),
        scale(scaling_dict, flow_der[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der_in"),#4
        scale(scaling_dict, flow_der2[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der2_in"),
        scale(scaling_dict, np.abs(flow_hist1[inlet_pts]), "flow_hist1_in"),#6
        scale(scaling_dict, np.abs(flow_hist2[inlet_pts]), "flow_hist2_in"),
        scale(scaling_dict, np.abs(flow_hist3[inlet_pts]), "flow_hist3_in"),#8
        scale(scaling_dict, area[inlet_pts], "area_in"),
        np.tile(time_interval, len(inlet_pts)))).T

    outlet_data = np.stack((scale(scaling_dict, np.abs(flow[outlet_pts]), "flow_out"), #0
        scale(scaling_dict, flow_der[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der_out"),
        scale(scaling_dict, flow_der2[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der2_out"),#2
        scale(scaling_dict, np.abs(flow_hist1[outlet_pts]), "flow_hist1_out"),
        scale(scaling_dict, np.abs(flow_hist2[outlet_pts]), "flow_hist2_out"),#4
        scale(scaling_dict, np.abs(flow_hist3[outlet_pts]), "flow_hist3_out"),
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle_diffs,
        np.tile(time_interval, len(outlet_pts)))).T#10

    pressure_out = (pressure[outlet_pts]-min_pressure_in).reshape(-1,1)
    flow_out = np.abs(flow[outlet_pts]).reshape(-1,1)
    res_out = scale(scaling_dict, pressure_out/flow_out, "res_out")
    #import pdb; pdb.set_trace()
    pressure_out_scaled = scale(scaling_dict, pressure_out, "pressure_out_rel")
    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_resistance"] = tf.convert_to_tensor(res_out, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out_scaled, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_flow"] = tf.convert_to_tensor(flow_out, dtype=tf.float32)

    return graph

def get_full_energy_res_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    print("getting full resistance graph")
    min_pressure_in = np.min(pressure[inlet_pts])
    inlet_angles = np.asarray([angle1[inlet_pts],
                            angle2[inlet_pts],#8
                            angle3[inlet_pts]])

    outlet_angles = np.asarray([angle1[outlet_pts],
                            angle2[outlet_pts],#8
                            angle3[outlet_pts]])

    angle_diffs = get_angle_diff(inlet_angles, outlet_angles)/np.pi
    angle_diffs = angle_diffs.reshape((len(outlet_pts),))

    min_pressure_in = np.min(pressure[inlet_pts])

    inlet_data = np.stack((scale(scaling_dict, pressure[inlet_pts], "pressure_in"), #0
        scale(scaling_dict, pressure_der[inlet_pts], "pressure_der_in"),
        pressure[inlet_pts]-min_pressure_in,#2
        scale(scaling_dict, np.abs(flow[inlet_pts]), "flow_in"),
        scale(scaling_dict, flow_der[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der_in"),#4
        scale(scaling_dict, flow_der2[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der2_in"),
        scale(scaling_dict, np.abs(flow_hist1[inlet_pts]), "flow_hist1_in"),#6
        scale(scaling_dict, np.abs(flow_hist2[inlet_pts]), "flow_hist2_in"),
        scale(scaling_dict, np.abs(flow_hist3[inlet_pts]), "flow_hist3_in"),#8
        scale(scaling_dict, area[inlet_pts], "area_in"),
        np.tile(time_interval, len(inlet_pts)))).T #10

    outlet_data = np.stack((scale(scaling_dict, np.abs(flow[outlet_pts]), "flow_out"), #0
        scale(scaling_dict, flow_der[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der_out"),
        scale(scaling_dict, flow_der2[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der2_out"),#2
        scale(scaling_dict, np.abs(flow_hist1[outlet_pts]), "flow_hist1_out"),
        scale(scaling_dict, np.abs(flow_hist2[outlet_pts]), "flow_hist2_out"),#4
        scale(scaling_dict, np.abs(flow_hist3[outlet_pts]), "flow_hist3_out"),
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle_diffs,
        np.tile(time_interval, len(outlet_pts)))).T#10

    pressure_out = (pressure[outlet_pts]-min_pressure_in).reshape(-1,1)
    flow_out = np.square(np.abs(flow[outlet_pts])).reshape(-1,1)
    res_out = scale(scaling_dict, pressure_out/flow_out, "energy_res_out")
    #import pdb; pdb.set_trace()
    pressure_out_scaled = scale(scaling_dict, pressure_out, "pressure_out_rel")
    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_energy_resistance"] = tf.convert_to_tensor(res_out, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out_scaled, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_flow"] = tf.convert_to_tensor(flow_out, dtype=tf.float32)

    return graph

def get_full_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """

    min_pressure_in = np.min(pressure[inlet_pts])

    inlet_data = np.stack((scale(scaling_dict, pressure[inlet_pts], "pressure_in"), #0
        scale(scaling_dict, pressure_der[inlet_pts], "pressure_der_in"),
        pressure[inlet_pts]-min_pressure_in,#2
        scale(scaling_dict, np.abs(flow[inlet_pts]), "flow_in"),
        scale(scaling_dict, flow_der[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der_in"),#4
        scale(scaling_dict, flow_der2[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der2_in"),
        scale(scaling_dict, np.abs(flow_hist1[inlet_pts]), "flow_hist1_in"),#6
        scale(scaling_dict, np.abs(flow_hist2[inlet_pts]), "flow_hist2_in"),
        scale(scaling_dict, np.abs(flow_hist3[inlet_pts]), "flow_hist3_in"),#8
        scale(scaling_dict, area[inlet_pts], "area_in"),
        angle1[inlet_pts],#10
        angle2[inlet_pts],
        angle3[inlet_pts],#12
        np.tile(time_interval, len(inlet_pts)))).T

    outlet_data = np.stack((scale(scaling_dict, np.abs(flow[outlet_pts]), "flow_out"), #0
        scale(scaling_dict, flow_der[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der_out"),
        scale(scaling_dict, flow_der2[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der2_out"),#2
        scale(scaling_dict, np.abs(flow_hist1[outlet_pts]), "flow_hist1_out"),
        scale(scaling_dict, np.abs(flow_hist2[outlet_pts]), "flow_hist2_out"),#4
        scale(scaling_dict, np.abs(flow_hist3[outlet_pts]), "flow_hist3_out"),
        scale(scaling_dict, area[outlet_pts], "area_out"),#6
        angle1[outlet_pts],
        angle2[outlet_pts],#8
        angle3[outlet_pts],
        np.tile(time_interval, len(outlet_pts)))).T#10

    pressure_out = scale(scaling_dict, pressure[outlet_pts]-min_pressure_in, "pressure_out_rel").reshape(-1,1)

    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out, dtype=tf.float32)

    return graph

def get_flow_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Build full-feature DGL graph
    Returns: full-feature graph
    """
    min_pressure_in = np.min(pressure[inlet_pts])

    inlet_data = np.stack((
        scale(scaling_dict, np.abs(flow[inlet_pts]), "flow_in"), # 0
        scale(scaling_dict, np.abs(flow[inlet_pts])/area[inlet_pts], "vel_in"), #1
        scale(scaling_dict, flow_der[inlet_pts]*np.sign(flow[inlet_pts]), "flow_der_in"), #2
        scale(scaling_dict, flow_der[inlet_pts]*np.sign(flow[inlet_pts])/area[inlet_pts], "vel_der_in"), #3
        scale(scaling_dict, area[inlet_pts], "area_in") #4
        )).T

    outlet_data = np.stack((
        scale(scaling_dict, np.abs(flow[outlet_pts]), "flow_out"), # 0
        scale(scaling_dict, np.abs(flow[outlet_pts])/area[outlet_pts], "vel_out"), #1
        scale(scaling_dict, flow_der[outlet_pts]*np.sign(flow[outlet_pts]), "flow_der_out"), #2
        scale(scaling_dict, flow_der[outlet_pts]*np.sign(flow[outlet_pts])/area[outlet_pts], "vel_der_out"), #3
        scale(scaling_dict, area[outlet_pts], "area_out") #4
        )).T # 5

    pressure_out = scale(scaling_dict, pressure[outlet_pts]-min_pressure_in, "pressure_out_rel").reshape(-1,1)

    inlet_outlet_pairs = get_inlet_outlet_pairs(len(inlet_pts), len(outlet_pts))
    outlet_pairs = get_outlet_pairs(len(outlet_pts))

    graph = dgl.heterograph({('inlet', 'inlet_to_outlet', 'outlet'): inlet_outlet_pairs,('outlet', 'outlet_to_outlet', 'outlet'): outlet_pairs})
    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_pressure"] = tf.convert_to_tensor(pressure_out, dtype=tf.float32)

    return graph

def add_synthetic_identifiers(graph, inlets, outlets, geo, flow_ind, time_index):
    """
    Add junction identifiers to graph (synthetic)
    Returns: graph including junction identifiers
    """
    inlet_identifiers = np.stack((np.asarray(inlets),
        np.tile(float(geo[5:]), len(inlets)),
        np.tile(flow_ind, len(inlets)),
        np.tile(time_index, len(inlets)))).T

    outlet_identifiers = np.stack((np.asarray(outlets),
        np.tile(float(geo[5:]), len(outlets)),
        np.tile(flow_ind, len(outlets)),
        np.tile(time_index, len(outlets)))).T

    graph.nodes["inlet"].data["inlet_identifiers"] = tf.convert_to_tensor(inlet_identifiers, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_identifiers"] = tf.convert_to_tensor(outlet_identifiers, dtype=tf.float32)
    return graph

def add_vmr_identifiers(graph, inlets, outlets, model, junction_id, time_index):
    """
    Add junction identifiers to graph (VMR)
    Returns: graph including junction identifiers
    """
    inlet_identifiers = np.stack((np.asarray(inlets),
        np.tile(float(model[0:4] + model[5:9]), len(inlets)),
        np.tile(junction_id, len(inlets)),
        np.tile(time_index, len(inlets)))).T

    outlet_identifiers = np.stack((np.asarray(outlets),
        np.tile(float(model[0:4] + model[5:9]), len(outlets)),
        np.tile(junction_id, len(outlets)),
        np.tile(time_index, len(outlets)))).T

    graph.nodes["inlet"].data["inlet_identifiers"] = tf.convert_to_tensor(inlet_identifiers, dtype=tf.float32)
    graph.nodes["outlet"].data["outlet_identifiers"] = tf.convert_to_tensor(outlet_identifiers, dtype=tf.float32)
    return graph

def get_graph(dataset_params, scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
    flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
    time_interval, inlet_pts, outlet_pts):
    """
    Get graph specified in dataset_params
    Returns: graph
    """
    if dataset_params["features"] == "reduced" :
        graph = get_reduced_graph(scaling_dict, pressure, flow, area, angle1, angle2, angle3, inlet_pts, outlet_pts)
    elif dataset_params["features"] == "full" and dataset_params["output"] == "dP":
        graph = get_full_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "geo" and dataset_params["output"] == "dP":
        graph = get_geo_graph_dP(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "flow":
        graph = get_flow_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "geo" and dataset_params["output"] == "res":
        graph = get_geo_graph_res(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "full" and dataset_params["output"] == "res":
        graph = get_full_res_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "geo" and dataset_params["output"] == "energy_res":
        graph = get_geo_graph_energy_res(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    elif dataset_params["features"] == "full" and dataset_params["output"] == "energy_res":
        graph = get_full_energy_res_graph(scaling_dict, pressure, pressure_der, flow, flow_der, flow_der2, \
            flow_hist1, flow_hist2, flow_hist3, area, angle1, angle2, angle3, \
            time_interval = 0.01, inlet_pts = inlet_pts, outlet_pts = outlet_pts)
    else:
        print(f"Didn't recognize graph type.")
    return graph

def save_junction_graphs(graph_list_dir, dataset_params, graph_list, model_list, flow_list = 0):
    """
    save graphs for later use.
    """
    dataset_name = f"{dataset_params['source']}_{dataset_params['features']}_{dataset_params['output']}_{dataset_params['filter']}_{dataset_params['scale_type']}_scale"
    if dataset_params["augmentation"] == "angle":
        dataset_name += "_angle_augment"
    dgl.save_graphs(f"{graph_list_dir}/graph_list_{dataset_name}", graph_list)
    np.save(f"{graph_list_dir}/model_list_{dataset_name}", np.asarray(model_list).astype(str), allow_pickle = True)
    if flow_list != 0:
        np.save(f"{graph_list_dir}/flow_list_{dataset_name}", np.asarray(flow_list), allow_pickle = True)
    return

def rotate_graph(original_graph):
    """
    Rotate inlet and outlet tangets for data augmentation
    """

    num_rotations = 3
    rot_inc = 2*np.pi/num_rotations
    rotated_graphs = []
    for rot_x in range(num_rotations):
        a = rot_inc * rot_x # angle of rotation around x-axis
        for rot_y in range(num_rotations):
            b = rot_inc * rot_y # angle of rotation around y-axis
            for rot_z in range(num_rotations):
                c = rot_inc * rot_z # angle of rotation around z-axis

                rot_mat = np.array([
                [np.cos(b)*np.cos(c) , np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c) , np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)],
                [np.cos(b)*np.sin(c) , np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c) , np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c)],
                [-np.sin(b) , np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)]
                ])

                graph = copy.deepcopy(original_graph)
                # Rotate inlets
                inlet_data = graph.nodes["inlet"].data["inlet_features"].numpy()
                for inlet_ind in range(inlet_data.shape[0]):
                    angles = inlet_data[inlet_ind, 10:13].reshape((-1,1))
                    #print(f"angles: {angles}")
                    new_angles = rot_mat @ angles
                    #print(f"new_angles: {new_angles}")
                    inlet_data[inlet_ind, 10:13] = new_angles.reshape((3,))
                    graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)

                # Rotate outlets
                outlet_data = graph.nodes["outlet"].data["outlet_features"].numpy()
                for outlet_ind in range(outlet_data.shape[0]):
                    angles = outlet_data[outlet_ind, 7:10].reshape((-1,1))
                    new_angles = rot_mat @ angles
                    outlet_data[outlet_ind, 7:10] = new_angles.reshape((3,))
                    graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)

                rotated_graphs.append(graph)

    assert len(rotated_graphs) == num_rotations**3;
    return rotated_graphs

def rotate_graph_rand(original_graph):
    """
    Rotate inlet and outlet tangets for data augmentation
    """

    num_rotations = 3
    rotated_graphs = []
    for i in range(num_rotations):
        a =  np.random.randint(low = 0, high = 2* np.pi) # angle of rotation around x-axis
        b =  np.random.randint(low = 0, high = 2* np.pi) # angle of rotation around y-axis
        c = np.random.randint(low = 0, high = 2* np.pi) # angle of rotation around z-axis

        rot_mat = np.array([
        [np.cos(b)*np.cos(c) , np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c) , np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)],
        [np.cos(b)*np.sin(c) , np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c) , np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c)],
        [-np.sin(b) , np.sin(a)*np.cos(b), np.cos(a)*np.cos(b)]
        ])

        graph = copy.deepcopy(original_graph)
        # Rotate inlets
        inlet_data = graph.nodes["inlet"].data["inlet_features"].numpy()
        for inlet_ind in range(inlet_data.shape[0]):
            angles = inlet_data[inlet_ind, 10:13].reshape((-1,1))
            #print(f"angles: {angles}")
            new_angles = rot_mat @ angles
            #print(f"new_angles: {new_angles}")
            inlet_data[inlet_ind, 10:13] = new_angles.reshape((3,))
            graph.nodes["inlet"].data["inlet_features"] = tf.convert_to_tensor(inlet_data, dtype=tf.float32)

        # Rotate outlets
        outlet_data = graph.nodes["outlet"].data["outlet_features"].numpy()
        for outlet_ind in range(outlet_data.shape[0]):
            angles = outlet_data[outlet_ind, 7:10].reshape((-1,1))
            new_angles = rot_mat @ angles
            outlet_data[outlet_ind, 7:10] = new_angles.reshape((3,))
            graph.nodes["outlet"].data["outlet_features"] = tf.convert_to_tensor(outlet_data, dtype=tf.float32)

        rotated_graphs.append(graph)

    assert len(rotated_graphs) == num_rotations;
    return rotated_graphs
