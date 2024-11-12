import pdb
import dgl
import tensorflow as tf
import dgl.function as fn
import numpy as np
import pdb

def MLP(in_feats, latent_space, out_feats, n_h_layers):

    encoder_in = tf.keras.layers.Dense(latent_space, activation=None, use_bias = True)
    encoder_out = tf.keras.layers.Dense(out_feats, activation=None, use_bias = True)
    n_h_layers = n_h_layers

    model = tf.keras.Sequential()
    model.add(encoder_in)
    model.add(tf.keras.layers.LeakyReLU())
    for i in range(n_h_layers):
        model.add(tf.keras.layers.Dense(latent_space, activation=None, use_bias = True))
        model.add(tf.keras.layers.LeakyReLU())
    model.add(encoder_out)

    return model

class GraphNet(tf.Module):
    def __init__(self, params):
        super(GraphNet, self).__init__()

        self.encoder_inlet_nodes = MLP(params["num_inlet_ft"],
                                        params['latent_size_mlp'],
                                        params['latent_size_mlp'],
                                        params['hl_mlp'])

        self.encoder_outlet_nodes = MLP(params["num_outlet_ft"],
                                        params['latent_size_mlp'],
                                        params['latent_size_mlp'],
                                        params['hl_mlp'])

        self.processor_outlet_nodes = []

        self.process_iters = params['process_iterations']
        for i in range(self.process_iters):
            def generate_proc_MLP(in_feat):
                return MLP(in_feat,
                           params['latent_size_mlp'],
                           params['latent_size_mlp'],
                           params['hl_mlp'])

            self.processor_outlet_nodes.append(generate_proc_MLP(params['latent_size_mlp']*3))

        self.output_pressure_drop = MLP(params['latent_size_mlp'],
                                   params['latent_size_mlp'],
                                   params['out_size'],
                                   params['hl_mlp'])

        self.params = params

    def get_model_list(self):
        model_list = [self.encoder_inlet_nodes,
                        self.encoder_outlet_nodes,
                        self.output_pressure_drop] + self.processor_outlet_nodes
        return model_list

    def encode_inlet_nodes(self, nodes):
        f = nodes.data["inlet_features"]
        enc_features = self.encoder_inlet_nodes(f)
        return {'proc_inlet': enc_features}

    def encode_outlet_nodes(self, nodes):
        f = nodes.data["outlet_features"]
        enc_features = self.encoder_outlet_nodes(f)
        return {'proc_outlet': enc_features}

    def process_nodes(self, nodes, processor):

        proc_node = processor(tf.concat((nodes.data['inlets_message'],
                                        nodes.data['outlets_message'],
                                        nodes.data['proc_outlet']), axis=1))
        return {'proc_node' : proc_node}

    def decode_pressure_drop(self, nodes):
        f = nodes.data['proc_node']
        h = self.output_pressure_drop(f)
        return {'h' : h}

    def forward(self, g):
        g.apply_nodes(self.encode_inlet_nodes, ntype="inlet")
        g.apply_nodes(self.encode_outlet_nodes, ntype="outlet")

        g.update_all(fn.copy_u('proc_inlet', 'm'), fn.mean('m', 'inlets_message'),
                               etype='inlet_to_outlet')
        g.update_all(fn.copy_u('proc_outlet', 'm'), fn.mean('m', 'outlets_message'),
                               etype='outlet_to_outlet')

        for i in range(self.process_iters):

            def pn_outlet(nodes):
                return self.process_nodes(nodes, self.processor_outlet_nodes[i])
            g.apply_nodes(pn_outlet, ntype='outlet')

        g.apply_nodes(self.decode_pressure_drop, ntype='outlet')

        return g.nodes['outlet'].data['h']

    def update_nn_weights(self, batched_graph, optimizer, loss, metric):
        with tf.GradientTape(persistent=True) as tape:
            tape.reset()
            pred_outlet_pressures = self.forward(batched_graph)
            true_outlet_pressures = tf.cast(batched_graph.nodes['outlet'].data['outlet_pressure'], dtype=tf.float32)

            loss_value = loss(pred_outlet_pressures, true_outlet_pressures)
            metric_value = metric(pred_outlet_pressures, true_outlet_pressures)

        model_list =  self.get_model_list()
        #pdb.set_trace()
        for model in model_list:
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, metric_value
