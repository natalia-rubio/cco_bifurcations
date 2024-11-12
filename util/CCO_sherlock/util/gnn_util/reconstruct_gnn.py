import pdb
import dgl
import tensorflow as tf
import dgl.function as fn
import numpy as np
import pdb
import pickle
class GraphNetReconstructed(tf.Module):
    def __init__(self, model_name):
        super(GraphNet, self).__init__()
        
        self.encoder_inlet_nodes = tf.keras.models.load_model("../cross_validation/models/"+model_name+"encoder_inlet_nodes", compile=False)
        self.encoder_outlet_nodes = tf.keras.models.load_model("../cross_validation/models/"+model_name+"encoder_outlet_nodes", compile=False)

        num_layers = len(self.encoder_inlet_nodes.layers)
        latent_size_mlp = self.encoder_inlet_nodes.layers[-1].output_shape[1]
        #import pdb; pdb.set_trace()
        self.processor_inlet_nodes = []
        self.processor_outlet_nodes = []
        self.process_iters = 1
        self.processor_outlet_nodes = [tf.keras.models.load_model("../cross_validation/models/"+model_name+"processor_outlet_nodes0", compile=False)]
        self.output_pressure_drop = tf.keras.models.load_model("../cross_validation/models/"+model_name+"output_pressure_drop", compile=False)


    def encode_inlet_nodes(self, nodes):
        #print('encoding inlets')
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
        #print("Applying graph net forward pass")
        g.apply_nodes(self.encode_inlet_nodes, ntype="inlet")
        g.apply_nodes(self.encode_outlet_nodes, ntype="outlet")
        #print("Inlet and outlets encoded")

        g.update_all(fn.copy_u('proc_inlet', 'm'), fn.mean('m', 'inlets_message'),
                               etype='inlet_to_outlet')
        g.update_all(fn.copy_u('proc_outlet', 'm'), fn.mean('m', 'outlets_message'),
                               etype='outlet_to_outlet')
        #print("Encoded messages sent to outlets.")

        for i in range(self.process_iters):
            def pn_outlet(nodes):
                return self.process_nodes(nodes, self.processor_outlet_nodes[i])
            g.apply_nodes(pn_outlet, ntype='outlet')
        #"Messages received at outlets processed."

        g.apply_nodes(self.decode_pressure_drop, ntype='outlet')

        return g.nodes['outlet'].data['h']
