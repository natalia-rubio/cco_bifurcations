import dgl
import tensorflow as tf
import numpy as np
from dgl.data import DGLDataset
from os.path import exists
import pickle
import pdb
import math
import random
import os

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        dict = pickle.load(f)
    return dict

def get_random_ind(num_pts, percent_train = 85 , seed = 0):

    ind = np.linspace(0,num_pts, num_pts, endpoint = False).astype(int); rng = np.random.default_rng(seed)
    train_ind = rng.choice(ind, size = int(num_pts * 0.01 * percent_train), replace = False)
    #print(train_ind)
    val_ind = ind[np.isin(ind, train_ind, invert = True)]
    return train_ind, val_ind

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

def generate_dataset(dataset_params = {}, seed = 0, num_geos = 130):

    print(os.getcwd())
    dataset_name = f'{dataset_params["source"]}_{dataset_params["features"]}'
    if exists(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset"):
        train_dataset = load_dict(f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
        val_dataset = load_dict(f"data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
    else:
        print(f"Generating Bifurcation Dataset ({dataset_name})")
        graph_list = dgl.load_graphs(f"data/master_data/graph_list_{dataset_name}")[0]; graph_arr = np.array(graph_list, dtype = "object")
        model_arr = np.load(f"data/master_data/model_list_{dataset_name}.npy");
        if dataset_params["source"] == "synthetic":
            model_list = [model[5:] for model in list(model_arr)]; model_arr = np.array(model_list)
        elif dataset_params["source"] == "vmr":
            model_list = [model[0:9] for model in list(model_arr)]; model_arr = np.array(model_list)

        random.seed(seed)
        geo_list = list(dict.fromkeys(model_list)); random.shuffle(geo_list); geo_arr = np.array(geo_list)
        print(f"Total number of unique models: {len(geo_list)}.")
        print(f"Total number of graphs: {len(graph_list)}.")
        #import pdb; pdb.set_trace()

        train_ind, val_ind = get_random_ind(num_pts = len(geo_list), seed = seed)
        num_train = int(0.85*num_geos); num_val = num_geos - num_train

        train_geos = geo_arr[train_ind[:num_train]]; val_geos = geo_arr[val_ind[:num_val]]
        print(f"train geos: {train_geos} \nval geos: {val_geos}")
        train_graph_list = (graph_arr[np.isin(model_arr, train_geos)])
        val_graph_list = (graph_arr[np.isin(model_arr, val_geos)])
        assert len(list(set(train_graph_list).intersection(val_graph_list))) == 0, "Common geometries between train and val sets."

        train_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, train_geos)]))
        val_dataset = DGL_Dataset(list(graph_arr[np.isin(model_arr, val_geos)]))

        np.save(f"data/split_indices/train_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", train_ind)
        np.save(f"data/split_indices/val_ind_{dataset_name}_num_geos_{num_geos}_seed_{seed}", val_ind)

        save_dict(train_graph_list, f"data/graph_lists/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")
        save_dict(val_graph_list, f"data/graph_lists/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_graph_list")

        save_dict(train_dataset, f"data/datasets/train_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")
        save_dict(val_dataset, f"data/datasets/val_{dataset_name}_num_geos_{num_geos}_seed_{seed}_dataset")

    return train_dataset, val_dataset

if __name__ == '__main__':
    dataset_params = {"source": "synthetic", # where to extraxt graphs from (synthetic or vmr)
                    "features": "full", # complexity of features (full or reduced)
                    "junction_type": "y"
                    }
    generate_dataset(dataset_params = dataset_params, seed = 0, num_geos = 150)
