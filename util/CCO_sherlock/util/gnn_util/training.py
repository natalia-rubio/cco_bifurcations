import sys
import os
import dgl
print(dgl.__version__)
from util.gnn_util.graphnet import GraphNet
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
from datetime import datetime
import random
import time
import json
import pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
#tf = __import__("tensorflow-gpu")
import random
random.seed(10)
font = {"size"   : 14}

def mse(input, target):
    return tf.math.reduce_mean(tf.square(input - target))

def mae(input, target, weight = None):
    if weight == None:
        return tf.math.reduce_mean(tf.math.abs(input - target))
    return tf.math.reduce_mean(weight * (tf.math.abs(input - target)))

def generate_gnn_model(params_dict):
    return GraphNet(params_dict)
    #return GraphNet()

def evaluate_model(gnn_model, train_dataloader, loss, metric = None,
                   optimizer = None, continuity_coeff = 0.0,
                   bc_coeff = 0.0,
                   validation_dataloader = None,
                   train = True):

    def loop_over(dataloader, c_optimizer = None):
        if c_optimizer == None:
            training = False
        else:
            training = True

        global_loss = 0
        global_metric = 0
        count = 0

        for batched_graph in dataloader:

            if training:
                loss_value, metric_value = gnn_model.update_nn_weights(batched_graph, optimizer, loss, metric)
                global_loss = global_loss + loss_value.numpy()
                global_metric = global_metric + metric_value.numpy()
            else:
                pred_outlet_pressures = gnn_model.forward(batched_graph)
                true_outlet_pressures = tf.cast(batched_graph.nodes['outlet'].data['outlet_pressure'], dtype=tf.float32)

                loss_value = loss(pred_outlet_pressures, true_outlet_pressures)
                global_loss = global_loss + loss_value.numpy()

                metric_v = metric(pred_outlet_pressures, true_outlet_pressures)
                global_metric = global_metric + metric_v.numpy()

            count = count + 1
        return {'global_loss': global_loss, 'count': count, 'global_metric': global_metric}

    validation_results = None
    start = time.time()
    if validation_dataloader:
        validation_results = loop_over(validation_dataloader)
    train_results = loop_over(train_dataloader, optimizer)
    end = time.time()

    return train_results, validation_results, end - start

def evaluate_model_get_dataloaders(train_params, gnn_model, train_dataset, loss, metric = None,
                   optimizer = None, continuity_coeff = 0.0,
                   bc_coeff = 0.0,
                   validation_dataset = None,
                   train = True):

    train_dataloader = get_graph_data_loader(train_dataset, batch_size=train_params['batch_size'])
    validation_dataloader = get_graph_data_loader(validation_dataset, batch_size=train_params['batch_size'])
    def loop_over(dataloader, c_optimizer = None):
        if c_optimizer == None:
            training = False
        else:
            training = True

        global_loss = 0
        global_metric = 0
        count = 0

        for batched_graph in dataloader:

            if training:
                loss_value, metric_value = gnn_model.update_nn_weights(batched_graph, optimizer, loss, metric)
                global_loss = global_loss + loss_value.numpy()
                global_metric = global_metric + metric_value.numpy()
            else:
                pred_outlet_pressures = gnn_model.forward(batched_graph)
                true_outlet_pressures = tf.cast(batched_graph.nodes['outlet'].data['outlet_pressure'], dtype=tf.float32)

                loss_value = loss(pred_outlet_pressures, true_outlet_pressures)
                global_loss = global_loss + loss_value.numpy()

                metric_v = metric(pred_outlet_pressures, true_outlet_pressures)
                global_metric = global_metric + metric_v.numpy()

            count = count + 1
        return {'global_loss': global_loss, 'count': count, 'global_metric': global_metric}

    validation_results = None
    start = time.time()
    if validation_dataloader:
        validation_results = loop_over(validation_dataloader)
    train_results = loop_over(train_dataloader, optimizer)
    end = time.time()

    return train_results, validation_results, end - start

def get_graph_data_loader(dataset, batch_size):
    graph_data_loader = []
    num_samples = len(dataset); num_batches = int(np.ceil(num_samples/batch_size))
    print(f"num batches: {num_batches}.  num graphs: {num_samples}.")
    indices_shuff = [i for i in range(num_samples)]
    random.shuffle(indices_shuff)
    for batch_ind in range(num_batches):
        try:
            batch_indices = indices_shuff[batch_size*batch_ind : batch_size*(batch_ind+1)]
        except:
            batch_indices = indices_shuff[batch_size*batch_ind :]
        graph_data_loader.append(dgl.batch([dataset[ind].to("/gpu:0") for ind in batch_indices]))
    return graph_data_loader

def train_gnn_model(gnn_model, train_dataset, validation_dataset, optimizer_name, train_params, params_dict, trial=1, percent_train = 60, model_name = None, index = 0,
                    checkpoint_fct = None):

    print('Training dataset contains {:.0f} graphs'.format(len(train_dataset)))

    train_dataloader = get_graph_data_loader(train_dataset, batch_size=train_params['batch_size'])
    validation_dataloader = get_graph_data_loader(validation_dataset, batch_size=train_params['batch_size'])
    #validation_dataloader_all = get_graph_data_loader(validation_dataset, batch_size=len(validation_dataset))

    nepochs = train_params['nepochs']
    scheduler_name = 'exponential'
    if scheduler_name == 'exponential':
        scheduler = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = train_params['learning_rate'], decay_steps=100000, decay_rate=train_params['lr_decay'])
    elif scheduler_name == 'cosine':
        eta_min = train_params['learning_rate'] * train_params['lr_decay']
        scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate = train_params['learning_rate'], decay_steps=nepochs, alpha = eta_min)
    learning_rate = scheduler

    if optimizer_name == 'adam':
        #optimizer = tf.keras.optimizers.Adam(learning_rate = train_params['learning_rate'])
        optimizer = tf.keras.optimizers.Adam(learning_rate = scheduler)
        #optimizer = tf.keras.optimizers.SGD(train_params['learning_rate'])
    else:
        raise ValueError('Optimizer ' + optimizer_name + ' not implemented')

    if checkpoint_fct != None:
        # 200 is the maximum number of sigopt checkpoint
        chckp_epochs = list(np.floor(np.linspace(0, nepochs, 200)))

    loss_train_list = []; loss_val_list = []; mae_train_list = []; mae_val_list = []

    for epoch in range(nepochs):

        train_results, val_results, elapsed = evaluate_model(gnn_model = gnn_model,
                                                            train_dataloader = train_dataloader,
                                                            loss = mse,
                                                            metric = mae,
                                                            optimizer = optimizer,
                                                            validation_dataloader = validation_dataloader)
        msg = '{:.0f}\t'.format(epoch)
        msg = msg + 'train_loss = {:.2e} '.format(train_results['global_loss']/train_results['count'])
        msg = msg + 'train_mae = {:.2e} '.format(train_results['global_metric']/train_results['count'])
        msg = msg + 'val_loss = {:.2e} '.format(val_results['global_loss']/val_results['count'])
        msg = msg + 'val_mae = {:.2e} '.format(val_results['global_metric']/val_results['count'])
        msg = msg + 'time = {:.2f} s'.format(elapsed)
        loss_train_list.append(train_results['global_loss']/train_results['count'])
        loss_val_list.append(val_results['global_loss']/val_results['count'])
        mae_train_list.append(train_results['global_metric']/train_results['count'])
        mae_val_list.append(val_results['global_metric']/val_results['count'])
        print(msg, flush=True)

        if checkpoint_fct != None:
            if epoch in chckp_epochs:
                checkpoint_fct(global_loss/count)

    # compute final loss
    train_results, val_results, elapsed = evaluate_model(gnn_model = gnn_model,
                                                        train_dataloader = train_dataloader,
                                                        loss = mse,
                                                        metric = mae,
                                                        optimizer = optimizer,
                                                        validation_dataloader = validation_dataloader)
    if model_name == None:
        model_name = str(params_dict["hl_mlp"])[0:4] + "_hl_" + str(params_dict["latent_size_mlp"])[0:4] + "_lsmlp_" + (str(train_params["learning_rate"])[0:6]).replace(".", "_") + "_lr_"+ (str(train_params["lr_decay"])[0:5]).replace(".", "_") + "_lrd_"+ (str(train_params["weight_decay"])[0:5]).replace(".", "_") + "_wd_" + "index_" + str(index) + "_bs_" + str(train_params["batch_size"]) + "_nepochs_" + str(train_params["nepochs"])

    plt.clf()
    plt.style.use('dark_background')
    plt.scatter(np.linspace(1,nepochs, nepochs, endpoint=True), np.asarray(loss_train_list), color = "cornflowerblue", s=40, alpha = 0.6, marker='o', label="training")
    plt.scatter(np.linspace(1,nepochs, nepochs, endpoint=True), np.asarray(loss_val_list),  color = "coral", s=40, alpha = 0.6, marker='d', label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.title(f"MSE Over Epochs (Split {index})"); plt.legend(); plt.yscale("log")
    plt.savefig("data/training_plots/" + model_name + "mse_over_epochs.png", bbox_inches='tight', transparent=True)

    msg = 'end\t'
    msg = msg + 'train_loss = {:.2e} '.format(train_results['global_loss']/train_results['count'])
    msg = msg + 'train_mae = {:.2e} '.format(train_results['global_metric']/train_results['count'])
    msg = msg + 'val_loss = {:.2e} '.format(val_results['global_loss']/val_results['count'])
    msg = msg + 'time = {:.2f} s'.format(elapsed)
    print(msg, flush=True)

    train_mse = train_results['global_loss']/train_results['count']
    val_mse = val_results['global_loss']/val_results['count']
    return gnn_model, val_mse, train_mse
