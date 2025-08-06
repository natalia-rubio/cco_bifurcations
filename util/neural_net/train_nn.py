from tarfile import tar_filter
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import optax
import os
from util.tools.basic import save_dict
from util.neural_net.nn_util import get_batch_indices, dill_save
from util.neural_net.nn_model import loss, loss_pure
import time
import matplotlib.pyplot as plt
import pdb
import multiprocessing
import dill



def train_nn(model, training_params):
    model_name2 = f"{model.output_type}_{model.anatomy}_pred_{model.target_coef_ind}_outlet_{model.outlet_number}"
    model_name = f"{model.output_type}_{model.anatomy}_ng_{model.num_geos}_nl_{model.num_layers}_lw_{model.layer_width}_ne_{training_params['num_epochs']}_bs_{training_params['batch_size']}_dr_{model.decay_rate}_{model.set_type}_pred_{model.target_coef_ind}_outlet_{model.outlet_number}"
    plotting = True
    train_hist = []
    val_hist = []
    
    num_offsets = training_params["num_offsets"]
    print("Number of offsets: ", num_offsets)
    train_inds = np.concatenate([training_params["train_inds"] * num_offsets  + i for i in range(num_offsets)])
    val_inds   = np.concatenate([training_params["val_inds"] * num_offsets + i for i in range(num_offsets)])

    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        batch_ind_list = get_batch_indices(train_inds, 
                                           training_params['batch_size']) # Split the training set into random batches
        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
            model.update(indices = batch_inds) # Update the model based on the batch
        # with multiprocessing.Pool() as pool:
        #         pool.map(model.update, batch_ind_list) # Update the model based on the batch

        epoch_time = time.time() - start_time
 
        train_loss = loss_pure(input = model.input[train_inds,:],
                        outputs= model.output[train_inds,:],
                        scaling_factors = model.data_dict["scaling_factors"][train_inds,:],
                        scaling_dict = model.scaling_dict,
                        target_coef_ind = model.target_coef_ind,
                        weights = model.weights)
        train_hist.append(train_loss)

        val_loss = loss_pure(input = model.input[val_inds,:],
                        outputs= model.output[val_inds,:],
                        scaling_factors = model.data_dict["scaling_factors"][val_inds,:],
                        scaling_dict = model.scaling_dict,
                        target_coef_ind = model.target_coef_ind,
                        weights = model.weights)
        val_hist.append(val_loss)

        print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
              "Training set accuracy {:e}  |  ".format(train_loss) + \
              "Validation set accuracy {:e}".format(val_loss))
        
        if (epoch+1)%100 == 0 and plotting:
            if epoch == 0:
                 continue
            plt.clf()
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(train_hist), label = "Training Loss", color = 'cornflowerblue')
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(val_hist), label = "Validation Loss", color = 'salmon')
            plt.xlabel("Epoch"); plt.ylabel("Loss (RMSE) (mmHg)"); plt.title("Training and Validation Loss")
            plt.yscale("log")
            plt.legend()
            if not os.path.exists(f"/Users/natalia/Desktop/cco_bifurcations/results/models/{model.anatomy}"):
                os.makedirs(f"/Users/natalia/Desktop/cco_bifurcations/results/models/{model.anatomy}")
            plt.savefig(f"/Users/natalia/Desktop/cco_bifurcations/results/models/{model.anatomy}/{model_name}_training_plot.png", bbox_inches='tight')

    dill_save(model, f"/Users/natalia/Desktop/cco_bifurcations/results/models/{model.anatomy}/{model_name}_model")
    dill_save(model, f"/Users/natalia/Desktop/cco_bifurcations/results/models/{model.anatomy}/{model_name2}_model")
    return val_loss.item()