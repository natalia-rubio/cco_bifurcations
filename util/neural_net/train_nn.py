import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import numpy as np
import optax
import os
from util.tools.basic import save_dict
from util.neural_net.nn_util import get_batch_indices, dill_save
from util.neural_net.nn_model import loss, coef_loss
import time
import matplotlib.pyplot as plt
import pdb
plt.rcParams.update(plt.rcParamsDefault)
def train_nn(model, training_params):

    model_name = f"{model.anatomy}_ng_{model.num_geos}_nl_{model.num_layers}_lw_{model.layer_width}_ne_{training_params['num_epochs']}_bs_{training_params['batch_size']}_dr_{model.decay_rate}"
    plotting = True
    # if plotting:
        # plt.show()
    train_hist = []
    val_hist = []
    train_mag = jnp.sqrt(jnp.mean(jnp.square((model.data_dict["dPs"][training_params["train_inds"],:,:])/1333)))
    val_mag = jnp.sqrt(jnp.mean(jnp.square((model.data_dict["dPs"][training_params["val_inds"],:,:])/1333)))
    
    # train_coef_loss = coef_loss(output = model.data_dict["output"][0:1,:],
    #         flow =  model.data_dict["flows"][0:1,:,:],
    #         dP_true = model.data_dict["dPs"][0:1,:,:],
    #         scaling_factors = model.data_dict["scaling_factors"][0:1,:],
    #         scaling_dict = model.scaling_dict)
    # pdb.set_trace()
    
    train_coef_loss = coef_loss(output = model.data_dict["output"][training_params["train_inds"],:],
                flow =  model.data_dict["flows"][training_params["train_inds"],:,:],
                dP_true = model.data_dict["dPs"][training_params["train_inds"],:,:],
                scaling_factors = model.data_dict["scaling_factors"][training_params["train_inds"],:],
                scaling_dict = model.scaling_dict)
    
    val_coef_loss = coef_loss(output = model.data_dict["output"][training_params["val_inds"],:],
            flow =  model.data_dict["flows"][training_params["val_inds"],:,:],
            dP_true = model.data_dict["dPs"][training_params["val_inds"],:,:],
            scaling_factors = model.data_dict["scaling_factors"][training_params["val_inds"],:],
            scaling_dict = model.scaling_dict)
    #pdb.set_trace()

    for epoch in range(training_params['num_epochs']): # Loop through the epochs
        start_time = time.time() # Time each epoch
        batch_ind_list = get_batch_indices(training_params["train_inds"], 
                                           training_params['batch_size']) # Split the training set into random batches
        for i, batch_inds in enumerate(batch_ind_list): # Loop through the batches
                model.update(indices = batch_inds) # Update the model based on the batch
        epoch_time = time.time() - start_time


        train_loss = loss(input = model.data_dict["input"][training_params["train_inds"],:],
                        flow =  model.data_dict["flows"][training_params["train_inds"],:,:],
                        dP_true = model.data_dict["dPs"][training_params["train_inds"],:,:],
                        scaling_factors = model.data_dict["scaling_factors"][training_params["train_inds"],:],
                        scaling_dict = model.scaling_dict,
                        weights = model.weights)
        train_hist.append(train_loss)

        val_loss = loss(input = model.data_dict["input"][training_params["val_inds"],:],
                        flow =  model.data_dict["flows"][training_params["val_inds"],:,:],
                        dP_true = model.data_dict["dPs"][training_params["val_inds"],:,:],
                        scaling_factors = model.data_dict["scaling_factors"][training_params["val_inds"],:],
                        scaling_dict = model.scaling_dict,
                        weights = model.weights)
        val_hist.append(val_loss)

        print("Epoch {} in {:0.2f} sec  |  ".format(epoch, epoch_time) + \
              "Training set accuracy {:e}  |  ".format(train_loss) + \
              "Validation set accuracy {:e}".format(val_loss))
        
                # Save epoch results
        
        if epoch%100 == 0 and plotting:
            if epoch == 0:
                 continue
            plt.clf()
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(train_hist), label = "Training Loss", color = 'cornflowerblue')
            plt.hlines(train_mag, 0, epoch, color = 'cornflowerblue', linestyle = '--', label = "Training Delta P RMSE")
            plt.hlines(train_coef_loss, 0, epoch, color = 'cornflowerblue', linestyle = ':', label = "Training Best Coef RMSE")
            plt.plot(np.linspace(0, epoch, epoch+1, True), np.asarray(val_hist), label = "Validation Loss", color = 'salmon')
            plt.hlines(val_mag, 0, epoch, color = 'salmon', linestyle = '--', label = "Validation Delta P RMSE")
            plt.hlines(val_coef_loss, 0, epoch, color = 'salmon', linestyle = ':', label = "Validation Best Coef RMSE")
            plt.xlabel("Epoch"); plt.ylabel("Loss (RMSE) (mmHg)"); plt.title("Training and Validation Loss")
            plt.yscale("log")
            plt.legend()
            if not os.path.exists(f"results/models/{model.anatomy}"):
                os.makedirs(f"results/models/{model.anatomy}")
            plt.savefig(f"results/models/{model.anatomy}/{model_name}_training_plot.png")
            #pdb.set_trace()

    dill_save(model, f"results/models/{model.anatomy}/{model_name}_model")
    
    return train_loss, val_loss