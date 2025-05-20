import json
import pdb
import sys
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from util.tools.junction_proc import *
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load
import jax.numpy as jnp
from util.neural_net.nn_model import NeuralNet, predict
from util.tree.get_0d_junction_dict import get_input_file_junction_dict_master
from util.tree.centerline_proj import extract_results
from util.tools.basic import *
from util.tools.junction_proc import get_angle_diff
from util.neural_net.nn_util import scale_jax, inv_scale_jax, dill_load 
from util.zerod.standard_to_RR import check_out_of_dist

from fpdf import FPDF
import matplotlib.pyplot as plt
import io
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10
#plt.rc('text', usetex=True)
colors = ["royalblue", "orangered", "seagreen", "peru", "blueviolet"]

get_input_file_junction_dict_master(tree_name)