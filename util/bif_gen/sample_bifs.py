import sys
import copy
sys.path.append("/Users/natalia/Desktop/CCO_junctions")
from util.tools.basic import *
from scipy.stats import qmc
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16


sampler = qmc.LatinHypercube(d=4,seed = 0)
samples = sampler.random(n=100)
tree_data_dict = load_dict("data/CCO_tree/tree_data_dict")
CCO_sampled_params_dict = {"daughter1_angle": [],
                           "daughter2_angle": [],
                           "daughter1_area_ratio": [],
                           "daughter2_area_ratio": []}

for i in range(100):
    for param_ind, param in enumerate(CCO_sampled_params_dict.keys()):

        CCO_sampled_params_dict[param].append(0.9 * tree_data_dict[param]["lowest"] + \
            samples[i, param_ind]*tree_data_dict[param]["range"] * 1.2)

save_dict(CCO_sampled_params_dict, "data/CCO_tree/CCO_sampled_params_dict")

if not os.path.exists("results/sampled_geo"):
    os.mkdir("results/sampled_geo")
bins = np.linspace(0, 2, 30)
plt.hist(CCO_sampled_params_dict["daughter1_angle"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(CCO_sampled_params_dict["daughter2_angle"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.vlines(np.pi/4, 0, 100, colors='r', linestyles='dashed', label = "45 degrees")
plt.ylim(0, 7)
plt.title("Daughter Angles")
plt.legend(loc='upper right')
plt.savefig("results/sampled_geo/CCO_daughter_angles.png", bbox_inches = 'tight')

plt.clf()
bins = np.linspace(0, 1.5, 30)
plt.hist(CCO_sampled_params_dict["daughter1_area_ratio"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(CCO_sampled_params_dict["daughter2_area_ratio"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Normalized Daughter Area")
plt.legend(loc='upper right')
plt.savefig("results/sampled_geo/CCO_daughter_area_x.png", bbox_inches = 'tight')
