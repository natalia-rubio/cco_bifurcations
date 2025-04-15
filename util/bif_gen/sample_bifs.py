import sys
import copy
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from scipy.stats import qmc, uniform, norm
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

anatomy = "ideal_fix_areas"
num_geos = 400
sampler = qmc.LatinHypercube(d=4,seed = 0)
samples = sampler.random(n=num_geos*10)

samples_uniform = uniform(loc=0, scale=1).ppf(samples)
samples_normal = norm(loc=0, scale=1).ppf(samples)


tree_data_dict =    {"daughter1_angle": {"lowest": 0.0, "highest": np.pi, "mean": np.pi/4, "range": np.pi/2},
                    "daughter2_angle": {"lowest": 0.0, "highest": np.pi, "mean": np.pi/4, "range": np.pi/2},
                    "total_daughter_area_ratio": {"lowest": 0.7, "highest": 1.5, "mean": 1.1, "range": 0.8},
                    "daughter1_area_ratio": {"lowest": 0.2, "highest": 1, "mean": 0.8, "range": 1.1}}
CCO_sampled_params_dict =   {"daughter1_angle": [],
                            "daughter2_angle": [],
                            "total_daughter_area_ratio": [],
                            "daughter1_area_ratio": []}
success_counter = 0
i = 0
while success_counter < num_geos:
    area_consistency = False
    while not area_consistency:
        for param_ind, param in enumerate(CCO_sampled_params_dict.keys()):
                CCO_sampled_params_dict[param].append(
                tree_data_dict[param]["lowest"] + samples_uniform[i, param_ind] * tree_data_dict[param]["range"])
        d1ar = CCO_sampled_params_dict["daughter1_area_ratio"][-1]
        d2ar = CCO_sampled_params_dict["total_daughter_area_ratio"][-1] - d1ar
        print(d1ar, d2ar)
        if d1ar < 1 and d2ar < 1 and d2ar > 0.2:
            area_consistency = True
        else:
            for param_ind, param in enumerate(CCO_sampled_params_dict.keys()):
                CCO_sampled_params_dict[param].pop(-1)
            area_consistency = False
        i += 1
    success_counter += 1
        
if not os.path.exists("data/sampled_params_dict"):
    os.mkdir("data/sampled_params_dict")
save_dict(CCO_sampled_params_dict, f"data/sampled_params_dict/{anatomy}_sampled_params_dict")

if not os.path.exists("results/sampled_geo"):
    os.mkdir("results/sampled_geo")

num_bins = 50

for param in CCO_sampled_params_dict.keys():
    plt.clf()
    bins = np.linspace(tree_data_dict[param]["lowest"]*0.6, tree_data_dict[param]["lowest"]+tree_data_dict[param]["range"]*1.4, num_bins)
    plt.hist(CCO_sampled_params_dict[param], bins, color="lightskyblue")
    plt.vlines(tree_data_dict[param]["lowest"], 0, 100, colors='black', linestyles='dashed', label = "lowest")
    plt.vlines(tree_data_dict[param]["lowest"]+tree_data_dict[param]["range"], 0, 100, colors='black', linestyles='dashed', label = "highest")
    plt.vlines(tree_data_dict[param]["mean"], 0, 100, colors='red', linestyles='dashed', label = "mean")
    plt.ylim(0, 4*num_geos/num_bins)
    plt.xlabel(param)
    plt.title(f"{param} distribution")
    plt.savefig(f"results/sampled_geo/{anatomy}_{param}.png", bbox_inches = 'tight')


# bins = np.linspace(0, 2, num_bins)
# plt.hist(CCO_sampled_params_dict["daughter1_angle"], bins, alpha=0.5, label='Primary Daughter')
# #plt.hist(CCO_sampled_params_dict["daughter2_angle"], bins, alpha=0.5, label='Auxilliary Daughter')
# plt.vlines(np.pi/4, 0, 100, colors='r', linestyles='dashed', label = "45 degrees")
# plt.vlines(np.pi/4, 0, 100, colors='r', linestyles='dashed', label = "45 degrees")
# plt.ylim(0, num_geos/num_bins)
# plt.title("Daughter Angles")
# plt.legend(loc='upper right')
# plt.savefig("results/sampled_geo/CCO_daughter_angles.png", bbox_inches = 'tight')


# plt.clf()
# bins = np.linspace(0, 1.5, 30)
# plt.hist(CCO_sampled_params_dict["daughter1_area_ratio"], bins, alpha=0.5, label='Primary Daughter')
# #plt.hist(CCO_sampled_params_dict["daughter2_area_ratio"], bins, alpha=0.5, label='Auxilliary Daughter')
# plt.title("Normalized Daughter Area")
# plt.legend(loc='upper right')
# plt.savefig("results/sampled_geo/CCO_daughter_area_x.png", bbox_inches = 'tight')

# plt.clf()
# plt.hist(CCO_sampled_params_dict["daughter12_area_ratio"], bins, alpha=0.5)
# plt.title("Normalized Daughter Area")
# plt.savefig(f"results/sampled_geo/CCO_daughter12_area_ratio.png")

# plt.clf()
# bins = np.linspace(0, 2, 30)
# plt.hist(CCO_sampled_params_dict["daughter12_angle_diff"], bins, alpha=0.5, label='Primary Daughter')
# plt.vlines(np.pi/4, 0, 40, colors='r', linestyles='dashed', label = "45 degrees")
# plt.title("Daughter Angle Difference")
# plt.legend(loc='upper right')
# plt.savefig(f"results/sampled_geo/CCO_daughter_angle_diff.png")
