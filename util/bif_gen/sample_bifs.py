import sys
import copy
import os
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from scipy.stats import qmc, uniform, norm
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

anatomy = "tree_20"
num_geos = 300
sampler = qmc.LatinHypercube(d=6,seed = 1)
samples = sampler.random(n=num_geos*2)

samples_uniform = uniform(loc=0, scale=1).ppf(samples)
samples_normal = norm(loc=0, scale=1).ppf(samples)

stats_dict = load_dict(f"trees/reports/tree_20/junction_statistics")


CCO_sampled_params_dict =   {"daughter1_angle": [],
                                "daughter2_angle": [],
                                "daughter1_area_ratio": [],
                                "daughter2_area_ratio": [],
                                "flow_split": [],
                                }
success_counter = 0
i = 0
while success_counter < num_geos:
    area_consistency = False
    while not area_consistency:
        for param_ind, param in enumerate(CCO_sampled_params_dict.keys()):
                CCO_sampled_params_dict[param].append(
                stats_dict[param]["min"]*0.8 + samples_uniform[i, param_ind] * 1.2*(stats_dict[param]["max"] - stats_dict[param]["min"]))

        d1ar = CCO_sampled_params_dict["daughter1_area_ratio"][-1]
        d2ar = CCO_sampled_params_dict["daughter2_area_ratio"][-1]
        print(d1ar, d2ar)
        if d1ar+d2ar < 1.8 and d2ar > 0.1:
            area_consistency = True
        else:
            for param_ind, param in enumerate(CCO_sampled_params_dict.keys()):
                CCO_sampled_params_dict[param].pop(-1)
            area_consistency = False
        i += 1
    success_counter += 1
        
if not os.path.exists("data/sampled_params_dict"):
    os.mkdir("data/sampled_params_dict")
save_dict(CCO_sampled_params_dict, f"data/sampled_params_dict/{anatomy}_sampled_params_dict_random2")

if not os.path.exists("results/sampled_geo"):
    os.mkdir("results/sampled_geo")

num_bins = 50
pdb.set_trace()

for param in CCO_sampled_params_dict.keys():
    plt.clf()
    bins = np.linspace(stats_dict[param]["min"]*0.7, stats_dict[param]["max"]*1.3, num_bins)
    plt.hist(CCO_sampled_params_dict[param], bins, color="lightskyblue")
    plt.vlines(stats_dict[param]["min"], 0, 100, colors='black', linestyles='dashed', label = "min")
    plt.vlines(stats_dict[param]["max"], 0, 100, colors='black', linestyles='dashed', label = "highest")
    plt.vlines(stats_dict[param]["mean"], 0, 100, colors='red', linestyles='dashed', label = "mean")
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
