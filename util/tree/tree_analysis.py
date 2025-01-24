import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *
from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

tree_name = "tree_80"
data_dict = load_dict(f"data/characteristic_value_dictionaries/{tree_name}_char_val_dict")

if not os.path.exists(f"results/CCO_hist_{tree_name}"):
    os.mkdir(f"results/CCO_hist_{tree_name}")   
bins = np.linspace(0, 2, 30)
plt.hist(data_dict["CCO"]["daughter1_angle"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(data_dict["CCO"]["daughter2_angle"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.vlines(np.pi/4, 0, 40, colors='r', linestyles='dashed', label = "45 degrees")
plt.title("Daughter Angles")
plt.legend(loc='upper right')
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_angles.png")

plt.clf()
bins = np.linspace(0, 2, 30)
plt.hist([data_dict["CCO"]["daughter1_angle"][i]+data_dict["CCO"]["daughter2_angle"][i] for i in range(len(data_dict["CCO"]["daughter1_angle"]))], bins, alpha=0.5, label='Primary Daughter')
plt.vlines(np.pi/4, 0, 40, colors='r', linestyles='dashed', label = "45 degrees")
plt.title("Daughter Angle Difference")
plt.legend(loc='upper right')
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_angle_diff.png")

plt.clf()
bins = np.linspace(0, 1.5, 30)
daughter1_area = np.pi*np.asarray(data_dict["CCO"]["daughter1_radius"])**2
daughter2_area = np.pi*np.asarray(data_dict["CCO"]["daughter2_radius"])**2
inlet_area = np.pi*np.asarray(data_dict["CCO"]["inlet_radius"])**2

Lc = np.asarray(data_dict["CCO"]["inlet_radius"])
Lc2 = np.pi * np.asarray(data_dict["CCO"]["inlet_radius"])**2
plt.hist(daughter1_area/Lc2, bins, alpha=0.5, label='Primary Daughter')
plt.hist(daughter2_area/Lc2, bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Normalized Daughter Area")
plt.legend(loc='upper right')
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_area_x.png")

plt.clf()
plt.hist(daughter2_area/daughter1_area, bins, alpha=0.5)
plt.title("Normalized Daughter Area")
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter12_area_ratio.png")

plt.clf()
bins = np.linspace(0, 0.6, 30)
plt.hist(data_dict["CCO"]["daughter1_radius"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(data_dict["CCO"]["daughter2_radius"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Daughter Area")
plt.legend(loc='upper right')
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_radii.png")

plt.clf()
inlet_rad = np.asarray(data_dict["CCO"]["inlet_radius"])
daughter1_rad = np.asarray(data_dict["CCO"]["daughter1_radius"])
daughter2_rad = np.asarray(data_dict["CCO"]["daughter2_radius"])
daughter2_rad_pred = (inlet_rad[inlet_rad > daughter1_rad]**3 - daughter1_rad[inlet_rad > daughter1_rad]**3)**(1/3)
daughter2_rad_pred_neg = -1*((-inlet_rad[inlet_rad <= daughter1_rad]**3 + daughter1_rad[inlet_rad <= daughter1_rad]**3))**(1/3)
plt.scatter(daughter2_rad[inlet_rad > daughter1_rad], daughter2_rad_pred)
plt.scatter(daughter2_rad[inlet_rad <= daughter1_rad], daughter2_rad_pred_neg)
plt.plot(daughter2_rad_pred, daughter2_rad_pred,'r', label= "Murray's Law")
plt.xlabel("Actual Daughter 2 Radius")
plt.ylabel("Predicted (Murray's Law) Daughter 2 Radius")
plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_radii_murray.png", bbox_inches='tight')

# plt.clf()
# bins = np.linspace(0, 2, 30)
# daughter1_vel = np.asarray(data_dict["CCO"]["daughter1_flow"])/daughter1_area
# daughter2_vel = np.asarray(data_dict["CCO"]["daughter2_flow"])/daughter2_area
# inlet_vel = np.pi * np.asarray(data_dict["CCO"]["inlet_flow"])**2
# Re = 1000
# Uc = 0.04*Re/(1.06*Lc)
# plt.hist(daughter1_vel/Uc, bins, alpha=0.5, label='Primary Daughter')
# plt.hist(daughter2_vel/Uc, bins, alpha=0.5, label='Auxilliary Daughter')
# plt.title("Normalized Daughter Velocity")
# plt.legend(loc='upper right')
# plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_velocities_x.png")

# plt.clf()

# inlet_vel = np.pi * np.asarray(data_dict["CCO"]["inlet_flow"])/inlet_area
# Re = 1.06 * inlet_vel * 2 *  data_dict["CCO"]["inlet_radius"]/ 0.04
# bins = np.linspace(0, np.max(Re), 30)
# plt.hist(Re, bins, alpha=0.5)
# plt.title("Normalized Daughter Re")

# plt.savefig(f"results/CCO_hist_{tree_name}/CCO_inlet_re.png")
# Re = 1000
# Uc = 0.04*Re/(1.06*Lc)

# plt.clf()
# bins = np.linspace(-1, 1.2, 30)
# daughter1_dp = np.asarray(data_dict["CCO"]["daughter1_dP"])
# daughter2_dp = np.asarray(data_dict["CCO"]["daughter2_dP"])
# plt.hist(daughter1_dp/(1.06*Uc**2), bins, alpha=0.5, label='Primary Daughter')
# plt.hist(daughter2_dp/(1.06*Uc**2), bins, alpha=0.5, label='Auxilliary Daughter')
# plt.title("Normalized Daughter Pressure Difference")
# plt.legend(loc='upper right')
# plt.savefig(f"results/CCO_hist_{tree_name}/CCO_daughter_pressure_x.png")

tree_data_dict = {}
tree_data_dict.update({"daughter1_area_ratio": daughter1_area/Lc2})
tree_data_dict.update({"daughter2_area_ratio": daughter2_area/Lc2})
tree_data_dict.update({"daughter12_area_ratio": daughter2_area/daughter1_area})
tree_data_dict.update({"daughter1_angle": data_dict["CCO"]["daughter1_angle"]})
tree_data_dict.update({"daughter2_angle": data_dict["CCO"]["daughter2_angle"]})
tree_data_dict.update({"daughter12_angle_diff": [data_dict["CCO"]["daughter1_angle"][i]+data_dict["CCO"]["daughter2_angle"][i] for i in range(len(data_dict["CCO"]["daughter1_angle"]))]})

for param in tree_data_dict.keys():
    mean, std = norm.fit(tree_data_dict[param])
    tree_data_dict[param] = {"full": tree_data_dict[param],
                             "lowest": np.min(tree_data_dict[param]),
                             "range": np.max(tree_data_dict[param]) - np.min(tree_data_dict[param]),
                             "mean": mean,
                             "std": std}
save_dict(tree_data_dict, f"data/CCO_tree/{tree_name}_data_dict")