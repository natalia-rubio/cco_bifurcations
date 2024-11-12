import sys
sys.path.append("/Users/natalia/Desktop/CCO_junctions")
from util.tools.basic import *

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16

data_dict = load_dict("data/CCO_tree/char_val_dict")

bins = np.linspace(0, 2, 30)
plt.hist(data_dict["CCO"]["daughter1_angle"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(data_dict["CCO"]["daughter2_angle"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.vlines(np.pi/4, 0, 100, colors='r', linestyles='dashed', label = "45 degrees")
plt.ylim(0, 7)
plt.title("Daughter Angles")
plt.legend(loc='upper right')
plt.savefig("results/CCO_hist/CCO_daughter_angles.png")

plt.clf()
bins = np.linspace(0, 1.5, 30)
daughter1_area = np.pi*np.asarray(data_dict["CCO"]["daughter1_radius"])**2
daughter2_area = np.pi*np.asarray(data_dict["CCO"]["daughter2_radius"])**2
Lc = np.asarray(data_dict["CCO"]["inlet_radius"])
Lc2 = np.pi * np.asarray(data_dict["CCO"]["inlet_radius"])**2
plt.hist(daughter1_area/Lc2, bins, alpha=0.5, label='Primary Daughter')
plt.hist(daughter2_area/Lc2, bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Normalized Daughter Area")
plt.legend(loc='upper right')
plt.savefig("results/CCO_hist/CCO_daughter_area_x.png")

plt.clf()
bins = np.linspace(0, 0.6, 30)
plt.hist(data_dict["CCO"]["daughter1_radius"], bins, alpha=0.5, label='Primary Daughter')
plt.hist(data_dict["CCO"]["daughter2_radius"], bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Daughter Area")
plt.legend(loc='upper right')
plt.savefig("results/CCO_hist/CCO_daughter_radii.png")

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
plt.savefig("results/CCO_hist/CCO_daughter_radii_murray.png", bbox_inches='tight')

plt.clf()
bins = np.linspace(0, 2, 30)
daughter1_vel = np.asarray(data_dict["CCO"]["daughter1_flow"])/daughter1_area
daughter2_vel = np.asarray(data_dict["CCO"]["daughter2_flow"])/daughter2_area
inlet_vel = np.pi * np.asarray(data_dict["CCO"]["inlet_flow"])**2
Re = 1000
Uc = 0.04*Re/(1.06*Lc)
plt.hist(daughter1_vel/Uc, bins, alpha=0.5, label='Primary Daughter')
plt.hist(daughter2_vel/Uc, bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Normalized Daughter Velocity")
plt.legend(loc='upper right')
plt.savefig("results/CCO_hist/CCO_daughter_velocities_x.png")

plt.clf()
bins = np.linspace(-1, 1.2, 30)
daughter1_dp = np.asarray(data_dict["CCO"]["daughter1_dP"])
daughter2_dp = np.asarray(data_dict["CCO"]["daughter2_dP"])
plt.hist(daughter1_dp/(1.06*Uc**2), bins, alpha=0.5, label='Primary Daughter')
plt.hist(daughter2_dp/(1.06*Uc**2), bins, alpha=0.5, label='Auxilliary Daughter')
plt.title("Normalized Daughter Pressure Difference")
plt.legend(loc='upper right')
plt.savefig("results/CCO_hist/CCO_daughter_pressure_x.png")

tree_data_dict = {}
tree_data_dict.update({"daughter1_area_ratio": daughter1_area/Lc2})
tree_data_dict.update({"daughter2_area_ratio": daughter2_area/Lc2})
tree_data_dict.update({"daughter1_angle": data_dict["CCO"]["daughter1_angle"]})
tree_data_dict.update({"daughter2_angle": data_dict["CCO"]["daughter2_angle"]})

for param in tree_data_dict.keys():
    tree_data_dict[param] = {"full": tree_data_dict[param],
                             "lowest": np.min(tree_data_dict[param]),
                             "range": np.max(tree_data_dict[param]) - np.min(tree_data_dict[param])}
save_dict(tree_data_dict, "data/CCO_tree/tree_data_dict")