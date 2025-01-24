import sys
import copy
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.basic import *

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16


contour_list = load_dict("results/path_planning/sample_rand_path")
norm_list = load_dict("results/path_planning/sample_rand_norms")
radii_list = load_dict("results/path_planning/sample_rand_radii")
# contour_list = load_dict("results/path_planning/sample_rand_path")
# norm_list = load_dict("results/path_planning/sample_rand_norms")
# radii_list = load_dict("results/path_planning/sample_rand_radii")
bif_pt_coords = load_dict("data/sample_bif/bif_pt_coords")
fig = plt.figure()
ax = plt.axes(projection='3d')
pdb.set_trace()
for contour in contour_list:
    ax.scatter3D(contour[0], contour[1], contour[2], label = f"Contour {contour}", marker = "o")
# ax.scatter3D(bif_pt_coords["inlet_coords"][0], bif_pt_coords["inlet_coords"][1], bif_pt_coords["inlet_coords"][2], label = "Inlet", marker = "X", s=100, color = "black")
# ax.scatter3D(bif_pt_coords["daughter1_coords"][0], bif_pt_coords["daughter1_coords"][1], bif_pt_coords["daughter1_coords"][2], label = "Daughter 1", marker = "X", s=100, color = "blue")
# ax.scatter3D(bif_pt_coords["daughter2_coords"][0], bif_pt_coords["daughter2_coords"][1], bif_pt_coords["daughter2_coords"][2], label = "Daughter 2", marker = "X", s=100, color = "red")
ax.set_title('Bifurcation Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.savefig("results/path_planning/sample_bifurcation_path.png", bbox_inches = 'tight')


contour1 = np.asarray(contour_list[0]).T
contour2 = np.asarray(contour_list[1]).T

radii1 = np.asarray(radii_list[0])
radii2 = np.asarray(radii_list[1])


norm1 = np.asarray(norm_list[0]).T/10000
norm2 = np.asarray(norm_list[1]).T/10000

intersection_pt_ind = np.argmin(np.linalg.norm(contour1 - contour2[0,:], axis = 1))
intersection_pt = contour1[intersection_pt_ind]

plt.clf()
plt.scatter(np.linspace(0,radii1.size, radii1.size), radii1, label = "Contour 1")
plt.scatter(np.linspace(intersection_pt_ind, intersection_pt_ind+radii2.size, radii2.size), radii2, label = "Contour 2")
plt.vlines(intersection_pt_ind, 0, 1, colors='r', linestyles='dashed', label = "Intersection Point")
plt.title("Sample Bifurcation Radii")
plt.legend()
plt.savefig("results/path_planning/sample_bifurcation_radii.png", bbox_inches = 'tight')



plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(1,1, 1, projection='3d')
for i in range(len(norm_list)):
    if i == 1:
        continue
    contour = np.asarray(contour_list[i])
    norm = np.asarray(norm_list[i])/5
    ax1.quiver(contour[0,:], contour[1,:], contour[2,:],\
              norm[0,:], norm[1,:], norm[2,:],\
              label = f"Contour {contour}", )


ax.set_title('Sample Bifurcation Norms')
fig.savefig("results/path_planning/sample_bifurcation_norms.png")



# plot original bifurcation
plt.clf()
contour_list = load_dict("results/path_planning/sample_path")
contour1 = np.asarray(contour_list[0]).T - intersection_pt
contour2 = np.asarray(contour_list[1]).T - intersection_pt

contour = contour1
ax.scatter3D(contour[:,0], contour[:,1], contour[:,2], label = f"Contour {contour}", marker = ".", color = "mediumblue")

contour = contour2
ax.scatter3D(contour[:,0], contour[:,1], contour[:,2], label = f"Contour {contour}", marker = ".", color = "firebrick")


ax.set_title('Sample Bifurcation Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.savefig("results/path_planning/sample_bifurcation_path_adjusted.png", bbox_inches = 'tight')


