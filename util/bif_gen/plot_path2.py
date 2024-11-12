import sys
import copy
sys.path.append("/Users/natalia/Desktop/CCO_junctions")
from util.tools.basic import *

from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16


contour_list = load_dict("results/path_planning/sample_path")
norm_list = load_dict("results/path_planning/sample_norms")
radii_list = load_dict("results/path_planning/sample_radii")
# contour_list = load_dict("results/path_planning/sample_rand_path")
# norm_list = load_dict("results/path_planning/sample_rand_norms")
# radii_list = load_dict("results/path_planning/sample_rand_radii")
bif_pt_coords = load_dict("data/sample_bif/bif_pt_coords")
fig = plt.figure()
ax = plt.axes(projection='3d')

for contour in contour_list:
    ax.scatter3D(contour[0], contour[1], contour[2], label = f"Contour {contour}", marker = "o")
ax.scatter3D(bif_pt_coords["inlet_coords"][0], bif_pt_coords["inlet_coords"][1], bif_pt_coords["inlet_coords"][2], label = "Inlet", marker = "X", s=100, color = "black")
ax.scatter3D(bif_pt_coords["daughter1_coords"][0], bif_pt_coords["daughter1_coords"][1], bif_pt_coords["daughter1_coords"][2], label = "Daughter 1", marker = "X", s=100, color = "blue")
ax.scatter3D(bif_pt_coords["daughter2_coords"][0], bif_pt_coords["daughter2_coords"][1], bif_pt_coords["daughter2_coords"][2], label = "Daughter 2", marker = "X", s=100, color = "red")
ax.set_title('Sample Bifurcation Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.savefig("results/path_planning/sample_bifurcation_path.png", bbox_inches = 'tight')


contour1 = np.asarray(contour_list[0]).T
contour2 = np.asarray(contour_list[1]).T

radii1 = np.asarray(radii_list[0])
radii2 = np.asarray(radii_list[1])


norm1 = np.asarray(norm_list[0]).T
norm2 = np.asarray(norm_list[1]).T

intersection_pt_ind = np.argmin(np.linalg.norm(contour1 - contour2[0,:], axis = 1))
intersection_pt = contour1[intersection_pt_ind]

plt.clf()
plt.plot(np.linspace(0,radii1.size, radii1.size), radii1, label = "Contour 1")
plt.plot(np.linspace(intersection_pt_ind,intersection_pt_ind+radii2.size, radii2.size), radii2, label = "Contour 2")
plt.vlines(intersection_pt_ind, 0, 1, colors='r', linestyles='dashed', label = "Intersection Point")
plt.title("Sample Bifurcation Radii")
plt.legend()
plt.savefig("results/path_planning/sample_bifurcation_radii.png", bbox_inches = 'tight')

plt.clf()
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
for i in range(len(norm_list)):
    contour = contour_list[i]
    norm = norm_list[i]
    ax1.quiver(contour[0], contour[1], contour[2],\
              norm[0], norm[1], norm[2],\
              label = f"Contour {contour}", )

ax2 = fig.add_subplot(2, 2, 2, projection='3d')
ax2.quiver(contour1[0:intersection_pt_ind, 0],
            contour1[0:intersection_pt_ind, 1],
            contour1[0:intersection_pt_ind, 2],
            norm1[0:intersection_pt_ind, 0],
            norm1[0:intersection_pt_ind, 1],
            norm1[0:intersection_pt_ind, 2],
            )

ax3 = fig.add_subplot(2, 2, 3, projection='3d')
ax3.quiver(contour1[intersection_pt_ind:, 0],
            contour1[intersection_pt_ind:, 1],
            contour1[intersection_pt_ind:, 2],
            norm1[intersection_pt_ind:, 0],
            norm1[intersection_pt_ind:, 1],
            norm1[intersection_pt_ind:, 2],
            )
ax4 = fig.add_subplot(2, 2, 4, projection='3d')
ax4.quiver(contour2[:, 0],
            contour2[:, 1],
            contour2[:, 2],
            norm2[:, 0],
            norm2[:, 1],
            norm2[:, 2],
            )

ax.set_title('Sample Bifurcation Norms')
fig.savefig("results/path_planning/sample_bifurcation_norms.png", bbox_inches = 'tight')

# Centering the intersection point
contour1 = contour1 - intersection_pt
contour2 = contour2 - intersection_pt

geo_params = {'daughter2_area_ratio': 1.075819054568883, 
              'daughter2_angle': 1.655705730146607, 
              'daughter1_area_ratio': 0.9125293563339761, 
              'daughter1_angle': 0.16916611144823895}

inlet_ref1 = np.inner(-1*contour1[0,:],contour1[-1,:])/np.linalg.norm(contour1[0,:])* (-1*contour1[0,:] / np.linalg.norm(contour1[0,:]))
inlet_ref_length1 = np.linalg.norm(inlet_ref1)
seg1 = contour1[-1,:] - inlet_ref1
seg_length1 = np.linalg.norm(seg1)
angle1 = np.arcsin(seg_length1/np.linalg.norm(contour1[-1,:]))
new_angle1 = geo_params["daughter1_angle"]
if new_angle1 < np.pi/2:
    new_seg_length1 = np.linalg.norm(contour1[-1,:])*np.sin(new_angle1)
    seg_add1 = (new_seg_length1 - seg_length1) * seg1/np.linalg.norm(seg1)
else:
    rev_length = seg_length1/np.tan(np.pi - new_angle1)
    seg_add1 = (inlet_ref_length1 + rev_length) * contour1[0,:]/np.linalg.norm(contour1[0,:])
incs = np.linspace(0, 1, contour1.shape[0]-intersection_pt_ind , endpoint = True)
for point in range(contour1.shape[0]-intersection_pt_ind):
    contour1[point+intersection_pt_ind,:] = contour1[point+intersection_pt_ind,:] + incs[point] * seg_add1

# Adjust angle 2
inlet_ref2 = (np.inner(-1*contour1[0,:],contour2[-1,:])/np.linalg.norm(contour1[0,:])) * (-1*contour1[0,:] / np.linalg.norm(contour1[0,:]))
inlet_ref_length2 = np.linalg.norm(inlet_ref2)
seg2 = contour2[-1,:] - inlet_ref2
seg_length2 = np.linalg.norm(seg2)
angle2 = np.arcsin(seg_length2/np.linalg.norm(contour2[-1,:]))
new_angle2 = angle2 + np.pi/2
if new_angle2 < np.pi/2:
    new_seg_length2 = np.linalg.norm(contour2[-1,:])*np.sin(new_angle2)
    seg_add2 = (new_seg_length2 - seg_length2) * seg2/np.linalg.norm(seg2)
else:
    rev_length = seg_length2/np.tan(np.pi - new_angle2)
    seg_add2 = (inlet_ref_length2 + rev_length) * contour1[0,:]/np.linalg.norm(contour1[0,:])
incs = np.linspace(0, 1, contour2.shape[0] , endpoint = True)
for point in range(contour2.shape[0]):
    contour2[point,:] = contour2[point,:] + incs[point] * seg_add2


fig = plt.figure()
ax = plt.axes(projection='3d')
contour_list = [contour1, contour2]
contour = contour_list[0]
ax.scatter3D(contour[:,0], contour[:,1], contour[:,2], label = f"Contour {contour}", marker = "o", color = "skyblue")
contour = contour_list[1]
ax.scatter3D(contour[:,0], contour[:,1], contour[:,2], label = f"Contour {contour}", marker = "o", color= "lightcoral")
ax.scatter3D(seg_add2[0], seg_add2[1], seg_add2[2], label = f"Contour {contour}", marker = "o", color= "lightcoral")
ax.scatter3D(inlet_ref2[0], inlet_ref2[1], inlet_ref2[2], label = f"Contour {contour}", marker = ".", color = "mediumblue")

# plot original bifurcation
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


