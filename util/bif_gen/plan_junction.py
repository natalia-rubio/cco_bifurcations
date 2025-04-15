import sys
sys.path.append('/Users/natalia/Desktop/dec_tree_gen')
import numpy as np
import pickle
# import matplotlib.pyplot as plt
#from util.tools.basic import save_dict, load_dict

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def plan_junction(geo_params):
    pts_per_branch = 4
    #tree_1
    branch_dict = {"branch1": {"branch_radii": [(1/3.14)**0.5, (geo_params["daughter1_area_ratio"]/3.14)**0.5,], 
                               "angles": [90, 90 + geo_params["daughter1_angle"]*180/np.pi], 
                               "connection_branch":None, "connection_seg":None},
                "branch2": {"branch_radii": [((geo_params["total_daughter_area_ratio"] - geo_params["daughter1_area_ratio"])/3.14)**0.5,],
                            "angles": [90 - geo_params["daughter2_angle"]*180/np.pi,], 
                            "connection_branch":"branch1", "connection_seg":0},
                }

    for branch_name in ["branch1", "branch2"]:
        #print(branch_dict)
        starts = []
        ends = []
        locs = []
        norms = []
        radii = []

        connection_branch = branch_dict[branch_name]["connection_branch"]
        connection_point = branch_dict[branch_name]["connection_seg"]
        branch_radii = branch_dict[branch_name]["branch_radii"]
        angles = branch_dict[branch_name]["angles"]
        num_segs = len(branch_radii)
        if connection_branch is None:
                connection_loc = (0, 0, 0)
        else:
            connection_loc = branch_dict[connection_branch]["ends"][connection_point]
            #connection_angle = branch_dict[connection_branch]["angles"][connection_point]
        for i in range(num_segs):
            
            if i == 0:
                starts.append(connection_loc)
                print("connection_loc", connection_loc)
            else:
                starts.append(ends[i-1])
            if branch_name == "branch1" and i == 0:
                length = np.sqrt(branch_radii[i]) * 10
                pts_per_branch = 4
            else:
                length = np.sqrt(branch_radii[i]) * 20
                pts_per_branch = 10
            end = (starts[i][0] + length * np.cos(np.radians(angles[i])),
                    starts[i][1] + length * np.sin(np.radians(angles[i])),
                    0)
            ends.append(end)

            for j in range(pts_per_branch):

                locs.append([starts[i][0] + j / pts_per_branch * (end[0] - starts[i][0]),
                            starts[i][1] + j / pts_per_branch * (end[1] - starts[i][1]),
                            starts[i][2] + j / pts_per_branch * (end[2] - starts[i][2])])
                norms.append([end[0] - starts[i][0], end[1] - starts[i][1], end[2] - starts[i][2]])
                if j == 0 and i > 0:
                    radii.append(float(branch_radii[i-1]))
                else:
                    radii.append(float(branch_radii[i]))

                # plt.scatter(locs[-1][0], locs[-1][1], c='r')
        branch_dict[branch_name]["ends"] = ends
        branch_dict[branch_name]["starts"] = starts
        branch_dict[branch_name]["locs"] = locs
        branch_dict[branch_name]["norms"] = norms
        branch_dict[branch_name]["radii"] = radii
        
    # plt.axis('equal')
    # plt.savefig(geo_params["geo_dir"]+'/tree_path.png')
    save_dict(branch_dict, geo_params["geo_dir"]+'/tree_dict.pkl')

        
        