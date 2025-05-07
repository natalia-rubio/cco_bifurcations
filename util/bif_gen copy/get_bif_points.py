import sys
sys.path.append("/Users/natalia/Desktop/CCO_junctions")
from util.tools.junction_proc import *

cent = read_centerline('data/sample_bif/centerlines.vtp').GetOutput()
cent_array = get_all_arrays(cent)

#Extract Geometry ----------------------------------------------------
pt_id = cent_array["GlobalNodeId"].astype(int)
num_pts = np.size(pt_id)  # number of points in mesh
branch_id = cent_array["BranchId"].astype(int)
junction_id = cent_array["BifurcationId"].astype(int)
axial_distance = cent_array["Path"]  # distance along centerline
area = cent_array["CenterlineSectionArea"]
direction = cent_array["CenterlineSectionNormal"]  # vector normal direction
direction_norm = np.linalg.norm( direction, axis=1, keepdims=True)  # norm of direction vector
direction = np.transpose(np.divide(direction,direction_norm))  # normalized direction vector
angle1 = direction[0,:].reshape(-1,)
angle2 = direction[1,:].reshape(-1,)
angle3 = direction[2,:].reshape(-1,)

points = v2n(read_centerline('data/sample_bif/centerlines.vtp').GetOutput().GetPoints().GetData())
inlet_id = np.max(pt_id[np.argwhere(branch_id == 0)])
daughter1_id = np.min(pt_id[np.argwhere(branch_id == 1)])
daughter2_id = np.min(pt_id[np.argwhere(branch_id == 2)])

inlet_ind = np.argwhere(pt_id == inlet_id)
daughter1_ind = np.argwhere(pt_id == daughter1_id)
daughter2_ind = np.argwhere(pt_id == daughter2_id)
bif_coord_dict = {"inlet_coords": points[inlet_ind,:][0][0], 
        "daughter1_coords": points[daughter1_ind,:][0][0],
        "daughter2_coords": points[daughter2_ind,:][0][0],
           }
save_dict(bif_coord_dict, "data/sample_bif/bif_pt_coords")
pdb.set_trace()
