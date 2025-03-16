import sys
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.tools.junction_proc import *
from util.tools.vtk_functions import *
import matplotlib.pyplot as plt

def plot_centerline_values(tree_name):
    """
    Compile list of junction graphs (from synthetic data)
    """
    #soln_dir = f"data/synthetic_junctions/CCO_80/mesh_convergence_4/{tree_name}/centerline_sol.vtp"
    soln_dir = f"data/synthetic_junctions/angles_CCO/random/{tree_name}/centerline_sol.vtp"
    reader_1d = read_geo(soln_dir).GetOutput()

    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output

    arrays = get_all_arrays(reader_1d)[0]
    num_branches = np.max(arrays["BranchId"]) + 1
    branch_dict = {}
    for branch_id in range(num_branches):
        branch_dict[branch_id] = {}
        branch_dict[branch_id]["path"] = arrays["Path"][arrays["BranchId"] == branch_id]
        path_order = np.argsort(branch_dict[branch_id]["path"])
        branch_dict[branch_id]["path"] = branch_dict[branch_id]["path"][path_order]
        if branch_id > 0:
            branch_dict[branch_id]["path"] = branch_dict[branch_id]["path"] + branch_dict[0]["path"][-1]
        branch_dict[branch_id]["area"] = arrays["area"][arrays["BranchId"] == branch_id][path_order]
        branch_dict[branch_id]["flow"] = arrays["Velocity"][arrays["BranchId"] == branch_id][path_order]
        branch_dict[branch_id]["pressure"] = arrays["Pressure"][arrays["BranchId"] == branch_id][path_order]/1333.22
        branch_dict[branch_id]["energy"] = arrays["Energy"][arrays["BranchId"] == branch_id][path_order]/1333.22
        branch_dict[branch_id]["total_pressure"] = arrays["Energy"][arrays["BranchId"] == branch_id][path_order]/1333.22 + arrays["Pressure"][arrays["BranchId"] == branch_id][path_order]/1333.22
    offset = 20
    colors = ["red", "blue", "green"]
    fig, ax = plt.subplots(5, 1, figsize=(10, 12))
    min_branch_length = np.min([branch_dict[1]["path"].size, branch_dict[2]["path"].size])
    for branch_id in range(num_branches):
        ax[0].plot(branch_dict[branch_id]["path"], branch_dict[branch_id]["area"], color = colors[branch_id], label=f"Branch {branch_id}")
        ax[1].plot(branch_dict[branch_id]["path"], branch_dict[branch_id]["flow"], colors[branch_id], label=f"Branch {branch_id}")
        ax[2].plot(branch_dict[branch_id]["path"], branch_dict[branch_id]["pressure"], colors[branch_id], label=f"Branch {branch_id}")
        ax[3].plot(branch_dict[branch_id]["path"], ((branch_dict[branch_id]["pressure"]+branch_dict[branch_id]["energy"])* \
                                                    branch_dict[branch_id]["flow"][10]), colors[branch_id], label=f"Branch {branch_id}")
        ax[4].plot(branch_dict[branch_id]["path"], branch_dict[branch_id]["energy"], colors[branch_id], label=f"Branch {branch_id}")
        if branch_id > 0:
            ax[1].vlines(branch_dict[branch_id]["path"][offset], np.min(branch_dict[2]["flow"]), np.max(branch_dict[0]["flow"]), color = colors[branch_id])
            ax[2].vlines(branch_dict[branch_id]["path"][offset], np.min(branch_dict[2]["pressure"]), np.max(branch_dict[0]["pressure"]), color = colors[branch_id])

    ax[1].plot(branch_dict[2]["path"][0:min_branch_length], branch_dict[2]["flow"][0:min_branch_length]+branch_dict[1]["flow"][0:min_branch_length], "--", color="black", label="Total Flow")
    # ax[3].plot(branch_dict[0]["path"], 
    #         branch_dict[1]["total_pressure"][0:min_branch_length]*branch_dict[1]["flow"][0:min_branch_length]/branch_dict[0]["flow"][10], 
    #         "--", color="black")
    ax[3].plot(branch_dict[2]["path"][0:min_branch_length], 
               branch_dict[1]["total_pressure"][0:min_branch_length]*branch_dict[1]["flow"][0:min_branch_length] +
               branch_dict[2]["total_pressure"][0:min_branch_length]*branch_dict[2]["flow"][0:min_branch_length], 
               "--", color="black", label="Total Pressure Weighted by Flow")

    ax[0].set_title(f"Centerline Values for {tree_name}")
    ax[0].set_ylabel("Area (cm^2)")
    ax[1].set_ylabel("Flow (cm^3/s)")
    ax[2].set_ylabel("Pressure (mmHg)")
    ax[3].set_ylabel("Total Pressure (mmHg)")
    ax[4].set_ylabel("Dynamic Pressure (mmHg)")
    ax[1].legend()
    ax[2].legend()

    if not os.path.exists(f"results/bifurcation_analysis"):
        os.mkdir(f"results/bifurcation_analysis")
    fig.savefig(f"results/bifurcation_analysis/{tree_name}_centerline_values.png")
    return



plot_centerline_values("CCO_021")
# plot_centerline_values("CCO_020_32706")
# plot_centerline_values("CCO_020_48766")