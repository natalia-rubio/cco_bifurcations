import pdb
import sys
import os
import vtk
import re
import numpy as np
import pandas as pd
import pysvzerod
from collections import defaultdict
from scipy.interpolate import interp1d
from vtk.util.numpy_support import numpy_to_vtk
sys.path.append("/Users/natalia/Desktop/cco_bifurcations")
from util.centerline_projection.reader.centerline_handler import CenterlineHandler
from util.centerline_projection.reader.svzerodsolver_input_handler import SvZeroDSolverInputHandler
from typing import Any
def rec_dd() -> defaultdict:
    return defaultdict(rec_dd)

def convert_csv_to_branch_result(
    df: pd.DataFrame, zerod_handler: SvZeroDSolverInputHandler
) -> dict:
    # loop branches and segments
    names = list(sorted(set(df["name"])))
    out: dict[str, Any] = {"flow": {}, "pressure": {}, "distance": {}}

    for name in names:
        # extract ids
        br, seg = [int(s) for s in re.findall(r"\d+", name)]

        # add 0d results
        for field in ["flow", "pressure"]:
            if seg == 0:
                out[field][br] = [
                    list(df[df.name == name][field + "_in"])[-1:]
                ]
            out[field][br] += [
                list(df[df.name == name][field + "_out"])[-1:]
            ]
            #pdb.set_trace()
        out["time"] = list(df[df.name == name]["time"])[-1:]
        

        # add path distance
        for vessel in zerod_handler.data["vessels"]:
            if vessel["vessel_name"] == name:
                if seg == 0:
                    out["distance"][br] = [0]
                l_new = (
                    out["distance"][br][-1] + vessel["vessel_length"]
                )
                out["distance"][br] += [l_new]

    # convert to numpy
    for field in ["flow", "pressure", "distance"]:
        for br in out[field].keys():
            out[field][br] = np.array(out[field][br])
    out["time"] = np.array(out["time"])

    # save to file
    return out

def project_to_centerline(tree_name, junction_mode):
    
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])

    centerline_handler = CenterlineHandler.from_file(f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/centerlines/centerlines.vtp")
    input_file = f"trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json"
    zerod_handler = SvZeroDSolverInputHandler.from_file(input_file)
    casadi = False
    #print(f"Input file: {input_file}")
    
    zerod_handler.update_simparams(last_cycle_only=True)
    #print("in project_to_centerline: got handler")
    casadi = True
    if junction_mode == "standard" or junction_mode == "RI":
        zerod_solver = pysvzerod.Solver(input_file)
        #print("in project_to_centerline: got solver")
        zerod_solver.run()
        #print("in project_to_centerline: ran solver")
        results_df = zerod_solver.get_full_result()
        os.makedirs(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}", exist_ok=True)
        results_df.to_csv(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}/results.csv")
    # 
    else:
        if casadi:
            results_df = pd.read_csv(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}/sol_casadi.csv")
        else:
            results_df = pd.read_csv(f"trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}/results.csv")
    #print("in project_to_centerline: loaded results")
    branch_results = convert_csv_to_branch_result(results_df, zerod_handler)
    arrays = rec_dd()

    points = centerline_handler.points
    branch_ids = centerline_handler.get_point_data_array("BranchId")
    path = centerline_handler.get_point_data_array("Path")
    cl_id = centerline_handler.get_point_data_array("CenterlineId")
    bif_id = centerline_handler.get_point_data_array("BifurcationId")

    # all branch ids in centerline
    ids_cent = np.unique(branch_ids).tolist()
    ids_cent.remove(-1)


    # loop all result fields
    for f in ["flow", "pressure"]:
        if f not in branch_results:
            continue

        # check if ROM branch has same ids as centerline
        ids_rom = list(branch_results[f].keys())
        ids_rom.sort()
        assert (
            all(x in ids_rom for x in ids_cent)
        ), "Centerline and ROM branch_results have different branch ids"

        # initialize output arrays
        array_f = np.zeros((path.shape[0], len(branch_results["time"])))
        n_outlet = np.zeros(path.shape[0])

        # loop all branches
        for br in branch_results[f].keys():
            if br not in ids_cent:
                continue
            # branch_results of this branch
            res_br = branch_results[f][br]

            # get centerline path
            path_cent = path[branch_ids == br]

            # get node locations from 0D branch_results
            path_1d_res = branch_results["distance"][br]
            f_res = res_br
            #pdb.set_trace()

            # interpolate ROM onto centerline
            # limit to interval [0,1] to avoid extrapolation error interp1d
            # due to slightly incompatible lenghts
            try:
                f_cent = interp1d(path_1d_res / path_1d_res[-1], f_res.T)(
                    path_cent / np.max(path_cent)
                ).T
            except:
                pdb.set_trace()

            # store branch_results of this path
            array_f[branch_ids == br] = f_cent

            # add upstream part of branch within junction
            if br == 0:
                continue

            # first point of branch
            ip = np.where(branch_ids == br)[0][0]

            # centerline that passes through branch (first occurence)
            cid = np.where(cl_id[ip])[0][0]

            # id of upstream junction
            jc = bif_id[ip - 1]

            # centerline within junction
            is_jc = bif_id == jc
            jc_cent = np.where(np.logical_and(is_jc, cl_id[:, cid]))[0]

            # length of centerline within junction
            jc_path = np.append(
                0,
                np.cumsum(
                    np.linalg.norm(
                        np.diff(points[jc_cent], axis=0), axis=1
                    )
                ),
            )
            jc_path /= jc_path[-1]

            # branch_results at upstream branch
            res_br_u = branch_results[f][branch_ids[jc_cent[0] - 1]]

            # branch_results at beginning and end of centerline within junction
            f0 = res_br_u[-1]
            f1 = res_br[0]

            # map 1d branch_results to centerline using paths
            array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T)(
                jc_path
            ).T

            # count number of outlets of this junction
            n_outlet[jc_cent] += 1

        # normalize branch_results within junctions by number of junction outlets
        is_jc = n_outlet > 0
        array_f[is_jc] = (array_f[is_jc].T / n_outlet[is_jc]).T

        # assemble time steps
        arrays[f] = array_f[:, 0]

    # add arrays to centerline and write to file
    for f, a in arrays.items():
        out_array = numpy_to_vtk(a)
        out_array.SetName(f)
        centerline_handler.data.GetPointData().AddArray(out_array)
    if not os.path.exists(f"trees/zerod_output_cent/{junction_mode}/{tree_name_base}/{tree_name}"):
        os.makedirs(f"trees/zerod_output_cent/{junction_mode}/{tree_name_base}/{tree_name}")
    centerline_handler.to_file(f"trees/zerod_output_cent/{junction_mode}/{tree_name_base}/{tree_name}/centerline_sol.vtp")
    #print(f"Centerline projection for {tree_name} in {junction_mode} mode completed.")
    return

tree_name = sys.argv[1]
junction_mode = sys.argv[2]
project_to_centerline(tree_name, junction_mode)