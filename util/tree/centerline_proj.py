import sys
import vtk
import os
import numpy as np
import pdb
sys.path.append("/Users/natalia/Desktop/CCO_junctions")
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm

from util.tools.get_bc_integrals import get_res_names
#from util.junction_proc import *
from geo_processing import *
from util.tools.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle
from sklearn.linear_model import LinearRegression

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def get_length(locs):
    length = 0
    for i in range(1, locs.shape[0]):
        length += np.linalg.norm(locs[i, :] - locs[i-1, :])
    return length


def slice_vessel(inp_3d, origin, normal):
    """
    Slice 3d geometry at certain plane
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # cut 3d geometry
    cut_3d = cut_plane(inp_3d, origin, normal)
    #write_geo(f'slice_{origin[0]}.vtp', cut_3d.GetOutput())

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)
    #write_geo(f'con_{origin[0]}.vtp', con.GetOutput())
    return con


def get_integral(inp_3d, origin, normal):
    """
    Slice simulation at certain plane and integrate
    Args:
        inp_1d: vtk InputConnection for 1d centerline
        inp_3d: vtk InputConnection for 3d volume model
        origin: plane origin
        normal: plane normal
    Returns:
        Integration object
    """
    # slice vessel at given location
    inp = slice_vessel(inp_3d, origin, normal)

    # recursively add calculators for normal velocities
    for v in get_res_names(inp_3d, 'velocity'):
        #fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        fun = 'dot(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+',' + v + ")"
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

def extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 1000):

    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])# get point and normals from centerline
    points = v2n(reader_1d.GetPoints().GetData())
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))# initialize output

    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(reader_1d.GetNumberOfPoints())
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array) # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3 # integrate results on all points of intergration cells
    print(f"Extracting solution at {reader_1d.GetNumberOfPoints()} points.")
    for i in tqdm(range(reader_1d.GetNumberOfPoints())):
        # check if point is cap
        reader_1d.GetPointCells(i, ids)
        if ids.GetNumberOfIds() == 1:
            if gid[i] == 0:
                # inlet
                points[i] += eps_norm * normals[i]
            else:
                # outlets
                points[i] -= eps_norm * normals[i]
        else:
            if only_caps:
                continue # create integration object (slice geometry at point/normal)

        try:
            #import pdb; pdb.set_trace()
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            continue # integrate all output arrays

        for name in res_names:
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    write_geo(fpath_out, reader_1d)
    return

if __name__ == "__main__":
    fpath_1d = "data/CCO_tree/geometry/centerlines.vtp"
    fpath_3d = "data/CCO_tree/sim_3D/solution_CCO.vtu"
    fpath_out = "data/CCO_tree/centerline_sol.vtp"
    extract_results(fpath_1d, fpath_3d, fpath_out, only_caps=False, num_time_steps = 50)
# def get_res_dict(sol_path1d, only_caps=False, num_time_steps = 2):
#     """
#     Get all result array names
#     Args:
#         inp: vtk InputConnection
#         names: list of names to search for
#     Returns:
#         list of result array names
#     """
#     soln = read_geo(sol_path1d).GetOutput()
#     soln_dict = get_all_arrays(soln)
#     res_names = list(soln_dict.keys())
#     res_names.sort()

#     pt_inds = [0,-1]
#     pressure_in_time= np.zeros((num_time_steps,len(pt_inds))) # initialize pressure_in_time matrix, each column is a mesh point, each row is a timestep
#     flow_in_time = np.zeros((num_time_steps,len(pt_inds))) # initialize flow_in_time matrix, each column is a mesh point, each row is a timestep
#     area= np.array([soln_dict["area"][0], soln_dict["area"][-1]])
#     times = list()  # list of timesteps

#     p_cnt = 0
#     v_cnt = 0
#     for name in res_names:
#         if name[0:8] == "pressure":
#             if name[9:14] == "error":
#                 continue
#             pressure_in_time[p_cnt, 0] = soln_dict[name][0]  # add timestep row to pressure_in_time
#             pressure_in_time[p_cnt, 1] = soln_dict[name][-1]  # add timestep row to pressure_in_time
#             p_cnt = p_cnt + 1
#             times.append(float(name[-5:]))  # add timestep to times

#         elif name[0:8] == "velocity": # NOTE: keys are labeled "velocity" but are actually flow!!!
#             if name[9:14] == "error":
#                 continue
#             flow_in_time[v_cnt, 0] = soln_dict[name][0]  # add timestep row to flow_in_time
#             flow_in_time[v_cnt, 1] = soln_dict[name][-1]  # add timestep row to flow_in_time
#             v_cnt = v_cnt + 1


#     conv = True; print(pressure_in_time); print(flow_in_time)
#     dP = (pressure_in_time[:,1]-pressure_in_time[:,0])
#     print((dP[1]-dP[0])/dP[1])

#     if np.any(np.abs((dP[1]-dP[0])/dP[1]) > 0.01):
#         print("Pressure change not converged!")
#         print(f"pressure_in_time: {pressure_in_time}")
#         print(f"flow_in_time: {flow_in_time}"); print(((dP[1]-dP[0])/dP[1]))
#         conv = False

#     res_dict = {"flow_in_time": flow_in_time,
#             "pressure_in_time": pressure_in_time,
#             "area": area,
#             "times" : [times[-1]]}
#     return res_dict, conv