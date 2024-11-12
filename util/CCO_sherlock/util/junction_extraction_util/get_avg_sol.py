# credit: Martin Pfaller (mrp89)
import vtk
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm
from util.junction_extraction_util.get_bc_integrals import get_res_names
from util.junction_extraction_util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

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

    # extract region closest to centerline
    con = connectivity(cut_3d, origin)

    return con

def get_inds(arr, vals):
    arr = np.asarray(arr); vals = np.asarray(vals)
    inds = 0 * vals
    for i in range(vals.size):
        inds[i] = np.where(arr == vals[i])[0][0]
    inds = list(inds)
    return inds


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
        fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)


def get_avg_results(fpath_1d, fpath_3d, fpath_out, pt_inds, only_caps=False):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results
        fpath_out: output path
        only_caps: extract solution only at caps, not in interior (much faster)
    Returns:
        res: dictionary of results at inlet and outlet locations
    """
    # read 1d and 3d model
    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()

    # get all result array names
    res_names = get_res_names(reader_3d, ['pressure', 'velocity'])

    # get point and normals from centerline
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))
    #junc_inds = np.isin(gid, np.asarray(pt_inds))
    junc_inds = get_inds(arr = gid, vals = pt_inds)
    assert np.all(gid[junc_inds] == np.asarray(pt_inds)), "check that junc ids are in the right order"
    points = v2n(reader_1d.GetPoints().GetData())[junc_inds]

    # locator = vtk.vtkPointLocator()
    # locator.Initialize()
    # locator.SetDataSet(vtk.vtkPolyData().SetPoints(reader_3d.GetPoints()))
    # locator.BuildLocator()
    # points = [locator.FindClosestPoint(p) for p in points_cent]

    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))[junc_inds]


    # initialize output
    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(len(pt_inds))
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array)

    # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3

    pressure_in_time= np.zeros((100,len(pt_inds))) # initialize pressure_in_time matrix, each column is a mesh point, each row is a timestep
    flow_in_time = np.zeros((100,len(pt_inds))) # initialize flow_in_time matrix, each column is a mesh point, each row is a timestep
    times = list()  # list of timesteps

    # integrate results on all points of intergration cells
    for i in tqdm(range(len(pt_inds))):

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
                continue

        # create integration object (slice geometry at point/normal)
        try:
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            print("Integration error.")
            continue

        count = int(0)

        # integrate all output arrays
        for name in res_names:
            print(name)
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
            soln = np.array(reader_1d.GetPointData().GetArray(name)).reshape(1,-1)
            if name[0:8] == "pressure_in_time":
                if name[9:14] == "error":
                    continue
                pressure_in_time[count, i] = soln[0][i]  # add timestep row to pressure_in_time
                if i == 0:
                    times.append(float(name[-5:]))  # add timestep to times

            elif name[0:8] == "velocity": # NOTE: keys are labeled "velocity" but are actually flow!!!
                if name[9:14] == "error":
                    continue
                flow_in_time[count, i] = soln[0][i]
                count += 1

        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())

    res_dict = {"flow_in_time": flow_in_time,
                "pressure_in_time": pressure_in_time,
                "times" : times}

    save_dict(res_dict, fpath_out)
    return res_dict
