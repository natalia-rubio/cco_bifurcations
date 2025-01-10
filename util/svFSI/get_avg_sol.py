import vtk
import os
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy as v2n
from tqdm import tqdm
from util.get_bc_integrals import get_res_names
from util.vtk_functions import read_geo, write_geo, calculator, cut_plane, connectivity, get_points_cells, clean, Integration
import util.junction_proc
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

    for v in get_res_names(inp_3d, 'Velocity'):
        #fun = '(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+').' + v
        fun = 'dot(iHat*'+repr(normal[0])+'+jHat*'+repr(normal[1])+'+kHat*'+repr(normal[2])+',' + v + ")"
        inp = calculator(inp, fun, [v], 'normal_' + v)

    return Integration(inp)

def get_inds(arr, vals):
    arr = np.asarray(arr); vals = np.asarray(vals)
    inds = 0 * vals
    for i in range(vals.size):
        inds[i] = np.where(arr == vals[i])[0][0]
    inds = list(inds)
    return inds

def get_avg_unsteady_results(ss_tol, 
                             inc, 
                             max_time_step, 
                             fpath_1d, 
                             fpath_3d_base,
                             fpath_out, 
                             pt_inds, 
                             only_caps=False, 
                             only_area = False):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results
        fpath_out: output path
        only_caps: extract solution only at caps, not in interior (much faster)
    Returns:
        res: dictionary of results in all branches, in all segments for all result arrays
    """
    # read 1d and 3d model
    reader_1d = read_geo(fpath_1d).GetOutput()
    # get point and normals from centerline
    # res_names = get_res_names(reader_3d, ['Pressure', 'Velocity'])
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))
    
    junc_inds = get_inds(arr = gid, vals = pt_inds)
    points = v2n(reader_1d.GetPoints().GetData())[junc_inds]
    path = v2n(reader_1d.GetPointData().GetArray('Path'))[junc_inds]
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))[junc_inds]

    # initialize output
    # for name in res_names + ['area']:
    #     array = vtk.vtkDoubleArray()
    #     array.SetName(name)
    #     array.SetNumberOfValues(len(pt_inds))
    #     array.Fill(0)
    #     reader_1d.GetPointData().AddArray(array)

    # for name in res_names:
    #     if name[0:8] == "pressure":
    #         if name[9:14] == "error":
    #             continue
    #         time = float(name[-5:])
    #         if time > max_timestep:
    #             max_timestep = time
    #             #print(time)

    # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3

    num_time_steps = int(max_time_step/inc)
    times = [int(i*inc) for i in range(1,num_time_steps)]
    pressure_in_time =  np.zeros((num_time_steps,len(pt_inds))) # initialize pressure_in_time matrix, each column is a mesh point, each row is a timestep
    flow_in_time =      np.zeros((num_time_steps,len(pt_inds))) # initialize flow_in_time matrix, each column is a mesh point, each row is a timestep
    areas =            np.zeros((1,len(pt_inds))) # initialize area matrix, each column is a mesh point, each row is a timestep
    tangents =         np.zeros((3,len(pt_inds))) # initialize tangent matrix
    #times = [int(fpath_3d.split("_")[-1][:-4]), int(fpath_3d_prev.split("_")[-1][:-4])]  # list of timesteps
    paths = np.zeros((1, len(pt_inds), num_time_steps))
    #print(f"Reducing timesteps {times}.")
    for time_step_index, time_step in enumerate(times):
        print(f"Reducing timestep {time_step}.")

        reader_3d = read_geo(fpath_3d_base + f"{int(time_step):03d}.vtu").GetOutput()
        #res_names = get_res_names(reader_3d, ['Pressure', 'Velocity']) # get all result array names
        
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
            except:
                print("integration error")
                continue

            # integrate all output arrays
            pressure_in_time[time_step_index, i] = integral.evaluate("Pressure")  # add timestep row to pressure_in_time
            flow_in_time[time_step_index, i] = integral.evaluate("Velocity")  # add timestep row to pressure_in_time
    
            # add 1d geometric information (only once)
            if time_step_index == 0:
                areas[0, i] = integral.area()  # add timestep row to pressure_in_time
                paths[0, i] =  path[i] # add timestep row to pressure_in_time
                tangents[:, i] = normals[i].reshape(3,)  # add timestep row to pressure_in_time



    print("Pressure in time: ")
    print(pressure_in_time)
    print("Flow in time: ") 
    print(flow_in_time)

    res_dict = {"flow_in_time": flow_in_time,
                "pressure_in_time": pressure_in_time,
                "areas": areas,
                "tangents": tangents,
                "times" : times,
                "paths" : paths}

    res_dict.update({"pt_inds": pt_inds})
    save_dict(res_dict, fpath_out + "_red_sol")

    save_centerline_sol = False
    # if save_centerline_sol:
    #     cent_sol = project_to_cent(fpath_1d, fpath_3d, only_caps=only_caps)
    #     write_geo(fpath_3d[:-4] + "_cent.vtp", cent_sol)
    return res_dict


def get_avg_steady_results(ss_tol, inc, fpath_1d, fpath_3d, fpath_3d_prev, fpath_out, pt_inds, only_caps=False, only_area = False):
    """
    Extract 3d results at 1d model nodes (integrate over cross-section)
    Args:
        fpath_1d: path to 1d model
        fpath_3d: path to 3d simulation results
        fpath_out: output path
        only_caps: extract solution only at caps, not in interior (much faster)
    Returns:
        res: dictionary of results in all branches, in all segments for all result arrays
    """
    # read 1d and 3d model
    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()
    reader_3d_prev = read_geo(fpath_3d_prev).GetOutput()


    # get all result array names
    res_names = get_res_names(reader_3d, ['Pressure', 'Velocity'])

    # get point and normals from centerline
    gid = v2n(reader_1d.GetPointData().GetArray('GlobalNodeId'))
    
    junc_inds = get_inds(arr = gid, vals = pt_inds)
    points = v2n(reader_1d.GetPoints().GetData())[junc_inds]
    path = v2n(reader_1d.GetPointData().GetArray('Path'))[junc_inds]
    normals = v2n(reader_1d.GetPointData().GetArray('CenterlineSectionNormal'))[junc_inds]

    # initialize output
    max_timestep = 0
    for name in res_names + ['area']:
        array = vtk.vtkDoubleArray()
        array.SetName(name)
        array.SetNumberOfValues(len(pt_inds))
        array.Fill(0)
        reader_1d.GetPointData().AddArray(array)

    for name in res_names:
        if name[0:8] == "pressure":
            if name[9:14] == "error":
                continue
            time = float(name[-5:])
            if time > max_timestep:
                max_timestep = time
                #print(time)

    # move points on caps slightly to ensure nice integration
    ids = vtk.vtkIdList()
    eps_norm = 1.0e-3


    num_time_steps = int(max_timestep)
    
    pressure_in_time =  np.zeros((2,len(pt_inds))) # initialize pressure_in_time matrix, each column is a mesh point, each row is a timestep
    flow_in_time =      np.zeros((2,len(pt_inds))) # initialize flow_in_time matrix, each column is a mesh point, each row is a timestep
    areas =            np.zeros((1,len(pt_inds))) # initialize area matrix, each column is a mesh point, each row is a timestep
    tangents =         np.zeros((3,len(pt_inds))) # initialize tangent matrix
    times = [int(fpath_3d.split("_")[-1][:-4]), int(fpath_3d_prev.split("_")[-1][:-4])]  # list of timesteps
    paths = np.zeros((1, len(pt_inds),))
    num_time_steps = len(times)
    print(f"Reducing {num_time_steps} timesteps.")

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
            integral_prev = get_integral(reader_3d_prev, points[i], normals[i])
        except:
            print("integration error")
            continue

        # integrate all output arrays
        pressure_in_time[0, i] = integral.evaluate("Pressure")  # add timestep row to pressure_in_time
        pressure_in_time[1, i] = integral_prev.evaluate("Pressure")  # add timestep row to pressure_in_time
        flow_in_time[0, i] = integral.evaluate("Velocity")  # add timestep row to pressure_in_time
        flow_in_time[1, i] = integral_prev.evaluate("Velocity")  # add timestep row to pressure_in_time
        areas[0, i] = integral.area()  # add timestep row to pressure_in_time
        paths[0, i] =  path[i] # add timestep row to pressure_in_time
        tangents[:, i] = normals[i].reshape(3,)  # add timestep row to pressure_in_time



    print("Pressure in time: ")
    print(pressure_in_time)
    print("Flow in time: ") 
    print(flow_in_time)

    dPlast = np.abs(pressure_in_time[-1,0] - pressure_in_time[-1,[1, 2]])
    dP2last = np.abs(pressure_in_time[-2,0] - pressure_in_time[-2,[1, 2]])

    print(f"dPlast: {dPlast}.  dP2last: {dP2last}.")
    conv = True
    if np.any(np.abs(dPlast - dP2last)/dPlast > ss_tol):
        conv = False
        print("not converged")
        pressure_in_time = np.zeros((0,))
        flow_in_time = np.zeros((0,))
        times = [0]
    else:
        pressure_in_time = pressure_in_time[-1,:]
        flow_in_time = flow_in_time[-1,:]
        if np.abs((flow_in_time[0] - np.sum(flow_in_time[1:]))/flow_in_time[0]) > 0.05:
            print("Mass not conserved!")
            conv = False

    res_dict = {"flow_in_time": flow_in_time,
                "pressure_in_time": pressure_in_time,
                "areas": areas,
                "tangents": tangents,
                "times" : times,
                "paths" : paths}

    if conv == True:
        res_dict.update({"pt_inds": pt_inds})
        save_dict(res_dict, fpath_out + "_red_sol")
    else:
        os.system(f"rm {fpath_out}")

    save_centerline_sol = False
    if save_centerline_sol:
        cent_sol = project_to_cent(fpath_1d, fpath_3d, only_caps=only_caps)
        write_geo(fpath_3d[:-4] + "_cent.vtp", cent_sol)
    return res_dict, conv


def project_to_cent(fpath_1d, fpath_3d, only_caps=False):

    reader_1d = read_geo(fpath_1d).GetOutput()
    reader_3d = read_geo(fpath_3d).GetOutput()# get all result array names
    res_names = get_res_names(reader_3d, ['Pressure', 'Velocity'])# get point and normals from centerline
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
            integral = get_integral(reader_3d, points[i], normals[i])
        except Exception:
            continue # integrate all output arrays

        for name in res_names:
            reader_1d.GetPointData().GetArray(name).SetValue(i, integral.evaluate(name))
        reader_1d.GetPointData().GetArray('area').SetValue(i, integral.area())
    return reader_1d