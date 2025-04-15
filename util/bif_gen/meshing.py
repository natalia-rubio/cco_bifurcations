from sv import *
import vtk
import os
import platform
import numpy as np
from vtk.util import numpy_support
import pdb
from util.bif_gen.mesh_utils import *
import sys
sys.path.append('/Users/natalia/Desktop/cco_bifurcations')

def build_model(contours):
    options = geometry.LoftNurbsOptions()
    solids = create_vessels(contours)
    for i, solid in enumerate(solids):
        solid.write('/Users/natalia/Desktop/solid_{}'.format(i),'vtp')
    unioned_model,terminating = union_all(solids)
    model = modeling.PolyData()
    tmp = unioned_model.get_polydata()
    NUM_CAPS = 4 # Should be 3????
    ############################
    # COMBINE FACES
    ############################
    if not terminating:
        model.set_surface(tmp)
        model.compute_boundary_faces(45)
        caps = model.identify_caps()
        ids = model.get_face_ids()
        walls = [ids[i] for i,x in enumerate(caps) if not x]
        while len(walls) > 1:
            target = walls[0]
            lose = walls[1]
            model.combine_faces(target,[lose])
            #combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
            #model.set_surface(combined)
            ids = model.get_face_ids()
            caps = model.identify_caps()
            walls = [ids[i] for i,x in enumerate(caps) if not x]
            print(walls)
        ids = model.get_face_ids()
        if True:
            print("no dmg") #dmg.add_model('CCO_11',model)
        if len(ids) > NUM_CAPS:
            face_cells = []
            for idx in ids:
                face = model.get_face_polydata(idx)
                cells = face.GetNumberOfCells()
                print(cells)
                face_cells.append(cells)
            data_to_remove = len(ids) - NUM_CAPS
            remove_list = []
            for i in range(data_to_remove):
                remove_list.append(ids[face_cells.index(min(face_cells))])
                face_cells[face_cells.index(min(face_cells))] += 1000
            print(remove_list)
            while len(remove_list) > 0:
                target = walls[0]
                lose = remove_list.pop(-1)
                model.combine_faces(target,[lose])
                #combined = mesh_utils.combine_faces(model.get_polydata(),target,lose)
                #model.set_surface(combined)
                print(remove_list)
            print(model.get_face_ids())
        ###############################
        # LOCAL SMOOTHING (not included)
        ###############################
        #smoothing_params = {'method':'constrained', 'num_iterations':5, 'constrain_factor':0.2, 'num_cg_solves':30}
        smooth_model = model.get_polydata()
        for idx, contour_set in enumerate(contours):
            if idx == 0:
                continue
            smoothing_params = {'method':'constrained', 'num_iterations':3, 'constrain_factor':1, 'num_cg_solves':30}
            #smoothing_params = {'method':'constrained', 'num_iterations':10, 'constrain_factor':1, 'num_cg_solves':50}
            smooth_model = geometry.local_sphere_smooth(smooth_model,contours[0][0].get_radius()*10,contour_set[0].get_center(),smoothing_params)
            print('local sphere smoothing {}'.format(idx))
        model.set_surface(smooth_model)
    model = clean(model)
    return model, walls

def get_mesh(model, contours, walls, edge_size=0.1):
    done = False
    min_edge = 0.001
    #edge_size = 0.1
    attempt = 0

    v2_start = contours[1][0].get_center()
    v2_end = contours[1][-1].get_center()
    v2_mid = [(v2_start[i] + v2_end[i])/2 for i in range(3)]
    v2_len = np.linalg.norm(np.array(v2_end) - np.array(v2_start))
    v2_rad = contours[1][-1].get_radius()
    v1_rad = contours[0][0].get_radius()
    edge_size = v1_rad/3
    
    cap_indicators = model.identify_caps()
    ids = model.get_face_ids()
    caps= [ids[i] for i,x in enumerate(cap_indicators) if x]

    cap_areas = []
    cap_radii = []
    cap_edge_size = []
    cap_dict = {}
    for id in caps:
        cap_areas.append(surf_area(model.get_face_polydata(id)))
        cap_radii.append(np.sqrt(cap_areas[-1]/np.pi))
        cap_edge_size.append(min([cap_radii[-1]/3, edge_size]))
        cap_dict.update({id:surf_area(model.get_face_polydata(id))})

    # while edge_size > min_edge and not done:
    #     try:
    faces = model.get_face_ids()
    mesher = meshing.create_mesher(meshing.Kernel.TETGEN)
    mesher.set_model(model)
    #mesher.load_model("model_tmp.vtp")
    tet_options = meshing.TetGenOptions(edge_size,True,True)
    #tet_options = mesher.get_options()
    #tet_options.no_merge = False
    #tet_options.global_edge_size = 5
    
    tet_options.optimization = 3
    #tet_options.minimum_dihedral_angle = 18.0
    tet_options.quality_ratio = 1.4
    #tet_options.no_bisect = True

    # tet_options.local_edge_size_on =  True
    # tet_options.local_edge_size = []
    # for i in range(len(caps)):
    #     tet_options.local_edge_size.append({'face_id':caps[i], 'edge_size':cap_edge_size[i]})
    # tet_options.local_edge_size_on = True 

    mesher.set_boundary_layer_options(number_of_layers=4, edge_size_fraction=0.8, layer_decreasing_ratio=0.8, constant_thickness=False)
    for idx, contour_set in enumerate(contours):
        if idx == 0:
            continue
        print('adding sphere refinement')
        print("edge_size: {}".format(edge_size))

        tet_options.sphere_refinement.append({'edge_size':edge_size*min([0.5, 2*v2_rad/v1_rad]), 'radius':v2_len*0.7, #*0.5, 
                        'center':v2_mid})
        # tet_options.sphere_refinement.append({'edge_size':edge_size*min([0.5, 2*v2_rad/v1_rad]), 'radius':1*v2_len, #*0.5, 
        #                 'center':v2_end})
        # tet_options.sphere_refinement.append({'edge_size':edge_size*min([0.5, 2*v2_rad/v1_rad]), 'radius':v1_rad*2, 
        #                         'center':contour_set[0].get_center()})
        
        
    tet_options.sphere_refinement_on = True

    # tet_options.radius_meshing_centerlines = vtk_centerlines
    # tet_options.radius_meshing_scale = 0.002*edge_size/0.0528
    # tet_options.radius_meshing_on = True

    print("Options values: ")
    [ print("  {0:s}:{1:s}".format(key,str(value))) for (key, value) in sorted(tet_options.get_values().items()) ]

    
    mesher.set_walls(walls)
    mesher.generate_mesh(tet_options)
    msh = mesher.get_mesh()
    done = True
        # except:
        #     done = False
        #     edge_size = edge_size - 0.1*edge_size
        #     attempt += 1
    return mesher, msh, cap_dict


def get_inlet_cap(mesher, walls):
    print("finding max area cap")
    y_locs = []
    mass = vtk.vtkMassProperties()
    mesh = modeling.PolyData()
    faces = mesher.get_model_face_ids()
    print("got model face ids")
    assert walls[0] == faces[0], "first face is wall"
    for face in faces:
        if face == walls[0]:
            continue
        pts = numpy_support.vtk_to_numpy(mesher.get_face_polydata(face).GetPoints().GetData())
        y_loc = np.min(pts[:,1])
        y_locs.append(y_loc)
        print("y_locs: "); print(y_locs)
    ind = np.argmin(np.asarray(y_locs)) + 1
    max_area_cap = faces[ind]
    print("faces" + str(faces))
    print("max_area_cap" + str(max_area_cap))
    return max_area_cap

def save_mesh(mesher, model, walls, cap_dict, geo_dir):
    os.mkdir(geo_dir+'/mesh-complete')
    os.mkdir(geo_dir+'/mesh-complete/mesh-surfaces')
    os.mkdir(geo_dir+'/centerlines')
    model.write(geo_dir + '/mesh-complete'+os.sep+'model_tmp','vtp')
    mesher.write_mesh(geo_dir + '/mesh-complete'+os.sep+'mesh-complete.mesh.vtu')
    mesh_out = modeling.PolyData()
    mesh_out.set_surface(mesher.get_surface())
    mesh_out.write(geo_dir + '/mesh-complete'+os.sep+'mesh-complete','exterior.vtp')
    mesh_out.set_surface(mesher.get_face_polydata(walls[0]))
    mesh_out.write(geo_dir + '/mesh-complete'+os.sep+'walls_combined','vtp')

    for face in mesher.get_model_face_ids():
        if face == walls[0]:
            continue
        mesh_out.set_surface(mesher.get_face_polydata(face))
        mesh_out.write(geo_dir + '/mesh-complete/mesh-surfaces'+os.sep+'cap_{}'.format(face),'vtp')
        

    max_area_cap = get_inlet_cap(mesher, walls)
    print("Max Area Cap: ")
    print(max_area_cap)
    faces = model.get_face_ids()
    out_caps = faces[1:]
    out_caps.remove(max_area_cap)
    assert len(out_caps) == 2, "Wrong number of caps."
    np.save(geo_dir+"/max_area_cap", np.asarray([max_area_cap]), allow_pickle = True)
    #pdb.set_trace()
    save_dict(cap_dict, geo_dir+"/cap_dict")

    cent_solid = modeling.PolyData()
    cent = vmtk.centerlines(model.get_polydata(), inlet_ids = [max_area_cap], outlet_ids = out_caps, use_face_ids = True)
    print("Centerlines generated.")
    cent_solid.set_surface(cent)
    print("Centerline surface set.")
    cent_solid.write(geo_dir +  '/centerlines/centerline', "vtp")