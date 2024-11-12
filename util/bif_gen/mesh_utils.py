from sv import *
import vtk
import os
import platform
import numpy as np
from vtk.util import numpy_support

def bad_edges(model):
    # Determine the number of non-manifold and degenerate
    # edges within a sv solid model
    #
    # Parameters:
    #
    # model     (sv.modeling.POLYDATA): sv solid modeling object
    #
    # Returns
    #
    # bad_edges_number   (int): number of non-manifold & degenerate edges
    fe = vtk.vtkFeatureEdges()
    fe.FeatureEdgesOff()
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.SetInputData(model.get_polydata())
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()

def clean(model):
    # Merge duplicate points, clean up bad edges, perform basic filtering
    #
    # Parameters:
    #
    # model    (sv.modeling.POLYDATA): sv solid model
    #
    # Returns:
    #
    # model    (sv.modeling.POLYDATA): sv solid model with cleaned polydata
    #                                  surface mesh
    clean_filter = vtk.vtkCleanPolyData()
    clean_filter.PointMergingOn()
    clean_filter.SetInputData(model.get_polydata())
    clean_filter.Update()
    model.set_surface(clean_filter.GetOutput())
    return model

def tri(model):
    # Triangle Filter for sv solid model
    #
    # Parameters
    #
    # model    (sv.modeling.POLYDATA): sv solid model object
    #
    # Returns
    #
    # model    (sv.modeling.POLYDATA): sv solid model object with
    #                                  surface mesh containing only triangles
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(model.get_polydata())
    tri_filter.Update()
    model.set_surface(tri_filter.GetOutput())
    return model

def fill(model):
    # Fill holes within a sv solid model in order to close an open manifold
    #
    # Parameters
    #
    # model    (sv.modeling.POLYDATA): sv solid modeling object
    #
    # Returns
    #
    # model    (sv.modeling.POLYDATA): sv solid modeling object with a surface
    #                                  mesh containing no holes
    poly = vmtk.cap(surface=model.get_polydata(),use_center=False)
    model.set_surface(poly)
    return model

def surf_area(poly):
    # Calculate the area of a polydata surface
    #
    # Parameters
    #
    # poly    (vtk.PolyData): polydata object
    #
    # Returns
    #
    # surface_area   (float): area of the polydata surface mesh
    mass = vtk.vtkMassProperties()
    mass.SetInputData(poly)
    mass.Update()
    return mass.GetSurfaceArea()

def remesh(model,radius_factor=10):
    # Remesh the surface model using MMG.
    #
    # Parameters
    #
    # model  (sv.modeling.POLYDATA): SV solid modeling object
    # radius_factor           (int): scale mesh global minimum edge size
    #                                based on outlet radius
    #
    # Returns
    #
    # model  (sv.modeling.POLYDATA): remeshed model object
    face_ids = model.get_face_ids()
    smallest = 1e8
    biggest = 0
    for id in face_ids:
        if surf_area(model.get_face_polydata(id)) < smallest:
            smallest = surf_area(model.get_face_polydata(id))
        if surf_area(model.get_face_polydata(id)) > biggest:
            biggest = surf_area(model.get_face_polydata(id))
    radius = (smallest/3.14)**(1/2)
    hmin = radius/radius_factor
    hmax = radius
    print("Remeshing Model:\nhmin: ----> {}\nhmax ----> {}".format(hmin,hmax))
    remeshed_polydata = mesh_utils.remesh(model.get_polydata(),hmin=hmin,hmax=hmax)
    model.set_surface(remeshed_polydata)
    return model

def remesh_face(model,face_id,radius_scale=10):
    # Remesh Faces of a surface model using MMG.
    #
    # Parameters
    #
    # model  (sv.modeling.POLYDATA): SV solid modeling object
    # face_id                 (int): face_id index within solid model
    # radius_factor           (int): scale mesh global minimum edge size
    #                                based on outlet radius
    #
    # Returns
    #
    # model  (sv.modeling.POLYDATA): model with remeshed face
    face_poly = model.get_face_polydata(face_id)
    edge_size = (surf_area(face_poly)/3.14)**(1/2)/radius_scale
    print("Remeshing Face: {} ----> Edge Size: {}".format(face_id,edge_size))
    remeshed_poly = mesh_utils.remesh_faces(model.get_polydata(),[face_id],edge_size)
    model.set_surface(remeshed_poly)
    return model

def remesh_wall(model,wall_id):
    wall_poly = model.get_face_polydata(wall_id)
    cell_number = wall_poly.GetNumberOfCells()
    edge_size = ((surf_area(wall_poly)/cell_number)*0.5)**(1/2)
    remeshed_poly = mesh_utils.remesh_faces(model.get_polydata(),[wall_id],edge_size)
    model.set_surface(remeshed_poly)
    new_cell_number = model.get_face_polydata(wall_id).GetNumberOfCells()
    print("Face Elements: {} ----> Edge Size: {}".format(cell_number,new_cell_number))
    return model

def remesh_caps(model):
    # Remesh all caps of a surface model using MMG. By definition
    # this function will not remesh the walls of the surface model.
    #
    # Parameters
    #
    # model  (sv.modeling.POLYDATA): SV solid modeling object
    #
    # Returns
    #
    # model  (sv.modeling.POLYDATA): model with remeshed caps
    cap_ids = model.identify_caps()
    face_ids = model.get_face_ids()
    for i,c in enumerate(cap_ids):
        if c:
            model = remesh_face(model,face_ids[i])
    return model

def norm(model):
    # Determine the normal vectors along the
    # polydata surface.
    #
    # PARAMETERS
    # model    (sv.modeling.POLYDATA): SV solid modeling object
    #
    # Returns
    #
    # model    (sv.modeling.POLYDATA): SV solid model with calculated normals
    norm_filter = vtk.vtkPolyDataNormals()
    norm_filter.AutoOrientNormalsOn()
    norm_filter.ComputeCellNormalsOn()
    norm_filter.ConsistencyOn()
    norm_filter.SplittingOn()
    norm_filter.NonManifoldTraversalOn()
    norm_filter.SetInputData(model.get_polydata())
    norm_filter.Update()
    model.set_surface(norm_filter.GetOutput())
    return model

def loft(contours,num_pts=50,distance=False):
    # Generate an open lofted NURBS surface along a given
    # vessel contour group.
    #
    # PARAMETERS:
    # contours (list):  list of contour polydata objects defining one vessel.
    # num_pts  (int) :  number of sample points to take along each contour.
    # distance (bool):  flag to use distance based method for contour alignment
    #
    # Return
    #
    # loft_solid  (sv.modeling.POLYDATA): sv solid model for the open lofted surface
    for idx in range(len(contours)):
        contours[idx] = geometry.interpolate_closed_curve(polydata=contours[idx],number_of_points=num_pts)
        if idx != 0:
            contours[idx] = geometry.align_profile(contours[idx-1],contours[idx],distance)
    options = geometry.LoftNurbsOptions()
    loft_polydata = geometry.loft_nurbs(polydata_list=contours,loft_options=options)
    loft_solid = modeling.PolyData()
    loft_solid.set_surface(surface=loft_polydata)
    return loft_solid

def loft_all(contour_list):
    # Loft all vessels defining the total model that you want to create.
    #
    # PARAMETERS
    # contour_list: (list): list of lists that contain polydata contour groups
    #                      Example for two vessels:
    #
    #                      contour_list -> [[polydataContourObject1,polydataContourObject2],[polydataContourObject1,polydataContourObject2]]
    #
    # RETURNS:
    # lofts:        (list): list of open sv solid models of the lofted 3D surface. Note that
    #                       the loft is not yet capped.
    lofts = []
    for group in contour_list:
        contours,polydata =  clean_contours(group)
        lofts.append(loft(polydata))
    return lofts


def cap_all(loft_list):
    # Cap all lofted vessels.
    #
    # PARAMETERS:
    # loft_list  (list): list of sv modeling solid objects that are open lofts generated from
    #                    the 'loft_all' function.
    #
    # RETURNS:
    # capped     (list): list of capped solids
    capped = []
    for loft_solid in loft_list:
        capped_solid = modeling.PolyData()
        capped_solid.set_surface(vmtk.cap(surface=loft_solid.get_polydata(),use_center=False))
        capped_solid.compute_boundary_faces(angle=45)
        capped.append(capped_solid)
    return capped

def check_cap_solids(cap_solid_list):
    # Check capped solids for bad edges
    #
    # Parameters
    # cap_solid_list  (list): list of sv.modeling.POLYDATA solids
    #
    # Returns
    #
    # bad_edges_exist (bool): True/False value for if any bad edges
    #                         exist within any of the solids within
    #                         list
    for solid in cap_solid_list:
        if bad_edges(solid) > 0:
            return False
    return True

def clean_contours(contours):
    num = len(contours)-2
    new_contours = [contours[0]]
    new_poly     = [contours[0].get_polydata()]
    for i in range(1,len(contours)-1):
        n1 = np.array(new_contours[-1].get_normal())
        n2 = np.array(contours[i].get_normal())
        n3 = np.array(contours[i+1].get_normal())
        #if ((np.arccos(np.dot(n1,n2.T))/np.pi)*180 > 10) or ((np.arccos(np.dot(n2,n3.T))/np.pi)*180 > 10) or check_connection(contours[i], contour_list=contours):
        new_contours.append(contours[i])
        new_poly.append(contours[i].get_polydata())
    if len(new_contours) == 1:
        mid = len(contours)//2
        new_contours.append(contours[mid])
        new_poly.append(contours[mid].get_polydata())
    new_contours.append(contours[-1])
    new_poly.append(contours[-1].get_polydata())
    return new_contours, new_poly

def check_connection(contour,contour_list):
    keep = False
    check_center = np.array(contour.get_center())
    for group in contour_list:
        c = np.array(group[0].get_center())
        if np.linalg.norm(check_center - c) < (group[0].get_radius()*2*np.pi+contour.get_radius()*2*np.pi):
            keep = True
            break
    return keep

def create_vessels(contour_list,attempts=5):
    # create seperate capped vessels for all contour groups defining a model of interest.
    #
    # PARAMETERS:
    # contour_list: (list): list of lists of contour polydata objects defining individual vessels
    #                       within the total model.
    # attemps:      (int) : the number of times that bad edges correction will be attemped during loft
    #                       alignment
    i = 0
    success = False
    while not success and i < attempts:
        lofts = loft_all(contour_list)
        cap_solids = cap_all(lofts)
        success = check_cap_solids(cap_solids)
        i += 1
    if success:
        print('Lofting Passed')
    else:
        print('Lofting Failed')
    return cap_solids

def show(model):
    polydata = model.get_polydata()

    colors = vtk.vtkNamedColors()
    background = 'white'
    mapper = vtk.vtkPolyDataMapper()
    actor  = vtk.vtkActor()
    mapper.SetInputDataObject(polydata)
    actor.SetMapper(mapper)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(colors.GetColor3d(background))

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetWindowName('Model View')

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    renderer.AddActor(actor)
    render_window.Render()
    interactor.Start()

# capped_vessels = create_vessels(contour_list)
###############################
# INITIALIZE MODELING KERNEL
###############################
def robust_union(model_1,model_2):
    modeler = modeling.Modeler(modeling.Kernel.POLYDATA)
    model_1_be = bad_edges(model_1)
    model_2_be = bad_edges(model_2)
    print("Model 1 Bad Edges: {}\n Model 2 Bad Edges: {}".format(model_1_be,model_2_be))
    if model_1_be == 0 and model_2_be == 0:
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        if bad_edges(unioned_model) > 0:
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Filling')
            unioned_model = fill(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            print('Cleaning')
            unioned_model = clean(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
            unioned_model = tri(unioned_model)
            print('Unioned Model Bad Edges: {}'.format(bad_edges(unioned_model)))
        print('union successful')
        return unioned_model
    else:
        print('1 or both models have bad edges.')
        unioned_model = modeler.union(model_1,model_2)
        unioned_model = clean(unioned_model)
        unioned_model = norm(unioned_model)
        return unioned_model

def union_all(solids,n_cells=100):
    for i in range(len(solids)):
        solids[i] = norm(solids[i])
        solids[i] = remesh(solids[i])
        solids[i] = remesh_caps(solids[i])
    joined = robust_union(solids[0],solids[1])
    for i in range(2,len(solids)):
        print("UNION NUMBER: "+str(i)+"/"+str(len(solids)-1))
        joined = robust_union(joined,solids[i])
        if joined is None:
            print("unioning failed")
            return None,True
    print("unioning passed")
    return joined,False

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
            smoothing_params = {'method':'constrained', 'num_iterations':3, 'constrain_factor':0.1+(0.9*(1-contour_set[10].get_radius()/contours[0][0].get_radius())), 'num_cg_solves':30}
            #smoothing_params = {'method':'constrained', 'num_iterations':10, 'constrain_factor':1, 'num_cg_solves':50}
            smooth_model = geometry.local_sphere_smooth(smooth_model,contour_set[0].get_radius()*4,contour_set[2].get_center(),smoothing_params)
            print('local sphere smoothing {}'.format(idx))
        model.set_surface(smooth_model)
    model = clean(model)
    return model, walls

def get_mesh(model, contours, walls):
    done = False
    min_edge = 0.001
    edge_size = 0.1
    attempt = 0

    while edge_size > min_edge and not done:
        try:
            faces = model.get_face_ids()
            mesher = meshing.create_mesher(meshing.Kernel.TETGEN)
            tet_options = meshing.TetGenOptions(edge_size,True,True)
            tet_options.no_merge = False
            tet_options.optimization = 5
            tet_options.minimum_dihedral_angle = 18.0
            mesher.set_boundary_layer_options(number_of_layers=2, edge_size_fraction=0.5, layer_decreasing_ratio=0.8, constant_thickness=False)
            for idx, contour_set in enumerate(contours):
                if idx == 0:
                    continue
                print('adding sphere refinement')
                tet_options.sphere_refinement.append({'edge_size':edge_size*0.5, 'radius':contour_set[0].get_radius()*3, 
                                        'center':contour_set[2].get_center()})
            tet_options.sphere_refinement_on = True
            mesher.set_model(model)
            mesher.set_walls(walls)
            mesher.generate_mesh(tet_options)
            msh = mesher.get_mesh()
            done = True
        except:
            done = False
            edge_size = edge_size - 0.1*edge_size
            attempt += 1
    return mesher, msh


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

def save_mesh(mesher, model, walls, geo_dir):
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

    cent_solid = modeling.PolyData()
    cent = vmtk.centerlines(model.get_polydata(), inlet_ids = [max_area_cap], outlet_ids = out_caps, use_face_ids = True)
    print("Centerlines generated.")
    cent_solid.set_surface(cent)
    print("Centerline surface set.")
    cent_solid.write(geo_dir +  '/centerlines/centerline', "vtp")