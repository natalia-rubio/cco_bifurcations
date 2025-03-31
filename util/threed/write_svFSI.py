import xml.etree.ElementTree as ET
import os
import glob
import numpy as np

def write_svfsiplus_xml(file_dir, sim_dir, n_tsteps=800, dt=0.001, mesh_complete='mesh-complete'):
    '''
    write an svFSIplus.xml file from a simulation directory which contains a mesh surfaces directory
    '''

    # set mesh complete diretory
    mesh_complete_local = os.path.join(file_dir,'mesh-complete')
    mesh_complete_sher = os.path.join(sim_dir,'mesh-complete')
    inlet_cap = "cap_" + os.listdir(mesh_complete_local + "/inlet_cap")[0]+".vtp"
    caps = os.listdir(mesh_complete_local + "/mesh-surfaces")
    outlet_caps = []
    for cap in caps:
        if cap==inlet_cap:
            continue
        outlet_caps.append(cap)

    

    print('writing svFSIplus.xml...')
    # generate XML tree
    svfsifile = ET.Element("svFSIFile")
    svfsifile.set("version", "0.1")
    # General Simulation Parameters
    gensimparams = ET.SubElement(svfsifile, "GeneralSimulationParameters")
    cont_prev_sim = ET.SubElement(gensimparams, "Continue_previous_simulation")
    cont_prev_sim.text = "false"
    num_spatial_dims = ET.SubElement(gensimparams, "Number_of_spatial_dimensions")
    num_spatial_dims.text = "3"
    num_time_steps = ET.SubElement(gensimparams, "Number_of_time_steps")
    num_time_steps.text = str(n_tsteps)
    time_step_size = ET.SubElement(gensimparams, "Time_step_size")
    time_step_size.text = str(dt)
    spec_radius = ET.SubElement(gensimparams, "Spectral_radius_of_infinite_time_step")
    spec_radius.text = "0.5"
    stop_trigger = ET.SubElement(gensimparams, "Searched_file_name_to_trigger_stop")
    stop_trigger.text = "STOP_SIM"
    save_results_to_vtk = ET.SubElement(gensimparams, "Save_results_to_VTK_format")
    save_results_to_vtk.text = "1"
    name_prefix = ET.SubElement(gensimparams, "Name_prefix_of_saved_VTK_files")
    name_prefix.text = "result"
    increment_vtk = ET.SubElement(gensimparams, "Increment_in_saving_VTK_files")
    increment_vtk.text = "100"
    start_saving_tstep = ET.SubElement(gensimparams, "Start_saving_after_time_step")
    start_saving_tstep.text = "1"
    incrememnt_restart = ET.SubElement(gensimparams, "Increment_in_saving_restart_files")
    incrememnt_restart.text = "10"
    convert_bin_vtk = ET.SubElement(gensimparams, "Convert_BIN_to_VTK_format")
    convert_bin_vtk.text = "0"
    verbose = ET.SubElement(gensimparams, "Verbose")
    verbose.text = "1"
    warning = ET.SubElement(gensimparams, "Warning")
    warning.text = "0"
    debug = ET.SubElement(gensimparams, "Debug")
    debug.text = "0"
    # add mesh
    add_mesh = ET.SubElement(svfsifile, "Add_mesh")
    add_mesh.set("name", "msh")
    msh_file_path = ET.SubElement(add_mesh, "Mesh_file_path")
    msh_file_path.text = os.path.join(mesh_complete_sher, 'mesh-complete.mesh.vtu')
    # add faces to mesh
    filelist = glob.glob(os.path.join(mesh_complete_local, 'mesh-surfaces/*.vtp'))
    # filelist = [file for file in filelist_raw if 'wall' not in file]
    # filelist.sort()
    # inflow = filelist.pop(-1)
    # filelist.insert(0, inflow)
    for file in filelist:
        add_face = ET.SubElement(add_mesh, "Add_face")
        add_face.set("name", os.path.basename(file).split('.')[0])
        face_file_path = ET.SubElement(add_face, "Face_file_path")
        face_file_path.text = mesh_complete_sher + '/mesh-surfaces/' + os.path.basename(file)
    
    add_wall = ET.SubElement(add_mesh, "Add_face")
    add_wall.set("name", "wall")
    wall_file_path = ET.SubElement(add_wall, "Face_file_path")
    wall_file_path.text = os.path.join(mesh_complete_sher, 'walls_combined.vtp')
    # add equation
    add_eqn = ET.SubElement(svfsifile, "Add_equation")
    add_eqn.set("type", "fluid")
    coupled = ET.SubElement(add_eqn, "Coupled")
    coupled.text = "1"
    min_iterations = ET.SubElement(add_eqn, "Min_iterations")
    min_iterations.text = "3"
    max_iterations = ET.SubElement(add_eqn, "Max_iterations")
    max_iterations.text = "10"
    tolerance = ET.SubElement(add_eqn, "Tolerance")
    tolerance.text = "1e-3"
    backflow_stab = ET.SubElement(add_eqn, "Backflow_stabilization_coefficient")
    backflow_stab.text = "0.2"
    density = ET.SubElement(add_eqn, "Density")
    density.text = "1.06"
    viscosity = ET.SubElement(add_eqn, "Viscosity", {"model": "Constant"})
    value = ET.SubElement(viscosity, "Value")
    value.text = "0.04"
    output = ET.SubElement(add_eqn, "Output", {"type": "Spatial"})
    velocity = ET.SubElement(output, "Velocity")
    velocity.text = "true"
    pressure = ET.SubElement(output, "Pressure")
    pressure.text = "true"
    traction = ET.SubElement(output, "Traction")
    traction.text = "true"
    wss = ET.SubElement(output, "WSS")
    wss.text = "true"
    vorticity = ET.SubElement(output, "Vorticity")
    vorticity.text = "true"
    divergence = ET.SubElement(output, "Divergence")
    divergence.text = "true"
    ls = ET.SubElement(add_eqn, "LS", {"type": "NS"})
    linear_algebra = ET.SubElement(ls, "Linear_algebra", {"type": "fsils"})
    preconditioner = ET.SubElement(linear_algebra, "Preconditioner")
    preconditioner.text = "fsils"
    ls_max_iterations = ET.SubElement(ls, "Max_iterations")
    ls_max_iterations.text = "10"
    ns_gm_max_iterations = ET.SubElement(ls, "NS_GM_max_iterations")
    ns_gm_max_iterations.text = "10"
    ns_cg_max_iterations = ET.SubElement(ls, "NS_CG_max_iterations")
    ns_cg_max_iterations.text = "500"
    ls_tolerance = ET.SubElement(ls, "Tolerance")
    ls_tolerance.text = "1e-3"
    ns_gm_tolerance = ET.SubElement(ls, "NS_GM_tolerance")
    ns_gm_tolerance.text = "1e-3"
    ns_cg_tolerance = ET.SubElement(ls, "NS_CG_tolerance")
    ns_cg_tolerance.text = "1e-3"
    krylov_space_dim = ET.SubElement(ls, "Krylov_space_dimension")
    krylov_space_dim.text = "50"

    # add boundary conditions
    add_bc = ET.SubElement(add_eqn, "Add_BC")
    add_bc.set("name", inlet_cap.split('.')[0])
    
    typ = ET.SubElement(add_bc, "Type")
    typ.text = "Dir"
    time_dep = ET.SubElement(add_bc, "Time_dependence")
    time_dep.text = "Unsteady"
    fpath_temp_vals = ET.SubElement(add_bc, "Temporal_values_file_path")
    fpath_temp_vals.text = sim_dir + "inflow_svFSI.flow"
    profile = ET.SubElement(add_bc, "Profile")
    profile.text = "Parabolic"

    

    
    for outlet in outlet_caps:
        add_bc = ET.SubElement(add_eqn, "Add_BC")
        add_bc.set("name", outlet.split('.')[0])
        typ = ET.SubElement(add_bc, "Type")
        typ.text = "Neu"
        time_dep = ET.SubElement(add_bc, "Time_dependence")
        time_dep.text = "Resistance"
        value = ET.SubElement(add_bc, "Value")
        value.text = "61.56"        

    # add wall bc
    add_wall_bc = ET.SubElement(add_eqn, "Add_BC")
    add_wall_bc.set("name", "wall")
    typ = ET.SubElement(add_wall_bc, "Type")
    typ.text = "Dir"
    time_dep = ET.SubElement(add_wall_bc, "Time_dependence")
    time_dep.text = "Steady"
    value = ET.SubElement(add_wall_bc, "Value")
    value.text = "0.0"
    # Create the XML tree
    tree = ET.ElementTree(svfsifile)
    ET.indent(tree.getroot())
    # def prettify(elem):
    #     """Return a pretty-printed XML string for the Element."""
    #     rough_string = ET.tostring(elem, 'utf-8')
    #     reparsed = xml.dom.minidom.parseString(rough_string)
    #     return reparsed.toprettyxml(indent="  ")
    
    # pretty_xml_str = prettify(svfsifile)
    # print(pretty_xml_str)
    # Write the XML to a file
    with open(os.path.join(file_dir, "svFSIplus.xml"), "wb") as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)

    
    num_time_steps = n_tsteps
    flow = f"{num_time_steps}    16\n"
    t = np.linspace(start = 0, stop = num_time_steps, num = num_time_steps)
    q = t*0
    for i in range(t.size):
        if i < 0.1 * t.size:
            q_fac = i/(0.1 * t.size)
        else:
            q_fac = 1

        #q[i] = q_fac * -2 * 0.04*5500/(1.06*0.28*2) #-1*0.5*200 #*3.14*1.0476766883**2#-1 * 85 * 2 / 3.4215284204218883
        q[i] = q_fac * -2 * 0.04*5500/(1.06*2*2)  #q_fac * -2 * 0.04*5500/(1.06*1.5*2) tree_dec

        flow = flow + "%1.5f    %1.3f\n" %(i*dt, q[i])
    f = open(file_dir + f"inflow_svFSI.flow", "w")
    f.write(flow)
    f.close()
    return

    # file_dir = "/Users/natalia/Desktop/cco_bifurcations/trees/geo_files/tree_80/"

# sim_dir = "/scratch/users/nrubio/synthetic_junctions/CCO/test/simulation_data35/"
# file_dir = "/Users/natalia/Desktop/cco_bifurcations/trees/geo_files/tree_80/"
sim_dir = "/scratch/users/nrubio/synthetic_junctions/CCO/tree_dec1/"
file_dir = "/Users/natalia/Desktop/cco_bifurcations/trees/geo_files/tree_dec1/"
write_svfsiplus_xml(file_dir, sim_dir)
