import numpy as np

def write_job_steady(anatomy, set_type, geo_name, flow_name, flow_index, num_cores, num_time_steps, inc):

    geo_job_script = f"#!/bin/bash\n\
#SBATCH --job-name={geo_name}_{flow_name}\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
#SBATCH --time=00:30:00\n\
#SBATCH --mem=50000\n\
#SBATCH --nodes={int(num_cores/24)}\n\
#SBATCH --tasks-per-node=24\n\
\n\
export UCX_TLS=ib\n\
export PMIX_MCA_gds=hash\n\
export OMPI_MCA_btl_tcp_if_include=ib0\n\
export F1='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}'\n\
export IMAGE_PATH='/home/users/nrubio/SV_scripts/solver_latest.sif'\n\
\n\
# Load Modules\n\
module purge\n\
module load openmpi\n\
module load openblas\n\
module load system\n\
module load x11\n\
module load mesa\n\
module load viz\n\
module load gcc\n\
module load valgrind\n\
module load python/3.9.0\n\
module load py-numpy/1.20.3_py39\n\
module load py-scipy/1.6.3_py39\n\
module load py-scikit-learn/1.0.2_py39\n\
module load gcc/10.1.0\n\
# Name of the executable you want to run\n\
\n\
singularity exec $IMAGE_PATH bash -c 'mpirun --mca opal_cuda_support 0 -n {int(num_cores)} /build-trilinos/svFSIplus-build/bin/svfsiplus $F1/{geo_name}_f{flow_index}.xml'\n\
python3 /home/users/nrubio/SV_scripts/svFSI/check_convergence.py {geo_name} {flow_index} {anatomy} {set_type} {num_time_steps} {inc}\n\
"
    f = open(f"/scratch/users/nrubio/job_scripts/{geo_name}_f{flow_index}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return

def write_svfsi(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size, inc):
    geo_dir = f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}"
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    svfsi = f"<?xml version='1.0' encoding='UTF-8' ?>\n\
    <svFSIFile version='0.1'>\n\
    \n\
    <GeneralSimulationParameters>\n\
    \n\
    <Continue_previous_simulation> false </Continue_previous_simulation>\n\
    <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions> \n\
    <Number_of_time_steps> {num_time_steps} </Number_of_time_steps> \n\
    <Time_step_size> {time_step_size} </Time_step_size> \n\
    <Spectral_radius_of_infinite_time_step> 0.50 </Spectral_radius_of_infinite_time_step> \n\
    <Searched_file_name_to_trigger_stop> STOP_SIM </Searched_file_name_to_trigger_stop> \n\
    \n\
    <Save_results_to_VTK_format> 1 </Save_results_to_VTK_format> \n\
    <Name_prefix_of_saved_VTK_files> solution_flow_{flow_index} </Name_prefix_of_saved_VTK_files> \n\
    <Increment_in_saving_VTK_files> {inc} </Increment_in_saving_VTK_files> \n\
    <Save_results_in_folder> {geo_dir}/flow_{flow_index} </Save_results_in_folder> \n\
    <Start_saving_after_time_step> {num_time_steps-inc} </Start_saving_after_time_step> \n\
    \n\
    <Increment_in_saving_restart_files> {inc} </Increment_in_saving_restart_files> \n\
    <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format> \n\
    \n\
    <Verbose> 1 </Verbose> \n\
    <Warning> 0 </Warning> \n\
    <Debug> 0 </Debug> \n\
    \n\
    </GeneralSimulationParameters>\n\
    \n\
    <Add_mesh name='msh' > \n\
    \n\
    <Mesh_file_path> {geo_dir}/mesh-complete/mesh-complete.mesh.vtu </Mesh_file_path>\n\
    \n\
    <Add_face name='inlet'>\n\
        <Face_file_path> {geo_dir}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp </Face_file_path>\n\
    </Add_face>\n\
    \n\
    <Add_face name='outlet0'>\n\
        <Face_file_path> {geo_dir}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp </Face_file_path>\n\
    </Add_face>\n\
    \n\
    <Add_face name='outlet1'>\n\
        <Face_file_path> {geo_dir}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp </Face_file_path>\n\
    </Add_face>\n\
    \n\
    <Add_face name='walls'>\n\
        <Face_file_path> {geo_dir}/mesh-complete/walls_combined.vtp </Face_file_path>\n\
    </Add_face>\n\
    \n\
    </Add_mesh>\n\
    \n\
    <Add_equation type='fluid' > \n\
        <Coupled> true </Coupled>\n\
        <Min_iterations> 2 </Min_iterations>  \n\
        <Max_iterations> 12 </Max_iterations> \n\
        <Tolerance> 1e-7 </Tolerance> \n\
        <Backflow_stabilization_coefficient> 0.2 </Backflow_stabilization_coefficient> \n\
        \n\
        <Density> 1.06 </Density> \n\
        <Viscosity model='Constant' >\n\
            <Value> 0.04 </Value>\n\
        </Viscosity>\n\
        \n\
        <Output type='Spatial' >\n\
            <Velocity> true </Velocity>\n\
            <Pressure> true </Pressure>\n\
            <Traction> true </Traction>\n\
            <Vorticity> true</Vorticity>\n\
            <Divergence> true</Divergence>\n\
            <WSS> true </WSS>\n\
        </Output>\n\
        \n\
        <LS type='NS' >\n\
            <Linear_algebra type='fsils' >\n\
                <Preconditioner> fsils </Preconditioner>\n\
            </Linear_algebra>\n\
            <Max_iterations> 15 </Max_iterations>\n\
            <NS_GM_max_iterations> 10 </NS_GM_max_iterations>\n\
            <NS_CG_max_iterations> 300 </NS_CG_max_iterations>\n\
            <Tolerance> 1e-3 </Tolerance>\n\
            <NS_GM_tolerance> 1e-3 </NS_GM_tolerance>\n\
            <NS_CG_tolerance> 1e-3 </NS_CG_tolerance>\n\
            <Absolute_tolerance> 1e-17 </Absolute_tolerance>\n\
            <Krylov_space_dimension> 250 </Krylov_space_dimension>\n\
        </LS>\n\
        \n\
        <Add_BC name='inlet' > \n\
            <Type> Dir </Type> \n\
            <Time_dependence> Unsteady </Time_dependence> \n\
            <Temporal_values_file_path> {geo_dir}/flow_{flow_index}/{flow_index}.flow </Temporal_values_file_path> \n\
            <Profile> Parabolic </Profile> \n\
            <Impose_flux> true </Impose_flux> \n\
        </Add_BC> \n\
        \n\
        <Add_BC name='outlet0' > \n\
            <Type> Neu </Type> \n\
            <Time_dependence> Resistance </Time_dependence> \n\
            <Value> 10000 </Value> \n\
        </Add_BC> \n\
        \n\
        <Add_BC name='outlet1' > \n\
            <Type> Neu </Type> \n\
            <Time_dependence> Resistance </Time_dependence> \n\
            <Value> 10000 </Value> \n\
        </Add_BC> \n\
        \n\
        <Add_BC name='walls' > \n\
            <Type> Dir </Type> \n\
            <Time_dependence> Steady </Time_dependence> \n\
            <Value> 0.0 </Value> \n\
        </Add_BC> \n\
        \n\
    </Add_equation>\n\
    \n\
    </svFSIFile>"
    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{geo}_f{flow_index}.xml", "w")
    f.write(svfsi)
    f.close()
    return


def write_flow_steady(anatomy, set_type, geo, flow_index, flow_amp, cap_number, num_time_steps, time_step_size):
    flow_name = f"flow_{flow_index}"
    flow = f"{int(num_time_steps)}    16\n"
    t = np.linspace(start = 0, stop = num_time_steps, num = num_time_steps)
    q = t*0
    for i in range(t.size):
        if i < 20:
            #q[i] = -1 * flow_amp * 0.5 * (1 - np.cos(np.pi * i / 20))
            q[i] = -1 * flow_amp
        else:
            q[i] = -1 * flow_amp

        flow = flow + "%1.5f    %1.3f\n" %(i*time_step_size*2, q[i])
    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return

def write_unsteady_flow(anatomy, set_type, geo, flow_amp, cap_number, num_time_steps, time_step_size):
    flow_name = f"unsteady"
    flow_index = "unsteady"
    flow = f"{int(num_time_steps)}    16\n"
    t = np.linspace(start = 0, stop = 4*np.pi, num = num_time_steps)
    q = (flow_amp/2) * (np.cos(t)-1)
    for i in range(t.size):
        flow = flow + "%1.3f    %1.3f\n" %(i*time_step_size, q[i])

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return
