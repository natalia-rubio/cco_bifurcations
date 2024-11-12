import numpy as np

def write_job_steady(anatomy, set_type, geo_name, flow_name, flow_index, num_cores, num_time_steps):

    geo_job_script = f"#!/bin/bash\n\
# Name of your job\n\
#SBATCH --job-name={geo_name}_flow_sweep\n\
# Name of partition\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
# The walltime you require for your simulation\n\
#SBATCH --time=00:90:00\n\
# Amount of memory you require per node. The default is 4000 MB (or 4 GB) per node\n\
#SBATCH --mem=50000\n\
#SBATCH --nodes={int(num_cores/24)}\n\
#SBATCH --tasks-per-node=24\n\
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
source /home/users/nrubio/junctionenv/bin/activate\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpre.exe /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/{flow_name}_job.svpre\n\
echo 'Done with svPre.'\n\
conv=false\n\
conv_attempts=1\n\
indir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}/{num_cores}-procs_case'\n\
outdir='/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}'\n\
echo 'Launching Simulation'\n\
cd /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}\n\
mpirun -n {num_cores} /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svsolver-openmpi.exe {flow_name}_solver.inp\n\
echo 'Simulation completed'\n\
/home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpost.exe -start {num_time_steps-100} -stop {num_time_steps} -incr 100 -vtkcombo -indir $indir -outdir $outdir -vtu solution_flow_{flow_index}.vtu > /dev/null\n\
python3 /home/users/nrubio/SV_scripts/junction_sim/check_convergence.py {geo_name} {flow_index} {anatomy} {set_type} {num_time_steps}\n\
kkrm -r $outdir"
    f = open(f"/scratch/users/nrubio/job_scripts/{geo_name[0]}_f{flow_index}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return

# /home/groups/amarsden/svSolver-github/BuildWithMake/Bin/svpost.exe -start 0 -stop 300 -incr 1 -vtkcombo -indir 24-procs_case -outdir . -vtu solution_flow_3.vtu

def write_svpre_steady_pipe(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    svpre = f"mesh_and_adjncy_vtu /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.mesh.vtu\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.exterior.vtp 1\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp {inlet_cap_number}\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp {res_caps[0]}\n\
fluid_density 1.06\n\
fluid_viscosity 0.04\n\
initial_pressure 0\n\
initial_velocity 0.0001 {flow_params['vel_in']} 0.0001\n\
prescribed_velocities_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp\n\
bct_analytical_shape parabolic\n\
bct_period {num_time_steps*2*time_step_size}\n\
bct_point_number {2}\n\
bct_fourier_mode_number 10\n\
bct_create /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow\n\
bct_write_dat /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.dat\n\
bct_write_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.vtp\n\
pressure_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 0\n\
noslip_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/walls_combined.vtp\n\
write_geombc /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/geombc.dat.1\n\
write_restart /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/restart.0.1"

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_job.svpre", "w")
    f.write(svpre)
    f.close()
    return

def write_svpre_steady(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    svpre = f"mesh_and_adjncy_vtu /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.mesh.vtu\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.exterior.vtp 1\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp {inlet_cap_number}\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp {res_caps[0]}\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp {res_caps[1]}\n\
fluid_density 1.06\n\
fluid_viscosity 0.04\n\
initial_pressure 0\n\
initial_velocity 0.0001 {flow_params['vel_in']} 0.0001\n\
prescribed_velocities_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp\n\
bct_analytical_shape parabolic\n\
bct_period {2*num_time_steps*time_step_size}\n\
bct_point_number {num_time_steps} \n\
bct_fourier_mode_number 10\n\
bct_create /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow\n\
bct_write_dat /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.dat\n\
bct_write_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.vtp\n\
pressure_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 0\n\
pressure_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 0\n\
noslip_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/walls_combined.vtp\n\
write_geombc /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/geombc.dat.1\n\
write_restart /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/restart.0.1"
#read_all_variables_vtu /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/initial_soln.vtu\n\
    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_job.svpre", "w")
    f.write(svpre)
    f.close()
    return

def write_svpre_unsteady(anatomy, set_type, geo, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"unsteady"
    flow_index = "unsteady"
    svpre = f"mesh_and_adjncy_vtu /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.mesh.vtu\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-complete.exterior.vtp 1\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp {inlet_cap_number}\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp {res_caps[0]}\n\
set_surface_id_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp {res_caps[1]}\n\
fluid_density 1.06\n\
fluid_viscosity 0.04\n\
initial_pressure 0\n\
initial_velocity 0.0001 0.0001 0.0001\n\
prescribed_velocities_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp\n\
bct_analytical_shape parabolic\n\
bct_period {num_time_steps*time_step_size}\n\
bct_point_number {num_time_steps}\n\
bct_fourier_mode_number 10\n\
bct_create /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{inlet_cap_number}.vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow\n\
bct_write_dat /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.dat\n\
bct_write_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/bct.vtp\n\
pressure_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[0]}.vtp 0\n\
pressure_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/mesh-surfaces/cap_{res_caps[1]}.vtp 0\n\
noslip_vtp /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/mesh-complete/walls_combined.vtp\n\
write_geombc /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/geombc.dat.1\n\
write_restart /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/restart.0.1"

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_job.svpre", "w")

    f.write(svpre)
    f.close()
    return

def write_steady_inp_pipe(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    inp = f"Density: 1.06\n\
Viscosity: 0.04\n\
\n\
Number of Timesteps: {num_time_steps}\n\
Time Step Size: {time_step_size}\n\
\n\
Number of Timesteps between Restarts: 100\n\
Number of Force Surfaces: 1\n\
Surface ID's for Force Calculation: 1\n\
Force Calculation Method: Velocity Based\n\
Print Average Solution: True\n\
Print Error Indicators: False\n\
\n\
Time Varying Boundary Conditions From File: True\n\
\n\
Step Construction: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1\n\
\n\
Number of Resistance Surfaces: 0\n\
List of Resistance Surfaces: \n\
Resistance Values: \n\
\n\
Pressure Coupling: Implicit\n\
Number of Coupled Surfaces: 2\n\
\n\
Backflow Stabilization Coefficient: 0.2\n\
Residual Control: True\n\
Residual Criteria: 0.0001\n\
Minimum Required Iterations: 2\n\
svLS Type: NS\n\
Number of Krylov Vectors per GMRES Sweep: 10\n\
Number of Solves per Left-hand-side Formation: 1\n\
Tolerance on Momentum Equations: 0.0001\n\
Tolerance on Continuity Equations: 0.0001\n\
Tolerance on svLS NS Solver: 0.01\n\
Maximum Number of Iterations for svLS NS Solver: 10\n\
Maximum Number of Iterations for svLS Momentum Loop: 20\n\
Maximum Number of Iterations for svLS Continuity Loop: 400\n\
Time Integration Rule: Second Order\n\
Time Integration Rho Infinity: 0.5\n\
Flow Advection Form: Convective\n\
Quadrature Rule on Interior: 2\n\
Quadrature Rule on Boundary: 3"

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_solver.inp", "w")
    f.write(inp)
    f.close()
    return

def write_inp_steady(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"flow_{flow_index}"
    inp = f"Density: 1.06\n\
Viscosity: 0.04\n\
\n\
Number of Timesteps: {num_time_steps}\n\
Time Step Size: {time_step_size}\n\
\n\
Number of Timesteps between Restarts: 100\n\
Number of Force Surfaces: 1\n\
Surface ID's for Force Calculation: 1\n\
Force Calculation Method: Velocity Based\n\
Print Average Solution: True\n\
Print Error Indicators: False\n\
\n\
Time Varying Boundary Conditions From File: True\n\
\n\
Step Construction: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1\n\
\n\
Number of Resistance Surfaces: 2\n\
List of Resistance Surfaces: {res_caps[0]} {res_caps[1]}\n\
Resistance Values: {flow_params['res_1']} {flow_params['res_2']}\n\
\n\
Pressure Coupling: Implicit\n\
Number of Coupled Surfaces: 2\n\
\n\
Backflow Stabilization Coefficient: 0.2\n\
Residual Control: True\n\
Residual Criteria: 0.0001\n\
Minimum Required Iterations: 2\n\
svLS Type: NS\n\
Number of Krylov Vectors per GMRES Sweep: 10\n\
Number of Solves per Left-hand-side Formation: 1\n\
Tolerance on Momentum Equations: 0.001\n\
Tolerance on Continuity Equations: 0.001\n\
Tolerance on svLS NS Solver: 0.001\n\
Maximum Number of Iterations for svLS NS Solver: 10\n\
Maximum Number of Iterations for svLS Momentum Loop: 20\n\
Maximum Number of Iterations for svLS Continuity Loop: 400\n\
Time Integration Rule: Second Order\n\
Time Integration Rho Infinity: 0.5\n\
Flow Advection Form: Convective\n\
Quadrature Rule on Interior: 2\n\
Quadrature Rule on Boundary: 3"

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_solver.inp", "w")
    f.write(inp)
    f.close()
    return

def write_unsteady_inp(anatomy, set_type, geo, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
    res_caps = cap_numbers
    res_caps.remove(inlet_cap_number)
    flow_name = f"unsteady"
    flow_index = "unsteady"
    inp = f"Density: 1.06\n\
Viscosity: 0.04\n\
\n\
Number of Timesteps: {num_time_steps}\n\
Time Step Size: {time_step_size}\n\
\n\
Number of Timesteps between Restarts: 1\n\
Number of Force Surfaces: 1\n\
Surface ID's for Force Calculation: 1\n\
Force Calculation Method: Velocity Based\n\
Print Average Solution: True\n\
Print Error Indicators: False\n\
\n\
Time Varying Boundary Conditions From File: True\n\
\n\
Step Construction: 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1\n\
\n\
Number of Resistance Surfaces: 2\n\
List of Resistance Surfaces: {res_caps[0]} {res_caps[1]}\n\
Resistance Values: {flow_params['res_1']} {flow_params['res_2']}\n\
\n\
Pressure Coupling: Implicit\n\
Number of Coupled Surfaces: 2\n\
\n\
Backflow Stabilization Coefficient: 0.01\n\
Residual Control: True\n\
Residual Criteria: 0.001\n\
Minimum Required Iterations: 2\n\
svLS Type: NS\n\
Number of Krylov Vectors per GMRES Sweep: 10\n\
Number of Solves per Left-hand-side Formation: 1\n\
Tolerance on Momentum Equations: 0.01\n\
Tolerance on Continuity Equations: 0.01\n\
Tolerance on svLS NS Solver: 0.01\n\
Maximum Number of Iterations for svLS NS Solver: 10\n\
Maximum Number of Iterations for svLS Momentum Loop: 20\n\
Maximum Number of Iterations for svLS Continuity Loop: 400\n\
Time Integration Rule: Second Order\n\
Time Integration Rho Infinity: 0.5\n\
Flow Advection Form: Convective\n\
Quadrature Rule on Interior: 2\n\
Quadrature Rule on Boundary: 3"

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_name}_solver.inp", "w")
    f.write(inp)
    f.close()
    return

def write_flow_steady(anatomy, set_type, geo, flow_index, flow_amp, cap_number, num_time_steps, time_step_size):
    flow_name = f"flow_{flow_index}"
    flow = ""
    t = np.linspace(start = 0, stop = num_time_steps, num = num_time_steps)
    q = t*0
    for i in range(t.size):
        if i < 20:
            #q[i] = -1 * flow_amp * 0.5 * (1 - np.cos(np.pi * i / 20))
            q[i] = -1 * flow_amp
        else:
            q[i] = -1 * flow_amp

        flow = flow + "%1.5f %1.3f\n" %(i*time_step_size*2, q[i])
    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return

def write_unsteady_flow(anatomy, set_type, geo, flow_amp, cap_number, num_time_steps, time_step_size):
    flow_name = f"unsteady"
    flow_index = "unsteady"
    flow = ""
    t = np.linspace(start = 0, stop = 4*np.pi, num = num_time_steps)
    q = (flow_amp/2) * (np.cos(t)-1)
    for i in range(t.size):
        flow = flow + "%1.3f %1.3f\n" %(i*time_step_size, q[i])

    f = open(f"/scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo}/{flow_name}/{flow_index}.flow", "w")
    f.write(flow)
    f.close()
    return
