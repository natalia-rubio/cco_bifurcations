import numpy as np

def write_job_steady(anatomy, set_type, geo_name, flow_name, flow_index, num_cores, num_time_steps):

    geo_job_script = f"#!/bin/bash\n\
#SBATCH --job-name={geo_name}_flow_sweep\n\
#SBATCH --partition=amarsden\n\
#SBATCH --output=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.o%j\n\
#SBATCH --error=/scratch/users/nrubio/job_scripts/{geo_name}_{flow_name}.e%j\n\
#SBATCH --time=00:90:00\n\
#SBATCH --mem=50000\n\
#SBATCH --nodes={int(num_cores/24)}\n\
#SBATCH --tasks-per-node=24\n\
\n\
export UCX_TLS=ib\n\
export PMIX_MCA_gds=hash\n\
export OMPI_MCA_btl_tcp_if_include=ib0\n\
F1 = /scratch/users/nrubio/synthetic_junctions/{anatomy}/{set_type}/{geo_name}/{flow_name}\n\
IMAGE_PATH = /scratch/users/nrubio/singularity_images/svFSIplus.sif\n\
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
source /home/users/nrubio/junctionenv/bin/activate\n\
mpirun -n {int(num_cores)} singularity run --bind $F1 $IMAGE_PATH \
/build-trilinos/svFSIplus-build/bin/svfsiplus svFSIplus.xml
python3 /home/users/nrubio/SV_scripts/junction_sim/check_convergence.py {geo_name} {flow_index} {anatomy} {set_type} {num_time_steps}\n\
kkrm -r $outdir"
    f = open(f"/scratch/users/nrubio/job_scripts/{geo_name[0]}_f{flow_index}.sh", "w")
    f.write(geo_job_script)
    f.close()
    return

def write_svfsi(anatomy, set_type, geo, flow_index, flow_params, cap_numbers, inlet_cap_number, num_time_steps, time_step_size):
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
