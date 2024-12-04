
input_file = f"import os\n\
from pathlib import Path\n\
import sv\n\
import sys\n\
import vtk\n\
\n\
tree_name = 'sample'\n\
## Set some directory paths. \n\
script_path = 'synthetic_tree'\n\
\n\
## Create a ROM simulation.\n\
input_dir = str(script_path + '/' + 'input')\n\
rom_simulation = sv.simulation.ROM()\n\
\n\
## Create ROM simulation parameters.\n\
params = sv.simulation.ROMParameters()\n\
\n\
## Mesh parameters.\n\
mesh_params = params.MeshParameters()\n\
\n\
## Model parameters.\n\
model_params = params.ModelParameters()\n\
model_params.name = 'synthetic_tree'\n\
model_params.inlet_face_names = ['cap_2' ]\n\
model_params.outlet_face_names = ['cap_4', 'cap_41', 'cap_42']\n\
model_params.centerlines_file_name = 'synthetic_tree/centerlines/centerline.vtp'\n\
\n\
## Fluid properties.\n\
fluid_props = params.FluidProperties()\n\
\n\
## Set wall properties.\n\
\n\
print('Set wall properties ...')\n\
material = params.WallProperties.OlufsenMaterial()\n\
\n\
## Set boundary conditions.\n\
bcs = params.BoundaryConditions()\n\"

input_file += f"#bcs.add_resistance(face_name='outlet', resistance=1333)\n\
bcs.add_velocities(face_name='cap_2', file_name='synthetic_tree/inflow.flow')\n\
bcs.add_resistance(face_name='cap_4', resistance=2500.0)\n\
bcs.add_resistance(face_name='cap_41', resistance=2500.0)\n\
bcs.add_resistance(face_name='cap_42', resistance=2500.0)\n\"

## Set solution parameters. 

solution_params = params.Solution()
solution_params.time_step = 0.2
solution_params.num_time_steps = 5

## Write a 1D solver input file.
output_dir = str("trees/zerod_input_standard/" + tree_name)
rom_simulation.write_input_file(model_order=0, model=model_params, mesh=mesh_params, fluid=fluid_props, material=material, boundary_conditions=bcs, solution=solution_params, directory=output_dir)"

