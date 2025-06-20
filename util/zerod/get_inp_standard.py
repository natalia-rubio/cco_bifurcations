import os
import sys
import numpy as np

def write_standard0d_input_generator_file(tree_dict):
   input_file = f"import os\n\
from pathlib import Path\n\
import sv\n\
import sys\n\
import vtk\n\
\n\
## Create a ROM simulation.\n\
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
model_params.name = '{tree_dict['tree_name']}'\n\
model_params.inlet_face_names = ['{tree_dict['inlet_cap']}']\n\
model_params.outlet_face_names = {tree_dict['outlet_cap_list']}\n\
model_params.centerlines_file_name = 'trees/geo_files/{tree_dict['tree_name_base']}/{tree_dict['tree_name_base']}_original/centerlines/centerlines.vtp'\n\
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
bcs = params.BoundaryConditions()\n\
bcs.add_velocities(face_name='{tree_dict['inlet_cap']}', file_name='trees/standard0d_input_file_generators/{tree_dict['tree_name']}_inflow_0D.flow')\n"

   for outlet_cap in tree_dict['outlet_cap_list']:
      input_file += f"bcs.add_resistance(face_name='{outlet_cap}', resistance=61.56)\n"

   input_file += f"solution_params = params.Solution()\n\
solution_params.time_step = {tree_dict['dt']}\n\
solution_params.num_time_steps = {tree_dict['num_time_steps']}\n\
\n\
## Write a 1D solver input file.\n\
output_dir = str('trees/zerod_input/standard/' + '{tree_dict['tree_name_base']}/{tree_dict['tree_name']}')\n\
if not os.path.exists(output_dir):\n\
      os.makedirs(output_dir)\n\
rom_simulation.write_input_file(model_order=0, model=model_params, mesh=mesh_params, fluid=fluid_props, material=material, boundary_conditions=bcs, solution=solution_params, directory=output_dir)"

   if not os.path.exists(f"trees/standard0d_input_file_generators"):
      os.makedirs(f"trees/standard0d_input_file_generators")
   f = open(f"trees/standard0d_input_file_generators/{tree_dict["tree_name"]}_standard0d_input_file_generator.py", "w")
   f.write(input_file)
   f.close()

   flow = f""
   t = np.linspace(start = 0, stop = tree_dict["num_time_steps"], num = tree_dict["num_time_steps"])
   q = t*0
   for i in range(t.size):
      q[i] = 1 * tree_dict["inflow"]

      flow = flow + "%1.5f    %1.3f\n" %(i*tree_dict["dt"], q[i])
   f = open(f"trees/standard0d_input_file_generators/{tree_dict["tree_name"]}_inflow_0D.flow", "w")
   f.write(flow)
   f.close()
   return

if __name__ == "__main__":
   tree_name = sys.argv[1]
   inflow = sys.argv[2]
   tree_name_split = tree_name.split("_")
   tree_name_base = "_".join(tree_name_split[0:2])

   inlet_cap = "cap_" + os.listdir(f'trees/geo_files/{tree_name_base}/{tree_name_base}_original/mesh-complete/inlet_cap')[0]+".vtp"
   caps = os.listdir(f"trees/geo_files/{tree_name_base}/{tree_name_base}_original/mesh-complete/mesh-surfaces")
   outlet_caps = []
   for cap in caps:
      if cap==inlet_cap:
         continue
      outlet_caps.append(cap)


   tree_dict = {"tree_name": tree_name, 
               "tree_name_base": tree_name_base,
               "inlet_cap": inlet_cap, 
               "outlet_cap_list": outlet_caps,
               "num_time_steps": 10,
               "dt": 0.1,
               "inflow": inflow}

   write_standard0d_input_generator_file(tree_dict)
   os.system(f"/Applications/SimVascular.app/Contents/Resources/simvascular --python -- trees/standard0d_input_file_generators/{tree_name}_standard0d_input_file_generator.py")
   os.system(f"sed -i -e 's/internal_junction/NORMAL_JUNCTION/g' trees/zerod_input/standard/{tree_name_base}/{tree_name}/solver_0d.json")
