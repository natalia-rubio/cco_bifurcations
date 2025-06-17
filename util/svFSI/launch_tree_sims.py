from util.sherlock_util import *
# from util.svFSI.util.projection import * For solution initialization
tree_name = sys.argv[1]

dir = f"/scratch/users/nrubio/synthetic_junctions/CCO/{tree_name}"
flow_names = os.listdir(dir); flow_names.sort()

for flow_name in flow_names:

    flow_name_split = flow_name.split("_")
    flow_mag = flow_name_split[-1]

    if os.path.exists(f"{dir}/{flow_name}/24-procs/{tree_name}_{flow_mag}_800.vtu"):
        continue
    if os.path.exists(f"{dir}/{flow_name}/48-procs/{tree_name}_{flow_mag}_800.vtu"):
        continue
    if os.path.exists(f"{dir}/{flow_name}/72-procs/{tree_name}_{flow_mag}_800.vtu"):
        continue
    if os.path.exists(f"{dir}/{flow_name}/96-procs/{tree_name}_{flow_mag}_800.vtu"):
        continue

    os.system(f"cd {dir}/{flow_name} && sbatch svFSI_{flow_name}.sh")
    print(f"Started job for {flow_name}")