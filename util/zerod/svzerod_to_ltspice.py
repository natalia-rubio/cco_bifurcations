from PySpice.Spice.Netlist import Circuit # for creating circuits
from PySpice.Unit import *                # for using units
from collections import defaultdict
import json
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging(logging_level=Logging.logging.ERROR)
import pdb
import matplotlib

# matplotlib.rcParams.update({'font.size': 14,
#                             'text.usetex': True,      # use TeX backend
#                             'mathtext.fontset': 'cm', # computer modern
#                             'mathtext.rm': 'serif'})
tree_name = "tree_80"
with open(f'trees/zerod_input_sv_RR_inlet/tree_80_full/solver_0d.json') as json_file:
    input_file = json.load(json_file)


circuit = Circuit(tree_name)
# Steady inflow
#circuit.I('inflow',  circuit.gnd, 'branch0_seg0_in', input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]) 
circuit.I('inflow',  circuit.gnd, 'branch0_seg0_in', input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]) 
vessel_dict = defaultdict(str)
for vessel in input_file["vessels"]:
    v_id = vessel["vessel_id"]
    v_name = vessel["vessel_name"]
    #seg_id = int(v_name.split("_")[-1][3:])
    vessel_dict[v_id] = v_name

    circuit.R(f'R_lin_{v_name}', f'{v_name}_in', f'{v_name}_post_R_lin', vessel["zero_d_element_values"]["R_poiseuille"])
    #circuit.R(f'R_lin_{v_name}', f'{v_name}_in', f'{v_name}_out', abs(vessel["zero_d_element_values"]["R_poiseuille"]))

    circuit.B(f'R_sten_{v_name}', f'{v_name}_post_R_lin', f'{v_name}_post_R_sten', 
                v=f'{vessel["zero_d_element_values"]["stenosis_coefficient"]}*abs(i(BR_quad_{v_name}))*i(BR_quad_{v_name})')
    
    circuit.B(f'R_quad_{v_name}', f'{v_name}_post_R_sten', f'{v_name}_post_R_quad', 
                v=f'{vessel["zero_d_element_values"]["stenosis_coefficient"]}*i(BR_quad_{v_name})*i(BR_quad_{v_name})')

    circuit.C(f'C_{v_name}', f'{v_name}_post_R_quad', circuit.gnd, vessel["zero_d_element_values"]["C"])

    circuit.L(f'L_{v_name}', f'{v_name}_post_R_quad', f'{v_name}_out', vessel["zero_d_element_values"]["L"])
    # Add resistance BC if applicable
    if "boundary_conditions" in vessel.keys():
        if "outlet" in vessel["boundary_conditions"].keys():
            circuit.R(f"{vessel['boundary_conditions']['outlet']}", f'{v_name}_out', circuit.gnd, 61.56)

for junction in input_file["junctions"]:
    j_name = junction["junction_name"]
    inlet_vessel = junction["inlet_vessels"][0]
    inlet_node = f"{vessel_dict[inlet_vessel]}_out"
    outlet_nodes = [f"{vessel_dict[outlet_vessel]}_in" for outlet_vessel in junction["outlet_vessels"]]
    # Blood vessel junction
    if junction["junction_type"] == "BloodVesselJunction":
        for i, outlet_vessel in enumerate(junction["outlet_vessels"]):

            circuit.R(f'R_lin_{j_name}_{outlet_vessel}', inlet_node, f'{j_name}_{outlet_vessel}_post_R_lin', junction["junction_values"]["R_poiseuille"][i])
            #circuit.R(f'R_lin_{j_name}_{outlet_vessel}', inlet_node, outlet_nodes[i], abs(junction["junction_values"]["R_poiseuille"][i]))

            circuit.B(f'R_sten_{j_name}_{outlet_vessel}', f'{j_name}_{outlet_vessel}_post_R_lin', f'{j_name}_{outlet_vessel}_post_R_sten', 
                    v=f'{junction["junction_values"]["stenosis_coefficient"][i]}*abs(i(BR_quad_{j_name}_{outlet_vessel}))*i(BR_quad_{j_name}_{outlet_vessel})')
            circuit.B(f'R_quad_{j_name}_{outlet_vessel}', f'{j_name}_{outlet_vessel}_post_R_sten', f'{j_name}_{outlet_vessel}_post_R_quad', 
                    v=f'{junction["junction_values"]["stenosis_coefficient"][i]}*abs(i(BR_quad_{j_name}_{outlet_vessel}))*i(BR_quad_{j_name}_{outlet_vessel})')
            # if junction["junction_values"]["C"][i] != 0:
            #     circuit.C(f'C_{j_name}_{outlet_vessel}', f'{j_name}_{outlet_vessel}_post_R_quad', circuit.gnd, abs(junction["junction_values"]["C"][i]))
            circuit.L(f'L_{j_name}_{outlet_vessel}', f'{j_name}_{outlet_vessel}_post_R_quad', outlet_nodes[i], junction["junction_values"]["L"][i])
    # Bifurcation junction
    elif junction["junction_type"] == "NORMAL_JUNCTION":
        for i, outlet_vessel in enumerate(junction["outlet_vessels"]):
            circuit.B(f"wire_{inlet_vessel}_{outlet_vessel}", inlet_node, outlet_nodes[i], v="0")
print(circuit)
simulator = circuit.simulator()
simulator.save_currents = True
analysis = simulator.operating_point()
for node in analysis.nodes.values():
    print(f"{str(node)}: {float(node)}V")
for param in analysis.internal_parameters.values():
    print(f"{str(param)}: {float(param)}A")

# export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/lib