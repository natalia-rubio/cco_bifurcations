from PySpice.Spice.Netlist import Circuit # for creating circuits
from PySpice.Unit import *                # for using units


import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging(logging_level=Logging.logging.ERROR)

import matplotlib

# matplotlib.rcParams.update({'font.size': 14,
#                             'text.usetex': True,      # use TeX backend
#                             'mathtext.fontset': 'cm', # computer modern
#                             'mathtext.rm': 'serif'})

circuit = Circuit('tree_name')

circuit.I('inflow',  circuit.gnd, 'v_1_in', 5)
circuit.R('R_lin0', 'v_1_in', 'v_1_res_lin', 0)
circuit.B('R_quad0', 'v_1_res_lin', 'v_1_res_quad', v="1*i(BR_quad0)*abs(i(BR_quad0))")
#circuit.B('R_quad0', 'v_1_res_lin', 'v_1_res_quad', v="0")
circuit.C('C_0', 'v_1_res_quad', circuit.gnd, 2)
circuit.L('L_0', 'v_1_res_quad', 'v_1_res_ind', 2) # you can also pass the actual value directly
circuit.B('wire', 'v_1_res_ind', 'v_1_bc', v="0")
circuit.R('R_1', 'v_1_bc', circuit.gnd, 1)
print(circuit)
simulator = circuit.simulator()
print(simulator)
analysis = simulator.operating_point()
analysis.nodes.keys()
for node in analysis.nodes.values():
    print(f"{str(node)}: {float(node)}V")
for branch in analysis.branches.values():
    print(f"{str(branch)}: {float(branch)}A")

simulator.save_currents = True
analysis = simulator.operating_point()
for param in analysis.internal_parameters.values():
    print(f"{str(param)}: {float(param)}A")

# export DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH:/opt/homebrew/lib