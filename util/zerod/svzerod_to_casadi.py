
from collections import defaultdict
import json
import pdb
import matplotlib
import casadi
import pandas as pd
import os
# Solve a zerod vascular flow with CasADi

# Load in svZeroDSolver input file
inflow = 344.655
tree_name = "tree_20"
junction_mode = "RR"
with open(f'trees/zerod_input_sv_{junction_mode}/{tree_name}_full/solver_0d.json') as json_file:
    input_file = json.load(json_file)
num_vessels = len(input_file["vessels"])
print(f"Number of vessels: {num_vessels}")
num_junctions = len(input_file["junctions"])
print(f"Number of junctions: {num_junctions}")
num_BCs = len(input_file["boundary_conditions"])
print(f"Number of boundary conditions: {num_BCs}")
print(f"Number of unknowns: {num_vessels*8}")
print("\n ---------------- \n")

vessel_constraint_counter = 0
junction_constraint_counter = 0
BC_constraint_counter = 0
SS_constraint_counter = 0

# Create a CasADi Opti object
opti = casadi.Opti()

# Decision variables
Q_in = opti.variable(num_vessels)
Q_out = opti.variable(num_vessels)
Q_in_dt = opti.variable(num_vessels)
Q_out_dt = opti.variable(num_vessels)

P_in = opti.variable(num_vessels)
P_out = opti.variable(num_vessels)
P_in_dt = opti.variable(num_vessels)
P_out_dt = opti.variable(num_vessels)


inflow_extractors = opti.parameter(num_vessels, num_junctions)
opti.set_value(inflow_extractors, 0)
outflow_extractors = opti.parameter(num_vessels, num_junctions)
opti.set_value(outflow_extractors, 0)

# Dictionary to organize vessel information (needing for wiring junctions)
vessel_dict = defaultdict(dict)

for i, vessel in enumerate(input_file["vessels"]):

    vessel_dict[vessel["vessel_id"]]["v_ind"] = i
    vessel_dict[vessel["vessel_id"]]["v_name"] = vessel["vessel_name"]

    R_lin = vessel["zero_d_element_values"]["R_poiseuille"]
    R_sten = vessel["zero_d_element_values"]["stenosis_coefficient"]
    if "pressure_recovery_coefficient" in vessel["zero_d_element_values"].keys():
        R_quad = vessel["zero_d_element_values"]["pressure_recovery_coefficient"]
    else:
        R_quad = 0
    C = vessel["zero_d_element_values"]["C"]
    L = vessel["zero_d_element_values"]["L"]

    # Pressure equation
    opti.subject_to(
        P_in[i] +
        - P_out[i] +
        - (R_lin + R_sten * (10**-2 + Q_in[i]**2)**0.5 + R_quad * Q_in[i]) * Q_in[i] + # abs removed
        - L * Q_out_dt[i] == 0    
    )
    vessel_constraint_counter += 1
    
    # Flow equation
    opti.subject_to(
        Q_in[i] + 
        - Q_out[i] + 
        - C * P_in_dt[i] +
        C * (R_lin + 2*R_sten*(10**-10 + Q_in[i]**2)**0.5  + 2*R_quad*Q_in[i])*Q_in_dt[i] == 0 # abs removed
    )
    vessel_constraint_counter += 1
    
    # Outlet boundary conditions
    if "boundary_conditions" in vessel.keys():
        if "outlet" in vessel["boundary_conditions"].keys():
            opti.subject_to(P_out[i] - Q_out[i] * 61.56 == 0)
            BC_constraint_counter += 1
    
    # Inlet boundary conditions
        if "inlet" in vessel["boundary_conditions"].keys():
            opti.subject_to(Q_in[i] == inflow)#344.655)
            BC_constraint_counter += 1
            
for i, junction in enumerate(input_file["junctions"]):
    j_name = junction["junction_name"]
    inlet_vessel_id = junction["inlet_vessels"][0]
    inlet_vessel_ind = vessel_dict[inlet_vessel_id]["v_ind"]
    outlet_vessel_inds = [vessel_dict[outlet_vessel]["v_ind"] for outlet_vessel in junction["outlet_vessels"]]

    #Q_sum[i] = Q_out[inlet_vessel_ind]

    for j, outlet_vessel_ind in enumerate(outlet_vessel_inds):

        if junction["junction_type"] == "BloodVesselJunction":

            R_lin = junction["junction_values"]["R_poiseuille"][j]
            R_sten = junction["junction_values"]["stenosis_coefficient"][j]
            if "pressure_recovery_coefficient" in junction["junction_values"].keys():
                R_quad = junction["junction_values"]["pressure_recovery_coefficient"][j]
            else:
                R_quad = 0
            L = junction["junction_values"]["L"][j]
            C = 0

            # Pressure equations
            opti.subject_to(
                P_out[inlet_vessel_ind] +
                - P_in[outlet_vessel_ind] +
                - (R_lin + R_sten * (10**-2 + Q_in[outlet_vessel_ind]**2)**0.5 + R_quad * Q_in[outlet_vessel_ind]) * Q_in[outlet_vessel_ind] + # abs removed
                - L * Q_in_dt[outlet_vessel_ind]
                == 0
            )
            junction_constraint_counter += 1

        elif junction["junction_type"] == "NORMAL_JUNCTION":

            opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] == 0)
            junction_constraint_counter += 1

        
        opti.set_value(outflow_extractors[outlet_vessel_ind, i], -1)
        #Q_sum[i] += -Q_in[outlet_vessel_ind]
        # Flow equation
    #opti.subject_to(Q_sum[i] == 0)
    #pdb.set_trace()
    opti.set_value(inflow_extractors[inlet_vessel_ind, i], 1)
    opti.subject_to(Q_out.T@inflow_extractors[:,i] + Q_in.T@outflow_extractors[:,i] == 0)
    junction_constraint_counter += 1

# Enforce steady state
steady_state = True
if steady_state:
    opti.subject_to(casadi.vec(Q_in_dt)     == 0)
    opti.subject_to(casadi.vec(Q_out_dt)    == 0)
    opti.subject_to(casadi.vec(P_in_dt)     == 0)
    opti.subject_to(casadi.vec(P_out_dt)    == 0)
    SS_constraint_counter += 4 * num_vessels

# Enforce positive flows
positive_flows = False
if positive_flows:
    opti.subject_to(casadi.vec(Q_in)  >= 0)
    opti.subject_to(casadi.vec(Q_out) >= 0)

print(f"Number of vessel constraints: {vessel_constraint_counter}")
print(f"Number of junction constraints: {junction_constraint_counter}")
print(f"Number of boundary condition constraints: {BC_constraint_counter}")
print(f"Number of steady state constraints: {SS_constraint_counter}")
print(f"Total number of constraints: {vessel_constraint_counter + junction_constraint_counter + BC_constraint_counter + SS_constraint_counter}")
# Solve NLP with IPOPT
opti.minimize(0) # Dummy objective
opti.solver('ipopt')
#pdb.set_trace()
sol = opti.solve()


# Casadi solution to Pandas df
df = pd.DataFrame(columns=['name', 'time','flow_in', 'flow_out', 'pressure_in', 'pressure_out'])
for i, vessel in enumerate(vessel_dict.keys()):
    # pdb.set_trace()
    df.loc[i] = [vessel_dict[vessel]["v_name"],
                    1, 
                    sol.value(Q_in)[vessel_dict[vessel]["v_ind"]],
                    sol.value(Q_out)[vessel_dict[vessel]["v_ind"]],
                    sol.value(P_in)[vessel_dict[vessel]["v_ind"]],
                    sol.value(P_out)[vessel_dict[vessel]["v_ind"]]]
if not os.path.exists(f'trees/zerod_output_sv_{junction_mode}'):
    os.makedirs(f'trees/zerod_output_sv_{junction_mode}/tree_80_full')
df.to_csv(f'trees/zerod_output_sv_{junction_mode}/tree_80_full/sol_casadi_{inflow}.csv', index=False)

pdb.set_trace()