
from collections import defaultdict
import json
import pdb
import matplotlib
import casadi
import pandas as pd
import os
import sys
# Solve a zerod vascular flow with CasADi

# Load in svZeroDSolver input file
def solve_casadi_single(tree_name, junction_mode, sol_prev = None, coef_factor = 1.0):
    
    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    flow_mag = tree_name_split[-1]

    with open(f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json') as json_file:
        input_file = json.load(json_file)
    num_vessels = len(input_file["vessels"])
    num_junctions = len(input_file["junctions"])
    num_BCs = len(input_file["boundary_conditions"])

    vessel_constraint_counter = 0
    junction_constraint_counter = 0
    BC_constraint_counter = 0
    SS_constraint_counter = 0

    # Create a CasADi Opti object
    opti = casadi.Opti()
    # Objective to minimize (squared residuals of the vessel and junction pressure equations)
    objective = 0

    # Decision variables
    Q_in = opti.variable(num_vessels)
    Q_out = opti.variable(num_vessels)
    Q_in_dt = opti.variable(num_vessels)
    Q_out_dt = opti.variable(num_vessels)

    P_in = opti.variable(num_vessels)
    P_out = opti.variable(num_vessels)
    P_in_dt = opti.variable(num_vessels)
    P_out_dt = opti.variable(num_vessels)

    if sol_prev is not None:
        # Set initial guess for decision variables
        opti.set_initial(Q_in, sol_prev.value(Q_in))
        opti.set_initial(Q_out, sol_prev.value(Q_out))
        opti.set_initial(Q_in_dt, sol_prev.value(Q_in_dt))
        opti.set_initial(Q_out_dt, sol_prev.value(Q_out_dt))

        opti.set_initial(P_in, sol_prev.value(P_in))
        opti.set_initial(P_out, sol_prev.value(P_out))
        opti.set_initial(P_in_dt, sol_prev.value(P_in_dt))
        opti.set_initial(P_out_dt, sol_prev.value(P_out_dt))


    inflow_extractors = opti.parameter(num_vessels, num_junctions)
    opti.set_value(inflow_extractors, 0)
    outflow_extractors = opti.parameter(num_vessels, num_junctions)
    opti.set_value(outflow_extractors, 0)

    # Dictionary to organize vessel information (needing for wiring junctions)
    vessel_dict = defaultdict(dict)

    bc_ind_dict = {}
    for ind, bc in enumerate(input_file["boundary_conditions"]):
        bc_ind_dict[bc["bc_name"]] = ind

    for i, vessel in enumerate(input_file["vessels"]):

        vessel_dict[vessel["vessel_id"]]["v_ind"] = i
        vessel_dict[vessel["vessel_id"]]["v_name"] = vessel["vessel_name"]

        R_lin = vessel["zero_d_element_values"]["R_poiseuille"]
        R_sten = vessel["zero_d_element_values"]["stenosis_coefficient"]
        if "pressure_recovery_coefficient" in vessel["zero_d_element_values"].keys():
            R_quad = vessel["zero_d_element_values"]["pressure_recovery_coefficient"]
        else:
            R_quad = 0
        C = vessel["zero_d_element_values"]["C"]*0
        L = vessel["zero_d_element_values"]["L"]*0

        if "branch0" in vessel["vessel_name"]:
            #print(f"Branch 0 vessel {vessel['vessel_name']} found, adding vessel equation to objective")
            
            objective += (
                P_in[i] +
                - P_out[i] +
                - (R_lin + R_sten * (10**-2 + Q_in[i]**2)**0.5 + R_quad * Q_in[i]) * Q_in[i] + # abs removed
                - L * Q_out_dt[i]  
                )**2
        else:
            objective += (
                P_out[i] +
                - P_in[i] 
            )**2
        vessel_constraint_counter += 1
        
        # Conservation of mass (to satisfy exactly)
        opti.subject_to(
            Q_in[i] + 
            - Q_out[i] + 
            - C * P_in_dt[i] +
            C * (R_lin + 2*R_sten*(10**-10 + Q_in[i]**2)**0.5  + 2*R_quad*Q_in[i])*Q_in_dt[i] == 0 # abs removed
        )
        vessel_constraint_counter += 1
        
        # Outlet boundary conditions (to satisfy exactly)
        if "boundary_conditions" in vessel.keys():
            if "outlet" in vessel["boundary_conditions"].keys():
                bc_name = vessel["boundary_conditions"]["outlet"]
                bc_ind = bc_ind_dict[bc_name]
                resistance = input_file["boundary_conditions"][bc_ind]["bc_values"]["R"]
                #opti.subject_to(P_out[i] - Q_out[i] * 61.56 == 0)
                opti.subject_to(P_out[i] - Q_out[i] * resistance == 0)
                BC_constraint_counter += 1
        
        # Inlet boundary conditions (to satisfy exactly)
            if "inlet" in vessel["boundary_conditions"].keys():
                
                inflow = input_file["boundary_conditions"][0]["bc_values"]["Q"][-1]
                opti.subject_to(Q_in[i] == inflow)
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
                L = junction["junction_values"]["L"][j] * coef_factor
                C = 0
                #pdb.set_trace()
                #Junction pressure equation residual (to minimize)
                objective += ((
                    P_out[inlet_vessel_ind] + # THIS IS THE INLET PRESSURE
                    - P_in[outlet_vessel_ind] +
                    - (R_lin + R_sten * (10**-2 + Q_in[outlet_vessel_ind]**2)**0.5 + R_quad * Q_in[outlet_vessel_ind]) * Q_in[outlet_vessel_ind] + # abs removed
                    - L * Q_in_dt[outlet_vessel_ind])/(1333*20)
                )**2

                junction_constraint_counter += 1
                
                enforce_pressure_loss = False
                if enforce_pressure_loss:
                    opti.subject_to(
                        P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] >= 0
                    )

                enforce_flow_splits = True
                if enforce_flow_splits:
                    # objective += (
                    #     (Q_in[outlet_vessel_ind] - junction["junction_values"]["flow_split"][j] *  Q_out[inlet_vessel_ind])**6
                    # )
                    opti.subject_to(
                        Q_in[outlet_vessel_ind] - junction["junction_values"]["flow_split"][j] *  Q_out[inlet_vessel_ind] == 0
                    )

            elif junction["junction_type"] == "NORMAL_JUNCTION":
                # Continuity of pressure (to satisfy exactly)
                opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] == 0)
                junction_constraint_counter += 1

            
            opti.set_value(outflow_extractors[outlet_vessel_ind, i], -1)


        # Conservation of mass
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

    # print(f"Number of vessel constraints: {vessel_constraint_counter}")
    # print(f"Number of junction constraints: {junction_constraint_counter}")
    # print(f"Number of boundary condition constraints: {BC_constraint_counter}")
    # print(f"Number of steady state constraints: {SS_constraint_counter}")
    # print(f"Total number of constraints: {vessel_constraint_counter + junction_constraint_counter + BC_constraint_counter + SS_constraint_counter}")
    # Solve NLP with IPOPT
    opti.minimize(objective) # Dummy objective
    opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
    opts = {}
    opti.solver('ipopt', opts)

    try:
        sol = opti.solve()
    except:
        opti.debug.value(objective)
        # print("Objective value: ", opti.debug.value(objective))
        opti.debug.value(Q_in)
        sol = opti.debug


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
    if not os.path.exists(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}')
    df.to_csv(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}/sol_casadi.csv', index=False)
    return (sol)



if __name__ == "__main__":
    #num_iters = 10
    tree_name = sys.argv[1]
    junction_mode = sys.argv[2]
    sol = solve_casadi_single(tree_name, junction_mode, sol_prev = None, coef_factor=1)
