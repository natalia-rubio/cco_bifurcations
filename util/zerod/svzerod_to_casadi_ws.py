
from collections import defaultdict
import json
import pdb
from unittest import result
import matplotlib
import casadi
import pandas as pd
import os
import sys

#from util.neural_net.nn_model import coef_loss
# Solve a zerod vascular flow with CasADi

# Load in svZeroDSolver input file
def solve_casadi_unsteady(time_step = 0, sol_prev = None, input_file = None, result_df = None):

    time = input_file["boundary_conditions"][0]["bc_values"]["t"][time_step]
    num_time_steps = len(input_file["boundary_conditions"][0]["bc_values"]["t"])
    dt = input_file["boundary_conditions"][0]["bc_values"]["t"][1] - input_file["boundary_conditions"][0]["bc_values"]["t"][0]
    inlet_Q = input_file["boundary_conditions"][0]["bc_values"]["Q"][time_step] * (time_step+1)/num_time_steps

    num_vessels = len(input_file["vessels"])
    # print(f"Number of vessels: {num_vessels}")
    num_junctions = len(input_file["junctions"])
    # print(f"Number of junctions: {num_junctions}")
    num_BCs = len(input_file["boundary_conditions"])
    # print(f"Number of boundary conditions: {num_BCs}")
    # print(f"Number of unknowns: {num_vessels*8}")
    # print("\n ---------------- \n")

    vessel_constraint_counter = 0
    junction_constraint_counter = 0
    BC_constraint_counter = 0
    SS_constraint_counter = 0
    coef_factor = 1

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
        opti.set_initial(Q_in, sol_prev["Q_in"])
        opti.set_initial(Q_out, sol_prev["Q_out"])
        opti.set_initial(Q_in_dt, sol_prev["Q_in_dt"])
        opti.set_initial(Q_out_dt, sol_prev["Q_out_dt"])

        opti.set_initial(P_in, sol_prev["P_in"])
        opti.set_initial(P_out, sol_prev["P_out"])
        opti.set_initial(P_in_dt, sol_prev["P_in_dt"])
        opti.set_initial(P_out_dt, sol_prev["P_out_dt"])


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
        C = vessel["zero_d_element_values"]["C"]
        L = vessel["zero_d_element_values"]["L"]

        if "branch0" in vessel["vessel_name"] or junction_mode == "standard":
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

        # objective += (
        #     P_in[i] +
        #     - P_out[i] +
        #     - (R_lin + R_sten * (10**-2 + Q_in[i]**2)**0.5 + R_quad * Q_in[i]) * Q_in[i] + # abs removed
        #     - L * Q_out_dt[i]  
        #     )**2
            
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
                opti.subject_to(Q_in[i] == inlet_Q)
                BC_constraint_counter += 1
        # if vessel["vessel_name"] == "branch32_seg0":
        #     pdb.set_trace()
                
    for i, junction in enumerate(input_file["junctions"]):
        j_name = junction["junction_name"]
        inlet_vessel_id = junction["inlet_vessels"][0]
        inlet_vessel_ind = vessel_dict[inlet_vessel_id]["v_ind"]
        outlet_vessel_inds = [vessel_dict[outlet_vessel]["v_ind"] for outlet_vessel in junction["outlet_vessels"]]

        #Q_sum[i] = Q_out[inlet_vessel_ind]

        for j, outlet_vessel_ind in enumerate(outlet_vessel_inds):

            if junction["junction_type"] == "BloodVesselJunction":

                R_lin = junction["junction_values"]["R_poiseuille"][j] * coef_factor
                R_sten = junction["junction_values"]["stenosis_coefficient"][j] * coef_factor
                if "pressure_recovery_coefficient" in junction["junction_values"].keys():
                    R_quad = junction["junction_values"]["pressure_recovery_coefficient"][j] * coef_factor
                else:
                    R_quad = 0


                L = junction["junction_values"]["L"][j] * coef_factor
                C = 0
                #pdb.set_trace()
                #print("Inductance: ", L)

                # Junction pressure equation residual (to minimize)
                objective += ((
                    P_out[inlet_vessel_ind] + # THIS IS THE INLET PRESSURE
                    - P_in[outlet_vessel_ind] +
                    - (R_lin + R_sten * (10**-2 + Q_in[outlet_vessel_ind]**2)**0.5 + R_quad * Q_in[outlet_vessel_ind]) * Q_in[outlet_vessel_ind] + # abs removed
                    - L * Q_in_dt[outlet_vessel_ind])/(1333 * 20)
                )**2
                junction_constraint_counter += 1
                
                enforce_pressure_loss = True
                if enforce_pressure_loss:
                    opti.subject_to(
                        P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] >= 0
                    )

                enforce_flow_splits = True
                if enforce_flow_splits:
                    # objective += (
                    #     100 * (Q_in[outlet_vessel_ind] - junction["junction_values"]["flow_split"][j] *  Q_out[inlet_vessel_ind])
                    # )
                    opti.subject_to(
                        Q_in[outlet_vessel_ind] - junction["junction_values"]["flow_split"][j] *  Q_out[inlet_vessel_ind] == 0
                    )

            elif junction["junction_type"] == "NORMAL_JUNCTION":
                # Continuity of pressure (to satisfy exactly)
                opti.subject_to(P_out[inlet_vessel_ind] - P_in[outlet_vessel_ind] == 0)
                junction_constraint_counter += 1

            
            opti.set_value(outflow_extractors[outlet_vessel_ind, i], -1)

            enforce_flow_splits = True

        # Conservation of mass
        opti.set_value(inflow_extractors[inlet_vessel_ind, i], 1)
        opti.subject_to(Q_out.T@inflow_extractors[:,i] + Q_in.T@outflow_extractors[:,i] == 0)
        junction_constraint_counter += 1

    # Enforce steady state
    steady = True
    if time_step == 0 or steady:
        opti.subject_to(casadi.vec(Q_in_dt)     == 0)
        opti.subject_to(casadi.vec(Q_out_dt)    == 0)
        opti.subject_to(casadi.vec(P_in_dt)     == 0)
        opti.subject_to(casadi.vec(P_out_dt)    == 0)
        SS_constraint_counter += 4 * num_vessels
    else:
        # Steady state conditions
        opti.subject_to(casadi.vec(Q_in_dt)     == (Q_in  - sol_prev["Q_in"])  / dt)
        opti.subject_to(casadi.vec(Q_out_dt)    == (Q_out - sol_prev["Q_out"]) / dt)
        opti.subject_to(casadi.vec(P_in_dt)     == (P_in  - sol_prev["P_in"])  / dt)
        opti.subject_to(casadi.vec(P_out_dt)    == (P_out - sol_prev["P_out"]) / dt)
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
    #opti.solver('ipopt')
    opts = {'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes'}
    #opts = {}
    opti.solver('ipopt', opts)
    try:
        sol = opti.solve()
    except:
        opti.debug.value(objective)
        #print("Objective value: ", opti.debug.value(objective))
        opti.debug.value(Q_in)
        sol = opti.debug


    # Casadi solution to Pandas df
    
    num_vessels = len(vessel_dict.keys())
    for i, vessel in enumerate(vessel_dict.keys()):
        # pdb.set_trace()
        # result_df.loc[i + time_step*num_vessels] = [vessel_dict[vessel]["v_name"],
        result_df.loc[i] = [vessel_dict[vessel]["v_name"],
                        time, # time?
                        sol.value(Q_in)[vessel_dict[vessel]["v_ind"]],
                        sol.value(Q_out)[vessel_dict[vessel]["v_ind"]],
                        sol.value(P_in)[vessel_dict[vessel]["v_ind"]],
                        sol.value(P_out)[vessel_dict[vessel]["v_ind"]]]
    sol_dict = {"Q_in": sol.value(Q_in),
                "Q_out": sol.value(Q_out),
                "P_in": sol.value(P_in),
                "P_out": sol.value(P_out),
                "Q_in_dt": sol.value(Q_in_dt),
                "Q_out_dt": sol.value(Q_out_dt),
                "P_in_dt": sol.value(P_in_dt),
                "P_out_dt": sol.value(P_out_dt)}
    return (sol_dict)



if __name__ == "__main__":
    #num_iters = 10
    tree_name = sys.argv[1]
    junction_mode = sys.argv[2]

    tree_name_split = tree_name.split("_")
    tree_name_base = "_".join(tree_name_split[0:2])
    with open(f'trees/zerod_input/{junction_mode}/{tree_name_base}/{tree_name}/solver_0d.json') as json_file:
        input_file = json.load(json_file)
    
    df = pd.DataFrame(columns=['name', 'time','flow_in', 'flow_out', 'pressure_in', 'pressure_out'])
    num_time_steps = len(input_file["boundary_conditions"][0]["bc_values"]["t"])
    for time_step in range(num_time_steps):
        #print(f"Solving time step {time_step + 1} of {num_time_steps}.")
        if time_step == 0:
            sol_prev = None
        
        sol_prev = solve_casadi_unsteady(time_step = time_step, sol_prev = sol_prev, input_file= input_file, result_df = df)
    
    df.sort_values(by=['name', 'time'], inplace=True)
    tree_name_split = tree_name.split("_")
    tree_name_split[-1] = "unsteady"

    tree_name_unsteady = "_".join(tree_name_split)   
    if not os.path.exists(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}'):
        os.makedirs(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}')
    df.to_csv(f'trees/zerod_output/{junction_mode}/{tree_name_base}/{tree_name}/sol_casadi.csv', index=False)
