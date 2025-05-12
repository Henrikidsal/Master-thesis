##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo, classically
##### No QUBO solver is used here.

##### basic imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import math

# Sets
generators = [1, 2, 3]
time_periods = [x for x in range(1, 4)] # T=3 hours
time_periods_with_0 = [x for x in range(0, 4)] # Including t=0 for initial conditions

# Generator Parameters
gen_data = {
    1: {'Pmin': 50,  'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
    2: {'Pmin': 80,  'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
    3: {'Pmin': 40,  'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5,  'Csd': 1.0, 'Cv': 0.150}
}


# Demand and Reserve Parameters
demand = {1: 160, 2: 500, 3: 400}


# Initial Conditions for T = 0 
# u_it: ON/OFF status
u_initial = {1: 0, 2: 0, 3: 1}
# p_it: Power output 
p_initial = {1: 0, 2: 0, 3: 100}

# Large Penalty for slack variables, in sub problem
M_penalty = 10

# Number of bits for beta variable
#num_beta_bits = 21

# Function creating the sub problem
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    
    model = pyo.ConcreteModel(name="UCP_Subproblem") # Just creates an empty model

    # Sets
    model.I = pyo.Set(initialize=generators) # Creates the generators set
    model.T = pyo.Set(initialize=time_periods) # Creates the time periods set
    model.T0 = pyo.Set(initialize=time_periods_with_0) # Creates a set for the time periods, but also includes t=0

    # Fixed parameters from master problem
    u_fixed_param_vals = {(i,t): u_fixed_vals[i,t] for i in model.I for t in model.T}
    zON_fixed_param_vals = {(i,t): zON_fixed_vals[i,t] for i in model.I for t in model.T}
    zOFF_fixed_param_vals = {(i,t): zOFF_fixed_vals[i,t] for i in model.I for t in model.T}

    model.u_fixed = pyo.Param(model.I, model.T, initialize=u_fixed_param_vals, mutable=True)
    model.zON_fixed = pyo.Param(model.I, model.T, initialize=zON_fixed_param_vals, mutable=True)
    model.zOFF_fixed = pyo.Param(model.I, model.T, initialize=zOFF_fixed_param_vals, mutable=True)

    # Parameters
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})
    model.Cv = pyo.Param(model.I, initialize={i: gen_data[i]['Cv'] for i in model.I})
    model.D = pyo.Param(model.T, initialize=demand)
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.p_init = pyo.Param(model.I, initialize=p_initial)
    model.M = pyo.Param(initialize=M_penalty)

    # Variabls
    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)
    model.demand_slack = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # objective Functio
    def objective_rule(m):
        variable_cost = sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
        penalty_cost = m.M * sum(m.demand_slack[t] for t in m.T)
        return variable_cost + penalty_cost
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # A smart placeholder that always gives the correct power output for the previous time period, even if t=1
    def p_prev_rule(m, i, t):
        if t == 1:
            return m.p_init[i]
        else:
            return m.p[i, t-1]
    model.p_prev = pyo.Expression(model.I, model.T, rule=p_prev_rule)

    # Uses the fixed value from the master problem, which is different in t=1
    def u_prev_fixed_rule(m, i, t):
         if t == 1:
             return m.u_init[i]
         else:
             return m.u_fixed[i, t-1]
    model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=u_prev_fixed_rule)

    # Constraint: Minimum power output
    def min_power_rule(m, i, t):
        return m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t]
    model.MinPower = pyo.Constraint(model.I, model.T, rule=min_power_rule)

    # Constraint: Maximum power output
    def max_power_rule(m, i, t):
        return m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t]
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=max_power_rule)

    # Constrainr: Ramping up limit
    def ramp_up_rule(m, i, t):
        return m.p[i, t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t]
    model.RampUp = pyo.Constraint(model.I, model.T, rule=ramp_up_rule)

    # Constraint: Ramping down limit
    def ramp_down_rule(m, i, t):
        return m.p_prev[i,t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t]
    model.RampDown = pyo.Constraint(model.I, model.T, rule=ramp_down_rule)

    # Constraint: Demand balance (with slack)
    def demand_rule(m, t):
        return sum(m.p[i, t] for i in m.I) + m.demand_slack[t] >= m.D[t]
    model.Demand = pyo.Constraint(model.T, rule=demand_rule)

    # Collecting dual variables for Benders cuts
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model

def _calculate_cut_expr_for_master(model, cut_data):
    sub_obj_k = cut_data['sub_obj']
    duals_k = cut_data['duals']
    u_k = cut_data['u_vals']
    zON_k = cut_data['zON_vals']
    zOFF_k = cut_data['zOFF_vals']

    # Initialize terms
    constant_term = sub_obj_k # Start with subproblem obj (includes var cost + sub penalty)
    variable_expr = 0.0

    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_min'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                term = -dual_val * model.Pmin[i] # df/du = -Pmin
                variable_expr += term * model.u[i, t]
                constant_term -= term * u_k.get((i, t), 0.0) # Subtract lambda*df/du*u_k

    # Max Power: Pmax*u - p >= 0. Dual lambda_max >= 0. Term: lambda_max * (Pmax * (u - u_k))
    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_max'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                term = dual_val * model.Pmax[i] # df/du = Pmax
                variable_expr += term * model.u[i, t]
                constant_term -= term * u_k.get((i, t), 0.0)

    # Ramp Up: (Ru*u_prev + Rsu*zON) - p + p_prev >= 0. Dual lambda_ru >= 0.
    # Term: lambda_ru * (Ru*(u_prev - u_prev_k) + Rsu*(zON - zON_k))
    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_ru'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                # u_prev term (df/du_prev = Ru)
                term_u_prev = dual_val * model.Ru[i]
                u_prev_var = model.u_prev[i, t] # Master variable u[i, t-1] or u_init[i]
                u_prev_k = u_k.get((i, t - 1), model.u_init[i]) if t > 1 else model.u_init[i]
                variable_expr += term_u_prev * u_prev_var
                constant_term -= term_u_prev * u_prev_k

                # zON term (df/dzON = Rsu)
                term_zON = dual_val * model.Rsu[i]
                zON_var = model.zON[i, t]
                zON_val_k = zON_k.get((i, t), 0.0)
                variable_expr += term_zON * zON_var
                constant_term -= term_zON * zON_val_k

    # Ramp Down: (Rd*u + Rsd*zOFF) - p_prev + p >= 0. Dual lambda_rd >= 0.
    # Term: lambda_rd * (Rd*(u - u_k) + Rsd*(zOFF - zOFF_k))
    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_rd'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                # u term (df/du = Rd)
                term_u = dual_val * model.Rd[i]
                u_var = model.u[i,t]
                u_val_k = u_k.get((i, t), 0.0)
                variable_expr += term_u * u_var
                constant_term -= term_u * u_val_k

                # zOFF term (df/dzOFF = Rsd)
                term_zOFF = dual_val * model.Rsd[i]
                zOFF_var = model.zOFF[i, t]
                zOFF_val_k = zOFF_k.get((i, t), 0.0)
                variable_expr += term_zOFF * zOFF_var
                constant_term -= term_zOFF * zOFF_val_k

    return constant_term + variable_expr

# Function creating the master problem
def build_master(iteration_data):
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Constrained") # Use descriptive name

    # Sets
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.T0 = pyo.Set(initialize=time_periods_with_0)
    #model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)
    model.Cuts = pyo.Set(initialize=range(len(iteration_data)))

    # Paramters 
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.D = pyo.Param(model.T, initialize=demand)
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})
    model.lambda_logic1 = pyo.Param(initialize=200) 
    model.lambda_logic2 = pyo.Param(initialize=200)
    lamda_Ben = 300
    model.lambda_benders = pyo.Param(initialize=lamda_Ben)

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    #model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary)
    model.s_continuous = pyo.Var(model.Cuts, within=pyo.NonNegativeReals, bounds=(1/(2*lamda_Ben), 1e6))
    model.beta = pyo.Var(within=pyo.NonNegativeReals, bounds=(0, 1e7)) # Continuous variable for beta

    # Expression for u_prev
    def u_prev_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    # Objective function
    def master_objective_rule(m):

        BETA = m.beta

        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)

        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )

        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )

        benders_penalty_total = 0
        for k in m.Cuts: # k is the index (0, 1, ..., num_cuts-1)
            cut_data_k = iteration_data[k] # Get data for this specific cut

            # Calculate the cut expression: SubObj_k + sum(Duals_k * (MasterVar - MasterVar_k))
            cut_expr_k = _calculate_cut_expr_for_master(m, cut_data_k)

            # Get the continuous slack variable for this cut
            slack_k = m.s_continuous[k]

            penalty_k = m.lambda_benders * (BETA - cut_expr_k - slack_k)**2
            benders_penalty_total += penalty_k

        return commitment_cost + logic1_term + logic2_term + benders_penalty_total + BETA
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

    return model

def check_master_solution_feasibility(master_problem, iteration_data, 
                                     lambda_benders_value, 
                                     numerical_tolerance=1e-6):
    
    print("--- Checking Feasibility (Logic Exact, Benders w/ Offset Tol) ---")
    
    # Get current solution values - ROUND binary vars to handle solver tolerance
    try:
        u_sol = {(i,t): round(pyo.value(master_problem.u[i,t])) 
                 for i in master_problem.I for t in master_problem.T}
        zON_sol = {(i,t): round(pyo.value(master_problem.zON[i,t])) 
                   for i in master_problem.I for t in master_problem.T}
        zOFF_sol = {(i,t): round(pyo.value(master_problem.zOFF[i,t])) 
                    for i in master_problem.I for t in master_problem.T}
        beta_sol = pyo.value(master_problem.beta) 
        
        # Check if beta is None (shouldn't happen if optimal)
        if beta_sol is None:
             print("ERROR: Continuous variable beta is None. Cannot check feasibility.")
             return False
             
    except Exception as e:
        # Handle cases where variable might not exist or pyo.value fails
        print(f"ERROR getting master solution values: {e}. Assuming infeasible.")
        return False

    # 1. Check Logic 1: u_it - u_{i,t-1} - zON_it + zOFF_it == 0 (Exact Check)
    logic1_satisfied = True
    for i in master_problem.I:
        for t in master_problem.T:
            # u_prev is based on the *initial* condition param OR the rounded solution value
            # Need access to u_init parameter from the master_problem object
            u_prev_val = master_problem.u_init[i] if t == 1 else u_sol.get((i, t-1), master_problem.u_init[i]) 
            
            # Ensure keys exist before accessing - belt-and-suspenders check
            if (i,t) not in u_sol or (i,t) not in zON_sol or (i,t) not in zOFF_sol:
                 print(f"ERROR: Missing solution value for i={i}, t={t}. Assuming infeasible.")
                 return False # Cannot check if values are missing
                 
            u_val = u_sol[i,t]
            zON_val = zON_sol[i,t]
            zOFF_val = zOFF_sol[i,t]
            
            # Calculate residual using integers
            residual = (u_val - u_prev_val) - (zON_val - zOFF_val)
            
            # Exact check for zero
            if residual != 0:
                print(f"DEBUG: Logic 1 VIOLATED for i={i}, t={t}. Exact Residual: {residual}")
                logic1_satisfied = False
                # break # You could break inner loop if needed
        # if not logic1_satisfied: break # You could break outer loop too
                
    if not logic1_satisfied: 
        print("Overall Logic 1 status: violated")
    else:
        print("Overall Logic 1 status: OK")

    # 2. Check Logic 2: zON_it + zOFF_it <= 1 (Exact Check)
    logic2_satisfied = True
    for i in master_problem.I:
        for t in master_problem.T:
            # Ensure keys exist
            if (i,t) not in zON_sol or (i,t) not in zOFF_sol:
                 print(f"ERROR: Missing solution value for i={i}, t={t}. Assuming infeasible.")
                 return False
                 
            zON_val = zON_sol[i,t]
            zOFF_val = zOFF_sol[i,t]
            
            # Exact check using integers
            if zON_val + zOFF_val > 1:
                print(f"DEBUG: Logic 2 VIOLATED for i={i}, t={t}. Sum: {zON_val + zOFF_val}")
                logic2_satisfied = False
                # break
        # if not logic2_satisfied: break
                
    if not logic2_satisfied:
        print("Overall Logic 2 status: violated")
    else:
        print("Overall Logic 2 status: OK")
        
    # 3. Check Benders Cuts: beta >= cut_expr_k - offset (within numerical tolerance)
    benders_satisfied = True
    if len(iteration_data) > 0: 
        if lambda_benders_value <= 1e-9: # Avoid division by zero or huge offsets if lambda is effectively zero
             print("warnin: lambda_benders is very small or zero, cannot calculate offset reliably. Checking Benders cuts with zero tolerance.")
             theoretical_offset = 0.0
        else:
            # --- Calculate Theoretical Offset ---
            #theoretical_offset = 1.0 / (2.0 * lambda_benders_value)
            theoretical_offset = 0.0
            print(f"just for debugging: Allowing Benders violation up to theoretical offset: {theoretical_offset:.6g}")
        
        # Define evaluate_cut_expr helper function here
        def evaluate_cut_expr(solved_u_rounded, solved_zON_rounded, solved_zOFF_rounded, 
                              cut_data, master_params):
            # This function takes the *numerical* solution values
            sub_obj_k = cut_data['sub_obj']
            duals_k = cut_data['duals']
            u_k = cut_data['u_vals']
            zON_k = cut_data['zON_vals']
            zOFF_k = cut_data['zOFF_vals']
            
            constant_term = sub_obj_k 
            variable_part_value = 0.0 

            I_set = master_params['I']
            T_set = master_params['T']
            # Access parameters via the dictionary
            pmin = master_params['Pmin']
            pmax = master_params['Pmax']
            ru = master_params['Ru']
            rsu = master_params['Rsu']
            rd = master_params['Rd']
            rsd = master_params['Rsd']
            u_init = master_params['u_init']

            for i in I_set:
                for t in T_set:
                    # Min Power Term
                    dual_val_min = duals_k['lambda_min'].get((i, t), 0.0)
                    if abs(dual_val_min) > 1e-9: 
                        term_coeff = -dual_val_min * pmin[i]
                        variable_part_value += term_coeff * solved_u_rounded.get((i,t), 0) # Use get for safety
                        constant_term -= term_coeff * u_k.get((i, t), 0.0)
                    
                    # Max Power Term
                    dual_val_max = duals_k['lambda_max'].get((i, t), 0.0)
                    if abs(dual_val_max) > 1e-9:
                        term_coeff = dual_val_max * pmax[i]
                        variable_part_value += term_coeff * solved_u_rounded.get((i,t), 0) # Use get for safety
                        constant_term -= term_coeff * u_k.get((i, t), 0.0)

                    # Ramp Up Term
                    dual_val_ru = duals_k['lambda_ru'].get((i, t), 0.0)
                    if abs(dual_val_ru) > 1e-9:
                        term_coeff_u_prev = dual_val_ru * ru[i]
                        u_prev_sol_val = u_init[i] if t == 1 else solved_u_rounded.get((i, t-1), u_init[i]) 
                        u_prev_k_val = u_init[i] if t == 1 else u_k.get((i, t - 1), u_init[i])
                        variable_part_value += term_coeff_u_prev * u_prev_sol_val
                        constant_term -= term_coeff_u_prev * u_prev_k_val
                        
                        term_coeff_zON = dual_val_ru * rsu[i]
                        variable_part_value += term_coeff_zON * solved_zON_rounded.get((i,t), 0) # Use get for safety
                        constant_term -= term_coeff_zON * zON_k.get((i, t), 0.0)

                    # Ramp Down Term
                    dual_val_rd = duals_k['lambda_rd'].get((i, t), 0.0)
                    if abs(dual_val_rd) > 1e-9:
                        term_coeff_u = dual_val_rd * rd[i]
                        variable_part_value += term_coeff_u * solved_u_rounded.get((i,t), 0) # Use get for safety
                        constant_term -= term_coeff_u * u_k.get((i, t), 0.0)

                        term_coeff_zOFF = dual_val_rd * rsd[i]
                        variable_part_value += term_coeff_zOFF * solved_zOFF_rounded.get((i,t), 0) # Use get for safety
                        constant_term -= term_coeff_zOFF * zOFF_k.get((i, t), 0.0)

            return constant_term + variable_part_value

        # Store master params needed by evaluation function
        # Need to access the actual parameter values from the Pyomo model object
        master_params_dict = {
            'I': list(master_problem.I), 'T': list(master_problem.T),
            'Pmin': {i: master_problem.Pmin[i] for i in master_problem.I}, 
            'Pmax': {i: master_problem.Pmax[i] for i in master_problem.I},
            'Ru':   {i: master_problem.Ru[i] for i in master_problem.I}, 
            'Rsu':  {i: master_problem.Rsu[i] for i in master_problem.I},
            'Rd':   {i: master_problem.Rd[i] for i in master_problem.I}, 
            'Rsd':  {i: master_problem.Rsd[i] for i in master_problem.I},
            'u_init': {i: master_problem.u_init[i] for i in master_problem.I}
        }

        # Use the master_problem.Cuts set directly if available and populated
        cuts_to_check = master_problem.Cuts if hasattr(master_problem, 'Cuts') else range(len(iteration_data))

        for k_idx in cuts_to_check:
            # Ensure index is valid for iteration_data
            if k_idx >= len(iteration_data):
                 print(f"WARNING: Cut index {k_idx} out of range for iteration_data (len={len(iteration_data)}). Skipping check.")
                 continue
                 
            cut_data_k = iteration_data[k_idx]
            # Evaluate using ROUNDED binary values from the solution
            cut_rhs_value = evaluate_cut_expr(u_sol, zON_sol, zOFF_sol, cut_data_k, master_params_dict)
            
            # --- MODIFIED CHECK ---
            # Check if beta is "close enough" considering the offset
            # beta >= cut_rhs_value - theoretical_offset  (allowing for numerical tolerance)
            # beta - cut_rhs_value >= -theoretical_offset - numerical_tolerance
            allowed_lower_limit = -theoretical_offset - numerical_tolerance
            actual_difference = beta_sol - cut_rhs_value
            
            if actual_difference < allowed_lower_limit:
                print(f"DEBUG: Benders Cut {k_idx} VIOLATED. beta: {beta_sol:.6f}, cut_rhs: {cut_rhs_value:.6f}, Diff: {actual_difference:.6g}, Allowed Lower Limit: {allowed_lower_limit:.6g}")
                benders_satisfied = False
                # break # Stop checking cuts if one fails
            # --- END MODIFIED CHECK ---

    if not benders_satisfied:
         print("Overall Benders Cuts status: VIOLATED (beyond offset)")
    else:
         print("Overall Benders Cuts status: OK (within offset tolerance)")


    # Final Decision - Return False if ANY check failed
    all_satisfied = logic1_satisfied and logic2_satisfied and benders_satisfied
    print(f"--- Feasibility Check Result (allowing offset): {all_satisfied} ---")
    return all_satisfied

# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 # Maximum number of iterations for Benders loop
    epsilon = 1 # Convergence tolerance for gap
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')

    # Initialize master variables for first subproblem solve
    # Basically, this just says that all generators are ON at t=1 for the first guess in iter 1.
    u_current = {}
    zON_current = {}
    zOFF_current = {}
    for t in time_periods:
        for i in generators:
             u_current[i, t] = 1.0 # Start all ON for robustness with penalty
             # Determine zON/zOFF based on u_current and u_initial
             u_prev = u_initial[i] if t == 1 else u_current.get((i, t-1), u_initial[i])
             if u_current[i, t] > 0.5 and u_prev < 0.5:
                 zON_current[i, t] = 1.0
                 zOFF_current[i, t] = 0.0
             elif u_current[i, t] < 0.5 and u_prev > 0.5:
                 zON_current[i, t] = 0.0
                 zOFF_current[i, t] = 1.0
             else:
                 zON_current[i, t] = 0.0
                 zOFF_current[i, t] = 0.0

    # Use Gurobi for both
    solver = "gurobi"
    master_solver = SolverFactory(solver)
    sub_solver = SolverFactory(solver)

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Using Solver: {solver}")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")


    for k in range(1, max_iter + 1):
        print(f"========================= Iteration {k} =========================")

        # Solving Subproblem
        print("--- Solving Subproblem ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)

        # Ensure dual suffix exists before solving subproblem, just makes sure the duals can be extracted
        if not hasattr(subproblem, 'dual') or not isinstance(subproblem.dual, pyo.Suffix):
            subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif subproblem.dual.direction != pyo.Suffix.IMPORT:
             subproblem.dual.set_direction(pyo.Suffix.IMPORT)

        # Solve the subproblem
        sub_results = sub_solver.solve(subproblem, tee=False)

        # Checks Subproblem Status
        if sub_results.solver.termination_condition == TerminationCondition.optimal or \
           sub_results.solver.termination_condition == TerminationCondition.feasible:

            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {sub_results.solver.termination_condition}")
            print(f"Subproblem Objective (Variable Cost + Penalty): {sub_obj_val:.4f}")

            # Calculate Commitment Cost for upper bound
            commitment_cost = sum(gen_data[i]['Csu'] * zON_current[i, t] +
                                  gen_data[i]['Csd'] * zOFF_current[i, t] +
                                  gen_data[i]['Cf'] * u_current[i, t]
                                  for i in generators for t in time_periods)

            current_total_cost = commitment_cost + sub_obj_val
            upper_bound = min(upper_bound, current_total_cost)
            print(f"Commitment Cost: {commitment_cost:.2f}")
            print(f"Current Total Cost (Commitment + Sub Obj): {current_total_cost:.4f}")
            print(f"Best Upper Bound (Z_UB): {upper_bound:.4f}")

            # Extracts Duals and Store Iteration Data
            duals = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}}
            try:
                 for i in generators:
                     for t in time_periods:
                         # Use .get on the suffix itself for robustness
                         duals['lambda_min'][(i,t)] = subproblem.dual.get(subproblem.MinPower[i,t], 0.0)
                         duals['lambda_max'][(i,t)] = subproblem.dual.get(subproblem.MaxPower[i,t], 0.0)
                         duals['lambda_ru'][(i,t)]  = subproblem.dual.get(subproblem.RampUp[i,t], 0.0)
                         duals['lambda_rd'][(i,t)]  = subproblem.dual.get(subproblem.RampDown[i,t], 0.0)
            except Exception as e:
                 print(f"Warning: Error extracting duals: {e}. Using 0.0 for cut generation.")
                 for key in duals:
                     for i in generators:
                         for t in time_periods:
                            duals[key][(i,t)] = 0.0

            # Store data needed for the Benders cut
            iteration_data.append({
                'iter': k,
                'sub_obj': sub_obj_val,
                'duals': duals,
                'u_vals': u_current.copy(),
                'zON_vals': zON_current.copy(),
                'zOFF_vals': zOFF_current.copy()
            })

        else:
            print(f"Subproblem FAILED to solve optimally! Status: {sub_results.solver.termination_condition}")
            print("Terminating Benders loop due to error.")
            break

        # Checking the convergence
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
             gap = (upper_bound - lower_bound) 
             print(f"Current Gap: {gap:.6f} (Tolerance: {epsilon})")
             if gap <= epsilon:
                 print("\nConvergence tolerance met.")
                 break
        else:
            print("Gap cannot be calculated yet.")


        if k == max_iter:
            print("\nMaximum iterations reached.")
            break

        # Solving the master Problem
        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data)
        master_solver.options['NumericFocus'] = 3
        master_results = master_solver.solve(master_problem, tee=True)

        # Checks status
        if master_results.solver.termination_condition == TerminationCondition.optimal:
            master_obj_val = pyo.value(master_problem.OBJ)

            commitment_cost_master_sol = sum(
                pyo.value(master_problem.Cf[i] * master_problem.u[i, t]) +
                pyo.value(master_problem.Csu[i] * master_problem.zON[i, t]) +
                pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t])
                for i in master_problem.I for t in master_problem.T
            )
            beta_master_sol = pyo.value(master_problem.beta) # Use the continuous beta variable

            print(f"Commitment Cost (Master): {commitment_cost_master_sol:.4f}")
            print(f"Beta (Master): {beta_master_sol:.4f}")
            print(f"Commitment Cost + Beta (Master): {commitment_cost_master_sol + beta_master_sol:.4f}")

            current_lambda_benders_value = pyo.value(master_problem.lambda_benders)

            is_feasible = check_master_solution_feasibility(
                                              master_problem,         # Pass the solved model
                                              iteration_data,         # Pass the cut data
                                              current_lambda_benders_value, # Pass the lambda value
                                              numerical_tolerance=1e-6  # Use desired tolerance
                                          )
            
            if is_feasible:
                print("Master solution is FEASIBLE (Logic Exact, Benders w/ Offset Tol).")
                current_lb_candidate = commitment_cost_master_sol + beta_master_sol
                lower_bound = max(lower_bound, current_lb_candidate)
                print(f"Lower Bound updated: {lower_bound:.4f}")
            else:
                print("Master solution is NOTt feasible. Lower Bound not updated.")

            print(f"Master Status: Optimal (for penalized obj)") # Clarify status
            print(f"Master Objective (Value from Solver): {master_obj_val:.4f}") # Show solver objective
            print(f"Updated Lower Bound (Z_LB): {lower_bound:.4f}") # Show current valid LB

            # Updates u_current, zON_current, zOFF_current for next iteration
            u_current = {(i,t): (master_problem.u[i,t].value if master_problem.u[i,t].value is not None else 0.0)
                         for i in generators for t in time_periods}
            zON_current = {(i,t): (master_problem.zON[i,t].value if master_problem.zON[i,t].value is not None else 0.0)
                           for i in generators for t in time_periods}
            zOFF_current = {(i,t): (master_problem.zOFF[i,t].value if master_problem.zOFF[i,t].value is not None else 0.0)
                            for i in generators for t in time_periods}

        else: # Handle master failure
            print(f"Master Problem FAILED to solve optimally! Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop due to master error.")
            break

    # --- End of Loop ---
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) / (abs(upper_bound) + 1e-9) if upper_bound < float('inf') and lower_bound > -float('inf') else float('inf')
    print(f"Final Gap: {final_gap:.6f}")
    print(f"Iterations Performed: {k if k <= max_iter else max_iter}") # Show correct iter count if max_iter reached
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    # Prints Final Solution (from the iteration that gave the upper bound ofc)
    best_iter_data = None
    min_total_cost = float('inf')
    # Finds the iteration data corresponding to the best upper bound
    for data in iteration_data:
         commit_cost = sum(gen_data[i]['Csu'] * data['zON_vals'][i, t] +
                           gen_data[i]['Csd'] * data['zOFF_vals'][i, t] +
                           gen_data[i]['Cf'] * data['u_vals'][i, t]
                           for i in generators for t in time_periods)
         if data['sub_obj'] is not None:
             total_cost = commit_cost + data['sub_obj']
             if total_cost < min_total_cost - 1e-6:
                 min_total_cost = total_cost
                 best_iter_data = data
             elif abs(total_cost - min_total_cost) < 1e-6 and best_iter_data and data['iter'] > best_iter_data['iter']:
                  best_iter_data = data

    if best_iter_data:
        print("\n--- Best Solution Found (from iteration {}) ---".format(best_iter_data['iter']))
        print(f"Best Total Cost (Upper Bound): {min_total_cost:.4f}")
        print("Commitment Schedule (u_it):")
        for t in time_periods:
            print(f"  t={t}: ", {i: round(best_iter_data['u_vals'][i,t]) for i in generators})

        # Re-solves the subproblem one last time with the best u, zON, zOFF to get final p values
        print("\nFinal Dispatch (p_it):")
        final_subproblem = build_subproblem(best_iter_data['u_vals'], best_iter_data['zON_vals'], best_iter_data['zOFF_vals'])
        # Ensure dual suffix exists for final solve too
        if not hasattr(final_subproblem, 'dual') or not isinstance(final_subproblem.dual, pyo.Suffix):
            final_subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif final_subproblem.dual.direction != pyo.Suffix.IMPORT:
             final_subproblem.dual.set_direction(pyo.Suffix.IMPORT)

        sub_solver.solve(final_subproblem, tee=False) #Solves the final sub problem with the best found config, its to retrieve the final values.
        if pyo.value(final_subproblem.OBJ) is not None:
            for t in time_periods:
                 # Use value() function for potentially indexed variables
                print(f"  t={t}: ", {i: f"{pyo.value(final_subproblem.p[i,t]):.2f}" for i in generators})
            print(f"Final Subproblem Objective (Var Cost + Penalty): {pyo.value(final_subproblem.OBJ):.4f}")
            final_commit_cost = sum(gen_data[i]['Csu'] * best_iter_data['zON_vals'][i, t] +
                           gen_data[i]['Csd'] * best_iter_data['zOFF_vals'][i, t] +
                           gen_data[i]['Cf'] * best_iter_data['u_vals'][i, t]
                           for i in generators for t in time_periods)
            print(f"Final Commitment Cost: {final_commit_cost:.2f}")
            print(f"Final Total Cost (recalculated): {final_commit_cost + pyo.value(final_subproblem.OBJ):.4f}")
            print("Final Demand Slack:")
            for t in time_periods:
                print(f"  t={t}: {pyo.value(final_subproblem.demand_slack[t]):.4f}")
        else:
             print("Could not resolve final subproblem to show dispatch.")

    else:
        print("\nNo feasible solution found or Benders loop terminated early.")


if __name__ == '__main__':
    main()