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
M_penalty = 1e2 # Using the value from the last user-provided code

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
             # Ensure t-1 is a valid index for u_fixed if t=1 (it won't be used due to if condition)
             # This assumes u_fixed is indexed starting from 1
             if t > 1:
                 return m.u_fixed[i, t-1]
             else: # Should not happen based on outer if t==1
                 return m.u_init[i] 
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

# Function to evaluate cut expression RHS using numerical values
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
                # Use ROUNDED value for u_prev_sol_val calculation
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

    # This calculates the value of: sub_obj_k + sum(Duals * (MasterVar_sol - MasterVar_k))
    return constant_term + variable_part_value

# Function used by build_master to generate the cut expression (returns Pyomo expression)
def _calculate_cut_expr_for_master(model, cut_data):
    # --- This function remains unchanged ---
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

    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_max'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                term = dual_val * model.Pmax[i] # df/du = Pmax
                variable_expr += term * model.u[i, t]
                constant_term -= term * u_k.get((i, t), 0.0)

    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_ru'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                term_u_prev = dual_val * model.Ru[i]
                u_prev_var = model.u_prev[i, t] 
                u_prev_k = u_k.get((i, t - 1), model.u_init[i]) if t > 1 else model.u_init[i]
                variable_expr += term_u_prev * u_prev_var
                constant_term -= term_u_prev * u_prev_k

                term_zON = dual_val * model.Rsu[i]
                zON_var = model.zON[i, t]
                zON_val_k = zON_k.get((i, t), 0.0)
                variable_expr += term_zON * zON_var
                constant_term -= term_zON * zON_val_k

    for i in model.I:
        for t in model.T:
            dual_val = duals_k['lambda_rd'].get((i, t), 0.0)
            if abs(dual_val) > 1e-9:
                term_u = dual_val * model.Rd[i]
                u_var = model.u[i,t]
                u_val_k = u_k.get((i, t), 0.0)
                variable_expr += term_u * u_var
                constant_term -= term_u * u_val_k

                term_zOFF = dual_val * model.Rsd[i]
                zOFF_var = model.zOFF[i, t]
                zOFF_val_k = zOFF_k.get((i, t), 0.0)
                variable_expr += term_zOFF * zOFF_var
                constant_term -= term_zOFF * zOFF_val_k

    return constant_term + variable_expr

# Function creating the master problem
def build_master(iteration_data):
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Constrained") 

    # Sets
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.T0 = pyo.Set(initialize=time_periods_with_0)
    model.Cuts = pyo.Set(initialize=range(len(iteration_data)))

    # Parameters (Using values from user's last provided code)
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
    
    # --- Penalty Parameters ---
    model.lambda_logic1 = pyo.Param(initialize=30) 
    model.lambda_logic2 = pyo.Param(initialize=30)
    model.lambda_benders = pyo.Param(initialize=300) 
    # --------------------------

    # Calculate beta upper bound based on GLOBAL M_penalty
    max_total_demand = sum(demand.values())
    max_slack_penalty_part = M_penalty * max_total_demand 
    max_var_cost_part = sum(gen_data[i]['Pmax'] * gen_data[i]['Cv'] * len(time_periods) for i in generators) 
    estimated_beta_upper_bound = max_slack_penalty_part + max_var_cost_part + 5000 # Buffer
    print(f"DEBUG: M_penalty={M_penalty}, Estimated Beta Upper Bound: {estimated_beta_upper_bound}")

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.s_continuous = pyo.Var(model.Cuts, within=pyo.NonNegativeReals) # Removed explicit bound
    model.beta = pyo.Var(bounds=(0.0, estimated_beta_upper_bound), within=pyo.Reals) # Using calculated bound

    # Expression for u_prev
    def u_prev_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    # Objective function
    def master_objective_rule(m):
        BETA = m.beta # This is beta_old conceptually
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        benders_penalty_total = 0
        if len(iteration_data) > 0: # Check if Cuts set is non-empty
            for k in m.Cuts: 
                cut_data_k = iteration_data[k] 
                cut_expr_k = _calculate_cut_expr_for_master(m, cut_data_k) # Pyomo expression
                slack_k = m.s_continuous[k]
                penalty_k = m.lambda_benders * (BETA - cut_expr_k - slack_k)**2
                benders_penalty_total += penalty_k
        return commitment_cost + logic1_term + logic2_term + benders_penalty_total + BETA
        
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

    return model

# --- NEW Feasibility Check Function ---
def check_master_solution_feasibility_with_beta_new(master_problem, iteration_data, 
                                                   beta_to_check, # Pass beta_new here
                                                   strict_tolerance=1e-6):
    """
    Checks if the master solution's u, zON, zOFF satisfy logic exactly,
    and if the provided beta_to_check satisfies Benders cuts strictly.

    Args:
        master_problem: Solved master problem model (used for binary vars & params).
        iteration_data: List of cut data.
        beta_to_check: The value of beta (e.g., beta_new) to check against cuts.
        strict_tolerance: Small tolerance for Benders cut floating point check.

    Returns:
        True if all constraints are satisfied within criteria, False otherwise.
    """
    print("--- Checking Feasibility (Logic Exact, Benders Strict w/ beta_new) ---")
    
    # Get current solution values - ROUND binary vars 
    try:
        u_sol = {(i,t): round(pyo.value(master_problem.u[i,t])) 
                 for i in master_problem.I for t in master_problem.T}
        zON_sol = {(i,t): round(pyo.value(master_problem.zON[i,t])) 
                   for i in master_problem.I for t in master_problem.T}
        zOFF_sol = {(i,t): round(pyo.value(master_problem.zOFF[i,t])) 
                    for i in master_problem.I for t in master_problem.T}
        # We don't need beta_old_sol inside this check function anymore
    except Exception as e:
        print(f"ERROR getting master BINARY solution values: {e}. Assuming infeasible.")
        return False

    # 1. Check Logic 1 (Exact)
    logic1_satisfied = True
    for i in master_problem.I:
        for t in master_problem.T:
            u_prev_val = master_problem.u_init[i] if t == 1 else u_sol.get((i, t-1), master_problem.u_init[i]) 
            if (i,t) not in u_sol or (i,t) not in zON_sol or (i,t) not in zOFF_sol: return False # Error check
            u_val, zON_val, zOFF_val = u_sol[i,t], zON_sol[i,t], zOFF_sol[i,t]
            residual = (u_val - u_prev_val) - (zON_val - zOFF_val)
            if residual != 0:
                print(f"DEBUG: Logic 1 VIOLATED for i={i}, t={t}. Exact Residual: {residual}")
                logic1_satisfied = False
    if not logic1_satisfied: print("Overall Logic 1 status: VIOLATED")
    else: print("Overall Logic 1 status: OK")

    # 2. Check Logic 2 (Exact)
    logic2_satisfied = True
    for i in master_problem.I:
        for t in master_problem.T:
            if (i,t) not in zON_sol or (i,t) not in zOFF_sol: return False # Error check
            zON_val, zOFF_val = zON_sol[i,t], zOFF_sol[i,t]
            if zON_val + zOFF_val > 1:
                print(f"DEBUG: Logic 2 VIOLATED for i={i}, t={t}. Sum: {zON_val + zOFF_val}")
                logic2_satisfied = False
    if not logic2_satisfied: print("Overall Logic 2 status: VIOLATED")
    else: print("Overall Logic 2 status: OK")
        
    # 3. Check Benders Cuts: beta_to_check >= cut_expr_k (within STRICT tolerance)
    benders_satisfied = True
    if len(iteration_data) > 0: 
        # Store master params needed by evaluation function
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
        
        cuts_to_check = master_problem.Cuts if hasattr(master_problem, 'Cuts') else range(len(iteration_data))

        for k_idx in cuts_to_check:
            if k_idx >= len(iteration_data): continue # Skip if index invalid
            cut_data_k = iteration_data[k_idx]
            # Evaluate RHS using ROUNDED binary values 
            cut_rhs_value = evaluate_cut_expr(u_sol, zON_sol, zOFF_sol, cut_data_k, master_params_dict)
            
            # --- STRICT BENDERS CHECK using beta_to_check (beta_new) ---
            # beta_new >= cut_rhs_value - numerical_tolerance
            actual_difference = beta_to_check - cut_rhs_value
            
            if actual_difference < -strict_tolerance: # Check against strict tolerance
                print(f"DEBUG: Benders Cut {k_idx} VIOLATED by beta_new. beta_new: {beta_to_check:.6f}, cut_rhs: {cut_rhs_value:.6f}, Diff: {actual_difference:.6g}")
                benders_satisfied = False
            # --- END STRICT CHECK ---

    if not benders_satisfied:
         print("Overall Benders Cuts status: VIOLATED (Strict Check w/ beta_new)")
    else:
         print("Overall Benders Cuts status: OK (Strict Check w/ beta_new)")

    # Final Decision - Return False if ANY check failed
    all_satisfied = logic1_satisfied and logic2_satisfied and benders_satisfied
    print(f"--- Feasibility Check Result (using beta_new): {all_satisfied} ---")
    return all_satisfied

# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 
    epsilon = 1 
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')
    actual_iterations = 0 # Track iterations run

    # --- Initialize u_current, zON_current, zOFF_current ---
    u_current, zON_current, zOFF_current = {}, {}, {}
    for t in time_periods:
        for i in generators:
             u_current[i, t] = 1.0 
             u_prev = u_initial[i] if t == 1 else u_current.get((i, t-1), u_initial[i])
             if u_current[i, t] > 0.5 and u_prev < 0.5: zON_current[i, t], zOFF_current[i, t] = 1.0, 0.0
             elif u_current[i, t] < 0.5 and u_prev > 0.5: zON_current[i, t], zOFF_current[i, t] = 0.0, 1.0
             else: zON_current[i, t], zOFF_current[i, t] = 0.0, 0.0

    # Solvers
    solver = "gurobi"
    master_solver = SolverFactory(solver)
    sub_solver = SolverFactory(solver)

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Using Solver: {solver}")
    print(f"Parameters: M_penalty={M_penalty}") # Add other lambdas if needed
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")


    for k in range(1, max_iter + 1):
        actual_iterations = k
        print(f"========================= Iteration {k} =========================")

        # --- Solving Subproblem ---
        print("--- Solving Subproblem ---")
        # Ensure values passed are floats if necessary (should be okay)
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)
        if not hasattr(subproblem, 'dual'): subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        sub_results = sub_solver.solve(subproblem, tee=False)

        # --- Check Subproblem Status & Update UB ---
        if sub_results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {sub_results.solver.termination_condition}")
            print(f"Subproblem Objective (Variable Cost + Penalty): {sub_obj_val:.4f}")
            commitment_cost = sum(gen_data[i]['Csu'] * u_current.get((i,t),0) + # Use u_current for costs
                                  gen_data[i]['Csd'] * u_current.get((i,t),0) +
                                  gen_data[i]['Cf'] * u_current.get((i,t),0) 
                                  for i in generators for t in time_periods)
            current_total_cost = commitment_cost + sub_obj_val
            upper_bound = min(upper_bound, current_total_cost)
            print(f"Commitment Cost (current): {commitment_cost:.2f}")
            print(f"Candidate UB: {current_total_cost:.4f}")
            print(f"Best Upper Bound (Z_UB): {upper_bound:.4f}")

            # --- Extract Duals ---
            duals = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}}
            # (Dual extraction code as before)
            try:
                 for i in generators:
                     for t in time_periods:
                         duals['lambda_min'][(i,t)] = subproblem.dual.get(subproblem.MinPower[i,t], 0.0)
                         duals['lambda_max'][(i,t)] = subproblem.dual.get(subproblem.MaxPower[i,t], 0.0)
                         duals['lambda_ru'][(i,t)]  = subproblem.dual.get(subproblem.RampUp[i,t], 0.0)
                         duals['lambda_rd'][(i,t)]  = subproblem.dual.get(subproblem.RampDown[i,t], 0.0)
            except Exception as e:
                 print(f"Warning: Error extracting duals: {e}.") # Handle error
                 # Optionally break or set duals to zero

            # Store iteration data
            iteration_data.append({
                'iter': k, 'sub_obj': sub_obj_val, 'duals': duals,
                'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
            })
        else:
            print(f"Subproblem FAILED! Status: {sub_results.solver.termination_condition}. Terminating.")
            break

        # --- Convergence Check ---
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
             gap = upper_bound - lower_bound 
             print(f"Current Gap (UB-LB): {gap:.6f} (Tolerance: {epsilon})")
             if gap <= epsilon:
                 print("\nConvergence tolerance met.")
                 break
             if lower_bound > upper_bound + epsilon: # Check for significant crossover
                  print(f"\nWARNING: Lower Bound ({lower_bound:.4f}) significantly exceeds Upper Bound ({upper_bound:.4f}). Terminating.")
                  break
        else:
            print("Gap cannot be calculated yet.")

        if k == max_iter:
            print("\nMaximum iterations reached.")
            break

        # --- Solving the master Problem ---
        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data)
        master_solver.options['NumericFocus'] = 3
        master_results = master_solver.solve(master_problem, tee=True)

        # --- Process Master Solution ---
        if master_results.solver.termination_condition == TerminationCondition.optimal:
            master_obj_val = pyo.value(master_problem.OBJ) # Penalized objective value

            # --- Calculate Commit Cost and Betas ---
            commitment_cost_master_sol = sum(
                pyo.value(master_problem.Cf[i] * master_problem.u[i, t]) +
                pyo.value(master_problem.Csu[i] * master_problem.zON[i, t]) +
                pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t])
                for i in master_problem.I for t in master_problem.T
            )
            beta_old_sol = pyo.value(master_problem.beta) # Beta from penalized solve
            
            # --- Calculate beta_new ---
            current_lambda_benders_value = pyo.value(master_problem.lambda_benders)
            beta_new = beta_old_sol # Default if lambda is zero
            offset = 0.0
            if current_lambda_benders_value > 1e-9:
                offset = 1.0 / (2.0 * current_lambda_benders_value)
                beta_new = beta_old_sol + offset
            
            print(f"Commitment Cost (Master): {commitment_cost_master_sol:.4f}")
            print(f"Beta_old (Master Solution): {beta_old_sol:.4f}")
            print(f"Calculated Offset: {offset:.6g}")
            print(f"Calculated Beta_new (for LB): {beta_new:.4f}")
            print(f"Commitment Cost + Beta_new: {commitment_cost_master_sol + beta_new:.4f}") # This is the LB candidate

            # --- Feasibility Check using beta_new ---
            is_feasible = check_master_solution_feasibility_with_beta_new(
                                              master_problem,         
                                              iteration_data,         
                                              beta_new, # Pass the corrected beta
                                              strict_tolerance=1e-6 
                                          )
            
            # --- LB Update ---
            if is_feasible:
                print("Master solution FEASIBLE (Logic Exact, Benders Strict w/ beta_new).")
                current_lb_candidate = commitment_cost_master_sol + beta_new # Use beta_new for LB
                lower_bound = max(lower_bound, current_lb_candidate)
                print(f"Lower Bound updated: {lower_bound:.4f}")
            else:
                print("Master solution is NOT feasible. Lower Bound not updated.")
            
            # --- Print Master Status ---
            total_penalties_value = master_obj_val - commitment_cost_master_sol - beta_old_sol # For info
            print(f"total_penalties_value (for info only): {total_penalties_value:.4f}")
            print(f"Master Status: Optimal (for penalized obj)") 
            print(f"Master Objective (Value from Solver): {master_obj_val:.4f}") 
            print(f"Updated Lower Bound (Z_LB): {lower_bound:.4f}") 

                        # Updates u_current, zON_current, zOFF_current for next iteration
            u_current = {}
            for i in generators:
                for t in time_periods:
                    u_val = pyo.value(master_problem.u[i,t], exception=False) # Get value, allow None
                    u_current[i,t] = u_val if u_val is not None else 0.0 # Use 0.0 if None

            zON_current = {}
            for i in generators:
                for t in time_periods:
                    zON_val = pyo.value(master_problem.zON[i,t], exception=False)
                    zON_current[i,t] = zON_val if zON_val is not None else 0.0

            zOFF_current = {}
            for i in generators:
                for t in time_periods:
                    zOFF_val = pyo.value(master_problem.zOFF[i,t], exception=False)
                    zOFF_current[i,t] = zOFF_val if zOFF_val is not None else 0.0

        else: # Handle master failure
            print(f"Master Problem FAILED! Status: {master_results.solver.termination_condition}. Terminating.")
            break

    # --- End of Loop ---
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    if upper_bound < float('inf') and lower_bound > -float('inf'):
         final_gap = upper_bound - lower_bound
         print(f"Final Absolute Gap (UB-LB): {final_gap:.6f}")
         if abs(upper_bound) > 1e-9:
             final_rel_gap = final_gap / abs(upper_bound)
             print(f"Final Relative Gap: {final_rel_gap:.6f}")
    else:
         print(f"Final Gap: inf")
         
    print(f"Iterations Performed: {actual_iterations}") 
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    # --- Final Solution Printing (no changes needed here) ---
    best_iter_data = None
    min_total_cost = float('inf')
    if iteration_data: # Check if list is not empty
        for data in iteration_data:
            # Calculate cost corresponding to this iteration's u,z values and sub_obj
            commit_cost_iter = sum(gen_data[i]['Csu'] * data['zON_vals'].get((i,t),0) +
                                   gen_data[i]['Csd'] * data['zOFF_vals'].get((i,t),0) +
                                   gen_data[i]['Cf'] * data['u_vals'].get((i,t),0)
                                   for i in generators for t in time_periods)
            if data['sub_obj'] is not None:
                total_cost_iter = commit_cost_iter + data['sub_obj']
                if total_cost_iter < min_total_cost - 1e-6:
                    min_total_cost = total_cost_iter
                    best_iter_data = data
                elif abs(total_cost_iter - min_total_cost) < 1e-6 and best_iter_data and data['iter'] > best_iter_data['iter']:
                     best_iter_data = data # Prefer later iterations if cost is same

    if best_iter_data:
        print(f"\n--- Best Solution Found (Iteration {best_iter_data['iter']}) ---")
        print(f"Best Total Cost (Upper Bound): {min_total_cost:.4f}")
        print("Commitment Schedule (u_it):")
        for t in time_periods:
            print(f"  t={t}: ", {i: round(best_iter_data['u_vals'].get((i,t),0)) for i in generators})

        print("\nFinal Dispatch (p_it) for Best Solution:")
        final_subproblem = build_subproblem(best_iter_data['u_vals'], best_iter_data['zON_vals'], best_iter_data['zOFF_vals'])
        if not hasattr(final_subproblem, 'dual'): final_subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        
        sub_results_final = sub_solver.solve(final_subproblem, tee=False) 
        if sub_results_final.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible]:
            final_sub_obj_recalc = pyo.value(final_subproblem.OBJ)
            final_commit_cost_recalc = sum(gen_data[i]['Csu'] * best_iter_data['zON_vals'].get((i,t),0) +
                                           gen_data[i]['Csd'] * best_iter_data['zOFF_vals'].get((i,t),0) +
                                           gen_data[i]['Cf'] * best_iter_data['u_vals'].get((i,t),0)
                                           for i in generators for t in time_periods)
            for t in time_periods:
                 print(f"  t={t}: ", {i: f"{pyo.value(final_subproblem.p[i,t]):.2f}" for i in generators})
            print(f"Final Subproblem Objective (Var Cost + Penalty): {final_sub_obj_recalc:.4f}")
            print(f"Final Commitment Cost: {final_commit_cost_recalc:.2f}")
            print(f"Final Total Cost (recalculated): {final_commit_cost_recalc + final_sub_obj_recalc:.4f}")
            print("Final Demand Slack:")
            for t in time_periods:
                print(f"  t={t}: {pyo.value(final_subproblem.demand_slack[t]):.4f}")
        else:
             print("Could not re-solve final subproblem for dispatch details.")
    else:
        print("\nNo feasible solution found or recorded.")

if __name__ == '__main__':
    main()