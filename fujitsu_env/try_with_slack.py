##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo, classically
##### THe logic constraints, both types, are penalty terms
##### Benders optimality cuts are also penalty terms.
##### The LB could be calculated as only commitment cost + beta, but here its the full QUBO value.

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
M_penalty = 1e4

# Number of bits for beta variable
num_beta_bits =9

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

    # --- Parameters ---
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

    # --- Variables ---
    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)
    model.demand_slack = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # --- Objective Function ---
    def objective_rule(m):
        variable_cost = sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
        penalty_cost = m.M * sum(m.demand_slack[t] for t in m.T)
        return variable_cost + penalty_cost
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # --- Constraints ---

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

    # --- Dual Suffix ---
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model

# Function creating the master problem
def build_master(iteration_data):
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Unconstrained")

    # Sets
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.T0 = pyo.Set(initialize=time_periods_with_0)
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)
    model.Cuts = pyo.Set(initialize=range(len(iteration_data))) # Set of cut indices

    # Paramters (penalty values, etc.)
    model.Pmax_param = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I}) # Renamed to avoid conflict
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.D_param = pyo.Param(model.T, initialize=demand) # Renamed
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.lambda_logic1 = pyo.Param(initialize=20) # Increased penalty
    model.lambda_logic2 = pyo.Param(initialize= 18) # Increased penalty
    model.lambda_benderscut = pyo.Param(initialize=3) # Penalty for Benders cuts, can be tuned

    # Parameters for Benders cut coefficients (used in objective)
    model.Pmin_param = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I}) # Renamed
    model.Rd_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for  i in model.I})       # Renamed
    model.Rsd_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})    # Renamed
    model.Ru_param = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})       # Renamed
    model.Rsu_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})    # Renamed

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary)
    model.s_cuts = pyo.Var(model.Cuts, within=pyo.Reals, bounds=(0, None)) # Non-negative slack variables for Benders cuts

    # Expression for u_prev
    def u_prev_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    # Objective function
    def master_objective_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )

        benders_penalty_term = 0
        for k_idx in m.Cuts: # k_idx is the integer index for the cut
            data = iteration_data[k_idx]
            sub_obj_k = data['sub_obj']
            duals_k = data['duals']
            u_k_vals = data['u_vals']
            zON_k_vals = data['zON_vals']
            zOFF_k_vals = data['zOFF_vals']

            # This is θ_j + Φ_j^T (x - x_j)
            cut_rhs_expr = sub_obj_k
            for i in m.I: # MinPower
                for t_loop in m.T: # Renamed t to t_loop to avoid conflict with m.T
                    dual_val = duals_k['lambda_min'].get((i, t_loop), 0.0)
                    cut_rhs_expr += dual_val * (m.Pmin_param[i] * (m.u[i, t_loop] - u_k_vals.get((i,t_loop), 0.0)))
            for i in m.I: # MaxPower
                for t_loop in m.T:
                    dual_val = duals_k['lambda_max'].get((i, t_loop), 0.0)
                    cut_rhs_expr += dual_val * (m.Pmax_param[i] * (m.u[i, t_loop] - u_k_vals.get((i,t_loop), 0.0))) # Note: max power dual is typically non-positive for <= constraint
            for i in m.I: # RampUp
                for t_loop in m.T:
                    dual_val = duals_k['lambda_ru'].get((i, t_loop), 0.0) # Duals for p_it - p_prev <= RHS, so they are non-negative
                    # Term for u_prev
                    u_prev_term_val = 0
                    u_prev_k_val = u_k_vals.get((i, t_loop-1), m.u_init[i]) if t_loop > 1 else m.u_init[i]
                    if t_loop > 1:
                         u_prev_term_val = m.Ru_param[i] * (m.u[i, t_loop-1] - u_prev_k_val)
                    else: # if t_loop == 1, u_prev is u_init, which is fixed, so (m.u_init - u_init_k) = 0 if u_init_k is also m.u_init.

                          pass


                    # Term for zON
                    zON_term_val = m.Rsu_param[i] * (m.zON[i, t_loop] - zON_k_vals.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val * (u_prev_term_val + zON_term_val)

            for i in m.I: # RampDown
                for t_loop in m.T:
                    dual_val = duals_k['lambda_rd'].get((i, t_loop), 0.0) # Duals for p_prev - p_it <= RHS, non-negative
                    # Term for u_current (related to RHS: Rd * u_fixed)
                    u_term_val = m.Rd_param[i] * (m.u[i, t_loop] - u_k_vals.get((i, t_loop), 0.0))
                    # Term for zOFF_current (related to RHS: Rsd * zOFF_fixed)
                    zOFF_term_val = m.Rsd_param[i] * (m.zOFF[i, t_loop] - zOFF_k_vals.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val * (u_term_val + zOFF_term_val)


            penalty_for_this_cut = (binary_beta_expr - m.s_cuts[k_idx] - cut_rhs_expr)**2
            benders_penalty_term += m.lambda_benderscut * penalty_for_this_cut

        return commitment_cost + logic1_term + logic2_term + binary_beta_expr + benders_penalty_term

    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

    # Benders Cuts Constraints are REMOVED
    # model.BendersCuts = pyo.Constraint(model.Cuts, rule=benders_cut_rule)

    return model


# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 # Maximum number of iterations for Benders loop
    epsilon = 1 # Convergence tolerance for gap
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')

    # Initialize master variables for first subproblem solve
    u_current = {}
    zON_current = {}
    zOFF_current = {}
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0
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

    solver = "gurobi" 
    master_solver = SolverFactory(solver)
    sub_solver = SolverFactory("gurobi") 

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Using Master Solver: {solver}, Sub Solver: gurobi")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")

    best_master_solution_for_lb = {
        'u': u_current.copy(),
        'zON': zON_current.copy(),
        'zOFF': zOFF_current.copy(),
        'beta_binary': {b: 0 for b in range(num_beta_bits)}, 
        's_cuts': {} 
    }

    # Initialize best_solution_for_ub outside the loop for broader scope if needed for final display logic
    best_solution_for_ub = None


    for k in range(1, max_iter + 1):
        print(f"========================= Iteration {k} =========================")

        print("--- Solving Subproblem ---")
        sub_u_vals = best_master_solution_for_lb['u'] if k > 1 else u_current
        sub_zON_vals = best_master_solution_for_lb['zON'] if k > 1 else zON_current
        sub_zOFF_vals = best_master_solution_for_lb['zOFF'] if k > 1 else zOFF_current

        subproblem = build_subproblem(sub_u_vals, sub_zON_vals, sub_zOFF_vals)

        if not hasattr(subproblem, 'dual') or not isinstance(subproblem.dual, pyo.Suffix):
            subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif subproblem.dual.direction != pyo.Suffix.IMPORT:
            subproblem.dual.set_direction(pyo.Suffix.IMPORT)

        sub_results = sub_solver.solve(subproblem, tee=False)

        if sub_results.solver.termination_condition == TerminationCondition.optimal or \
           sub_results.solver.termination_condition == TerminationCondition.feasible:

            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {sub_results.solver.termination_condition}")
            print(f"Subproblem Objective (Variable Cost + Penalty): {sub_obj_val:.4f}")

            commitment_cost = sum(gen_data[i]['Csu'] * sub_zON_vals.get((i,t),0) +
                                  gen_data[i]['Csd'] * sub_zOFF_vals.get((i,t),0) +
                                  gen_data[i]['Cf'] * sub_u_vals.get((i,t),0)
                                  for i in generators for t in time_periods)
            current_total_cost = commitment_cost + sub_obj_val
            
            is_logically_feasible = True
            for i_gen in generators:
                for t_time in time_periods:
                    u_val = sub_u_vals.get((i_gen, t_time), 0)
                    u_prev_val = u_initial[i_gen] if t_time == 1 else sub_u_vals.get((i_gen, t_time - 1), 0)
                    zon_val = sub_zON_vals.get((i_gen, t_time), 0)
                    zoff_val = sub_zOFF_vals.get((i_gen, t_time), 0)
                    if abs((u_val - u_prev_val) - (zon_val - zoff_val)) > 1e-4:
                        is_logically_feasible = False
                        print(f"Logic1 violated at G{i_gen}, T{t_time} by solution for UB update.")
                        break
                    if abs(zon_val * zoff_val) > 1e-4 : 
                        is_logically_feasible = False
                        print(f"Logic2 violated at G{i_gen}, T{t_time} by solution for UB update.")
                        break
                if not is_logically_feasible:
                    break
            
            if is_logically_feasible:
                if current_total_cost < upper_bound:
                    upper_bound = current_total_cost
                    best_solution_for_ub = {
                        'u_vals': sub_u_vals.copy(),
                        'zON_vals': sub_zON_vals.copy(),
                        'zOFF_vals': sub_zOFF_vals.copy(),
                        'sub_obj': sub_obj_val,
                        'total_cost': current_total_cost,
                        'iter': k
                    }
                    print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f} from iter {k}")

            print(f"Commitment Cost (for this subproblem's u,z): {commitment_cost:.2f}")
            print(f"Current Total Cost (Commitment + Sub Obj): {current_total_cost:.4f}")
            print(f"Best Upper Bound (Z_UB) so far: {upper_bound:.4f}")

            duals = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}}
            try:
                for i in generators:
                    for t_time_loop in time_periods: # Renamed t to avoid conflict
                        duals['lambda_min'][(i,t_time_loop)] = subproblem.dual.get(subproblem.MinPower[i,t_time_loop], 0.0)
                        duals['lambda_max'][(i,t_time_loop)] = subproblem.dual.get(subproblem.MaxPower[i,t_time_loop], 0.0)
                        duals['lambda_ru'][(i,t_time_loop)]  = subproblem.dual.get(subproblem.RampUp[i,t_time_loop], 0.0)
                        duals['lambda_rd'][(i,t_time_loop)]  = subproblem.dual.get(subproblem.RampDown[i,t_time_loop], 0.0)
            except Exception as e:
                print(f"Warning: Error extracting duals: {e}. Using 0.0 for cut generation.")
                for key_dual in duals: # Renamed key to avoid conflict
                    for i_gen in generators:
                        for t_time_loop in time_periods:
                            duals[key_dual][(i_gen,t_time_loop)] = 0.0

            iteration_data.append({
                'iter': k,
                'sub_obj': sub_obj_val, 
                'duals': duals,      
                'u_vals': sub_u_vals.copy(), 
                'zON_vals': sub_zON_vals.copy(),
                'zOFF_vals': sub_zOFF_vals.copy()
            })
        else:
            print(f"Subproblem FAILED to solve optimally! Status: {sub_results.solver.termination_condition}")
            print("Terminating Benders loop due to subproblem error.")
            break

        print(f"Current Lower Bound (Z_LB from master): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB from subproblem): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
            gap = (upper_bound - lower_bound)
            print(f"Current Gap: {gap:.6f} (Tolerance: {epsilon})")
            if gap <= epsilon and k > 1: 
                print("\nConvergence tolerance met.")
                break
        else:
            print("Gap cannot be calculated yet (LB or UB is inf).")

        if k == max_iter:
            print("\nMaximum iterations reached.")
            break

        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data)

        if k > 1:
            for i_gen in master_problem.I:
                for t_time in master_problem.T:
                    master_problem.u[i_gen, t_time].value = best_master_solution_for_lb['u'].get((i_gen, t_time), 0)
                    master_problem.zON[i_gen, t_time].value = best_master_solution_for_lb['zON'].get((i_gen, t_time), 0)
                    master_problem.zOFF[i_gen, t_time].value = best_master_solution_for_lb['zOFF'].get((i_gen, t_time), 0)
            for bit_idx in master_problem.BETA_BITS:
                master_problem.beta_binary[bit_idx].value = best_master_solution_for_lb['beta_binary'].get(bit_idx,0)
            for cut_idx in master_problem.Cuts: 
                master_problem.s_cuts[cut_idx].value = best_master_solution_for_lb['s_cuts'].get(cut_idx, 0)

        master_solver.options['NumericFocus'] = 3
        #master_solver.options['Presolve'] = 0
        #master_solver.options['NonConvex'] = 2
        master_results = master_solver.solve(master_problem, tee=True) 

        if master_results.solver.termination_condition == TerminationCondition.optimal or \
           master_results.solver.termination_condition == TerminationCondition.locallyOptimal or \
           master_results.solver.termination_condition == TerminationCondition.feasible :

            master_obj_val_from_solver = pyo.value(master_problem.OBJ)
            print(f"Master Status: {master_results.solver.termination_condition}")
            print(f"Master Problem Raw Objective Value (Full QUBO): {master_obj_val_from_solver:.4f}")
            
            # --- START MOVED BLOCK ---
            # Extract solution values from master_problem IMMEDIATELY after solve
            beta_val = sum((2**j) * pyo.value(master_problem.beta_binary[j]) for j in master_problem.BETA_BITS)
            commitment_cost_master = sum(pyo.value(master_problem.Cf[i] * master_problem.u[i, t_loop]) +
                                         pyo.value(master_problem.Csu[i] * master_problem.zON[i, t_loop]) +
                                         pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t_loop])
                                         for i in master_problem.I for t_loop in master_problem.T) # Renamed t to t_loop
            
            u_sol = {(i_gen,t_time): pyo.value(master_problem.u[i_gen,t_time]) for i_gen in generators for t_time in time_periods}
            zON_sol = {(i_gen,t_time): pyo.value(master_problem.zON[i_gen,t_time]) for i_gen in generators for t_time in time_periods}
            zOFF_sol = {(i_gen,t_time): pyo.value(master_problem.zOFF[i_gen,t_time]) for i_gen in generators for t_time in time_periods}
            s_cuts_sol = {c_idx: pyo.value(master_problem.s_cuts[c_idx]) for c_idx in master_problem.Cuts}
            beta_binary_sol = {j_bit: pyo.value(master_problem.beta_binary[j_bit]) for j_bit in master_problem.BETA_BITS} # Renamed j to j_bit
            # --- END MOVED BLOCK ---

            print(f"Master Problem Beta Value: {beta_val:.4f}")
            print(f"Master Problem Commitment Cost: {commitment_cost_master:.4f}")
            print("beta + commitment cost = ", beta_val + commitment_cost_master)

            print("\n--- Verifying Benders Optimality Cuts Satisfaction (using current master solution) ---")
            if not iteration_data:
                print("No Benders cuts to verify yet.")
            else:
                for k_idx, cut_data in enumerate(iteration_data): 
                    sub_obj_k = cut_data['sub_obj']
                    duals_k = cut_data['duals']
                    u_k = cut_data['u_vals']         
                    zON_k = cut_data['zON_vals']     
                    zOFF_k = cut_data['zOFF_vals']  
                    cut_rhs_value_verified = sub_obj_k
                    
                    for i_gen_loop in generators: # Renamed i to avoid conflict
                        for t_time_loop in time_periods: # Renamed t to avoid conflict
                            cut_rhs_value_verified += duals_k['lambda_min'].get((i_gen_loop, t_time_loop), 0.0) * \
                                gen_data[i_gen_loop]['Pmin'] * (u_sol.get((i_gen_loop,t_time_loop),0.0) - u_k.get((i_gen_loop,t_time_loop), 0.0))
                            cut_rhs_value_verified += duals_k['lambda_max'].get((i_gen_loop, t_time_loop), 0.0) * \
                                (gen_data[i_gen_loop]['Pmax']) * (u_sol.get((i_gen_loop,t_time_loop),0.0) - u_k.get((i_gen_loop,t_time_loop), 0.0))
                            

                    for i_gen_loop in generators: # Renamed i to avoid conflict
                        for t_time_loop in time_periods: # Renamed t to avoid conflict
                            term_ramp_up_verified = 0
                            u_prev_k_val = u_initial[i_gen_loop] if t_time_loop == 1 else u_k.get((i_gen_loop,t_time_loop-1), 0.0)
                            if t_time_loop > 1:
                                term_ramp_up_verified += gen_data[i_gen_loop]['Ru'] * (u_sol.get((i_gen_loop,t_time_loop-1),0.0) - u_prev_k_val)
                            term_ramp_up_verified += gen_data[i_gen_loop]['Rsu'] * (zON_sol.get((i_gen_loop,t_time_loop),0.0) - zON_k.get((i_gen_loop,t_time_loop), 0.0))
                            cut_rhs_value_verified += duals_k['lambda_ru'].get((i_gen_loop,t_time_loop),0.0) * term_ramp_up_verified

                            term_ramp_down_verified = 0
                            term_ramp_down_verified += gen_data[i_gen_loop]['Rd'] * (u_sol.get((i_gen_loop,t_time_loop),0.0) - u_k.get((i_gen_loop,t_time_loop), 0.0))
                            term_ramp_down_verified += gen_data[i_gen_loop]['Rsd'] * (zOFF_sol.get((i_gen_loop,t_time_loop),0.0) - zOFF_k.get((i_gen_loop,t_time_loop), 0.0))
                            cut_rhs_value_verified += duals_k['lambda_rd'].get((i_gen_loop,t_time_loop),0.0) * term_ramp_down_verified
                    
                    h_value_verified = beta_val - cut_rhs_value_verified 
                    s_cut_val_for_this_cut = s_cuts_sol.get(k_idx, 0.0) 

                    term_in_square_verified = h_value_verified - s_cut_val_for_this_cut
                    individual_penalty_contribution = pyo.value(master_problem.lambda_benderscut) * (term_in_square_verified**2)

                    print(f"Cut {k_idx+1} (from iter {cut_data['iter']}):")
                    print(f"  beta_val: {beta_val:.4f}")
                    print(f"  Calculated cut_RHS_verified (using master sol): {cut_rhs_value_verified:.4f}")
                    print(f"  h_verified (beta - cut_RHS_verified): {h_value_verified:.4f}")
                    print(f"  s_cuts[{k_idx}]_val from master sol: {s_cut_val_for_this_cut:.4f}") 
                    print(f"  Term in square (h_verified - s_cut_val): {term_in_square_verified:.4f}")
                    print(f"  Individual Penalty Contr. (lambda * term^2): {individual_penalty_contribution:.4f}")

                    tolerance = 1e-4 
                    if h_value_verified >= -tolerance:
                        print(f"  Original Cut Sense (beta >= cut_RHS_verified): SATISFIED.")
                    else:
                        print(f"  Original Cut Sense (beta >= cut_RHS_verified): VIOLATED by {-h_value_verified:.4f}.")
                    
                    if abs(term_in_square_verified) < 0.01: 
                         print(f"  Penalty term effect: constraint violation effectively handled by s_cut or small difference.")
                    else:
                         print(f"  Penalty term effect: constraint violation NOT fully handled to make term small (h_verified={h_value_verified:.4f}, s_cut={s_cut_val_for_this_cut:.4f}).")
            print("--- End of Benders Cuts Verification (Corrected) ---")
            
            # Now calculations for logic1_penalty_val, logic2_penalty_val, benders_penalty_total_val
            # These use pyo.value(master_problem.VAR) which is fine as they re-evaluate based on the solved model.
            logic1_penalty_val = pyo.value(master_problem.lambda_logic1) * sum( ( (pyo.value(master_problem.u[i_gen, t_time]) - pyo.value(master_problem.u_prev[i_gen, t_time])) - (pyo.value(master_problem.zON[i_gen, t_time]) - pyo.value(master_problem.zOFF[i_gen, t_time])) )**2 for i_gen in master_problem.I for t_time in master_problem.T )
            logic2_penalty_val = pyo.value(master_problem.lambda_logic2) * sum( pyo.value(master_problem.zON[i_gen, t_time]) * pyo.value(master_problem.zOFF[i_gen, t_time]) for i_gen in master_problem.I for t_time in master_problem.T )
            print(f"Master Sol: Logic1 Penalty = {logic1_penalty_val:.4f}, Logic2 Penalty = {logic2_penalty_val:.4f}")

            benders_penalty_total_val = 0
            if len(iteration_data)>0: 
                binary_beta_expr_val = beta_val # Use already calculated beta_val
                for k_idx_loop in master_problem.Cuts:
                    data = iteration_data[k_idx_loop]
                    sub_obj_k_loop = data['sub_obj'] # Renamed sub_obj_k
                    duals_k_loop = data['duals']     # Renamed duals_k
                    u_k_vals_loop = data['u_vals']
                    zON_k_vals_loop = data['zON_vals']
                    zOFF_k_vals_loop = data['zOFF_vals']
                    
                    cut_rhs_expr_val = sub_obj_k_loop
                    for i_gen_loop in master_problem.I: # Renamed i
                        for t_time_loop in master_problem.T: # Renamed t_loop
                            dual_val_loop = duals_k_loop['lambda_min'].get((i_gen_loop, t_time_loop), 0.0) # Renamed dual_val
                            cut_rhs_expr_val += dual_val_loop * (pyo.value(master_problem.Pmin_param[i_gen_loop]) * (pyo.value(master_problem.u[i_gen_loop, t_time_loop]) - u_k_vals_loop.get((i_gen_loop,t_time_loop), 0.0)))
                    for i_gen_loop in master_problem.I: # Renamed i
                        for t_time_loop in master_problem.T: # Renamed t_loop
                            dual_val_loop = duals_k_loop['lambda_max'].get((i_gen_loop, t_time_loop), 0.0) # Renamed dual_val
                            cut_rhs_expr_val += dual_val_loop * (pyo.value(master_problem.Pmax_param[i_gen_loop]) * (pyo.value(master_problem.u[i_gen_loop, t_time_loop]) - u_k_vals_loop.get((i_gen_loop,t_time_loop), 0.0)))
                    for i_gen_loop in master_problem.I: # Renamed i
                        for t_time_loop in master_problem.T: # Renamed t_loop
                            dual_val_loop = duals_k_loop['lambda_ru'].get((i_gen_loop, t_time_loop), 0.0) # Renamed dual_val
                            u_prev_term_val_loop = 0 # Renamed u_prev_term_val
                            u_prev_k_val_loop = u_k_vals_loop.get((i_gen_loop, t_time_loop-1), pyo.value(master_problem.u_init[i_gen_loop])) if t_time_loop > 1 else pyo.value(master_problem.u_init[i_gen_loop])
                            current_u_prev_val_loop = pyo.value(master_problem.u_init[i_gen_loop]) if t_time_loop == 1 else pyo.value(master_problem.u[i_gen_loop, t_time_loop-1])
                            if t_time_loop > 1:
                                u_prev_term_val_loop = pyo.value(master_problem.Ru_param[i_gen_loop]) * (current_u_prev_val_loop - u_prev_k_val_loop)
                            zON_term_val_loop = pyo.value(master_problem.Rsu_param[i_gen_loop]) * (pyo.value(master_problem.zON[i_gen_loop, t_time_loop]) - zON_k_vals_loop.get((i_gen_loop, t_time_loop), 0.0))
                            cut_rhs_expr_val += dual_val_loop * (u_prev_term_val_loop + zON_term_val_loop)
                    for i_gen_loop in master_problem.I: # Renamed i
                        for t_time_loop in master_problem.T: # Renamed t_loop
                            dual_val_loop = duals_k_loop['lambda_rd'].get((i_gen_loop, t_time_loop), 0.0) # Renamed dual_val
                            u_term_val_loop = pyo.value(master_problem.Rd_param[i_gen_loop]) * (pyo.value(master_problem.u[i_gen_loop, t_time_loop]) - u_k_vals_loop.get((i_gen_loop, t_time_loop), 0.0))
                            zOFF_term_val_loop = pyo.value(master_problem.Rsd_param[i_gen_loop]) * (pyo.value(master_problem.zOFF[i_gen_loop, t_time_loop]) - zOFF_k_vals_loop.get((i_gen_loop, t_time_loop), 0.0))
                            cut_rhs_expr_val += dual_val_loop * (u_term_val_loop + zOFF_term_val_loop)
                    
                    s_cut_val_loop = s_cuts_sol.get(k_idx_loop, 0.0) # Use the extracted s_cuts_sol
                    penalty_for_this_cut_val = (binary_beta_expr_val - s_cut_val_loop - cut_rhs_expr_val)**2
                    benders_penalty_total_val += pyo.value(master_problem.lambda_benderscut) * penalty_for_this_cut_val
            print(f"Master Sol: Benders Cuts Penalty Total (recalculated): {benders_penalty_total_val:.4f}")
            
            # Lower bound update logic
            current_potential_lb = commitment_cost_master + beta_val # Use beta_val not beta_val_master
            if logic1_penalty_val < 1e-3 and logic2_penalty_val < 1e-3 and benders_penalty_total_val < 1e-3 : 
                lower_bound = max(lower_bound, current_potential_lb)
                print(f"Master problem penalties are small. Commitment Cost + Beta = {current_potential_lb:.4f}")
            else:
                # Retain previous lower_bound if penalties are not small, 
                # as current_potential_lb is not a true LB to original problem then.
                # Or, use master_obj_val_from_solver if it's guaranteed to be non-decreasing for the QUBO.
                # For now, keeping old LB is safer if penalties are large.
                print(f"Master problem penalties are NOT small ({logic1_penalty_val:.2f}, {logic2_penalty_val:.2f}, {benders_penalty_total_val:.2f}). Using old LB or QUBO obj based LB strategy.")
                # If penalties are high, the master_obj_val_from_solver is a lower bound to the penalized problem.
                # We might want to ensure lower_bound is non-decreasing.
                lower_bound = max(lower_bound, master_obj_val_from_solver) # Update LB with raw QUBO value


            print(f"Master Commitment Cost: {commitment_cost_master:.4f}") # Already printed
            print(f"Master Beta Value: {beta_val:.4f}") # Already printed
            print(f"Updated Lower Bound (Z_LB): {lower_bound:.4f}")

            # Store values for the next subproblem solve / warm start
            best_master_solution_for_lb['u'] = u_sol
            best_master_solution_for_lb['zON'] = zON_sol
            best_master_solution_for_lb['zOFF'] = zOFF_sol
            best_master_solution_for_lb['beta_binary'] = beta_binary_sol 
            best_master_solution_for_lb['s_cuts'] = s_cuts_sol

        else: # Master problem failed
            print(f"Master Problem FAILED to solve optimally! Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop due to master error.")
            break # out of Benders loop for k

    # --- End of Benders Loop ---
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    # ... (rest of your final print statements) ...
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound)
    print(f"Final Gap: {final_gap:.6f}")
    iterations_run = 0
    if 'k' in locals() and k > 0 : # Check if k was defined (i.e. loop ran at least once)
        iterations_run = k if k <= max_iter else max_iter
        if master_results.solver.termination_condition != TerminationCondition.optimal and \
           master_results.solver.termination_condition != TerminationCondition.locallyOptimal and \
           master_results.solver.termination_condition != TerminationCondition.feasible and \
           sub_results.solver.termination_condition != TerminationCondition.optimal and \
           sub_results.solver.termination_condition != TerminationCondition.feasible:
             # If loop broke early due to solver failure, k might be the iter it failed ON.
             # If it completed all iters or converged, k is iter+1, so k-1 or the actual iter count.
             # The current 'k' is the one that *would have been* next or is max_iter + 1
             pass # iteration_data[-1]['iter'] would be the last successful one.
    
    print(f"Iterations Performed: {len(iteration_data)}") # More reliable count of completed iterations
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    if best_solution_for_ub: # Check if a feasible UB was ever found
        print("\n--- Best Solution Found (leading to best Z_UB in iter {}) ---".format(best_solution_for_ub['iter']))
        print(f"Best Total Cost (Upper Bound): {best_solution_for_ub['total_cost']:.4f}")
        print("Commitment Schedule (u_it):")
        for t_time_loop in time_periods: # Renamed t
            print(f"  t={t_time_loop}: ", {i_gen: round(best_solution_for_ub['u_vals'].get((i_gen,t_time_loop),0)) for i_gen in generators})

        print("\nFinal Dispatch (p_it) for Best UB Solution:")
        final_subproblem = build_subproblem(best_solution_for_ub['u_vals'], best_solution_for_ub['zON_vals'], best_solution_for_ub['zOFF_vals'])
        if not hasattr(final_subproblem, 'dual') or not isinstance(final_subproblem.dual, pyo.Suffix):
            final_subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif final_subproblem.dual.direction != pyo.Suffix.IMPORT:
             final_subproblem.dual.set_direction(pyo.Suffix.IMPORT)

        sub_solver.solve(final_subproblem, tee=False)
        if pyo.value(final_subproblem.OBJ) is not None:
            for t_time_loop in time_periods: # Renamed t
                print(f"  t={t_time_loop}: ", {i_gen: f"{pyo.value(final_subproblem.p[i_gen,t_time_loop]):.2f}" for i_gen in generators})
            final_sub_obj_val = pyo.value(final_subproblem.OBJ)
            print(f"Final Subproblem Objective (Var Cost + Penalty): {final_sub_obj_val:.4f}")
            final_commit_cost = sum(gen_data[i_gen]['Csu'] * best_solution_for_ub['zON_vals'].get((i_gen, t_time_loop),0) +
                                    gen_data[i_gen]['Csd'] * best_solution_for_ub['zOFF_vals'].get((i_gen, t_time_loop),0) +
                                    gen_data[i_gen]['Cf'] * best_solution_for_ub['u_vals'].get((i_gen, t_time_loop),0)
                                    for i_gen in generators for t_time_loop in time_periods)
            print(f"Final Commitment Cost: {final_commit_cost:.2f}")
            print(f"Final Total Cost (recalculated for best UB): {final_commit_cost + final_sub_obj_val:.4f}")
            print("Final Demand Slack:")
            for t_time_loop in time_periods: # Renamed t
                print(f"  t={t_time_loop}: {pyo.value(final_subproblem.demand_slack[t_time_loop]):.4f}")
        else:
            print("Could not resolve final subproblem to show dispatch for best UB.")
    else:
        print("\nNo feasible solution found or Benders loop terminated before a UB was established.")

if __name__ == '__main__':
    main()