##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo, classically
##### THe logic constraints, both types, are penalty terms
##### Benders optimality cuts are also penalty terms.
##### An additional penalty ensures at least one generator is ON per time period.
##### Fix for iteration 1 unbounded master problem.

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
M_penalty = 5 # User's current value, strongly recommend increasing this (e.g., to 200 or 500)

# Number of bits for beta variable
num_beta_bits = 18 # This contributes to large coefficients. Consider reducing if possible for testing.

# Function creating the sub problem
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    model = pyo.ConcreteModel(name="UCP_Subproblem")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)

    if not u_fixed_vals:
        print("ERROR: u_fixed_vals is empty or None in build_subproblem. This should not happen.")
        # Fallback or raise error
        # For now, let's create a default to avoid crashing Pyomo Param initialization
        # but this indicates a logic error upstream.
        u_fixed_vals = {(i,t): 0.0 for i in model.I for t in model.T} 
        # A better approach might be to raise an error:
        # raise ValueError("u_fixed_vals is empty in build_subproblem")


    u_fixed_param_vals = {(i,t): u_fixed_vals.get((i,t), 0.0) for i in model.I for t in model.T}
    zON_fixed_param_vals = {(i,t): zON_fixed_vals.get((i,t), 0.0) for i in model.I for t in model.T}
    zOFF_fixed_param_vals = {(i,t): zOFF_fixed_vals.get((i,t), 0.0) for i in model.I for t in model.T}

    model.u_fixed = pyo.Param(model.I, model.T, initialize=u_fixed_param_vals, mutable=True)
    model.zON_fixed = pyo.Param(model.I, model.T, initialize=zON_fixed_param_vals, mutable=True)
    model.zOFF_fixed = pyo.Param(model.I, model.T, initialize=zOFF_fixed_param_vals, mutable=True)

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

    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)
    model.demand_slack = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)

    def objective_rule(m):
        variable_cost = sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
        penalty_cost = m.M * sum(m.demand_slack[t] for t in m.T)
        return variable_cost + penalty_cost
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def p_prev_rule(m, i, t):
        if t == 1: return m.p_init[i]
        else: return m.p[i, t-1]
    model.p_prev = pyo.Expression(model.I, model.T, rule=p_prev_rule)

    def u_prev_fixed_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u_fixed[i, t-1]
    model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=u_prev_fixed_rule)

    def min_power_rule(m, i, t): return m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t]
    model.MinPower = pyo.Constraint(model.I, model.T, rule=min_power_rule)

    def max_power_rule(m, i, t): return m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t]
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=max_power_rule)

    def ramp_up_rule(m, i, t): return m.p[i, t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t]
    model.RampUp = pyo.Constraint(model.I, model.T, rule=ramp_up_rule)

    def ramp_down_rule(m, i, t): return m.p_prev[i,t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t]
    model.RampDown = pyo.Constraint(model.I, model.T, rule=ramp_down_rule)

    def demand_rule(m, t): return sum(m.p[i, t] for i in m.I) + m.demand_slack[t] >= m.D[t]
    model.Demand = pyo.Constraint(model.T, rule=demand_rule)

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model

# Function creating the master problem
def build_master(iteration_data_param): # Renamed to avoid conflict with global iteration_data
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Unconstrained_With_OnPenalty")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)
    model.Cuts = pyo.Set(initialize=range(len(iteration_data_param)))

    model.Pmax_param = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.lambda_logic1 = pyo.Param(initialize=100) 
    model.lambda_logic2 = pyo.Param(initialize=100) 
    model.lambda_benderscut = pyo.Param(initialize=500) 
    model.lambda_on = pyo.Param(initialize=100)

    model.Pmin_param = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
    model.Rd_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})   
    model.Rsd_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})  
    model.Ru_param = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})    
    model.Rsu_param = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})  

    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary)
    model.s_cuts = pyo.Var(model.Cuts, within=pyo.NonNegativeReals) 
    model.s_on = pyo.Var(model.T, within=pyo.NonNegativeReals)

    def u_prev_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    def master_objective_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )
        on_penalty_term = m.lambda_on * sum( ( (sum(m.u[i, t_on_loop] for i in m.I)) - m.s_on[t_on_loop] - 1 )**2 for t_on_loop in m.T )

        obj_expr_val = commitment_cost + logic1_term + logic2_term + binary_beta_expr + on_penalty_term

        if len(iteration_data_param) > 0: # Only add Benders penalty term if there are cuts
            benders_penalty_term_sum_val = 0
            for k_idx_obj in m.Cuts: 
                data_obj = iteration_data_param[k_idx_obj]
                sub_obj_k_obj = data_obj['sub_obj']
                duals_k_obj = data_obj['duals']
                u_k_vals_obj = data_obj['u_vals']
                zON_k_vals_obj = data_obj['zON_vals']
                zOFF_k_vals_obj = data_obj['zOFF_vals']
                
                cut_rhs_terms_list = [sub_obj_k_obj] 

                for i_obj in m.I: 
                    for t_obj in m.T: 
                        dual_min_obj = duals_k_obj['lambda_min'].get((i_obj, t_obj), 0.0)
                        if abs(dual_min_obj) > 1e-9:
                            cut_rhs_terms_list.append(dual_min_obj * (m.Pmin_param[i_obj] * (m.u[i_obj, t_obj] - u_k_vals_obj.get((i_obj,t_obj), 0.0))))
                        
                        dual_max_obj = duals_k_obj['lambda_max'].get((i_obj, t_obj), 0.0)
                        if abs(dual_max_obj) > 1e-9:
                             cut_rhs_terms_list.append(dual_max_obj * (-m.Pmax_param[i_obj] * (m.u[i_obj, t_obj] - u_k_vals_obj.get((i_obj,t_obj), 0.0))))
                
                for i_obj in m.I: 
                    for t_obj in m.T:
                        dual_ru_obj = duals_k_obj['lambda_ru'].get((i_obj, t_obj), 0.0) 
                        if abs(dual_ru_obj) > 1e-9:
                            u_prev_master_var = m.u[i_obj, t_obj-1] if t_obj > 1 else m.u_init[i_obj]
                            u_prev_k_val_for_cut = u_k_vals_obj.get((i_obj, t_obj-1), m.u_init[i_obj]) if t_obj > 1 else m.u_init[i_obj]
                            
                            u_prev_term_expr = 0
                            if t_obj > 1: # Only apply ramp if not t=1
                                u_prev_term_expr = m.Ru_param[i_obj] * (u_prev_master_var - u_prev_k_val_for_cut)
                            
                            zON_term_expr = m.Rsu_param[i_obj] * (m.zON[i_obj, t_obj] - zON_k_vals_obj.get((i_obj, t_obj), 0.0))
                            cut_rhs_terms_list.append(dual_ru_obj * (u_prev_term_expr + zON_term_expr))

                        dual_rd_obj = duals_k_obj['lambda_rd'].get((i_obj, t_obj), 0.0) 
                        if abs(dual_rd_obj) > 1e-9:
                            u_term_expr = m.Rd_param[i_obj] * (m.u[i_obj, t_obj] - u_k_vals_obj.get((i_obj, t_obj), 0.0))
                            zOFF_term_expr = m.Rsd_param[i_obj] * (m.zOFF[i_obj, t_obj] - zOFF_k_vals_obj.get((i_obj, t_obj), 0.0))
                            cut_rhs_terms_list.append(dual_rd_obj * (u_term_expr + zOFF_term_expr))
                
                cut_rhs_final_expr = sum(cut_rhs_terms_list)
                penalty_for_this_cut_expr = (binary_beta_expr - m.s_cuts[k_idx_obj] - cut_rhs_final_expr)**2
                benders_penalty_term_sum_val += m.lambda_benderscut * penalty_for_this_cut_expr
            obj_expr_val += benders_penalty_term_sum_val
            
        return obj_expr_val
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)
    return model

# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 
    epsilon = 1 
    iteration_data_main_loop = [] # Renamed to avoid conflict with build_master param
    lower_bound = -float('inf')
    upper_bound = float('inf')

    u_current_init = {}
    zON_current_init = {}
    zOFF_current_init = {}
    for t_init_loop in time_periods:
        for i_init_loop in generators:
            u_current_init[i_init_loop, t_init_loop] = 1.0 
            u_prev_init_val_loop = u_initial[i_init_loop] if t_init_loop == 1 else u_current_init.get((i_init_loop, t_init_loop-1), u_initial[i_init_loop])
            if u_current_init[i_init_loop, t_init_loop] > 0.5 and u_prev_init_val_loop < 0.5:
                zON_current_init[i_init_loop, t_init_loop] = 1.0
                zOFF_current_init[i_init_loop, t_init_loop] = 0.0
            elif u_current_init[i_init_loop, t_init_loop] < 0.5 and u_prev_init_val_loop > 0.5:
                zON_current_init[i_init_loop, t_init_loop] = 0.0
                zOFF_current_init[i_init_loop, t_init_loop] = 1.0
            else:
                zON_current_init[i_init_loop, t_init_loop] = 0.0
                zOFF_current_init[i_init_loop, t_init_loop] = 0.0

    solver_name = "gurobi" 
    master_solver = SolverFactory(solver_name)
    sub_solver = SolverFactory("gurobi") 

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Using Master Solver: {solver_name}, Sub Solver: gurobi")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")

    best_master_solution_for_lb = {
        'u': u_current_init.copy(),
        'zON': zON_current_init.copy(),
        'zOFF': zOFF_current_init.copy(),
        'beta_binary': {b: 0 for b in range(num_beta_bits)}, 
        's_cuts': {},
        's_on': {t: 0.0 for t in time_periods} 
    }
    
    best_solution_for_ub = None
    last_master_results = None 
    last_sub_results = None    

    for k in range(1, max_iter + 1):
        print(f"========================= Iteration {k} =========================")
        print("--- Solving Subproblem ---")
        sub_u_vals = best_master_solution_for_lb['u'] 
        sub_zON_vals = best_master_solution_for_lb['zON']
        sub_zOFF_vals = best_master_solution_for_lb['zOFF']
        
        if not sub_u_vals:
             print("CRITICAL WARNING: sub_u_vals is empty before building subproblem in iteration", k)
             if k == 1:
                 print("Fallback to u_current_init for subproblem input in iter 1")
                 sub_u_vals = u_current_init.copy()
                 sub_zON_vals = zON_current_init.copy()
                 sub_zOFF_vals = zOFF_current_init.copy()
             else:
                 print("ERROR: sub_u_vals became empty in a later iteration. Terminating.")
                 break
        
        subproblem = build_subproblem(sub_u_vals, sub_zON_vals, sub_zOFF_vals)

        if not hasattr(subproblem, 'dual') or not isinstance(subproblem.dual, pyo.Suffix):
            subproblem.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif subproblem.dual.direction != pyo.Suffix.IMPORT:
            subproblem.dual.set_direction(pyo.Suffix.IMPORT)

        last_sub_results = sub_solver.solve(subproblem, tee=False)

        if last_sub_results.solver.termination_condition == TerminationCondition.optimal or \
           last_sub_results.solver.termination_condition == TerminationCondition.feasible:

            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {last_sub_results.solver.termination_condition}")
            print(f"Subproblem Objective (Variable Cost + Penalty): {sub_obj_val:.4f}")

            commitment_cost_sub_input = sum(gen_data[i_gen]['Csu'] * sub_zON_vals.get((i_gen,t_time),0) +
                                   gen_data[i_gen]['Csd'] * sub_zOFF_vals.get((i_gen,t_time),0) +
                                   gen_data[i_gen]['Cf'] * sub_u_vals.get((i_gen,t_time),0)
                                   for i_gen in generators for t_time in time_periods)
            current_total_cost = commitment_cost_sub_input + sub_obj_val
            
            is_logically_feasible_for_ub = True
            for i_gen_ub in generators:
                for t_time_ub in time_periods:
                    u_val_ub = sub_u_vals.get((i_gen_ub, t_time_ub), 0.0) # Default to 0.0 if key missing
                    u_prev_val_ub = u_initial[i_gen_ub] if t_time_ub == 1 else sub_u_vals.get((i_gen_ub, t_time_ub - 1), 0.0)
                    zon_val_ub = sub_zON_vals.get((i_gen_ub, t_time_ub), 0.0)
                    zoff_val_ub = sub_zOFF_vals.get((i_gen_ub, t_time_ub), 0.0)
                    if abs((u_val_ub - u_prev_val_ub) - (zon_val_ub - zoff_val_ub)) > 1e-4:
                        is_logically_feasible_for_ub = False
                        print(f"Logic1 VIOLATED by master solution G{i_gen_ub}, T{t_time_ub} used for UB update. Diff: {(u_val_ub - u_prev_val_ub) - (zon_val_ub - zoff_val_ub)}")
                        break
                    if abs(zon_val_ub * zoff_val_ub) > 1e-4 : 
                        is_logically_feasible_for_ub = False
                        print(f"Logic2 VIOLATED by master solution G{i_gen_ub}, T{t_time_ub} used for UB update. Product: {zon_val_ub * zoff_val_ub}")
                        break
                if not is_logically_feasible_for_ub:
                    break
            
            is_at_least_one_on_for_ub = True
            if is_logically_feasible_for_ub:
                for t_time_ub in time_periods:
                    sum_u_t = sum(sub_u_vals.get((i_gen_ub, t_time_ub), 0) for i_gen_ub in generators)
                    if sum_u_t < 0.5: 
                        is_at_least_one_on_for_ub = False
                        print(f"At-least-one-on VIOLATED by master solution at T{t_time_ub} for UB update. Sum u: {sum_u_t}")
                        break
            
            if is_logically_feasible_for_ub and is_at_least_one_on_for_ub:
                if current_total_cost < upper_bound:
                    upper_bound = current_total_cost
                    best_solution_for_ub = {
                        'u_vals': sub_u_vals.copy(), 'zON_vals': sub_zON_vals.copy(),
                        'zOFF_vals': sub_zOFF_vals.copy(), 'sub_obj': sub_obj_val,
                        'total_cost': current_total_cost, 'iter': k
                    }
                    print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f} from iter {k} (Master sol was feasible for UB)")
            else:
                print(f"Master solution for subproblem was NOT suitable for UB update. Current total cost: {current_total_cost:.4f}")

            print(f"Commitment Cost (for this subproblem's u,z): {commitment_cost_sub_input:.2f}")
            print(f"Current Total Cost (Commitment + Sub Obj): {current_total_cost:.4f}")
            print(f"Best Upper Bound (Z_UB) so far: {upper_bound:.4f}")

            duals = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}}
            # Dual extraction logic (remains same)
            try:
                for i_dual in generators:
                    for t_dual in time_periods: 
                        duals['lambda_min'][(i_dual,t_dual)] = subproblem.dual.get(subproblem.MinPower[i_dual,t_dual], 0.0)
                        duals['lambda_max'][(i_dual,t_dual)] = subproblem.dual.get(subproblem.MaxPower[i_dual,t_dual], 0.0)
                        duals['lambda_ru'][(i_dual,t_dual)]  = subproblem.dual.get(subproblem.RampUp[i_dual,t_dual], 0.0)
                        duals['lambda_rd'][(i_dual,t_dual)]  = subproblem.dual.get(subproblem.RampDown[i_dual,t_dual], 0.0)
            except Exception as e_dual: # Catching specific errors can be better
                print(f"Warning: Error extracting duals: {e_dual}. Using 0.0 for cut generation.")
                # Initialize all duals to 0.0 if any error occurs
                for key_dual_loop in duals: 
                    for i_gen_dual_loop in generators:
                        for t_time_dual_loop in time_periods:
                            duals[key_dual_loop][(i_gen_dual_loop,t_time_dual_loop)] = 0.0


            iteration_data_main_loop.append({
                'iter': k, 'sub_obj': sub_obj_val, 'duals': duals,     
                'u_vals': sub_u_vals.copy(), 'zON_vals': sub_zON_vals.copy(),
                'zOFF_vals': sub_zOFF_vals.copy()
            })
        else:
            print(f"Subproblem FAILED to solve optimally! Status: {last_sub_results.solver.termination_condition}")
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
        master_problem = build_master(iteration_data_main_loop) # Pass the correct iteration_data
        
        # Set Gurobi options for numerical stability
        master_solver.options['NumericFocus'] = 3 
        master_solver.options['ScaleFlag'] = 2    # Aggressive scaling
        # master_solver.options['FeasibilityTol'] = 1e-9 # Tighter feasibility tolerance if needed
        # master_solver.options['OptimalityTol'] = 1e-9 # Tighter optimality tolerance if needed


        # Warm start master problem
        for i_gen_warm in master_problem.I:
            for t_time_warm in master_problem.T:
                master_problem.u[i_gen_warm, t_time_warm].value = best_master_solution_for_lb['u'].get((i_gen_warm, t_time_warm), 0)
                master_problem.zON[i_gen_warm, t_time_warm].value = best_master_solution_for_lb['zON'].get((i_gen_warm, t_time_warm), 0)
                master_problem.zOFF[i_gen_warm, t_time_warm].value = best_master_solution_for_lb['zOFF'].get((i_gen_warm, t_time_warm), 0)
        for bit_idx_warm in master_problem.BETA_BITS:
            master_problem.beta_binary[bit_idx_warm].value = best_master_solution_for_lb['beta_binary'].get(bit_idx_warm,0)
        
        # s_cuts might not exist in master_problem if iteration_data_main_loop was empty (k=1)
        if len(iteration_data_main_loop) > 0:
            for cut_idx_warm in master_problem.Cuts: 
                master_problem.s_cuts[cut_idx_warm].value = best_master_solution_for_lb['s_cuts'].get(cut_idx_warm, 0)
        
        for t_on_warm in master_problem.T: 
            master_problem.s_on[t_on_warm].value = best_master_solution_for_lb['s_on'].get(t_on_warm, 0)

        last_master_results = master_solver.solve(master_problem, tee=True) 

        if last_master_results.solver.termination_condition == TerminationCondition.optimal or \
           last_master_results.solver.termination_condition == TerminationCondition.locallyOptimal or \
           last_master_results.solver.termination_condition == TerminationCondition.feasible :

            master_obj_val_from_solver = pyo.value(master_problem.OBJ)
            print(f"Master Status: {last_master_results.solver.termination_condition}")
            print(f"Master Problem Raw Objective Value (Full QUBO): {master_obj_val_from_solver:.4f}")
            
            # Extract solution values (remains largely same)
            beta_val_master = sum((2**j_bit) * pyo.value(master_problem.beta_binary[j_bit]) for j_bit in master_problem.BETA_BITS)
            commitment_cost_master_sol = sum(pyo.value(master_problem.Cf[i_m] * master_problem.u[i_m, t_m]) +
                                         pyo.value(master_problem.Csu[i_m] * master_problem.zON[i_m, t_m]) +
                                         pyo.value(master_problem.Csd[i_m] * master_problem.zOFF[i_m, t_m])
                                         for i_m in master_problem.I for t_m in master_problem.T) 
            
            u_sol_master = {(i_g_m,t_t_m): pyo.value(master_problem.u[i_g_m,t_t_m]) for i_g_m in generators for t_t_m in time_periods}
            zON_sol_master = {(i_g_m,t_t_m): pyo.value(master_problem.zON[i_g_m,t_t_m]) for i_g_m in generators for t_t_m in time_periods}
            zOFF_sol_master = {(i_g_m,t_t_m): pyo.value(master_problem.zOFF[i_g_m,t_t_m]) for i_g_m in generators for t_t_m in time_periods}
            
            s_cuts_sol_master = {}
            if len(iteration_data_main_loop) > 0: # s_cuts only exist if cuts exist
                s_cuts_sol_master = {c_idx_m: pyo.value(master_problem.s_cuts[c_idx_m]) for c_idx_m in master_problem.Cuts if master_problem.s_cuts[c_idx_m].value is not None}

            beta_binary_sol_master = {j_bit_m: pyo.value(master_problem.beta_binary[j_bit_m]) for j_bit_m in master_problem.BETA_BITS}
            s_on_sol_master = {t_m_on: pyo.value(master_problem.s_on[t_m_on]) for t_m_on in master_problem.T if master_problem.s_on[t_m_on].value is not None}

            print(f"Master Problem Beta Value: {beta_val_master:.4f}")
            print(f"Master Problem Commitment Cost: {commitment_cost_master_sol:.4f}")
            print(f"Master Problem Sum (Commitment + Beta): {beta_val_master + commitment_cost_master_sol:.4f}")

            # Recalculate penalties based on master solution
            logic1_penalty_val_master = pyo.value(master_problem.lambda_logic1) * sum( ( (u_sol_master.get((i_v,t_v),0) - (u_initial[i_v] if t_v == 1 else u_sol_master.get((i_v,t_v-1),0)) ) - (zON_sol_master.get((i_v,t_v),0) - zOFF_sol_master.get((i_v,t_v),0)) )**2 for i_v in master_problem.I for t_v in master_problem.T )
            logic2_penalty_val_master = pyo.value(master_problem.lambda_logic2) * sum( zON_sol_master.get((i_v,t_v),0) * zOFF_sol_master.get((i_v,t_v),0) for i_v in master_problem.I for t_v in master_problem.T )
            on_penalty_val_master = pyo.value(master_problem.lambda_on) * sum( ( (sum(u_sol_master.get((i_v, t_v_on),0) for i_v in master_problem.I)) - s_on_sol_master.get(t_v_on,0.0) - 1 )**2 for t_v_on in master_problem.T ) # Added .get default for s_on_sol_master
            
            print(f"Master Sol: Logic1 Penalty = {logic1_penalty_val_master:.4f}, Logic2 Penalty = {logic2_penalty_val_master:.4f}, AtLeastOneOn Penalty = {on_penalty_val_master:.4f}")

            benders_penalty_total_val_master = 0
            if iteration_data_main_loop: 
                # Recalculation logic for benders_penalty_total_val_master (remains same)
                for k_idx_verify in master_problem.Cuts:
                    data_verify = iteration_data_main_loop[k_idx_verify]
                    sub_obj_k_verify = data_verify['sub_obj'] 
                    duals_k_verify = data_verify['duals']   
                    u_k_vals_verify = data_verify['u_vals']
                    zON_k_vals_verify = data_verify['zON_vals']
                    zOFF_k_vals_verify = data_verify['zOFF_vals']
                    
                    cut_rhs_expr_val_verify = sub_obj_k_verify # Start with constant
                    
                    # MinPower and MaxPower terms
                    for i_v_b in master_problem.I: 
                        for t_v_b in master_problem.T: 
                            cut_rhs_expr_val_verify += duals_k_verify['lambda_min'].get((i_v_b, t_v_b), 0.0) * (pyo.value(master_problem.Pmin_param[i_v_b]) * (u_sol_master.get((i_v_b,t_v_b),0.0) - u_k_vals_verify.get((i_v_b,t_v_b), 0.0)))
                            cut_rhs_expr_val_verify += duals_k_verify['lambda_max'].get((i_v_b, t_v_b), 0.0) * (-pyo.value(master_problem.Pmax_param[i_v_b]) * (u_sol_master.get((i_v_b,t_v_b),0.0) - u_k_vals_verify.get((i_v_b,t_v_b), 0.0)))
                    
                    # RampUp and RampDown terms
                    for i_v_b in master_problem.I: 
                        for t_v_b in master_problem.T: 
                            # RampUp
                            u_prev_master_val_verify = u_initial[i_v_b] if t_v_b == 1 else u_sol_master.get((i_v_b,t_v_b-1),0.0)
                            u_prev_k_val_verify = u_initial[i_v_b] if t_v_b == 1 else u_k_vals_verify.get((i_v_b,t_v_b-1),0.0)
                            
                            term_ramp_up_u_prev_verify = 0
                            if t_v_b > 1: # Ramp-up u_prev term only if t > 1
                                term_ramp_up_u_prev_verify = gen_data[i_v_b]['Ru'] * (u_prev_master_val_verify - u_prev_k_val_verify)
                            
                            term_ramp_up_zON_verify = gen_data[i_v_b]['Rsu'] * (zON_sol_master.get((i_v_b,t_v_b),0.0) - zON_k_vals_verify.get((i_v_b,t_v_b), 0.0))
                            cut_rhs_expr_val_verify += duals_k_verify['lambda_ru'].get((i_v_b,t_v_b),0.0) * (term_ramp_up_u_prev_verify + term_ramp_up_zON_verify)

                            # RampDown
                            term_ramp_down_u_verify = gen_data[i_v_b]['Rd'] * (u_sol_master.get((i_v_b,t_v_b),0.0) - u_k_vals_verify.get((i_v_b,t_v_b), 0.0))
                            term_ramp_down_zOFF_verify = gen_data[i_v_b]['Rsd'] * (zOFF_sol_master.get((i_v_b,t_v_b),0.0) - zOFF_k_vals_verify.get((i_v_b,t_v_b), 0.0))
                            cut_rhs_expr_val_verify += duals_k_verify['lambda_rd'].get((i_v_b,t_v_b),0.0) * (term_ramp_down_u_verify + term_ramp_down_zOFF_verify)
                    
                    s_cut_val_verify = s_cuts_sol_master.get(k_idx_verify, 0.0) 
                    penalty_for_this_cut_val_verify = (beta_val_master - s_cut_val_verify - cut_rhs_expr_val_verify)**2
                    benders_penalty_total_val_master += pyo.value(master_problem.lambda_benderscut) * penalty_for_this_cut_val_verify

            print(f"Master Sol: Benders Cuts Penalty Total (recalculated): {benders_penalty_total_val_master:.4f}")
            
            current_potential_lb_master = commitment_cost_master_sol + beta_val_master
            penalties_are_small = (logic1_penalty_val_master < 1e-3 and \
                                   logic2_penalty_val_master < 1e-3 and \
                                   on_penalty_val_master < 1e-3 and \
                                   benders_penalty_total_val_master < 1e-3)

            if penalties_are_small: 
                lower_bound = max(lower_bound, current_potential_lb_master)
                print(f"Master problem penalties are small. Commitment Cost + Beta = {current_potential_lb_master:.4f}")
            else:
                print(f"Master problem penalties are NOT sufficiently small ({logic1_penalty_val_master:.2f}, {logic2_penalty_val_master:.2f}, {on_penalty_val_master:.2f}, {benders_penalty_total_val_master:.2f}). Using QUBO obj for LB consideration.")
                lower_bound = max(lower_bound, master_obj_val_from_solver)

            print(f"Updated Lower Bound (Z_LB): {lower_bound:.4f}")

            best_master_solution_for_lb['u'] = u_sol_master
            best_master_solution_for_lb['zON'] = zON_sol_master
            best_master_solution_for_lb['zOFF'] = zOFF_sol_master
            best_master_solution_for_lb['beta_binary'] = beta_binary_sol_master 
            best_master_solution_for_lb['s_cuts'] = s_cuts_sol_master
            best_master_solution_for_lb['s_on'] = s_on_sol_master

        else: 
            print(f"Master Problem FAILED! Status: {last_master_results.solver.termination_condition}")
            if last_master_results.solver.message:
                 print(f"Solver message: {last_master_results.solver.message}")
            print("Terminating Benders loop due to master error.")
            break 

    # --- End of Benders Loop ---
    # Final print statements (remain same)
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    if upper_bound != float('inf') and lower_bound != -float('inf'):
        final_gap = (upper_bound - lower_bound)
        print(f"Final Gap: {final_gap:.6f}")
    else:
        print("Final Gap: Inf")
    
    print(f"Iterations Performed: {len(iteration_data_main_loop)}") 
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    if best_solution_for_ub: 
        print("\n--- Best Solution Found (leading to best Z_UB in iter {}) ---".format(best_solution_for_ub['iter']))
        print(f"Best Total Cost (Upper Bound): {best_solution_for_ub['total_cost']:.4f}")
        print("Commitment Schedule (u_it):")
        for t_print_u in time_periods: 
            print(f"  t={t_print_u}: ", {i_print_u: round(best_solution_for_ub['u_vals'].get((i_print_u,t_print_u),0)) for i_print_u in generators})

        print("\nFinal Dispatch (p_it) for Best UB Solution:")
        final_subproblem_print = build_subproblem(best_solution_for_ub['u_vals'], best_solution_for_ub['zON_vals'], best_solution_for_ub['zOFF_vals'])
        if not hasattr(final_subproblem_print, 'dual') or not isinstance(final_subproblem_print.dual, pyo.Suffix):
            final_subproblem_print.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        elif final_subproblem_print.dual.direction != pyo.Suffix.IMPORT:
            final_subproblem_print.dual.set_direction(pyo.Suffix.IMPORT)

        sub_solver.solve(final_subproblem_print, tee=False)
        if pyo.value(final_subproblem_print.OBJ, exception_flag=False) is not None:
            for t_print_p in time_periods: 
                print(f"  t={t_print_p}: ", {i_print_p: f"{pyo.value(final_subproblem_print.p[i_print_p,t_print_p]):.2f}" for i_print_p in generators})
            final_sub_obj_val_print = pyo.value(final_subproblem_print.OBJ)
            print(f"Final Subproblem Objective (Var Cost + Penalty): {final_sub_obj_val_print:.4f}")
            final_commit_cost_print = sum(gen_data[i_p_c]['Csu'] * best_solution_for_ub['zON_vals'].get((i_p_c, t_p_c),0) +
                                     gen_data[i_p_c]['Csd'] * best_solution_for_ub['zOFF_vals'].get((i_p_c, t_p_c),0) +
                                     gen_data[i_p_c]['Cf'] * best_solution_for_ub['u_vals'].get((i_p_c, t_p_c),0)
                                     for i_p_c in generators for t_p_c in time_periods)
            print(f"Final Commitment Cost: {final_commit_cost_print:.2f}")
            print(f"Final Total Cost (recalculated for best UB): {final_commit_cost_print + final_sub_obj_val_print:.4f}")
            print("Final Demand Slack:")
            for t_print_s in time_periods: 
                print(f"  t={t_print_s}: {pyo.value(final_subproblem_print.demand_slack[t_print_s]):.4f}")
        else:
            print("Could not resolve final subproblem to show dispatch for best UB.")
    else:
        print("\nNo feasible solution found or Benders loop terminated before a UB was established.")


if __name__ == '__main__':
    main()
