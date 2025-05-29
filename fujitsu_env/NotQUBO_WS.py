##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo.
##### Logic constraints (types 1 and 2) are penalty terms.
##### Benders optimality AND feasibility cuts are ALSO penalty terms.
##### Feasibility cuts use FarkasDual attributes obtained via gurobi_persistent.

##### basic imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import math

# Choose the number of time periods wanted:
Periods = 4

# Sets
generators = [1, 2, 3]
time_periods = [x for x in range(1, Periods+1)] # T=3 hours
time_periods_with_0 = [x for x in range(0, Periods+1)] # Including t=0 for initial conditions

# Generator Parameters
gen_data = {
    1: {'Pmin': 50,  'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
    2: {'Pmin': 80,  'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
    3: {'Pmin': 40,  'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5,  'Csd': 1.0, 'Cv': 0.150}
}

# Demand Parameters
demand = {1: 160, 2: 500, 3: 400, 4:400} 

# Initial Conditions for T = 0
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Number of bits for beta variable
num_beta_bits = 13 # Or more, depending on expected range of subproblem costs

# Function creating the sub problem (LP)
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    model = pyo.ConcreteModel(name="UCP_Subproblem")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)

    u_fixed_param_vals = {(i,t): u_fixed_vals[i,t] for i in model.I for t in model.T}
    zON_fixed_param_vals = {(i,t): zON_fixed_vals[i,t] for i in model.I for t in model.T}
    zOFF_fixed_param_vals = {(i,t): zOFF_fixed_vals[i,t] for i in model.I for t in model.T}

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

    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)

    def objective_rule(m):
        return sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def p_prev_rule(m, i, t):
        return m.p_init[i] if t == 1 else m.p[i, t-1]
    model.p_prev = pyo.Expression(model.I, model.T, rule=p_prev_rule)

    def u_prev_fixed_rule(m, i, t):
        return m.u_init[i] if t == 1 else m.u_fixed[i, t-1]
    model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=u_prev_fixed_rule)

    # Subproblem Constraints [cite: 31]
    model.MinPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t])
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t])
    model.RampUp = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p[i,t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t])
    model.RampDown = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p_prev[i,t] - m.p[i,t] <= m.Rd[i] * m.u_fixed[i,t] + m.Rsd[i] * m.zOFF_fixed[i,t])
    model.Demand = pyo.Constraint(model.T, rule=lambda m, t: sum(m.p[i, t] for i in m.I) >= m.D[t])

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model

# Function creating the master problem
def build_master(iteration_data, latest_u_warm=None, latest_zON_warm=None, latest_zOFF_warm=None, latest_beta_bin_warm=None, latest_s_opt_warm=None, latest_s_feas_warm=None):
    model = pyo.ConcreteModel(name="UCP_MasterProblem_PenalizedCuts")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)

    # Parameters for costs and penalties
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I}) # Renamed from Pmax_param if it was
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.D_param = pyo.Param(model.T, initialize=demand) # Renamed from D
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    
    model.lambda_logic1 = pyo.Param(initialize=50) # Penalty for logic constraint 1
    model.lambda_logic2 = pyo.Param(initialize=1)  # Penalty for logic constraint 2
    model.lambda_opt_cut = pyo.Param(initialize=8) # Penalty for optimality cuts
    model.lambda_feas_cut = pyo.Param(initialize=80) # Penalty for feasibility cuts (typically needs to be high)

    # Parameters for Benders cut coefficients (used in objective terms)
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I}) # Renamed from Pmin_param
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I}) # Renamed from Rd_param
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I}) # Renamed from Rsd_param
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I}) # Renamed from Ru_param
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I}) # Renamed from Rsu_param

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary)

    # Create sets of indices for optimality and feasibility cuts based on iteration_data
    opt_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'optimality']
    feas_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'feasibility']
    
    model.OptCutIndices = pyo.Set(initialize=opt_cut_indices_list)
    model.FeasCutIndices = pyo.Set(initialize=feas_cut_indices_list)

    model.s_opt_cuts = pyo.Var(model.OptCutIndices, within=pyo.Reals, bounds=(0, 1e9)) # Slack for optimality cuts [cite: 18]
    model.s_feas_cuts = pyo.Var(model.FeasCutIndices, within=pyo.Reals, bounds=(0, 1e9))# Slack for feasibility cuts [cite: 33]

    # Warmstart if values provided
    if latest_u_warm:
        for i in model.I:
            for t in model.T:
                model.u[i,t].value = latest_u_warm.get((i,t), 0)
                model.zON[i,t].value = latest_zON_warm.get((i,t), 0)
                model.zOFF[i,t].value = latest_zOFF_warm.get((i,t), 0)
    if latest_beta_bin_warm:
        for b_idx in model.BETA_BITS:
            model.beta_binary[b_idx].value = latest_beta_bin_warm.get(b_idx, 0)
    if latest_s_opt_warm:
        for k_idx in model.OptCutIndices:
            model.s_opt_cuts[k_idx].value = latest_s_opt_warm.get(k_idx, 0)
    if latest_s_feas_warm:
        for k_idx in model.FeasCutIndices:
            model.s_feas_cuts[k_idx].value = latest_s_feas_warm.get(k_idx, 0)


    # Expression for u_prev
    def u_prev_rule(m, i, t):
        return m.u_init[i] if t == 1 else m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    # Objective function
    def master_objective_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        
        logic1_penalty_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_penalty_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )

        # Optimality Cuts Penalty [cite: 17]
        optimality_cuts_penalty = 0
        for k_idx in m.OptCutIndices:
            data = iteration_data[k_idx]
            sub_obj_k = data['sub_obj']
            duals_k = data['duals']
            u_k_iter = data['u_vals']    
            zON_k_iter = data['zON_vals']
            zOFF_k_iter = data['zOFF_vals']
            
            # Calculate cut_rhs_expr = θ_k + Φ_k^T (x - x_k)
            cut_rhs_expr = sub_obj_k 
            for i in m.I:
                for t_loop in m.T: # Use t_loop to avoid conflict with m.T if used inside sum
                    # MinPower: dual * Pmin * (u_current - u_k_iter)
                    cut_rhs_expr += duals_k['lambda_min'].get((i, t_loop), 0.0) * \
                                    m.Pmin[i] * (m.u[i, t_loop] - u_k_iter.get((i,t_loop), 0.0))

                    cut_rhs_expr += duals_k['lambda_max'].get((i, t_loop), 0.0) * \
                                    m.Pmax[i] * (m.u[i, t_loop] - u_k_iter.get((i,t_loop), 0.0)) # This implies lambda_max is for Pmax*u fixed term

                    # RampUp: p_it - p_prev <= Ru*u_prev_fixed + Rsu*zON_fixed
                    # Term: dual_ru * ( Ru*(u_prev_current - u_prev_k) + Rsu*(zON_current - zON_k) )
                    dual_val_ru = duals_k['lambda_ru'].get((i, t_loop), 0.0) # Dual for <=, so non-positive
                    u_prev_term_expr_ru = 0
                    u_prev_k_val_ru = u_k_iter.get((i, t_loop-1), m.u_init[i]) if t_loop > 1 else m.u_init[i]
                    if t_loop > 1:
                         u_prev_term_expr_ru = m.Ru[i] * (m.u[i, t_loop-1] - u_prev_k_val_ru)
                    
                    zON_term_expr_ru = m.Rsu[i] * (m.zON[i, t_loop] - zON_k_iter.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val_ru * (u_prev_term_expr_ru + zON_term_expr_ru)
                    
                    # RampDown: p_prev - p_it <= Rd*u_fixed + Rsd*zOFF_fixed
                    # Term: dual_rd * ( Rd*(u_current - u_k) + Rsd*(zOFF_current - zOFF_k) )
                    dual_val_rd = duals_k['lambda_rd'].get((i, t_loop), 0.0) # Dual for <=, so non-positive
                    u_term_expr_rd = m.Rd[i] * (m.u[i, t_loop] - u_k_iter.get((i, t_loop), 0.0))
                    zOFF_term_expr_rd = m.Rsd[i] * (m.zOFF[i, t_loop] - zOFF_k_iter.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val_rd * (u_term_expr_rd + zOFF_term_expr_rd)

            optimality_cuts_penalty += (binary_beta_expr - cut_rhs_expr - m.s_opt_cuts[k_idx])**2
        
        optimality_cuts_penalty *= m.lambda_opt_cut

        # Feasibility Cuts Penalty [cite: 32]
        feasibility_cuts_penalty = 0
        for k_idx in m.FeasCutIndices:
            data = iteration_data[k_idx]
            rays_k = data['rays']
            
            current_feas_cut_lhs_expr = 0
            
            for i_gen in m.I:
                for t_period in m.T:
                    current_feas_cut_lhs_expr += rays_k['min_power'].get((i_gen, t_period), 0.0) * (m.Pmin[i_gen] * m.u[i_gen, t_period])
                    current_feas_cut_lhs_expr += rays_k['max_power'].get((i_gen, t_period), 0.0) * (m.Pmax[i_gen] * m.u[i_gen, t_period]) # As per user's working code
                    
                    # Ramp up: term in L(x) based on ray_val * (Ru*u_prev + Rsu*zON)
                    current_feas_cut_lhs_expr += rays_k['ramp_up'].get((i_gen, t_period), 0.0) * \
                                                 (m.Ru[i_gen] * m.u_prev[i_gen, t_period] + m.Rsu[i_gen] * m.zON[i_gen, t_period])
                    # Ramp down: term in L(x) based on ray_val * (Rd*u + Rsd*zOFF)
                    current_feas_cut_lhs_expr += rays_k['ramp_down'].get((i_gen, t_period), 0.0) * \
                                                 (m.Rd[i_gen] * m.u[i_gen, t_period] + m.Rsd[i_gen] * m.zOFF[i_gen, t_period])
            for t_period in m.T:
                 current_feas_cut_lhs_expr += rays_k['demand'].get(t_period, 0.0) * m.D_param[t_period] # D_param used here

            # Penalty: lambda * (-L_k(x) + s_k)^2
            feasibility_cuts_penalty += (-current_feas_cut_lhs_expr + m.s_feas_cuts[k_idx])**2
            
        feasibility_cuts_penalty *= m.lambda_feas_cut
        
        return commitment_cost + logic1_penalty_term + logic2_penalty_term + \
               binary_beta_expr + optimality_cuts_penalty + feasibility_cuts_penalty

    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)
    return model

# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 
    epsilon = 1 
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')

    # Initial master variable values for the first subproblem solve
    u_current = {}
    zON_current = {}
    zOFF_current = {}
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0 # Start all ON for robustness with penalty methods
            u_prev_val = u_initial[i] if t == 1 else u_current.get((i, t-1), u_initial[i]) # careful with get here if t-1 not populated
            if u_current[i, t] > 0.5 and u_prev_val < 0.5:
                zON_current[i, t] = 1.0; zOFF_current[i, t] = 0.0
            elif u_current[i, t] < 0.5 and u_prev_val > 0.5:
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 1.0
            else: # u_current[i,t] == u_prev_val
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 0.0
    
    # For warmstarting master
    latest_u_warm = u_current.copy()
    latest_zON_warm = zON_current.copy()
    latest_zOFF_warm = zOFF_current.copy()
    latest_beta_bin_warm = {b:0 for b in range(num_beta_bits)}
    latest_s_opt_warm = {}
    latest_s_feas_warm = {}


    master_solver_name = "gurobi" # Or any MINLP solver if penalties make it non-linear (e.g. "baron", "scip")
                                # Gurobi can handle quadratic objectives and binary vars.
    master_solver = SolverFactory(master_solver_name)
    
    sub_solver_name = "gurobi_persistent"
    sub_solver_is_persistent = False
    try:
        sub_solver = SolverFactory(sub_solver_name, solver_io='python') 
        sub_solver_is_persistent = True
        print(f"Successfully initialized persistent subproblem solver: {sub_solver.name}")
    except Exception as e:
        print(f"Could not create gurobi_persistent solver: {e}")
        print("Falling back to standard gurobi solver for subproblem.")
        sub_solver = SolverFactory("gurobi")

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Master Solver: {master_solver_name}, Subproblem Solver Name: {sub_solver.name}")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")

    k_iter_count = 0
    best_solution_for_ub_display = None

    for k_loop_idx in range(1, max_iter + 1):
        k_iter_count = k_loop_idx
        print(f"========================= Iteration {k_iter_count} =========================")

        print("--- Solving Subproblem ---")
        # Use u_current, zON_current, zOFF_current from previous master solve (or initial)
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)

        sub_solve_completed = False 
        is_infeasible = False
        sub_obj_val = float('nan') 

        if sub_solver_is_persistent:
            try:
                sub_solver.set_instance(subproblem)
                sub_solver.set_gurobi_param('InfUnbdInfo', 1) # To get Farkas duals
                sub_solver.set_gurobi_param('DualReductions', 0) # Important for Farkas duals
                results = sub_solver.solve(tee=False) 
                sub_solve_completed = True
                
                if results.solver.termination_condition == TerminationCondition.optimal or \
                   results.solver.termination_condition == TerminationCondition.feasible:
                    sub_obj_val = pyo.value(subproblem.OBJ) 
                    print(f"Subproblem Status (Persistent): {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
                elif results.solver.termination_condition == TerminationCondition.infeasible:
                    print("Subproblem Status (Persistent): INFEASIBLE")
                    is_infeasible = True
                else:
                    print(f"Subproblem FAILED (Persistent) with status: {results.solver.termination_condition}")
                    sub_solve_completed = False 
            except Exception as e:
                print(f"Error with persistent subproblem solver: {e}")
                sub_solve_completed = False
        else: # Standard solver
            sub_solver.options['InfUnbdInfo'] = 1 # Gurobi specific option
            sub_solver.options['DualReductions'] = 0 # Gurobi specific
            results = sub_solver.solve(subproblem, load_solutions=True, tee=False) 
            sub_solve_completed = True
            if results.solver.termination_condition == TerminationCondition.optimal or \
               results.solver.termination_condition == TerminationCondition.feasible:
                sub_obj_val = pyo.value(subproblem.OBJ)
                print(f"Subproblem Status (Standard): {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            elif results.solver.termination_condition == TerminationCondition.infeasible:
                print("Subproblem Status (Standard): INFEASIBLE")
                is_infeasible = True
                # For non-persistent Gurobi, need to re-solve with specific parameters to get Farkas duals if not available
                # This part is tricky with generic SolverFactory; gurobi_persistent handles it better.
                # If FarkasDuals are not available through suffix after an infeasible solve, feasibility cuts might be zero.
            else:
                print(f"Subproblem FAILED (Standard) with status: {results.solver.termination_condition}")
                sub_solve_completed = False

        if not sub_solve_completed: 
            print("Terminating Benders loop due to subproblem solver issue.")
            break

        if not is_infeasible: 
            commitment_cost_current_iter = sum(gen_data[i]['Csu'] * zON_current.get((i,t),0) +
                                               gen_data[i]['Csd'] * zOFF_current.get((i,t),0) +
                                               gen_data[i]['Cf'] * u_current.get((i,t),0)
                                               for i in generators for t in time_periods)
            current_total_cost = commitment_cost_current_iter + sub_obj_val
            
            # Check if u_current, zON_current, zOFF_current are logically feasible
            logically_sound_for_ub = True
            for i_gen in generators:
                for t_time in time_periods:
                    u_val = u_current.get((i_gen,t_time),0)
                    u_prev_val = u_initial[i_gen] if t_time == 1 else u_current.get((i_gen, t_time-1),0)
                    zon_val = zON_current.get((i_gen,t_time),0)
                    zoff_val = zOFF_current.get((i_gen,t_time),0)
                    if abs((u_val - u_prev_val) - (zon_val - zoff_val)) > 1e-4: # Logic 1
                        logically_sound_for_ub = False; break
                    if abs(zon_val * zoff_val) > 1e-4: # Logic 2
                        logically_sound_for_ub = False; break
                if not logically_sound_for_ub: break

            if logically_sound_for_ub and current_total_cost < upper_bound :
                upper_bound = current_total_cost
                best_solution_for_ub_display = {
                    'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 
                    'zOFF_vals': zOFF_current.copy(), 'iter': k_iter_count,
                    'total_cost': upper_bound
                }
                print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f} from total cost {current_total_cost:.4f} (logically sound master solution)")
            elif not logically_sound_for_ub:
                 print(f"Current master solution for UB calc is not logically sound. UB not updated. Cost was {current_total_cost:.4f}")
            else:
                print(f"Current Total Cost {current_total_cost:.4f} did not improve UB {upper_bound:.4f}")

            duals_for_cut = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}, 'lambda_dem': {}}
            try: 
                for i in generators:
                    for t_p in time_periods:
                        duals_for_cut['lambda_min'][(i,t_p)] = subproblem.dual.get(subproblem.MinPower[i,t_p], 0.0)
                        duals_for_cut['lambda_max'][(i,t_p)] = subproblem.dual.get(subproblem.MaxPower[i,t_p], 0.0)
                        duals_for_cut['lambda_ru'][(i,t_p)]  = subproblem.dual.get(subproblem.RampUp[i,t_p], 0.0)
                        duals_for_cut['lambda_rd'][(i,t_p)]  = subproblem.dual.get(subproblem.RampDown[i,t_p], 0.0)
                for t_p in time_periods:
                    duals_for_cut['lambda_dem'][t_p] = subproblem.dual.get(subproblem.Demand[t_p], 0.0)
            except Exception as e:
                print(f"Warning: Error extracting duals (suffix) for optimality cut: {e}. Using 0.0.")
            
            iteration_data.append({
                'type': 'optimality', 'iter': k_iter_count, 'sub_obj': sub_obj_val,
                'duals': duals_for_cut, 'u_vals': u_current.copy(), # u_vals are x_k for the cut
                'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
            })

        else: # Subproblem is Infeasible
            print("Generating Feasibility Cut using FarkasDual.")
            rays_for_cut = {'min_power': {}, 'max_power': {}, 'ramp_up': {}, 'ramp_down': {}, 'demand': {}}
            can_add_feas_cut = False
            
            if sub_solver_is_persistent and hasattr(sub_solver, 'get_linear_constraint_attr'):
                try:
                    non_zero_ray_found = False
                    for c in subproblem.component_data_objects(Constraint, active=True):
                        ray_val = sub_solver.get_linear_constraint_attr(c, 'FarkasDual')
                        if ray_val is None: ray_val = 0.0
                        if abs(ray_val) > 1e-9: non_zero_ray_found = True
                        
                        parent_component = c.parent_component()
                        idx = c.index()
                        if parent_component is subproblem.MinPower: rays_for_cut['min_power'][idx] = ray_val
                        elif parent_component is subproblem.MaxPower: rays_for_cut['max_power'][idx] = ray_val
                        elif parent_component is subproblem.RampUp: rays_for_cut['ramp_up'][idx] = ray_val
                        elif parent_component is subproblem.RampDown: rays_for_cut['ramp_down'][idx] = ray_val
                        elif parent_component is subproblem.Demand: rays_for_cut['demand'][idx] = ray_val
                    
                    if not non_zero_ray_found: print("WARNING: All extracted FarkasDuals are zero/None (Persistent). Feasibility cut may be trivial.")
                    can_add_feas_cut = True
                    # print("Extracted Rays (Persistent):", rays_for_cut) 
                except Exception as e:
                    print(f"ERROR: Failed to extract FarkasDuals with persistent solver: {e}")
            else: # Standard solver, try suffix (less reliable for Farkas)
                print("WARNING: Subproblem solver is not gurobi_persistent or get_linear_constraint_attr not available. Attempting ray extraction via 'dual' suffix (may not be Farkas rays).")
                if hasattr(subproblem, 'dual'):
                    non_zero_ray_found_std = False
                    for i_gen in generators:
                        for t_p in time_periods:
                            val_mp = subproblem.dual.get(subproblem.MinPower[i_gen,t_p], 0.0); rays_for_cut['min_power'][(i_gen,t_p)] = val_mp
                            val_maxp = subproblem.dual.get(subproblem.MaxPower[i_gen,t_p], 0.0); rays_for_cut['max_power'][(i_gen,t_p)] = val_maxp
                            val_ru = subproblem.dual.get(subproblem.RampUp[i_gen,t_p], 0.0); rays_for_cut['ramp_up'][(i_gen,t_p)] = val_ru
                            val_rd = subproblem.dual.get(subproblem.RampDown[i_gen,t_p], 0.0); rays_for_cut['ramp_down'][(i_gen,t_p)] = val_rd
                            if any(abs(v) > 1e-9 for v in [val_mp, val_maxp, val_ru, val_rd]): non_zero_ray_found_std = True
                    for t_p in time_periods:
                        val_dem = subproblem.dual.get(subproblem.Demand[t_p], 0.0); rays_for_cut['demand'][t_p] = val_dem
                        if abs(val_dem) > 1e-9: non_zero_ray_found_std = True
                    
                    if not non_zero_ray_found_std: print("WARNING: All rays from 'dual' suffix are zero. Feasibility cut will be trivial.")
                    can_add_feas_cut = True
                else:
                    print("ERROR: No 'dual' suffix on subproblem for standard solver ray extraction.")

            if can_add_feas_cut:
                iteration_data.append({
                    'type': 'feasibility', 'iter': k_iter_count, 'rays': rays_for_cut,
                    'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
                })
            else:
                print("Skipping feasibility cut addition due to issues in ray extraction.")

        # --- Convergence Check & Master Problem ---
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
            gap = (upper_bound - lower_bound) 
            print(f"Current Gap: {gap:.6f} (Tolerance: {epsilon})")
            if gap <= epsilon and k_iter_count > 1: # Avoid premature convergence if UB not well established
                print("\nConvergence tolerance met.")
                break
        else:
            print("Gap cannot be calculated yet.")

        if k_iter_count == max_iter:
            print("\nMaximum iterations reached.")
            break

        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data, latest_u_warm, latest_zON_warm, latest_zOFF_warm, latest_beta_bin_warm, latest_s_opt_warm, latest_s_feas_warm)
        
        master_solver.options['NumericFocus'] = 1 # Adjust as needed for Gurobi
        #master_solver.options['NonConvex'] = 2 # If using Gurobi for MIQCP
        #master_solver.options['Presolve'] = 0
        master_results = master_solver.solve(master_problem, tee=False) # Set tee=True for solver logs

        if master_results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]:
            master_obj_val_total_qubo = pyo.value(master_problem.OBJ)
            
            # Extract values for LB calculation and warmstart
            u_current = {(i,t): pyo.value(master_problem.u[i,t]) for i in generators for t in time_periods}
            zON_current = {(i,t): pyo.value(master_problem.zON[i,t]) for i in generators for t in time_periods}
            zOFF_current = {(i,t): pyo.value(master_problem.zOFF[i,t]) for i in generators for t in time_periods}
            beta_binary_current = {j: pyo.value(master_problem.beta_binary[j]) for j in master_problem.BETA_BITS}
            s_opt_cuts_current = {k: pyo.value(master_problem.s_opt_cuts[k]) for k in master_problem.OptCutIndices if master_problem.s_opt_cuts[k].value is not None}
            s_feas_cuts_current = {k: pyo.value(master_problem.s_feas_cuts[k]) for k in master_problem.FeasCutIndices if master_problem.s_feas_cuts[k].value is not None}

            #printing the slacks for debug
            print("Optimality Cut Slacks:", s_opt_cuts_current)
            print("Feasibility Cut Slacks:", s_feas_cuts_current)

            # Update warmstart values
            latest_u_warm, latest_zON_warm, latest_zOFF_warm = u_current, zON_current, zOFF_current
            latest_beta_bin_warm, latest_s_opt_warm, latest_s_feas_warm = beta_binary_current, s_opt_cuts_current, s_feas_cuts_current

            commitment_cost_master_sol = sum(
                pyo.value(master_problem.Cf[i] * master_problem.u[i, t]) +
                pyo.value(master_problem.Csu[i] * master_problem.zON[i, t]) +
                pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t])
                for i in master_problem.I for t in master_problem.T)
            beta_val_master_sol = sum((2**j) * pyo.value(master_problem.beta_binary[j]) for j in master_problem.BETA_BITS)
            
            # Calculate penalty contributions from master solution
            logic1_pen_val = pyo.value(master_problem.lambda_logic1 * sum( ( (master_problem.u[i, t] - master_problem.u_prev[i, t]) - (master_problem.zON[i, t] - master_problem.zOFF[i, t]) )**2 for i in master_problem.I for t in master_problem.T ))
            logic2_pen_val = pyo.value(master_problem.lambda_logic2 * sum( master_problem.zON[i, t] * master_problem.zOFF[i, t] for i in master_problem.I for t in master_problem.T ))
            
            opt_pen_total_val = 0
            if hasattr(master_problem, 'OptCutIndices') and len(master_problem.OptCutIndices) > 0 :
                # Reconstruct penalty to report (master_objective_rule is not directly callable with solved values easily)
                temp_opt_penalty = 0
                for k_idx in master_problem.OptCutIndices:
                    data = iteration_data[k_idx]
                    sub_obj_k = data['sub_obj']; duals_k = data['duals']
                    u_k_iter = data['u_vals']; zON_k_iter = data['zON_vals']; zOFF_k_iter = data['zOFF_vals']
                    cut_rhs_expr_val = sub_obj_k
                    for i in master_problem.I:
                        for t_loop in master_problem.T:
                            cut_rhs_expr_val += duals_k['lambda_min'].get((i,t_loop),0.) * master_problem.Pmin[i] * (u_current[i,t_loop] - u_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += duals_k['lambda_max'].get((i,t_loop),0.) * master_problem.Pmax[i] * (u_current[i,t_loop] - u_k_iter.get((i,t_loop),0.))
                            dual_val_ru = duals_k['lambda_ru'].get((i,t_loop),0.)
                            u_prev_term_expr_ru_val = 0
                            if t_loop > 1: u_prev_term_expr_ru_val = master_problem.Ru[i] * (u_current[i,t_loop-1] - (u_k_iter.get((i,t_loop-1), master_problem.u_init[i]) if t_loop > 1 else master_problem.u_init[i]))
                            zON_term_expr_ru_val = master_problem.Rsu[i] * (zON_current[i,t_loop] - zON_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += dual_val_ru * (u_prev_term_expr_ru_val + zON_term_expr_ru_val)
                            dual_val_rd = duals_k['lambda_rd'].get((i,t_loop),0.)
                            u_term_expr_rd_val = master_problem.Rd[i] * (u_current[i,t_loop] - u_k_iter.get((i,t_loop),0.))
                            zOFF_term_expr_rd_val = master_problem.Rsd[i] * (zOFF_current[i,t_loop] - zOFF_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += dual_val_rd * (u_term_expr_rd_val + zOFF_term_expr_rd_val)
                    temp_opt_penalty += (beta_val_master_sol - cut_rhs_expr_val - s_opt_cuts_current.get(k_idx,0))**2
                opt_pen_total_val = pyo.value(master_problem.lambda_opt_cut) * temp_opt_penalty

            feas_pen_total_val = 0
            if hasattr(master_problem, 'FeasCutIndices') and len(master_problem.FeasCutIndices) > 0:
                temp_feas_penalty = 0
                for k_idx in master_problem.FeasCutIndices:
                    data = iteration_data[k_idx]; rays_k = data['rays']
                    current_feas_cut_lhs_expr_val = 0
                    for i_gen in master_problem.I:
                        for t_period in master_problem.T:
                            current_feas_cut_lhs_expr_val += rays_k['min_power'].get((i_gen,t_period),0.)*(master_problem.Pmin[i_gen]*u_current[i_gen,t_period])
                            current_feas_cut_lhs_expr_val += rays_k['max_power'].get((i_gen,t_period),0.)*(master_problem.Pmax[i_gen]*u_current[i_gen,t_period])
                            current_feas_cut_lhs_expr_val += rays_k['ramp_up'].get((i_gen,t_period),0.)*(master_problem.Ru[i_gen]*(u_initial[i_gen] if t_period==1 else u_current[i_gen,t_period-1]) + master_problem.Rsu[i_gen]*zON_current[i_gen,t_period])
                            current_feas_cut_lhs_expr_val += rays_k['ramp_down'].get((i_gen,t_period),0.)*(master_problem.Rd[i_gen]*u_current[i_gen,t_period] + master_problem.Rsd[i_gen]*zOFF_current[i_gen,t_period])
                    for t_period in master_problem.T:
                        current_feas_cut_lhs_expr_val += rays_k['demand'].get(t_period,0.)*master_problem.D_param[t_period]
                    temp_feas_penalty += (-current_feas_cut_lhs_expr_val + s_feas_cuts_current.get(k_idx,0))**2
                feas_pen_total_val = pyo.value(master_problem.lambda_feas_cut) * temp_feas_penalty

            print(f"Master Status: {master_results.solver.termination_condition}")
            print(f"Master QUBO Objective Value: {master_obj_val_total_qubo:.4f}")
            print(f"  Commitment Cost part: {commitment_cost_master_sol:.4f}")
            print(f"  Beta Value part: {beta_val_master_sol:.4f}")
            print(f"  Logic1 Penalty part: {logic1_pen_val:.4f}")
            print(f"  Logic2 Penalty part: {logic2_pen_val:.4f}")
            print(f"  Optimality Cuts Penalty part: {opt_pen_total_val:.4f}")
            print(f"  Feasibility Cuts Penalty part: {feas_pen_total_val:.4f}")

            pen_tol = 1e-3 # Tolerance for penalties being "small"
            if logic1_pen_val < pen_tol and logic2_pen_val < pen_tol and \
               opt_pen_total_val < pen_tol and feas_pen_total_val < pen_tol:
                true_lower_bound_candidate = commitment_cost_master_sol + beta_val_master_sol
                lower_bound = max(lower_bound, true_lower_bound_candidate)
                print(f"All penalties small. Updated True Lower Bound (Z_LB): {lower_bound:.4f}")
            else:
                # If penalties are large, commitment_cost + beta is not a reliable LB for original problem.
                # The QUBO objective itself is a LB for *this penalized master problem*.
                # For simplicity, we might not update LB or use a more conservative approach if penalties are high.
                # Or, as in user's previous code, use the QUBO obj if penalties are high.
                # For now, only update LB if penalties are small to ensure validity for original problem.
                print(f"Penalties are not all small. Lower bound {lower_bound:.4f} not updated by Cmnt+Beta.")
                # Alternative: lower_bound = max(lower_bound, master_obj_val_total_qubo) but this is for the QUBO.

        elif master_results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Master Problem INFEASIBLE. Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop.")
            break
        else:
            print(f"Master Problem FAILED to solve optimally/feasibly! Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop.")
            break
            
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) if upper_bound != float('inf') and lower_bound != -float('inf') else float('inf')
    print(f"Final Absolute Gap: {final_gap:.6f}")
    print(f"Iterations Performed: {k_iter_count}")
    used_time = end_time - start_time
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    if best_solution_for_ub_display:
        print("\n--- Best Feasible Solution Found (leading to best Z_UB in iter {}) ---".format(best_solution_for_ub_display['iter']))
        print(f"Best Total Cost (Upper Bound): {best_solution_for_ub_display['total_cost']:.4f}")
        print("Commitment Schedule (u_it):")
        u_best = best_solution_for_ub_display['u_vals']
        for t_p in time_periods: print(f"  t={t_p}: ", {i: round(u_best.get((i,t_p),0)) for i in generators})

        print("\nFinal Dispatch (p_it) for the best UB solution:")
        final_subproblem = build_subproblem(best_solution_for_ub_display['u_vals'], 
                                            best_solution_for_ub_display['zON_vals'], 
                                            best_solution_for_ub_display['zOFF_vals'])
        
        final_sub_solver = SolverFactory('gurobi') 
        final_sub_results = final_sub_solver.solve(final_subproblem, tee=False)

        if final_sub_results.solver.termination_condition == TerminationCondition.optimal:
            final_sub_obj_resolved = pyo.value(final_subproblem.OBJ)
            final_commit_c_resolved = sum(gen_data[i]['Csu'] * best_solution_for_ub_display['zON_vals'].get((i,t),0) + 
                                          gen_data[i]['Csd'] * best_solution_for_ub_display['zOFF_vals'].get((i,t),0) + 
                                          gen_data[i]['Cf'] * best_solution_for_ub_display['u_vals'].get((i,t),0) 
                                          for i in generators for t in time_periods)
            print(f"  Final Variable Cost (re-solve): {final_sub_obj_resolved:.4f}")
            print(f"  Final Commitment Cost: {final_commit_c_resolved:.2f}")
            print(f"  Final Total Cost (recalculated): {final_commit_c_resolved + final_sub_obj_resolved:.4f}")
            for t_p in time_periods: print(f"    t={t_p}: ", {i: f"{pyo.value(final_subproblem.p[i,t_p]):.2f}" for i in generators})
            print("  Final Demand Check:")
            for t_p in time_periods:
                actual_prod = sum(pyo.value(final_subproblem.p[i,t_p]) for i in generators)
                print(f"    t={t_p}: Prod={actual_prod:.2f}, Demand={demand[t_p]}, Met={actual_prod >= demand[t_p] - 1e-4}")
        else: print(f"Could not re-solve final subproblem for display. Status: {final_sub_results.solver.termination_condition}")
    else: print("\nNo feasible solution matching UB found for final printout, or UB was not updated from initial inf.")

    return used_time

if __name__ == '__main__':
    '''
    avg_used_time = 0
    N=100
    for _ in range(N):
        used_time = main()
        avg_used_time+=used_time/N
    print("average used time = ", avg_used_time)
    '''
    main()