##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo, classically
##### THe logic constraints, both types, are penalty terms
##### Benders optimality and feasibility cuts are constraints.
##### Feasibility cuts use FarkasDual attributes obtained via gurobi_persistent.

##### basic imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import math

# Choose the number of time periods wanted:
Periods = 1

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

# Demand and Reserve Parameters
demand = {1: 160} # Demand for each time period


# Initial Conditions for T = 0
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Number of bits for beta variable
num_beta_bits = 5

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

    model.MinPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t])
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t])
    model.RampUp = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p[i,t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t])
    model.RampDown = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p_prev[i,t] - m.p[i,t] <= m.Rd[i] * m.u_fixed[i,t] + m.Rsd[i] * m.zOFF_fixed[i,t])
    model.Demand = pyo.Constraint(model.T, rule=lambda m, t: sum(m.p[i, t] for i in m.I) >= m.D[t])

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model

# Function creating the master problem
def build_master(iteration_data):
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Constrained")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)

    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.D = pyo.Param(model.T, initialize=demand)
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.lambda_logic1 = pyo.Param(initialize=20) 
    model.lambda_logic2 = pyo.Param(initialize=1)  

    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})

    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary)

    def u_prev_rule(m, i, t):
        return m.u_init[i] if t == 1 else m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    def master_objective_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )
        return commitment_cost + logic1_term + logic2_term + binary_beta_expr
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

    model.BendersCuts = pyo.ConstraintList()
    binary_beta_expr_master = sum( (2**j) * model.beta_binary[j] for j in model.BETA_BITS )

    for k_iter_idx, data in enumerate(iteration_data):
        if data['type'] == 'optimality':
            sub_obj_k = data['sub_obj']
            duals_k = data['duals']
            u_k = data['u_vals']
            zON_k = data['zON_vals']
            zOFF_k = data['zOFF_vals']
            cut_expr_rhs = sub_obj_k
            for i in model.I:
                for t in model.T:
                    cut_expr_rhs += duals_k['lambda_min'].get((i, t), 0.0) * model.Pmin[i] * (model.u[i, t] - u_k.get((i,t), 0.0))
                    cut_expr_rhs += duals_k['lambda_max'].get((i,t),0.0) * model.Pmax[i] * (model.u[i,t] - u_k.get((i,t),0.0))
                    
                    dual_val = duals_k['lambda_ru'].get((i, t), 0.0) 
                    u_prev_term_model = model.Ru[i] * (model.u_prev[i,t] - (u_k.get((i, t-1), model.u_init[i]) if t > 1 else model.u_init[i]))
                    zON_term_model = model.Rsu[i] * (model.zON[i,t] - zON_k.get((i,t),0.0))
                    cut_expr_rhs += dual_val * (u_prev_term_model + zON_term_model)

                    dual_val = duals_k['lambda_rd'].get((i, t), 0.0) 
                    u_term_model = model.Rd[i] * (model.u[i,t] - u_k.get((i,t),0.0))
                    zOFF_term_model = model.Rsd[i] * (model.zOFF[i,t] - zOFF_k.get((i,t),0.0))
                    cut_expr_rhs += dual_val * (u_term_model + zOFF_term_model)
            model.BendersCuts.add(binary_beta_expr_master >= cut_expr_rhs)

        elif data['type'] == 'feasibility':
            rays_k = data['rays']
            feas_cut_lhs = 0
            for i_gen in model.I:
                for t_period in model.T:
                    ray_val = rays_k['min_power'].get((i_gen, t_period), 0.0)
                    feas_cut_lhs += ray_val * (model.Pmin[i_gen] * model.u[i_gen, t_period])
                    ray_val = rays_k['max_power'].get((i_gen, t_period), 0.0)
                    feas_cut_lhs += ray_val * (model.Pmax[i_gen] * model.u[i_gen, t_period])
                    ray_val = rays_k['ramp_up'].get((i_gen, t_period), 0.0)
                    feas_cut_lhs += ray_val * (model.Ru[i_gen] * model.u_prev[i_gen, t_period] + model.Rsu[i_gen] * model.zON[i_gen, t_period])
                    ray_val = rays_k['ramp_down'].get((i_gen, t_period), 0.0)
                    feas_cut_lhs += ray_val * (model.Rd[i_gen] * model.u[i_gen, t_period] + model.Rsd[i_gen] * model.zOFF[i_gen, t_period])
            for t_period in model.T:
                ray_val = rays_k['demand'].get(t_period, 0.0)
                feas_cut_lhs += ray_val * model.D[t_period]
            model.BendersCuts.add(feas_cut_lhs >= 0)
    return model

# Main Benders Loop
def main():
    start_time = time.time()
    max_iter = 30 
    epsilon = 1 
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')

    u_current = {}
    zON_current = {}
    zOFF_current = {}
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0 
            u_prev = u_initial[i] if t == 1 else u_current.get((i, t-1), u_initial[i])
            if u_current[i, t] > 0.5 and u_prev < 0.5:
                zON_current[i, t] = 1.0; zOFF_current[i, t] = 0.0
            elif u_current[i, t] < 0.5 and u_prev > 0.5:
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 1.0
            else:
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 0.0

    master_solver_name = "gurobi"
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
    for k_loop_idx in range(1, max_iter + 1):
        k_iter_count = k_loop_idx
        print(f"========================= Iteration {k_iter_count} =========================")

        print("--- Solving Subproblem ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)

        sub_solve_completed = False 
        is_infeasible = False
        sub_obj_val = float('nan') 

        if sub_solver_is_persistent:
            try:
                sub_solver.set_instance(subproblem)
                sub_solver.set_gurobi_param('InfUnbdInfo', 1)
                sub_solver.set_gurobi_param('DualReductions', 0) 
                results = sub_solver.solve(tee=False) # load_solutions=True is default
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
        else: 
            sub_solver.options['InfUnbdInfo'] = 1 
            results = sub_solver.solve(subproblem, load_solutions=True, tee=False) 
            sub_solve_completed = True
            if results.solver.termination_condition == TerminationCondition.optimal or \
               results.solver.termination_condition == TerminationCondition.feasible:
                sub_obj_val = pyo.value(subproblem.OBJ)
                print(f"Subproblem Status (Standard): {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            elif results.solver.termination_condition == TerminationCondition.infeasible:
                print("Subproblem Status (Standard): INFEASIBLE")
                is_infeasible = True
            else:
                print(f"Subproblem FAILED (Standard) with status: {results.solver.termination_condition}")
                sub_solve_completed = False

        if not sub_solve_completed: 
            print("Terminating Benders loop due to subproblem solver issue.")
            break

        if not is_infeasible: 
            commitment_cost_current = sum(gen_data[i]['Csu'] * zON_current[i, t] +
                                       gen_data[i]['Csd'] * zOFF_current[i, t] +
                                       gen_data[i]['Cf'] * u_current[i, t]
                                       for i in generators for t in time_periods)
            current_total_cost = commitment_cost_current + sub_obj_val
            if current_total_cost < upper_bound :
                 upper_bound = current_total_cost
                 print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f} from total cost {current_total_cost:.4f}")
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
                'duals': duals_for_cut, 'u_vals': u_current.copy(),
                'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
            })

        else: # Subproblem is Infeasible
            print("Generating Feasibility Cut using FarkasDual.")
            rays_for_cut = {'min_power': {}, 'max_power': {}, 'ramp_up': {}, 'ramp_down': {}, 'demand': {}}
            can_add_feas_cut = False
            if sub_solver_is_persistent:
                try:
                    temp_rays_min_power, temp_rays_max_power, temp_rays_ramp_up, temp_rays_ramp_down, temp_rays_demand = {}, {}, {}, {}, {}
                    non_zero_ray_found = False
                    for c in subproblem.component_data_objects(Constraint, active=True):
                        ray_val = sub_solver.get_linear_constraint_attr(c, 'FarkasDual')
                        if ray_val is None: # Handle case where attribute might not be set (e.g. Gurobi returns None)
                            ray_val = 0.0
                        
                        if abs(ray_val) > 1e-9: 
                            non_zero_ray_found = True

                        parent_component = c.parent_component()
                        idx = c.index()
                        if parent_component is subproblem.MinPower: temp_rays_min_power[idx] = ray_val
                        elif parent_component is subproblem.MaxPower: temp_rays_max_power[idx] = ray_val
                        elif parent_component is subproblem.RampUp: temp_rays_ramp_up[idx] = ray_val
                        elif parent_component is subproblem.RampDown: temp_rays_ramp_down[idx] = ray_val
                        elif parent_component is subproblem.Demand: temp_rays_demand[idx] = ray_val
                    
                    if not non_zero_ray_found:
                        print("WARNING: All extracted FarkasDuals are zero or None. Feasibility cut may be trivial.")
                    
                    rays_for_cut['min_power'] = temp_rays_min_power
                    rays_for_cut['max_power'] = temp_rays_max_power
                    rays_for_cut['ramp_up'] = temp_rays_ramp_up
                    rays_for_cut['ramp_down'] = temp_rays_ramp_down
                    rays_for_cut['demand'] = temp_rays_demand
                    can_add_feas_cut = True
                    print("Extracted Rays (Persistent):", rays_for_cut) # For debugging
                except AttributeError as ae: # Specifically catch if get_linear_constraint_attr itself is the issue for some reason
                    print(f"ERROR: AttributeError during FarkasDual extraction with persistent solver: {ae}")
                    print("This might indicate an issue with the persistent solver's API or the attribute name.")
                    print("Cannot add feasibility cut this iteration.")
                except Exception as e:
                    print(f"ERROR: Failed to extract FarkasDuals with persistent solver: {e}")
                    print("Cannot add feasibility cut this iteration.")
            else: 
                print("WARNING: Subproblem solver is not gurobi_persistent. Attempting ray extraction via suffix (likely all zeros).")
                try:
                    temp_rays_min_power, temp_rays_max_power, temp_rays_ramp_up, temp_rays_ramp_down, temp_rays_demand = {}, {}, {}, {}, {}
                    non_zero_ray_found_std = False
                    if hasattr(subproblem, 'dual'): # Check if dual suffix exists
                         for i in generators:
                            for t_p in time_periods:
                                val_mp = subproblem.dual.get(subproblem.MinPower[i,t_p], 0.0); temp_rays_min_power[(i,t_p)] = val_mp
                                val_maxp = subproblem.dual.get(subproblem.MaxPower[i,t_p], 0.0); temp_rays_max_power[(i,t_p)] = val_maxp
                                val_ru = subproblem.dual.get(subproblem.RampUp[i,t_p], 0.0); temp_rays_ramp_up[(i,t_p)] = val_ru
                                val_rd = subproblem.dual.get(subproblem.RampDown[i,t_p], 0.0); temp_rays_ramp_down[(i,t_p)] = val_rd
                                if any(abs(v) > 1e-9 for v in [val_mp, val_maxp, val_ru, val_rd]): non_zero_ray_found_std = True
                         for t_p in time_periods:
                            val_dem = subproblem.dual.get(subproblem.Demand[t_p], 0.0); temp_rays_demand[t_p] = val_dem
                            if abs(val_dem) > 1e-9: non_zero_ray_found_std = True
                         
                         rays_for_cut['min_power'] = temp_rays_min_power
                         rays_for_cut['max_power'] = temp_rays_max_power
                         rays_for_cut['ramp_up'] = temp_rays_ramp_up
                         rays_for_cut['ramp_down'] = temp_rays_ramp_down
                         rays_for_cut['demand'] = temp_rays_demand
                         can_add_feas_cut = True
                         if not non_zero_ray_found_std:
                            print("WARNING: All FarkasDuals (via suffix method for std solver) are zero.")
                    else:
                        print("WARNING: No 'dual' suffix on subproblem for standard solver ray extraction.")
                except Exception as e:
                    print(f"Error extracting rays via suffix for standard solver: {e}")

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
            if gap <= epsilon:
                print("\nConvergence tolerance met.")
                break
        else:
            print("Gap cannot be calculated yet.")

        if k_iter_count == max_iter:
            print("\nMaximum iterations reached.")
            break

        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data)
        master_results = master_solver.solve(master_problem, tee=False) 

        if master_results.solver.termination_condition == TerminationCondition.optimal:
            master_obj_val_total = pyo.value(master_problem.OBJ)
            current_commitment_cost_master = sum(
                pyo.value(master_problem.Cf[i] * master_problem.u[i, t]) +
                pyo.value(master_problem.Csu[i] * master_problem.zON[i, t]) +
                pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t])
                for i in master_problem.I for t in master_problem.T)
            current_beta_master = sum((2**j) * pyo.value(master_problem.beta_binary[j]) for j in master_problem.BETA_BITS)
            true_lower_bound = current_commitment_cost_master + current_beta_master
            
            Total_penalty = 0
            logic1_penalty_val = pyo.value(master_problem.lambda_logic1 * sum( ( (pyo.value(master_problem.u[i, t]) - pyo.value(master_problem.u_prev[i, t])) - (pyo.value(master_problem.zON[i, t]) - pyo.value(master_problem.zOFF[i, t])) )**2 for i in master_problem.I for t in master_problem.T ))
            logic2_penalty_val = pyo.value(master_problem.lambda_logic2 * sum( pyo.value(master_problem.zON[i, t]) * pyo.value(master_problem.zOFF[i, t]) for i in master_problem.I for t in master_problem.T ))
            if logic1_penalty_val > 1e-3 or logic2_penalty_val > 1e-3:
                Total_penalty +=1
                print(f"WARNING: Logic penalties are non-zero in master: L1={logic1_penalty_val:.4f}, L2={logic2_penalty_val:.4f}")
            
            lower_bound = max(lower_bound, true_lower_bound) 

            print(f"Master Status: Optimal")
            print(f"Master QUBO Objective Value (incl. penalties): {master_obj_val_total:.4f}")
            print(f"Master Commitment Cost part: {current_commitment_cost_master:.4f}")
            print(f"Master Beta Value part: {current_beta_master:.4f}")
            print(f"Updated True Lower Bound (Z_LB): {lower_bound:.4f}")

            u_current = {(i,t): (pyo.value(master_problem.u[i,t]) if master_problem.u[i,t].value is not None else 0.0) for i in generators for t in time_periods}
            zON_current = {(i,t): (pyo.value(master_problem.zON[i,t]) if master_problem.zON[i,t].value is not None else 0.0) for i in generators for t in time_periods}
            zOFF_current = {(i,t): (pyo.value(master_problem.zOFF[i,t]) if master_problem.zOFF[i,t].value is not None else 0.0) for i in generators for t in time_periods}
        elif master_results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Master Problem INFEASIBLE. Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop.")
            break
        else:
            print(f"Master Problem FAILED to solve optimally! Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop.")
            break
            
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) if upper_bound != float('inf') and lower_bound != -float('inf') else float('inf')
    print(f"Final Absolute Gap: {final_gap:.6f}")
    print(f"Iterations Performed: {k_iter_count}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print(f"Are any penalty terms non-zero? {'Yes' if Total_penalty > 0 else 'No'}")

    best_solution_u, best_solution_zON, best_solution_zOFF = None, None, None
    min_cost_for_printing = float('inf')

    for data_item in iteration_data:
        if data_item['type'] == 'optimality' and data_item.get('sub_obj') is not None: 
            commit_c = sum(gen_data[i]['Csu'] * data_item['zON_vals'][i, t] +
                           gen_data[i]['Csd'] * data_item['zOFF_vals'][i, t] +
                           gen_data[i]['Cf'] * data_item['u_vals'][i, t]
                           for i in generators for t in time_periods)
            total_c = commit_c + data_item['sub_obj']
            if abs(total_c - upper_bound) < 1e-5 : 
                if total_c < min_cost_for_printing : 
                    min_cost_for_printing = total_c
                    best_solution_u = data_item['u_vals']
                    best_solution_zON = data_item['zON_vals']
                    best_solution_zOFF = data_item['zOFF_vals']

    if best_solution_u:
        print("\n--- Best Feasible Solution Found (corresponds to Z_UB) ---")
        print(f"Best Total Cost (Upper Bound): {upper_bound:.4f}")
        print("Commitment Schedule (u_it):")
        for t_p in time_periods: print(f"  t={t_p}: ", {i: round(best_solution_u[i,t_p]) for i in generators})

        print("\nFinal Dispatch (p_it) for the best solution:")
        final_subproblem = build_subproblem(best_solution_u, best_solution_zON, best_solution_zOFF)
        
        final_sub_solver = SolverFactory('gurobi') 
        final_sub_results = final_sub_solver.solve(final_subproblem)

        if final_sub_results.solver.termination_condition == TerminationCondition.optimal:
            final_sub_obj = pyo.value(final_subproblem.OBJ)
            final_commit_c = sum(gen_data[i]['Csu'] * best_solution_zON[i,t] + gen_data[i]['Csd'] * best_solution_zOFF[i,t] + gen_data[i]['Cf'] * best_solution_u[i,t] for i in generators for t in time_periods)
            print(f"Final Variable Cost (re-solve): {final_sub_obj:.4f}")
            print(f"Final Commitment Cost: {final_commit_c:.2f}")
            print(f"Final Total Cost (recalculated): {final_commit_c + final_sub_obj:.4f}")
            for t_p in time_periods: print(f"  t={t_p}: ", {i: f"{pyo.value(final_subproblem.p[i,t_p]):.2f}" for i in generators})
            print("Final Demand Check:")
            for t_p in time_periods:
                actual_prod = sum(pyo.value(final_subproblem.p[i,t_p]) for i in generators)
                print(f"  t={t_p}: Prod={actual_prod:.2f}, Demand={demand[t_p]}, Met={actual_prod >= demand[t_p] - 1e-4}")
        else: print(f"Could not re-solve final subproblem. Status: {final_sub_results.solver.termination_condition}")
    else: print("\nNo feasible solution matching UB found for final printout, or UB was not updated from initial inf.")

if __name__ == '__main__':
    main()