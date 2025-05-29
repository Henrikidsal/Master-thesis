##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo/gurobi.
##### All constraints in master problem; Logic 1, Logic 2, optimality cuts and feasibility cuts are penalty terms, not hard constraints.
##### All variables in the master problem are binary.
##### Beta is binary encoded, and can take integer values
##### The slack variables in optimality and feasibility cuts are also binary encoded, they can take float valeus with step lenght d.
##### Feasibility cuts use FarkasDual attributes obtained via gurobi_persistent.
##### This script uses cold starts for the master problem variables.

##### Neccecary imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time

# Choose the number of time periods wanted:
Periods = 3
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
demand = {1: 160, 2: 500, 3: 400} # Demand for each time period

# Initial Conditions for time period = 0
# Only generator 3 is on at t=0, producing 100 MW
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Number of bits for beta variable
# With 7 bits, beta can take values from 0 to 127
num_beta_bits = 7

# Number of bits for slack variables in optimality and feasibility cuts
# With 10 bits, slack variables can take 1024 discrete values, the range depends on the step length
num_slack_bits = 10

# This function creates the continous sub problem (LP)
# Its straightforward, it is a LP with the objective of minimizing the cost of generation
# dual suffixes at the borrom collects dual variables for optimality cuts, and dual rays for feasibility cuts
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):

    # Create a concrete model
    model = pyo.ConcreteModel(name="Sub Problem")

    # Define sets for generators and time periods
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)

    # This part collects the master problem variables
    u_fixed_param_vals = {(i,t): u_fixed_vals[i,t] for i in model.I for t in model.T}
    zON_fixed_param_vals = {(i,t): zON_fixed_vals[i,t] for i in model.I for t in model.T}
    zOFF_fixed_param_vals = {(i,t): zOFF_fixed_vals[i,t] for i in model.I for t in model.T}

    # This part fixes the master problem variables to parameters.
    model.u_fixed = pyo.Param(model.I, model.T, initialize=u_fixed_param_vals)
    model.zON_fixed = pyo.Param(model.I, model.T, initialize=zON_fixed_param_vals)
    model.zOFF_fixed = pyo.Param(model.I, model.T, initialize=zOFF_fixed_param_vals)

    # Here the parameters for the problem are defined
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I}) # Minimum power output for each generator
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I}) # Maximum power output for each generator
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I}) # Ramp down limit for each generator
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I}) # Ramp down start-up limit for each generator
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I}) # Ramp up limit for each generator
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I}) # Ramp up start-up limit for each generator
    model.Cv = pyo.Param(model.I, initialize={i: gen_data[i]['Cv'] for i in model.I}) # Variable cost for each generator
    model.D = pyo.Param(model.T, initialize=demand) # Demand for each time period
    model.u_init = pyo.Param(model.I, initialize=u_initial) # Initial state of each generator t=0
    model.p_init = pyo.Param(model.I, initialize=p_initial) # Initial power output for each generator t=0

    # We only have this one variable in the sub problem, the power output of each generator
    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)

    # Objective function, minimizing the total cost of generation
    def objective_rule(m):
        return sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # just a clever way of always ensuring that we correctly uses the previous time periods power output
    def p_prev_rule(m, i, t):
        return m.p_init[i] if t == 1 else m.p[i, t-1]
    model.p_prev = pyo.Expression(model.I, model.T, rule=p_prev_rule)

    # A clever way of ennsuring that we uses the correct on/off state of the generator in each time period
    def u_prev_fixed_rule(m, i, t):
        return m.u_init[i] if t == 1 else m.u_fixed[i, t-1]
    model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=u_prev_fixed_rule)

    # The constraints
    model.MinPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t])
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t])
    model.RampUp = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p[i,t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t])
    model.RampDown = pyo.Constraint(model.I, model.T, rule=lambda m,i,t: m.p_prev[i,t] - m.p[i,t] <= m.Rd[i] * m.u_fixed[i,t] + m.Rsd[i] * m.zOFF_fixed[i,t])
    model.Demand = pyo.Constraint(model.T, rule=lambda m, t: sum(m.p[i, t] for i in m.I) >= m.D[t])

    # Collects dual variables for optimality cuts, and dual rays for feasibility cuts.
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    return model

# Function creating the master problem
def build_master(iteration_data):

    # Creates an empty model
    model = pyo.ConcreteModel(name="Master Problem")

    # This creates one list for optimality cuts and one for feasibility cuts. The list contains the indices of the cuts.
    opt_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'optimality']
    feas_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'feasibility']
    
    # This creates the sets for both optimality and feasibility cuts, using the lists above
    model.OptCutIndices = pyo.Set(initialize=opt_cut_indices_list)
    model.FeasCutIndices = pyo.Set(initialize=feas_cut_indices_list)

    # Sets
    model.I = pyo.Set(initialize=generators) # Set of generators
    model.T = pyo.Set(initialize=time_periods) # Set of time periods
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1) # Set for binary encoding of beta
    model.SLACK_BITS = pyo.RangeSet(0, num_slack_bits - 1) # Set for binary encoding of slacks

    # Parameter for slack variables step length
    model.slack_step_length = pyo.Param(initialize=0.5)

    # Parameters 
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I}) # Maximum power output for each generator
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I}) # Minimum power output for each generator
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I}) # Ramp down limit for each generator
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I}) # Ramp down start-up limit for each generator
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I}) # Ramp up limit for each generator
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I}) # Ramp up start-up limit for each generator
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I}) # Fixed cost for having generators on
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I}) # Start-up cost for each generator
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I}) # Shut-down cost for each generator
    model.D_param = pyo.Param(model.T, initialize=demand) # Demand for each time period
    model.u_init = pyo.Param(model.I, initialize=u_initial) # Initial state of each generator t=0
    model.lambda_logic1 = pyo.Param(initialize=20) # Logic 1 penalty term lambda
    model.lambda_logic2 = pyo.Param(initialize=1)  # Logic 2 penalty term lambda
    model.lambda_opt_cut = pyo.Param(initialize=5) # Optimality cut penalty term lambda
    model.lambda_feas_cut = pyo.Param(initialize=1) # Feasibility cut penalty term lambda

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary) # on/off status
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary) # start-up status
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary) # shut-down status
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary) # binary encoding of beta
    model.slack_optimality = pyo.Var(model.OptCutIndices, model.SLACK_BITS, within=pyo.Binary) # binary encoding of optimality cuts slack
    model.slack_feasibility = pyo.Var(model.FeasCutIndices, model.SLACK_BITS, within=pyo.Binary) # binary encoding of feasibility cuts slack

    # A clever way of ennsuring that we uses the correct on/off state of the generator in each time period
    def u_prev_rule(m, i, t):
        return m.u_init[i] if t == 1 else m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    # Objective function
    def master_objective_rule(m):

        # Commitment cost (Original objective function)
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)

        # Logic 1 and Logic 2 penalty terms
        logic1_penalty_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_penalty_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )

        # The beta term, where all bits are summed up to represent the sub problem future cost.
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )

        # Penalty term for the optimality cuts
        # The code iterates through each k_idx in m.OptCutIndices, where k_idx represents one optimality cut.
        optimality_cuts_penalty = 0
        for k_idx in m.OptCutIndices:
            data = iteration_data[k_idx]
            sub_obj_k = data['sub_obj']; duals_k = data['duals']
            u_k_iter = data['u_vals']; zON_k_iter = data['zON_vals']; zOFF_k_iter = data['zOFF_vals']
            
            cut_rhs_expr = sub_obj_k 
            for i in m.I:
                for t_loop in m.T:
                    cut_rhs_expr += duals_k['lambda_min'].get((i, t_loop), 0.0) * m.Pmin[i] * (m.u[i, t_loop] - u_k_iter.get((i,t_loop), 0.0))
                    cut_rhs_expr += duals_k['lambda_max'].get((i, t_loop), 0.0) * m.Pmax[i] * (m.u[i, t_loop] - u_k_iter.get((i,t_loop), 0.0))
                    dual_val_ru = duals_k['lambda_ru'].get((i, t_loop), 0.0)
                    u_prev_term_expr_ru = 0
                    u_prev_k_val_ru = u_k_iter.get((i, t_loop-1), m.u_init[i]) if t_loop > 1 else m.u_init[i]
                    if t_loop > 1: u_prev_term_expr_ru = m.Ru[i] * (m.u[i, t_loop-1] - u_prev_k_val_ru)
                    zON_term_expr_ru = m.Rsu[i] * (m.zON[i, t_loop] - zON_k_iter.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val_ru * (u_prev_term_expr_ru + zON_term_expr_ru)
                    dual_val_rd = duals_k['lambda_rd'].get((i, t_loop), 0.0)
                    u_term_expr_rd = m.Rd[i] * (m.u[i, t_loop] - u_k_iter.get((i, t_loop), 0.0))
                    zOFF_term_expr_rd = m.Rsd[i] * (m.zOFF[i, t_loop] - zOFF_k_iter.get((i, t_loop), 0.0))
                    cut_rhs_expr += dual_val_rd * (u_term_expr_rd + zOFF_term_expr_rd)
            
            
            s_opt_k_expr = m.slack_step_length * sum( (2**l) * m.slack_optimality[k_idx, l] for l in m.SLACK_BITS )
            optimality_cuts_penalty += (binary_beta_expr - cut_rhs_expr - s_opt_k_expr)**2
        optimality_cuts_penalty *= m.lambda_opt_cut

        # Penalty term for the feasibility cuts
        # The code iterates through each k_idx in m.FeasCutIndices, where k_idx represents one feasibility cut.
        feasibility_cuts_penalty = 0
        for k_idx in m.FeasCutIndices:
            data = iteration_data[k_idx]; rays_k = data['rays']
            current_feas_cut_lhs_expr = 0
            for i_gen in m.I:
                for t_period in m.T:
                    current_feas_cut_lhs_expr += rays_k['min_power'].get((i_gen, t_period), 0.0) * (m.Pmin[i_gen] * m.u[i_gen, t_period])
                    current_feas_cut_lhs_expr += rays_k['max_power'].get((i_gen, t_period), 0.0) * (m.Pmax[i_gen] * m.u[i_gen, t_period])
                    current_feas_cut_lhs_expr += rays_k['ramp_up'].get((i_gen, t_period), 0.0) * \
                                                 (m.Ru[i_gen] * m.u_prev[i_gen, t_period] + m.Rsu[i_gen] * m.zON[i_gen, t_period])
                    current_feas_cut_lhs_expr += rays_k['ramp_down'].get((i_gen, t_period), 0.0) * \
                                                 (m.Rd[i_gen] * m.u[i_gen, t_period] + m.Rsd[i_gen] * m.zOFF[i_gen, t_period])
            for t_period in m.T:
                    current_feas_cut_lhs_expr += rays_k['demand'].get(t_period, 0.0) * m.D_param[t_period]
            
            # Reconstruct slack value from binary variables, to calculate the penalty term
            s_feas_k_expr = m.slack_step_length * sum( (2**l) * m.slack_feasibility[k_idx, l] for l in m.SLACK_BITS )
            feasibility_cuts_penalty += (-current_feas_cut_lhs_expr + s_feas_k_expr)**2
        feasibility_cuts_penalty *= m.lambda_feas_cut
        
        return commitment_cost + logic1_penalty_term + logic2_penalty_term + \
               binary_beta_expr + optimality_cuts_penalty + feasibility_cuts_penalty

    # Minimizing this objective function which is our master problem
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)
    return model

# Main Benders Loop, the algorithm.
def main():
    start_time = time.time()
    max_iter = 30 
    epsilon = 1 # Tolerance for convergence
    iteration_data = []
    lower_bound = -float('inf')
    upper_bound = float('inf')

    # Here I am making the initial guess for the master problem variables
    # The sub problem is solved first, and needs master variables.
    # start by creating three empty dicts for u, zON and zOFF
    u_current = {}
    zON_current = {}
    zOFF_current = {}

    # Then the initial guess says all gens are on in each time period (standard to start with)
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0 

            # Then we must find out with this initial guess, how does that affect the zON and zOFF variables
            u_prev_val = u_initial[i] if t == 1 else u_current.get((i, t-1), u_initial[i]) 
            if u_current[i, t] > 0.5 and u_prev_val < 0.5:
                zON_current[i, t] = 1.0; zOFF_current[i, t] = 0.0
            elif u_current[i, t] < 0.5 and u_prev_val > 0.5:
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 1.0
            else: 
                zON_current[i, t] = 0.0; zOFF_current[i, t] = 0.0
    
    # Uses guvobi as MP solver
    master_solver_name = "gurobi"
    master_solver = SolverFactory(master_solver_name)
    
    # Uses gurobi_persistent as SP solver, this solver can be used to extract FarkasDuals.
    sub_solver_name = "gurobi_persistent"
    sub_solver_is_persistent = False
    sub_solver = SolverFactory(sub_solver_name, solver_io='python') 
    sub_solver_is_persistent = True

    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Master Solver: {master_solver_name}, Subproblem Solver Name: {sub_solver.name}")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}, Num Slack Bits: {num_slack_bits} (Max Slack Value: {0.1*(2**num_slack_bits-1):.1f})\n")


    k_iter_count = 0
    best_solution_for_ub_display = None

    for k_loop_idx in range(1, max_iter + 1):
        k_iter_count = k_loop_idx
        print(f"========================= Iteration {k_iter_count} =========================")

        print("--- Solving Subproblem ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)

        is_infeasible = False
        sub_obj_val = float('nan') 

        # Set the initial conditions for the subproblem
        sub_solver.set_instance(subproblem)
        sub_solver.set_gurobi_param('InfUnbdInfo', 1) # Enable unboundedness information
        sub_solver.set_gurobi_param('DualReductions', 0) # Disable dual reductions
        results = sub_solver.solve(tee=False) 
        if results.solver.termination_condition == TerminationCondition.optimal or \
           results.solver.termination_condition == TerminationCondition.feasible:
            sub_obj_val = pyo.value(subproblem.OBJ) 
            print(f"Subproblem Status: {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            print("Subproblem Status: INFEASIBLE")
            is_infeasible = True

        # If the subproblem is feasible, we need to generate an optimality cut
        if not is_infeasible: 
            commitment_cost_current_iter = sum(gen_data[i]['Csu'] * zON_current.get((i,t),0) +
                                               gen_data[i]['Csd'] * zOFF_current.get((i,t),0) +
                                               gen_data[i]['Cf'] * u_current.get((i,t),0)
                                               for i in generators for t in time_periods)
            current_total_cost = commitment_cost_current_iter + sub_obj_val
            logically_sound_for_ub = True
            for i_gen in generators:
                for t_time in time_periods:
                    u_val = u_current.get((i_gen,t_time),0)
                    u_prev_val = u_initial[i_gen] if t_time == 1 else u_current.get((i_gen, t_time-1),0)
                    zon_val = zON_current.get((i_gen,t_time),0)
                    zoff_val = zOFF_current.get((i_gen,t_time),0)
                    if abs((u_val - u_prev_val) - (zon_val - zoff_val)) > 1e-4: 
                        logically_sound_for_ub = False; break
                    if abs(zon_val * zoff_val) > 1e-4: 
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
                print(f"Warning: Error extracting duals for optimality cut: {e}. Using 0.0.")
            iteration_data.append({
                'type': 'optimality', 'iter': k_iter_count, 'sub_obj': sub_obj_val,
                'duals': duals_for_cut, 'u_vals': u_current.copy(), 
                'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
            })
        else:
            # If the subproblem is infeasible, we need to generate a feasibility cut
            print("Generating Feasibility Cut using FarkasDual.")
            rays_for_cut = {'min_power': {}, 'max_power': {}, 'ramp_up': {}, 'ramp_down': {}, 'demand': {}}
            can_add_feas_cut = False
            
            try:
                non_zero_ray_found = False
                for c in subproblem.component_data_objects(Constraint, active=True):
                    ray_val = sub_solver.get_linear_constraint_attr(c, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    if abs(ray_val) > 1e-9: non_zero_ray_found = True
                    parent_component = c.parent_component(); idx = c.index()
                    if parent_component is subproblem.MinPower: rays_for_cut['min_power'][idx] = ray_val
                    elif parent_component is subproblem.MaxPower: rays_for_cut['max_power'][idx] = ray_val
                    elif parent_component is subproblem.RampUp: rays_for_cut['ramp_up'][idx] = ray_val
                    elif parent_component is subproblem.RampDown: rays_for_cut['ramp_down'][idx] = ray_val
                    elif parent_component is subproblem.Demand: rays_for_cut['demand'][idx] = ray_val
                if not non_zero_ray_found: print("WARNING: All extracted FarkasDuals are zero/None.")
                can_add_feas_cut = True
            except Exception as e:
                print(f"ERROR: Failed to extract FarkasDuals with persistent solver: {e}")
            
            if can_add_feas_cut:
                iteration_data.append({
                    'type': 'feasibility', 'iter': k_iter_count, 'rays': rays_for_cut,
                    'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()
                })
            else: print("Skipping feasibility cut addition due to issues in ray extraction.")

        # Checking bounds and convergence
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
            gap = (upper_bound - lower_bound) 
            print(f"Current Gap: {gap:.6f} (Tolerance: {epsilon})")
            if gap <= epsilon and k_iter_count > 1: 
                print("\nConvergence tolerance met.")
                break
        else: print("Gap cannot be calculated yet.")
        if k_iter_count == max_iter:
            print("\nMaximum iterations reached.")
            break

        # Now the sub problem is solved, a cut is generated and its time to solve the master problem.
        print("\n--- Solving Master Problem ---")
        master_problem = build_master(iteration_data) 
        
        master_solver.options['NumericFocus'] = 1 # Can activate this if numerical issues arise
        #master_results = master_solver.solve(master_problem, options={'MIPGap': 0.02}, tee=True) 
        master_results = master_solver.solve(master_problem, tee=True) 

        # Updates the date for the next iteration if the master problem is solved successfully
        if master_results.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.locallyOptimal, TerminationCondition.feasible]:
            master_obj_val_total_qubo = pyo.value(master_problem.OBJ)
            u_current = {(i,t): pyo.value(master_problem.u[i,t]) for i in generators for t in time_periods}
            zON_current = {(i,t): pyo.value(master_problem.zON[i,t]) for i in generators for t in time_periods}
            zOFF_current = {(i,t): pyo.value(master_problem.zOFF[i,t]) for i in generators for t in time_periods}
            
            # calculates information
            commitment_cost_master_sol = sum(pyo.value(master_problem.Cf[i] * master_problem.u[i, t]) + \
                                             pyo.value(master_problem.Csu[i] * master_problem.zON[i, t]) + \
                                             pyo.value(master_problem.Csd[i] * master_problem.zOFF[i, t]) \
                                             for i in master_problem.I for t in master_problem.T)
            beta_val_master_sol = sum((2**j) * pyo.value(master_problem.beta_binary[j]) for j in master_problem.BETA_BITS)
            logic1_pen_val = pyo.value(master_problem.lambda_logic1 * sum( ( (master_problem.u[i, t] - master_problem.u_prev[i, t]) - (master_problem.zON[i, t] - master_problem.zOFF[i, t]) )**2 for i in master_problem.I for t in master_problem.T ))
            logic2_pen_val = pyo.value(master_problem.lambda_logic2 * sum( master_problem.zON[i, t] * master_problem.zOFF[i, t] for i in master_problem.I for t in master_problem.T ))
            
            # This block calculates the optimality cuts penalty value
            opt_pen_total_val = 0
            if hasattr(master_problem, 'OptCutIndices') and len(master_problem.OptCutIndices) > 0 :
                temp_opt_penalty = 0
                for k_idx in master_problem.OptCutIndices:
                    data = iteration_data[k_idx]; sub_obj_k = data['sub_obj']; duals_k = data['duals']
                    u_k_iter = data['u_vals']; zON_k_iter = data['zON_vals']; zOFF_k_iter = data['zOFF_vals']
                    cut_rhs_expr_val = sub_obj_k
                    for i in master_problem.I:
                        for t_loop in master_problem.T:
                            cut_rhs_expr_val += duals_k['lambda_min'].get((i,t_loop),0.)*master_problem.Pmin[i]*(u_current[i,t_loop] - u_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += duals_k['lambda_max'].get((i,t_loop),0.)*master_problem.Pmax[i]*(u_current[i,t_loop] - u_k_iter.get((i,t_loop),0.))
                            dual_val_ru = duals_k['lambda_ru'].get((i,t_loop),0.)
                            u_prev_term_expr_ru_val = 0
                            if t_loop > 1: u_prev_term_expr_ru_val = master_problem.Ru[i]*(u_current[i,t_loop-1]-(u_k_iter.get((i,t_loop-1),master_problem.u_init[i]) if t_loop > 1 else master_problem.u_init[i]))
                            zON_term_expr_ru_val = master_problem.Rsu[i]*(zON_current[i,t_loop]-zON_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += dual_val_ru * (u_prev_term_expr_ru_val + zON_term_expr_ru_val)
                            dual_val_rd = duals_k['lambda_rd'].get((i,t_loop),0.)
                            u_term_expr_rd_val = master_problem.Rd[i]*(u_current[i,t_loop]-u_k_iter.get((i,t_loop),0.))
                            zOFF_term_expr_rd_val = master_problem.Rsd[i]*(zOFF_current[i,t_loop]-zOFF_k_iter.get((i,t_loop),0.))
                            cut_rhs_expr_val += dual_val_rd * (u_term_expr_rd_val + zOFF_term_expr_rd_val)
                    s_opt_k_val = pyo.value(master_problem.slack_step_length * sum((2**l) * master_problem.slack_optimality[k_idx,l] for l in master_problem.SLACK_BITS))
                    temp_opt_penalty += (beta_val_master_sol - cut_rhs_expr_val - s_opt_k_val)**2
                opt_pen_total_val = pyo.value(master_problem.lambda_opt_cut) * temp_opt_penalty

            # This block calculates the feasibility cuts penalty value
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
                    s_feas_k_val = pyo.value(master_problem.slack_step_length * sum((2**l) * master_problem.slack_feasibility[k_idx,l] for l in master_problem.SLACK_BITS))
                    temp_feas_penalty += (-current_feas_cut_lhs_expr_val + s_feas_k_val)**2
                feas_pen_total_val = pyo.value(master_problem.lambda_feas_cut) * temp_feas_penalty
            
            print(f"Master Status: {master_results.solver.termination_condition}")
            print(f"Master QUBO Objective Value: {master_obj_val_total_qubo:.4f}")
            print(f"  Commitment Cost part: {commitment_cost_master_sol:.4f}")
            print(f"  Beta Value part: {beta_val_master_sol:.4f}")
            print(f"  Logic1 Penalty part: {logic1_pen_val:.4f}")
            print(f"  Logic2 Penalty part: {logic2_pen_val:.4f}")
            print(f"  Optimality Cuts Penalty part: {opt_pen_total_val:.4f}")
            print(f"  Feasibility Cuts Penalty part: {feas_pen_total_val:.4f}")

            for k_idx in master_problem.OptCutIndices:
                decoded_integer_slack_value_opt = 0
                for l in master_problem.SLACK_BITS:
                    # Ensure slack_optimality has a value before accessing it
                    if master_problem.slack_optimality[k_idx, l].value is not None:
                        decoded_integer_slack_value_opt += (2**l) * pyo.value(master_problem.slack_optimality[k_idx, l])
                
                # Multiply by the step length to get the actual slack value
                actual_slack_value_opt = master_problem.slack_step_length.value * decoded_integer_slack_value_opt
                print(f"  Optimality Cut {k_idx} Decoded Slack Value: {actual_slack_value_opt:.4f} (Integer part: {decoded_integer_slack_value_opt:.0f})")

            # For Feasibility Cuts
            for k_idx in master_problem.FeasCutIndices:
                decoded_integer_slack_value_feas = 0
                for l in master_problem.SLACK_BITS:
                    # Ensure slack_feasibility has a value before accessing it
                    if master_problem.slack_feasibility[k_idx, l].value is not None:
                        decoded_integer_slack_value_feas += (2**l) * pyo.value(master_problem.slack_feasibility[k_idx, l])
                
                # Multiply by the step length to get the actual slack value
                actual_slack_value_feas = master_problem.slack_step_length.value * decoded_integer_slack_value_feas
                print(f"  Feasibility Cut {k_idx} Decoded Slack Value: {actual_slack_value_feas:.4f} (Integer part: {decoded_integer_slack_value_feas:.0f})")

            Total_penalty=0
            pen_tol = 1e-3 
            if logic1_pen_val < pen_tol and logic2_pen_val < pen_tol and \
               opt_pen_total_val < pen_tol and feas_pen_total_val < pen_tol:
                true_lower_bound_candidate = commitment_cost_master_sol + beta_val_master_sol -1
                lower_bound = max(lower_bound, true_lower_bound_candidate)
                print(f"All penalties small. Updated Lower Bound: {lower_bound:.4f}")
            else:
                Total_penalty += 1
                print(f"Penalties are NOT small. Lower bound {lower_bound:.4f} not updated")

        elif master_results.solver.termination_condition == TerminationCondition.infeasible:
            print(f"Master Problem INFEASIBLE. Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop."); break
        else:
            print(f"Master Problem FAILED to solve optimally/feasibly! Status: {master_results.solver.termination_condition}")
            print("Terminating Benders loop."); break
            
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) if upper_bound != float('inf') and lower_bound != -float('inf') else float('inf')
    print(f"Final Absolute Gap: {final_gap:.6f}")
    print(f"Iterations Performed: {k_iter_count}")
    used_time = end_time - start_time
    print(f"Total Time: {used_time:.2f} seconds")
    print(f"Are any penalty terms violated? {'Yes' if Total_penalty > 0 else 'No'}")

    # Here we look at the best solution found for the upper bound and prints a final report
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
    avg_used_time = 0
    N=1
    for _ in range(N):
        used_time = main()
        avg_used_time+=used_time/N
    print("average used time = ", avg_used_time)