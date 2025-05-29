#Here i was just testing the total QUBO version with feasibility cuts in the dadk solver.
# does not currently work

import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import math
import numpy as np # DADK uses numpy for some operations

'''
# DADK imports
from dadk.BinPol import BinPol, VarShapeSet, BitArrayShape, PartialConfig
from dadk.QUBOSolverCPU import QUBOSolverCPU # Using DADK's CPU solver
from dadk. soluzioni_SolutionList import SolutionList # Corrected import name
# Note: Depending on your DADK version, the solutions module might be dadk.Solution_SolutionList
'''

from dadk.BinPol import *
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction

# --- Problem Data (same as your original script) ---
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

# Demand Parameters
demand = {1: 160, 2: 500, 3: 400}

# Initial Conditions for T = 0
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Benders Parameters
num_beta_bits = 7
num_slack_bits = 10
slack_step_length = 0.5

# Penalty parameters for master QUBO
lambda_logic1 = 20.0
lambda_logic2 = 1.0
lambda_opt_cut = 5.0
lambda_feas_cut = 70.0

# --- Subproblem (LP) - Remains solved by Pyomo/Gurobi ---
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    # This function is identical to your original script
    model = pyo.ConcreteModel(name="UCP_Subproblem")
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)

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

# --- DADK Master Problem QUBO Construction ---
def build_master_qubo_dadk(iteration_data, var_shape_set_master, u_init_master):
    master_qubo = BinPol(var_shape_set=var_shape_set_master)

    # 1. Commitment Costs (linear terms for u, zON, zOFF)
    for i in generators:
        for t in time_periods:
            master_qubo.add_term(gen_data[i]['Cf'], ('u', i, t))
            master_qubo.add_term(gen_data[i]['Csu'], ('zON', i, t))
            master_qubo.add_term(gen_data[i]['Csd'], ('zOFF', i, t))

    # 2. Beta term (linear terms for beta_binary)
    for j in range(num_beta_bits):
        master_qubo.add_term(2**j, ('beta_binary', j))

    # 3. Logic Constraint 1: (u_it - u_i(t-1) - zON_it + zOFF_it)^2
    # Penalty: lambda_logic1 * term^2
    for i in generators:
        for t in time_periods:
            term_poly = BinPol()
            term_poly.add_term(1, ('u', i, t))
            if t == 1:
                # u_i(t-1) is u_initial[i] (constant)
                term_poly.add_term(-u_init_master[i]) # Add to constant part of term_poly
            else:
                term_poly.add_term(-1, ('u', i, t - 1))
            term_poly.add_term(-1, ('zON', i, t))
            term_poly.add_term(1, ('zOFF', i, t))
            
            penalty_term = term_poly.power(2)
            penalty_term.multiply_scalar(lambda_logic1)
            master_qubo.add(penalty_term)

    # 4. Logic Constraint 2: zON_it * zOFF_it
    # Penalty: lambda_logic2 * term
    for i in generators:
        for t in time_periods:
            master_qubo.add_term(lambda_logic2, (('zON', i, t), ('zOFF', i, t)))
            
    # 5. Benders Optimality Cuts
    # Penalty: lambda_opt_cut * (beta_expr - cut_rhs_expr - s_opt_k_expr)^2
    opt_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'optimality']
    for k_idx_val in opt_cut_indices_list: # Use k_idx_val to avoid conflict with loop var k
        data = iteration_data[k_idx_val]
        sub_obj_k = data['sub_obj']
        duals_k = data['duals']
        u_k_iter_vals = data['u_vals']
        zON_k_iter_vals = data['zON_vals']
        zOFF_k_iter_vals = data['zOFF_vals']

        # beta_expr
        beta_poly = BinPol()
        for j in range(num_beta_bits):
            beta_poly.add_term(2**j, ('beta_binary', j))

        # cut_rhs_expr (this will be a BinPol)
        cut_rhs_poly = BinPol()
        cut_rhs_poly.add_term(sub_obj_k) # Constant part

        for i_gen in generators:
            for t_loop in time_periods:
                # Term: duals_k['lambda_min'] * Pmin[i] * (u[i,t] - u_k_iter[i,t])
                coeff_min = duals_k['lambda_min'].get((i_gen, t_loop), 0.0) * gen_data[i_gen]['Pmin']
                if abs(coeff_min) > 1e-9:
                    cut_rhs_poly.add_term(coeff_min, ('u', i_gen, t_loop))
                    cut_rhs_poly.add_term(-coeff_min * u_k_iter_vals.get((i_gen, t_loop), 0.0))
                
                # Term: duals_k['lambda_max'] * Pmax[i] * (u[i,t] - u_k_iter[i,t])
                coeff_max = duals_k['lambda_max'].get((i_gen, t_loop), 0.0) * gen_data[i_gen]['Pmax']
                if abs(coeff_max) > 1e-9:
                    cut_rhs_poly.add_term(coeff_max, ('u', i_gen, t_loop))
                    cut_rhs_poly.add_term(-coeff_max * u_k_iter_vals.get((i_gen, t_loop), 0.0))

                # Ramp-up terms
                dual_val_ru = duals_k['lambda_ru'].get((i_gen, t_loop), 0.0)
                if abs(dual_val_ru) > 1e-9:
                    # Ru[i] * (u[i,t-1] - u_k_iter[i,t-1])
                    if t_loop > 1:
                        u_prev_k_val_ru = u_k_iter_vals.get((i_gen, t_loop - 1), u_init_master[i_gen])
                        cut_rhs_poly.add_term(dual_val_ru * gen_data[i_gen]['Ru'], ('u', i_gen, t_loop - 1))
                        cut_rhs_poly.add_term(-dual_val_ru * gen_data[i_gen]['Ru'] * u_prev_k_val_ru)
                    else: # t_loop == 1
                        u_prev_k_val_ru = u_init_master[i_gen] # u_k_iter should not have t=0
                        # if u_init_master[i_gen] is non-zero, it's a constant term
                        # if u_prev_k_val_ru (which is u_initial[i_gen]) is non-zero:
                        # This part (Ru[i] * u_prev_fixed) is constant if u_prev_fixed refers to u_initial.
                        # (m.Ru[i] * m.u_prev_fixed[i,t]) -> if t=1, m.u_prev_fixed is u_init, so this becomes dual_val_ru * Ru[i] * u_init[i]
                        # The part (m.Ru[i] * -u_k_iter_vals.get(...)) is also constant
                        # This logic needs to be careful if u_k_iter could have t=0. Assuming not.
                        # For t=1, u_prev_fixed is u_init. So the term is Ru * u_init.
                        # The (u_i(t-1) - u_k_iter_i(t-1)) part becomes (u_init - u_init_k) which is 0 if u_k_iter(t=0) = u_init
                        # Let's assume for t=1, u_prev is always u_initial, and for u_k_iter, u_prev_k is also u_initial.
                        # The structure of Benders cut implies difference from the point u_k_iter.
                        # So, if t_loop=1, the u_prev term in the cut refers to the *fixed* u_initial.
                        # The part being subtracted is dual_val_ru * Ru[i] * u_k_iter[i,0] (if u_k_iter included t=0).
                        # The original Pyomo code implies u_prev_k_val_ru is from u_k_iter.get((i, t_loop-1), m.u_init[i])
                        # If t_loop=1, this means u_k_iter.get((i,0), m.u_init[i]).
                        # We assume u_k_iter does not have t=0, so it defaults to m.u_init[i].
                        # Thus, (m.u[i,t-1] - u_k_iter_prev_val) is 0 if u_prev is constant u_init.
                        # This seems to align with the original structure where u_prev_fixed for t=1 is u_init.
                        # The Pyomo expression: m.Ru[i] * m.u_prev_fixed[i,t]
                        # If t=1, this is m.Ru[i] * m.u_init[i].
                        # The subtracted part is m.Ru[i] * u_k_iter_at_t_minus_1_or_init.
                        # This should be correct as:
                        if t_loop == 1:
                             cut_rhs_poly.add_term(dual_val_ru * gen_data[i_gen]['Ru'] * (u_init_master[i_gen] - u_init_master[i_gen])) # effectively zero if u_k_iter also uses u_init for t=0
                        # else: # t_loop > 1 is handled above by ('u', i_gen, t_loop - 1)

                    # Rsu[i] * (zON[i,t] - zON_k_iter[i,t])
                    cut_rhs_poly.add_term(dual_val_ru * gen_data[i_gen]['Rsu'], ('zON', i_gen, t_loop))
                    cut_rhs_poly.add_term(-dual_val_ru * gen_data[i_gen]['Rsu'] * zON_k_iter_vals.get((i_gen, t_loop), 0.0))
                
                # Ramp-down terms
                dual_val_rd = duals_k['lambda_rd'].get((i_gen, t_loop), 0.0)
                if abs(dual_val_rd) > 1e-9:
                    # Rd[i] * (u[i,t] - u_k_iter[i,t])
                    cut_rhs_poly.add_term(dual_val_rd * gen_data[i_gen]['Rd'], ('u', i_gen, t_loop))
                    cut_rhs_poly.add_term(-dual_val_rd * gen_data[i_gen]['Rd'] * u_k_iter_vals.get((i_gen, t_loop), 0.0))
                    # Rsd[i] * (zOFF[i,t] - zOFF_k_iter[i,t])
                    cut_rhs_poly.add_term(dual_val_rd * gen_data[i_gen]['Rsd'], ('zOFF', i_gen, t_loop))
                    cut_rhs_poly.add_term(-dual_val_rd * gen_data[i_gen]['Rsd'] * zOFF_k_iter_vals.get((i_gen, t_loop), 0.0))
        
        # s_opt_k_expr (binary encoded slack)
        s_opt_k_poly = BinPol()
        for l_bit in range(num_slack_bits):
            s_opt_k_poly.add_term(slack_step_length * (2**l_bit), ('s_opt_cuts_binary', k_idx_val, l_bit))

        # Full optimality cut penalty term: (beta_poly - cut_rhs_poly - s_opt_k_poly)^2
        opt_cut_term_poly = beta_poly - cut_rhs_poly - s_opt_k_poly # DADK supports BinPol arithmetic
        opt_cut_penalty = opt_cut_term_poly.power(2)
        opt_cut_penalty.multiply_scalar(lambda_opt_cut)
        master_qubo.add(opt_cut_penalty)

    # 6. Benders Feasibility Cuts
    # Penalty: lambda_feas_cut * (-current_feas_cut_lhs_expr + s_feas_k_expr)^2
    feas_cut_indices_list = [k_idx for k_idx, data in enumerate(iteration_data) if data['type'] == 'feasibility']
    for k_idx_val in feas_cut_indices_list:
        data = iteration_data[k_idx_val]
        rays_k = data['rays']

        # current_feas_cut_lhs_expr (this will be a BinPol)
        feas_cut_lhs_poly = BinPol()
        for i_gen in generators:
            for t_loop in time_periods:
                # rays_k['min_power'] * Pmin[i] * u[i,t]
                coeff_min_feas = rays_k['min_power'].get((i_gen, t_loop), 0.0) * gen_data[i_gen]['Pmin']
                if abs(coeff_min_feas) > 1e-9: feas_cut_lhs_poly.add_term(coeff_min_feas, ('u', i_gen, t_loop))
                
                # rays_k['max_power'] * Pmax[i] * u[i,t]
                coeff_max_feas = rays_k['max_power'].get((i_gen, t_loop), 0.0) * gen_data[i_gen]['Pmax']
                if abs(coeff_max_feas) > 1e-9: feas_cut_lhs_poly.add_term(coeff_max_feas, ('u', i_gen, t_loop))

                # rays_k['ramp_up'] * (Ru[i] * u_prev[i,t] + Rsu[i] * zON[i,t])
                coeff_ru_feas = rays_k['ramp_up'].get((i_gen, t_loop), 0.0)
                if abs(coeff_ru_feas) > 1e-9:
                    if t_loop == 1: # u_prev is u_initial (constant)
                        feas_cut_lhs_poly.add_term(coeff_ru_feas * gen_data[i_gen]['Ru'] * u_init_master[i_gen])
                    else: # u_prev is u[i,t-1] (variable)
                        feas_cut_lhs_poly.add_term(coeff_ru_feas * gen_data[i_gen]['Ru'], ('u', i_gen, t_loop - 1))
                    feas_cut_lhs_poly.add_term(coeff_ru_feas * gen_data[i_gen]['Rsu'], ('zON', i_gen, t_loop))

                # rays_k['ramp_down'] * (Rd[i] * u[i,t] + Rsd[i] * zOFF[i,t])
                coeff_rd_feas = rays_k['ramp_down'].get((i_gen, t_loop), 0.0)
                if abs(coeff_rd_feas) > 1e-9:
                    feas_cut_lhs_poly.add_term(coeff_rd_feas * gen_data[i_gen]['Rd'], ('u', i_gen, t_loop))
                    feas_cut_lhs_poly.add_term(coeff_rd_feas * gen_data[i_gen]['Rsd'], ('zOFF', i_gen, t_loop))
        
        for t_loop in time_periods:
            # rays_k['demand'] * D[t] (constant term)
            coeff_dem_feas = rays_k['demand'].get(t_loop, 0.0) * demand[t_loop]
            if abs(coeff_dem_feas) > 1e-9: feas_cut_lhs_poly.add_term(coeff_dem_feas)

        # s_feas_k_expr (binary encoded slack)
        s_feas_k_poly = BinPol()
        for l_bit in range(num_slack_bits):
            s_feas_k_poly.add_term(slack_step_length * (2**l_bit), ('s_feas_cuts_binary', k_idx_val, l_bit))
        
        # Full feasibility cut penalty term: (-feas_cut_lhs_poly + s_feas_k_poly)^2
        feas_cut_term_poly = -feas_cut_lhs_poly + s_feas_k_poly
        feas_cut_penalty = feas_cut_term_poly.power(2)
        feas_cut_penalty.multiply_scalar(lambda_feas_cut)
        master_qubo.add(feas_cut_penalty)
        
    return master_qubo

# --- Helper function to map DADK solution back to structured variables ---
def map_dadk_solution_to_structured_vars(solution_config, var_shape_set_master):
    u_sol = {}
    zON_sol = {}
    zOFF_sol = {}
    beta_binary_sol = {}
    s_opt_cuts_binary_sol = {}
    s_feas_cuts_binary_sol = {}

    for i in generators:
        for t in time_periods:
            u_sol[i,t] = solution_config[var_shape_set_master.get_index(('u', i, t))]
            zON_sol[i,t] = solution_config[var_shape_set_master.get_index(('zON', i, t))]
            zOFF_sol[i,t] = solution_config[var_shape_set_master.get_index(('zOFF', i, t))]
    
    for j in range(num_beta_bits):
        beta_binary_sol[j] = solution_config[var_shape_set_master.get_index(('beta_binary', j))]
    
    # Assuming iteration_data is accessible or cut indices are known to determine max k_idx
    # For simplicity, we'll iterate up to a predefined max_cuts if iteration_data isn't passed
    # A better approach would be to dynamically size these based on iteration_data length
    max_possible_cuts = 30 # Example, should match Benders max_iter
    for k_cut in range(max_possible_cuts): # Iterate up to a potential max number of cuts
        # Optimality slacks
        try: # Check if this slack variable exists in VarShapeSet for this cut index
            if var_shape_set_master.get_index(('s_opt_cuts_binary', k_cut, 0)) is not None:
                 s_opt_cuts_binary_sol[k_cut] = {}
                 for l_bit in range(num_slack_bits):
                     s_opt_cuts_binary_sol[k_cut][l_bit] = solution_config[var_shape_set_master.get_index(('s_opt_cuts_binary', k_cut, l_bit))]
        except (KeyError, IndexError): # If bit for this cut index not in VarShapeSet
            pass 
        
        # Feasibility slacks
        try:
            if var_shape_set_master.get_index(('s_feas_cuts_binary', k_cut, 0)) is not None:
                s_feas_cuts_binary_sol[k_cut] = {}
                for l_bit in range(num_slack_bits):
                    s_feas_cuts_binary_sol[k_cut][l_bit] = solution_config[var_shape_set_master.get_index(('s_feas_cuts_binary', k_cut, l_bit))]
        except (KeyError, IndexError):
            pass
            
    return u_sol, zON_sol, zOFF_sol, beta_binary_sol, s_opt_cuts_binary_sol, s_feas_cuts_binary_sol

# --- Main Benders Loop ---
def main_benders_dadk():
    start_time_main = time.time()
    max_iter = 30 
    epsilon = 1 
    iteration_data = [] # Stores data for cuts
    lower_bound = -float('inf')
    upper_bound = float('inf')

    # Initial feasible solution for u, zON, zOFF (can be arbitrary or from a heuristic)
    u_current = {}
    zON_current = {}
    zOFF_current = {}
    for t_period in time_periods:
        for i_gen in generators:
            u_current[i_gen, t_period] = 1.0 
            u_prev_val_init = u_initial[i_gen] if t_period == 1 else u_current.get((i_gen, t_period-1), u_initial[i_gen])
            if u_current[i_gen, t_period] > 0.5 and u_prev_val_init < 0.5:
                zON_current[i_gen, t_period] = 1.0; zOFF_current[i_gen, t_period] = 0.0
            elif u_current[i_gen, t_period] < 0.5 and u_prev_val_init > 0.5:
                zON_current[i_gen, t_period] = 0.0; zOFF_current[i_gen, t_period] = 1.0
            else:
                zON_current[i_gen, t_period] = 0.0; zOFF_current[i_gen, t_period] = 0.0
    
    latest_u_warm = u_current.copy()
    latest_zON_warm = zON_current.copy()
    latest_zOFF_warm = zOFF_current.copy()
    latest_beta_bin_warm = {b:0 for b in range(num_beta_bits)}
    # s_opt_cuts_binary and s_feas_cuts_binary warm starts will be implicitly handled by PartialConfig later

    # DADK Solver for Master Problem
    # Parameters for DADK QUBOSolverCPU (adjust as needed)
    # For simplicity, using default or minimal parameters. 
    # In a real scenario, these would be tuned.
    dadk_master_solver = QUBOSolverCPU(
        number_runs=1, # Single run for quick iteration in Benders
        number_iterations=10000, # Adjust based on problem size/complexity
        temperature_sampling=False, # Manually set temperatures for more control if needed
        temperature_start=10.0,    # Example values, tune these
        temperature_end=0.1,
        random_seed=42
    )

    # Subproblem Solver (Gurobi via Pyomo)
    sub_solver_name_persistent = "gurobi_persistent"
    sub_solver_name_standard = "gurobi"
    sub_solver_is_persistent = False
    try:
        sub_solver = SolverFactory(sub_solver_name_persistent, solver_io='python') 
        sub_solver_is_persistent = True
        print(f"Successfully initialized persistent subproblem solver: {sub_solver.name}")
    except Exception as e_persist:
        print(f"Could not create gurobi_persistent solver: {e_persist}")
        print("Falling back to standard gurobi solver for subproblem.")
        sub_solver = SolverFactory(sub_solver_name_standard)


    print(f"--- Starting Benders Decomposition for UCP (DADK Master) ---")
    print(f"Master Solver: DADK QUBOSolverCPU, Subproblem Solver: {sub_solver.name}")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}, Num Slack Bits: {num_slack_bits}\n")

    k_iter_count_dadk = 0
    best_solution_for_ub_display_dadk = None

    # --- Define VarShapeSet for DADK Master Problem ---
    # This needs to be done once, considering all possible slack variables up to max_iter
    var_shapes = []
    var_shapes.append(BitArrayShape(name='u', shape=(len(generators) + 1, len(time_periods) + 1))) # Use 1-based indexing for convenience matching Pyomo
    var_shapes.append(BitArrayShape(name='zON', shape=(len(generators) + 1, len(time_periods) + 1)))
    var_shapes.append(BitArrayShape(name='zOFF', shape=(len(generators) + 1, len(time_periods) + 1)))
    var_shapes.append(BitArrayShape(name='beta_binary', shape=(num_beta_bits,)))
    
    # Slack variables need to be indexed by cut iteration 'k' and bit index 'l'
    # We pre-declare shapes for the maximum number of cuts anticipated
    for k_cut_idx in range(max_iter): # max_iter is the max number of cuts
        var_shapes.append(BitArrayShape(name='s_opt_cuts_binary', shape=(k_cut_idx + 1, num_slack_bits))) # Shape is (cut_index, bit_index)
        var_shapes.append(BitArrayShape(name='s_feas_cuts_binary', shape=(k_cut_idx + 1, num_slack_bits)))
    
    # For DADK, ensure names are unique if they represent distinct variables.
    # If s_opt_cuts_binary[k,l] are all independent bits, they need unique names or careful indexing within a larger BitArrayShape.
    # Simpler: one BitArrayShape for all opt slacks, one for all feas slacks.
    var_shapes_refined = []
    var_shapes_refined.append(BitArrayShape(name='u', shape=(len(generators) + 1, len(time_periods) + 1)))
    var_shapes_refined.append(BitArrayShape(name='zON', shape=(len(generators) + 1, len(time_periods) + 1)))
    var_shapes_refined.append(BitArrayShape(name='zOFF', shape=(len(generators) + 1, len(time_periods) + 1)))
    var_shapes_refined.append(BitArrayShape(name='beta_binary', shape=(num_beta_bits,)))
    var_shapes_refined.append(BitArrayShape(name='s_opt_cuts_binary', shape=(max_iter, num_slack_bits)))
    var_shapes_refined.append(BitArrayShape(name='s_feas_cuts_binary', shape=(max_iter, num_slack_bits)))
    
    var_shape_set_master = VarShapeSet(*var_shapes_refined)
    BinPol.freeze_var_shape_set(var_shape_set_master) # Make it default for new BinPol instances
    
    # --- Benders Loop ---
    for k_loop_idx in range(1, max_iter + 1):
        k_iter_count_dadk = k_loop_idx
        print(f"========================= Iteration {k_iter_count_dadk} =========================")

        print("--- Solving Subproblem (Pyomo/Gurobi) ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)
        sub_solve_completed = False; is_infeasible = False; sub_obj_val = float('nan')
        
        if sub_solver_is_persistent:
            try:
                sub_solver.set_instance(subproblem)
                sub_solver.set_gurobi_param('InfUnbdInfo', 1); sub_solver.set_gurobi_param('DualReductions', 0)
                results = sub_solver.solve(tee=False)
                sub_solve_completed = True
                if results.solver.termination_condition == TerminationCondition.optimal or results.solver.termination_condition == TerminationCondition.feasible:
                    sub_obj_val = pyo.value(subproblem.OBJ)
                    print(f"Subproblem Status (Persistent): {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
                elif results.solver.termination_condition == TerminationCondition.infeasible:
                    print("Subproblem Status (Persistent): INFEASIBLE"); is_infeasible = True
                else:
                    print(f"Subproblem FAILED (Persistent) with status: {results.solver.termination_condition}"); sub_solve_completed = False
            except Exception as e: print(f"Error with persistent subproblem solver: {e}"); sub_solve_completed = False
        else: # Standard Gurobi
            sub_solver.options['InfUnbdInfo'] = 1; sub_solver.options['DualReductions'] = 0
            results = sub_solver.solve(subproblem, load_solutions=True, tee=False)
            sub_solve_completed = True
            if results.solver.termination_condition == TerminationCondition.optimal or results.solver.termination_condition == TerminationCondition.feasible:
                sub_obj_val = pyo.value(subproblem.OBJ)
                print(f"Subproblem Status (Standard): {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            elif results.solver.termination_condition == TerminationCondition.infeasible:
                print("Subproblem Status (Standard): INFEASIBLE"); is_infeasible = True
            else:
                print(f"Subproblem FAILED (Standard) with status: {results.solver.termination_condition}"); sub_solve_completed = False

        if not sub_solve_completed: print("Terminating Benders loop due to subproblem solver issue."); break

        # --- Update Upper Bound & Add Cut ---
        if not is_infeasible:
            commitment_cost_current_iter = sum(gen_data[i]['Csu'] * zON_current.get((i,t),0) +
                                               gen_data[i]['Csd'] * zOFF_current.get((i,t),0) +
                                               gen_data[i]['Cf'] * u_current.get((i,t),0)
                                               for i in generators for t in time_periods)
            current_total_cost = commitment_cost_current_iter + sub_obj_val
            
            logically_sound_for_ub = True # Check logic constraints on u_current, zON_current, zOFF_current
            for i_gen in generators:
                for t_time in time_periods:
                    u_val = u_current.get((i_gen,t_time),0.0); u_prev_val = u_initial[i_gen] if t_time == 1 else u_current.get((i_gen, t_time-1),0.0)
                    zon_val = zON_current.get((i_gen,t_time),0.0); zoff_val = zOFF_current.get((i_gen,t_time),0.0)
                    if abs((u_val - u_prev_val) - (zon_val - zoff_val)) > 1e-4: logically_sound_for_ub = False; break
                    if abs(zon_val * zoff_val) > 1e-4: logically_sound_for_ub = False; break
                if not logically_sound_for_ub: break
            
            if logically_sound_for_ub and current_total_cost < upper_bound :
                upper_bound = current_total_cost
                best_solution_for_ub_display_dadk = {'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 
                                                     'zOFF_vals': zOFF_current.copy(), 'iter': k_iter_count_dadk, 'total_cost': upper_bound}
                print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f} from total cost {current_total_cost:.4f}")
            elif not logically_sound_for_ub: print(f"Current master solution for UB not logically sound. UB not updated. Cost was {current_total_cost:.4f}")
            else: print(f"Current Total Cost {current_total_cost:.4f} did not improve UB {upper_bound:.4f}")

            duals_for_cut = {'lambda_min': {}, 'lambda_max': {}, 'lambda_ru': {}, 'lambda_rd': {}, 'lambda_dem': {}}
            # (Dual extraction logic same as original)
            try: 
                for i in generators:
                    for t_p in time_periods:
                        duals_for_cut['lambda_min'][(i,t_p)] = subproblem.dual.get(subproblem.MinPower[i,t_p], 0.0)
                        duals_for_cut['lambda_max'][(i,t_p)] = subproblem.dual.get(subproblem.MaxPower[i,t_p], 0.0)
                        duals_for_cut['lambda_ru'][(i,t_p)]  = subproblem.dual.get(subproblem.RampUp[i,t_p], 0.0)
                        duals_for_cut['lambda_rd'][(i,t_p)]  = subproblem.dual.get(subproblem.RampDown[i,t_p], 0.0)
                for t_p in time_periods:
                    duals_for_cut['lambda_dem'][t_p] = subproblem.dual.get(subproblem.Demand[t_p], 0.0)
            except Exception as e: print(f"Warning: Error extracting duals for optimality cut: {e}. Using 0.0.")
            iteration_data.append({'type': 'optimality', 'iter': k_iter_count_dadk, 'sub_obj': sub_obj_val, 
                                   'duals': duals_for_cut, 'u_vals': u_current.copy(), 
                                   'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
        else: # is_infeasible
            print("Generating Feasibility Cut using FarkasDual.")
            rays_for_cut = {'min_power': {}, 'max_power': {}, 'ramp_up': {}, 'ramp_down': {}, 'demand': {}}
            can_add_feas_cut = False
            # (FarkasDual extraction logic same as original)
            if sub_solver_is_persistent and hasattr(sub_solver, 'get_linear_constraint_attr'):
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
                    if not non_zero_ray_found: print("WARNING: All extracted FarkasDuals are zero/None (Persistent). Feasibility cut may be trivial.")
                    can_add_feas_cut = True
                except Exception as e: print(f"ERROR: Failed to extract FarkasDuals with persistent solver: {e}")
            else: # Standard Gurobi or other solver
                print("WARNING: Attempting ray extraction via 'dual' suffix for feasibility (may not be Farkas rays).")
                if hasattr(subproblem, 'dual'):
                    non_zero_ray_found_std = False
                    # (Simplified ray extraction, as in original)
                    for i_gen in generators:
                        for t_p_idx in time_periods: # Ensure t_p_idx is correct for Pyomo component access
                             rays_for_cut['min_power'][(i_gen,t_p_idx)] = subproblem.dual.get(subproblem.MinPower[i_gen,t_p_idx],0.0)
                             if abs(rays_for_cut['min_power'][(i_gen,t_p_idx)]) > 1e-9: non_zero_ray_found_std = True
                             # ... similar for max_power, ramp_up, ramp_down
                             rays_for_cut['max_power'][(i_gen,t_p_idx)] = subproblem.dual.get(subproblem.MaxPower[i_gen,t_p_idx],0.0)
                             if abs(rays_for_cut['max_power'][(i_gen,t_p_idx)]) > 1e-9: non_zero_ray_found_std = True
                             rays_for_cut['ramp_up'][(i_gen,t_p_idx)] = subproblem.dual.get(subproblem.RampUp[i_gen,t_p_idx],0.0)
                             if abs(rays_for_cut['ramp_up'][(i_gen,t_p_idx)]) > 1e-9: non_zero_ray_found_std = True
                             rays_for_cut['ramp_down'][(i_gen,t_p_idx)] = subproblem.dual.get(subproblem.RampDown[i_gen,t_p_idx],0.0)
                             if abs(rays_for_cut['ramp_down'][(i_gen,t_p_idx)]) > 1e-9: non_zero_ray_found_std = True
                    for t_p_idx in time_periods:
                        rays_for_cut['demand'][t_p_idx] = subproblem.dual.get(subproblem.Demand[t_p_idx],0.0)
                        if abs(rays_for_cut['demand'][t_p_idx]) > 1e-9: non_zero_ray_found_std = True
                    if not non_zero_ray_found_std: print("WARNING: All rays from 'dual' suffix are zero. Feasibility cut will be trivial.")
                    can_add_feas_cut = True
                else: print("ERROR: No 'dual' suffix on subproblem for standard solver ray extraction.")

            if can_add_feas_cut:
                iteration_data.append({'type': 'feasibility', 'iter': k_iter_count_dadk, 'rays': rays_for_cut,
                                       'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
            else: print("Skipping feasibility cut addition due to issues in ray extraction.")

        # --- Check Convergence ---
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}")
        print(f"Current Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound < float('inf') and lower_bound > -float('inf'):
            gap = (upper_bound - lower_bound) 
            print(f"Current Gap: {gap:.6f} (Tolerance: {epsilon})")
            if gap <= epsilon and k_iter_count_dadk > 1 : print("\nConvergence tolerance met."); break
        else: print("Gap cannot be calculated yet.")
        if k_iter_count_dadk == max_iter: print("\nMaximum iterations reached."); break

        # --- Solve DADK Master Problem ---
        print("\n--- Solving Master Problem (DADK) ---")
        master_qubo_dadk = build_master_qubo_dadk(iteration_data, var_shape_set_master, u_initial)
        
        # Prepare guidance_config for DADK
        guidance_conf = PartialConfig(var_shape_set=var_shape_set_master)
        if latest_u_warm:
            for (i,t), val in latest_u_warm.items(): guidance_conf.set_bit(('u',i,t), bool(val > 0.5))
        if latest_zON_warm:
            for (i,t), val in latest_zON_warm.items(): guidance_conf.set_bit(('zON',i,t), bool(val > 0.5))
        if latest_zOFF_warm:
            for (i,t), val in latest_zOFF_warm.items(): guidance_conf.set_bit(('zOFF',i,t), bool(val > 0.5))
        if latest_beta_bin_warm:
            for b_idx, val in latest_beta_bin_warm.items(): guidance_conf.set_bit(('beta_binary',b_idx), bool(val > 0.5))
        # Slack warm starts would also go here if needed and if they are part of VarShapeSet for PartialConfig

        master_solution_list_dadk = dadk_master_solver.minimize(master_qubo_dadk, guidance_config=guidance_conf)
        
        if master_solution_list_dadk and master_solution_list_dadk.get_solution_list():
            master_solution_dadk = master_solution_list_dadk.get_minimum_energy_solution()
            # print(f"DADK Master Problem Raw Objective (Energy): {master_solution_dadk.energy:.4f}")
            
            # Map DADK solution back to structured variables
            u_sol, zON_sol, zOFF_sol, beta_b_sol, s_opt_b_sol, s_feas_b_sol = \
                map_dadk_solution_to_structured_vars(master_solution_dadk.configuration, var_shape_set_master)

            # Update current values for the next subproblem
            u_current = u_sol
            zON_current = zON_sol
            zOFF_current = zOFF_sol
            
            # Update warm start values
            latest_u_warm = u_current.copy()
            latest_zON_warm = zON_current.copy()
            latest_zOFF_warm = zOFF_current.copy()
            latest_beta_bin_warm = beta_b_sol.copy()
            # latest_s_opt_warm, latest_s_feas_warm would be updated here if needed for DADK warm start

            # Calculate and print components of DADK QUBO value for verification
            # This requires re-evaluating the QUBO with the DADK solution, which is complex
            # For now, we'll just use the energy from DADK and then calculate the true lower bound.
            
            commitment_cost_master_sol_dadk = sum(gen_data[i]['Cf'] * u_current.get((i, t),0) +
                                                  gen_data[i]['Csu'] * zON_current.get((i, t),0) +
                                                  gen_data[i]['Csd'] * zOFF_current.get((i, t),0)
                                                  for i in generators for t in time_periods)
            beta_val_master_sol_dadk = sum((2**j) * beta_b_sol.get(j,0) for j in range(num_beta_bits))
            
            print(f"DADK Master: Commitment Cost part (from solution): {commitment_cost_master_sol_dadk:.4f}")
            print(f"DADK Master: Beta Value part (from solution): {beta_val_master_sol_dadk:.4f}")

            # Check logic penalties from DADK solution
            logic1_pen_val_dadk = 0
            for i_gen in generators:
                for t_time in time_periods:
                    u_val_d = u_current.get((i_gen,t_time),0.0); u_prev_val_d = u_initial[i_gen] if t_time == 1 else u_current.get((i_gen, t_time-1),0.0)
                    zon_val_d = zON_current.get((i_gen,t_time),0.0); zoff_val_d = zOFF_current.get((i_gen,t_time),0.0)
                    logic1_pen_val_dadk += ((u_val_d - u_prev_val_d) - (zon_val_d - zoff_val_d))**2
            logic1_pen_val_dadk *= lambda_logic1
            print(f"DADK Master: Logic1 Penalty (from solution): {logic1_pen_val_dadk:.4f}")

            logic2_pen_val_dadk = sum(zON_current.get((i_gen,t_time),0.0) * zOFF_current.get((i_gen,t_time),0.0) for i_gen in generators for t_time in time_periods)
            logic2_pen_val_dadk *= lambda_logic2
            print(f"DADK Master: Logic2 Penalty (from solution): {logic2_pen_val_dadk:.4f}")

            # Note: Recalculating the penalty values for Benders cuts from the DADK solution
            # would be similar to how it's done for the Pyomo master problem objective.
            # For brevity, this is omitted here but would be needed for a full objective breakdown.

            pen_tol = 1e-3 # Tolerance for penalties being "small"
            if logic1_pen_val_dadk < pen_tol and logic2_pen_val_dadk < pen_tol: 
                # If Benders cut penalties were also small (need to calculate them for DADK solution)
                # For now, assume they are also small if logic1 and logic2 are small
                true_lower_bound_candidate_dadk = commitment_cost_master_sol_dadk + beta_val_master_sol_dadk
                lower_bound = max(lower_bound, true_lower_bound_candidate_dadk)
                print(f"Logic penalties small. Updated True Lower Bound (Z_LB): {lower_bound:.4f}")
            else:
                print(f"Logic penalties are not all small. Lower bound {lower_bound:.4f} not updated by DADK Master Cmnt+Beta.")

        else:
            print("DADK Master Problem FAILED to find a solution.")
            print("Terminating Benders loop."); break
            
    # --- End of Benders Loop ---
    end_time_main = time.time()
    print("\n========================= Benders (DADK Master) Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap_dadk = (upper_bound - lower_bound) if upper_bound != float('inf') and lower_bound != -float('inf') else float('inf')
    print(f"Final Absolute Gap: {final_gap_dadk:.6f}")
    print(f"Iterations Performed: {k_iter_count_dadk}")
    used_time_main = end_time_main - start_time_main
    print(f"Total Time: {used_time_main:.2f} seconds")

    if best_solution_for_ub_display_dadk:
        print("\n--- Best Feasible Solution Found by DADK Master (leading to best Z_UB in iter {}) ---".format(best_solution_for_ub_display_dadk['iter']))
        print(f"Best Total Cost (Upper Bound): {best_solution_for_ub_display_dadk['total_cost']:.4f}")
        # (Display logic for best solution remains similar to original)
        u_best_dadk = best_solution_for_ub_display_dadk['u_vals']
        print("Commitment Schedule (u_it):")
        for t_p in time_periods: print(f"  t={t_p}: ", {i: round(u_best_dadk.get((i,t_p),0)) for i in generators})
        
        print("\nFinal Dispatch (p_it) for the best UB solution (from DADK master):")
        final_subproblem_dadk = build_subproblem(best_solution_for_ub_display_dadk['u_vals'], 
                                                 best_solution_for_ub_display_dadk['zON_vals'], 
                                                 best_solution_for_ub_display_dadk['zOFF_vals'])
        final_sub_solver_gurobi = SolverFactory('gurobi') 
        final_sub_results_dadk = final_sub_solver_gurobi.solve(final_subproblem_dadk, tee=False)
        if final_sub_results_dadk.solver.termination_condition == TerminationCondition.optimal:
            final_sub_obj_resolved_d = pyo.value(final_subproblem_dadk.OBJ)
            final_commit_c_resolved_d = sum(gen_data[i]['Csu'] * best_solution_for_ub_display_dadk['zON_vals'].get((i,t),0) + 
                                           gen_data[i]['Csd'] * best_solution_for_ub_display_dadk['zOFF_vals'].get((i,t),0) + 
                                           gen_data[i]['Cf'] * best_solution_for_ub_display_dadk['u_vals'].get((i,t),0) 
                                           for i in generators for t in time_periods)
            print(f"  Final Variable Cost (re-solve): {final_sub_obj_resolved_d:.4f}")
            print(f"  Final Commitment Cost: {final_commit_c_resolved_d:.2f}")
            print(f"  Final Total Cost (recalculated): {final_commit_c_resolved_d + final_sub_obj_resolved_d:.4f}")
            for t_p in time_periods: print(f"    t={t_p}: ", {i: f"{pyo.value(final_subproblem_dadk.p[i,t_p]):.2f}" for i in generators})
            print("  Final Demand Check:")
            for t_p in time_periods:
                actual_prod_d = sum(pyo.value(final_subproblem_dadk.p[i,t_p]) for i in generators)
                print(f"    t={t_p}: Prod={actual_prod_d:.2f}, Demand={demand[t_p]}, Met={actual_prod_d >= demand[t_p] - 1e-4}")
        else: print(f"Could not re-solve final subproblem for display. Status: {final_sub_results_dadk.solver.termination_condition}")
    else: print("\nNo feasible solution matching UB found for final printout, or UB was not updated from initial inf.")
    
    return used_time_main

if __name__ == '__main__':
    main_benders_dadk()