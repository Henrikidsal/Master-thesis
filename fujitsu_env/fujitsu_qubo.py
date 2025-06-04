##### This is a script that solves the UCP using a hybrid Benders Decomposition.
##### The subproblem is a Pyomo/Gurobi LP.
##### The master problem is solved using DADK's QUBOSolverCPU and Pyomo/Gurobi for the relaxed LP.

# Neccecary imports
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import time
import numpy as np
from datetime import datetime

# DADK imports
from dadk.BinPol import *
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction

num_tries = 20
num_runs = 128
num_replicas = 128
num_iter = 70_000
BETA_BITS = 7
num_slack_bits = 8
SCALE_FACTOR = 1
lambda1 = 110*SCALE_FACTOR
lambda2 = 100*SCALE_FACTOR
lambda3 = 100*SCALE_FACTOR
lambda4 = 100*SCALE_FACTOR

# Choose the number of time periods wanted:
Periods = 3
# Sets
generators = [1, 2, 3] # 1-indexed
time_periods = [x for x in range(1, Periods + 1)] # 1-indexed, T=3 hours

# Generator Parameters from the PDF
gen_data = {
    1: {'Pmin': 50,  'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
    2: {'Pmin': 80,  'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
    3: {'Pmin': 40,  'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5,  'Csd': 1.0, 'Cv': 0.150}
}

# Demand Parameters from the PDF
demand = {1: 160, 2: 500, 3: 400} # Demand for each time period

# Initial Conditions for time period = 0, from the PDF
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Global mapping for DADK symbolic names
gen_map = {orig_g: idx for idx, orig_g in enumerate(generators)}
time_map = {orig_t: idx for idx, orig_t in enumerate(time_periods)}

# --- Benders Decomposition Functions ---

def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    model = pyo.ConcreteModel(name="Sub_Problem")
    model.I_set = pyo.Set(initialize=generators)
    model.T_set = pyo.Set(initialize=time_periods)

    u_fixed_param_vals = {(i, t): u_fixed_vals.get((i, t), 0.0) for i in model.I_set for t in model.T_set}
    zON_fixed_param_vals = {(i, t): zON_fixed_vals.get((i, t), 0.0) for i in model.I_set for t in model.T_set}
    zOFF_fixed_param_vals = {(i, t): zOFF_fixed_vals.get((i, t), 0.0) for i in model.I_set for t in model.T_set}

    model.u_fixed = pyo.Param(model.I_set, model.T_set, initialize=u_fixed_param_vals)
    model.zON_fixed = pyo.Param(model.I_set, model.T_set, initialize=zON_fixed_param_vals)
    model.zOFF_fixed = pyo.Param(model.I_set, model.T_set, initialize=zOFF_fixed_param_vals)

    model.Pmin = pyo.Param(model.I_set, initialize={i: gen_data[i]['Pmin'] for i in model.I_set})
    model.Pmax = pyo.Param(model.I_set, initialize={i: gen_data[i]['Pmax'] for i in model.I_set})
    model.Rd = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rd'] for i in model.I_set})
    model.Rsd = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rsd'] for i in model.I_set})
    model.Ru = pyo.Param(model.I_set, initialize={i: gen_data[i]['Ru'] for i in model.I_set})
    model.Rsu = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rsu'] for i in model.I_set})
    model.Cv = pyo.Param(model.I_set, initialize={i: gen_data[i]['Cv'] for i in model.I_set})
    model.D = pyo.Param(model.T_set, initialize=demand)
    model.u_init = pyo.Param(model.I_set, initialize=u_initial)
    model.p_init = pyo.Param(model.I_set, initialize=p_initial)

    model.p = pyo.Var(model.I_set, model.T_set, within=pyo.NonNegativeReals)

    model.OBJ = pyo.Objective(rule=lambda m: sum(m.Cv[i] * m.p[i, t] for i in m.I_set for t in m.T_set), sense=pyo.minimize)

    model.p_prev = pyo.Expression(model.I_set, model.T_set, rule=lambda m, i, t: m.p_init[i] if t == 1 else m.p[i, t - 1])
    model.u_prev_fixed = pyo.Expression(model.I_set, model.T_set, rule=lambda m, i, t: m.u_init[i] if t == 1 else m.u_fixed[i, t - 1])

    model.MinPower = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t])
    model.MaxPower = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t])
    model.RampUp = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.p[i, t] - m.p_prev[i, t] <= m.Ru[i] * m.u_prev_fixed[i, t] + m.Rsu[i] * m.zON_fixed[i, t])
    model.RampDown = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.p_prev[i, t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t])
    model.Demand = pyo.Constraint(model.T_set, rule=lambda m, t: sum(m.p[i, t_val] for i in m.I_set for t_val in [t]) >= m.D[t])

    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return model

def build_master_fujitsu(iteration_data, num_beta_bits=BETA_BITS, num_slack_bits_per_cut_param=num_slack_bits): # Added num_slack_bits_per_cut_param
    u_shape = BitArrayShape('u', shape=(len(generators), len(time_periods)))
    zON_shape = BitArrayShape('zON', shape=(len(generators), len(time_periods)))
    zOFF_shape = BitArrayShape('zOFF', shape=(len(generators), len(time_periods)))
    beta_shape = BitArrayShape('beta', shape=(num_beta_bits,))

    # --- Dynamically determine the number of active cuts ---
    num_current_optimality_cuts = sum(1 for d in iteration_data if d['type'] == 'optimality')
    num_current_feasibility_cuts = sum(1 for d in iteration_data if d['type'] == 'feasibility')

    master_vss_elements = [u_shape, zON_shape, zOFF_shape, beta_shape]

    # --- Dynamically define slack variable shapes ---
    if num_current_optimality_cuts > 0:
        # Define s_opt shape based on the *actual* number of optimality cuts present
        slack_opt_shape = BitArrayShape('s_opt', shape=(num_current_optimality_cuts, num_slack_bits_per_cut_param))
        master_vss_elements.append(slack_opt_shape)
    
    if num_current_feasibility_cuts > 0:
        # Define s_feas shape based on the *actual* number of feasibility cuts present
        slack_feas_shape = BitArrayShape('s_feas', shape=(num_current_feasibility_cuts, num_slack_bits_per_cut_param))
        master_vss_elements.append(slack_feas_shape)
        
    # --- Create and freeze the VarShapeSet for THIS iteration ---
    master_vss = VarShapeSet(*master_vss_elements)
    BinPol.freeze_var_shape_set(master_vss) # This is crucial

    main_objective_qubo = BinPol()
    penalty_qubo_dadk = BinPol()

    lambda_logic1, lambda_logic2, lambda_opt_cut, lambda_feas_cut = lambda1, lambda2, lambda3, lambda4

    # --- Objective and Logic Penalties (remain the same) ---
    for i in generators:
        for t in time_periods:
            main_objective_qubo.add_term(gen_data[i]['Cf'], ('u', gen_map[i], time_map[t]))
            main_objective_qubo.add_term(gen_data[i]['Csu'], ('zON', gen_map[i], time_map[t]))
            main_objective_qubo.add_term(gen_data[i]['Csd'], ('zOFF', gen_map[i], time_map[t]))

    for j in range(num_beta_bits):
        main_objective_qubo.add_term(2**j, ('beta', j))

    for i in generators:
        for t in time_periods:
            logic1_expr = BinPol()
            logic1_expr.add_term(1, ('u', gen_map[i], time_map[t]))
            logic1_expr.add_term(-1, ('zON', gen_map[i], time_map[t]))
            logic1_expr.add_term(1, ('zOFF', gen_map[i], time_map[t]))
            if t == 1:
                logic1_expr.add_term(-u_initial[i])
            else:
                logic1_expr.add_term(-1, ('u', gen_map[i], time_map[t-1]))
            penalty_qubo_dadk.add(logic1_expr.power(2), scalar=lambda_logic1)

    for i in generators:
        for t in time_periods:
            penalty_qubo_dadk.add_term(2 * lambda_logic2, 
                                       ('zON', gen_map[i], time_map[t]), 
                                       ('zOFF', gen_map[i], time_map[t]))
    
    # --- Benders Cuts ---
    opt_cut_idx_counter = 0  # This will now index from 0 to num_current_optimality_cuts-1
    feas_cut_idx_counter = 0 # This will now index from 0 to num_current_feasibility_cuts-1

    for data in iteration_data: # Iterate through all historical cuts
        if data['type'] == 'optimality':
            # ... (logic for building lhs_opt_poly_scaled_rounded remains the same)
            sub_obj_k, duals_k, u_k_vals, zON_k_vals, zOFF_k_vals = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
            lhs_opt_poly_unscaled = BinPol()
            for j in range(num_beta_bits):
                lhs_opt_poly_unscaled.add_term(-1 * (2**j), ('beta', j))
            constant_part_opt = sub_obj_k
            for i in generators:
                for t in time_periods:
                    u_prev_k_val = u_k_vals.get((i, t - 1), u_initial[i]) if t > 1 else u_initial[i]
                    u_prev_var_symbol = ('u', gen_map[i], time_map[t-1]) if t > 1 else None
                    dual_val_min = duals_k['lambda_min'].get((i, t), 0.)
                    lhs_opt_poly_unscaled.add_term(dual_val_min * gen_data[i]['Pmin'], ('u', gen_map[i], time_map[t]))
                    constant_part_opt -= dual_val_min * gen_data[i]['Pmin'] * u_k_vals.get((i,t),0.)
                    dual_val_max = duals_k['lambda_max'].get((i, t), 0.)
                    lhs_opt_poly_unscaled.add_term(dual_val_max * gen_data[i]['Pmax'], ('u', gen_map[i], time_map[t]))
                    constant_part_opt -= dual_val_max * gen_data[i]['Pmax'] * u_k_vals.get((i,t),0.)
                    dual_val_ru = duals_k['lambda_ru'].get((i, t), 0.)
                    lhs_opt_poly_unscaled.add_term(dual_val_ru * gen_data[i]['Rsu'], ('zON', gen_map[i], time_map[t]))
                    constant_part_opt -= dual_val_ru * gen_data[i]['Rsu'] * zON_k_vals.get((i,t),0.)
                    if u_prev_var_symbol:
                        lhs_opt_poly_unscaled.add_term(dual_val_ru * gen_data[i]['Ru'], u_prev_var_symbol)
                        constant_part_opt -= dual_val_ru * gen_data[i]['Ru'] * u_prev_k_val
                    else:
                        constant_part_opt -= dual_val_ru * gen_data[i]['Ru'] * u_initial[i]
                    dual_val_rd = duals_k['lambda_rd'].get((i, t), 0.)
                    lhs_opt_poly_unscaled.add_term(dual_val_rd * gen_data[i]['Rd'], ('u', gen_map[i], time_map[t]))
                    lhs_opt_poly_unscaled.add_term(dual_val_rd * gen_data[i]['Rsd'], ('zOFF', gen_map[i], time_map[t]))
                    constant_part_opt -= dual_val_rd * gen_data[i]['Rd'] * u_k_vals.get((i,t),0.)
                    constant_part_opt -= dual_val_rd * gen_data[i]['Rsd'] * zOFF_k_vals.get((i,t),0.)
            lhs_opt_poly_unscaled.add_term(constant_part_opt)

            lhs_opt_poly_scaled_unrounded = lhs_opt_poly_unscaled.clone().multiply_scalar(SCALE_FACTOR)
            lhs_opt_poly_scaled_rounded = BinPol()
            for term_indices, coeff in lhs_opt_poly_scaled_unrounded.p.items():
                if not term_indices: lhs_opt_poly_scaled_rounded.add_term(int(round(coeff)))
                else: lhs_opt_poly_scaled_rounded.add_term(int(round(coeff)), *term_indices)

            slack_opt_poly = BinPol()
            # num_slack_bits_per_cut_param is used here
            for l_bit in range(num_slack_bits_per_cut_param):
                 slack_opt_poly.add_term((2**l_bit), ('s_opt', opt_cut_idx_counter, l_bit)) # opt_cut_idx_counter correctly indexes the current cut
            
            opt_cut_penalty_term = (lhs_opt_poly_scaled_rounded - slack_opt_poly).power(2)
            penalty_qubo_dadk.add(opt_cut_penalty_term, scalar=lambda_opt_cut)
            opt_cut_idx_counter += 1 # Increment for the next optimality cut

        elif data['type'] == 'feasibility':
            # ... (logic for building neg_lhs_feas_poly_scaled_rounded remains the same)
            rays_k = data['rays']
            lhs_feas_poly_unscaled = BinPol()
            constant_term_feas = 0.0
            for i in generators:
                for t_loop in time_periods:
                    u_coeff_val = (rays_k['min_power'].get((i, t_loop), 0.) * gen_data[i]['Pmin'] +
                                   rays_k['max_power'].get((i, t_loop), 0.) * gen_data[i]['Pmax'] +
                                   rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rd'])
                    if t_loop < Periods:
                        u_coeff_val += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * gen_data[i]['Ru']
                    lhs_feas_poly_unscaled.add_term(u_coeff_val, ('u', gen_map[i], time_map[t_loop]))
                    zON_coeff_val = rays_k['ramp_up'].get((i, t_loop), 0.) * gen_data[i]['Rsu']
                    lhs_feas_poly_unscaled.add_term(zON_coeff_val, ('zON', gen_map[i], time_map[t_loop]))
                    zOFF_coeff_val = rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rsd']
                    lhs_feas_poly_unscaled.add_term(zOFF_coeff_val, ('zOFF', gen_map[i], time_map[t_loop]))
            for t_loop in time_periods:
                constant_term_feas += rays_k['demand'].get(t_loop, 0.) * demand[t_loop]
                for i in generators:
                    if t_loop == 1:
                        constant_term_feas += rays_k['ramp_up'].get((i, 1), 0.) * gen_data[i]['Ru'] * u_initial[i]
            lhs_feas_poly_unscaled.add_term(constant_term_feas)

            neg_lhs_feas_poly_scaled_rounded = BinPol()
            temp_scaled = lhs_feas_poly_unscaled.clone().multiply_scalar(-SCALE_FACTOR)
            for term_indices, coeff in temp_scaled.p.items():
                if not term_indices: neg_lhs_feas_poly_scaled_rounded.add_term(int(round(coeff)))
                else: neg_lhs_feas_poly_scaled_rounded.add_term(int(round(coeff)), *term_indices)
            
            slack_feas_poly = BinPol()
            # num_slack_bits_per_cut_param is used here
            for l_bit in range(num_slack_bits_per_cut_param):
                 slack_feas_poly.add_term((2**l_bit), ('s_feas', feas_cut_idx_counter, l_bit)) # feas_cut_idx_counter correctly indexes the current cut
            
            feas_cut_penalty_term = (neg_lhs_feas_poly_scaled_rounded + slack_feas_poly).power(2)
            penalty_qubo_dadk.add(feas_cut_penalty_term, scalar=lambda_feas_cut)
            feas_cut_idx_counter += 1 # Increment for the next feasibility cut
            
    return main_objective_qubo, penalty_qubo_dadk, master_vss # Return the dynamically created master_vss


def build_relaxed_master_pyomo(iteration_data):
    model = pyo.ConcreteModel(name="Relaxed_Master_LP")
    model.I_set = pyo.Set(initialize=generators)
    model.T_set = pyo.Set(initialize=time_periods)
    model.Cf = pyo.Param(model.I_set, initialize={i: gen_data[i]['Cf'] for i in model.I_set})
    model.Csu = pyo.Param(model.I_set, initialize={i: gen_data[i]['Csu'] for i in model.I_set})
    model.Csd = pyo.Param(model.I_set, initialize={i: gen_data[i]['Csd'] for i in model.I_set})
    model.u_init = pyo.Param(model.I_set, initialize=u_initial)
    model.Pmin = pyo.Param(model.I_set, initialize={i: gen_data[i]['Pmin'] for i in model.I_set})
    model.Pmax = pyo.Param(model.I_set, initialize={i: gen_data[i]['Pmax'] for i in model.I_set})
    model.Rd = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rd'] for i in model.I_set})
    model.Rsd = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rsd'] for i in model.I_set})
    model.Ru = pyo.Param(model.I_set, initialize={i: gen_data[i]['Ru'] for i in model.I_set})
    model.Rsu = pyo.Param(model.I_set, initialize={i: gen_data[i]['Rsu'] for i in model.I_set})
    model.D_param = pyo.Param(model.T_set, initialize=demand)
    model.u = pyo.Var(model.I_set, model.T_set, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.zON = pyo.Var(model.I_set, model.T_set, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.zOFF = pyo.Var(model.I_set, model.T_set, within=pyo.NonNegativeReals, bounds=(0, 1))
    model.beta = pyo.Var(within=pyo.NonNegativeReals)
    def master_obj_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I_set for t in m.T_set)
        return commitment_cost + m.beta
    model.OBJ = pyo.Objective(rule=master_obj_rule, sense=pyo.minimize)
    model.u_prev = pyo.Expression(model.I_set, model.T_set, rule=lambda m, i, t: m.u_init[i] if t == 1 else m.u[i, t - 1])
    model.Logic1 = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.u[i, t] - m.u_prev[i, t] == m.zON[i, t] - m.zOFF[i, t])
    model.Logic2 = pyo.Constraint(model.I_set, model.T_set, rule=lambda m, i, t: m.zON[i, t] + m.zOFF[i, t] <= 1)
    model.OptimalityCuts = pyo.ConstraintList()
    model.FeasibilityCuts = pyo.ConstraintList()
    for data in iteration_data:
        if data['type'] == 'optimality':
            sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
            cut_rhs_expr = sub_obj_k
            for i in model.I_set:
                for t in model.T_set:
                    u_prev_k_val = u_k.get((i, t-1), model.u_init[i]) if t > 1 else model.u_init[i]
                    u_prev_var = model.u_init[i] if t == 1 else model.u[i, t-1]
                    cut_rhs_expr += duals_k['lambda_min'].get((i, t), 0.) * (model.Pmin[i] * (model.u[i,t] - u_k.get((i,t),0.)))
                    cut_rhs_expr += duals_k['lambda_max'].get((i, t), 0.) * (model.Pmax[i] * (model.u[i,t] - u_k.get((i,t),0.)))
                    cut_rhs_expr += duals_k['lambda_rd'].get((i, t), 0.) * (model.Rd[i] * (model.u[i,t] - u_k.get((i,t),0.)) + model.Rsd[i] * (model.zOFF[i,t] - zOFF_k.get((i,t),0.)))
                    cut_rhs_expr += duals_k['lambda_ru'].get((i, t), 0.) * (model.Ru[i] * (u_prev_var - u_prev_k_val) + model.Rsu[i] * (model.zON[i,t] - zON_k.get((i,t),0.)))
            model.OptimalityCuts.add(model.beta >= cut_rhs_expr)
        elif data['type'] == 'feasibility':
            rays_k = data['rays']
            cut_lhs_expr = 0
            for i in model.I_set:
                for t_loop in model.T_set:
                    u_coeff_val = (rays_k['min_power'].get((i, t_loop), 0.) * model.Pmin[i] +
                                   rays_k['max_power'].get((i, t_loop), 0.) * model.Pmax[i] +
                                   rays_k['ramp_down'].get((i, t_loop), 0.) * model.Rd[i])
                    if t_loop < Periods:
                        u_coeff_val += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * model.Ru[i]
                    cut_lhs_expr += u_coeff_val * model.u[i, t_loop]
                    cut_lhs_expr += (rays_k['ramp_up'].get((i, t_loop), 0.) * model.Rsu[i]) * model.zON[i, t_loop]
                    cut_lhs_expr += (rays_k['ramp_down'].get((i, t_loop), 0.) * model.Rsd[i]) * model.zOFF[i, t_loop]
            for t_loop in model.T_set:
                cut_lhs_expr += rays_k['demand'].get(t_loop, 0.) * model.D_param[t_loop]
                for i in model.I_set:
                    if t_loop == 1:
                         cut_lhs_expr += rays_k['ramp_up'].get((i, 1), 0.) * model.Ru[i] * model.u_init[i]
            model.FeasibilityCuts.add(cut_lhs_expr >= 0)
    return model

def get_symbolic_sample_from_dadk_solution(solution_obj, master_vss_local, gen_map_local, time_map_local, num_beta_bits_local):
    sample_dict = {}
    u_array = solution_obj.extract_bit_array('u').data
    for i_orig in generators:
        for t_orig in time_periods:
            sample_dict[f'u_{i_orig}_{t_orig}'] = u_array[gen_map_local[i_orig], time_map_local[t_orig]]

    zON_array = solution_obj.extract_bit_array('zON').data
    for i_orig in generators:
        for t_orig in time_periods:
            sample_dict[f'zON_{i_orig}_{t_orig}'] = zON_array[gen_map_local[i_orig], time_map_local[t_orig]]
    
    zOFF_array = solution_obj.extract_bit_array('zOFF').data
    for i_orig in generators:
        for t_orig in time_periods:
            sample_dict[f'zOFF_{i_orig}_{t_orig}'] = zOFF_array[gen_map_local[i_orig], time_map_local[t_orig]]

    beta_array = solution_obj.extract_bit_array('beta').data
    for j in range(num_beta_bits_local):
        sample_dict[f'beta_{j}'] = beta_array[j]
    
    if 's_opt' in master_vss_local.var_by_name:
        s_opt_array = solution_obj.extract_bit_array('s_opt').data
        # MODIFIED: Infer number of slack bits from array shape
        num_slack_bits_for_this_array = s_opt_array.shape[1] 
        for k in range(s_opt_array.shape[0]): # Iterate through cuts
            for l_s in range(num_slack_bits_for_this_array): # Iterate through slack bits for that cut
                 sample_dict[f's_opt_{k}_{l_s}'] = s_opt_array[k,l_s]
    
    if 's_feas' in master_vss_local.var_by_name:
        s_feas_array = solution_obj.extract_bit_array('s_feas').data
        # MODIFIED: Infer number of slack bits from array shape
        num_slack_bits_for_this_array = s_feas_array.shape[1]
        for k in range(s_feas_array.shape[0]):
            for l_s in range(num_slack_bits_for_this_array):
                 sample_dict[f's_feas_{k}_{l_s}'] = s_feas_array[k,l_s]
        
    return sample_dict

def check_logic1_feasibility_dadk(solution_obj, master_vss_local, gen_map_local, time_map_local):
    u_data = solution_obj.extract_bit_array('u').data
    zON_data = solution_obj.extract_bit_array('zON').data
    zOFF_data = solution_obj.extract_bit_array('zOFF').data
    for i_orig in generators:
        for t_orig in time_periods:
            u_it = u_data[gen_map_local[i_orig], time_map_local[t_orig]]
            zon_it = zON_data[gen_map_local[i_orig], time_map_local[t_orig]]
            zoff_it = zOFF_data[gen_map_local[i_orig], time_map_local[t_orig]]
            u_prev = u_initial[i_orig] if t_orig == 1 else u_data[gen_map_local[i_orig], time_map_local[t_orig-1]]
            if abs((u_it - u_prev) - (zon_it - zoff_it)) > 1e-6:
                return False
    return True

def check_logic2_feasibility_dadk(solution_obj, master_vss_local, gen_map_local, time_map_local):
    zON_data = solution_obj.extract_bit_array('zON').data
    zOFF_data = solution_obj.extract_bit_array('zOFF').data
    for i_orig in generators:
        for t_orig in time_periods:
            zon_it = zON_data[gen_map_local[i_orig], time_map_local[t_orig]]
            zoff_it = zOFF_data[gen_map_local[i_orig], time_map_local[t_orig]]
            if zon_it * zoff_it != 0: # This means zON and zOFF cannot both be 1.
                return False
    return True

# MODIFIED: Removed num_slack_bits and slack_step from parameters
def check_optimality_cuts_dadk_manual(solution_obj, master_vss_local, iteration_data_local, gen_map_local, time_map_local, num_beta_bits_local=BETA_BITS):
    u_data = solution_obj.extract_bit_array('u').data
    zON_data = solution_obj.extract_bit_array('zON').data
    zOFF_data = solution_obj.extract_bit_array('zOFF').data
    beta_data = solution_obj.extract_bit_array('beta').data
    s_opt_data = None
    if 's_opt' in master_vss_local.var_by_name:
        if 's_opt' in solution_obj.var_shape_set.var_by_name: 
            s_opt_data = solution_obj.extract_bit_array('s_opt').data
        
    beta_val_sample = sum((2**j) * beta_data[j] for j in range(num_beta_bits_local))
    
    opt_cut_idx_counter = 0
    for data_idx, data in enumerate(iteration_data_local):
        if data['type'] == 'optimality':
            sub_obj_k, duals_k, u_k_vals, zON_k_vals, zOFF_k_vals = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
            lhs_val = sub_obj_k - beta_val_sample 
            for i_orig in generators:
                for t_orig in time_periods:
                    u_it_sample = u_data[gen_map_local[i_orig], time_map_local[t_orig]]
                    zon_it_sample = zON_data[gen_map_local[i_orig], time_map_local[t_orig]]
                    zoff_it_sample = zOFF_data[gen_map_local[i_orig], time_map_local[t_orig]]
                    u_prev_sample = u_initial[i_orig] if t_orig == 1 else u_data[gen_map_local[i_orig], time_map_local[t_orig-1]]
                    u_prev_k_val = u_k_vals.get((i_orig, t_orig - 1), u_initial[i_orig]) if t_orig > 1 else u_initial[i_orig]
                    

                    
                    lhs_val += duals_k['lambda_min'].get((i_orig, t_orig), 0.) * gen_data[i_orig]['Pmin'] * (u_it_sample - u_k_vals.get((i_orig,t_orig),0.))
                    lhs_val += duals_k['lambda_max'].get((i_orig, t_orig), 0.) * gen_data[i_orig]['Pmax'] * (u_it_sample - u_k_vals.get((i_orig,t_orig),0.))
                    lhs_val += duals_k['lambda_ru'].get((i_orig, t_orig), 0.) * ( gen_data[i_orig]['Ru'] * (u_prev_sample - u_prev_k_val) + \
                                                                          gen_data[i_orig]['Rsu'] * (zon_it_sample - zON_k_vals.get((i_orig,t_orig),0.)) )
                    lhs_val += duals_k['lambda_rd'].get((i_orig, t_orig), 0.) * ( gen_data[i_orig]['Rd'] * (u_it_sample - u_k_vals.get((i_orig,t_orig),0.)) + \
                                                                          gen_data[i_orig]['Rsd'] * (zoff_it_sample - zOFF_k_vals.get((i_orig,t_orig),0.)) )
            
            slack_val_opt = 0
            if s_opt_data is not None and opt_cut_idx_counter < s_opt_data.shape[0] :
                num_bits_for_this_slack = s_opt_data.shape[1]
                # MODIFIED: slack_step is 1
                slack_val_opt = sum((2**l_bit) * s_opt_data[opt_cut_idx_counter, l_bit] for l_bit in range(num_bits_for_this_slack))
            

            current_lhs_opt_poly_unscaled_val = 0
            
            scaled_lhs_val_for_check = lhs_val * SCALE_FACTOR
            rounded_scaled_lhs_val_for_check = int(round(scaled_lhs_val_for_check))

            if abs(rounded_scaled_lhs_val_for_check - slack_val_opt) > 1e-3 : # Allow small tolerance for floating point comparison if not exactly integer
                # print(f"Debug Opt Cut {opt_cut_idx_counter}: RoundedLHS={rounded_scaled_lhs_val_for_check}, Slack={slack_val_opt}, Diff={abs(rounded_scaled_lhs_val_for_check - slack_val_opt)}")
                # print(f"  Unscaled LHS_val = {lhs_val}")
                return False
            opt_cut_idx_counter +=1
    return True

def check_feasibility_cuts_dadk_manual(solution_obj, master_vss_local, iteration_data_local, gen_map_local, time_map_local):
    u_data = solution_obj.extract_bit_array('u').data
    zON_data = solution_obj.extract_bit_array('zON').data
    zOFF_data = solution_obj.extract_bit_array('zOFF').data
    s_feas_data = None
    if 's_feas' in master_vss_local.var_by_name:
        if 's_feas' in solution_obj.var_shape_set.var_by_name:
            s_feas_data = solution_obj.extract_bit_array('s_feas').data

    feas_cut_idx_counter = 0
    for data_idx, data in enumerate(iteration_data_local):
        if data['type'] == 'feasibility':
            rays_k = data['rays']
            # This is lhs_feas_poly_unscaled evaluated with sample values
            lhs_feas_val_sample = 0.0 
            # Constant part from build_master_fujitsu
            constant_term_feas_bmf = 0.0
            for t_loop_orig_bmf in time_periods:
                constant_term_feas_bmf += rays_k['demand'].get(t_loop_orig_bmf, 0.) * demand[t_loop_orig_bmf]
                for i_orig_bmf in generators:
                    if t_loop_orig_bmf == 1:
                        constant_term_feas_bmf += rays_k['ramp_up'].get((i_orig_bmf, 1), 0.) * gen_data[i_orig_bmf]['Ru'] * u_initial[i_orig_bmf]
            lhs_feas_val_sample += constant_term_feas_bmf

            for i_orig in generators:
                for t_loop_orig in time_periods:
                    u_it_sample = u_data[gen_map_local[i_orig], time_map_local[t_loop_orig]]
                    zon_it_sample = zON_data[gen_map_local[i_orig], time_map_local[t_loop_orig]]
                    zoff_it_sample = zOFF_data[gen_map_local[i_orig], time_map_local[t_loop_orig]]
                    
                    u_coeff_val = (rays_k['min_power'].get((i_orig, t_loop_orig), 0.) * gen_data[i_orig]['Pmin'] +
                                   rays_k['max_power'].get((i_orig, t_loop_orig), 0.) * gen_data[i_orig]['Pmax'] +
                                   rays_k['ramp_down'].get((i_orig, t_loop_orig), 0.) * gen_data[i_orig]['Rd'])
                    if t_loop_orig < Periods: # Match logic in build_master_fujitsu
                        u_coeff_val += rays_k['ramp_up'].get((i_orig, t_loop_orig + 1), 0.) * gen_data[i_orig]['Ru']
                    
                    lhs_feas_val_sample += u_coeff_val * u_it_sample
                    lhs_feas_val_sample += (rays_k['ramp_up'].get((i_orig, t_loop_orig), 0.) * gen_data[i_orig]['Rsu']) * zon_it_sample
                    lhs_feas_val_sample += (rays_k['ramp_down'].get((i_orig, t_loop_orig), 0.) * gen_data[i_orig]['Rsd']) * zoff_it_sample
            
            slack_val_feas = 0
            if s_feas_data is not None and feas_cut_idx_counter < s_feas_data.shape[0]:
                num_bits_for_this_slack = s_feas_data.shape[1]
                # MODIFIED: slack_step is 1
                slack_val_feas = sum((2**l_bit) * s_feas_data[feas_cut_idx_counter, l_bit] for l_bit in range(num_bits_for_this_slack))
            
            # Penalty is (neg_lhs_feas_poly_scaled_rounded + slack_feas_poly).power(2)
            # So, round(-SCALE_FACTOR * lhs_feas_val_sample) + slack_val_feas should be 0
            
            neg_scaled_lhs_feas_val_sample = -SCALE_FACTOR * lhs_feas_val_sample
            rounded_neg_scaled_lhs_feas_val_sample = int(round(neg_scaled_lhs_feas_val_sample))
            
            if abs(rounded_neg_scaled_lhs_feas_val_sample + slack_val_feas) > 1e-3:
                # print(f"Debug Feas Cut {feas_cut_idx_counter}: RoundedNegLHS={rounded_neg_scaled_lhs_feas_val_sample}, Slack={slack_val_feas}, Sum={rounded_neg_scaled_lhs_feas_val_sample + slack_val_feas}")
                # print(f"  Unscaled LHS_feas_val_sample = {lhs_feas_val_sample}")
                return False
            feas_cut_idx_counter += 1
    return True


def main():
    start_time = time.time()
    max_iter, epsilon, num_beta_bits_main = 30, 1.0, BETA_BITS # Renamed num_beta_bits to avoid conflict
    iteration_data, lower_bound, upper_bound = [], -float('inf'), float('inf')
    stagnation_limit = 10
    stagnation_counter = 0
    best_ub_solution_dadk_master_inputs = None

    u_current, zON_current, zOFF_current = {}, {}, {}
    for t_orig in time_periods:
        for i_orig in generators:
            u_current[i_orig, t_orig] = 1.0
            u_prev_val = u_initial[i_orig] if t_orig == 1 else u_current.get((i_orig, t_orig-1), 1.0)
            if u_current[i_orig, t_orig] > 0.5 and u_prev_val < 0.5: 
                zON_current[i_orig, t_orig], zOFF_current[i_orig, t_orig] = 1.0, 0.0
            elif u_current[i_orig, t_orig] < 0.5 and u_prev_val > 0.5: 
                zON_current[i_orig, t_orig], zOFF_current[i_orig, t_orig] = 0.0, 1.0
            else: 
                zON_current[i_orig, t_orig], zOFF_current[i_orig, t_orig] = 0.0, 0.0
    
    gurobi_solver = SolverFactory("gurobi")
    sub_solver_persistent = SolverFactory("gurobi_persistent", solver_io='python')

    dadk_solver = QUBOSolverCPU(
        optimization_method='parallel_tempering',
        number_runs=num_runs, 
        number_replicas=num_replicas, 
        number_iterations=num_iter,
        temperature_sampling=True,
        scaling_action=ScalingAction.AUTO_SCALING,
        random_seed=42
    )

    print("--- Starting Hybrid Benders Decomposition for UCP ---")
    print(f"Master Solvers: DADK QUBOSolverCPU (Heuristic), Gurobi (Relaxed LP for LB)")
    print(f"Subproblem Solver: Gurobi")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")

    for k_iter_count in range(1, max_iter + 1):
        print(f"========================= Iteration {k_iter_count} =========================")
        print("--- 1. Solving Subproblem (LP) ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)
        sub_solver_persistent.set_instance(subproblem)
        # Ensure Gurobi parameters are set to get Farkas duals if infeasible
        sub_solver_persistent.set_gurobi_param('InfUnbdInfo', 1) 
        results = sub_solver_persistent.solve(tee=False)
        is_infeasible = results.solver.termination_condition == TerminationCondition.infeasible or \
                        results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded


        if not is_infeasible:
            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            commitment_cost = sum(gen_data[i]['Csu']*zON_current.get((i,t),0) + 
                                  gen_data[i]['Csd']*zOFF_current.get((i,t),0) + 
                                  gen_data[i]['Cf']*u_current.get((i,t),0) 
                                  for i in generators for t in time_periods)
            current_total_cost = commitment_cost + sub_obj_val
            if current_total_cost < upper_bound - 1e-6: # Added tolerance for UB update
                upper_bound = current_total_cost
                best_ub_solution_dadk_master_inputs = {
                    'u_vals': u_current.copy(),
                    'zON_vals': zON_current.copy(),
                    'zOFF_vals': zOFF_current.copy(),
                    'iter': k_iter_count,
                    'total_cost': upper_bound
                }
                print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f}")
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            duals = {
                'lambda_min': { (i,t): subproblem.dual.get(subproblem.MinPower[i,t], 0.0) for i in subproblem.I_set for t in subproblem.T_set},
                'lambda_max': { (i,t): subproblem.dual.get(subproblem.MaxPower[i,t], 0.0) for i in subproblem.I_set for t in subproblem.T_set},
                'lambda_ru':  { (i,t): subproblem.dual.get(subproblem.RampUp[i,t], 0.0) for i in subproblem.I_set for t in subproblem.T_set},
                'lambda_rd':  { (i,t): subproblem.dual.get(subproblem.RampDown[i,t], 0.0) for i in subproblem.I_set for t in subproblem.T_set},
                'lambda_dem': { t: subproblem.dual.get(subproblem.Demand[t], 0.0) for t in subproblem.T_set}
            }
            iteration_data.append({'type': 'optimality', 'sub_obj': sub_obj_val, 'duals': duals, 
                                   'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
        else:
            print("Subproblem Status: INFEASIBLE. Generating Feasibility Cut.")
            # For Gurobi Persistent, FarkasDual is an attribute of the constraint object itself after solving
            rays = {
                'min_power': {(i,t): subproblem.MinPower[i,t].FarkasDual if hasattr(subproblem.MinPower[i,t], 'FarkasDual') else 0.0 for i in subproblem.I_set for t in subproblem.T_set},
                'max_power': {(i,t): subproblem.MaxPower[i,t].FarkasDual if hasattr(subproblem.MaxPower[i,t], 'FarkasDual') else 0.0 for i in subproblem.I_set for t in subproblem.T_set},
                'ramp_up':   {(i,t): subproblem.RampUp[i,t].FarkasDual if hasattr(subproblem.RampUp[i,t], 'FarkasDual') else 0.0 for i in subproblem.I_set for t in subproblem.T_set},
                'ramp_down': {(i,t): subproblem.RampDown[i,t].FarkasDual if hasattr(subproblem.RampDown[i,t], 'FarkasDual') else 0.0 for i in subproblem.I_set for t in subproblem.T_set},
                'demand':    {t: subproblem.Demand[t].FarkasDual if hasattr(subproblem.Demand[t], 'FarkasDual') else 0.0 for t in subproblem.T_set}
            }
            iteration_data.append({'type': 'feasibility', 'rays': rays})

        print(f"Current Bounds -> LB: {lower_bound:.4f}, UB: {upper_bound:.4f}")
        if upper_bound - lower_bound <= epsilon: print("\nConvergence tolerance met."); break
        if k_iter_count == max_iter: print("\nMaximum iterations reached."); break
        if stagnation_counter >= stagnation_limit: print(f"\nTerminating: UB unchanged for {stagnation_limit} iterations."); break
        
        print("\n--- 2. Solving Master Problem (DADK QUBOSolverCPU) ---")
        main_qubo, penalty_qubo_all_terms, master_vss_instance = build_master_fujitsu(iteration_data, num_beta_bits_main)
        master_problem_qubo_for_dadk = main_qubo + penalty_qubo_all_terms 
        
        best_feasible_dadk_solution = None
        for attempt in range(num_tries): # Reduced to 1 for quicker test, can be increased
            solution_list_master = dadk_solver.minimize(qubo=master_problem_qubo_for_dadk)
            current_best_iter_energy = float('inf')
            current_best_iter_solution = None

            if not solution_list_master.solutions:
                print("  DADK solver returned no solutions on this attempt.")
                continue # Skip to next attempt if no solutions

            for sol_obj in solution_list_master.get_solution_list():
                # Check basic DADK feasibility first (are all variables present)
                try:
                    # Test extraction of a core variable. If this fails, sol_obj is likely problematic.
                    sol_obj.extract_bit_array('u') 
                except Exception as e:
                    # print(f"  Skipping a DADK solution object due to extraction error: {e}")
                    continue # Skip this malformed solution object

                logic1_ok = check_logic1_feasibility_dadk(sol_obj, master_vss_instance, gen_map, time_map)
                logic2_ok = check_logic2_feasibility_dadk(sol_obj, master_vss_instance, gen_map, time_map)
                opt_cuts_ok = check_optimality_cuts_dadk_manual(sol_obj, master_vss_instance, iteration_data, gen_map, time_map, num_beta_bits_local=num_beta_bits_main)
                feas_cuts_ok = check_feasibility_cuts_dadk_manual(sol_obj, master_vss_instance, iteration_data, gen_map, time_map)

                if logic1_ok and logic2_ok and opt_cuts_ok and feas_cuts_ok:
                    symbolic_sample = get_symbolic_sample_from_dadk_solution(sol_obj, master_vss_instance, gen_map, time_map, num_beta_bits_main)
                    master_objective_for_this_sample = 0
                    for i_orig_obj in generators:
                        for t_orig_obj in time_periods:
                            master_objective_for_this_sample += gen_data[i_orig_obj]['Cf'] * symbolic_sample.get(f'u_{i_orig_obj}_{t_orig_obj}',0)
                            master_objective_for_this_sample += gen_data[i_orig_obj]['Csu'] * symbolic_sample.get(f'zON_{i_orig_obj}_{t_orig_obj}',0)
                            master_objective_for_this_sample += gen_data[i_orig_obj]['Csd'] * symbolic_sample.get(f'zOFF_{i_orig_obj}_{t_orig_obj}',0)
                    for j_obj in range(num_beta_bits_main):
                         master_objective_for_this_sample += (2**j_obj) * symbolic_sample.get(f'beta_{j_obj}',0)
                    
                    if master_objective_for_this_sample < current_best_iter_energy:
                        current_best_iter_energy = master_objective_for_this_sample
                        current_best_iter_solution = sol_obj
            
            if current_best_iter_solution:
                best_feasible_dadk_solution = current_best_iter_solution
                print(f"Found a feasible integer solution from DADK on attempt {attempt + 1} with master obj: {current_best_iter_energy:.4f}.")
                break 
            else:
                print(f"No DADK solution satisfied all constraints on attempt {attempt+1}.")
                # Optionally, print details of a low-energy infeasible solution for debugging
                if solution_list_master.solutions and solution_list_master.min_solution:
                    debug_sol = solution_list_master.min_solution
                    print("  Debug: Min energy solution failed constraints. Checking its status:")
                    print(f"    Logic1: {check_logic1_feasibility_dadk(debug_sol, master_vss_instance, gen_map, time_map)}")
                    print(f"    Logic2: {check_logic2_feasibility_dadk(debug_sol, master_vss_instance, gen_map, time_map)}")
                    print(f"    OptCuts: {check_optimality_cuts_dadk_manual(debug_sol, master_vss_instance, iteration_data, gen_map, time_map, num_beta_bits_local=num_beta_bits_main)}")
                    print(f"    FeasCuts: {check_feasibility_cuts_dadk_manual(debug_sol, master_vss_instance, iteration_data, gen_map, time_map)}")


        if best_feasible_dadk_solution is None: print("CRITICAL: Master Problem (DADK) FAILED to find a feasible solution after attempts. Stopping."); break
        
        u_data_best = best_feasible_dadk_solution.extract_bit_array('u').data
        zON_data_best = best_feasible_dadk_solution.extract_bit_array('zON').data
        zOFF_data_best = best_feasible_dadk_solution.extract_bit_array('zOFF').data
        for i_orig in generators:
            for t_orig in time_periods:
                u_current[i_orig, t_orig] = u_data_best[gen_map[i_orig], time_map[t_orig]]
                zON_current[i_orig, t_orig] = zON_data_best[gen_map[i_orig], time_map[t_orig]]
                zOFF_current[i_orig, t_orig] = zOFF_data_best[gen_map[i_orig], time_map[t_orig]]
        
        print(f"\n--- 3. Solving Relaxed Master (LP for Lower Bound) ---")
        relaxed_master_model = build_relaxed_master_pyomo(iteration_data)
        master_results_lp = gurobi_solver.solve(relaxed_master_model, tee=False)
        if master_results_lp.solver.termination_condition == TerminationCondition.optimal:
            master_obj_val_lp = pyo.value(relaxed_master_model.OBJ)
            lower_bound = max(lower_bound, master_obj_val_lp)
            print(f"Relaxed Master Solved. New LB candidate: {master_obj_val_lp:.4f}. Updated Z_LB: {lower_bound:.4f}")
            print(f"The beta value from the relaxed master is: {pyo.value(relaxed_master_model.beta):.4f}")
        else: print(f"CRITICAL: Relaxed Master LP FAILED! Status: {master_results_lp.solver.termination_condition}. Stopping."); break
            
    end_time = time.time()
    
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) / (abs(upper_bound) + 1e-9) if upper_bound != float('inf') and abs(upper_bound) > 1e-9 else float('inf')
    print(f"Final Gap: {final_gap:.6f}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")

    if best_ub_solution_dadk_master_inputs:
        print(f"\n--- Best Feasible Solution Found (corresponds to Z_UB from iteration {best_ub_solution_dadk_master_inputs['iter']}) ---")
        print(f"Best Total Cost (Upper Bound): {best_ub_solution_dadk_master_inputs['total_cost']:.4f}")
        print("Commitment Schedule (u_it):")
        for t_val in time_periods:
            print(f"  t={t_val}: ", {i: round(best_ub_solution_dadk_master_inputs['u_vals'].get((i, t_val), 0)) for i in generators})

        final_subproblem = build_subproblem(
            best_ub_solution_dadk_master_inputs['u_vals'],
            best_ub_solution_dadk_master_inputs['zON_vals'],
            best_ub_solution_dadk_master_inputs['zOFF_vals']
        )
        final_solver = SolverFactory("gurobi")
        final_res = final_solver.solve(final_subproblem)

        if final_res.solver.termination_condition == TerminationCondition.optimal:
            var_cost = pyo.value(final_subproblem.OBJ)
            com_cost = sum(gen_data[i]['Csu'] * best_ub_solution_dadk_master_inputs['zON_vals'].get((i,t),0) +
                           gen_data[i]['Csd'] * best_ub_solution_dadk_master_inputs['zOFF_vals'].get((i,t),0) +
                           gen_data[i]['Cf']  * best_ub_solution_dadk_master_inputs['u_vals'].get((i,t),0)
                           for i in generators for t in time_periods)

            print("\nFinal Dispatch (p_it):")
            for t_orig_final in time_periods: 
                print(f"  t={t_orig_final}: ", {i_orig_final: f"{pyo.value(final_subproblem.p[i_orig_final, t_orig_final]):.2f}" for i_orig_final in generators})


            print("\nCost recap:")
            print(f"  Commitment cost : {com_cost:.4f}")
            print(f"  Variable cost   : {var_cost:.4f}")
            print(f"  Total cost      : {com_cost + var_cost:.4f}") # Should match best_ub_solution_dadk_master_inputs['total_cost']

            print("\nDemand check:")
            for t_orig_final in time_periods: 
                total_prod = sum(pyo.value(final_subproblem.p[i_orig_final, t_orig_final]) for i_orig_final in generators)
                print(f"  t={t_orig_final}: produced={total_prod:.2f}  demand={demand[t_orig_final]}"
                      f"  met={total_prod >= demand[t_orig_final] - 1e-4}")
        else:
            print("WARNING: Could not re-solve dispatch for the final schedule.")
    else:
        print("\nNo feasible solution was found during the process.")
        

if __name__ == '__main__':
    main()