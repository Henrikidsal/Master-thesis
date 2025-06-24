##### This is a script that solves the UCP using a hybrid Benders Decomposition.
##### The subproblem is a Pyomo/Gurobi LP.
##### The master problem is solved using DADK's QUBOSolverCPU and Pyomo/Gurobi for the relaxed LP.

# Necessary imports
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import time
import numpy as np
from datetime import datetime
import pandas as pd
import os
import gc 

# DADK imports
from dadk.BinPol import *
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction

def main(time_periods, iterations):
    num_tries = 10
    num_runs = 1
    num_replicas = 128
    num_iter = iterations
    BETA_BITS = 7
    num_slack_bits = 8
    SCALE_FACTOR = 1
    lambda1 = 7000
    lambda2 = 1500
    lambda3 = 50
    lambda4 = 10

    lower_bounds = []
    upper_bounds = []

    # Choose the number of time periods wanted:
    Periods = time_periods
    # Sets
    generators = [1, 2, 3]  # 1-indexed
    time_periods_set = [x for x in range(1, Periods + 1)]  # 1-indexed, T=3 hours

    # Generator Parameters from the PDF
    gen_data = {
        1: {'Pmin': 50, 'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
        2: {'Pmin': 80, 'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
        3: {'Pmin': 40, 'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5, 'Csd': 1.0, 'Cv': 0.150}
    }

    if Periods == 3:
        demand = {1: 160, 2: 500, 3: 400}  # 3
        BETA_BITS = 7
    elif Periods == 5:
        demand = {1: 160, 2: 500, 3: 400, 4: 160, 5: 500}
        BETA_BITS = 8
    elif Periods == 6:
        demand = {1: 160, 2: 500, 3: 400, 4: 160, 5: 500, 6: 400}
        BETA_BITS = 8
    else:
        raise ValueError("Invalid number of time periods. Choose between 3 and 7.")

    # Initial Conditions for time period = 0, from the PDF
    u_initial = {1: 0, 2: 0, 3: 1}
    p_initial = {1: 0, 2: 0, 3: 100}

    # Global mapping for DADK symbolic names
    gen_map = {orig_g: idx for idx, orig_g in enumerate(generators)}
    time_map = {orig_t: idx for idx, orig_t in enumerate(time_periods_set)}

    # --- Benders Decomposition Functions ---
    def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
        model = pyo.ConcreteModel(name="Sub_Problem")
        model.I_set = pyo.Set(initialize=generators)
        model.T_set = pyo.Set(initialize=time_periods_set)
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

    def build_master_fujitsu(iteration_data, num_beta_bits=BETA_BITS, num_slack_bits_per_cut_param=num_slack_bits):
        u_shape = BitArrayShape('u', shape=(len(generators), len(time_periods_set)))
        zON_shape = BitArrayShape('zON', shape=(len(generators), len(time_periods_set)))
        zOFF_shape = BitArrayShape('zOFF', shape=(len(generators), len(time_periods_set)))
        beta_shape = BitArrayShape('beta', shape=(num_beta_bits,))
        num_current_optimality_cuts = sum(1 for d in iteration_data if d['type'] == 'optimality')
        num_current_feasibility_cuts = sum(1 for d in iteration_data if d['type'] == 'feasibility')
        master_vss_elements = [u_shape, zON_shape, zOFF_shape, beta_shape]
        if num_current_optimality_cuts > 0:
            slack_opt_shape = BitArrayShape('s_opt', shape=(num_current_optimality_cuts, num_slack_bits_per_cut_param))
            master_vss_elements.append(slack_opt_shape)
        if num_current_feasibility_cuts > 0:
            slack_feas_shape = BitArrayShape('s_feas', shape=(num_current_feasibility_cuts, num_slack_bits_per_cut_param))
            master_vss_elements.append(slack_feas_shape)
        master_vss = VarShapeSet(*master_vss_elements)
        BinPol.freeze_var_shape_set(master_vss)
        main_objective_qubo = BinPol()
        penalty_qubo_dadk = BinPol()
        lambda_logic1, lambda_logic2, lambda_opt_cut, lambda_feas_cut = lambda1, lambda2, lambda3, lambda4
        for i in generators:
            for t in time_periods_set:
                main_objective_qubo.add_term(gen_data[i]['Cf'], ('u', gen_map[i], time_map[t]))
                main_objective_qubo.add_term(gen_data[i]['Csu'], ('zON', gen_map[i], time_map[t]))
                main_objective_qubo.add_term(gen_data[i]['Csd'], ('zOFF', gen_map[i], time_map[t]))
        for j in range(num_beta_bits):
            main_objective_qubo.add_term(2**j, ('beta', j))
        for i in generators:
            for t in time_periods_set:
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
            for t in time_periods_set:
                penalty_qubo_dadk.add_term(2 * lambda_logic2, ('zON', gen_map[i], time_map[t]), ('zOFF', gen_map[i], time_map[t]))
        opt_cut_idx_counter = 0
        feas_cut_idx_counter = 0
        max_slack_val_possible = (2**num_slack_bits_per_cut_param) - 1
        for data_idx, data in enumerate(iteration_data):
            if data['type'] == 'optimality':
                sub_obj_k, duals_k, u_k_vals, zON_k_vals, zOFF_k_vals = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
                lhs_opt_poly_unscaled = BinPol()
                for j in range(num_beta_bits):
                    lhs_opt_poly_unscaled.add_term(-1 * (2**j), ('beta', j))
                constant_part_opt = sub_obj_k
                for i in generators:
                    for t in time_periods_set:
                        u_it_k_val = u_k_vals.get((i,t),0.)
                        zON_it_k_val = zON_k_vals.get((i,t),0.)
                        zOFF_it_k_val = zOFF_k_vals.get((i,t),0.)
                        u_prev_k_val = u_k_vals.get((i, t - 1), u_initial[i]) if t > 1 else u_initial[i]
                        dual_val_min = duals_k['lambda_min'].get((i, t), 0.)
                        if abs(dual_val_min) > 1e-9:
                            lhs_opt_poly_unscaled.add_term(dual_val_min * gen_data[i]['Pmin'], ('u', gen_map[i], time_map[t]))
                            constant_part_opt -= dual_val_min * gen_data[i]['Pmin'] * u_it_k_val
                        dual_val_max = duals_k['lambda_max'].get((i, t), 0.)
                        if abs(dual_val_max) > 1e-9:
                            lhs_opt_poly_unscaled.add_term(dual_val_max * gen_data[i]['Pmax'], ('u', gen_map[i], time_map[t]))
                            constant_part_opt -= dual_val_max * gen_data[i]['Pmax'] * u_it_k_val
                        dual_val_ru = duals_k['lambda_ru'].get((i, t), 0.)
                        if abs(dual_val_ru) > 1e-9:
                            lhs_opt_poly_unscaled.add_term(dual_val_ru * gen_data[i]['Rsu'], ('zON', gen_map[i], time_map[t]))
                            constant_part_opt -= dual_val_ru * gen_data[i]['Rsu'] * zON_it_k_val
                            if t > 1:
                                lhs_opt_poly_unscaled.add_term(dual_val_ru * gen_data[i]['Ru'], ('u', gen_map[i], time_map[t-1]))
                                constant_part_opt -= dual_val_ru * gen_data[i]['Ru'] * u_prev_k_val
                            else:
                                constant_part_opt += dual_val_ru * gen_data[i]['Ru'] * u_initial[i]
                                constant_part_opt -= dual_val_ru * gen_data[i]['Ru'] * u_prev_k_val
                        dual_val_rd = duals_k['lambda_rd'].get((i, t), 0.)
                        if abs(dual_val_rd) > 1e-9:
                            lhs_opt_poly_unscaled.add_term(dual_val_rd * gen_data[i]['Rd'], ('u', gen_map[i], time_map[t]))
                            lhs_opt_poly_unscaled.add_term(dual_val_rd * gen_data[i]['Rsd'], ('zOFF', gen_map[i], time_map[t]))
                            constant_part_opt -= dual_val_rd * gen_data[i]['Rd'] * u_it_k_val
                            constant_part_opt -= dual_val_rd * gen_data[i]['Rsd'] * zOFF_it_k_val
                lhs_opt_poly_unscaled.add_term(constant_part_opt)
                lhs_opt_poly_scaled_unrounded = lhs_opt_poly_unscaled.clone().multiply_scalar(SCALE_FACTOR)
                lhs_opt_poly_scaled_rounded = BinPol()
                for term_indices, coeff in lhs_opt_poly_scaled_unrounded.p.items():
                    if not term_indices: lhs_opt_poly_scaled_rounded.add_term(int(round(coeff)))
                    else: lhs_opt_poly_scaled_rounded.add_term(int(round(coeff)), *term_indices)
                const_term_scaled_opt = lhs_opt_poly_scaled_rounded.p.get((), 0.0)
                sum_abs_var_coeffs_scaled_opt = sum(abs(c) for ti, c in lhs_opt_poly_scaled_rounded.p.items() if ti)
                max_abs_val_to_represent_opt = abs(const_term_scaled_opt) + sum_abs_var_coeffs_scaled_opt
                print(f"      Opt Cut {opt_cut_idx_counter}: Est. max val for slack bits: {max_abs_val_to_represent_opt:.1f}. Slack capacity: {max_slack_val_possible}. Scaled Constant part: {const_term_scaled_opt:.1f}")
                slack_opt_poly = BinPol()
                for l_bit in range(num_slack_bits_per_cut_param):
                    slack_opt_poly.add_term((2**l_bit), ('s_opt', opt_cut_idx_counter, l_bit))
                opt_cut_penalty_term = (lhs_opt_poly_scaled_rounded + slack_opt_poly).power(2)
                penalty_qubo_dadk.add(opt_cut_penalty_term, scalar=lambda_opt_cut)
                opt_cut_idx_counter += 1
            elif data['type'] == 'feasibility':
                rays_k = data['rays']
                lhs_feas_poly_unscaled = BinPol()
                constant_term_feas = 0.0
                for i in generators:
                    for t_loop in time_periods_set:
                        u_coeff_val = (rays_k['min_power'].get((i, t_loop), 0.) * gen_data[i]['Pmin'] +
                                       rays_k['max_power'].get((i, t_loop), 0.) * gen_data[i]['Pmax'] +
                                       rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rd'])
                        if t_loop < Periods:
                            u_coeff_val += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * gen_data[i]['Ru']
                        if abs(u_coeff_val) > 1e-9:
                            lhs_feas_poly_unscaled.add_term(u_coeff_val, ('u', gen_map[i], time_map[t_loop]))
                        zON_coeff_val = rays_k['ramp_up'].get((i, t_loop), 0.) * gen_data[i]['Rsu']
                        if abs(zON_coeff_val) > 1e-9:
                            lhs_feas_poly_unscaled.add_term(zON_coeff_val, ('zON', gen_map[i], time_map[t_loop]))
                        zOFF_coeff_val = rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rsd']
                        if abs(zOFF_coeff_val) > 1e-9:
                            lhs_feas_poly_unscaled.add_term(zOFF_coeff_val, ('zOFF', gen_map[i], time_map[t_loop]))
                for t_loop in time_periods_set:
                    constant_term_feas += rays_k['demand'].get(t_loop, 0.) * demand[t_loop]
                    for i in generators:
                        if t_loop == 1:
                            constant_term_feas += rays_k['ramp_up'].get((i, 1), 0.) * gen_data[i]['Ru'] * u_initial[i]
                lhs_feas_poly_unscaled.add_term(constant_term_feas)
                temp_scaled_positive = lhs_feas_poly_unscaled.clone().multiply_scalar(SCALE_FACTOR)
                neg_lhs_feas_poly_scaled_unrounded = lhs_feas_poly_unscaled.clone().multiply_scalar(-SCALE_FACTOR)
                neg_lhs_feas_poly_scaled_rounded = BinPol()
                for term_indices, coeff in neg_lhs_feas_poly_scaled_unrounded.p.items():
                    if not term_indices: neg_lhs_feas_poly_scaled_rounded.add_term(int(round(coeff)))
                    else: neg_lhs_feas_poly_scaled_rounded.add_term(int(round(coeff)), *term_indices)
                temp_scaled_positive_rounded_const = int(round(temp_scaled_positive.p.get((), 0.0)))
                temp_scaled_positive_rounded_sum_abs_vars = sum(abs(int(round(c))) for ti, c in temp_scaled_positive.p.items() if ti)
                max_abs_val_to_represent_feas = abs(temp_scaled_positive_rounded_const) + temp_scaled_positive_rounded_sum_abs_vars
                slack_feas_poly = BinPol()
                for l_bit in range(num_slack_bits_per_cut_param):
                    slack_feas_poly.add_term((2**l_bit), ('s_feas', feas_cut_idx_counter, l_bit))
                feas_cut_penalty_term = (neg_lhs_feas_poly_scaled_rounded + slack_feas_poly).power(2)
                penalty_qubo_dadk.add(feas_cut_penalty_term, scalar=lambda_feas_cut)
                feas_cut_idx_counter += 1
        return main_objective_qubo, penalty_qubo_dadk, master_vss

    def build_relaxed_master_pyomo(iteration_data):
        model = pyo.ConcreteModel(name="Relaxed_Master_LP")
        model.I_set = pyo.Set(initialize=generators)
        model.T_set = pyo.Set(initialize=time_periods_set)
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
            for t_orig in time_periods_set:
                sample_dict[f'u_{i_orig}_{t_orig}'] = u_array[gen_map_local[i_orig], time_map_local[t_orig]]
        zON_array = solution_obj.extract_bit_array('zON').data
        for i_orig in generators:
            for t_orig in time_periods_set:
                sample_dict[f'zON_{i_orig}_{t_orig}'] = zON_array[gen_map_local[i_orig], time_map_local[t_orig]]
        zOFF_array = solution_obj.extract_bit_array('zOFF').data
        for i_orig in generators:
            for t_orig in time_periods_set:
                sample_dict[f'zOFF_{i_orig}_{t_orig}'] = zOFF_array[gen_map_local[i_orig], time_map_local[t_orig]]
        beta_array = solution_obj.extract_bit_array('beta').data
        for j in range(num_beta_bits_local):
            sample_dict[f'beta_{j}'] = beta_array[j]
        if 's_opt' in master_vss_local.var_by_name:
            s_opt_array = solution_obj.extract_bit_array('s_opt').data
            num_slack_bits_for_this_array = s_opt_array.shape[1]
            for k in range(s_opt_array.shape[0]):
                for l_s in range(num_slack_bits_for_this_array):
                    sample_dict[f's_opt_{k}_{l_s}'] = s_opt_array[k,l_s]
        if 's_feas' in master_vss_local.var_by_name:
            s_feas_array = solution_obj.extract_bit_array('s_feas').data
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
            for t_orig in time_periods_set:
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
            for t_orig in time_periods_set:
                zon_it = zON_data[gen_map_local[i_orig], time_map_local[t_orig]]
                zoff_it = zOFF_data[gen_map_local[i_orig], time_map_local[t_orig]]
                if zon_it * zoff_it != 0:
                    return False
        return True

    def check_optimality_cuts_dadk_type1(solution_obj, master_vss_local, iteration_data_local, gen_map_local, time_map_local, num_beta_bits_local=BETA_BITS, tolerance=1e-6):
        u_data = solution_obj.extract_bit_array('u').data
        zON_data = solution_obj.extract_bit_array('zON').data
        zOFF_data = solution_obj.extract_bit_array('zOFF').data
        beta_data = solution_obj.extract_bit_array('beta').data
        beta_val_solution = sum((2**j) * beta_data[j] for j in range(num_beta_bits_local))
        for data_idx, data in enumerate(iteration_data_local):
            if data['type'] == 'optimality':
                sub_obj_k, duals_k, u_k_vals, zON_k_vals, zOFF_k_vals = \
                    data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
                cut_rhs_val_float = float(sub_obj_k)
                for i_orig in generators:
                    for t_orig in time_periods_set:
                        u_it_solution = float(u_data[gen_map_local[i_orig], time_map_local[t_orig]])
                        zon_it_solution = float(zON_data[gen_map_local[i_orig], time_map_local[t_orig]])
                        zoff_it_solution = float(zOFF_data[gen_map_local[i_orig], time_map_local[t_orig]])
                        u_prev_solution = float(u_initial[i_orig]) if t_orig == 1 else \
                                           float(u_data[gen_map_local[i_orig], time_map_local[t_orig-1]])
                        u_it_k = float(u_k_vals.get((i_orig,t_orig),0.))
                        zon_it_k = float(zON_k_vals.get((i_orig,t_orig),0.))
                        zoff_it_k = float(zOFF_k_vals.get((i_orig,t_orig),0.))
                        u_prev_k = float(u_k_vals.get((i_orig, t_orig - 1), u_initial[i_orig])) if t_orig > 1 else \
                                   float(u_initial[i_orig])
                        cut_rhs_val_float += float(duals_k['lambda_min'].get((i_orig, t_orig), 0.)) * \
                                             float(gen_data[i_orig]['Pmin']) * (u_it_solution - u_it_k)
                        cut_rhs_val_float += float(duals_k['lambda_max'].get((i_orig, t_orig), 0.)) * \
                                             float(gen_data[i_orig]['Pmax']) * (u_it_solution - u_it_k)
                        dual_ru_val = float(duals_k['lambda_ru'].get((i_orig, t_orig), 0.))
                        cut_rhs_val_float += dual_ru_val * ( \
                            float(gen_data[i_orig]['Ru']) * (u_prev_solution - u_prev_k) + \
                            float(gen_data[i_orig]['Rsu']) * (zon_it_solution - zon_it_k) )
                        dual_rd_val = float(duals_k['lambda_rd'].get((i_orig, t_orig), 0.))
                        cut_rhs_val_float += dual_rd_val * ( \
                            float(gen_data[i_orig]['Rd']) * (u_it_solution - u_it_k) + \
                            float(gen_data[i_orig]['Rsd']) * (zoff_it_solution - zoff_it_k) )
                if not (beta_val_solution >= cut_rhs_val_float - tolerance):
                    return False
        return True

    def check_feasibility_cuts_dadk_type1(solution_obj, master_vss_local, iteration_data_local, gen_map_local, time_map_local, tolerance=1e-6):
        u_data = solution_obj.extract_bit_array('u').data
        zON_data = solution_obj.extract_bit_array('zON').data
        zOFF_data = solution_obj.extract_bit_array('zOFF').data
        for data_idx, data in enumerate(iteration_data_local):
            if data['type'] == 'feasibility':
                rays_k = data['rays']
                lhs_feas_val_float = 0.0
                for t_loop_orig in time_periods_set:
                    lhs_feas_val_float += float(rays_k['demand'].get(t_loop_orig, 0.)) * float(demand[t_loop_orig])
                    for i_orig in generators:
                        if t_loop_orig == 1:
                            lhs_feas_val_float += float(rays_k['ramp_up'].get((i_orig, 1), 0.)) * \
                                                  float(gen_data[i_orig]['Ru']) * float(u_initial[i_orig])
                for i_orig in generators:
                    for t_loop_orig in time_periods_set:
                        u_it_solution = float(u_data[gen_map_local[i_orig], time_map_local[t_loop_orig]])
                        zon_it_solution = float(zON_data[gen_map_local[i_orig], time_map_local[t_loop_orig]])
                        zoff_it_solution = float(zOFF_data[gen_map_local[i_orig], time_map_local[t_loop_orig]])
                        u_coeff_val = (float(rays_k['min_power'].get((i_orig, t_loop_orig), 0.)) * float(gen_data[i_orig]['Pmin']) +
                                       float(rays_k['max_power'].get((i_orig, t_loop_orig), 0.)) * float(gen_data[i_orig]['Pmax']) +
                                       float(rays_k['ramp_down'].get((i_orig, t_loop_orig), 0.)) * float(gen_data[i_orig]['Rd']))
                        if t_loop_orig < Periods:
                            u_coeff_val += float(rays_k['ramp_up'].get((i_orig, t_loop_orig + 1), 0.)) * float(gen_data[i_orig]['Ru'])
                        lhs_feas_val_float += u_coeff_val * u_it_solution
                        lhs_feas_val_float += (float(rays_k['ramp_up'].get((i_orig, t_loop_orig), 0.)) * \
                                               float(gen_data[i_orig]['Rsu'])) * zon_it_solution
                        lhs_feas_val_float += (float(rays_k['ramp_down'].get((i_orig, t_loop_orig), 0.)) * \
                                               float(gen_data[i_orig]['Rsd'])) * zoff_it_solution
                if not (lhs_feas_val_float >= -tolerance):
                    return False
        return True

    start_time = time.time()
    max_iter, epsilon, num_beta_bits_main = 30, 1.0, BETA_BITS
    iteration_data, lower_bound, upper_bound = [], -float('inf'), float('inf')
    stagnation_limit = 10
    stagnation_counter = 0
    best_ub_solution_dadk_master_inputs = None
    u_current, zON_current, zOFF_current = {}, {}, {}
    for t_orig in time_periods_set:
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
        sub_solver_persistent.set_gurobi_param('InfUnbdInfo', 1)
        sub_solver_persistent.set_gurobi_param('DualReductions', 0)
        results = sub_solver_persistent.solve(tee=False)
        is_infeasible = results.solver.termination_condition == TerminationCondition.infeasible or \
                        results.solver.termination_condition == TerminationCondition.infeasibleOrUnbounded
        if not is_infeasible:
            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            commitment_cost = sum(gen_data[i]['Csu']*zON_current.get((i,t),0) +
                                  gen_data[i]['Csd']*zOFF_current.get((i,t),0) +
                                  gen_data[i]['Cf']*u_current.get((i,t),0)
                                  for i in generators for t in time_periods_set)
            current_total_cost = commitment_cost + sub_obj_val
            if current_total_cost < upper_bound - 1e-6:
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
            rays = {'min_power': {}, 'max_power': {}, 'ramp_up': {}, 'ramp_down': {}, 'demand': {}}
            non_zero_rays_found = False
            total_rays_extracted = 0
            problematic_ray_extraction = False
            try:
                for (i_ray, t_ray_idx), constraint_object in subproblem.MinPower.items():
                    ray_val = sub_solver_persistent.get_linear_constraint_attr(constraint_object, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    rays['min_power'][(i_ray, t_ray_idx)] = ray_val
                    if abs(ray_val) > 1e-9: non_zero_rays_found = True
                for (i_ray, t_ray_idx), constraint_object in subproblem.MaxPower.items():
                    ray_val = sub_solver_persistent.get_linear_constraint_attr(constraint_object, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    rays['max_power'][(i_ray, t_ray_idx)] = ray_val
                    if abs(ray_val) > 1e-9: non_zero_rays_found = True
                for (i_ray, t_ray_idx), constraint_object in subproblem.RampUp.items():
                    ray_val = sub_solver_persistent.get_linear_constraint_attr(constraint_object, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    rays['ramp_up'][(i_ray, t_ray_idx)] = ray_val
                    if abs(ray_val) > 1e-9: non_zero_rays_found = True
                for (i_ray, t_ray_idx), constraint_object in subproblem.RampDown.items():
                    ray_val = sub_solver_persistent.get_linear_constraint_attr(constraint_object, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    rays['ramp_down'][(i_ray, t_ray_idx)] = ray_val
                    if abs(ray_val) > 1e-9: non_zero_rays_found = True
                for t_ray_idx, constraint_object in subproblem.Demand.items():
                    ray_val = sub_solver_persistent.get_linear_constraint_attr(constraint_object, 'FarkasDual')
                    if ray_val is None: ray_val = 0.0
                    rays['demand'][t_ray_idx] = ray_val
                    if abs(ray_val) > 1e-9: non_zero_rays_found = True
            except AttributeError as ae:
                print(f"AttributeError during FarkasDual extraction: {ae}. This might mean a constraint doesn't exist for an index or FarkasDual attribute is missing unexpectedly.")
                problematic_ray_extraction = True
            except Exception as e:
                print(f"General ERROR during FarkasDual extraction: {e}")
                problematic_ray_extraction = True
            print("\n--- Extracted Farkas Duals (Rays) for Feasibility Cut ---")
            if problematic_ray_extraction:
                print("  Extraction failed or was incomplete due to an error.")
            else:
                for key, ray_dict_content in rays.items():
                    print(f"  {key}:")
                    if not ray_dict_content:
                        print("    <empty_or_all_zero>")
                    has_nonzero_in_dict = any(abs(V_content) > 1e-9 for V_content in ray_dict_content.values())
                    if has_nonzero_in_dict:
                        for K_content, V_content in ray_dict_content.items():
                            if abs(V_content) > 1e-9:
                                print(f"        {K_content}: {V_content:.4f}")
                    elif ray_dict_content:
                        print("    (all zero)")
                print(f"  Summary: Attempted to extract {total_rays_extracted} potential ray components.")
                if non_zero_rays_found:
                    print("  Status: At least one non-zero FarkasDual component was found overall.")
                else:
                    print("  WARNING: All extracted FarkasDual components are zero (or very close to zero). Feasibility cut will be ineffective.")
            print("-----------------------------------------------------------\n")
            if not problematic_ray_extraction and non_zero_rays_found:
                iteration_data.append({'type': 'feasibility', 'rays': rays,
                                       'u_vals': u_current.copy(),
                                       'zON_vals': zON_current.copy(),
                                       'zOFF_vals': zOFF_current.copy()})
            else:
                print("Skipping feasibility cut addition due to issues in ray extraction or all rays being zero.")

        termination_reason = None
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)
        print(f"Current Bounds -> LB: {lower_bound:.4f}, UB: {upper_bound:.4f}")
        if upper_bound - lower_bound <= epsilon:
            print("\nConvergence tolerance met.")
            termination_reason = "Convergence Tolerance Met"
            break
        if k_iter_count == max_iter:
            print("\nMaximum iterations reached.")
            termination_reason = "Max Iterations Reached"
            break
        if stagnation_counter >= stagnation_limit:
            print(f"\nTerminating: UB unchanged for {stagnation_limit} iterations.")
            termination_reason = "Stagnation Limit Reached"
            break
        if Periods == 3 and upper_bound == 191.8:
            print("\nTerminating: Known optimal solution found for 3-period case.")
            termination_reason = "Known Optimal Solution Found for 3-Period Case"
            break
        if Periods == 5 and upper_bound == 293.5:
            print("\nTerminating: Known optimal solution found for 5-period case.")
            termination_reason = "Known Optimal Solution Found for 5-Period Case"
            break
        if Periods == 6 and upper_bound == 347.5:
            print("\nTerminating: Known optimal solution found for 6-period case.")
            termination_reason = "Known Optimal Solution Found for 6-Period Case"
            break
        print("\n--- 2. Solving Master Problem (DADK QUBOSolverCPU) ---")
        main_qubo, penalty_qubo_all_terms, master_vss_instance = build_master_fujitsu(iteration_data, num_beta_bits_main)
        master_problem_qubo_for_dadk = main_qubo + penalty_qubo_all_terms
        best_feasible_dadk_solution = None
        for attempt in range(num_tries):
            solution_list_master = dadk_solver.minimize(qubo=master_problem_qubo_for_dadk)
            current_best_iter_energy = float('inf')
            current_best_iter_solution = None
            if not solution_list_master.solutions:
                print(f"  DADK solver returned no solutions on attempt {attempt + 1}.")
                continue
            for sol_obj in solution_list_master.get_solution_list():
                try:
                    sol_obj.extract_bit_array('u')
                except Exception as e:
                    continue
                logic1_ok = check_logic1_feasibility_dadk(sol_obj, master_vss_instance, gen_map, time_map)
                logic2_ok = check_logic2_feasibility_dadk(sol_obj, master_vss_instance, gen_map, time_map)
                opt_cuts_ok = check_optimality_cuts_dadk_type1(sol_obj, master_vss_instance, iteration_data, gen_map, time_map, num_beta_bits_local=num_beta_bits_main)
                feas_cuts_ok = check_feasibility_cuts_dadk_type1(sol_obj, master_vss_instance, iteration_data, gen_map, time_map)
                if logic1_ok and logic2_ok and opt_cuts_ok and feas_cuts_ok:
                    symbolic_sample = get_symbolic_sample_from_dadk_solution(sol_obj, master_vss_instance, gen_map, time_map, num_beta_bits_main)
                    master_objective_for_this_sample = 0
                    for i_orig_obj in generators:
                        for t_orig_obj in time_periods_set:
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
                if solution_list_master.solutions:
                    print("  Detailed constraint violation analysis for this attempt's DADK solutions:")
                    num_solutions_analyzed = 0
                    logic1_violations = 0
                    logic2_violations = 0
                    opt_cuts_violations = 0
                    feas_cuts_violations = 0
                    solutions_to_analyze = solution_list_master.get_solution_list()
                    total_solutions_in_list = len(solutions_to_analyze)
                    for sol_idx, debug_sol in enumerate(solutions_to_analyze):
                        try:
                            debug_sol.extract_bit_array('u')
                        except Exception as e:
                            continue
                        num_solutions_analyzed += 1
                        if not check_logic1_feasibility_dadk(debug_sol, master_vss_instance, gen_map, time_map):
                            logic1_violations += 1
                        if not check_logic2_feasibility_dadk(debug_sol, master_vss_instance, gen_map, time_map):
                            logic2_violations += 1
                        if not check_optimality_cuts_dadk_type1(debug_sol, master_vss_instance, iteration_data, gen_map, time_map, num_beta_bits_local=num_beta_bits_main):
                            opt_cuts_violations += 1
                        if not check_feasibility_cuts_dadk_type1(debug_sol, master_vss_instance, iteration_data, gen_map, time_map):
                            feas_cuts_violations += 1
                    if num_solutions_analyzed > 0:
                        print(f"    Analyzed {num_solutions_analyzed} DADK solutions (out of {total_solutions_in_list} returned by solver):")
                        print(f"      Logic1 violations:          {logic1_violations:>5} ({logic1_violations/num_solutions_analyzed*100:6.1f}%)")
                        print(f"      Logic2 violations:          {logic2_violations:>5} ({logic2_violations/num_solutions_analyzed*100:6.1f}%)")
                        print(f"      Optimality Cuts violations: {opt_cuts_violations:>5} ({opt_cuts_violations/num_solutions_analyzed*100:6.1f}%)")
                        print(f"      Feasibility Cuts violations:{feas_cuts_violations:>5} ({feas_cuts_violations/num_solutions_analyzed*100:6.1f}%)")
                    else:
                        print("    No DADK solutions could be meaningfully analyzed from this attempt.")
                else:
                    print("    No DADK solutions were available to analyze for this attempt (e.g. solver returned empty list).")
        if best_feasible_dadk_solution is None:
            print("CRITICAL: Master Problem (DADK) FAILED to find a feasible solution after all attempts. Stopping.")
            termination_reason = "Master Problem DADK Failed"
            break
        if best_feasible_dadk_solution:
            symbolic_sol_dadk = get_symbolic_sample_from_dadk_solution(best_feasible_dadk_solution, master_vss_instance, gen_map, time_map, num_beta_bits_main)
            commitment_cost_from_dadk_vars = 0
            for i_orig_cc in generators:
                for t_orig_cc in time_periods_set:
                    commitment_cost_from_dadk_vars += gen_data[i_orig_cc]['Cf'] * symbolic_sol_dadk.get(f'u_{i_orig_cc}_{t_orig_cc}',0)
                    commitment_cost_from_dadk_vars += gen_data[i_orig_cc]['Csu'] * symbolic_sol_dadk.get(f'zON_{i_orig_cc}_{t_orig_cc}',0)
                    commitment_cost_from_dadk_vars += gen_data[i_orig_cc]['Csd'] * symbolic_sol_dadk.get(f'zOFF_{i_orig_cc}_{t_orig_cc}',0)
            beta_val_from_dadk_bits = sum((2**j) * symbolic_sol_dadk.get(f'beta_{j}',0) for j in range(num_beta_bits_main))
            print(f"    DADK Master Sol Info: DADK Reported Obj = {current_best_iter_energy:.4f}")
            print(f"    DADK Master Sol Info: Commitment Cost from DADK u,zON,zOFF = {commitment_cost_from_dadk_vars:.4f}")
            print(f"    DADK Master Sol Info: Beta value from DADK bits = {beta_val_from_dadk_bits:.4f}")
            print(f"    DADK Master Sol Info: Sum (Commitment + Beta DADK) = {commitment_cost_from_dadk_vars + beta_val_from_dadk_bits:.4f}")
        u_data_best = best_feasible_dadk_solution.extract_bit_array('u').data
        zON_data_best = best_feasible_dadk_solution.extract_bit_array('zON').data
        zOFF_data_best = best_feasible_dadk_solution.extract_bit_array('zOFF').data
        for i_orig in generators:
            for t_orig in time_periods_set:
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
        else:
            print(f"CRITICAL: Relaxed Master LP FAILED! Status: {master_results_lp.solver.termination_condition}. Stopping.")
            termination_reason = "Relaxed Master LP Failed"
            break

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
        for t_val in time_periods_set:
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
                           for i in generators for t in time_periods_set)
            print("\nFinal Dispatch (p_it):")
            for t_orig_final in time_periods_set:
                print(f"  t={t_orig_final}: ", {i_orig_final: f"{pyo.value(final_subproblem.p[i_orig_final, t_orig_final]):.2f}" for i_orig_final in generators})
            print("final turning ON schedule (zON_it):")
            for t_orig_final in time_periods_set:
                print(f"  t={t_orig_final}: ", {i_orig_final: round(best_ub_solution_dadk_master_inputs['zON_vals'].get((i_orig_final, t_orig_final), 0)) for i_orig_final in generators})
            print("final turning OFF schedule (zOFF_it):")
            for t_orig_final in time_periods_set:
                print(f"  t={t_orig_final}: ", {i_orig_final: round(best_ub_solution_dadk_master_inputs['zOFF_vals'].get((i_orig_final, t_orig_final), 0)) for i_orig_final in generators})
            print("\nCost recap:")
            print(f"  Commitment cost : {com_cost:.4f}")
            print(f"  Variable cost   : {var_cost:.4f}")
            print(f"  Total cost      : {com_cost + var_cost:.4f}")
            print("\nDemand check:")
            for t_orig_final in time_periods_set:
                total_prod = sum(pyo.value(final_subproblem.p[i_orig_final, t_orig_final]) for i_orig_final in generators)
                print(f"  t={t_orig_final}: produced={total_prod:.2f}  demand={demand[t_orig_final]}"
                      f"  met={total_prod >= demand[t_orig_final] - 1e-4}")
        else:
            print("WARNING: Could not re-solve dispatch for the final schedule.")
    else:
        print("\nNo feasible solution was found during the process.")

    result_data = {
        "number_of_time_periods": len(time_periods_set),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "best_solution": best_ub_solution_dadk_master_inputs['total_cost'] if best_ub_solution_dadk_master_inputs else None,
        "time_used": end_time - start_time,
        "number_of_benders_iterations": k_iter_count,
        "number_of_runs": num_iter,
        "commitment_schedule": {t: {i: round(best_ub_solution_dadk_master_inputs['u_vals'].get((i, t), 0)) for i in generators} for t in time_periods_set} if best_ub_solution_dadk_master_inputs else None,
        "turning_on_schedule": {t: {i: round(best_ub_solution_dadk_master_inputs['zON_vals'].get((i, t), 0)) for i in generators} for t in time_periods_set} if best_ub_solution_dadk_master_inputs else None,
        "turning_off_schedule": {t: {i: round(best_ub_solution_dadk_master_inputs['zOFF_vals'].get((i, t), 0)) for i in generators} for t in time_periods_set} if best_ub_solution_dadk_master_inputs else None,
        "termination_reason": termination_reason
    }

    # Explicitly delete large objects to help free up memory
    del iteration_data
    del lower_bounds
    del upper_bounds
    del subproblem
    del dadk_solver
    if 'relaxed_master_model' in locals():
        del relaxed_master_model
    if 'master_problem_qubo_for_dadk' in locals():
        del master_problem_qubo_for_dadk
    if 'final_subproblem' in locals():
        del final_subproblem

    return result_data

def append_to_csv(result_dict, csv_filepath):
    """Appends a result dictionary as a new row to a CSV file."""
    file_exists = os.path.exists(csv_filepath)
    df_to_append = pd.DataFrame([result_dict])
    df_to_append.to_csv(csv_filepath, mode='a', header=not file_exists, index=False)


if __name__ == '__main__':
    csv_file = "fujitsu_qubo_results.csv"

    # Remove the results file at the very start of the experiment
    if os.path.exists(csv_file):
        os.remove(csv_file)

    run_configurations = [
        (50, 2_000, 3),
        (50, 5_000, 3),
        (50, 5_000, 5),
        (50, 5_000, 6)
    ]

    # Loop through each configuration
    for total_runs, iterations, periods in run_configurations:
        print(f"\n--- Starting New Batch: {total_runs} runs with {iterations} iterations for {periods} time periods ---")
        for i in range(total_runs):
            print(f"\n--- Running Experiment {i+1}/{total_runs} for this batch ---")
            
            # Run the main function and get the result
            result = main(iterations=iterations, time_periods=periods)

            # Append the result directly to the CSV file
            if result:
                append_to_csv(result, csv_file)

            # Manually trigger the garbage collector to free up memory
            gc.collect()

    print("\nAll experiments have been completed.")