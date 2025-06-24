##### This is a script that solves the UCP using a hybrid Benders Decomposition.
##### The subproblem is a Pyomo/Gurobi LP.
##### The master problem is solved in two ways:
##### 1. As a QUBO with D-Wave's SimulatedAnnealingSampler to find a feasible integer solution.
##### 2. As a relaxed LP with Pyomo/Gurobi to find a valid lower bound.

# Neccecary imports
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition
import time
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np
from dwave.system import LeapHybridBQMSampler
from dwave.system import DWaveSampler
import joblib
from datetime import datetime

import pandas as pd
import os


#For solving 3 time periods, the following parameters are used:
#Beta bits = 7, Periods = 3, demand = {1: 160, 2: 500, 3: 400}, lambda1 = 110, lambda2 = 40, lambda3 = 10, lambda4 = 50
#Often finds optimal solution, 191.8

#For solving 6 time periods, the following parameters are used:
#Beta bits = 8, Periods = 6, demand = {1: 160, 2: 500, 3: 400, 4: 160, 5: 500, 6: 400}, lambda1 = 110, lambda2 = 40, lambda3 = 10, lambda4 = 50
#Used num_reads = 3000, which is high
#Often finds optimal solution, 




def main(iterations, time_periods):
    global df
        
    num_iter = iterations
    # --- Global Parameters ---
    save_results_to_file=False
    
    lambda1, lambda2, lambda3, lambda4 = 1800, 900, 100, 500 # Lagrange multipliers for the master problem
    # Choose the number of time periods wanted:
    Periods = time_periods
    # Sets
    generators = [1, 2, 3]
    time_periods = [x for x in range(1, Periods + 1)] # T=3 hours

    # Beta bits for the master problem
    if Periods == 3:
        BETA_BITS = 7
        demand = {1: 160, 2: 500, 3: 400} #3 #beta 7

    elif Periods == 5:
        BETA_BITS = 8
        demand = {1: 160, 2: 500, 3: 400, 4: 160, 5: 500} #5 #beta 8

    elif Periods == 6:
        BETA_BITS = 8
        demand = {1: 160, 2: 500, 3: 400, 4: 160, 5: 500, 6: 400} #6 #beta 8

    else:
        raise ValueError("Unsupported number of time periods. Supported: 3, 5, or 6.")



    # Generator Parameters from the PDF
    gen_data = {
        1: {'Pmin': 50,  'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
        2: {'Pmin': 80,  'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
        3: {'Pmin': 40,  'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5,  'Csd': 1.0, 'Cv': 0.150}
    }


    # Initial Conditions for time period = 0, from the PDF
    u_initial = {1: 0, 2: 0, 3: 1}
    p_initial = {1: 0, 2: 0, 3: 100}

    # --- Benders Decomposition Functions ---

    def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
        model = pyo.ConcreteModel(name="Sub_Problem")
        model.I = pyo.Set(initialize=generators)
        model.T = pyo.Set(initialize=time_periods)
        u_fixed_param_vals = {(i, t): u_fixed_vals.get((i, t), 0.0) for i in model.I for t in model.T}
        zON_fixed_param_vals = {(i, t): zON_fixed_vals.get((i, t), 0.0) for i in model.I for t in model.T}
        zOFF_fixed_param_vals = {(i, t): zOFF_fixed_vals.get((i, t), 0.0) for i in model.I for t in model.T}
        model.u_fixed = pyo.Param(model.I, model.T, initialize=u_fixed_param_vals)
        model.zON_fixed = pyo.Param(model.I, model.T, initialize=zON_fixed_param_vals)
        model.zOFF_fixed = pyo.Param(model.I, model.T, initialize=zOFF_fixed_param_vals)
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
        model.OBJ = pyo.Objective(rule=lambda m: sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T), sense=pyo.minimize)
        model.p_prev = pyo.Expression(model.I, model.T, rule=lambda m, i, t: m.p_init[i] if t == 1 else m.p[i, t - 1])
        model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=lambda m, i, t: m.u_init[i] if t == 1 else m.u_fixed[i, t - 1])
        model.MinPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t])
        model.MaxPower = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t])
        model.RampUp = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p[i, t] - m.p_prev[i, t] <= m.Ru[i] * m.u_prev_fixed[i, t] + m.Rsu[i] * m.zON_fixed[i, t])
        model.RampDown = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.p_prev[i, t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t])
        model.Demand = pyo.Constraint(model.T, rule=lambda m, t: sum(m.p[i, t] for i in m.I) >= m.D[t])
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
        return model

    def build_master_dwave(iteration_data, num_beta_bits=BETA_BITS):
        bqm = dimod.BinaryQuadraticModel('BINARY')
        lambda_logic1, lambda_logic2, lambda_opt_cut, lambda_feas_cut = lambda1, lambda2, lambda3, lambda4
        SCALE_FACTOR = 1

        u_vars = {(i, t): f'u_{i}_{t}' for i in generators for t in time_periods}
        zON_vars = {(i, t): f'zON_{i}_{t}' for i in generators for t in time_periods}
        zOFF_vars = {(i, t): f'zOFF_{i}_{t}' for i in generators for t in time_periods}
        beta_vars = {j: f'beta_{j}' for j in range(num_beta_bits)}

        for i in generators:
            for t in time_periods:
                bqm.add_linear(u_vars[i, t], gen_data[i]['Cf'])
                bqm.add_linear(zON_vars[i, t], gen_data[i]['Csu'])
                bqm.add_linear(zOFF_vars[i, t], gen_data[i]['Csd'])

        for j in range(num_beta_bits):
            bqm.add_linear(beta_vars[j], 2**j)

        for i in generators:
            for t in time_periods:
                terms = [(u_vars[i, t], 1), (zON_vars[i, t], -1), (zOFF_vars[i, t], 1)]
                constant = -u_initial[i] if t == 1 else 0
                if t > 1:
                    terms.append((u_vars[i, t-1], -1))
                bqm.add_linear_equality_constraint(terms, lagrange_multiplier=lambda_logic1, constant=constant)

        for i in generators:
            for t in time_periods:
                bqm.add_quadratic(zON_vars[i,t], zOFF_vars[i,t], 2 * lambda_logic2)

        for k_idx, data in enumerate(iteration_data):
            if data['type'] == 'optimality':
                sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
                
                cut_terms = [] # Stores (variable_name, coefficient)
                cut_constant = sub_obj_k # Start with sub_obj_k for the constant part of (sub_obj + dual*(x-x_k) - beta)

                # -beta term
                for j in range(num_beta_bits):
                    cut_terms.append((beta_vars[j], -1 * (2**j))) 

                # dual^T * x terms and -dual^T * x_k (added to cut_constant)
                for i in generators:
                    for t in time_periods:
                        u_prev_k = u_k.get((i, t - 1), u_initial[i]) if t > 1 else u_initial[i]
                        u_prev_var_name = u_vars.get((i, t-1)) if t > 1 else None

                        # MinPower: dual * Pmin * u
                        dual_val = duals_k['lambda_min'].get((i, t), 0.)
                        cut_terms.append((u_vars[i,t], dual_val * gen_data[i]['Pmin']))
                        cut_constant -= dual_val * gen_data[i]['Pmin'] * u_k.get((i,t),0.)

                        # MaxPower: dual * Pmax * u
                        dual_val = duals_k['lambda_max'].get((i, t), 0.)
                        cut_terms.append((u_vars[i,t], dual_val * gen_data[i]['Pmax']))
                        cut_constant -= dual_val * gen_data[i]['Pmax'] * u_k.get((i,t),0.)
                        
                        # RampUp: dual * (Ru*u_prev + Rsu*zON)
                        dual_val = duals_k['lambda_ru'].get((i, t), 0.)
                        cut_terms.append((zON_vars[i,t], dual_val * gen_data[i]['Rsu']))
                        cut_constant -= dual_val * gen_data[i]['Rsu'] * zON_k.get((i,t),0.)
                        if u_prev_var_name: # if t > 1
                            cut_terms.append((u_prev_var_name, dual_val * gen_data[i]['Ru']))
                        else: # t == 1, u_prev is u_initial[i], constant contribution to the fixed part
                            cut_constant -= dual_val * gen_data[i]['Ru'] * u_initial[i] 
                        cut_constant -= dual_val * gen_data[i]['Ru'] * u_prev_k # This is -dual*Ru*u_prev_k
                                            
                        # RampDown: dual * (Rd*u + Rsd*zOFF)
                        dual_val = duals_k['lambda_rd'].get((i, t), 0.)
                        cut_terms.append((u_vars[i,t], dual_val * gen_data[i]['Rd']))
                        cut_terms.append((zOFF_vars[i,t], dual_val * gen_data[i]['Rsd']))
                        cut_constant -= dual_val * gen_data[i]['Rd'] * u_k.get((i,t),0.)
                        cut_constant -= dual_val * gen_data[i]['Rsd'] * zOFF_k.get((i,t),0.)

                # Constraint: sub_obj_k + sum(dual*(x-x_k)) - beta <= 0
                scaled_terms = [(var, int(round(c * SCALE_FACTOR))) for var, c in cut_terms]
                scaled_constant = int(round(cut_constant * SCALE_FACTOR))
                
                bqm.add_linear_inequality_constraint(
                    scaled_terms,
                    lagrange_multiplier=lambda_opt_cut,
                    label=f"opt_cut_{k_idx}",
                    constant=scaled_constant, # This is (sub_obj_k - sum(dual*x_k))
                    ub=0 
                )

            elif data['type'] == 'feasibility':
                # ** REVERTING TO USER'S ORIGINAL FEASIBILITY CUT LOGIC **
                rays_k = data['rays']
                terms, constant_term_original = [], 0 # Renamed to avoid conflict with loop variable
                
                for i in generators:
                    for t_loop in time_periods: # Renamed t to t_loop to avoid conflict
                        u_coeff = (rays_k['min_power'].get((i, t_loop), 0.) * gen_data[i]['Pmin'] +
                                rays_k['max_power'].get((i, t_loop), 0.) * gen_data[i]['Pmax'] +
                                rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rd'])
                        if t_loop < Periods:
                            u_coeff += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * gen_data[i]['Ru']
                        terms.append((u_vars[i, t_loop], u_coeff))

                        zON_coeff = rays_k['ramp_up'].get((i, t_loop), 0.) * gen_data[i]['Rsu']
                        terms.append((zON_vars[i, t_loop], zON_coeff))

                        zOFF_coeff = rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rsd']
                        terms.append((zOFF_vars[i, t_loop], zOFF_coeff))
                
                for t_loop in time_periods: # Renamed t to t_loop
                    constant_term_original += rays_k['demand'].get(t_loop, 0.) * demand[t_loop]
                    for i in generators:
                        if t_loop == 1:
                            constant_term_original += rays_k['ramp_up'].get((i, 1), 0.) * gen_data[i]['Ru'] * u_initial[i]

                # Original constraint: sum(terms) + constant >= 0  (lb=0)
                scaled_terms = [(var, int(round(c * SCALE_FACTOR))) for var, c in terms]
                scaled_constant = int(round(constant_term_original * SCALE_FACTOR))
                
                bqm.add_linear_inequality_constraint(
                    scaled_terms, 
                    lagrange_multiplier=lambda_feas_cut, 
                    label=f"feas_cut_{k_idx}", 
                    constant=scaled_constant, 
                    lb=0,  # Lower bound for ">=" constraint
                    ub=np.iinfo(np.int32).max
                )
                
        return bqm

    def build_relaxed_master_pyomo(iteration_data):
        model = pyo.ConcreteModel(name="Relaxed_Master_LP")
        model.I = pyo.Set(initialize=generators)
        model.T = pyo.Set(initialize=time_periods)
        model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
        model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
        model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
        model.u_init = pyo.Param(model.I, initialize=u_initial)
        model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
        model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
        model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})
        model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})
        model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})
        model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})
        model.D_param = pyo.Param(model.T, initialize=demand) # Renamed to avoid conflict
        
        
        model.u = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.zON = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
        model.zOFF = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals, bounds=(0, 1))
        '''
        model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
        model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
        model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
        '''
        model.beta = pyo.Var(within=pyo.NonNegativeReals)
        def master_obj_rule(m):
            commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
            return commitment_cost + m.beta
            
        model.OBJ = pyo.Objective(rule=master_obj_rule, sense=pyo.minimize)
        model.u_prev = pyo.Expression(model.I, model.T, rule=lambda m, i, t: m.u_init[i] if t == 1 else m.u[i, t - 1])
        model.Logic1 = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.u[i, t] - m.u_prev[i, t] == m.zON[i, t] - m.zOFF[i, t])
        model.Logic2 = pyo.Constraint(model.I, model.T, rule=lambda m, i, t: m.zON[i, t] + m.zOFF[i, t] <= 1)
        model.OptimalityCuts = pyo.ConstraintList()
        model.FeasibilityCuts = pyo.ConstraintList()

        for data in iteration_data:
            if data['type'] == 'optimality':
                sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
                cut_rhs_expr = sub_obj_k
                for i in model.I:
                    for t in model.T:
                        u_prev_k = u_k.get((i, t-1), model.u_init[i]) if t > 1 else model.u_init[i]
                        u_prev_var = model.u_init[i] if t == 1 else model.u[i, t-1]
                        cut_rhs_expr += duals_k['lambda_min'].get((i, t), 0.) * (model.Pmin[i] * (model.u[i,t] - u_k.get((i,t),0.)))
                        cut_rhs_expr += duals_k['lambda_max'].get((i, t), 0.) * (model.Pmax[i] * (model.u[i,t] - u_k.get((i,t),0.)))
                        cut_rhs_expr += duals_k['lambda_rd'].get((i, t), 0.) * (model.Rd[i] * (model.u[i,t] - u_k.get((i,t),0.)) + model.Rsd[i] * (model.zOFF[i,t] - zOFF_k.get((i,t),0.)))
                        cut_rhs_expr += duals_k['lambda_ru'].get((i, t), 0.) * (model.Ru[i] * (u_prev_var - u_prev_k) + model.Rsu[i] * (model.zON[i,t] - zON_k.get((i,t),0.)))
                model.OptimalityCuts.add(model.beta >= cut_rhs_expr)
            
            elif data['type'] == 'feasibility':

                rays_k = data['rays']
                cut_lhs_expr = 0
                for i in model.I:
                    for t_loop in model.T:
                        u_coeff_val = (rays_k['min_power'].get((i, t_loop), 0.) * model.Pmin[i] +
                                    rays_k['max_power'].get((i, t_loop), 0.) * model.Pmax[i] +
                                    rays_k['ramp_down'].get((i, t_loop), 0.) * model.Rd[i])
                        if t_loop < Periods:
                            u_coeff_val += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * model.Ru[i]
                        cut_lhs_expr += u_coeff_val * model.u[i, t_loop]

                        cut_lhs_expr += (rays_k['ramp_up'].get((i, t_loop), 0.) * model.Rsu[i]) * model.zON[i, t_loop]
                        cut_lhs_expr += (rays_k['ramp_down'].get((i, t_loop), 0.) * model.Rsd[i]) * model.zOFF[i, t_loop]

                for t_loop in model.T:
                    cut_lhs_expr += rays_k['demand'].get(t_loop, 0.) * model.D_param[t_loop]
                    for i in model.I:
                        if t_loop == 1:
                            cut_lhs_expr += rays_k['ramp_up'].get((i, 1), 0.) * model.Ru[i] * model.u_init[i]
                
                model.FeasibilityCuts.add(cut_lhs_expr >= 0) # Original was sum(terms) + constant >= 0

        return model

    def check_logic1_feasibility(sample):
        for i in generators:
            for t in time_periods:
                u_it = sample.get(f'u_{i}_{t}', 0)
                zon_it = sample.get(f'zON_{i}_{t}', 0)
                zoff_it = sample.get(f'zOFF_{i}_{t}', 0)
                u_prev = u_initial[i] if t == 1 else sample.get(f'u_{i}_{t-1}', 0)
                if abs((u_it - u_prev) - (zon_it - zoff_it)) > 1e-6:
                    return False
        return True

    def check_logic2_feasibility(sample):
        for i in generators:
            for t in time_periods:
                zon_it = sample.get(f'zON_{i}_{t}', 0)
                zoff_it = sample.get(f'zOFF_{i}_{t}', 0)
                if zon_it * zoff_it != 0: # Check product is 0
                    return False
        return True

    def check_optimality_cuts(sample, iteration_data, num_beta_bits=BETA_BITS):

        
        for data in iteration_data:
            if data['type'] == 'optimality':
                sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
                beta_val = sum((2**j) * sample.get(f'beta_{j}', 0) for j in range(num_beta_bits))
                
                cut_rhs_val = sub_obj_k # This is the sub_obj part
                for i in generators:
                    for t in time_periods:
                        u_it_sample = sample.get(f'u_{i}_{t}', 0)
                        zon_it_sample = sample.get(f'zON_{i}_{t}', 0)
                        zoff_it_sample = sample.get(f'zOFF_{i}_{t}', 0)
                        u_prev_sample = u_initial[i] if t == 1 else sample.get(f'u_{i}_{t-1}', 0)
                        u_prev_k_val = u_k.get((i, t - 1), u_initial[i]) if t > 1 else u_initial[i]

                        # MinPower: dual * Pmin * (u_sample - u_k)
                        cut_rhs_val += duals_k['lambda_min'].get((i, t), 0.) * gen_data[i]['Pmin'] * (u_it_sample - u_k.get((i,t),0.))
                        # MaxPower: dual * Pmax * (u_sample - u_k)
                        cut_rhs_val += duals_k['lambda_max'].get((i, t), 0.) * gen_data[i]['Pmax'] * (u_it_sample - u_k.get((i,t),0.))
                        # RampUp: dual * (Ru*(u_prev_sample - u_prev_k) + Rsu*(zON_sample - zON_k))
                        cut_rhs_val += duals_k['lambda_ru'].get((i, t), 0.) * ( gen_data[i]['Ru'] * (u_prev_sample - u_prev_k_val) + \
                                                                            gen_data[i]['Rsu'] * (zon_it_sample - zON_k.get((i,t),0.)) )
                        # RampDown: dual * (Rd*(u_sample - u_k) + Rsd*(zOFF_sample - zOFF_k))
                        cut_rhs_val += duals_k['lambda_rd'].get((i, t), 0.) * ( gen_data[i]['Rd'] * (u_it_sample - u_k.get((i,t),0.)) + \
                                                                            gen_data[i]['Rsd'] * (zoff_it_sample - zOFF_k.get((i,t),0.)) )
                if beta_val < cut_rhs_val - 1e-6: # beta_sample must be >= cut_rhs_val
                    return False
        return True

    def check_feasibility_cuts(sample, iteration_data):
        # This check needs to align with how the feasibility cut is added to the BQM
        # BQM constraint: scaled_terms + scaled_constant >= 0
        for data in iteration_data:
            if data['type'] == 'feasibility':
                rays_k = data['rays']
                cut_lhs_val = 0
                # Variable terms
                for i in generators:
                    for t_loop in time_periods:
                        u_it_sample = sample.get(f'u_{i}_{t_loop}', 0)
                        zon_it_sample = sample.get(f'zON_{i}_{t_loop}', 0)
                        zoff_it_sample = sample.get(f'zOFF_{i}_{t_loop}', 0)

                        u_coeff_val = (rays_k['min_power'].get((i, t_loop), 0.) * gen_data[i]['Pmin'] +
                                    rays_k['max_power'].get((i, t_loop), 0.) * gen_data[i]['Pmax'] +
                                    rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rd'])
                        if t_loop < Periods:
                            u_coeff_val += rays_k['ramp_up'].get((i, t_loop + 1), 0.) * gen_data[i]['Ru']
                        cut_lhs_val += u_coeff_val * u_it_sample

                        cut_lhs_val += (rays_k['ramp_up'].get((i, t_loop), 0.) * gen_data[i]['Rsu']) * zon_it_sample
                        cut_lhs_val += (rays_k['ramp_down'].get((i, t_loop), 0.) * gen_data[i]['Rsd']) * zoff_it_sample
                
                # Constant terms
                for t_loop in time_periods:
                    cut_lhs_val += rays_k['demand'].get(t_loop, 0.) * demand[t_loop]
                    for i in generators:
                        if t_loop == 1:
                            cut_lhs_val += rays_k['ramp_up'].get((i, 1), 0.) * gen_data[i]['Ru'] * u_initial[i]
                
                if cut_lhs_val < 0 - 1e-6: # sum(terms) + constant must be >= 0
                    return False
        return True


    solutions_found = []
    



    start_time = time.time()
    max_iter, epsilon, num_beta_bits = 35, 1.0, BETA_BITS
    iteration_data, lower_bound, upper_bound = [], -float('inf'), float('inf')
    stagnation_limit   = 20
    stagnation_counter = 0
    best_ub_solution = None
    u_current, zON_current, zOFF_current = {}, {}, {}
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0
            u_prev_val = u_initial[i] if t == 1 else u_current.get((i, t-1), 1.0)
            if u_current[i, t] > 0.5 and u_prev_val < 0.5: zON_current[i, t], zOFF_current[i, t] = 1.0, 0.0
            elif u_current[i, t] < 0.5 and u_prev_val > 0.5: zON_current[i, t], zOFF_current[i, t] = 0.0, 1.0
            else: zON_current[i, t], zOFF_current[i, t] = 0.0, 0.0
    
    gurobi_solver = SolverFactory("gurobi")
    sub_solver = SolverFactory("gurobi_persistent", solver_io='python')
    print("--- Starting Hybrid Benders Decomposition for UCP ---")
    print(f"Master Solvers: D-Wave SA (Heuristic), Gurobi (Relaxed LP for LB)")
    print(f"Subproblem Solver: Gurobi")
    sampleset_dict = {}
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")


    lower_bounds = []
    upper_bounds = []




    for k_iter_count in range(1, max_iter + 1):
        print(f"========================= Iteration {k_iter_count} =========================")
        print("--- 1. Solving Subproblem (LP) ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)
        sub_solver.set_instance(subproblem)
        sub_solver.set_gurobi_param('InfUnbdInfo', 1)
        results = sub_solver.solve(tee=False)
        is_infeasible = results.solver.termination_condition == TerminationCondition.infeasible
        if not is_infeasible:
            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            commitment_cost = sum(gen_data[i]['Csu']*zON_current.get((i,t),0) + gen_data[i]['Csd']*zOFF_current.get((i,t),0) + gen_data[i]['Cf']*u_current.get((i,t),0) for i in generators for t in time_periods)
            current_total_cost = commitment_cost + sub_obj_val
            if current_total_cost < upper_bound - 1e-6:
                upper_bound = current_total_cost
                best_ub_solution = {
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
            duals = {f'lambda_{k}': {idx: subproblem.dual.get(con[idx], 0.0) for idx in con} for k, con in {'min': subproblem.MinPower, 'max': subproblem.MaxPower, 'ru': subproblem.RampUp, 'rd': subproblem.RampDown, 'dem': subproblem.Demand}.items()}
            iteration_data.append({'type': 'optimality', 'sub_obj': sub_obj_val, 'duals': duals, 'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
        else:
            print("Subproblem Status: INFEASIBLE. Generating Feasibility Cut.")
            rays = {f'{c_name}': {idx: sub_solver.get_linear_constraint_attr(c, 'FarkasDual') or 0.0 for idx, c in con.items()} for c_name, con in {'min_power': subproblem.MinPower, 'max_power': subproblem.MaxPower, 'ramp_up': subproblem.RampUp, 'ramp_down': subproblem.RampDown}.items()}
            rays['demand'] = {idx: sub_solver.get_linear_constraint_attr(subproblem.Demand[idx], 'FarkasDual') or 0.0 for idx in subproblem.Demand}
            iteration_data.append({'type': 'feasibility', 'rays': rays})


        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

        termination_reason = None

        print(f"Current Bounds -> LB: {lower_bound:.4f}, UB: {upper_bound:.4f}")
        if upper_bound - lower_bound <= epsilon:
            termination_reason = "Convergence tolerance met"
            print("\nConvergence tolerance met."); break
        if k_iter_count == max_iter:
            termination_reason = "Maximum iterations reached"
            print("\nMaximum iterations reached."); break
        
        if stagnation_counter >= stagnation_limit:
            termination_reason = "Stagnation limit reached"
            print(f"\nTerminating: Upper bound unchanged for {stagnation_limit} "
                f"consecutive iterations.")
            break
        if Periods == 3 and upper_bound ==191.8:
            print("\nTerminating: Known optimal solution found for 3-period case.")
            termination_reason = "Known optimal solution for 3-period case"
            break

        if Periods == 5 and upper_bound ==293.5: 
            print("\nTerminating: Known optimal solution found for 3-period case.")
            termination_reason = "Known optimal solution for 5-period case"
            break

        if Periods == 6 and upper_bound ==347.5: 
            print("\nTerminating: Known optimal solution found for 3-period case.")
            termination_reason = "Known optimal solution for 6-period case"
            break
        

        print("\n--- 2. Solving Master Problem ")
        master_bqm = build_master_dwave(iteration_data, num_beta_bits)
        best_feasible_sample, lowest_energy = None, float('inf')
        for attempt in range(10):

            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(master_bqm, num_reads=1, num_sweeps=num_iter, label=f'HENRIK-UCP-Master-Iter-{k_iter_count}')
            
            #sampler = LeapHybridBQMSampler(token="token")
            #sampleset = sampler.sample(master_bqm, label=f'HENRIK-UCP-Master-Iter-{k_iter_count}', time_limit=5)
            
            sampleset_dict[k_iter_count] = sampleset.copy()
            for i_sample in range(len(sampleset)):
                sample = sampleset.samples()[i_sample]
                if (check_logic1_feasibility(sample) and
                    check_logic2_feasibility(sample) and
                    check_optimality_cuts(sample, iteration_data, num_beta_bits) and
                    check_feasibility_cuts(sample, iteration_data)):
                    energy = sampleset.record.energy[i_sample]
                    if energy < lowest_energy:
                        lowest_energy = energy
                        best_feasible_sample = sample
            if best_feasible_sample:
                print(f"Found a feasible integer solution on attempt {attempt + 1}.")
                break
            else:
                print(f"No feasible solution found on attempt {attempt + 1}. Retrying...")
                print("Is logic 1 violated?", not check_logic1_feasibility(sample))
                print("Is logic 2 violated?", not check_logic2_feasibility(sample))
                print("Are optimality cuts violated?", not check_optimality_cuts(sample, iteration_data, num_beta_bits))
                print("Are feasibility cuts violated?", not check_feasibility_cuts(sample, iteration_data))
        if best_feasible_sample is None:
            print("CRITICAL: Master Problem (QUBO) FAILED to find a feasible solution after all attempts. Stopping.")
            termination_reason = "Master problem failed to find feasible solution"
            break

        #printing beta values, the combined value of beta is the sum of all beta bits
        print("combined beta value:", sum((2**j) * best_feasible_sample.get(f'beta_{j}', 0) for j in range(num_beta_bits)))
        for var, val in best_feasible_sample.items():
            p = var.split('_'); vtype=p[0]
            if vtype == 'u': u_current[int(p[1]), int(p[2])] = val
            elif vtype == 'zON': zON_current[int(p[1]), int(p[2])] = val
            elif vtype == 'zOFF': zOFF_current[int(p[1]), int(p[2])] = val
        print("\n--- 3. Solving Relaxed Master (LP for Lower Bound) ---")
        relaxed_master_model = build_relaxed_master_pyomo(iteration_data)
        master_results = gurobi_solver.solve(relaxed_master_model, tee=False)
        if master_results.solver.termination_condition == TerminationCondition.optimal:
            master_obj_val = pyo.value(relaxed_master_model.OBJ)
            lower_bound = max(lower_bound, master_obj_val)
            print(f"Relaxed Master Solved. New LB candidate: {master_obj_val:.4f}. Updated Z_LB: {lower_bound:.4f}")
            print(f"The beta value from the relaxed master is: {relaxed_master_model.beta.value:.4f}")
        else:
            print(f"CRITICAL: Relaxed Master LP FAILED to solve! Status: {master_results.solver.termination_condition}. Stopping.")
            termination_reason = "Relaxed master LP failed to solve"
            break
    end_time = time.time()
    print("\n========================= Benders Terminated =========================")
    print(f"Final Lower Bound (Z_LB): {lower_bound:.4f}")
    print(f"Final Upper Bound (Z_UB): {upper_bound:.4f}")
    final_gap = (upper_bound - lower_bound) / (abs(upper_bound) + 1e-9) if upper_bound < float('inf') else float('inf')
    print(f"Final Gap: {final_gap:.6f}")
    print(f"Total Time: {end_time - start_time:.2f} seconds")
    print("\n--- Best Feasible Solution Found (corresponds to Z_UB) ---")
    print(f"Best Total Cost (Upper Bound): {upper_bound:.4f}")
    print("Commitment Schedule (u_it):")
    for t in time_periods:
        print(f"  t={t}: ",
            {i: round(best_ub_solution['u_vals'][(i, t)]) for i in generators})
    final_subproblem = build_subproblem(best_ub_solution['u_vals'],
                                        best_ub_solution['zON_vals'],
                                        best_ub_solution['zOFF_vals'])
    
    print("Final turning ON schedule (z^On):")
    for t in time_periods:
        print(f"  t={t}: ", {i: round(best_ub_solution['zON_vals'][(i, t)]) for i in generators})
    
    print("Final turning OFF schedule (z^Off):")
    for t in time_periods:
        print(f"  t={t}: ", {i: round(best_ub_solution['zOFF_vals'][(i, t)]) for i in generators})


    final_solver = SolverFactory("gurobi")
    final_res = final_solver.solve(final_subproblem)
    if final_res.solver.termination_condition == TerminationCondition.optimal:
        var_cost = pyo.value(final_subproblem.OBJ)
        com_cost = sum(gen_data[i]['Csu'] * best_ub_solution['zON_vals'][(i, t)] +
                    gen_data[i]['Csd'] * best_ub_solution['zOFF_vals'][(i, t)] +
                    gen_data[i]['Cf']  * best_ub_solution['u_vals'][(i, t)]
                    for i in generators for t in time_periods)
        print("\nFinal Dispatch (p_it):")
        for t in time_periods:
            print(f"  t={t}: ", {i: f"{pyo.value(final_subproblem.p[i, t]):.2f}" for i in generators})
        print("\nCost recap:")
        print(f"  Commitment cost : {com_cost:.4f}")
        print(f"  Variable cost   : {var_cost:.4f}")
        print(f"  Total cost      : {com_cost + var_cost:.4f}")
        # Demand check (sanity)
        print("\nDemand check:")
        for t in time_periods:
            total_prod = sum(pyo.value(final_subproblem.p[i, t]) for i in generators)
            print(f"  t={t}: produced={total_prod:.2f}  demand={demand[t]}"
                f"  met={total_prod >= demand[t] - 1e-4}")
    else:
        print("WARNING: Could not re-solve dispatch for the final schedule.")
    if best_ub_solution:
        print(f"\n--- Best Found Solution (from iteration {best_ub_solution['iter']}) ---")
        print(f"Best Total Cost: {best_ub_solution['total_cost']:.4f}")
        print("Commitment Schedule (u_it):")
        for t_val in time_periods: 
            print(f"  t={t_val}: ", {i: round(best_ub_solution['u_vals'].get((i, t_val), 0)) for i in generators})
    else:
        print("\nNo feasible solution was found during the process.")

    solutions_found.append(upper_bound)

    
    commitment_schedule = {t: {i: round(best_ub_solution['u_vals'].get((i, t), 0)) for i in generators} for t in time_periods}
    turning_on_schedule = {t: {i: round(best_ub_solution['zON_vals'].get((i, t), 0)) for i in generators} for t in time_periods}
    turning_off_schedule = {t: {i: round(best_ub_solution['zOFF_vals'].get((i, t), 0)) for i in generators} for t in time_periods}

    # Create a new DataFrame with the data you want to append
    new_row_df = pd.DataFrame([{
        "number_of_time_periods": len(time_periods),
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "best_solution": best_ub_solution['total_cost'] if best_ub_solution else None,
        "time_used": end_time - start_time,
        "number_of_benders_iterations": k_iter_count,
        "number_of_sweeps": num_iter,
        "commitment_schedule": commitment_schedule,
        "turning_on_schedule": turning_on_schedule,
        "turning_off_schedule": turning_off_schedule,
        "termination_reason": termination_reason
    }])

    # Use pd.concat to append the new row(s) to the DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    # if csv file exists, delete it and replace with new one
    csv_file = "dwave_qubo_results.csv"
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

if __name__ == '__main__':


    df = pd.DataFrame(columns=["number_of_time_periods", "lower_bounds", "upper_bounds", "best_solution", "time_used", "number_of_benders_iterations","number_of_sweeps", "commitment_schedule", "turning_on_schedule", "turning_off_schedule","termination_reason"])
    
    
    for _ in range(30):
        main(iterations=256_000, time_periods=6)

    for _ in range(100):
        main(iterations=640_000, time_periods=6)
    




#number of time periods
#upper bound in each benders iteration
#lower bound in each benders iteration
#final best solution, also final upper bound
#final lower bound
#time used
#number of benders iterations
#number of sweeps/iterations in each master problem solve
#commitment schedule, which generators are on/off in each time period
#turning on schedule
#turning off schedule

#with 3 time periods
#100 with 50_000 iter
#100 with 1_000_000 iter
#100 with 5_000_000 iter

#with 5 time periods
#100 with 50_000 iter
#100 with 1_000_000 iter
#100 with 5_000_000 iter

#with 6 time periods
#100 with 50_000 iter
#100 with 1_000_000 iter
#100 with 5_000_000 iter