##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition. The subproblem is a Pyomo/Gurobi LP.
##### The master problem is a QUBO formulated with D-Wave's dimod and
##### solved with the SimulatedAnnealingSampler. Feasibility is checked manually.

# Neccecary imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import dimod
from neal import SimulatedAnnealingSampler
import numpy as np

# Choose the number of time periods wanted:
Periods = 3
# Sets
generators = [1, 2, 3]
time_periods = [x for x in range(1, Periods+1)] # T=3 hours

# Generator Parameters
gen_data = {
    1: {'Pmin': 50,  'Pmax': 350, 'Rd': 300, 'Rsd': 300, 'Ru': 200, 'Rsu': 200, 'Cf': 5, 'Csu': 20, 'Csd': 0.5, 'Cv': 0.100},
    2: {'Pmin': 80,  'Pmax': 200, 'Rd': 150, 'Rsd': 150, 'Ru': 100, 'Rsu': 100, 'Cf': 7, 'Csu': 18, 'Csd': 0.3, 'Cv': 0.125},
    3: {'Pmin': 40,  'Pmax': 140, 'Rd': 100, 'Rsd': 100, 'Ru': 100, 'Rsu': 100, 'Cf': 6, 'Csu': 5,  'Csd': 1.0, 'Cv': 0.150}
}

# Demand Parameters
demand = {1: 160, 2: 500, 3: 400} # Demand for each time period

# Initial Conditions for time period = 0
u_initial = {1: 0, 2: 0, 3: 1}
p_initial = {1: 0, 2: 0, 3: 100}

# Number of bits for beta variable
num_beta_bits = 7

# This function creates the continous sub problem (LP)
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):
    model = pyo.ConcreteModel(name="Sub Problem")
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

def build_master_dwave(iteration_data):
    bqm = dimod.BinaryQuadraticModel('BINARY')
    SCALE_FACTOR = 1000
    lambda_logic1, lambda_logic2, lambda_opt_cut, lambda_feas_cut = 25, 20, 80, 80
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
            if t > 1: terms.append((u_vars[i, t-1], -1))
            bqm.add_linear_equality_constraint(terms, lagrange_multiplier=lambda_logic1, constant=constant)
            bqm.add_quadratic(zON_vars[i,t], zOFF_vars[i,t], lambda_logic2)
    for k_idx, data in enumerate(iteration_data):
        if data['type'] == 'optimality':
            sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
            terms, constant_term = [], -sub_obj_k
            for j in range(num_beta_bits): terms.append((beta_vars[j], 2**j))
            for i in generators:
                for t in time_periods:
                    u_coeff = -duals_k['lambda_min'].get((i, t), 0.) * gen_data[i]['Pmin'] - duals_k['lambda_max'].get((i, t), 0.) * gen_data[i]['Pmax'] - duals_k['lambda_rd'].get((i, t), 0.) * gen_data[i]['Rd']
                    if t < Periods: u_coeff -= duals_k['lambda_ru'].get((i, t + 1), 0.) * gen_data[i]['Ru']
                    terms.append((u_vars[i,t], u_coeff))
                    terms.append((zON_vars[i,t], -duals_k['lambda_ru'].get((i,t), 0.)*gen_data[i]['Rsu']))
                    terms.append((zOFF_vars[i,t], -duals_k['lambda_rd'].get((i,t), 0.)*gen_data[i]['Rsd']))
                    u_prev_k = u_k.get((i, t-1), u_initial[i]) if t > 1 else u_initial[i]
                    constant_term += duals_k['lambda_min'].get((i, t), 0.) * gen_data[i]['Pmin'] * u_k.get((i, t), 0.) + duals_k['lambda_max'].get((i, t), 0.) * gen_data[i]['Pmax'] * u_k.get((i, t), 0.)
                    constant_term += duals_k['lambda_ru'].get((i, t), 0.) * (gen_data[i]['Ru'] * u_prev_k + gen_data[i]['Rsu'] * zON_k.get((i, t), 0.))
                    constant_term += duals_k['lambda_rd'].get((i, t), 0.) * (gen_data[i]['Rd'] * u_k.get((i, t), 0.) + gen_data[i]['Rsd'] * zOFF_k.get((i, t), 0.))
            scaled_terms = [(var, int(round(c * SCALE_FACTOR))) for var, c in terms]
            scaled_constant = int(round(constant_term * SCALE_FACTOR))
            bqm.add_linear_inequality_constraint(scaled_terms, lagrange_multiplier=lambda_opt_cut, label=f"opt_cut_{k_idx}", constant=scaled_constant, lb=0, ub=np.iinfo(np.int64).max)
        elif data['type'] == 'feasibility':
            rays_k = data['rays']
            terms, constant_term = [], 0
            # ** START OF CORRECTION **
            for i in generators:
                for t in time_periods:
                    # Coefficient for u_it
                    u_coeff = (rays_k['min_power'].get((i, t), 0.) * gen_data[i]['Pmin'] +
                               rays_k['max_power'].get((i, t), 0.) * gen_data[i]['Pmax'] +
                               rays_k['ramp_down'].get((i, t), 0.) * gen_data[i]['Rd'])
                    # u_it is u_prev in ramp-up constraint for t+1
                    if t < Periods:
                        u_coeff += rays_k['ramp_up'].get((i, t + 1), 0.) * gen_data[i]['Ru']
                    terms.append((u_vars[i, t], u_coeff))

                    # Coefficient for zON_it
                    zON_coeff = rays_k['ramp_up'].get((i, t), 0.) * gen_data[i]['Rsu']
                    terms.append((zON_vars[i, t], zON_coeff))

                    # Coefficient for zOFF_it
                    zOFF_coeff = rays_k['ramp_down'].get((i, t), 0.) * gen_data[i]['Rsd']
                    terms.append((zOFF_vars[i, t], zOFF_coeff))
            
            # Constant part of the expression
            for t in time_periods:
                constant_term += rays_k['demand'].get(t, 0.) * demand[t]
                for i in generators:
                    if t == 1:
                        # Contribution from u_prev at t=1 (which is u_initial)
                        constant_term += rays_k['ramp_up'].get((i, 1), 0.) * gen_data[i]['Ru'] * u_initial[i]

            # The constraint is sum(terms) + constant >= 0
            scaled_terms = [(var, int(round(c * SCALE_FACTOR))) for var, c in terms]
            scaled_constant = int(round(constant_term * SCALE_FACTOR))
            bqm.add_linear_inequality_constraint(scaled_terms, lagrange_multiplier=lambda_feas_cut, label=f"feas_cut_{k_idx}", constant=scaled_constant, lb=0, ub=np.iinfo(np.int64).max)
            # ** END OF CORRECTION **
    return bqm

# --- MANUAL FEASIBILITY CHECKING FUNCTIONS ---
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
            # Constraint is zON_it + zOFF_it <= 1
            # For binary variables, this is equivalent to checking zON_it * zOFF_it == 0
            if zon_it * zoff_it != 0: # If both are 1, product is 1
                return False
    return True

def check_optimality_cuts(sample, iteration_data):
    for data in iteration_data:
        if data['type'] == 'optimality':
            sub_obj_k, duals_k, u_k, zON_k, zOFF_k = data['sub_obj'], data['duals'], data['u_vals'], data['zON_vals'], data['zOFF_vals']
            beta_val = sum((2**j) * sample.get(f'beta_{j}', 0) for j in range(num_beta_bits))
            cut_rhs = sub_obj_k
            for i in generators:
                for t in time_periods:
                    u_it = sample.get(f'u_{i}_{t}', 0)
                    zon_it = sample.get(f'zON_{i}_{t}', 0)
                    zoff_it = sample.get(f'zOFF_{i}_{t}', 0)
                    u_prev = u_initial[i] if t == 1 else sample.get(f'u_{i}_{t-1}', 0)
                    u_prev_k = u_k.get((i, t-1), u_initial[i]) if t > 1 else u_initial[i]
                    cut_rhs += duals_k['lambda_min'].get((i, t), 0.) * gen_data[i]['Pmin'] * (u_it - u_k.get((i, t), 0.))
                    cut_rhs += duals_k['lambda_max'].get((i, t), 0.) * gen_data[i]['Pmax'] * (u_it - u_k.get((i, t), 0.))
                    cut_rhs += duals_k['lambda_ru'].get((i, t), 0.) * (gen_data[i]['Ru'] * (u_prev - u_prev_k) + gen_data[i]['Rsu'] * (zon_it - zON_k.get((i, t), 0.)))
                    cut_rhs += duals_k['lambda_rd'].get((i, t), 0.) * (gen_data[i]['Rd'] * (u_it - u_k.get((i, t), 0.)) + gen_data[i]['Rsd'] * (zoff_it - zOFF_k.get((i, t), 0.)))
            if beta_val < cut_rhs - 1e-6:
                return False
    return True

def check_feasibility_cuts(sample, iteration_data):
    for data in iteration_data:
        if data['type'] == 'feasibility':
            rays_k = data['rays']
            cut_lhs = 0
            for i in generators:
                for t in time_periods:
                    u_it = sample.get(f'u_{i}_{t}', 0)
                    zon_it = sample.get(f'zON_{i}_{t}', 0)
                    zoff_it = sample.get(f'zOFF_{i}_{t}', 0)
                    u_prev = u_initial[i] if t == 1 else sample.get(f'u_{i}_{t-1}', 0)
                    cut_lhs += rays_k['min_power'].get((i, t), 0.) * (gen_data[i]['Pmin'] * u_it)
                    cut_lhs += rays_k['max_power'].get((i, t), 0.) * (gen_data[i]['Pmax'] * u_it)
                    cut_lhs += rays_k['ramp_up'].get((i, t), 0.) * (gen_data[i]['Ru'] * u_prev + gen_data[i]['Rsu'] * zon_it)
                    cut_lhs += rays_k['ramp_down'].get((i, t), 0.) * (gen_data[i]['Rd'] * u_it + gen_data[i]['Rsd'] * zoff_it)
            for t in time_periods:
                cut_lhs += rays_k['demand'].get(t, 0.) * demand[t]
            if cut_lhs < -1e-6:
                return False
    return True

def main():
    start_time = time.time()
    max_iter, epsilon = 30, 1.0
    iteration_data, lower_bound, upper_bound = [], -float('inf'), float('inf')
    best_solution_for_ub_display = None
    u_current, zON_current, zOFF_current = {}, {}, {}
    for t in time_periods:
        for i in generators:
            u_current[i, t] = 1.0
            u_prev_val = u_initial[i] if t == 1 else u_current.get((i, t-1), 1.0)
            if u_current[i, t] > 0.5 and u_prev_val < 0.5: zON_current[i, t], zOFF_current[i, t] = 1.0, 0.0
            elif u_current[i, t] < 0.5 and u_prev_val > 0.5: zON_current[i, t], zOFF_current[i, t] = 0.0, 1.0
            else: zON_current[i, t], zOFF_current[i, t] = 0.0, 0.0
    master_solver = SimulatedAnnealingSampler()
    sub_solver = SolverFactory("gurobi_persistent", solver_io='python')
    print(f"--- Starting Benders Decomposition for UCP ---")
    print(f"Master Solver: D-Wave SimulatedAnnealingSampler, Subproblem Solver: {sub_solver.name}")
    print(f"Max Iterations: {max_iter}, Tolerance: {epsilon}\n")
    for k_iter_count in range(1, max_iter + 1):
        print(f"========================= Iteration {k_iter_count} =========================")
        print("--- Solving Subproblem ---")
        subproblem = build_subproblem(u_current, zON_current, zOFF_current)
        sub_solver.set_instance(subproblem)
        sub_solver.set_gurobi_param('InfUnbdInfo', 1)
        sub_solver.set_gurobi_param('DualReductions', 0)
        results = sub_solver.solve(tee=False)
        is_infeasible = results.solver.termination_condition == TerminationCondition.infeasible
        if not is_infeasible:
            sub_obj_val = pyo.value(subproblem.OBJ)
            print(f"Subproblem Status: {results.solver.termination_condition}, Objective: {sub_obj_val:.4f}")
            commitment_cost = sum(gen_data[i]['Csu']*zON_current.get((i,t),0) + gen_data[i]['Csd']*zOFF_current.get((i,t),0) + gen_data[i]['Cf']*u_current.get((i,t),0) for i in generators for t in time_periods)
            current_total_cost = commitment_cost + sub_obj_val
            if check_logic1_feasibility({f'u_{i}_{t}': v for (i,t),v in u_current.items()} | {f'zON_{i}_{t}': v for (i,t),v in zON_current.items()} | {f'zOFF_{i}_{t}': v for (i,t),v in zOFF_current.items()}) and current_total_cost < upper_bound:
                upper_bound = current_total_cost
                best_solution_for_ub_display = {'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy(), 'iter': k_iter_count, 'total_cost': upper_bound}
                print(f"New Best Upper Bound (Z_UB): {upper_bound:.4f}")
            duals = {f'lambda_{k}': {idx: subproblem.dual.get(con[idx], 0.0) for idx in con} for k, con in {'min': subproblem.MinPower, 'max': subproblem.MaxPower, 'ru': subproblem.RampUp, 'rd': subproblem.RampDown}.items()}
            duals['lambda_dem'] = {idx: subproblem.dual.get(subproblem.Demand[idx], 0.0) for idx in subproblem.Demand}
            iteration_data.append({'type': 'optimality', 'sub_obj': sub_obj_val, 'duals': duals, 'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
        else:
            print("Subproblem Status: INFEASIBLE. Generating Feasibility Cut.")
            rays = {f'{c_name}': {idx: sub_solver.get_linear_constraint_attr(c, 'FarkasDual') or 0.0 for idx, c in con.items()} for c_name, con in {'min_power': subproblem.MinPower, 'max_power': subproblem.MaxPower, 'ramp_up': subproblem.RampUp, 'ramp_down': subproblem.RampDown}.items()}
            rays['demand'] = {idx: sub_solver.get_linear_constraint_attr(subproblem.Demand[idx], 'FarkasDual') or 0.0 for idx in subproblem.Demand}
            iteration_data.append({'type': 'feasibility', 'rays': rays, 'u_vals': u_current.copy(), 'zON_vals': zON_current.copy(), 'zOFF_vals': zOFF_current.copy()})
        print(f"Current Lower Bound (Z_LB): {lower_bound:.4f}, Upper Bound (Z_UB): {upper_bound:.4f}")
        if upper_bound - lower_bound <= epsilon and k_iter_count > 1: print("\nConvergence tolerance met."); break
        if k_iter_count == max_iter: print("\nMaximum iterations reached."); break
        
        
        print("\n--- Solving Master Problem with D-Wave Neal ---")
        master_bqm = build_master_dwave(iteration_data)
        best_feasible_sample = None
        for attempt in range(5):
            print(f"  Attempt {attempt + 1}/5...")
            sampleset = master_solver.sample(master_bqm, num_reads=2000)
            lowest_energy, current_best_sample = float('inf'), None
            if len(sampleset) > 0: # Ensure there are samples to check
                for i_sample in range(len(sampleset)): # Iterate by index to access energy later
                    sample = sampleset.samples()[i_sample]
                    
                    # Perform all manual feasibility checks
                    is_logic1_feasible = check_logic1_feasibility(sample)
                    is_logic2_feasible = check_logic2_feasibility(sample) # ADDED CHECK
                    are_opt_cuts_feasible = check_optimality_cuts(sample, iteration_data)
                    are_feas_cuts_feasible = check_feasibility_cuts(sample, iteration_data)
                    
                    if (is_logic1_feasible and 
                        is_logic2_feasible and  # ADDED CHECK
                        are_opt_cuts_feasible and 
                        are_feas_cuts_feasible):
                        
                        energy = sampleset.record.energy[i_sample]
                        if energy < lowest_energy:
                            lowest_energy, current_best_sample = energy, sample
            
            if current_best_sample:
                print(f"  Found a fully feasible solution with energy {lowest_energy:.2f}.")
                best_feasible_sample = current_best_sample
                break # Exit retry loop if a feasible solution is found
            else:
                print(f"  No fully feasible solutions found in this attempt.")
        if best_feasible_sample is None: print("Master Problem FAILED to find a feasible solution."); break
        temp_u, temp_zON, temp_zOFF, temp_beta = {}, {}, {}, {}
        for var, val in best_feasible_sample.items():
            p = var.split('_'); vtype=p[0]
            if vtype == 'u': temp_u[int(p[1]), int(p[2])] = val
            elif vtype == 'zON': temp_zON[int(p[1]), int(p[2])] = val
            elif vtype == 'zOFF': temp_zOFF[int(p[1]), int(p[2])] = val
            elif vtype == 'beta': temp_beta[int(p[1])] = val
        u_current, zON_current, zOFF_current = temp_u, temp_zON, temp_zOFF
        commitment_cost = sum(gen_data[i]['Cf']*u_current.get((i,t),0) + gen_data[i]['Csu']*zON_current.get((i,t),0) + gen_data[i]['Csd']*zOFF_current.get((i,t),0) for i in generators for t in time_periods)
        beta_val = sum((2**j) * temp_beta.get(j, 0) for j in range(num_beta_bits))
        lower_bound = max(lower_bound, commitment_cost + beta_val)
        print(f"Master solved. New Z_LB candidate: {commitment_cost + beta_val:.4f}. Updated Z_LB: {lower_bound:.4f}")
if __name__ == '__main__':
    main()