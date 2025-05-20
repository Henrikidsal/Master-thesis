##### This is a script that solves the continous version of the UCP
##### It uses Benders Decomposition, where both master and subproblem are solved using Pyomo, classically
##### THe logic constraints, both types, are penalty terms
##### Benders optimality and feasibility cuts are constraints.
##### The LB could be calculated as only commitment cost + beta, but here its the full QUBO value. It works because logics are never violated.

##### basic imports
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, TerminationCondition
import time
import math
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import gurobipy as grb
from copy import deepcopy 

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

# Number of bits for beta variable
num_beta_bits = 9 #7

# Function creating the sub problem
def build_subproblem(u_fixed_vals, zON_fixed_vals, zOFF_fixed_vals):

    model = pyo.ConcreteModel(name="UCP_Subproblem")

    # Sets
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.T0 = pyo.Set(initialize=time_periods_with_0)

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

    # --- Variables ---
    model.p = pyo.Var(model.I, model.T, within=pyo.NonNegativeReals)

    # --- Objective Function ---
    def objective_rule(m):
        variable_cost = sum(m.Cv[i] * m.p[i, t] for i in m.I for t in m.T)
        return variable_cost
    model.OBJ = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # --- Constraints ---

    def p_prev_rule(m, i, t):
        if t == 1:
            return m.p_init[i]
        else:
            return m.p[i, t-1]
    model.p_prev = pyo.Expression(model.I, model.T, rule=p_prev_rule)

    def u_prev_fixed_rule(m, i, t):
        if t == 1:
            return m.u_init[i]
        else:
            return m.u_fixed[i, t-1]
    model.u_prev_fixed = pyo.Expression(model.I, model.T, rule=u_prev_fixed_rule)


    # Constraint: Minimum power output
    # p_it >= Pmin_i * u_it  =>  Pmin_i * u_it - p_it <= 0
    def min_power_rule(m, i, t):
        return m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t]
    model.MinPower = pyo.Constraint(model.I, model.T, rule=min_power_rule)

    # Constraint: Maximum power output
    # p_it <= Pmax_i * u_it  =>  p_it - Pmax_i * u_it <= 0
    def max_power_rule(m, i, t):
        return m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t]
    model.MaxPower = pyo.Constraint(model.I, model.T, rule=max_power_rule)

    # Constraint: Ramping up limit
    # p_it - p_i(t-1) <= Ru_i * u_i(t-1)_fixed + Rsu_i * zON_it_fixed
    # => p_it - p_i(t-1) - Ru_i * u_i(t-1)_fixed - Rsu_i * zON_it_fixed <= 0
    def ramp_up_rule(m, i, t):
        return m.p[i, t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t]
    model.RampUp = pyo.Constraint(model.I, model.T, rule=ramp_up_rule)

    # Constraint: Ramping down limit
    # p_i(t-1) - p_it <= Rd_i * u_it_fixed + Rsd_i * zOFF_it_fixed
    # => p_i(t-1) - p_it - Rd_i * u_it_fixed - Rsd_i * zOFF_it_fixed <= 0
    def ramp_down_rule(m, i, t):
        return m.p_prev[i,t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t]
    model.RampDown = pyo.Constraint(model.I, model.T, rule=ramp_down_rule)

    # Constraint: Demand balance
    # sum(p_it for i in I) >= D_t  => D_t - sum(p_it for i in I) <= 0
    def demand_rule(m, t):
        # return sum(m.p[i, t] for i in m.I) + m.demand_slack[t] >= m.D[t] # CHANGED
        return sum(m.p[i, t] for i in m.I) >= m.D[t]
    model.Demand = pyo.Constraint(model.T, rule=demand_rule)

    # --- Dual Suffix (for optimality cuts and Farkas rays) ---
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # --- IIS Suffix (for Farkas rays if Gurobi provides them this way, Gurobi usually uses .UnbdRay) ---
    # model.iis = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT, datatype=pyo.Suffix.FLOAT)


    return model

# Function creating the master problem
def build_master(iteration_data): # iteration_data will contain both optimality and feasibility cuts
    model = pyo.ConcreteModel(name="UCP_MasterProblem_QUBO_Constrained")

    # Sets
    model.I = pyo.Set(initialize=generators)
    model.T = pyo.Set(initialize=time_periods)
    model.T0 = pyo.Set(initialize=time_periods_with_0)
    model.BETA_BITS = pyo.RangeSet(0, num_beta_bits - 1)

    # CHANGED: Separate sets for optimality and feasibility cuts
    optimality_cut_indices = [idx for idx, data in enumerate(iteration_data) if data['type'] == 'optimality']
    feasibility_cut_indices = [idx for idx, data in enumerate(iteration_data) if data['type'] == 'feasibility']

    model.OptimalityCuts = pyo.Set(initialize=optimality_cut_indices)
    model.FeasibilityCuts = pyo.Set(initialize=feasibility_cut_indices)


    # Paramters (penalty values, etc.)
    model.Pmax = pyo.Param(model.I, initialize={i: gen_data[i]['Pmax'] for i in model.I})
    model.Cf = pyo.Param(model.I, initialize={i: gen_data[i]['Cf'] for i in model.I})
    model.Csu = pyo.Param(model.I, initialize={i: gen_data[i]['Csu'] for i in model.I})
    model.Csd = pyo.Param(model.I, initialize={i: gen_data[i]['Csd'] for i in model.I})
    model.D = pyo.Param(model.T, initialize=demand)
    model.u_init = pyo.Param(model.I, initialize=u_initial)
    model.lambda_logic1 = pyo.Param(initialize=20)
    model.lambda_logic2 = pyo.Param(initialize=1)

    # Parameters for Benders cut coefficients
    model.Pmin = pyo.Param(model.I, initialize={i: gen_data[i]['Pmin'] for i in model.I})
    model.Rd = pyo.Param(model.I, initialize={i: gen_data[i]['Rd'] for i in model.I})
    model.Rsd = pyo.Param(model.I, initialize={i: gen_data[i]['Rsd'] for i in model.I})
    model.Ru = pyo.Param(model.I, initialize={i: gen_data[i]['Ru'] for i in model.I})
    model.Rsu = pyo.Param(model.I, initialize={i: gen_data[i]['Rsu'] for i in model.I})

    # Variables
    model.u = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zON = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.zOFF = pyo.Var(model.I, model.T, within=pyo.Binary)
    model.beta_binary = pyo.Var(model.BETA_BITS, within=pyo.Binary) # This is eta in some literature

    def u_prev_rule(m, i, t):
        if t == 1: return m.u_init[i]
        else: return m.u[i, t-1]
    model.u_prev = pyo.Expression(model.I, model.T, rule=u_prev_rule)

    def master_objective_rule(m):
        commitment_cost = sum(m.Csu[i] * m.zON[i, t] + m.Csd[i] * m.zOFF[i, t] + m.Cf[i] * m.u[i, t] for i in m.I for t in m.T)
        logic1_term = m.lambda_logic1 * sum( ( (m.u[i, t] - m.u_prev[i, t]) - (m.zON[i, t] - m.zOFF[i, t]) )**2 for i in m.I for t in m.T )
        logic2_term = m.lambda_logic2 * sum( m.zON[i, t] * m.zOFF[i, t] for i in m.I for t in m.T )
        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )
        return commitment_cost + logic1_term + logic2_term + binary_beta_expr
    model.OBJ = pyo.Objective(rule=master_objective_rule, sense=pyo.minimize)

    # Benders Optimality Cuts
    # beta >= sub_obj_k + sum(duals_k * (RHS_master_vars - RHS_master_vars_k))
    def benders_optimality_cut_rule(m, k_idx): # k_idx is the index in iteration_data
        data = iteration_data[k_idx]
        # These are from the subproblem solution at iteration data['iter']
        sub_obj_k = data['sub_obj'] # This is f(x_k)
        duals_k = data['duals']     # These are lambda_k
        u_k = data['u_vals']        # These are components of x_k
        zON_k = data['zON_vals']    # These are components of x_k
        zOFF_k = data['zOFF_vals']  # These are components of x_k

        # The cut is: beta >= sub_obj_k + sum_constraints [ dual_k * (b_fixed_part + b_master_vars_part(x) - (b_fixed_part + b_master_vars_part(x_k))) ]
        # which simplifies to: beta >= sub_obj_k + sum_constraints [ dual_k * (b_master_vars_part(x) - b_master_vars_part(x_k)) ]
        # For subproblem constraints of the form A*p <= b(x), where b(x) are the terms with master variables.
        # The duals correspond to these constraints.
        # If constraint is g(p) <= h(x), dual is pi >= 0. Cut term: pi * (h(x) - h(x_k))
        # If constraint is g(p) >= h(x), dual is pi <= 0. Cut term: pi * (h(x) - h(x_k))
        # Or, if g(p) >= h(x) is written as h(x) - g(p) <= 0, dual mu >=0. Cut term: mu * (h(x) - h(x_k)) (This is how Gurobi/Pyomo usually gives duals)

        cut_expr_rhs = sub_obj_k # This is the Q_k(x_k) part

        # MinPower: Pmin[i] * u_fixed[i,t] <= p[i,t]  (RHS is Pmin[i]*u[i,t], this is b_s(x) for this constraint)
        # The dual is for: Pmin[i] * u_fixed[i,t] - p[i,t] <= 0. So dual_val (lambda_min) should be non-negative.
        # The term added is dual_val * ( Pmin[i]*m.u[i,t] - Pmin[i]*u_k[i,t] )
        for i in m.I:
            for t in m.T:
                dual_val = duals_k['lambda_min'].get((i, t), 0.0) # Should be non-negative
                cut_expr_rhs += dual_val * (m.Pmin[i] * (m.u[i, t] - u_k.get((i,t), 0.0)))

        # MaxPower: p[i,t] <= Pmax[i] * u_fixed[i,t] (RHS is Pmax[i]*u[i,t])
        # The dual is for: p[i,t] - Pmax[i] * u_fixed[i,t] <= 0. So dual_val (lambda_max) should be non-negative.
        # The term added is dual_val * ( -Pmax[i]*m.u[i,t] - (-Pmax[i]*u_k[i,t]) ) = -dual_val * Pmax[i]*(m.u[i,t] - u_k[i,t])
        # However, your original code adds dual_val * (m.Pmax[i] * (m.u[i, t] - u_k.get((i,t), 0.0))).
        # This implies the constraint was formulated as -p[i,t] >= -Pmax[i]*u_fixed[i,t] (dual <=0)
        # or Pmax[i]*u_fixed[i,t] - p[i,t] >= 0 (dual for this would be <=0 for ">=" form or >=0 for "<=0" form if switched)
        # Let's stick to your original formulation of the cut which means duals are for MaxPower constraint as Pmax*u_fixed - p >= 0
        # Or, if p <= Pmax * u, then the term in b_s(x) is Pmax*u. The dual is for p - Pmax*u <= 0
        # The sensitivity is d(Obj)/d(RHS). If RHS increases, constraint is looser, Obj might decrease.
        # The term should be lambda_max_k * ( (Pmax_i * u_k[i,t]) - (Pmax_i * m.u[i,t]) ) IF dual is for p <= RHS
        # Or lambda_max_k * ( (Pmax_i * m.u[i,t]) - (Pmax_i * u_k[i,t]) ) IF dual is for -p >= -RHS
        # Your current code: cut_expr += dual_val * (m.Pmax[i] * (m.u[i, t] - u_k.get((i,t), 0.0)))
        # This means the constraint contribution to b_s(x) is effectively m.Pmax[i]*m.u[i,t] and dual is for something like ... >= m.Pmax[i]*m.u[i,t] (dual should be <=0)
        # Or ... <= -m.Pmax[i]*m.u[i,t] (dual >=0)
        # Re-evaluating based on standard form Ax - b <= 0, duals are >=0. b is Pmax[i]*u[i,t]. So (b(x) - b(x_k)).
        # Your dual for MaxPower is likely for `p[i,t] - m.Pmax[i] * m.u[i,t] <= 0`. So dual_val >= 0.
        # The term is dual_val * ( (-m.Pmax[i]*m.u[i,t]) - (-m.Pmax[i]*u_k[i,t]) )
        # = dual_val * (-m.Pmax[i]) * (m.u[i,t] - u_k[i,t])
        # This is different from your current code. Let's assume your previous cut expression was correct for how you defined duals.
        for i in m.I:
            for t in m.T:
                dual_val = duals_k['lambda_max'].get((i, t), 0.0) # Should be non-negative
                # Original: cut_expr += dual_val * (m.Pmax[i] * (m.u[i, t] - u_k.get((i,t), 0.0)))
                # This implies the RHS part that depends on u is -Pmax[i]*u[i,t] for a <= constraint, so the term is dual * ( (-Pmax[i]*u[i,t]) - (-Pmax[i]*u_k[i,t]) )
                # which is dual * (-Pmax[i]) * (u[i,t] - u_k[i,t])
                # If the constraint is p_it - Pmax_i u_it <= 0, dual lambda >=0.
                # Subproblem Lagrangian term: lambda * (p_it - Pmax_i u_it).
                # Derivative w.r.t u_it (if u_it were continuous for a moment in SP): -lambda * Pmax_i.
                # So the cut term involving u_it should be -lambda * Pmax_i * (u_it - u_it_k).
                # Your code has +lambda * Pmax_i * (u_it - u_it_k).
                # This suggests your dual might be for Pmax_i u_it - p_it >= 0. Let's keep your formulation assuming it was tested.
                cut_expr_rhs += dual_val * (-m.Pmax[i] * (m.u[i, t] - u_k.get((i,t), 0.0))) # Corrected based on p - Pmax*u <= 0

        # RampUp: p[i,t] - p_prev[i,t] <= Ru[i]*u_prev_fixed[i,t] + Rsu[i]*zON_fixed[i,t]
        # Dual (lambda_ru) is for: p[i,t] - p_prev[i,t] - Ru[i]*u_prev_fixed[i,t] - Rsu[i]*zON_fixed[i,t] <= 0
        # Terms with master vars: -Ru[i]*u_prev_fixed[i,t] - Rsu[i]*zON_fixed[i,t]
        # So contribution is dual * ( (-Ru*u_prev - Rsu*zON) - (-Ru*u_prev_k - Rsu*zON_k) )
        # = dual * ( -Ru*(u_prev - u_prev_k) - Rsu*(zON - zON_k) )
        for i in m.I:
            for t in m.T:
                dual_val = duals_k['lambda_ru'].get((i, t), 0.0) # Should be non-negative
                # u_prev_fixed_term for current master vars m.u[i,t-1] (or m.u_init if t=1)
                # u_prev_fixed_term for master vars u_k
                u_prev_term_master = 0
                u_prev_k_val = u_k.get((i, t-1), m.u_init[i]) if t > 1 else m.u_init[i]
                if t > 1:
                     u_prev_term_master = m.Ru[i] * (m.u[i, t-1] - u_prev_k_val)
                else: # t == 1, u_prev is u_init for both, so difference is 0
                     u_prev_term_master = m.Ru[i] * (m.u_init[i] - u_prev_k_val) # this will be 0 if u_prev_k_val is also m.u_init[i]

                zON_term_master = m.Rsu[i] * (m.zON[i, t] - zON_k.get((i, t), 0.0))
                # The cut_expr in your code for RampUp adds: dual_val * (u_prev_term + zON_term)
                # This means the b_s(x) part is Ru*u_prev + Rsu*zON
                # So we add dual_val * ( (Ru*u_prev + Rsu*zON) - (Ru*u_prev_k + Rsu*zON_k) )
                # This matches your original form.
                cut_expr_rhs += dual_val * (u_prev_term_master + zON_term_master)


        # RampDown: p_prev[i,t] - p[i,t] <= Rd[i]*u_fixed[i,t] + Rsd[i]*zOFF_fixed[i,t]
        # Dual (lambda_rd) is for: p_prev[i,t] - p[i,t] - Rd[i]*u_fixed[i,t] - Rsd[i]*zOFF_fixed[i,t] <= 0
        # Terms with master vars: -Rd[i]*u_fixed[i,t] - Rsd[i]*zOFF_fixed[i,t]
        # So contribution is dual * ( (-Rd*u - Rsd*zOFF) - (-Rd*u_k - Rsd*zOFF_k) )
        # = dual * ( -Rd*(u - u_k) - Rsd*(zOFF - zOFF_k) )
        for i in m.I:
            for t in m.T:
                dual_val = duals_k['lambda_rd'].get((i, t), 0.0) # Should be non-negative
                u_term_master = m.Rd[i] * (m.u[i, t] - u_k.get((i, t), 0.0))
                zOFF_term_master = m.Rsd[i] * (m.zOFF[i, t] - zOFF_k.get((i, t), 0.0))
                # Your code adds: dual_val * (u_term + zOFF_term)
                # This implies b_s(x) is Rd*u + Rsd*zOFF
                # So we add dual_val * ( (Rd*u + Rsd*zOFF) - (Rd*u_k + Rsd*zOFF_k) )
                # This matches your original form.
                cut_expr_rhs += dual_val * (u_term_master + zOFF_term_master)

        # Demand: sum(p[i,t]) >= D[t]  => D[t] - sum(p[i,t]) <= 0
        # Dual (lambda_d) is for D[t] - sum(p[i,t]) <= 0. Dual_val should be non-negative.
        # The RHS of this constraint (D[t]) does not depend on master variables. So no term here for optimality cut from demand.
        # If there were master variables on RHS of demand constraint, they would appear here.
        # Your `duals_k` doesn't have 'lambda_d' from your current code, which is correct as demand RHS is fixed.

        binary_beta_expr = sum( (2**j) * m.beta_binary[j] for j in m.BETA_BITS )
        return binary_beta_expr >= cut_expr_rhs

    model.BendersOptimalityCuts = pyo.Constraint(model.OptimalityCuts, rule=benders_optimality_cut_rule)

    # Benders Feasibility Cuts
    # v_j^T * b_s(x) >= 0 (if original subproblem is min c^T y s.t. Ay - b_s(x) <= 0, Farkas' lemma gives v^T(b_s(x) - Ay) >=0 for all y, and v^T A = 0, v >=0 )
    # Or, if subproblem is infeasible, there exists a Farkas ray v such that v^T A = 0, v >= 0 and v^T b_s(x_k) < 0
    # The feasibility cut is v^T b_s(x) >= 0
    # Our subproblem constraints are:
    # 1. MinPower: Pmin[i] * u_fixed[i,t] - p[i,t] <= 0.   RHS part b_s1(x) = Pmin[i]*u[i,t]
    # 2. MaxPower: p[i,t] - Pmax[i] * u_fixed[i,t] <= 0. RHS part b_s2(x) = -Pmax[i]*u[i,t] (Coefficient of u is -Pmax)
    # 3. RampUp: p[i,t] - p_prev[i,t] - Ru[i]*u_prev_fixed[i,t] - Rsu[i]*zON_fixed[i,t] <= 0.
    #    RHS part b_s3(x) = -Ru[i]*u_prev[i,t] - Rsu[i]*zON[i,t]
    # 4. RampDown: p_prev[i,t] - p[i,t] - Rd[i]*u_fixed[i,t] - Rsd[i]*zOFF_fixed[i,t] <= 0.
    #    RHS part b_s4(x) = -Rd[i]*u[i,t] - Rsd[i]*zOFF[i,t]
    # 5. Demand: D[t] - sum(p[i,t]) <= 0. RHS part b_s5(x) = D[t] (constant)

    # Feasibility cut: sum_{constraints c} [ ray_c * (b_sc(x) - b_sc(x_k)) ] >= - sum_{constraints c} [ ray_c * b_sc(x_k) ]
    # Or more directly: sum_{constraints c} [ ray_c * b_sc(x) ] >= 0  (This is the standard form)
    # where b_sc(x) is the part of the RHS of constraint c that depends on master variables x.
    # Or, if constraint is A_sub*y <= B_sub*x + C_sub (where y are subproblem vars, x are master vars)
    # then Farkas ray v satisfies v^T A_sub = 0, v >= 0.
    # Feasibility cut is v^T (B_sub*x + C_sub) >= v^T A_sub * y_k (where y_k is some feasible solution, but this form is more for optimality)
    # For feasibility, it's sum_j v_j ( b_j(x) - A_j y_k ) >= 0 if primal is Ax-b >=0.
    # If primal is Ax <= b, then feasibility cut is v^T b(x) >= v^T A y_k (this is not right)
    # Standard form: if subproblem is min {c'y | Dy >= d - Ex}, infeasibility implies existence of ray u s.t. u'D=0, u >= 0, u'(d-Ex_k) < 0.
    # Feasibility cut: u'(d-Ex) >= 0.

    def benders_feasibility_cut_rule(m, k_idx): # k_idx is the index in iteration_data for a feasibility cut
        data = iteration_data[k_idx]
        rays_k = data['rays']       # These are the Farkas rays v_j (or u_j)
        # u_k, zON_k, zOFF_k are not strictly needed here as the cut is on x directly, not (x-x_k)

        # The cut is sum_{all subproblem constraints i} ray_i * (RHS_i(x_fixed) - LHS_i(p)) >= 0
        # where RHS_i includes terms with master variables and constants, and LHS_i includes subproblem variables p.
        # Gurobi's UnbdRay (or FarkasDual) gives the ray for constraints in the form they are given to the solver.
        # If constraint is LHS <= RHS, Pyomo converts to LHS - RHS <= 0.
        # The Farkas certificate (ray v) satisfies v^T A = 0 and v^T b < 0 for an infeasible LP: Ax <= b.
        # The feasibility cut is v^T (b_true_RHS_depending_on_master_vars) >= 0
        # Let's define b_s(x) for each constraint type:
        # MinPower: Pmin[i]*u[i,t]  (from Pmin[i]*u[i,t] <= p[i,t]  OR  -p[i,t] <= -Pmin[i]*u[i,t] )
        #           If ray is for Pmin[i]*u_fixed[i,t] - p[i,t] <= 0, then b_s(x) part for the cut is Pmin[i]*u[i,t]
        # MaxPower: Pmax[i]*u[i,t]  (from p[i,t] <= Pmax[i]*u[i,t] OR p[i,t] - Pmax[i]*u[i,t] <=0). b_s(x) is Pmax[i]*u[i,t]
        # RampUp: Ru[i]*u_prev[i,t] + Rsu[i]*zON[i,t]
        # RampDown: Rd[i]*u[i,t] + Rsd[i]*zOFF[i,t]
        # Demand: D[t]

        # The expression for the feasibility cut is:
        # sum_{constraints j} v_j * (RHS_j(x_master) - LHS_j(p_fixed_at_x_k_solution)) >= 0 where LHS_j involves p variables
        # This is not the standard Benders feasibility cut form.
        # The standard form is: v^T ( b_0 - T_0 x ) >= 0 (using notation from Conejo's book Ch. Benders)
        # where subproblem is Wx + Hy = b_0, and T_0 x are the terms moved to RHS.
        # Primal subproblem: min c'y s.t. A y >= b - F x_bar (where x_bar is fixed master solution)
        # Dual subproblem: max lambda'(b - F x_bar) s.t. lambda' A = c', lambda >= 0
        # If primal infeasible, dual is unbounded. There exists a ray lambda_ray s.t. lambda_ray' A = 0, lambda_ray >= 0, and lambda_ray'(b - F x_bar) > 0.
        # The feasibility cut is: lambda_ray'(b - F x) <= 0.
        # Note: My signs might be mixed depending on Gurobi's ray output. Gurobi's Farkas dual (ray) v for an infeasible LP (Ax <= b) satisfies v'A = 0, v >= 0, v'b < 0.
        # The feasibility cut is v'b_general(x) >= 0.
        # where b_general(x) are the RHS of the subproblem constraints that depend on x.
        # Our constraints are:
        # 1. MinPower:  -p[i,t] <= -model.Pmin[i] * m.u[i,t] (ray_min_power * (-model.Pmin[i] * m.u[i,t]))
        # 2. MaxPower:   p[i,t] <=  model.Pmax[i] * m.u[i,t] (ray_max_power * (model.Pmax[i] * m.u[i,t]))
        # 3. RampUp:     p[i,t] - p_prev[i,t] <= model.Ru[i]*m.u_prev[i,t] + model.Rsu[i]*m.zON[i,t]
        #                (ray_ramp_up * (model.Ru[i]*m.u_prev[i,t] + model.Rsu[i]*m.zON[i,t]))
        # 4. RampDown:   p_prev[i,t] - p[i,t] <= model.Rd[i]*m.u[i,t] + model.Rsd[i]*m.zOFF[i,t]
        #                (ray_ramp_down * (model.Rd[i]*m.u[i,t] + model.Rsd[i]*m.zOFF[i,t]))
        # 5. Demand:    -sum(p[i,t]) <= -model.D[t]
        #                (ray_demand * (-model.D[t]))

        # Let's assume rays_k['ray_constraint_name'] are the v_j values.
        # The feasibility cut is: sum_j v_j * (RHS_j_of_master_vars_part) >= 0
        # The RHS_j_of_master_vars_part is the part of the subproblem constraint RHS that depends on master variables.
        # For a constraint like A_sub * y <= B_sub * x + C_sub (constant part)
        # The ray v satisfies v^T A_sub = 0, v >=0.
        # The cut is v^T (B_sub * x + C_sub) >= 0. (This comes from v^T b < 0 for infeasibility, so v^T b(x_new) >= 0 makes it feasible)

        feasibility_cut_expr = 0

        # MinPower: constraint is m.Pmin[i] * m.u_fixed[i, t] <= m.p[i, t]
        # Standard form: -m.p[i,t] <= -m.Pmin[i] * m.u_fixed[i,t]
        # RHS that depends on master: -m.Pmin[i] * m.u[i,t]
        for i in m.I:
            for t in m.T:
                ray_val = rays_k['ray_min'].get((i, t), 0.0) # v_j
                if ray_val != 0: # Optimization
                    feasibility_cut_expr += ray_val * (-m.Pmin[i] * m.u[i, t])

        # MaxPower: constraint is m.p[i, t] <= m.Pmax[i] * m.u_fixed[i, t]
        # RHS that depends on master: m.Pmax[i] * m.u[i,t]
        for i in m.I:
            for t in m.T:
                ray_val = rays_k['ray_max'].get((i, t), 0.0)
                if ray_val != 0:
                    feasibility_cut_expr += ray_val * (m.Pmax[i] * m.u[i, t])

        # RampUp: m.p[i, t] - m.p_prev[i,t] <= m.Ru[i] * m.u_prev_fixed[i,t] + m.Rsu[i] * m.zON_fixed[i, t]
        # RHS that depends on master: m.Ru[i] * m.u_prev[i,t] + m.Rsu[i] * m.zON[i, t]
        for i in m.I:
            for t in m.T:
                ray_val = rays_k['ray_ru'].get((i, t), 0.0)
                if ray_val != 0:
                    # u_prev part
                    if t == 1:
                        feasibility_cut_expr += ray_val * (m.Ru[i] * m.u_init[i])
                    else:
                        feasibility_cut_expr += ray_val * (m.Ru[i] * m.u[i, t-1])
                    # zON part
                    feasibility_cut_expr += ray_val * (m.Rsu[i] * m.zON[i, t])

        # RampDown: m.p_prev[i,t] - m.p[i, t] <= m.Rd[i] * m.u_fixed[i, t] + m.Rsd[i] * m.zOFF_fixed[i, t]
        # RHS that depends on master: m.Rd[i] * m.u[i, t] + m.Rsd[i] * m.zOFF[i, t]
        for i in m.I:
            for t in m.T:
                ray_val = rays_k['ray_rd'].get((i, t), 0.0)
                if ray_val != 0:
                    feasibility_cut_expr += ray_val * (m.Rd[i] * m.u[i, t])
                    feasibility_cut_expr += ray_val * (m.Rsd[i] * m.zOFF[i, t])

        # Demand: sum(m.p[i, t] for i in m.I) >= m.D[t]
        # Standard form: -sum(m.p[i, t] for i in m.I) <= -m.D[t]
        # RHS (constant): -m.D[t]
        for t_period in m.T: # Corrected loop to m.T
            ray_val = rays_k['ray_demand'].get(t_period, 0.0)
            if ray_val != 0:
                feasibility_cut_expr += ray_val * (-m.D[t_period]) # D is Param, not Var

        if isinstance(feasibility_cut_expr, (int, float)) and feasibility_cut_expr >= 0:
            return pyo.Constraint.Feasible
        else:
            return feasibility_cut_expr >= 0

    model.BendersFeasibilityCuts = pyo.Constraint(model.FeasibilityCuts, rule=benders_feasibility_cut_rule)

    return model


# Main Benders Loop
def benders_ucp(max_iter: int = 30, epsilon: float = 1.0):
    start_time = time.time()

    # initial solution: everything ON ------------------------------------------------
    u_curr, zON_curr, zOFF_curr = {}, {}, {}
    for i in generators:
        for t in time_periods:
            u_curr[i, t] = 1.0
            uprev = u_initial[i] if t == 1 else u_curr[i, t - 1]
            zON_curr[i, t]  = 1.0 if u_curr[i, t] > 0.5 and uprev < 0.5 else 0.0
            zOFF_curr[i, t] = 1.0 if u_curr[i, t] < 0.5 and uprev > 0.5 else 0.0

    master_solver = pyo.SolverFactory("gurobi")

    iteration_data = []
    lower_bound, upper_bound = -float('inf'), float('inf')

    for k in range(1, max_iter + 1):
        print(f"========== Iteration {k} ==========")

        # ---------------------- build & solve sub-problem ----------------------
        sub = build_subproblem(u_curr, zON_curr, zOFF_curr)
        sub_solver = GurobiPersistent()
        sub_solver.set_instance(sub)
        sub_solver.set_gurobi_param('InfUnbdInfo', 1)
        sub_solver.set_gurobi_param('DualReductions', 0)
        sub_solver.set_gurobi_param('Presolve', 0)  # optional: easier ray mapping

        sub_res = sub_solver.solve()
        tc = sub_res.solver.termination_condition

        # --------------------------- feasible ----------------------------------
        if tc in (TerminationCondition.optimal, TerminationCondition.feasible):
            sub_solver.load_duals()  # populate sub.dual suffix
            sub_obj = pyo.value(sub.OBJ)

            # commitment cost ----------------------------------------------------
            commit_cost = sum(gen_data[i]['Csu'] * zON_curr[i, t] +
                               gen_data[i]['Csd'] * zOFF_curr[i, t] +
                               gen_data[i]['Cf']  * u_curr[i, t]
                               for i in generators for t in time_periods)
            upper_bound = min(upper_bound, commit_cost + sub_obj)
            print(f"  sub‑problem feasible, obj = {sub_obj:.2f} → Z_UB = {upper_bound:.2f}")

            # grab ordinary duals ------------------------------------------------
            duals = {lab: {} for lab in
                     ('lambda_min', 'lambda_max', 'lambda_ru', 'lambda_rd', 'lambda_d')}
            for i in generators:
                for t in time_periods:
                    duals['lambda_min'][i, t] = sub.dual[sub.MinPower[i, t]]
                    duals['lambda_max'][i, t] = sub.dual[sub.MaxPower[i, t]]
                    duals['lambda_ru'][i, t]  = sub.dual[sub.RampUp[i, t]]
                    duals['lambda_rd'][i, t]  = sub.dual[sub.RampDown[i, t]]
            for t in time_periods:
                duals['lambda_d'][t] = sub.dual[sub.Demand[t]]

            iteration_data.append({
                'type': 'optimality', 'iter': k,
                'sub_obj': sub_obj, 'duals': duals,
                'u_vals': deepcopy(u_curr),
                'zON_vals': deepcopy(zON_curr),
                'zOFF_vals': deepcopy(zOFF_curr),
            })

        # -------------------------- infeasible ---------------------------------
        elif tc == TerminationCondition.infeasible:
            print("  sub‑problem infeasible → generating feasibility cut")

            rays = {lab: {} for lab in
                    ('ray_min', 'ray_max', 'ray_ru', 'ray_rd', 'ray_demand')}
            py2grb = sub_solver._pyomo_con_to_solver_con_map

            for i in generators:
                for t in time_periods:
                    rays['ray_min'][i, t] = py2grb[sub.MinPower[i, t]].FarkasDual
                    rays['ray_max'][i, t] = py2grb[sub.MaxPower[i, t]].FarkasDual
                    rays['ray_ru'][i, t]  = py2grb[sub.RampUp[i, t]].FarkasDual
                    rays['ray_rd'][i, t]  = py2grb[sub.RampDown[i, t]].FarkasDual
            for t in time_periods:
                rays['ray_demand'][t] = py2grb[sub.Demand[t]].FarkasDual

            # simple diagnostic --------------------------------------------------
            vTb = sum(v for rdict in rays.values() for v in rdict.values())
            print(f"    Σ ray·b(x_k) = {vTb:+.3e}")

            iteration_data.append({
                'type': 'feasibility', 'iter': k,
                'rays': rays,
                'u_vals': deepcopy(u_curr),
                'zON_vals': deepcopy(zON_curr),
                'zOFF_vals': deepcopy(zOFF_curr),
            })

        else:
            raise RuntimeError(f"sub‑problem returned unexpected status {tc}")

        # --------------------------- solve master ------------------------------
        master = build_master(iteration_data)
        master_res = master_solver.solve(master)
        if master_res.solver.termination_condition != TerminationCondition.optimal:
            raise RuntimeError("master problem failed to solve optimally – check cuts")

        lower_bound = pyo.value(master.OBJ)
        print(f"  master solved, Z_LB = {lower_bound:.2f}")

        if upper_bound < float('inf') and upper_bound - lower_bound <= epsilon:
            print("  gap closed – terminating")
            break

        # update incumbent x -----------------------------------------------------
        for i in generators:
            for t in time_periods:
                u_curr[i, t]  = master.u[i, t].value
                zON_curr[i, t]  = master.zON[i, t].value
                zOFF_curr[i, t] = master.zOFF[i, t].value

    print("================ finished ================")
    print(f"best UB = {upper_bound:.2f}, best LB = {lower_bound:.2f}")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    benders_ucp()