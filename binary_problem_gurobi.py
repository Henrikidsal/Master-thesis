
##### This code uses gurobi to solve the all binary problem.
##### It uses the slack method, and have no trouble because of the exact solvers nature.
##### It finds the optimal solution for the problem (215.0)


import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB

''''''''''''' The code changes the problem from "summation QUBO" into standard QUBO form '''''''''''''

T = 3  # Number of time periods
N = 3   # Number of units

# --- ADDED: Initial status from t=0 ---
# Previous time period status (initial conditions):
# Unit 1: off, Unit 2: off, Unit 3: on.
u_prev = {1: 0, 2: 0, 3: 1}
# ---------------------------------------

# Global penalty factors (one per constraint)
penalty_factors = {
    "unit_transition": 100,
    "mutual_exclusion": 100,
    "demand": 100,
    "ramp_up": 100,
    "ramp_down": 100
}

# Parameter dictionaries
P_max = {1:350, 2:200, 3:140}
D = {1: 160, 2: 500, 3: 400}
C_startup = {1:20, 2:18, 3:5}
C_shutdown = {1:0.5, 2:0.3, 3:1.0}
b_cost = {1:0.1, 2:0.125, 3:0.15}
c_cost = {1:5, 2:7, 3:6}
R_up = {1:350, 2:200, 3:140}
R_down = {1:350, 2:200, 3:140}

# ---------------------------
# Determine Dynamic Bit-lengths for Slack Variables
# ---------------------------
# For the demand constraint at time t: slack = (sum_i P_max[i]*u_i) - D[t]
demand_K = {}  # For each time period t, the number of bits needed for the demand slack.
total_Pmax = sum(P_max.values())
for t in range(1, T+1):
    max_slack = total_Pmax - D[t]
    if max_slack < 0:
         print(f"Warning: Demand at t={t} ({D[t]}) might be unsatisfiable even with all units on ({total_Pmax}). Setting max_slack to 0.")
         max_slack = 0 # Slack must be non-negative
    demand_K[t] = int(math.ceil(math.log2(max_slack + 1))) if max_slack >= 0 else 0 # Handle potential max_slack= -1 edge case if log2(0)
    print(f"Time {t}: Demand slack max = {max_slack}, bits = {demand_K[t]}")

# For ramp-up: slack = R_up[i] - P_max[i]*(u_{i,t} - u_{i,t-1}). Max value is R_up[i] + P_max[i] (when u_t=0, u_{t-1}=1)
max_slack_ramp_up = max(R_up[i] + P_max[i] for i in P_max)
K_ramp_up = int(math.ceil(math.log2(max_slack_ramp_up + 1)))
print("Ramp-up slack bits (uniform for all units):", K_ramp_up)

# For ramp-down: slack = R_down[i] - P_max[i]*(u_{i,t-1} - u_{i,t}). Max value is R_down[i] + P_max[i] (when u_t=1, u_{t-1}=0)
max_slack_ramp_down = max(R_down[i] + P_max[i] for i in P_max)
K_ramp_down = int(math.ceil(math.log2(max_slack_ramp_down + 1)))
print("Ramp-down slack bits (uniform for all units):", K_ramp_down)

# ---------------------------
# Create a Mapping for Variables
# ---------------------------
# Each variable (including slack bits) is assigned a unique index in the vector x.
variable_mapping = {}
current_index = 0

def add_variable(name):
    """Add a variable to the mapping and assign a unique index."""
    global current_index
    variable_mapping[name] = current_index
    current_index += 1

# 1. Unit status variables: u_{i,t} for t=1..T
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"u_{i}_{t}")

# 2. Startup decision variables: zOn_{i,t} for t=1..T
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOn_{i}_{t}")

# 3. Shutdown decision variables: zOff_{i,t} for t=1..T
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOff_{i}_{t}")

# 4. Slack variables for demand constraint: s_{t,k} for t=1..T
if penalty_factors["demand"] != 0:
    for t in range(1, T+1):
        for k in range(demand_K[t]):
            add_variable(f"s_{t}_{k}")

# 5. Slack variables for ramp-up constraints: sUp_{t,i,k} for t=1..T
if penalty_factors["ramp_up"] != 0:
    # Ramp constraints apply from t=1 (using u_prev) up to t=T
    for t in range(1, T+1):
        for i in range(1, N+1):
            for k in range(K_ramp_up):
                add_variable(f"sUp_{t}_{i}_{k}")

# 6. Slack variables for ramp-down constraints: sDown_{t,i,k} for t=1..T
if penalty_factors["ramp_down"] != 0:
    # Ramp constraints apply from t=1 (using u_prev) up to t=T
    for t in range(1, T+1):
        for i in range(1, N+1):
            for k in range(K_ramp_down):
                add_variable(f"sDown_{t}_{i}_{k}")

num_variables = current_index
print("Total number of variables:", num_variables)

# ---------------------------
# Initialize Q Matrix and Constant Term
# ---------------------------
Q = np.zeros((num_variables, num_variables))
c_term = 0

# ---------------------------
# Helper Functions to Update Q and c_term
# ---------------------------
def add_quadratic_term(i, j, coeff):
    """
    Add a quadratic term: coeff * x_i * x_j.
    For off-diagonal entries, add half the coefficient to each to preserve symmetry.
    """
    if i == j:
        Q[i, i] += coeff
    else:
        Q[i, j] += coeff / 2.0
        Q[j, i] += coeff / 2.0

def add_linear_term(i, coeff):
    """
    Add a linear term: coeff * x_i.
    (For binary variables, x_i^2 = x_i, so add to the diagonal.)
    """
    Q[i, i] += coeff

def add_constant_term(value):
    """Add a constant term to the overall constant."""
    global c_term
    c_term += value

def add_penalty_term(term_list, constant_term, weight):
    """
    Adds the penalty term weight * (sum(coeff_k * x_k) + constant_term)^2 to the QUBO.
    term_list is a list of tuples (variable_index, coefficient).
    """
    # Quadratic terms: coeff_i * coeff_j * x_i * x_j
    for idx1, coeff1 in term_list:
        for idx2, coeff2 in term_list:
            add_quadratic_term(idx1, idx2, weight * coeff1 * coeff2)

    # Linear terms: 2 * constant_term * coeff_k * x_k
    for idx, coeff in term_list:
        add_linear_term(idx, weight * 2 * coeff * constant_term)

    # Constant term: constant_term^2
    add_constant_term(weight * constant_term * constant_term)

# ---------------------------
# Add Cost Function Terms (Linear Terms)
# ---------------------------
# Costs are incurred from t=1 to T
for t in range(1, T+1):
    for i in range(1, N+1):
        idx_u = variable_mapping[f"u_{i}_{t}"]
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]

        # Production cost = (b_cost * P_max + c_cost) * u_{i,t}
        cost_u = b_cost[i] * P_max[i] + c_cost[i]
        add_linear_term(idx_u, cost_u)

        # Startup cost = C_startup * zOn_{i,t}
        add_linear_term(idx_zOn, C_startup[i])

        # Shutdown cost = C_shutdown * zOff_{i,t}
        add_linear_term(idx_zOff, C_shutdown[i])

# ---------------------------
# Add Unit Status Transition Constraint
# For t=1: u_{i,1} - u_prev[i] - zOn_{i,1} + zOff_{i,1} = 0
# For t>=2: u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t} = 0.
# QUBO form: penalty * (expression)^2
if penalty_factors["unit_transition"] != 0:
    penalty = penalty_factors["unit_transition"]
    for i in range(1, N+1):
        # Case t = 1
        idx_u_1 = variable_mapping[f"u_{i}_1"]
        idx_zOn_1 = variable_mapping[f"zOn_{i}_1"]
        idx_zOff_1 = variable_mapping[f"zOff_{i}_1"]
        terms_t1 = [(idx_u_1, 1.0),
                    (idx_zOn_1, -1.0),
                    (idx_zOff_1, 1.0)]
        constant_t1 = -u_prev[i] # Correct: (u1 - zOn1 + zOff1) - u_prev = 0
        add_penalty_term(terms_t1, constant_t1, penalty)

        # Case t >= 2
        for t in range(2, T+1):
            idx_u_t = variable_mapping[f"u_{i}_{t}"]
            idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
            idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
            idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
            terms_t_ge_2 = [(idx_u_t, 1.0),
                            (idx_u_prev, -1.0),
                            (idx_zOn, -1.0),
                            (idx_zOff, 1.0)]
            constant_t_ge_2 = 0 # Correct: (u_t - u_{t-1} - zOn + zOff) = 0
            add_penalty_term(terms_t_ge_2, constant_t_ge_2, penalty)


# ---------------------------
# Add Mutual Exclusion Constraint: zOn_{i,t} * zOff_{i,t} = 0
# QUBO penalty term is penalty * zOn_{i,t} * zOff_{i,t}
if penalty_factors["mutual_exclusion"] != 0:
    penalty = penalty_factors["mutual_exclusion"]
    for t in range(1, T+1):
        for i in range(1, N+1):
            idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
            idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
            add_quadratic_term(idx_zOn, idx_zOff, penalty) # Correct

# ---------------------------
# Add Demand Satisfaction Constraint
# sum_i (P_max[i] * u_{i,t}) - sum_k (s_{t,k} * 2^k) = D[t]
# QUBO form: penalty * (expression - D[t])^2
if penalty_factors["demand"] != 0:
    penalty = penalty_factors["demand"]
    for t in range(1, T+1):
        term_list = []
        # Add production terms
        for i in range(1, N+1):
            idx_u = variable_mapping[f"u_{i}_{t}"]
            term_list.append((idx_u, P_max[i]))
        # Add slack terms
        for k in range(demand_K[t]):
            idx_s = variable_mapping[f"s_{t}_{k}"]
            term_list.append((idx_s, -(2 ** k)))
        # Constant term is -D[t]
        constant_val = -D[t] # Correct: (sum(P*u) - sum(s*2^k)) - D = 0
        add_penalty_term(term_list, constant_val, penalty)

# ---------------------------
# Add Ramp-Up Constraint
# Target EQUALITY (after adding slack):
# For t=1: P_max[i]*(u_{i,1} - u_prev[i]) + sum_k (sUp_{1,i,k} * 2^k) = R_up[i]
#          => ( P_max[i]*u_{i,1} + sum_k(sUp*2^k) ) + ( -P_max[i]*u_prev[i] - R_up[i] ) = 0
# For t>=2: P_max[i]*(u_{i,t} - u_{i,t-1}) + sum_k (sUp_{t,i,k} * 2^k) = R_up[i]
#          => ( P_max[i]*u_{i,t} - P_max[i]*u_{i,t-1} + sum_k(sUp*2^k) ) + ( -R_up[i] ) = 0
# QUBO form: penalty * ( sum(variable_terms) + constant_term )^2
if penalty_factors["ramp_up"] != 0:
    penalty = penalty_factors["ramp_up"]
    for i in range(1, N+1):
        # Case t = 1
        term_list_t1 = []
        idx_u_1 = variable_mapping[f"u_{i}_1"]
        term_list_t1.append((idx_u_1, P_max[i])) # P_max[i]*u_{i,1}
        for k in range(K_ramp_up):
            idx_sUp = variable_mapping[f"sUp_1_{i}_{k}"] # Slack for t=1
            term_list_t1.append((idx_sUp, 2 ** k))      # + sum(sUp*2^k)

        # *** CORRECTED CONSTANT TERM FOR t=1 ***
        constant_t1 = - P_max[i] * u_prev[i] - R_up[i]
        add_penalty_term(term_list_t1, constant_t1, penalty)

        # Case t >= 2
        for t in range(2, T+1):
            term_list_t_ge_2 = []
            idx_u_t = variable_mapping[f"u_{i}_{t}"]
            idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
            term_list_t_ge_2.append((idx_u_t, P_max[i]))     # P_max[i]*u_{i,t}
            term_list_t_ge_2.append((idx_u_prev, -P_max[i])) # -P_max[i]*u_{i,t-1}
            for k in range(K_ramp_up):
                idx_sUp = variable_mapping[f"sUp_{t}_{i}_{k}"] # Slack for t
                term_list_t_ge_2.append((idx_sUp, 2 ** k))   # + sum(sUp*2^k)
            # Constant term: -R_up[i]
            constant_t_ge_2 = -R_up[i] # Correct
            add_penalty_term(term_list_t_ge_2, constant_t_ge_2, penalty)

# ---------------------------
# Add Ramp-Down Constraint
# Target EQUALITY (after adding slack):
# For t=1: P_max[i]*(u_prev[i] - u_{i,1}) + sum_k (sDown_{1,i,k} * 2^k) = R_down[i]
#          => ( -P_max[i]*u_{i,1} + sum_k(sDown*2^k) ) + ( P_max[i]*u_prev[i] - R_down[i] ) = 0
# For t>=2: P_max[i]*(u_{i,t-1} - u_{i,t}) + sum_k (sDown_{t,i,k} * 2^k) = R_down[i]
#          => ( P_max[i]*u_{i,t-1} - P_max[i]*u_{i,t} + sum_k(sDown*2^k) ) + ( -R_down[i] ) = 0
# QUBO form: penalty * ( sum(variable_terms) + constant_term )^2
if penalty_factors["ramp_down"] != 0:
    penalty = penalty_factors["ramp_down"]
    for i in range(1, N+1):
         # Case t = 1
        term_list_t1 = []
        idx_u_1 = variable_mapping[f"u_{i}_1"]
        term_list_t1.append((idx_u_1, -P_max[i])) # -P_max[i]*u_{i,1}
        for k in range(K_ramp_down):
            idx_sDown = variable_mapping[f"sDown_1_{i}_{k}"] # Slack for t=1
            term_list_t1.append((idx_sDown, 2 ** k))       # + sum(sDown*2^k)
        # Constant term
        constant_t1 = P_max[i] * u_prev[i] - R_down[i] # Correct
        add_penalty_term(term_list_t1, constant_t1, penalty)

        # Case t >= 2
        for t in range(2, T+1):
            term_list_t_ge_2 = []
            idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
            idx_u_t = variable_mapping[f"u_{i}_{t}"]
            term_list_t_ge_2.append((idx_u_prev, P_max[i]))   # P_max[i]*u_{i,t-1}
            term_list_t_ge_2.append((idx_u_t, -P_max[i]))     # -P_max[i]*u_{i,t}
            for k in range(K_ramp_down):
                idx_sDown = variable_mapping[f"sDown_{t}_{i}_{k}"] # Slack for t
                term_list_t_ge_2.append((idx_sDown, 2 ** k))   # + sum(sDown*2^k)
            # Constant term
            constant_t_ge_2 = -R_down[i] # Correct
            add_penalty_term(term_list_t_ge_2, constant_t_ge_2, penalty)

# ---------------------------
# Final Output (Optional)
# print("Final Q matrix (dense format):")
# print(Q)
print("Shape of Q matrix:", Q.shape)
print("\nConstant term c:")
print(c_term)
print("Is the matrix symmetric?", np.allclose(Q, Q.T))

''''''''''''' Solving the QUBO '''''''''''''
# [NO CHANGES NEEDED in solve_qubo, calculate_original_cost, check_feasibility, or __main__ execution block]
def solve_qubo(Q, c):

    n = Q.shape[0]

    # Create a new Gurobi model
    m = gp.Model("qubo")
    m.setParam('OutputFlag', 1)  # Set to 1 to see Gurobi output, 0 to suppress
    # m.setParam('TimeLimit', 120) # Example: Optional time limit

    # Add binary variables x[i] for i in range(n)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # Build the quadratic objective expression: x^T Q x + c
    # Use the explicit loop method for robustness
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0.0: # Avoid adding zero terms
                obj.add(x[i] * Q[i, j] * x[j])

    m.setObjective(obj + c, GRB.MINIMIZE) # Add 'c' to the objective expression

    # Optimize the model
    m.optimize()

    # Retrieve the solution
    if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL or (m.status == GRB.TIME_LIMIT and m.SolCount > 0):
        if m.SolCount > 0: # Check if a solution exists
             sol = np.array([x[i].X for i in range(n)])
             obj_val = m.objVal # Gurobi's objVal includes the constant 'c'

             print(f"Gurobi optimization finished with status: {m.status}")
             print(f"Solution count: {m.SolCount}")
             print(f"Objective value (QUBO): {obj_val}")
             return sol, obj_val
        else:
            print(f"Gurobi finished with status {m.status} but found no feasible solution.")
            return None, None
    else:
        print(f"Gurobi optimization failed or found no solution. Status: {m.status}")
        return None, None


# --- Helper function to calculate original cost ---
def calculate_original_cost(solution_vector, variable_mapping):
    cost = 0
    # Production, Startup, Shutdown costs
    for t in range(1, T+1):
        for i in range(1, N+1):
            idx_u = variable_mapping[f"u_{i}_{t}"]
            idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
            idx_zOff = variable_mapping[f"zOff_{i}_{t}"]

            u_val = round(solution_vector[idx_u]) # Use rounded value for cost calculation
            zOn_val = round(solution_vector[idx_zOn])
            zOff_val = round(solution_vector[idx_zOff])

            cost += u_val * (b_cost[i] * P_max[i] + c_cost[i])
            cost += zOn_val * C_startup[i]
            cost += zOff_val * C_shutdown[i]
    return cost

# --- Helper function to check feasibility ---
def check_feasibility(solution_vector, variable_mapping, u_prev, tolerance=1e-4):
    feasible = True
    print("\n--- Feasibility Check ---")

    # Extract main variables
    u = {}
    zOn = {}
    zOff = {}
    for t in range(1, T+1):
        for i in range(1, N+1):
            u[(i, t)] = round(solution_vector[variable_mapping[f"u_{i}_{t}"]])
            zOn[(i, t)] = round(solution_vector[variable_mapping[f"zOn_{i}_{t}"]])
            zOff[(i, t)] = round(solution_vector[variable_mapping[f"zOff_{i}_{t}"]])

    # 1. Unit Transition
    for i in range(1, N+1):
        # t=1
        lhs = u[(i, 1)] - u_prev[i]
        rhs = zOn[(i, 1)] - zOff[(i, 1)]
        if abs(lhs - rhs) > tolerance:
            print(f" FAILED: Unit Transition Unit {i}, t=1 ({lhs} != {rhs})")
            feasible = False
        # t>=2
        for t in range(2, T+1):
            lhs = u[(i, t)] - u[(i, t-1)]
            rhs = zOn[(i, t)] - zOff[(i, t)]
            if abs(lhs - rhs) > tolerance:
                print(f" FAILED: Unit Transition Unit {i}, t={t} ({lhs} != {rhs})")
                feasible = False

    # 2. Mutual Exclusion
    for t in range(1, T+1):
        for i in range(1, N+1):
            if zOn[(i, t)] + zOff[(i, t)] > 1 + tolerance:
                 print(f" FAILED: Mutual Exclusion Unit {i}, t={t} ({zOn[(i,t)]}+{zOff[(i,t)]} > 1)")
                 feasible = False

    # 3. Demand Satisfaction
    for t in range(1, T+1):
        production = sum(P_max[i] * u[(i, t)] for i in range(1, N+1))
        # Check original constraint: production >= D[t]
        if production < D[t] - tolerance:
             print(f" FAILED: Demand Met t={t} (Prod={production} < Demand={D[t]})")
             feasible = False
        # Check equality with slack (optional, but good for debugging penalties)
        if penalty_factors["demand"] != 0:
            s_t_val = 0
            for k in range(demand_K[t]):
                 idx_s = variable_mapping.get(f"s_{t}_{k}", -1) # Use .get for safety
                 if idx_s != -1:
                    s_t_val += round(solution_vector[idx_s]) * (2**k)
            if abs(production - s_t_val - D[t]) > tolerance:
                print(f" WARNING: Demand Slack Equality t={t} (Prod={production} - Slack={s_t_val} = {production - s_t_val} != Demand={D[t]})")

    # 4. Ramp-Up
    for i in range(1, N+1):
         # t=1
         prod_change = P_max[i] * (u[(i, 1)] - u_prev[i])
         # Check original constraint: prod_change <= R_up[i]
         if prod_change > R_up[i] + tolerance:
              print(f" FAILED: Ramp-Up Unit {i}, t=1 (Change={prod_change} > Limit={R_up[i]})")
              feasible = False
         # Check equality with slack
         if penalty_factors["ramp_up"] != 0:
             s_up_val = 0
             for k in range(K_ramp_up):
                 idx_sUp = variable_mapping.get(f"sUp_1_{i}_{k}", -1)
                 if idx_sUp != -1:
                     s_up_val += round(solution_vector[idx_sUp]) * (2**k)
             if abs(prod_change + s_up_val - R_up[i]) > tolerance:
                 print(f" WARNING: Ramp-Up Slack Equality Unit {i}, t=1 (Change={prod_change} + Slack={s_up_val} = {prod_change + s_up_val} != Limit={R_up[i]})")

         # t>=2
         for t in range(2, T+1):
            prod_change = P_max[i] * (u[(i, t)] - u[(i, t-1)])
            # Check original constraint
            if prod_change > R_up[i] + tolerance:
                 print(f" FAILED: Ramp-Up Unit {i}, t={t} (Change={prod_change} > Limit={R_up[i]})")
                 feasible = False
            # Check equality with slack
            if penalty_factors["ramp_up"] != 0:
                s_up_val = 0
                for k in range(K_ramp_up):
                    idx_sUp = variable_mapping.get(f"sUp_{t}_{i}_{k}", -1)
                    if idx_sUp != -1:
                       s_up_val += round(solution_vector[idx_sUp]) * (2**k)
                if abs(prod_change + s_up_val - R_up[i]) > tolerance:
                    print(f" WARNING: Ramp-Up Slack Equality Unit {i}, t={t} (Change={prod_change} + Slack={s_up_val} = {prod_change + s_up_val} != Limit={R_up[i]})")


    # 5. Ramp-Down
    for i in range(1, N+1):
        # t=1
        prod_change = P_max[i] * (u_prev[i] - u[(i, 1)])
        # Check original constraint
        if prod_change > R_down[i] + tolerance:
             print(f" FAILED: Ramp-Down Unit {i}, t=1 (Change={prod_change} > Limit={R_down[i]})")
             feasible = False
        # Check equality with slack
        if penalty_factors["ramp_down"] != 0:
            s_down_val = 0
            for k in range(K_ramp_down):
                 idx_sDown = variable_mapping.get(f"sDown_1_{i}_{k}", -1)
                 if idx_sDown != -1:
                    s_down_val += round(solution_vector[idx_sDown]) * (2**k)
            if abs(prod_change + s_down_val - R_down[i]) > tolerance:
                 print(f" WARNING: Ramp-Down Slack Equality Unit {i}, t=1 (Change={prod_change} + Slack={s_down_val} = {prod_change+s_down_val} != Limit={R_down[i]})")

        # t>=2
        for t in range(2, T+1):
            prod_change = P_max[i] * (u[(i, t-1)] - u[(i, t)])
            # Check original constraint
            if prod_change > R_down[i] + tolerance:
                 print(f" FAILED: Ramp-Down Unit {i}, t={t} (Change={prod_change} > Limit={R_down[i]})")
                 feasible = False
            # Check equality with slack
            if penalty_factors["ramp_down"] != 0:
                s_down_val = 0
                for k in range(K_ramp_down):
                    idx_sDown = variable_mapping.get(f"sDown_{t}_{i}_{k}", -1)
                    if idx_sDown != -1:
                        s_down_val += round(solution_vector[idx_sDown]) * (2**k)
                if abs(prod_change + s_down_val - R_down[i]) > tolerance:
                    print(f" WARNING: Ramp-Down Slack Equality Unit {i}, t={t} (Change={prod_change} + Slack={s_down_val} = {prod_change+s_down_val} != Limit={R_down[i]})")

    print("--- Feasibility Check Complete ---")
    return feasible


# --- Main execution ---
if __name__ == "__main__":

    solution, obj_val = solve_qubo(Q, c_term)

    if solution is not None:
        # print("Optimal binary solution vector x:")
        # print(solution)
        print(f"\nOptimal QUBO objective value: {obj_val}")

        # Print the u values
        print("\nUnit status values (u_i_t):")
        u_sol = np.zeros((N, T))
        for t in range(1, T+1):
            for i in range(1, N+1):
                idx = variable_mapping[f"u_{i}_{t}"]
                u_sol[i-1, t-1] = round(solution[idx]) # Store in 0-indexed array
                print(f"u_{i}_{t} = {round(solution[idx])}")
        print("u solution array:")
        print(u_sol)

        # Print zOn/zOff values (Optional)
        print("\nStartup decisions (zOn_i_t):")
        zOn_sol = np.zeros((N, T))
        for t in range(1, T+1):
            for i in range(1, N+1):
                idx = variable_mapping[f"zOn_{i}_{t}"]
                zOn_sol[i-1, t-1] = round(solution[idx])
        print(zOn_sol)

        print("\nShutdown decisions (zOff_i_t):")
        zOff_sol = np.zeros((N, T))
        for t in range(1, T+1):
            for i in range(1, N+1):
                idx = variable_mapping[f"zOff_{i}_{t}"]
                zOff_sol[i-1, t-1] = round(solution[idx])
        print(zOff_sol)

        # Calculate and print original cost
        original_cost = calculate_original_cost(solution, variable_mapping)
        print(f"\nOriginal problem objective cost: {original_cost}")

        # Check feasibility
        is_feasible = check_feasibility(solution, variable_mapping, u_prev)
        print(f"\nSolution is feasible regarding original constraints: {is_feasible}")

        # Optional: Print slack variable values
        print("\nSlack Variable Values (decoded):")
        # Demand Slack
        if penalty_factors["demand"] != 0:
            for t in range(1, T+1):
                 s_t_val = 0
                 for k in range(demand_K[t]):
                      idx_s = variable_mapping.get(f"s_{t}_{k}", -1)
                      if idx_s != -1:
                          s_t_val += round(solution[idx_s]) * (2**k)
                 print(f" Demand Slack s_{t}: {s_t_val}")
        # Ramp-Up Slack
        if penalty_factors["ramp_up"] != 0:
             for t in range(1, T+1):
                 for i in range(1, N+1):
                      s_up_val = 0
                      for k in range(K_ramp_up):
                          idx_sUp = variable_mapping.get(f"sUp_{t}_{i}_{k}", -1)
                          if idx_sUp != -1:
                              s_up_val += round(solution[idx_sUp]) * (2**k)
                      print(f" Ramp-Up Slack sUp_{t}_{i}: {s_up_val}")
        # Ramp-Down Slack
        if penalty_factors["ramp_down"] != 0:
            for t in range(1, T+1):
                 for i in range(1, N+1):
                      s_down_val = 0
                      for k in range(K_ramp_down):
                          idx_sDown = variable_mapping.get(f"sDown_{t}_{i}_{k}", -1)
                          if idx_sDown != -1:
                              s_down_val += round(solution[idx_sDown]) * (2**k)
                      print(f" Ramp-Down Slack sDown_{t}_{i}: {s_down_val}")

    else:
        print("\nNo solution found by Gurobi.")