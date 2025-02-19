import numpy as np
import math
# ---------------------------
# Problem Dimensions & Parameters
# ---------------------------
T = 3   # Number of time periods
N = 3   # Number of units

# Global penalty factors (one per constraint)
penalty_factors = {
    "unit_transition": 0,
    "mutual_exclusion": 0,
    "demand": 0,
    "ramp_up": 0,
    "ramp_down": 0
}

# Parameter dictionaries
P_max = {1:350, 2:200, 3:140}
D = {1:160, 2:500, 3:400}
C_startup = {1:20, 2:18, 3:5}
C_shutdown = {1:0.5, 2:0.3, 3:1.0}
b_cost = {1:0.1, 2:0.125, 3:0.15}
c_cost = {1:5, 2:7, 3:6}
R_up = {1:200, 2:100, 3:100}
R_down = {1:300, 2:150, 3:100}

# ---------------------------
# Determine Dynamic Bit-lengths for Slack Variables
# ---------------------------
# For the demand constraint at time t: s_t = (sum_i P_max[i]) - D[t]
demand_K = {}  # For each time period t, the number of bits needed for the demand slack.
total_Pmax = sum(P_max.values())
for t in range(1, T+1):
    max_slack = total_Pmax - D[t]
    demand_K[t] = int(math.ceil(math.log2(max_slack + 1)))
    print(f"Time {t}: Demand slack max = {max_slack}, bits = {demand_K[t]}")

# For ramp-up: constraint is P_max[i]*(u_{i,t} - u_{i,t-1]) + sUp_{t,i} = R_up[i]
# Worst-case slack: when u_{i,t} - u_{i,t-1} is most negative, we assume worst-case = R_up[i] + P_max[i].
max_slack_ramp_up = max(R_up[i] + P_max[i] for i in P_max)
K_ramp_up = int(math.ceil(math.log2(max_slack_ramp_up + 1)))
print("Ramp-up slack bits (uniform for all units):", K_ramp_up)

# For ramp-down: worst-case slack = R_down[i] + P_max[i]
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

# 1. Unit status variables: u_{i,t}
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"u_{i}_{t}")

# 2. Startup decision variables: zOn_{i,t}
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOn_{i}_{t}")

# 3. Shutdown decision variables: zOff_{i,t}
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOff_{i}_{t}")

# 4. Slack variables for demand constraint: s_{t,k} (using dynamic bits per time period)
for t in range(1, T+1):
    for k in range(demand_K[t]):
        add_variable(f"s_{t}_{k}")

# 5. Slack variables for ramp-up constraints: sUp_{t,i,k} (uniform K_ramp_up)
for t in range(1, T+1):
    for i in range(1, N+1):
        for k in range(K_ramp_up):
            add_variable(f"sUp_{t}_{i}_{k}")

# 6. Slack variables for ramp-down constraints: sDown_{t,i,k} (uniform K_ramp_down)
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
    Given a list of terms (index, coefficient) and a constant,
    add the penalty weight * (sum_j (coeff_j * x_j) + constant)^2.
    This expands to:
      weight * [sum_{j,l} (coeff_j*coeff_l * x_j*x_l) + 2*constant*sum_j (coeff_j*x_j) + constant^2]
    """
    for idx1, coeff1 in term_list:
        for idx2, coeff2 in term_list:
            add_quadratic_term(idx1, idx2, weight * coeff1 * coeff2)
    for idx, coeff in term_list:
        add_linear_term(idx, weight * 2 * coeff * constant_term)
    add_constant_term(weight * constant_term * constant_term)

# ---------------------------
# Add Cost Function Terms (Linear Terms)
# ---------------------------
for t in range(1, T+1):
    for i in range(1, N+1):
        idx_u = variable_mapping[f"u_{i}_{t}"]
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
        cost_u = b_cost[i] * P_max[i] + c_cost[i]
        add_linear_term(idx_u, cost_u)
        add_linear_term(idx_zOn, C_startup[i])
        add_linear_term(idx_zOff, C_shutdown[i])

# ---------------------------
# Add Unit Status Transition Constraint
# For t>=2, for each unit i: u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t} = 0.
for t in range(2, T+1):
    for i in range(1, N+1):
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
        terms = [(idx_u_t, 1.0),
                 (idx_u_prev, -1.0),
                 (idx_zOn, -1.0),
                 (idx_zOff, 1.0)]
        add_penalty_term(terms, 0, penalty_factors["unit_transition"])

# ---------------------------
# Add Mutual Exclusion Constraint
# For each t and i: zOff_{i,t} + zOn_{i,t} ≤ 1 (penalize if both are 1).
for t in range(1, T+1):
    for i in range(1, N+1):
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
        add_quadratic_term(idx_zOn, idx_zOff, penalty_factors["mutual_exclusion"])

# ---------------------------
# Add Demand Satisfaction Constraint
# For each time t: sum_i P_max[i]*u_{i,t} - (sum_{k=0}^{K_demand[t]-1} 2^k * s_{t,k}) - D[t] = 0.
for t in range(1, T+1):
    term_list = []
    for i in range(1, N+1):
        idx_u = variable_mapping[f"u_{i}_{t}"]
        term_list.append((idx_u, P_max[i]))
    for k in range(demand_K[t]):
        idx_s = variable_mapping[f"s_{t}_{k}"]
        term_list.append((idx_s, - (2 ** k)))
    constant_val = -D[t]
    add_penalty_term(term_list, constant_val, penalty_factors["demand"])

# ---------------------------
# Add Ramp-Up Constraint
# For each t>=2, for each unit i: P_max[i]*(u_{i,t} - u_{i,t-1]) + (sum_{k=0}^{K_ramp_up-1} 2^k * sUp_{t,i,k}) - R_up[i] = 0.
for t in range(2, T+1):
    for i in range(1, N+1):
        term_list = []
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        term_list.append((idx_u_t, P_max[i]))
        term_list.append((idx_u_prev, -P_max[i]))
        for k in range(K_ramp_up):
            idx_sUp = variable_mapping[f"sUp_{t}_{i}_{k}"]
            term_list.append((idx_sUp, 2 ** k))
        constant_val = -R_up[i]
        add_penalty_term(term_list, constant_val, penalty_factors["ramp_up"])

# ---------------------------
# Add Ramp-Down Constraint
# For each t>=2, for each unit i: P_max[i]*(u_{i,t-1} - u_{i,t]) + (sum_{k=0}^{K_ramp_down-1} 2^k * sDown_{t,i,k}) - R_down[i] = 0.
for t in range(2, T+1):
    for i in range(1, N+1):
        term_list = []
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        term_list.append((idx_u_prev, P_max[i]))
        term_list.append((idx_u_t, -P_max[i]))
        for k in range(K_ramp_down):
            idx_sDown = variable_mapping[f"sDown_{t}_{i}_{k}"]
            term_list.append((idx_sDown, 2 ** k))
        constant_val = -R_down[i]
        add_penalty_term(term_list, constant_val, penalty_factors["ramp_down"])

# ---------------------------
# Final Output
# ---------------------------
print("Final Q matrix (dense format):")
print(Q)
print("Shape of Q matrix:", Q.shape)
print("\nConstant term c:")
print(c_term)
print("Is the matrix symmetric?", np.allclose(Q, Q.T))



