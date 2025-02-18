import numpy as np

# ---------------------------
# Problem Dimensions & Parameters
# ---------------------------
T = 3   # Number of time periods
N = 3   # Number of units
K = 4   # Number of bits for each slack variable

# Define individual penalty factors for each constraint:
penalty_factors = {
    "unit_transition": 30,
    "mutual_exclusion": 30,
    "demand": 30,
    "ramp_up": 30,
    "ramp_down": 30
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
# Create a Mapping for Variables
# ---------------------------
# Each variable (including slack bits) is assigned a unique index in x.
variable_mapping = {}
current_index = 0

def add_variable(name):
    """Add a variable to the mapping and assign a unique index."""
    global current_index
    variable_mapping[name] = current_index
    current_index += 1

# For each time period t = 1,...,T and unit i = 1,...,N
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"u_{i}_{t}")      # Unit on/off status u_i,t
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOn_{i}_{t}")    # Startup decision z_i,t^On
for t in range(1, T+1):
    for i in range(1, N+1):
        add_variable(f"zOff_{i}_{t}")   # Shutdown decision z_i,t^Off

# Slack variables for demand constraint: s_{t,k}
for t in range(1, T+1):
    for k in range(K):
        add_variable(f"s_{t}_{k}")

# Slack variables for ramp-up constraints: sUp_{t,i,k}
for t in range(1, T+1):
    for i in range(1, N+1):
        for k in range(K):
            add_variable(f"sUp_{t}_{i}_{k}")

# Slack variables for ramp-down constraints: sDown_{t,i,k}
for t in range(1, T+1):
    for i in range(1, N+1):
        for k in range(K):
            add_variable(f"sDown_{t}_{i}_{k}")

num_variables = current_index
print("Total number of variables:", num_variables)

# ---------------------------
# Initialize Q Matrix and Constant Term
# ---------------------------
# For this example we use a dense representation.
Q = np.zeros((num_variables, num_variables))
c_term = 0

# ---------------------------
# Helper Functions to Update Q and c_term
# ---------------------------
def add_quadratic_term(i, j, coeff):
    """
    Add a quadratic term: coeff * x_i * x_j.
    For off-diagonal entries, split the coefficient to maintain symmetry.
    """
    if i == j:
        Q[i, i] += coeff
    else:
        Q[i, j] += coeff / 2.0
        Q[j, i] += coeff / 2.0

def add_linear_term(i, coeff):
    """
    Add a linear term: coeff * x_i.
    For binary variables, x_i^2 = x_i so we add to the diagonal.
    """
    Q[i, i] += coeff

def add_constant_term(value):
    """Add a constant term to the overall constant."""
    global c_term
    c_term += value

def add_penalty_term(term_list, constant_term, weight):
    """
    Given a list of terms of the form (index, coeff) and a constant,
    add the penalty weight * (sum_j coeff_j * x_j + constant)^2 to Q and c_term.
    
    This expands to:
      weight * [sum_{j,l} (coeff_j*coeff_l * x_j*x_l) + 2*constant*sum_j (coeff_j*x_j) + constant^2]
    """
    # Add quadratic terms (including linear when j==l)
    for idx1, coeff1 in term_list:
        for idx2, coeff2 in term_list:
            add_quadratic_term(idx1, idx2, weight * coeff1 * coeff2)
    # Add linear terms from constant cross-terms
    for idx, coeff in term_list:
        add_linear_term(idx, weight * 2 * coeff * constant_term)
    # Add constant term
    add_constant_term(weight * constant_term * constant_term)

# ---------------------------
# Add Cost Function Terms (Linear Terms)
# ---------------------------
# For each time period and unit, add: 
#   zOn_{i,t}*C_startup,i + zOff_{i,t}*C_shutdown,i + u_{i,t}*(b_i*P_max[i] + c_i)
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
# Constraint: u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t} = 0 for t>=2, for each unit i.
# Penalty term: penalty_factors["unit_transition"]*(u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t})^2
for t in range(2, T+1):
    for i in range(1, N+1):
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
        # Build the term list: (index, coefficient)
        terms = [(idx_u_t, 1.0),
                 (idx_u_prev, -1.0),
                 (idx_zOn, -1.0),
                 (idx_zOff, 1.0)]
        add_penalty_term(terms, 0, penalty_factors["unit_transition"])

# ---------------------------
# Add Mutual Exclusion Constraint
# Constraint: zOff_{i,t} + zOn_{i,t} ≤ 1.
# In our QUBO, we penalize the case when both are 1: add penalty_factors["mutual_exclusion"]*(zOff_{i,t} * zOn_{i,t}).
for t in range(1, T+1):
    for i in range(1, N+1):
        idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
        idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
        add_quadratic_term(idx_zOn, idx_zOff, penalty_factors["mutual_exclusion"])

# ---------------------------
# Add Demand Satisfaction Constraint
# Constraint (for each time t): ∑_(i=1)^N (P_max[i] * u_{i,t}) - ∑_(k=0)^(K-1) (2^k * s_{t,k}) - D[t] = 0.
for t in range(1, T+1):
    term_list = []
    # Terms from units
    for i in range(1, N+1):
        idx_u = variable_mapping[f"u_{i}_{t}"]
        term_list.append((idx_u, P_max[i]))
    # Terms from demand slack variables
    for k in range(K):
        idx_s = variable_mapping[f"s_{t}_{k}"]
        term_list.append((idx_s, - (2 ** k)))
    # Constant is -D[t]
    constant_val = -D[t]
    add_penalty_term(term_list, constant_val, penalty_factors["demand"])

# ---------------------------
# Add Ramp-Up Constraint
# Constraint (for t>=2, for each unit i): 
#   P_max[i]*(u_{i,t} - u_{i,t-1}) + ∑_(k=0)^(K-1) (2^k * sUp_{t,i,k}) - R_up[i] = 0.
for t in range(2, T+1):
    for i in range(1, N+1):
        term_list = []
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        term_list.append((idx_u_t, P_max[i]))
        term_list.append((idx_u_prev, -P_max[i]))
        for k in range(K):
            idx_sUp = variable_mapping[f"sUp_{t}_{i}_{k}"]
            term_list.append((idx_sUp, 2 ** k))
        constant_val = -R_up[i]
        add_penalty_term(term_list, constant_val, penalty_factors["ramp_up"])

# ---------------------------
# Add Ramp-Down Constraint
# Constraint (for t>=2, for each unit i):
#   P_max[i]*(u_{i,t-1} - u_{i,t}) + ∑_(k=0)^(K-1) (2^k * sDown_{t,i,k}) - R_down[i] = 0.
for t in range(2, T+1):
    for i in range(1, N+1):
        term_list = []
        idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
        idx_u_t = variable_mapping[f"u_{i}_{t}"]
        term_list.append((idx_u_prev, P_max[i]))
        term_list.append((idx_u_t, -P_max[i]))
        for k in range(K):
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

# The final Q and c_term represent the QUBO in the standard form: x^T Q x + c.
