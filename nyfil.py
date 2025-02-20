import numpy as np
import math
import gurobipy as gp
from gurobipy import GRB
import optuna

# ---------------------------
# Global Parameters
# ---------------------------
T = 3   # Number of time periods
N = 3   # Number of units

# Parameter dictionaries
P_max = {1: 350, 2: 200, 3: 140}
D = {1: 160, 2: 500, 3: 400}
C_startup = {1: 20, 2: 18, 3: 5}
C_shutdown = {1: 0.5, 2: 0.3, 3: 1.0}
b_cost = {1: 0.1, 2: 0.125, 3: 0.15}
c_cost = {1: 5, 2: 7, 3: 6}
R_up = {1: 200, 2: 100, 3: 100}
R_down = {1: 300, 2: 150, 3: 100}

# ---------------------------
# Function to Build the QUBO
# ---------------------------
def build_qubo(penalty_factors):
    # --- Determine Dynamic Bit-lengths for Slack Variables ---
    # For the demand constraint at time t: slack = (sum_i P_max[i]) - D[t]
    demand_K = {}
    total_Pmax = sum(P_max.values())
    for t in range(1, T+1):
        max_slack = total_Pmax - D[t]
        demand_K[t] = int(math.ceil(math.log2(max_slack + 1)))
        print(f"Time {t}: Demand slack max = {max_slack}, bits = {demand_K[t]}")
        
    # For ramp-up and ramp-down: worst-case slack = R[i] + P_max[i]
    max_slack_ramp_up = max(R_up[i] + P_max[i] for i in P_max)
    K_ramp_up = int(math.ceil(math.log2(max_slack_ramp_up + 1)))
    print("Ramp-up slack bits (uniform for all units):", K_ramp_up)
    
    max_slack_ramp_down = max(R_down[i] + P_max[i] for i in P_max)
    K_ramp_down = int(math.ceil(math.log2(max_slack_ramp_down + 1)))
    print("Ramp-down slack bits (uniform for all units):", K_ramp_down)
    
    # --- Create a Mapping for Variables ---
    variable_mapping = {}
    current_index = 0

    def add_variable(name):
        nonlocal current_index
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
            
    # 4. Slack variables for demand constraint: s_{t,k}
    if penalty_factors["demand"] != 0:
        for t in range(1, T+1):
            for k in range(demand_K[t]):
                add_variable(f"s_{t}_{k}")
                
    # 5. Slack variables for ramp-up constraints: sUp_{t,i,k}
    if penalty_factors["ramp_up"] != 0:
        for t in range(1, T+1):
            for i in range(1, N+1):
                for k in range(K_ramp_up):
                    add_variable(f"sUp_{t}_{i}_{k}")
                    
    # 6. Slack variables for ramp-down constraints: sDown_{t,i,k}
    if penalty_factors["ramp_down"] != 0:
        for t in range(1, T+1):
            for i in range(1, N+1):
                for k in range(K_ramp_down):
                    add_variable(f"sDown_{t}_{i}_{k}")
                    
    num_variables = current_index
    print("Total number of variables:", num_variables)
    
    # --- Initialize Q Matrix and Constant Term ---
    Q = np.zeros((num_variables, num_variables))
    c_term = 0

    def add_quadratic_term(i, j, coeff):
        if i == j:
            Q[i, i] += coeff
        else:
            Q[i, j] += coeff / 2.0
            Q[j, i] += coeff / 2.0

    def add_linear_term(i, coeff):
        Q[i, i] += coeff

    def add_constant_term(value):
        nonlocal c_term
        c_term += value

    def add_penalty_term(term_list, constant_term_val, weight):
        for idx1, coeff1 in term_list:
            for idx2, coeff2 in term_list:
                add_quadratic_term(idx1, idx2, weight * coeff1 * coeff2)
        for idx, coeff in term_list:
            add_linear_term(idx, weight * 2 * coeff * constant_term_val)
        add_constant_term(weight * constant_term_val * constant_term_val)

    # --- Add Cost Function Terms ---
    for t in range(1, T+1):
        for i in range(1, N+1):
            idx_u = variable_mapping[f"u_{i}_{t}"]
            idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
            idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
            cost_u = b_cost[i] * P_max[i] + c_cost[i]
            add_linear_term(idx_u, cost_u)
            add_linear_term(idx_zOn, C_startup[i])
            add_linear_term(idx_zOff, C_shutdown[i])

    # --- Add Constraints with Penalty Terms ---
    # 1. Unit Status Transition: u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t} = 0 for t>=2.
    if penalty_factors["unit_transition"] != 0:
        for t in range(2, T+1):
            for i in range(1, N+1):
                idx_u_t = variable_mapping[f"u_{i}_{t}"]
                idx_u_prev = variable_mapping[f"u_{i}_{t-1}"]
                idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
                idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
                terms = [(idx_u_t, 1.0), (idx_u_prev, -1.0),
                         (idx_zOn, -1.0), (idx_zOff, 1.0)]
                add_penalty_term(terms, 0, penalty_factors["unit_transition"])
    
    # 2. Mutual Exclusion: zOn_{i,t} + zOff_{i,t} ≤ 1.
    if penalty_factors["mutual_exclusion"] != 0:
        for t in range(1, T+1):
            for i in range(1, N+1):
                idx_zOn = variable_mapping[f"zOn_{i}_{t}"]
                idx_zOff = variable_mapping[f"zOff_{i}_{t}"]
                add_quadratic_term(idx_zOn, idx_zOff, penalty_factors["mutual_exclusion"])
    
    # 3. Demand Satisfaction: sum_i P_max[i]*u_{i,t} - sum_k (2^k * s_{t,k}) = D[t].
    if penalty_factors["demand"] != 0:
        for t in range(1, T+1):
            term_list = []
            for i in range(1, N+1):
                idx_u = variable_mapping[f"u_{i}_{t}"]
                term_list.append((idx_u, P_max[i]))
            for k in range(demand_K[t]):
                idx_s = variable_mapping[f"s_{t}_{k}"]
                term_list.append((idx_s, - (2 ** k)))
            add_penalty_term(term_list, -D[t], penalty_factors["demand"])
    
    # 4. Ramp-Up Constraint: P_max[i]*(u_{i,t} - u_{i,t-1]) + sum_k (2^k * sUp_{t,i,k}) = R_up[i] for t>=2.
    if penalty_factors["ramp_up"] != 0:
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
                add_penalty_term(term_list, -R_up[i], penalty_factors["ramp_up"])
    
    # 5. Ramp-Down Constraint: P_max[i]*(u_{i,t-1} - u_{i,t]) + sum_k (2^k * sDown_{t,i,k}) = R_down[i] for t>=2.
    if penalty_factors["ramp_down"] != 0:
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
                add_penalty_term(term_list, -R_down[i], penalty_factors["ramp_down"])
    
    print("Final Q matrix (dense format):")
    print(Q)
    print("Shape of Q matrix:", Q.shape)
    print("\nConstant term c:")
    print(c_term)
    print("Is the matrix symmetric?", np.allclose(Q, Q.T))
    
    return Q, c_term, variable_mapping, demand_K, K_ramp_up, K_ramp_down

# ---------------------------
# Function to Solve the QUBO using Gurobi
# ---------------------------
def solve_qubo(Q, c):
    n = Q.shape[0]
    m = gp.Model("qubo")
    m.setParam('OutputFlag', 0)  # Turn off solver output
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0:
                obj.add(Q[i, j] * x[i] * x[j])
    obj.add(c)
    m.setObjective(obj, GRB.MINIMIZE)
    
    m.optimize()
    solution = np.array([x[i].X for i in range(n)])
    obj_val = m.objVal
    return solution, obj_val

# ---------------------------
# Function to Check Constraint Violations
# ---------------------------
def check_constraints(solution, variable_mapping, demand_K, K_ramp_up, K_ramp_down, tol=1e-6):
    total_violation = 0.0
    # 1. Unit Status Transition: u_{i,t} - u_{i,t-1} - zOn_{i,t} + zOff_{i,t} = 0 for t>=2.
    for t in range(2, T+1):
        for i in range(1, N+1):
            lhs = (solution[variable_mapping[f"u_{i}_{t}"]] -
                   solution[variable_mapping[f"u_{i}_{t-1}"]] -
                   solution[variable_mapping[f"zOn_{i}_{t}"]] +
                   solution[variable_mapping[f"zOff_{i}_{t}"]])
            total_violation += abs(lhs)
    # 2. Mutual Exclusion: zOn_{i,t} + zOff_{i,t} ≤ 1.
    for t in range(1, T+1):
        for i in range(1, N+1):
            summ = (solution[variable_mapping[f"zOn_{i}_{t}"]] +
                    solution[variable_mapping[f"zOff_{i}_{t}"]])
            if summ > 1 + tol:
                total_violation += summ - 1
    # 3. Demand Satisfaction: sum_i P_max[i]*u_{i,t} - sum_k (2^k * s_{t,k}) = D[t].
    for t in range(1, T+1):
        lhs = 0
        for i in range(1, N+1):
            lhs += P_max[i] * solution[variable_mapping[f"u_{i}_{t}"]]
        for k in range(demand_K[t]):
            lhs -= (2 ** k) * solution[variable_mapping[f"s_{t}_{k}"]]
        total_violation += abs(lhs - D[t])
    # 4. Ramp-Up: P_max[i]*(u_{i,t} - u_{i,t-1]) + sum_k (2^k * sUp_{t,i,k}) = R_up[i] for t>=2.
    for t in range(2, T+1):
        for i in range(1, N+1):
            lhs = P_max[i]*(solution[variable_mapping[f"u_{i}_{t}"]] - solution[variable_mapping[f"u_{i}_{t-1}"]])
            for k in range(K_ramp_up):
                lhs += (2 ** k)*solution[variable_mapping[f"sUp_{t}_{i}_{k}"]]
            total_violation += abs(lhs - R_up[i])
    # 5. Ramp-Down: P_max[i]*(u_{i,t-1} - u_{i,t]) + sum_k (2^k * sDown_{t,i,k}) = R_down[i] for t>=2.
    for t in range(2, T+1):
        for i in range(1, N+1):
            lhs = P_max[i]*(solution[variable_mapping[f"u_{i}_{t-1}"]] - solution[variable_mapping[f"u_{i}_{t}"]])
            for k in range(K_ramp_down):
                lhs += (2 ** k)*solution[variable_mapping[f"sDown_{t}_{i}_{k}"]]
            total_violation += abs(lhs - R_down[i])
    return total_violation

# ---------------------------
# Optuna Objective Function
# ---------------------------
def objective(trial):
    # Suggest penalty parameters on a log scale (adjust ranges if needed)
    unit_transition = trial.suggest_float("unit_transition", 0, 5)
    mutual_exclusion = trial.suggest_float("mutual_exclusion", 0, 5)
    demand = trial.suggest_float("demand", 0, 5)
    ramp_up = trial.suggest_float("ramp_up", 0, 5)
    ramp_down = trial.suggest_float("ramp_down", 0, 5)
    
    penalty_factors = {
        "unit_transition": unit_transition,
        "mutual_exclusion": mutual_exclusion,
        "demand": demand,
        "ramp_up": ramp_up,
        "ramp_down": ramp_down
    }
    
    # Build QUBO with the current penalty factors
    Q, c_term, variable_mapping, demand_K, K_ramp_up, K_ramp_down = build_qubo(penalty_factors)
    solution, _ = solve_qubo(Q, c_term)
    
    # Check the solution for constraint violations
    total_violation = check_constraints(solution, variable_mapping, demand_K, K_ramp_up, K_ramp_down)
    
    if total_violation > 1e-6:
        # Infeasible solution: return a large penalty
        return 1e6 + total_violation
    # If feasible, the objective is the sum of penalty parameters (we want these as low as possible)
    return unit_transition + mutual_exclusion + demand + ramp_up + ramp_down

# ---------------------------
# Main: Run Optuna Study and Solve Final QUBO
# ---------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)  # Adjust n_trials as needed
    
    print("Best penalty parameters found:")
    print(study.best_params)
    
    # Build final QUBO with best penalty parameters
    best_penalty_factors = study.best_params
    Q, c_term, variable_mapping, demand_K, K_ramp_up, K_ramp_down = build_qubo(best_penalty_factors)
    solution, obj_val = solve_qubo(Q, c_term)
    
    print("Optimal binary solution x:")
    print(solution)
    print("Optimal objective value:")
    print(obj_val)
    
    total_violation = check_constraints(solution, variable_mapping, demand_K, K_ramp_up, K_ramp_down)
    print("Total constraint violation:", total_violation)
