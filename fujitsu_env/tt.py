import math
import optuna
from dadk.BinPol import *
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction

###############################
# Global Problem Parameters
###############################
T = 3
N = 3

# (These lambda values will be tuned by Optuna.)
# Other parameters
P_max = {1:350, 2:200, 3:140}
D = {1: 160, 2: 500, 3: 300}
C_startup = {1:20, 2:18, 3:5}
C_shutdown = {1:0.5, 2:0.3, 3:1.0}
b_cost = {1:0.1, 2:0.125, 3:0.15}
c_cost = {1:5, 2:7, 3:6}
R_up = {1:200, 2:100, 3:100}
R_down = {1:300, 2:150, 3:100}

#####################################
# Preprocessing: Slack Bit Calculations
#####################################
total_Pmax = sum(P_max.values())
demand_bits = {} 
for t in range(1, T+1):
    max_slack = total_Pmax - D[t]
    demand_bits[t] = int(math.ceil(math.log2(max_slack + 1)))
    print(f"Time {t}: Demand slack max = {max_slack}, bits = {demand_bits[t]}")

max_slack_ramp = max(R_up[i] + P_max[i] for i in P_max)
K_ramp = int(math.ceil(math.log2(max_slack_ramp + 1)))
print("Ramp-up slack bits (uniform):", K_ramp)

max_slack_ramp_down = max(R_down[i] + P_max[i] for i in P_max)
K_ramp_down = int(math.ceil(math.log2(max_slack_ramp_down + 1)))
print("Ramp-down slack bits (uniform):", K_ramp_down)

#####################################
# Define Variable Shapes
#####################################
u_shape    = BitArrayShape(name='u', shape=(N, T))
zOn_shape  = BitArrayShape(name='zOn', shape=(N, T))
zOff_shape = BitArrayShape(name='zOff', shape=(N, T))
s_shapes = []
for t in range(1, T+1):
    s_shapes.append(BitArrayShape(name=f's_{t}', shape=(demand_bits[t],)))
sUp_shape = BitArrayShape(name='sUp', shape=(N, T - 1, K_ramp))
sDown_shape = BitArrayShape(name='sDown', shape=(N, T - 1, K_ramp_down))
var_shape_set = VarShapeSet(u_shape, zOn_shape, zOff_shape, *s_shapes, sUp_shape, sDown_shape)
BinPol.freeze_var_shape_set(var_shape_set)

#####################################
# Global settings for penalties
#####################################
tolerance = 1e-5
# k scales the total violation penalty relative to the sum of lambda parameters.
k = 0.1

# -------------------------------
# Penalty Functions
# -------------------------------
def inequality_penalty(delta, tol=tolerance, high_scale=10.0, low_scale=1.0):
    if delta < 0:
        return high_scale * abs(delta)
    else:
        return low_scale * abs(delta)

def equality_penalty(delta, tol=tolerance, scale=10.0):
    """
    For equality constraints, any deviation (positive or negative) is penalized.
    """
    if abs(delta) > tol:
        return scale * abs(delta)
    else:
        return 0.0

def compute_total_violation(sol):
    total_pen = 0.0
    u_arr = sol['u'].data.astype(int)
    zOn_arr = sol['zOn'].data.astype(int)
    zOff_arr = sol['zOff'].data.astype(int)
    
    # 1. Unit Transition Constraint (Equality):
    # u[i,t] - u[i,t-1] - zOn[i,t] + zOff[i,t] = 0
    for t in range(1, T):
        for i in range(N):
            delta = u_arr[i, t] - u_arr[i, t-1] - zOn_arr[i, t] + zOff_arr[i, t]
            total_pen += equality_penalty(delta)
    
    # 2. Mutual Exclusion Constraint (Inequality):
    # zOn[i,t] + zOff[i,t] <= 1  <=> 1 - (zOn+zOff) >= 0
    for t in range(T):
        for i in range(N):
            delta = 1 - (zOn_arr[i, t] + zOff_arr[i, t])
            total_pen += inequality_penalty(delta)
    
    # 3. Demand Balance Constraint (Equality):
    # sum_i P_max[i]*u[i,t] - slack = D[t+1]
    for t in range(T):
        total_power = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        slack_bits = sol[f's_{t+1}'].data.astype(int)
        slack_val = sum((2 ** bit) * slack_bits[bit] for bit in range(len(slack_bits)))
        delta = total_power - slack_val - D[t+1]
        total_pen += equality_penalty(delta)
    
    # 4. Ramp-up Constraint (Equality):
    # P_max[i]*(u[i,t]-u[i,t-1]) + slack = R_up[i]
    sUp_arr = sol['sUp'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_up = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            slack_up = sum((2 ** bit) * sUp_arr[i, t-1, bit] for bit in range(K_ramp))
            delta = delta_up + slack_up - R_up[i+1]
            total_pen += equality_penalty(delta)
    
    # 5. Ramp-down Constraint (Equality):
    # P_max[i]*(u[i,t-1]-u[i,t]) + slack = R_down[i]
    sDown_arr = sol['sDown'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_down = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            slack_down = sum((2 ** bit) * sDown_arr[i, t-1, bit] for bit in range(K_ramp_down))
            delta = delta_down + slack_down - R_down[i+1]
            total_pen += equality_penalty(delta)
    
    return total_pen

#####################################
# Feasibility Check Function
#####################################
def is_feasible(sol):
    u_arr = sol['u'].data.astype(int)
    zOn_arr = sol['zOn'].data.astype(int)
    zOff_arr = sol['zOff'].data.astype(int)
    
    # Unit transition constraint
    for t in range(1, T):
        for i in range(N):
            if abs(u_arr[i, t] - u_arr[i, t-1] - zOn_arr[i, t] + zOff_arr[i, t]) > tolerance:
                return False

    # Mutual exclusion constraint
    for t in range(T):
        for i in range(N):
            if (zOn_arr[i, t] + zOff_arr[i, t]) - 1 > tolerance:
                return False

    # Demand balance constraint
    for t in range(T):
        total_power = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        slack_bits = sol[f's_{t+1}'].data.astype(int)
        slack_val = sum((2 ** bit) * slack_bits[bit] for bit in range(len(slack_bits)))
        if abs(total_power - slack_val - D[t+1]) > tolerance:
            return False

    # Ramp-up constraint
    sUp_arr = sol['sUp'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_up = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            slack_up = sum((2 ** bit) * sUp_arr[i, t-1, bit] for bit in range(K_ramp))
            if abs(delta_up + slack_up - R_up[i+1]) > tolerance:
                return False

    # Ramp-down constraint
    sDown_arr = sol['sDown'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_down = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            slack_down = sum((2 ** bit) * sDown_arr[i, t-1, bit] for bit in range(K_ramp_down))
            if abs(delta_down + slack_down - R_down[i+1]) > tolerance:
                return False

    return True

#####################################
# Constraint Status Checker
#####################################
def check_constraints_status(sol):
    status = {}
    u_arr = sol['u'].data.astype(int)
    zOn_arr = sol['zOn'].data.astype(int)
    zOff_arr = sol['zOff'].data.astype(int)
    
    # 1. Unit Transition Constraint (logic1)
    logic1_ok = True
    logic1_violations = []
    for t in range(1, T):
        for i in range(N):
            if abs(u_arr[i, t] - u_arr[i, t-1] - zOn_arr[i, t] + zOff_arr[i, t]) > tolerance:
                logic1_ok = False
                logic1_violations.append((i, t))
    status["logic1 (unit transition)"] = (logic1_ok, logic1_violations)
    
    # 2. Mutual Exclusion Constraint (logic2)
    logic2_ok = True
    logic2_violations = []
    for t in range(T):
        for i in range(N):
            if (zOn_arr[i, t] + zOff_arr[i, t]) - 1 > tolerance:
                logic2_ok = False
                logic2_violations.append((i, t))
    status["logic2 (mutual exclusion)"] = (logic2_ok, logic2_violations)
    
    # 3. Demand Balance Constraint
    demand_ok = True
    demand_violations = []
    for t in range(T):
        total_power = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        slack_bits = sol[f's_{t+1}'].data.astype(int)
        slack_val = sum((2 ** bit) * slack_bits[bit] for bit in range(len(slack_bits)))
        if abs(total_power - slack_val - D[t+1]) > tolerance:
            demand_ok = False
            demand_violations.append(t)
    status["demand balance"] = (demand_ok, demand_violations)
    
    # 4. Ramp-up Constraint
    ramp_up_ok = True
    ramp_up_violations = []
    sUp_arr = sol['sUp'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_up = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            slack_up = sum((2 ** bit) * sUp_arr[i, t-1, bit] for bit in range(K_ramp))
            if abs(delta_up + slack_up - R_up[i+1]) > tolerance:
                ramp_up_ok = False
                ramp_up_violations.append((i, t))
    status["ramp-up"] = (ramp_up_ok, ramp_up_violations)
    
    # 5. Ramp-down Constraint
    ramp_down_ok = True
    ramp_down_violations = []
    sDown_arr = sol['sDown'].data.astype(int)
    for t in range(1, T):
        for i in range(N):
            delta_down = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            slack_down = sum((2 ** bit) * sDown_arr[i, t-1, bit] for bit in range(K_ramp_down))
            if abs(delta_down + slack_down - R_down[i+1]) > tolerance:
                ramp_down_ok = False
                ramp_down_violations.append((i, t))
    status["ramp-down"] = (ramp_down_ok, ramp_down_violations)
    
    return status

#####################################
# Optuna Objective Function
#####################################
def objective(trial):
    # Tune the penalty multipliers (lambda parameters)
    logic1_penalty = trial.suggest_float("logic1_penalty", 0.0001, 15)
    logic2_penalty = trial.suggest_float("logic2_penalty", 0.0001, 15)
    demand_penalty  = trial.suggest_float("demand_penalty",  0.0001, 15)
    ramp_up_penalty = trial.suggest_float("ramp_up_penalty", 0.0001, 15)
    ramp_down_penalty = trial.suggest_float("ramp_down_penalty", 0.0001, 15)
    
    # Build the QUBO
    qubo = BinPol()
    
    # Cost terms: generation, startup, shutdown costs
    for t in range(T):
        for i in range(N):
            cost_u = b_cost[i+1] * P_max[i+1] + c_cost[i+1]
            qubo.add_term(cost_u, ('u', i, t))
            qubo.add_term(C_startup[i+1], ('zOn', i, t))
            qubo.add_term(C_shutdown[i+1], ('zOff', i, t))
    
    # Constraint 1: Startup/Shut-down logic (Equality)
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(1, ('u', i, t))
                    .add_term(-1, ('u', i, t-1))
                    .add_term(-1, ('zOn', i, t))
                    .add_term(1, ('zOff', i, t)))
            qubo.add(poly.power2(), logic1_penalty)
    
    # Constraint 2: Mutual Exclusion (Inequality)
    for t in range(T):
        for i in range(N):
            qubo.add_term(logic2_penalty, ('zOn', i, t), ('zOff', i, t))
    
    # Constraint 3: Demand Balance (Equality)
    for t in range(T):
        poly = BinPol()
        for i in range(N):
            poly.add_term(P_max[i+1], ('u', i, t))
        bits_length = demand_bits[t+1]
        for bit in range(bits_length):
            poly.add_term(- (2 ** bit), (f's_{t+1}', bit))
        poly.add_term(-D[t+1])
        qubo.add(poly.power2(), demand_penalty)
    
    # Constraint 4: Ramp-up (Equality)
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(P_max[i+1], ('u', i, t))
                    .add_term(-P_max[i+1], ('u', i, t-1)))
            for bit in range(K_ramp):
                poly.add_term(2 ** bit, ('sUp', i, t - 1, bit))
            poly.add_term(-R_up[i+1])
            qubo.add(poly.power2(), ramp_up_penalty)
    
    # Constraint 5: Ramp-down (Equality)
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(P_max[i+1], ('u', i, t-1))
                    .add_term(-P_max[i+1], ('u', i, t)))
            for bit in range(K_ramp_down):
                poly.add_term(2 ** bit, ('sDown', i, t - 1, bit))
            poly.add_term(-R_down[i+1])
            qubo.add(poly.power2(), ramp_down_penalty)
    
    # Solve the QUBO using dadk.QUBOSolverCPU
    solver = QUBOSolverCPU(
        optimization_method='parallel_tempering',
        number_runs=10,
        number_replicas=10,
        number_iterations=10000,
        temperature_sampling=True,
        random_seed=42,
        scaling_action=ScalingAction.AUTO_SCALING,
    )
    solution_list = solver.minimize(qubo)
    solution = solution_list.get_minimum_energy_solution()
    
    # Sum of lambda parameters
    lambda_sum = logic1_penalty + logic2_penalty + demand_penalty + ramp_up_penalty + ramp_down_penalty
    
    # If the solution is infeasible, add the scaled total violation penalty.
    violation = compute_total_violation(solution)
    obj_value = lambda_sum + k * violation
    return obj_value

#####################################
# Run the Optuna Study
#####################################
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print("Best penalty parameters found:")
    print(study.best_params)
    
    # -------------- Final Run with Best Parameters --------------
    best_params = study.best_params
    logic1_penalty = best_params["logic1_penalty"]
    logic2_penalty = best_params["logic2_penalty"]
    demand_penalty  = best_params["demand_penalty"]
    ramp_up_penalty = best_params["ramp_up_penalty"]
    ramp_down_penalty = best_params["ramp_down_penalty"]
    
    qubo = BinPol()
    for t in range(T):
        for i in range(N):
            cost_u = b_cost[i+1] * P_max[i+1] + c_cost[i+1]
            qubo.add_term(cost_u, ('u', i, t))
            qubo.add_term(C_startup[i+1], ('zOn', i, t))
            qubo.add_term(C_shutdown[i+1], ('zOff', i, t))
    
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(1, ('u', i, t))
                    .add_term(-1, ('u', i, t-1))
                    .add_term(-1, ('zOn', i, t))
                    .add_term(1, ('zOff', i, t)))
            qubo.add(poly.power2(), logic1_penalty)
    
    for t in range(T):
        for i in range(N):
            qubo.add_term(logic2_penalty, ('zOn', i, t), ('zOff', i, t))
    
    for t in range(T):
        poly = BinPol()
        for i in range(N):
            poly.add_term(P_max[i+1], ('u', i, t))
        bits_length = demand_bits[t+1]
        for bit in range(bits_length):
            poly.add_term(- (2 ** bit), (f's_{t+1}', bit))
        poly.add_term(-D[t+1])
        qubo.add(poly.power2(), demand_penalty)
    
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(P_max[i+1], ('u', i, t))
                    .add_term(-P_max[i+1], ('u', i, t-1)))
            for bit in range(K_ramp):
                poly.add_term(2 ** bit, ('sUp', i, t - 1, bit))
            poly.add_term(-R_up[i+1])
            qubo.add(poly.power2(), ramp_up_penalty)
    
    for t in range(1, T):
        for i in range(N):
            poly = (BinPol()
                    .add_term(P_max[i+1], ('u', i, t-1))
                    .add_term(-P_max[i+1], ('u', i, t)))
            for bit in range(K_ramp_down):
                poly.add_term(2 ** bit, ('sDown', i, t - 1, bit))
            poly.add_term(-R_down[i+1])
            qubo.add(poly.power2(), ramp_down_penalty)
    
    solver = QUBOSolverCPU(
        optimization_method='parallel_tempering',
        number_runs=100,
        number_replicas=100,
        number_iterations=10000,
        temperature_sampling=True,
        random_seed=42,
        scaling_action=ScalingAction.AUTO_SCALING,
    )
    solution_list = solver.minimize(qubo)
    solution = solution_list.get_minimum_energy_solution()
    
    u_sol = solution['u'].data.astype(int)
    zOn_sol = solution['zOn'].data.astype(int)
    zOff_sol = solution['zOff'].data.astype(int)
    
    orig_obj = 0
    for t in range(T):
        for i in range(N):
            orig_obj += zOn_sol[i, t] * C_startup[i+1]
            orig_obj += zOff_sol[i, t] * C_shutdown[i+1]
            orig_obj += u_sol[i, t] * (b_cost[i+1] * P_max[i+1] + c_cost[i+1])
    
    print("Final results with best penalty parameters:")
    print("Original objective value (costs only):", orig_obj)
    print("Full QUBO objective value (with penalties):", solution.energy)
    
    sorted_solutions = solution_list.get_sorted_solution_list()
    print(f"Total number of solutions: {len(sorted_solutions)}")
    min_solution = solution_list.get_minimum_energy_solution()
    print("Lowest energy among all solutions:", min_solution.energy)
    
    # Check each constraint and print status
    cons_status = check_constraints_status(solution)
    for cons, (ok, violations) in cons_status.items():
        if ok:
            print(f"{cons} is satisfied.")
        else:
            print(f"{cons} is violated. Violations at indices: {violations}")
    
    # Also, iterate to find one feasible solution among the sorted solutions:
    index = 0
    found_feasible = False
    while index < len(sorted_solutions):
        sol = sorted_solutions[index]
        if is_feasible(sol):
            print("FEASIBLE SOLUTION FOUND, best one with energy:", sol.energy)
            found_feasible = True
            break
        index += 1
    if not found_feasible:
        print("No feasible solution found among the sorted solutions.")
