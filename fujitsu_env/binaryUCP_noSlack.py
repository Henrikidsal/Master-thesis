##### This is the all binary problem
##### Using the "Unbalanced Penalization" method without slack variables.
##### Ramping constraints for t=1 transition are included.
##### Reduced Print Output Version.

##### This first part imports the necessary tools #####
from dadk.BinPol import *
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction
import math
import timeit
import numpy as np

##### This part defines the parameters #####

# Number of time periods and generating units
T = 3
N = 3

# Previous time period status (initial conditions):
u_prev = {1: 0, 2: 0, 3: 1}

# --- Penalty Weights (Lambdas) ---
lambda_0 = 30           # Logic 1 
lambda_logic2 = 20      # Logic 2 
demand_lambda1 = 730000  # Demand 
demand_lambda2 = 8600
ramp_up_lambda1 = 1     # Ramp-Up 
ramp_up_lambda2 = 0.5
ramp_down_lambda1 = 1   # Ramp-Down 
ramp_down_lambda2 = 0.5

# parameters
P_max = {1:350, 2:200, 3:140}
D = {1: 160, 2: 500, 3: 400}
C_startup = {1:20, 2:18, 3:5}
C_shutdown = {1:0.5, 2:0.3, 3:1.0}
b_cost = {1:0.1, 2:0.125, 3:0.15}
c_cost = {1:5, 2:7, 3:6}
R_up = {1:350, 2:200, 3:140}
R_down = {1:350, 2:200, 3:140}

##### Variable Shapes #####
u_shape    = BitArrayShape(name='u', shape=(N, T))
zOn_shape  = BitArrayShape(name='zOn', shape=(N, T))
zOff_shape = BitArrayShape(name='zOff', shape=(N, T))
var_shape_set = VarShapeSet(u_shape, zOn_shape, zOff_shape)
BinPol.freeze_var_shape_set(var_shape_set)

# Calculate total variables
total_vars = 0
shape_names = ['u', 'zOn', 'zOff']
for name in shape_names:
    try:
        shape_obj = var_shape_set[name]
        total_vars += np.prod(shape_obj.shape)
    except KeyError:
        # This error shouldn't happen with the defined shapes
        pass
# --- KEEP ---
print("\nTotal number of binary variables:", total_vars)

##### Here the QUBO is built #####
qubo = BinPol()

# --- Objective Function ---
# (Print removed)
for t in range(T):
    for i in range(N):
        cost_u = b_cost[i+1] * P_max[i+1] + c_cost[i+1]
        qubo.add_term(cost_u, ('u', i, t))
        qubo.add_term(C_startup[i+1], ('zOn', i, t))
        qubo.add_term(C_shutdown[i+1], ('zOff', i, t))

# --- Constraint 1: Start-up/Shut-down Logic (Equality) ---
# (Print removed)
# t = 0
for i in range(N):
    poly = BinPol(); poly.add_term(1, ('u', i, 0)); poly.add_term(-1, ('zOn', i, 0)); poly.add_term(1, ('zOff', i, 0)); poly.add_term(-u_prev[i+1])
    qubo.add(poly.power2(), lambda_0)
# t >= 1
for t in range(1, T):
    for i in range(N):
        poly = BinPol(); poly.add_term(1, ('u', i, t)); poly.add_term(-1, ('u', i, t-1)); poly.add_term(-1, ('zOn', i, t)); poly.add_term(1, ('zOff', i, t))
        qubo.add(poly.power2(), lambda_0)

# --- Constraint 2: Mutual Exclusion (Simple Inequality) ---
# (Print removed)
for t in range(T):
    for i in range(N):
        qubo.add_term(lambda_logic2, ('zOn', i, t), ('zOff', i, t))

# --- Constraint 3: Demand Balance Constraint (Inequality) ---
# (Print removed)
for t in range(T):
    h_t = BinPol()
    for i in range(N): h_t.add_term(P_max[i+1], ('u', i, t))
    h_t.add_term(-D[t+1])
    qubo.add(h_t, -demand_lambda1); qubo.add(h_t.power2(), demand_lambda2)

# --- Constraint 4: Ramp-Up Constraint (Inequality) ---
# (Print removed)
# t=1 transition
for i in range(N):
    h_up_i0 = BinPol(); h_up_i0.add_term(R_up[i+1]); h_up_i0.add_term(-P_max[i+1], ('u', i, 0)); h_up_i0.add_term(P_max[i+1] * u_prev[i+1])
    qubo.add(h_up_i0, -ramp_up_lambda1); qubo.add(h_up_i0.power2(), ramp_up_lambda2)
# t >= 2 transitions
for t in range(1, T):
    for i in range(N):
        h_up_it = BinPol(); h_up_it.add_term(R_up[i+1]); h_up_it.add_term(-P_max[i+1], ('u', i, t)); h_up_it.add_term(P_max[i+1], ('u', i, t-1))
        qubo.add(h_up_it, -ramp_up_lambda1); qubo.add(h_up_it.power2(), ramp_up_lambda2)

for i in range(N):
    h_down_i0 = BinPol(); h_down_i0.add_term(R_down[i+1]); h_down_i0.add_term(-P_max[i+1] * u_prev[i+1]); h_down_i0.add_term(P_max[i+1], ('u', i, 0))
    qubo.add(h_down_i0, -ramp_down_lambda1); qubo.add(h_down_i0.power2(), ramp_down_lambda2)

# t >= 2 transitions
for t in range(1, T):
    for i in range(N):
        h_down_it = BinPol(); h_down_it.add_term(R_down[i+1]); h_down_it.add_term(-P_max[i+1], ('u', i, t-1)); h_down_it.add_term(P_max[i+1], ('u', i, t))
        qubo.add(h_down_it, -ramp_down_lambda1); qubo.add(h_down_it.power2(), ramp_down_lambda2)

start = timeit.default_timer()
solver = QUBOSolverCPU(
    optimization_method='parallel_tempering',
    number_runs=100,
    number_replicas=100,
    number_iterations=10000, # Adjusted iterations
    temperature_sampling=True,
    scaling_action=ScalingAction.AUTO_SCALING,
)

solution_list = solver.minimize(qubo)
stop = timeit.default_timer()

print(f"Solver finished in {stop - start:.2f} seconds.")

# --- Analyzing the Solution ---
solution = solution_list.get_minimum_energy_solution()
u_sol = solution['u'].data.astype(int)
zOn_sol = solution['zOn'].data.astype(int)
zOff_sol = solution['zOff'].data.astype(int)

# Calculate original objective
orig_obj = 0
for t in range(T):
    for i in range(N):
        orig_obj += zOn_sol[i, t] * C_startup[i+1]
        orig_obj += zOff_sol[i, t] * C_shutdown[i+1]
        orig_obj += u_sol[i, t] * (b_cost[i+1] * P_max[i+1] + c_cost[i+1])

# --- KEEP ---
print(f"\nLowest QUBO value found (objective + penalties): {solution.energy:.4f}")
# --- KEEP ---
print(f"Original Objective Cost (excluding penalties): {orig_obj:.4f}")

# --- KEEP: Feasibility Check ---
tolerance = 1e-6
def feasibility_check_original_constraints_full(sol_dict):
    # (Function definition remains unchanged)
    u_arr    = sol_dict['u'].data.astype(int)
    zOn_arr  = sol_dict['zOn'].data.astype(int)
    zOff_arr = sol_dict['zOff'].data.astype(int)
    violations = []
    # 1. Logic 1 (t=1)
    for i in range(N):
        diff = u_arr[i, 0] - u_prev[i+1] - (zOn_arr[i, 0] - zOff_arr[i, 0])
        if abs(diff) > tolerance: violations.append(f"Logic1 violated at t=1 for unit {i+1}: diff={diff}")
    # 2. Logic 1 (t>=2)
    for t in range(1, T):
        for i in range(N):
            diff = u_arr[i, t] - u_arr[i, t-1] - (zOn_arr[i, t] - zOff_arr[i, t])
            if abs(diff) > tolerance: violations.append(f"Logic1 violated at t={t+1} for unit {i+1}: diff={diff}")
    # 3. Logic 2
    for t in range(T):
        for i in range(N):
            if zOn_arr[i, t] + zOff_arr[i, t] > 1 + tolerance: violations.append(f"Logic2 violated at t={t+1} for unit {i+1}: zOn={zOn_arr[i,t]}, zOff={zOff_arr[i,t]}")
    # 4. Demand Balance
    for t in range(T):
        total_production = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        if total_production < D[t+1] - tolerance: violations.append(f"Demand balance violated at t={t+1}: Gen={total_production:.1f}, Demand={D[t+1]}")
    # 5. Ramp-Up
    for i in range(N): # t=1
        production_increase = P_max[i+1] * (u_arr[i, 0] - u_prev[i+1])
        if production_increase > R_up[i+1] + tolerance: violations.append(f"Ramp-up violated at t=1 for unit {i+1}: increase = {production_increase:.1f}, limit = {R_up[i+1]}")
    for t in range(1, T): # t>=2
        for i in range(N):
            production_increase = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            if production_increase > R_up[i+1] + tolerance: violations.append(f"Ramp-up violated at t={t+1} for unit {i+1}: increase = {production_increase:.1f}, limit = {R_up[i+1]}")
    # 6. Ramp-Down
    for i in range(N): # t=1
        production_decrease = P_max[i+1] * (u_prev[i+1] - u_arr[i, 0])
        if production_decrease > R_down[i+1] + tolerance: violations.append(f"Ramp-down violated at t=1 for unit {i+1}: decrease = {production_decrease:.1f}, limit = {R_down[i+1]}")
    for t in range(1, T): # t>=2
        for i in range(N):
            production_decrease = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            if production_decrease > R_down[i+1] + tolerance: violations.append(f"Ramp-down violated at t={t+1} for unit {i+1}: decrease = {production_decrease:.1f}, limit = {R_down[i+1]}")
    return violations

min_energy_violations = feasibility_check_original_constraints_full(solution)

# --- KEEP ---
print("\n--- Feasibility Check ---")
if not min_energy_violations:
    print("Minimum energy solution is FEASIBLE with respect to ALL original constraints.")
else:
    print("Minimum energy solution is INFEASIBLE with respect to original constraints:")
    for v in min_energy_violations:
        print(f"  - {v}")

# --- KEEP: Variable configuration ---
print("\nSolution Variables (Lowest Energy):")
print("u (unit on/off):")
print(u_sol)
print("zOn (startup decisions):")
print(zOn_sol)
print("zOff (shutdown decisions):")
print(zOff_sol)

# (Removed penalty contribution analysis and tuning guidance)
print("\n--- End of Script ---") # Added for clarity