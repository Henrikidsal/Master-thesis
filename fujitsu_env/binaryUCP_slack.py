##### This is the all binary problem
##### It is solved with the original slack variable method
##### The inputs are modified so that the ramping constraints are effectively removed
##### The solver finds feasible solutions, but not optimal
##### It focuses to much on flipping slack variables
##### It cannot get the slack values correctly, due to heuristic nature.

##### This first part imports the necessary tools #####
from dadk.BinPol import *  
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction
import math
import timeit

##### This part defines the parameters #####

# Number of time periods and generating units
T = 3
N = 3   

# Previous time period status (initial conditions): 
# Unit 1: off, Unit 2: off, Unit 3: on.
u_prev = {1: 0, 2: 0, 3: 1}

# Penalty weights for penalty terms (temporary values)
logic1_penalty = 7000000       #700
logic2_penalty = 7000000       #700
demand_penalty = 10000      #4
ramp_up_penalty = 1000      #0.3
ramp_down_penalty = 1000  #0.3

# parameters
P_max = {1:350, 2:200, 3:140}
D = {1: 160, 2: 500, 3: 400}
C_startup = {1:20, 2:18, 3:5}
C_shutdown = {1:0.5, 2:0.3, 3:1.0}
b_cost = {1:0.1, 2:0.125, 3:0.15}
c_cost = {1:5, 2:7, 3:6}
R_up = {1:350, 2:200, 3:140}
R_down = {1:350, 2:200, 3:140}      

##### This part finds out how many bits are needed for slack variables #####

# For the demand slack, there are different number of slack variables per time period
total_Pmax = sum(P_max.values())
demand_bits = {} 
for t in range(1, T+1):
    max_slack = total_Pmax - D[t]
    demand_bits[t] = int(math.ceil(math.log2(max_slack + 1)))
    print(f"Time {t}: Demand slack max = {max_slack}, bits = {demand_bits[t]}")

# For the ramp-up slack, we can use the same number of bits for all time periods
max_slack_ramp = max(R_up[i] + P_max[i] for i in P_max)
K_ramp = int(math.ceil(math.log2(max_slack_ramp + 1)))
print("Ramp-up slack bits (uniform):", K_ramp)

# For the ramp-down slack, again we can use the same number of bits for all time periods
max_slack_ramp_down = max(R_down[i] + P_max[i] for i in P_max)
K_ramp_down = int(math.ceil(math.log2(max_slack_ramp_down + 1)))
print("Ramp-down slack bits (uniform):", K_ramp_down)

##### Here the shapes of the variables are defined #####
##### Also, the variables are gathered together in a var_shape_set #####

# On/off variable
u_shape    = BitArrayShape(name='u', shape=(N, T))

# Startup and shutdown variables
zOn_shape  = BitArrayShape(name='zOn', shape=(N, T))
zOff_shape = BitArrayShape(name='zOff', shape=(N, T))

s_shapes = []
for t in range(1, T+1):
    s_shapes.append(BitArrayShape(name=f's_{t}', shape=(demand_bits[t],)))

sUp_shape = BitArrayShape(name='sUp', shape=(N, T - 1, K_ramp))
sDown_shape = BitArrayShape(name='sDown', shape=(N, T - 1, K_ramp_down))

var_shape_set = VarShapeSet(u_shape, zOn_shape, zOff_shape, *s_shapes, sUp_shape, sDown_shape)
BinPol.freeze_var_shape_set(var_shape_set)

##### Here the QUBO is built using Binary polynomial (BinPol) objects #####

# Create an empty QUBO object
qubo = BinPol()

# Add terms for unit generation, startup, and shutdown costs over time periods and units.
for t in range(T):
    for i in range(N):
        cost_u = b_cost[i+1] * P_max[i+1] + c_cost[i+1]  # Unit generation cost: b*P_max + c
        qubo.add_term(cost_u, ('u', i, t))
        qubo.add_term(C_startup[i+1], ('zOn', i, t))
        qubo.add_term(C_shutdown[i+1], ('zOff', i, t))

# --- Startup/Shut-down Logic Constraints ---
# For t = 0, use the previous period status.
for i in range(N):
    # Enforce: u[i,0] - u_prev - zOn[i,0] + zOff[i,0] = 0
    poly = (BinPol()
            .add_term(1, ('u', i, 0))
            .add_term(-1, ('zOn', i, 0))
            .add_term(1, ('zOff', i, 0))
            .add_term(-u_prev[i+1]))  # constant term from previous state
    qubo.add(poly.power2(), logic1_penalty)

# For t >= 1, use the previous time period decision variables.
for t in range(1, T):
    for i in range(N):
        poly = (BinPol()
                .add_term(1, ('u', i, t))
                .add_term(-1, ('u', i, t-1))
                .add_term(-1, ('zOn', i, t))
                .add_term(1, ('zOff', i, t)))
        qubo.add(poly.power2(), logic1_penalty)

# --- Mutual Exclusion Constraint: zOn * zOff <= 1 ---
for t in range(T):
    for i in range(N):
        qubo.add_term(logic2_penalty, ('zOn', i, t), ('zOff', i, t))

# --- Demand Balance Constraint: sum_i (P_max[i] * u[i,t]) - slack = D[t] ---
for t in range(T):
    poly = BinPol()
    for i in range(N):
        poly.add_term(P_max[i+1], ('u', i, t))
    bits_length = demand_bits[t+1]
    for k in range(bits_length):
        poly.add_term(- (2 ** k), (f's_{t+1}', k))
    poly.add_term(-D[t+1])
    qubo.add(poly.power2(), demand_penalty)

# --- Ramp-Up Constraint: P_max[i]*(u[i,t] - u[i,t-1]) + slack_up = R_up[i] ---
for t in range(1, T):
    for i in range(N):
        poly = (BinPol()
                .add_term(P_max[i+1], ('u', i, t))
                .add_term(-P_max[i+1], ('u', i, t-1)))
        for k in range(K_ramp):
            poly.add_term(2 ** k, ('sUp', i, t - 1, k))
        poly.add_term(-R_up[i+1])
        qubo.add(poly.power2(), ramp_up_penalty)

# --- Ramp-Down Constraint: P_max[i]*(u[i,t-1] - u[i,t]) + slack_down = R_down[i] ---
for t in range(1, T):
    for i in range(N):
        poly = (BinPol()
                .add_term(P_max[i+1], ('u', i, t-1))
                .add_term(-P_max[i+1], ('u', i, t)))
        for k in range(K_ramp_down):
            poly.add_term(2 ** k, ('sDown', i, t - 1, k))
        poly.add_term(-R_down[i+1])
        qubo.add(poly.power2(), ramp_down_penalty)

start = timeit.default_timer()
# --- Solving the QUBO ---
solver = QUBOSolverCPU(
    #optimization_method='annealing',
    optimization_method='parallel_tempering',
    number_runs=128,
    number_replicas=128,
    number_iterations=10000,
    temperature_sampling=True,
    #temperature_start=1.030992598e+04,
    #temperature_end=1.0e+03,
    #temperature_mode=0,
    #temperature_interval=5,
    #offset_increase_rate=4.7578e+03,
    #pt_temperature_model='Exponential',
    #pt_replica_exchange_model='Neighbours',
    #random_seed=42,
    #scaling_factor=1.0
    scaling_action=ScalingAction.AUTO_SCALING,
)



solution_list = solver.minimize(qubo)


# Uncomment the following lines to print graphs or the solution list
# solution_list.display_graphs(figsize=(9.0, 7.0), file=None)
# print(solution_list)

# Retrieve the minimum energy solution
solution = solution_list.get_minimum_energy_solution()

# Extract the solution values
u_sol = solution['u'].data.astype(int)
zOn_sol = solution['zOn'].data.astype(int)
zOff_sol = solution['zOff'].data.astype(int)

# Calculate the original objective value (costs only, not with penalty terms)
orig_obj = 0
for t in range(T):
    for i in range(N):
        orig_obj += zOn_sol[i, t] * C_startup[i+1]
        orig_obj += zOff_sol[i, t] * C_shutdown[i+1]
        orig_obj += u_sol[i, t] * (b_cost[i+1] * P_max[i+1] + c_cost[i+1])

#Calculate if any constriants are violated, only this.
stop = timeit.default_timer()
print('Time: ', stop - start)
# Uncomment below to display intermediate results
'''
print("Optimal solution:")
print("u (unit on/off):")
print(u_sol)
print("zOn (startup decisions):")
print(zOn_sol)
print("zOff (shutdown decisions):")
print(zOff_sol)
for t in range(1, T+1):
    print(f"s_{t} (demand slack bits):")
    print(solution[f's_{t}'].data)
print("sUp (ramp-up slack bits):")
print(solution['sUp'].data)
print("sDown (ramp-down slack bits):")
print(solution['sDown'].data)
'''

print("Full QUBO value (with penalties) (from the lowest energy solution):", solution.energy)
print("The original objective value (from the lowest energy solution):", orig_obj)


print("\n###########################################################\n")
##### Feasibility Check: Verify that the original constraints are satisfied #####

tolerance = 1e-6  # Tolerance for floating-point comparisons

# Get the sorted list of solutions (lowest energy first)
sorted_solutions = solution_list.get_sorted_solution_list()

def feasibility_check_original(sol):
    # Convert decision variables to integer arrays
    u_arr    = sol['u'].data.astype(int)
    zOn_arr  = sol['zOn'].data.astype(int)
    zOff_arr = sol['zOff'].data.astype(int)
    
    # 1. Unit Transition Constraint for t = 0: u[i,0] - u_prev = zOn[i,0] - zOff[i,0]
    for i in range(N):
        diff = u_arr[i, 0] - u_prev[i+1] - (zOn_arr[i, 0] - zOff_arr[i, 0])
        if abs(diff) > tolerance:
            return False

    # 2. Unit Transition Constraint for t >= 1: u[i,t] - u[i,t-1] = zOn[i,t] - zOff[i,t]
    for t in range(1, T):
        for i in range(N):
            diff = u_arr[i, t] - u_arr[i, t-1] - (zOn_arr[i, t] - zOff_arr[i, t])
            if abs(diff) > tolerance:
                return False

    # 3. Mutual Exclusion Constraint: zOn[i,t] + zOff[i,t] <= 1
    for t in range(T):
        for i in range(N):
            if zOn_arr[i, t] + zOff_arr[i, t] > 1 + tolerance:
                return False

    # 4. Demand Balance Constraint: Total production must be >= demand.
    for t in range(T):
        total_production = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        if total_production < D[t+1] - tolerance:
            return False

    # 5. Ramp-Up Constraint: P_max[i]*(u[i,t] - u[i,t-1]) <= R_up[i] for t>=1.
    for t in range(1, T):
        for i in range(N):
            production_increase = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            if production_increase > R_up[i+1] + tolerance:
                return False

    # 6. Ramp-Down Constraint: P_max[i]*(u[i,t-1] - u[i,t]) <= R_down[i] for t>=1.
    for t in range(1, T):
        for i in range(N):
            production_decrease = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            if production_decrease > R_down[i+1] + tolerance:
                return False

    return True


print("Looking for feasible solutions...")

feasible_solution = None
#printing lentgh of sorted_solutions
print(len(sorted_solutions))
# Iterate through the sorted solutions (lowest energy first). If a feasible solution is found, store it and break.
for sol in sorted_solutions:
    if feasibility_check_original(sol):
        feasible_solution = sol
        print("Feasible solution found! Original objective value:", orig_obj)
        print("\nOptimal u (unit on/off) values:")
        print(sol['u'].data.astype(int))
        print("\nOptimal zOn (startup decisions):")
        print(sol['zOn'].data.astype(int))
        print("\nOptimal zOff (shutdown decisions):")
        print(sol['zOff'].data.astype(int))
        break

if feasible_solution is None:
    print("No feasible solutions were found in the solution list.")
    print("\nAnalyzing which original constraints are violated for the lowest energy solution:")

    u_arr    = solution['u'].data.astype(int)
    zOn_arr  = solution['zOn'].data.astype(int)
    zOff_arr = solution['zOff'].data.astype(int)

    # Check initial time (t=0) startup/shutdown consistency
    for i in range(N):
        diff = u_arr[i, 0] - u_prev[i+1] - (zOn_arr[i, 0] - zOff_arr[i, 0])
        if abs(diff) > tolerance:
            print(f"Logic1 violated at t=1 for unit {i+1}: difference = {diff}")

    # Check unit transition for t >= 1
    for t in range(1, T):
        for i in range(N):
            diff = u_arr[i, t] - u_arr[i, t-1] - (zOn_arr[i, t] - zOff_arr[i, t])
            if abs(diff) > tolerance:
                print(f"Logic1 violated on (unit {i+1}, time {t+1}): difference = {diff}")

    # Mutual Exclusion Constraint
    for t in range(T):
        for i in range(N):
            total = zOn_arr[i, t] + zOff_arr[i, t]
            if total > 1 + tolerance:
                print(f"Logic2 violated on (unit {i+1}, time {t+1}): sum = {total}")

    # Demand Balance Constraint
    for t in range(T):
        total_production = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        if total_production < D[t+1] - tolerance:
            print(f"Demand balance violated at time {t+1}: total_production = {total_production}, demand = {D[t+1]}")

    # Ramp-Up Constraint
    for t in range(1, T):
        for i in range(N):
            production_increase = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            if production_increase > R_up[i+1] + tolerance:
                print(f"Ramp-up violated on (unit {i+1}, time {t+1}): production increase = {production_increase}, limit = {R_up[i+1]}")

    # Ramp-Down Constraint
    for t in range(1, T):
        for i in range(N):
            production_decrease = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            if production_decrease > R_down[i+1] + tolerance:
                print(f"Ramp-down violated on (unit {i+1}, time {t+1}): production decrease = {production_decrease}, limit = {R_down[i+1]}")

# Checking constraints including slack variables for the feasible solution
if feasible_solution is not None:
    print("\nChecking constraints including slack variables for the feasible solution:")

    # Extract solution arrays
    u_arr = feasible_solution['u'].data.astype(int)
    zOn_arr = feasible_solution['zOn'].data.astype(int)
    zOff_arr = feasible_solution['zOff'].data.astype(int)

    # Tolerance for floating-point comparisons
    tolerance = 1e-6

    # Check Demand Balance with Slack: sum_i P_max[i] * u[i,t] - s_t = D[t]
    for t in range(T):
        total_production = sum(P_max[i+1] * u_arr[i, t] for i in range(N))
        s_t_bits = feasible_solution[f's_{t+1}'].data
        s_t_value = sum((2 ** k) * int(s_t_bits[k]) for k in range(demand_bits[t+1]))
        lhs = total_production - s_t_value
        rhs = D[t+1]
        if abs(lhs - rhs) > tolerance:
            print(f"Demand balance with slack violated at time {t+1}: {lhs} != {rhs} (slack = {s_t_value})")

    # Check Ramp-Up with Slack: P_max[i]*(u[i,t] - u[i,t-1]) + s_up[i,t] = R_up[i]
    for t in range(1, T):
        for i in range(N):
            production_increase = P_max[i+1] * (u_arr[i, t] - u_arr[i, t-1])
            s_up_bits = feasible_solution['sUp'].data[i, t-1, :]
            s_up_value = sum((2 ** k) * int(s_up_bits[k]) for k in range(K_ramp))
            lhs = production_increase + s_up_value
            rhs = R_up[i+1]
            if abs(lhs - rhs) > tolerance:
                print(f"Ramp-up with slack violated for unit {i+1} at time {t+1}: {lhs} != {rhs} (slack = {s_up_value})")

    # Check Ramp-Down with Slack: P_max[i]*(u[i,t-1] - u[i,t]) + s_down[i,t] = R_down[i]
    for t in range(1, T):
        for i in range(N):
            production_decrease = P_max[i+1] * (u_arr[i, t-1] - u_arr[i, t])
            s_down_bits = feasible_solution['sDown'].data[i, t-1, :]
            s_down_value = sum((2 ** k) * int(s_down_bits[k]) for k in range(K_ramp_down))
            lhs = production_decrease + s_down_value
            rhs = R_down[i+1]
            if abs(lhs - rhs) > tolerance:
                print(f"Ramp-down with slack violated for unit {i+1} at time {t+1}: {lhs} != {rhs} (slack = {s_down_value})")

    print("Slack variable constraint checks completed.")
else:
    print("No feasible solution was found, so slack variable checks cannot be performed.")