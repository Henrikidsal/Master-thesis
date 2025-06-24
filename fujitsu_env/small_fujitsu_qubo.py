from dadk.BinPol import BinPol, BitArrayShape, VarShapeSet
from dadk.QUBOSolverCPU import QUBOSolverCPU, ScalingAction
import time
import pandas as pd
import os


# 1-A ─────────────────────────────────────────────────────────────────────────
# Data
T_periods_list = [1, 2, 3]
generators_list = [1, 2, 3]

D_data = {1: 160, 2: 500, 3: 400}
P_max_data = {1: 350, 2: 200, 3: 140}
C_fixed_data = {1: 5, 2: 7, 3: 6}
C_startup_data = {1: 20, 2: 18, 3: 5}
C_shutdown_data = {1: 0.5, 2: 0.3, 3: 1.0}
C_variable_data = {1:0.100,2:0.125,3:0.150}
u_initial_data = {1: 0, 2: 0, 3: 1} # Initial state of generators (on=1, off=0)

# pandas dataframe with obj_value, feasibility, and elapsed_time
df = pd.DataFrame(columns=["obj_value", "feasibility", "elapsed_time", "num_iterations", "penalty_method"])

def main(iterations, penalty_method):

    global df

    # 1-B ─────────────────────────────────────────────────────────────────────────
    # --- MASTER SWITCH FOR DEMAND PENALTY ---
    #PENALTY_METHOD = "slack"
    #num_iter = 5_000

    PENALTY_METHOD = penalty_method
    num_iter = iterations


    # 2 ───────────────────────────────────────────────────────────────────────────
    # Penalty coefficients for QUBO formulation
    if PENALTY_METHOD == "slack": #50-5_000, 50-10_000, 50-100_000 and 10-500_000
        print("Using 'Slack Variable' method for demand constraint.")
        λ_logic1 = 1e6 # Penalty for Logic 1 violation
        λ_logic2 = 1e5 # Penalty for Logic 2 violation
        λ_demand = 100 # Penalty for demand violation with slack variables
        K_SLACK = 8    # Number of bits for slack variable
    elif PENALTY_METHOD == "unbalanced": #50-5_000, 50-10_000, 50-100_000 and 10-500_000
        print("Using 'Unbalanced Penalty' method for demand constraint.")
        λ_logic1 = 100 # Penalty for Logic 1 violation
        λ_logic2 = 100 # Penalty for Logic 2 violation
        λ_demand1 = 150 # Linear penalty for demand (to make it positive)
        λ_demand2 = 3   # Quadratic penalty for demand
    else:
        raise ValueError(f"Invalid PENALTY_METHOD: '{PENALTY_METHOD}'. Choose 'slack' or 'unbalanced'.")


    # 3 ───────────────────────────────────────────────────────────────────────────
    # Variable-shape definitions for binary variables
    n_gen = len(generators_list)
    n_time = len(T_periods_list)
    gen_map = {g: idx for idx, g in enumerate(generators_list)} # Map generator ID to index
    time_map = {t: idx for idx, t in enumerate(T_periods_list)} # Map time period to index

    # Define BitArrayShapes for different variable types
    u_shape = BitArrayShape('u', (n_gen, n_time))       # Generator on/off status
    zON_shape = BitArrayShape('zON', (n_gen, n_time))   # Generator startup status
    zOFF_shape = BitArrayShape('zOFF', (n_gen, n_time)) # Generator shutdown status

    # Define VarShapeSet based on the penalty method
    if PENALTY_METHOD == "slack":
        s_shape = BitArrayShape('s', (n_time, K_SLACK)) # Slack variables for demand
        vss = VarShapeSet(u_shape, zON_shape, zOFF_shape, s_shape)
        def S(t, k): return ('s', time_map[t], k) # Helper function to access slack variable
    elif PENALTY_METHOD == "unbalanced":
        vss = VarShapeSet(u_shape, zON_shape, zOFF_shape)

    # Freeze the variable shape set to register variables in BinPol
    BinPol.freeze_var_shape_set(vss)

    # Helper functions to access variables by their logical names and indices
    def U(i,t): return ('u', gen_map[i], time_map[t])
    def ZON(i,t): return ('zON', gen_map[i], time_map[t])
    def ZOFF(i,t): return ('zOFF',gen_map[i], time_map[t])

        



    # 4 ───────────────────────────────────────────────────────────────────────────
    # Build the QUBO (Quadratic Unconstrained Binary Optimization)
    Q = BinPol()

    # 4-A: Cost Terms (Objective Function)
    # Add fixed, startup, shutdown, and variable costs to the objective
    for i in generators_list:
        for t in T_periods_list:
            Q.add_term(C_startup_data[i], ZON(i,t))                     # Startup cost
            Q.add_term(C_shutdown_data[i], ZOFF(i,t))                   # Shutdown cost
            Q.add_term(C_variable_data[i] * P_max_data[i], U(i,t))      # Variable production cost (proportional to P_max)
            Q.add_term(C_fixed_data[i], U(i,t))                         # Fixed cost for being online

    # 4-B: Logic 1 Penalty: u_it - u_i(t-1) - z_it^ON + z_it^Off = 0
    # This constraint ensures consistency between generator status and startup/shutdown events.
    for i in generators_list:
        for t in T_periods_list:
            logic1_expr = BinPol()
            logic1_expr.add_term(1, U(i,t))       # u_it
            logic1_expr.add_term(-1, ZON(i,t))    # -z_it^ON
            logic1_expr.add_term(1, ZOFF(i,t))    # +z_it^Off

            # Handle initial state for t=1, otherwise use u_i(t-1)
            if t == 1:
                logic1_expr.add_term(-u_initial_data[i]) # -u_i(t-1) which is u_initial
            else:
                logic1_expr.add_term(-1, U(i, t-1))      # -u_i(t-1)

            Q.add(logic1_expr.power(2), scalar=λ_logic1) # Add quadratic penalty term

    # 4-C: Logic 2 Penalty: z_it^ON + z_it^Off <= 1
    # This constraint ensures that a generator cannot both start up and shut down simultaneously.
    for i in generators_list:
        for t in T_periods_list:
            Q.add_term(λ_logic2, ZON(i,t), ZOFF(i,t)) # If both are 1, penalty is added

    # 4-D: Demand Penalty - Conditional block based on PENALTY_METHOD
    if PENALTY_METHOD == "slack":
        for t in T_periods_list:
            diff = BinPol()
            # Sum of maximum production minus demand
            for i in generators_list:
                diff.add_term(P_max_data[i], U(i,t))
            diff.add_term(-D_data[t])

            # Subtract slack variables (sum_k 2^k * s_tk)
            for k in range(K_SLACK):
                diff.add_term(-(2**k), S(t,k))
            Q += diff.power(2) * λ_demand # Add quadratic penalty

    elif PENALTY_METHOD == "unbalanced":
        for t in T_periods_list:
            # Define h(u) = Production - Demand
            h_poly = BinPol()
            for i in generators_list:
                h_poly.add_term(P_max_data[i], U(i, t)) # Add Production
            h_poly.add_term(-D_data[t]) # Subtract Demand

            # Add the penalty: -λ₁*h(u) + λ₂*h(u)² to the QUBO
            # This formulation encourages h(u) to be positive and close to zero
            Q.add(h_poly, scalar=-λ_demand1)
            Q.add(h_poly.power(2), scalar=λ_demand2)


    # Solve with Fujitsu Digital Annealer (simulated by QUBOSolverCPU)
    solver = QUBOSolverCPU(
        optimization_method='parallel_tempering', # A metaheuristic optimization algorithm
        number_runs=1,
        number_replicas=128,
        number_iterations=num_iter,
        temperature_sampling=True,
        scaling_action=ScalingAction.AUTO_SCALING, # Automatically scale the QUBO problem
    )
    print("\nRunning Fujitsu Digital Annealer (CPU simulation) …")
    #start timer
    start_time = time.time()
    res = solver.minimize(qubo=Q)

    #end timer
    end_time = time.time()

    # time im seconds
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.2f} seconds\n")

    # 6 & 7 ───────────────────────────────────────────────────────────────────────
    # Results processing
    if not res.solutions:
        raise RuntimeError("DADK returned no solution")

    best = res.get_solution_list()[0] # Get the best solution found
    u_bits = best.extract_bit_array('u').data       # Extract u variables
    zON_bits = best.extract_bit_array('zON').data   # Extract zON variables
    zOFF_bits = best.extract_bit_array('zOFF').data # Extract zOFF variables

    # Helper function to get bit value for (generator, time)
    def bit(arr,i,t): return int(arr[gen_map[i], time_map[t]])

    # Calculate true cost based on the solution
    true_cost = 0.0
    for i in generators_list:
        for t in T_periods_list:
            true_cost += bit(zON_bits,i,t) * C_startup_data[i]
            true_cost += bit(zOFF_bits,i,t) * C_shutdown_data[i]
            true_cost += bit(u_bits,i,t) * P_max_data[i] * C_variable_data[i]
            true_cost += bit(u_bits,i,t) * C_fixed_data[i]

    print(f"\nBest Solution Found (Fujitsu DADK using '{PENALTY_METHOD}' method)")
    print(f"Total Cost: ${true_cost:.2f}")
    print(f"QUBO Objective Value (energy): {best.energy:.2f}\n")

    # Print Generator Schedules
    print("Generator Schedules (u_it - 1=ON, 0=OFF):")
    print("Gen\\Time |  1  |  2  |  3  |")
    print("----------------------------")
    for i in generators_list:
        print(f"  G{i}     |  {bit(u_bits,i,1)}  |  {bit(u_bits,i,2)}  |  {bit(u_bits,i,3)}  |")

    # Print Production Plan and Demand
    print("\nProduction Plan (P_i^max * u_it):")
    print("Gen\\Time |   1   |   2   |   3   |")
    print("-----------------------------------")
    tot_prod = {t:0 for t in T_periods_list} # Dictionary to store total production per time period
    for i in generators_list:
        row = f"  G{i}     |"
        for t in T_periods_list:
            prod = bit(u_bits,i,t) * P_max_data[i]
            tot_prod[t] += prod
            row += f" {prod:5.0f} |"
        print(row)
    print("-----------------------------------")
    dem_row = "Demand   |"
    tot_row = "TotalPrd |"
    for t in T_periods_list:
        dem_row += f" {D_data[t]:5.0f} |"
        tot_row += f" {tot_prod[t]:5.0f} |"
    print(dem_row)
    print(tot_row)

    # 8 ───────────────────────────────────────────────────────────────────────
    # Feasibility Check against Original Constraints

    print("\n--- Feasibility Check ---")
    is_feasible = True
    violations = []

    # Check Logic 1: u_it - u_i(t-1) - z_it^ON + z_it^Off = 0
    # This logic ensures that u_it correctly reflects startup/shutdown events.
    for i in generators_list:
        for t in T_periods_list:
            u_prev = u_initial_data[i] if t == 1 else bit(u_bits, i, t-1)
            u_curr = bit(u_bits, i, t)
            z_on = bit(zON_bits, i, t)
            z_off = bit(zOFF_bits, i, t)

            # The equation must be 0 for feasibility
            # u_curr - u_prev - z_on + z_off == 0
            if u_curr - u_prev - z_on + z_off != 0:
                is_feasible = False
                violations.append(f"Logic 1 (Gen {i}, Time {t}): {u_curr} - {u_prev} - {z_on} + {z_off} = {u_curr - u_prev - z_on + z_off} (Expected 0)")

    # Check Logic 2: z_it^ON + z_it^Off <= 1
    # A generator cannot both start up and shut down at the same time.
    for i in generators_list:
        for t in T_periods_list:
            z_on = bit(zON_bits, i, t)
            z_off = bit(zOFF_bits, i, t)
            if z_on + z_off > 1:
                is_feasible = False
                violations.append(f"Logic 2 (Gen {i}, Time {t}): zON + zOFF = {z_on + z_off} (Expected <= 1)")

    # Check Demand: sum(P_max_data[i] * u_it) >= D_t
    # Total production must meet or exceed demand for each time period.
    for t in T_periods_list:
        production_at_t = 0
        for i in generators_list:
            production_at_t += bit(u_bits, i, t) * P_max_data[i]
        if production_at_t < D_data[t]:
            is_feasible = False
            violations.append(f"Demand (Time {t}): Production {production_at_t} < Demand {D_data[t]} (Violation)")

    # Print feasibility status
    if is_feasible:
        print("The solution is FEASIBLE against all original constraints.")
    else:
        print("The solution is INFEASIBLE. Violations found:")
        for violation in violations:
            print(f"- {violation}")
    
    # Create a new DataFrame with the data you want to append
    new_row_df = pd.DataFrame([{
        "obj_value": true_cost,
        "feasibility": is_feasible,
        "elapsed_time": elapsed_time,
        "num_iterations": num_iter,
        "penalty_method": PENALTY_METHOD,
    }])

    # Use pd.concat to append the new row(s) to the DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    # if csv file exists, delete it and replace with new one
    csv_file = "fujitsu_small_results.csv"
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)

    

if __name__ == "__main__":

    df = pd.DataFrame(columns=["obj_value", "feasibility", "elapsed_time", "num_iterations", "penalty_method"])

    for _ in range(100):
        main(iterations=5000, penalty_method="slack")
    
    for _ in range(100):
        main(iterations=5000, penalty_method="unbalanced")
    
    for _ in range(100):
        main(iterations=20_000, penalty_method="slack")
    
    for _ in range(100):
        main(iterations=20_000, penalty_method="unbalanced")

    for _ in range(100):    
        main(iterations=100_000, penalty_method="slack")
    
    for _ in range(100):
        main(iterations=100_000, penalty_method="unbalanced")

