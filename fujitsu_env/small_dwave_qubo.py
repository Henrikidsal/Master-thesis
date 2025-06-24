import dimod
from neal import SimulatedAnnealingSampler
import numpy as np # Needed for np.iinfo
import time
import os 
import pandas as pd



def main(iterations):

    global df

    # Time periods
    T_periods_list = [1, 2, 3]

    # Demand
    D_data = {1: 160, 2: 500, 3: 400}

    # Generators
    generators_list = [1, 2, 3]

    # Generator Parameters
    P_max_data = {1: 350, 2: 200, 3: 140}          # Capacity (P_i^max) [MW]
    C_fixed_data = {1: 5, 2: 7, 3: 6}              # Fixed cost (C_i^Fixed) [$]
    C_startup_data = {1: 20, 2: 18, 3: 5}          # Start-up cost (C_i^Startup) [$]
    C_shutdown_data = {1: 0.5, 2: 0.3, 3: 1.0}     # Shut-down cost (C_i^Shutdown) [$]
    C_variable_data = {1: 0.100, 2: 0.125, 3: 0.150} # Variable cost (C_i^Variable) [$/MWh]

    # Initial state for t=0
    u_initial_data = {
        1: 0,  # Generator 1 was OFF
        2: 0,  # Generator 2 was OFF
        3: 1   # Generator 3 was ON
    }

    # QUBO Parameters from your formulation
    lambda_logic1 = 100
    lambda_logic2 = 100
    lambda_demand = 100

    # --- D-Wave Ocean SDK BQM Construction ---

    # Initialize BQM with BINARY variables
    bqm = dimod.BinaryQuadraticModel(vartype='BINARY')

    # Define variable names (string representation for BQM)
    u_vars = {}
    zON_vars = {}
    zOFF_vars = {}

    for i in generators_list:
        for t in T_periods_list:
            u_vars[(i, t)] = f'u_{i}_{t}'
            zON_vars[(i, t)] = f'zON_{i}_{t}'
            zOFF_vars[(i, t)] = f'zOFF_{i}_{t}'

    # --- Objective Function Components (as per your QUBO formulation) ---

    # Part 1: Original Operational Costs
    for t in T_periods_list:
        for i in generators_list:
            bqm.add_linear(zON_vars[(i,t)], C_startup_data[i])
            bqm.add_linear(zOFF_vars[(i,t)], C_shutdown_data[i])
            bqm.add_linear(u_vars[(i,t)], C_variable_data[i] * P_max_data[i])
            bqm.add_linear(u_vars[(i,t)], C_fixed_data[i])

    # Part 2: Penalty for Logic 1
    for i in generators_list:
            for t in T_periods_list:
                terms = [(u_vars[i, t], 1), (zON_vars[i, t], -1), (zOFF_vars[i, t], 1)]
                constant = -u_initial_data[i] if t == 1 else 0
                if t > 1:
                    terms.append((u_vars[i, t-1], -1))
                bqm.add_linear_equality_constraint(terms, lagrange_multiplier=lambda_logic1, constant=constant)

    # Part 3: Penalty for Logic 2
    # +λ^logic2*∑_(t=1)^T▒∑_(i=1)^N▒(〖z_(i,t)^Off*z〗_(i,t)^On )
    for i in generators_list:
        for t in T_periods_list:
            bqm.add_quadratic(zOFF_vars[(i,t)], zON_vars[(i,t)], lambda_logic2)

    max_total_production = int(sum(P_max_data.values()))

    for t in T_periods_list:
        terms_for_demand_ineq = []
        for i_gen in generators_list:
            terms_for_demand_ineq.append((u_vars[(i_gen, t)], int(P_max_data[i_gen])))

        # Constraint: D_data[t] <= sum(P_max[i]*u_it) <= max_total_production
        # The function handles lb <= sum(terms) + constant <= ub
        # Here, constant is 0.
        bqm.add_linear_inequality_constraint(
            terms=terms_for_demand_ineq,
            lagrange_multiplier=lambda_demand,
            label=f"demand_t{t}",
            constant=0, # The constant part of the sum expression
            lb=int(D_data[t]),
            ub=max_total_production,
            penalization_method="slack" 
        )

    # --- Solve with Simulated Annealer ---
    sampler = SimulatedAnnealingSampler()
    #start timer
    start_time = time.time()
    sampleset = sampler.sample(bqm, num_reads=1, num_sweeps=iterations, label='Binary_UCP_QUBO_DWave_SA')
    # 50-100_000, 50-500_000, 50-5_000_000, 50-10_000_000.

    #end timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nD-Wave Simulated Annealing completed in {elapsed_time:.2f} seconds.")

    # --- Feasibility Check Function ---
    def check_feasibility(u_solution_dict, zON_solution_dict, zOFF_solution_dict):
        """
        Checks if a given solution (u, zON, zOFF dictionaries) is feasible
        against the original constraints.
        Returns (is_feasible, violations_list).
        """
        is_feasible_local = True
        violations_local = []

        # Check Logic 1: u_it - u_i(t-1) - z_it^ON + z_it^Off = 0
        for i in generators_list:
            for t in T_periods_list:
                u_prev = u_initial_data[i] if t == 1 else u_solution_dict.get((i, t-1), 0)
                u_curr = u_solution_dict.get((i, t), 0)
                z_on = zON_solution_dict.get((i, t), 0)
                z_off = zOFF_solution_dict.get((i, t), 0)

                # The equation must be 0 for feasibility
                # u_curr - u_prev - z_on + z_off == 0
                if u_curr - u_prev - z_on + z_off != 0:
                    is_feasible_local = False
                    violations_local.append(f"Logic 1 (Gen {i}, Time {t}): {u_curr} - {u_prev} - {z_on} + {z_off} = {u_curr - u_prev - z_on + z_off} (Expected 0)")

        # Check Logic 2: z_it^ON + z_it^Off <= 1
        # A generator cannot both start up and shut down at the same time.
        for i in generators_list:
            for t in T_periods_list:
                z_on = zON_solution_dict.get((i, t), 0)
                z_off = zOFF_solution_dict.get((i, t), 0)
                if z_on + z_off > 1:
                    is_feasible_local = False
                    violations_local.append(f"Logic 2 (Gen {i}, Time {t}): zON + zOFF = {z_on + z_off} (Expected <= 1)")

        # Check Demand: sum(P_max_data[i] * u_it) >= D_t
        # Total production must meet or exceed demand for each time period.
        for t in T_periods_list:
            production_at_t = 0
            for i in generators_list:
                production_at_t += u_solution_dict.get((i, t), 0) * P_max_data[i]
            if production_at_t < D_data[t]:
                is_feasible_local = False
                violations_local.append(f"Demand (Time {t}): Production {production_at_t} < Demand {D_data[t]} (Violation)")
                
        return is_feasible_local, violations_local

    # --- Process and Print Results ---
    if sampleset and len(sampleset) > 0:
        # Get the lowest energy solution
        best_solution_record = sampleset.first
        sample = best_solution_record.sample
        energy = best_solution_record.energy
        print("lenght of sampleset:", len(sampleset))
        print("\nBest Solution Found (via D-Wave Simulated Annealing with inequality constraint)")
        print(f"QUBO Objective Value: {energy:.2f}")

        calculated_operational_cost = 0.0
        u_sol = {}
        zON_sol = {}
        zOFF_sol = {}

        for t_val in T_periods_list:
            for i_val in generators_list:
                u_s = sample.get(u_vars[(i_val, t_val)], 0)
                zON_s = sample.get(zON_vars[(i_val, t_val)], 0)
                zOFF_s = sample.get(zOFF_vars[(i_val, t_val)], 0)
                
                u_sol[(i_val, t_val)] = u_s
                zON_sol[(i_val, t_val)] = zON_s
                zOFF_sol[(i_val, t_val)] = zOFF_s

                calculated_operational_cost += zON_s * C_startup_data[i_val]
                calculated_operational_cost += zOFF_s * C_shutdown_data[i_val]
                calculated_operational_cost += C_variable_data[i_val] * P_max_data[i_val] * u_s
                calculated_operational_cost += C_fixed_data[i_val] * u_s
        
        print(f"Total Operational Cost (calculated from solution): ${calculated_operational_cost:.2f}")

        # --- Feasibility Check for the Best Solution ---
        print("\n--- Feasibility Check for Lowest Energy Solution ---")
        is_feasible, violations = check_feasibility(u_sol, zON_sol, zOFF_sol)

        if is_feasible:
            print("The lowest energy solution is FEASIBLE against all original constraints.")
        else:
            print("The lowest energy solution is INFEASIBLE. Violations found:")
            for violation in violations:
                print(f"- {violation}")

        print("\nGenerator Schedules (u_it):")
        print("Gen\\Time |  1  |  2  |  3  |")
        print("-----------------------------")
        for i_gen in generators_list:
            print(f"   G{i_gen}    |  {u_sol.get((i_gen,1),0):.0f}  |  {u_sol.get((i_gen,2),0):.0f}  |  {u_sol.get((i_gen,3),0):.0f}  |")

        print("\nGenerator Start-ups (z_it^ON):")
        print("Gen\\Time |  1  |  2  |  3  |")
        print("-----------------------------")
        for i_gen in generators_list:
            print(f"   G{i_gen}    |  {zON_sol.get((i_gen,1),0):.0f}  |  {zON_sol.get((i_gen,2),0):.0f}  |  {zON_sol.get((i_gen,3),0):.0f}  |")

        print("\nGenerator Shut-downs (z_it^OFF):")
        print("Gen\\Time |  1  |  2  |  3  |")
        print("-----------------------------")
        for i_gen in generators_list:
            print(f"   G{i_gen}    |  {zOFF_sol.get((i_gen,1),0):.0f}  |  {zOFF_sol.get((i_gen,2),0):.0f}  |  {zOFF_sol.get((i_gen,3),0):.0f}  |")

        print("\nProduction Plan (P_i^max * u_it):")
        print("Gen\\Time |   1   |   2   |   3   |")
        print("-----------------------------------")
        total_production = {t_p:0.0 for t_p in T_periods_list}
        for i_gen in generators_list:
            row = f"   G{i_gen}    |"
            for t_p_loop in T_periods_list:
                prod = u_sol.get((i_gen,t_p_loop),0) * P_max_data[i_gen]
                total_production[t_p_loop] += prod
                row += f" {prod:5.0f} |"
            print(row)
        print("-----------------------------------")
        demand_row = "Demand   |"
        total_prod_row = "TotalPrd |"
        for t_p_loop in T_periods_list:
            demand_row += f" {D_data[t_p_loop]:5.0f} |"
            total_prod_row += f" {total_production[t_p_loop]:5.0f} |"
        print(demand_row)
        print(total_prod_row)

        # --- Analyze and print the values of the slack variables ---
        print("\n" + "="*50)
        print("SLACK VARIABLE ANALYSIS (for the lowest energy solution)")
        print("="*50)
        
        for t_val in T_periods_list:
            # Reconstruct what the slack coefficients would have been for this time period
            max_total_production = int(sum(P_max_data.values()))
            ub_c = max_total_production # upper bound for production without considering demand
            lb_c = D_data[t_val] # lower bound for production (demand)

            # Calculate range for slack variables
            slack_upper_bound = ub_c - lb_c

            # This logic determines the binary representation of the slack variable value.
            # D-Wave automatically creates these variables and names them.
            # We're trying to infer what they *would* be called and their contribution.
            # This part is highly dependent on D-Wave's internal implementation for
            # `add_linear_inequality_constraint` with `penalization_method="slack"`.
            # The variables created are typically named 'slack_label_j' where j is the bit index.
            # Here, the label is "demand_t{t}".
            
            calculated_surplus = 0
            print(f"\n--- Slack Variables for t={t_val} ---")
            
            # Iterate through possible slack bits and check if they exist in the sample
            # D-Wave usually adds slacks up to log2(range), plus potentially one more for remainder.
            # This reconstruction of slack names might not be perfect for all cases but covers common ones.
            # A more robust approach would be to capture the BQM's auxiliary variables directly.
            
            # Estimate maximum number of slack bits for this constraint range
            if slack_upper_bound >= 0:
                # max_slack_bits is essentially ceil(log2(slack_upper_bound + 1))
                num_slack_variables_estimate = int(np.ceil(np.log2(slack_upper_bound + 1))) if slack_upper_bound > 0 else 0
            else:
                num_slack_variables_estimate = 0 # No positive range, no slacks

            # D-Wave's naming convention for slack variables when using add_linear_inequality_constraint
            # is typically "slack_LABEL_BITINDEX". Here LABEL is "demand_t{t_val}".
            
            # We need to consider how dimod translates the inequality.
            # `sum(P_max[i]*u_it) >= D_t` is `sum(P_max[i]*u_it) - D_t - S = 0` where S >= 0.
            # So S represents the surplus.
            
            # The penalty creates new variables. Let's try to infer them from the sample keys.
            # A more direct way to get slack values is to query `bqm.aux_variables`
            # if using the `GeneratedBQM` object returned by the constraint functions.

            slack_found = False
            for k_bit in range(num_slack_variables_estimate + 2): # Check a few extra bits just in case
                slack_var_name = f'slack_demand_t{t_val}_{k_bit}'
                if slack_var_name in sample:
                    bit_value = sample[slack_var_name]
                    coeff = 2**k_bit # For standard binary representation
                    calculated_surplus += bit_value * coeff
                    print(f"  {slack_var_name}: value = {bit_value:.0f}, coefficient = {coeff}")
                    slack_found = True
                # There might also be a 'remainder' slack variable with a non-power-of-2 coefficient.
                # D-Wave's `add_linear_inequality_constraint` uses a "one-hot" or "binary" representation internally.
                # The general form is usually `sum(x_i * C_i) - S = K`, where S is a sum of powers of 2.
                # It's challenging to perfectly reconstruct the specific slack variable names and their coefficients
                # without inspecting the BQM structure after compilation or knowing the exact internal naming.

            if not slack_found:
                print("  No explicit slack variables found in solution for this time period (constraint might be tight or trivially satisfied).")

            prod_at_t = total_production[t_val] # Assuming total_production dict is calculated
            print(f"\n  Production at t={t_val}: {prod_at_t:.0f}")
            print(f"  Demand at t={t_val}:   {D_data[t_val]:.0f}")
            print(f"  Calculated Surplus (from identified slacks): S = {calculated_surplus:.0f}")
            print(f"  CHECK: Production - Demand = {prod_at_t - D_data[t_val]:.0f}")
            # Note: For strict adherence, `prod_at_t - D_data[t_val]` should equal `calculated_surplus`.
            # Differences might indicate issues with slack variable reconstruction or
            # the solution not perfectly satisfying the softened constraint.

        print("="*50 + "\n")

    elif sampleset and len(sampleset) == 0:
        print("\nSampler returned an empty sampleset. No solution found. Try increasing num_reads or adjusting annealing parameters.")
    else:
        print("\nSampling failed or did not produce results.")
    

    '''
    # Create a new DataFrame with the data you want to append
    new_row_df = pd.DataFrame([{
        "obj_value":calculated_operational_cost,
        "feasibility": is_feasible,
        "elapsed_time": elapsed_time,
        "num_iterations": iterations,
    }])

    # Use pd.concat to append the new row(s) to the DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    # if csv file exists, delete it and replace with new one
    csv_file = "dwave_small_results.csv"
    
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    '''

if __name__ == "__main__":
    #df = pd.DataFrame(columns=["obj_value", "feasibility", "elapsed_time", "num_iterations"])

    for _ in range(1):
        main(iterations=500_000) 
    
    for _ in range(0):
        main(iterations=5_000_000)
    
    for _ in range(0):
        main(iterations=25_000_000)


