import pyomo.environ as pyo

def minimize_function_pyomo(p_val, k_vals):

    # --- 1. Create a Pyomo Model ---
    model = pyo.ConcreteModel()

    # --- 2. Define Parameters and Sets ---
    # Create a set for the indices of s and k (1, 2, 3, ...)
    model.indices = pyo.Set(initialize=k_vals.keys())

    # p is a single, global parameter
    model.p = pyo.Param(initialize=p_val)
    # k is now an indexed parameter, initialized with the dictionary
    model.k = pyo.Param(model.indices, initialize=k_vals)

    # --- 3. Define Variables ---
    # b is still a single non-negative variable
    model.b = pyo.Var(domain=pyo.Integers, bounds=(0, None))
    # s is now an indexed variable over our set of indices
    model.s = pyo.Var(model.indices, domain=pyo.NonNegativeReals)

    # --- 4. Define the Objective Function ---
    # The rule now sums the penalty terms for each index i.
    def objective_rule(m):
        penalty_sum = sum(m.p * (m.b - m.s[i] - m.k[i])**2 for i in m.indices)
        return penalty_sum + m.b
    
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # --- 5. Solve the Model ---
    solver = pyo.SolverFactory('gurobi')
    results = solver.solve(model, tee=False)

    # --- 6. Print the Results ---
    if (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("Optimization was successful!")
        opt_b = pyo.value(model.b)
        # Store the optimal 's' values in a dictionary
        opt_s = {i: pyo.value(model.s[i]) for i in model.indices}
        
        print(f"Optimal value for b: {opt_b:.4f}")
        for i in sorted(opt_s.keys()):
            print(f"Optimal value for s_{i}: {opt_s[i]:.4f}")
            
        return opt_b, opt_s
    else:
        print("Optimization did not find an optimal solution.")
        print(f"Solver Status: {results.solver.status}")
        return None, None


# --- Main execution block ---
if __name__ == "__main__":
    # --- CHOOSE YOUR CONSTANTS HERE ---
    p_constant = 1000
    
    # Define the k constants as a dictionary. 
    # The keys (1, 2, 3) will be the indices.
    k_constants = {
        1: 50,
        2: 70.1,
        3: 30
    }

    print(f"Running optimization with p = {p_constant} and k = {k_constants}")
    print("-" * 30)

    # Run the optimization function
    minimize_function_pyomo(p_constant, k_constants)