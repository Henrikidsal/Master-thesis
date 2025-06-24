import pyomo.environ as pyo
import time
# 1. Data (Same as before)
# Time periods
T_periods_list = [1, 2, 3]

#demand
D_data          = {1: 160, 2: 500, 3: 400}       # MW demand
#D_data          = {1: 200, 2: 550, 3: 550}       # MW demand

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

# QUBO Parameters
lambda_logic1 = 10000.0
lambda_logic2 = 10000.0
lambda_demand = 10000.0

# Number of slack bits for demand penalty term
SLACK_BITS_K_COUNT = 8# k from 0 to 9, for 2^0 to 2^9


# 2. Pyomo Model
model = pyo.ConcreteModel(name="Binary_UCP_QUBO_Simple_Output")

# 3. Sets
model.GENERATORS = pyo.Set(initialize=generators_list)
model.TIME_PERIODS = pyo.Set(initialize=T_periods_list)
model.SLACK_K_RANGE = pyo.Set(initialize=range(SLACK_BITS_K_COUNT)) # k from 0 to K-1

# 4. Parameters
model.P_max = pyo.Param(model.GENERATORS, initialize=P_max_data)
model.C_fixed = pyo.Param(model.GENERATORS, initialize=C_fixed_data)
model.C_startup = pyo.Param(model.GENERATORS, initialize=C_startup_data)
model.C_shutdown = pyo.Param(model.GENERATORS, initialize=C_shutdown_data)
model.C_variable = pyo.Param(model.GENERATORS, initialize=C_variable_data)
model.D = pyo.Param(model.TIME_PERIODS, initialize=D_data)
model.u_initial = pyo.Param(model.GENERATORS, initialize=u_initial_data)

# 5. Variables
model.u = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)
model.z_ON = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)
model.z_OFF = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)
model.s_demand = pyo.Var(model.TIME_PERIODS, model.SLACK_K_RANGE, domain=pyo.Binary)


# 6. Objective Function (Original Cost + Penalties)
def objective_rule_qubo(m):
    # Original cost part
    original_cost = 0
    for t in m.TIME_PERIODS:
        for i in m.GENERATORS:
            original_cost += m.z_ON[i,t] * m.C_startup[i]
            original_cost += m.z_OFF[i,t] * m.C_shutdown[i]
            original_cost += m.C_variable[i] * m.P_max[i] * m.u[i,t]
            original_cost += m.C_fixed[i] * m.u[i,t]

    # Penalty for Logic 1: (u_it - u_i(t-1) - z_it^ON + z_it^OFF)^2
    penalty_logic1 = 0
    for i in m.GENERATORS:
        for t in m.TIME_PERIODS:
            if t == m.TIME_PERIODS.first():
                u_previous = m.u_initial[i]
            else:
                u_previous = m.u[i, t-1]
            
            logic1_term = m.u[i,t] - u_previous - m.z_ON[i,t] + m.z_OFF[i,t]
            penalty_logic1 += logic1_term**2

    # Penalty for Logic 2: (z_it^OFF * z_it^ON)
    penalty_logic2 = 0
    for i in m.GENERATORS:
        for t in m.TIME_PERIODS:
            penalty_logic2 += m.z_OFF[i,t] * m.z_ON[i,t]
            
    # Penalty for Demand: (sum(P_i^max * u_it) - sum(2^k * s_tk^demand) - D_t)^2
    penalty_demand = 0
    for t in m.TIME_PERIODS:
        production_at_t = sum(m.P_max[i] * m.u[i,t] for i in m.GENERATORS)
        s_tk_sum_at_t = sum((2**k) * m.s_demand[t,k] for k in m.SLACK_K_RANGE)
        demand_term = production_at_t - s_tk_sum_at_t - m.D[t]
        penalty_demand += demand_term**2
        
    total_objective = original_cost + \
                      lambda_logic1 * penalty_logic1 + \
                      lambda_logic2 * penalty_logic2 + \
                      lambda_demand * penalty_demand
    return total_objective

# This is the QUBO objective. The first script's output refers to "Total Cost"
# as only the operational part, so we will calculate that separately for printing.
model.objective_qubo_full = pyo.Objective(rule=objective_rule_qubo, sense=pyo.minimize)

# 7. Constraints are now part of the objective function as penalties

# 8. Solve
solver = pyo.SolverFactory('gurobi')
#start time
start_time = time.time()
results = solver.solve(model, tee=False) # tee=False as requested
#end time
end_time = time.time()
total_time = end_time - start_time
print(f"Solver finished in {total_time:.2f} seconds")

# 9. Results
if (results.solver.status == pyo.SolverStatus.ok) and \
   (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\nOptimal Solution Found")

    # Calculate the "original cost" component for printing, to match the first script's output
    calculated_operational_cost = 0
    for t_val in model.TIME_PERIODS:
        for i_val in model.GENERATORS:
            calculated_operational_cost += pyo.value(model.z_ON[i_val,t_val]) * model.C_startup[i_val]
            calculated_operational_cost += pyo.value(model.z_OFF[i_val,t_val]) * model.C_shutdown[i_val]
            calculated_operational_cost += model.C_variable[i_val] * model.P_max[i_val] * pyo.value(model.u[i_val,t_val])
            calculated_operational_cost += model.C_fixed[i_val] * pyo.value(model.u[i_val,t_val])
    
    print(f"Total Cost: ${calculated_operational_cost:.2f}") # This now prints the operational cost
    print("QUBO Objective Value: ",pyo.value(model.objective_qubo_full))

    print("\nGenerator Schedules (u_it):")
    print("Gen\\Time |  1  |  2  |  3  |")
    print("-----------------------------")
    for i in model.GENERATORS:
        print(f"   G{i}    |  {pyo.value(model.u[i,1]):.0f}  |  {pyo.value(model.u[i,2]):.0f}  |  {pyo.value(model.u[i,3]):.0f}  |")

    print("\nGenerator Start-ups (z_it^ON):")
    print("Gen\\Time |  1  |  2  |  3  |")
    print("-----------------------------")
    for i in model.GENERATORS:
        print(f"   G{i}    |  {pyo.value(model.z_ON[i,1]):.0f}  |  {pyo.value(model.z_ON[i,2]):.0f}  |  {pyo.value(model.z_ON[i,3]):.0f}  |")

    print("\nGenerator Shut-downs (z_it^OFF):")
    print("Gen\\Time |  1  |  2  |  3  |")
    print("-----------------------------")
    for i in model.GENERATORS:
        print(f"   G{i}    |  {pyo.value(model.z_OFF[i,1]):.0f}  |  {pyo.value(model.z_OFF[i,2]):.0f}  |  {pyo.value(model.z_OFF[i,3]):.0f}  |")

    print("\nProduction Plan (P_i^max * u_it):")
    print("Gen\\Time |   1   |   2   |   3   |")
    print("-----------------------------------")
    total_production = {t:0 for t in model.TIME_PERIODS}
    for i in model.GENERATORS:
        row = f"   G{i}    |"
        for t in model.TIME_PERIODS:
            prod = pyo.value(model.u[i,t]) * model.P_max[i]
            total_production[t] += prod
            row += f" {prod:5.0f} |"
        print(row)
    print("-----------------------------------")
    demand_row = "Demand   |"
    total_prod_row = "TotalPrd |"
    for t in model.TIME_PERIODS:
        demand_row += f" {model.D[t]:5.0f} |"
        total_prod_row += f" {total_production[t]:5.0f} |"
    print(demand_row)
    print(total_prod_row)


elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
    print("Problem is Infeasible") # Less likely for QUBO unless numerical issues
else:
    print("Solver Status: ", results.solver.status)
    print("Termination Condition: ", results.solver.termination_condition)

# To see the LP file (it will be a quadratic program):
# model.write("ucp_qubo_simple_output.lp", io_options={'symbolic_solver_labels': True})
# model.pprint()