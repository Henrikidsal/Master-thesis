import pyomo.environ as pyo
import time

# 1. Data
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
# u_i0: Generator i status at t=0 (1 if on, 0 if off)
u_initial_data = {
    1: 0,  # Generator 1 was OFF
    2: 0,  # Generator 2 was OFF
    3: 1   # Generator 3 was ON
}

# 2. Pyomo Model
model = pyo.ConcreteModel(name="Binary_UCP")

# 3. Sets
model.GENERATORS = pyo.Set(initialize=generators_list)
model.TIME_PERIODS = pyo.Set(initialize=T_periods_list)

# 4. Parameters
model.P_max = pyo.Param(model.GENERATORS, initialize=P_max_data)
model.C_fixed = pyo.Param(model.GENERATORS, initialize=C_fixed_data)
model.C_startup = pyo.Param(model.GENERATORS, initialize=C_startup_data)
model.C_shutdown = pyo.Param(model.GENERATORS, initialize=C_shutdown_data)
model.C_variable = pyo.Param(model.GENERATORS, initialize=C_variable_data)
model.D = pyo.Param(model.TIME_PERIODS, initialize=D_data)
model.u_initial = pyo.Param(model.GENERATORS, initialize=u_initial_data)

# 5. Variables
# u_it – Generator i is on at time t
model.u = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)
# z_it^ON – generator i is turning on at time t
model.z_ON = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)
# z_it^OFF – generator i is turning off at time t
model.z_OFF = pyo.Var(model.GENERATORS, model.TIME_PERIODS, domain=pyo.Binary)

# 6. Objective Function
def objective_rule(m):
    cost = 0
    for t in m.TIME_PERIODS:
        for i in m.GENERATORS:
            cost += m.z_ON[i,t] * m.C_startup[i]
            cost += m.z_OFF[i,t] * m.C_shutdown[i]
            cost += m.C_variable[i] * m.P_max[i] * m.u[i,t]
            cost += m.C_fixed[i] * m.u[i,t]
    return cost

model.total_cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# 7. Constraints

# Logic 1: u_it - u_i(t-1) - z_it^ON + z_it^OFF = 0
model.logic1_constraints = pyo.ConstraintList()
for i in model.GENERATORS:
    for t in model.TIME_PERIODS:
        if t == model.TIME_PERIODS.first(): # Special handling for t=1 using u_initial
            u_previous = model.u_initial[i]
        else:
            u_previous = model.u[i, t-1]
        
        model.logic1_constraints.add(
            model.u[i,t] - u_previous - model.z_ON[i,t] + model.z_OFF[i,t] == 0
        )

# Logic 2: z_it^ON + z_it^OFF <= 1
model.logic2_constraints = pyo.ConstraintList()
for i in model.GENERATORS:
    for t in model.TIME_PERIODS:
        model.logic2_constraints.add(
            model.z_ON[i,t] + model.z_OFF[i,t] <= 1
        )

# Demand: sum(P_i^max * u_it) >= D_t
model.demand_constraints = pyo.ConstraintList()
for t in model.TIME_PERIODS:
    production_at_t = sum(model.P_max[i] * model.u[i,t] for i in model.GENERATORS)
    model.demand_constraints.add(
        production_at_t >= model.D[t]
    )


solver = pyo.SolverFactory('gurobi')
#start timer
start_time = time.time()
results = solver.solve(model, tee=False) # tee=True will show solver output
#end timer
end_time = time.time()
total_time = end_time - start_time
print(f"\nSolver finished in {total_time:.2f} seconds")

if (results.solver.status == pyo.SolverStatus.ok) and \
   (results.solver.termination_condition == pyo.TerminationCondition.optimal):
    print("\nOptimal Solution Found")
    print(f"Total Cost: ${model.total_cost():.2f}")

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
    print("Problem is Infeasible")
else:
    print("Solver Status: ", results.solver.status)
    print("Termination Condition: ", results.solver.termination_condition)

# To see the LP file:
# model.write("ucp_binary.lp", io_options={'symbolic_solver_labels': True})
# model.pprint()