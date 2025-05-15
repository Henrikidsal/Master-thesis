import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.Parameter1_1 = pyo.Param(initialize=100, mutable=True)  
model.Parameter1_2 = pyo.Param(initialize=50, mutable=True)

model.Parameter2 = pyo.Param(initialize=235.7, mutable=True)  

model.Parameter3 = pyo.Param(initialize=185.1, mutable=True)  

model.Parameter4 = pyo.Param(initialize=3164.91, mutable=True)

model.x = pyo.Var(domain=pyo.Integers)

def objective_rule(model):
    
    term = 0
    term+= - model.Parameter1_1 * (model.x - model.Parameter2) + model.Parameter1_2 * (model.x - model.Parameter2)**2
    term+= - model.Parameter1_1 * (model.x - model.Parameter3) + model.Parameter1_2 * (model.x - model.Parameter3)**2
    term+= - model.Parameter1_1 * (model.x - model.Parameter4) + model.Parameter1_2 * (model.x - model.Parameter4)**2
    term+= model.x
    
    
    return term
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


solver = pyo.SolverFactory('gurobi')

print("Solving the optimization problem...")
results = solver.solve(model, tee=False)
print("\n--- Optimal Solution Found ---")
print(f"Objective Function Value: {pyo.value(model.objective)}")
print(f"Value of x: {pyo.value(model.x)}")