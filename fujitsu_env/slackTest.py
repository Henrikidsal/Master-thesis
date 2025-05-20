import pyomo.environ as pyo

model = pyo.ConcreteModel()

model.Lambda = pyo.Param(initialize=10000, mutable=True)  
model.RHS_cut1 = pyo.Param(initialize=235.007, mutable=True)  
#model.RHS_cut2 = pyo.Param(initialize=185.1, mutable=True)  
#model.RHS_cut3 = pyo.Param(initialize=364.0099, mutable=True)

model.beta = pyo.Var(domain=pyo.Integers)
model.slack_cut1 = pyo.Var(domain=pyo.NonNegativeReals)
#model.slack_cut2 = pyo.Var(domain=pyo.NonNegativeReals)  
#model.slack_cut3 = pyo.Var(domain=pyo.NonNegativeReals) 

def objective_rule(model):
    
    term = 0
    term+= model.Lambda * (model.beta - model.slack_cut1 - model.RHS_cut1)**2
    #term+= model.Lambda * (model.beta - model.slack_cut2 - model.RHS_cut2)**2
    #term+= model.Lambda * (model.beta - model.slack_cut3 - model.RHS_cut3)**2
    term+= model.beta
    
    
    return term
model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


solver = pyo.SolverFactory('gurobi')
results = solver.solve(model, tee=False)
print(f"Objective Function Value: {pyo.value(model.objective)}")
print(f"Value of beta: {pyo.value(model.beta)}")
print(f"Value of slack_cut1: {pyo.value(model.slack_cut1)}")   
#print(f"Value of slack_cut2: {pyo.value(model.slack_cut2)}")
#print(f"Value of slack_cut3: {pyo.value(model.slack_cut3)}")      
        

    