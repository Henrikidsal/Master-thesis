import gurobipy as gp
from gurobipy import GRB
import numpy as np
from Small_problem_solver import Q as Qubo
from Small_problem_solver import c_term as constant_term

def solve_qubo(Q, c):

    n = Q.shape[0]
    
    # Create a new Gurobi model
    m = gp.Model("qubo")
    m.setParam('OutputFlag', 0)  # turn off solver output
    
    # Add binary variables x[i] for i in range(n)
    x = m.addVars(n, vtype=GRB.BINARY, name="x")
    
    # Build the quadratic objective expression: x^T Q x + c
    obj = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            # Only add nonzero coefficients for efficiency
            if Q[i, j] != 0:
                # Note: For off-diagonal terms, Q is assumed to be fully populated;
                # we add them all. Gurobi will combine terms appropriately.
                obj.add(Q[i, j] * x[i] * x[j])
    
    obj.add(c)  # add the constant term
    m.setObjective(obj, GRB.MINIMIZE)
    
    # Optimize the model
    m.optimize()
    
    # Retrieve the solution
    sol = np.array([x[i].X for i in range(n)])
    obj_val = m.objVal
    
    return sol, obj_val

if __name__ == "__main__":
    # 
    Q = Qubo
    
    c = constant_term  # constant term
    
    solution, obj_val = solve_qubo(Q, c)
    print("Optimal binary solution x:")
    print(solution)
    print("Optimal objective value:")
    print(obj_val)

