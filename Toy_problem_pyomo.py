import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory

def build_subproblem(y_vals):
    
    model = pyo.ConcreteModel(name="SubProblem")
    
    # Sets
    model.lo = pyo.Set(initialize=[1, 2])
    model.n  = pyo.Set(initialize=[1, 2, 3])
    AL_data = [(1, 2), (1, 3), (2, 3)]
    model.AL = pyo.Set(initialize=AL_data, dimen=2)
    
    # Parameters
    model.d = pyo.Param(initialize=0.85)
    B_data = {(1,2): 2.5, (1,3): 3.5, (2,3): 3.0}
    model.B = pyo.Param(model.AL, initialize=B_data)
    emax_data = {(1,2): 0.3, (1,3): 0.7, (2,3): 0.7}
    model.emax = pyo.Param(model.AL, initialize=emax_data)
    Pmax_data = {1: 0.9, 2: 0.9}
    model.Pmax = pyo.Param(model.lo, initialize=Pmax_data)
    f_data = {1: 10, 2: 5}
    v_data = {1: 6, 2: 7}
    model.f = pyo.Param(model.lo, initialize=f_data)
    model.v = pyo.Param(model.lo, initialize=v_data)
    
    # Variables
    model.P = pyo.Var(model.lo, within=NonNegativeReals)
    model.ud = pyo.Var(within=NonNegativeReals)
    model.c = pyo.Var(model.lo, within=Reals)
    model.e = pyo.Var(model.AL, within=Reals)
    model.delta = pyo.Var(model.n, within=Reals, bounds=(-3.14, 3.14))
    model.cost = pyo.Var(within=Reals)
    model.y = pyo.Var(model.lo, within=Reals)
    
    # Instead of fixing y, introduce a mutable parameter and an equality constraint.
    model.y_fixed = pyo.Param(model.lo, initialize=y_vals, mutable=True)
    def yfix_rule(m, l):
        return m.y[l] == m.y_fixed[l]
    model.yfix = pyo.Constraint(model.lo, rule=yfix_rule)
    
    # Dual suffix
    model.dual = pyo.Suffix(direction=Suffix.IMPORT)
    
    # Constraints
    def balance1_rule(m):
        return m.P[1] - m.e[1,2] - m.e[1,3] == 0
    model.balance1 = pyo.Constraint(rule=balance1_rule)
    
    def balance2_rule(m):
        return m.P[2] + m.e[1,2] - m.e[2,3] == 0
    model.balance2 = pyo.Constraint(rule=balance2_rule)
    
    def balance3_rule(m):
        return -m.d + m.e[1,3] + m.e[2,3] + m.ud == 0
    model.balance3 = pyo.Constraint(rule=balance3_rule)
    
    def translim_rule(m, i, j):
        return m.e[i,j] <= m.emax[i,j]
    model.translim = pyo.Constraint(model.AL, rule=translim_rule)
    
    def edf_rule(m, i, j):
        return m.e[i,j] == m.B[i,j] * sin(m.delta[i] - m.delta[j])
    model.edf = pyo.Constraint(model.AL, rule=edf_rule)
    
    def copar_rule(m, l):
        return m.c[l] == m.f[l]*m.y[l] + m.v[l]*m.P[l]
    model.copar = pyo.Constraint(model.lo, rule=copar_rule)
    
    # Production capacity constraint; note that GAMS uses "=l=" which means ≤.
    def prodlim_rule(m, l):
        return m.P[l] <= m.y[l] * m.Pmax[l]
    model.prodlim = pyo.Constraint(model.lo, rule=prodlim_rule)
    
    def costdf_rule(m):
        return m.cost == sum(m.c[l]*m.P[l] for l in m.lo) + 10 * sum(10*m.f[l] for l in m.lo) * m.ud
    model.costdf = pyo.Constraint(rule=costdf_rule)
    
    model.obj = pyo.Objective(expr=model.cost, sense=pyo.minimize)
    
    return model

def build_master(iteration_data):
   
    master = pyo.ConcreteModel(name="MasterProblem")
    master.lo = pyo.Set(initialize=[1,2])
    master.yaux = pyo.Var(master.lo, within=Binary)
    master.alpha = pyo.Var(within=NonNegativeReals)
    master.Cuts = pyo.Set(initialize=range(len(iteration_data)))
    
    def benders_cut_rule(m, k):
        data = iteration_data[k]
        z_s = data['z_s']
        lambdas = data['lambda']
        y_s = data['y_s']
        return m.alpha >= z_s + sum(lambdas[l]*(m.yaux[l] - y_s[l]) for l in m.lo)
    master.benders_cut = pyo.Constraint(master.Cuts, rule=benders_cut_rule)
    master.obj = pyo.Objective(expr=master.alpha, sense=pyo.minimize)
    
    return master

def main():
    max_iter = 15
    epsilon = 1e-3
    iteration_data = []
    z_lo = 0.0
    z_up = float('inf')
    error = 1.0
    
    # In iteration 1, we force y = 1.
    y_current = {1: 1.0, 2: 1.0}
    
    mip_solver = SolverFactory('gurobi')
    nlp_solver = SolverFactory('ipopt')
    
    it = 1
    while it <= max_iter and error > epsilon:
        print(f"=========================Iteration {it}=========================")
        
        if it == 1:
            print("Iteration 1: Setting y = 1 for all facilities.")
        else:
            # For iterations > 1, solve the master first to update y.
            master_problem = build_master(iteration_data)
            result_master = mip_solver.solve(master_problem, tee=False)
            master_problem.solutions.load_from(result_master)
            alpha_val = pyo.value(master_problem.alpha)
            new_yaux = {l: pyo.value(master_problem.yaux[l]) for l in master_problem.lo}
            print(" Master problem objective (alpha):", alpha_val)
            for l in master_problem.lo:
                print(f"  Master solution: yaux({l}) = {new_yaux[l]:.6f}")
            z_lo = alpha_val
            y_current = new_yaux  # update y_current from master solution
            print(" Updated lower bound z_lo =", z_lo)
        
        # Now, solve the subproblem using the current y_current.
        subproblem = build_subproblem(y_current)
        result_sub = nlp_solver.solve(subproblem, tee=False)
        subproblem.solutions.load_from(result_sub)
        z_s = pyo.value(subproblem.cost)
        print(" Subproblem objective (cost):", z_s)
        print(" Subproblem solver status:", result_sub.solver.status, result_sub.solver.termination_condition)
        
        # Retrieve the multiplier on the y-fixing constraint (this corresponds to lambda in GAMS)
        lambdas = {}
        for l in subproblem.lo:
            if subproblem.yfix[l] in subproblem.dual:
                lambdas[l] = subproblem.dual[subproblem.yfix[l]]
            else:
                lambdas[l] = 0.0
        y_s = {l: pyo.value(subproblem.y[l]) for l in subproblem.lo}
        ud_val = pyo.value(subproblem.ud)
        for l in subproblem.lo:
            print(f"  Facility {l}: Production P = {pyo.value(subproblem.P[l]):.6f}, lambda = {lambdas[l]:.6f}")
        print("  Unsatisfied demand =", ud_val)
        
        z_up = z_s
        print(" Current upper bound z_up =", z_up)
        print(" Current lower bound z_lo =", z_lo)
        
        # Store iteration data for the master.
        iteration_data.append({'it': it, 'z_s': z_s, 'lambda': lambdas, 'y_s': y_s})
        
        # Update error.
        if abs(z_lo) > 1e-15:
            error = abs((z_up - z_lo) / z_lo)
        else:
            error = abs(z_up - z_lo)
        print(" Error =", error, "\n")
        
        it += 1
    
    print("===== Terminated =====")
    print(f"Final bounds: z_lo = {z_lo:.6f}, z_up = {z_up:.6f}, error = {error:.6f}")
    print("Number of iterations performed =", it - 1)

if __name__ == '__main__':
    main()

