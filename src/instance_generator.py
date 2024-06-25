# std lib dependencies
import sys
import os
import random
from datetime import date

# third party dependencies
import numpy as np
from logger import logger
from cplex import Cplex
import cplex
from pyscipopt import Model, quicksum

# project dependencies
from common_utils import (
    cplex_problem_from_numpy_inequality_form,
    cplex_problem_to_numpy_inequality_form,
)


def completepathwithintermediatefolder(filepath, problemtype, midstring, i):
    intermediate_folder = f"{problemtype}_{midstring}"
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    if not os.path.exists(os.path.join(filepath, intermediate_folder)):
        os.mkdir(os.path.join(filepath, intermediate_folder))
    if not os.path.exists(
        os.path.join(filepath, intermediate_folder, "sample_" + str(i))
    ):
        os.mkdir(os.path.join(filepath, intermediate_folder, "sample_" + str(i)))
    return os.path.join(filepath, intermediate_folder, "sample_" + str(i))


def check_ILP_is_solvable(A, b, c, vtype="I") -> bool:
    "vtype: I for integer, B for binary"
    A = np.array(A)
    b = np.array(b)
    c = np.array(c)
    # Create a SCIP model
    model = Model("ILP_Check")

    # Define the variables
    n_vars = A.shape[1]
    x = []
    for i in range(n_vars):
        x.append(model.addVar(vtype=vtype, name=f"x_{i}"))

    # Define the constraints
    for i in range(A.shape[0]):
        logger.info(f"Add constraint {A[i]} * x <= {b[i]}")
        model.addCons(quicksum(A[i, j] * x[j] for j in range(n_vars)) <= b[i])

    # Define the objective function (if needed)
    if c is not None:
        model.setObjective(quicksum(c[i] * x[i] for i in range(n_vars)), "minimize")

    # display the problem
    logger.info(model)

    # Solve the ILP problem
    model.optimize()

    # Check the status of the optimization
    status = model.getStatus()

    # Check if the problem has an optimal solution and it's integral
    if status == "optimal":
        bestSol = {"x_" + str(i): model.getVal(x[i]) for i in range(n_vars)}
        # get bestSol as a numpy array:
        bestSolarray = np.array([bestSol[f"x_{i}"] for i in range(n_vars)])
        # check that A * bestSol <= b
        b_diff = A @ bestSolarray - b
        all_satistfy = True
        for element in b_diff:
            if element > 0:
                all_satistfy = False
        if all_satistfy:
            objval = model.getObjVal()
            logger.info(f"SCIP Optimal solution: {bestSol}")
            logger.info(f"SCIP Optimal value: {objval}")
            return {
                "best_sol": bestSol,
                "obj_val": objval,
                "status": status,
            }
    return {
        "best_sol": None,
        "obj_val": None,
        "status": "unfeasible",
    }


def generate_random_binpacking_instance(n: int, m: int):
    assert n <= m, "n must be less than or equal to m"
    # pick item sizes at random between 1 and 10
    item_sizes = [random.randint(1, 10) for _ in range(n)]
    # pick bin capacities at random between 1 and 10
    bin_capacities = [random.randint(1, 10) for _ in range(m)]
    # make sure that the sum of the item sizes is less than the sum of the bin capacities
    iteration_counter = 0
    while sum(item_sizes) > sum(bin_capacities) and iteration_counter < 1000:
        bin_capacities = [random.randint(1, 10) for _ in range(m)]
        iteration_counter += 1
    assert (
        iteration_counter < 1000
    ), "Too many iterations, could not find a feasible solution"
    # 1 weights for all selected bins yi
    c = [0 for _ in range(n * m)] + [1 for _ in range(m)]
    A = []  # [x0, ..., xn*m, y0, ..., ym]
    b = []
    # each item must be packed in exactly one bin
    for i in range(n):
        # we want constraints in <= form so we add two in positive and negative form for the equality constraint
        A.append([0 for _ in range(n * m + m)])
        A[-1][i * m : (i + 1) * m] = [1 for _ in range(m)]
        b.append(1)
        A.append([0 for _ in range(n * m + m)])
        A[-1][i * m : (i + 1) * m] = [-1 for _ in range(m)]
        b.append(-1)
    # each bin must not exceed its capacity
    for j in range(m):
        A.append([0 for _ in range(n * m + m)])
        A[-1][j] = -bin_capacities[j]
        b.append(0)
    return np.array(A), np.array(b), np.array(c)


def generate_random_binpacking_instance_v2(n: int, m: int):
    # sample coefficients, ranges as in Tang et al.
    rng = np.random.default_rng()
    A = rng.integers(low=5, high=30, size=(m, n), endpoint=True)
    b = rng.integers(low=10 * n, high=20 * n, size=m, endpoint=True)
    c = rng.integers(low=1, high=10, size=n, endpoint=True)
    for i in range(n):
        integrality_constraint = [0 for _ in range(n)]
        integrality_constraint[i] = 1
        A = np.vstack((A, integrality_constraint))
        b = np.append(b, 1)
    A = -A
    b = -b
    return A, b, c


def generate_random_packing_instance(n: int, m: int):
    A = np.random.randint(0, 5 + 1, size=(m, n))
    b = np.random.randint(9 * n, 10 * n + 1, size=m)
    c = np.random.randint(1, 10 + 1, size=n)
    A = -A
    b = -b
    return A, b, c


def generate_random_maxcut_instance(N: int, E: int):
    V = list(range(N))
    possible_edges = [(i, j) for i in V for j in V if i < j]
    assert E <= len(possible_edges), "E must be less than or equal to N choose 2"
    edges = random.sample(possible_edges, E)
    w = [0 for i in range(N)] + [random.randint(1, 10) for _ in range(E)]
    A = []  # (2*E + N) X (N+E)
    b = []
    c = w
    for i in range(E):
        edge = edges[i]
        # add y_u_v - x_u - x_v <= 0
        A.append([0 for _ in range(N + E)])
        A[-1][edge[0]] = -1
        A[-1][edge[1]] = -1
        A[-1][N + i] = 1
        b.append(0)
        # add y_u_v + x_u + x_v <= 2
        A.append([0 for _ in range(N + E)])
        A[-1][edge[0]] = 1
        A[-1][edge[1]] = 1
        A[-1][N + i] = 1
        b.append(2)
    for i in range(N):
        # add x_u <= 1
        A.append([0 for _ in range(N + E)])
        A[-1][i] = 1
        b.append(1)

    for i in range(E):
        # add y_u_v <= 1
        A.append([0 for _ in range(N + E)])
        A[-1][N + i] = 1
        b.append(1)
    return np.array(A), np.array(b), -np.array(c)


def generate_random_set_cover_instance(n: int, m: int):
    # n: number of sets
    # m: number of elements

    # generate random sets by iterating over the elements between 1 and m and adding them to some sets, each element must be added to at least one set
    # make it so that no element is included in more than 20% of the sets
    # make it so that no set is empty
    sets = []
    # create m empty sets
    for i in range(m):
        sets.append([])
    # add each element to a random set
    for i in range(n):
        for j in range(m):
            if random.random() < 0.2:
                sets[j].append(i + 1)
    # make sure no set is empty
    for j in range(m):
        if len(sets[j]) == 0:
            sets[j].append(random.randint(1, n))
    # make sure all elements are included in at least one set
    for i in range(n):
        if not any(i + 1 in sets[j] for j in range(m)):
            sets[random.randint(0, m - 1)].append(i + 1)

    # uniform cost for all sets
    c = [1 for _ in range(n)]

    # each element has to be covered by at least one set
    A = []
    for j in range(m):
        A.append([-1 if j + 1 in sets[i] else 0 for i in range(n)])
    b = [-1 for _ in range(m)]

    return np.array(A), np.array(b), np.array(c)


def generate_production_planning_instance(T):
    # Constants and parameters
    M = 100
    s_0 = 0
    s_T_star = 20

    # Generate revenue parameters
    p_prime = [random.randint(1, 10) for _ in range(1, T + 1)]
    h_prime = [random.randint(1, 10) for _ in range(T + 1)]
    q = [random.randint(1, 10) for _ in range(T + 1)]
    d = [random.randint(1, 10) for _ in range(T + 1)]

    # Create a CPLEX problem instance
    problem = Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Decision variables
    x = [f"x_{i}" for i in range(1, T + 1)]
    s = [f"s_{i}" for i in range(T + 1)]
    y = [f"y_{i}" for i in range(T + 1)]

    # Objective function
    problem.variables.add(obj=p_prime, names=x)
    problem.variables.add(obj=h_prime, names=s)
    problem.variables.add(obj=q, names=y)

    # Constraints
    # s[i] = s[i-1] + x[i] - d[i] for all 1 <= i <= T
    for i in range(1, T + 1):
        problem.linear_constraints.add(
            lin_expr=[[[f"s_{i-1}", f"x_{i}", f"s_{i}"], [1, 1, -1]]],
            senses=["E"],
            rhs=[d[i]],
        )

    # x[i] <= M * y[i] for all 1 <= i <= T
    for i in range(1, T + 1):
        problem.linear_constraints.add(
            lin_expr=[[[f"x_{i}", f"y_{i}"], [1, -M]]], senses=["L"], rhs=[0]
        )

    # s_0 = 0 and s_T = s_T_star
    problem.linear_constraints.add(lin_expr=[[[f"s_0"], [1]]], senses=["E"], rhs=[s_0])

    problem.linear_constraints.add(
        lin_expr=[[[f"s_{T}"], [1]]], senses=["E"], rhs=[s_T_star]
    )

    return cplex_problem_to_numpy_inequality_form(problem)


def generate_random_instances(
    n: int = None,
    m: int = None,
    output_file_prefix: str = ".",
    n_samples=1,
    model_type="packing",
    T_param_production_planning: int = None,
    N_param_max_cut: int = None,
    E_param_max_cut: int = None,
):
    assert model_type in [
        "packing",
        "binpacking",
        "maxcut",
        "production_planning",
        "set_cover",
    ]
    assert (model_type != "maxcut") or (
        N_param_max_cut is not None and E_param_max_cut is not None
    ), "N_param_max_cut and E_param_max_cut must be specified for maxcut"
    assert (model_type != "production_planning") or (
        T_param_production_planning is not None
    ), "T_param_production_planning must be specified for production_planning"
    assert model_type not in ["packing", "binpacking"] or (
        n is not None and m is not None
    ), "n and m must be specified for packing and binpacking"
    original_otuput_file_prefix = output_file_prefix

    for i in range(n_samples):
        solvable = False
        iterations_counter = 0
        while not solvable:
            if model_type == "packing":
                midstring = f"{n}_{m}"
                A, b, c = generate_random_packing_instance(n, m)
            elif model_type == "binpacking":
                midstring = f"{n}_{m}"
                A, b, c = generate_random_binpacking_instance_v2(n, m)
            elif model_type == "maxcut":
                midstring = f"{N_param_max_cut}_{E_param_max_cut}"
                A, b, c = generate_random_maxcut_instance(
                    N_param_max_cut, E_param_max_cut
                )
            elif model_type == "production_planning":
                midstring = f"{T_param_production_planning}"
                A, b, c = generate_production_planning_instance(
                    T_param_production_planning
                )
            elif model_type == "set_cover":
                midstring = f"{n}_{m}"
                A, b, c = generate_random_set_cover_instance(n, m)
            logger.info(f"\n {A}")
            logger.info(f"\n {b}")
            logger.info(f"\n {c}")
            logger.info(f"Checking instance {i}...")
            res = check_ILP_is_solvable(A, b, c)
            if res["status"] == "optimal":
                solvable = True
                logger.info(
                    f"Instance {i} is solvable, obtained in {iterations_counter+1} iterations"
                )
            iterations_counter += 1
            assert iterations_counter < 1000, "Too many iterations"
        output_file_prefix = completepathwithintermediatefolder(
            original_otuput_file_prefix, model_type, midstring, i
        )

        # save optimal solution and its value to metadata txt
        with open(output_file_prefix + "/solution_data.txt", "w") as f:
            f.write(f"Optimal solution: {res['best_sol']}\n")
            f.write(f"Optimal value: {res['obj_val']}\n")
            f.write(f"Status: {res['status']}\n")
            f.write(f"Date: {date.today()}\n")
            f.write(f"Solver: SCIP\n")

        # Save A, b, and c as NumPy .npy objects
        np.save(output_file_prefix + "/A.npy", A)
        np.save(output_file_prefix + "/b.npy", b)
        np.save(output_file_prefix + "/c.npy", c)

        # Write the instance to an MPS file
        with open(f"{output_file_prefix}/instance.mps", "w") as f:
            # Write the NAME record
            f.write(f"NAME: {model_type}\n")
            f.write("ROWS\n")

            # Write constraint rows
            for i in range(A.shape[0]):
                f.write(f" L  C{i+1}\n")

            # Write objective row
            f.write(" N  OBJ\n")

            f.write("COLUMNS\n")
            # Write variable columns
            for j in range(A.shape[1]):
                f.write(f"    x{j+1}  OBJ  {c[j]}\n")
                for i in range(A.shape[0]):
                    f.write(f"    x{j+1}  C{i+1}  {A[i, j]}\n")

            f.write("RHS\n")
            # Write RHS coefficients
            for i in range(A.shape[0]):
                f.write(f"    RHS1  C{i+1}  {b[i]}\n")

            f.write("BOUNDS\n")
            # Write variable bounds (non-negative)
            for j in range(A.shape[1]):
                f.write(f" LO BND x{j+1} 0\n")

            f.write("ENDATA\n")


def isinteger(x):
    return abs(x - round(x)) < 1e-5


def save_to_mps(A, b, c, filename):
    pr = cplex_problem_from_numpy_inequality_form(A, b, c)
    cplex_problem_to_mps(pr, filename)


def cplex_problem_to_mps(problem, filename):
    # Write the problem to an MPS file
    if not filename.endswith(".mps"):
        filename += ".mps"
    problem.write(filename)
