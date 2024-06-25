# std lib dependencies
import sys
import os

# third party dependencies
import numpy as np
from cplex import Cplex
import cplex
from logger import logger

VERBOSE=False

def cplex_problem_from_numpy_inequality_form(A, b, c):
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    # Get the number of variables and constraints
    num_variables = A.shape[1]
    num_constraints = A.shape[0]

    # Create a Cplex object
    problem = Cplex()

    # Set the sense of the objective function to minimize
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Add variables
    lb = [0.0] * num_variables  # Lower bounds (default is 0.0)
    ub = [cplex.infinity] * num_variables  # Upper bounds (default is +inf)
    var_names = [f"x{i}" for i in range(num_variables)]
    problem.variables.add(obj=c.tolist(), lb=lb, ub=ub, names=var_names)

    # Add constraints
    rows = []
    senses = [
        "L"
    ] * num_constraints  # Assuming all constraints are "less than or equal to"
    rhs = b.tolist()
    row_names = [f"c{i}" for i in range(num_constraints)]

    for i in range(num_constraints):
        row = [list(range(num_variables)), A[i].tolist()]
        rows.append(row)

    problem.linear_constraints.add(
        lin_expr=rows, senses=senses, rhs=rhs, names=row_names
    )
    return problem


def cplex_problem_from_numpy_equality_form(A, b, c):
    """Adds positive slacks to the problem and turns it into an equality problem"""
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(b, np.ndarray):
        b = np.array(b)
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    # add slacks
    A_prime = np.concatenate((A, np.eye(A.shape[0])), axis=1)
    c_prime = np.concatenate((c, np.zeros(A.shape[0])), axis=0)

    # Get the number of variables and constraints
    num_variables = A_prime.shape[1]
    num_constraints = A_prime.shape[0]

    # Create a Cplex object
    problem = Cplex()

    # Set the sense of the objective function to minimize
    problem.objective.set_sense(problem.objective.sense.minimize)

    # A_primedd variables
    lb = [0.0] * num_variables  # Lower bounds (default is 0.0)
    ub = [cplex.infinity] * num_variables  # Upper bounds (default is +inf)
    var_names = [f"x{i}" for i in range(num_variables)]
    problem.variables.add(obj=c_prime.tolist(), lb=lb, ub=ub, names=var_names)

    # A_primedd constraints
    rows = []
    senses = ["E"] * num_constraints  # A_prime assuming all constraints are "equal to"
    rhs = b.tolist()
    row_names = [f"c_prime{i}" for i in range(num_constraints)]

    for i in range(num_constraints):
        row = [list(range(num_variables)), A_prime[i].tolist()]
        rows.append(row)

    problem.linear_constraints.add(
        lin_expr=rows, senses=senses, rhs=rhs, names=row_names
    )
    return problem, A_prime, c_prime


def cplex_problem_to_numpy_inequality_form(problem):
    num_variables = problem.variables.get_num()
    num_constraints = problem.linear_constraints.get_num()

    # Initialize A and b as zeros
    A = np.zeros((num_constraints, num_variables))
    b = np.zeros(num_constraints)

    c = np.array(problem.objective.get_linear())
    
    # Get the objective sense
    sense = problem.objective.get_sense()
    logger.info(f"Optimization sense is {sense}")

    # If the objective sense is minimization, negate the c vector
    if sense == -1: # maximization
        logger.info(f"Maximization problem detected, negating objective function")
        c = -c

    # Initialize a list to track the unique variable indices used in the constraints
    used_variable_indices = []

    for i in range(num_constraints):
        constraint = problem.linear_constraints.get_rows(i)

        # Initialize a dictionary to store the variable indices and coefficients for the current constraint
        constraint_variables = {}

        for idx in range(len(constraint.ind)):
            variable_idx = constraint.ind[idx]
            coefficient = constraint.val[idx]
            constraint_variables[variable_idx] = coefficient

            # Add the variable index to the list of used indices if not already present
            if variable_idx not in used_variable_indices:
                used_variable_indices.append(variable_idx)

        # Assign the coefficients to the A matrix using the unique variable indices
        for variable_idx in used_variable_indices:
            A[i, variable_idx] = constraint_variables.get(variable_idx, 0.0)

        # Handle the right-hand side of the constraint (rhs)
        rhs = problem.linear_constraints.get_rhs()[i]
        sense = problem.linear_constraints.get_senses()[i]

        if sense == 'L':
            b[i] = rhs
        elif sense == 'G':
            b[i] = -rhs
            A[i, :] = -A[i, :]
        elif sense == 'E':
            # an equality constraint generates two inequality constraints, add one at index i and the other one at the end
            b[i] = rhs
            A[i, :] = -A[i, :]
            A = np.vstack((A, A[i, :]))
            b = np.append(b, -rhs)
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")
    return A, b, c


def scip_blackbox_checker(A, b, c):
    """Recieves a INTEGER constrained optimization problem c^T*x s.t. Ax <= b and returns the optimal INTEGER solution using scip"""
    from pyscipopt import Model, quicksum

    model = Model("mip1")
    x = {}
    for j in range(len(c)):
        x[j] = model.addVar(vtype="I", name="x(%s)" % j)
    model.setObjective(quicksum(c[j] * x[j] for j in range(len(c))), "minimize")
    for i in range(len(b)):
        model.addCons(
            quicksum(A[i][j] * x[j] for j in range(len(c))) <= b[i], "c(%s)" % i
        )
    model.optimize()
    return {
        "optimal_value": model.getObjVal(),
        "x": [model.getVal(x[j]) for j in range(len(c))],
    }


def solve_LP(A, b, c):
    """Given the inequality form of an LP, solves it using cplex"""
    problem, A_prime, c_prime = cplex_problem_from_numpy_equality_form(
            A, b, c
        )
    if VERBOSE:
        logger.info(A_prime, c_prime)
    problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.primal)
    problem.solve()
    tab = []
    for tab_row in problem.solution.advanced.binvarow():
        tab.append(tab_row)
    is_optimal = problem.solution.get_status() == problem.solution.status.optimal
    if not is_optimal and VERBOSE:
        logger.error("LP did not converge")
    return {
        "is_optimal": is_optimal,
        "optimal_status": problem.solution.status.optimal,
        "solution": problem.solution.get_values(),
        "objective_value": problem.solution.get_objective_value(),
        "tableau": tab,
        "rhs": problem.linear_constraints.get_rhs(),
    }


def get_path_of_largest_epoch(checkpoint_dir):
    # List all files in the directory
    files = os.listdir(checkpoint_dir)

    # Filter out files that do not match the checkpoint pattern
    checkpoint_files = [file for file in files if file.startswith('checkpoint-epoch') and file.endswith('.pt')]

    if not checkpoint_files:
        return None  # No matching checkpoint files found

    # Extract the epoch number from each file
    epoch_numbers = [int(file.split('-')[-1].split('.')[0]) for file in checkpoint_files]

    # Find the index of the file with the maximum epoch
    max_epoch_index = epoch_numbers.index(max(epoch_numbers))

    # Select the path corresponding to the largest epoch
    path_of_largest_epoch = os.path.join(checkpoint_dir, checkpoint_files[max_epoch_index])

    return path_of_largest_epoch


def repr_integers_as_integers(x: np.array, epsilon: float = 1e-6):
    """"Converts all floats that are integers to integers"""
    return np.array([int(i) if np.abs(i-np.rint(i)) < epsilon else i for i in x])