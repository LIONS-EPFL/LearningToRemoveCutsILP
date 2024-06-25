# std lib dependencies
import math
import pdb
import os
from typing import Dict, List

# third party dependencies
from logger import logger
import numpy as np
import time
from cplex import Cplex
import cplex as cpx
import cplex.exceptions

# project dependencies
from agents import BaseCutSelectAgent
from common_dtypes import (
    GomoryCut,
    CutSelResult,
    list_cuts_to_npy,
    StorageTrajectoryDatapoint,
    StorageILP,
    StorageLP,
    StorageCuts,
    StorageExpert,
    CutpoolQualityLog,
    CutpoolQualityLogList,
)
from common_utils import (
    cplex_problem_from_numpy_equality_form,
    scip_blackbox_checker,
    solve_LP,
)

VERBOSE = False
DEBUG = False


class BaseCutSelEnv:
    """Base class for an optimizer environment, defines the methods that need to be implemented by a subclass"""

    def __init__(
        self,
        instance_file_path: str,
        optimizer_seed: int = 0,
        environment_seed: int = 0,
        optimizer_time_limit_sc: int = 1e10,
    ):
        self.instance_file_path = instance_file_path
        self.problem_name = instance_file_path.split("/")[-1]
        self.instances = os.listdir(instance_file_path)
        self.optimizer_seed = optimizer_seed
        self.optimizer_time_limit_sc = optimizer_time_limit_sc
        self.set_seed(seed=environment_seed)

    def set_seed(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random.RandomState(self.seed)

    def set_random_seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def step(self, cutsel_agent: BaseCutSelectAgent) -> Dict[str, any]:
        """Given a cutselector policy the step method runs the IP solver and returns the statistics"""
        raise NotImplementedError

    def reset(self):
        """Resets the environment"""
        raise NotImplementedError


class CutSelEnv(BaseCutSelEnv):
    """Environment for the cutselector policy which starting from a ILP instance uses the cutting plane method to solve linear relaxations while adding Gomory Cuts iteratively"""

    def __init__(
        self,
        instance_file_path: str,
        optimizer_seed: int = 0,
        environment_seed: int = 0,
        optimizer_time_limit_sc: int = 1e10,
        inference_mode: str = "cutremove",
        fixed_ncuts_limit: int = 30,
        save_trajectories_path: str = None,
        iterations_limit: int = 30,
        collect_and_save_cpl: bool = False,
        save_cpls_path: str = None,
    ):
        super().__init__(
            instance_file_path,
            optimizer_seed,
            environment_seed,
            optimizer_time_limit_sc,
        )
        assert inference_mode in [
            "cutremove",
            "cutselect",
        ], f"Inference mode '{inference_mode}' not supported"
        self.inference_mode = inference_mode
        self.instance_choice = None
        self.mps_filepath = None
        self.sanity_check_metadata = None
        self.A = None
        self.b = None
        self.c = None
        self.x = None
        self.n_vars = None
        self.n_constraints = None
        self.initial_number_of_constraints = None
        self.objvalue = None
        self.A_prime = None
        self.c_prime = None
        self.x_prime = None
        self.g = []
        self.igc_seq = []
        self.LP_lb_list = []
        self.ncuts_limit = None
        self.fixed_ncuts_limit = fixed_ncuts_limit
        self.iterations_limit = iterations_limit
        self.save_trajectories_path = save_trajectories_path
        self.cpl = CutpoolQualityLogList(values=[])
        self.save_cpls_path = save_cpls_path
        self.collect_and_save_cpl = collect_and_save_cpl
        self.save_removed_and_kept_cuts_mode = False

    def select_random_instance(self):
        list_instances = os.listdir(self.instance_file_path)
        list_instances = [li for li in list_instances]
        return self.rng.choice(list_instances)

    def load_instance(self):
        # List all the files in the instance folder and pick one
        if VERBOSE:
            logger.info(f"List of instances: {list_instances}")
        if VERBOSE:
            logger.info(f"List of instances: {list_instances}")
        if self.instance_choice is None:
            self.instance_choice = self.select_random_instance()
        self.mps_filepath = os.path.join(
            self.instance_file_path, self.instance_choice, "instance.mps"
        )
        self.A = np.load(
            os.path.join(self.instance_file_path, self.instance_choice, "A.npy")
        )
        self.b = np.load(
            os.path.join(self.instance_file_path, self.instance_choice, "b.npy")
        )
        self.c = np.load(
            os.path.join(self.instance_file_path, self.instance_choice, "c.npy")
        )
        self.n_vars = len(self.c)
        self.n_constraints = len(self.b)
        self.initial_number_of_constraints = len(self.b)
        self.load_sanity_check_metadata(
            os.path.join(
                self.instance_file_path, self.instance_choice, "solution_data.txt"
            )
        )
        if self.fixed_ncuts_limit is not None:
            self.ncuts_limit = self.fixed_ncuts_limit
        else:
            raise NotImplementedError
        self.g = []
        self.igc_seq = []
        self.LP_lb_list = []
        self.cpl = CutpoolQualityLogList(values=[])
        self.instance_execution_time = 0

    def load_sanity_check_metadata(self, path):
        # Initialize an empty metadata dictionary
        metadata = {
            "optimal_value": None,
            "x": {},
            "status": None,
            "date": None,
            "solver": None,
        }

        # Read the content of the file
        with open(path, "r") as file:
            lines = file.readlines()

        # Parse the lines and populate the metadata dictionary
        for line in lines:
            if line.startswith("Optimal solution:"):
                # Extract the dictionary part and parse it into a Python dictionary
                solution_str = line.replace("Optimal solution: ", "").strip()
                metadata["x"] = eval(solution_str)

            elif line.startswith("Optimal value:"):
                # Extract the optimal value as a float
                value_str = line.replace("Optimal value: ", "").strip()
                metadata["optimal_value"] = float(value_str)

            elif line.startswith("Status:"):
                # Extract the status as a string
                status_str = line.replace("Status: ", "").strip()
                metadata["status"] = status_str

            elif line.startswith("Date:"):
                # Extract the date as a string
                date_str = line.replace("Date: ", "").strip()
                metadata["date"] = date_str

            elif line.startswith("Solver:"):
                # Extract the solver as a string
                solver_str = line.replace("Solver: ", "").strip()
                metadata["solver"] = solver_str

        # ensure that x is an array of integers
        metadata["x"] = np.array([el for el in metadata["x"].values()])
        metadata["x"] = self.ensure_integer_array(metadata["x"])

        # Store the metadata in the class attribute
        self.sanity_check_metadata = metadata

    def solve_linear_relaxation(self):
        problem, self.A_prime, self.c_prime = cplex_problem_from_numpy_equality_form(
            self.A, self.b, self.c
        )
        if VERBOSE:
            logger.info(self.A_prime)
            logger.info(self.c_prime)
        problem.parameters.lpmethod.set(problem.parameters.lpmethod.values.primal)
        try:
            problem.solve()
        except cplex.exceptions.CplexSolverError:
            logger.error("LP relaxation did not converge")
            return {
                "is_optimal": False,
                "optimal_status": problem.solution.status.optimal,
                "solution": None,
                "objective_value": None,
                "tableau": None,
                "rhs": None,
            }
        tab = []
        for tab_row in problem.solution.advanced.binvarow():
            tab.append(tab_row)
        is_optimal = problem.solution.get_status() == problem.solution.status.optimal
        if not is_optimal and VERBOSE:
            logger.error("LP relaxation did not converge")
        solution = problem.solution.get_values()
        # update the x params with the obtained solution in the LP relaxation
        self.x = np.array(solution[: self.n_vars])
        self.x_prime = np.array(solution)
        return {
            "is_optimal": is_optimal,
            "optimal_status": problem.solution.status.optimal,
            "solution": solution,
            "objective_value": problem.solution.get_objective_value(),
            "tableau": tab,
            "rhs": problem.linear_constraints.get_rhs(),
        }

    def cut_sanity_check(self, cut, rhs, solution, tabrow, lp_solution):
        cut = np.array(cut)
        lhs = np.dot(cut, solution)
        if VERBOSE:
            logger.info(f"\n For LP solution {lp_solution}")
        bv_indices = set(
            [idx for idx in range(len(lp_solution)) if lp_solution[idx] != 0]
        )
        if VERBOSE:
            logger.info(f"\n Basic variable indices are {bv_indices}")
            logger.info(f"\n For tableau row {tabrow}")
        one_indices = set([idx for idx in range(len(tabrow)) if tabrow[idx] == 1])
        if VERBOSE:
            logger.info(f"\n 1 indices are {one_indices}")
            logger.info(
                f"\n Cut corresponding to var {bv_indices.intersection(one_indices)}"
            )
            logger.info(f"\n Obtained cut is {cut}")
            logger.info(f"\n For solution {solution}")
            logger.info(
                f"\n Lhs of the cut replacing the solution values is {lhs} \n Rhs: {rhs}"
            )
        try:
            assert lhs <= rhs
        except AssertionError:
            import pdb

            pdb.set_trace()
        return True

    def get_all_gomory_cuts_fractional_form(self, solution, tableau) -> List[GomoryCut]:
        """Generates Gomory cuts from the tableau of the LP relaxation solution using the fractional form of the cuts so that it is easier to project to the original problem space"""
        tableau = np.array(tableau)
        solution = np.array(solution)
        rhs = np.matmul(tableau, solution)

        # Generate the fractional form of the cuts
        cut_lhs_coefficients = np.floor(tableau) - tableau
        cut_rhs_coefficients = np.floor(rhs) - rhs

        # Project the cuts into the original variable space
        r = cut_lhs_coefficients[:, self.n_vars :]
        e = cut_lhs_coefficients[:, : self.n_vars]
        d = cut_rhs_coefficients
        original_space_cut_lhs_coefficients = e - np.dot(r, self.A)
        original_space_cut_rhs_coefficients = d - np.dot(r, self.b)

        # Ensure the coefficients are integers to avoid numerical issues
        final_cuts_lhs = self.ensure_integer_array(original_space_cut_lhs_coefficients)
        final_cuts_rhs = self.ensure_integer_array(original_space_cut_rhs_coefficients)

        # Perform a sanity check on the cuts and remove non valid cuts
        gomory_cuts = []
        for i in range(len(tableau)):
            # if the cut had the original rhs as an integer skip the cut as we produced a 0vec. cut
            if not self.is_integer(rhs[i]):
                scip_solution = self.sanity_check_metadata["x"]
                scip_solution_with_slack = np.concatenate(
                    [
                        scip_solution,
                        np.zeros(len(cut_lhs_coefficients[i]) - len(scip_solution)),
                    ]
                )
                retvalue = self.cut_sanity_check(
                    final_cuts_lhs[i],
                    final_cuts_rhs[i],
                    scip_solution,
                    tableau[i],
                    solution,
                )
                # get the index of the basic variable corresponding to the cut
                basic_var_idx = -1
                for j in range(len(tableau[i])):
                    if tableau[i][j] == 1 and solution[j] != 0:
                        basic_var_idx = j
                        break
                assert basic_var_idx != -1, "Basic variable index not found"

                if retvalue:
                    gomory_cuts.append(
                        GomoryCut(
                            coefficients=final_cuts_lhs[i],
                            rhs=final_cuts_rhs[i],
                            tableau_idx=i,
                            nonbasic_var_idx=basic_var_idx,
                            features=None,  # features are computed a posteriori
                        )
                    )
                    if gomory_cuts[-1].tableau_idx is None:
                        print("tableau idx is None")
                        import pdb

                        pdb.set_trace()
        if self.collect_and_save_cpl:
            self.collect_cpl(cuts=gomory_cuts, x=self.x, c=self.c, A=self.A, b=self.b)
        return gomory_cuts

    def collect_cpl(
        self, cuts: List[GomoryCut], x: np.array, c: np.array, A: np.array, b: np.array
    ):
        """Given a list of cuts, computes the bound improvement for each cut and saves it in a file"""
        current_cpl = CutpoolQualityLog(previous_LP_value=np.dot(c, x), new_values=[])
        for cut in cuts:
            A_cut = np.concatenate([A, cut.coefficients.reshape(1, -1)], axis=0)
            b_cut = np.concatenate([b, [cut.rhs]], axis=0)
            res = solve_LP(A=A_cut, b=b_cut, c=c)
            current_cpl.new_values.append(res["objective_value"])
        self.cpl.values.append(current_cpl)

    def remove_LP_lb(self):
        """Removes -c^T x <= -np.ceil(x_optLR) as a constraint to the problem. This constraint should be in the last row of A"""
        assert np.equal(self.A[-1, :], -self.c).all(), "Last row of A is not -c^T"
        self.A = self.A[:-1, :]
        self.b = self.b[:-1]
        self.n_constraints -= 1

    def add_LP_lb(self, LP_solution_value):
        """Adds c^T x >= np.ceil(x_optLR) as a constraint to the problem
        equivalent to -c^T x <= -np.ceil(x_optLR)
        """
        if VERBOSE:
            logger.info(f"A shape {self.A.shape}")
            logger.info(f"c shape {self.c.shape}")
        self.A = np.concatenate([self.A, -self.c.reshape(1, -1)], axis=0)
        self.Lpsolutionvalue = LP_solution_value
        LP_lb = (
            -np.ceil(LP_solution_value)
            if not self.is_integer(LP_solution_value)
            else -np.rint(LP_solution_value)
        )
        self.b = np.concatenate([self.b, [LP_lb]], axis=0)
        self.n_constraints += 1
        self.LP_lb_list.append(
            {
                "LP_lb": LP_lb,
                "LP_solution_value": LP_solution_value,
            }
        )

    @staticmethod
    def is_integer(x, tolerance=1e-6):
        return abs(x - round(x)) < tolerance

    @staticmethod
    def ensure_integer_array(x: np.array, assertiontolerance=1e-6):
        if VERBOSE:
            logger.info(f"Rounding {x}")
        assert np.all(
            np.abs(x - np.rint(x)) < assertiontolerance * np.ones(x.shape) for el in x
        ), "Array is not integer"
        return [np.rint(el) for el in x]

    def add_cuts(self, cuts: List[GomoryCut]):
        """Adds multiple gomory cuts to the problem"""
        if len(cuts) == 0:
            return
        if VERBOSE:
            logger.info(f"Adding cuts: {cuts}")
        cut_coefficients = np.array([cut.coefficients for cut in cuts])
        cut_rhs = np.array([cut.rhs for cut in cuts])
        if VERBOSE:
            logger.info(
                f"Cut coefficients: {cut_coefficients} shape {cut_coefficients.shape}"
            )
            logger.info(f"Cut rhs: {cut_rhs}")
            logger.info(f"A shape: {self.A.shape}")
        self.A = np.concatenate([self.A, cut_coefficients], axis=0)
        self.b = np.concatenate([self.b, cut_rhs], axis=0)
        self.n_constraints += len(cuts)

    def remove_cuts(self, cuts: List[GomoryCut]):
        """Removes multiple gomory cuts from the problem"""
        if len(cuts) == 0:
            return
        reserved_m = self.initial_number_of_constraints
        A_non_reserved = self.A[reserved_m:, :]
        b_non_reserved = self.b[reserved_m:]
        full_constraints = np.concatenate(
            [A_non_reserved, b_non_reserved.reshape(-1, 1)], axis=1
        )
        cut_coefficients = np.array([cut.coefficients for cut in cuts])
        cut_rhs = np.array([cut.rhs for cut in cuts])
        cuts_constraints = np.concatenate(
            [cut_coefficients, cut_rhs.reshape(-1, 1)], axis=1
        )
        mask = np.logical_not(
            np.any(
                np.all(full_constraints[:, None, :] == cuts_constraints, axis=-1),
                axis=1,
            )
        )
        A_non_reserved = full_constraints[mask, :-1]
        b_non_reserved = full_constraints[mask, -1]
        self.A = np.concatenate([self.A[:reserved_m, :], A_non_reserved], axis=0)
        self.b = np.concatenate([self.b[:reserved_m], b_non_reserved], axis=0)
        self.n_constraints -= len(cuts)

    @staticmethod
    def round_integer_array_tolerance(array, tolerance=1e-6):
        """Rounds an array of floats to integers with a given tolerance"""
        return np.array(
            [round(el) if abs(el - round(el)) < tolerance else el for el in array]
        )

    def check_convergence(self):
        """Checks if the x solution for the A,b,c LP relaxation solve is valid for the Integer case"""
        corrected_x = self.round_integer_array_tolerance(self.x)
        if VERBOSE:
            logger.info(f"Checking convergence for solution {self.x}")
            logger.info(f"Constraints A: {self.A}")
            logger.info(f"Constraints b: {self.b}")
        n_vars = len(self.x)
        is_valid_solution = True

        for i in range(n_vars):
            # Check if each variable in the solution is integer
            if not self.is_integer(self.x[i]):
                if VERBOSE:
                    logger.info(f"Variable {i} is not integer")
                is_valid_solution = False
                break

        # If the solution is integer, check if it satisfies the ILP constraints
        if is_valid_solution:
            for j in range(len(self.b)):
                constraint_lhs = np.dot(self.A[j, :], self.x)
                if np.abs(constraint_lhs - self.b[j]) > 1e6:
                    if VERBOSE:
                        logger.info(
                            f"Constraint {j} is not satisfied: {constraint_lhs} > {self.b[j]}"
                        )
                        logger.info(f"corresponding constraint row: {self.A[j, :]}")
                        logger.info(f"corresponding row indice {j}")
                    is_valid_solution = False
                    break

        return is_valid_solution

    def remove_duplicate_cuts(self):
        """Removes duplicate cuts from the problem"""
        reserved_m = self.initial_number_of_constraints
        A_non_reserved = self.A[reserved_m:, :]
        b_non_reserved = self.b[reserved_m:]
        full_constraints = np.concatenate(
            [A_non_reserved, b_non_reserved.reshape(-1, 1)], axis=1
        )
        unique_constraints = np.unique(full_constraints, axis=0)
        self.A = np.concatenate(
            [self.A[:reserved_m, :], unique_constraints[:, :-1]], axis=0
        )
        self.b = np.concatenate(
            [self.b[:reserved_m], unique_constraints[:, -1]], axis=0
        )
        self.n_constraints = self.A.shape[0]

    def step(self, cutsel_agent: BaseCutSelectAgent) -> Dict[str, any]:
        """Loads an instance and solves it with the Cutting Plane method"""
        starting_time = time.time()
        self.load_instance()
        converged = False
        iteration_counter = 0
        optimizer_timeout = False
        added_cuts_counter = 0
        log_tab_info = False
        lp_convergence_error = False
        objvalue = None
        is_optimal_scip = False
        nocuts_error = False
        LP_solutions_value = []
        starting_time = time.time()
        while (
            not converged
            and not optimizer_timeout
            and not nocuts_error
            and added_cuts_counter < self.ncuts_limit
        ):
            # Solve LP relaxation and update the parameters
            ret_sol = self.solve_linear_relaxation()
            if not ret_sol["is_optimal"]:
                lp_convergence_error = True
                break
            objvalue = ret_sol["objective_value"]
            LP_solutions_value.append(objvalue)
            if log_tab_info:
                if VERBOSE:
                    logger.info(f"LP relaxation solution: {self.x}")
                    logger.info("Tableau")
                np_tableau = np.array(ret_sol["tableau"])
                if VERBOSE:
                    logger.info(f"Tableau np")
                    for i in range(len(np_tableau)):
                        logger.info(f"tab row:{np_tableau[i]}\n")

            converged = self.check_convergence()
            if not converged:
                # Generate cuts from the LP tableau
                tableau = np.array(ret_sol["tableau"])
                solution = np.array(ret_sol["solution"])
                cuts = self.get_all_gomory_cuts_fractional_form(
                    solution,
                    tableau,
                )
                cuts = cutsel_agent.remove_duplicate_cuts(cuts)
                if len(cuts) == 0 and VERBOSE:
                    logger.warning("No cuts generated")
                if len(cuts) == 0 or any(cut == [] for cut in cuts):
                    nocuts_error = True
                    break

                # Update the greedy checkpoint after LP relaxation solve
                # Note that this update has no effect on previous lines cut generation as it is post LP solve so we do it a posteriori
                if self.inference_mode == "cutremove":
                    if iteration_counter > 0:
                        self.remove_LP_lb()
                    self.add_LP_lb(objvalue)

                # CUT INFERENCE
                try:
                    if self.inference_mode == "cutremove":
                        # Remove previous greedy checkpoint and we will re-add it because we always keep it in the last row
                        self.remove_LP_lb()

                        # Add ALL cuts to the model
                        self.add_cuts(cuts)
                        added_cuts_counter += len(cuts)

                        # Remove duplicate cuts
                        self.remove_duplicate_cuts()

                        # Add the greedy checkpoint cut back
                        self.add_LP_lb(objvalue)

                        # Re-solve the LP relaxation with all the added cuts
                        ret_sol = self.solve_linear_relaxation()
                        if not ret_sol["is_optimal"]:
                            lp_convergence_error = True
                            break
                        objvalue = ret_sol["objective_value"]
                        LP_solutions_value.append(objvalue)

                        # Check for convergence
                        converged = self.check_convergence()
                        if converged:
                            self.x = self.round_integer_array_tolerance(self.x)
                            break

                        # Remove the greedy checkpoint cut, we will update it after dealing with the cuts so it is at the last row
                        self.remove_LP_lb()

                        # Remove the cuts from the constraints
                        assert (
                            cutsel_agent.mode == "reselect"
                        ), f"Agent should be in reselect mode for cutremoval inference but is {cutsel_agent.mode}"
                        n_cuts_added_previous_iterations = added_cuts_counter - len(
                            cuts
                        )
                        assert (
                            n_cuts_added_previous_iterations
                            + self.initial_number_of_constraints
                            + len(cuts)
                            == self.A.shape[0]
                        ), f"Mistmaching parameters for n_cuts_added_previous_iterations {n_cuts_added_previous_iterations} + self.initial_number_of_constraints {self.initial_number_of_constraints} + len(cuts) {len(cuts)} != self.A.shape[0] {self.A.shape[0]}"
                        cuts_to_add, cuts_to_remove = cutsel_agent.cutselect(
                            reserved_m=self.initial_number_of_constraints,
                            cuts=[],
                            n_cuts_added_previous_iterations=n_cuts_added_previous_iterations,  # count the cuts added in previous iterations, the greedy checkpoint is out of the cutpool
                            A=self.A,
                            b=self.b,
                            c=self.c,
                            x=self.x,
                            SimplexTableau=tableau,
                            SimplexSolution=solution,
                        )
                        if self.save_removed_and_kept_cuts_mode:
                            self.save_removed_and_kept_cuts(
                                iteration_counter, cuts, cuts_to_remove
                            )
                        assert (
                            len(cuts_to_add) == 0
                        ), "No cuts should be added in cutremove inference mode as we have added all the cuts previously"
                        previous_A_shape = self.A.shape
                        self.remove_cuts(cuts_to_remove)
                        try:
                            assert (
                                previous_A_shape[0] - len(cuts_to_remove)
                                == self.A.shape[0]
                            ), f"Previous A shape does not match current A shape after removing cuts {previous_A_shape[0]} - {len(cuts_to_remove)} != {self.A.shape[0]}"
                            assert (
                                self.A.shape[0]
                                == 1
                                + self.initial_number_of_constraints
                                + iteration_counter
                            )
                        except:
                            import pdb

                            pdb.set_trace()
                        added_cuts_counter -= len(cuts_to_remove)
                        # Update the greedy checkpoint with the previous objective value of solving ALL the cuts
                        self.add_LP_lb(objvalue)
                    elif self.inference_mode == "cutselect":
                        n_cuts_added_previous_iterations = added_cuts_counter
                        # Add cuts to the model
                        cuts_to_add, cuts_to_remove = cutsel_agent.cutselect(
                            reserved_m=self.initial_number_of_constraints,
                            cuts=cuts,
                            n_cuts_added_previous_iterations=n_cuts_added_previous_iterations,
                            A=self.A,
                            b=self.b,
                            c=self.c,
                            x=self.x,
                            SimplexTableau=tableau,
                            SimplexSolution=solution,
                        )
                        assert (
                            len(cuts_to_remove) == 0
                        ), "No cuts should be removed in cutselect inference mode"
                        if (
                            self.save_trajectories_path is not None
                            and not self.save_removed_and_kept_cuts_mode
                        ):
                            self.save_trajectory(iteration_counter, cuts, objvalue)
                        self.add_cuts(cuts_to_add)
                        added_cuts_counter += len(cuts_to_add)

                except Exception as e:
                    nocuts_error = True
                    import traceback

                    traceback.print_exc()
                    print(e)
                    import pdb

                    pdb.set_trace()
                    break
                if VERBOSE:
                    logger.info(f"Adding cuts: {cuts_to_add}")

                # compute integrality gap
                g_t = np.around(
                    self.sanity_check_metadata["optimal_value"]
                    - ret_sol["objective_value"],
                    decimals=10,
                )
                if self.is_integer(g_t):
                    g_t = np.rint(g_t)
                assert g_t >= 0
                self.g.append(g_t)
                if self.g[0] != 0:
                    igc_t = (self.g[0] - self.g[-1]) / self.g[0]
                else:
                    igc_t = 1
                self.igc_seq.append(igc_t)

            else:
                self.x = self.round_integer_array_tolerance(self.x)
            optimizer_timeout = (
                time.time() - starting_time > self.optimizer_time_limit_sc
            )
            iteration_counter += 1
        if converged:
            if VERBOSE:
                logger.info(
                    f"Cutting Plane method converged with:\n*Objective value: {objvalue}\n*Number of cuts: {added_cuts_counter}\n*Solution: {self.x}\n*Iteration counter: {iteration_counter}"
                )
                logger.info(
                    f"Optimal value from Cutting Plane: \n value {objvalue}\n solution{self.x}"
                )
                logger.info(
                    f"Optimal value from SCIP: \n value {self.sanity_check_metadata['optimal_value']}\n solution{self.sanity_check_metadata['x']}"
                )
            is_optimal_scip = (
                np.abs(self.sanity_check_metadata["optimal_value"] - objvalue) < 1e-6
            )
            optimal_value_difference = (
                objvalue - self.sanity_check_metadata["optimal_value"]
            )
        elif optimizer_timeout:
            if VERBOSE:
                logger.warning(
                    f"Cutting Plance method did not converge under timeout {self.optimizer_time_limit_sc}"
                )
        elif lp_convergence_error:
            if VERBOSE:
                logger.warning(
                    f"Cutting Plance method did not converge due to LP relaxation error"
                )
        if len(self.g) > 0:
            if self.g[0] == 0:
                igc = 1
            else:
                igc = (self.g[0] - self.g[-1]) / self.g[0]
        else:
            igc = 1
        self.igc_seq.append(igc)
        if len(self.g) == 0:
            self.g = [0]

        if self.collect_and_save_cpl:
            self.save_cpl(policy_name=cutsel_agent.policy.name)

        end_time = time.time()
        elapsed_time = end_time - starting_time
        result = CutSelResult(
            instance=self.instance_choice,
            problem_name=self.problem_name,
            policy=cutsel_agent.policy.name,
            converged=converged,
            lp_convergence_error=lp_convergence_error,
            optimizer_timeout=optimizer_timeout,
            objvalue=objvalue,
            ncuts=added_cuts_counter,
            x=self.x,
            iteration_counter=iteration_counter,
            sanity_check=is_optimal_scip,
            igc=igc,
            nocuts_error=nocuts_error,
            cuts_limit_exceeded=added_cuts_counter >= self.ncuts_limit,
            gap_array=self.g,
            igc_seq=self.igc_seq,
            LP_solutions_value=LP_solutions_value,
            LP_lb_list=self.LP_lb_list,
            elapsed_time=elapsed_time,
        )
        return result

    @staticmethod
    def igc(g0: float, g: float):
        """Computes the integrality gap given a sequence of values"""
        if g0 == 0:
            return 1
        return (g0 - g) / g0

    def save_removed_and_kept_cuts(
        self, iteration: int, cuts: List[GomoryCut], cuts_to_remove: List[GomoryCut]
    ):
        if isinstance(iteration, int):
            iteration = str(iteration)
        problem_trajectory_path = os.path.join(
            self.save_trajectories_path, self.instance_choice
        )
        if not os.path.exists(problem_trajectory_path):
            os.makedirs(problem_trajectory_path)
        current_iteration_path = os.path.join(problem_trajectory_path, iteration)
        if not os.path.exists(current_iteration_path):
            os.makedirs(current_iteration_path)
        removed_cuts_path = os.path.join(current_iteration_path, "removed_cuts")
        if not os.path.exists(removed_cuts_path):
            os.makedirs(removed_cuts_path)
        generated_cuts_path = os.path.join(current_iteration_path, "generated_cuts")
        if not os.path.exists(generated_cuts_path):
            os.makedirs(generated_cuts_path)
        for i, cut in enumerate(cuts):
            final_path = os.path.join(generated_cuts_path, f"cut_{i}.pkl")
            cut.to_pkl(final_path)
        for i, cut in enumerate(cuts_to_remove):
            final_path = os.path.join(removed_cuts_path, f"cut_{i}.pkl")
            cut.to_pkl(final_path)
        A_path = os.path.join(current_iteration_path, "A.npy")
        np.save(A_path, self.A)
        x_path = os.path.join(current_iteration_path, "x.npy")
        np.save(x_path, self.x)
        c_path = os.path.join(current_iteration_path, "c.npy")
        np.save(c_path, self.c)
        b_path = os.path.join(current_iteration_path, "b.npy")
        np.save(b_path, self.b)

    def save_cpl(self, policy_name: str):
        full_path = os.path.join(
            self.save_cpls_path, policy_name, self.instance_choice, "cpl_list.pkl"
        )
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
        self.cpl.to_pkl(full_path)

    def save_trajectory(self, iteration: int, cuts: List[GomoryCut], objvalue: float):
        if isinstance(iteration, int):
            iteration = str(iteration)
        problem_trajectory_path = os.path.join(
            self.save_trajectories_path, self.instance_choice
        )
        if not os.path.exists(problem_trajectory_path):
            os.makedirs(problem_trajectory_path)
        current_iteration_path = os.path.join(problem_trajectory_path, iteration)
        if not os.path.exists(current_iteration_path):
            os.makedirs(current_iteration_path)
        scores = np.array([cut.score for cut in cuts])
        trajectory_datapoint = StorageTrajectoryDatapoint.from_ILP_LP_cuts_expert(
            A=self.A,
            b=self.b,
            c=self.c,
            x_star_LP=self.x,
            x_star_value_LP=objvalue,
            cuts=cuts,
            expert=scores,
        )
        final_path = os.path.join(current_iteration_path, "trajectory_datapoint.pkl")
        trajectory_datapoint.save_pkl(final_path)

    def reset(self):
        self.A = None
        self.b = None
        self.c = None
        self.x = None
        self.n_vars = None
        self.n_constraints = None
        self.objvalue = None
        self.A_prime = None
        self.c_prime = None
        self.x_prime = None
        self.g = []
        self.igc_seq = []
        self.ncuts_limit = None
        self.LP_lb_list = []
        self.cpl = CutpoolQualityLogList(values=[])
