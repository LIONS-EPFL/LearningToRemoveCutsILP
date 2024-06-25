import os
from pyscipopt import Model
from pyscipopt import Model, quicksum, SCIP_RESULT, SCIP_PARAMSETTING
from pyscipopt.scip import Cutsel
import itertools
import pyscipopt
from typing import List, Tuple
import numpy as np
import torch
from common_dtypes import GenericCut, RowFeatures
import time
import pandas as pd
import pickle


class Scorer:
    def __init__(
        self, input_size: int, device: str, checkpoint_path: str = None
    ) -> None:
        self.input_size = input_size
        self.model = SimpleMLP(input_size=14, hidden_layers=3, hidden_size=512).to(
            device
        )
        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()

    def forward(self, torch_constraints):
        return self.model(torch_constraints)

    def preprocess(self, np_constraints, np_rhs, np_solution, np_obj):
        cuts = [
            GenericCut(lhs=np_constraints[i], rhs=np_rhs[i])
            for i in range(len(np_constraints))
        ]
        for cut in cuts:
            cut.features = RowFeatures.from_cut_solution_c(
                cut_lhs=cut.lhs, cut_rhs=cut.rhs, solution=np_solution, c=np_obj
            )
        cut_features_ndarray = np.array([cut.features.to_numpy() for cut in cuts])
        torch_cut_features = torch.tensor(cut_features_ndarray)
        return torch_cut_features

    def __call__(self, torch_constraints):
        return self.forward(torch_constraints)


class CustomCutSelector(Cutsel):
    def __init__(self, scorerclass, var_indices, mode):
        # scorer takes (nconst, nvars) tensor as input and returns a (nconst,) tensor
        super().__init__()
        self.rounds = 0
        self.scorerclass = scorerclass
        self.var_indices = var_indices
        self.nvars = len(var_indices)
        self.LP_bound_pairs = (
            []
        )  # list of tuples (LP value, improved bound) for each round
        self.mode = mode

    def setNvars(self):
        if self.nvars is None:
            self.nvars = self.model.getNVars()

    def setScorer(self):
        assert self.nvars is not None
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.scorer = self.scorerclass(input_size=self.nvars, device=device)

    def constrainttoNumpy(self, constraint) -> Tuple[np.array, float]:
        # assumes that variables are named as var_i in [0, nvars-1]
        if isinstance(constraint, pyscipopt.scip.ExprCons):
            return self.ExprConstoNumpy(constraint)
        elif isinstance(constraint, pyscipopt.scip.Constraint):
            return self.ConstoNumpy(constraint)
        else:
            raise ValueError(f"constraint type {type(constraint)} not supported")

    def ExprConstoNumpy(self, constraint) -> Tuple[np.array, float]:
        # Assumes that variables are named as var_i in [0, nvars-1]
        scip = self.model
        np_constraint = np.zeros(self.nvars)
        for varterm, coef in constraint.expr.terms.items():
            var_index = varterm.vartuple[0].getIndex()
            if var_index >= self.nvars:
                # extend the np constraint array
                np_constraint = np.pad(
                    np_constraint,
                    (0, var_index - self.nvars + 1),
                    "constant",
                    constant_values=(0),
                )
                self.nvars = var_index + 1
            np_constraint[var_index] = coef
        rhs = constraint._rhs
        return np_constraint, rhs

    def ConstoNumpy(self, constraint) -> Tuple[np.array, float]:
        # assumes that variables are named as var_i in [0, nvars-1]
        scip = self.model
        np_constraint = np.zeros(self.nvars)
        values_dict = scip.getValsLinear(constraint)
        return np_constraint, scip.getRhs(constraint)

    def solutiontoNumpy(self, solution) -> np.array:
        np_solution = np.zeros(self.nvars)
        solution_dict = solution.getValsDict()
        for varname, val in solution_dict.items():
            var_index = self.var_indices[varname]
            np_solution[var_index] = val
        return np_solution

    def objtoNumpy(self, objective) -> np.array:
        np_obj = np.zeros(self.nvars)
        for varterm, coef in objective.terms.items():
            var_index = varterm.vartuple[0].getIndex()
            if var_index >= self.nvars:
                # extend the np constraint array
                np_constraint = np.pad(
                    np_constraint,
                    (0, var_index - self.nvars + 1),
                    "constant",
                    constant_values=(0),
                )
                self.nvars = var_index + 1
            np_obj[var_index] = coef
        return np_obj

    def consttoTorch(self, constraint) -> torch.Tensor:
        np_constraint = self.constrainttoNumpy(constraint)
        return torch.tensor(np_constraint).float()

    def conslisttoNumpy(self, constraints) -> Tuple[np.ndarray, np.array]:
        # assumes that variables are named as var_i in [0, nvars-1]
        np_constraints = np.zeros((len(constraints), self.nvars))
        np_rhs = np.zeros(len(constraints))
        for i in range(len(constraints)):
            new_constraint, new_rhs = self.constrainttoNumpy(constraints[i])
            if new_constraint.shape[0] > np_constraints.shape[1]:
                np_constraints = np.pad(
                    np_constraints,
                    ((0, 0), (0, new_constraint.shape[0] - np_constraints.shape[1])),
                    "constant",
                    constant_values=(0),
                )
            np_constraints[i] = new_constraint
            np_rhs[i] = new_rhs
        return np_constraints, np_rhs

    def getCutsAsConstraints(self, cuts):
        scip = self.model
        constraints = []
        for i in range(len(cuts)):
            cut = cuts[i]
            lhs_nonzero_vars = cut.getCols()
            lhs_nonzero_coefs = cut.getVals()
            lin_expr = [
                (lhs_nonzero_vars[i].getVar(), lhs_nonzero_coefs[i])
                for i in range(len(lhs_nonzero_vars))
            ]
            rhs = cut.getRhs()
            constraints.append(
                pyscipopt.quicksum(coef * var for var, coef in lin_expr) <= rhs
            )
        return constraints

    def scoreConstraints(self, constraints, solution, objective) -> np.array:
        np_constraints, np_rhs = self.conslisttoNumpy(constraints)
        np_solution = self.solutiontoNumpy(solution)
        np_objective = self.objtoNumpy(objective)
        assert (
            np_constraints.shape[1] == self.nvars
        ), f"nvars {self.nvars} != np_constraints.shape[1] {np_constraints.shape[1]}"
        assert (
            np_rhs.shape[0] == np_constraints.shape[0]
        ), f"np_rhs.shape[0] {np_rhs.shape[0]} != np_constraints.shape[0] {np_constraints.shape[0]}"
        assert (
            np_solution.shape[0] == self.nvars
        ), f"nvars {self.nvars} != np_solution.shape[0] {np_solution.shape[0]}"
        assert (
            np_objective.shape[0] == self.nvars
        ), f"nvars {self.nvars} != np_objective.shape[0] {np_objective.shape[0]}"
        scorer_input = self.scorer.preprocess(
            np_constraints, np_rhs, np_solution, np_objective
        )
        scores = self.scorer(scorer_input)
        return scores.detach().numpy()

    def addtopkConstraints(self, constraints, scores, k, added_key="alg2"):
        scip = self.model
        k = min(k, len(constraints))
        for i in range(k):
            constraint = constraints[i]
            if not isinstance(constraint, pyscipopt.scip.ExprCons):
                rhs = scip.getRhs(constraint)
                expr_constraint = scip.getExprLinear(constraint)
                constraint = pyscipopt.scip.ExprCons(expr_constraint, rhs=rhs)
            scip.addCons(
                constraint,
                removable=True,
                local=True,
                name=f"{added_key}_{time.time()}",
            )

    def addObjectiveBoundConstraint(self, objective, bound, added_key: str = "alg2"):
        scip = self.model
        if np.isclose(bound, np.round(bound)):
            bound = np.round(bound)
        scip.addCons(
            -objective <= -np.ceil(bound),
            removable=False,
            local=True,
            name=f"{added_key}_{time.time()}",
        )

    def getBoundIfAddedAllCuts(self, cuts):
        """Create a new LP instance with the current LP + all the cuts and return the bound"""
        scip = self.model
        previous_value = scip.getLPObjVal()
        # start a dive mode
        scip.startDive()
        for cut in cuts:
            scip.addRowDive(cut)
        err, cutoff = scip.solveDiveLP()
        assert not err and not cutoff, f"Error {err} or cutoff {cutoff} in solveDiveLP"
        value = scip.getLPObjVal()
        scip.endDive()
        self.LP_bound_pairs.append((previous_value, value))
        # assert value <= previous_value, f"Value {value} > previous_value {previous_value}"
        return value

    def getLinearActiveRemovableAddedConstraints(
        self, added_key="alg2", and_remove: bool = False
    ):
        scip = self.model
        constraints = scip.getConss()
        active_constraints = []
        for constraint in constraints:
            if (
                constraint.isLinear()
                and constraint.isRemovable()
                and added_key in constraint.name
            ):
                active_constraints.append(constraint)
                if and_remove:
                    scip.delCons(constraint)
        return active_constraints

    def cutselselect(self, cuts, forcedcuts, root, maxnselectedcuts):
        mode = self.mode
        assert mode in ["base", "cutremove"]
        # format + add all cuts and add the greedy checkpoint
        scip = self.model
        solution = scip.getBestSol()
        objective = scip.getObjective()
        if mode == "cutremove":
            cutsbound = self.getBoundIfAddedAllCuts(cuts)
            self.addObjectiveBoundConstraint(objective, cutsbound)
        # get cuts and constraints and score them
        self.setScorer()
        cutConstraints = self.getCutsAsConstraints(cuts)
        if mode == "cutremove":
            constraints = self.getLinearActiveRemovableAddedConstraints(
                added_key="alg2", and_remove=True
            )
            all_constraints = constraints + cutConstraints
        elif mode == "base":
            all_constraints = cutConstraints
        scores = self.scoreConstraints(all_constraints, solution, objective)
        # re-add the best k scored constraints
        self.addtopkConstraints(all_constraints, scores, maxnselectedcuts)
        return {"cuts": cuts, "nselectedcuts": 0, "result": SCIP_RESULT.SUCCESS}


def test_cut_selector(
    instance_base_path: str, n_nodes: int, save_path: str, checkpoint_path: str = None
):
    """Builds and runs the instances in instance_path for a
    maximum of n_nodes with cut remove algorithm implementation
    and default cut selection. Then gets some stats."""
    MODES = ["base", "cutremove"]
    # create a dataframe with Gap, LP_val, Sol_val, Mode, Bound_pairs as columns
    df = pd.DataFrame(
        columns=[
            "Gap",
            "LP_val",
            "Sol_val",
            "Primal_Bound",
            "Dual_Bound",
            "Mode",
            "Instance",
            "Bound_pairs",
        ],
        index=[],
    )
    if not os.path.exists(save_path):
        df.to_csv(save_path)
    instances = [f for f in os.listdir(instance_base_path) if f.endswith(".lp")]
    processed_instances_modes = []
    # read the dataframe and get the pairs of instance and mode that appear
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        processed_instances_modes = list(zip(df["Instance"], df["Mode"]))

    SKIP_INSTANCESSEGFAULTNUMERICALERR = []
    if os.path.exists("skip_instances.pkl"):
        with open("skip_instances.pkl", "rb") as f:
            SKIP_INSTANCESSEGFAULTNUMERICALERR = pickle.load(f)
    else:
        with open("skip_instances.pkl", "wb") as f:
            pickle.dump(SKIP_INSTANCESSEGFAULTNUMERICALERR, f)
    for instance_path in instances:
        for mode in MODES:
            if (
                instance_path.replace(".lp", ""),
                mode,
            ) in processed_instances_modes:
                continue
            if instance_path in SKIP_INSTANCESSEGFAULTNUMERICALERR:
                print(f"Skipping instance {instance_path}")
                print(f"{SKIP_INSTANCESSEGFAULTNUMERICALERR}")
                continue
            scip = Model()
            # scip.setParam("separating/maxcuts", 0)
            scip.setParam("limits/nodes", n_nodes)
            # scip.setParam("separating/maxcutsroot", 0)
            # scip.setParam("separating/maxrounds", 0)
            # scip.setParam("separating/maxroundsroot", 0)
            scip.setObjective(scip.getObjective(), sense="minimize")
            scip.readProblem(os.path.join(instance_base_path, instance_path))
            var_indices = {}
            for v in scip.getVars():
                var_indices[v.name] = v.getIndex()
            cutsel = CustomCutSelector(
                scorerclass=Scorer, var_indices=var_indices, mode=mode
            )
            scip.includeCutsel(
                cutsel, "customCutSelector", "maximises efficacy", 5000000
            )
            try:
                scip.optimize()
            except:
                print(f"Error in instance {instance_path} with mode {mode}")
                scip.freeProb()
                # update the skip instances
                SKIP_INSTANCESSEGFAULTNUMERICALERR.append(instance_path)
                with open("skip_instances.pkl", "wb") as f:
                    pickle.dump(SKIP_INSTANCESSEGFAULTNUMERICALERR, f)
                continue
            solution = scip.getBestSol()
            bound_pairs = cutsel.LP_bound_pairs if mode == "cutremove" else None
            # for some stages of the problem this can't be called
            try:
                solution_value = scip.getSolVal(solution)
            except:
                solution_value = None
            stats = {
                "Gap": scip.getGap(),
                "LP_val": scip.getLPObjVal(),
                "Sol_val": solution_value,
                "Primal_Bound": scip.getPrimalbound(),
                "Dual_Bound": scip.getDualbound(),
                "Mode": mode,
                "Instance": instance_path.replace(".lp", ""),
                "Bound_pairs": bound_pairs,
            }
            df = pd.DataFrame([stats])
            df.to_csv(save_path, mode="a", header=False, index=False)
            # free the model
            scip.freeProb()


test_cut_selector(
    instance_base_path="./data/nn_verification_instances",
    n_nodes=100,
    save_path="scip_experiment.csv",
    checkpoint_path="./misc/scip_experiment_checkpoint.pth",
)
