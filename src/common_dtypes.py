# std lib dependencies
import os
from dataclasses import dataclass, field, asdict
import ast
import pickle as pkl
from typing import List, Dict

# third party dependencies
import pandas as pd
import numpy as np


@dataclass
class CutpoolQualityLog:
    previous_LP_value: float
    new_values: List[float]


@dataclass
class CutpoolQualityLogList:
    values: List[CutpoolQualityLog]

    def to_pkl(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def from_pkl(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)


@dataclass
class RowFeaturesHuang:
    """Row features as defined in Huang et al. 2022"""

    # Cut Stats
    cut_coefs_mean: float
    cut_coefs_min: float
    cut_coefs_max: float
    cut_coefs_std: float
    # Objective Stats
    obj_coefs_mean: float
    obj_coefs_min: float
    obj_coefs_max: float
    obj_coefs_std: float
    # Relation Metrics between cut and objective
    parallelism: float  # {the parallelism between the objective (c) and the cut (a)}[def] := (c^t a) / (norm(c) * norm(a))
    efficacy: float  # {euclidean distance of the cut hyperplane to the current LP solution}[def] := norm(c - a)
    support: float  # {proportion of non-zero coefs of the cut}[def] := |supp(a)| / |a|
    integral_support: float  # {proportion of non-zero coefs wrt. to integer variables of the cut}[def] := |supp(a) \cap Z| / |supp(a)|
    normalized_violation: float  # {max(0, (a^t*x_LP - b)/norm(b))}[def] := max(0, (a^t*x_LP - b)/norm(b)) on the original space variables of lpsol

    @classmethod
    def from_coefs_and_lpsol(
        cls, alpha: np.array, beta: float, obj_coefs: np.array, lpsol: np.array
    ):
        if isinstance(alpha, list):
            alpha = np.array(alpha)
        if isinstance(obj_coefs, list):
            obj_coefs = np.array(obj_coefs)
        if isinstance(lpsol, list):
            lpsol = np.array(lpsol)
        assert len(alpha.shape) == 1
        assert len(obj_coefs.shape) == 1
        assert len(lpsol.shape) == 1
        assert (
            alpha.shape[0] == lpsol.shape[0]
        ), f"alpha.shape[0] = {alpha.shape[0]}, lpsol.shape[0] = {lpsol.shape[0]}"
        assert (
            alpha.shape[0] == obj_coefs.shape[0]
        ), f"alpha.shape[0] = {alpha.shape[0]}, obj_coefs.shape[0] = {obj_coefs.shape[0]}"
        norm_violation = max(0, np.dot(alpha, lpsol[: alpha.size]) - beta) / np.abs(
            beta
        )
        return cls(
            # Cut Stats
            cut_coefs_mean=np.mean(alpha),
            cut_coefs_min=np.min(alpha),
            cut_coefs_max=np.max(alpha),
            cut_coefs_std=np.std(alpha),
            # Objective Stats
            obj_coefs_mean=np.mean(obj_coefs),
            obj_coefs_min=np.min(obj_coefs),
            obj_coefs_max=np.max(obj_coefs),
            obj_coefs_std=np.std(obj_coefs),
            # Relation Metrics between cut and objective
            parallelism=(np.dot(obj_coefs, alpha))
            / (
                np.linalg.norm(obj_coefs) * np.linalg.norm(alpha)
            ),  # (c^t a) / (norm(c) * norm(a))
            efficacy=np.linalg.norm(
                obj_coefs - alpha
            ),  # euclidean distance of the cut hyperplane to the current LP solution
            support=np.count_nonzero(alpha)
            / alpha.size,  # proportion of non-zero coefs of the cut
            integral_support=np.count_nonzero(
                alpha[~np.isclose(alpha, alpha.astype(int))]
            )
            / alpha.size,  # proportion of non-zero coefs wrt. to integer variables of the cut
            normalized_violation=norm_violation,  # max(0, (a^t*x_LP - b)/norm(b)) on the original space variables of lpsol
        )

    def to_numpy(self):
        return np.array(
            [
                # Cut Stats
                self.cut_coefs_mean,
                self.cut_coefs_min,
                self.cut_coefs_max,
                self.cut_coefs_std,
                # Objective Stats
                self.obj_coefs_mean,
                self.obj_coefs_min,
                self.obj_coefs_max,
                self.obj_coefs_std,
                # Relation Metrics between cut and objective
                self.parallelism,
                self.efficacy,
                self.support,
                self.integral_support,
                self.normalized_violation,
            ]
        )

    def to_dict(self):
        return {
            # Cut Stats
            "cut_coefs_mean": self.cut_coefs_mean,
            "cut_coefs_min": self.cut_coefs_min,
            "cut_coefs_max": self.cut_coefs_max,
            "cut_coefs_std": self.cut_coefs_std,
            # Objective Stats
            "obj_coefs_mean": self.obj_coefs_mean,
            "obj_coefs_min": self.obj_coefs_min,
            "obj_coefs_max": self.obj_coefs_max,
            "obj_coefs_std": self.obj_coefs_std,
            # Relation Metrics between cut and objective
            "parallelism": self.parallelism,
            "efficacy": self.efficacy,
            "support": self.support,
            "integral_support": self.integral_support,
            "normalized_violation": self.normalized_violation,
        }


@dataclass
class RowFeatures:
    features_huang: RowFeaturesHuang
    is_historic: int

    @classmethod
    def from_datapoint_and_cut_idx(cls, datapoint, cut_idx: int, is_historic: int = 0):
        assert is_historic in [0, 1], "is_historic must be 0 or 1"
        return cls(
            features_huang=RowFeaturesHuang.from_coefs_and_lpsol(
                alpha=datapoint.Cuts.cuts[cut_idx].coefficients,
                beta=datapoint.Cuts.cuts[cut_idx].rhs,
                obj_coefs=datapoint.ILP.c,
                lpsol=datapoint.LP.x_star_LP,
            ),
            is_historic=is_historic,
        )

    @classmethod
    def from_cut_solution_c(cls, cut, solution, c, is_historic: int = 0):
        assert is_historic in [0, 1], "is_historic must be 0 or 1"
        return cls(
            features_huang=RowFeaturesHuang.from_coefs_and_lpsol(
                alpha=cut.coefficients, beta=cut.rhs, obj_coefs=c, lpsol=solution
            ),
            is_historic=is_historic,
        )

    @classmethod
    def from_datapoint_and_constraint_idx(
        cls, datapoint, constraint_idx: int, is_historic: int = 0
    ):
        assert is_historic in [0, 1], "is_historic must be 0 or 1"
        return cls(
            features_huang=RowFeaturesHuang.from_coefs_and_lpsol(
                alpha=datapoint.ILP.A[constraint_idx],
                beta=datapoint.ILP.b[constraint_idx],
                obj_coefs=datapoint.ILP.c,
                lpsol=datapoint.LP.x_star_LP,
            ),
            is_historic=is_historic,
        )

    @classmethod
    def from_cut_solution_c(
        cls,
        cut_lhs: np.array,
        cut_rhs: float,
        solution: np.array,
        c: np.array,
        is_historic: int = 0,
    ):
        assert is_historic in [0, 1], "is_historic must be 0 or 1"
        return cls(
            features_huang=RowFeaturesHuang.from_coefs_and_lpsol(
                alpha=cut_lhs, beta=cut_rhs, obj_coefs=c, lpsol=solution
            ),
            is_historic=is_historic,
        )

    def to_numpy(self):
        assert self.is_historic in [0, 1], "is_historic must be 0 or 1"
        features_huang_np = self.features_huang.to_numpy()
        return np.array([*features_huang_np, self.is_historic])

    def to_dict(self):
        return {**self.features_huang.to_dict(), "is_historic": self.is_historic}


@dataclass
class GomoryCut:
    coefficients: List[float]
    rhs: float
    tableau_idx: int
    nonbasic_var_idx: int
    features: RowFeatures = None
    score: float = 0

    def to_numpy(self):
        return np.array([*self.coefficients, self.rhs])

    def to_pkl(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def from_pkl(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)

    def to_dict(self):
        # returns itself as dict and the row features as dict
        return {
            "coefficients": self.coefficients,
            "rhs": self.rhs,
            "tableau_idx": self.tableau_idx,
            "nonbasic_var_idx": self.nonbasic_var_idx,
            **self.features.to_dict(),
            "score": self.score,
        }


@dataclass
class GenericCut:
    lhs: np.array
    rhs: float
    features: RowFeatures = None
    score: float = 0

    def to_numpy(self):
        return np.array([*self.lhs, self.rhs])

    def to_pkl(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def from_pkl(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)


@dataclass
class StorageILP:
    A: np.array
    b: np.array
    c: np.array


@dataclass
class StorageLP:
    x_star_LP: np.array
    x_star_value_LP: float


@dataclass
class StorageCuts:
    cuts: List[GomoryCut]


@dataclass
class StorageExpert:
    label: List[float]


@dataclass
class StorageTrajectoryDatapoint:
    """Dataclass to store a datapoint of the storage trajectory"""

    ILP: StorageILP
    LP: StorageLP
    Cuts: StorageCuts
    Expert: StorageExpert

    def save_pkl(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load_pkl(cls, filename):
        with open(filename, "rb") as f:
            return pkl.load(f)

    @classmethod
    def from_ILP_LP_cuts_expert(
        cls,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x_star_LP: np.ndarray,
        x_star_value_LP: float,
        cuts: List[GomoryCut],
        expert: List[float],
    ):
        return cls(
            ILP=StorageILP(A=A, b=b, c=c),
            LP=StorageLP(x_star_LP=x_star_LP, x_star_value_LP=x_star_value_LP),
            Cuts=StorageCuts(cuts=cuts),
            Expert=StorageExpert(label=expert),
        )


def list_cuts_to_npy(cuts: List[GomoryCut], filename: str):
    instances_dict_list = [asdict(instance) for instance in cuts]
    np.save(filename, instances_dict_list)


def from_npy_ndarray_cut_features(filename: str) -> np.ndarray:
    cuts = from_npy_list_cuts(filename)
    np_cuts = np.array([cut.features.to_numpy() for cut in cuts])
    return np_cuts


@dataclass
class CutSelResult:
    """Class to store the results of a CutSel instance"""

    instance: str
    policy: str
    problem_name: str
    converged: bool
    optimizer_timeout: bool
    objvalue: float
    ncuts: int
    x: List[float]
    iteration_counter: int
    lp_convergence_error: float
    igc: float
    nocuts_error: bool
    sanity_check: bool
    cuts_limit_exceeded: bool
    gap_array: List[float]
    igc_seq: List[float]
    LP_solutions_value: List[float]
    LP_lb_list: List[float]
    save_date: str = None
    elapsed_time: float = None

    def to_pd(self):
        data_dict = asdict(self)
        df = pd.DataFrame([data_dict])
        return df

    @classmethod
    def from_pd(cls, df):
        instance = cls(
            instance=df["instance"],
            policy=df["policy"],
            problem_name=df["problem_name"],
            converged=df["converged"],
            optimizer_timeout=df["optimizer_timeout"],
            objvalue=df["objvalue"],
            ncuts=df["ncuts"],
            x=df["x"],
            iteration_counter=df["iteration_counter"],
            lp_convergence_error=df["lp_convergence_error"],
            igc=df["igc"],
            nocuts_error=df["nocuts_error"],
            sanity_check=df["sanity_check"],
            cuts_limit_exceeded=df["cuts_limit_exceeded"],
            gap_array=df["gap_array"],
            igc_seq=df["igc_seq"],
            LP_solutions_value=df["LP_solutions_value"],
            LP_lb_list=df["LP_lb_list"],
            save_date=df["save_date"],
            elapsed_time=df["elapsed_time"],
        )
        return instance


@dataclass
class ListCutSelResult:
    values: List[CutSelResult]

    def to_pd(self):
        df = pd.concat([x.to_pd() for x in self.values])
        return df

    @classmethod
    def from_pd(cls, df):
        instance = cls(values=[CutSelResult.from_pd(row) for idx, row in df.iterrows()])
        return instance

    @classmethod
    def from_csv(cls, filename):
        instance = cls(values=[])  # Create an instance of the class with an empty list
        df = pd.read_csv(
            filename,
            converters={
                "x": np.fromstring,
                "gap_array": ast.literal_eval,
                "igc_seq": np.fromstring,
                "LP_solutions_value": ast.literal_eval,
            },
        )
        instance.values = [CutSelResult.from_pd(row) for idx, row in df.iterrows()]
        return instance

    @classmethod
    def from_pkl(cls, filename):
        instance = cls(values=[])
        with open(filename, "rb") as f:
            instance.values = pkl.load(f)
        return instance

    def to_pkl(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self.values, f)

    def to_csv(self, filename, append_mode=True):
        df = self.to_pd()
        df["save_date"] = pd.to_datetime("today")
        if not os.path.exists(filename):
            df.to_csv(filename, mode="w", header=True, index=False)
        else:
            df.to_csv(
                filename,
                mode="a" if append_mode else "w",
                header=not append_mode,
                index=False,
            )
