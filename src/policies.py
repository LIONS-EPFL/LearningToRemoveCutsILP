# std lib
import os
import random
from logger import logger
from typing import List, Tuple, Optional, Callable, Dict
import pdb
import copy

# third part dependencies
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# project dependencies
from common_dtypes import GomoryCut
from common_utils import solve_LP
import datasets as custom_datasets
VERBOSE = False


class BaseCutSelectPolicy:
    def __init__(self, name="", ncuts=1) -> None:
        self.name = name
        self.type = "cutselect"
        self.ncuts = ncuts

    def act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ):
        assert self.ncuts <= len(
            cuts
        ), f"ncuts should be less than length of cuts received a cuts list of length {len(cuts)} and the policy is expected to select {self.ncuts} cuts"
        assert all(cut!=[] for cut in cuts), f"cuts should not be empty"
        print(f"Started Acting for {self.name}")
        return self._act(cuts, A, b, c, x, SimplexTableau, SimplexSolution)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ):
        """Given a state, return an action."""
        raise NotImplementedError
    
    def sort_and_crop_cuts(self, metrics: np.array, cuts: List[GomoryCut]):
        """Given a list of metrics and a list of cuts. Adds metrics as cut socres and sorts the cuts in a descending order according to the metrics"""
        assert(len(metrics) == len(cuts))
        self.ensure_metrics_as_npfloats(metrics)
        for i in range(len(metrics)):
            cuts[i].score = metrics[i]
        idxes = np.argsort(metrics)
        cut_dict = {i: cuts[i] for i in range(len(cuts))}
        selected_cuts = [cut_dict[idx] for idx in idxes if idx in cut_dict]
        selected_cuts = list(reversed(selected_cuts)) # descending order
        return selected_cuts[: self.ncuts]

    @staticmethod
    def ensure_metrics_as_npfloats(metrics: np.array, input_value: float = 0.0):
        if isinstance(metrics, list):
            metrics = np.array(metrics, dtype=np.float32)
        starting_metrics = copy.deepcopy(metrics)
        final_metrics = np.array([metrics[i] if (isinstance(metrics[i], np.float32) or isinstance(metrics[i], np.float64)) else input_value for i in range(len(metrics))])
        if not np.all(starting_metrics == final_metrics):
            import pdb; pdb.set_trace()
        return final_metrics

class RandomCutPolicy(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="random", ncuts=ncuts)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        """Returns ncuts different random elements of the cuts list"""
        idxes = np.random.choice(np.arange(len(cuts)), size=self.ncuts, replace=False)
        seleced_cuts = [cuts[idx] for idx in idxes]
        return seleced_cuts


class MVPolicy(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="most_violated", ncuts=ncuts)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        # it = arg max{|[x∗B (t)]i − round([x∗B (t)]i )|}.
        xb_frac = np.abs(SimplexSolution - np.round(SimplexSolution))
        tableau_idxs = [cut.tableau_idx for cut in cuts]
        xb_frac = [xb_frac[i] for i in tableau_idxs]
        return self.sort_and_crop_cuts(metrics=xb_frac, cuts=cuts)

class MNVPolicy(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="most_negative_violated", ncuts=ncuts)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        assert (
            SimplexTableau is not None
        ), "SimplexTableau should not be None for MNV Policy"
        #  it = arg max{|[x∗B (t)]i − round([x∗B (t)]i )|/∥A ̃i ∥}.
        xb_frac = np.abs(SimplexSolution - np.round(SimplexSolution))
        xb_frac_norm = xb_frac / np.linalg.norm(SimplexTableau)
        tableau_idxs = [cut.tableau_idx for cut in cuts]
        xb_frac_norm = [xb_frac_norm[i] for i in tableau_idxs]
        return self.sort_and_crop_cuts(metrics=xb_frac_norm, cuts=cuts)

class LEPolicy(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="lexicographical", ncuts=ncuts)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        # it = arg min{i, [x∗B (t)]i is fractional}.
        idxes = np.where(np.abs(SimplexSolution - np.round(SimplexSolution)) > 1e-6)[0]
        cut_dict = {cut.nonbasic_var_idx: cut for cut in cuts}
        selected_cuts = [cut_dict[idx] for idx in idxes if idx in cut_dict]
        return selected_cuts[: self.ncuts]

class MinSimilar(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="minsimilar", ncuts=ncuts)

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        # it = argmin_row{np.abs(Cuts_Matrix^t c)}}.
        Cuts_Matrix = np.array([cut.coefficients for cut in cuts])
        scores = np.matmul(Cuts_Matrix, c)
        return self.sort_and_crop_cuts(metrics=scores, cuts=cuts)


class MAXLPBoundPolicy(BaseCutSelectPolicy):
    def __init__(self, ncuts=1) -> None:
        super().__init__(name="max_lp_bound", ncuts=ncuts)
    
    def _act(
        self,
        cuts: List[GomoryCut],
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        x = None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        # solves the next LP adding the cut and returns the cuts with the largest difference in the objective function (ie. the cuts with the smallest objective function value)
        cut_objective_function_values = []
        # prev_lp_obj_value = np.dot(x, c)
        for cut in cuts:
            A = np.concatenate((A, np.array([cut.coefficients])), axis=0)
            b = np.concatenate((b, np.array([cut.rhs])), axis=0)
            res = solve_LP(A, b, c)
            new_bound = res["objective_value"]
            cut_objective_function_values.append(new_bound)
            A = A[:-1]
            b = b[:-1]
        return self.sort_and_crop_cuts(metrics=cut_objective_function_values, cuts=cuts)

class SimpleNeuralPolicy(BaseCutSelectPolicy):
    """
    Interface for a neural policy. A neural policy is has the following pipeline input -> preprocesser(input) -> model(·) -> ?postprocesser(·) -> cut selection
    """
    def __init__(self, name:str, model: torch.nn.Module, ncuts: int=1) -> None:
        super().__init__(name=name, ncuts=ncuts)
        print(self.name)
        self.model = model

    def _act(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> List[GomoryCut]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        model_inputs = self.preprocesser(cuts, A, b, c, x, SimplexTableau, SimplexSolution)
        model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
        scores = self.model(**model_inputs)
        if self.postprocesser is not None:
            scores = self.postprocesser(scores)
        return self.sort_and_crop_cuts(metrics=scores, cuts=cuts)
    
    @staticmethod
    def preprocessor(
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    @staticmethod
    def postprocesser(
        scores: torch.Tensor
    ) -> np.array:
        raise NotImplementedError
    
class NeuralCutsPolicy(SimpleNeuralPolicy):
    def __init__(self, name:str, model: torch.nn.Module, ncuts: int=1) -> None:
        super().__init__(name=name, model=model, ncuts=ncuts)
    
    def preprocesser(
        self,
        cuts: List[GomoryCut],
        A=None,
        b=None,
        c=None,
        x=None,
        SimplexTableau=None,
        SimplexSolution=None,
    ) -> Dict[str, torch.Tensor]:
        return {"cuts": custom_datasets.CutsDataset.preprocess_cuts_from_cut_list_solution_c(cuts, x, c)}
    
    def postprocesser(
        self,
        scores: torch.Tensor,
        mode="sigmoid"
    ) -> np.array:
        if mode == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            scores = sigmoid(scores)
            if len(scores.shape) > 1:
                scores = scores.view(-1)
            scores = scores.detach().cpu().numpy()
        else:
            raise NotImplementedError
        return scores