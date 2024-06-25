# std lib dependencies
import os
from typing import List, Dict, Optional, Tuple
import sys

# third party dependencies
import torch
from torch.utils.data import Dataset
import numpy as np

# project dependencies
from common_utils import repr_integers_as_integers, solve_LP
from common_dtypes import GomoryCut, StorageTrajectoryDatapoint, RowFeatures


class CutsDataset(Dataset):
    def __init__(self, dataset_path: str, truncate: Optional[int]=None, preprocessed_key: str = "preprocessed"):
        # dataset_path:
        #   - instance_folder
        #       - iteration_folder
        instance_folders = [instance_foler for instance_foler in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, instance_foler))]
        self.dataset = [os.path.join(dataset_path, instance_folder, iteration_folder, "trajectory_datapoint.pkl") for instance_folder in instance_folders for iteration_folder in os.listdir(os.path.join(dataset_path, instance_folder)) if os.path.isdir(os.path.join(dataset_path, instance_folder, iteration_folder))]
        self.preprocessed_key = preprocessed_key
        if truncate is not None:
            self.dataset = self.dataset[:truncate]
        self.index_data = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        base_path = os.path.dirname(self.dataset[index])
        if os.path.exists(os.path.join(base_path, self.preprocessed_key)):
            return self.load_item_offline_preprocessing(index)
        return self.load_item_online_preprocessing(index)
    
    def load_item_online_preprocessing(self, index):
        # load index data
        self.index_data = self.dataset[index]
        datapoint = StorageTrajectoryDatapoint.load_pkl(self.index_data)
        cuts = self.preprocess_cuts(datapoint)
        label = self.preprocess_label(datapoint)
        base_path = os.path.dirname(self.index_data)
        preprocessed_path = os.path.join(base_path, self.preprocessed_key)
        # save preprocessed data
        if not os.path.exists(preprocessed_path):
            os.makedirs(preprocessed_path)
        torch.save(cuts, os.path.join(preprocessed_path, "cuts.pt"))
        torch.save(label, os.path.join(preprocessed_path, "label.pt"))
        np.save(os.path.join(preprocessed_path, "raw_label.npy"), np.array(datapoint.Expert.label))
        return {"cuts": cuts, "label": label, "raw_label": datapoint.Expert.label}
    
    def load_item_offline_preprocessing(self, index):
        # load index data
        self.index_data = self.dataset[index]
        base_path = os.path.dirname(self.index_data)
        preprocessed_path = os.path.join(base_path, self.preprocessed_key)
        cuts = torch.load(os.path.join(preprocessed_path, "cuts.pt"))
        label = torch.load(os.path.join(preprocessed_path, "label.pt"))
        raw_label = np.load(os.path.join(preprocessed_path, "raw_label.npy"))
        return {"cuts": cuts, "label": label, "raw_label": raw_label}

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]):
        # At different iterations, the number of cuts and constraints varies, thus, we pad the data to the maximum number of cuts, constraints and label with/zeros
        max_number_cuts = max([len(data["cuts"]) for data in batch])
        batch_data = {
            "cuts": torch.zeros((len(batch), max_number_cuts, batch[0]["cuts"].shape[1])),
            "label": torch.zeros((len(batch), max_number_cuts, 1))
        }
        raw_data = {
                "raw_label": [None] * len(batch)
        }
        # batch = [{cuts: <tensor>, constraints: <tensor>, label: <tensor>}, ...] -> batch_data = {cuts: <tensor>, constraints: <tensor>, label: <tensor>}
        for i, data in enumerate(batch):
            batch_data["cuts"][i, :data["cuts"].shape[0], :] = data["cuts"]
            batch_data["label"][i, :data["label"].shape[0], :] = data["label"]
            raw_data["raw_label"][i] = data["raw_label"]
        return batch_data, raw_data

    @staticmethod
    def collate_fn_with_masking(batch: List[Dict[str, torch.Tensor]]):
        max_number_cuts = max([len(data["cuts"]) for data in batch])
        batch_data = {
            "cuts": torch.zeros((len(batch), max_number_cuts, batch[0]["cuts"].shape[1])),
            "label": torch.zeros((len(batch), max_number_cuts, 1)),
            "mask": torch.zeros((len(batch), max_number_cuts, 1), dtype=torch.bool)
        }
        # batch = [{cuts: <tensor>, constraints: <tensor>, label: <tensor>}, ...] -> batch_data = {cuts: <tensor>, constraints: <tensor>, label: <tensor>}
        for i, data in enumerate(batch):
            batch_data["cuts"][i, :data["cuts"].shape[0], :] = data["cuts"]
            batch_data["label"][i, :data["label"].shape[0], :] = data["label"]
            batch_data["mask"][i, :data["cuts"].shape[0], :] = torch.ones((data["cuts"].shape[0], 1), dtype=torch.bool)
        return batch_data

    def preprocess_cuts(self, datapoint: StorageTrajectoryDatapoint):
        cuts = datapoint.Cuts.cuts
        solution = datapoint.LP.x_star_LP
        c = datapoint.ILP.c
        return self.preprocess_cuts_from_cut_list_solution_c(cuts, solution, c)

    @staticmethod
    def preprocess_cuts_from_cut_list_solution_c(cuts: List[GomoryCut], solution: np.array, c: np.array, is_historic: int = 0):
        assert is_historic in [0, 1], "is_historic must be 0 or 1"
        row_features = [
            RowFeatures.from_cut_solution_c(cut, solution, c, is_historic=is_historic)
            for cut in cuts
        ]
        np_features = np.array([row_feature.to_numpy() for row_feature in row_features])
        torch_features = torch.tensor(np_features, dtype=torch.float32)
        return torch_features

    def preprocess_label(self, datapoint: np.array):
        return common_preprocess_labels(datapoint)

class HistoricCutsDataset(CutsDataset):
    """This considers the new features of the historic cuts with the old label scores as the label."""
    def __init__(self, dataset_path: str, truncate: Optional[int]=None, preprocessed_key: str = "preprocessed_historic", ncuts: int = 1):
        super().__init__(dataset_path, truncate, preprocessed_key=preprocessed_key)
        self.dataset = sorted(self.dataset, key=lambda x: (x.split("/")[-3], int(x.split("/")[-2])))
        self.ncuts = ncuts
    
    def preprocess_cuts(self, datapoint: StorageTrajectoryDatapoint) -> torch.Tensor:
        cuts = datapoint.Cuts.cuts
        solution = datapoint.LP.x_star_LP
        c = datapoint.ILP.c
        current_candidate_cuts = self.preprocess_cuts_from_cut_list_solution_c(cuts, solution, c, is_historic=0)
        raw_historic_candidate_cuts, _ = self.get_historic_cuts_and_labels()
        if len(raw_historic_candidate_cuts) == 0:
            return current_candidate_cuts
        historic_candidate_cuts = self.format_previous_cuts(raw_historic_candidate_cuts, solution, c)
        final_cuts = torch.cat([historic_candidate_cuts, current_candidate_cuts], dim=0)
        return final_cuts
    
    def format_previous_cuts(self, previous_cuts: List[List[GomoryCut]], solution: float, c: np.array) -> torch.Tensor:
        previous_cuts = [cut for pc in previous_cuts for cut in pc]
        previous_cuts = self.preprocess_cuts_from_cut_list_solution_c(previous_cuts, solution, c, is_historic=1)
        return previous_cuts

    def preprocess_label(self, datapoint: StorageTrajectoryDatapoint) -> torch.Tensor:
        current_candidate_labels = preprocess_labels_from_bounds_and_value_LP(datapoint.Expert.label, datapoint.LP.x_star_value_LP)
        previous_cuts, previous_labels = self.get_historic_cuts_and_labels()
        if len (previous_labels) == 0:
            return current_candidate_labels
        previous_labels = self.format_previous_labels(previous_labels, previous_cuts)
        final_labels = torch.cat([previous_labels, current_candidate_labels], dim=0)
        return final_labels
    
    def format_previous_labels(self, previous_labels: List[List[float]], previous_cuts: List[GomoryCut]) -> torch.Tensor:
        previous_labels = [label for pl in previous_labels for label in pl]
        previous_labels = torch.tensor(previous_labels, dtype=torch.float32)
        previous_labels = previous_labels.unsqueeze(1)
        return previous_labels
    
    def get_previous_iteration_path(self) -> str or None:
        """Gets the path of the previous iteration. If current iteration is 0 returns None"""
        base_path_iteration = os.path.dirname(self.index_data)
        iteration = int(base_path_iteration.split("/")[-1])
        if iteration == 0:
            return None
        instance_path = os.path.dirname(base_path_iteration)
        return os.path.join(instance_path, str(iteration - 1))
    
    def get_historic_cuts_and_labels(self)-> Tuple[List[List[GomoryCut]], List[List[float]]]:
        best_previous_cuts = []
        best_previous_labels = []
        initial_index_data = self.index_data
        previous_iteration_path = self.get_previous_iteration_path()
        while previous_iteration_path is not None:
            best_cut, best_label = self.get_best_cut_and_label(previous_iteration_path)
            best_previous_cuts.append(best_cut)
            best_previous_labels.append(best_label)
            self.index_data = os.path.join(previous_iteration_path, "trajectory_datapoint.pkl")
            previous_iteration_path = self.get_previous_iteration_path()
        self.index_data = initial_index_data
        best_previous_cuts.reverse()
        best_previous_labels.reverse()
        return best_previous_cuts, best_previous_labels

    def get_best_cut_and_label(self, path: str)-> Tuple[GomoryCut, np.array]:
        datapoint = StorageTrajectoryDatapoint.load_pkl(os.path.join(path, "trajectory_datapoint.pkl"))
        cuts = datapoint.Cuts.cuts
        label = datapoint.Expert.label
        best_cut_idxs = np.argsort(label)[-self.ncuts:]
        best_cuts = [cuts[best_cut_idx] for best_cut_idx in best_cut_idxs]
        best_labels = [label[best_cut_idx] for best_cut_idx in best_cut_idxs]
        return best_cuts, best_labels

class UpdatedScoresHistoricCutsDataset(HistoricCutsDataset):
    def __init__(self, dataset_path: str, truncate: Optional[int]=None, preprocessed_key: str = "preprocessed_updated_scores_historic_2"):
        super().__init__(dataset_path, truncate, preprocessed_key=preprocessed_key)
        self.reserved_m = None

    def set_reserved_m(self):
        base_path_iteration = os.path.dirname(self.index_data)
        iteration = int(base_path_iteration.split("/")[-1])
        if iteration == 1: # this won't be accessed in iteration 0
            iteration_0_datapoint_path = os.path.join(os.path.dirname(base_path_iteration), "0", "trajectory_datapoint.pkl")
            self.reserved_m = StorageTrajectoryDatapoint.load_pkl(iteration_0_datapoint_path).ILP.A.shape[0]
        assert self.reserved_m is not None, "reserved_m must be set before calling this method or be called in first iteration"

    def format_previous_labels(self, previous_labels: List[List[float]], previous_cuts: List[GomoryCut]) -> torch.Tensor:
        """Refreshes the previous labels with the updated scores of the expert policy.add()
        The labels for cut removal are defined in the analogous way as the labels for cut addition. We measure the bound improvement and scale it. Ignores previous labels.
        """
        previous_cuts = [cut for pc in previous_cuts for cut in pc] # flatten previous cuts
        datapoint = StorageTrajectoryDatapoint.load_pkl(self.index_data)
        A = datapoint.ILP.A
        b = datapoint.ILP.b
        c = datapoint.ILP.c
        x_star_value_LP = datapoint.LP.x_star_value_LP
        scores = []
        labels = []
        for cut in previous_cuts:
            A_prime, b_prime = self.A_b_no_cut(cut, A, b)
            ret_sol = solve_LP(A_prime, b_prime, c)
            score = ret_sol["objective_value"]
            label = preprocess_label_from_bounds_and_value_LP_to_float(new_bound=x_star_value_LP, value_LP=score)
            scores.append(score)
            labels.append(label)
        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.unsqueeze(1)
        return labels

    def A_b_no_cut(self, cut: GomoryCut, A: np.array, b: np.array) -> Tuple[np.array, np.array, np.array]:
        """Removes the cut from the A, b and c matrices on match with the cut coefficients outside of the reserved section (for the constraints)."""
        self.set_reserved_m()
        A_non_reserved = A[self.reserved_m:, :]
        b_non_reserved = b[self.reserved_m:]
        full_constraints = np.concatenate([A_non_reserved, b_non_reserved.reshape(-1, 1)], axis=1)
        cut_coefficients = cut.coefficients
        cut_rhs = np.array([cut.rhs])
        cuts_constraints = np.concatenate([cut_coefficients, cut_rhs], axis=0)
        cuts_constraints = cuts_constraints.reshape(1, -1)
        mask = np.logical_not(np.any(np.all(full_constraints[:, None, :] == cuts_constraints, axis=-1), axis=1))
        A_no_cut = A_non_reserved[mask, :]
        b_no_cut = b_non_reserved[mask]
        A_complete_no_cut = np.concatenate([A[:self.reserved_m, :], A_no_cut], axis=0)
        b_complete_no_cut = np.concatenate([b[:self.reserved_m], b_no_cut], axis=0)
        return A_complete_no_cut, b_complete_no_cut


# PREPROCESSING UTILS ------------------------------------------------------
def common_preprocess_labels(datapoint: StorageTrajectoryDatapoint):
    """Scale label to be in [0, 1]"""
    return preprocess_labels_from_bounds_and_value_LP(datapoint.Expert.label, datapoint.LP.x_star_value_LP)

def preprocess_labels_from_bounds_and_value_LP(new_LP_bounds_arr: np.array, value_LP: float) -> torch.Tensor:
    """Scale label to be in [0, 1]"""
    if value_LP == 0:
        old_LP_bounds_arr = np.ones(new_LP_bounds_arr.size)
    else:
        old_LP_bounds_arr = np.ones(new_LP_bounds_arr.size) * value_LP
    label = (new_LP_bounds_arr - old_LP_bounds_arr) / old_LP_bounds_arr
    label = repr_integers_as_integers(label)
    torch_label = torch.tensor(label, dtype=torch.float32)
    torch_label = torch_label.unsqueeze(1)
    return torch_label

def preprocess_label_from_bounds_and_value_LP_to_float(new_bound: float, value_LP: float) -> float:
    """Scale label to be in [0, 1]"""
    if value_LP == 0:
        old_bound = 1
    else:
        old_bound = value_LP
    label = (new_bound - old_bound) / old_bound
    label = float_int_to_int(label)
    return label

def float_int_to_int(value: float, epsilon: float = 1e-6):
    return int(value) if np.abs(value-np.rint(value)) < epsilon else value

def unit_1d_array(input_array: np.array):
    norm = np.linalg.norm(input_array)
    if norm == 0:
        return input_array
    return input_array / norm

def unit_2d_array(input_array_2d: np.ndarray):
    assert len(input_array_2d.shape) == 2 
    return np.apply_along_axis(unit_1d_array, 1, input_array_2d)

def scale_maxvalue_1d_array(input_array: np.array):
    maxvalue = np.max(np.abs(input_array))
    if maxvalue == 0:
        return input_array
    return input_array / maxvalue

def scale_maxvalue_2d_array(input_array_2d: np.ndarray):
    assert len(input_array_2d.shape) == 2 
    return np.apply_along_axis(scale_maxvalue_1d_array, 1, input_array_2d)

def has_duplicate_rows(array):
    # Convert the 2D array to a structured array with a view dtype
    view = array.view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    
    # Use numpy.unique to find unique rows
    unique_rows, counts = np.unique(view, return_counts=True)
    
    # Check if any row has a count greater than 1
    return np.any(counts > 1)
# ------------------------------------------------------------------------------