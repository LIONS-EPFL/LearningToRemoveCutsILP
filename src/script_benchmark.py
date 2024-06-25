# std lib dependencies
import os
import sys
import copy
from argparse import ArgumentParser

# third party dependencies
import torch
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# project dependencies
from policies import (
    RandomCutPolicy,
    MAXLPBoundPolicy,
    NeuralCutsPolicy,
    MinSimilar,
    MNVPolicy,
    MVPolicy,
    LEPolicy,
)
from models import SimpleMLP
from datasets import UpdatedScoresHistoricCutsDataset
from evaluating_utils import multipolicy_data_gathering
from common_dtypes import GomoryCut, StorageTrajectoryDatapoint

value = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"Variable CUDA_VISIBLE_DEVICES {value}")
# assert torch.cuda.is_available(), "CUDA not available"

SUPPORTED_ARCHITECTURES = ["MLP-Cuts-Features"]

argparse = ArgumentParser()
# Non-Default Arguments -------------------------------------------------------
argparse.add_argument(
    "--benchmarking_dataset",
    type=str,
    help="same to benchmark on the same dataset, name to benchmark on another",
    default="same",
)
argparse.add_argument(
    "--neural_checkpoints_path_list",
    type=str,
    nargs="+",
    help="List of paths to the neural checkpoints to benchmark",
)
argparse.add_argument(
    "--benchmarking_samples", type=int, help="Number of samples to benchmark on"
)
argparse.add_argument(
    "--benchmarking_results_path",
    type=str,
    help="Path to save the benchmarking results",
)
argparse.add_argument(
    "--acting_mode",
    type=str,
    default="cutremove",
    choices=["cutselect", "cutremove"],
    help="acting_mode for the neural policies, if cutremove, agent_mode should be reselect, if cutselect, agent_mode should be cutselect",
)
argparse.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Name of the run, one experiment might contain multiple runs",
)
argparse.add_argument(
    "--experiment_name",
    type=str,
    default="sample_experiment",
    help="Name of the experiment, an experiment might contain multiple runs which will be grouped in the collections.csv file for the experiment folder in the results",
)
argparse.add_argument("--use_wandb", type=int, default=0, help="Set to 1 to use wandb")
argparse.add_argument(
    "--architecture",
    type=str,
    choices=SUPPORTED_ARCHITECTURES,
    help="Architecture to use",
    default="MLP-Cuts-Features",
)
argparse.add_argument(
    "--ncuts_limit_benchmarks",
    type=int,
    help="Number of cuts to limit the benchmarking to",
    default=30,
)

args = argparse.parse_args()

if args.use_wandb:
    import wandb

    wandb.init(
        project="LearningCuts",
        entity="pol-puigdemont",
        config=args,
        name=run_name,
        mode="online" if args.use_wandb else "disabled",
    )
run_name = args.run_name if args.run_name is not None else wandb.util.generate_id()

benchmarking_dataset = args.benchmarking_dataset
benchmarking_path = f"./data/instances/{benchmarking_dataset}_test"
os.makedirs(benchmarking_path, exist_ok=True)
# device = "cuda:0"
benchmarking_results_path = os.path.join(args.benchmarking_results_path, run_name)
if not os.path.exists(benchmarking_results_path):
    os.makedirs(benchmarking_results_path)

loaded_models = []
# load the models from the checkpoint list
if args.architecture == "MLP-Cuts-Features":
    for i, checkpoint_path in enumerate(args.neural_checkpoints_path_list):
        model = SimpleMLP(input_size=14, hidden_layers=3, hidden_size=512)
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device).eval()
        loaded_models.append(model)
else:
    raise NotImplementedError
neural_policies = [
    NeuralCutsPolicy(name=f"neural_{i}", model=model, ncuts=1)
    for i, model in enumerate(loaded_models)
]
# test against other policies
best_policy_no_cutremove = NeuralCutsPolicy(
    name="neural_no_cutremove", model=loaded_models[-1], ncuts=1
)
baseline_policies = [
    best_policy_no_cutremove,
    RandomCutPolicy(),
    MAXLPBoundPolicy(),
    MinSimilar(),
    MNVPolicy(),
    MVPolicy(),
    LEPolicy(),
]
policies = neural_policies + baseline_policies
agent_mode_neural_policies = (
    "reselect" if args.acting_mode == "cutremove" else "cutselect"
)
acting_mode = [args.acting_mode for _ in range(len(neural_policies))] + [
    "cutselect" for _ in range(len(baseline_policies))
]
agent_mode = [agent_mode_neural_policies for _ in range(len(neural_policies))] + [
    "cutselect" for _ in range(len(baseline_policies))
]
fixed_selected_instances = [f for f in os.listdir(benchmarking_path)]
n__fixed_instances = 50
fixed_selected_instances = [
    fixed_selected_instances[i] for i in range(n__fixed_instances)
]
results = multipolicy_data_gathering(
    problem_filepath=benchmarking_path,
    policies=policies,
    ninstances=args.benchmarking_samples,
    inference_mode=acting_mode,
    csv_path=os.path.join(benchmarking_results_path, "benchmark.csv"),
    pickle_path=os.path.join(benchmarking_results_path, "benchmark.pkl"),
    agent_mode=agent_mode,
    ncuts_limit=args.ncuts_limit_benchmarks,
    save_trajectories_path=None,
    fixed_selected_instances=fixed_selected_instances,
)
experiment_collections_path = "./data/experiment_results/collections"
os.makedirs(experiment_collections_path, exist_ok=True)
# if a csv with experiment name.csv does not exist, create it, otherwise append to it the run_name (found in args.run_name), benchmarking_pickle_path os.path.join(benchmarking_results_path, "benchmark.csv")
if not os.path.exists(
    os.path.join(experiment_collections_path, f"{args.experiment_name}.csv")
):
    with open(
        os.path.join(experiment_collections_path, f"{args.experiment_name}.csv"), "w"
    ) as f:
        f.write("run_name,benchmarking_pickle_path\n")
        f.write(f"{args.run_name},{benchmarking_results_path}/benchmark.pkl\n")
else:
    with open(
        os.path.join(experiment_collections_path, f"{args.experiment_name}.csv"), "a"
    ) as f:
        f.write(f"{args.run_name},{benchmarking_results_path}/benchmark.pkl\n")
if args.use_wandb:
    wandb.finish()
