# std lib dependencies
import pdb
import pickle as pkl
import os
import sys
import shutil
from datetime import date, time

# third party dependencies
import numpy as np

# project dependencies
from policies import MAXLPBoundPolicy
from agents import BaseCutSelectAgent
from environment import CutSelEnv
from instance_generator import generate_random_instances


def trajectory_generation(
    instance_name: str,
    n_samples: int,
    training_ratio: int = 0.67,
    validation_size: int = 0.16,
    test_size: int = 0.16,
    instance_base_path: str = "./data/instances",
    trajectories_base_path: str = "./data/trajectories",
):
    """Rolls out the MAXLPBoundPolicy on the instances in <instance_base_path>/<instance_name> and saves the trajectories in <trajectories_base_path>/<instance_name>. The trajectories are split into training, validation and test sets.
    The extension of the labels for the previous cuts is done at a dataset level.
    """
    instance_file_path = os.path.join(instance_base_path, instance_name)
    save_trajectories_path = os.path.join(trajectories_base_path, instance_name)
    os.makedirs(save_trajectories_path, exist_ok=True)
    list_instances = os.listdir(instance_file_path)
    list_instances = np.array([li for li in list_instances])
    training_size = int(training_ratio * n_samples)
    validation_size = int(validation_size * n_samples)
    test_size = int(test_size * n_samples)
    assert len(list_instances) >= (
        training_size + validation_size + test_size
    ), f"Not enough instances in {instance_file_path}: {len(list_instances)} to generate {(training_size + validation_size + test_size)} trajectories"
    env = CutSelEnv(
        instance_file_path=instance_file_path,
        optimizer_seed=0,
        environment_seed=0,
        fixed_ncuts_limit=30,
        save_trajectories_path=save_trajectories_path,
        inference_mode="cutselect",
    )
    supervisor_policy = MAXLPBoundPolicy()
    agent = BaseCutSelectAgent(supervisor_policy)
    train_instances = np.random.choice(
        list_instances, size=training_size, replace=False
    )
    nonselected = list(set(list_instances) - set(train_instances))
    validation_instances = np.random.choice(
        nonselected, size=validation_size, replace=False
    )
    nonselected = list(set(nonselected) - set(validation_instances))
    test_instances = np.random.choice(nonselected, size=test_size, replace=False)
    print(f"Training instances {train_instances}")
    print(f"Validation instances {validation_instances}")
    print(f"Test instances {test_instances}")
    for key, instance_list in {
        "train": train_instances,
        "validation": validation_instances,
        "test": test_instances,
    }.items():
        mode_path = save_trajectories_path + f"_{key}"
        if not os.path.exists(mode_path):
            os.makedirs(mode_path)
        for i in range(len(instance_list)):
            env.instance_choice = instance_list[i]
            env.save_trajectories_path = mode_path
            results = env.step(agent)
            assert env.instance_choice == instance_list[i]
            # ensure that it converged or cut limit reached
            try:
                assert results.converged or results.cuts_limit_exceeded
            except AssertionError:
                pdb.set_trace()
    # copy instances to the corresponding folders
    test_instance_filepath = instance_file_path + "_test"
    if not os.path.exists(test_instance_filepath):
        os.makedirs(test_instance_filepath)
    for instance in test_instances:
        shutil.copytree(
            os.path.join(instance_file_path, instance),
            os.path.join(test_instance_filepath, instance),
        )


def instance_generation(
    instance_name: str,
    n_samples: int = 3000,
    instance_base_path: str = "./data/instances",
):
    """Generates <n_samples> instances of <instance_name> in <instance_base_path>/<instance_name>"""
    problem_family = instance_name.split("_")[0]
    problem_family = (
        "production_planning" if problem_family == "production" else problem_family
    )
    problem_family = "set_cover" if problem_family == "set" else problem_family
    assert (
        problem_family in SUPPORTED_FAMILIES
    ), f"Problem family {problem_family} not supported. Supported families are {SUPPORTED_FAMILIES}"
    n, m, V, E, T = None, None, None, None, None
    if problem_family in ["packing", "binpacking", "set_cover"]:
        if problem_family == "set_cover":
            n, m = instance_name.split("_")[2:]
        else:
            n, m = instance_name.split("_")[1:]
        n = int(n)
        m = int(m)
    elif problem_family == "maxcut":
        V, E = instance_name.split("_")[1:]
        V = int(V)
        E = int(E)
    elif problem_family == "production_planning":
        T = instance_name.split("_")[2]
        T = int(T)
    generate_random_instances(
        n=n,
        m=m,
        output_file_prefix=instance_base_path,
        n_samples=n_samples,
        model_type=problem_family,
        T_param_production_planning=T,
        N_param_max_cut=V,
        E_param_max_cut=E,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    SUPPORTED_FAMILIES = [
        "binpacking",
        "maxcut",
        "set_cover",
        "production_planning",
        "packing",
    ]
    parser.add_argument(
        "--t", action="store_true", help="Generate trajectories instead of instances"
    )
    parser.add_argument(
        "--instances",
        type=str,
        help="The instance name and dimensions format for supported families is: packing_<n>_<m>, binpacking_<n>_<m>, maxcut_<N>_<E>, production_planning_<T>, set_cover_<n>_<m>",
    )
    parser.add_argument("--n_samples", type=int, default=2)
    args = parser.parse_args()
    if args.t:
        print("Starting generation of trajectories")
        os.makedirs("./data/trajectories", exist_ok=True)
        trajectory_generation(args.instances, args.n_samples)
    else:
        print("Starting generation of instances")
        os.makedirs("./data/instances", exist_ok=True)
        instance_generation(args.instances, args.n_samples)
