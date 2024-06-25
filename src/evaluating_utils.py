# std lib dependencies
import sys
import os
from datetime import date, time
import pdb

# third party dependencies
from tqdm import tqdm
import pickle as pkl
from logger import logger
import numpy as np
import torch
from typing import List, Tuple, Dict

# project dependencies
from environment import CutSelEnv, BaseCutSelEnv
from policies import BaseCutSelectPolicy
from agents import BaseCutSelectAgent
from common_dtypes import CutSelResult, ListCutSelResult


def select_different_instances(
    instance_path: str, n_instances: int, seed: int = 0
) -> List[str]:
    """Selects n_instances different instances from instance_path."""
    instance_list = os.listdir(instance_path)
    assert len(instance_list) >= n_instances, f"n_instances={n_instances} is too large"
    np.random.seed(42)  # fixed seed
    instance_list = np.random.choice(instance_list, n_instances, replace=False)
    return instance_list


def multipolicy_data_gathering(
    problem_filepath: str,
    policies: List[BaseCutSelectPolicy],
    ninstances: int,
    inference_mode: List[str],
    csv_path: str,
    pickle_path: str,
    agent_mode: List[str],
    ncuts_limit: int = 30,
    skip_instances: List[str] = None,
    save_trajectories_path: str = None,
    fixed_selected_instances: List[str] = None,
) -> ListCutSelResult:
    assert all(
        [imode in ["cutselect", "cutremove"] for imode in inference_mode]
    ), "All inference mode should be in ['cutselect', 'cutremove']"
    results = []
    saving_results = []
    if fixed_selected_instances is not None:
        instance_list = fixed_selected_instances
    else:
        instance_list = select_different_instances(problem_filepath, ninstances)
    if skip_instances is not None:
        instance_list = [inst for inst in instance_list if inst not in skip_instances]
    env = CutSelEnv(
        instance_file_path=problem_filepath,
        fixed_ncuts_limit=ncuts_limit,
        save_trajectories_path=save_trajectories_path,
    )
    for i in tqdm(range(len(instance_list))):
        env.instance_choice = instance_list[i]
        for j in range(len(policies)):
            policy = policies[j]
            env.inference_mode = inference_mode[j]
            agent = BaseCutSelectAgent(policy=policy, mode=agent_mode[j])
            result = env.step(agent)
            results.append(result)
            saving_results.append(result)
            logger.info(f"Completed step result: {result}")
        if isinstance(csv_path, str):
            saving_results = ListCutSelResult(saving_results)
            saving_results.to_csv(filename=csv_path, append_mode=True)
            saving_results = []
    if isinstance(pickle_path, str):
        with open(pickle_path, "wb") as f:
            pkl.dump(results, f)
    return results


def fix_all_seeds(seed: int = None):
    if seed is None:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
