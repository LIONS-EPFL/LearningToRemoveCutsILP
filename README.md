[![License](https://img.shields.io/badge/MIT_License-lightgreen?style=for-the-badge)](./LICENSE)
# Learning to Remove Cuts in Integer Linear Programming
Official implementation of the **ICML'24 paper "Learning to Remove Cuts in Integer Linear Programming"**.

## Repository Installation:
* Install the [SCIP solver](https://www.scipopt.org/index.php#download) on your machine. The code was tested on version 8.0 for our OS.
* Get a working [python installation](https://www.python.org/downloads/) on your machine. The code was tested on version 3.8.10.
* Install the dependencies for the project specified in the 'requirements.txt' file. For example, using virtualenv and pip you could:
```bash
python -m virtualenv <your-venv-name> # create a virtualenv
```
```bash
source <your-venv-name>/bin/activate # activate a virtualenv
```
```bash
python -m pip install -r 'requirements.txt' # install the dependencies
```

## Sample Scripts:
### Generating instances:
In order to generate and save multiple instances in the data/instances folder inside the project run:  
```bash
python src/script_generate_instances_and_trajectories.py --instances <instance-name-and dims: str> --n_samples  <number-of-instances: int>
```

The formatting for 'instance name and dims' for the benchmarks is as follows:
packing_&lt;n&gt;_&lt;m&gt;, binpacking_&lt;n&gt;_&lt;m&gt;, maxcut_&lt;N&gt;_&lt;E&gt;, production_planning_&lt;T&gt;, set_cover_&lt;n&gt;_&lt;m&gt;

For example, the command to generate 3000 instances for Max Cut N=14, E=40 would be:  
```bash
python src/script_generate_instances_and_trajectories.py --instances maxcut_14_40 --n_samples  3000
```

The instances will be saved inside a subfolder of data/instances named as '&lt;instance name and dims&gt;' and each generation will have a subfolder named 'sample_&lt;i&gt;' containing A, b, c numpy objects for the optimization problem an '.mps' representation and a '.txt' containing SCIP solution data for the IGC computation and environment's sanity checks.

### Generating Trajectories
In order to generate trajectories of the expert policy for some instances and save them in data/trajectories run:
```bash
python src/script_generate_instances_and_trajectories.py --t --instances <instance-name-and dims: str> --n_samples  <number-of-instances: int>
```

The formatting is the same as specified in "Generating the Instances".

For example, the command to generate 3000 trajectories for Max Cut N=14, E=40 would be:  
```bash
python src/script_generate_instances_and_trajectories.py --t --instances maxcut_14_40 --n_samples  10
```

Note that in order to properly generate the trajectories the instances must exist.

The trajectories will be saved inside a subfolder of data/trajectories named as '&lt;instance name and dims&gt;', each generation will have a subfolder named 'sample_&lt;i&gt;' containing one folder per iteration '&lt;j&gt;' with a 'trajectory_datapoint.pkl' with the stored data for that iteration.

### Training a Neural Policy $\pi_\theta$:
Training a Neural Policy $\pi_\theta$ requires two simple steps:
1. Generating the training dataset from the trajectories. We compute and preprocess the X,y datapoints from the trajectories offline by running a unshuffled one batch run. This is only required once per trajectory and will generate a 'preprocessed' folder inside data/trajectories/&lt;instance name and dims&gt;/sample_&lt;i&gt;/&lt;j&gt;/ that the Pytorch training dataset will access.
```bash
python src/script_train.py --instances <instance-name-and-dims: str> --epochs 1 --shuffle 0 --batch_size 1
```
2. Training the model: After we can train our model as we wish, see the script args for details on the options.
```bash
python src/script_train.py --instances <instance-name-and-dims: str>
```
### Benchmarking policies:
In order to benchmark muliple policies against each other and collect the results we run the following command.
```bash
python src/script_benchmark.py --benchmarking_dataset <instance-name-and-dims: str> --benchmarking_samples <n> --neural_checkpoints_path_list './data/experiment_results/checkpoints/<run_id>/model_<i>_weights.pth'
```

### Cutpool quality:
To collect data for cutpool quality analysis set the `collect_and_save_cpl` parameter in the environment class to true.

### SCIP + NN verification dataset:
The pyscipopt interface implementation used to benchmark the `nn verification` dataset can be found in [this script](src/script_scip_nn_verification.py). The dataset is available at [deepmind's github](https://github.com/google-deepmind/deepmind-research/tree/master/neural_mip_solving).

## Extending Functionalities:
You can add your own policies and architectures to re-use the pipelines by extending the base classes in `policies.py`, `models.py` and `datasets.py` files respectively and then adding them as supported in the scripts.

## Contact:
Please refer to first author's email for any inquiries or questions, happy to chat about it!