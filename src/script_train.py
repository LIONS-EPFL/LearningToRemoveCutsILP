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

from models import SimpleMLP
from datasets import UpdatedScoresHistoricCutsDataset, CutsDataset
from evaluating_utils import multipolicy_data_gathering
from common_dtypes import GomoryCut, StorageTrajectoryDatapoint

value = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"Variable CUDA_VISIBLE_DEVICES {value}")
assert torch.cuda.is_available(), "CUDA not available"

SUPPORTED_ARCHITECTURES = ["MLP-Cuts-Features"]
SUPPORTED_OPTIMIZERS = ["SGD", "Adam", "Adagrad"]

argparse = ArgumentParser()
# Non-Default Arguments -------------------------------------------------------
argparse.add_argument("--instances", type=str, help="Dataset to train on")
argparse.add_argument(
    "--architecture",
    type=str,
    choices=SUPPORTED_ARCHITECTURES,
    help="Architecture to use",
    default="MLP-Cuts-Features",
)
argparse.add_argument(
    "--optimizer",
    type=str,
    default="SGD",
    choices=["SGD", "Adam", "Adagrad"],
    help="Optimizer to use",
)
argparse.add_argument(
    "--loss", type=str, default="MSE", choices=["MSE"], help="Loss function to use"
)
argparse.add_argument(
    "--learning_rate", type=float, default=0.005, help="Learning rate for the optimizer"
)
argparse.add_argument(
    "--epochs", type=int, default=30, help="Number of epochs to train for"
)
argparse.add_argument(
    "--batch_size",
    help="Batch size, if 'full' is passed, the full dataset will be used as a batch",
    default=10_000,
)
argparse.add_argument(
    "--patience_max", type=int, default=5, help="Patience for early stopping"
)
argparse.add_argument(
    "--checkpoints_path",
    type=str,
    default="./data/experiment_results/checkpoints",
    help="Path to save the checkpoints",
)
argparse.add_argument(
    "--post_model_function",
    type=str,
    default="sigmoid",
    choices=["sigmoid"],
    help="Post model function to apply to the output of the model",
)
argparse.add_argument(
    "--truncate",
    type=int,
    default=None,
    help="Truncate the dataset to <truncate> datapoints, useful for debugging. If None, no truncation is done",
)
argparse.add_argument(
    "--run_name",
    type=str,
    default=None,
    help="Name of the run, one experiment might contain multiple runs",
)
argparse.add_argument("--use_wandb", type=int, default=1, help="Set to 1 to use wandb")
argparse.add_argument(
    "--scheduler",
    type=str,
    default=None,
    choices=["ReduceLROnPlateau", None],
    help="Scheduler to use",
)
argparse.add_argument("--shuffle", type=int, default=0, help="Shuffle the dataset")
args = argparse.parse_args()

if isinstance(args.batch_size, str) and args.batch_size.isdigit():
    args.batch_size = int(args.batch_size)

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

path = f"./data/trajectories/{args.instances}_train/"
validation_path = f"./data/trajectories/{args.instances}_validation/"
test_path = f"./data/trajectories/{args.instances}_test/"
batch_size = args.batch_size
device = "cuda:0"
epochs = args.epochs
patience_max = args.patience_max
checkpoints_path = os.path.join(args.checkpoints_path, run_name)
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

# load dataset
dataset = UpdatedScoresHistoricCutsDataset(dataset_path=path, truncate=args.truncate)
validation_dataset = UpdatedScoresHistoricCutsDataset(
    dataset_path=validation_path, truncate=args.truncate
)
test_dataset = UpdatedScoresHistoricCutsDataset(
    dataset_path=test_path, truncate=args.truncate
)

if args.batch_size == "full":
    batch_size = len(dataset)
    batch_size_validation = len(validation_dataset)
else:
    batch_size = args.batch_size
    batch_size_validation = args.batch_size
batch_size_test = 1
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=bool(args.shuffle),
    collate_fn=CutsDataset.collate_fn,
)
print(f"Train dataset with Dataloader len {len(dataloader)}")
validation_dataloader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=batch_size_validation,
    shuffle=bool(args.shuffle),
    collate_fn=CutsDataset.collate_fn,
)
print(f"Validation dataset with Dataloader len {len(validation_dataloader)}")
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size_test,
    shuffle=bool(args.shuffle),
    collate_fn=CutsDataset.collate_fn,
)
print(f"Test dataset with Dataloader len {len(test_dataloader)}")

if args.architecture == "MLP-Cuts-Features":
    model = SimpleMLP(input_size=14, hidden_layers=3, hidden_size=512).to(device)
    input_keys = ["cuts"]

if args.post_model_function == "sigmoid":
    post_model_function = torch.sigmoid
elif args.post_model_function is None:
    post_model_function = lambda x: x

if args.optimizer == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "Adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)

if args.scheduler == "ReduceLROnPlateau":
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10, factor=0.7, verbose=True
    )
else:
    scheduler = None

if args.loss == "MSE":
    loss = torch.nn.MSELoss()

print("Starting dry run with no gradients")
loss_vals = []
validation_loss_vals = []
with torch.no_grad():
    batch_loss = 0
    print(f"Starting dry training loop for {len(dataloader)} batches")
    for batch, raw in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        model_input = {key: value for key, value in batch.items() if key in input_keys}
        output = model(**model_input)
        output = post_model_function(output)
        loss_val = loss(output, batch["label"])
        batch_loss += loss_val.item()
    mean_batch_loss = batch_loss / len(dataloader)
    loss_vals.append(mean_batch_loss)

    epoch_validation_loss_val = 0
    print(f"Starting dry validation loop for {len(dataloader)} batches")
    for batch, raw in validation_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        model_input = {key: value for key, value in batch.items() if key in input_keys}
        output = model(**model_input)
        output = post_model_function(output)
        loss_val = loss(output, batch["label"])
        epoch_validation_loss_val += loss_val.item()
    epoch_validation_loss_val /= len(validation_dataloader)
    if args.use_wandb:
        wandb.log(
            {
                "train_loss": mean_batch_loss,
                "validation_loss": epoch_validation_loss_val,
            }
        )

best_validation_loss = epoch_validation_loss_val
index_best_model = 0
patience_counter = 0

for i in tqdm(range(epochs)):
    batch_loss = 0
    for batch, raw in dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        model_input = {key: value for key, value in batch.items() if key in input_keys}
        optimizer.zero_grad()
        output = model(**model_input)
        output = post_model_function(output)
        loss_val = loss(output, batch["label"])
        loss_val.backward()
        optimizer.step()
        batch_loss += loss_val.item()
    mean_batch_loss = batch_loss / len(dataloader)
    if scheduler is not None:
        scheduler.step(mean_batch_loss)
    loss_vals.append(mean_batch_loss)

    epoch_validation_loss_val = 0
    for batch, raw in validation_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        model_input = {key: value for key, value in batch.items() if key in input_keys}
        output = model(**model_input)
        output = post_model_function(output)
        loss_val = loss(output, batch["label"])
        epoch_validation_loss_val += loss_val.item()
    epoch_validation_loss_val /= len(validation_dataloader)
    if epoch_validation_loss_val < best_validation_loss:
        best_validation_loss = epoch_validation_loss_val
        torch.save(
            model.state_dict(), os.path.join(checkpoints_path, f"model_{i}_weights.pth")
        )
        index_best_model = i
        patience_counter = 0
    else:
        patience_counter += 1
    validation_loss_vals.append(epoch_validation_loss_val)
    if args.use_wandb:
        wandb.log(
            {
                "train_loss": mean_batch_loss,
                "validation_loss": epoch_validation_loss_val,
            }
        )
    if patience_counter == patience_max:
        print("Early stopping")
        break

# test against test set
if args.architecture == "MLP-Cuts-Features":
    print(f"Best model is model {i} with validation loss {best_validation_loss}")
    best_model = SimpleMLP(input_size=14, hidden_layers=3, hidden_size=512)
    best_model.load_state_dict(
        torch.load(
            os.path.join(checkpoints_path, f"model_{index_best_model}_weights.pth")
        )
    )
    torch.save(
        best_model.state_dict(),
        os.path.join(checkpoints_path, "best_model_weights.pth"),
    )
    best_model.to(device).eval()
test_loss = 0

with torch.no_grad():
    for batch, raw in test_dataloader:
        batch = {key: value.to(device) for key, value in batch.items()}
        model_input = {key: value for key, value in batch.items() if key in input_keys}
        output = best_model(**model_input)
        output = post_model_function(output)
        loss_val = loss(output, batch["label"])
        test_loss += loss_val.item()
    test_loss /= len(test_dataloader)
    if args.use_wandb:
        wandb.log(
            {
                "test_loss": test_loss,
            }
        )
if args.use_wandb:
    wandb.finish()
