import argparse
import copy
import csv
import os
import random

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from data import get_dataset
from models import get_model
from train_model import LocalUpdate
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, required=True, choices=["SyntheticBinary", "LocalBinaryCSV"])
parser.add_argument("--model", type=str, required=True, choices=["SyntheticMLP", "SyntheticMLPDeep"])
parser.add_argument(
    "--algs_to_run",
    nargs="+",
    type=str,
    required=True,
    choices=["fedavg", "fedfisher_diag", "fedfisher_kfac"],
)
parser.add_argument("--seed", type=int, required=False, default=0)
parser.add_argument("--alpha", type=float, required=False, default=0.1)
parser.add_argument("--num_clients", type=int, required=False, default=5)
parser.add_argument("--num_rounds", type=int, required=False, default=1)
parser.add_argument("--local_epochs", type=int, required=False, default=30)
parser.add_argument("--output_dir", type=str, required=False, default=".")
parser.add_argument("--synthetic_split", type=str, required=False, default="noniid", choices=["iid", "noniid"])
parser.add_argument("--synthetic_num_train", type=int, required=False, default=10000)
parser.add_argument("--synthetic_num_test", type=int, required=False, default=10000)
parser.add_argument("--synthetic_dim", type=int, required=False, default=100)
parser.add_argument("--synthetic_signal_dim", type=int, required=False, default=10)
parser.add_argument("--synthetic_signal_strength", type=float, required=False, default=0.7)
parser.add_argument("--synthetic_noise_std", type=float, required=False, default=1.0)
parser.add_argument("--local_train_csv", type=str, required=False, default=None)
parser.add_argument("--local_test_csv", type=str, required=False, default=None)
parser.add_argument("--local_label_col", type=str, required=False, default="-1")
parser.add_argument("--local_has_header", action="store_true")
parser.add_argument("--local_partition", type=str, required=False, default="noniid", choices=["iid", "noniid"])
parser.add_argument("--local_client_col", type=str, required=False, default=None)

args_parser = parser.parse_args()

seed = args_parser.seed
dataset = args_parser.dataset
model_name = args_parser.model
algs_to_run = args_parser.algs_to_run
local_epochs = args_parser.local_epochs
alpha = args_parser.alpha
num_clients = args_parser.num_clients
num_rounds = args_parser.num_rounds
output_dir = args_parser.output_dir
print_every_test = 1
print_every_train = 1
n_c = 2

alpha_tag = str(alpha).replace(".", "p")
if dataset == "SyntheticBinary":
    signal_tag = str(args_parser.synthetic_signal_strength).replace(".", "p")
    noise_tag = str(args_parser.synthetic_noise_std).replace(".", "p")
    filename_extra = (
        "_split"
        + args_parser.synthetic_split
        + "_train"
        + str(args_parser.synthetic_num_train)
        + "_test"
        + str(args_parser.synthetic_num_test)
        + "_dim"
        + str(args_parser.synthetic_dim)
        + "_sdim"
        + str(args_parser.synthetic_signal_dim)
        + "_sig"
        + signal_tag
        + "_noise"
        + noise_tag
    )
else:
    filename_extra = "_partition" + args_parser.local_partition
filename_base = (
    "one_shot_results_seed"
    + str(seed)
    + "_"
    + dataset
    + "_"
    + model_name
    + "_epochs"
    + str(local_epochs)
    + "_alpha"
    + alpha_tag
    + "_clients"
    + str(num_clients)
    + "_rounds"
    + str(num_rounds)
    + filename_extra
)
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, filename_base)
filename_csv = filename + ".csv"
print("Writing results to", filename_csv)

dict_results = {}

for alg in algs_to_run:
    print("Running algorithm", alg)

    np.random.seed(3)
    dataset_train, dataset_train_global, dataset_test_global, net_cls_counts = get_dataset(
        dataset,
        num_clients,
        n_c,
        alpha,
        False,
        synthetic_split=args_parser.synthetic_split,
        synthetic_num_train=args_parser.synthetic_num_train,
        synthetic_num_test=args_parser.synthetic_num_test,
        synthetic_dim=args_parser.synthetic_dim,
        synthetic_signal_dim=args_parser.synthetic_signal_dim,
        synthetic_signal_strength=args_parser.synthetic_signal_strength,
        synthetic_noise_std=args_parser.synthetic_noise_std,
        local_train_csv=args_parser.local_train_csv,
        local_test_csv=args_parser.local_test_csv,
        local_label_col=args_parser.local_label_col,
        local_has_header=args_parser.local_has_header,
        local_partition=args_parser.local_partition,
        local_client_col=args_parser.local_client_col,
        seed=seed,
    )

    val_size = min(500, len(dataset_train_global))
    ind = np.random.choice(len(dataset_train_global), val_size, replace=False)
    dataset_val = torch.utils.data.Subset(dataset_train_global, ind)
    input_dim = dataset_train_global.tensors[0].shape[1]

    args = {
        "bs": 64,
        "local_epochs": local_epochs,
        "device": "cuda",
        "rounds": num_rounds,
        "num_clients": num_clients,
        "augmentation": False,
        "eta": 0.01,
        "dataset": dataset,
        "model": model_name,
        "n_c": n_c,
        "synthetic_dim": input_dim,
    }

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    net_glob_org = get_model(
        args["model"],
        n_c,
        bias=False,
        synthetic_dim=args["synthetic_dim"],
    ).to(args["device"])

    n = len(dataset_train)
    print("No. of clients", n)

    p = np.zeros((n))
    for i in range(n):
        p[i] = len(dataset_train[i])
    p = p / np.sum(p)

    local_model_accs = []
    local_model_loss = []
    d = parameters_to_vector(net_glob_org.parameters()).numel()
    net_glob = copy.deepcopy(net_glob_org)

    for t in range(0, args["rounds"]):
        ind = [i for i in range(n)]
        F_kfac_list = []
        F_diag_list = []
        model_vectors = []
        models = []

        for i in ind:
            print("Training Local Model ", i)
            net_glob.train()
            local = LocalUpdate(args=args, dataset=dataset_train[i])
            model_vector, model, F_kfac, F_diag = local.train_and_compute_fisher(copy.deepcopy(net_glob), args["n_c"])
            model_vectors.append(model_vector)
            models.append(model)
            test_acc, test_loss = test_img(model, dataset_test_global, args)
            print("Local Model ", i, "Test Acc. ", test_acc, "Test Loss ", test_loss)
            local_model_accs.append(test_acc.flatten()[0])
            local_model_loss.append(test_loss)

            F_diag = F_diag * p[i]
            for layer_id in F_kfac.data.keys():
                F_kfac.data[layer_id] = list(F_kfac.data[layer_id])
                F_kfac.data[layer_id][0].mul_(p[i])
                F_kfac.data[layer_id][1].mul_(p[i])
            F_kfac_list.append(F_kfac)
            F_diag_list.append(F_diag)

    dict_results["local_model_test_accuracies_" + str(alpha) + "_" + str(t)] = local_model_accs
    dict_results["local_model_test_losses_" + str(alpha) + "_" + str(t)] = local_model_loss

    net_glob = get_one_shot_model(
        alg,
        d,
        n,
        p,
        args,
        net_glob,
        models,
        model_vectors,
        F_kfac_list,
        F_diag_list,
        dataset_val,
        dataset_train,
        dataset_train_global,
        dataset_test_global,
        filename,
        net_cls_counts,
    )

    test_acc, test_loss = test_img(net_glob, dataset_test_global, args)
    print("Test Acc. ", test_acc, "Test Loss", test_loss)
    dict_results[alg + "_test_loss_" + str(seed) + "_" + str(t)] = test_loss
    dict_results[alg + "_test_acc_" + str(seed) + "_" + str(t)] = test_acc

    with open(filename_csv, "w") as csv_file:
        writer = csv.writer(csv_file)
        for i in dict_results.keys():
            writer.writerow([i, dict_results[i]])


with open(filename_csv, "w") as csv_file:
    writer = csv.writer(csv_file)
    for i in dict_results.keys():
        writer.writerow([i, dict_results[i]])
