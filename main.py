import argparse
import copy
import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from data import (
    build_effect_modifier_site_evaluation_datasets,
    get_dataset,
)
from models import get_model
from train_model import LocalUpdate
from run_one_shot_algs import get_one_shot_model
from utils.compute_accuracy import test_img
from utils.feature_importance import (
    add_model_feature_importance,
    train_pooled_model,
    write_feature_importance_outputs,
)
from utils.prediction_intervention import (
    add_model_prediction_intervention,
    train_pooled_prediction_model,
    write_prediction_intervention_outputs,
)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["SyntheticBinary", "SyntheticEffectModifier", "LocalBinaryCSV"],
)
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
parser.add_argument("--num_clients", type=int, required=False, default=None)
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
parser.add_argument("--effect_modifier_covariate_dim", type=int, required=False, default=100)
parser.add_argument("--effect_modifier_signal_dim", type=int, required=False, default=2)
parser.add_argument("--effect_modifier_signal_strength", type=float, required=False, default=2.0)
parser.add_argument("--effect_modifier_main_effect_strength", type=float, required=False, default=1.0)
parser.add_argument("--effect_modifier_confounding_strength", type=float, required=False, default=1.0)
parser.add_argument("--effect_modifier_intercept", type=float, required=False, default=0.0)
parser.add_argument("--effect_modifier_treatment_prob", type=float, required=False, default=0.5)
parser.add_argument("--effect_modifier_propensity_clip", type=float, required=False, default=0.05)
parser.add_argument("--effect_modifier_site_min_samples", type=int, required=False, default=500)
parser.add_argument("--effect_modifier_site_max_samples", type=int, required=False, default=1000)
parser.add_argument("--effect_modifier_site_gaussian_shift", action="store_true")
parser.add_argument(
    "--effect_modifier_combined_ablation",
    action="store_true",
    help=(
        "Explicitly authorize the factorial combination of site-Gaussian "
        "training covariates and site-local prediction intervention."
    ),
)
parser.add_argument("--effect_modifier_site_gaussian_mean_span", type=float, default=0.75)
parser.add_argument("--effect_modifier_site_gaussian_std_low", type=float, default=0.75)
parser.add_argument("--effect_modifier_site_gaussian_std_high", type=float, default=1.25)
parser.add_argument("--local_train_csv", type=str, required=False, default=None)
parser.add_argument("--local_test_csv", type=str, required=False, default=None)
parser.add_argument("--local_label_col", type=str, required=False, default="-1")
parser.add_argument("--local_has_header", action="store_true")
parser.add_argument("--local_partition", type=str, required=False, default="noniid", choices=["iid", "noniid"])
parser.add_argument("--local_client_col", type=str, required=False, default=None)
parser.add_argument("--feature_importance", action="store_true")
parser.add_argument("--feature_importance_repeats", type=int, required=False, default=5)
parser.add_argument(
    "--feature_importance_modes",
    nargs="+",
    type=str,
    required=False,
    default=["permute", "zero"],
    choices=["permute", "zero"],
)
parser.add_argument("--feature_importance_output_suffix", type=str, required=False, default="feature_importance")
parser.add_argument("--feature_importance_batch_size", type=int, required=False, default=1024)
parser.add_argument("--feature_importance_no_pooled_baseline", action="store_true")
parser.add_argument("--prediction_intervention", action="store_true")
parser.add_argument(
    "--prediction_intervention_modes",
    nargs="+",
    type=str,
    required=False,
    default=["permute", "zero"],
    choices=["permute", "zero"],
)
parser.add_argument("--prediction_intervention_repeats", type=int, required=False, default=5)
parser.add_argument("--prediction_intervention_batch_size", type=int, required=False, default=1024)
parser.add_argument(
    "--prediction_intervention_output_suffix",
    type=str,
    required=False,
    default="prediction_intervention",
)
parser.add_argument(
    "--prediction_intervention_scope",
    type=str,
    default="global",
    choices=["global", "site"],
    help="Permute globally or independently within site-local held-out shards.",
)
parser.add_argument("--prediction_intervention_include_local_models", action="store_true")
parser.add_argument("--prediction_intervention_no_pooled_baseline", action="store_true")

args_parser = parser.parse_args()

if args_parser.effect_modifier_site_gaussian_shift and args_parser.dataset != "SyntheticEffectModifier":
    parser.error("--effect_modifier_site_gaussian_shift requires SyntheticEffectModifier")
if args_parser.prediction_intervention_scope == "site" and args_parser.dataset != "SyntheticEffectModifier":
    parser.error("--prediction_intervention_scope site requires SyntheticEffectModifier")
if args_parser.prediction_intervention_scope == "site" and not args_parser.prediction_intervention:
    parser.error("--prediction_intervention_scope site requires --prediction_intervention")
combined_effect_modifier_ablation = (
    args_parser.effect_modifier_site_gaussian_shift
    and args_parser.prediction_intervention_scope == "site"
)
if combined_effect_modifier_ablation and not args_parser.effect_modifier_combined_ablation:
    parser.error(
        "combining --effect_modifier_site_gaussian_shift with "
        "--prediction_intervention_scope site requires "
        "--effect_modifier_combined_ablation"
    )
if args_parser.effect_modifier_combined_ablation and not combined_effect_modifier_ablation:
    parser.error(
        "--effect_modifier_combined_ablation requires both "
        "--effect_modifier_site_gaussian_shift and "
        "--prediction_intervention_scope site"
    )

allowed_output_roots = [
    Path("/home/del6500/projects/FedFisher").resolve(),
    Path("/data/del6500/FedFisher").resolve(),
    Path("/scr/del6500/FedFisher").resolve(),
]
resolved_output_dir = Path(args_parser.output_dir).resolve(strict=False)
if not any(
    resolved_output_dir == root or root in resolved_output_dir.parents
    for root in allowed_output_roots
):
    parser.error(
        "--output_dir resolves outside the allowed FedFisher roots: %s"
        % resolved_output_dir
    )
args_parser.output_dir = str(resolved_output_dir)

seed = args_parser.seed
dataset = args_parser.dataset
model_name = args_parser.model
algs_to_run = args_parser.algs_to_run
local_epochs = args_parser.local_epochs
alpha = args_parser.alpha
num_clients = args_parser.num_clients
if num_clients is None:
    num_clients = 10 if dataset == "SyntheticEffectModifier" else 5
num_rounds = args_parser.num_rounds
output_dir = args_parser.output_dir
print_every_test = 1
print_every_train = 1
n_c = 2
SYNTHETIC_RECOVERY_DATASETS = {"SyntheticBinary", "SyntheticEffectModifier"}


def float_tag(value):
    return str(value).replace(".", "p").replace("-", "m")


def get_recovery_indices(dataset_name, parsed_args, input_dim):
    if dataset_name == "SyntheticEffectModifier":
        covariate_dim = parsed_args.effect_modifier_covariate_dim
        signal_dim = parsed_args.effect_modifier_signal_dim
        expected_dim = 1 + 2 * covariate_dim
        if input_dim != expected_dim:
            raise ValueError("SyntheticEffectModifier input dimension should be %d, got %d" % (expected_dim, input_dim))
        # Legacy feature importance and direct_interaction diagnostics use the
        # interaction block; tau-intervention computes raw-X indices separately.
        interaction_start = 1 + covariate_dim
        candidate_indices = list(range(interaction_start, interaction_start + covariate_dim))
        signal_indices = list(range(interaction_start, interaction_start + signal_dim))
        return signal_dim, signal_indices, candidate_indices

    return parsed_args.synthetic_signal_dim, None, None


def write_site_gaussian_manifest(path, datasets, parsed_args):
    fields = [
        "seed",
        "model",
        "split",
        "alpha",
        "site_idx",
        "num_examples",
        "label_0_count",
        "label_1_count",
        "gaussian_mean",
        "gaussian_std",
        "empirical_raw_mean",
        "empirical_raw_std",
        "empirical_x1_mean",
        "empirical_x1_std",
        "empirical_x2_mean",
        "empirical_x2_std",
    ]
    rows = []
    covariate_dim = parsed_args.effect_modifier_covariate_dim
    for site_idx, site_dataset in enumerate(datasets):
        features, labels = site_dataset.tensors
        raw_x = features[:, 1 : 1 + covariate_dim]
        label_counts = torch.bincount(labels, minlength=2)
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "split": parsed_args.synthetic_split,
                "alpha": alpha,
                "site_idx": site_idx,
                "num_examples": len(site_dataset),
                "label_0_count": int(label_counts[0]),
                "label_1_count": int(label_counts[1]),
                "gaussian_mean": site_dataset.site_gaussian_mean,
                "gaussian_std": site_dataset.site_gaussian_std,
                "empirical_raw_mean": float(raw_x.mean()),
                "empirical_raw_std": float(raw_x.std(unbiased=False)),
                "empirical_x1_mean": float(raw_x[:, 0].mean()),
                "empirical_x1_std": float(raw_x[:, 0].std(unbiased=False)),
                "empirical_x2_mean": float(raw_x[:, 1].mean()),
                "empirical_x2_std": float(raw_x[:, 1].std(unbiased=False)),
            }
        )
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


alpha_tag = float_tag(alpha)
if dataset == "SyntheticBinary":
    signal_tag = float_tag(args_parser.synthetic_signal_strength)
    noise_tag = float_tag(args_parser.synthetic_noise_std)
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
elif dataset == "SyntheticEffectModifier":
    strength_tag = float_tag(args_parser.effect_modifier_signal_strength)
    main_effect_tag = float_tag(args_parser.effect_modifier_main_effect_strength)
    confounding_tag = float_tag(args_parser.effect_modifier_confounding_strength)
    intercept_tag = float_tag(args_parser.effect_modifier_intercept)
    treat_tag = float_tag(args_parser.effect_modifier_treatment_prob)
    propensity_clip_tag = float_tag(args_parser.effect_modifier_propensity_clip)
    filename_extra = (
        "_split"
        + args_parser.synthetic_split
        + "_sites"
        + str(num_clients)
        + "_smin"
        + str(args_parser.effect_modifier_site_min_samples)
        + "_smax"
        + str(args_parser.effect_modifier_site_max_samples)
        + "_test"
        + str(args_parser.synthetic_num_test)
        + "_xdim"
        + str(args_parser.effect_modifier_covariate_dim)
        + "_emdim"
        + str(args_parser.effect_modifier_signal_dim)
        + "_emstr"
        + strength_tag
        + "_mainstr"
        + main_effect_tag
        + "_confstr"
        + confounding_tag
        + "_pclip"
        + propensity_clip_tag
        + "_intercept"
        + intercept_tag
        + "_treat"
        + treat_tag
    )
else:
    filename_extra = "_partition" + args_parser.local_partition

if dataset == "SyntheticEffectModifier":
    if args_parser.effect_modifier_site_gaussian_shift:
        filename_extra += "_sg"
    if args_parser.prediction_intervention_scope == "site":
        filename_extra += "_lp"

# Keep effect-modifier output basenames below Linux's 255-byte component limit
# even after the longest prediction-intervention suffix is appended. All DGP
# parameters remain encoded in filename_extra and in the output rows.
dataset_filename_tag = "SEM" if dataset == "SyntheticEffectModifier" else dataset
filename_base = (
    "one_shot_results_seed"
    + str(seed)
    + "_"
    + dataset_filename_tag
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
site_gaussian_manifest_csv = filename + "_site_gaussian_manifest.csv"
print("Writing results to", filename_csv)

dict_results = {}
feature_importance_enabled = args_parser.feature_importance and dataset in SYNTHETIC_RECOVERY_DATASETS
if args_parser.feature_importance and dataset not in SYNTHETIC_RECOVERY_DATASETS:
    print("Feature importance is currently only available for synthetic datasets; skipping it.")

feature_detail_rows = []
feature_summary_rows = []
feature_detail_csv = filename + "_" + args_parser.feature_importance_output_suffix + ".csv"
feature_summary_csv = filename + "_" + args_parser.feature_importance_output_suffix + "_summary.csv"
latest_feature_context = None

prediction_intervention_enabled = args_parser.prediction_intervention and dataset in SYNTHETIC_RECOVERY_DATASETS
if args_parser.prediction_intervention and dataset not in SYNTHETIC_RECOVERY_DATASETS:
    print("Prediction intervention is currently only available for synthetic datasets; skipping it.")

prediction_detail_rows = []
prediction_summary_rows = []
prediction_model_summary_rows = []
prediction_site_summary_rows = []
prediction_detail_csv = filename + "_" + args_parser.prediction_intervention_output_suffix + ".csv"
prediction_summary_csv = filename + "_" + args_parser.prediction_intervention_output_suffix + "_summary.csv"
prediction_model_summary_csv = filename + "_" + args_parser.prediction_intervention_output_suffix + "_model_summary.csv"
prediction_site_summary_csv = filename + "_" + args_parser.prediction_intervention_output_suffix + "_site_summary.csv"
latest_prediction_context = None
local_prediction_intervention_done = False

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
        effect_modifier_covariate_dim=args_parser.effect_modifier_covariate_dim,
        effect_modifier_signal_dim=args_parser.effect_modifier_signal_dim,
        effect_modifier_signal_strength=args_parser.effect_modifier_signal_strength,
        effect_modifier_main_effect_strength=args_parser.effect_modifier_main_effect_strength,
        effect_modifier_confounding_strength=args_parser.effect_modifier_confounding_strength,
        effect_modifier_intercept=args_parser.effect_modifier_intercept,
        effect_modifier_treatment_prob=args_parser.effect_modifier_treatment_prob,
        effect_modifier_propensity_clip=args_parser.effect_modifier_propensity_clip,
        effect_modifier_site_min_samples=args_parser.effect_modifier_site_min_samples,
        effect_modifier_site_max_samples=args_parser.effect_modifier_site_max_samples,
        effect_modifier_site_gaussian_shift=args_parser.effect_modifier_site_gaussian_shift,
        effect_modifier_site_gaussian_mean_span=args_parser.effect_modifier_site_gaussian_mean_span,
        effect_modifier_site_gaussian_std_low=args_parser.effect_modifier_site_gaussian_std_low,
        effect_modifier_site_gaussian_std_high=args_parser.effect_modifier_site_gaussian_std_high,
        local_train_csv=args_parser.local_train_csv,
        local_test_csv=args_parser.local_test_csv,
        local_label_col=args_parser.local_label_col,
        local_has_header=args_parser.local_has_header,
        local_partition=args_parser.local_partition,
        local_client_col=args_parser.local_client_col,
        seed=seed,
    )

    site_evaluation_datasets = None
    if dataset == "SyntheticEffectModifier":
        site_sizes = [len(site_dataset) for site_dataset in dataset_train]
        dict_results["effect_modifier_site_train_sizes_" + str(alpha)] = site_sizes
        dict_results["effect_modifier_total_train_examples_" + str(alpha)] = sum(site_sizes)
        if args_parser.effect_modifier_site_gaussian_shift and alg == algs_to_run[0]:
            write_site_gaussian_manifest(
                site_gaussian_manifest_csv,
                dataset_train,
                args_parser,
            )
        if args_parser.prediction_intervention_scope == "site":
            site_evaluation_datasets = build_effect_modifier_site_evaluation_datasets(
                dataset_test_global,
                dataset_train,
                args_parser.synthetic_split,
                alpha,
                seed,
            )

    val_size = min(500, len(dataset_train_global))
    ind = np.random.choice(len(dataset_train_global), val_size, replace=False)
    dataset_val = torch.utils.data.Subset(dataset_train_global, ind)
    input_dim = dataset_train_global.tensors[0].shape[1]
    recovery_signal_dim, recovery_signal_indices, recovery_candidate_indices = get_recovery_indices(
        dataset,
        args_parser,
        input_dim,
    )

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
        "feature_importance_batch_size": args_parser.feature_importance_batch_size,
        "prediction_intervention_batch_size": args_parser.prediction_intervention_batch_size,
        "effect_modifier_covariate_dim": args_parser.effect_modifier_covariate_dim,
        "effect_modifier_signal_dim": args_parser.effect_modifier_signal_dim,
        "effect_modifier_main_effect_strength": args_parser.effect_modifier_main_effect_strength,
        "effect_modifier_confounding_strength": args_parser.effect_modifier_confounding_strength,
        "effect_modifier_propensity_clip": args_parser.effect_modifier_propensity_clip,
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

    latest_feature_context = {
        "args": args,
        "dataset_train_global": dataset_train_global,
        "dataset_test_global": dataset_test_global,
        "recovery_signal_dim": recovery_signal_dim,
        "recovery_signal_indices": recovery_signal_indices,
        "recovery_candidate_indices": recovery_candidate_indices,
    }
    latest_prediction_context = {
        "args": args,
        "dataset_train_global": dataset_train_global,
        "dataset_test_global": dataset_test_global,
        "site_evaluation_datasets": site_evaluation_datasets,
        "recovery_signal_dim": recovery_signal_dim,
        "recovery_signal_indices": recovery_signal_indices,
        "recovery_candidate_indices": recovery_candidate_indices,
    }
    if feature_importance_enabled:
        F_diag_sum_for_importance = torch.zeros_like(F_diag_list[0])
        for F_diag in F_diag_list:
            F_diag_sum_for_importance += F_diag
        feature_metadata = {
            "seed": seed,
            "alg": alg,
            "model": model_name,
            "split": args_parser.synthetic_split,
            "alpha": alpha,
            "synthetic_dim": args["synthetic_dim"],
            "synthetic_signal_dim": recovery_signal_dim,
        }
        add_model_feature_importance(
            feature_detail_rows,
            feature_summary_rows,
            alg,
            net_glob,
            F_diag_sum_for_importance,
            dataset_train_global,
            dataset_test_global,
            args,
            feature_metadata,
            args_parser.feature_importance_modes,
            args_parser.feature_importance_repeats,
            seed,
            args_parser.feature_importance_batch_size,
            signal_indices=recovery_signal_indices,
            candidate_indices=recovery_candidate_indices,
        )
        write_feature_importance_outputs(
            feature_detail_rows,
            feature_summary_rows,
            feature_detail_csv,
            feature_summary_csv,
        )

    if prediction_intervention_enabled:
        prediction_metadata = {
            "seed": seed,
            "model": model_name,
            "split": args_parser.synthetic_split,
            "alpha": alpha,
            "synthetic_dim": args["synthetic_dim"],
            "synthetic_signal_dim": recovery_signal_dim,
        }
        prediction_seed = seed + sum(ord(ch) for ch in alg)
        add_model_prediction_intervention(
            prediction_detail_rows,
            prediction_summary_rows,
            prediction_model_summary_rows,
            alg,
            net_glob,
            dataset_test_global,
            args,
            prediction_metadata,
            args_parser.prediction_intervention_modes,
            args_parser.prediction_intervention_repeats,
            prediction_seed,
            signal_indices=recovery_signal_indices,
            candidate_indices=recovery_candidate_indices,
            permutation_scope=args_parser.prediction_intervention_scope,
            site_datasets=site_evaluation_datasets,
            site_summary_rows=prediction_site_summary_rows,
        )
        if (
            args_parser.prediction_intervention_include_local_models
            and not local_prediction_intervention_done
        ):
            # Local models are trained before aggregation; avoid duplicate rows across algs.
            for local_idx, local_model in enumerate(models):
                local_seed = seed + 10000 + local_idx
                add_model_prediction_intervention(
                    prediction_detail_rows,
                    prediction_summary_rows,
                    prediction_model_summary_rows,
                    "local_client_" + str(local_idx),
                    local_model,
                    dataset_test_global,
                    args,
                    prediction_metadata,
                    args_parser.prediction_intervention_modes,
                    args_parser.prediction_intervention_repeats,
                    local_seed,
                    signal_indices=recovery_signal_indices,
                    candidate_indices=recovery_candidate_indices,
                    permutation_scope=args_parser.prediction_intervention_scope,
                    site_datasets=site_evaluation_datasets,
                    site_summary_rows=prediction_site_summary_rows,
                )
            local_prediction_intervention_done = True
        write_prediction_intervention_outputs(
            prediction_detail_rows,
            prediction_summary_rows,
            prediction_model_summary_rows,
            prediction_detail_csv,
            prediction_summary_csv,
            prediction_model_summary_csv,
            site_summary_rows=prediction_site_summary_rows,
            site_summary_path=prediction_site_summary_csv,
        )

    with open(filename_csv, "w") as csv_file:
        writer = csv.writer(csv_file)
        for i in dict_results.keys():
            writer.writerow([i, dict_results[i]])


with open(filename_csv, "w") as csv_file:
    writer = csv.writer(csv_file)
    for i in dict_results.keys():
        writer.writerow([i, dict_results[i]])

if (
    feature_importance_enabled
    and not args_parser.feature_importance_no_pooled_baseline
    and latest_feature_context is not None
):
    print("Running pooled baseline for feature importance")
    args = latest_feature_context["args"]
    dataset_train_global = latest_feature_context["dataset_train_global"]
    dataset_test_global = latest_feature_context["dataset_test_global"]
    recovery_signal_dim = latest_feature_context["recovery_signal_dim"]
    recovery_signal_indices = latest_feature_context["recovery_signal_indices"]
    recovery_candidate_indices = latest_feature_context["recovery_candidate_indices"]
    torch.manual_seed(seed)
    random.seed(seed)
    pooled_model = train_pooled_model(args, dataset_train_global)
    test_acc, test_loss = test_img(pooled_model, dataset_test_global, args)
    print("Pooled Test Acc. ", test_acc, "Test Loss", test_loss)
    feature_metadata = {
        "seed": seed,
        "alg": "pooled",
        "model": model_name,
        "split": args_parser.synthetic_split,
        "alpha": alpha,
        "synthetic_dim": args["synthetic_dim"],
        "synthetic_signal_dim": recovery_signal_dim,
    }
    add_model_feature_importance(
        feature_detail_rows,
        feature_summary_rows,
        "pooled",
        pooled_model,
        None,
        dataset_train_global,
        dataset_test_global,
        args,
        feature_metadata,
        args_parser.feature_importance_modes,
        args_parser.feature_importance_repeats,
        seed,
        args_parser.feature_importance_batch_size,
        signal_indices=recovery_signal_indices,
        candidate_indices=recovery_candidate_indices,
    )
    write_feature_importance_outputs(
        feature_detail_rows,
        feature_summary_rows,
        feature_detail_csv,
        feature_summary_csv,
    )

if (
    prediction_intervention_enabled
    and not args_parser.prediction_intervention_no_pooled_baseline
    and latest_prediction_context is not None
):
    print("Running pooled baseline for prediction intervention")
    args = latest_prediction_context["args"]
    dataset_train_global = latest_prediction_context["dataset_train_global"]
    dataset_test_global = latest_prediction_context["dataset_test_global"]
    site_evaluation_datasets = latest_prediction_context["site_evaluation_datasets"]
    recovery_signal_dim = latest_prediction_context["recovery_signal_dim"]
    recovery_signal_indices = latest_prediction_context["recovery_signal_indices"]
    recovery_candidate_indices = latest_prediction_context["recovery_candidate_indices"]
    torch.manual_seed(seed)
    random.seed(seed)
    pooled_model = train_pooled_prediction_model(args, dataset_train_global)
    test_acc, test_loss = test_img(pooled_model, dataset_test_global, args)
    print("Pooled Prediction Test Acc. ", test_acc, "Test Loss", test_loss)
    prediction_metadata = {
        "seed": seed,
        "model": model_name,
        "split": args_parser.synthetic_split,
        "alpha": alpha,
        "synthetic_dim": args["synthetic_dim"],
        "synthetic_signal_dim": recovery_signal_dim,
    }
    add_model_prediction_intervention(
        prediction_detail_rows,
        prediction_summary_rows,
        prediction_model_summary_rows,
        "pooled",
        pooled_model,
        dataset_test_global,
        args,
        prediction_metadata,
        args_parser.prediction_intervention_modes,
        args_parser.prediction_intervention_repeats,
        seed + 20000,
        signal_indices=recovery_signal_indices,
        candidate_indices=recovery_candidate_indices,
        permutation_scope=args_parser.prediction_intervention_scope,
        site_datasets=site_evaluation_datasets,
        site_summary_rows=prediction_site_summary_rows,
    )
    write_prediction_intervention_outputs(
        prediction_detail_rows,
        prediction_summary_rows,
        prediction_model_summary_rows,
        prediction_detail_csv,
        prediction_summary_csv,
        prediction_model_summary_csv,
        site_summary_rows=prediction_site_summary_rows,
        site_summary_path=prediction_site_summary_csv,
    )
