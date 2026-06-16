import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag
from torch.utils.data import DataLoader

from models import get_model


FEATURE_DETAIL_FIELDS = [
    "seed",
    "alg",
    "model",
    "split",
    "alpha",
    "synthetic_dim",
    "synthetic_signal_dim",
    "feature_idx",
    "is_signal",
    "method",
    "importance",
    "rank",
]

FEATURE_SUMMARY_FIELDS = [
    "seed",
    "alg",
    "model",
    "split",
    "alpha",
    "synthetic_dim",
    "synthetic_signal_dim",
    "method",
    "topk_hits",
    "topk_precision",
    "mean_signal_rank",
    "median_signal_rank",
    "mean_noise_rank",
    "auroc",
]


def collect_tensor_dataset(dataset, device=None, batch_size=8192):
    """Return full X/y tensors from a TensorDataset-like object or Subset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs = []
    ys = []
    for x_batch, y_batch in loader:
        xs.append(x_batch)
        ys.append(y_batch)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0).long()
    if device is not None:
        x = x.to(device)
        y = y.to(device)
    return x, y


def evaluate_logits_loss_acc_margin(model, x, y, device=None, batch_size=1024):
    """Evaluate CE loss, accuracy percentage, and binary correct-class margin."""
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_margin = 0.0
    total_count = 0

    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            end = min(start + batch_size, len(y))
            x_batch = x[start:end].to(device)
            y_batch = y[start:end].to(device)
            logits = model(x_batch)
            total_loss += F.cross_entropy(logits, y_batch, reduction="sum").item()
            total_correct += logits.argmax(dim=1).eq(y_batch).long().sum().item()
            signs = (2 * y_batch.float()) - 1
            margins = signs * (logits[:, 1] - logits[:, 0])
            total_margin += margins.sum().item()
            total_count += y_batch.numel()

    if was_training:
        model.train()

    return {
        "loss": total_loss / total_count,
        "accuracy": 100.0 * total_correct / total_count,
        "margin": total_margin / total_count,
    }


def input_ablation_importance(model, dataset, args, mode="permute", repeats=5, seed=0):
    """Rank input dimensions by test-time supervised ablation importance.

    This is a supervised feature-selection / signal-recovery diagnostic: it
    perturbs one input coordinate at a time and measures how much prediction of
    the binary label degrades. It is not a causal treatment-effect experiment.
    """
    if mode not in {"zero", "permute"}:
        raise ValueError("mode must be 'zero' or 'permute'")

    device = args.get("device", next(model.parameters()).device)
    batch_size = args.get("feature_importance_batch_size", args.get("bs", 1024))
    x, y = collect_tensor_dataset(dataset, device=None)
    baseline = evaluate_logits_loss_acc_margin(model, x, y, device=device, batch_size=batch_size)

    repeat_count = 1 if mode == "zero" else max(1, int(repeats))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    dim = x.shape[1]
    loss_increase = np.zeros(dim, dtype=np.float64)
    accuracy_drop = np.zeros(dim, dtype=np.float64)
    margin_drop = np.zeros(dim, dtype=np.float64)

    for feature_idx in range(dim):
        feature_loss = []
        feature_acc_drop = []
        feature_margin_drop = []
        for _ in range(repeat_count):
            x_modified = x.clone()
            if mode == "zero":
                x_modified[:, feature_idx] = 0.0
            else:
                permutation = torch.randperm(x.shape[0], generator=generator)
                x_modified[:, feature_idx] = x[permutation, feature_idx]

            metrics = evaluate_logits_loss_acc_margin(
                model,
                x_modified,
                y,
                device=device,
                batch_size=batch_size,
            )
            feature_loss.append(metrics["loss"] - baseline["loss"])
            feature_acc_drop.append(baseline["accuracy"] - metrics["accuracy"])
            feature_margin_drop.append(baseline["margin"] - metrics["margin"])

        loss_increase[feature_idx] = float(np.mean(feature_loss))
        accuracy_drop[feature_idx] = float(np.mean(feature_acc_drop))
        margin_drop[feature_idx] = float(np.mean(feature_margin_drop))

    return {
        "loss": loss_increase,
        "accuracy_drop": accuracy_drop,
        "margin_drop": margin_drop,
        "baseline": baseline,
    }


def _first_linear_layer(model):
    if not hasattr(model, "net") or len(model.net) == 0:
        raise ValueError("feature importance expects a model with a Sequential 'net'")
    first_layer = model.net[0]
    if not isinstance(first_layer, torch.nn.Linear):
        raise ValueError("model.net[0] must be a Linear layer")
    return first_layer


def first_layer_weight_importance(model):
    first_layer = _first_linear_layer(model)
    scores = first_layer.weight.detach().pow(2).sum(dim=0)
    return scores.cpu().numpy()


def fisher_weighted_first_layer_importance(model, f_diag):
    first_layer = _first_linear_layer(model)
    weight = first_layer.weight.detach()
    weight_numel = weight.numel()
    if f_diag.numel() < weight_numel:
        raise ValueError("f_diag is shorter than the first-layer weight block")
    fisher_weight = f_diag.detach()[:weight_numel].reshape_as(weight).to(weight.device)
    scores = 0.5 * fisher_weight * weight.pow(2)
    return scores.sum(dim=0).cpu().numpy()


def compute_diagonal_fisher(model, dataset, args, batch_size=None):
    """Compute diagonal Fisher for the final global model on a dataset."""
    device = args.get("device", next(model.parameters()).device)
    n_output = args["n_c"]
    if batch_size is None:
        batch_size = args.get("feature_importance_batch_size", args.get("bs", 1024))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    fisher = FIM(
        model=model,
        loader=loader,
        representation=PMatDiag,
        device=device,
        n_output=n_output,
    )
    if was_training:
        model.train()
    return fisher.get_diag().detach()


def rank_features(importance):
    scores = np.asarray(importance, dtype=np.float64)
    order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def evaluate_signal_recovery(importance, signal_dim):
    """Evaluate whether top-ranked dimensions recover the known signal set."""
    scores = np.asarray(importance, dtype=np.float64)
    if signal_dim < 1 or signal_dim > len(scores):
        raise ValueError("signal_dim must be in [1, len(importance)]")

    signal_mask = np.zeros(len(scores), dtype=bool)
    signal_mask[:signal_dim] = True
    ranks = rank_features(scores)
    topk = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind="mergesort")[:signal_dim]
    topk_hits = int(signal_mask[topk].sum())

    signal_ranks = ranks[signal_mask]
    noise_ranks = ranks[~signal_mask]
    signal_scores = scores[signal_mask]
    noise_scores = scores[~signal_mask]
    if len(noise_scores) == 0:
        auroc = float("nan")
    else:
        comparisons = signal_scores[:, None] - noise_scores[None, :]
        wins = (comparisons > 0).sum()
        ties = (comparisons == 0).sum()
        auroc = float((wins + 0.5 * ties) / comparisons.size)

    return {
        "topk_hits": topk_hits,
        "topk_precision": topk_hits / signal_dim,
        "mean_signal_rank": float(np.mean(signal_ranks)),
        "median_signal_rank": float(np.median(signal_ranks)),
        "mean_noise_rank": float(np.mean(noise_ranks)) if len(noise_ranks) else float("nan"),
        "auroc": auroc,
    }


def write_feature_importance_outputs(detail_rows, summary_rows, detail_path, summary_path):
    if detail_rows:
        with open(detail_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FEATURE_DETAIL_FIELDS)
            writer.writeheader()
            writer.writerows(detail_rows)
    if summary_rows:
        with open(summary_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FEATURE_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)


def _add_importance_method_rows(detail_rows, summary_rows, metadata, method, importance):
    ranks = rank_features(importance)
    signal_dim = metadata["synthetic_signal_dim"]
    metrics = evaluate_signal_recovery(importance, signal_dim)
    for feature_idx, score in enumerate(importance):
        row = dict(metadata)
        row.update(
            {
                "feature_idx": feature_idx,
                "is_signal": int(feature_idx < signal_dim),
                "method": method,
                "importance": float(score),
                "rank": int(ranks[feature_idx]),
            }
        )
        detail_rows.append(row)

    row = dict(metadata)
    row.update({"method": method})
    row.update(metrics)
    summary_rows.append(row)


def add_model_feature_importance(
    detail_rows,
    summary_rows,
    alg,
    model,
    f_diag_sum_for_importance,
    dataset_train_global,
    dataset_test_global,
    args,
    metadata,
    modes,
    repeats,
    seed,
    batch_size,
):
    """Run supervised signal-dimension recovery for one trained global model.

    The SyntheticBinary generator makes the first synthetic_signal_dim input
    coordinates informative for the binary label and the remaining coordinates
    Gaussian noise. This analysis ranks input dimensions and checks recovery of
    those known signal coordinates. It is not a causal treatment-effect or HTE
    experiment.
    """
    print("Computing feature importance for", alg)
    _add_importance_method_rows(
        detail_rows,
        summary_rows,
        metadata,
        "weight_norm",
        first_layer_weight_importance(model),
    )

    if f_diag_sum_for_importance is not None:
        _add_importance_method_rows(
            detail_rows,
            summary_rows,
            metadata,
            "fisher_weighted",
            fisher_weighted_first_layer_importance(model, f_diag_sum_for_importance),
        )

    final_f_diag = compute_diagonal_fisher(
        model,
        dataset_train_global,
        args,
        batch_size=batch_size,
    )
    _add_importance_method_rows(
        detail_rows,
        summary_rows,
        metadata,
        "global_fisher_weighted",
        fisher_weighted_first_layer_importance(model, final_f_diag),
    )

    for mode in modes:
        mode_seed = seed + sum(ord(ch) for ch in (alg + mode))
        ablation_scores = input_ablation_importance(
            model,
            dataset_test_global,
            args,
            mode=mode,
            repeats=repeats,
            seed=mode_seed,
        )
        _add_importance_method_rows(
            detail_rows,
            summary_rows,
            metadata,
            "ablation_" + mode + "_loss",
            ablation_scores["loss"],
        )
        _add_importance_method_rows(
            detail_rows,
            summary_rows,
            metadata,
            "ablation_" + mode + "_margin",
            ablation_scores["margin_drop"],
        )
        _add_importance_method_rows(
            detail_rows,
            summary_rows,
            metadata,
            "ablation_" + mode + "_acc_drop",
            ablation_scores["accuracy_drop"],
        )


def train_pooled_model(args, dataset_train_global):
    model = get_model(
        args["model"],
        args["n_c"],
        bias=False,
        synthetic_dim=args["synthetic_dim"],
    ).to(args["device"])
    loader = DataLoader(dataset_train_global, batch_size=args["bs"], shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args["eta"], momentum=0.9)

    model.train()
    for epoch in range(args["local_epochs"]):
        batch_loss = []
        for data, target in loader:
            data, target = data.to(args["device"]), target.to(args["device"])
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())
        print("Pooled Epoch No. ", epoch, "Loss ", sum(batch_loss) / len(batch_loss))

    return model
